"""Layout CHATTING (una cara grande): la cámara sigue la cara, suave, a 9:16.

Pasos:
1. Detectar la cara cada ``sample_interval`` s (por defecto 0,33 s) a lo largo del clip.
2. Elegir LA cara a seguir: la más grande, y entre muestras la más cercana a la
   anterior (si alguien cruza por detrás no cambiamos de persona).
3. Suavizar en 3 capas para que no tiemble:
   - mediana de 5 muestras (quita detecciones sueltas erróneas),
   - zona muerta: si la cara se mueve poco, la cámara se queda quieta,
   - media exponencial + velocidad máxima: cuando se mueve, se desliza.
4. ffmpeg recorta con una ventana 9:16 que se mueve según esa trayectoria
   (sendcmd cambia la x del recorte 10 veces por segundo).
"""
import json
import os
import subprocess
import threading

import cv2
import numpy as np

OUT_W, OUT_H = 1080, 1920
_LOCAL = threading.local()


def _detector():
    fd = getattr(_LOCAL, "fd", None)
    if fd is None:
        import mediapipe as mp
        fd = mp.solutions.face_detection.FaceDetection(model_selection=1,
                                                        min_detection_confidence=0.5)
        _LOCAL.fd = fd
    return fd


def detect_big_faces(frame, min_width_frac=0.06):
    """Caras grandes (modelo de largo alcance sobre el frame entero)."""
    H, W = frame.shape[:2]
    res = _detector().process(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
    out = []
    for d in res.detections or []:
        b = d.location_data.relative_bounding_box
        if b.width >= min_width_frac:
            out.append((b.xmin * W, b.ymin * H, b.width * W, b.height * H, float(d.score[0])))
    return out


def sample_track(video_path, interval=0.33):
    """[(t, (cx, cy, w, h) o None)] siguiendo a la cara principal."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    step = max(1, int(round(fps * interval)))
    samples, prev, idx = [], None, 0
    while True:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, fr = cap.retrieve()
            if not ok:
                break
            faces = detect_big_faces(fr)
            pick = None
            if faces:
                if prev is None:
                    f = max(faces, key=lambda f: f[2] * f[3])
                else:
                    # La más cercana a la anterior, salvo que haya otra mucho más grande.
                    f = min(faces, key=lambda f: abs(f[0] + f[2] / 2 - prev[0]) +
                            abs(f[1] + f[3] / 2 - prev[1]))
                pick = (f[0] + f[2] / 2, f[1] + f[3] / 2, f[2], f[3])
                prev = pick
            samples.append((idx / fps, pick))
        idx += 1
    W = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    H = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return samples, (W, H), fps


def smooth_path(samples, crop_w, W, cfg):
    """Trayectoria x (borde izquierdo del recorte) por muestra.

    La cámara está QUIETA mientras la cara cabe cómoda en el encuadre y solo
    se mueve (suave) cuando la cara se acerca a un borde. Además hay un límite
    duro: la cara nunca puede quedar cortada. Medido en un clip de Ibai: seguir
    el centro con suavizado dejaba la cara cortada en 8-21 de 185 muestras."""
    def _series(fn):
        v = np.array([fn(s[1]) if s[1] else np.nan for s in samples], dtype=float)
        good = ~np.isnan(v)
        if not good.any():
            return None
        idx = np.arange(len(v))
        v = np.interp(idx, idx[good], v[good])  # huecos: mantener/interpolar
        k = 2  # mediana de 5: quita detecciones sueltas erróneas
        return np.array([np.median(v[max(0, i - k):i + k + 1]) for i in range(len(v))])

    lefts = _series(lambda f: f[0] - f[2] / 2)
    rights = _series(lambda f: f[0] + f[2] / 2)
    # Sin filtrar (para el límite duro: la mediana va ~0,6 s por detrás).
    raw_l = [s[1][0] - s[1][2] / 2 if s[1] else None for s in samples]
    raw_r = [s[1][0] + s[1][2] / 2 if s[1] else None for s in samples]
    if lefts is None:
        return np.full(len(samples), (W - crop_w) / 2)
    hard = cfg.get("face_margin_frac", 0.08) * crop_w     # la cara nunca más cerca del borde
    comfy = cfg.get("deadzone_frac", 0.18) * crop_w       # zona cómoda: no mover
    alpha = cfg.get("ema_alpha", 0.35)
    cam = float(np.clip((lefts[0] + rights[0]) / 2 - crop_w / 2, 0, W - crop_w))
    out = []
    for l, r, rl, rr in zip(lefts, rights, raw_l, raw_r):
        # Zona cómoda proporcional al hueco libre (una cara grande deja poco).
        free = max(0.0, crop_w - (r - l))
        cm = min(comfy, free * 0.3)
        c_lo, c_hi = r + cm - crop_w, l - cm
        hl = min(l, rl) if rl is not None else l
        hr = max(r, rr) if rr is not None else r
        h_lo, h_hi = hr + hard - crop_w, hl - hard
        if c_lo > c_hi:  # cara muy grande: centrarla
            c_lo = c_hi = (l + r) / 2 - crop_w / 2
        target = min(max(cam, c_lo), c_hi)  # quieta si ya está cómoda
        cam += alpha * (target - cam)
        if h_lo <= h_hi:
            cam = min(max(cam, h_lo), h_hi)  # límite duro: nunca cortar la cara
        cam = min(max(cam, 0), W - crop_w)
        out.append(cam)
    return np.array(out)


def plan_clip(clip_path, cfg):
    samples, (W, H), fps = sample_track(clip_path, cfg.get("sample_interval", 0.33))
    found = sum(1 for s in samples if s[1])
    zoom = cfg.get("zoom", 1.0)          # 1.0 = alto completo; 0.8 = un poco más cerca
    crop_h = int(H * zoom) // 2 * 2
    crop_w = int(crop_h * 9 / 16) // 2 * 2
    if crop_w > W:
        crop_w, crop_h = W // 2 * 2, int(W * 16 / 9) // 2 * 2
    xs = smooth_path(samples, crop_w, W, cfg)
    # Vertical: con zoom < 1, la cara a ~38% de la altura (aire sobre la cabeza).
    if crop_h < H:
        cys = [s[1][1] for s in samples if s[1]]
        cy = float(np.median(cys)) if cys else H / 2
        y = int(min(max(cy - crop_h * 0.38, 0), H - crop_h)) // 2 * 2
    else:
        y = 0
    return {"mode": "face_track", "faces_found": round(found / max(1, len(samples)), 2),
            "crop_w": crop_w, "crop_h": crop_h, "y": y,
            "path": [[round(s[0], 2), int(x)] for s, x in zip(samples, xs)],
            "source": [W, H], "fps": fps}


def _per_frame(path, fps):
    """Interpola la trayectoria (una x cada ~0,33 s) a cada fotograma, para que
    la ventana se deslice en vez de dar saltos 3 veces por segundo."""
    if len(path) < 2:
        return [(t, x) for t, x in path]
    ts = np.array([p[0] for p in path])
    xs = np.array([p[1] for p in path], dtype=float)
    grid = np.arange(ts[0], ts[-1], 1.0 / fps)
    xi = np.interp(grid, ts, xs)
    out, last = [], None
    for t, x in zip(grid, xi):
        x = int(round(x)) // 2 * 2
        if x != last:
            out.append((float(t), x))
            last = x
    return out


def render(clip_path, out_path, cfg):
    plan = plan_clip(clip_path, cfg)
    cmds = out_path + ".sendcmd.txt"
    with open(cmds, "w") as f:
        for t, x in _per_frame(plan["path"], plan.get("fps", 30)):
            f.write(f"{t:.3f} crop@c x {x};\n")
    x0 = plan["path"][0][1] if plan["path"] else 0
    cmdpath = cmds.replace("\\", "/").replace(":", "\\:")
    vf = (f"sendcmd=f='{cmdpath}',crop@c=w={plan['crop_w']}:h={plan['crop_h']}:x={x0}:y={plan['y']},"
          f"scale={OUT_W}:{OUT_H},setsar=1,format=yuv420p")
    try:
        import ffmpeg_utils
        venc = ffmpeg_utils.video_encode_args(ffmpeg_utils.QUALITY)
    except Exception:
        venc = ["-c:v", "libx264", "-preset", "medium", "-crf", "18"]
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", clip_path, "-vf", vf,
                    "-map", "0:v", "-map", "0:a?", *venc, "-c:a", "aac", "-b:a", "192k",
                    "-movflags", "+faststart", out_path], check=True)
    os.remove(cmds)
    if cfg.get("debug_video", True):
        debug_video(clip_path, out_path + ".debug.mp4", plan)
    return plan


def debug_video(clip_path, out_path, plan):
    """Clip original con la ventana 9:16 que se usa (rojo) moviéndose."""
    W = plan["source"][0]
    s = 960 / W
    cmds = out_path + ".sendcmd.txt"
    with open(cmds, "w") as f:
        for t, x in _per_frame(plan["path"], plan.get("fps", 30)):
            f.write(f"{t:.3f} drawbox@b x {int(x * s)};\n")
    cmdpath = cmds.replace("\\", "/").replace(":", "\\:")
    x0 = int(plan["path"][0][1] * s) if plan["path"] else 0
    vf = (f"scale=960:-2,sendcmd=f='{cmdpath}',drawbox@b=x={x0}:y={int(plan['y'] * s)}:"
          f"w={int(plan['crop_w'] * s)}:h={int(plan['crop_h'] * s)}:color=red:t=4")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", clip_path, "-vf", vf, "-an",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", out_path], check=True)
    os.remove(cmds)

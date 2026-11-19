"""Layout PODCAST: 2+ personas en el mismo plano.

Quién habla: movimiento de la boca de cada cara + volumen del audio, en
ventanas de 0,4 s (detector de OpenShorts, active_speaker.py; gratis, sin
diarización). Encima, nuestras reglas:

- SPEAKER SWITCH: se encuadra a quien habla (recorte 9:16 centrado en su
  cara). Un cambio solo se acepta si dura >= ``min_switch_seconds`` (1,5 s),
  para que no parpadee.
- SPLIT: las dos caras apiladas (arriba / abajo) cuando hablan a la vez o el
  intercambio es tan rápido que el speaker switch cortaría cada segundo.
- Si nadie habla claro, se mantiene el plano anterior.
"""
import json
import math
import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np

from . import face_track, webcam

OUT_W, OUT_H = 1080, 1920


_YOLO = {}


def _yolo():
    """Detector de personas (Ultralytics YOLO, ya incluido con OpenShorts)."""
    import threading
    key = threading.get_ident()
    if key not in _YOLO:
        from ultralytics import YOLO
        root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        _YOLO[key] = YOLO(os.path.join(root, "yolov8n.pt"))
    return _YOLO[key]


def detect_heads(frame, cfg):
    """Cabezas de las personas sentadas en plano, como cajas (x, y, w, h, score).

    YOLO encuentra personas aunque estén de perfil (MediaPipe no). Se descartan
    las que llegan al borde de abajo o a un lateral: es público en primer plano
    de espaldas. La cabeza es la cara detectada dentro de la persona o, si no
    hay (perfil), la parte de arriba de su caja."""
    H, W = frame.shape[:2]
    r = _yolo()(frame, classes=[0], conf=cfg.get("person_min_conf", 0.35), verbose=False)[0]
    faces = webcam.detect_faces(frame)
    heads = []
    for (x1, y1, x2, y2), conf in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
        w, h = x2 - x1, y2 - y1
        if y2 > 0.97 * H:
            continue  # cortado por abajo: primer plano / público
        if (x1 < 0.01 * W or x2 > 0.99 * W) and w < 0.2 * W:
            continue  # cortado por un lateral
        inside = [f for f in faces if x1 <= f[0] + f[2] / 2 <= x2 and y1 <= f[1] + f[3] / 2 <= y1 + h * 0.5]
        if inside:
            f = max(inside, key=lambda f: f[2])
            heads.append((f[0], f[1], f[2], f[3], conf))
        else:
            hh = min(0.35 * h, 0.7 * w)
            heads.append((x1 + w / 2 - hh * 0.4, y1, hh * 0.8, hh, conf))
    return heads


def find_speakers(frames, W, cfg):
    """Cajas (x, y, w, h) de las personas presentes en la escena, de izquierda
    a derecha. Ignora caras quietas (pósters) y las que salen poco."""
    min_w = cfg.get("podcast_min_face_frac", 0.015) * W
    face_lists = [[f for f in detect_heads(fr, cfg) if f[2] >= min_w] for _, fr in frames]
    clusters = webcam._vote_clusters(face_lists, W, 1.0, top=6)
    people = []
    for faces, ratio, idx in clusters:
        if ratio < cfg.get("podcast_min_presence", 0.5):
            continue
        cxs = [f[0] + f[2] / 2 for f in faces]
        fw = float(np.median([f[2] for f in faces]))
        if len(faces) >= 3 and np.std(cxs) / max(1.0, fw) < 0.003:
            continue  # póster / dibujo
        people.append(tuple(float(np.median([f[i] for f in faces])) for i in range(4)))
    # Público: cabezas mucho más abajo que las de los que están en la mesa.
    if len(people) > 2:
        H = frames[0][1].shape[0]
        top = min(p[1] for p in people)
        people = [p for p in people if p[1] <= top + cfg.get("podcast_max_head_drop", 0.18) * H]
    return sorted(people, key=lambda b: b[0])


def _attribute(activity, loudness, floor, margin):
    """Por ventana: índice del que habla, "both" si hablan a la vez, o None."""
    out = []
    for i, row in enumerate(activity):
        if loudness and i < len(loudness) and loudness[i] < floor:
            out.append(None)
            continue
        order = np.argsort(row)[::-1]
        a, b = row[order[0]], (row[order[1]] if len(row) > 1 else 0.0)
        if a <= 0:
            out.append(None)
        elif b >= 0.6 and b >= a * 0.75:
            out.append("both")
        elif (a - b) / (a + b + 1e-6) >= margin:
            out.append(int(order[0]))
        else:
            out.append(None)
    return out


def timeline(verdicts, window_s, cfg):
    """Tramos [(inicio, fin, estado)] con estado = índice de persona o "split".

    Reglas: ventanas dudosas heredan el estado anterior; un estado nuevo
    necesita durar >= min_switch_seconds; si en ``rapid_window`` segundos hay
    >= ``rapid_switches`` cambios de persona, ese tramo pasa a split."""
    min_w = max(1, math.ceil(cfg.get("min_switch_seconds", 1.5) / window_s))
    # 1) rellenar dudas
    filled, cur = [], None
    for v in verdicts:
        cur = v if v is not None else cur
        filled.append(cur)
    first = next((v for v in filled if v is not None), 0)
    filled = [first if v is None else v for v in filled]
    # 2) intercambio rápido -> split
    rapid_w = max(1, int(cfg.get("rapid_window", 4.0) / window_s))
    rapid_n = cfg.get("rapid_switches", 3)
    raw = list(filled)
    for i in range(len(raw)):
        seg = raw[max(0, i - rapid_w // 2):i + rapid_w // 2 + 1]
        changes = sum(1 for a, b in zip(seg, seg[1:]) if a != b and a != "both" and b != "both")
        if changes >= rapid_n:
            filled[i] = "both"
    # 3) duración mínima de cada estado
    runs = []
    for v in filled:
        if runs and runs[-1][0] == v:
            runs[-1][1] += 1
        else:
            runs.append([v, 1])
    merged = []
    for v, n in runs:
        if merged and (n < min_w or merged[-1][0] == v):
            merged[-1][1] += n
        else:
            merged.append([v, n])
    # el primero, si es corto, se une al siguiente
    if len(merged) > 1 and merged[0][1] < min_w:
        merged[1][1] += merged[0][1]
        merged.pop(0)
    out, t = [], 0.0
    for v, n in merged:
        out.append((round(t, 2), round(t + n * window_s, 2), "split" if v == "both" else v))
        t += n * window_s
    return out


def _speaker_crop(box, W, H, aspect, cfg):
    """Recorte de proporción ``aspect`` centrado en una cara (con aire arriba)."""
    x, y, w, h = box
    # Nunca recortar menos del X% de la altura: en un plano general las caras
    # son pequeñas y un recorte ceñido saldría muy pixelado al ampliarlo.
    crop_h = min(H, max(h * cfg.get("podcast_face_scale", 4.2),
                        H * cfg.get("podcast_min_crop_frac", 0.6) * (OUT_W / OUT_H) / aspect))
    crop_w = crop_h * aspect
    if crop_w > W:
        crop_w, crop_h = W, W / aspect
    cx, cy = x + w / 2, y + h / 2
    cx0 = min(max(cx - crop_w / 2, 0), W - crop_w)
    cy0 = min(max(cy - crop_h * 0.38, 0), H - crop_h)
    e = lambda v: int(v) // 2 * 2
    return e(cx0), e(cy0), e(crop_w), e(crop_h)


def _venc():
    try:
        import ffmpeg_utils
        return ffmpeg_utils.video_encode_args(ffmpeg_utils.QUALITY)
    except Exception:
        return ["-c:v", "libx264", "-preset", "medium", "-crf", "18"]


def render(clip_path, out_path, cfg, frames=None):
    """Renderiza un tramo con 2+ personas. Devuelve el plan (con "ranges"
    relativos al tramo, para colocar los subtítulos)."""
    import active_speaker

    cap = cv2.VideoCapture(clip_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    W, H = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)), int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    cap.release()
    frames = frames or webcam.sample_frames(clip_path, 10, skip_edges=0.0)
    people = find_speakers(frames, W, cfg)
    if len(people) < 2:
        plan = face_track.render(clip_path, out_path, {**cfg, "debug_video": False})
        plan["ranges"] = [(0.0, dur, "track")]
        return plan

    ws = active_speaker.WINDOW_SECONDS
    activity = active_speaker.decode_activity(clip_path, 0, dur, people, fps)
    loud = active_speaker.audio_envelope(clip_path, 0, dur)
    norm = active_speaker.normalise_activity(activity)
    verdicts = _attribute(norm, loud, active_speaker.AUDIO_FLOOR, active_speaker.MIN_MARGIN)
    mode = cfg.get("podcast_mode", "auto")  # auto | switch | split
    if mode == "split":
        tl = [(0.0, dur, "split")]
    else:
        tl = timeline(verdicts, ws, cfg)
        if mode == "switch":
            # sin split: el tramo split se queda con quien más habla en él
            tl = [(a, b, s if s != "split" else _dominant(verdicts, a, b, ws)) for a, b, s in tl]
    if tl:
        tl[-1] = (tl[-1][0], dur, tl[-1][2])

    # Las dos personas que más hablan (para el split).
    counts = [sum(1 for v in verdicts if v == i) for i in range(len(people))]
    duo = sorted(sorted(range(len(people)), key=lambda i: -counts[i])[:2])

    tmp = tempfile.mkdtemp(prefix="pod_")
    parts = []
    try:
        for k, (a, b, state) in enumerate(tl):
            part = os.path.join(tmp, f"p{k}.mp4")
            if state == "split":
                half = OUT_H // 2
                ca = _speaker_crop(people[duo[0]], W, H, OUT_W / half, {**cfg, "podcast_face_scale": 3.0})
                cb = _speaker_crop(people[duo[1]], W, H, OUT_W / half, {**cfg, "podcast_face_scale": 3.0})
                vf = (f"[0:v]split=2[a][b];[a]crop={ca[2]}:{ca[3]}:{ca[0]}:{ca[1]},scale={OUT_W}:{half},setsar=1[t];"
                      f"[b]crop={cb[2]}:{cb[3]}:{cb[0]}:{cb[1]},scale={OUT_W}:{half},setsar=1[d];"
                      f"[t][d]vstack,format=yuv420p[v]")
                args = ["-filter_complex", vf, "-map", "[v]"]
            else:
                c = _speaker_crop(people[state], W, H, OUT_W / OUT_H, cfg)
                args = ["-vf", f"crop={c[2]}:{c[3]}:{c[0]}:{c[1]},scale={OUT_W}:{OUT_H},setsar=1,format=yuv420p"]
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", clip_path, "-ss", f"{a:.3f}",
                            "-to", f"{b:.3f}", *args, "-an", *_venc(), part], check=True)
            parts.append(part)
        lst = os.path.join(tmp, "list.txt")
        with open(lst, "w") as f:
            for p in parts:
                f.write(f"file '{p.replace(os.sep, '/')}'\n")
        joined = os.path.join(tmp, "joined.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                        "-c", "copy", joined], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", joined, "-i", clip_path,
                        "-map", "0:v", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                        "-shortest", "-movflags", "+faststart", out_path], check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    shares = [round(c / max(1, sum(counts)), 2) for c in counts]
    return {"mode": "podcast", "people": [[round(v) for v in p] for p in people],
            "talk_share": shares, "timeline": tl,
            "ranges": [(a, b, "split" if s == "split" else "track") for a, b, s in tl]}


def _dominant(verdicts, a, b, ws):
    seg = [v for v in verdicts[int(a / ws):int(b / ws) + 1] if isinstance(v, int)]
    return max(set(seg), key=seg.count) if seg else 0

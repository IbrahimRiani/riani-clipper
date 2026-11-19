"""Layout GAMING: webcam arriba, juego abajo, en 1080x1920.

Geometría (decidida una vez por video, para que la costura donde van los
subtítulos no salte de un clip a otro):

    ┌──────────────┐  0
    │   WEBCAM     │  cam_h  (~30-40 %, según la forma de la cam)
    ├──────────────┤  <- subtítulos aquí (la "costura")
    │    JUEGO     │  la ventana de juego sin overlay ni webcam
    │              │
    ├──────────────┤
    │ juego borroso│  relleno si el juego no llega al final
    └──────────────┘  1920

Por clip se decide: si la cam está visible (si no, solo juego), quitar barras
negras del juego y hacia dónde mirar en juegos panorámicos (centro de la acción,
esquivando la webcam para que no salga dos veces).
"""
import json
import os
import subprocess

import cv2
import numpy as np

from . import webcam

OUT_W, OUT_H = 1080, 1920


def _even(v):
    return int(v) // 2 * 2


def cam_panel_height(cam_rect, cfg):
    """Alto del panel de la cam: el que respeta su forma, entre un mínimo y un máximo."""
    _, _, w, h = cam_rect
    lo = cfg.get("cam_min_frac", 0.30) * OUT_H
    hi = cfg.get("cam_max_frac", 0.40) * OUT_H
    return _even(min(hi, max(lo, OUT_W * h / w)))


def _crop_to_aspect(rect, aspect, center=None):
    """Recorta ``rect`` (x, y, w, h) a la proporción ``aspect`` (ancho/alto),
    centrado en ``center`` (x, y) si se da, sin salirse del rectángulo."""
    x, y, w, h = rect
    if w / h > aspect:            # sobra ancho
        nw, nh = h * aspect, h
    else:                         # sobra alto
        nw, nh = w, w / aspect
    cx, cy = center if center else (x + w / 2, y + h / 2)
    nx = min(max(cx - nw / 2, x), x + w - nw)
    ny = min(max(cy - nh / 2, y), y + h - nh)
    return (_even(nx), _even(ny), _even(nw), _even(nh))


def _inset(rect, W, H, frac):
    """Mete el rectángulo de la cam hacia dentro en los lados que NO tocan el
    borde del vídeo. La compresión difumina el borde de la cam con el juego y,
    al ampliar, esos 1-2 píxeles se veían como un hilo de gameplay."""
    x, y, w, h = rect
    d = max(3, int(round(frac * min(w, h))))
    x0 = x + (d if x > 2 else 0)
    y0 = y + (d if y > 2 else 0)
    x1 = x + w - (d if x + w < W - 2 else 0)
    y1 = y + h - (d if y + h < H - 2 else 0)
    return (x0, y0, x1 - x0, y1 - y0)


def trim_black_bars(frames, rect, thr=18):
    """Quita barras negras (letterbox) de ``rect`` mirando el máximo de brillo."""
    x, y, w, h = rect
    stack = np.stack([cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)[y:y + h, x:x + w] for _, fr in frames])
    mx = stack.max(axis=0)
    rows = np.where(np.percentile(mx, 90, axis=1) > thr)[0]
    cols = np.where(np.percentile(mx, 90, axis=0) > thr)[0]
    if len(rows) < h * 0.4 or len(cols) < w * 0.4:
        return rect
    return (x + int(cols[0]), y + int(rows[0]), int(cols[-1] - cols[0] + 1), int(rows[-1] - rows[0] + 1))


def action_center(frames, rect, cam_rect=None):
    """Centro (x, y) de lo que se mueve dentro de ``rect`` a lo largo del clip."""
    x, y, w, h = rect
    acc = np.zeros((h, w), np.float32)
    prev = None
    for _, fr in frames:
        g = cv2.cvtColor(fr[y:y + h, x:x + w], cv2.COLOR_BGR2GRAY).astype(np.float32)
        if prev is not None:
            acc += np.abs(g - prev)
        prev = g
    if cam_rect:
        cx, cy, cw, ch = cam_rect
        acc[max(0, cy - y):max(0, cy - y + ch), max(0, cx - x):max(0, cx - x + cw)] = 0
    acc = cv2.GaussianBlur(acc, (0, 0), max(w, h) / 20)
    if acc.sum() <= 0:
        return (x + w / 2, y + h / 2)
    ys, xs = np.mgrid[0:h, 0:w]
    tot = acc.sum()
    mx, my = (acc * xs).sum() / tot, (acc * ys).sum() / tot
    # Mezcla con el centro: la acción manda, pero sin irse a una esquina.
    k = 0.6
    return (x + k * mx + (1 - k) * w / 2, y + k * my + (1 - k) * h / 2)


def _overlap(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    return ix * iy


def _avoid_cam(crop, game, cam):
    """Desplaza horizontalmente el recorte del juego para no incluir la cam."""
    if not cam or not _overlap(crop, cam):
        return crop
    x, y, w, h = crop
    gx, _, gw, _ = game
    cx, _, cw, _ = cam
    options = []
    if cx + cw + w <= gx + gw:                 # a la derecha de la cam
        options.append(cx + cw)
    if cx - w >= gx:                           # a la izquierda de la cam
        options.append(cx - w)
    if not options:
        return crop  # no cabe sin solaparse: mejor algo de cam que nada de juego
    nx = min(options, key=lambda o: abs(o - x))
    return (_even(nx), y, w, h)


def plan_clip(clip_path, analysis, cfg):
    """Decide la geometría de un clip. Devuelve un dict serializable (debug)."""
    W, H = analysis["width"], analysis["height"]
    frames = webcam.sample_frames(clip_path, cfg.get("clip_sample_frames", 24), skip_edges=0.0)
    cam = analysis.get("webcam")
    cam_rect = tuple(cam["rect"]) if cam else None
    plan = {"cam_present": None, "mode": "game_only"}
    if cam_rect:
        x, y, w, h = cam_rect
        hits = 0
        sub = frames[:: max(1, len(frames) // 8)]
        tpl = None
        if cam.get("template") and os.path.exists(cam["template"]):
            tpl = cv2.imread(cam["template"])
        # ¿La cam se ha movido en este clip? (en un VOD de 90 min Rubius la bajó
        # 156 px a mitad de directo). Buscarla por su aspecto en todo el frame.
        if tpl is not None:
            ms = [webcam.find_by_template(fr, tpl) for _, fr in sub]
            ms = [m for m in ms if m and m[2] >= cfg.get("cam_template_min_score", 0.35)]
            if len(ms) >= max(2, 0.5 * len(sub)):
                mx = float(np.median([m[0] for m in ms]))
                my = float(np.median([m[1] for m in ms]))
                same = sum(1 for m in ms if abs(m[0] - mx) <= 0.03 * W and abs(m[1] - my) <= 0.03 * H)
                if same >= max(2, 0.5 * len(sub)) and (abs(mx - x) > 0.03 * W or abs(my - y) > 0.03 * H):
                    dx, dy = int(mx) - x, int(my) - y
                    cam = {**cam, "rect": [int(mx), int(my), w, h],
                           "face": [cam["face"][0] + dx, cam["face"][1] + dy, *cam["face"][2:]]}
                    cam_rect = tuple(cam["rect"])
                    x, y = cam_rect[0], cam_rect[1]
                    plan["cam_moved_to"] = [x, y]
        for _, fr in sub:
            if any(x <= f[0] + f[2] / 2 <= x + w and y <= f[1] + f[3] / 2 <= y + h
                   for f in webcam.detect_faces(fr)):
                hits += 1
            elif tpl is not None:
                # Sin cara (mira abajo / de lado): ¿se ve la cam por su aspecto
                # en su sitio? En un VOD de 90 min, 5 clips salían "solo juego"
                # con la cam bien visible.
                m = webcam.find_by_template(fr, tpl)
                if m and m[2] >= cfg.get("cam_template_min_score", 0.35) \
                        and abs(m[0] - x) <= 0.03 * W and abs(m[1] - y) <= 0.03 * H:
                    hits += 1
        plan["cam_present"] = round(hits / max(1, len(sub)), 2)
        if plan["cam_present"] >= cfg.get("cam_present_min", 0.3):
            plan["mode"] = "cam_top"

    game = trim_black_bars(frames, tuple(analysis["game"]["rect"]))
    plan["game_rect"] = list(game)
    center = action_center(frames, game, cam_rect)
    plan["action_center"] = [round(center[0]), round(center[1])]

    geo = cfg.get("force_geometry") if plan["mode"] == "cam_top" and cfg.get("force_geometry")         else video_geometry(analysis, cfg, with_cam=plan["mode"] == "cam_top")
    plan.update(geo)
    if plan["mode"] == "cam_top":
        face = cam["face"]
        face_c = (face[0] + face[2] / 2, face[1] + face[3] / 2)
        inner = _inset(cam_rect, W, H, cfg.get("cam_inset_frac", 0.025))
        plan["cam_crop"] = list(_crop_to_aspect(inner, OUT_W / geo["cam_h"], face_c))
    # El juego de ESTE clip (sin barras negras) se recorta a la forma del panel.
    crop = _crop_to_aspect(game, OUT_W / geo["game_h"], center)
    crop = _avoid_cam(crop, game, cam_rect if plan["mode"] == "cam_top" else None)
    plan["game_crop"] = list(crop)
    plan["fill"] = cfg.get("fill", "blur")
    return plan


def video_geometry(analysis, cfg, with_cam=True):
    """Tamaños de los paneles, iguales para todos los clips del video (así la
    costura de los subtítulos no salta). Se calculan con la zona de juego global.

    Devuelve cam_y, cam_h, game_y, game_h (en píxeles de la salida 1080x1920)."""
    gx, gy, gw, gh = analysis["game"]["rect"]
    ga = gw / gh
    cam = analysis.get("webcam") if with_cam else None
    cam_h = cam_panel_height(cam["rect"], cfg) if cam else 0
    avail = OUT_H - cam_h
    if ga <= cfg.get("game_fit_max_aspect", 1.6):
        # Juego "cuadradote" (retro, 4:3): casi entero, como mucho pierde
        # game_fit_keep por los lados (los HUD suelen estar dentro).
        crop_aspect = max(OUT_W / avail, ga * cfg.get("game_fit_keep", 0.9))
    else:
        # Panorámico (16:9): recortado, sin perder más de game_min_keep.
        keep = cfg.get("game_min_keep", 0.5) if cam else cfg.get("game_min_keep_no_cam", 0.42)
        crop_aspect = max(OUT_W / avail, ga * keep)
    game_h = _even(min(avail, OUT_W / crop_aspect))
    spare = OUT_H - cam_h - game_h
    if cam and spare > 0:
        # Lo que sobra, primero a la webcam (hasta cam_grow_max_frac)...
        grow = min(spare, _even(cfg.get("cam_grow_max_frac", 0.42) * OUT_H) - cam_h)
        if grow > 0:
            cam_h += _even(grow)
            spare = OUT_H - cam_h - game_h
    # ...y el resto repartido arriba y abajo (bloque centrado).
    top = _even(max(0, spare) / 2) if cfg.get("center_stack", True) else 0
    return {"cam_y": top, "cam_h": cam_h, "game_y": top + cam_h, "game_h": game_h}


def _filter_graph(plan):
    gx, gy, gw, gh = plan["game_crop"]
    parts = ["[0:v]split=3[a][b][c]"]
    # Fondo: el juego difuminado y oscurecido, por si queda hueco.
    if plan["fill"] == "black":
        parts.append(f"color=black:s={OUT_W}x{OUT_H}[bg0]")
        parts.append("[c]nullsink")
    else:
        parts.append(f"[c]crop={gw}:{gh}:{gx}:{gy},scale=270:480:force_original_aspect_ratio=increase,"
                     f"crop=270:480,boxblur=12:2,eq=brightness=-0.12,scale={OUT_W}:{OUT_H},setsar=1[bg0]")
    parts.append(f"[b]crop={gw}:{gh}:{gx}:{gy},scale={OUT_W}:{plan['game_h']},setsar=1[game]")
    if plan["mode"] == "cam_top":
        cx, cy, cw, ch = plan["cam_crop"]
        parts.append(f"[a]crop={cw}:{ch}:{cx}:{cy},scale={OUT_W}:{plan['cam_h']},setsar=1[cam]")
        parts.append(f"[bg0][cam]overlay=0:{plan['cam_y']}[t1]")
        parts.append(f"[t1][game]overlay=0:{plan['game_y']},format=yuv420p[v]")
    else:
        parts.append("[a]nullsink")
        parts.append(f"[bg0][game]overlay=0:{plan['game_y']},format=yuv420p[v]")
    return ";".join(parts)


def _encode_args():
    try:
        import ffmpeg_utils
        return ffmpeg_utils.video_encode_args(ffmpeg_utils.QUALITY)
    except Exception:
        return ["-c:v", "libx264", "-preset", "medium", "-crf", "18"]


def render(clip_path, out_path, analysis, cfg):
    """Renderiza el clip con el layout gaming y escribe el sidecar de layout
    (para que los subtítulos vayan en la costura) y un JSON de debug."""
    plan = plan_clip(clip_path, analysis, cfg)
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", clip_path,
           "-filter_complex", _filter_graph(plan), "-map", "[v]", "-map", "0:a?",
           *_encode_args(), "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out_path]
    subprocess.run(cmd, check=True)

    import layout_ranges
    cap = cv2.VideoCapture(out_path)
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / (cap.get(cv2.CAP_PROP_FPS) or 30)
    cap.release()
    layout = "split" if plan["mode"] == "cam_top" else "track"
    layout_ranges.write(out_path, [(0.0, dur, layout)])
    with open(out_path + ".gaming.json", "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)
    if cfg.get("debug_video", True):
        debug_video(clip_path, out_path + ".debug.mp4", plan, analysis)
    return plan


def debug_video(clip_path, out_path, plan, analysis):
    """El clip original a 960 px con las cajas: cam (verde), cara (amarillo),
    zona de juego (azul) y recorte del juego que se usa (rojo)."""
    W = analysis["width"]
    s = 960 / W
    boxes = []
    if analysis.get("webcam"):
        boxes.append((analysis["webcam"]["rect"], "green"))
        boxes.append((analysis["webcam"]["face"], "yellow"))
    if plan.get("cam_crop"):
        boxes.append((plan["cam_crop"], "lime"))
    boxes.append((plan["game_rect"], "blue"))
    boxes.append((plan["game_crop"], "red"))
    draws = ",".join(
        f"drawbox=x={int(b[0] * s)}:y={int(b[1] * s)}:w={int(b[2] * s)}:h={int(b[3] * s)}:color={c}:t=3"
        for b, c in boxes)
    vf = f"scale=960:-2,{draws}"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", clip_path, "-vf", vf, "-an",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", out_path], check=True)

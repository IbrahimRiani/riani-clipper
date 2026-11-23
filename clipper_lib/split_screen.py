"""Layout PANTALLA PARTIDA: la persona en un panel lateral a toda altura y el
contenido (otro directo, un vídeo, una web) en el resto, separados por una
línea vertical. Pasa a 1080x1920 como persona arriba y contenido abajo, 50/50:

    ┌──────────────┐  0
    │   PERSONA    │  recorte del panel lateral alrededor de la cara
    ├──────────────┤  960  <- subtítulos aquí
    │  CONTENIDO   │  el otro panel, casi entero (zoom mínimo)
    └──────────────┘  1920

Detección: en cada frame se busca la columna con un salto de color fuerte en
casi toda la altura (la costura entre paneles). La costura puede moverse poco a
poco (el streamer redimensiona el panel), por eso se mide frame a frame y no
con la media. Para no confundirla con el borde de una ventana de juego, además
tiene que haber una cara grande entera a un lado de la costura.
"""
import json
import os
import subprocess

import cv2
import numpy as np

from . import webcam

OUT_W, OUT_H = 1080, 1920
HALF = OUT_H // 2


def seam(frame, lo=0.15, hi=0.85):
    """(x, cobertura): la columna entre lo y hi del ancho con más filas con un
    salto de color fuerte entre la izquierda y la derecha (4 px)."""
    W = frame.shape[1]
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    d = np.abs(g[:, 4:, :] - g[:, :-4, :]).sum(axis=2)
    p = np.pad((d > 25).mean(axis=0), (2, 2))
    a, b = int(lo * W), int(hi * W)
    i = int(np.argmax(p[a:b])) + a
    return i, float(p[i])


def _frame_hit(frame, cfg):
    """Si el frame es pantalla partida: (x_costura, {lado: cara más grande})
    o None. El contenido también puede tener caras (otro streamer), así que se
    guardan las de los dos lados y el llamador decide cuál es la persona."""
    W = frame.shape[1]
    x, cov = seam(frame)
    if cov < cfg.get("split_seam_min_cover", 0.92):
        return None
    sides = {}
    for f in sorted(webcam.detect_faces(frame), key=lambda f: -f[2]):
        if f[2] < cfg.get("split_min_face_frac", 0.1) * W:
            continue
        side = "left" if f[0] + f[2] <= x else "right" if f[0] >= x else None
        if side and side not in sides:
            sides[side] = f
    return (x, sides) if sides else None


def detect(frames, cfg):
    """Análisis global sobre los frames muestreados. Devuelve un dict con el
    lado de la persona y las costuras vistas, o None si no es pantalla partida."""
    if not cfg.get("split_enabled", True) or not frames:
        return None
    step = max(1, len(frames) // cfg.get("split_sample_frames", 20))
    sample = frames[::step]
    hits = [h for h in (_frame_hit(f, cfg) for _, f in sample) if h]
    W = frames[0][1].shape[1]
    count = {s: sum(s in h[1] for h in hits) for s in ("left", "right")}
    # La persona es el lado con cara en más frames; si empatan, el panel más
    # estrecho (el contenido suele ocupar más).
    mid = np.median([h[0] for h in hits]) if hits else W / 2
    side = max(count, key=lambda s: (count[s], (mid < W / 2) == (s == "left")))
    ratio = count[side] / len(sample)
    if ratio < cfg.get("split_min_ratio", 0.5):
        return None
    xs = [h[0] for h in hits if side in h[1]]
    return {"side": side, "ratio": round(ratio, 2),
            "seam_min": int(min(xs)), "seam_max": int(max(xs)),
            "seam_frac": round(float(np.median(xs)) / W, 3)}


def _crop_aspect(x0, x1, H, aspect, cy):
    """Recorte de aspecto w/h dentro de la franja [x0, x1] y toda la altura,
    centrado en vertical en cy y en horizontal en la franja."""
    w = x1 - x0
    h = int(round(w / aspect))
    if h > H:
        h = H
        w = int(round(H * aspect))
    w -= w % 2
    h -= h % 2
    x = x0 + ((x1 - x0) - w) // 2
    y = int(min(max(cy - h / 2, 0), H - h))
    return [int(x), y, w, h]


def plan_clip(clip_path, info, cfg):
    """Recortes fijos para todo el clip, o None si el clip no es pantalla
    partida (entonces se usa el layout chatting)."""
    cap = cv2.VideoCapture(clip_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    dur = cap.get(cv2.CAP_PROP_FRAME_COUNT) / fps
    cap.release()
    n = int(min(30, max(6, dur)))
    frames = webcam.sample_frames(clip_path, n, 0, dur, skip_edges=0)
    if not frames:
        return None
    H, W = frames[0][1].shape[:2]
    hits = [(h[0], info["side"], h[1][info["side"]])
            for h in (_frame_hit(f, cfg) for _, f in frames) if h and info["side"] in h[1]]
    if len(hits) / len(frames) < cfg.get("split_clip_min_ratio", 0.6):
        return None
    # Fuera costuras sueltas (un frame de transición o un borde del contenido
    # que casualmente cruza todo el alto): solo las cercanas a la mediana.
    med = float(np.median([h[0] for h in hits]))
    hits = [h for h in hits if abs(h[0] - med) <= cfg.get("split_seam_max_jump", 0.06) * W]
    xs = [h[0] for h in hits]
    pad = 6
    # La costura se mueve un poco dentro del clip: el panel de la persona hasta
    # la costura más cerrada y el contenido desde la más abierta, así no entra
    # nunca un trozo del otro panel.
    if info["side"] == "left":
        person = (0, min(xs) - pad)
        content = (max(xs) + pad, W)
    else:
        person = (max(xs) + pad, W)
        content = (0, min(xs) - pad)
    aspect = OUT_W / HALF
    face_cy = float(np.median([h[2][1] + h[2][3] / 2 for h in hits]))
    # Cara al ~40 % de la altura del recorte: aire arriba para pelo/gorra y
    # hombros/micro debajo.
    ph = (person[1] - person[0]) / aspect
    person_crop = _crop_aspect(*person, H, aspect, face_cy + 0.1 * min(ph, H))
    content_crop = _crop_aspect(*content, H, aspect, H / 2)
    return {"mode": "split_screen", "side": info["side"], "seam_xs": [int(min(xs)), int(max(xs))],
            "person_crop": person_crop, "content_crop": content_crop,
            "hit_ratio": round(len(hits) / len(frames), 2), "duration": dur}


def _filter_graph(plan):
    px, py, pw, ph = plan["person_crop"]
    cx, cy, cw, ch = plan["content_crop"]
    return ";".join([
        "[0:v]split=2[a][b]",
        f"[a]crop={pw}:{ph}:{px}:{py},scale={OUT_W}:{HALF}:force_original_aspect_ratio=increase,"
        f"crop={OUT_W}:{HALF},setsar=1[top]",
        f"[b]crop={cw}:{ch}:{cx}:{cy},scale={OUT_W}:{HALF}:force_original_aspect_ratio=increase,"
        f"crop={OUT_W}:{HALF},setsar=1[bot]",
        "[top][bot]vstack=inputs=2,format=yuv420p[v]",
    ])


def render(clip_path, out_path, analysis, cfg):
    """Renderiza un clip. Devuelve el plan, o None si el clip no tiene la
    pantalla partida (el llamador usa entonces el layout chatting)."""
    from .gaming_render import _encode_args
    plan = plan_clip(clip_path, analysis["split_screen"], cfg)
    if plan is None:
        return None
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", clip_path,
           "-filter_complex", _filter_graph(plan), "-map", "[v]", "-map", "0:a?",
           *_encode_args(), "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", out_path]
    subprocess.run(cmd, check=True)
    import layout_ranges
    layout_ranges.write(out_path, [(0.0, plan["duration"], "split")])
    with open(out_path + ".split.json", "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)
    return plan

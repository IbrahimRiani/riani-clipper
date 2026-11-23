"""Layout PANTALLA PARTIDA: dos paneles lado a lado. Pasa a 1080x1920 como un
panel arriba y el otro abajo, 50/50:

    ┌──────────────┐  0
    │   PERSONA    │  recorte del panel alrededor de la cara
    ├──────────────┤  960  <- subtítulos aquí
    │ CONTENIDO o  │  el otro panel: si hay otra persona (videochat, debate a
    │ OTRA PERSONA │  dos cámaras), recortado en su cara; si no, casi entero
    └──────────────┘  1920

Casos:
- Streamer en un panel lateral a toda altura + contenido (otro directo, un
  vídeo): costura vertical en casi toda la altura y una cara grande a un lado.
- Videochat / dos webcams lado a lado (Omegle, Umingle, debates): dos caras,
  una a cada lado, con una costura vertical entre ellas en la franja de las
  cámaras (no hace falta que cruce toda la pantalla: alrededor suele haber
  navegador, chat, botones).

La costura puede moverse poco a poco (el streamer redimensiona el panel), por
eso se mide frame a frame y no con la media.
"""
import json
import os
import subprocess

import cv2
import numpy as np

from . import webcam

OUT_W, OUT_H = 1080, 1920
HALF = OUT_H // 2
ASPECT = OUT_W / HALF


def _col_cover(frame, rows=None):
    """Por columna, fracción de filas (de ``rows``) con salto de color fuerte
    entre la izquierda y la derecha (4 px)."""
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB).astype(np.float32)
    if rows:
        g = g[max(0, rows[0]):rows[1]]
    d = np.abs(g[:, 4:, :] - g[:, :-4, :]).sum(axis=2)
    return np.pad((d > 25).mean(axis=0), (2, 2))


def seam(frame, lo=0.15, hi=0.85, rows=None):
    """(x, cobertura) de la mejor costura vertical entre lo y hi del ancho
    (fracciones) o entre los píxeles lo..hi si son > 1."""
    W = frame.shape[1]
    p = _col_cover(frame, rows)
    a = int(lo * W) if lo <= 1 else int(lo)
    b = int(hi * W) if hi <= 1 else int(hi)
    if b - a < 3:
        return a, 0.0
    i = int(np.argmax(p[a:b])) + a
    return i, float(p[i])


def _frame_hit(frame, cfg):
    """Si el frame es pantalla partida: {"x", "sides": {lado: cara}} o None."""
    H, W = frame.shape[:2]
    faces = sorted((f for f in webcam.detect_faces(frame)
                    if f[2] >= cfg.get("split_min_face_frac_any", 0.06) * W), key=lambda f: -f[2])
    if not faces:
        return None

    def sides_for(x):
        sides = {}
        for f in faces:
            side = "left" if f[0] + f[2] <= x else "right" if f[0] >= x else None
            if side and side not in sides:
                sides[side] = f
        return sides

    # 1) Costura a toda altura + una cara grande (panel del streamer).
    x, cov = seam(frame)
    if cov >= cfg.get("split_seam_min_cover", 0.92):
        sides = sides_for(x)
        if any(f[2] >= cfg.get("split_min_face_frac", 0.1) * W for f in sides.values()):
            return {"x": x, "sides": sides}
    # 2) Dos caras separadas en horizontal con una costura entre ellas en la
    #    franja de las caras (dos webcams lado a lado).
    for i, a in enumerate(faces):
        for b in faces[i + 1:]:
            left, right = (a, b) if a[0] < b[0] else (b, a)
            if left[0] + left[2] >= right[0]:
                continue
            fh = max(left[3], right[3])
            rows = (int(min(left[1], right[1])), int(max(left[1] + left[3], right[1] + right[3])))
            x, cov = seam(frame, left[0] + left[2] + 2, right[0] - 2, rows=(max(0, rows[0]), min(H, rows[1])))
            if cov >= cfg.get("split_local_min_cover", 0.85):
                return {"x": x, "sides": {"left": left, "right": right}}
            return None  # solo se prueba el par de las dos caras más grandes
    return None


def detect(frames, cfg):
    """Análisis global sobre los frames muestreados. Devuelve un dict con qué
    va arriba y las costuras vistas, o None si no es pantalla partida."""
    if not cfg.get("split_enabled", True) or not frames:
        return None
    step = max(1, len(frames) // cfg.get("split_sample_frames", 20))
    sample = frames[::step]
    hits = [h for h in (_frame_hit(f, cfg) for _, f in sample) if h]
    if not hits:
        return None
    W = frames[0][1].shape[1]
    count = {s: sum(s in h["sides"] for h in hits) for s in ("left", "right")}
    min_ratio = cfg.get("split_min_ratio", 0.5)
    mid = np.median([h["x"] for h in hits])
    if min(count.values()) / len(sample) >= min_ratio:
        # Una persona a cada lado: la de la izquierda arriba.
        top, kind = "left", "dual"
    else:
        # La persona es el lado con cara en más frames; si empatan, el panel
        # más estrecho (el contenido suele ocupar más).
        top = max(count, key=lambda s: (count[s], (mid < W / 2) == (s == "left")))
        kind = "single"
    ratio = count[top] / len(sample)
    if ratio < min_ratio:
        return None
    xs = [h["x"] for h in hits if top in h["sides"]]
    return {"side": top, "kind": kind, "ratio": round(ratio, 2),
            "seam_min": int(min(xs)), "seam_max": int(max(xs)),
            "seam_frac": round(float(np.median(xs)) / W, 3)}


def _panel_rows(frames, x0, x1, face, H):
    """Borde de arriba y de abajo del panel de una cámara: la fila más cercana
    a la cara con un salto de color en casi todo el ancho del panel. Si no hay
    (panel a toda altura), 0 y H."""
    fx, fy, fw, fh = face
    # Solo las columnas alrededor de la cara: a los lados suele haber chat o
    # texto encima del borde y bajaría la medida.
    cx = fx + fw / 2
    c0, c1 = int(max(x0, cx - 1.6 * fw)), int(min(x1, cx + 1.6 * fw))
    tops, bots = [], []
    for _, f in frames[:12]:
        g = cv2.cvtColor(f[:, c0:c1], cv2.COLOR_BGR2LAB).astype(np.float32)
        d = np.abs(g[4:] - g[:-4]).sum(axis=2)
        cov = np.pad((d > 25).mean(axis=1), (2, 2))
        strong = np.where(cov >= 0.8)[0]
        above = strong[strong < fy - 0.6 * fh]
        below = strong[strong > fy + fh * 1.15]
        tops.append(int(above.max()) + 3 if len(above) else 0)
        bots.append(int(below.min()) - 3 if len(below) else H)
    return int(np.median(tops)), int(np.median(bots))


def _panel_cols(frames, x0, x1, y0, y1, face):
    """Borde izquierdo y derecho del panel (dentro de [x0, x1]): la columna más
    cercana a la cara con salto de color en casi toda la altura del panel.
    Evita franjas negras o de la web a los lados de una cámara más estrecha."""
    fx, fy, fw, fh = face
    if y1 - y0 < 20:
        return x0, x1
    lefts, rights = [], []
    for _, f in frames[:12]:
        cov = _col_cover(f, (y0 + 4, y1 - 4))
        strong = np.where(cov >= 0.85)[0]
        lft = strong[(strong >= x0) & (strong < fx - 0.6 * fw)]
        rgt = strong[(strong <= x1) & (strong > fx + fw * 1.6)]
        lefts.append(int(lft.max()) + 3 if len(lft) else x0)
        rights.append(int(rgt.min()) - 3 if len(rgt) else x1)
    return int(np.median(lefts)), int(np.median(rights))


def _face_crop(x0, x1, y0, y1, face):
    """Recorte 1080:960 lo más grande posible dentro del panel, centrado en la
    cara en horizontal y con la cara al ~40 % de la altura."""
    fx, fy, fw, fh = face
    w, h = x1 - x0, y1 - y0
    if w / h > ASPECT:
        w = int(h * ASPECT)
    else:
        h = int(w / ASPECT)
    w -= w % 2
    h -= h % 2
    cx, cy = fx + fw / 2, fy + fh / 2
    x = int(min(max(cx - w / 2, x0), x1 - w))
    y = int(min(max(cy - 0.4 * h, y0), y1 - h))
    return [x, y, w, h]


def _content_crop(x0, x1, H):
    """Recorte 1080:960 centrado del panel de contenido, a toda altura si cabe."""
    w = x1 - x0
    h = int(round(w / ASPECT))
    if h > H:
        h = H
        w = int(round(H * ASPECT))
    w -= w % 2
    h -= h % 2
    return [int(x0 + ((x1 - x0) - w) // 2), int((H - h) // 2), w, h]


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
    top = info["side"]
    hits = [h for h in (_frame_hit(f, cfg) for _, f in frames) if h and top in h["sides"]]
    if len(hits) / len(frames) < cfg.get("split_clip_min_ratio", 0.6):
        return None
    # Fuera costuras sueltas (un frame de transición o un borde del contenido
    # que casualmente cruza todo el alto): solo las cercanas a la mediana.
    med = float(np.median([h["x"] for h in hits]))
    hits = [h for h in hits if abs(h["x"] - med) <= cfg.get("split_seam_max_jump", 0.06) * W]
    xs = [h["x"] for h in hits]
    pad = 6
    # La costura se mueve un poco dentro del clip: cada panel hasta la costura
    # más cerrada para él, así no entra nunca un trozo del otro panel.
    region = {"left": (0, min(xs) - pad), "right": (max(xs) + pad, W)}
    crops = {}
    for side in ("left", "right"):
        faces = [h["sides"][side] for h in hits if side in h["sides"]]
        x0, x1 = region[side]
        if side == top or len(faces) >= 0.4 * len(hits):
            face = tuple(float(np.median([f[k] for f in faces])) for k in range(4))
            y0, y1 = _panel_rows(frames, x0, x1, face, H)
            x0, x1 = _panel_cols(frames, x0, x1, y0, y1, face)
            crops[side] = _face_crop(x0, x1, y0, y1, face)
        else:
            crops[side] = _content_crop(x0, x1, H)
    bottom = "right" if top == "left" else "left"
    return {"mode": "split_screen", "side": top, "kind": info.get("kind", "single"),
            "seam_xs": [int(min(xs)), int(max(xs))],
            "person_crop": crops[top], "content_crop": crops[bottom],
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

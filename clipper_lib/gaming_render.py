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
                k = float(np.median([m[3] for m in ms]))
                moved = abs(mx - x) > 0.03 * W or abs(my - y) > 0.03 * H or abs(k - 1) > 0.1
                cand = [int(mx), int(my), int(w * k), int(h * k)]
                if same >= max(2, 0.5 * len(sub)) and moved and                         webcam.face_fits(sub, cand, cam["face"][2] * k):
                    # Movida y/o agrandada (escena "ampliada" del streamer).
                    fx, fy, fw, fh = cam["face"][:4]
                    nw, nh = int(w * k), int(h * k)
                    cam = {**cam, "rect": [int(mx), int(my), nw, nh],
                           "face": [mx + (fx - x) * k, my + (fy - y) * k, fw * k, fh * k]}
                    cam_rect = tuple(cam["rect"])
                    x, y, w, h = cam_rect
                    plan["cam_moved_to"] = [x, y, w, h]
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
    full = cfg.get("game_view", "full") == "full"
    if full:
        # Zoom fijo al CENTRO de la pantalla: sin seguir la acción, sin
        # movimientos de cámara (preferencia del usuario).
        center = (game[0] + game[2] / 2, game[1] + game[3] / 2)
    crop = _crop_to_aspect(game, OUT_W / geo["game_h"], center)
    if not full:
        crop = _avoid_cam(crop, game, cam_rect if plan["mode"] == "cam_top" else None)
    plan["game_crop"] = list(crop)
    plan["game_view"] = "full" if full else "crop"
    if not full and plan["mode"] == "cam_top" and cam_rect and _overlap(crop, cam_rect):
        # No se pudo esquivar (cam muy grande): se difumina dentro del juego
        # para que no salga dos veces.
        ix0, iy0 = max(crop[0], cam_rect[0]), max(crop[1], cam_rect[1])
        ix1 = min(crop[0] + crop[2], cam_rect[0] + cam_rect[2])
        iy1 = min(crop[1] + crop[3], cam_rect[1] + cam_rect[3])
        plan["cam_in_game"] = [_even(ix0 - crop[0]), _even(iy0 - crop[1]),
                               _even(ix1 - ix0), _even(iy1 - iy0)]
    plan["fill"] = cfg.get("fill", "blur")
    return plan


def video_geometry(analysis, cfg, with_cam=True):
    """Tamaños de los paneles, iguales para todos los clips del video (así la
    costura de los subtítulos no salta). Se calculan con la zona de juego global.

    Devuelve cam_y, cam_h, game_y, game_h (en píxeles de la salida 1080x1920)."""
    gx, gy, gw, gh = analysis["game"]["rect"]
    ga = gw / gh
    cam = analysis.get("webcam") if with_cam else None
    if cam and cfg.get("game_view", "full") == "full" and cfg.get("full_cam_frac"):
        # Reparto fijo de la pantalla (por defecto 50/50): webcam arriba y
        # juego abajo llenando su mitad, con zoom fijo al centro. Sin franjas.
        cam_h = _even(cfg["full_cam_frac"] * OUT_H)
        return {"cam_y": 0, "cam_h": cam_h, "game_y": cam_h, "game_h": OUT_H - cam_h}
    cam_h = cam_panel_height(cam["rect"], cfg) if cam else 0
    avail = OUT_H - cam_h
    full = cfg.get("game_view", "full") == "full"
    if full:
        # Pantalla de juego ENTERA a lo ancho (aunque la cam se vea también
        # dentro): preferencia del usuario frente a recortar el centro.
        # Con un zoom suave: se pierde un poco por los lados (game_full_keep,
        # 85 % del ancho), toda la altura, y el juego gana algo de alto.
        crop_aspect = max(OUT_W / avail, ga * cfg.get("game_full_keep", 0.85))
    elif ga <= cfg.get("game_fit_max_aspect", 1.6):
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
        max_frac = cfg.get("cam_grow_max_frac_full", 0.5) if full else cfg.get("cam_grow_max_frac", 0.42)
        grow = min(spare, _even(max_frac * OUT_H) - cam_h)
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
    if plan.get("cam_in_game") and plan["cam_in_game"][2] > 4 and plan["cam_in_game"][3] > 4:
        bx, by, bw, bh = plan["cam_in_game"]
        parts.append(f"[b]crop={gw}:{gh}:{gx}:{gy},split=2[g0][g1]")
        parts.append(f"[g1]crop={bw}:{bh}:{bx}:{by},boxblur=25:3[gb]")
        parts.append(f"[g0][gb]overlay={bx}:{by},scale={OUT_W}:{plan['game_h']},setsar=1[game]")
    else:
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


def cam_timeline(clip_path, analysis, cfg, every=1.0):
    """Posición/tamaño de la cam a lo largo del clip, por su aspecto.

    Devuelve [(inicio, fin, rect)] si la cam CAMBIA durante el clip (zoom
    temporal de la escena del streamer, o la mueve), o [] si no cambia."""
    cam = analysis.get("webcam")
    if not cam or not cam.get("template") or not os.path.exists(cam["template"]):
        return []
    tpl = cv2.imread(cam["template"])
    W, H = analysis["width"], analysis["height"]
    thr = cfg.get("cam_template_min_score", 0.35)
    cap = cv2.VideoCapture(clip_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(fps * every))
    x0, y0, w0, h0 = cam["rect"]
    labels = []  # (t, rect o None)
    for idx in range(total):
        # Lectura de corrido (saltar con set() en vídeo comprimido era lo lento).
        if not cap.grab():
            break
        if idx % step:
            continue
        ok, fr = cap.retrieve()
        if not ok:
            break
        m = webcam.find_by_template(fr, tpl)
        if m and m[2] >= thr:
            x, y, s, k = m
            # Redondear: misma posición/tamaño = misma etiqueta.
            if abs(x - x0) <= 0.03 * W and abs(y - y0) <= 0.03 * H and abs(k - 1) <= 0.1:
                rect = (x0, y0, w0, h0)
            else:
                rect = (int(round(x / 8) * 8), int(round(y / 8) * 8), int(w0 * k), int(h0 * k))
                # Posición/tamaño nuevos: solo si dentro hay una cara coherente.
                if not webcam.face_fits([(0, fr)], rect, cam["face"][2] * k):
                    rect = None
        else:
            rect = None
        labels.append((idx / fps, rect))
    cap.release()
    dur = total / fps
    # Tramos de etiqueta igual; los huecos (no se encontró) heredan el anterior.
    def same(r1, r2):
        if r1 is None or r2 is None:
            return r1 is r2
        return all(abs(p - q) <= 0.03 * max(W, H) for p, q in zip(r1, r2))

    runs = []
    for t, r in labels:
        if r is None and runs:
            r = runs[-1][2]
        if runs and same(runs[-1][2], r):
            runs[-1][1] = t + every
        else:
            runs.append([t, t + every, r])
    # Tramos cortos (< min) se unen al anterior: evita parpadeos.
    min_len = cfg.get("cam_change_min_seconds", 2.0)
    merged = []
    for a, b, r in runs:
        if merged and (b - a < min_len or same(merged[-1][2], r)):
            merged[-1][1] = b
        else:
            merged.append([a, b, r])
    merged = [m for m in merged if m[2] is not None] or merged
    if len(merged) <= 1:
        return []
    merged[0][0], merged[-1][1] = 0.0, dur
    for i in range(1, len(merged)):
        merged[i][0] = merged[i - 1][1]
    return [(round(a, 2), round(b, 2), list(r)) for a, b, r in merged]


def _render_segments(clip_path, out_path, analysis, cfg, tl):
    """Renderiza cada tramo con su posición de cam y los pega (audio del clip)."""
    import shutil
    import tempfile
    tmp = tempfile.mkdtemp(prefix="cam_")
    geo = video_geometry(analysis, cfg)
    plans, parts = [], []
    try:
        for k, (a, b, rect) in enumerate(tl):
            seg = os.path.join(tmp, f"s{k}.mp4")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", clip_path, "-ss", f"{a:.3f}",
                            "-to", f"{b:.3f}", "-an", *_encode_args(), seg], check=True)
            cam = analysis["webcam"]
            kf = rect[2] / max(1, cam["rect"][2])
            fx, fy, fw, fh = cam["face"][:4]
            moved = {**cam, "rect": rect,
                     "face": [rect[0] + (fx - cam["rect"][0]) * kf, rect[1] + (fy - cam["rect"][1]) * kf,
                              fw * kf, fh * kf], "template": None}
            ana = {**analysis, "webcam": moved}
            plan = plan_clip(seg, ana, {**cfg, "cam_present_min": 0.0, "force_geometry": geo})
            part = os.path.join(tmp, f"p{k}.mp4")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", seg, "-filter_complex", _filter_graph(plan),
                            "-map", "[v]", "-an", *_encode_args(), part], check=True)
            parts.append(part)
            plans.append({"start": a, "end": b, **plan})
        lst = os.path.join(tmp, "list.txt")
        with open(lst, "w") as f:
            for p in parts:
                f.write(f"file '{p.replace(os.sep, '/')}'\n")
        joined = os.path.join(tmp, "joined.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                        "-c", "copy", joined], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", joined, "-i", clip_path, "-map", "0:v",
                        "-map", "1:a?", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k", "-shortest",
                        "-movflags", "+faststart", out_path], check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return plans


def render(clip_path, out_path, analysis, cfg):
    """Renderiza el clip con el layout gaming y escribe el sidecar de layout
    (para que los subtítulos vayan en la costura) y un JSON de debug."""
    tl = cam_timeline(clip_path, analysis, cfg) if cfg.get("cam_follow_changes", True) else []
    if tl:
        # La cam cambia de sitio/tamaño durante el clip: un tramo por posición.
        plans = _render_segments(clip_path, out_path, analysis, cfg, tl)
        import layout_ranges
        layout_ranges.write(out_path, [(p["start"], p["end"], "split") for p in plans])
        plan = {"mode": "cam_top", "cam_present": 1.0, "cam_timeline": tl, "segments": plans,
                **{k: plans[0][k] for k in ("game_rect", "game_crop", "cam_crop", "cam_y", "cam_h",
                                            "game_y", "game_h", "fill")}}
        with open(out_path + ".gaming.json", "w", encoding="utf-8") as f:
            json.dump(plan, f, indent=2)
        if cfg.get("debug_video", True):
            debug_video(clip_path, out_path + ".debug.mp4", plan, analysis)
        return plan
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

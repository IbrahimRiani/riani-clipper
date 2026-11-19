"""Layout CHATTING: decide POR ESCENA dentro de cada clip.

Un stream de just chatting / reacción alterna escenas: cámara a pantalla
completa (cara grande) y reacción (el vídeo o la web ocupa la pantalla y la
webcam es pequeña). Un mismo clip de 60 s puede tener las dos, así que:

1. Se buscan los cortes de escena del clip (cambio brusco de colores).
2. En cada escena se miran las caras en unos pocos frames:
   - cara pequeña dentro de la webcam detectada -> "cam_top" (como gaming:
     webcam arriba, contenido abajo, sin duplicar la cam)
   - cara grande                                -> "face_track" (cámara que
     sigue la cara, suave)
   - sin cara                                   -> "content" (contenido a
     lo ancho con fondo difuminado)
3. Cada escena se renderiza con su modo y se pegan; el audio es el del clip.
   Los subtítulos van en la costura solo en las escenas "cam_top".
"""
import json
import os
import shutil
import subprocess
import tempfile

import cv2
import numpy as np

from . import face_track, gaming_render, podcast_render, webcam

OUT_W, OUT_H = 1080, 1920


def scene_cuts(clip_path, fps_sample=4.0, thr=0.55, min_len=1.5, diff_thr=38.0):
    """Tiempos de corte de escena (s), por cambio de histograma de color."""
    cap = cv2.VideoCapture(clip_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    step = max(1, int(round(fps / fps_sample)))
    prev, prev_g, cuts, idx = None, None, [], 0
    while idx < total:
        ok = cap.grab()
        if not ok:
            break
        if idx % step == 0:
            ok, fr = cap.retrieve()
            if not ok:
                break
            small = cv2.resize(fr, (160, 90))
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            h = cv2.calcHist([hsv], [0, 1], None, [24, 16], [0, 180, 0, 256])
            cv2.normalize(h, h)
            g = cv2.cvtColor(cv2.resize(small, (64, 36)), cv2.COLOR_BGR2GRAY).astype(np.float32)
            # Corte = cambian mucho los colores O cambia mucho la imagen (de plano
            # general a primer plano con el mismo fondo los colores apenas cambian).
            diff = float(np.mean(np.abs(g - prev_g))) if prev_g is not None else 0.0
            if prev is not None and (cv2.compareHist(prev, h, cv2.HISTCMP_CORREL) < thr
                                     or diff > diff_thr):
                t = idx / fps
                if not cuts or t - cuts[-1] >= min_len:
                    cuts.append(t)
            prev, prev_g = h, g
        idx += 1
    cap.release()
    return cuts, total / fps


def cam_border_visible(frame, rect, thr=0.45):
    """¿Se ve el marco de la webcam en este frame? (aunque la cara no se
    detecte porque mira hacia otro lado). Fracción de píxeles de los 4 lados
    del rectángulo que son borde."""
    x, y, w, h = rect
    H, W = frame.shape[:2]
    g = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY).astype(np.float32)
    gx = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))
    gy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
    sides = []
    for xx in (x, x + w - 1):
        if 2 < xx < W - 3:
            sides.append((gx[y:y + h, xx - 2:xx + 3].max(axis=1) > 40).mean())
    for yy in (y, y + h - 1):
        if 2 < yy < H - 3:
            sides.append((gy[yy - 2:yy + 3, x:x + w].max(axis=0) > 40).mean())
    return bool(sides) and float(np.mean(sides)) >= thr


_TEMPLATES = {}


def _template(cam):
    path = (cam or {}).get("template")
    if not path or not os.path.exists(path):
        return None
    if path not in _TEMPLATES:
        _TEMPLATES[path] = cv2.imread(path)
    return _TEMPLATES[path]


def _cam_by_template(frames, cam, W, cfg):
    """Busca la cam por su aspecto en los frames de la escena. Devuelve el
    rectángulo si aparece en el mismo sitio en la mayoría de frames."""
    tpl = _template(cam)
    if tpl is None:
        return None
    thr = cfg.get("cam_template_min_score", 0.35)
    hits = []
    for _, fr in frames:
        m = webcam.find_by_template(fr, tpl)
        if m and m[2] >= thr:
            hits.append(m)
    if len(hits) < max(2, 0.5 * len(frames)):
        return None
    xs, ys = np.array([h[0] for h in hits]), np.array([h[1] for h in hits])
    mx, my = float(np.median(xs)), float(np.median(ys))
    tol = 0.03 * W
    consistent = sum(1 for x, y in zip(xs, ys) if abs(x - mx) <= tol and abs(y - my) <= tol)
    if consistent < max(2, 0.5 * len(frames)):
        return None  # sale en sitios distintos: coincidencia casual
    k = float(np.median([m[3] for m in hits]))
    h, w = tpl.shape[:2]
    return [int(mx), int(my), int(w * k), int(h * k)]


def _scale_face(cam, rect):
    """Posición de la cara dentro de una cam movida y/o agrandada."""
    k = rect[2] / max(1, cam["rect"][2])
    fx, fy, fw, fh = cam["face"][:4]
    return [rect[0] + (fx - cam["rect"][0]) * k, rect[1] + (fy - cam["rect"][1]) * k, fw * k, fh * k]


def _classify(frames, analysis, cfg):
    """(modo, análisis a usar) de una escena a partir de sus frames.

    El análisis puede ser distinto del global si el streamer ha movido la cam
    en esta escena."""
    W = analysis["width"]
    cam = analysis.get("webcam")
    # 0) Buscar la cam por su aspecto: funciona aunque no se vea la cara y
    #    aunque la haya movido de sitio.
    if cam and _template(cam) is not None:
        rect = _cam_by_template(frames, cam, W, cfg)
        if rect:
            if _iou(rect, cam["rect"]) > 0.5:
                return "cam_top", analysis
            moved = {**cam, "rect": rect, "face": _scale_face(cam, rect)}
            return "cam_top", {**analysis, "webcam": moved}
    in_cam = border = big = 0
    small_elsewhere = 0
    face_lists = []
    for _, fr in frames:
        faces = webcam.detect_faces(fr)
        face_lists.append(faces)
        if cam:
            x, y, w, h = cam["rect"]
            ref = cam["face"][2]
            # Cualquier cara (no solo la más grande) dentro de la cam y de su
            # tamaño: el vídeo reaccionado puede tener caras más grandes.
            if any(x <= f[0] + f[2] / 2 <= x + w and y <= f[1] + f[3] / 2 <= y + h
                   and f[2] <= ref * 1.6 for f in faces):
                in_cam += 1
                continue
            if cam_border_visible(fr, cam["rect"]):
                border += 1
        if faces and max(f[2] for f in faces) >= W * cfg.get("track_min_face_frac", 0.05):
            big += 1
        if any(f[2] < W * cfg.get("max_cam_face_frac", 0.12) for f in faces):
            small_elsewhere += 1
    n = max(1, len(frames))
    # 1) ¿La cam está en otro sitio en esta escena? (algunos streamers la
    #    mueven). Solo se acepta si se parece en tamaño a la cam original: las
    #    "cams" que salen de estanterías o marcos de fotos son mucho mayores.
    # Solo si el vídeo tiene webcam: sin referencia, cualquier recuadro con
    # una cara del decorado pasaba por "cam" (falso positivo medido en un podcast).
    if cam and small_elsewhere >= 2:
        local = webcam.scene_cam(frames, cfg)
        if local and _compatible(local, cam, analysis, cfg):
            if cam and _iou(local["rect"], cam["rect"]) > 0.5:
                return "cam_top", analysis
            return "cam_top", {**analysis, "webcam": local}
    # 2) La cam de siempre: cara dentro y marco visible.
    if in_cam >= 1 and (in_cam + border) / n >= 0.5 and in_cam + border >= 2:
        return "cam_top", analysis
    # 3) Dos o más personas en plano -> podcast (quién habla / split).
    if cfg.get("podcast_enabled", True) and len(podcast_render.find_speakers(frames, W, cfg)) >= 2:
        return "podcast", analysis
    if big / n >= 0.6:
        return "face_track", analysis
    return "content", analysis


def _iou(a, b):
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / max(1, aw * ah + bw * bh - inter)


def _compatible(local, cam, analysis, cfg):
    """¿Esta cam encontrada en una escena es creíble?"""
    W, H = analysis["width"], analysis["height"]
    area = local["rect"][2] * local["rect"][3] / (W * H)
    if not cam:
        return area <= cfg.get("max_cam_area", 0.35)
    ref_area = cam["rect"][2] * cam["rect"][3] / (W * H)
    size_ok = 0.5 <= area / max(1e-6, ref_area) <= 2.0
    face_ok = 0.6 <= local["face"][2] / max(1.0, cam["face"][2]) <= 1.6
    return size_ok and face_ok


def _content_render(seg, out, analysis, cfg):
    """Sin cara: el contenido (zona que cambia) a lo ancho sobre fondo difuminado."""
    gx, gy, gw, gh = analysis["game"]["rect"]
    h = int(OUT_W * gh / gw) // 2 * 2
    y = (OUT_H - h) // 2 // 2 * 2
    vf = (f"[0:v]split=2[a][b];[a]crop={gw}:{gh}:{gx}:{gy},scale=270:480:force_original_aspect_ratio=increase,"
          f"crop=270:480,boxblur=12:2,eq=brightness=-0.12,scale={OUT_W}:{OUT_H},setsar=1[bg];"
          f"[b]crop={gw}:{gh}:{gx}:{gy},scale={OUT_W}:{h},setsar=1[fg];"
          f"[bg][fg]overlay=0:{y},format=yuv420p[v]")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", seg, "-filter_complex", vf,
                    "-map", "[v]", "-an", *_venc(), out], check=True)
    return {"mode": "content"}


def _venc():
    try:
        import ffmpeg_utils
        return ffmpeg_utils.video_encode_args(ffmpeg_utils.QUALITY)
    except Exception:
        return ["-c:v", "libx264", "-preset", "medium", "-crf", "18"]


def render(clip_path, out_path, analysis, cfg):
    cuts, dur = scene_cuts(clip_path, thr=cfg.get("scene_cut_threshold", 0.55),
                           diff_thr=cfg.get("scene_diff_threshold", 38.0))
    bounds = [0.0] + cuts + [dur]
    segs = [(bounds[i], bounds[i + 1]) for i in range(len(bounds) - 1) if bounds[i + 1] - bounds[i] > 0.05]
    tmp = tempfile.mkdtemp(prefix="chat_")
    plans = []
    try:
        # 1) clasificar cada escena
        modes = []
        for a, b in segs:
            n_fr = 8 if b - a > 4 else 5
            fr = webcam.sample_frames(clip_path, n_fr, a + 0.1, max(a + 0.2, b - 0.1), skip_edges=0.0)
            modes.append(_classify(fr, analysis, cfg))
        # Solo las escenas MUY cortas (< min_scene_seconds, por defecto 0,8 s)
        # heredan el modo anterior. Con 2 s, un primer plano de 1,7 s heredaba
        # el encuadre del plano general y salía un brazo gigante.
        for i in range(1, len(segs)):
            if segs[i][1] - segs[i][0] < cfg.get("min_scene_seconds", 0.8):
                modes[i] = modes[i - 1]
        # Unir escenas consecutivas con el mismo modo.
        merged = []
        for (a, b), (m, ana) in zip(segs, modes):
            same_cam = merged and (merged[-1][3].get("webcam") or {}).get("rect") ==                 (ana.get("webcam") or {}).get("rect")
            if merged and merged[-1][2] == m and same_cam:
                merged[-1] = (merged[-1][0], b, m, merged[-1][3])
            else:
                merged.append((a, b, m, ana))

        # 2) renderizar cada tramo (sin audio)
        parts = []
        for k, (a, b, m, ana) in enumerate(merged):
            seg = os.path.join(tmp, f"seg{k}.mp4")
            subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", clip_path, "-ss", f"{a:.3f}",
                            "-to", f"{b:.3f}", "-an", *_venc(), seg], check=True)
            part = os.path.join(tmp, f"part{k}.mp4")
            if m == "cam_top":
                # Ya sabemos que la cam está (cara o marco): no volver a exigir cara.
                # Misma geometría que el resto del vídeo aunque la cam se haya
                # movido: así la línea de los subtítulos no salta.
                geo = gaming_render.video_geometry(analysis, cfg) if analysis.get("webcam") else None
                p = gaming_render.render(seg, part, ana,
                                         {**cfg, "debug_video": False, "cam_present_min": 0.0,
                                          "force_geometry": geo})
                if p["mode"] != "cam_top":  # la cam no se vio al final: tratar como contenido
                    m = "content"
            elif m == "face_track":
                p = face_track.render(seg, part, {**cfg, "debug_video": False})
            elif m == "podcast":
                p = podcast_render.render(seg, part, cfg)
            if m == "content":
                p = _content_render(seg, part, ana, cfg)
            for side in (part + ".layout.json", part + ".gaming.json"):
                if os.path.exists(side):
                    os.remove(side)
            plans.append({"start": round(a, 2), "end": round(b, 2), "mode": m,
                          "sub": [(round(a + x, 2), round(a + y, 2), z) for x, y, z in p.get("ranges", [])],
                          "timeline": p.get("timeline"), "talk_share": p.get("talk_share"),
                          "people": p.get("people"),
                          "cam_rect": (ana.get("webcam") or {}).get("rect"),
                          **{k2: v for k2, v in p.items() if k2 in ("cam_crop", "game_crop", "crop_w")}})
            parts.append(part)

        # 3) pegar y poner el audio del clip
        lst = os.path.join(tmp, "list.txt")
        with open(lst, "w") as f:
            for p in parts:
                f.write(f"file '{p.replace(os.sep, '/')}'\n")
        joined = os.path.join(tmp, "joined.mp4")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "concat", "-safe", "0", "-i", lst,
                        "-an", "-c", "copy", joined], check=True)
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", joined, "-i", clip_path,
                        "-map", "0:v", "-map", "1:a?", "-c:v", "copy", "-c:a", "aac", "-b:a", "192k",
                        "-shortest", "-movflags", "+faststart", out_path], check=True)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    import layout_ranges
    ranges = []
    for p in plans:
        if p.get("sub"):
            ranges += p["sub"]  # podcast: split / track por tramos dentro de la escena
        else:
            ranges.append((p["start"], p["end"], "split" if p["mode"] == "cam_top" else "track"))
    layout_ranges.write(out_path, ranges)
    plan = {"mode": "chatting", "scenes": plans}
    with open(out_path + ".gaming.json", "w", encoding="utf-8") as f:
        json.dump(plan, f, indent=2)
    if cfg.get("debug_video", True):
        try:
            debug_video(clip_path, out_path + ".debug.mp4", plan, analysis)
        except subprocess.CalledProcessError as e:
            print(f"   ⚠️ Vídeo de debug omitido ({e}).")
    return plan


def debug_video(clip_path, out_path, plan, analysis):
    """Clip original a 960 px con el modo de cada escena escrito y la cam (verde)."""
    s = 960 / analysis["width"]
    fonts = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                         "vendor", "openshorts", "fonts", "Anton-Regular.ttf")
    font = fonts.replace("\\", "/").replace(":", "\\:")
    vf = ["scale=960:-2"]
    if analysis.get("webcam"):
        x, y, w, h = [int(v * s) for v in analysis["webcam"]["rect"]]
        vf.append(f"drawbox=x={x}:y={y}:w={w}:h={h}:color=green:t=3")
    colors = ["red", "cyan", "magenta", "orange"]
    for sc in plan["scenes"]:
        a0, b0 = sc["start"], sc["end"]
        if sc["mode"] == "podcast" and sc.get("timeline") and sc.get("people"):
            # Cajas de cada persona (gruesa la que habla) y quién habla en cada tramo.
            for k, (x, y, w, h) in enumerate(sc["people"]):
                vf.append(f"drawbox=x={int(x * s)}:y={int(y * s)}:w={int(w * s)}:h={int(h * s)}:"
                          f"color={colors[k % 4]}:t=2:enable='between(t,{a0},{b0})'")
            for ta, tb, st in sc["timeline"]:
                label = "podcast split" if st == "split" else f"podcast habla persona {int(st) + 1}"
                col = "white" if st == "split" else colors[int(st) % 4]
                vf.append(f"drawtext=fontfile='{font}':text='{label}':x=20:y=20:fontsize=40:"
                          f"fontcolor={col}:box=1:boxcolor=black@0.6:"
                          f"enable='between(t,{a0 + ta},{a0 + tb})'")
                if st != "split":
                    x, y, w, h = sc["people"][int(st)]
                    vf.append(f"drawbox=x={int(x * s)}:y={int(y * s)}:w={int(w * s)}:h={int(h * s)}:"
                              f"color={col}:t=8:enable='between(t,{a0 + ta},{a0 + tb})'")
            continue
        vf.append(f"drawtext=fontfile='{font}':text='{sc['mode']}':x=20:y=20:fontsize=40:"
                  f"fontcolor=yellow:box=1:boxcolor=black@0.6:"
                  f"enable='between(t,{a0},{b0})'")
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", clip_path, "-vf", ",".join(vf), "-an",
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "28", out_path], check=True)

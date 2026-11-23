"""Detección de la webcam y de la zona de juego en streams.

Idea en sencillo:
1. Muestreamos muchos frames repartidos por TODO el video.
2. Buscamos caras en cada uno. La cara del streamer aparece casi siempre en el
   MISMO sitio; las "caras" del juego (NPCs, sprites) cambian de sitio. Votamos
   la posición más repetida.
3. El borde de la webcam es una línea que está quieta en todos los frames
   mientras el juego cambia: buscamos esas líneas "persistentes" alrededor de la
   cara para recortar el rectángulo exacto de la cam.
4. La zona de juego es lo que más cambia entre frames (el overlay decorativo
   está quieto), quitando la webcam.

Todas las cajas están en píxeles del video original: (x, y, w, h).
"""
import cv2
import numpy as np

import threading

# MediaPipe no admite usar el mismo detector desde dos hilos a la vez (los
# clips se renderizan en paralelo): un detector por hilo.
_LOCAL = threading.local()


def _face_detector():
    fd = getattr(_LOCAL, "fd", None)
    if fd is None:
        import mediapipe as mp
        fd = mp.solutions.face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.4)
        _LOCAL.fd = fd
    return fd


def detect_faces(frame):
    """Caras en un frame. Corre el detector de corto alcance sobre 5 trozos
    ampliados (4 cuadrantes + centro): una cara de webcam de 60 px es
    invisible para el detector a tamaño completo pero clara ampliada x2.
    Devuelve [(x, y, w, h, score)] en píxeles, sin duplicados."""
    H, W = frame.shape[:2]
    fd = _face_detector()
    # 9 trozos de medio frame que se solapan a cuartos: una cara cae entera en
    # al menos uno (con 5 trozos, una cam pegada a la línea del medio se partía
    # en dos y no se detectaba en ninguno).
    tiles = [(x, y) for y in (0, H // 4, H // 2) for x in (0, W // 4, W // 2)]
    faces = []
    for x0, y0 in tiles:
        tile = frame[y0:y0 + H // 2, x0:x0 + W // 2]
        big = cv2.resize(tile, (W, H))
        res = fd.process(cv2.cvtColor(big, cv2.COLOR_BGR2RGB))
        for d in res.detections or []:
            b = d.location_data.relative_bounding_box
            faces.append((x0 + b.xmin * W / 2, y0 + b.ymin * H / 2,
                          b.width * W / 2, b.height * H / 2, float(d.score[0])))
    # La misma cara sale en varios trozos solapados: quedarse con la mejor.
    faces.sort(key=lambda f: -f[4])
    kept = []
    for f in faces:
        cx, cy = f[0] + f[2] / 2, f[1] + f[3] / 2
        if all(abs(cx - (k[0] + k[2] / 2)) > k[2] * 0.5 or
               abs(cy - (k[1] + k[3] / 2)) > k[3] * 0.5 for k in kept):
            kept.append(f)
    return kept


def sample_frames(video_path, n, t_start=None, t_end=None, skip_edges=0.03):
    """``n`` frames repartidos entre t_start y t_end (segundos)."""
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    a = int((t_start or 0) * fps) if t_start is not None else int(total * skip_edges)
    b = int(t_end * fps) if t_end is not None else int(total * (1 - skip_edges))
    b = max(a + 1, min(b, total - 1))
    frames = []
    for i in range(n):
        idx = int(a + (b - a) * (i + 0.5) / n)
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ok, fr = cap.read()
        if ok:
            frames.append((idx / fps, fr))
    cap.release()
    return frames


def _near(c, f):
    cx, cy, cw = c[0] + c[2] / 2, c[1] + c[3] / 2, c[2]
    return abs(cx - (f[0] + f[2] / 2)) < cw * 0.8 and abs(cy - (f[1] + f[3] / 2)) < cw * 0.8 \
        and 0.6 < f[2] / cw < 1.6


def _vote_clusters(face_lists, W, max_face_frac, top=8):
    """Posiciones de cara que más se repiten, de más a menos votada.

    Devuelve [(caras_del_cluster, ratio, índices_de_frames)]. Se devuelven
    varias porque en un stream mixto (charla a pantalla completa + reacción con
    cam pequeña) la cara grande puede salir más veces que la cam."""
    cands = [f for fl in face_lists for f in fl if f[2] < W * max_face_frac]
    scored = []
    for c in cands:
        idx = [i for i, fl in enumerate(face_lists) if any(_near(c, f) for f in fl)]
        scored.append((len(idx), c, idx))
    scored.sort(key=lambda s: -s[0])
    clusters = []
    for hits, c, idx in scored:
        if any(_near(k[0], c) for k in clusters):
            continue  # misma posición que un cluster ya elegido
        faces = [max((f for f in face_lists[i] if _near(c, f)), key=lambda f: f[4]) for i in idx]
        clusters.append((c, faces, hits / max(1, len(face_lists)), idx))
        if len(clusters) >= top:
            break
    return [(faces, ratio, idx) for _, faces, ratio, idx in clusters]


def edge_persistence(frames):
    """Fracción de frames en la que cada píxel es borde vertical / horizontal."""
    acc_v = acc_h = None
    for _, fr in frames:
        g = cv2.GaussianBlur(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), (3, 3), 0).astype(np.float32)
        gx = np.abs(cv2.Sobel(g, cv2.CV_32F, 1, 0, ksize=3))
        gy = np.abs(cv2.Sobel(g, cv2.CV_32F, 0, 1, ksize=3))
        v = (gx > 40).astype(np.float32)
        h = (gy > 40).astype(np.float32)
        acc_v = v if acc_v is None else acc_v + v
        acc_h = h if acc_h is None else acc_h + h
    n = max(1, len(frames))
    return acc_v / n, acc_h / n


def temporal_std(frames, scale=0.25):
    """Cuánto cambia cada píxel a lo largo de los frames (0-255)."""
    small = [cv2.resize(cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY), None, fx=scale, fy=scale)
             .astype(np.float32) for _, fr in frames]
    std = np.std(np.stack(small), axis=0)
    H, W = frames[0][1].shape[:2]
    return cv2.resize(std, (W, H), interpolation=cv2.INTER_LINEAR)


def _best_line(profile, lo, hi, min_score):
    """Índice en [lo, hi) con mayor persistencia, si supera min_score."""
    lo, hi = max(0, int(lo)), min(len(profile), int(hi))
    if hi - lo < 2:
        return None
    seg = profile[lo:hi]
    i = int(np.argmax(seg))
    return lo + i if seg[i] >= min_score else None


def _contrast_profiles(std, ys, xs, k=12):
    """Para cada columna/fila: cuánto más cambia un lado que el otro.

    El borde real de una webcam separa la habitación (casi quieta) del juego
    (cambia mucho). Devuelve 4 perfiles en [0, 1]: "fuera a la izquierda",
    "fuera a la derecha", "fuera arriba" y "fuera abajo"."""
    col = std[ys, :].mean(axis=0)
    row = std[:, xs].mean(axis=1)

    def side_means(p):
        c = np.concatenate([[0.0], np.cumsum(p)])
        n = len(p)
        idx = np.arange(n)
        lo = np.clip(idx - k, 0, n)
        hi = np.clip(idx + k, 0, n)
        before = (c[idx] - c[lo]) / np.maximum(1, idx - lo)
        after = (c[hi] - c[idx]) / np.maximum(1, hi - idx)
        return before, after

    def norm(a, b):
        return np.clip((a - b) / (a + b + 8.0), 0, 1)

    cb, ca = side_means(col)
    rb, ra = side_means(row)
    # borde izquierdo de la cam: fuera (antes) cambia más que dentro (después)
    return norm(cb, ca), norm(ca, cb), norm(rb, ra), norm(ra, rb)


def find_cam_rect(face, ev, eh, W, H, cfg, std=None):
    """Rectángulo de la webcam alrededor de la cara ``face`` (x, y, w, h).

    Cada lado se elige donde hay una línea "fija" (persistente en todos los
    frames) y/o un salto entre zona quieta (dentro) y zona que cambia (fuera)."""
    fx, fy, fw, fh = face
    min_score = cfg.get("edge_min_score", 0.5)
    w_contrast = cfg.get("edge_contrast_weight", 1.0)
    near_edge = cfg.get("edge_snap_faces", 2.5)  # a menos de N caras del borde -> el borde
    # Primera estimación del rango de filas/columnas que cubre la cam.
    y_lo, y_hi = fy - 0.5 * fh, fy + 1.5 * fh
    x_lo, x_hi = fx - 0.5 * fw, fx + 1.5 * fw
    rect = None
    for _ in range(2):  # refinar con el rectángulo de la vuelta anterior
        ys = slice(int(max(0, y_lo)), int(min(H, y_hi)))
        xs = slice(int(max(0, x_lo)), int(min(W, x_hi)))
        col = ev[ys, :].mean(axis=0)
        row = eh[:, xs].mean(axis=1)
        # Continuidad: fracción del lateral donde la línea está casi siempre.
        # Un marco dibujado da ~1.0; un borde de la habitación, a trozos.
        k = np.ones((1, 3), np.uint8)
        cont_col = cv2.dilate((ev[ys, :] > 0.7).astype(np.uint8), k).mean(axis=0)
        cont_row = cv2.dilate((eh[:, xs] > 0.7).astype(np.uint8), k.T).mean(axis=1)
        if std is not None:
            c_left, c_right, c_top, c_bottom = _contrast_profiles(std, ys, xs)
        else:
            c_left = c_right = np.zeros(W)
            c_top = c_bottom = np.zeros(H)
        # Medido a mano en 3 streams (12 lados): un borde real tiene línea
        # continua (marco dibujado) y/o salto quieto->movimiento (sin marco).
        # Las líneas de la habitación son continuas pero sin salto, por eso
        # se suman las dos pistas.
        def score(cont, contrast):
            return cont + w_contrast * contrast

        left = _best_line(score(cont_col, c_left), fx - 4.5 * fw, fx - 0.1 * fw, min_score)
        right = _best_line(score(cont_col, c_right), fx + 1.1 * fw, fx + fw + 4.5 * fw, min_score)
        top = _best_line(score(cont_row, c_top), fy - 3.0 * fh, fy - 0.05 * fh, min_score)
        bottom = _best_line(score(cont_row, c_bottom), fy + 1.05 * fh, fy + fh + 4.0 * fh, min_score)
        if left is None and fx < near_edge * fw:
            left = 0
        if top is None and fy < near_edge * fh:
            top = 0
        if right is None and W - (fx + fw) < near_edge * fw:
            right = W
        if bottom is None and H - (fy + fh) < near_edge * fh:
            bottom = H
        real_sides = sum(v is not None for v in (left, right, top, bottom))
        # Lados sin línea clara: tamaño típico de cam (cara ~ 1/4.5 del ancho, 16:9).
        cam_w = fw * cfg.get("fallback_cam_width_faces", 4.5)
        cam_h = cam_w * 9 / 16
        cx, cy = fx + fw / 2, fy + fh / 2
        if left is None and right is not None:
            left = max(0, right - cam_w)
        if right is None and left is not None:
            right = min(W, left + cam_w)
        if left is None and right is None:
            left, right = max(0, cx - cam_w / 2), min(W, cx + cam_w / 2)
        if top is None and bottom is not None:
            top = max(0, bottom - cam_h)
        if bottom is None and top is not None:
            bottom = min(H, top + cam_h)
        if top is None and bottom is None:
            top, bottom = max(0, cy - cam_h * 0.45), min(H, cy + cam_h * 0.55)
        rect = (int(left), int(top), int(right - left), int(bottom - top))
        y_lo, y_hi = top, bottom
        x_lo, x_hi = left, right
    if cfg.get("_return_sides"):
        return rect, real_sides
    return rect


def find_game_rect(frames, cam_rect, W, H, ev, eh, cfg):
    """Zona de juego: la parte del frame que más cambia, sin la webcam.

    Si el juego ocupa casi todo el frame (la cam va encima), devuelve el frame
    completo: el recorte posterior se encarga de esquivar la cam."""
    std = temporal_std(frames)
    thr = cfg.get("game_motion_std", 12.0)
    active = (std > thr).astype(np.uint8)
    if cam_rect:
        x, y, w, h = cam_rect
        m = int(0.05 * max(w, h))
        active[max(0, y - m):y + h + m, max(0, x - m):x + w + m] = 0
    col_frac = active.mean(axis=0)
    row_frac = active.mean(axis=1)
    min_frac = cfg.get("game_active_frac", 0.25)
    cols = np.where(col_frac > min_frac)[0]
    rows = np.where(row_frac > min_frac)[0]
    if len(cols) < W * 0.2 or len(rows) < H * 0.2:
        return (0, 0, W, H), std
    x0, x1, y0, y1 = cols[0], cols[-1] + 1, rows[0], rows[-1] + 1
    # Ajustar a líneas persistentes cercanas (el marco de la ventana del juego).
    snap = int(0.04 * W)
    min_score = cfg.get("edge_min_persistence", 0.45)
    ys, xs = slice(y0, y1), slice(x0, x1)
    colp, rowp = ev[ys, :].mean(axis=0), eh[:, xs].mean(axis=1)

    def snap_to(prof, v, lo, hi):
        b = _best_line(prof, max(lo, v - snap), min(hi, v + snap), min_score)
        return v if b is None else b

    x0 = snap_to(colp, x0, 0, W)
    x1 = snap_to(colp, x1, 0, W)
    y0 = snap_to(rowp, y0, 0, H)
    y1 = snap_to(rowp, y1, 0, H)
    # Casi todo el frame -> frame completo (bordes de 1-2% no cuentan).
    if x0 < W * 0.03 and y0 < H * 0.03 and x1 > W * 0.97 and y1 > H * 0.97:
        return (0, 0, W, H), std
    return (int(x0), int(y0), int(x1 - x0), int(y1 - y0)), std



def pick_cam(frames, face_lists, clusters, W, H, cfg, min_sides=None):
    """Elige la webcam entre las posiciones de cara más votadas.

    Prueba cada candidata y se queda con la primera que tenga un marco de
    webcam real. Los bordes se miden SOLO en los frames donde está esa cara:
    en un stream mixto la cam pequeña no sale en todos.
    Devuelve (webcam o None, lista de candidatas para el debug)."""
    cands = []
    for faces, ratio, idx in clusters:
        if len(idx) < cfg.get("min_cam_hits", 5):
            break
        # Una cara pintada (póster, mural, logo) está siempre en el mismo
        # píxel; la de una persona se mueve algo a lo largo del vídeo.
        cxs = np.array([f[0] + f[2] / 2 for f in faces])
        cys = np.array([f[1] + f[3] / 2 for f in faces])
        fw_med = float(np.median([f[2] for f in faces]))
        moving = max(np.std(cxs), np.std(cys)) / max(1.0, fw_med)
        if moving < cfg.get("min_face_motion", 0.02):
            cands.append({"face": [round(float(v), 1) for v in faces[0][:4]],
                          "hit_ratio": round(ratio, 3), "rejected": "cara quieta (póster/dibujo)"})
            continue
        sub = [frames[i] for i in idx]
        ev_s, eh_s = edge_persistence(sub)
        face = tuple(float(np.median([f[i] for f in faces])) for i in range(4))
        rect, sides = find_cam_rect(face, ev_s, eh_s, W, H, {**cfg, "_return_sides": True},
                                    std=temporal_std(sub))
        area = rect[2] * rect[3] / (W * H)
        cand = {"rect": list(rect), "face": [round(v, 1) for v in face], "hit_ratio": round(ratio, 3),
                "real_sides": sides, "area_frac": round(area, 3)}
        cands.append(cand)
        # Una sola posición muy dominante (gaming típico) vale con 2 lados;
        # si compite con otras, se exige un marco claro (3-4 lados).
        need = min_sides or (2 if ratio >= cfg.get("min_cam_hit_ratio", 0.35) else 3)
        # Una webcam superpuesta va pegada a algún borde de la pantalla (esquina
        # o lateral). En un plató con cámara fija, la mesa y el decorado forman
        # un "marco" de 4 lados en mitad de la imagen: eso no es una webcam.
        e = cfg.get("cam_max_edge_dist", 0.08)
        x, y, w, h = rect
        anchored = x <= e * W or x + w >= (1 - e) * W or y <= e * H or y + h >= (1 - e) * H
        if not anchored:
            cand["rejected"] = "no está pegada a ningún borde"
            continue
        # Las webcams son apaisadas (medido: 1,5-1,8 en 4 streams). Una "cam"
        # más alta que ancha era el público en primer plano de un plató.
        if w / max(1, h) < cfg.get("cam_min_aspect", 0.9):
            cand["rejected"] = "más alta que ancha"
            continue
        # En una webcam se ve cabeza y hombros: la cara ocupa el 26-37 % de su
        # ancho (medido en 4 streams). Una persona pequeña en un plano general
        # de plató ocupaba el 16 % de su "marco".
        # Con YOLO la "cara" es una estimación de la cabeza (más pequeña que la
        # caja de MediaPipe): umbral más bajo. Medido: 0,195 y 0,206 en dos
        # directos de ElMillor con la cam bien detectada.
        if cfg.get("_via_person"):
            min_ratio = cfg.get("min_face_to_cam_width_person", 0.12)
        else:
            min_ratio = cfg.get("min_face_to_cam_width", 0.2)
        if face[2] / max(1, w) < min_ratio:
            cand["rejected"] = "cara demasiado pequeña para su marco"
            continue
        fcx, fcy = face[0] + face[2] / 2, face[1] + face[3] / 2
        if not (x <= fcx <= x + w and y <= fcy <= y + h):
            cand["rejected"] = "la cara queda fuera de su marco"
            continue
        if sides >= need and area <= cfg.get("max_cam_area", 0.35)                 and face[2] / rect[2] <= cfg.get("max_face_to_cam_width", 0.6):
            return ({"rect": list(rect), "face": cand["face"], "area_frac": cand["area_frac"],
                     "hit_ratio": cand["hit_ratio"], "frame_index": int(idx[len(idx) // 2])}, cands)
    return None, cands


def person_heads(frame, cfg):
    """Cabezas (x, y, w, h, score) de las personas que ve YOLO, incluidas las
    cortadas por el borde (una webcam pegada abajo corta el cuerpo)."""
    from .podcast_render import _yolo
    H, W = frame.shape[:2]
    r = _yolo()(frame, classes=[0], conf=cfg.get("person_min_conf", 0.35), verbose=False)[0]
    heads = []
    for (x1, y1, x2, y2), conf in zip(r.boxes.xyxy.tolist(), r.boxes.conf.tolist()):
        w, h = x2 - x1, y2 - y1
        hh = min(0.45 * h, 0.75 * w)  # cabeza: parte de arriba de la caja
        heads.append((x1 + w / 2 - hh * 0.4, y1, hh * 0.8, hh, float(conf)))
    return heads


def scene_cam(frames, cfg):
    """Busca la webcam usando solo los frames de UNA escena (por si el
    streamer la ha movido). Más estricto: marco claro de 3+ lados."""
    if not frames:
        return None
    H, W = frames[0][1].shape[:2]
    face_lists = [detect_faces(fr) for _, fr in frames]
    clusters = _vote_clusters(face_lists, W, cfg.get("max_cam_face_frac", 0.12))
    # Escena corta: la cara se mueve poco, pero un póster no se mueve NADA.
    local = {**cfg, "min_cam_hits": max(2, len(frames) // 3), "min_face_motion": 0.005}
    cam, _ = pick_cam(frames, face_lists, clusters, W, H, local, min_sides=3)
    return cam

def analyze(video_path, cfg=None, n_frames=None):
    """Análisis global del video: webcam, zona de juego y señales para el
    auto-layout. Devuelve un dict serializable a JSON (para debug)."""
    cfg = cfg or {}
    n = n_frames or cfg.get("sample_frames", 60)
    frames = sample_frames(video_path, n)
    if not frames:
        raise RuntimeError(f"No se pudieron leer frames de {video_path}")
    H, W = frames[0][1].shape[:2]
    face_lists = [detect_faces(fr) for _, fr in frames]
    clusters = _vote_clusters(face_lists, W, cfg.get("max_cam_face_frac", 0.12))
    ev, eh = edge_persistence(frames)

    result = {"width": W, "height": H, "frames_sampled": len(frames),
              "face_hit_ratio": round(clusters[0][1], 3) if clusters else 0.0, "webcam": None,
              "faces_per_frame": [len(fl) for fl in face_lists], "cam_candidates": []}
    webcam_info, cands = pick_cam(frames, face_lists, clusters, W, H, cfg)
    if webcam_info is None and cfg.get("cam_person_fallback", True):
        # Plan B: cabezas de personas (YOLO) en vez de caras. Con luz lateral o
        # la cara en diagonal, MediaPipe solo veía a ElMillor en 1 de cada 5
        # frames; YOLO lo encontraba en 54 de 60.
        head_lists = [person_heads(fr, cfg) for _, fr in frames]
        hclusters = _vote_clusters(head_lists, W, cfg.get("max_cam_face_frac", 0.12))
        # Más exigente que con caras: la persona debe verse en >=35% del vídeo
        # (ElMillor: 73%; una "cam" falsa de público en un plató: 18%).
        # Más exigente que con caras. Medido: cams reales de ElMillor en el
        # 73-75 % del vídeo y 4,5 % de la pantalla; la "cam" falsa de un plató,
        # 37 % y 22 %.
        min_hits = max(cfg.get("min_cam_hits", 5),
                       int(cfg.get("person_min_hit_ratio", 0.5) * len(frames)))
        webcam_info, hcands = pick_cam(frames, head_lists, hclusters, W, H,
                                       {**cfg, "min_cam_hits": min_hits, "_via_person": True,
                                        "max_cam_area": cfg.get("person_max_cam_area", 0.12)})
        for c in hcands:
            c["via"] = "personas (YOLO)"
        cands += hcands
        if webcam_info:
            webcam_info["via"] = "personas (YOLO)"
    result["cam_candidates"] = cands
    result["webcam"] = webcam_info
    cam_rect = tuple(webcam_info["rect"]) if webcam_info else None
    game_rect, std = find_game_rect(frames, cam_rect, W, H, ev, eh, cfg)
    result["game"] = {"rect": list(game_rect)}
    # Dinamismo de la escena fuera de la cam (para distinguir gaming de chatting).
    mask = np.ones((H, W), bool)
    if cam_rect:
        x, y, w, h = cam_rect
        mask[y:y + h, x:x + w] = False
    result["scene_motion"] = round(float(std[mask].mean()), 2)
    big_faces = [max((f[2] for f in fl), default=0) / W for fl in face_lists]
    result["median_largest_face_frac"] = round(float(np.median(big_faces)), 3)
    # Parte del vídeo con una cara GRANDE (escenas de cámara a pantalla completa).
    # Solo cuenta si las caras grandes salen casi siempre en el MISMO sitio
    # (el streamer en su escena de cámara completa); los personajes de las
    # cinemáticas de un juego salen cada vez en un sitio distinto.
    big_lists = [[f for f in fl if f[2] >= cfg.get("big_face_frac", 0.1) * W] for fl in face_lists]
    bc = _vote_clusters(big_lists, W, 1.0, top=1)
    result["big_face_ratio"] = round(bc[0][1], 3) if bc else 0.0
    return result, frames


def cam_present(video_path, cam_rect, t_start=None, t_end=None, n=8):
    """Fracción de frames del tramo con una cara dentro de la cam."""
    x, y, w, h = cam_rect
    frames = sample_frames(video_path, n, t_start, t_end, skip_edges=0.0)
    hits = 0
    for _, fr in frames:
        for f in detect_faces(fr):
            cx, cy = f[0] + f[2] / 2, f[1] + f[3] / 2
            if x <= cx <= x + w and y <= cy <= y + h:
                hits += 1
                break
    return hits / max(1, len(frames))


def draw_debug(frame, analysis, extra_boxes=None):
    """Frame con las cajas dibujadas: cam (verde), juego (azul), extras (rojo)."""
    img = frame.copy()
    if analysis.get("webcam"):
        x, y, w, h = analysis["webcam"]["rect"]
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 3)
        fx, fy, fw, fh = [int(v) for v in analysis["webcam"]["face"]]
        cv2.rectangle(img, (fx, fy), (fx + fw, fy + fh), (0, 255, 255), 2)
    x, y, w, h = analysis["game"]["rect"]
    cv2.rectangle(img, (x + 2, y + 2), (x + w - 2, y + h - 2), (255, 128, 0), 3)
    for label, (x, y, w, h) in (extra_boxes or {}).items():
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(img, label, (x + 4, y + 22), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    return img


# --- Buscar la webcam por su ASPECTO (aunque no se vea la cara) -------------

TEMPLATE_SCALE = 0.25


def save_template(frame, rect, path):
    """Guarda la imagen de la webcam (marco + habitación) para buscarla luego."""
    x, y, w, h = rect
    cv2.imwrite(path, frame[y:y + h, x:x + w])


def find_by_template(frame, template, scale=TEMPLATE_SCALE,
                     sizes=(1.0, 0.8, 1.25, 1.5, 1.75, 2.0, 2.5, 3.0)):
    """(x, y, score, tamaño) de donde mejor encaja la imagen de la cam.

    Se prueba a varios tamaños: ElMillor usaba una escena "ampliada" con la cam
    1,5 veces más grande y a un solo tamaño la búsqueda fallaba (0,44, y en
    otro sitio). ``tamaño`` es el factor respecto a la cam original."""
    small = cv2.resize(frame, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
    best = None
    for k in sizes:
        tpl = cv2.resize(template, None, fx=scale * k, fy=scale * k, interpolation=cv2.INTER_AREA)
        if tpl.shape[0] < 8 or tpl.shape[0] >= small.shape[0] or tpl.shape[1] >= small.shape[1]:
            continue
        res = cv2.matchTemplate(small, tpl, cv2.TM_CCOEFF_NORMED)
        _, score, _, loc = cv2.minMaxLoc(res)
        # Pequeña preferencia por el tamaño original ante empates.
        adj = score - (0.0 if k == 1.0 else 0.02)
        if best is None or adj > best[4]:
            best = (int(loc[0] / scale), int(loc[1] / scale), float(score), k, adj)
    return best[:4] if best else None

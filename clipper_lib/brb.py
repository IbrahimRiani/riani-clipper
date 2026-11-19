"""Detecta tramos "muertos": pantallas de BRB / espera / intro / "empezamos pronto".

Un tramo es muerto si a la vez:
1. Casi no hay voz: pocas palabras transcritas, o solo frases que whisper se
   inventa sobre música ("Subtítulos por la comunidad de Amara.org"...).
2. La imagen está casi quieta, o es una animación en bucle (se parece mucho a
   lo que se veía unos segundos antes).
3. No hay ninguna cara (descarta a un streamer que juega en silencio).

Uso: se marcan esas zonas en el transcript para DeepSeek y, después, se
descartan los clips que caigan sobre todo dentro de ellas.
"""
import re
import subprocess

import numpy as np

# Frases que whisper "alucina" sobre música o silencio (es/en).
HALLUCINATIONS = re.compile(
    r"amara\.org|subt[ií]tulos (realizados|por)|gracias por ver|thanks for watching|"
    r"suscr[ií]bete|subscribe|♪|\[m[uú]sica\]|\[music\]|música de fondo",
    re.IGNORECASE)


def frame_signals(video_path, step=2.0, size=(64, 36)):
    """Imágenes diminutas en gris y el segundo de cada una.

    Lee solo los fotogramas clave (``-skip_frame nokey``): en Twitch hay uno
    cada ~2 s y así 90 min se leen en ~1 min en vez de 5,5 (decodificar todos
    los fotogramas a 1080p60 era lo lento). ``step`` se ignora si hay keyframes."""
    w, h = size
    cmd = ["ffmpeg", "-v", "error", "-skip_frame", "nokey", "-i", video_path,
           "-vf", f"scale={w}:{h},format=gray", "-vsync", "vfr", "-f", "rawvideo", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    n = len(raw) // (w * h)
    frames = np.frombuffer(raw[:n * w * h], dtype=np.uint8).reshape(n, h, w).astype(np.float32)
    dur = _duration(video_path)
    if dur and n > dur:
        # El códec ignoró -skip_frame (p. ej. AV1/VP9 de YouTube) y salieron
        # todos los fotogramas: quedarse con uno cada ~step segundos.
        every = max(1, int(round(step * n / dur)))
        frames = frames[::every]
        return frames, dur / max(1, len(frames))
    if n < max(3, dur / 30):  # vídeo con muy pocos keyframes: muestreo fijo
        cmd = ["ffmpeg", "-v", "error", "-i", video_path, "-vf",
               f"fps=1/{step},scale={w}:{h},format=gray", "-f", "rawvideo", "-"]
        raw = subprocess.run(cmd, capture_output=True, check=True).stdout
        n = len(raw) // (w * h)
        frames = np.frombuffer(raw[:n * w * h], dtype=np.uint8).reshape(n, h, w).astype(np.float32)
    return frames, dur / max(1, n)


def _duration(video_path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "csv=p=0", video_path], capture_output=True, text=True).stdout
    try:
        return float(out.strip())
    except ValueError:
        return 0.0


def words_per_window(transcript, n, step):
    """Palabras "reales" (no alucinaciones) por ventana de ``step`` s."""
    counts = np.zeros(n)
    for seg in transcript.get("segments", []):
        if HALLUCINATIONS.search(seg.get("text") or ""):
            continue
        for w in seg.get("words", []):
            i = int(float(w["start"]) // step)
            if 0 <= i < n:
                counts[i] += 1
    return counts


def find_dead_zones(video_path, transcript, cfg=None):
    """[(inicio, fin)] en segundos de los tramos muertos."""
    cfg = cfg or {}
    frames, step = frame_signals(video_path, cfg.get("sample_step", 2.0))
    n = len(frames)
    if n < 3:
        return []
    # Movimiento: diferencia con la imagen anterior.
    motion = np.zeros(n)
    motion[1:] = np.mean(np.abs(frames[1:] - frames[:-1]), axis=(1, 2))
    # Bucle: se parece a alguna imagen de los últimos ~20 s (animación que se repite).
    look = max(2, int(cfg.get("loop_seconds", 20) / step))
    loop = np.zeros(n, bool)
    for i in range(n):
        past = frames[max(0, i - look):max(0, i - 1)]
        if len(past):
            loop[i] = np.min(np.mean(np.abs(past - frames[i]), axis=(1, 2))) < cfg.get("loop_diff", 4.0)
    words = words_per_window(transcript, n, step)
    # Suavizar las palabras en ~10 s (una frase suelta no salva una pantalla de espera).
    k = max(1, int(10 / step))
    words_s = np.convolve(words, np.ones(k), mode="same")
    # Una partida cambia mucho entre fotogramas clave (>15); una pantalla de
    # espera ilustrada con cuenta atrás y algo animado, 3,5-8,7 (medido).
    still = motion < cfg.get("max_motion", 10.0)
    quiet = words_s <= cfg.get("max_words_10s", 2)
    dead = quiet & (still | loop)
    # Por mayoría en ~10 s: un pico suelto (transición, alerta de donación)
    # partía una pantalla de espera de 12 min en trozos de menos de 20 s.
    dead = np.convolve(dead.astype(float), np.ones(k) / k, mode="same") >= cfg.get("majority", 0.6)

    # Tramos de al menos min_seconds.
    zones, start = [], None
    for i, d in enumerate(list(dead) + [False]):
        if d and start is None:
            start = i
        elif not d and start is not None:
            if (i - start) * step >= cfg.get("min_seconds", 20):
                zones.append((start * step, i * step))
            start = None
    # Comprobar caras es opcional: un personaje DIBUJADO y animado de la
    # pantalla de espera pasaba por persona real. Sin voz 20 s + imagen casi
    # quieta ya no es material de clip, haya o no alguien en pantalla.
    if cfg.get("check_faces", False):
        zones = _without_faces(video_path, zones, cfg)
    return [(round(a, 1), round(b, 1)) for a, b in zones]


def _without_faces(video_path, zones, cfg):
    """Descarta las zonas donde hay una persona de verdad (streamer callado).

    Una cara DIBUJADA (ilustración de la pantalla de espera) no cuenta: está
    siempre en el mismo sitio. Una persona real se mueve entre fotogramas."""
    from . import webcam
    keep = []
    for a, b in zones:
        frames = webcam.sample_frames(video_path, 6, a + 1, b - 1, skip_edges=0.0)
        found = [max(fs, key=lambda f: f[2]) for fs in (webcam.detect_faces(fr) for _, fr in frames) if fs]
        real = False
        if len(found) > cfg.get("max_frames_with_face", 1):
            cx = np.array([f[0] + f[2] / 2 for f in found])
            cy = np.array([f[1] + f[3] / 2 for f in found])
            fw = float(np.median([f[2] for f in found]))
            real = max(np.std(cx), np.std(cy)) / max(1.0, fw) >= cfg.get("min_face_motion", 0.02)
        if not real:
            keep.append((round(a, 1), round(b, 1)))
    return keep


def overlap(a, b, zones):
    """Fracción de [a, b] dentro de zonas muertas."""
    tot = 0.0
    for za, zb in zones:
        tot += max(0.0, min(b, zb) - max(a, za))
    return tot / max(1e-6, b - a)


def annotate_windows(windows, zones):
    """Marca las ventanas del transcript que caen en zonas muertas, para DeepSeek."""
    for w in windows:
        if overlap(w["start"], w["end"], zones) > 0.3:
            w["text"] = f"[DEAD AIR / WAITING SCREEN: do not clip] {w['text']}"
    return windows

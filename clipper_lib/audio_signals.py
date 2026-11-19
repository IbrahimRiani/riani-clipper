"""Picos de volumen del audio: señal gratuita de "aquí pasó algo".

En streams, un grito, una risa o un clutch suelen ser más fuertes que la
charla normal, y a veces el transcript no lo refleja ("AAAH" se transcribe
mal o nada). Medimos el volumen (RMS) cada 0.5 s con ffmpeg y marcamos como
pico los instantes muy por encima del nivel típico de ESE video.
"""
import subprocess

import numpy as np

SAMPLE_RATE = 8000


def loudness_curve(media_path, hop_seconds=0.5):
    """Volumen en dB por bloque de ``hop_seconds``. Devuelve (tiempos, db)."""
    cmd = ["ffmpeg", "-v", "error", "-i", media_path, "-vn", "-ac", "1",
           "-ar", str(SAMPLE_RATE), "-f", "s16le", "-"]
    raw = subprocess.run(cmd, capture_output=True, check=True).stdout
    audio = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
    hop = int(SAMPLE_RATE * hop_seconds)
    n = len(audio) // hop
    if n == 0:
        return np.array([]), np.array([])
    blocks = audio[: n * hop].reshape(n, hop)
    rms = np.sqrt(np.mean(blocks ** 2, axis=1)) + 1e-9
    db = 20 * np.log10(rms)
    times = np.arange(n) * hop_seconds
    return times, db


def find_peaks(media_path, cfg=None):
    """Lista de picos [{"t": segundo, "db_over": dB sobre la mediana}].

    Umbral: percentil ``percentile`` del video y al menos ``min_db_over`` dB
    sobre la mediana del habla. Picos más cercanos que ``merge_seconds`` se
    funden en uno (el más fuerte).
    """
    cfg = cfg or {}
    pct = float(cfg.get("percentile", 98))
    min_over = float(cfg.get("min_db_over", 8))
    merge = float(cfg.get("merge_seconds", 4))
    max_peaks = int(cfg.get("max_peaks", 60))

    times, db = loudness_curve(media_path)
    if len(db) == 0:
        return []
    voiced = db[db > -50]  # ignora silencios para la referencia
    if len(voiced) < 10:
        return []
    median = float(np.median(voiced))
    thr = max(float(np.percentile(voiced, pct)), median + min_over)

    peaks = []
    for t, v in zip(times, db):
        if v < thr:
            continue
        if peaks and t - peaks[-1]["t"] < merge:
            if v - median > peaks[-1]["db_over"]:
                peaks[-1] = {"t": float(t), "db_over": round(float(v - median), 1)}
            continue
        peaks.append({"t": float(t), "db_over": round(float(v - median), 1)})
    peaks.sort(key=lambda p: -p["db_over"])
    return sorted(peaks[:max_peaks], key=lambda p: p["t"])


def annotate_windows(windows, peaks):
    """Añade al texto de cada ventana una marca corta con sus picos, p. ej.
    ``[LOUD: 412s +14dB, 430s +11dB]``. El prompt explica qué significa."""
    if not peaks:
        return windows
    for w in windows:
        inside = [p for p in peaks if w["start"] <= p["t"] <= w["end"]]
        if inside:
            marks = ", ".join(f"{p['t']:.0f}s +{p['db_over']:.0f}dB" for p in inside[:4])
            w["text"] = f"{w['text']} [LOUD: {marks}]"
    return windows

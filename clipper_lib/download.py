"""Descarga de YouTube / Twitch con yt-dlp, opcionalmente solo un tramo.

Un VOD de Twitch puede durar 8 h (más de 15 GB). Con ``start``/``minutes`` solo se
baja ese trozo: yt-dlp pide a ffmpeg el rango, sin descargar el resto.
"""
import os
import re
import subprocess

import yt_dlp
from yt_dlp.utils import download_range_func


def _safe(name):
    name = re.sub(r"[^\w\-. ]+", "", name, flags=re.UNICODE).strip()
    return (name or "video")[:80]


def parse_time(value):
    """'1:23:45', '83:45' o '5025' -> segundos."""
    if value is None:
        return None
    parts = [float(p) for p in str(value).split(":")]
    secs = 0.0
    for p in parts:
        secs = secs * 60 + p
    return secs


def _best_muxed_hls(info, max_height):
    """El mejor formato HLS que ya lleva audio y video juntos (caso Twitch)."""
    best = None
    for f in info.get("formats") or []:
        if not str(f.get("protocol", "")).startswith("m3u8"):
            continue
        if f.get("vcodec") in (None, "none") or f.get("acodec") == "none":
            continue
        h = f.get("height") or 0
        if h > max_height:
            continue
        key = (h, f.get("fps") or 0, f.get("tbr") or 0)
        if best is None or key > best[0]:
            best = (key, f)
    return best[1] if best else None


def _download_hls_range(playlist_url, start, end, out_path, workers=8):
    """Baja los segmentos HLS que cubren [start, end] y los remuxa a MP4."""
    import shutil
    import tempfile
    from concurrent.futures import ThreadPoolExecutor
    from urllib.parse import urljoin

    import httpx

    with httpx.Client(timeout=60, follow_redirects=True) as http:
        text = http.get(playlist_url).raise_for_status().text
        init_uri, segs, t, dur = None, [], 0.0, None
        for line in text.splitlines():
            line = line.strip()
            if line.startswith("#EXT-X-MAP:"):
                m = re.search(r'URI="([^"]+)"', line)
                init_uri = m.group(1) if m else None
            elif line.startswith("#EXTINF:"):
                dur = float(line[8:].split(",")[0])
            elif line and not line.startswith("#") and dur is not None:
                if t + dur > start and t < end:
                    segs.append((t, line))
                t += dur
                dur = None
        if not segs:
            raise RuntimeError(f"El tramo {start:.0f}-{end:.0f}s está fuera del video.")
        first_t = segs[0][0]
        print(f"   {len(segs)} trozos HLS a descargar.")

        tmp = tempfile.mkdtemp(prefix="hls_")
        try:
            def fetch(i_uri):
                i, uri = i_uri
                p = os.path.join(tmp, f"{i:06d}.part")
                with open(p, "wb") as f:
                    f.write(http.get(urljoin(playlist_url, uri)).raise_for_status().content)
                return p

            jobs = list(enumerate(u for _, u in segs))
            with ThreadPoolExecutor(workers) as pool:
                parts = list(pool.map(fetch, jobs))
            joined = os.path.join(tmp, "joined" + (".mp4" if init_uri else ".ts"))
            with open(joined, "wb") as out:
                if init_uri:
                    out.write(http.get(urljoin(playlist_url, init_uri)).raise_for_status().content)
                for p in parts:
                    with open(p, "rb") as f:
                        shutil.copyfileobj(f, out)
            # Recorte fino al segundo pedido (los trozos son de ~10 s).
            cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{start - first_t:.3f}",
                   "-i", joined, "-t", f"{end - start:.3f}", "-c", "copy",
                   "-movflags", "+faststart", out_path]
            subprocess.run(cmd, check=True)
        finally:
            shutil.rmtree(tmp, ignore_errors=True)
    if not os.path.exists(out_path) or os.path.getsize(out_path) < 1024:
        raise RuntimeError(f"No se pudo generar {out_path}")


def download(url, out_dir, start=None, minutes=None, max_height=1080):
    """Descarga ``url`` en ``out_dir`` y devuelve (ruta_mp4, título)."""
    os.makedirs(out_dir, exist_ok=True)
    with yt_dlp.YoutubeDL({"quiet": True, "skip_download": True}) as ydl:
        info = ydl.extract_info(url, download=False)
    title = info.get("title") or info.get("id") or "video"
    duration = info.get("duration")

    base = _safe(title)
    opts = {
        "format": f"bv*[height<={max_height}]+ba/b[height<={max_height}]/b",
        "merge_output_format": "mp4",
        "outtmpl": os.path.join(out_dir, base + ".%(ext)s"),
        "overwrites": True,
        "noprogress": False,
    }
    if start is not None or minutes is not None:
        s = float(start or 0)
        e = s + float(minutes) * 60 if minutes else (duration or s + 3600)
        if duration:
            e = min(e, duration)
        base = f"{base}_{int(s)}-{int(e)}"
        print(f"✂️ Descargando solo el tramo {s:.0f}s → {e:.0f}s de "
              f"{(duration or 0) / 3600:.1f} h.")
        muxed = _best_muxed_hls(info, max_height)
        if muxed:
            # Twitch: un solo flujo HLS con audio+video, en trozos de ~10 s.
            # El playlist viene marcado como EVENT ("en directo") y ffmpeg se
            # queda esperando en vez de saltar, así que bajamos nosotros solo
            # los trozos que cubren el tramo y los unimos.
            path = os.path.join(out_dir, base + ".mp4")
            _download_hls_range(muxed["url"], s, e, path)
            return path, title
        # Sin force_keyframes_at_cuts: cortar exacto obliga a recodificar todo
        # (4 min para 8 min de video). Un corte en el keyframe más cercano basta.
        opts["download_ranges"] = download_range_func(None, [(s, e)])
        opts["outtmpl"] = os.path.join(out_dir, base + ".%(ext)s")
    else:
        print(f"📥 Descargando video completo ({(duration or 0) / 60:.0f} min).")

    with yt_dlp.YoutubeDL(opts) as ydl:
        ydl.download([url])
    path = os.path.join(out_dir, base + ".mp4")
    if not os.path.exists(path):
        raise RuntimeError(f"yt-dlp no generó {path}")
    return path, title

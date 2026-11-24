"""Solo descargar (y trocear) vídeos, sin sacar clips.

Uso:
    .venv\\Scripts\\python.exe tools\\descargar.py URL_o_archivo [más...] [--tramo 30] [--entrada]
        [--start 1:00:00] [--minutes 90]

- Descarga cada URL (YouTube, Twitch...) a downloads/. Un archivo local se usa tal cual.
- Si --tramo > 0, lo parte en trozos de esos minutos sin recodificar (segundos y sin
  perder calidad; cada corte cae en el fotograma clave más cercano).
- Los trozos van a downloads/<vídeo>_tramos/, o a entrada/ con --entrada para que
  "Procesar la carpeta entrada/" de la app los convierta en clips uno detrás de otro.
"""
import argparse
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

from clipper_lib import download  # noqa: E402


def duration_of(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration",
                          "-of", "json", path], capture_output=True, text=True)
    try:
        return float(json.loads(out.stdout)["format"]["duration"])
    except (ValueError, KeyError):
        return 0.0


def hms(seconds):
    s = int(round(seconds))
    return f"{s // 3600}h{s % 3600 // 60:02d}"


def nice(seconds):
    h, m = int(seconds // 3600), int(seconds % 3600 // 60)
    return f"{h} h {m} min" if h else f"{m} min"


def split(path, minutes, out_dir):
    """Parte ``path`` en trozos de ``minutes`` minutos. Devuelve las rutas."""
    total = duration_of(path)
    step = minutes * 60
    n = max(1, int(-(-total // step)))  # techo
    # Un último trozo de menos de 3 min se une al anterior (no da para clips).
    if n > 1 and total - (n - 1) * step < 180:
        n -= 1
    os.makedirs(out_dir, exist_ok=True)
    base = os.path.splitext(os.path.basename(path))[0]
    print(f"✂️ {nice(total)} → {n} tramo(s) de {minutes} min")
    parts = []
    for i in range(n):
        s = i * step
        e = total if i == n - 1 else (i + 1) * step
        name = f"{base}_parte{i + 1:02d}_{hms(s)}-{hms(e)}.mp4"
        dst = os.path.join(out_dir, name)
        cmd = ["ffmpeg", "-v", "error", "-y", "-ss", f"{s:.2f}", "-i", path, "-t", f"{e - s:.2f}",
               "-map", "0:v:0?", "-map", "0:a:0?", "-c", "copy", "-avoid_negative_ts", "make_zero",
               "-movflags", "+faststart", dst]
        subprocess.run(cmd, check=True)
        parts.append(dst)
        print(f"   📦 Tramo {i + 1} de {n}: {name}")
    return parts


def main():
    p = argparse.ArgumentParser(description="Descargar y trocear vídeos sin sacar clips.")
    p.add_argument("sources", nargs="+", help="URLs o archivos locales")
    p.add_argument("--tramo", type=float, default=30, help="minutos por tramo (0 = no trocear)")
    p.add_argument("--entrada", action="store_true", help="dejar los tramos en entrada/")
    p.add_argument("--start", help="solo URLs: empezar en (p. ej. 1:30:00)")
    p.add_argument("--minutes", type=float, help="solo URLs: minutos a bajar desde --start")
    args = p.parse_args()

    downloads = os.path.join(ROOT, "downloads")
    for k, src in enumerate(args.sources, start=1):
        print(f"🎬 Vídeo {k} de {len(args.sources)}: {src}")
        try:
            if src.startswith("http"):
                path, _title = download.download(src, downloads, start=download.parse_time(args.start),
                                                  minutes=args.minutes)
            else:
                path = src
            total = duration_of(path)
            print(f"✅ Descargado: {os.path.basename(path)} ({nice(total)})")
            if args.tramo and args.tramo > 0 and total > args.tramo * 60 + 180:
                base = os.path.splitext(os.path.basename(path))[0]
                out = os.path.join(ROOT, "entrada") if args.entrada else \
                    os.path.join(downloads, base + "_tramos")
                split(path, args.tramo, out)
                print(f"📁 Tramos en {out}")
            elif args.entrada:
                import shutil
                dst = os.path.join(ROOT, "entrada", os.path.basename(path))
                os.makedirs(os.path.dirname(dst), exist_ok=True)
                shutil.copy2(path, dst)
                print(f"📁 Copiado a {dst} (no hace falta trocear)")
            else:
                print(f"📁 En {path} (no hace falta trocear)")
        except Exception as e:  # que un vídeo roto no pare la lista
            print(f"❌ {src}: {type(e).__name__}: {e}")


if __name__ == "__main__":
    main()

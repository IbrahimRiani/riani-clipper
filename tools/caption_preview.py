"""Hoja de muestra con todos los estilos de subtítulos sobre un clip.

Uso:
    .venv\\Scripts\\python.exe tools\\caption_preview.py output\\<video> [n_clip] [segundo]

Genera output/<video>/debug/estilos_subtitulos.jpg con un fotograma por estilo.
"""
import glob
import json
import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT, "vendor", "openshorts"))
sys.path.insert(0, ROOT)
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

import subtitles  # noqa: E402


def main():
    out_dir = sys.argv[1]
    n = int(sys.argv[2]) if len(sys.argv) > 2 else 1
    at = float(sys.argv[3]) if len(sys.argv) > 3 else 5.0
    debug = os.path.join(out_dir, "debug")
    meta = json.load(open(glob.glob(os.path.join(debug, "*_metadata.json"))[0], encoding="utf-8"))
    clip = meta["shorts"][n - 1]
    base = glob.glob(os.path.join(debug, f"*_clip_{n}.mp4"))[0]
    seam = os.environ.get("CAPTION_SEAM_Y")
    splits = [(0, 999)] if seam else None
    tiles = []
    variants = list(subtitles.CAPTION_PRESETS.items())
    if os.environ.get("FONT_SHEET"):
        # Un preset (p. ej. oneword) con cada fuente de assets/fonts.
        import clipper
        font_files = clipper.register_fonts(subtitles)
        from fontTools.ttLib import TTFont
        base_style = subtitles.CAPTION_PRESETS[os.environ["FONT_SHEET"]]
        variants = []
        for f in sorted(glob.glob(os.path.join(ROOT, "assets", "fonts", "*.ttf"))):
            fam = TTFont(f)["name"].getDebugName(1)
            variants.append((fam, {**base_style, "font_name": fam}))
    for name, preset in variants:
        style = {**subtitles.AUTO_CAPTION_STYLE, **preset}
        ass = os.path.join(debug, f"_preview_{len(tiles)}.ass")
        kwargs = dict(max_chars=style.get("max_chars", 18), max_duration=style.get("max_duration", 1.4),
                      alignment=style.get("alignment", "bottom"), fontsize=style.get("font_size", 40),
                      font_name=style.get("font_name"), font_color=style.get("font_color"),
                      highlight_color=style.get("highlight_color"), border_color="#000000",
                      border_width=style.get("border_width", 2), effect=style.get("effect", "pop"),
                      uppercase=style.get("uppercase", True), base_opacity=style.get("base_opacity", 1.0),
                      shadow=style.get("shadow", 0), reveal=style.get("reveal", False),
                      split_ranges=splits)
        try:
            subtitles.generate_ass(meta["transcript"], clip["start"], clip["end"], ass, **kwargs)
            if os.environ.get("FONT_SHEET"):
                from clipper_lib import caption_fit
                caption_fit.fit(ass, font_files)
        except TypeError:
            # Firma distinta: pasar solo lo básico.
            subtitles.generate_ass(meta["transcript"], clip["start"], clip["end"], ass)
        tile = os.path.join(debug, f"_preview_{len(tiles)}.jpg")
        assf = ass.replace("\\", "/").replace(":", "\\:")
        fonts = os.path.join(ROOT, "vendor", "openshorts", "fonts").replace("\\", "/").replace(":", "\\:")
        band = (f"crop=1080:560:0:{max(0, int(float(seam) * 1920) - 330)}," if seam else "")
        vf = (f"ass=filename='{assf}':fontsdir='{fonts}',{band}scale=360:-2,"
              f"drawtext=text='{name}':fontfile='{fonts}/Anton-Regular.ttf':x=10:y=10:fontsize=28:fontcolor=white:box=1:boxcolor=black@0.6")
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-copyts", "-ss", str(at), "-i", base, "-frames:v", "1",
                        "-vf", vf, tile], check=True)
        tiles.append(tile)
        os.remove(ass)
    sheet = os.path.join(debug, "fuentes_subtitulos.jpg" if os.environ.get("FONT_SHEET")
                         else "estilos_subtitulos.jpg")
    cols = 5 if len(tiles) > 7 else len(tiles)
    inputs = sum([["-i", t] for t in tiles], [])
    subprocess.run(["ffmpeg", "-v", "error", "-y", *inputs, "-filter_complex",
                    f"xstack=inputs={len(tiles)}:grid={cols}x{-(-len(tiles) // cols)}:fill=black"
                    if len(tiles) > 1 else "null", sheet], check=True)
    for t in tiles:
        os.remove(t)
    print(sheet)


if __name__ == "__main__":
    main()

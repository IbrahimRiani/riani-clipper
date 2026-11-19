"""Encoge las palabras de subtítulo que no caben a lo ancho.

El estilo "oneword" usa letra muy grande (70) y una palabra larga en una
fuente ancha ("PERSONALIZACIÓN" en Rubik Mono) se sale de la pantalla. Tras
generar el .ass, medimos cada línea con la fuente real y, si pasa del
``max_frac`` del ancho, le añadimos {\\fscx..\\fscy..} para reducirla.
"""
import re

from PIL import ImageFont

OUT_W, OUT_H, PLAY_RES_Y = 1080, 1920, 288


def _px_per_unit(path):
    """Píxeles de salida por unidad de PIL a ese tamaño de letra. libass ajusta
    el tamaño a la altura total de la fuente (usWinAscent+usWinDescent), no al
    "em" como PIL. Medido renderizando con libass: error del 2-6 %."""
    from fontTools.ttLib import TTFont
    t = TTFont(path)
    o = t["OS/2"]
    ratio = (o.usWinAscent + o.usWinDescent) / t["head"].unitsPerEm
    return (OUT_H / PLAY_RES_Y) / ratio
_TAG = re.compile(r"\{[^}]*\}")
_SCALE = re.compile(r"\\fsc[xy]([\d.]+)")
_SCALE_SUB = re.compile(r"(\\fsc[xy])([\d.]+)")


def _styles(lines):
    fmt, styles = None, {}
    for ln in lines:
        if ln.startswith("Format:") and fmt is None and "Fontname" in ln:
            fmt = [x.strip() for x in ln[7:].split(",")]
        elif ln.startswith("Style:") and fmt:
            vals = [x.strip() for x in ln[6:].split(",")]
            d = dict(zip(fmt, vals))
            styles[d["Name"]] = (d["Fontname"], float(d["Fontsize"]), float(d.get("Outline", 0) or 0),
                                 float(d.get("ScaleX", 100) or 100))
    return styles


_TRAIL_PUNCT = re.compile(r"[,.;:…]+(?=(\{[^}]*\})*$)")


def strip_punct(ass_path):
    """Quita comas/puntos al final de cada bloque ("HOSTIA," -> "HOSTIA").
    Se mantienen ¿? ¡! porque cambian el sentido."""
    with open(ass_path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    for i, ln in enumerate(lines):
        if ln.startswith("Dialogue:"):
            parts = ln.split(",", 9)
            if len(parts) == 10:
                parts[9] = _TRAIL_PUNCT.sub("", parts[9])
                lines[i] = ",".join(parts)
    with open(ass_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def fit(ass_path, font_files, max_frac=0.86):
    """``font_files``: {nombre de fuente: ruta .ttf}. Devuelve nº de líneas encogidas."""
    with open(ass_path, encoding="utf-8") as f:
        lines = f.read().split("\n")
    styles = _styles(lines)
    cache, changed = {}, 0
    budget = OUT_W * max_frac  # en píxeles de salida
    ppu = {}
    for i, ln in enumerate(lines):
        if not ln.startswith("Dialogue:"):
            continue
        parts = ln.split(",", 9)
        if len(parts) < 10:
            continue
        st = styles.get(parts[3].strip())
        if not st:
            continue
        font, size, outline, scale_x = st
        path = font_files.get(font)
        if not path:
            continue
        text = _TAG.sub("", parts[9]).replace("\\N", " ").replace("\\h", " ")
        if not text.strip():
            continue
        key = (path, int(size))
        if key not in cache:
            cache[key] = ImageFont.truetype(path, max(1, int(size)))
        # La escala máxima que aparece en la línea (el efecto "pop" de
        # OpenShorts anima de \fscx90 a \fscx108): el ancho real es el de ese pico.
        scales = [float(v) for v in _SCALE.findall(parts[9])] or [scale_x]
        peak = max(scales)
        if path not in ppu:
            ppu[path] = _px_per_unit(path)
        width = (cache[key].getlength(text) * peak / 100 + 2 * outline) * ppu[path]
        if width > budget:
            k = budget / width
            body = parts[9]
            if _SCALE.search(body):
                body = _SCALE_SUB.sub(lambda m: f"{m.group(1)}{float(m.group(2)) * k:.0f}", body)
            else:
                body = "{\\fscx%d\\fscy%d}" % (scale_x * k, scale_x * k) + body
            parts[9] = body
            lines[i] = ",".join(parts)
            changed += 1
    if changed:
        with open(ass_path, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
    return changed

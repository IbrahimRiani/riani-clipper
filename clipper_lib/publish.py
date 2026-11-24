"""Textos para publicar cada clip (título, descripción, hashtags) y campañas.

DeepSeek ya escribe el título y las descripciones al elegir los clips; aquí se
juntan con lo obligatorio de la campaña (hashtags y nombres) y se guardan en
``publicar.json`` (lo editado en la app) y ``publicar.txt`` (para copiar).
"""
import glob
import json
import os
import re

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CAMPAIGNS = os.path.join(ROOT, "campanas")
_TAG = re.compile(r"#\w+", re.UNICODE)


def campaigns():
    """Nombres de los archivos campanas/*.yaml."""
    return sorted(os.path.splitext(os.path.basename(p))[0]
                  for p in glob.glob(os.path.join(CAMPAIGNS, "*.yaml")))


def load_campaign(name):
    """La sección ``campaign`` de una campaña ({} si no hay)."""
    path = os.path.join(CAMPAIGNS, f"{name}.yaml")
    if not name or not os.path.exists(path):
        return {}
    import yaml
    with open(path, encoding="utf-8") as f:
        return (yaml.safe_load(f) or {}).get("campaign") or {}


def missing_overlay(name):
    """Ruta del overlay obligatorio de la campaña si falta el archivo, o None."""
    f = (load_campaign(name).get("overlay") or {}).get("file")
    path = os.path.join(CAMPAIGNS, f) if f else None
    return path if path and not os.path.exists(path) else None


def brand_overlay(camp, tmp_dir):
    """La etiqueta fija de la campaña lista para superponer: {"path", "width",
    "x", "y"}, sin el margen transparente de la imagen. None si no hay."""
    ov = (camp or {}).get("overlay") or {}
    src = os.path.join(CAMPAIGNS, ov["file"]) if ov.get("file") else None
    if not src:
        return None
    if not os.path.exists(src):
        raise FileNotFoundError(f"Falta el overlay obligatorio de la campaña: {src}")
    from PIL import Image
    img = Image.open(src).convert("RGBA")
    # Umbral: algunas imágenes traen píxeles casi invisibles hasta el borde.
    solid = img.getchannel("A").point(lambda a: 255 if a > 40 else 0)
    img = img.crop(solid.getbbox() or (0, 0, *img.size))
    os.makedirs(os.path.join(tmp_dir, "debug"), exist_ok=True)
    path = os.path.join(tmp_dir, "debug", "overlay_campana.png")
    img.save(path)
    return {"path": path, "width": ov.get("width", 460), "x": ov.get("x", 540), "y": ov.get("y", 1440)}


def _read(path, default):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except (OSError, ValueError):
        return default


def merge_tags(*texts):
    """Hashtags de varios textos, sin repetir y en orden."""
    seen, out = set(), []
    for t in texts:
        for tag in _TAG.findall(t or ""):
            if tag.lower() not in seen:
                seen.add(tag.lower())
                out.append(tag)
    return " ".join(out)


def campaign_of(out_dir):
    return _read(os.path.join(out_dir, "clips.json"), {}).get("campaign")


def posts(out_dir, camp=None):
    """[{file, title, description, hashtags}] de una carpeta de resultados.
    Lo editado a mano manda sobre lo que escribió DeepSeek."""
    camp = camp or {}
    saved = _read(os.path.join(out_dir, "publicar.json"), {})
    out = []
    for c in _read(os.path.join(out_dir, "clips.json"), {}).get("clips", []):
        name = c.get("file")
        if not name or not os.path.exists(os.path.join(out_dir, name)):
            continue
        if name in saved:
            out.append({"file": name, **saved[name]})
            continue
        raw = _read(os.path.join(out_dir, "debug", f"clip_{name[:2]}.json"), {}).get("raw", {})
        desc = raw.get("video_description_for_tiktok") or c.get("title") or ""
        out.append({"file": name, "title": c.get("title") or "",
                    "description": re.sub(r"\s+", " ", _TAG.sub("", desc)).strip(),
                    # Primero los obligatorios: si hay que recortar, que no caigan.
                    "hashtags": merge_tags(camp.get("hashtags"), desc)})
    return out


def final_text(post, camp=None):
    """El texto que se pega al publicar: descripción, nombres y hashtags."""
    camp = camp or {}
    parts = [post.get("description", "").strip()]
    mention = (camp.get("mention") or "").strip()
    names = camp.get("names") or []
    body = parts[0].lower()
    if mention and not any(n.lower() in body for n in names):
        parts.append(mention)
    parts.append(merge_tags(camp.get("hashtags"), post.get("hashtags")))
    return "\n".join(p for p in parts if p)


def problems(post, camp=None):
    """Avisos sobre lo obligatorio de la campaña y los límites de las redes."""
    camp, text, out = camp or {}, final_text(post, camp), []
    title = post.get("title", "")
    if len(title) > 100:
        out.append(f"El título tiene {len(title)} letras; YouTube admite 100.")
    if len(text) > 2200:
        out.append(f"El texto tiene {len(text)} letras; Instagram admite 2.200.")
    if len(_TAG.findall(text)) > 30:
        out.append("Hay más de 30 hashtags, el máximo de Instagram.")
    if camp.get("english_only") and re.search(r"[¿¡ñáéíóúÑÁÉÍÓÚ]", title + text):
        out.append("Parece que hay texto en español y la campaña solo admite inglés.")
    return out


def save(out_dir, name, title, description, hashtags, camp=None):
    """Guarda lo editado de un clip y rehace publicar.txt."""
    path = os.path.join(out_dir, "publicar.json")
    data = _read(path, {})
    data[name] = {"title": title.strip(), "description": description.strip(),
                  "hashtags": merge_tags(hashtags)}
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    write_txt(out_dir, camp)


def write_txt(out_dir, camp=None):
    with open(os.path.join(out_dir, "publicar.txt"), "w", encoding="utf-8") as f:
        for p in posts(out_dir, camp):
            f.write(f"=== {p['file']}\nTÍTULO: {p['title']}\n{final_text(p, camp)}\n\n")

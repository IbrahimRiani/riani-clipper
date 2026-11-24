"""Ordena la carpeta de salida tras el render de OpenShorts.

Resultado:
    output/<video>/01_titulo-del-clip.mp4   (clip final con subtítulos)
    output/<video>/clips.json               (resumen: tiempos, score, títulos)
    output/<video>/debug/clip_01.json       (todo lo que se sabe del clip)
    output/<video>/debug/...                (clips sin subtítulos, .ass, layout)
"""
import glob
import json
import os
import re
import shutil
import unicodedata


def _slug(text, n=50):
    text = unicodedata.normalize("NFKD", text or "").encode("ascii", "ignore").decode()
    text = re.sub(r"[^\w\s-]", "", text).strip().lower()
    return re.sub(r"[\s_-]+", "-", text)[:n].strip("-") or "clip"


def _face_box_out(raw, debug_dir):
    """Caja de la cara del streamer en la salida 1080x1920 (layout gaming), o None."""
    try:
        with open(raw + ".gaming.json", encoding="utf-8") as f:
            plan = json.load(f)
        with open(os.path.join(debug_dir, "layout_detection.json"), encoding="utf-8") as f:
            face = json.load(f)["webcam"]["face"]
    except (OSError, ValueError, KeyError, TypeError):
        return None
    if plan.get("mode") != "cam_top" or not plan.get("cam_crop"):
        return None
    cx, cy, cw, ch = plan["cam_crop"]
    s = 1080 / cw
    fx, fy, fw, fh = face
    # La cara se mueve un poco: margen del 10 %.
    m = 0.10
    return ((fx - cx - m * fw) * s, plan["cam_y"] + (fy - cy - m * fh) * s,
            fw * (1 + 2 * m) * s, fh * (1 + 2 * m) * s)


def clip_overlay_items(emoji_items, transcript, clip, lol_index, lol_cfg):
    """Emojis del clip + iconos del LoL mencionados (que sustituyen al emoji cercano)."""
    if not lol_index:
        return emoji_items
    from . import lol
    icons = lol.mentions(transcript, clip["start"], clip["end"], lol_index, lol_cfg)
    if icons:
        print(f"   🎮 Iconos LoL: " + ", ".join(f"{m['name']} ({m['t']:.0f}s)" for m in icons))
    # El sonido elegido para un emoji se conserva aunque el icono lo sustituya.
    merged = lol.merge_with_emojis(emoji_items, icons)
    for m in merged:
        if m.get("png") and not m.get("s"):
            near = [e for e in emoji_items if e.get("s") and abs(e["t"] - m["t"]) <= 1.5]
            if near:
                m["s"] = near[0]["s"]
    return merged


def place_items(raw, items, cfg, debug_dir=None):
    """Copia de ``items`` con "x"/"y" de cada emoji según la escena en su segundo."""
    out = []
    for it in items:
        x, y = emoji_position(raw, cfg, debug_dir, t=it.get("t"))
        out.append({**it, "x": x, "y": y})
    return out


def emoji_position(raw, cfg, debug_dir=None, t=None):
    """(x, y) del centro del emoji: encima del subtítulo y, si ahí taparía la
    cara del streamer, desplazado al lado con más hueco."""
    y = _emoji_y(raw, cfg)
    size = int(cfg.get("size", 230))
    box = _face_box_out(raw, debug_dir or os.path.dirname(raw))
    if not box:
        return None, y
    fx, fy, fw, fh = box
    # Parte del emoji (centrado) que caería sobre la cara.
    ex0, ey0 = 540 - size / 2, y - size / 2
    ix = max(0, min(ex0 + size, fx + fw) - max(ex0, fx))
    iy = max(0, min(ey0 + size, fy + fh) - max(ey0, fy))
    if ix * iy <= cfg.get("face_overlap_max", 0.35) * size * size:
        return None, y  # tapa poco o nada: centrado encima del texto
    left, right = fx, 1080 - (fx + fw)
    if max(left, right) >= size * 0.9:
        return (left / 2 if left > right else fx + fw + right / 2), y
    return None, y


def _emoji_y(raw, cfg, t=None):
    """Altura del centro del emoji: encima del subtítulo."""
    size = int(cfg.get("size", 200))
    gap = int(cfg.get("gap_above_caption", 85))
    seam = os.environ.get("CAPTION_SEAM_Y")
    split = False
    try:
        with open(raw + ".layout.json", encoding="utf-8") as f:
            ranges = json.load(f).get("ranges", [])
        split = any(r.get("layout") == "split" and (t is None or r["start"] <= t < r["end"])
                    for r in ranges)
    except (OSError, ValueError):
        pass
    if split:
        # Costura cam/juego si hay webcam; si no (podcast en split), el centro.
        caption_y = float(seam) * 1920 if seam else 960
    else:
        caption_y = 1920 - 330  # subtítulo abajo (MarginV de OpenShorts)
    return caption_y - gap - size / 2


def organize(out_dir, cfg=None):
    cfg = cfg or {}
    metas = glob.glob(os.path.join(out_dir, "*_metadata.json"))
    if not metas:
        return
    meta_path = metas[0]
    with open(meta_path, encoding="utf-8") as f:
        meta = json.load(f)
    base = os.path.basename(meta_path)[: -len("_metadata.json")]
    debug = os.path.join(out_dir, "debug")
    os.makedirs(debug, exist_ok=True)

    emoji_cfg = cfg.get("emojis", {})
    emoji_plan = {}
    if emoji_cfg.get("enabled", False):
        try:
            from . import emojis
            emoji_plan = emojis.plan(meta, emoji_cfg)
        except Exception as e:
            print(f"   ⚠️ Emojis omitidos ({type(e).__name__}: {e}).")
    root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    lol_cfg = cfg.get("lol_icons", {})
    lol_index = None
    if lol_cfg.get("enabled", False):
        try:
            from . import lol
            lol_index = lol.load()
            if not lol.is_lol_video(meta["transcript"], lol_index, lol_cfg):
                print("   🎮 Iconos LoL: el vídeo no va de LoL, no se ponen.")
                lol_index = None
        except Exception as e:
            print(f"   ⚠️ Iconos del LoL omitidos ({type(e).__name__}: {e}).")

    from . import publish
    brand = publish.brand_overlay(cfg.get("campaign"), out_dir)
    summary = []
    for i, clip in enumerate(meta.get("shorts", []), start=1):
        title = clip.get("video_title_for_youtube_short") or f"clip {i}"
        final = os.path.join(out_dir, f"{i:02d}_{_slug(title)}.mp4")
        subbed = glob.glob(os.path.join(out_dir, f"subtitled_*_{base}_clip_{i}.mp4"))
        raw = os.path.join(out_dir, f"{base}_clip_{i}.mp4")
        done = False
        overlay_items = clip_overlay_items(emoji_plan.get(i, []), meta["transcript"], clip,
                                           lol_index, lol_cfg)
        if overlay_items:
            emoji_plan[i] = overlay_items
        if subbed and (emoji_plan.get(i) or brand):
            try:
                from . import emojis
                items = place_items(raw, emoji_plan[i], emoji_cfg, out_dir) if emoji_plan.get(i) else []
                if emojis.overlay(subbed[0], final, items,
                                  os.path.join(root, "assets", "emoji"), None, emoji_cfg, brand=brand):
                    os.remove(subbed[0])
                    done = True
            except Exception as e:
                print(f"   ⚠️ Emojis del clip {i} omitidos ({e}).")
                if brand:
                    print(f"   ❌ El clip {i} ha salido SIN el overlay obligatorio de la campaña: no lo subas.")
        if done:
            pass
        elif subbed:
            shutil.move(subbed[0], final)
        elif os.path.exists(raw):
            shutil.copy2(raw, final)
        else:
            final = None
        gaming = None
        if os.path.exists(raw + ".gaming.json"):
            with open(raw + ".gaming.json", encoding="utf-8") as f:
                gaming = json.load(f)
        if os.path.exists(raw + ".debug.mp4"):
            shutil.move(raw + ".debug.mp4", os.path.join(debug, f"clip_{i:02d}_cajas.mp4"))
        for p in [raw, raw + ".layout.json", raw + ".gaming.json", raw + ".split.json"]:
            if os.path.exists(p):
                shutil.move(p, os.path.join(debug, os.path.basename(p)))
        info = {
            "file": os.path.basename(final) if final else None,
            "start": clip.get("start"), "end": clip.get("end"),
            "duration": round((clip.get("end") or 0) - (clip.get("start") or 0), 2),
            "score": clip.get("predicted_score"),
            "title": title,
            "hook": clip.get("viral_hook_text"),
            "why": clip.get("why"),
            "emojis": emoji_plan.get(i, []),
        }
        summary.append(info)
        with open(os.path.join(debug, f"clip_{i:02d}.json"), "w", encoding="utf-8") as f:
            json.dump({**info, "layout_plan": gaming, "raw": clip}, f, ensure_ascii=False, indent=2)

    for p in glob.glob(os.path.join(out_dir, "autosubs_*.ass")):
        shutil.move(p, os.path.join(debug, os.path.basename(p)))
    shutil.move(meta_path, os.path.join(debug, os.path.basename(meta_path)))
    with open(os.path.join(out_dir, "clips.json"), "w", encoding="utf-8") as f:
        json.dump({"cost": meta.get("cost_analysis"), "campaign": cfg.get("campaign_id"), "clips": summary},
                  f, ensure_ascii=False, indent=2)
    # Texto listo para copiar al publicar, con lo obligatorio de la campaña.
    publish.write_txt(out_dir, cfg.get("campaign"))
    print(f"\n📁 {len(summary)} clip(s) en {out_dir}")
    for s in summary:
        print(f"   {s['file']}  ({s['duration']:.0f}s, score {s['score']})")

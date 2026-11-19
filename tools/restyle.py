"""Vuelve a poner subtítulos (y emojis) a clips ya generados.

Útil al cambiar la fuente o el estilo en config.yaml: no repite transcripción,
DeepSeek ni reencuadre. Usa los clips reencuadrados sin subtítulos que quedan
en output/<video>/debug/ y los emojis ya elegidos (clips.json).

Uso:
    .venv\\Scripts\\python.exe tools\\restyle.py output\\<video> [--new-emojis] [--no-emojis]
"""
import argparse
import glob
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "vendor", "openshorts"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("out_dir")
    p.add_argument("--new-emojis", action="store_true", help="Pedir emojis nuevos a DeepSeek.")
    p.add_argument("--no-emojis", action="store_true")
    p.add_argument("--relayout", action="store_true",
                   help="Volver a cortar y reencuadrar (layout gaming) desde el vídeo original.")
    p.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    args = p.parse_args()
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

    import clipper
    cfg = clipper.load_config(args.config)
    clipper.apply_config(cfg)
    clipper._install_windows_shims()
    clipper._install_hooks(cfg)

    out = os.path.abspath(args.out_dir)
    debug = os.path.join(out, "debug")
    meta = json.load(open(glob.glob(os.path.join(debug, "*_metadata.json"))[0], encoding="utf-8"))
    summary = json.load(open(os.path.join(out, "clips.json"), encoding="utf-8"))
    base = os.path.basename(glob.glob(os.path.join(debug, "*_metadata.json"))[0])[:-len("_metadata.json")]

    # Costura de los subtítulos, igual que en la ejecución original.
    det = os.path.join(debug, "layout_detection.json")
    if os.path.exists(det):
        from clipper_lib import gaming_render
        analysis = json.load(open(det, encoding="utf-8"))
        if analysis.get("webcam"):
            geo = gaming_render.video_geometry(analysis, cfg.get("layout", {}))
            os.environ["CAPTION_SEAM_Y"] = f"{(geo['game_y'] + cfg.get('layout', {}).get('caption_offset_px', 0)) / gaming_render.OUT_H:.4f}"

    emoji_cfg = cfg.get("emojis", {})
    plan = {}
    if not args.no_emojis and emoji_cfg.get("enabled", False):
        if args.new_emojis:
            from clipper_lib import emojis
            plan = emojis.plan(meta, emoji_cfg)
        else:
            plan = {i: c.get("emojis", []) for i, c in enumerate(summary["clips"], start=1)}

    import layout_ranges
    import main as osmain  # funciones de OpenShorts (auto_caption_clip)
    from clipper_lib import emojis, organize

    source = None
    if args.relayout:
        name = meta.get("source_video")
        for d in ("samples", "downloads"):
            cand = os.path.join(ROOT, d, name or "")
            if name and os.path.exists(cand):
                source = cand
        if not source or not os.path.exists(det):
            sys.exit(f"--relayout: no encuentro el vídeo original ({name}) en samples/ ni downloads/")
        # La costura de los subtítulos también aplica al layout chatting.
        analysis = json.load(open(det, encoding="utf-8"))
        from clipper_lib import chatting_render, gaming_render, layout as lay
        from ffmpeg_utils import cut_clip
        lcfg = cfg.get("layout", {})
        chosen, why = lay.decide(analysis, lcfg.get("mode", "auto"), lcfg)
        print(f"🧩 Layout: {chosen} — {why}")

    for i, clip in enumerate(meta["shorts"], start=1):
        raw = os.path.join(debug, f"{base}_clip_{i}.mp4")
        if source:
            tmp = os.path.join(debug, f"_tmp_cut_{i}.mp4")
            cut_clip(source, tmp, clip["start"], clip["end"], i)
            if chosen == "gaming":
                gaming_render.render(tmp, raw, analysis, lcfg)
            else:
                chatting_render.render(tmp, raw, analysis, lcfg)
            os.remove(tmp)
            dbg = raw + ".debug.mp4"
            if os.path.exists(dbg):
                os.replace(dbg, os.path.join(debug, f"clip_{i:02d}_cajas.mp4"))
        info = summary["clips"][i - 1]
        final = os.path.join(out, info["file"])
        if not os.path.exists(raw):
            print(f"   ⚠️ Falta {raw}")
            continue
        splits = layout_ranges.split_ranges(layout_ranges.read(raw))
        captioned = osmain.auto_caption_clip(raw, meta["transcript"], clip["start"], clip["end"],
                                             split_ranges=splits)
        src = captioned or raw
        if plan.get(i):
            items = organize.place_items(raw, plan[i], emoji_cfg, debug)
            emojis.overlay(src, final, items, os.path.join(ROOT, "assets", "emoji"), None, emoji_cfg)
            if captioned:
                os.remove(captioned)
        else:
            os.replace(src, final) if captioned else None
        info["emojis"] = plan.get(i, [])
        print(f"   ✅ {info['file']}")
    json.dump(summary, open(os.path.join(out, "clips.json"), "w", encoding="utf-8"),
              ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()

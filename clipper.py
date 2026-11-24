"""Claude Clipper: convierte videos largos en clips verticales 9:16.

Uso:
    python clipper.py --url "https://www.twitch.tv/videos/123"
    python clipper.py --input samples/video.mp4 -o output/mi_video

Por debajo usa el pipeline de OpenShorts (MIT, ver vendor/openshorts/UPSTREAM.md),
con DeepSeek como LLM de texto y la configuración de config.yaml.
"""
import argparse
import os
import runpy
import sys
import types

ROOT = os.path.dirname(os.path.abspath(__file__))
VENDOR = os.path.join(ROOT, "vendor", "openshorts")


def _install_windows_shims():
    """OpenShorts está pensado para Linux/Docker: usa fcntl (bloqueos de
    archivo) y resource (medir CPU), que no existen en Windows. Aquí ponemos
    sustitutos que no hacen nada. Solo afectan a bloqueos entre procesos
    paralelos y a una estadística final, que este CLI no necesita."""
    if os.name != "nt":
        return
    if "fcntl" not in sys.modules:
        fcntl = types.ModuleType("fcntl")
        fcntl.LOCK_EX, fcntl.LOCK_NB, fcntl.LOCK_UN, fcntl.LOCK_SH = 2, 4, 8, 1
        fcntl.flock = lambda *a, **k: None
        sys.modules["fcntl"] = fcntl
    if "resource" not in sys.modules:
        resource = types.ModuleType("resource")
        resource.RUSAGE_SELF, resource.RUSAGE_CHILDREN = 0, -1
        usage = types.SimpleNamespace(ru_utime=0.0, ru_stime=0.0)
        resource.getrusage = lambda who: usage
        sys.modules["resource"] = resource


def load_config(path):
    import yaml
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def merge_campaign(cfg, name):
    """Mezcla campanas/<name>.yaml (o una ruta) encima de la configuración."""
    path = name if os.path.exists(name) else os.path.join(ROOT, "campanas", name + ".yaml")

    def merge(a, b):
        for k, v in b.items():
            if isinstance(v, dict) and isinstance(a.get(k), dict):
                merge(a[k], v)
            else:
                a[k] = v
    merge(cfg, load_config(path))
    from clipper_lib import publish
    if publish.missing_overlay(name):
        raise SystemExit(f"❌ Falta el overlay obligatorio de la campaña: {publish.missing_overlay(name)}")
    cfg["campaign_id"] = os.path.splitext(os.path.basename(path))[0]
    return cfg


def apply_config(cfg):
    """Traduce config.yaml a las variables de entorno que lee OpenShorts."""
    from dotenv import load_dotenv
    load_dotenv(os.path.join(ROOT, ".env"))

    llm = cfg.get("llm", {})
    key = os.environ.get("DEEPSEEK_API_KEY", "").strip()
    if not key:
        sys.exit("Falta DEEPSEEK_API_KEY en el archivo .env (ver .env.example).")
    env = {
        "LLM_PROVIDER": "openai",
        "LLM_BASE_URL": llm.get("base_url", "https://api.deepseek.com"),
        "LLM_MODEL": llm.get("model", "deepseek-flash"),
        "LLM_API_KEY": key,
        "LLM_TIMEOUT": str(llm.get("timeout_seconds", 180)),
        # DeepSeek tiene contexto grande: lotes de puntuación más grandes = menos llamadas.
        "LLM_SCORE_BATCH": "8",
        "LLM_JSON_SCHEMA": "0",
    }
    if llm.get("thinking") in ("disabled", "enabled"):
        import json
        env["LLM_EXTRA_BODY"] = json.dumps({"thinking": {"type": llm["thinking"]}})
    tr = cfg.get("transcription", {})
    env.update({
        "TRANSCRIBE_BACKEND": "whisper",
        "WHISPER_MODEL": str(tr.get("model", "small")),
        "WHISPER_DEVICE": str(tr.get("device", "cpu")),
        "WHISPER_COMPUTE": str(tr.get("compute", "int8")),
    })
    clips = cfg.get("clips", {})
    env.update({
        "CLIP_MIN_SECONDS": str(clips.get("min_seconds", 20)),
        "CLIP_MAX_SECONDS": str(clips.get("max_seconds", 60)),
    })
    render = cfg.get("render", {})
    env.update({
        "CLIP_WORKERS": str(render.get("workers", 2)),
        "FFMPEG_ENCODER": str(render.get("encoder", "auto")),
        "SCENE_ENGINE": str(render.get("scene_engine", "pyscenedetect")),
        # Funciones de OpenShorts que necesitan Gemini (ver video): apagadas.
        "AUTO_LAYOUT": "0",
        "GEMINI_API_KEY": "",
        # Sin marca de agua ni cosas del SaaS.
        "WATERMARK": "0",
        "EARLY_AUDIO": "0",
        "ASR_LOCK_DIR": os.path.join(ROOT, "output", ".locks"),
    })
    for k, v in env.items():
        os.environ[k] = v


def register_fonts(subtitles):
    """Copia las fuentes de assets/fonts (Google Fonts, OFL) a la carpeta que
    libass mira al quemar subtítulos, y las registra para medir el ancho."""
    import glob
    import shutil
    from fontTools.ttLib import TTFont
    dst = os.path.join(VENDOR, "fonts")
    files = {}
    for src in glob.glob(os.path.join(ROOT, "assets", "fonts", "*.ttf")) +             glob.glob(os.path.join(dst, "*.ttf")):
        name = os.path.basename(src)
        if not os.path.exists(os.path.join(dst, name)):
            shutil.copy2(src, os.path.join(dst, name))
        # libass busca por el nombre "legacy" (nameID 1): "Poppins Black", no "Poppins".
        family = TTFont(src)["name"].getDebugName(1)
        subtitles._FONT_FILES.setdefault(family, name)
        files[family] = os.path.join(dst, name)
    return files


def _install_hooks(cfg):
    """Engancha nuestras mejoras en OpenShorts sin editar sus archivos:
    sustituimos funciones de sus módulos antes de que main.py las importe."""
    from clipper_lib import audio_signals, prompts
    import clip_selection
    import gemini_worker
    import transcribe_backends

    peaks_cfg = cfg.get("audio_peaks", {})
    prompts.patch(gemini_worker, (cfg.get("campaign") or {}).get("hint", ""))

    import subtitles

    # Idioma de la transcripción. En streams el juego puede hablar otro idioma
    # (GTA en inglés): whisper decidía con los primeros 30 s y, en modo inglés,
    # TRADUCÍA al inglés lo que decía el streamer. Por eso se puede fijar.
    lang = str(cfg.get("transcription", {}).get("language", "auto")).strip().lower()
    if lang and lang != "auto":
        subtitles.WHISPER_TRANSCRIBE_PARAMS["language"] = lang
    else:
        # Auto: votar el idioma en 10 trozos del audio, no solo en el primero.
        subtitles.WHISPER_TRANSCRIBE_PARAMS["language_detection_segments"] = 10
    subtitles.WHISPER_TRANSCRIBE_PARAMS["task"] = "transcribe"  # nunca traducir

    font_files = register_fonts(subtitles)

    # Palabras que no caben a lo ancho -> se encogen (fuentes anchas, oneword).
    from clipper_lib import caption_fit
    orig_gen = subtitles.generate_ass

    def generate_ass_fitted(transcript, clip_start, clip_end, output_path, *a, **k):
        out = orig_gen(transcript, clip_start, clip_end, output_path, *a, **k)
        try:
            if cfg.get("captions", {}).get("strip_punctuation", True):
                caption_fit.strip_punct(output_path)
            caption_fit.fit(output_path, font_files, cfg.get("captions", {}).get("max_width", 0.86))
        except Exception as e:
            print(f"   ⚠️ Ajuste de ancho de subtítulos omitido ({e}).")
        return out

    subtitles.generate_ass = generate_ass_fitted

    # Estilo de subtítulos: uno de los presets de OpenShorts + retoques.
    cap = cfg.get("captions", {})
    preset = cap.get("preset", "default")
    if preset in subtitles.CAPTION_PRESETS:
        subtitles.AUTO_CAPTION_STYLE.update(subtitles.CAPTION_PRESETS[preset])
    else:
        print(f"   ⚠️ Preset de subtítulos desconocido: {preset}. "
              f"Opciones: {', '.join(subtitles.CAPTION_PRESETS)}")
    subtitles.AUTO_CAPTION_STYLE.update(cap.get("overrides") or {})

    brb_cfg = cfg.get("dead_air", {})
    orig_transcribe = transcribe_backends.transcribe_media

    def transcribe_plus(path, *a, **k):
        """Transcribe y añade señales gratis: picos de volumen y zonas muertas."""
        result = orig_transcribe(path, *a, **k)
        if peaks_cfg.get("enabled", True):
            try:
                result["audio_peaks"] = audio_signals.find_peaks(path, peaks_cfg)
                print(f"   🔊 {len(result['audio_peaks'])} pico(s) de volumen detectados.")
            except Exception as e:  # una señal extra nunca debe tumbar el trabajo
                print(f"   ⚠️ Picos de volumen omitidos ({e}).")
        if brb_cfg.get("enabled", True):
            try:
                from clipper_lib import brb
                result["dead_zones"] = brb.find_dead_zones(path, result, brb_cfg)
                mins = sum(b - a for a, b in result["dead_zones"]) / 60
                print(f"   💤 {len(result['dead_zones'])} zona(s) muerta(s) (BRB/espera/intro), "
                      f"{mins:.1f} min en total: {result['dead_zones'][:6]}")
            except Exception as e:
                print(f"   ⚠️ Detección de BRB omitida ({e}).")
        return result

    transcribe_backends.transcribe_media = transcribe_plus

    from clipper_lib import sentence_snap
    _TRANSCRIPT = []
    orig_windows = clip_selection.build_transcript_windows

    def windows_hook(transcript_result, *a, **k):
        # Se llama justo antes del snap de cortes: guardamos los segmentos.
        sentence_snap.set_segments(transcript_result)
        _TRANSCRIPT[:] = [transcript_result]
        ws = orig_windows(transcript_result, *a, **k)
        if peaks_cfg.get("enabled", True):
            ws = audio_signals.annotate_windows(ws, transcript_result.get("audio_peaks"))
        zones = transcript_result.get("dead_zones") or []
        _DEAD[:] = zones
        if zones:
            from clipper_lib import brb
            ws = brb.annotate_windows(ws, zones)
        return ws

    clip_selection.build_transcript_windows = windows_hook

    # Red de seguridad: quitar clips que caen sobre todo en zonas muertas.
    _DEAD = []
    orig_dedupe = clip_selection.dedupe_overlapping

    def dedupe_and_drop_dead(shorts, *a, **k):
        cuts = cfg.get("cuts", {})
        if cuts.get("llm_refine", True) and _TRANSCRIPT and shorts:
            from clipper_lib import boundaries
            clips_cfg = cfg.get("clips", {})
            boundaries.refine(shorts, _TRANSCRIPT[0], cuts, clips_cfg.get("min_seconds", 20),
                              cuts.get("hard_max_seconds", clips_cfg.get("max_seconds", 60)))
        out = orig_dedupe(shorts, *a, **k)
        if not _DEAD:
            return out
        from clipper_lib import brb
        keep = [s for s in out if brb.overlap(float(s["start"]), float(s["end"]), _DEAD)
                <= brb_cfg.get("max_clip_overlap", 0.25)]
        if len(keep) < len(out):
            print(f"   💤 Descartados {len(out) - len(keep)} clip(s) sobre pantallas de espera/BRB.")
        return keep or out

    clip_selection.dedupe_overlapping = dedupe_and_drop_dead

    snap_cfg = cfg.get("cuts", {})
    if snap_cfg.get("sentence_snap", True):
        orig_snap = clip_selection.snap_clip_to_words

        def snap_to_sentences(start, end, words, video_duration, *a, **k):
            s, e = orig_snap(start, end, words, video_duration, *a, **k)
            mn = k.get("min_duration", 15.0)
            mx = k.get("max_duration", 60.0)
            ns, ne = sentence_snap.snap(s, e, words, mn, mx, snap_cfg)
            return ns, min(ne, video_duration)

        clip_selection.snap_clip_to_words = snap_to_sentences


def main():
    p = argparse.ArgumentParser(description="Clips verticales 9:16 desde videos largos.")
    src = p.add_mutually_exclusive_group(required=True)
    src.add_argument("--url", help="URL de YouTube o Twitch (VOD o clip).")
    src.add_argument("--input", help="Archivo de video local.")
    p.add_argument("-o", "--output", help="Carpeta de salida (por defecto output/<nombre>).")
    p.add_argument("--config", default=os.path.join(ROOT, "config.yaml"))
    p.add_argument("--start", help="Solo con --url: inicio del tramo (p. ej. 1:30:00).")
    p.add_argument("--minutes", type=float,
                   help="Solo con --url: minutos a descargar desde --start.")
    p.add_argument("--layout", default=None, choices=["auto", "gaming", "chatting", "podcast", "split"],
                   help="Forzar layout (por defecto: el de config.yaml, normalmente auto).")
    p.add_argument("--campaign", help="Reglas de una campaña: campanas/<nombre>.yaml")
    args = p.parse_args()

    # La consola de Windows usa cp1252 y no sabe imprimir emojis: forzamos UTF-8.
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):
            pass

    cfg = load_config(args.config)
    if args.campaign:
        merge_campaign(cfg, args.campaign)
    apply_config(cfg)
    _install_windows_shims()
    sys.path.insert(0, VENDOR)
    sys.path.insert(0, ROOT)
    _install_hooks(cfg)

    if args.url:
        # Descarga propia (no la de OpenShorts) para poder bajar solo un tramo.
        from clipper_lib import download
        video, _title = download.download(
            args.url, os.path.join(ROOT, "downloads"),
            start=download.parse_time(args.start), minutes=args.minutes)
    else:
        video = os.path.abspath(args.input)
    name = os.path.splitext(os.path.basename(video))[0]
    out = os.path.abspath(args.output or os.path.join(ROOT, "output", name))
    os.makedirs(out, exist_ok=True)
    os.makedirs(os.environ["ASR_LOCK_DIR"], exist_ok=True)

    from clipper_lib import layout
    lcfg = cfg.get("layout", {})
    print("🔎 Analizando el video (webcam, zona de juego, caras)...")
    analysis = layout.analyze_and_save(video, out, lcfg)
    chosen, why = layout.decide(analysis, args.layout or lcfg.get("mode", "auto"), lcfg)
    print(f"🧩 Layout: {chosen} — {why}")
    if analysis.get("webcam"):
        print(f"   Webcam: {analysis['webcam']['rect']} | juego: {analysis['game']['rect']}")
    layout.install(chosen, analysis, lcfg)

    sys.argv = ["main.py", "-i", video, "-o", out]
    runpy.run_path(os.path.join(VENDOR, "main.py"), run_name="__main__")

    from clipper_lib import organize
    organize.organize(out, cfg)


if __name__ == "__main__":
    main()

"""Claude Clipper — interfaz en el navegador (solo en tu PC).

Arrancar: doble clic en Clipper.bat (o: .venv\\Scripts\\python.exe app.py).
Se abre http://127.0.0.1:7860 en tu navegador. Nada sale de tu PC salvo el
texto que se manda a DeepSeek para elegir los momentos.
"""
import copy
import glob
import json
import os
import re
import shutil
import subprocess
import sys
import time

import gradio as gr
import yaml

ROOT = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
OUT = os.path.join(ROOT, "output")
INBOX = os.path.join(ROOT, "entrada")
DONE = os.path.join(INBOX, "hechos")
VIDEO_EXT = (".mp4", ".mkv", ".mov", ".webm", ".avi", ".flv", ".ts")

# Líneas del log que se enseñan (el resto es ruido técnico) y su avance aproximado.
STEPS = [
    ("✂️", "Descargando", 0.02), ("📥", "Descargando", 0.02),
    ("🔎", "Analizando la imagen (webcam, juego, caras)", 0.06),
    ("🧩", None, 0.10),
    ("🎙️  Transcribing", "Transcribiendo (whisper)", 0.12),
    ("💤", None, 0.35), ("🔊", None, 0.33),
    ("🤖  Analyzing", "DeepSeek está eligiendo los mejores momentos", 0.40),
    ("🔥 Found", None, 0.50),
    ("🎬 Processing Clip", None, None),
    ("😀", None, 0.95),
    ("📁", None, 1.0),
]


def fonts():
    from fontTools.ttLib import TTFont
    names = set()
    for d in (os.path.join(ROOT, "assets", "fonts"), os.path.join(ROOT, "vendor", "openshorts", "fonts")):
        for f in glob.glob(os.path.join(d, "*.ttf")):
            try:
                names.add(TTFont(f)["name"].getDebugName(1))
            except Exception:
                pass
    return sorted(names)


def base_config():
    with open(os.path.join(ROOT, "config.yaml"), encoding="utf-8") as f:
        return yaml.safe_load(f)


def write_config(font, emojis, lang, min_s, max_s, layout_mode, game_view="full",
                 lol_icons=True, sfx=True):
    """Copia de config.yaml con lo elegido en la interfaz (config.yaml no se toca)."""
    cfg = copy.deepcopy(base_config())
    cfg.setdefault("captions", {}).setdefault("overrides", {})["font_name"] = font
    cfg.setdefault("emojis", {})["enabled"] = bool(emojis)
    cfg["emojis"]["sfx_enabled"] = bool(sfx)
    cfg.setdefault("lol_icons", {})["enabled"] = bool(lol_icons)
    cfg.setdefault("transcription", {})["language"] = lang
    cfg.setdefault("clips", {}).update({"min_seconds": int(min_s), "max_seconds": int(max_s)})
    cfg.setdefault("layout", {})["mode"] = layout_mode
    cfg["layout"]["game_view"] = game_view
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, ".app_config.yaml")
    with open(path, "w", encoding="utf-8") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    return path


def _slug(text):
    text = re.sub(r"[^\w\- ]+", "", text, flags=re.UNICODE).strip()
    return re.sub(r"\s+", "_", text)[:60] or "video"


def run_one(source, is_url, cfg_path, start, minutes, state):
    """Procesa un vídeo con clipper.py y va actualizando ``state`` con el progreso."""
    if is_url:
        out = os.path.join(OUT, f"url_{time.strftime('%Y%m%d_%H%M%S')}")
        cmd = [PY, "-u", "clipper.py", "--url", source, "-o", out, "--config", cfg_path]
        if start:
            cmd += ["--start", str(start)]
        if minutes:
            cmd += ["--minutes", str(minutes)]
    else:
        name = _slug(os.path.splitext(os.path.basename(source))[0])
        out = os.path.join(OUT, name)
        cmd = [PY, "-u", "clipper.py", "--input", source, "-o", out, "--config", cfg_path]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env=env, text=True, encoding="utf-8", errors="replace")
    n_clips, done_clips = 0, 0
    for line in proc.stdout:
        line = line.rstrip()
        for key, label, frac in STEPS:
            if key in line:
                if key == "🔥 Found":
                    m = re.search(r"Found (\d+)", line)
                    n_clips = int(m.group(1)) if m else 0
                if key == "🎬 Processing Clip":
                    done_clips += 1
                    frac = 0.5 + 0.45 * done_clips / max(1, n_clips)
                    label = f"Montando clip {done_clips} de {n_clips}"
                if key == "🎙️  Transcribing":
                    pass
                state["log"].append(label or line.strip())
                if frac is not None:
                    state["frac"] = frac
                break
        else:
            if "Transcribing…" in line:
                m = re.search(r"(\d+)%", line)
                if m:
                    state["frac"] = 0.12 + 0.2 * int(m.group(1)) / 100
                    state["log"][-1:] = [f"Transcribiendo… {m.group(1)}%"]
            elif "Traceback" in line or "Error" in line or "❌" in line:
                state["log"].append("⚠️ " + line.strip()[:200])
    proc.wait()
    state["out"] = out
    return proc.returncode, out


def clips_of(out_dir):
    """[(ruta_mp4, "score · título")] de una carpeta de resultados."""
    p = os.path.join(out_dir, "clips.json")
    if not os.path.exists(p):
        return []
    with open(p, encoding="utf-8") as f:
        data = json.load(f)
    items = []
    for c in data.get("clips", []):
        f = os.path.join(out_dir, c.get("file") or "")
        if c.get("file") and os.path.exists(f):
            items.append((f, f"{c.get('score', '?')} · {c.get('title', '')}"))
    return items


def process(files, urls, start, minutes, layout_mode, font, emojis, lang, min_s, max_s,
            game_view="full", lol_icons=True, sfx=True):
    jobs = [(f if isinstance(f, str) else f.name, False) for f in (files or [])]
    jobs += [(u.strip(), True) for u in (urls or "").splitlines() if u.strip().startswith("http")]
    if not jobs:
        yield "Añade algún vídeo o URL.", [], []
        return
    if not os.environ.get("DEEPSEEK_API_KEY") and not os.path.exists(os.path.join(ROOT, ".env")):
        yield "Falta la clave de DeepSeek: crea el archivo .env (ver README).", [], []
        return
    cfg_path = write_config(font, emojis, lang, min_s, max_s, layout_mode, game_view, lol_icons, sfx)
    all_clips, files_out, lines = [], [], []
    for k, (src, is_url) in enumerate(jobs, start=1):
        state = {"log": [], "frac": 0.0, "out": None}
        head = f"Vídeo {k} de {len(jobs)}: {os.path.basename(src) if not is_url else src}"
        import threading
        t = threading.Thread(target=run_one, args=(src, is_url, cfg_path, start, minutes, state), daemon=True)
        t.start()
        t0 = time.time()
        while t.is_alive():
            mins = (time.time() - t0) / 60
            bar = "█" * int(state["frac"] * 20) + "░" * (20 - int(state["frac"] * 20))
            yield ("\n".join(lines + [head, f"{bar} {state['frac']:.0%}  ·  {mins:.1f} min"] +
                             ["   " + s for s in state["log"][-8:]]), all_clips, files_out)
            time.sleep(1.5)
        new = clips_of(state["out"]) if state["out"] else []
        all_clips += new
        files_out += [c[0] for c in new]
        lines.append(f"{head}  →  {len(new)} clip(s) en {state['out']}" if new
                     else f"{head}  →  ⚠️ sin clips (mira {state['out']})")
        yield "\n".join(lines), all_clips, files_out
    lines.append("\n✅ Terminado.")
    yield "\n".join(lines), all_clips, files_out


def process_inbox(*opts):
    """Procesa todo lo que haya en entrada/ (vídeos y entrada/urls.txt)."""
    os.makedirs(DONE, exist_ok=True)
    vids = sorted(f for f in glob.glob(os.path.join(INBOX, "*")) if f.lower().endswith(VIDEO_EXT))
    urls_file = os.path.join(INBOX, "urls.txt")
    urls = ""
    if os.path.exists(urls_file):
        with open(urls_file, encoding="utf-8") as f:
            urls = f.read()
    if not vids and not urls.strip():
        yield f"La carpeta {INBOX} está vacía. Mete vídeos (o un urls.txt) y vuelve a pulsar.", [], []
        return
    last = None
    for last in process(vids, urls, None, None, *opts):
        yield last
    # Mover lo procesado a entrada/hechos para no repetirlo.
    for v in vids:
        shutil.move(v, os.path.join(DONE, os.path.basename(v)))
    if urls.strip():
        shutil.move(urls_file, os.path.join(DONE, f"urls_{time.strftime('%Y%m%d_%H%M%S')}.txt"))


def list_results():
    dirs = sorted((d for d in glob.glob(os.path.join(OUT, "*")) if os.path.exists(os.path.join(d, "clips.json"))),
                  key=os.path.getmtime, reverse=True)
    return [os.path.basename(d) for d in dirs]


def show_results(name):
    if not name:
        return [], []
    items = clips_of(os.path.join(OUT, name))
    return items, [i[0] for i in items]


def build():
    cfg = base_config()
    font_list = fonts()
    cur_font = (cfg.get("captions", {}).get("overrides") or {}).get("font_name", "Luckiest Guy")
    with gr.Blocks(title="Claude Clipper", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🎬 Claude Clipper\nVídeos largos → clips verticales con subtítulos. "
                    "Todo en tu PC; solo el texto va a DeepSeek.")
        with gr.Tab("Crear clips"):
            with gr.Row():
                with gr.Column(scale=3):
                    files = gr.File(label="Arrastra aquí tus vídeos", file_count="multiple",
                                    file_types=list(VIDEO_EXT))
                    urls = gr.Textbox(label="…o pega URLs de YouTube / Twitch (una por línea)", lines=3,
                                      placeholder="https://www.twitch.tv/videos/1234567890")
                    with gr.Row():
                        start = gr.Textbox(label="Empezar en (solo URLs, opcional)", placeholder="1:30:00")
                        minutes = gr.Number(label="Minutos a bajar (solo URLs, opcional)", value=None)
                with gr.Column(scale=2):
                    layout_mode = gr.Dropdown([("auto", "auto"), ("gaming", "gaming"), ("chatting", "chatting"), ("podcast", "podcast"),
                                               ("pantalla partida", "split")], value="auto",
                                              label="Tipo de vídeo",
                                              info="auto lo detecta solo (webcam + juego, charla, podcast, pantalla partida)")
                    game_view = gr.Radio([("Pantalla entera", "full"), ("Recortada al centro", "crop")],
                                         value=cfg.get("layout", {}).get("game_view", "full"),
                                         label="Juego / contenido (debajo de la webcam)")
                    font = gr.Dropdown(font_list, value=cur_font if cur_font in font_list else None,
                                       label="Fuente de los subtítulos")
                    with gr.Row():
                        emojis = gr.Checkbox(value=cfg.get("emojis", {}).get("enabled", True), label="Emojis")
                        lol_icons = gr.Checkbox(value=cfg.get("lol_icons", {}).get("enabled", True),
                                                label="Iconos del LoL")
                        sfx = gr.Checkbox(value=cfg.get("emojis", {}).get("sfx_enabled", True),
                                          label="Sonidos meme")
                    lang = gr.Dropdown(["es", "en", "auto"],
                                       value=str(cfg.get("transcription", {}).get("language", "es")),
                                       label="Idioma del que habla")
                    with gr.Row():
                        min_s = gr.Slider(10, 60, value=cfg.get("clips", {}).get("min_seconds", 20), step=5,
                                          label="Duración mínima (s)")
                        max_s = gr.Slider(20, 90, value=cfg.get("clips", {}).get("max_seconds", 60), step=5,
                                          label="Duración máxima (s)")
            with gr.Row():
                go = gr.Button("✨ Crear clips", variant="primary", scale=3)
                inbox = gr.Button(f"📂 Procesar la carpeta entrada/", scale=2)
            status = gr.Textbox(label="Progreso", lines=12, interactive=False)
            gallery = gr.Gallery(label="Clips", columns=4, height="auto", object_fit="contain")
            downloads = gr.File(label="Descargar", file_count="multiple", interactive=False)
            opts = [layout_mode, font, emojis, lang, min_s, max_s, game_view, lol_icons, sfx]
            go.click(process, [files, urls, start, minutes] + opts, [status, gallery, downloads])
            inbox.click(process_inbox, opts, [status, gallery, downloads])
        with gr.Tab("Mis clips") as mis_clips:
            with gr.Row():
                pick = gr.Dropdown(list_results(), value=None, label="Vídeo procesado", scale=4)
                refresh = gr.Button("🔄", scale=1)
            g2 = gr.Gallery(label="Clips", columns=4, height="auto", object_fit="contain")
            d2 = gr.File(label="Descargar", file_count="multiple", interactive=False)
            pick.change(show_results, pick, [g2, d2])
            refresh.click(lambda: gr.update(choices=list_results()), None, pick)
            mis_clips.select(lambda: gr.update(choices=list_results()), None, pick)
        gr.Markdown(f"Carpeta de resultados: `{OUT}` · Carpeta de entrada para lotes: `{INBOX}`")
    return demo


if __name__ == "__main__":
    os.makedirs(INBOX, exist_ok=True)
    build().queue().launch(server_name="127.0.0.1", server_port=7860, inbrowser=True,
                           allowed_paths=[OUT], show_api=False)

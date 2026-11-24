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
                 lol_icons=True, sfx=True, color=None):
    """Copia de config.yaml con lo elegido en la interfaz (config.yaml no se toca)."""
    cfg = copy.deepcopy(base_config())
    cfg.setdefault("captions", {}).setdefault("overrides", {})["font_name"] = font
    if color:
        cfg["captions"]["overrides"]["font_color"] = _hex(color)
        cfg["captions"]["overrides"]["highlight_color"] = _hex(color)
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


def _hex(color):
    """'#ffc93c' o 'rgba(255, 201, 60, 1)' (según la versión de Gradio) -> '#FFC93C'."""
    c = str(color or "").strip()
    m = re.match(r"rgba?\(\s*([\d.]+)\s*,\s*([\d.]+)\s*,\s*([\d.]+)", c)
    if m:
        return "#" + "".join(f"{round(float(v)):02X}" for v in m.groups())
    c = c.lstrip("#")
    return "#" + c[:6].upper() if re.fullmatch(r"[0-9a-fA-F]{6,8}", c) else "#FFFFFF"


def _preview_background():
    """Un fotograma de un clip ya hecho (sin subtítulos) como fondo; si no hay
    ninguno, un degradado oscuro."""
    bg = os.path.join(OUT, ".preview_bg.png")
    if os.path.exists(bg):
        return bg
    os.makedirs(OUT, exist_ok=True)
    raws = sorted(glob.glob(os.path.join(OUT, "*", "debug", "*_clip_1.mp4")), key=os.path.getmtime)
    cmd = (["ffmpeg", "-v", "error", "-y", "-ss", "8", "-i", raws[-1]] if raws else
           ["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "gradients=s=1080x1920:c0=0x202638:c1=0x4a3a5a"])
    subprocess.run(cmd + ["-frames:v", "1", "-vf", "scale=1080:1920", bg], check=False)
    return bg


def caption_preview(font, color, word="BRUTAL"):
    """Imagen 1080x1920 (reducida) con una palabra en la fuente y el color
    elegidos, quemada con libass igual que los subtítulos de verdad."""
    cfg = base_config()
    ov = (cfg.get("captions") or {}).get("overrides") or {}
    size = ov.get("font_size", 25)
    border = ov.get("border_width", 3)
    h = _hex(color).lstrip("#")
    ass_col = f"&H00{h[4:6]}{h[2:4]}{h[0:2]}"
    fonts_dir = os.path.join(ROOT, "vendor", "openshorts", "fonts")
    for f in glob.glob(os.path.join(ROOT, "assets", "fonts", "*.ttf")):
        if not os.path.exists(os.path.join(fonts_dir, os.path.basename(f))):
            shutil.copy2(f, fonts_dir)
    word = (word or "BRUTAL").strip().upper()[:20] or "BRUTAL"
    ass = ("[Script Info]\nScriptType: v4.00+\nPlayResY: 288\nScaledBorderAndShadow: yes\n\n"
           "[V4+ Styles]\nFormat: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, "
           "BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, "
           "Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding\n"
           f"Style: Default,{font},{size},{ass_col},{ass_col},&H00000000,&HFF000000,1,0,0,0,100,100,0,0,1,"
           f"{border},0,2,10,10,43,1\n\n"
           "[Events]\nFormat: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text\n"
           f"Dialogue: 0,0:00:00.00,0:00:05.00,Default,,0,0,0,,{{\\an5\\pos(192,157.5)\\fscx108\\fscy108}}{word}\n")
    tmp = os.path.join(OUT, ".preview")
    os.makedirs(tmp, exist_ok=True)
    with open(os.path.join(tmp, "p.ass"), "w", encoding="utf-8") as f:
        f.write(ass)
    out = os.path.join(tmp, f"preview_{int(time.time() * 1000)}.png")
    for old in glob.glob(os.path.join(tmp, "preview_*.png")):
        try:
            os.remove(old)
        except OSError:
            pass
    # Rutas relativas: el filtro ass de ffmpeg se lía con "C:" y los espacios.
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-i", os.path.relpath(_preview_background(), tmp),
                    "-vf", "ass=p.ass:fontsdir=" + os.path.relpath(fonts_dir, tmp).replace("\\", "/") +
                    ",scale=405:720", "-frames:v", "1", os.path.basename(out)], cwd=tmp, check=False)
    return out if os.path.exists(out) else None


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
            game_view="full", lol_icons=True, sfx=True, color=None):
    jobs = [(f if isinstance(f, str) else f.name, False) for f in (files or [])]
    jobs += [(u.strip(), True) for u in (urls or "").splitlines() if u.strip().startswith("http")]
    if not jobs:
        yield "Añade algún vídeo o URL.", [], []
        return
    if not os.environ.get("DEEPSEEK_API_KEY") and not os.path.exists(os.path.join(ROOT, ".env")):
        yield "Falta la clave de DeepSeek: crea el archivo .env (ver README).", [], []
        return
    cfg_path = write_config(font, emojis, lang, min_s, max_s, layout_mode, game_view, lol_icons, sfx, color)
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


def only_download(files, urls, start, minutes, tramo, to_inbox):
    """Solo descarga (y trocea) con tools/descargar.py, sin sacar clips."""
    srcs = [f if isinstance(f, str) else f.name for f in (files or [])]
    srcs += [u.strip() for u in (urls or "").splitlines() if u.strip().startswith("http")]
    keep = (gr.update(), None, gr.update())
    if not srcs:
        yield ("Pega alguna URL (o arrastra un vídeo para solo trocearlo).",) + keep
        return
    cmd = [PY, "-u", os.path.join("tools", "descargar.py"), *srcs, "--tramo", str(float(tramo or 0))]
    if to_inbox:
        cmd.append("--entrada")
    if start:
        cmd += ["--start", str(start)]
    if minutes:
        cmd += ["--minutes", str(minutes)]
    env = {**os.environ, "PYTHONIOENCODING": "utf-8", "PYTHONUNBUFFERED": "1"}
    proc = subprocess.Popen(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                            env=env, text=True, encoding="utf-8", errors="replace")
    lines, last, results = [], 0.0, []  # results: [(carpeta, [archivos])] por vídeo
    for line in proc.stdout:
        line = line.rstrip()
        mt = re.match(r"\s*📦 Tramo \d+ de \d+: (.+)$", line)
        if mt:
            results.append((None, [mt.group(1)])) if not results or results[-1][0] else \
                results[-1][1].append(mt.group(1))
        mf = re.match(r"📁 Tramos en (.+)$", line)
        if mf and results and results[-1][0] is None:
            results[-1] = (mf.group(1), results[-1][1])
        mo = re.match(r"📁 (?:En|Copiado a) (.+?) \(no hace falta trocear\)$", line)
        if mo:
            results.append((os.path.dirname(mo.group(1)), [os.path.basename(mo.group(1))]))
        m = re.search(r"\[download\]\s+([\d.]+)%", line)
        if m:
            # Barra de yt-dlp: una sola línea que se va actualizando.
            txt = f"   Descargando… {float(m.group(1)):.0f}%"
            if lines and lines[-1].startswith("   Descargando…"):
                lines[-1] = txt
            else:
                lines.append(txt)
        elif line and (line[0] in "🎬✅✂📦📁❌📥" or line.startswith("   📦")):
            lines.append(line)
        else:
            continue
        if time.time() - last > 0.5:
            last = time.time()
            yield ("\n".join(lines),) + keep
    proc.wait()
    tail = ("\n\nListo. Pulsa \"📂 Procesar la carpeta entrada/\" para sacar los clips de todos los tramos."
            if to_inbox else "\n\nListo.")
    # Resumen: cada tramo con su duración y tamaño, y dónde está.
    md, paths, folder = [], [], None
    for d, names in results:
        if not d:
            continue
        folder = d
        md.append(f"**📁 Carpeta:** `{d}`\n\n| Tramo | Duración | Tamaño |\n|---|---|---|")
        for n in names:
            p = os.path.join(d, n)
            if os.path.exists(p):
                paths.append(p)
                md.append(f"| {n} | {_dur_text(p)} | {os.path.getsize(p) / 2**20:,.0f} MB |")
        md.append("")
    summary = "\n".join(md) or "No se ha generado ningún archivo (mira el progreso)."
    yield ("\n".join(lines) + tail, summary, folder, [_thumb(p) for p in paths])


def _thumb(path):
    """(miniatura, pie) de un tramo: un fotograma a los 60 s, en output/.thumbs."""
    tdir = os.path.join(OUT, ".thumbs")
    os.makedirs(tdir, exist_ok=True)
    img = os.path.join(tdir, os.path.splitext(os.path.basename(path))[0] + ".jpg")
    if not os.path.exists(img):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-ss", "60", "-i", path, "-frames:v", "1",
                        "-vf", "scale=480:-2", img], check=False)
    return (img, f"{os.path.basename(path)} · {_dur_text(path)}")


def _dur_text(path):
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0", path],
                         capture_output=True, text=True)
    try:
        s = int(float(out.stdout.strip()))
    except ValueError:
        return "?"
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def _bring_to_front(title):
    """Pone delante la ventana del Explorador de esa carpeta. Windows no deja
    que un programa en segundo plano robe el foco, así que se abre detrás del
    navegador; ponerla "siempre visible" un instante sí la trae delante."""
    import ctypes
    from ctypes import wintypes
    u = ctypes.WinDLL("user32", use_last_error=True)
    proto = ctypes.WINFUNCTYPE(wintypes.BOOL, wintypes.HWND, wintypes.LPARAM)
    u.SetWindowPos.argtypes = [wintypes.HWND, wintypes.HWND] + [ctypes.c_int] * 4 + [wintypes.UINT]
    found = []

    @proto
    def visit(hwnd, _):
        cls, txt = ctypes.create_unicode_buffer(64), ctypes.create_unicode_buffer(512)
        u.GetClassNameW(hwnd, cls, 64)
        u.GetWindowTextW(hwnd, txt, 512)
        if cls.value == "CabinetWClass" and title.lower() in txt.value.lower():
            found.append(hwnd)
        return True

    for _ in range(20):  # la ventana tarda un poco en aparecer
        u.EnumWindows(visit, 0)
        if found:
            break
        time.sleep(0.2)
    for hwnd in found[:1]:
        if u.IsIconic(hwnd):
            u.ShowWindow(hwnd, 9)  # restaurar si estaba minimizada
        for order in (-1, -2):  # siempre visible -> normal
            u.SetWindowPos(hwnd, order, 0, 0, 0, 0, 0x0001 | 0x0002 | 0x0040)
    return bool(found)


def open_folder(folder):
    """Abre la carpeta de los tramos en el Explorador (la app corre en tu PC)."""
    # Al recargar la página se pierde cuál era: entonces se abre entrada/.
    folder = folder if folder and os.path.isdir(folder) else INBOX
    try:
        os.makedirs(folder, exist_ok=True)
        os.startfile(folder)
        _bring_to_front(os.path.basename(os.path.normpath(folder)))
    except Exception as e:
        gr.Warning(f"No se pudo abrir {folder}: {e}")
        return
    gr.Info(f"Carpeta abierta: {folder}")


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
                    cur_color = (cfg.get("captions", {}).get("overrides") or {}).get("font_color", "#FFFFFF")
                    with gr.Row():
                        with gr.Column(scale=3, min_width=160):
                            font = gr.Dropdown(font_list, value=cur_font if cur_font in font_list else None,
                                               label="Fuente de los subtítulos")
                            color = gr.ColorPicker(value=cur_color, label="Color de la letra")
                            sample = gr.Textbox(value="BRUTAL", label="Palabra de prueba", max_lines=1)
                        preview = gr.Image(value=caption_preview(cur_font, cur_color), label="Vista previa",
                                           show_label=True, interactive=False, height=360,
                                           show_download_button=False, scale=2, min_width=150)
                    for ctl in (font, color, sample):
                        ctl.change(caption_preview, [font, color, sample], preview)
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
            with gr.Accordion("📥 Solo descargar (y trocear un directo largo)", open=False):
                gr.Markdown("Baja las URLs de arriba (respeta \"Empezar en\" y \"Minutos\") sin sacar clips, "
                            "y las parte en tramos. Un vídeo arrastrado arriba también se puede trocear.")
                with gr.Row():
                    tramo = gr.Number(value=30, label="Trocear cada (minutos, 0 = no trocear)", minimum=0, scale=2)
                    to_inbox = gr.Checkbox(value=True, label="Dejar los tramos en entrada/ para procesarlos luego",
                                           scale=3)
                    dl_only = gr.Button("📥 Solo descargar", scale=2)
                dl_summary = gr.Markdown()
                dl_folder = gr.State(None)
                open_btn = gr.Button("📂 Abrir carpeta de los tramos")
                # Miniaturas (no el vídeo: Gradio recodificaría cada tramo para el navegador).
                dl_pick = gr.Gallery(label="Tramos", columns=4, height="auto", object_fit="contain")
                open_btn.click(open_folder, dl_folder, None)
            status = gr.Textbox(label="Progreso", lines=12, interactive=False)
            gallery = gr.Gallery(label="Clips", columns=4, height="auto", object_fit="contain")
            downloads = gr.File(label="Descargar", file_count="multiple", interactive=False)
            opts = [layout_mode, font, emojis, lang, min_s, max_s, game_view, lol_icons, sfx, color]
            go.click(process, [files, urls, start, minutes] + opts, [status, gallery, downloads])
            inbox.click(process_inbox, opts, [status, gallery, downloads])
            dl_only.click(only_download, [files, urls, start, minutes, tramo, to_inbox],
                          [status, dl_summary, dl_folder, dl_pick])
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
    build().queue().launch(server_name="127.0.0.1", server_port=int(os.environ.get("CLIPPER_PORT", 7860)),
                           inbrowser=not os.environ.get("CLIPPER_NO_BROWSER"),
                           allowed_paths=[OUT, INBOX, os.path.join(ROOT, "downloads")], show_api=False)

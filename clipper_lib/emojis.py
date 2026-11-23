"""Emojis animados encima del subtítulo, elegidos por DeepSeek.

1. UNA llamada a DeepSeek para todos los clips del video (pocos tokens): para
   cada clip le pasamos sus frases con el segundo en que empiezan y devuelve
   2-4 momentos con el emoji que mejor los refuerza.
2. Las imágenes son los emojis 3D de Noto (Google, github.com/googlefonts/noto-emoji,
   licencia Apache 2.0). Se descargan la primera vez y quedan en assets/emoji.
3. ffmpeg los superpone centrados encima del subtítulo, con un "pop" de entrada
   y un fundido de salida.
"""
import json
import re
import os
import subprocess

import httpx

NOTO_URL = ("https://raw.githubusercontent.com/googlefonts/noto-emoji/main/"
            "{style}/png/{size}/emoji_u{code}.png")

PROMPT = """You add emojis to short vertical videos (TikTok style).
For each clip below, pick {min_n}-{max_n} key moments (reactions, jokes, surprise,
anger, hype, fails) and ONE emoji each that a viewer would instantly get.
Use the "t" of the phrase where the moment happens. Spread them out (>= 4 s apart).
Prefer expressive faces and objects: 😂 😱 💀 🔥 😡 🤯 😭 👀 🤔 💯 😳 🫠 ⚡ 🎉 😎 🤡 🙏 👏 ❤️.
{sfx_rules}Return only json: {{"clips": [{{"id": 1, "emojis": [{{"t": 3.2, "e": "😂"{sfx_example}}}]}}]}}

CLIPS:
{clips}
"""


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SFX_DIR = os.path.join(ROOT, "assets", "sfx")
# Sonido -> cuándo usarlo (se lo explicamos a DeepSeek en una línea).
SFX_MEANING = {
    "boom": "punchline, shocking statement or dramatic reveal",
    "faah": "fail, death, big mistake or embarrassing moment",
    "rizz": "flirting, mentions of a girl/boyfriend/girlfriend or romance",
    "bruh": "something dumb, absurd or disappointing; a facepalm moment",
    "gunshot": "a brutal roast, a savage comeback or someone getting destroyed in an argument",
}


def _sfx_key(name):
    """Clave del sonido a partir del nombre del archivo. Los conocidos se
    reconocen por parte del nombre ("vine-boom" -> boom, "faaah" -> faah); el
    resto usa su propio nombre y DeepSeek decide por él cuándo encaja."""
    low = name.lower()
    for key in SFX_MEANING:
        if key in low:
            return key
    if re.search(r"fa+h", low):
        return "faah"
    return re.sub(r"[^a-z0-9]+", "-", low).strip("-") or None


def sfx_meaning(key):
    return SFX_MEANING.get(key, f'meme sound "{key}": use it only where that sound clearly fits')


def available_sfx():
    """{"boom": ruta, ...} con los sonidos que el usuario ha puesto en assets/sfx/."""
    out = {}
    if os.path.isdir(SFX_DIR):
        for f in os.listdir(SFX_DIR):
            name, ext = os.path.splitext(f)
            if ext.lower() not in (".mp3", ".wav", ".ogg", ".m4a"):
                continue
            key = _sfx_key(name)
            if key and key not in out:
                out[key] = os.path.join(SFX_DIR, f)
    return out


def _phrases(transcript, start, end, max_chars=900):
    """Frases del clip como '[t] texto' con t relativo al inicio del clip."""
    out, total = [], 0
    for seg in transcript.get("segments", []):
        if seg["end"] <= start or seg["start"] >= end:
            continue
        line = f"[{max(0.0, seg['start'] - start):.1f}] {seg['text'].strip()}"
        total += len(line)
        if total > max_chars:
            break
        out.append(line)
    return "\n".join(out)


def plan(meta, cfg):
    """Devuelve {n_clip: [{"t": s, "e": "😂"}]} pidiendo a DeepSeek una sola vez."""
    shorts = meta.get("shorts", [])
    if not shorts:
        return {}
    blocks = []
    for i, c in enumerate(shorts, start=1):
        blocks.append(f"# clip {i} ({c['end'] - c['start']:.0f}s)\n"
                      f"{_phrases(meta['transcript'], c['start'], c['end'])}")
    sfx = available_sfx() if cfg.get("sfx_enabled", True) else {}
    max_sfx = int(cfg.get("max_sfx_per_clip", 2))
    if sfx and max_sfx > 0:
        rules = (f'Also add "s" (a meme sound) to AT MOST {max_sfx} of those moments per clip, '
                 "only when it clearly fits; otherwise leave it out:\n" +
                 "".join(f'- "{k}": {sfx_meaning(k)}\n' for k in sfx))
        example = ', "s": "' + next(iter(sfx)) + '"'
    else:
        rules, example = "", ""
    prompt = PROMPT.format(min_n=cfg.get("min_per_clip", 2), max_n=cfg.get("max_per_clip", 4),
                           clips="\n\n".join(blocks), sfx_rules=rules, sfx_example=example)
    body = {"model": os.environ["LLM_MODEL"], "temperature": 0.4, "stream": False,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": prompt}]}
    body.update(json.loads(os.environ.get("LLM_EXTRA_BODY") or "{}"))
    headers = {"Authorization": f"Bearer {os.environ['LLM_API_KEY']}"}
    url = os.environ["LLM_BASE_URL"].rstrip("/") + "/chat/completions"
    data = None
    for _ in range(3):
        r = httpx.post(url, json=body, headers=headers, timeout=120)
        r.raise_for_status()
        resp = r.json()
        text = resp["choices"][0]["message"].get("content") or ""
        try:
            data = json.loads(text)
            break
        except ValueError:
            continue  # DeepSeek a veces devuelve contenido vacío: reintentar
    if data is None:
        raise RuntimeError("DeepSeek no devolvió JSON válido para los emojis")
    usage = resp.get("usage", {})
    print(f"   😀 Emojis elegidos por DeepSeek ({usage.get('prompt_tokens', '?')} tokens "
          f"de entrada, {usage.get('completion_tokens', '?')} de salida).")
    result = {}
    for c in data.get("clips", []):
        try:
            n = int(c["id"])
        except (KeyError, ValueError, TypeError):
            continue
        dur = shorts[n - 1]["end"] - shorts[n - 1]["start"] if 0 < n <= len(shorts) else 0
        items = []
        for e in c.get("emojis", []):
            try:
                t = float(e["t"])
            except (KeyError, ValueError, TypeError):
                continue
            if 0 <= t < dur - 0.5 and isinstance(e.get("e"), str) and e["e"].strip():
                item = {"t": round(t, 2), "e": e["e"].strip()}
                snd = str(e.get("s") or "").strip().lower()
                if snd in sfx and sum(1 for x in items if x.get("s")) < max_sfx:
                    item["s"] = snd
                items.append(item)
        result[n] = items[: cfg.get("max_per_clip", 4)]
    return result


def _codepoints(emoji):
    """'❤️' -> '2764' (sin el selector de variación FE0F, como nombra Noto)."""
    cps = [f"{ord(ch):x}" for ch in emoji if ord(ch) != 0xFE0F]
    return "_".join(cps)


def image_for(emoji, cache_dir, style="3D", size=512):
    """Ruta al PNG del emoji (lo descarga de Noto la primera vez) o None."""
    code = _codepoints(emoji)
    os.makedirs(cache_dir, exist_ok=True)
    path = os.path.join(cache_dir, f"{style}_{size}_{code}.png")
    if os.path.exists(path):
        return path
    for st in (style, "2D"):
        url = NOTO_URL.format(style=st, size=size, code=code)
        r = httpx.get(url, timeout=30, follow_redirects=True)
        if r.status_code == 200 and r.content[:4] == b"\x89PNG":
            with open(path, "wb") as f:
                f.write(r.content)
            return path
    return None


def overlay(video_in, video_out, items, cache_dir, y_center, cfg, x_center=None):
    """Superpone emojis / iconos y mezcla los sonidos meme en ``video_in``.

    Cada item: {"t", "e" (emoji) o "png" (icono, p. ej. del LoL), "s" (sonido,
    opcional), "x"/"y" (opcionales)}. ``y_center``/``x_center``: centro por
    defecto en el frame 1080x1920 (x: centrado)."""
    def xexpr(it):
        x = it.get("x", x_center)
        return "(W-w)/2" if x is None else f"{int(x)}-w/2"
    size = int(cfg.get("size", 230))
    show = float(cfg.get("duration", 1.4))
    sfx_files = available_sfx()
    inputs, chains, last = [], [], "[0:v]"
    k = 0
    sounds = []
    for it in items:
        if it.get("s") in sfx_files:
            sounds.append((it["t"], sfx_files[it["s"]]))
        png = it.get("png") or (image_for(it["e"], cache_dir, cfg.get("style", "3D")) if it.get("e") else None)
        if not png:
            if it.get("e"):
                print(f"   ⚠️ Emoji sin imagen en Noto: {it['e']}")
            continue
        k += 1
        t0 = it["t"]
        inputs += ["-loop", "1", "-t", f"{show:.2f}", "-i", png]
        # Los iconos del LoL son pequeños (64 px): algo menos grandes que un
        # emoji, reescalados con lanczos y con un borde oscuro fino.
        is_icon = bool(it.get("png"))
        sz = int(size * cfg.get("icon_scale", 0.8)) if is_icon else size
        # Pop: crece de 55% a 115% en 0,15 s y se asienta al 100%; fundido al final.
        w = (f"'{sz}*if(lt(t,0.15),0.55+t*4,if(lt(t,0.25),1.15-(t-0.15)*1.5,1))'")
        prep = "format=rgba,"
        if is_icon:
            prep = "format=rgba,scale=256:256:flags=lanczos,pad=iw+16:ih+16:8:8:color=0x101010,"
        chains.append(
            f"[{k}:v]{prep}scale=w={w}:h=-1:eval=frame:flags=lanczos,"
            f"fade=t=out:st={show - 0.25:.2f}:d=0.25:alpha=1,"
            f"setpts=PTS-STARTPTS+{t0:.2f}/TB[e{k}]")
        chains.append(
            f"{last}[e{k}]overlay=x={xexpr(it)}:y={int(it.get('y') or y_center)}-h/2:eof_action=pass:"
            f"enable='between(t,{t0:.2f},{t0 + show:.2f})'[v{k}]")
        last = f"[v{k}]"
    if not k and not sounds:
        return False
    vmap = last if k else "0:v"
    amap, aenc = "0:a?", ["-c:a", "copy"]
    if sounds:
        # Sonidos: cada uno retrasado a su segundo y mezclado con el audio del clip.
        vol = float(cfg.get("sfx_volume", 0.8))
        labels = []
        for j, (t0, path) in enumerate(sounds):
            n = k + 1 + j
            inputs += ["-i", path]
            ms = int(t0 * 1000)
            chains.append(f"[{n}:a]aresample=48000,adelay={ms}|{ms},volume={vol}[s{j}]")
            labels.append(f"[s{j}]")
        chains.append(f"[0:a]{''.join(labels)}amix=inputs={len(labels) + 1}:normalize=0:"
                      f"duration=first[aout]")
        amap, aenc = "[aout]", ["-c:a", "aac", "-b:a", "192k"]
    try:
        import ffmpeg_utils
        venc = ffmpeg_utils.video_encode_args(ffmpeg_utils.QUALITY)
    except Exception:
        venc = ["-c:v", "libx264", "-preset", "medium", "-crf", "18"]
    vargs = venc if k else ["-c:v", "copy"]
    fc = ["-filter_complex", ";".join(chains)] if chains else []
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", video_in, *inputs, *fc,
           "-map", vmap, "-map", amap, *vargs, *aenc, "-movflags", "+faststart", video_out]
    subprocess.run(cmd, check=True)
    return True

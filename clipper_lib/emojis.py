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
Return only json: {{"clips": [{{"id": 1, "emojis": [{{"t": 3.2, "e": "😂"}}]}}]}}

CLIPS:
{clips}
"""


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
    prompt = PROMPT.format(min_n=cfg.get("min_per_clip", 2), max_n=cfg.get("max_per_clip", 4),
                           clips="\n\n".join(blocks))
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
                items.append({"t": round(t, 2), "e": e["e"].strip()})
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
    """Superpone los emojis en ``video_in``. ``y_center``/``x_center``: centro
    del emoji en el frame 1080x1920 (x por defecto: centrado)."""
    def xexpr(it):
        x = it.get("x", x_center)
        return "(W-w)/2" if x is None else f"{int(x)}-w/2"
    size = int(cfg.get("size", 230))
    show = float(cfg.get("duration", 1.4))
    inputs, chains, last = [], [], "[0:v]"
    k = 0
    for it in items:
        png = image_for(it["e"], cache_dir, cfg.get("style", "3D"))
        if not png:
            print(f"   ⚠️ Emoji sin imagen en Noto: {it['e']}")
            continue
        k += 1
        t0 = it["t"]
        inputs += ["-loop", "1", "-t", f"{show:.2f}", "-i", png]
        # Pop: crece de 55% a 115% en 0,15 s y se asienta al 100%; fundido al final.
        w = (f"'{size}*if(lt(t,0.15),0.55+t*4,if(lt(t,0.25),1.15-(t-0.15)*1.5,1))'")
        chains.append(
            f"[{k}:v]format=rgba,scale=w={w}:h=-1:eval=frame,"
            f"fade=t=out:st={show - 0.25:.2f}:d=0.25:alpha=1,"
            f"setpts=PTS-STARTPTS+{t0:.2f}/TB[e{k}]")
        chains.append(
            f"{last}[e{k}]overlay=x={xexpr(it)}:y={int(it.get('y') or y_center)}-h/2:eof_action=pass:"
            f"enable='between(t,{t0:.2f},{t0 + show:.2f})'[v{k}]")
        last = f"[v{k}]"
    if not k:
        return False
    try:
        import ffmpeg_utils
        venc = ffmpeg_utils.video_encode_args(ffmpeg_utils.QUALITY)
    except Exception:
        venc = ["-c:v", "libx264", "-preset", "medium", "-crf", "18"]
    cmd = ["ffmpeg", "-v", "error", "-y", "-i", video_in, *inputs,
           "-filter_complex", ";".join(chains), "-map", last, "-map", "0:a?",
           *venc, "-c:a", "copy", "-movflags", "+faststart", video_out]
    subprocess.run(cmd, check=True)
    return True

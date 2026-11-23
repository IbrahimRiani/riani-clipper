"""Inicio y final de cada clip con sentido (DeepSeek, una sola llamada).

El ajuste mecánico (sentence_snap) corta en puntos y pausas. Con streamers que
hablan sin parar y sin que whisper ponga puntos, eso no basta: salían clips que
empezaban a mitad de frase ("ser muy lento, muy corto...") o que acababan justo
antes del remate ("me invitó a su casa" | "¿y pasaron cosas? Y me dijo, sí").

Aquí, ya elegidos los clips, se le enseñan a DeepSeek las líneas del transcript
de cada clip con margen antes y después, numeradas y con su tiempo, y elige la
línea donde empieza la idea y la línea donde se cierra (remate incluido).
"""
import json
import os

import httpx

PROMPT = """You are a short-form video editor (TikTok/Shorts/Reels). Each clip below is a
candidate cut from a long video or stream. Its transcript lines are numbered with their
time in seconds. Lines marked with ">" are the current cut; the unmarked lines around them
are context you MAY include.

For each clip choose:
- "start": the line where the moment really begins, so a viewer with no context understands
  it. Never start mid-sentence or mid-thought (e.g. a line that continues the previous one
  like "ser muy lento..." or "y entonces..."). Prefer starting on a hook or the setup of the
  story/argument. If the current start is fine, keep it.
- "end": the line where the idea closes: the punchline, answer, reaction or conclusion is
  INCLUDED and the last sentence is complete. Never end on a setup or a question whose
  answer comes in the next lines. Never end mid-list or mid-sentence. If the current end is
  fine, keep it.
- Read the lines as one continuous speech: whisper splits sentences across lines, so a line
  that starts in lowercase or with "y", "pero", "que", "porque" usually continues the previous
  one. If the best idea does not fit in the time limit, cut a shorter complete idea inside it
  rather than a long one that is cut off.
- The result (end time of the end line minus start time of the start line) must be between
  {min_s} and {max_s} seconds.

Answer ONLY JSON: {{"clips": [{{"id": 1, "start": <line number>, "end": <line number>}}]}}

{clips}"""


def _lines(segments, a, b):
    return [s for s in segments if float(s["end"]) > a and float(s["start"]) < b
            and str(s.get("text", "")).strip()]


def _chat(prompt, think=True):
    body = {"model": os.environ["LLM_MODEL"], "temperature": 0.2, "stream": False,
            "response_format": {"type": "json_object"},
            "messages": [{"role": "user", "content": prompt}]}
    body.update(json.loads(os.environ.get("LLM_EXTRA_BODY") or "{}"))
    if think:
        # Decidir dónde empieza y acaba una idea pide razonar: aquí sí se deja
        # pensar al modelo (el resto del pipeline va sin "thinking").
        body["thinking"] = {"type": "enabled"}
        body.pop("temperature", None)
    headers = {"Authorization": f"Bearer {os.environ['LLM_API_KEY']}"}
    url = os.environ["LLM_BASE_URL"].rstrip("/") + "/chat/completions"
    for _ in range(3):
        r = httpx.post(url, json=body, headers=headers, timeout=300)
        r.raise_for_status()
        resp = r.json()
        try:
            return json.loads(resp["choices"][0]["message"].get("content") or ""), resp.get("usage", {})
        except ValueError:
            continue  # a veces devuelve contenido vacío: reintentar
    raise RuntimeError("DeepSeek no devolvió JSON válido para los cortes")


def refine(shorts, transcript, cfg, min_s, max_s):
    """Ajusta start/end de cada clip (in place) y devuelve la lista."""
    segments = transcript.get("segments") or []
    if not shorts or not segments:
        return shorts
    before = cfg.get("context_before_seconds", 35)
    after = cfg.get("context_after_seconds", 25)
    lead, tail = cfg.get("lead_seconds", 0.2), cfg.get("tail_seconds", 0.35)
    blocks, tables = [], []
    for i, c in enumerate(shorts, start=1):
        s, e = float(c["start"]), float(c["end"])
        lines = _lines(segments, s - before, e + after)
        tables.append(lines)
        rows = []
        for n, g in enumerate(lines, start=1):
            mark = ">" if float(g["start"]) < e and float(g["end"]) > s else " "
            rows.append(f"{mark}{n} [{float(g['start']):.1f}-{float(g['end']):.1f}] {g['text'].strip()}")
        blocks.append(f"# clip {i} (current cut {s:.1f}-{e:.1f})\n" + "\n".join(rows))
    try:
        data, usage = _chat(PROMPT.format(min_s=int(min_s), max_s=int(max_s), clips="\n\n".join(blocks)),
                            think=cfg.get("llm_refine_thinking", True))
    except Exception as ex:  # un ajuste extra nunca debe tumbar el trabajo
        print(f"   ⚠️ Ajuste de cortes con DeepSeek omitido ({ex}).")
        return shorts
    changed = 0
    for item in data.get("clips", []):
        try:
            i, a, b = int(item["id"]), int(item["start"]), int(item["end"])
        except (KeyError, ValueError, TypeError):
            continue
        if not (0 < i <= len(shorts)):
            continue
        lines = tables[i - 1]
        if not (1 <= a <= b <= len(lines)):
            continue
        ns = max(0.0, float(lines[a - 1]["start"]) - lead)
        ne = float(lines[b - 1]["end"]) + tail
        if b < len(lines):
            ne = min(ne, float(lines[b]["start"]) - 0.05)
        if not (min_s * 0.9 <= ne - ns <= max_s):
            continue  # fuera de rango: mejor el corte que ya había
        c = shorts[i - 1]
        if abs(ns - float(c["start"])) > 0.3 or abs(ne - float(c["end"])) > 0.3:
            c.setdefault("cut_before_refine", [c["start"], c["end"]])
            c["start"], c["end"] = round(ns, 3), round(ne, 3)
            changed += 1
    print(f"   ✂️ Cortes revisados por DeepSeek: {changed} de {len(shorts)} ajustados "
          f"({usage.get('prompt_tokens', '?')} tokens de entrada).")
    return shorts

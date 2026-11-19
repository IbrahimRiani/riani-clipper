"""Cortes en final de frase, no en mitad.

OpenShorts ajusta el corte a la palabra más cercana. En la prueba con Folagor
un clip acababa en el segundo 240,3, en mitad de "10 equipos, hostia puta",
justo en el remate. Aquí, después de su ajuste, alargamos el final hasta
que termine la frase (o, si no cabe, lo recortamos a la frase anterior) y
empezamos en el inicio de una frase.

Fin de frase = la palabra acaba en . ? ! … o va seguida de una pausa.
"""
import re

_END_PUNCT = re.compile(r"[.?!…]+[\"'»”)]*$")

# Finales de segmento de whisper (sus "frases"). Los rellena clipper.py desde
# el transcript, porque la lista de palabras que recibe el snap no los trae.
SEGMENT_ENDS = []


def set_segments(transcript):
    SEGMENT_ENDS[:] = sorted(float(s.get("end", 0)) for s in transcript.get("segments", []))


def _ends_segment(t, tol=0.08):
    import bisect
    i = bisect.bisect_left(SEGMENT_ENDS, t - tol)
    return i < len(SEGMENT_ENDS) and SEGMENT_ENDS[i] <= t + tol


def _is_sentence_end(words, i, pause):
    if i >= len(words) - 1:
        return True
    w = str(words[i].get("w", "")).strip()
    gap = float(words[i + 1]["s"]) - float(words[i]["e"])
    # Un final de segmento de whisper solo cuenta si lo siguiente empieza
    # frase nueva (mayúscula, ¿, ¡) o hay una pausita. Whisper a veces parte
    # una frase en dos sin pausa: "los mejores entradores" | "del mundo" (no
    # es fin) frente a "hostia puta" | "Hostia, Silvio" (sí lo es).
    nxt = str(words[i + 1].get("w", "")).strip()
    starts_new = bool(nxt) and (nxt[0].isupper() or nxt[0] in "¿¡")
    return bool(_END_PUNCT.search(w)) or gap >= pause or \
        ((gap >= 0.25 or starts_new) and _ends_segment(float(words[i]["e"])))


def snap(start, end, words, min_duration, max_duration, cfg):
    """Devuelve (start, end) ajustados a frases. ``words``: [{'w','s','e'}]."""
    if not words:
        return start, end
    pause = cfg.get("pause_seconds", 0.7)
    extend = cfg.get("max_extend_seconds", 6.0)
    back = cfg.get("max_start_back_seconds", 4.0)
    tail = cfg.get("tail_seconds", 0.35)
    lead = cfg.get("lead_seconds", 0.2)

    # Última palabra que empieza antes del final propuesto.
    last = max((i for i, w in enumerate(words) if float(w["s"]) < end), default=None)
    first = min((i for i, w in enumerate(words) if float(w["e"]) > start), default=None)
    if last is None or first is None or first > last:
        return start, end

    # INICIO: si empezamos a mitad de frase, retroceder a su inicio (poco).
    i = first
    while i > 0 and not _is_sentence_end(words, i - 1, pause) \
            and start - float(words[i - 1]["s"]) <= back:
        i -= 1
    new_start = max(0.0, float(words[i]["s"]) - lead)

    # FINAL: alargar hasta el fin de frase si cabe; si no, volver a la anterior.
    j = last
    while not _is_sentence_end(words, j, pause) and j + 1 < len(words) \
            and float(words[j + 1]["e"]) - end <= extend \
            and float(words[j + 1]["e"]) - new_start <= max_duration:
        j += 1
    if not _is_sentence_end(words, j, pause):
        k = last
        while k > i and not _is_sentence_end(words, k, pause):
            k -= 1
        if float(words[k]["e"]) - new_start >= min_duration:
            j = k
    new_end = float(words[j]["e"]) + tail
    if j + 1 < len(words):
        new_end = min(new_end, float(words[j + 1]["s"]) - 0.05)
    # Límite duro de duración (Shorts/TikTok): salían clips de 61-62 s porque el
    # inicio retrocedía y el final se alargaba. Volver a la última frase que cabe;
    # si ninguna cabe, cortar en la última palabra que cabe.
    hard = cfg.get("hard_max_seconds", max_duration)
    if new_end - new_start > hard:
        k = j
        while k > i and (float(words[k]["e"]) + tail - new_start > hard
                         or not _is_sentence_end(words, k, pause)):
            k -= 1
        if float(words[k]["e"]) + tail - new_start > hard or k <= i:
            k = j
            while k > i and float(words[k]["e"]) + tail - new_start > hard:
                k -= 1
        new_end = min(float(words[k]["e"]) + tail, new_start + hard)
    if new_end - new_start < min_duration * 0.8:
        return start, min(end, start + hard)  # algo raro: mejor el ajuste original
    return round(new_start, 3), round(new_end, 3)

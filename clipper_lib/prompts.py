"""Ajustes a los prompts de OpenShorts para streams y DeepSeek.

Se añaden unas pocas líneas (pocos tokens) en vez de reescribir los prompts:
los de OpenShorts ya están probados. Ojo: las plantillas pasan por
str.format, así que aquí NO se pueden usar llaves sueltas.
"""

STREAM_HINT = """
STREAM / GAMING NOTES:
- Window text may end with [LOUD: <second>s +<n>dB, ...]: audio peaks far above
  this video's normal level (shouting, laughing, screaming, clutch moments).
  These are strong signals in streams even if the transcript looks thin.
- Prioritize reactions, laughs, screams, rage, clutch plays and funny fails.
- Never pick waiting screens, BRB, intros, ads or "the stream starts soon".
  Text starting with [DEAD AIR / WAITING SCREEN] is exactly that: score it 0.
- Every clip must be about a DIFFERENT moment or topic: never two clips about
  the same event, person or joke (keep only the best one).
- Write titles and hooks in the transcript language.
"""


def _insert_before(template, marker, extra):
    i = template.find(marker)
    return template[:i] + extra + "\n" + template[i:] if i >= 0 else template + extra


def patch(gemini_worker, extra=""):
    """``extra``: reglas de una campaña (las llaves se doblan por str.format)."""
    hint = STREAM_HINT + extra.replace("{", "{{").replace("}", "}}")
    gemini_worker.SCORE_PROMPT_TEMPLATE = _insert_before(
        gemini_worker.SCORE_PROMPT_TEMPLATE, "TRANSCRIPT_LANGUAGE:", hint)
    gemini_worker.DETAIL_PROMPT_TEMPLATE = _insert_before(
        gemini_worker.DETAIL_PROMPT_TEMPLATE, "TRANSCRIPT_LANGUAGE:", hint)

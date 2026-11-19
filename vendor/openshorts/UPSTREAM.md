# Código de terceros: OpenShorts

Estos archivos vienen de https://github.com/mutonby/openshorts
(commit 7e3468cdd3d9d998c903b11227a6ca81273a67c4), licencia MIT (ver LICENSE y NOTICE).
Copyright (c) 2024 OpenShorts.

Solo se copió el núcleo MIT del pipeline de clips. NO se incluye nada de la carpeta
`cloud/` (licencia comercial), ni el dashboard, Remotion, render-service u otras partes SaaS.

Las modificaciones locales sobre estos archivos se anotan aquí:
- `llm_backend.py`: variables `LLM_EXTRA_BODY` (campos extra en la petición, p. ej. desactivar
  el "thinking" de DeepSeek) y `LLM_JSON_SCHEMA=0` (saltar `json_schema`, que DeepSeek no admite).
  Marcado con `[Claude Clipper]`.
- `subtitles.py`: variable `CAPTION_SEAM_Y` para colocar los subtítulos en la costura entre
  webcam y juego (layout gaming) en vez de en el centro exacto. Marcado con `[Claude Clipper]`.

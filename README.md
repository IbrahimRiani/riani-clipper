# Claude Clipper

Convierte videos largos (VODs de Twitch, streams, gameplays, podcasts de YouTube) en
clips verticales 9:16 con subtítulos, de forma local y gratuita. El único servicio de
pago es tu clave de **DeepSeek**, que solo lee texto (el transcript) para elegir los momentos.

> **Estado: fase 4** (layouts gaming, chatting/reacción y podcast).

## Cómo funciona

1. **Descarga**: yt-dlp baja el video de YouTube o Twitch (o usas un archivo local).
2. **Transcripción**: faster-whisper, en local, con tiempos por palabra.
3. **Picos de volumen**: ffmpeg mide el volumen cada 0,5 s y marca gritos y risas como pista extra.
4. **Elección de momentos**: DeepSeek puntúa trozos del transcript y luego elige los
   mejores clips de 20-60 s, con título, hook y puntuación. Responde en JSON validado.
5. **Reencuadre a 9:16 y subtítulos** palabra por palabra. Exporta MP4.

## Instalación en Windows

Necesitas Python 3.11, ffmpeg en el PATH y, para ir rápido, una GPU NVIDIA.

```powershell
uv venv --python 3.11 .venv
```

```powershell
uv pip install --python .venv\Scripts\python.exe torch==2.11.0 torchvision==0.26.0 --index-url https://download.pytorch.org/whl/cu128
```

```powershell
uv pip install --python .venv\Scripts\python.exe -r requirements.txt
```

Sin GPU, cambia `cu128` por `cpu` y en `config.yaml` pon `device: cpu`, `compute: int8` y `model: small`.

Después copia `.env.example` como `.env` y pon tu clave:

```
DEEPSEEK_API_KEY=sk-...
```

## Instalación en Linux

Es lo mismo, pero con `.venv/bin/python` y ffmpeg instalado con tu gestor de paquetes (`apt install ffmpeg`).

## Uso fácil: la app

Doble clic en **`Clipper.bat`**. Se abre una página en tu navegador (solo en tu PC,
`http://127.0.0.1:7860`) donde puedes:

- **Arrastrar vídeos** o **pegar URLs** de YouTube/Twitch (con "empezar en" y "minutos" para
  bajar solo un tramo de un VOD largo).
- Elegir tipo de vídeo (auto/gaming/chatting/podcast), fuente, emojis, idioma y duración.
- Ver el progreso y los clips terminados con su título y puntuación, y descargarlos.
- **Lotes**: mete vídeos (o un `urls.txt` con una URL por línea) en la carpeta `entrada/` y
  pulsa "Procesar la carpeta entrada/". Lo procesado se mueve a `entrada/hechos/`.
- Pestaña **Mis clips**: todo lo que has procesado antes.
- **📥 Solo descargar**: baja las URLs sin sacar clips y las parte en tramos (30 min por
  defecto, sin recodificar: tarda segundos). Si dejas marcado "Dejar los tramos en entrada/",
  luego "Procesar la carpeta entrada/" saca los clips de todos los tramos seguidos. Por
  comandos: `.venv\Scripts\python.exe tools\descargar.py URL --tramo 30 --entrada`.

Para apagarla, cierra la ventana negra que se abre con el `.bat`.

## Uso por comandos

```powershell
.venv\Scripts\python.exe clipper.py --input samples\gaming_rubius_gta6.mp4
```

Para un VOD largo, baja solo un tramo (aquí 30 min a partir de la hora 2):

```powershell
.venv\Scripts\python.exe clipper.py --url "https://www.twitch.tv/videos/1234567890" --start 2:00:00 --minutes 30
```

Las descargas se guardan en `downloads/`. El resultado queda así:

```
output/<video>/01_titulo-del-clip.mp4   clip final 1080x1920 con subtítulos
output/<video>/clips.json               resumen: tiempos, puntuación, título, hook, motivo
output/<video>/debug/clip_01.json       todo lo que se sabe de cada clip
output/<video>/debug/...                clips sin subtítulos, subtítulos .ass, layout usado
```

Para forzar un layout: `--layout gaming` (o `auto`, `chatting`, `podcast`).

### Cambiar el estilo sin reprocesar

Si cambias la fuente, el estilo o los emojis en `config.yaml`, rehaz solo los subtítulos:

```powershell
.venv\Scripts\python.exe tools\restyle.py output\<video>
```

Para ver todas las fuentes sobre un clip: `tools\caption_preview.py` (ver el propio archivo).

## Layout gaming

1. Se muestrean 60 frames repartidos por todo el vídeo y se buscan caras (MediaPipe sobre
   trozos ampliados, para caras pequeñas). La posición que más se repite es la webcam.
2. El rectángulo exacto de la cam sale de sus bordes: un marco dibujado (línea continua) o
   el salto entre habitación quieta y juego en movimiento.
3. La zona de juego es lo que más cambia entre frames, sin overlay decorativo ni webcam.
4. Por clip: si la cara no aparece, solo juego. Se quitan barras negras y, en juegos
   panorámicos, se recorta centrado en la acción esquivando la webcam.
5. Los subtítulos van en la costura entre webcam y juego, y los emojis encima (al lado de la cara si la taparían).

En `output/<video>/debug/` quedan `layout_detection.jpg` (cajas detectadas) y
`clip_XX_cajas.mp4` (cada clip con las cajas dibujadas).

## Layout chatting / reacción

Los directos de charla alternan escenas: cámara a pantalla completa y reacciones (un vídeo o
una web ocupa la pantalla y la webcam es pequeña). Por eso se decide **escena a escena**:

- **Cara grande** → la cámara sigue la cara. Está quieta mientras la cara cabe cómoda y solo
  se desliza cuando se acerca al borde; nunca la deja cortada.
- **Webcam pequeña + contenido** → igual que gaming: webcam arriba, contenido abajo.
  La cam se busca también **por su aspecto**, así que se encuentra aunque el streamer la mueva
  de sitio o aunque no se le vea la cara.
- **Sin cara** → contenido a lo ancho sobre fondo difuminado.

Las caras que nunca se mueven (pósters, murales, dibujos) se ignoran.
En `debug/clip_XX_cajas.mp4` se ve el modo elegido en cada escena.

## Layout podcast / entrevista

Se aplica escena a escena dentro del layout chatting, cuando hay **2 o más personas en plano**:

- **Quién habla**: movimiento de la boca de cada persona + volumen del audio, en ventanas de
  0,4 s (detector de OpenShorts, gratis, sin token de Hugging Face).
- **Speaker switch**: se encuadra a quien habla; un cambio solo vale si dura ≥ 1,5 s.
- **Split**: las dos personas apiladas cuando hablan a la vez o el intercambio es muy rápido.
- **Una sola cámara con varias personas**: cada una se recorta por su caja. Las personas se
  detectan con YOLO (encuentra también perfiles) y se ignora el público en primer plano.
- **Multicámara** (un primer plano por persona): cada escena tiene una cara → seguimiento de cara.

`podcast_mode` en `config.yaml`: `auto`, `switch` (sin split) o `split` (siempre las dos).

## Layout pantalla partida

Para directos con la persona en un panel lateral a toda altura y el contenido (otro directo,
un vídeo) al lado. Se detecta por la **línea vertical** entre paneles en casi todo el alto
(se mide frame a frame, porque el streamer la mueve) más una **cara grande** entera a un
lado. Resultado: persona arriba (recortada alrededor de la cara) y contenido abajo, 50/50,
con los subtítulos en la costura. Si un clip concreto no tiene la pantalla partida, se hace
escena a escena como en chatting. Para forzarlo: `--layout split`.

También cubre **dos cámaras lado a lado** (videochats tipo Omegle/Umingle, debates a dos
cámaras): dos caras, una a cada lado, con una línea entre ellas a la altura de las caras.
La de la izquierda va arriba y la de la derecha abajo, cada una recortada en su cara y sin
el navegador ni los botones de alrededor (se detecta el borde de arriba y de abajo de cada
cámara).

## Pantallas de espera / BRB / intro

Antes de elegir clips se marcan los tramos "muertos": al menos 20 s sin voz (o solo frases que
whisper se inventa con música) y con la imagen casi quieta o en bucle. DeepSeek los recibe
marcados como "no clipear" y, además, se descarta cualquier clip que caiga más de un 25 % en
ellos. Probado con la cuenta atrás de 8 min del inicio de un directo de 90 min, sin falsos
positivos en el resto de vídeos de prueba. Ajustes en la sección `dead_air` de `config.yaml`.

## VODs largos

Probado con 90 min de un directo de Twitch (1080p60): ~19 min en total en una RTX 3080
(transcripción ~5 min, análisis de pantallas de espera ~1 min, DeepSeek ~1 min, 10 clips ~6 min).
Si el streamer mueve la webcam a mitad de directo, se encuentra en su nueva posición.

## Iconos del LoL y sonidos meme

- **Iconos del LoL**: al mencionar un objeto grande o un hechizo de invocador ("rabadon",
  "black cleaver", "flash", "smite", "tp"…) aparece su icono oficial en vez del emoji. Los
  iconos son de Riot (Data Dragon) y se descargan solos a `assets/lol/`.
- **Sonidos meme**: pon tus archivos en `assets/sfx/` (vine boom, faaah, rizz; ver
  `LEEME.txt`). DeepSeek elige como mucho 2 por clip en los momentos que encajan (remate,
  fail, ligoteo). No se suben a GitHub.

## Configuración

Todo lo ajustable está en [config.yaml](config.yaml): modelo de DeepSeek, modelo de
whisper, duración de los clips y umbrales de los picos de volumen.

## Estructura

- `clipper.py`: comando principal. Lee `config.yaml` y `.env` y lanza el pipeline.
- `clipper_lib/`: nuestras piezas (picos de volumen, ajustes de prompts y, más adelante, layouts).
- `vendor/openshorts/`: núcleo de [OpenShorts](https://github.com/mutonby/openshorts) (MIT).
  Ver [UPSTREAM.md](vendor/openshorts/UPSTREAM.md).
- `samples/`: videos de prueba.

## Créditos y licencias

- [OpenShorts](https://github.com/mutonby/openshorts), MIT, © 2024 OpenShorts: pipeline
  de transcripción, selección de clips, reencuadre y subtítulos. No se incluye su carpeta
  `cloud/`, que tiene licencia comercial.
- [faster-whisper](https://github.com/SYSTRAN/faster-whisper) (MIT), [yt-dlp](https://github.com/yt-dlp/yt-dlp) (Unlicense),
  [MediaPipe](https://github.com/google-ai-edge/mediapipe) (Apache-2.0), [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) (BSD-3).
- [Ultralytics YOLO](https://github.com/ultralytics/ultralytics) es **AGPL-3.0**. Para uso
  personal no hay problema; si algún día distribuyes esto como servicio, revisa esa licencia.
- Fuentes en `vendor/openshorts/fonts` y `assets/fonts` (Luckiest Guy, Bebas Neue, Titan One…):
  [Google Fonts](https://github.com/google/fonts), licencias OFL/Apache en `assets/fonts/licenses`.
- Emojis: [Noto Emoji](https://github.com/googlefonts/noto-emoji) de Google (imágenes Apache 2.0),
  descargados a `assets/emoji` la primera vez que se usan.

"""Elección de layout (auto / forzado) y enganche al render de OpenShorts.

Heurísticas de ``auto`` sobre frames muestreados de todo el video:
- webcam pequeña y fija + escena que cambia mucho  -> gaming
- cara grande, o webcam que aparece y desaparece    -> chatting (decide escena a escena)
- dos o más caras grandes                           -> podcast: lo resuelve el layout
  chatting escena a escena (speaker switch / split)
"""
import json
import os

import cv2

from . import chatting_render, gaming_render, webcam

LAYOUTS = ("auto", "gaming", "chatting", "podcast")


def decide(analysis, forced, cfg):
    """Devuelve (layout, motivo) para enseñar en el log y en el debug."""
    if forced and forced != "auto":
        return forced, "forzado con --layout"
    cam = analysis.get("webcam")
    face_frac = analysis.get("median_largest_face_frac", 0)
    motion = analysis.get("scene_motion", 0)
    if cam and cam["area_frac"] <= cfg.get("gaming_max_cam_area", 0.3) \
            and cam.get("hit_ratio", 1) >= cfg.get("gaming_min_cam_ratio", 0.6) \
            and motion >= cfg.get("gaming_min_scene_motion", 15):
        return "gaming", (f"webcam fija ({cam['area_frac']:.0%} del frame, visible en el "
                          f"{cam.get('hit_ratio', 1):.0%} del vídeo) y escena dinámica "
                          f"(movimiento {motion:.0f})")
    if cam:
        return "chatting", (f"webcam que aparece en el {cam.get('hit_ratio', 0):.0%} del vídeo: "
                            f"se decide escena a escena (cam arriba / cara / contenido)")
    if face_frac >= cfg.get("chatting_min_face_frac", 0.05):
        return "chatting", f"cara grande ({face_frac:.0%} del ancho): seguimiento de cara"
    return "chatting", "sin webcam ni cara clara: escena a escena (sobre todo contenido)"


def analyze_and_save(video, out_dir, cfg):
    """Análisis global + imagen y JSON de debug en out_dir/debug."""
    analysis, frames = webcam.analyze(video, cfg)
    debug = os.path.join(out_dir, "debug")
    os.makedirs(debug, exist_ok=True)
    if analysis.get("webcam") and analysis["webcam"].get("frame_index") is not None:
        tpl = os.path.join(debug, "cam_template.png")
        webcam.save_template(frames[analysis["webcam"]["frame_index"]][1],
                             analysis["webcam"]["rect"], tpl)
        analysis["webcam"]["template"] = tpl
    img = webcam.draw_debug(frames[len(frames) // 2][1], analysis)
    cv2.imwrite(os.path.join(debug, "layout_detection.jpg"), img)
    with open(os.path.join(debug, "layout_detection.json"), "w", encoding="utf-8") as f:
        json.dump(analysis, f, indent=2)
    return analysis


def install(layout, analysis, cfg):
    """Sustituye reframe_v2.render por el layout elegido (si es nuestro)."""
    if layout not in ("gaming", "chatting", "podcast"):
        return
    if layout == "podcast":
        # El layout chatting ya detecta las escenas con 2+ personas y aplica
        # speaker switch / split en ellas.
        layout = "chatting"
    if not analysis.get("webcam"):
        if layout == "gaming":
            print("   ⚠️ Layout gaming sin webcam detectada: solo juego.")
    else:
        geo = gaming_render.video_geometry(analysis, cfg)
        # Subtítulos en la costura entre webcam y juego.
        os.environ["CAPTION_SEAM_Y"] = f"{(geo['game_y'] + cfg.get('caption_offset_px', 0)) / gaming_render.OUT_H:.4f}"
        print(f"   Paneles: cam {geo['cam_h']}px, juego {geo['game_h']}px, "
              f"margen arriba {geo['cam_y']}px")

    import reframe_v2
    original = reframe_v2.render

    def render(input_video, final_output_video, aspect_ratio, *a, **k):
        if abs(aspect_ratio - 9 / 16) > 0.01:
            return original(input_video, final_output_video, aspect_ratio, *a, **k)
        if layout == "gaming":
            plan = gaming_render.render(input_video, final_output_video, analysis, cfg)
            print(f"   🎮 Layout gaming: {plan['mode']} (cam visible {plan['cam_present']})")
        else:
            plan = chatting_render.render(input_video, final_output_video, analysis, cfg)
            print("   💬 Layout chatting: " + ", ".join(
                f"{s['start']:.0f}-{s['end']:.0f}s {s['mode']}" for s in plan["scenes"]))
        return True

    reframe_v2.render = render

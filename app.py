"""
app.py
------
GameLens AI - Gradio web app and entry point (local + Hugging Face Spaces).

When the user clicks "Load Fast Sample Demo", cached metrics/charts/report
return without YOLO. When the user clicks "Analyze Uploaded Video / Reprocess
with YOLO", uploaded clips or explicit sample reprocessing run YOLO + ByteTrack.

A separate "Ask" button answers questions using only the computed metrics.
The app runs even without an OpenAI API key by showing rule-based text.
"""

from __future__ import annotations

import inspect
import json
import os
from pathlib import Path
import traceback

import gradio as gr

print("GameLens AI app imported")

PROJECT_ROOT = Path(__file__).resolve().parent
SAMPLE_VIDEO_PATH = PROJECT_ROOT / "sample_videos" / "default_soccer_clip.mp4"
SAMPLE_OUTPUTS_DIR = PROJECT_ROOT / "sample_outputs"
SAMPLE_METRICS_PATH = SAMPLE_OUTPUTS_DIR / "sample_metrics.json"
SAMPLE_TRACKING_CSV_PATH = SAMPLE_OUTPUTS_DIR / "sample_tracking_data.csv"
SAMPLE_REPORT_PATH = SAMPLE_OUTPUTS_DIR / "sample_report.txt"
SAMPLE_CHART_PATHS = [
    SAMPLE_OUTPUTS_DIR / "sample_chart_zone_activity.png",
    SAMPLE_OUTPUTS_DIR / "sample_chart_top_track_segments.png",
    SAMPLE_OUTPUTS_DIR / "sample_chart_player_count.png",
    SAMPLE_OUTPUTS_DIR / "sample_chart_heatmap.png",
]
SAMPLE_VIDEO_POINTER_MAX_BYTES = 1024

# --- UI theme (visual only; no effect on the pipeline) -----------------
THEME = gr.themes.Soft(
    primary_hue=gr.themes.colors.emerald,
    secondary_hue=gr.themes.colors.teal,
    neutral_hue=gr.themes.colors.slate,
    radius_size=gr.themes.sizes.radius_lg,
    # Fonts (Inter / Sora) are loaded and applied via CUSTOM_CSS below, which
    # avoids a Gradio theme-comparison issue with string fonts.
).set(
    # Primary button — emerald -> teal gradient with soft depth on hover/press
    button_primary_background_fill="linear-gradient(135deg, #059669, #0d9488)",
    button_primary_background_fill_hover="linear-gradient(135deg, #047857, #0f766e)",
    button_primary_text_color="#ffffff",
    button_primary_shadow="0 10px 24px -12px rgba(5,150,105,.70)",
    button_primary_shadow_hover="0 16px 32px -12px rgba(5,150,105,.90)",
    button_primary_shadow_active="0 6px 14px -10px rgba(5,150,105,.90)",
    button_secondary_shadow="0 6px 16px -12px rgba(2,6,23,.50)",
    button_secondary_shadow_hover="0 10px 22px -12px rgba(2,6,23,.60)",
    button_large_radius="14px",
    button_small_radius="10px",
    # Blocks — softer corners and a subtle floating depth
    block_radius="16px",
    block_shadow="0 16px 40px -30px rgba(2,6,23,.55)",
    block_label_text_weight="600",
    # Inputs — rounded fields with a brand-tinted focus ring
    input_radius="12px",
    input_border_color_focus="#10b981",
    input_shadow_focus="0 0 0 3px rgba(16,185,129,.16)",
    # On-brand accents for sliders and checkboxes
    slider_color="#0d9488",
    checkbox_background_color_selected="#059669",
)

# --- Custom CSS for a more premium look --------------------------------
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=Sora:wght@600;700;800&display=swap');

/* ---- Base & layout ------------------------------------------------- */
.gradio-container {
  max-width: 1180px !important;
  margin: 0 auto !important;
  background:
    radial-gradient(1200px 520px at 50% -14%, rgba(16,185,129,.12), transparent 62%),
    radial-gradient(760px 460px at 100% 2%, rgba(13,148,136,.08), transparent 60%),
    var(--body-background-fill);
}
.gradio-container, .gradio-container button,
.gradio-container input, .gradio-container textarea,
.gradio-container select {
  font-family: 'Inter', ui-sans-serif, system-ui, -apple-system, sans-serif;
  -webkit-font-smoothing: antialiased;
  -moz-osx-font-smoothing: grayscale;
}
.gradio-container button, .gradio-container input,
.gradio-container textarea, .gl-card, .gl-pill {
  transition: all .22s cubic-bezier(.4, 0, .2, 1);
}

/* ---- Hero banner --------------------------------------------------- */
.gl-hero {
  position: relative; overflow: hidden;
  border-radius: 24px; padding: 42px 44px; margin-bottom: 6px;
  color: #ecfdf5; border: 1px solid rgba(255,255,255,.08);
  background:
    radial-gradient(620px 240px at 86% -20%, rgba(45,212,191,.38), transparent 62%),
    linear-gradient(135deg, #04231f 0%, #065f46 46%, #0e7490 100%);
  box-shadow: 0 34px 66px -30px rgba(6,95,70,.75), inset 0 1px 0 rgba(255,255,255,.09);
}
.gl-hero::before {
  content: ""; position: absolute; inset: 0; pointer-events: none; opacity: .5;
  background-image:
    linear-gradient(rgba(255,255,255,.05) 1px, transparent 1px),
    linear-gradient(90deg, rgba(255,255,255,.05) 1px, transparent 1px);
  background-size: 34px 34px;
  -webkit-mask-image: radial-gradient(circle at 18% 12%, #000, transparent 72%);
          mask-image: radial-gradient(circle at 18% 12%, #000, transparent 72%);
}
.gl-hero::after {
  content: ""; position: absolute; right: -90px; top: -100px; pointer-events: none;
  width: 320px; height: 320px;
  background: radial-gradient(circle, rgba(255,255,255,.18), transparent 70%);
}
.gl-hero > * { position: relative; z-index: 1; }
.gl-eyebrow {
  display: inline-block; margin-bottom: 12px;
  font-size: .72rem; font-weight: 700; letter-spacing: .2em; text-transform: uppercase;
  color: #6ee7b7;
}
.gl-hero h1 {
  font-family: 'Sora','Inter',sans-serif; font-size: 2.45rem; font-weight: 800;
  margin: 0 0 10px; letter-spacing: -.025em; color: #fff; line-height: 1.04;
}
.gl-hero p {
  margin: 0; max-width: 780px; font-size: 1.04rem; line-height: 1.6;
  color: #d1fae5; opacity: .95;
}
.gl-pills { margin-top: 22px; display: flex; flex-wrap: wrap; gap: 9px; }
.gl-pill {
  background: rgba(255,255,255,.10); border: 1px solid rgba(255,255,255,.22);
  color: #fff; padding: 7px 14px; border-radius: 999px;
  font-size: .78rem; font-weight: 600; backdrop-filter: blur(6px);
}
.gl-pill:hover {
  transform: translateY(-1px);
  background: rgba(255,255,255,.20); border-color: rgba(255,255,255,.40);
}

/* ---- Cards --------------------------------------------------------- */
.gl-card {
  position: relative;
  border: 1px solid var(--border-color-primary) !important;
  border-radius: 20px !important; padding: 22px 24px !important;
  background: var(--block-background-fill) !important;
  box-shadow: 0 18px 44px -32px rgba(2,6,23,.60) !important;
}
.gl-card:hover {
  border-color: rgba(16,185,129,.35) !important;
  box-shadow: 0 26px 52px -30px rgba(6,95,70,.42) !important;
}

/* ---- Section & step titles ---------------------------------------- */
.gl-card-title {
  display: flex; align-items: center; gap: 10px; margin: 0 0 15px;
  font-weight: 700; font-size: .78rem; letter-spacing: .1em; text-transform: uppercase;
  color: var(--body-text-color-subdued);
}
.gl-card-title::before {
  content: ""; flex: none; width: 18px; height: 3px; border-radius: 999px;
  background: linear-gradient(90deg, #10b981, #0d9488);
}
.gl-card-title.gl-step::before { display: none; }
.gl-step-num {
  display: inline-flex; align-items: center; justify-content: center; flex: none;
  width: 27px; height: 27px; border-radius: 9px;
  background: linear-gradient(135deg, #059669, #0d9488);
  color: #fff; font-size: .82rem; font-weight: 800;
  box-shadow: 0 8px 16px -8px rgba(6,95,70,.85);
}

/* ---- Primary call-to-action --------------------------------------- */
.gl-cta { width: 100% !important; font-weight: 700 !important; letter-spacing: .01em; }
.gl-cta:hover { transform: translateY(-2px); }
.gl-cta:active { transform: translateY(0); }

/* ---- Markdown content (metrics / report) -------------------------- */
.gl-card :is(h1, h2, h3, h4) {
  font-family: 'Sora','Inter',sans-serif; letter-spacing: -.01em;
  color: var(--body-text-color);
}
.gl-card h3 { font-size: 1.12rem; margin: 2px 0 12px; }
.gl-card ul { list-style: none; padding-left: 2px; margin: 8px 0; }
.gl-card ul li { position: relative; padding-left: 20px; margin: 8px 0; line-height: 1.62; }
.gl-card ul li::before {
  content: ""; position: absolute; left: 3px; top: .58em;
  width: 6px; height: 6px; border-radius: 50%;
  background: linear-gradient(135deg, #10b981, #0d9488);
}
.gl-card strong { color: var(--body-text-color); font-weight: 700; }
.gl-card em { color: var(--body-text-color-subdued); }
.gl-card code {
  background: rgba(16,185,129,.12); color: #0d9488;
  padding: 1px 6px; border-radius: 6px; font-size: .88em;
}

/* ---- API-key notice ----------------------------------------------- */
.gl-notice {
  display: flex; gap: 10px; align-items: flex-start;
  border-radius: 14px; padding: 13px 17px; margin: 6px 0; font-size: .9rem;
  background: rgba(245,158,11,.12); border: 1px solid rgba(245,158,11,.38);
  color: var(--body-text-color);
}
.gl-notice::before { content: "\\26A0"; color: #d97706; font-size: 1rem; line-height: 1.35; }
.gl-notice code { background: rgba(0,0,0,.08); padding: 1px 6px; border-radius: 6px; }

/* ---- Q&A answer + footer ------------------------------------------ */
.gl-answer { min-height: 2.1rem; }
.gl-footer {
  text-align: center; color: var(--body-text-color-subdued);
  font-size: .82rem; line-height: 1.7; padding: 22px 0 6px; margin-top: 6px;
  border-top: 1px solid var(--border-color-primary);
}

/* ---- Media polish -------------------------------------------------- */
.gl-card video, .gl-card img { border-radius: 12px; }
"""

HERO_HTML = """
<div class="gl-hero">
  <span class="gl-eyebrow">AI Soccer Analytics</span>
  <h1>⚽ GameLens AI</h1>
  <p>Computer Vision · Data Science · LLM-powered soccer analytics. Upload a short
  clip and get player tracking, structured metrics, charts, and grounded AI
  insights &mdash; all in one place.</p>
  <div class="gl-pills">
    <span class="gl-pill">YOLO + ByteTrack</span>
    <span class="gl-pill">Data-science metrics</span>
    <span class="gl-pill">LangGraph · GPT-4o mini</span>
    <span class="gl-pill">Annotated video</span>
  </div>
</div>
"""

NOTICE_HTML = (
    '<div class="gl-notice">No <code>OPENAI_API_KEY</code> detected &mdash; AI '
    "text uses a built-in fallback. Add a key to enable the GPT-4o mini agents.</div>"
)

FOOTER_HTML = (
    '<div class="gl-footer">Built with YOLOv8 · ByteTrack · OpenCV · Pandas · '
    "LangGraph · Gradio<br/>Movement is measured in image pixels, and insights are "
    "grounded only in the extracted metrics.</div>"
)

METRICS_PLACEHOLDER = (
    "_Load the fast sample demo or run real YOLO analysis — your match metrics will appear here._"
)
REPORT_PLACEHOLDER = (
    "_The cached or generated match report will appear here after analysis._"
)
ANSWER_PLACEHOLDER = "_Load the fast sample demo or analyze a clip, then ask a question to see the answer here._"
STATUS_PLACEHOLDER = "_Ready._"



def _has_openai_key() -> bool:
    """Check Space/local env vars without importing dotenv during UI startup."""
    return bool(os.getenv("OPENAI_API_KEY", "").strip())


def _metrics_to_markdown(m: dict) -> str:
    """Render metrics without importing pandas-heavy analytics at startup."""
    lines = [
        "### Match Metrics",
        f"- **Duration:** {m.get('video_duration_seconds', 0)} s "
        f"({m.get('frames_processed', 0)} frames processed)",
        f"- **Players visible on screen:** ~{m.get('avg_players_on_screen', 0)} on "
        f"average, peak {m.get('peak_players_on_screen', 0)} "
        "_(realistic count)_",
        f"- **Distinct stable tracks:** {m.get('distinct_stable_tracks', 0)} "
        "_(upper bound - one player can span several IDs after occlusion)_",
        f"- **Raw track IDs created:** {m.get('raw_track_ids_created', 0)} "
        "_(tracker artifact - includes brief detections & anyone in view)_",
        f"- **Most active track segment:** {m.get('most_active_track_segment', 'N/A')}",
        f"- **Highest activity zone:** {m.get('highest_activity_zone', 'N/A')}",
        f"- **Most active time segment:** {m.get('most_active_time_segment', 'N/A')}",
        f"- **Player count trend:** {m.get('player_count_trend', 'unknown')}",
        f"- **Overall movement level:** {m.get('activity_level', 'Unknown')} "
        f"(~{m.get('avg_pixel_speed_px_per_s', 0)} px/s average pixel movement)",
    ]
    zones = m.get("activity_by_zone", {})
    if zones:
        z = ", ".join(f"{k}: {v}" for k, v in zones.items())
        lines.append(f"- **Field activity by zone (pixels):** {z}")
    roi, mtf = m.get("field_roi_top_ratio"), m.get("min_track_frames")
    if roi is not None or mtf is not None:
        lines.append(
            f"- **Filters:** field ROI top ratio = {roi}, min track frames = {mtf}")
    return "\n".join(lines)


def _sample_error(message: str):
    return (None, f"**Status:** {message}", "", "", [], "", None, None)


def _display_path(path: Path) -> str:
    try:
        return str(path.relative_to(PROJECT_ROOT))
    except ValueError:
        return str(path)


def _sample_video_preview_path() -> tuple[str | None, str | None]:
    """Return a preview path when the sample MP4 is present and materialized."""
    if not SAMPLE_VIDEO_PATH.exists():
        return None, (
            "Sample video preview is unavailable because "
            "sample_videos/default_soccer_clip.mp4 is missing."
        )
    try:
        size = SAMPLE_VIDEO_PATH.stat().st_size
    except OSError:
        return None, "Sample video preview is unavailable right now."
    if size <= SAMPLE_VIDEO_POINTER_MAX_BYTES:
        return None, (
            "Sample video preview appears to be a storage pointer rather than "
            "the materialized MP4. Cached metrics/charts/report are still shown."
        )
    return str(SAMPLE_VIDEO_PATH), None


def load_cached_sample_demo():
    """Return the fast sample demo without queueing or heavy imports.

    This function only reads cached JSON/CSV/TXT/PNG files and optionally
    returns the sample MP4 path as a preview. It must not import or call the
    real YOLO/OpenCV/LangGraph pipeline.
    """
    required = [SAMPLE_METRICS_PATH, SAMPLE_TRACKING_CSV_PATH,
                SAMPLE_REPORT_PATH, *SAMPLE_CHART_PATHS]
    missing = [_display_path(p) for p in required if not p.exists()]
    if missing:
        return _sample_error(
            "The cached sample outputs are unavailable. Missing: "
            f"{', '.join(missing)}. Upload a video, or reprocess the sample "
            "with YOLO after the sample MP4 is available."
        )

    try:
        metrics = json.loads(SAMPLE_METRICS_PATH.read_text(encoding="utf-8"))
        report_md = SAMPLE_REPORT_PATH.read_text(encoding="utf-8")
    except Exception as exc:
        return _sample_error(f"Could not load cached sample outputs: {exc}")

    preview_path, preview_warning = _sample_video_preview_path()
    status = "Using cached sample outputs. No YOLO processing."
    if preview_warning:
        status = f"{status} {preview_warning}"

    metrics_md = (
        "**Cached sample demo:** showing precomputed sample metrics, charts, "
        "report text, and downloads. Uploaded videos and sample reprocessing "
        "use the real YOLO + ByteTrack pipeline.\n\n"
        + _metrics_to_markdown(metrics)
    )
    return (
        preview_path,
        status,
        metrics_md,
        report_md,
        [str(p) for p in SAMPLE_CHART_PATHS],
        "",
        metrics,
        [str(SAMPLE_TRACKING_CSV_PATH), str(SAMPLE_METRICS_PATH)],
    )


def _run_real_analysis(selected_video_path, frame_skip, max_width, confidence,
                       field_roi_top_ratio, min_track_frames):
    """Run the real CV pipeline. Heavy imports stay inside this path."""
    from src.analytics import (
        build_tracking_dataframe,
        compute_metrics,
        save_metrics_json,
        save_tracking_csv,
    )
    from src.detect_track import run_detection_and_tracking
    from src.langgraph_agents import generate_report
    from src.utils import (
        ANNOTATED_VIDEO_NAME,
        METRICS_JSON_NAME,
        REPORT_TXT_NAME,
        TRACKING_CSV_NAME,
        TrackingConfig,
        ensure_output_dir,
        output_path,
    )
    from src.visualizations import generate_all_charts

    ensure_output_dir()
    config = TrackingConfig(
        frame_skip=int(frame_skip),
        max_width=int(max_width),
        confidence=float(confidence),
        field_roi_top_ratio=float(field_roi_top_ratio),
        min_track_frames=int(min_track_frames),
    )

    annotated_path = output_path(ANNOTATED_VIDEO_NAME)
    rows, info = run_detection_and_tracking(
        selected_video_path, annotated_path, config)

    df = build_tracking_dataframe(rows, info["processed_width"],
                                  config.min_track_frames)
    csv_path = save_tracking_csv(df, output_path(TRACKING_CSV_NAME))

    metrics = compute_metrics(df, info)
    json_path = save_metrics_json(metrics, output_path(METRICS_JSON_NAME))

    charts = generate_all_charts(df, metrics, info)

    report_md = generate_report(metrics)["report"]
    with open(output_path(REPORT_TXT_NAME), "w", encoding="utf-8") as f:
        f.write(report_md)

    return (
        annotated_path,
        "Finished real YOLO processing. First run may take longer because model weights may download.",
        _metrics_to_markdown(metrics),
        report_md,
        charts,
        "",
        metrics,
        [csv_path, json_path],
    )


def real_processing_status(video_path, reprocess_sample):
    if video_path or reprocess_sample:
        return (
            "Running real YOLO processing. First run may take longer because "
            "model weights may download."
        )
    return "Upload a video, click Load Fast Sample Demo, or enable sample reprocessing."


def analyze_real_video(video_path, reprocess_sample, frame_skip, max_width,
                       confidence, field_roi_top_ratio, min_track_frames):
    """Queued real YOLO + ByteTrack path for uploads or explicit reprocessing."""
    try:
        if video_path:
            return _run_real_analysis(
                str(video_path), frame_skip, max_width, confidence,
                field_roi_top_ratio, min_track_frames)

        if reprocess_sample:
            if not SAMPLE_VIDEO_PATH.exists():
                return _sample_error(
                    "The sample soccer clip is not available. Upload your own "
                    "video, or add sample_videos/default_soccer_clip.mp4 to this app."
                )
            return _run_real_analysis(
                str(SAMPLE_VIDEO_PATH), frame_skip, max_width, confidence,
                field_roi_top_ratio, min_track_frames)

        return _sample_error(
            "Upload a video, click Load Fast Sample Demo, or enable Reprocess "
            "sample with YOLO."
        )

    except ValueError as exc:
        return _sample_error(f"Real YOLO processing could not continue: {exc}")
    except Exception as exc:  # unexpected - log to console, show short message
        traceback.print_exc()
        return _sample_error(f"Real YOLO processing failed: {exc}")


def ask_question(question, metrics):
    """Answer a follow-up question from the stored metrics."""
    if not metrics:
        return "Please analyze a video first, then ask a question."
    if not (question or "").strip():
        return "Please type a question."
    try:
        from src.langgraph_agents import answer_question

        return answer_question(metrics, question)
    except Exception as exc:  # never crash the UI on a bad question
        return f"Could not answer right now: {exc}"


# Gradio 6 moved `theme`/`css` from Blocks() to launch(); older versions keep
# them on Blocks(). Detect which one applies so styling works on both.
_LAUNCH_ACCEPTS_THEME = "theme" in inspect.signature(gr.Blocks.launch).parameters


def _theme_css_for_launch() -> dict:
    return {"theme": THEME, "css": CUSTOM_CSS} if _LAUNCH_ACCEPTS_THEME else {}


def _theme_css_for_blocks() -> dict:
    return {} if _LAUNCH_ACCEPTS_THEME else {"theme": THEME, "css": CUSTOM_CSS}


def build_ui() -> gr.Blocks:
    with gr.Blocks(title="GameLens AI", **_theme_css_for_blocks()) as demo:
        gr.HTML(HERO_HTML)
        if not _has_openai_key():
            gr.HTML(NOTICE_HTML)

        metrics_state = gr.State()

        with gr.Row(equal_height=False):
            with gr.Column(scale=1):
                with gr.Group(elem_classes="gl-card"):
                    gr.HTML('<div class="gl-card-title gl-step">'
                            '<span class="gl-step-num">1</span> Upload clip</div>')
                    video_in = gr.Video(label="Soccer clip (.mp4, ~30-90s)")
                    gr.Markdown(
                        "Fast sample demo uses cached outputs and does not run YOLO. "
                        "Real YOLO processing can take several minutes on Hugging Face CPU."
                    )
                    fast_sample_btn = gr.Button(
                        "Load Fast Sample Demo", variant="primary", size="lg",
                        elem_classes="gl-cta")
                    with gr.Accordion("Advanced options", open=False):
                        reprocess_sample = gr.Checkbox(
                            label="Reprocess sample with YOLO",
                            value=False,
                            info="Slower on CPU; runs the same real pipeline used for uploaded clips.",
                        )
                        frame_skip = gr.Slider(
                            1, 10, value=8, step=1,
                            label="Process every Nth frame",
                            info="Higher = faster on CPU, but less smooth tracking")
                        max_width = gr.Slider(
                            320, 1280, value=640, step=32,
                            label="Max frame width",
                            info="Smaller = faster on CPU; increase for higher quality")
                        confidence = gr.Slider(
                            0.1, 0.7, value=0.3, step=0.05,
                            label="Detection confidence",
                            info="Lower detects more players (and more false positives)")
                        field_roi = gr.Slider(
                            0.0, 0.8, value=0.35, step=0.05,
                            label="Field area top ratio",
                            info="Ignore detections above this fraction of the frame "
                                 "(crops out stands / spectators)")
                        min_track = gr.Slider(
                            1, 60, value=10, step=1,
                            label="Min frames for a stable track",
                            info="A field track segment must appear in at least this "
                                 "many processed frames")
                    analyze_btn = gr.Button(
                        "Analyze Uploaded Video / Reprocess with YOLO",
                        variant="secondary", size="lg")
            with gr.Column(scale=1):
                with gr.Group(elem_classes="gl-card"):
                    gr.HTML('<div class="gl-card-title gl-step">'
                            '<span class="gl-step-num">2</span> Annotated video</div>')
                    video_out = gr.Video(label="Video preview / annotated output", interactive=False)

        with gr.Group(elem_classes="gl-card"):
            gr.HTML('<div class="gl-card-title">Status</div>')
            status_md = gr.Markdown(STATUS_PLACEHOLDER)

        with gr.Group(elem_classes="gl-card"):
            gr.HTML('<div class="gl-card-title">Match metrics</div>')
            metrics_md = gr.Markdown(METRICS_PLACEHOLDER)

        with gr.Group(elem_classes="gl-card"):
            gr.HTML('<div class="gl-card-title">Visual analytics</div>')
            gallery = gr.Gallery(
                show_label=False, columns=2, height=460, object_fit="contain")

        with gr.Group(elem_classes="gl-card"):
            gr.HTML('<div class="gl-card-title">AI match report</div>')
            report_md = gr.Markdown(REPORT_PLACEHOLDER)

        with gr.Group(elem_classes="gl-card"):
            gr.HTML('<div class="gl-card-title">Download data</div>')
            files_out = gr.File(
                label="Tracking CSV + metrics JSON", interactive=False)

        with gr.Group(elem_classes="gl-card"):
            gr.HTML('<div class="gl-card-title">Ask about this clip</div>')
            with gr.Row(equal_height=True):
                question = gr.Textbox(
                    show_label=False, scale=5, lines=1,
                    placeholder="e.g. Which track segment moved the most? Which side had more activity?")
                ask_btn = gr.Button("Ask", variant="secondary", scale=1)
            gr.Examples(
                examples=[
                    "Which track segment moved the most?",
                    "Which side had the most activity?",
                    "How many players were tracked?",
                    "When was the most active moment?",
                    "Was the clip high or low activity?",
                ],
                inputs=question,
                label="Try one of these",
            )
            answer_md = gr.Markdown(ANSWER_PLACEHOLDER, elem_classes="gl-answer")

        gr.HTML(FOOTER_HTML)

        analysis_outputs = [
            video_out, status_md, metrics_md, report_md, gallery, answer_md,
            metrics_state, files_out,
        ]
        fast_sample_btn.click(
            load_cached_sample_demo,
            inputs=None,
            outputs=analysis_outputs,
            queue=False,
        )
        analyze_event = analyze_btn.click(
            real_processing_status,
            inputs=[video_in, reprocess_sample],
            outputs=[status_md],
            queue=False,
        )
        analyze_event.then(
            analyze_real_video,
            inputs=[
                video_in, reprocess_sample, frame_skip, max_width, confidence,
                field_roi, min_track,
            ],
            outputs=analysis_outputs,
            queue=True,
        )
        ask_btn.click(
            ask_question,
            inputs=[question, metrics_state],
            outputs=[answer_md],
        )
        return demo


demo = build_ui()
print("Gradio demo built")
print("Fast cached sample route ready")


def launch_app(**kwargs):
    """Launch the top-level demo for local runs."""
    launch_kwargs = _theme_css_for_launch()
    if "ssr_mode" in inspect.signature(gr.Blocks.launch).parameters:
        launch_kwargs["ssr_mode"] = False
    launch_kwargs.update(kwargs)
    return demo.launch(**launch_kwargs)


if __name__ == "__main__":
    launch_app()

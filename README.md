---
title: GameLens AI
emoji: ⚽
colorFrom: green
colorTo: blue
sdk: gradio
app_file: app.py
pinned: false
---

# ⚽ GameLens AI

**Computer Vision, Data Science, and LLM-Powered Soccer Analytics**

GameLens AI is a practical, end-to-end soccer video analytics app. Upload a
short match clip and it detects and tracks players with computer vision,
converts the video into a structured tracking dataset, computes simple
data-science metrics and charts, and uses a small **LangGraph** multi-agent
workflow (**GPT-4o Mini**) to produce grounded, coach-style insights and
answer questions — all through a **Gradio** UI that runs locally or on
**Hugging Face Spaces**.

> The design principle: **Computer Vision extracts data → Data Science
> analyzes it → LLM agents explain it.** The LLM never "watches" the video;
> it only reasons over the structured metrics the pipeline produces.

---

## Overview

Raw sports video is hard to analyze by hand. GameLens AI turns a short clip
into structured player-movement data and then into simple, understandable
insights. It is intentionally an **MVP**: reliable and readable rather than
an over-engineered research system.

## Features

- **Video upload** through a Gradio interface, plus an optional bundled sample clip.
- **Player detection & tracking** with YOLOv8 (nano) + ByteTrack.
- **Spectator filtering (field ROI)** — detections in the top of the frame
  (the stands) are ignored, so crowds don't inflate the counts.
- **Honest counts & neutral labels** — separates raw tracker IDs and *distinct
  stable tracks* (both over-count) from the realistic **players-on-screen**
  estimate (peak / average concurrent), and labels stable tracks as neutral
  **track segments** (Track Segment 1, 2, …) rather than implying real
  identities.
- **Annotated H.264 video** (browser-playable via bundled ffmpeg) with
  bounding boxes, raw track IDs, a field-ROI line, and a live field count.
- **Structured dataset** (`tracking_data.csv`) with frame, timestamp,
  track ID, bounding-box coordinates, center point, box size, zone, per-frame
  movement in pixels, plus `is_stable` and `display_id` (clean label).
- **Data-science metrics**: duration, raw track IDs created, distinct stable
  tracks, estimated / avg / peak players on screen, most active track segment,
  activity by zone, player-count trend, most active time segment, and an
  overall movement level.
- **Charts**: field activity by zone, top active field track segments (clean
  labels), field players on screen over time, and a movement heatmap.
- **LangGraph agents** (GPT-4o Mini): Data Analyst, Sports Insight, and QA.
- **Grounded Q&A** answered only from the computed metrics.
- **Runs without an API key** — LLM features degrade to clear, rule-based
  fallback text instead of crashing.

## Tech Stack

| Layer | Tool |
|---|---|
| Language | Python |
| Computer Vision | OpenCV |
| Detection | Ultralytics YOLO (`yolov8n.pt`) |
| Tracking | ByteTrack (via Ultralytics) |
| Data | Pandas, NumPy |
| Charts | Matplotlib |
| LLM Orchestration | LangGraph |
| LLM Provider | OpenAI API (`gpt-4o-mini`) |
| UI | Gradio |
| Deployment | Hugging Face Spaces |

## Architecture

```text
User uploads or selects sample soccer video (Gradio)
        │
        ▼
OpenCV read + resize + frame skip     ── src/video_processor.py
        │
        ▼
YOLO detection + ByteTrack tracking   ── src/detect_track.py
        │
        ├── annotated_video.mp4
        ▼
Structured tracking rows → CSV        ── src/analytics.py
        │
        ▼
Data-science metrics → metrics.json   ── src/analytics.py
        │
        ├────────────► Charts (PNG)   ── src/visualizations.py
        ▼
LangGraph agents (GPT-4o Mini)        ── src/langgraph_agents.py + prompts.py
   Data Analyst → Sports Insight → QA
        │
        ▼
Report + Q&A shown in Gradio UI       ── app.py
```

## How It Works

1. **Read** the clip with OpenCV; optionally shrink wide frames and process
   every Nth frame for speed.
2. **Track** the `person` class with YOLOv8n + ByteTrack.
3. **Filter spectators (field ROI)** — YOLO detects *all* humans, including
   people in the stands. Detections whose center is above
   `field_roi_top_ratio × frame_height` (the top of the frame) are dropped, so
   only field-area detections are kept.
4. **Structure** the field detections into a tidy CSV, adding a
   left/middle/right `zone` and per-frame `movement_pixels`.
5. **Find stable track segments** — a track ID that appears in at least
   `min_track_frames` processed frames is a *stable field track segment*; each
   gets a neutral sequential label (`display_id` = Track Segment 1, 2, …).
   Short-lived / flickering IDs are excluded from segment-level metrics.
6. **Analyze** with Pandas/NumPy → `metrics.json`. Segment-level metrics use
   stable field tracks; spatial metrics (zone, heatmap) use all field
   detections; both raw and stable counts are reported.
7. **Visualize** the metrics as Matplotlib charts.
8. **Explain** the metrics with three LangGraph agents that only see the
   structured numbers, then answer user questions from the same metrics.

### How many players? (raw IDs vs. stable tracks vs. on-screen)

A tracker creates **far more IDs than there are players**. YOLO detects
spectators too, and — crucially — ByteTrack assigns a **new ID whenever a
player is occluded or leaves and re-enters the frame** (there is no
re-identification, which is intentionally out of scope). So even *after*
filtering, the count of distinct tracks over-counts real players. GameLens AI
reports three numbers so nothing is misleading:

| Metric | Meaning | Use it as… |
|---|---|---|
| `raw_track_ids_created` | Every ID the tracker made (incl. the stands) | not a headcount |
| `distinct_stable_tracks` | Field IDs that persisted ≥ `min_track_frames` | an **upper bound** |
| `estimated_players_on_screen` (+ `avg_players_on_screen`) | Peak / average players tracked **at the same time** | the **realistic** count |

Example: a clip may show `90` raw IDs and `56` distinct stable tracks, but only
`~12` players on screen on average (peak `15`) — that ~12–15 is the honest
figure, and the AI report is grounded to say so.

This is **pixel-based tracking, not real-world player identity** — labels like
Track Segment 1 are stable *within a clip* only, are not confirmed players, and
are not jersey numbers.

## LangGraph Agent Workflow

| Agent | Responsibility |
|---|---|
| **Data Analyst** | Explains the metrics in plain language. |
| **Sports Insight** | Turns metrics into short, hedged, coach-style notes. |
| **QA** | Answers a user question using only the metrics. |

The report graph runs `Data Analyst → Sports Insight`. A separate one-node
QA graph answers follow-up questions so they don't re-run the whole report.
Prompts live in [`src/prompts.py`](src/prompts.py) and keep the model
grounded (no invented names, scores, teams, or real-world distances). If the
metrics can't answer a question, the QA agent says the MVP does not measure
that yet.

## Folder Structure

```text
gamelens-ai/
├── app.py                     # Gradio app (entry point)
├── requirements.txt           # Python dependencies
├── packages.txt               # apt packages for Hugging Face Spaces (OpenCV libs)
├── README.md
├── PROGRESS.md                # Running handoff / progress log
├── .gitignore
├── .env.example
├── src/
│   ├── __init__.py
│   ├── video_processor.py     # OpenCV read/write helpers
│   ├── detect_track.py        # YOLO + ByteTrack tracking
│   ├── analytics.py           # CSV + metrics
│   ├── visualizations.py      # Matplotlib charts
│   ├── langgraph_agents.py    # 3-agent LangGraph workflow
│   ├── prompts.py             # LLM prompt templates
│   └── utils.py               # Shared config + helpers
├── sample_videos/             # Local sample clip + notes
│   ├── README.md
│   └── default_soccer_clip.mp4 # Uploaded separately to HF Space storage
├── outputs/                   # Generated artifacts (git-ignored)
│   └── .gitkeep
└── notebooks/
    └── README.md
```

## How to Run Locally

**Python 3.11–3.13 recommended** (verified on 3.13). Some Python 3.10 Windows
wheel combinations of this ML stack can be unstable — see Troubleshooting.

```bash
# 1) Create and activate a virtual environment (use Python 3.13 if available)
py -3.13 -m venv .venv          # Windows;  macOS/Linux: python3.13 -m venv .venv
# Windows (PowerShell). If activation is blocked by the execution policy, run
# once:  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
.venv\Scripts\Activate.ps1
# macOS / Linux:
source .venv/bin/activate

# 2) Install requirements
pip install -r requirements.txt

# 3) Copy the example env file and add your key (optional)
cp .env.example .env
#   then edit .env and set OPENAI_API_KEY=...   (leave blank to use fallback)

# 4) Run the app
python app.py
```

Open the local URL that Gradio prints (usually `http://127.0.0.1:7860`),
upload a short soccer clip or select **Use sample soccer clip**, then click **Analyze Video**.

> The first run downloads the small `yolov8n.pt` weights automatically.
> Without an `OPENAI_API_KEY`, the app still works and shows grounded,
> rule-based text instead of GPT-4o Mini output.

## Deployment on Hugging Face Spaces

1. **Create a new Space** at <https://huggingface.co/new-space>.
2. **Select the Gradio SDK.**
3. **Push the repo files** (`app.py`, `requirements.txt`, `packages.txt`,
   `README.md`, `src/`, `sample_videos/README.md`, `notebooks/README.md`, and
   `outputs/.gitkeep`). Do **not** commit `.mp4` videos as normal Git blobs.
4. **Upload the sample video separately** with Hugging Face Hub/Xet-compatible
   storage so it lands at the same path the app expects:

   ```bash
   hf upload abhinavvathadi/gamelens-ai sample_videos/default_soccer_clip.mp4 sample_videos/default_soccer_clip.mp4 --repo-type space
   ```

   Locally and on the Space, the sample video path remains
   `sample_videos/default_soccer_clip.mp4`.
5. **Add your secret**: in the Space -> *Settings* -> *Variables and secrets*,
   add `OPENAI_API_KEY` (and optionally `OPENAI_MODEL=gpt-4o-mini`).
6. **Run the app.** The Space builds automatically; then upload a short clip
   and analyze. Users without their own clip can select **Use sample soccer clip**.

> Notes: `packages.txt` installs the system libraries OpenCV needs on a
> headless Space (`libgl1`, `libglib2.0-0`). CPU Spaces work for short clips —
> increase *frame skip* and lower *max width* in the Advanced options if
> processing is slow.

## Troubleshooting

- **Random crash / weird error during `import` (e.g. `re.error: bad escape`,
  `'str' object is not callable`, `internal error in regular expression
  engine`, or exit code `-1073741819`)** → a native library binary conflict in
  that particular environment, not a code bug. It is intermittent, so simply
  **re-run** `python app.py`. If it happens often, use **Python 3.13** (most
  stable here) and recreate the venv from scratch:
  `py -3.13 -m venv .venv` then `pip install -r requirements.txt`.
- **PowerShell: "running scripts is disabled on this system"** when activating
  the venv → run once: `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned`.
- **App crashes on startup on Windows with exit code `-1073741819`
  (0xC0000005)** → two OpenCV builds are installed at the same time. Keep only
  one: `pip uninstall -y opencv-python-headless` (leaves `opencv-python`). The
  pinned `requirements.txt` already avoids this; it only happens if the headless
  build was installed separately.
- **A one-off import error mentioning `packaging` right after installing** →
  usually a half-finished install; just run `python app.py` again.
- **Tracking error mentioning `lap`** → `pip install lap` (already pinned in
  `requirements.txt`).
- **`cv2` import error about `libGL.so.1` on a server/Space** → make sure
  `packages.txt` (with `libgl1` and `libglib2.0-0`) is present in the repo.
- **`OpenH264`/`libopenh264` VideoWriter errors, or the annotated video won't
  play** → the app now writes H.264 via the bundled `imageio-ffmpeg` (no system
  ffmpeg or OpenH264 needed). Make sure `imageio` and `imageio-ffmpeg` are
  installed (`pip install -r requirements.txt`). It falls back to OpenCV `mp4v`
  only if imageio is missing.

## Limitations

- This is **pixel-based tracking, not real-world player identity tracking**.
  Labelled entities are **track segments** (Track Segment 1, 2, …), not
  confirmed players — one player can span several segments.
- **The AI report describes movement / positioning only.** Because the MVP
  tracks neither the ball nor teams, it does not (and is instructed not to)
  claim possession, passes, shots, attacking / defending, "key plays", or team
  strategy — high movement can only indicate higher movement / positioning
  activity.
- **Neither the raw ID count nor the distinct-track count is a headcount.**
  Both over-count; use `estimated_players_on_screen` / `avg_players_on_screen`
  for "how many players".
- **ByteTrack IDs may skip numbers or switch** after occlusion or camera cuts,
  so one real player can span several IDs — this is why `distinct_stable_tracks`
  is only an upper bound (no re-identification is done, by design).
- **Spectator filtering is a simple horizontal ROI**, not true field detection.
  If the camera angle is unusual (e.g. players high in the frame), tune the
  *Field area top ratio*; some crowd members near the field edge may still slip
  through.
- Movement is measured in **image pixels, not real-world meters**.
- The system does **not** identify real players by name or jersey number.
- **No accurate ball tracking**, pass detection, xG, or formation analysis
  in this MVP.
- Tactical insights are **approximate** and based only on extracted metrics.
- The **LLM does not watch the video**; it reasons over the metrics only.
- Performance depends on video quality, camera angle, and hardware.

## Future Improvements

- Near-real-time / live-like mode (webcam or looping clip).
- Team classification via jersey-color clustering.
- Richer heatmaps and player movement trails.
- Optional ball detection.
- LangSmith tracing for the agent workflow.

## Resume Bullet

> Built **GameLens AI**, a Hugging Face-deployed soccer video analytics app
> using YOLOv8, ByteTrack, OpenCV, Pandas, and LangGraph to convert match
> clips into structured player-tracking data, generate movement and
> zone-based analytics with charts, and produce grounded, GPT-4o Mini
> coach-style insights and Q&A through a Gradio interface.

## License

Provided for educational and portfolio use. Review the licenses of the
underlying dependencies (Ultralytics YOLO, etc.) before any commercial use.

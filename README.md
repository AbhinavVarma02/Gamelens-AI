# ⚽ GameLens AI

**Computer Vision, Data Science, and LLM-Powered Soccer Analytics**

GameLens AI is an end-to-end soccer video analytics app that turns short match clips into structured tracking data, visual analytics, and grounded AI-generated insights.

The app uses **YOLOv8 + ByteTrack** for player detection and tracking, **OpenCV + Pandas** for video/data processing, **Matplotlib** for visual analytics, and a **LangGraph + GPT-4o-mini** workflow for reporting and Q&A.

> **Core idea:** Computer Vision extracts data → Data Science analyzes it → LLM agents explain it.  
> The LLM does not watch the video. It only reasons over structured metrics produced by the pipeline.

## Live Demo

[Launch GameLens AI on Hugging Face Spaces](https://huggingface.co/spaces/abhinavvathadi/gamelens-ai)

## Demo Screenshots

<img width="1086" height="883" alt="image" src="https://github.com/user-attachments/assets/17c69d98-9cce-4ddc-b349-02d0d94ffd5c" />

<img width="1119" height="665" alt="image" src="https://github.com/user-attachments/assets/53195efe-5c8f-44d3-aa14-28b3b6413242" />

<img width="1084" height="579" alt="image" src="https://github.com/user-attachments/assets/a2cc7ea4-7834-4662-825e-dea93832a0a5" />

<img width="1062" height="534" alt="image" src="https://github.com/user-attachments/assets/b63846f1-74b0-40d5-b3dc-8a5cbf02867d" />

<img width="934" height="867" alt="image" src="https://github.com/user-attachments/assets/e1e82ead-b1b1-4af5-926e-60b0515724a9" />



## Demo Results

On the bundled 20-second soccer clip, GameLens AI produced:

| Output | Result |
|---|---:|
| Processed frames | 400 |
| Tracking rows | 4,718 |
| Raw tracker IDs | 90 |
| Stable track segments | 56 |
| Average on-screen detections | ~11.5 |
| Peak on-screen detections | 15 |
| Visual analytics charts | 4 |
| Exports | CSV + JSON |

The sample demo uses cached outputs on Hugging Face CPU Spaces so visitors can see results quickly. Uploaded videos still run the real YOLO + ByteTrack pipeline.

## Features

- **Soccer video upload** through a Gradio interface
- **Fast sample demo** using cached outputs for CPU-friendly Hugging Face deployment
- **Player detection and tracking** with YOLOv8 nano + ByteTrack
- **Field ROI filtering** to reduce spectator/crowd detections
- **Stable track filtering** to reduce short-lived tracker noise
- **H.264 annotated video output** for uploaded clips and reprocessed samples
- **Structured tracking exports** as CSV and JSON
- **Data-science metrics** for movement, zones, track segments, and player-count trends
- **Four visual analytics charts**
  - field activity by zone
  - top active track segments
  - on-screen detections over time
  - movement heatmap
- **LangGraph reporting workflow** with Data Analyst, Sports Insight, and Q&A agents
- **Grounded Q&A** answered only from computed metrics
- **No-key fallback** so the app still works without an OpenAI API key

## Tech Stack

| Layer | Tools |
|---|---|
| UI | Gradio |
| Deployment | Hugging Face Spaces |
| Computer Vision | OpenCV |
| Detection | Ultralytics YOLOv8 |
| Tracking | ByteTrack |
| Data Processing | Pandas, NumPy |
| Visualization | Matplotlib |
| LLM Workflow | LangGraph |
| LLM Provider | OpenAI GPT-4o-mini |
| Video Encoding | imageio-ffmpeg |

## Architecture

```text
User uploads video or loads cached sample demo
        │
        ▼
Gradio UI
        │
        ├── Fast sample path
        │       └── sample_outputs/
        │           ├── cached metrics
        │           ├── cached charts
        │           ├── cached report
        │           └── CSV/JSON downloads
        │
        └── Real YOLO path
                │
                ▼
        OpenCV video read + resize + frame skip
                │
                ▼
        YOLOv8 person detection + ByteTrack tracking
                │
                ├── H.264 annotated video
                ▼
        Structured tracking rows
                │
                ├── tracking_data.csv
                ├── metrics.json
                └── visual analytics charts
                │
                ▼
        LangGraph reporting + Q&A
                │
                ▼
        Grounded report and answers in the UI
```

## How It Works

### 1. Fast Sample Demo

The **Load Fast Sample Demo** button reads precomputed files from `sample_outputs/`.

This path does **not** run YOLO, OpenCV video processing, Matplotlib chart generation, OpenAI, or LangGraph. It is designed so the deployed Hugging Face Space can show a complete demo quickly on CPU.

### 2. Real Video Analysis

For uploaded videos, or when **Reprocess sample with YOLO** is enabled, the app runs the real pipeline:

1. Read the video with OpenCV
2. Resize wide frames for faster CPU processing
3. Process every Nth frame using frame skipping
4. Detect people with YOLOv8
5. Track detections with ByteTrack
6. Filter detections using a field ROI
7. Generate tracking rows, metrics, charts, and annotated video
8. Pass structured metrics to LangGraph agents for report generation and Q&A

## Tracking Metrics Explained

A tracker can create more IDs than the number of real players. This happens because YOLO can detect spectators, and ByteTrack may assign a new ID after occlusion, camera cuts, or a player leaving and re-entering the frame.

GameLens AI reports multiple counts to avoid misleading results:

| Metric | Meaning | Interpretation |
|---|---|---|
| `raw_track_ids_created` | Every ID created by the tracker | Not a headcount |
| `distinct_stable_tracks` | Field-area track IDs that persisted long enough | Upper bound |
| `estimated_players_on_screen` | Peak concurrent on-screen detections | More realistic count |
| `avg_players_on_screen` | Average concurrent on-screen detections | More realistic count |

Example from the sample clip:

```text
Raw tracker IDs: 90
Stable track segments: 56
Average on-screen detections: ~11.5
Peak on-screen detections: 15
```

The app labels tracked entities as **Track Segment 1, Track Segment 2, ...** instead of real player names or jersey numbers. These are tracker labels within a clip, not confirmed player identities.

## LangGraph Agent Workflow

GameLens AI uses a lightweight LangGraph workflow to convert metrics into readable insights.

| Agent | Role |
|---|---|
| Data Analyst | Explains the computed metrics in plain language |
| Sports Insight | Converts metrics into short, careful soccer-style observations |
| QA Agent | Answers user questions using only available metrics |

The prompts are designed to avoid unsupported claims. The AI report does not invent teams, player names, scores, tactics, possession, passes, shots, or ball events.

If no `OPENAI_API_KEY` is provided, the app uses grounded fallback text instead of crashing.

## Folder Structure

```text
gamelens-ai/
├── app.py
├── requirements.txt
├── packages.txt
├── README.md
├── PROGRESS.md
├── .env.example
├── .gitignore
├── src/
│   ├── __init__.py
│   ├── video_processor.py
│   ├── detect_track.py
│   ├── analytics.py
│   ├── visualizations.py
│   ├── langgraph_agents.py
│   ├── prompts.py
│   └── utils.py
├── sample_outputs/
│   ├── README.md
│   ├── sample_metrics.json
│   ├── sample_tracking_data.csv
│   ├── sample_report.txt
│   └── sample_chart_*.png
├── sample_videos/
│   └── README.md
├── outputs/
│   └── .gitkeep
└── notebooks/
    └── README.md
```

## Run Locally

Python 3.11 or later is recommended.

```bash
# 1. Create a virtual environment
python -m venv .venv

# 2. Activate it
# Windows PowerShell
.venv\Scripts\Activate.ps1

# macOS/Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Optional: create a local env file for GPT output
cp .env.example .env

# 5. Run the app
python app.py
```

Then open the local Gradio URL printed in the terminal.

Without an OpenAI API key, the app still runs and uses fallback report/Q&A text.

## Environment Variables

| Variable | Required | Purpose |
|---|---|---|
| `OPENAI_API_KEY` | No | Enables GPT-4o-mini report and Q&A generation |
| `OPENAI_MODEL` | No | Defaults to `gpt-4o-mini` |

Do not commit `.env` or any real API keys.

## Hugging Face Notes

The deployed Space uses:

- `app.py` as the Gradio entry point
- Python 3.11 for deployment stability
- `packages.txt` for Linux/OpenCV system libraries
- cached sample outputs for fast CPU demo behavior
- separate Hugging Face storage for the bundled MP4 sample video

The sample MP4 is not committed as a normal Git blob. It is uploaded separately to Hugging Face Hub/Xet storage at:

```text
sample_videos/default_soccer_clip.mp4
```

Uploaded user videos still run the real YOLO + ByteTrack pipeline and may be slower on Hugging Face CPU Basic.

## Limitations

- This is pixel-based tracking, not real-world player identity tracking.
- Track Segment labels are not jersey numbers or real identities.
- One real player can appear as multiple track segments after occlusion or re-entry.
- Raw tracker IDs and stable track counts are not true player headcounts.
- Movement is measured in image pixels, not meters.
- Field ROI filtering is a simple horizontal filter, not full field segmentation.
- The app does not track the ball.
- The app does not detect passes, shots, possession, formations, xG, or team tactics.
- The LLM does not watch the video. It only explains structured metrics.
- Real video processing speed depends on clip length, resolution, camera angle, and hardware.

## Future Improvements

- Team classification using jersey-color clustering
- Optional ball detection
- Richer movement trails and heatmaps
- Better field segmentation
- Re-identification to reduce track fragmentation
- Near-real-time webcam or live-like mode
- LangSmith tracing for the agent workflow

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

Note: This project uses third-party libraries and models, including Ultralytics YOLO. Users are responsible for reviewing and complying with the licenses of all external dependencies and model weights before commercial use.

# GameLens AI Progress

## Current Status
Working, verified MVP on Python 3.13. Hugging Face Spaces deployment recovery is prepared after normal Git rejected the sample MP4 blob.

The core pipeline is built and mostly working:
- Gradio app entry point: `app.py`
- YOLOv8 + ByteTrack detection/tracking in `src/detect_track.py`
- CSV + metrics generation in `src/analytics.py`
- Matplotlib charts in `src/visualizations.py`
- LangGraph report and Q&A workflow in `src/langgraph_agents.py`
- Safe rule-based fallback text when `OPENAI_API_KEY` is not set

## Recent Fixes Completed
- Tracker labels were changed from player-identity wording to neutral `Track Segment N` labels.
- Metrics now separate realistic on-screen count, distinct stable track segments, and raw tracker IDs.
- AI prompts and fallbacks avoid unsupported claims about real player identity, jersey numbers, ball possession, passes, shots, teams, tactics, or attacking/defending actions.
- Movement is described as pixel-based image movement, not real-world speed or distance.
- Annotated video output prefers H.264 through bundled `imageio-ffmpeg`, with OpenCV `mp4v` as a fallback.
- Dependencies were cleaned up for one OpenCV package, `lap`, `imageio`, and `imageio-ffmpeg`.
- UI emoji cleanup was completed for app UI text: the only app UI emoji is the main heading `GameLens AI` soccer-ball heading.
- Final readiness cleanup removed player-centric Q&A examples/comments/README wording, renamed the top-active chart artifact to `chart_top_track_segments.png`, and prepared generated local artifacts for removal.
- Added an MVP sample-video option: users can upload their own clip or select `Use sample soccer clip`; uploads take priority, and missing sample files return a friendly error.
- Polished AI report/Q&A wording to avoid broad phrases like "dynamic environment" and prefer "field detections visible on screen" where it improves honesty.

## Important Decisions
- This is an MVP for short soccer clips, not a full match-analysis system.
- The LLM never watches the video. It only receives structured metrics JSON.
- `raw_track_ids_created` is a tracker artifact and not a headcount.
- `distinct_stable_tracks` is an upper bound because ByteTrack can fragment one real player into multiple IDs after occlusion or re-entry.
- `estimated_players_on_screen`, `avg_players_on_screen`, and `peak_players_on_screen` are the realistic on-screen count metrics.
- Labelled entities are stable track segments, not confirmed player identities and not jersey numbers.
- Segment movement and all-segments movement are separate so a single segment is not credited with the whole clip's movement.
- Spatial activity can only mean higher movement or positioning activity in image space.
- The app should run without `OPENAI_API_KEY`; with no key it uses deterministic fallback text.
- The bundled sample video path remains `sample_videos/default_soccer_clip.mp4`, but the MP4 must not be committed as a normal Git blob. It should be uploaded separately to the Hugging Face Space with Hub/Xet-compatible storage.
- `.env` is local-only and must not be read, edited, committed, or uploaded.
- Hugging Face Spaces uses `app.py` as the Gradio entry point, declared in the README YAML metadata.

## Files Created or Updated
| File | Purpose | Status |
|---|---|---|
| app.py | Gradio app / entry point | Updated |
| requirements.txt | Python dependencies | Updated |
| packages.txt | Hugging Face apt packages | Created |
| README.md | Setup, deployment, limitations, resume bullet | Updated |
| PROGRESS.md | Handoff / progress log | Updated |
| .gitignore | Ignore secrets, outputs, weights, local media, scope docs | Created |
| .env.example | Placeholder environment config | Created |
| src/__init__.py | Package marker | Created |
| src/utils.py | Shared config, output paths, OpenAI helper | Updated |
| src/video_processor.py | Video metadata and H.264 writer | Updated |
| src/detect_track.py | YOLO + ByteTrack tracking and ROI filtering | Updated |
| src/analytics.py | CSV shaping and metrics | Updated |
| src/visualizations.py | Charts | Updated |
| src/prompts.py | LLM grounding prompts | Updated |
| src/langgraph_agents.py | Report and Q&A workflow with fallbacks | Updated |
| sample_videos/README.md | Local sample-video note | Updated |
| sample_videos/default_soccer_clip.mp4 | Demo clip under 8 MB; upload separately to Hugging Face Space storage, not normal Git | Local only / HF storage |
| notebooks/README.md | Optional notebook note | Created |
| outputs/.gitkeep | Keeps outputs folder in repo | Created |

## Validation Notes
Final review completed on 2026-07-04, then rerun after the sample-video MVP addition:
- `python -m compileall app.py src`: OK
- Safe import/UI smoke with `PYTHON_DOTENV_DISABLED=1` and empty `OPENAI_API_KEY`: OK
- Gradio Blocks build: OK (`43` blocks)
- Friendly callback checks: OK (`analyze_video` with no upload asks for upload or sample selection; missing sample path returns a friendly message)
- Upload priority check: OK (a provided upload path is used even when `Use sample soccer clip` is selected)
- Default sample path check: OK (`sample_videos/default_soccer_clip.mp4` is selected when no upload is provided and the checkbox is selected)
- No-key LLM fallback: OK (`generate_report` used deterministic fallback text in smoke checks)
- Local Gradio launch smoke from previous final review: OK (`http://127.0.0.1:7861/` returned HTTP 200, then server closed)
- Synthetic metrics separation: OK (raw tracker IDs, field IDs, distinct stable track segments, and realistic on-screen count remained separate)
- Dependency check from previous final review: OK (`opencv-python` installed, `opencv-python-headless` absent, `imageio`, `imageio-ffmpeg`, and `lap` installed)
- Sample video file: OK (`sample_videos/default_soccer_clip.mp4`, 7,731,760 bytes)
- `.gitignore` video rule: OK (`sample_videos/*.mp4` is ignored, while `sample_videos/README.md` stays tracked)
- Git tracking check: OK (`sample_videos/default_soccer_clip.mp4` removed from normal Git tracking, local file still exists, and `.gitignore` ignores it)
- Stale metric-key scan: OK for legacy player-centric metric keys
- UI emoji scan: OK; app UI has only the main `GameLens AI` soccer-ball heading
- `.env`: not read or edited; validation imports disabled dotenv loading

Latest deployment recovery:
- Updated `.gitignore` so `.mp4` files, `sample_videos/*.mp4`, generated videos, outputs, weights, caches, `.venv/`, and `.env` remain out of normal Git tracking.
- Updated README and `sample_videos/README.md` to document the separate Hugging Face Hub/Xet sample-video upload at `sample_videos/default_soccer_clip.mp4`.
- Added Hugging Face Space YAML metadata to README with `sdk: gradio`, `app_file: app.py`, and valid Space colors.
- Removed `sample_videos/default_soccer_clip.mp4` from normal Git tracking while keeping the local file in place.
- Verified `python -m compileall app.py src`: OK.
- Verified safe UI import/build with `.venv`, `PYTHON_DOTENV_DISABLED=1`, and empty `OPENAI_API_KEY`: OK (`43` blocks).
- `.env` was not read or edited.

## Known Remaining Limitations
- The app does not perform player re-identification, so one real player can span multiple track segments.
- The app does not identify real players, jersey numbers, teams, formations, tactics, possession, passes, shots, or ball events.
- Movement is measured in image pixels only, not meters or calibrated speed.
- Spectator filtering is a simple horizontal field ROI and may need tuning for unusual camera angles.
- Performance depends on video length, resolution, camera angle, model download time, and CPU/GPU availability.
- Python 3.10 on this Windows environment previously showed intermittent native import crashes; Python 3.13 was more stable.

## Next Steps
- Push source files to Hugging Face Space Git, excluding `.env`, `.venv`, generated outputs, local videos, model weights, caches, and the original scope Markdown. Do not include `sample_videos/default_soccer_clip.mp4` in normal Git.
- Upload `sample_videos/default_soccer_clip.mp4` separately to the Hugging Face Space at the same path with `hf upload abhinavvathadi/gamelens-ai sample_videos/default_soccer_clip.mp4 sample_videos/default_soccer_clip.mp4 --repo-type space`.
- Add `OPENAI_API_KEY` as a Hugging Face Space secret only if GPT output is desired; the app still runs without it.
- Run one real short soccer clip after deployment to eyeball the annotated video, charts, metrics, and fallback/GPT report text.
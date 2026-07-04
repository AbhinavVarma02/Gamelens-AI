"""
detect_track.py
---------------
Player detection and tracking using Ultralytics YOLO + ByteTrack.

What it does:
- Loads a small YOLO model (yolov8n by default) and caches it.
- Reads the uploaded video frame by frame with OpenCV, optionally skipping
  frames and shrinking large frames to keep processing fast.
- Runs YOLO tracking on the 'person' class only.
- Draws bounding boxes + track IDs and writes an annotated output video.
- Returns one row of raw tracking data per detected player per processed
  frame, which analytics.py turns into a CSV and metrics.

Only the 'person' class is tracked for this MVP - no ball, jersey numbers,
or player identities.
"""

from __future__ import annotations

from typing import Optional

import cv2

from .utils import TrackingConfig
from .video_processor import (
    compute_output_size,
    create_video_writer,
    get_video_metadata,
)

# Cache loaded models so we do not reload weights on every request.
_MODEL_CACHE: dict = {}


def load_model(model_path: str):
    """Load (and cache) a YOLO model.

    ultralytics is imported lazily so the rest of the app can be imported
    even in environments where it (or torch) is not installed yet.
    """
    if model_path in _MODEL_CACHE:
        return _MODEL_CACHE[model_path]
    from ultralytics import YOLO

    model = YOLO(model_path)
    _MODEL_CACHE[model_path] = model
    return model


def _color_for_id(track_id: int) -> tuple[int, int, int]:
    """Deterministic BGR color derived from a track id."""
    track_id = int(track_id)
    r = (37 * track_id) % 255
    g = (17 * track_id + 90) % 255
    b = (29 * track_id + 160) % 255
    return int(b), int(g), int(r)


def _draw_box(frame, x1, y1, x2, y2, label, color):
    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
    (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
    cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
    cv2.putText(frame, label, (x1 + 2, y1 - 4), cv2.FONT_HERSHEY_SIMPLEX,
                0.5, (255, 255, 255), 1, cv2.LINE_AA)


def run_detection_and_tracking(
    video_path: str,
    output_video_path: str,
    config: Optional[TrackingConfig] = None,
):
    """Run YOLO tracking over a video and write an annotated copy.

    Detections above the field ROI line (top `field_roi_top_ratio` of the
    frame, i.e. the stands) are ignored so spectators do not inflate the
    metrics.

    Returns (rows, info):
      rows - list of dicts (field detections only): frame, timestamp,
             track_id, x1, y1, x2, y2, x_center, y_center, bbox_width,
             bbox_height
      info - dict: processed_width/height, fps, duration_seconds,
             frames_processed, original_width/height, raw_track_ids_created
             (all ids incl. stands), field_roi_top_ratio, min_track_frames

    Raises ValueError with a friendly message if nothing can be tracked.
    """
    config = config or TrackingConfig()
    meta = get_video_metadata(video_path)

    out_w, out_h = compute_output_size(meta.width, meta.height, config.max_width)
    # Output plays at roughly original speed even though we skip frames.
    out_fps = max(meta.fps / max(config.frame_skip, 1), 1.0)
    writer = create_video_writer(output_video_path, out_fps, (out_w, out_h))

    model = load_model(config.model_path)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        writer.release()
        raise ValueError(
            "Could not open the video file. Please upload a valid .mp4 clip."
        )

    rows = []
    frame_index = -1
    frames_processed = 0
    max_frames = int(config.max_seconds * meta.fps) if meta.fps else 0
    resize_needed = (out_w, out_h) != (meta.width, meta.height)
    # Everything above this y is treated as stands/spectators and ignored.
    roi_y = out_h * max(0.0, min(config.field_roi_top_ratio, 0.95))
    # Every raw tracker id we ever see (incl. stands) - for the "raw" count.
    all_track_ids: set = set()

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            frame_index += 1

            if max_frames and frame_index > max_frames:
                break
            # Frame skipping for speed.
            if frame_index % max(config.frame_skip, 1) != 0:
                continue

            if resize_needed:
                frame = cv2.resize(frame, (out_w, out_h))

            timestamp = frame_index / meta.fps if meta.fps else 0.0

            # persist=True keeps ByteTrack state across the frames we feed it.
            results = model.track(
                frame,
                persist=True,
                classes=[config.person_class_id],
                conf=config.confidence,
                tracker=config.tracker,
                verbose=False,
            )

            frames_processed += 1
            boxes = results[0].boxes
            field_count = 0

            if boxes is not None and boxes.id is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                ids = boxes.id.cpu().numpy().astype(int)
                for (x1, y1, x2, y2), tid in zip(xyxy, ids):
                    all_track_ids.add(int(tid))
                    x1, y1, x2, y2 = float(x1), float(y1), float(x2), float(y2)
                    xc, yc = (x1 + x2) / 2.0, (y1 + y2) / 2.0

                    # Field ROI filter: skip detections above the line
                    # (spectators / stands sit near the top of the frame).
                    if yc < roi_y:
                        continue

                    field_count += 1
                    rows.append({
                        "frame": frame_index,
                        "timestamp": round(timestamp, 3),
                        "track_id": int(tid),
                        "x1": round(x1, 1), "y1": round(y1, 1),
                        "x2": round(x2, 1), "y2": round(y2, 1),
                        "x_center": round(xc, 1), "y_center": round(yc, 1),
                        "bbox_width": round(x2 - x1, 1),
                        "bbox_height": round(y2 - y1, 1),
                    })
                    # Draw the raw tracker id; clean "Track Segment N" labels
                    # are assigned later, only for stable field track segments.
                    _draw_box(frame, int(x1), int(y1), int(x2), int(y2),
                              f"ID {int(tid)}", _color_for_id(tid))

            # Faint line marking the top of the analysed field area.
            cv2.line(frame, (0, int(roi_y)), (out_w, int(roi_y)), (0, 200, 255), 1)
            cv2.putText(frame, "field area below line", (10, max(int(roi_y) - 6, 12)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, (0, 200, 255), 1, cv2.LINE_AA)
            # Frame-level count of field detections on screen.
            cv2.putText(frame, f"On-screen: {field_count}", (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2, cv2.LINE_AA)
            writer.write(frame)
    finally:
        cap.release()
        writer.release()

    if not rows:
        raise ValueError(
            "No field players could be detected in this clip. If players sit low "
            "in the frame, try lowering the 'Field area top ratio'. Otherwise try "
            "a clearer, wider soccer clip."
        )

    info = {
        "processed_width": out_w,
        "processed_height": out_h,
        "fps": meta.fps,
        "duration_seconds": round(meta.duration_seconds, 2),
        "frames_processed": frames_processed,
        "original_width": meta.width,
        "original_height": meta.height,
        # Raw = every tracker id seen, including anyone in the stands (this is
        # the inflated count). Field/stable counts are derived in analytics.
        "raw_track_ids_created": len(all_track_ids),
        "field_roi_top_ratio": round(float(config.field_roi_top_ratio), 3),
        "min_track_frames": int(config.min_track_frames),
    }
    return rows, info

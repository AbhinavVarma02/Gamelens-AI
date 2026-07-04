"""
analytics.py
------------
Turn raw tracking rows into a clean dataset and simple, explainable metrics.

Steps:
- Build a pandas DataFrame from the tracking rows.
- Add 'zone' (left/middle/right third of the frame) and 'movement_pixels'
  (distance each track segment moved since its previous frame).
- Save tracking_data.csv.
- Compute easy-to-explain metrics (activity, zones, counts, trend, time
  segment, overall level) and save metrics.json.

All movement is in image pixels, not real-world meters - a deliberate MVP
simplification.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from .utils import ZONES, classify_zone, format_seconds

# Final CSV column order. `track_id` is the raw tracker id (kept for
# debugging); `display_id` is the clean sequential label (Track Segment 1,
# Track Segment 2, ...) assigned only to stable field track segments.
TRACKING_COLUMNS = [
    "frame", "timestamp", "track_id",
    "x1", "y1", "x2", "y2",
    "x_center", "y_center",
    "bbox_width", "bbox_height",
    "zone", "movement_pixels",
    "is_stable", "display_id",
]

MEASUREMENT_NOTE = (
    "Counts are subtle with tracking. 'raw_track_ids_created' is every ID the "
    "tracker produced (including brief detections and anyone in the stands "
    "within view), so it is NOT a headcount. 'distinct_stable_tracks' counts "
    "field-area IDs that lasted at least 'min_track_frames' frames, but this "
    "still OVER-counts real players because ByteTrack assigns a NEW id to a "
    "player after occlusion or after leaving and re-entering the frame (there "
    "is no re-identification). The realistic number of players is "
    "'estimated_players_on_screen' (the peak number tracked at the same time), "
    "with 'avg_players_on_screen' as the average. Movement is in image pixels, "
    "not meters. Track Segment 1, Track Segment 2, ... are clean labels for "
    "stable track SEGMENTS, not confirmed players (one player can span several "
    "segments) and not jersey numbers or identities. The system does not track "
    "the ball or classify teams, so movement in a zone or by a segment can only "
    "indicate higher movement / positioning activity, not plays, possession, "
    "attacking or defending."
)


def build_tracking_dataframe(rows, frame_width: int,
                             min_track_frames: int = 10) -> pd.DataFrame:
    """Create the tracking DataFrame with zone, movement, stability and clean
    track-segment labels.

    A track id is a "stable field track segment" if it appears in at least
    `min_track_frames` processed frames. Stable ids get clean sequential labels
    (Track Segment 1, Track Segment 2, ...) in `display_id`, ordered by first
    appearance. The raw `track_id` column is always kept for debugging.
    """
    df = pd.DataFrame(rows)
    if df.empty:
        return pd.DataFrame(columns=TRACKING_COLUMNS)

    # Zone from the horizontal position of the detection center.
    df["zone"] = df["x_center"].apply(lambda x: classify_zone(x, frame_width))

    # Movement = distance from this track id's previous center point.
    df = df.sort_values(["track_id", "frame"]).reset_index(drop=True)
    prev_x = df.groupby("track_id")["x_center"].shift()
    prev_y = df.groupby("track_id")["y_center"].shift()
    dx = df["x_center"] - prev_x
    dy = df["y_center"] - prev_y
    df["movement_pixels"] = np.sqrt(dx * dx + dy * dy).fillna(0.0).round(1)

    # Stability: how many processed frames each track id appears in.
    frames_per_id = df.groupby("track_id")["frame"].nunique()
    stable_ids = frames_per_id[frames_per_id >= max(int(min_track_frames), 1)].index
    df["is_stable"] = df["track_id"].isin(stable_ids)

    # Clean sequential labels for stable field track segments, ordered by first
    # appearance (Track Segment 1 = first stable segment to appear).
    first_seen = (
        df[df["is_stable"]].groupby("track_id")["frame"].min().sort_values(kind="stable")
    )
    label_map = {tid: f"Track Segment {i + 1}" for i, tid in enumerate(first_seen.index)}
    df["display_id"] = df["track_id"].map(label_map).fillna("")

    # Tidy row order for the CSV.
    df = df.sort_values(["frame", "track_id"]).reset_index(drop=True)
    return df[TRACKING_COLUMNS]


def save_tracking_csv(df: pd.DataFrame, path: str) -> str:
    df.to_csv(path, index=False)
    return path


def _activity_level(avg_speed_ratio: float) -> str:
    """High/Medium/Low from average player speed expressed as a fraction of the
    frame width per second.

    Using speed (pixels/second) instead of raw per-frame movement makes the
    label independent of resolution AND of fps / frame-skip - otherwise a
    60fps clip with frame-skip looks "Low" only because each step covers a tiny
    slice of time. Thresholds are calibrated for broadcast soccer.
    """
    if avg_speed_ratio <= 0:
        return "Unknown"
    if avg_speed_ratio >= 0.09:
        return "High"
    if avg_speed_ratio >= 0.035:
        return "Medium"
    return "Low"


def _count_trend(df: pd.DataFrame) -> str:
    """Compare average visible players in the first vs second half of the clip."""
    if df.empty:
        return "unknown"
    t_min, t_max = df["timestamp"].min(), df["timestamp"].max()
    if t_max <= t_min:
        return "stable"
    mid = (t_min + t_max) / 2.0
    first = df[df["timestamp"] <= mid].groupby("frame")["track_id"].nunique().mean()
    second = df[df["timestamp"] > mid].groupby("frame")["track_id"].nunique().mean()
    if pd.isna(first) or pd.isna(second):
        return "stable"
    diff = second - first
    if diff > 1:
        return "increasing"
    if diff < -1:
        return "decreasing"
    return "stable"


def compute_metrics(df: pd.DataFrame, info: dict) -> dict:
    """Compute the summary metrics dict (pure Python types, JSON-safe).

    Segment-level metrics (top segments, counts, trend) use STABLE field
    tracks only. Spatial metrics (zone activity, heatmap) use all
    field-filtered detections. Raw, stable, and on-screen counts are reported
    so the LLM can tell tracker artifacts apart from realistic player counts.
    """
    frame_width = int(info.get("processed_width") or 0)
    duration = float(info.get("duration_seconds") or 0.0)
    raw_ids = int(info.get("raw_track_ids_created") or 0)

    metrics = {
        "video_duration_seconds": round(duration, 1),
        "frames_processed": int(info.get("frames_processed") or 0),
        "field_roi_top_ratio": info.get("field_roi_top_ratio"),
        "min_track_frames": info.get("min_track_frames"),
        "raw_track_ids_created": raw_ids,
        "field_track_ids_created": 0,
        # distinct stable track IDs = an UPPER BOUND on players (ID switches
        # inflate it). estimated/avg/peak "on screen" are the realistic counts.
        "distinct_stable_tracks": 0,
        "estimated_players_on_screen": 0,
        "avg_players_on_screen": 0.0,
        "peak_players_on_screen": 0,
        # A "track segment" is one stable tracker id, NOT a confirmed player
        # (one player can span several segments). Raw track_id kept internally.
        "most_active_track_segment": "N/A",
        "most_active_track_segment_id": None,
        "most_active_track_segment_movement_pixels": 0.0,
        "top_active_track_segments": [],
        # Summed over ALL stable segments - NOT a single segment's movement.
        "total_movement_pixels_all_segments": 0.0,
        "activity_by_zone": {z: 0.0 for z in ZONES},
        "highest_activity_zone": "N/A",
        "most_active_time_segment": "N/A",
        "player_count_trend": "unknown",
        "avg_pixel_speed_px_per_s": 0.0,
        "activity_level": "Unknown",
        "measurement_note": MEASUREMENT_NOTE,
    }
    if df.empty:
        if not raw_ids:
            metrics["measurement_note"] = "No players were tracked in this clip."
        return metrics

    # df already holds only field-area detections (spectators removed upstream).
    field_df = df
    metrics["field_track_ids_created"] = int(field_df["track_id"].nunique())

    # Zone activity uses ALL field detections.
    zone_series = (
        field_df.groupby("zone")["movement_pixels"].sum().reindex(ZONES).fillna(0.0)
    )
    metrics["activity_by_zone"] = {z: round(float(v), 1) for z, v in zone_series.items()}
    metrics["highest_activity_zone"] = max(
        metrics["activity_by_zone"], key=metrics["activity_by_zone"].get
    )

    # Segment-level metrics use STABLE field track segments only.
    stable_df = field_df[field_df["is_stable"]]
    metrics["distinct_stable_tracks"] = int(stable_df["track_id"].nunique())

    # Concurrent counts = how many stable segments are visible at the same time.
    # The peak is the realistic estimate of players on screen; distinct tracks
    # over-count because one player can span several segments.
    per_frame_stable = stable_df.groupby("frame")["track_id"].nunique()
    if not per_frame_stable.empty:
        metrics["avg_players_on_screen"] = round(float(per_frame_stable.mean()), 1)
        metrics["peak_players_on_screen"] = int(per_frame_stable.max())
        metrics["estimated_players_on_screen"] = int(per_frame_stable.max())

    if not stable_df.empty:
        movement_by_segment = (
            stable_df.groupby(["track_id", "display_id"])["movement_pixels"]
            .sum().sort_values(ascending=False)
        )
        metrics["top_active_track_segments"] = [
            {
                "segment": disp or f"Track Segment (id {int(tid)})",
                "track_id": int(tid),
                "movement_pixels": round(float(val), 1),
            }
            for (tid, disp), val in movement_by_segment.head(5).items()
        ]
        top_tid, top_disp = movement_by_segment.index[0]
        metrics["most_active_track_segment"] = top_disp or f"Track Segment (id {int(top_tid)})"
        metrics["most_active_track_segment_id"] = int(top_tid)
        metrics["most_active_track_segment_movement_pixels"] = round(
            float(movement_by_segment.iloc[0]), 1)
        metrics["total_movement_pixels_all_segments"] = round(
            float(stable_df["movement_pixels"].sum()), 1)

    # Time segment / activity level / trend: stable if available, else field.
    metric_df = stable_df if not stable_df.empty else field_df

    seg_len = 10 if duration > 40 else 5
    seg = (metric_df["timestamp"] // seg_len).astype(int)
    seg_movement = metric_df.groupby(seg)["movement_pixels"].sum()
    if not seg_movement.empty and seg_movement.max() > 0:
        start = int(seg_movement.idxmax()) * seg_len
        metrics["most_active_time_segment"] = (
            f"{format_seconds(start)} to {format_seconds(start + seg_len)}"
        )

    steps = metric_df.loc[metric_df["movement_pixels"] > 0, "movement_pixels"]
    avg_step = float(steps.mean()) if not steps.empty else 0.0
    # Convert the per-frame step to a speed (px/second) using the real time
    # between processed frames, so the activity level does not depend on
    # fps / frame-skip.
    times = np.sort(metric_df["timestamp"].unique())
    dt = float(np.median(np.diff(times))) if times.size > 1 else 0.0
    avg_speed = (avg_step / dt) if dt > 0 else 0.0
    metrics["avg_pixel_speed_px_per_s"] = round(avg_speed, 1)
    speed_ratio = (avg_speed / frame_width) if frame_width > 0 else 0.0
    metrics["activity_level"] = _activity_level(speed_ratio)
    metrics["player_count_trend"] = _count_trend(metric_df)

    return metrics


def save_metrics_json(metrics: dict, path: str) -> str:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return path


def metrics_to_markdown(m: dict) -> str:
    """Render the metrics as a compact Markdown summary for the UI."""
    lines = [
        "### Match Metrics",
        f"- **Duration:** {m.get('video_duration_seconds', 0)} s "
        f"({m.get('frames_processed', 0)} frames processed)",
        f"- **Players visible on screen:** ~{m.get('avg_players_on_screen', 0)} on "
        f"average, peak {m.get('peak_players_on_screen', 0)} "
        "_(realistic count)_",
        f"- **Distinct stable tracks:** {m.get('distinct_stable_tracks', 0)} "
        "_(upper bound — one player can span several IDs after occlusion)_",
        f"- **Raw track IDs created:** {m.get('raw_track_ids_created', 0)} "
        "_(tracker artifact — includes brief detections & anyone in view)_",
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

"""
visualizations.py
-----------------
Create simple charts from the tracking data and metrics using Matplotlib.

Charts produced:
1. Activity by zone (bar)
2. Top active track segments (horizontal bar)
3. Player count over time (line)
4. Movement heatmap (2D histogram of field detection positions)

Each chart is saved as a PNG in outputs/ and the file paths are returned so
the Gradio gallery can display them. Uses the headless 'Agg' backend so it
works on servers and Hugging Face Spaces.
"""

from __future__ import annotations

import matplotlib

matplotlib.use("Agg")  # headless backend - safe on servers / Spaces

import matplotlib.pyplot as plt  # noqa: E402  (must come after use())
import numpy as np  # noqa: E402

from .utils import ZONES, output_path  # noqa: E402


def _save(fig, name: str) -> str:
    path = output_path(name)
    fig.savefig(path, bbox_inches="tight", dpi=110)
    plt.close(fig)
    return path


def chart_activity_by_zone(metrics: dict):
    zones = metrics.get("activity_by_zone", {})
    values = [zones.get(z, 0) for z in ZONES]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.bar(ZONES, values, color=["#4C72B0", "#55A868", "#C44E52"])
    ax.set_title("Field Activity by Zone")
    ax.set_ylabel("Total movement (pixels)")
    ax.set_xlabel("Field zone (image thirds)")
    return _save(fig, "chart_zone_activity.png")


def chart_top_track_segments(metrics: dict):
    # Uses stable field track segments with clean labels (from compute_metrics).
    segments = metrics.get("top_active_track_segments", [])
    if not segments:
        return None
    # Reverse so the most active segment sits at the top of the bar chart.
    names = [s["segment"] for s in segments][::-1]
    values = [s["movement_pixels"] for s in segments][::-1]
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.barh(names, values, color="#4C72B0")
    ax.set_title("Top Active Field Track Segments")
    ax.set_xlabel("Total movement (pixels)")
    return _save(fig, "chart_top_track_segments.png")


def chart_player_count_over_time(df):
    # Concurrent stable players on screen: count per frame, then average
    # within each second (per-second distinct IDs would over-count when IDs
    # switch mid-second).
    if df.empty:
        return None
    stable = df[df["is_stable"]] if "is_stable" in df.columns else df
    if stable.empty:
        return None
    per_frame = stable.groupby("frame").agg(
        second=("timestamp", "first"),
        count=("track_id", "nunique"),
    )
    per_frame["second"] = np.floor(per_frame["second"]).astype(int)
    timeline = per_frame.groupby("second")["count"].mean()
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.plot(timeline.index, timeline.values, marker="o", color="#55A868")
    ax.set_title("Field Players On Screen Over Time")
    ax.set_xlabel("Time (seconds)")
    ax.set_ylabel("Players on screen")
    ax.grid(True, alpha=0.3)
    return _save(fig, "chart_player_count.png")


def chart_heatmap(df, info: dict):
    # Uses all field-filtered detections (spectators already removed).
    if df.empty:
        return None
    w = int(info.get("processed_width") or 0) or int(df["x_center"].max() + 1)
    h = int(info.get("processed_height") or 0) or int(df["y_center"].max() + 1)
    fig, ax = plt.subplots(figsize=(5, 3.5))
    ax.hist2d(df["x_center"], df["y_center"], bins=(24, 16),
              range=[[0, w], [0, h]], cmap="hot")
    ax.set_title("Field Player Position Heatmap")
    ax.set_xlabel("x (pixels)")
    ax.set_ylabel("y (pixels)")
    ax.invert_yaxis()  # image coordinates grow downward
    return _save(fig, "chart_heatmap.png")


def generate_all_charts(df, metrics: dict, info: dict):
    """Return a list of chart image paths (skipping any that cannot render)."""
    charts = [
        chart_activity_by_zone(metrics),
        chart_top_track_segments(metrics),
        chart_player_count_over_time(df),
        chart_heatmap(df, info),
    ]
    return [c for c in charts if c]

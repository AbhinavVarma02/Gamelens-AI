"""
utils.py
--------
Shared helpers and configuration for GameLens AI.

This keeps the "glue" in one place: loading environment variables,
default settings for the computer-vision pipeline, geometry/zone helpers,
standard output paths, and a tiny OpenAI client factory that fails
gracefully when no API key is present.

Nothing here does heavy lifting - the goal is to keep the other modules
short and readable.
"""

from __future__ import annotations

import os
from dataclasses import dataclass

from dotenv import load_dotenv

# Load variables from a local .env file if one exists. On Hugging Face
# Spaces the values come from Space secrets instead, and this is a no-op.
load_dotenv()


# --- Project paths ------------------------------------------------------
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")

# Standard output file names (kept together so every module agrees).
ANNOTATED_VIDEO_NAME = "annotated_video.mp4"
TRACKING_CSV_NAME = "tracking_data.csv"
METRICS_JSON_NAME = "metrics.json"
REPORT_TXT_NAME = "match_report.txt"


def ensure_output_dir() -> str:
    """Make sure the outputs/ folder exists and return its path."""
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    return OUTPUT_DIR


def output_path(name: str) -> str:
    """Full path for a file inside outputs/ (creating the folder if needed)."""
    ensure_output_dir()
    return os.path.join(OUTPUT_DIR, name)


# --- OpenAI / LLM config ------------------------------------------------
DEFAULT_OPENAI_MODEL = "gpt-4o-mini"


def get_openai_model() -> str:
    """Model id to use, from OPENAI_MODEL or the gpt-4o-mini default."""
    return os.getenv("OPENAI_MODEL", DEFAULT_OPENAI_MODEL) or DEFAULT_OPENAI_MODEL


def has_openai_key() -> bool:
    """True only if a non-empty OPENAI_API_KEY is set."""
    return bool(os.getenv("OPENAI_API_KEY", "").strip())


def get_openai_client():
    """Return an OpenAI client, or None if it is unavailable.

    Returns None when the API key is missing or the openai package cannot
    be imported. Callers treat None as "use fallback text" so the app never
    crashes just because there is no key.
    """
    if not has_openai_key():
        return None
    try:
        from openai import OpenAI

        return OpenAI()  # reads OPENAI_API_KEY from the environment
    except Exception:
        return None


# --- Computer-vision config ---------------------------------------------
@dataclass
class TrackingConfig:
    """Settings that trade off speed vs. quality in the CV pipeline."""

    model_path: str = "yolov8n.pt"     # small, fast default model
    tracker: str = "bytetrack.yaml"    # Ultralytics ByteTrack config
    person_class_id: int = 0           # COCO 'person' class
    confidence: float = 0.3            # detection confidence threshold
    frame_skip: int = 8                # process every Nth frame
    max_width: int = 640               # shrink wide frames to this width
    max_seconds: int = 120             # safety cap on video length processed
    # Spectator/stands filtering: the top of a broadcast frame is usually the
    # stands. Keep a detection only if y_center >= frame_height * this ratio.
    field_roi_top_ratio: float = 0.35
    # A stable field track segment is a track id that appears in at least this
    # many processed frames (filters out flickering / very short-lived IDs).
    min_track_frames: int = 10


# --- Zone + geometry helpers -------------------------------------------
ZONE_LEFT = "Left"
ZONE_MIDDLE = "Middle"
ZONE_RIGHT = "Right"
ZONES = [ZONE_LEFT, ZONE_MIDDLE, ZONE_RIGHT]


def classify_zone(x_center: float, frame_width: int) -> str:
    """Split the frame into left / middle / right thirds by x position."""
    if frame_width <= 0:
        return ZONE_MIDDLE
    third = frame_width / 3.0
    if x_center < third:
        return ZONE_LEFT
    if x_center < 2 * third:
        return ZONE_MIDDLE
    return ZONE_RIGHT


def format_seconds(seconds: float) -> str:
    """Format a number of seconds as an integer label, e.g. '35s'."""
    return f"{int(round(seconds))}s"

"""
video_processor.py
------------------
Thin OpenCV helpers for reading and writing video.

Responsibilities:
- Read basic video metadata (fps, size, frame count, duration).
- Work out a good output size when we shrink wide frames.
- Create an OpenCV VideoWriter for the annotated output clip.

The actual detection/tracking loop lives in detect_track.py; this file
keeps the low-level video plumbing out of the way.
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2


@dataclass
class VideoMetadata:
    fps: float
    width: int
    height: int
    frame_count: int
    duration_seconds: float


def get_video_metadata(video_path: str) -> VideoMetadata:
    """Read metadata from a video file using OpenCV."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        cap.release()
        raise ValueError(
            "Could not open the video file. Please upload a valid .mp4 clip."
        )

    fps = cap.get(cv2.CAP_PROP_FPS) or 0.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()

    # Some files report a missing/zero fps; fall back to a sane default.
    if fps <= 0:
        fps = 25.0

    duration = frame_count / fps if fps > 0 and frame_count > 0 else 0.0
    return VideoMetadata(fps, width, height, frame_count, duration)


def compute_output_size(width: int, height: int, max_width: int) -> tuple[int, int]:
    """Scale (width, height) down so width <= max_width, keeping aspect ratio.

    Dimensions are forced even, which most video codecs prefer.
    """
    if width <= 0 or height <= 0:
        return width, height

    if max_width <= 0 or width <= max_width:
        out_w, out_h = width, height
    else:
        scale = max_width / float(width)
        out_w = int(width * scale)
        out_h = int(height * scale)

    out_w -= out_w % 2
    out_h -= out_h % 2
    return max(out_w, 2), max(out_h, 2)


class AnnotatedVideoWriter:
    """Writes the annotated clip and exposes a tiny `.write()/.release()` API.

    Prefers browser-playable H.264 via imageio-ffmpeg (its wheel bundles a
    static ffmpeg with libx264, so no system ffmpeg / OpenH264 is needed).
    Falls back to OpenCV's built-in 'mp4v' if imageio is unavailable, so the
    app never crashes - and, importantly, this avoids the noisy OpenH264
    encoder errors that OpenCV's 'avc1' path prints when the codec is missing.

    `fps` should already account for frame skipping. Frames are written in
    OpenCV's BGR order.
    """

    def __init__(self, path: str, fps: float, size: tuple[int, int]):
        self._backend = None
        self._writer = None
        fps = max(float(fps), 1.0)
        try:
            import imageio

            self._writer = imageio.get_writer(
                path, fps=fps, codec="libx264", format="FFMPEG",
                macro_block_size=None,   # keep our exact (even) size
                # imageio already defaults libx264 output to yuv420p, which is
                # the broadly browser-compatible pixel format.
            )
            self._backend = "imageio"
        except Exception:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            self._writer = cv2.VideoWriter(path, fourcc, fps, size)
            self._backend = "cv2"

    def write(self, frame_bgr) -> None:
        if self._backend == "imageio":
            self._writer.append_data(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        else:
            self._writer.write(frame_bgr)

    def release(self) -> None:
        try:
            if self._backend == "imageio":
                self._writer.close()
            else:
                self._writer.release()
        except Exception:
            pass


def create_video_writer(path: str, fps: float, size: tuple[int, int]) -> AnnotatedVideoWriter:
    """Create a writer for the annotated clip (H.264 if possible, else mp4v)."""
    return AnnotatedVideoWriter(path, fps, size)

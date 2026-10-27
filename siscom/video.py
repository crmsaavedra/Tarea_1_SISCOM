"""Video metadata and mean grayscale intensity extraction."""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np

from .models import ROI, VideoMetadata, VideoSignal


class VideoReadError(ValueError):
    """An actionable error for files that cannot be decoded as video."""


def roi_to_pixels(roi: ROI, width: int, height: int) -> tuple[int, int, int, int]:
    """Map a normalized ROI to `(x, y, width, height)` pixels for one frame."""
    if width <= 0 or height <= 0:
        raise ValueError("Frame dimensions must be greater than zero")
    x0 = math.floor(roi.left * width)
    y0 = math.floor(roi.top * height)
    x1 = math.ceil((roi.left + roi.width) * width)
    y1 = math.ceil((roi.top + roi.height) * height)
    x0 = max(0, min(x0, width - 1))
    y0 = max(0, min(y0, height - 1))
    x1 = max(x0 + 1, min(x1, width))
    y1 = max(y0 + 1, min(y1, height))
    return x0, y0, x1 - x0, y1 - y0


def _open_video(path: Path) -> cv2.VideoCapture:
    if not path.is_file():
        raise VideoReadError(f"Video file does not exist: {path}")
    capture = cv2.VideoCapture(str(path))
    if not capture.isOpened():
        capture.release()
        raise VideoReadError(f"Video could not be opened: {path.name}. Check the file and codec.")
    return capture


def _read_metadata(capture: cv2.VideoCapture, first_frame: np.ndarray) -> VideoMetadata:
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    if not np.isfinite(fps) or fps <= 0.0:
        raise VideoReadError("Video frame rate is missing or invalid; re-export the recording with a valid fps.")
    height, width = first_frame.shape[:2]
    frame_count = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
    if frame_count <= 0:
        # Some containers omit the frame-count header. Count readable frames
        # from the current position instead of returning a misleading duration.
        frame_count = 1
        while True:
            ok, _ = capture.read()
            if not ok:
                break
            frame_count += 1
    return VideoMetadata(
        fps=fps,
        frame_count=frame_count,
        width=int(width),
        height=int(height),
        duration_seconds=frame_count / fps,
    )


def probe_video(path: Path) -> VideoMetadata:
    """Inspect a video and verify that its first frame can be decoded."""
    capture = _open_video(Path(path))
    try:
        ok, first_frame = capture.read()
        if not ok or first_frame is None:
            raise VideoReadError(f"Video could not be decoded: {Path(path).name}. Re-record or re-export it.")
        return _read_metadata(capture, first_frame)
    finally:
        capture.release()


def extract_video_signal(path: Path, roi: ROI) -> VideoSignal:
    """Return mean grayscale intensity in the selected ROI for every frame."""
    capture = _open_video(Path(path))
    intensities: list[float] = []
    try:
        fps = float(capture.get(cv2.CAP_PROP_FPS))
        if not np.isfinite(fps) or fps <= 0.0:
            raise VideoReadError("Video frame rate is missing or invalid; re-export the recording with a valid fps.")

        frame_width = 0
        frame_height = 0
        pixels: tuple[int, int, int, int] | None = None
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            if frame is None or frame.size == 0:
                raise VideoReadError("Video contains an empty frame and cannot be processed.")
            current_height, current_width = frame.shape[:2]
            if not frame_width:
                frame_width, frame_height = int(current_width), int(current_height)
                pixels = roi_to_pixels(roi, frame_width, frame_height)
            elif (current_width, current_height) != (frame_width, frame_height):
                raise VideoReadError("Video frame dimensions change during the clip; re-export it at a fixed resolution.")

            if frame.ndim == 2:
                gray = frame
            else:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            assert pixels is not None
            x, y, width, height = pixels
            intensities.append(float(np.mean(gray[y : y + height, x : x + width], dtype=np.float64)))
    finally:
        capture.release()

    if not intensities:
        raise VideoReadError(f"Video could not be decoded: {Path(path).name}. Re-record or re-export it.")
    values = np.asarray(intensities, dtype=np.float64)
    return VideoSignal(
        times=np.arange(values.size, dtype=np.float64) / fps,
        intensity=values,
        fps=fps,
        width=frame_width,
        height=frame_height,
        roi=roi,
    )

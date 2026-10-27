"""Shared workflow labels and video-preview geometry for the desktop UI."""

from __future__ import annotations

import cv2
import numpy as np
from .models import ROI


WORKFLOW_PAGES = (
    "Proyecto",
    "Prueba rápida",
    "Calibración",
    "Decodificación",
    "Análisis",
    "Exportación",
)
PREVIEW_BACKGROUND_RGB = (23, 33, 43)


def fit_portrait_preview(
    frame_rgb: np.ndarray,
    width: int = 360,
    height: int = 640,
) -> np.ndarray:
    """Rotate landscape frames and fit them into a dark 9:16 RGB canvas."""
    frame = np.asarray(frame_rgb)
    if frame.ndim != 3 or frame.shape[2] != 3 or not frame.shape[0] or not frame.shape[1]:
        raise ValueError("Preview input must be a nonempty RGB image")
    if width <= 0 or height <= 0:
        raise ValueError("Preview dimensions must be greater than zero")
    if frame.dtype != np.uint8:
        if not np.issubdtype(frame.dtype, np.number) or not np.isfinite(frame).all():
            raise ValueError("Preview pixels must be finite numeric values")
        frame = np.clip(frame, 0, 255).astype(np.uint8)

    source_height, source_width = frame.shape[:2]
    if source_width > source_height:
        frame = cv2.rotate(frame, cv2.ROTATE_90_CLOCKWISE)
        source_height, source_width = frame.shape[:2]

    scale = min(width / source_width, height / source_height)
    scaled_width = max(1, min(width, int(round(source_width * scale))))
    scaled_height = max(1, min(height, int(round(source_height * scale))))
    interpolation = cv2.INTER_AREA if scale < 1.0 else cv2.INTER_NEAREST
    resized = cv2.resize(frame, (scaled_width, scaled_height), interpolation=interpolation)

    canvas = np.full((height, width, 3), PREVIEW_BACKGROUND_RGB, dtype=np.uint8)
    left = (width - scaled_width) // 2
    top = (height - scaled_height) // 2
    canvas[top : top + scaled_height, left : left + scaled_width] = resized
    return canvas


def _preview_transform(source_width: int, source_height: int, width: int, height: int):
    if min(source_width, source_height, width, height) <= 0:
        raise ValueError("Source and preview dimensions must be greater than zero")
    rotated = source_width > source_height
    rotated_width, rotated_height = (
        (source_height, source_width) if rotated else (source_width, source_height)
    )
    scale = min(width / rotated_width, height / rotated_height)
    offset_x = (width - rotated_width * scale) / 2.0
    offset_y = (height - rotated_height * scale) / 2.0
    return rotated, scale, offset_x, offset_y


def roi_to_preview_bounds(
    roi: ROI,
    source_width: int,
    source_height: int,
    *,
    width: int = 360,
    height: int = 640,
) -> tuple[int, int, int, int]:
    """Map original-frame ROI edges into the rotated, letterboxed preview."""
    rotated, scale, offset_x, offset_y = _preview_transform(
        source_width, source_height, width, height
    )
    x0, x1 = roi.left * source_width, (roi.left + roi.width) * source_width
    y0, y1 = roi.top * source_height, (roi.top + roi.height) * source_height
    if rotated:
        x0, y0, x1, y1 = source_height - y1, x0, source_height - y0, x1
    bounds = (
        offset_x + x0 * scale,
        offset_y + y0 * scale,
        offset_x + x1 * scale,
        offset_y + y1 * scale,
    )
    return tuple(int(round(value)) for value in bounds)


def preview_bounds_to_roi(
    bounds: tuple[float, float, float, float],
    source_width: int,
    source_height: int,
    *,
    width: int = 360,
    height: int = 640,
) -> ROI:
    """Convert a drawn preview rectangle back to normalized source coordinates."""
    rotated, scale, offset_x, offset_y = _preview_transform(
        source_width, source_height, width, height
    )
    preview_width = source_height if rotated else source_width
    preview_height = source_width if rotated else source_height
    image_x0, image_x1 = offset_x, offset_x + preview_width * scale
    image_y0, image_y1 = offset_y, offset_y + preview_height * scale
    x0, y0, x1, y1 = map(float, bounds)
    x0, x1 = sorted((max(image_x0, min(image_x1, x0)), max(image_x0, min(image_x1, x1))))
    y0, y1 = sorted((max(image_y0, min(image_y1, y0)), max(image_y0, min(image_y1, y1))))
    if x1 - x0 < 1.0 or y1 - y0 < 1.0:
        raise ValueError("Draw a region at least one preview pixel wide and high")
    rx0, rx1 = (x0 - offset_x) / scale, (x1 - offset_x) / scale
    ry0, ry1 = (y0 - offset_y) / scale, (y1 - offset_y) / scale
    if rotated:
        source_x0, source_x1 = ry0, ry1
        source_y0, source_y1 = source_height - rx1, source_height - rx0
    else:
        source_x0, source_x1 = rx0, rx1
        source_y0, source_y1 = ry0, ry1
    source_x0 = min(max(source_x0, 0.0), source_width - 1.0)
    source_y0 = min(max(source_y0, 0.0), source_height - 1.0)
    source_x1 = min(max(source_x1, source_x0 + 1.0), float(source_width))
    source_y1 = min(max(source_y1, source_y0 + 1.0), float(source_height))
    return ROI(
        source_x0 / source_width,
        source_y0 / source_height,
        (source_x1 - source_x0) / source_width,
        (source_y1 - source_y0) / source_height,
    )

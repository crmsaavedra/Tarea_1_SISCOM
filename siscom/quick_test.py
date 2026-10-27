"""Automatic, single-clip Morse analysis for the HOLA quick test."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .models import Calibration, DecodeResult, ROI, VideoSignal
from .morse import decode_signal
from .video import VideoReadError, extract_video_signal, probe_video


_MAX_ROI_SAMPLE_FRAMES = 96
_ROI_SIDE_FRACTION = 0.16
_DEFAULT_TOLERANCE = 0.4


class QuickVideoError(ValueError):
    """A video cannot support an honest single-clip Morse estimate."""


@dataclass(frozen=True)
class QuickVideoAnalysis:
    signal: VideoSignal
    calibration: Calibration
    decode: DecodeResult
    intensity_contrast: float
    dot_dash_ratio: float | None
    diagnostics: tuple[str, ...]


def _estimate_roi(video_path: Path) -> tuple[ROI, float]:
    metadata = probe_video(video_path)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        capture.release()
        raise QuickVideoError(f"No se pudo abrir el video {video_path.name}.")

    sample_count = min(metadata.frame_count, _MAX_ROI_SAMPLE_FRAMES)
    frame_indices = np.unique(
        np.linspace(0, metadata.frame_count - 1, sample_count, dtype=np.int64)
    )
    frames: list[np.ndarray] = []
    try:
        for frame_index in frame_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            frames.append(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    finally:
        capture.release()

    if len(frames) < 2:
        raise QuickVideoError("El video no contiene suficientes cuadros para localizar la luz.")
    if any(frame.shape != frames[0].shape for frame in frames):
        raise QuickVideoError("La resolución cambia durante el video; no se puede estimar la ROI.")

    stack = np.stack(frames).astype(np.float32)
    variance = np.var(stack, axis=0)
    local_variance = cv2.GaussianBlur(variance, (0, 0), sigmaX=2.0)
    peak_y, peak_x = np.unravel_index(int(np.argmax(local_variance)), local_variance.shape)
    peak = float(local_variance[peak_y, peak_x])
    median = float(np.median(local_variance))
    if not np.isfinite(peak) or peak < 1.0:
        raise QuickVideoError(
            "No se detecta contraste temporal: revisa que la luz parpadee y se vea en el video."
        )

    height, width = frames[0].shape
    roi_width = min(1.0, max(_ROI_SIDE_FRACTION, 8.0 / width))
    roi_height = min(1.0, max(_ROI_SIDE_FRACTION, 8.0 / height))
    roi_width_px = max(1, int(round(roi_width * width)))
    roi_height_px = max(1, int(round(roi_height * height)))
    left_px = min(max(int(peak_x) - roi_width_px // 2, 0), width - roi_width_px)
    top_px = min(max(int(peak_y) - roi_height_px // 2, 0), height - roi_height_px)
    roi = ROI(
        left_px / width,
        top_px / height,
        roi_width_px / width,
        roi_height_px / height,
    )

    confidence = max(0.0, min(1.0, (peak - median) / max(peak, 1.0)))
    return roi, confidence


def _estimate_threshold(intensity: np.ndarray) -> tuple[float, float]:
    pixels = np.clip(np.rint(intensity), 0, 255).astype(np.uint8).reshape(-1, 1)
    otsu_value, binary = cv2.threshold(
        pixels, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU
    )
    is_light = binary.reshape(-1) > 0
    dark_values = intensity[~is_light]
    light_values = intensity[is_light]
    if dark_values.size == 0 or light_values.size == 0:
        raise QuickVideoError(
            "No se pudo separar la luz encendida de la apagada; revisa el encuadre del video."
        )

    contrast = float(np.mean(light_values) - np.mean(dark_values))
    pooled_noise = float(
        np.sqrt((np.var(light_values) + np.var(dark_values)) / 2.0)
    )
    separation = contrast / max(pooled_noise, 1.0)
    if contrast < 12.0 or separation < 1.5:
        raise QuickVideoError(
            "El contraste entre luz encendida y apagada es insuficiente para decodificar."
        )
    # The existing decoder classifies values >= threshold as light. Move half
    # a digital level above Otsu's integer threshold to keep its dark class off.
    return float(otsu_value) + 0.5, contrast


def _estimate_point_duration(
    intensity: np.ndarray, threshold: float, fps: float
) -> tuple[float, float | None, list[str]]:
    states = intensity >= threshold
    changes = np.flatnonzero(states[1:] != states[:-1]) + 1
    boundaries = np.r_[0, changes, states.size]
    on_durations = np.asarray(
        [end - start for start, end in zip(boundaries[:-1], boundaries[1:]) if states[start]],
        dtype=np.float64,
    )
    if on_durations.size < 2:
        raise QuickVideoError("No se detectaron suficientes pulsos de luz para estimar T.")

    shortest = float(on_durations.min())
    dot_runs = on_durations[on_durations <= shortest * 1.35]
    if dot_runs.size < 2:
        raise QuickVideoError(
            "No hay puntos repetidos suficientes para estimar T con confianza."
        )
    dot_frames = float(np.median(dot_runs))
    dash_runs = on_durations[on_durations >= dot_frames * 2.0]
    diagnostics: list[str] = []
    if dash_runs.size == 0:
        ratio = None
        diagnostics.append(
            "No se detectaron rayas; T se estimó con puntos y no se pudo verificar la duración 3T."
        )
    else:
        ratio = float(np.median(dash_runs) / dot_frames)
        if not 2.0 <= ratio <= 4.0:
            diagnostics.append(
                "La duración de puntos y rayas no sigue claramente la relación aproximada 1:3."
            )

    return dot_frames / fps, ratio, diagnostics


def analyze_quick_video(video_path: Path) -> QuickVideoAnalysis:
    """Estimate a one-video calibration, then decode from signal only."""
    video_path = Path(video_path)
    try:
        roi, roi_confidence = _estimate_roi(video_path)
        signal = extract_video_signal(video_path, roi)
    except VideoReadError as error:
        raise QuickVideoError(f"No se pudo leer el video: {error}") from error

    threshold, contrast = _estimate_threshold(signal.intensity)
    point_seconds, dot_dash_ratio, diagnostics = _estimate_point_duration(
        signal.intensity, threshold, signal.fps
    )
    if roi_confidence < 0.2:
        diagnostics.append(
            "La ubicación automática de la luz es incierta; revisa que el LED esté quieto y visible."
        )

    calibration = Calibration(
        threshold=threshold,
        point_seconds=point_seconds,
        tolerance_fraction=_DEFAULT_TOLERANCE,
        roi=roi,
        reference_fps=signal.fps,
    )
    decoded = decode_signal(signal.intensity, signal.fps, calibration)
    return QuickVideoAnalysis(
        signal=signal,
        calibration=calibration,
        decode=decoded,
        intensity_contrast=contrast,
        dot_dash_ratio=dot_dash_ratio,
        diagnostics=tuple(diagnostics),
    )

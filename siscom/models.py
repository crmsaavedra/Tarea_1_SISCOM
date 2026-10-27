"""Small immutable values shared by the video, Morse, and analysis layers."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class ROI:
    """A reproducible region of interest expressed as frame fractions."""

    left: float
    top: float
    width: float
    height: float

    def __post_init__(self) -> None:
        values = (self.left, self.top, self.width, self.height)
        if any(not 0.0 <= value <= 1.0 for value in values):
            raise ValueError("ROI coordinates and size must be between 0 and 1")
        if self.width <= 0.0 or self.height <= 0.0:
            raise ValueError("ROI width and height must be greater than zero")
        if self.left + self.width > 1.0 or self.top + self.height > 1.0:
            raise ValueError("ROI must fit inside the frame")


@dataclass(frozen=True)
class Calibration:
    threshold: float
    point_seconds: float
    tolerance_fraction: float
    roi: ROI
    reference_fps: float

    def __post_init__(self) -> None:
        if not 0.0 <= self.threshold <= 255.0:
            raise ValueError("Threshold must be between 0 and 255 digital levels")
        if self.point_seconds <= 0.0:
            raise ValueError("Reference point duration T must be greater than zero")
        if not 0.0 < self.tolerance_fraction < 0.5:
            raise ValueError("Timing tolerance must be between 0 and 50 percent")
        if self.reference_fps <= 0.0:
            raise ValueError("Calibration frame rate must be greater than zero")


@dataclass(frozen=True)
class PulseRun:
    """A contiguous light-on or light-off interval; end_frame is exclusive."""

    start_frame: int
    end_frame: int
    is_light: bool
    fps: float

    def __post_init__(self) -> None:
        if self.start_frame < 0 or self.end_frame <= self.start_frame:
            raise ValueError("A pulse run must cover at least one frame")
        if self.fps <= 0.0:
            raise ValueError("Pulse-run frame rate must be greater than zero")

    @property
    def duration_seconds(self) -> float:
        return (self.end_frame - self.start_frame) / self.fps


@dataclass(frozen=True)
class DecodeResult:
    text: str
    runs: tuple[PulseRun, ...]
    decisions: tuple[str, ...]


@dataclass(frozen=True)
class VideoMetadata:
    fps: float
    frame_count: int
    width: int
    height: int
    duration_seconds: float

    def __post_init__(self) -> None:
        if not np.isfinite(self.fps) or self.fps <= 0.0:
            raise ValueError("Video frame rate must be greater than zero")
        if self.frame_count <= 0:
            raise ValueError("Video must contain at least one readable frame")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Video dimensions must be greater than zero")
        if not np.isfinite(self.duration_seconds) or self.duration_seconds <= 0.0:
            raise ValueError("Video duration must be greater than zero")


@dataclass(frozen=True)
class VideoSignal:
    times: np.ndarray
    intensity: np.ndarray
    fps: float
    width: int
    height: int
    roi: ROI

    def __post_init__(self) -> None:
        times = np.asarray(self.times, dtype=np.float64)
        intensity = np.asarray(self.intensity, dtype=np.float64)
        if times.ndim != 1 or intensity.ndim != 1 or times.size != intensity.size:
            raise ValueError("Video times and intensity must be one-dimensional arrays of equal length")
        if times.size == 0:
            raise ValueError("Video signal must contain at least one frame")
        if not np.isfinite(times).all() or not np.isfinite(intensity).all():
            raise ValueError("Video signal contains non-finite values")
        if not np.isfinite(self.fps) or self.fps <= 0.0:
            raise ValueError("Video frame rate must be greater than zero")
        if self.width <= 0 or self.height <= 0:
            raise ValueError("Video dimensions must be greater than zero")
        object.__setattr__(self, "times", times)
        object.__setattr__(self, "intensity", intensity)


@dataclass(frozen=True)
class TeamMember:
    name: str
    contribution_percent: int

    def __post_init__(self) -> None:
        if not self.name.strip():
            raise ValueError("Each team member needs a name")
        if not 0 <= self.contribution_percent <= 100:
            raise ValueError("Contribution must be between 0 and 100")


@dataclass(frozen=True)
class SetupMetadata:
    emission_mode: str
    source: str
    control_method: str
    materials: str
    lighting: str
    brightness: str

    def __post_init__(self) -> None:
        if self.emission_mode not in {"manual", "automática"}:
            raise ValueError("Emission mode must be manual or automática")


@dataclass(frozen=True)
class Transmission:
    distance_m: float
    video_path: "Path"
    roi: ROI
    acquisition_valid: bool
    invalid_reason: str | None

    def __post_init__(self) -> None:
        if not np.isfinite(self.distance_m) or self.distance_m <= 0.0:
            raise ValueError("Transmission distance must be greater than zero")
        if not self.acquisition_valid and not (self.invalid_reason or "").strip():
            raise ValueError("An invalid acquisition requires a reason")


@dataclass(frozen=True)
class Project:
    root: "Path"
    calibration_video: "Path"
    transmissions: list[Transmission]
    calibration: Calibration | None
    expected_text: str
    team: tuple[TeamMember, TeamMember]
    setup: SetupMetadata
    references: tuple[str, ...]
    ai_assistance: str
    montage_video: "Path | None"
    montage_photo: "Path | None"

    def __post_init__(self) -> None:
        if len(self.team) != 2:
            raise ValueError("The assignment project requires exactly two team members")

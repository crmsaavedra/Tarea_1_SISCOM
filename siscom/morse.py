"""Morse encoding and duration-based decoding for optical OOK signals."""

from __future__ import annotations

from collections.abc import Mapping
from math import isclose

import numpy as np

from .models import Calibration, DecodeResult, PulseRun


CHAR_TO_MORSE: Mapping[str, str] = {
    "A": ".-", "B": "-...", "C": "-.-.", "D": "-..", "E": ".",
    "F": "..-.", "G": "--.", "H": "....", "I": "..", "J": ".---",
    "K": "-.-", "L": ".-..", "M": "--", "N": "-.", "O": "---",
    "P": ".--.", "Q": "--.-", "R": ".-.", "S": "...", "T": "-",
    "U": "..-", "V": "...-", "W": ".--", "X": "-..-", "Y": "-.--",
    "Z": "--..",
}
MORSE_TO_CHAR: Mapping[str, str] = {pattern: letter for letter, pattern in CHAR_TO_MORSE.items()}


def encode_text_morse(message: str) -> str:
    """Encode A–Z and spaces, separating letters by spaces and words by `/`."""
    normalized = message.upper()
    if not normalized:
        return ""
    unsupported = sorted(
        {character for character in normalized if character != " " and character not in CHAR_TO_MORSE}
    )
    if unsupported:
        raise ValueError(f"Unsupported character(s): {', '.join(unsupported)}")
    words = normalized.split()
    if not words:
        return ""
    return " / ".join(" ".join(CHAR_TO_MORSE[letter] for letter in word) for word in words)


def _runs_from_binary(binary: np.ndarray, fps: float) -> tuple[PulseRun, ...]:
    if binary.size == 0:
        return ()
    changes = np.flatnonzero(binary[1:] != binary[:-1]) + 1
    boundaries = np.r_[0, changes, binary.size]
    return tuple(
        PulseRun(int(start), int(end), bool(binary[start]), fps)
        for start, end in zip(boundaries[:-1], boundaries[1:])
    )


def _classify_duration(
    duration: float,
    targets: tuple[tuple[str, float], ...],
    tolerance: float,
) -> str | None:
    matches = [
        (label, abs(duration - target) / target)
        for label, target in targets
        if (
            abs(duration - target) <= target * tolerance
            or isclose(abs(duration - target), target * tolerance, rel_tol=1e-12, abs_tol=1e-12)
        )
    ]
    if not matches:
        return None
    return min(matches, key=lambda match: match[1])[0]


def decode_signal(intensity: np.ndarray, fps: float, calibration: Calibration) -> DecodeResult:
    """Decode thresholded light runs using timing only; expected text is not an input."""
    signal = np.asarray(intensity, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("Intensity signal must be one-dimensional")
    if signal.size == 0:
        return DecodeResult(text="", runs=(), decisions=("no_frames",))
    if not np.isfinite(signal).all():
        raise ValueError("Intensity signal contains non-finite values")
    if fps <= 0.0:
        raise ValueError("Video frame rate must be greater than zero")

    runs = _runs_from_binary(signal >= calibration.threshold, fps)
    light_indices = [index for index, run in enumerate(runs) if run.is_light]
    if not light_indices:
        return DecodeResult(text="", runs=runs, decisions=("no_light_pulses",))
    active_runs = runs[light_indices[0] : light_indices[-1] + 1]

    point = calibration.point_seconds
    tolerance = calibration.tolerance_fraction
    decoded: list[str] = []
    decisions: list[str] = []
    current_pattern: list[str] = []
    current_invalid = False

    def finish_letter() -> None:
        nonlocal current_pattern, current_invalid
        if not current_pattern and not current_invalid:
            return
        pattern = "".join(current_pattern)
        letter = "?" if current_invalid else MORSE_TO_CHAR.get(pattern, "?")
        decoded.append(letter)
        if letter == "?":
            decisions.append(f"invalid_morse:{pattern or 'pulse_timing'}")
        else:
            decisions.append(f"letter:{letter}({pattern})")
        current_pattern = []
        current_invalid = False

    for run in active_runs:
        duration = run.duration_seconds
        if run.is_light:
            symbol = _classify_duration(
                duration,
                (("dot", point), ("dash", 3.0 * point)),
                tolerance,
            )
            if symbol is None:
                current_invalid = True
                decisions.append(f"invalid_pulse:{duration:.4f}s")
            else:
                current_pattern.append("." if symbol == "dot" else "-")
                decisions.append(f"{symbol}:{duration:.4f}s")
            continue

        # Leading/trailing off margins are excluded above. Every remaining
        # off-run lies between pulses and therefore carries a separator.
        gap = _classify_duration(
            duration,
            (("symbol_gap", point), ("letter_gap", 3.0 * point), ("word_gap", 7.0 * point)),
            tolerance,
        )
        if gap is None:
            decisions.append(f"invalid_gap:{duration:.4f}s")
            finish_letter()
            decoded.append("?")
            continue
        decisions.append(f"{gap}:{duration:.4f}s")
        if gap == "letter_gap":
            finish_letter()
        elif gap == "word_gap":
            finish_letter()
            if decoded and decoded[-1] != " ":
                decoded.append(" ")

    finish_letter()
    return DecodeResult(text="".join(decoded).strip(), runs=runs, decisions=tuple(decisions))

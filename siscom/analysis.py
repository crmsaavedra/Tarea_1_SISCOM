"""Error-rate and frequency-domain calculations for captured Morse video."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.signal import welch


@dataclass(frozen=True)
class ErrorCounts:
    substitutions: int
    deletions: int
    insertions: int

    @property
    def total(self) -> int:
        return self.substitutions + self.deletions + self.insertions


@dataclass(frozen=True)
class SpectralAnalysis:
    fft_frequencies: np.ndarray
    fft_amplitude: np.ndarray
    psd_frequencies: np.ndarray
    psd: np.ndarray
    b95_hz: float | None
    psd_method: str
    window: str
    nperseg: int


def levenshtein_counts(reference: str, received: str) -> ErrorCounts:
    """Return a minimum-edit S/D/I decomposition with deterministic tie breaks."""
    rows, columns = len(reference), len(received)
    costs = [[0] * (columns + 1) for _ in range(rows + 1)]
    operations = [[""] * (columns + 1) for _ in range(rows + 1)]

    for row in range(1, rows + 1):
        costs[row][0] = row
        operations[row][0] = "deletion"
    for column in range(1, columns + 1):
        costs[0][column] = column
        operations[0][column] = "insertion"

    for row in range(1, rows + 1):
        for column in range(1, columns + 1):
            if reference[row - 1] == received[column - 1]:
                costs[row][column] = costs[row - 1][column - 1]
                operations[row][column] = "match"
                continue
            candidates = (
                (costs[row - 1][column - 1] + 1, 0, "substitution"),
                (costs[row - 1][column] + 1, 1, "deletion"),
                (costs[row][column - 1] + 1, 2, "insertion"),
            )
            cost, _, operation = min(candidates)
            costs[row][column] = cost
            operations[row][column] = operation

    substitutions = deletions = insertions = 0
    row, column = rows, columns
    while row or column:
        operation = operations[row][column]
        if operation in {"match", "substitution"}:
            substitutions += operation == "substitution"
            row -= 1
            column -= 1
        elif operation == "deletion":
            deletions += 1
            row -= 1
        else:
            insertions += 1
            column -= 1
    return ErrorCounts(int(substitutions), deletions, insertions)


def character_error_rate(reference: str, received: str) -> float:
    """Calculate CER after uppercasing, retaining spaces and all error marks."""
    reference = reference.upper()
    received = received.upper()
    if not reference:
        if not received:
            return 0.0
        raise ValueError("CER reference cannot be empty when received text is nonempty")
    return levenshtein_counts(reference, received).total / len(reference)


def analyze_spectrum(intensity: np.ndarray, fs: float) -> SpectralAnalysis:
    """Compute a one-sided amplitude FFT, Welch PSD, and integrated B95."""
    signal = np.asarray(intensity, dtype=np.float64)
    if signal.ndim != 1:
        raise ValueError("Spectrum input must be a one-dimensional signal")
    if signal.size < 2:
        raise ValueError("At least two samples are required for spectrum analysis")
    if not np.isfinite(signal).all():
        raise ValueError("Spectrum input contains non-finite values")
    if not np.isfinite(fs) or fs <= 0.0:
        raise ValueError("Sampling frequency fs must be greater than zero")

    centered = signal - np.mean(signal)
    sample_count = centered.size
    fft_frequencies = np.fft.rfftfreq(sample_count, d=1.0 / fs)
    fft_amplitude = np.abs(np.fft.rfft(centered)) / sample_count
    if sample_count % 2 == 0:
        fft_amplitude[1:-1] *= 2.0
    else:
        fft_amplitude[1:] *= 2.0

    nperseg = min(sample_count, 256)
    if nperseg > 2 and nperseg % 2:
        nperseg -= 1
    psd_frequencies, psd = welch(
        centered,
        fs=fs,
        window="hann",
        nperseg=nperseg,
        scaling="density",
        return_onesided=True,
    )
    cumulative_power = cumulative_trapezoid(psd, psd_frequencies, initial=0.0)
    total_power = float(cumulative_power[-1])
    if total_power <= 0.0:
        b95_hz = None
    else:
        target_power = 0.95 * total_power
        crossing = int(np.searchsorted(cumulative_power, target_power, side="left"))
        if crossing == 0:
            b95_hz = float(psd_frequencies[0])
        else:
            left_power, right_power = cumulative_power[crossing - 1 : crossing + 1]
            left_frequency, right_frequency = psd_frequencies[crossing - 1 : crossing + 1]
            fraction = (target_power - left_power) / (right_power - left_power)
            b95_hz = float(left_frequency + fraction * (right_frequency - left_frequency))

    return SpectralAnalysis(
        fft_frequencies=fft_frequencies,
        fft_amplitude=fft_amplitude,
        psd_frequencies=psd_frequencies,
        psd=psd,
        b95_hz=b95_hz,
        psd_method="Welch",
        window="hann",
        nperseg=nperseg,
    )

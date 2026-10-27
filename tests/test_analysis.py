import numpy as np
import pytest

from siscom.analysis import analyze_spectrum, character_error_rate, levenshtein_counts


@pytest.mark.parametrize(
    ("reference", "received", "expected"),
    [
        ("abc", "abc", (0, 0, 0)),
        ("abc", "axc", (1, 0, 0)),
        ("abc", "ac", (0, 1, 0)),
        ("ac", "abc", (0, 0, 1)),
        ("SIS COM", "SISCOM", (0, 1, 0)),
    ],
)
def test_levenshtein_counts_minimal_character_edits(reference, received, expected):
    counts = levenshtein_counts(reference, received)

    assert (counts.substitutions, counts.deletions, counts.insertions) == expected


def test_cer_uses_seven_original_characters_including_the_space():
    assert character_error_rate("SIS COM", "SIS CUM") == pytest.approx(1 / 7)
    assert character_error_rate("SIS COM", "SIS COM") == 0.0


def test_cer_compares_text_in_uppercase_and_keeps_spaces():
    assert character_error_rate("sis com", "SIS COM") == 0.0
    assert character_error_rate("SIS COM", "SISCOM") == pytest.approx(1 / 7)


def test_complete_recording_with_empty_received_text_has_cer_one():
    assert character_error_rate("SIS COM", "") == 1.0


def test_cer_rejects_nonempty_received_text_when_reference_is_empty():
    with pytest.raises(ValueError, match="reference cannot be empty"):
        character_error_rate("", "A")


def test_fft_places_known_sinusoid_peak_at_five_hz():
    fs = 40.0
    times = np.arange(1024) / fs
    signal = 2.0 + np.sin(2 * np.pi * 5.0 * times)

    result = analyze_spectrum(signal, fs)

    peak_index = int(np.argmax(result.fft_amplitude[1:]) + 1)
    assert result.fft_frequencies[peak_index] == pytest.approx(5.0)
    assert result.fft_amplitude[peak_index] == pytest.approx(1.0, abs=0.01)
    assert result.psd_method == "Welch"
    assert result.window == "hann"
    assert result.nperseg == 256


def test_b95_is_inside_one_sided_nyquist_interval():
    fs = 60.0
    times = np.arange(600) / fs
    signal = np.sin(2 * np.pi * 4.0 * times) + 0.3 * np.sin(2 * np.pi * 12.0 * times)

    result = analyze_spectrum(signal, fs)

    assert result.psd_frequencies[0] == 0.0
    assert result.psd_frequencies[-1] == pytest.approx(fs / 2)
    assert result.b95_hz is not None
    assert 0.0 <= result.b95_hz <= fs / 2


def test_zero_power_signal_has_no_fabricated_b95():
    result = analyze_spectrum(np.full(64, 23.0), fs=32.0)

    assert result.b95_hz is None
    assert np.all(result.psd == 0.0)


@pytest.mark.parametrize("signal, fs", [(np.array([1.0]), 10.0), (np.arange(10), 0.0), (np.arange(10), -1.0)])
def test_spectrum_rejects_too_short_or_invalid_sample_rate(signal, fs):
    with pytest.raises(ValueError):
        analyze_spectrum(signal, fs)

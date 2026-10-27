import numpy as np
import pytest

from siscom.models import Calibration, ROI
from siscom.morse import decode_signal, encode_text_morse


FPS = 20.0
POINT_SECONDS = 0.2


def make_signal(letters, *, point_frames=4, fps=FPS):
    """Build a fixture from hand-written Morse groups, not the app encoder."""
    values = [0] * 8
    for word_index, word in enumerate(letters):
        for letter_index, pattern in enumerate(word):
            for symbol_index, symbol in enumerate(pattern):
                duration = point_frames if symbol == "." else point_frames * 3
                values.extend([255] * duration)
                if symbol_index < len(pattern) - 1:
                    values.extend([0] * point_frames)
            if letter_index < len(word) - 1:
                values.extend([0] * (point_frames * 3))
        if word_index < len(letters) - 1:
            values.extend([0] * (point_frames * 7))
    values.extend([0] * 10)
    return np.asarray(values, dtype=np.float64)


def calibration(*, tolerance=0.2):
    return Calibration(
        threshold=127.5,
        point_seconds=POINT_SECONDS,
        tolerance_fraction=tolerance,
        roi=ROI(left=0.1, top=0.1, width=0.2, height=0.2),
        reference_fps=FPS,
    )


def test_encoder_preserves_letters_and_word_space():
    assert encode_text_morse("SIS COM") == "... .. ... / -.-. --- --"


def test_encoder_supports_the_complete_a_to_z_alphabet():
    expected = ".- -... -.-. -.. . ..-. --. .... .. .--- -.- .-.. -- -. --- .--. --.- .-. ... - ..- ...- .-- -..- -.-- --.."

    assert encode_text_morse("ABCDEFGHIJKLMNOPQRSTUVWXYZ") == expected


def test_encoder_normalizes_lowercase_and_repeated_spaces():
    assert encode_text_morse("sis   com") == "... .. ... / -.-. --- --"


def test_decoder_recovers_sis_com_from_measured_pulses():
    signal = make_signal([["...", "..", "..."], ["-.-.", "---", "--"]])

    result = decode_signal(signal, FPS, calibration())

    assert result.text == "SIS COM"
    assert len(result.runs) > 20
    assert any("dot" in decision for decision in result.decisions)
    assert any("word_gap" in decision for decision in result.decisions)


def test_decoder_drops_leading_and_trailing_off_margins():
    result = decode_signal(np.r_[np.zeros(45), np.full(4, 255), np.zeros(50)], FPS, calibration())

    assert result.text == "E"


def test_dot_at_tolerance_boundary_is_still_a_dot():
    signal = np.r_[np.zeros(8), np.full(5, 255), np.zeros(10)]

    result = decode_signal(signal, FPS, calibration(tolerance=0.25))

    assert result.text == "E"


@pytest.mark.parametrize(
    ("light_frames", "expected"),
    [(32, "E"), (48, "E"), (96, "T"), (144, "T")],
)
def test_point_and_dash_accept_both_inclusive_tolerance_boundaries(light_frames, expected):
    signal = np.r_[np.zeros(8), np.full(light_frames, 255), np.zeros(10)]
    candidate = Calibration(127.5, 0.4, 0.2, calibration().roi, 100.0)

    result = decode_signal(signal, 100.0, candidate)

    assert result.text == expected


@pytest.mark.parametrize(
    ("gap_frames", "expected"),
    [(48, "I"), (96, "EE"), (144, "EE"), (224, "E E"), (336, "E E")],
)
def test_symbol_letter_and_word_pauses_accept_inclusive_tolerance_boundaries(gap_frames, expected):
    signal = np.r_[np.zeros(8), np.full(40, 255), np.zeros(gap_frames), np.full(40, 255), np.zeros(8)]
    candidate = Calibration(127.5, 0.4, 0.2, calibration().roi, 100.0)

    result = decode_signal(signal, 100.0, candidate)

    assert result.text == expected


@pytest.mark.parametrize("light_frames", [31, 49, 95, 145])
def test_durations_just_outside_pulse_tolerance_are_not_accepted(light_frames):
    signal = np.r_[np.zeros(8), np.full(light_frames, 255), np.zeros(10)]
    candidate = Calibration(127.5, 0.4, 0.2, calibration().roi, 100.0)

    result = decode_signal(signal, 100.0, candidate)

    assert result.text == "?"


def test_unknown_morse_pattern_is_preserved_as_question_mark():
    result = decode_signal(make_signal([["......."]]), FPS, calibration())

    assert result.text == "?"


def test_pulse_with_unrecognized_duration_is_not_guessed():
    signal = np.r_[np.zeros(8), np.full(8, 255), np.zeros(10)]

    result = decode_signal(signal, FPS, calibration())

    assert result.text == "?"
    assert any("invalid" in decision for decision in result.decisions)


def test_encoder_rejects_characters_outside_letters_and_spaces():
    try:
        encode_text_morse("SIS-COM")
    except ValueError as error:
        assert "-" in str(error)
    else:
        raise AssertionError("unsupported punctuation must be reported")

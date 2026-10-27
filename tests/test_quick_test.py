import importlib
import importlib.util
from pathlib import Path

import cv2
import numpy as np
import pytest

from siscom.morse import CHAR_TO_MORSE
from siscom.video import roi_to_pixels


def quick_test_module():
    assert importlib.util.find_spec("siscom.quick_test") is not None, (
        "the standalone quick-test analyzer module is missing"
    )
    return importlib.import_module("siscom.quick_test")


def make_morse_video(
    path: Path,
    message: str,
    *,
    fps: float = 20.0,
    point_seconds: float = 0.2,
    background: int = 12,
    light: int = 245,
) -> Path:
    width, height = 96, 64
    dot_frames = round(point_seconds * fps)
    off_frame = np.full((height, width, 3), background, dtype=np.uint8)
    on_frame = off_frame.copy()
    led_x, led_y, led_size = 73, 23, 6
    on_frame[led_y : led_y + led_size, led_x : led_x + led_size] = light

    writer = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*"MJPG"), fps, (width, height)
    )
    if not writer.isOpened():
        pytest.skip("MJPG writer is not available in this OpenCV build")

    def write_frames(frame: np.ndarray, count: int) -> None:
        for _ in range(count):
            writer.write(frame)

    try:
        write_frames(off_frame, dot_frames * 4)
        words = message.upper().split()
        for word_index, word in enumerate(words):
            for letter_index, letter in enumerate(word):
                pattern = CHAR_TO_MORSE[letter]
                for symbol_index, symbol in enumerate(pattern):
                    write_frames(on_frame, dot_frames * (3 if symbol == "-" else 1))
                    if symbol_index < len(pattern) - 1:
                        write_frames(off_frame, dot_frames)
                if letter_index < len(word) - 1:
                    write_frames(off_frame, dot_frames * 3)
            if word_index < len(words) - 1:
                write_frames(off_frame, dot_frames * 7)
        write_frames(off_frame, dot_frames * 4)
    finally:
        writer.release()
    return path


def test_auto_analysis_finds_an_off_center_led_and_decodes_hola(tmp_path):
    video = make_morse_video(tmp_path / "hola.avi", "HOLA")

    analysis = quick_test_module().analyze_quick_video(video)

    x, y, width, height = roi_to_pixels(
        analysis.calibration.roi, analysis.signal.width, analysis.signal.height
    )
    assert x <= 73 < x + width
    assert y <= 23 < y + height
    assert analysis.decode.text == "HOLA"
    assert analysis.calibration.point_seconds == pytest.approx(0.2, abs=0.03)
    assert analysis.intensity_contrast > 12
    assert analysis.dot_dash_ratio == pytest.approx(3.0, abs=0.5)


def test_auto_analysis_rejects_a_video_without_light_dark_contrast(tmp_path):
    video = make_morse_video(
        tmp_path / "flat.avi", "HOLA", background=40, light=40
    )

    with pytest.raises(quick_test_module().QuickVideoError, match="contraste"):
        quick_test_module().analyze_quick_video(video)


def test_auto_analysis_warns_when_clip_has_no_dash_pulses(tmp_path):
    video = make_morse_video(tmp_path / "dots.avi", "EEE")

    analysis = quick_test_module().analyze_quick_video(video)

    assert analysis.decode.text == "EEE"
    assert any("raya" in item.lower() for item in analysis.diagnostics)


def test_auto_analysis_reports_an_unreadable_video(tmp_path):
    video = tmp_path / "broken.avi"
    video.write_bytes(b"not a video")

    with pytest.raises(quick_test_module().QuickVideoError, match="video"):
        quick_test_module().analyze_quick_video(video)

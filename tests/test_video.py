from pathlib import Path

import cv2
import numpy as np
import pytest

from siscom.models import ROI, VideoMetadata
from siscom.video import VideoReadError, extract_video_signal, probe_video, roi_to_pixels


def write_clip(path: Path, frames: list[np.ndarray], fps: float = 10.0) -> Path:
    height, width = frames[0].shape[:2]
    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        pytest.skip("MJPG writer is not available in this OpenCV build")
    try:
        for frame in frames:
            writer.write(frame)
    finally:
        writer.release()
    return path


def test_extracts_mean_grayscale_roi_and_frame_times(tmp_path):
    roi = ROI(left=0.25, top=0.25, width=0.5, height=0.5)
    frames = []
    for value in (60, 140, 220):
        frame = np.zeros((16, 20, 3), dtype=np.uint8)
        frame[4:12, 5:15] = value
        frames.append(frame)
    video = write_clip(tmp_path / "gray_pulses.avi", frames)

    signal = extract_video_signal(video, roi)

    assert signal.fps == pytest.approx(10.0)
    assert signal.times.tolist() == pytest.approx([0.0, 0.1, 0.2])
    assert signal.intensity == pytest.approx([60, 140, 220], abs=8)
    assert signal.width == 20
    assert signal.height == 16


def test_probe_video_reports_dimensions_and_duration(tmp_path):
    frames = [np.full((12, 18, 3), 100, dtype=np.uint8) for _ in range(4)]
    metadata = probe_video(write_clip(tmp_path / "short.avi", frames, fps=8.0))

    assert metadata.fps == pytest.approx(8.0)
    assert metadata.frame_count == 4
    assert (metadata.width, metadata.height) == (18, 12)
    assert metadata.duration_seconds == pytest.approx(0.5)


def test_roi_scales_proportionally_across_resolutions():
    roi = ROI(left=0.1, top=0.2, width=0.3, height=0.25)

    assert roi_to_pixels(roi, 100, 80) == (10, 16, 30, 20)
    assert roi_to_pixels(roi, 200, 160) == (20, 32, 60, 40)


def test_roi_aligned_to_bottom_right_frame_edge_is_valid():
    roi = ROI(left=0.8, top=0.75, width=0.2, height=0.25)

    assert roi_to_pixels(roi, 100, 80) == (80, 60, 20, 20)


def test_roi_outside_normalized_frame_is_rejected():
    with pytest.raises(ValueError, match="fit inside the frame"):
        ROI(left=0.9, top=0.0, width=0.2, height=0.5)


def test_unreadable_video_returns_an_actionable_error(tmp_path):
    invalid_video = tmp_path / "not_a_video.avi"
    invalid_video.write_text("not video bytes", encoding="utf-8")

    with pytest.raises(VideoReadError, match="could not be opened|Could not open"):
        probe_video(invalid_video)


def test_metadata_rejects_nonpositive_frame_rate():
    with pytest.raises(ValueError, match="frame rate"):
        VideoMetadata(fps=0.0, frame_count=10, width=20, height=20, duration_seconds=0.0)

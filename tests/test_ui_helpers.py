from inspect import signature
import json
from dataclasses import replace
from pathlib import Path

import cv2
import numpy as np
import pytest
import siscom.ui_helpers as ui_helpers

from siscom.models import (
    Calibration,
    DecodeResult,
    Project,
    ROI,
    SetupMetadata,
    TeamMember,
    Transmission,
    VideoSignal,
)
from siscom.ui_helpers import (
    ProcessedRecording,
    apply_calibration_roi_size,
    build_results_table,
    comparison_segment,
    decode_video_file,
    decoder_recording_is_current,
    decoder_configuration_signature,
    export_readiness,
    find_saved_projects,
    load_comparison_starts,
    required_result_files,
    reset_project_session,
    replace_transmission_by_path,
    validate_distance_set,
)
from siscom.project import analysis_output_signature


def make_clip(path: Path, patterns: list[str], *, fps: float = 20.0) -> Path:
    point_frames = 4
    values = [0] * 12
    for letter_index, pattern in enumerate(patterns):
        for symbol_index, symbol in enumerate(pattern):
            values.extend([255] * (point_frames * (3 if symbol == "-" else 1)))
            if symbol_index < len(pattern) - 1:
                values.extend([0] * point_frames)
        if letter_index < len(patterns) - 1:
            values.extend([0] * (point_frames * 3))
    values.extend([0] * 16)

    writer = cv2.VideoWriter(
        str(path),
        cv2.VideoWriter_fourcc(*"MJPG"),
        fps,
        (16, 16),
    )
    if not writer.isOpened():
        pytest.skip("MJPG writer is not available in this OpenCV build")
    try:
        for value in values:
            frame = np.zeros((16, 16, 3), dtype=np.uint8)
            frame[4:12, 4:12] = value
            writer.write(frame)
    finally:
        writer.release()
    return path


def make_transmission(distance: float, path: Path, *, valid: bool = True, reason: str | None = None):
    return Transmission(distance, path, ROI(0.25, 0.25, 0.5, 0.5), valid, reason)


def make_calibration() -> Calibration:
    return Calibration(
        threshold=120,
        point_seconds=0.2,
        tolerance_fraction=0.2,
        roi=ROI(0.25, 0.25, 0.5, 0.5),
        reference_fps=20.0,
    )


def test_validate_distance_set_requires_one_clip_at_each_assignment_distance(tmp_path):
    full_set = [make_transmission(distance, tmp_path / f"{distance}.avi") for distance in (0.5, 1.0, 2.0, 3.0)]
    incomplete = full_set[:-1]

    validate_distance_set(full_set)
    with pytest.raises(ValueError, match="3.0 m"):
        validate_distance_set(incomplete)


def test_duplicate_distance_is_rejected(tmp_path):
    transmissions = [
        make_transmission(0.5, tmp_path / "one.avi"),
        make_transmission(0.5, tmp_path / "two.avi"),
        make_transmission(2.0, tmp_path / "three.avi"),
        make_transmission(3.0, tmp_path / "four.avi"),
    ]

    with pytest.raises(ValueError, match="una sola grabación"):
        validate_distance_set(transmissions)


def test_calibration_roi_dimensions_are_applied_and_each_roi_stays_in_frame(tmp_path):
    transmissions = [
        make_transmission(0.5, tmp_path / "near.avi"),
        Transmission(1.0, tmp_path / "far.avi", ROI(0.8, 0.8, 0.1, 0.1), True, None),
    ]
    calibrated_roi = ROI(0.1, 0.15, 0.4, 0.3)

    updated = apply_calibration_roi_size(transmissions, calibrated_roi)

    assert [(item.roi.width, item.roi.height) for item in updated] == [(0.4, 0.3), (0.4, 0.3)]
    assert updated[0].roi == ROI(0.3, 0.35, 0.4, 0.3)
    assert updated[1].roi == ROI(0.6, 0.7, 0.4, 0.3)


def test_required_analysis_files_follow_invalid_acquisition_status(tmp_path):
    project = make_incomplete_project(tmp_path)
    full_set = required_result_files(project)
    no_far = replace(
        project,
        transmissions=[
            replace(item, acquisition_valid=False, invalid_reason="video cortado")
            if item.distance_m == 3.0
            else item
            for item in project.transmissions
        ],
    )
    no_near = replace(
        project,
        transmissions=[
            replace(item, acquisition_valid=False, invalid_reason="video cortado")
            if item.distance_m == 0.5
            else item
            for item in project.transmissions
        ],
    )

    assert "plots/comparacion_0.5m_3.0m.png" in full_set
    assert "plots/comparacion_0.5m_3.0m.png" not in required_result_files(no_far)
    assert "fft_0.5m.csv" not in required_result_files(no_near)


@pytest.mark.parametrize("invalid_distance", [0.5, 3.0])
def test_export_readiness_allows_reasoned_invalid_edge_acquisition(tmp_path, invalid_distance):
    project = make_incomplete_project(tmp_path)
    project = replace(
        project,
        transmissions=[
            replace(item, acquisition_valid=False, invalid_reason="video cortado")
            if item.distance_m == invalid_distance
            else item
            for item in project.transmissions
        ],
        montage_photo=tmp_path / "montaje.jpg",
    )
    project.montage_photo.write_bytes(b"photo")
    results_dir = tmp_path / "results"
    for relative_path in required_result_files(project):
        output = results_dir / relative_path
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text("resultado", encoding="utf-8")
    (results_dir / "analysis_manifest.json").write_text(
        json.dumps({"signature": analysis_output_signature(project)}), encoding="utf-8"
    )
    source_root = tmp_path / "source"
    source_root.mkdir()
    for filename in ("app.py", "requirements.txt", "requirements-build.txt", "README.md", "run_app.bat", "build_exe.bat", "arduino_morse_leds.ino"):
        (source_root / filename).write_text("fuente", encoding="utf-8")
    package_root = source_root / "siscom"
    package_root.mkdir()
    for module in (
        "__init__.py", "analysis.py", "export.py", "models.py", "morse.py",
        "project.py", "quick_test.py", "paths.py", "reporting.py", "tk_app.py", "ui.py", "ui_helpers.py", "ui_layout.py", "video.py",
    ):
        (package_root / module).write_text("fuente", encoding="utf-8")
    recordings = []
    for transmission in project.transmissions:
        if not transmission.acquisition_valid:
            continue
        signal = VideoSignal(
            times=np.arange(2) / 20.0,
            intensity=np.array([0.0, 255.0]),
            fps=20.0,
            width=16,
            height=16,
            roi=transmission.roi,
        )
        recordings.append(
            ProcessedRecording(
                transmission.video_path,
                signal,
                DecodeResult("E", (), ()),
                decoder_configuration_signature(
                    transmission.video_path, transmission.roi, project.calibration
                ),
            )
        )

    missing = export_readiness(project, recordings, results_dir, source_root)

    assert missing == []


def test_analysis_manifest_must_match_current_project_configuration(tmp_path):
    project = make_incomplete_project(tmp_path)
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "analysis_manifest.json").write_text(
        json.dumps({"signature": analysis_output_signature(project)}), encoding="utf-8"
    )
    changed = replace(
        project,
        calibration=replace(project.calibration, threshold=project.calibration.threshold + 1),
    )

    missing = export_readiness(changed, [], results_dir, tmp_path / "app_source")

    assert any("configuración actual" in item.lower() for item in missing)
    assert analysis_output_signature(project, (1.0, 2.0)) != analysis_output_signature(project)


def test_comparison_start_times_restore_from_saved_analysis_manifest(tmp_path):
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "analysis_manifest.json").write_text(
        json.dumps({"comparison_starts": [1.25, 2.5]}), encoding="utf-8"
    )

    assert load_comparison_starts(results_dir) == (1.25, 2.5)


def test_decoded_recording_is_stale_after_calibration_changes(tmp_path):
    video = make_clip(tmp_path / "captured.avi", ["."])
    original_calibration = make_calibration()
    recording = decode_video_file(video, original_calibration.roi, original_calibration)
    transmission = make_transmission(0.5, video)

    assert decoder_recording_is_current(recording, transmission, original_calibration)
    changed_calibration = replace(original_calibration, threshold=original_calibration.threshold + 1)
    assert not decoder_recording_is_current(recording, transmission, changed_calibration)
    changed_roi = replace(transmission, roi=ROI(0.1, 0.1, 0.5, 0.5))
    assert not decoder_recording_is_current(recording, changed_roi, original_calibration)


def test_comparison_segment_keeps_weak_samples_without_detected_light(tmp_path):
    signal = VideoSignal(
        times=np.arange(100) / 20.0,
        intensity=np.full(100, 90.0),
        fps=20.0,
        width=16,
        height=16,
        roi=ROI(0.25, 0.25, 0.5, 0.5),
    )
    from siscom.ui_helpers import ProcessedRecording

    recording = ProcessedRecording(
        tmp_path / "weak.avi", signal, DecodeResult("", (), ("no_light_pulses",))
    )

    times, intensity = comparison_segment(recording, start_seconds=0.5, duration_seconds=1.0)

    assert len(times) == 20
    assert times[0] == 0.0
    assert intensity.tolist() == [90.0] * 20


def test_switching_projects_clears_old_export_and_processed_state():
    state = {
        "siscom_export_path": "old-project.zip",
        "siscom_export_project_root": "old-project",
        "siscom_recordings": ["old-recording"],
        "siscom_demo_recording": "old-demo",
        "comparison_start_05": 2.5,
        "report_limitation": "old notes",
    }

    reset_project_session(state)

    assert state["siscom_export_path"] is None
    assert state["siscom_recordings"] == []
    assert state["siscom_demo_recording"] is None
    assert "siscom_export_project_root" not in state
    assert "comparison_start_05" not in state
    assert "report_limitation" not in state


def test_find_saved_projects_includes_project_at_extracted_package_root(tmp_path):
    extracted_app = tmp_path / "package"
    managed_projects = extracted_app / "projects"
    (managed_projects / "current").mkdir(parents=True)
    (managed_projects / "current" / "project.json").write_text("{}", encoding="utf-8")
    (extracted_app / "project.json").write_text("{}", encoding="utf-8")

    saved = find_saved_projects(extracted_app, managed_projects)

    assert extracted_app / "project.json" in saved
    assert managed_projects / "current" / "project.json" in saved


def test_replace_transmission_by_path_preserves_distance_with_unsorted_project(tmp_path):
    transmissions = [
        make_transmission(distance, tmp_path / f"{distance}.avi")
        for distance in (2.0, 0.5, 3.0, 1.0)
    ]
    updated_roi = ROI(0.1, 0.2, 0.5, 0.4)
    updated = transmissions[1].__class__(
        transmissions[1].distance_m,
        transmissions[1].video_path,
        updated_roi,
        transmissions[1].acquisition_valid,
        transmissions[1].invalid_reason,
    )

    result = replace_transmission_by_path(transmissions, updated)

    assert [item.distance_m for item in result] == [2.0, 0.5, 3.0, 1.0]
    assert result[1].roi == updated_roi
    assert result[0].roi == transmissions[0].roi


def test_decoder_handles_arbitrary_four_letter_demo_without_cer(tmp_path):
    # Hand-written Morse: T -, E ., S ..., T -.
    video = make_clip(tmp_path / "demo_test.avi", ["-", ".", "...", "-"])

    decoded = decode_video_file(video, ROI(0.25, 0.25, 0.5, 0.5), make_calibration())

    assert decoded.decode.text == "TEST"
    assert not hasattr(decoded, "cer")
    assert "expected_text" not in signature(decode_video_file).parameters


def test_demo_roi_origin_is_clamped_to_keep_the_full_roi_inside_the_frame():
    roi = ROI(0.35, 0.35, 0.3, 0.35)

    assert callable(getattr(ui_helpers, "roi_with_origin", None))
    moved = ui_helpers.roi_with_origin(roi, 0.99, 0.99)

    assert moved == ROI(0.7, 0.65, 0.3, 0.35)


def test_result_table_includes_required_metrics_and_keeps_empty_valid_decode_as_cer_one(tmp_path):
    video = make_clip(tmp_path / "empty_valid.avi", [])
    signal = VideoSignal(
        times=np.arange(28) / 20.0,
        intensity=np.zeros(28),
        fps=20.0,
        width=16,
        height=16,
        roi=ROI(0.25, 0.25, 0.5, 0.5),
    )
    from siscom.ui_helpers import ProcessedRecording

    empty_record = ProcessedRecording(video, signal, DecodeResult("", (), ("no_light_pulses",)))
    transmission = make_transmission(0.5, video)

    table = build_results_table([transmission], [empty_record], "SIS COM")
    row = table.iloc[0]

    assert {"Distancia (m)", "Duración (s)", "S", "D", "I", "S + D + I", "CER", "Texto recibido", "Adquisición"}.issubset(table.columns)
    assert row["CER"] == 1.0
    assert row["Texto recibido"] == ""
    assert row["Adquisición"] == "válida"


def test_invalid_acquisition_table_has_status_and_no_cer(tmp_path):
    invalid_video = tmp_path / "failed.avi"
    transmission = make_transmission(1.0, invalid_video, valid=False, reason="grabación incompleta")

    table = build_results_table([transmission], [], "SIS COM")
    row = table.iloc[0]

    assert row["CER"] is None
    assert row["Adquisición"] == "sin adquisición válida"
    assert row["Motivo"] == "grabación incompleta"


def make_incomplete_project(tmp_path: Path) -> Project:
    video_dir = tmp_path / "videos" / "transmissions"
    video_dir.mkdir(parents=True)
    calibration_video = tmp_path / "videos" / "calibration.avi"
    calibration_video.parent.mkdir(parents=True, exist_ok=True)
    calibration_video.write_bytes(b"calibration")
    transmissions = [
        make_transmission(distance, video_dir / f"{distance}.avi")
        for distance in (0.5, 1.0, 2.0, 3.0)
    ]
    for transmission in transmissions:
        transmission.video_path.write_bytes(b"video")
    return Project(
        root=tmp_path,
        calibration_video=calibration_video,
        transmissions=transmissions,
        calibration=make_calibration(),
        expected_text="SIS COM",
        team=(TeamMember("Ana", 100), TeamMember("Luis", 100)),
        setup=SetupMetadata("manual", "LED", "botón", "trípodes", "interior", "constante"),
        references=("ITU-R M.1677-1",),
        ai_assistance="Codex para programar; citar el uso en el informe.",
        montage_video=None,
        montage_photo=None,
    )


def test_export_readiness_lists_missing_results_and_required_setup_photo(tmp_path):
    project = make_incomplete_project(tmp_path)
    source_root = tmp_path / "app_source"
    source_root.mkdir()

    missing = export_readiness(project, [], tmp_path / "results", source_root)

    assert any("foto del montaje" in item.lower() for item in missing)
    assert any("procesa" in item.lower() or "resultados" in item.lower() for item in missing)


def test_export_readiness_requires_the_actual_result_table_and_analysis_files(tmp_path):
    project = make_incomplete_project(tmp_path)
    results_dir = tmp_path / "results"
    results_dir.mkdir()
    (results_dir / "unrelated.txt").write_text("not an analysis", encoding="utf-8")
    source_root = tmp_path / "app_source"
    source_root.mkdir()

    missing = export_readiness(project, [], results_dir, source_root)

    assert any("tabla de resultados" in item.lower() for item in missing)


def test_export_readiness_requires_launcher_and_core_package_sources(tmp_path):
    project = make_incomplete_project(tmp_path)
    source_root = tmp_path / "source"
    source_root.mkdir()
    for filename in ("app.py", "requirements.txt", "README.md"):
        (source_root / filename).write_text("source", encoding="utf-8")

    missing = export_readiness(project, [], tmp_path / "results", source_root)

    assert any("run_app.bat" in item for item in missing)
    assert any("arduino_morse_leds.ino" in item for item in missing)
    assert any("siscom/ui.py" in item for item in missing)
    assert any("siscom/quick_test.py" in item for item in missing)


def test_desktop_entrypoint_is_tkinter_and_does_not_use_a_web_framework():
    app_root = Path(__file__).resolve().parents[1]
    app_source = (app_root / "app.py").read_text(encoding="utf-8").lower()
    ui_source = (app_root / "siscom" / "ui.py").read_text(encoding="utf-8").lower()
    requirements = (app_root / "requirements.txt").read_text(encoding="utf-8").lower()

    assert "streamlit" not in app_source
    assert "streamlit" not in ui_source
    assert "streamlit" not in requirements
    assert "tkinter" in ui_source or "tkinter" in (app_root / "siscom" / "tk_app.py").read_text(encoding="utf-8").lower()


def test_desktop_app_module_exposes_all_six_workflow_pages():
    from siscom.ui_layout import WORKFLOW_PAGES

    assert WORKFLOW_PAGES == (
        "Proyecto", "Prueba rápida", "Calibración", "Decodificación", "Análisis", "Exportación"
    )

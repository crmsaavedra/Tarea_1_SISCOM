from pathlib import Path
from types import SimpleNamespace
from tempfile import TemporaryDirectory

import pytest

from siscom.models import Calibration, ROI, Project, SetupMetadata, TeamMember, Transmission
from siscom.project import (
    evaluate_transmission,
    import_video,
    load_project,
    save_project,
    validate_montage_duration,
)


def make_project(root: Path) -> Project:
    video_dir = root / "videos"
    calibration_dir = video_dir / "calibration"
    transmission_dir = video_dir / "transmissions"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    transmission_dir.mkdir(parents=True, exist_ok=True)
    calibration_video = calibration_dir / "calibracion.avi"
    calibration_video.write_bytes(b"calibration video")
    transmissions = []
    for distance in (0.5, 1.0, 2.0, 3.0):
        video = transmission_dir / f"sis_com_{distance:.1f}m.avi"
        video.write_bytes(f"full video at {distance}m".encode())
        transmissions.append(
            Transmission(distance, video, ROI(0.2, 0.2, 0.4, 0.4), True, None)
        )
    calibration = Calibration(128.0, 0.4, 0.2, ROI(0.2, 0.2, 0.4, 0.4), 25.0)
    return Project(
        root=root,
        calibration_video=calibration_video,
        transmissions=transmissions,
        calibration=calibration,
        expected_text="SIS COM",
        team=(TeamMember("Ana Pérez", 60), TeamMember("Luis Soto", 60)),
        setup=SetupMetadata("manual", "linterna LED", "botón", "trípode y soporte", "interior", "100 %"),
        references=("UIT-R M.1677-1",),
        ai_assistance="Asistencia en código y depuración.",
        montage_video=None,
        montage_photo=None,
    )


def test_project_saves_and_reopens_relative_video_paths():
    # Workspace temp is intentional: the sandbox denies os.replace inside its
    # redirected AC Temp directory, while real project folders use workspace IO.
    with TemporaryDirectory(prefix=".project-test-", dir=Path.cwd()) as folder:
        root = Path(folder) / "proyecto"
        project = make_project(root)

        project_file = save_project(project, root)
        loaded = load_project(project_file)

        assert project_file == root / "project.json"
        assert loaded.root == root
        assert loaded.expected_text == "SIS COM"
        assert [item.distance_m for item in loaded.transmissions] == [0.5, 1.0, 2.0, 3.0]
        assert loaded.transmissions[0].video_path == root / "videos" / "transmissions" / "sis_com_0.5m.avi"
        assert loaded.calibration.threshold == 128.0
        assert loaded.team[0].contribution_percent == 60
        assert loaded.setup.emission_mode == "manual"


def test_import_video_copies_source_without_changing_it(tmp_path):
    project_root = tmp_path / "project"
    source = tmp_path / "original.avi"
    original_bytes = b"the original recording"
    source.write_bytes(original_bytes)

    imported = import_video(project_root, source, "transmission")

    assert imported.read_bytes() == original_bytes
    assert source.read_bytes() == original_bytes
    assert imported.parent == project_root / "videos" / "transmissions"


def test_import_video_uses_collision_safe_local_name(tmp_path):
    project_root = tmp_path / "project"
    first = tmp_path / "first.avi"
    second = tmp_path / "second.avi"
    first.write_bytes(b"first")
    second.write_bytes(b"second")

    imported_first = import_video(project_root, first, "calibration")
    imported_second = import_video(project_root, second, "calibration")

    assert imported_first.name == "first.avi"
    assert imported_second.name == "second.avi"
    imported_duplicate = import_video(project_root, first, "calibration")
    assert imported_duplicate.name == "first_2.avi"


def test_native_file_import_keeps_a_safe_version_of_the_original_filename(tmp_path):
    source = tmp_path / "Morse 0.5m video.avi"
    source.write_bytes(b"uploaded bytes")

    imported = import_video(tmp_path / "project", source, "transmission")

    assert imported.name == "Morse_0.5m_video.avi"
    assert imported.read_bytes() == b"uploaded bytes"


def test_import_video_rejects_unknown_category_and_path_traversal(tmp_path):
    source = tmp_path / "clip.avi"
    source.write_bytes(b"video")

    with pytest.raises(ValueError, match="category"):
        import_video(tmp_path / "project", source, "../../outside")


def test_team_has_exactly_two_members_and_percentages_need_not_sum_to_100(tmp_path):
    project = make_project(tmp_path / "test-project")

    assert sum(member.contribution_percent for member in project.team) == 120
    with pytest.raises(ValueError, match="exactly two"):
        Project(
            root=project.root,
            calibration_video=project.calibration_video,
            transmissions=project.transmissions,
            calibration=project.calibration,
            expected_text=project.expected_text,
            team=(TeamMember("Ana", 100),),
            setup=project.setup,
            references=(),
            ai_assistance="",
            montage_video=None,
            montage_photo=None,
        )


@pytest.mark.parametrize("contribution", [-1, 101])
def test_team_contribution_percentage_must_be_between_zero_and_100(contribution):
    with pytest.raises(ValueError, match="0 and 100"):
        TeamMember("Ana", contribution)


def test_invalid_acquisition_requires_a_reason(tmp_path):
    with pytest.raises(ValueError, match="reason"):
        Transmission(1.0, tmp_path / "missing.avi", ROI(0.1, 0.1, 0.2, 0.2), False, " ")


def test_invalid_acquisition_has_no_cer_but_valid_empty_decode_has_cer_one():
    invalid = evaluate_transmission("SIS COM", "", acquisition_valid=False, invalid_reason="Clip cortado")
    empty = evaluate_transmission("SIS COM", "", acquisition_valid=True, invalid_reason=None)

    assert invalid.cer is None
    assert invalid.acquisition_status == "sin adquisición válida"
    assert empty.cer == 1.0
    assert empty.acquisition_status == "válida"


def test_montage_duration_is_limited_to_two_minutes():
    validate_montage_duration(120.0)

    with pytest.raises(ValueError, match="120"):
        validate_montage_duration(120.01)

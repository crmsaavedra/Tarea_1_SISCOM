from pathlib import Path
from tempfile import TemporaryDirectory
from zipfile import ZipFile
from dataclasses import replace

from siscom.export import create_results_zip
from siscom.models import Calibration, Project, ROI, SetupMetadata, TeamMember, Transmission
from siscom.project import load_project
from siscom.ui_helpers import find_saved_projects


def make_export_fixture(tmp_path: Path, with_montage: bool = False):
    project_root = tmp_path / "project"
    calibration_video = project_root / "videos" / "calibration" / "calibration.avi"
    calibration_video.parent.mkdir(parents=True)
    calibration_video.write_bytes(b"calibration")
    transmissions = []
    for distance in (0.5, 1.0, 2.0, 3.0):
        video = project_root / "videos" / "transmissions" / f"{distance:.1f}m.avi"
        video.parent.mkdir(parents=True, exist_ok=True)
        video.write_bytes(f"video-{distance}".encode())
        transmissions.append(Transmission(distance, video, ROI(0.2, 0.2, 0.4, 0.4), True, None))

    montage = project_root / "evidence" / "montage.mp4" if with_montage else None
    photo = project_root / "evidence" / "setup.jpg"
    photo.parent.mkdir(parents=True)
    photo.write_bytes(b"setup photo")
    if montage:
        montage.write_bytes(b"demo montage")

    calibration = Calibration(127, 0.4, 0.2, ROI(0.2, 0.2, 0.4, 0.4), 25)
    project = Project(
        root=project_root,
        calibration_video=calibration_video,
        transmissions=transmissions,
        calibration=calibration,
        expected_text="SIS COM",
        team=(TeamMember("Ana Pérez", 60), TeamMember("Luis Soto", 60)),
        setup=SetupMetadata("manual", "linterna", "botón", "trípode", "interior", "alto"),
        references=("UIT-R M.1677-1",),
        ai_assistance="Asistencia declarada.",
        montage_video=montage,
        montage_photo=photo,
    )

    source_root = tmp_path / "source"
    (source_root / "siscom").mkdir(parents=True)
    (source_root / "app.py").write_text("print('app')\n", encoding="utf-8")
    for module in (
        "__init__.py", "analysis.py", "export.py", "models.py", "morse.py",
        "project.py", "ui.py", "ui_helpers.py", "video.py",
    ):
        (source_root / "siscom" / module).write_text("# source\n", encoding="utf-8")
    (source_root / "requirements.txt").write_text("opencv-python-headless\n", encoding="utf-8")
    (source_root / "README.md").write_text("Run the app.\n", encoding="utf-8")
    (source_root / "run_app.bat").write_text("@echo off\n", encoding="utf-8")
    (source_root / "arduino_morse_leds.ino").write_text("// Arduino transmitter\n", encoding="utf-8")

    results = tmp_path / "results"
    (results / "plots").mkdir(parents=True)
    (results / "tabla_resultados.csv").write_text("distance,cer\n0.5,0\n", encoding="utf-8")
    (results / "plots" / "cer.png").write_bytes(b"plot")
    return project, results, source_root


def test_results_zip_contains_source_videos_calibration_outputs_and_readme():
    with TemporaryDirectory(prefix=".export-test-", dir=Path.cwd()) as folder:
        tmp_path = Path(folder)
        project, results, source_root = make_export_fixture(tmp_path)
        output = tmp_path / "entrega.zip"

        create_results_zip(project, results, output, source_root)

        with ZipFile(output) as archive:
            names = set(archive.namelist())
            assert "app.py" in names
            assert "siscom/morse.py" in names
            assert "siscom/ui.py" in names
            assert "requirements.txt" in names
            assert "README.md" in names
            assert "run_app.bat" in names
            assert "arduino_morse_leds.ino" in names
            assert "videos/calibration/calibration.avi" in names
            assert all(f"videos/transmissions/{distance:.1f}m.avi" in names for distance in (0.5, 1.0, 2.0, 3.0))
            assert "results/tabla_resultados.csv" in names
            assert "results/plots/cer.png" in names
            assert "evidence/setup.jpg" in names
            assert "expected_morse.txt" in names
            assert "project.json" in names
            assert all(not name.startswith(("C:/", "C:\\", "/")) for name in names)
            assert archive.read("expected_morse.txt").decode().strip() == "... .. ... / -.-. --- --"


def test_results_zip_includes_montage_only_when_project_has_one():
    with TemporaryDirectory(prefix=".export-montage-test-", dir=Path.cwd()) as folder:
        tmp_path = Path(folder)
        project, results, source_root = make_export_fixture(tmp_path, with_montage=True)
        output = tmp_path / "entrega.zip"

        create_results_zip(project, results, output, source_root)

        with ZipFile(output) as archive:
            assert "evidence/montage.mp4" in archive.namelist()


def test_results_zip_omits_comparison_for_invalid_far_acquisition():
    with TemporaryDirectory(prefix=".export-invalid-test-", dir=Path.cwd()) as folder:
        tmp_path = Path(folder)
        project, results, source_root = make_export_fixture(tmp_path)
        project = replace(
            project,
            transmissions=[
                replace(item, acquisition_valid=False, invalid_reason="video cortado")
                if item.distance_m == 3.0
                else item
                for item in project.transmissions
            ],
        )
        comparison = results / "plots" / "comparacion_0.5m_3.0m.png"
        comparison.write_bytes(b"stale comparison")
        output = tmp_path / "entrega.zip"

        create_results_zip(project, results, output, source_root)

        with ZipFile(output) as archive:
            assert "results/plots/comparacion_0.5m_3.0m.png" not in archive.namelist()


def test_results_zip_omits_spectrum_and_comparison_for_invalid_near_acquisition():
    with TemporaryDirectory(prefix=".export-invalid-near-test-", dir=Path.cwd()) as folder:
        tmp_path = Path(folder)
        project, results, source_root = make_export_fixture(tmp_path)
        project = replace(
            project,
            transmissions=[
                replace(item, acquisition_valid=False, invalid_reason="video cortado")
                if item.distance_m == 0.5
                else item
                for item in project.transmissions
            ],
        )
        stale_results = (
            "fft_0.5m.csv",
            "psd_0.5m.csv",
            "intensidad_0.5m.csv",
            "plots/senal_temporal_0.5m.png",
            "plots/fft_0.5m.png",
            "plots/psd_b95_0.5m.png",
            "plots/comparacion_0.5m_3.0m.png",
        )
        for relative in stale_results:
            path = results / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(b"stale")
        output = tmp_path / "entrega.zip"

        create_results_zip(project, results, output, source_root)

        with ZipFile(output) as archive:
            names = set(archive.namelist())
            assert all(f"results/{relative}" not in names for relative in stale_results)


def test_extracted_results_zip_reopens_project_at_package_root():
    with TemporaryDirectory(prefix=".export-reopen-test-", dir=Path.cwd()) as folder:
        tmp_path = Path(folder)
        project, results, source_root = make_export_fixture(tmp_path)
        output = tmp_path / "entrega.zip"
        extracted = tmp_path / "extracted"

        create_results_zip(project, results, output, source_root)
        with ZipFile(output) as archive:
            archive.extractall(extracted)

        reopened = load_project(extracted / "project.json")

        assert reopened.root == extracted
        assert reopened.team[0].name == "Ana Pérez"
        assert reopened.transmissions[3].video_path == extracted / "videos" / "transmissions" / "3.0m.avi"
        assert find_saved_projects(extracted, extracted / "projects") == [extracted / "project.json"]

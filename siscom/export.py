"""Build a portable ZIP containing the app, project media, and analysis outputs."""

from __future__ import annotations

from pathlib import Path
from zipfile import ZIP_DEFLATED, ZipFile

from .morse import encode_text_morse
from .models import Project
from .project import ProjectError, save_project


_SOURCE_FILES = {"app.py", "requirements.txt", "requirements-build.txt", "README.md", "run_app.bat", "build_exe.bat", "arduino_morse_leds.ino"}


def _add_project_file(archive: ZipFile, root: Path, path: Path) -> None:
    root = root.resolve()
    path = path.resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ProjectError(f"Export asset is outside its project folder: {path.name}") from error
    if not path.is_file():
        raise ProjectError(f"Export asset is missing: {relative.as_posix()}")
    archive.write(path, relative.as_posix())


def _source_files(source_root: Path) -> list[Path]:
    source_root = Path(source_root).resolve()
    files = [source_root / name for name in sorted(_SOURCE_FILES) if (source_root / name).is_file()]
    package_dir = source_root / "siscom"
    if package_dir.is_dir():
        files.extend(sorted(path for path in package_dir.rglob("*.py") if path.is_file()))
    if not files:
        raise ProjectError("No app source files were found for the reproducibility ZIP")
    return files


def create_results_zip(
    project: Project,
    results_dir: Path,
    output_path: Path,
    source_root: Path,
) -> Path:
    """Package relative, project-owned data and app source without `.venv` or absolute paths."""
    root = Path(project.root).resolve()
    results_dir = Path(results_dir).resolve()
    output_path = Path(output_path).resolve()
    source_root = Path(source_root).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    save_project(project, root)

    with ZipFile(output_path, mode="w", compression=ZIP_DEFLATED, compresslevel=6) as archive:
        for source in _source_files(source_root):
            archive.write(source, source.relative_to(source_root).as_posix())

        _add_project_file(archive, root, project.calibration_video)
        for transmission in project.transmissions:
            _add_project_file(archive, root, transmission.video_path)
        for evidence in (project.montage_photo, project.montage_video):
            if evidence is not None:
                _add_project_file(archive, root, evidence)
        _add_project_file(archive, root, root / "project.json")

        archive.writestr("expected_morse.txt", encode_text_morse(project.expected_text) + "\n")
        results_resolved = results_dir.resolve()
        valid_distances = {
            round(item.distance_m, 2)
            for item in project.transmissions
            if item.acquisition_valid
        }
        excluded_results = set()
        if 0.5 not in valid_distances:
            excluded_results.update(
                {
                    "intensidad_0.5m.csv",
                    "fft_0.5m.csv",
                    "psd_0.5m.csv",
                    "plots/senal_temporal_0.5m.png",
                    "plots/fft_0.5m.png",
                    "plots/psd_b95_0.5m.png",
                }
            )
        if 0.5 not in valid_distances or 3.0 not in valid_distances:
            excluded_results.add("plots/comparacion_0.5m_3.0m.png")
        for transmission in project.transmissions:
            if not transmission.acquisition_valid:
                excluded_results.add(f"intensidad_{transmission.distance_m:.1f}m.csv")
        if results_resolved.is_dir():
            for result in sorted(results_resolved.rglob("*")):
                if not result.is_file() or result.resolve() == output_path:
                    continue
                try:
                    relative = result.resolve().relative_to(results_resolved)
                except ValueError as error:
                    raise ProjectError(f"Result file is outside results folder: {result.name}") from error
                if relative.as_posix() in excluded_results:
                    continue
                archive.write(result, (Path("results") / relative).as_posix())

    return output_path

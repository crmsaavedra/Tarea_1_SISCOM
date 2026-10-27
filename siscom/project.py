"""Portable project storage, safe video imports, and acquisition scoring."""

from __future__ import annotations

import json
import hashlib
import re
import shutil
from dataclasses import asdict, dataclass
from pathlib import Path

from .analysis import ErrorCounts, character_error_rate, levenshtein_counts
from .models import Calibration, Project, ROI, SetupMetadata, TeamMember, Transmission


class ProjectError(ValueError):
    """A project file or referenced local asset cannot be used safely."""


@dataclass(frozen=True)
class TransmissionEvaluation:
    received_text: str
    acquisition_status: str
    invalid_reason: str | None
    errors: ErrorCounts | None
    cer: float | None


_IMPORT_DIRECTORIES = {
    "calibration": Path("videos") / "calibration",
    "transmission": Path("videos") / "transmissions",
    "demo": Path("videos") / "demos",
    "photo": Path("evidence"),
}


def analysis_output_signature(
    project: Project, comparison_starts: tuple[float, float] = (0.0, 0.0)
) -> str:
    """Fingerprint every project setting that changes decoded or plotted results."""
    if project.calibration is None:
        raise ProjectError("Project must be calibrated before result signature can be generated")
    root = Path(project.root).resolve()

    def asset_record(path: Path) -> dict[str, str | int]:
        resolved = Path(path).resolve()
        try:
            relative = resolved.relative_to(root)
        except ValueError as error:
            raise ProjectError("Project media must remain inside its project folder") from error
        try:
            size = resolved.stat().st_size
        except OSError as error:
            raise ProjectError(f"Missing analysis input: {relative.as_posix()}") from error
        return {"path": relative.as_posix(), "size": size}

    transmissions = []
    for item in sorted(project.transmissions, key=lambda entry: entry.distance_m):
        transmissions.append(
            {
                "distance_m": item.distance_m,
                "video": asset_record(item.video_path),
                "roi": asdict(item.roi),
                "acquisition_valid": item.acquisition_valid,
                "invalid_reason": item.invalid_reason,
            }
        )
    payload = {
        "expected_text": project.expected_text,
        "calibration_video": asset_record(project.calibration_video),
        "calibration": _calibration_to_json(project.calibration),
        "transmissions": transmissions,
        "comparison_starts": [round(float(value), 4) for value in comparison_starts],
    }
    normalized = json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def _safe_filename(name: str) -> str:
    filename = Path(name).name
    filename = re.sub(r"[^A-Za-z0-9._-]+", "_", filename).strip("._")
    if not filename:
        raise ValueError("File name has no usable characters")
    return filename


def import_video(
    project_root: Path,
    source: Path,
    category: str,
    *,
    filename: str | None = None,
) -> Path:
    """Copy a video or evidence image into a category folder without overwriting."""
    if category not in _IMPORT_DIRECTORIES:
        raise ValueError(f"Unknown import category: {category}")
    source = Path(source)
    if not source.is_file():
        raise FileNotFoundError(f"Source file does not exist: {source}")

    destination_directory = Path(project_root) / _IMPORT_DIRECTORIES[category]
    destination_directory.mkdir(parents=True, exist_ok=True)
    safe_filename = _safe_filename(filename if filename is not None else source.name)
    destination = destination_directory / safe_filename
    stem, suffix = destination.stem, destination.suffix
    counter = 2
    while destination.exists():
        destination = destination_directory / f"{stem}_{counter}{suffix}"
        counter += 1
    shutil.copy2(source, destination)
    return destination


def validate_montage_duration(duration_seconds: float) -> None:
    if duration_seconds <= 0.0:
        raise ValueError("Montage video duration must be greater than zero")
    if duration_seconds > 120.0:
        raise ValueError("Montage video must be no longer than 120 seconds")


def evaluate_transmission(
    reference: str,
    received_text: str,
    *,
    acquisition_valid: bool,
    invalid_reason: str | None,
) -> TransmissionEvaluation:
    if not acquisition_valid:
        if not (invalid_reason or "").strip():
            raise ValueError("An invalid acquisition requires a reason")
        return TransmissionEvaluation(
            received_text=received_text,
            acquisition_status="sin adquisición válida",
            invalid_reason=invalid_reason.strip(),
            errors=None,
            cer=None,
        )
    return TransmissionEvaluation(
        received_text=received_text,
        acquisition_status="válida",
        invalid_reason=None,
        errors=levenshtein_counts(reference, received_text),
        cer=character_error_rate(reference, received_text),
    )


def _relative_file(root: Path, path: Path | None) -> str | None:
    if path is None:
        return None
    root = root.resolve()
    path = Path(path).resolve()
    try:
        relative = path.relative_to(root)
    except ValueError as error:
        raise ProjectError(f"Project asset is outside the project folder: {path.name}") from error
    if not path.is_file():
        raise ProjectError(f"Project asset is missing: {relative.as_posix()}")
    return relative.as_posix()


def _calibration_to_json(calibration: Calibration | None) -> dict | None:
    if calibration is None:
        return None
    return {
        "threshold": calibration.threshold,
        "point_seconds": calibration.point_seconds,
        "tolerance_fraction": calibration.tolerance_fraction,
        "roi": asdict(calibration.roi),
        "reference_fps": calibration.reference_fps,
    }


def save_project(project: Project, root: Path) -> Path:
    """Write portable relative asset paths and settings to `project.json`."""
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": 1,
        "expected_text": project.expected_text,
        "calibration_video": _relative_file(root, project.calibration_video),
        "calibration": _calibration_to_json(project.calibration),
        "transmissions": [
            {
                "distance_m": item.distance_m,
                "video_path": _relative_file(root, item.video_path),
                "roi": asdict(item.roi),
                "acquisition_valid": item.acquisition_valid,
                "invalid_reason": item.invalid_reason,
            }
            for item in project.transmissions
        ],
        "team": [asdict(member) for member in project.team],
        "setup": asdict(project.setup),
        "references": list(project.references),
        "ai_assistance": project.ai_assistance,
        "montage_video": _relative_file(root, project.montage_video),
        "montage_photo": _relative_file(root, project.montage_photo),
    }
    destination = root / "project.json"
    temporary = root / "project.json.tmp"
    temporary.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(destination)
    return destination


def _project_asset(root: Path, relative: str | None, label: str) -> Path | None:
    if relative is None:
        return None
    root = root.resolve()
    path = (root / relative).resolve()
    try:
        path.relative_to(root)
    except ValueError as error:
        raise ProjectError(f"Unsafe path for {label}; project assets must be relative") from error
    if not path.is_file():
        raise ProjectError(f"Missing {label}: {relative}")
    return path


def load_project(project_file: Path) -> Project:
    """Load a project and verify all serialized asset paths remain local."""
    project_file = Path(project_file).resolve()
    root = project_file.parent
    try:
        payload = json.loads(project_file.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ProjectError(f"Cannot read project file: {project_file.name}") from error
    if payload.get("schema_version") != 1:
        raise ProjectError("Unsupported project file version")

    calibration_data = payload.get("calibration")
    calibration = None
    if calibration_data is not None:
        calibration = Calibration(
            threshold=calibration_data["threshold"],
            point_seconds=calibration_data["point_seconds"],
            tolerance_fraction=calibration_data["tolerance_fraction"],
            roi=ROI(**calibration_data["roi"]),
            reference_fps=calibration_data["reference_fps"],
        )

    calibration_video = _project_asset(root, payload.get("calibration_video"), "calibration video")
    if calibration_video is None:
        raise ProjectError("Project is missing its calibration video")
    transmissions = []
    for item in payload.get("transmissions", []):
        video_path = _project_asset(root, item.get("video_path"), "transmission video")
        if video_path is None:
            raise ProjectError("Project transmission is missing a video path")
        transmissions.append(
            Transmission(
                distance_m=item["distance_m"],
                video_path=video_path,
                roi=ROI(**item["roi"]),
                acquisition_valid=item["acquisition_valid"],
                invalid_reason=item.get("invalid_reason"),
            )
        )

    team = tuple(TeamMember(**member) for member in payload.get("team", []))
    if len(team) != 2:
        raise ProjectError("Project must contain exactly two team members")
    optional_video = _project_asset(root, payload.get("montage_video"), "montage video")
    optional_photo = _project_asset(root, payload.get("montage_photo"), "montage photo")
    return Project(
        root=root,
        calibration_video=calibration_video,
        transmissions=transmissions,
        calibration=calibration,
        expected_text=payload.get("expected_text", "SIS COM"),
        team=(team[0], team[1]),
        setup=SetupMetadata(**payload["setup"]),
        references=tuple(payload.get("references", [])),
        ai_assistance=payload.get("ai_assistance", ""),
        montage_video=optional_video,
        montage_photo=optional_photo,
    )

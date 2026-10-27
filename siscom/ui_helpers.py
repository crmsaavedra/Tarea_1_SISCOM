"""Testable workflow helpers used by the desktop interface."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from math import isfinite
from pathlib import Path

import numpy as np
import pandas as pd

from .analysis import character_error_rate
from .models import Calibration, DecodeResult, Project, ROI, Transmission, VideoSignal
from .morse import decode_signal
from .project import analysis_output_signature, evaluate_transmission, validate_montage_duration
from .video import VideoReadError, extract_video_signal, probe_video


EXPECTED_DISTANCES_M = (0.5, 1.0, 2.0, 3.0)
CORE_RESULT_FILES = {
    "tabla_resultados.csv": "la tabla de resultados",
    "textos_recibidos.txt": "los textos recibidos",
    "decisiones_morse.csv": "las decisiones Morse",
    "notas_para_informe.md": "las notas para el informe",
    "plots/cer_distancia.png": "el gráfico CER por distancia",
}
SPECTRUM_RESULT_FILES = {
    "fft_0.5m.csv": "los datos FFT de 0,5 m",
    "psd_0.5m.csv": "los datos PSD de 0,5 m",
    "plots/senal_temporal_0.5m.png": "el gráfico temporal de 0,5 m",
    "plots/fft_0.5m.png": "el gráfico FFT de 0,5 m",
    "plots/psd_b95_0.5m.png": "el gráfico PSD y B95 de 0,5 m",
}
COMPARISON_RESULT_FILES = {
    "plots/comparacion_0.5m_3.0m.png": "la comparación entre 0,5 m y 3,0 m",
}


def find_saved_projects(app_root: Path, projects_root: Path) -> list[Path]:
    """Find managed projects and a project stored at an extracted ZIP's root."""
    app_root = Path(app_root)
    projects_root = Path(projects_root)
    candidates = []
    package_project = app_root / "project.json"
    if package_project.is_file():
        candidates.append(package_project)
    if projects_root.is_dir():
        candidates.extend(path for path in projects_root.glob("*/project.json") if path.is_file())
    return sorted(set(candidates), key=lambda path: str(path).casefold())


def replace_transmission_by_path(
    transmissions: list[Transmission], updated: Transmission
) -> list[Transmission]:
    """Replace one clip without depending on the project's list ordering."""
    target = Path(updated.video_path).resolve()
    if not any(Path(item.video_path).resolve() == target for item in transmissions):
        raise ValueError("No se encontró la grabación que se intenta actualizar.")
    return [
        updated if Path(item.video_path).resolve() == target else item
        for item in transmissions
    ]


def apply_calibration_roi_size(
    transmissions: list[Transmission], calibration_roi: ROI
) -> list[Transmission]:
    """Share the calibrated ROI dimensions while preserving each clip's center."""
    updated = []
    for transmission in transmissions:
        center_x = transmission.roi.left + transmission.roi.width / 2.0
        center_y = transmission.roi.top + transmission.roi.height / 2.0
        left = min(max(center_x - calibration_roi.width / 2.0, 0.0), 1.0 - calibration_roi.width)
        top = min(max(center_y - calibration_roi.height / 2.0, 0.0), 1.0 - calibration_roi.height)
        roi = ROI(left, top, calibration_roi.width, calibration_roi.height)
        updated.append(replace(transmission, roi=roi))
    return updated


def roi_with_origin(template: ROI, left: float, top: float) -> ROI:
    """Move an ROI while keeping its full extent inside the normalized frame."""
    max_left = 1.0 - template.width
    max_top = 1.0 - template.height
    safe_left = min(max(float(left), 0.0), max_left)
    safe_top = min(max(float(top), 0.0), max_top)
    return ROI(safe_left, safe_top, template.width, template.height)


@dataclass(frozen=True)
class ProcessedRecording:
    video_path: Path
    signal: VideoSignal
    decode: DecodeResult
    configuration_signature: str | None = None

    @property
    def message_duration_seconds(self) -> float:
        light_runs = [run for run in self.decode.runs if run.is_light]
        if not light_runs:
            return 0.0
        return (light_runs[-1].end_frame - light_runs[0].start_frame) / self.signal.fps


def validate_distance_set(transmissions: list[Transmission]) -> None:
    distances = [round(transmission.distance_m, 2) for transmission in transmissions]
    duplicates = sorted({distance for distance in distances if distances.count(distance) > 1})
    if duplicates:
        rendered = ", ".join(f"{distance:.1f} m" for distance in duplicates)
        raise ValueError(f"Debe haber una sola grabación por distancia; se repite {rendered}.")
    missing = [distance for distance in EXPECTED_DISTANCES_M if distance not in distances]
    extra = [distance for distance in distances if distance not in EXPECTED_DISTANCES_M]
    if missing:
        rendered = ", ".join(f"{distance:.1f} m" for distance in missing)
        raise ValueError(f"Falta una grabación a {rendered}.")
    if extra or len(distances) != len(EXPECTED_DISTANCES_M):
        raise ValueError("Carga solo las cuatro distancias requeridas: 0,5 m, 1,0 m, 2,0 m y 3,0 m.")


def decode_video_file(video_path: Path, roi: ROI, calibration: Calibration) -> ProcessedRecording:
    """Decode a video from its measured intensity only; it accepts no expected text."""
    signal = extract_video_signal(Path(video_path), roi)
    decoded = decode_signal(signal.intensity, signal.fps, calibration)
    return ProcessedRecording(
        Path(video_path), signal, decoded,
        decoder_configuration_signature(Path(video_path), roi, calibration),
    )


def decoder_configuration_signature(video_path: Path, roi: ROI, calibration: Calibration) -> str:
    path = Path(video_path).resolve()
    payload = {
        "path": str(path),
        "size": path.stat().st_size,
        "roi": {"left": roi.left, "top": roi.top, "width": roi.width, "height": roi.height},
        "calibration": {
            "threshold": calibration.threshold,
            "point_seconds": calibration.point_seconds,
            "tolerance_fraction": calibration.tolerance_fraction,
            "roi": {
                "left": calibration.roi.left,
                "top": calibration.roi.top,
                "width": calibration.roi.width,
                "height": calibration.roi.height,
            },
            "reference_fps": calibration.reference_fps,
        },
    }
    normalized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def decoder_recording_is_current(
    recording: ProcessedRecording, transmission: Transmission, calibration: Calibration
) -> bool:
    try:
        expected = decoder_configuration_signature(
            transmission.video_path, transmission.roi, calibration
        )
    except OSError:
        return False
    return (
        recording.video_path.resolve() == transmission.video_path.resolve()
        and recording.configuration_signature == expected
    )


def current_recordings(
    project: Project, recordings: list[ProcessedRecording]
) -> list[ProcessedRecording]:
    """Keep only decodes made with the current calibration, ROI, and valid clips."""
    by_path = {
        transmission.video_path.resolve(): transmission
        for transmission in project.transmissions
        if transmission.acquisition_valid
    }
    return [
        recording
        for recording in recordings
        if (transmission := by_path.get(recording.video_path.resolve())) is not None
        and project.calibration is not None
        and decoder_recording_is_current(recording, transmission, project.calibration)
    ]


def comparison_segment(
    recording: ProcessedRecording, start_seconds: float, duration_seconds: float
) -> tuple[np.ndarray, np.ndarray]:
    """Extract an explicitly time-selected segment, without relying on pulse detection."""
    if not isfinite(start_seconds) or start_seconds < 0:
        raise ValueError("El inicio del segmento debe ser un tiempo válido igual o mayor que cero.")
    if not isfinite(duration_seconds) or duration_seconds <= 0:
        raise ValueError("La duración del segmento debe ser mayor que cero.")
    times = recording.signal.times
    start_index = int(np.searchsorted(times, start_seconds, side="left"))
    if start_index >= len(times):
        raise ValueError("El inicio del segmento queda fuera de la grabación.")
    stop_index = int(np.searchsorted(times, start_seconds + duration_seconds, side="left"))
    stop_index = max(start_index + 1, min(stop_index, len(times)))
    selected_times = times[start_index:stop_index]
    return selected_times - selected_times[0], recording.signal.intensity[start_index:stop_index]


def required_result_files(project: Project) -> dict[str, str]:
    """Require only outputs supported by the project's valid acquisitions."""
    required = dict(CORE_RESULT_FILES)
    by_distance = {round(item.distance_m, 2): item for item in project.transmissions}
    near = by_distance.get(0.5)
    far = by_distance.get(3.0)
    if near is not None and near.acquisition_valid:
        required.update(SPECTRUM_RESULT_FILES)
    if near is not None and far is not None and near.acquisition_valid and far.acquisition_valid:
        required.update(COMPARISON_RESULT_FILES)
    return required


def reset_project_session(state) -> None:
    """Drop recordings and downloads belonging to the previously open project."""
    state["siscom_recordings"] = []
    state["siscom_demo_recording"] = None
    state["siscom_export_path"] = None
    state.pop("siscom_export_project_root", None)
    state.pop("comparison_start_05", None)
    state.pop("comparison_start_30", None)
    for key in list(state.keys()):
        if str(key).startswith("report_"):
            state.pop(key, None)


def load_comparison_starts(results_dir: Path) -> tuple[float, float]:
    """Restore the reproducible manual alignment stored with saved figures."""
    manifest_path = Path(results_dir) / "analysis_manifest.json"
    try:
        starts = json.loads(manifest_path.read_text(encoding="utf-8")).get("comparison_starts")
        if not isinstance(starts, list) or len(starts) != 2:
            return (0.0, 0.0)
        values = tuple(float(value) for value in starts)
        if any(not isfinite(value) or value < 0 for value in values):
            return (0.0, 0.0)
        return values
    except (OSError, ValueError, TypeError, json.JSONDecodeError):
        return (0.0, 0.0)


def build_results_table(
    transmissions: list[Transmission],
    recordings: list[ProcessedRecording],
    expected_text: str,
) -> pd.DataFrame:
    """Create the distance table; only evaluated, valid acquisitions receive CER."""
    by_path = {record.video_path.resolve(): record for record in recordings}
    rows: list[dict] = []
    for transmission in sorted(transmissions, key=lambda item: item.distance_m):
        recording = by_path.get(transmission.video_path.resolve())
        received = recording.decode.text if recording else ""
        if not transmission.acquisition_valid:
            evaluation = evaluate_transmission(
                expected_text,
                received,
                acquisition_valid=False,
                invalid_reason=transmission.invalid_reason,
            )
        elif recording is not None:
            evaluation = evaluate_transmission(
                expected_text,
                received,
                acquisition_valid=True,
                invalid_reason=None,
            )
        else:
            evaluation = None

        counts = evaluation.errors if evaluation else None
        rows.append(
            {
                "Distancia (m)": transmission.distance_m,
                "Duración (s)": recording.message_duration_seconds if recording else None,
                "S": counts.substitutions if counts else None,
                "D": counts.deletions if counts else None,
                "I": counts.insertions if counts else None,
                "S + D + I": counts.total if counts else None,
                "CER": evaluation.cer if evaluation else None,
                "Texto recibido": received,
                "Adquisición": evaluation.acquisition_status if evaluation else "pendiente",
                "Motivo": evaluation.invalid_reason if evaluation else None,
            }
        )
    return pd.DataFrame(rows)


def export_readiness(
    project: Project,
    recordings: list[ProcessedRecording],
    results_dir: Path,
    source_root: Path,
    comparison_starts: tuple[float, float] = (0.0, 0.0),
) -> list[str]:
    """Return human-readable missing items required for a complete results bundle."""
    missing: list[str] = []
    try:
        validate_distance_set(project.transmissions)
    except ValueError as error:
        missing.append(str(error))
    if project.calibration is None:
        missing.append("Completa la calibración a 0,5 m: umbral, T y tolerancia.")
    if not Path(project.calibration_video).is_file():
        missing.append("Carga el video breve de calibración.")
    for transmission in project.transmissions:
        if not Path(transmission.video_path).is_file():
            missing.append(f"Falta el video SIS COM de {transmission.distance_m:.1f} m.")
    valid_paths = {
        transmission.video_path.resolve()
        for transmission in project.transmissions
        if transmission.acquisition_valid
    }
    processed_paths = {
        recording.video_path.resolve()
        for recording in current_recordings(project, recordings)
    } if project.calibration is not None else set()
    if not valid_paths.issubset(processed_paths):
        missing.append("Procesa las grabaciones válidas de las cuatro distancias.")
    if project.montage_photo is None or not Path(project.montage_photo).is_file():
        missing.append("Agrega una foto del montaje para el informe.")

    setup_values = (
        project.setup.source,
        project.setup.control_method,
        project.setup.materials,
        project.setup.lighting,
        project.setup.brightness,
    )
    if any(not value.strip() for value in setup_values):
        missing.append("Completa fuente, control, materiales, iluminación y brillo del montaje.")
    if any(not member.name.strip() for member in project.team):
        missing.append("Completa los nombres de ambos integrantes.")
    if not project.references:
        missing.append("Agrega las referencias utilizadas.")
    if not project.ai_assistance.strip():
        missing.append("Declara el apoyo de IA utilizado o indica que no se usó.")

    results_dir = Path(results_dir)
    for relative_path, description in required_result_files(project).items():
        if not (results_dir / relative_path).is_file():
            missing.append(f"Falta {description} ({relative_path}); guarda el análisis completo.")
    manifest_path = results_dir / "analysis_manifest.json"
    if not manifest_path.is_file():
        missing.append("Guarda de nuevo el análisis para verificar que coincide con la configuración actual.")
    else:
        try:
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            current_signature = analysis_output_signature(project, comparison_starts)
        except (OSError, ValueError):
            missing.append("No se pudo verificar la vigencia del análisis; vuelve a guardarlo.")
        else:
            if manifest.get("signature") != current_signature:
                missing.append("El análisis guardado no corresponde a la configuración actual; reprocesa y guarda los resultados de nuevo.")
    source_root = Path(source_root)
    for name in (
        "app.py",
        "requirements.txt",
        "requirements-build.txt",
        "README.md",
        "run_app.bat",
        "build_exe.bat",
        "arduino_morse_leds.ino",
        "siscom/__init__.py",
        "siscom/analysis.py",
        "siscom/export.py",
        "siscom/models.py",
        "siscom/morse.py",
        "siscom/project.py",
        "siscom/quick_test.py",
        "siscom/paths.py",
        "siscom/reporting.py",
        "siscom/tk_app.py",
        "siscom/ui.py",
        "siscom/ui_layout.py",
        "siscom/ui_helpers.py",
        "siscom/video.py",
    ):
        if not (source_root / name).is_file():
            missing.append(f"Falta {name} en la carpeta de la aplicación.")

    if project.montage_video is not None:
        try:
            metadata = probe_video(Path(project.montage_video))
            validate_montage_duration(metadata.duration_seconds)
        except (VideoReadError, ValueError) as error:
            missing.append(f"Revisa el video de demostración: {error}")
    return missing

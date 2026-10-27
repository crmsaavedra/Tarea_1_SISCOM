"""Matplotlib views and reproducible analysis outputs for the Tkinter app."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from .analysis import SpectralAnalysis
from .models import Project
from .project import analysis_output_signature
from .ui_helpers import ProcessedRecording, comparison_segment


def signal_figure(times, values, *, threshold=None, title="Señal temporal"):
    fig, axis = plt.subplots(figsize=(8.2, 3.0), constrained_layout=True)
    axis.plot(times, values, color="#3157d5", linewidth=1.1)
    if threshold is not None:
        axis.axhline(threshold, color="#e98b2a", linestyle="--", linewidth=1.3, label=f"Umbral {threshold:.1f}")
        axis.legend(frameon=False, loc="upper right")
    axis.set_title(title, loc="left", fontsize=11, fontweight="bold", color="#16213b")
    axis.set_xlabel("Tiempo (s)")
    axis.set_ylabel("Intensidad media (niveles digitales)")
    axis.grid(True, alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    return fig


def fft_figure(spectrum: SpectralAnalysis):
    fig, axis = plt.subplots(figsize=(8.2, 2.8), constrained_layout=True)
    axis.plot(spectrum.fft_frequencies, spectrum.fft_amplitude, color="#3157d5", linewidth=1.1)
    axis.set_title("Espectro de amplitud · FFT", loc="left", fontsize=11, fontweight="bold", color="#16213b")
    axis.set_xlabel("Frecuencia (Hz)")
    axis.set_ylabel("Amplitud (niveles digitales)")
    axis.grid(True, alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    return fig


def psd_figure(spectrum: SpectralAnalysis, fs: float):
    fig, axis = plt.subplots(figsize=(8.2, 2.8), constrained_layout=True)
    axis.plot(spectrum.psd_frequencies, spectrum.psd, color="#0f9b8e", linewidth=1.2)
    axis.set_xlim(0, fs / 2)
    if spectrum.b95_hz is not None:
        axis.axvline(spectrum.b95_hz, color="#e98b2a", linestyle="--", linewidth=1.3, label=f"B₉₅ = {spectrum.b95_hz:.2f} Hz")
        axis.legend(frameon=False, loc="upper right")
    axis.set_title("Densidad espectral de potencia · Welch", loc="left", fontsize=11, fontweight="bold", color="#16213b")
    axis.set_xlabel("Frecuencia (Hz)")
    axis.set_ylabel("PSD (niveles digitales²/Hz)")
    axis.grid(True, alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    return fig


def cer_figure(table: pd.DataFrame):
    valid = table.dropna(subset=["CER"])
    fig, axis = plt.subplots(figsize=(7.2, 2.9), constrained_layout=True)
    axis.plot(valid["Distancia (m)"], valid["CER"], color="#3157d5", marker="o", linewidth=1.8)
    axis.set_title("Tasa de error de caracteres por distancia", loc="left", fontsize=11, fontweight="bold", color="#16213b")
    axis.set_xlabel("Distancia (m)")
    axis.set_ylabel("CER")
    axis.grid(True, alpha=0.2)
    axis.spines[["top", "right"]].set_visible(False)
    return fig


def comparison_figure(near, far, threshold, near_start, far_start, duration):
    near_times, near_values = comparison_segment(near, near_start, duration)
    far_times, far_values = comparison_segment(far, far_start, duration)
    fig, axis = plt.subplots(figsize=(8.2, 3.0), constrained_layout=True)
    near_label = "0,5 m" if any(run.is_light for run in near.decode.runs) else "0,5 m · sin pulsos detectados"
    far_label = "3,0 m" if any(run.is_light for run in far.decode.runs) else "3,0 m · sin pulsos detectados"
    axis.plot(near_times, near_values, color="#3157d5", linewidth=1.0, label=near_label)
    axis.plot(far_times, far_values, color="#0f9b8e", linewidth=1.0, alpha=0.85, label=far_label)
    axis.axhline(threshold, color="#e98b2a", linestyle="--", linewidth=1.3, label=f"Umbral {threshold:.1f}")
    axis.set_title("Segmentos de captura seleccionados", loc="left", fontsize=11, fontweight="bold", color="#16213b")
    axis.set_xlabel("Tiempo relativo al comienzo del segmento (s)")
    axis.set_ylabel("Intensidad media (niveles digitales)")
    axis.grid(True, alpha=0.2)
    axis.legend(frameon=False, loc="upper right")
    axis.spines[["top", "right"]].set_visible(False)
    return fig


def _save_figure(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def write_analysis_outputs(
    project: Project,
    recordings: list[ProcessedRecording],
    table: pd.DataFrame,
    spectrum: SpectralAnalysis | None,
    report_notes: str,
    comparison_starts: tuple[float, float],
) -> Path:
    """Write the CSV, text, figures, and configuration manifest used by export."""
    results_dir = project.root / "results"
    plots_dir = results_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    table.to_csv(results_dir / "tabla_resultados.csv", index=False, encoding="utf-8-sig")
    by_path = {record.video_path.resolve(): record for record in recordings}
    decisions = []
    received_lines = []
    for transmission in sorted(project.transmissions, key=lambda item: item.distance_m):
        recording = by_path.get(transmission.video_path.resolve())
        label = f"{transmission.distance_m:.1f} m"
        received_lines.append(f"{label}: {recording.decode.text if recording else 'sin adquisición válida'}")
        if recording is None:
            continue
        pd.DataFrame({"tiempo_s": recording.signal.times, "intensidad_digital": recording.signal.intensity}).to_csv(
            results_dir / f"intensidad_{transmission.distance_m:.1f}m.csv", index=False, encoding="utf-8-sig"
        )
        decisions.extend({"distancia_m": transmission.distance_m, "decision": decision} for decision in recording.decode.decisions)
    (results_dir / "textos_recibidos.txt").write_text("\n".join(received_lines) + "\n", encoding="utf-8")
    pd.DataFrame(decisions, columns=["distancia_m", "decision"]).to_csv(results_dir / "decisiones_morse.csv", index=False, encoding="utf-8-sig")
    invalid = [item for item in project.transmissions if not item.acquisition_valid]
    invalid_notes = ""
    if invalid:
        invalid_notes = "\n## Adquisiciones inválidas\n" + "\n".join(
            f"- {item.distance_m:.1f} m: {item.invalid_reason}" for item in invalid
        ) + "\nLos cálculos que dependen de esas grabaciones se omitieron.\n"
    comparison_notes = (
        "\n## Segmento de comparación\n"
        f"Inicio elegido: {comparison_starts[0]:.3f} s en 0,5 m y "
        f"{comparison_starts[1]:.3f} s en 3,0 m; duración común: "
        "la menor duración disponible desde ambos inicios.\n"
    )
    (results_dir / "notas_para_informe.md").write_text(
        report_notes.strip() + comparison_notes + invalid_notes + "\n", encoding="utf-8"
    )
    by_distance = {round(item.distance_m, 2): item for item in project.transmissions}
    near = by_path.get(by_distance[0.5].video_path.resolve()) if 0.5 in by_distance else None
    far = by_path.get(by_distance[3.0].video_path.resolve()) if 3.0 in by_distance else None
    if near is not None:
        _save_figure(signal_figure(near.signal.times, near.signal.intensity, threshold=project.calibration.threshold, title="Intensidad recibida · 0,5 m"), plots_dir / "senal_temporal_0.5m.png")
        if spectrum is not None:
            _save_figure(fft_figure(spectrum), plots_dir / "fft_0.5m.png")
            _save_figure(psd_figure(spectrum, near.signal.fps), plots_dir / "psd_b95_0.5m.png")
            pd.DataFrame({"frecuencia_hz": spectrum.fft_frequencies, "amplitud": spectrum.fft_amplitude}).to_csv(results_dir / "fft_0.5m.csv", index=False, encoding="utf-8-sig")
            pd.DataFrame({"frecuencia_hz": spectrum.psd_frequencies, "psd": spectrum.psd}).to_csv(results_dir / "psd_0.5m.csv", index=False, encoding="utf-8-sig")
    _save_figure(cer_figure(table), plots_dir / "cer_distancia.png")
    if near is not None and far is not None:
        near_duration = len(near.signal.times) / near.signal.fps
        far_duration = len(far.signal.times) / far.signal.fps
        common = min(near_duration - comparison_starts[0], far_duration - comparison_starts[1])
        if common > 0:
            _save_figure(comparison_figure(near, far, project.calibration.threshold, *comparison_starts, common), plots_dir / "comparacion_0.5m_3.0m.png")
    (results_dir / "analysis_manifest.json").write_text(
        json.dumps({"signature": analysis_output_signature(project, comparison_starts), "comparison_starts": list(comparison_starts)}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return results_dir

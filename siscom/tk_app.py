"""Native Tkinter desktop interface for the SisCom Morse assignment."""

from __future__ import annotations

import base64
import os
import queue
import re
import threading
import traceback
from dataclasses import replace
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
import tkinter as tk

import cv2
import matplotlib

matplotlib.use("TkAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np
import pandas as pd

from .analysis import analyze_spectrum
from .export import create_results_zip
from .models import Calibration, Project, ROI, SetupMetadata, TeamMember, Transmission
from .paths import app_data_root, bundled_source_root
from .project import import_video, load_project, save_project, validate_montage_duration
from .quick_test import QuickVideoAnalysis, analyze_quick_video
from .reporting import (
    cer_figure,
    comparison_figure,
    fft_figure,
    psd_figure,
    signal_figure,
    write_analysis_outputs,
)
from .ui_helpers import (
    ProcessedRecording,
    apply_calibration_roi_size,
    build_results_table,
    current_recordings,
    decode_video_file,
    export_readiness,
    find_saved_projects,
    replace_transmission_by_path,
    roi_with_origin,
    validate_distance_set,
)
from .ui_layout import (
    WORKFLOW_PAGES,
    fit_portrait_preview,
    preview_bounds_to_roi,
    roi_to_preview_bounds,
)
from .video import VideoReadError, extract_video_signal, probe_video, roi_to_pixels


APP_ROOT = bundled_source_root()
PROJECTS_ROOT = app_data_root() / "projects"
DEFAULT_ROI = ROI(0.35, 0.35, 0.3, 0.3)
VIDEO_TYPES = (".mp4", ".mov", ".avi", ".mkv", ".m4v")
PHOTO_TYPES = (".png", ".jpg", ".jpeg")
QUICK_TEST_MISSING_EVIDENCE = (
    "Video breve de calibración con puntos, rayas y pausas.",
    "Cuatro grabaciones completas SIS COM, una por distancia: 0,5; 1,0; 2,0 y 3,0 m.",
    "Foto del montaje y datos de fuente, control, materiales, iluminación, equipo y aportes.",
    "Comparación de señales y CER entre distancias, incluida la comparación 0,5 m–3,0 m.",
    "Informe final en PDF con tablas, gráficos, análisis, referencias y conclusiones.",
    "Demostración en clase, si la pauta la exige.",
)
REPORT_CHECKLIST = (
    "Diagrama de bloques con mensaje, codificador, fuente, canal, cámara y decodificador",
    "Fotografía del montaje y descripción del control de la fuente y los materiales",
    "Frase Morse prevista, T, distancia, iluminación, fps y método manual o automático",
    "Transmisión óptica encendida/apagada (OOK) y su relación con el patrón Morse",
    "Señal temporal de intensidad en niveles digitales, ROI y frecuencia de muestreo",
    "Calibración a 0,5 m: umbral, tolerancia y muestra de puntos, rayas y pausas",
    "Texto reconstruido, decisiones de decodificación y CER con espacios conservados",
    "Tabla y gráficos de distancia frente a CER, FFT, PSD y B95",
    "Comparación de 0,5 m y 3,0 m, dificultad principal y mejora propuesta",
    "Referencias y declaración de apoyo de IA revisadas por el equipo",
    "Informe PDF, nombres de ambos integrantes y aportes individuales en portada",
)


def _preview_frame(video_path: Path, frame_index: int, roi: ROI) -> tuple[np.ndarray, tuple[int, int, int, int], int, int]:
    capture = cv2.VideoCapture(str(video_path))
    try:
        capture.set(cv2.CAP_PROP_POS_FRAMES, int(frame_index))
        ok, frame = capture.read()
    finally:
        capture.release()
    if not ok or frame is None:
        raise VideoReadError("No se pudo mostrar este cuadro. Revisa el archivo y su códec.")
    height, width = frame.shape[:2]
    pixel_roi = roi_to_pixels(roi, width, height)
    x, y, roi_width, roi_height = pixel_roi
    cv2.rectangle(frame, (x, y), (x + roi_width - 1, y + roi_height - 1), (35, 218, 190), 2)
    return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB), pixel_roi, width, height


def _member_value(value: str, label: str) -> int:
    try:
        parsed = int(value.strip())
    except ValueError as error:
        raise ValueError(f"El aporte de {label} debe ser un porcentaje entero.") from error
    if parsed < 0 or parsed > 100:
        raise ValueError(f"El aporte de {label} debe estar entre 0 y 100 %.")
    return parsed


def _create_project_from_form(data: dict) -> Project:
    name = re.sub(r"[^A-Za-z0-9_-]+", "_", data["name"].strip()).strip("_") or "siscom_morse"
    root = PROJECTS_ROOT / name
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Ya existe un proyecto guardado en {root}.")
    required_videos = [Path(data["calibration_video"])] + [Path(data[f"video_{distance:.1f}"]) for distance in (0.5, 1.0, 2.0, 3.0)]
    for video in required_videos:
        probe_video(video)
    if data.get("montage"):
        validate_montage_duration(probe_video(Path(data["montage"])).duration_seconds)
    team = (
        TeamMember(data["member_1"].strip(), _member_value(data["contribution_1"], "integrante 1")),
        TeamMember(data["member_2"].strip(), _member_value(data["contribution_2"], "integrante 2")),
    )
    if sum(member.contribution_percent for member in team) != 100:
        raise ValueError("Los aportes individuales deben sumar 100 %.")
    setup = SetupMetadata(
        emission_mode=data["emission_mode"],
        source=data["source"].strip(),
        control_method=data["control_method"].strip(),
        materials=data["materials"].strip(),
        lighting=data["lighting"].strip(),
        brightness=data["brightness"].strip(),
    )
    root.mkdir(parents=True, exist_ok=True)
    calibration_video = import_video(root, Path(data["calibration_video"]), "calibration")
    transmissions = []
    for distance in (0.5, 1.0, 2.0, 3.0):
        video = import_video(root, Path(data[f"video_{distance:.1f}"]), "transmission")
        transmissions.append(Transmission(distance, video, DEFAULT_ROI, True, None))
    photo = import_video(root, Path(data["photo"]), "photo") if data.get("photo") else None
    montage = import_video(root, Path(data["montage"]), "demo") if data.get("montage") else None
    project = Project(
        root=root,
        calibration_video=calibration_video,
        transmissions=transmissions,
        calibration=None,
        expected_text=data["expected_text"].strip().upper(),
        team=team,
        setup=setup,
        references=tuple(line.strip() for line in data["references"].splitlines() if line.strip()),
        ai_assistance=data["ai_assistance"].strip(),
        montage_video=montage,
        montage_photo=photo,
    )
    save_project(project, root)
    return project


class NewProjectDialog:
    """Collect the required project metadata and original media paths."""

    def __init__(self, parent: tk.Misc, on_created):
        self.parent = parent
        self.on_created = on_created
        self.window = tk.Toplevel(parent)
        self.window.title("Crear proyecto SisCom")
        self.window.geometry("800x850")
        self.window.minsize(720, 680)
        self.window.transient(parent)
        self.window.grab_set()
        self.window.configure(bg="#f4f6fb")
        self.fields: dict[str, tk.StringVar] = {}
        self.paths: dict[str, tk.StringVar] = {}
        self._build()

    def _build(self):
        outer = ttk.Frame(self.window, padding=18)
        outer.pack(fill="both", expand=True)
        ttk.Label(outer, text="Nuevo proyecto", style="Title.TLabel").pack(anchor="w")
        ttk.Label(outer, text="Carga las evidencias base y completa los datos de la tarea.", style="Muted.TLabel").pack(anchor="w", pady=(2, 12))
        canvas = tk.Canvas(outer, bg="#f4f6fb", highlightthickness=0)
        scrollbar = ttk.Scrollbar(outer, orient="vertical", command=canvas.yview)
        inner = ttk.Frame(canvas)
        inner.bind("<Configure>", lambda _event: canvas.configure(scrollregion=canvas.bbox("all")))
        inner_window = canvas.create_window((0, 0), window=inner, anchor="nw", width=730)
        canvas.configure(yscrollcommand=scrollbar.set)
        canvas.pack(side="left", fill="both", expand=True)
        scrollbar.pack(side="right", fill="y")
        canvas.bind("<MouseWheel>", lambda event: canvas.yview_scroll(int(-event.delta / 120), "units"))
        canvas.bind("<Configure>", lambda event: canvas.itemconfigure(inner_window, width=event.width))

        self._entry(inner, "Nombre del proyecto", "name", "SisCom_Morse")
        self._file_row(inner, "Video breve de calibración", "calibration_video", VIDEO_TYPES, required=True)
        for distance in (0.5, 1.0, 2.0, 3.0):
            self._file_row(inner, f"Video SIS COM · {distance:.1f} m", f"video_{distance:.1f}", VIDEO_TYPES, required=True)
        self._file_row(inner, "Foto del montaje", "photo", PHOTO_TYPES, required=False)
        self._file_row(inner, "Video demostrativo (máx. 120 s)", "montage", VIDEO_TYPES, required=False)
        ttk.Separator(inner).pack(fill="x", pady=12)
        self._entry(inner, "Texto esperado en las cuatro transmisiones", "expected_text", "SIS COM")
        self._entry(inner, "Integrante 1", "member_1", "")
        self._entry(inner, "Aporte integrante 1 (%)", "contribution_1", "50")
        self._entry(inner, "Integrante 2", "member_2", "")
        self._entry(inner, "Aporte integrante 2 (%)", "contribution_2", "50")
        self._combo(inner, "Modo de emisión", "emission_mode", ("manual", "automática"), "manual")
        for label, key in (("Fuente de luz", "source"), ("Método de control", "control_method"), ("Materiales y montaje", "materials"), ("Condiciones de iluminación", "lighting"), ("Brillo / configuración de la fuente", "brightness")):
            self._entry(inner, label, key, "")
        self._multiline(inner, "Referencias consultadas (una por línea)", "references", "ITU-R M.1677-1 — https://www.itu.int/rec/R-REC-M.1677-1-200910-I/es")
        self._multiline(inner, "Declaración de uso de IA (o indicar que no se usó)", "ai_assistance", "")
        footer = ttk.Frame(outer)
        footer.pack(fill="x", pady=(10, 0))
        ttk.Label(footer, text=f"Se guardará en: {PROJECTS_ROOT}", style="Muted.TLabel").pack(side="left")
        ttk.Button(footer, text="Crear proyecto", style="Primary.TButton", command=self._submit).pack(side="right")
        ttk.Button(footer, text="Cancelar", command=self.window.destroy).pack(side="right", padx=(0, 8))

    def _entry(self, parent, label, key, default):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=4)
        ttk.Label(row, text=label, width=34).pack(side="left", anchor="nw")
        variable = tk.StringVar(value=default)
        self.fields[key] = variable
        ttk.Entry(row, textvariable=variable).pack(side="left", fill="x", expand=True)

    def _combo(self, parent, label, key, values, default):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=4)
        ttk.Label(row, text=label, width=34).pack(side="left")
        variable = tk.StringVar(value=default)
        self.fields[key] = variable
        ttk.Combobox(row, textvariable=variable, values=values, state="readonly").pack(side="left", fill="x", expand=True)

    def _multiline(self, parent, label, key, default):
        ttk.Label(parent, text=label).pack(anchor="w", pady=(7, 2))
        text = tk.Text(parent, height=3, wrap="word", font=("Segoe UI", 9), relief="solid", bd=1)
        text.insert("1.0", default)
        text.pack(fill="x", expand=True)
        self.fields[key] = text

    def _file_row(self, parent, label, key, extensions, required):
        row = ttk.Frame(parent)
        row.pack(fill="x", pady=3)
        ttk.Label(row, text=label + (" *" if required else ""), width=34).pack(side="left")
        variable = tk.StringVar()
        self.paths[key] = variable
        ttk.Entry(row, textvariable=variable).pack(side="left", fill="x", expand=True)
        kind = [("Archivos compatibles", " ".join("*" + extension for extension in extensions)), ("Todos los archivos", "*.*")]
        command = lambda: variable.set(filedialog.askopenfilename(parent=self.window, filetypes=kind))
        ttk.Button(row, text="Elegir…", command=command).pack(side="left", padx=(6, 0))

    def _submit(self):
        data = {key: (value.get("1.0", "end").strip() if isinstance(value, tk.Text) else value.get().strip()) for key, value in self.fields.items()}
        data.update({key: value.get().strip() for key, value in self.paths.items()})
        required = ("calibration_video", "video_0.5", "video_1.0", "video_2.0", "video_3.0")
        if not data["name"] or not data["member_1"] or not data["member_2"]:
            messagebox.showerror("Faltan datos", "Escribe el nombre del proyecto y los dos integrantes.", parent=self.window)
            return
        if any(not data[key] for key in required):
            messagebox.showerror("Faltan videos", "Selecciona el video de calibración y las cuatro grabaciones por distancia.", parent=self.window)
            return
        for key in ("source", "control_method", "materials", "lighting", "brightness", "ai_assistance", "expected_text"):
            if not data[key]:
                messagebox.showerror("Faltan datos", "Completa fuente, control, materiales, iluminación, brillo, texto esperado y declaración de IA.", parent=self.window)
                return
        self.window.grab_release()
        self.window.destroy()
        self.on_created(data)


class DesktopApp:
    """Three-panel Tkinter workspace with all six assignment stages."""

    def __init__(self, root: tk.Tk):
        self.root = root
        self.project: Project | None = None
        self.recordings: list[ProcessedRecording] = []
        self.quick_analysis: QuickVideoAnalysis | None = None
        self.quick_video: Path | None = None
        self.current_page = WORKFLOW_PAGES[0]
        self.workflow_buttons: dict[str, ttk.Button] = {}
        self.worker_queue: queue.Queue = queue.Queue()
        self.worker_busy = False
        self._image = None
        self._preview_path: Path | None = None
        self._preview_roi = DEFAULT_ROI
        self._preview_frame_count = 1
        self._preview_width = 0
        self._preview_height = 0
        self._roi_editable = False
        self._roi_drag_start: tuple[int, int] | None = None
        self._roi_drag_rect = None
        self._page_vars: dict[str, tk.Variable] = {}
        self._build_window()
        self.show_page("Proyecto")
        self.root.after(100, self._poll_worker)

    def _build_window(self):
        self.root.title("Laboratorio óptico Morse · Sistemas de Comunicaciones")
        self.root.geometry("1500x930")
        self.root.minsize(1200, 760)
        self.root.configure(bg="#f4f6fb")
        style = ttk.Style(self.root)
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        style.configure("TFrame", background="#f4f6fb")
        style.configure("Panel.TFrame", background="#ffffff", relief="solid", borderwidth=1)
        style.configure("TLabel", background="#f4f6fb", foreground="#16213b", font=("Segoe UI", 9))
        style.configure("Muted.TLabel", foreground="#68758c", font=("Segoe UI", 9))
        style.configure("Title.TLabel", font=("Segoe UI", 17, "bold"), foreground="#16213b")
        style.configure("Heading.TLabel", font=("Segoe UI", 11, "bold"), foreground="#16213b")
        style.configure("PanelTitle.TLabel", background="#ffffff", foreground="#596780", font=("Segoe UI", 8, "bold"))
        style.configure("TButton", font=("Segoe UI", 9), padding=(10, 6))
        style.configure("Primary.TButton", background="#3157d5", foreground="white", font=("Segoe UI", 9, "bold"))
        style.map("Primary.TButton", background=[("active", "#2547bd"), ("disabled", "#9aa9df")])
        style.configure("Nav.TButton", background="#ffffff", padding=(13, 7), font=("Segoe UI", 9, "bold"))
        style.configure("ActiveNav.TButton", background="#3157d5", foreground="white", padding=(13, 7), font=("Segoe UI", 9, "bold"))
        style.configure("TNotebook", background="#ffffff", borderwidth=0)
        style.configure("TNotebook.Tab", padding=(12, 8), font=("Segoe UI", 9, "bold"))
        style.configure("Treeview", rowheight=25, font=("Segoe UI", 9))
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))

        header = ttk.Frame(self.root, padding=(18, 12, 18, 6))
        header.pack(fill="x")
        title_row = ttk.Frame(header)
        title_row.pack(fill="x")
        ttk.Label(title_row, text="◉", foreground="#3157d5", font=("Segoe UI Symbol", 22, "bold")).pack(side="left", padx=(0, 9))
        title_text = ttk.Frame(title_row)
        title_text.pack(side="left", anchor="w")
        ttk.Label(title_text, text="Laboratorio óptico Morse", style="Title.TLabel").pack(anchor="w")
        ttk.Label(title_text, text="Sistemas de Comunicaciones · Transmisión, decodificación y análisis", style="Muted.TLabel").pack(anchor="w")
        self.project_label = ttk.Label(title_row, text="Sin proyecto activo", style="Muted.TLabel")
        self.project_label.pack(side="right", padx=12)
        ttk.Separator(header).pack(fill="x", pady=(10, 8))
        nav = ttk.Frame(header)
        nav.pack(fill="x")
        for page in WORKFLOW_PAGES:
            button = ttk.Button(nav, text=page, style="Nav.TButton", command=lambda value=page: self.show_page(value))
            button.pack(side="left", padx=(0, 5))
            self.workflow_buttons[page] = button
        self.status_label = ttk.Label(nav, text="Listo", style="Muted.TLabel")
        self.status_label.pack(side="right")

        self.panes = tk.PanedWindow(self.root, orient="horizontal", sashwidth=6, sashrelief="flat", bg="#e0e5ee", bd=0, showhandle=False)
        self.panes.pack(fill="both", expand=True, padx=14, pady=(4, 14))
        self.left_panel = ttk.Frame(self.panes, style="Panel.TFrame", padding=12, width=380)
        self.center_outer = ttk.Frame(self.panes, style="Panel.TFrame", padding=0, width=395)
        self.right_panel = ttk.Frame(self.panes, style="Panel.TFrame", padding=10, width=680)
        self.panes.add(self.left_panel, minsize=340, width=380, stretch="never")
        self.panes.add(self.center_outer, minsize=340, width=400, stretch="always")
        self.panes.add(self.right_panel, minsize=470, width=680, stretch="always")
        self._build_preview_panel()
        self._build_center_scroll()
        self._build_results_panel()

    def _build_preview_panel(self):
        ttk.Label(self.left_panel, text="VIDEO Y REGIÓN DE INTERÉS", style="PanelTitle.TLabel").pack(anchor="w", pady=(0, 8))
        self.preview_caption = ttk.Label(self.left_panel, text="Elige un video para ver la vista previa.", style="Muted.TLabel", wraplength=340)
        self.preview_caption.pack(anchor="w", pady=(0, 7))
        preview_holder = ttk.Frame(self.left_panel, style="Panel.TFrame", padding=6)
        preview_holder.pack(anchor="center", fill="x")
        self.preview_canvas = tk.Canvas(preview_holder, width=360, height=640, bg="#17212b", highlightthickness=0)
        self.preview_canvas.pack(anchor="center")
        self.preview_canvas.bind("<ButtonPress-1>", self._roi_press)
        self.preview_canvas.bind("<B1-Motion>", self._roi_motion)
        self.preview_canvas.bind("<ButtonRelease-1>", self._roi_release)
        self.frame_var = tk.IntVar(value=0)
        self.frame_scale = ttk.Scale(self.left_panel, from_=0, to=0, command=self._frame_changed)
        self.frame_scale.pack(fill="x", pady=(8, 2))
        self.frame_label = ttk.Label(self.left_panel, text="Cuadro 0", style="Muted.TLabel")
        self.frame_label.pack(anchor="w")
        self.roi_label = ttk.Label(self.left_panel, text="ROI no seleccionada", style="Muted.TLabel", wraplength=340)
        self.roi_label.pack(anchor="w", pady=(8, 0))

    def _build_center_scroll(self):
        self.center_canvas = tk.Canvas(self.center_outer, bg="#ffffff", highlightthickness=0)
        self.center_scrollbar = ttk.Scrollbar(self.center_outer, orient="vertical", command=self.center_canvas.yview)
        self.center_canvas.configure(yscrollcommand=self.center_scrollbar.set)
        self.center_scrollbar.pack(side="right", fill="y")
        self.center_canvas.pack(side="left", fill="both", expand=True)
        self.center = ttk.Frame(self.center_canvas, padding=16)
        self.center_window = self.center_canvas.create_window((0, 0), window=self.center, anchor="nw")
        self.center.bind("<Configure>", lambda _event: self.center_canvas.configure(scrollregion=self.center_canvas.bbox("all")))
        self.center_canvas.bind("<Configure>", lambda event: self.center_canvas.itemconfigure(self.center_window, width=event.width))

    def _build_results_panel(self):
        ttk.Label(self.right_panel, text="RESULTADOS", style="PanelTitle.TLabel").pack(anchor="w", pady=(0, 8))
        self.results_notebook = ttk.Notebook(self.right_panel)
        self.results_notebook.pack(fill="both", expand=True)
        self.result_frames: dict[str, ttk.Frame] = {}
        for name in ("Resultado", "Señal", "FFT y PSD"):
            frame = ttk.Frame(self.results_notebook, padding=10)
            self.results_notebook.add(frame, text=name)
            self.result_frames[name] = frame
            ttk.Label(frame, text="Los resultados aparecerán al procesar el video.", style="Muted.TLabel", wraplength=560).pack(anchor="w")

    def _clear_center(self):
        for child in self.center.winfo_children():
            child.destroy()
        self._page_vars = {}
        self.center_canvas.yview_moveto(0)

    def _heading(self, step: str, title: str, description: str):
        ttk.Label(self.center, text=step.upper(), foreground="#0f9b8e", font=("Segoe UI", 8, "bold")).pack(anchor="w", pady=(1, 4))
        ttk.Label(self.center, text=title, style="Title.TLabel", wraplength=350).pack(anchor="w")
        ttk.Label(self.center, text=description, style="Muted.TLabel", wraplength=350).pack(anchor="w", pady=(4, 13))

    def _section(self, title: str):
        ttk.Label(self.center, text=title, style="Heading.TLabel").pack(anchor="w", pady=(12, 5))

    def _button(self, text: str, command, primary=False, *, fill="x", pady=(5, 4)):
        button = ttk.Button(self.center, text=text, command=command, style="Primary.TButton" if primary else "TButton")
        button.pack(fill=fill, pady=pady)
        return button

    def _field(self, label: str, key: str, default: str, width=None):
        ttk.Label(self.center, text=label).pack(anchor="w", pady=(6, 2))
        variable = tk.StringVar(value=default)
        self._page_vars[key] = variable
        entry = ttk.Entry(self.center, textvariable=variable, width=width)
        entry.pack(fill="x")
        return variable

    def _text(self, label: str, key: str, default: str = "", height=4):
        ttk.Label(self.center, text=label).pack(anchor="w", pady=(7, 3))
        widget = tk.Text(self.center, height=height, wrap="word", font=("Segoe UI", 9), relief="solid", bd=1)
        widget.insert("1.0", default)
        widget.pack(fill="x")
        self._page_vars[key] = widget
        return widget

    def _set_text(self, parent: ttk.Frame, content: str):
        for child in parent.winfo_children():
            child.destroy()
        widget = tk.Text(parent, wrap="word", font=("Segoe UI", 9), bg="white", fg="#24314d", relief="flat", padx=10, pady=10)
        widget.insert("1.0", content)
        widget.configure(state="disabled")
        widget.pack(fill="both", expand=True)

    def _set_result(self, content: str, figures=(), extra=None):
        frame = self.result_frames["Resultado"]
        for child in frame.winfo_children():
            child.destroy()
        text = tk.Text(frame, wrap="word", font=("Segoe UI", 9), bg="white", fg="#24314d", relief="flat", padx=10, pady=8, height=11)
        text.insert("1.0", content)
        text.configure(state="disabled")
        text.pack(fill="x", expand=False)
        if extra:
            extra(frame)
        for fig in figures:
            self._embed_figure(frame, fig)
        self._set_figures_tab("Señal", ())
        self._set_figures_tab("FFT y PSD", ())

    def _set_figures_tab(self, name: str, figures, note: str = ""):
        frame = self.result_frames[name]
        for child in frame.winfo_children():
            child.destroy()
        if note:
            ttk.Label(frame, text=note, style="Muted.TLabel", wraplength=570).pack(anchor="w", pady=(0, 8))
        for fig in figures:
            self._embed_figure(frame, fig)

    def _embed_figure(self, parent, fig):
        holder = ttk.Frame(parent)
        holder.pack(fill="x", expand=True, pady=6)
        canvas = FigureCanvasTkAgg(fig, master=holder)
        canvas.draw()
        canvas.get_tk_widget().pack(fill="both", expand=True)

    def _set_table(self, parent, dataframe: pd.DataFrame):
        holder = ttk.Frame(parent)
        holder.pack(fill="both", expand=True, pady=6)
        columns = list(dataframe.columns)
        tree = ttk.Treeview(holder, columns=columns, show="headings", height=min(10, max(4, len(dataframe))))
        vertical = ttk.Scrollbar(holder, orient="vertical", command=tree.yview)
        horizontal = ttk.Scrollbar(holder, orient="horizontal", command=tree.xview)
        tree.configure(yscrollcommand=vertical.set, xscrollcommand=horizontal.set)
        tree.pack(side="top", fill="both", expand=True)
        vertical.pack(side="right", fill="y")
        horizontal.pack(side="bottom", fill="x")
        for column in columns:
            tree.heading(column, text=column)
            tree.column(column, width=max(90, min(220, len(str(column)) * 12)), anchor="center", stretch=True)
        for _, row in dataframe.iterrows():
            values = ["—" if pd.isna(value) else (f"{value:.4f}" if isinstance(value, (float, np.floating)) and column == "CER" else str(value)) for column, value in row.items()]
            tree.insert("", "end", values=values)

    def _poll_worker(self):
        try:
            while True:
                kind, payload = self.worker_queue.get_nowait()
                self.worker_busy = False
                if kind == "ok":
                    result, on_success = payload
                    self.status_label.configure(text="Listo")
                    try:
                        on_success(result)
                    except Exception as error:
                        messagebox.showerror("Error", str(error), parent=self.root)
                else:
                    error, title = payload
                    self.status_label.configure(text="Error")
                    messagebox.showerror(title, str(error), parent=self.root)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_worker)

    def _run_worker(self, label: str, function, on_success, error_title="No se pudo completar"):
        if self.worker_busy:
            return
        self.worker_busy = True
        self.status_label.configure(text=label)

        def run():
            try:
                result = function()
            except Exception as error:
                self.worker_queue.put(("error", (error, error_title)))
            else:
                self.worker_queue.put(("ok", (result, on_success)))

        threading.Thread(target=run, daemon=True, name="siscom-worker").start()

    def _set_project(self, project: Project):
        previous_root = self.project.root.resolve() if self.project else None
        self.project = project
        if previous_root != project.root.resolve():
            self.recordings = []
        else:
            self.recordings = current_recordings(project, self.recordings)
        save_project(project, project.root)
        self.project_label.configure(text=f"Proyecto activo: {project.root.name}")
        if self.current_page != "Proyecto":
            self.show_page(self.current_page)

    def show_page(self, page: str):
        if page not in WORKFLOW_PAGES:
            raise ValueError(f"Página desconocida: {page}")
        self.current_page = page
        for name, button in self.workflow_buttons.items():
            button.configure(style="ActiveNav.TButton" if name == page else "Nav.TButton")
        self._clear_center()
        if page == "Proyecto":
            self._page_project()
        elif page == "Prueba rápida":
            self._page_quick_test()
        elif page == "Calibración":
            self._page_calibration()
        elif page == "Decodificación":
            self._page_decoding()
        elif page == "Análisis":
            self._page_analysis()
        else:
            self._page_export()

    def _page_project(self):
        self._heading("01 · Proyecto", "Prepara las grabaciones", "Organiza la calibración, las cuatro distancias y los datos que acompañan el informe.")
        self._button("Crear proyecto", lambda: NewProjectDialog(self.root, self._start_project_creation), primary=True)
        self._button("Abrir proyecto guardado", self._open_project)
        self._button("Guardar proyecto", self._save_current_project, fill="x")
        self._section("Qué debe tener el proyecto")
        guidance = (
            "1. Un video breve de calibración con puntos, rayas y pausas.\n"
            "2. Cuatro videos SIS COM completos, grabados a 0,5; 1,0; 2,0 y 3,0 m.\n"
            "3. Texto esperado, nombres y aportes de los dos integrantes.\n"
            "4. Fuente, control, materiales, iluminación, brillo, referencias y declaración de IA.\n"
            "5. Foto del montaje. La demostración es opcional y debe durar hasta 120 s."
        )
        ttk.Label(self.center, text=guidance, wraplength=350, justify="left").pack(anchor="w", pady=5)
        self._section("Proyecto activo")
        if self.project is None:
            ttk.Label(self.center, text="Aún no hay proyecto abierto. También puedes usar Prueba rápida para revisar un único video con HOLA.", wraplength=350, style="Muted.TLabel").pack(anchor="w")
            self._set_result("Crea o abre un proyecto para ver su estado. La prueba rápida funciona sin proyecto.")
            self._clear_preview("Selecciona un video desde el flujo de trabajo.")
        else:
            project = self.project
            summary = [f"Carpeta: {project.root}", f"Texto esperado: {project.expected_text}", f"Video calibración: {project.calibration_video.name}", f"Grabaciones cargadas: {len(project.transmissions)} / 4", f"Calibración: {'lista' if project.calibration else 'pendiente'}", f"Foto del montaje: {'lista' if project.montage_photo else 'pendiente'}"]
            ttk.Label(self.center, text="\n".join(summary), wraplength=350, style="Muted.TLabel").pack(anchor="w")
            self._set_result("Proyecto guardado localmente.\n\n" + "\n".join(summary))
            self._load_preview(project.calibration_video, project.calibration.roi if project.calibration else DEFAULT_ROI, editable=False)

    def _start_project_creation(self, data):
        self._run_worker("Importando videos y guardando proyecto…", lambda: _create_project_from_form(data), self._project_created, "No se pudo crear el proyecto")

    def _project_created(self, project):
        self._set_project(project)
        messagebox.showinfo("Proyecto creado", f"El proyecto quedó guardado en:\n{project.root}", parent=self.root)
        self.show_page("Proyecto")

    def _open_project(self):
        path = filedialog.askopenfilename(parent=self.root, title="Abrir proyecto SisCom", filetypes=[("Proyecto SisCom", "project.json"), ("JSON", "*.json")])
        if not path:
            return
        self._run_worker("Abriendo proyecto…", lambda: load_project(Path(path)), self._project_created, "No se pudo abrir el proyecto")

    def _save_current_project(self):
        if self.project is None:
            messagebox.showinfo("Sin proyecto", "Crea o abre un proyecto primero.", parent=self.root)
            return
        save_project(self.project, self.project.root)
        self.status_label.configure(text="Proyecto guardado")

    def _page_quick_test(self):
        self._heading("PRUEBA INDEPENDIENTE", "Prueba rápida", "Selecciona un clip grabado con HOLA. La aplicación estima ROI, umbral y T; luego decodifica y compara.")
        self._button("Seleccionar video…", self._choose_quick_video)
        ttk.Label(self.center, text=self.quick_video.name if self.quick_video else "No hay video seleccionado.", style="Muted.TLabel", wraplength=350).pack(anchor="w", pady=(4, 0))
        ttk.Label(self.center, text="La comparación contra HOLA ocurre después de decodificar. Este resultado no es el CER de las cuatro distancias.", wraplength=350, foreground="#8a5a12").pack(anchor="w", pady=(9, 0))
        self._button("Analizar video", self._analyze_quick, primary=True)
        self._section("Lo que todavía falta para completar la tarea")
        listbox = tk.Text(self.center, height=12, wrap="word", font=("Segoe UI", 9), bg="#f8f9fc", relief="flat", padx=8, pady=7)
        listbox.insert("1.0", "\n".join(f"☐ {item}" for item in QUICK_TEST_MISSING_EVIDENCE))
        listbox.configure(state="disabled")
        listbox.pack(fill="x")
        if self.quick_video:
            self._load_preview(self.quick_video, self.quick_analysis.calibration.roi if self.quick_analysis else DEFAULT_ROI, editable=False)
        if self.quick_analysis:
            self._display_quick_result(self.quick_analysis)
        else:
            self._set_result("Selecciona el video que contiene HOLA y pulsa «Analizar video». La aplicación procesa el archivo localmente.")
            self._set_text(self.result_frames["Señal"], "La señal temporal aparecerá al terminar el análisis.")
            self._set_text(self.result_frames["FFT y PSD"], "FFT y PSD aparecerán cuando la señal tenga muestras suficientes.")

    def _choose_quick_video(self):
        path = filedialog.askopenfilename(parent=self.root, title="Seleccionar video HOLA", filetypes=[("Videos", "*.mp4 *.mov *.avi *.mkv *.m4v"), ("Todos los archivos", "*.*")])
        if path:
            self.quick_video = Path(path)
            self.quick_analysis = None
            self.show_page("Prueba rápida")

    def _analyze_quick(self):
        if self.quick_video is None:
            self._choose_quick_video()
            if self.quick_video is None:
                return
        path = self.quick_video
        self.quick_analysis = None
        self._run_worker("Analizando video HOLA…", lambda: analyze_quick_video(path), self._quick_complete, "No se pudo analizar el video")

    def _quick_complete(self, analysis):
        self.quick_analysis = analysis
        self._display_quick_result(analysis)
        self._load_preview(self.quick_video, analysis.calibration.roi, editable=False)

    def _display_quick_result(self, analysis: QuickVideoAnalysis):
        from .project import evaluate_transmission

        evaluation = evaluate_transmission("HOLA", analysis.decode.text, acquisition_valid=True, invalid_reason=None)
        ratio = f"{analysis.dot_dash_ratio:.2f}:1" if analysis.dot_dash_ratio is not None else "sin rayas detectadas"
        details = [
            "PRUEBA DE UN SOLO VIDEO · HOLA",
            "Coincide con HOLA." if evaluation.cer == 0 else "No coincide completamente con HOLA.",
            f"\nReferencia: HOLA    Recibido: {analysis.decode.text or '(vacío)'}",
            f"Sustituciones (S): {evaluation.errors.substitutions}",
            f"Eliminaciones (D): {evaluation.errors.deletions}",
            f"Inserciones (I): {evaluation.errors.insertions}",
            f"CER contra HOLA: {evaluation.cer:.4f}",
            f"Umbral: {analysis.calibration.threshold:.1f}/255    T estimada: {analysis.calibration.point_seconds:.3f} s",
            f"ROI: {analysis.calibration.roi}    fs: {analysis.signal.fps:.2f} Hz",
            f"Contraste: {analysis.intensity_contrast:.1f} niveles    Relación raya/punto: {ratio}",
            "\nDecisiones Morse:", *(analysis.decode.decisions or ("No se generaron decisiones.",)),
        ]
        if analysis.diagnostics:
            details += ["\nObservaciones:", *(f"• {line}" for line in analysis.diagnostics)]
        self._set_result("\n".join(details))
        self._set_figures_tab("Señal", (signal_figure(analysis.signal.times, analysis.signal.intensity, threshold=analysis.calibration.threshold, title="Prueba rápida · intensidad media de la ROI"),), f"{self.quick_video.name if self.quick_video else 'Video'} · {analysis.signal.fps:.2f} Hz")
        try:
            spectrum = analyze_spectrum(analysis.signal.intensity, analysis.signal.fps)
        except ValueError as error:
            self._set_figures_tab("FFT y PSD", (), f"No se pudo calcular FFT/PSD: {error}")
        else:
            suffix = f"Welch · ventana Hann · nperseg={spectrum.nperseg} · B95={spectrum.b95_hz:.2f} Hz" if spectrum.b95_hz is not None else "Welch · ventana Hann · B95 no definido"
            self._set_figures_tab("FFT y PSD", (fft_figure(spectrum), psd_figure(spectrum, analysis.signal.fps)), suffix)

    def _page_calibration(self):
        self._heading("02 · Calibración", "Define la recepción a 0,5 m", "Ajusta la región del LED, el umbral y T. Esta calibración se reutiliza en las cuatro distancias.")
        if self.project is None:
            self._no_project("Crea o abre un proyecto con el video breve de calibración.")
            return
        project = self.project
        existing = project.calibration
        roi = existing.roi if existing else DEFAULT_ROI
        self.cal_roi = roi
        self._load_preview(project.calibration_video, roi, editable=True)
        self._field("Umbral digital (0–255)", "threshold", f"{existing.threshold:.1f}" if existing else "127.5")
        self._field("Duración del punto T (segundos)", "point_seconds", f"{existing.point_seconds:.3f}" if existing else "0.400")
        self._field("Tolerancia temporal (0–49 %)", "tolerance", f"{100 * existing.tolerance_fraction:.0f}" if existing else "20")
        ttk.Label(self.center, text="Arrastra sobre la vista previa para dibujar la ROI. La raya dura 3T; las pausas entre símbolos, letras y palabras son T, 3T y 7T.", style="Muted.TLabel", wraplength=350).pack(anchor="w", pady=(8, 0))
        self._button("Medir señal y revisar calibración", self._measure_calibration, primary=True)
        self._button("Guardar calibración", self._save_calibration)
        if self._calibration_candidate:
            self._display_calibration_candidate()
        else:
            self._set_result("Mide la señal para comprobar la separación de niveles y los intervalos antes de guardar la calibración.")

    @property
    def _calibration_candidate(self):
        return getattr(self, "calibration_candidate", None)

    def _measure_calibration(self):
        if not self.project:
            return
        try:
            threshold = float(self._page_vars["threshold"].get())
            point = float(self._page_vars["point_seconds"].get())
            tolerance = float(self._page_vars["tolerance"].get()) / 100.0
            candidate = Calibration(threshold, point, tolerance, self.cal_roi, 1.0)
        except (ValueError, AttributeError) as error:
            messagebox.showerror("Revisa la calibración", str(error), parent=self.root)
            return
        video_path, roi = self.project.calibration_video, self.cal_roi

        def measure():
            signal = extract_video_signal(video_path, roi)
            result = replace(candidate, reference_fps=signal.fps)
            recording = decode_video_file(video_path, roi, result)
            low = signal.intensity[signal.intensity < result.threshold]
            high = signal.intensity[signal.intensity >= result.threshold]
            separation = float(np.mean(high) - np.mean(low)) if low.size and high.size else None
            return result, recording, separation

        self._run_worker("Midiendo señal de calibración…", measure, self._calibration_measured, "No se pudo medir el video")

    def _calibration_measured(self, result):
        self.calibration_candidate, self.calibration_recording, self.calibration_separation = result
        self._display_calibration_candidate()

    def _display_calibration_candidate(self):
        calibration = self.calibration_candidate
        recording = self.calibration_recording
        low_values = recording.signal.intensity[recording.signal.intensity < calibration.threshold]
        high_values = recording.signal.intensity[recording.signal.intensity >= calibration.threshold]
        text = ["CALIBRACIÓN · VIDEO BREVE", f"Umbral {calibration.threshold:.1f}/255 · T={calibration.point_seconds:.3f} s · tolerancia {calibration.tolerance_fraction:.0%}", f"Frecuencia de muestreo: {recording.signal.fps:.2f} Hz", f"Apagado medio: {np.mean(low_values):.1f}" if low_values.size else "Apagado: sin muestras", f"Encendido medio: {np.mean(high_values):.1f}" if high_values.size else "Encendido: sin muestras", f"Separación: {self.calibration_separation:.1f} niveles" if self.calibration_separation is not None else "Separación: no calculable", f"Intervalos detectados: {len(recording.decode.runs)}", "\nDecisiones Morse:", *(recording.decode.decisions or ("Sin decisiones.",))]
        if self.calibration_separation is not None and self.calibration_separation < 12:
            text.append("\nLos niveles están próximos; revisa la iluminación o el encuadre.")
        self._set_result("\n".join(text))
        self._set_figures_tab("Señal", (signal_figure(recording.signal.times, recording.signal.intensity, threshold=calibration.threshold, title="Muestra de calibración · intensidad de la ROI"),))
        try:
            spectrum = analyze_spectrum(recording.signal.intensity, recording.signal.fps)
            self._set_figures_tab("FFT y PSD", (fft_figure(spectrum), psd_figure(spectrum, recording.signal.fps)))
        except ValueError as error:
            self._set_figures_tab("FFT y PSD", (), str(error))

    def _save_calibration(self):
        if not self.project:
            return
        if not self._calibration_candidate:
            messagebox.showinfo("Primero mide la señal", "Pulsa «Medir señal y revisar calibración» antes de guardar.", parent=self.root)
            return
        transmissions = apply_calibration_roi_size(self.project.transmissions, self.calibration_candidate.roi)
        self._set_project(replace(self.project, calibration=self.calibration_candidate, transmissions=transmissions))
        self.recordings = []
        self.status_label.configure(text="Calibración guardada")
        messagebox.showinfo("Calibración guardada", "Umbral, T, tolerancia y ROI quedaron guardados en el proyecto.", parent=self.root)

    def _page_decoding(self):
        self._heading("03 · Recepción", "Decodifica las grabaciones", "Usa la calibración guardada y revisa las decisiones antes de analizar el CER.")
        if self.project is None:
            self._no_project("Crea un proyecto con las cuatro grabaciones para decodificar.")
            return
        if self.project.calibration is None:
            self._no_project("Guarda la calibración a 0,5 m antes de decodificar.")
            return
        try:
            validate_distance_set(self.project.transmissions)
        except ValueError as error:
            self._no_project(str(error))
            return
        ordered = sorted(self.project.transmissions, key=lambda item: item.distance_m)
        distances = [item.distance_m for item in ordered]
        current = getattr(self, "decode_distance", distances[0])
        if current not in distances:
            current = distances[0]
        self.decode_distance = current
        self._section("Grabación seleccionada")
        selection = tk.StringVar(value=f"{current:.1f} m")
        ttk.Combobox(self.center, textvariable=selection, values=[f"{distance:.1f} m" for distance in distances], state="readonly").pack(fill="x")
        selection.trace_add("write", lambda *_: self._select_decode_distance(selection.get()))
        transmission = next(item for item in ordered if item.distance_m == current)
        self.decode_roi = transmission.roi
        self._load_preview(transmission.video_path, transmission.roi, editable=True)
        ttk.Label(self.center, text=f"Calibración fija · umbral {self.project.calibration.threshold:.1f} · T={self.project.calibration.point_seconds:.3f} s · tolerancia {self.project.calibration.tolerance_fraction:.0%}", style="Muted.TLabel", wraplength=350).pack(anchor="w", pady=7)
        self._field("ROI izquierda (0–1)", "roi_left", f"{transmission.roi.left:.4f}")
        self._field("ROI arriba (0–1)", "roi_top", f"{transmission.roi.top:.4f}")
        self._field("ROI ancho (0–1)", "roi_width", f"{transmission.roi.width:.4f}")
        self._field("ROI alto (0–1)", "roi_height", f"{transmission.roi.height:.4f}")
        self.valid_var = tk.BooleanVar(value=transmission.acquisition_valid)
        ttk.Checkbutton(self.center, text="Grabación completa y válida", variable=self.valid_var).pack(anchor="w", pady=(9, 2))
        self._field("Motivo si la adquisición está incompleta", "invalid_reason", transmission.invalid_reason or "")
        self._button("Guardar ROI y estado", self._save_transmission_status)
        self._button("Procesar las grabaciones válidas", self._decode_all, primary=True)
        ttk.Label(self.center, text="Una recepción vacía o débil sigue siendo válida. Márcala inválida solo cuando la captura esté incompleta por un fallo de adquisición.", style="Muted.TLabel", wraplength=350).pack(anchor="w", pady=7)
        self.recordings = current_recordings(self.project, self.recordings)
        self._display_decoding_result(transmission)

    def _select_decode_distance(self, value):
        try:
            self.decode_distance = float(value.split()[0])
        except (ValueError, IndexError):
            return
        self.show_page("Decodificación")

    def _save_transmission_status(self):
        if not self.project:
            return
        item = next((entry for entry in self.project.transmissions if entry.distance_m == self.decode_distance), None)
        if item is None:
            return
        try:
            roi = ROI(*(float(self._page_vars[key].get()) for key in ("roi_left", "roi_top", "roi_width", "roi_height")))
            valid = bool(self.valid_var.get())
            reason = self._page_vars["invalid_reason"].get().strip()
            updated = replace(item, roi=roi, acquisition_valid=valid, invalid_reason=None if valid else reason)
            transmissions = replace_transmission_by_path(self.project.transmissions, updated)
            self._set_project(replace(self.project, transmissions=transmissions))
        except (ValueError, TypeError) as error:
            messagebox.showerror("Revisa los datos", str(error), parent=self.root)

    def _decode_all(self):
        if not self.project or not self.project.calibration:
            return
        project = self.project
        valid = sorted((item for item in project.transmissions if item.acquisition_valid), key=lambda item: item.distance_m)
        if not valid:
            messagebox.showinfo("Sin capturas válidas", "Marca como válida al menos una grabación completa.", parent=self.root)
            return

        def decode_all():
            return [decode_video_file(item.video_path, item.roi, project.calibration) for item in valid]

        self._run_worker("Decodificando grabaciones…", decode_all, self._decoding_complete, "No se completó la decodificación")

    def _decoding_complete(self, records):
        self.recordings = records
        self._set_result(f"Procesamiento finalizado: {len(records)} grabaciones válidas.\nLa salida se obtuvo solo de la señal; el texto esperado no se usó para corregir la decodificación.")
        self.show_page("Decodificación")

    def _display_decoding_result(self, transmission):
        record = next((item for item in self.recordings if item.video_path.resolve() == transmission.video_path.resolve()), None)
        if record is None:
            message = f"{transmission.distance_m:.1f} m · pendiente de procesar" if transmission.acquisition_valid else f"{transmission.distance_m:.1f} m · adquisición inválida\nMotivo: {transmission.invalid_reason}"
            self._set_result(message)
            self._set_text(self.result_frames["Señal"], "La señal aparecerá después de procesar esta distancia.")
            self._set_text(self.result_frames["FFT y PSD"], "El espectro aparecerá después de procesar esta distancia.")
            return
        calibration = self.project.calibration
        self._set_result(f"DISTANCIA {transmission.distance_m:.1f} m\n\nTexto reconstruido: {record.decode.text or '(vacío)'}\nDuración del mensaje: {record.message_duration_seconds:.2f} s\nFrecuencia de muestreo: {record.signal.fps:.2f} Hz\nIntervalos detectados: {len(record.decode.runs)}\n\nDecisiones Morse:\n" + "\n".join(record.decode.decisions))
        self._set_figures_tab("Señal", (signal_figure(record.signal.times, record.signal.intensity, threshold=calibration.threshold, title=f"Intensidad recibida · {transmission.distance_m:.1f} m"),))
        try:
            spectrum = analyze_spectrum(record.signal.intensity, record.signal.fps)
        except ValueError as error:
            self._set_figures_tab("FFT y PSD", (), str(error))
        else:
            self._set_figures_tab("FFT y PSD", (fft_figure(spectrum), psd_figure(spectrum, record.signal.fps)), f"B95={spectrum.b95_hz:.2f} Hz" if spectrum.b95_hz is not None else "B95 no definido")

    def _page_analysis(self):
        self._heading("04 · Análisis", "Mide errores y espectro", "Compara las cuatro distancias y guarda los datos y figuras que acompañan el informe.")
        if self.project is None or self.project.calibration is None:
            self._no_project("Abre un proyecto y guarda la calibración para analizar los resultados.")
            return
        try:
            validate_distance_set(self.project.transmissions)
        except ValueError as error:
            self._no_project(str(error))
            return
        project = self.project
        self.recordings = current_recordings(project, self.recordings)
        table = build_results_table(project.transmissions, self.recordings, project.expected_text)
        self._set_result(f"Texto esperado: {project.expected_text}\nGrabaciones procesadas: {len(self.recordings)} / {sum(item.acquisition_valid for item in project.transmissions)} válidas")
        self._set_table(self.result_frames["Resultado"], table)
        self._embed_figure(self.result_frames["Resultado"], cer_figure(table))
        near_t = next((item for item in project.transmissions if item.distance_m == 0.5), None)
        far_t = next((item for item in project.transmissions if item.distance_m == 3.0), None)
        by_path = {record.video_path.resolve(): record for record in self.recordings}
        near = by_path.get(near_t.video_path.resolve()) if near_t else None
        far = by_path.get(far_t.video_path.resolve()) if far_t else None
        signal_figs = [signal_figure(near.signal.times, near.signal.intensity, threshold=project.calibration.threshold, title="Intensidad recibida · 0,5 m")] if near else []
        start05, start30 = self._saved_comparison_starts()
        compare_figs = []
        if near and far:
            duration = min(len(near.signal.times) / near.signal.fps - start05, len(far.signal.times) / far.signal.fps - start30)
            if duration > 0:
                compare_figs.append(comparison_figure(near, far, project.calibration.threshold, start05, start30, duration))
        self._set_figures_tab("Señal", signal_figs + compare_figs, "Se conserva la intensidad completa, también bajo el umbral Morse.")
        try:
            spectrum = analyze_spectrum(near.signal.intensity, near.signal.fps) if near else None
        except ValueError as error:
            spectrum = None
            self._set_figures_tab("FFT y PSD", (), str(error))
        else:
            if spectrum:
                self._set_figures_tab("FFT y PSD", (fft_figure(spectrum), psd_figure(spectrum, near.signal.fps)), f"Welch · ventana Hann · nperseg={spectrum.nperseg} · B95={spectrum.b95_hz:.2f} Hz" if spectrum.b95_hz is not None else "B95 no definido")
        self._section("Alineación de la comparación")
        starts = self._saved_comparison_starts()
        self._field("Inicio del segmento · 0,5 m (s)", "comparison_05", f"{starts[0]:.3f}")
        self._field("Inicio del segmento · 3,0 m (s)", "comparison_30", f"{starts[1]:.3f}")
        self._section("Notas para el informe")
        self._text("Dificultad principal observada", "limitation")
        self._text("Mejora propuesta y por qué ayudaría", "improvement")
        self._text("Si CER = 0, ¿por qué no garantiza las siguientes transmisiones?", "cer_zero", height=3)
        self._button("Guardar tabla, señales y gráficos", lambda: self._save_analysis(table, spectrum), primary=True)

    def _saved_comparison_starts(self):
        try:
            import json
            path = self.project.root / "results" / "analysis_manifest.json"
            values = json.loads(path.read_text(encoding="utf-8")).get("comparison_starts", [0.0, 0.0])
            if len(values) == 2 and all(float(value) >= 0 for value in values):
                return float(values[0]), float(values[1])
        except (OSError, ValueError, TypeError):
            pass
        return (0.0, 0.0)

    def _save_analysis(self, table, spectrum):
        if not self.project:
            return
        try:
            starts = (float(self._page_vars["comparison_05"].get()), float(self._page_vars["comparison_30"].get()))
            if min(starts) < 0:
                raise ValueError("Los inicios deben ser cero o mayores.")
        except ValueError as error:
            messagebox.showerror("Revisa los tiempos", str(error), parent=self.root)
            return
        def text_value(key):
            return self._page_vars[key].get("1.0", "end").strip()
        notes = ("# Notas para el informe\n\n## Dificultad principal\n" + (text_value("limitation") or "[Completar]") + "\n\n## Mejora propuesta\n" + (text_value("improvement") or "[Completar]") + "\n\n## Explicación de CER = 0\n" + (text_value("cer_zero") or "[No aplica o completar]") + "\n")
        project = self.project
        recordings = list(self.recordings)
        self._run_worker("Guardando CSV y gráficos…", lambda: write_analysis_outputs(project, recordings, table, spectrum, notes, starts), self._analysis_saved, "No se pudo guardar el análisis")

    def _analysis_saved(self, results_dir):
        save_project(self.project, self.project.root)
        self.status_label.configure(text=f"Resultados guardados · {results_dir}")
        messagebox.showinfo("Análisis guardado", f"Tabla, señales y gráficos guardados en:\n{results_dir}", parent=self.root)

    def _page_export(self):
        self._heading("05 · Entrega", "Prepara el paquete", "Revisa la evidencia pendiente y genera un ZIP reproducible cuando el análisis esté vigente.")
        if self.project is None:
            self._no_project("Crea o abre un proyecto para preparar la entrega.")
            return
        project = self.project
        records = current_recordings(project, self.recordings)
        results = project.root / "results"
        missing = export_readiness(project, records, results, APP_ROOT, self._saved_comparison_starts())
        self._section("Estado de la entrega")
        if missing:
            ttk.Label(self.center, text=f"Faltan {len(missing)} elementos:", foreground="#9a5a12", font=("Segoe UI", 9, "bold")).pack(anchor="w")
            text = tk.Text(self.center, height=min(18, max(8, len(missing) + 1)), wrap="word", font=("Segoe UI", 9), bg="#fffaf0", relief="flat", padx=8, pady=6)
            text.insert("1.0", "\n".join(f"☐ {item}" for item in missing))
            text.configure(state="disabled")
            text.pack(fill="x", pady=(5, 5))
        else:
            ttk.Label(self.center, text="La revisión automática está completa. Recuerda redactar el PDF y preparar la demostración de clase.", wraplength=350, foreground="#087a6b").pack(anchor="w")
        self._section("Lista del informe")
        checklist = tk.Text(self.center, height=15, wrap="word", font=("Segoe UI", 9), bg="#f8f9fc", relief="flat", padx=8, pady=6)
        checklist.insert("1.0", "\n".join(f"☐ {item}" for item in REPORT_CHECKLIST))
        checklist.configure(state="disabled")
        checklist.pack(fill="x")
        self._button("Crear ZIP reproducible", lambda: self._create_export(missing), primary=True)
        self._button("Abrir carpeta del proyecto", lambda: os.startfile(str(project.root)))
        if (project.root / "entrega_siscom_morse.zip").is_file():
            self._button("Abrir ZIP generado", lambda: os.startfile(str(project.root / "entrega_siscom_morse.zip")))
        self._set_result(f"Proyecto: {project.root.name}\nGrabaciones decodificadas vigentes: {len(records)}\nResultados guardados: {'sí' if (results / 'tabla_resultados.csv').is_file() else 'no'}\n\nEl ZIP no crea el informe PDF ni reemplaza la demostración oral.")

    def _create_export(self, missing):
        if missing:
            messagebox.showwarning("Faltan requisitos", "Completa la lista de requisitos de esta sección antes de crear el paquete.", parent=self.root)
            return
        project = self.project
        target = project.root / "entrega_siscom_morse.zip"
        self._run_worker("Creando ZIP reproducible…", lambda: create_results_zip(project, project.root / "results", target, APP_ROOT), self._export_created, "No se pudo crear el ZIP")

    def _export_created(self, path):
        messagebox.showinfo("Paquete creado", f"ZIP reproducible guardado en:\n{path}", parent=self.root)
        self.show_page("Exportación")

    def _no_project(self, message):
        ttk.Label(self.center, text=message, wraplength=350, style="Muted.TLabel").pack(anchor="w", pady=5)
        self._button("Ir a Proyecto", lambda: self.show_page("Proyecto"), primary=True)
        self._set_result(message)
        self._clear_preview(message)

    def _clear_preview(self, message):
        self._preview_path = None
        self.preview_canvas.delete("all")
        self.preview_canvas.configure(bg="#17212b")
        self.preview_caption.configure(text=message)
        self.roi_label.configure(text="ROI no seleccionada")
        self.frame_scale.configure(from_=0, to=0)

    def _load_preview(self, path: Path | None, roi: ROI, *, editable: bool):
        self._roi_editable = editable
        if path is None or not Path(path).is_file():
            self._clear_preview("No se puede encontrar el video seleccionado.")
            return
        try:
            metadata = probe_video(path)
            self._preview_path = Path(path)
            self._preview_roi = roi
            self._preview_frame_count = metadata.frame_count
            self.frame_scale.configure(from_=0, to=max(0, metadata.frame_count - 1))
            self.frame_scale.set(0)
            self.preview_caption.configure(text=f"{Path(path).name} · {metadata.fps:.2f} fps · {metadata.duration_seconds:.2f} s · {metadata.width} × {metadata.height} px")
            self._render_preview(0)
        except (VideoReadError, OSError, ValueError) as error:
            self._clear_preview(str(error))

    def _render_preview(self, frame_index):
        if not self._preview_path:
            return
        try:
            rgb, pixels, source_width, source_height = _preview_frame(self._preview_path, int(float(frame_index)), self._preview_roi)
            preview = fit_portrait_preview(rgb, width=360, height=640)
            ok, encoded = cv2.imencode(".png", cv2.cvtColor(preview, cv2.COLOR_RGB2BGR))
            if not ok:
                raise VideoReadError("No se pudo convertir el cuadro de video para la vista previa.")
            self._image = tk.PhotoImage(data=base64.b64encode(encoded).decode("ascii"), format="png")
            self.preview_canvas.delete("all")
            self.preview_canvas.create_image(0, 0, anchor="nw", image=self._image)
            bounds = roi_to_preview_bounds(self._preview_roi, source_width, source_height, width=360, height=640)
            self.preview_canvas.create_rectangle(*bounds, outline="#23dabe", width=2, tags="roi")
            self.preview_canvas.tag_lower("roi")
            self.frame_label.configure(text=f"Cuadro {int(float(frame_index))} / {self._preview_frame_count - 1}")
            self.roi_label.configure(text=f"ROI · x={pixels[0]}, y={pixels[1]} · {pixels[2]}×{pixels[3]} px · arrastra el marco para cambiarla" if self._roi_editable else f"ROI · x={pixels[0]}, y={pixels[1]} · {pixels[2]}×{pixels[3]} px")
        except (VideoReadError, OSError, ValueError, tk.TclError) as error:
            self.preview_canvas.delete("all")
            self.preview_caption.configure(text=str(error))

    def _frame_changed(self, value):
        if self._preview_path:
            self._render_preview(value)

    def _roi_press(self, event):
        if not self._roi_editable or self._preview_path is None:
            return
        self._roi_drag_start = (event.x, event.y)
        if self._roi_drag_rect:
            self.preview_canvas.delete(self._roi_drag_rect)
        self._roi_drag_rect = self.preview_canvas.create_rectangle(event.x, event.y, event.x, event.y, outline="#ffb547", width=2)

    def _roi_motion(self, event):
        if self._roi_drag_start and self._roi_drag_rect:
            self.preview_canvas.coords(self._roi_drag_rect, *self._roi_drag_start, event.x, event.y)

    def _roi_release(self, event):
        if not self._roi_drag_start or not self._preview_path:
            return
        start = self._roi_drag_start
        self._roi_drag_start = None
        try:
            metadata = probe_video(self._preview_path)
            roi = preview_bounds_to_roi((start[0], start[1], event.x, event.y), metadata.width, metadata.height, width=360, height=640)
        except ValueError as error:
            self.status_label.configure(text=str(error))
            self._render_preview(self.frame_scale.get())
            return
        self._preview_roi = roi
        if self.current_page == "Calibración":
            self.cal_roi = roi
        elif self.current_page == "Decodificación":
            self.decode_roi = roi
            for key, value in zip(("roi_left", "roi_top", "roi_width", "roi_height"), (roi.left, roi.top, roi.width, roi.height)):
                variable = self._page_vars.get(key)
                if variable:
                    variable.set(f"{value:.4f}")
        self._render_preview(self.frame_scale.get())


def run_app() -> None:
    """Start the local windowed UI. No browser or local web server is used."""
    PROJECTS_ROOT.mkdir(parents=True, exist_ok=True)
    root = tk.Tk()
    DesktopApp(root)
    root.mainloop()

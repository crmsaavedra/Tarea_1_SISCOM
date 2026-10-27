import tkinter as tk
from tkinter import ttk

import cv2
import numpy as np
import pytest

from siscom.models import Calibration, DecodeResult, ROI, VideoSignal
from siscom.ui_layout import WORKFLOW_PAGES
from siscom.quick_test import QuickVideoAnalysis


def test_tkinter_shell_renders_all_workflows_and_result_tabs():
    from siscom.tk_app import DesktopApp

    try:
        root = tk.Tk()
    except tk.TclError as error:
        pytest.skip(f"Tk display is unavailable: {error}")
    root.withdraw()
    try:
        app = DesktopApp(root)
        assert tuple(app.workflow_buttons) == WORKFLOW_PAGES
        assert isinstance(app.results_notebook, ttk.Notebook)
        assert tuple(app.results_notebook.tab(tab, "text") for tab in app.results_notebook.tabs()) == (
            "Resultado", "Señal", "FFT y PSD"
        )

        for page in WORKFLOW_PAGES:
            app.show_page(page)
            root.update_idletasks()
            assert app.current_page == page
    finally:
        root.destroy()


def test_tkinter_preview_renders_a_portrait_video_frame(tmp_path):
    from siscom.tk_app import DesktopApp

    path = tmp_path / "preview.avi"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"MJPG"), 10, (40, 24))
    if not writer.isOpened():
        pytest.skip("MJPG writer is not available in this OpenCV build")
    try:
        frame = np.zeros((24, 40, 3), dtype=np.uint8)
        frame[8:16, 12:28] = 255
        for _ in range(4):
            writer.write(frame)
    finally:
        writer.release()

    try:
        root = tk.Tk()
    except tk.TclError as error:
        pytest.skip(f"Tk display is unavailable: {error}")
    root.withdraw()
    try:
        app = DesktopApp(root)
        app._load_preview(path, ROI(0.25, 0.25, 0.5, 0.5), editable=True)
        root.update_idletasks()
        assert app._image is not None, app.preview_caption.cget("text")
        assert app._image.width() == 360
        assert app._image.height() == 640
        assert app._preview_frame_count == 4
    finally:
        root.destroy()


def test_quick_test_results_embed_signal_and_spectrum_in_tk_notebook():
    from siscom.tk_app import DesktopApp

    try:
        root = tk.Tk()
    except tk.TclError as error:
        pytest.skip(f"Tk display is unavailable: {error}")
    root.withdraw()
    try:
        app = DesktopApp(root)
        roi = ROI(0.2, 0.2, 0.3, 0.3)
        signal = VideoSignal(
            times=np.arange(128) / 32.0,
            intensity=np.tile(np.r_[np.zeros(16), np.full(16, 220.0)], 4),
            fps=32.0,
            width=96,
            height=64,
            roi=roi,
        )
        analysis = QuickVideoAnalysis(
            signal=signal,
            calibration=Calibration(110, 0.25, 0.4, roi, 32),
            decode=DecodeResult("HOLA", (), ("texto recibido: HOLA",)),
            intensity_contrast=210,
            dot_dash_ratio=3.0,
            diagnostics=(),
        )

        app._display_quick_result(analysis)
        root.update_idletasks()

        assert len(app.result_frames["Señal"].winfo_children()) >= 2
        assert len(app.result_frames["FFT y PSD"].winfo_children()) >= 3
        result_text = app.result_frames["Resultado"].winfo_children()[0]
        assert "CER contra HOLA: 0.0000" in result_text.get("1.0", "end")
    finally:
        root.destroy()

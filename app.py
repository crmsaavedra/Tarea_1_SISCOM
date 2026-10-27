"""Launch the local SisCom Morse Tkinter desktop application."""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path


def main() -> None:
    # `--windowed` sets both streams to None; some native libraries expect
    # file-like streams even when they never emit console output.
    if sys.stdout is None:
        sys.stdout = open(os.devnull, "w", encoding="utf-8")
    if sys.stderr is None:
        sys.stderr = open(os.devnull, "w", encoding="utf-8")
    try:
        from siscom.ui import run_app

        run_app()
    except Exception as error:
        base = Path(os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or Path.home())
        log_path = base / "SisCom Morse" / "startup-error.log"
        try:
            log_path.parent.mkdir(parents=True, exist_ok=True)
            log_path.write_text(traceback.format_exc(), encoding="utf-8")
        except OSError:
            pass
        try:
            import tkinter as tk
            from tkinter import messagebox

            root = tk.Tk()
            root.withdraw()
            messagebox.showerror(
                "No se pudo iniciar SisCom Morse",
                f"{error}\n\nSe guardaron detalles en:\n{log_path}",
                parent=root,
            )
            root.destroy()
        except Exception:
            pass
        raise


if __name__ == "__main__":
    main()

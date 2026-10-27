"""Runtime resource and user-data locations for source and frozen builds."""

from __future__ import annotations

import os
import sys
from pathlib import Path


APP_DIRECTORY_NAME = "SisCom Morse"


def bundled_source_root() -> Path:
    """Return embedded source files in a PyInstaller build, or the app folder."""
    if getattr(sys, "frozen", False):
        return Path(getattr(sys, "_MEIPASS")) / "source"
    return Path(__file__).resolve().parents[1]


def app_data_root() -> Path:
    """Keep frozen-user projects out of the temporary one-file extraction dir."""
    if not getattr(sys, "frozen", False):
        return bundled_source_root()
    base = os.environ.get("LOCALAPPDATA") or os.environ.get("APPDATA") or str(Path.home())
    return Path(base) / APP_DIRECTORY_NAME

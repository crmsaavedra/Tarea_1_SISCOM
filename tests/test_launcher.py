from pathlib import Path


def test_windows_launcher_and_exe_builder_start_the_desktop_app_without_web():
    project_root = Path(__file__).resolve().parents[1]
    launcher = (project_root / "run_app.bat").read_text(encoding="utf-8")
    builder = (project_root / "build_exe.bat").read_text(encoding="utf-8")

    assert "py -3.11 -m venv .venv" in launcher
    assert "py -3.12 -m venv .venv" in launcher
    assert "pythonw.exe\" app.py" in launcher
    assert "--onefile" in builder
    assert "--windowed" in builder
    assert "SisCom Morse.exe" in builder
    assert "streamlit" not in launcher.lower() + builder.lower()

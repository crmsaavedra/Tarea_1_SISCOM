import sys

from siscom.paths import app_data_root, bundled_source_root


def test_frozen_app_uses_local_appdata_for_projects_and_embedded_sources(monkeypatch, tmp_path):
    monkeypatch.setattr(sys, "frozen", True, raising=False)
    monkeypatch.setattr(sys, "_MEIPASS", str(tmp_path / "bundle"), raising=False)
    monkeypatch.setenv("LOCALAPPDATA", str(tmp_path / "local"))

    assert app_data_root() == tmp_path / "local" / "SisCom Morse"
    assert bundled_source_root() == tmp_path / "bundle" / "source"

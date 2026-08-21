from __future__ import annotations

from pathlib import Path

import app


def test_streamlit_cloud_runtime_detects_cloud_env(monkeypatch):
    monkeypatch.setenv("STREAMLIT_CLOUD", "true")
    assert app._is_streamlit_cloud_runtime()


def test_streamlit_cloud_runtime_detects_mount_src(monkeypatch):
    for name in ("STREAMLIT_CLOUD", "STREAMLIT_SHARING_MODE", "IS_RUNNING_IN_STREAMLIT_SHARING"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(app.Path, "cwd", staticmethod(lambda: Path("/mount/src/keiba-lab-publisher")))
    assert app._is_streamlit_cloud_runtime()

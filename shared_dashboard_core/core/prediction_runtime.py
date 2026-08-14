from __future__ import annotations

import contextlib
import io
import sys
import types
from typing import Any, Callable

import pandas as pd


class DisplayCapture:
    def __init__(self) -> None:
        self.objects: list[Any] = []

    def display(self, obj: Any) -> None:
        self.objects.append(obj)

    def reset(self) -> None:
        self.objects.clear()


class HtmlObject:
    def __init__(self, data: Any = "") -> None:
        self.data = data

    def _repr_html_(self) -> str:
        return str(self.data or "")


def install_notebook_shims(capture: DisplayCapture) -> None:
    ipython_mod = types.ModuleType("IPython")
    ipython_display_mod = types.ModuleType("IPython.display")
    ipython_display_mod.display = capture.display
    ipython_display_mod.HTML = HtmlObject
    sys.modules["IPython"] = ipython_mod
    sys.modules["IPython.display"] = ipython_display_mod

    google_mod = sys.modules.get("google") or types.ModuleType("google")
    colab_mod = types.ModuleType("google.colab")
    files_mod = types.ModuleType("google.colab.files")
    files_mod.upload = lambda: {}
    colab_mod.files = files_mod
    google_mod.colab = colab_mod
    sys.modules["google"] = google_mod
    sys.modules["google.colab"] = colab_mod
    sys.modules["google.colab.files"] = files_mod


def build_overall_table(result_df: Any, display_cols: list[str] | tuple[str, ...] | None) -> pd.DataFrame | None:
    if not isinstance(result_df, pd.DataFrame):
        return None
    columns = [column for column in (display_cols or []) if column in result_df.columns]
    if columns:
        return result_df[columns].copy()
    return result_df.copy()


def capture_text(capture: DisplayCapture, func: Callable[[], Any]) -> str:
    capture.reset()
    buffer = io.StringIO()
    with contextlib.redirect_stdout(buffer):
        func()
    return buffer.getvalue()


def capture_first_dataframe(capture: DisplayCapture, func: Callable[[], Any]) -> pd.DataFrame | None:
    capture_text(capture, func)
    for obj in capture.objects:
        frame = display_object_to_dataframe(obj)
        if frame is not None:
            return frame
    return None


def display_object_to_dataframe(obj: Any) -> pd.DataFrame | None:
    if isinstance(obj, pd.DataFrame):
        return obj.copy()
    data = getattr(obj, "data", None)
    if isinstance(data, pd.DataFrame):
        return data.copy()
    return None


def split_attention_horses(text: str) -> list[str]:
    blocks: list[str] = []
    current: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped or stripped in {"【注目馬】", "注目馬"}:
            if current:
                blocks.append("\n".join(current))
                current = []
            continue
        current.append(stripped)
    if current:
        blocks.append("\n".join(current))
    return blocks

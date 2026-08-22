from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any


class KeibaUploadError(ValueError):
    """A Streamlit upload is not a usable .keiba file."""


@dataclass(frozen=True)
class UploadedKeiba:
    filename: str
    data: bytes
    sha256: str

    @property
    def cache_key(self) -> str:
        return f"loaded:{self.sha256}:{len(self.data)}"


def _filename(uploaded_file: Any) -> str:
    return str(getattr(uploaded_file, "name", "") or "").strip()


def _read_bytes(uploaded_file: Any) -> bytes:
    if hasattr(uploaded_file, "getvalue"):
        data = uploaded_file.getvalue()
    elif hasattr(uploaded_file, "read"):
        data = uploaded_file.read()
    else:
        raise KeibaUploadError("アップロードされたファイルを読み取れません。")
    if isinstance(data, memoryview):
        data = data.tobytes()
    elif isinstance(data, bytearray):
        data = bytes(data)
    if not isinstance(data, bytes):
        raise KeibaUploadError("アップロードされたファイルの形式が不正です。")
    if not data:
        raise KeibaUploadError(".keibaファイルが空です。")
    return data


def read_uploaded_keiba(uploaded_file: Any) -> UploadedKeiba:
    """Read an UploadedFile without relying on a local Windows path."""

    name = _filename(uploaded_file)
    if name and not name.lower().endswith(".keiba"):
        raise KeibaUploadError(".keibaファイルを選択してください。")
    data = _read_bytes(uploaded_file)
    return UploadedKeiba(filename=name, data=data, sha256=hashlib.sha256(data).hexdigest())

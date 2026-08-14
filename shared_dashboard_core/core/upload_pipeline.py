from __future__ import annotations

import io
import re
import shutil
import stat
import tempfile
import zipfile
from dataclasses import dataclass
from pathlib import Path, PurePosixPath
from typing import Any, Callable, Literal

from .summary_loader import SummaryLoadError, load_summary


ArchiveMode = Literal["jra", "nar"]
Builder = Callable[..., tuple[Path, int, list[dict[str, str]]]]

MAX_ARCHIVE_ENTRIES = 5_000
MAX_ARCHIVE_BYTES = 1_000_000_000
MAX_FILE_BYTES = 200_000_000


class UploadProcessingError(ValueError):
    """Raised when an uploaded archive cannot produce a valid Summary."""


@dataclass(frozen=True)
class UploadResult:
    mode: ArchiveMode
    summary_path: Path
    analyzed_races: int
    errors: tuple[dict[str, str], ...] = ()


def process_html_zip(
    file_name: str,
    archive_bytes: bytes,
    analysis_directory: str | Path,
    *,
    weekend_builder: Builder | None = None,
    daily_builder: Builder | None = None,
    fetch_past_detail: bool = True,
) -> UploadResult:
    mode = detect_archive_mode(file_name)
    summary_date = date_from_archive_name(file_name)
    analysis_dir = Path(analysis_directory)

    with tempfile.TemporaryDirectory(prefix=f"keiba_dashboard_{mode}_") as temporary:
        temporary_root = Path(temporary)
        extracted = temporary_root / "extracted"
        staging_analysis = temporary_root / "analysis"
        safe_extract_zip(archive_bytes, extracted)

        if mode == "jra":
            builder = weekend_builder or _load_weekend_builder()
            summary_name = "weekend_summary.json"
            detail_directory = "results"
        else:
            builder = daily_builder or _load_daily_builder()
            summary_name = "nar_daily_summary.json"
            detail_directory = "nar_results"

        staging_summary = staging_analysis / summary_name
        _output, success_count, errors = builder(
            extracted,
            staging_summary,
            summary_date=summary_date,
            fetch_past_detail=fetch_past_detail,
        )
        if success_count <= 0:
            detail = errors[0].get("message", "") if errors else ""
            message = "解析できるレースがありませんでした。"
            raise UploadProcessingError(f"{message} {detail}".strip())
        try:
            summary = load_summary(staging_summary)
        except SummaryLoadError as exc:
            raise UploadProcessingError(str(exc)) from exc
        if summary is None:
            raise UploadProcessingError("Summary JSONが生成されませんでした。")

        destination_summary = _publish_analysis(
            staging_analysis,
            analysis_dir,
            summary_name=summary_name,
            detail_directory=detail_directory,
        )
        return UploadResult(
            mode=mode,
            summary_path=destination_summary,
            analyzed_races=success_count,
            errors=tuple(dict(item) for item in errors),
        )


def detect_archive_mode(file_name: str) -> ArchiveMode:
    name = Path(str(file_name or "")).name.lower()
    if re.fullmatch(r"collected_html_jra_[^/\\]+\.zip", name):
        return "jra"
    if re.fullmatch(r"collected_html_nar_[^/\\]+\.zip", name):
        return "nar"
    raise UploadProcessingError(
        "ファイル名は collected_html_jra_xxxxx.zip または collected_html_nar_xxxxx.zip にしてください。"
    )


def date_from_archive_name(file_name: str) -> str:
    match = re.search(r"(?<!\d)(20\d{6})(?!\d)", str(file_name or ""))
    if not match:
        return ""
    value = match.group(1)
    return f"{value[:4]}-{value[4:6]}-{value[6:8]}"


def safe_extract_zip(archive_bytes: bytes, destination: str | Path) -> Path:
    target_root = Path(destination)
    target_root.mkdir(parents=True, exist_ok=True)
    try:
        archive = zipfile.ZipFile(io.BytesIO(archive_bytes))
    except (zipfile.BadZipFile, OSError) as exc:
        raise UploadProcessingError("ZIPファイルを開けませんでした。") from exc

    with archive:
        members = archive.infolist()
        if not members:
            raise UploadProcessingError("ZIPファイルが空です。")
        if len(members) > MAX_ARCHIVE_ENTRIES:
            raise UploadProcessingError("ZIP内のファイル数が上限を超えています。")
        total_size = sum(max(0, item.file_size) for item in members)
        if total_size > MAX_ARCHIVE_BYTES:
            raise UploadProcessingError("ZIP展開後のサイズが上限を超えています。")

        for item in members:
            relative = _safe_member_path(item)
            output = target_root.joinpath(*relative.parts)
            if item.is_dir():
                output.mkdir(parents=True, exist_ok=True)
                continue
            if item.file_size > MAX_FILE_BYTES:
                raise UploadProcessingError(f"ZIP内のファイルが大きすぎます: {item.filename}")
            output.parent.mkdir(parents=True, exist_ok=True)
            with archive.open(item) as source, output.open("wb") as destination_file:
                shutil.copyfileobj(source, destination_file)
    return target_root


def _safe_member_path(item: zipfile.ZipInfo) -> PurePosixPath:
    relative = PurePosixPath(item.filename.replace("\\", "/"))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise UploadProcessingError(f"安全でないZIPパスです: {item.filename}")
    if ":" in relative.parts[0]:
        raise UploadProcessingError(f"安全でないZIPパスです: {item.filename}")
    file_type = (item.external_attr >> 16) & 0o170000
    if file_type == stat.S_IFLNK:
        raise UploadProcessingError(f"ZIP内のシンボリックリンクは使用できません: {item.filename}")
    return relative


def _publish_analysis(
    staging_analysis: Path,
    analysis_directory: Path,
    *,
    summary_name: str,
    detail_directory: str,
) -> Path:
    analysis_directory.mkdir(parents=True, exist_ok=True)
    source_details = staging_analysis / detail_directory
    if source_details.is_dir():
        shutil.copytree(source_details, analysis_directory / detail_directory, dirs_exist_ok=True)

    source_summary = staging_analysis / summary_name
    destination_summary = analysis_directory / summary_name
    temporary_summary = analysis_directory / f".{summary_name}.tmp"
    shutil.copy2(source_summary, temporary_summary)
    temporary_summary.replace(destination_summary)
    return destination_summary


def _load_weekend_builder() -> Builder:
    from tools.build_weekend_summary import build_from_directory

    return build_from_directory


def _load_daily_builder() -> Builder:
    from tools.build_daily_summary import build_daily_from_directory

    return build_daily_from_directory

from __future__ import annotations

import hashlib
import io
import re
import stat
import zipfile
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath
from typing import Callable, Iterable, Mapping

from .html_classifier import (
    classify_html,
    decode_uploaded_html,
    extract_meta,
    kind_label,
    required_kinds,
)
from .prediction_input import predict_from_html_inputs
from .prediction_snapshot import KeibaSnapshotError, build_event_snapshot, race_snapshot_from_result


MAX_ARCHIVE_ENTRIES = 10_000
MAX_ARCHIVE_BYTES = 1_500_000_000
MAX_FILE_BYTES = 250_000_000
HTML_SUFFIXES = {".html", ".htm"}


class BatchPredictionError(ValueError):
    """The uploaded files cannot be converted into any Mobile predictions."""


@dataclass(frozen=True)
class UploadedSource:
    file_name: str
    data: bytes


@dataclass
class RaceHtmlBundle:
    race_id: str
    race_mode: str
    html_files: dict[str, str] = field(default_factory=dict)
    file_names: dict[str, str] = field(default_factory=dict)
    digests: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class BatchPredictionReport:
    event_snapshot: dict
    input_file_count: int
    html_file_count: int
    recognized_file_count: int
    predicted_race_count: int
    skipped_race_count: int
    warnings: tuple[str, ...]
    errors: tuple[str, ...]


ProgressCallback = Callable[[int, int, str], None]


def predict_uploaded_sources(
    sources: Iterable[UploadedSource],
    *,
    prediction_logic_version: str = "market",
    progress: ProgressCallback | None = None,
) -> BatchPredictionReport:
    """Classify every HTML then call the canonical Mobile predictor per race."""

    source_list = list(sources)
    if not source_list:
        raise BatchPredictionError("HTMLまたはZIPを追加してください。")
    entries, expansion_warnings = expand_uploaded_sources(source_list)
    bundles, classification_warnings, classification_errors, recognized = group_html_by_race(entries)
    race_snapshots: list[dict] = []
    prediction_errors: list[str] = []
    total = len(bundles)
    for index, race_id in enumerate(sorted(bundles), start=1):
        bundle = bundles[race_id]
        if progress:
            progress(index, total, f"{bundle.race_mode.upper()} {race_id}")
        missing = [kind for kind in required_kinds(bundle.race_mode) if kind not in bundle.html_files]
        if missing:
            bundle.errors.append(
                "Mobile必須HTML不足: " + ", ".join(kind_label(kind) for kind in missing)
            )
        if bundle.errors:
            prediction_errors.extend(f"{race_id}: {message}" for message in bundle.errors)
            continue
        try:
            result = predict_from_html_inputs(
                bundle.race_mode,  # type: ignore[arg-type]
                bundle.html_files,
                bundle.file_names,
                prediction_logic_version=prediction_logic_version,
            )
            if result.status != "ok":
                raise RuntimeError(result.message or "PredictionResult status != ok")
            if result.overall_table is None or result.horse_evaluation is None:
                raise RuntimeError("PredictionResultの全頭表または馬評価がありません")
            race_snapshots.append(
                race_snapshot_from_result(result, source_files=bundle.file_names)
            )
        except Exception as exc:
            prediction_errors.append(f"{race_id}: 予想失敗: {exc}")
    if not race_snapshots:
        details = (classification_errors + prediction_errors)[:5]
        suffix = " / ".join(details)
        raise BatchPredictionError(("予想可能なレースがありませんでした。 " + suffix).strip())
    try:
        event = build_event_snapshot(race_snapshots)
    except KeibaSnapshotError as exc:
        raise BatchPredictionError(str(exc)) from exc
    warnings = tuple(expansion_warnings + classification_warnings)
    errors = tuple(classification_errors + prediction_errors)
    return BatchPredictionReport(
        event_snapshot=event,
        input_file_count=len(source_list),
        html_file_count=len(entries),
        recognized_file_count=recognized,
        predicted_race_count=len(race_snapshots),
        skipped_race_count=max(0, len(bundles) - len(race_snapshots)),
        warnings=warnings,
        errors=errors,
    )


def expand_uploaded_sources(
    sources: Iterable[UploadedSource],
) -> tuple[list[UploadedSource], list[str]]:
    entries: list[UploadedSource] = []
    warnings: list[str] = []
    for source in sources:
        suffix = Path(source.file_name).suffix.lower()
        if suffix in HTML_SUFFIXES:
            entries.append(source)
            continue
        if suffix != ".zip":
            warnings.append(f"対象外ファイルを無視しました: {source.file_name}")
            continue
        entries.extend(_html_entries_from_zip(source))
    if not entries:
        raise BatchPredictionError("HTMLを含む入力がありません。")
    return entries, warnings


def group_html_by_race(
    entries: Iterable[UploadedSource],
) -> tuple[dict[str, RaceHtmlBundle], list[str], list[str], int]:
    bundles: dict[str, RaceHtmlBundle] = {}
    warnings: list[str] = []
    errors: list[str] = []
    recognized = 0
    for entry in entries:
        text = decode_uploaded_html(entry.data)
        meta = extract_meta(entry.file_name, text)
        if _is_result_page(meta):
            warnings.append(f"確定結果HTMLは予想入力に使用しません: {entry.file_name}")
            continue
        if not meta.race_id:
            warnings.append(f"race_id不明のHTMLを無視しました: {entry.file_name}")
            continue
        if meta.detected_mode not in {"jra", "nar"}:
            warnings.append(f"JRA/NAR不明のHTMLを無視しました: {entry.file_name}")
            continue
        classified = classify_html(entry.file_name, text, meta.detected_mode)  # type: ignore[arg-type]
        if classified.kind == "unknown":
            warnings.append(f"種別不明HTMLを解析しません: {entry.file_name}")
            continue
        recognized += 1
        bundle = bundles.setdefault(
            meta.race_id,
            RaceHtmlBundle(race_id=meta.race_id, race_mode=meta.detected_mode),
        )
        if bundle.race_mode != meta.detected_mode:
            bundle.errors.append(
                f"同じrace_idにJRA/NARが混在しています ({bundle.race_mode}/{meta.detected_mode})"
            )
            continue
        digest = hashlib.sha256(entry.data).hexdigest()
        previous = bundle.digests.get(classified.kind)
        if previous:
            if previous == digest:
                bundle.warnings.append(
                    f"同一HTML重複を1件に統合: {kind_label(classified.kind)}"
                )
                continue
            bundle.errors.append(
                f"同一race_id・同一kindが複数あります: {kind_label(classified.kind)}"
            )
            continue
        bundle.html_files[classified.kind] = text
        bundle.file_names[classified.kind] = entry.file_name
        bundle.digests[classified.kind] = digest
    for race_id, bundle in bundles.items():
        warnings.extend(f"{race_id}: {message}" for message in bundle.warnings)
        errors.extend(f"{race_id}: {message}" for message in bundle.errors)
    return bundles, warnings, errors, recognized


def _html_entries_from_zip(source: UploadedSource) -> list[UploadedSource]:
    try:
        archive = zipfile.ZipFile(io.BytesIO(source.data))
    except (zipfile.BadZipFile, OSError) as exc:
        raise BatchPredictionError(f"ZIPを開けません: {source.file_name}") from exc
    entries: list[UploadedSource] = []
    with archive:
        members = archive.infolist()
        if not members:
            raise BatchPredictionError(f"空のZIPです: {source.file_name}")
        if len(members) > MAX_ARCHIVE_ENTRIES:
            raise BatchPredictionError(f"ZIP内ファイル数が上限を超えています: {source.file_name}")
        total = sum(max(0, item.file_size) for item in members)
        if total > MAX_ARCHIVE_BYTES:
            raise BatchPredictionError(f"ZIP展開サイズが上限を超えています: {source.file_name}")
        for item in members:
            relative = _safe_member_path(item)
            if item.is_dir() or relative.suffix.lower() not in HTML_SUFFIXES:
                continue
            if item.file_size > MAX_FILE_BYTES:
                raise BatchPredictionError(f"ZIP内HTMLが大きすぎます: {item.filename}")
            try:
                payload = archive.read(item)
            except (RuntimeError, zipfile.BadZipFile) as exc:
                raise BatchPredictionError(f"ZIP内HTMLを読めません: {item.filename}") from exc
            entries.append(
                UploadedSource(
                    file_name=f"{Path(source.file_name).name}!/{relative.as_posix()}",
                    data=payload,
                )
            )
    return entries


def _safe_member_path(item: zipfile.ZipInfo) -> PurePosixPath:
    relative = PurePosixPath(item.filename.replace("\\", "/"))
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise BatchPredictionError(f"ZIP内に安全でないパスがあります: {item.filename}")
    if ":" in relative.parts[0]:
        raise BatchPredictionError(f"ZIP内に安全でないパスがあります: {item.filename}")
    file_type = (item.external_attr >> 16) & 0o170000
    if file_type == stat.S_IFLNK:
        raise BatchPredictionError(f"ZIP内シンボリックリンクは使用できません: {item.filename}")
    return relative


def race_prediction_signature(race_snapshot: Mapping) -> tuple[tuple, ...]:
    """Stable comparison of the six outputs required to match Mobile."""

    mobile = race_snapshot.get("mobile_snapshot") or {}
    horses = mobile.get("horses") or []
    signature: list[tuple] = []
    for horse in horses:
        prediction = horse.get("prediction") or {}
        market = horse.get("market_compare") or horse.get("market") or {}
        signature.append(
            (
                str(horse.get("horse_no") or ""),
                prediction.get("ability_value"),
                prediction.get("ability_rank") or market.get("market_ability_rank"),
                prediction.get("ability_band") or market.get("ability_band"),
                market.get("current_evaluation_rank") or prediction.get("ai_current_rank"),
                market.get("ai_current_mark") or prediction.get("mark"),
                bool(horse.get("value_support", {}).get("is_value") or market.get("value_signal")),
            )
        )
    return tuple(signature)


def _is_result_page(meta) -> bool:
    trusted = " ".join((meta.canonical, meta.og_url, *meta.page_urls)).lower()
    return bool(
        "/race/result.html" in trusted
        or "pid=race_result" in trusted
        or str(meta.body_id).lower() == "netkeiba_race_result"
        or "結果・払戻" in str(meta.title)
    )

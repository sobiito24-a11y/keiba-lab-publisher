from __future__ import annotations

import copy
import hashlib
import hmac
import io
import json
import math
import re
import stat
import zipfile
from datetime import date, datetime
from pathlib import PurePosixPath
from typing import Any, Iterable, Mapping

import pandas as pd

from .models import PredictionResult
from .prediction_history import build_prediction_snapshot
from .version import APP_VERSION, PREDICTION_LOGIC_VERSION


KEIBA_FORMAT = "keiba-prediction-snapshot"
SCHEMA_VERSION = 1
MAX_KEIBA_BYTES = 250_000_000
MAX_MEMBER_BYTES = 200_000_000


class KeibaSnapshotError(ValueError):
    """A saved prediction cannot be opened safely."""


class UnsupportedSchemaError(KeibaSnapshotError):
    """The file uses a schema this application does not support."""


def race_snapshot_from_result(
    result: PredictionResult,
    *,
    source_files: Mapping[str, str] | None = None,
) -> dict[str, Any]:
    """Freeze one Mobile PredictionResult without running another prediction."""

    mobile_snapshot = build_prediction_snapshot(result)
    race_info = dict(mobile_snapshot.get("race_info") or {})
    race_id = _text(race_info.get("race_id"))
    if not race_id:
        raise KeibaSnapshotError("PredictionResultにrace_idがありません。")
    race_number = _race_number_from_id(race_id) or _text(race_info.get("race_number"))
    return _json_ready(
        {
            "race_id": race_id,
            "race_mode": "nar" if result.race_mode == "nar" else "jra",
            "date": _text(race_info.get("date")),
            "venue": _text(race_info.get("venue")),
            "race_number": race_number,
            "race_name": _text(race_info.get("race_name") or result.race_name),
            "distance": race_info.get("distance", ""),
            "surface": race_info.get("surface", ""),
            "head_count": race_info.get("head_count", ""),
            "prediction_created_at": result.created_at,
            "app_version": result.version or APP_VERSION,
            "prediction_logic_version": result.logic_version or "market",
            "horses": _dashboard_horse_snapshots(result, mobile_snapshot),
            "mobile_snapshot": mobile_snapshot,
            "prediction_result": serialize_prediction_result(result),
            "input_summary": {
                "kinds": sorted((source_files or {}).keys()),
                "file_names": dict(source_files or {}),
                "html_embedded": False,
            },
        }
    )


def build_event_snapshot(races: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    race_list = [_json_ready(dict(item)) for item in races]
    if not race_list:
        raise KeibaSnapshotError("保存できる予想レースがありません。")
    _validate_unique_race_ids(race_list)
    race_list.sort(key=_race_sort_key)
    created_values = [_text(item.get("prediction_created_at")) for item in race_list]
    created_at = min((value for value in created_values if value), default=datetime.now().isoformat(timespec="seconds"))
    snapshot = {
        "format": KEIBA_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "app_version": APP_VERSION,
        "prediction_logic_version": PREDICTION_LOGIC_VERSION,
        "prediction_created_at": created_at,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "scope": _scope_for_races(race_list),
        "races": race_list,
    }
    validate_event_snapshot(snapshot)
    return snapshot


def serialize_prediction_result(result: PredictionResult) -> dict[str, Any]:
    return _json_ready(
        {
            "race_mode": result.race_mode,
            "version": result.version,
            "created_at": result.created_at,
            "race_name": result.race_name,
            "race_info": result.race_info,
            "overall_table": _serialize_frame(result.overall_table),
            "horse_evaluation": _serialize_frame(result.horse_evaluation),
            "attention_horses": result.attention_horses,
            "ai_race_review": result.ai_race_review,
            "betting_structure": result.betting_structure,
            "source_files": result.source_files,
            "status": result.status,
            "message": result.message,
            "raw_output": result.raw_output,
            "debug_info": result.debug_info,
            "logic_version": result.logic_version,
            "ver4_summary": result.ver4_summary,
        }
    )


def restore_prediction_result(race_snapshot: Mapping[str, Any]) -> PredictionResult:
    payload = race_snapshot.get("prediction_result")
    if not isinstance(payload, Mapping):
        raise KeibaSnapshotError("保存ファイルにPredictionResultがありません。")
    mode = _text(payload.get("race_mode")).lower()
    if mode not in {"jra", "nar"}:
        raise KeibaSnapshotError("PredictionResultのJRA/NAR情報が不正です。")
    result = PredictionResult(
        race_mode=mode,  # type: ignore[arg-type]
        version=_text(payload.get("version")) or APP_VERSION,
        created_at=_text(payload.get("created_at")),
        race_name=_text(payload.get("race_name")),
        race_info=dict(payload.get("race_info") or {}),
        overall_table=_restore_frame(payload.get("overall_table")),
        horse_evaluation=_restore_frame(payload.get("horse_evaluation")),
        attention_horses=list(payload.get("attention_horses") or []),
        ai_race_review=_text(payload.get("ai_race_review")),
        betting_structure=_text(payload.get("betting_structure")),
        source_files=dict(payload.get("source_files") or {}),
        status=_text(payload.get("status")),
        message=_text(payload.get("message")),
        raw_output=_text(payload.get("raw_output")),
        debug_info=dict(payload.get("debug_info") or {}),
        logic_version=_text(payload.get("logic_version")) or "market",
        ver4_summary=dict(payload.get("ver4_summary") or {}),
    )
    return result


def replace_race_result(
    event_snapshot: Mapping[str, Any],
    race_id: str,
    result: PredictionResult,
) -> dict[str, Any]:
    """Persist UI-only changes such as the user's selections in memory."""

    event = copy.deepcopy(dict(event_snapshot))
    found = False
    for index, race in enumerate(event.get("races") or []):
        if _text(race.get("race_id")) != _text(race_id):
            continue
        source = ((race.get("input_summary") or {}).get("file_names") or {})
        event["races"][index] = race_snapshot_from_result(result, source_files=source)
        found = True
        break
    if not found:
        raise KeibaSnapshotError(f"race_id={race_id} は保存データにありません。")
    event["saved_at"] = datetime.now().isoformat(timespec="seconds")
    event["scope"] = _scope_for_races(event["races"])
    validate_event_snapshot(event)
    return event


def update_user_selection(
    event_snapshot: Mapping[str, Any],
    race_id: str,
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    """Update only user-authored fields; never rebuild historical prediction facts."""

    validate_event_snapshot(event_snapshot)
    event = copy.deepcopy(dict(event_snapshot))
    selected = _json_ready(dict(selection))
    for race in event.get("races") or []:
        if _text(race.get("race_id")) != _text(race_id):
            continue
        mobile = race.setdefault("mobile_snapshot", {})
        market = mobile.setdefault("market_compare", {})
        market["user_selection"] = copy.deepcopy(selected)
        prediction = race.setdefault("prediction_result", {})
        debug = prediction.setdefault("debug_info", {})
        prediction_market = debug.setdefault("market_compare", {})
        prediction_market["user_selection"] = copy.deepcopy(selected)
        race["user_selection"] = copy.deepcopy(selected)
        event["saved_at"] = datetime.now().isoformat(timespec="seconds")
        validate_event_snapshot(event)
        return event
    raise KeibaSnapshotError(f"race_id={race_id} は保存データにありません。")


def subset_event_snapshot(
    event_snapshot: Mapping[str, Any],
    *,
    race_date: str = "",
    race_mode: str = "",
    venue: str = "",
) -> dict[str, Any]:
    validate_event_snapshot(event_snapshot)
    races = [
        race
        for race in event_snapshot.get("races", [])
        if (not race_date or _text(race.get("date")) == race_date)
        and (not race_mode or _text(race.get("race_mode")) == race_mode)
        and (not venue or _text(race.get("venue")) == venue)
    ]
    return build_event_snapshot(races)


def keiba_bytes(event_snapshot: Mapping[str, Any]) -> bytes:
    validate_event_snapshot(event_snapshot)
    snapshot_bytes = json.dumps(
        _json_ready(dict(event_snapshot)),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
    manifest = {
        "format": KEIBA_FORMAT,
        "schema_version": SCHEMA_VERSION,
        "app_version": _text(event_snapshot.get("app_version")) or APP_VERSION,
        "prediction_logic_version": _text(event_snapshot.get("prediction_logic_version")),
        "saved_at": _text(event_snapshot.get("saved_at")),
        "race_count": len(event_snapshot.get("races") or []),
        "snapshot_sha256": hashlib.sha256(snapshot_bytes).hexdigest(),
    }
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False, indent=2))
        archive.writestr("snapshot.json", snapshot_bytes)
    return output.getvalue()


def load_keiba(data: bytes) -> dict[str, Any]:
    if not data:
        raise KeibaSnapshotError(".keibaファイルが空です。")
    if len(data) > MAX_KEIBA_BYTES:
        raise KeibaSnapshotError(".keibaファイルがサイズ上限を超えています。")
    try:
        archive = zipfile.ZipFile(io.BytesIO(data))
    except (zipfile.BadZipFile, OSError) as exc:
        raise KeibaSnapshotError("破損した.keibaファイルです。") from exc
    with archive:
        members = archive.infolist()
        names = [item.filename for item in members]
        if names.count("manifest.json") != 1 or names.count("snapshot.json") != 1:
            raise KeibaSnapshotError(".keiba内部のmanifest/snapshot構成が不正です。")
        if set(names) != {"manifest.json", "snapshot.json"}:
            raise KeibaSnapshotError(".keiba内部に未対応ファイルがあります。")
        for item in members:
            _validate_keiba_member(item)
        try:
            manifest_raw = archive.read("manifest.json")
            snapshot_raw = archive.read("snapshot.json")
        except (KeyError, RuntimeError, zipfile.BadZipFile) as exc:
            raise KeibaSnapshotError(".keiba内部データを読み取れません。") from exc
    try:
        manifest = json.loads(manifest_raw.decode("utf-8"))
        snapshot = json.loads(snapshot_raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise KeibaSnapshotError(".keiba内部JSONが破損しています。") from exc
    if not isinstance(manifest, Mapping) or not isinstance(snapshot, Mapping):
        raise KeibaSnapshotError(".keiba内部JSONの型が不正です。")
    if _text(manifest.get("format")) != KEIBA_FORMAT:
        raise KeibaSnapshotError(".keiba manifestの形式が不正です。")
    if manifest.get("schema_version") != snapshot.get("schema_version"):
        raise KeibaSnapshotError(".keiba manifestとsnapshotのschema_versionが一致しません。")
    if manifest.get("race_count") not in (None, len(snapshot.get("races") or [])):
        raise KeibaSnapshotError(".keiba manifestのレース件数が一致しません。")
    expected = _text(manifest.get("snapshot_sha256"))
    if not expected or not _constant_equal(expected, hashlib.sha256(snapshot_raw).hexdigest()):
        raise KeibaSnapshotError(".keibaの整合性確認に失敗しました。")
    validate_event_snapshot(snapshot)
    return copy.deepcopy(dict(snapshot))


def validate_event_snapshot(snapshot: Mapping[str, Any]) -> None:
    if _text(snapshot.get("format")) != KEIBA_FORMAT:
        raise KeibaSnapshotError("Keiba AI Prediction Snapshotではありません。")
    schema = snapshot.get("schema_version")
    if not isinstance(schema, int):
        raise KeibaSnapshotError("schema_versionがありません。")
    if schema != SCHEMA_VERSION:
        direction = "旧" if schema < SCHEMA_VERSION else "新"
        raise UnsupportedSchemaError(
            f"{direction}schema_version={schema}はこの版では読み込めません（対応={SCHEMA_VERSION}）。"
        )
    races = snapshot.get("races")
    if not isinstance(races, list) or not races:
        raise KeibaSnapshotError("保存ファイルにレース予想がありません。")
    _validate_unique_race_ids(races)
    for race in races:
        if not isinstance(race, Mapping):
            raise KeibaSnapshotError("レースSnapshotの型が不正です。")
        if not _text(race.get("race_id")):
            raise KeibaSnapshotError("race_idがないレースSnapshotがあります。")
        if _text(race.get("race_mode")) not in {"jra", "nar"}:
            raise KeibaSnapshotError("レースSnapshotのJRA/NAR情報が不正です。")
        if not isinstance(race.get("prediction_result"), Mapping):
            raise KeibaSnapshotError("レースSnapshotにPredictionResultがありません。")


def keiba_file_name(event_snapshot: Mapping[str, Any]) -> str:
    scope = event_snapshot.get("scope") if isinstance(event_snapshot.get("scope"), Mapping) else {}
    dates = list(scope.get("dates") or [])
    modes = list(scope.get("race_modes") or [])
    venues = list(scope.get("venues") or [])
    date_text = _safe_name(dates[0] if len(dates) == 1 else "multiple_dates")
    mode_text = _safe_name(modes[0].upper() if len(modes) == 1 else "JRA_NAR")
    venue_text = _safe_name(venues[0] if len(venues) == 1 else "all_venues")
    return f"{date_text}_{mode_text}_{venue_text}.keiba"


def race_by_id(event_snapshot: Mapping[str, Any], race_id: str) -> dict[str, Any]:
    for race in event_snapshot.get("races") or []:
        if _text(race.get("race_id")) == _text(race_id):
            return dict(race)
    raise KeibaSnapshotError(f"race_id={race_id} は保存データにありません。")


def _serialize_frame(value: Any) -> dict[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, pd.DataFrame):
        try:
            value = pd.DataFrame(value)
        except Exception as exc:  # pragma: no cover - defensive boundary
            raise KeibaSnapshotError("PredictionResult内の表を保存できません。") from exc
    return {
        "columns": [str(column) for column in value.columns],
        "records": _json_ready(value.to_dict(orient="records")),
    }


def _dashboard_horse_snapshots(
    result: PredictionResult,
    mobile_snapshot: Mapping[str, Any],
) -> list[dict[str, Any]]:
    """Copy display facts into stable, machine-matchable horse records."""

    rows: dict[str, dict[str, Any]] = {}
    for table in (result.overall_table, result.horse_evaluation):
        if not isinstance(table, pd.DataFrame) or table.empty:
            continue
        for record in table.to_dict(orient="records"):
            number = _horse_number(_first(record, "馬番", "馬", "horse_no", "horse_number"))
            if not number:
                continue
            merged = rows.setdefault(number, {})
            for key, value in record.items():
                if not _is_missing(value):
                    merged[str(key)] = value
    saved_by_number = {
        _horse_number(item.get("horse_no")): item
        for item in mobile_snapshot.get("horses", []) or []
        if isinstance(item, Mapping)
    }
    horses: list[dict[str, Any]] = []
    for number in sorted(rows, key=lambda value: int(value) if value.isdigit() else 999):
        row = rows[number]
        saved = saved_by_number.get(number, {})
        saved_support = saved.get("support") if isinstance(saved.get("support"), Mapping) else {}
        saved_value = saved.get("value_support") if isinstance(saved.get("value_support"), Mapping) else {}
        horses.append(
            _json_ready(
                {
                    "horse_no": number,
                    "horse_name": _first(row, "馬名", "horse_name", "name"),
                    "sex_age": _first(row, "馬年齢", "性齢", "馬齢", "sex_age"),
                    "ability_value": _first(
                        row,
                        "market_ability_score",
                        "能力評価値",
                        "ability_display_score",
                    ),
                    "ability_rank": _first(
                        row,
                        "market_ability_rank",
                        "能力順位",
                        "ability_rank",
                    ),
                    "ability_band": _first(
                        row,
                        "ability_band_v2",
                        "能力帯",
                        "ability_band",
                    ),
                    "current_evaluation_rank": _first(
                        row,
                        "current_evaluation_rank",
                        "今回評価順位",
                        "ai_current_rank",
                    ),
                    "mark": _first(row, "ai_current_mark", "表示印", "display_mark", "最終印"),
                    "value_signal": _truthy(
                        _first(row, "value_signal", "妙味あり")
                        or saved_value.get("value_signal")
                    ),
                    "odds_at_prediction": _first(row, "actual_odds_market", "単勝オッズ", "オッズ"),
                    "weight": _first(row, "weight_market", "斤量詳細", "斤量"),
                    "jockey": _first(row, "jockey_market", "騎手", "jockey"),
                    "jockey_display": _first(row, "jockey_display_market", "騎手詳細", "jockey_detail"),
                    "jockey_change": _first(row, "jockey_change_market", "騎手継続/乗替", "jockey_change"),
                    "jockey_course_stats": _first(
                        row,
                        "jockey_course_stats_market",
                        "騎手コース成績",
                    ),
                    "state": _first(row, "state_label_market", "状態", "form_state"),
                    "class": _first(row, "current_class_market", "今回クラス", "クラス"),
                    "interval": _first(row, "race_interval_market", "レース間隔", "間隔"),
                    "development": _first(
                        row,
                        "pace_reason_market",
                        "pace_material_label",
                        "今回の展開",
                    ),
                    "course_material": _first(
                        row,
                        "course_material_label",
                        "今回のコース材料",
                    ) or saved_support.get("course_material_label", ""),
                    "netkeiba_favorable": _first(
                        row,
                        "netkeiba_favorable_label",
                        "netkeiba推定有利馬",
                    ) or saved_support.get("netkeiba_favorable_label", ""),
                    "estimated_position": _first(
                        row,
                        "position_path_market",
                        "estimated_position_label",
                        "想定位置",
                    ),
                    "training_short": _first(
                        row,
                        "training_display",
                        "training_market",
                        "調教短縮評価",
                        "調教評価",
                    ) or saved_support.get("training_display", ""),
                    "stable_comment_summary": _first(
                        row,
                        "stable_comment_display",
                        "厩舎コメント要約",
                        "厩舎コメント",
                    ) or saved_support.get("stable_comment_display", ""),
                    "decision_material": _first(
                        row,
                        "current_evaluation_reason",
                        "評価／検討材料",
                        "評価/検討材料",
                    ),
                    "plus_materials": _list_value(
                        _first(row, "value_plus_materials", "主要＋材料", "＋材料")
                        or saved_value.get("plus_materials")
                    ),
                    "minus_materials": _list_value(
                        _first(row, "value_minus_materials", "主要－材料", "－材料")
                        or saved_value.get("minus_materials")
                    ),
                }
            )
        )
    return horses


def _restore_frame(value: Any) -> pd.DataFrame | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise KeibaSnapshotError("保存された表データの型が不正です。")
    columns = value.get("columns")
    records = value.get("records")
    if not isinstance(columns, list) or not isinstance(records, list):
        raise KeibaSnapshotError("保存された表データの構造が不正です。")
    frame = pd.DataFrame.from_records(records)
    for column in columns:
        if column not in frame.columns:
            frame[column] = None
    return frame.loc[:, columns]


def _scope_for_races(races: list[Mapping[str, Any]]) -> dict[str, Any]:
    return {
        "dates": sorted({_text(item.get("date")) for item in races if _text(item.get("date"))}),
        "race_modes": sorted({_text(item.get("race_mode")) for item in races}),
        "venues": sorted({_text(item.get("venue")) for item in races if _text(item.get("venue"))}),
        "race_count": len(races),
    }


def _validate_unique_race_ids(races: Iterable[Mapping[str, Any]]) -> None:
    seen: set[str] = set()
    duplicates: set[str] = set()
    for race in races:
        race_id = _text(race.get("race_id"))
        if race_id in seen:
            duplicates.add(race_id)
        seen.add(race_id)
    if duplicates:
        raise KeibaSnapshotError("race_idが重複しています: " + ", ".join(sorted(duplicates)))


def _validate_keiba_member(item: zipfile.ZipInfo) -> None:
    relative = PurePosixPath(item.filename.replace("\\", "/"))
    if relative.is_absolute() or ".." in relative.parts or not relative.parts:
        raise KeibaSnapshotError(".keiba内部に安全でないパスがあります。")
    file_type = (item.external_attr >> 16) & 0o170000
    if file_type == stat.S_IFLNK:
        raise KeibaSnapshotError(".keiba内部のシンボリックリンクは使用できません。")
    if item.file_size > MAX_MEMBER_BYTES:
        raise KeibaSnapshotError(".keiba内部データがサイズ上限を超えています。")


def _json_ready(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (datetime, date, pd.Timestamp)):
        return value.isoformat()
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if hasattr(value, "item"):
        try:
            return _json_ready(value.item())
        except (TypeError, ValueError):
            pass
    try:
        if bool(pd.isna(value)):
            return None
    except (TypeError, ValueError):
        pass
    return str(value)


def _race_sort_key(race: Mapping[str, Any]) -> tuple[Any, ...]:
    number = re.search(r"\d+", _text(race.get("race_number")))
    return (
        _text(race.get("date")),
        _text(race.get("race_mode")),
        _text(race.get("venue")),
        int(number.group()) if number else 999,
        _text(race.get("race_id")),
    )


def _race_number_from_id(race_id: str) -> str:
    if not re.fullmatch(r"\d{10,14}", race_id):
        return ""
    number = int(race_id[-2:])
    return f"{number}R" if 1 <= number <= 18 else ""


def _text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if bool(pd.isna(value)):
            return ""
    except (TypeError, ValueError):
        pass
    return re.sub(r"\s+", " ", str(value)).strip()


def _first(source: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in source and not _is_missing(source.get(name)):
            return source.get(name)
    return ""


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
        if isinstance(missing, bool):
            return missing
    except (TypeError, ValueError):
        pass
    return str(value).strip() in {"", "None", "nan", "NaN", "<NA>", "NaT"}


def _horse_number(value: Any) -> str:
    text = _text(value)
    match = re.search(r"\d+", text)
    return str(int(match.group())) if match else ""


def _list_value(value: Any) -> list[Any]:
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value if not _is_missing(item)]
    text = _text(value)
    return [text] if text else []


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)) and not _is_missing(value):
        return value != 0
    return _text(value).lower() in {"true", "1", "yes", "妙味あり", "○"}


def _safe_name(value: Any) -> str:
    text = re.sub(r"[\\/:*?\"<>|\s]+", "_", _text(value)).strip("._")
    return text or "unknown"


def _constant_equal(left: str, right: str) -> bool:
    return hmac.compare_digest(left, right)

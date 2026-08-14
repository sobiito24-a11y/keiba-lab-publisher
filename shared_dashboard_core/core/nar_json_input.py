from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
from html import escape, unescape
from typing import Any, Iterable

import pandas as pd

from .html_classifier import classify_html, classify_netkeiba_page_url
from .nar_courseanalysis_parser import (
    NarCourseAnalysisParseError,
    is_courseanalysis_html,
    parse_courseanalysis_html,
)
from .nar_newspaper_parser import (
    NarNewspaperParseError,
    build_entry_from_nar_newspaper,
    is_nar_newspaper_html,
    parse_nar_newspaper_html,
)
from .star_trace import candidate_summary, log_star_trace, star_trace_row
from .star_trace import clear_star_trace


REQUIRED_NAR_JSON_TYPES = {"entry", "speed", "courseanalysis"}

CURRENT_LOAD_WEIGHT_KEYS = ("weight", "斤量", "load_weight", "current_weight", "current_load_weight")
PREVIOUS_LOAD_WEIGHT_KEYS = (
    "previous_weight",
    "prev_weight",
    "last_weight",
    "前走斤量",
    "previous_load_weight",
    "prev_load_weight",
    "last_load_weight",
    "_previous_load_weight",
)
LOAD_WEIGHT_CHANGE_KEYS = ("weight_change", "load_weight_change", "前走比", "斤量増減", "_load_weight_change")
CURRENT_JOCKEY_KEYS = ("jockey", "騎手", "current_jockey", "_current_jockey")
PREVIOUS_JOCKEY_KEYS = ("previous_jockey", "prev_jockey", "last_jockey", "前走騎手", "_previous_jockey")
JOCKEY_CHANGED_KEYS = ("jockey_changed", "乗り替わり", "_jockey_changed")


class NarJsonDataError(ValueError):
    """Raised when uploaded shortcut data cannot be used for NAR prediction."""


@dataclass(frozen=True)
class NarJsonPredictionInput:
    race_id: str
    html_files: dict[str, str]
    file_names: dict[str, str]
    entry_count: int
    speed_count: int
    running_styles: tuple[str, ...]
    horse_style_count: int = 0
    entry_source: str = "entry"
    debug_logs: tuple[dict[str, Any], ...] = ()
    star_debug_logs: tuple[dict[str, Any], ...] = ()


def build_nar_prediction_inputs_from_uploads(
    uploaded_files: Iterable[tuple[str, bytes]],
) -> NarJsonPredictionInput:
    clear_star_trace()
    classified = classify_nar_uploaded_files(uploaded_files)
    star_debug_logs = _build_star_trace_from_speed_json(classified.get("speed", {}))
    if "entry" not in classified and "newspaper" in classified:
        classified["entry"] = build_entry_from_nar_newspaper(classified["newspaper"])
    elif "entry" in classified and "newspaper" in classified:
        classified["entry"] = apply_newspaper_jockey_priority(classified["entry"], classified["newspaper"])
    validate_nar_uploaded_data(classified)

    entry_data = classified["entry"]
    speed_data = classified["speed"]
    courseanalysis_data = classified["courseanalysis"]
    race_id = str(entry_data.get("race_id", "")).strip()
    jockey_race_id = _safe_text((classified.get("jockey") or {}).get("race_id"))
    if jockey_race_id and jockey_race_id != race_id:
        raise NarJsonDataError(
            f"騎手コース成績HTMLのrace_idが一致していません（必須データ={race_id} / 騎手={jockey_race_id}）"
        )
    merged_horses = merge_entry_and_speed(entry_data, speed_data, courseanalysis_data)
    star_debug_logs.extend(_build_star_trace_from_merged_horses("01b nar_json_input merged", merged_horses))
    running_styles = tuple(
        _safe_text(item.get("style"))
        for item in courseanalysis_data.get("running_styles", [])
        if _safe_text(item.get("style"))
    )

    html_files = {
        "shutuba": build_shutuba_html(entry_data, merged_horses),
        "speed": build_speed_html(speed_data, entry_data, merged_horses),
        "style": build_courseanalysis_html(courseanalysis_data, merged_horses),
    }
    newspaper_source = _safe_text((classified.get("newspaper") or {}).get("_source_html"))
    if newspaper_source:
        # Keep the raw newspaper only for the independent course/development
        # display parser.  The legacy NAR notebook path does not consume this
        # key, so its prediction inputs remain unchanged.
        html_files["newspaper_context"] = newspaper_source
    jockey_source = _safe_text((classified.get("jockey") or {}).get("_source_html"))
    if jockey_source:
        html_files["jockey"] = jockey_source
    star_debug_logs.extend(_extract_speed_star_trace(html_files.get("speed", ""), "02b build_speed_html output attrs"))
    file_names = {
        "shutuba": _suggested_name(entry_data, race_id, "entry"),
        "speed": _suggested_name(speed_data, race_id, "speed"),
        "style": _suggested_name(courseanalysis_data, race_id, "courseanalysis"),
    }
    if newspaper_source:
        file_names["newspaper_context"] = _safe_text(classified["newspaper"].get("_uploaded_file_name"))
    if jockey_source:
        file_names["jockey"] = _safe_text(classified["jockey"].get("_uploaded_file_name"))
    return NarJsonPredictionInput(
        race_id=race_id,
        html_files=html_files,
        file_names=file_names,
        entry_count=len(entry_data.get("horses", [])),
        speed_count=len(speed_data.get("horses", [])),
        running_styles=running_styles,
        horse_style_count=sum(1 for horse in merged_horses if _safe_text(horse.get("running_style"))),
        entry_source=_safe_text(entry_data.get("source")) or "entry",
        debug_logs=tuple(_build_nar_previous_jockey_debug_logs(classified, entry_data, merged_horses, html_files)),
        star_debug_logs=tuple(star_debug_logs),
    )


def apply_newspaper_jockey_priority(entry_data: dict[str, Any], newspaper_data: dict[str, Any]) -> dict[str, Any]:
    """Use per-horse display data from newspaper HTML without changing horse order."""

    result = dict(entry_data)
    newspaper_horses = [horse for horse in newspaper_data.get("horses", []) if isinstance(horse, dict)]

    by_number = {
        _horse_number(horse): horse
        for horse in newspaper_horses
        if _horse_number(horse)
    }
    by_id = {
        str(horse.get("horse_id", "")).strip(): horse
        for horse in newspaper_horses
        if str(horse.get("horse_id", "")).strip()
    }
    by_name = {
        _normalize_name(horse.get("horse_name")): horse
        for horse in newspaper_horses
        if _normalize_name(horse.get("horse_name"))
    }

    merged_horses = []
    for entry_horse in entry_data.get("horses", []):
        horse = dict(entry_horse)
        newspaper_horse = (
            by_number.get(_horse_number(horse))
            or by_id.get(str(horse.get("horse_id", "")).strip())
            or by_name.get(_normalize_name(horse.get("horse_name")))
        )
        jockey = _safe_text((newspaper_horse or {}).get("jockey"))
        if jockey:
            horse["jockey"] = jockey
            horse["_jockey_source"] = "newspaper"
        if newspaper_horse:
            for key in (
                "past_runs",
                "recent_runs",
                "race1_racecourse",
                "race1_venue",
                "race1_track",
                "race1_surface",
                "race1_distance",
                "race1_turn",
                "race1_direction",
                "race1_condition",
                "race2_racecourse",
                "race2_venue",
                "race2_track",
                "race2_surface",
                "race2_distance",
                "race2_turn",
                "race2_direction",
                "race2_condition",
                "race3_racecourse",
                "race3_venue",
                "race3_track",
                "race3_surface",
                "race3_distance",
                "race3_turn",
                "race3_direction",
                "race3_condition",
                "previous_date",
                "previous_track",
                "previous_race",
                "previous_finish",
                "previous_jockey",
                "previous_weight",
                "previous_body_weight",
                "前走日付",
                "前走競馬場",
                "前走レース",
                "前走着順",
                "前走騎手",
                "前走斤量",
                "前走馬体重",
            ):
                raw_value = newspaper_horse.get(key)
                if isinstance(raw_value, (list, dict)):
                    if raw_value:
                        horse[key] = raw_value
                    continue
                value = _safe_text(raw_value)
                if value:
                    horse[key] = value
        merged_horses.append(horse)

    result["horses"] = merged_horses
    return result


def classify_nar_uploaded_files(uploaded_files: Iterable[tuple[str, bytes]]) -> dict[str, dict[str, Any]]:
    classified: dict[str, dict[str, Any]] = {}
    duplicates: dict[str, list[str]] = {}
    invalid_files: list[str] = []

    for file_name, raw in uploaded_files:
        text = decode_uploaded_text(raw)
        data = try_load_json(text)
        if data is not None:
            data_type = str(data.get("data_type", "")).strip()
            if data_type == "error" and classify_netkeiba_page_url(str(data.get("url") or "")) == "jockey":
                # A failed optional shortcut fetch must not block the three
                # required prediction inputs. It contains no jockey data and
                # is therefore deliberately ignored rather than parsed.
                continue
            if data_type not in REQUIRED_NAR_JSON_TYPES:
                invalid_files.append(f"{file_name}: data_type が不正です（{data_type or '未取得'}）")
                continue
            _add_classified_data(classified, duplicates, data_type, data, file_name)
            continue

        if "<html" in text.lower():
            html_item = classify_html(file_name, text, "nar")
            if html_item.kind == "jockey":
                jockey_data = {
                    "race_id": html_item.meta.race_id,
                    "data_type": "jockey",
                    "_source_html": text,
                }
                _add_classified_data(classified, duplicates, "jockey", jockey_data, file_name)
                continue

        if is_courseanalysis_html(text):
            try:
                courseanalysis_data = parse_courseanalysis_html(text)
            except NarCourseAnalysisParseError as exc:
                invalid_files.append(f"{file_name}: コース分析HTMLの解析に失敗しました（{exc}）")
                continue
            _add_classified_data(classified, duplicates, "courseanalysis", courseanalysis_data, file_name)
            continue

        if is_nar_newspaper_html(text):
            try:
                newspaper_data = parse_nar_newspaper_html(text)
            except NarNewspaperParseError as exc:
                invalid_files.append(f"{file_name}: 競馬新聞HTMLの解析に失敗しました（{exc}）")
                continue
            newspaper_data["_source_html"] = text
            _add_classified_data(classified, duplicates, "newspaper", newspaper_data, file_name)
            continue

        invalid_files.append(f"{file_name}: entry/speed JSON、courseanalysis HTML、または競馬新聞HTMLとして判定できません")

    if invalid_files:
        raise NarJsonDataError("読み込めないファイルがあります。\n" + "\n".join(invalid_files))
    if duplicates:
        details = [f"{data_type}: {', '.join(names)}" for data_type, names in duplicates.items()]
        raise NarJsonDataError("同じ種類のファイルが複数あります。採用するファイルを1つにしてください。\n" + "\n".join(details))
    return classified


def classify_uploaded_json_files(uploaded_files: Iterable[tuple[str, bytes]]) -> dict[str, dict[str, Any]]:
    return classify_nar_uploaded_files(uploaded_files)


def _add_classified_data(
    classified: dict[str, dict[str, Any]],
    duplicates: dict[str, list[str]],
    data_type: str,
    data: dict[str, Any],
    file_name: str,
) -> None:
    if data_type in classified:
        duplicates.setdefault(data_type, []).append(file_name)
        return
    data["_uploaded_file_name"] = file_name
    classified[data_type] = data


def decode_uploaded_file(uploaded_file: Any) -> str:
    if isinstance(uploaded_file, bytes):
        return decode_uploaded_text(uploaded_file)
    if hasattr(uploaded_file, "getvalue"):
        return decode_uploaded_text(uploaded_file.getvalue())
    return decode_uploaded_text(bytes(uploaded_file))


def try_load_json(text: str) -> dict[str, Any] | None:
    source = str(text or "").strip()
    if not source:
        return None
    try:
        data = json.loads(source)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", source, flags=re.S)
        if not match:
            return None
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError:
            return None
    return data if isinstance(data, dict) else None


def load_uploaded_json(raw: bytes) -> dict[str, Any]:
    data = try_load_json(decode_uploaded_text(raw))
    if data is None:
        raise NarJsonDataError("JSON本文を取得できませんでした。")
    return data


def decode_uploaded_text(raw: bytes) -> str:
    for encoding in ("utf-8-sig", "utf-8", "cp932", "euc-jp"):
        try:
            return raw.decode(encoding)
        except UnicodeDecodeError:
            continue
    return raw.decode("utf-8", errors="replace")


def validate_nar_uploaded_data(classified: dict[str, dict[str, Any]]) -> None:
    missing = REQUIRED_NAR_JSON_TYPES - set(classified)
    if missing:
        labels = ", ".join(_type_label(value) for value in sorted(missing))
        raise NarJsonDataError(f"必要なデータが不足しています: {labels}")

    race_ids = {str(classified[key].get("race_id", "")).strip() for key in REQUIRED_NAR_JSON_TYPES}
    if len(race_ids) != 1 or "" in race_ids:
        details = ", ".join(f"{_type_label(key)}={classified[key].get('race_id', '')}" for key in sorted(REQUIRED_NAR_JSON_TYPES))
        raise NarJsonDataError("3ファイルのrace_idが一致していません。\n" + details)

    entry_horses = classified["entry"].get("horses", [])
    speed_horses = classified["speed"].get("horses", [])
    running_styles = classified["courseanalysis"].get("running_styles", [])
    if not isinstance(entry_horses, list) or not entry_horses:
        raise NarJsonDataError("出走表JSONのhorsesが空です。")
    if not isinstance(speed_horses, list) or not speed_horses:
        raise NarJsonDataError("タイム指数JSONのhorsesが空です。")
    if len(entry_horses) != len(speed_horses):
        raise NarJsonDataError(
            f"出走表とタイム指数の頭数が一致しません（出走表{len(entry_horses)}頭 / タイム指数{len(speed_horses)}頭）。"
        )

    entry_numbers = _number_list(entry_horses)
    speed_numbers = _number_list(speed_horses)
    if any(not number for number in entry_numbers + speed_numbers):
        raise NarJsonDataError("horse_numberが空の馬があります。")
    _raise_if_duplicate(entry_numbers, "出走表")
    _raise_if_duplicate(speed_numbers, "タイム指数")
    if set(entry_numbers) != set(speed_numbers):
        missing_speed = sorted(set(entry_numbers) - set(speed_numbers), key=_number_sort_key)
        missing_entry = sorted(set(speed_numbers) - set(entry_numbers), key=_number_sort_key)
        detail = []
        if missing_speed:
            detail.append("タイム指数にない馬番: " + ", ".join(missing_speed))
        if missing_entry:
            detail.append("出走表にない馬番: " + ", ".join(missing_entry))
        raise NarJsonDataError("出走表とタイム指数の馬番が一致しません。\n" + "\n".join(detail))

    _validate_horse_identity(entry_horses, speed_horses)

    if not isinstance(running_styles, list) or not running_styles:
        raise NarJsonDataError("コース脚質データのrunning_stylesが空です。")
    has_invalid_style = False
    for item in running_styles:
        if not isinstance(item, dict) or not str(item.get("style", "")).strip() or item.get("win_rate") is None:
            has_invalid_style = True
            break
    if has_invalid_style:
        raise NarJsonDataError("コース脚質データにstyleまたはwin_rateが不足しています。")


def validate_nar_json_bundle(classified: dict[str, dict[str, Any]]) -> None:
    validate_nar_uploaded_data(classified)


def merge_entry_and_speed(
    entry_data: dict[str, Any],
    speed_data: dict[str, Any],
    courseanalysis_data: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    speed_by_number = {
        _horse_number(horse): horse
        for horse in speed_data.get("horses", [])
        if _horse_number(horse)
    }
    course_style_by_number = _horse_running_style_map(courseanalysis_data or {})
    merged_horses: list[dict[str, Any]] = []
    for entry_horse in entry_data.get("horses", []):
        horse_number = _horse_number(entry_horse)
        speed_horse = speed_by_number.get(horse_number)
        if not speed_horse:
            raise NarJsonDataError(f"馬番{horse_number}のタイム指数が見つかりません。")
        merged = dict(entry_horse)
        for key in ("max", "avg5", "distance", "course", "race3", "race2", "race1"):
            merged[key] = parse_speed_index(speed_horse.get(key))
        for key in ("odds", "popularity", "style", "running_style", "jockey"):
            if not _safe_text(merged.get(key)) and _safe_text(speed_horse.get(key)):
                merged[key] = speed_horse.get(key)
        running_style = (
            course_style_by_number.get(horse_number)
            or _extract_horse_running_style(entry_horse)
            or _extract_horse_running_style(speed_horse)
        )
        if running_style:
            merged["running_style"] = running_style
            if not _safe_text(merged.get("style")):
                merged["style"] = running_style
        _apply_display_detail_fields(merged, entry_horse, speed_horse)
        weight_value, weight_diff = parse_horse_weight(merged.get("horse_weight"))
        merged["horse_weight_value"] = weight_value
        merged["horse_weight_diff"] = weight_diff
        merged["_speed_horse"] = speed_horse
        merged_horses.append(merged)
    return merged_horses


def _apply_display_detail_fields(merged: dict[str, Any], *sources: dict[str, Any]) -> None:
    """Carry shortcut-supplied previous weight/jockey data as display-only fields."""

    all_sources = (merged, *sources)
    current_weight = _first_horse_text(all_sources, CURRENT_LOAD_WEIGHT_KEYS)
    previous_weight = _first_horse_text(all_sources, PREVIOUS_LOAD_WEIGHT_KEYS)
    explicit_change = _first_horse_text(all_sources, LOAD_WEIGHT_CHANGE_KEYS)
    current_jockey = _first_horse_text(all_sources, CURRENT_JOCKEY_KEYS)
    previous_jockey = _first_horse_text(all_sources, PREVIOUS_JOCKEY_KEYS)
    explicit_jockey_changed = _first_horse_text(all_sources, JOCKEY_CHANGED_KEYS)

    current_weight_number = _parse_float_text(current_weight)
    previous_weight_number = _parse_float_text(previous_weight)
    change_number = _parse_float_text(explicit_change)
    if change_number is None and current_weight_number is not None and previous_weight_number is not None:
        change_number = current_weight_number - previous_weight_number

    if current_weight_number is not None:
        merged["_display_current_load_weight"] = current_weight_number
    if previous_weight_number is not None:
        merged["_display_previous_load_weight"] = previous_weight_number
    if change_number is not None:
        merged["_display_load_weight_change"] = round(change_number, 3)
    if current_jockey:
        merged["_display_current_jockey"] = current_jockey
    if previous_jockey:
        merged["_display_previous_jockey"] = previous_jockey
    if current_jockey and previous_jockey:
        merged["_display_jockey_changed"] = _nar_jockey_changed_value(current_jockey, previous_jockey)
    elif explicit_jockey_changed:
        merged["_display_jockey_changed"] = _parse_bool_text(explicit_jockey_changed)


def _display_detail_attrs(horse: dict[str, Any]) -> str:
    attrs = []
    for key, attr_name in (
        ("_display_current_load_weight", "current-load-weight"),
        ("_display_previous_load_weight", "previous-load-weight"),
        ("_display_load_weight_change", "load-weight-change"),
        ("_display_current_jockey", "current-jockey"),
        ("_display_previous_jockey", "previous-jockey"),
        ("_display_jockey_changed", "jockey-changed"),
    ):
        if key not in horse:
            continue
        value = horse.get(key)
        if value is None:
            continue
        text = _safe_text(value)
        if text:
            attrs.append(f' data-display-{attr_name}="{_e(text)}"')
    return "".join(attrs)


def _build_nar_previous_jockey_debug_logs(
    classified: dict[str, dict[str, Any]],
    entry_data: dict[str, Any],
    merged_horses: list[dict[str, Any]],
    html_files: dict[str, str],
) -> list[dict[str, Any]]:
    newspaper = classified.get("newspaper") or {}
    newspaper_by_number = {
        _horse_number(horse): horse
        for horse in newspaper.get("horses", [])
        if isinstance(horse, dict) and _horse_number(horse)
    }
    entry_by_number = {
        _horse_number(horse): horse
        for horse in entry_data.get("horses", [])
        if isinstance(horse, dict) and _horse_number(horse)
    }
    merged_by_number = {
        _horse_number(horse): horse
        for horse in merged_horses
        if isinstance(horse, dict) and _horse_number(horse)
    }
    speed_attrs_by_number = _extract_speed_previous_jockey_attrs(html_files.get("speed", ""))
    numbers = sorted(
        set(newspaper_by_number) | set(entry_by_number) | set(merged_by_number) | set(speed_attrs_by_number),
        key=_number_sort_key,
    )
    rows: list[dict[str, Any]] = []
    for number in numbers:
        newspaper_horse = newspaper_by_number.get(number, {})
        entry_horse = entry_by_number.get(number, {})
        merged_horse = merged_by_number.get(number, {})
        speed_attrs = speed_attrs_by_number.get(number, {})
        rows.append(
            {
                "horse_number": number,
                "horse_name": (
                    _safe_text(merged_horse.get("horse_name"))
                    or _safe_text(entry_horse.get("horse_name"))
                    or _safe_text(newspaper_horse.get("horse_name"))
                ),
                "raw_previous_jockey": _safe_text(newspaper_horse.get("_debug_previous_jockey_raw")),
                "normalized_previous_jockey": _safe_text(newspaper_horse.get("_debug_previous_jockey_normalized")),
                "entry_prev_jockey": _first_horse_text((entry_horse,), PREVIOUS_JOCKEY_KEYS),
                "merged_prev_jockey": _safe_text(merged_horse.get("_display_previous_jockey")),
                "speed_html_previous_jockey": _safe_text(speed_attrs.get("previous_jockey")),
                "current_jockey": (
                    _safe_text(merged_horse.get("_display_current_jockey"))
                    or _safe_text(merged_horse.get("jockey"))
                    or _safe_text(speed_attrs.get("current_jockey"))
                ),
                "jockey_changed": _safe_text(merged_horse.get("_display_jockey_changed")),
            }
        )
    return rows


def _extract_speed_previous_jockey_attrs(html_text: str) -> dict[str, dict[str, str]]:
    attrs_by_number: dict[str, dict[str, str]] = {}
    for match in re.finditer(r"<tr\b(?P<attrs>[^>]*)>(?P<body>[\s\S]*?)</tr>", str(html_text or ""), flags=re.I):
        attrs = match.group("attrs") or ""
        body = match.group("body") or ""
        if "data-display-" not in attrs:
            continue
        number_match = re.search(
            r'class=["\'][^"\']*(?:Speed_List01|UmaBan)[^"\']*["\'][^>]*>\s*(\d{1,2})',
            body,
            flags=re.I,
        )
        if not number_match:
            number_match = re.search(r">\s*(\d{1,2})\s*</", body)
        if not number_match:
            continue
        number = number_match.group(1)
        attrs_by_number[number] = {
            "current_jockey": _extract_generated_attr(attrs, "current-jockey"),
            "previous_jockey": _extract_generated_attr(attrs, "previous-jockey"),
            "jockey_changed": _extract_generated_attr(attrs, "jockey-changed"),
        }
    return attrs_by_number


def _extract_generated_attr(attrs: str, name: str) -> str:
    match = re.search(rf'data-display-{re.escape(name)}=["\']([^"\']*)["\']', attrs or "", flags=re.I)
    return unescape(match.group(1)).strip() if match else ""


def _first_horse_text(sources: tuple[dict[str, Any], ...], keys: tuple[str, ...]) -> str:
    for source in sources:
        for key in keys:
            value = _safe_text(source.get(key))
            if value:
                return value
    return ""


def _parse_float_text(value: Any) -> float | None:
    text = _safe_text(value).replace("＋", "+").replace("－", "-")
    if not text:
        return None
    match = re.search(r"[+-]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except ValueError:
        return None


def _parse_bool_text(value: Any) -> bool:
    text = _safe_text(value).lower()
    return text in {"true", "1", "yes", "y", "○", "あり", "替", "乗り替わり", "乗替"}


def _normalize_jockey_for_compare(value: Any) -> str:
    text = unicodedata.normalize("NFKC", _safe_text(value))
    text = re.sub(r"[\(（]\s*替\s*[\)）]", "", text)
    text = re.sub(r"[\(（][^\)）]{1,4}所属[\)）]", "", text)
    text = re.sub(r"^[一-龥ぁ-んァ-ン]{1,4}[・･]", "", text)
    text = re.sub(r"^(?:替|乗替|乗り替わり|初騎乗)", "", text)
    text = text.replace("騎手", "")
    text = re.sub(r"^[▲△☆★◇◆▽▼]+", "", text)
    return re.sub(r"\s+", "", text)


def _same_nar_jockey_name(current: Any, previous: Any) -> bool | None:
    current_text = _normalize_jockey_for_compare(current)
    previous_text = _normalize_jockey_for_compare(previous)
    if not current_text or not previous_text:
        return None
    if current_text == previous_text:
        return True

    short, long = (
        (current_text, previous_text)
        if len(current_text) <= len(previous_text)
        else (previous_text, current_text)
    )
    if long.startswith(short):
        diff = len(long) - len(short)
        if len(short) >= 3 and 1 <= diff <= 2:
            return True
        return None
    return False


def _nar_jockey_changed_value(current: Any, previous: Any) -> bool | str | None:
    same = _same_nar_jockey_name(current, previous)
    if same is None:
        return "pending" if _safe_text(current) and _safe_text(previous) else None
    return not same


def parse_index(value: Any) -> int | None:
    if _is_missing(value):
        return None
    text = str(value).strip().replace("*", "")
    if text in {"", "-", "未", "未取得", "None", "none", "nan", "<NA>", "NaT"}:
        return None
    try:
        return int(float(text))
    except ValueError:
        return None


def parse_speed_index(value: Any) -> int | None:
    parsed = parse_index(value)
    text = _safe_text(value).replace("*", "")
    # netkeiba's saved HTML uses hidden sort value 100 for missing speed cells.
    # Some Shortcut JSON captures that hidden value instead of the displayed "-"/"未".
    if parsed == 100 and text in {"100", "100.0"}:
        return None
    return parsed


def parse_horse_weight(value: Any) -> tuple[int | None, int | None]:
    text = _safe_text(value)
    match = re.search(r"(\d+)\s*\(([+-]?\d+)\)", text)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def build_speed_html(
    speed_data: dict[str, Any],
    entry_data: dict[str, Any],
    merged_horses: list[dict[str, Any]],
) -> str:
    race_id = str(speed_data.get("race_id") or entry_data.get("race_id") or "").strip()
    race = _race_dict(speed_data, entry_data)
    race_name = _race_value(race, "race_name") or "地方競馬"
    race_data_1 = _race_value(race, "race_data_1", "race_data", "race_info") or ""
    race_data_2 = _race_value(race, "race_data_2") or ""
    rows = []
    for horse in merged_horses:
        detail_attrs = _display_detail_attrs(horse)
        rows.append(
            f'<tr class="List HorseList"{detail_attrs}>'
            f"<td>{_e(horse.get('frame_number'))}</td>"
            f'<td><span class="Speed_List01 UmaBan">{_e(horse.get("horse_number"))}</span></td>'
            f'<td><span class="Horse_Name"><a href="https://nar.netkeiba.com/horse/{_e(horse.get("horse_id"))}">{_e(horse.get("horse_name"))}</a></span></td>'
            "<td></td>"
            f"<td>{_e(horse.get('sex_age'))}</td>"
            f'<td><span class="Speed_List02">{_e(horse.get("weight"))}</span></td>'
            f'<td><span class="Jockey">{_e(horse.get("jockey"))}</span></td>'
            f'<td class="Speed_List07 Odds">{_e(horse.get("odds"))}</td>'
            f'<td class="Speed_List08 Ninki">{_e(horse.get("popularity"))}</td>'
            f'<td class="Speed_List03 MaxIndex"><a>{_index_text(horse.get("max"))}</a></td>'
            f'<td class="Speed_List04 Avg5Index">{_index_text(horse.get("avg5"))}</td>'
            f'<td class="Speed_List05">{_index_text(horse.get("distance"))}</td>'
            f'<td class="Speed_List06">{_index_text(horse.get("course"))}</td>'
            f'{_speed_index_td("Speed_List09", horse, "race3", "3走前")}'
            f'{_speed_index_td("Speed_List10", horse, "race2", "2走前")}'
            f'{_speed_index_td("Speed_List11", horse, "race1", "前走")}'
            "</tr>"
        )
    _build_star_trace_from_merged_horses("02 build_speed_html", merged_horses)
    return _html_document(
        race_id,
        race_name,
        race_data_1,
        race_data_2,
        "speed",
        f'<div id="Speed_List"><table class="SpeedIndex_Table"><tbody>{"".join(rows)}</tbody></table></div>',
    )


def _speed_index_td(class_name: str, horse: dict[str, Any], run_key: str, label: str) -> str:
    return f'<td class="{_e(class_name)}"{_past_run_condition_attrs(horse, run_key, label)}>{_index_text(horse.get(run_key))}</td>'


def _past_run_condition_attrs(horse: dict[str, Any], run_key: str, label: str) -> str:
    run = _past_run_condition_dict(horse, run_key, label)
    attrs = []
    for key, attr_name in (
        ("racecourse", "venue"),
        ("surface", "surface"),
        ("distance", "distance"),
        ("direction", "turn"),
        ("condition", "condition"),
    ):
        value = _safe_text(run.get(key))
        if value:
            attrs.append(f' data-star-{attr_name}="{_e(value)}"')
    return "".join(attrs)


def _past_run_condition_dict(horse: dict[str, Any], run_key: str, label: str) -> dict[str, Any]:
    speed_horse = horse.get("_speed_horse") if isinstance(horse.get("_speed_horse"), dict) else {}
    sources = [speed_horse, horse]
    nested = _find_nested_past_run(sources, run_key, label)
    result: dict[str, Any] = {}
    for source in [nested, *sources]:
        if not isinstance(source, dict):
            continue
        generic_keys = bool(source is nested and source)
        result["racecourse"] = result.get("racecourse") or _first_value(
            source,
            (
                f"{run_key}_racecourse",
                f"{run_key}_venue",
                f"{run_key}_track",
                f"{run_key}_競馬場",
                *(( "racecourse", "venue", "track", "競馬場") if generic_keys else ()),
            ),
        )
        result["surface"] = result.get("surface") or _first_value(
            source,
            (
                f"{run_key}_surface",
                f"{run_key}_芝ダ",
                f"{run_key}_course_type",
                *(("surface", "芝ダ", "course_type") if generic_keys else ()),
            ),
        )
        result["distance"] = result.get("distance") or _first_value(
            source,
            (
                f"{run_key}_distance",
                f"{run_key}_距離",
                *(("distance", "距離") if generic_keys else ()),
            ),
        )
        result["direction"] = result.get("direction") or _first_value(
            source,
            (
                f"{run_key}_turn",
                f"{run_key}_direction",
                f"{run_key}_回り",
                *(("turn", "direction", "回り") if generic_keys else ()),
            ),
        )
        result["condition"] = result.get("condition") or _first_value(
            source,
            (
                f"{run_key}_condition",
                f"{run_key}_条件",
                *(("condition", "条件", "course_label") if generic_keys else ()),
            ),
        )
    return result


def _find_nested_past_run(sources: list[dict[str, Any]], run_key: str, label: str) -> dict[str, Any]:
    index_by_key = {"race3": 0, "race2": 1, "race1": 2}
    target_index = index_by_key.get(run_key)
    for source in sources:
        if not isinstance(source, dict):
            continue
        for list_key in ("past_runs", "recent_runs", "runs", "races", "history"):
            value = source.get(list_key)
            if not isinstance(value, list):
                continue
            candidates = [item for item in value if isinstance(item, dict)]
            for item in candidates:
                item_label = _safe_text(item.get("label") or item.get("race_label") or item.get("走"))
                item_key = _safe_text(item.get("key") or item.get("run_key"))
                if item_key == run_key or item_label == label:
                    return item
            if target_index is not None and len(candidates) > target_index:
                return candidates[target_index]
    return {}


def _build_star_trace_from_speed_json(speed_data: dict[str, Any]) -> list[dict[str, Any]]:
    horses = speed_data.get("horses", []) if isinstance(speed_data, dict) else []
    rows = []
    for horse in horses:
        if not isinstance(horse, dict):
            continue
        rows.append(
            star_trace_row(
                horse_no=_horse_number(horse),
                horse_name=horse.get("horse_name"),
                year_max_index=parse_speed_index(horse.get("max")),
                star_max_index=_first_value(
                    horse,
                    (
                        "star_max_index",
                        "star_max",
                        "same_condition_max",
                        "same_condition_high",
                    ),
                ),
                star_candidates=candidate_summary(_star_candidate_runs_from_horse(horse)),
            )
        )
    return log_star_trace("01 nar_json_input JSON loaded", rows)


def _build_star_trace_from_merged_horses(stage: str, horses: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for horse in horses:
        rows.append(
            star_trace_row(
                horse_no=horse.get("horse_number"),
                horse_name=horse.get("horse_name"),
                year_max_index=horse.get("max"),
                star_max_index=horse.get("star_max_index"),
                star_candidates=candidate_summary(_star_candidate_runs_from_horse(horse)),
            )
        )
    return log_star_trace(stage, rows)


def _star_candidate_runs_from_horse(horse: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        {
            "label": label,
            "value": horse.get(run_key),
            **_past_run_condition_dict(horse, run_key, label),
        }
        for run_key, label in (("race3", "3back"), ("race2", "2back"), ("race1", "last"))
    ]


def _extract_speed_star_trace(html_text: str, stage: str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for match in re.finditer(r"<tr\b(?P<attrs>[^>]*)>(?P<body>[\s\S]*?)</tr>", str(html_text or ""), flags=re.I):
        body = match.group("body") or ""
        number_match = re.search(
            r'class=["\'][^"\']*(?:Speed_List01|UmaBan)[^"\']*["\'][^>]*>\s*(\d{1,2})',
            body,
            flags=re.I,
        )
        name_match = re.search(
            r'class=["\'][^"\']*Horse_Name[^"\']*["\'][^>]*>[\s\S]*?<a[^>]*>([\s\S]*?)</a>',
            body,
            flags=re.I,
        )
        year_match = re.search(
            r'class=["\'][^"\']*(?:Speed_List03|MaxIndex)[^"\']*["\'][^>]*>[\s\S]*?([+-]?\d+(?:\.\d+)?)',
            body,
            flags=re.I,
        )
        star_attrs = []
        for cell in re.findall(r"<td\b[^>]*data-star-[^>]*>[\s\S]*?</td>", body, flags=re.I):
            value_text = _clean_html_text(cell)
            attrs = re.search(r"<td\b([^>]*)>", cell, flags=re.I)
            attr_text = attrs.group(1) if attrs else ""
            star_attrs.append(
                {
                    "label": _extract_star_attr(attr_text, "label") or "cell",
                    "value": value_text,
                    "racecourse": _extract_star_attr(attr_text, "venue"),
                    "surface": _extract_star_attr(attr_text, "surface"),
                    "distance": _extract_star_attr(attr_text, "distance"),
                    "direction": _extract_star_attr(attr_text, "turn"),
                }
            )
        rows.append(
            star_trace_row(
                horse_no=number_match.group(1) if number_match else "",
                horse_name=_clean_html_text(name_match.group(1)) if name_match else "",
                year_max_index=year_match.group(1) if year_match else None,
                star_max_index=None,
                star_candidates=candidate_summary(star_attrs),
            )
        )
    return log_star_trace(stage, rows)


def _extract_star_attr(attrs: str, name: str) -> str:
    match = re.search(rf'data-star-{re.escape(name)}=["\']([^"\']*)["\']', attrs or "", flags=re.I)
    return unescape(match.group(1)).strip() if match else ""


def _clean_html_text(source: str) -> str:
    text = re.sub(r"<[^>]+>", " ", source or "")
    return re.sub(r"\s+", " ", unescape(text)).strip()


def _first_value(source: dict[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = source.get(key)
        if _safe_text(value):
            return value
    return ""


def build_shutuba_html(entry_data: dict[str, Any], merged_horses: list[dict[str, Any]]) -> str:
    race_id = str(entry_data.get("race_id", "")).strip()
    race = _race_dict(entry_data)
    race_name = _race_value(race, "race_name") or "地方競馬"
    race_data_1 = _race_value(race, "race_data_1", "race_data", "race_info") or ""
    race_data_2 = _race_value(race, "race_data_2") or ""
    rows = []
    for horse in merged_horses:
        rows.append(
            '<tr class="HorseList">'
            f'<td class="Waku">{_e(horse.get("frame_number"))}</td>'
            f'<td class="Umaban Horse_Num HorseNum Num HorseList_Num">{_e(horse.get("horse_number"))}</td>'
            f'<td class="Horse_Info"><a href="https://nar.netkeiba.com/horse/{_e(horse.get("horse_id"))}">{_e(horse.get("horse_name"))}</a></td>'
            f"<td>{_e(horse.get('sex_age'))}</td>"
            f"<td>{_e(horse.get('weight'))}</td>"
            f"<td>{_e(horse.get('jockey'))}</td>"
            f'<td class="Weight HorseWeight Horse_Weight">{_e(horse.get("horse_weight"))}</td>'
            "</tr>"
        )
    return _html_document(
        race_id,
        race_name,
        race_data_1,
        race_data_2,
        "shutuba",
        f'<table class="Shutuba_Table"><tbody>{"".join(rows)}</tbody></table>',
    )


def build_courseanalysis_html(
    courseanalysis_data: dict[str, Any],
    merged_horses: list[dict[str, Any]],
) -> str:
    race_id = str(courseanalysis_data.get("race_id", "")).strip()
    race = _race_dict(courseanalysis_data)
    race_name = _race_value(race, "race_name") or "地方競馬"
    race_data_1 = _race_value(race, "race_data_1", "race_data", "race_info") or ""
    race_data_2 = _race_value(race, "race_data_2") or ""
    style_stats = _course_style_stats(courseanalysis_data)

    horse_rows = []
    for horse in merged_horses:
        style = _safe_text(horse.get("running_style")) or _safe_text(horse.get("style"))
        if not style:
            continue
        stats = style_stats.get(_normalize_style(style), {})
        horse_rows.append(
            '<tr class="HorseList">'
            f"<td>{_e(horse.get('horse_number'))}</td>"
            f'<td class="Horse_Info"><a>{_e(horse.get("horse_name"))}</a></td>'
            f'<td class="DataTitle_Cell">{_e(style)}</td>'
            "<td></td><td></td><td></td><td></td><td></td>"
            f"<td>{_e(stats.get('win_rate'))}</td>"
            f"<td>{_e(stats.get('quinella_rate'))}</td>"
            f"<td>{_e(stats.get('place_rate'))}</td>"
            "<td></td><td></td>"
            "</tr>"
        )

    trend_rows = []
    for item in courseanalysis_data.get("running_styles", []):
        trend_rows.append(
            "<tr>"
            f"<td>{_e(item.get('style'))}</td>"
            f"<td>{_e(item.get('win_rate'))}</td>"
            f"<td>{_e(item.get('quinella_rate'))}</td>"
            f"<td>{_e(item.get('place_rate'))}</td>"
            f"<td>{_e(item.get('outside_rate'))}</td>"
            "</tr>"
        )

    log_star_trace(
        "03 build_courseanalysis_html",
        [
            star_trace_row(
                horse_no=horse.get("horse_number"),
                horse_name=horse.get("horse_name"),
                year_max_index=horse.get("max"),
                star_max_index=None,
                running_style=horse.get("running_style") or horse.get("style"),
            )
            for horse in merged_horses
        ],
    )

    body = (
        f'<canvas id="score1"></canvas>'
        f'<table id="table_sort_back" class="Data01_Table"><tbody>{"".join(horse_rows)}</tbody></table>'
        f'<table class="CourseAnalysis"><tbody>{"".join(trend_rows)}</tbody></table>'
    )
    return _html_document(race_id, race_name, race_data_1, race_data_2, "courseanalysis", body)


def _html_document(
    race_id: str,
    race_name: str,
    race_data_1: str,
    race_data_2: str,
    page_kind: str,
    body: str,
) -> str:
    canonical = {
        "speed": f"https://nar.netkeiba.com/race/speed.html?race_id={race_id}",
        "shutuba": f"https://nar.netkeiba.com/race/shutuba.html?race_id={race_id}",
        "courseanalysis": f"https://nar.netkeiba.com/race/data_list.html?race_id={race_id}&mode=courseanalysis&cid=1",
    }.get(page_kind, f"https://nar.netkeiba.com/race/shutuba.html?race_id={race_id}")
    return (
        "<!doctype html><html><head>"
        f"<title>{_e(race_name)}</title>"
        f'<link rel="canonical" href="{_e(canonical)}">'
        f'<meta property="og:url" content="{_e(canonical)}">'
        "</head>"
        '<body id="Netkeiba_Race_Nar_Shutuba">'
        f'<a href="{_e(canonical)}">race_id={_e(race_id)}</a>'
        f'<h1 class="RaceName">{_e(race_name)}</h1>'
        f'<div class="RaceData01">{_e(race_data_1)}</div>'
        f'<div class="RaceData02">{_e(race_data_2)}</div>'
        f"{body}</body></html>"
    )


def _suggested_name(data: dict[str, Any], race_id: str, fallback_type: str) -> str:
    return (
        _safe_text(data.get("suggested_file_name"))
        or _safe_text(data.get("_uploaded_file_name"))
        or f"{race_id}_{fallback_type}.html"
    )


def _race_dict(*sources: dict[str, Any]) -> dict[str, Any]:
    for source in sources:
        race = source.get("race")
        if isinstance(race, dict):
            return race
    return {}


def _race_value(race: dict[str, Any], *keys: str) -> str:
    for key in keys:
        value = _safe_text(race.get(key))
        if value:
            return value
    return ""


def _horse_number(horse: dict[str, Any]) -> str:
    return _safe_text(horse.get("horse_number"))


def _number_list(horses: list[dict[str, Any]]) -> list[str]:
    return [_horse_number(horse) for horse in horses]


def _raise_if_duplicate(numbers: list[str], label: str) -> None:
    duplicates = sorted({number for number in numbers if numbers.count(number) > 1}, key=_number_sort_key)
    if duplicates:
        raise NarJsonDataError(f"{label}に重複馬番があります: {', '.join(duplicates)}")


def _validate_horse_identity(entry_horses: list[dict[str, Any]], speed_horses: list[dict[str, Any]]) -> None:
    speed_by_number = {_horse_number(horse): horse for horse in speed_horses}
    mismatches: list[str] = []
    for entry_horse in entry_horses:
        number = _horse_number(entry_horse)
        speed_horse = speed_by_number.get(number, {})
        entry_id = _safe_text(entry_horse.get("horse_id"))
        speed_id = _safe_text(speed_horse.get("horse_id"))
        if entry_id and speed_id and entry_id != speed_id:
            mismatches.append(f"馬番{number}: horse_id {entry_id} / {speed_id}")
        entry_name = _normalize_name(entry_horse.get("horse_name"))
        speed_name = _normalize_name(speed_horse.get("horse_name"))
        if entry_name and speed_name and entry_name != speed_name:
            mismatches.append(f"馬番{number}: 馬名 {entry_horse.get('horse_name')} / {speed_horse.get('horse_name')}")
    if mismatches:
        raise NarJsonDataError("出走表とタイム指数で馬情報が一致しません。\n" + "\n".join(mismatches[:8]))


def _horse_running_style_map(courseanalysis_data: dict[str, Any]) -> dict[str, str]:
    style_by_number: dict[str, str] = {}
    candidates = []
    for key in ("horse_running_styles", "horses"):
        value = courseanalysis_data.get(key)
        if isinstance(value, list):
            candidates.extend(item for item in value if isinstance(item, dict))
    for item in candidates:
        number = _horse_number(item)
        style = _extract_horse_running_style(item)
        if number and style:
            style_by_number[number] = style
    return style_by_number


def _extract_horse_running_style(*sources: dict[str, Any]) -> str:
    keys = (
        "running_style",
        "style",
        "脚質",
        "kyakushitsu",
        "running_style_name",
        "running_style_label",
    )
    for source in sources:
        for key in keys:
            value = _safe_text(source.get(key))
            if value:
                normalized = _normalize_style(value)
                return normalized or value
    return ""


def _course_style_stats(courseanalysis_data: dict[str, Any]) -> dict[str, dict[str, Any]]:
    stats: dict[str, dict[str, Any]] = {}
    for item in courseanalysis_data.get("running_styles", []):
        if not isinstance(item, dict):
            continue
        style = _normalize_style(item.get("style"))
        if style:
            stats[style] = item
    return stats


def _normalize_style(value: Any) -> str:
    text = _safe_text(value)
    if "逃" in text:
        return "逃"
    if "先" in text:
        return "先"
    if "差" in text:
        return "差"
    if "追" in text:
        return "追"
    return ""


def _normalize_name(value: Any) -> str:
    return re.sub(r"\s+", "", _safe_text(value))


def _number_sort_key(value: str) -> tuple[int, str]:
    try:
        return int(value), value
    except ValueError:
        return 999, value


def _index_text(value: Any) -> str:
    parsed = parse_index(value)
    return "" if parsed is None else str(parsed)


def _type_label(data_type: str) -> str:
    return {
        "entry": "出走表JSON",
        "speed": "タイム指数JSON",
        "courseanalysis": "コース脚質HTML/JSON",
    }.get(data_type, data_type)


def _e(value: Any) -> str:
    return escape(_safe_text(value), quote=True)


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        missing = pd.isna(value)
        try:
            if bool(missing):
                return True
        except (TypeError, ValueError):
            pass
    except (TypeError, ValueError):
        pass
    return str(value).strip() in {"", "None", "none", "nan", "NaN", "<NA>", "NaT"}


def _safe_text(value: Any) -> str:
    return "" if _is_missing(value) else str(value).strip()

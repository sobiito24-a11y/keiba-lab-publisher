# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from copy import deepcopy
from collections.abc import Mapping
from typing import Any

import pandas as pd

from .purchase_conditions import clean_text, horse_no, to_float
from .recent_races import build_recent_races


CONDITION_FIT_LABELS = {
    "same_venue_distance": ("★", "同会場距離"),
    "same_turn_distance": ("☆", "同回り距離"),
    "same_distance": ("※", "同距離"),
    "none": ("", "条件実績なし"),
}
CONDITION_FIT_PRIORITY = ("same_venue_distance", "same_turn_distance", "same_distance")
CONDITION_FIT_LEVEL_ALIASES = {
    "same_venue_distance": "same_venue_distance",
    "same_turn_distance": "same_turn_distance",
    "same_distance": "same_distance",
    "none": "none",
    # Legacy star_match_level values all prove the same venue + distance.
    # They do not encode the Ver4 ☆/※ concepts and must not be mapped to them.
    "venue_distance": "same_venue_distance",
    "venue_distance_surface": "same_venue_distance",
    "venue_distance_surface_turn": "same_venue_distance",
}


def canonical_condition_fit_level(value: Any, mark: Any = "") -> str | None:
    """Translate only condition levels whose semantics are explicitly known."""

    text = clean_text(value).lower()
    if text in CONDITION_FIT_LEVEL_ALIASES:
        return CONDITION_FIT_LEVEL_ALIASES[text]
    if not text:
        return {"★": "same_venue_distance", "☆": "same_turn_distance", "※": "same_distance"}.get(
            clean_text(mark)
        )
    return None


def extract_condition_fit_sources(table: Any) -> dict[str, dict[str, Any]]:
    """Copy result_df-only condition facts into an immutable horse-number map.

    The returned mapping is suitable for ``PredictionResult.debug_info``.  It
    deliberately contains no result/payoff fields and does not mutate the
    source DataFrame.
    """

    if table is None or not isinstance(table, pd.DataFrame) or table.empty:
        return {}
    source_fields = (
        "_past_runs",
        "recent_runs",
        "past_runs",
        "condition_fit_mark",
        "condition_fit_level",
        "condition_fit_reason",
        "condition_fit_data_status",
        "matched_past_runs",
        "star_match_level",
    )
    result: dict[str, dict[str, Any]] = {}
    for _, raw in table.iterrows():
        record = raw.to_dict()
        key = horse_no(_first_present(record, ("馬番", "horse_no", "horse_number", "馬")))
        if not key:
            continue
        values = {
            name: deepcopy(record.get(name))
            for name in source_fields
            if name in record and not _missing(record.get(name))
        }
        if values:
            result[key] = values
    return result


def evaluate_condition_fit(
    row: Mapping[str, Any],
    race_info: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return direct-condition experience labels without changing prediction scores.

    This is display/audit-only.  It uses existing recent-race records and the
    current race metadata that the prediction result already carries.
    """

    current = _current_condition(row, race_info)
    runs = build_recent_races(row)
    matched_by_level: dict[str, list[dict[str, Any]]] = {
        "same_venue_distance": [],
        "same_turn_distance": [],
        "same_distance": [],
    }
    current_distance = current.get("distance")
    for run in runs:
        run_distance = _distance_int(run.get("distance"))
        if current_distance is None or run_distance is None or current_distance != run_distance:
            continue
        run_record = _matched_run_record(run)
        current_venue = _venue_key(current.get("venue"))
        run_venue = _venue_key(run.get("venue"))
        current_turn = _turn_key(current.get("turn"))
        run_turn = _turn_key(run.get("turn"))
        same_venue = bool(current_venue and run_venue and current_venue == run_venue)
        same_turn = bool(current_turn and run_turn and current_turn == run_turn)
        if same_venue:
            matched_by_level["same_venue_distance"].append(run_record)
        elif same_turn:
            matched_by_level["same_turn_distance"].append(run_record)
        else:
            matched_by_level["same_distance"].append(run_record)

    level = "none"
    matched: list[dict[str, Any]] = []
    for candidate in CONDITION_FIT_PRIORITY:
        if matched_by_level[candidate]:
            level = candidate
            matched = matched_by_level[candidate]
            break

    if level != "none":
        data_status = "ok"
    elif _condition_source_is_complete(current_distance, runs):
        data_status = "no_match"
    else:
        data_status = "missing_source_data"

    mark, label = CONDITION_FIT_LABELS[level]
    reason = _reason(level, current, matched, data_status)
    return {
        "condition_fit_mark": mark or None,
        "condition_fit_level": level,
        "condition_fit_label": label,
        "condition_fit_reason": reason,
        "condition_fit_data_status": data_status,
        "matched_past_runs": matched,
        "current_condition": current,
    }


def resolved_condition_fit(
    row: Mapping[str, Any],
    race_info: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Prefer already-plumbed diagnostics, otherwise evaluate source runs."""

    mark = clean_text(row.get("condition_fit_mark"))
    level = canonical_condition_fit_level(row.get("condition_fit_level"), mark)
    status = clean_text(row.get("condition_fit_data_status"))
    if level is not None and status in {"ok", "no_match", "missing_source_data"}:
        return {
            "condition_fit_mark": mark or None,
            "condition_fit_level": level,
            "condition_fit_label": CONDITION_FIT_LABELS[level][1],
            "condition_fit_reason": clean_text(row.get("condition_fit_reason")),
            "condition_fit_data_status": status,
            "matched_past_runs": deepcopy(row.get("matched_past_runs"))
            if isinstance(row.get("matched_past_runs"), list)
            else [],
        }
    return evaluate_condition_fit(row, race_info)


def condition_fit_badge_text(
    row: Mapping[str, Any],
    race_info: Mapping[str, Any] | None = None,
) -> str:
    result = resolved_condition_fit(row, race_info)
    mark = clean_text(result.get("condition_fit_mark"))
    label = clean_text(result.get("condition_fit_label"))
    if mark:
        return f"{mark}{label}"
    return f"—{label}" if label else ""


def _current_condition(row: Mapping[str, Any], race_info: Mapping[str, Any] | None) -> dict[str, Any]:
    race_info = race_info or {}
    venue = _first(
        race_info,
        row,
        ("venue", "racecourse", "place", "競馬場", "開催場"),
        ("_racecourse", "_current_racecourse", "_current_venue", "racecourse", "venue", "競馬場", "開催場"),
    )
    distance = _distance_int(
        _first(
            race_info,
            row,
            ("distance", "距離"),
            ("_race_distance", "race_distance", "distance", "距離", "star_max_distance", "_star_max_distance"),
        )
    )
    surface = _first(
        race_info,
        row,
        ("surface", "track_type", "芝ダ", "馬場種別"),
        ("_race_surface", "_current_surface", "surface", "芝ダ", "馬場種別"),
    )
    turn = _first(
        race_info,
        row,
        ("turn", "direction", "回り"),
        ("_race_turn", "_current_turn", "turn", "direction", "回り", "star_max_turn", "_star_max_turn"),
    )
    return {
        "venue": clean_text(venue),
        "surface": clean_text(surface),
        "distance": distance,
        "turn": _turn_key(turn),
    }


def _first(
    race_info: Mapping[str, Any],
    row: Mapping[str, Any],
    info_names: tuple[str, ...],
    row_names: tuple[str, ...],
) -> Any:
    for name in info_names:
        value = race_info.get(name)
        if not _missing(value):
            return value
    for name in row_names:
        value = row.get(name)
        if not _missing(value):
            return value
    return ""


def _matched_run_record(run: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "label": clean_text(run.get("label")),
        "date": clean_text(run.get("date")),
        "venue": clean_text(run.get("venue")),
        "surface": clean_text(run.get("surface")),
        "distance": _distance_int(run.get("distance")),
        "direction": _turn_key(run.get("turn")),
        "race_name": clean_text(run.get("race_name")),
        "finish": clean_text(run.get("finish")),
        "popularity": clean_text(run.get("popularity")),
        "time_index": clean_text(run.get("time_index")),
        "passing_order": clean_text(run.get("passing_order")),
        "running_style": clean_text(run.get("running_style")),
    }


def _reason(
    level: str,
    current: Mapping[str, Any],
    matched: list[dict[str, Any]],
    data_status: str,
) -> str:
    if level == "none":
        if data_status == "missing_source_data":
            return "条件適性判定に必要な過去走条件データ不足"
        distance = f"{current.get('distance')}m" if current.get("distance") is not None else "今回距離"
        return f"{distance}の近3走実績なし"
    first = matched[0] if matched else {}
    distance = f"{current.get('distance')}m" if current.get("distance") is not None else ""
    if level == "same_venue_distance":
        venue = clean_text(first.get("venue") or current.get("venue"))
        return f"{' '.join(part for part in [venue, distance] if part)}の過去走あり"
    if level == "same_turn_distance":
        turn = clean_text(first.get("direction") or current.get("turn"))
        return f"{' '.join(part for part in [turn + '回り' if turn else '', distance] if part)}の過去走あり"
    return f"{distance}の過去走あり" if distance else "同距離の過去走あり"


def _distance_int(value: Any) -> int | None:
    if _missing(value):
        return None
    number = to_float(value)
    if number is not None:
        return int(number)
    match = re.search(r"\d{3,4}", clean_text(value).replace(",", ""))
    return int(match.group(0)) if match else None


def _venue_key(value: Any) -> str:
    text = clean_text(value)
    text = text.replace("競馬場", "").replace("レース場", "")
    return re.sub(r"\s+", "", text)


def _turn_key(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    if "右" in text:
        return "右"
    if "左" in text:
        return "左"
    if "直" in text:
        return "直"
    return text


def _condition_source_is_complete(current_distance: int | None, runs: list[dict[str, Any]]) -> bool:
    if current_distance is None or not runs:
        return False
    return all(_distance_int(run.get("distance")) is not None for run in runs)


def _first_present(row: Mapping[str, Any], names: tuple[str, ...]) -> Any:
    for name in names:
        if name in row and not _missing(row.get(name)):
            return row.get(name)
    return ""


def _missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, (list, tuple, dict)):
        return len(value) == 0
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return clean_text(value).lower() in {"", "-", "—", "nan", "none", "null", "データなし", "未取得"}

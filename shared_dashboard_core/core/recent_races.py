# -*- coding: utf-8 -*-
from __future__ import annotations

import ast
import json
import re
from collections.abc import Mapping
from typing import Any

import pandas as pd

from .purchase_conditions import clean_text, to_float


RUN_LABEL_ORDER = {
    "前走": 0,
    "蜑崎ｵｰ": 0,
    "last": 0,
    "race1": 0,
    "2走前": 1,
    "2襍ｰ蜑・": 1,
    "2back": 1,
    "two_back": 1,
    "race2": 1,
    "3走前": 2,
    "3襍ｰ蜑・": 2,
    "3back": 2,
    "three_back": 2,
    "race3": 2,
}


def build_recent_races(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Return existing recent-race data as a stable display/history structure."""

    flattened = _flattened_records(row)
    nested = _nested_runs(row)
    if nested:
        records = [_normalize_nested_run(run) for run in nested if isinstance(run, Mapping)]
        records = [_merge_run_material(record, flattened) for record in records]
        records = [record for record in records if _has_material(record)]
        records.sort(key=lambda record: RUN_LABEL_ORDER.get(clean_text(record.get("label")), 99))
        return records[:3]

    return flattened


def _flattened_records(row: Mapping[str, Any]) -> list[dict[str, Any]]:
    records = [
        _flattened_run(row, "前走", ["race1", "last"], ["前走", "蜑崎ｵｰ", "race1", "last_index"]),
        _flattened_run(row, "2走前", ["race2", "two_back"], ["2走前", "2襍ｰ蜑・", "race2", "two_back_index"]),
        _flattened_run(row, "3走前", ["race3", "three_back"], ["3走前", "3襍ｰ蜑・", "race3", "three_back_index"]),
    ]
    return [record for record in records if _has_material(record)]


def recent_race_preview_text(row: Mapping[str, Any]) -> str:
    runs = build_recent_races(row)
    if not runs:
        return ""
    latest = runs[0]
    condition = _condition_text(latest)
    result = _finish_popularity_text(latest)
    index_text = _index_text(latest)
    parts = [clean_text(latest.get("label")) or "前走", condition, result, index_text]
    return " ".join(part for part in parts if part)


def recent_races_summary_text(row: Mapping[str, Any]) -> str:
    runs = build_recent_races(row)
    if not runs:
        return ""
    lines: list[str] = []
    for index, run in enumerate(runs[:3]):
        label = clean_text(run.get("label")) or ("前走" if index == 0 else f"{index + 1}走前")
        parts = [
            clean_text(run.get("venue")),
            clean_text(run.get("surface")),
            clean_text(run.get("distance")),
            clean_text(run.get("finish")),
            _index_text(run),
        ]
        body = " ".join(part for part in parts if part)
        lines.append(f"{label}：{body or '-'}")
    return "\n".join(lines)


def recent_races_detail_text(row: Mapping[str, Any]) -> str:
    runs = build_recent_races(row)
    if not runs:
        return "データなし"
    blocks: list[str] = []
    for run in runs:
        label = clean_text(run.get("label")) or "-"
        condition = _detail_condition_text(run) or "-"
        details = [
            _finish_popularity_text(run),
            _index_text(run),
            _passing_text(run),
            _running_style_text(run),
        ]
        detail_line = "｜".join(part for part in details if part) or "-"
        blocks.append("\n".join([label, condition, detail_line]))
    return "\n\n".join(blocks)


def _nested_runs(row: Mapping[str, Any]) -> list[Any]:
    for key in ("_past_runs", "past_runs", "recent_runs", "runs", "races", "history"):
        value = row.get(key)
        if _missing(value):
            continue
        parsed = _parse_maybe_serialized(value)
        if isinstance(parsed, list):
            return parsed
    return []


def _parse_maybe_serialized(value: Any) -> Any:
    if isinstance(value, list):
        return value
    if isinstance(value, tuple):
        return list(value)
    text = clean_text(value)
    if not text or not text.startswith("["):
        return value
    for loader in (json.loads, ast.literal_eval):
        try:
            parsed = loader(text)
        except Exception:
            continue
        if isinstance(parsed, list):
            return parsed
    return value


def _normalize_nested_run(run: Mapping[str, Any]) -> dict[str, Any]:
    label = _normalize_label(_pick(run, "label", "key", "run_key", "race_label"))
    return {
        "label": label,
        "date": _pick(run, "race_date", "date", "previous_date", "日付"),
        "venue": _pick(run, "racecourse", "venue", "track", "previous_track", "競馬場"),
        "surface": _pick(run, "surface", "芝ダ", "track_type"),
        "distance": _format_distance(_pick(run, "distance", "距離")),
        "turn": _pick(run, "direction", "turn", "回り"),
        "race_name": _pick(run, "race_name", "previous_race", "レース"),
        "finish": _format_finish(_pick(run, "position", "finish", "previous_finish", "着順")),
        "popularity": _format_popularity(_pick(run, "popularity", "ninki", "人気")),
        "time_index": _format_index(_pick(run, "value", "time_index", "index", "指数")),
        "passing_order": _pick(run, "passing_order", "passing", "通過順位", "通過"),
        "running_style": _pick(run, "running_style", "style", "脚質"),
    }


def _merge_run_material(record: dict[str, Any], flattened: list[dict[str, Any]]) -> dict[str, Any]:
    order = RUN_LABEL_ORDER.get(clean_text(record.get("label")), 99)
    fallback = next(
        (
            item
            for item in flattened
            if RUN_LABEL_ORDER.get(clean_text(item.get("label")), 99) == order
        ),
        {},
    )
    if not fallback:
        return record
    merged = dict(record)
    for key, value in fallback.items():
        if _missing(merged.get(key)) and not _missing(value):
            merged[key] = value
    return merged


def _flattened_run(
    row: Mapping[str, Any],
    label: str,
    prefixes: list[str],
    index_names: list[str],
) -> dict[str, Any]:
    japanese_prefix = label
    return {
        "label": label,
        "date": _pick_prefixed(row, prefixes, ["date", "race_date"], [f"{japanese_prefix}日付"]),
        "venue": _pick_prefixed(
            row,
            prefixes,
            ["venue", "racecourse", "track", "place"],
            [f"{japanese_prefix}競馬場"],
        ),
        "surface": _pick_prefixed(row, prefixes, ["surface"], [f"{japanese_prefix}芝ダ"]),
        "distance": _format_distance(
            _pick_prefixed(row, prefixes, ["distance"], [f"{japanese_prefix}距離"])
        ),
        "turn": _pick_prefixed(row, prefixes, ["turn", "direction"], [f"{japanese_prefix}回り"]),
        "race_name": _pick_prefixed(row, prefixes, ["race", "race_name"], [f"{japanese_prefix}レース"]),
        "finish": _format_finish(
            _pick_prefixed(row, prefixes, ["finish", "position"], [f"{japanese_prefix}着順"])
        ),
        "popularity": _format_popularity(
            _pick_prefixed(row, prefixes, ["popularity", "ninki"], [f"{japanese_prefix}人気"])
        ),
        "time_index": _format_index(_pick(row, *index_names)),
        "passing_order": _pick_prefixed(
            row,
            prefixes,
            ["passing_order", "passing"],
            [f"{japanese_prefix}通過順位", f"{japanese_prefix}通過"],
        ),
        "running_style": _pick_prefixed(row, prefixes, ["running_style", "style"], [f"{japanese_prefix}脚質"]),
    }


def _pick_prefixed(
    row: Mapping[str, Any],
    prefixes: list[str],
    suffixes: list[str],
    direct_names: list[str],
) -> Any:
    names: list[str] = []
    for prefix in prefixes:
        for suffix in suffixes:
            names.append(f"{prefix}_{suffix}")
            names.append(f"{prefix}{suffix}")
    names.extend(direct_names)
    return _pick(row, *names)


def _pick(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name not in row:
            continue
        value = row.get(name)
        if not _missing(value):
            return value
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


def _has_material(record: Mapping[str, Any]) -> bool:
    return any(
        not _missing(record.get(key))
        for key in (
            "date",
            "venue",
            "surface",
            "distance",
            "finish",
            "popularity",
            "time_index",
            "passing_order",
            "running_style",
            "race_name",
        )
    )


def _normalize_label(value: Any) -> str:
    text = clean_text(value)
    if text in {"前走", "蜑崎ｵｰ", "last", "race1"}:
        return "前走"
    if text in {"2走前", "2襍ｰ蜑・", "2back", "two_back", "race2"}:
        return "2走前"
    if text in {"3走前", "3襍ｰ蜑・", "3back", "three_back", "race3"}:
        return "3走前"
    if "前走" in text:
        return "前走"
    if "2" in text:
        return "2走前"
    if "3" in text:
        return "3走前"
    return text or "前走"


def _format_distance(value: Any) -> str:
    if _missing(value):
        return ""
    number = to_float(value)
    if number is not None:
        return f"{int(number)}m" if number.is_integer() else f"{number}m"
    return clean_text(value)


def _format_finish(value: Any) -> str:
    if _missing(value):
        return ""
    text = clean_text(value)
    if re.fullmatch(r"\d{1,2}", text):
        return f"{text}着"
    return text


def _format_popularity(value: Any) -> str:
    if _missing(value):
        return ""
    text = clean_text(value)
    if re.fullmatch(r"\d{1,2}", text):
        return f"{text}人気"
    return text


def _format_index(value: Any) -> str:
    if _missing(value):
        return ""
    number = to_float(value)
    if number is not None:
        return str(int(number)) if number.is_integer() else f"{number:.1f}".rstrip("0").rstrip(".")
    text = clean_text(value)
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    return match.group(0) if match else text


def _condition_text(run: Mapping[str, Any]) -> str:
    pieces = [
        clean_text(run.get("venue")),
        clean_text(run.get("surface")),
        clean_text(run.get("distance")),
        clean_text(run.get("turn")),
    ]
    return " ".join(piece for piece in pieces if piece)


def _detail_condition_text(run: Mapping[str, Any]) -> str:
    pieces = [
        clean_text(run.get("date")),
        clean_text(run.get("venue")),
        clean_text(run.get("surface")),
        clean_text(run.get("distance")),
        clean_text(run.get("turn")),
        clean_text(run.get("race_name")),
    ]
    return " / ".join(piece for piece in pieces if piece)


def _finish_popularity_text(run: Mapping[str, Any]) -> str:
    finish = clean_text(run.get("finish"))
    popularity = clean_text(run.get("popularity"))
    if finish and popularity:
        return f"{finish}（{popularity}）"
    return finish or (f"（{popularity}）" if popularity else "")


def _index_text(run: Mapping[str, Any]) -> str:
    value = clean_text(run.get("time_index"))
    return f"指数{value}" if value else ""


def _passing_text(run: Mapping[str, Any]) -> str:
    value = clean_text(run.get("passing_order"))
    return f"通過{value}" if value else ""


def _running_style_text(run: Mapping[str, Any]) -> str:
    value = clean_text(run.get("running_style"))
    return f"脚質{value}" if value else ""

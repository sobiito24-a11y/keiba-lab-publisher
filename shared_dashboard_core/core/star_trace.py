from __future__ import annotations

import json
import logging
from typing import Any, Iterable

import pandas as pd


LOGGER = logging.getLogger("keiba.star_trace")
_TRACE_ROWS: list[dict[str, Any]] = []


def clear_star_trace() -> None:
    _TRACE_ROWS.clear()


def get_star_trace() -> list[dict[str, Any]]:
    return [dict(row) for row in _TRACE_ROWS]


def log_star_trace(stage: str, rows: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    normalized = [_normalize_row(stage, row) for row in rows]
    _TRACE_ROWS.extend(normalized)
    LOGGER.warning("[STAR_TRACE] stage=%s count=%d", stage, len(normalized))
    for row in normalized:
        LOGGER.warning("[STAR_TRACE] %s", json.dumps(row, ensure_ascii=False, default=str))
    return normalized


def star_trace_row(
    *,
    horse_no: Any = "",
    horse_name: Any = "",
    year_max_index: Any = None,
    star_max_index: Any = None,
    **extra: Any,
) -> dict[str, Any]:
    row = {
        "horse_no": _safe_value(horse_no),
        "horse_name": _safe_value(horse_name),
        "year_max_index": _safe_value(year_max_index),
        "star_max_index": _safe_value(star_max_index),
    }
    for key, value in extra.items():
        row[key] = _safe_value(value)
    return row


def candidate_summary(runs: Iterable[dict[str, Any]]) -> str:
    parts: list[str] = []
    for run in runs or []:
        if not isinstance(run, dict):
            continue
        label = _safe_value(run.get("label") or run.get("race_label") or run.get("race"))
        value = _safe_value(run.get("value"))
        venue = _safe_value(run.get("racecourse") or run.get("venue"))
        distance = _safe_value(run.get("distance"))
        surface = _safe_value(run.get("surface"))
        turn = _safe_value(run.get("direction") or run.get("turn"))
        detail = "".join(str(item) for item in (venue, surface, distance, turn) if item not in ("", None))
        parts.append(f"{label}:{value}:{detail or 'no_condition'}")
    return " | ".join(parts)


def _normalize_row(stage: str, row: dict[str, Any]) -> dict[str, Any]:
    result = {"stage": stage}
    result.update({key: _safe_value(value) for key, value in dict(row or {}).items()})
    for key in ("horse_no", "horse_name", "year_max_index", "star_max_index"):
        result.setdefault(key, "")
    return result


def _safe_value(value: Any) -> Any:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if isinstance(value, (int, float, bool)):
        return value
    text = str(value).strip()
    return text if text else None

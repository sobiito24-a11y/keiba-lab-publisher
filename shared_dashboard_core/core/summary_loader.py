from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping


class SummaryLoadError(ValueError):
    """Raised when a summary file exists but is not valid dashboard JSON."""


def load_summary(path: str | Path) -> dict[str, Any] | None:
    source = Path(path)
    if not source.is_file():
        return None
    try:
        data = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise SummaryLoadError(f"Summary JSONを読み込めません: {source.name}") from exc
    if not isinstance(data, dict):
        raise SummaryLoadError(f"Summary JSONのルートはobjectである必要があります: {source.name}")
    return data


def summary_date(summary: Mapping[str, Any]) -> str:
    return str(summary.get("date") or "").strip()


def summary_counts(summary: Mapping[str, Any]) -> tuple[int, int, int]:
    counts = summary.get("counts")
    if isinstance(counts, Mapping):
        return (
            _count_value(counts.get("buy")),
            _count_value(counts.get("hold")),
            _count_value(counts.get("skip")),
        )
    return (
        _list_count(summary.get("buy")),
        _list_count(summary.get("hold")),
        _list_count(summary.get("skip")),
    )


def summary_venues(summary: Mapping[str, Any]) -> tuple[str, ...]:
    venues = summary.get("venues")
    if isinstance(venues, list):
        return tuple(dict.fromkeys(str(item).strip() for item in venues if str(item).strip()))
    derived: list[str] = []
    for decision in ("buy", "hold", "skip"):
        items = summary.get(decision)
        if not isinstance(items, list):
            continue
        for item in items:
            if not isinstance(item, Mapping):
                continue
            venue = str(item.get("venue") or "").strip()
            if venue:
                derived.append(venue)
    return tuple(dict.fromkeys(derived))


def _count_value(value: Any) -> int:
    try:
        return max(0, int(value))
    except (TypeError, ValueError):
        return 0


def _list_count(value: Any) -> int:
    return len(value) if isinstance(value, list) else 0

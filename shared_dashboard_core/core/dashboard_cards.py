from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


CORE_MARKS = ("◎", "○", "▲")
MARK_ORDER = {mark: index for index, mark in enumerate(CORE_MARKS)}


class DashboardDetailError(ValueError):
    """Raised when a referenced PredictionResult detail cannot be read safely."""


@dataclass(frozen=True)
class CardHorse:
    mark: str
    number: str
    name: str
    ai_score: str = "—"
    ability: str = "—"
    trust_summary: str = ""


@dataclass(frozen=True)
class RaceCard:
    source: str
    race_id: str
    venue: str
    race_number: str
    post_time: str
    ticket: str
    strategy_score: float
    roi: str
    investment_rank: str
    condition_match: str
    adopted_strategy: str
    buy_reasons: tuple[str, ...]
    horses: tuple[CardHorse, ...]
    detail_path: str
    detail_available: bool
    reason: str = ""


def summary_decisions(summary: Mapping[str, Any], decision: str) -> tuple[Mapping[str, Any], ...]:
    items = summary.get(decision.lower())
    if not isinstance(items, list):
        return ()
    return tuple(item for item in items if isinstance(item, Mapping))


def prepare_race_cards(
    summary: Mapping[str, Any],
    analysis_dir: str | Path,
    *,
    source: str,
    decision: str = "buy",
    venue: str = "",
    sort_mode: str = "score",
) -> tuple[RaceCard, ...]:
    cards = [
        prepare_race_card(item, analysis_dir, source=source)
        for item in summary_decisions(summary, decision)
        if _venue_matches(item, venue)
    ]
    return tuple(sorted(cards, key=_race_card_sort_key if sort_mode == "race" else _score_sort_key))


def today_best_five(
    sources: Iterable[tuple[str, Mapping[str, Any], str | Path]],
    *,
    venue: str = "",
) -> tuple[RaceCard, ...]:
    cards: list[RaceCard] = []
    for source, summary, analysis_dir in sources:
        cards.extend(prepare_race_cards(summary, analysis_dir, source=source, decision="buy", venue=venue))
    return tuple(sorted(cards, key=_score_sort_key)[:5])


def filtered_summary_counts(summary: Mapping[str, Any], venue: str = "") -> tuple[int, int, int]:
    return tuple(
        len([item for item in summary_decisions(summary, decision) if _venue_matches(item, venue)])
        for decision in ("buy", "hold", "skip")
    )  # type: ignore[return-value]


def prepare_race_card(
    item: Mapping[str, Any],
    analysis_dir: str | Path,
    *,
    source: str,
) -> RaceCard:
    detail_path = _text(item.get("detail_path"))
    detail: Mapping[str, Any] = {}
    detail_available = False
    if detail_path:
        try:
            loaded = load_detail_json(analysis_dir, detail_path)
        except DashboardDetailError:
            loaded = None
        if loaded is not None:
            detail = loaded
            detail_available = True

    race_info = detail.get("race_info")
    if not isinstance(race_info, Mapping):
        race_info = {}

    venue = _first_text(item, ("venue", "開催場")) or _first_text(race_info, ("racecourse", "venue", "place", "競馬場"))
    race_number = _race_number(item, detail, race_info)
    post_time = _post_time(item, race_info)
    ticket = _first_text(item, ("ticket", "券種", "買い目")) or "—"
    condition_match = _first_text(item, ("condition_match", "matched_condition", "一致条件")) or _ticket_condition(ticket)
    adopted_strategy = _first_text(item, ("selected_strategy", "adopted_strategy", "strategy_name", "strategy", "採用戦略")) or ticket
    strategy_score = _safe_number(item.get("strategy_score", item.get("score"))) or 0.0
    roi = _format_percent(item.get("expected_roi", item.get("roi")))

    return RaceCard(
        source=source,
        race_id=_text(item.get("race_id")),
        venue=venue or "開催場不明",
        race_number=race_number or "R不明",
        post_time=post_time or "—",
        ticket=ticket,
        strategy_score=strategy_score,
        roi=roi,
        investment_rank=_first_text(item, ("investment_rank", "confidence", "信頼度")) or "—",
        condition_match=condition_match or "—",
        adopted_strategy=adopted_strategy,
        buy_reasons=_buy_reason_lines(item, condition_match, roi, strategy_score),
        horses=_card_horses(item, detail),
        detail_path=detail_path,
        detail_available=detail_available,
        reason=_text(item.get("reason")),
    )


def resolve_detail_path(analysis_dir: str | Path, detail_path: str) -> Path:
    base = Path(analysis_dir).resolve()
    relative = Path(_text(detail_path))
    if not str(relative) or relative.is_absolute():
        raise DashboardDetailError("detail_path is invalid.")
    target = (base / relative).resolve()
    if target == base or base not in target.parents:
        raise DashboardDetailError("detail_path points outside assets/analysis.")
    return target


def load_detail_json(analysis_dir: str | Path, detail_path: str) -> dict[str, Any] | None:
    target = resolve_detail_path(analysis_dir, detail_path)
    if not target.is_file():
        return None
    try:
        data = json.loads(target.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise DashboardDetailError(f"Could not read detail JSON: {target.name}") from exc
    if not isinstance(data, dict):
        raise DashboardDetailError(f"Detail JSON must contain an object: {target.name}")
    return data


def format_strategy_score(value: float) -> str:
    return _format_scalar(value)


def _buy_reason_lines(
    item: Mapping[str, Any],
    condition_match: str,
    roi: str,
    strategy_score: float,
) -> tuple[str, ...]:
    raw_reason = _first_text(item, ("buy_reason", "buy_reasons", "reason", "理由"))
    reasons: list[str] = []
    for part in re.split(r"\r?\n|[／/]", raw_reason):
        clean = re.sub(r"^[・\-\s]+", "", part).strip()
        if clean:
            _append_unique(reasons, clean)
    if condition_match and condition_match != "—":
        _append_unique(reasons, f"{condition_match} 条件一致")
    if roi != "—":
        _append_unique(reasons, f"回収率 {roi}")
    if not _is_missing(item.get("strategy_score", item.get("score"))):
        _append_unique(reasons, f"Score {format_strategy_score(strategy_score)}")
    return tuple(reasons[:4])


def _ticket_condition(ticket: str) -> str:
    text = _text(ticket)
    if not text or text == "—":
        return ""
    parts = text.split(maxsplit=1)
    return parts[1].strip() if len(parts) == 2 else ""


def _append_unique(values: list[str], candidate: str) -> None:
    key = re.sub(r"\s+", "", candidate).lower()
    if key and all(re.sub(r"\s+", "", value).lower() != key for value in values):
        values.append(candidate)


def _card_horses(item: Mapping[str, Any], detail: Mapping[str, Any]) -> tuple[CardHorse, ...]:
    rows_by_number: dict[str, dict[str, Any]] = {}
    for table_name in ("horse_evaluation", "overall_table"):
        rows = detail.get(table_name)
        if not isinstance(rows, list):
            continue
        for raw_row in rows:
            if not isinstance(raw_row, Mapping):
                continue
            number = _horse_number_key(_first_value(raw_row, ("馬番", "horse_no", "horse_number", "number")))
            if not number:
                continue
            merged = rows_by_number.setdefault(number, {})
            for key, value in raw_row.items():
                if not _is_missing(value):
                    merged[str(key)] = value

    horses: list[CardHorse] = []
    seen: set[str] = set()
    marked_rows: list[tuple[int, int, CardHorse]] = []
    for index, (number, row) in enumerate(rows_by_number.items()):
        mark = _normalize_mark(_first_value(row, ("表示印", "display_mark", "old_final_mark", "最終印", "印", "mark")))
        if mark not in MARK_ORDER:
            continue
        marked_rows.append((MARK_ORDER[mark], index, _horse_from_row(mark, number, row)))
        seen.add(number)
    horses.extend(item[2] for item in sorted(marked_rows, key=lambda value: (value[0], value[1])))

    summary_horses = item.get("horses")
    if isinstance(summary_horses, list):
        for raw_horse in summary_horses:
            if not isinstance(raw_horse, Mapping):
                continue
            number = _horse_number_key(_first_value(raw_horse, ("number", "馬番", "horse_no")))
            if not number or number in seen:
                continue
            group = _text(raw_horse.get("group")).upper()
            mark = _normalize_mark(_first_value(raw_horse, ("role", "mark", "印"))) or _mark_from_group(group)
            detail_row = rows_by_number.get(number, {})
            name = _first_text(raw_horse, ("name", "馬名")) or _first_text(detail_row, ("馬名", "horse_name", "name"))
            horses.append(
                CardHorse(
                    mark=mark,
                    number=number,
                    name=name or "馬名不明",
                    ai_score=_display_ai(detail_row),
                    ability=_display_ability(detail_row),
                    trust_summary=_first_text(raw_horse, ("horse_trust_summary", "trust_summary", "信頼根拠"))
                    or _first_text(detail_row, ("horse_trust_summary", "trust_summary", "信頼根拠")),
                )
            )
            seen.add(number)

    return tuple(sorted(horses, key=lambda horse: (MARK_ORDER.get(horse.mark, len(MARK_ORDER)), _race_number_sort_value(horse.number))))


def _horse_from_row(mark: str, number: str, row: Mapping[str, Any]) -> CardHorse:
    return CardHorse(
        mark=mark,
        number=number,
        name=_first_text(row, ("馬名", "horse_name", "name")) or "馬名不明",
        ai_score=_display_ai(row),
        ability=_display_ability(row),
        trust_summary=_first_text(row, ("horse_trust_summary", "trust_summary", "信頼根拠")),
    )


def _display_ai(row: Mapping[str, Any]) -> str:
    value = _first_value(row, ("AI点", "normalized_ai_score", "旧AI点"))
    return "—" if _is_missing(value) else _format_scalar(value)


def _display_ability(row: Mapping[str, Any]) -> str:
    band = _first_value(row, ("能力帯", "ability_band", "能力ランク", "ability_rank"))
    score = _first_value(row, ("能力評価値", "ability_display_score"))
    band_text = "" if _is_missing(band) else _format_scalar(band)
    score_text = "" if _is_missing(score) else _format_scalar(score)
    if band_text and score_text:
        return f"{band_text} / {score_text}"
    return band_text or score_text or "—"


def _race_number(item: Mapping[str, Any], detail: Mapping[str, Any], race_info: Mapping[str, Any]) -> str:
    value = _first_text(item, ("race_number", "race_no", "R", "レース番号")) or _first_text(
        race_info, ("race_number", "race_no", "R", "レース番号")
    )
    if not value:
        text = " ".join(filter(None, (_text(item.get("race_name")), _text(item.get("race_title")), _text(detail.get("race_name")))))
        match = re.search(r"(?<!\d)(\d{1,2})\s*R", text, flags=re.IGNORECASE)
        value = match.group(1) if match else ""
    if not value:
        return ""
    compact = value.strip()
    return compact if compact.upper().endswith("R") else f"{compact}R"


def _post_time(item: Mapping[str, Any], race_info: Mapping[str, Any]) -> str:
    direct = _first_text(item, ("post_time", "start_time", "発走時刻", "発走")) or _first_text(
        race_info, ("post_time", "start_time", "発走時刻", "発走")
    )
    if direct:
        match = re.search(r"(?<!\d)\d{1,2}:\d{2}(?!\d)", direct)
        return match.group(0) if match else direct
    race_data = _first_text(race_info, ("race_data", "raceData", "レース情報"))
    match = re.search(r"(?<!\d)\d{1,2}:\d{2}(?!\d)", race_data)
    return match.group(0) if match else ""


def _venue_matches(item: Mapping[str, Any], venue: str) -> bool:
    if not venue or venue == "すべて":
        return True
    item_venue = _first_text(item, ("venue", "開催場"))
    return item_venue == venue


def _score_sort_key(card: RaceCard) -> tuple[float, int, str]:
    return (-card.strategy_score, _race_number_sort_value(card.race_number), card.race_id)


def _race_card_sort_key(card: RaceCard) -> tuple[str, int, str]:
    return (card.venue, _race_number_sort_value(card.race_number), card.race_id)


def _race_number_sort_value(value: str) -> int:
    match = re.search(r"([1-9]|1[0-2])", str(value or ""))
    return int(match.group(1)) if match else 99


def _normalize_mark(value: Any) -> str:
    text = _text(value)
    for mark in CORE_MARKS:
        if mark in text:
            return mark
    return ""


def _mark_from_group(group: str) -> str:
    if group == "SS":
        return "◎"
    if group == "A":
        return "○"
    if group == "B":
        return "▲"
    return ""


def _horse_number_key(value: Any) -> str:
    if _is_missing(value):
        return ""
    number = _safe_number(value)
    if number is not None and number.is_integer():
        return str(int(number))
    return _text(value)


def _first_text(source: Mapping[str, Any], keys: Sequence[str]) -> str:
    value = _first_value(source, keys)
    return "" if _is_missing(value) else _text(value)


def _first_value(source: Mapping[str, Any], keys: Sequence[str]) -> Any:
    for key in keys:
        value = source.get(key)
        if not _is_missing(value):
            return value
    return None


def _format_percent(value: Any) -> str:
    number = _safe_number(value)
    if number is None:
        return "—"
    return f"{_format_scalar(number)}%"


def _format_scalar(value: Any) -> str:
    number = _safe_number(value)
    if number is not None:
        if number.is_integer():
            return str(int(number))
        return f"{number:.1f}".rstrip("0").rstrip(".")
    return _text(value) or "—"


def _safe_number(value: Any) -> float | None:
    if _is_missing(value):
        return None
    try:
        number = float(_text(value).replace(",", "").replace("%", ""))
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip() or value.strip().lower() in {"nan", "none", "null"}
    if isinstance(value, float):
        return math.isnan(value)
    return False


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()

from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from .models import PredictionResult


ASSETS_ANALYSIS_DIR = Path(__file__).resolve().parents[1] / "assets" / "analysis"
DEFAULT_JRA_STRATEGY_PATH = ASSETS_ANALYSIS_DIR / "jra_strategy_selection.json"

BUY_SCORE_MIN = 70.0
HOLD_SCORE_MIN = 50.0
BUY_ROI_MIN = 110.0
HOLD_ROI_MIN = 100.0
BUY_MIN_RACES = 8
BUY_MIN_HITS = 2
MAX_PAYOUT_CONTRIBUTION_LIMIT = 80.0
POINT_YEN = 100

GROUPS = ("SS", "A", "B", "C", "Z")
AI_ROLES = ("AI1", "AI2", "AI3")


@dataclass(frozen=True)
class StrategyHorse:
    number: str
    name: str
    group: str
    mark: str
    ai_rank: int | None
    ai_score: float | None
    odds: float | None
    popularity: int | None

    def to_summary_horse(self) -> dict[str, Any]:
        return {
            "number": int(self.number) if self.number.isdigit() else self.number,
            "name": self.name,
            "group": self.group,
            "role": self.mark,
            "odds": self.odds,
        }


@dataclass(frozen=True)
class StrategyEvaluation:
    strategy_id: str
    ticket_type: str
    label: str
    decision: str
    strategy_score: float
    roi: float | None
    hit_rate: float | None
    sample_size: int
    hits: int
    max_losing_streak: int | None
    max_drawdown: float | None
    max_payout_contribution: float | None
    selected_horses: tuple[StrategyHorse, ...]
    combinations: tuple[tuple[str, ...], ...]
    points: int
    condition_matches: tuple[str, ...]
    condition_misses: tuple[str, ...]
    avoid_matches: tuple[str, ...]
    reasons: tuple[str, ...]
    non_adoption_reason: str = ""

    @property
    def ticket_label(self) -> str:
        return f"{self.ticket_type} {self.label}".strip()

    @property
    def investment(self) -> int:
        return self.points * POINT_YEN if self.decision == "BUY" else 0


def load_jra_strategy_selection(path: str | Path = DEFAULT_JRA_STRATEGY_PATH) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("jra_strategy_selection.json must contain an object.")
    race_type = str(data.get("race_type") or data.get("scope") or "").lower()
    if race_type != "jra":
        raise ValueError("JRA strategy selection must have race_type=jra.")
    if not isinstance(data.get("strategies"), list):
        raise ValueError("JRA strategy selection requires strategies.")
    return dict(data)


def select_jra_investment_strategy(
    result: PredictionResult,
    strategy_selection: Mapping[str, Any],
) -> dict[str, Any]:
    if result.race_mode != "jra":
        raise ValueError("JRA strategy selection accepts JRA PredictionResult only.")
    horses = _current_horses(result)
    if not horses:
        return _skip_payload(strategy_selection, "評価対象馬を取得できませんでした。", ())

    role_map = _role_map(horses)
    evaluations = tuple(
        _evaluate_strategy(dict(strategy), role_map)
        for strategy in strategy_selection.get("strategies", [])
        if isinstance(strategy, Mapping)
    )
    valid = [item for item in evaluations if item.decision in {"BUY", "HOLD"}]
    if not valid:
        return _skip_payload(strategy_selection, "正式購入条件に一致する戦略がありません。", evaluations)
    selected = sorted(valid, key=_strategy_priority_key)[0]
    return _evaluation_payload(strategy_selection, selected, _mark_adoption(evaluations, selected.strategy_id))


def _evaluate_strategy(strategy: dict[str, Any], role_map: Mapping[str, tuple[StrategyHorse, ...]]) -> StrategyEvaluation:
    condition_matches: list[str] = []
    condition_misses: list[str] = []
    for condition in _condition_list(strategy.get("conditions")):
        label = _condition_label(condition)
        if _condition_matches(condition, role_map):
            condition_matches.append(label)
        else:
            condition_misses.append(label)

    avoid_matches: list[str] = []
    for condition in _condition_list(strategy.get("avoid_conditions")):
        if _condition_matches(condition, role_map):
            avoid_matches.append(_condition_label(condition))

    combinations, selected_horses = _build_tickets(role_map, strategy.get("role_pattern"), str(strategy.get("ticket_type") or ""))
    points = len(combinations)
    score = _strategy_score(strategy, points)
    roi = _safe_float(strategy.get("return_rate", strategy.get("roi")))
    hit_rate = _safe_float(strategy.get("hit_rate"))
    sample_size = int(_safe_float(strategy.get("target_races", strategy.get("sample_size"))) or 0)
    hits = int(_safe_float(strategy.get("hits")) or 0)
    max_losing = _safe_int(strategy.get("max_losing_streak"))
    max_dd = _safe_float(strategy.get("max_drawdown"))
    max_contrib = _safe_float(strategy.get("max_payout_contribution"))

    reasons: list[str] = []
    decision = "SKIP"
    if condition_misses:
        reasons.append("条件不一致")
    if avoid_matches:
        reasons.append("回避条件該当")
    if points <= 0:
        reasons.append("買い目生成不可")

    tier = str(strategy.get("recommendation_tier") or "").lower()
    formal = tier in {"formal", "buy", "official"} or str(strategy.get("formal") or "").lower() == "true"
    if not reasons:
        if (
            formal
            and score >= BUY_SCORE_MIN
            and _number_gte(roi, BUY_ROI_MIN)
            and sample_size >= BUY_MIN_RACES
            and hits >= BUY_MIN_HITS
            and (max_contrib is None or max_contrib <= MAX_PAYOUT_CONTRIBUTION_LIMIT)
        ):
            decision = "BUY"
            reasons.append("正式条件成立")
        elif tier != "avoid" and (score >= HOLD_SCORE_MIN or _number_gte(roi, HOLD_ROI_MIN)):
            decision = "HOLD"
            reasons.append("参考条件成立")
        else:
            reasons.append("期待値不足")

    return StrategyEvaluation(
        strategy_id=str(strategy.get("strategy_id") or ""),
        ticket_type=str(strategy.get("ticket_type") or ""),
        label=str(strategy.get("label") or ""),
        decision=decision,
        strategy_score=score,
        roi=roi,
        hit_rate=hit_rate,
        sample_size=sample_size,
        hits=hits,
        max_losing_streak=max_losing,
        max_drawdown=max_dd,
        max_payout_contribution=max_contrib,
        selected_horses=selected_horses,
        combinations=combinations,
        points=points,
        condition_matches=tuple(condition_matches),
        condition_misses=tuple(condition_misses),
        avoid_matches=tuple(avoid_matches),
        reasons=tuple(reasons),
    )


def _condition_matches(condition: Mapping[str, Any], role_map: Mapping[str, tuple[StrategyHorse, ...]]) -> bool:
    role = str(condition.get("role") or "").upper()
    field = str(condition.get("field") or "").lower()
    op = str(condition.get("op") or "eq").lower()
    expected = condition.get("value")
    horses = role_map.get(role, ())
    if field == "count":
        return _compare(len(horses), op, expected)
    if not horses:
        return False
    values = [_horse_field_value(horse, field) for horse in horses]
    values = [value for value in values if value is not None]
    return bool(values) and any(_compare(value, op, expected) for value in values)


def _build_tickets(
    role_map: Mapping[str, tuple[StrategyHorse, ...]],
    pattern: Any,
    ticket_type: str,
) -> tuple[tuple[tuple[str, ...], ...], tuple[StrategyHorse, ...]]:
    if not isinstance(pattern, Mapping):
        return (), ()
    pattern_type = str(pattern.get("type") or "").lower()
    selected: list[StrategyHorse] = []
    combos: set[tuple[str, ...]] = set()
    if pattern_type == "single":
        for horse in _horses_for_roles(role_map, pattern.get("roles", [])):
            combos.add((horse.number,))
            selected.append(horse)
    elif pattern_type == "pair":
        left = _horses_for_roles(role_map, pattern.get("left_roles", []))
        right = _horses_for_roles(role_map, pattern.get("right_roles", []))
        for first in left:
            for second in right:
                if first.number == second.number:
                    continue
                combos.add(tuple(sorted((first.number, second.number), key=_number_sort_key)))
                selected.extend((first, second))
    elif pattern_type == "exacta":
        left = _horses_for_roles(role_map, pattern.get("left_roles", []))
        right = _horses_for_roles(role_map, pattern.get("right_roles", []))
        for first in left:
            for second in right:
                if first.number != second.number:
                    combos.add((first.number, second.number))
                    selected.extend((first, second))
    elif pattern_type == "box":
        horses = _horses_for_roles(role_map, pattern.get("roles", []))
        size = int(_safe_float(pattern.get("size")) or (3 if "三連" in ticket_type else 2))
        for combo in itertools.combinations(horses, size):
            combos.add(tuple(sorted((horse.number for horse in combo), key=_number_sort_key)))
            selected.extend(combo)
    return tuple(sorted(combos, key=lambda combo: tuple(_number_sort_key(value) for value in combo))), _unique_horses(selected)


def _strategy_score(strategy: Mapping[str, Any], ticket_count: int) -> float:
    score = _safe_float(strategy.get("strategy_score", strategy.get("score"))) or 0.0
    roi = _safe_float(strategy.get("return_rate", strategy.get("roi")))
    hit_rate = _safe_float(strategy.get("hit_rate"))
    sample_size = _safe_float(strategy.get("target_races", strategy.get("sample_size"))) or 0.0
    max_losing = _safe_float(strategy.get("max_losing_streak"))
    max_contrib = _safe_float(strategy.get("max_payout_contribution"))
    if roi is not None and roi > 100:
        score += min(10.0, (roi - 100.0) / 5.0)
    if hit_rate is not None:
        score += min(8.0, hit_rate / 10.0)
    score += min(8.0, sample_size / 4.0)
    if max_losing is not None:
        score -= min(8.0, max_losing / 2.0)
    if max_contrib is not None and max_contrib > 80:
        score -= min(10.0, (max_contrib - 80.0) / 2.0)
    if ticket_count > 4:
        score -= min(8.0, float(ticket_count - 4))
    return round(max(0.0, min(100.0, score)), 1)


def _current_horses(result: PredictionResult) -> tuple[StrategyHorse, ...]:
    source = result.overall_table if isinstance(result.overall_table, pd.DataFrame) and not result.overall_table.empty else result.horse_evaluation
    if not isinstance(source, pd.DataFrame) or source.empty:
        return ()
    horses: list[StrategyHorse] = []
    for _, row in source.iterrows():
        data = {str(key): row.get(key) for key in source.columns}
        number = _horse_number(_first_value(data, ("horse_no", "horse_number", "number", "馬番")))
        if not number:
            continue
        mark = _text(_first_value(data, ("display_mark", "old_final_mark", "mark", "final_mark", "印", "最終印", "表示印")))
        group = _text(_first_value(data, ("display_group", "group", "グループ"))).upper()
        if group not in GROUPS:
            group = _group_from_mark(mark)
        horses.append(
            StrategyHorse(
                number=number,
                name=_text(_first_value(data, ("horse_name", "name", "馬名"))) or f"{number}番",
                group=group,
                mark=mark,
                ai_rank=_safe_int(_first_value(data, ("ai_rank", "AI順位"))),
                ai_score=_safe_float(_first_value(data, ("normalized_ai_score", "AI点", "score"))),
                odds=_safe_float(_first_value(data, ("odds", "win_odds", "単勝オッズ", "オッズ"))),
                popularity=_safe_int(_first_value(data, ("popularity", "人気"))),
            )
        )
    return tuple(horses)


def _role_map(horses: Iterable[StrategyHorse]) -> dict[str, tuple[StrategyHorse, ...]]:
    ordered = sorted(horses, key=_horse_ai_sort_key)
    result: dict[str, list[StrategyHorse]] = {group: [] for group in GROUPS}
    for horse in ordered:
        result.setdefault(horse.group, []).append(horse)
    for index, role in enumerate(AI_ROLES):
        if index < len(ordered):
            result[role] = [ordered[index]]
    return {key: tuple(value) for key, value in result.items()}


def _horses_for_roles(role_map: Mapping[str, tuple[StrategyHorse, ...]], roles: Any) -> tuple[StrategyHorse, ...]:
    result: list[StrategyHorse] = []
    for role in roles if isinstance(roles, list) else []:
        result.extend(role_map.get(str(role).upper(), ()))
    return _unique_horses(result)


def _unique_horses(horses: Iterable[StrategyHorse]) -> tuple[StrategyHorse, ...]:
    seen: set[str] = set()
    result: list[StrategyHorse] = []
    for horse in horses:
        if horse.number in seen:
            continue
        seen.add(horse.number)
        result.append(horse)
    return tuple(sorted(result, key=lambda horse: _number_sort_key(horse.number)))


def _evaluation_payload(
    strategy_selection: Mapping[str, Any],
    selected: StrategyEvaluation,
    evaluations: tuple[StrategyEvaluation, ...],
) -> dict[str, Any]:
    return {
        "race_type": "jra",
        "decision": selected.decision,
        "selected_strategy": selected.ticket_label,
        "strategy_id": selected.strategy_id,
        "ticket": selected.ticket_label,
        "combinations": list(_combo_texts(selected.combinations, selected.ticket_type)),
        "points": selected.points,
        "investment": selected.investment,
        "strategy_score": selected.strategy_score,
        "expected_roi": selected.roi,
        "roi": selected.roi,
        "hit_rate": selected.hit_rate,
        "sample_size": selected.sample_size,
        "confidence": _confidence_from_score(selected.strategy_score, selected.decision),
        "reason": " / ".join((*selected.reasons, *selected.condition_matches)),
        "avoid_reason": " / ".join(selected.avoid_matches),
        "horses": [horse.to_summary_horse() for horse in selected.selected_horses],
        "strategy_audit": _audit_payload(strategy_selection, evaluations, selected.strategy_id),
    }


def _skip_payload(
    strategy_selection: Mapping[str, Any],
    reason: str,
    evaluations: Iterable[StrategyEvaluation],
) -> dict[str, Any]:
    evaluation_tuple = tuple(evaluations)
    return {
        "race_type": "jra",
        "decision": "SKIP",
        "selected_strategy": "",
        "strategy_id": "",
        "ticket": "",
        "combinations": [],
        "points": 0,
        "investment": 0,
        "strategy_score": 0,
        "expected_roi": None,
        "roi": None,
        "hit_rate": None,
        "sample_size": 0,
        "confidence": "☆☆☆☆☆",
        "reason": reason,
        "avoid_reason": "",
        "horses": [],
        "strategy_audit": _audit_payload(strategy_selection, evaluation_tuple, ""),
    }


def _mark_adoption(evaluations: Iterable[StrategyEvaluation], selected_strategy_id: str) -> tuple[StrategyEvaluation, ...]:
    result: list[StrategyEvaluation] = []
    for item in evaluations:
        if item.strategy_id == selected_strategy_id:
            result.append(item)
            continue
        result.append(
            StrategyEvaluation(
                strategy_id=item.strategy_id,
                ticket_type=item.ticket_type,
                label=item.label,
                decision=item.decision,
                strategy_score=item.strategy_score,
                roi=item.roi,
                hit_rate=item.hit_rate,
                sample_size=item.sample_size,
                hits=item.hits,
                max_losing_streak=item.max_losing_streak,
                max_drawdown=item.max_drawdown,
                max_payout_contribution=item.max_payout_contribution,
                selected_horses=item.selected_horses,
                combinations=item.combinations,
                points=item.points,
                condition_matches=item.condition_matches,
                condition_misses=item.condition_misses,
                avoid_matches=item.avoid_matches,
                reasons=item.reasons,
                non_adoption_reason="優先順位で非採用" if item.decision in {"BUY", "HOLD"} else "条件不成立",
            )
        )
    return tuple(result)


def _audit_payload(
    strategy_selection: Mapping[str, Any],
    evaluations: Iterable[StrategyEvaluation],
    selected_strategy_id: str,
) -> dict[str, Any]:
    return {
        "race_type": "jra",
        "analysis_json": "jra_strategy_selection.json",
        "strategy_selection_id": strategy_selection.get("strategy_id"),
        "validation": strategy_selection.get("validation") or strategy_selection.get("source") or {},
        "selected_strategy_id": selected_strategy_id,
        "candidates": [
            {
                "strategy_id": item.strategy_id,
                "ticket_type": item.ticket_type,
                "label": item.label,
                "decision": item.decision,
                "condition_matches": list(item.condition_matches),
                "condition_misses": list(item.condition_misses),
                "avoid_matches": list(item.avoid_matches),
                "horses": [horse.to_summary_horse() for horse in item.selected_horses],
                "combinations": list(_combo_texts(item.combinations, item.ticket_type)),
                "points": item.points,
                "return_rate": item.roi,
                "hit_rate": item.hit_rate,
                "target_races": item.sample_size,
                "max_losing_streak": item.max_losing_streak,
                "max_drawdown": item.max_drawdown,
                "max_payout_contribution": item.max_payout_contribution,
                "strategy_score": item.strategy_score,
                "adoption_reason": "最終採用" if item.strategy_id == selected_strategy_id else "",
                "non_adoption_reason": item.non_adoption_reason,
            }
            for item in evaluations
        ],
    }


def _combo_texts(combinations: Iterable[tuple[str, ...]], ticket_type: str) -> tuple[str, ...]:
    connector = "→" if "馬単" in ticket_type or "三連単" in ticket_type else "-"
    return tuple(connector.join(combo) for combo in combinations)


def _strategy_priority_key(item: StrategyEvaluation) -> tuple[int, float, int, float, float, int]:
    decision_rank = {"BUY": 0, "HOLD": 1, "SKIP": 2}.get(item.decision, 2)
    roi = item.roi if item.roi is not None else -1.0
    hit_rate = item.hit_rate if item.hit_rate is not None else -1.0
    return (decision_rank, -item.strategy_score, -item.sample_size, -roi, -hit_rate, item.points)


def _condition_list(value: Any) -> tuple[Mapping[str, Any], ...]:
    if not isinstance(value, list):
        return ()
    return tuple(item for item in value if isinstance(item, Mapping))


def _condition_label(condition: Mapping[str, Any]) -> str:
    return str(condition.get("label") or f"{condition.get('role')}.{condition.get('field')}")


def _horse_field_value(horse: StrategyHorse, field: str) -> float | int | str | None:
    if field == "odds":
        return horse.odds
    if field == "ai_score":
        return horse.ai_score
    if field == "popularity":
        return horse.popularity
    if field == "group":
        return horse.group
    return None


def _compare(actual: Any, op: str, expected: Any) -> bool:
    actual_number = _safe_float(actual)
    expected_number = _safe_float(expected)
    if op in {"gte", ">="}:
        return actual_number is not None and expected_number is not None and actual_number >= expected_number
    if op in {"gt", ">"}:
        return actual_number is not None and expected_number is not None and actual_number > expected_number
    if op in {"lte", "<="}:
        return actual_number is not None and expected_number is not None and actual_number <= expected_number
    if op in {"lt", "<"}:
        return actual_number is not None and expected_number is not None and actual_number < expected_number
    if op in {"ne", "!="}:
        return actual != expected
    return actual == expected


def _number_gte(value: float | None, threshold: float) -> bool:
    return value is not None and value >= threshold


def _horse_ai_sort_key(horse: StrategyHorse) -> tuple[int, float, int]:
    rank = horse.ai_rank if horse.ai_rank is not None else 999
    score = horse.ai_score if horse.ai_score is not None else -1.0
    return (rank, -score, _number_sort_key(horse.number))


def _number_sort_key(value: Any) -> int:
    try:
        return int(float(str(value)))
    except (TypeError, ValueError):
        return 999


def _horse_number(value: Any) -> str:
    if _is_missing(value):
        return ""
    number = _safe_float(value)
    if number is not None and number.is_integer():
        return str(int(number))
    return _text(value)


def _group_from_mark(mark: str) -> str:
    text = str(mark or "")
    if "◎" in text:
        return "SS"
    if "○" in text or "▲" in text:
        return "A"
    if "△" in text:
        return "B"
    if "✓" in text or "✔" in text:
        return "C"
    return "Z"


def _first_value(data: Mapping[str, Any], keys: tuple[str, ...]) -> Any:
    for key in keys:
        value = data.get(key)
        if not _is_missing(value):
            return value
    return None


def _safe_float(value: Any) -> float | None:
    if _is_missing(value):
        return None
    try:
        number = float(str(value).replace(",", "").replace("%", "").strip())
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _safe_int(value: Any) -> int | None:
    number = _safe_float(value)
    return int(number) if number is not None else None


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    if isinstance(value, str):
        return not value.strip() or value.strip().lower() in {"nan", "none", "null", "-"}
    try:
        return bool(pd.isna(value))
    except (TypeError, ValueError):
        return False


def _confidence_from_score(score: float, decision: str) -> str:
    if decision == "SKIP":
        return "☆☆☆☆☆"
    count = 5 if score >= 85 else 4 if score >= 70 else 3 if score >= 55 else 2 if score >= 40 else 1
    return "★" * count + "☆" * (5 - count)

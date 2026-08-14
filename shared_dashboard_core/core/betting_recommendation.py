# -*- coding: utf-8 -*-
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import pandas as pd

from .purchase_conditions import (
    ASSETS_ANALYSIS_DIR,
    DEFAULT_REPORT_DIR,
    ConditionSpec,
    condition_mask,
    enrich_current_table,
    to_float,
)
from .ticket_strategy_analysis import POINT_LIMITS, build_tickets_for_race, role_numbers, unique_nums


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ANALYSIS_JSON = ASSETS_ANALYSIS_DIR / "betting_recommendations.json"
LEGACY_TICKET_JSON = DEFAULT_REPORT_DIR / "ticket_strategy_ranked.json"
LEGACY_CONDITION_JSON = DEFAULT_REPORT_DIR / "purchase_condition_ranked.json"


@dataclass(frozen=True)
class BettingRecommendation:
    ticket_type: str
    label: str
    stars: str
    expected_roi: float | None
    condition: str
    reason: str
    source: str = "fixed"
    risk_label: str = ""
    strategy_id: str = ""
    hit_rate: float | None = None
    sample_races: int = 0
    ticket_count: int = 0
    tickets: tuple[str, ...] = ()
    ticket_numbers: tuple[tuple[str, ...], ...] = ()
    ticket_horses: tuple[str, ...] = ()
    matched_conditions: tuple[str, ...] = ()
    unmatched_conditions: tuple[str, ...] = ()
    adopted_reason: str = ""
    audit: dict[str, Any] = field(default_factory=dict)


RECOMMENDATION_RULES = {
    "win_ai3_value": {
        "ticket_type": "単勝",
        "label": "AI3位",
        "condition": "AI3位 / オッズ8〜20倍",
        "expected_roi": 289.0,
    },
    "win_ai1_value": {
        "ticket_type": "単勝",
        "label": "AI1位",
        "condition": "AI1位 / オッズ8〜20倍",
        "expected_roi": 263.0,
    },
    "wide_ss_c": {
        "ticket_type": "ワイド",
        "label": "SS-C",
        "condition": "SSとC穴候補の組み合わせ",
        "expected_roi": 133.0,
    },
    "trio_ss_a_b_c": {
        "ticket_type": "三連複",
        "label": "SS/A→B→SS/A/B/C",
        "condition": "SS・A本線にB/Cを絡める形",
        "expected_roi": 269.0,
    },
}


LAST_LOAD_DIAGNOSTIC: dict[str, Any] = {}
LAST_MATCH_AUDIT: list[dict[str, Any]] = []


def build_betting_recommendations(
    table: Any,
    *,
    max_items: int = 4,
    json_paths: list[Path] | None = None,
) -> list[BettingRecommendation]:
    """Build display-only betting hints.

    Analysis JSON is preferred when available.  The old fixed rules are used
    only when no JSON exists, so stale hard-coded ROI does not mask a broken
    or newer analysis file.
    """

    if table is None or not isinstance(table, pd.DataFrame) or table.empty:
        return []

    payload, diagnostic = load_recommendation_payload(json_paths)
    LAST_LOAD_DIAGNOSTIC.clear()
    LAST_LOAD_DIAGNOSTIC.update(diagnostic)
    if payload is not None:
        return build_recommendations_from_payload(table, payload, max_items=max_items)
    # Fallback is used only when the analysis JSON cannot be loaded.  A loaded
    # JSON with no current-race match returns no recommendation instead.
    return build_fixed_betting_recommendations(table, max_items=max_items)


def load_recommendation_payload(json_paths: list[Path] | None = None) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    paths = json_paths if json_paths is not None else [DEFAULT_ANALYSIS_JSON, LEGACY_TICKET_JSON, LEGACY_CONDITION_JSON]
    diagnostics: list[dict[str, Any]] = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            diagnostics.append({"path": str(path), "status": "missing"})
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return None, {
                "status": "parse_error",
                "path": str(path),
                "fallback_reason": f"analysis json parse failed: {exc}",
                "checked": diagnostics,
            }
        recommendations = payload.get("recommendations")
        return payload, {
            "status": "loaded",
            "path": str(path),
            "mtime": path.stat().st_mtime,
            "recommendation_count": len(recommendations) if isinstance(recommendations, list) else 0,
            "source_race_count": (payload.get("source") or payload.get("meta") or {}).get("race_count")
            or (payload.get("source") or {}).get("source_race_count"),
            "checked": diagnostics,
        }
    return None, {
        "status": "missing",
        "path": "",
        "fallback_reason": "analysis json not found",
        "checked": diagnostics,
    }


def build_recommendations_from_payload(
    table: pd.DataFrame,
    payload: dict[str, Any],
    *,
    max_items: int,
) -> list[BettingRecommendation]:
    current = enrich_current_table(table)
    raw_items = payload.get("recommendations", [])
    if not isinstance(raw_items, list):
        return []

    LAST_MATCH_AUDIT.clear()
    candidates: list[BettingRecommendation] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        recommendation, audit = evaluate_payload_item(current, item)
        LAST_MATCH_AUDIT.append(audit)
        if recommendation is not None:
            candidates.append(recommendation)

    if not any(item.risk_label == "正式" for item in candidates):
        for audit in LAST_MATCH_AUDIT:
            if audit.get("matched"):
                audit["adopted"] = False
                audit["non_adoption_reason"] = "正式推奨が0件のため見送り"
        return []

    candidates = sorted(candidates, key=recommendation_priority_key)
    matched: list[BettingRecommendation] = []
    used_types: set[str] = set()
    high_risk_used = False
    for recommendation in candidates:
        if recommendation.risk_label == "高リスク" and high_risk_used:
            _mark_audit_not_adopted(recommendation.strategy_id, "高リスクは最大1件まで")
            continue
        # Keep the section varied; do not fill all slots with one ticket type.
        if recommendation.ticket_type in used_types and len(used_types) < max_items:
            _mark_audit_not_adopted(recommendation.strategy_id, "券種の偏りを避けるため非採用")
            continue
        matched.append(_with_adopted_reason(recommendation, "条件一致かつ優先順位内"))
        if recommendation.risk_label == "高リスク":
            high_risk_used = True
        used_types.add(recommendation.ticket_type)
        _mark_audit_adopted(recommendation.strategy_id, "条件一致かつ優先順位内")
        if len(matched) >= max_items:
            break

    if len(matched) < max_items:
        already_added = {item.strategy_id for item in matched}
        for recommendation in candidates:
            if recommendation.strategy_id in already_added:
                continue
            if recommendation.risk_label == "高リスク" and high_risk_used:
                _mark_audit_not_adopted(recommendation.strategy_id, "高リスクは最大1件まで")
                continue
            matched.append(_with_adopted_reason(recommendation, "条件一致かつ追加採用"))
            already_added.add(recommendation.strategy_id)
            if recommendation.risk_label == "高リスク":
                high_risk_used = True
            _mark_audit_adopted(recommendation.strategy_id, "条件一致かつ追加採用")
            if len(matched) >= max_items:
                break

    adopted_ids = {item.strategy_id for item in matched}
    for audit in LAST_MATCH_AUDIT:
        if audit.get("matched") and audit.get("strategy_id") not in adopted_ids and not audit.get("non_adoption_reason"):
            audit["adopted"] = False
            audit["non_adoption_reason"] = "最大表示件数外"
    return matched[:max_items]


def evaluate_payload_item(current: pd.DataFrame, item: dict[str, Any]) -> tuple[BettingRecommendation | None, dict[str, Any]]:
    strategy_id = str(item.get("strategy_id") or item.get("label") or "")
    audit = {
        "strategy_id": strategy_id,
        "recommendation_kind": str(item.get("recommendation_kind") or item.get("kind") or ""),
        "ticket_type": str(item.get("ticket_type") or ""),
        "label": str(item.get("label") or ""),
        "json_conditions": item.get("condition_labels") or item.get("conditions") or item.get("role_pattern") or [],
        "matched_conditions": [],
        "unmatched_conditions": [],
        "target_horses": [],
        "generated_tickets": [],
        "purchase_points": 0,
        "matched": False,
        "adopted": False,
        "adopted_reason": "",
        "non_adoption_reason": "",
    }
    if not is_displayable_payload_item(item):
        audit["non_adoption_reason"] = "表示基準外"
        return None, audit
    kind = str(item.get("recommendation_kind") or item.get("kind") or "")
    if not (kind == "ticket_strategy" or item.get("role_pattern")):
        audit["non_adoption_reason"] = "買い方セクションでは購入条件JSONを直接表示しない"
        return None, audit
    recommendation = match_ticket_strategy_item(current, item, base_audit=audit)
    if recommendation is None:
        return None, audit
    return recommendation, dict(recommendation.audit)


def recommendation_priority_key(item: BettingRecommendation) -> tuple[Any, ...]:
    risk_order = {"正式": 0, "参考": 1, "高リスク": 2}
    score = to_float(item.audit.get("reliability_score")) or 0.0
    roi = item.expected_roi or 0.0
    hit = item.hit_rate or 0.0
    points = item.ticket_count or 999
    return (
        risk_order.get(item.risk_label, 3),
        -score,
        -item.sample_races,
        -roi,
        -hit,
        points,
    )


def _mark_audit_adopted(strategy_id: str, reason: str) -> None:
    for audit in LAST_MATCH_AUDIT:
        if audit.get("strategy_id") == strategy_id:
            audit["adopted"] = True
            audit["adopted_reason"] = reason
            audit["non_adoption_reason"] = ""


def _mark_audit_not_adopted(strategy_id: str, reason: str) -> None:
    for audit in LAST_MATCH_AUDIT:
        if audit.get("strategy_id") == strategy_id and not audit.get("adopted"):
            audit["non_adoption_reason"] = reason


def _with_adopted_reason(item: BettingRecommendation, reason: str) -> BettingRecommendation:
    audit = dict(item.audit)
    audit["adopted"] = True
    audit["adopted_reason"] = reason
    return BettingRecommendation(
        ticket_type=item.ticket_type,
        label=item.label,
        stars=item.stars,
        expected_roi=item.expected_roi,
        condition=item.condition,
        reason=item.reason,
        source=item.source,
        risk_label=item.risk_label,
        strategy_id=item.strategy_id,
        hit_rate=item.hit_rate,
        sample_races=item.sample_races,
        ticket_count=item.ticket_count,
        tickets=item.tickets,
        ticket_numbers=item.ticket_numbers,
        ticket_horses=item.ticket_horses,
        matched_conditions=item.matched_conditions,
        unmatched_conditions=item.unmatched_conditions,
        adopted_reason=reason,
        audit=audit,
    )


def is_displayable_payload_item(item: dict[str, Any]) -> bool:
    risk = str(item.get("risk_label") or item.get("ranking_type") or "")
    roi = to_float(item.get("return_rate"))
    if roi is None:
        roi = max(to_float(item.get("win_roi")) or 0.0, to_float(item.get("place_roi")) or 0.0)
    if roi <= 0:
        return False
    if risk == "高リスク" and roi < 120:
        return False
    return True


def match_payload_item(current: pd.DataFrame, item: dict[str, Any]) -> BettingRecommendation | None:
    kind = str(item.get("recommendation_kind") or item.get("kind") or "")
    if kind == "ticket_strategy" or item.get("role_pattern"):
        return match_ticket_strategy_item(current, item)
    return None


def match_ticket_strategy_item(
    current: pd.DataFrame,
    item: dict[str, Any],
    *,
    base_audit: dict[str, Any] | None = None,
) -> BettingRecommendation | None:
    ticket_type = str(item.get("ticket_type") or "")
    pattern = item.get("role_pattern")
    if not ticket_type or not isinstance(pattern, dict):
        return None
    tickets = build_tickets_for_race(current, pattern, ticket_type)
    matched_conditions, unmatched_conditions = ticket_condition_status(current, pattern, ticket_type, tickets)
    audit = dict(base_audit or {})
    audit.update(
        {
            "strategy_id": str(item.get("strategy_id") or item.get("label") or ""),
            "recommendation_kind": "ticket_strategy",
            "ticket_type": ticket_type,
            "label": str(item.get("label") or ""),
            "json_conditions": item.get("condition_labels") or pattern,
            "matched_conditions": list(matched_conditions),
            "unmatched_conditions": list(unmatched_conditions),
            "generated_tickets": format_ticket_lines(tickets, ticket_type),
            "purchase_points": len(tickets),
            "target_horses": ticket_horse_labels(current, tickets),
            "reliability_score": to_float(item.get("reliability_score")) or 0.0,
            "matched": False,
        }
    )
    if not tickets:
        audit["non_adoption_reason"] = "現在レースで買い目が作れない"
        if base_audit is not None:
            base_audit.update(audit)
        return None
    min_points, max_points = POINT_LIMITS.get(ticket_type, (1, 999))
    if len(tickets) < min_points:
        audit["unmatched_conditions"].append(f"購入点数が不足（{len(tickets)}点）")
        audit["non_adoption_reason"] = "必要点数不足"
        if base_audit is not None:
            base_audit.update(audit)
        return None
    if len(tickets) > max_points:
        audit["unmatched_conditions"].append(f"購入点数が多すぎます（{len(tickets)}点）")
        audit["non_adoption_reason"] = "購入点数上限超過"
        if base_audit is not None:
            base_audit.update(audit)
        return None
    picks = format_tickets(tickets, ticket_type)
    roi = to_float(item.get("return_rate"))
    risk = str(item.get("risk_label") or "")
    condition = str(item.get("label") or "")
    hit_rate = to_float(item.get("hit_rate"))
    sample_races = int(to_float(item.get("purchase_races")) or 0)
    note = str(item.get("current_odds_note") or "保存HTMLのレース前オッズ基準。最終オッズで条件外となる可能性があります。")
    audit["matched"] = True
    reason = (
        f"{risk} / 過去実績 {sample_races}R"
        f" / 的中率 {hit_rate or 0:.1f}%"
        f" / 買い目 {picks}"
        f" / {note}"
    )
    return BettingRecommendation(
        ticket_type=ticket_type,
        label=str(item.get("label") or item.get("strategy_id") or ""),
        stars=str(item.get("stars") or stars_for_roi(roi)),
        expected_roi=roi,
        condition=condition,
        reason=reason,
        source="analysis_json",
        risk_label=risk,
        strategy_id=str(item.get("strategy_id") or item.get("label") or ""),
        hit_rate=hit_rate,
        sample_races=sample_races,
        ticket_count=len(tickets),
        tickets=tuple(format_ticket_lines(tickets, ticket_type)),
        ticket_numbers=tuple(sorted(tickets, key=ticket_sort_key)),
        ticket_horses=tuple(ticket_horse_labels(current, tickets)),
        matched_conditions=tuple(matched_conditions),
        unmatched_conditions=tuple(unmatched_conditions),
        audit=audit,
    )


def match_purchase_condition_item(current: pd.DataFrame, item: dict[str, Any]) -> BettingRecommendation | None:
    specs = [ConditionSpec.from_dict(spec) for spec in item.get("conditions", []) if isinstance(spec, dict)]
    if not specs:
        return None
    mask = pd.Series(True, index=current.index)
    for spec in specs:
        mask &= condition_mask(current, spec)
    matched = current[mask].copy()
    if matched.empty:
        return None
    labels = [str(label) for label in item.get("condition_labels", [])] or [spec.label for spec in specs]
    horses = " / ".join(horse_label(row) for _, row in matched.iterrows())
    roi = to_float(item.get("win_roi") if item.get("ticket_type", "").startswith("単勝") else item.get("place_roi"))
    if roi is None:
        roi = max(to_float(item.get("win_roi")) or 0, to_float(item.get("place_roi")) or 0)
    return BettingRecommendation(
        ticket_type=str(item.get("ticket_type") or "条件一致"),
        label=" × ".join(labels),
        stars=str(item.get("stars") or stars_for_roi(roi)),
        expected_roi=roi,
        condition=" / ".join(labels),
        reason=(
            f"一致馬 {horses} / 対象{int(to_float(item.get('target_races')) or 0)}R"
            f" / 単勝回収率{to_float(item.get('win_roi')) or 0:.0f}%"
            f" / 複勝回収率{to_float(item.get('place_roi')) or 0:.0f}%"
        ),
        source="analysis_json",
        risk_label=str(item.get("ranking_type") or ""),
        strategy_id=str(item.get("strategy_id") or "purchase_condition"),
    )


def ticket_condition_status(
    current: pd.DataFrame,
    pattern: dict[str, Any],
    ticket_type: str,
    tickets: set[tuple[str, ...]],
) -> tuple[list[str], list[str]]:
    matched: list[str] = []
    unmatched: list[str] = []
    kind = str(pattern.get("type", ""))

    def role_line(role: str) -> None:
        count = len(unique_nums(role_numbers(current, [role])))
        label = role_label(role)
        if count > 0:
            matched.append(f"{label}が{count}頭")
        else:
            unmatched.append(f"{label}が不在")

    if kind == "single":
        for role in pattern.get("roles", []):
            role_line(str(role))
    elif kind == "pair":
        for role in [*pattern.get("left_roles", []), *pattern.get("right_roles", [])]:
            role_line(str(role))
        if tickets:
            matched.append(f"{ticket_type}の組み合わせが成立")
        else:
            unmatched.append(f"{ticket_type}の組み合わせが不成立")
    elif kind == "exacta":
        for role in [*pattern.get("first_roles", []), *pattern.get("second_roles", [])]:
            role_line(str(role))
        if tickets:
            matched.append("馬単の順序付き組み合わせが成立")
        else:
            unmatched.append("馬単の組み合わせが不成立")
    elif kind == "box":
        roles = [str(role) for role in pattern.get("roles", [])]
        nums = unique_nums(role_numbers(current, roles))
        size = int(pattern.get("size", 3) or 3)
        if len(nums) >= size:
            matched.append(f"BOX対象が{len(nums)}頭")
        else:
            unmatched.append(f"BOX対象が{size}頭未満")
        for role in roles:
            role_line(role)
    elif kind == "trifecta":
        for role in [*pattern.get("first_roles", []), *pattern.get("second_roles", []), *pattern.get("third_roles", [])]:
            role_line(str(role))
        if tickets:
            matched.append("三連単の順序付き組み合わせが成立")
        else:
            unmatched.append("三連単の組み合わせが不成立")
    else:
        unmatched.append("未対応の買い方条件")
    return matched, unmatched


def role_label(role: str) -> str:
    if role.startswith("AI"):
        rank = role.replace("AI", "")
        return f"AI{rank}位"
    return role


def format_ticket_lines(tickets: set[tuple[str, ...]], ticket_type: str) -> list[str]:
    sep = "→" if ticket_type in {"馬単", "三連単"} else "-"
    return [sep.join(ticket) for ticket in sorted(tickets, key=ticket_sort_key)]


def ticket_sort_key(ticket: tuple[str, ...]) -> tuple[int, ...]:
    out: list[int] = []
    for value in ticket:
        try:
            out.append(int(value))
        except (TypeError, ValueError):
            out.append(999)
    return tuple(out)


def ticket_horse_labels(current: pd.DataFrame, tickets: set[tuple[str, ...]]) -> list[str]:
    numbers = unique_nums(no for ticket in tickets for no in ticket)
    by_no = {str(row.get("horse_no_eval")): horse_label(row) for _, row in current.iterrows()}
    return [by_no.get(no, no) for no in numbers]


def adoption_map_from_recommendations(recommendations: list[BettingRecommendation]) -> dict[str, list[str]]:
    mapping: dict[str, list[str]] = {}
    for item in recommendations:
        label = f"{item.ticket_type} {item.label}".strip()
        for ticket in item.ticket_numbers:
            for no in ticket:
                mapping.setdefault(str(no), [])
                if label not in mapping[str(no)]:
                    mapping[str(no)].append(label)
    return mapping


def build_fixed_betting_recommendations(table: pd.DataFrame, *, max_items: int = 3) -> list[BettingRecommendation]:
    rows = [dict(row) for row in table.to_dict("records")]
    recommendations: list[BettingRecommendation] = []

    ai3 = _find_by_ai_rank(rows, 3)
    if ai3 and _odds_in_range(ai3, 8.0, 20.0):
        rule = RECOMMENDATION_RULES["win_ai3_value"]
        recommendations.append(_build(rule, f"{_horse_label(ai3)}がAI3位かつ実測妙味帯です。"))

    ai1 = _find_by_ai_rank(rows, 1)
    if ai1 and _odds_in_range(ai1, 8.0, 20.0):
        rule = RECOMMENDATION_RULES["win_ai1_value"]
        recommendations.append(_build(rule, f"{_horse_label(ai1)}がAI1位で、単勝妙味帯に入っています。"))

    ss = _rows_by_group(rows, "SS")
    group_a = _rows_by_group(rows, "A")
    group_b = _rows_by_group(rows, "B")
    group_c = _rows_by_group(rows, "C")
    if ss and group_c:
        rule = RECOMMENDATION_RULES["wide_ss_c"]
        recommendations.append(_build(rule, f"{_horse_label(ss[0])}からC穴候補をワイドで確認。"))
    if ss and group_a and (group_b or group_c):
        rule = RECOMMENDATION_RULES["trio_ss_a_b_c"]
        recommendations.append(_build(rule, "軸・相手本線に押さえ/穴を絡める実測上位パターンです。"))
    return recommendations[:max_items]


def _build(rule: dict[str, Any], reason: str) -> BettingRecommendation:
    roi = to_float(rule.get("expected_roi"))
    return BettingRecommendation(
        ticket_type=str(rule["ticket_type"]),
        label=str(rule["label"]),
        stars=stars_for_roi(roi),
        expected_roi=roi,
        condition=str(rule["condition"]),
        reason=reason,
        source="fixed",
    )


def format_tickets(tickets: set[tuple[str, ...]], ticket_type: str) -> str:
    sample = sorted(tickets, key=ticket_sort_key)[:5]
    sep = "→" if ticket_type in {"馬単", "三連単"} else "-"
    labels = [sep.join(ticket) for ticket in sample]
    if len(tickets) > len(sample):
        labels.append(f"ほか{len(tickets) - len(sample)}点")
    return " / ".join(labels)


def horse_label(row: pd.Series) -> str:
    no = str(row.get("horse_no_eval") or "").strip()
    name = str(row.get("horse_name_eval") or "").strip()
    mark = str(row.get("mark_eval") or "").strip()
    return " ".join(part for part in [no, mark, name] if part)


def _find_by_ai_rank(rows: list[dict[str, Any]], rank: int) -> dict[str, Any] | None:
    for row in rows:
        value = to_float(_pick(row, "ai_rank", "AI順位", "AI点順位"))
        if value is not None and int(value) == rank:
            return row
    return None


def _rows_by_group(rows: list[dict[str, Any]], group: str) -> list[dict[str, Any]]:
    return [row for row in rows if _text(_pick(row, "display_group", "グループ", "勢力図グループ")) == group]


def _odds_in_range(row: dict[str, Any], low: float, high: float) -> bool:
    odds = to_float(_pick(row, "単勝オッズ", "オッズ", "odds"))
    return odds is not None and low <= odds <= high


def stars_for_roi(roi: float | None) -> str:
    if roi is None:
        return "★★☆☆☆"
    if roi >= 160:
        return "★★★★★"
    if roi >= 130:
        return "★★★★☆"
    if roi >= 105:
        return "★★★☆☆"
    if roi >= 80:
        return "★★☆☆☆"
    return "★☆☆☆☆"


def _horse_label(row: dict[str, Any]) -> str:
    no = _text(_pick(row, "馬番", "馬", "horse_no"))
    name = _text(_pick(row, "馬名", "horse_name"))
    return " ".join(part for part in [no, name] if part)


def _pick(row: dict[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and not _is_missing(row.get(name)):
            return row.get(name)
    return None


def _is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return _text(value).lower() in {"", "-", "—", "nan", "none", "null"}


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()

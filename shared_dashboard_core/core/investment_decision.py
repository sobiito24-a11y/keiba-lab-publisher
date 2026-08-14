# -*- coding: utf-8 -*-
from __future__ import annotations

import itertools
import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from .betting_recommendation import BettingRecommendation, build_fixed_betting_recommendations
from .final_betting_context import (
    build_final_betting_context,
    build_ticket_alignment,
    context_summary_lines,
    ticket_alignment_lines,
)
from .horse_trust import build_horse_trust_for_numbers, compact_trust_lines, trust_rows_to_audit_text
from .purchase_conditions import ASSETS_ANALYSIS_DIR, clean_text, enrich_current_table, horse_no, to_float
from .ticket_strategy_analysis import pair_key, unique_nums


JRA_STRATEGY_JSON = ASSETS_ANALYSIS_DIR / "jra_strategy_selection.json"
NAR_STRATEGY_JSON = ASSETS_ANALYSIS_DIR / "nar_strategy_selection.json"
JRA_FALLBACK_JSON = ASSETS_ANALYSIS_DIR / "betting_recommendations.json"
NAR_WORK_STRATEGY_JSON = (
    Path(__file__).resolve().parents[1]
    / "work"
    / "nar_betting_expectation_report"
    / "nar_strategy_selection_report.json"
)

STAKE_PER_POINT = 100
BUY_SCORE_MIN = 70.0
HOLD_SCORE_MIN = 50.0
BUY_ROI_MIN = 110.0
HOLD_ROI_MIN = 100.0
BUY_MIN_RACES = 8
BUY_MIN_HITS = 2
MAX_PAYOUT_CONTRIBUTION_LIMIT = 80.0

JUDGEMENT_BUY = "買い"
JUDGEMENT_HOLD = "保留"
JUDGEMENT_PASS = "見送り"


@dataclass(frozen=True)
class InvestmentDecision:
    race_type: str
    judgement: str
    selected: BettingRecommendation | None = None
    candidates: tuple[BettingRecommendation, ...] = ()
    audit_rows: tuple[dict[str, Any], ...] = ()
    reason_lines: tuple[str, ...] = ()
    source_path: str = ""
    source_race_count: int = 0
    updated_at: str = ""
    source_note: str = ""
    fallback_used: bool = False
    total_stake: int = 0
    target_horses: tuple[str, ...] = ()
    horse_trust: tuple[dict[str, Any], ...] = ()
    horse_trust_summary: tuple[str, ...] = ()
    ticket_rationale: dict[str, Any] = field(default_factory=dict)
    final_betting_context: tuple[dict[str, Any], ...] = ()
    final_context_summary: tuple[str, ...] = ()
    ticket_alignment: tuple[dict[str, Any], ...] = ()
    ticket_alignment_summary: tuple[str, ...] = ()
    logic_version: str = "v3"
    decision_v4: str = ""
    axis_score: float = 0.0
    axis_confidence_v4: str = ""
    ticket_candidate_score: float = 0.0
    ticket_veto_reason: str = ""
    practical_decision: str = ""
    practical_reason: str = ""
    practical_reason_lines: tuple[str, ...] = ()
    practical_config_version: str = ""
    honmei_horse_no: str = ""
    honmei_horse_name: str = ""


def build_investment_decision(
    table: Any,
    race_mode: str,
    *,
    json_paths: list[Path] | None = None,
    race_info: Mapping[str, Any] | None = None,
    prediction_logic_version: str = "v3",
) -> InvestmentDecision:
    """Select exactly one betting strategy for display.

    This layer consumes audit JSON and current-race rows only.  It deliberately
    does not alter AI scores, marks, groups, parsers, or existing prediction
    outputs.
    """

    race_type = "nar" if str(race_mode).lower() == "nar" else "jra"
    if str(prediction_logic_version).lower() == "practical":
        from .practical_mode import build_practical_decision

        return build_practical_decision(table, race_type, race_info=race_info)
    if str(prediction_logic_version).lower() in {"v4", "v4.1"}:
        from .ver4_engine import build_ver4_investment_decision, evaluate_ver4_table

        current = table
        if isinstance(table, pd.DataFrame) and "horse_score_v4" not in table.columns:
            current = evaluate_ver4_table(table, race_type, race_info)
        return build_ver4_investment_decision(current, race_type)
    if table is None or not isinstance(table, pd.DataFrame) or table.empty:
        return InvestmentDecision(
            race_type=race_type,
            judgement=JUDGEMENT_PASS,
            reason_lines=("出走馬データがないため馬券提案を作成できません。",),
        )

    payload, diagnostic = load_strategy_payload(race_type, json_paths=json_paths)
    if payload is None:
        return build_missing_json_fallback(table, race_type, diagnostic, race_info=race_info)

    current = enrich_current_table(table)
    strategies = payload.get("strategies") or payload.get("recommendations") or []
    if not isinstance(strategies, list):
        strategies = []

    source = payload.get("source") or {}
    source_race_count = int(to_float(source.get("race_count") or payload.get("race_count")) or 0)
    updated_at = str(payload.get("updated_at") or source.get("updated_at") or "")
    source_note = str(source.get("note") or payload.get("note") or "")

    candidates: list[BettingRecommendation] = []
    audit_rows: list[dict[str, Any]] = []
    for item in strategies:
        if not isinstance(item, dict):
            continue
        recommendation, audit = evaluate_strategy(current, item, race_type, payload)
        audit["analysis_json"] = diagnostic.get("path", "")
        audit["race_type"] = race_type.upper()
        audit_rows.append(audit)
        if recommendation is not None and audit.get("judgement") in {JUDGEMENT_BUY, JUDGEMENT_HOLD}:
            candidates.append(recommendation)

    candidates = sorted(candidates, key=strategy_priority_key)
    selected = candidates[0] if candidates else None
    for index, audit in enumerate(audit_rows, start=1):
        audit["adoption_rank"] = ""
        audit["final_selected_strategy"] = selected.strategy_id if selected else ""
        if selected and audit.get("strategy_id") == selected.strategy_id:
            audit["adoption_rank"] = 1
            audit["adopted"] = True
            audit["adopted_reason"] = "成立候補の中で戦略スコアと優先順位が最上位"
        elif audit.get("judgement") in {JUDGEMENT_BUY, JUDGEMENT_HOLD}:
            audit["adopted"] = False
            audit["non_adoption_reason"] = "より優先度の高い戦略を1件だけ採用"

    if selected is None:
        return InvestmentDecision(
            race_type=race_type,
            judgement=JUDGEMENT_PASS,
            audit_rows=tuple(audit_rows),
            reason_lines=pass_reason_lines(audit_rows),
            source_path=str(diagnostic.get("path", "")),
            source_race_count=source_race_count,
            updated_at=updated_at,
            source_note=source_note,
        )

    selected_numbers = unique_nums(no for ticket in selected.ticket_numbers for no in ticket)
    horse_trust = build_horse_trust_for_numbers(current, race_type, selected_numbers)
    horse_trust_summary = compact_trust_lines(horse_trust)
    final_context = build_final_betting_context(current, race_type, ticket_numbers=selected_numbers, race_info=race_info)
    final_context_summary = context_summary_lines(final_context)
    ticket_alignment = build_ticket_alignment(
        final_context,
        selected.ticket_numbers,
        strategy_id=selected.strategy_id,
        strategy_label=selected.label,
    )
    ticket_alignment_summary = ticket_alignment_lines(ticket_alignment)
    ticket_rationale = build_ticket_rationale(selected)
    selected_audit = {
        **selected.audit,
        "horse_trust": list(horse_trust),
        "horse_trust_summary": list(horse_trust_summary),
        "horse_trust_audit_text": trust_rows_to_audit_text(horse_trust),
        "final_betting_context": list(final_context),
        "final_context_summary": list(final_context_summary),
        "ticket_alignment": list(ticket_alignment),
        "ticket_alignment_summary": list(ticket_alignment_summary),
        "ticket_rationale": ticket_rationale,
    }
    selected = with_audit(selected, selected_audit)
    audit_rows = attach_selected_context_to_audit_rows(
        audit_rows,
        selected.strategy_id,
        horse_trust,
        horse_trust_summary,
        final_context,
        final_context_summary,
        ticket_alignment,
        ticket_alignment_summary,
        ticket_rationale,
    )

    judgement = selected.audit.get("judgement") or JUDGEMENT_HOLD
    total_stake = selected.ticket_count * STAKE_PER_POINT
    return InvestmentDecision(
        race_type=race_type,
        judgement=str(judgement),
        selected=selected,
        candidates=tuple(candidates),
        audit_rows=tuple(audit_rows),
        reason_lines=tuple(selected.audit.get("adopted_reason_lines") or selected.audit.get("matched_conditions") or ()),
        source_path=str(diagnostic.get("path", "")),
        source_race_count=source_race_count,
        updated_at=updated_at,
        source_note=source_note,
        total_stake=total_stake,
        target_horses=tuple(selected.ticket_horses),
        horse_trust=horse_trust,
        horse_trust_summary=horse_trust_summary,
        ticket_rationale=ticket_rationale,
        final_betting_context=final_context,
        final_context_summary=final_context_summary,
        ticket_alignment=ticket_alignment,
        ticket_alignment_summary=ticket_alignment_summary,
    )


def load_strategy_payload(
    race_type: str,
    *,
    json_paths: list[Path] | None = None,
) -> tuple[dict[str, Any] | None, dict[str, Any]]:
    paths = json_paths if json_paths is not None else default_strategy_paths(race_type)
    checked: list[dict[str, Any]] = []
    for path in paths:
        path = Path(path)
        if not path.exists():
            checked.append({"path": str(path), "status": "missing"})
            continue
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            return None, {
                "status": "parse_error",
                "path": str(path),
                "reason": f"analysis json parse failed: {exc}",
                "checked": checked,
            }
        return payload, {
            "status": "loaded",
            "path": str(path),
            "checked": checked,
        }
    return None, {"status": "missing", "path": "", "reason": "analysis json not found", "checked": checked}


def default_strategy_paths(race_type: str) -> list[Path]:
    if race_type == "nar":
        return [NAR_STRATEGY_JSON, NAR_WORK_STRATEGY_JSON]
    return [JRA_STRATEGY_JSON, JRA_FALLBACK_JSON]


def build_missing_json_fallback(
    table: pd.DataFrame,
    race_type: str,
    diagnostic: dict[str, Any],
    *,
    race_info: Mapping[str, Any] | None = None,
) -> InvestmentDecision:
    if race_type != "jra":
        return InvestmentDecision(
            race_type=race_type,
            judgement=JUDGEMENT_PASS,
            reason_lines=("分析JSONが見つからないため、地方は固定提案へフォールバックしません。",),
            source_path=str(diagnostic.get("path", "")),
            fallback_used=True,
        )

    fixed = build_fixed_betting_recommendations(table, max_items=1)
    if not fixed:
        return InvestmentDecision(
            race_type=race_type,
            judgement=JUDGEMENT_PASS,
            reason_lines=("分析JSONが見つからず、固定フォールバック条件にも一致しません。",),
            fallback_used=True,
        )
    selected = fixed[0]
    audit = {
        "race_type": "JRA",
        "strategy_id": selected.strategy_id or "fixed_fallback",
        "ticket_type": selected.ticket_type,
        "strategy_name": selected.label,
        "analysis_json": "",
        "judgement": JUDGEMENT_HOLD,
        "adopted": True,
        "adopted_reason": "分析JSON欠損時のみ固定フォールバック",
        "generated_tickets": list(selected.tickets),
        "purchase_points": selected.ticket_count,
        "total_stake": selected.ticket_count * STAKE_PER_POINT,
    }
    current = enrich_current_table(table)
    selected_numbers = unique_nums(no for ticket in selected.ticket_numbers for no in ticket)
    horse_trust = build_horse_trust_for_numbers(current, race_type, selected_numbers)
    horse_trust_summary = compact_trust_lines(horse_trust)
    final_context = build_final_betting_context(current, race_type, ticket_numbers=selected_numbers, race_info=race_info)
    final_context_summary = context_summary_lines(final_context)
    ticket_alignment = build_ticket_alignment(
        final_context,
        selected.ticket_numbers,
        strategy_id=selected.strategy_id,
        strategy_label=selected.label,
    )
    ticket_alignment_summary = ticket_alignment_lines(ticket_alignment)
    ticket_rationale = build_ticket_rationale(selected)
    audit = {
        **audit,
        "horse_trust": list(horse_trust),
        "horse_trust_summary": list(horse_trust_summary),
        "horse_trust_audit_text": trust_rows_to_audit_text(horse_trust),
        "final_betting_context": list(final_context),
        "final_context_summary": list(final_context_summary),
        "ticket_alignment": list(ticket_alignment),
        "ticket_alignment_summary": list(ticket_alignment_summary),
        "ticket_rationale": ticket_rationale,
    }
    selected = with_audit(selected, audit)
    return InvestmentDecision(
        race_type=race_type,
        judgement=JUDGEMENT_HOLD,
        selected=selected,
        candidates=(selected,),
        audit_rows=(audit,),
        reason_lines=("分析JSON欠損時のみ、従来の固定条件を参考表示しています。",),
        fallback_used=True,
        total_stake=selected.ticket_count * STAKE_PER_POINT,
        target_horses=tuple(selected.ticket_horses),
        horse_trust=horse_trust,
        horse_trust_summary=horse_trust_summary,
        ticket_rationale=ticket_rationale,
        final_betting_context=final_context,
        final_context_summary=final_context_summary,
        ticket_alignment=ticket_alignment,
        ticket_alignment_summary=ticket_alignment_summary,
    )


def evaluate_strategy(
    current: pd.DataFrame,
    item: dict[str, Any],
    race_type: str,
    payload: dict[str, Any],
) -> tuple[BettingRecommendation | None, dict[str, Any]]:
    strategy_id = str(item.get("strategy_id") or item.get("label") or "")
    ticket_type = str(item.get("ticket_type") or "")
    label = str(item.get("label") or item.get("strategy_label") or strategy_id)
    pattern = item.get("role_pattern") or item.get("pattern") or {}

    tickets = build_current_tickets(current, pattern, ticket_type)
    role_matched, role_unmatched = readable_ticket_conditions(current, pattern, ticket_type, tickets)
    required_conditions = normalize_conditions(item.get("conditions"))
    avoid_conditions = normalize_conditions(item.get("avoid_conditions"))

    matched_conditions = list(role_matched)
    unmatched_conditions = list(role_unmatched)
    avoid_matched: list[str] = []
    avoid_unmatched: list[str] = []

    for condition in required_conditions:
        ok, label_text, _targets = evaluate_condition(current, condition)
        if ok:
            matched_conditions.append(label_text)
        else:
            unmatched_conditions.append(label_text)
    for condition in avoid_conditions:
        ok, label_text, _targets = evaluate_condition(current, condition)
        if ok:
            avoid_matched.append(label_text)
        else:
            avoid_unmatched.append(label_text)

    min_points, max_points = point_limits(ticket_type)
    if len(tickets) < min_points:
        unmatched_conditions.append(f"必要点数未満（{len(tickets)}点）")
    if len(tickets) > max_points:
        unmatched_conditions.append(f"購入点数上限超過（{len(tickets)}点）")

    score = strategy_score(item, len(tickets))
    roi = to_float(item.get("return_rate"))
    hit_rate = to_float(item.get("hit_rate"))
    sample_races = int(to_float(item.get("target_races") or item.get("purchase_races") or item.get("sample_races")) or 0)
    hits = int(to_float(item.get("hits")) or 0)
    max_losing = to_float(item.get("max_losing_streak"))
    max_drawdown_value = to_float(item.get("max_drawdown"))
    dependency = max_payout_dependency(item)
    risk = str(item.get("risk_label") or item.get("ranking_type") or "正式")
    formal = risk in {"正式", "official", "正式推奨"} or bool(item.get("official", False))

    judgement, reason = judgement_for_strategy(
        formal=formal,
        score=score,
        sample_races=sample_races,
        roi=roi,
        hits=hits,
        dependency=dependency,
        unmatched_conditions=unmatched_conditions,
        avoid_matched=avoid_matched,
        ticket_count=len(tickets),
    )

    ticket_lines = format_ticket_lines(tickets, ticket_type)
    target_horses = ticket_horse_labels(current, tickets)
    audit = {
        "race_type": race_type.upper(),
        "strategy_id": strategy_id,
        "ticket_type": ticket_type,
        "strategy_name": label,
        "analysis_json": "",
        "analysis_race_count": (payload.get("source") or {}).get("race_count") or payload.get("race_count"),
        "conditions_json": required_conditions,
        "matched_conditions": matched_conditions,
        "unmatched_conditions": unmatched_conditions,
        "avoid_conditions": [condition_label(condition) for condition in avoid_conditions],
        "avoid_matched": avoid_matched,
        "avoid_unmatched": avoid_unmatched,
        "target_horses": target_horses,
        "generated_tickets": ticket_lines,
        "purchase_points": len(tickets),
        "total_stake": len(tickets) * STAKE_PER_POINT,
        "return_rate": roi,
        "hit_rate": hit_rate,
        "sample_races": sample_races,
        "hits": hits,
        "max_losing_streak": max_losing,
        "max_drawdown": max_drawdown_value,
        "max_payout_contribution": dependency,
        "reliability_score": to_float(item.get("reliability_score") or item.get("selection_score")),
        "strategy_score": score,
        "selection_rank": int(to_float(item.get("selection_rank")) or 999),
        "risk_label": risk,
        "judgement": judgement,
        "adopted": False,
        "adoption_rank": "",
        "adopted_reason": "",
        "non_adoption_reason": reason if judgement == JUDGEMENT_PASS else "",
        "final_selected_strategy": "",
    }
    if judgement == JUDGEMENT_PASS:
        return None, audit

    reason_text = (
        f"{risk} / {sample_races}R / 的中率{hit_rate or 0:.1f}% / "
        f"回収率{roi or 0:.1f}% / {len(tickets)}点"
    )
    recommendation = BettingRecommendation(
        ticket_type=ticket_type,
        label=label,
        stars=stars_for_strategy(score),
        expected_roi=roi,
        condition=" / ".join(matched_conditions),
        reason=reason_text,
        source="strategy_selection_json",
        risk_label=risk,
        strategy_id=strategy_id,
        hit_rate=hit_rate,
        sample_races=sample_races,
        ticket_count=len(tickets),
        tickets=tuple(ticket_lines),
        ticket_numbers=tuple(sorted(tickets, key=ticket_sort_key)),
        ticket_horses=tuple(target_horses),
        matched_conditions=tuple(matched_conditions),
        unmatched_conditions=tuple(unmatched_conditions),
        adopted_reason="",
        audit={**audit, "adopted_reason_lines": matched_conditions},
    )
    return recommendation, audit


def build_ticket_rationale(item: BettingRecommendation) -> dict[str, Any]:
    audit = item.audit or {}
    return {
        "strategy": item.label,
        "ticket_type": item.ticket_type,
        "sample_races": item.sample_races,
        "hits": int(to_float(audit.get("hits")) or 0),
        "hit_rate": item.hit_rate,
        "return_rate": item.expected_roi,
        "max_losing_streak": audit.get("max_losing_streak"),
        "max_drawdown": audit.get("max_drawdown"),
        "max_payout_contribution": audit.get("max_payout_contribution"),
        "strategy_score": audit.get("strategy_score"),
        "risk_label": item.risk_label,
        "ticket_count": item.ticket_count,
    }


def attach_selected_context_to_audit_rows(
    audit_rows: list[dict[str, Any]],
    strategy_id: str,
    horse_trust: tuple[dict[str, Any], ...],
    horse_trust_summary: tuple[str, ...],
    final_context: tuple[dict[str, Any], ...],
    final_context_summary: tuple[str, ...],
    ticket_alignment: tuple[dict[str, Any], ...],
    ticket_alignment_summary: tuple[str, ...],
    ticket_rationale: dict[str, Any],
) -> tuple[dict[str, Any], ...]:
    updated: list[dict[str, Any]] = []
    for row in audit_rows:
        item = dict(row)
        if item.get("strategy_id") == strategy_id:
            item["horse_trust"] = list(horse_trust)
            item["horse_trust_summary"] = list(horse_trust_summary)
            item["horse_trust_audit_text"] = trust_rows_to_audit_text(horse_trust)
            item["final_betting_context"] = list(final_context)
            item["final_context_summary"] = list(final_context_summary)
            item["ticket_alignment"] = list(ticket_alignment)
            item["ticket_alignment_summary"] = list(ticket_alignment_summary)
            item["ticket_rationale"] = ticket_rationale
        updated.append(item)
    return tuple(updated)


def normalize_conditions(value: Any) -> list[dict[str, Any]]:
    if not isinstance(value, list):
        return []
    return [item for item in value if isinstance(item, dict)]


def judgement_for_strategy(
    *,
    formal: bool,
    score: float,
    sample_races: int,
    roi: float | None,
    hits: int,
    dependency: float | None,
    unmatched_conditions: list[str],
    avoid_matched: list[str],
    ticket_count: int,
) -> tuple[str, str]:
    if ticket_count <= 0:
        return JUDGEMENT_PASS, "買い目生成不可"
    if unmatched_conditions:
        return JUDGEMENT_PASS, "正式条件が現在レースで不成立"
    if avoid_matched:
        return JUDGEMENT_PASS, "回避条件に該当"
    roi_value = roi or 0.0
    dependency_value = dependency if dependency is not None else 0.0
    if (
        formal
        and score >= BUY_SCORE_MIN
        and sample_races >= BUY_MIN_RACES
        and roi_value >= BUY_ROI_MIN
        and hits >= BUY_MIN_HITS
        and dependency_value < MAX_PAYOUT_CONTRIBUTION_LIMIT
    ):
        return JUDGEMENT_BUY, ""
    if score >= HOLD_SCORE_MIN and roi_value >= HOLD_ROI_MIN and dependency_value < MAX_PAYOUT_CONTRIBUTION_LIMIT:
        return JUDGEMENT_HOLD, ""
    return JUDGEMENT_PASS, "回収率・サンプル数・再現性スコアの基準未満"


def strategy_score(item: dict[str, Any], ticket_count: int) -> float:
    base = to_float(item.get("strategy_score") or item.get("selection_score") or item.get("reliability_score")) or 0.0
    roi = to_float(item.get("return_rate")) or 0.0
    hit = to_float(item.get("hit_rate")) or 0.0
    sample = to_float(item.get("target_races") or item.get("purchase_races") or item.get("sample_races")) or 0.0
    max_losing = to_float(item.get("max_losing_streak")) or 0.0
    drawdown = to_float(item.get("max_drawdown")) or 0.0
    dependency = max_payout_dependency(item) or 0.0
    risk = str(item.get("risk_label") or item.get("ranking_type") or "")

    score = base
    if risk in {"正式", "official", "正式推奨"}:
        score += 8.0
    elif risk in {"参考", "reference"}:
        score -= 2.0
    elif risk in {"高リスク", "high_risk"}:
        score -= 14.0
    score += min(10.0, max(0.0, roi - 100.0) * 0.12)
    score += min(8.0, hit * 0.10)
    score += min(6.0, sample * 0.35)
    score -= max(0.0, max_losing - 5.0) * 2.0
    score -= max(0.0, dependency - 45.0) * 0.25
    score -= max(0.0, drawdown - 2000.0) / 400.0
    score -= max(0, ticket_count - 4) * 1.5
    return round(max(0.0, min(100.0, score)), 1)


def max_payout_dependency(item: dict[str, Any]) -> float | None:
    for key in ["max_payout_contribution", "high_payout_dependency"]:
        value = to_float(item.get(key))
        if value is not None:
            return value
    return None


def strategy_priority_key(item: BettingRecommendation) -> tuple[Any, ...]:
    judgement = str(item.audit.get("judgement") or "")
    judgement_order = {JUDGEMENT_BUY: 0, JUDGEMENT_HOLD: 1}.get(judgement, 2)
    score = to_float(item.audit.get("strategy_score")) or 0.0
    sample = int(to_float(item.audit.get("sample_races")) or 0)
    roi = to_float(item.audit.get("return_rate")) or 0.0
    hit = to_float(item.audit.get("hit_rate")) or 0.0
    points = int(to_float(item.audit.get("purchase_points")) or 999)
    rank = int(to_float(item.audit.get("selection_rank")) or 999)
    return (judgement_order, rank, -score, -sample, -roi, -hit, points)


def build_current_tickets(current: pd.DataFrame, pattern: Any, ticket_type: str) -> set[tuple[str, ...]]:
    if not isinstance(pattern, dict):
        return set()
    kind = str(pattern.get("type", ""))
    if kind == "single":
        nums = role_numbers_readable(current, list(pattern.get("roles", [])))
        return {(no,) for no in unique_nums(nums)}
    if kind == "pair":
        left = unique_nums(role_numbers_readable(current, list(pattern.get("left_roles", []))))
        right = unique_nums(role_numbers_readable(current, list(pattern.get("right_roles", []))))
        return {pair_key([a, b]) for a in left for b in right if a != b}
    if kind == "exacta":
        first = unique_nums(role_numbers_readable(current, list(pattern.get("first_roles", []))))
        second = unique_nums(role_numbers_readable(current, list(pattern.get("second_roles", []))))
        return {(a, b) for a in first for b in second if a != b}
    if kind == "box":
        nums = unique_nums(role_numbers_readable(current, list(pattern.get("roles", []))))
        size = int(pattern.get("size", 3) or 3)
        if canonical_ticket_type(ticket_type) == "三連単":
            return {tuple(combo) for combo in itertools.permutations(nums, size)}
        return {tuple(sorted(combo, key=int)) for combo in itertools.combinations(nums, size)}
    if kind == "trifecta":
        first = unique_nums(role_numbers_readable(current, list(pattern.get("first_roles", []))))
        second = unique_nums(role_numbers_readable(current, list(pattern.get("second_roles", []))))
        third = unique_nums(role_numbers_readable(current, list(pattern.get("third_roles", []))))
        return {(a, b, c) for a in first for b in second for c in third if len({a, b, c}) == 3}
    return set()


def readable_ticket_conditions(
    current: pd.DataFrame,
    pattern: Any,
    ticket_type: str,
    tickets: set[tuple[str, ...]],
) -> tuple[list[str], list[str]]:
    if not isinstance(pattern, dict):
        return [], ["買い方パターンが未定義"]
    matched: list[str] = []
    unmatched: list[str] = []
    roles: list[str] = []
    kind = str(pattern.get("type", ""))
    if kind == "single":
        roles = [str(role) for role in pattern.get("roles", [])]
    elif kind == "pair":
        roles = [str(role) for role in pattern.get("left_roles", [])] + [str(role) for role in pattern.get("right_roles", [])]
    elif kind == "exacta":
        roles = [str(role) for role in pattern.get("first_roles", [])] + [str(role) for role in pattern.get("second_roles", [])]
    elif kind == "box":
        roles = [str(role) for role in pattern.get("roles", [])]
    elif kind == "trifecta":
        roles = (
            [str(role) for role in pattern.get("first_roles", [])]
            + [str(role) for role in pattern.get("second_roles", [])]
            + [str(role) for role in pattern.get("third_roles", [])]
        )
    for role in roles:
        count = len(unique_nums(role_numbers_readable(current, [role])))
        if count:
            matched.append(f"{role}が{count}頭")
        else:
            unmatched.append(f"{role}が不在")
    if tickets:
        matched.append(f"{canonical_ticket_type(ticket_type)}の買い目生成")
    else:
        unmatched.append(f"{canonical_ticket_type(ticket_type)}の買い目生成不可")
    return matched, unmatched


def role_numbers_readable(group: pd.DataFrame, roles: Iterable[Any]) -> list[str]:
    nums: list[str] = []
    for role in roles:
        role_text = str(role)
        if role_text.startswith("AI"):
            rank = to_float(role_text.replace("AI", ""))
            if rank is None or "ai_rank_eval" not in group:
                continue
            subset = group[pd.to_numeric(group["ai_rank_eval"], errors="coerce").eq(rank)]
        else:
            if "display_group_eval" not in group:
                continue
            subset = group[group["display_group_eval"].astype(str).eq(role_text)]
        nums.extend(subset.get("horse_no_eval", pd.Series(dtype=object)).map(horse_no).tolist())
    return nums


def rows_for_role(group: pd.DataFrame, role: str) -> pd.DataFrame:
    if role.startswith("AI"):
        rank = to_float(role.replace("AI", ""))
        if rank is None or "ai_rank_eval" not in group:
            return group.iloc[0:0]
        return group[pd.to_numeric(group["ai_rank_eval"], errors="coerce").eq(rank)]
    if "display_group_eval" not in group:
        return group.iloc[0:0]
    return group[group["display_group_eval"].astype(str).eq(role)]


def evaluate_condition(current: pd.DataFrame, condition: dict[str, Any]) -> tuple[bool, str, list[str]]:
    label = condition_label(condition)
    role = str(condition.get("role") or "")
    field = str(condition.get("field") or "")
    operator = str(condition.get("op") or condition.get("kind") or "")
    subset = rows_for_role(current, role) if role else current
    if operator in {"exists", "present"}:
        return (not subset.empty), label, subset_numbers(subset)
    if field == "count":
        count = len(unique_nums(subset_numbers(subset)))
        return compare_number(float(count), condition), label, subset_numbers(subset)
    if subset.empty:
        return False, label, []

    values = [field_value(row, field) for _, row in subset.iterrows()]
    if operator in {"eq", "equals"}:
        target = clean_text(condition.get("value"))
        ok = any(clean_text(value) == target for value in values if clean_text(value))
    elif operator in {"in"}:
        targets = {clean_text(value) for value in condition.get("values", [])}
        ok = any(clean_text(value) in targets for value in values if clean_text(value))
    else:
        ok = any(compare_number(to_float(value), condition) for value in values)
    return ok, label, subset_numbers(subset)


def compare_number(value: float | None, condition: dict[str, Any]) -> bool:
    if value is None:
        return False
    operator = str(condition.get("op") or condition.get("kind") or "")
    low = to_float(condition.get("low"))
    high = to_float(condition.get("high"))
    include_high = bool(condition.get("include_high", True))
    target = to_float(condition.get("value"))
    if operator == "range":
        if low is not None and value < low:
            return False
        if high is not None:
            return value <= high if include_high else value < high
        return True
    if operator == "lt":
        threshold = high if high is not None else target
        return threshold is not None and value < threshold
    if operator == "lte":
        threshold = high if high is not None else target
        return threshold is not None and value <= threshold
    if operator == "gt":
        threshold = low if low is not None else target
        return threshold is not None and value > threshold
    if operator == "gte":
        threshold = low if low is not None else target
        return threshold is not None and value >= threshold
    if operator == "eq":
        return target is not None and value == target
    return False


def field_value(row: pd.Series, field: str) -> Any:
    names = {
        "odds": ["odds_eval", "オッズ", "単勝オッズ", "odds"],
        "popularity": ["popularity_eval", "人気", "popularity"],
        "ai_score": ["ai_score_eval", "AI点", "ai_score"],
        "ability_value": ["ability_value_eval", "能力評価値", "ability_value"],
        "momentum": ["momentum_rank", "勢いランク", "勢い"],
    }.get(field, [field])
    for name in names:
        if name in row:
            value = row.get(name)
            try:
                if pd.notna(value):
                    return value
            except TypeError:
                return value
    return None


def subset_numbers(subset: pd.DataFrame) -> list[str]:
    if subset.empty or "horse_no_eval" not in subset:
        return []
    return unique_nums(subset["horse_no_eval"].map(horse_no).tolist())


def condition_label(condition: dict[str, Any]) -> str:
    explicit = clean_text(condition.get("label"))
    if explicit:
        return explicit
    role = clean_text(condition.get("role"))
    field = clean_text(condition.get("field"))
    op = clean_text(condition.get("op") or condition.get("kind"))
    low = condition.get("low")
    high = condition.get("high")
    value = condition.get("value")
    if op == "range":
        return f"{role} {field} {low}～{high}"
    if value is not None:
        return f"{role} {field}={value}"
    return "条件"


def point_limits(ticket_type: str) -> tuple[int, int]:
    ticket = canonical_ticket_type(ticket_type)
    return {
        "単勝": (1, 2),
        "複勝": (1, 2),
        "ワイド": (1, 5),
        "馬連": (1, 5),
        "馬単": (1, 8),
        "三連複": (1, 15),
        "三連単": (1, 30),
    }.get(ticket, (1, 999))


def canonical_ticket_type(value: Any) -> str:
    text = str(value or "")
    if "単勝" in text or "蜊伜享" in text:
        return "単勝"
    if "複勝" in text or "隍" in text and "享" in text:
        return "複勝"
    if "ワイド" in text or "繝ｯ" in text:
        return "ワイド"
    if "馬連" in text or "騾" in text:
        return "馬連"
    if "馬単" in text or "蜊" in text and "鬥" in text:
        return "馬単"
    if "三連複" in text:
        return "三連複"
    if "三連単" in text:
        return "三連単"
    return text


def format_ticket_lines(tickets: set[tuple[str, ...]], ticket_type: str) -> list[str]:
    sep = "→" if canonical_ticket_type(ticket_type) in {"馬単", "三連単"} else "-"
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
    by_no: dict[str, str] = {}
    for _, row in current.iterrows():
        no = horse_no(row.get("horse_no_eval"))
        name = clean_text(row.get("horse_name_eval"))
        group = clean_text(row.get("display_group_eval"))
        mark = clean_text(row.get("mark_eval"))
        label = " ".join(part for part in [group, no, mark, name] if part)
        if no:
            by_no[no] = label or no
    return [by_no.get(no, no) for no in numbers]


def stars_for_strategy(score: float) -> str:
    if score >= 85:
        return "★★★★★"
    if score >= 72:
        return "★★★★☆"
    if score >= 58:
        return "★★★☆☆"
    if score >= 45:
        return "★★☆☆☆"
    return "★☆☆☆☆"


def confidence_label(score: float | None) -> str:
    value = score or 0.0
    if value >= 85:
        return "暫定S"
    if value >= 70:
        return "暫定A"
    if value >= 55:
        return "暫定B"
    return "暫定C"


def pass_reason_lines(audit_rows: list[dict[str, Any]]) -> tuple[str, ...]:
    if not audit_rows:
        return ("分析JSONに有効な戦略がありません。",)
    reasons = []
    if not any(row.get("matched_conditions") for row in audit_rows):
        reasons.append("正式購入条件の一致なし")
    if any(row.get("avoid_matched") for row in audit_rows):
        reasons.append("回避条件に該当")
    if any(row.get("unmatched_conditions") for row in audit_rows):
        reasons.append("必要な馬・条件が不足")
    if not reasons:
        reasons.append("回収率・サンプル数・再現性スコアの基準未満")
    return tuple(dict.fromkeys(reasons))


def with_audit(item: BettingRecommendation, audit: dict[str, Any]) -> BettingRecommendation:
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
        adopted_reason=item.adopted_reason,
        audit=audit,
    )

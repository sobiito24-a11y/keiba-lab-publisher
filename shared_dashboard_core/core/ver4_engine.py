# -*- coding: utf-8 -*-
"""Keiba AI Ver4 absolute-score prediction layer.

Ver3 notebook output is treated as immutable input.  This module only reads
existing race/horse facts and appends explicitly named ``*_v4`` columns to a
copy.  It never uses finishing results, payoffs, or within-race min/max
normalisation to calculate Horse Score.
"""
from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence
from typing import Any

import pandas as pd

from .condition_fit import canonical_condition_fit_level, evaluate_condition_fit
from .purchase_conditions import clean_text, horse_no, to_float


NAR_WEIGHTS = {
    "base_ability_score": 0.45,
    "condition_score": 0.30,
    "jockey_score": 0.08,
    "age_weight_score": 0.07,
    "momentum_score_v4": 0.07,
    "race_shape_score": 0.03,
}

JRA_WEIGHTS = {
    "base_ability_score": 0.40,
    "condition_score": 0.25,
    "jockey_score": 0.10,
    "age_weight_score": 0.08,
    "training_score": 0.08,
    "momentum_score_v4": 0.05,
    "race_shape_score": 0.04,
}

GROUP_THRESHOLDS = ((82.0, "SS"), (70.0, "A"), (58.0, "B"), (45.0, "C"))
INDEX_ANCHORS = {
    # Fixed historical scale anchors.  These are deliberately shared by every
    # horse in a race; race entrants never define each other's absolute score.
    # The NAR limits cover approximately the 1st--99th percentile (-39.6 to
    # 82.6) of 382 saved, result-free index observations in the supplied five
    # Prediction History files.  They are rounded outward and are not fitted
    # against finishing positions or payoffs.
    "nar": (-35.0, 85.0),
    "jra": (-20.0, 100.0),
}

V4_COMPONENT_COLUMNS = tuple(dict.fromkeys((*NAR_WEIGHTS, *JRA_WEIGHTS)))
V4_OUTPUT_COLUMNS = (
    "horse_score_v4",
    "race_rank_v4",
    *V4_COMPONENT_COLUMNS,
    "condition_fit_mark",
    "condition_fit_level",
    "condition_fit_reason",
    "condition_matched_quality",
    "condition_distance_score",
    "condition_course_score",
    "group_v4",
    "mark_v4",
    "warning_reason",
    "positive_reasons_v4",
    "negative_reasons_v4",
    "watch_reason_v4",
    "axis_score",
    "axis_confidence_v4",
    "top_score_gap_v4",
    "third_score_gap_v4",
    "race_competitiveness_v4",
    "opponent_eligible_v4",
    "opponent_veto_reason_v4",
    "ticket_candidate_score",
)
V41_OUTPUT_COLUMNS = (*V4_OUTPUT_COLUMNS, "condition_fit_data_status")


def prediction_logic_version(value: Any) -> str:
    """Return a supported logic label while preserving the Ver4 baseline."""

    normalized = clean_text(value).lower().replace("ver", "v")
    if normalized in {"market", "market-compare", "能力×価格比較", "能力価格比較", "比較モード"}:
        return "market"
    if normalized in {"practical", "実戦", "実戦モード"}:
        return "practical"
    if normalized in {"v4.1", "v41"}:
        return "v4.1"
    return "v4" if normalized == "v4" else "v3"


def validate_component_weights(race_type: str) -> bool:
    weights = weights_for_race_type(race_type)
    return math.isclose(sum(weights.values()), 1.0, rel_tol=0.0, abs_tol=1e-12)


def weights_for_race_type(race_type: str) -> dict[str, float]:
    return dict(NAR_WEIGHTS if clean_text(race_type).lower() == "nar" else JRA_WEIGHTS)


def normalize_index(value: Any, race_type: str) -> float | None:
    """Normalize an existing time index against a fixed absolute scale."""

    number = index_number(value)
    if number is None:
        return None
    low, high = INDEX_ANCHORS["nar" if clean_text(race_type).lower() == "nar" else "jra"]
    return round(_clamp((number - low) / (high - low) * 100.0), 1)


def index_number(value: Any) -> float | None:
    """Read numeric indices including display strings such as ``48/良``."""

    direct = to_float(value)
    if direct is not None:
        return direct
    text = clean_text(value).replace(",", "")
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    return float(match.group(0)) if match else None


def merge_prediction_tables(result: Any) -> pd.DataFrame:
    """Build an ephemeral horse-number merge without mutating either source."""

    rows: dict[str, dict[str, Any]] = {}
    # overall_table is the authoritative index source; horse_evaluation fills
    # display-only facts that are absent there.
    for table in (getattr(result, "overall_table", None), getattr(result, "horse_evaluation", None)):
        if table is None or not isinstance(table, pd.DataFrame) or table.empty:
            continue
        for _, raw in table.iterrows():
            record = raw.to_dict()
            key = horse_no(_pick(record, "馬番", "horse_no", "horse_number", "馬"))
            if not key:
                continue
            target = rows.setdefault(key, {"horse_no_key_v4": key})
            for name, value in record.items():
                if name not in target or _missing(target.get(name)):
                    if not _missing(value):
                        target[str(name)] = value
    return pd.DataFrame(sorted(rows.values(), key=lambda item: _horse_sort_key(item.get("horse_no_key_v4"))))


def evaluate_ver4_table(
    table: pd.DataFrame | Sequence[Mapping[str, Any]] | None,
    race_type: str,
    race_info: Mapping[str, Any] | None = None,
    *,
    condition_fit_plumbing: bool = False,
) -> pd.DataFrame:
    """Calculate Ver4 output columns from existing, result-free horse facts."""

    frame = _as_frame(table)
    if frame.empty:
        return _empty_result(frame, V41_OUTPUT_COLUMNS if condition_fit_plumbing else V4_OUTPUT_COLUMNS)
    race_type = "nar" if clean_text(race_type).lower() == "nar" else "jra"
    if not validate_component_weights(race_type):
        raise ValueError(f"Ver4 component weights do not sum to 1.0: {race_type}")

    evaluated: list[dict[str, Any]] = []
    for _, source in frame.iterrows():
        row = source.to_dict()
        components, diagnostics = component_scores(
            row,
            race_type,
            race_info,
            condition_fit_plumbing=condition_fit_plumbing,
        )
        weights = weights_for_race_type(race_type)
        score = round(sum(components[name] * weight for name, weight in weights.items()), 1)
        evaluated.append({**row, **components, **diagnostics, "horse_score_v4": score})

    result = pd.DataFrame(evaluated, index=frame.index)
    result["race_rank_v4"] = (
        pd.to_numeric(result["horse_score_v4"], errors="coerce")
        .rank(method="min", ascending=False)
        .astype("Int64")
    )
    result["group_v4"] = result.apply(lambda row: group_for_horse(row, race_type), axis=1)
    result = _add_race_context(result, race_type)
    result = _add_marks(result, race_type)
    result = _add_opponent_context(result)
    return result


def component_scores(
    row: Mapping[str, Any],
    race_type: str,
    race_info: Mapping[str, Any] | None = None,
    *,
    condition_fit_plumbing: bool = False,
) -> tuple[dict[str, float], dict[str, Any]]:
    race_type = "nar" if clean_text(race_type).lower() == "nar" else "jra"
    fit = _condition_fit(row, race_info, plumbing_fix=condition_fit_plumbing)
    base = _base_ability_score(row, race_type)
    condition, matched_quality, distance_score, course_score = _condition_score(
        row,
        race_type,
        fit,
        plumbing_fix=condition_fit_plumbing,
    )
    jockey = _jockey_score(row)
    age_weight = _age_weight_score(row)
    momentum = _momentum_score(row, race_type)
    race_shape = _race_shape_score(row)
    training = _training_score(row) if race_type == "jra" else None

    components: dict[str, float] = {
        "base_ability_score": base,
        "condition_score": condition,
        "jockey_score": jockey,
        "age_weight_score": age_weight,
        "momentum_score_v4": momentum,
        "race_shape_score": race_shape,
    }
    if training is not None:
        components["training_score"] = training

    positives, negatives = _component_reasons(
        components,
        fit,
        row,
        race_type,
        plumbing_fix=condition_fit_plumbing,
    )
    warning = " / ".join(negatives[:3])
    diagnostics = {
        "condition_fit_mark": fit.get("condition_fit_mark", ""),
        "condition_fit_level": fit.get("condition_fit_level", "none"),
        "condition_fit_reason": fit.get("condition_fit_reason", ""),
        "condition_fit_data_status": fit.get("condition_fit_data_status", ""),
        "matched_past_runs": fit.get("matched_past_runs", []),
        "condition_matched_quality": matched_quality,
        "condition_distance_score": distance_score,
        "condition_course_score": course_score,
        "warning_reason": warning,
        "positive_reasons_v4": positives,
        "negative_reasons_v4": negatives,
    }
    return components, diagnostics


def group_for_horse(row: Mapping[str, Any], race_type: str) -> str:
    score = _number(row.get("horse_score_v4")) or 0.0
    if score >= 82.0 and _ss_qualified(row, race_type):
        return "SS"
    # A score above the SS line without the minimum all-round qualification is
    # an A, not a forced SS.  Therefore a race may have zero or many SS horses.
    if score >= 70.0:
        return "A"
    if score >= 58.0:
        return "B"
    if score >= 45.0:
        return "C"
    return "Z"


def legacy_decision_from_v4(decision_v4: Any) -> str:
    return {"BUY": "BUY", "LIGHT": "HOLD", "WATCH": "HOLD", "SKIP": "SKIP"}.get(
        clean_text(decision_v4).upper(), "SKIP"
    )


def decision_label_ja(decision_v4: Any) -> str:
    return {"BUY": "買い", "LIGHT": "保留", "WATCH": "保留", "SKIP": "見送り"}.get(
        clean_text(decision_v4).upper(), "見送り"
    )


def build_ver4_race_summary(table: pd.DataFrame | None) -> dict[str, Any]:
    if table is None or not isinstance(table, pd.DataFrame) or table.empty:
        return {
            "decision_v4": "SKIP",
            "legacy_decision": "SKIP",
            "axis_horse_no": "",
            "axis_score": 0.0,
            "axis_confidence": "なし",
            "ticket_candidate_score": 0.0,
            "ticket_veto_reason": "出走馬データなし",
            "ticket_type": "",
            "tickets": [],
            "opponent_horse_numbers": [],
        }

    ranked = table.sort_values(["race_rank_v4", "horse_score_v4"], ascending=[True, False])
    axes = ranked[ranked["mark_v4"].astype(str).eq("◎")]
    axis = axes.iloc[0] if not axes.empty else None
    eligible = ranked[ranked["opponent_eligible_v4"].eq(True)]
    if axis is not None:
        axis_no = horse_no(_pick(axis.to_dict(), "馬番", "horse_no", "horse_number", "馬", "horse_no_key_v4"))
        eligible = eligible[
            eligible.apply(
                lambda row: horse_no(_pick(row.to_dict(), "馬番", "horse_no", "horse_number", "馬", "horse_no_key_v4")) != axis_no,
                axis=1,
            )
        ]
    opponents = eligible.head(3)

    if axis is None:
        credible = int((ranked["horse_score_v4"] >= 58.0).sum())
        decision = "WATCH" if credible >= 2 else "SKIP"
        veto = "◎に必要な絶対水準・適性・弱点条件を満たす軸馬なし"
        axis_no = ""
        axis_score = 0.0
        axis_confidence = "なし"
        ticket_score = round(float(ranked["horse_score_v4"].head(3).mean()), 1) if credible else 0.0
        ticket_type = ""
        tickets: list[str] = []
    else:
        axis_data = axis.to_dict()
        axis_no = horse_no(_pick(axis_data, "馬番", "horse_no", "horse_number", "馬", "horse_no_key_v4"))
        axis_score = float(_number(axis_data.get("axis_score")) or 0.0)
        axis_confidence = clean_text(axis_data.get("axis_confidence_v4")) or "標準"
        opponent_scores = [float(value) for value in pd.to_numeric(opponents["ticket_candidate_score"], errors="coerce").dropna()]
        ticket_score = round(axis_score * 0.65 + (sum(opponent_scores) / len(opponent_scores) if opponent_scores else 0.0) * 0.35, 1)
        if opponents.empty:
            if axis_confidence == "高" and float(axis_data.get("horse_score_v4", 0.0)) >= 85.0:
                decision = "LIGHT"
                veto = ""
                ticket_type = "単勝"
                tickets = [axis_no]
                ticket_score = round(axis_score, 1)
            else:
                decision = "WATCH"
                veto = "相手候補が適性・弱点のVETO条件を通過しない"
                ticket_type = ""
                tickets = []
        else:
            decision = "BUY" if axis_confidence == "高" and ticket_score >= 70.0 else "LIGHT" if ticket_score >= 60.0 else "WATCH"
            veto = ""
            opponent_numbers = [
                horse_no(_pick(row, "馬番", "horse_no", "horse_number", "馬", "horse_no_key_v4"))
                for row in opponents.to_dict("records")
            ]
            strong_top3 = len(opponents) >= 2 and all(
                float(row.get("horse_score_v4", 0.0)) >= 70.0 and float(row.get("condition_score", 0.0)) >= 50.0
                for row in opponents.head(2).to_dict("records")
            )
            if axis_confidence == "高" and strong_top3:
                ticket_type = "三連複"
                tickets = [f"{axis_no}-{opponent_numbers[0]}-{opponent_numbers[1]}"]
            else:
                ticket_type = "馬連" if axis_confidence == "高" and ticket_score >= 70.0 else "ワイド"
                tickets = [f"{axis_no}-{number}" for number in opponent_numbers if number]

    opponent_numbers = [
        horse_no(_pick(row, "馬番", "horse_no", "horse_number", "馬", "horse_no_key_v4"))
        for row in opponents.to_dict("records")
    ]
    return {
        "decision_v4": decision,
        "legacy_decision": legacy_decision_from_v4(decision),
        "axis_horse_no": axis_no,
        "axis_score": round(axis_score, 1),
        "axis_confidence": axis_confidence,
        "ticket_candidate_score": ticket_score,
        "ticket_veto_reason": veto,
        "ticket_type": ticket_type,
        "tickets": tickets,
        "opponent_horse_numbers": opponent_numbers,
        "top_score_gap": float(_number(ranked.iloc[0].get("top_score_gap_v4")) or 0.0),
        "third_score_gap": float(_number(ranked.iloc[0].get("third_score_gap_v4")) or 0.0),
        "race_competitiveness": clean_text(ranked.iloc[0].get("race_competitiveness_v4")),
    }


def build_ver4_investment_decision(table: pd.DataFrame, race_type: str) -> Any:
    """Return the existing InvestmentDecision shape using only V4 horse output."""

    from .betting_recommendation import BettingRecommendation
    from .investment_decision import InvestmentDecision

    summary = build_ver4_race_summary(table)
    decision_v4 = summary["decision_v4"]
    selected = None
    if summary["tickets"]:
        ticket_numbers = tuple(tuple(ticket.split("-")) for ticket in summary["tickets"])
        selected = BettingRecommendation(
            ticket_type=summary["ticket_type"],
            label=f"V4 {summary['axis_horse_no']}軸",
            stars="",
            expected_roi=None,
            condition="Ver4絶対評価・相手VETO通過",
            reason="馬→印→軸→相手→券種の順で構成",
            source="ver4",
            risk_label="正式" if decision_v4 == "BUY" else "参考",
            strategy_id="ver4_absolute_selection",
            sample_races=0,
            ticket_count=len(ticket_numbers),
            tickets=tuple(summary["tickets"]),
            ticket_numbers=ticket_numbers,
            ticket_horses=tuple([summary["axis_horse_no"], *summary["opponent_horse_numbers"]]),
            matched_conditions=("Ver4絶対評価", "相手VETO通過"),
            adopted_reason="予測結果の軸・相手評価から低点数で構成",
            audit={
                "logic_version": "v4",
                "decision_v4": decision_v4,
                "strategy_score": summary["ticket_candidate_score"],
                "confidence": summary["axis_confidence"],
                "axis_score": summary["axis_score"],
                "ticket_candidate_score": summary["ticket_candidate_score"],
                "ticket_veto_reason": summary["ticket_veto_reason"],
            },
        )
    reason = summary["ticket_veto_reason"] or f"Ver4判断: {decision_v4}"
    return InvestmentDecision(
        race_type="nar" if clean_text(race_type).lower() == "nar" else "jra",
        judgement=decision_label_ja(decision_v4),
        selected=selected,
        candidates=(selected,) if selected is not None else (),
        reason_lines=(reason,),
        total_stake=(selected.ticket_count * 100) if selected is not None else 0,
        target_horses=tuple(selected.ticket_horses) if selected is not None else (),
        logic_version="v4",
        decision_v4=decision_v4,
        axis_score=summary["axis_score"],
        axis_confidence_v4=summary["axis_confidence"],
        ticket_candidate_score=summary["ticket_candidate_score"],
        ticket_veto_reason=summary["ticket_veto_reason"],
    )


def apply_ver4_to_result(result: Any) -> Any:
    """Attach V4 views to a PredictionResult without changing Ver3 columns."""

    merged = merge_prediction_tables(result)
    evaluated = evaluate_ver4_table(merged, getattr(result, "race_mode", "jra"), getattr(result, "race_info", {}) or {})
    return _attach_ver4_result(result, evaluated, "v4", V4_OUTPUT_COLUMNS)


def apply_ver41_to_result(result: Any) -> Any:
    """Attach Ver4.1 using result_df condition sources and unchanged Ver4 rules."""

    merged = merge_prediction_tables(result)
    merged = attach_condition_fit_sources(merged, getattr(result, "debug_info", {}) or {})
    evaluated = evaluate_ver4_table(
        merged,
        getattr(result, "race_mode", "jra"),
        getattr(result, "race_info", {}) or {},
        condition_fit_plumbing=True,
    )
    return _attach_ver4_result(result, evaluated, "v4.1", V41_OUTPUT_COLUMNS)


def _attach_ver4_result(
    result: Any,
    evaluated: pd.DataFrame,
    logic_version: str,
    output_columns: Sequence[str],
) -> Any:
    by_no = {
        horse_no(_pick(row, "馬番", "horse_no", "horse_number", "馬", "horse_no_key_v4")): row
        for row in evaluated.to_dict("records")
    }
    for attr in ("overall_table", "horse_evaluation"):
        source = getattr(result, attr, None)
        if source is None or not isinstance(source, pd.DataFrame):
            continue
        target = source.copy()
        for column in output_columns:
            if column not in target.columns:
                target[column] = None
        for index, raw in target.iterrows():
            key = horse_no(_pick(raw.to_dict(), "馬番", "horse_no", "horse_number", "馬"))
            values = by_no.get(key, {})
            for column in output_columns:
                if column in values:
                    target.at[index, column] = values[column]
        setattr(result, attr, target)
    summary = build_ver4_race_summary(evaluated)
    result.logic_version = logic_version
    result.ver4_summary = summary
    debug = dict(getattr(result, "debug_info", {}) or {})
    debug_key = "ver4_1" if logic_version == "v4.1" else "ver4"
    debug[debug_key] = {"summary": summary, "horses": evaluated.to_dict("records")}
    result.debug_info = debug
    return result


def apply_prediction_logic(result: Any, version: Any = "v3") -> Any:
    version = prediction_logic_version(version)
    if version == "market":
        from .market_compare import apply_market_compare_to_result

        return apply_market_compare_to_result(result)
    if version == "practical":
        from .practical_mode import apply_practical_to_result

        return apply_practical_to_result(result)
    if version == "v4.1":
        return apply_ver41_to_result(result)
    if version == "v4":
        return apply_ver4_to_result(result)
    result.logic_version = "v3"
    return result


def attach_condition_fit_sources(
    frame: pd.DataFrame,
    debug_info: Mapping[str, Any],
) -> pd.DataFrame:
    """Return an ephemeral frame enriched with result_df-only past runs."""

    sources = debug_info.get("condition_fit_sources")
    if frame.empty or not isinstance(sources, Mapping):
        return frame.copy()
    result = frame.copy()
    source_columns = sorted(
        {
            str(name)
            for source in sources.values()
            if isinstance(source, Mapping)
            for name in source
        }
    )
    for column in source_columns:
        if column not in result.columns:
            result[column] = pd.Series([None] * len(result), index=result.index, dtype="object")
    for index, raw in result.iterrows():
        key = horse_no(_pick(raw.to_dict(), "馬番", "horse_no", "horse_number", "馬", "horse_no_key_v4"))
        source = sources.get(key)
        if not isinstance(source, Mapping):
            continue
        for name, value in source.items():
            if name == "_past_runs" or _missing(result.at[index, name]):
                if not _missing(value):
                    result.at[index, name] = value
    return result


def _base_ability_score(row: Mapping[str, Any], race_type: str) -> float:
    recent = [_normal(row, names, race_type) for names in (("3走前", "race3"), ("2走前", "race2"), ("前走", "race1"))]
    valid_recent = [value for value in recent if value is not None]
    average = _normal(row, ("平均指数", "3走平均", "近3走平均", "avg5"), race_type)
    year_max = _normal(row, ("過去1年最高指数", "year_max_index", "最高指数"), race_type)
    if average is None and valid_recent:
        average = sum(valid_recent) / len(valid_recent)
    if year_max is None and valid_recent:
        year_max = max(valid_recent)
    last = recent[-1]
    recent_max = max(valid_recent) if valid_recent else None
    values = [(average, 0.35), (last, 0.25), (year_max, 0.20), (recent_max, 0.20)]
    score = _weighted_available(values)
    if score is None:
        raw = index_number(_pick(row, "raw_score", "_raw_score", "能力評価値", "ability_display_score"))
        if raw is not None:
            low, high = (-40.0, 120.0) if race_type == "nar" else (-20.0, 140.0)
            score = _clamp((raw - low) / (high - low) * 100.0)
    return round(score if score is not None else 50.0, 1)


def _condition_score(
    row: Mapping[str, Any],
    race_type: str,
    fit: Mapping[str, Any],
    *,
    plumbing_fix: bool = False,
) -> tuple[float, float, float, float]:
    matched = fit.get("matched_past_runs") if isinstance(fit.get("matched_past_runs"), list) else []
    quality_values = [normalize_index(item.get("time_index"), race_type) for item in matched if isinstance(item, Mapping)]
    quality_values = [value for value in quality_values if value is not None]
    level = clean_text(fit.get("condition_fit_level")) or "none"
    factor = {"same_venue_distance": 1.0, "same_turn_distance": 0.90, "same_distance": 0.80}.get(level, 0.0)
    shallow = _career_shallow(row)
    if quality_values:
        matched_quality = max(quality_values) * factor
    elif level != "none":
        matched_quality = 50.0 * factor
    else:
        data_missing = clean_text(fit.get("condition_fit_data_status")) == "missing_source_data"
        matched_quality = 50.0 if shallow or (plumbing_fix and data_missing) else 30.0
    distance = _normal(row, ("距離指数", "distance_index"), race_type)
    course = _normal(row, ("コース指数", "course_index"), race_type)
    distance_score = distance if distance is not None else (50.0 if shallow else 45.0)
    course_score = course if course is not None else (50.0 if shallow else 45.0)
    score = matched_quality * 0.50 + distance_score * 0.25 + course_score * 0.25
    return round(_clamp(score), 1), round(matched_quality, 1), round(distance_score, 1), round(course_score, 1)


def _jockey_score(row: Mapping[str, Any]) -> float:
    text = " ".join(clean_text(_pick(row, name)) for name in ("騎手評価", "騎手詳細", "騎手継続/乗替", "jockey_change"))
    score = 50.0
    if any(token in text for token in ("◎", "大幅強化", "強化")):
        score = 75.0
    elif any(token in text for token in ("○", "継続", "好相性")):
        score = 65.0
    elif any(token in text for token in ("△", "弱化", "不安")):
        score = 35.0
    adjustment = _number(_pick(row, "騎手補正", "jockey_adjustment", "騎手評価点"))
    if adjustment is not None:
        score = adjustment if 0.0 <= adjustment <= 100.0 else score + _clamp(adjustment, -20.0, 20.0)
    return round(_clamp(score), 1)


def _age_weight_score(row: Mapping[str, Any]) -> float:
    score = 50.0
    for names in (("年齢補正", "age_adjustment"), ("斤量補正", "weight_adjustment")):
        value = _number(_pick(row, *names))
        if value is not None:
            score += _clamp(value, -15.0, 15.0)
    text = " ".join(clean_text(_pick(row, name)) for name in ("斤量詳細", "年齢評価", "weight_detail"))
    if any(token in text for token in ("有利", "減", "好条件")):
        score += 8.0
    if any(token in text for token in ("不利", "増", "重い")):
        score -= 8.0
    return round(_clamp(score), 1)


def _momentum_score(row: Mapping[str, Any], race_type: str) -> float:
    raw = [index_number(_pick(row, *names)) for names in (("3走前", "race3"), ("2走前", "race2"), ("前走", "race1"))]
    values = [value for value in raw if value is not None]
    score = 50.0
    if len(values) >= 2:
        slope = values[-1] - values[0]
        score = 82.0 if slope >= 15 else 70.0 if slope >= 7 else 58.0 if slope >= 2 else 48.0 if slope > -7 else 32.0
        last_absolute = normalize_index(values[-1], race_type)
        if last_absolute is not None:
            score = score * 0.70 + last_absolute * 0.30
    state = clean_text(_pick(row, "状態", "form_state", "勢いランク", "momentum_rank", "近3走傾向", "recent3_trend"))
    if any(token in state for token in ("上昇", "良化", "反発", "持ち直し")):
        score = max(score, 68.0)
    if any(token in state for token in ("下降", "急落", "不安")):
        score = min(score, 38.0)
    existing = _number(_pick(row, "momentum_score"))
    if existing is not None and 0 <= existing <= 100:
        score = score * 0.7 + existing * 0.3
    return round(_clamp(score), 1)


def _race_shape_score(row: Mapping[str, Any]) -> float:
    texts = " ".join(
        clean_text(_pick(row, name))
        for name in ("展開印", "pace_fit", "4角評価", "corner4_evaluation", "直線評価", "straight_evaluation", "脚質評価")
    )
    if any(token in texts for token in ("◎", "かなり有利", "最適")):
        return 80.0
    if any(token in texts for token in ("○", "有利", "向く")):
        return 68.0
    if any(token in texts for token in ("×", "不利", "向かない")):
        return 28.0
    if "△" in texts:
        return 42.0
    return 50.0


def _training_score(row: Mapping[str, Any]) -> float:
    text = clean_text(_pick(row, "調教評価", "追切評価", "training_grade", "_調教評価記号"))
    if "◎" in text or text.upper() == "S":
        return 85.0
    if "○" in text or text.upper() == "A":
        return 72.0
    if "△" in text or text.upper() == "C":
        return 42.0
    if "×" in text or text.upper() == "D":
        return 25.0
    return 50.0


def _component_reasons(
    components: Mapping[str, float],
    fit: Mapping[str, Any],
    row: Mapping[str, Any],
    race_type: str,
    *,
    plumbing_fix: bool = False,
) -> tuple[list[str], list[str]]:
    positives: list[str] = []
    negatives: list[str] = []
    labels = {
        "base_ability_score": "基礎能力",
        "condition_score": "条件適性",
        "jockey_score": "騎手材料",
        "age_weight_score": "年齢・斤量",
        "training_score": "調教",
        "momentum_score_v4": "近走勢い",
        "race_shape_score": "展開適性",
    }
    for name, value in components.items():
        if value >= 70:
            positives.append(f"{labels[name]}{value:.0f}")
        elif value < 35:
            negatives.append(f"{labels[name]}{value:.0f}")
    mark = clean_text(fit.get("condition_fit_mark"))
    if mark:
        positives.insert(0, f"条件実績{mark}")
    elif (
        not _career_shallow(row)
        and (
            not plumbing_fix
            or clean_text(fit.get("condition_fit_data_status")) == "no_match"
        )
    ):
        negatives.append("近3走に同距離条件実績なし")
    if race_type == "jra" and components.get("training_score", 50.0) == 50.0:
        # Missing training is neutral, never invented as a positive/negative.
        pass
    return positives[:5], negatives[:5]


def _ss_qualified(row: Mapping[str, Any], race_type: str) -> bool:
    base = _number(row.get("base_ability_score")) or 0.0
    condition = _number(row.get("condition_score")) or 0.0
    momentum = _number(row.get("momentum_score_v4")) or 0.0
    warning = clean_text(row.get("warning_reason"))
    if race_type == "nar":
        return base >= 72.0 and condition >= 62.0 and momentum >= 45.0 and not warning
    training = _number(row.get("training_score")) or 0.0
    return base >= 70.0 and condition >= 60.0 and training >= 50.0 and momentum >= 45.0 and not warning


def _add_race_context(frame: pd.DataFrame, race_type: str) -> pd.DataFrame:
    result = frame.copy()
    ranked = result.sort_values("horse_score_v4", ascending=False)
    scores = [float(value) for value in ranked["horse_score_v4"].tolist()]
    top = scores[0] if scores else 0.0
    second = scores[1] if len(scores) > 1 else top
    third = scores[2] if len(scores) > 2 else second
    gap2 = round(top - second, 1)
    gap3 = round(top - third, 1)
    competitiveness = "軸明確" if gap2 >= 6.0 else "上位拮抗" if gap3 <= 5.0 else "相手比較"
    result["top_score_gap_v4"] = gap2
    result["third_score_gap_v4"] = gap3
    result["race_competitiveness_v4"] = competitiveness
    result["axis_score"] = result.apply(
        lambda row: round(_clamp(float(row["horse_score_v4"]) + (gap2 * 1.5 if int(row["race_rank_v4"]) == 1 else 0.0)), 1),
        axis=1,
    )
    result["axis_confidence_v4"] = result.apply(
        lambda row: (
            "高"
            if int(row["race_rank_v4"]) == 1 and row["axis_score"] >= 82 and gap2 >= 4
            else "中"
            if int(row["race_rank_v4"]) == 1 and row["axis_score"] >= 72
            else "低"
        ),
        axis=1,
    )
    return result


def _add_marks(frame: pd.DataFrame, race_type: str) -> pd.DataFrame:
    result = frame.copy()
    marks: dict[Any, str] = {}
    watch_reasons: dict[Any, str] = {}
    for index, row in result.sort_values(["race_rank_v4", "horse_score_v4"], ascending=[True, False]).iterrows():
        rank = int(row["race_rank_v4"])
        score = float(row["horse_score_v4"])
        warning = clean_text(row.get("warning_reason"))
        condition = float(row.get("condition_score", 0.0))
        base = float(row.get("base_ability_score", 0.0))
        if rank == 1 and score >= 72.0 and base >= 68.0 and condition >= 50.0 and not warning:
            marks[index] = "◎"
        elif rank <= 2 and score >= 66.0 and condition >= 45.0:
            marks[index] = "○"
        elif rank <= 3 and score >= 58.0:
            marks[index] = "▲"
        else:
            reason = _watch_signal(row)
            if clean_text(row.get("group_v4")) in {"C", "Z"} and reason:
                marks[index] = "✓"
                watch_reasons[index] = reason
            elif rank <= 5 and score >= 52.0:
                marks[index] = "△"
            else:
                marks[index] = ""
    result["mark_v4"] = pd.Series(marks, index=result.index).fillna("")
    result["watch_reason_v4"] = pd.Series(watch_reasons, index=result.index).fillna("")
    return result


def _watch_signal(row: Mapping[str, Any]) -> str:
    signals = (
        ("condition_score", "条件適性"),
        ("condition_distance_score", "距離適性"),
        ("condition_course_score", "コース適性"),
        ("momentum_score_v4", "近走勢い"),
        ("race_shape_score", "展開適性"),
    )
    best_name = ""
    best_score = 0.0
    for name, label in signals:
        value = _number(row.get(name)) or 0.0
        if value >= 72.0 and value > best_score:
            best_name, best_score = label, value
    if not best_name:
        last = index_number(_pick(row, "前走", "race1"))
        average = index_number(_pick(row, "平均指数", "3走平均", "近3走平均"))
        if last is not None and average is not None and last - average >= 12.0:
            return "前走指数の上振れ"
        return ""
    return f"{best_name}{best_score:.0f}"


def _add_opponent_context(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    eligible: list[bool] = []
    vetoes: list[str] = []
    scores: list[float] = []
    for _, row in result.iterrows():
        group = clean_text(row.get("group_v4"))
        mark = clean_text(row.get("mark_v4"))
        condition = float(row.get("condition_score", 0.0))
        strong_support = max(
            float(row.get("condition_distance_score", 0.0)),
            float(row.get("condition_course_score", 0.0)),
            float(row.get("momentum_score_v4", 0.0)),
            float(row.get("race_shape_score", 0.0)),
        ) >= 72.0
        reasons: list[str] = []
        if group not in {"SS", "A", "B"} and mark != "✓":
            reasons.append("B以上でも強い✓でもない")
        if clean_text(row.get("warning_reason")):
            reasons.append("重大な弱点あり")
        if condition < 50.0 and not strong_support:
            reasons.append("条件50未満かつ強い補完材料なし")
        ok = not reasons
        score = float(row.get("horse_score_v4", 0.0)) * 0.65 + condition * 0.20 + max(
            float(row.get("momentum_score_v4", 0.0)), float(row.get("race_shape_score", 0.0))
        ) * 0.15
        eligible.append(ok)
        vetoes.append(" / ".join(reasons))
        scores.append(round(_clamp(score), 1))
    result["opponent_eligible_v4"] = eligible
    result["opponent_veto_reason_v4"] = vetoes
    result["ticket_candidate_score"] = scores
    return result


def _condition_fit(
    row: Mapping[str, Any],
    race_info: Mapping[str, Any] | None,
    *,
    plumbing_fix: bool = False,
) -> dict[str, Any]:
    if plumbing_fix:
        evaluated = evaluate_condition_fit(row, race_info)
        if clean_text(evaluated.get("condition_fit_data_status")) != "missing_source_data":
            return evaluated

        existing_mark = clean_text(row.get("condition_fit_mark"))
        level = canonical_condition_fit_level(row.get("condition_fit_level"), existing_mark)
        if level is None:
            level = canonical_condition_fit_level(row.get("star_match_level"), existing_mark)
        existing_runs = row.get("matched_past_runs")
        if level and level != "none":
            mark = existing_mark or {"same_venue_distance": "★", "same_turn_distance": "☆", "same_distance": "※"}.get(
                level,
                "",
            )
            return {
                "condition_fit_mark": mark or None,
                "condition_fit_level": level,
                "condition_fit_reason": clean_text(row.get("condition_fit_reason"))
                or "既存の構造化条件一致情報から接続",
                "condition_fit_data_status": "ok",
                "matched_past_runs": existing_runs if isinstance(existing_runs, list) else [],
            }
        return evaluated

    existing_level = clean_text(row.get("condition_fit_level"))
    existing_mark = clean_text(row.get("condition_fit_mark"))
    existing_runs = row.get("matched_past_runs")
    if existing_level not in {"", "none"} or existing_mark or (isinstance(existing_runs, list) and bool(existing_runs)):
        level = existing_level or {"★": "same_venue_distance", "☆": "same_turn_distance", "※": "same_distance"}.get(existing_mark, "none")
        return {
            "condition_fit_mark": existing_mark,
            "condition_fit_level": level,
            "condition_fit_reason": clean_text(row.get("condition_fit_reason")),
            "matched_past_runs": existing_runs if isinstance(existing_runs, list) else [],
        }
    return evaluate_condition_fit(row, race_info)


def _career_shallow(row: Mapping[str, Any]) -> bool:
    age_text = clean_text(_pick(row, "馬年齢", "性齢", "馬齢", "age"))
    age_match = re.search(r"(\d{1,2})", age_text)
    if age_match and int(age_match.group(1)) <= 2:
        return True
    starts = _number(_pick(row, "通算出走数", "career_starts", "出走数"))
    if starts is not None and starts <= 3:
        return True
    recent_count = sum(index_number(_pick(row, *names)) is not None for names in (("3走前", "race3"), ("2走前", "race2"), ("前走", "race1")))
    return recent_count <= 1


def _normal(row: Mapping[str, Any], names: Sequence[str], race_type: str) -> float | None:
    return normalize_index(_pick(row, *names), race_type)


def _weighted_available(values: Sequence[tuple[float | None, float]]) -> float | None:
    present = [(value, weight) for value, weight in values if value is not None]
    if not present:
        return None
    weight_sum = sum(weight for _, weight in present)
    return sum(float(value) * weight for value, weight in present) / weight_sum


def _as_frame(table: pd.DataFrame | Sequence[Mapping[str, Any]] | None) -> pd.DataFrame:
    if table is None:
        return pd.DataFrame()
    if isinstance(table, pd.DataFrame):
        return table.copy()
    return pd.DataFrame([dict(row) for row in table])


def _empty_result(frame: pd.DataFrame, output_columns: Sequence[str] = V4_OUTPUT_COLUMNS) -> pd.DataFrame:
    result = frame.copy()
    for column in output_columns:
        if column not in result.columns:
            result[column] = pd.Series(dtype="object")
    return result


def _pick(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and not _missing(row.get(name)):
            return row.get(name)
    return ""


def _number(value: Any) -> float | None:
    return index_number(value)


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


def _clamp(value: float, low: float = 0.0, high: float = 100.0) -> float:
    return max(low, min(high, float(value)))


def _horse_sort_key(value: Any) -> tuple[int, str]:
    number = to_float(value)
    return (int(number) if number is not None else 999, clean_text(value))

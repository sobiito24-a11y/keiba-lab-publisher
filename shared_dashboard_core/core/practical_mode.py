# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

import pandas as pd

from .betting_recommendation import BettingRecommendation
from .condition_fit import evaluate_condition_fit
from .investment_decision import InvestmentDecision, STAKE_PER_POINT
from .purchase_conditions import clean_text, horse_no, to_float


PRACTICAL_CONFIG_VERSION = "practical-1.0"
PRACTICAL_DECISION_BUY = "BUY"
PRACTICAL_DECISION_WATCH = "WATCH"
PRACTICAL_STAKE_YEN = STAKE_PER_POINT

# These rules are intentionally semantic and fixed.  They reuse Ver3 audit
# labels rather than searching numeric cut-offs against the 188-race result set.
PRACTICAL_RULES = {
    "prediction_basis": "Ver3 display marks",
    "axis_confidence_required": "A",
    "accepted_ability_gaps": ("大", "中"),
    "minimum_recent_indexes": 2,
    "blocked_recent_trends": ("連続下降", "下降傾向", "下降", "急落"),
    "accepted_jra_training_grades": ("S", "A", "B"),
    "condition_mark_required": False,
    "stake_yen": PRACTICAL_STAKE_YEN,
    "primary_ticket": "◎単勝",
}

CONDITION_OUTPUT_COLUMNS = (
    "condition_fit_mark",
    "condition_fit_level",
    "condition_fit_reason",
    "condition_fit_data_status",
    "matched_past_runs",
)
PRACTICAL_OUTPUT_COLUMNS = (*CONDITION_OUTPUT_COLUMNS, "practical_mark", "practical_warning_reason")


def apply_practical_to_result(result: Any) -> Any:
    """Attach result-free condition diagnostics while preserving all Ver3 marks."""

    from .ver4_engine import attach_condition_fit_sources, merge_prediction_tables

    merged = merge_prediction_tables(result)
    enriched = attach_condition_fit_sources(merged, getattr(result, "debug_info", {}) or {})
    evaluated = evaluate_practical_table(
        enriched,
        getattr(result, "race_mode", "jra"),
        getattr(result, "race_info", {}) or {},
    )
    by_no = {
        horse_no(_pick(row, "馬番", "horse_no", "horse_number", "馬", "horse_no_key_v4")): row
        for row in evaluated.to_dict("records")
    }
    for attr in ("overall_table", "horse_evaluation"):
        source = getattr(result, attr, None)
        if source is None or not isinstance(source, pd.DataFrame):
            continue
        target = source.copy()
        for column in PRACTICAL_OUTPUT_COLUMNS:
            if column not in target.columns:
                target[column] = pd.Series([None] * len(target), index=target.index, dtype="object")
        for index, raw in target.iterrows():
            key = horse_no(_pick(raw.to_dict(), "馬番", "horse_no", "horse_number", "馬"))
            values = by_no.get(key, {})
            for column in PRACTICAL_OUTPUT_COLUMNS:
                if column in values:
                    target.at[index, column] = values[column]
        setattr(result, attr, target)

    decision = build_practical_decision(
        evaluated,
        getattr(result, "race_mode", "jra"),
        race_info=getattr(result, "race_info", {}) or {},
    )
    result.logic_version = "practical"
    debug = dict(getattr(result, "debug_info", {}) or {})
    debug["practical"] = {
        "config_version": PRACTICAL_CONFIG_VERSION,
        "rules": practical_rules_snapshot(),
        "summary": practical_decision_snapshot(decision),
        "horses": evaluated.to_dict("records"),
    }
    result.debug_info = debug
    return result


def evaluate_practical_table(
    table: pd.DataFrame | None,
    race_type: str,
    race_info: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Add only supporting facts; never recalculate scores, ranks, or Ver3 marks."""

    if table is None or not isinstance(table, pd.DataFrame):
        return pd.DataFrame()
    result = table.copy()
    if result.empty:
        for column in PRACTICAL_OUTPUT_COLUMNS:
            if column not in result.columns:
                result[column] = pd.Series(dtype="object")
        return result

    race_type = "nar" if clean_text(race_type).lower() == "nar" else "jra"
    diagnostics = [evaluate_condition_fit(raw.to_dict(), race_info) for _, raw in result.iterrows()]
    for column in CONDITION_OUTPUT_COLUMNS:
        result[column] = pd.Series(
            [item.get(column) for item in diagnostics],
            index=result.index,
            dtype="object",
        )
    result["practical_mark"] = [ver3_mark(raw.to_dict()) for _, raw in result.iterrows()]
    result["practical_warning_reason"] = [
        " / ".join(practical_warning_reasons(raw.to_dict(), race_type))
        for _, raw in result.iterrows()
    ]
    return result


def build_practical_decision(
    table: Any,
    race_type: str,
    *,
    race_info: Mapping[str, Any] | None = None,
) -> InvestmentDecision:
    """Return a conservative BUY/WATCH decision with one fixed win ticket."""

    race_type = "nar" if clean_text(race_type).lower() == "nar" else "jra"
    if table is None or not isinstance(table, pd.DataFrame) or table.empty:
        return _watch_decision(race_type, ("出走馬データがないためWATCH。",))

    working = table.copy()
    if "condition_fit_data_status" not in working.columns:
        working = evaluate_practical_table(working, race_type, race_info)
    honmei_rows = [raw.to_dict() for _, raw in working.iterrows() if ver3_mark(raw.to_dict()) == "◎"]
    if not honmei_rows:
        return _watch_decision(race_type, ("Ver3の◎がないためWATCH。",))
    if len(honmei_rows) != 1:
        return _watch_decision(race_type, ("Ver3の◎が複数あり軸を1頭へ固定できないためWATCH。",))

    honmei = honmei_rows[0]
    reasons: list[str] = []
    blockers: list[str] = []
    axis = clean_text(_pick(honmei, "軸信頼度", "axis_confidence"))
    gap = clean_text(_pick(honmei, "能力差", "ability_gap_level"))
    trend = clean_text(_pick(honmei, "近3走傾向", "recent3_trend"))
    condition_mark = clean_text(honmei.get("condition_fit_mark"))
    condition_status = clean_text(honmei.get("condition_fit_data_status"))

    if axis == PRACTICAL_RULES["axis_confidence_required"]:
        reasons.append("◎の絶対評価は既存の軸信頼度A")
    else:
        blockers.append(f"◎の軸信頼度が{axis or '未評価'}")

    if gap in PRACTICAL_RULES["accepted_ability_gaps"]:
        reasons.append(f"能力差{gap}で2位との差を確認")
    else:
        blockers.append("◎と上位馬の評価差が小さい")

    if trend in PRACTICAL_RULES["blocked_recent_trends"]:
        blockers.append(f"近3走が{trend}")
    elif trend:
        reasons.append(f"近3走は{trend}")

    if condition_status == "missing_source_data":
        blockers.append("条件適性の元データ不足")
    elif condition_mark:
        reasons.append(f"今回条件に{condition_mark}実績あり")
    else:
        reasons.append("条件実績は非該当（★必須にはしない）")

    if race_type == "jra":
        training_grade = _training_grade(_pick(honmei, "調教評価", "追切評価", "_調教評価記号"))
        if training_grade in PRACTICAL_RULES["accepted_jra_training_grades"]:
            reasons.append(f"調教評価{training_grade}")

    blockers.extend(practical_warning_reasons(honmei, race_type))
    blockers = _unique(blockers)
    reasons = _unique(reasons)
    number = horse_no(_pick(honmei, "馬番", "horse_no", "horse_number", "馬"))
    name = clean_text(_pick(honmei, "馬名", "horse_name", "name"))
    if blockers:
        return _watch_decision(
            race_type,
            tuple(f"{line}ためWATCH。" if not line.endswith("。") else line for line in blockers),
            honmei_no=number,
            honmei_name=name,
            audit={"supporting_reasons": reasons, "blockers": blockers},
        )

    buy_reason = "◎の絶対評価が高く、2位との差も確認でき、重大な不安材料がない。"
    ticket = BettingRecommendation(
        ticket_type="単勝",
        label="◎単勝（100円固定）",
        stars="",
        expected_roi=None,
        condition=" / ".join(reasons),
        reason=buy_reason,
        source="practical_fixed_rules",
        risk_label="実戦固定",
        strategy_id=PRACTICAL_CONFIG_VERSION,
        ticket_count=1,
        tickets=(number,),
        ticket_numbers=((number,),),
        ticket_horses=(" ".join(part for part in (number, name) if part),),
        matched_conditions=tuple(reasons),
        adopted_reason=buy_reason,
        audit={
            "config_version": PRACTICAL_CONFIG_VERSION,
            "strategy_score": "",
            "adopted_reason": buy_reason,
            "adopted_reason_lines": reasons,
            "practical_decision": PRACTICAL_DECISION_BUY,
            "honmei_horse_no": number,
            "honmei_horse_name": name,
            "stake_yen": PRACTICAL_STAKE_YEN,
            "rules": practical_rules_snapshot(),
        },
    )
    return InvestmentDecision(
        race_type=race_type,
        judgement=PRACTICAL_DECISION_BUY,
        selected=ticket,
        candidates=(ticket,),
        reason_lines=tuple(reasons),
        total_stake=PRACTICAL_STAKE_YEN,
        target_horses=(number,),
        logic_version="practical",
        practical_decision=PRACTICAL_DECISION_BUY,
        practical_reason=buy_reason,
        practical_reason_lines=tuple(reasons),
        practical_config_version=PRACTICAL_CONFIG_VERSION,
        honmei_horse_no=number,
        honmei_horse_name=name,
    )


def practical_warning_reasons(row: Mapping[str, Any], race_type: str) -> list[str]:
    """Return conservative blockers from existing, result-free facts only."""

    reasons: list[str] = []
    if to_float(_pick(row, "AI点", "normalized_ai_score", "ai_score")) is None:
        reasons.append("AI点が不足")
    if to_float(_pick(row, "raw_score", "_raw_score", "能力評価値", "ability_display_score")) is None:
        reasons.append("能力評価データが不足")
    odds = to_float(_pick(row, "単勝オッズ", "オッズ", "odds"))
    if odds is None or odds <= 0:
        reasons.append("単勝オッズが未取得")
    # Existing index cells may be display strings such as
    # ``85/札幌ダ1700m右★``. Reuse the established result-free parser instead
    # of treating those valid cells as missing.
    from .ver4_engine import index_number

    recent_count = sum(
        index_number(_pick(row, *names)) is not None
        for names in (("3走前", "race3"), ("2走前", "race2"), ("前走", "race1"))
    )
    if recent_count < int(PRACTICAL_RULES["minimum_recent_indexes"]):
        reasons.append("近走指数データが不足")
    if _truthy(_pick(row, "_地方指数データ不足", "data_shortage")):
        reasons.append("地方指数データが不足")
    major_count = to_float(_pick(row, "_重大マイナス数", "major_negative_count"))
    if major_count is not None and major_count > 0:
        reasons.append("重大マイナス材料あり")
    supplement = clean_text(
        _pick(row, "補足", "supplement_note", "評価／検討材料", "評価/検討材料")
    )
    if "長期休養明け" in supplement:
        reasons.append("長期休養明け")
    if clean_text(row.get("condition_fit_data_status")) == "missing_source_data":
        reasons.append("条件適性の元データ不足")
    trend = clean_text(_pick(row, "近3走傾向", "recent3_trend"))
    if trend in PRACTICAL_RULES["blocked_recent_trends"]:
        reasons.append(f"近3走が{trend}")
    if race_type == "jra":
        grade = _training_grade(_pick(row, "調教評価", "追切評価", "_調教評価記号"))
        if not grade:
            reasons.append("調教評価が未取得")
        elif grade not in PRACTICAL_RULES["accepted_jra_training_grades"]:
            reasons.append(f"調教評価{grade}は慎重材料")
    return _unique(reasons)


def practical_rules_snapshot() -> dict[str, Any]:
    return {
        key: list(value) if isinstance(value, tuple) else value
        for key, value in PRACTICAL_RULES.items()
    }


def practical_decision_snapshot(decision: InvestmentDecision) -> dict[str, Any]:
    selected = decision.selected
    return {
        "decision": decision.practical_decision or clean_text(decision.judgement).upper(),
        "reason": decision.practical_reason,
        "reason_lines": list(decision.practical_reason_lines or decision.reason_lines),
        "honmei_horse_no": decision.honmei_horse_no,
        "honmei_horse_name": decision.honmei_horse_name,
        "ticket_type": selected.ticket_type if selected else "",
        "tickets": list(selected.tickets) if selected else [],
        "investment": decision.total_stake,
        "config_version": decision.practical_config_version or PRACTICAL_CONFIG_VERSION,
    }


def ver3_mark(row: Mapping[str, Any]) -> str:
    """Read only Ver3-era mark fields; never fall back to mark_v4."""

    return clean_text(
        _pick(row, "表示印", "display_mark", "最終印", "old_final_mark", "元印", "印", "mark")
    )


def _watch_decision(
    race_type: str,
    lines: tuple[str, ...],
    *,
    honmei_no: str = "",
    honmei_name: str = "",
    audit: Mapping[str, Any] | None = None,
) -> InvestmentDecision:
    reason = clean_text(lines[0]) if lines else "購入条件が揃わないためWATCH。"
    audit_rows = (dict(audit or {}),) if audit else ()
    return InvestmentDecision(
        race_type=race_type,
        judgement=PRACTICAL_DECISION_WATCH,
        audit_rows=audit_rows,
        reason_lines=lines,
        total_stake=0,
        logic_version="practical",
        practical_decision=PRACTICAL_DECISION_WATCH,
        practical_reason=reason,
        practical_reason_lines=lines,
        practical_config_version=PRACTICAL_CONFIG_VERSION,
        honmei_horse_no=honmei_no,
        honmei_horse_name=honmei_name,
    )


def _training_grade(value: Any) -> str:
    match = re.match(r"\s*([A-D])", clean_text(value).upper())
    return match.group(1) if match else ""


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
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return clean_text(value).lower() in {"", "-", "—", "nan", "none", "null", "データなし", "未取得"}


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return clean_text(value).lower() in {"1", "true", "yes", "y", "○", "あり", "該当"}


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = clean_text(value)
        if text and text not in result:
            result.append(text)
    return result

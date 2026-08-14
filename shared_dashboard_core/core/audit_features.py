# -*- coding: utf-8 -*-
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd

from .condition_fit import evaluate_condition_fit
from .form_rank import add_form_rank_columns


AXIS_CONFIDENCE_CONFIG = {
    "raw_a": 95.0,
    "raw_b": 85.0,
    "top_gap_a": 3.0,
    "top_gap_warn": 1.5,
    "recent_range_warn": 12.0,
    "recent_range_stable": 6.0,
    "last_vs_average_warn": -6.0,
    "last_vs_average_good": 0.0,
    "market_gap_warn": 4.0,
}

ABILITY_BAND_CONFIG = {
    "top_band_gap": 5.0,
    "middle_band_gap": 12.0,
    "middle_rank_ratio": 0.55,
    "gap_large_top2": 5.0,
    "gap_large_top3": 8.0,
    "gap_large_top5": 12.0,
    "gap_middle_top2": 2.5,
    "gap_middle_top3": 5.0,
    "gap_middle_top5": 8.0,
}


AUDIT_OUTPUT_COLUMNS = [
    "過去1年最高指数",
    "year_max_index",
    "★最高指数",
    "★該当走",
    "★条件",
    "star_max_source",
    "star_max_index",
    "star_max_race",
    "star_max_venue",
    "star_max_distance",
    "star_max_surface",
    "star_max_turn",
    "star_match_level",
    "condition_fit_mark",
    "condition_fit_level",
    "condition_fit_reason",
    "condition_fit_data_status",
    "matched_past_runs",
    "practical_mark",
    "practical_warning_reason",
    "old_ai_score",
    "raw_score",
    "ability_display_score",
    "normalized_ai_score",
    "ai_rank",
    "old_final_mark",
    "original_mark",
    "final_mark_score",
    "market_score",
    "axis_confidence",
    "axis_confidence_reason",
    "ability_band",
    "ability_gap_level",
    "race_difficulty",
    "race_difficulty_reason",
    "display_comment",
    "display_mark",
    "display_group",
    "running_style_display",
    "old_watch_mark",
    "hole_candidate",
    "watch_horse",
    "ability_rank",
    "ability_rank_reason",
    "momentum_score",
    "momentum_rank",
    "momentum_reason",
    "recent3_trend",
    "recent3_slope",
    "recent3_volatility",
    "recent3_valid_count",
    "form_state",
    "overall_rank",
    "overall_rank_reason",
    "power_group",
    "power_group_label",
    "has_same_course",
    "has_same_distance",
    "has_same_turn",
    "has_heavy_track",
    "check_summary",
    "supplement_note",
]

AUDIT_EXPORT_COLUMNS = [
    "馬番",
    "馬名",
    "脚質",
    "running_style_display",
    "過去1年最高指数",
    "year_max_index",
    "★最高指数",
    "★該当走",
    "★条件",
    "★最高指数の取得元",
    "star_max_source",
    "star_max_index",
    "star_max_race",
    "star_max_venue",
    "star_max_distance",
    "star_max_surface",
    "star_max_turn",
    "star_match_level",
    "condition_fit_mark",
    "condition_fit_level",
    "condition_fit_reason",
    "condition_fit_data_status",
    "matched_past_runs",
    "practical_mark",
    "practical_warning_reason",
    "old_ai_score",
    "raw_score",
    "ability_display_score",
    "normalized_ai_score",
    "ai_rank",
    "old_final_mark",
    "original_mark",
    "final_mark_score",
    "market_score",
    "axis_confidence",
    "axis_confidence_reason",
    "ability_band",
    "ability_gap_level",
    "race_difficulty",
    "race_difficulty_reason",
    "display_comment",
    "display_mark",
    "display_group",
    "old_watch_mark",
    "hole_candidate",
    "watch_horse",
    "旧AI点",
    "能力評価値",
    "正規化AI点",
    "AI順位",
    "旧印",
    "元印",
    "総合評価監査点",
    "市場評価点",
    "軸信頼度",
    "軸信頼度理由",
    "能力帯",
    "能力差",
    "レース難易度",
    "レース難易度理由",
    "表示コメント",
    "表示印",
    "グループ",
    "脚質表示",
    "旧✓",
    "穴候補",
    "注意馬",
    "ability_rank",
    "ability_rank_reason",
    "momentum_score",
    "momentum_rank",
    "momentum_reason",
    "recent3_trend",
    "recent3_slope",
    "recent3_volatility",
    "recent3_valid_count",
    "form_state",
    "overall_rank",
    "overall_rank_reason",
    "power_group",
    "power_group_label",
    "has_same_course",
    "has_same_distance",
    "has_same_turn",
    "has_heavy_track",
    "能力ランク",
    "能力ランク理由",
    "勢いスコア",
    "勢いランク",
    "勢い理由",
    "近3走傾向",
    "状態",
    "総合ランク",
    "総合ランク理由",
    "勢力図グループ",
    "勢力図役割",
    "同競馬場",
    "同距離",
    "同回り",
    "重馬場実績",
    "チェック項目",
    "補足",
    "horse_score_v4",
    "race_rank_v4",
    "base_ability_score",
    "condition_score",
    "jockey_score",
    "age_weight_score",
    "training_score",
    "momentum_score_v4",
    "race_shape_score",
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
    "ver3_ability_core",
    "ability_core_source",
    "legacy_raw_score",
    "market_removed_adjustment",
    "market_ability_score",
    "market_ability_rank",
    "ability_band_v2",
    "能力帯比較",
    "ability_band_reason",
    "actual_odds",
    "fair_odds_display",
    "fair_odds_status",
    "fair_odds_reason",
    "running_style_market",
    "pace_mark_market",
    "pace_reason_market",
    "pace_scenario_market",
    "race_interval_market",
    "state_arrow",
    "state_label_market",
    "state_transition",
    "current_class_market",
    "previous_class_market",
    "best_recent_class_market",
    "class_shift_market",
    "class_basis_market",
    "jockey_market",
    "jockey_change_market",
    "weight_market",
    "weight_change_market",
    "condition_mark_market",
    "condition_reason_market",
    "training_market",
    "stable_comment_market",
    "positive_materials",
    "negative_materials",
    "plus_materials_display",
    "minus_materials_display",
]


@dataclass(frozen=True)
class AxisContext:
    top_raw: float | None
    second_raw: float | None
    top_gap: float | None


@dataclass(frozen=True)
class AbilityRaceContext:
    gap_level: str
    difficulty: str
    reason: str


def add_audit_evaluation_columns(df: pd.DataFrame | None, *, race_type: str = "nar") -> pd.DataFrame | None:
    """Add display/audit-only evaluation columns without changing scores or marks."""
    if df is None:
        return df
    result = df.copy()
    if result.empty:
        for column in AUDIT_OUTPUT_COLUMNS:
            if column not in result.columns:
                result[column] = []
        return result

    result["old_ai_score"] = _numeric_series(result, "AI点")
    result["raw_score"] = _numeric_series(result, "_raw_score")
    result["ability_display_score"] = result["raw_score"].round(1)
    result["normalized_ai_score"] = _numeric_series(result, "AI点")
    if "AI順位" in result.columns:
        result["ai_rank"] = _numeric_series(result, "AI順位")
    else:
        result["ai_rank"] = result["normalized_ai_score"].rank(method="min", ascending=False)
    result["old_final_mark"] = _text_series(result, "最終印")
    result["original_mark"] = result["old_final_mark"]
    result["final_mark_score"] = _first_numeric_series(result, ["総合評価点", "_最終印点", "総合評価"])
    result["market_score"] = _first_numeric_series(result, ["市場反映勝率", "推定勝率", "単勝期待値"])
    if "year_max_index" not in result.columns:
        result["year_max_index"] = _first_numeric_series(result, ["過去1年最高指数", "_year_max_index"])
    if "過去1年最高指数" not in result.columns:
        result["過去1年最高指数"] = result["year_max_index"]
    if "star_max_index" not in result.columns:
        result["star_max_index"] = _first_numeric_series(result, ["★最高指数", "★最高"])
    if "★最高指数" not in result.columns:
        result["★最高指数"] = result["star_max_index"]
    for column, default in (
        ("star_max_race", ""),
        ("star_max_venue", ""),
        ("star_max_distance", pd.NA),
        ("star_max_surface", ""),
        ("star_max_turn", ""),
        ("star_match_level", "none"),
        ("★該当走", ""),
        ("★条件", ""),
    ):
        if column not in result.columns:
            result[column] = default
    if "star_max_condition" in result.columns:
        condition = _text_series(result, "star_max_condition")
        result["★条件"] = _text_series(result, "★条件").where(_text_series(result, "★条件").astype(str).str.len().gt(0), condition)
    result["★該当走"] = _text_series(result, "★該当走").where(
        _text_series(result, "★該当走").astype(str).str.len().gt(0),
        _text_series(result, "star_max_race"),
    )
    if "star_max_source" in result.columns:
        star_source = _text_series(result, "star_max_source")
    else:
        star_source = _text_series(result, "★最高指数の取得元")
    result["star_max_source"] = star_source.where(star_source.astype(str).str.len().gt(0), "missing")
    result["★最高指数の取得元"] = result["star_max_source"]
    condition_fit_rows = [evaluate_condition_fit(row.to_dict()) for _, row in result.iterrows()]
    result["condition_fit_mark"] = [item.get("condition_fit_mark", "") for item in condition_fit_rows]
    result["condition_fit_level"] = [item.get("condition_fit_level", "none") for item in condition_fit_rows]
    result["condition_fit_reason"] = [item.get("condition_fit_reason", "") for item in condition_fit_rows]
    result["matched_past_runs"] = [item.get("matched_past_runs", []) for item in condition_fit_rows]

    axis_context = _axis_context(result["raw_score"])
    axis_values: list[str] = []
    axis_reasons: list[str] = []
    for _, row in result.iterrows():
        confidence, reason = _axis_confidence_for_row(row, axis_context, race_type=race_type)
        axis_values.append(confidence)
        axis_reasons.append(reason)
    result["axis_confidence"] = axis_values
    result["axis_confidence_reason"] = axis_reasons

    split = _split_watch_and_hole_candidates(result, race_type=race_type)
    result["old_watch_mark"] = split["old_watch_mark"]
    result["hole_candidate"] = split["hole_candidate"]
    result["watch_horse"] = split["watch_horse"]

    ability_context, ability_band = _ability_band_context(result["raw_score"])
    result["ability_band"] = ability_band
    result["ability_gap_level"] = ability_context.gap_level
    result["race_difficulty"] = ability_context.difficulty
    result["race_difficulty_reason"] = ability_context.reason
    result["display_comment"] = result.apply(_display_comment_for_row, axis=1)
    result["display_mark"] = _display_mark_series(result)
    result["display_group"] = result["display_mark"].map(_display_group_from_mark)
    result["running_style_display"] = _running_style_display_series(result)

    # Japanese aliases are kept for normal UI/CSV readability. They mirror the
    # snake_case audit columns and do not feed back into prediction logic.
    result["旧AI点"] = result["old_ai_score"]
    result["能力評価値"] = result["ability_display_score"]
    result["正規化AI点"] = result["normalized_ai_score"]
    result["旧印"] = result["old_final_mark"]
    result["元印"] = result["original_mark"]
    result["総合評価監査点"] = result["final_mark_score"]
    result["市場評価点"] = result["market_score"]
    result["軸信頼度"] = result["axis_confidence"]
    result["軸信頼度理由"] = result["axis_confidence_reason"]
    result["能力帯"] = result["ability_band"]
    result["能力差"] = result["ability_gap_level"]
    result["レース難易度"] = result["race_difficulty"]
    result["レース難易度理由"] = result["race_difficulty_reason"]
    result["表示コメント"] = result["display_comment"]
    result["表示印"] = result["display_mark"]
    result["グループ"] = result["display_group"]
    result["脚質表示"] = result["running_style_display"]
    result["旧✓"] = result["old_watch_mark"].map(_bool_label)
    result["穴候補"] = result["hole_candidate"].map(_bool_label)
    result["注意馬"] = result["watch_horse"].map(_bool_label)
    result = add_form_rank_columns(result, race_type=race_type)
    if "check_summary" not in result.columns:
        result["check_summary"] = _text_series(result, "チェック項目")
    if "supplement_note" not in result.columns:
        result["supplement_note"] = _text_series(result, "補足")
    return result


def build_audit_export_table(df: pd.DataFrame | None) -> pd.DataFrame:
    if df is None or not isinstance(df, pd.DataFrame) or df.empty:
        return pd.DataFrame()
    columns = [column for column in AUDIT_EXPORT_COLUMNS if column in df.columns]
    if not columns:
        return pd.DataFrame()
    return df.loc[:, columns].copy()


def audit_table_to_csv_bytes(df: pd.DataFrame) -> bytes:
    frame = _export_clean_frame(df)
    return frame.to_csv(index=False).encode("utf-8-sig")


def audit_table_to_json_bytes(df: pd.DataFrame) -> bytes:
    frame = _export_clean_frame(df)
    return frame.to_json(orient="records", force_ascii=False, indent=2).encode("utf-8")


def audit_table_to_markdown(df: pd.DataFrame) -> str:
    frame = _export_clean_frame(df)
    if frame.empty:
        return ""
    headers = [str(column) for column in frame.columns]
    lines = [
        "| " + " | ".join(_escape_markdown_cell(header) for header in headers) + " |",
        "| " + " | ".join("---" for _ in headers) + " |",
    ]
    for _, row in frame.iterrows():
        lines.append("| " + " | ".join(_escape_markdown_cell(row.get(column)) for column in frame.columns) + " |")
    return "\n".join(lines)


def _axis_context(raw: pd.Series) -> AxisContext:
    values = pd.to_numeric(raw, errors="coerce").dropna().sort_values(ascending=False)
    if values.empty:
        return AxisContext(None, None, None)
    top = float(values.iloc[0])
    second = float(values.iloc[1]) if len(values) >= 2 else None
    gap = (top - second) if second is not None else None
    return AxisContext(top, second, gap)


def _ability_band_context(raw: pd.Series) -> tuple[AbilityRaceContext, pd.Series]:
    values = pd.to_numeric(raw, errors="coerce")
    band = pd.Series("未評価", index=raw.index, dtype="object")
    valid = values.dropna().sort_values(ascending=False)
    if valid.empty:
        context = AbilityRaceContext("不明", "未判定", "能力評価値が不足しています")
        return context, band

    config = ABILITY_BAND_CONFIG
    top = float(valid.iloc[0])
    second = float(valid.iloc[1]) if len(valid) >= 2 else None
    third = float(valid.iloc[2]) if len(valid) >= 3 else None
    fifth_pos = min(4, len(valid) - 1)
    fifth_or_last = float(valid.iloc[fifth_pos])
    top2_gap = (top - second) if second is not None else None
    top3_gap = (top - third) if third is not None else None
    top5_range = top - fifth_or_last

    gap_level = _ability_gap_level(top2_gap, top3_gap, top5_range)
    difficulty = {"大": "絞りやすい", "中": "やや混戦", "小": "混戦"}.get(gap_level, "未判定")
    reason = {
        "大": "上位馬と他馬の能力評価値に差があります",
        "中": "上位数頭に差はありますが逆転余地があります",
        "小": "上位馬の能力評価値が接近しています",
    }.get(gap_level, "能力評価値の分布を判定できませんでした")

    ranks = values.rank(method="min", ascending=False)
    middle_rank_limit = max(3, int(len(valid) * config["middle_rank_ratio"] + 0.999))
    for idx, value in values.items():
        number = _safe_float(value)
        if number is None:
            continue
        gap = top - number
        rank = _safe_float(ranks.get(idx))
        if gap <= config["top_band_gap"]:
            band.loc[idx] = "上位帯"
        elif gap <= config["middle_band_gap"] or (rank is not None and rank <= middle_rank_limit):
            band.loc[idx] = "中位帯"
        else:
            band.loc[idx] = "下位帯"

    return AbilityRaceContext(gap_level, difficulty, reason), band


def _ability_gap_level(top2_gap: float | None, top3_gap: float | None, top5_range: float | None) -> str:
    config = ABILITY_BAND_CONFIG
    safe_top2 = top2_gap if top2_gap is not None else 999.0
    safe_top3 = top3_gap if top3_gap is not None else safe_top2
    safe_top5 = top5_range if top5_range is not None else safe_top3
    if (
        safe_top2 >= config["gap_large_top2"]
        and safe_top3 >= config["gap_large_top3"]
        and safe_top5 >= config["gap_large_top5"]
    ):
        return "大"
    if (
        safe_top2 >= config["gap_middle_top2"]
        or safe_top3 >= config["gap_middle_top3"]
        or safe_top5 >= config["gap_middle_top5"]
    ):
        return "中"
    return "小"


def _display_comment_for_row(row: pd.Series) -> str:
    gap_level = _text_value(row.get("ability_gap_level"))
    band = _text_value(row.get("ability_band"))
    if gap_level == "小":
        if _truthy(row.get("hole_candidate")) or _truthy(row.get("watch_horse")):
            return "能力差が小さい混戦で、配当面や展開材料も確認したい馬です。"
        if band == "上位帯":
            return "上位帯の一頭。能力差が小さいため、調教・展開・オッズも含めて比較したい馬です。"
        if band == "中位帯":
            return "中位帯ですが逆転余地があります。適性・展開・オッズをあわせて確認したい馬です。"
        return "能力差が小さいため、他馬との比較材料を確認したい馬です。"
    return ""


def _display_mark_series(df: pd.DataFrame) -> pd.Series:
    mark = _text_series(df, "old_final_mark")
    display = pd.Series("", index=df.index, dtype="object")
    core_mark = mark.isin(["◎", "○", "▲", "△"])
    display.loc[core_mark] = mark.loc[core_mark]
    hole = _bool_series(df, "hole_candidate")
    display.loc[hole] = "✓"
    return display


def _display_group_from_mark(value: Any) -> str:
    mark = _text_value(value)
    if mark == "◎":
        return "SS"
    if mark in {"○", "▲"}:
        return "A"
    if mark == "△":
        return "B"
    if mark in {"✓", "✔"}:
        return "C"
    return "Z"


def _running_style_display_series(df: pd.DataFrame) -> pd.Series:
    result = pd.Series("", index=df.index, dtype="object")
    for column in ("脚質", "running_style", "style"):
        if column not in df.columns:
            continue
        values = df[column].map(_display_running_style)
        result = result.where(result.astype(str).str.len().gt(0), values)
    return result


def _display_running_style(value: Any) -> str:
    text = _text_value(value)
    if not text:
        return ""
    if "逃" in text:
        return "逃げ"
    if "先" in text:
        return "先行"
    if "差" in text:
        return "差し"
    if "追" in text:
        return "追込"
    return text


def _axis_confidence_for_row(row: pd.Series, context: AxisContext, *, race_type: str) -> tuple[str, str]:
    raw = _safe_float(row.get("raw_score"))
    ai_rank = _safe_float(row.get("ai_rank"))
    if raw is None:
        return "C", "raw score欠損"

    points = 0
    reasons: list[str] = []
    config = AXIS_CONFIDENCE_CONFIG

    if raw >= config["raw_a"]:
        points += 2
        reasons.append("能力水準高め")
    elif raw >= config["raw_b"]:
        points += 1
        reasons.append("能力水準あり")
    else:
        points -= 1
        reasons.append("能力水準控えめ")

    if ai_rank is not None and ai_rank <= 2:
        points += 1
        reasons.append("AI順位上位")
    elif ai_rank is not None and ai_rank >= 7:
        points -= 1

    if ai_rank == 1 and context.top_gap is not None:
        if context.top_gap >= config["top_gap_a"]:
            points += 1
            reasons.append("2位差あり")
        elif context.top_gap < config["top_gap_warn"]:
            points -= 1
            reasons.append("上位僅差")
    elif ai_rank is not None and ai_rank <= 2 and context.top_gap is not None and context.top_gap < config["top_gap_warn"]:
        points -= 1
        reasons.append("上位僅差")

    recent_range = _recent_index_range(row)
    if recent_range is not None:
        if recent_range >= config["recent_range_warn"]:
            points -= 1
            reasons.append("近走の振れ幅あり")
        elif recent_range <= config["recent_range_stable"]:
            points += 1
            reasons.append("近走安定")

    last_value = _first_numeric_value(row, ["前走", "race1", "_last"])
    average_value = _first_numeric_value(row, ["平均指数", "3走平均", "avg5"])
    if last_value is not None and average_value is not None:
        diff = last_value - average_value
        if diff <= config["last_vs_average_warn"]:
            points -= 1
            reasons.append("前走が平均より低め")
        elif diff >= config["last_vs_average_good"]:
            points += 1
            reasons.append("前走が平均以上")

    major_negative = _safe_float(row.get("_重大マイナス数"))
    if major_negative is not None:
        if major_negative >= 2:
            points -= 2
            reasons.append("重大マイナス複数")
        elif major_negative >= 1:
            points -= 1
            reasons.append("マイナス材料あり")

    style = _text_value(row.get("脚質"))
    pace_mark = _text_value(row.get("展開印"))
    if style == "追" and pace_mark != "展":
        points -= 1
        reasons.append("脚質リスク")

    market_rank = _first_numeric_value(row, ["勝率順位", "市場順位", "人気"])
    if ai_rank is not None and market_rank is not None:
        gap = market_rank - ai_rank
        if gap >= config["market_gap_warn"]:
            points -= 1
            reasons.append("市場評価との乖離")
        elif ai_rank <= 3 and market_rank <= 3:
            points += 1
            reasons.append("市場評価と一致")

    if str(race_type).lower() == "jra":
        grade = _training_grade(row)
        if grade in {"S", "A"}:
            points += 1
            reasons.append("調教評価良好")
        elif grade in {"C", "D"}:
            points -= 1
            reasons.append("調教評価は慎重材料")

    if points >= 3:
        confidence = "A"
    elif points >= 1:
        confidence = "B"
    else:
        confidence = "C"
    return confidence, "、".join(_unique(reasons)[:2]) or "確認材料少なめ"


def _split_watch_and_hole_candidates(df: pd.DataFrame, *, race_type: str) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)
    mark = _text_series(df, "最終印")
    core = mark.isin(["◎", "○", "▲", "△"])
    old_watch = mark.eq("✓") | mark.eq("☆")
    data_shortage = _bool_series(df, "_地方指数データ不足") if str(race_type).lower() == "nar" else pd.Series(False, index=df.index)

    odds = _first_numeric_series(df, ["単勝オッズ", "オッズ"])
    ai_rank = _first_numeric_series(df, ["ai_rank", "AI順位"])
    final_score = _first_numeric_series(df, ["final_mark_score", "総合評価点", "_最終印点", "総合評価"]).fillna(-9999)
    ev = _first_numeric_series(df, ["単勝期待値"]).fillna(0)
    market_score = _first_numeric_series(df, ["market_score", "市場反映勝率", "推定勝率"]).fillna(0)
    material = (
        _text_series(df, "評価/検討材料")
        + " / "
        + _text_series(df, "評価／検討材料")
        + " / "
        + _text_series(df, "調教/評価/検討材料")
        + " / "
        + _text_series(df, "印理由")
        + " / "
        + _text_series(df, "クラス変動")
        + " / "
        + _text_series(df, "展開印")
        + " / "
        + _text_series(df, "馬タイプ")
    )

    has_material = material.str.contains(
        "高指数|最高指数|距離|コース|クラス降級|相手弱化|展開|対戦|調教|好気配|配当|穴|中穴|大穴|単勝",
        regex=True,
        na=False,
    )
    odds_band = odds.between(8.0, 60.0, inclusive="both").fillna(False).astype(bool)
    rank_band = (
        ai_rank.le(8)
        | final_score.rank(method="min", ascending=False).le(8)
        | market_score.rank(method="min", ascending=False).le(6)
    ).fillna(False).astype(bool)
    ev_signal = ev.ge(1.10).fillna(False).astype(bool)
    hole_pool = (~core) & (~data_shortage) & (old_watch | ((odds_band | ev_signal) & has_material & rank_band))

    score = final_score.copy()
    score += old_watch.astype(float) * 3.0
    score += has_material.astype(float) * 2.0
    score += odds.ge(10).fillna(False).astype(float) * 1.0
    score += odds.ge(20).fillna(False).astype(float) * 0.8
    score += ev_signal.astype(float) * 2.0
    score += market_score.rank(method="min", ascending=False).le(6).fillna(False).astype(float) * 0.8
    score = score.where(hole_pool, -9999)

    hole = pd.Series(False, index=df.index)
    if bool(hole_pool.any()):
        selected = score.sort_values(ascending=False).head(2).index
        selected = [idx for idx in selected if bool(hole_pool.loc[idx])]
        hole.loc[selected] = True

    watch_signal = old_watch | ((~core) & (~data_shortage) & (has_material | rank_band | odds.ge(10).fillna(False)))
    watch = watch_signal & ~hole
    result["old_watch_mark"] = old_watch.fillna(False).astype(bool)
    result["hole_candidate"] = hole.fillna(False).astype(bool)
    result["watch_horse"] = watch.fillna(False).astype(bool)
    return result


def _numeric_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(pd.NA, index=df.index, dtype="Float64")
    return pd.to_numeric(df[column], errors="coerce")


def _first_numeric_series(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    result = pd.Series(pd.NA, index=df.index, dtype="Float64")
    for column in columns:
        if column not in df.columns:
            continue
        values = pd.to_numeric(df[column], errors="coerce")
        result = result.where(result.notna(), values)
    return result


def _text_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series("", index=df.index)
    return df[column].map(_text_value)


def _bool_series(df: pd.DataFrame, column: str) -> pd.Series:
    if column not in df.columns:
        return pd.Series(False, index=df.index)
    return df[column].map(_truthy).fillna(False).astype(bool)


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
    except Exception:
        if value is None:
            return None
    try:
        number = pd.to_numeric(value, errors="coerce")
    except Exception:
        return None
    try:
        if pd.isna(number):
            return None
    except Exception:
        return None
    return float(number)


def _first_numeric_value(row: pd.Series, columns: list[str]) -> float | None:
    for column in columns:
        if column not in row.index:
            continue
        value = _safe_float(row.get(column))
        if value is not None:
            return value
    return None


def _recent_index_range(row: pd.Series) -> float | None:
    values: list[float] = []
    raw_prev = row.get("_prev_values")
    if isinstance(raw_prev, list):
        for value in raw_prev:
            number = _safe_float(value)
            if number is not None:
                values.append(number)
    for column in ["3走前", "2走前", "前走", "race3", "race2", "race1"]:
        value = _safe_float(row.get(column))
        if value is not None:
            values.append(value)
    if len(values) < 2:
        return None
    return max(values) - min(values)


def _training_grade(row: pd.Series) -> str:
    text = ""
    for column in ["調教評価", "_oikiri_grade", "追切評価"]:
        text = _text_value(row.get(column))
        if text:
            break
    if not text:
        return ""
    upper = text.upper()
    for grade in ("S", "A", "B", "C", "D"):
        if grade in upper:
            return grade
    if "◎" in text:
        return "A"
    if "○" in text:
        return "B"
    if "▲" in text:
        return "C"
    return ""


def _text_value(value: Any) -> str:
    try:
        if value is None or pd.isna(value):
            return ""
    except Exception:
        if value is None:
            return ""
    text = str(value).strip()
    if text.lower() in {"", "nan", "none", "<na>", "nat", "-"}:
        return ""
    return text


def _truthy(value: Any) -> bool:
    try:
        if value is None or pd.isna(value):
            return False
    except Exception:
        if value is None:
            return False
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    return text in {"true", "1", "yes", "y", "○", "あり"}


def _bool_label(value: Any) -> str:
    return "○" if _truthy(value) else ""


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result


def _export_clean_frame(df: pd.DataFrame) -> pd.DataFrame:
    frame = df.copy()
    frame = frame.astype("object")
    return frame.where(pd.notna(frame), "")


def _escape_markdown_cell(value: Any) -> str:
    text = _text_value(value)
    return text.replace("|", "\\|").replace("\n", "<br>")

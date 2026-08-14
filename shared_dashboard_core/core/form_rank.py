# -*- coding: utf-8 -*-
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Any

import pandas as pd


ABILITY_RANK_THRESHOLDS = {
    "S": 82.0,
    "A": 76.0,
    "B": 69.0,
    "C": 62.0,
}

MOMENTUM_RANK_THRESHOLDS = {
    "S": 75.0,
    "A": 62.0,
    "B": 48.0,
    "C": 35.0,
}

MOMENTUM_SCORE_CONFIG = {
    "base": 50.0,
    "slope_strong": 5.0,
    "slope_mid": 2.5,
    "slope_weak": 1.0,
    "last_vs_avg_good": 5.0,
    "last_vs_avg_ok": 0.0,
    "last_vs_avg_bad": -3.0,
    "last_vs_avg_very_bad": -8.0,
    "sharp_change": 8.0,
    "volatility_warn": 18.0,
    "volatility_bad": 28.0,
    "layoff_warn_days": 90.0,
    "layoff_bad_days": 180.0,
}

CHECK_ITEMS = [
    ("has_same_course", "同競馬場"),
    ("has_same_distance", "同距離"),
    ("has_same_turn", "同回り"),
    ("has_heavy_track", "重馬場実績"),
]


@dataclass(frozen=True)
class MomentumResult:
    score: float | None
    rank: str
    reason: str
    trend: str
    slope: float | None
    volatility: float | None
    valid_count: int


def add_form_rank_columns(df: pd.DataFrame | None, *, race_type: str = "nar") -> pd.DataFrame | None:
    """Add display/audit columns only; prediction scores and marks are not modified."""
    if df is None:
        return df
    result = df.copy()
    if result.empty:
        for column in _empty_columns():
            if column not in result.columns:
                result[column] = []
        return result

    ability_values = _first_numeric_series(result, ["ability_display_score", "raw_score", "_raw_score", "能力評価値"])
    ranks: list[str] = []
    reasons: list[str] = []
    for value in ability_values:
        rank, reason = ability_rank(value)
        ranks.append(rank)
        reasons.append(reason)
    result["ability_rank"] = ranks
    result["ability_rank_reason"] = reasons
    result["能力ランク"] = result["ability_rank"]
    result["能力ランク理由"] = result["ability_rank_reason"]

    momentum_rows: list[MomentumResult] = []
    for _, row in result.iterrows():
        momentum_rows.append(momentum_for_row(row, race_type=race_type))
    result["momentum_score"] = [item.score for item in momentum_rows]
    result["momentum_rank"] = [item.rank for item in momentum_rows]
    result["momentum_reason"] = [item.reason for item in momentum_rows]
    result["recent3_trend"] = [item.trend for item in momentum_rows]
    result["recent3_slope"] = [item.slope for item in momentum_rows]
    result["recent3_volatility"] = [item.volatility for item in momentum_rows]
    result["recent3_valid_count"] = [item.valid_count for item in momentum_rows]
    result["form_state"] = result["recent3_trend"].map(form_state_from_trend)
    result["勢いスコア"] = result["momentum_score"]
    result["勢いランク"] = result["momentum_rank"]
    result["勢い理由"] = result["momentum_reason"]
    result["近3走傾向"] = result["recent3_trend"]
    result["状態"] = result["form_state"]

    check_frame = build_check_columns(result)
    for column in check_frame.columns:
        result[column] = check_frame[column]
    result["チェック項目"] = result.apply(check_summary_text, axis=1)
    result["補足"] = result.apply(supplement_text_for_row, axis=1)
    result["check_summary"] = result["チェック項目"]
    result["supplement_note"] = result["補足"]

    overall_ranks: list[str] = []
    overall_reasons: list[str] = []
    power_groups: list[str] = []
    power_labels: list[str] = []
    for _, row in result.iterrows():
        overall, reason = overall_rank_for_row(row, race_type=race_type)
        overall_ranks.append(overall)
        overall_reasons.append(reason)
        group, label = power_group_for_row(row)
        power_groups.append(group)
        power_labels.append(label)
    result["overall_rank"] = overall_ranks
    result["overall_rank_reason"] = overall_reasons
    result["power_group"] = power_groups
    result["power_group_label"] = power_labels
    result["総合ランク"] = result["overall_rank"]
    result["総合ランク理由"] = result["overall_rank_reason"]
    result["勢力図グループ"] = result["power_group"]
    result["勢力図役割"] = result["power_group_label"]
    return result


def ability_rank(value: Any) -> tuple[str, str]:
    number = _safe_float(value)
    if number is None:
        return "未評価", "能力評価値が不足しています"
    if number >= ABILITY_RANK_THRESHOLDS["S"]:
        return "S", f"能力評価値{number:.1f}がS基準以上"
    if number >= ABILITY_RANK_THRESHOLDS["A"]:
        return "A", f"能力評価値{number:.1f}がA基準"
    if number >= ABILITY_RANK_THRESHOLDS["B"]:
        return "B", f"能力評価値{number:.1f}がB基準"
    if number >= ABILITY_RANK_THRESHOLDS["C"]:
        return "C", f"能力評価値{number:.1f}がC基準"
    return "D", f"能力評価値{number:.1f}がD基準"


def momentum_for_row(row: pd.Series, *, race_type: str = "nar") -> MomentumResult:
    values = [
        _safe_float(row.get("3走前")),
        _safe_float(row.get("2走前")),
        _safe_float(row.get("前走")),
    ]
    valid = [value for value in values if value is not None]
    valid_count = len(valid)
    if valid_count == 0:
        return MomentumResult(None, "未判定", "近3走データが不足", "未判定", None, None, 0)
    if valid_count == 1:
        return MomentumResult(None, "判定保留", "近3走データが1走のみ", "判定保留", None, 0.0, 1)

    first = valid[0]
    last = valid[-1]
    slope = (last - first) / max(valid_count - 1, 1)
    volatility = max(valid) - min(valid)
    trend = recent3_trend(values)
    score = float(MOMENTUM_SCORE_CONFIG["base"])
    reasons: list[str] = []

    if slope >= MOMENTUM_SCORE_CONFIG["slope_strong"]:
        score += 18
        reasons.append("上昇傾向が強い")
    elif slope >= MOMENTUM_SCORE_CONFIG["slope_mid"]:
        score += 12
        reasons.append("上昇傾向")
    elif slope >= MOMENTUM_SCORE_CONFIG["slope_weak"]:
        score += 6
        reasons.append("やや上向き")
    elif slope <= -MOMENTUM_SCORE_CONFIG["slope_strong"]:
        score -= 18
        reasons.append("下降傾向が強い")
    elif slope <= -MOMENTUM_SCORE_CONFIG["slope_mid"]:
        score -= 12
        reasons.append("下降傾向")
    elif slope <= -MOMENTUM_SCORE_CONFIG["slope_weak"]:
        score -= 6
        reasons.append("やや下降")
    else:
        reasons.append("横ばい")

    if trend == "連続上昇":
        score += 10
        reasons.append("連続上昇")
    elif trend == "連続下降":
        score -= 10
        reasons.append("連続下降")
    elif trend in {"持ち直し", "反発"}:
        score += 8
        reasons.append(trend)

    average = _first_numeric_value(row, ["平均指数", "3走平均", "avg5"])
    if average is not None:
        last_vs_avg = last - average
        if last_vs_avg >= MOMENTUM_SCORE_CONFIG["last_vs_avg_good"]:
            score += 8
            reasons.append("前走が平均以上")
        elif last_vs_avg >= MOMENTUM_SCORE_CONFIG["last_vs_avg_ok"]:
            score += 4
            reasons.append("前走が平均並み以上")
        elif last_vs_avg <= MOMENTUM_SCORE_CONFIG["last_vs_avg_very_bad"]:
            score -= 10
            reasons.append("前走が平均を大きく下回る")
        elif last_vs_avg <= MOMENTUM_SCORE_CONFIG["last_vs_avg_bad"]:
            score -= 5
            reasons.append("前走が平均以下")

    if valid_count >= 2:
        last_change = valid[-1] - valid[-2]
        if last_change >= MOMENTUM_SCORE_CONFIG["sharp_change"]:
            score += 8
            reasons.append("前走急上昇")
        elif last_change <= -MOMENTUM_SCORE_CONFIG["sharp_change"]:
            score -= 10
            reasons.append("前走急落")

    if volatility >= MOMENTUM_SCORE_CONFIG["volatility_bad"]:
        score -= 8
        reasons.append("振れ幅大")
    elif volatility >= MOMENTUM_SCORE_CONFIG["volatility_warn"]:
        score -= 4
        reasons.append("振れ幅あり")

    interval_days = _first_numeric_value(row, ["_days_since_last", "レース間隔日数"])
    if interval_days is not None:
        if interval_days >= MOMENTUM_SCORE_CONFIG["layoff_bad_days"]:
            score -= 8
            reasons.append("長期休養明け")
        elif interval_days >= MOMENTUM_SCORE_CONFIG["layoff_warn_days"]:
            score -= 4
            reasons.append("間隔長め")

    if str(race_type).lower() == "jra":
        grade = training_grade(row)
        if grade in {"S", "A", "A相当"}:
            score += 3
            reasons.append("調教良好")
        elif grade in {"B", "B相当"}:
            score += 1
        elif grade in {"C", "C相当"}:
            score -= 1
        elif grade == "D":
            score -= 3

    score = max(0.0, min(100.0, score))
    rank = momentum_rank(score)
    reason = "、".join(_unique(reasons)[:3]) or "近走推移を確認"
    return MomentumResult(round(score, 1), rank, reason, trend, round(slope, 2), round(volatility, 1), valid_count)


def recent3_trend(values: list[Any]) -> str:
    nums = [_safe_float(value) for value in values]
    if sum(value is not None for value in nums) < 2:
        return "判定保留"
    if all(value is not None for value in nums):
        a, b, c = nums  # type: ignore[misc]
        if a < b < c:
            return "連続上昇"
        if a > b > c:
            return "連続下降"
        if b < a and c > b:
            return "持ち直し" if c >= a - 5 else "反発"
        if max(a, b, c) - min(a, b, c) <= 4:
            return "横ばい"
        if c > a:
            return "上昇傾向"
        if c < a:
            return "下降傾向"
        return "横ばい"
    valid = [value for value in nums if value is not None]
    if valid[-1] > valid[0]:
        return "上昇傾向"
    if valid[-1] < valid[0]:
        return "下降傾向"
    return "横ばい"


def momentum_rank(score: Any) -> str:
    number = _safe_float(score)
    if number is None:
        return "未判定"
    if number >= MOMENTUM_RANK_THRESHOLDS["S"]:
        return "S"
    if number >= MOMENTUM_RANK_THRESHOLDS["A"]:
        return "A"
    if number >= MOMENTUM_RANK_THRESHOLDS["B"]:
        return "B"
    if number >= MOMENTUM_RANK_THRESHOLDS["C"]:
        return "C"
    return "D"


def form_state_from_trend(value: Any) -> str:
    trend = _text_value(value)
    if trend in {"連続上昇", "上昇"}:
        return "上昇"
    if trend in {"横ばい", "安定"}:
        return "安定"
    if trend in {"連続下降", "下降", "急落"}:
        return "下降"
    if trend in {"持ち直し", "反発"}:
        return "反発"
    if trend in {"判定保留", "未判定"}:
        return "判定なし"
    return "判定なし"


def overall_rank_for_row(row: pd.Series, *, race_type: str = "nar") -> tuple[str, str]:
    points = 0.0
    reasons: list[str] = []
    ability = _text_value(row.get("ability_rank"))
    momentum = _text_value(row.get("momentum_rank"))
    ability_points = {"S": 5.0, "A": 4.0, "B": 3.0, "C": 2.0, "D": 1.0}.get(ability, 0.0)
    momentum_points = {"S": 2.0, "A": 1.5, "B": 1.0, "C": 0.0, "D": -1.0}.get(momentum, 0.0)
    points += ability_points + momentum_points
    if ability:
        reasons.append(f"能力{ability}")
    if momentum:
        reasons.append(f"勢い{momentum}")

    axis = _text_value(row.get("axis_confidence"))
    if axis == "A":
        points += 1.0
        reasons.append("軸信頼A")
    elif axis == "B":
        points += 0.5
    elif axis == "C":
        points -= 0.5

    band = _text_value(row.get("ability_band"))
    if band == "上位帯":
        points += 0.6
    elif band == "下位帯":
        points -= 0.5

    difficulty = _text_value(row.get("race_difficulty"))
    if difficulty == "絞りやすい" and band == "上位帯":
        points += 0.6
    elif difficulty == "混戦":
        points -= 0.2

    if _text_value(row.get("展開印")):
        points += 0.5
        reasons.append("展開材料")

    market = _safe_float(row.get("market_score"))
    if market is None:
        market = _first_numeric_value(row, ["市場反映勝率", "推定勝率"])
    if market is not None:
        if market >= 25:
            points += 0.7
            reasons.append("市場評価")
        elif market <= 8:
            points -= 0.3

    if str(race_type).lower() == "jra":
        grade = training_grade(row)
        if grade in {"S", "A", "A相当"}:
            points += 0.8
            reasons.append("調教良好")
        elif grade in {"C", "C相当"}:
            points -= 0.2
        elif grade == "D":
            points -= 0.5

    if points >= 8.0:
        return "S", "、".join(_unique(reasons)[:3]) or "総合材料上位"
    if points >= 6.5:
        return "A", "、".join(_unique(reasons)[:3]) or "総合材料あり"
    if points >= 5.0:
        return "B", "、".join(_unique(reasons)[:3]) or "比較対象"
    if points >= 3.5:
        return "C", "、".join(_unique(reasons)[:3]) or "押さえ候補"
    return "D", "、".join(_unique(reasons)[:3]) or "材料控えめ"


def power_group_for_row(row: pd.Series) -> tuple[str, str]:
    mark = _text_value(row.get("display_mark")) or _text_value(row.get("表示印")) or _text_value(row.get("最終印"))
    ability_band = _text_value(row.get("ability_band"))
    ability_rank_value = _text_value(row.get("ability_rank"))
    overall = _text_value(row.get("overall_rank"))
    hole = _truthy(row.get("hole_candidate")) or mark == "✓"
    watch = _truthy(row.get("watch_horse"))
    if mark == "◎" or (ability_band == "上位帯" and overall in {"S", "A"}):
        return "SS", "軸"
    if mark in {"○", "▲"} or (ability_band == "上位帯" and ability_rank_value in {"S", "A"}):
        return "A", "相手本線"
    if hole or mark == "△" or watch:
        return "B", "相手・穴"
    if ability_band == "中位帯":
        return "C", "押さえ"
    return "D", "消し寄り"


def build_check_columns(df: pd.DataFrame) -> pd.DataFrame:
    result = pd.DataFrame(index=df.index)
    same_flags = df.apply(_same_condition_flags, axis=1)
    result["has_same_course"] = same_flags.map(lambda item: bool(item.get("course")))
    result["has_same_distance"] = same_flags.map(lambda item: bool(item.get("distance")))
    result["has_same_turn"] = same_flags.map(lambda item: bool(item.get("turn")))
    result["has_heavy_track"] = df.apply(_has_heavy_track, axis=1)
    result["同競馬場"] = result["has_same_course"]
    result["同距離"] = result["has_same_distance"]
    result["同回り"] = result["has_same_turn"]
    result["重馬場実績"] = result["has_heavy_track"]
    return result


def check_summary_text(row: pd.Series) -> str:
    lines = []
    for column, label in CHECK_ITEMS:
        mark = "○" if _truthy(row.get(column)) else "×"
        lines.append(f"{mark} {label}")
    return "\n".join(lines)


def supplement_text_for_row(row: pd.Series) -> str:
    candidates = []
    text = " / ".join(
        _text_value(row.get(column))
        for column in ["評価/検討材料", "評価／検討材料", "調教/評価/検討材料", "コメント", "新聞コメント", "厩舎コメント"]
    )
    for keyword in ["初ブリンカー", "初ダート", "初芝", "去勢明け", "長期休養明け"]:
        if keyword in text and keyword not in candidates:
            candidates.append(keyword)
    if _truthy(row.get("_is_layoff")) and "長期休養明け" not in candidates:
        candidates.append("長期休養明け")
    return " / ".join(candidates) if candidates else "なし"


def training_grade(row: pd.Series) -> str:
    text = _text_value(_first_value(row, ["調教評価", "追切評価", "_調教評価記号", "_追切評価記号"]))
    if not text:
        return ""
    for grade in ("S", "A", "B", "C", "D"):
        if grade in text:
            return grade
    if "◎" in text:
        return "A相当"
    if "○" in text:
        return "B相当"
    if "▲" in text:
        return "C相当"
    return text


def recent3_display(row: pd.Series) -> str:
    values = [_safe_float(row.get("3走前")), _safe_float(row.get("2走前")), _safe_float(row.get("前走"))]
    parts = ["-" if value is None else f"{value:g}" for value in values]
    trend = _text_value(row.get("recent3_trend"))
    arrow = "↗" if trend in {"連続上昇", "上昇傾向", "持ち直し", "反発"} else ("↘" if trend in {"連続下降", "下降傾向"} else "→")
    return f"{' → '.join(parts)} {arrow} {trend or '未判定'}"


def _same_condition_flags(row: pd.Series) -> dict[str, bool]:
    flags_text = _text_value(row.get("_same_condition_flags"))
    same_condition = "True" in flags_text or "true" in flags_text
    star_index = _safe_float(row.get("star_max_index"))
    star_level = _text_value(row.get("star_match_level"))
    star_venue = _text_value(row.get("star_max_venue"))
    star_distance = _safe_float(row.get("star_max_distance"))
    condition_text = _text_value(row.get("★条件")) or _text_value(row.get("star_max_condition"))
    same_course = bool(same_condition or star_venue or condition_text or star_index is not None)
    same_distance = bool(star_distance is not None or star_index is not None or "m" in condition_text or re.search(r"\d{3,4}", condition_text))
    same_turn = bool(star_level == "venue_distance_surface_turn" or _text_value(row.get("star_max_turn")) or "右" in condition_text or "左" in condition_text)
    return {"course": same_course, "distance": same_distance, "turn": same_turn}


def _has_heavy_track(row: pd.Series) -> bool:
    text = " / ".join(
        _text_value(row.get(column))
        for column in ["馬場実績", "馬場適性", "評価/検討材料", "評価／検討材料", "調教/評価/検討材料", "コメント"]
    )
    if not text:
        return False
    return any(keyword in text for keyword in ["重", "不良", "道悪", "稍重", "馬場実績"])


def _empty_columns() -> list[str]:
    return [
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
        "チェック項目",
        "状態",
        "補足",
    ]


def _first_numeric_series(df: pd.DataFrame, columns: list[str]) -> pd.Series:
    result = pd.Series(pd.NA, index=df.index, dtype="Float64")
    for column in columns:
        if column not in df.columns:
            continue
        values = df[column].map(_safe_float)
        result = result.where(result.notna(), pd.Series(values, index=df.index, dtype="Float64"))
    return result


def _first_numeric_value(row: pd.Series, columns: list[str]) -> float | None:
    for column in columns:
        if column not in row.index:
            continue
        value = _safe_float(row.get(column))
        if value is not None:
            return value
    return None


def _first_value(row: pd.Series, columns: list[str]) -> Any:
    for column in columns:
        if column not in row.index:
            continue
        value = row.get(column)
        if _text_value(value):
            return value
    return ""


def _safe_float(value: Any) -> float | None:
    try:
        if value is None or pd.isna(value):
            return None
    except Exception:
        if value is None:
            return None
    if isinstance(value, str):
        text = value.strip()
        if not text or text in {"-", "未", "未取得", "データなし", "None", "nan", "<NA>"}:
            return None
        match = re.search(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
        if not match:
            return None
        text = match.group(0)
    else:
        text = value
    try:
        number = float(text)
    except Exception:
        return None
    return None if math.isnan(number) else number


def _text_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    text = str(value).strip()
    return "" if text.lower() in {"nan", "none", "<na>"} else text


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    text = _text_value(value).lower()
    return text in {"true", "1", "yes", "y", "○", "あり", "該当"}


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        if value and value not in result:
            result.append(value)
    return result

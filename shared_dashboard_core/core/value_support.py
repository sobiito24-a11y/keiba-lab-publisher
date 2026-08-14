# -*- coding: utf-8 -*-
from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Iterable, Mapping

import pandas as pd

from .horse_trust import build_horse_trust_materials
from .purchase_conditions import clean_text, horse_no, to_float


VALUE_FIELD_NAMES = (
    "value_signal",
    "value_candidate",
    "value_reason",
    "value_score",
    "value_odds_at_decision",
    "value_ability_band",
    "value_ability_rank",
    "value_current_rank",
    "value_mark",
    "value_plus_materials",
    "value_minus_materials",
    "training_display",
    "training_rank",
    "training_short_comment",
    "stable_comment_display",
    "course_material_label",
    "course_material_detail",
    "netkeiba_favorable_label",
    "netkeiba_favorable_source",
    "estimated_position_label",
    "value_audit",
)

PROJECT_ROOT = Path(__file__).resolve().parents[1]
BACKTEST_JSON = PROJECT_ROOT / "work" / "mark_betting_backtest" / "mark_betting_backtest_summary.json"

MARK_ALIASES = {"✓": "☆", "✔": "☆"}
POSITIVE_GRADES = {"◎", "○", "+", "A", "B", "S"}
NEGATIVE_GRADES = {"△", "×", "D"}

DEFAULT_BACKTEST_REFERENCE = {
    "meta": {"usable_races": 176, "jra_races": 64, "nar_races": 112, "source": "baseline_reference"},
    "mark_summary": [
        {"印": "◎", "券種": "単勝", "購入数": 143, "的中率": 28.0, "回収率": 72.8, "最大払戻": 0, "購入参考": "見送り参考"},
        {"印": "◎", "券種": "複勝", "購入数": 143, "的中率": 59.4, "回収率": 80.4, "最大払戻": 0, "購入参考": "参考"},
        {"印": "○", "券種": "単勝", "購入数": 209, "的中率": 14.8, "回収率": 63.1, "最大払戻": 0, "購入参考": "見送り参考"},
        {"印": "○", "券種": "複勝", "購入数": 209, "的中率": 40.7, "回収率": 73.1, "最大払戻": 0, "購入参考": "見送り参考"},
        {"印": "▲", "券種": "単勝", "購入数": 176, "的中率": 12.5, "回収率": 67.5, "最大払戻": 0, "購入参考": "見送り参考"},
        {"印": "▲", "券種": "複勝", "購入数": 176, "的中率": 45.5, "回収率": 101.2, "最大払戻": 0, "購入参考": "参考"},
        {"印": "△", "券種": "単勝", "購入数": 352, "的中率": 8.8, "回収率": 66.8, "最大払戻": 0, "購入参考": "見送り参考"},
        {"印": "△", "券種": "複勝", "購入数": 352, "的中率": 26.1, "回収率": 59.9, "最大払戻": 0, "購入参考": "見送り参考"},
        {"印": "☆", "券種": "単勝", "購入数": 310, "的中率": 8.1, "回収率": 69.4, "最大払戻": 0, "購入参考": "見送り参考"},
        {"印": "☆", "券種": "複勝", "購入数": 310, "的中率": 24.5, "回収率": 72.2, "最大払戻": 0, "購入参考": "見送り参考"},
    ],
    "box_summary": [
        {"対象印": "◎○▲", "券種": "馬連", "レース数": 176, "的中率": 14.8, "回収率": 93.9, "購入参考": "参考"},
    ],
    "condition_summary": [],
    "value_summary": [],
}


def attach_value_signals(
    rows: Iterable[Mapping[str, Any]],
    race_type: str = "jra",
    *,
    max_horses: int = 2,
) -> list[dict[str, Any]]:
    """Attach display-only value labels to current prediction rows.

    This function never changes marks, ranks, ability values, groups, or scores.
    It only appends explanatory fields used by UI/history/backtest audit.
    """

    enriched = [dict(row) for row in rows]
    candidates: list[tuple[int, float, int]] = []
    for idx, row in enumerate(enriched):
        support = build_value_support(row, race_type)
        enriched[idx].update(support)
        if support["value_candidate"]:
            candidates.append((idx, float(support["value_score"] or 0), _odds_sort_value(support["value_odds_at_decision"])))

    candidates.sort(key=lambda item: (item[1], item[2]), reverse=True)
    selected = {idx for idx, _score, _odds in candidates[:max_horses]}
    for idx, row in enumerate(enriched):
        if idx in selected:
            row["value_signal"] = True
        else:
            if row.get("value_candidate") and candidates:
                row["value_audit"] = {**_mapping(row.get("value_audit")), "selection_note": "妙味候補だが同一レース上限外"}
            row["value_signal"] = False
    return enriched


def build_value_support(row: Mapping[str, Any], race_type: str = "jra") -> dict[str, Any]:
    training = training_display(row, race_type)
    course = course_material_display(row)
    ability_band = _pick_text(row, "能力帯", "ability_band", "ability_rank_label", "display_ability_band")
    ability_rank = _to_int(_pick(row, "能力順位", "ability_rank_for_backtest", "ability_rank", "能力順位_v3"))
    current_rank = _to_int(_pick(row, "AI今回評価順位", "ai_current_rank", "ai_rank", "AI順位", "race_rank_v4"))
    mark = normalize_value_mark(_pick_text(row, "表示印", "display_mark", "最終印", "original_mark", "旧印", "印", "mark"))
    odds = to_float(_pick(row, "単勝オッズ", "オッズ", "odds", "単勝"))

    plus, minus, score = value_materials(row, race_type, ability_band, ability_rank, current_rank, mark, training, course)
    market_ok = odds is not None and odds >= 8.0
    high_odds_guard = odds is not None and odds >= 50.0
    enough_materials = len(plus) >= 2 and score >= (6 if high_odds_guard else 4)
    severe_low = _is_low_ability(ability_band, ability_rank, current_rank)
    candidate = bool(market_ok and enough_materials and not severe_low)
    reason = ""
    if candidate:
        material_text = "＋".join(plus[:4])
        odds_text = f"{odds:.1f}倍" if odds is not None else "市場評価低め"
        reason = f"{material_text}に対して{odds_text}"

    return {
        "value_signal": False,
        "value_candidate": candidate,
        "value_reason": reason,
        "value_score": score,
        "value_odds_at_decision": odds,
        "value_ability_band": ability_band,
        "value_ability_rank": ability_rank,
        "value_current_rank": current_rank,
        "value_mark": mark,
        "value_plus_materials": plus,
        "value_minus_materials": minus,
        "training_display": training.get("display", ""),
        "training_rank": training.get("rank", ""),
        "training_short_comment": training.get("comment", ""),
        "stable_comment_display": stable_comment_display(row, race_type),
        "course_material_label": course.get("label", ""),
        "course_material_detail": course.get("detail", ""),
        "netkeiba_favorable_label": course.get("netkeiba_label", ""),
        "netkeiba_favorable_source": course.get("netkeiba_source", ""),
        "estimated_position_label": course.get("position", ""),
        "value_audit": {
            "market_ok": market_ok,
            "high_odds_guard": high_odds_guard,
            "enough_materials": enough_materials,
            "severe_low": severe_low,
            "score": score,
            "plus_materials": plus,
            "minus_materials": minus,
            "course_material": course,
            "training": training,
        },
    }


def value_materials(
    row: Mapping[str, Any],
    race_type: str,
    ability_band: str,
    ability_rank: int | None,
    current_rank: int | None,
    mark: str,
    training: Mapping[str, Any],
    course: Mapping[str, Any],
) -> tuple[list[str], list[str], int]:
    plus: list[str] = []
    minus: list[str] = []
    score = 0

    band = ability_band.upper()
    band_label = ability_band if "帯" in ability_band else f"{ability_band}帯"
    if band in {"AA", "SS", "S", "A"} or "上位" in ability_band:
        plus.append(band_label)
        score += 2
    elif band == "B" or "中位" in ability_band:
        plus.append(band_label or "B帯")
        score += 1
    elif band in {"C", "Z", "D"} or "下位" in ability_band:
        minus.append(band_label)
        score -= 1

    if ability_rank is not None:
        if ability_rank <= 3:
            plus.append(f"能力{ability_rank}位")
            score += 2
        elif ability_rank <= 5:
            plus.append(f"能力{ability_rank}位")
            score += 1
        elif ability_rank >= 8:
            minus.append(f"能力{ability_rank}位")
            score -= 1

    if current_rank is not None:
        if current_rank <= 4:
            plus.append(f"今回{current_rank}位")
            score += 1
        elif current_rank >= 8:
            minus.append(f"今回{current_rank}位")
            score -= 1

    if mark in {"◎", "○"}:
        plus.append(f"印{mark}")
        score += 2
    elif mark in {"▲", "△", "☆"}:
        plus.append(f"印{mark}")
        score += 1

    for label, key in (("距離実績", "距離指数"), ("コース実績", "コース指数")):
        number = to_float(_pick(row, key, f"{key}_value"))
        if number is None:
            continue
        if number >= 60:
            plus.append(f"{label}◎")
            score += 1
        elif number < 35:
            minus.append(f"{label}△")

    state = _pick_text(row, "状態", "form_state", "勢いランク", "momentum_rank", "勢い", "recent3_trend")
    if any(token in state for token in ("上昇", "良化", "反発", "持ち直し", "安定", "A", "S")):
        plus.append("近走上昇")
        score += 1
    elif any(token in state for token in ("下降", "急落", "不安", "D")):
        minus.append("近走下降")
        score -= 1

    if course.get("tone") == "plus":
        plus.append(clean_text(course.get("short")) or "展開向き")
        score += 1
    elif course.get("tone") == "minus":
        minus.append(clean_text(course.get("short")) or "展開注意")
        score -= 1

    if clean_text(training.get("rank")) in {"S", "A", "B"}:
        plus.append(f"調教{training.get('rank')}")
        score += 1
    elif clean_text(training.get("rank")) in {"D"}:
        minus.append("調教D")
        score -= 1

    jockey = _pick_text(row, "騎手詳細", "jockey_detail", "騎手", "jockey")
    if "継続" in jockey:
        plus.append("継続騎乗")
    elif any(token in jockey for token in ("乗替", "乗り替", "替")):
        minus.append("乗替")

    return _unique(plus), _unique(minus), score


def training_display(row: Mapping[str, Any], race_type: str = "jra") -> dict[str, str]:
    if clean_text(race_type).lower() == "nar":
        return {"display": "", "rank": "", "comment": "", "source": ""}
    raw = _pick_text(row, "調教評価", "追切評価", "_調教評価記号", "training_grade", "調教ランク")
    rank = _extract_training_rank(raw)
    if not rank:
        return {"display": "", "rank": "", "comment": "", "source": raw}
    comment = _short_training_comment(
        _pick_text(row, "調教短評", "追切短評", "追切内容", "調教コメント", "training_comment")
    )
    arrow, default_comment = _training_rank_display(rank)
    comment = comment or default_comment
    display = f"調教{rank}{arrow}" + (f" {comment}" if comment else "")
    return {"display": display, "rank": rank, "comment": comment, "source": raw}


def stable_comment_display(row: Mapping[str, Any], race_type: str = "jra", *, max_length: int = 64) -> str:
    if clean_text(race_type).lower() == "nar":
        return ""
    text = _pick_text(row, "厩舎コメント", "新聞コメント", "stable_comment", "stable_comment_market", "一言コメント")
    if not text:
        return ""
    return _stable_comment_summary(text)


def course_material_display(row: Mapping[str, Any]) -> dict[str, str]:
    raw_mark = _pick_text(row, "展開印", "pace_mark", "展開評価", "枠脚質評価", "枠順×脚質評価")
    raw_reason = _pick_text(row, "course_development_reason", "コース材料理由", "course_material_reason")
    position = normalize_position(_pick_text(row, "推定位置", "想定位置", "position_band", "脚質", "running_style"))
    netkeiba = _pick_text(
        row,
        "netkeiba推定有利馬",
        "推定有利馬",
        "有利馬",
        "netkeiba_favorable",
        "estimated_favorable",
        "_position_favorable_horse",
    )
    distance = _pick_text(row, "距離指数")
    course = _pick_text(row, "コース指数")
    label = ""
    tone = "neutral"
    short = ""

    if any(token in raw_mark for token in ("◎", "○", "有利", "向き", "一致", "好")):
        label = f"○ {position}向き" if position else "○ 展開/コース向き"
        tone = "plus"
        short = "展開向き"
    elif any(token in raw_mark for token in ("△", "×", "厳", "注意", "不利")):
        label = f"△ {position}想定は注意" if position else "△ 展開注意"
        tone = "minus"
        short = "展開注意"
    elif position:
        label = f"± {position}想定"
    elif "推定有利馬" in raw_reason:
        label = "○ 推定有利馬"
    else:
        condition_mark = _pick_text(row, "condition_fit_mark", "条件実績マーク")
        if condition_mark == "★":
            label = "＋ 同場同距離実績"
        elif condition_mark == "☆":
            label = "＋ 同回り同距離実績"
        elif condition_mark == "※":
            label = "＋ 同距離実績"

    netkeiba_label = ""
    if (netkeiba and netkeiba.lower() not in {"false", "0", "なし", "—", "-"}) or "推定有利馬" in raw_reason:
        netkeiba_label = "○ 推定有利馬"

    detail_parts = [
        f"今回コース材料={raw_mark}" if raw_mark else "",
        f"コース材料理由={raw_reason}" if raw_reason else "",
        f"推定位置={position}" if position else "",
        f"距離指数={distance}" if distance else "",
        f"コース指数={course}" if course else "",
        netkeiba_label,
    ]
    return {
        "label": label,
        "tone": tone,
        "short": short,
        "detail": " / ".join(part for part in detail_parts if part),
        "position": position,
        "netkeiba_label": netkeiba_label,
        "netkeiba_source": netkeiba,
    }


def normalize_position(value: Any) -> str:
    text = clean_text(value)
    if not text:
        return ""
    if any(token in text for token in ("逃", "ハナ")):
        return "逃げ"
    if any(token in text for token in ("先", "好位", "前")):
        return "先団"
    if any(token in text for token in ("中", "差")):
        return "中団"
    if any(token in text for token in ("後", "追")):
        return "後方"
    return text[:12]


def value_reference_rows(summary_path: str | Path | None = None) -> list[dict[str, Any]]:
    payload = load_backtest_reference(summary_path)
    rows: list[dict[str, Any]] = []
    for item in payload.get("mark_summary", []) or []:
        mark = clean_text(item.get("印"))
        bet = clean_text(item.get("券種"))
        if (mark, bet) in {("▲", "複勝"), ("◎", "単勝"), ("◎", "複勝")}:
            rows.append(_reference_item(f"{mark} {bet}", item))
    for item in payload.get("box_summary", []) or []:
        if clean_text(item.get("対象印")) == "◎○▲" and clean_text(item.get("券種")) == "馬連":
            rows.append(_reference_item("◎○▲ 馬連BOX", item))
    for item in payload.get("value_summary", []) or []:
        if clean_text(item.get("対象")) in {"妙味あり", "全体"}:
            rows.append(_reference_item("妙味あり 単複", item))
            break
    return rows[:5]


def current_mark_reference(row: Mapping[str, Any], summary_path: str | Path | None = None) -> dict[str, Any] | None:
    mark = normalize_value_mark(_pick_text(row, "表示印", "display_mark", "最終印", "印", "mark"))
    odds = to_float(_pick(row, "単勝オッズ", "オッズ", "odds", "単勝"))
    if not mark:
        return None
    payload = load_backtest_reference(summary_path)
    target = None
    for item in payload.get("mark_summary", []) or []:
        if clean_text(item.get("印")) == mark and clean_text(item.get("券種")) == "単勝":
            target = item
            break
    if not target:
        return None
    odds_text = f"{odds:.1f}倍" if odds is not None else "オッズ欠損"
    out = _reference_item(f"{mark} {odds_text} 単勝", target)
    out["odds"] = odds
    return out


def load_backtest_reference(summary_path: str | Path | None = None) -> dict[str, Any]:
    path = Path(summary_path) if summary_path else BACKTEST_JSON
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            pass
    return DEFAULT_BACKTEST_REFERENCE


def normalize_value_mark(value: Any) -> str:
    text = clean_text(value)
    return MARK_ALIASES.get(text, text)


def _reference_item(label: str, item: Mapping[str, Any]) -> dict[str, Any]:
    sample = _pick(item, "購入数", "レース数", "対象レース数", "購入レース数") or 0
    roi = to_float(_pick(item, "回収率", "単勝回収率", "複勝回収率"))
    hit_rate = to_float(_pick(item, "的中率", "単勝的中率", "複勝的中率"))
    top1_roi = to_float(_pick(item, "最大払戻除外回収率", "上位1件除外回収率"))
    category = reference_category(sample, roi, hit_rate, top1_roi)
    return {
        "label": label,
        "sample": int(to_float(sample) or 0),
        "hit_rate": hit_rate,
        "roi": roi,
        "top1_excluded_roi": top1_roi,
        "max_payout": to_float(_pick(item, "最大払戻", "最高払戻1件")),
        "category": category,
    }


def reference_category(sample: Any, roi: float | None, hit_rate: float | None, top1_roi: float | None = None) -> str:
    count = int(to_float(sample) or 0)
    if count < 30:
        return "サンプル不足"
    if roi is None:
        return "未校正"
    if roi >= 95 and (top1_roi is None or top1_roi >= 75) and (hit_rate is None or hit_rate >= 12):
        return "参考"
    if roi < 80:
        return "見送り参考"
    return "未校正"


def _extract_training_rank(raw: str) -> str:
    text = clean_text(raw)
    if not text:
        return ""
    upper = text.upper()
    symbol_map = {"◎": "A", "○": "B", "△": "C", "×": "D"}
    for symbol, rank in symbol_map.items():
        if symbol in text:
            return rank
    patterns = [
        r"調教\s*([SABCD])",
        r"追切\s*([SABCD])",
        r"評価\s*([SABCD])",
        r"ランク\s*([SABCD])",
        r"^([SABCD])(?:$|[：:\s）)\]])",
    ]
    for pattern in patterns:
        match = re.search(pattern, upper)
        if match:
            return match.group(1)
    return ""


def _training_rank_display(rank: str) -> tuple[str, str]:
    mapping = {
        "S": ("↑", "動き抜群"),
        "A": ("↑", "動き抜群"),
        "B": ("↑", "仕上上々"),
        "C": ("→", "平行線"),
        "D": ("↓", "物足りず"),
    }
    return mapping.get(clean_text(rank).upper(), ("", ""))


def _short_training_comment(text: str) -> str:
    comment = clean_text(text)
    if not comment or _looks_like_lap_text(comment):
        return ""
    positive_strong = ("抜群", "鋭", "迫力", "絶好", "好時計", "伸び")
    positive = ("上昇", "好調", "維持", "良化", "順調", "軽快", "気配", "上々", "仕上")
    neutral = ("平行", "変わらず", "まずまず", "普通")
    negative = ("物足", "重い", "一息", "不安", "遅れ")
    if any(word in comment for word in positive_strong):
        return "動き抜群"
    if any(word in comment for word in positive):
        return "仕上上々"
    if any(word in comment for word in neutral):
        return "平行線"
    if any(word in comment for word in negative):
        return "物足りず"
    return ""


def _stable_comment_summary(text: str) -> str:
    comment = clean_text(text)
    if not comment:
        return ""
    positive = ("好気配", "順調", "良化", "上向", "上昇", "仕上", "期待", "力を出せ", "態勢", "充実")
    negative = ("一息", "重い", "不安", "慎重", "使ってから", "良化途上", "時間", "割引", "厳しい", "物足")
    if any(word in comment for word in positive):
        return "厩舎コメント：↑ 好気配"
    if any(word in comment for word in negative):
        return "厩舎コメント：↓ 慎重"
    return "厩舎コメント：→ 平常"


def _looks_like_lap_text(text: str) -> bool:
    chars = [char for char in text if not char.isspace()]
    if not chars:
        return False
    numeric = sum(1 for char in chars if char.isdigit() or char in ".()[]-/:")
    return numeric / len(chars) >= 0.55


def _is_low_ability(ability_band: str, ability_rank: int | None, current_rank: int | None) -> bool:
    band = clean_text(ability_band).upper()
    if band in {"Z", "D"} or "下位" in ability_band:
        return True
    return (ability_rank is not None and ability_rank >= 8) and (current_rank is not None and current_rank >= 8)


def _odds_sort_value(value: Any) -> int:
    odds = to_float(value)
    if odds is None:
        return 0
    return int(min(odds, 50) * 10)


def _pick(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name not in row:
            continue
        value = row.get(name)
        if not _missing(value):
            return value
    return None


def _pick_text(row: Mapping[str, Any], *names: str) -> str:
    return clean_text(_pick(row, *names))


def _to_int(value: Any) -> int | None:
    number = to_float(value)
    if number is None:
        return None
    return int(number)


def _missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return clean_text(value).lower() in {"", "-", "—", "nan", "none", "null", "データなし", "未取得"}


def _mapping(value: Any) -> dict[str, Any]:
    return dict(value) if isinstance(value, Mapping) else {}


def _unique(values: Iterable[str]) -> list[str]:
    out: list[str] = []
    for value in values:
        text = clean_text(value)
        if text and text not in out:
            out.append(text)
    return out


def value_by_horse_number(rows: Iterable[Mapping[str, Any]]) -> dict[str, dict[str, Any]]:
    return {horse_no(_pick(row, "馬番", "馬", "horse_no", "horse_number")): dict(row) for row in rows}

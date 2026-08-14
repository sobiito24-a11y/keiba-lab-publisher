# -*- coding: utf-8 -*-
"""Result-free ability/price comparison layer.

Ability and its bands read the explicit, unadjusted Ver3 time-index core.
Odds, popularity, jockey, weight, interval, state, pace, and generated material
lists are attached as independent comparison columns only. For old saved
snapshots without the explicit core, ``legacy raw - recorded adjustment`` is a
compatibility fallback; current predictions do not need that reconstruction.
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import unicodedata
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import pandas as pd

from .condition_fit import evaluate_condition_fit
from .course_materials import JOCKEY_COURSE_MIN_STARTS
from .purchase_conditions import clean_text, horse_no, to_float


MARKET_COMPARE_VERSION = "market-compare-1.4"
MARKET_HISTORY_ROOT = Path("prediction_history") / "market_compare"

# AA is deliberately exceptional: a race-leading raw ability value must clear
# both an absolute level and a separation test.  A is the normal leading band.
# All thresholds are fixed before result inspection and use the Ver3 core only.
ABILITY_BAND_RULES = {
    "aa_min_score": 95.0,
    "aa_min_gap": 5.0,
    "a_gap": 5.0,
    "b_gap": 12.0,
    "c_gap": 20.0,
}

MARKET_OUTPUT_COLUMNS = (
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
    "course_condition_market",
    "provider_pace_market",
    "position_start_market",
    "position_corner3_market",
    "position_corner4_market",
    "position_start_label_market",
    "position_corner3_label_market",
    "position_corner4_label_market",
    "position_path_market",
    "position_coverage_market",
    "favorable_position_label_market",
    "four_corner_place_rates_market",
    "course_development_mark",
    "course_development_reason",
    "course_development_source",
    "track_bias_status_market",
    "lap_prediction_status_market",
    "predicted_3f_coverage_market",
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
    "previous_jockey_market",
    "jockey_display_market",
    "jockey_change_market",
    "jockey_course_stats_market",
    "jockey_course_sample_market",
    "jockey_course_mark_market",
    "jockey_course_reason_market",
    "jockey_course_source_market",
    "weight_market",
    "weight_change_market",
    "body_weight_market",
    "body_weight_change_market",
    "condition_mark_market",
    "condition_reason_market",
    "training_market",
    "stable_comment_market",
    "positive_materials",
    "negative_materials",
    "plus_materials_display",
    "minus_materials_display",
    "current_evaluation_balance",
    "current_evaluation_positive_count",
    "current_evaluation_negative_count",
    "current_evaluation_rank",
    "ai_current_mark",
    "ai_current_reason",
)

SNAPSHOT_COLUMNS = (
    "馬番",
    "馬名",
    "馬年齢",
    "性齢",
    "market_ability_score",
    "ver3_ability_core",
    "ability_core_source",
    "legacy_raw_score",
    "market_removed_adjustment",
    "market_ability_rank",
    "ability_band_v2",
    "actual_odds",
    "人気",
    "fair_odds_display",
    "fair_odds_status",
    "running_style_market",
    "pace_mark_market",
    "pace_reason_market",
    "course_condition_market",
    "provider_pace_market",
    "position_start_market",
    "position_corner3_market",
    "position_corner4_market",
    "position_start_label_market",
    "position_corner3_label_market",
    "position_corner4_label_market",
    "position_path_market",
    "position_coverage_market",
    "favorable_position_label_market",
    "four_corner_place_rates_market",
    "course_development_mark",
    "course_development_reason",
    "course_development_source",
    "track_bias_status_market",
    "lap_prediction_status_market",
    "predicted_3f_coverage_market",
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
    "previous_jockey_market",
    "jockey_display_market",
    "jockey_change_market",
    "jockey_course_stats_market",
    "jockey_course_sample_market",
    "jockey_course_mark_market",
    "jockey_course_reason_market",
    "jockey_course_source_market",
    "weight_market",
    "weight_change_market",
    "body_weight_market",
    "body_weight_change_market",
    "距離指数",
    "コース指数",
    "condition_mark_market",
    "condition_reason_market",
    "3走前",
    "2走前",
    "前走",
    "平均指数",
    "過去1年最高指数",
    "training_market",
    "stable_comment_market",
    "positive_materials",
    "negative_materials",
    "current_evaluation_balance",
    "current_evaluation_positive_count",
    "current_evaluation_negative_count",
    "current_evaluation_rank",
    "ai_current_mark",
    "ai_current_reason",
)

_RESULT_FIELD_PATTERN = re.compile(
    r"(?:着順|確定|払戻|配当|result|finish|payoff|payout|profit|hit)",
    flags=re.IGNORECASE,
)


def apply_market_compare_to_result(result: Any) -> Any:
    """Attach the new comparison view without changing legacy prediction data."""

    from .ver4_engine import attach_condition_fit_sources, merge_prediction_tables

    merged = merge_prediction_tables(result)
    merged = attach_condition_fit_sources(merged, getattr(result, "debug_info", {}) or {})
    evaluated = evaluate_market_table(
        merged,
        getattr(result, "race_mode", "jra"),
        getattr(result, "race_info", {}) or {},
    )
    by_number = {
        horse_no(_pick(row, "馬番", "horse_no", "horse_number", "馬", "horse_no_key_v4")): row
        for row in evaluated.to_dict("records")
    }
    for attribute in ("overall_table", "horse_evaluation"):
        source = getattr(result, attribute, None)
        if source is None or not isinstance(source, pd.DataFrame):
            continue
        target = source.copy()
        for column in MARKET_OUTPUT_COLUMNS:
            if column not in target.columns:
                target[column] = pd.Series([None] * len(target), index=target.index, dtype="object")
        for index, raw in target.iterrows():
            key = horse_no(_pick(raw.to_dict(), "馬番", "horse_no", "horse_number", "馬"))
            values = by_number.get(key, {})
            for column in MARKET_OUTPUT_COLUMNS:
                if column in values:
                    target.at[index, column] = values[column]
        setattr(result, attribute, target)

    signature = market_prediction_signature(evaluated, result.race_mode, result.race_info)
    debug = dict(getattr(result, "debug_info", {}) or {})
    previous_selection = ((debug.get("market_compare") or {}).get("user_selection") or {})
    debug["market_compare"] = {
        "version": MARKET_COMPARE_VERSION,
        "ability_band_rules": dict(ABILITY_BAND_RULES),
        "ability_source": "explicit unadjusted Ver3 time-index core (legacy adjustment fallback for old snapshots only)",
        "calibration": calibration_status(),
        "pace": race_pace_snapshot(evaluated),
        "race_summary": build_race_summary(evaluated),
        "prediction_signature": signature,
        "horses": safe_snapshot_records(evaluated),
        "user_selection": previous_selection,
    }
    result.debug_info = debug
    result.logic_version = "market"
    return result


def evaluate_market_table(
    table: pd.DataFrame | Sequence[Mapping[str, Any]] | None,
    race_type: str,
    race_info: Mapping[str, Any] | None = None,
) -> pd.DataFrame:
    """Build independent comparison facts from pre-race inputs only."""

    frame = _as_frame(table)
    if frame.empty:
        for column in MARKET_OUTPUT_COLUMNS:
            if column not in frame.columns:
                frame[column] = pd.Series(dtype="object")
        return frame

    result = frame.copy()
    race_type = "nar" if clean_text(race_type).lower() == "nar" else "jra"
    race_info = race_info or {}
    explicit_core = _first_numeric_series(
        result,
        ("_ver3_ability_core", "ver3_ability_core"),
    )
    legacy_ability = _first_numeric_series(
        result,
        ("raw_score", "_raw_score", "能力評価値", "ability_display_score"),
    )
    removed_adjustment = _first_numeric_series(result, ("_market_non_ability_adjustment",)).fillna(0.0)
    legacy_fallback = legacy_ability - removed_adjustment
    ability = explicit_core.where(explicit_core.notna(), legacy_fallback)
    result["ver3_ability_core"] = ability.round(1)
    result["ability_core_source"] = pd.Series(
        ["explicit_ver3_core" if pd.notna(value) else "legacy_adjustment_fallback" for value in explicit_core],
        index=result.index,
        dtype="object",
    )
    result["legacy_raw_score"] = legacy_ability.round(1)
    result["market_removed_adjustment"] = removed_adjustment.round(1)
    result["market_ability_score"] = ability.round(1)
    result["market_ability_rank"] = ability.rank(method="min", ascending=False).astype("Int64")
    bands, band_reasons = _ability_bands(ability)
    result["ability_band_v2"] = bands
    result["能力帯比較"] = bands
    result["ability_band_reason"] = band_reasons
    result["actual_odds"] = _first_numeric_series(result, ("単勝オッズ", "オッズ", "単勝", "odds"))
    result["fair_odds_display"] = "未校正"
    result["fair_odds_status"] = "uncalibrated"
    result["fair_odds_reason"] = calibration_status()["reason"]

    styles = [_normalize_style(_pick(raw.to_dict(), "脚質", "running_style_display", "running_style", "style")) for _, raw in result.iterrows()]
    result["running_style_market"] = styles
    pace = _pace_context(styles)
    result["pace_scenario_market"] = pace["scenario"]

    row_outputs: list[dict[str, Any]] = []
    for position, (_, raw) in enumerate(result.iterrows()):
        row = raw.to_dict()
        style = styles[position]
        condition = evaluate_condition_fit(row, race_info)
        state = _state(row)
        class_info = _class_info(row, race_info)
        interval = _interval(row)
        weight, weight_change = _weight(row)
        body_weight, body_weight_change = _body_weight(row)
        jockey, previous_jockey, jockey_change = _jockey(row)
        provider_pace = clean_text(_pick(row, "_netkeiba_pace")).upper()
        pace_mark, pace_reason = _human_pace_for_horse(style, provider_pace, pace)
        course_development = _course_development(row, style, pace_mark, pace_reason)
        jockey_course = _jockey_course(row)
        training = _training(row, race_type)
        stable_comment = clean_text(_pick(row, "厩舎コメント", "新聞コメント", "stable_comment"))
        plus, minus = _materials(
            row=row,
            race_type=race_type,
            condition=condition,
            state=state,
            class_info=class_info,
            interval=interval,
            weight_change=weight_change,
            course_development=course_development,
            jockey_course=jockey_course,
            training=training,
        )
        row_outputs.append(
            {
                "running_style_market": style or "未取得",
                "pace_mark_market": pace_mark,
                "pace_reason_market": pace_reason,
                "race_interval_market": interval,
                "course_condition_market": clean_text(_pick(row, "_course_condition_html")) or "未取得",
                "provider_pace_market": provider_pace or "未取得",
                "position_start_market": clean_text(_pick(row, "_estimated_position_start")) or "未取得",
                "position_corner3_market": clean_text(_pick(row, "_estimated_position_corner3")) or "未取得",
                "position_corner4_market": clean_text(_pick(row, "_estimated_position_corner4")) or "未取得",
                "position_start_label_market": clean_text(_pick(row, "_estimated_position_start_label")) or "位置不明",
                "position_corner3_label_market": clean_text(_pick(row, "_estimated_position_corner3_label")) or "位置不明",
                "position_corner4_label_market": clean_text(_pick(row, "_estimated_position_corner4_label")) or "位置不明",
                "position_path_market": clean_text(_pick(row, "_estimated_position_path")),
                "position_coverage_market": clean_text(_pick(row, "_position_coverage")) or "未取得",
                "favorable_position_label_market": clean_text(_pick(row, "_favorable_position_label")) or "未取得",
                "four_corner_place_rates_market": clean_text(_pick(row, "_four_corner_place_rates")) or "未取得",
                "course_development_mark": course_development["mark"],
                "course_development_reason": course_development["reason"],
                "course_development_source": course_development["source"],
                "track_bias_status_market": clean_text(_pick(row, "_track_bias_status")) or "html内に存在しない",
                "lap_prediction_status_market": clean_text(_pick(row, "_lap_prediction_status")) or "html内に存在しない",
                "predicted_3f_coverage_market": clean_text(_pick(row, "_predicted_3f_coverage")) or "未取得",
                "state_arrow": state["arrow"],
                "state_label_market": state["label"],
                "state_transition": state["transition"],
                "current_class_market": class_info["current"],
                "previous_class_market": class_info["previous"],
                "best_recent_class_market": class_info["best"],
                "class_shift_market": class_info["shift"],
                "class_basis_market": class_info["basis"],
                "jockey_market": jockey,
                "previous_jockey_market": previous_jockey,
                "jockey_display_market": _jockey_display(
                    jockey,
                    previous_jockey,
                    jockey_change,
                    jockey_course,
                ),
                "jockey_change_market": jockey_change,
                "jockey_course_stats_market": jockey_course["display"],
                "jockey_course_sample_market": jockey_course["sample"],
                "jockey_course_mark_market": jockey_course["mark"],
                "jockey_course_reason_market": jockey_course["reason"],
                "jockey_course_source_market": jockey_course["source"],
                "weight_market": weight,
                "weight_change_market": weight_change,
                "body_weight_market": body_weight,
                "body_weight_change_market": body_weight_change,
                "condition_mark_market": condition.get("condition_fit_mark") or "",
                "condition_reason_market": condition.get("condition_fit_reason") or "未取得",
                "training_market": training,
                "stable_comment_market": stable_comment,
                "positive_materials": plus,
                "negative_materials": minus,
                "plus_materials_display": " / ".join(plus),
                "minus_materials_display": " / ".join(minus),
            }
        )
    for column in MARKET_OUTPUT_COLUMNS:
        if column in result.columns:
            continue
        result[column] = [item.get(column) for item in row_outputs]
    # These columns were created before the row pass and should be retained.
    for column in row_outputs[0]:
        result[column] = [item.get(column) for item in row_outputs]
    return _attach_current_evaluation(result)


def _ability_bands(values: pd.Series) -> tuple[pd.Series, pd.Series]:
    numeric = pd.to_numeric(values, errors="coerce")
    bands = pd.Series("Z", index=values.index, dtype="object")
    reasons = pd.Series("能力材料不足", index=values.index, dtype="object")
    valid = numeric.dropna().sort_values(ascending=False)
    if valid.empty:
        return bands, reasons
    top = float(valid.iloc[0])
    second = float(valid.iloc[1]) if len(valid) >= 2 else None
    top_gap = top - second if second is not None else 0.0
    rules = ABILITY_BAND_RULES
    aa_index = valid.index[0] if top >= rules["aa_min_score"] and top_gap >= rules["aa_min_gap"] else None
    for index, value in numeric.items():
        if pd.isna(value):
            continue
        score = float(value)
        gap = top - score
        if index == aa_index:
            band = "AA"
            reason = f"能力{score:.1f}・2位差{top_gap:.1f}"
        elif gap <= rules["a_gap"]:
            band = "A"
            reason = f"首位差{gap:.1f}"
        elif gap <= rules["b_gap"]:
            band = "B"
            reason = f"首位差{gap:.1f}"
        elif gap <= rules["c_gap"]:
            band = "C"
            reason = f"首位差{gap:.1f}"
        else:
            band = "Z"
            reason = f"首位差{gap:.1f}"
        bands.at[index] = band
        reasons.at[index] = reason
    return bands, reasons


def calibration_status() -> dict[str, Any]:
    return {
        "status": "uncalibrated",
        "display": "AI適正オッズ：未校正",
        "reason": (
            "同梱188Rは各レースの本命1頭のみで、全出走馬の能力帯・順位と勝敗を"
            "development/holdoutに分けて校正できないため"
        ),
        "required_next": "全頭の予測時点スナップショットと確定勝敗を蓄積して時系列holdoutで検証",
    }


def price_band_rows(table: pd.DataFrame | None) -> dict[str, list[dict[str, Any]]]:
    result = {band: [] for band in ("AA", "A", "B", "C", "Z")}
    if table is None or not isinstance(table, pd.DataFrame) or table.empty:
        return result
    for row in table.to_dict("records"):
        band = clean_text(_pick(row, "ability_band_v2", "能力帯比較")) or "Z"
        if band not in result:
            band = "Z"
        result[band].append(
            {
                "horse_no": horse_no(_pick(row, "馬番", "horse_no", "馬")),
                "horse_name": clean_text(_pick(row, "馬名", "horse_name")),
                "odds": to_float(_pick(row, "actual_odds", "単勝オッズ", "オッズ", "odds")),
                "ability": to_float(_pick(row, "market_ability_score", "raw_score", "能力評価値")),
                "fair_odds": clean_text(_pick(row, "fair_odds_display")) or "未校正",
            }
        )
    for rows in result.values():
        rows.sort(key=lambda item: (item["odds"] is None, item["odds"] or math.inf, _horse_sort_key(item["horse_no"])))
    return result


def race_pace_snapshot(table: pd.DataFrame | None) -> dict[str, Any]:
    if table is None or not isinstance(table, pd.DataFrame) or table.empty:
        return {"scenario": "判定保留", "counts": {}, "horses": {}}
    styles = [clean_text(value) for value in table.get("running_style_market", pd.Series("", index=table.index))]
    context = _pace_context(styles)
    horses = {key: [] for key in ("逃", "先", "差", "追", "未取得")}
    for row in table.to_dict("records"):
        style = clean_text(_pick(row, "running_style_market"))
        key = style if style in horses else "未取得"
        number = horse_no(_pick(row, "馬番", "horse_no", "馬"))
        if number:
            horses[key].append(number)
    return {"scenario": context["scenario"], "counts": context["counts"], "horses": horses}


def build_race_summary(table: pd.DataFrame | None) -> list[str]:
    if table is None or not isinstance(table, pd.DataFrame) or table.empty:
        return ["能力上位：データ不足", "A帯オッズ：未取得", "展開：脚質データ不足", "主な変化：判定保留"]
    ranked = table.sort_values(["market_ability_rank", "market_ability_score"], ascending=[True, False])
    top = ranked.head(3)
    labels = [f"{horse_no(_pick(row, '馬番', '馬'))}{clean_text(_pick(row, '馬名'))}" for row in top.to_dict("records")]
    ability_line = "能力上位：" + "、".join(label for label in labels if label)
    a_rows = price_band_rows(table).get("A", [])
    if a_rows:
        price_bits = [f"{item['horse_no']} {item['odds']:.1f}倍" for item in a_rows if item["odds"] is not None]
        market_line = "A帯オッズ：" + "、".join(price_bits) if price_bits else "A帯オッズ：未取得"
    else:
        market_line = "A帯オッズ：該当なし"
    pace = race_pace_snapshot(table)
    counts = pace.get("counts") or {}
    provider_values = [
        clean_text(value).upper()
        for value in table.get("provider_pace_market", pd.Series("", index=table.index))
        if clean_text(value).upper() in {"H", "M", "S"}
    ]
    provider_pace = provider_values[0] if provider_values else ""
    front_count = int(counts.get("逃", 0)) + int(counts.get("先", 0))
    if provider_pace == "H":
        pace_detail = "先行馬多数 → 前半は流れやすく、差し馬に展開利の可能性" if front_count >= 3 else "前半は流れやすい想定"
        pace_line = f"展開：想定ペース H / {pace_detail}"
    elif provider_pace == "M":
        pace_line = "展開：想定ペース M / 極端な偏りなし"
    elif provider_pace == "S":
        pace_line = "展開：想定ペース S / 前残りの可能性"
    else:
        pace_line = f"展開：{pace.get('scenario', '判定保留')}（逃{counts.get('逃', 0)}・先{counts.get('先', 0)}）"
    changes: list[str] = []
    for row in table.to_dict("records"):
        number = horse_no(_pick(row, "馬番", "馬"))
        state = clean_text(_pick(row, "state_arrow"))
        weight_change = to_float(_pick(row, "weight_change_market"))
        jockey_change = clean_text(_pick(row, "jockey_change_market"))
        interval = clean_text(_pick(row, "race_interval_market"))
        if interval in {"休み明け", "長期休養"}:
            changes.append(f"{number}{interval}")
        if weight_change is not None and abs(weight_change) >= 2.0:
            changes.append(f"{number}斤量{weight_change:+.1f}kg")
        if jockey_change == "乗替":
            changes.append(f"{number}騎手替")
        if state in {"↑", "↓"}:
            changes.append(f"{number}状態{state}")
    change_line = "主な変化：" + ("、".join(changes[:5]) if changes else "大きな取得済み変化なし")
    return [ability_line, market_line, pace_line, change_line]


def market_prediction_signature(
    table: pd.DataFrame | Sequence[Mapping[str, Any]] | None,
    race_type: str,
    race_info: Mapping[str, Any] | None = None,
) -> str:
    frame = _as_frame(table)
    if not frame.empty and "ability_band_v2" not in frame.columns:
        frame = evaluate_market_table(frame, race_type, race_info)
    payload = {
        "version": MARKET_COMPARE_VERSION,
        "race": {
            key: _json_value((race_info or {}).get(key))
            for key in ("race_id", "date", "venue", "racecourse", "race_number", "race_name", "distance", "surface", "turn", "class_label")
        },
        "horses": safe_snapshot_records(frame),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def safe_snapshot_records(table: pd.DataFrame | None) -> list[dict[str, Any]]:
    if table is None or not isinstance(table, pd.DataFrame) or table.empty:
        return []
    columns = [column for column in SNAPSHOT_COLUMNS if column in table.columns and not _RESULT_FIELD_PATTERN.search(column)]
    records = []
    for row in table.loc[:, columns].to_dict("records"):
        records.append({str(key): _json_value(value) for key, value in row.items()})
    return sorted(records, key=lambda row: _horse_sort_key(_pick(row, "馬番", "horse_no", "馬")))


def freeze_market_prediction(result: Any, *, root: str | Path = MARKET_HISTORY_ROOT) -> Path:
    """Write an immutable pre-race snapshot and its SHA-256 sidecar."""

    if clean_text(getattr(result, "logic_version", "")) != "market":
        raise ValueError("能力×価格比較モードの予測だけを固定できます。")
    from .prediction_history import build_prediction_snapshot, result_stub_schema

    snapshot = build_prediction_snapshot(result, None)
    audit = snapshot.get("audit") if isinstance(snapshot.get("audit"), dict) else {}
    if audit:
        audit["prediction_generated_at"] = clean_text(getattr(result, "created_at", ""))
    race_info = snapshot.get("race_info") if isinstance(snapshot.get("race_info"), Mapping) else {}
    race_id = clean_text(race_info.get("race_id"))
    if not race_id:
        raise ValueError("race_idがないため予測を固定できません。")
    race_directory = Path(root) / "races" / race_id
    race_directory.mkdir(parents=True, exist_ok=True)
    prediction_path = race_directory / "prediction.json"
    data = json.dumps(snapshot, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")
    if prediction_path.exists() and prediction_path.read_bytes() != data:
        raise RuntimeError("固定済み予測と内容が異なるため上書きを拒否しました。")
    if not prediction_path.exists():
        prediction_path.write_bytes(data)
    digest = hashlib.sha256(prediction_path.read_bytes()).hexdigest()
    signature_path = race_directory / "prediction.sha256"
    if signature_path.exists() and clean_text(signature_path.read_text(encoding="utf-8")) != digest:
        raise RuntimeError("固定済み予測のSHA-256が一致しません。")
    if not signature_path.exists():
        signature_path.write_text(digest + "\n", encoding="utf-8")
    result_template = race_directory / "result_template.json"
    if not result_template.exists():
        result_template.write_text(
            json.dumps(result_stub_schema(race_info), ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return prediction_path


def _pace_context(styles: Sequence[str]) -> dict[str, Any]:
    counts = {key: 0 for key in ("逃", "先", "差", "追", "未取得")}
    for style in styles:
        key = style if style in counts else "未取得"
        counts[key] += 1
    known = sum(counts[key] for key in ("逃", "先", "差", "追"))
    front = counts["逃"] + counts["先"]
    if known < 2:
        scenario = "脚質データ不足"
    elif counts["逃"] == 1 and front <= 3:
        scenario = "単騎逃げ候補・スロー寄り"
    elif counts["逃"] >= 2 or front >= max(4, math.ceil(known * 0.45)):
        scenario = "先行馬多数・平均以上"
    elif front <= 2:
        scenario = "前少なめ・スロー寄り"
    else:
        scenario = "平均想定"
    return {"counts": counts, "known": known, "front": front, "scenario": scenario}


def _pace_for_horse(style: str, pace: Mapping[str, Any]) -> tuple[str, str]:
    scenario = clean_text(pace.get("scenario"))
    counts = pace.get("counts") if isinstance(pace.get("counts"), Mapping) else {}
    if not style or "不足" in scenario:
        return "±", "脚質データ不足"
    if style == "逃" and int(counts.get("逃", 0)) == 1:
        return "○", "単騎逃げ候補"
    if "先行馬多数" in scenario and style in {"差", "追"}:
        return "○", "前が競れば差し浮上"
    if "先行馬多数" in scenario and style in {"逃", "先"}:
        return "－", "同型多数"
    if "スロー" in scenario and style == "追":
        return "－", "追込＋スロー想定"
    if "スロー" in scenario and style in {"逃", "先"}:
        return "○", "好位を取りやすい"
    return "±", "大きな偏りなし"


def _human_pace_for_horse(
    style: str,
    provider_pace: str,
    pace: Mapping[str, Any],
) -> tuple[str, str]:
    """Describe this race's likely flow without changing ability."""

    if not style:
        return "±", "脚質データ不足"
    if provider_pace == "H":
        return {
            "逃": ("△", "ハイペースで逃げには厳しめ"),
            "先": ("△", "好位確保もハイペースは厳しめ"),
            "差": ("○", "前が流れれば浮上"),
            "追": ("○", "展開待ちだが流れは向く"),
        }.get(style, ("±", "展開判定保留"))
    if provider_pace == "M":
        return {
            "逃": ("±", "自分のペースなら粘り込み"),
            "先": ("±", "好位確保・展開平均"),
            "差": ("±", "流れ次第で浮上"),
            "追": ("△", "展開待ち"),
        }.get(style, ("±", "展開判定保留"))
    if provider_pace == "S":
        return {
            "逃": ("○", "前残りなら有利"),
            "先": ("○", "好位確保・前残りなら有利"),
            "差": ("△", "スローでは差し届かない懸念"),
            "追": ("△", "展開待ち"),
        }.get(style, ("±", "展開判定保留"))

    mark, reason = _pace_for_horse(style, pace)
    human = {
        "単騎逃げ候補": "単騎で運べれば有利",
        "前が競れば差し浮上": "前が流れれば浮上",
        "同型多数": "先行争いが厳しくなる懸念",
        "追込＋スロー想定": "スローでは追込届かない懸念",
        "好位を取りやすい": "好位確保・前残りなら有利",
        "大きな偏りなし": "展開平均",
    }.get(reason, reason)
    return mark, human


def _state(row: Mapping[str, Any]) -> dict[str, str]:
    values = [_index_number(_pick(row, *names)) for names in (("3走前", "race3"), ("2走前", "race2"), ("前走", "race1"))]
    valid = [value for value in values if value is not None]
    transition = "→".join(_format_number(value) if value is not None else "?" for value in values)
    if len(valid) < 2:
        return {"arrow": "？", "label": "判定保留", "transition": transition}
    if len(values) == 3 and all(value is not None for value in values):
        first, middle, last = (float(value) for value in values)
        if first < middle < last and last - first >= 6:
            return {"arrow": "↑", "label": "上昇", "transition": transition}
        if first > middle > last and first - last >= 6:
            return {"arrow": "↓", "label": "下降", "transition": transition}
        if middle < first and last > middle:
            return {"arrow": "↗", "label": "持ち直し", "transition": transition}
        if middle > first and last < middle:
            return {"arrow": "↘", "label": "弱含み", "transition": transition}
        if max(valid) - min(valid) <= 4:
            return {"arrow": "→", "label": "安定", "transition": transition}
    change = float(valid[-1]) - float(valid[0])
    if change >= 6:
        return {"arrow": "↑", "label": "上昇", "transition": transition}
    if change > 1:
        return {"arrow": "↗", "label": "持ち直し", "transition": transition}
    if change <= -6:
        return {"arrow": "↓", "label": "下降", "transition": transition}
    if change < -1:
        return {"arrow": "↘", "label": "弱含み", "transition": transition}
    return {"arrow": "→", "label": "安定", "transition": transition}


def _class_info(row: Mapping[str, Any], race_info: Mapping[str, Any]) -> dict[str, str]:
    current = _known_text(_pick(row, "_current_class_label", "今回クラス", "current_class"))
    if not current:
        current = _known_text(_pick(race_info, "class_label", "class", "race_class", "クラス"))
    previous = _known_text(_pick(row, "_previous_class_label", "前走クラス", "previous_class"))
    best = _known_text(_pick(row, "_best_past_class_label", "近3走最高クラス", "best_recent_class"))
    shift = _known_text(_pick(row, "クラス変動", "_class_shift", "class_shift"))
    past_labels = _as_text_list(row.get("_past_class_labels"))
    past_runs = _market_past_runs(row)
    previous_run = next((run for run in past_runs if _market_past_run_role(run) == "前走"), None)
    if previous_run is None and past_runs:
        previous_run = past_runs[-1]
    if not previous and isinstance(previous_run, Mapping):
        previous = _known_text(previous_run.get("class_label"))
    if not past_labels:
        past_labels = [
            _known_text(run.get("class_label"))
            for run in past_runs
            if isinstance(run, Mapping) and _known_text(run.get("class_label"))
        ]
    ranked_runs = [
        (to_float(run.get("class_rank")), _known_text(run.get("class_label")))
        for run in past_runs
        if isinstance(run, Mapping)
        and to_float(run.get("class_rank")) is not None
        and _known_text(run.get("class_label"))
    ]
    if not best:
        best = max(ranked_runs, key=lambda item: item[0])[1] if ranked_runs else (past_labels[0] if past_labels else "")
    current_rank = to_float(_pick(row, "_current_class_rank"))
    if current_rank is None:
        current_rank = to_float(_pick(race_info, "class_rank"))
    previous_rank = to_float(_pick(row, "_previous_class_rank"))
    if previous_rank is None and isinstance(previous_run, Mapping):
        previous_rank = to_float(previous_run.get("class_rank"))
    current_key = _class_label_key(current)
    previous_key = _class_label_key(previous)
    if current_rank is not None and previous_rank is not None and current_key and previous_key and current_key != previous_key:
        difference = current_rank - previous_rank
        derived_shift = "クラス昇級" if difference > 0 else "クラス降級" if difference < 0 else "同級"
        if shift in {"", "同級", "同級近辺"}:
            shift = derived_shift
    elif not shift and current_rank is not None and previous_rank is not None:
        shift = "同級"
    experienced = bool(current_key and any(_class_label_key(label) == current_key for label in past_labels))
    good_at_class = any(
        _class_label_key(run.get("class_label")) == current_key
        and (_class_finish_position(run) or 99) <= 3
        for run in past_runs
        if isinstance(run, Mapping)
    )
    evidence = []
    if current:
        evidence.append(f"今回{current}")
    if previous:
        evidence.append(f"前走{previous}")
    if best:
        evidence.append(f"近3走最高{best}")
    if shift:
        evidence.append(shift)
    if experienced:
        evidence.append(f"{current}経験あり")
    if good_at_class:
        evidence.append(f"{current}好走歴")
    if not evidence:
        existing = clean_text(_pick(row, "クラス根拠", "class_basis"))
        if existing:
            evidence.append(existing)
    return {
        "current": current or "未取得",
        "previous": previous or "未取得",
        "best": best or "未取得",
        "shift": shift or "判定保留",
        "basis": " / ".join(_unique(evidence)) if evidence else "取得不能",
        "experienced": "yes" if experienced else "no",
        "good_at_class": "yes" if good_at_class else "no",
    }


def _known_text(value: Any) -> str:
    text = clean_text(value)
    return "" if text in {"-", "—", "未取得", "未確認", "取得不能", "判定保留", "None", "nan"} else text


def _market_past_runs(row: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    result: list[Mapping[str, Any]] = []
    seen: set[tuple[str, str, str]] = set()
    for key in ("_past_runs", "_newspaper_past_runs", "_新聞過去走", "past_runs", "recent_runs"):
        runs = row.get(key)
        if not isinstance(runs, list):
            continue
        for raw in runs:
            if not isinstance(raw, Mapping):
                continue
            identity = (
                _market_past_run_role(raw),
                _known_text(raw.get("race_id") or raw.get("race_name") or raw.get("previous_race")),
                _known_text(raw.get("race_date") or raw.get("previous_date")),
            )
            if identity in seen:
                continue
            seen.add(identity)
            result.append(raw)
    return result


def _market_past_run_role(run: Mapping[str, Any]) -> str:
    label = _known_text(run.get("label") or run.get("key") or run.get("run_key"))
    return {
        "前走": "前走", "last": "前走", "race1": "前走", "1走前": "前走",
        "2走前": "2走前", "2back": "2走前", "race2": "2走前",
        "3走前": "3走前", "3back": "3走前", "race3": "3走前",
    }.get(label, "")


def _class_label_key(value: Any) -> str:
    return re.sub(r"\s+", "", _known_text(value)).upper()


def _class_finish_position(run: Mapping[str, Any]) -> float | None:
    for key in ("position", "finish", "finish_position", "previous_finish", "前走着順"):
        position = _index_number(run.get(key))
        if position is not None:
            return position
    return None


def _interval(row: Mapping[str, Any]) -> str:
    direct = _known_text(_pick(row, "レース間隔", "間隔", "race_interval"))
    if direct:
        return direct
    days = to_float(_pick(row, "_days_since_last", "レース間隔日数", "days_since_last"))
    if days is None:
        return "未取得"
    if days >= 120:
        return "長期休養"
    if days >= 56:
        return "休み明け"
    weeks = max(0, int(round(days / 7.0)) - 1)
    return f"中{weeks}週"


def _jockey(row: Mapping[str, Any]) -> tuple[str, str, str]:
    raw_current = clean_text(_pick(row, "_display_current_jockey", "_current_jockey", "騎手", "jockey"))
    current = _clean_jockey_display_name(raw_current) or "未取得"
    previous = _clean_jockey_display_name(
        clean_text(
            _pick(
                row,
                "_display_previous_jockey",
                "_previous_jockey",
                "前走騎手",
                "previous_jockey",
            )
        )
    )
    explicit = clean_text(
        _pick(
            row,
            "騎手継続/乗替",
            "jockey_change",
            "_display_jockey_changed",
            "_jockey_changed",
            "jockey_changed",
        )
    )
    inline_change = bool(re.search(r"(?:乗り?替|\(替\)|（替）|【乗り替わり】)", raw_current))
    if previous:
        same_name = _same_jockey_display_name(current, previous)
        change = "乗替" if _truthy(explicit) or "替" in explicit or same_name is False else "継続"
    elif explicit:
        change = "乗替" if _truthy(explicit) or "替" in explicit else "継続" if "継" in explicit else "未取得"
    elif inline_change:
        change = "乗替"
    else:
        change = "未取得"
    return current, previous, change


def _clean_jockey_display_name(value: Any) -> str:
    text = clean_text(value)
    text = re.sub(r"[\(（]\s*(?:替|継|乗替|乗り替わり|継続)\s*[\)）]", "", text)
    text = re.sub(r"【\s*(?:替|継|乗替|乗り替わり|継続|前走データなし|判定保留)\s*】", "", text)
    return text.strip()


def _jockey_compare_name(value: Any) -> str:
    text = unicodedata.normalize("NFKC", _clean_jockey_display_name(value))
    text = re.sub(r"^[▲△☆★◇◆▽▼]+", "", text)
    return re.sub(r"\s+", "", text)


def _same_jockey_display_name(current: Any, previous: Any) -> bool | None:
    """Treat provider-truncated three-character names as the same jockey."""

    current_text = _jockey_compare_name(current)
    previous_text = _jockey_compare_name(previous)
    if not current_text or not previous_text:
        return None
    if current_text == previous_text:
        return True
    short, long = (
        (current_text, previous_text)
        if len(current_text) <= len(previous_text)
        else (previous_text, current_text)
    )
    if long.startswith(short) and len(short) >= 3 and 1 <= len(long) - len(short) <= 2:
        return True
    return False


def _preferred_jockey_display_name(current: Any, previous: Any) -> str:
    current_text = _clean_jockey_display_name(current)
    previous_text = _clean_jockey_display_name(previous)
    if _same_jockey_display_name(current_text, previous_text) is True and len(previous_text) > len(current_text):
        return previous_text
    return current_text


def _course_development(
    row: Mapping[str, Any],
    style: str,
    fallback_mark: str,
    fallback_reason: str,
) -> dict[str, str]:
    """Create one grouped material from provider-derived course facts.

    Pace, position, course tendency, and provider picks can describe the same
    underlying scenario.  This function deliberately emits one mark/reason,
    never one point per derivative fact.
    """

    status = clean_text(_pick(row, "_course_context_status"))
    provider_pace = clean_text(_pick(row, "_netkeiba_pace")).upper()
    label = clean_text(_pick(row, "_favorable_position_label"))
    positions = [
        clean_text(_pick(row, "_estimated_position_start")),
        clean_text(_pick(row, "_estimated_position_corner3")),
        clean_text(_pick(row, "_estimated_position_corner4")),
    ]
    is_provider_context = status == "取得" and bool(
        provider_pace or label or any(positions) or clean_text(_pick(row, "_course_condition_html"))
    )
    if not is_provider_context:
        return {
            "mark": fallback_mark or "±",
            "reason": fallback_reason or "脚質データ不足",
            "source": "既存の全頭脚質構成",
        }

    favorable = row.get("_position_favorable_horse") is True
    if favorable:
        reason = "推定有利馬"
        if label and label != "未取得":
            reason += f"（{label}）"
        return {"mark": "○", "reason": reason, "source": "netkeiba競馬新聞HTML"}

    if "前有利" in label:
        if style in {"逃", "先"}:
            return {"mark": "○", "reason": "前残りなら有利", "source": "netkeiba競馬新聞HTML"}
        if style == "追":
            return {"mark": "△", "reason": "前有利で追込には不向き", "source": "netkeiba競馬新聞HTML"}
    if "後有利" in label:
        if style in {"差", "追"}:
            return {"mark": "○", "reason": "差し向き", "source": "netkeiba競馬新聞HTML"}
        if style in {"逃", "先"}:
            return {"mark": "△", "reason": "後方有利で先行には不向き", "source": "netkeiba競馬新聞HTML"}
    if "フラット" in label:
        return {"mark": "±", "reason": "4角傾向フラット", "source": "netkeiba競馬新聞HTML"}

    if provider_pace == "H":
        if style in {"差", "追"}:
            return {"mark": "○", "reason": "差し向き", "source": "netkeiba競馬新聞HTML"}
        if style in {"逃", "先"}:
            return {"mark": "△", "reason": "ハイペースで先行には厳しめ", "source": "netkeiba競馬新聞HTML"}
    if provider_pace == "S":
        if style in {"逃", "先"}:
            return {"mark": "○", "reason": "前残りなら有利", "source": "netkeiba競馬新聞HTML"}
        if style == "追":
            return {"mark": "△", "reason": "スローで追込には不向き", "source": "netkeiba競馬新聞HTML"}
    if any(positions):
        return {"mark": "±", "reason": "推定位置取得・評価保留", "source": "netkeiba競馬新聞HTML"}
    return {"mark": "±", "reason": "コース情報取得・評価保留", "source": "netkeiba競馬新聞HTML"}


def _jockey_course(row: Mapping[str, Any]) -> dict[str, str]:
    """Display objective course statistics only; never infer jockey quality."""

    win = to_float(_pick(row, "_jockey_course_win_rate", "jockey_course_win_rate", "騎手コース勝率"))
    quinella = to_float(
        _pick(row, "_jockey_course_quinella_rate", "jockey_course_quinella_rate", "騎手コース連対率")
    )
    place = to_float(_pick(row, "_jockey_course_place_rate", "jockey_course_place_rate", "騎手コース複勝率"))
    starts = to_float(_pick(row, "_jockey_course_starts", "jockey_course_starts", "騎手コース出走回数"))
    condition = clean_text(_pick(row, "_jockey_course_condition", "jockey_course_condition", "騎手コース条件"))
    source = clean_text(_pick(row, "_jockey_course_source", "jockey_course_source", "騎手コース成績取得元"))
    available_rates = [value for value in (win, quinella, place) if value is not None]
    if available_rates:
        rate_text = "-".join(_percent_text(value) for value in (win, quinella, place))
        display = f"{condition}｜{rate_text}" if condition else rate_text
        if starts is None:
            return {
                "display": display,
                "sample": "サンプル数未取得",
                "mark": "参考",
                "reason": "率は取得・出走回数なしのため参考値",
                "source": source or "供給された構造化データ",
            }
        sample = f"n={int(starts)}"
        if starts < JOCKEY_COURSE_MIN_STARTS:
            return {
                "display": display,
                "sample": f"サンプル不足 {sample}",
                "mark": "参考",
                "reason": f"{sample}<{JOCKEY_COURSE_MIN_STARTS}",
                "source": source or "供給された構造化データ",
            }
        if win is not None and place is not None and win >= 15.0 and place >= 40.0:
            return {
                "display": display,
                "sample": sample,
                "mark": "○",
                "reason": "客観コース成績が高水準",
                "source": source or "供給された構造化データ",
            }
        if win is not None and place is not None and win <= 5.0 and place <= 15.0:
            return {
                "display": display,
                "sample": sample,
                "mark": "△",
                "reason": "客観コース成績が低水準",
                "source": source or "供給された構造化データ",
            }
        return {
            "display": display,
            "sample": sample,
            "mark": "±",
            "reason": "客観値取得・強弱なし",
            "source": source or "供給された構造化データ",
        }

    rank = to_float(_pick(row, "_jockey_course_rank"))
    if rank is not None:
        return {
            "display": f"該当コース上位候補{int(rank)}位（率・件数なし）",
            "sample": "サンプル数未取得",
            "mark": "参考",
            "reason": "HTML内の参考順位のみ・強評価しない",
            "source": "競馬新聞HTML AnaBestTable",
        }
    return {
        "display": "騎手成績なし",
        "sample": "参考値なし",
        "mark": "—",
        "reason": "騎手コース勝率・連対率・複勝率・出走回数なし",
        "source": "参考値なし",
    }


def _jockey_display(
    jockey: str,
    previous_jockey: str,
    change: str,
    stats: Mapping[str, str],
) -> str:
    """Compact normal-UI label; sampling diagnostics stay in audit data."""

    name = clean_text(jockey)
    if change == "継続":
        name = _preferred_jockey_display_name(name, previous_jockey)
    display = clean_text(stats.get("display"))
    place = None
    if display and display not in {"取得不能", "騎手成績なし"}:
        rate_part = display.split("｜")[-1]
        parts = rate_part.split("-")
        if len(parts) >= 3 and parts[2] not in {"", "—"}:
            place = parts[2]
    rate = f"複{place}" if place else ""
    if not name or name == "未取得":
        return ""
    if change == "乗替" and previous_jockey:
        current = f"{name}（{rate}）" if rate else name
        return f"{previous_jockey} → {current}"
    if change == "乗替":
        details = "・".join(item for item in ("替", "前走騎手不明", rate) if item)
        return f"{name}（{details}）"
    if change == "継続":
        details = "・".join(item for item in ("継", rate) if item)
        return f"{name}（{details}）"
    return f"{name}（{rate}）" if rate else name


def _attach_current_evaluation(result: pd.DataFrame) -> pd.DataFrame:
    """Rank the current setup independently from the immutable ability rank.

    No market price is added to the evaluation balance.  Actual odds is used
    only as the final ordering key when ability and every current-condition
    comparison key are exactly equal, so market price stays a separate axis.
    """

    band_base = {"AA": 10.0, "A": 8.0, "B": 5.0, "C": 2.0, "Z": 0.0}
    evaluations: list[dict[str, Any]] = []
    for position, row in enumerate(result.to_dict("records")):
        balance, positive_count, negative_count = _current_factor_balance(row)
        band = clean_text(_pick(row, "ability_band_v2")) or "Z"
        ability = to_float(_pick(row, "market_ability_score"))
        odds = to_float(_pick(row, "actual_odds"))
        evaluations.append(
            {
                "position": position,
                "band": band,
                "balance": balance,
                "positive_count": positive_count,
                "negative_count": negative_count,
                "comparison": band_base.get(band, 0.0) + balance,
                "ability": ability if ability is not None else -math.inf,
                "odds": odds if odds is not None else -math.inf,
                "horse_no": horse_no(_pick(row, "馬番", "horse_no", "馬")),
                "plus": list(row.get("positive_materials") or []),
                "minus": list(row.get("negative_materials") or []),
            }
        )
    ordered = sorted(
        evaluations,
        key=lambda item: (
            -item["comparison"],
            -item["positive_count"],
            item["negative_count"],
            -item["ability"],
            -item["odds"],
            _horse_sort_key(item["horse_no"]),
        ),
    )
    # C/Z can remain comparison candidates, but do not become ◎ while a horse
    # with an AA/A/B ability foundation exists.
    eligible_index = next(
        (index for index, item in enumerate(ordered) if item["band"] in {"AA", "A", "B"}),
        None,
    )
    if eligible_index not in {None, 0}:
        ordered.insert(0, ordered.pop(int(eligible_index)))

    marks = ("◎", "○", "▲", "△", "☆")
    by_position: dict[int, dict[str, Any]] = {}
    for rank, item in enumerate(ordered, start=1):
        plus_reason = item["plus"][0] if item["plus"] else ""
        minus_reason = item["minus"][0] if item["minus"] else ""
        reason_bits = [f"能力{item['band']}を土台"]
        if plus_reason:
            reason_bits.append(plus_reason)
        if minus_reason:
            reason_bits.append(f"注意：{minus_reason}")
        if not plus_reason and not minus_reason:
            reason_bits.append("今回材料は中立")
        by_position[item["position"]] = {
            "balance": item["balance"],
            "positive_count": item["positive_count"],
            "negative_count": item["negative_count"],
            "rank": rank,
            "mark": marks[rank - 1] if rank <= len(marks) else "",
            "reason": " / ".join(reason_bits),
        }

    result = result.copy()
    result["current_evaluation_balance"] = [by_position[index]["balance"] for index in range(len(result))]
    result["current_evaluation_positive_count"] = [by_position[index]["positive_count"] for index in range(len(result))]
    result["current_evaluation_negative_count"] = [by_position[index]["negative_count"] for index in range(len(result))]
    result["current_evaluation_rank"] = pd.Series(
        [by_position[index]["rank"] for index in range(len(result))],
        index=result.index,
        dtype="Int64",
    )
    result["ai_current_mark"] = [by_position[index]["mark"] for index in range(len(result))]
    result["ai_current_reason"] = [by_position[index]["reason"] for index in range(len(result))]
    return result


def _current_factor_balance(row: Mapping[str, Any]) -> tuple[float, int, int]:
    factors: list[float] = []
    if clean_text(_pick(row, "condition_mark_market")):
        factors.append(0.75)
    state = clean_text(_pick(row, "state_arrow"))
    if state in {"↑", "↗"}:
        factors.append(1.0)
    elif state in {"↓", "↘"}:
        factors.append(-1.0)
    shift = clean_text(_pick(row, "class_shift_market"))
    if "降級" in shift or "好走歴" in clean_text(_pick(row, "class_basis_market")):
        factors.append(0.75)
    elif "昇級" in shift and "経験あり" not in clean_text(_pick(row, "class_basis_market")):
        factors.append(-0.75)
    interval = clean_text(_pick(row, "race_interval_market"))
    if "長期休養" in interval:
        factors.append(-1.0)
    elif "休み明け" in interval:
        factors.append(-0.75)
    weight_change = to_float(_pick(row, "weight_change_market"))
    if weight_change is not None and weight_change <= -1.0:
        factors.append(0.5)
    elif weight_change is not None and weight_change >= 1.0:
        factors.append(-0.5)
    development = clean_text(_pick(row, "course_development_mark"))
    if development == "○":
        factors.append(1.0)
    elif development in {"△", "－"}:
        factors.append(-1.0)
    jockey = clean_text(_pick(row, "jockey_course_mark_market"))
    if jockey == "○":
        factors.append(0.25)
    elif jockey == "△":
        factors.append(-0.25)
    training = clean_text(_pick(row, "training_market")).upper()
    if training.startswith("A") or any(word in training for word in ("良化", "上向", "好気配", "復調")):
        factors.append(0.5)
    elif any(word in training for word in ("平凡", "一息", "重い")):
        factors.append(-0.5)
    raw = sum(factors)
    return round(max(-3.0, min(3.0, raw)), 2), sum(value > 0 for value in factors), sum(value < 0 for value in factors)


def _percent_text(value: float | None) -> str:
    if value is None:
        return "—"
    return f"{value:g}%"


def _weight(row: Mapping[str, Any]) -> tuple[str, float | None]:
    current_value = to_float(_pick(row, "_current_load_weight", "斤量", "weight"))
    display = f"{current_value:.1f}kg" if current_value is not None else clean_text(_pick(row, "斤量", "weight")) or "未取得"
    change = to_float(_pick(row, "_load_weight_change", "斤量増減", "weight_change"))
    if change is None:
        detail = clean_text(_pick(row, "斤量詳細", "weight_detail"))
        match = re.search(r"([+-]\d+(?:\.\d+)?|±\s*0)", detail)
        if match:
            token = match.group(1).replace("±", "").replace(" ", "")
            change = to_float(token)
    return display, change


def _body_weight(row: Mapping[str, Any]) -> tuple[str, float | None]:
    """Read only uploaded/parser-provided body weight and its change."""

    raw = _known_text(_pick(row, "馬体重", "body_weight", "horse_weight"))
    value = to_float(_pick(row, "_body_weight", "body_weight_value", "horse_weight_value"))
    change = to_float(
        _pick(
            row,
            "_body_weight_change",
            "body_weight_change",
            "horse_weight_diff",
            "馬体重増減",
        )
    )
    match = re.search(r"(\d{3})\s*(?:kg)?\s*[（(]\s*([+\-−]?\d+)\s*[）)]", raw)
    if match:
        if value is None:
            value = to_float(match.group(1))
        if change is None:
            change = to_float(match.group(2).replace("−", "-"))
    if value is None:
        value = to_float(raw)
    if value is None:
        return "未取得", change
    if change is None:
        return f"{value:.0f}kg", None
    change_text = "±0" if abs(change) < 0.0001 else f"{change:+.0f}"
    return f"{value:.0f}kg（{change_text}）", change


def _training(row: Mapping[str, Any], race_type: str) -> str:
    if race_type != "jra":
        return "対象外"
    grade = clean_text(_pick(row, "調教評価", "追切評価", "_調教評価記号"))
    comment = clean_text(_pick(row, "追切内容", "調教コメント", "追切材料", "調教/評価/検討材料", "training_comment"))
    if grade and comment and comment not in grade:
        return f"{grade}｜{comment}"
    return grade or comment or "未取得"


def _materials(
    *,
    row: Mapping[str, Any],
    race_type: str,
    condition: Mapping[str, Any],
    state: Mapping[str, str],
    class_info: Mapping[str, str],
    interval: str,
    weight_change: float | None,
    course_development: Mapping[str, str],
    jockey_course: Mapping[str, str],
    training: str,
) -> tuple[list[str], list[str]]:
    plus: dict[str, str] = {}
    minus: dict[str, str] = {}

    def add(target: dict[str, str], family: str, text: str) -> None:
        cleaned = clean_text(text)
        if cleaned and family not in target:
            target[family] = cleaned

    mark = clean_text(condition.get("condition_fit_mark"))
    reason = clean_text(condition.get("condition_fit_reason"))
    if mark:
        condition_text = re.sub(r"の過去走あり$", "実績", reason)
        add(plus, "condition", condition_text)
    if state.get("arrow") in {"↑", "↗"}:
        add(plus, "state", f"近走{state.get('label')}")
    elif state.get("arrow") in {"↓", "↘"}:
        add(minus, "state", f"近走{state.get('label')}")
    shift = class_info.get("shift", "")
    current = class_info.get("current", "")
    if "降級" in shift:
        add(plus, "class", "クラス降級")
    elif "昇級" in shift:
        if class_info.get("experienced") == "yes":
            add(plus, "class", f"{current}経験あり")
        else:
            add(minus, "class", f"初{current}" if current and current != "未取得" else "初昇級")
    elif class_info.get("good_at_class") == "yes":
        add(plus, "class", f"{current}好走歴")
    if "長期休養" in interval:
        add(minus, "interval", "長期休養")
    elif "休み明け" in interval:
        add(minus, "interval", "休み明け")
    if weight_change is not None:
        if weight_change <= -1.0:
            add(plus, "weight", f"斤量{weight_change:+.1f}kg")
        elif weight_change >= 1.0:
            add(minus, "weight", f"斤量{weight_change:+.1f}kg")
    development_mark = clean_text(course_development.get("mark"))
    development_reason = clean_text(course_development.get("reason"))
    if development_mark == "○" and development_reason:
        add(plus, "course", f"展開/コース：{development_reason}")
    elif development_mark in {"△", "－"} and development_reason:
        add(minus, "course", f"展開/コース：{development_reason}")
    jockey_mark = clean_text(jockey_course.get("mark"))
    jockey_reason = clean_text(jockey_course.get("reason"))
    if jockey_mark == "○" and jockey_reason:
        add(plus, "jockey", "騎手：コース成績良好")
    elif jockey_mark == "△" and jockey_reason:
        add(minus, "jockey", "騎手：コース成績低調")
    if race_type == "jra" and training not in {"", "未取得", "対象外"}:
        grade_match = re.match(r"\s*([A-D])", training.upper())
        if grade_match and grade_match.group(1) == "A":
            add(plus, "training", f"調教{grade_match.group(1)}")
        if any(word in training for word in ("良化", "上向", "好気配", "復調")):
            add(plus, "training", next(word for word in ("良化", "上向", "好気配", "復調") if word in training))
        if any(word in training for word in ("平凡", "一息", "重い")):
            add(minus, "training", next(word for word in ("平凡", "一息", "重い") if word in training))
    # Odds, popularity, prediction marks, and jockey fame are intentionally
    # absent. Course/development and jockey facts are display materials only.
    plus_order = ("course", "state", "weight", "condition", "class", "jockey", "training")
    minus_order = ("course", "interval", "state", "class", "weight", "jockey", "training")
    return (
        [plus[key] for key in plus_order if key in plus][:3],
        [minus[key] for key in minus_order if key in minus][:3],
    )


def _normalize_style(value: Any) -> str:
    text = clean_text(value)
    if "逃" in text:
        return "逃"
    if "先" in text:
        return "先"
    if "差" in text:
        return "差"
    if "追" in text:
        return "追"
    return ""


def _first_numeric_series(frame: pd.DataFrame, names: Sequence[str]) -> pd.Series:
    result = pd.Series(pd.NA, index=frame.index, dtype="Float64")
    for name in names:
        if name not in frame.columns:
            continue
        candidate = frame[name].map(_index_number).astype("Float64")
        result = result.where(result.notna(), candidate)
    return result


def _index_number(value: Any) -> float | None:
    number = to_float(value)
    if number is not None:
        return number
    match = re.search(r"[-+]?\d+(?:\.\d+)?", clean_text(value).replace(",", ""))
    return float(match.group(0)) if match else None


def _as_frame(table: pd.DataFrame | Sequence[Mapping[str, Any]] | None) -> pd.DataFrame:
    if table is None:
        return pd.DataFrame()
    if isinstance(table, pd.DataFrame):
        return table.copy()
    return pd.DataFrame([dict(row) for row in table])


def _pick(source: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name not in source:
            continue
        value = source.get(name)
        if not _missing(value):
            return value
    return ""


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
    return clean_text(value).lower() in {"", "-", "—", "nan", "none", "null", "未取得", "データなし"}


def _truthy(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return clean_text(value).lower() in {"1", "true", "yes", "y", "○", "あり", "該当", "乗替", "乗り替わり"}


def _as_text_list(value: Any) -> list[str]:
    if isinstance(value, (list, tuple)):
        return [clean_text(item) for item in value if clean_text(item)]
    text = clean_text(value)
    return [part for part in re.split(r"[／/,、\s]+", text) if part] if text else []


def _unique(values: Sequence[str]) -> list[str]:
    result: list[str] = []
    for value in values:
        text = clean_text(value)
        if text and text not in result:
            result.append(text)
    return result


def _format_number(value: float) -> str:
    return f"{value:.1f}".rstrip("0").rstrip(".")


def _horse_sort_key(value: Any) -> tuple[int, str]:
    number = to_float(value)
    return (int(number) if number is not None else 999, clean_text(value))


def _json_value(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_value(item) for key, item in value.items() if not _RESULT_FIELD_PATTERN.search(str(key))}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if _missing(value):
        return ""
    if isinstance(value, bool):
        return value
    if hasattr(value, "item"):
        try:
            return _json_value(value.item())
        except Exception:
            pass
    if isinstance(value, (int, float)):
        number = to_float(value)
        if number is not None:
            return int(number) if number.is_integer() else number
    return value

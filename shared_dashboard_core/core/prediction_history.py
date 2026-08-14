# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

import pandas as pd

from .condition_fit import resolved_condition_fit
from .final_betting_context import build_final_betting_context
from .horse_trust import build_horse_trust_materials, build_horse_trust_summary
from .models import PredictionResult
from .purchase_conditions import clean_text, horse_no, to_float
from .recent_races import build_recent_races
from .value_support import attach_value_signals
from .version import APP_VERSION


HISTORY_ROOT = Path("prediction_history")

# Dashboard 0.x exported its BUY/HOLD/SKIP summary archive through this
# module.  Keep those two public entry points available while the Mobile
# PredictionResult snapshot implementation remains the canonical code below.
from .dashboard_prediction_history import (  # noqa: E402
    build_prediction_history_zip,
    history_zip_file_name,
)


def build_prediction_snapshot(result: PredictionResult, investment_decision: Any = None) -> dict[str, Any]:
    race_type = "nar" if clean_text(result.race_mode).lower() == "nar" else "jra"
    logic_version = clean_text(getattr(result, "logic_version", "v3")) or "v3"
    if investment_decision is None and logic_version in {"v4", "v4.1", "practical"}:
        from .investment_decision import build_investment_decision

        table = result.overall_table
        if table is None or not isinstance(table, pd.DataFrame) or table.empty:
            table = result.horse_evaluation
        investment_decision = build_investment_decision(
            table,
            race_type,
            race_info=result.race_info,
            prediction_logic_version=logic_version,
        )
    race_info = _race_info(result)
    horses = _horse_snapshots(result, race_type, race_info)
    investment = _investment_snapshot(investment_decision)
    payload = {
            "schema_version": 1,
            "logic_version": logic_version,
            "race_info": race_info,
            "horses": horses,
            "prediction": {
                "attention_horses": list(result.attention_horses or []),
                "ai_race_review": result.ai_race_review,
                "betting_structure": result.betting_structure,
            },
            "investment_decision": investment,
            "audit": {
                "strategy_json_path": getattr(investment_decision, "source_path", "") if investment_decision else "",
                "strategy_json_updated_at": getattr(investment_decision, "updated_at", "") if investment_decision else "",
                "strategy_json_note": getattr(investment_decision, "source_note", "") if investment_decision else "",
                "strategy_source_race_count": getattr(investment_decision, "source_race_count", 0) if investment_decision else 0,
                "prediction_generated_at": datetime.now().isoformat(timespec="seconds"),
                "prediction_created_at": result.created_at,
                "app_version": result.version or APP_VERSION,
                "commit_hash": "",
                "race_mode": race_type,
                "logic_version": logic_version,
                "source_files": result.source_files,
                "status": result.status,
                "message": result.message,
            },
            "result_file": result_stub_schema(race_info),
        }
    if logic_version in {"v4", "v4.1"}:
        payload["ver4_summary"] = getattr(result, "ver4_summary", {}) or {}
    if logic_version == "practical":
        payload["practical_summary"] = (
            ((getattr(result, "debug_info", {}) or {}).get("practical") or {}).get("summary", {})
        )
    if logic_version == "market":
        market = ((getattr(result, "debug_info", {}) or {}).get("market_compare") or {})
        payload["market_compare"] = {
            "version": market.get("version", ""),
            "ability_band_rules": market.get("ability_band_rules", {}),
            "ability_source": market.get("ability_source", ""),
            "calibration": market.get("calibration", {}),
            "pace": market.get("pace", {}),
            "race_summary": market.get("race_summary", []),
            "prediction_signature": market.get("prediction_signature", ""),
            "horses": market.get("horses", []),
            "user_selection": market.get("user_selection", {}),
        }
    return _json_ready(payload)


def result_stub_schema(race_info: Mapping[str, Any] | None = None) -> dict[str, Any]:
    race_info = race_info or {}
    return {
        "schema_version": 1,
        "race_id": race_info.get("race_id", ""),
        "race_type": race_info.get("race_type", ""),
        "results": [],
        "payoffs": {
            "win": [],
            "place": [],
            "wide": [],
            "quinella": [],
            "exacta": [],
            "trio": [],
            "trifecta": [],
        },
        "note": "結果はprediction.jsonへ上書きせず、result.jsonとして別保存します。",
    }


def prediction_csv_bytes(snapshot: Mapping[str, Any]) -> bytes:
    rows = prediction_csv_rows(snapshot)
    if not rows:
        return b""
    headers: list[str] = []
    for row in rows:
        for key in row:
            if key not in headers:
                headers.append(key)
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=headers)
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8-sig")


def prediction_csv_rows(snapshot: Mapping[str, Any]) -> list[dict[str, Any]]:
    race_info = snapshot.get("race_info") if isinstance(snapshot.get("race_info"), Mapping) else {}
    investment = snapshot.get("investment_decision") if isinstance(snapshot.get("investment_decision"), Mapping) else {}
    rows: list[dict[str, Any]] = []
    for horse in snapshot.get("horses", []) or []:
        if not isinstance(horse, Mapping):
            continue
        rows.append(
            {
                "race_id": race_info.get("race_id", ""),
                "race_type": race_info.get("race_type", ""),
                "logic_version": snapshot.get("logic_version", "v3"),
                "date": race_info.get("date", ""),
                "venue": race_info.get("venue", ""),
                "race_number": race_info.get("race_number", ""),
                "race_name": race_info.get("race_name", ""),
                "distance": race_info.get("distance", ""),
                "surface": race_info.get("surface", ""),
                "class": race_info.get("class", ""),
                "decision": investment.get("decision", ""),
                "selected_strategy": investment.get("selected_strategy", ""),
                "strategy_score": investment.get("strategy_score", ""),
                "expected_roi": investment.get("expected_roi", ""),
                "buy_reason": investment.get("buy_reason", ""),
                "watch_reason": investment.get("watch_reason", ""),
                "recommended_ticket": investment.get("ticket_type", ""),
                "investment": investment.get("investment", 0),
                **_flatten_for_csv(horse),
            }
        )
    return rows


def summary_text(snapshot: Mapping[str, Any]) -> str:
    race_info = snapshot.get("race_info") if isinstance(snapshot.get("race_info"), Mapping) else {}
    investment = snapshot.get("investment_decision") if isinstance(snapshot.get("investment_decision"), Mapping) else {}
    horses = [horse for horse in snapshot.get("horses", []) or [] if isinstance(horse, Mapping)]
    logic_version = clean_text(snapshot.get("logic_version")).lower()
    if logic_version in {"v4", "v4.1"}:
        return _ver4_summary_text(race_info, investment, horses)
    if logic_version == "practical":
        return _practical_summary_text(race_info, investment, horses)
    if logic_version == "market":
        return _market_summary_text(snapshot, race_info)
    lines = [
        _join_nonempty([race_info.get("venue"), race_info.get("race_number"), race_info.get("race_name")], sep=" "),
        "",
        f"総合判定：{investment.get('decision', 'SKIP')}",
        "",
        "今回買うべき馬券",
    ]
    if investment.get("selected_strategy"):
        lines.extend(
            [
                clean_text(investment.get("selected_strategy")),
                _join_nonempty(investment.get("tickets", []), sep=" / ") or "買い目なし",
                "",
                "【今回の馬の根拠】",
                _join_nonempty(investment.get("final_context_summary", []) or investment.get("horse_trust_summary", []), sep="\n") or "記録なし",
                "",
                "【馬券側の根拠】",
                _ticket_rationale_text(investment),
                "",
                "【今回評価との一致】",
                _join_nonempty(investment.get("ticket_alignment_summary", []), sep="\n") or "記録なし",
            ]
        )
    else:
        lines.extend(["今回は正式購入条件に一致する馬券がありません。", "", "【見送り理由】"])
        lines.extend(investment.get("reason_lines", []) or ["記録なし"])

    lines.extend(["", "【上位馬】"])
    for horse in horses[:5]:
        prediction = horse.get("prediction") if isinstance(horse.get("prediction"), Mapping) else {}
        lines.append(
            _join_nonempty(
                [
                    prediction.get("mark"),
                    horse.get("horse_no"),
                    horse.get("horse_name"),
                    f"AI点{prediction.get('ai_score')}" if prediction.get("ai_score") not in (None, "") else "",
                    f"能力{prediction.get('ability_value')}" if prediction.get("ability_value") not in (None, "") else "",
                ],
                sep=" ",
            )
        )
    return "\n".join(line for line in lines if line is not None)


def _ver4_summary_text(
    race_info: Mapping[str, Any],
    investment: Mapping[str, Any],
    horses: list[Mapping[str, Any]],
) -> str:
    lines = [
        _join_nonempty([race_info.get("venue"), race_info.get("race_number"), race_info.get("race_name")], sep=" "),
        "",
        f"Ver4判断：{investment.get('decision_v4') or 'SKIP'}",
        f"互換判定：{investment.get('legacy_decision') or investment.get('decision') or 'SKIP'}",
        f"軸Score：{investment.get('axis_score', '')}",
        f"軸信頼度：{investment.get('axis_confidence_v4', '') or 'なし'}",
        f"買い候補Score：{investment.get('ticket_candidate_score', '')}",
    ]
    if investment.get("ticket_veto_reason"):
        lines.append(f"VETO：{investment.get('ticket_veto_reason')}")
    lines.extend(["", "今回の買い目"])
    lines.append(_join_nonempty(investment.get("tickets", []), sep=" / ") or "買い目なし")
    lines.extend(["", "【Horse Score上位】"])
    ranked = sorted(
        horses,
        key=lambda item: to_float((_mapping(item.get("ver4"))).get("race_rank_v4")) or 999,
    )
    for horse in ranked[:5]:
        ver4 = _mapping(horse.get("ver4"))
        lines.append(
            _join_nonempty(
                [
                    ver4.get("mark_v4"),
                    horse.get("horse_no"),
                    horse.get("horse_name"),
                    f"Score {ver4.get('horse_score_v4')}" if ver4.get("horse_score_v4") not in (None, "") else "",
                    f"Rank {ver4.get('race_rank_v4')}" if ver4.get("race_rank_v4") not in (None, "") else "",
                    ver4.get("group_v4"),
                ],
                sep=" ",
            )
        )
    return "\n".join(line for line in lines if line is not None)


def _practical_summary_text(
    race_info: Mapping[str, Any],
    investment: Mapping[str, Any],
    horses: list[Mapping[str, Any]],
) -> str:
    decision = clean_text(investment.get("decision")) or "WATCH"
    lines = [
        _join_nonempty([race_info.get("venue"), race_info.get("race_number"), race_info.get("race_name")], sep=" "),
        "",
        f"実戦判定：{decision}",
        f"理由：{investment.get('buy_reason') or investment.get('watch_reason') or '記録なし'}",
        f"主推奨：{investment.get('ticket_type') or '購入なし'}",
        f"投資額：{investment.get('investment') or 0}円",
        "",
        "【Ver3印】",
    ]
    for horse in horses:
        prediction = _mapping(horse.get("prediction"))
        mark = clean_text(prediction.get("mark"))
        if mark:
            lines.append(
                _join_nonempty(
                    [mark, horse.get("horse_no"), horse.get("horse_name"), horse.get("condition_fit_mark")],
                    sep=" ",
                )
            )
    return "\n".join(line for line in lines if line is not None)


def _market_summary_text(snapshot: Mapping[str, Any], race_info: Mapping[str, Any]) -> str:
    market = snapshot.get("market_compare") if isinstance(snapshot.get("market_compare"), Mapping) else {}
    lines = [
        _join_nonempty([race_info.get("venue"), race_info.get("race_number"), race_info.get("race_name")], sep=" "),
        "",
        "能力×市場価格×今日の条件×展開 比較",
        f"予測signature：{market.get('prediction_signature') or '未生成'}",
        "",
    ]
    lines.extend(str(line) for line in market.get("race_summary", []) or [])
    calibration = market.get("calibration") if isinstance(market.get("calibration"), Mapping) else {}
    lines.extend(["", calibration.get("display") or "AI適正オッズ：未校正"])
    selection = market.get("user_selection") if isinstance(market.get("user_selection"), Mapping) else {}
    if selection:
        lines.extend(
            [
                "",
                "【ユーザー選択】",
                "選択馬：" + _join_nonempty(selection.get("horses", []), sep=" / "),
                "理由：" + clean_text(selection.get("reason")),
                "券種：" + clean_text(selection.get("ticket")),
            ]
        )
    return "\n".join(line for line in lines if line is not None)


def prediction_zip_bytes(result: PredictionResult, investment_decision: Any = None) -> bytes:
    snapshot = build_prediction_snapshot(result, investment_decision)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        archive.writestr("prediction.json", json.dumps(snapshot, ensure_ascii=False, indent=2))
        archive.writestr("prediction.csv", prediction_csv_bytes(snapshot))
        archive.writestr("summary.txt", summary_text(snapshot).encode("utf-8"))
        if clean_text(snapshot.get("logic_version")) in {"practical", "market"}:
            archive.writestr(
                "result_template.json",
                json.dumps(snapshot.get("result_file", {}), ensure_ascii=False, indent=2),
            )
    return buffer.getvalue()


def prediction_zip_filename(result: PredictionResult) -> str:
    info = _race_info(result)
    date = clean_text(info.get("date")).replace("-", "") or datetime.now().strftime("%Y%m%d")
    venue = _safe_filename(info.get("venue") or "race")
    race_no = _safe_filename(info.get("race_number") or "")
    return f"keiba_prediction_{date}_{venue}{race_no}.zip"


def save_prediction_history(
    result: PredictionResult,
    investment_decision: Any = None,
    *,
    root: str | Path = HISTORY_ROOT,
    overwrite: bool = False,
) -> Path:
    snapshot = build_prediction_snapshot(result, investment_decision)
    info = snapshot["race_info"]
    race_type = clean_text(info.get("race_type")) or "jra"
    race_date = clean_text(info.get("date")).replace("-", "") or datetime.now().strftime("%Y%m%d")
    race_id = clean_text(info.get("race_id")) or _safe_filename(_join_nonempty([info.get("venue"), info.get("race_number")], sep="_"))
    target_dir = Path(root) / race_type / race_date / race_id
    target_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = target_dir / "prediction.json"
    if overwrite or not prediction_path.exists():
        prediction_path.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    result_path = target_dir / "result.json"
    if not result_path.exists():
        result_path.write_text(json.dumps(result_stub_schema(info), ensure_ascii=False, indent=2), encoding="utf-8")
    return prediction_path


def _race_info(result: PredictionResult) -> dict[str, Any]:
    info = result.race_info or {}
    race_type = "nar" if clean_text(result.race_mode).lower() == "nar" else "jra"
    race_id = _pick(info, "race_id", "id", "raceId") or _race_id_from_source_files(result.source_files)
    race_name = result.race_name or _pick(info, "race_name", "レース名", "name")
    return {
        "race_id": race_id,
        "race_type": race_type,
        "date": _race_date(info, race_id),
        "venue": _pick(info, "venue", "racecourse", "競馬場", "place", "場所"),
        "race_number": _race_number(_pick(info, "race_number", "race_no", "R", "レース番号") or race_name),
        "race_name": race_name,
        "distance": _pick(info, "distance", "距離"),
        "surface": _pick(info, "surface", "芝ダート", "馬場種別"),
        "turn": _pick(info, "turn", "回り"),
        "class": _pick(info, "class", "race_class", "クラス"),
        "head_count": _pick(info, "頭数", "horse_count", "runners", "出走頭数"),
        "post_time": _pick(info, "post_time", "start_time", "発走時刻", "発走"),
    }


def _horse_snapshots(
    result: PredictionResult,
    race_type: str,
    race_info: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    merged = _merged_horse_rows(result)
    value_by_no = {
        horse_no(_pick(item, "馬番", "馬", "horse_no", "horse_number")): item
        for item in attach_value_signals(merged, race_type)
    }
    context_by_no = {
        clean_text(item.get("horse_number")): item
        for item in build_final_betting_context(pd.DataFrame(merged), race_type, race_info=race_info)
    }
    rows: list[dict[str, Any]] = []
    for row in merged:
        trust = build_horse_trust_materials(row, race_type)
        number = horse_no(_pick(row, "馬番", "horse_no", "horse_number", "馬"))
        value_support = value_by_no.get(number, {})
        final_context = context_by_no.get(number, {})
        momentum = final_context.get("momentum") if isinstance(final_context.get("momentum"), Mapping) else {}
        race_shape = final_context.get("race_shape") if isinstance(final_context.get("race_shape"), Mapping) else {}
        recent_races = build_recent_races(row)
        condition_fit = resolved_condition_fit(row, race_info)
        rows.append(
            {
                "horse_no": number,
                "horse_name": _pick(row, "馬名", "horse_name", "name"),
                "sex_age": _pick(row, "馬年齢", "性齢", "馬齢"),
                "age": _age_number(_pick(row, "馬年齢", "性齢", "馬齢")),
                "weight": _pick(row, "斤量", "weight"),
                "weight_detail": _pick(row, "斤量詳細", "weight_detail"),
                "jockey": _pick(row, "騎手", "jockey"),
                "jockey_detail": _pick(row, "騎手詳細", "jockey_detail"),
                "jockey_change": _pick(row, "騎手継続/乗替", "jockey_change", "_jockey_changed"),
                "popularity": _pick(row, "人気", "popularity"),
                "odds": _pick(row, "単勝オッズ", "オッズ", "odds"),
                "prediction": {
                    "ai_score": _pick(row, "AI点", "normalized_ai_score", "ai_score"),
                    "ai_rank": _pick(row, "AI順位", "ai_rank"),
                    "raw_score": _pick(row, "raw_score", "_raw_score"),
                    "ability_value": _pick(row, "能力評価値", "ability_display_score"),
                    "mark": _pick(row, "表示印", "display_mark", "最終印", "印", "mark"),
                    "display_group": _pick(row, "グループ", "display_group"),
                    "original_mark": _pick(row, "original_mark", "old_final_mark", "元印"),
                    "horse_score_v4": _pick(row, "horse_score_v4"),
                    "race_rank_v4": _pick(row, "race_rank_v4"),
                    "group_v4": _pick(row, "group_v4"),
                    "mark_v4": _pick(row, "mark_v4"),
                },
                "ver4": {
                    "horse_score_v4": _pick(row, "horse_score_v4"),
                    "race_rank_v4": _pick(row, "race_rank_v4"),
                    "base_ability_score": _pick(row, "base_ability_score"),
                    "condition_score": _pick(row, "condition_score"),
                    "jockey_score": _pick(row, "jockey_score"),
                    "age_weight_score": _pick(row, "age_weight_score"),
                    "training_score": _pick(row, "training_score"),
                    "momentum_score_v4": _pick(row, "momentum_score_v4"),
                    "race_shape_score": _pick(row, "race_shape_score"),
                    "condition_fit_mark": _pick(row, "condition_fit_mark"),
                    "condition_fit_level": _pick(row, "condition_fit_level"),
                    "condition_matched_quality": _pick(row, "condition_matched_quality"),
                    "group_v4": _pick(row, "group_v4"),
                    "mark_v4": _pick(row, "mark_v4"),
                    "warning_reason": _pick(row, "warning_reason"),
                    "positive_reasons": _pick(row, "positive_reasons_v4"),
                    "negative_reasons": _pick(row, "negative_reasons_v4"),
                    "watch_reason": _pick(row, "watch_reason_v4"),
                    "axis_score": _pick(row, "axis_score"),
                    "axis_confidence": _pick(row, "axis_confidence_v4"),
                    "ticket_candidate_score": _pick(row, "ticket_candidate_score"),
                    "opponent_eligible": _pick(row, "opponent_eligible_v4"),
                    "opponent_veto_reason": _pick(row, "opponent_veto_reason_v4"),
                },
                "indices": {
                    "distance_index": _pick(row, "距離指数", "distance_index"),
                    "course_index": _pick(row, "コース指数", "course_index"),
                    "race3": _pick(row, "3走前", "race3"),
                    "race2": _pick(row, "2走前", "race2"),
                    "race1": _pick(row, "前走", "race1"),
                    "recent3_average": _pick(row, "平均指数", "3走平均", "近3走平均", "avg5"),
                    "year_max_index": _pick(row, "過去1年最高指数", "year_max_index", "最高指数"),
                    "star_max_index": _pick(row, "★最高指数", "star_max_index", "★最高"),
                    "star_max_race": _pick(row, "★該当走", "star_max_race"),
                    "star_max_condition": _pick(row, "★条件", "star_max_condition"),
                },
                "support": {
                    "time_index_evaluation": _material_detail(trust, "time_index"),
                    "jockey_evaluation": _material_detail(trust, "jockey"),
                    "age_evaluation": _material_detail(trust, "age"),
                    "age_adjustment": _pick(row, "年齢補正", "age_adjustment"),
                    "frame_style_evaluation": _material_detail(trust, "frame_style"),
                    "training_evaluation": _pick(row, "調教評価", "追切評価", "_調教評価記号") if race_type == "jra" else "",
                    "state": _pick(row, "状態", "form_state", "勢いランク", "momentum_rank"),
                    "weight_material": _material_detail(trust, "weight"),
                    "first_blinker": bool(_material_detail(trust, "first_blinker")) if race_type == "jra" else False,
                    "supplement": _pick(row, "補足", "supplement_note", "評価／検討材料", "評価/検討材料"),
                    "value_signal": bool(value_support.get("value_signal")),
                    "value_reason": value_support.get("value_reason", ""),
                    "value_odds_at_decision": value_support.get("value_odds_at_decision"),
                    "value_ability_band": value_support.get("value_ability_band"),
                    "value_ability_rank": value_support.get("value_ability_rank"),
                    "value_current_rank": value_support.get("value_current_rank"),
                    "value_mark": value_support.get("value_mark"),
                    "value_plus_materials": list(value_support.get("value_plus_materials") or []),
                    "value_minus_materials": list(value_support.get("value_minus_materials") or []),
                    "training_display": value_support.get("training_display", ""),
                    "stable_comment_display": value_support.get("stable_comment_display", ""),
                    "course_material_label": value_support.get("course_material_label", ""),
                    "netkeiba_favorable_label": value_support.get("netkeiba_favorable_label", ""),
                },
                "horse_trust": trust,
                "horse_trust_summary": build_horse_trust_summary(row, race_type),
                "value_support": {
                    "value_signal": bool(value_support.get("value_signal")),
                    "value_candidate": bool(value_support.get("value_candidate")),
                    "value_reason": value_support.get("value_reason", ""),
                    "value_score": value_support.get("value_score"),
                    "odds_at_decision": value_support.get("value_odds_at_decision"),
                    "ability_band": value_support.get("value_ability_band"),
                    "ability_rank": value_support.get("value_ability_rank"),
                    "current_rank": value_support.get("value_current_rank"),
                    "mark": value_support.get("value_mark"),
                    "plus_materials": list(value_support.get("value_plus_materials") or []),
                    "minus_materials": list(value_support.get("value_minus_materials") or []),
                    "audit": value_support.get("value_audit", {}),
                },
                "recent_races": recent_races,
                "condition_fit_mark": condition_fit.get("condition_fit_mark"),
                "condition_fit_level": condition_fit.get("condition_fit_level"),
                "condition_fit_reason": condition_fit.get("condition_fit_reason"),
                "condition_fit_data_status": condition_fit.get("condition_fit_data_status"),
                "matched_past_runs": condition_fit.get("matched_past_runs", []),
                "final_betting_context": final_context,
                "gauge": momentum.get("gauge"),
                "trend": momentum.get("trend"),
                "trend_label": momentum.get("trend_label"),
                "start_evaluation": race_shape.get("start_evaluation"),
                "corner4_evaluation": race_shape.get("corner4_evaluation"),
                "corner4_rank": race_shape.get("corner4_rank"),
                "straight_evaluation": race_shape.get("straight_evaluation"),
                "straight_rank": race_shape.get("straight_rank"),
                "pace_fit": race_shape.get("pace_fit"),
                "front_survival_flag": race_shape.get("front_survival_flag"),
                "race_comment_role": race_shape.get("race_comment_role"),
            }
        )
    if clean_text(getattr(result, "logic_version", "v3")).lower() not in {"v4", "v4.1"}:
        for snapshot in rows:
            snapshot.pop("ver4", None)
            prediction = snapshot.get("prediction")
            if isinstance(prediction, dict):
                for key in ("horse_score_v4", "race_rank_v4", "group_v4", "mark_v4"):
                    prediction.pop(key, None)
    return rows


def _investment_snapshot(decision: Any) -> dict[str, Any]:
    if decision is None:
        return {"decision": "SKIP", "reason_lines": ["購入判断は未生成です。"]}
    selected = getattr(decision, "selected", None)
    judgement = _decision_label(getattr(decision, "judgement", ""))
    audit = getattr(selected, "audit", {}) if selected is not None else {}
    horse_trust = getattr(decision, "horse_trust", ()) or audit.get("horse_trust", ())
    horse_trust_summary = list(getattr(decision, "horse_trust_summary", ()) or audit.get("horse_trust_summary", ()) or [])
    final_context = list(getattr(decision, "final_betting_context", ()) or audit.get("final_betting_context", ()) or [])
    final_context_summary = list(getattr(decision, "final_context_summary", ()) or audit.get("final_context_summary", ()) or [])
    ticket_alignment = list(getattr(decision, "ticket_alignment", ()) or audit.get("ticket_alignment", ()) or [])
    ticket_alignment_summary = list(getattr(decision, "ticket_alignment_summary", ()) or audit.get("ticket_alignment_summary", ()) or [])
    payload = {
        "decision": judgement,
        "decision_label": getattr(decision, "judgement", ""),
        "selected_strategy": getattr(selected, "label", "") if selected is not None else "",
        "strategy_id": getattr(selected, "strategy_id", "") if selected is not None else "",
        "strategy_score": audit.get("strategy_score", ""),
        "expected_roi": getattr(selected, "expected_roi", None) if selected is not None else None,
        "confidence": audit.get("confidence", ""),
        "ticket_type": getattr(selected, "ticket_type", "") if selected is not None else "",
        "tickets": list(getattr(selected, "tickets", ()) if selected is not None else ()),
        "purchase_points": getattr(selected, "ticket_count", 0) if selected is not None else 0,
        "investment": getattr(decision, "total_stake", 0),
        "horse_trust": horse_trust,
        "horse_trust_summary": horse_trust_summary,
        "final_betting_context": final_context,
        "final_context_summary": final_context_summary,
        "ticket_alignment": ticket_alignment,
        "ticket_alignment_summary": ticket_alignment_summary,
        "ticket_rationale": audit.get("ticket_rationale", {}),
        "matched_conditions": list(getattr(selected, "matched_conditions", ()) if selected is not None else ()),
        "reason_lines": list(getattr(decision, "reason_lines", ()) or []),
        "buy_reason": audit.get("adopted_reason", ""),
        "hold_reason": "" if judgement != "HOLD" else audit.get("adopted_reason", ""),
        "skip_reason": "" if judgement != "SKIP" else " / ".join(getattr(decision, "reason_lines", ()) or []),
        "avoid_conditions": audit.get("avoid_matched", []),
        "audit_rows": list(getattr(decision, "audit_rows", ()) or []),
    }
    logic_version = clean_text(getattr(decision, "logic_version", "v3")) or "v3"
    if logic_version == "v4":
        payload.update(
            {
                "logic_version": logic_version,
                "decision_v4": clean_text(getattr(decision, "decision_v4", "")),
                "legacy_decision": judgement,
                "axis_score": getattr(decision, "axis_score", 0.0),
                "axis_confidence_v4": getattr(decision, "axis_confidence_v4", ""),
                "ticket_candidate_score": getattr(decision, "ticket_candidate_score", 0.0),
                "ticket_veto_reason": getattr(decision, "ticket_veto_reason", ""),
            }
        )
    elif logic_version == "practical":
        practical_decision = clean_text(getattr(decision, "practical_decision", "")) or "WATCH"
        practical_reason = clean_text(getattr(decision, "practical_reason", ""))
        payload.update(
            {
                "logic_version": "practical",
                "decision": practical_decision,
                "decision_label": practical_decision,
                "buy_reason": practical_reason if practical_decision == "BUY" else "",
                "watch_reason": practical_reason if practical_decision == "WATCH" else "",
                "reason_lines": list(
                    getattr(decision, "practical_reason_lines", ()) or getattr(decision, "reason_lines", ()) or ()
                ),
                "practical_config_version": getattr(decision, "practical_config_version", ""),
                "honmei_horse_no": getattr(decision, "honmei_horse_no", ""),
                "honmei_horse_name": getattr(decision, "honmei_horse_name", ""),
                "ticket_type": getattr(selected, "ticket_type", "") if selected is not None else "",
                "tickets": list(getattr(selected, "tickets", ()) if selected is not None else ()),
                "purchase_points": getattr(selected, "ticket_count", 0) if selected is not None else 0,
                "investment": getattr(decision, "total_stake", 0),
                "expected_roi": None,
            }
        )
    return payload


def _merged_horse_rows(result: PredictionResult) -> list[dict[str, Any]]:
    rows_by_no: dict[str, dict[str, Any]] = {}
    for table in (result.overall_table, result.horse_evaluation):
        if table is None or not isinstance(table, pd.DataFrame) or table.empty:
            continue
        for _, raw in table.iterrows():
            record = raw.to_dict()
            number = horse_no(_pick(record, "馬番", "horse_no", "horse_number", "馬"))
            if not number:
                continue
            merged = rows_by_no.setdefault(number, {})
            for key, value in record.items():
                if not _missing(value):
                    merged[str(key)] = value
    return sorted(rows_by_no.values(), key=lambda row: int(horse_no(_pick(row, "馬番", "horse_no", "馬")) or 999))


def _flatten_for_csv(source: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in source.items():
        name = f"{prefix}{key}" if not prefix else f"{prefix}_{key}"
        if isinstance(value, Mapping):
            out.update(_flatten_for_csv(value, name))
        elif isinstance(value, list):
            out[name] = json.dumps(_json_ready(value), ensure_ascii=False)
        else:
            out[name] = _json_ready(value)
    return out


def _ticket_rationale_text(investment: Mapping[str, Any]) -> str:
    rationale = investment.get("ticket_rationale")
    if not isinstance(rationale, Mapping):
        rationale = {}
    pieces = [
        clean_text(investment.get("selected_strategy")),
        f"{investment.get('purchase_points', 0)}点",
        f"回収率 {investment.get('expected_roi')}%" if investment.get("expected_roi") not in (None, "") else "",
        f"対象{rationale.get('sample_races')}R" if rationale.get("sample_races") not in (None, "") else "",
        f"的中率{rationale.get('hit_rate')}%" if rationale.get("hit_rate") not in (None, "") else "",
    ]
    return "\n".join(piece for piece in pieces if clean_text(piece)) or "記録なし"


def _material_detail(materials: list[dict[str, Any]], key: str) -> Any:
    for material in materials:
        if material.get("key") == key:
            return material.get("detail") or material.get("display")
    return ""


def _pick(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name not in row:
            continue
        value = row.get(name)
        if not _missing(value):
            return value
    return ""


def _mapping(value: Any) -> Mapping[str, Any]:
    return value if isinstance(value, Mapping) else {}


def _missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return clean_text(value).lower() in {"", "-", "—", "nan", "none", "null", "データなし", "未取得"}


def _decision_label(value: Any) -> str:
    text = clean_text(value)
    return {"買い": "BUY", "保留": "HOLD", "見送り": "SKIP"}.get(text, text.upper() if text else "SKIP")


def _race_id_from_source_files(source_files: Mapping[str, str] | None) -> str:
    text = " ".join(str(value) for value in (source_files or {}).values())
    match = re.search(r"race_id=(\d{10,12})", text)
    if match:
        return match.group(1)
    match = re.search(r"(\d{10,12})", text)
    return match.group(1) if match else ""


def _race_date(info: Mapping[str, Any], race_id: Any) -> str:
    direct = clean_text(_pick(info, "date", "race_date", "開催日", "日付"))
    if direct:
        digits = re.sub(r"\D", "", direct)
        if len(digits) >= 8:
            return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
        return direct
    race_id_text = clean_text(race_id)
    if re.match(r"20\d{6}", race_id_text):
        return f"{race_id_text[:4]}-{race_id_text[4:6]}-{race_id_text[6:8]}"
    return datetime.now().strftime("%Y-%m-%d")


def _race_number(value: Any) -> str:
    text = clean_text(value)
    match = re.search(r"([1-9]|1[0-2])\s*R", text, flags=re.IGNORECASE)
    if match:
        return f"{match.group(1)}R"
    match = re.search(r"(?<!\d)([1-9]|1[0-2])(?!\d)", text)
    return f"{match.group(1)}R" if match else text


def _age_number(value: Any) -> int | None:
    match = re.search(r"(\d{1,2})", clean_text(value))
    return int(match.group(1)) if match else None


def _join_nonempty(values: Any, sep: str = " ") -> str:
    if isinstance(values, (str, bytes)):
        return clean_text(values)
    return sep.join(clean_text(value) for value in values if clean_text(value))


def _safe_filename(value: Any) -> str:
    text = clean_text(value) or "race"
    return re.sub(r'[\\/:*?"<>|\s]+', "_", text).strip("_")


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if isinstance(value, pd.DataFrame):
        return _json_ready(value.to_dict("records"))
    if isinstance(value, pd.Series):
        return _json_ready(value.to_dict())
    if hasattr(value, "item"):
        try:
            return value.item()
        except Exception:
            pass
    if _missing(value):
        return ""
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        number = to_float(value)
        if number is not None:
            return int(number) if number.is_integer() else number
    return value

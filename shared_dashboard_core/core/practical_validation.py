# -*- coding: utf-8 -*-
from __future__ import annotations

import csv
import hashlib
import json
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path
from typing import Any

from .prediction_history import build_prediction_snapshot
from .practical_mode import (
    PRACTICAL_CONFIG_VERSION,
    PRACTICAL_DECISION_BUY,
    PRACTICAL_STAKE_YEN,
    practical_rules_snapshot,
)
from .purchase_conditions import clean_text, horse_no, to_float


PRACTICAL_HISTORY_ROOT = Path("prediction_history") / "practical_100r"
PRACTICAL_TARGET_RACES = 100
MANIFEST_NAME = "manifest.json"
PREDICTIONS_CSV_NAME = "practical_100r_predictions.csv"
RESULTS_CSV_NAME = "practical_100r_results.csv"

PREDICTION_FIELDS = (
    "race_id",
    "date",
    "race_type",
    "venue",
    "race_number",
    "race_name",
    "logic_version",
    "practical_config_version",
    "config_signature",
    "marks_json",
    "mark_◎",
    "mark_○",
    "mark_▲",
    "mark_△",
    "mark_✓",
    "top3_horse_numbers_json",
    "honmei_horse_no",
    "honmei_horse_name",
    "honmei_odds",
    "honmei_popularity",
    "honmei_ai_score",
    "honmei_ability_value",
    "honmei_condition_mark",
    "honmei_condition_level",
    "honmei_condition_data_status",
    "honmei_matched_past_runs_json",
    "decision",
    "buy_reason",
    "watch_reason",
    "reason_lines_json",
    "recommended_ticket",
    "ticket_numbers_json",
    "investment_yen",
    "prediction_created_at",
    "prediction_file_sha256",
)

RESULT_FIELDS = (
    "race_id",
    "race_type",
    "decision",
    "honmei_horse_no",
    "honmei_finish",
    "winner_horse_no",
    "top3_horse_numbers_json",
    "top3_winner_captured",
    "win_payoff_per_100_yen",
    "investment_yen",
    "payout_yen",
    "profit_yen",
    "hit",
    "settled_at",
    "prediction_file_sha256",
)


def practical_config_signature() -> str:
    payload = {
        "config_version": PRACTICAL_CONFIG_VERSION,
        "rules": practical_rules_snapshot(),
    }
    encoded = json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def freeze_practical_prediction(
    result: Any,
    decision: Any = None,
    *,
    root: str | Path = PRACTICAL_HISTORY_ROOT,
) -> Path:
    """Save one immutable pre-race prediction and append the frozen CSV once."""

    if clean_text(getattr(result, "logic_version", "")) != "practical":
        raise ValueError("100R検証へ保存できるのは実戦モードの予想だけです。")
    snapshot = build_prediction_snapshot(result, decision)
    info = snapshot.get("race_info") if isinstance(snapshot.get("race_info"), Mapping) else {}
    race_id = clean_text(info.get("race_id"))
    if not race_id:
        raise ValueError("race_idがないため100R検証へ固定できません。")

    root_path = Path(root)
    root_path.mkdir(parents=True, exist_ok=True)
    manifest = _load_or_create_manifest(root_path)
    prediction_csv = root_path / PREDICTIONS_CSV_NAME
    existing_rows = _read_csv(prediction_csv)
    race_dir = root_path / "races" / race_id
    prediction_path = race_dir / "prediction.json"
    if any(clean_text(row.get("race_id")) == race_id for row in existing_rows) and prediction_path.exists():
        return prediction_path
    if len(existing_rows) >= int(manifest["target_races"]):
        raise RuntimeError("100R固定検証は上限へ到達しています。新しい予想は追加できません。")

    race_dir.mkdir(parents=True, exist_ok=True)
    prediction_bytes = json.dumps(snapshot, ensure_ascii=False, indent=2).encode("utf-8")
    if prediction_path.exists():
        existing_bytes = prediction_path.read_bytes()
        if existing_bytes != prediction_bytes:
            raise RuntimeError("既存の固定予想と内容が異なるため上書きを拒否しました。")
    else:
        prediction_path.write_bytes(prediction_bytes)
    prediction_hash = _sha256(prediction_path.read_bytes())

    row = _prediction_row(snapshot, prediction_hash)
    _append_csv_once(prediction_csv, PREDICTION_FIELDS, row, key="race_id")
    count = len(_read_csv(prediction_csv))
    manifest["prediction_count"] = count
    manifest["status"] = "complete" if count >= int(manifest["target_races"]) else "active"
    manifest["updated_at"] = datetime.now().isoformat(timespec="seconds")
    _write_json(root_path / MANIFEST_NAME, manifest)
    return prediction_path


def settle_practical_result(
    race_id: Any,
    result_payload: Mapping[str, Any],
    *,
    root: str | Path = PRACTICAL_HISTORY_ROOT,
) -> Path:
    """Join a result only after an immutable prediction exists."""

    race_id_text = clean_text(race_id)
    root_path = Path(root)
    prediction_path = root_path / "races" / race_id_text / "prediction.json"
    if not prediction_path.exists():
        raise FileNotFoundError("固定済みprediction.jsonがないため結果を照合できません。")
    snapshot = json.loads(prediction_path.read_text(encoding="utf-8"))
    frozen_race_id = clean_text((snapshot.get("race_info") or {}).get("race_id"))
    if frozen_race_id != race_id_text:
        raise ValueError("race_idと固定予想が一致しません。")
    payload_race_id = clean_text(result_payload.get("race_id"))
    if payload_race_id and payload_race_id != race_id_text:
        raise ValueError("resultのrace_idが固定予想と一致しません。")

    race_dir = prediction_path.parent
    settlement_path = race_dir / "settlement.json"
    if settlement_path.exists():
        return settlement_path
    settlement = _settlement(snapshot, result_payload, _sha256(prediction_path.read_bytes()))
    result_path = race_dir / "result.json"
    _write_json(result_path, dict(result_payload))
    _write_json(settlement_path, settlement)
    _append_csv_once(root_path / RESULTS_CSV_NAME, RESULT_FIELDS, settlement, key="race_id")
    return settlement_path


def practical_validation_summary(
    *,
    root: str | Path = PRACTICAL_HISTORY_ROOT,
) -> dict[str, Any]:
    root_path = Path(root)
    predictions = _read_csv(root_path / PREDICTIONS_CSV_NAME)
    results_by_id = {
        clean_text(row.get("race_id")): row
        for row in _read_csv(root_path / RESULTS_CSV_NAME)
        if clean_text(row.get("race_id"))
    }
    scopes: dict[str, Any] = {}
    for scope in ("ALL", "JRA", "NAR"):
        selected = [
            row
            for row in predictions
            if scope == "ALL" or clean_text(row.get("race_type")).upper() == scope
        ]
        settled = [results_by_id[clean_text(row.get("race_id"))] for row in selected if clean_text(row.get("race_id")) in results_by_id]
        scopes[scope] = _scope_summary(selected, settled)
    manifest = _read_json(root_path / MANIFEST_NAME) or {}
    payload = {
        "schema_version": 1,
        "config_version": PRACTICAL_CONFIG_VERSION,
        "config_signature": practical_config_signature(),
        "target_races": PRACTICAL_TARGET_RACES,
        "prediction_count": len(predictions),
        "settled_count": len(results_by_id),
        "status": manifest.get("status", "not_started"),
        "scopes": scopes,
        "generated_at": datetime.now().isoformat(timespec="seconds"),
    }
    if root_path.exists():
        _write_json(root_path / "practical_100r_summary.json", payload)
    return payload


def _load_or_create_manifest(root: Path) -> dict[str, Any]:
    path = root / MANIFEST_NAME
    signature = practical_config_signature()
    existing = _read_json(path)
    if existing:
        if clean_text(existing.get("config_signature")) != signature:
            raise RuntimeError("100R検証開始後に実戦ルールが変化したため保存を停止しました。")
        return dict(existing)
    payload = {
        "schema_version": 1,
        "config_version": PRACTICAL_CONFIG_VERSION,
        "config_signature": signature,
        "target_races": PRACTICAL_TARGET_RACES,
        "prediction_count": 0,
        "status": "active",
        "started_at": datetime.now().isoformat(timespec="seconds"),
        "rules": practical_rules_snapshot(),
        "note": "100R終了まで予想・BUY条件・券種ルールを変更しない固定検証。",
    }
    _write_json(path, payload)
    return payload


def _prediction_row(snapshot: Mapping[str, Any], prediction_hash: str) -> dict[str, Any]:
    race_info = snapshot.get("race_info") if isinstance(snapshot.get("race_info"), Mapping) else {}
    investment = snapshot.get("investment_decision") if isinstance(snapshot.get("investment_decision"), Mapping) else {}
    horses = [horse for horse in snapshot.get("horses", []) or [] if isinstance(horse, Mapping)]
    marks: dict[str, list[dict[str, str]]] = {mark: [] for mark in ("◎", "○", "▲", "△", "✓")}
    for item in horses:
        prediction = item.get("prediction") if isinstance(item.get("prediction"), Mapping) else {}
        mark = clean_text(prediction.get("mark"))
        if mark in marks:
            marks[mark].append(
                {
                    "number": horse_no(item.get("horse_no")),
                    "name": clean_text(item.get("horse_name")),
                }
            )
    honmei_no = horse_no(investment.get("honmei_horse_no"))
    honmei = next((item for item in horses if horse_no(item.get("horse_no")) == honmei_no), {})
    honmei_prediction = honmei.get("prediction") if isinstance(honmei.get("prediction"), Mapping) else {}
    top3_numbers = [item["number"] for mark in ("◎", "○", "▲") for item in marks[mark]]
    decision = clean_text(investment.get("decision")) or "WATCH"
    return {
        "race_id": race_info.get("race_id", ""),
        "date": race_info.get("date", ""),
        "race_type": race_info.get("race_type", ""),
        "venue": race_info.get("venue", ""),
        "race_number": race_info.get("race_number", ""),
        "race_name": race_info.get("race_name", ""),
        "logic_version": snapshot.get("logic_version", ""),
        "practical_config_version": investment.get("practical_config_version", PRACTICAL_CONFIG_VERSION),
        "config_signature": practical_config_signature(),
        "marks_json": json.dumps(marks, ensure_ascii=False, separators=(",", ":")),
        **{f"mark_{mark}": _mark_text(marks[mark]) for mark in marks},
        "top3_horse_numbers_json": json.dumps(top3_numbers, ensure_ascii=False),
        "honmei_horse_no": honmei_no,
        "honmei_horse_name": honmei.get("horse_name", investment.get("honmei_horse_name", "")),
        "honmei_odds": honmei.get("odds", ""),
        "honmei_popularity": honmei.get("popularity", ""),
        "honmei_ai_score": honmei_prediction.get("ai_score", ""),
        "honmei_ability_value": honmei_prediction.get("ability_value", ""),
        "honmei_condition_mark": honmei.get("condition_fit_mark", ""),
        "honmei_condition_level": honmei.get("condition_fit_level", ""),
        "honmei_condition_data_status": honmei.get("condition_fit_data_status", ""),
        "honmei_matched_past_runs_json": json.dumps(honmei.get("matched_past_runs", []), ensure_ascii=False),
        "decision": decision,
        "buy_reason": investment.get("buy_reason", ""),
        "watch_reason": investment.get("watch_reason", ""),
        "reason_lines_json": json.dumps(investment.get("reason_lines", []), ensure_ascii=False),
        "recommended_ticket": investment.get("ticket_type", ""),
        "ticket_numbers_json": json.dumps(investment.get("tickets", []), ensure_ascii=False),
        "investment_yen": int(to_float(investment.get("investment")) or 0),
        "prediction_created_at": (snapshot.get("audit") or {}).get("prediction_generated_at", ""),
        "prediction_file_sha256": prediction_hash,
    }


def _settlement(
    snapshot: Mapping[str, Any],
    result_payload: Mapping[str, Any],
    prediction_hash: str,
) -> dict[str, Any]:
    info = snapshot.get("race_info") if isinstance(snapshot.get("race_info"), Mapping) else {}
    investment = snapshot.get("investment_decision") if isinstance(snapshot.get("investment_decision"), Mapping) else {}
    honmei_no = horse_no(investment.get("honmei_horse_no"))
    finishes = _finish_map(result_payload)
    if not finishes:
        raise ValueError("resultに確定着順がありません。")
    honmei_finish = finishes.get(honmei_no)
    winner_no = next((number for number, finish in finishes.items() if finish == 1), "")
    horses = [horse for horse in snapshot.get("horses", []) or [] if isinstance(horse, Mapping)]
    top3_numbers = [
        horse_no(item.get("horse_no"))
        for item in horses
        if clean_text((item.get("prediction") or {}).get("mark")) in {"◎", "○", "▲"}
    ]
    decision = clean_text(investment.get("decision")) or "WATCH"
    investment_yen = PRACTICAL_STAKE_YEN if decision == PRACTICAL_DECISION_BUY else 0
    win_payoff = _win_payoff(result_payload, honmei_no)
    hit = bool(decision == PRACTICAL_DECISION_BUY and honmei_finish == 1)
    payout = win_payoff if hit else 0
    return {
        "race_id": info.get("race_id", ""),
        "race_type": info.get("race_type", ""),
        "decision": decision,
        "honmei_horse_no": honmei_no,
        "honmei_finish": honmei_finish if honmei_finish is not None else "",
        "winner_horse_no": winner_no,
        "top3_horse_numbers_json": json.dumps(top3_numbers, ensure_ascii=False),
        "top3_winner_captured": int(bool(winner_no and winner_no in top3_numbers)),
        "win_payoff_per_100_yen": win_payoff,
        "investment_yen": investment_yen,
        "payout_yen": payout,
        "profit_yen": payout - investment_yen,
        "hit": int(hit),
        "settled_at": datetime.now().isoformat(timespec="seconds"),
        "prediction_file_sha256": prediction_hash,
    }


def _scope_summary(predictions: list[dict[str, Any]], settlements: list[dict[str, Any]]) -> dict[str, Any]:
    settled_count = len(settlements)
    honmei_wins = sum(_int(row.get("honmei_finish")) == 1 for row in settlements)
    honmei_top3 = sum(0 < _int(row.get("honmei_finish")) <= 3 for row in settlements)
    top3_capture = sum(_int(row.get("top3_winner_captured")) == 1 for row in settlements)
    buys = [row for row in settlements if clean_text(row.get("decision")) == PRACTICAL_DECISION_BUY]
    investment = sum(_int(row.get("investment_yen")) for row in buys)
    payout = sum(_int(row.get("payout_yen")) for row in buys)
    hits = sum(_int(row.get("hit")) for row in buys)
    payouts_desc = sorted((_int(row.get("payout_yen")) for row in buys), reverse=True)
    return {
        "prediction_count": len(predictions),
        "settled_count": settled_count,
        "honmei_wins": honmei_wins,
        "honmei_win_rate": _rate(honmei_wins, settled_count),
        "honmei_top3": honmei_top3,
        "honmei_top3_rate": _rate(honmei_top3, settled_count),
        "top3_winner_capture_rate": _rate(top3_capture, settled_count),
        "buy_count": len(buys),
        "buy_rate": _rate(len([row for row in predictions if clean_text(row.get("decision")) == PRACTICAL_DECISION_BUY]), len(predictions)),
        "buy_honmei_win_rate": _rate(hits, len(buys)),
        "buy_honmei_top3_rate": _rate(sum(0 < _int(row.get("honmei_finish")) <= 3 for row in buys), len(buys)),
        "hits": hits,
        "hit_rate": _rate(hits, len(buys)),
        "investment_yen": investment,
        "payout_yen": payout,
        "profit_yen": payout - investment,
        "return_rate": _rate(payout, investment),
        "return_rate_without_top1_payout": _excluded_return_rate(payouts_desc, 1),
        "return_rate_without_top2_payouts": _excluded_return_rate(payouts_desc, 2),
        "highest_payout_yen": payouts_desc[0] if payouts_desc else 0,
        "highest_payout_share": _rate(payouts_desc[0] if payouts_desc else 0, payout),
    }


def _excluded_return_rate(payouts_desc: list[int], remove_count: int) -> float | None:
    if len(payouts_desc) <= remove_count:
        return None
    remaining = payouts_desc[remove_count:]
    return _rate(sum(remaining), len(remaining) * PRACTICAL_STAKE_YEN)


def _finish_map(payload: Mapping[str, Any]) -> dict[str, int]:
    rows = payload.get("results") or payload.get("着順") or []
    result: dict[str, int] = {}
    if isinstance(rows, Mapping):
        rows = [{"horse_no": key, "finish": value} for key, value in rows.items()]
    if not isinstance(rows, list):
        return result
    for item in rows:
        if not isinstance(item, Mapping):
            continue
        number = horse_no(_first(item, "horse_no", "horse_number", "number", "馬番"))
        finish = _int(_first(item, "finish", "position", "着順"))
        if number and finish > 0:
            result[number] = finish
    return result


def _win_payoff(payload: Mapping[str, Any], honmei_no: str) -> int:
    payoffs = payload.get("payoffs") if isinstance(payload.get("payoffs"), Mapping) else {}
    win = payoffs.get("win") if isinstance(payoffs, Mapping) else None
    if win is None:
        win = payload.get("win_payoffs") or payload.get("単勝") or []
    if isinstance(win, Mapping):
        if honmei_no in win:
            return _int(win.get(honmei_no))
        win = [win]
    if not isinstance(win, list):
        return 0
    for item in win:
        if isinstance(item, Mapping):
            number = horse_no(_first(item, "horse_no", "horse_number", "number", "馬番"))
            if number == honmei_no:
                return _int(_first(item, "payout", "payoff", "amount", "払戻"))
        elif isinstance(item, (list, tuple)) and len(item) >= 2 and horse_no(item[0]) == honmei_no:
            return _int(item[1])
    return 0


def _mark_text(items: list[dict[str, str]]) -> str:
    return " / ".join(" ".join(part for part in (item.get("number"), item.get("name")) if part) for item in items)


def _append_csv_once(path: Path, fields: tuple[str, ...], row: Mapping[str, Any], *, key: str) -> None:
    existing = _read_csv(path)
    key_value = clean_text(row.get(key))
    if any(clean_text(item.get(key)) == key_value for item in existing):
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not path.exists() or path.stat().st_size == 0
    with path.open("a", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fields), extrasaction="ignore")
        if write_header:
            writer.writeheader()
        writer.writerow({field: row.get(field, "") for field in fields})


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists() or path.stat().st_size == 0:
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    value = json.loads(path.read_text(encoding="utf-8"))
    return value if isinstance(value, dict) else None


def _sha256(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _first(row: Mapping[str, Any], *names: str) -> Any:
    for name in names:
        if name in row and clean_text(row.get(name)):
            return row.get(name)
    return ""


def _int(value: Any) -> int:
    return int(to_float(value) or 0)


def _rate(numerator: float, denominator: float) -> float | None:
    if denominator <= 0:
        return None
    return round(numerator / denominator * 100.0, 1)

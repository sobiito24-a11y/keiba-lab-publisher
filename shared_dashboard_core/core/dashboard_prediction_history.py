from __future__ import annotations

import csv
import io
import json
import re
import zipfile
from datetime import datetime
from pathlib import Path
from typing import Any, Mapping

from .dashboard_cards import DashboardDetailError, load_detail_json


def build_prediction_history_zip(summary: Mapping[str, Any], analysis_dir: str | Path, *, source: str = "") -> bytes:
    race_type = _race_type(summary, source)
    summary_day = _summary_date(summary)
    buffer = io.BytesIO()
    with zipfile.ZipFile(buffer, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for item in _all_decisions(summary):
            detail = _load_detail(item, analysis_dir)
            snapshot = build_prediction_snapshot(item, detail, race_type=race_type, summary_date=summary_day, source=source)
            race_id = _text(snapshot["race_info"].get("race_id")) or _safe_filename(snapshot["race_info"].get("race_name") or "race")
            folder = f"{race_type}/{summary_day.replace('-', '')}/{race_id}"
            archive.writestr(f"{folder}/prediction.json", json.dumps(snapshot, ensure_ascii=False, indent=2))
            archive.writestr(f"{folder}/prediction.csv", _prediction_csv_bytes(snapshot))
            archive.writestr(f"{folder}/summary.txt", _summary_text(snapshot).encode("utf-8"))
    return buffer.getvalue()


def history_zip_file_name(summary: Mapping[str, Any], *, source: str = "") -> str:
    race_type = _race_type(summary, source).upper()
    date = _summary_date(summary).replace("-", "")
    return f"{race_type}_{date}_prediction_history.zip"


def build_prediction_snapshot(
    item: Mapping[str, Any],
    detail: Mapping[str, Any] | None,
    *,
    race_type: str,
    summary_date: str,
    source: str = "",
) -> dict[str, Any]:
    detail = detail or {}
    race_info = detail.get("race_info") if isinstance(detail.get("race_info"), Mapping) else {}
    horses = _horse_rows(detail, item)
    return _json_ready(
        {
            "schema_version": 1,
            "race_info": {
                "race_id": _first(item, "race_id") or _first(race_info, "race_id"),
                "race_type": race_type,
                "date": summary_date,
                "venue": _first(item, "venue", "開催場") or _first(race_info, "venue", "racecourse", "競馬場"),
                "race_number": _first(item, "race_number", "race_no", "R") or _first(race_info, "race_number", "race_no", "R"),
                "race_name": _first(item, "race_name", "race_title") or _first(detail, "race_name"),
                "distance": _first(race_info, "distance", "距離"),
                "surface": _first(race_info, "surface", "芝ダート", "馬場種別"),
                "class": _first(race_info, "class", "race_class", "クラス"),
                "post_time": _first(item, "post_time") or _first(race_info, "post_time", "start_time", "発走"),
                "head_count": len(horses),
            },
            "horses": horses,
            "investment_decision": {
                "decision": _first(item, "decision"),
                "selected_strategy": _first(item, "selected_strategy"),
                "strategy_id": _first(item, "strategy_id"),
                "strategy_score": _first(item, "strategy_score", "score"),
                "expected_roi": _first(item, "expected_roi", "roi"),
                "confidence": _first(item, "confidence", "investment_rank"),
                "ticket": _first(item, "ticket"),
                "combinations": item.get("combinations", []),
                "points": _first(item, "points"),
                "investment": _first(item, "investment"),
                "reason": _first(item, "reason"),
                "avoid_reason": _first(item, "avoid_reason"),
                "horses": item.get("horses", []),
            },
            "audit": {
                "source": source,
                "detail_path": _first(item, "detail_path"),
                "prediction_generated_at": datetime.now().isoformat(timespec="seconds"),
                "summary_generated_at": _first(item, "generated_at"),
                "summary_race_type": race_type,
            },
            "result_file": {
                "schema_version": 1,
                "race_id": _first(item, "race_id"),
                "race_type": race_type,
                "results": [],
                "payoffs": {},
                "note": "結果はprediction.jsonへ上書きせず、result.jsonとして別保存します。",
            },
        }
    )


def _horse_rows(detail: Mapping[str, Any], item: Mapping[str, Any]) -> list[dict[str, Any]]:
    rows_by_no: dict[str, dict[str, Any]] = {}
    for table_name in ("overall_table", "horse_evaluation"):
        rows = detail.get(table_name)
        if not isinstance(rows, list):
            continue
        for raw in rows:
            if not isinstance(raw, Mapping):
                continue
            number = _horse_number(_first(raw, "馬番", "horse_no", "horse_number"))
            if not number:
                continue
            merged = rows_by_no.setdefault(number, {})
            for key, value in raw.items():
                if not _missing(value):
                    merged[str(key)] = value
    if not rows_by_no:
        for raw in item.get("horses", []) or []:
            if not isinstance(raw, Mapping):
                continue
            number = _horse_number(_first(raw, "number", "馬番", "horse_no"))
            if number:
                rows_by_no[number] = dict(raw)
    horses: list[dict[str, Any]] = []
    for number in sorted(rows_by_no, key=lambda value: int(value) if value.isdigit() else 999):
        row = rows_by_no[number]
        horses.append(
            {
                "horse_no": number,
                "horse_name": _first(row, "馬名", "horse_name", "name"),
                "sex_age": _first(row, "馬年齢", "性齢", "馬齢"),
                "weight": _first(row, "斤量", "weight"),
                "weight_detail": _first(row, "斤量詳細", "weight_detail"),
                "jockey": _first(row, "騎手", "jockey"),
                "jockey_detail": _first(row, "騎手詳細", "jockey_detail"),
                "popularity": _first(row, "人気", "popularity"),
                "odds": _first(row, "単勝オッズ", "オッズ", "odds"),
                "prediction": {
                    "ai_score": _first(row, "AI点", "normalized_ai_score", "ai_score"),
                    "ai_rank": _first(row, "AI順位", "ai_rank"),
                    "raw_score": _first(row, "raw_score", "_raw_score"),
                    "ability_value": _first(row, "能力評価値", "ability_display_score"),
                    "mark": _first(row, "表示印", "display_mark", "最終印", "印", "mark", "role"),
                    "display_group": _first(row, "グループ", "display_group", "group"),
                    "original_mark": _first(row, "original_mark", "old_final_mark", "元印"),
                },
                "indices": {
                    "distance_index": _first(row, "距離指数", "distance_index"),
                    "course_index": _first(row, "コース指数", "course_index"),
                    "race3": _first(row, "3走前", "race3"),
                    "race2": _first(row, "2走前", "race2"),
                    "race1": _first(row, "前走", "race1"),
                    "recent3_average": _first(row, "平均指数", "3走平均", "近3走平均", "avg5"),
                    "year_max_index": _first(row, "過去1年最高指数", "year_max_index", "最高指数"),
                    "star_max_index": _first(row, "★最高指数", "star_max_index", "★最高"),
                },
                "support": {
                    "trust_summary": _first(row, "horse_trust_summary", "信頼根拠"),
                    "training_evaluation": _first(row, "調教評価", "追切評価", "_調教評価記号"),
                    "state": _first(row, "状態", "form_state", "勢いランク", "momentum_rank"),
                    "supplement": _first(row, "補足", "supplement_note", "評価／検討材料", "評価/検討材料"),
                },
            }
        )
    return horses


def _prediction_csv_bytes(snapshot: Mapping[str, Any]) -> bytes:
    race = snapshot.get("race_info") if isinstance(snapshot.get("race_info"), Mapping) else {}
    investment = snapshot.get("investment_decision") if isinstance(snapshot.get("investment_decision"), Mapping) else {}
    rows: list[dict[str, Any]] = []
    for horse in snapshot.get("horses", []) or []:
        if not isinstance(horse, Mapping):
            continue
        rows.append(
            {
                "race_id": race.get("race_id", ""),
                "race_type": race.get("race_type", ""),
                "date": race.get("date", ""),
                "venue": race.get("venue", ""),
                "race_number": race.get("race_number", ""),
                "decision": investment.get("decision", ""),
                "selected_strategy": investment.get("selected_strategy", ""),
                **_flatten(horse),
            }
        )
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


def _summary_text(snapshot: Mapping[str, Any]) -> str:
    race = snapshot.get("race_info") if isinstance(snapshot.get("race_info"), Mapping) else {}
    investment = snapshot.get("investment_decision") if isinstance(snapshot.get("investment_decision"), Mapping) else {}
    lines = [
        " ".join(_text(part) for part in [race.get("venue"), race.get("race_number"), race.get("race_name")] if _text(part)),
        "",
        f"総合判定：{_text(investment.get('decision')) or 'SKIP'}",
        f"今回買うべき馬券：{_text(investment.get('ticket')) or 'なし'}",
        f"採用戦略：{_text(investment.get('selected_strategy')) or 'なし'}",
        f"期待回収率：{_text(investment.get('expected_roi')) or '—'}",
        "",
        "【対象馬】",
    ]
    for horse in snapshot.get("horses", []) or []:
        if not isinstance(horse, Mapping):
            continue
        prediction = horse.get("prediction") if isinstance(horse.get("prediction"), Mapping) else {}
        lines.append(
            " ".join(
                _text(part)
                for part in [
                    prediction.get("mark"),
                    horse.get("horse_no"),
                    horse.get("horse_name"),
                    f"AI点{prediction.get('ai_score')}" if not _missing(prediction.get("ai_score")) else "",
                ]
                if _text(part)
            )
        )
    return "\n".join(lines)


def _all_decisions(summary: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    items: list[Mapping[str, Any]] = []
    for key in ("buy", "hold", "skip"):
        value = summary.get(key)
        if isinstance(value, list):
            items.extend(item for item in value if isinstance(item, Mapping))
    return items


def _load_detail(item: Mapping[str, Any], analysis_dir: str | Path) -> Mapping[str, Any] | None:
    detail_path = _first(item, "detail_path")
    if not detail_path:
        return None
    try:
        return load_detail_json(analysis_dir, detail_path)
    except DashboardDetailError:
        return None


def _race_type(summary: Mapping[str, Any], source: str) -> str:
    text = f"{_first(summary, 'race_type')} {source}".lower()
    return "nar" if "nar" in text or "daily" in text else "jra"


def _summary_date(summary: Mapping[str, Any]) -> str:
    raw = _first(summary, "date", "summary_date")
    digits = re.sub(r"\D", "", raw)
    if len(digits) >= 8:
        return f"{digits[:4]}-{digits[4:6]}-{digits[6:8]}"
    return raw or datetime.now().strftime("%Y-%m-%d")


def _first(data: Mapping[str, Any], *keys: str) -> Any:
    for key in keys:
        value = data.get(key)
        if not _missing(value):
            return value
    return ""


def _horse_number(value: Any) -> str:
    number = _safe_float(value)
    if number is not None and number.is_integer():
        return str(int(number))
    return _text(value)


def _safe_float(value: Any) -> float | None:
    if _missing(value):
        return None
    try:
        return float(_text(value).replace(",", "").replace("倍", "").replace("%", ""))
    except ValueError:
        return None


def _missing(value: Any) -> bool:
    if value is None:
        return True
    return _text(value).lower() in {"", "-", "—", "nan", "none", "null", "データなし"}


def _text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def _safe_filename(value: Any) -> str:
    return re.sub(r'[\\/:*?"<>|\s]+', "_", _text(value) or "race").strip("_")


def _flatten(source: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in source.items():
        name = f"{prefix}_{key}" if prefix else str(key)
        if isinstance(value, Mapping):
            out.update(_flatten(value, name))
        elif isinstance(value, list):
            out[name] = json.dumps(_json_ready(value), ensure_ascii=False)
        else:
            out[name] = _json_ready(value)
    return out


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_json_ready(item) for item in value]
    if isinstance(value, tuple):
        return [_json_ready(item) for item in value]
    if _missing(value):
        return ""
    return value

from __future__ import annotations

import copy
import json
from typing import Any, Mapping

from .content import marked_map, text


class ResultDataError(ValueError):
    pass


def empty_result(race_id: str = "") -> dict[str, Any]:
    return {"schema_version": 1, "race_id": text(race_id), "results": [], "payoffs": {}}


def load_result_json(data: bytes) -> dict[str, Any]:
    try:
        payload = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ResultDataError("正式結果JSONを読み込めません。") from exc
    if not isinstance(payload, Mapping):
        raise ResultDataError("結果JSONの形式が不正です。")
    if "races" in payload and isinstance(payload["races"], list):
        return {"races": [normalize_result(item) for item in payload["races"] if isinstance(item, Mapping)]}
    return normalize_result(payload)


def normalize_result(payload: Mapping[str, Any]) -> dict[str, Any]:
    result = empty_result(text(payload.get("race_id")))
    rows = payload.get("results") or []
    if not isinstance(rows, list):
        raise ResultDataError("resultsは配列である必要があります。")
    normalized = []
    for row in rows:
        if not isinstance(row, Mapping):
            continue
        horse_no = text(row.get("horse_no") or row.get("number") or row.get("馬番"))
        rank = text(row.get("rank") or row.get("finish") or row.get("着順"))
        if horse_no and rank:
            normalized.append({"horse_no": horse_no, "rank": rank, "horse_name": text(row.get("horse_name") or row.get("馬名"))})
    result["results"] = normalized
    payoffs = payload.get("payoffs") or {}
    result["payoffs"] = copy.deepcopy(payoffs) if isinstance(payoffs, Mapping) else {}
    return result


def result_map_from_snapshot(snapshot: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    found = {}
    for race in snapshot.get("races") or []:
        race_id = text(race.get("race_id"))
        raw = (race.get("mobile_snapshot") or {}).get("result_file") or {}
        if isinstance(raw, Mapping):
            normalized = normalize_result(raw)
            normalized["race_id"] = normalized.get("race_id") or race_id
            found[race_id] = normalized
    return found


def merge_result_upload(target: dict[str, dict[str, Any]], payload: Mapping[str, Any]) -> None:
    items = payload.get("races") if isinstance(payload.get("races"), list) else [payload]
    for item in items:
        if not isinstance(item, Mapping):
            continue
        normalized = normalize_result(item)
        race_id = text(normalized.get("race_id"))
        if not race_id:
            raise ResultDataError("結果データにrace_idがありません。")
        target[race_id] = normalized


def _payoff_lines(payoffs: Mapping[str, Any]) -> list[str]:
    labels = {"win": "単勝", "place": "複勝", "wide": "ワイド", "quinella": "馬連", "exacta": "馬単", "trio": "三連複", "trifecta": "三連単"}
    lines: list[str] = []
    for key, label in labels.items():
        entries = payoffs.get(key) or []
        if isinstance(entries, Mapping): entries = [entries]
        if not isinstance(entries, list): continue
        for entry in entries:
            if not isinstance(entry, Mapping): continue
            amount = text(entry.get("payout") or entry.get("amount") or entry.get("払戻"))
            combination = text(entry.get("combination") or entry.get("numbers") or entry.get("組合せ"))
            if amount:
                lines.append(f"{label}{(' ' + combination) if combination else ''} {amount}{'' if '円' in amount else '円'}")
    return lines


def result_reply(race: Mapping[str, Any], result: Mapping[str, Any] | None, owner_comment: str = "") -> str:
    if not result or not result.get("results"):
        return "結果データ未取得"
    marks = marked_map(race)
    by_number = {text(row.get("horse_no")): row for row in result.get("results") or [] if isinstance(row, Mapping)}
    lines: list[str] = []
    emojis = {"1": "🏇", "2": "✨", "3": "💥"}
    for mark in ("◎", "○", "▲", "△", "☆"):
        horse = marks.get(mark)
        if not horse: continue
        row = by_number.get(text(horse.get("horse_no")))
        if not row: continue
        rank = text(row.get("rank"))
        if rank in {"1", "2", "3"}:
            lines.append(f"{mark}{text(horse.get('horse_no'))}番 {rank}着{emojis[rank]}")
    lines.extend(_payoff_lines(result.get("payoffs") or {}))
    if text(owner_comment):
        lines.extend(["", text(owner_comment)])
    return "\n".join(lines) if lines else "正式結果は取得済みですが、公開用に表示できる印馬・配当はありません。"

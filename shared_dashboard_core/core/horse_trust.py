# -*- coding: utf-8 -*-
from __future__ import annotations

import re
from typing import Any, Iterable, Mapping

import pandas as pd

from .purchase_conditions import clean_text, horse_no, to_float


HORSE_TRUST_ALIASES: dict[str, tuple[str, ...]] = {
    "horse_no": ("horse_no_eval", "馬番", "horse_no", "horse_number", "馬"),
    "horse_name": ("horse_name_eval", "馬名", "horse_name", "name"),
    "mark": ("mark_eval", "表示印", "display_mark", "最終印", "印", "mark"),
    "group": ("display_group_eval", "グループ", "display_group"),
    "age": ("馬年齢", "性齢", "馬齢", "age"),
    "jockey": ("騎手詳細", "jockey_detail", "騎手", "jockey"),
    "weight": ("斤量詳細", "weight_detail", "斤量", "weight"),
    "training": ("調教評価", "追切評価", "_調教評価記号", "training_grade"),
    "distance": ("距離指数", "distance_index"),
    "course": ("コース指数", "course_index"),
    "star": ("★最高指数", "star_max_index", "★最高"),
    "recent_high": ("近3走最高", "recent3_high"),
    "avg": ("平均指数", "3走平均", "近3走平均", "avg5"),
    "state": ("状態", "form_state", "勢いランク", "momentum_rank", "勢い"),
    "pace": ("枠脚質評価", "枠順×脚質評価", "枠脚質", "脚質勝率", "コース脚質勝率"),
    "supplement": ("補足", "supplement_note", "評価／検討材料", "評価/検討材料", "馬具"),
}


def build_horse_trust_materials(row: Mapping[str, Any], race_type: str = "jra") -> list[dict[str, Any]]:
    """Build display/audit-only trust materials from existing prediction columns.

    The returned values are not fed back into AI score, marks, groups, or ticket
    selection.  They only explain why the already-selected horse is worth
    inspecting in the current race.
    """

    race = "nar" if clean_text(race_type).lower() == "nar" else "jra"
    materials: list[dict[str, Any]] = []

    _append(materials, _index_material(row))
    _append(materials, _jockey_material(row))
    _append(materials, _age_material(row))
    _append(materials, _pace_material(row))
    if race == "jra":
        _append(materials, _training_material(row))
    _append(materials, _index_badge(row, "距離", "distance"))
    _append(materials, _index_badge(row, "コース", "course"))
    _append(materials, _weight_material(row))
    _append(materials, _state_material(row))
    if race == "jra":
        _append(materials, _first_blinker_material(row))

    return materials


def build_horse_trust_summary(
    row: Mapping[str, Any],
    race_type: str = "jra",
    *,
    max_items: int = 5,
) -> str:
    materials = build_horse_trust_materials(row, race_type)
    return " / ".join(item["display"] for item in materials[:max_items])


def build_horse_trust_for_numbers(
    table: Any,
    race_type: str,
    numbers: Iterable[Any],
) -> tuple[dict[str, Any], ...]:
    if table is None or not isinstance(table, pd.DataFrame) or table.empty:
        return ()
    by_no = rows_by_horse_number(table)
    out: list[dict[str, Any]] = []
    for number in _unique_numbers(numbers):
        row = by_no.get(number)
        if row is None:
            continue
        materials = build_horse_trust_materials(row, race_type)
        out.append(
            {
                "horse_no": number,
                "horse_name": _value(row, "horse_name"),
                "mark": _value(row, "mark"),
                "display_group": _value(row, "group"),
                "summary": " / ".join(item["display"] for item in materials[:5]),
                "materials": materials,
            }
        )
    return tuple(out)


def rows_by_horse_number(table: pd.DataFrame) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for _, raw in table.iterrows():
        record = raw.to_dict()
        number = horse_no(_value(record, "horse_no"))
        if not number:
            continue
        merged = rows.setdefault(number, {})
        for key, value in record.items():
            if not _missing(value):
                merged[str(key)] = value
    return rows


def compact_trust_lines(trust_rows: Iterable[Mapping[str, Any]]) -> tuple[str, ...]:
    lines: list[str] = []
    for item in trust_rows:
        name = clean_text(item.get("horse_name"))
        number = horse_no(item.get("horse_no"))
        summary = clean_text(item.get("summary"))
        if not summary:
            continue
        label = " ".join(part for part in [number, name] if part)
        lines.append(f"{label}：{summary}" if label else summary)
    return tuple(lines)


def trust_rows_to_audit_text(trust_rows: Iterable[Mapping[str, Any]]) -> str:
    return " / ".join(line for line in compact_trust_lines(trust_rows) if line)


def _append(materials: list[dict[str, Any]], item: dict[str, Any] | None) -> None:
    if item is not None and clean_text(item.get("display")):
        materials.append(item)


def _index_material(row: Mapping[str, Any]) -> dict[str, Any] | None:
    values = {
        "近3走最高": _number(row, "recent_high"),
        "★最高": _number(row, "star"),
        "平均": _number(row, "avg"),
        "距離": _number(row, "distance"),
        "コース": _number(row, "course"),
    }
    valid = {key: value for key, value in values.items() if value is not None}
    if not valid:
        return None
    best = max(valid.values())
    if best >= 65:
        grade = "◎"
    elif best >= 50:
        grade = "○"
    elif best >= 35:
        grade = "△"
    else:
        grade = "注"
    return {
        "key": "time_index",
        "label": "タイム指数",
        "grade": grade,
        "display": f"指数{grade}",
        "detail": " / ".join(f"{key}{_format_number(value)}" for key, value in valid.items()),
        "source_columns": list(valid.keys()),
    }


def _jockey_material(row: Mapping[str, Any]) -> dict[str, Any] | None:
    jockey = clean_text(_value(row, "jockey"))
    if not jockey:
        return None
    if "継続" in jockey:
        grade = "○"
        display = "騎手○"
    elif any(token in jockey for token in ("乗替", "乗り替", "替")):
        grade = "△"
        display = "騎手△"
    else:
        grade = "-"
        display = "騎手-"
    return {
        "key": "jockey",
        "label": "騎手",
        "grade": grade,
        "display": display,
        "detail": jockey,
        "source_columns": list(HORSE_TRUST_ALIASES["jockey"]),
    }


def _age_material(row: Mapping[str, Any]) -> dict[str, Any] | None:
    text = clean_text(_value(row, "age"))
    if not text:
        return None
    match = re.search(r"(\d{1,2})", text)
    age = int(match.group(1)) if match else None
    if age is None:
        return None
    if age == 3:
        display, grade = "3歳+", "+"
    elif age == 4:
        display, grade = "4歳±", "0"
    elif age in {5, 6}:
        display, grade = "年齢±0", "0"
    elif age >= 7:
        display, grade = "年齢△", "△"
    else:
        display, grade = f"{age}歳", "-"
    return {
        "key": "age",
        "label": "年齢",
        "grade": grade,
        "display": display,
        "detail": text,
        "source_columns": list(HORSE_TRUST_ALIASES["age"]),
    }


def _pace_material(row: Mapping[str, Any]) -> dict[str, Any] | None:
    raw = _value(row, "pace")
    number = to_float(raw)
    text = clean_text(raw)
    if number is None and not text:
        return None
    if number is not None:
        if number >= 18:
            grade = "◎"
        elif number >= 10:
            grade = "○"
        else:
            grade = "△"
    elif any(token in text for token in ("◎", "有利", "好", "高")):
        grade = "◎"
    elif any(token in text for token in ("○", "中", "注意")):
        grade = "○"
    else:
        grade = "△"
    return {
        "key": "frame_style",
        "label": "枠順×脚質",
        "grade": grade,
        "display": f"枠脚質{grade}",
        "detail": text or _format_number(number),
        "source_columns": list(HORSE_TRUST_ALIASES["pace"]),
    }


def _training_material(row: Mapping[str, Any]) -> dict[str, Any] | None:
    text = clean_text(_value(row, "training"))
    if not text:
        return None
    upper = text.upper()
    if upper.startswith("A") or "◎" in text:
        grade = "◎"
    elif upper.startswith("B") or "○" in text:
        grade = "○"
    elif upper.startswith("C") or "△" in text:
        grade = "△"
    elif upper.startswith("D") or "×" in text:
        grade = "×"
    else:
        grade = "-"
    return {
        "key": "training",
        "label": "調教",
        "grade": grade,
        "display": f"調教{grade}",
        "detail": text,
        "source_columns": list(HORSE_TRUST_ALIASES["training"]),
    }


def _index_badge(row: Mapping[str, Any], label: str, key: str) -> dict[str, Any] | None:
    number = _number(row, key)
    if number is None:
        return None
    if number >= 60:
        grade = "◎"
    elif number >= 45:
        grade = "○"
    else:
        grade = "△"
    return {
        "key": key,
        "label": label,
        "grade": grade,
        "display": f"{label}{grade}",
        "detail": _format_number(number),
        "source_columns": list(HORSE_TRUST_ALIASES[key]),
    }


def _weight_material(row: Mapping[str, Any]) -> dict[str, Any] | None:
    text = clean_text(_value(row, "weight"))
    if not text:
        return None
    if "前走データなし" in text:
        display, grade = "斤量-", "-"
    elif any(token in text for token in ("+", "＋", "増")):
        display, grade = "斤量△", "△"
    elif any(token in text for token in ("-", "－", "減")):
        display, grade = "斤量○", "○"
    elif "±" in text or "0.0" in text:
        display, grade = "斤量±", "0"
    else:
        display, grade = "斤量-", "-"
    return {
        "key": "weight",
        "label": "斤量",
        "grade": grade,
        "display": display,
        "detail": text,
        "source_columns": list(HORSE_TRUST_ALIASES["weight"]),
    }


def _state_material(row: Mapping[str, Any]) -> dict[str, Any] | None:
    text = clean_text(_value(row, "state"))
    if not text:
        return None
    if any(token in text for token in ("S", "A", "上昇", "良化", "反発", "安定")):
        display, grade = "状態○", "○"
    elif any(token in text for token in ("D", "下降", "急落", "不安")):
        display, grade = "状態△", "△"
    else:
        display, grade = "状態-", "-"
    return {
        "key": "state",
        "label": "状態",
        "grade": grade,
        "display": display,
        "detail": text,
        "source_columns": list(HORSE_TRUST_ALIASES["state"]),
    }


def _first_blinker_material(row: Mapping[str, Any]) -> dict[str, Any] | None:
    text = " ".join(clean_text(_first_existing(row, aliases)) for aliases in [
        HORSE_TRUST_ALIASES["supplement"],
        ("厩舎コメント", "新聞コメント", "stable_comment"),
    ])
    if not text:
        return None
    if any(pattern in text for pattern in ("初ブリンカー", "初Ｂ", "初B", "B初", "Ｂ初", "ブリンカー初", "初めてブリンカー")):
        return {
            "key": "first_blinker",
            "label": "初ブリンカー",
            "grade": "info",
            "display": "初B",
            "detail": text,
            "source_columns": list(HORSE_TRUST_ALIASES["supplement"]),
        }
    return None


def _value(row: Mapping[str, Any], key: str) -> Any:
    return _first_existing(row, HORSE_TRUST_ALIASES.get(key, (key,)))


def _first_existing(row: Mapping[str, Any], names: Iterable[str]) -> Any:
    for name in names:
        if name not in row:
            continue
        value = row.get(name)
        if not _missing(value):
            return value
    return None


def _number(row: Mapping[str, Any], key: str) -> float | None:
    return to_float(_value(row, key))


def _missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return clean_text(value).lower() in {"", "-", "—", "nan", "none", "null", "データなし", "未取得"}


def _format_number(value: Any) -> str:
    number = to_float(value)
    if number is None:
        return clean_text(value)
    return f"{number:.1f}".rstrip("0").rstrip(".")


def _unique_numbers(numbers: Iterable[Any]) -> list[str]:
    out: list[str] = []
    for value in numbers:
        number = horse_no(value)
        if number and number not in out:
            out.append(number)
    return out

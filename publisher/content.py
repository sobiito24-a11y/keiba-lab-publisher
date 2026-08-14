from __future__ import annotations

import copy
import re
from collections import defaultdict
from typing import Any, Iterable, Mapping

from .jockey import jockey_display


MARK_ORDER = ("◎", "○", "▲", "△", "☆")
BANNED_WORDS = ("絶対", "鉄板", "必勝", "確実")


def text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def race_number(race: Mapping[str, Any]) -> str:
    value = text(race.get("race_number"))
    if value:
        return value if value.upper().endswith("R") else f"{value}R"
    race_id = text(race.get("race_id"))
    return f"{int(race_id[-2:])}R" if race_id[-2:].isdigit() else ""


def is_debut_race(race: Mapping[str, Any]) -> bool:
    """Exclude only when saved text explicitly identifies a debut race."""

    candidates = [race.get("race_name"), race.get("class"), race.get("race_class")]
    info = race.get("mobile_snapshot", {}).get("race_info", {}) if isinstance(race.get("mobile_snapshot"), Mapping) else {}
    candidates.extend([info.get("race_name"), info.get("class")])
    return any(re.search(r"新馬|メイクデビュー", text(value)) for value in candidates)


def group_by_venue(snapshot: Mapping[str, Any], *, exclude_debut: bool = True) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in snapshot.get("races") or []:
        race = copy.deepcopy(source)
        if exclude_debut and is_debut_race(race):
            continue
        grouped[text(race.get("venue")) or "会場不明"].append(race)
    for races in grouped.values():
        races.sort(key=lambda item: _number_value(race_number(item)))
    return dict(grouped)


def _number_value(value: Any) -> int:
    match = re.search(r"\d+", text(value))
    return int(match.group()) if match else 999


def marked_horses(race: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    horses = list(race.get("horses") or [])
    order = {mark: index for index, mark in enumerate(MARK_ORDER)}
    selected = [horse for horse in horses if text(horse.get("mark")) in order]
    return sorted(selected, key=lambda horse: (order[text(horse.get("mark"))], _number_value(horse.get("horse_no"))))


def value_horses(race: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [horse for horse in race.get("horses") or [] if _truthy_value(horse.get("value_signal"))]


def _truthy_value(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return text(value).lower() in {"1", "true", "yes", "妙味あり", "あり", "○"}


def _horse_label(horse: Mapping[str, Any]) -> str:
    return f"{text(horse.get('horse_no'))} {text(horse.get('horse_name'))}".strip()


def _material_items(horse: Mapping[str, Any], key: str) -> list[str]:
    raw = horse.get(key)
    if isinstance(raw, str):
        raw = [raw]
    if not isinstance(raw, Iterable) or isinstance(raw, Mapping):
        return []
    seen: set[str] = set()
    result: list[str] = []
    for item in raw:
        clean = re.sub(r"^[＋+○△▲☆\-－]\s*", "", text(item))
        if clean and clean not in seen:
            seen.add(clean)
            result.append(clean)
    return result


def _top_horse(race: Mapping[str, Any], rank_key: str) -> Mapping[str, Any] | None:
    candidates = [horse for horse in race.get("horses") or [] if _number_value(horse.get(rank_key)) < 999]
    return min(candidates, key=lambda horse: _number_value(horse.get(rank_key)), default=None)


def _safe_sentence(value: str) -> str:
    clean = "\n".join(" ".join(line.split()) for line in text(value).splitlines())
    for word in BANNED_WORDS:
        clean = clean.replace(word, "上位評価")
    return clean


def short_commentary(race: Mapping[str, Any]) -> str:
    ability = _top_horse(race, "ability_rank")
    current = _top_horse(race, "current_evaluation_rank")
    sentences: list[str] = []
    if ability:
        sentences.append(f"能力面では{_horse_label(ability)}が上位。")
    if current:
        if not ability or text(current.get("horse_no")) != text(ability.get("horse_no")):
            sentences.append(f"今回評価では{text(current.get('mark')) or '上位'} {_horse_label(current)}を中心に見る。")
        else:
            sentences.append(f"今回評価でも{_horse_label(current)}を中心に見る。")
        materials = _material_items(current, "plus_materials")[:2]
        if materials:
            sentences.append("、".join(materials) + "をプラス材料としている。")
        elif text(current.get("development")) or text(current.get("course_material")):
            facts = [text(current.get("development")), text(current.get("course_material"))]
            sentences.append("、".join(item for item in facts if item) + "を比較材料とした。")
    if not sentences:
        saved = text((race.get("prediction_result") or {}).get("ai_race_review"))
        body = re.sub(r"^【AIレース考察】\s*", "", saved)
        first = re.split(r"(?<=。)", body)[0] if body else ""
        sentences.append(first or "保存済みの印と今回評価を比較したいレース。")
    return _safe_sentence("".join(sentences))


def _mark_lines(race: Mapping[str, Any]) -> list[str]:
    return [f"{text(horse.get('mark'))} {text(horse.get('horse_no'))} {text(horse.get('horse_name'))}" for horse in marked_horses(race)]


def note_race_section(race: Mapping[str, Any]) -> str:
    lines = [f"### {text(race.get('venue'))}{race_number(race)}"]
    marks = _mark_lines(race)
    lines.extend(["", *(marks or ["印：保存データ内に該当なし"])])
    lines.extend(["", "【AIレース考察】", short_commentary(race)])
    current = _top_horse(race, "current_evaluation_rank")
    if current:
        details = [f"{text(current.get('mark'))} {_horse_label(current)}"]
        facts = []
        if text(current.get("ability_band")):
            facts.append(f"能力{text(current.get('ability_band'))}")
        if text(current.get("ability_value")):
            facts.append(f"能力値{text(current.get('ability_value'))}")
        if text(current.get("current_evaluation_rank")):
            facts.append(f"今回評価{text(current.get('current_evaluation_rank'))}位")
        details.append(" / ".join(facts))
        details.extend(f"＋{item}" for item in _material_items(current, "plus_materials")[:3])
        jockey = jockey_display(current)
        if jockey.text and jockey.relationship != "unknown":
            details.append(f"騎手：{jockey.text}")
        lines.extend(["", "【注目馬】", *[item for item in details if item]])
    values = value_horses(race)
    if values:
        lines.extend(["", "【妙味あり】", "、".join(_horse_label(horse) for horse in values)])
    return "\n".join(lines).rstrip()


def note_article(venue: str, races: Iterable[Mapping[str, Any]], race_date: str) -> str:
    month_day = _month_day(race_date)
    lines = [
        f"【{month_day} {venue}｜KEIBA LAB AI全レース予想】",
        "",
        "自作競馬AI KEIBA LABによる公開検証予想です。",
        "能力・近走・展開・コース・調教などを全頭比較。",
        "予想はレース前に保存し、結果に関係なく検証しています。",
    ]
    for race in races:
        lines.extend(["", note_race_section(race)])
    return "\n".join(lines).rstrip() + "\n"


def _month_day(value: str) -> str:
    match = re.search(r"\d{4}[-/]?(\d{1,2})[-/]?(\d{1,2})", text(value))
    return f"{int(match.group(1))}/{int(match.group(2))}" if match else text(value)


_CIRCLED = {1: "①", 2: "②", 3: "③", 4: "④", 5: "⑤", 6: "⑥", 7: "⑦", 8: "⑧", 9: "⑨", 10: "⑩", 11: "⑪", 12: "⑫", 13: "⑬", 14: "⑭", 15: "⑮", 16: "⑯", 17: "⑰", 18: "⑱"}


def _x_horse(horse: Mapping[str, Any]) -> str:
    no = _number_value(horse.get("horse_no"))
    return f"{_CIRCLED.get(no, text(horse.get('horse_no')))}{text(horse.get('horse_name'))}"


def x_post(race: Mapping[str, Any], note_url: str = "") -> str:
    venue, rno = text(race.get("venue")), race_number(race)
    lines = [f"【{venue}{rno}｜KEIBA LAB AI予想】", ""]
    marked = marked_horses(race)
    if marked:
        lines.extend(f"{text(horse.get('mark'))} {_x_horse(horse)}" for horse in marked)
    else:
        lines.append("保存データ内に印情報なし")
    top = _top_horse(race, "current_evaluation_rank")
    if top:
        facts = []
        if text(top.get("ability_band")):
            facts.append(f"能力{text(top.get('ability_band'))}")
        if text(top.get("current_evaluation_rank")):
            facts.append(f"今回評価{text(top.get('current_evaluation_rank'))}位")
        materials = _material_items(top, "plus_materials")[:1]
        description = "・".join(facts + materials)
        if description:
            lines.extend(["", f"{text(top.get('mark')) or '注目'}{_x_horse(top)}は{description}。"])
    if note_url:
        lines.extend(["", f"{venue}全レースのAI予想・考察はこちら👇", note_url])
    lines.extend(["", f"#{venue}{rno} #競馬予想 #AI競馬予想"])
    return _safe_sentence("\n".join(lines)).strip()

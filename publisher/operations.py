from __future__ import annotations

from typing import Any, Mapping

from .content import marked_map, race_number, text, value_horses

OPERATION_MODES = ("🏢 通常平日", "🏃 忙しい日", "🌴 休日")


def publication_candidate(races: list[Mapping[str, Any]]) -> dict[str, Any] | None:
    """Suggest interesting public content using saved facts only; never predict."""
    candidates: list[tuple[int, int, Mapping[str, Any], list[str]]] = []
    for index, race in enumerate(races):
        marks = marked_map(race)
        honmei = marks.get("◎")
        if not honmei:
            continue
        reasons: list[str] = []
        score = 0
        if str(honmei.get("current_evaluation_rank") or "") == "1":
            score += 3
            reasons.append("◎が保存済み総合評価1位")
        if str(honmei.get("ability_rank") or "") == "1":
            score += 2
            reasons.append("◎が能力評価1位")
        elif honmei.get("ability_rank") not in (None, ""):
            score += 2
            reasons.append("能力1位と◎が異なる比較型")
        materials = []
        for key in ("decision_material", "plus_materials", "development", "course_material", "training_short"):
            raw = honmei.get(key)
            materials.extend(raw if isinstance(raw, list) else ([raw] if raw else []))
        material_count = len({text(v) for v in materials if text(v) not in {"-", "対象外", "データなし"}})
        score += min(material_count, 4)
        if material_count >= 3:
            reasons.append("考察材料が比較的豊富")
        if value_horses(race):
            score += 2
            reasons.append("保存済みの妙味材料あり")
        candidates.append((score, -index, race, reasons))
    if not candidates:
        return None
    score, _, race, reasons = max(candidates, key=lambda item: (item[0], item[1]))
    return {
        "race_id": text(race.get("race_id")),
        "label": f"{text(race.get('venue'))}{race_number(race)}",
        "score": score,
        "reason": "、".join(reasons[:2]) + "という特徴があり、公開コンテンツとして組み立てやすいレースです。",
    }


def visible_x_races(mode: str, races: list[Mapping[str, Any]], free_race_id: str = "") -> list[Mapping[str, Any]]:
    if mode == OPERATION_MODES[0]:
        return list(races)
    return [race for race in races if text(race.get("race_id")) == text(free_race_id)]

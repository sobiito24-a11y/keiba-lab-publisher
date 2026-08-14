from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class StarMaxResult:
    value: float | None
    race: str
    venue: str
    distance: int | None
    surface: str
    turn: str
    match_level: str
    source: str

    @property
    def condition(self) -> str:
        parts = [self.venue]
        course = ""
        if self.surface:
            course += self.surface
        if self.distance is not None:
            course += f"{self.distance}m"
        if self.turn:
            course += self.turn
        if course:
            parts.append(course)
        return "".join(parts)


def star_match_level(current: dict[str, Any], past: dict[str, Any]) -> str:
    current_venue = _text(current.get("racecourse") or current.get("venue"))
    past_venue = _text(past.get("racecourse") or past.get("venue"))
    current_distance = _int(current.get("distance"))
    past_distance = _int(past.get("distance"))
    if not current_venue or not past_venue or current_venue != past_venue:
        return "none"
    if current_distance is None or past_distance is None or current_distance != past_distance:
        return "none"

    level = "venue_distance"
    current_surface = _surface(current.get("surface"))
    past_surface = _surface(past.get("surface"))
    if current_surface and past_surface:
        if current_surface != past_surface:
            return "none"
        level = "venue_distance_surface"

    current_turn = _turn(current.get("direction") or current.get("turn"))
    past_turn = _turn(past.get("direction") or past.get("turn"))
    if current_turn and past_turn:
        if current_turn != past_turn:
            return "none"
        if level == "venue_distance_surface":
            level = "venue_distance_surface_turn"

    return level


def build_star_max_result(current: dict[str, Any], runs: list[dict[str, Any]]) -> StarMaxResult:
    matches: list[tuple[float, int, str, dict[str, Any]]] = []
    for order, run in enumerate(runs):
        value = _float(run.get("value"))
        if value is None:
            continue
        level = star_match_level(current, run)
        if level == "none":
            continue
        matches.append((value, -order, level, run))

    if not matches:
        return StarMaxResult(None, "", "", None, "", "", "none", "missing")

    value, _, level, run = max(matches, key=lambda item: (item[0], item[1]))
    distance = _int(run.get("distance"))
    return StarMaxResult(
        value=value,
        race=_text(run.get("label") or run.get("race_label") or run.get("race")),
        venue=_text(run.get("racecourse") or run.get("venue")),
        distance=distance,
        surface=_surface(run.get("surface")),
        turn=_turn(run.get("direction") or run.get("turn")),
        match_level=level,
        source="recent3_same_condition",
    )


def _text(value: Any) -> str:
    if value is None:
        return ""
    text = str(value).strip()
    if text.lower() in {"", "none", "nan", "<na>", "nat", "-"}:
        return ""
    return text


def _int(value: Any) -> int | None:
    if value is None:
        return None
    try:
        return int(float(str(value).replace(",", "").strip()))
    except Exception:
        return None


def _float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(str(value).replace(",", "").strip())
    except Exception:
        return None


def _surface(value: Any) -> str:
    text = _text(value)
    if "芝" in text:
        return "芝"
    if "ダ" in text:
        return "ダ"
    return text


def _turn(value: Any) -> str:
    text = _text(value)
    if "直" in text:
        return "直"
    if "左" in text:
        return "左"
    if "右" in text:
        return "右"
    return text

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Mapping


SHOW_PUBLIC_SS_DEFAULT = False
INTERNAL_RANKS = ("SS", "S", "A", "対象外")


def _text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _number(value: Any) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _saved_materials(horse: Mapping[str, Any]) -> tuple[set[str], set[str]]:
    """Return saved facts, excluding stale explanatory mark/rank labels."""
    plus = {_text(v) for v in horse.get("plus_materials") or []}
    minus = {_text(v) for v in horse.get("minus_materials") or []}
    plus = {v for v in plus if v and not (v.startswith("印") or v.startswith("今回"))}
    minus = {v for v in minus if v and not (v.startswith("印") or v.startswith("今回"))}
    return plus, minus


def _jockey_flags(horse: Mapping[str, Any], plus: set[str], minus: set[str]) -> tuple[bool, bool]:
    """The structured field wins when saved labels conflict."""
    change = _text(horse.get("jockey_change"))
    if change in {"継続", "乗替"}:
        return change == "継続", change == "乗替"
    return "継続騎乗" in plus, "乗替" in minus


def _training_ab(horse: Mapping[str, Any], plus: set[str]) -> bool:
    short = _text(horse.get("training_short"))
    return "調教A" in plus or "調教B" in plus or short.startswith("A ") or short.startswith("B ")


def _training_d(horse: Mapping[str, Any], minus: set[str]) -> bool:
    return "調教D" in minus or _text(horse.get("training_short")).startswith("D ")


def _honmei(race: Mapping[str, Any]) -> Mapping[str, Any] | None:
    return next((h for h in race.get("horses") or [] if _text(h.get("mark")) == "◎"), None)


def ability_gap(race: Mapping[str, Any]) -> float | None:
    values = sorted(
        (value for value in (_float(h.get("ability_value")) for h in race.get("horses") or []) if value is not None),
        reverse=True,
    )
    return round(values[0] - values[1], 1) if len(values) >= 2 else None


@dataclass(frozen=True)
class ConfidenceAssessment:
    race_id: str
    internal_confidence_rank: str
    public_confidence_rank: str
    confidence_reason: str
    confidence_positive_materials: tuple[str, ...]
    confidence_warning_materials: tuple[str, ...]
    ability_gap: float | None
    mode: str
    score: int | None = None

    def to_state(self) -> dict[str, Any]:
        value = asdict(self)
        value["confidence_positive_materials"] = list(self.confidence_positive_materials)
        value["confidence_warning_materials"] = list(self.confidence_warning_materials)
        return value


def _warning_materials(horse: Mapping[str, Any], minus: set[str], rider_change: bool) -> list[str]:
    warnings: list[str] = []
    for item in horse.get("minus_materials") or []:
        clean = _text(item)
        if clean == "乗替":
            clean = "乗り替わり"
        if clean and not (clean.startswith("印") or clean.startswith("今回")) and clean not in warnings:
            warnings.append(clean)
    if rider_change and not any("乗替" in item or "乗り替" in item for item in warnings):
        warnings.append("乗り替わり")
    development = _text(horse.get("development"))
    if development and any(word in development for word in ("厳", "懸念", "展開待ち", "届かない", "不利")) and development not in warnings:
        warnings.append(development)
    state = _text(horse.get("state"))
    if state in {"弱含み", "下降", "判定保留"}:
        label = f"状態は{state}"
        if label not in warnings:
            warnings.append(label)
    return warnings


def assess_confidence(race: Mapping[str, Any], *, show_public_ss: bool = SHOW_PUBLIC_SS_DEFAULT) -> ConfidenceAssessment:
    horse = _honmei(race)
    race_id = _text(race.get("race_id"))
    mode = _text(race.get("race_mode") or (race.get("prediction_result") or {}).get("race_mode")).lower()
    if horse is None:
        return ConfidenceAssessment(race_id, "対象外", "", "最終◎が保存されていません。", (), (), None, mode)

    plus, minus = _saved_materials(horse)
    continued, rider_change = _jockey_flags(horse, plus, minus)
    gap = ability_gap(race)
    ability_rank = _number(horse.get("ability_rank"))
    evaluation_rank = _number(horse.get("current_evaluation_rank"))
    aligned = ability_rank == 1 and evaluation_rank == 1
    distance = "距離実績◎" in plus
    course = "コース実績◎" in plus
    recent_up = "近走上昇" in plus
    recent_down = "近走下降" in minus
    training_ab = _training_ab(horse, plus)
    training_d = _training_d(horse, minus)

    positives: list[str] = []
    if aligned: positives.append("◎が能力1位・今回評価1位")
    if gap is not None: positives.append(f"能力1位と2位の差{gap:.1f}")
    if distance: positives.append("距離実績◎")
    if course: positives.append("コース実績◎")
    if recent_up: positives.append("近走上昇")
    if mode == "jra" and training_ab: positives.append("調教A/B")
    if continued: positives.append("継続騎乗")
    warnings = _warning_materials(horse, minus, rider_change)
    score: int | None = None

    if mode == "nar":
        gap_points = 3 if gap is not None and gap >= 10 else 2 if gap is not None and gap >= 5 else 1 if gap is not None and gap >= 3 else 0
        score = gap_points + sum((distance, course, recent_up, continued)) - int(rider_change)
        if score:
            positives.append(f"NAR評価点{score}")
        internal = "対象外"
        if aligned and score >= 5 and not minus:
            internal = "SS"
        elif aligned and gap is not None and gap >= 3 and score >= 4 and not recent_down and "距離実績△" not in minus and "コース実績△" not in minus:
            internal = "S"
        elif (evaluation_rank == 1 and ability_rank is not None and ability_rank <= 2) or (ability_rank == 1 and evaluation_rank is not None and evaluation_rank <= 2):
            internal = "A"
    else:
        axes = sum((distance, course, recent_up, training_ab, continued))
        internal = "対象外"
        if aligned and gap is not None and gap >= 5 and distance and course and recent_up and training_ab and continued and not minus:
            internal = "SS"
        elif aligned and gap is not None and gap >= 3 and axes >= 4 and not recent_down and not training_d:
            internal = "S"
        elif (evaluation_rank == 1 and ability_rank is not None and ability_rank <= 2) or (ability_rank == 1 and evaluation_rank is not None and evaluation_rank <= 2):
            internal = "A"

    public = "SS" if internal == "SS" and show_public_ss else "S" if internal in {"SS", "S"} else "A" if internal == "A" else ""
    if internal in {"SS", "S"}:
        reason = "能力順位・今回評価と、保存済みの条件材料を固定基準で確認した注目度です。"
    elif internal == "A":
        reason = "◎が能力順位または今回評価の上位に該当するため、注目度Aです。"
    else:
        reason = "公開注目度の固定条件には該当しません。"
    return ConfidenceAssessment(race_id, internal, public, reason, tuple(positives), tuple(warnings), gap, mode, score)


def assess_snapshot(races: Iterable[Mapping[str, Any]], *, show_public_ss: bool = SHOW_PUBLIC_SS_DEFAULT) -> dict[str, dict[str, Any]]:
    return {assessment.race_id: assessment.to_state() for assessment in (assess_confidence(r, show_public_ss=show_public_ss) for r in races)}

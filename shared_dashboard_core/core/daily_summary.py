from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any, Iterable, Literal, Mapping

import pandas as pd

from .models import PredictionResult


DecisionLabel = Literal["BUY", "HOLD", "SKIP"]
NAR_RACE_TYPE = "nar"
DEFAULT_STRATEGY_PATH = Path(__file__).resolve().parents[1] / "assets" / "analysis" / "nar_strategy_selection.json"

NAR_VENUES = {
    "30": "門別",
    "31": "盛岡",
    "35": "盛岡",
    "36": "水沢",
    "42": "浦和",
    "43": "船橋",
    "44": "大井",
    "45": "川崎",
    "46": "金沢",
    "47": "笠松",
    "48": "名古屋",
    "50": "園田",
    "51": "姫路",
    "54": "高知",
    "55": "佐賀",
}
NAR_VENUE_ORDER = (
    "門別", "盛岡", "水沢", "浦和", "船橋", "大井", "川崎",
    "金沢", "笠松", "名古屋", "園田", "姫路", "高知", "佐賀",
)


@dataclass(frozen=True)
class DailyHorse:
    number: int | str
    name: str
    group: str = ""
    role: str = ""
    odds: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "number": self.number,
            "name": self.name,
            "group": self.group,
            "role": self.role,
            "odds": self.odds,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DailyHorse":
        return cls(
            number=data.get("number", ""),
            name=str(data.get("name") or ""),
            group=str(data.get("group") or ""),
            role=str(data.get("role") or ""),
            odds=_safe_float(data.get("odds")),
        )


@dataclass(frozen=True)
class DailyInvestmentDecision:
    race_id: str
    race_name: str
    race_title: str
    venue: str
    race_number: str
    post_time: str
    ticket: str
    combinations: tuple[str, ...]
    decision: DecisionLabel
    strategy_score: int
    reproducibility: str
    roi: float | None
    hit_rate: float | None
    sample_size: int
    investment: int
    points: int
    horses: tuple[DailyHorse, ...] = ()
    confidence: str = "★☆☆☆☆"
    reason: str = ""
    validation_label: str = ""
    validation_races: int | None = None
    detail_path: str = ""
    race_type: str = NAR_RACE_TYPE

    def to_dict(self) -> dict[str, Any]:
        return {
            "race_type": self.race_type,
            "race_id": self.race_id,
            "race_name": self.race_name,
            "race_title": self.race_title,
            "venue": self.venue,
            "race_number": self.race_number,
            "post_time": self.post_time,
            "ticket": self.ticket,
            "combinations": list(self.combinations),
            "decision": self.decision,
            "score": self.strategy_score,
            "strategy_score": self.strategy_score,
            "reproducibility": self.reproducibility,
            "roi": self.roi,
            "hit_rate": self.hit_rate,
            "sample_size": self.sample_size,
            "investment": self.investment,
            "points": self.points,
            "confidence": self.confidence,
            "reason": self.reason,
            "validation_label": self.validation_label,
            "validation_races": self.validation_races,
            "detail_path": self.detail_path,
            "horses": [horse.to_dict() for horse in self.horses],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DailyInvestmentDecision":
        _require_nar_race_type(data, "Daily decision")
        decision = str(data.get("decision") or "SKIP").upper()
        if decision not in {"BUY", "HOLD", "SKIP"}:
            decision = "SKIP"
        validation_races = _safe_int(data.get("validation_races"))
        return cls(
            race_id=str(data.get("race_id") or ""),
            race_name=str(data.get("race_name") or ""),
            race_title=str(data.get("race_title") or ""),
            venue=str(data.get("venue") or ""),
            race_number=str(data.get("race_number") or ""),
            post_time=str(data.get("post_time") or ""),
            ticket=str(data.get("ticket") or ""),
            combinations=tuple(str(item) for item in data.get("combinations", [])),
            decision=decision,  # type: ignore[arg-type]
            strategy_score=int(_safe_float(data.get("strategy_score", data.get("score"))) or 0),
            reproducibility=str(data.get("reproducibility") or ""),
            roi=_safe_float(data.get("roi")),
            hit_rate=_safe_float(data.get("hit_rate")),
            sample_size=int(_safe_float(data.get("sample_size")) or 0),
            investment=int(_safe_float(data.get("investment")) or 0),
            points=int(_safe_float(data.get("points")) or 0),
            horses=tuple(DailyHorse.from_dict(item) for item in data.get("horses", []) if isinstance(item, Mapping)),
            confidence=str(data.get("confidence") or "★☆☆☆☆"),
            reason=str(data.get("reason") or ""),
            validation_label=str(data.get("validation_label") or ""),
            validation_races=validation_races,
            detail_path=str(data.get("detail_path") or ""),
        )


@dataclass(frozen=True)
class DailySummary:
    date: str
    buy: tuple[DailyInvestmentDecision, ...] = ()
    hold: tuple[DailyInvestmentDecision, ...] = ()
    skip: tuple[DailyInvestmentDecision, ...] = ()
    strategy_selection: dict[str, Any] = field(default_factory=dict)
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    errors: tuple[dict[str, str], ...] = ()
    race_type: str = NAR_RACE_TYPE

    @property
    def all_decisions(self) -> tuple[DailyInvestmentDecision, ...]:
        return self.buy + self.hold + self.skip

    @property
    def venues(self) -> tuple[str, ...]:
        venues = {item.venue for item in self.all_decisions if item.venue}
        order = {venue: index for index, venue in enumerate(NAR_VENUE_ORDER)}
        return tuple(sorted(venues, key=lambda venue: (order.get(venue, len(order)), venue)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "date": self.date,
            "race_type": self.race_type,
            "generated_at": self.generated_at,
            "strategy_selection": _json_ready(self.strategy_selection),
            "counts": {
                "buy": len(self.buy),
                "hold": len(self.hold),
                "skip": len(self.skip),
                "errors": len(self.errors),
            },
            "venues": list(self.venues),
            "buy": [item.to_dict() for item in self.buy],
            "hold": [item.to_dict() for item in self.hold],
            "skip": [item.to_dict() for item in self.skip],
            "errors": [dict(item) for item in self.errors],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "DailySummary":
        _require_nar_race_type(data, "Daily summary")
        strategy_selection = dict(data.get("strategy_selection") or {})
        if strategy_selection:
            _require_nar_race_type(strategy_selection, "Daily strategy metadata")
        return cls(
            date=str(data.get("date") or ""),
            generated_at=str(data.get("generated_at") or ""),
            strategy_selection=strategy_selection,
            buy=tuple(DailyInvestmentDecision.from_dict(item) for item in data.get("buy", []) if isinstance(item, Mapping)),
            hold=tuple(DailyInvestmentDecision.from_dict(item) for item in data.get("hold", []) if isinstance(item, Mapping)),
            skip=tuple(DailyInvestmentDecision.from_dict(item) for item in data.get("skip", []) if isinstance(item, Mapping)),
            errors=tuple(dict(item) for item in data.get("errors", []) if isinstance(item, Mapping)),
        )


def load_nar_strategy_selection(path: str | Path = DEFAULT_STRATEGY_PATH) -> dict[str, Any]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("nar_strategy_selection.json must contain a JSON object.")
    _require_nar_race_type(data, "Strategy selection")
    mapping = data.get("decision_mapping")
    if not isinstance(mapping, Mapping):
        raise ValueError("NAR strategy selection requires decision_mapping.")
    invalid = {str(value) for value in mapping.values()} - {"BUY", "HOLD", "SKIP"}
    if invalid:
        raise ValueError("NAR strategy selection has an invalid decision label.")
    return dict(data)


def build_daily_investment_decision(
    result: PredictionResult,
    *,
    strategy_selection: Mapping[str, Any],
    race_id: str = "",
    detail_path: str = "",
) -> DailyInvestmentDecision:
    if result.race_mode != NAR_RACE_TYPE:
        raise ValueError("Keiba AI Daily accepts NAR PredictionResult only.")
    _require_nar_race_type(strategy_selection, "Strategy selection")

    resolved_race_id = race_id or _race_id_from_result(result)
    venue, race_number, post_time, display_name, race_title = _race_metadata(result, resolved_race_id)
    frame = _ticket_input_frame(result)
    if frame.empty:
        return _skip_decision(
            resolved_race_id, display_name, race_title, venue, race_number, post_time,
            detail_path, strategy_selection, "馬データなし",
        )

    ticket_logic = _load_nar_ticket_logic()
    venue_profile = getattr(ticket_logic, "VENUE_PROFILES", {}).get(venue)
    confidence_summary = ticket_logic.build_ai_confidence_summary(
        frame,
        result.race_info,
        venue,
        venue_profile,
        race_type="nar",
    )
    candidate_rows = ticket_logic._ticket_candidate_rows(frame, confidence_summary=confidence_summary, race_type="nar")
    top_rows = ticket_logic._ticket_top_rows(candidate_rows, confidence_summary=confidence_summary)
    if top_rows:
        selected = top_rows[0]
    elif candidate_rows:
        selected = sorted(candidate_rows, key=ticket_logic._ticket_ranking_sort_key, reverse=True)[0]
    else:
        return _skip_decision(
            resolved_race_id, display_name, race_title, venue, race_number, post_time,
            detail_path, strategy_selection, "有力買い目なし",
        )

    judgement_code = ticket_logic._ticket_judgement_code(selected)
    decision = _decision_for_code(strategy_selection, judgement_code)
    score = int(_safe_float(selected.get("_score", selected.get("買い目スコア"))) or 0)
    roi = _safe_float(selected.get("_roi"))
    hit_rate = _safe_float(selected.get("_hit_rate"))
    sample_size = int(_safe_float(selected.get("_n")) or 0)
    reproducibility = _first_text(selected, ["再現性", "_reproducibility", "reproducibility"])
    horses = tuple(_candidate_horses(selected, frame))
    combinations = _candidate_combinations(selected, horses)
    ticket = _daily_ticket_text(selected, horses, ticket_logic)
    stars, _label = ticket_logic._ticket_recommendation(selected)
    points = max(1, len(combinations)) if decision != "SKIP" else 0
    investment = _investment_for_code(strategy_selection, judgement_code) * points if decision == "BUY" else 0
    validation = strategy_selection.get("validation") if isinstance(strategy_selection.get("validation"), Mapping) else {}
    validation_races = _safe_int(validation.get("race_count"))
    return DailyInvestmentDecision(
        race_id=resolved_race_id,
        race_name=display_name,
        race_title=race_title,
        venue=venue,
        race_number=race_number,
        post_time=post_time,
        ticket=ticket,
        combinations=combinations,
        decision=decision,
        strategy_score=score,
        reproducibility=reproducibility,
        roi=roi,
        hit_rate=hit_rate,
        sample_size=sample_size,
        investment=investment,
        points=points,
        horses=horses,
        confidence=stars or _stars_from_score(score),
        reason=str(selected.get("理由") or selected.get("_judgement") or ""),
        validation_label=str(validation.get("label") or ""),
        validation_races=validation_races,
        detail_path=detail_path,
    )


def build_daily_summary(
    results: Iterable[tuple[str, PredictionResult]],
    *,
    summary_date: str,
    strategy_selection: Mapping[str, Any],
    detail_paths: Mapping[str, str] | None = None,
    errors: Iterable[Mapping[str, Any]] = (),
) -> DailySummary:
    _require_nar_race_type(strategy_selection, "Strategy selection")
    decisions = [
        build_daily_investment_decision(
            result,
            strategy_selection=strategy_selection,
            race_id=race_id,
            detail_path=str((detail_paths or {}).get(race_id) or ""),
        )
        for race_id, result in results
    ]
    buy = tuple(sort_daily_decisions(item for item in decisions if item.decision == "BUY"))
    hold = tuple(sort_daily_decisions(item for item in decisions if item.decision == "HOLD"))
    skip = tuple(sort_daily_decisions(item for item in decisions if item.decision == "SKIP"))
    clean_errors = tuple({str(key): str(value) for key, value in item.items()} for item in errors)
    strategy_meta = {
        "race_type": "nar",
        "strategy_id": str(strategy_selection.get("strategy_id") or ""),
        "strategy_source": str(strategy_selection.get("strategy_source") or ""),
        "validation": _json_ready(strategy_selection.get("validation") or {}),
    }
    return DailySummary(
        date=summary_date,
        buy=buy,
        hold=hold,
        skip=skip,
        strategy_selection=strategy_meta,
        errors=clean_errors,
    )


def sort_daily_decisions(
    decisions: Iterable[DailyInvestmentDecision],
    *,
    mode: str = "strategy",
) -> list[DailyInvestmentDecision]:
    items = list(decisions)
    if mode == "post_time":
        return sorted(items, key=lambda item: (_post_time_sort_value(item.post_time), item.venue, item.race_id))
    return sorted(items, key=_strategy_sort_key)


def filter_daily_decisions(
    decisions: Iterable[DailyInvestmentDecision],
    venue: str,
) -> list[DailyInvestmentDecision]:
    return [item for item in decisions if venue in {"", "全て", item.venue}]


def write_daily_summary(summary: DailySummary, path: str | Path) -> Path:
    return _write_json(summary.to_dict(), path)


def load_daily_summary(path: str | Path) -> DailySummary:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("nar_daily_summary.json must contain a JSON object.")
    return DailySummary.from_dict(data)


def write_daily_prediction_result(result: PredictionResult, path: str | Path) -> Path:
    if result.race_mode != NAR_RACE_TYPE:
        raise ValueError("Daily detail JSON accepts NAR PredictionResult only.")
    return _write_json(_prediction_result_to_dict(result), path)


def load_daily_prediction_result(path: str | Path) -> PredictionResult:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("Daily PredictionResult JSON must contain a JSON object.")
    _require_nar_race_type(data, "Daily PredictionResult")
    if str(data.get("race_mode") or "nar").lower() != "nar":
        raise ValueError("Daily PredictionResult must have race_mode=nar.")
    return PredictionResult(
        race_mode="nar",
        version=str(data.get("version") or ""),
        created_at=str(data.get("created_at") or ""),
        race_name=str(data.get("race_name") or ""),
        race_info=dict(data.get("race_info") or {}),
        overall_table=_records_frame(data.get("overall_table")),
        horse_evaluation=_records_frame(data.get("horse_evaluation")),
        attention_horses=[str(item) for item in data.get("attention_horses", [])],
        ai_race_review=str(data.get("ai_race_review") or ""),
        betting_structure=str(data.get("betting_structure") or ""),
        source_files={str(key): str(value) for key, value in dict(data.get("source_files") or {}).items()},
        status=str(data.get("status") or "not_started"),
        message=str(data.get("message") or ""),
        raw_output=str(data.get("raw_output") or ""),
        debug_info=dict(data.get("debug_info") or {}),
    )


def resolve_daily_detail_path(summary_path: str | Path, detail_path: str) -> Path:
    base = Path(summary_path).resolve().parent
    target = (base / detail_path).resolve()
    if target != base and base not in target.parents:
        raise ValueError("detail_path points outside assets/analysis.")
    return target


def normalize_daily_date(value: Any) -> str:
    text = str(value or "").strip()
    match = re.search(r"(20\d{2})[-/年]?(\d{2})[-/月]?(\d{2})", text)
    if not match:
        return ""
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3))).isoformat()
    except ValueError:
        return ""


def _ticket_input_frame(result: PredictionResult) -> pd.DataFrame:
    source = result.overall_table
    if not isinstance(source, pd.DataFrame) or source.empty:
        source = result.horse_evaluation
    if not isinstance(source, pd.DataFrame):
        return pd.DataFrame()
    frame = source.copy(deep=True)
    if "最終印" not in frame.columns:
        frame["最終印"] = _first_text_series(frame, ["old_final_mark", "旧印", "表示印", "display_mark", "印"])
    if "総合評価点" not in frame.columns:
        frame["総合評価点"] = _first_series(frame, ["final_mark_score", "総合評価監査点", "総合評価", "AI点"])
    if "AI順位" not in frame.columns and "ai_rank" in frame.columns:
        frame["AI順位"] = frame["ai_rank"]
    if "単勝オッズ" not in frame.columns and "オッズ" in frame.columns:
        frame["単勝オッズ"] = frame["オッズ"]
    return frame


def _first_series(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    result = pd.Series(pd.NA, index=frame.index, dtype="object")
    for column in columns:
        if column in frame.columns:
            result = result.where(result.notna(), frame[column])
    return result


def _first_text_series(frame: pd.DataFrame, columns: list[str]) -> pd.Series:
    result = pd.Series("", index=frame.index, dtype="object")
    for column in columns:
        if column not in frame.columns:
            continue
        values = frame[column].fillna("").astype(str).str.strip()
        result = result.where(result.str.len().gt(0), values)
    return result


def _candidate_horses(row: Mapping[str, Any], frame: pd.DataFrame) -> list[DailyHorse]:
    lookup = _horse_group_lookup(frame)
    text = str(row.get("単勝オッズ構成") or "")
    horses: list[DailyHorse] = []
    for part in re.split(r"[／/]", text):
        match = re.match(r"^\s*(.*?)(\d+)\s+(.+?)[：:]\s*([^倍]+)倍?\s*$", part)
        if not match:
            continue
        role, number_text, name, odds_text = match.groups()
        group = lookup.get(_horse_number_key(number_text)) or _group_from_role(role)
        horses.append(
            DailyHorse(
                number=int(number_text),
                name=name.strip(),
                group=group,
                role=role.strip(),
                odds=_safe_float(odds_text),
            )
        )
    return horses


def _horse_group_lookup(frame: pd.DataFrame) -> dict[str, str]:
    result: dict[str, str] = {}
    for _, row in frame.iterrows():
        key = _horse_number_key(row.get("馬番", row.get("horse_no")))
        if not key:
            continue
        group = str(row.get("display_group") or row.get("グループ") or "").strip()
        if group not in {"SS", "A", "B", "C", "Z"}:
            group = _group_from_role(str(row.get("表示印") or row.get("display_mark") or row.get("最終印") or ""))
        result[key] = group
    return result


def _candidate_combinations(row: Mapping[str, Any], horses: tuple[DailyHorse, ...]) -> tuple[str, ...]:
    if len(horses) < 2:
        return ()
    connector = "→" if "→" in str(row.get("買い目") or "") else "－"
    return (f"{horses[0].number}{connector}{horses[1].number}",)


def _daily_ticket_text(row: Mapping[str, Any], horses: tuple[DailyHorse, ...], ticket_logic: Any) -> str:
    bet_type = ticket_logic._ticket_bet_type_text(row)
    connector = "→" if "→" in str(row.get("買い目") or "") else "-"
    groups = [horse.group or _group_from_role(horse.role) for horse in horses]
    return f"{bet_type} {connector.join(groups)}".strip()


def _group_from_role(role: str) -> str:
    text = str(role or "").strip()
    if text.startswith("◎"):
        return "SS"
    if text.startswith(("○", "▲")):
        return "A"
    if text.startswith("△"):
        return "B"
    if text.startswith(("✓", "✔", "☆")):
        return "C"
    return "Z"


def _race_metadata(result: PredictionResult, race_id: str) -> tuple[str, str, str, str, str]:
    info = result.race_info or {}
    venue = _first_text(info, ["racecourse", "venue", "競馬場", "場所"])
    if not venue and len(race_id) >= 6:
        venue = NAR_VENUES.get(race_id[4:6], "")
    race_number = _first_text(info, ["race_number", "race_no", "R", "レース番号"])
    number_match = re.search(r"(1[0-2]|[1-9])", race_number)
    if number_match:
        race_number = f"{int(number_match.group(1))}R"
    elif len(race_id) >= 2 and race_id[-2:].isdigit():
        race_number = f"{int(race_id[-2:])}R"
    else:
        race_number = ""
    post_time = _first_text(info, ["post_time", "start_time", "発走時刻", "発走"])
    if not post_time:
        time_match = re.search(r"\b([01]?\d|2[0-3]):[0-5]\d\b", str(info.get("race_data") or ""))
        post_time = time_match.group(0) if time_match else ""
    race_title = str(result.race_name or info.get("race_name") or "").strip()
    display_name = f"{venue}{race_number}" if venue or race_number else race_title or race_id
    return venue, race_number, post_time, display_name, race_title


def _race_id_from_result(result: PredictionResult) -> str:
    info = result.race_info or {}
    for value in (info.get("race_id"), *result.source_files.values()):
        match = re.search(r"(?<!\d)(\d{12})(?!\d)", str(value or ""))
        if match:
            return match.group(1)
    return ""


def _decision_for_code(strategy: Mapping[str, Any], code: str) -> DecisionLabel:
    mapping = strategy.get("decision_mapping") if isinstance(strategy.get("decision_mapping"), Mapping) else {}
    value = str(mapping.get(code, "SKIP"))
    return value if value in {"BUY", "HOLD", "SKIP"} else "SKIP"  # type: ignore[return-value]


def _investment_for_code(strategy: Mapping[str, Any], code: str) -> int:
    investment = strategy.get("investment_yen") if isinstance(strategy.get("investment_yen"), Mapping) else {}
    return max(0, int(_safe_float(investment.get(code)) or 0))


def _skip_decision(
    race_id: str,
    race_name: str,
    race_title: str,
    venue: str,
    race_number: str,
    post_time: str,
    detail_path: str,
    strategy: Mapping[str, Any],
    reason: str,
) -> DailyInvestmentDecision:
    validation = strategy.get("validation") if isinstance(strategy.get("validation"), Mapping) else {}
    return DailyInvestmentDecision(
        race_id=race_id,
        race_name=race_name,
        race_title=race_title,
        venue=venue,
        race_number=race_number,
        post_time=post_time,
        ticket="",
        combinations=(),
        decision="SKIP",
        strategy_score=0,
        reproducibility="",
        roi=None,
        hit_rate=None,
        sample_size=0,
        investment=0,
        points=0,
        confidence="★☆☆☆☆",
        reason=reason,
        validation_label=str(validation.get("label") or ""),
        validation_races=_safe_int(validation.get("race_count")),
        detail_path=detail_path,
    )


def _strategy_sort_key(item: DailyInvestmentDecision) -> tuple[float, float, float, int, str]:
    reproducibility_rank = {"高": 3, "中": 2, "低": 1}.get(item.reproducibility, 0)
    roi = item.roi if item.roi is not None else -1.0
    return (
        -float(item.strategy_score),
        -float(reproducibility_rank),
        -roi,
        _post_time_sort_value(item.post_time),
        item.race_id,
    )


def _post_time_sort_value(value: str) -> int:
    match = re.search(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", str(value or ""))
    return int(match.group(1)) * 60 + int(match.group(2)) if match else 24 * 60 + 1


def _stars_from_score(score: int) -> str:
    count = 5 if score >= 85 else 4 if score >= 70 else 3 if score >= 55 else 2 if score >= 40 else 1
    return "★" * count + "☆" * (5 - count)


def _load_nar_ticket_logic() -> Any:
    from . import nar_notebook_logic

    return nar_notebook_logic


def _require_nar_race_type(data: Mapping[str, Any], label: str) -> None:
    if str(data.get("race_type") or "").lower() != NAR_RACE_TYPE:
        raise ValueError(f"{label} must have race_type=nar.")


def _first_text(data: Mapping[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = data.get(key)
        if value is not None and str(value).strip() and str(value).lower() != "nan":
            return str(value).strip()
    return ""


def _horse_number_key(value: Any) -> str:
    if value is None:
        return ""
    try:
        number = float(value)
        if math.isnan(number):
            return ""
        if number.is_integer():
            return str(int(number))
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _safe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        number = float(str(value).replace(",", "").replace("%", "").strip())
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _safe_int(value: Any) -> int | None:
    number = _safe_float(value)
    return int(number) if number is not None else None


def _prediction_result_to_dict(result: PredictionResult) -> dict[str, Any]:
    return _json_ready({
        "race_type": "nar",
        "race_mode": result.race_mode,
        "version": result.version,
        "created_at": result.created_at,
        "race_name": result.race_name,
        "race_info": result.race_info,
        "overall_table": _frame_records(result.overall_table),
        "horse_evaluation": _frame_records(result.horse_evaluation),
        "attention_horses": result.attention_horses,
        "ai_race_review": result.ai_race_review,
        "betting_structure": result.betting_structure,
        "source_files": result.source_files,
        "status": result.status,
        "message": result.message,
        "raw_output": result.raw_output,
        "debug_info": result.debug_info,
    })


def _frame_records(value: Any) -> list[dict[str, Any]] | None:
    if not isinstance(value, pd.DataFrame):
        return None
    return value.to_dict("records")


def _records_frame(value: Any) -> pd.DataFrame | None:
    if value is None:
        return None
    if not isinstance(value, list):
        raise ValueError("Serialized DataFrame must be a list of records.")
    return pd.DataFrame(value)


def _json_ready(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _json_ready(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_ready(item) for item in value]
    if isinstance(value, (datetime, date)):
        return value.isoformat()
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "item"):
        try:
            return _json_ready(value.item())
        except (TypeError, ValueError):
            pass
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def _write_json(data: Mapping[str, Any], path: str | Path) -> Path:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(json.dumps(_json_ready(data), ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    temporary.replace(target)
    return target

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
from .race_investment_strategy import (
    load_jra_strategy_selection,
    select_jra_investment_strategy,
)


DecisionLabel = Literal["BUY", "HOLD", "SKIP"]

JRA_VENUES = {
    "01": "札幌",
    "02": "函館",
    "03": "福島",
    "04": "新潟",
    "05": "東京",
    "06": "中山",
    "07": "中京",
    "08": "京都",
    "09": "阪神",
    "10": "小倉",
}
JRA_VENUE_ORDER = ("札幌", "函館", "福島", "新潟", "東京", "中山", "中京", "京都", "阪神", "小倉")


@dataclass(frozen=True)
class WeekendHorse:
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
    def from_dict(cls, data: Mapping[str, Any]) -> "WeekendHorse":
        return cls(
            number=data.get("number", ""),
            name=str(data.get("name") or ""),
            group=str(data.get("group") or ""),
            role=str(data.get("role") or ""),
            odds=_safe_float(data.get("odds")),
        )


@dataclass(frozen=True)
class InvestmentDecision:
    race_id: str
    race_name: str
    race_title: str
    venue: str
    race_number: str
    ticket: str
    decision: DecisionLabel
    strategy_score: int
    roi: float | None
    investment: int
    selected_strategy: str = ""
    strategy_id: str = ""
    expected_roi: float | None = None
    hit_rate: float | None = None
    sample_size: int = 0
    points: int = 0
    combinations: tuple[str, ...] = ()
    horses: tuple[WeekendHorse, ...] = ()
    confidence: str = "☆☆☆☆☆"
    reason: str = ""
    avoid_reason: str = ""
    detail_path: str = ""
    strategy_audit: Mapping[str, Any] = field(default_factory=dict)
    race_type: str = "jra"

    def to_dict(self) -> dict[str, Any]:
        return {
            "race_type": self.race_type,
            "race_id": self.race_id,
            "race_name": self.race_name,
            "race_title": self.race_title,
            "venue": self.venue,
            "race_number": self.race_number,
            "ticket": self.ticket,
            "decision": self.decision,
            "selected_strategy": self.selected_strategy,
            "strategy_id": self.strategy_id,
            "score": self.strategy_score,
            "strategy_score": self.strategy_score,
            "expected_roi": self.expected_roi,
            "roi": self.roi,
            "hit_rate": self.hit_rate,
            "sample_size": self.sample_size,
            "investment": self.investment,
            "points": self.points,
            "combinations": list(self.combinations),
            "confidence": self.confidence,
            "reason": self.reason,
            "avoid_reason": self.avoid_reason,
            "detail_path": self.detail_path,
            "horses": [horse.to_dict() for horse in self.horses],
            "strategy_audit": _json_ready(self.strategy_audit),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "InvestmentDecision":
        decision = str(data.get("decision") or "SKIP").upper()
        if decision not in {"BUY", "HOLD", "SKIP"}:
            decision = "SKIP"
        return cls(
            race_id=str(data.get("race_id") or ""),
            race_name=str(data.get("race_name") or ""),
            race_title=str(data.get("race_title") or ""),
            venue=str(data.get("venue") or ""),
            race_number=str(data.get("race_number") or ""),
            ticket=str(data.get("ticket") or ""),
            decision=decision,  # type: ignore[arg-type]
            strategy_score=int(_safe_float(data.get("strategy_score", data.get("score"))) or 0),
            roi=_safe_float(data.get("roi")),
            investment=int(_safe_float(data.get("investment")) or 0),
            selected_strategy=str(data.get("selected_strategy") or ""),
            strategy_id=str(data.get("strategy_id") or ""),
            expected_roi=_safe_float(data.get("expected_roi")),
            hit_rate=_safe_float(data.get("hit_rate")),
            sample_size=int(_safe_float(data.get("sample_size")) or 0),
            points=int(_safe_float(data.get("points")) or 0),
            combinations=tuple(str(item) for item in data.get("combinations", []) if str(item).strip()),
            horses=tuple(WeekendHorse.from_dict(item) for item in data.get("horses", []) if isinstance(item, Mapping)),
            confidence=str(data.get("confidence") or "☆☆☆☆☆"),
            reason=str(data.get("reason") or ""),
            avoid_reason=str(data.get("avoid_reason") or ""),
            detail_path=str(data.get("detail_path") or ""),
            strategy_audit=dict(data.get("strategy_audit") or {}),
        )


@dataclass(frozen=True)
class WeekendSummary:
    date: str
    buy: tuple[InvestmentDecision, ...] = ()
    hold: tuple[InvestmentDecision, ...] = ()
    skip: tuple[InvestmentDecision, ...] = ()
    strategy_selection: dict[str, Any] = field(default_factory=dict)
    generated_at: str = field(default_factory=lambda: datetime.now().isoformat(timespec="seconds"))
    errors: tuple[dict[str, str], ...] = ()
    race_type: str = "jra"

    @property
    def all_decisions(self) -> tuple[InvestmentDecision, ...]:
        return self.buy + self.hold + self.skip

    @property
    def venues(self) -> tuple[str, ...]:
        venues = {item.venue for item in self.all_decisions if item.venue}
        order = {venue: index for index, venue in enumerate(JRA_VENUE_ORDER)}
        return tuple(sorted(venues, key=lambda venue: (order.get(venue, len(order)), venue)))

    def to_dict(self) -> dict[str, Any]:
        return {
            "race_type": self.race_type,
            "date": self.date,
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
    def from_dict(cls, data: Mapping[str, Any]) -> "WeekendSummary":
        return cls(
            date=str(data.get("date") or ""),
            generated_at=str(data.get("generated_at") or ""),
            strategy_selection=dict(data.get("strategy_selection") or {}),
            buy=tuple(InvestmentDecision.from_dict(item) for item in data.get("buy", []) if isinstance(item, Mapping)),
            hold=tuple(InvestmentDecision.from_dict(item) for item in data.get("hold", []) if isinstance(item, Mapping)),
            skip=tuple(InvestmentDecision.from_dict(item) for item in data.get("skip", []) if isinstance(item, Mapping)),
            errors=tuple(dict(item) for item in data.get("errors", []) if isinstance(item, Mapping)),
        )


def build_investment_decision(
    result: PredictionResult,
    *,
    race_id: str = "",
    detail_path: str = "",
    strategy_selection: Mapping[str, Any] | None = None,
) -> InvestmentDecision:
    """Build a JRA race-level investment decision without changing prediction logic."""
    if result.race_mode != "jra":
        raise ValueError("Keiba AI Weekend supports JRA PredictionResult only.")

    resolved_race_id = race_id or _race_id_from_result(result)
    venue, race_number, display_name, race_title = _race_metadata(result, resolved_race_id)
    loaded_strategy = strategy_selection
    if loaded_strategy is None:
        try:
            loaded_strategy = load_jra_strategy_selection()
        except Exception:
            loaded_strategy = None
    if loaded_strategy is not None:
        payload = select_jra_investment_strategy(result, loaded_strategy)
        return _decision_from_strategy_payload(
            payload,
            resolved_race_id,
            display_name,
            race_title,
            venue,
            race_number,
            detail_path,
        )
    return _build_ticket_fallback_decision(result, resolved_race_id, display_name, race_title, venue, race_number, detail_path)


def build_weekend_summary(
    results: Iterable[tuple[str, PredictionResult]],
    *,
    summary_date: str,
    detail_paths: Mapping[str, str] | None = None,
    errors: Iterable[Mapping[str, Any]] = (),
) -> WeekendSummary:
    try:
        strategy_selection = load_jra_strategy_selection()
    except Exception:
        strategy_selection = None

    decisions = [
        build_investment_decision(
            result,
            race_id=race_id,
            detail_path=str((detail_paths or {}).get(race_id) or ""),
            strategy_selection=strategy_selection,
        )
        for race_id, result in results
    ]
    buy = tuple(sorted((item for item in decisions if item.decision == "BUY"), key=_decision_sort_key))
    hold = tuple(sorted((item for item in decisions if item.decision == "HOLD"), key=_decision_sort_key))
    skip = tuple(sorted((item for item in decisions if item.decision == "SKIP"), key=_race_sort_key))
    clean_errors = tuple({str(key): str(value) for key, value in item.items()} for item in errors)
    strategy_meta = {
        "race_type": "jra",
        "strategy_id": str((strategy_selection or {}).get("strategy_id") or ""),
        "strategy_source": str((strategy_selection or {}).get("strategy_source") or ""),
        "validation": _json_ready((strategy_selection or {}).get("validation") or (strategy_selection or {}).get("source") or {}),
    }
    return WeekendSummary(
        date=summary_date,
        buy=buy,
        hold=hold,
        skip=skip,
        strategy_selection=strategy_meta,
        errors=clean_errors,
    )


def write_weekend_summary(summary: WeekendSummary, path: str | Path) -> Path:
    return _write_json(summary.to_dict(), path)


def load_weekend_summary(path: str | Path) -> WeekendSummary:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("weekend_summary.json must contain an object.")
    return WeekendSummary.from_dict(data)


def write_prediction_result(result: PredictionResult, path: str | Path) -> Path:
    return _write_json(prediction_result_to_dict(result), path)


def load_prediction_result(path: str | Path) -> PredictionResult:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, Mapping):
        raise ValueError("PredictionResult JSON must contain an object.")
    return prediction_result_from_dict(data)


def prediction_result_to_dict(result: PredictionResult) -> dict[str, Any]:
    return _json_ready({
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


def prediction_result_from_dict(data: Mapping[str, Any]) -> PredictionResult:
    race_mode = str(data.get("race_mode") or "jra")
    if race_mode not in {"jra", "nar"}:
        raise ValueError(f"Unsupported race_mode: {race_mode}")
    return PredictionResult(
        race_mode=race_mode,  # type: ignore[arg-type]
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


def resolve_detail_path(summary_path: str | Path, detail_path: str) -> Path:
    base = Path(summary_path).resolve().parent
    target = (base / detail_path).resolve()
    if target != base and base not in target.parents:
        raise ValueError("detail_path points outside assets/analysis.")
    return target


def normalize_summary_date(value: Any) -> str:
    text = str(value or "").strip()
    match = re.search(r"(20\d{2})[-/年]?(\d{2})[-/月]?(\d{2})", text)
    if not match:
        return ""
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3))).isoformat()
    except ValueError:
        return ""


def _decision_from_strategy_payload(
    payload: Mapping[str, Any],
    race_id: str,
    race_name: str,
    race_title: str,
    venue: str,
    race_number: str,
    detail_path: str,
) -> InvestmentDecision:
    decision = str(payload.get("decision") or "SKIP").upper()
    if decision not in {"BUY", "HOLD", "SKIP"}:
        decision = "SKIP"
    horses = tuple(WeekendHorse.from_dict(item) for item in payload.get("horses", []) if isinstance(item, Mapping))
    return InvestmentDecision(
        race_id=race_id,
        race_name=race_name,
        race_title=race_title,
        venue=venue,
        race_number=race_number,
        ticket=str(payload.get("ticket") or ""),
        decision=decision,  # type: ignore[arg-type]
        strategy_score=int(_safe_float(payload.get("strategy_score")) or 0),
        roi=_safe_float(payload.get("roi", payload.get("expected_roi"))),
        investment=int(_safe_float(payload.get("investment")) or 0),
        selected_strategy=str(payload.get("selected_strategy") or ""),
        strategy_id=str(payload.get("strategy_id") or ""),
        expected_roi=_safe_float(payload.get("expected_roi", payload.get("roi"))),
        hit_rate=_safe_float(payload.get("hit_rate")),
        sample_size=int(_safe_float(payload.get("sample_size")) or 0),
        points=int(_safe_float(payload.get("points")) or 0),
        combinations=tuple(str(item) for item in payload.get("combinations", []) if str(item).strip()),
        horses=horses,
        confidence=str(payload.get("confidence") or "☆☆☆☆☆"),
        reason=str(payload.get("reason") or ""),
        avoid_reason=str(payload.get("avoid_reason") or ""),
        detail_path=detail_path,
        strategy_audit=dict(payload.get("strategy_audit") or {}),
    )


def _build_ticket_fallback_decision(
    result: PredictionResult,
    resolved_race_id: str,
    display_name: str,
    race_title: str,
    venue: str,
    race_number: str,
    detail_path: str,
) -> InvestmentDecision:
    """Fallback only for missing/corrupt JRA strategy JSON."""
    frame = _ticket_input_frame(result)
    if frame.empty:
        return _skip_decision(resolved_race_id, display_name, race_title, venue, race_number, detail_path, "horse data missing")

    ticket_logic = _load_jra_ticket_logic()
    candidate_rows = ticket_logic._ticket_candidate_rows(frame, confidence_summary=None, race_type="jra")
    top_rows = ticket_logic._ticket_top_rows(candidate_rows, confidence_summary=None)
    if top_rows:
        selected = top_rows[0]
    elif candidate_rows:
        selected = sorted(candidate_rows, key=ticket_logic._ticket_ranking_sort_key, reverse=True)[0]
    else:
        return _skip_decision(resolved_race_id, display_name, race_title, venue, race_number, detail_path, "ticket candidate missing")

    judgement_code = ticket_logic._ticket_judgement_code(selected)
    decision: DecisionLabel = "BUY" if judgement_code in {"A", "B"} else "HOLD" if judgement_code == "C" else "SKIP"
    score = int(_safe_float(selected.get("_score", selected.get("買い目スコア"))) or 0)
    roi = _safe_float(selected.get("_roi"))
    horses = tuple(_candidate_horses(selected, frame))
    ticket = _weekend_ticket_text(selected, horses, ticket_logic)
    stars, _label = ticket_logic._ticket_recommendation(selected)
    investment = 200 if decision == "BUY" and judgement_code == "A" else 100 if decision == "BUY" else 0
    reason = str(selected.get("理由") or selected.get("_judgement") or "")
    return InvestmentDecision(
        race_id=resolved_race_id,
        race_name=display_name,
        race_title=race_title,
        venue=venue,
        race_number=race_number,
        ticket=ticket,
        decision=decision,
        strategy_score=score,
        roi=roi,
        investment=investment,
        selected_strategy=ticket,
        hit_rate=_safe_float(selected.get("_hit_rate")),
        sample_size=int(_safe_float(selected.get("_n")) or 0),
        horses=horses,
        confidence=stars or _stars_from_score(score),
        reason=reason,
        detail_path=detail_path,
    )


def _skip_decision(
    race_id: str,
    race_name: str,
    race_title: str,
    venue: str,
    race_number: str,
    detail_path: str,
    reason: str,
) -> InvestmentDecision:
    return InvestmentDecision(
        race_id=race_id,
        race_name=race_name,
        race_title=race_title,
        venue=venue,
        race_number=race_number,
        ticket="",
        decision="SKIP",
        strategy_score=0,
        roi=None,
        investment=0,
        confidence="☆☆☆☆☆",
        reason=reason,
        detail_path=detail_path,
    )


def _ticket_input_frame(result: PredictionResult) -> pd.DataFrame:
    source = result.overall_table
    if not isinstance(source, pd.DataFrame) or source.empty:
        source = result.horse_evaluation
    if not isinstance(source, pd.DataFrame):
        return pd.DataFrame()
    frame = source.copy(deep=True)
    if "最終印" not in frame.columns:
        frame["最終印"] = _first_text_series(frame, ["old_final_mark", "display_mark", "印"])
    if "総合評価点" not in frame.columns:
        frame["総合評価点"] = _first_series(frame, ["final_mark_score", "総合評価", "AI点"])
    if "AI順位" not in frame.columns and "ai_rank" in frame.columns:
        frame["AI順位"] = frame["ai_rank"]
    if "単勝オッズ" not in frame.columns and "オッズ" in frame.columns:
        frame["単勝オッズ"] = frame["オッズ"]
    return frame


def _candidate_horses(row: Mapping[str, Any], frame: pd.DataFrame) -> list[WeekendHorse]:
    lookup = _horse_group_lookup(frame)
    text = str(row.get("単勝オッズ構成") or row.get("蜊伜享繧ｪ繝・ぜ讒区・") or "")
    horses: list[WeekendHorse] = []
    for part in re.split(r"[・/]", text):
        match = re.match(r"^\s*(.*?)(\d+)\s+(.+?)[・/]\s*([^倍]+)倍\s*$", part)
        if not match:
            continue
        role, number_text, name, odds_text = match.groups()
        key = _horse_number_key(number_text)
        horses.append(
            WeekendHorse(
                number=int(number_text),
                name=name.strip(),
                group=lookup.get(key) or _group_from_role(role),
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


def _weekend_ticket_text(row: Mapping[str, Any], horses: tuple[WeekendHorse, ...], ticket_logic: Any) -> str:
    bet_type = ticket_logic._ticket_bet_type_text(row)
    source = str(row.get("買い目") or "")
    connector = "→" if "→" in source else "-"
    groups = [horse.group or _group_from_role(horse.role) for horse in horses]
    combo = connector.join(groups) if groups else _group_combo_from_text(source)
    return f"{bet_type} {combo}".strip()


def _group_combo_from_text(value: str) -> str:
    combo = str(value or "").split(" ", 1)[-1]
    connector = "→" if "→" in combo else "-"
    roles = re.split(r"[・×/-]", combo)
    return connector.join(_group_from_role(role) for role in roles if role)


def _group_from_role(role: str) -> str:
    text = str(role or "").strip()
    if text.startswith("◎"):
        return "SS"
    if text.startswith(("○", "▲")):
        return "A"
    if text.startswith("△"):
        return "B"
    if text.startswith(("✓", "✔")):
        return "C"
    return "Z"


def _race_metadata(result: PredictionResult, race_id: str) -> tuple[str, str, str, str]:
    info = result.race_info or {}
    venue = _first_text(info, ["racecourse", "venue", "競馬場", "場所"])
    if not venue and len(race_id) >= 6:
        venue = JRA_VENUES.get(race_id[4:6], "")
    race_number = _first_text(info, ["race_number", "race_no", "R", "レース番号"])
    number_match = re.search(r"([1-9]|1[0-2])", race_number)
    if number_match:
        race_number = f"{int(number_match.group(1))}R"
    elif len(race_id) >= 2 and race_id[-2:].isdigit():
        race_number = f"{int(race_id[-2:])}R"
    else:
        race_number = ""
    race_title = str(result.race_name or info.get("race_name") or "").strip()
    display_name = f"{venue}{race_number}" if venue or race_number else race_title or race_id
    return venue, race_number, display_name, race_title


def _race_id_from_result(result: PredictionResult) -> str:
    info = result.race_info or {}
    for value in (info.get("race_id"), *result.source_files.values()):
        match = re.search(r"(?<!\d)(\d{12})(?!\d)", str(value or ""))
        if match:
            return match.group(1)
    return ""


def _decision_sort_key(item: InvestmentDecision) -> tuple[float, float, str]:
    roi = item.roi if item.roi is not None else -1.0
    return (-float(item.strategy_score), -roi, item.race_id)


def _race_sort_key(item: InvestmentDecision) -> tuple[int, int, str]:
    venue_order = {venue: index for index, venue in enumerate(JRA_VENUE_ORDER)}
    return (venue_order.get(item.venue, len(venue_order)), _race_number_sort_value(item.race_number), item.race_id)


def _race_number_sort_value(value: str) -> int:
    match = re.search(r"([1-9]|1[0-2])", str(value or ""))
    return int(match.group(1)) if match else 99


def _stars_from_score(score: int) -> str:
    count = 5 if score >= 85 else 4 if score >= 70 else 3 if score >= 55 else 2 if score >= 40 else 1
    return "★" * count + "☆" * (5 - count)


def _load_jra_ticket_logic() -> Any:
    from . import jra_notebook_logic

    return jra_notebook_logic


def _first_text(data: Mapping[str, Any], keys: list[str]) -> str:
    for key in keys:
        value = data.get(key)
        if value is not None and str(value).strip() and str(value).lower() != "nan":
            return str(value).strip()
    return ""


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

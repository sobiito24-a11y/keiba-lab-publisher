# -*- coding: utf-8 -*-
from __future__ import annotations

import itertools
from dataclasses import dataclass
from typing import Any, Iterable

import pandas as pd

from .purchase_conditions import (
    clean_text,
    display_group_from_row,
    horse_no,
    max_contribution,
    max_drawdown,
    max_losing_streak,
    normalize_mark,
    pct,
    to_float,
)


POINT_LIMITS: dict[str, tuple[int, int]] = {
    "単勝": (1, 2),
    "複勝": (1, 2),
    "ワイド": (1, 5),
    "馬連": (1, 5),
    "馬単": (2, 8),
    "三連複": (3, 15),
    "三連単": (6, 30),
}


@dataclass(frozen=True)
class TicketStrategySpec:
    strategy_id: str
    ticket_type: str
    label: str
    pattern: dict[str, Any]
    stance: str


def build_default_ticket_strategy_specs() -> list[TicketStrategySpec]:
    specs: list[TicketStrategySpec] = []
    for role in ["AI1", "AI2", "AI3", "SS", "A", "B", "C"]:
        specs.append(TicketStrategySpec(f"win_{role.lower()}", "単勝", role, {"type": "single", "roles": [role]}, "妙味"))
        specs.append(TicketStrategySpec(f"place_{role.lower()}", "複勝", role, {"type": "single", "roles": [role]}, "安定"))

    pair_specs = [
        ("SS-A", ["SS"], ["A"]),
        ("SS-B", ["SS"], ["B"]),
        ("SS-C", ["SS"], ["C"]),
        ("A-B", ["A"], ["B"]),
        ("A-C", ["A"], ["C"]),
        ("AI1-AI2", ["AI1"], ["AI2"]),
        ("AI1-AI3", ["AI1"], ["AI3"]),
        ("AI2-AI3", ["AI2"], ["AI3"]),
    ]
    for label, left, right in pair_specs:
        specs.append(
            TicketStrategySpec(
                f"wide_{safe_id(label)}",
                "ワイド",
                label,
                {"type": "pair", "left_roles": left, "right_roles": right},
                "安定",
            )
        )
        specs.append(
            TicketStrategySpec(
                f"quinella_{safe_id(label)}",
                "馬連",
                label,
                {"type": "pair", "left_roles": left, "right_roles": right},
                "本線",
            )
        )

    exacta_specs = [
        ("SS→A", ["SS"], ["A"]),
        ("A→SS", ["A"], ["SS"]),
        ("SS→B", ["SS"], ["B"]),
        ("A→B", ["A"], ["B"]),
        ("AI1→AI2", ["AI1"], ["AI2"]),
        ("AI2→AI1", ["AI2"], ["AI1"]),
    ]
    for label, first, second in exacta_specs:
        specs.append(
            TicketStrategySpec(
                f"exacta_{safe_id(label)}",
                "馬単",
                label,
                {"type": "exacta", "first_roles": first, "second_roles": second},
                "妙味",
            )
        )

    trio_specs = [
        ("AI1-AI2-AI3 BOX", ["AI1", "AI2", "AI3"]),
        ("SS-A-B BOX", ["SS", "A", "B"]),
        ("SS-A-C BOX", ["SS", "A", "C"]),
        ("SS-A-B-C BOX", ["SS", "A", "B", "C"]),
    ]
    for label, roles in trio_specs:
        specs.append(
            TicketStrategySpec(
                f"trio_{safe_id(label)}",
                "三連複",
                label,
                {"type": "box", "roles": roles, "size": 3},
                "本線" if "C" not in roles else "妙味",
            )
        )

    trifecta_specs = [
        ("AI1→AI2/AI3→AI2/AI3/C", ["AI1"], ["AI2", "AI3"], ["AI2", "AI3", "C"]),
        ("SS→A/B→A/B/C", ["SS"], ["A", "B"], ["A", "B", "C"]),
        ("A→SS/B→SS/B/C", ["A"], ["SS", "B"], ["SS", "B", "C"]),
    ]
    for label, first, second, third in trifecta_specs:
        specs.append(
            TicketStrategySpec(
                f"trifecta_{safe_id(label)}",
                "三連単",
                label,
                {"type": "trifecta", "first_roles": first, "second_roles": second, "third_roles": third},
                "高リスク",
            )
        )
    return specs


def evaluate_ticket_strategies(
    records: pd.DataFrame,
    payouts_by_race: dict[str, dict[str, Any]],
    *,
    source_race_count: int | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if records is None or records.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty

    enriched = prepare_ticket_records(records)
    race_count = int(source_race_count or enriched["race_id"].nunique())
    rows = [
        evaluate_ticket_strategy(enriched, payouts_by_race, spec, race_count)
        for spec in build_default_ticket_strategy_specs()
    ]
    frame = pd.DataFrame([row for row in rows if row and row["purchase_races"] > 0])
    if frame.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty
    frame = frame.sort_values(
        ["reliability_score", "return_rate", "hit_rate", "purchase_races"],
        ascending=[False, False, False, False],
    ).reset_index(drop=True)
    official = frame[frame["risk_label"].eq("正式")].copy()
    reference = frame[frame["risk_label"].isin(["参考", "高リスク"])].copy()
    avoid = frame[(frame["return_rate"] < 60) & (frame["purchase_races"] >= 10)].copy()
    return frame, official, reference, avoid


def prepare_ticket_records(records: pd.DataFrame) -> pd.DataFrame:
    frame = records.copy()
    frame["race_id"] = frame["race_id"].astype(str)
    frame["horse_no_eval"] = frame.apply(lambda row: horse_no(first_existing(row, ["horse_no_eval", "馬番", "horse_no"])), axis=1)
    frame["ai_rank_eval"] = pd.to_numeric(frame.get("ai_rank_eval", frame.get("ai_rank")), errors="coerce")
    if "ai_rank_eval" not in frame or frame["ai_rank_eval"].isna().all():
        ai_col = first_column(frame, ["AI順位", "AI点順位"])
        if ai_col:
            frame["ai_rank_eval"] = pd.to_numeric(frame[ai_col], errors="coerce")
    if "ai_rank_eval" not in frame or frame["ai_rank_eval"].isna().all():
        score_col = first_column(frame, ["AI点", "ai_score"])
        if score_col:
            frame["ai_rank_eval"] = frame.groupby("race_id")[score_col].rank(method="first", ascending=False)
    frame["display_group_eval"] = frame.apply(display_group_from_row, axis=1)
    frame["mark_eval"] = frame.apply(lambda row: normalize_mark(first_existing(row, ["mark_eval", "表示印", "最終印", "印", "mark"])), axis=1)
    frame["odds_eval"] = frame.apply(lambda row: to_float(first_existing(row, ["odds_eval", "オッズ", "単勝オッズ", "odds"])), axis=1)
    frame["popularity_eval"] = frame.apply(lambda row: to_float(first_existing(row, ["popularity_eval", "人気", "popularity"])), axis=1)
    frame["finish_eval"] = pd.to_numeric(frame.get("finish_eval", frame.get("実際の着順")), errors="coerce")
    return frame


def evaluate_ticket_strategy(
    records: pd.DataFrame,
    payouts_by_race: dict[str, dict[str, Any]],
    spec: TicketStrategySpec,
    source_race_count: int,
) -> dict[str, Any] | None:
    race_rows: list[dict[str, Any]] = []
    for race_id, group in records.groupby("race_id", sort=True):
        tickets = build_tickets_for_race(group, spec.pattern, spec.ticket_type)
        min_points, max_points = POINT_LIMITS.get(spec.ticket_type, (1, 999))
        if len(tickets) < min_points or len(tickets) > max_points:
            continue
        payout = payout_for_tickets(tickets, spec.ticket_type, payouts_by_race.get(str(race_id), {}))
        race_rows.append(
            {
                "race_id": str(race_id),
                "points": len(tickets),
                "stake": len(tickets) * 100,
                "payout": payout,
                "hit": payout > 0,
            }
        )
    if not race_rows:
        return None
    purchase_races = len(race_rows)
    points = sum(row["points"] for row in race_rows)
    stake = sum(row["stake"] for row in race_rows)
    payout = sum(row["payout"] for row in race_rows)
    hits = sum(1 for row in race_rows if row["hit"])
    profits = [row["payout"] - row["stake"] for row in race_rows]
    max_contrib = max_contribution(pd.Series([row["payout"] for row in race_rows]))
    time_split = ticket_time_split(race_rows)
    risk = classify_ticket_strategy(purchase_races, hits, max_contrib, spec.ticket_type, points / purchase_races, time_split)
    score = ticket_recommendation_score(purchase_races, hits, payout, stake, max_contrib, spec.ticket_type, points / purchase_races, risk)
    return {
        "strategy_id": spec.strategy_id,
        "recommendation_kind": "ticket_strategy",
        "ticket_type": spec.ticket_type,
        "label": spec.label,
        "stance": spec.stance,
        "role_pattern": spec.pattern,
        "source_race_count": source_race_count,
        "target_race_count": source_race_count,
        "purchase_races": purchase_races,
        "purchase_points": points,
        "average_points": round(points / purchase_races, 2) if purchase_races else 0.0,
        "stake": stake,
        "payout": payout,
        "hits": hits,
        "hit_rate": pct(hits, purchase_races),
        "return_rate": pct(payout, stake),
        "profit": round(payout - stake, 1),
        "average_payout": round(payout / hits, 1) if hits else 0.0,
        "max_losing_streak": max_losing_streak([row["hit"] for row in race_rows]),
        "max_drawdown": round(max_drawdown(profits), 1),
        "max_payout_contribution": round(max_contrib, 1),
        "high_payout_dependency": dependency_label_ja(max_contrib),
        "risk_label": risk,
        "reliability_score": round(score, 1),
        "stars": stars_for_ticket_score(score),
        "time_split_result": time_split,
        "odds_basis": "保存HTMLのレース前オッズで条件判定、払戻は結果HTMLの確定払戻",
        "current_odds_note": "現在オッズで暫定一致。最終オッズにより条件外となる可能性があります。",
    }


def build_tickets_for_race(group: pd.DataFrame, pattern: dict[str, Any], ticket_type: str) -> set[tuple[str, ...]]:
    kind = str(pattern.get("type", ""))
    if kind == "single":
        nums = role_numbers(group, list(pattern.get("roles", [])))
        return {(no,) for no in unique_nums(nums)}
    if kind == "pair":
        left = unique_nums(role_numbers(group, list(pattern.get("left_roles", []))))
        right = unique_nums(role_numbers(group, list(pattern.get("right_roles", []))))
        return {pair_key([a, b]) for a in left for b in right if a != b}
    if kind == "exacta":
        first = unique_nums(role_numbers(group, list(pattern.get("first_roles", []))))
        second = unique_nums(role_numbers(group, list(pattern.get("second_roles", []))))
        return {(a, b) for a in first for b in second if a != b}
    if kind == "box":
        nums = unique_nums(role_numbers(group, list(pattern.get("roles", []))))
        size = int(pattern.get("size", 3) or 3)
        if ticket_type == "三連複":
            return {tuple(sorted(combo, key=int)) for combo in itertools.combinations(nums, size)}
        if ticket_type == "三連単":
            return {tuple(combo) for combo in itertools.permutations(nums, size)}
    if kind == "trifecta":
        first = unique_nums(role_numbers(group, list(pattern.get("first_roles", []))))
        second = unique_nums(role_numbers(group, list(pattern.get("second_roles", []))))
        third = unique_nums(role_numbers(group, list(pattern.get("third_roles", []))))
        return {(a, b, c) for a in first for b in second for c in third if len({a, b, c}) == 3}
    return set()


def role_numbers(group: pd.DataFrame, roles: list[str]) -> list[str]:
    nums: list[str] = []
    for role in roles:
        role_text = str(role)
        if role_text.startswith("AI"):
            rank = to_float(role_text.replace("AI", ""))
            if rank is None:
                continue
            subset = group[pd.to_numeric(group["ai_rank_eval"], errors="coerce").eq(rank)]
        else:
            subset = group[group["display_group_eval"].astype(str).eq(role_text)]
        nums.extend(subset["horse_no_eval"].map(horse_no).tolist())
    return nums


def payout_for_tickets(tickets: set[tuple[str, ...]], ticket_type: str, payouts: dict[str, Any]) -> float:
    if not tickets:
        return 0.0
    if ticket_type == "単勝":
        return sum(float(payouts.get("win", {}).get(ticket[0], 0) or 0) for ticket in tickets)
    if ticket_type == "複勝":
        return sum(float(payouts.get("place", {}).get(ticket[0], 0) or 0) for ticket in tickets)
    if ticket_type == "ワイド":
        return sum(float(payouts.get("wide", {}).get(pair_key(ticket), 0) or 0) for ticket in tickets)
    if ticket_type == "馬連":
        return sum(float(payouts.get("quinella", {}).get(pair_key(ticket), 0) or 0) for ticket in tickets)
    if ticket_type == "馬単":
        return sum(float(payouts.get("exacta", {}).get(order_key(ticket), 0) or 0) for ticket in tickets)
    if ticket_type == "三連複":
        return sum(float(payouts.get("trio", {}).get(trio_key(ticket), 0) or 0) for ticket in tickets)
    if ticket_type == "三連単":
        return sum(float(payouts.get("trifecta", {}).get(order_key(ticket), 0) or 0) for ticket in tickets)
    return 0.0


def ticket_time_split(race_rows: list[dict[str, Any]]) -> dict[str, Any]:
    ordered = sorted(race_rows, key=lambda row: str(row["race_id"]))
    split_at = max(1, int(len(ordered) * 0.7))
    train = ordered[:split_at]
    test = ordered[split_at:]
    return {
        "train_races": len(train),
        "train_return_rate": roi_for_rows(train),
        "train_hit_rate": pct(sum(1 for row in train if row["hit"]), len(train)),
        "test_races": len(test),
        "test_return_rate": roi_for_rows(test),
        "test_hit_rate": pct(sum(1 for row in test if row["hit"]), len(test)),
    }


def classify_ticket_strategy(
    purchase_races: int,
    hits: int,
    max_contrib: float,
    ticket_type: str,
    avg_points: float,
    time_split: dict[str, Any],
) -> str:
    _min_points, max_points = POINT_LIMITS.get(ticket_type, (1, 999))
    collapsed = (
        time_split.get("test_races", 0) >= 5
        and time_split.get("test_return_rate", 0) < 40
        and time_split.get("train_return_rate", 0) >= 150
    )
    if purchase_races >= 20 and hits >= 2 and max_contrib < 80 and avg_points <= max_points and not collapsed:
        return "正式"
    if hits >= 1 and purchase_races >= 10:
        return "参考"
    return "高リスク"


def ticket_recommendation_score(
    purchase_races: int,
    hits: int,
    payout: float,
    stake: float,
    max_contrib: float,
    ticket_type: str,
    avg_points: float,
    risk: str,
) -> float:
    roi = pct(payout, stake)
    hit_rate = pct(hits, purchase_races)
    sample_score = min(28.0, purchase_races * 0.9)
    roi_score = min(38.0, max(0.0, roi - 80.0) * 0.35)
    hit_score = min(22.0, hit_rate * (0.45 if ticket_type in {"単勝", "馬単", "三連複", "三連単"} else 0.7))
    point_penalty = max(0.0, avg_points - POINT_LIMITS.get(ticket_type, (1, 99))[1]) * 8
    dependency_penalty = max(0.0, max_contrib - 45.0) * 0.45
    risk_penalty = {"正式": 0.0, "参考": 10.0, "高リスク": 22.0}.get(risk, 12.0)
    return max(0.0, min(100.0, sample_score + roi_score + hit_score - point_penalty - dependency_penalty - risk_penalty))


def roi_for_rows(rows: list[dict[str, Any]]) -> float:
    stake = sum(float(row.get("stake", 0) or 0) for row in rows)
    payout = sum(float(row.get("payout", 0) or 0) for row in rows)
    return pct(payout, stake)


def stars_for_ticket_score(score: float) -> str:
    if score >= 80:
        return "★★★★★"
    if score >= 65:
        return "★★★★☆"
    if score >= 50:
        return "★★★☆☆"
    if score >= 35:
        return "★★☆☆☆"
    return "★☆☆☆☆"


def dependency_label_ja(value: float) -> str:
    if value >= 80:
        return "一発依存大"
    if value >= 60:
        return "一発依存"
    if value >= 40:
        return "注意"
    return "分散"


def pair_key(values: Iterable[Any]) -> tuple[str, str]:
    nums = sorted((horse_no(value) for value in values if horse_no(value)), key=int)
    return tuple(nums[:2])  # type: ignore[return-value]


def trio_key(values: Iterable[Any]) -> tuple[str, str, str]:
    nums = sorted((horse_no(value) for value in values if horse_no(value)), key=int)
    return tuple(nums[:3])  # type: ignore[return-value]


def order_key(values: Iterable[Any]) -> tuple[str, ...]:
    return tuple(horse_no(value) for value in values if horse_no(value))


def unique_nums(values: Iterable[Any]) -> list[str]:
    out: list[str] = []
    for value in values:
        no = horse_no(value)
        if no and no not in out:
            out.append(no)
    return out


def first_existing(row: pd.Series, names: list[str]) -> Any:
    for name in names:
        if name in row and not pd.isna(row[name]):
            return row[name]
    return None


def first_column(frame: pd.DataFrame, names: list[str]) -> str | None:
    for name in names:
        if name in frame.columns:
            return name
    return None


def safe_id(value: str) -> str:
    return "".join(ch.lower() if ch.isalnum() else "_" for ch in str(value)).strip("_") or "strategy"

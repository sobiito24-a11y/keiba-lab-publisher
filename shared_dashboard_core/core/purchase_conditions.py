# -*- coding: utf-8 -*-
from __future__ import annotations

import itertools
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
WORK_ROOT = Path(__file__).resolve().parents[2] / "work"
DEFAULT_DATA_DIR = WORK_ROOT / "audit_ver20_outputs"
DEFAULT_REPORT_DIR = WORK_ROOT / "jra_betting_expectation_report"
ASSETS_ANALYSIS_DIR = PROJECT_ROOT / "assets" / "analysis"
DEFAULT_CONDITION_JSON = ASSETS_ANALYSIS_DIR / "purchase_condition_ranked.json"
LEGACY_CONDITION_JSON = DEFAULT_REPORT_DIR / "purchase_condition_ranked.json"


OFFICIAL_MIN_HORSES = 30
OFFICIAL_MIN_RACES = 20
REFERENCE_MIN_HORSES = 15
REFERENCE_MIN_RACES = 10
ABSOLUTE_MIN_HORSES = 10

ODDS_BANDS = [
    ("2倍未満", None, 2.0),
    ("2～5倍", 2.0, 5.0),
    ("5～8倍", 5.0, 8.0),
    ("8～12倍", 8.0, 12.0),
    ("12～20倍", 12.0, 20.0),
    ("20～50倍", 20.0, 50.0),
    ("50倍以上", 50.0, None),
]

POPULARITY_BANDS = [
    ("1人気", 1, 1),
    ("2～3人気", 2, 3),
    ("4～6人気", 4, 6),
    ("7～9人気", 7, 9),
    ("10人気以下", 10, None),
]

NUMERIC_BANDS = {
    "AI点": [95, 90, 85, 80, 75],
    "補正AI点": [95, 90, 85, 80, 75],
    "総合評価点": [100, 95, 90, 85, 80],
    "距離指数": [70, 60, 50, 40],
    "コース指数": [70, 60, 50, 40],
    "近3走最高": [70, 60, 50, 40],
    "平均指数": [70, 60, 50, 40],
    "最高指数": [70, 60, 50, 40],
}

SAME_FAMILY_GROUPS = {
    "ai_rank": "ai_position",
    "mark": "ai_position",
    "display_group": "ai_position",
    "ai_score": "score",
    "adjusted_ai_score": "score",
    "total_score": "score",
    "ability_value": "score",
    "odds": "market_price",
    "popularity": "market_price",
}


@dataclass(frozen=True)
class ConditionSpec:
    id: str
    label: str
    family: str
    source: str
    kind: str
    value: Any = None
    low: float | None = None
    high: float | None = None
    include_high: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "family": self.family,
            "source": self.source,
            "kind": self.kind,
            "value": self.value,
            "low": self.low,
            "high": self.high,
            "include_high": self.include_high,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "ConditionSpec":
        return cls(
            id=str(data.get("id", "")),
            label=str(data.get("label", "")),
            family=str(data.get("family", "")),
            source=str(data.get("source", "")),
            kind=str(data.get("kind", "")),
            value=data.get("value"),
            low=to_float(data.get("low")),
            high=to_float(data.get("high")),
            include_high=bool(data.get("include_high", False)),
        )


@dataclass(frozen=True)
class PurchaseConditionRecommendation:
    ticket_type: str
    stars: str
    condition_score: float
    condition_labels: list[str]
    matched_horses: list[str]
    sample_label: str
    target_horses: int
    target_races: int
    win_roi: float
    place_roi: float
    win_rate: float
    place_rate: float
    reliability: str
    horse_no: str = ""
    horse_name: str = ""
    adopted_betting_labels: list[str] = field(default_factory=list)
    recommended_ticket_types: list[str] = field(default_factory=list)
    audit: dict[str, Any] = field(default_factory=dict)


def load_jra_analysis_records(data_dir: Path = DEFAULT_DATA_DIR) -> tuple[pd.DataFrame, dict[str, Any]]:
    records_path = data_dir / "jra_folder_records.csv"
    payoff_path = data_dir / "horse_individual_records.csv"
    if not records_path.exists():
        raise FileNotFoundError(f"JRA records not found: {records_path}")

    records = pd.read_csv(records_path, encoding="utf-8-sig").copy()
    records["race_id"] = records["race_id"].astype(str)
    records["horse_no_key"] = records.apply(lambda row: horse_no(first_value(row, ["馬番", "horse_no"])), axis=1)

    if payoff_path.exists():
        payoff = pd.read_csv(payoff_path, encoding="utf-8-sig")
        if "区分" in payoff.columns:
            payoff = payoff[payoff["区分"].astype(str).eq("中央")].copy()
        payoff["race_id"] = payoff["race_id"].astype(str)
        payoff["horse_no_key"] = payoff.apply(lambda row: horse_no(first_value(row, ["馬番", "horse_no"])), axis=1)
        payoff_cols = [
            "race_id",
            "horse_no_key",
            "単勝払戻",
            "複勝払戻",
            "実際の着順",
            "結果人気",
            "結果オッズ",
        ]
        payoff = payoff[[col for col in payoff_cols if col in payoff.columns]].drop_duplicates(["race_id", "horse_no_key"])
        records = records.merge(payoff, on=["race_id", "horse_no_key"], how="left", suffixes=("", "_payoff"))

    records = enrich_analysis_records(records)
    meta = {
        "data_dir": str(data_dir),
        "records_path": str(records_path),
        "payoff_path": str(payoff_path),
        "race_count": int(records["race_id"].nunique()),
        "horse_count": int(len(records)),
        "source_columns": list(records.columns),
    }
    return records, meta


def enrich_analysis_records(records: pd.DataFrame) -> pd.DataFrame:
    frame = records.copy()
    frame["horse_no_eval"] = frame.apply(lambda row: horse_no(first_value(row, ["馬番", "horse_no", "horse_no_key"])), axis=1)
    frame["horse_name_eval"] = frame.apply(lambda row: clean_text(first_value(row, ["馬名", "horse_name"])), axis=1)
    frame["ai_rank_eval"] = frame.apply(lambda row: first_number(row, ["AI順位", "ai_rank", "AI点順位"]), axis=1)
    frame["ai_score_eval"] = frame.apply(lambda row: first_number(row, ["AI点", "ai_score"]), axis=1)
    frame["adjusted_ai_score_eval"] = frame.apply(lambda row: first_number(row, ["補正AI点", "ai_score"]), axis=1)
    frame["total_score_eval"] = frame.apply(lambda row: first_number(row, ["総合評価点", "総合評価", "total_score"]), axis=1)
    frame["ability_value_eval"] = frame.apply(lambda row: first_number(row, ["能力評価値", "ability_display_score", "_raw_score", "raw_score"]), axis=1)
    frame["odds_eval"] = frame.apply(lambda row: first_number(row, ["オッズ", "単勝オッズ", "odds"]), axis=1)
    frame["popularity_eval"] = frame.apply(lambda row: first_number(row, ["人気", "popularity"]), axis=1)
    frame["finish_eval"] = frame.apply(lambda row: first_number(row, ["実際の着順", "finish"]), axis=1)
    frame["result_odds_eval"] = frame.apply(lambda row: first_number(row, ["結果オッズ", "result_odds"]), axis=1)
    frame["win_payoff_eval"] = frame.apply(lambda row: first_number(row, ["単勝払戻"]), axis=1).fillna(0)
    frame["place_payoff_eval"] = frame.apply(lambda row: first_number(row, ["複勝払戻"]), axis=1).fillna(0)
    frame["mark_eval"] = frame.apply(lambda row: normalize_mark(first_value(row, ["表示印", "最終印", "mark", "印"])), axis=1)
    frame["display_group_eval"] = frame.apply(display_group_from_row, axis=1)
    frame["jockey_changed_eval"] = frame.apply(jockey_changed_from_row, axis=1)
    frame["star_available_eval"] = frame.apply(lambda row: first_number(row, ["★最高指数", "star_max_index", "★最高"]) is not None, axis=1)
    for column in ["距離指数", "コース指数", "近3走最高", "平均指数", "最高指数", "3走前", "2走前", "前走"]:
        if column in frame.columns:
            frame[f"{column}_num"] = pd.to_numeric(frame[column], errors="coerce")
    frame["split_order"] = frame["race_id"].astype(str)
    return frame


def build_condition_catalog(records: pd.DataFrame) -> tuple[list[ConditionSpec], list[str]]:
    excluded: list[str] = []
    specs: list[ConditionSpec] = []

    def add(spec: ConditionSpec) -> None:
        mask = condition_mask(records, spec)
        if int(mask.sum()) >= ABSOLUTE_MIN_HORSES:
            specs.append(spec)

    for rank in [1, 2, 3, 4, 5]:
        add(ConditionSpec(f"ai_rank_{rank}", f"AI順位{rank}位", "ai_rank", "AI順位", "eq", rank))
    add(ConditionSpec("ai_rank_1_3", "AI順位1～3位", "ai_rank", "AI順位", "range", low=1, high=3, include_high=True))
    add(ConditionSpec("ai_rank_4_6", "AI順位4～6位", "ai_rank", "AI順位", "range", low=4, high=6, include_high=True))

    for group in ["SS", "A", "B", "C", "Z"]:
        add(ConditionSpec(f"group_{group.lower()}", f"勢力図{group}", "display_group", "display_group", "eq", group))
    for mark in ["◎", "○", "▲", "△", "☆", "✓"]:
        add(ConditionSpec(f"mark_{safe_id(mark)}", f"印{mark}", "mark", "最終印", "eq", mark))

    for label, low, high in ODDS_BANDS:
        add(ConditionSpec(f"odds_{safe_id(label)}", f"オッズ{label}", "odds", "オッズ", "range", low=low, high=high))
    for label, low, high in POPULARITY_BANDS:
        add(ConditionSpec(f"pop_{safe_id(label)}", label, "popularity", "人気", "range", low=low, high=high, include_high=True))

    numeric_sources = {
        "ai_score": ("AI点", "ai_score_eval"),
        "adjusted_ai_score": ("補正AI点", "adjusted_ai_score_eval"),
        "total_score": ("総合評価点", "total_score_eval"),
        "ability_value": ("能力評価値", "ability_value_eval"),
        "distance_index": ("距離指数", "距離指数_num"),
        "course_index": ("コース指数", "コース指数_num"),
        "recent_high": ("近3走最高", "近3走最高_num"),
        "avg_index": ("平均指数", "平均指数_num"),
        "year_high": ("最高指数", "最高指数_num"),
    }
    for family, (label, column) in numeric_sources.items():
        if column not in records.columns or records[column].notna().sum() < ABSOLUTE_MIN_HORSES:
            continue
        thresholds = numeric_thresholds(records[column], label)
        for low, high, band_label in thresholds:
            add(ConditionSpec(f"{family}_{safe_id(band_label)}", f"{label}{band_label}", family, label, "range", low=low, high=high))

    categorical_sources = [
        ("style", "脚質", ["脚質", "style"]),
        ("ability_label", "能力", ["能力"]),
        ("aptitude", "適性", ["適性"]),
        ("momentum_label", "勢い", ["勢い"]),
        ("horse_type", "馬タイプ", ["馬タイプ"]),
        ("class_basis", "クラス根拠", ["クラス根拠"]),
        ("buy_judgement", "購入判定", ["購入判定"]),
        ("training_mark", "調教評価", ["_調教評価記号", "追切評価", "調教評価"]),
        ("jockey_change", "騎乗", ["_jockey_changed"]),
    ]
    for family, label, columns in categorical_sources:
        values, source_column = categorical_values(records, columns)
        for value, count in values.items():
            if count >= ABSOLUTE_MIN_HORSES:
                add(ConditionSpec(f"{family}_{safe_id(value)}", f"{label}:{value}", family, source_column or label, "eq", value))

    if "★最高指数" in records.columns and records["★最高指数"].notna().sum() < ABSOLUTE_MIN_HORSES:
        excluded.append("★最高指数: 保存CSVで値が不足しているため探索対象外")
    else:
        excluded.append("★最高指数: 今回は保存データの欠損状況を優先し探索対象外")

    unique: dict[str, ConditionSpec] = {}
    for spec in specs:
        unique.setdefault(spec.id, spec)
    return list(unique.values()), excluded


def search_purchase_conditions(
    records: pd.DataFrame,
    *,
    max_conditions: int = 5,
    beam_size: int = 80,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    specs, excluded_features = build_condition_catalog(records)
    masks = {spec.id: condition_mask(records, spec) for spec in specs}
    spec_by_id = {spec.id: spec for spec in specs}
    rows: list[dict[str, Any]] = []
    explored = 0
    previous_layer: list[tuple[str, ...]] = []

    for spec in specs:
        combo = (spec.id,)
        row = evaluate_condition_combo(records, combo, spec_by_id, masks)
        explored += 1
        if row:
            rows.append(row)
            previous_layer.append(combo)

    previous_layer = top_combo_ids(rows, 1, beam_size)
    extension_ids = {item for combo in previous_layer[:beam_size] for item in combo}
    extension_specs = [spec for spec in specs if spec.id in extension_ids] or specs[:beam_size]
    for size in range(2, max_conditions + 1):
        next_rows: list[dict[str, Any]] = []
        seen: set[tuple[str, ...]] = set()
        source_layer = previous_layer if previous_layer else list(itertools.combinations([s.id for s in specs], size - 1))
        for base in source_layer:
            base_set = set(base)
            for spec in extension_specs:
                if spec.id in base_set:
                    continue
                combo = tuple(sorted((*base, spec.id)))
                if combo in seen or len(combo) != size:
                    continue
                seen.add(combo)
                combo_specs = [spec_by_id[item] for item in combo]
                if has_conflict(combo_specs):
                    continue
                row = evaluate_condition_combo(records, combo, spec_by_id, masks)
                explored += 1
                if row:
                    rows.append(row)
                    next_rows.append(row)
        layer_limit = beam_size if size <= 3 else max(30, beam_size // 2)
        previous_layer = top_combo_ids(next_rows, size, layer_limit)

    all_frame = pd.DataFrame(rows)
    if all_frame.empty:
        empty = pd.DataFrame()
        return empty, empty, empty, empty, {"explored_conditions": explored, "excluded_features": excluded_features, "used_features": []}

    all_frame = all_frame.sort_values(
        ["condition_score", "条件数", "単勝回収率", "複勝回収率", "該当馬数"],
        ascending=[False, True, False, False, False],
    ).reset_index(drop=True)
    all_frame = deduplicate_similar_conditions(all_frame)
    official = all_frame[all_frame["ranking_type"].eq("正式")].copy()
    reference = all_frame[all_frame["ranking_type"].eq("参考")].copy()
    avoid = build_avoid_conditions(all_frame)
    meta = {
        "explored_conditions": explored,
        "condition_specs": len(specs),
        "used_features": sorted({spec.source for spec in specs}),
        "excluded_features": excluded_features,
    }
    return all_frame, official, reference, avoid, meta


def evaluate_condition_combo(
    records: pd.DataFrame,
    combo: tuple[str, ...],
    spec_by_id: dict[str, ConditionSpec],
    masks: dict[str, pd.Series],
) -> dict[str, Any] | None:
    mask = pd.Series(True, index=records.index)
    for spec_id in combo:
        mask &= masks[spec_id]
    horse_count = int(mask.sum())
    if horse_count < ABSOLUTE_MIN_HORSES:
        return None
    race_count = int(records.loc[mask, "race_id"].nunique())
    ranking_type = ranking_type_for(horse_count, race_count)
    if ranking_type == "除外":
        return None

    group = records.loc[mask].copy()
    specs = [spec_by_id[item] for item in combo]
    labels = [spec.label for spec in specs]
    stats = horse_stats(group)
    train_stats, test_stats = time_split_stats(records, mask)
    score = condition_score(stats, train_stats, test_stats, len(combo), ranking_type)
    row = {
        "条件数": len(combo),
        "条件ID": " | ".join(combo),
        "条件内容": " × ".join(labels),
        "conditions": [spec.to_dict() for spec in specs],
        "ranking_type": ranking_type,
        "condition_score": round(score, 1),
        "評価": stars_for_score(score),
        **stats,
        "探索期間_該当馬数": train_stats["該当馬数"],
        "探索期間_単勝回収率": train_stats["単勝回収率"],
        "探索期間_複勝回収率": train_stats["複勝回収率"],
        "検証期間_該当馬数": test_stats["該当馬数"],
        "検証期間_単勝回収率": test_stats["単勝回収率"],
        "検証期間_複勝回収率": test_stats["複勝回収率"],
        "検証メモ": "検証件数不足" if test_stats["該当馬数"] < 5 else "",
        "match_signature": match_signature(group),
    }
    return row


def deduplicate_similar_conditions(frame: pd.DataFrame) -> pd.DataFrame:
    if frame.empty or "match_signature" not in frame.columns:
        return frame
    kept: list[int] = []
    seen: set[str] = set()
    ordered = frame.sort_values(
        ["condition_score", "条件数", "単勝回収率", "複勝回収率", "該当馬数"],
        ascending=[False, True, False, False, False],
    )
    for idx, row in ordered.iterrows():
        signature = str(row.get("match_signature", ""))
        if signature and signature in seen:
            continue
        if signature:
            seen.add(signature)
        kept.append(idx)
    return frame.loc[kept].reset_index(drop=True)


def match_signature(group: pd.DataFrame) -> str:
    keys = [
        f"{row.race_id}:{row.horse_no_eval}"
        for row in group[["race_id", "horse_no_eval"]].sort_values(["race_id", "horse_no_eval"]).itertuples(index=False)
    ]
    return "|".join(keys)


def horse_stats(group: pd.DataFrame) -> dict[str, Any]:
    ordered = group.sort_values(["split_order", "race_id", "horse_no_eval"]).copy()
    n = int(len(ordered))
    race_count = int(ordered["race_id"].nunique())
    finish = pd.to_numeric(ordered["finish_eval"], errors="coerce")
    wins = int(finish.eq(1).sum())
    seconds = int(finish.between(1, 2).sum())
    places = int(finish.between(1, 3).sum())
    win_payoff = pd.to_numeric(ordered["win_payoff_eval"], errors="coerce").fillna(0)
    place_payoff = pd.to_numeric(ordered["place_payoff_eval"], errors="coerce").fillna(0)
    win_stake = n * 100
    place_stake = n * 100
    win_total = float(win_payoff.sum())
    place_total = float(place_payoff.sum())
    return {
        "該当馬数": n,
        "該当レース数": race_count,
        "勝数": wins,
        "連対数": seconds,
        "複勝数": places,
        "勝率": pct(wins, n),
        "連対率": pct(seconds, n),
        "複勝率": pct(places, n),
        "単勝投資額": win_stake,
        "単勝払戻額": round(win_total, 1),
        "単勝回収率": pct(win_total, win_stake),
        "複勝投資額": place_stake,
        "複勝払戻額": round(place_total, 1),
        "複勝回収率": pct(place_total, place_stake),
        "平均オッズ": round(float(pd.to_numeric(ordered["odds_eval"], errors="coerce").mean()), 1) if ordered["odds_eval"].notna().any() else None,
        "中央値オッズ": round(float(pd.to_numeric(ordered["odds_eval"], errors="coerce").median()), 1) if ordered["odds_eval"].notna().any() else None,
        "平均人気": round(float(pd.to_numeric(ordered["popularity_eval"], errors="coerce").mean()), 1) if ordered["popularity_eval"].notna().any() else None,
        "最大連敗": max_losing_streak(win_payoff.gt(0).tolist()),
        "最大ドローダウン": round(max_drawdown((win_payoff - 100).tolist()), 1),
        "的中の偏り": dependency_label(max_contribution(win_payoff)),
        "最大単勝払戻寄与率": round(max_contribution(win_payoff), 1),
        "最大複勝払戻寄与率": round(max_contribution(place_payoff), 1),
        "特定1レースへの依存率": round(race_dependency(ordered, "win_payoff_eval"), 1),
    }


def time_split_stats(records: pd.DataFrame, mask: pd.Series) -> tuple[dict[str, Any], dict[str, Any]]:
    race_ids = sorted(records["race_id"].astype(str).unique())
    split_at = max(1, int(math.ceil(len(race_ids) * 0.7)))
    train_ids = set(race_ids[:split_at])
    test_ids = set(race_ids[split_at:])
    train = records[mask & records["race_id"].astype(str).isin(train_ids)]
    test = records[mask & records["race_id"].astype(str).isin(test_ids)]
    return compact_stats(train), compact_stats(test)


def compact_stats(group: pd.DataFrame) -> dict[str, Any]:
    if group.empty:
        return {"該当馬数": 0, "該当レース数": 0, "単勝回収率": 0.0, "複勝回収率": 0.0, "勝率": 0.0, "複勝率": 0.0}
    stats = horse_stats(group)
    return {key: stats[key] for key in ["該当馬数", "該当レース数", "単勝回収率", "複勝回収率", "勝率", "複勝率"]}


def build_avoid_conditions(all_frame: pd.DataFrame) -> pd.DataFrame:
    if all_frame.empty:
        return all_frame.copy()
    candidates = all_frame[
        (all_frame["該当馬数"].ge(OFFICIAL_MIN_HORSES))
        & (all_frame["該当レース数"].ge(OFFICIAL_MIN_RACES))
        & ((all_frame["単勝回収率"].lt(60)) | (all_frame["複勝回収率"].lt(60)))
    ].copy()
    if candidates.empty:
        return candidates
    candidates["回避理由"] = candidates.apply(
        lambda row: "単勝・複勝とも低回収" if row["単勝回収率"] < 60 and row["複勝回収率"] < 60 else ("単勝回収率が低い" if row["単勝回収率"] < 60 else "複勝回収率が低い"),
        axis=1,
    )
    return candidates.sort_values(["単勝回収率", "複勝回収率", "該当馬数"], ascending=[True, True, False]).reset_index(drop=True)


def build_ticket_strategy_detailed(records: pd.DataFrame, data_dir: Path = DEFAULT_DATA_DIR) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    single_selectors = {
        "AI1位": records["ai_rank_eval"].eq(1),
        "AI2位": records["ai_rank_eval"].eq(2),
        "AI3位": records["ai_rank_eval"].eq(3),
        "SS": records["display_group_eval"].eq("SS"),
        "A": records["display_group_eval"].eq("A"),
        "B": records["display_group_eval"].eq("B"),
        "C": records["display_group_eval"].eq("C"),
    }
    for label, mask in single_selectors.items():
        group = records[mask]
        if not group.empty:
            win = horse_stats(group)
            rows.append(ticket_row("単勝", label, win, source="馬単位条件"))
            place = horse_stats(group)
            rows.append(ticket_row("複勝", label, place, source="馬単位条件", roi_key="複勝回収率", payoff_key="複勝払戻額"))

    pair_specs = [
        ("SS-A", ["SS"], ["A"]),
        ("SS-B", ["SS"], ["B"]),
        ("SS-C", ["SS"], ["C"]),
        ("A-A", ["A"], ["A"]),
        ("A-B", ["A"], ["B"]),
        ("A-C", ["A"], ["C"]),
        ("AI1位-AI2位", ["AI1"], ["AI2"]),
        ("AI1位-AI3位", ["AI1"], ["AI3"]),
        ("AI2位-AI3位", ["AI2"], ["AI3"]),
    ]
    pair_detail = load_pair_detail(data_dir / "full_ticket_rank_pair_detail.csv")
    wide_detail = load_pair_detail(data_dir / "full_ticket_rank_wide_detail.csv")
    for label, left, right in pair_specs:
        pair_keys_by_race = make_pair_keys(records, left, right)
        rows.append(pair_ticket_row("馬連", label, pair_keys_by_race, pair_detail))
        rows.append(pair_ticket_row("ワイド", label, pair_keys_by_race, wide_detail))

    precomputed = data_dir / "best_hit_combinations_summary.csv"
    if precomputed.exists():
        pre = pd.read_csv(precomputed, encoding="utf-8-sig")
        if "区分" in pre.columns:
            pre = pre[pre["区分"].astype(str).eq("中央")].copy()
        for _, row in pre.iterrows():
            ticket = clean_text(row.get("カテゴリ"))
            if ticket not in {"馬単", "3連複", "3連単", "3連複F", "3連単F"}:
                continue
            rows.append(
                {
                    "券種": ticket,
                    "買い方": clean_text(row.get("組み合わせ")),
                    "対象レース数": int(to_float(row.get("購入R")) or 0),
                    "購入点数": int(to_float(row.get("投資")) or 0) // 100,
                    "投資額": float(to_float(row.get("投資")) or 0),
                    "払戻額": float(to_float(row.get("払戻")) or 0),
                    "的中数": int(to_float(row.get("的中R")) or 0),
                    "的中率": round(float(to_float(row.get("的中率")) or 0), 1),
                    "回収率": round(float(to_float(row.get("ROI")) or 0), 1),
                    "平均配当": round(float(to_float(row.get("平均配当")) or 0), 1),
                    "最大連敗": None,
                    "最大ドローダウン": None,
                    "最大払戻寄与率": None,
                    "分析元": "既存組み合わせ集計",
                }
            )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(["回収率", "的中率", "対象レース数"], ascending=[False, False, False]).reset_index(drop=True)


def load_pair_detail(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame()
    detail = pd.read_csv(path, encoding="utf-8-sig")
    if "区分" in detail.columns:
        detail = detail[detail["区分"].astype(str).eq("中央")].copy()
    detail["race_id"] = detail["race_id"].astype(str)
    detail["pair_key"] = detail.apply(lambda row: pair_key([row.get("馬番1"), row.get("馬番2")]), axis=1)
    detail["払戻"] = pd.to_numeric(detail.get("払戻"), errors="coerce").fillna(0)
    return detail


def make_pair_keys(records: pd.DataFrame, left_roles: list[str], right_roles: list[str]) -> dict[str, set[str]]:
    out: dict[str, set[str]] = {}
    for race_id, group in records.groupby("race_id"):
        left = role_numbers(group, left_roles)
        right = role_numbers(group, right_roles)
        pairs = {pair_key([a, b]) for a in left for b in right if a and b and a != b}
        if pairs:
            out[str(race_id)] = pairs
    return out


def role_numbers(group: pd.DataFrame, roles: list[str]) -> list[str]:
    nums: list[str] = []
    for role in roles:
        if role.startswith("AI"):
            rank = to_float(role.replace("AI", ""))
            if rank is None:
                continue
            subset = group[group["ai_rank_eval"].eq(rank)]
        else:
            subset = group[group["display_group_eval"].eq(role)]
        nums.extend(subset["horse_no_eval"].map(horse_no).tolist())
    return nums


def pair_ticket_row(ticket_type: str, label: str, pairs_by_race: dict[str, set[str]], detail: pd.DataFrame) -> dict[str, Any]:
    lookup = {(str(row.race_id), row.pair_key): float(row.払戻) for row in detail.itertuples(index=False)} if not detail.empty else {}
    race_rows = []
    for race_id, pairs in pairs_by_race.items():
        payout = sum(lookup.get((str(race_id), key), 0.0) for key in pairs)
        race_rows.append({"race_id": race_id, "points": len(pairs), "payout": payout, "hit": payout > 0})
    points = sum(row["points"] for row in race_rows)
    payout_total = sum(row["payout"] for row in race_rows)
    stake = points * 100
    hit_count = sum(1 for row in race_rows if row["hit"])
    profits = [row["payout"] - row["points"] * 100 for row in sorted(race_rows, key=lambda item: str(item["race_id"]))]
    max_payout = max([row["payout"] for row in race_rows] or [0])
    return {
        "券種": ticket_type,
        "買い方": label,
        "対象レース数": len(race_rows),
        "購入点数": points,
        "投資額": stake,
        "払戻額": round(payout_total, 1),
        "的中数": hit_count,
        "的中率": pct(hit_count, len(race_rows)),
        "回収率": pct(payout_total, stake),
        "平均配当": round(payout_total / hit_count, 1) if hit_count else 0.0,
        "最大連敗": max_losing_streak([row["hit"] for row in sorted(race_rows, key=lambda item: str(item["race_id"]))]),
        "最大ドローダウン": round(max_drawdown(profits), 1),
        "最大払戻寄与率": pct(max_payout, payout_total) if payout_total else 0.0,
        "分析元": "組み合わせdetail",
    }


def ticket_row(ticket_type: str, label: str, stats: dict[str, Any], *, source: str, roi_key: str = "単勝回収率", payoff_key: str = "単勝払戻額") -> dict[str, Any]:
    return {
        "券種": ticket_type,
        "買い方": label,
        "対象レース数": stats["該当レース数"],
        "購入点数": stats["該当馬数"],
        "投資額": stats["該当馬数"] * 100,
        "払戻額": stats[payoff_key],
        "的中数": stats["勝数"] if ticket_type == "単勝" else stats["複勝数"],
        "的中率": stats["勝率"] if ticket_type == "単勝" else stats["複勝率"],
        "回収率": stats[roi_key],
        "平均配当": round(stats[payoff_key] / max(1, stats["勝数"] if ticket_type == "単勝" else stats["複勝数"]), 1),
        "最大連敗": stats["最大連敗"],
        "最大ドローダウン": stats["最大ドローダウン"],
        "最大払戻寄与率": stats["最大単勝払戻寄与率"] if ticket_type == "単勝" else stats["最大複勝払戻寄与率"],
        "分析元": source,
    }


def write_condition_outputs(
    records: pd.DataFrame,
    all_frame: pd.DataFrame,
    official: pd.DataFrame,
    reference: pd.DataFrame,
    avoid: pd.DataFrame,
    time_split: pd.DataFrame,
    ticket_detail: pd.DataFrame,
    meta: dict[str, Any],
    out_dir: Path = DEFAULT_REPORT_DIR,
) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "all": out_dir / "purchase_condition_all.csv",
        "ranked": out_dir / "purchase_condition_ranked.csv",
        "reference": out_dir / "purchase_condition_reference.csv",
        "avoid": out_dir / "purchase_condition_avoid.csv",
        "time_split": out_dir / "purchase_condition_time_split.csv",
        "ticket": out_dir / "ticket_strategy_detailed.csv",
        "json": out_dir / "purchase_condition_ranked.json",
        "markdown": out_dir / "jra_purchase_condition_report.md",
    }
    csv_columns = [col for col in all_frame.columns if col not in {"conditions", "match_signature"}]
    all_frame[csv_columns].to_csv(paths["all"], index=False, encoding="utf-8-sig")
    official[csv_columns].to_csv(paths["ranked"], index=False, encoding="utf-8-sig")
    reference[csv_columns].to_csv(paths["reference"], index=False, encoding="utf-8-sig")
    avoid[[col for col in avoid.columns if col not in {"conditions", "match_signature"}]].to_csv(paths["avoid"], index=False, encoding="utf-8-sig")
    time_split.to_csv(paths["time_split"], index=False, encoding="utf-8-sig")
    ticket_detail.to_csv(paths["ticket"], index=False, encoding="utf-8-sig")
    payload = build_condition_json_payload(official, reference, meta)
    paths["json"].write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    paths["markdown"].write_text(build_purchase_condition_report(records, all_frame, official, reference, avoid, ticket_detail, meta), encoding="utf-8")
    return paths


def build_condition_json_payload(official: pd.DataFrame, reference: pd.DataFrame, meta: dict[str, Any]) -> dict[str, Any]:
    combined = pd.concat([official.head(30), reference.head(30)], ignore_index=True)
    entries: list[dict[str, Any]] = []
    for _, row in combined.iterrows():
        score = float(row.get("condition_score", 0) or 0)
        if score < 50:
            continue
        entries.append(
            {
                "ticket_type": "単勝候補" if row.get("単勝回収率", 0) >= row.get("複勝回収率", 0) else "複勝候補",
                "stars": stars_for_score(score),
                "condition_score": round(score, 1),
                "ranking_type": row.get("ranking_type", ""),
                "condition_labels": str(row.get("条件内容", "")).split(" × ") if row.get("条件内容") else [],
                "conditions": row.get("conditions", []),
                "target_horses": int(row.get("該当馬数", 0) or 0),
                "target_races": int(row.get("該当レース数", 0) or 0),
                "win_rate": float(row.get("勝率", 0) or 0),
                "place_rate": float(row.get("複勝率", 0) or 0),
                "win_roi": float(row.get("単勝回収率", 0) or 0),
                "place_roi": float(row.get("複勝回収率", 0) or 0),
                "max_win_contribution": float(row.get("最大単勝払戻寄与率", 0) or 0),
                "validation_note": row.get("検証メモ", ""),
            }
        )
    return {
        "version": 1,
        "scope": "jra",
        "note": "49R時点の暫定検証結果。AI予想ロジックには使用しません。",
        "meta": meta,
        "recommendations": entries,
    }


def build_purchase_condition_report(
    records: pd.DataFrame,
    all_frame: pd.DataFrame,
    official: pd.DataFrame,
    reference: pd.DataFrame,
    avoid: pd.DataFrame,
    ticket_detail: pd.DataFrame,
    meta: dict[str, Any],
) -> str:
    lines = [
        "# JRA購入条件探索レポート",
        "",
        "このレポートは保存済み中央競馬データだけを使った暫定検証です。AI点・印・買い目ロジックは変更していません。",
        "",
        "## データ件数",
        f"- 対象レース数: {meta.get('race_count')}R",
        f"- 対象馬数: {meta.get('horse_count')}頭",
        f"- 探索条件総数: {meta.get('explored_conditions')}",
        f"- 条件候補数: {meta.get('condition_specs')}",
        "",
        "## 使用特徴量",
        ", ".join(meta.get("used_features", [])) or "-",
        "",
        "## 除外特徴量",
        *[f"- {item}" for item in meta.get("excluded_features", [])],
        "",
        "## 最低サンプル基準",
        f"- 正式ランキング: {OFFICIAL_MIN_HORSES}頭以上、{OFFICIAL_MIN_RACES}R以上",
        f"- 参考ランキング: {REFERENCE_MIN_HORSES}頭以上、{REFERENCE_MIN_RACES}R以上",
        "- 10頭未満はランキング対象外",
        "",
        "## 再現性スコア定義",
        "サンプル数、対象レース数、単勝/複勝回収率、勝率、複勝率、最大払戻寄与率、最大連敗、最大ドローダウン、検証期間成績を合成した100点満点の表示用スコアです。AI予想には使用しません。",
        "",
        "## 一発依存判定",
        "最大払戻寄与率が40%未満は比較的分散、40～60%は注意、60%以上は一発依存として評価を下げています。",
        "",
        "## 買う条件ランキング（正式）",
        markdown_table(official.head(20), ["評価", "条件内容", "該当馬数", "該当レース数", "勝率", "複勝率", "単勝回収率", "複勝回収率", "condition_score", "最大単勝払戻寄与率"]),
        "",
        "## 買う条件ランキング（参考）",
        markdown_table(reference.head(20), ["評価", "条件内容", "該当馬数", "該当レース数", "勝率", "複勝率", "単勝回収率", "複勝回収率", "condition_score", "検証メモ"]),
        "",
        "## 買わない条件ランキング",
        markdown_table(avoid.head(20), ["条件内容", "該当馬数", "該当レース数", "勝率", "複勝率", "単勝回収率", "複勝回収率", "回避理由"]),
        "",
        "## 馬券種別結果",
        markdown_table(ticket_detail.head(30), ["券種", "買い方", "対象レース数", "購入点数", "的中率", "回収率", "平均配当", "最大払戻寄与率"]),
        "",
        "## 注意",
        "49R・641頭のため、4条件・5条件は参考値です。今後100R、300R、500R、1000Rと増えるほど精度が上がる設計です。",
    ]
    return "\n".join(lines)


def markdown_table(frame: pd.DataFrame, columns: list[str]) -> str:
    if frame is None or frame.empty:
        return "該当なし"
    cols = [col for col in columns if col in frame.columns]
    if not cols:
        return "該当なし"
    lines = [
        "| " + " | ".join(cols) + " |",
        "| " + " | ".join(["---"] * len(cols)) + " |",
    ]
    for _, row in frame[cols].iterrows():
        values = [clean_text(row.get(col)).replace("|", "/") for col in cols]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def condition_mask(records: pd.DataFrame, spec: ConditionSpec) -> pd.Series:
    series = feature_series(records, spec)
    if spec.kind == "eq":
        numeric_value = to_float(spec.value)
        if numeric_value is not None and pd.to_numeric(series, errors="coerce").notna().any():
            return pd.to_numeric(series, errors="coerce").eq(numeric_value).fillna(False)
        return series.astype(str).eq(str(spec.value))
    if spec.kind == "range":
        numeric = pd.to_numeric(series, errors="coerce")
        mask = pd.Series(True, index=records.index)
        if spec.low is not None:
            mask &= numeric.ge(spec.low)
        if spec.high is not None:
            mask &= numeric.le(spec.high) if spec.include_high else numeric.lt(spec.high)
        return mask.fillna(False)
    if spec.kind == "notna":
        return series.notna()
    return pd.Series(False, index=records.index)


def feature_series(records: pd.DataFrame, spec: ConditionSpec) -> pd.Series:
    if spec.family == "ai_rank":
        return records["ai_rank_eval"]
    if spec.family == "mark":
        return records["mark_eval"]
    if spec.family == "display_group":
        return records["display_group_eval"]
    if spec.family == "odds":
        return records["odds_eval"]
    if spec.family == "popularity":
        return records["popularity_eval"]
    if spec.family == "ai_score":
        return records["ai_score_eval"]
    if spec.family == "adjusted_ai_score":
        return records["adjusted_ai_score_eval"]
    if spec.family == "total_score":
        return records["total_score_eval"]
    if spec.family == "ability_value":
        return records["ability_value_eval"]
    if spec.family == "distance_index":
        return records.get("距離指数_num", pd.Series(index=records.index, dtype=float))
    if spec.family == "course_index":
        return records.get("コース指数_num", pd.Series(index=records.index, dtype=float))
    if spec.family == "recent_high":
        return records.get("近3走最高_num", pd.Series(index=records.index, dtype=float))
    if spec.family == "avg_index":
        return records.get("平均指数_num", pd.Series(index=records.index, dtype=float))
    if spec.family == "year_high":
        return records.get("最高指数_num", pd.Series(index=records.index, dtype=float))
    if spec.family == "jockey_change":
        return records["jockey_changed_eval"].map({True: "乗り替わり", False: "継続騎乗"})
    for column in [spec.source, spec.family]:
        if column in records.columns:
            return records[column].map(clean_text)
    return pd.Series("", index=records.index)


def has_conflict(specs: list[ConditionSpec]) -> bool:
    families: set[str] = set()
    for spec in specs:
        grouped = SAME_FAMILY_GROUPS.get(spec.family, spec.family)
        if grouped in families:
            return True
        families.add(grouped)
    return False


def ranking_type_for(horse_count: int, race_count: int) -> str:
    if horse_count >= OFFICIAL_MIN_HORSES and race_count >= OFFICIAL_MIN_RACES:
        return "正式"
    if horse_count >= REFERENCE_MIN_HORSES and race_count >= REFERENCE_MIN_RACES:
        return "参考"
    return "除外"


def condition_score(stats: dict[str, Any], train_stats: dict[str, Any], test_stats: dict[str, Any], condition_count: int, ranking_type: str) -> float:
    sample_score = min(25.0, stats["該当馬数"] / OFFICIAL_MIN_HORSES * 15.0 + stats["該当レース数"] / OFFICIAL_MIN_RACES * 10.0)
    roi_score = min(30.0, max(0.0, (max(stats["単勝回収率"], stats["複勝回収率"]) - 70.0) / 3.0))
    hit_score = min(20.0, stats["勝率"] * 0.45 + stats["複勝率"] * 0.25)
    dependency_penalty = 0.0
    max_contrib = max(stats["最大単勝払戻寄与率"], stats["最大複勝払戻寄与率"])
    if max_contrib >= 80:
        dependency_penalty += 30
    elif max_contrib >= 60:
        dependency_penalty += 20
    elif max_contrib >= 40:
        dependency_penalty += 10
    losing_penalty = min(12.0, stats["最大連敗"] * 0.8)
    complexity_penalty = max(0, condition_count - 3) * 4
    if ranking_type == "参考":
        dependency_penalty += 8
    validation_bonus = 0.0
    if test_stats["該当馬数"] >= 5:
        if max(test_stats["単勝回収率"], test_stats["複勝回収率"]) >= 90:
            validation_bonus += 8
        if test_stats["複勝率"] >= 25:
            validation_bonus += 4
    else:
        dependency_penalty += 5
    return max(0.0, min(100.0, sample_score + roi_score + hit_score + validation_bonus - dependency_penalty - losing_penalty - complexity_penalty))


def top_combo_ids(rows: list[dict[str, Any]], size: int, limit: int) -> list[tuple[str, ...]]:
    selected = [row for row in rows if row.get("条件数") == size]
    selected.sort(key=lambda row: (row.get("condition_score", 0), row.get("単勝回収率", 0), row.get("複勝回収率", 0)), reverse=True)
    combos: list[tuple[str, ...]] = []
    for row in selected[:limit]:
        combos.append(tuple(str(row["条件ID"]).split(" | ")))
    return combos


def numeric_thresholds(series: pd.Series, label: str) -> list[tuple[float | None, float | None, str]]:
    clean = pd.to_numeric(series, errors="coerce").dropna()
    if clean.empty:
        return []
    if label in NUMERIC_BANDS:
        thresholds = NUMERIC_BANDS[label]
        bands: list[tuple[float | None, float | None, str]] = []
        prev: float | None = None
        for threshold in thresholds:
            if prev is None:
                bands.append((threshold, None, f"{threshold}以上"))
            else:
                bands.append((threshold, prev, f"{threshold}～{prev - 0.1:.1f}"))
            prev = threshold
        bands.append((None, prev, f"{prev}未満"))
        return bands
    quantiles = clean.quantile([0.25, 0.5, 0.75]).round(1).drop_duplicates().tolist()
    if len(quantiles) < 2:
        return []
    q1, q2, q3 = quantiles[0], quantiles[len(quantiles) // 2], quantiles[-1]
    return [
        (q3, None, f"{q3}以上"),
        (q2, q3, f"{q2}～{q3 - 0.1:.1f}"),
        (q1, q2, f"{q1}～{q2 - 0.1:.1f}"),
        (None, q1, f"{q1}未満"),
    ]


def categorical_values(records: pd.DataFrame, columns: list[str]) -> tuple[dict[str, int], str]:
    series = None
    source_column = ""
    for column in columns:
        if column in records.columns:
            series = records[column]
            source_column = column
            break
    if series is None:
        return {}, source_column
    if columns == ["_jockey_changed"]:
        series = records["jockey_changed_eval"].map({True: "乗り替わり", False: "継続騎乗"})
        source_column = "_jockey_changed"
    values = series.map(clean_text)
    values = values[~values.isin(["", "-", "nan", "None"])]
    return values.value_counts().head(12).to_dict(), source_column


def display_group_from_row(row: pd.Series) -> str:
    group = clean_text(first_value(row, ["display_group", "勢力図グループ", "グループ"]))
    if group in {"SS", "A", "B", "C", "Z"}:
        return group
    mark = normalize_mark(first_value(row, ["表示印", "最終印", "mark", "印"]))
    if mark == "◎":
        return "SS"
    if mark in {"○", "▲"}:
        return "A"
    if mark == "△":
        return "B"
    if mark in {"✓", "☆"}:
        return "C"
    return "Z"


def normalize_mark(value: Any) -> str:
    text = clean_text(value)
    replacements = {
        "БЭ": "◎",
        "БЫ": "○",
        "Бг": "▲",
        "Бв": "△",
        "БЩ": "☆",
        "✔": "✓",
    }
    return replacements.get(text, text)


def jockey_changed_from_row(row: pd.Series) -> bool:
    value = first_value(row, ["_jockey_changed", "jockey_changed", "乗り替わり"])
    text = clean_text(value).lower()
    if text in {"true", "1", "yes", "乗り替わり"}:
        return True
    return False


def max_losing_streak(hits: Iterable[bool]) -> int:
    longest = 0
    current = 0
    for hit in hits:
        if hit:
            longest = max(longest, current)
            current = 0
        else:
            current += 1
    return max(longest, current)


def max_drawdown(profits: Iterable[float]) -> float:
    peak = 0.0
    value = 0.0
    worst = 0.0
    for profit in profits:
        value += float(profit)
        peak = max(peak, value)
        worst = min(worst, value - peak)
    return abs(worst)


def max_contribution(payoffs: pd.Series) -> float:
    pay = pd.to_numeric(payoffs, errors="coerce").fillna(0)
    total = float(pay.sum())
    if total <= 0:
        return 0.0
    return pct(float(pay.max()), total)


def race_dependency(frame: pd.DataFrame, payoff_col: str) -> float:
    pay = pd.to_numeric(frame[payoff_col], errors="coerce").fillna(0)
    total = float(pay.sum())
    if total <= 0:
        return 0.0
    by_race = frame.assign(_pay=pay).groupby("race_id")["_pay"].sum()
    return pct(float(by_race.max()), total)


def dependency_label(value: float) -> str:
    if value >= 80:
        return "一発依存大"
    if value >= 60:
        return "一発依存"
    if value >= 40:
        return "注意"
    return "分散"


def stars_for_score(score: float) -> str:
    if score >= 80:
        return "★★★★★"
    if score >= 65:
        return "★★★★☆"
    if score >= 50:
        return "★★★☆☆"
    if score >= 35:
        return "★★☆☆☆"
    return "★☆☆☆☆"


def pct(numerator: float, denominator: float) -> float:
    if not denominator:
        return 0.0
    return round(float(numerator) / float(denominator) * 100, 1)


def first_value(row: pd.Series | dict[str, Any], names: list[str]) -> Any:
    for name in names:
        if name in row and not is_missing(row[name]):
            return row[name]
    return None


def first_number(row: pd.Series | dict[str, Any], names: list[str]) -> float | None:
    return to_float(first_value(row, names))


def to_float(value: Any) -> float | None:
    if is_missing(value):
        return None
    try:
        return float(str(value).replace(",", "").replace("倍", "").replace("%", "").strip())
    except ValueError:
        return None


def is_missing(value: Any) -> bool:
    if value is None:
        return True
    try:
        if pd.isna(value):
            return True
    except (TypeError, ValueError):
        pass
    return clean_text(value).lower() in {"", "-", "—", "nan", "none", "null", "データなし"}


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    return str(value).strip()


def horse_no(value: Any) -> str:
    number = to_float(value)
    if number is None:
        return clean_text(value)
    return str(int(number))


def pair_key(values: Iterable[Any]) -> str:
    nums = sorted(horse_no(value) for value in values if horse_no(value))
    return "-".join(nums)


def safe_id(value: Any) -> str:
    text = clean_text(value)
    table = str.maketrans({
        "～": "_",
        "倍": "x",
        "以": "",
        "上": "up",
        "下": "down",
        "未": "under",
        "満": "",
        "人": "pop",
        "気": "",
        "位": "",
        "◎": "honmei",
        "○": "taikou",
        "▲": "tan",
        "△": "ren",
        "☆": "star",
        "✓": "check",
        " ": "_",
        "/": "_",
        "・": "_",
        ":": "_",
    })
    cleaned = text.translate(table)
    return "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in cleaned).strip("_") or "value"


def build_purchase_condition_recommendations(
    table: Any,
    *,
    json_path: Path = DEFAULT_CONDITION_JSON,
    max_items: int = 4,
    adopted_horse_numbers: set[str] | None = None,
    adoption_map: dict[str, list[str]] | None = None,
) -> list[PurchaseConditionRecommendation]:
    json_path = resolve_condition_json_path(json_path)
    if table is None or not isinstance(table, pd.DataFrame) or table.empty or not json_path.exists():
        return []
    try:
        payload = json.loads(json_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return []
    recommendations: list[PurchaseConditionRecommendation] = []
    current = enrich_current_table(table)
    for item in payload.get("recommendations", []):
        specs = [ConditionSpec.from_dict(spec) for spec in item.get("conditions", [])]
        if not specs:
            continue
        mask = pd.Series(True, index=current.index)
        for spec in specs:
            mask &= condition_mask(current, spec)
        matched = current[mask].copy()
        if adopted_horse_numbers is not None:
            matched = matched[matched["horse_no_eval"].astype(str).isin({str(no) for no in adopted_horse_numbers})].copy()
        if matched.empty:
            continue
        score = float(item.get("condition_score", 0) or 0)
        if score < 50:
            continue
        if adopted_horse_numbers is None:
            recommendations.append(
                build_purchase_condition_recommendation(item, matched, score)
            )
        else:
            for _, row in matched.iterrows():
                no = clean_text(row.get("horse_no_eval"))
                used_in = list((adoption_map or {}).get(no, []))
                if not used_in:
                    continue
                recommendations.append(
                    build_purchase_condition_recommendation(item, pd.DataFrame([row]), score, adopted_labels=used_in)
                )
                if len(recommendations) >= max_items:
                    break
        if len(recommendations) >= max_items:
            break
    return recommendations


def build_purchase_condition_recommendation(
    item: dict[str, Any],
    matched: pd.DataFrame,
    score: float,
    *,
    adopted_labels: list[str] | None = None,
) -> PurchaseConditionRecommendation:
    first = matched.iloc[0] if not matched.empty else pd.Series(dtype=object)
    adopted = list(adopted_labels or [])
    ticket_types = sorted({label.split()[0] for label in adopted if label.strip()})
    condition_labels = [str(label) for label in item.get("condition_labels", [])]
    return PurchaseConditionRecommendation(
        ticket_type=str(item.get("ticket_type", "購入候補")),
        stars=str(item.get("stars", stars_for_score(score))),
        condition_score=round(score, 1),
        condition_labels=condition_labels,
        matched_horses=[horse_label(row) for _, row in matched.iterrows()],
        sample_label=str(item.get("ranking_type", "暫定")),
        target_horses=int(item.get("target_horses", 0) or 0),
        target_races=int(item.get("target_races", 0) or 0),
        win_roi=float(item.get("win_roi", 0) or 0),
        place_roi=float(item.get("place_roi", 0) or 0),
        win_rate=float(item.get("win_rate", 0) or 0),
        place_rate=float(item.get("place_rate", 0) or 0),
        reliability="暫定" if item.get("ranking_type") == "正式" else "参考",
        horse_no=clean_text(first.get("horse_no_eval")),
        horse_name=clean_text(first.get("horse_name_eval")),
        adopted_betting_labels=adopted,
        recommended_ticket_types=ticket_types,
        audit={
            "conditions": condition_labels,
            "matched_horses": [horse_label(row) for _, row in matched.iterrows()],
            "adopted_betting_labels": adopted,
            "used_in_betting": bool(adopted),
        },
    )


def resolve_condition_json_path(json_path: Path) -> Path:
    if json_path.exists():
        return json_path
    if json_path == DEFAULT_CONDITION_JSON and LEGACY_CONDITION_JSON.exists():
        return LEGACY_CONDITION_JSON
    return json_path


def enrich_current_table(table: pd.DataFrame) -> pd.DataFrame:
    current = table.copy()
    current["race_id"] = current.get("race_id", "")
    current["horse_no_eval"] = current.apply(lambda row: horse_no(first_value(row, ["馬番", "horse_no", "馬", "馬番"])), axis=1)
    current["horse_name_eval"] = current.apply(lambda row: clean_text(first_value(row, ["馬名", "horse_name"])), axis=1)
    current["ai_rank_eval"] = current.apply(lambda row: first_number(row, ["AI順位", "ai_rank", "AI点順位"]), axis=1)
    current["ai_score_eval"] = current.apply(lambda row: first_number(row, ["AI点", "ai_score"]), axis=1)
    current["adjusted_ai_score_eval"] = current.apply(lambda row: first_number(row, ["補正AI点", "ai_score"]), axis=1)
    current["total_score_eval"] = current.apply(lambda row: first_number(row, ["総合評価点", "総合評価", "total_score"]), axis=1)
    current["ability_value_eval"] = current.apply(lambda row: first_number(row, ["能力評価値", "ability_display_score", "_raw_score", "raw_score"]), axis=1)
    current["odds_eval"] = current.apply(lambda row: first_number(row, ["オッズ", "単勝オッズ", "odds"]), axis=1)
    current["popularity_eval"] = current.apply(lambda row: first_number(row, ["人気", "popularity"]), axis=1)
    current["mark_eval"] = current.apply(lambda row: normalize_mark(first_value(row, ["表示印", "最終印", "mark", "印"])), axis=1)
    current["display_group_eval"] = current.apply(display_group_from_row, axis=1)
    current["jockey_changed_eval"] = current.apply(jockey_changed_from_row, axis=1)
    for column in ["距離指数", "コース指数", "近3走最高", "平均指数", "最高指数", "3走前", "2走前", "前走"]:
        if column in current.columns:
            current[f"{column}_num"] = pd.to_numeric(current[column], errors="coerce")
    return current


def horse_label(row: pd.Series) -> str:
    no = clean_text(row.get("horse_no_eval"))
    name = clean_text(row.get("horse_name_eval"))
    mark = normalize_mark(row.get("mark_eval"))
    return " ".join(part for part in [no, mark, name] if part)

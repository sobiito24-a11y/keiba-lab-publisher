from __future__ import annotations

import itertools
import json
import math
import re
from dataclasses import dataclass
from io import StringIO
from pathlib import Path
from typing import Any, Callable, Iterable

import pandas as pd

from .value_support import attach_value_signals


MARK_ORDER = ["◎", "○", "▲", "△", "☆"]
MARK_SET_SPECS: dict[str, list[str]] = {
    "◎○": ["◎", "○"],
    "◎○▲": ["◎", "○", "▲"],
    "◎○▲△": ["◎", "○", "▲", "△"],
    "◎○▲△☆": ["◎", "○", "▲", "△", "☆"],
}
PAIR_BET_TYPES = {"quinella": "馬連", "wide": "ワイド"}
TRIO_BET_TYPES = {"trio": "三連複"}


@dataclass(frozen=True)
class RaceSource:
    race_id: str
    race_type: str
    race_dir: Path
    result_path: Path


def read_text(path: Path) -> str:
    data = path.read_bytes()
    for encoding in ("utf-8", "utf-8-sig", "cp932", "euc-jp"):
        try:
            text = data.decode(encoding)
        except UnicodeDecodeError:
            continue
        if "<html" in text.lower() or "netkeiba" in text.lower():
            return text
    return data.decode("utf-8", errors="replace")


def normalize_horse_no(value: Any) -> str:
    number = to_int(value)
    return str(number) if number is not None else ""


def normalize_mark(value: Any) -> str:
    text = clean_text(value)
    if text in MARK_ORDER:
        return text
    if text in {"✓", "✔"}:
        return "☆"
    return ""


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
    return re.sub(r"\s+", " ", str(value).replace("\xa0", " ")).strip()


def to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    match = re.search(r"-?\d+(?:\.\d+)?", str(value).replace(",", ""))
    return float(match.group(0)) if match else None


def to_int(value: Any) -> int | None:
    number = to_float(value)
    if number is None:
        return None
    return int(number)


def pct(numerator: float, denominator: float) -> float:
    return round(float(numerator) / float(denominator) * 100, 1) if denominator else 0.0


def first_value(row: dict[str, Any] | pd.Series, keys: Iterable[str]) -> Any:
    for key in keys:
        if key in row:
            value = row[key]
            if clean_text(value) != "":
                return value
    return None


def first_column(df: pd.DataFrame, *needles: str) -> str | None:
    for col in df.columns:
        name = re.sub(r"\s+", "", str(col))
        if all(needle in name for needle in needles):
            return str(col)
    return None


def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if isinstance(out.columns, pd.MultiIndex):
        out.columns = [
            " ".join(str(part) for part in col if str(part) and not str(part).startswith("Unnamed")).strip()
            for col in out.columns
        ]
    else:
        out.columns = [str(col) for col in out.columns]
    return out


def parse_result_html(path: Path) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    return parse_result_html_text(read_text(path))


def parse_result_html_text(html: str) -> tuple[dict[str, dict[str, Any]], dict[str, Any]]:
    tables = [flatten_columns(table) for table in pd.read_html(StringIO(html), flavor="lxml")]
    finish = parse_finish_table(tables)
    payouts = parse_payoff_tables(tables)
    return finish, payouts


def parse_finish_table(tables: list[pd.DataFrame]) -> dict[str, dict[str, Any]]:
    result_table = find_finish_table(tables)
    if result_table is None:
        return {}
    rank_col = first_column(result_table, "着", "順") or first_column(result_table, "着") or str(result_table.columns[0])
    horse_col = first_column(result_table, "馬", "番")
    name_col = first_column(result_table, "馬名")
    pop_col = first_column(result_table, "人", "気")
    odds_col = first_column(result_table, "単勝", "オッズ")
    finish: dict[str, dict[str, Any]] = {}
    if horse_col is None:
        return finish
    for _, row in result_table.iterrows():
        no = normalize_horse_no(row.get(horse_col))
        rank = to_int(row.get(rank_col))
        if not no or rank is None:
            continue
        finish[no] = {
            "finish": rank,
            "result_name": clean_text(row.get(name_col)) if name_col else "",
            "result_popularity": to_int(row.get(pop_col)) if pop_col else None,
            "result_odds": to_float(row.get(odds_col)) if odds_col else None,
        }
    return finish


def find_finish_table(tables: list[pd.DataFrame]) -> pd.DataFrame | None:
    for table in tables:
        cols = " ".join(str(col) for col in table.columns)
        compact = re.sub(r"\s+", "", cols)
        if "着順" in compact and "馬番" in compact:
            return table
    return None


def empty_payouts() -> dict[str, Any]:
    return {
        "win": {},
        "place": {},
        "wide": {},
        "quinella": {},
        "exacta": {},
        "trio": {},
        "trifecta": {},
    }


def parse_payoff_tables(tables: list[pd.DataFrame]) -> dict[str, Any]:
    payouts = empty_payouts()
    for table in tables:
        if table.shape[1] < 3:
            continue
        for _, row in table.iterrows():
            kind = clean_text(row.iloc[0])
            combo = clean_text(row.iloc[1])
            amount = clean_text(row.iloc[2])
            nums = re.findall(r"\d+", combo)
            pays = [int(value.replace(",", "")) for value in re.findall(r"\d[\d,]*", amount)]
            if not kind or not nums or not pays:
                continue
            if "単勝" in kind:
                payouts["win"][nums[0]] = pays[0]
            elif "複勝" in kind:
                for no, pay in zip(nums, pays):
                    payouts["place"][no] = pay
            elif "ワイド" in kind:
                for pair, pay in zip(pairwise_numbers(nums), pays):
                    payouts["wide"][pair_key(pair)] = pay
            elif "馬連" in kind:
                payouts["quinella"][pair_key(nums[:2])] = pays[0]
            elif "馬単" in kind:
                payouts["exacta"][tuple(nums[:2])] = pays[0]
            elif "3連複" in kind or "三連複" in kind:
                payouts["trio"][trio_key(nums[:3])] = pays[0]
            elif "3連単" in kind or "三連単" in kind:
                payouts["trifecta"][tuple(nums[:3])] = pays[0]
    return payouts


def pairwise_numbers(nums: list[str]) -> list[tuple[str, str]]:
    return [(nums[i], nums[i + 1]) for i in range(0, len(nums) - 1, 2)]


def pair_key(nums: Iterable[Any]) -> tuple[str, str]:
    values = [normalize_horse_no(value) for value in nums]
    values = [value for value in values if value]
    return tuple(sorted(values, key=lambda item: int(item)))[:2]  # type: ignore[return-value]


def trio_key(nums: Iterable[Any]) -> tuple[str, str, str]:
    values = [normalize_horse_no(value) for value in nums]
    values = [value for value in values if value]
    return tuple(sorted(values, key=lambda item: int(item)))[:3]  # type: ignore[return-value]


def prediction_html_files(race_dir: Path, race_type: str) -> tuple[dict[str, str], dict[str, str]]:
    kinds = ["newspaper", "speed", "style", "oikiri"] if race_type == "jra" else ["newspaper", "speed", "style"]
    html_files: dict[str, str] = {}
    file_names: dict[str, str] = {}
    for kind in kinds:
        path = find_kind_file(race_dir, kind)
        if path is None:
            continue
        html_files[kind] = read_text(path)
        file_names[kind] = path.name
    return html_files, file_names


def find_kind_file(race_dir: Path, kind: str) -> Path | None:
    candidates = sorted(race_dir.glob(f"*_{kind}.html"))
    if candidates:
        return candidates[0]
    for path in sorted(race_dir.glob("*.html")):
        if path.stem.endswith(kind):
            return path
    return None


def discover_race_sources(roots: Iterable[Path], race_type: str) -> tuple[list[RaceSource], dict[str, Any]]:
    sources: list[RaceSource] = []
    seen: set[str] = set()
    duplicates = 0
    missing_prediction_pages = 0
    for root in roots:
        if root is None or not root.exists():
            continue
        for result_path in sorted(root.rglob("*_result.html")):
            race_dir = result_path.parent
            race_id = result_path.stem.removesuffix("_result")
            if not re.fullmatch(r"\d{10,12}", race_id):
                race_id = race_dir.name
            if race_id in seen:
                duplicates += 1
                continue
            html_files, _ = prediction_html_files(race_dir, race_type)
            if not html_files:
                missing_prediction_pages += 1
                continue
            seen.add(race_id)
            sources.append(RaceSource(race_id=race_id, race_type=race_type, race_dir=race_dir, result_path=result_path))
    meta = {
        "race_type": race_type,
        "roots": [str(root) for root in roots if root is not None],
        "discovered_races": len(sources),
        "duplicate_result_files_skipped": duplicates,
        "missing_prediction_pages": missing_prediction_pages,
    }
    return sources, meta


def extract_prediction_rows(result: Any, race_id: str, race_type: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    table = getattr(result, "overall_table", None)
    if table is None or getattr(table, "empty", True):
        return [], {}
    frame = table.copy()
    if "ai_rank" not in frame.columns:
        score_col = find_existing_column(frame, ["AI点", "normalized_ai_score"])
        if score_col:
            frame["ai_rank"] = pd.to_numeric(frame[score_col], errors="coerce").rank(method="first", ascending=False)
    if "ability_rank_for_backtest" not in frame.columns:
        ability_col = find_existing_column(frame, ["能力評価値", "ability_display_score", "raw_score"])
        if ability_col:
            frame["ability_rank_for_backtest"] = pd.to_numeric(frame[ability_col], errors="coerce").rank(method="first", ascending=False)
    rows: list[dict[str, Any]] = []
    race_info = dict(getattr(result, "race_info", {}) or {})
    for _, row in frame.iterrows():
        raw = row.to_dict()
        no = normalize_horse_no(first_value(raw, ["馬番", "horse_no", "鬥ｬ逡ｪ"]))
        if not no:
            continue
        rows.append(
            {
                "race_id": race_id,
                "race_type": race_type,
                "venue": infer_venue(result, race_info),
                "race_name": clean_text(getattr(result, "race_name", "")) or clean_text(race_info.get("race_name")),
                "distance": to_int(first_value(race_info, ["distance", "距離"])) or to_int(first_value(raw, ["距離"])),
                "surface": clean_text(first_value(race_info, ["surface", "course_type", "芝ダ"])) or clean_text(first_value(raw, ["芝ダ"])),
                "field_size": len(frame),
                "horse_no": no,
                "horse_name": clean_text(first_value(raw, ["馬名", "horse_name"])),
                "mark": normalize_mark(first_value(raw, ["表示印", "display_mark", "最終印", "original_mark", "旧印", "old_final_mark", "印", "mark"])),
                "raw_mark": clean_text(first_value(raw, ["表示印", "display_mark", "最終印", "original_mark", "旧印", "old_final_mark", "印", "mark"])),
                "ability_band": clean_text(first_value(raw, ["能力帯", "ability_band", "ability_rank"])),
                "ability_rank": to_int(first_value(raw, ["能力順位", "ability_rank_for_backtest"])),
                "ability_value": to_float(first_value(raw, ["能力評価値", "ability_display_score", "raw_score", "_raw_score"])),
                "ai_current_rank": to_int(first_value(raw, ["AI今回評価順位", "ai_rank", "AI順位"])),
                "ai_score": to_float(first_value(raw, ["AI点", "normalized_ai_score", "ai_score"])),
                "odds": to_float(first_value(raw, ["オッズ", "単勝オッズ", "odds"])),
                "popularity": to_int(first_value(raw, ["人気", "単勝人気", "popularity"])),
            }
        )
    return rows, race_info


def infer_venue(result: Any, race_info: dict[str, Any]) -> str:
    for key in ["venue", "racecourse", "競馬場", "場所", "開催場"]:
        text = clean_text(race_info.get(key))
        if text:
            return text
    race_name = clean_text(getattr(result, "race_name", ""))
    match = re.search(r"(札幌|函館|福島|新潟|東京|中山|中京|京都|阪神|小倉|門別|盛岡|水沢|浦和|船橋|大井|川崎|金沢|笠松|名古屋|園田|姫路|高知|佐賀)", race_name)
    return match.group(1) if match else ""


def find_existing_column(frame: pd.DataFrame, candidates: Iterable[str]) -> str | None:
    for candidate in candidates:
        if candidate in frame.columns:
            return candidate
    return None


def attach_results(prediction_rows: list[dict[str, Any]], finish: dict[str, dict[str, Any]], payouts: dict[str, Any]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for row in prediction_rows:
        out = dict(row)
        no = out.get("horse_no", "")
        result_row = finish.get(no, {})
        out["finish"] = result_row.get("finish")
        out["result_popularity"] = result_row.get("result_popularity")
        out["result_odds"] = result_row.get("result_odds")
        out["win_payoff"] = payouts.get("win", {}).get(no, 0)
        out["place_payoff"] = payouts.get("place", {}).get(no, 0)
        rows.append(out)
    return rows


def attach_value_signals_to_records(records: pd.DataFrame) -> pd.DataFrame:
    if records is None or records.empty:
        return records
    groups: list[pd.DataFrame] = []
    for _race_id, group in records.groupby("race_id", sort=False):
        race_type = clean_text(group.iloc[0].get("race_type")) or "jra"
        enriched = attach_value_signals(group.to_dict(orient="records"), race_type)
        groups.append(pd.DataFrame(enriched))
    return pd.concat(groups, ignore_index=True) if groups else records.copy()


def evaluate_mark_singles(records: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for mark in MARK_ORDER:
        targets = records[records["mark"].eq(mark)].copy()
        for bet_type, payoff_col in [("単勝", "win_payoff"), ("複勝", "place_payoff")]:
            stake = len(targets) * 100
            payout_values = pd.to_numeric(targets.get(payoff_col, pd.Series(dtype=float)), errors="coerce").fillna(0)
            hits = int((payout_values > 0).sum())
            payout = float(payout_values.sum())
            dependency = payout_dependency_metrics(payout_values, stake)
            rows.append(
                {
                    "印": mark,
                    "券種": bet_type,
                    "対象レース数": int(targets["race_id"].nunique()) if not targets.empty else 0,
                    "購入数": int(len(targets)),
                    "購入額": int(stake),
                    "的中数": hits,
                    "的中率": pct(hits, len(targets)),
                    "払戻額": int(payout),
                    "回収率": pct(payout, stake),
                    "平均払戻": round(payout / hits, 1) if hits else 0.0,
                    "最大払戻": int(payout_values.max()) if len(payout_values) else 0,
                    "最大払戻除外回収率": dependency["top1_excluded_roi"],
                    "上位2件除外回収率": dependency["top2_excluded_roi"],
                    "最大払戻依存度": dependency["max_payout_dependency"],
                    "購入参考": classify_reference(len(targets), hits, pct(payout, stake)),
                }
            )
    return pd.DataFrame(rows)


def evaluate_value_singles(records: pd.DataFrame) -> pd.DataFrame:
    if records is None or records.empty or "value_signal" not in records.columns:
        return pd.DataFrame()
    rows: list[dict[str, Any]] = []
    targets_all = records[records["value_signal"].astype(bool)].copy()
    for label, targets in [("妙味あり", targets_all), *[(f"妙味あり＋{mark}", targets_all[targets_all["mark"].eq(mark)]) for mark in MARK_ORDER]]:
        if targets.empty:
            rows.append(empty_value_summary_row(label))
            continue
        win_values = pd.to_numeric(targets.get("win_payoff", pd.Series(dtype=float)), errors="coerce").fillna(0)
        place_values = pd.to_numeric(targets.get("place_payoff", pd.Series(dtype=float)), errors="coerce").fillna(0)
        stake = len(targets) * 100
        win_hits = int((win_values > 0).sum())
        place_hits = int((place_values > 0).sum())
        win_dependency = payout_dependency_metrics(win_values, stake)
        place_dependency = payout_dependency_metrics(place_values, stake)
        rows.append(
            {
                "対象": label,
                "対象レース数": int(targets["race_id"].nunique()),
                "購入数": int(len(targets)),
                "単勝的中数": win_hits,
                "単勝的中率": pct(win_hits, len(targets)),
                "単勝購入額": int(stake),
                "単勝払戻額": int(win_values.sum()),
                "単勝回収率": pct(win_values.sum(), stake),
                "単勝最大払戻": int(win_values.max()) if len(win_values) else 0,
                "単勝最大払戻除外回収率": win_dependency["top1_excluded_roi"],
                "単勝上位2件除外回収率": win_dependency["top2_excluded_roi"],
                "複勝的中数": place_hits,
                "複勝的中率": pct(place_hits, len(targets)),
                "複勝購入額": int(stake),
                "複勝払戻額": int(place_values.sum()),
                "複勝回収率": pct(place_values.sum(), stake),
                "複勝最大払戻": int(place_values.max()) if len(place_values) else 0,
                "複勝最大払戻除外回収率": place_dependency["top1_excluded_roi"],
                "複勝上位2件除外回収率": place_dependency["top2_excluded_roi"],
                "参考区分": classify_reference(len(targets), max(win_hits, place_hits), max(pct(win_values.sum(), stake), pct(place_values.sum(), stake))),
            }
        )
    return pd.DataFrame(rows)


def empty_value_summary_row(label: str) -> dict[str, Any]:
    return {
        "対象": label,
        "対象レース数": 0,
        "購入数": 0,
        "単勝的中数": 0,
        "単勝的中率": 0.0,
        "単勝購入額": 0,
        "単勝払戻額": 0,
        "単勝回収率": 0.0,
        "単勝最大払戻": 0,
        "単勝最大払戻除外回収率": 0.0,
        "単勝上位2件除外回収率": 0.0,
        "複勝的中数": 0,
        "複勝的中率": 0.0,
        "複勝購入額": 0,
        "複勝払戻額": 0,
        "複勝回収率": 0.0,
        "複勝最大払戻": 0,
        "複勝最大払戻除外回収率": 0.0,
        "複勝上位2件除外回収率": 0.0,
        "参考区分": "サンプル不足",
    }


def payout_dependency_metrics(payout_values: pd.Series, stake: float) -> dict[str, float]:
    values = pd.to_numeric(payout_values, errors="coerce").fillna(0).sort_values(ascending=False)
    total = float(values.sum())
    top1 = float(values.iloc[0]) if len(values) else 0.0
    top2 = float(values.iloc[:2].sum()) if len(values) else 0.0
    return {
        "top1_excluded_roi": pct(total - top1, stake),
        "top2_excluded_roi": pct(total - top2, stake),
        "max_payout_dependency": round(top1 / total * 100, 1) if total else 0.0,
    }


def evaluate_box_strategies(records: pd.DataFrame, payouts_by_race: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for label, marks in MARK_SET_SPECS.items():
        for bet_type, bet_label in {**PAIR_BET_TYPES, **TRIO_BET_TYPES}.items():
            if bet_type == "trio" and len(marks) < 3:
                continue
            race_rows = []
            for race_id, group in records.groupby("race_id", sort=True):
                horses = marked_horses(group, marks)
                size = 3 if bet_type == "trio" else 2
                tickets = list(itertools.combinations(horses, size))
                if not tickets:
                    continue
                payout = payout_for_tickets(tickets, bet_type, payouts_by_race.get(str(race_id), {}))
                race_rows.append({"race_id": str(race_id), "points": len(tickets), "stake": len(tickets) * 100, "payout": payout})
            total_points = sum(row["points"] for row in race_rows)
            stake = sum(row["stake"] for row in race_rows)
            payout = sum(row["payout"] for row in race_rows)
            hits = sum(1 for row in race_rows if row["payout"] > 0)
            dependency = payout_dependency_metrics(pd.Series([row["payout"] for row in race_rows]), stake)
            rows.append(
                {
                    "対象印": label,
                    "券種": bet_label,
                    "レース数": len(race_rows),
                    "総点数": total_points,
                    "総購入額": stake,
                    "的中レース数": hits,
                    "的中率": pct(hits, len(race_rows)),
                    "総払戻額": int(payout),
                    "回収率": pct(payout, stake),
                    "平均点数": round(total_points / len(race_rows), 2) if race_rows else 0.0,
                    "最大払戻": int(max([row["payout"] for row in race_rows] or [0])),
                    "最大払戻除外回収率": dependency["top1_excluded_roi"],
                    "上位2件除外回収率": dependency["top2_excluded_roi"],
                    "最大払戻依存度": dependency["max_payout_dependency"],
                    "購入参考": classify_reference(len(race_rows), hits, pct(payout, stake)),
                }
            )
    return pd.DataFrame(rows)


def marked_horses(group: pd.DataFrame, marks: list[str]) -> list[str]:
    horses: list[str] = []
    for mark in marks:
        subset = group[group["mark"].eq(mark)].copy()
        sort_cols = [column for column in ["ai_current_rank", "horse_no"] if column in subset.columns]
        if sort_cols:
            subset = subset.sort_values(sort_cols, na_position="last")
        if subset.empty:
            continue
        no = str(subset.iloc[0].get("horse_no", ""))
        if no and no not in horses:
            horses.append(no)
    return horses


def payout_for_tickets(tickets: Iterable[tuple[str, ...]], bet_type: str, payouts: dict[str, Any]) -> float:
    total = 0.0
    payout_map = payouts.get(bet_type, {})
    for ticket in tickets:
        if bet_type in {"wide", "quinella"}:
            key = pair_key(ticket)
        elif bet_type == "trio":
            key = trio_key(ticket)
        else:
            key = tuple(str(value) for value in ticket)
        total += float(payout_map.get(key, 0) or 0)
    return total


def build_condition_summary(records: pd.DataFrame) -> pd.DataFrame:
    honmei = records[records["mark"].eq("◎")].copy()
    if honmei.empty:
        return pd.DataFrame()
    honmei["距離帯"] = honmei["distance"].apply(distance_band)
    honmei["頭数帯"] = honmei["field_size"].apply(field_size_band)
    honmei["◎単勝オッズ帯"] = honmei["odds"].apply(odds_band)
    if "◎○能力値差" in honmei.columns:
        honmei["◎○能力値差帯"] = honmei["◎○能力値差"].apply(diff_band)
    if "◎○今回評価差" in honmei.columns:
        honmei["◎○今回評価差帯"] = honmei["◎○今回評価差"].apply(diff_band)
    condition_columns = [
        ("JRA/NAR", "race_type"),
        ("競馬場", "venue"),
        ("芝/ダート", "surface"),
        ("距離帯", "距離帯"),
        ("頭数帯", "頭数帯"),
        ("◎の能力帯", "ability_band"),
        ("◎の能力順位", "ability_rank"),
        ("◎のAI今回評価順位", "ai_current_rank"),
        ("◎の単勝オッズ帯", "◎単勝オッズ帯"),
        ("◎と○の能力値差", "◎○能力値差帯"),
        ("◎と○の今回評価差", "◎○今回評価差帯"),
    ]
    rows: list[dict[str, Any]] = []
    for label, column in condition_columns:
        if column not in honmei.columns:
            continue
        for value, group in honmei.groupby(column, dropna=False):
            if clean_text(value) == "":
                value = "欠損"
            win_pay = pd.to_numeric(group["win_payoff"], errors="coerce").fillna(0)
            place_pay = pd.to_numeric(group["place_payoff"], errors="coerce").fillna(0)
            stake = len(group) * 100
            rows.append(
                {
                    "条件": label,
                    "値": value,
                    "サンプル数": len(group),
                    "対象レース数": group["race_id"].nunique(),
                    "単勝的中率": pct((win_pay > 0).sum(), len(group)),
                    "単勝回収率": pct(win_pay.sum(), stake),
                    "複勝的中率": pct((place_pay > 0).sum(), len(group)),
                    "複勝回収率": pct(place_pay.sum(), stake),
                    "参考区分": "参考値" if len(group) < 30 or group["race_id"].nunique() < 20 else "集計対象",
                }
            )
    return pd.DataFrame(rows)


def distance_band(value: Any) -> str:
    distance = to_float(value)
    if distance is None:
        return "欠損"
    if distance < 1400:
        return "短距離"
    if distance < 1800:
        return "マイル前後"
    if distance < 2200:
        return "中距離"
    return "長距離"


def field_size_band(value: Any) -> str:
    size = to_int(value)
    if size is None:
        return "欠損"
    if size <= 8:
        return "少頭数"
    if size <= 12:
        return "中頭数"
    return "多頭数"


def odds_band(value: Any) -> str:
    odds = to_float(value)
    if odds is None:
        return "欠損"
    if odds < 2:
        return "2倍未満"
    if odds < 5:
        return "2〜5倍"
    if odds < 10:
        return "5〜10倍"
    if odds < 20:
        return "10〜20倍"
    return "20倍以上"


def diff_band(value: Any) -> str:
    diff = to_float(value)
    if diff is None:
        return "欠損"
    if diff < 0:
        return "○が上"
    if diff < 2:
        return "0〜2差"
    if diff < 5:
        return "2〜5差"
    if diff < 10:
        return "5〜10差"
    return "10差以上"


def classify_reference(sample_size: int, hits: int, roi: float) -> str:
    if sample_size < 30:
        return "サンプル不足"
    if roi >= 110 and hits >= 3:
        return "購入参考候補"
    if roi >= 80:
        return "参考"
    return "見送り参考"


def build_report_payload(
    records: pd.DataFrame,
    payouts_by_race: dict[str, dict[str, Any]],
    meta: dict[str, Any],
) -> dict[str, Any]:
    if records is not None and not records.empty and "value_signal" not in records.columns:
        records = attach_value_signals_to_records(records)
    mark_summary = evaluate_mark_singles(records)
    box_summary = evaluate_box_strategies(records, payouts_by_race)
    condition_summary = build_condition_summary(records)
    value_summary = evaluate_value_singles(records)
    return {
        "meta": meta,
        "mark_summary": mark_summary,
        "box_summary": box_summary,
        "condition_summary": condition_summary,
        "value_summary": value_summary,
    }


def write_outputs(payload: dict[str, Any], records: pd.DataFrame, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    mark_summary = payload["mark_summary"]
    box_summary = payload["box_summary"]
    condition_summary = payload["condition_summary"]
    value_summary = payload.get("value_summary", pd.DataFrame())
    paths = {
        "race_mark_details": out_dir / "race_mark_details.csv",
        "mark_summary": out_dir / "mark_single_summary.csv",
        "box_summary": out_dir / "box_summary.csv",
        "condition_summary": out_dir / "condition_summary.csv",
        "value_summary": out_dir / "value_signal_summary.csv",
        "json": out_dir / "mark_betting_backtest_summary.json",
        "markdown": out_dir / "mark_betting_backtest_report.md",
    }
    records.to_csv(paths["race_mark_details"], index=False, encoding="utf-8-sig")
    mark_summary.to_csv(paths["mark_summary"], index=False, encoding="utf-8-sig")
    box_summary.to_csv(paths["box_summary"], index=False, encoding="utf-8-sig")
    condition_summary.to_csv(paths["condition_summary"], index=False, encoding="utf-8-sig")
    value_summary.to_csv(paths["value_summary"], index=False, encoding="utf-8-sig")
    json_payload = {
        "meta": payload["meta"],
        "mark_summary": mark_summary.to_dict(orient="records"),
        "box_summary": box_summary.to_dict(orient="records"),
        "condition_summary": condition_summary.to_dict(orient="records"),
        "value_summary": value_summary.to_dict(orient="records"),
    }
    paths["json"].write_text(json.dumps(json_payload, ensure_ascii=False, indent=2, default=json_default), encoding="utf-8")
    paths["markdown"].write_text(build_markdown_report(payload), encoding="utf-8")
    return paths


def add_honmei_maruta_difference_columns(records: pd.DataFrame) -> pd.DataFrame:
    if records is None or records.empty:
        return records
    frame = records.copy()
    frame["◎○能力値差"] = None
    frame["◎○今回評価差"] = None
    for race_id, group in frame.groupby("race_id", sort=False):
        honmei = group[group["mark"].eq("◎")]
        maru = group[group["mark"].eq("○")]
        if honmei.empty or maru.empty:
            continue
        honmei_row = honmei.sort_values(["ai_current_rank", "horse_no"], na_position="last").iloc[0]
        maru_row = maru.sort_values(["ai_current_rank", "horse_no"], na_position="last").iloc[0]
        ability_diff = numeric_diff(honmei_row.get("ability_value"), maru_row.get("ability_value"))
        ai_diff = numeric_diff(honmei_row.get("ai_score"), maru_row.get("ai_score"))
        mask = frame["race_id"].astype(str).eq(str(race_id))
        frame.loc[mask, "◎○能力値差"] = ability_diff
        frame.loc[mask, "◎○今回評価差"] = ai_diff
    return frame


def numeric_diff(left: Any, right: Any) -> float | None:
    left_number = to_float(left)
    right_number = to_float(right)
    if left_number is None or right_number is None:
        return None
    return round(left_number - right_number, 3)


def build_markdown_report(payload: dict[str, Any]) -> str:
    meta = payload["meta"]
    lines = [
        "# 印・馬券バックテストレポート",
        "",
        "現行の予想ロジックを変更せず、保存済みHTMLから予想を再生成して結果HTMLと照合した検証です。",
        "結果HTMLは予想生成後の照合にのみ使用し、予想入力からは除外しています。",
        "",
        "## データ監査",
        "",
        f"- 使用レース数: {meta.get('usable_races', 0)}R",
        f"- JRA: {meta.get('jra_races', 0)}R",
        f"- NAR: {meta.get('nar_races', 0)}R",
        f"- 対象馬数: {meta.get('horse_count', 0)}頭",
        f"- 予想エラー: {meta.get('prediction_error_count', 0)}件",
        f"- 結果HTMLは予想生成入力から除外: {meta.get('future_info_isolated', True)}",
        "",
        "## 印別 単勝・複勝",
        "",
        markdown_table(payload["mark_summary"]),
        "",
        "## BOX",
        "",
        markdown_table(payload["box_summary"]),
        "",
        "## 妙味あり 単勝・複勝",
        "",
        markdown_table(payload.get("value_summary", pd.DataFrame())),
        "",
        "## 条件別（◎）",
        "",
        markdown_table(payload["condition_summary"].head(80)),
        "",
        "## 注意",
        "",
        "- 現行表示で穴系の第5印が `✓/✔` の場合、バックテスト上は依頼対象の `☆` として正規化しています。元の表示値は `race_mark_details.csv` の `raw_mark` に保持しています。",
        "- `妙味あり` は印とは別の表示補助です。能力帯・順位・印・材料・判定時オッズから結果前に判定し、印やAI点へは反映していません。",
        "- 回収率が高い方式でも、サンプル不足の場合は正式推奨ではなく参考値です。",
        "- 最大払戻除外回収率は、1件の高配当に依存していないかを見るための監査値です。",
        "- 今回は測定のみで、印・AI点・能力評価・買い目ロジックは変更していません。",
    ]
    return "\n".join(lines) + "\n"


def markdown_table(frame: pd.DataFrame) -> str:
    if frame is None or frame.empty:
        return "_データなし_"
    show = frame.copy()
    columns = list(show.columns)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for _, row in show.iterrows():
        values = [clean_text(row.get(column)).replace("|", "/") for column in columns]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def json_default(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, (pd.Timestamp,)):
        return value.isoformat()
    if isinstance(value, float) and math.isnan(value):
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)

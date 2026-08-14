# -*- coding: utf-8 -*-
"""Parse result-free course, pace, position, and jockey context.

This module intentionally does not calculate ability.  It only carries values
that are explicitly present in a saved netkeiba newspaper HTML document into
the independent Market Compare display layer.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field
import math
import re
from typing import Any, Mapping

import pandas as pd
from bs4 import BeautifulSoup


POSITION_CORNER_NAMES = {
    "Corner01": "start",
    "Corner02": "corner3",
    "Corner03": "corner4",
}
POSITION_LABELS = {
    "start": "スタート後",
    "corner3": "3コーナー",
    "corner4": "4コーナー",
}
MATRIX_ROWS = ("front", "middle", "rear")
MATRIX_COLUMNS = ("inner", "middle", "outer")
JOCKEY_COURSE_MIN_STARTS = 20


@dataclass
class ParsedCourseMaterials:
    race_id: str = ""
    detected_mode: str = ""
    source_status: str = "html内に存在しない"
    course_condition: str = ""
    pace: str = ""
    positions: dict[str, dict[int, dict[str, str]]] = field(default_factory=dict)
    position_categories: dict[str, dict[int, str]] = field(default_factory=dict)
    position_coverage: dict[str, int] = field(default_factory=dict)
    horse_count: int = 0
    four_corner_place_rates: dict[str, dict[str, int]] = field(default_factory=dict)
    favorable_position_label: str = ""
    favorable_horses: list[dict[str, Any]] = field(default_factory=list)
    favorable_horses_complete: bool = False
    ai_opinion: str = ""
    ai_opinion_complete: bool = False
    predicted_3f: dict[int, dict[str, Any]] = field(default_factory=dict)
    predicted_3f_coverage: int = 0
    predicted_3f_complete: bool = False
    predicted_3f_usable: bool = False
    track_bias_status: str = "html内に存在しない"
    track_bias_code: str = ""
    track_bias_text: list[str] = field(default_factory=list)
    lap_prediction_status: str = "html内に存在しない"
    lap_prediction: list[dict[str, Any]] = field(default_factory=list)
    jockey_course_ranking: list[dict[str, Any]] = field(default_factory=list)
    jockey_course_stats_status: str = "html内に勝率・連対率・複勝率・出走回数なし"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class ParsedJockeyCourseStats:
    race_id: str = ""
    detected_mode: str = ""
    source_status: str = "html内に存在しない"
    course_condition: str = ""
    horses: dict[int, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def parse_netkeiba_course_materials(
    html: str,
    *,
    expected_mode: str = "",
) -> ParsedCourseMaterials:
    """Return only values explicitly recoverable from saved newspaper HTML."""

    source = str(html or "")
    parsed = ParsedCourseMaterials()
    if not source.strip():
        return parsed

    soup = BeautifulSoup(source, "html.parser")
    parsed.race_id = _race_id(source)
    parsed.detected_mode = _mode(source)
    if expected_mode and parsed.detected_mode and parsed.detected_mode != expected_mode:
        parsed.source_status = "JRA/NAR不一致"
        return parsed

    ai_area = soup.select_one(".AiTenkaiArea01, .AiTenkaiArea02")
    position_area = soup.select_one(".PositionMapBlock")
    if ai_area is None and position_area is None:
        parsed.source_status = "AI展開予測がHTML内に存在しない"
        return parsed
    parsed.source_status = "取得"

    course_heading = soup.select_one(".CourseDataArea.Time .CourseDataTitle h2, .CourseDataArea.Time h2")
    course_text = _text(course_heading)
    parsed.course_condition = re.sub(r"^コース情報\s*", "", course_text).strip()

    pace_node = soup.select_one(".CourseDataArea.Time .Data")
    pace_text = _text(pace_node).upper()
    pace_class = " ".join(pace_node.get("class", [])) if pace_node else ""
    pace_match = re.search(r"(?:^|\s)Pace_([HMS])(?:\s|$)", pace_class, flags=re.I)
    if pace_text in {"H", "M", "S"}:
        parsed.pace = pace_text
    elif pace_match:
        parsed.pace = pace_match.group(1).upper()

    actual_horse_numbers = {
        int(match.group(1))
        for node in soup.select(".DevelopImgWrap .HorseIcon[id^='Horse']")
        if (match := re.search(r"Horse(\d+)", node.get("id", "")))
    }
    parsed.positions = _parse_position_javascript(source)
    if actual_horse_numbers:
        parsed.positions = {
            corner: {
                number: values
                for number, values in horses.items()
                if number in actual_horse_numbers
            }
            for corner, horses in parsed.positions.items()
        }
    active_position = _parse_active_position_dom(soup)
    for corner, horses in active_position.items():
        # The rendered DOM is authoritative for the currently selected corner.
        # Inline JavaScript can retain commented-out/stale branches.
        parsed.positions.setdefault(corner, {}).update(horses)
    parsed.position_coverage = {
        corner: len(parsed.positions.get(corner, {}))
        for corner in ("start", "corner3", "corner4")
    }
    parsed.horse_count = len(actual_horse_numbers) or max(parsed.position_coverage.values(), default=0)
    parsed.position_categories = _position_categories(parsed.positions)

    parsed.four_corner_place_rates = _parse_four_corner_rates(soup)
    position_map = soup.select_one(".PositionMapImg")
    parsed.favorable_position_label = _text(position_map.select_one("dt")) if position_map else ""
    parsed.favorable_horses = _parse_favorable_horses(soup)
    favorite_wrap = soup.select_one(".PositionPickupHorseWrap")
    parsed.favorable_horses_complete = bool(
        favorite_wrap
        and not favorite_wrap.select_one(".DummyBox02, .MasterPush, [class*='Dummy']")
    )

    opinion_node = soup.select_one("section.DevelopOpinionArea dd.NoCheckData p, .DevelopOpinionArea dd p")
    parsed.ai_opinion = _text(opinion_node)
    opinion_parent = opinion_node.find_parent("section") if opinion_node else None
    opinion_dummy = bool(
        opinion_parent
        and opinion_parent.select_one("img[src*='dummy'], .FreemiumDummy01, .MasterPush")
    )
    parsed.ai_opinion_complete = bool(
        parsed.ai_opinion
        and not opinion_dummy
        and not re.search(r"(?:\.\.\.|…)$", parsed.ai_opinion)
    )

    parsed.predicted_3f = _parse_predicted_three_furlongs(soup)
    parsed.predicted_3f_coverage = len(parsed.predicted_3f)
    parsed.predicted_3f_complete = bool(
        parsed.horse_count
        and parsed.predicted_3f_coverage == parsed.horse_count
    )
    # Ranking one or two preview rows creates a misleading comparison.  Three
    # explicit rows is the existing minimum before this can become a material.
    parsed.predicted_3f_usable = parsed.predicted_3f_coverage >= 3

    bias_node = soup.select_one(".DevelopImg01")
    bias_classes = bias_node.get("class", []) if bias_node else []
    bias_code = next((value for value in bias_classes if re.fullmatch(r"BiasPattern\d{3}", value)), "")
    bias_text = [_text(item) for item in soup.select(".DevelopBiasTxt span") if _text(item)]
    parsed.track_bias_code = bias_code
    parsed.track_bias_text = bias_text
    if bias_text:
        parsed.track_bias_status = "取得"
    elif bias_code:
        parsed.track_bias_status = "コードのみ・意味未確定"
    elif soup.select_one(".CourseDataArea.Bias, .BiasSwitch, .DevelopImg01"):
        parsed.track_bias_status = "HTML内に実値なし"

    parsed.lap_prediction, parsed.lap_prediction_status = _parse_lap_prediction(soup)
    parsed.jockey_course_ranking = _parse_jockey_course_ranking(soup)
    return parsed


def parse_netkeiba_jockey_course_stats(
    html: str,
    *,
    expected_mode: str = "",
) -> ParsedJockeyCourseStats:
    """Parse only the explicit per-horse course jockey-statistics table.

    The saved ``data_list.html?mode=courseanalysis&cid=2`` page contains a
    result-free table with starts and three rates.  A cid=1 style-analysis
    page, a newspaper ranking preview, and chart-only placeholders are not
    accepted as substitutes.
    """

    source = str(html or "")
    parsed = ParsedJockeyCourseStats()
    if not source.strip():
        return parsed
    parsed.race_id = _race_id(source)
    parsed.detected_mode = _mode(source)
    if expected_mode and parsed.detected_mode and parsed.detected_mode != expected_mode:
        parsed.source_status = "JRA/NAR不一致"
        return parsed
    lowered = source[:300_000].lower().replace("&amp;", "&")
    if not re.search(r"mode=courseanalysis(?:[^\"'<>])*[&?]cid=2(?:\D|$)", lowered):
        parsed.source_status = "騎手コース成績HTMLではない"
        return parsed

    soup = BeautifulSoup(source, "html.parser")
    table = soup.select_one("table#table_sort_back")
    if table is None:
        parsed.source_status = "騎手コース成績表がHTML内に存在しない"
        return parsed
    headers = [_normalize_header(_text(cell)) for cell in table.select("thead tr.Header th")]
    required = ("馬番", "項目", "出走回数", "勝率", "連対率", "複勝率", "馬名")
    indexes = {name: headers.index(name) for name in required if name in headers}
    if any(name not in indexes for name in required):
        parsed.source_status = "騎手コース成績表の列不足"
        return parsed

    title_text = _text(soup.select_one("title"))
    condition_match = re.search(r"([^\s|_]+?[芝ダ]\d{3,4}m)が得意な騎手", title_text)
    parsed.course_condition = condition_match.group(1) if condition_match else ""
    for row in table.select("tbody tr.HorseList"):
        cells = row.find_all("td", recursive=False)
        if any(index >= len(cells) for index in indexes.values()):
            continue
        number_match = re.search(r"\d+", _text(cells[indexes["馬番"]]))
        if not number_match:
            continue
        number = int(number_match.group(0))
        starts = _integer_cell(cells[indexes["出走回数"]])
        win = _percentage_cell(cells[indexes["勝率"]])
        quinella = _percentage_cell(cells[indexes["連対率"]])
        place = _percentage_cell(cells[indexes["複勝率"]])
        if starts is None or any(value is None for value in (win, quinella, place)):
            continue
        parsed.horses[number] = {
            "horse_number": number,
            "horse_name": _text(cells[indexes["馬名"]]),
            "jockey_name": _text(cells[indexes["項目"]]),
            "starts": starts,
            "win_rate": win,
            "quinella_rate": quinella,
            "place_rate": place,
        }
    parsed.source_status = "取得" if parsed.horses else "騎手コース成績表に実値なし"
    return parsed


def attach_course_materials_to_result(
    result: Any,
    html_files: Mapping[str, str] | None,
) -> Any:
    """Attach parsed context columns without modifying any ability input/value."""

    files = dict(html_files or {})
    source = files.get("newspaper_context") or files.get("newspaper") or ""
    parsed = parse_netkeiba_course_materials(
        source,
        expected_mode=str(getattr(result, "race_mode", "") or ""),
    )
    race_info = getattr(result, "race_info", {}) or {}
    expected_race_id = str(race_info.get("race_id") or "").strip()
    if expected_race_id and parsed.race_id and expected_race_id != parsed.race_id:
        parsed.source_status = "race_id不一致"
        _store_debug(result, parsed, ParsedJockeyCourseStats())
        return result

    jockey_parsed = parse_netkeiba_jockey_course_stats(
        files.get("jockey") or "",
        expected_mode=str(getattr(result, "race_mode", "") or ""),
    )
    if expected_race_id and jockey_parsed.race_id and expected_race_id != jockey_parsed.race_id:
        jockey_parsed.source_status = "race_id不一致"
        jockey_by_number: dict[int, dict[str, Any]] = {}
    else:
        jockey_by_number = jockey_parsed.horses

    favorite_numbers = {
        int(item["horse_number"])
        for item in parsed.favorable_horses
        if item.get("horse_number") is not None
    }
    ranking_by_number = {
        int(item["horse_number"]): item
        for item in parsed.jockey_course_ranking
        if item.get("horse_number") is not None
    }
    rates_display = four_corner_rates_display(parsed.four_corner_place_rates)

    for attribute in ("overall_table", "horse_evaluation"):
        source_table = getattr(result, attribute, None)
        if not isinstance(source_table, pd.DataFrame) or source_table.empty:
            continue
        target = source_table.copy()
        for column in COURSE_CONTEXT_COLUMNS:
            if column not in target.columns:
                target[column] = pd.Series([None] * len(target), index=target.index, dtype="object")
        for index, raw in target.iterrows():
            number = _horse_number(raw.to_dict())
            target.at[index, "_course_context_status"] = parsed.source_status
            target.at[index, "_course_condition_html"] = parsed.course_condition
            target.at[index, "_netkeiba_pace"] = parsed.pace
            target.at[index, "_favorable_position_label"] = parsed.favorable_position_label
            target.at[index, "_four_corner_place_rates"] = rates_display
            target.at[index, "_position_coverage"] = _coverage_display(parsed)
            target.at[index, "_ai_opinion"] = parsed.ai_opinion
            target.at[index, "_ai_opinion_complete"] = parsed.ai_opinion_complete
            target.at[index, "_track_bias_status"] = parsed.track_bias_status
            target.at[index, "_track_bias_code"] = parsed.track_bias_code
            target.at[index, "_lap_prediction_status"] = parsed.lap_prediction_status
            target.at[index, "_predicted_3f_coverage"] = (
                f"{parsed.predicted_3f_coverage}/{parsed.horse_count or '?'}"
            )
            if number is None:
                continue
            for corner, column in (
                ("start", "_estimated_position_start"),
                ("corner3", "_estimated_position_corner3"),
                ("corner4", "_estimated_position_corner4"),
            ):
                target.at[index, column] = position_display(parsed.positions.get(corner, {}).get(number))
            for corner, column in (
                ("start", "_estimated_position_start_label"),
                ("corner3", "_estimated_position_corner3_label"),
                ("corner4", "_estimated_position_corner4_label"),
            ):
                target.at[index, column] = parsed.position_categories.get(corner, {}).get(number, "")
            target.at[index, "_estimated_position_path"] = position_path_display(
                parsed.position_categories,
                number,
            )
            if number in favorite_numbers:
                target.at[index, "_position_favorable_horse"] = True
            elif parsed.favorable_horses_complete:
                target.at[index, "_position_favorable_horse"] = False
            ranking = ranking_by_number.get(number)
            if ranking:
                target.at[index, "_jockey_course_rank"] = ranking.get("rank")
                target.at[index, "_jockey_course_rank_name"] = ranking.get("jockey_name")
            if parsed.predicted_3f_usable and number in parsed.predicted_3f:
                target.at[index, "_ai_predicted_early3f"] = parsed.predicted_3f[number].get("early_3f")
                target.at[index, "_ai_predicted_late3f"] = parsed.predicted_3f[number].get("late_3f")
            jockey_stats = jockey_by_number.get(number)
            if jockey_stats:
                target.at[index, "_jockey_course_win_rate"] = jockey_stats.get("win_rate")
                target.at[index, "_jockey_course_quinella_rate"] = jockey_stats.get("quinella_rate")
                target.at[index, "_jockey_course_place_rate"] = jockey_stats.get("place_rate")
                target.at[index, "_jockey_course_starts"] = jockey_stats.get("starts")
                target.at[index, "_jockey_course_condition"] = jockey_parsed.course_condition
                target.at[index, "_jockey_course_source"] = "netkeiba courseanalysis cid=2"
                target.at[index, "_jockey_course_html_name"] = jockey_stats.get("jockey_name")
        setattr(result, attribute, target)

    _store_debug(result, parsed, jockey_parsed)
    return result


COURSE_CONTEXT_COLUMNS = (
    "_course_context_status",
    "_course_condition_html",
    "_netkeiba_pace",
    "_estimated_position_start",
    "_estimated_position_corner3",
    "_estimated_position_corner4",
    "_estimated_position_start_label",
    "_estimated_position_corner3_label",
    "_estimated_position_corner4_label",
    "_estimated_position_path",
    "_position_coverage",
    "_position_favorable_horse",
    "_favorable_position_label",
    "_four_corner_place_rates",
    "_ai_opinion",
    "_ai_opinion_complete",
    "_predicted_3f_coverage",
    "_ai_predicted_early3f",
    "_ai_predicted_late3f",
    "_track_bias_status",
    "_track_bias_code",
    "_lap_prediction_status",
    "_jockey_course_rank",
    "_jockey_course_rank_name",
    "_jockey_course_win_rate",
    "_jockey_course_quinella_rate",
    "_jockey_course_place_rate",
    "_jockey_course_starts",
    "_jockey_course_condition",
    "_jockey_course_source",
    "_jockey_course_html_name",
)


def position_display(value: Mapping[str, str] | None) -> str:
    if not value:
        return ""
    top = str(value.get("top") or "").strip()
    left = str(value.get("left") or "").strip()
    speed = str(value.get("speed_class") or "").strip()
    bits = [f"top={top}" if top else "", f"left={left}" if left else "", speed]
    return ", ".join(bit for bit in bits if bit)


def position_path_display(categories: Mapping[str, Mapping[int, str]], horse_number: int) -> str:
    labels = [
        clean
        for corner in ("start", "corner3", "corner4")
        if (clean := str((categories.get(corner) or {}).get(horse_number) or "").strip())
    ]
    return " → ".join(labels) if len(labels) == 3 else ""


def four_corner_rates_display(rates: Mapping[str, Mapping[str, Any]] | None) -> str:
    if not rates:
        return ""
    row_labels = {"front": "前", "middle": "中", "rear": "後"}
    column_labels = {"inner": "内", "middle": "中", "outer": "外"}
    parts: list[str] = []
    for row in MATRIX_ROWS:
        values = rates.get(row) if isinstance(rates, Mapping) else None
        if not isinstance(values, Mapping):
            continue
        cells = [
            f"{column_labels[column]}{int(values[column])}%"
            for column in MATRIX_COLUMNS
            if values.get(column) is not None
        ]
        if cells:
            parts.append(f"{row_labels[row]}({','.join(cells)})")
    return " / ".join(parts)


def _parse_position_javascript(source: str) -> dict[str, dict[int, dict[str, str]]]:
    positions: dict[str, dict[int, dict[str, str]]] = {}
    start = source.find("function updateHorsePosition()")
    if start < 0:
        return positions
    end = source.find("// コーナーのクリックイベント", start)
    chunk = source[start : end if end >= 0 else start + 200_000]
    # Saved NAR pages can retain a previous field's assignments as `// ...`
    # immediately before the live assignments.  Commented code is not data.
    chunk = re.sub(r"(?m)^[ \t]*//.*$", "", chunk)
    base_match = re.search(
        r"if\s*\(\s*!checkbox1Checked\s*&&\s*!checkbox2Checked\s*\)\s*\{([\s\S]*?)\}\s*else\s+if",
        chunk,
    )
    base = base_match.group(1) if base_match else chunk
    for corner_id, corner_name in POSITION_CORNER_NAMES.items():
        case_match = re.search(
            rf"case\s*['\"]{re.escape(corner_id)}['\"]\s*:\s*([\s\S]*?)\bbreak\s*;",
            base,
        )
        if not case_match:
            continue
        records: dict[int, dict[str, str]] = {}
        statement_pattern = re.compile(
            r"\$\(\s*['\"]#Horse(\d+)['\"]\s*\)\.css\(\s*\{\s*"
            r"['\"]top['\"]\s*:\s*['\"]([^'\"]+)['\"]\s*,\s*"
            r"['\"]left['\"]\s*:\s*['\"]([^'\"]+)['\"]\s*,?\s*\}\s*\)"
            r"([\s\S]*?);"
        )
        for horse, top, left, suffix in statement_pattern.findall(case_match.group(1)):
            speed_match = re.search(r"class=['\"](Speed(?:Up|Down)_\d+)['\"]", suffix)
            records[int(horse)] = {
                "top": top.strip(),
                "left": left.strip(),
                "speed_class": speed_match.group(1) if speed_match else "",
                "source": "inline_javascript",
            }
        if records:
            positions[corner_name] = records
    return positions


def _parse_active_position_dom(soup: BeautifulSoup) -> dict[str, dict[int, dict[str, str]]]:
    active = soup.select_one("#CornerSwitch li.Active a")
    corner_name = POSITION_CORNER_NAMES.get(active.get("id", "") if active else "")
    if not corner_name:
        return {}
    records: dict[int, dict[str, str]] = {}
    for node in soup.select(".DevelopImgWrap .HorseIcon[id^='Horse']"):
        number_match = re.search(r"Horse(\d+)", node.get("id", ""))
        if not number_match:
            continue
        style = node.get("style", "")
        top_match = re.search(r"top\s*:\s*([^;]+)", style)
        left_match = re.search(r"left\s*:\s*([^;]+)", style)
        speed = ""
        for child in node.find_all("span", recursive=False):
            speed = next(
                (value for value in child.get("class", []) if re.fullmatch(r"Speed(?:Up|Down)_\d+", value)),
                speed,
            )
        records[int(number_match.group(1))] = {
            "top": top_match.group(1).strip() if top_match else "",
            "left": left_match.group(1).strip() if left_match else "",
            "speed_class": speed,
            "source": "static_dom",
        }
    return {corner_name: records} if records else {}


def _position_categories(
    positions: Mapping[str, Mapping[int, Mapping[str, str]]],
) -> dict[str, dict[int, str]]:
    """Convert the visual left-to-right order into field-relative labels.

    The position map explicitly draws the front at the left and the rear at
    the right.  ``top`` only separates overlapping icons, so it is never used
    as a pace/position signal.  Labels are based on each corner's relative
    horizontal order; fewer than three explicit horses remains unknown.
    """

    categorized: dict[str, dict[int, str]] = {}
    for corner in ("start", "corner3", "corner4"):
        ordered: list[tuple[float, int]] = []
        for number, values in (positions.get(corner) or {}).items():
            match = re.search(r"-?\d+(?:\.\d+)?", str(values.get("left") or ""))
            if match:
                ordered.append((float(match.group(0)), int(number)))
        ordered.sort(key=lambda item: (item[0], item[1]))
        count = len(ordered)
        if count < 3:
            continue
        front_cut = max(2, math.ceil(count / 3))
        middle_cut = max(front_cut + 1, math.ceil(count * 2 / 3))
        unique_leader = count == 1 or ordered[1][0] > ordered[0][0]
        labels: dict[int, str] = {}
        for rank, (_, number) in enumerate(ordered):
            if rank == 0 and unique_leader:
                label = "逃げ"
            elif rank < front_cut:
                label = "先団"
            elif rank < middle_cut:
                label = "中団"
            else:
                label = "後方"
            labels[number] = label
        categorized[corner] = labels
    return categorized


def _parse_four_corner_rates(soup: BeautifulSoup) -> dict[str, dict[str, int]]:
    rows = soup.select(".PositionMapImg .PositionMarkList > li")
    if len(rows) != 3:
        return {}
    result: dict[str, dict[str, int]] = {}
    for row_name, row in zip(MATRIX_ROWS, rows):
        values: list[int] = []
        for cell in row.select(":scope > div"):
            match = re.search(r"(\d{1,3})\s*%", _text(cell))
            if match:
                values.append(int(match.group(1)))
        if len(values) != 3:
            return {}
        result[row_name] = dict(zip(MATRIX_COLUMNS, values))
    return result


def _parse_favorable_horses(soup: BeautifulSoup) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for item in soup.select(".PositionPickupHorseWrap li"):
        number_node = item.select_one(".Umaban_Num")
        link = item.select_one("a")
        number_match = re.search(r"\d+", _text(number_node))
        if not number_match or link is None:
            continue
        horse_id_match = re.search(r"/horse/(\d+)", link.get("href", ""))
        result.append(
            {
                "horse_number": int(number_match.group(0)),
                "horse_name": _text(link),
                "horse_id": horse_id_match.group(1) if horse_id_match else "",
            }
        )
    return result


def _parse_predicted_three_furlongs(soup: BeautifulSoup) -> dict[int, dict[str, Any]]:
    table = soup.select_one("table.PredictRap_Table")
    if table is None:
        return {}
    headers = [_text(item).replace("\n", "") for item in table.select("thead th")]
    header_indexes = {header: index for index, header in enumerate(headers)}
    number_index = next((index for name, index in header_indexes.items() if "馬" in name and "番" in name), None)
    name_index = next((index for name, index in header_indexes.items() if "馬名" in name), None)
    early_index = next((index for name, index in header_indexes.items() if "前半" in name and "3F" in name), None)
    late_index = next((index for name, index in header_indexes.items() if "後半" in name and "3F" in name), None)
    if number_index is None or (early_index is None and late_index is None):
        return {}
    result: dict[int, dict[str, Any]] = {}
    for row in table.select("tbody tr"):
        if row.select_one("img[src*='dummy']"):
            continue
        cells = row.find_all("td", recursive=False)
        if number_index >= len(cells):
            continue
        number_match = re.search(r"\d+", _text(cells[number_index]))
        if not number_match:
            continue
        early = _float_cell(cells, early_index)
        late = _float_cell(cells, late_index)
        if early is None and late is None:
            continue
        number = int(number_match.group(0))
        result[number] = {
            "horse_number": number,
            "horse_name": _text(cells[name_index]) if name_index is not None and name_index < len(cells) else "",
            "early_3f": early,
            "late_3f": late,
        }
    return result


def _parse_lap_prediction(soup: BeautifulSoup) -> tuple[list[dict[str, Any]], str]:
    table = soup.select_one("table.Race_HaronTime")
    if table is None:
        return [], "html内に存在しない"
    headers = [_text(item) for item in table.select("tr.Header th")]
    value_row = table.select_one("tr.HaronTime")
    if value_row is None:
        return [], "HTML内に実値なし"
    values: list[dict[str, Any]] = []
    for index, cell in enumerate(value_row.find_all("td", recursive=False)):
        if cell.select_one("img[src*='dummy']"):
            continue
        match = re.search(r"\d+(?:\.\d+)?", _text(cell))
        if not match:
            continue
        values.append(
            {
                "distance": headers[index] if index < len(headers) else "",
                "seconds": float(match.group(0)),
            }
        )
    if not values:
        return [], "HTML内に実値なし"
    return values, "取得" if len(values) == len(headers) else "一部取得"


def _parse_jockey_course_ranking(soup: BeautifulSoup) -> list[dict[str, Any]]:
    for table in soup.select("table.AnaBestTable"):
        if _text(table.select_one(".PickupHorseTableTitle")) != "騎手":
            continue
        link_text = _text(table.select_one("a"))
        if "コースランキング" not in link_text:
            continue
        result: list[dict[str, Any]] = []
        for rank, box in enumerate(table.select(".AnaBest_HorseBox .Kyaku_Type_box"), start=1):
            number_match = re.search(r"\d+", _text(box.select_one(".Kyaku_Type_Num")))
            if not number_match:
                continue
            result.append(
                {
                    "rank": rank,
                    "horse_number": int(number_match.group(0)),
                    "jockey_name": _text(box.select_one(".UmaName")),
                    "status": "率・出走回数なしの参考順位",
                }
            )
        return result
    return []


def _coverage_display(parsed: ParsedCourseMaterials) -> str:
    denominator: Any = parsed.horse_count or "?"
    return " / ".join(
        f"{POSITION_LABELS[corner]} {parsed.position_coverage.get(corner, 0)}/{denominator}"
        for corner in ("start", "corner3", "corner4")
    )


def _float_cell(cells: list[Any], index: int | None) -> float | None:
    if index is None or index >= len(cells):
        return None
    match = re.search(r"\d+(?:\.\d+)?", _text(cells[index]))
    return float(match.group(0)) if match else None


def _integer_cell(cell: Any) -> int | None:
    match = re.search(r"\d+", _text(cell).replace(",", ""))
    return int(match.group(0)) if match else None


def _percentage_cell(cell: Any) -> float | None:
    match = re.search(r"\d+(?:\.\d+)?", _text(cell).replace(",", ""))
    return float(match.group(0)) if match else None


def _normalize_header(value: str) -> str:
    return re.sub(r"\s+", "", str(value or ""))


def _horse_number(row: Mapping[str, Any]) -> int | None:
    for key in ("馬番", "horse_no", "horse_number", "馬"):
        value = row.get(key)
        match = re.search(r"\d+", str(value or ""))
        if match:
            return int(match.group(0))
    return None


def _race_id(source: str) -> str:
    head = source[:300_000]
    candidates = re.findall(r"race_id(?:=|%3D)(\d{10,14})", head, flags=re.I)
    return candidates[0] if candidates else ""


def _mode(source: str) -> str:
    head = source[:150_000]
    # Prefer the document's own canonical/og URL.  A JRA page also contains
    # links and explanatory text mentioning NAR.
    metadata_tags = re.findall(
        r"<(?:link|meta)\b[^>]*(?:rel\s*=\s*['\"]canonical['\"]|property\s*=\s*['\"]og:url['\"])[^>]*>",
        head,
        flags=re.I,
    )
    metadata = " ".join(metadata_tags).lower()
    if re.search(r"https?://nar(?:\.sp)?\.netkeiba\.com/", metadata):
        return "nar"
    if re.search(r"https?://race(?:\.sp)?\.netkeiba\.com/", metadata):
        return "jra"
    lowered = head.lower()
    if re.search(r"https?://nar(?:\.sp)?\.netkeiba\.com/race/newspaper", lowered):
        return "nar"
    if re.search(r"https?://race(?:\.sp)?\.netkeiba\.com/race/newspaper", lowered):
        return "jra"
    if "地方競馬レース情報" in head:
        return "nar"
    if "レース情報(jra)" in lowered or "JRA" in head:
        return "jra"
    return ""


def _text(node: Any) -> str:
    if node is None:
        return ""
    return re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()


def _store_debug(
    result: Any,
    parsed: ParsedCourseMaterials,
    jockey_parsed: ParsedJockeyCourseStats | None = None,
) -> None:
    debug = dict(getattr(result, "debug_info", {}) or {})
    debug["course_materials"] = parsed.to_dict()
    if jockey_parsed is not None:
        debug["jockey_course_materials"] = jockey_parsed.to_dict()
    result.debug_info = debug

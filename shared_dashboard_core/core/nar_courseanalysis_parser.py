from __future__ import annotations

import html as html_lib
import re
from typing import Any


class NarCourseAnalysisParseError(ValueError):
    """Raised when NAR course-analysis HTML cannot be converted to data."""


def is_courseanalysis_html(text: str) -> bool:
    normalized = html_lib.unescape(str(text or ""))
    lower = normalized.lower()
    head = lower[:300_000]
    own_page_tags = re.findall(
        r"<(?:link|meta)\b[^>]*(?:canonical|og:url)[^>]*>",
        head,
        flags=re.I,
    )
    own_page_text = " ".join(own_page_tags)
    if (
        "mode=courseanalysis" in own_page_text
        and re.search(r"(?:[?&])cid=2(?:\D|$)", own_page_text)
    ):
        return False
    has_page_markers = (
        "<html" in lower
        and "mode=courseanalysis" in lower
        and "cid=1" in lower
    )
    has_graph_marker = (
        'id="score1"' in lower
        or "id='score1'" in lower
        or "datagraphwrap1" in lower
    )
    has_chart = "new chart" in lower or "chart(" in lower
    return has_page_markers and has_graph_marker and has_chart


def extract_race_id_from_courseanalysis_html(html: str) -> str:
    source = html_lib.unescape(str(html or ""))
    head = source[:150_000]
    candidates: list[str] = []

    for tag_pattern in (
        r"<link\b[^>]*rel=['\"]?canonical['\"]?[^>]*>",
        r"<meta\b[^>]*(?:property|name)=['\"]?og:url['\"]?[^>]*>",
    ):
        for tag in re.findall(tag_pattern, head, flags=re.I):
            value = _attr(tag, "href") or _attr(tag, "content")
            if value:
                candidates.extend(_race_ids_in_text(value))

    candidates.extend(_race_ids_in_text(head))
    for race_id in candidates:
        if re.fullmatch(r"\d{12}", race_id):
            return race_id
    return candidates[0] if candidates else ""


def parse_courseanalysis_html(html: str) -> dict[str, Any]:
    if not is_courseanalysis_html(html):
        raise NarCourseAnalysisParseError("courseanalysis HTMLとして判定できませんでした。")

    race_id = extract_race_id_from_courseanalysis_html(html)
    script_text = _find_score1_chart_script(html)
    labels = _extract_labels(script_text)
    if not labels:
        raise NarCourseAnalysisParseError("脚質ラベルを取得できませんでした。")

    datasets = _extract_datasets(script_text)
    win_values = _pick_dataset(datasets, "1着")
    second_values = _pick_dataset(datasets, "2着")
    third_values = _pick_dataset(datasets, "3着")
    outside_values = _pick_dataset(datasets, "着外率", "4着以下", "着外")
    if not win_values:
        raise NarCourseAnalysisParseError("1着率データを取得できませんでした。")

    running_styles: list[dict[str, Any]] = []
    for index, style in enumerate(labels):
        win_rate = _value_at(win_values, index)
        second_rate = _value_at(second_values, index)
        third_rate = _value_at(third_values, index)
        outside_rate = _value_at(outside_values, index)
        running_styles.append(
            {
                "style": style,
                "win_rate": win_rate,
                "second_rate": second_rate,
                "third_rate": third_rate,
                "quinella_rate": _sum_if_complete(win_rate, second_rate),
                "place_rate": _sum_if_complete(win_rate, second_rate, third_rate),
                "outside_rate": outside_rate,
            }
        )

    if not any(item.get("style") and item.get("win_rate") is not None for item in running_styles):
        raise NarCourseAnalysisParseError("有効な脚質データを取得できませんでした。")

    return {
        "race_id": race_id,
        "data_type": "courseanalysis",
        "race": _extract_race_info(html),
        "running_styles": running_styles,
        "horse_running_styles": _extract_horse_running_styles(html),
    }


def parse_number_array(source: str) -> list[int | float | None]:
    values: list[int | float | None] = []
    for item in str(source or "").split(","):
        text = item.strip().strip("'\"")
        if not text:
            values.append(None)
            continue
        try:
            number = float(text)
        except ValueError:
            values.append(None)
            continue
        values.append(int(number) if number.is_integer() else number)
    return values


def _find_score1_chart_script(html: str) -> str:
    source = html_lib.unescape(str(html or ""))
    scripts = re.findall(r"<script\b[^>]*>([\s\S]*?)</script>", source, flags=re.I)
    for script in scripts:
        lower = script.lower()
        if (
            "score1" in lower
            and "new chart" in lower
            and "labels" in lower
            and "datasets" in lower
        ):
            return script

    score1_position = source.lower().find("score1")
    if score1_position >= 0:
        return source[score1_position : score1_position + 80_000]
    return source[:120_000]


def _extract_labels(script_text: str) -> list[str]:
    match = re.search(r"labels\s*:\s*\[([\s\S]*?)\]", script_text)
    if not match:
        return []
    return [html_lib.unescape(value.strip()) for value in re.findall(r"""["']([^"']+)["']""", match.group(1))]


def _extract_datasets(script_text: str) -> dict[str, list[int | float | None]]:
    dataset_pattern = re.compile(
        r"""label\s*:\s*["']([^"']+)["'][\s\S]*?data\s*:\s*\[([^\]]+)\]""",
        re.VERBOSE,
    )
    datasets: dict[str, list[int | float | None]] = {}
    for label, values in dataset_pattern.findall(script_text):
        datasets[html_lib.unescape(label.strip())] = parse_number_array(values)
    return datasets


def _pick_dataset(
    datasets: dict[str, list[int | float | None]],
    *keywords: str,
) -> list[int | float | None]:
    for label, values in datasets.items():
        if any(keyword in label for keyword in keywords):
            return values
    return []


def _value_at(values: list[Any], index: int) -> Any:
    return values[index] if index < len(values) else None


def _sum_if_complete(*values: Any) -> int | float | None:
    if any(value is None for value in values):
        return None
    total = sum(float(value) for value in values)
    return int(total) if total.is_integer() else total


def _extract_race_info(html: str) -> dict[str, str]:
    race_name = _class_text(html, "RaceName") or _title_text(html)
    race_number = _class_text(html, "RaceNum")
    race_data_1 = _class_text(html, "RaceData01")
    race_data_2 = _class_text(html, "RaceData02")
    return {
        "race_name": race_name,
        "race_number": race_number,
        "race_data_1": race_data_1,
        "race_data_2": race_data_2,
    }


def _extract_horse_running_styles(html: str) -> list[dict[str, str]]:
    source = html_lib.unescape(str(html or ""))
    table_match = re.search(
        r"<table\b(?=[^>]*(?:id=['\"]table_sort_back['\"]|class=['\"][^'\"]*Data01_Table))[^>]*>([\s\S]*?)</table>",
        source,
        flags=re.I,
    )
    if not table_match:
        return []

    records: list[dict[str, str]] = []
    for row in re.findall(r"<tr\b[^>]*>([\s\S]*?)</tr>", table_match.group(1), flags=re.I):
        cells = re.findall(r"<td\b[^>]*>([\s\S]*?)</td>", row, flags=re.I)
        if len(cells) < 3:
            continue
        horse_number_match = re.search(r"\d{1,2}", _clean_text(cells[0]))
        if not horse_number_match:
            continue
        style_match = re.search(
            r"<td\b[^>]*class=['\"][^'\"]*DataTitle_Cell[^'\"]*['\"][^>]*>([\s\S]*?)</td>",
            row,
            flags=re.I,
        )
        style = _clean_text(style_match.group(1)) if style_match else _clean_text(cells[2])
        if not style:
            continue
        records.append(
            {
                "horse_number": horse_number_match.group(0),
                "horse_name": _clean_text(cells[1]),
                "running_style": style,
            }
        )
    return records


def _class_text(html: str, class_name: str) -> str:
    pattern = (
        r"<(?P<tag>[a-z0-9]+)\b[^>]*class=['\"][^'\"]*\b"
        + re.escape(class_name)
        + r"\b[^'\"]*['\"][^>]*>(?P<body>[\s\S]*?)</(?P=tag)>"
    )
    match = re.search(pattern, html, flags=re.I)
    if not match:
        return ""
    return _clean_text(match.group("body"))


def _title_text(html: str) -> str:
    match = re.search(r"<title[^>]*>([\s\S]*?)</title>", html, flags=re.I)
    return _clean_text(match.group(1)) if match else ""


def _clean_text(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(value or ""))
    text = html_lib.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _attr(tag: str, attr_name: str) -> str:
    match = re.search(
        rf"\b{re.escape(attr_name)}\s*=\s*(['\"])(.*?)\1",
        tag,
        flags=re.I,
    )
    if match:
        return html_lib.unescape(match.group(2))
    return ""


def _race_ids_in_text(text: str) -> list[str]:
    source = str(text or "")
    found = re.findall(r"race_id(?:=|%3D)(\d{12})", source, flags=re.I)
    found.extend(
        re.findall(r"race_id['\"]?\s*[:=]\s*['\"](\d{12})", source, flags=re.I)
    )
    return found

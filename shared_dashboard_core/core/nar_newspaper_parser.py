from __future__ import annotations

import html as html_lib
import re
from typing import Any


_PREVIOUS_RUN_KEYS = (
    "past_runs",
    "recent_runs",
    "race1_racecourse",
    "race1_venue",
    "race1_track",
    "race1_surface",
    "race1_distance",
    "race1_turn",
    "race1_direction",
    "race1_condition",
    "race2_racecourse",
    "race2_venue",
    "race2_track",
    "race2_surface",
    "race2_distance",
    "race2_turn",
    "race2_direction",
    "race2_condition",
    "race3_racecourse",
    "race3_venue",
    "race3_track",
    "race3_surface",
    "race3_distance",
    "race3_turn",
    "race3_direction",
    "race3_condition",
    "previous_date",
    "previous_track",
    "previous_race",
    "previous_finish",
    "previous_jockey",
    "previous_weight",
    "previous_body_weight",
    "前走日付",
    "前走競馬場",
    "前走レース",
    "前走着順",
    "前走騎手",
    "前走斤量",
    "前走馬体重",
)


class NarNewspaperParseError(ValueError):
    """Raised when NAR newspaper HTML cannot be converted to entry data."""


def is_nar_newspaper_html(text: str) -> bool:
    source = html_lib.unescape(str(text or ""))
    lower = source.lower()
    head = lower[:150_000]
    return (
        "<html" in lower
        and ("newspaper" in head or "競馬新聞" in source[:150_000])
        and ("nar.netkeiba.com" in head or "地方競馬" in source[:150_000])
    )


def extract_race_id_from_nar_newspaper_html(html: str) -> str:
    source = html_lib.unescape(str(html or ""))
    head = source[:150_000]
    for tag_pattern in (
        r"<link\b[^>]*rel=['\"]?canonical['\"]?[^>]*>",
        r"<meta\b[^>]*(?:property|name)=['\"]?og:url['\"]?[^>]*>",
    ):
        for tag in re.findall(tag_pattern, head, flags=re.I):
            value = _attr(tag, "href") or _attr(tag, "content")
            race_id = _first_race_id(value)
            if race_id:
                return race_id
    return _first_race_id(head)


def parse_nar_newspaper_html(html: str) -> dict[str, Any]:
    if not is_nar_newspaper_html(html):
        raise NarNewspaperParseError("地方競馬新聞HTMLとして判定できませんでした。")

    records = _extract_horse_records(html)
    if not records:
        raise NarNewspaperParseError("競馬新聞HTMLから馬データを取得できませんでした。")

    race_id = extract_race_id_from_nar_newspaper_html(html)
    return {
        "race_id": race_id,
        "data_type": "newspaper",
        "race": _extract_race_info(html),
        "horses": records,
    }


def build_entry_from_nar_newspaper(newspaper_data: dict[str, Any]) -> dict[str, Any]:
    horses = []
    for item in newspaper_data.get("horses", []):
        horse = {
            "frame_number": item.get("frame_number", ""),
            "horse_number": item.get("horse_number", ""),
            "horse_id": item.get("horse_id", ""),
            "horse_name": item.get("horse_name", ""),
            "sex_age": item.get("sex_age", ""),
            "weight": item.get("weight", ""),
            "jockey": item.get("jockey", ""),
            "trainer": item.get("trainer", ""),
            "affiliation": item.get("affiliation", ""),
            "horse_weight": item.get("horse_weight", ""),
            "odds": item.get("odds", ""),
            "popularity": item.get("popularity", ""),
            "running_style": item.get("running_style", ""),
            "style": item.get("running_style", ""),
            "race_interval": item.get("race_interval", ""),
            "stable_comment": item.get("stable_comment", ""),
            "pace_prediction": item.get("pace_prediction", ""),
            "ai_mark": item.get("ai_mark", ""),
            "early_3f": item.get("early_3f", ""),
            "late_3f": item.get("late_3f", ""),
        }
        for key in _PREVIOUS_RUN_KEYS:
            value = item.get(key)
            if value not in (None, ""):
                horse[key] = value
        horses.append(horse)

    return {
        "race_id": str(newspaper_data.get("race_id", "")).strip(),
        "data_type": "entry",
        "race": newspaper_data.get("race") or {},
        "horses": horses,
        "source": "nar_newspaper_html",
        "suggested_file_name": str(newspaper_data.get("_uploaded_file_name") or "nar_newspaper_entry.html"),
    }


def _extract_horse_records(html: str) -> list[dict[str, Any]]:
    source = html_lib.unescape(str(html or ""))
    vertical_records = _extract_horse_records_from_vertical_blocks(source)
    if vertical_records:
        return sorted(vertical_records, key=lambda item: _number_sort_key(str(item.get("horse_number", ""))))

    records: list[dict[str, Any]] = []
    seen_numbers: set[str] = set()
    for row_html in re.findall(r"<tr\b[^>]*>([\s\S]*?)</tr>", source, flags=re.I):
        if "/horse/" not in row_html and "Horse_Info" not in row_html and "HorseName" not in row_html:
            continue
        record = _record_from_row(row_html)
        number = str(record.get("horse_number", "")).strip()
        name = str(record.get("horse_name", "")).strip()
        if not number or not name or number in seen_numbers:
            continue
        seen_numbers.add(number)
        records.append(record)
    return sorted(records, key=lambda item: _number_sort_key(str(item.get("horse_number", ""))))


def _extract_horse_records_from_vertical_blocks(source: str) -> list[dict[str, Any]]:
    starts = []
    for match in re.finditer(r"<dl\b(?P<attrs>[^>]*)>", source, flags=re.I):
        attrs = match.group("attrs") or ""
        if "HorseList" not in attrs or "past_tr_" not in attrs:
            continue
        starts.append((match.start(), attrs))

    records: list[dict[str, Any]] = []
    seen_numbers: set[str] = set()
    for index, (start, attrs) in enumerate(starts):
        end = starts[index + 1][0] if index + 1 < len(starts) else len(source)
        block = source[start:end]
        record = _record_from_vertical_block(block, attrs)
        number = str(record.get("horse_number", "")).strip()
        name = str(record.get("horse_name", "")).strip()
        if not number or not name or number in seen_numbers:
            continue
        seen_numbers.add(number)
        records.append(record)
    return records


def _record_from_vertical_block(block: str, attrs: str) -> dict[str, Any]:
    horse_number = _attr_from_text(attrs, "id")
    match = re.search(r"past_tr_(\d{1,2})", horse_number)
    horse_number = match.group(1) if match else ""
    if not horse_number:
        horse_number = _class_text(block, "Waku_Horse")

    frame_number = _extract_vertical_frame_number(block)
    horse_link_html = _class_body(block, "HorseName") or _class_body(block, "Horse02")
    horse_id, horse_name = _extract_horse_link(horse_link_html or block[:2500])

    style_text = _class_text(block, "Horse06")
    running_style = _normalize_style(style_text)
    race_interval = _extract_race_interval(style_text)

    horse07_body = _class_body(block, "Horse07")
    horse07_text = _clean_text(horse07_body)
    body_weight = _extract_body_weight(horse07_text) or _extract_body_weight(block)
    body_value, body_diff = _split_body_weight(body_weight)

    jockey_body = _class_body(block, "Jockey")
    jockey_text = _clean_text(jockey_body)
    info_end = block.find("Horse06")
    info_prefix = block[:info_end] if info_end > 0 else block[:3500]
    raw_trainer = _link_text(info_prefix, "/trainer/")
    affiliation, trainer = _split_trainer_affiliation(raw_trainer)
    if not affiliation:
        affiliation = _extract_affiliation(_clean_text(info_prefix), trainer)

    record = {
        "frame_number": frame_number,
        "horse_number": horse_number,
        "horse_id": horse_id,
        "horse_name": horse_name,
        "sex_age": _extract_sex_age(jockey_text),
        "weight": _extract_carried_weight(jockey_text),
        "jockey": _clean_jockey_name(_link_text(jockey_body, "/jockey/")),
        "trainer": trainer,
        "affiliation": affiliation,
        "horse_weight": body_weight,
        "horse_weight_value": body_value,
        "horse_weight_diff": body_diff,
        "odds": _extract_vertical_odds(horse07_body),
        "popularity": _extract_vertical_popularity(horse07_text),
        "running_style": running_style,
        "race_interval": race_interval,
        "stable_comment": _extract_comment([], ("Comment", "コメント", "厩舎")),
        "pace_prediction": _extract_comment([], ("Pace", "展開")),
        "ai_mark": "",
        "early_3f": _extract_3f(_clean_text(block), "前半"),
        "late_3f": _extract_3f(_clean_text(block), "後半"),
    }
    _attach_recent_past_runs(record, block)
    return record


def _record_from_row(row_html: str) -> dict[str, Any]:
    cells = _extract_cells(row_html)
    row_text = _clean_text(row_html)
    horse_link = re.search(r"<a\b[^>]*href=['\"][^'\"]*/horse/([^/'\"?]+)[^'\"]*['\"][^>]*>([\s\S]*?)</a>", row_html, flags=re.I)
    horse_id = horse_link.group(1).strip() if horse_link else ""
    horse_name = _clean_text(horse_link.group(2)) if horse_link else _first_nonempty_cell(cells, ("Horse_Info", "HorseName", "Horse_Name"))

    horse_number = _cell_number(cells, ("Umaban", "Horse_Num", "HorseNum", "HorseList_Num", "UmaBan"))
    if not horse_number:
        horse_number = _infer_horse_number(cells, horse_name)

    frame_number = _cell_number(cells, ("Waku", "Frame", "枠"))
    if not frame_number:
        frame_number = _infer_frame_number(cells, horse_number)

    weight, sex_age = _extract_weight_and_sex_age(cells, row_text)
    body_weight = _extract_body_weight(row_text)
    body_value, body_diff = _split_body_weight(body_weight)
    odds = _extract_odds(cells, row_html)
    popularity = _extract_popularity(cells, row_html)

    raw_trainer = _link_text(row_html, "/trainer/")
    affiliation, trainer = _split_trainer_affiliation(raw_trainer)
    if not affiliation:
        affiliation = _extract_affiliation(row_text, trainer)

    record = {
        "frame_number": frame_number,
        "horse_number": horse_number,
        "horse_id": horse_id,
        "horse_name": horse_name,
        "sex_age": sex_age,
        "weight": weight,
        "jockey": _link_text(row_html, "/jockey/") or _first_nonempty_cell(cells, ("Jockey", "騎手")),
        "trainer": trainer,
        "affiliation": affiliation,
        "horse_weight": body_weight,
        "horse_weight_value": body_value,
        "horse_weight_diff": body_diff,
        "odds": odds,
        "popularity": popularity,
        "running_style": _extract_running_style(cells, row_text),
        "stable_comment": _extract_comment(cells, ("Comment", "コメント", "厩舎")),
        "pace_prediction": _extract_comment(cells, ("Pace", "展開")),
        "ai_mark": _extract_mark(cells, row_text),
        "early_3f": _extract_3f(row_text, "前半"),
        "late_3f": _extract_3f(row_text, "後半"),
    }
    _attach_recent_past_runs(record, row_html)
    return record


def _extract_cells(row_html: str) -> list[tuple[str, str, str]]:
    cells = []
    for match in re.finditer(r"<td\b(?P<attrs>[^>]*)>(?P<body>[\s\S]*?)</td>", row_html, flags=re.I):
        attrs = match.group("attrs") or ""
        body = match.group("body") or ""
        cells.append((attrs, body, _clean_text(body)))
    return cells


def _extract_latest_past_run(source: str) -> dict[str, Any]:
    best: dict[str, Any] = {}
    for segment in _past_run_segments(source):
        record = _extract_past_run_from_segment(segment)
        if not (record.get("previous_jockey") or record.get("previous_weight")):
            continue
        if not best:
            best = record
        elif _same_previous_run_fragment(best, record):
            for key, value in record.items():
                if value not in (None, "") and best.get(key) in (None, ""):
                    best[key] = value
        else:
            break
        if best.get("previous_jockey") and best.get("previous_weight"):
            return best
    return best


def _attach_recent_past_runs(record: dict[str, Any], source: str) -> None:
    past_runs = _extract_recent_past_runs(source)
    if not past_runs:
        record.update(_extract_latest_past_run(source))
        return

    latest = dict(past_runs[0])
    record.update({key: value for key, value in latest.items() if value not in (None, "")})
    record["past_runs"] = past_runs
    record["recent_runs"] = past_runs
    for index, run in enumerate(past_runs[:3], start=1):
        run_key = f"race{index}"
        record[f"{run_key}_racecourse"] = run.get("racecourse", "")
        record[f"{run_key}_venue"] = run.get("venue", "")
        record[f"{run_key}_track"] = run.get("track", "")
        record[f"{run_key}_surface"] = run.get("surface", "")
        record[f"{run_key}_distance"] = run.get("distance")
        record[f"{run_key}_turn"] = run.get("turn", "")
        record[f"{run_key}_direction"] = run.get("direction", "")
        record[f"{run_key}_condition"] = run.get("condition", "")


def _extract_recent_past_runs(source: str, limit: int = 3) -> list[dict[str, Any]]:
    runs: list[dict[str, Any]] = []
    for segment in _past_run_segments(source):
        record = _extract_past_run_from_segment(segment)
        if not _has_past_run_material(record):
            continue
        if runs and _same_previous_run_fragment(runs[-1], record):
            for key, value in record.items():
                if value not in (None, "") and runs[-1].get(key) in (None, ""):
                    runs[-1][key] = value
        else:
            runs.append(record)
        if len(runs) >= limit:
            break

    labels = ("last", "2back", "3back")
    keys = ("race1", "race2", "race3")
    for index, run in enumerate(runs):
        run["label"] = labels[index] if index < len(labels) else f"{index + 1}back"
        run["key"] = keys[index] if index < len(keys) else f"race{index + 1}"
        run["run_key"] = run["key"]
    return runs[:limit]


def _has_past_run_material(record: dict[str, Any]) -> bool:
    return any(
        record.get(key)
        for key in (
            "previous_date",
            "previous_track",
            "previous_race",
            "previous_finish",
            "previous_jockey",
            "previous_weight",
            "distance",
            "racecourse",
        )
    )


def _same_previous_run_fragment(base: dict[str, Any], candidate: dict[str, Any]) -> bool:
    """Return true when adjacent HTML fragments appear to describe the same latest run."""

    for key in ("previous_date", "previous_race", "previous_finish"):
        left = str(base.get(key) or "").strip()
        right = str(candidate.get(key) or "").strip()
        if left and right and left != right:
            return False
    return True


def _past_run_segments(source: str) -> list[str]:
    segments: list[str] = []
    seen: set[str] = set()
    for tag in ("li", "div", "tr", "dd", "dt"):
        pattern = rf"<{tag}\b[^>]*>[\s\S]*?</{tag}>"
        for match in re.finditer(pattern, source, flags=re.I):
            segment = match.group(0)
            if _looks_like_past_run(segment):
                key = _clean_text(segment)[:240]
                if key and key not in seen:
                    seen.add(key)
                    segments.append(segment)

    if not segments:
        for match in re.finditer(r"<div\b[^>]*>[\s\S]*?</div>", source, flags=re.I):
            segment = match.group(0)
            if _looks_like_past_run(segment):
                key = _clean_text(segment)[:240]
                if key and key not in seen:
                    seen.add(key)
                    segments.append(segment)

    if not segments and _looks_like_past_run(source):
        segments.append(source)
    return segments


def _looks_like_past_run(segment: str) -> bool:
    text = _clean_text(segment)
    if not text:
        return False
    start_tag = re.match(r"<[a-z0-9]+\b[^>]*>", segment, flags=re.I)
    if start_tag and "HorseList" in start_tag.group(0) and ("Horse_Info" in segment or "HorseName" in segment):
        return False
    has_date = _extract_previous_date(text) != ""
    has_race_link = re.search(r"/race/|race_id=", segment, flags=re.I) is not None
    has_past_hint = re.search(r"Past|past|History|history|Result|result|過去走|前走|近走|戦績", segment) is not None
    has_finish = re.search(r"(?:\d{1,2}\s*着|中止|取消|除外|失格)", text) is not None
    has_jockey = _extract_previous_jockey(segment, text) != ""
    has_weight = _extract_past_load_weight(segment) != ""
    # Current horse header cells can contain a jockey and carried weight, so require
    # a past-run cue such as a date, race link, or explicit past-run class/text.
    return (has_date or has_race_link or has_past_hint) and (has_finish or has_jockey or has_weight)


def _extract_past_run_from_segment(segment: str) -> dict[str, Any]:
    text = _clean_text(segment)
    previous_jockey_raw = _extract_previous_jockey_raw(segment, text)
    previous_jockey = _clean_jockey_name(previous_jockey_raw)
    course_text = _class_text(segment, "Course") or text
    previous_track = _class_text(segment, "Place") or _extract_previous_track(text)
    previous_surface = _extract_previous_surface(course_text)
    previous_distance = _extract_previous_distance(course_text)
    previous_turn = _extract_previous_turn(course_text)
    condition = _build_previous_condition(previous_track, previous_surface, previous_distance, previous_turn)
    record = {
        "previous_date": _extract_previous_date(text),
        "previous_track": previous_track,
        "previous_race": _extract_previous_race(segment),
        "previous_finish": _extract_previous_finish(text),
        "previous_jockey": previous_jockey,
        "previous_weight": _extract_past_load_weight(segment),
        "previous_body_weight": _extract_body_weight(text),
        # Keep only the class evidence that is actually present in this
        # saved past-run block.  The NAR/JRA-specific rank conversion is done
        # later by the prediction parser; this layer never guesses a class.
        "class_text": _extract_previous_class_text(segment, text),
        "racecourse": previous_track,
        "venue": previous_track,
        "track": previous_track,
        "surface": previous_surface,
        "distance": previous_distance,
        "turn": previous_turn,
        "direction": previous_turn,
        "condition": condition,
        "_debug_previous_jockey_raw": previous_jockey_raw,
        "_debug_previous_jockey_normalized": previous_jockey,
    }
    aliases = {
        "前走日付": record["previous_date"],
        "前走競馬場": record["previous_track"],
        "前走レース": record["previous_race"],
        "前走着順": record["previous_finish"],
        "前走騎手": record["previous_jockey"],
        "前走斤量": record["previous_weight"],
        "前走馬体重": record["previous_body_weight"],
    }
    record.update({key: value for key, value in aliases.items() if value})
    return {key: value for key, value in record.items() if value not in (None, "")}


def _extract_previous_class_text(segment: str, visible_text: str) -> str:
    """Return provider class/grade evidence from one past-run fragment."""

    parts: list[str] = []

    def add(value: str) -> None:
        value = _clean_text(value)
        if value and value not in parts:
            parts.append(value)

    for match in re.finditer(
        r"<(?P<tag>[a-z0-9]+)\b(?P<attrs>[^>]*)class=['\"](?P<class>[^'\"]*(?:Grade|Class|Kumi)[^'\"]*)['\"](?P<rest>[^>]*)>(?P<body>[\s\S]*?)</(?P=tag)>",
        segment or "",
        flags=re.I,
    ):
        add(" ".join([match.group("class") or "", _clean_text(match.group("body") or "")]))

    # Local-class race names often contain the only usable evidence (B3, C2,
    # A級など), so retain those literal tokens without assigning a rank here.
    token_source = " ".join(parts + [_extract_previous_race(segment)])
    for match in re.finditer(
        r"(?<![A-Za-z0-9])(?:Jpn[123]|G(?:I{1,3}|[123])|L|OP|OPEN|重賞|準重賞|"
        r"[ABCＡＢＣ]\s*(?:級\s*)?\d{1,2}|[ABCＡＢＣ]\s*級|\d勝クラス)(?!\d)",
        token_source,
        flags=re.I,
    ):
        add(match.group(0))
    return " ".join(parts)


def _extract_previous_date(text: str) -> str:
    source = str(text or "")
    for match in re.finditer(r"(\d{4})[./-](\d{1,2})[./-](\d{1,2})|(?<!\d)(\d{1,2})[./-](\d{1,2})(?!\d)", source):
        if match.group(1):
            month = int(match.group(2))
            day = int(match.group(3))
        else:
            month = int(match.group(4))
            day = int(match.group(5))
        if 1 <= month <= 12 and 1 <= day <= 31:
            return match.group(0)
    return ""


def _extract_previous_track(text: str) -> str:
    tracks = (
        "門別",
        "盛岡",
        "水沢",
        "浦和",
        "船橋",
        "大井",
        "川崎",
        "金沢",
        "笠松",
        "名古屋",
        "園田",
        "姫路",
        "高知",
        "佐賀",
        "帯広",
        "札幌",
        "函館",
        "福島",
        "新潟",
        "東京",
        "中山",
        "中京",
        "京都",
        "阪神",
        "小倉",
    )
    for track in tracks:
        if track in str(text or ""):
            return track
    return ""


def _extract_previous_surface(text: str) -> str:
    source = str(text or "")
    if "ダート" in source or "ダ" in source:
        return "ダ"
    if "芝" in source:
        return "芝"
    return ""


def _extract_previous_distance(text: str) -> int | None:
    source = str(text or "")
    for pattern in (
        r"(?:芝|ダート|ダ)\s*(\d{3,4})\s*m?",
        r"(\d{3,4})\s*m",
    ):
        match = re.search(pattern, source, flags=re.I)
        if match:
            try:
                return int(match.group(1))
            except ValueError:
                return None
    return None


def _extract_previous_turn(text: str) -> str:
    source = str(text or "")
    if "右" in source:
        return "右"
    if "左" in source:
        return "左"
    if "直" in source:
        return "直"
    return ""


def _build_previous_condition(track: str, surface: str, distance: int | None, turn: str) -> str:
    course = ""
    if surface:
        course += surface
    if distance is not None:
        course += f"{distance}m"
    if turn:
        course += turn
    return "".join(part for part in (track, course) if part)


def _extract_previous_race(segment: str) -> str:
    for match in re.finditer(
        r"<a\b[^>]*href=['\"][^'\"]*(?:/race/|race_id=)[^'\"]*['\"][^>]*>([\s\S]*?)</a>",
        segment,
        flags=re.I,
    ):
        text = _clean_text(match.group(1))
        link_html = match.group(0).lower()
        if text and "/horse/" not in link_html and "jockey" not in link_html:
            return text[:80]
    return ""


def _extract_previous_finish(text: str) -> str:
    match = re.search(r"(\d{1,2})\s*着", str(text or ""))
    if match:
        return f"{match.group(1)}着"
    match = re.search(r"(中止|取消|除外|失格)", str(text or ""))
    return match.group(1) if match else ""


def _extract_previous_jockey(segment: str, text: str) -> str:
    return _clean_jockey_name(_extract_previous_jockey_raw(segment, text))


def _extract_previous_jockey_raw(segment: str, text: str) -> str:
    jockey = _link_text(segment, "/jockey/")
    if jockey:
        return jockey
    for class_name in ("Jockey", "Data14"):
        jockey = _class_text(segment, class_name)
        if jockey:
            return jockey
    for pattern in (
        r"(?:騎手|鞍上)\s*[:：]?\s*([^\s　/／・,，、]+)",
        r"([一-龥ぁ-んァ-ン]{2,6})\s*騎手",
    ):
        match = re.search(pattern, text)
        if match:
            return _clean_text(match.group(1))
    return ""


def _extract_past_load_weight(segment: str) -> str:
    text = _clean_text(segment)
    for pattern in (
        r"(?:斤量|負担重量)\s*[:：]?\s*(\d{2}(?:\.\d)?)",
        r"(\d{2}(?:\.\d)?)\s*kg\s*(?:騎手|[一-龥ぁ-んァ-ン]{2,6})",
    ):
        match = re.search(pattern, text, flags=re.I)
        if match and _is_carried_weight(match.group(1)):
            return match.group(1)

    cells = _extract_cells(segment)
    for attrs, _, cell_text in cells:
        lower = attrs.lower()
        if (
            ("斤量" in attrs or "futan" in lower or "load" in lower or "carried" in lower or "weight" in lower)
            and "horseweight" not in lower
            and "body" not in lower
            and "馬体" not in cell_text
        ):
            value = _extract_carried_weight(cell_text)
            if value:
                return value

    cleaned = re.sub(r"\d{3}\s*(?:kg)?\s*\([+-]?\d+\)", " ", text, flags=re.I)
    cleaned = re.sub(r"\d{3,4}\s*m", " ", cleaned, flags=re.I)
    cleaned = re.sub(r"\d{4}[./-]\d{1,2}[./-]\d{1,2}|(?<!\d)\d{1,2}[/-]\d{1,2}(?!\d)", " ", cleaned)
    candidates = [match.group(1) for match in re.finditer(r"(?<!\d)(\d{2}(?:\.\d)?)(?!\d)", cleaned)]
    weights = [value for value in candidates if _is_carried_weight(value)]
    return weights[-1] if weights else ""


def _is_carried_weight(value: str) -> bool:
    try:
        number = float(str(value).strip())
    except ValueError:
        return False
    return 45 <= number <= 65


def _extract_vertical_frame_number(block: str) -> str:
    for match in re.finditer(r"<dt\b(?P<attrs>[^>]*)>(?P<body>[\s\S]*?)</dt>", block, flags=re.I):
        attrs = match.group("attrs") or ""
        if "Waku_Horse" in attrs:
            continue
        class_match = re.search(r"\bWaku(\d)\b", attrs)
        if class_match:
            text = _clean_text(match.group("body"))
            number = re.search(r"\d", text)
            return number.group(0) if number else class_match.group(1)
    return ""


def _extract_horse_link(source: str) -> tuple[str, str]:
    match = re.search(
        r"<a\b[^>]*href=['\"][^'\"]*/horse/([^/'\"?]+)[^'\"]*['\"][^>]*>([\s\S]*?)</a>",
        source,
        flags=re.I,
    )
    if not match:
        return "", ""
    return match.group(1).strip(), _clean_text(match.group(2))


def _extract_race_interval(text: str) -> str:
    source = _clean_text(text)
    match = re.search(r"(中\s*\d+\s*週|休み明け|連闘)", source)
    return re.sub(r"\s+", "", match.group(1)) if match else ""


def _extract_sex_age(text: str) -> str:
    match = re.search(r"([牡牝セ騙]\s*\d{1,2})", str(text or ""))
    return match.group(1).replace(" ", "") if match else ""


def _extract_carried_weight(text: str) -> str:
    weights = []
    for match in re.finditer(r"(?<!\d)(\d{2}(?:\.\d)?)(?!\d)", str(text or "")):
        value = float(match.group(1))
        if 45 <= value <= 65:
            weights.append(match.group(1))
    return weights[-1] if weights else ""


def _clean_jockey_name(value: str) -> str:
    text = _clean_text(value)
    text = re.sub(r"[\(（]\s*替\s*[\)）]", "", text)
    text = re.sub(r"^\s*(?:替|乗替|初騎乗)\s*", "", text)
    text = re.sub(r"(?<!\d)\d{2}(?:\.\d)?(?!\d).*", "", text)
    text = re.sub(r"\s+", "", text)
    return text.strip()


def _extract_vertical_odds(source: str) -> str:
    match = re.search(
        r"<span\b[^>]*class=['\"][^'\"]*\bOddsDataTxt\b[^'\"]*['\"][^>]*>([\s\S]*?)</span>",
        source,
        flags=re.I,
    )
    if match:
        return _first_float(_clean_text(match.group(1)))
    text = _clean_text(source)
    text = re.sub(r"\d{3}\s*\([+-]?\d+\)", " ", text, count=1)
    return _first_float(text)


def _extract_vertical_popularity(text: str) -> str:
    match = re.search(r"\(\s*(\d{1,2})\s*人気\)", str(text or ""))
    if match:
        return match.group(1)
    match = re.search(r"\(\s*(\d{1,2})\s*人", str(text or ""))
    return match.group(1) if match else ""


def _cell_number(cells: list[tuple[str, str, str]], class_keywords: tuple[str, ...]) -> str:
    for attrs, _, text in cells:
        if any(keyword in attrs for keyword in class_keywords):
            match = re.search(r"\d{1,2}", text)
            if match:
                return match.group(0)
    return ""


def _infer_horse_number(cells: list[tuple[str, str, str]], horse_name: str) -> str:
    for _, _, text in cells[:5]:
        if text and text != horse_name:
            match = re.fullmatch(r"\d{1,2}", text)
            if match:
                return match.group(0)
    return ""


def _infer_frame_number(cells: list[tuple[str, str, str]], horse_number: str) -> str:
    number_seen = False
    for _, _, text in cells[:4]:
        if text == horse_number:
            number_seen = True
            continue
        if number_seen:
            break
        if re.fullmatch(r"\d", text):
            return text
    return ""


def _extract_weight_and_sex_age(cells: list[tuple[str, str, str]], row_text: str) -> tuple[str, str]:
    sex_age = ""
    weight = ""
    for _, _, text in cells:
        if not sex_age:
            match = re.search(r"([牡牝セせ騸]\s*\d{1,2})", text)
            if match:
                sex_age = match.group(1).replace(" ", "").replace("せ", "セ")
        if not weight:
            match = re.fullmatch(r"(\d{2}(?:\.\d)?)", text)
            if match and 45 <= float(match.group(1)) <= 65:
                weight = match.group(1)
    if not sex_age:
        match = re.search(r"([牡牝セせ騸]\s*\d{1,2})", row_text)
        if match:
            sex_age = match.group(1).replace(" ", "").replace("せ", "セ")
    if not weight:
        for match in re.finditer(r"(?<!\d)(\d{2}(?:\.\d)?)(?!\d)", row_text):
            value = float(match.group(1))
            if 45 <= value <= 65:
                weight = match.group(1)
                break
    return weight, sex_age


def _extract_body_weight(row_text: str) -> str:
    match = re.search(r"(\d{3})\s*(?:kg)?\s*\(([+-]?\d+)\)", row_text, flags=re.I)
    return f"{match.group(1)}({match.group(2)})" if match else ""


def _split_body_weight(value: str) -> tuple[int | None, int | None]:
    match = re.search(r"(\d{3})\(([+-]?\d+)\)", str(value or ""))
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def _extract_odds(cells: list[tuple[str, str, str]], row_html: str) -> str:
    for attrs, _, text in cells:
        if "odds" in attrs.lower() or "Odds" in attrs:
            value = _first_float(text)
            if value:
                return value
    match = re.search(r"id=['\"]odds-[^'\"]*['\"][^>]*>([\s\S]*?)<", row_html, flags=re.I)
    if match:
        return _first_float(_clean_text(match.group(1)))
    return ""


def _extract_popularity(cells: list[tuple[str, str, str]], row_html: str) -> str:
    for attrs, _, text in cells:
        lower = attrs.lower()
        if "ninki" in lower or "popular" in lower or "人気" in attrs:
            match = re.search(r"\d{1,2}", text)
            if match:
                return match.group(0)
    match = re.search(r"id=['\"]ninki-[^'\"]*['\"][^>]*>([\s\S]*?)<", row_html, flags=re.I)
    if match:
        text = _clean_text(match.group(1))
        number = re.search(r"\d{1,2}", text)
        return number.group(0) if number else ""
    return ""


def _extract_running_style(cells: list[tuple[str, str, str]], row_text: str) -> str:
    for attrs, _, text in cells:
        if "DataTitle_Cell" in attrs or "脚質" in attrs or "style" in attrs.lower():
            style = _normalize_style(text)
            if style:
                return style
    for _, _, text in cells:
        if len(text) <= 3:
            style = _normalize_style(text)
            if style:
                return style
    return _normalize_style(row_text)


def _extract_comment(cells: list[tuple[str, str, str]], class_keywords: tuple[str, ...]) -> str:
    candidates = []
    for attrs, _, text in cells:
        if any(keyword.lower() in attrs.lower() for keyword in class_keywords) and len(text) >= 3:
            candidates.append(text)
    return max(candidates, key=len)[:160] if candidates else ""


def _extract_mark(cells: list[tuple[str, str, str]], row_text: str) -> str:
    marks = ("◎", "○", "◯", "▲", "△", "☆", "✓")
    for _, _, text in cells[:6]:
        for mark in marks:
            if mark in text:
                return "○" if mark == "◯" else mark
    for mark in marks:
        if mark in row_text[:80]:
            return "○" if mark == "◯" else mark
    return ""


def _extract_3f(row_text: str, label: str) -> str:
    pattern = rf"{label}\s*3F?\s*[:：]?\s*(\d{{1,2}}\.\d)"
    match = re.search(pattern, row_text, flags=re.I)
    return match.group(1) if match else ""


def _link_text(row_html: str, href_part: str) -> str:
    href_patterns = [re.escape(href_part)]
    if href_part == "/jockey/":
        href_patterns.append("jockey")
    match = re.search(
        rf"<a\b[^>]*href=['\"][^'\"]*(?:{'|'.join(href_patterns)})[^'\"]*['\"][^>]*>([\s\S]*?)</a>",
        row_html,
        flags=re.I,
    )
    return _clean_text(match.group(1)) if match else ""


def _extract_affiliation(row_text: str, trainer: str) -> str:
    if not trainer:
        return ""
    match = re.search(r"([^\s・/／]+)[・/／]\s*" + re.escape(trainer), row_text)
    return match.group(1) if match else ""


def _split_trainer_affiliation(value: str) -> tuple[str, str]:
    text = str(value or "").strip()
    for separator in ("・", "/", "／"):
        if separator in text:
            left, right = text.split(separator, 1)
            return left.strip(), right.strip()
    return "", text


def _first_nonempty_cell(cells: list[tuple[str, str, str]], class_keywords: tuple[str, ...]) -> str:
    for attrs, _, text in cells:
        if any(keyword in attrs for keyword in class_keywords) and text:
            return text
    return ""


def _first_float(text: str) -> str:
    match = re.search(r"\d+(?:\.\d+)?", str(text or ""))
    return match.group(0) if match else ""


def _extract_race_info(html: str) -> dict[str, str]:
    return {
        "race_name": _class_text(html, "RaceName") or _title_text(html),
        "race_number": _class_text(html, "RaceNum"),
        "race_data_1": _class_text(html, "RaceData01"),
        "race_data_2": _class_text(html, "RaceData02"),
    }


def _class_text(html: str, class_name: str) -> str:
    return _clean_text(_class_body(html, class_name))


def _class_body(html: str, class_name: str) -> str:
    pattern = (
        r"<(?P<tag>[a-z0-9]+)\b[^>]*class=['\"][^'\"]*\b"
        + re.escape(class_name)
        + r"\b[^'\"]*['\"][^>]*>(?P<body>[\s\S]*?)</(?P=tag)>"
    )
    match = re.search(pattern, html, flags=re.I)
    return match.group("body") if match else ""


def _title_text(html: str) -> str:
    match = re.search(r"<title[^>]*>([\s\S]*?)</title>", html, flags=re.I)
    return _clean_text(match.group(1)) if match else ""


def _clean_text(value: str) -> str:
    text = re.sub(r"<[^>]+>", " ", str(value or ""))
    text = html_lib.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _normalize_style(value: Any) -> str:
    text = str(value or "").strip()
    if "逃" in text:
        return "逃"
    if "先" in text:
        return "先"
    if "差" in text:
        return "差"
    if "追" in text:
        return "追"
    return ""


def _attr(tag: str, attr_name: str) -> str:
    match = re.search(rf"\b{re.escape(attr_name)}\s*=\s*(['\"])(.*?)\1", tag, flags=re.I)
    return html_lib.unescape(match.group(2)) if match else ""


def _attr_from_text(text: str, attr_name: str) -> str:
    return _attr(str(text or ""), attr_name)


def _first_race_id(text: str) -> str:
    source = str(text or "")
    for pattern in (
        r"race_id(?:=|%3D)(\d{12})",
        r"race_id['\"]?\s*[:=]\s*['\"](\d{12})",
    ):
        match = re.search(pattern, source, flags=re.I)
        if match:
            return match.group(1)
    return ""


def _number_sort_key(value: str) -> tuple[int, str]:
    try:
        return int(value), value
    except ValueError:
        return 999, value

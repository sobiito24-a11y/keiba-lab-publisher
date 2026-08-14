# -*- coding: utf-8 -*-
from __future__ import annotations

#@title 地方競馬版 解析ロジック
import math
import re
import time
import unicodedata
from datetime import date
from functools import lru_cache
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .audit_features import add_audit_evaluation_columns
from .condition_fit import extract_condition_fit_sources
from .nar_newspaper_parser import parse_nar_newspaper_html as parse_uploaded_nar_newspaper_html
from .star_index import build_star_max_result, star_match_level
from .ver3_ability import calculate_ver3_ability_core
from .star_trace import candidate_summary, clear_star_trace, get_star_trace, log_star_trace, star_trace_row


USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)

LOAD_WEIGHT_INDEX_PER_KG = 3.0
RELATIVE_WEIGHT_INDEX_PER_KG = 1.5
MAX_TOTAL_WEIGHT_ADJUSTMENT = 8.0


def make_session():
    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
    })
    return session


def decode_netkeiba_response(response):
    content = response.content
    for enc in ("utf-8", "utf-8-sig", "EUC-JP", "cp932"):
        try:
            text = content.decode(enc)
            if "<html" in text.lower() or "netkeiba" in text.lower():
                return text
        except UnicodeDecodeError:
            pass
    return content.decode(response.apparent_encoding or "utf-8", errors="replace")


def fetch_html(url, session=None):
    session = session or make_session()
    response = session.get(url, timeout=25)
    response.raise_for_status()
    return decode_netkeiba_response(response)


def norm_text(text):
    return re.sub(r"\s+", " ", (text or "").replace("\xa0", " ")).strip()


def text_of(element):
    if element is None:
        return ""
    return norm_text(element.get_text(" ", strip=True))


def visible_text(element):
    if element is None:
        return ""
    clone = BeautifulSoup(str(element), "html.parser")
    for hidden in clone.select(".Sort_Function_Data_Hidden, script, style, select"):
        hidden.decompose()
    return text_of(clone)


def first(row, selectors):
    for selector in selectors:
        found = row.select_one(selector)
        if found is not None:
            return found
    return None


def parse_int_from_text(text):
    match = re.search(r"-?\d+", text or "")
    return int(match.group(0)) if match else None


def extract_age_from_sex_age(value):
    match = re.search(r"(\d+)", str(value or ""))
    return int(match.group(1)) if match else None


def parse_going_from_text(text):
    text = norm_text(text)
    patterns = [
        r"馬場\s*[:：]?\s*(不良|稍重|稍|重|良|不)",
        r"(?:芝|ダート|ダ)\s*[:：]?\s*(不良|稍重|稍|重|良|不)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            value = match.group(1)
            return {"稍": "稍重", "不": "不良"}.get(value, value)
    return ""


def parse_course_text(text):
    text = norm_text(text)
    surface = ""
    if "障" in text:
        surface = "障"
    elif "ダ" in text:
        surface = "ダ"
    elif "芝" in text:
        surface = "芝"

    distance_match = re.search(r"(\d{3,4})\s*m", text)
    distance = int(distance_match.group(1)) if distance_match else None

    if "直" in text:
        direction = "直"
    elif "左" in text:
        direction = "左"
    elif "右" in text:
        direction = "右"
    else:
        direction = ""

    side = "外" if "外" in text else ("内" if "内" in text else "")
    distance_label = f"{surface}{distance}m" if surface and distance else ""
    full_label = f"{distance_label}{direction}{side}" if distance_label else ""
    going = parse_going_from_text(text)

    return {
        "raw": text,
        "surface": surface,
        "distance": distance,
        "direction": direction,
        "side": side,
        "going": going,
        "label": full_label,
        "distance_label": distance_label,
    }


NAR_TRACK_CODES = {
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

NAR_TRACK_NAMES = [
    "門別", "盛岡", "水沢", "浦和", "船橋", "大井", "川崎", "金沢", "笠松", "名古屋", "園田", "姫路", "高知", "佐賀"
]


def parse_nar_track_from_race_id(text):
    race_id = parse_race_id_from_text(text)
    if not race_id or len(race_id) < 6:
        return ""
    return NAR_TRACK_CODES.get(race_id[4:6], "")


def parse_racecourse_from_text(*texts):
    combined = " ".join(norm_text(str(text)) for text in texts if text)
    for name in NAR_TRACK_NAMES:
        if name in combined:
            return name
    return parse_nar_track_from_race_id(combined)


CLASS_RULES = [
    (r"Jpn\s*1|JPN\s*1|JpnI|JPNI|JpnⅠ|Ｊｐｎ１", 88, "Jpn1"),
    (r"Jpn\s*2|JPN\s*2|JpnII|JPNII|JpnⅡ|Ｊｐｎ２", 78, "Jpn2"),
    (r"Jpn\s*3|JPN\s*3|JpnIII|JPNIII|JpnⅢ|Ｊｐｎ３", 68, "Jpn3"),
    (r"\bG\s*1\b|\bGI\b|GⅠ|Ｇ１", 90, "G1"),
    (r"\bG\s*2\b|\bGII\b|GⅡ|Ｇ２", 80, "G2"),
    (r"\bG\s*3\b|\bGIII\b|GⅢ|Ｇ３", 70, "G3"),
    (r"準重賞|準重", 62, "準重賞"),
    (r"重賞|S\s*1|S1|S\s*2|S2", 66, "重賞"),
    (r"リステッド|Listed|\(L\)|（L）", 65, "L"),
    (r"オープン特別|オープン|OPEN|\bOP\b", 60, "OP"),
    (r"3\s*勝クラス|3\s*勝|３\s*勝|1600万", 50, "3勝"),
    (r"2\s*勝クラス|2\s*勝|２\s*勝|1000万", 40, "2勝"),
    (r"1\s*勝クラス|1\s*勝|１\s*勝|500万", 30, "1勝"),
    (r"未勝利", 20, "未勝利"),
    (r"新馬|メイクデビュー", 10, "新馬"),
    (r"\bA\s*1\b|A1|Ａ１", 70, "A1"),
    (r"\bA\s*2\b|A2|Ａ２", 65, "A2"),
    (r"\bA\s*3\b|A3|Ａ３", 60, "A3"),
    (r"A\s*級|Ａ級|\bA\b", 64, "A級"),
    (r"\bB\s*1\b|B1|Ｂ１", 55, "B1"),
    (r"\bB\s*2\b|B2|Ｂ２", 50, "B2"),
    (r"\bB\s*3\b|B3|Ｂ３", 45, "B3"),
    (r"B\s*級|Ｂ級|\bB\b", 50, "B級"),
    (r"\bC\s*1\b|C1|Ｃ１", 40, "C1"),
    (r"\bC\s*2\b|C2|Ｃ２", 30, "C2"),
    (r"\bC\s*3\b|C3|Ｃ３", 20, "C3"),
    (r"C\s*級|Ｃ級|\bC\b", 30, "C級"),
    (r"条件戦|一般戦|一般", 25, "条件戦"),
]

GRADE_CLASS_HINT_RE = re.compile(
    r"GradeType|Icon_Grade|GradeIcon|Jpn|G1|G2|G3|GI|GII|GIII|重賞|準重賞|"
    r"リステッド|Listed|オープン|OPEN|OP|クラス|条件|A級|B級|C級|A1|A2|A3|B1|B2|B3|C1|C2|C3|[ABCＡＢＣ]\\s*(?:級\\s*)?\\d+",
    flags=re.IGNORECASE,
)

GRADE_TYPE_RULES = [
    (r"GradeType\s*1|GradeType1|grade_type_?1", 90, "G1"),
    (r"GradeType\s*2|GradeType2|grade_type_?2", 80, "G2"),
    (r"GradeType\s*3|GradeType3|grade_type_?3", 70, "G3"),
]


LOCAL_NAR_CLASS_BASE = {"A": 75, "B": 60, "C": 50}
LOCAL_NAR_CLASS_STEP = {"A": 5, "B": 5, "C": 10}


def local_abc_class_info(text):
    text = norm_text(text).translate(str.maketrans("ＡＢＣａｂｃ０１２３４５６７８９", "ABCabc0123456789")).upper()
    matches = []
    pattern = re.compile(r"(?<![A-Z0-9])([ABC])\s*(?:級\s*)?(\d{1,2})(?!\d)", flags=re.IGNORECASE)
    for match in pattern.finditer(text):
        letter = match.group(1).upper()
        number = int(match.group(2))
        if number <= 0:
            continue
        base = LOCAL_NAR_CLASS_BASE.get(letter)
        step = LOCAL_NAR_CLASS_STEP.get(letter)
        if base is None or step is None:
            continue
        rank = base - (number * step)
        matches.append((rank, f"{letter}{number}"))
    if not matches:
        return None, ""
    return max(matches, key=lambda item: item[0])


def race_class_info(text):
    text = norm_text(text).translate(str.maketrans("ＡＢＣａｂｃ０１２３４５６７８９", "ABCabc0123456789"))
    local_rank, local_label = local_abc_class_info(text)
    for pattern, rank, label in CLASS_RULES:
        if re.search(pattern, text, flags=re.IGNORECASE):
            if local_rank is not None and label in ("A級", "B級", "C級", "条件戦"):
                return local_rank, local_label
            return rank, label
    if local_rank is not None:
        return local_rank, local_label
    for pattern, rank, label in GRADE_TYPE_RULES:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return rank, label
    return None, ""


def extract_class_text_candidates(soup):
    if soup is None:
        return []
    candidates = []
    selectors = [
        ".RaceName",
        ".RaceData01",
        ".RaceData02",
        ".RaceList_NameBox",
        ".RaceList_Item02",
        ".RaceList_ItemLong",
        ".data_intro h1",
        ".data_intro .racedata",
        ".data_intro .smalltxt",
        "[class*='Grade']",
        "[class*='grade']",
        "[class*='Class']",
        "[class*='class']",
    ]
    for selector in selectors:
        for node in soup.select(selector):
            text = text_of(node)
            if text:
                candidates.append(text)
            attrs = []
            for attr_name in ("class", "id", "title", "alt", "aria-label", "src"):
                value = node.get(attr_name)
                if isinstance(value, list):
                    value = " ".join(str(x) for x in value)
                if value:
                    attrs.append(str(value))
            attr_text = " ".join(attrs)
            if attr_text and GRADE_CLASS_HINT_RE.search(attr_text):
                candidates.append(attr_text)
    for node in soup.find_all(True):
        attrs = []
        for attr_name in ("class", "id", "title", "alt", "aria-label", "src"):
            value = node.get(attr_name)
            if isinstance(value, list):
                value = " ".join(str(x) for x in value)
            if value:
                attrs.append(str(value))
        attr_text = " ".join(attrs)
        if attr_text and GRADE_CLASS_HINT_RE.search(attr_text):
            node_text = text_of(node)
            candidates.append(" ".join([attr_text, node_text]).strip())
    unique = []
    for text in candidates:
        text = norm_text(text)
        if text and text not in unique:
            unique.append(text)
    return unique


def race_class_info_from_soup(soup, *fallback_texts):
    for text in extract_class_text_candidates(soup):
        rank, label = race_class_info(text)
        if rank is not None:
            return rank, label
    compact_fallback = " ".join(norm_text(str(text)) for text in fallback_texts if text)
    return race_class_info(compact_fallback)


def class_shift_label(current_rank, past_rank):
    if current_rank is None or past_rank is None:
        return ""
    diff = current_rank - past_rank
    if diff >= 8:
        return "クラス昇級"
    if diff <= -8:
        return "クラス降級"
    return "同級"



def parse_race_date_from_race_id(text):
    match = re.search(r"race_id=(\d{12})", text or "") or re.search(r"\b(\d{12})\b", text or "")
    if not match:
        return None
    race_id = match.group(1)
    try:
        return date(int(race_id[:4]), int(race_id[6:8]), int(race_id[8:10]))
    except ValueError:
        return None


def parse_race_id_from_text(text):
    match = re.search(r"race_id=(\d{12})", text or "") or re.search(r"\b(\d{12})\b", text or "")
    return match.group(1) if match else ""


def parse_race_date_from_text(text):
    match = re.search(r"(\d{4})年\s*(\d{1,2})月\s*(\d{1,2})日", text or "")
    if not match:
        return None
    try:
        return date(int(match.group(1)), int(match.group(2)), int(match.group(3)))
    except ValueError:
        return None


def days_between(current_date, past_date):
    if not current_date or not past_date:
        return None
    days = (current_date - past_date).days
    return days if days >= 0 else None


def format_interval_from_days(days):
    if days is None or pd.isna(days):
        return "-"
    try:
        days = int(days)
    except Exception:
        return "-"
    # 前走日・今回日が取れず0日扱いになった場合の誤表示を避ける。
    if days <= 0:
        return "-"
    if days >= 60:
        return "休み明け"
    weeks = max(0, days // 7)
    return f"中{weeks}週"

def parse_current_race_info(html):
    soup = BeautifulSoup(html, "html.parser")
    race_name = text_of(soup.select_one(".RaceName"))
    race_data = text_of(soup.select_one(".RaceData01"))
    race_data2 = text_of(soup.select_one(".RaceData02"))
    title_text = text_of(soup.title)
    race_id = parse_race_id_from_text(html)
    info = parse_course_text(race_data)
    class_rank, class_label = race_class_info_from_soup(soup, race_name, race_data, race_data2, title_text)
    info.update({
        "race_name": race_name,
        "race_data": race_data,
        "race_data2": race_data2,
        "racecourse": parse_nar_track_from_race_id(race_id) or parse_racecourse_from_text(race_name, race_data, race_data2, title_text),
        "race_date": parse_race_date_from_race_id(html) or parse_race_date_from_text(" ".join([race_name, race_data, race_data2, title_text])),
        "class_rank": class_rank,
        "class_label": class_label,
    })
    return info


def parse_index_cell(cell):
    raw = ""
    if cell is not None:
        # netkeiba puts a hidden sort value (often 100) in empty index cells.
        # Colab reads the displayed link text, so prefer the anchor text here too.
        anchor = cell.find("a")
        raw = visible_text(anchor) if anchor is not None else visible_text(cell)
    cleaned = raw.replace("*", "").strip()
    if cleaned in {"", "-", "未", "未取得", "None", "none", "nan", "<NA>"}:
        value = None
    else:
        value = parse_int_from_text(cleaned)
    link = ""
    if cell is not None:
        a = cell.find("a", href=True)
        if a:
            link = urljoin("https://db.netkeiba.com", a["href"])
    info = {"value": value, "raw": raw, "url": link, "race_date": parse_race_date_from_race_id(link)}
    attr_get = getattr(cell, "get", None)
    if cell is not None and callable(attr_get):
        info.update({
            "racecourse": str(attr_get("data-star-venue", "") or attr_get("data-star-racecourse", "") or "").strip(),
            "surface": str(attr_get("data-star-surface", "") or "").strip(),
            "distance": parse_int_from_text(str(attr_get("data-star-distance", "") or "")),
            "direction": str(attr_get("data-star-turn", "") or attr_get("data-star-direction", "") or "").strip(),
            "label": str(attr_get("data-star-condition", "") or "").strip(),
        })
    return info


def parse_float_from_text(text):
    match = re.search(r"-?\d+(?:\.\d+)?", text or "")
    return float(match.group(0)) if match else None


def parse_bool_from_text(text):
    value = str(text or "").strip().lower()
    return value in {"true", "1", "yes", "y", "○", "あり", "替", "乗り替わり", "乗替"}


def parse_jockey_changed_from_text(text):
    value = str(text or "").strip().lower()
    if value in {"pending", "unknown", "hold", "判定保留", "保留"}:
        return "pending"
    return parse_bool_from_text(text)


def display_attr(row, name):
    if row is None:
        return ""
    try:
        value = row.get(f"data-display-{name}", "")
    except Exception:
        return ""
    return str(value or "").strip()


def is_same_racecourse(current, past):
    current_course = current.get("racecourse") or ""
    past_course = past.get("racecourse") or ""
    return bool(current_course and past_course and current_course == past_course)


def is_same_condition(current, past):
    return star_match_level(current, past) != "none"


def is_same_distance(current, past):
    return bool(
        current.get("surface")
        and current.get("distance")
        and current.get("surface") == past.get("surface")
        and current.get("distance") == past.get("distance")
    )


def is_similar_condition(current, past):
    return bool(
        not is_same_racecourse(current, past)
        and current.get("surface")
        and current.get("distance")
        and current.get("direction")
        and current.get("surface") == past.get("surface")
        and current.get("distance") == past.get("distance")
        and current.get("direction") == past.get("direction")
    )


def format_prev_run(cell_info, past_info, same_condition):
    value = cell_info.get("value")
    raw = cell_info.get("raw") or ""
    label = past_info.get("label") or ""
    racecourse = past_info.get("racecourse") or ""
    display_label = f"{racecourse}{label}" if racecourse and label else label
    star = "★" if same_condition else ""

    if value is None:
        display_value = raw if raw else "-"
    else:
        display_value = str(value)

    return f"{display_value}/{display_label}{star}" if display_label else f"{display_value}{star}"


def compact_number(value):
    try:
        number = float(value)
    except Exception:
        return str(value)
    if abs(number - round(number)) < 0.05:
        return str(int(round(number)))
    return f"{number:.1f}".rstrip("0").rstrip(".")


def format_load_weight_with_change(current_text, previous_weight):
    current_weight = parse_float_from_text(current_text)
    if current_weight is None or previous_weight is None:
        return current_text
    diff = current_weight - float(previous_weight)
    if abs(diff) < 0.05:
        return current_text
    sign = "+" if diff > 0 else ""
    return f"{compact_number(current_weight)}({sign}{compact_number(diff)})"


def load_weight_index_adjustment(change):
    if change is None or pd.isna(change):
        return 0.0
    return round(-float(change) * LOAD_WEIGHT_INDEX_PER_KG, 1)


def relative_load_weight_adjustment(current_weight, field_average):
    if current_weight is None or pd.isna(current_weight) or field_average is None or pd.isna(field_average):
        return 0.0
    return round((float(field_average) - float(current_weight)) * RELATIVE_WEIGHT_INDEX_PER_KG, 1)


def format_percent_value(value):
    number = parse_float_from_text(str(value or ""))
    return f"{compact_number(number)}%" if number is not None else ""


def get_race_id_from_html_or_name(html, fallback="nar_race"):
    for pattern in [r"race_id=(\d+)", r"MyRace-Item-(\d+)", r"/race/speed\.html\?race_id=(\d+)"]:
        match = re.search(pattern, html or "")
        if match:
            return match.group(1)
    return fallback


def extract_horse_id_from_link(link):
    href = link.get("href", "") if link else ""
    match = re.search(r"/horse/(?:result/)?(\d+)", href)
    return match.group(1) if match else ""


def compact_name(text):
    return re.sub(r"\s+", "", text or "")


def circled_number_text(value):
    chars = {
        1: "①", 2: "②", 3: "③", 4: "④", 5: "⑤", 6: "⑥", 7: "⑦", 8: "⑧", 9: "⑨", 10: "⑩",
        11: "⑪", 12: "⑫", 13: "⑬", 14: "⑭", 15: "⑮", 16: "⑯", 17: "⑰", 18: "⑱",
    }
    try:
        number = int(value)
        return chars.get(number, str(number))
    except Exception:
        return str(value)


def extract_past_race_results(past_soup):
    table = past_soup.select_one("table.race_table_01")
    if table is None:
        return {}

    header_cells = table.select_one("tr").find_all(["td", "th"], recursive=False) if table.select_one("tr") else []
    headers = [visible_text(cell) for cell in header_cells]
    load_weight_index = next((i for i, header in enumerate(headers) if "斤量" in header), None)

    results = {}
    for row in table.select("tr")[1:]:
        cells = row.find_all(["td", "th"], recursive=False)
        if len(cells) < 4:
            continue
        position = parse_int_from_text(visible_text(cells[0]))
        horse_no = parse_int_from_text(visible_text(cells[2])) if len(cells) > 2 else None
        horse_name = visible_text(cells[3])
        if load_weight_index is not None and len(cells) > load_weight_index:
            load_weight = parse_float_from_text(visible_text(cells[load_weight_index]))
        else:
            load_weight = parse_float_from_text(visible_text(cells[5])) if len(cells) > 5 else None
        if position is None or not horse_name:
            continue
        results[compact_name(horse_name)] = {
            "position": position,
            "horse_no": horse_no,
            "horse_name": horse_name,
            "load_weight": load_weight,
        }
    return results


def extract_jockey_from_row(row):
    if row is None:
        return ""

    for selector in [
        ".Jockey a",
        ".Jockey",
        ".JockeyName a",
        ".JockeyName",
        "[class*='Jockey'] a",
        "[class*='Jockey']",
    ]:
        value = visible_text(row.select_one(selector))
        if value:
            return value.replace("騎手", "").strip()

    table = row.find_parent("table")
    if table is not None:
        headers = [visible_text(th) for th in table.select("thead th")]
        jockey_index = next((i for i, header in enumerate(headers) if "騎手" in header), None)
        if jockey_index is not None:
            cells = row.find_all(["td", "th"], recursive=False)
            if len(cells) > jockey_index:
                value = visible_text(cells[jockey_index])
                if value:
                    return value.replace("騎手", "").strip()

    cells = row.find_all(["td", "th"], recursive=False)
    horse_cell_index = None
    for index, cell in enumerate(cells):
        if cell.select_one('a[href*="/horse/"]') or cell.select_one('a[href*="horse/result"]'):
            horse_cell_index = index
            break
    if horse_cell_index is not None:
        for offset in (3, 2, 4):
            target_index = horse_cell_index + offset
            if len(cells) > target_index:
                value = visible_text(cells[target_index])
                if value and not re.search(r"\d{1,2}:\d{2}", value):
                    return value.replace("騎手", "").strip()

    return ""


def extract_jockey_from_past_race(past_soup, horse_name, horse_id=""):
    target_anchors = []
    if horse_id:
        target_anchors = past_soup.select(f'a[href*="{horse_id}"]')
    if not target_anchors and horse_name:
        target_name = compact_name(horse_name)
        target_anchors = [
            a for a in past_soup.find_all("a")
            if compact_name(text_of(a)) == target_name
        ]

    for anchor in target_anchors:
        row = anchor.find_parent("tr")
        jockey = extract_jockey_from_row(row)
        if jockey:
            return jockey
    return ""


def parse_nar_style_table(html):
    if not html:
        return pd.DataFrame()

    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("#table_sort_back") or soup.select_one("table.Data01_Table")
    if table is None:
        return pd.DataFrame()

    records = []
    for row in table.select("tbody tr.HorseList") or table.select("tbody tr"):
        cells = row.find_all("td", recursive=False)
        if len(cells) < 4:
            continue

        horse_no = parse_int_from_text(visible_text(cells[0]))
        horse_name = visible_text(row.select_one(".Horse_Info a"))
        running_style = visible_text(row.select_one(".DataTitle_Cell"))
        if not horse_no or not horse_name or not running_style:
            continue

        def cell_text(index):
            return visible_text(cells[index]) if len(cells) > index else ""

        records.append({
            "馬番": horse_no,
            "馬名": horse_name,
            "脚質": running_style,
            "脚質勝率": cell_text(8),
            "脚質連対率": cell_text(9),
            "脚質複勝率": cell_text(10),
            "脚質単回収": cell_text(11),
            "脚質複回収": cell_text(12),
        })

    style_df = pd.DataFrame(records)
    if style_df.empty:
        return style_df

    odds_records = []
    for row in soup.select(".DataModal tr.HorseList"):
        cells = row.find_all("td", recursive=False)
        if not cells:
            continue
        horse_no = parse_int_from_text(visible_text(cells[0]))
        if not horse_no:
            continue
        odds = parse_float_from_text(visible_text(row.select_one('[id^="odds-1_"]')))
        popularity = parse_int_from_text(visible_text(row.select_one('[id^="ninki-1_"]')))
        odds_records.append({
            "馬番": horse_no,
            "_脚質HTML単勝オッズ": odds,
            "_脚質HTML人気": popularity,
        })

    if odds_records:
        odds_df = pd.DataFrame(odds_records).drop_duplicates("馬番")
        style_df = style_df.merge(odds_df, on="馬番", how="left")

    return style_df.drop_duplicates("馬番")


def apply_nar_style_features(df, style_html):
    df = df.copy()
    if not style_html:
        df["脚質"] = ""
        return df, pd.DataFrame()

    style_df = parse_nar_style_table(style_html)
    if style_df.empty:
        df["脚質"] = ""
        return df, style_df

    merged = df.merge(
        style_df.drop(columns=["馬名"], errors="ignore"),
        on="馬番",
        how="left",
    )
    merged["脚質"] = merged["脚質"].fillna("")

    if "_脚質HTML単勝オッズ" in merged.columns:
        merged["単勝オッズ"] = pd.to_numeric(merged["単勝オッズ"], errors="coerce")
        merged["単勝オッズ"] = merged["単勝オッズ"].fillna(merged["_脚質HTML単勝オッズ"])
    if "_脚質HTML人気" in merged.columns:
        merged["人気"] = pd.to_numeric(merged["人気"], errors="coerce")
        merged["人気"] = merged["人気"].fillna(merged["_脚質HTML人気"])

    drop_cols = [c for c in merged.columns if c.startswith("_脚質HTML")]
    if drop_cols:
        merged = merged.drop(columns=drop_cols)
    return merged, style_df


def parse_body_weight_text(text):
    text = norm_text(text)
    if not text or "計不" in text:
        return None, None, text
    match = re.search(r"(\d{3})\s*(?:\(\s*([+-]?\d+)\s*\))?", text)
    if not match:
        return None, None, text
    weight = int(match.group(1))
    change = int(match.group(2)) if match.group(2) is not None else None
    if change is None:
        display = str(weight)
    else:
        sign = "+" if change > 0 else ""
        display = f"{weight}({sign}{change})"
    return weight, change, display


def parse_shutuba_html(html):
    if not html:
        return {}, pd.DataFrame()
    soup = BeautifulSoup(html, "html.parser")
    race_name = text_of(soup.select_one(".RaceName"))
    race_data = text_of(soup.select_one(".RaceData01"))
    race_data2 = text_of(soup.select_one(".RaceData02"))
    title_text = text_of(soup.title)
    full_text = text_of(soup)
    info = parse_course_text(" ".join([race_data, race_data2, title_text, full_text[:2000]]))
    if not info.get("going"):
        info["going"] = parse_going_from_text(full_text)
    info.update({
        "race_name": race_name,
        "race_data": race_data,
        "race_data2": race_data2,
    })

    records = []
    rows = soup.select("table.Shutuba_Table tr.HorseList") or soup.select("tr.HorseList")
    for row in rows:
        cells = row.find_all(["td", "th"], recursive=False)
        cell_texts = [visible_text(cell) for cell in cells]
        horse_no = parse_int_from_text(visible_text(first(row, [
            ".Umaban", ".Horse_Num", ".HorseNum", ".Num", ".HorseList_Num"
        ])))
        if not horse_no:
            small_numbers = []
            for text in cell_texts[:5]:
                value = parse_int_from_text(text)
                if value is not None and 1 <= value <= 18:
                    small_numbers.append(value)
            if len(small_numbers) >= 2:
                horse_no = small_numbers[1]
            elif small_numbers:
                horse_no = small_numbers[0]
        if not horse_no:
            continue

        horse_link = row.select_one('a[href*="/horse/"]')
        horse_name = text_of(horse_link) if horse_link else ""

        weight_text = visible_text(first(row, [".Weight", ".HorseWeight", ".Horse_Weight"]))
        if not weight_text:
            for text in reversed(cell_texts):
                if re.search(r"\d{3}\s*(?:\([+-]?\d+\))?", text) or "計不" in text:
                    weight_text = text
                    break
        body_weight, body_change, body_display = parse_body_weight_text(weight_text)
        if body_display:
            records.append({
                "馬番": int(horse_no),
                "_shutuba_horse_name": horse_name,
                "馬体重": body_display,
                "_body_weight": body_weight,
                "_body_weight_change": body_change,
            })

    body_df = pd.DataFrame(records)
    if not body_df.empty:
        body_df = body_df.drop_duplicates("馬番")
    return info, body_df


def body_weight_comment(change):
    if change is None or pd.isna(change):
        return ""
    try:
        change = int(change)
    except Exception:
        return ""
    if change <= -12:
        return "馬体大幅減注意"
    if change <= -8:
        return "馬体減注意"
    if change >= 12:
        return "馬体大幅増注意"
    if change >= 8:
        return "馬体増注意"
    if -3 <= change <= 3:
        return "馬体安定"
    return ""


def append_comment_part(text, part):
    if not part:
        return text
    existing = [item for item in str(text or "").split("、") if item]
    if part not in existing:
        existing.append(part)
    return "、".join(existing[:4])


def recompute_same_going_features(df, race_info):
    result = df.copy()
    going = race_info.get("going") or ""
    counts = []
    highs = []
    for _, row in result.iterrows():
        values = []
        for run in row.get("_past_runs") or []:
            value = run.get("value")
            if going and run.get("going") == going and value is not None:
                values.append(value)
        counts.append(len(values))
        highs.append(max(values) if values else None)
    result["_current_going"] = going
    result["_same_going_count"] = counts
    result["_same_going_high"] = highs
    return add_condition_context_features(result, race_info)


def apply_shutuba_features(df, race_info, shutuba_html):
    result = df.copy()
    result["馬体重"] = result.get("馬体重", pd.Series("", index=result.index)).fillna("")
    if "同馬場実績" not in result.columns:
        result["同馬場実績"] = result.get("馬場適性", pd.Series("", index=result.index)).fillna("")
    shutuba_info = {"used": False, "going": "", "body_count": 0}
    if not shutuba_html:
        return result, race_info, shutuba_info

    parsed_info, body_df = parse_shutuba_html(shutuba_html)
    shutuba_info["used"] = True
    if parsed_info.get("going"):
        race_info["going"] = parsed_info.get("going")
        shutuba_info["going"] = parsed_info.get("going")
        result = recompute_same_going_features(result, race_info)
    else:
        result = add_condition_context_features(result, race_info)

    if not body_df.empty:
        result = result.merge(
            body_df.drop(columns=["_shutuba_horse_name"], errors="ignore"),
            on="馬番",
            how="left",
            suffixes=("", "_shutuba"),
        )
        if "馬体重_shutuba" in result.columns:
            result["馬体重"] = result["馬体重_shutuba"].fillna(result["馬体重"])
            result = result.drop(columns=["馬体重_shutuba"])
        shutuba_info["body_count"] = int(result["馬体重"].astype(str).str.len().gt(0).sum())
        if "コメント" in result.columns:
            result["コメント"] = result.apply(
                lambda row: append_comment_part(row.get("コメント", ""), body_weight_comment(row.get("_body_weight_change"))),
                axis=1,
            )
    return result, race_info, shutuba_info


def normalize_jockey_for_compare(value):
    text = unicodedata.normalize("NFKC", norm_text(str(value or "")))
    text = re.sub(r"[\(（]\s*替\s*[\)）]", "", text)
    text = re.sub(r"[\(（][^\)）]{1,4}所属[\)）]", "", text)
    text = re.sub(r"^[一-龥ぁ-んァ-ン]{1,4}[・･]", "", text)
    text = re.sub(r"^(?:替|乗替|乗り替わり|初騎乗)", "", text)
    text = text.replace("騎手", "")
    text = re.sub(r"^[▲△☆★◇◆▽▼]+", "", text)
    return re.sub(r"\s+", "", text)


def same_nar_jockey_name(current, previous):
    current_text = normalize_jockey_for_compare(current)
    previous_text = normalize_jockey_for_compare(previous)
    if not current_text or not previous_text:
        return None
    if current_text == previous_text:
        return True

    short, long = (
        (current_text, previous_text)
        if len(current_text) <= len(previous_text)
        else (previous_text, current_text)
    )
    if long.startswith(short):
        diff = len(long) - len(short)
        if len(short) >= 3 and 1 <= diff <= 2:
            return True
        return None
    return False


def nar_jockey_changed_value(current, previous):
    same = same_nar_jockey_name(current, previous)
    if same is None:
        return "pending" if norm_text(str(current or "")) and norm_text(str(previous or "")) else None
    return not same


def parse_nar_newspaper_feature_html(html):
    if not html:
        return {}, pd.DataFrame()
    data = parse_uploaded_nar_newspaper_html(html)
    race = data.get("race") or {}
    soup = BeautifulSoup(html, "html.parser")
    full_text = text_of(soup)
    info = parse_course_text(
        " ".join(
            [
                str(race.get("race_data_1") or ""),
                str(race.get("race_data_2") or ""),
                str(race.get("race_name") or ""),
                full_text[:2000],
            ]
        )
    )
    if not info.get("going"):
        info["going"] = parse_going_from_text(full_text)
    class_fallback_text = " ".join(
        norm_text(str(value or ""))
        for value in (race.get("race_name"), race.get("race_data_1"), race.get("race_data_2"))
        if value
    )
    if hasattr(soup, "select"):
        current_class_rank, current_class_label = race_class_info_from_soup(soup, class_fallback_text)
    else:
        current_class_rank, current_class_label = race_class_info(class_fallback_text)
    current_race_date = parse_race_date_from_race_id(str(data.get("race_id") or html))
    info.update(
        {
            "race_name": race.get("race_name") or text_of(soup.select_one(".RaceName")),
            "race_data": race.get("race_data_1") or text_of(soup.select_one(".RaceData01")),
            "race_data2": race.get("race_data_2") or text_of(soup.select_one(".RaceData02")),
            "class_rank": current_class_rank,
            "class_label": current_class_label,
            "race_date": current_race_date,
        }
    )

    records = []
    for item in data.get("horses", []):
        horse_no = parse_int_from_text(str(item.get("horse_number") or ""))
        if not horse_no:
            continue
        body_value, body_change, body_display = parse_body_weight_text(str(item.get("horse_weight") or ""))
        current_weight = parse_float_from_text(str(item.get("weight") or ""))
        previous_weight = parse_float_from_text(str(item.get("previous_weight") or item.get("前走斤量") or ""))
        load_weight_change = (
            round(current_weight - previous_weight, 3)
            if current_weight is not None and previous_weight is not None
            else None
        )
        current_jockey = norm_text(str(item.get("jockey") or ""))
        previous_jockey = norm_text(str(item.get("previous_jockey") or item.get("前走騎手") or ""))
        newspaper_past_runs = _nar_newspaper_past_runs(item, current_race_date)
        previous_run = newspaper_past_runs[0] if newspaper_past_runs else {}
        previous_class_rank = previous_run.get("class_rank")
        previous_class_label = norm_text(str(previous_run.get("class_label") or ""))
        ranked_runs = [
            run for run in newspaper_past_runs
            if run.get("class_rank") is not None and norm_text(str(run.get("class_label") or ""))
        ]
        best_run = max(ranked_runs, key=lambda run: run.get("class_rank")) if ranked_runs else {}
        days_since_last = days_between(current_race_date, previous_run.get("race_date"))

        record = {
            "馬番": int(horse_no),
            "_newspaper_horse_name": item.get("horse_name", ""),
            "馬体重": body_display,
            "_body_weight": body_value,
            "_body_weight_change": body_change,
            "レース間隔": norm_text(str(item.get("race_interval") or "")),
            "_newspaper_past_runs": newspaper_past_runs,
            "_current_class_rank": current_class_rank,
            "_current_class_label": current_class_label,
            "_previous_class_rank": previous_class_rank,
            "_previous_class_label": previous_class_label,
            "_best_past_class_rank": best_run.get("class_rank"),
            "_best_past_class_label": norm_text(str(best_run.get("class_label") or "")),
            "_past_class_labels": [
                norm_text(str(run.get("class_label") or ""))
                for run in newspaper_past_runs
                if norm_text(str(run.get("class_label") or ""))
            ],
            "_class_shift": class_shift_label(current_class_rank, previous_class_rank),
            "_days_since_last": days_since_last,
        }
        if current_weight is not None:
            record["_display_current_load_weight"] = current_weight
        if previous_weight is not None:
            record["_display_previous_load_weight"] = previous_weight
        if load_weight_change is not None:
            record["_display_load_weight_change"] = load_weight_change
        if current_jockey:
            record["_display_current_jockey"] = current_jockey
        if previous_jockey:
            record["_display_previous_jockey"] = previous_jockey
        if current_jockey and previous_jockey:
            record["_display_jockey_changed"] = nar_jockey_changed_value(current_jockey, previous_jockey)
        records.append(record)

    newspaper_df = pd.DataFrame(records)
    if not newspaper_df.empty:
        newspaper_df = newspaper_df.drop_duplicates("馬番")
    return info, newspaper_df


def _nar_newspaper_past_runs(item, current_race_date):
    source_runs = item.get("past_runs") or item.get("recent_runs") or []
    if not isinstance(source_runs, list):
        return []
    result = []
    seen = set()
    labels = ("前走", "2走前", "3走前")
    for index, raw in enumerate(source_runs[:3]):
        if not isinstance(raw, dict):
            continue
        run = dict(raw)
        identity = (
            norm_text(str(raw.get("previous_date") or raw.get("前走日付") or "")),
            norm_text(str(raw.get("previous_race") or raw.get("前走レース") or "")),
        )
        if identity != ("", "") and identity in seen:
            continue
        seen.add(identity)
        class_rank, class_label = race_class_info(
            " ".join(
                value
                for value in (
                    norm_text(str(raw.get("class_text") or "")),
                    norm_text(str(raw.get("previous_race") or raw.get("前走レース") or "")),
                )
                if value
            )
        )
        race_date = _nar_newspaper_date(raw.get("previous_date") or raw.get("前走日付"), current_race_date)
        position = parse_int_from_text(str(raw.get("previous_finish") or raw.get("前走着順") or ""))
        run.update(
            {
                "label": labels[index],
                "race_date": race_date,
                "class_rank": class_rank,
                "class_label": class_label,
                "position": position,
            }
        )
        result.append(run)
    return result


def _nar_newspaper_date(value, current_race_date):
    text = norm_text(str(value or ""))
    match = re.search(r"(?:(\d{4})[./-])?(\d{1,2})[./-](\d{1,2})", text)
    if not match:
        return None
    year = int(match.group(1)) if match.group(1) else getattr(current_race_date, "year", 0)
    month = int(match.group(2))
    day = int(match.group(3))
    if not year:
        return None
    try:
        parsed = date(year, month, day)
    except ValueError:
        return None
    if current_race_date and parsed > current_race_date and not match.group(1):
        try:
            parsed = date(year - 1, month, day)
        except ValueError:
            return None
    return parsed


def _coalesce_newspaper_column(result, column):
    newspaper_column = f"{column}_newspaper"
    if newspaper_column not in result.columns:
        return result
    newspaper_values = result[newspaper_column]
    has_newspaper_value = newspaper_values.notna()
    if newspaper_values.dtype == object:
        has_newspaper_value &= newspaper_values.astype(str).str.strip().ne("")
    if not has_newspaper_value.any():
        # Assigning an empty object Series to a numeric column is rejected by
        # pandas 3 (and warned by pandas 2) even though the row mask is empty.
        # There is nothing to coalesce, so preserve the original values/dtype.
        return result.drop(columns=[newspaper_column])
    if column in result.columns:
        replacement = newspaper_values.loc[has_newspaper_value]
        target = result[column]
        if pd.api.types.is_numeric_dtype(target.dtype):
            numeric_replacement = pd.to_numeric(replacement, errors="coerce")
            non_numeric = replacement.notna() & numeric_replacement.isna()
            if non_numeric.any():
                # This is a genuinely non-numeric field whose existing column
                # became numeric only because every original value was NaN.
                # Promote this column alone and retain the factual HTML value.
                result[column] = target.astype("object")
            else:
                replacement = numeric_replacement
        result.loc[has_newspaper_value, column] = replacement
    else:
        result[column] = newspaper_values
    return result.drop(columns=[newspaper_column])


def apply_nar_newspaper_html_features(df, race_info, newspaper_html):
    result = df.copy()
    result["馬体重"] = result.get("馬体重", pd.Series("", index=result.index)).fillna("")
    if "同馬場実績" not in result.columns:
        result["同馬場実績"] = result.get("馬場適性", pd.Series("", index=result.index)).fillna("")
    newspaper_info = {"used": False, "going": "", "body_count": 0, "previous_detail_count": 0}
    if not newspaper_html:
        return result, race_info, newspaper_info

    parsed_info, newspaper_df = parse_nar_newspaper_feature_html(newspaper_html)
    newspaper_info["used"] = True
    if parsed_info.get("going"):
        race_info["going"] = parsed_info.get("going")
        newspaper_info["going"] = parsed_info.get("going")
        result = recompute_same_going_features(result, race_info)
    else:
        result = add_condition_context_features(result, race_info)
    if parsed_info.get("class_label") and not race_info.get("class_label"):
        race_info["class_label"] = parsed_info.get("class_label")
        race_info["class_rank"] = parsed_info.get("class_rank")
    if parsed_info.get("race_date") and not race_info.get("race_date"):
        race_info["race_date"] = parsed_info.get("race_date")

    if not newspaper_df.empty:
        result = result.merge(
            newspaper_df.drop(columns=["_newspaper_horse_name"], errors="ignore"),
            on="馬番",
            how="left",
            suffixes=("", "_newspaper"),
        )
        for column in (
            "馬体重",
            "_body_weight",
            "_body_weight_change",
            "レース間隔",
            "_current_class_rank",
            "_current_class_label",
            "_previous_class_rank",
            "_previous_class_label",
            "_best_past_class_rank",
            "_best_past_class_label",
            "_past_class_labels",
            "_class_shift",
            "_days_since_last",
            "_display_current_load_weight",
            "_display_previous_load_weight",
            "_display_load_weight_change",
            "_display_current_jockey",
            "_display_previous_jockey",
            "_display_jockey_changed",
        ):
            result = _coalesce_newspaper_column(result, column)

        if "_newspaper_past_runs" in result.columns:
            if "_past_runs" not in result.columns:
                result["_past_runs"] = pd.Series([[] for _ in range(len(result))], index=result.index, dtype="object")
            result["_past_runs"] = result.apply(
                lambda row: _merge_nar_past_run_evidence(
                    row.get("_past_runs"),
                    row.get("_newspaper_past_runs"),
                ),
                axis=1,
            )
            result = result.drop(columns=["_newspaper_past_runs"])

        newspaper_info["body_count"] = int(result["馬体重"].astype(str).str.len().gt(0).sum())
        previous_cols = [
            col for col in ("_display_previous_load_weight", "_display_previous_jockey") if col in result.columns
        ]
        if previous_cols:
            newspaper_info["previous_detail_count"] = int(
                result[previous_cols].notna().any(axis=1).sum()
            )
        if "コメント" in result.columns:
            result["コメント"] = result.apply(
                lambda row: append_comment_part(row.get("コメント", ""), body_weight_comment(row.get("_body_weight_change"))),
                axis=1,
            )
    return result, race_info, newspaper_info


def _merge_nar_past_run_evidence(existing, newspaper):
    """Supplement speed-index runs with saved-newspaper facts by run role."""

    base = [dict(run) for run in existing if isinstance(run, dict)] if isinstance(existing, list) else []
    additions = [dict(run) for run in newspaper if isinstance(run, dict)] if isinstance(newspaper, list) else []
    by_role = {_nar_past_run_role(run): run for run in base if _nar_past_run_role(run)}
    for run in additions:
        role = _nar_past_run_role(run)
        target = by_role.get(role)
        if target is None:
            base.append(run)
            if role:
                by_role[role] = run
            continue
        for key, value in run.items():
            if value not in (None, "", []) and target.get(key) in (None, "", []):
                target[key] = value
    order = {"3走前": 0, "2走前": 1, "前走": 2}
    return sorted(base, key=lambda run: order.get(_nar_past_run_role(run), 9))


def _nar_past_run_role(run):
    label = norm_text(str((run or {}).get("label") or (run or {}).get("key") or (run or {}).get("run_key") or ""))
    aliases = {
        "前走": "前走", "last": "前走", "race1": "前走", "1走前": "前走",
        "2走前": "2走前", "2back": "2走前", "race2": "2走前",
        "3走前": "3走前", "3back": "3走前", "race3": "3走前",
    }
    return aliases.get(label, label if label in {"前走", "2走前", "3走前"} else "")


def parse_nar_speed_table(html, session, fetch_past_detail=True, sleep_sec=0.35):
    current = parse_current_race_info(html)
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("#Speed_List table") or soup.select_one("table.SpeedIndex_Table")
    if table is None:
        raise ValueError("地方競馬のタイム指数テーブルが見つかりません。nar.netkeiba.comのタイム指数HTMLか確認してください。")

    rows = table.select("tbody tr.List") or table.select("tbody tr")

    @lru_cache(maxsize=256)
    def fetch_past_info(url, horse_name="", horse_id=""):
        if not fetch_past_detail or not url:
            return {}
        try:
            time.sleep(float(sleep_sec or 0))
            past_html = fetch_html(url, session=session)
            past_soup = BeautifulSoup(past_html, "html.parser")
            candidates = [
                past_soup.select_one(".data_intro .racedata span"),
                past_soup.select_one(".RaceData01"),
            ]
            detail_text = next((text_of(x) for x in candidates if text_of(x)), "")
            info = parse_course_text(detail_text) if detail_text else {}
            info["race_id"] = parse_race_id_from_text(url)
            info["race_name"] = text_of(past_soup.select_one(".data_intro h1")) or text_of(past_soup.select_one(".RaceName"))
            past_title = text_of(past_soup.title)
            past_meta = text_of(past_soup.select_one(".data_intro .smalltxt")) or text_of(past_soup.select_one(".RaceData02"))
            class_rank, class_label = race_class_info_from_soup(past_soup, info.get("race_name", ""), detail_text, past_meta, past_title)
            info["class_rank"] = class_rank
            info["class_label"] = class_label
            info["racecourse"] = parse_racecourse_from_text(url, detail_text, info["race_name"], past_meta, past_title)
            info["race_date"] = parse_race_date_from_race_id(url) or parse_race_date_from_text(detail_text)
            info["jockey"] = extract_jockey_from_past_race(past_soup, horse_name, horse_id)
            info["results"] = extract_past_race_results(past_soup)
            return info
        except Exception as exc:
            return {"label": "", "error": str(exc)}

    records = []
    for row in rows:
        direct_cells = row.find_all("td", recursive=False)

        def td_text(index):
            return visible_text(direct_cells[index]) if len(direct_cells) > index else ""

        waku = td_text(0)
        umaban = visible_text(first(row, [".Speed_List01", ".sk__umaban", ".UmaBan"]))
        horse_cell = first(row, [".Horse_Name", ".sk__horse_name"])
        horse_link = horse_cell.find("a") if horse_cell else None
        horse_name = text_of(horse_link) if horse_link else visible_text(horse_cell)
        horse_id = extract_horse_id_from_link(horse_link)
        if not umaban or not horse_name:
            continue

        sex_age = td_text(4)
        load_weight = visible_text(first(row, [".Speed_List02", ".sk__load_weight"])) or td_text(5)
        jockey = visible_text(first(row, [".Jockey"]))
        odds = parse_float_from_text(visible_text(first(row, [".Speed_List07", ".sk__odds", ".Odds"])))
        popularity = parse_int_from_text(visible_text(first(row, [".Speed_List08", ".sk__ninki", ".Ninki"])))
        display_current_load_weight = parse_float_from_text(display_attr(row, "current-load-weight"))
        display_previous_load_weight = parse_float_from_text(display_attr(row, "previous-load-weight"))
        display_load_weight_change = parse_float_from_text(display_attr(row, "load-weight-change"))
        display_current_jockey = display_attr(row, "current-jockey")
        display_previous_jockey = display_attr(row, "previous-jockey")
        display_jockey_changed_text = display_attr(row, "jockey-changed")
        display_jockey_changed = parse_jockey_changed_from_text(display_jockey_changed_text) if display_jockey_changed_text else None
        max_index = parse_index_cell(first(row, [".Speed_List03", ".sk__max_index", ".MaxIndex"]))["value"]
        avg5_index = parse_index_cell(first(row, [".Speed_List04", ".sk__avg5_index", ".Avg5Index"]))["value"]
        distance_index = parse_index_cell(first(row, [".Speed_List05", ".sk__max_distance_index"]))["value"]
        course_index = parse_index_cell(first(row, [".Speed_List06", ".sk__max_course_index"]))["value"]

        prev_defs = [
            ("3走前", [".Speed_List09", ".sk__index3"]),
            ("2走前", [".Speed_List10", ".sk__index2"]),
            ("前走", [".Speed_List11", ".sk__index1"]),
        ]
        prev_values = []
        prev_display = {}
        star_count = 0
        same_distance_values = []
        similar_condition_values = []
        star_values = []
        same_condition_flags = []
        same_going_values = []
        previous_jockey = ""
        previous_load_weight = None
        days_since_last = None
        previous_class_rank = None
        previous_class_label = ""
        past_class_ranks = []
        past_class_labels = []
        past_runs = []
        star_candidate_runs = []

        for label, selectors in prev_defs:
            cell_info = parse_index_cell(first(row, selectors))
            past_info = fetch_past_info(cell_info["url"], horse_name, horse_id)
            past_condition = dict(past_info)
            for key in ("racecourse", "surface", "distance", "direction", "label"):
                if not past_condition.get(key) and cell_info.get(key):
                    past_condition[key] = cell_info.get(key)
            if not past_condition.get("label"):
                past_condition["label"] = cell_info.get("label") or ""
            if past_info.get("class_rank") is not None:
                past_class_ranks.append(past_info.get("class_rank"))
                past_class_labels.append(past_info.get("class_label", ""))
            same_cond = is_same_condition(current, past_condition)
            same_dist = is_same_distance(current, past_condition)
            similar_cond = is_similar_condition(current, past_condition)
            same_condition_flags.append(bool(same_cond))
            if same_cond:
                star_count += 1
                if cell_info["value"] is not None:
                    star_values.append(cell_info["value"])
            if same_dist and cell_info["value"] is not None:
                same_distance_values.append(cell_info["value"])
            if similar_cond and cell_info["value"] is not None:
                similar_condition_values.append(cell_info["value"])
            if current.get("going") and past_info.get("going") == current.get("going") and cell_info["value"] is not None:
                same_going_values.append(cell_info["value"])
            prev_values.append(cell_info["value"])
            prev_display[label] = format_prev_run(cell_info, past_condition, same_cond)
            result_entry = (past_info.get("results") or {}).get(compact_name(horse_name), {})
            if label == "前走":
                previous_jockey = past_info.get("jockey", "")
                previous_load_weight = result_entry.get("load_weight")
                past_race_date = past_info.get("race_date") or cell_info.get("race_date")
                days_since_last = days_between(current.get("race_date"), past_race_date)
                previous_class_rank = past_info.get("class_rank")
                previous_class_label = past_info.get("class_label", "")
            past_runs.append({
                "label": label,
                "url": cell_info.get("url", ""),
                "race_id": past_info.get("race_id") or parse_race_id_from_text(cell_info.get("url", "")),
                "race_name": past_info.get("race_name", ""),
                "race_date": past_info.get("race_date") or cell_info.get("race_date"),
                "racecourse": past_info.get("racecourse", ""),
                "course_label": past_info.get("label", ""),
                "surface": past_info.get("surface", ""),
                "distance": past_info.get("distance"),
                "direction": past_info.get("direction", ""),
                "going": past_info.get("going", ""),
                "class_rank": past_info.get("class_rank"),
                "class_label": past_info.get("class_label", ""),
                "error": past_info.get("error", ""),
                "value": cell_info.get("value"),
                "position": result_entry.get("position"),
            })
            star_candidate_runs.append({
                "label": label,
                "value": cell_info.get("value"),
                "racecourse": past_condition.get("racecourse", ""),
                "surface": past_condition.get("surface", ""),
                "distance": past_condition.get("distance"),
                "direction": past_condition.get("direction", ""),
            })

        valid_prev = [v for v in prev_values if v is not None]
        avg3 = round(sum(valid_prev) / len(valid_prev), 1) if valid_prev else None
        trend = None
        if prev_values[0] is not None and prev_values[-1] is not None:
            trend = prev_values[-1] - prev_values[0]
        jockey_change_value = nar_jockey_changed_value(jockey, previous_jockey)
        jockey_changed = bool(jockey_change_value is True)
        jockey_display = f"{jockey}(替)" if jockey_changed else jockey
        load_weight_display = format_load_weight_with_change(load_weight, previous_load_weight)
        current_load_weight = parse_float_from_text(load_weight)
        load_weight_change = current_load_weight - previous_load_weight if current_load_weight is not None and previous_load_weight is not None else None
        log_star_trace(
            "05 before star_index.py call",
            [
                star_trace_row(
                    horse_no=umaban,
                    horse_name=horse_name,
                    year_max_index=max_index,
                    star_max_index=None,
                    star_candidates=candidate_summary(star_candidate_runs),
                )
            ],
        )
        star_result = build_star_max_result(current, star_candidate_runs)
        star_high_value = star_result.value
        log_star_trace(
            "06 after star_index.py call",
            [
                star_trace_row(
                    horse_no=umaban,
                    horse_name=horse_name,
                    year_max_index=max_index,
                    star_max_index=star_high_value,
                    star_source=star_result.source,
                    star_race=star_result.race,
                    star_condition=star_result.condition,
                    star_match_level=star_result.match_level,
                )
            ],
        )
        star_count_value = sum(1 for item in star_candidate_runs if star_match_level(current, item) != "none" and item.get("value") is not None)

        records.append({
            "馬番": int(parse_int_from_text(umaban) or 0),
            "枠": int(parse_int_from_text(waku) or 0),
            "馬名": horse_name,
            "性齢": sex_age,
            "斤量": load_weight_display,
            "騎手": jockey_display,
            "単勝オッズ": odds,
            "人気": popularity,
            "間隔": format_interval_from_days(days_since_last),
            "最高指数": max_index,
            "year_max_index": max_index,
            "過去1年最高指数": max_index,
            "平均指数": avg5_index,
            "距離指数": distance_index,
            "コース指数": course_index,
            "3走前": prev_display["3走前"],
            "2走前": prev_display["2走前"],
            "前走": prev_display["前走"],
            "3走平均": avg3,
            "_prev_values": prev_values,
            "_last": prev_values[-1] if prev_values else None,
            "_trend": trend,
            "_star_count": star_count_value,
            "_star_high": star_high_value,
            "_star_max_race": star_result.race,
            "_star_max_venue": star_result.venue,
            "_star_max_distance": star_result.distance,
            "_star_max_surface": star_result.surface,
            "_star_max_turn": star_result.turn,
            "_star_match_level": star_result.match_level,
            "_star_max_condition": star_result.condition,
            "_star_high_source": star_result.source,
            "_same_condition_flags": same_condition_flags,
            "_last_same_condition": bool(same_condition_flags[-1]) if same_condition_flags else False,
            "_same_distance_high": max(same_distance_values) if same_distance_values else None,
            "_similar_condition_count": len(similar_condition_values),
            "_similar_condition_high": max(similar_condition_values) if similar_condition_values else None,
            "_current_going": current.get("going", ""),
            "_same_going_count": len(same_going_values),
            "_same_going_high": max(same_going_values) if same_going_values else None,
            "_current_class_rank": current.get("class_rank"),
            "_current_class_label": current.get("class_label", ""),
            "_previous_class_rank": previous_class_rank,
            "_previous_class_label": previous_class_label,
            "_best_past_class_rank": max(past_class_ranks) if past_class_ranks else previous_class_rank,
            "_best_past_class_label": past_class_labels[past_class_ranks.index(max(past_class_ranks))] if past_class_ranks else previous_class_label,
            "_past_class_labels": past_class_labels,
            "_class_shift": class_shift_label(current.get("class_rank"), previous_class_rank),
            "_current_jockey": jockey,
            "_previous_jockey": previous_jockey,
            "_jockey_changed": jockey_changed,
            "_jockey_change_pending": jockey_change_value == "pending",
            "_previous_load_weight": previous_load_weight,
            "_current_load_weight": current_load_weight,
            "_load_weight_change": load_weight_change,
            "_display_previous_load_weight": display_previous_load_weight,
            "_display_current_load_weight": display_current_load_weight,
            "_display_load_weight_change": display_load_weight_change,
            "_display_previous_jockey": display_previous_jockey,
            "_display_current_jockey": display_current_jockey,
            "_display_jockey_changed": display_jockey_changed,
            "_load_weight_adjustment": load_weight_index_adjustment(load_weight_change),
            "_race_distance": current.get("distance"),
            "_days_since_last": days_since_last,
            "_is_layoff": bool(days_since_last is not None and days_since_last >= 60),
            "_past_runs": past_runs,
        })

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("馬データを抽出できませんでした。保存HTMLにタイム指数表が含まれているか確認してください。")

    log_star_trace(
        "04 parse_nar_speed_table DataFrame",
        [
            star_trace_row(
                horse_no=row.iloc[0] if len(row) > 0 else "",
                horse_name=row.iloc[2] if len(row) > 2 else "",
                year_max_index=row.get("year_max_index"),
                star_max_index=row.get("_star_high"),
                star_source=row.get("_star_high_source"),
                star_race=row.get("_star_max_race"),
                star_condition=row.get("_star_max_condition"),
            )
            for _, row in df.iterrows()
        ],
    )

    df = add_head_to_head_features(df)
    df = add_condition_context_features(df, current)
    df = add_scores_and_comments(df)
    return df, current


def add_condition_context_features(df, current):
    df = df.copy()
    going = current.get("going") or ""
    same_going_high = pd.to_numeric(df.get("_same_going_high"), errors="coerce")
    top_same_going = same_going_high.max()

    def going_label(row):
        if not going:
            return ""
        high = row.get("_same_going_high")
        count = row.get("_same_going_count", 0) or 0
        if high is None or pd.isna(high):
            return "同馬場未知" if going in ["稍重", "重", "不良"] else ""
        prefix = "道悪" if going in ["稍重", "重", "不良"] else "良馬場"
        if pd.notna(top_same_going) and high == top_same_going:
            return f"{prefix}上位{compact_number(high)}"
        if count >= 2:
            return f"{prefix}実績{compact_number(high)}"
        return f"{prefix}経験{compact_number(high)}"

    df["同馬場実績"] = df.apply(going_label, axis=1)
    df["馬場適性"] = df["同馬場実績"]
    df["クラス変動"] = df.get("_class_shift", pd.Series("", index=df.index)).fillna("")
    return df


def safe_num(value, fallback):
    if value is None:
        return fallback
    if isinstance(value, float) and math.isnan(value):
        return fallback
    return value


def add_head_to_head_features(df):
    df = df.copy()
    if "_past_runs" not in df.columns:
        df["_h2h_wins"] = 0
        df["_h2h_losses"] = 0
        df["_h2h_score"] = 0
        df["_h2h_label"] = ""
        df["_h2h_latest"] = ""
        df["対戦"] = ""
        return df

    race_map = {}
    for idx, row in df.iterrows():
        for run in row.get("_past_runs") or []:
            position = run.get("position")
            race_key = run.get("race_id") or run.get("url")
            if not race_key or position is None:
                continue
            race_map.setdefault(race_key, {
                "race_date": run.get("race_date"),
                "race_name": run.get("race_name", ""),
                "course_label": run.get("course_label", ""),
                "entries": [],
            })
            race_map[race_key]["entries"].append({
                "idx": idx,
                "horse_no": row.get("馬番"),
                "horse_name": row.get("馬名"),
                "position": int(position),
                "label": run.get("label", ""),
                "value": run.get("value"),
            })

    wins = {idx: 0 for idx in df.index}
    losses = {idx: 0 for idx in df.index}
    latest = {idx: None for idx in df.index}
    details = {idx: [] for idx in df.index}

    def date_key(value):
        return value.toordinal() if hasattr(value, "toordinal") else -1

    for race in race_map.values():
        entries = sorted(race["entries"], key=lambda item: item["position"])
        if len(entries) < 2:
            continue
        race_date = race.get("race_date")
        for i, winner in enumerate(entries):
            for loser in entries[i + 1:]:
                win_idx = winner["idx"]
                lose_idx = loser["idx"]
                wins[win_idx] += 1
                losses[lose_idx] += 1

                loser_no = circled_number_text(loser["horse_no"])
                winner_no = circled_number_text(winner["horse_no"])
                win_phrase = f"{loser_no}に先着"
                lose_phrase = f"{winner_no}に敗戦"
                details[win_idx].append(win_phrase)
                details[lose_idx].append(lose_phrase)

                win_latest = f"直近{loser_no}に先着"
                lose_latest = f"直近{winner_no}に敗戦"
                if latest[win_idx] is None or date_key(race_date) >= latest[win_idx][0]:
                    latest[win_idx] = (date_key(race_date), win_latest)
                if latest[lose_idx] is None or date_key(race_date) >= latest[lose_idx][0]:
                    latest[lose_idx] = (date_key(race_date), lose_latest)

    h2h_labels = []
    h2h_latest = []
    display_values = []
    for idx in df.index:
        win_count = wins[idx]
        loss_count = losses[idx]
        if win_count == 0 and loss_count == 0:
            label = ""
        elif win_count >= 2 and loss_count == 0:
            label = "対戦◎"
        elif win_count > loss_count:
            label = "対戦○"
        elif loss_count > win_count:
            label = "対戦△"
        else:
            label = "対戦五分"

        latest_phrase = latest[idx][1] if latest[idx] else ""
        display = latest_phrase
        h2h_labels.append(label)
        h2h_latest.append(latest_phrase)
        display_values.append(display)

    df["_h2h_wins"] = pd.Series(wins)
    df["_h2h_losses"] = pd.Series(losses)
    df["_h2h_score"] = df["_h2h_wins"] - df["_h2h_losses"]
    df["_h2h_label"] = h2h_labels
    df["_h2h_latest"] = h2h_latest
    df["対戦"] = display_values
    return df


def _nar_has_any_prev_index(values):
    if not isinstance(values, list):
        return False
    return any(_nar_has_index_value(value) for value in values)


def _nar_has_index_value(value):
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    text = str(value).strip().replace("*", "")
    if text in {"", "-", "未", "未取得", "None", "none", "nan", "<NA>"}:
        return False
    return pd.to_numeric(pd.Series([text]), errors="coerce").notna().iloc[0]


def _nar_is_missing_scalar(value):
    if value is None:
        return True
    try:
        missing = pd.isna(value)
        try:
            if bool(missing):
                return True
        except (TypeError, ValueError):
            pass
    except (TypeError, ValueError):
        pass
    return str(value).strip() in {"", "None", "none", "nan", "NaN", "<NA>", "NaT"}


def _nar_safe_text(value):
    return "" if _nar_is_missing_scalar(value) else str(value).strip()


def _nar_safe_bool(value, default=False):
    if _nar_is_missing_scalar(value):
        return default
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "t"}
    return bool(value)


def _nar_safe_int(value, default=0):
    if _nar_is_missing_scalar(value):
        return default
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _nar_local_index_data_shortage_mask(df):
    """True when a NAR horse has no local index material at all.

    Central transfers and other first-start-in-NAR horses can have odds/style
    information while every local speed-index source is missing.  Those horses
    must stay visible, but they should not be normalized into an AI score.
    """

    if df is None or len(df) == 0:
        return pd.Series([], dtype=bool)

    valid = pd.Series(False, index=df.index)
    for column in [
        "最高指数",
        "平均指数",
        "距離指数",
        "コース指数",
        "近3走最高",
        "3走平均",
        "★最高",
        "★最高指数",
        "_star_high",
        "_same_distance_high",
        "_similar_condition_high",
        "max",
        "avg5",
        "distance",
        "course",
        "race3",
        "race2",
        "race1",
    ]:
        if column in df.columns:
            valid = valid | df[column].map(_nar_has_index_value).fillna(False).astype(bool)

    if "_prev_values" in df.columns:
        valid = valid | df["_prev_values"].map(_nar_has_any_prev_index).fillna(False).astype(bool)

    return ~valid


def add_scores_and_comments(df):
    df = df.copy()
    for col in ["距離指数", "コース指数", "3走平均"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    field_avg3 = df["3走平均"].mean()
    field_dist = df["距離指数"].mean()
    field_course = df["コース指数"].mean()
    star_high_series = pd.to_numeric(df.get("_star_high"), errors="coerce")
    field_star_high = star_high_series.mean()
    top_star_high = star_high_series.max()
    current_load_weights = pd.to_numeric(df.get("_current_load_weight"), errors="coerce")
    field_average_load_weight = current_load_weights.mean()
    df["_relative_load_weight"] = (field_average_load_weight - current_load_weights).round(1)
    df["_relative_load_weight_adjustment"] = current_load_weights.map(
        lambda value: relative_load_weight_adjustment(value, field_average_load_weight)
    )
    df["_total_load_weight_adjustment"] = (
        pd.to_numeric(df.get("_load_weight_adjustment"), errors="coerce").fillna(0)
        + pd.to_numeric(df.get("_relative_load_weight_adjustment"), errors="coerce").fillna(0)
    ).clip(-MAX_TOTAL_WEIGHT_ADJUSTMENT, MAX_TOTAL_WEIGHT_ADJUSTMENT).round(1)
    df["近3走最高"] = df["_prev_values"].apply(
        lambda values: max([v for v in values if v is not None]) if isinstance(values, list) and any(v is not None for v in values) else None
    )
    df["_地方指数データ不足"] = _nar_local_index_data_shortage_mask(df)

    raw_scores = []
    ver3_ability_cores = []
    market_non_ability_adjustments = []
    for idx, row in df.iterrows():
        if bool(df.at[idx, "_地方指数データ不足"]):
            raw_scores.append(pd.NA)
            ver3_ability_cores.append(pd.NA)
            market_non_ability_adjustments.append(pd.NA)
            continue
        avg3 = safe_num(row["3走平均"], field_avg3)
        dist = safe_num(row["距離指数"], avg3)
        course = safe_num(row["コース指数"], avg3)
        latest = safe_num(row["_last"], avg3)
        star_high = safe_num(row.get("_star_high"), None)
        similar_condition_high = safe_num(row.get("_similar_condition_high"), None)
        best_recent = safe_num(row["近3走最高"], avg3)
        weight_adjustment = safe_num(row.get("_total_load_weight_adjustment"), 0)

        star_component = star_high if star_high is not None else field_avg3
        condition_bonus = 0.0
        if star_high is None and similar_condition_high is not None:
            if similar_condition_high >= avg3 + 6:
                condition_bonus = 1.2
            elif similar_condition_high >= avg3:
                condition_bonus = 0.7
            else:
                condition_bonus = 0.3
        ability_core = calculate_ver3_ability_core(
            recent_average=avg3,
            star_index=star_component,
            recent_best=best_recent,
            latest_index=latest,
            distance_index=dist,
            course_index=course,
        )
        raw = ability_core + weight_adjustment + condition_bonus
        raw_scores.append(raw)
        ver3_ability_cores.append(ability_core)
        # Legacy Ver3 compatibility keeps these terms in _raw_score. Market
        # mode reads _ver3_ability_core directly; this adjustment column exists
        # only to audit legacy values and old saved snapshots.
        market_non_ability_adjustments.append(weight_adjustment + condition_bonus)

    df["_raw_score"] = raw_scores
    df["_ver3_ability_core"] = ver3_ability_cores
    df["_market_non_ability_adjustment"] = market_non_ability_adjustments
    raw_numeric = pd.to_numeric(df["_raw_score"], errors="coerce")
    valid_score_mask = (~df["_地方指数データ不足"]) & raw_numeric.notna()
    min_raw = raw_numeric.loc[valid_score_mask].min()
    max_raw = raw_numeric.loc[valid_score_mask].max()
    df["AI点"] = pd.NA
    if not bool(valid_score_mask.any()):
        df["AI点"] = pd.NA
    elif max_raw == min_raw:
        df.loc[valid_score_mask, "AI点"] = 80.0
    else:
        df.loc[valid_score_mask, "AI点"] = (
            60 + 40 * (raw_numeric.loc[valid_score_mask] - min_raw) / (max_raw - min_raw)
        ).round(1)
    df["★最高"] = df["_star_high"]
    if "_star_high_source" in df.columns:
        star_source = df["_star_high_source"].fillna("missing").astype(str).replace("", "missing")
    else:
        star_source = pd.Series("missing", index=df.index)
    df["star_max_index"] = df["_star_high"]
    df["star_max_race"] = df.get("_star_max_race", pd.Series("", index=df.index))
    df["star_max_venue"] = df.get("_star_max_venue", pd.Series("", index=df.index))
    df["star_max_distance"] = df.get("_star_max_distance", pd.Series(pd.NA, index=df.index))
    df["star_max_surface"] = df.get("_star_max_surface", pd.Series("", index=df.index))
    df["star_max_turn"] = df.get("_star_max_turn", pd.Series("", index=df.index))
    df["star_match_level"] = df.get("_star_match_level", pd.Series("none", index=df.index))
    df["star_max_source"] = star_source
    df["★最高指数の取得元"] = star_source
    df["star_max_condition"] = df.get("_star_max_condition", pd.Series("", index=df.index))
    log_star_trace(
        "07 add_scores_and_comments",
        [
            star_trace_row(
                horse_no=row.iloc[0] if len(row) > 0 else "",
                horse_name=row.iloc[2] if len(row) > 2 else "",
                year_max_index=row.get("year_max_index"),
                star_max_index=row.get("star_max_index"),
                star_source=row.get("star_max_source"),
                raw_score=row.get("_raw_score"),
                normalized_ai_score=row.get("AI轤ｹ"),
            )
            for _, row in df.iterrows()
        ],
    )
    df["★該当走"] = df["star_max_race"]
    df["★条件"] = df["star_max_condition"]

    df = df.sort_values(["AI点", "3走平均", "距離指数"], ascending=[False, False, False]).reset_index(drop=True)
    ai_rank = pd.to_numeric(df["AI点"], errors="coerce").rank(method="min", ascending=False)
    df.insert(0, "AI順位", ai_rank.astype("Int64"))
    top_avg3 = df["3走平均"].max()
    top_dist = df["距離指数"].max()
    top_course = df["コース指数"].max()
    top_ai_for_comment = df["AI点"].max()
    top_recent_for_comment = df["近3走最高"].max()
    field_recent_high = pd.to_numeric(df["近3走最高"], errors="coerce").mean()

    def comment(row):
        if _nar_safe_bool(row.get("_地方指数データ不足", False)):
            return "データ不足"
        parts = []
        star_high = safe_num(row.get("_star_high"), None)
        star_count_value = safe_num(row.get("_star_count"), 0)
        popularity_value = safe_num(row.get("人気"), 99)
        if star_count_value >= 2:
            if star_high is not None and pd.notna(top_star_high) and star_high == top_star_high and row["AI点"] >= top_ai_for_comment - 12:
                parts.append("同条件◎")
            else:
                parts.append("同条件続く")
            if popularity_value >= 5:
                parts.append("紐穴")
        elif star_high is not None and pd.notna(top_star_high) and star_high == top_star_high:
            if row["AI点"] >= top_ai_for_comment - 10:
                parts.append("同条件◎")
            else:
                parts.append("条件揃うも相手強化")
        elif star_high is not None and pd.notna(row["3走平均"]) and star_high >= row["3走平均"] + 6:
            parts.append("条件替わり妙味")
        elif star_high is not None:
            parts.append("条件揃う")
        elif row.get("AI順位", 99) <= 6 and pd.notna(row.get("近3走最高")) and row["近3走最高"] >= top_recent_for_comment - 8:
            parts.append("近況上位")

        similar_condition_high = safe_num(row.get("_similar_condition_high"), None)
        if star_high is None and similar_condition_high is not None:
            if pd.notna(row.get("3走平均")) and similar_condition_high >= row["3走平均"]:
                parts.append("他場同条件○")
            else:
                parts.append("他場同条件")

        going_note = _nar_safe_text(row.get("同馬場実績")) or _nar_safe_text(row.get("馬場適性"))
        if going_note and going_note != "同馬場未知":
            parts.append(going_note)

        class_note = _nar_safe_text(row.get("クラス変動"))
        if class_note in ("相手弱化", "クラス降級"):
            parts.append("クラス降級")
        elif class_note in ("相手強化", "クラス昇級"):
            parts.append("クラス昇級注意")

        age = extract_age_from_sex_age(row.get("性齢"))
        relative_load_weight = safe_num(row.get("_relative_load_weight"), None)
        young_lightweight = bool(
            age == 3
            and relative_load_weight is not None
            and relative_load_weight >= 2
        )
        if young_lightweight:
            parts.append("軽量3歳注意")
        elif relative_load_weight is not None and relative_load_weight >= 2.5:
            parts.append("軽斤量")

        h2h_latest = _nar_safe_text(row.get("_h2h_latest"))
        h2h_label = _nar_safe_text(row.get("_h2h_label"))
        if h2h_latest and "敗戦" in h2h_latest:
            parts.append(h2h_latest)
        elif h2h_label in ("対戦◎", "対戦○"):
            parts.append(h2h_label)
        elif h2h_label == "対戦△":
            parts.append("対戦劣勢")

        days_since = row.get("_days_since_last")
        if pd.notna(days_since) and days_since >= 60:
            parts.append("休み明け注意")
        elif pd.notna(days_since) and days_since >= 45:
            parts.append("間隔空き")

        if pd.notna(row["3走平均"]) and row["3走平均"] == top_avg3:
            parts.append("近走最上位")
        elif pd.notna(row["3走平均"]) and row["3走平均"] >= field_avg3 + 5:
            parts.append("能力上位")

        if pd.notna(row["距離指数"]) and row["距離指数"] == top_dist:
            parts.append("距離巧者")
        elif pd.notna(row["コース指数"]) and row["コース指数"] == top_course:
            parts.append("コース巧者")

        weight_change = safe_num(row.get("_load_weight_change"), None)
        if weight_change is not None:
            if weight_change <= -1 and not young_lightweight:
                parts.append("斤量減")
            elif weight_change >= 1:
                parts.append("斤量増注意")

        trend = safe_num(row["_trend"], 0)
        popularity = safe_num(row.get("人気"), None)
        if age == 3:
            if popularity is not None and popularity <= 5 and row["AI点"] < top_ai_for_comment - 10:
                parts.append("3歳指数以上")
            elif popularity is not None and popularity <= 5:
                parts.append("3歳上積み")
            elif trend >= 4 or star_high is not None:
                parts.append("3歳穴")
        elif age == 4:
            if popularity is not None and popularity <= 5 and row["AI点"] < top_ai_for_comment - 10:
                parts.append("指数以上の支持")
            elif popularity is not None and popularity <= 5:
                parts.append("4歳上積み")
            elif trend >= 4 or star_high is not None:
                parts.append("4歳穴")

        if trend >= 8:
            parts.append("上昇中")
        elif trend >= 4:
            parts.append("良化気配")
        if age is not None and age >= 7:
            recent_high = safe_num(row.get("近3走最高"), None)
            if recent_high is not None and pd.notna(field_recent_high) and recent_high <= field_recent_high - 6:
                parts.append("近走一息")
        if row.get("_jockey_changed"):
            parts.append("乗替注")
        if not parts:
            parts.append("一変待ち")

        unique = []
        for item in parts:
            if item not in unique:
                unique.append(item)
        return "、".join(unique[:3])

    df["コメント"] = df.apply(comment, axis=1)

    top_ai = pd.to_numeric(df["AI点"], errors="coerce").max()

    def merit_score(row):
        if _nar_safe_bool(row.get("_地方指数データ不足", False)) or pd.isna(row.get("AI順位")):
            return 0
        return int(row["人気"]) - int(row["AI順位"]) if pd.notna(row.get("人気")) else 0

    df["妙味スコア"] = df.apply(merit_score, axis=1)

    def recommendation_bonus(row):
        if _nar_safe_bool(row.get("_地方指数データ不足", False)) or pd.isna(row.get("AI点")) or pd.isna(top_ai):
            return 0.0
        value = safe_num(row["妙味スコア"], 0)
        bonus = max(min(value, 8), -4) * 0.35
        if pd.notna(row.get("人気")) and int(row["人気"]) == 1:
            bonus += 1.0
        ai_value = safe_num(row.get("AI点"), None)
        if ai_value is not None and ai_value >= top_ai - 8 and value >= 4:
            bonus += 1.2
        h2h_score = safe_num(row.get("_h2h_score"), 0)
        bonus += max(min(h2h_score, 2), -2) * 0.6
        latest_h2h = _nar_safe_text(row.get("_h2h_latest"))
        if "敗戦" in latest_h2h:
            bonus -= 0.5
        elif "先着" in latest_h2h:
            bonus += 0.3
        return bonus

    df["推奨点"] = (pd.to_numeric(df["AI点"], errors="coerce") + df.apply(recommendation_bonus, axis=1)).round(1)

    def age_recommendation_bonus(row):
        age = extract_age_from_sex_age(row.get("性齢"))
        trend = safe_num(row.get("_trend"), 0)
        relative_load_weight = safe_num(row.get("_relative_load_weight"), 0)
        bonus = 0.0
        if age == 3:
            bonus += 3.0
            if trend >= 4:
                bonus += 1.5
            if relative_load_weight >= 2:
                bonus += 1.5
        elif age == 4:
            bonus += 2.2
            if trend >= 4:
                bonus += 1.2
            if relative_load_weight >= 2:
                bonus += 0.5
        elif age is not None and age >= 11:
            bonus -= 3.0 if trend >= 4 else 4.0
        elif age is not None and age >= 9:
            bonus -= 0.5 if trend >= 4 else 1.5
        return bonus

    df["_年齢"] = df["性齢"].map(extract_age_from_sex_age)
    df["_age_bonus"] = df.apply(age_recommendation_bonus, axis=1).round(1)
    df["推奨点"] = (df["推奨点"] + df["_age_bonus"]).round(1)

    df["役割"] = "消し寄り"
    selected = set()
    popularity = pd.to_numeric(df["人気"], errors="coerce")
    layoff_mask = df.get("_is_layoff", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    star_high_numeric = pd.to_numeric(df["★最高"], errors="coerce")

    def assign_role(mask, role_name, limit):
        candidates = df[mask & ~df.index.isin(selected)].sort_values(
            ["推奨点", "AI点", "3走平均"],
            ascending=[False, False, False],
        ).head(limit)
        for idx in candidates.index:
            df.at[idx, "役割"] = role_name
            selected.add(idx)

    condition_main_mask = (
        star_high_numeric.notna()
        & (star_high_numeric >= field_star_high + 4)
        & (df["AI点"] >= top_ai - 12)
        & (popularity.fillna(99) <= 4)
        & ~layoff_mask
    )
    stable_mask = (
        condition_main_mask
        | (
        (df["AI順位"] <= 3)
        & (df["AI点"] >= top_ai - 5)
        & (df["妙味スコア"] <= 2)
        & (popularity <= 4)
        & df["_last"].notna()
        & (df["_last"] >= df["3走平均"] - 5)
        & ~layoff_mask
        )
    )
    if not stable_mask.any():
        stable_mask = (
            (df["AI順位"] == 1)
            & (df["AI点"] >= top_ai - 3)
            & (popularity <= 3)
            & ~layoff_mask
        )
    assign_role(stable_mask, "本軸", 1)

    assign_role(
        (df["AI点"] >= top_ai - 10)
        & (df["妙味スコア"] >= 4)
        & (pd.to_numeric(df["人気"], errors="coerce") >= 5),
        "穴軸",
        1,
    )
    assign_role(
        (df["AI点"] >= top_ai - 14)
        & (df["妙味スコア"] >= 3)
        & (pd.to_numeric(df["人気"], errors="coerce") >= 5),
        "妙味あり",
        1,
    )
    assign_role(
        (df["AI順位"] <= 5)
        | (df["AI点"] >= top_ai - 9)
        | ((pd.to_numeric(df["人気"], errors="coerce") <= 3) & (df["AI点"] >= top_ai - 15)),
        "相手有力",
        3,
    )
    assign_role(
        (df["AI順位"] <= 8)
        | (df["AI点"] >= top_ai - 16)
        | ((pd.to_numeric(df["★最高"], errors="coerce") >= field_star_high + 5) & (df["AI点"] >= top_ai - 20)),
        "押さえ",
        2,
    )

    role_order = {"本軸": 0, "穴軸": 1, "妙味あり": 2, "相手有力": 3, "押さえ": 4, "消し寄り": 5}
    df["_role_order"] = df["役割"].map(role_order).fillna(9)
    df = df.sort_values(["AI点", "推奨点", "近3走最高", "★最高", "_role_order"], ascending=[False, False, False, False, True]).reset_index(drop=True)
    df.insert(0, "推奨順位", range(1, len(df) + 1))

    def betting_mark(row):
        return {
            "本軸": "◎",
            "穴軸": "穴軸",
            "妙味あり": "妙",
            "相手有力": "○",
            "押さえ": "△",
            "消し寄り": "",
        }.get(row["役割"], "")

    def betting_note(mark):
        return {
            "◎": "本軸",
            "穴軸": "穴軸",
            "妙": "妙味",
            "○": "本線",
            "△": "押さえ",
        }.get(mark, "")

    df["印"] = df.apply(betting_mark, axis=1)
    df["買い目メモ"] = df["印"].map(betting_note)
    df["展開メモ"] = ""

    final_cols = ["推奨順位", "印", "役割", "買い目メモ", "妙味スコア", "AI順位", "枠", "馬番", "馬名", "性齢", "斤量", "騎手", "単勝オッズ", "人気", "コメント", "距離指数", "コース指数", "3走前", "2走前", "前走", "3走平均", "過去1年最高指数", "year_max_index", "★最高", "★該当走", "★条件", "★最高指数の取得元", "star_max_index", "star_max_race", "star_max_venue", "star_max_distance", "star_max_surface", "star_max_turn", "star_match_level", "star_max_source", "近3走最高", "対戦", "AI点", "推奨点"]
    return df[final_cols + [c for c in df.columns if c.startswith("_")]]


def circled_number(value):
    chars = {
        1: "①", 2: "②", 3: "③", 4: "④", 5: "⑤", 6: "⑥", 7: "⑦", 8: "⑧", 9: "⑨", 10: "⑩",
        11: "⑪", 12: "⑫", 13: "⑬", 14: "⑭", 15: "⑮", 16: "⑯", 17: "⑰", 18: "⑱",
    }
    try:
        number = int(value)
        return chars.get(number, str(number))
    except Exception:
        return str(value)


def format_horses(values):
    values = [v for v in values if pd.notna(v)]
    return "".join(circled_number(v) for v in values) if values else "なし"


def unique_horse_numbers(*groups, limit=None):
    result = []
    for group in groups:
        for value in group:
            if pd.isna(value):
                continue
            number = int(value)
            if number not in result:
                result.append(number)
    return result[:limit] if limit else result


def numeric_col(df, column, fallback=0):
    if column in df.columns:
        return pd.to_numeric(df[column], errors="coerce").fillna(fallback)
    return pd.Series(fallback, index=df.index, dtype="float64")


def add_focus_scores(df):
    df = df.copy()
    ai = numeric_col(df, "AI点")
    recommend = numeric_col(df, "推奨点", ai)
    avg3 = numeric_col(df, "3走平均", ai)
    recent = numeric_col(df, "近3走最高", avg3)
    star_high = numeric_col(df, "★最高", 0)
    similar_condition_high = numeric_col(df, "_similar_condition_high", 0)
    course = numeric_col(df, "コース指数", avg3)
    popularity = numeric_col(df, "人気", 8)
    h2h_score = numeric_col(df, "_h2h_score", 0)
    star_count = numeric_col(df, "_star_count", 0)
    layoff = df.get("_is_layoff", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    ages = df["性齢"].map(extract_age_from_sex_age) if "性齢" in df.columns else pd.Series(None, index=df.index)
    age_bonus = numeric_col(df, "_age_bonus", 0)

    pop_bonus = pd.Series(0.0, index=df.index)
    pop_bonus += (popularity <= 1).astype(float) * 5.0
    pop_bonus += ((popularity >= 2) & (popularity <= 3)).astype(float) * 4.0
    pop_bonus += ((popularity >= 4) & (popularity <= 5)).astype(float) * 1.5
    pop_bonus -= (popularity >= 10).astype(float) * 2.5

    focus = (
        ai * 0.45
        + recommend * 0.15
        + recent * 0.15
        + star_high * 0.15
        + similar_condition_high * 0.04
        + course * 0.10
        + pop_bonus
        + age_bonus * 0.8
        + star_count.clip(0, 3) * 1.2
        + h2h_score.clip(-2, 2) * 0.8
    )
    focus -= layoff.astype(float) * 3.0
    focus -= ((ages.fillna(0) >= 8) & (ai <= ai.max() - 25)).astype(float) * 2.0
    df["_focus_score"] = focus.round(1)
    return df


def get_betting_groups(df):
    return {
        "main_axis": df[df["役割"].eq("本軸")]["馬番"].tolist(),
        "value_axis": df[df["役割"].eq("穴軸")]["馬番"].tolist(),
        "value": df[df["役割"].eq("妙味あり")]["馬番"].tolist(),
        "main_partners": df[df["役割"].eq("相手有力")]["馬番"].tolist(),
        "reserves": df[df["役割"].eq("押さえ")]["馬番"].tolist(),
    }


def get_buy_candidates(groups):
    return unique_horse_numbers(
        groups["main_axis"],
        groups["value_axis"],
        groups["main_partners"],
        limit=2,
    )


def get_buy_candidates_from_df(df, limit=2):
    tmp = add_focus_scores(df)
    pool = tmp[tmp["役割"].isin(["本軸", "穴軸", "相手有力"])]
    if pool.empty:
        pool = tmp[tmp["役割"].isin(["押さえ"])]
    if pool.empty:
        pool = tmp
    return (
        pool.sort_values(["_focus_score", "推奨点", "AI点"], ascending=[False, False, False])
        ["馬番"]
        .astype(int)
        .tolist()[:limit]
    )


def get_watch_candidates(groups):
    return unique_horse_numbers(
        groups["main_axis"],
        groups["value_axis"],
        groups["main_partners"],
        groups["reserves"],
        limit=4,
    )


def front_condition_watch_mask(df):
    if "脚質" not in df.columns:
        return pd.Series(False, index=df.index)

    def numeric_series(column, default=None):
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce")
        return pd.Series(default, index=df.index, dtype="float64")

    styles = df["脚質"].map(normalize_running_style)
    distance = numeric_series("距離指数")
    course = numeric_series("コース指数")
    popularity = numeric_series("人気")
    ai = numeric_series("AI点")
    star_high = numeric_series("★最高")

    top_distance = distance.max()
    top_course = course.max()
    top_ai = ai.max()
    top_star = star_high.max()

    distance_high = distance.notna() & pd.notna(top_distance) & (distance >= top_distance - 6)
    course_high = course.notna() & pd.notna(top_course) & (course >= top_course - 6)
    star_fit = star_high.notna() & pd.notna(top_star) & (star_high >= top_star - 5)
    popular = popularity.fillna(99) <= 4
    not_too_low_ai = ai.notna() & pd.notna(top_ai) & (ai >= top_ai - 25)

    escape_pick = styles.eq("逃") & popular & (distance_high | course_high) & not_too_low_ai
    leader_pick = styles.eq("先") & popular & distance_high & course_high & (not_too_low_ai | star_fit)
    return escape_pick | leader_pick


def candidate_reason_masks(df):
    index = df.index

    def numeric_series(column, default=None):
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce")
        return pd.Series(default, index=index, dtype="float64")

    ai = numeric_series("AI点")
    ai_rank = numeric_series("AI順位")
    recommend = numeric_series("推奨点").fillna(ai)
    recent_high = numeric_series("近3走最高")
    star_high = numeric_series("★最高")
    similar_condition_high = numeric_series("_similar_condition_high")
    popularity = numeric_series("人気")
    value_score = numeric_series("妙味スコア", 0).fillna(0)
    h2h_score = numeric_series("_h2h_score", 0).fillna(0)
    ages = df["性齢"].map(extract_age_from_sex_age) if "性齢" in df.columns else pd.Series(None, index=index)
    trend = numeric_series("_trend", 0).fillna(0)
    relative_load_weight = numeric_series("_relative_load_weight", 0).fillna(0)
    comments = df.get("コメント", pd.Series("", index=index)).astype(str)
    race_comments = df.get("展開コメント", pd.Series("", index=index)).astype(str)
    pace_memo = df.get("展開メモ", pd.Series("", index=index)).astype(str)

    top_ai = ai.max()
    top_recommend = recommend.max()
    top_recent = recent_high.max()
    top_star = star_high.max()
    top_similar = similar_condition_high.max()

    old_low = (ages.fillna(0) >= 8) & ai.notna() & pd.notna(top_ai) & (ai <= top_ai - 25)
    not_too_low = ai.notna() & pd.notna(top_ai) & (ai >= top_ai - 30)
    popular = popularity.fillna(99) <= 5
    value_pop = popularity.fillna(0) >= 5

    ability = (
        (ai_rank <= 3)
        | (ai >= top_ai - 12)
        | ((recent_high >= top_recent - 6) & (ai >= top_ai - 24))
    )
    condition = (
        star_high.notna()
        & pd.notna(top_star)
        & (star_high >= top_star - 5)
        & not_too_low
    )
    similar_condition = (
        star_high.isna()
        & similar_condition_high.notna()
        & pd.notna(top_similar)
        & (similar_condition_high >= top_similar - 5)
        & not_too_low
    )
    pace = (
        front_condition_watch_mask(df)
        | pace_memo.str.contains("展開穴|注意", na=False)
        | race_comments.str.contains("展開穴|差し届く|能力で粘る|単騎なら粘る|好位で粘る|前残り警戒", na=False)
    ) & not_too_low
    development = (
        ages.isin([3, 4])
        & (
            popular
            | (trend >= 4)
            | ((ages == 3) & (relative_load_weight >= 2))
            | comments.str.contains("3歳|4歳|指数以上の支持|上昇", na=False)
        )
        & (ai >= top_ai - 40)
    )
    h2h = (
        (h2h_score >= 1)
        | comments.str.contains("対戦◎|対戦○|先着", na=False)
    ) & not_too_low
    value_hole = (
        value_pop
        & (
            (value_score >= 3)
            | condition
            | similar_condition
            | pace
            | h2h
            | ((recommend >= top_recommend - 12) & not_too_low)
        )
    )

    return {
        "能力上位": ability & ~old_low,
        "同条件": condition & ~old_low,
        "準適性": similar_condition & ~old_low,
        "展開": pace & ~old_low,
        "3-4歳上積み": development,
        "対戦": h2h & ~old_low,
        "妙味穴": value_hole & ~old_low,
    }


def get_candidate_reason_texts(df):
    masks = candidate_reason_masks(df)
    reasons = []
    for idx in df.index:
        labels = [label for label, mask in masks.items() if bool(mask.loc[idx])]
        reasons.append("、".join(labels))
    return pd.Series(reasons, index=df.index)


def get_watch_candidates_from_df(df, limit=None):
    tmp = add_focus_scores(df)
    masks = candidate_reason_masks(tmp)
    watch_mask = pd.Series(False, index=tmp.index)
    for mask in masks.values():
        watch_mask = watch_mask | mask
    pool = tmp[watch_mask]
    sort_cols = [col for col in ["_focus_score", "推奨点", "AI点"] if col in tmp.columns]
    if pool.empty:
        pool = tmp.sort_values(sort_cols, ascending=[False] * len(sort_cols)).head(3) if sort_cols else tmp.head(3)
    sorted_pool = pool.sort_values(sort_cols, ascending=[False] * len(sort_cols)) if sort_cols else pool
    horses = sorted_pool["馬番"].astype(int).tolist()
    return horses[:limit] if limit else horses


def add_candidate_marks(df, candidates=None):
    df = df.copy()
    candidates = set(int(x) for x in (candidates or get_watch_candidates_from_df(df)))
    df["候補"] = df["馬番"].astype(int).map(lambda value: "✓" if value in candidates else "")
    df["_候補理由"] = get_candidate_reason_texts(df)
    return df


def refresh_betting_labels(df):
    df = df.copy()

    def betting_mark(row):
        return {
            "本軸": "◎",
            "穴軸": "穴軸",
            "妙味あり": "妙",
            "相手有力": "○",
            "押さえ": "△",
            "消し寄り": "",
        }.get(row["役割"], "")

    def betting_note(mark):
        return {
            "◎": "本軸",
            "穴軸": "穴軸",
            "妙": "妙味",
            "○": "本線",
            "△": "押さえ",
        }.get(mark, "")

    df["印"] = df.apply(betting_mark, axis=1)
    df["買い目メモ"] = df["印"].map(betting_note)
    return df


def normalize_running_style(value):
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


def analyze_running_style(df):
    if "脚質" not in df.columns:
        return {
            "available": False,
            "ペース": "脚質HTMLなし",
            "展開傾向": "",
            "流れ": "",
            "有利脚質": [],
            "展開穴": [],
            "展開向く馬番": [],
            "脚質構成": {},
        }

    styles = df["脚質"].map(normalize_running_style)
    counts = {key: int((styles == key).sum()) for key in ["逃", "先", "差", "追"]}
    race_distances = pd.to_numeric(
        df.get("_race_distance", pd.Series(index=df.index, dtype="float64")),
        errors="coerce",
    ).dropna()
    race_distance = int(race_distances.iloc[0]) if not race_distances.empty else None
    if sum(counts.values()) == 0:
        return {
            "available": False,
            "ペース": "脚質HTMLなし",
            "展開傾向": "",
            "流れ": "",
            "有利脚質": [],
            "展開穴": [],
            "展開向く馬番": [],
            "脚質構成": counts,
            "距離": race_distance,
        }

    early_count = counts["逃"] + counts["先"]
    if race_distance is not None and race_distance <= 1000:
        pace = "短距離の先行争い"
        tendency = "逃げ・先行力重視"
        favored = ["逃", "先"]
    elif race_distance is not None and race_distance <= 1200 and early_count >= 4:
        pace = "短距離の先行激化"
        tendency = "先行力重視・差し警戒"
        favored = ["先", "差"]
    elif race_distance is not None and race_distance <= 1200:
        pace = "短距離ペース"
        tendency = "前残り注意"
        favored = ["逃", "先"]
    elif counts["逃"] == 1 and early_count <= 2:
        pace = "単騎逃げ注意"
        tendency = "逃げ残り警戒"
        favored = ["逃", "先"]
    elif early_count <= 1:
        pace = "スロー想定"
        tendency = "前残り注意"
        favored = ["逃", "先"]
    elif early_count >= 4:
        pace = "速くなりそう"
        tendency = "差し浮上"
        favored = ["差", "追"]
    else:
        pace = "平均ペース"
        tendency = "先行〜差し互角"
        favored = ["先", "差"]

    tmp = df.copy()
    tmp["_脚質正規化"] = styles
    popularity = pd.to_numeric(tmp.get("人気"), errors="coerce")
    odds = pd.to_numeric(tmp.get("単勝オッズ"), errors="coerce")
    style_win_rate = tmp.get("脚質勝率", pd.Series("", index=tmp.index)).map(
        lambda value: safe_num(parse_float_from_text(str(value or "")), 0)
    )
    score_source = tmp.get("推奨点", tmp.get("AI点", pd.Series(0, index=tmp.index)))
    tmp["_展開候補点"] = pd.to_numeric(score_source, errors="coerce").fillna(0)
    tmp["_展開候補点"] += tmp["_脚質正規化"].map({"逃": 6.0, "先": 3.0, "差": 2.0, "追": 1.0}).fillna(0)
    if counts["逃"] == 1:
        tmp.loc[tmp["_脚質正規化"].eq("逃"), "_展開候補点"] += 4.0
    tmp["_展開候補点"] += popularity.fillna(6).clip(1, 12) * 0.25
    tmp["_展開候補点"] += odds.fillna(0).clip(0, 30) * 0.03
    tmp["_展開候補点"] += style_win_rate.clip(0, 40) * 0.12

    lone_escape_numbers = tmp.loc[tmp["_脚質正規化"].eq("逃"), "馬番"].tolist()
    if counts["逃"] == 1 and early_count <= 2 and lone_escape_numbers:
        # レース考察で単騎逃げを主役にする時は、表の展開印も同じ馬へ寄せる。
        pace_holes = lone_escape_numbers[:1]
    else:
        hole_mask = (
            tmp["_脚質正規化"].isin(favored)
            & (popularity.fillna(99) >= 5)
            & ~tmp.get("役割", pd.Series("", index=tmp.index)).isin(["本軸"])
        )
        pace_holes = (
            tmp[hole_mask]
            .sort_values(["_展開候補点", "AI点"], ascending=[False, False])
            ["馬番"]
            .tolist()[:1]
        )

    return {
        "available": True,
        "ペース": pace,
        "展開傾向": tendency,
        "流れ": tendency,
        "有利脚質": favored,
        "展開穴": pace_holes,
        "展開向く馬番": pace_holes,
        "脚質構成": counts,
        "距離": race_distance,
    }


def add_corner_stretch_features(df, running_info=None):
    df = df.copy()
    running_info = running_info or {}
    favored = set(running_info.get("有利脚質", []))
    pace = str(running_info.get("ペース", ""))
    tendency = str(running_info.get("展開傾向", ""))
    pace_holes = set(int(x) for x in running_info.get("展開穴", []) if pd.notna(x))
    counts = running_info.get("脚質構成", {}) or {}
    early_count = counts.get("逃", 0) + counts.get("先", 0)
    early_compete = counts.get("逃", 0) >= 2 or counts.get("先", 0) >= 2 or early_count >= 3
    fast_flow = "速" in pace or "差し" in tendency
    front_flow = "スロー" in pace or "前残り" in tendency or "逃げ残り" in tendency

    styles = df["脚質"].map(normalize_running_style) if "脚質" in df.columns else pd.Series("", index=df.index)
    ai = pd.to_numeric(df.get("AI点"), errors="coerce")
    recommend = pd.to_numeric(df.get("推奨点"), errors="coerce")
    recent = pd.to_numeric(df.get("近3走最高"), errors="coerce")
    star_high = pd.to_numeric(df.get("★最高"), errors="coerce")
    course = pd.to_numeric(df.get("コース指数"), errors="coerce")
    popularity = pd.to_numeric(df.get("人気"), errors="coerce")
    style_win_rate = df.get("脚質勝率", pd.Series("", index=df.index)).apply(format_percent_value)

    top_ai = ai.max()
    top_recent = recent.max()
    top_course = course.max()

    corner_values = []
    stretch_values = []
    comments = []

    for idx, row in df.iterrows():
        style = styles.loc[idx]
        ai_value = ai.loc[idx]
        recent_value = recent.loc[idx]
        star_value = star_high.loc[idx]
        course_value = course.loc[idx]
        pop_value = popularity.loc[idx]

        if style == "逃":
            corner = "先頭争い" if counts.get("逃", 0) >= 2 else "先頭"
        elif style == "先":
            corner = "好位争い" if early_compete else "好位"
        elif style == "差":
            corner = "中団"
        elif style == "追":
            corner = "後方"
        else:
            corner = "中団想定"

        win_level = pd.notna(ai_value) and pd.notna(top_ai) and ai_value >= top_ai - 5
        strong_level = pd.notna(ai_value) and pd.notna(top_ai) and ai_value >= top_ai - 12
        mid_level = pd.notna(ai_value) and pd.notna(top_ai) and ai_value >= top_ai - 22
        recent_high = pd.notna(recent_value) and pd.notna(top_recent) and recent_value >= top_recent - 6
        course_high = pd.notna(course_value) and pd.notna(top_course) and course_value >= top_course - 5
        condition_fit = pd.notna(star_value)
        value_pop = pd.notna(pop_value) and pop_value >= 5
        style_win_text = style_win_rate.loc[idx]
        horse_no = int(row.get("馬番", 0) or 0)
        is_pace_hole = horse_no in pace_holes

        if style in ("逃", "先"):
            if front_flow and win_level:
                stretch = "押切候補"
                comment = "前残り有利"
            elif front_flow and (strong_level or condition_fit or course_high):
                stretch = "粘る"
                comment = "前残り警戒"
            elif win_level:
                stretch = "押切候補"
                comment = "同型次第" if early_compete else "勝ち負け"
            elif style == "逃" and counts.get("逃", 0) == 1 and (strong_level or condition_fit):
                stretch = "逃げ粘る"
                comment = "単騎なら粘る"
            elif fast_flow and not strong_level:
                stretch = "粘り課題"
                comment = "流れ厳しい"
            elif strong_level or condition_fit or course_high:
                stretch = "粘る"
                comment = "同型次第" if early_compete else "前で運べる"
            else:
                stretch = "粘り込み"
                comment = "流れひとつ"
        elif style == "差":
            if fast_flow and win_level:
                stretch = "差し届く"
                comment = "展開合う"
            elif fast_flow and (strong_level or recent_high or condition_fit):
                stretch = "差し浮上"
                comment = "差し届くか"
            elif front_flow and not win_level:
                stretch = "届き課題"
                comment = "前残り注意"
            elif win_level:
                stretch = "勝ち負け"
                comment = "直線勝負"
            elif strong_level or recent_high:
                stretch = "伸びる"
                comment = "直線伸びる"
            else:
                stretch = "展開待ち"
                comment = "差し届くか"
        elif style == "追":
            if fast_flow and win_level:
                stretch = "差し届く"
                comment = "前崩れなら届く"
            elif fast_flow and (strong_level or recent_high or condition_fit):
                stretch = "追込警戒"
                comment = "差し届くか"
            elif front_flow and not win_level:
                stretch = "届き待ち"
                comment = "前残りで割引"
            elif win_level and (condition_fit or recent_high):
                stretch = "差切警戒"
                comment = "能力で届く"
            elif win_level:
                stretch = "末脚勝負"
                comment = "能力上位"
            elif strong_level or condition_fit or recent_high:
                stretch = "差切警戒"
                comment = "末脚警戒"
            elif mid_level:
                stretch = "末脚勝負"
                comment = "展開ひとつ"
            else:
                stretch = "届き待ち"
                comment = "展開待ち"
        else:
            if win_level:
                stretch = "勝ち負け"
                comment = "AI上位"
            elif strong_level or recent_high:
                stretch = "伸びる"
                comment = "近況上位"
            else:
                stretch = "流れ次第"
                comment = "展開不明"

        if style in favored and value_pop and (strong_level or condition_fit or recent_high):
            comment = f"{comment}・妙味"
        if is_pace_hole and "展開穴" not in comment:
            comment = f"展開穴・{comment}"
        if style_win_text:
            comment = f"{comment}(脚質勝率{style_win_text})"

        corner_values.append(corner)
        stretch_values.append(stretch)
        comments.append(comment)

    df["4角予想"] = corner_values
    df["直線評価"] = stretch_values
    df["展開コメント"] = comments
    return df


def append_short_comment(text, phrase, max_parts=2):
    parts = [part for part in str(text or "").split("、") if part]
    if phrase not in parts:
        parts.append(phrase)
    return "、".join(parts[:max_parts])


def find_caution_horses(df, limit=1):
    if "脚質" not in df.columns:
        return []

    tmp = df.copy()
    tmp["_脚質正規化"] = tmp["脚質"].map(normalize_running_style)
    popularity = pd.to_numeric(tmp.get("人気"), errors="coerce")
    star_high = pd.to_numeric(tmp.get("★最高"), errors="coerce")
    top_ai = float(tmp["AI点"].max()) if len(tmp) else 0

    condition_text = tmp.get("コメント", pd.Series("", index=tmp.index)).astype(str)
    caution_mask = (
        tmp["役割"].eq("消し寄り")
        & tmp["_脚質正規化"].isin(["逃", "先"])
        & (popularity <= 3)
        & (
            star_high.notna()
            | condition_text.str.contains("条件", na=False)
            | (tmp["AI点"] >= top_ai - 25)
        )
    )

    style_rank = tmp["_脚質正規化"].map({"逃": 0, "先": 1}).fillna(9)
    candidates = tmp[caution_mask].assign(_style_rank=style_rank[caution_mask])
    if candidates.empty:
        return []

    return (
        candidates.sort_values(["人気", "_style_rank", "推奨点", "AI点"], ascending=[True, True, False, False])
        ["馬番"]
        .astype(int)
        .tolist()[:limit]
    )


def rebalance_betting_roles(df, pace_holes=None, target_total=6):
    df = df.copy()
    df = add_focus_scores(df)
    pace_holes = set(int(x) for x in (pace_holes or []))
    df["_元役割"] = df["役割"]
    df["役割"] = "消し寄り"
    selected = set()

    def sorted_candidates(mask):
        return df[mask & ~df.index.isin(selected)].sort_values(
            ["_focus_score", "推奨点", "AI点", "★最高", "3走平均"],
            ascending=[False, False, False, False, False],
        )

    def choose(mask, role_name, limit, preferred_horse_numbers=None):
        nonlocal selected
        if limit <= 0:
            return []
        chosen = []
        preferred_horse_numbers = [int(x) for x in (preferred_horse_numbers or [])]
        for horse_no in preferred_horse_numbers:
            preferred = df[(df["馬番"].astype(int) == horse_no) & mask & ~df.index.isin(selected)]
            for idx in preferred.index:
                if len(chosen) >= limit:
                    break
                df.at[idx, "役割"] = role_name
                selected.add(idx)
                chosen.append(idx)
        if len(chosen) < limit:
            for idx in sorted_candidates(mask).head(limit - len(chosen)).index:
                df.at[idx, "役割"] = role_name
                selected.add(idx)
                chosen.append(idx)
        return chosen

    popularity = pd.to_numeric(df.get("人気"), errors="coerce")
    top_ai = float(df["AI点"].max()) if len(df) else 0
    layoff_mask = df.get("_is_layoff", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    star_high = pd.to_numeric(df.get("★最高"), errors="coerce")
    field_star_high = star_high.mean()
    star_count = pd.to_numeric(df.get("_star_count"), errors="coerce").fillna(0)
    recent_high = pd.to_numeric(df.get("近3走最高"), errors="coerce")
    ai_rank = pd.to_numeric(df.get("AI順位"), errors="coerce")
    course_index = pd.to_numeric(df.get("コース指数"), errors="coerce")
    top_recent = recent_high.max()
    top_course = course_index.max()
    top_focus = pd.to_numeric(df.get("_focus_score"), errors="coerce").max()

    main_mask = (
        df["_元役割"].eq("本軸")
        & ~layoff_mask
    )
    if not main_mask.any():
        main_mask = (
            (
                ((star_high >= field_star_high + 4) & (df["AI点"] >= top_ai - 12) & (popularity.fillna(99) <= 4))
                | ((df["AI点"] >= top_ai - 3) & (popularity.fillna(99) <= 3))
            )
            & ~layoff_mask
        )
    main_indices = choose(main_mask, "本軸", 1)

    hole_mask = (
        df["_元役割"].isin(["穴軸", "妙味あり"])
        | ((df["妙味スコア"].fillna(0) >= 4) & (df["AI点"] >= top_ai - 14) & (popularity.fillna(99) >= 5))
    )
    remaining = max(0, target_total - len(selected))
    hole_indices = choose(hole_mask, "穴軸", min(1, remaining))

    remaining = max(0, target_total - len(selected))
    partner_mask = (
        df["_元役割"].isin(["相手有力", "本軸", "妙味あり"])
        | (df["AI点"] >= top_ai - 12)
        | ((popularity.fillna(99) <= 3) & (df["AI点"] >= top_ai - 16))
        | (df["_focus_score"] >= top_focus - 10)
        | (recent_high >= top_recent - 3)
        | (course_index >= top_course - 3)
        | df["馬番"].astype(int).isin(pace_holes)
    )
    if (main_indices or hole_indices) and remaining:
        choose(partner_mask, "相手有力", min(2, remaining), preferred_horse_numbers=list(pace_holes))

    remaining = max(0, target_total - len(selected))
    reserve_mask = (
        df["_元役割"].eq("押さえ")
        | df["馬番"].astype(int).isin(pace_holes)
        | ((ai_rank <= 6) & (df["AI点"] >= top_ai - 28))
        | ((recent_high >= top_recent - 8) & (df["AI点"] >= top_ai - 30))
        | (
            star_high.notna()
            & (
                (star_count >= 2)
                | (star_high >= field_star_high)
                | (df["_focus_score"] >= top_focus - 16)
            )
        )
        | ((popularity.fillna(99) <= 3) & (df["_focus_score"] >= top_focus - 18))
    )
    choose(reserve_mask, "押さえ", remaining, preferred_horse_numbers=list(pace_holes))

    df = refresh_betting_labels(df)
    role_order = {"本軸": 0, "穴軸": 1, "相手有力": 2, "押さえ": 3, "消し寄り": 4}
    df["_role_order"] = df["役割"].map(role_order).fillna(9)
    df = df.sort_values(["AI点", "推奨点", "近3走最高", "★最高", "_role_order"], ascending=[False, False, False, False, True]).reset_index(drop=True)
    return df


def apply_running_style_features(df):
    df = df.copy()
    if "展開メモ" not in df.columns:
        df["展開メモ"] = ""

    info = analyze_running_style(df)
    pace_holes = [int(x) for x in info.get("展開穴", [])]
    if pace_holes:
        df.loc[df["馬番"].astype(int).isin(pace_holes), "展開メモ"] = "展開穴"

    df = rebalance_betting_roles(df, pace_holes=pace_holes, target_total=6)
    if pace_holes:
        df.loc[df["馬番"].astype(int).isin(pace_holes), "展開メモ"] = "展開穴"
    caution_horses = find_caution_horses(df, limit=1)
    if caution_horses:
        mask = df["馬番"].astype(int).isin(caution_horses)
        df.loc[mask, "展開メモ"] = df.loc[mask, "展開メモ"].apply(
            lambda value: "注意" if not value else f"{value}・注意"
        )
        for idx in df[mask].index:
            comment = df.at[idx, "コメント"]
            style = normalize_running_style(df.at[idx, "脚質"])
            has_ability = df.at[idx, "AI点"] >= df["AI点"].max() - 25
            if style == "逃":
                phrase = "能力あり逃げ残り注意" if has_ability else "逃げ残り注意"
            else:
                phrase = "能力あり先行注意" if has_ability else "先行残り注意"
            df.at[idx, "コメント"] = append_short_comment(comment, phrase, max_parts=3)
    info["注意"] = caution_horses

    df = add_corner_stretch_features(df, info)
    return df, info


def add_final_race_evaluation(df, running_info=None):
    df = df.copy()
    info = running_info or analyze_running_style(df)
    ai = pd.to_numeric(df.get("AI点"), errors="coerce")
    popularity = pd.to_numeric(df.get("人気"), errors="coerce")
    odds = pd.to_numeric(df.get("単勝オッズ"), errors="coerce")
    ai_rank = ai.rank(method="min", ascending=False)
    styles = df.get("脚質", pd.Series("", index=df.index)).map(normalize_running_style)
    style_win_rate = df.get("脚質勝率", pd.Series("", index=df.index)).map(
        lambda value: safe_num(parse_float_from_text(str(value or "")), 0)
    )
    pace = str(info.get("ペース", ""))
    flow = str(info.get("流れ") or info.get("展開傾向", ""))
    favored = set(info.get("有利脚質", []))
    pace_horses = set(
        int(value)
        for value in (info.get("展開向く馬番") or info.get("展開穴", []))
        if pd.notna(value)
    )
    counts = info.get("脚質構成", {}) or {}
    top_ai = ai.max()

    pace_adjustments = []
    risks = []
    values = []

    for idx, row in df.iterrows():
        style = styles.loc[idx]
        horse_no = int(row.get("馬番", 0) or 0)
        rank_value = safe_num(ai_rank.loc[idx], len(df))
        pop_value = safe_num(popularity.loc[idx], None)
        odds_value = safe_num(odds.loc[idx], None)
        ai_value = safe_num(ai.loc[idx], 0)

        pace_adjustment = 0.0
        risk = 0.0
        value = 0.0

        if info.get("available"):
            if style in favored:
                pace_adjustment += 1.2
            elif style:
                risk += 0.6
            if horse_no in pace_horses:
                pace_adjustment += 2.5

            if "逃げ・先行力重視" in flow or "前残り" in flow or "逃げ残り" in flow:
                pace_adjustment += {"逃": 2.2, "先": 1.4, "差": -1.0, "追": -1.8}.get(style, 0)
                if style in ("差", "追"):
                    risk += 0.8
            elif "差し浮上" in flow:
                pace_adjustment += {"逃": -2.0, "先": -0.8, "差": 2.2, "追": 1.8}.get(style, 0)
                if style in ("逃", "先"):
                    risk += 1.0
            elif "好位組" in flow:
                pace_adjustment += {"逃": 0.2, "先": 1.6, "差": 0.8, "追": -0.8}.get(style, 0)
            elif "互角" in flow:
                pace_adjustment += {"先": 0.8, "差": 0.8}.get(style, 0)

            if "短距離" in pace and style == "追":
                risk += 0.8
            if counts.get("逃", 0) == 1 and style == "逃":
                pace_adjustment += 1.0
            rate = safe_num(style_win_rate.loc[idx], 0)
            if rate >= 15:
                pace_adjustment += 0.8
            elif rate >= 10:
                pace_adjustment += 0.4

        if pop_value is not None:
            rank_gap = pop_value - rank_value
            if rank_gap > 0:
                value += min(rank_gap, 8) * 0.45
            elif rank_gap <= -3:
                risk += min(abs(rank_gap), 6) * 0.45

        ability_near_top = pd.notna(top_ai) and ai_value >= top_ai - 15
        if odds_value is not None and ability_near_top:
            if odds_value >= 30:
                value += 1.5
            elif odds_value >= 15:
                value += 1.1
            elif odds_value >= 8:
                value += 0.6
        if horse_no in pace_horses and odds_value is not None and odds_value >= 8:
            value += 0.8

        if odds_value is not None:
            if odds_value <= 3 and rank_value >= 4:
                risk += 1.8
            elif odds_value <= 5 and rank_value >= 6:
                risk += 1.2

        pace_adjustments.append(round(max(-4.0, min(6.0, pace_adjustment)), 1))
        risks.append(round(max(0.0, min(6.0, risk)), 1))
        values.append(round(max(0.0, min(5.0, value)), 1))

    df["展開補正"] = pace_adjustments
    df["危険度"] = risks
    df["妙味"] = values
    df["最終評価"] = (
        ai.fillna(0)
        + pd.Series(pace_adjustments, index=df.index)
        + pd.Series(values, index=df.index)
        - pd.Series(risks, index=df.index)
    ).round(1)

    final_rank = df["最終評価"].rank(method="min", ascending=False)
    top_final = df["最終評価"].max()
    buy_labels = []
    for idx, row in df.iterrows():
        rank_value = int(final_rank.loc[idx])
        risk = float(row["危険度"])
        value = float(row["妙味"])
        final_value = float(row["最終評価"])
        horse_no = int(row.get("馬番", 0) or 0)
        if rank_value <= 2 and final_value >= top_final - 6 and risk <= 2.5:
            label = "本線"
        elif rank_value <= 4 and final_value >= top_final - 10 and risk <= 3.5:
            label = "相手"
        elif rank_value <= 7 and final_value >= top_final - 14 and value >= 2.0 and risk <= 4.0:
            label = "妙味"
        elif horse_no in pace_horses and rank_value <= 8 and final_value >= top_final - 16:
            label = "展開穴"
        elif rank_value <= 6 and final_value >= top_final - 12 and risk <= 4.5:
            label = "押さえ"
        else:
            label = ""
        buy_labels.append(label)
    df["買い候補"] = buy_labels
    return df.sort_values(
        ["最終評価", "AI点", "推奨点"],
        ascending=[False, False, False],
    ).reset_index(drop=True)


def _legacy_add_newspaper_features(df, running_info=None):
    df = df.copy()
    info = running_info or analyze_running_style(df)
    ai = pd.to_numeric(df.get("AI点"), errors="coerce")
    popularity = pd.to_numeric(df.get("人気"), errors="coerce")
    styles = df.get("脚質", pd.Series("", index=df.index)).map(normalize_running_style)
    style_win_rate = df.get("脚質勝率", pd.Series("", index=df.index)).map(
        lambda value: safe_num(parse_float_from_text(str(value or "")), 0)
    )
    favored = set(info.get("有利脚質", []))
    pace_horses = set(
        int(value)
        for value in (info.get("展開向く馬番") or info.get("展開穴", []))
        if pd.notna(value)
    )
    flow = str(info.get("流れ") or info.get("展開傾向", ""))
    counts = info.get("脚質構成", {}) or {}

    df["AI順位"] = ai.rank(method="min", ascending=False).astype("Int64")
    df["人気差"] = (popularity - df["AI順位"]).astype("Int64")

    pace_adjustments = []
    pace_types = []
    horse_comments = []
    top_ai = ai.max()

    for idx, row in df.iterrows():
        style = styles.loc[idx]
        horse_no = int(row.get("馬番", 0) or 0)
        ai_rank = int(df.at[idx, "AI順位"]) if pd.notna(df.at[idx, "AI順位"]) else len(df)
        popularity_gap = int(df.at[idx, "人気差"]) if pd.notna(df.at[idx, "人気差"]) else 0
        adjustment = 0.0

        if not info.get("available") or not style:
            pace_type = "展開待ち"
        elif "差し浮上" in flow:
            if style in ("差", "追"):
                pace_type = "ハイペース恩恵"
                adjustment += 2.2 if style == "差" else 1.8
            else:
                pace_type = "展開不向き"
                adjustment -= 1.8 if style == "逃" else 0.8
        elif "逃げ・先行力重視" in flow or "前残り" in flow or "逃げ残り" in flow:
            if style == "逃":
                pace_type = "追走楽" if counts.get("逃", 0) == 1 else "先行有利"
                adjustment += 2.2
            elif style == "先":
                pace_type = "先行有利"
                adjustment += 1.4
            elif style == "差":
                pace_type = "展開待ち"
                adjustment -= 0.8
            else:
                pace_type = "展開不向き"
                adjustment -= 1.5
        elif "好位組" in flow:
            if style == "先":
                pace_type = "先行有利"
                adjustment += 1.6
            elif style == "差":
                pace_type = "差し警戒"
                adjustment += 0.8
            elif style == "逃":
                pace_type = "自在性あり"
                adjustment += 0.2
            else:
                pace_type = "展開待ち"
                adjustment -= 0.6
        else:
            if style in ("先", "差"):
                pace_type = "自在性あり"
                adjustment += 0.8
            elif style == "逃":
                pace_type = "追走楽" if counts.get("逃", 0) == 1 else "展開待ち"
                adjustment += 0.6 if counts.get("逃", 0) == 1 else 0
            else:
                pace_type = "展開待ち"

        if style in favored:
            adjustment += 1.0
        if horse_no in pace_horses:
            adjustment += 2.0
            if "差し浮上" in flow:
                pace_type = "ハイペース恩恵"
            elif style in ("逃", "先"):
                pace_type = "追走楽"
        rate = safe_num(style_win_rate.loc[idx], 0)
        if rate >= 15:
            adjustment += 0.8
        elif rate >= 10:
            adjustment += 0.4
        adjustment = round(max(-3.0, min(5.0, adjustment)), 1)

        trend = safe_num(row.get("_trend"), 0)
        star_high = safe_num(row.get("★最高"), None)
        if ai_rank == 1:
            if adjustment < 0:
                comment = f"能力最上位。{pace_type}で流れ次第では取りこぼしも。"
            else:
                comment = f"能力最上位。{pace_type}の中で力を出せるか。"
        elif ai_rank <= 3:
            comment = f"能力上位。{pace_type}で持ち味を出せるか注目。"
        elif popularity_gap >= 3:
            comment = f"市場より能力指数が上。{pace_type}なら変化も。"
        elif popularity_gap <= -3:
            comment = f"人気が能力指数を先行。{pace_type}で内容を見たい。"
        elif trend >= 5:
            comment = f"近走上向き。{pace_type}なら指数以上の余地。"
        elif star_high is not None:
            comment = f"同条件実績あり。{pace_type}で指数再現に注目。"
        elif pd.notna(top_ai) and safe_num(ai.loc[idx], 0) >= top_ai - 15:
            comment = f"能力差は小さい。{pace_type}で着順変動の余地。"
        else:
            comment = f"能力指数は中位以下。{pace_type}で浮上余地を探る。"

        pace_adjustments.append(adjustment)
        pace_types.append(pace_type)
        horse_comments.append(comment[:40])

    df["展開補正"] = pace_adjustments
    df["展開タイプ"] = pace_types
    df["馬コメント"] = horse_comments
    return df.sort_values(["AI順位", "AI点"], ascending=[True, False]).reset_index(drop=True)


def _legacy_add_newspaper_features_v2(df, running_info=None):
    df = df.copy()
    info = running_info or analyze_running_style(df)
    df = df.drop(
        columns=[
            "展開補正",
            "危険度",
            "妙味",
            "最終評価",
            "買い候補",
            "推奨点",
            "展開タイプ",
            "馬コメント",
        ],
        errors="ignore",
    )

    ai = pd.to_numeric(df.get("AI点"), errors="coerce")
    popularity = pd.to_numeric(df.get("人気"), errors="coerce")
    odds = pd.to_numeric(df.get("単勝オッズ"), errors="coerce")
    latest = pd.to_numeric(df.get("_last", pd.Series(index=df.index, dtype="float64")), errors="coerce")
    average = pd.to_numeric(df.get("3走平均", pd.Series(index=df.index, dtype="float64")), errors="coerce")
    recent_high = pd.to_numeric(df.get("近3走最高", pd.Series(index=df.index, dtype="float64")), errors="coerce")
    styles = df.get("脚質", pd.Series("", index=df.index)).map(normalize_running_style)

    df["AI順位"] = ai.rank(method="min", ascending=False).astype("Int64")
    df["人気差"] = (popularity - df["AI順位"]).astype("Int64")
    latest_rank = latest.rank(method="min", ascending=False)
    average_rank = average.rank(method="min", ascending=False)
    high_rank = recent_high.rank(method="min", ascending=False)
    distance_index = pd.to_numeric(df.get("距離指数", pd.Series(index=df.index, dtype="float64")), errors="coerce")
    course_index = pd.to_numeric(df.get("コース指数", pd.Series(index=df.index, dtype="float64")), errors="coerce")
    distance_rank = distance_index.rank(method="min", ascending=False)
    course_rank = course_index.rank(method="min", ascending=False)
    star_rank = star_high.rank(method="min", ascending=False)
    popularity = pd.to_numeric(df.get("人気", pd.Series(index=df.index, dtype="float64")), errors="coerce")
    top_ai = ai.max()
    recent_ranges = df.get("_prev_values", pd.Series([[]] * len(df), index=df.index)).map(
        lambda values: (
            max(valid) - min(valid)
            if isinstance(values, list)
            and len(valid := [value for value in values if value is not None]) >= 2
            else None
        )
    )
    range_median = pd.to_numeric(recent_ranges, errors="coerce").median()
    star_high = pd.to_numeric(df.get("★最高"), errors="coerce")

    material_values = []
    comment_values = []
    for idx, row in df.iterrows():
        ai_rank = int(df.at[idx, "AI順位"]) if pd.notna(df.at[idx, "AI順位"]) else len(df)
        pop_value = safe_num(popularity.loc[idx], None)
        odds_value = safe_num(odds.loc[idx], None)
        style = styles.loc[idx]
        materials = []
        if ai_rank <= 5:
            materials.append("能力上位")
        if pd.notna(latest_rank.loc[idx]) and latest_rank.loc[idx] <= 3:
            materials.append("前走上位")
        row_range = safe_num(recent_ranges.loc[idx], None)
        if (
            pd.notna(average_rank.loc[idx])
            and average_rank.loc[idx] <= 5
            and row_range is not None
            and pd.notna(range_median)
            and row_range <= range_median
        ):
            materials.append("平均安定")
        if pd.notna(high_rank.loc[idx]) and high_rank.loc[idx] <= 3 and pop_value is not None and pop_value >= 6:
            materials.append("一発あり")
        if pd.notna(star_high.loc[idx]):
            materials.append("同条件実績")
        if style in ("逃", "先"):
            materials.append("先行力")
        if style in ("差", "追"):
            materials.append("差し脚")
        if odds_value is not None and 5 <= odds_value <= 30:
            materials.append("人気手頃")
        if pop_value is not None and pop_value >= 8 and ai_rank <= 10:
            materials.append("人気薄注意")

        selected_materials = materials[:3]
        material_values.append(" / ".join(selected_materials))

        if style == "逃":
            base = "主導権を握れれば粘り込み可能。競られる展開は課題。"
        elif style == "先":
            base = "好位で流れに乗れれば力を発揮。直線の粘り込みが鍵。"
        elif style == "差":
            base = "前が流れれば差し込み十分。進路と仕掛けのタイミングが鍵。"
        elif style == "追":
            base = "展開待ちの面はあるが、流れが向けば末脚を生かせる。"
        else:
            base = "位置取りは読みづらいが、流れに合わせて持ち味を出せるか。"

        if "前走上位" in selected_materials:
            detail = "前走指数も上位で、状態面の裏付けがある。"
        elif "同条件実績" in selected_materials:
            detail = "同条件での指数実績があり、再現できるか注目。"
        elif "能力上位" in selected_materials:
            detail = "能力指数は上位で、展開への対応が焦点。"
        elif "平均安定" in selected_materials:
            detail = "近3走の指数は安定しており、大崩れは少ない。"
        elif "一発あり" in selected_materials:
            detail = "高い指数を持ち、流れ次第で着順を上げる余地。"
        elif "人気薄注意" in selected_materials:
            detail = "市場人気より能力順位が高く、内容を見直したい。"
        else:
            detail = "能力指数と展開の噛み合いを見極めたい。"
        comment_values.append((base + detail)[:60])

    df["買い材料"] = material_values
    df["展開コメント"] = comment_values
    return df.sort_values(["AI順位", "AI点"], ascending=[True, False]).reset_index(drop=True)


def add_newspaper_features(df, running_info=None):
    df = df.copy()
    df = df.drop(
        columns=[
            "展開補正",
            "危険度",
            "妙味",
            "最終評価",
            "推奨点",
            "買い候補",
            "買い材料",
            "展開タイプ",
            "馬コメント",
            "役割",
            "印",
            "買い目メモ",
            "候補",
            "推奨順位",
            "実戦評価",
            "馬券材料",
            "妙味スコア",
            "人気差",
        ],
        errors="ignore",
    )

    ai = pd.to_numeric(df.get("AI点"), errors="coerce")
    average = pd.to_numeric(df.get("3走平均", pd.Series(index=df.index, dtype="float64")), errors="coerce")
    recent_high = pd.to_numeric(df.get("近3走最高", pd.Series(index=df.index, dtype="float64")), errors="coerce")
    styles = df.get("脚質", pd.Series("", index=df.index)).map(normalize_running_style)
    prev_values = df.get("_prev_values", pd.Series([[]] * len(df), index=df.index))
    star_high = pd.to_numeric(df.get("★最高", pd.Series(index=df.index, dtype="float64")), errors="coerce")
    distance_index = pd.to_numeric(df.get("距離指数", pd.Series(index=df.index, dtype="float64")), errors="coerce")
    course_index = pd.to_numeric(df.get("コース指数", pd.Series(index=df.index, dtype="float64")), errors="coerce")
    distance_rank = distance_index.rank(method="min", ascending=False)
    course_rank = course_index.rank(method="min", ascending=False)
    star_rank = star_high.rank(method="min", ascending=False)
    popularity = pd.to_numeric(df.get("人気", pd.Series(index=df.index, dtype="float64")), errors="coerce")
    top_ai = ai.max()
    same_distance = pd.to_numeric(df.get("_same_distance_high", pd.Series(index=df.index, dtype="float64")), errors="coerce")

    df["平均指数"] = average
    df["最高指数"] = recent_high
    df["★最高指数"] = star_high
    df["AI順位"] = ai.rank(method="min", ascending=False).astype("Int64")

    def newspaper_age_adjustment(row):
        age = extract_age_from_sex_age(row.get("性齢"))
        trend = safe_num(row.get("_trend"), 0)
        relative_load_weight = safe_num(row.get("_relative_load_weight"), 0)
        adjustment = 0.0
        if age == 3:
            adjustment += 3.0
            if trend >= 4:
                adjustment += 1.5
            if relative_load_weight >= 2:
                adjustment += 1.5
        elif age == 4:
            adjustment += 2.2
            if trend >= 4:
                adjustment += 1.2
            if relative_load_weight >= 2:
                adjustment += 0.5
        elif age is not None and age >= 11:
            adjustment -= 3.0 if trend >= 4 else 4.0
        elif age is not None and age >= 9:
            adjustment -= 0.5 if trend >= 4 else 1.5
        return round(adjustment, 1)

    df["年齢補正"] = df.apply(newspaper_age_adjustment, axis=1)

    average_rank = average.rank(method="min", ascending=False)
    high_rank = recent_high.rank(method="min", ascending=False)
    ranges = prev_values.map(
        lambda values: (
            max(valid) - min(valid)
            if isinstance(values, list)
            and len(valid := [value for value in values if value is not None]) >= 2
            else None
        )
    )
    range_median = pd.to_numeric(ranges, errors="coerce").median()

    abilities = []
    aptitudes = []
    momentums = []
    horse_types = []
    comments = []

    for idx, row in df.iterrows():
        ai_rank = int(df.at[idx, "AI順位"]) if pd.notna(df.at[idx, "AI順位"]) else len(df)
        avg_rank = safe_num(average_rank.loc[idx], len(df))
        max_rank = safe_num(high_rank.loc[idx], len(df))
        row_range = safe_num(ranges.loc[idx], None)
        values = row.get("_prev_values")
        valid = [value for value in values if value is not None] if isinstance(values, list) else []
        rising = len(valid) == 3 and valid[0] < valid[1] < valid[2]
        style = styles.loc[idx]
        has_condition = pd.notna(star_high.loc[idx]) or pd.notna(same_distance.loc[idx])

        if ai_rank <= 3:
            ability = "能力上位"
        elif ai_rank <= 6:
            ability = "能力中上位"
        elif ai_rank <= 10:
            ability = "能力中位"
        else:
            ability = "能力下位"

        if pd.notna(star_high.loc[idx]):
            aptitude = "同条件実績"
        elif pd.notna(same_distance.loc[idx]):
            aptitude = "距離実績"
        else:
            aptitude = "条件未知"

        if rising:
            momentum = "上昇"
        elif len(valid) >= 2 and valid[-1] >= valid[0] + 5:
            momentum = "良化"
        elif row_range is not None and pd.notna(range_median) and row_range <= range_median:
            momentum = "安定"
        elif len(valid) >= 2 and valid[-1] <= valid[0] - 5:
            momentum = "下降"
        else:
            momentum = "横ばい"

        stable = avg_rank <= 5 and row_range is not None and pd.notna(range_median) and row_range <= range_median
        one_shot = max_rank <= 3 and avg_rank >= 7
        if ai_rank <= 3:
            horse_type = "能力型"
        elif stable:
            horse_type = "安定型"
        elif rising:
            horse_type = "上昇型"
        elif one_shot:
            horse_type = "一発型"
        elif style in ("逃", "先"):
            horse_type = "先行型"
        elif style in ("差", "追"):
            horse_type = "差し型"
        elif has_condition:
            horse_type = "条件型"
        else:
            horse_type = "安定型"

        comment_parts = []
        if ai_rank <= 3:
            comment_parts.append("能力上位で中心候補")
        elif ai_rank <= 6:
            comment_parts.append("能力は相手圏")
        elif ai_rank <= 10:
            comment_parts.append("能力は中位、条件で浮上余地")
        else:
            comment_parts.append("能力面は展開待ち")

        row_star_rank = safe_num(star_rank.loc[idx], None)
        row_high_rank = safe_num(high_rank.loc[idx], None)
        row_dist_rank = safe_num(distance_rank.loc[idx], None)
        row_course_rank = safe_num(course_rank.loc[idx], None)
        if row_star_rank is not None and row_star_rank <= 1:
            comment_parts.append("★最高最上位で条件実績が強い")
        elif row_star_rank is not None and row_star_rank <= 3:
            comment_parts.append("★最高上位で条件実績あり")
        elif row_high_rank is not None and row_high_rank <= 3:
            comment_parts.append("近走最高指数は上位")

        if row_dist_rank is not None and row_course_rank is not None and row_dist_rank <= 3 and row_course_rank <= 3:
            comment_parts.append("距離・コース指数とも上位")
        elif row_dist_rank is not None and row_dist_rank <= 3:
            comment_parts.append("距離指数上位")
        elif row_course_rank is not None and row_course_rank <= 3:
            comment_parts.append("コース指数上位")

        class_note = str(row.get("クラス変動") or "")
        if class_note in ("クラス降級", "相手弱化"):
            comment_parts.append("クラス降級で相手関係は楽")
        elif class_note in ("クラス昇級", "相手強化"):
            comment_parts.append("昇級で相手強化は注意")

        pop_value = safe_num(popularity.loc[idx], None)
        if pop_value is not None and pd.notna(ai_rank) and pop_value - ai_rank >= 3:
            comment_parts.append("人気以上に評価できる")

        if momentum in ("上昇", "良化"):
            comment_parts.append("近走は上向き")
        elif momentum == "下降":
            comment_parts.append("近走低下は割引")

        if style == "逃":
            style_note = "前で運べるが競られると粘りが課題"
        elif style == "先":
            style_note = "好位で運べれば指数を生かせる"
        elif style == "差":
            style_note = "差し脚はあるが展開の後押しが欲しい"
        elif style == "追":
            style_note = "後方からで展開待ち"
        else:
            style_note = "位置取り次第"
        comment_parts.append(style_note)

        unique_comment_parts = []
        for part in comment_parts:
            if part and part not in unique_comment_parts:
                unique_comment_parts.append(part)
        comment = "。".join(unique_comment_parts[:4]) + "。"

        abilities.append(ability)
        aptitudes.append(aptitude)
        momentums.append(momentum)
        horse_types.append(horse_type)
        comments.append(comment[:90])

    df["能力"] = abilities
    df["適性"] = aptitudes
    df["勢い"] = momentums
    df["馬タイプ"] = horse_types
    df["展開コメント"] = comments
    return df.sort_values(["AI順位", "AI点"], ascending=[True, False]).reset_index(drop=True)


def analyze_race_shape(df):
    by_ai = df.sort_values("AI点", ascending=False).reset_index(drop=True)
    top_ai = float(by_ai.loc[0, "AI点"])
    second_gap = top_ai - float(by_ai.loc[1, "AI点"]) if len(by_ai) > 1 else 99
    fifth_gap = top_ai - float(by_ai.loc[min(4, len(by_ai) - 1), "AI点"]) if len(by_ai) else 0
    top3 = by_ai.head(3)
    top6 = by_ai.head(6)
    top3_popular = int((pd.to_numeric(top3["人気"], errors="coerce") <= 3).sum())
    value_count = int((pd.to_numeric(top6["妙味スコア"], errors="coerce") >= 3).sum())

    stars = 1
    if second_gap <= 2.0:
        stars += 1
    if fifth_gap <= 8.0:
        stars += 1
    if value_count >= 1:
        stars += 1
    if value_count >= 2 or top3_popular <= 1:
        stars += 1
    stars = max(1, min(5, stars))

    if top3_popular >= 2 and second_gap >= 4 and value_count == 0:
        race_eval = "堅い"
    elif stars <= 2:
        race_eval = "やや堅い"
    elif stars <= 4:
        race_eval = "波乱含み"
    else:
        race_eval = "荒れそう"

    groups = get_betting_groups(df)
    if stars >= 5 and not groups["main_axis"] and not groups["value_axis"]:
        policy = "見送り寄り"
    elif groups["main_axis"] and groups["value_axis"]:
        policy = "実戦2頭まで"
    elif groups["value_axis"]:
        policy = "妙味を確認"
    elif groups["main_axis"]:
        policy = "上位中心"
    else:
        policy = "見送り寄り"

    return {
        "波乱度": "★" * stars + "☆" * (5 - stars),
        "レース評価": race_eval,
        "推奨方針": policy,
    }


def print_race_evaluation(df):
    shape = analyze_race_shape(df)
    running = analyze_running_style(df)
    print("【レース評価】")
    print(f"波乱度：{shape['波乱度']}")
    print(f"レース評価：{shape['レース評価']}")
    print(f"推奨方針：{shape['推奨方針']}")
    if running.get("available"):
        counts = running["脚質構成"]
        print("")
        print("【展開予想】")
        print(f"脚質構成：逃{counts.get('逃', 0)} 先{counts.get('先', 0)} 差{counts.get('差', 0)} 追{counts.get('追', 0)}")
        if running.get("距離"):
            print(f"距離補正：{running['距離']}m")
        print(f"ペース：{running['ペース']} / {running['展開傾向']}")
        print(f"有利脚質：{'・'.join(running['有利脚質'])}")
        print(f"展開穴：{format_horses(running['展開穴'])}")


def horse_label(row):
    return f"{circled_number(row.get('馬番'))}{row.get('馬名', '')}"


def _legacy_print_race_scenario(df):
    tmp = df.copy()
    if "AI順位" not in tmp.columns or "展開タイプ" not in tmp.columns:
        tmp = add_newspaper_features(tmp)
    running = analyze_running_style(tmp)
    counts = running.get("脚質構成", {}) or {}
    escape_count = counts.get("逃", 0)
    early_count = escape_count + counts.get("先", 0)
    race_distance = running.get("距離")

    if not running.get("available"):
        first_paragraph = (
            "脚質データがないため隊列は読み切れない。"
            "能力指数と同条件実績を基準に、各馬の位置取りと仕掛けを想像したい。"
        )
    elif race_distance is not None and race_distance <= 1200 and early_count >= 4:
        first_paragraph = (
            f"{race_distance}mの短距離で逃げ・先行が{early_count}頭。"
            "序盤から位置取り争いが続き、前の組は息を入れる間を作れるかが焦点になる。"
        )
    elif early_count >= 5 or escape_count >= 2:
        first_paragraph = (
            "序盤から先行争いが激しくなりそう。"
            "逃げ・先行勢は道中で脚を使い、中団待機組にも出番が生まれる流れ。"
        )
    elif escape_count == 1 and early_count <= 3:
        first_paragraph = (
            "逃げ馬が単騎で運べる可能性がある。"
            "道中が落ち着けば前が残り、早めに動く馬が出れば差しも届く形。"
        )
    elif early_count <= 1:
        first_paragraph = (
            "前へ行く馬が少なく、序盤は落ち着いた流れを想定。"
            "後方勢は仕掛けのタイミングが遅れると、位置取りの差が残りやすい。"
        )
    else:
        first_paragraph = (
            "極端な先行争いにはなりにくく、平均的な流れを想定。"
            "好位勢と中団勢の仕掛けどころで、直線の並びが変わりそう。"
        )

    by_ai = tmp.sort_values(["AI順位", "AI点"], ascending=[True, False])
    top_labels = "、".join(horse_label(row) for _, row in by_ai.head(2).iterrows())
    second_parts = [f"能力指数では{top_labels}が上位。"]

    popularity_gap = pd.to_numeric(tmp.get("人気差"), errors="coerce")
    gap_pool = tmp[popularity_gap >= 3].sort_values(["人気差", "AI点"], ascending=[False, False])
    if not gap_pool.empty:
        row = gap_pool.iloc[0]
        second_parts.append(f"{horse_label(row)}は市場評価よりAI順位が高い。")

    star_high = pd.to_numeric(tmp.get("★最高"), errors="coerce")
    if star_high.notna().any():
        star_row = tmp.loc[star_high.idxmax()]
        second_parts.append(f"{horse_label(star_row)}は同条件指数に裏付けがある。")

    second_parts.append("能力差だけでなく、位置取りと仕掛けの順番で着順が入れ替わる余地がある。")

    print("【レース想定】")
    print(first_paragraph)
    print("")
    print("".join(second_parts))


def _legacy_print_race_scenario_v2(df):
    tmp = df.copy()
    if "AI順位" not in tmp.columns or "買い材料" not in tmp.columns:
        tmp = add_newspaper_features(tmp)
    running = analyze_running_style(tmp)
    counts = running.get("脚質構成", {}) or {}
    escape_count = counts.get("逃", 0)
    leader_count = counts.get("先", 0)
    early_count = escape_count + leader_count
    distance = running.get("距離")
    flow = str(running.get("流れ") or running.get("展開傾向", ""))

    if not running.get("available"):
        opening = "脚質データがないため隊列は読み切れず、各馬の出方を見ながら進む形。"
        middle = "道中の位置取りとペース変化が読みの中心となり、能力差だけでは決まりにくい。"
        stretch = "直線では余力を残した馬が伸びる形で、仕掛けのタイミングが着順を左右する。"
    elif distance is not None and distance <= 1200:
        opening = f"{distance}mの短距離で逃げ・先行が{early_count}頭。序盤から位置取り争いが続きそう。"
        middle = "前の組は息を入れる区間を作れるかが鍵で、中団勢は離されず追走したい。"
        stretch = "前残りの可能性を残しつつ、先行争いが長引けば差し馬にも出番が生まれる。"
    elif early_count >= 5 or escape_count >= 2:
        opening = "逃げ・先行馬が揃い、前半からある程度流れる展開になりそう。"
        middle = "先行勢は脚を使わされる可能性があり、好位から中団で脚を溜める馬にも機会。"
        stretch = "前の余力が薄れれば差し馬が浮上し、直線で着順が入れ替わる形も考えられる。"
    elif escape_count == 1 and early_count <= 3:
        opening = "逃げ馬が単騎で運べる可能性があり、序盤は落ち着いた流れを想定。"
        middle = "先行勢は好位で折り合いやすく、後方勢は早めに差を詰める必要がある。"
        stretch = "前残りを意識しつつ、途中で動く馬がいれば差し馬の末脚も届く余地。"
    elif early_count <= 1:
        opening = "前へ行く馬が少なく、序盤はゆったりした流れになりやすい。"
        middle = "位置取りの差が残りやすく、後方勢は仕掛けを遅らせないことが重要。"
        stretch = "先行勢の粘りが焦点で、差し馬は直線だけで届くかが課題となる。"
    else:
        opening = "極端な先行争いにはなりにくく、平均的な流れで隊列が決まりそう。"
        middle = "好位勢と中団勢が余力を残し、各馬の仕掛けどころが重なる展開。"
        stretch = "前後の有利不利は小さく、能力指数と直線での反応が着順を左右する。"

    by_ai = tmp.sort_values(["AI順位", "AI点"], ascending=[True, False])
    top_names = "、".join(horse_label(row) for _, row in by_ai.head(2).iterrows())
    gap = pd.to_numeric(tmp.get("人気差"), errors="coerce")
    gap_pool = tmp[gap >= 3].sort_values(["人気差", "AI点"], ascending=[False, False])
    note = f"能力指数では{top_names}が上位。"
    if not gap_pool.empty:
        row = gap_pool.iloc[0]
        note += f"{horse_label(row)}は人気よりAI順位が高く、流れとの噛み合いで着順が動く余地。"
    elif "差し浮上" in flow:
        note += "能力上位馬だけでなく、流れを受ける差し馬の位置取りにも注目したい。"
    else:
        note += "能力差だけでなく、道中の位置取りと仕掛けの順番も見比べたい。"

    print("【レース想定】")
    print("")
    print("序盤：")
    print(opening)
    print("")
    print("中盤：")
    print(middle)
    print("")
    print("直線：")
    print(stretch)
    print("")
    print("注目点：")
    print(note)


def build_value_angle_sentence(df, limit=2):
    if df is None or len(df) == 0:
        return ""
    tmp = df.copy()

    def numeric(column):
        if column in tmp.columns:
            return pd.to_numeric(tmp[column], errors="coerce")
        return pd.Series(float("nan"), index=tmp.index, dtype="float64")

    ai_rank = numeric("AI順位")
    popularity = numeric("人気")
    odds = numeric("単勝オッズ")
    ai = numeric("AI点")
    distance = numeric("距離指数")
    course = numeric("コース指数")
    star_high = numeric("★最高指数").fillna(numeric("★最高"))

    mark = tmp.get("最終印", pd.Series("", index=tmp.index)).astype(str)
    pace_mark = tmp.get("展開印", pd.Series("", index=tmp.index)).astype(str)
    reason = tmp.get("印理由", pd.Series("", index=tmp.index)).astype(str)
    comment = tmp.get("展開コメント", pd.Series("", index=tmp.index)).astype(str)
    h2h = tmp.get("対戦", pd.Series("", index=tmp.index)).astype(str)
    class_shift = tmp.get("クラス変動", pd.Series("", index=tmp.index)).astype(str)

    distance_rank = distance.rank(method="min", ascending=False)
    course_rank = course.rank(method="min", ascending=False)
    star_rank = star_high.rank(method="min", ascending=False)

    value_gap = popularity - ai_rank
    material_mask = (
        pace_mark.eq("展")
        | reason.str.contains("展開|クラス降級|人気以上|オッズ妙味|同条件実績|距離指数上位|コース指数上位", regex=True, na=False)
        | comment.str.contains("展開向く|人気以上|距離指数上位|コース指数上位|浮上余地", regex=True, na=False)
        | h2h.str.contains("先着|対戦", regex=True, na=False)
        | class_shift.eq("クラス降級")
        | distance_rank.le(3).fillna(False)
        | course_rank.le(3).fillna(False)
        | star_rank.le(3).fillna(False)
    )
    value_mask = (
        value_gap.ge(2).fillna(False)
        | odds.ge(8).fillna(False)
        | (popularity.ge(5).fillna(False) & ai_rank.le(8).fillna(False))
    )
    core_mask = mark.isin(["◎", "○"])
    candidates = tmp[material_mask & value_mask & ~core_mask].copy()
    if candidates.empty:
        return ""

    candidates["_妙味差"] = value_gap.reindex(candidates.index).fillna(0)
    candidates["_妙味オッズ"] = odds.reindex(candidates.index).fillna(0)
    candidates["_妙味AI"] = ai.reindex(candidates.index).fillna(0)
    candidates["_妙味印"] = pace_mark.reindex(candidates.index).eq("展").astype(int)
    candidates = candidates.sort_values(
        ["_妙味印", "_妙味差", "_妙味AI", "_妙味オッズ"],
        ascending=[False, False, False, False],
    ).head(limit)

    labels = []
    for _, row in candidates.iterrows():
        number = pd.to_numeric(row.get("馬番"), errors="coerce")
        no_text = circled_number(int(number)) if pd.notna(number) else str(row.get("馬番", "")).strip()
        name = str(row.get("馬名", "")).strip()
        labels.append(f"{no_text}{name}" if name else no_text)
    if not labels:
        return ""

    material_text = "条件・展開・対戦材料" if "対戦" in tmp.columns else "条件・展開・クラス材料"
    return f"妙味を取るなら{'、'.join(labels)}。人気より{material_text}を優先して確認。"


def print_race_scenario(df):
    tmp = df.copy()
    if "AI順位" not in tmp.columns or "馬タイプ" not in tmp.columns:
        tmp = add_newspaper_features(tmp)
    styles = tmp.get("脚質", pd.Series("", index=tmp.index)).map(normalize_running_style)
    tmp["_脚質表示"] = styles

    def names_for(style_values, limit):
        pool = tmp[tmp["_脚質表示"].isin(style_values)].sort_values(
            ["AI順位", "AI点"], ascending=[True, False]
        ).head(limit)
        labels = []
        for _, row in pool.iterrows():
            horse_no = pd.to_numeric(row.get("馬番"), errors="coerce")
            no_text = str(int(horse_no)) if pd.notna(horse_no) else str(row.get("馬番", "")).strip()
            name = str(row.get("馬名", "")).strip()
            labels.append(f"{no_text} {name}".strip())
        return labels

    escape_names = names_for(["逃"], 3)
    leader_names = names_for(["先"], 5)
    closer_names = names_for(["差", "追"], 5)

    def names_text(names):
        return "、".join(names) if names else "該当馬なし"

    escape_count = len(tmp[tmp["_脚質表示"].eq("逃")])
    early_count = len(tmp[tmp["_脚質表示"].isin(["逃", "先"])])
    if escape_count >= 2 or early_count >= 5:
        pace = "ハイペース"
    elif escape_count == 1 and early_count <= 3:
        pace = "スロー"
    else:
        pace = "平均"

    if escape_names:
        lead_name = escape_names[0]
        if len(escape_names) >= 2:
            race_text = (
                f"{lead_name}がハナを主張し、{escape_names[1]}も前へ出る構え。"
                f"逃げ争いが続けば、ペースは{pace}まで上がりそう。"
            )
        else:
            race_text = f"{lead_name}が主導権を握りそう。単騎で運べればペースは{pace}を想定。"
    elif leader_names:
        lead_name = leader_names[0]
        race_text = f"明確な逃げ馬は少なく、{lead_name}が押し出される形。ペースは{pace}を想定。"
    else:
        lead_name = ""
        race_text = f"前へ行く馬が読みづらく、序盤の出方次第。ペースは{pace}を想定。"

    if leader_names:
        race_text += f"{names_text(leader_names[:3])}は好位付近を追走しそう。"
    if closer_names:
        race_text += f"{names_text(closer_names[:3])}は中団より後ろで脚を溜める形。"

    front_straight = (escape_names + leader_names)[:2]
    close_straight = closer_names[:2]
    straight_parts = []
    if front_straight:
        straight_parts.append(f"{names_text(front_straight)}が前から粘り込みを図る")
    if close_straight:
        straight_parts.append(f"{names_text(close_straight)}が外や馬群の間から差を詰める")
    straight_text = "。".join(straight_parts) + "。" if straight_parts else "各馬の位置取りと仕掛けの差が出る直線になりそう。"

    if escape_count == 1 and escape_names:
        attention = (
            f"{escape_names[0]}が単騎で運べれば前の余力が残りやすい。"
            f"早めに先行勢が動けば、{names_text(close_straight)}の末脚が届く流れへ変わる。"
        )
    elif escape_count >= 2 and len(escape_names) >= 2:
        attention = (
            f"{escape_names[0]}と{escape_names[1]}の主導権争いが分岐点。"
            f"競り合いが長引けば、{names_text(close_straight)}が直線で差を縮める形。"
        )
    else:
        attention = (
            f"{names_text(front_straight)}が楽に好位を取れるかが分岐点。"
            f"前半が想定より速くなれば、{names_text(close_straight)}の出番が増える。"
        )

    print("【逃げ候補】")
    print(names_text(escape_names))
    print("")
    print("【先行候補】")
    print(names_text(leader_names))
    print("")
    print("【差し候補】")
    print(names_text(closer_names))
    print("")
    print("【レース想定】")
    print(race_text)
    print("")
    print("【直線】")
    print(straight_text)
    print("")
    print("【注目点】")
    print(attention)
    value_sentence = build_value_angle_sentence(tmp)
    if value_sentence:
        print(value_sentence)


def _legacy_print_race_insight(df):
    tmp = df.copy()
    if "_候補理由" not in tmp.columns:
        tmp["_候補理由"] = get_candidate_reason_texts(tmp)
    if "_focus_score" not in tmp.columns:
        tmp = add_focus_scores(tmp)
    if "推奨点" not in tmp.columns:
        tmp["推奨点"] = pd.to_numeric(tmp.get("AI点", pd.Series(0, index=tmp.index)), errors="coerce").fillna(0)

    ai = pd.to_numeric(tmp.get("AI点"), errors="coerce")
    popularity = pd.to_numeric(tmp.get("人気"), errors="coerce")
    value_score = pd.to_numeric(tmp.get("妙味スコア", pd.Series(0, index=tmp.index)), errors="coerce").fillna(0)
    ages = tmp["性齢"].map(extract_age_from_sex_age) if "性齢" in tmp.columns else pd.Series(None, index=tmp.index)
    styles = tmp["脚質"].map(normalize_running_style) if "脚質" in tmp.columns else pd.Series("", index=tmp.index)
    relative_weight = pd.to_numeric(
        tmp.get("_relative_load_weight", pd.Series(index=tmp.index, dtype="float64")),
        errors="coerce",
    )
    style_win_rate = tmp.get("脚質勝率", pd.Series("", index=tmp.index)).map(
        lambda value: safe_num(parse_float_from_text(str(value or "")), 0)
    )
    course_index = pd.to_numeric(tmp.get("コース指数"), errors="coerce")
    star_high = pd.to_numeric(tmp.get("★最高"), errors="coerce")
    recent_high = pd.to_numeric(tmp.get("近3走最高"), errors="coerce")
    similar_high = pd.to_numeric(tmp.get("_similar_condition_high"), errors="coerce")
    trend = pd.to_numeric(
        tmp.get("_trend", pd.Series(index=tmp.index, dtype="float64")),
        errors="coerce",
    )
    latest = pd.to_numeric(
        tmp.get("_last", pd.Series(index=tmp.index, dtype="float64")),
        errors="coerce",
    )
    last_same_condition = tmp.get(
        "_last_same_condition",
        pd.Series(False, index=tmp.index, dtype="bool"),
    ).fillna(False).astype(bool)
    focus_score = pd.to_numeric(tmp.get("_focus_score"), errors="coerce").fillna(0)

    def fill_with_field_mean(series, fallback=0):
        mean_value = series.mean()
        return series.fillna(mean_value if pd.notna(mean_value) else fallback)

    tmp["_insight_course_score"] = (
        fill_with_field_mean(course_index) * 0.35
        + star_high.fillna(0) * 0.35
        + fill_with_field_mean(recent_high) * 0.20
        + focus_score * 0.05
        + style_win_rate.clip(0, 40) * 0.05
    )
    tmp["_insight_pace_score"] = (
        tmp["_insight_course_score"]
        + focus_score * 0.10
    )
    running = analyze_running_style(tmp)

    lines = []
    used_horses = set()

    def horse_no(row):
        return int(row.get("馬番", 0) or 0)

    def unused(pool):
        return pool[~pool["馬番"].astype(int).isin(used_horses)].copy()

    def mark_used(rows):
        for _, row in rows.iterrows():
            used_horses.add(horse_no(row))

    def material_text(row):
        materials = []
        comment_parts = [part for part in str(row.get("コメント", "")).split("、") if part]
        for part in comment_parts:
            if part not in materials:
                materials.append(part)

        reason_text = str(row.get("_候補理由", ""))
        age = extract_age_from_sex_age(row.get("性齢"))
        reason_map = {
            "同条件": "同条件実績",
            "準適性": "他場実績注意",
            "対戦": "対戦材料",
            "展開": "展開利",
            "妙味穴": "人気薄",
            "3-4歳上積み": f"{age}歳上積み" if age in (3, 4) else "若駒上積み",
        }
        for reason in reason_text.split("、"):
            phrase = reason_map.get(reason)
            if phrase and phrase not in materials:
                materials.append(phrase)
        return "、".join(materials[:3]) or "指数上位"

    def index_material_text(row):
        materials = []
        row_course = safe_num(row.get("コース指数"), None)
        row_star = safe_num(row.get("★最高"), None)
        row_recent = safe_num(row.get("近3走最高"), None)
        if row_course is not None:
            materials.append(f"コース指数{compact_number(row_course)}")
        if row_star is not None:
            materials.append(f"★最高{compact_number(row_star)}")
        if row_recent is not None:
            materials.append(f"近3走最高{compact_number(row_recent)}")
        return "・".join(materials[:3]) or "今回条件の指数上位"

    def value_form_text(row):
        values = row.get("_prev_values")
        valid_values = [safe_num(value, None) for value in values] if isinstance(values, list) else []
        valid_values = [value for value in valid_values if value is not None]
        row_latest = safe_num(row.get("_last"), None)
        row_trend = safe_num(row.get("_trend"), 0)
        parts = ["前走同条件"]
        if len(valid_values) == 3 and valid_values[0] < valid_values[1] < valid_values[2]:
            parts.append("→".join(compact_number(value) for value in valid_values) + "と上昇")
        elif row_latest is not None and row_trend >= 5:
            parts.append(f"前走{compact_number(row_latest)}へ反発")
        elif row_latest is not None:
            parts.append(f"前走{compact_number(row_latest)}")
        row_course = safe_num(row.get("コース指数"), None)
        if row_course is not None:
            parts.append(f"コース指数{compact_number(row_course)}")
        return "・".join(parts[:3])

    def pace_horse_label(row):
        label = horse_label(row)
        style = normalize_running_style(row.get("脚質"))
        rate = format_percent_value(row.get("脚質勝率"))
        details = []
        if style:
            details.append(style)
        row_relative_weight = safe_num(row.get("_relative_load_weight"), 0)
        if extract_age_from_sex_age(row.get("性齢")) == 3 and row_relative_weight >= 2:
            details.append("軽量3歳")
        elif row_relative_weight >= 2.5:
            details.append("軽斤量")
        if rate:
            details.append(f"脚質勝率{rate}")
        return f"{label}({'・'.join(details)})" if details else label

    def ability_caution_text(row):
        cautions = []
        comment = str(row.get("コメント", ""))
        caution_map = {
            "斤量増注意": "斤量増",
            "休み明け注意": "休み明け",
            "間隔空き": "間隔空き",
            "乗替注": "乗り替わり",
        }
        for keyword, phrase in caution_map.items():
            if keyword in comment and phrase not in cautions:
                cautions.append(phrase)

        counts = running.get("脚質構成", {}) or {}
        early_count = counts.get("逃", 0) + counts.get("先", 0)
        style = normalize_running_style(row.get("脚質"))
        favored = set(running.get("有利脚質", []))
        tendency = str(running.get("展開傾向", ""))
        race_distance = safe_num(running.get("距離"), 9999)
        if style in ("逃", "先") and early_count >= 4:
            cautions.append("同型次第")
        elif style in ("差", "追") and ("前残り" in tendency or race_distance <= 1000):
            cautions.append("展開待ち")
        elif style and favored and style not in favored:
            cautions.append("展開待ち")

        unique = []
        for item in cautions:
            if item not in unique:
                unique.append(item)
        return "、".join(unique[:2])

    ability_pool = tmp[tmp["_候補理由"].str.contains("能力上位|同条件", na=False)].copy()
    if ability_pool.empty:
        ability_pool = tmp.sort_values(["_insight_course_score", "AI点"], ascending=[False, False]).head(1)
    ability_row = (
        ability_pool.sort_values(["_insight_course_score", "AI点"], ascending=[False, False]).iloc[0]
        if not ability_pool.empty
        else None
    )

    if running.get("available"):
        counts = running.get("脚質構成", {})
        escape_count = counts.get("逃", 0)
        leader_count = counts.get("先", 0)
        early_count = escape_count + leader_count
        race_distance = running.get("距離")

        if race_distance is not None and race_distance <= 1000:
            front_pool = unused(tmp[styles.isin(["逃", "先"])]).sort_values(
                ["_insight_pace_score", "推奨点", "AI点"],
                ascending=[False, False, False],
            ).head(2)
            if front_pool.empty:
                front_pool = tmp[styles.isin(["逃", "先"])].sort_values(
                    ["_insight_pace_score", "推奨点", "AI点"],
                    ascending=[False, False, False],
                ).head(2)
            names = "、".join(pace_horse_label(row) for _, row in front_pool.iterrows())
            line = f"{race_distance}mの短距離だけに、多少前が重なっても逃げ・先行力を重視"
            if names:
                line += f"。前では{names}が展開の中心"
                mark_used(front_pool)

            lightweight = unused(tmp[relative_weight >= 2]).sort_values(
                ["_insight_pace_score", "推奨点", "AI点"],
                ascending=[False, False, False],
            ).head(1)
            if not lightweight.empty:
                row = lightweight.iloc[0]
                label = "軽量3歳" if extract_age_from_sex_age(row.get("性齢")) == 3 else "軽斤量"
                line += f"。{label}の{pace_horse_label(row)}もスタートを決めれば注意"
                used_horses.add(horse_no(row))
            lines.append("展開：" + line + "。")
        elif race_distance is not None and race_distance <= 1200 and early_count >= 4:
            front_pool = unused(tmp[styles.isin(["逃", "先"])]).sort_values(
                ["_insight_pace_score", "推奨点", "AI点"],
                ascending=[False, False, False],
            ).head(1)
            closer_pool = unused(tmp[styles.isin(["差", "追"])]).sort_values(
                ["_insight_pace_score", "推奨点", "AI点"],
                ascending=[False, False, False],
            ).head(1)
            parts = [f"{race_distance}mの短距離で先行力は重要だが、逃げ・先行が計{early_count}頭なら競り合いにも注意"]
            if not front_pool.empty:
                row = front_pool.iloc[0]
                parts.append(f"前では{pace_horse_label(row)}")
                used_horses.add(horse_no(row))
            if not closer_pool.empty:
                row = closer_pool.iloc[0]
                parts.append(f"差しなら{pace_horse_label(row)}の浮上を警戒")
                used_horses.add(horse_no(row))
            lines.append("展開：" + "。".join(parts) + "。")
        elif race_distance is not None and race_distance <= 1200:
            front_pool = unused(tmp[styles.isin(["逃", "先"])]).sort_values(
                ["_insight_pace_score", "推奨点", "AI点"],
                ascending=[False, False, False],
            ).head(2)
            if not front_pool.empty:
                names = "、".join(pace_horse_label(row) for _, row in front_pool.iterrows())
                lines.append(f"展開：{race_distance}mの短距離で前有利を想定。{names}の先行力を重視。")
                mark_used(front_pool)
        elif early_count >= 4:
            pace_parts = [f"逃げ・先行が計{early_count}頭と多く、流れは速くなって差し浮上を想定"]
            closers = unused(tmp[styles.isin(["差", "追"])]).sort_values(
                ["_insight_pace_score", "推奨点", "AI点"],
                ascending=[False, False, False],
            ).head(2)
            if not closers.empty:
                names = "、".join(pace_horse_label(row) for _, row in closers.iterrows())
                pace_parts.append(f"差し馬では{names}に警戒")
                mark_used(closers)

            lightweight = unused(tmp[relative_weight >= 2]).sort_values(
                ["_insight_pace_score", "推奨点", "AI点"],
                ascending=[False, False, False],
            ).head(1)
            if not lightweight.empty:
                row = lightweight.iloc[0]
                label = "軽量3歳" if extract_age_from_sex_age(row.get("性齢")) == 3 else "軽斤量"
                pace_parts.append(f"{label}の{pace_horse_label(row)}も流れに乗れば注意")
                used_horses.add(horse_no(row))
            lines.append("展開：" + "。".join(pace_parts) + "。")
        elif escape_count == 1 and early_count <= 2:
            escape_pool = tmp[styles.eq("逃")].sort_values(["_focus_score", "推奨点"], ascending=[False, False])
            row = escape_pool.iloc[0]
            lines.append(f"展開：前は手薄で{pace_horse_label(row)}の単騎逃げが焦点。楽に運べれば前残りまで。")
            used_horses.add(horse_no(row))
        elif early_count <= 1:
            front_pool = tmp[styles.isin(["逃", "先"])].sort_values(["_focus_score", "推奨点"], ascending=[False, False]).head(2)
            if not front_pool.empty:
                names = "、".join(pace_horse_label(row) for _, row in front_pool.iterrows())
                lines.append(f"展開：前へ行く馬が少なくスロー寄り。{names}の前残りを警戒。")
                mark_used(front_pool)
        else:
            pace_pool = unused(tmp[styles.isin(["先", "差"])]).sort_values(
                ["_focus_score", "推奨点", "AI点"],
                ascending=[False, False, False],
            ).head(2)
            if not pace_pool.empty:
                names = "、".join(pace_horse_label(row) for _, row in pace_pool.iterrows())
                lines.append(f"展開：平均ペースなら先行と差しは互角。展開面では{names}を相手候補に。")
                mark_used(pace_pool)

    course_mask = star_high.notna()
    course_pool = unused(tmp[course_mask])
    if not course_pool.empty:
        selected = course_pool.sort_values(
            ["_insight_course_score", "★最高", "コース指数", "近3走最高"],
            ascending=[False, False, False, False],
        ).head(2)
        descriptions = [f"{horse_label(row)}は{index_material_text(row)}" for _, row in selected.iterrows()]
        lines.append(f"今回条件：{'、'.join(descriptions)}。今回会場の実績と指数を優先。")
        mark_used(selected)

    course_floor = course_index.median()
    form_mask = (
        (popularity.fillna(0) >= 7)
        & star_high.notna()
        & last_same_condition
        & (
            (trend.fillna(0) >= 5)
            | (latest.notna() & course_index.notna() & (latest >= course_index - 3))
        )
    )
    if pd.notna(course_floor):
        form_mask = form_mask & (course_index >= course_floor - 3)
    form_pool = unused(tmp[form_mask]).copy()
    if not form_pool.empty:
        form_pool["_insight_value_form_score"] = (
            trend.loc[form_pool.index].fillna(0).clip(-20, 30) * 0.8
            + latest.loc[form_pool.index].fillna(0) * 0.35
            + course_index.loc[form_pool.index].fillna(0) * 0.25
            + value_score.loc[form_pool.index].fillna(0) * 0.25
        )
        selected = form_pool.sort_values(
            ["_insight_value_form_score", "人気"],
            ascending=[False, False],
        ).head(2)
        descriptions = [f"{horse_label(row)}は{value_form_text(row)}" for _, row in selected.iterrows()]
        lines.append(f"穴の形：{'、'.join(descriptions)}。人気薄でも同条件の再現に注意。")
        mark_used(selected)

    value_pool = unused(tmp[
        (popularity.fillna(0) >= 5)
        & tmp["_候補理由"].str.contains("妙味穴|展開|同条件|対戦", na=False)
    ])
    if not value_pool.empty:
        selected = value_pool.assign(_value_score=value_score.loc[value_pool.index]).sort_values(
            ["_value_score", "_insight_course_score", "AI点"],
            ascending=[False, False, False],
        ).head(2)
        descriptions = [f"{horse_label(row)}は{material_text(row)}" for _, row in selected.iterrows()]
        lines.append(f"妙味注目：{'、'.join(descriptions)}。人気より条件・展開・対戦材料を重視。")
        mark_used(selected)

    young_pool = unused(tmp[
        (tmp.get("候補", "") == "✓")
        & ages.isin([3, 4])
        & (popularity.fillna(99) <= 5)
    ])
    if not young_pool.empty:
        selected = young_pool.sort_values(["人気", "推奨点"], ascending=[True, False]).head(2)
        descriptions = [f"{horse_label(row)}は{material_text(row)}" for _, row in selected.iterrows()]
        lines.append(f"上積み注目：{'、'.join(descriptions)}。成長力込みで見落とし注意。")
        mark_used(selected)

    similar_threshold = recent_high.quantile(0.75)
    similar_pool = tmp[
        similar_high.notna()
        & star_high.isna()
        & (similar_high >= similar_threshold if pd.notna(similar_threshold) else True)
    ].copy()
    other_venue_line = ""
    if not similar_pool.empty:
        row = similar_pool.sort_values(["_similar_condition_high", "近3走最高"], ascending=[False, False]).iloc[0]
        other_venue_line = (
            f"他場注意：{horse_label(row)}は他会場の同条件で指数"
            f"{compact_number(row.get('_similar_condition_high'))}。能力は警戒するが、"
            "今回会場の裏付けが薄く押さえまで。"
        )

    h2h_pool = unused(tmp[
        (tmp.get("候補", "") == "✓")
        & tmp["_候補理由"].str.contains("対戦", na=False)
    ])
    if not h2h_pool.empty:
        row = h2h_pool.sort_values(["推奨点", "AI点"], ascending=[False, False]).iloc[0]
        detail = row.get("対戦", "")
        if detail:
            lines.append(f"対戦注目：{horse_label(row)}に材料。{detail}は確認しておきたい。")
        else:
            lines.append(f"対戦注目：{horse_label(row)}に材料あり。")
        used_horses.add(horse_no(row))

    ability_line = ""
    if ability_row is not None and horse_no(ability_row) not in used_horses:
        caution = ability_caution_text(ability_row)
        if caution:
            ability_line = f"能力面では{horse_label(ability_row)}が上位。ただし{caution}で、軸固定より相手評価。"
        else:
            ability_line = f"能力面では{horse_label(ability_row)}が上位。大きな割引材料は少ないが、展開とオッズを見て最終判断。"

    if not lines:
        lines.append("指数差が小さく、オッズ・条件・展開を見比べて絞るレース。")

    print("【レース考察】")
    output_lines = lines[:3]
    if other_venue_line:
        output_lines.append(other_venue_line)
    elif len(lines) > 3:
        output_lines.append(lines[3])
    if ability_line:
        output_lines.append(ability_line)
    no_honmei_market_gap = (
        "最終印" in tmp.columns
        and not tmp["最終印"].fillna("").astype(str).eq("◎").any()
        and tmp.get("印理由", pd.Series("", index=tmp.index)).fillna("").astype(str).str.contains("市場評価差で○", na=False).any()
    )
    if no_honmei_market_gap:
        output_lines.insert(0, "本命不在（混戦評価）：市場との評価乖離が大きく、軸固定は慎重。")
    for line in output_lines[:5]:
        print(line)


def print_corner_scenario(df):
    tmp = df.copy()
    tmp["_脚質正規化"] = tmp["脚質"].map(normalize_running_style) if "脚質" in tmp.columns else ""
    ai = pd.to_numeric(tmp.get("AI点"), errors="coerce")
    tmp["_scenario_ai"] = ai

    def horse_list(mask):
        horses = (
            tmp[mask]
            .sort_values(["_scenario_ai", "推奨点"], ascending=[False, False])
            ["馬番"]
            .astype(int)
            .tolist()
        )
        return format_horses(horses)

    front_mask = tmp["_脚質正規化"].isin(["逃", "先"])
    middle_mask = tmp["_脚質正規化"].eq("差") | tmp["_脚質正規化"].eq("")
    back_mask = tmp["_脚質正規化"].eq("追")

    print("【4コーナー展開予想】")
    print(f"先団：{horse_list(front_mask)}")
    print(f"中団：{horse_list(middle_mask)}")
    print(f"後方：{horse_list(back_mask)}")

    front_count = int(front_mask.sum())
    escape_count = int((tmp["_脚質正規化"] == "逃").sum())
    leader_count = int((tmp["_脚質正規化"] == "先").sum())
    early_compete = escape_count >= 2 or leader_count >= 2 or front_count >= 3
    if early_compete:
        print(f"隊列メモ：先行重なり。先団{horse_list(front_mask)}は同型次第")
    elif escape_count == 1:
        escape = horse_list(tmp["_脚質正規化"].eq("逃"))
        print(f"隊列メモ：{escape}の単騎逃げに注意")

    focus_mask = tmp["直線評価"].isin(["勝ち負け", "押切候補", "差切警戒", "末脚勝負", "差し浮上"])
    focus = (
        tmp[focus_mask]
        .sort_values(["AI点", "推奨点"], ascending=[False, False])
        .head(4)
    )
    if not focus.empty:
        notes = [f"{circled_number(row['馬番'])}{row['直線評価']}" for _, row in focus.iterrows()]
        print(f"直線注目：{' / '.join(notes)}")


def format_number_for_display(value):
    if pd.isna(value):
        return ""
    try:
        number = float(value)
    except Exception:
        return value
    return f"{number:.1f}".rstrip("0").rstrip(".")


DISPLAY_NUMBER_COLUMNS = [
    "斤量",
    "単勝オッズ",
    "オッズ",
    "距離指数",
    "コース指数",
    "3走平均",
    "平均指数",
    "★最高",
    "★最高指数",
    "近3走最高",
    "最高指数",
    "AI点",
    "補正AI点",
    "総合評価点",
    "総合評価",
    "推定勝率",
    "市場反映勝率",
    "勝率順位",
    "適正オッズ",
    "単勝期待値",
    "AI順位",
    "人気",
    "人気差",
]


ROLE_DISPLAY_MAP = {
    "本軸": "◎",
    "相手有力": "○",
    "穴軸": "★",
    "妙味あり": "★",
    "押さえ": "△",
    "消し寄り": "",
}


def format_result_for_output(df):
    formatted = df.copy()
    if "役割" in formatted.columns:
        formatted["役割"] = formatted["役割"].map(ROLE_DISPLAY_MAP).fillna("")
    for col in DISPLAY_NUMBER_COLUMNS:
        if col in formatted.columns:
            formatted[col] = formatted[col].map(format_number_for_display)
    if "年齢補正" in formatted.columns:
        formatted["年齢補正"] = formatted["年齢補正"].map(
            lambda value: (
                ""
                if pd.isna(value)
                else f"+{format_number_for_display(value)}"
                if float(value) > 0
                else format_number_for_display(value)
            )
        )
    return formatted


def _display_float_or_none(value):
    try:
        if pd.isna(value):
            return None
        return float(value)
    except Exception:
        return None


def _add_material_unique(parts, label):
    if label and label not in parts:
        parts.append(label)


def build_nar_evaluation_material(row):
    parts = []
    average = _display_float_or_none(row.get("平均指数"))
    star_high = _display_float_or_none(row.get("★最高指数") or row.get("最高指数"))
    distance = _display_float_or_none(row.get("距離指数"))
    course = _display_float_or_none(row.get("コース指数"))
    style = normalize_running_style(row.get("脚質", ""))
    class_shift = str(row.get("クラス変動") or "")
    ability_text = str(row.get("能力") or "")
    aptitude_text = str(row.get("適性") or "")
    same_going_text = str(row.get("同馬場実績") or row.get("馬場適性") or "")
    reason_text = str(row.get("印理由") or "")
    h2h_label = str(row.get("_h2h_label") or "")
    h2h_latest = str(row.get("_h2h_latest") or row.get("対戦") or "")
    h2h_score = _display_float_or_none(row.get("_h2h_score")) or 0
    condition_axis = _display_float_or_none(row.get("_条件軸点")) or 0

    if condition_axis >= 5:
        _add_material_unique(parts, "条件軸")
    elif condition_axis >= 3:
        _add_material_unique(parts, "条件合う")

    if h2h_label == "対戦◎" or h2h_score >= 2:
        _add_material_unique(parts, "対戦◎")
    elif h2h_score > 0 or h2h_label == "対戦○" or "先着" in h2h_latest:
        _add_material_unique(parts, "対戦先着")
    elif h2h_score < 0 or h2h_label == "対戦△" or "敗戦" in h2h_latest:
        _add_material_unique(parts, "対戦劣勢")

    if bool(row.get("_course_hole", False)):
        _add_material_unique(parts, "コース穴")
    if class_shift == "クラス降級":
        _add_material_unique(parts, "クラス降級")
    if average is not None and average >= 100:
        _add_material_unique(parts, "高指数")
    if star_high is not None and star_high >= 100:
        _add_material_unique(parts, "最高指数")
    if course is not None:
        _add_material_unique(parts, "コース実績")
    if distance is not None:
        _add_material_unique(parts, "距離実績")
    if average is not None and course is not None and average >= 100 and course >= 95:
        _add_material_unique(parts, "平均×コース")
    if style in ("逃", "先") and average is not None and course is not None and average >= 95 and course >= 95:
        _add_material_unique(parts, "前受け実績")
    if "上位" in ability_text:
        _add_material_unique(parts, "能力上位")
    if "上位" in aptitude_text or "実績" in aptitude_text:
        _add_material_unique(parts, "適性あり")
    if same_going_text and "未知" not in same_going_text:
        _add_material_unique(parts, "同馬場実績")
    if "展開材料" in reason_text or "単騎逃げ" in reason_text:
        _add_material_unique(parts, "展開向く")

    if not parts:
        for piece in reason_text.split(" / "):
            piece = piece.strip()
            if piece:
                _add_material_unique(parts, piece)
            if len(parts) >= 4:
                break
    return " / ".join(parts[:5])


def prepare_nar_display_columns(df):
    prepared = df.copy()
    if "馬年齢" not in prepared.columns and "性齢" in prepared.columns:
        prepared["馬年齢"] = prepared["性齢"].map(lambda value: "" if pd.isna(value) else str(value).strip())
    if "騎手" not in prepared.columns:
        prepared["騎手"] = "―"
    else:
        jockey = prepared["騎手"].map(lambda value: "" if pd.isna(value) else str(value).strip())
        prepared["騎手"] = jockey.mask(jockey.eq(""), "―")
    if "オッズ" not in prepared.columns and "単勝オッズ" in prepared.columns:
        prepared["オッズ"] = pd.to_numeric(prepared["単勝オッズ"], errors="coerce")
    if "総合評価" not in prepared.columns and "総合評価点" in prepared.columns:
        prepared["総合評価"] = pd.to_numeric(prepared["総合評価点"], errors="coerce")
    if "市場反映勝率" not in prepared.columns and "推定勝率" in prepared.columns:
        prepared["市場反映勝率"] = pd.to_numeric(prepared["推定勝率"], errors="coerce")

    def resolve_interval(row):
        raw_interval = str(row.get("間隔") or "").strip()
        formatted = format_interval_from_days(row.get("_days_since_last"))
        if formatted and formatted != "-":
            return formatted
        if raw_interval and raw_interval not in ("nan", "None", "中0週"):
            return raw_interval
        return "-"

    prepared["レース間隔"] = prepared.apply(resolve_interval, axis=1)
    prepared["間隔"] = prepared["レース間隔"]
    if "評価/検討材料" not in prepared.columns:
        prepared["評価/検討材料"] = prepared.apply(build_nar_evaluation_material, axis=1)
    if "_地方指数データ不足" in prepared.columns:
        shortage = prepared["_地方指数データ不足"].fillna(False).astype(bool)
        for column in ["AI点", "総合評価", "総合評価点", "補正AI点", "市場反映勝率", "推定勝率", "単勝期待値", "勝率順位"]:
            if column in prepared.columns:
                prepared[column] = prepared[column].astype("object")
                prepared.loc[shortage, column] = "データ不足"
        if "評価/検討材料" in prepared.columns:
            current = prepared.loc[shortage, "評価/検討材料"].fillna("").astype(str)
            prepared.loc[shortage, "評価/検討材料"] = current.mask(
                current.eq(""),
                "データ不足",
            ).where(
                current.str.contains("データ不足", na=False),
                (current + " / データ不足").str.strip(" /"),
            )
    return prepared

def make_result_cell_styles(df):
    styles = pd.DataFrame("", index=df.index, columns=df.columns)

    if "候補" in df.columns:
        styles.loc[df["候補"].astype(str).eq("✓"), "候補"] = (
            "font-weight: 800; color: #047857; background-color: #ecfdf5;"
        )

    if "単勝オッズ" in df.columns:
        odds = pd.to_numeric(df["単勝オッズ"], errors="coerce")
        for idx, value in odds.items():
            if pd.isna(value):
                continue
            if value <= 5:
                css = "font-weight: 800; color: #1d4ed8; background-color: #eff6ff;"
            elif value <= 15:
                css = "font-weight: 800; color: #047857; background-color: #ecfdf5;"
            elif value <= 50:
                css = "font-weight: 800; color: #c2410c; background-color: #fff7ed;"
            else:
                css = "font-weight: 800; color: #b91c1c; background-color: #fef2f2;"
            styles.at[idx, "単勝オッズ"] = css

    star_column = "★最高指数" if "★最高指数" in df.columns else "★最高"
    if star_column in df.columns:
        star_high = pd.to_numeric(df[star_column], errors="coerce")
        top_star = star_high.max()
        if pd.notna(top_star):
            for idx, value in star_high.items():
                if pd.isna(value):
                    continue
                if value >= top_star:
                    styles.at[idx, star_column] = "font-weight: 800; color: #92400e; background-color: #fef3c7;"
                elif value >= top_star - 5:
                    styles.at[idx, star_column] = "font-weight: 700; color: #a16207; background-color: #fffbeb;"

    if "年齢補正" in df.columns:
        age_adjustment = pd.to_numeric(df["年齢補正"], errors="coerce")
        styles.loc[age_adjustment > 0, "年齢補正"] = "font-weight: 700; color: #047857;"
        styles.loc[age_adjustment < 0, "年齢補正"] = "font-weight: 700; color: #b91c1c;"

    if "最終評価" in df.columns:
        final_score = pd.to_numeric(df["最終評価"], errors="coerce")
        top_final = final_score.max()
        if pd.notna(top_final):
            styles.loc[final_score >= top_final - 3, "最終評価"] = (
                "font-weight: 800; color: #7f1d1d; background-color: #fee2e2;"
            )

    if "最終印" in df.columns:
        mark_styles = {
            "◎": "font-weight: 900; color: #7f1d1d; background-color: #fee2e2;",
            "○": "font-weight: 900; color: #1d4ed8; background-color: #eff6ff;",
            "▲": "font-weight: 900; color: #92400e; background-color: #fef3c7;",
            "△": "font-weight: 800; color: #374151; background-color: #f3f4f6;",
            "✓": "font-weight: 900; color: #0f766e; background-color: #ccfbf1;",
        }
        for mark, css in mark_styles.items():
            styles.loc[df["最終印"].astype(str).eq(mark), "最終印"] = css

    if "クラス変動" in df.columns:
        styles.loc[df["クラス変動"].astype(str).eq("クラス降級"), "クラス変動"] = "font-weight: 800; color: #047857; background-color: #ecfdf5;"
        styles.loc[df["クラス変動"].astype(str).eq("クラス昇級"), "クラス変動"] = "font-weight: 800; color: #b91c1c; background-color: #fef2f2;"

    return styles


def result_display_styler(df):
    raw = df.copy()
    return format_result_for_output(raw).style.hide(axis="index").apply(lambda _: make_result_cell_styles(raw), axis=None)


def format_result_for_export(df):
    return format_result_for_output(df)


BETTING_OUTPUT_COLUMNS = [
    "推奨券種",
    "券種理由",
    "買い候補",
    "買い目メモ",
    "印",
    "役割",
    "推奨順位",
    "勝負タイプ",
    "評価",
    "総合評価",
    "総合理由",
    "実戦評価",
    "馬券材料",
    "候補",
    "向く買い方",
    "中心評価",
    "中心理由",
    "再現性",
    "展開依存",
    "妙味スコア",
    "妙味",
    "危険度",
    "最終評価",
    "推奨点",
]


def remove_betting_output_columns(df):
    return df.drop(columns=[c for c in BETTING_OUTPUT_COLUMNS if c in df.columns], errors="ignore")


FINAL_MARK_SEQUENCE = ["◎", "○", "▲", "△", "△", "✓"]
FINAL_MARK_LABELS = {"◎": "本命", "○": "対抗", "▲": "単穴", "△": "抑え", "✓": "穴候補"}


def final_mark_class_basis(row):
    current_label = str(row.get("_current_class_label") or "").strip()
    previous_label = str(row.get("_previous_class_label") or "").strip()
    best_label = str(row.get("_best_past_class_label") or "").strip()
    shift = str(row.get("クラス変動") or row.get("_class_shift") or "").strip()
    past_runs = row.get("_past_runs") or []
    parts = []
    parts.append(f"今回{current_label}" if current_label else "今回クラス不明")

    previous_run = next((run for run in past_runs if run.get("label") == "前走"), {})
    if previous_label:
        parts.append(f"前走{previous_label}")
    elif previous_run.get("url"):
        parts.append("前走クラス未取得")

    order = {"前走": 0, "2走前": 1, "3走前": 2}
    past_class_parts = []
    missing_detail = False
    for run in sorted(past_runs, key=lambda item: order.get(item.get("label"), 99)):
        run_label = str(run.get("label") or "").strip()
        class_label = str(run.get("class_label") or "").strip()
        if run_label and class_label:
            past_class_parts.append(f"{run_label}{class_label}")
        elif run_label and run.get("url"):
            missing_detail = True
    if past_class_parts:
        parts.append("近3走クラス:" + "・".join(past_class_parts))
    elif missing_detail:
        parts.append("近3走クラス未取得")

    if best_label and best_label != previous_label:
        parts.append(f"近3走最高クラス{best_label}")
    if shift:
        parts.append(shift)
    return " / ".join(parts)


def final_mark_class_score(row):
    current_rank = safe_num(row.get("_current_class_rank"), None)
    previous_rank = safe_num(row.get("_previous_class_rank"), None)
    best_rank = safe_num(row.get("_best_past_class_rank"), previous_rank)
    best_label = str(row.get("_best_past_class_label") or row.get("_previous_class_label") or "").strip()
    shift = str(row.get("クラス変動") or row.get("_class_shift") or "")
    score = 0.0
    reasons = []

    if shift in ("クラス降級", "相手弱化"):
        score += 4.0
        reasons.append("クラス降級")
    elif shift in ("クラス昇級", "相手強化"):
        score -= 4.0
        reasons.append("クラス昇級で慎重")
    elif shift in ("同級", "同級近辺"):
        score += 1.0
        reasons.append("同級安定")

    if current_rank is not None and best_rank is not None and best_rank - current_rank >= 8:
        score += 3.0
        if best_label:
            reasons.append(f"{best_label}実績")
        else:
            reasons.append("上級実績")
    if best_rank is not None and best_rank >= 68:
        score += 1.8
        reasons.append("重賞級経験")
    if current_rank is None and previous_rank is None:
        score -= 0.5
    return score, reasons


def add_final_marks_v1_legacy(df, running_info=None):
    """Legacy Ver1.0 final-mark scorer kept for notebook parity checks.

    This function is not used by the normal Keiba AI Mobile prediction path.
    The active add_final_marks() keeps AI点 as the ability score and adds only
    class/state/pace/matchup corrections, while this legacy scorer also mixes
    AI rank, index ranks, market value, venue, condition-axis, and hole signals
    into the final mark score.
    """
    df = df.copy()
    df = df.drop(columns=["最終印", "展開印", "印理由", "クラス根拠", "_最終印点", "_最終印順", "_穴評価点", "_条件軸点", "補正AI点", "_course_hole"], errors="ignore")
    if df.empty:
        df["最終印"] = ""
        df["展開印"] = ""
        df["印理由"] = ""
        df["クラス根拠"] = ""
        df["補正AI点"] = ""
        df["_course_hole"] = False
        df["_条件軸点"] = ""
        return df

    def numeric(column, default=None):
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce")
        return pd.Series(default, index=df.index, dtype="float64")

    data_shortage = (
        df.get("_地方指数データ不足", _nar_local_index_data_shortage_mask(df))
        .fillna(False)
        .astype(bool)
    )
    df["_地方指数データ不足"] = data_shortage

    ai = numeric("AI点")
    if "AI順位" not in df.columns or pd.to_numeric(df.get("AI順位"), errors="coerce").isna().all():
        df["AI順位"] = ai.rank(method="min", ascending=False).astype("Int64")
    ai_rank = pd.to_numeric(df.get("AI順位"), errors="coerce")
    popularity = numeric("人気")
    odds = numeric("単勝オッズ")
    average = numeric("平均指数").fillna(numeric("3走平均"))
    recent_high = numeric("最高指数").fillna(numeric("近3走最高"))
    distance = numeric("距離指数")
    course = numeric("コース指数")
    star_high = numeric("★最高指数").fillna(numeric("★最高"))
    support_count = numeric("_指数裏付け数", 0).fillna(0)
    venue_score = numeric("_会場評価点").fillna(numeric("_JRA会場評価点", 0)).fillna(0)
    trend = numeric("_trend", 0).fillna(0)
    h2h_score = numeric("_h2h_score", 0).fillna(0)
    layoff = df.get("_is_layoff", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    styles = df.get("脚質", pd.Series("", index=df.index)).map(normalize_running_style)
    running = running_info or analyze_running_style(df)
    favored_styles = set(running.get("有利脚質", []))
    counts = running.get("脚質構成", {}) or {}
    pace_horses = set(
        int(value)
        for value in (running.get("展開向く馬番") or running.get("展開穴", []))
        if pd.notna(value)
    )

    average_rank = average.rank(method="min", ascending=False)
    recent_rank = recent_high.rank(method="min", ascending=False)
    distance_rank = distance.rank(method="min", ascending=False)
    course_rank = course.rank(method="min", ascending=False)
    star_rank = star_high.rank(method="min", ascending=False)
    top_ai = ai.max()
    field_average_value = safe_num(average.mean(), None)

    scores = []
    reason_texts = []
    class_basis_texts = []
    hole_scores = []
    adjusted_ai_scores = []
    course_hole_flags = []
    condition_axis_scores = []

    def add_unique(parts, text):
        if text and text not in parts:
            parts.append(text)

    for idx, row in df.iterrows():
        base_ai_value = float(ai.loc[idx]) if pd.notna(ai.loc[idx]) else 0.0
        score = base_ai_value
        condition_adjustment = 0.0
        reasons = []
        rank_value = safe_num(ai_rank.loc[idx], len(df))
        pop_value = safe_num(popularity.loc[idx], None)
        odds_value = safe_num(odds.loc[idx], None)
        class_score, class_reasons = final_mark_class_score(row)
        score += class_score
        for reason in class_reasons:
            add_unique(reasons, reason)

        if rank_value <= 1:
            score += 7.0
            add_unique(reasons, "AI順位最上位")
        elif rank_value <= 3:
            score += 5.0
            add_unique(reasons, "AI順位上位")
        elif rank_value <= 6:
            score += 2.0

        for rank_series, label, first_bonus, top3_bonus in [
            (average_rank, "平均指数上位", 2.5, 1.5),
            (recent_rank, "最高指数上位", 2.5, 1.5),
            (distance_rank, "距離指数上位", 1.5, 0.8),
            (course_rank, "コース指数上位", 1.8, 1.0),
            (star_rank, "同条件実績", 1.8, 1.0),
        ]:
            rank_item = rank_series.loc[idx]
            if pd.notna(rank_item) and rank_item <= 1:
                score += first_bonus
                add_unique(reasons, label)
            elif pd.notna(rank_item) and rank_item <= 3:
                score += top3_bonus
                add_unique(reasons, label)

        # 地方は「近3走の★最高指数」と今回の距離/コース適性が揃う馬を軸候補として強めに扱う。
        star_rank_value = safe_num(star_rank.loc[idx], None)
        recent_rank_value = safe_num(recent_rank.loc[idx], None)
        distance_rank_value = safe_num(distance_rank.loc[idx], None)
        course_rank_value_for_axis = safe_num(course_rank.loc[idx], None)
        condition_top_count = int(distance_rank_value is not None and distance_rank_value <= 3) + int(
            course_rank_value_for_axis is not None and course_rank_value_for_axis <= 3
        )
        condition_axis_score = 0.0
        if star_rank_value is not None and star_rank_value <= 1 and condition_top_count >= 1:
            condition_axis_score += 5.5
            add_unique(reasons, "★最高×条件")
        elif star_rank_value is not None and star_rank_value <= 3 and condition_top_count >= 1:
            condition_axis_score += 4.0
            add_unique(reasons, "★上位×条件")
        elif recent_rank_value is not None and recent_rank_value <= 3 and condition_top_count >= 1:
            condition_axis_score += 2.5
            add_unique(reasons, "近3走最高×条件")
        if condition_top_count >= 2:
            condition_axis_score += 1.8
            add_unique(reasons, "距離コース両方")
        if condition_axis_score >= 5.0 and not layoff.loc[idx]:
            condition_axis_score += 1.0
            add_unique(reasons, "軸材料")
        elif condition_axis_score >= 5.0 and layoff.loc[idx]:
            condition_axis_score -= 1.5
            add_unique(reasons, "休み明け割引")
        score += condition_axis_score

        ability_text = str(row.get("能力") or "")
        aptitude_text = str(row.get("適性") or "")
        momentum_text = str(row.get("勢い") or "")
        same_going_text = str(row.get("同馬場実績") or row.get("馬場適性") or "")
        if "上位" in ability_text:
            score += 2.0
            add_unique(reasons, "能力上位")
        if "実績" in aptitude_text or "同条件" in aptitude_text:
            score += 1.5
            add_unique(reasons, "適性あり")
        if momentum_text in ("上昇", "良化", "安定"):
            score += 1.0
            add_unique(reasons, momentum_text)
        if same_going_text and "未知" not in same_going_text:
            score += 1.0
            add_unique(reasons, "同馬場実績")

        h2h_value = safe_num(h2h_score.loc[idx], 0) or 0
        h2h_label = str(row.get("_h2h_label") or "")
        h2h_latest = str(row.get("_h2h_latest") or row.get("対戦") or "")
        h2h_recent_win = "先着" in h2h_latest
        h2h_recent_loss = "敗戦" in h2h_latest
        h2h_positive = h2h_recent_win or h2h_value > 0 or h2h_label in ("対戦◎", "対戦○")
        h2h_strong = h2h_recent_win or h2h_value >= 2 or h2h_label == "対戦◎"
        h2h_negative = (not h2h_recent_win) and (h2h_value < 0 or h2h_label == "対戦△" or h2h_recent_loss)
        if h2h_strong:
            add_unique(reasons, "対戦◎" if h2h_label == "対戦◎" else "対戦先着")
        elif h2h_positive:
            add_unique(reasons, "対戦先着")
        elif h2h_negative:
            add_unique(reasons, "対戦劣勢")

        style = styles.loc[idx]
        average_value = safe_num(average.loc[idx], None)
        course_value = safe_num(course.loc[idx], None)
        distance_value = safe_num(distance.loc[idx], None)
        star_value = safe_num(star_high.loc[idx], None)
        course_rank_value = safe_num(course_rank.loc[idx], None)
        average_rank_value = safe_num(average_rank.loc[idx], None)
        average_low_for_course = (
            average_value is not None
            and (
                (field_average_value is not None and average_value <= field_average_value - 5)
                or (average_rank_value is not None and average_rank_value >= 6)
            )
        )
        course_hole = (
            course_rank_value is not None
            and course_rank_value <= 3
            and odds_value is not None
            and odds_value >= 8
            and star_value is None
            and average_low_for_course
        )
        if course_value is not None:
            score += 1.0
            condition_adjustment += 0.6
            add_unique(reasons, "コース実績")
        if distance_value is not None:
            condition_adjustment += 0.2
        if average_value is not None and average_value >= 100:
            score += 1.0
            condition_adjustment += 0.8
            add_unique(reasons, "高指数")
        if star_value is not None and star_value >= 100:
            condition_adjustment += 0.5
        if average_value is not None and course_value is not None and average_value >= 100 and course_value >= 95:
            score += 1.5
            condition_adjustment += 1.0
            add_unique(reasons, "平均×コース")
        if style in ("逃", "先") and average_value is not None and course_value is not None and average_value >= 95 and course_value >= 95:
            score += 1.2
            condition_adjustment += 0.8
            add_unique(reasons, "前受け実績")
        if style in ("逃", "先") and average_value is not None and course_value is not None and average_value >= 100 and course_value >= 100:
            score += 1.2
            condition_adjustment += 0.8
            add_unique(reasons, "軸向き")
        if course_hole:
            add_unique(reasons, "コース穴")

        horse_no = int(row.get("馬番", 0) or 0)
        pace_comment = str(row.get("展開コメント") or "")
        pace_focus = horse_no in pace_horses or "展開穴" in pace_comment or "単騎" in pace_comment
        if pace_focus:
            score += 2.6
            if style == "逃" and (counts.get("逃", 0) == 1 or "単騎" in pace_comment):
                add_unique(reasons, "単騎逃げ")
            add_unique(reasons, "展開向く")
        elif style in favored_styles:
            score += 0.5
            add_unique(reasons, "脚質合う")
        elif "展開合う" in pace_comment or "好位で粘る" in pace_comment:
            score += 1.0
            add_unique(reasons, "展開材料")
        if "展開待ち" in pace_comment or "課題" in pace_comment:
            score -= 0.8

        score += min(max(float(venue_score.loc[idx]) if pd.notna(venue_score.loc[idx]) else 0.0, -2.0), 8.0) * 0.35
        if venue_score.loc[idx] >= 4:
            add_unique(reasons, "会場材料")

        value_gap = 0.0
        if pop_value is not None and pd.notna(rank_value):
            value_gap = float(pop_value) - float(rank_value)
            if value_gap >= 3:
                score += min(value_gap, 8) * 0.55
                add_unique(reasons, "人気以上に評価")
            elif value_gap <= -3 and rank_value >= 4:
                score -= 1.3
                add_unique(reasons, "人気先行注意")
        if odds_value is not None and pd.notna(top_ai) and ai.loc[idx] >= top_ai - 15:
            if odds_value >= 15:
                score += 1.2
                add_unique(reasons, "オッズ妙味")
            elif odds_value >= 8:
                score += 0.7
                add_unique(reasons, "オッズ妙味")
            elif odds_value <= 2 and rank_value >= 4:
                score -= 1.0

        if trend.loc[idx] >= 5:
            score += 1.0
            add_unique(reasons, "近走上向き")
        elif trend.loc[idx] <= -8:
            score -= 1.2
        if layoff.loc[idx]:
            score -= 1.0
            condition_adjustment -= 1.4
            add_unique(reasons, "休み明け注意")

        class_basis = final_mark_class_basis(row)
        if not reasons:
            add_unique(reasons, "総合評価上位")
        scores.append(round(score, 2))
        reason_texts.append(" / ".join(reasons[:5]))
        class_basis_texts.append(class_basis)

        hole_score = score
        if value_gap >= 3:
            hole_score += min(value_gap, 8) * 0.8
        if odds_value is not None and odds_value >= 8:
            hole_score += 1.4
        if any(reason in reasons for reason in ("展開材料", "クラス降級", "人気以上に評価", "コース実績", "平均×コース", "軸向き", "前受け実績", "対戦◎", "対戦先着")):
            hole_score += 1.4
        if h2h_strong:
            hole_score += 5.0
        elif h2h_positive:
            hole_score += 3.5
        if course_hole:
            hole_score += 9.0
        adjusted_ai = min(max(base_ai_value + condition_adjustment + condition_axis_score * 0.35, 0.0), 100.0)
        adjusted_ai_scores.append(round(adjusted_ai, 1))
        condition_axis_scores.append(round(condition_axis_score, 1))
        course_hole_flags.append(bool(course_hole))
        hole_scores.append(round(hole_score, 2))

    df["_最終印点"] = scores
    df["_穴評価点"] = hole_scores
    df["_条件軸点"] = condition_axis_scores
    df["補正AI点"] = adjusted_ai_scores
    df["_course_hole"] = course_hole_flags
    df["印理由"] = reason_texts
    df["クラス根拠"] = class_basis_texts
    pace_reason = df["印理由"].astype(str)
    df["展開印"] = ""
    pace_numbers = set(
        int(value)
        for value in (running.get("展開向く馬番") or running.get("展開穴", []))
        if pd.notna(value)
    )
    if pace_numbers:
        pace_candidates = df[df["馬番"].astype(int).isin(pace_numbers)].copy()
    else:
        pace_candidates = df[pace_reason.str.contains("展開向く", na=False)].copy()
    if not pace_candidates.empty:
        pace_candidates["_展開印候補点"] = pd.to_numeric(
            pace_candidates.get("_最終印点"), errors="coerce"
        ).fillna(0)
        pace_candidates["_展開印AI点"] = pd.to_numeric(
            pace_candidates.get("AI点"), errors="coerce"
        ).fillna(0)
        pace_idx = pace_candidates.sort_values(
            ["_展開印候補点", "_展開印AI点"], ascending=[False, False]
        ).index[0]
        df.at[pace_idx, "展開印"] = "展"
    df["最終印"] = ""
    df["_最終印順"] = pd.NA

    ordered_indices = (
        df.sort_values(["_最終印点", "AI点"], ascending=[False, False])
        .index
        .tolist()
    )
    assigned = set()
    mark_order = 0
    for mark in ["◎", "○", "▲", "△", "△"]:
        if not ordered_indices:
            break
        idx = next((x for x in ordered_indices if x not in assigned), None)
        if idx is None:
            break
        df.at[idx, "最終印"] = mark
        df.at[idx, "_最終印順"] = mark_order
        assigned.add(idx)
        mark_order += 1

    remaining = [idx for idx in ordered_indices if idx not in assigned]
    star_idx = None

    def pool_numeric(frame, column, default=None):
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce")
        return pd.Series(default, index=frame.index, dtype="float64")

    def append_mark_reason(idx, reason):
        current = str(df.at[idx, "印理由"] or "")
        parts = [part for part in current.split(" / ") if part]
        if reason not in parts:
            parts.append(reason)
        df.at[idx, "印理由"] = " / ".join(parts[:6])

    if remaining:
        hole_pool = df.loc[remaining].copy()
        hole_ai_rank = pool_numeric(hole_pool, "AI順位")
        hole_ai = pool_numeric(hole_pool, "AI点")
        hole_popularity = pool_numeric(hole_pool, "人気")
        hole_odds = pool_numeric(hole_pool, "単勝オッズ")
        hole_weight_change = pool_numeric(hole_pool, "_load_weight_change")
        hole_best_class = pool_numeric(hole_pool, "_best_past_class_rank")
        hole_current_class = pool_numeric(hole_pool, "_current_class_rank")
        hole_age = hole_pool.get(
            "性齢", pd.Series("", index=hole_pool.index)
        ).map(extract_age_from_sex_age)
        hole_reason = hole_pool["印理由"].astype(str)
        hole_class_basis = hole_pool["クラス根拠"].astype(str)
        hole_shift = hole_pool.get(
            "クラス変動", pd.Series("", index=hole_pool.index)
        ).astype(str)
        hole_h2h_score = pool_numeric(hole_pool, "_h2h_score", 0).fillna(0)
        hole_h2h_label = hole_pool.get("_h2h_label", pd.Series("", index=hole_pool.index)).fillna("").astype(str)
        hole_h2h_latest = hole_pool.get(
            "_h2h_latest", hole_pool.get("対戦", pd.Series("", index=hole_pool.index))
        ).fillna("").astype(str)
        hole_course_rank = course_rank.reindex(hole_pool.index)
        hole_average_rank = average_rank.reindex(hole_pool.index)
        hole_average = pool_numeric(hole_pool, "平均指数").fillna(pool_numeric(hole_pool, "3走平均"))
        hole_star = pool_numeric(hole_pool, "★最高指数").fillna(pool_numeric(hole_pool, "★最高"))
        rank_gap = hole_popularity.fillna(99) - hole_ai_rank.fillna(len(df))

        class_down = (
            hole_reason.str.contains("クラス降級", na=False)
            | hole_shift.isin(["クラス降級", "相手弱化"])
        )
        upper_experience = (
            hole_class_basis.str.contains("G1|G2|G3|Jpn|重賞|OP|L", regex=True, na=False)
            | hole_reason.str.contains("重賞級経験|G1実績|G2実績|G3実績|Jpn", regex=True, na=False)
            | (
                hole_best_class.notna()
                & hole_current_class.notna()
                & (hole_best_class - hole_current_class >= 8)
            )
        )
        ai_window = hole_ai_rank.ge(4) & hole_ai_rank.le(10)
        ai_not_too_low = hole_ai.notna() & pd.notna(top_ai) & hole_ai.ge(top_ai - 18)
        value_signal = hole_odds.ge(8) | rank_gap.ge(2)
        h2h_recent_win_mask = hole_h2h_latest.str.contains("先着", na=False)
        h2h_recent_loss_mask = hole_h2h_latest.str.contains("敗戦", na=False)
        h2h_positive_mask = (
            h2h_recent_win_mask
            | hole_h2h_score.gt(0)
            | hole_h2h_label.isin(["対戦◎", "対戦○"])
        )
        h2h_strong_mask = h2h_recent_win_mask | hole_h2h_score.ge(2) | hole_h2h_label.eq("対戦◎")
        h2h_bad_mask = (~h2h_recent_win_mask) & (hole_h2h_score.lt(0) | h2h_recent_loss_mask)
        average_low_for_course_mask = (
            (field_average_value is not None and hole_average.le(field_average_value - 5))
            | hole_average_rank.ge(6)
        )
        course_hole_mask = (
            hole_course_rank.le(3)
            & hole_odds.ge(8)
            & hole_star.isna()
            & average_low_for_course_mask
        )
        weight_ok = hole_weight_change.isna() | hole_weight_change.le(0)
        age_ok = hole_age.isna() | hole_age.le(8)
        layoff_ok = ~df.loc[hole_pool.index].get(
            "_is_layoff", pd.Series(False, index=hole_pool.index)
        ).fillna(False).astype(bool)
        relative_index_hole_mask = (
            hole_average_rank.le(3)
            & hole_odds.ge(6)
            & age_ok
            & layoff_ok
        )

        special_hole_mask = (
            ai_window
            & class_down
            & upper_experience
            & ai_not_too_low
            & value_signal
            & weight_ok
            & age_ok
            & layoff_ok
        )
        h2h_hole_mask = (
            h2h_positive_mask
            & ~h2h_bad_mask
            & hole_ai_rank.ge(4)
            & (hole_ai_rank.le(12) | hole_ai.ge(top_ai - 45))
            & age_ok
            & layoff_ok
        )
        hole_pool["_☆専用点"] = pool_numeric(hole_pool, "_穴評価点", 0).fillna(0)
        hole_pool["_☆専用点"] += class_down.astype(float) * 6.0
        hole_pool["_☆専用点"] += upper_experience.astype(float) * 3.0
        hole_pool["_☆専用点"] += value_signal.astype(float) * 2.0
        hole_pool["_☆専用点"] += relative_index_hole_mask.astype(float) * 10.0
        hole_pool["_☆専用点"] += h2h_recent_win_mask.astype(float) * 12.0
        hole_pool["_☆専用点"] += course_hole_mask.astype(float) * 11.0
        hole_pool["_☆専用点"] += h2h_strong_mask.astype(float) * 9.0
        hole_pool["_☆専用点"] += h2h_positive_mask.astype(float) * 6.0
        hole_pool["_☆専用点"] -= h2h_bad_mask.astype(float) * 3.0
        hole_pool["_☆専用点"] += weight_ok.astype(float) * 1.5
        hole_pool["_☆専用点"] += age_ok.astype(float) * 1.0
        hole_pool["_☆専用点"] -= hole_ai_rank.le(3).astype(float) * 10.0
        hole_pool["_☆専用点"] -= hole_weight_change.gt(0).astype(float) * 3.0
        hole_pool["_☆専用点"] -= hole_age.ge(9).fillna(False).astype(float) * 4.0

        special_holes = hole_pool[special_hole_mask | h2h_hole_mask | course_hole_mask | relative_index_hole_mask]
        if not special_holes.empty:
            star_idx = special_holes.sort_values(
                ["_☆専用点", "_穴評価点", "_最終印点"],
                ascending=[False, False, False],
            ).index[0]
            if bool(relative_index_hole_mask.reindex([star_idx]).fillna(False).iloc[0]):
                append_mark_reason(star_idx, "指数妙味")
            elif bool(h2h_hole_mask.reindex([star_idx]).fillna(False).iloc[0]):
                append_mark_reason(star_idx, "対戦穴")
            elif bool(course_hole_mask.reindex([star_idx]).fillna(False).iloc[0]):
                append_mark_reason(star_idx, "コース穴")
            else:
                append_mark_reason(star_idx, "降級妙味")
        else:
            hole_signal = rank_gap.ge(3) | hole_odds.ge(8)
            hole_signal = hole_signal | hole_reason.str.contains("展開材料|人気以上|対戦◎|対戦先着|コース穴|指数妙味", na=False) | hole_shift.eq("クラス降級") | h2h_positive_mask | course_hole_mask
            regular_holes = hole_pool[hole_signal & hole_ai_rank.ge(4)]
            if not regular_holes.empty:
                star_idx = regular_holes.sort_values(["_穴評価点", "_最終印点"], ascending=[False, False]).index[0]
            elif len(remaining) >= 3:
                non_top_ai = [idx for idx in remaining if pd.to_numeric(df.at[idx, "AI順位"], errors="coerce") > 3]
                star_idx = non_top_ai[2] if len(non_top_ai) >= 3 else (non_top_ai[-1] if non_top_ai else remaining[2])
            elif remaining:
                star_idx = remaining[-1]

    delta_order = remaining
    if remaining:
        delta_pool = df.loc[remaining].copy()
        delta_ai_rank = pool_numeric(delta_pool, "AI順位")
        delta_ai = pool_numeric(delta_pool, "AI点")
        delta_odds = pool_numeric(delta_pool, "単勝オッズ")
        delta_final = pool_numeric(delta_pool, "_最終印点", 0).fillna(0)
        delta_h2h_score = pool_numeric(delta_pool, "_h2h_score", 0).fillna(0)
        delta_h2h_label = delta_pool.get("_h2h_label", pd.Series("", index=delta_pool.index)).fillna("").astype(str)
        delta_h2h_latest = delta_pool.get(
            "_h2h_latest", delta_pool.get("対戦", pd.Series("", index=delta_pool.index))
        ).fillna("").astype(str)
        delta_recent_win = delta_h2h_latest.str.contains("先着", na=False)
        delta_recent_loss = delta_h2h_latest.str.contains("敗戦", na=False)
        delta_h2h_positive = (
            delta_recent_win
            | delta_h2h_score.gt(0)
            | delta_h2h_label.isin(["対戦◎", "対戦○"])
        )
        delta_h2h_strong = delta_recent_win | delta_h2h_score.ge(2) | delta_h2h_label.eq("対戦◎")
        delta_h2h_bad = (~delta_recent_win) & (delta_h2h_score.lt(0) | delta_recent_loss)
        delta_course_rank = course_rank.reindex(delta_pool.index)
        delta_average_rank = average_rank.reindex(delta_pool.index)
        delta_average = pool_numeric(delta_pool, "平均指数").fillna(pool_numeric(delta_pool, "3走平均"))
        delta_star = pool_numeric(delta_pool, "★最高指数").fillna(pool_numeric(delta_pool, "★最高"))
        delta_average_low_for_course = (
            (field_average_value is not None and delta_average.le(field_average_value - 5))
            | delta_average_rank.ge(6)
        )
        delta_course_hole = (
            delta_course_rank.le(3)
            & delta_odds.ge(8)
            & delta_star.isna()
            & delta_average_low_for_course
        )
        delta_h2h_candidate = (
            delta_h2h_positive
            & ~delta_h2h_bad
            & delta_ai_rank.ge(4)
            & (delta_ai_rank.le(12) | delta_ai.ge(top_ai - 45))
        )
        delta_pool["_△対戦押上点"] = delta_final
        delta_pool["_△対戦押上点"] += delta_recent_win.astype(float) * 30.0
        delta_pool["_△対戦押上点"] += delta_course_hole.astype(float) * 24.0
        delta_pool["_△対戦押上点"] += delta_h2h_strong.astype(float) * 18.0
        delta_pool["_△対戦押上点"] += delta_h2h_positive.astype(float) * 12.0
        delta_pool["_△対戦押上点"] += delta_odds.ge(8).fillna(False).astype(float) * 2.0
        delta_pool["_△対戦押上点"] -= delta_h2h_bad.astype(float) * 12.0
        if bool((delta_h2h_candidate | delta_course_hole).any()):
            delta_order = delta_pool.sort_values(
                ["_△対戦押上点", "_最終印点"], ascending=[False, False]
            ).index.tolist()

    delta_slots = 0
    for idx in delta_order:
        if idx == star_idx or idx in assigned or delta_slots <= 0:
            continue
        df.at[idx, "最終印"] = "△"
        delta_h2h_score_value = pd.to_numeric(df.at[idx, "_h2h_score"], errors="coerce") if "_h2h_score" in df.columns else 0
        delta_h2h_label_value = str(df.at[idx, "_h2h_label"] if "_h2h_label" in df.columns else "")
        delta_h2h_latest_value = str(df.at[idx, "_h2h_latest"] if "_h2h_latest" in df.columns else df.at[idx, "対戦"] if "対戦" in df.columns else "")
        if pd.notna(delta_h2h_score_value) and delta_h2h_score_value > 0 or delta_h2h_label_value in ("対戦◎", "対戦○") or "先着" in delta_h2h_latest_value:
            append_mark_reason(idx, "対戦穴")
        if bool(df.at[idx, "_course_hole"]) if "_course_hole" in df.columns else False:
            append_mark_reason(idx, "コース穴")
        df.at[idx, "_最終印順"] = mark_order
        assigned.add(idx)
        mark_order += 1
        delta_slots -= 1

    if star_idx is not None and star_idx not in assigned:
        star_h2h_score = pd.to_numeric(df.at[star_idx, "_h2h_score"], errors="coerce") if "_h2h_score" in df.columns else 0
        star_h2h_label = str(df.at[star_idx, "_h2h_label"] if "_h2h_label" in df.columns else "")
        star_h2h_latest = str(df.at[star_idx, "_h2h_latest"] if "_h2h_latest" in df.columns else df.at[star_idx, "対戦"] if "対戦" in df.columns else "")
        if pd.notna(star_h2h_score) and star_h2h_score > 0 or star_h2h_label in ("対戦◎", "対戦○") or "先着" in star_h2h_latest:
            append_mark_reason(star_idx, "対戦穴")
        star_course_hole = bool(df.at[star_idx, "_course_hole"]) if "_course_hole" in df.columns else False
        if star_course_hole:
            append_mark_reason(star_idx, "コース穴")
        if str(df.at[star_idx, "クラス変動"] or "") == "クラス降級":
            append_mark_reason(star_idx, "降級妙味")
        df.at[star_idx, "最終印"] = "✓"
        df.at[star_idx, "_最終印順"] = 5
        assigned.add(star_idx)

    sort_columns = [col for col in ["_最終印点", "補正AI点", "AI点"] if col in df.columns]
    if sort_columns:
        df = df.sort_values(sort_columns, ascending=[False] * len(sort_columns)).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    return add_purchase_value_columns(df)



# ===== Ver2.0 評価ロジック =====
# AI点は能力評価としてそのまま使い、総合評価は条件補正だけを加える。
# Ver1.0の能力重複加点は add_final_marks_v1_legacy に残して比較可能にする。

def compute_market_reflection_rank_for_mark_decision(df):
    """◎の信頼度判定専用。表示用の市場反映勝率と同じ思想で順位だけを内部計算する。"""
    import numpy as np

    if df is None or len(df) == 0:
        return pd.Series(dtype="float64")

    score_source = df.get(
        "_最終印点",
        df.get("総合評価点", df.get("AI点", pd.Series(index=df.index, dtype="float64"))),
    )
    score = pd.to_numeric(score_source, errors="coerce")
    odds = pd.to_numeric(
        df.get("単勝オッズ", df.get("オッズ", pd.Series(index=df.index, dtype="float64"))),
        errors="coerce",
    )

    score_min = score.min()
    score_max = score.max()
    if pd.isna(score_min) or pd.isna(score_max) or score_max == score_min:
        normalized = pd.Series(0.5, index=df.index, dtype="float64")
    else:
        normalized = ((score - score_min) / (score_max - score_min)).fillna(0.5)

    model_weight = np.exp((normalized - normalized.max()) * 2.4)
    model_total = float(model_weight.sum())
    model_probability = (
        model_weight / model_total
        if model_total > 0
        else pd.Series(1.0 / len(df), index=df.index, dtype="float64")
    )

    market_raw = pd.Series(0.0, index=df.index, dtype="float64")
    valid_odds = odds.notna() & odds.gt(0)
    market_raw.loc[valid_odds] = 1.0 / odds.loc[valid_odds]
    market_total = float(market_raw.sum())
    market_probability = (
        market_raw / market_total if market_total > 0 else model_probability.copy()
    )

    probability = (model_probability * 0.5 + market_probability * 0.5).clip(0.001, 0.95)
    probability_cap = pd.Series(0.35, index=df.index, dtype="float64")
    probability_cap.loc[odds.gt(12)] = 0.09
    probability_cap.loc[odds.gt(20)] = 0.035
    probability_cap.loc[odds.gt(35)] = 0.015
    probability_cap.loc[odds.gt(80)] = 0.002
    probability = pd.concat([probability, probability_cap], axis=1).min(axis=1)

    return (probability * 100).rank(method="min", ascending=False)


def add_final_marks(df, running_info=None):
    df = df.copy()
    df = df.drop(columns=[
        "最終印", "展開印", "印理由", "クラス根拠", "総合評価点", "_最終印点", "_最終印順",
        "_穴評価点", "補正AI点", "補正値", "クラス補正", "状態補正",
        "展開補正", "対戦補正", "_重大マイナス数", "_course_hole", "_条件軸点",
    ], errors="ignore")
    if df.empty:
        for column in [
            "最終印", "展開印", "印理由", "クラス根拠", "総合評価点", "補正AI点", "_最終印点",
            "_最終印順", "_穴評価点", "補正値", "クラス補正", "状態補正",
            "展開補正", "対戦補正", "_重大マイナス数", "_course_hole", "_条件軸点",
        ]:
            df[column] = ""
        return df

    def numeric(column, default=None):
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce")
        return pd.Series(default, index=df.index, dtype="float64")

    data_shortage = (
        df.get("_地方指数データ不足", _nar_local_index_data_shortage_mask(df))
        .fillna(False)
        .astype(bool)
    )
    df["_地方指数データ不足"] = data_shortage

    ai = numeric("AI点").mask(data_shortage)
    if "AI順位" not in df.columns or pd.to_numeric(df.get("AI順位"), errors="coerce").isna().all():
        df["AI順位"] = ai.rank(method="min", ascending=False).astype("Int64")

    trend = numeric("_trend", 0).fillna(0)
    h2h_score = numeric("_h2h_score", 0).fillna(0)
    days_since_last = numeric("_days_since_last")
    layoff = df.get("_is_layoff", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    styles = df.get("脚質", pd.Series("", index=df.index)).map(normalize_running_style)
    running = running_info or analyze_running_style(df)
    favored_styles = set(running.get("有利脚質", []))
    pace_horses = set(
        int(value)
        for value in (running.get("展開向く馬番") or running.get("展開穴", []))
        if pd.notna(value)
    )

    recent_high = numeric("最高指数").fillna(numeric("近3走最高"))
    star_high = numeric("★最高指数").fillna(numeric("★最高"))
    recent_rank = recent_high.rank(method="min", ascending=False)
    star_rank = star_high.rank(method="min", ascending=False)

    class_adjustments = []
    condition_adjustments = []
    pace_adjustments = []
    matchup_adjustments = []
    total_adjustments = []
    total_scores = []
    reason_texts = []
    class_basis_texts = []
    class_shift_texts = []
    major_negative_counts = []
    course_hole_flags = []
    condition_axis_scores = []
    hole_scores = []

    def add_unique(parts, text):
        if text and text not in parts:
            parts.append(text)

    def clean_class_shift_value(value):
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except Exception:
            pass
        text = str(value).strip()
        if text.lower() in ("", "nan", "none", "<na>", "nat", "-"):
            return ""
        if "降級" in text or "相手弱化" in text:
            return "クラス降級"
        if "昇級" in text or "相手強化" in text:
            return "クラス昇級"
        if "同級" in text:
            return "同級"
        return text

    def rank_shift(current_rank, compared_rank):
        if current_rank is None or compared_rank is None:
            return ""
        diff = current_rank - compared_rank
        if diff >= 8:
            return "クラス昇級"
        if diff <= -8:
            return "クラス降級"
        return "同級"

    def class_shift_for(row):
        explicit_shift = ""
        for key in ("クラス変動", "_class_shift"):
            explicit_shift = clean_class_shift_value(row.get(key))
            if explicit_shift:
                break
        if not explicit_shift and "final_mark_class_shift" in globals():
            try:
                explicit_shift = clean_class_shift_value(final_mark_class_shift(row))
            except Exception:
                explicit_shift = ""

        current_rank = safe_num(row.get("_current_class_rank"), None)
        previous_rank = safe_num(row.get("_previous_class_rank"), None)
        best_rank = safe_num(row.get("_best_past_class_rank"), previous_rank)
        previous_shift = rank_shift(current_rank, previous_rank)
        best_shift = rank_shift(current_rank, best_rank)

        # 前走が同級でも、近3走最高クラスが明確に上なら「今回降級」として扱う。
        if best_shift in ("クラス降級", "クラス昇級") and explicit_shift in ("", "同級", "同級近辺"):
            return best_shift
        if explicit_shift:
            return explicit_shift
        if previous_shift:
            return previous_shift
        if best_shift:
            return best_shift
        basis = clean_class_shift_value(row.get("クラス根拠"))
        return basis

    for idx, row in df.iterrows():
        reasons = []

        # ===== Ver1.0 legacy notes =====
        # score += AI順位加点
        # score += 平均/最高/距離/コース/★最高の順位加点
        # score += ★最高×条件、近3走最高×条件、距離コース両方、条件軸材料
        # score += 人気以上、人気先行、オッズ妙味
        # Ver2.0では上記を総合評価から除外し、AI点に含まれる能力評価として扱う。

        class_shift = class_shift_for(row)
        if bool(data_shortage.loc[idx]):
            class_adjustments.append(pd.NA)
            condition_adjustments.append(pd.NA)
            pace_adjustments.append(pd.NA)
            matchup_adjustments.append(pd.NA)
            total_adjustments.append(pd.NA)
            total_scores.append(pd.NA)
            major_negative_counts.append(0)
            reason_texts.append("データ不足")
            class_shift_texts.append(class_shift)
            try:
                row_for_class_basis = row.copy()
                row_for_class_basis["クラス変動"] = class_shift
                row_for_class_basis["_class_shift"] = class_shift
                class_basis = final_mark_class_basis(row_for_class_basis)
            except Exception:
                class_basis = ""
            if class_shift and class_shift not in str(class_basis):
                class_basis = (str(class_basis) + " / " + class_shift).strip(" /")
            class_basis_texts.append(class_basis)
            course_hole_flags.append(False)
            condition_axis_scores.append(0.0)
            hole_scores.append(pd.NA)
            continue

        base_ai_value = float(ai.loc[idx]) if pd.notna(ai.loc[idx]) else 0.0
        class_adj = 0.0
        if class_shift in ("クラス降級", "相手弱化"):
            class_adj = 1.5
            add_unique(reasons, "クラス降級")
        elif class_shift in ("同級", "同級近辺"):
            class_adj = 0.5
            add_unique(reasons, "同級")
        elif class_shift in ("クラス昇級", "相手強化"):
            class_adj = -1.0
            add_unique(reasons, "クラス昇級注意")

        state_adj = 0.0
        state_down = False
        long_layoff = bool(pd.notna(days_since_last.loc[idx]) and days_since_last.loc[idx] >= 90)
        if long_layoff:
            state_adj -= 2.0
            add_unique(reasons, "長期休養")
        elif bool(layoff.loc[idx]) or (pd.notna(days_since_last.loc[idx]) and days_since_last.loc[idx] >= 45):
            state_adj -= 1.0
            add_unique(reasons, "休み明け")
        if trend.loc[idx] >= 5:
            state_adj += 0.5
            add_unique(reasons, "近走上向き")
        elif trend.loc[idx] <= -8:
            state_adj -= 1.0
            state_down = True
            add_unique(reasons, "指数下降")

        style = styles.loc[idx]
        horse_no = int(row.get("馬番", 0) or 0)
        pace_comment = str(row.get("展開コメント") or "")
        pace_focus = horse_no in pace_horses or "展開穴" in pace_comment or "単騎" in pace_comment
        pace_wait = (
            "展開待ち" in pace_comment
            or "展開不利" in pace_comment
            or "課題" in pace_comment
        )
        closer_risk = style == "追" and not pace_focus
        pace_adj = 0.0
        if pace_focus:
            pace_adj += 1.0
            add_unique(reasons, "展開向く")
        elif style in favored_styles:
            pace_adj += 0.5
            add_unique(reasons, "有利脚質")
        if pace_wait:
            pace_adj -= 1.0
            add_unique(reasons, "展開待ち")
        if closer_risk:
            pace_adj -= 0.5
            add_unique(reasons, "追込リスク")

        h2h_value = safe_num(h2h_score.loc[idx], 0) or 0
        h2h_label = str(row.get("_h2h_label") or "")
        h2h_latest = str(row.get("_h2h_latest") or row.get("対戦") or "")
        h2h_recent_win = "先着" in h2h_latest
        h2h_negative = (
            (not h2h_recent_win)
            and (h2h_value < 0 or h2h_label == "対戦△" or "敗戦" in h2h_latest or "負け越し" in h2h_latest)
        )
        matchup_adj = 0.0
        if h2h_recent_win or h2h_value > 0 or h2h_label in ("対戦◎", "対戦○"):
            matchup_adj += 1.0
            add_unique(reasons, "対戦先着")
        elif h2h_negative:
            matchup_adj -= 0.5
            add_unique(reasons, "対戦劣勢")

        raw_adjustment = class_adj + state_adj + pace_adj + matchup_adj
        total_adjustment = min(max(raw_adjustment, -6.0), 4.0)
        total_score = base_ai_value + total_adjustment

        major_negative_count = 0
        if style == "追":
            major_negative_count += 1
        if pace_wait:
            major_negative_count += 1
        if long_layoff:
            major_negative_count += 1
        if state_down:
            major_negative_count += 1

        if not reasons:
            add_unique(reasons, "条件補正なし")

        class_adjustments.append(round(class_adj, 1))
        condition_adjustments.append(round(state_adj, 1))
        pace_adjustments.append(round(pace_adj, 1))
        matchup_adjustments.append(round(matchup_adj, 1))
        total_adjustments.append(round(total_adjustment, 1))
        total_scores.append(round(total_score, 2))
        major_negative_counts.append(int(major_negative_count))
        reason_texts.append(" / ".join(reasons[:6]))
        class_shift_texts.append(class_shift)
        try:
            row_for_class_basis = row.copy()
            row_for_class_basis["クラス変動"] = class_shift
            row_for_class_basis["_class_shift"] = class_shift
            class_basis = final_mark_class_basis(row_for_class_basis)
        except Exception:
            class_basis = ""
        if class_shift and class_shift not in str(class_basis):
            class_basis = (str(class_basis) + " / " + class_shift).strip(" /")
        class_basis_texts.append(class_basis)
        course_hole_flags.append(False)
        condition_axis_scores.append(0.0)
        hole_scores.append(round(total_score, 2))

    df["_最終印点"] = total_scores
    df["_穴評価点"] = hole_scores
    df["補正AI点"] = total_scores
    df["補正値"] = total_adjustments
    df["クラス補正"] = class_adjustments
    df["状態補正"] = condition_adjustments
    df["展開補正"] = pace_adjustments
    df["対戦補正"] = matchup_adjustments
    df["_重大マイナス数"] = major_negative_counts
    df["_course_hole"] = course_hole_flags
    df["_条件軸点"] = condition_axis_scores
    df["印理由"] = reason_texts
    df["クラス変動"] = class_shift_texts
    df["クラス根拠"] = class_basis_texts
    df["総合評価点"] = pd.to_numeric(df["_最終印点"], errors="coerce").round(1)

    df["展開印"] = ""
    pace_numbers = set(
        int(value)
        for value in (running.get("展開向く馬番") or running.get("展開穴", []))
        if pd.notna(value)
    )
    if pace_numbers:
        pace_candidates = df[(~data_shortage) & df["馬番"].astype(int).isin(pace_numbers)].copy()
    else:
        pace_candidates = df[(~data_shortage) & df["印理由"].astype(str).str.contains("展開向く", na=False)].copy()
    if not pace_candidates.empty:
        pace_idx = pace_candidates.sort_values(["_最終印点", "AI点"], ascending=[False, False]).index[0]
        df.at[pace_idx, "展開印"] = "展"

    df["最終印"] = ""
    df["_最終印順"] = pd.NA
    mark_eligible = (~data_shortage) & pd.to_numeric(df["_最終印点"], errors="coerce").notna()
    ordered_indices = df.loc[mark_eligible].sort_values(["_最終印点", "AI点"], ascending=[False, False]).index.tolist()
    assigned = set()
    market_rank_for_mark = compute_market_reflection_rank_for_mark_decision(df)

    def market_gap_for_mark(idx):
        ai_rank_value = safe_num(df.at[idx, "AI順位"] if "AI順位" in df.columns else None, None)
        market_rank_value = safe_num(
            market_rank_for_mark.reindex([idx]).iloc[0]
            if idx in market_rank_for_mark.index
            else None,
            None,
        )
        if ai_rank_value is None or market_rank_value is None:
            return None
        return market_rank_value - ai_rank_value

    def should_market_downgrade_honmei(idx):
        gap = market_gap_for_mark(idx)
        return gap is not None and gap >= 3

    if ordered_indices:
        top_idx = ordered_indices[0]
        honmei_idx = top_idx
        downgraded_top = None
        downgrade_reasons = []
        if int(df.at[top_idx, "_重大マイナス数"] or 0) >= 2:
            replacement = next(
                (idx for idx in ordered_indices[1:] if int(df.at[idx, "_重大マイナス数"] or 0) < 2),
                None,
            )
            if replacement is not None:
                honmei_idx = replacement
                downgraded_top = top_idx
                downgrade_reasons.append("重大マイナスで○")
                add_unique(reason_texts, "")

        if honmei_idx == top_idx and should_market_downgrade_honmei(top_idx):
            honmei_idx = None
            downgraded_top = top_idx
            downgrade_reasons.append("市場評価差で○")

        if honmei_idx is not None:
            df.at[honmei_idx, "最終印"] = "◎"
            df.at[honmei_idx, "_最終印順"] = 0
            assigned.add(honmei_idx)
        if downgraded_top is not None:
            df.at[downgraded_top, "最終印"] = "○"
            df.at[downgraded_top, "_最終印順"] = 1
            current = str(df.at[downgraded_top, "印理由"] or "")
            for downgrade_reason in downgrade_reasons:
                if downgrade_reason and downgrade_reason not in current:
                    current = (current + " / " + downgrade_reason).strip(" /")
            df.at[downgraded_top, "印理由"] = current
            assigned.add(downgraded_top)

    has_honmei = any(df["最終印"].astype(str).eq("◎"))
    has_taikou = any(df["最終印"].astype(str).eq("○"))
    if not has_honmei and has_taikou:
        next_marks = ["○", "▲", "△", "△"]
        mark_order_start = 1
    elif has_honmei and has_taikou:
        next_marks = ["▲", "△", "△"]
        mark_order_start = 2
    else:
        next_marks = ["○", "▲", "△", "△"]
        mark_order_start = 1
    mark_order = mark_order_start
    for mark in next_marks:
        idx = next((x for x in ordered_indices if x not in assigned), None)
        if idx is None:
            break
        df.at[idx, "最終印"] = mark
        df.at[idx, "_最終印順"] = mark_order
        assigned.add(idx)
        mark_order += 1

    remaining = [idx for idx in ordered_indices if idx not in assigned]
    star_idx = None
    if remaining:
        star_pool = df.loc[remaining].copy()
        star_pool["_☆ピーク順位"] = pd.concat(
            [star_rank.reindex(star_pool.index), recent_rank.reindex(star_pool.index)],
            axis=1,
        ).min(axis=1)
        peak_pool = star_pool[star_pool["_☆ピーク順位"].le(3)]
        if not peak_pool.empty:
            star_idx = peak_pool.sort_values(
                ["_☆ピーク順位", "_最終印点", "AI点"], ascending=[True, False, False]
            ).index[0]
        else:
            star_idx = star_pool.sort_values(["_最終印点", "AI点"], ascending=[False, False]).index[0]
    if star_idx is not None:
        df.at[star_idx, "最終印"] = "✓"
        df.at[star_idx, "_最終印順"] = 5
        current = str(df.at[star_idx, "印理由"] or "")
        if "ピーク能力" not in current and (
            safe_num(star_rank.reindex([star_idx]).iloc[0], None) is not None
            and star_rank.reindex([star_idx]).iloc[0] <= 3
            or safe_num(recent_rank.reindex([star_idx]).iloc[0], None) is not None
            and recent_rank.reindex([star_idx]).iloc[0] <= 3
        ):
            df.at[star_idx, "印理由"] = (current + " / ピーク能力").strip(" /")

    sort_columns = [col for col in ["_最終印点", "補正AI点", "AI点"] if col in df.columns]
    if sort_columns:
        df = df.sort_values(sort_columns, ascending=[False] * len(sort_columns)).reset_index(drop=True)
    else:
        df = df.reset_index(drop=True)
    return add_purchase_value_columns(df)


def build_evaluation_adjustment_log(df):
    log_df = df.copy()
    if "総合評価点" not in log_df.columns and "_最終印点" in log_df.columns:
        log_df["総合評価点"] = pd.to_numeric(log_df["_最終印点"], errors="coerce").round(1)
    if "総合評価" not in log_df.columns and "総合評価点" in log_df.columns:
        log_df["総合評価"] = pd.to_numeric(log_df["総合評価点"], errors="coerce").round(1)
    if "補正値" not in log_df.columns and {"総合評価点", "AI点"}.issubset(log_df.columns):
        log_df["補正値"] = (
            pd.to_numeric(log_df["総合評価点"], errors="coerce")
            - pd.to_numeric(log_df["AI点"], errors="coerce")
        ).round(1)
    if "印" not in log_df.columns and "最終印" in log_df.columns:
        log_df["印"] = log_df["最終印"].fillna("").astype(str)
    if "市場反映勝率" not in log_df.columns and "推定勝率" in log_df.columns:
        log_df["市場反映勝率"] = pd.to_numeric(log_df["推定勝率"], errors="coerce")

    ai_rank = pd.to_numeric(log_df.get("AI点"), errors="coerce").rank(method="min", ascending=False)
    final_score = pd.to_numeric(
        log_df.get("総合評価点", log_df.get("_最終印点")),
        errors="coerce",
    )
    final_rank = final_score.rank(method="min", ascending=False)

    def rank_change_text(idx):
        ai_value = ai_rank.loc[idx] if idx in ai_rank.index else pd.NA
        final_value = final_rank.loc[idx] if idx in final_rank.index else pd.NA
        if pd.isna(ai_value) or pd.isna(final_value):
            return ""
        ai_int = int(ai_value)
        final_int = int(final_value)
        diff = ai_int - final_int
        if diff > 0:
            arrow = f"▲{diff}"
        elif diff < 0:
            arrow = f"▼{abs(diff)}"
        else:
            arrow = "±0"
        return f"AI{ai_int}位→総合{final_int}位 {arrow}"

    log_df["順位変動"] = [rank_change_text(idx) for idx in log_df.index]
    columns = [
        "馬番", "馬名", "AI点", "総合評価", "補正値", "クラス補正", "状態補正",
        "展開補正", "対戦補正", "印", "市場反映勝率", "順位変動",
    ]
    existing = [column for column in columns if column in log_df.columns]
    return log_df[existing].reset_index(drop=True)


# ===== Ver2.5 Final UI: AI信頼度（表示のみ） =====
AI_CONFIDENCE_VENUE_EVAL = {
    "nar": {
        "盛岡": "A",
        "名古屋": "A",
        "門別": "A",
        "金沢": "A",
        "浦和": "B",
        "川崎": "B",
        "船橋": "B",
        "大井": "B",
        "園田": "B",
        "高知": "B",
        "佐賀": "B",
        "笠松": "B",
        "水沢": "B",
    },
    "jra": {
        "福島": "A",
        "函館": "B",
        "小倉": "B",
        "東京": "B",
        "中山": "B",
        "京都": "B",
        "阪神": "B",
        "中京": "B",
        "札幌": "B",
        "新潟": "B",
    },
}

AI_CONFIDENCE_COMMENT = {
    "★★★★★": {
        "label": "【軸勝負】",
        "comment": (
            "本命信頼度が高いレースです。\n\n"
            "地方\n"
            "◎複勝\n"
            "◎軸ワイド\n\n"
            "中央\n"
            "◎単勝\n\n"
            "を基本に積極購入してください。"
        ),
    },
    "★★★★☆": {
        "label": "【通常購入】",
        "comment": (
            "本命は信頼できます。\n\n"
            "地方\n"
            "ワイド・馬連\n\n"
            "中央\n"
            "単勝・馬連\n\n"
            "を基本に購入してください。"
        ),
    },
    "★★★☆☆": {
        "label": "【混戦・BOX検討】",
        "comment": (
            "混戦レースです。\n\n"
            "軸固定よりBOX向きですが、\n\n"
            "点数と想定配当を確認し、\n"
            "期待値が低い場合は見送りも有効です。"
        ),
    },
    "★★☆☆☆": {
        "label": "【BOXまたは少額】",
        "comment": (
            "本命の信頼度は低めです。\n\n"
            "BOXまたは少額購入向きです。"
        ),
    },
    "★☆☆☆☆": {
        "label": "【軸非推奨】",
        "comment": (
            "本命としては信頼できません。\n\n"
            "BOXで遊ぶ程度、\n"
            "または見送りを推奨します。"
        ),
    },
}


def _confidence_safe_float(value):
    try:
        number = pd.to_numeric(value, errors="coerce")
        if pd.isna(number):
            return None
        return float(number)
    except Exception:
        return None


def _confidence_format_number(value):
    if value is None or pd.isna(value):
        return "-"
    return f"{float(value):.1f}".rstrip("0").rstrip(".")


def _confidence_rank_text(value):
    if value is None or pd.isna(value):
        return "-"
    return f"{int(value)}位"


def _confidence_check(status):
    return {"good": "✓", "warn": "△", "bad": "×"}.get(status, "△")


def _confidence_mark_score(value, good, warn):
    if value is None or pd.isna(value):
        return "bad"
    if value >= good:
        return "good"
    if value >= warn:
        return "warn"
    return "bad"


def _confidence_market_status(rank):
    if rank is None or pd.isna(rank):
        return "bad"
    if rank <= 3:
        return "good"
    if rank <= 5:
        return "warn"
    return "bad"


def _confidence_eval_status(value):
    value = str(value or "").strip().upper()
    if value == "A":
        return "good"
    if value == "B":
        return "warn"
    return "bad"


def _confidence_diff_status(value, race_type="nar"):
    if value is None or pd.isna(value):
        return "bad"
    if race_type == "jra":
        if 5 <= value < 10:
            return "good"
        if 0 <= value < 5:
            return "warn"
        if value >= 10:
            return "bad"
        return "warn"
    if value >= 10:
        return "good"
    if value >= 0:
        return "warn"
    return "bad"


def classify_ai_confidence_condition(race_info=None, df=None, race_type="nar"):
    race_info = race_info or {}
    text = " ".join(
        str(race_info.get(key) or "")
        for key in ["race_name", "race_data", "race_class", "class_name"]
    )
    if df is not None and "クラス根拠" in df.columns:
        text += " " + " ".join(df["クラス根拠"].fillna("").astype(str).head(6).tolist())
    if df is not None and "クラス変動" in df.columns:
        text += " " + " ".join(df["クラス変動"].fillna("").astype(str).head(6).tolist())

    if race_type == "jra":
        if any(token in text for token in ["G1", "G2", "G3", "重賞", "リステッド", "OP", "オープン"]):
            return "重賞/OP", "A"
        if "3歳未勝利" in text or ("3歳" in text and "未勝利" in text):
            return "3歳未勝利", "B"
        if "3歳1勝" in text or ("3歳" in text and "1勝" in text):
            return "3歳1勝", "A"
        if "2歳" in text or "新馬" in text:
            return "2歳/新馬", "B"
        return "古馬混合", "A"

    if "2歳" in text:
        return "2歳", "A"
    if "3歳" in text and not any(token in text for token in ["C1", "C2", "C3", "B1", "B2", "A1", "A2"]):
        return "地方3歳", "C"
    if any(token in text for token in ["A1", "A2", "A級"]):
        return "地方A級", "B"
    if any(token in text for token in ["B1", "B2", "B3", "B級"]):
        return "地方B級", "A"
    if any(token in text for token in ["C1", "C2", "C3", "C級"]):
        return "地方C級", "B"
    return "古馬混合", "A"


def build_ai_confidence_summary(df, race_info=None, detected_venue="", venue_profile=None, race_type="nar"):
    result = df.copy()
    if result.empty:
        return {
            "stars": "★☆☆☆☆",
            "label": AI_CONFIDENCE_COMMENT["★☆☆☆☆"]["label"],
            "comment": "購入非推奨です。",
            "ai_diff": None,
            "total_diff": None,
            "market_rank": None,
            "venue_eval": "C",
            "condition_eval": "C",
            "condition_label": "取得不可",
            "has_honmei": False,
            "reasons": ["◎が不在、または評価対象を取得できません。"],
        }

    mark = result.get("最終印", pd.Series("", index=result.index)).fillna("").astype(str)
    honmei_pool = result[mark.eq("◎")].copy()
    has_honmei = not honmei_pool.empty

    if "総合評価点" not in result.columns and "_最終印点" in result.columns:
        result["総合評価点"] = pd.to_numeric(result["_最終印点"], errors="coerce")
    if "総合評価" not in result.columns and "総合評価点" in result.columns:
        result["総合評価"] = pd.to_numeric(result["総合評価点"], errors="coerce")
    if "市場反映勝率" not in result.columns and "推定勝率" in result.columns:
        result["市場反映勝率"] = pd.to_numeric(result["推定勝率"], errors="coerce")

    final_order = pd.to_numeric(result.get("_最終印順", pd.Series(99, index=result.index)), errors="coerce").fillna(99)
    final_score = pd.to_numeric(
        result.get("総合評価点", result.get("総合評価", result.get("_最終印点", result.get("AI点")))),
        errors="coerce",
    )
    candidate = result.assign(_confidence_order=final_order, _confidence_score=final_score)

    if has_honmei:
        honmei_idx = honmei_pool.index[0]
        other_marked = candidate[
            (candidate.index != honmei_idx)
            & mark.isin(["○", "▲", "△", "✓", "☆"])
        ].sort_values(["_confidence_order", "_confidence_score", "AI点"], ascending=[True, False, False])
        if other_marked.empty:
            other_marked = candidate[candidate.index.ne(honmei_idx)].sort_values(
                ["_confidence_score", "AI点"], ascending=[False, False]
            )
        ref_row = other_marked.iloc[0] if not other_marked.empty else None
        honmei_row = result.loc[honmei_idx]
        ai_diff = None if ref_row is None else (
            _confidence_safe_float(honmei_row.get("AI点")) - _confidence_safe_float(ref_row.get("AI点"))
            if _confidence_safe_float(honmei_row.get("AI点")) is not None and _confidence_safe_float(ref_row.get("AI点")) is not None
            else None
        )
        honmei_total = _confidence_safe_float(honmei_row.get("総合評価点", honmei_row.get("総合評価")))
        ref_total = None if ref_row is None else _confidence_safe_float(ref_row.get("総合評価点", ref_row.get("総合評価")))
        total_diff = honmei_total - ref_total if honmei_total is not None and ref_total is not None else None
    else:
        honmei_idx = None
        honmei_row = None
        ai_diff = None
        total_diff = None

    market_rank = None
    if has_honmei:
        if "市場反映勝率" in result.columns:
            market_series = pd.to_numeric(result["市場反映勝率"], errors="coerce")
            market_rank_series = market_series.rank(method="min", ascending=False)
            market_rank = _confidence_safe_float(market_rank_series.loc[honmei_idx])
        else:
            try:
                market_rank_series = compute_market_reflection_rank_for_mark_decision(result)
                market_rank = _confidence_safe_float(market_rank_series.loc[honmei_idx])
            except Exception:
                market_rank = None

    venue = str(detected_venue or (race_info or {}).get("racecourse") or "").strip()
    venue_eval = AI_CONFIDENCE_VENUE_EVAL.get(race_type, {}).get(venue, "B")
    condition_label, condition_eval = classify_ai_confidence_condition(race_info, result, race_type)

    def ai_diff_score(value):
        if value is None or pd.isna(value):
            return -3
        if race_type == "jra":
            if 5 <= value < 10:
                return 3
            if 0 <= value < 2:
                return 1
            if 2 <= value < 5:
                return 0
            if value >= 10:
                return -1
            return 0
        if value >= 10:
            return 3
        if value >= 5:
            return 1
        if value >= 0:
            return 0
        return -2

    def total_diff_score(value):
        return ai_diff_score(value)

    def market_score(rank):
        if rank is None or pd.isna(rank):
            return -3
        if rank <= 3:
            return 2
        if rank <= 5:
            return 0
        return -3

    def eval_score(value):
        value = str(value or "").strip().upper()
        if value == "A":
            return 2
        if value == "B":
            return 1
        return -1

    score = (
        ai_diff_score(ai_diff)
        + total_diff_score(total_diff)
        + market_score(market_rank)
        + min(2, eval_score(venue_eval) + eval_score(condition_eval))
    )
    if not has_honmei:
        score = min(score, -1)

    if score >= 9:
        stars = "★★★★★"
    elif score >= 5:
        stars = "★★★★☆"
    elif score >= 3:
        stars = "★★★☆☆"
    elif score >= 0:
        stars = "★★☆☆☆"
    else:
        stars = "★☆☆☆☆"

    info = AI_CONFIDENCE_COMMENT[stars]
    reasons = []
    if not has_honmei:
        reasons.append("市場警戒により◎を設定していません。")

    return {
        "stars": stars,
        "label": info["label"],
        "comment": info["comment"],
        "ai_diff": None if ai_diff is None else round(float(ai_diff), 1),
        "total_diff": None if total_diff is None else round(float(total_diff), 1),
        "market_rank": None if market_rank is None else int(market_rank),
        "venue_eval": venue_eval,
        "condition_eval": condition_eval,
        "condition_label": condition_label,
        "has_honmei": has_honmei,
        "score": score,
        "reasons": reasons,
    }


def _ai_confidence_ticket_lines(stars, race_type="nar"):
    if stars == "★★★★★":
        if race_type == "jra":
            return ["◎単勝", "", "馬連BOX", "◎○▲"]
        return ["◎複勝", "", "ワイド", "◎－○", "◎－▲", "◎－△①"]
    if stars == "★★★★☆":
        if race_type == "jra":
            return ["◎単勝", "", "馬連BOX", "◎○▲"]
        return ["ワイド", "馬連"]
    if stars == "★★★☆☆":
        return ["印6頭BOX", "3連複BOX"]
    if stars == "★★☆☆☆":
        return ["印6頭BOX（少額）"]
    return ["BOXで遊ぶ程度", "または見送り"]


def _ai_confidence_marked_horses(df):
    if df is None or df.empty:
        return []
    source = df.copy()
    mark = source.get("最終印", pd.Series("", index=source.index)).fillna("").astype(str)
    source["_confidence_mark"] = mark
    source["_confidence_order"] = pd.to_numeric(
        source.get("_最終印順", pd.Series(99, index=source.index)),
        errors="coerce",
    ).fillna(99)
    source["_confidence_score"] = pd.to_numeric(
        source.get("総合評価点", source.get("総合評価", source.get("_最終印点", source.get("AI点")))),
        errors="coerce",
    )
    marked = source[source["_confidence_mark"].isin(["◎", "○", "▲", "△", "✓", "☆"])].copy()
    marked = marked.sort_values(["_confidence_order", "_confidence_score", "AI点"], ascending=[True, False, False])
    horses = []
    delta_count = 0
    for _, row in marked.iterrows():
        mark_value = str(row.get("_confidence_mark") or "").strip()
        horse_no = pd.to_numeric(row.get("馬番"), errors="coerce")
        if pd.isna(horse_no):
            continue
        if mark_value == "△":
            delta_count += 1
            role = f"△{delta_count}"
        else:
            role = mark_value
        horses.append({
            "role": role,
            "mark": mark_value,
            "no": int(horse_no),
            "name": str(row.get("馬名") or "").strip(),
        })
    return horses


def _odds_combo_label(horses):
    return "－".join(str(horse) for horse in horses)


def _odds_role_label(horse):
    label = str(horse.get("role") or "")
    name = str(horse.get("name") or "").strip()
    no = horse.get("no")
    return f"{label}{no} {name}".strip()


def _parse_odds_number(text):
    import re
    if text is None:
        return None
    normalized = str(text).replace(",", "").replace("〜", "-").replace("～", "-")
    values = re.findall(r"\d+(?:\.\d+)?", normalized)
    if not values:
        return None
    numbers = [float(value) for value in values[:2]]
    if len(numbers) == 1:
        return (numbers[0], numbers[0])
    return (min(numbers), max(numbers))


def _format_odds_range(value):
    if not value:
        return "取得不可"
    low, high = value
    def fmt(number):
        return f"{float(number):.1f}".rstrip("0").rstrip(".")
    if abs(low - high) < 0.001:
        return f"{fmt(low)}倍"
    return f"{fmt(low)}〜{fmt(high)}倍"


def _odds_average(value):
    if not value:
        return None
    return (float(value[0]) + float(value[1])) / 2


def _merge_odds(current, candidate):
    if not candidate:
        return current
    if not current:
        return candidate
    return (min(current[0], candidate[0]), max(current[1], candidate[1]))


def _odds_key(numbers):
    return tuple(sorted(int(number) for number in numbers))


def _classify_odds_table(text):
    text = str(text or "")
    if any(token in text for token in ["ワイド", "Wide", "wide"]):
        return "wide"
    if any(token in text for token in ["馬連", "Umaren", "umaren"]):
        return "umaren"
    if any(token in text for token in ["3連複", "三連複", "Sanrenpuku", "sanrenpuku"]):
        return "sanrenpuku"
    if any(token in text for token in ["単勝", "Tansho", "tansho"]):
        return "tansho"
    return ""


def _extract_horse_numbers_from_text(text):
    import re
    numbers = []
    for token in re.findall(r"(?<!\d)(1[0-8]|[1-9])(?!\d)", str(text or "")):
        number = int(token)
        if number not in numbers:
            numbers.append(number)
    return numbers


def _strip_html_tags(text):
    import re
    text = re.sub(r"<script.*?</script>", " ", str(text or ""), flags=re.I | re.S)
    text = re.sub(r"<style.*?</style>", " ", text, flags=re.I | re.S)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&nbsp;", " ").replace("&amp;", "&")
    return re.sub(r"\s+", " ", text).strip()


def _parse_optional_odds_html_regex(odds_html):
    import re
    data = {"wide": {}, "umaren": {}, "sanrenpuku": {}, "tansho": {}}
    tables = re.findall(r"<table.*?</table>", str(odds_html or ""), flags=re.I | re.S)
    for table_html in tables:
        table_text = _strip_html_tags(table_html)
        bet_type = _classify_odds_table(table_text)
        rows = re.findall(r"<tr.*?</tr>", table_html, flags=re.I | re.S)
        if not rows:
            continue
        header_cells = re.findall(r"<t[hd].*?</t[hd]>", rows[0], flags=re.I | re.S)
        header_numbers = []
        for cell in header_cells:
            nums = _extract_horse_numbers_from_text(_strip_html_tags(cell))
            header_numbers.append(nums[0] if nums else None)
        for row_html in rows[1:]:
            cells = re.findall(r"<t[hd].*?</t[hd]>", row_html, flags=re.I | re.S)
            if not cells:
                continue
            row_numbers = _extract_horse_numbers_from_text(_strip_html_tags(cells[0]))
            row_no = row_numbers[0] if row_numbers else None
            for col_idx, cell in enumerate(cells[1:], start=1):
                odds_value = _parse_odds_number(_strip_html_tags(cell))
                if not odds_value:
                    continue
                col_no = header_numbers[col_idx] if col_idx < len(header_numbers) else None
                if row_no and col_no and row_no != col_no:
                    use_type = bet_type or "wide"
                    key = _odds_key([row_no, col_no])
                    data[use_type][key] = _merge_odds(data[use_type].get(key), odds_value)
            row_text = _strip_html_tags(row_html)
            nums = _extract_horse_numbers_from_text(row_text)
            odds_candidates = re.findall(r"\d+(?:\.\d+)?\s*(?:[-〜～]\s*\d+(?:\.\d+)?)?\s*倍?", row_text)
            if not odds_candidates:
                continue
            odds_value = _parse_odds_number(odds_candidates[-1])
            if len(nums) >= 3 and odds_value:
                key = _odds_key(nums[:3])
                data["sanrenpuku"][key] = _merge_odds(data["sanrenpuku"].get(key), odds_value)
            elif len(nums) >= 2 and odds_value:
                use_type = bet_type if bet_type in ("wide", "umaren") else "wide"
                key = _odds_key(nums[:2])
                data[use_type][key] = _merge_odds(data[use_type].get(key), odds_value)
    return data


def parse_optional_odds_html(odds_html):
    if not odds_html:
        return {"wide": {}, "umaren": {}, "sanrenpuku": {}, "tansho": {}}
    import re
    try:
        from bs4 import BeautifulSoup
    except Exception:
        return _parse_optional_odds_html_regex(odds_html)

    soup = BeautifulSoup(odds_html, "html.parser")
    data = {"wide": {}, "umaren": {}, "sanrenpuku": {}, "tansho": {}}

    for table in soup.find_all("table"):
        table_text = table.get_text(" ", strip=True)
        bet_type = _classify_odds_table(table_text)
        rows = table.find_all("tr")
        if not rows:
            continue

        header_numbers = []
        first_cells = rows[0].find_all(["th", "td"])
        for cell in first_cells:
            nums = _extract_horse_numbers_from_text(cell.get_text(" ", strip=True))
            header_numbers.append(nums[0] if nums else None)

        for row in rows[1:]:
            cells = row.find_all(["th", "td"])
            if not cells:
                continue
            row_numbers = _extract_horse_numbers_from_text(cells[0].get_text(" ", strip=True))
            row_no = row_numbers[0] if row_numbers else None

            for col_idx, cell in enumerate(cells[1:], start=1):
                odds_value = _parse_odds_number(cell.get_text(" ", strip=True))
                if not odds_value:
                    continue
                col_no = header_numbers[col_idx] if col_idx < len(header_numbers) else None
                if row_no and col_no and row_no != col_no:
                    use_type = bet_type or "wide"
                    key = _odds_key([row_no, col_no])
                    data.setdefault(use_type, {})
                    data[use_type][key] = _merge_odds(data[use_type].get(key), odds_value)

            row_text = row.get_text(" ", strip=True)
            nums = _extract_horse_numbers_from_text(row_text)
            odds_candidates = re.findall(r"\d+(?:\.\d+)?\s*(?:[-〜～]\s*\d+(?:\.\d+)?)?\s*倍?", row_text)
            if not odds_candidates:
                continue
            odds_value = _parse_odds_number(odds_candidates[-1])
            if not odds_value:
                continue
            if len(nums) >= 3:
                use_type = bet_type if bet_type in ("sanrenpuku",) else "sanrenpuku"
                key = _odds_key(nums[:3])
                data[use_type][key] = _merge_odds(data[use_type].get(key), odds_value)
            elif len(nums) >= 2:
                use_type = bet_type if bet_type in ("wide", "umaren") else "wide"
                key = _odds_key(nums[:2])
                data[use_type][key] = _merge_odds(data[use_type].get(key), odds_value)
            elif len(nums) == 1 and bet_type == "tansho":
                data["tansho"][(nums[0],)] = _merge_odds(data["tansho"].get((nums[0],)), odds_value)

    return data


def inspect_optional_odds_html(odds_html):
    info = {
        "available": bool(odds_html),
        "body_id": "",
        "title": "",
        "table_count": 0,
        "length": len(str(odds_html or "")),
    }
    if not odds_html:
        return info
    import re

    def fallback_body_id():
        if "Netkeiba_Race_OddsView" in str(odds_html):
            return "Netkeiba_Race_OddsView"
        body_match = re.search(r"<body[^>]*id\s*=\s*[\"']?([^\"'\s>]+)", str(odds_html), flags=re.I)
        return body_match.group(1).strip() if body_match else ""

    def fallback_title():
        title_match = re.search(r"<title[^>]*>(.*?)</title>", str(odds_html), flags=re.I | re.S)
        return _strip_html_tags(title_match.group(1)).strip() if title_match else ""

    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(odds_html, "html.parser")
        body = soup.find("body")
        info["body_id"] = str(body.get("id", "")).strip() if body else ""
        title = soup.find("title")
        info["title"] = title.get_text(" ", strip=True) if title else ""
        if not info["body_id"]:
            info["body_id"] = fallback_body_id()
        if not info["title"]:
            info["title"] = fallback_title()
        count = 0
        for table in soup.find_all("table"):
            table_text = table.get_text(" ", strip=True)
            table_id = " ".join([
                str(table.get("id", "")),
                " ".join(table.get("class", []) if isinstance(table.get("class"), list) else [str(table.get("class", ""))]),
                table_text[:500],
            ])
            if _classify_odds_table(table_id) or any(key in table_id for key in ["Odds", "オッズ", "RaceOdds"]):
                count += 1
        info["table_count"] = count
    except Exception:
        info["body_id"] = fallback_body_id()
        info["title"] = fallback_title()
        tables = re.findall(r"<table.*?</table>", str(odds_html), flags=re.I | re.S)
        info["table_count"] = sum(
            1 for table in tables
            if _classify_odds_table(_strip_html_tags(table)) or any(key in table for key in ["Odds", "オッズ", "RaceOdds"])
        )
    if not info["table_count"]:
        odds_data = parse_optional_odds_html(odds_html)
        info["table_count"] = sum(1 for values in odds_data.values() if values)
    return info


def print_optional_odds_html_debug(odds_html="", odds_file_name=""):
    info = inspect_optional_odds_html(odds_html)
    print(f"オッズHTML判定：{'あり' if info['available'] else 'なし'}")
    if not info["available"]:
        return
    print(f"ファイル：{odds_file_name or '-'}")
    print(f"オッズHTML文字数：{info['length']:,}")
    print(f"body id：{info['body_id'] or '-'}")
    print(f"title：{info['title'] or '-'}")
    print(f"抽出オッズ表数：{info['table_count']}")


def _lookup_combo_odds(odds_data, bet_type, numbers):
    if not odds_data:
        return None
    key = _odds_key(numbers)
    return odds_data.get(bet_type, {}).get(key)


def _recommended_combo_specs(horses, stars, race_type="nar"):
    by_role = {horse["role"]: horse for horse in horses}
    top6 = horses[:6]
    specs = []

    def add(bet_type, roles, title=None):
        selected = [by_role.get(role) for role in roles]
        if all(selected):
            specs.append({
                "bet_type": bet_type,
                "roles": roles,
                "horses": selected,
                "title": title or bet_type,
            })

    if stars in ("★★★★★", "★★★★☆"):
        if race_type == "jra":
            add("tansho", ["◎"], "単勝")
            for roles in [("◎", "○"), ("◎", "▲"), ("○", "▲")]:
                add("umaren", list(roles), "馬連")
        else:
            for roles in [("◎", "○"), ("◎", "▲"), ("◎", "△1")]:
                add("wide", list(roles), "ワイド")
            if stars == "★★★★☆":
                for roles in [("◎", "○"), ("◎", "▲"), ("○", "▲")]:
                    add("umaren", list(roles), "馬連")
    elif stars == "★★★☆☆":
        for left in range(min(6, len(top6))):
            for right in range(left + 1, min(6, len(top6))):
                specs.append({"bet_type": "wide", "roles": [], "horses": [top6[left], top6[right]], "title": "ワイドBOX"})
    elif stars == "★★☆☆☆":
        for left in range(min(6, len(top6))):
            for right in range(left + 1, min(6, len(top6))):
                specs.append({"bet_type": "wide", "roles": [], "horses": [top6[left], top6[right]], "title": "ワイドBOX（少額）"})
    return specs


def _expectation_rating(odds_value):
    avg = _odds_average(odds_value)
    if avg is None:
        return "取得不可", ""
    if avg >= 10:
        return "★★★★★", ""
    if avg >= 6:
        return "★★★★☆", ""
    if avg >= 3.5:
        return "★★★☆☆", ""
    if avg >= 2:
        return "★★☆☆☆", "（妙味薄）"
    return "★☆☆☆☆", "（妙味薄）"




# ===== Ver2.5 Final UI: 単勝オッズ構成による配当妙味メモ（表示のみ） =====
def _single_odds_safe_float(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    text = str(value).strip().replace(",", "").replace("倍", "")
    if not text or text in {"-", "―", "nan", "None"}:
        return None
    match = re.search(r"[-+]?\d+(?:\.\d+)?", text)
    if not match:
        return None
    try:
        return float(match.group(0))
    except Exception:
        return None


def _single_odds_from_row(row):
    for column in ["オッズ", "単勝オッズ", "単勝", "単勝人気"]:
        if column in row.index:
            value = _single_odds_safe_float(row.get(column))
            if value is not None:
                return value
    return None


def _single_odds_format(value):
    value = _single_odds_safe_float(value)
    if value is None:
        return "-"
    return f"{value:.1f}".rstrip("0").rstrip(".")


def _single_odds_role_display(role):
    role_text = _nar_safe_text(role)
    return {"△1": "△①", "△2": "△②"}.get(role_text, role_text)


def _single_odds_marked_horses(df):
    if df is None or df.empty:
        return []
    source = df.copy()
    mark = source.get("最終印", pd.Series("", index=source.index)).fillna("").astype(str)
    source["_single_odds_mark"] = mark
    source["_single_odds_order"] = pd.to_numeric(
        source.get("_最終印順", pd.Series(99, index=source.index)),
        errors="coerce",
    )
    fallback_order = source["_single_odds_mark"].map({"◎": 0, "○": 1, "▲": 2, "△": 3, "✓": 5, "☆": 5})
    source["_single_odds_order"] = source["_single_odds_order"].where(
        source["_single_odds_order"].notna(), fallback_order
    ).fillna(99)
    source["_single_odds_score"] = pd.to_numeric(
        source.get("総合評価点", source.get("総合評価", source.get("_最終印点", source.get("AI点")))),
        errors="coerce",
    )
    marked = source[source["_single_odds_mark"].isin(["◎", "○", "▲", "△", "✓", "☆"])].copy()
    marked = marked.sort_values(["_single_odds_order", "_single_odds_score", "AI点"], ascending=[True, False, False])
    horses = []
    delta_count = 0
    circle_count = 0
    for _, row in marked.iterrows():
        mark_value = _nar_safe_text(row.get("_single_odds_mark"))
        if mark_value == "△":
            delta_count += 1
            role = f"△{delta_count}"
        elif mark_value == "○":
            circle_count += 1
            role = "○" if circle_count == 1 else f"○{circle_count}"
        else:
            role = mark_value
        horse_no = pd.to_numeric(row.get("馬番"), errors="coerce")
        horse_no_text = "" if pd.isna(horse_no) else str(int(horse_no))
        horses.append({
            "role": role,
            "mark": mark_value,
            "no": horse_no_text,
            "name": _nar_safe_text(row.get("馬名")),
            "odds": _single_odds_from_row(row),
        })
    return horses


def _single_odds_horse_text(horse, include_name=True):
    if not horse:
        return ""
    role = _single_odds_role_display(horse.get("role"))
    odds = _single_odds_format(horse.get("odds"))
    no = _nar_safe_text(horse.get("no"))
    name = _nar_safe_text(horse.get("name"))
    label = role
    if no:
        label += no
    if include_name and name:
        label += f" {name}"
    return f"{label} {odds}倍"


def _single_odds_combo_text(horses):
    return "／".join(_single_odds_horse_text(horse) for horse in horses if horse)


def _single_odds_max(horses):
    values = [_single_odds_safe_float(horse.get("odds")) for horse in horses if horse]
    values = [value for value in values if value is not None]
    return max(values) if values else None


def _single_odds_reference_rows(df, confidence_summary=None, race_type="nar"):
    horses = _single_odds_marked_horses(df)
    if not horses:
        return []
    by_role = {}
    for horse in horses:
        by_role.setdefault(horse["role"], horse)
    top6 = horses[:6]
    top3 = horses[:3]
    rows = []

    def add(ticket, roles, judgement, note=""):
        selected = []
        for role in roles:
            if isinstance(role, dict):
                selected.append(role)
            elif by_role.get(role):
                selected.append(by_role[role])
        if len(selected) != len(roles):
            return
        rows.append({
            "買い目": ticket,
            "単勝オッズ構成": _single_odds_combo_text(selected),
            "判定": judgement,
            "メモ": note,
        })

    if race_type == "jra":
        add("◎単勝", ["◎"], "購入検討")
        add("◎－○馬連", ["◎", "○"], "購入検討")
        if len(top3) >= 3:
            rows.append({
                "買い目": "馬連BOX印上位3頭",
                "単勝オッズ構成": _single_odds_combo_text(top3),
                "判定": "購入検討",
                "メモ": "",
            })
        add("◎－△②ワイド", ["◎", "△2"], "購入検討")
        add("◎→○馬単", ["◎", "○"], "高配当候補・少額向き")
        stars = (confidence_summary or {}).get("stars", "")
        venue_eval = (confidence_summary or {}).get("venue_eval", "")
        condition_eval = (confidence_summary or {}).get("condition_eval", "")
        if stars in {"★★★★★", "★★★★☆"} or venue_eval == "A" or condition_eval == "A":
            if len(top6) >= 6:
                rows.append({
                    "買い目": "3連複系",
                    "単勝オッズ構成": _single_odds_combo_text(top6),
                    "判定": "購入検討・点数負け注意",
                    "メモ": "AI信頼度または得意条件を確認",
                })
    else:
        wide_main = "妙味薄め"
        if by_role.get("◎") and by_role.get("○"):
            max_odds = _single_odds_max([by_role["◎"], by_role["○"]])
            if max_odds is not None and max_odds >= 10:
                wide_main = "購入検討"
        add("◎－○ワイド", ["◎", "○"], wide_main)
        add("◎－△①ワイド", ["◎", "△1"], "購入検討")
        add("◎－○馬連", ["◎", "○"], "購入検討・少額向き")
        add("◎→○馬単", ["◎", "○"], "高配当候補・少額向き")
        if len(top6) >= 6:
            rows.append({
                "買い目": "3連複BOX印6頭",
                "単勝オッズ構成": _single_odds_combo_text(top6),
                "判定": "点数負け注意",
                "メモ": "原則見送り寄り",
            })
    return rows


def print_single_odds_value_reference(df, confidence_summary=None, race_type="nar"):
    rows = _single_odds_reference_rows(df, confidence_summary, race_type)
    if not rows:
        return
    print("【配当妙味の参考（単勝オッズ構成）】")
    print("※表示専用。配当妙味の参考であり、断定ではありません。")
    value_df = pd.DataFrame(rows)
    try:
        display(format_result_for_output(value_df))
    except Exception:
        display(value_df)











# ===== Ver2.5 Final UI: 印同士の買い目ランキング（表示のみ） =====
import json as _ticket_ranking_json

_TICKET_RANKING_STATS = _ticket_ranking_json.loads(r'''{"pair_summary":[{"division":"地方","ticket":"◎－○","bet_type":"wide","n":83,"hits":24,"hit_rate":28.9,"roi":88.1,"avg_payout":304.6,"low_odds":4.1,"high_odds":16.1,"avg_win_odds":10.1,"median_win_odds":10.1},{"division":"地方","ticket":"◎－▲","bet_type":"wide","n":82,"hits":12,"hit_rate":14.6,"roi":49.6,"avg_payout":339.2,"low_odds":6.2,"high_odds":25.0,"avg_win_odds":15.6,"median_win_odds":15.6},{"division":"地方","ticket":"◎－△①","bet_type":"wide","n":83,"hits":16,"hit_rate":19.3,"roi":102.0,"avg_payout":529.4,"low_odds":5.8,"high_odds":27.5,"avg_win_odds":16.6,"median_win_odds":16.6},{"division":"地方","ticket":"◎－△②","bet_type":"wide","n":83,"hits":13,"hit_rate":15.7,"roi":74.6,"avg_payout":476.2,"low_odds":4.4,"high_odds":39.9,"avg_win_odds":22.2,"median_win_odds":22.2},{"division":"地方","ticket":"◎－☆","bet_type":"wide","n":81,"hits":6,"hit_rate":7.4,"roi":39.4,"avg_payout":531.7,"low_odds":4.6,"high_odds":47.5,"avg_win_odds":26.1,"median_win_odds":26.1},{"division":"地方","ticket":"○－▲","bet_type":"wide","n":90,"hits":12,"hit_rate":13.3,"roi":81.6,"avg_payout":611.7,"low_odds":6.6,"high_odds":28.6,"avg_win_odds":17.6,"median_win_odds":17.6},{"division":"地方","ticket":"○－△①","bet_type":"wide","n":91,"hits":15,"hit_rate":16.5,"roi":57.3,"avg_payout":347.3,"low_odds":7.6,"high_odds":29.2,"avg_win_odds":18.4,"median_win_odds":18.4},{"division":"地方","ticket":"○－△②","bet_type":"wide","n":91,"hits":11,"hit_rate":12.1,"roi":93.7,"avg_payout":775.5,"low_odds":6.8,"high_odds":47.1,"avg_win_odds":27.0,"median_win_odds":27.0},{"division":"地方","ticket":"○－☆","bet_type":"wide","n":89,"hits":11,"hit_rate":12.4,"roi":77.6,"avg_payout":628.2,"low_odds":7.3,"high_odds":48.2,"avg_win_odds":27.7,"median_win_odds":27.7},{"division":"地方","ticket":"▲－△①","bet_type":"wide","n":91,"hits":5,"hit_rate":5.5,"roi":46.6,"avg_payout":848.0,"low_odds":12.4,"high_odds":33.1,"avg_win_odds":22.7,"median_win_odds":22.7},{"division":"地方","ticket":"▲－△②","bet_type":"wide","n":91,"hits":5,"hit_rate":5.5,"roi":77.7,"avg_payout":1414.0,"low_odds":11.5,"high_odds":50.4,"avg_win_odds":30.9,"median_win_odds":30.9},{"division":"地方","ticket":"▲－☆","bet_type":"wide","n":89,"hits":7,"hit_rate":7.9,"roi":47.0,"avg_payout":597.1,"low_odds":12.3,"high_odds":49.2,"avg_win_odds":30.7,"median_win_odds":30.7},{"division":"地方","ticket":"△①－△②","bet_type":"wide","n":92,"hits":7,"hit_rate":7.6,"roi":62.8,"avg_payout":825.7,"low_odds":11.2,"high_odds":52.8,"avg_win_odds":32.0,"median_win_odds":32.0},{"division":"地方","ticket":"△①－☆","bet_type":"wide","n":90,"hits":6,"hit_rate":6.7,"roi":100.8,"avg_payout":1511.7,"low_odds":14.6,"high_odds":51.4,"avg_win_odds":33.0,"median_win_odds":33.0},{"division":"地方","ticket":"△②－☆","bet_type":"wide","n":90,"hits":4,"hit_rate":4.4,"roi":59.7,"avg_payout":1342.5,"low_odds":15.8,"high_odds":66.7,"avg_win_odds":41.3,"median_win_odds":41.3},{"division":"中央","ticket":"◎－○","bet_type":"wide","n":50,"hits":7,"hit_rate":14.0,"roi":54.2,"avg_payout":387.1,"low_odds":3.4,"high_odds":16.3,"avg_win_odds":9.9,"median_win_odds":9.9},{"division":"中央","ticket":"◎－▲","bet_type":"wide","n":50,"hits":8,"hit_rate":16.0,"roi":63.6,"avg_payout":397.5,"low_odds":4.8,"high_odds":25.1,"avg_win_odds":15.0,"median_win_odds":15.0},{"division":"中央","ticket":"◎－△①","bet_type":"wide","n":50,"hits":8,"hit_rate":16.0,"roi":96.0,"avg_payout":600.0,"low_odds":4.3,"high_odds":27.9,"avg_win_odds":16.1,"median_win_odds":16.1},{"division":"中央","ticket":"◎－△②","bet_type":"wide","n":50,"hits":7,"hit_rate":14.0,"roi":163.4,"avg_payout":1167.1,"low_odds":4.6,"high_odds":48.3,"avg_win_odds":26.4,"median_win_odds":26.4},{"division":"中央","ticket":"◎－☆","bet_type":"wide","n":49,"hits":4,"hit_rate":8.2,"roi":115.3,"avg_payout":1412.5,"low_odds":4.8,"high_odds":42.5,"avg_win_odds":23.7,"median_win_odds":23.7},{"division":"中央","ticket":"○－▲","bet_type":"wide","n":60,"hits":5,"hit_rate":8.3,"roi":61.7,"avg_payout":740.0,"low_odds":6.7,"high_odds":34.1,"avg_win_odds":20.4,"median_win_odds":20.4},{"division":"中央","ticket":"○－△①","bet_type":"wide","n":60,"hits":4,"hit_rate":6.7,"roi":48.7,"avg_payout":730.0,"low_odds":8.6,"high_odds":33.1,"avg_win_odds":20.9,"median_win_odds":20.9},{"division":"中央","ticket":"○－△②","bet_type":"wide","n":60,"hits":4,"hit_rate":6.7,"roi":94.3,"avg_payout":1415.0,"low_odds":7.4,"high_odds":52.5,"avg_win_odds":30.0,"median_win_odds":30.0},{"division":"中央","ticket":"○－☆","bet_type":"wide","n":59,"hits":4,"hit_rate":6.8,"roi":75.4,"avg_payout":1112.5,"low_odds":8.8,"high_odds":49.0,"avg_win_odds":28.9,"median_win_odds":28.9},{"division":"中央","ticket":"▲－△①","bet_type":"wide","n":60,"hits":2,"hit_rate":3.3,"roi":30.0,"avg_payout":900.0,"low_odds":10.1,"high_odds":36.9,"avg_win_odds":23.5,"median_win_odds":23.5},{"division":"中央","ticket":"▲－△②","bet_type":"wide","n":60,"hits":2,"hit_rate":3.3,"roi":13.7,"avg_payout":410.0,"low_odds":14.1,"high_odds":51.1,"avg_win_odds":32.6,"median_win_odds":32.6},{"division":"中央","ticket":"▲－☆","bet_type":"wide","n":59,"hits":5,"hit_rate":8.5,"roi":102.2,"avg_payout":1206.0,"low_odds":12.1,"high_odds":51.8,"avg_win_odds":31.9,"median_win_odds":31.9},{"division":"中央","ticket":"△①－△②","bet_type":"wide","n":60,"hits":4,"hit_rate":6.7,"roi":125.0,"avg_payout":1875.0,"low_odds":11.8,"high_odds":54.3,"avg_win_odds":33.0,"median_win_odds":33.0},{"division":"中央","ticket":"△①－☆","bet_type":"wide","n":59,"hits":1,"hit_rate":1.7,"roi":26.8,"avg_payout":1580.0,"low_odds":10.5,"high_odds":54.1,"avg_win_odds":32.3,"median_win_odds":32.3},{"division":"中央","ticket":"△②－☆","bet_type":"wide","n":59,"hits":2,"hit_rate":3.4,"roi":108.1,"avg_payout":3190.0,"low_odds":16.8,"high_odds":66.1,"avg_win_odds":41.4,"median_win_odds":41.4},{"division":"地方","ticket":"◎－○","bet_type":"quinella","n":83,"hits":14,"hit_rate":16.9,"roi":122.9,"avg_payout":728.6,"low_odds":4.1,"high_odds":16.1,"avg_win_odds":10.1,"median_win_odds":10.1},{"division":"地方","ticket":"◎－▲","bet_type":"quinella","n":82,"hits":6,"hit_rate":7.3,"roi":54.3,"avg_payout":741.7,"low_odds":6.2,"high_odds":25.0,"avg_win_odds":15.6,"median_win_odds":15.6},{"division":"地方","ticket":"◎－△①","bet_type":"quinella","n":83,"hits":5,"hit_rate":6.0,"roi":80.5,"avg_payout":1336.0,"low_odds":5.8,"high_odds":27.5,"avg_win_odds":16.6,"median_win_odds":16.6},{"division":"地方","ticket":"◎－△②","bet_type":"quinella","n":83,"hits":4,"hit_rate":4.8,"roi":92.4,"avg_payout":1917.5,"low_odds":4.4,"high_odds":39.9,"avg_win_odds":22.2,"median_win_odds":22.2},{"division":"地方","ticket":"◎－☆","bet_type":"quinella","n":81,"hits":3,"hit_rate":3.7,"roi":60.4,"avg_payout":1630.0,"low_odds":4.6,"high_odds":47.5,"avg_win_odds":26.1,"median_win_odds":26.1},{"division":"地方","ticket":"○－▲","bet_type":"quinella","n":90,"hits":6,"hit_rate":6.7,"roi":123.7,"avg_payout":1855.0,"low_odds":6.6,"high_odds":28.6,"avg_win_odds":17.6,"median_win_odds":17.6},{"division":"地方","ticket":"○－△①","bet_type":"quinella","n":91,"hits":2,"hit_rate":2.2,"roi":30.1,"avg_payout":1370.0,"low_odds":7.6,"high_odds":29.2,"avg_win_odds":18.4,"median_win_odds":18.4},{"division":"地方","ticket":"○－△②","bet_type":"quinella","n":91,"hits":2,"hit_rate":2.2,"roi":18.1,"avg_payout":825.0,"low_odds":6.8,"high_odds":47.1,"avg_win_odds":27.0,"median_win_odds":27.0},{"division":"地方","ticket":"○－☆","bet_type":"quinella","n":89,"hits":8,"hit_rate":9.0,"roi":119.7,"avg_payout":1331.2,"low_odds":7.3,"high_odds":48.2,"avg_win_odds":27.7,"median_win_odds":27.7},{"division":"地方","ticket":"▲－△①","bet_type":"quinella","n":91,"hits":3,"hit_rate":3.3,"roi":66.5,"avg_payout":2016.7,"low_odds":12.4,"high_odds":33.1,"avg_win_odds":22.7,"median_win_odds":22.7},{"division":"地方","ticket":"▲－△②","bet_type":"quinella","n":91,"hits":1,"hit_rate":1.1,"roi":13.5,"avg_payout":1230.0,"low_odds":11.5,"high_odds":50.4,"avg_win_odds":30.9,"median_win_odds":30.9},{"division":"地方","ticket":"▲－☆","bet_type":"quinella","n":89,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0,"low_odds":12.3,"high_odds":49.2,"avg_win_odds":30.7,"median_win_odds":30.7},{"division":"地方","ticket":"△①－△②","bet_type":"quinella","n":92,"hits":4,"hit_rate":4.3,"roi":45.3,"avg_payout":1042.5,"low_odds":11.2,"high_odds":52.8,"avg_win_odds":32.0,"median_win_odds":32.0},{"division":"地方","ticket":"△①－☆","bet_type":"quinella","n":90,"hits":2,"hit_rate":2.2,"roi":19.9,"avg_payout":895.0,"low_odds":14.6,"high_odds":51.4,"avg_win_odds":33.0,"median_win_odds":33.0},{"division":"地方","ticket":"△②－☆","bet_type":"quinella","n":90,"hits":1,"hit_rate":1.1,"roi":46.4,"avg_payout":4180.0,"low_odds":15.8,"high_odds":66.7,"avg_win_odds":41.3,"median_win_odds":41.3},{"division":"地方","ticket":"◎→○","bet_type":"exacta","n":83,"hits":9,"hit_rate":10.8,"roi":151.4,"avg_payout":1396.7,"low_odds":4.1,"high_odds":16.1,"avg_win_odds":10.1,"median_win_odds":10.1},{"division":"地方","ticket":"◎→▲","bet_type":"exacta","n":82,"hits":3,"hit_rate":3.7,"roi":68.7,"avg_payout":1876.7,"low_odds":6.2,"high_odds":25.0,"avg_win_odds":15.6,"median_win_odds":15.6},{"division":"地方","ticket":"◎→△①","bet_type":"exacta","n":83,"hits":1,"hit_rate":1.2,"roi":20.2,"avg_payout":1680.0,"low_odds":5.8,"high_odds":27.5,"avg_win_odds":16.6,"median_win_odds":16.6},{"division":"地方","ticket":"◎→△②","bet_type":"exacta","n":83,"hits":1,"hit_rate":1.2,"roi":13.5,"avg_payout":1120.0,"low_odds":4.4,"high_odds":39.9,"avg_win_odds":22.2,"median_win_odds":22.2},{"division":"地方","ticket":"◎→☆","bet_type":"exacta","n":81,"hits":3,"hit_rate":3.7,"roi":110.6,"avg_payout":2986.7,"low_odds":4.6,"high_odds":47.5,"avg_win_odds":26.1,"median_win_odds":26.1},{"division":"地方","ticket":"○→◎","bet_type":"exacta","n":83,"hits":5,"hit_rate":6.0,"roi":40.5,"avg_payout":672.0,"low_odds":4.1,"high_odds":16.1,"avg_win_odds":10.1,"median_win_odds":10.1},{"division":"地方","ticket":"○→▲","bet_type":"exacta","n":90,"hits":5,"hit_rate":5.6,"roi":65.2,"avg_payout":1174.0,"low_odds":6.6,"high_odds":28.6,"avg_win_odds":17.6,"median_win_odds":17.6},{"division":"地方","ticket":"○→△①","bet_type":"exacta","n":91,"hits":2,"hit_rate":2.2,"roi":38.8,"avg_payout":1765.0,"low_odds":7.6,"high_odds":29.2,"avg_win_odds":18.4,"median_win_odds":18.4},{"division":"地方","ticket":"○→△②","bet_type":"exacta","n":91,"hits":1,"hit_rate":1.1,"roi":9.9,"avg_payout":900.0,"low_odds":6.8,"high_odds":47.1,"avg_win_odds":27.0,"median_win_odds":27.0},{"division":"地方","ticket":"○→☆","bet_type":"exacta","n":89,"hits":4,"hit_rate":4.5,"roi":72.8,"avg_payout":1620.0,"low_odds":7.3,"high_odds":48.2,"avg_win_odds":27.7,"median_win_odds":27.7},{"division":"地方","ticket":"▲→◎","bet_type":"exacta","n":82,"hits":3,"hit_rate":3.7,"roi":27.2,"avg_payout":743.3,"low_odds":6.2,"high_odds":25.0,"avg_win_odds":15.6,"median_win_odds":15.6},{"division":"地方","ticket":"▲→○","bet_type":"exacta","n":90,"hits":1,"hit_rate":1.1,"roi":345.6,"avg_payout":31100.0,"low_odds":6.6,"high_odds":28.6,"avg_win_odds":17.6,"median_win_odds":17.6},{"division":"地方","ticket":"▲→△①","bet_type":"exacta","n":91,"hits":2,"hit_rate":2.2,"roi":65.2,"avg_payout":2965.0,"low_odds":12.4,"high_odds":33.1,"avg_win_odds":22.7,"median_win_odds":22.7},{"division":"地方","ticket":"▲→△②","bet_type":"exacta","n":91,"hits":1,"hit_rate":1.1,"roi":21.4,"avg_payout":1950.0,"low_odds":11.5,"high_odds":50.4,"avg_win_odds":30.9,"median_win_odds":30.9},{"division":"地方","ticket":"▲→☆","bet_type":"exacta","n":89,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0,"low_odds":12.3,"high_odds":49.2,"avg_win_odds":30.7,"median_win_odds":30.7},{"division":"地方","ticket":"△①→◎","bet_type":"exacta","n":83,"hits":4,"hit_rate":4.8,"roi":132.4,"avg_payout":2747.5,"low_odds":5.8,"high_odds":27.5,"avg_win_odds":16.6,"median_win_odds":16.6},{"division":"地方","ticket":"△①→○","bet_type":"exacta","n":91,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0,"low_odds":7.6,"high_odds":29.2,"avg_win_odds":18.4,"median_win_odds":18.4},{"division":"地方","ticket":"△①→▲","bet_type":"exacta","n":91,"hits":1,"hit_rate":1.1,"roi":32.0,"avg_payout":2910.0,"low_odds":12.4,"high_odds":33.1,"avg_win_odds":22.7,"median_win_odds":22.7},{"division":"地方","ticket":"△①→△②","bet_type":"exacta","n":92,"hits":1,"hit_rate":1.1,"roi":18.3,"avg_payout":1680.0,"low_odds":11.2,"high_odds":52.8,"avg_win_odds":32.0,"median_win_odds":32.0},{"division":"地方","ticket":"△①→☆","bet_type":"exacta","n":90,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0,"low_odds":14.6,"high_odds":51.4,"avg_win_odds":33.0,"median_win_odds":33.0},{"division":"地方","ticket":"△②→◎","bet_type":"exacta","n":83,"hits":3,"hit_rate":3.6,"roi":149.6,"avg_payout":4140.0,"low_odds":4.4,"high_odds":39.9,"avg_win_odds":22.2,"median_win_odds":22.2},{"division":"地方","ticket":"△②→○","bet_type":"exacta","n":91,"hits":1,"hit_rate":1.1,"roi":33.1,"avg_payout":3010.0,"low_odds":6.8,"high_odds":47.1,"avg_win_odds":27.0,"median_win_odds":27.0},{"division":"地方","ticket":"△②→▲","bet_type":"exacta","n":91,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0,"low_odds":11.5,"high_odds":50.4,"avg_win_odds":30.9,"median_win_odds":30.9},{"division":"地方","ticket":"△②→△①","bet_type":"exacta","n":92,"hits":3,"hit_rate":3.3,"roi":58.8,"avg_payout":1803.3,"low_odds":11.2,"high_odds":52.8,"avg_win_odds":32.0,"median_win_odds":32.0},{"division":"地方","ticket":"△②→☆","bet_type":"exacta","n":90,"hits":1,"hit_rate":1.1,"roi":73.2,"avg_payout":6590.0,"low_odds":15.8,"high_odds":66.7,"avg_win_odds":41.3,"median_win_odds":41.3},{"division":"地方","ticket":"☆→◎","bet_type":"exacta","n":81,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0,"low_odds":4.6,"high_odds":47.5,"avg_win_odds":26.1,"median_win_odds":26.1},{"division":"地方","ticket":"☆→○","bet_type":"exacta","n":89,"hits":4,"hit_rate":4.5,"roi":191.1,"avg_payout":4252.5,"low_odds":7.3,"high_odds":48.2,"avg_win_odds":27.7,"median_win_odds":27.7},{"division":"地方","ticket":"☆→▲","bet_type":"exacta","n":89,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0,"low_odds":12.3,"high_odds":49.2,"avg_win_odds":30.7,"median_win_odds":30.7},{"division":"地方","ticket":"☆→△①","bet_type":"exacta","n":90,"hits":2,"hit_rate":2.2,"roi":34.3,"avg_payout":1545.0,"low_odds":14.6,"high_odds":51.4,"avg_win_odds":33.0,"median_win_odds":33.0},{"division":"地方","ticket":"☆→△②","bet_type":"exacta","n":90,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0,"low_odds":15.8,"high_odds":66.7,"avg_win_odds":41.3,"median_win_odds":41.3},{"division":"中央","ticket":"◎－○","bet_type":"quinella","n":50,"hits":6,"hit_rate":12.0,"roi":142.2,"avg_payout":1185.0,"low_odds":3.4,"high_odds":16.3,"avg_win_odds":9.9,"median_win_odds":9.9},{"division":"中央","ticket":"◎－▲","bet_type":"quinella","n":50,"hits":5,"hit_rate":10.0,"roi":76.8,"avg_payout":768.0,"low_odds":4.8,"high_odds":25.1,"avg_win_odds":15.0,"median_win_odds":15.0},{"division":"中央","ticket":"◎－△①","bet_type":"quinella","n":50,"hits":6,"hit_rate":12.0,"roi":166.6,"avg_payout":1388.3,"low_odds":4.3,"high_odds":27.9,"avg_win_odds":16.1,"median_win_odds":16.1},{"division":"中央","ticket":"◎－△②","bet_type":"quinella","n":50,"hits":2,"hit_rate":4.0,"roi":127.0,"avg_payout":3175.0,"low_odds":4.6,"high_odds":48.3,"avg_win_odds":26.4,"median_win_odds":26.4},{"division":"中央","ticket":"◎－☆","bet_type":"quinella","n":49,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0,"low_odds":4.8,"high_odds":42.5,"avg_win_odds":23.7,"median_win_odds":23.7},{"division":"中央","ticket":"○－▲","bet_type":"quinella","n":60,"hits":3,"hit_rate":5.0,"roi":160.8,"avg_payout":3216.7,"low_odds":6.7,"high_odds":34.1,"avg_win_odds":20.4,"median_win_odds":20.4},{"division":"中央","ticket":"○－△①","bet_type":"quinella","n":60,"hits":2,"hit_rate":3.3,"roi":22.2,"avg_payout":665.0,"low_odds":8.6,"high_odds":33.1,"avg_win_odds":20.9,"median_win_odds":20.9},{"division":"中央","ticket":"○－△②","bet_type":"quinella","n":60,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0,"low_odds":7.4,"high_odds":52.5,"avg_win_odds":30.0,"median_win_odds":30.0},{"division":"中央","ticket":"○－☆","bet_type":"quinella","n":59,"hits":1,"hit_rate":1.7,"roi":24.2,"avg_payout":1430.0,"low_odds":8.8,"high_odds":49.0,"avg_win_odds":28.9,"median_win_odds":28.9},{"division":"中央","ticket":"▲－△①","bet_type":"quinella","n":60,"hits":1,"hit_rate":1.7,"roi":16.0,"avg_payout":960.0,"low_odds":10.1,"high_odds":36.9,"avg_win_odds":23.5,"median_win_odds":23.5},{"division":"中央","ticket":"▲－△②","bet_type":"quinella","n":60,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0,"low_odds":14.1,"high_odds":51.1,"avg_win_odds":32.6,"median_win_odds":32.6},{"division":"中央","ticket":"▲－☆","bet_type":"quinella","n":59,"hits":2,"hit_rate":3.4,"roi":121.0,"avg_payout":3570.0,"low_odds":12.1,"high_odds":51.8,"avg_win_odds":31.9,"median_win_odds":31.9},{"division":"中央","ticket":"△①－△②","bet_type":"quinella","n":60,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0,"low_odds":11.8,"high_odds":54.3,"avg_win_odds":33.0,"median_win_odds":33.0},{"division":"中央","ticket":"△①－☆","bet_type":"quinella","n":59,"hits":1,"hit_rate":1.7,"roi":90.5,"avg_payout":5340.0,"low_odds":10.5,"high_odds":54.1,"avg_win_odds":32.3,"median_win_odds":32.3},{"division":"中央","ticket":"△②－☆","bet_type":"quinella","n":59,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0,"low_odds":16.8,"high_odds":66.1,"avg_win_odds":41.4,"median_win_odds":41.4},{"division":"中央","ticket":"◎→○","bet_type":"exacta","n":50,"hits":5,"hit_rate":10.0,"roi":247.6,"avg_payout":2476.0,"low_odds":3.4,"high_odds":16.3,"avg_win_odds":9.9,"median_win_odds":9.9},{"division":"中央","ticket":"◎→▲","bet_type":"exacta","n":50,"hits":4,"hit_rate":8.0,"roi":119.0,"avg_payout":1487.5,"low_odds":4.8,"high_odds":25.1,"avg_win_odds":15.0,"median_win_odds":15.0},{"division":"中央","ticket":"◎→△①","bet_type":"exacta","n":50,"hits":3,"hit_rate":6.0,"roi":122.6,"avg_payout":2043.3,"low_odds":4.3,"high_odds":27.9,"avg_win_odds":16.1,"median_win_odds":16.1},{"division":"中央","ticket":"◎→△②","bet_type":"exacta","n":50,"hits":2,"hit_rate":4.0,"roi":219.8,"avg_payout":5495.0,"low_odds":4.6,"high_odds":48.3,"avg_win_odds":26.4,"median_win_odds":26.4},{"division":"中央","ticket":"◎→☆","bet_type":"exacta","n":49,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0,"low_odds":4.8,"high_odds":42.5,"avg_win_odds":23.7,"median_win_odds":23.7},{"division":"中央","ticket":"○→◎","bet_type":"exacta","n":50,"hits":1,"hit_rate":2.0,"roi":17.8,"avg_payout":890.0,"low_odds":3.4,"high_odds":16.3,"avg_win_odds":9.9,"median_win_odds":9.9},{"division":"中央","ticket":"○→▲","bet_type":"exacta","n":60,"hits":1,"hit_rate":1.7,"roi":189.3,"avg_payout":11360.0,"low_odds":6.7,"high_odds":34.1,"avg_win_odds":20.4,"median_win_odds":20.4},{"division":"中央","ticket":"○→△①","bet_type":"exacta","n":60,"hits":2,"hit_rate":3.3,"roi":39.2,"avg_payout":1175.0,"low_odds":8.6,"high_odds":33.1,"avg_win_odds":20.9,"median_win_odds":20.9},{"division":"中央","ticket":"○→△②","bet_type":"exacta","n":60,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0,"low_odds":7.4,"high_odds":52.5,"avg_win_odds":30.0,"median_win_odds":30.0},{"division":"中央","ticket":"○→☆","bet_type":"exacta","n":59,"hits":1,"hit_rate":1.7,"roi":48.6,"avg_payout":2870.0,"low_odds":8.8,"high_odds":49.0,"avg_win_odds":28.9,"median_win_odds":28.9},{"division":"中央","ticket":"▲→◎","bet_type":"exacta","n":50,"hits":1,"hit_rate":2.0,"roi":27.2,"avg_payout":1360.0,"low_odds":4.8,"high_odds":25.1,"avg_win_odds":15.0,"median_win_odds":15.0},{"division":"中央","ticket":"▲→○","bet_type":"exacta","n":60,"hits":2,"hit_rate":3.3,"roi":97.8,"avg_payout":2935.0,"low_odds":6.7,"high_odds":34.1,"avg_win_odds":20.4,"median_win_odds":20.4},{"division":"中央","ticket":"▲→△①","bet_type":"exacta","n":60,"hits":1,"hit_rate":1.7,"roi":25.7,"avg_payout":1540.0,"low_odds":10.1,"high_odds":36.9,"avg_win_odds":23.5,"median_win_odds":23.5},{"division":"中央","ticket":"▲→△②","bet_type":"exacta","n":60,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0,"low_odds":14.1,"high_odds":51.1,"avg_win_odds":32.6,"median_win_odds":32.6},{"division":"中央","ticket":"▲→☆","bet_type":"exacta","n":59,"hits":2,"hit_rate":3.4,"roi":309.5,"avg_payout":9130.0,"low_odds":12.1,"high_odds":51.8,"avg_win_odds":31.9,"median_win_odds":31.9},{"division":"中央","ticket":"△①→◎","bet_type":"exacta","n":50,"hits":3,"hit_rate":6.0,"roi":216.0,"avg_payout":3600.0,"low_odds":4.3,"high_odds":27.9,"avg_win_odds":16.1,"median_win_odds":16.1},{"division":"中央","ticket":"△①→○","bet_type":"exacta","n":60,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0,"low_odds":8.6,"high_odds":33.1,"avg_win_odds":20.9,"median_win_odds":20.9},{"division":"中央","ticket":"△①→▲","bet_type":"exacta","n":60,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0,"low_odds":10.1,"high_odds":36.9,"avg_win_odds":23.5,"median_win_odds":23.5},{"division":"中央","ticket":"△①→△②","bet_type":"exacta","n":60,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0,"low_odds":11.8,"high_odds":54.3,"avg_win_odds":33.0,"median_win_odds":33.0},{"division":"中央","ticket":"△①→☆","bet_type":"exacta","n":59,"hits":1,"hit_rate":1.7,"roi":183.9,"avg_payout":10850.0,"low_odds":10.5,"high_odds":54.1,"avg_win_odds":32.3,"median_win_odds":32.3},{"division":"中央","ticket":"△②→◎","bet_type":"exacta","n":50,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0,"low_odds":4.6,"high_odds":48.3,"avg_win_odds":26.4,"median_win_odds":26.4},{"division":"中央","ticket":"△②→○","bet_type":"exacta","n":60,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0,"low_odds":7.4,"high_odds":52.5,"avg_win_odds":30.0,"median_win_odds":30.0},{"division":"中央","ticket":"△②→▲","bet_type":"exacta","n":60,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0,"low_odds":14.1,"high_odds":51.1,"avg_win_odds":32.6,"median_win_odds":32.6},{"division":"中央","ticket":"△②→△①","bet_type":"exacta","n":60,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0,"low_odds":11.8,"high_odds":54.3,"avg_win_odds":33.0,"median_win_odds":33.0},{"division":"中央","ticket":"△②→☆","bet_type":"exacta","n":59,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0,"low_odds":16.8,"high_odds":66.1,"avg_win_odds":41.4,"median_win_odds":41.4},{"division":"中央","ticket":"☆→◎","bet_type":"exacta","n":49,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0,"low_odds":4.8,"high_odds":42.5,"avg_win_odds":23.7,"median_win_odds":23.7},{"division":"中央","ticket":"☆→○","bet_type":"exacta","n":59,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0,"low_odds":8.8,"high_odds":49.0,"avg_win_odds":28.9,"median_win_odds":28.9},{"division":"中央","ticket":"☆→▲","bet_type":"exacta","n":59,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0,"low_odds":12.1,"high_odds":51.8,"avg_win_odds":31.9,"median_win_odds":31.9},{"division":"中央","ticket":"☆→△①","bet_type":"exacta","n":59,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0,"low_odds":10.5,"high_odds":54.1,"avg_win_odds":32.3,"median_win_odds":32.3},{"division":"中央","ticket":"☆→△②","bet_type":"exacta","n":59,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0,"low_odds":16.8,"high_odds":66.1,"avg_win_odds":41.4,"median_win_odds":41.4}],"pair_band_summary":[{"division":"地方","ticket":"◎－○","bet_type":"wide","band":"両馬5倍未満","n":21,"hits":9,"hit_rate":42.9,"roi":77.6,"avg_payout":181.1},{"division":"地方","ticket":"◎－▲","bet_type":"wide","band":"片方が10～19.9倍","n":21,"hits":4,"hit_rate":19.0,"roi":94.8,"avg_payout":497.5},{"division":"地方","ticket":"◎－△①","bet_type":"wide","band":"片方が50倍以上","n":11,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎－△②","bet_type":"wide","band":"両馬5倍未満","n":6,"hits":2,"hit_rate":33.3,"roi":71.7,"avg_payout":215.0},{"division":"地方","ticket":"◎－☆","bet_type":"wide","band":"片方が50倍以上","n":22,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"○－▲","bet_type":"wide","band":"片方が10～19.9倍","n":29,"hits":2,"hit_rate":6.9,"roi":63.4,"avg_payout":920.0},{"division":"地方","ticket":"○－△①","bet_type":"wide","band":"片方が50倍以上","n":14,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"○－△②","bet_type":"wide","band":"両馬5倍未満","n":5,"hits":2,"hit_rate":40.0,"roi":106.0,"avg_payout":265.0},{"division":"地方","ticket":"○－☆","bet_type":"wide","band":"片方が50倍以上","n":26,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"▲－△①","bet_type":"wide","band":"片方が10～19.9倍","n":35,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"▲－△①","bet_type":"wide","band":"片方が50倍以上","n":17,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"▲－△①","bet_type":"wide","band":"両馬10倍以上","n":27,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"▲－△②","bet_type":"wide","band":"片方が10～19.9倍","n":33,"hits":2,"hit_rate":6.1,"roi":23.6,"avg_payout":390.0},{"division":"地方","ticket":"▲－☆","bet_type":"wide","band":"片方が10～19.9倍","n":31,"hits":3,"hit_rate":9.7,"roi":56.1,"avg_payout":580.0},{"division":"地方","ticket":"▲－☆","bet_type":"wide","band":"片方が50倍以上","n":27,"hits":1,"hit_rate":3.7,"roi":50.0,"avg_payout":1350.0},{"division":"地方","ticket":"▲－☆","bet_type":"wide","band":"両馬10倍以上","n":31,"hits":1,"hit_rate":3.2,"roi":21.6,"avg_payout":670.0},{"division":"地方","ticket":"△①－△②","bet_type":"wide","band":"片方が50倍以上","n":31,"hits":1,"hit_rate":3.2,"roi":76.5,"avg_payout":2370.0},{"division":"地方","ticket":"△①－☆","bet_type":"wide","band":"片方が50倍以上","n":26,"hits":1,"hit_rate":3.8,"roi":257.3,"avg_payout":6690.0},{"division":"地方","ticket":"△①－☆","bet_type":"wide","band":"両馬10倍以上","n":35,"hits":1,"hit_rate":2.9,"roi":191.1,"avg_payout":6690.0},{"division":"地方","ticket":"△②－☆","bet_type":"wide","band":"片方が50倍以上","n":39,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎－▲","bet_type":"wide","band":"両馬5倍未満","n":13,"hits":4,"hit_rate":30.8,"roi":56.9,"avg_payout":185.0},{"division":"地方","ticket":"◎－△①","bet_type":"wide","band":"片方が20～49.9倍","n":20,"hits":4,"hit_rate":20.0,"roi":203.5,"avg_payout":1017.5},{"division":"地方","ticket":"◎－△②","bet_type":"wide","band":"片方が5～9.9倍","n":36,"hits":6,"hit_rate":16.7,"roi":84.4,"avg_payout":506.7},{"division":"地方","ticket":"○－▲","bet_type":"wide","band":"両馬5倍未満","n":9,"hits":2,"hit_rate":22.2,"roi":32.2,"avg_payout":145.0},{"division":"地方","ticket":"○－△①","bet_type":"wide","band":"片方が20～49.9倍","n":29,"hits":3,"hit_rate":10.3,"roi":37.9,"avg_payout":366.7},{"division":"地方","ticket":"○－△②","bet_type":"wide","band":"片方が5～9.9倍","n":32,"hits":4,"hit_rate":12.5,"roi":57.2,"avg_payout":457.5},{"division":"地方","ticket":"▲－△①","bet_type":"wide","band":"片方が20～49.9倍","n":34,"hits":3,"hit_rate":8.8,"roi":100.0,"avg_payout":1133.3},{"division":"地方","ticket":"▲－△②","bet_type":"wide","band":"片方が5～9.9倍","n":41,"hits":2,"hit_rate":4.9,"roi":147.8,"avg_payout":3030.0},{"division":"地方","ticket":"△①－△②","bet_type":"wide","band":"片方が5～9.9倍","n":44,"hits":4,"hit_rate":9.1,"roi":41.8,"avg_payout":460.0},{"division":"地方","ticket":"△①－△②","bet_type":"wide","band":"片方が20～49.9倍","n":29,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△①－☆","bet_type":"wide","band":"片方が20～49.9倍","n":38,"hits":1,"hit_rate":2.6,"roi":176.1,"avg_payout":6690.0},{"division":"地方","ticket":"△②－☆","bet_type":"wide","band":"片方が5～9.9倍","n":33,"hits":2,"hit_rate":6.1,"roi":119.1,"avg_payout":1965.0},{"division":"地方","ticket":"◎－△①","bet_type":"wide","band":"片方が5～9.9倍","n":42,"hits":7,"hit_rate":16.7,"roi":71.4,"avg_payout":428.6},{"division":"地方","ticket":"◎－△②","bet_type":"wide","band":"片方が50倍以上","n":23,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"○－△①","bet_type":"wide","band":"片方が5～9.9倍","n":41,"hits":8,"hit_rate":19.5,"roi":70.5,"avg_payout":361.2},{"division":"地方","ticket":"○－△②","bet_type":"wide","band":"片方が50倍以上","n":27,"hits":2,"hit_rate":7.4,"roi":175.2,"avg_payout":2365.0},{"division":"地方","ticket":"▲－△①","bet_type":"wide","band":"片方が5～9.9倍","n":44,"hits":1,"hit_rate":2.3,"roi":13.6,"avg_payout":600.0},{"division":"地方","ticket":"▲－△②","bet_type":"wide","band":"片方が50倍以上","n":30,"hits":1,"hit_rate":3.3,"roi":185.7,"avg_payout":5570.0},{"division":"地方","ticket":"△①－☆","bet_type":"wide","band":"片方が5～9.9倍","n":40,"hits":4,"hit_rate":10.0,"roi":40.2,"avg_payout":402.5},{"division":"地方","ticket":"△②－☆","bet_type":"wide","band":"両馬10倍以上","n":38,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎－△①","bet_type":"wide","band":"片方が10～19.9倍","n":23,"hits":3,"hit_rate":13.0,"roi":77.0,"avg_payout":590.0},{"division":"地方","ticket":"◎－△②","bet_type":"wide","band":"片方が10～19.9倍","n":22,"hits":6,"hit_rate":27.3,"roi":161.8,"avg_payout":593.3},{"division":"地方","ticket":"◎－☆","bet_type":"wide","band":"片方が20～49.9倍","n":23,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"○－△①","bet_type":"wide","band":"片方が10～19.9倍","n":31,"hits":3,"hit_rate":9.7,"roi":58.4,"avg_payout":603.3},{"division":"地方","ticket":"○－△②","bet_type":"wide","band":"片方が10～19.9倍","n":31,"hits":3,"hit_rate":9.7,"roi":56.5,"avg_payout":583.3},{"division":"地方","ticket":"○－☆","bet_type":"wide","band":"片方が20～49.9倍","n":33,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"▲－☆","bet_type":"wide","band":"片方が20～49.9倍","n":38,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△①－△②","bet_type":"wide","band":"片方が10～19.9倍","n":31,"hits":3,"hit_rate":9.7,"roi":127.1,"avg_payout":1313.3},{"division":"地方","ticket":"△①－△②","bet_type":"wide","band":"両馬10倍以上","n":32,"hits":1,"hit_rate":3.1,"roi":74.1,"avg_payout":2370.0},{"division":"地方","ticket":"△①－☆","bet_type":"wide","band":"片方が10～19.9倍","n":34,"hits":1,"hit_rate":2.9,"roi":22.6,"avg_payout":770.0},{"division":"地方","ticket":"△②－☆","bet_type":"wide","band":"片方が10～19.9倍","n":35,"hits":1,"hit_rate":2.9,"roi":27.7,"avg_payout":970.0},{"division":"地方","ticket":"△②－☆","bet_type":"wide","band":"片方が20～49.9倍","n":36,"hits":3,"hit_rate":8.3,"roi":122.2,"avg_payout":1466.7},{"division":"地方","ticket":"○－▲","bet_type":"wide","band":"片方が50倍以上","n":14,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"○－▲","bet_type":"wide","band":"両馬10倍以上","n":15,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"○－△②","bet_type":"wide","band":"両馬10倍以上","n":15,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"○－☆","bet_type":"wide","band":"片方が10～19.9倍","n":31,"hits":5,"hit_rate":16.1,"roi":162.9,"avg_payout":1010.0},{"division":"地方","ticket":"○－☆","bet_type":"wide","band":"両馬10倍以上","n":21,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"▲－△②","bet_type":"wide","band":"両馬10倍以上","n":29,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎－○","bet_type":"wide","band":"片方が10～19.9倍","n":16,"hits":5,"hit_rate":31.2,"roi":188.1,"avg_payout":602.0},{"division":"地方","ticket":"◎－▲","bet_type":"wide","band":"片方が20～49.9倍","n":18,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎－▲","bet_type":"wide","band":"両馬10倍以上","n":6,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎－△①","bet_type":"wide","band":"両馬10倍以上","n":5,"hits":1,"hit_rate":20.0,"roi":216.0,"avg_payout":1080.0},{"division":"地方","ticket":"◎－△②","bet_type":"wide","band":"片方が20～49.9倍","n":15,"hits":1,"hit_rate":6.7,"roi":29.3,"avg_payout":440.0},{"division":"地方","ticket":"◎－△②","bet_type":"wide","band":"両馬10倍以上","n":5,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎－☆","bet_type":"wide","band":"片方が10～19.9倍","n":20,"hits":2,"hit_rate":10.0,"roi":74.0,"avg_payout":740.0},{"division":"地方","ticket":"○－▲","bet_type":"wide","band":"片方が20～49.9倍","n":27,"hits":3,"hit_rate":11.1,"roi":115.9,"avg_payout":1043.3},{"division":"地方","ticket":"○－△②","bet_type":"wide","band":"片方が20～49.9倍","n":23,"hits":1,"hit_rate":4.3,"roi":18.7,"avg_payout":430.0},{"division":"地方","ticket":"○－☆","bet_type":"wide","band":"両馬5倍未満","n":4,"hits":3,"hit_rate":75.0,"roi":137.5,"avg_payout":183.3},{"division":"地方","ticket":"▲－△②","bet_type":"wide","band":"片方が20～49.9倍","n":28,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎－○","bet_type":"wide","band":"片方が5～9.9倍","n":37,"hits":10,"hit_rate":27.0,"roi":92.2,"avg_payout":341.0},{"division":"地方","ticket":"◎－▲","bet_type":"wide","band":"片方が5～9.9倍","n":38,"hits":4,"hit_rate":10.5,"roi":35.3,"avg_payout":335.0},{"division":"地方","ticket":"◎－△①","bet_type":"wide","band":"両馬5倍未満","n":6,"hits":3,"hit_rate":50.0,"roi":118.3,"avg_payout":236.7},{"division":"地方","ticket":"◎－☆","bet_type":"wide","band":"両馬5倍未満","n":2,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"○－▲","bet_type":"wide","band":"片方が5～9.9倍","n":34,"hits":6,"hit_rate":17.6,"roi":103.5,"avg_payout":586.7},{"division":"地方","ticket":"○－☆","bet_type":"wide","band":"片方が5～9.9倍","n":27,"hits":4,"hit_rate":14.8,"roi":117.0,"avg_payout":790.0},{"division":"地方","ticket":"▲－☆","bet_type":"wide","band":"片方が5～9.9倍","n":32,"hits":3,"hit_rate":9.4,"roi":40.3,"avg_payout":430.0},{"division":"地方","ticket":"△①－☆","bet_type":"wide","band":"両馬5倍未満","n":1,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎－☆","bet_type":"wide","band":"片方が5～9.9倍","n":32,"hits":4,"hit_rate":12.5,"roi":53.4,"avg_payout":427.5},{"division":"地方","ticket":"○－△①","bet_type":"wide","band":"両馬5倍未満","n":5,"hits":3,"hit_rate":60.0,"roi":130.0,"avg_payout":216.7},{"division":"地方","ticket":"◎－○","bet_type":"wide","band":"片方が50倍以上","n":5,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"○－△①","bet_type":"wide","band":"両馬10倍以上","n":18,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎－○","bet_type":"wide","band":"片方が20～49.9倍","n":11,"hits":2,"hit_rate":18.2,"roi":55.5,"avg_payout":305.0},{"division":"地方","ticket":"◎－▲","bet_type":"wide","band":"片方が50倍以上","n":10,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"▲－☆","bet_type":"wide","band":"両馬5倍未満","n":3,"hits":1,"hit_rate":33.3,"roi":113.3,"avg_payout":340.0},{"division":"地方","ticket":"▲－△②","bet_type":"wide","band":"両馬5倍未満","n":1,"hits":1,"hit_rate":100.0,"roi":230.0,"avg_payout":230.0},{"division":"地方","ticket":"◎－☆","bet_type":"wide","band":"両馬10倍以上","n":5,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎－○","bet_type":"wide","band":"両馬10倍以上","n":2,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"▲－△①","bet_type":"wide","band":"両馬5倍未満","n":2,"hits":1,"hit_rate":50.0,"roi":120.0,"avg_payout":240.0},{"division":"地方","ticket":"△②－☆","bet_type":"wide","band":"両馬5倍未満","n":2,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△①－△②","bet_type":"wide","band":"両馬5倍未満","n":2,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎－○","bet_type":"wide","band":"片方が5～9.9倍","n":28,"hits":3,"hit_rate":10.7,"roi":54.6,"avg_payout":510.0},{"division":"中央","ticket":"◎－▲","bet_type":"wide","band":"両馬5倍未満","n":8,"hits":4,"hit_rate":50.0,"roi":135.0,"avg_payout":270.0},{"division":"中央","ticket":"◎－△①","bet_type":"wide","band":"片方が20～49.9倍","n":10,"hits":1,"hit_rate":10.0,"roi":138.0,"avg_payout":1380.0},{"division":"中央","ticket":"◎－△②","bet_type":"wide","band":"片方が50倍以上","n":12,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎－☆","bet_type":"wide","band":"片方が5～9.9倍","n":19,"hits":3,"hit_rate":15.8,"roi":271.6,"avg_payout":1720.0},{"division":"中央","ticket":"○－▲","bet_type":"wide","band":"片方が5～9.9倍","n":32,"hits":4,"hit_rate":12.5,"roi":108.7,"avg_payout":870.0},{"division":"中央","ticket":"○－△①","bet_type":"wide","band":"片方が5～9.9倍","n":30,"hits":3,"hit_rate":10.0,"roi":89.0,"avg_payout":890.0},{"division":"中央","ticket":"○－△①","bet_type":"wide","band":"片方が20～49.9倍","n":12,"hits":1,"hit_rate":8.3,"roi":102.5,"avg_payout":1230.0},{"division":"中央","ticket":"○－△②","bet_type":"wide","band":"片方が5～9.9倍","n":24,"hits":2,"hit_rate":8.3,"roi":164.2,"avg_payout":1970.0},{"division":"中央","ticket":"○－△②","bet_type":"wide","band":"片方が50倍以上","n":18,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○－☆","bet_type":"wide","band":"片方が5～9.9倍","n":28,"hits":4,"hit_rate":14.3,"roi":158.9,"avg_payout":1112.5},{"division":"中央","ticket":"▲－△①","bet_type":"wide","band":"片方が20～49.9倍","n":20,"hits":1,"hit_rate":5.0,"roi":68.0,"avg_payout":1360.0},{"division":"中央","ticket":"▲－△②","bet_type":"wide","band":"片方が50倍以上","n":16,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"▲－☆","bet_type":"wide","band":"片方が5～9.9倍","n":25,"hits":4,"hit_rate":16.0,"roi":102.4,"avg_payout":640.0},{"division":"中央","ticket":"△①－△②","bet_type":"wide","band":"片方が20～49.9倍","n":24,"hits":2,"hit_rate":8.3,"roi":292.5,"avg_payout":3510.0},{"division":"中央","ticket":"△①－△②","bet_type":"wide","band":"片方が50倍以上","n":16,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△①－△②","bet_type":"wide","band":"両馬10倍以上","n":19,"hits":1,"hit_rate":5.3,"roi":285.8,"avg_payout":5430.0},{"division":"中央","ticket":"△①－☆","bet_type":"wide","band":"片方が5～9.9倍","n":26,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△①－☆","bet_type":"wide","band":"片方が20～49.9倍","n":22,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△②－☆","bet_type":"wide","band":"片方が5～9.9倍","n":18,"hits":1,"hit_rate":5.6,"roi":193.3,"avg_payout":3480.0},{"division":"中央","ticket":"△②－☆","bet_type":"wide","band":"片方が50倍以上","n":22,"hits":1,"hit_rate":4.5,"roi":131.8,"avg_payout":2900.0},{"division":"中央","ticket":"◎－▲","bet_type":"wide","band":"片方が5～9.9倍","n":28,"hits":3,"hit_rate":10.7,"roi":53.2,"avg_payout":496.7},{"division":"中央","ticket":"◎－▲","bet_type":"wide","band":"片方が10～19.9倍","n":13,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎－△①","bet_type":"wide","band":"片方が5～9.9倍","n":22,"hits":4,"hit_rate":18.2,"roi":77.3,"avg_payout":425.0},{"division":"中央","ticket":"◎－△②","bet_type":"wide","band":"片方が5～9.9倍","n":18,"hits":2,"hit_rate":11.1,"roi":63.3,"avg_payout":570.0},{"division":"中央","ticket":"◎－△②","bet_type":"wide","band":"片方が10～19.9倍","n":16,"hits":1,"hit_rate":6.2,"roi":49.4,"avg_payout":790.0},{"division":"中央","ticket":"◎－☆","bet_type":"wide","band":"片方が20～49.9倍","n":13,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○－▲","bet_type":"wide","band":"片方が10～19.9倍","n":21,"hits":1,"hit_rate":4.8,"roi":90.0,"avg_payout":1890.0},{"division":"中央","ticket":"○－△②","bet_type":"wide","band":"片方が10～19.9倍","n":22,"hits":1,"hit_rate":4.5,"roi":30.5,"avg_payout":670.0},{"division":"中央","ticket":"○－☆","bet_type":"wide","band":"片方が20～49.9倍","n":19,"hits":1,"hit_rate":5.3,"roi":155.8,"avg_payout":2960.0},{"division":"中央","ticket":"▲－△①","bet_type":"wide","band":"片方が5～9.9倍","n":31,"hits":1,"hit_rate":3.2,"roi":14.2,"avg_payout":440.0},{"division":"中央","ticket":"▲－△①","bet_type":"wide","band":"片方が10～19.9倍","n":17,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"▲－△②","bet_type":"wide","band":"片方が10～19.9倍","n":25,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"▲－△②","bet_type":"wide","band":"両馬10倍以上","n":18,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"▲－☆","bet_type":"wide","band":"片方が10～19.9倍","n":22,"hits":2,"hit_rate":9.1,"roi":219.5,"avg_payout":2415.0},{"division":"中央","ticket":"▲－☆","bet_type":"wide","band":"片方が20～49.9倍","n":22,"hits":1,"hit_rate":4.5,"roi":157.7,"avg_payout":3470.0},{"division":"中央","ticket":"▲－☆","bet_type":"wide","band":"両馬10倍以上","n":21,"hits":1,"hit_rate":4.8,"roi":165.2,"avg_payout":3470.0},{"division":"中央","ticket":"△①－△②","bet_type":"wide","band":"片方が5～9.9倍","n":18,"hits":1,"hit_rate":5.6,"roi":88.3,"avg_payout":1590.0},{"division":"中央","ticket":"△①－△②","bet_type":"wide","band":"片方が10～19.9倍","n":24,"hits":2,"hit_rate":8.3,"roi":237.1,"avg_payout":2845.0},{"division":"中央","ticket":"△②－☆","bet_type":"wide","band":"片方が10～19.9倍","n":27,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△②－☆","bet_type":"wide","band":"片方が20～49.9倍","n":26,"hits":1,"hit_rate":3.8,"roi":133.8,"avg_payout":3480.0},{"division":"中央","ticket":"△②－☆","bet_type":"wide","band":"両馬10倍以上","n":31,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○－△①","bet_type":"wide","band":"片方が10～19.9倍","n":18,"hits":1,"hit_rate":5.6,"roi":59.4,"avg_payout":1070.0},{"division":"中央","ticket":"○－☆","bet_type":"wide","band":"片方が10～19.9倍","n":19,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○－☆","bet_type":"wide","band":"片方が50倍以上","n":19,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○－☆","bet_type":"wide","band":"両馬10倍以上","n":14,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"▲－△②","bet_type":"wide","band":"片方が5～9.9倍","n":21,"hits":1,"hit_rate":4.8,"roi":25.7,"avg_payout":540.0},{"division":"中央","ticket":"▲－☆","bet_type":"wide","band":"片方が50倍以上","n":19,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△①－☆","bet_type":"wide","band":"片方が50倍以上","n":20,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎－△②","bet_type":"wide","band":"片方が20～49.9倍","n":14,"hits":3,"hit_rate":21.4,"roi":430.7,"avg_payout":2010.0},{"division":"中央","ticket":"◎－☆","bet_type":"wide","band":"片方が10～19.9倍","n":14,"hits":1,"hit_rate":7.1,"roi":35.0,"avg_payout":490.0},{"division":"中央","ticket":"○－△①","bet_type":"wide","band":"両馬5倍未満","n":4,"hits":1,"hit_rate":25.0,"roi":62.5,"avg_payout":250.0},{"division":"中央","ticket":"○－△②","bet_type":"wide","band":"片方が20～49.9倍","n":19,"hits":2,"hit_rate":10.5,"roi":231.6,"avg_payout":2200.0},{"division":"中央","ticket":"▲－△②","bet_type":"wide","band":"片方が20～49.9倍","n":22,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△①－☆","bet_type":"wide","band":"片方が10～19.9倍","n":21,"hits":1,"hit_rate":4.8,"roi":75.2,"avg_payout":1580.0},{"division":"中央","ticket":"◎－○","bet_type":"wide","band":"片方が20～49.9倍","n":5,"hits":1,"hit_rate":20.0,"roi":116.0,"avg_payout":580.0},{"division":"中央","ticket":"○－▲","bet_type":"wide","band":"片方が20～49.9倍","n":15,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○－△①","bet_type":"wide","band":"両馬10倍以上","n":11,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○－△②","bet_type":"wide","band":"両馬10倍以上","n":14,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎－△①","bet_type":"wide","band":"片方が50倍以上","n":7,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○－△①","bet_type":"wide","band":"片方が50倍以上","n":12,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"▲－△①","bet_type":"wide","band":"片方が50倍以上","n":12,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△①－☆","bet_type":"wide","band":"両馬10倍以上","n":15,"hits":1,"hit_rate":6.7,"roi":105.3,"avg_payout":1580.0},{"division":"中央","ticket":"◎－○","bet_type":"wide","band":"片方が10～19.9倍","n":10,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎－○","bet_type":"wide","band":"片方が50倍以上","n":4,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎－○","bet_type":"wide","band":"両馬10倍以上","n":1,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎－△①","bet_type":"wide","band":"片方が10～19.9倍","n":13,"hits":2,"hit_rate":15.4,"roi":112.3,"avg_payout":730.0},{"division":"中央","ticket":"◎－△①","bet_type":"wide","band":"両馬10倍以上","n":3,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎－☆","bet_type":"wide","band":"両馬10倍以上","n":4,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○－▲","bet_type":"wide","band":"片方が50倍以上","n":12,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○－△②","bet_type":"wide","band":"両馬5倍未満","n":2,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"▲－△①","bet_type":"wide","band":"両馬10倍以上","n":11,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎－☆","bet_type":"wide","band":"片方が50倍以上","n":13,"hits":1,"hit_rate":7.7,"roi":346.2,"avg_payout":4500.0},{"division":"中央","ticket":"◎－△②","bet_type":"wide","band":"両馬5倍未満","n":2,"hits":1,"hit_rate":50.0,"roi":105.0,"avg_payout":210.0},{"division":"中央","ticket":"▲－△②","bet_type":"wide","band":"両馬5倍未満","n":2,"hits":1,"hit_rate":50.0,"roi":140.0,"avg_payout":280.0},{"division":"中央","ticket":"◎－○","bet_type":"wide","band":"両馬5倍未満","n":7,"hits":3,"hit_rate":42.9,"roi":85.7,"avg_payout":200.0},{"division":"中央","ticket":"◎－☆","bet_type":"wide","band":"両馬5倍未満","n":4,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎－▲","bet_type":"wide","band":"片方が20～49.9倍","n":10,"hits":1,"hit_rate":10.0,"roi":61.0,"avg_payout":610.0},{"division":"中央","ticket":"◎－▲","bet_type":"wide","band":"両馬10倍以上","n":4,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎－△②","bet_type":"wide","band":"両馬10倍以上","n":4,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎－△①","bet_type":"wide","band":"両馬5倍未満","n":4,"hits":1,"hit_rate":25.0,"roi":65.0,"avg_payout":260.0},{"division":"中央","ticket":"○－▲","bet_type":"wide","band":"両馬10倍以上","n":8,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎－▲","bet_type":"wide","band":"片方が50倍以上","n":5,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○－▲","bet_type":"wide","band":"両馬5倍未満","n":4,"hits":1,"hit_rate":25.0,"roi":55.0,"avg_payout":220.0},{"division":"中央","ticket":"△①－△②","bet_type":"wide","band":"両馬5倍未満","n":3,"hits":1,"hit_rate":33.3,"roi":73.3,"avg_payout":220.0},{"division":"中央","ticket":"△②－☆","bet_type":"wide","band":"両馬5倍未満","n":1,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"▲－△①","bet_type":"wide","band":"両馬5倍未満","n":2,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"▲－☆","bet_type":"wide","band":"両馬5倍未満","n":1,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○－☆","bet_type":"wide","band":"両馬5倍未満","n":1,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎－○","bet_type":"quinella","band":"両馬5倍未満","n":21,"hits":7,"hit_rate":33.3,"roi":96.2,"avg_payout":288.6},{"division":"地方","ticket":"◎－▲","bet_type":"quinella","band":"片方が10～19.9倍","n":21,"hits":2,"hit_rate":9.5,"roi":128.6,"avg_payout":1350.0},{"division":"地方","ticket":"◎－△①","bet_type":"quinella","band":"片方が50倍以上","n":11,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎－△②","bet_type":"quinella","band":"両馬5倍未満","n":6,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎－☆","bet_type":"quinella","band":"片方が50倍以上","n":22,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"○－▲","bet_type":"quinella","band":"片方が10～19.9倍","n":29,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"○－△①","bet_type":"quinella","band":"片方が50倍以上","n":14,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"○－△②","bet_type":"quinella","band":"両馬5倍未満","n":5,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"○－☆","bet_type":"quinella","band":"片方が50倍以上","n":26,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"▲－△①","bet_type":"quinella","band":"片方が10～19.9倍","n":35,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"▲－△①","bet_type":"quinella","band":"片方が50倍以上","n":17,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"▲－△①","bet_type":"quinella","band":"両馬10倍以上","n":27,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"▲－△②","bet_type":"quinella","band":"片方が10～19.9倍","n":33,"hits":1,"hit_rate":3.0,"roi":37.3,"avg_payout":1230.0},{"division":"地方","ticket":"▲－☆","bet_type":"quinella","band":"片方が10～19.9倍","n":31,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"▲－☆","bet_type":"quinella","band":"片方が50倍以上","n":27,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"▲－☆","bet_type":"quinella","band":"両馬10倍以上","n":31,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△①－△②","bet_type":"quinella","band":"片方が50倍以上","n":31,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△①－☆","bet_type":"quinella","band":"片方が50倍以上","n":26,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△①－☆","bet_type":"quinella","band":"両馬10倍以上","n":35,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△②－☆","bet_type":"quinella","band":"片方が50倍以上","n":39,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎→○","bet_type":"exacta","band":"両馬5倍未満","n":21,"hits":3,"hit_rate":14.3,"roi":54.3,"avg_payout":380.0},{"division":"地方","ticket":"◎→▲","bet_type":"exacta","band":"片方が10～19.9倍","n":21,"hits":2,"hit_rate":9.5,"roi":203.8,"avg_payout":2140.0},{"division":"地方","ticket":"◎→△①","bet_type":"exacta","band":"片方が50倍以上","n":11,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎→△②","bet_type":"exacta","band":"両馬5倍未満","n":6,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎→☆","bet_type":"exacta","band":"片方が50倍以上","n":22,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"○→◎","bet_type":"exacta","band":"両馬5倍未満","n":21,"hits":4,"hit_rate":19.0,"roi":128.6,"avg_payout":675.0},{"division":"地方","ticket":"○→▲","bet_type":"exacta","band":"片方が10～19.9倍","n":29,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"○→△①","bet_type":"exacta","band":"片方が50倍以上","n":14,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"○→△②","bet_type":"exacta","band":"両馬5倍未満","n":5,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"○→☆","bet_type":"exacta","band":"片方が50倍以上","n":26,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"▲→◎","bet_type":"exacta","band":"片方が10～19.9倍","n":21,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"▲→○","bet_type":"exacta","band":"片方が10～19.9倍","n":29,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"▲→△①","bet_type":"exacta","band":"片方が10～19.9倍","n":35,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"▲→△①","bet_type":"exacta","band":"片方が50倍以上","n":17,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"▲→△①","bet_type":"exacta","band":"両馬10倍以上","n":27,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"▲→△②","bet_type":"exacta","band":"片方が10～19.9倍","n":33,"hits":1,"hit_rate":3.0,"roi":59.1,"avg_payout":1950.0},{"division":"地方","ticket":"▲→☆","bet_type":"exacta","band":"片方が10～19.9倍","n":31,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"▲→☆","bet_type":"exacta","band":"片方が50倍以上","n":27,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"▲→☆","bet_type":"exacta","band":"両馬10倍以上","n":31,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△①→◎","bet_type":"exacta","band":"片方が50倍以上","n":11,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△①→○","bet_type":"exacta","band":"片方が50倍以上","n":14,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△①→▲","bet_type":"exacta","band":"片方が10～19.9倍","n":35,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△①→▲","bet_type":"exacta","band":"片方が50倍以上","n":17,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△①→▲","bet_type":"exacta","band":"両馬10倍以上","n":27,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△①→△②","bet_type":"exacta","band":"片方が50倍以上","n":31,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△①→☆","bet_type":"exacta","band":"片方が50倍以上","n":26,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△①→☆","bet_type":"exacta","band":"両馬10倍以上","n":35,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△②→◎","bet_type":"exacta","band":"両馬5倍未満","n":6,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△②→○","bet_type":"exacta","band":"両馬5倍未満","n":5,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△②→▲","bet_type":"exacta","band":"片方が10～19.9倍","n":33,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△②→△①","bet_type":"exacta","band":"片方が50倍以上","n":31,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△②→☆","bet_type":"exacta","band":"片方が50倍以上","n":39,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"☆→◎","bet_type":"exacta","band":"片方が50倍以上","n":22,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"☆→○","bet_type":"exacta","band":"片方が50倍以上","n":26,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"☆→▲","bet_type":"exacta","band":"片方が10～19.9倍","n":31,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"☆→▲","bet_type":"exacta","band":"片方が50倍以上","n":27,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"☆→▲","bet_type":"exacta","band":"両馬10倍以上","n":31,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"☆→△①","bet_type":"exacta","band":"片方が50倍以上","n":26,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"☆→△①","bet_type":"exacta","band":"両馬10倍以上","n":35,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"☆→△②","bet_type":"exacta","band":"片方が50倍以上","n":39,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎－▲","bet_type":"quinella","band":"両馬5倍未満","n":13,"hits":4,"hit_rate":30.8,"roi":134.6,"avg_payout":437.5},{"division":"地方","ticket":"◎－△①","bet_type":"quinella","band":"片方が20～49.9倍","n":20,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎－△②","bet_type":"quinella","band":"片方が5～9.9倍","n":36,"hits":1,"hit_rate":2.8,"roi":70.0,"avg_payout":2520.0},{"division":"地方","ticket":"○－▲","bet_type":"quinella","band":"両馬5倍未満","n":9,"hits":1,"hit_rate":11.1,"roi":45.6,"avg_payout":410.0},{"division":"地方","ticket":"○－△①","bet_type":"quinella","band":"片方が20～49.9倍","n":29,"hits":1,"hit_rate":3.4,"roi":14.1,"avg_payout":410.0},{"division":"地方","ticket":"○－△②","bet_type":"quinella","band":"片方が5～9.9倍","n":32,"hits":2,"hit_rate":6.2,"roi":51.6,"avg_payout":825.0},{"division":"地方","ticket":"▲－△①","bet_type":"quinella","band":"片方が20～49.9倍","n":34,"hits":1,"hit_rate":2.9,"roi":102.1,"avg_payout":3470.0},{"division":"地方","ticket":"▲－△②","bet_type":"quinella","band":"片方が5～9.9倍","n":41,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△①－△②","bet_type":"quinella","band":"片方が5～9.9倍","n":44,"hits":2,"hit_rate":4.5,"roi":23.4,"avg_payout":515.0},{"division":"地方","ticket":"△①－△②","bet_type":"quinella","band":"片方が20～49.9倍","n":29,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△①－☆","bet_type":"quinella","band":"片方が20～49.9倍","n":38,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△②－☆","bet_type":"quinella","band":"片方が5～9.9倍","n":33,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎→▲","bet_type":"exacta","band":"両馬5倍未満","n":13,"hits":1,"hit_rate":7.7,"roi":103.8,"avg_payout":1350.0},{"division":"地方","ticket":"◎→△①","bet_type":"exacta","band":"片方が20～49.9倍","n":20,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎→△②","bet_type":"exacta","band":"片方が5～9.9倍","n":36,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"○→▲","bet_type":"exacta","band":"両馬5倍未満","n":9,"hits":1,"hit_rate":11.1,"roi":92.2,"avg_payout":830.0},{"division":"地方","ticket":"○→△①","bet_type":"exacta","band":"片方が20～49.9倍","n":29,"hits":1,"hit_rate":3.4,"roi":14.8,"avg_payout":430.0},{"division":"地方","ticket":"○→△②","bet_type":"exacta","band":"片方が5～9.9倍","n":32,"hits":1,"hit_rate":3.1,"roi":28.1,"avg_payout":900.0},{"division":"地方","ticket":"▲→◎","bet_type":"exacta","band":"両馬5倍未満","n":13,"hits":3,"hit_rate":23.1,"roi":171.5,"avg_payout":743.3},{"division":"地方","ticket":"▲→○","bet_type":"exacta","band":"両馬5倍未満","n":9,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"▲→△①","bet_type":"exacta","band":"片方が20～49.9倍","n":34,"hits":1,"hit_rate":2.9,"roi":140.6,"avg_payout":4780.0},{"division":"地方","ticket":"▲→△②","bet_type":"exacta","band":"片方が5～9.9倍","n":41,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△①→◎","bet_type":"exacta","band":"片方が20～49.9倍","n":20,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△①→○","bet_type":"exacta","band":"片方が20～49.9倍","n":29,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△①→▲","bet_type":"exacta","band":"片方が20～49.9倍","n":34,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△①→△②","bet_type":"exacta","band":"片方が5～9.9倍","n":44,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△①→△②","bet_type":"exacta","band":"片方が20～49.9倍","n":29,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△①→☆","bet_type":"exacta","band":"片方が20～49.9倍","n":38,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△②→◎","bet_type":"exacta","band":"片方が5～9.9倍","n":36,"hits":1,"hit_rate":2.8,"roi":150.6,"avg_payout":5420.0},{"division":"地方","ticket":"△②→○","bet_type":"exacta","band":"片方が5～9.9倍","n":32,"hits":1,"hit_rate":3.1,"roi":94.1,"avg_payout":3010.0},{"division":"地方","ticket":"△②→▲","bet_type":"exacta","band":"片方が5～9.9倍","n":41,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△②→△①","bet_type":"exacta","band":"片方が5～9.9倍","n":44,"hits":2,"hit_rate":4.5,"roi":35.7,"avg_payout":785.0},{"division":"地方","ticket":"△②→△①","bet_type":"exacta","band":"片方が20～49.9倍","n":29,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△②→☆","bet_type":"exacta","band":"片方が5～9.9倍","n":33,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"☆→△①","bet_type":"exacta","band":"片方が20～49.9倍","n":38,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"☆→△②","bet_type":"exacta","band":"片方が5～9.9倍","n":33,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎－△①","bet_type":"quinella","band":"片方が5～9.9倍","n":42,"hits":4,"hit_rate":9.5,"roi":142.9,"avg_payout":1500.0},{"division":"地方","ticket":"◎－△②","bet_type":"quinella","band":"片方が50倍以上","n":23,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"○－△①","bet_type":"quinella","band":"片方が5～9.9倍","n":41,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"○－△②","bet_type":"quinella","band":"片方が50倍以上","n":27,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"▲－△①","bet_type":"quinella","band":"片方が5～9.9倍","n":44,"hits":1,"hit_rate":2.3,"roi":45.2,"avg_payout":1990.0},{"division":"地方","ticket":"▲－△②","bet_type":"quinella","band":"片方が50倍以上","n":30,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△①－☆","bet_type":"quinella","band":"片方が5～9.9倍","n":40,"hits":2,"hit_rate":5.0,"roi":44.8,"avg_payout":895.0},{"division":"地方","ticket":"△②－☆","bet_type":"quinella","band":"両馬10倍以上","n":38,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎→△①","bet_type":"exacta","band":"片方が5～9.9倍","n":42,"hits":1,"hit_rate":2.4,"roi":40.0,"avg_payout":1680.0},{"division":"地方","ticket":"◎→△②","bet_type":"exacta","band":"片方が50倍以上","n":23,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"○→△①","bet_type":"exacta","band":"片方が5～9.9倍","n":41,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"○→△②","bet_type":"exacta","band":"片方が50倍以上","n":27,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"▲→△①","bet_type":"exacta","band":"片方が5～9.9倍","n":44,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"▲→△②","bet_type":"exacta","band":"片方が50倍以上","n":30,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△①→◎","bet_type":"exacta","band":"片方が5～9.9倍","n":42,"hits":3,"hit_rate":7.1,"roi":227.9,"avg_payout":3190.0},{"division":"地方","ticket":"△①→○","bet_type":"exacta","band":"片方が5～9.9倍","n":41,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△①→▲","bet_type":"exacta","band":"片方が5～9.9倍","n":44,"hits":1,"hit_rate":2.3,"roi":66.1,"avg_payout":2910.0},{"division":"地方","ticket":"△①→☆","bet_type":"exacta","band":"片方が5～9.9倍","n":40,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△②→◎","bet_type":"exacta","band":"片方が50倍以上","n":23,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△②→○","bet_type":"exacta","band":"片方が50倍以上","n":27,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△②→▲","bet_type":"exacta","band":"片方が50倍以上","n":30,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△②→☆","bet_type":"exacta","band":"両馬10倍以上","n":38,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"☆→△①","bet_type":"exacta","band":"片方が5～9.9倍","n":40,"hits":2,"hit_rate":5.0,"roi":77.2,"avg_payout":1545.0},{"division":"地方","ticket":"☆→△②","bet_type":"exacta","band":"両馬10倍以上","n":38,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎－△①","bet_type":"quinella","band":"片方が10～19.9倍","n":23,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎－△②","bet_type":"quinella","band":"片方が10～19.9倍","n":22,"hits":4,"hit_rate":18.2,"roi":348.6,"avg_payout":1917.5},{"division":"地方","ticket":"◎－☆","bet_type":"quinella","band":"片方が20～49.9倍","n":23,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"○－△①","bet_type":"quinella","band":"片方が10～19.9倍","n":31,"hits":1,"hit_rate":3.2,"roi":75.2,"avg_payout":2330.0},{"division":"地方","ticket":"○－△②","bet_type":"quinella","band":"片方が10～19.9倍","n":31,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"○－☆","bet_type":"quinella","band":"片方が20～49.9倍","n":33,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"▲－☆","bet_type":"quinella","band":"片方が20～49.9倍","n":38,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△①－△②","bet_type":"quinella","band":"片方が10～19.9倍","n":31,"hits":2,"hit_rate":6.5,"roi":101.3,"avg_payout":1570.0},{"division":"地方","ticket":"△①－△②","bet_type":"quinella","band":"両馬10倍以上","n":32,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△①－☆","bet_type":"quinella","band":"片方が10～19.9倍","n":34,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△②－☆","bet_type":"quinella","band":"片方が10～19.9倍","n":35,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△②－☆","bet_type":"quinella","band":"片方が20～49.9倍","n":36,"hits":1,"hit_rate":2.8,"roi":116.1,"avg_payout":4180.0},{"division":"地方","ticket":"◎→△①","bet_type":"exacta","band":"片方が10～19.9倍","n":23,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎→△②","bet_type":"exacta","band":"片方が10～19.9倍","n":22,"hits":1,"hit_rate":4.5,"roi":50.9,"avg_payout":1120.0},{"division":"地方","ticket":"◎→☆","bet_type":"exacta","band":"片方が20～49.9倍","n":23,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"○→△①","bet_type":"exacta","band":"片方が10～19.9倍","n":31,"hits":1,"hit_rate":3.2,"roi":100.0,"avg_payout":3100.0},{"division":"地方","ticket":"○→△②","bet_type":"exacta","band":"片方が10～19.9倍","n":31,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"○→☆","bet_type":"exacta","band":"片方が20～49.9倍","n":33,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"▲→☆","bet_type":"exacta","band":"片方が20～49.9倍","n":38,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△①→◎","bet_type":"exacta","band":"片方が10～19.9倍","n":23,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△①→○","bet_type":"exacta","band":"片方が10～19.9倍","n":31,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△①→△②","bet_type":"exacta","band":"片方が10～19.9倍","n":31,"hits":1,"hit_rate":3.2,"roi":54.2,"avg_payout":1680.0},{"division":"地方","ticket":"△①→△②","bet_type":"exacta","band":"両馬10倍以上","n":32,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△①→☆","bet_type":"exacta","band":"片方が10～19.9倍","n":34,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△②→◎","bet_type":"exacta","band":"片方が10～19.9倍","n":22,"hits":3,"hit_rate":13.6,"roi":564.5,"avg_payout":4140.0},{"division":"地方","ticket":"△②→○","bet_type":"exacta","band":"片方が10～19.9倍","n":31,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△②→△①","bet_type":"exacta","band":"片方が10～19.9倍","n":31,"hits":1,"hit_rate":3.2,"roi":123.9,"avg_payout":3840.0},{"division":"地方","ticket":"△②→△①","bet_type":"exacta","band":"両馬10倍以上","n":32,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△②→☆","bet_type":"exacta","band":"片方が10～19.9倍","n":35,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△②→☆","bet_type":"exacta","band":"片方が20～49.9倍","n":36,"hits":1,"hit_rate":2.8,"roi":183.1,"avg_payout":6590.0},{"division":"地方","ticket":"☆→◎","bet_type":"exacta","band":"片方が20～49.9倍","n":23,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"☆→○","bet_type":"exacta","band":"片方が20～49.9倍","n":33,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"☆→▲","bet_type":"exacta","band":"片方が20～49.9倍","n":38,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"☆→△①","bet_type":"exacta","band":"片方が10～19.9倍","n":34,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"☆→△②","bet_type":"exacta","band":"片方が10～19.9倍","n":35,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"☆→△②","bet_type":"exacta","band":"片方が20～49.9倍","n":36,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"○－▲","bet_type":"quinella","band":"片方が50倍以上","n":14,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"○－▲","bet_type":"quinella","band":"両馬10倍以上","n":15,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"○－△②","bet_type":"quinella","band":"両馬10倍以上","n":15,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"○－☆","bet_type":"quinella","band":"片方が10～19.9倍","n":31,"hits":3,"hit_rate":9.7,"roi":245.2,"avg_payout":2533.3},{"division":"地方","ticket":"○－☆","bet_type":"quinella","band":"両馬10倍以上","n":21,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"▲－△②","bet_type":"quinella","band":"両馬10倍以上","n":29,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"○→▲","bet_type":"exacta","band":"片方が50倍以上","n":14,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"○→▲","bet_type":"exacta","band":"両馬10倍以上","n":15,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"○→△②","bet_type":"exacta","band":"両馬10倍以上","n":15,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"○→☆","bet_type":"exacta","band":"片方が10～19.9倍","n":31,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"○→☆","bet_type":"exacta","band":"両馬10倍以上","n":21,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"▲→○","bet_type":"exacta","band":"片方が50倍以上","n":14,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"▲→○","bet_type":"exacta","band":"両馬10倍以上","n":15,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"▲→△②","bet_type":"exacta","band":"両馬10倍以上","n":29,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△②→○","bet_type":"exacta","band":"両馬10倍以上","n":15,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△②→▲","bet_type":"exacta","band":"両馬10倍以上","n":29,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"☆→○","bet_type":"exacta","band":"片方が10～19.9倍","n":31,"hits":3,"hit_rate":9.7,"roi":545.2,"avg_payout":5633.3},{"division":"地方","ticket":"☆→○","bet_type":"exacta","band":"両馬10倍以上","n":21,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎－○","bet_type":"quinella","band":"片方が10～19.9倍","n":16,"hits":3,"hit_rate":18.8,"roi":370.6,"avg_payout":1976.7},{"division":"地方","ticket":"◎－▲","bet_type":"quinella","band":"片方が20～49.9倍","n":18,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎－▲","bet_type":"quinella","band":"両馬10倍以上","n":6,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎－△①","bet_type":"quinella","band":"両馬10倍以上","n":5,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎－△②","bet_type":"quinella","band":"片方が20～49.9倍","n":15,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎－△②","bet_type":"quinella","band":"両馬10倍以上","n":5,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎－☆","bet_type":"quinella","band":"片方が10～19.9倍","n":20,"hits":2,"hit_rate":10.0,"roi":209.0,"avg_payout":2090.0},{"division":"地方","ticket":"○－▲","bet_type":"quinella","band":"片方が20～49.9倍","n":27,"hits":2,"hit_rate":7.4,"roi":309.6,"avg_payout":4180.0},{"division":"地方","ticket":"○－△②","bet_type":"quinella","band":"片方が20～49.9倍","n":23,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"○－☆","bet_type":"quinella","band":"両馬5倍未満","n":4,"hits":3,"hit_rate":75.0,"roi":210.0,"avg_payout":280.0},{"division":"地方","ticket":"▲－△②","bet_type":"quinella","band":"片方が20～49.9倍","n":28,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎→○","bet_type":"exacta","band":"片方が10～19.9倍","n":16,"hits":3,"hit_rate":18.8,"roi":591.9,"avg_payout":3156.7},{"division":"地方","ticket":"◎→▲","bet_type":"exacta","band":"片方が20～49.9倍","n":18,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎→▲","bet_type":"exacta","band":"両馬10倍以上","n":6,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎→△①","bet_type":"exacta","band":"両馬10倍以上","n":5,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎→△②","bet_type":"exacta","band":"片方が20～49.9倍","n":15,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎→△②","bet_type":"exacta","band":"両馬10倍以上","n":5,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎→☆","bet_type":"exacta","band":"片方が10～19.9倍","n":20,"hits":2,"hit_rate":10.0,"roi":347.0,"avg_payout":3470.0},{"division":"地方","ticket":"○→◎","bet_type":"exacta","band":"片方が10～19.9倍","n":16,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"○→▲","bet_type":"exacta","band":"片方が20～49.9倍","n":27,"hits":1,"hit_rate":3.7,"roi":60.4,"avg_payout":1630.0},{"division":"地方","ticket":"○→△②","bet_type":"exacta","band":"片方が20～49.9倍","n":23,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"○→☆","bet_type":"exacta","band":"両馬5倍未満","n":4,"hits":2,"hit_rate":50.0,"roi":330.0,"avg_payout":660.0},{"division":"地方","ticket":"▲→◎","bet_type":"exacta","band":"片方が20～49.9倍","n":18,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"▲→◎","bet_type":"exacta","band":"両馬10倍以上","n":6,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"▲→○","bet_type":"exacta","band":"片方が20～49.9倍","n":27,"hits":1,"hit_rate":3.7,"roi":1151.9,"avg_payout":31100.0},{"division":"地方","ticket":"▲→△②","bet_type":"exacta","band":"片方が20～49.9倍","n":28,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△①→◎","bet_type":"exacta","band":"両馬10倍以上","n":5,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△②→◎","bet_type":"exacta","band":"片方が20～49.9倍","n":15,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△②→◎","bet_type":"exacta","band":"両馬10倍以上","n":5,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△②→○","bet_type":"exacta","band":"片方が20～49.9倍","n":23,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△②→▲","bet_type":"exacta","band":"片方が20～49.9倍","n":28,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"☆→◎","bet_type":"exacta","band":"片方が10～19.9倍","n":20,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"☆→○","bet_type":"exacta","band":"両馬5倍未満","n":4,"hits":1,"hit_rate":25.0,"roi":27.5,"avg_payout":110.0},{"division":"地方","ticket":"◎－○","bet_type":"quinella","band":"片方が5～9.9倍","n":37,"hits":3,"hit_rate":8.1,"roi":61.1,"avg_payout":753.3},{"division":"地方","ticket":"◎－▲","bet_type":"quinella","band":"片方が5～9.9倍","n":38,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎－△①","bet_type":"quinella","band":"両馬5倍未満","n":6,"hits":1,"hit_rate":16.7,"roi":113.3,"avg_payout":680.0},{"division":"地方","ticket":"◎－☆","bet_type":"quinella","band":"両馬5倍未満","n":2,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"○－▲","bet_type":"quinella","band":"片方が5～9.9倍","n":34,"hits":3,"hit_rate":8.8,"roi":69.4,"avg_payout":786.7},{"division":"地方","ticket":"○－☆","bet_type":"quinella","band":"片方が5～9.9倍","n":27,"hits":2,"hit_rate":7.4,"roi":81.9,"avg_payout":1105.0},{"division":"地方","ticket":"▲－☆","bet_type":"quinella","band":"片方が5～9.9倍","n":32,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△①－☆","bet_type":"quinella","band":"両馬5倍未満","n":1,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎→○","bet_type":"exacta","band":"片方が5～9.9倍","n":37,"hits":3,"hit_rate":8.1,"roi":116.5,"avg_payout":1436.7},{"division":"地方","ticket":"◎→▲","bet_type":"exacta","band":"片方が5～9.9倍","n":38,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎→△①","bet_type":"exacta","band":"両馬5倍未満","n":6,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎→☆","bet_type":"exacta","band":"両馬5倍未満","n":2,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"○→◎","bet_type":"exacta","band":"片方が5～9.9倍","n":37,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"○→▲","bet_type":"exacta","band":"片方が5～9.9倍","n":34,"hits":3,"hit_rate":8.8,"roi":100.3,"avg_payout":1136.7},{"division":"地方","ticket":"○→☆","bet_type":"exacta","band":"片方が5～9.9倍","n":27,"hits":2,"hit_rate":7.4,"roi":191.1,"avg_payout":2580.0},{"division":"地方","ticket":"▲→◎","bet_type":"exacta","band":"片方が5～9.9倍","n":38,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"▲→○","bet_type":"exacta","band":"片方が5～9.9倍","n":34,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"▲→☆","bet_type":"exacta","band":"片方が5～9.9倍","n":32,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△①→◎","bet_type":"exacta","band":"両馬5倍未満","n":6,"hits":1,"hit_rate":16.7,"roi":236.7,"avg_payout":1420.0},{"division":"地方","ticket":"△①→☆","bet_type":"exacta","band":"両馬5倍未満","n":1,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"☆→◎","bet_type":"exacta","band":"両馬5倍未満","n":2,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"☆→○","bet_type":"exacta","band":"片方が5～9.9倍","n":27,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"☆→▲","bet_type":"exacta","band":"片方が5～9.9倍","n":32,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"☆→△①","bet_type":"exacta","band":"両馬5倍未満","n":1,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎－☆","bet_type":"quinella","band":"片方が5～9.9倍","n":32,"hits":1,"hit_rate":3.1,"roi":22.2,"avg_payout":710.0},{"division":"地方","ticket":"○－△①","bet_type":"quinella","band":"両馬5倍未満","n":5,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎→☆","bet_type":"exacta","band":"片方が5～9.9倍","n":32,"hits":1,"hit_rate":3.1,"roi":63.1,"avg_payout":2020.0},{"division":"地方","ticket":"○→△①","bet_type":"exacta","band":"両馬5倍未満","n":5,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△①→○","bet_type":"exacta","band":"両馬5倍未満","n":5,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"☆→◎","bet_type":"exacta","band":"片方が5～9.9倍","n":32,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎－○","bet_type":"quinella","band":"片方が50倍以上","n":5,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"○－△①","bet_type":"quinella","band":"両馬10倍以上","n":18,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎→○","bet_type":"exacta","band":"片方が50倍以上","n":5,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"○→◎","bet_type":"exacta","band":"片方が50倍以上","n":5,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"○→△①","bet_type":"exacta","band":"両馬10倍以上","n":18,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△①→○","bet_type":"exacta","band":"両馬10倍以上","n":18,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎－○","bet_type":"quinella","band":"片方が20～49.9倍","n":11,"hits":2,"hit_rate":18.2,"roi":174.5,"avg_payout":960.0},{"division":"地方","ticket":"◎→○","bet_type":"exacta","band":"片方が20～49.9倍","n":11,"hits":1,"hit_rate":9.1,"roi":139.1,"avg_payout":1530.0},{"division":"地方","ticket":"○→◎","bet_type":"exacta","band":"片方が20～49.9倍","n":11,"hits":1,"hit_rate":9.1,"roi":60.0,"avg_payout":660.0},{"division":"地方","ticket":"◎－▲","bet_type":"quinella","band":"片方が50倍以上","n":10,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎→▲","bet_type":"exacta","band":"片方が50倍以上","n":10,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"▲→◎","bet_type":"exacta","band":"片方が50倍以上","n":10,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"▲－☆","bet_type":"quinella","band":"両馬5倍未満","n":3,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"▲→☆","bet_type":"exacta","band":"両馬5倍未満","n":3,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"☆→▲","bet_type":"exacta","band":"両馬5倍未満","n":3,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"▲－△②","bet_type":"quinella","band":"両馬5倍未満","n":1,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"▲→△②","bet_type":"exacta","band":"両馬5倍未満","n":1,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△②→▲","bet_type":"exacta","band":"両馬5倍未満","n":1,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎－☆","bet_type":"quinella","band":"両馬10倍以上","n":5,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎→☆","bet_type":"exacta","band":"両馬10倍以上","n":5,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"☆→◎","bet_type":"exacta","band":"両馬10倍以上","n":5,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎－○","bet_type":"quinella","band":"両馬10倍以上","n":2,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"◎→○","bet_type":"exacta","band":"両馬10倍以上","n":2,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"○→◎","bet_type":"exacta","band":"両馬10倍以上","n":2,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"▲－△①","bet_type":"quinella","band":"両馬5倍未満","n":2,"hits":1,"hit_rate":50.0,"roi":295.0,"avg_payout":590.0},{"division":"地方","ticket":"▲→△①","bet_type":"exacta","band":"両馬5倍未満","n":2,"hits":1,"hit_rate":50.0,"roi":575.0,"avg_payout":1150.0},{"division":"地方","ticket":"△①→▲","bet_type":"exacta","band":"両馬5倍未満","n":2,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△②－☆","bet_type":"quinella","band":"両馬5倍未満","n":2,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△②→☆","bet_type":"exacta","band":"両馬5倍未満","n":2,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"☆→△②","bet_type":"exacta","band":"両馬5倍未満","n":2,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△①－△②","bet_type":"quinella","band":"両馬5倍未満","n":2,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△①→△②","bet_type":"exacta","band":"両馬5倍未満","n":2,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"地方","ticket":"△②→△①","bet_type":"exacta","band":"両馬5倍未満","n":2,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎－○","bet_type":"quinella","band":"片方が5～9.9倍","n":28,"hits":2,"hit_rate":7.1,"roi":115.4,"avg_payout":1615.0},{"division":"中央","ticket":"◎－▲","bet_type":"quinella","band":"両馬5倍未満","n":8,"hits":4,"hit_rate":50.0,"roi":288.8,"avg_payout":577.5},{"division":"中央","ticket":"◎－△①","bet_type":"quinella","band":"片方が20～49.9倍","n":10,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎－△②","bet_type":"quinella","band":"片方が50倍以上","n":12,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎－☆","bet_type":"quinella","band":"片方が5～9.9倍","n":19,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○－▲","bet_type":"quinella","band":"片方が5～9.9倍","n":32,"hits":3,"hit_rate":9.4,"roi":301.6,"avg_payout":3216.7},{"division":"中央","ticket":"○－△①","bet_type":"quinella","band":"片方が5～9.9倍","n":30,"hits":1,"hit_rate":3.3,"roi":29.3,"avg_payout":880.0},{"division":"中央","ticket":"○－△①","bet_type":"quinella","band":"片方が20～49.9倍","n":12,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○－△②","bet_type":"quinella","band":"片方が5～9.9倍","n":24,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○－△②","bet_type":"quinella","band":"片方が50倍以上","n":18,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○－☆","bet_type":"quinella","band":"片方が5～9.9倍","n":28,"hits":1,"hit_rate":3.6,"roi":51.1,"avg_payout":1430.0},{"division":"中央","ticket":"▲－△①","bet_type":"quinella","band":"片方が20～49.9倍","n":20,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"▲－△②","bet_type":"quinella","band":"片方が50倍以上","n":16,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"▲－☆","bet_type":"quinella","band":"片方が5～9.9倍","n":25,"hits":2,"hit_rate":8.0,"roi":285.6,"avg_payout":3570.0},{"division":"中央","ticket":"△①－△②","bet_type":"quinella","band":"片方が20～49.9倍","n":24,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△①－△②","bet_type":"quinella","band":"片方が50倍以上","n":16,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△①－△②","bet_type":"quinella","band":"両馬10倍以上","n":19,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△①－☆","bet_type":"quinella","band":"片方が5～9.9倍","n":26,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△①－☆","bet_type":"quinella","band":"片方が20～49.9倍","n":22,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△②－☆","bet_type":"quinella","band":"片方が5～9.9倍","n":18,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△②－☆","bet_type":"quinella","band":"片方が50倍以上","n":22,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎→○","bet_type":"exacta","band":"片方が5～9.9倍","n":28,"hits":2,"hit_rate":7.1,"roi":260.4,"avg_payout":3645.0},{"division":"中央","ticket":"◎→▲","bet_type":"exacta","band":"両馬5倍未満","n":8,"hits":3,"hit_rate":37.5,"roi":390.0,"avg_payout":1040.0},{"division":"中央","ticket":"◎→△①","bet_type":"exacta","band":"片方が20～49.9倍","n":10,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎→△②","bet_type":"exacta","band":"片方が50倍以上","n":12,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎→☆","bet_type":"exacta","band":"片方が5～9.9倍","n":19,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○→◎","bet_type":"exacta","band":"片方が5～9.9倍","n":28,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○→▲","bet_type":"exacta","band":"片方が5～9.9倍","n":32,"hits":1,"hit_rate":3.1,"roi":355.0,"avg_payout":11360.0},{"division":"中央","ticket":"○→△①","bet_type":"exacta","band":"片方が5～9.9倍","n":30,"hits":1,"hit_rate":3.3,"roi":49.0,"avg_payout":1470.0},{"division":"中央","ticket":"○→△①","bet_type":"exacta","band":"片方が20～49.9倍","n":12,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○→△②","bet_type":"exacta","band":"片方が5～9.9倍","n":24,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○→△②","bet_type":"exacta","band":"片方が50倍以上","n":18,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○→☆","bet_type":"exacta","band":"片方が5～9.9倍","n":28,"hits":1,"hit_rate":3.6,"roi":102.5,"avg_payout":2870.0},{"division":"中央","ticket":"▲→◎","bet_type":"exacta","band":"両馬5倍未満","n":8,"hits":1,"hit_rate":12.5,"roi":170.0,"avg_payout":1360.0},{"division":"中央","ticket":"▲→○","bet_type":"exacta","band":"片方が5～9.9倍","n":32,"hits":2,"hit_rate":6.2,"roi":183.4,"avg_payout":2935.0},{"division":"中央","ticket":"▲→△①","bet_type":"exacta","band":"片方が20～49.9倍","n":20,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"▲→△②","bet_type":"exacta","band":"片方が50倍以上","n":16,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"▲→☆","bet_type":"exacta","band":"片方が5～9.9倍","n":25,"hits":2,"hit_rate":8.0,"roi":730.4,"avg_payout":9130.0},{"division":"中央","ticket":"△①→◎","bet_type":"exacta","band":"片方が20～49.9倍","n":10,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△①→○","bet_type":"exacta","band":"片方が5～9.9倍","n":30,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△①→○","bet_type":"exacta","band":"片方が20～49.9倍","n":12,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△①→▲","bet_type":"exacta","band":"片方が20～49.9倍","n":20,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△①→△②","bet_type":"exacta","band":"片方が20～49.9倍","n":24,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△①→△②","bet_type":"exacta","band":"片方が50倍以上","n":16,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△①→△②","bet_type":"exacta","band":"両馬10倍以上","n":19,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△①→☆","bet_type":"exacta","band":"片方が5～9.9倍","n":26,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△①→☆","bet_type":"exacta","band":"片方が20～49.9倍","n":22,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△②→◎","bet_type":"exacta","band":"片方が50倍以上","n":12,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△②→○","bet_type":"exacta","band":"片方が5～9.9倍","n":24,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△②→○","bet_type":"exacta","band":"片方が50倍以上","n":18,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△②→▲","bet_type":"exacta","band":"片方が50倍以上","n":16,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△②→△①","bet_type":"exacta","band":"片方が20～49.9倍","n":24,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△②→△①","bet_type":"exacta","band":"片方が50倍以上","n":16,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△②→△①","bet_type":"exacta","band":"両馬10倍以上","n":19,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△②→☆","bet_type":"exacta","band":"片方が5～9.9倍","n":18,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△②→☆","bet_type":"exacta","band":"片方が50倍以上","n":22,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"☆→◎","bet_type":"exacta","band":"片方が5～9.9倍","n":19,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"☆→○","bet_type":"exacta","band":"片方が5～9.9倍","n":28,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"☆→▲","bet_type":"exacta","band":"片方が5～9.9倍","n":25,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"☆→△①","bet_type":"exacta","band":"片方が5～9.9倍","n":26,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"☆→△①","bet_type":"exacta","band":"片方が20～49.9倍","n":22,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"☆→△②","bet_type":"exacta","band":"片方が5～9.9倍","n":18,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"☆→△②","bet_type":"exacta","band":"片方が50倍以上","n":22,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎－▲","bet_type":"quinella","band":"片方が5～9.9倍","n":28,"hits":1,"hit_rate":3.6,"roi":54.6,"avg_payout":1530.0},{"division":"中央","ticket":"◎－▲","bet_type":"quinella","band":"片方が10～19.9倍","n":13,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎－△①","bet_type":"quinella","band":"片方が5～9.9倍","n":22,"hits":3,"hit_rate":13.6,"roi":144.5,"avg_payout":1060.0},{"division":"中央","ticket":"◎－△②","bet_type":"quinella","band":"片方が5～9.9倍","n":18,"hits":1,"hit_rate":5.6,"roi":133.3,"avg_payout":2400.0},{"division":"中央","ticket":"◎－△②","bet_type":"quinella","band":"片方が10～19.9倍","n":16,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎－☆","bet_type":"quinella","band":"片方が20～49.9倍","n":13,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○－▲","bet_type":"quinella","band":"片方が10～19.9倍","n":21,"hits":1,"hit_rate":4.8,"roi":303.8,"avg_payout":6380.0},{"division":"中央","ticket":"○－△②","bet_type":"quinella","band":"片方が10～19.9倍","n":22,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○－☆","bet_type":"quinella","band":"片方が20～49.9倍","n":19,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"▲－△①","bet_type":"quinella","band":"片方が5～9.9倍","n":31,"hits":1,"hit_rate":3.2,"roi":31.0,"avg_payout":960.0},{"division":"中央","ticket":"▲－△①","bet_type":"quinella","band":"片方が10～19.9倍","n":17,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"▲－△②","bet_type":"quinella","band":"片方が10～19.9倍","n":25,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"▲－△②","bet_type":"quinella","band":"両馬10倍以上","n":18,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"▲－☆","bet_type":"quinella","band":"片方が10～19.9倍","n":22,"hits":1,"hit_rate":4.5,"roi":271.8,"avg_payout":5980.0},{"division":"中央","ticket":"▲－☆","bet_type":"quinella","band":"片方が20～49.9倍","n":22,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"▲－☆","bet_type":"quinella","band":"両馬10倍以上","n":21,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△①－△②","bet_type":"quinella","band":"片方が5～9.9倍","n":18,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△①－△②","bet_type":"quinella","band":"片方が10～19.9倍","n":24,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△②－☆","bet_type":"quinella","band":"片方が10～19.9倍","n":27,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△②－☆","bet_type":"quinella","band":"片方が20～49.9倍","n":26,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△②－☆","bet_type":"quinella","band":"両馬10倍以上","n":31,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎→▲","bet_type":"exacta","band":"片方が5～9.9倍","n":28,"hits":1,"hit_rate":3.6,"roi":101.1,"avg_payout":2830.0},{"division":"中央","ticket":"◎→▲","bet_type":"exacta","band":"片方が10～19.9倍","n":13,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎→△①","bet_type":"exacta","band":"片方が5～9.9倍","n":22,"hits":2,"hit_rate":9.1,"roi":219.1,"avg_payout":2410.0},{"division":"中央","ticket":"◎→△②","bet_type":"exacta","band":"片方が5～9.9倍","n":18,"hits":1,"hit_rate":5.6,"roi":259.4,"avg_payout":4670.0},{"division":"中央","ticket":"◎→△②","bet_type":"exacta","band":"片方が10～19.9倍","n":16,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎→☆","bet_type":"exacta","band":"片方が20～49.9倍","n":13,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○→▲","bet_type":"exacta","band":"片方が10～19.9倍","n":21,"hits":1,"hit_rate":4.8,"roi":541.0,"avg_payout":11360.0},{"division":"中央","ticket":"○→△②","bet_type":"exacta","band":"片方が10～19.9倍","n":22,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○→☆","bet_type":"exacta","band":"片方が20～49.9倍","n":19,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"▲→◎","bet_type":"exacta","band":"片方が5～9.9倍","n":28,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"▲→◎","bet_type":"exacta","band":"片方が10～19.9倍","n":13,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"▲→○","bet_type":"exacta","band":"片方が10～19.9倍","n":21,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"▲→△①","bet_type":"exacta","band":"片方が5～9.9倍","n":31,"hits":1,"hit_rate":3.2,"roi":49.7,"avg_payout":1540.0},{"division":"中央","ticket":"▲→△①","bet_type":"exacta","band":"片方が10～19.9倍","n":17,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"▲→△②","bet_type":"exacta","band":"片方が10～19.9倍","n":25,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"▲→△②","bet_type":"exacta","band":"両馬10倍以上","n":18,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"▲→☆","bet_type":"exacta","band":"片方が10～19.9倍","n":22,"hits":1,"hit_rate":4.5,"roi":758.6,"avg_payout":16690.0},{"division":"中央","ticket":"▲→☆","bet_type":"exacta","band":"片方が20～49.9倍","n":22,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"▲→☆","bet_type":"exacta","band":"両馬10倍以上","n":21,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△①→◎","bet_type":"exacta","band":"片方が5～9.9倍","n":22,"hits":1,"hit_rate":4.5,"roi":63.2,"avg_payout":1390.0},{"division":"中央","ticket":"△①→▲","bet_type":"exacta","band":"片方が5～9.9倍","n":31,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△①→▲","bet_type":"exacta","band":"片方が10～19.9倍","n":17,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△①→△②","bet_type":"exacta","band":"片方が5～9.9倍","n":18,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△①→△②","bet_type":"exacta","band":"片方が10～19.9倍","n":24,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△②→◎","bet_type":"exacta","band":"片方が5～9.9倍","n":18,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△②→◎","bet_type":"exacta","band":"片方が10～19.9倍","n":16,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△②→○","bet_type":"exacta","band":"片方が10～19.9倍","n":22,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△②→▲","bet_type":"exacta","band":"片方が10～19.9倍","n":25,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△②→▲","bet_type":"exacta","band":"両馬10倍以上","n":18,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△②→△①","bet_type":"exacta","band":"片方が5～9.9倍","n":18,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△②→△①","bet_type":"exacta","band":"片方が10～19.9倍","n":24,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△②→☆","bet_type":"exacta","band":"片方が10～19.9倍","n":27,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△②→☆","bet_type":"exacta","band":"片方が20～49.9倍","n":26,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△②→☆","bet_type":"exacta","band":"両馬10倍以上","n":31,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"☆→◎","bet_type":"exacta","band":"片方が20～49.9倍","n":13,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"☆→○","bet_type":"exacta","band":"片方が20～49.9倍","n":19,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"☆→▲","bet_type":"exacta","band":"片方が10～19.9倍","n":22,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"☆→▲","bet_type":"exacta","band":"片方が20～49.9倍","n":22,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"☆→▲","bet_type":"exacta","band":"両馬10倍以上","n":21,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"☆→△②","bet_type":"exacta","band":"片方が10～19.9倍","n":27,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"☆→△②","bet_type":"exacta","band":"片方が20～49.9倍","n":26,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"☆→△②","bet_type":"exacta","band":"両馬10倍以上","n":31,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○－△①","bet_type":"quinella","band":"片方が10～19.9倍","n":18,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○－☆","bet_type":"quinella","band":"片方が10～19.9倍","n":19,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○－☆","bet_type":"quinella","band":"片方が50倍以上","n":19,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○－☆","bet_type":"quinella","band":"両馬10倍以上","n":14,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"▲－△②","bet_type":"quinella","band":"片方が5～9.9倍","n":21,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"▲－☆","bet_type":"quinella","band":"片方が50倍以上","n":19,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△①－☆","bet_type":"quinella","band":"片方が50倍以上","n":20,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○→△①","bet_type":"exacta","band":"片方が10～19.9倍","n":18,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○→☆","bet_type":"exacta","band":"片方が10～19.9倍","n":19,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○→☆","bet_type":"exacta","band":"片方が50倍以上","n":19,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○→☆","bet_type":"exacta","band":"両馬10倍以上","n":14,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"▲→△②","bet_type":"exacta","band":"片方が5～9.9倍","n":21,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"▲→☆","bet_type":"exacta","band":"片方が50倍以上","n":19,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△①→○","bet_type":"exacta","band":"片方が10～19.9倍","n":18,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△①→☆","bet_type":"exacta","band":"片方が50倍以上","n":20,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△②→▲","bet_type":"exacta","band":"片方が5～9.9倍","n":21,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"☆→○","bet_type":"exacta","band":"片方が10～19.9倍","n":19,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"☆→○","bet_type":"exacta","band":"片方が50倍以上","n":19,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"☆→○","bet_type":"exacta","band":"両馬10倍以上","n":14,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"☆→▲","bet_type":"exacta","band":"片方が50倍以上","n":19,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"☆→△①","bet_type":"exacta","band":"片方が50倍以上","n":20,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎－△②","bet_type":"quinella","band":"片方が20～49.9倍","n":14,"hits":1,"hit_rate":7.1,"roi":282.1,"avg_payout":3950.0},{"division":"中央","ticket":"◎－☆","bet_type":"quinella","band":"片方が10～19.9倍","n":14,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○－△①","bet_type":"quinella","band":"両馬5倍未満","n":4,"hits":1,"hit_rate":25.0,"roi":112.5,"avg_payout":450.0},{"division":"中央","ticket":"○－△②","bet_type":"quinella","band":"片方が20～49.9倍","n":19,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"▲－△②","bet_type":"quinella","band":"片方が20～49.9倍","n":22,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△①－☆","bet_type":"quinella","band":"片方が10～19.9倍","n":21,"hits":1,"hit_rate":4.8,"roi":254.3,"avg_payout":5340.0},{"division":"中央","ticket":"◎→△②","bet_type":"exacta","band":"片方が20～49.9倍","n":14,"hits":1,"hit_rate":7.1,"roi":451.4,"avg_payout":6320.0},{"division":"中央","ticket":"◎→☆","bet_type":"exacta","band":"片方が10～19.9倍","n":14,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○→△①","bet_type":"exacta","band":"両馬5倍未満","n":4,"hits":1,"hit_rate":25.0,"roi":220.0,"avg_payout":880.0},{"division":"中央","ticket":"○→△②","bet_type":"exacta","band":"片方が20～49.9倍","n":19,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"▲→△②","bet_type":"exacta","band":"片方が20～49.9倍","n":22,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△①→○","bet_type":"exacta","band":"両馬5倍未満","n":4,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△①→☆","bet_type":"exacta","band":"片方が10～19.9倍","n":21,"hits":1,"hit_rate":4.8,"roi":516.7,"avg_payout":10850.0},{"division":"中央","ticket":"△②→◎","bet_type":"exacta","band":"片方が20～49.9倍","n":14,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△②→○","bet_type":"exacta","band":"片方が20～49.9倍","n":19,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△②→▲","bet_type":"exacta","band":"片方が20～49.9倍","n":22,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"☆→◎","bet_type":"exacta","band":"片方が10～19.9倍","n":14,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"☆→△①","bet_type":"exacta","band":"片方が10～19.9倍","n":21,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎－○","bet_type":"quinella","band":"片方が20～49.9倍","n":5,"hits":1,"hit_rate":20.0,"roi":538.0,"avg_payout":2690.0},{"division":"中央","ticket":"○－▲","bet_type":"quinella","band":"片方が20～49.9倍","n":15,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○－△①","bet_type":"quinella","band":"両馬10倍以上","n":11,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○－△②","bet_type":"quinella","band":"両馬10倍以上","n":14,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎→○","bet_type":"exacta","band":"片方が20～49.9倍","n":5,"hits":1,"hit_rate":20.0,"roi":678.0,"avg_payout":3390.0},{"division":"中央","ticket":"○→◎","bet_type":"exacta","band":"片方が20～49.9倍","n":5,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○→▲","bet_type":"exacta","band":"片方が20～49.9倍","n":15,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○→△①","bet_type":"exacta","band":"両馬10倍以上","n":11,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○→△②","bet_type":"exacta","band":"両馬10倍以上","n":14,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"▲→○","bet_type":"exacta","band":"片方が20～49.9倍","n":15,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△①→○","bet_type":"exacta","band":"両馬10倍以上","n":11,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△②→○","bet_type":"exacta","band":"両馬10倍以上","n":14,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎－△①","bet_type":"quinella","band":"片方が50倍以上","n":7,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○－△①","bet_type":"quinella","band":"片方が50倍以上","n":12,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"▲－△①","bet_type":"quinella","band":"片方が50倍以上","n":12,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△①－☆","bet_type":"quinella","band":"両馬10倍以上","n":15,"hits":1,"hit_rate":6.7,"roi":356.0,"avg_payout":5340.0},{"division":"中央","ticket":"◎→△①","bet_type":"exacta","band":"片方が50倍以上","n":7,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○→△①","bet_type":"exacta","band":"片方が50倍以上","n":12,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"▲→△①","bet_type":"exacta","band":"片方が50倍以上","n":12,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△①→◎","bet_type":"exacta","band":"片方が50倍以上","n":7,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△①→○","bet_type":"exacta","band":"片方が50倍以上","n":12,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△①→▲","bet_type":"exacta","band":"片方が50倍以上","n":12,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△①→☆","bet_type":"exacta","band":"両馬10倍以上","n":15,"hits":1,"hit_rate":6.7,"roi":723.3,"avg_payout":10850.0},{"division":"中央","ticket":"☆→△①","bet_type":"exacta","band":"両馬10倍以上","n":15,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎－○","bet_type":"quinella","band":"片方が10～19.9倍","n":10,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎－○","bet_type":"quinella","band":"片方が50倍以上","n":4,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎－○","bet_type":"quinella","band":"両馬10倍以上","n":1,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎－△①","bet_type":"quinella","band":"片方が10～19.9倍","n":13,"hits":2,"hit_rate":15.4,"roi":348.5,"avg_payout":2265.0},{"division":"中央","ticket":"◎－△①","bet_type":"quinella","band":"両馬10倍以上","n":3,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎－☆","bet_type":"quinella","band":"両馬10倍以上","n":4,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○－▲","bet_type":"quinella","band":"片方が50倍以上","n":12,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎→○","bet_type":"exacta","band":"片方が10～19.9倍","n":10,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎→○","bet_type":"exacta","band":"片方が50倍以上","n":4,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎→○","bet_type":"exacta","band":"両馬10倍以上","n":1,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎→△①","bet_type":"exacta","band":"片方が10～19.9倍","n":13,"hits":1,"hit_rate":7.7,"roi":100.8,"avg_payout":1310.0},{"division":"中央","ticket":"◎→△①","bet_type":"exacta","band":"両馬10倍以上","n":3,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎→☆","bet_type":"exacta","band":"両馬10倍以上","n":4,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○→◎","bet_type":"exacta","band":"片方が10～19.9倍","n":10,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○→◎","bet_type":"exacta","band":"片方が50倍以上","n":4,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○→◎","bet_type":"exacta","band":"両馬10倍以上","n":1,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○→▲","bet_type":"exacta","band":"片方が50倍以上","n":12,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"▲→○","bet_type":"exacta","band":"片方が50倍以上","n":12,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△①→◎","bet_type":"exacta","band":"片方が10～19.9倍","n":13,"hits":1,"hit_rate":7.7,"roi":646.2,"avg_payout":8400.0},{"division":"中央","ticket":"△①→◎","bet_type":"exacta","band":"両馬10倍以上","n":3,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"☆→◎","bet_type":"exacta","band":"両馬10倍以上","n":4,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○－△②","bet_type":"quinella","band":"両馬5倍未満","n":2,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"▲－△①","bet_type":"quinella","band":"両馬10倍以上","n":11,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○→△②","bet_type":"exacta","band":"両馬5倍未満","n":2,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"▲→△①","bet_type":"exacta","band":"両馬10倍以上","n":11,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△①→▲","bet_type":"exacta","band":"両馬10倍以上","n":11,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△②→○","bet_type":"exacta","band":"両馬5倍未満","n":2,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎－☆","bet_type":"quinella","band":"片方が50倍以上","n":13,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎→☆","bet_type":"exacta","band":"片方が50倍以上","n":13,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"☆→◎","bet_type":"exacta","band":"片方が50倍以上","n":13,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎－△②","bet_type":"quinella","band":"両馬5倍未満","n":2,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"▲－△②","bet_type":"quinella","band":"両馬5倍未満","n":2,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎→△②","bet_type":"exacta","band":"両馬5倍未満","n":2,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"▲→△②","bet_type":"exacta","band":"両馬5倍未満","n":2,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△②→◎","bet_type":"exacta","band":"両馬5倍未満","n":2,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△②→▲","bet_type":"exacta","band":"両馬5倍未満","n":2,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎－○","bet_type":"quinella","band":"両馬5倍未満","n":7,"hits":3,"hit_rate":42.9,"roi":170.0,"avg_payout":396.7},{"division":"中央","ticket":"◎→○","bet_type":"exacta","band":"両馬5倍未満","n":7,"hits":2,"hit_rate":28.6,"roi":242.9,"avg_payout":850.0},{"division":"中央","ticket":"○→◎","bet_type":"exacta","band":"両馬5倍未満","n":7,"hits":1,"hit_rate":14.3,"roi":127.1,"avg_payout":890.0},{"division":"中央","ticket":"◎－☆","bet_type":"quinella","band":"両馬5倍未満","n":4,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎→☆","bet_type":"exacta","band":"両馬5倍未満","n":4,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"☆→◎","bet_type":"exacta","band":"両馬5倍未満","n":4,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎－▲","bet_type":"quinella","band":"片方が20～49.9倍","n":10,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎－▲","bet_type":"quinella","band":"両馬10倍以上","n":4,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎－△②","bet_type":"quinella","band":"両馬10倍以上","n":4,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎→▲","bet_type":"exacta","band":"片方が20～49.9倍","n":10,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎→▲","bet_type":"exacta","band":"両馬10倍以上","n":4,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎→△②","bet_type":"exacta","band":"両馬10倍以上","n":4,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"▲→◎","bet_type":"exacta","band":"片方が20～49.9倍","n":10,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"▲→◎","bet_type":"exacta","band":"両馬10倍以上","n":4,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△②→◎","bet_type":"exacta","band":"両馬10倍以上","n":4,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎－△①","bet_type":"quinella","band":"両馬5倍未満","n":4,"hits":1,"hit_rate":25.0,"roi":155.0,"avg_payout":620.0},{"division":"中央","ticket":"○－▲","bet_type":"quinella","band":"両馬10倍以上","n":8,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎→△①","bet_type":"exacta","band":"両馬5倍未満","n":4,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○→▲","bet_type":"exacta","band":"両馬10倍以上","n":8,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"▲→○","bet_type":"exacta","band":"両馬10倍以上","n":8,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△①→◎","bet_type":"exacta","band":"両馬5倍未満","n":4,"hits":1,"hit_rate":25.0,"roi":252.5,"avg_payout":1010.0},{"division":"中央","ticket":"◎－▲","bet_type":"quinella","band":"片方が50倍以上","n":5,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"◎→▲","bet_type":"exacta","band":"片方が50倍以上","n":5,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"▲→◎","bet_type":"exacta","band":"片方が50倍以上","n":5,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○－▲","bet_type":"quinella","band":"両馬5倍未満","n":4,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○→▲","bet_type":"exacta","band":"両馬5倍未満","n":4,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"▲→○","bet_type":"exacta","band":"両馬5倍未満","n":4,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△①－△②","bet_type":"quinella","band":"両馬5倍未満","n":3,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△①→△②","bet_type":"exacta","band":"両馬5倍未満","n":3,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△②→△①","bet_type":"exacta","band":"両馬5倍未満","n":3,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△②－☆","bet_type":"quinella","band":"両馬5倍未満","n":1,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△②→☆","bet_type":"exacta","band":"両馬5倍未満","n":1,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"☆→△②","bet_type":"exacta","band":"両馬5倍未満","n":1,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"▲－△①","bet_type":"quinella","band":"両馬5倍未満","n":2,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"▲→△①","bet_type":"exacta","band":"両馬5倍未満","n":2,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"△①→▲","bet_type":"exacta","band":"両馬5倍未満","n":2,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"▲－☆","bet_type":"quinella","band":"両馬5倍未満","n":1,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"▲→☆","bet_type":"exacta","band":"両馬5倍未満","n":1,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"☆→▲","bet_type":"exacta","band":"両馬5倍未満","n":1,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○－☆","bet_type":"quinella","band":"両馬5倍未満","n":1,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"○→☆","bet_type":"exacta","band":"両馬5倍未満","n":1,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0},{"division":"中央","ticket":"☆→○","bet_type":"exacta","band":"両馬5倍未満","n":1,"hits":0,"hit_rate":0.0,"roi":0.0,"avg_payout":0.0}],"box_summary":[{"division":"地方","ticket":"ワイドBOX印6頭","bet_type":"wide_box","n":88,"avg_points":15.0,"hits":75,"hit_rate":85.2,"roi":71.5,"avg_payout":1257.9},{"division":"地方","ticket":"馬連BOX印上位3頭","bet_type":"quinella_box","n":88,"avg_points":2.8,"hits":24,"hit_rate":27.3,"roi":95.5,"avg_payout":986.7},{"division":"地方","ticket":"3連複BOX印6頭","bet_type":"trio_box","n":88,"avg_points":20.0,"hits":40,"hit_rate":45.5,"roi":65.9,"avg_payout":2900.8},{"division":"中央","ticket":"ワイドBOX印6頭","bet_type":"wide_box","n":59,"avg_points":15.0,"hits":39,"hit_rate":66.1,"roi":75.0,"avg_payout":1702.1},{"division":"中央","ticket":"馬連BOX印上位3頭","bet_type":"quinella_box","n":59,"avg_points":2.7,"hits":14,"hit_rate":23.7,"roi":131.2,"avg_payout":1471.4},{"division":"中央","ticket":"3連複BOX印6頭","bet_type":"trio_box","n":59,"avg_points":20.0,"hits":15,"hit_rate":25.4,"roi":122.9,"avg_payout":9666.7}]}''')
_TICKET_ROLE_ORDER = {"◎": 0, "○": 1, "○2": 1.1, "▲": 2, "△1": 3, "△2": 4, "✓": 5, "☆": 5}
_TICKET_ROLE_BASE_ORDER = {"◎": 0, "○": 1, "▲": 2, "△1": 3, "△2": 4, "✓": 5, "☆": 5}


def _ticket_role_base(role):
    role = _nar_safe_text(role)
    if role.startswith("○"):
        return "○"
    if role.startswith("△"):
        return role
    return role


def _ticket_role_text(role):
    role = _nar_safe_text(role)
    return {"○2": "○②", "△1": "△①", "△2": "△②", "穴外1": "無印穴①", "穴外2": "無印穴②"}.get(role, role)


def _ticket_pair_label(left, right):
    left_base = _ticket_role_base(left)
    right_base = _ticket_role_base(right)
    ordered = sorted([left_base, right_base], key=lambda value: _TICKET_ROLE_BASE_ORDER.get(value, 99))
    if len(ordered) == 2 and ordered[0] == ordered[1]:
        return f"{_ticket_role_text(ordered[0])}－{_ticket_role_text(ordered[1])}"
    return f"{_ticket_role_text(ordered[0])}－{_ticket_role_text(ordered[1])}"


def _ticket_exacta_label(left, right):
    return f"{_ticket_role_text(_ticket_role_base(left))}→{_ticket_role_text(_ticket_role_base(right))}"


def _ticket_division(race_type="nar"):
    return "中央" if _nar_safe_text(race_type).lower() == "jra" else "地方"


def _ticket_pair_stats_index():
    if "_TICKET_PAIR_STATS_INDEX" not in globals():
        globals()["_TICKET_PAIR_STATS_INDEX"] = {
            (row.get("division"), row.get("bet_type"), row.get("ticket")): row
            for row in _TICKET_RANKING_STATS.get("pair_summary", [])
        }
    return globals()["_TICKET_PAIR_STATS_INDEX"]


def _ticket_band_stats_index():
    if "_TICKET_BAND_STATS_INDEX" not in globals():
        globals()["_TICKET_BAND_STATS_INDEX"] = {
            (row.get("division"), row.get("bet_type"), row.get("ticket"), row.get("band")): row
            for row in _TICKET_RANKING_STATS.get("pair_band_summary", [])
        }
    return globals()["_TICKET_BAND_STATS_INDEX"]


def _ticket_box_stats_index():
    if "_TICKET_BOX_STATS_INDEX" not in globals():
        globals()["_TICKET_BOX_STATS_INDEX"] = {
            (row.get("division"), row.get("bet_type"), row.get("ticket")): row
            for row in _TICKET_RANKING_STATS.get("box_summary", [])
        }
    return globals()["_TICKET_BOX_STATS_INDEX"]


def _ticket_odds_bands(low, high):
    low = _single_odds_safe_float(low)
    high = _single_odds_safe_float(high)
    if low is None or high is None:
        return []
    values = [low, high]
    labels = []
    if all(value < 5 for value in values):
        labels.append("両馬5倍未満")
    if any(5 <= value < 10 for value in values):
        labels.append("片方が5～9.9倍")
    if any(10 <= value < 20 for value in values):
        labels.append("片方が10～19.9倍")
    if any(20 <= value < 50 for value in values):
        labels.append("片方が20～49.9倍")
    if any(value >= 50 for value in values):
        labels.append("片方が50倍以上")
    if all(value >= 10 for value in values):
        labels.append("両馬10倍以上")
    return labels


def _ticket_lookup_band_stat(division, bet_type, ticket, low, high):
    band_index = _ticket_band_stats_index()
    candidates = []
    for band in _ticket_odds_bands(low, high):
        row = band_index.get((division, bet_type, ticket, band))
        if row:
            candidates.append(row)
    if not candidates:
        return None
    # Use the largest sample among matching bands to avoid overrating one-off high payout bands.
    return sorted(candidates, key=lambda row: (float(row.get("n") or 0), float(row.get("hit_rate") or 0)), reverse=True)[0]


def _ticket_sample_weight(n):
    n = float(n or 0)
    if n < 5:
        return 0.15
    if n < 10:
        return 0.35
    if n < 20:
        return 0.65
    return 1.0


def _ticket_confidence_bonus(confidence_summary):
    stars = (confidence_summary or {}).get("stars", "★★★☆☆")
    venue = (confidence_summary or {}).get("venue_eval", "")
    condition = (confidence_summary or {}).get("condition_eval", "")
    bonus = {"★★★★★": 6, "★★★★☆": 4, "★★★☆☆": 1, "★★☆☆☆": -2, "★☆☆☆☆": -4}.get(stars, 0)
    bonus += {"A": 2, "B": 1, "C": -1}.get(venue, 0)
    bonus += {"A": 2, "B": 1, "C": -1}.get(condition, 0)
    return bonus


def _ticket_current_odds_bonus(low, high):
    low = _single_odds_safe_float(low)
    high = _single_odds_safe_float(high)
    if low is None or high is None:
        return -8
    bonus = 0
    if low < 5 and high < 5:
        bonus -= 8
    if high >= 10:
        bonus += 4
    if 10 <= high < 50:
        bonus += 4
    if high >= 50:
        bonus -= 3
    if low >= 10 and high >= 10:
        bonus += 2
    if low <= 2 and high <= 6:
        bonus -= 5
    return bonus


def _horse_first_column(df, names):
    for name in names:
        if name in getattr(df, "columns", []):
            return name
    return None


def _horse_to_numeric(series):
    try:
        return pd.to_numeric(series, errors="coerce")
    except Exception:
        return pd.Series([None] * len(series), index=series.index)


def _horse_rank_series(df, value_column, ascending=False):
    if not value_column or value_column not in df.columns:
        return pd.Series([None] * len(df), index=df.index)
    values = _horse_to_numeric(df[value_column])
    return values.rank(method="min", ascending=ascending, na_option="bottom")


def _horse_numeric_value(row, *keys):
    for key in keys:
        if key in row.index:
            value = _single_odds_safe_float(row.get(key))
            if value is not None:
                return value
    return None


def _horse_rank_value(row, key):
    value = _horse_numeric_value(row, key)
    if value is None:
        return 999
    try:
        return int(value)
    except Exception:
        return 999


def _horse_mark_value(row):
    mark = _nar_safe_text(row.get("最終印", ""))
    return mark if mark in {"◎", "○", "▲", "△", "✓", "☆"} else ""


def _horse_market_warning(row):
    ai_rank = _horse_rank_value(row, "_馬_AI順位")
    market_rank = _horse_rank_value(row, "_馬_市場順位")
    return ai_rank < 999 and market_rank < 999 and (market_rank - ai_rank) >= 3


def _horse_type_group(horse_type):
    text = _nar_safe_text(horse_type)
    if "データ不足" in text:
        return "shortage"
    if "軸" in text:
        return "axis"
    if "安定" in text:
        return "stable"
    if "穴" in text:
        return "hole"
    if "相手" in text:
        return "opponent"
    if "消し" in text:
        return "fade"
    return "other"


def _horse_type_priority(horse_type):
    return {"axis": 0, "stable": 1, "hole": 2, "opponent": 3, "shortage": 8, "fade": 9}.get(_horse_type_group(horse_type), 5)


def _horse_evaluation_frame(df, race_type="nar"):
    work = df.copy()
    if work.empty:
        return work

    horse_no_col = _horse_first_column(work, ["馬番", "馬番号", "horse_no"])
    name_col = _horse_first_column(work, ["馬名", "horse_name"])
    odds_col = _horse_first_column(work, ["単勝オッズ", "オッズ", "odds"])
    ai_score_col = _horse_first_column(work, ["AI点", "AI", "ai_score"])
    total_score_col = _horse_first_column(work, ["総合評価点", "_最終印点", "総合評価", "total_score"])
    market_score_col = _horse_first_column(work, ["市場反映勝率", "推定勝率", "勝率", "win_probability"])

    work["_馬_馬番"] = work[horse_no_col].astype(str).str.strip() if horse_no_col else ""
    work["_馬_馬名"] = work[name_col].astype(str).str.strip() if name_col else ""
    work["_馬_単勝"] = work[odds_col].apply(_single_odds_safe_float) if odds_col else pd.Series([None] * len(work), index=work.index)

    if "AI順位" in work.columns:
        work["_馬_AI順位"] = _horse_to_numeric(work["AI順位"])
    elif ai_score_col:
        work["_馬_AI順位"] = _horse_rank_series(work, ai_score_col, ascending=False)
    else:
        work["_馬_AI順位"] = pd.Series([None] * len(work), index=work.index)

    if "総合評価順位" in work.columns:
        work["_馬_総合順位"] = _horse_to_numeric(work["総合評価順位"])
    elif total_score_col:
        work["_馬_総合順位"] = _horse_rank_series(work, total_score_col, ascending=False)
    else:
        work["_馬_総合順位"] = work["_馬_AI順位"]

    if "勝率順位" in work.columns:
        work["_馬_市場順位"] = _horse_to_numeric(work["勝率順位"])
    elif "市場順位" in work.columns:
        work["_馬_市場順位"] = _horse_to_numeric(work["市場順位"])
    elif market_score_col:
        work["_馬_市場順位"] = _horse_rank_series(work, market_score_col, ascending=False)
    elif "人気" in work.columns:
        work["_馬_市場順位"] = _horse_to_numeric(work["人気"])
    elif odds_col:
        work["_馬_市場順位"] = work["_馬_単勝"].rank(method="min", ascending=True, na_option="bottom")
    else:
        work["_馬_市場順位"] = pd.Series([None] * len(work), index=work.index)
    if "人気" in work.columns:
        popularity_rank = _horse_to_numeric(work["人気"])
        if market_score_col and market_score_col in work.columns:
            market_source = _horse_to_numeric(work[market_score_col])
            work["_馬_市場順位"] = work["_馬_市場順位"].where(market_source.notna(), popularity_rank)
        else:
            work["_馬_市場順位"] = work["_馬_市場順位"].where(work["_馬_市場順位"].notna(), popularity_rank)

    work["_馬_印"] = work.apply(_horse_mark_value, axis=1)
    work["_馬_市場警戒"] = work.apply(_horse_market_warning, axis=1)
    work["_馬タイプ"] = work.apply(lambda row: _horse_classify(row, race_type), axis=1)
    work["_馬コメント"] = work.apply(lambda row: " / ".join(_horse_comment_items(row, row.get("_馬タイプ"), race_type)[:2]), axis=1)
    work["_馬タイプ優先"] = work["_馬タイプ"].apply(_horse_type_priority)
    return work


def _horse_classify(row, race_type="nar"):
    race_type = _nar_safe_text(race_type).lower() or "nar"
    mark = _horse_mark_value(row)
    odds = _horse_numeric_value(row, "_馬_単勝")
    ai_rank = _horse_rank_value(row, "_馬_AI順位")
    total_rank = _horse_rank_value(row, "_馬_総合順位")
    market_rank = _horse_rank_value(row, "_馬_市場順位")
    market_warning = _horse_market_warning(row)

    if _nar_safe_bool(row.get("_地方指数データ不足", False)):
        return "データ不足"

    ai_top = ai_rank <= 3
    total_top = total_rank <= 3
    ai_mid = ai_rank <= 6
    total_mid = total_rank <= 6
    market_top = market_rank <= 3
    market_mid = market_rank <= 6
    top_material = ai_top or total_top or market_top
    mid_material = ai_mid or total_mid or market_mid
    marked = mark in {"◎", "○", "▲", "△", "✓", "☆"}

    if race_type == "jra":
        if ai_rank == 1 and market_top and not market_warning:
            return "軸候補"
        if market_top and (ai_top or total_top) and not market_warning:
            return "安定候補"
    else:
        if ai_rank == 1 and market_top and odds is not None and odds < 5 and not market_warning:
            return "軸候補"
        if market_rank == 1 and ai_rank <= 2 and total_rank <= 3 and odds is not None and odds < 5 and not market_warning:
            return "軸候補"
        if market_top and (ai_top or total_top) and odds is not None and odds < 10:
            return "安定候補"

    if odds is not None and 10 <= odds < 20 and (mid_material or mark in {"△", "☆"}):
        return "中穴警戒"
    if odds is not None and 20 <= odds < 50 and (mid_material or mark in {"△", "☆"}):
        return "穴警戒"
    if odds is not None and odds >= 50 and (market_mid or ai_mid or total_mid or mark in {"△", "☆"}):
        return "大穴警戒・参考"

    if marked and (mid_material or mark in {"◎", "○", "▲", "△"}):
        return "相手候補"
    if top_material and (odds is None or odds < 10):
        return "相手候補"
    return "消し候補"


def _horse_comment_items(row, horse_type, race_type="nar"):
    comments = []
    if _nar_safe_bool(row.get("_地方指数データ不足", False)):
        comments.append("地方指数データ不足")
        if _horse_rank_value(row, "_馬_市場順位") <= 4:
            comments.append("市場評価は別途確認")
        return comments
    odds = _horse_numeric_value(row, "_馬_単勝")
    ai_rank = _horse_rank_value(row, "_馬_AI順位")
    total_rank = _horse_rank_value(row, "_馬_総合順位")
    market_rank = _horse_rank_value(row, "_馬_市場順位")
    market_warning = _horse_market_warning(row)
    group = _horse_type_group(horse_type)

    if market_warning:
        comments.append("能力上位だが市場警戒")
    elif ai_rank <= 3 and market_rank <= 3:
        comments.append("能力と市場評価が一致")
    elif market_rank <= 3:
        comments.append("市場上位で安定")
    elif ai_rank <= 3 or total_rank <= 3:
        comments.append("能力上位")

    if group == "axis":
        comments.append("軸候補として扱いやすい")
    elif group == "stable":
        comments.append("馬券内期待を重視")
    elif group == "hole":
        if odds is not None and odds >= 50:
            comments.append("大穴警戒・参考")
        elif odds is not None and odds >= 20:
            comments.append("穴帯で材料あり")
        else:
            comments.append("中穴帯で材料あり")
    elif group == "opponent":
        comments.append("相手候補として確認")
    elif group == "fade":
        comments.append("AI・市場とも下位")

    seen = []
    for item in comments:
        if item and item not in seen:
            seen.append(item)
    return seen[:2] or ["材料確認"]


def _horse_type_map(df, race_type="nar"):
    work = _horse_evaluation_frame(df, race_type)
    mapping = {}
    for _, row in work.iterrows():
        no = _nar_safe_text(row.get("_馬_馬番"))
        if no:
            mapping[no] = {
                "type": _nar_safe_text(row.get("_馬タイプ")),
                "comment": _nar_safe_text(row.get("_馬コメント")),
                "ai_rank": row.get("_馬_AI順位", None),
                "total_rank": row.get("_馬_総合順位", None),
                "market_rank": row.get("_馬_市場順位", None),
            }
    return mapping


def _horse_unmarked_hole_horses(df, race_type="nar", limit=2):
    work = _horse_evaluation_frame(df, race_type)
    if work.empty:
        return []
    marked = work["_馬_印"].astype(str).isin(["◎", "○", "▲", "△", "✓", "☆"])
    holes = work[(~marked) & (work["_馬タイプ"].astype(str).str.contains("穴", na=False))].copy()
    if holes.empty:
        return []
    holes = holes.sort_values(["_馬タイプ優先", "_馬_市場順位", "_馬_AI順位", "_馬_単勝"], na_position="last").head(limit)
    result = []
    for idx, (_, row) in enumerate(holes.iterrows(), start=1):
        result.append({
            "role": f"穴外{idx}",
            "no": _nar_safe_text(row.get("_馬_馬番")),
            "name": _nar_safe_text(row.get("_馬_馬名")),
            "odds": row.get("_馬_単勝"),
            "type": _nar_safe_text(row.get("_馬タイプ")),
            "comment": _nar_safe_text(row.get("_馬コメント")),
        })
    return result


def _horse_format_rank(value):
    try:
        if pd.isna(value):
            return "-"
        return f"{int(float(value))}位"
    except Exception:
        return "-"


def _horse_attention_display_rows(df, race_type="nar"):
    work = _horse_evaluation_frame(df, race_type)
    if work.empty:
        return []
    normal = work[work["_馬タイプ"] != "消し候補"].copy()
    normal["_is_marked"] = normal["_馬_印"].astype(str).isin(["◎", "○", "▲", "△", "✓", "☆"])
    normal["_is_unmarked_hole"] = (~normal["_is_marked"]) & normal["_馬タイプ"].astype(str).str.contains("穴", na=False)
    base = normal[normal["_is_marked"]].copy()
    extra = normal[normal["_is_unmarked_hole"]].copy()
    base = base.sort_values(["_馬タイプ優先", "_馬_市場順位", "_馬_AI順位", "_馬_単勝"], na_position="last").head(6)
    extra = extra.sort_values(["_馬タイプ優先", "_馬_市場順位", "_馬_AI順位", "_馬_単勝"], na_position="last").head(2)
    selected = pd.concat([base, extra], ignore_index=True).drop_duplicates(subset=["_馬_馬番"], keep="first")
    selected = selected.sort_values(["_馬タイプ優先", "_馬_市場順位", "_馬_AI順位", "_馬_単勝"], na_position="last")
    rows = []
    for _, row in selected.iterrows():
        rows.append({
            "馬番": _nar_safe_text(row.get("_馬_馬番")),
            "印": _nar_safe_text(row.get("_馬_印")) or "無印",
            "馬名": _nar_safe_text(row.get("_馬_馬名")),
            "単勝": _single_odds_format(row.get("_馬_単勝")),
            "AI順位": _horse_format_rank(row.get("_馬_AI順位")),
            "総合順位": _horse_format_rank(row.get("_馬_総合順位")),
            "市場順位": _horse_format_rank(row.get("_馬_市場順位")),
            "馬タイプ": _nar_safe_text(row.get("_馬タイプ")),
            "一言コメント": _nar_safe_text(row.get("_馬コメント")),
        })
    return rows


def print_horse_individual_reference(df, race_type="nar"):
    print("【注目馬評価】")
    print("※表示専用。印・AI点・総合評価・信頼度には影響しません。")
    rows = _horse_attention_display_rows(df, race_type)
    if not rows:
        print("注目馬評価を作成できませんでした。")
        return
    display_df = pd.DataFrame(rows)
    try:
        display(format_result_for_output(display_df))
    except Exception:
        display(display_df)

    try:
        work = _horse_evaluation_frame(df, race_type)
        all_rows = []
        for _, row in work.sort_values(["_馬タイプ優先", "_馬_市場順位", "_馬_AI順位"], na_position="last").iterrows():
            all_rows.append({
                "馬番": _nar_safe_text(row.get("_馬_馬番")),
                "印": _nar_safe_text(row.get("_馬_印")) or "無印",
                "馬名": _nar_safe_text(row.get("_馬_馬名")),
                "単勝": _single_odds_format(row.get("_馬_単勝")),
                "AI順位": _horse_format_rank(row.get("_馬_AI順位")),
                "総合順位": _horse_format_rank(row.get("_馬_総合順位")),
                "市場順位": _horse_format_rank(row.get("_馬_市場順位")),
                "馬タイプ": _nar_safe_text(row.get("_馬タイプ")),
                "一言コメント": _nar_safe_text(row.get("_馬コメント")),
            })
        if all_rows:
            _ticket_display_collapsible_table("全馬タイプを表示", pd.DataFrame(all_rows))
    except Exception:
        pass


def _ticket_attach_horse_type(horse, horse_type_map):
    no = _nar_safe_text((horse or {}).get("no"))
    data = horse_type_map.get(no, {})
    horse["type"] = data.get("type", horse.get("type", ""))
    horse["type_comment"] = data.get("comment", horse.get("comment", ""))
    return horse


def _ticket_type_pair_text(left_type, right_type):
    left = _nar_safe_text(left_type) or "-"
    right = _nar_safe_text(right_type) or "-"
    return f"{left}－{right}"


def _ticket_type_adjustment(left_type, right_type, bet_type):
    groups = {_horse_type_group(left_type), _horse_type_group(right_type)}
    if "fade" in groups:
        return -18
    if groups == {"axis", "stable"}:
        return 8
    if "axis" in groups and "hole" in groups:
        return 8
    if "stable" in groups and "hole" in groups:
        return 10
    if groups == {"hole"}:
        return 2 if bet_type != "exacta" else -2
    if "axis" in groups and "opponent" in groups:
        return 5
    if "stable" in groups and "opponent" in groups:
        return 5
    if "opponent" in groups and "hole" in groups:
        return 4
    if groups == {"opponent"}:
        return 1
    return 0


def _ticket_buying_style_from_types(left_type, right_type):
    groups = {_horse_type_group(left_type), _horse_type_group(right_type)}
    if "fade" in groups:
        return "見送り寄り"
    if groups <= {"axis", "stable"}:
        return "安定重視"
    if ("axis" in groups or "stable" in groups) and ("opponent" in groups or "hole" in groups):
        return "バランス重視"
    if "hole" in groups:
        return "回収率重視"
    return "バランス重視"


def _ticket_type_one_line_comment(row):
    left_type = row.get("_left_type", "")
    right_type = row.get("_right_type", "")
    groups = {_horse_type_group(left_type), _horse_type_group(right_type)}
    if "fade" in groups:
        return "🚫 見送り候補"
    if groups <= {"axis", "stable"}:
        return "🎯 的中重視"
    if "axis" in groups and "hole" in groups:
        return "💰 配当妙味あり"
    if "stable" in groups and "hole" in groups:
        return "👍 バランス型"
    if groups == {"hole"}:
        return "🌟 穴狙い"
    return ""


def _ticket_type_reason(row):
    left_type = row.get("_left_type", "")
    right_type = row.get("_right_type", "")
    groups = {_horse_type_group(left_type), _horse_type_group(right_type)}
    if "fade" in groups:
        return "消し候補を含む"
    if groups <= {"axis", "stable"}:
        return "軸・安定候補の組み合わせ"
    if "axis" in groups and "hole" in groups:
        return "軸候補＋穴警戒"
    if "stable" in groups and "hole" in groups:
        return "安定候補＋穴警戒"
    if groups == {"hole"}:
        return "穴警戒同士で少額向き"
    if "hole" in groups:
        return "穴警戒を含む"
    return ""


def _ticket_score(stats, band_stats, low, high, bet_type, confidence_summary):
    if not stats:
        return 10
    n = float(stats.get("n") or 0)
    hit_rate = float(stats.get("hit_rate") or 0)
    roi = float(stats.get("roi") or 0)
    sample_weight = _ticket_sample_weight(n)
    score = 18 * sample_weight
    score += min(max(hit_rate, 0), 30) / 30 * 22
    score += min(max(roi - 45, 0), 120) / 120 * 35
    score += _ticket_current_odds_bonus(low, high)
    score += _ticket_confidence_bonus(confidence_summary)
    if band_stats and float(band_stats.get("n") or 0) >= 5:
        band_roi = float(band_stats.get("roi") or 0)
        band_hit = float(band_stats.get("hit_rate") or 0)
        if band_roi >= 120:
            score += 8
        elif band_roi >= 90:
            score += 4
        elif band_roi < 50:
            score -= 6
        if band_hit >= hit_rate and band_hit >= 10:
            score += 3
    if bet_type == "exacta":
        score -= 12
    elif bet_type == "quinella":
        score -= 3
    return int(round(max(0, min(100, score))))


def _ticket_judgement(stats, score, low, high, bet_type):
    if not stats:
        return "N：サンプル不足・参考値"
    n = float(stats.get("n") or 0)
    hit_rate = float(stats.get("hit_rate") or 0)
    roi = float(stats.get("roi") or 0)
    low = _single_odds_safe_float(low)
    high = _single_odds_safe_float(high)
    if n < 5:
        return "N：サンプル不足・参考値"
    if n < 10:
        return "C：攻め候補・少額向き"
    if bet_type == "exacta":
        if score >= 45 and roi >= 70:
            return "C：攻め候補・少額向き"
        return "N：サンプル不足・参考値" if score >= 35 else "－：見送り"
    if low is not None and high is not None and low < 5 and high < 5 and roi < 100:
        return "D：妙味薄め"
    if score >= 78 and n >= 20 and roi >= 100 and hit_rate >= 10:
        return "A：本線候補"
    if score >= 62 and n >= 10 and roi >= 75:
        return "B：購入検討"
    if score >= 45 and (roi >= 70 or (high is not None and high >= 10)):
        return "C：攻め候補・少額向き"
    if score >= 35:
        return "D：妙味薄め"
    return "－：見送り"


def _ticket_reason(stats, band_stats, low, high, bet_type):
    if not stats:
        return "過去統計なし"
    reasons = []
    n = int(stats.get("n") or 0)
    roi = float(stats.get("roi") or 0)
    hit_rate = float(stats.get("hit_rate") or 0)
    high = _single_odds_safe_float(high)
    low = _single_odds_safe_float(low)
    if n >= 20:
        reasons.append("サンプル20R以上")
    elif n >= 10:
        reasons.append("サンプル10R以上")
    elif n >= 5:
        reasons.append("サンプル少なめ")
    if roi >= 120:
        reasons.append("過去回収実績")
    elif roi >= 90:
        reasons.append("過去回収は許容範囲")
    if hit_rate >= 15:
        reasons.append("的中率高め")
    if high is not None and high >= 10:
        reasons.append("相手側の配当妙味あり")
    if low is not None and high is not None and low < 5 and high < 5:
        reasons.append("人気馬同士")
    if band_stats and float(band_stats.get("n") or 0) >= 5:
        reasons.append(f"単勝帯ROI{_confidence_format_number(band_stats.get('roi'))}%")
    if bet_type == "exacta":
        reasons.append("馬単は少額向き")
    return " / ".join(reasons[:4])


def _ticket_horse_label(horse):
    role = _ticket_role_text(horse.get("role"))
    no = str(horse.get("no") or "")
    name = str(horse.get("name") or "").strip()
    odds = _single_odds_format(horse.get("odds"))
    name_part = f" {name}" if name else ""
    return f"{role}{no}{name_part}：{odds}倍"


def _ticket_pair_odds_text(left_horse, right_horse):
    return f"{_ticket_horse_label(left_horse)}／{_ticket_horse_label(right_horse)}"


def _ticket_candidate_rows(df, confidence_summary=None, race_type="nar"):
    division = _ticket_division(race_type)
    stats_index = _ticket_pair_stats_index()
    horse_type_map = _horse_type_map(df, race_type)
    horses = [_ticket_attach_horse_type(dict(horse), horse_type_map) for horse in _single_odds_marked_horses(df)]
    horses.extend(_horse_unmarked_hole_horses(df, race_type, limit=2))
    if len(horses) < 2:
        return []
    rows = []

    def add_candidate(kind, bet_type, left_horse, right_horse, ticket):
        low_values = [
            _single_odds_safe_float(left_horse.get("odds")),
            _single_odds_safe_float(right_horse.get("odds")),
        ]
        if any(value is None for value in low_values):
            return
        low = min(low_values)
        high = max(low_values)
        stats = stats_index.get((division, bet_type, ticket))
        band_stats = _ticket_lookup_band_stat(division, bet_type, ticket, low, high)
        base_score = _ticket_score(stats, band_stats, low, high, bet_type, confidence_summary)
        left_type = left_horse.get("type", "")
        right_type = right_horse.get("type", "")
        type_adjustment = _ticket_type_adjustment(left_type, right_type, bet_type)
        score = int(round(max(0, min(100, base_score + type_adjustment))))
        judgement = _ticket_judgement(stats, score, low, high, bet_type)
        stats_data = stats or dict()
        band_data = band_stats or dict()
        rows.append({
            "_score": score,
            "_base_score": base_score,
            "_type_adjustment": type_adjustment,
            "_bet_type": bet_type,
            "_judgement": judgement,
            "_left_type": left_type,
            "_right_type": right_type,
            "_type_pair": _ticket_type_pair_text(left_type, right_type),
            "_buying_style": _ticket_buying_style_from_types(left_type, right_type),
            "_n": int(stats_data.get("n") or 0),
            "_hit_rate": stats_data.get("hit_rate"),
            "_roi": stats_data.get("roi"),
            "_avg_payout": stats_data.get("avg_payout"),
            "_low_odds": low,
            "_high_odds": high,
            "_band_n": int(band_data.get("n") or 0),
            "_band_roi": band_data.get("roi"),
            "買い目": kind,
            "馬タイプ": _ticket_type_pair_text(left_type, right_type),
            "買い方タイプ": _ticket_buying_style_from_types(left_type, right_type),
            "単勝オッズ構成": _ticket_pair_odds_text(left_horse, right_horse),
            "買い目スコア": score,
            "過去成績": (
                f"対象{int(stats_data.get('n') or 0)}R / "
                f"的中率{_confidence_format_number(stats_data.get('hit_rate'))}% / "
                f"回収率{_confidence_format_number(stats_data.get('roi'))}%"
            ),
            "判定": judgement,
            "理由": _ticket_reason(stats, band_stats, low, high, bet_type),
        })

    for left_index, left in enumerate(horses):
        for right in horses[left_index + 1:]:
            ticket = _ticket_pair_label(left.get("role"), right.get("role"))
            label = f"ワイド {_ticket_role_text(left.get('role'))}－{_ticket_role_text(right.get('role'))}"
            add_candidate(label, "wide", left, right, ticket)
            label = f"馬連 {_ticket_role_text(left.get('role'))}－{_ticket_role_text(right.get('role'))}"
            add_candidate(label, "quinella", left, right, ticket)
            exacta_ticket = _ticket_exacta_label(left.get("role"), right.get("role"))
            label = f"馬単 {_ticket_role_text(left.get('role'))}→{_ticket_role_text(right.get('role'))}"
            add_candidate(label, "exacta", left, right, exacta_ticket)
            exacta_ticket = _ticket_exacta_label(right.get("role"), left.get("role"))
            label = f"馬単 {_ticket_role_text(right.get('role'))}→{_ticket_role_text(left.get('role'))}"
            add_candidate(label, "exacta", right, left, exacta_ticket)
    return rows


def _ticket_ranking_sort_key(row):
    judge_rank = {"A": 5, "B": 4, "C": 3, "D": 2, "N": 1, "－": 0}
    judgement = str(row.get("_judgement") or "")
    rank = judge_rank.get(judgement.split("：", 1)[0], 0)
    bet_bonus = {"wide": 8, "quinella": 3, "exacta": 0}.get(row.get("_bet_type"), 0)
    return (rank, int(row.get("_score") or 0) + bet_bonus)


def _ticket_top_rows(rows, confidence_summary=None):
    if not rows:
        return []
    useful = _ticket_ranked_useful_rows(rows)
    if not useful:
        return []

    max_total = 8
    type_limits = {"wide": 5, "quinella": 2, "exacta": 1}
    selected = []
    selected_ids = set()
    counts = {"wide": 0, "quinella": 0, "exacta": 0}

    # A/B判定は優先表示。8件を超える場合はスコア順の上位だけ残す。
    for row in useful:
        if _ticket_judgement_code(row) not in {"A", "B"}:
            continue
        if len(selected) >= max_total:
            break
        selected.append(row)
        selected_ids.add(id(row))
        bet_type = row.get("_bet_type")
        counts[bet_type] = counts.get(bet_type, 0) + 1

    # C判定は券種の偏りを抑えながら追加。
    for row in useful:
        if len(selected) >= max_total:
            break
        if id(row) in selected_ids or _ticket_judgement_code(row) != "C":
            continue
        bet_type = row.get("_bet_type")
        if counts.get(bet_type, 0) >= type_limits.get(bet_type, max_total):
            continue
        selected.append(row)
        selected_ids.add(id(row))
        counts[bet_type] = counts.get(bet_type, 0) + 1

    # 対象券種が少ない場合は枠を空けず、スコア順で補完。
    for row in useful:
        if len(selected) >= max_total:
            break
        if id(row) in selected_ids:
            continue
        selected.append(row)
        selected_ids.add(id(row))

    return sorted(selected, key=_ticket_ranking_sort_key, reverse=True)[:max_total]


def _ticket_judgement_code(row):
    judgement = str(row.get("_judgement") or row.get("判定") or "")
    return judgement.split("：", 1)[0] if judgement else ""


def _ticket_ranked_useful_rows(rows):
    useful = [
        row for row in rows
        if _ticket_judgement_code(row) in {"A", "B", "C"}
    ]
    return sorted(useful, key=_ticket_ranking_sort_key, reverse=True)


def _ticket_next_rows(rows, selected_rows=None):
    ranked = _ticket_ranked_useful_rows(rows)
    selected_ids = {id(row) for row in (selected_rows or [])}
    next_rows = [row for row in ranked if id(row) not in selected_ids]
    return next_rows[:4]


def _ticket_bet_type_text(row):
    bet_type = str(row.get("_bet_type") or "")
    return {"wide": "ワイド", "quinella": "馬連", "exacta": "馬単"}.get(bet_type, bet_type)


def _ticket_combo_text_from_row(row):
    text = str(row.get("買い目") or "")
    parts = text.split(" ", 1)
    return parts[1] if len(parts) > 1 else text


def _ticket_display_judgement(row):
    judgement = str(row.get("_judgement") or row.get("判定") or "")
    if "：" in judgement:
        return judgement.split("：", 1)[1]
    if judgement.startswith("－"):
        return "見送り"
    return judgement or "見送り"


def _ticket_recommendation(row):
    code = _ticket_judgement_code(row)
    score = int(row.get("_score") or 0)
    bet_type = row.get("_bet_type")
    if code == "A":
        return "★★★★★", "【本線】"
    if code == "B":
        return "★★★★☆", "【購入候補】"
    if code == "C":
        if score >= 65 and bet_type != "exacta":
            return "★★★☆☆", "【押さえ】"
        return "★★☆☆☆", "【少額向き】"
    if code in {"D", "N"}:
        return "★☆☆☆☆", "【参考】"
    return "", "【見送り】"


def _ticket_recommendation_text(row):
    stars, label = _ticket_recommendation(row)
    return f"{stars}\n{label}" if stars else label


def _ticket_one_line_comment(row):
    code = _ticket_judgement_code(row)
    n = int(row.get("_n") or 0)
    roi = float(row.get("_roi") or 0)
    hit_rate = float(row.get("_hit_rate") or 0)
    high = _single_odds_safe_float(row.get("_high_odds"))
    low = _single_odds_safe_float(row.get("_low_odds"))
    bet_type = row.get("_bet_type")
    type_comment = _ticket_type_one_line_comment(row)
    if type_comment:
        return type_comment
    if code == "A":
        return "🔥 本線候補"
    if bet_type == "exacta":
        return "⚠ 少額推奨"
    if high is not None and high >= 50:
        return "🌟 穴狙い"
    if roi >= 100 and high is not None and high >= 10:
        return "💰 配当妙味あり"
    if n >= 20 and roi >= 90 and hit_rate >= 15:
        return "📊 実績安定"
    if hit_rate >= 20 and (high is None or high < 10):
        return "🎯 的中重視"
    if code == "B":
        return "👍 バランス型"
    if code == "C":
        return "⚠ 少額推奨"
    if low is not None and high is not None and low < 5 and high < 5:
        return "🎯 的中重視"
    return "🚫 見送り候補"


def _ticket_stat_text(value, suffix=""):
    if value is None or value == "":
        return "-"
    try:
        number = float(value)
        return f"{_confidence_format_number(number)}{suffix}"
    except Exception:
        return f"{value}{suffix}"


def _ticket_average_payout_text(row):
    value = row.get("_avg_payout")
    if value is None or value == "":
        return "-"
    try:
        number = float(value)
        if number <= 0:
            return "-"
        return f"{_confidence_format_number(number)}円"
    except Exception:
        return "-"


def _ticket_short_reason_text(row):
    code = _ticket_judgement_code(row)
    n = int(row.get("_n") or 0)
    roi = float(row.get("_roi") or 0)
    hit_rate = float(row.get("_hit_rate") or 0)
    high = _single_odds_safe_float(row.get("_high_odds"))
    low = _single_odds_safe_float(row.get("_low_odds"))
    bet_type = row.get("_bet_type")
    reasons = []
    if code == "N":
        reasons.append("サンプル不足")
    elif n >= 20:
        reasons.append("サンプル十分")
    elif n >= 10:
        reasons.append("サンプル10R以上")
    elif n >= 5:
        reasons.append("サンプル少なめ")
    if roi >= 120:
        reasons.append("高回収実績あり")
    elif roi >= 100:
        reasons.append("回収率100%以上")
    elif hit_rate >= 20:
        reasons.append("的中率は比較的高い")
    elif roi < 70 and code in {"D", "N"}:
        reasons.append("過去実績は控えめ")
    if high is not None and high >= 50:
        reasons.append("人気薄を含む")
    elif high is not None and high >= 10:
        reasons.append("相手側に配当妙味あり")
    elif low is not None and high is not None and low < 5 and high < 5:
        reasons.append("人気馬同士で妙味薄め")
    if bet_type == "exacta":
        reasons.append("馬単は少額向き")
    type_reason = _ticket_type_reason(row)
    if type_reason and type_reason not in reasons:
        reasons.insert(0, type_reason)
    if not reasons:
        reasons.append("過去成績を参考")
    return "\n".join(f"・{reason}" for reason in reasons[:3])


def _ticket_display_candidate_row(row, rank=None):
    output = {
        "券種": _ticket_bet_type_text(row),
        "組み合わせ": _ticket_combo_text_from_row(row),
        "馬タイプ": row.get("馬タイプ", row.get("_type_pair", "")),
        "買い方タイプ": row.get("買い方タイプ", row.get("_buying_style", "")),
        "単勝オッズ構成": row.get("単勝オッズ構成", ""),
        "おすすめ度": _ticket_recommendation_text(row),
        "判定": _ticket_display_judgement(row),
        "一言コメント": _ticket_one_line_comment(row),
        "サンプル": f"{int(row.get('_n') or 0)}R",
        "的中率": _ticket_stat_text(row.get("_hit_rate"), "%"),
        "回収率": _ticket_stat_text(row.get("_roi"), "%"),
        "平均払戻": _ticket_average_payout_text(row),
        "理由": _ticket_short_reason_text(row),
    }
    if rank is not None:
        output = {"順位": f"{rank}位", **output}
    return output


def _ticket_exclusion_reason(row, selected_ids):
    if id(row) in selected_ids:
        return ""
    code = _ticket_judgement_code(row)
    if code in {"A", "B", "C"}:
        return "表示枠外"
    if code == "D":
        return "D判定のため通常非表示"
    if code == "N":
        return "サンプル不足・参考値"
    return "見送り判定"


def _ticket_all_candidate_display_rows(rows, selected_rows):
    selected_ids = {id(row) for row in selected_rows}
    ranked = sorted(rows, key=_ticket_ranking_sort_key, reverse=True)
    display_rows = []
    for index, row in enumerate(ranked, start=1):
        display_rows.append({
            "順位": f"{index}位",
            "券種": _ticket_bet_type_text(row),
            "印の組み合わせ": _ticket_combo_text_from_row(row),
            "馬タイプ": row.get("馬タイプ", row.get("_type_pair", "")),
            "買い方タイプ": row.get("買い方タイプ", row.get("_buying_style", "")),
            "単勝オッズ構成": row.get("単勝オッズ構成", ""),
            "内部スコア": row.get("買い目スコア", row.get("_score", "")),
            "馬タイプ補正": row.get("_type_adjustment", ""),
            "おすすめ度": _ticket_recommendation_text(row),
            "判定": _ticket_display_judgement(row),
            "一言コメント": _ticket_one_line_comment(row),
            "サンプル数": f"{int(row.get('_n') or 0)}R",
            "的中率": _ticket_stat_text(row.get("_hit_rate"), "%"),
            "回収率": _ticket_stat_text(row.get("_roi"), "%"),
            "除外理由": _ticket_exclusion_reason(row, selected_ids),
        })
    return display_rows


def _ticket_display_collapsible_table(title, df):
    try:
        from IPython.display import HTML, display as ipy_display
        html = df.to_html(index=False, escape=True)
        ipy_display(HTML(f"<details><summary>{title}</summary>{html}</details>"))
    except Exception:
        print(f"【{title}】")
        try:
            display(format_result_for_output(df))
        except Exception:
            display(df)


def _ticket_simple_candidate_rows(rows, start_rank=9):
    simple_rows = []
    for offset, row in enumerate(rows, start=start_rank):
        simple_rows.append({
            "順位": f"{offset}位",
            "券種": _ticket_bet_type_text(row),
            "組み合わせ": _ticket_combo_text_from_row(row),
            "馬タイプ": row.get("馬タイプ", row.get("_type_pair", "")),
            "おすすめ度": _ticket_recommendation_text(row),
            "判定": _ticket_display_judgement(row),
            "一言コメント": _ticket_one_line_comment(row),
        })
    return simple_rows


def _ticket_box_reference_rows(df, confidence_summary=None, race_type="nar"):
    division = _ticket_division(race_type)
    box_index = _ticket_box_stats_index()
    horses = _single_odds_marked_horses(df)
    odds_values = [_single_odds_safe_float(horse.get("odds")) for horse in horses[:6]]
    odds_values = [value for value in odds_values if value is not None]
    if len(odds_values) < 6:
        return []
    series = pd.Series(odds_values, dtype="float64")
    median = float(series.median())
    ge10 = int((series >= 10).sum())
    rows = []
    specs = [
        ("ワイドBOX印6頭", "wide_box", 15),
        ("馬連BOX印上位3頭", "quinella_box", 3),
        ("3連複BOX印6頭", "trio_box", 20),
    ]
    for ticket, bet_type, points in specs:
        stats = box_index.get((division, bet_type, ticket))
        if not stats:
            continue
        roi = float(stats.get("roi") or 0)
        hit_rate = float(stats.get("hit_rate") or 0)
        if race_type != "jra" and ticket == "3連複BOX印6頭":
            judgement = "D：点数負け注意"
        elif roi >= 100 and points <= 15 and ge10 >= 2:
            judgement = "B：購入検討"
        elif roi >= 80 and median >= 7:
            judgement = "C：攻め候補・少額向き"
        else:
            judgement = "D：点数負け注意"
        if judgement.startswith("B") or (ticket == "3連複BOX印6頭" and race_type == "jra" and roi >= 100):
            rows.append({
                "BOX": f"{ticket}（{points}点）",
                "印6頭単勝中央値": _confidence_format_number(median),
                "過去成績": f"対象{int(stats.get('n') or 0)}R / 的中率{_confidence_format_number(hit_rate)}% / 回収率{_confidence_format_number(roi)}%",
                "判定": judgement,
                "メモ": "BOXは点数に対する配当妙味の参考",
            })
    return rows[:2]


def print_ticket_ranking_reference(df, confidence_summary=None, race_type="nar"):
    print("【買い目ランキング】")
    print("※表示専用。AI点・総合評価・印・信頼度には影響しません。")
    rows = _ticket_candidate_rows(df, confidence_summary, race_type)
    top_rows = _ticket_top_rows(rows, confidence_summary)
    if not top_rows:
        print("有力候補なし：見送り")
    else:
        display_rows = []
        for index, row in enumerate(top_rows, start=1):
            display_rows.append(_ticket_display_candidate_row(row, index))
        ranking_df = pd.DataFrame(display_rows)[[
            "順位", "券種", "組み合わせ", "馬タイプ", "買い方タイプ", "単勝オッズ構成", "おすすめ度", "判定", "一言コメント",
            "サンプル", "的中率", "回収率", "平均払戻", "理由",
        ]]
        try:
            display(format_result_for_output(ranking_df))
        except Exception:
            display(ranking_df)

    next_rows = _ticket_next_rows(rows, top_rows)
    if next_rows:
        print("")
        print("【次点候補】")
        next_df = pd.DataFrame(_ticket_simple_candidate_rows(next_rows, start_rank=9))
        try:
            display(format_result_for_output(next_df))
        except Exception:
            display(next_df)

    if rows:
        all_df = pd.DataFrame(_ticket_all_candidate_display_rows(rows, top_rows))
        _ticket_display_collapsible_table("全買い目候補を表示", all_df)

    box_rows = _ticket_box_reference_rows(df, confidence_summary, race_type)
    if box_rows:
        print("")
        print("【BOX参考】")
        try:
            display(format_result_for_output(pd.DataFrame(box_rows)))
        except Exception:
            display(pd.DataFrame(box_rows))









# ===== Ver3.0 Final UI: レース全体から判断する通常画面（表示のみ） =====
def _ver30_safe_float(value):
    try:
        return _single_odds_safe_float(value)
    except Exception:
        try:
            if value is None or pd.isna(value):
                return None
            return float(str(value).replace("倍", "").replace(",", "").strip())
        except Exception:
            return None


def _ver30_num(row, *keys):
    for key in keys:
        if key in row.index:
            value = _ver30_safe_float(row.get(key))
            if value is not None:
                return value
    return None


def _ver30_star(level):
    try:
        level = int(level)
    except Exception:
        level = 1
    level = max(1, min(5, level))
    return "★" * level + "☆" * (5 - level)


def _ver30_rank_level(rank, best=1):
    rank = _ver30_safe_float(rank)
    if rank is None:
        return 1
    if rank <= best:
        return 5
    if rank <= 3:
        return 4
    if rank <= 6:
        return 3
    if rank <= 9:
        return 2
    return 1


def _ver30_format_rank(value):
    try:
        if value is None or pd.isna(value):
            return "-"
        return f"{int(float(value))}位"
    except Exception:
        return "-"


def _ver30_format_odds(value):
    try:
        text = _single_odds_format(value)
    except Exception:
        value = _ver30_safe_float(value)
        text = "-" if value is None else f"{value:.1f}".rstrip("0").rstrip(".")
    return text if text == "-" else f"{text}倍"


def _ver30_text_value(value):
    if _nar_is_missing_scalar(value):
        return ""
    text = str(value).strip()
    if not text or text.lower() == "nan" or text == "-":
        return ""
    return text


def _ver30_pick_text(row, keys):
    for key in keys:
        text = _ver30_text_value(row.get(key, ""))
        if text:
            return text
    return ""


def _ver30_split_combined_training_material(text):
    text = _ver30_text_value(text)
    if not text:
        return "", ""
    parts = [part.strip() for part in text.replace("／", "/").split("/") if part.strip()]
    if len(parts) <= 1:
        return text, ""
    return parts[0], " / ".join(parts[1:])


def _ver30_ai_point_display(row):
    if _nar_safe_bool(row.get("_地方指数データ不足", False)):
        return "データ不足"
    ai_value = _ver30_num(row, "AI点")
    if ai_value is None:
        ai_value = _ver30_num(row, "_馬_AI点")
    if ai_value is None:
        return "-"
    return f"{ai_value:.1f}"


def _ver30_audit_score_display(row):
    value = _ver30_num(row, "ability_display_score", "raw_score", "_raw_score")
    if value is None:
        return "-"
    return f"{value:.1f}"


def _ver30_audit_rank_display(row):
    value = _ver30_num(row, "ai_rank", "AI順位", "_馬_AI順位")
    if value is None:
        return "-"
    return f"{int(value)}位"


def _ver30_audit_bool_label(value):
    if _nar_is_missing_scalar(value):
        return ""
    return "○" if bool(value) else ""


def _ver30_class_shift_short(row):
    text = _ver30_pick_text(row, ["クラス変動", "クラス判定", "クラス"])
    if not text:
        return "-"
    first = text.replace("／", "/").split("/")[0].strip()
    if "降級" in first:
        return "降級"
    if "昇級" in first:
        return "昇級"
    if "同級" in first or "同" in first:
        return "同級"
    return first[:8]


def _ver30_matchup_eval_short(row):
    text = " / ".join(
        _ver30_text_value(row.get(key, ""))
        for key in ["対戦評価", "対戦材料", "対戦", "評価/検討材料", "評価／検討材料", "検討材料", "印理由"]
        if _ver30_text_value(row.get(key, ""))
    )
    if "対戦◎" in text:
        return "対戦◎"
    if any(word in text for word in ["対戦先着", "先着", "直近①に先着", "直近1に先着"]):
        return "対戦先着"
    if any(word in text for word in ["対戦互角", "互角", "五分"]):
        return "対戦互角"
    if any(word in text for word in ["対戦劣勢", "劣勢", "敗戦", "負け", "に敗戦"]):
        return "対戦劣勢"
    score = _ver30_num(row, "対戦補正")
    if score is not None:
        if score > 0:
            return "対戦先着"
        if score < 0:
            return "対戦劣勢"
    return "未評価"


def _ver30_training_eval_short(row):
    combined_text = _ver30_pick_text(row, ["調教/評価/検討材料", "調教／評価／検討材料"])
    training_text = _ver30_pick_text(row, ["調教評価", "調教材料", "追切評価", "追切材料", "状態材料"])
    if not training_text and combined_text:
        training_text, _ = _ver30_split_combined_training_material(combined_text)
    training_text = _ver30_text_value(training_text)
    if not training_text:
        return "未取得"
    if training_text.startswith("A"):
        return "A 好調" if len(training_text) == 1 else training_text[:8]
    if training_text.startswith("B"):
        return "B 良好" if len(training_text) == 1 else training_text[:8]
    if training_text.startswith("C"):
        return "C 平行線" if len(training_text) == 1 else training_text[:8]
    return training_text[:8]


def _ver30_float_value(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        return float(value)
    except Exception:
        return parse_float_from_text(str(value))


def _ver30_format_kg(value):
    number = _ver30_float_value(value)
    if number is None:
        return ""
    return f"{number:.1f}kg"


def _ver30_signed_kg(value):
    number = _ver30_float_value(value)
    if number is None:
        return ""
    if abs(number) < 0.05:
        return "±0.0kg"
    sign = "＋" if number > 0 else "－"
    return f"{sign}{abs(number):.1f}kg"


def _ver30_load_weight_detail(row):
    current = _ver30_float_value(row.get("_display_current_load_weight"))
    if current is None:
        current = _ver30_float_value(row.get("_current_load_weight"))
    if current is None:
        current = _ver30_float_value(row.get("斤量"))
    if current is None:
        return "データなし"

    previous = _ver30_float_value(row.get("_display_previous_load_weight"))
    if previous is None:
        previous = _ver30_float_value(row.get("_previous_load_weight"))
    change = _ver30_float_value(row.get("_display_load_weight_change"))
    if change is None:
        change = _ver30_float_value(row.get("_load_weight_change"))
    if change is None and previous is not None:
        change = current - previous

    current_text = _ver30_format_kg(current)
    if previous is None or change is None:
        return f"{current_text}（前走データなし）"
    return f"{current_text}（前走比{_ver30_signed_kg(change)}）"


def _ver30_jockey_detail(row):
    current = (
        _ver30_text_value(row.get("_display_current_jockey"))
        or _ver30_text_value(row.get("_current_jockey"))
        or _ver30_text_value(row.get("騎手"))
    )
    previous = _ver30_text_value(row.get("_display_previous_jockey")) or _ver30_text_value(row.get("_previous_jockey"))
    if not current:
        return "データなし"
    if not previous:
        return f"{current}【前走データなし】"
    changed_value = row.get("_display_jockey_changed")
    if _ver30_text_value(changed_value) == "":
        changed_value = row.get("_jockey_changed")
    changed_text = _ver30_text_value(changed_value).lower()
    if changed_text in {"pending", "unknown", "hold", "判定保留", "保留"} or _nar_safe_bool(row.get("_jockey_change_pending"), False):
        return f"{current}【判定保留】"
    if changed_text == "":
        changed_value = nar_jockey_changed_value(current, previous)
        if changed_value == "pending":
            return f"{current}【判定保留】"
    changed = _nar_safe_bool(changed_value, False)
    if changed:
        return f"{previous} → {current}【乗り替わり】"
    return f"{current}【継続】"


def _ver30_display_running_style(value):
    style = normalize_running_style(value)
    if style == "逃":
        return "逃げ"
    if style == "先":
        return "先行"
    if style == "差":
        return "差し"
    if style == "追":
        return "追込"
    return _ver30_text_value(value) or "データなし"


def _ver30_short_text(value, max_len=72):
    text = _ver30_text_value(value)
    if len(text) <= max_len:
        return text
    return text[: max_len - 1] + "…"


def _ver30_stable_comment_short(row):
    for key in ["厩舎コメント", "新聞コメント", "馬コメント"]:
        text = _ver30_short_text(row.get(key, ""), max_len=72)
        if text:
            return text
    return "データなし"


def _ver30_material_tags(row, race_type="nar"):
    source_texts = []
    for key in [
        "評価/検討材料",
        "評価／検討材料",
        "検討材料",
        "評価根拠",
        "印理由",
        "対戦材料",
        "クラス根拠",
        "馬場実績",
    ]:
        value = _ver30_text_value(row.get(key, ""))
        if value:
            source_texts.append(value)

    combined_text = _ver30_pick_text(row, ["調教/評価/検討材料", "調教／評価／検討材料"])
    if combined_text:
        training_part, material_part = _ver30_split_combined_training_material(combined_text)
        if str(race_type).lower() == "jra":
            if material_part:
                source_texts.append(material_part)
        else:
            source_texts.append(combined_text)

    text = " / ".join(source_texts)
    tags = []

    def add(tag):
        if tag and tag not in tags:
            tags.append(tag)

    if "対戦◎" in text:
        add("対戦◎")
    elif "先着" in text:
        add("対戦先着")
    elif any(word in text for word in ["互角", "五分"]):
        add("対戦互角")
    elif any(word in text for word in ["劣勢", "敗戦", "負け"]):
        add("対戦劣勢")
    if any(word in text for word in ["高指数", "最高指数", "指数上位", "能力上位"]):
        add("高指数")
    if any(word in text for word in ["距離実績", "距離適性", "距離指数"]):
        add("距離実績")
    elif _ver30_text_value(row.get("距離指数", "")):
        add("距離実績")
    if any(word in text for word in ["コース実績", "コース穴", "コース指数"]):
        add("コース実績")
    elif _ver30_text_value(row.get("コース指数", "")):
        add("コース実績")
    class_short = _ver30_class_shift_short(row)
    if class_short == "降級":
        add("クラス降級")
    elif class_short == "昇級":
        add("クラス昇級")
    if any(word in text for word in ["好気配", "復調", "上向", "動き良", "良化"]):
        add("好気配")
    if any(word in text for word in ["展開向く", "展開", "脚質"]):
        add("展開材料")
    if any(word in text for word in ["馬場", "同馬場", "適性"]):
        add("適性")

    if not tags:
        for raw in text.replace("／", "/").split("/"):
            raw = raw.strip()
            if raw and len(raw) <= 10:
                add(raw)
            if len(tags) >= 4:
                break
    return "／".join(tags[:4]) if tags else "-"


def _ver30_mark_order(mark):
    return {"◎": 0, "○": 1, "▲": 2, "△": 3, "✓": 5, "☆": 5, "": 9}.get(_nar_safe_text(mark), 9)


def _ver30_prepare_horse_frame(df, race_type="nar"):
    work = _horse_evaluation_frame(df, race_type).copy()
    if work.empty:
        return work
    ai_rank = pd.to_numeric(work.get("_馬_AI順位", pd.Series(index=work.index)), errors="coerce")
    total_rank = pd.to_numeric(work.get("_馬_総合順位", pd.Series(index=work.index)), errors="coerce")
    market_rank = pd.to_numeric(work.get("_馬_市場順位", pd.Series(index=work.index)), errors="coerce")
    odds = pd.to_numeric(work.get("_馬_単勝", pd.Series(index=work.index)), errors="coerce")
    mark = work.get("_馬_印", pd.Series("", index=work.index)).fillna("").astype(str)
    horse_type = work.get("_馬タイプ", pd.Series("", index=work.index)).fillna("").astype(str)

    ability = []
    stability = []
    value = []
    market = []
    comments = []
    for idx, row in work.iterrows():
        ai_r = _ver30_num(row, "_馬_AI順位")
        total_r = _ver30_num(row, "_馬_総合順位")
        market_r = _ver30_num(row, "_馬_市場順位")
        odds_v = _ver30_num(row, "_馬_単勝")
        type_text = _nar_safe_text(row.get("_馬タイプ"))
        type_group = _horse_type_group(type_text)
        mark_text = _nar_safe_text(row.get("_馬_印"))

        if _nar_safe_bool(row.get("_地方指数データ不足", False)):
            ability_level = 1
            stability_level = _ver30_rank_level(market_r, 1)
            value_level = 2 if odds_v is not None and odds_v < 10 else 1
            market_level = _ver30_rank_level(market_r, 1)
            ability.append(_ver30_star(ability_level))
            stability.append(_ver30_star(stability_level))
            value.append(_ver30_star(value_level))
            market.append(_ver30_star(market_level))
            comments.append("地方指数データ不足のため、AI点は算出していません。市場評価や脚質は参考情報として確認してください。")
            continue

        ability_level = max(_ver30_rank_level(ai_r, 1), _ver30_rank_level(total_r, 1))
        if type_group == "fade":
            ability_level = min(ability_level, 2)
        ability.append(_ver30_star(ability_level))

        stability_level = _ver30_rank_level(market_r, 1)
        if ai_r is not None and market_r is not None and abs(ai_r - market_r) <= 1:
            stability_level = min(5, stability_level + 1)
        if "休み明け" in _nar_safe_text(row.get("レース間隔")):
            stability_level = max(1, stability_level - 1)
        if type_group == "axis":
            stability_level = max(stability_level, 4)
        elif type_group == "stable":
            stability_level = max(stability_level, 4)
        elif type_group == "fade":
            stability_level = min(stability_level, 2)
        stability.append(_ver30_star(stability_level))

        material = (
            (ai_r is not None and ai_r <= 6)
            or (total_r is not None and total_r <= 6)
            or (market_r is not None and market_r <= 6)
            or mark_text in {"◎", "○", "▲", "△", "✓", "☆"}
        )
        if odds_v is None:
            value_level = 2 if material else 1
        elif odds_v < 5:
            value_level = 2 if material else 1
        elif odds_v < 10:
            value_level = 3 if material else 2
        elif odds_v < 20:
            value_level = 4 if material else 2
        elif odds_v < 50:
            value_level = 4 if material else 2
        else:
            value_level = 2 if material else 1
        if type_group == "hole" and odds_v is not None and odds_v < 50:
            value_level = min(5, value_level + 1)
        if type_group == "fade":
            value_level = min(value_level, 2)
        value.append(_ver30_star(value_level))

        market_level = _ver30_rank_level(market_r, 1)
        market.append(_ver30_star(market_level))

        comments.append(_ver30_horse_comment(row, ability_level, stability_level, value_level, market_level))

    work["_Ver30能力評価"] = ability
    work["_Ver30安定評価"] = stability
    work["_Ver30妙味評価"] = value
    work["_Ver30市場評価"] = market
    work["_Ver30コメント"] = comments
    work["_Ver30妙味Lv"] = [s.count("★") for s in value]
    work["_Ver30能力Lv"] = [s.count("★") for s in ability]
    work["_Ver30安定Lv"] = [s.count("★") for s in stability]
    work["_Ver30市場Lv"] = [s.count("★") for s in market]
    work["_Ver30印順"] = mark.map(_ver30_mark_order)
    work["_Ver30タイプ優先"] = work.get("_馬タイプ優先", pd.Series(5, index=work.index))
    return work


def _ver30_material_notes(row, limit=2):
    text_parts = []
    for key in ["評価/検討材料", "調教/評価/検討材料", "状態材料", "クラス根拠", "馬場実績", "クラス変動"]:
        value = _nar_safe_text(row.get(key))
        if value and value != "nan":
            text_parts.append(value)
    text = " / ".join(text_parts)
    notes = []
    if "クラス降級" in text or _nar_safe_text(row.get("クラス変動")) == "クラス降級":
        notes.append("クラス降級")
    if "距離" in text or _ver30_num(row, "距離指数") is not None:
        notes.append("距離適性")
    if "コース" in text or _ver30_num(row, "コース指数") is not None:
        notes.append("コース実績")
    if any(word in text for word in ["復調", "上向", "好気配", "動き良", "良化"]):
        notes.append("復調気配")
    if any(word in text for word in ["高指数", "最高指数", "能力上位", "指数上位"]):
        notes.append("高指数")
    if "展開向く" in text or _nar_safe_text(row.get("展開印")):
        notes.append("展開向き")
    if "対戦" in text:
        notes.append("対戦材料")
    seen = []
    for note in notes:
        if note not in seen:
            seen.append(note)
    return seen[:limit]


def _ver30_material_phrase(row):
    notes = _ver30_material_notes(row, limit=2)
    if not notes:
        return ""
    return "、".join(notes)


def _ver30_horse_comment(row, ability_level, stability_level, value_level, market_level):
    if _nar_safe_bool(row.get("_地方指数データ不足", False)):
        return "地方指数データ不足のため、AI点は算出していません。市場評価や脚質は参考情報として確認してください。"
    ai_r = _ver30_num(row, "_馬_AI順位")
    market_r = _ver30_num(row, "_馬_市場順位")
    odds_v = _ver30_num(row, "_馬_単勝")
    type_group = _horse_type_group(row.get("_馬タイプ", ""))
    material = _ver30_material_phrase(row)
    odds_text = _ver30_format_odds(odds_v) if odds_v is not None else "-"
    if ai_r is not None and market_r is not None and ai_r <= 3 and market_r <= 3:
        if odds_v is not None and odds_v >= 10:
            return f"能力評価と市場評価が一致し、単勝{odds_text}なら配当面も確認したい馬です。"
        return "能力評価と市場評価が一致しています。"
    if ai_r is not None and market_r is not None and ai_r <= 3 and market_r >= 4:
        if odds_v is not None and odds_v >= 10:
            return f"能力評価は高く、単勝{odds_text}なら配当面も注目できます。"
        return "能力評価は高い一方で、市場評価とのズレがあります。"
    if type_group == "axis":
        return "能力と市場評価をあわせて中心候補として確認したい馬です。"
    if type_group == "stable":
        if material:
            return f"{material}があり、馬券内期待を確認したい馬です。"
        return "安定材料があり、馬券内期待を確認したい馬です。"
    if type_group == "hole":
        if odds_v is not None and odds_v >= 50:
            return "大穴警戒・参考として材料を確認したい馬です。"
        if odds_v is not None and odds_v >= 10:
            if material:
                return f"{material}が残り、単勝{odds_text}なら配当面も確認したい馬です。"
            return f"中穴帯で材料が残り、単勝{odds_text}なら配当面も確認したい馬です。"
        return "展開次第で相手候補として注目したい馬です。"
    if type_group == "opponent":
        if material:
            return f"{material}を確認し、相手候補として注目したい馬です。"
        return "相手候補として注目したい馬です。"
    if market_r is not None and market_r <= 4:
        return "AI評価に対して市場では一定の評価があります。"
    if ability_level <= 2 and market_level <= 2:
        return "評価下位のため、材料の確認が必要です。"
    if odds_v is not None and odds_v >= 10 and ability_level >= 3:
        return f"市場評価は低めですが、単勝{odds_text}以上の走りに注意したい馬です。"
    return "今回の扱い方を馬評価で確認したい馬です。"


def print_ver30_all_horse_rating(df, race_type="nar"):
    print("【馬評価（全頭）】")
    work = _ver30_prepare_horse_frame(df, race_type)
    if work.empty:
        print("馬評価を作成できませんでした。")
        return
    rows = []
    for _, row in work.sort_values(["_Ver30印順", "_Ver30能力Lv", "_Ver30市場Lv"], ascending=[True, False, False]).iterrows():
        base = {
            "馬番": row.get("_馬_馬番", ""),
            "印": _nar_safe_text(row.get("display_mark")) or "無印",
            "表示印": _nar_safe_text(row.get("display_mark")),
            "馬名": _nar_safe_text(row.get("_馬_馬名")),
            "馬年齢": _ver30_text_value(row.get("馬年齢", "")) or _ver30_text_value(row.get("性齢", "")) or "データなし",
            "騎手": _ver30_text_value(row.get("騎手", "")) or "―",
            "脚質": _ver30_display_running_style(row.get("脚質", "")),
            "単勝オッズ": _ver30_format_odds(row.get("_馬_単勝")),
            "斤量詳細": _ver30_load_weight_detail(row),
            "騎手詳細": _ver30_jockey_detail(row),
            "能力評価": row.get("_Ver30能力評価", ""),
            "安定評価": row.get("_Ver30安定評価", ""),
            "市場評価": row.get("_Ver30市場評価", ""),
            "能力評価値": _ver30_audit_score_display(row),
            "能力ランク": _nar_safe_text(row.get("ability_rank")) or "-",
            "能力ランク理由": _nar_safe_text(row.get("ability_rank_reason")) or "-",
            "勢いランク": _nar_safe_text(row.get("momentum_rank")) or "-",
            "勢いスコア": row.get("momentum_score", ""),
            "勢い理由": _nar_safe_text(row.get("momentum_reason")) or "-",
            "近3走傾向": _nar_safe_text(row.get("recent3_trend")) or "-",
            "総合ランク": _nar_safe_text(row.get("overall_rank")) or "-",
            "総合ランク理由": _nar_safe_text(row.get("overall_rank_reason")) or "-",
            "AI順位": _ver30_audit_rank_display(row),
            "軸信頼度": _nar_safe_text(row.get("axis_confidence")) or "-",
            "軸信頼度理由": _nar_safe_text(row.get("axis_confidence_reason")) or "-",
            "能力帯": _nar_safe_text(row.get("ability_band")) or "-",
            "能力差": _nar_safe_text(row.get("ability_gap_level")) or "-",
            "レース難易度": _nar_safe_text(row.get("race_difficulty")) or "-",
            "レース難易度理由": _nar_safe_text(row.get("race_difficulty_reason")) or "-",
            "AI点": _ver30_ai_point_display(row),
            "クラス変動": _ver30_class_shift_short(row),
            "チェック項目": _nar_safe_text(row.get("チェック項目")) or "-",
            "補足": _nar_safe_text(row.get("補足")) or "なし",
        }
        if str(race_type).lower() == "jra":
            base["調教評価"] = _ver30_training_eval_short(row)
            base["厩舎コメント"] = _ver30_stable_comment_short(row)
        else:
            base["対戦評価"] = _ver30_matchup_eval_short(row)
        base.update({
            "評価／検討材料": _ver30_material_tags(row, race_type),
            "馬タイプ": _nar_safe_text(row.get("_馬タイプ")),
            "穴候補": _ver30_audit_bool_label(row.get("hole_candidate")),
            "注意馬": _ver30_audit_bool_label(row.get("watch_horse")),
            "表示コメント": _nar_safe_text(row.get("display_comment")),
            "一言コメント": _nar_safe_text(row.get("display_comment")) or _nar_safe_text(row.get("_Ver30コメント")),
        })
        rows.append(base)
    rating_df = pd.DataFrame(rows)
    try:
        display(format_result_for_output(rating_df))
    except Exception:
        display(rating_df)


def _ver30_horse_label(row):
    no = _nar_safe_text(row.get("_馬_馬番"))
    name = _nar_safe_text(row.get("_馬_馬名"))
    return f"{no}番 {name}".strip()


def print_ver30_attention_horses(df, race_type="nar"):
    print("【注目馬】")
    work = _ver30_prepare_horse_frame(df, race_type)
    if work.empty:
        print("注目馬を作成できませんでした。")
        return
    marked = work.get("_馬_印", pd.Series("", index=work.index)).fillna("").astype(str).isin(["◎", "○", "▲", "△", "✓", "☆"])
    unmarked_hole = (~marked) & work["_馬タイプ"].astype(str).str.contains("穴", na=False)
    work["_Ver30注目点"] = (
        (5 - work["_Ver30タイプ優先"].fillna(5)) * 10
        + work["_Ver30能力Lv"].fillna(1) * 4
        + work["_Ver30市場Lv"].fillna(1) * 3
        + work["_Ver30妙味Lv"].fillna(1) * 2
        + marked.astype(int) * 12
        + unmarked_hole.astype(int) * 5
    )
    selected = work[marked].sort_values("_Ver30注目点", ascending=False).head(4).copy()
    if len(selected) < 4:
        extra = work[unmarked_hole & ~work.index.isin(selected.index)].sort_values("_Ver30注目点", ascending=False).head(1)
        selected = pd.concat([selected, extra]).head(4)
    if selected.empty:
        print("明確な注目馬は絞り込めませんでした。")
        return
    for _, row in selected.iterrows():
        mark = _nar_safe_text(row.get("display_mark")) or "無印"
        print("")
        print(_ver30_horse_label(row))
        print(f"印：{mark}")
        print(_ver30_attention_comment(row))


def _ver30_attention_comment(row):
    horse_type = _nar_safe_text(row.get("_馬タイプ"))
    value_level = _nar_safe_int(row.get("_Ver30妙味Lv"), 1)
    ability_level = _nar_safe_int(row.get("_Ver30能力Lv"), 1)
    market_level = _nar_safe_int(row.get("_Ver30市場Lv"), 1)
    odds_v = _ver30_num(row, "_馬_単勝")
    odds_text = _ver30_format_odds(odds_v) if odds_v is not None else "-"
    material = _ver30_material_phrase(row)
    if _horse_type_group(horse_type) == "axis":
        return "能力評価と市場評価がかみ合っており、今回の中心候補として確認したい馬です。"
    if _horse_type_group(horse_type) == "stable":
        if material:
            return f"{material}を評価でき、相手候補としても注目できます。"
        return "安定材料があり、相手候補としても注目できます。"
    if _horse_type_group(horse_type) == "hole" or value_level >= 4:
        if odds_v is not None and odds_v >= 10:
            if material:
                return f"{material}があり、単勝{odds_text}なら配当面も含めて確認したい馬です。"
            return f"単勝{odds_text}の中穴候補として、展開や適性が噛み合うか確認したい馬です。"
    return "相手候補として、展開や適性が噛み合うか確認したい馬です。"
    if ability_level >= 4 and market_level <= 2:
        return "能力評価に対して市場評価とのズレがあり、扱いを慎重に確認したい馬です。"
    if material:
        return f"{material}を材料に、馬評価と展開をあわせて確認したい馬です。"
    return "相手候補として、馬評価と展開をあわせて確認したい馬です。"


def _ver30_names_by_style(df, style, limit=3):
    if "脚質" not in df.columns:
        return []
    source = df.copy()
    source["_style"] = source["脚質"].map(normalize_running_style)
    source = source[source["_style"].eq(style)].copy()
    if source.empty:
        return []
    order_col = "_最終印順" if "_最終印順" in source.columns else None
    if order_col:
        source = source.sort_values(order_col)
    return [f"{str(row.get('馬番', '')).strip()}番{str(row.get('馬名', '')).strip()}" for _, row in source.head(limit).iterrows()]


def _ver30_join_names(names):
    names = [name for name in names if name and not name.startswith("番")]
    if not names:
        return ""
    if len(names) == 1:
        return names[0]
    return "、".join(names)


def _ver30_style_counts(df):
    if "脚質" not in df.columns:
        return {"逃": 0, "先": 0, "差": 0, "追": 0}
    styles = df["脚質"].map(normalize_running_style)
    return {key: int((styles == key).sum()) for key in ["逃", "先", "差", "追"]}


def _ver30_style_overview(counts):
    escape = counts.get("逃", 0)
    lead = counts.get("先", 0)
    stalk = counts.get("差", 0)
    closer = counts.get("追", 0)
    front = escape + lead
    back = stalk + closer
    if front == 0 and back > 0:
        return "今回は逃げ・先行型が少なく、前半は落ち着いた流れになる可能性があります。"
    if escape >= 2:
        return "逃げ候補が複数いるため、序盤の主導権争いでペースが緩みにくい可能性があります。"
    if lead <= 1 and back >= 4:
        return "先行型が少なく差し・追い込み型が多いため、前の位置を取れる馬の扱いを確認したい構成です。"
    if front >= 5:
        return "前へ行きたい馬が多く、序盤から位置取りが忙しくなる可能性があります。"
    if stalk >= 4:
        return "差し型が多く、3コーナー以降の進出タイミングが重要になりそうです。"
    if closer >= 4:
        return "追い込み型が多く、直線だけでなく道中の位置取りも確認したいレースです。"
    return "脚質構成は極端ではなく、序盤の並びと3コーナー以降の動きがポイントになりそうです。"


def print_ver30_ai_race_review(df, race_info=None, running_style_info=None, confidence_summary=None, race_type="nar"):
    print("【AIレース考察】")
    work = _ver30_prepare_horse_frame(df, race_type)
    if work.empty:
        print("評価対象を取得できないため、展開考察は控えめに確認してください。")
        return
    if confidence_summary is None:
        confidence_summary = {"stars": "★★★☆☆", "has_honmei": True}

    counts = _ver30_style_counts(df)
    escape = _ver30_names_by_style(df, "逃", 2)
    lead = _ver30_names_by_style(df, "先", 3)
    stalk = _ver30_names_by_style(df, "差", 3)
    closer = _ver30_names_by_style(df, "追", 3)
    style_available = sum(counts.values()) > 0

    center = work.sort_values(["_Ver30タイプ優先", "_馬_AI順位", "_馬_市場順位"], na_position="last").head(2)
    center_text = _ver30_join_names([_ver30_horse_label(row) for _, row in center.iterrows()])
    holes = work[work["_馬タイプ"].astype(str).str.contains("穴", na=False)].sort_values(["_Ver30妙味Lv", "_馬_単勝"], ascending=[False, True]).head(2)
    hole_text = _ver30_join_names([_ver30_horse_label(row) for _, row in holes.iterrows()])
    watch_mask = (
        work.get("_馬_印", pd.Series("", index=work.index)).fillna("").astype(str).eq("✓")
        | work.get("展開印", pd.Series("", index=work.index)).fillna("").astype(str).ne("")
        | work.get("クラス変動", pd.Series("", index=work.index)).fillna("").astype(str).eq("クラス降級")
    )
    watch_pool = work[watch_mask & ~work.index.isin(center.index)].sort_values(
        ["_Ver30能力Lv", "_Ver30市場Lv", "_馬_単勝"], ascending=[False, False, True], na_position="last"
    ).head(3)
    watch_text = _ver30_join_names([_ver30_horse_label(row) for _, row in watch_pool.iterrows()])

    if not style_available:
        print("脚質データ不足のため、展開は参考程度に確認します。")
        if not confidence_summary.get("has_honmei", True):
            print("市場との評価乖離が大きく、本命不在の混戦評価として確認します。")
        print(f"中心候補は{center_text or '上位評価馬'}で、相手候補との差を馬評価で確認したいレースです。")
        if hole_text:
            print(f"妙味候補としては{hole_text}の扱いも確認したいです。")
        return

    escape_text = _ver30_join_names(escape) or "逃げ候補"
    lead_text = _ver30_join_names(lead) or "先行勢"
    stalk_text = _ver30_join_names(stalk) or "差し勢"
    closer_text = _ver30_join_names(closer) or "追い込み勢"
    pace_text = ""
    if running_style_info:
        pace_text = str(running_style_info.get("ペース", "") or running_style_info.get("流れ", "") or "")

    print(_ver30_style_overview(counts))
    if counts.get("逃", 0) == 0 and counts.get("先", 0) == 0:
        print("スタート直後から明確に前を主張する馬は少なく、位置取りは騎手の判断に左右される可能性があります。")
    elif counts.get("逃", 0) == 0:
        print(f"スタートでは明確な逃げ候補が少なく、{lead_text}が自然に前の位置を取る形を想定します。")
    else:
        print(f"スタートでは{escape_text}が前の位置を取り、{lead_text}がその直後につける形を想定します。")
    if pace_text:
        print(f"序盤は{pace_text}を前提に、極端な決めつけはせず位置取りを確認します。")
    else:
        print("序盤は脚質構成を踏まえ、隊列が固まるまでの位置取りを確認したいです。")
    if counts.get("先", 0) <= 1:
        print(f"向正面では前の隊列が厚くなりにくく、{stalk_text}が早めに射程圏へ入れるかがポイントになります。")
    else:
        print(f"向正面では{lead_text}が好位を維持し、{stalk_text}は前を射程圏に入れながら進む展開が考えられます。")
    if counts.get("差", 0) + counts.get("追", 0) >= 5:
        print(f"3コーナーから4コーナーにかけては、{stalk_text}や{closer_text}の進出タイミングが結果を左右しそうです。")
    else:
        print(f"3コーナーから4コーナーにかけては先行勢が早めに動き、{stalk_text}が外から進出する余地があります。")
    print(f"直線では{center_text or '上位評価馬'}を中心に、好位勢と差し馬の比較になります。")
    if hole_text:
        print(f"ゴール前では上位評価馬が残る可能性を見つつ、{hole_text}のような妙味候補が2着または3着へ入る余地も警戒したいです。")
    else:
        print("ゴール前では上位評価馬の比較を中心に、相手候補の差を確認したいです。")
    if watch_text:
        print(f"展開有利や注意馬では{watch_text}も確認しておきたいです。")
    if not confidence_summary.get("has_honmei", True):
        print("市場との評価乖離が大きく、本命不在の混戦評価です。印上位を絶対視せず、馬評価を確認したいレースです。")
    print("全体としては、展開有利・能力上位・注意馬の材料をあわせて最終判断したい構成です。")


def _ver30_find_value_horse(work):
    if work.empty:
        return None
    candidates = work[
        (pd.to_numeric(work.get("_馬_単勝", pd.Series(index=work.index)), errors="coerce") >= 10)
        & (
            (pd.to_numeric(work.get("_馬_AI順位", pd.Series(index=work.index)), errors="coerce") <= 6)
            | (pd.to_numeric(work.get("_馬_総合順位", pd.Series(index=work.index)), errors="coerce") <= 6)
            | (pd.to_numeric(work.get("_馬_市場順位", pd.Series(index=work.index)), errors="coerce") <= 6)
            | work.get("_馬_印", pd.Series("", index=work.index)).astype(str).isin(["○", "▲", "△", "✓", "☆"])
        )
    ].copy()
    if candidates.empty:
        return None
    candidates["_value_pick"] = candidates["_Ver30妙味Lv"] * 10 + (6 - candidates["_Ver30能力Lv"]).abs()
    return candidates.sort_values(["_Ver30妙味Lv", "_馬_市場順位"], ascending=[False, True]).iloc[0]


def print_ver30_betting_structure(df, confidence_summary=None, race_type="nar"):
    print("【今回の馬券構成】")
    work = _ver30_prepare_horse_frame(df, race_type)
    if work.empty:
        print("無理に購入せず、見送りも選択肢となるレースです。")
        return
    stars = (confidence_summary or {}).get("stars", "★★★☆☆")
    has_honmei = bool((confidence_summary or {}).get("has_honmei", True))
    star_count = str(stars).count("★")
    axis_count = int(work["_馬タイプ"].astype(str).str.contains("軸候補|安定候補", regex=True).sum())
    hole_count = int(work["_馬タイプ"].astype(str).str.contains("穴", na=False).sum())
    value_horse = _ver30_find_value_horse(work)

    if not has_honmei:
        print("ワイド・馬連・3連複を検討できる混戦です。")
        print("ただしBOXは点数が増えるため、印をすべて機械的に買うより、馬評価を確認して候補を絞る方が適しています。")
    elif star_count >= 4 and axis_count >= 1:
        if str(race_type).lower() == "jra":
            print("単勝・ワイド・馬連向きです。")
            print("中央版ではAI1位・◎の単勝成績が地方より良かった監査結果がありますが、今回も◎だけを絶対視せず相手候補を確認します。")
        else:
            print("複勝・ワイド・馬連向きです。")
            print("地方版では市場評価との一致を重視し、注目馬を中心に絞った構成が合うと考えます。")
    elif hole_count >= 2 or star_count <= 2:
        print("ワイド・馬連・3連複を検討できる混戦です。")
        print("能力評価・市場評価・展開材料が分散しているため、BOXを使う場合も点数を抑えて確認したいレースです。")
    else:
        print("ワイド・馬連向きです。")
        print("中心候補はいますが、相手候補の評価差が小さいため、単勝一本よりも相手との組み合わせを検討したいレースです。")

    if value_horse is not None:
        mark = _nar_safe_text(value_horse.get("display_mark")) or "無印"
        label = _ver30_horse_label(value_horse)
        print(f"{mark}・{label}は単勝オッズと能力材料の差を確認できるため、単勝・複勝を少額で検討する選択肢があります。")

    if star_count <= 1:
        print("無理に購入せず、見送りも選択肢となるレースです。")
    print("最終判断では、印だけでなく馬評価・展開・単勝オッズのバランスを確認してください。")


def _watch_mark_text_series(df, column):
    if column in df.columns:
        return df[column].fillna("").astype(str)
    return pd.Series("", index=df.index)


def apply_watch_marks(df, race_type="nar"):
    """Ver3.0 UI layer: keep old watch materials separate from ✓ hole marks.

    This does not change AI点, 総合評価, 補正値, or the top-five mark ordering.
    ✓ is assigned to at most two non-core horses with stronger buying material.
    Other watch-material horses remain unmarked and are kept as 注意馬.
    """
    if df is None or len(df) == 0 or "最終印" not in df.columns:
        return df
    result = df.copy()
    mark = result["最終印"].fillna("").astype(str)
    core_mark = mark.isin(["◎", "○", "▲", "△"])

    # ☆ is no longer shown. Re-evaluate non-core marks as watch-only ✓.
    result.loc[~core_mark & mark.isin(["☆", "✓"]), "最終印"] = ""

    try:
        work = _ver30_prepare_horse_frame(result, race_type)
    except Exception:
        work = pd.DataFrame(index=result.index)
    if work.empty:
        return result

    work = work.reindex(result.index)
    current_mark = result["最終印"].fillna("").astype(str)
    core_mark = current_mark.isin(["◎", "○", "▲", "△"])
    data_shortage = (
        result.get("_地方指数データ不足", pd.Series(False, index=result.index))
        .fillna(False)
        .astype(bool)
    )

    class_text = _watch_mark_text_series(work, "クラス変動") + " / " + _watch_mark_text_series(work, "クラス根拠")
    material_text = (
        _watch_mark_text_series(work, "評価/検討材料")
        + " / "
        + _watch_mark_text_series(work, "調教/評価/検討材料")
        + " / "
        + _watch_mark_text_series(work, "印理由")
        + " / "
        + _watch_mark_text_series(work, "_Ver30コメント")
    )
    pace_text = _watch_mark_text_series(work, "展開印") + " / " + material_text

    ability_level = pd.to_numeric(work.get("_Ver30能力Lv", pd.Series(0, index=work.index)), errors="coerce").fillna(0)
    stability_level = pd.to_numeric(work.get("_Ver30安定Lv", pd.Series(0, index=work.index)), errors="coerce").fillna(0)
    market_level = pd.to_numeric(work.get("_Ver30市場Lv", pd.Series(0, index=work.index)), errors="coerce").fillna(0)
    odds = pd.to_numeric(work.get("_馬_単勝", pd.Series(index=work.index)), errors="coerce")
    horse_type = _watch_mark_text_series(work, "_馬タイプ")

    class_down = class_text.str.contains("クラス降級|相手弱化", na=False)
    single_or_payout_comment = material_text.str.contains("単勝.*検討|単勝.*確認|配当面も確認|配当面も注目|配当面", regex=True, na=False)
    pace_favorable = pace_text.str.contains("展開向く|展開有利|展開材料|先行想定|単騎|展", regex=True, na=False)
    high_eval = ability_level.ge(4) | stability_level.ge(4) | market_level.ge(4)
    other_watch = (
        horse_type.str.contains("穴警戒|大穴警戒|安定候補|相手候補", regex=True, na=False)
        & odds.between(10, 49.9, inclusive="both")
    )

    watch_candidate = (~core_mark) & (~data_shortage) & (class_down | single_or_payout_comment | pace_favorable | high_eval | other_watch)
    if bool(watch_candidate.any()):
        final_score = pd.to_numeric(
            result.get("_最終印点", result.get("総合評価点", pd.Series(0, index=result.index))),
            errors="coerce",
        ).fillna(0)
        market_score = pd.to_numeric(
            result.get("市場反映勝率", result.get("推定勝率", pd.Series(0, index=result.index))),
            errors="coerce",
        ).fillna(0)
        expected_value = pd.to_numeric(result.get("単勝期待値", pd.Series(0, index=result.index)), errors="coerce").fillna(0)
        candidate_score = final_score.copy()
        candidate_score += ability_level.reindex(result.index).fillna(0) * 2.0
        candidate_score += stability_level.reindex(result.index).fillna(0) * 1.0
        candidate_score += market_level.reindex(result.index).fillna(0) * 1.0
        candidate_score += class_down.reindex(result.index).fillna(False).astype(float) * 3.0
        candidate_score += single_or_payout_comment.reindex(result.index).fillna(False).astype(float) * 3.0
        candidate_score += pace_favorable.reindex(result.index).fillna(False).astype(float) * 2.0
        candidate_score += other_watch.reindex(result.index).fillna(False).astype(float) * 1.5
        candidate_score += odds.ge(10).reindex(result.index).fillna(False).astype(float) * 1.0
        candidate_score += odds.ge(20).reindex(result.index).fillna(False).astype(float) * 0.8
        candidate_score += expected_value.ge(1.10).astype(float) * 2.0
        candidate_score += market_score.rank(method="min", ascending=False).le(6).fillna(False).astype(float) * 0.8
        candidate_score = candidate_score.where(watch_candidate, -999999)

        selected_index = candidate_score.sort_values(ascending=False).head(2).index
        selected_mask = pd.Series(False, index=result.index)
        selected_mask.loc[[idx for idx in selected_index if bool(watch_candidate.loc[idx])]] = True
        watch_only = watch_candidate & ~selected_mask

        result.loc[selected_mask, "最終印"] = "✓"
        if "_最終印順" in result.columns:
            result.loc[selected_mask, "_最終印順"] = 5
        if "印理由" in result.columns:
            hole_reason = result.loc[selected_mask, "印理由"].fillna("").astype(str)
            hole_reason = hole_reason.where(hole_reason.eq("") | hole_reason.str.contains("穴候補", na=False), hole_reason + " / 穴候補")
            hole_reason = hole_reason.mask(hole_reason.eq(""), "穴候補")
            result.loc[selected_mask, "印理由"] = hole_reason

            watch_reason = result.loc[watch_only, "印理由"].fillna("").astype(str)
            watch_reason = watch_reason.where(watch_reason.eq("") | watch_reason.str.contains("注意馬", na=False), watch_reason + " / 注意馬")
            watch_reason = watch_reason.mask(watch_reason.eq(""), "注意馬")
            result.loc[watch_only, "印理由"] = watch_reason
    return result


def _target_audit_value(row, *columns):
    for col in columns:
        if col in row.index:
            value = row.get(col)
            if pd.notna(value) and str(value) != "":
                return value
    return ""


def print_target_horse_adjustment_audit(df, horse_no=12, horse_name_keyword="ヤングオーオー"):
    """Print available score/bias details for one horse without changing prediction logic."""
    print("【指定馬調査】")
    if df is None or len(df) == 0:
        print("解析表が空のため、指定馬調査はできませんでした。")
        return
    target = pd.Series(False, index=df.index)
    if horse_no is not None and "馬番" in df.columns:
        target = target | pd.to_numeric(df["馬番"], errors="coerce").eq(int(horse_no))
    if horse_name_keyword and "馬名" in df.columns:
        target = target | df["馬名"].fillna("").astype(str).str.contains(str(horse_name_keyword), na=False)
    rows = df[target].copy()
    if rows.empty:
        print(f"{horse_no}番 {horse_name_keyword} は解析表に見つかりませんでした。対象HTMLをアップロードして再実行してください。")
        return
    row = rows.iloc[0]
    ai_value = pd.to_numeric(pd.Series([_target_audit_value(row, "AI点")]), errors="coerce").iloc[0]
    total_value = pd.to_numeric(pd.Series([_target_audit_value(row, "総合評価", "総合評価点", "_最終印点")]), errors="coerce").iloc[0]
    correction_value = _target_audit_value(row, "補正値")
    if correction_value == "" and pd.notna(ai_value) and pd.notna(total_value):
        correction_value = round(float(total_value) - float(ai_value), 1)

    detail_rows = [
        {"項目": "馬番", "値": _target_audit_value(row, "馬番"), "補足": ""},
        {"項目": "馬名", "値": _target_audit_value(row, "馬名"), "補足": ""},
        {"項目": "最終印", "値": _target_audit_value(row, "最終印"), "補足": "旧最終印です。通常表示の✓は穴候補、注意馬は別表示です。"},
        {"項目": "AI点", "値": _target_audit_value(row, "AI点"), "補足": "タイム指数中心の能力評価。"},
        {"項目": "総合評価", "値": _target_audit_value(row, "総合評価", "総合評価点", "_最終印点"), "補足": "AI点に条件補正を加えた評価。"},
        {"項目": "補正値", "値": correction_value, "補足": "総合評価－AI点。"},
        {"項目": "クラス補正", "値": _target_audit_value(row, "クラス補正"), "補足": _target_audit_value(row, "クラス変動")},
        {"項目": "状態補正", "値": _target_audit_value(row, "状態補正"), "補足": _target_audit_value(row, "レース間隔")},
        {"項目": "展開補正", "値": _target_audit_value(row, "展開補正"), "補足": _target_audit_value(row, "展開印")},
        {"項目": "対戦補正", "値": _target_audit_value(row, "対戦補正"), "補足": _target_audit_value(row, "対戦")},
        {"項目": "距離指数", "値": _target_audit_value(row, "距離指数"), "補足": "補正列ではなく能力材料として確認。"},
        {"項目": "コース指数", "値": _target_audit_value(row, "コース指数"), "補足": "補正列ではなく適性材料として確認。"},
        {"項目": "市場反映勝率", "値": _target_audit_value(row, "市場反映勝率"), "補足": "能力補正ではなく市場込みの表示指標。"},
        {"項目": "単勝期待値", "値": _target_audit_value(row, "単勝期待値"), "補足": "購入判断の参考値。"},
        {"項目": "クラス根拠", "値": _target_audit_value(row, "クラス根拠"), "補足": ""},
        {"項目": "評価/検討材料", "値": _target_audit_value(row, "評価/検討材料", "調教/評価/検討材料"), "補足": ""},
        {"項目": "印理由", "値": _target_audit_value(row, "印理由"), "補足": ""},
    ]
    audit_df = pd.DataFrame(detail_rows)
    try:
        display(format_result_for_output(audit_df))
    except Exception:
        display(audit_df)

    mark_text = str(_target_audit_value(row, "最終印"))
    class_shift = str(_target_audit_value(row, "クラス変動"))
    print("")
    if class_shift == "クラス降級":
        print("クラス降級材料は評価されていますが、最終順位では能力・市場・展開・対戦材料との総合比較で上位5印に届かなかった可能性があります。")
    if mark_text == "✓":
        print("旧最終印が✓の場合は、注意馬候補として監査に保持します。通常表示の✓は穴候補のみです。")
    elif mark_text in ("", "無印"):
        print("無印の場合は、現時点の表示材料だけでは注意馬条件に届いていません。")
    print("AI点そのものの詳細な内部加点ログは現在保持していないため、距離指数・コース指数・近走指数・平均指数・最高指数を根拠として確認してください。")

def print_betting_diagnosis(df, odds_html="", confidence_summary=None, race_type="nar"):
    print("【馬券診断】")
    if not odds_html:
        print("オッズ未取得のため期待値診断は省略")
        return

    odds_data = parse_optional_odds_html(odds_html)
    horses = _ai_confidence_marked_horses(df)
    stars = (confidence_summary or {}).get("stars", "★★★☆☆")
    specs = _recommended_combo_specs(horses, stars, race_type)

    print("オッズHTML: あり")
    print("")
    print("【買い目別オッズ】")
    if not specs:
        print("表示対象の推奨買い目がありません。")
    displayed = 0
    for spec in specs[:15]:
        numbers = [horse["no"] for horse in spec["horses"]]
        odds_value = _lookup_combo_odds(odds_data, spec["bet_type"], numbers)
        if spec["bet_type"] == "tansho" and odds_value is None:
            odds_value = _lookup_combo_odds(odds_data, "tansho", numbers)
        label = _odds_combo_label(numbers)
        print(spec["title"])
        print(label)
        print(_format_odds_range(odds_value))
        print("")
        displayed += 1

    if stars in ("★★★☆☆", "★★☆☆☆", "★☆☆☆☆"):
        print("【BOX点数】")
        print("ワイドBOX 印6頭")
        print("15点")
        print("")
        print("3連複BOX 印6頭")
        print("20点")
        print("")
        print("点数に対して配当妙味が低い場合は見送りも有効")
        print("")

    candidate_specs = []
    by_role = {horse["role"]: horse for horse in horses}
    if by_role.get("◎"):
        for role in ["○", "▲", "△1", "△2", "☆"]:
            if by_role.get(role):
                candidate_specs.append({"bet_type": "wide", "horses": [by_role["◎"], by_role[role]]})
    else:
        top = horses[:6]
        for i in range(len(top)):
            for j in range(i + 1, len(top)):
                candidate_specs.append({"bet_type": "wide", "horses": [top[i], top[j]]})

    candidate_rows = []
    for spec in candidate_specs:
        numbers = [horse["no"] for horse in spec["horses"]]
        odds_value = _lookup_combo_odds(odds_data, "wide", numbers)
        if odds_value is None:
            odds_value = _lookup_combo_odds(odds_data, "umaren", numbers)
        if odds_value is None:
            continue
        rating, note = _expectation_rating(odds_value)
        candidate_rows.append((rating, _odds_average(odds_value) or 0, numbers, odds_value, note))

    print("【期待馬券候補】")
    if not candidate_rows:
        print("現在取得できる組み合わせオッズがありません。")
        return
    candidate_rows.sort(key=lambda item: (item[0], item[1]), reverse=True)
    for rating, _, numbers, odds_value, note in candidate_rows[:5]:
        print(_odds_combo_label(numbers))
        print("現在オッズ")
        print(_format_odds_range(odds_value))
        print("判定")
        print(rating)
        if note:
            print(note)
        print("--------------------------------")


def print_ai_confidence_summary(df, race_info=None, detected_venue="", venue_profile=None, race_type="nar"):
    summary = build_ai_confidence_summary(df, race_info, detected_venue, venue_profile, race_type)
    ai_status = _confidence_diff_status(summary["ai_diff"], race_type)
    total_status = _confidence_diff_status(summary["total_diff"], race_type)
    market_status = _confidence_market_status(summary["market_rank"])
    venue_status = _confidence_eval_status(summary["venue_eval"])
    condition_status = _confidence_eval_status(summary["condition_eval"])

    print("========================")
    print("AI信頼度")
    print("")
    print(summary["stars"])
    print("")
    print("理由")
    print(f"{_confidence_check(ai_status)} AI点差：{_confidence_format_number(summary['ai_diff'])}点")
    print(f"{_confidence_check(total_status)} 総合評価差：{_confidence_format_number(summary['total_diff'])}点")
    print(f"{_confidence_check(market_status)} 市場順位：{_confidence_rank_text(summary['market_rank'])}")
    print(f"{_confidence_check(venue_status)} 会場評価：{summary['venue_eval']}")
    print(f"{_confidence_check(condition_status)} 条件評価：{summary['condition_eval']}（{summary['condition_label']}）")
    if summary["reasons"]:
        for reason in summary["reasons"]:
            print(f"× {reason}")
    if not summary["has_honmei"]:
        print("")
        print("⚠ 本命不在（混戦評価）")
        print("市場警戒により")
        print("◎を設定していません。")
        print("")
        print("軸固定は非推奨です。")
        print("")
        print("印上位馬を中心に")
        print("BOX・相手重視をご検討ください。")
    print("")
    print("推奨")
    print(summary["label"])
    for line in str(summary["comment"]).splitlines():
        print(line)
    print("")
    print("推奨馬券")
    for line in _ai_confidence_ticket_lines(summary["stars"], race_type):
        print(line)
    print("========================")
    return summary

def add_purchase_value_columns(df):
    import numpy as np
    import re

    result = df.copy()
    if result.empty:
        for column in ["推定勝率", "市場反映勝率", "勝率順位", "単勝期待値", "購入判定", "馬場実績"]:
            result[column] = ""
        return result

    data_shortage = (
        result.get("_地方指数データ不足", _nar_local_index_data_shortage_mask(result))
        .fillna(False)
        .astype(bool)
    )
    result["_地方指数データ不足"] = data_shortage

    score = pd.to_numeric(result.get("_最終印点"), errors="coerce").mask(data_shortage)
    odds = pd.to_numeric(
        result.get("単勝オッズ", result.get("オッズ", pd.Series(index=result.index, dtype="float64"))),
        errors="coerce",
    )
    score_min = score.min()
    score_max = score.max()
    if pd.isna(score_min) or pd.isna(score_max) or score_max == score_min:
        normalized = pd.Series(0.5, index=result.index, dtype="float64")
    else:
        normalized = ((score - score_min) / (score_max - score_min)).fillna(0.5)

    valid_score = score.notna() & ~data_shortage
    model_weight = pd.Series(0.0, index=result.index, dtype="float64")
    if bool(valid_score.any()):
        model_weight.loc[valid_score] = np.exp((normalized.loc[valid_score] - normalized.loc[valid_score].max()) * 2.4)
    model_total = float(model_weight.sum())
    model_probability = (
        model_weight / model_total
        if model_total > 0
        else pd.Series(0.0, index=result.index)
    )

    market_raw = pd.Series(0.0, index=result.index, dtype="float64")
    valid_odds = odds.notna() & odds.gt(0) & ~data_shortage
    market_raw.loc[valid_odds] = 1.0 / odds.loc[valid_odds]
    market_total = float(market_raw.sum())
    market_probability = (
        market_raw / market_total if market_total > 0 else model_probability.copy()
    )

    # 勝率は「軸決定」ではなく、印の濃淡・単勝妙味・相手補助に使う。
    probability = (model_probability * 0.5 + market_probability * 0.5).clip(0.001, 0.95)
    probability_cap = pd.Series(0.35, index=result.index, dtype="float64")
    probability_cap.loc[odds.gt(12)] = 0.09
    probability_cap.loc[odds.gt(20)] = 0.035
    probability_cap.loc[odds.gt(35)] = 0.015
    probability_cap.loc[odds.gt(80)] = 0.002
    probability = pd.concat([probability, probability_cap], axis=1).min(axis=1)
    probability.loc[data_shortage] = np.nan

    fair_odds = (1.0 / probability).replace([np.inf, -np.inf], np.nan)
    win_ev = probability * odds
    probability_percent = probability * 100
    probability_rank = probability_percent.rank(method="min", ascending=False)

    style = result.get("脚質", pd.Series("", index=result.index)).fillna("").astype(str)
    pace_mark = result.get("展開印", pd.Series("", index=result.index)).fillna("").astype(str)
    mark = result.get("最終印", pd.Series("", index=result.index)).fillna("").astype(str)
    eligible = ~data_shortage
    marked = mark.ne("") & eligible
    honmei = mark.eq("◎") & eligible
    main_partner = mark.isin(["○", "▲"]) & eligible
    reserve = mark.eq("△") & eligible
    star = mark.eq("✓") & eligible
    top3_probability = probability_rank.le(3) & eligible
    top5_probability = probability_rank.le(5) & eligible
    late_closer_without_pace = style.str.contains("追", na=False) & ~pace_mark.eq("展")

    material_text = (
        result.get("評価/検討材料", pd.Series("", index=result.index)).fillna("").astype(str) + " / "
        + result.get("調教/評価", pd.Series("", index=result.index)).fillna("").astype(str) + " / "
        + result.get("クラス変動", pd.Series("", index=result.index)).fillna("").astype(str) + " / "
        + result.get("馬場実績", pd.Series("", index=result.index)).fillna("").astype(str) + " / "
        + result.get("同馬場実績", pd.Series("", index=result.index)).fillna("").astype(str)
    )
    course_value = pd.to_numeric(result.get("コース指数", pd.Series(index=result.index, dtype="float64")), errors="coerce")
    distance_value = pd.to_numeric(result.get("距離指数", pd.Series(index=result.index, dtype="float64")), errors="coerce")
    course_support = material_text.str.contains("コース実績|コース指数|同馬場実績|馬場実績", na=False) | course_value.notna()
    distance_support = material_text.str.contains("距離実績|距離指数", na=False) | distance_value.notna()
    ability_support = material_text.str.contains("能力上位|指数上位|最高指数|平均指数", na=False)
    class_support = material_text.str.contains("クラス降級|同級|クラス実績", na=False)
    h2h_support = result.get("対戦", pd.Series("", index=result.index)).fillna("").astype(str).str.contains("先着|対戦○|対戦優勢|勝ち越し", na=False)
    condition_axis_support = pd.to_numeric(result.get("_条件軸点", pd.Series(0, index=result.index)), errors="coerce").fillna(0).ge(3)
    wide_material_score = (
        course_support.astype(int)
        + distance_support.astype(int)
        + ability_support.astype(int)
        + class_support.astype(int)
        + h2h_support.astype(int)
        + condition_axis_support.astype(int)
        + pace_mark.eq("展").astype(int)
    )

    # 地方の穴拾いでは、純粋な「推定勝率×オッズ」だけだと中穴が沈みすぎる。
    # オッズがついていて、かつ材料がある馬は妙味値として押し上げる。
    raw_win_ev = win_ev.copy()
    odds_value_bonus = pd.Series(0.0, index=result.index, dtype="float64")
    odds_value_bonus.loc[odds.ge(5)] = 0.10
    odds_value_bonus.loc[odds.ge(8)] = 0.20
    odds_value_bonus.loc[odds.ge(12)] = 0.32
    odds_value_bonus.loc[odds.ge(20)] = 0.45
    odds_value_bonus.loc[odds.ge(30)] = 0.55
    material_value_bonus = wide_material_score.clip(lower=0, upper=5).astype(float) * 0.06
    material_value_bonus += h2h_support.astype(float) * 0.10
    material_value_bonus += condition_axis_support.astype(float) * 0.12
    material_value_bonus = material_value_bonus.where(odds.ge(5), 0.0)
    win_ev = (raw_win_ev + odds_value_bonus + material_value_bonus).clip(lower=0, upper=3.0)

    single_value_mask = (
        marked
        & win_ev.ge(1.18)
        & odds.between(2.0, 20.0, inclusive="both")
        & ~late_closer_without_pace
    )

    h2h_text = result.get("対戦", pd.Series("", index=result.index)).fillna("").astype(str)
    horse_numbers_raw = result.get("馬番", pd.Series(index=result.index, dtype="float64"))
    if not isinstance(horse_numbers_raw, pd.Series):
        horse_numbers_raw = pd.Series(horse_numbers_raw, index=result.index)
    horse_numbers = pd.to_numeric(horse_numbers_raw, errors="coerce")
    circled_map = {
        "①": 1, "②": 2, "③": 3, "④": 4, "⑤": 5, "⑥": 6, "⑦": 7, "⑧": 8, "⑨": 9,
        "⑩": 10, "⑪": 11, "⑫": 12, "⑬": 13, "⑭": 14, "⑮": 15, "⑯": 16, "⑰": 17, "⑱": 18,
    }

    def extract_horse_numbers(text):
        values = set()
        text = str(text or "")
        for value in re.findall(r"\d+", text):
            try:
                number = int(value)
                if 1 <= number <= 18:
                    values.add(number)
            except Exception:
                pass
        for symbol, number in circled_map.items():
            if symbol in text:
                values.add(number)
        return values

    marked_loss_targets = set()
    loss_mask = marked & h2h_text.str.contains("敗戦|負け|先着され", na=False)
    for text in h2h_text[loss_mask]:
        marked_loss_targets.update(extract_horse_numbers(text))
    h2h_direct_support = h2h_text.str.contains("先着|対戦○|対戦優勢|対戦有利|勝ち越し", na=False)
    h2h_cover_mask = (
        ~marked
        & (
            h2h_direct_support
            | horse_numbers.isin(marked_loss_targets).fillna(False)
        )
    )
    h2h_wide_candidate_mask = (
        h2h_cover_mask
        & odds.between(4.0, 35.0, inclusive="both")
        & (wide_material_score.ge(2) | condition_axis_support | h2h_support)
        & ~late_closer_without_pace
    )

    honmei_single_axis_mask = honmei & single_value_mask
    honmei_closer_axis_mask = honmei & late_closer_without_pace
    honmei_axis_mask = honmei & ~honmei_single_axis_mask & ~honmei_closer_axis_mask
    honmei_low_value_mask = honmei_axis_mask & odds.between(1.0, 4.0, inclusive="both") & win_ev.lt(0.95)
    honmei_axis_mask = honmei_axis_mask & ~honmei_low_value_mask
    main_single_partner_mask = main_partner & single_value_mask
    main_partner_strong_mask = main_partner & top5_probability & ~main_single_partner_mask
    main_partner_mask = main_partner & ~main_single_partner_mask & ~main_partner_strong_mask
    reserve_single_mask = reserve & single_value_mask
    reserve_wide_candidate_mask = (
        reserve
        & odds.between(2.0, 12.0, inclusive="both")
        & wide_material_score.ge(3)
        & ~late_closer_without_pace
        & ~reserve_single_mask
    )
    reserve_mask = reserve & ~reserve_single_mask & ~reserve_wide_candidate_mask
    star_single_mask = star & single_value_mask
    star_pace_mask = star & pace_mark.eq("展") & ~star_single_mask
    star_wide_candidate_mask = (
        star
        & odds.between(2.0, 12.0, inclusive="both")
        & wide_material_score.ge(3)
        & ~late_closer_without_pace
        & ~star_single_mask
        & ~star_pace_mask
    )
    star_probability_mask = star & top3_probability & ~star_single_mask & ~star_pace_mask & ~star_wide_candidate_mask
    star_short_odds_mask = star & odds.between(2.0, 10.0, inclusive="both") & ~star_single_mask & ~star_pace_mask & ~star_probability_mask & ~star_wide_candidate_mask
    star_mask = star & ~star_single_mask & ~star_pace_mask & ~star_probability_mask & ~star_short_odds_mask & ~star_wide_candidate_mask
    probability_partner_mask = ~marked & ~h2h_cover_mask & top3_probability

    result["推定勝率"] = probability_percent.round(1)
    result["市場反映勝率"] = result["推定勝率"]
    result["市場反映勝率"] = result["推定勝率"]
    result["勝率順位"] = probability_rank.astype("Int64")
    result["適正オッズ"] = fair_odds.round(1)
    result["単勝期待値"] = win_ev.round(2)
    result["購入判定"] = np.select(
        [
            honmei_single_axis_mask,
            honmei_closer_axis_mask,
            honmei_axis_mask,
            honmei_low_value_mask,
            main_single_partner_mask,
            main_partner_strong_mask,
            main_partner_mask,
            reserve_single_mask,
            reserve_wide_candidate_mask,
            reserve_mask,
            star_single_mask,
            star_pace_mask,
            star_wide_candidate_mask,
            star_probability_mask,
            star_short_odds_mask,
            star_mask,
            h2h_wide_candidate_mask,
            h2h_cover_mask & ~h2h_wide_candidate_mask,
            probability_partner_mask,
        ],
        [
            "単勝妙味/中心候補",
            "追込注意候補",
            "中心候補",
            "評価上位/妙味薄",
            "単勝妙味/相手有力",
            "相手有力",
            "印相手候補",
            "単勝妙味/押さえ候補",
            "ワイド本線候補",
            "押さえ候補",
            "単勝妙味/✓穴候補",
            "展開注意候補",
            "ワイド妙味候補",
            "✓穴候補有力",
            "妙味相手候補",
            "✓穴候補",
            "対戦ワイド候補",
            "対戦押さえ候補",
            "勝率補助候補",
        ],
        default="見送り",
    )
    for column in ["推定勝率", "市場反映勝率", "勝率順位", "適正オッズ", "単勝期待値"]:
        if column in result.columns:
            result.loc[data_shortage, column] = pd.NA
    result.loc[data_shortage, "購入判定"] = "データ不足"
    if "同馬場実績" in result.columns:
        result["馬場実績"] = result["同馬場実績"].fillna("").astype(str)
    elif "馬場適性" in result.columns:
        result["馬場実績"] = result["馬場適性"].fillna("").astype(str)
    else:
        result["馬場実績"] = ""
    return result


def build_win_probability_top3(df, limit=5):
    source = add_purchase_value_columns(df)
    if source.empty:
        return pd.DataFrame()
    pool = source.sort_values(
        ["推定勝率", "_最終印点"], ascending=[False, False]
    ).head(max(int(limit), 1)).copy()
    columns = [
        "勝率順位", "最終印", "展開印", "馬番", "馬名", "脚質",
        "推定勝率", "オッズ", "単勝期待値", "購入判定",
    ]
    return pool[[column for column in columns if column in pool.columns]].reset_index(drop=True)


def analyze_nar_purchase_shape(df):
    if df is None or len(df) == 0:
        return {
            "買い方方針": "見送り寄り",
            "推奨買い方": "材料不足のため見送り",
            "理由": "",
            "軸候補": "なし",
            "妙味候補": "",
            "堅さ": "不明",
        }

    tmp = df.copy()

    def text_series(column):
        if column in tmp.columns:
            return tmp[column].fillna("").astype(str)
        return pd.Series("", index=tmp.index, dtype="object")

    def numeric_series(column):
        if column in tmp.columns:
            return pd.to_numeric(tmp[column], errors="coerce")
        return pd.Series(float("nan"), index=tmp.index, dtype="float64")

    def short_number(value):
        try:
            if pd.isna(value):
                return ""
            return f"{float(value):.1f}".rstrip("0").rstrip(".")
        except Exception:
            return str(value or "")

    def row_label(row):
        number = pd.to_numeric(row.get("馬番"), errors="coerce")
        name = str(row.get("馬名", "") or "").strip()
        if pd.notna(number):
            try:
                no_text = circled_number(int(number))
            except Exception:
                no_text = str(int(number))
        else:
            no_text = str(row.get("馬番", "") or "").strip()
        return f"{no_text}{name}" if name else no_text

    mark = text_series("最終印")
    pace_mark = text_series("展開印")
    odds = numeric_series("単勝オッズ").fillna(numeric_series("オッズ"))
    score = numeric_series("_最終印点").fillna(numeric_series("総合評価点")).fillna(numeric_series("AI点"))
    h2h = text_series("対戦")
    material = (
        text_series("評価/検討材料") + " / "
        + text_series("調教/評価/検討材料") + " / "
        + text_series("印理由") + " / "
        + text_series("展開コメント")
    )
    class_shift = text_series("クラス変動")
    ground = text_series("馬場実績") + " / " + text_series("同馬場実績")
    distance = numeric_series("距離指数")
    course = numeric_series("コース指数")
    ev = numeric_series("単勝期待値")

    h2h_good = h2h.str.contains("対戦○|先着|対戦優勢|対戦有利|勝ち越し", regex=True, na=False)
    course_good = (
        material.str.contains("コース実績|コース指数|同馬場実績", regex=True, na=False)
        | ground.str.contains("実績|経験|同馬場", regex=True, na=False)
        | course.notna()
    )
    distance_good = (
        material.str.contains("距離実績|距離指数", regex=True, na=False)
        | distance.notna()
    )
    ability_good = material.str.contains("能力上位|指数上位|最高指数|平均指数", regex=True, na=False)
    class_good = class_shift.str.contains("クラス降級|同級", regex=True, na=False) | material.str.contains("クラス降級|同級", regex=True, na=False)
    pace_good = pace_mark.eq("展")

    material_count = (
        h2h_good.astype(int)
        + course_good.astype(int)
        + distance_good.astype(int)
        + ability_good.astype(int)
        + class_good.astype(int)
        + pace_good.astype(int)
    )

    marked = mark.ne("")
    main = mark.isin(["◎", "○", "▲"])
    lower = mark.isin(["△", "☆"])
    one_digit = odds.gt(0) & odds.lt(10)
    marked_one_digit_material = marked & one_digit & material_count.ge(2)
    lower_one_digit_material = lower & one_digit & material_count.ge(2)
    main_one_digit = main & one_digit

    marked_count = int(marked.sum())
    main_one_digit_count = int(main_one_digit.sum())
    lower_material_count = int(lower_one_digit_material.sum())
    marked_material_count = int(marked_one_digit_material.sum())

    marked_scores = score[marked & score.notna()]
    score_span = None
    score_gap = None
    if len(marked_scores) >= 2:
        sorted_scores = marked_scores.sort_values(ascending=False)
        score_gap = float(sorted_scores.iloc[0] - sorted_scores.iloc[1])
    if len(marked_scores) >= 4:
        score_span = float(marked_scores.max() - marked_scores.min())

    all_main_over_10 = bool(main.any() and odds[main].notna().all() and odds[main].ge(10).all())
    honmei_mask = mark.eq("◎")
    axis_mask = (
        honmei_mask
        & odds.between(1.5, 5.5, inclusive="both")
        & material_count.ge(3)
        & (pd.Series(score_gap if score_gap is not None else 0, index=tmp.index).ge(6))
    )
    axis_rows = tmp[axis_mask].copy()

    value_mask = (
        marked
        & odds.between(4.0, 35.0, inclusive="both")
        & material_count.ge(2)
        & (
            lower
            | pace_good
            | h2h_good
            | ev.ge(1.05).fillna(False)
        )
    )
    h2h_value_mask = (
        ~marked
        & h2h_good
        & odds.between(4.0, 35.0, inclusive="both")
        & material_count.ge(1)
    )
    value_rows = tmp[value_mask | h2h_value_mask].copy()
    if not value_rows.empty:
        value_rows["_材料数"] = material_count.reindex(value_rows.index).fillna(0)
        value_rows["_下位印"] = lower.reindex(value_rows.index).astype(int)
        value_rows["_対戦穴"] = h2h_value_mask.reindex(value_rows.index).astype(int)
        value_rows["_妙味EV"] = ev.reindex(value_rows.index).fillna(0)
        value_rows["_妙味オッズ"] = odds.reindex(value_rows.index).fillna(0)
        value_rows = value_rows.sort_values(
            ["_下位印", "_対戦穴", "_材料数", "_妙味オッズ", "_妙味EV"],
            ascending=[False, False, False, False, False],
        )
    value_labels = [row_label(row) for _, row in value_rows.head(2).iterrows()]
    axis_labels = [row_label(row) for _, row in axis_rows.head(1).iterrows()]

    horizontal = (
        lower_material_count >= 2
        or (marked_material_count >= 4 and (score_span is None or score_span <= 30))
        or (main_one_digit_count <= 1 and lower_material_count >= 1)
    )
    main_center = (
        not horizontal
        and main_one_digit_count >= 2
        and lower_material_count == 0
    )

    reasons = []
    if main_one_digit_count:
        reasons.append(f"◎○▲一桁{main_one_digit_count}頭")
    if lower_material_count:
        reasons.append(f"下位印の一桁＋材料{lower_material_count}頭")
    if marked_material_count:
        reasons.append(f"印内の一桁＋材料{marked_material_count}頭")
    if score_gap is not None:
        reasons.append(f"首位差{score_gap:.1f}")
    if score_span is not None:
        reasons.append(f"印内点差{score_span:.1f}")
    if all_main_over_10:
        reasons.append("◎○▲が全体的に人気薄")

    if axis_labels and not horizontal:
        policy = "軸あり：少点数"
        firmness = "軸あり"
        recommendation = f"軸候補{axis_labels[0]}から相手2〜3頭へワイド。馬連は配当がある時だけ"
    elif horizontal and value_labels:
        policy = "横並び：単複/少点数"
        firmness = "横並び"
        recommendation = f"軸固定しない。妙味候補{ '・'.join(value_labels) }の単複、または妙味候補同士のワイド。◎○▲ワイドBOX3点は配当確認後"
    elif horizontal:
        policy = "横並び：ワイド3点"
        firmness = "横並び"
        recommendation = "軸固定しない。◎○▲ワイドBOX3点を基本に、△/✓はオッズがつく時だけ追加"
    elif main_center:
        policy = "堅め：◎○▲"
        firmness = "堅め"
        recommendation = "◎○▲ワイドBOX3点中心。低配当なら見送りも含めて確認"
    elif value_labels:
        policy = "穴ワイド：少点数"
        firmness = "穴ワイド"
        recommendation = f"中心固定は弱め。妙味候補{ '・'.join(value_labels) }をワイド相手に少点数で確認"
    elif all_main_over_10:
        policy = "混戦：見送り/単複"
        firmness = "混戦"
        recommendation = "無理に連系で広げない。買うなら妙味候補の単複を少額"
    elif marked_count >= 6:
        policy = "標準：◎○▲ワイド"
        firmness = "標準"
        recommendation = "◎○▲ワイドBOX3点を基本。△/✓は材料が重なる時だけ追加"
    else:
        policy = "見送り寄り"
        firmness = "見送り"
        recommendation = "印・材料不足。買うなら少額"

    return {
        "買い方方針": policy,
        "推奨買い方": recommendation,
        "理由": " / ".join(reasons),
        "軸候補": "・".join(axis_labels) if axis_labels else "なし",
        "妙味候補": "・".join(value_labels),
        "堅さ": firmness,
        "下位印一桁材料数": lower_material_count,
        "印内一桁材料数": marked_material_count,
    }


def print_nar_betting_policy(df):
    policy = analyze_nar_purchase_shape(df)
    print("堅さ：" + policy.get("堅さ", ""))
    print("買い方方針：" + policy.get("買い方方針", ""))
    if policy.get("軸候補") and policy.get("軸候補") != "なし":
        print("軸候補：" + policy["軸候補"])
    if policy.get("妙味候補"):
        print("妙味候補：" + policy["妙味候補"])
    if policy.get("理由"):
        print("方針理由：" + policy["理由"])
    if policy.get("推奨買い方"):
        print("推奨買い方：" + policy["推奨買い方"])
    return policy

def build_purchase_candidate_table(df, limit=8):
    source = add_purchase_value_columns(df)
    source = prepare_nar_display_columns(source)
    if source.empty:
        return pd.DataFrame()

    mark = source.get("最終印", pd.Series("", index=source.index)).fillna("").astype(str)
    valid_marks = ["◎", "○", "▲", "△", "✓", "☆"]
    pool = source[mark.isin(valid_marks)].copy()
    columns = [
        "印", "馬番", "馬名", "騎手", "斤量", "脚質", "レース間隔", "オッズ",
        "AI点", "総合評価", "市場反映勝率", "単勝期待値", "評価根拠", "買い方メモ",
    ]
    if pool.empty:
        return pd.DataFrame(columns=columns)

    pool["印"] = pool["最終印"].fillna("").astype(str)
    mark_order_map = {"◎": 0, "○": 1, "▲": 2, "△": 3, "✓": 5, "☆": 5}
    pool["_印順"] = pd.to_numeric(pool.get("_最終印順", pd.Series(99, index=pool.index)), errors="coerce")
    fallback_order = pool["印"].map(mark_order_map)
    pool["_印順"] = pool["_印順"].where(pool["_印順"].notna(), fallback_order).fillna(99)
    pool = pool.sort_values(["_印順", "_最終印点", "AI点"], ascending=[True, False, False])
    if limit:
        pool = pool.head(max(int(limit), 1))

    if "総合評価点" in pool.columns:
        pool["総合評価"] = pd.to_numeric(pool["総合評価点"], errors="coerce")
    elif "総合評価" in pool.columns:
        pool["総合評価"] = pd.to_numeric(pool["総合評価"], errors="coerce")
    if "市場反映勝率" not in pool.columns and "推定勝率" in pool.columns:
        pool["市場反映勝率"] = pd.to_numeric(pool["推定勝率"], errors="coerce")
    if "評価根拠" not in pool.columns:
        if "評価/検討材料" in pool.columns:
            pool["評価根拠"] = pool["評価/検討材料"].fillna("").astype(str)
        else:
            pool["評価根拠"] = pool.apply(build_nar_evaluation_material, axis=1)

    def memo_for_mark(mark_value):
        return {
            "◎": "軸向き",
            "○": "相手本線",
            "▲": "連下候補",
            "△": "押さえ",
            "✓": "穴候補",
        }.get(str(mark_value or "").strip(), "")

    pool["買い方メモ"] = pool["印"].map(memo_for_mark)
    return pool[[column for column in columns if column in pool.columns]].reset_index(drop=True)


def purchase_candidate_display(df):
    formatted = format_result_for_output(df)
    if "オッズ" not in formatted.columns:
        return formatted
    try:
        return formatted.style.set_properties(
            subset=["オッズ"],
            **{"font-weight": "700"},
        )
    except Exception:
        return formatted

def refresh_horse_pace_comments(df, running_info=None):
    df = df.copy()
    if df.empty:
        df["展開コメント"] = ""
        return df

    running = running_info or analyze_running_style(df)
    counts = running.get("脚質構成", {}) or {}
    pace_horses = set(
        int(value)
        for value in (running.get("展開向く馬番") or running.get("展開穴", []))
        if pd.notna(value)
    )
    styles = df.get("脚質", pd.Series("", index=df.index)).map(normalize_running_style)

    def numeric(column):
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce")
        return pd.Series(float("nan"), index=df.index, dtype="float64")

    ai_rank = numeric("AI順位")
    if ai_rank.isna().all():
        ai_rank = numeric("AI点").rank(method="min", ascending=False)
    popularity = numeric("人気")
    odds = numeric("単勝オッズ")
    distance_rank = numeric("距離指数").rank(method="min", ascending=False)
    course_rank = numeric("コース指数").rank(method="min", ascending=False)
    star_rank = numeric("★最高指数").fillna(numeric("★最高")).rank(method="min", ascending=False)
    layoff = df.get("_is_layoff", pd.Series(False, index=df.index)).fillna(False).astype(bool)

    comments = []
    for idx, row in df.iterrows():
        horse_no = int(row.get("馬番", 0) or 0)
        style = styles.loc[idx]
        rank_value = safe_num(ai_rank.loc[idx], None)
        pop_value = safe_num(popularity.loc[idx], None)
        odds_value = safe_num(odds.loc[idx], None)
        class_shift = str(row.get("クラス変動") or "")
        h2h_text = str(row.get("対戦") or "")
        parts = []

        if horse_no in pace_horses:
            if style == "逃" and counts.get("逃", 0) == 1:
                parts.append("単騎逃げなら展開利")
            else:
                parts.append("今回の展開向く")
        elif style == "逃":
            parts.append("逃げ候補、競られ方が鍵")
        elif style == "先":
            parts.append("好位で運べれば粘り込み")
        elif style == "差":
            parts.append("中団から末脚勝負")
        elif style == "追":
            parts.append("後方からで展開待ち")
        else:
            parts.append("位置取り次第")

            if class_shift == "クラス降級":
                parts.append("クラス降級で妙味")
            elif class_shift == "クラス昇級":
                parts.append("昇級で相手強化")

            if not _nar_is_missing_scalar(rank_value) and rank_value <= 3:
                parts.append("能力上位")
            elif not _nar_is_missing_scalar(rank_value) and rank_value <= 6:
                parts.append("相手圏")

            if h2h_text and h2h_text not in ("対戦不明", "対戦なし"):
                parts.append(h2h_text)

        if layoff.loc[idx]:
            parts.append("休み明けで信頼度割引")

        if safe_num(star_rank.loc[idx], None) is not None and star_rank.loc[idx] <= 3:
            parts.append("同条件実績あり")
        elif safe_num(distance_rank.loc[idx], None) is not None and distance_rank.loc[idx] <= 3:
            parts.append("距離指数上位")
        elif safe_num(course_rank.loc[idx], None) is not None and course_rank.loc[idx] <= 3:
            parts.append("コース指数上位")

        if (
            not _nar_is_missing_scalar(pop_value)
            and not _nar_is_missing_scalar(rank_value)
            and pop_value - rank_value >= 3
        ) or (
            not _nar_is_missing_scalar(odds_value)
            and odds_value >= 8
            and not _nar_is_missing_scalar(rank_value)
            and rank_value <= 8
        ):
            parts.append("オッズ妙味")

        unique = []
        for part in parts:
            if part and part not in unique:
                unique.append(part)
        comments.append("。".join(unique[:5]) + "。")

    df["展開コメント"] = comments
    return df


def build_final_mark_summary_table(df):
    if "最終印" not in df.columns:
        df = add_final_marks(df)
    summary = df[df["最終印"].astype(str).str.len().gt(0)].copy()
    if summary.empty:
        return summary
    summary["_表示順"] = pd.to_numeric(summary.get("_最終印順"), errors="coerce").fillna(99)
    summary = summary.sort_values(["_表示順", "_最終印点"], ascending=[True, False])
    cols = ["最終印", "馬番", "馬名", "AI順位", "AI点", "人気", "単勝オッズ"]
    return summary[[c for c in cols if c in summary.columns]].reset_index(drop=True)


def print_final_mark_summary(df):
    summary = build_final_mark_summary_table(df)
    print("【最終評価】")
    if summary.empty:
        print("最終印を判定できませんでした。")
        return
    for _, row in summary.iterrows():
        horse_no = pd.to_numeric(row.get("馬番"), errors="coerce")
        no_text = str(int(horse_no)) if pd.notna(horse_no) else str(row.get("馬番", "")).strip()
        print(f"{row.get('最終印', '')} {no_text} {row.get('馬名', '')}")



#@title 会場別試験ロジック
# 保存済みレースの会場別検証結果を補助評価へ使用します。
# AI点は変更せず、会場評価は独立した列として表示します。

VENUE_PROFILES = {
    "名古屋": {
        "sample_races": 12,
        "headline": "近3走最高と展開型を重視。脚質単独より指数との組み合わせを確認",
        "primary": "最高指数",
        "secondary": "コース指数",
        "style_available": True,
        "note": "差し馬が多いため、脚質名だけでなく能力と条件指数を併用",
    },
    "水沢": {
        "sample_races": 9,
        "headline": "前走指数とコース指数を重視。前が競れば条件上位の差し浮上に注目",
        "primary": "_last",
        "secondary": "コース指数",
        "style_available": True,
        "note": "脚質だけで位置取りを固定せず、能力・条件上位の早め進出も想定",
    },
    "高知": {
        "sample_races": 5,
        "headline": "近3走最高と前走指数を重視。差し馬の4角押し上げに注目",
        "primary": "最高指数",
        "secondary": "_last",
        "style_available": True,
        "note": "差し中心だが、後方待機のままでは届きにくい",
    },
    "佐賀": {
        "sample_races": 5,
        "headline": "近3走平均と前走指数を重視。先行・差し中心で追込は慎重",
        "primary": "平均指数",
        "secondary": "_last",
        "style_available": True,
        "note": "追込は展開依存が高く、4角好位型を優先",
    },
    "浦和": {
        "sample_races": 7,
        "headline": "能力・近3走平均・前走指数を確認。4角までの位置取り変化を警戒",
        "primary": "平均指数",
        "secondary": "_last",
        "style_available": True,
        "note": "差し・追込表示でも早めに進出する例があり、脚質を固定視しない",
    },
    "金沢": {
        "sample_races": 7,
        "headline": "能力と近3走平均を重視。4角好位へ進出できる馬を確認",
        "primary": "平均指数",
        "secondary": "_last",
        "style_available": True,
        "note": "馬券内馬は4角4番手以内が中心。先行型を評価し、後方依存は慎重",
    },
    "門別": {
        "sample_races": 12,
        "headline": "近3走最高と平均指数を重視。短距離は単騎逃げ、流れる展開は差し追込も確認",
        "primary": "最高指数",
        "secondary": "平均指数",
        "style_available": True,
        "note": "距離・コース単体より近走の指数水準を優先。逃げは少数標本のため軽い補助評価",
    },
    "園田": {
        "sample_races": 11,
        "headline": "能力型合成と条件型合成を重視。4角4番手以内へ運べる馬を確認",
        "primary": "平均指数",
        "secondary": "_last",
        "style_available": True,
        "note": "馬券内馬は4角4番手以内が中心。差し表示でも早め進出できるかを重視",
    },
}


def venue_metric_rank(series):
    numeric = pd.to_numeric(series, errors="coerce")
    return numeric.rank(method="min", ascending=False)


def rank_bonus(rank_value, first=3.0, top3=2.0, top5=1.0):
    if pd.isna(rank_value):
        return 0.0
    if rank_value <= 1:
        return first
    if rank_value <= 3:
        return top3
    if rank_value <= 5:
        return top5
    return 0.0


def venue_reason_label(column):
    return {
        "_last": "前走指数",
        "コース指数": "コース指数",
        "最高指数": "近3走最高",
        "平均指数": "近3走平均",
    }.get(column, column)


def apply_venue_profile(df, race_info, has_style_html=False):
    result = df.copy()
    if "間隔" not in result.columns:
        if "_days_since_last" in result.columns:
            result["間隔"] = result["_days_since_last"].map(format_interval_from_days)
        else:
            result["間隔"] = ""
    if "同馬場実績" not in result.columns:
        result["同馬場実績"] = result.get("馬場適性", pd.Series("", index=result.index)).fillna("")
    if "馬体重" not in result.columns:
        result["馬体重"] = ""
    if "クラス変動" not in result.columns:
        result["クラス変動"] = ""
    venue = str(race_info.get("racecourse") or "").strip()
    profile = VENUE_PROFILES.get(venue)

    ai_rank = pd.to_numeric(result.get("AI順位"), errors="coerce")
    course_rank = venue_metric_rank(result.get("コース指数", pd.Series(index=result.index, dtype="float64")))
    recent_rank = venue_metric_rank(result.get("最高指数", pd.Series(index=result.index, dtype="float64")))
    average_rank = venue_metric_rank(result.get("平均指数", pd.Series(index=result.index, dtype="float64")))
    star_rank = venue_metric_rank(result.get("★最高指数", pd.Series(index=result.index, dtype="float64")))
    star_values = pd.to_numeric(
        result.get("★最高指数", pd.Series(index=result.index, dtype="float64")),
        errors="coerce",
    )
    recent_values = pd.to_numeric(
        result.get("最高指数", pd.Series(index=result.index, dtype="float64")),
        errors="coerce",
    )
    star_benchmark = recent_values.median()
    last_rank = venue_metric_rank(result.get("_last", pd.Series(index=result.index, dtype="float64")))
    styles = result.get("脚質", pd.Series("", index=result.index)).map(normalize_running_style)

    scores = []
    reasons = []
    for idx, row in result.iterrows():
        score = 0.0
        parts = []
        rank = ai_rank.loc[idx]
        age = extract_age_from_sex_age(row.get("性齢"))
        trend = pd.to_numeric(row.get("_trend"), errors="coerce")
        weight_change = pd.to_numeric(row.get("_load_weight_change"), errors="coerce")
        if pd.notna(rank):
            if rank <= 1:
                score += 4.0
                parts.append("能力最上位")
            elif rank <= 3:
                score += 3.0
                parts.append("能力上位")
            elif rank <= 5:
                score += 2.0
            elif rank <= 8:
                score += 1.0

        if profile:
            primary = profile["primary"]
            secondary = profile["secondary"]
            rank_sources = {
                "_last": last_rank,
                "コース指数": course_rank,
                "最高指数": recent_rank,
                "平均指数": average_rank,
            }
            primary_rank = rank_sources[primary].loc[idx]
            secondary_rank = rank_sources[secondary].loc[idx]
            primary_bonus = rank_bonus(primary_rank, 3.0, 2.0, 1.0)
            secondary_bonus = rank_bonus(secondary_rank, 2.5, 1.5, 0.8)
            score += primary_bonus + secondary_bonus
            if pd.notna(primary_rank) and primary_rank <= 3:
                parts.append(f"{venue_reason_label(primary)}上位")
            if pd.notna(secondary_rank) and secondary_rank <= 3:
                parts.append(f"{venue_reason_label(secondary)}上位")

            if venue == "水沢" and has_style_html:
                style = styles.loc[idx]
                if style == "逃":
                    score += 0.2
                    parts.append("前残り警戒")
                elif style == "差":
                    score += 0.9
                    parts.append("差し押上型")
                elif style == "先":
                    score += 0.1
                    parts.append("好位進出型")
                elif style == "追":
                    score -= 0.7
                    parts.append("後方展開待ち")
            elif venue == "高知" and has_style_html:
                style = styles.loc[idx]
                if style == "差":
                    score += 0.8
                    parts.append("差し優勢")
                elif style == "追":
                    score += 0.2
                    parts.append("流れ待ち")
                elif style in ("逃", "先"):
                    score -= 0.3
                    parts.append("前受け試金石")
            elif venue == "佐賀" and has_style_html:
                style = styles.loc[idx]
                if style in ("先", "差"):
                    score += 0.6
                    parts.append("好位進出型")
                elif style == "追":
                    score -= 1.5
                    parts.append("追込不振傾向")
            elif venue == "浦和" and has_style_html:
                style = styles.loc[idx]
                if style == "逃":
                    score += 0.8
                    parts.append("前受け警戒")
                elif style == "先":
                    score += 0.6
                    parts.append("好位候補")
                elif style == "差":
                    score += 0.2
                    parts.append("差し進出型")
                elif style == "追":
                    score -= 0.6
                    parts.append("展開待ち")
            elif venue == "金沢" and has_style_html:
                style = styles.loc[idx]
                if style == "先":
                    score += 0.9
                    parts.append("好位有利")
                elif style == "逃":
                    score += 0.4
                    parts.append("前残り警戒")
                elif style == "差":
                    score += 0.2
                    parts.append("早め進出なら")
                elif style == "追":
                    score -= 0.6
                    parts.append("後方展開待ち")
            elif venue == "門別" and has_style_html:
                style = styles.loc[idx]
                if style == "逃":
                    score += 0.6
                    parts.append("単騎なら注意")
                elif style == "追":
                    score += 0.3
                    parts.append("流れ向けば")
                elif style == "差":
                    score += 0.2
                    parts.append("差し浮上型")
                elif style == "先":
                    score += 0.1
                    parts.append("好位候補")
            elif venue == "園田" and has_style_html:
                style = styles.loc[idx]
                if style == "逃":
                    score += 0.7
                    parts.append("単騎なら注意")
                elif style in ("先", "差"):
                    score += 0.4
                    parts.append("4角好位警戒")
                elif style == "追":
                    score -= 0.5
                    parts.append("後方依存注意")
        else:
            score += rank_bonus(course_rank.loc[idx], 1.5, 1.0, 0.5)
            score += rank_bonus(recent_rank.loc[idx], 1.5, 1.0, 0.5)
            if pd.notna(course_rank.loc[idx]) and course_rank.loc[idx] <= 3:
                parts.append("コース指数上位")
            if pd.notna(recent_rank.loc[idx]) and recent_rank.loc[idx] <= 3:
                parts.append("近3走最高上位")

        star_value = star_values.loc[idx]
        star_is_strong = (
            pd.notna(star_value)
            and pd.notna(star_benchmark)
            and star_value >= star_benchmark
        )
        star_is_usable = (
            pd.notna(star_value)
            and pd.notna(star_benchmark)
            and star_value >= star_benchmark * 0.9
        )
        if star_is_strong:
            score += rank_bonus(star_rank.loc[idx], 4.0, 2.5, 0.5)
            if pd.notna(star_rank.loc[idx]) and star_rank.loc[idx] <= 1:
                parts.append("同条件★最上位")
            elif pd.notna(star_rank.loc[idx]) and star_rank.loc[idx] <= 3:
                parts.append("同条件★上位")
        elif star_is_usable:
            score += rank_bonus(star_rank.loc[idx], 1.5, 1.0, 0.3)
            parts.append("同条件★水準")
        elif pd.notna(star_value):
            parts.append("★指数水準低め")

        if pd.notna(weight_change):
            if weight_change <= -2:
                score += 1.0
                parts.append("斤量2kg以上減")
            elif weight_change <= -1:
                score += 0.5
                parts.append("斤量減")
            elif weight_change >= 2:
                score -= 1.5
                parts.append("斤量2kg以上増")
            elif weight_change >= 1:
                score -= 0.5
                parts.append("斤量増")

        if age is not None and age >= 11:
            score -= 2.5 if pd.notna(trend) and trend >= 4 else 3.5
            parts.append("11歳以上・良化中" if pd.notna(trend) and trend >= 4 else "11歳以上注意")
        elif age is not None and age >= 9:
            score -= 0.5 if pd.notna(trend) and trend >= 4 else 1.5
            parts.append("高齢も良化中" if pd.notna(trend) and trend >= 4 else "高齢注意")

        unique = []
        for part in parts:
            if part not in unique:
                unique.append(part)
        scores.append(round(score, 1))
        reasons.append(" / ".join(unique[:4]) or "会場別強調なし")

    result["_会場評価点"] = scores
    result["_★会場順位"] = star_rank
    result["_★水準十分"] = star_values.ge(star_benchmark) if pd.notna(star_benchmark) else False
    result["会場理由"] = reasons

    support_counts = []
    support_texts = []
    for idx in result.index:
        labels = []
        if pd.notna(recent_rank.loc[idx]) and recent_rank.loc[idx] <= 5:
            labels.append("近3走最高")
        if pd.notna(course_rank.loc[idx]) and course_rank.loc[idx] <= 5:
            labels.append("コース")
        if pd.notna(last_rank.loc[idx]) and last_rank.loc[idx] <= 5:
            labels.append("前走")
        support_counts.append(len(labels))
        support_texts.append(
            f"{len(labels)}/3（{'・'.join(labels)}）"
            if labels
            else "0/3"
        )
    result["_指数裏付け数"] = support_counts
    result["指数裏付け"] = support_texts

    hole_signals = []
    hole_reasons = []
    for idx, row in result.iterrows():
        rank = ai_rank.loc[idx]
        odds = pd.to_numeric(row.get("単勝オッズ"), errors="coerce")
        age = extract_age_from_sex_age(row.get("性齢"))
        age_adjustment = pd.to_numeric(row.get("年齢補正"), errors="coerce")
        relative_light = pd.to_numeric(row.get("_relative_load_weight"), errors="coerce")
        values = row.get("_prev_values")
        valid = [value for value in values if value is not None] if isinstance(values, list) else []
        uplift = max(valid[1:]) - valid[0] if len(valid) >= 2 else 0
        same_flags = row.get("_same_condition_flags")
        same_values = (
            [
                value
                for value, flag in zip(values, same_flags)
                if flag and value is not None
            ]
            if isinstance(values, list) and isinstance(same_flags, list)
            else []
        )
        same_range = max(same_values) - min(same_values) if len(same_values) >= 2 else None
        same_average = (
            sum(same_values) / len(same_values)
            if len(same_values) >= 2
            else None
        )
        index_support = (
            (pd.notna(recent_rank.loc[idx]) and recent_rank.loc[idx] <= 5)
            or (pd.notna(course_rank.loc[idx]) and course_rank.loc[idx] <= 5)
            or (pd.notna(last_rank.loc[idx]) and last_rank.loc[idx] <= 5)
        )
        young_or_light = (
            age in (3, 4)
            or (pd.notna(age_adjustment) and age_adjustment >= 2)
            or (pd.notna(relative_light) and relative_light >= 1.5)
        )
        is_improving_hole = (
            pd.notna(rank)
            and 4 <= rank <= 9
            and pd.notna(odds)
            and 8 <= odds <= 30
            and uplift >= 8
            and (index_support or uplift >= 15)
            and young_or_light
        )
        is_condition_stable_hole = (
            pd.notna(rank)
            and 4 <= rank <= 10
            and pd.notna(odds)
            and 8 <= odds <= 50
            and same_range is not None
            and same_range <= 8
            and same_average is not None
            and (
                pd.isna(star_benchmark)
                or same_average >= star_benchmark * 0.85
            )
        )
        signal_parts = []
        reason_parts = []
        if is_improving_hole:
            parts = ["中穴オッズ", f"指数上積み+{format_number_for_display(uplift)}"]
            if age in (3, 4):
                parts.append(f"{age}歳")
            elif pd.notna(relative_light) and relative_light >= 1.5:
                parts.append("軽斤量")
            if pd.notna(recent_rank.loc[idx]) and recent_rank.loc[idx] <= 5:
                parts.append("近3走最高上位")
            elif pd.notna(course_rank.loc[idx]) and course_rank.loc[idx] <= 5:
                parts.append("コース指数上位")
            else:
                parts.append("前走指数上位")
            signal_parts.append("良化穴")
            reason_parts.extend(parts[:4])
        if is_condition_stable_hole:
            signal_parts.append("条件安定穴")
            reason_parts.extend([
                "同条件2走以上",
                f"同条件振れ幅{format_number_for_display(same_range)}",
                f"同条件平均{format_number_for_display(same_average)}",
            ])
        unique_reasons = []
        for part in reason_parts:
            if part not in unique_reasons:
                unique_reasons.append(part)
        hole_signals.append(" / ".join(signal_parts))
        hole_reasons.append(" / ".join(unique_reasons[:5]))

    result["穴サイン"] = hole_signals
    result["穴理由"] = hole_reasons
    result = result.sort_values(
        ["_会場評価点", "AI点", "AI順位"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    result["材料順位"] = range(1, len(result) + 1)

    def race_style(row):
        odds = pd.to_numeric(row.get("単勝オッズ"), errors="coerce")
        rank = pd.to_numeric(row.get("AI順位"), errors="coerce")
        score = pd.to_numeric(row.get("_会場評価点"), errors="coerce")
        if (
            pd.notna(odds)
            and 5 <= odds <= 30
            and pd.notna(rank)
            and rank <= 6
            and pd.notna(score)
            and score >= 6
        ):
            return "単複型"
        if row.get("穴サイン"):
            return "穴ワイド型"
        return ""

    result["勝負タイプ"] = result.apply(race_style, axis=1)
    result["勝負理由"] = result.apply(
        lambda row: (
            row.get("会場理由", "")
            if row.get("勝負タイプ") == "単複型"
            else row.get("穴理由", "")
            if row.get("勝負タイプ") == "穴ワイド型"
            else ""
        ),
        axis=1,
    )

    def combined_evaluation(row):
        def clean_text(value):
            if pd.isna(value):
                return ""
            text = str(value).strip()
            return "" if text.lower() == "nan" else text

        ai_position = pd.to_numeric(row.get("AI順位"), errors="coerce")
        popularity = pd.to_numeric(row.get("人気"), errors="coerce")
        support_count = pd.to_numeric(row.get("_指数裏付け数"), errors="coerce")
        venue_score = pd.to_numeric(row.get("_会場評価点"), errors="coerce")
        h2h_score = pd.to_numeric(row.get("_h2h_score"), errors="coerce")
        hole_signal = clean_text(row.get("穴サイン", ""))
        race_style = clean_text(row.get("勝負タイプ", ""))
        support_value = int(support_count) if pd.notna(support_count) else 0
        support_text = f"指数{support_value}/3"

        if race_style == "単複型":
            return f"{support_text}・単複向き"
        if race_style == "穴ワイド型":
            return f"{support_text}・穴注意"
        if (
            pd.notna(ai_position)
            and ai_position <= 3
            and pd.notna(support_count)
            and support_count >= 2
        ):
            return f"{support_text}・軸向き"
        if (
            pd.notna(popularity)
            and popularity >= 5
            and (
                bool(hole_signal)
                or (pd.notna(support_count) and support_count >= 2)
                or (pd.notna(h2h_score) and h2h_score >= 1)
            )
        ):
            return f"{support_text}・穴注意"
        if (
            (pd.notna(support_count) and support_count >= 2)
            or (
                pd.notna(ai_position)
                and ai_position <= 5
                and pd.notna(venue_score)
                and venue_score >= 5
            )
        ):
            return f"{support_text}・相手向き"
        if (
            (pd.notna(support_count) and support_count >= 1)
            or (pd.notna(h2h_score) and h2h_score != 0)
            or str(row.get("会場理由", "")).strip() not in ("", "会場別強調なし")
        ):
            return f"{support_text}・条件注意"
        return f"{support_text}・様子見"

    def combined_reason(row):
        def clean_text(value):
            if pd.isna(value):
                return ""
            text = str(value).strip()
            return "" if text.lower() == "nan" else text

        parts = []
        venue_reason = clean_text(row.get("会場理由", ""))
        bet_reason = clean_text(row.get("勝負理由", ""))
        support = clean_text(row.get("指数裏付け", ""))
        hole_signal = clean_text(row.get("穴サイン", ""))
        matchup = clean_text(row.get("対戦", ""))
        momentum = clean_text(row.get("勢い", ""))
        age_adjustment = pd.to_numeric(row.get("年齢補正"), errors="coerce")
        age = extract_age_from_sex_age(row.get("性齢"))
        weight_change = pd.to_numeric(row.get("_load_weight_change"), errors="coerce")

        if hole_signal:
            parts.extend(part.strip() for part in hole_signal.split("/") if part.strip())
        if support:
            parts.append(f"指数{support}")
        if matchup:
            parts.append(matchup)
        if momentum in ("良化", "上昇", "上昇中"):
            parts.append(f"近走{momentum}")
        if pd.notna(weight_change):
            if weight_change >= 1:
                parts.append(f"斤量+{format_number_for_display(weight_change)}kg")
            elif weight_change <= -1:
                parts.append(f"斤量{format_number_for_display(weight_change)}kg")
        if pd.notna(age_adjustment) and age_adjustment > 0:
            parts.append(f"年齢補正+{format_number_for_display(age_adjustment)}")
        elif age is not None and age >= 11:
            parts.append("11歳以上注意")
        elif age is not None and age >= 9:
            parts.append("高齢注意")
        if bet_reason:
            parts.extend(part.strip() for part in bet_reason.split("/") if part.strip())
        if venue_reason and venue_reason != "会場別強調なし":
            parts.extend(part.strip() for part in venue_reason.split("/") if part.strip())

        unique = []
        for part in parts:
            if part and part not in unique:
                unique.append(part)
        if not unique:
            return "指数裏付け0/3。会場別の強調材料は少ない"
        return " / ".join(unique[:6])

    def ticket_type(row):
        odds = pd.to_numeric(row.get("単勝オッズ"), errors="coerce")
        material_rank = pd.to_numeric(row.get("材料順位"), errors="coerce")
        ai_rank = pd.to_numeric(row.get("AI順位"), errors="coerce")
        support_count = pd.to_numeric(row.get("_指数裏付け数"), errors="coerce")
        support_count = int(support_count) if pd.notna(support_count) else 0
        race_style_value = str(row.get("勝負タイプ", "") or "")
        hole_signal = str(row.get("穴サイン", "") or "").strip()
        style = normalize_running_style(row.get("脚質", ""))
        age = extract_age_from_sex_age(row.get("性齢"))

        if pd.isna(odds):
            return "見送り"
        if age is not None and age >= 11 and support_count < 3:
            return "見送り"
        if odds < 2:
            return "ワイド軸" if support_count >= 2 or (pd.notna(ai_rank) and ai_rank <= 3) else "見送り"
        if (
            pd.notna(material_rank)
            and material_rank <= 2
            and support_count >= 2
            and odds >= 2
        ):
            return "単勝＋ワイド"
        if race_style_value == "単複型" and support_count >= 2 and odds >= 2:
            return "単勝"
        if odds >= 10 and (hole_signal or support_count >= 1):
            return "複勝穴"
        if odds >= 5 and odds <= 30 and pd.notna(ai_rank) and ai_rank <= 6 and support_count >= 1:
            return "ワイド"
        return "見送り"

    def ticket_reason(row):
        kind = row.get("推奨券種", "")
        odds = pd.to_numeric(row.get("単勝オッズ"), errors="coerce")
        support_count = pd.to_numeric(row.get("_指数裏付け数"), errors="coerce")
        support_count = int(support_count) if pd.notna(support_count) else 0
        material_rank = pd.to_numeric(row.get("材料順位"), errors="coerce")
        hole_signal = str(row.get("穴サイン", "") or "").strip()
        style = normalize_running_style(row.get("脚質", ""))
        parts = []
        if pd.notna(material_rank) and material_rank <= 2:
            parts.append("材料上位")
        if support_count >= 2:
            parts.append(f"指数{support_count}/3")
        if pd.notna(odds):
            if odds < 2:
                parts.append("1倍台")
            elif odds >= 10:
                parts.append("配当妙味")
        if hole_signal:
            parts.append("穴材料")
        if style in ("逃", "先"):
            parts.append("前受け")
        elif style in ("差", "追"):
            parts.append("展開待ち")
        if not parts:
            parts.append("強調材料少")
        return f"{kind}：" + " / ".join(parts[:4])

    result["総合評価"] = result.apply(combined_evaluation, axis=1)
    result["総合理由"] = result.apply(combined_reason, axis=1)
    result["推奨券種"] = result.apply(ticket_type, axis=1)
    result["券種理由"] = result.apply(ticket_reason, axis=1)
    result["オッズ"] = result["単勝オッズ"]
    result["距離"] = result["距離指数"]
    result["★最高"] = result["★最高指数"]
    result["評価"] = result["総合理由"]
    result = result.sort_values(["AI順位", "AI点"], ascending=[True, False]).reset_index(drop=True)

    candidate_columns = [
        "材料順位", "馬番", "馬名", "性齢", "馬体重", "人気", "オッズ", "AI順位",
        "推奨券種", "券種理由", "総合評価", "脚質", "間隔", "同馬場実績", "クラス変動", "距離", "コース指数",
        "3走前", "2走前", "前走", "★最高", "評価", "対戦",
    ]
    candidates = result.sort_values("材料順位")[candidate_columns].reset_index(drop=True)
    return result, candidates, venue, profile


def prepare_venue_scenario(df):
    tmp = df.copy()
    if "AI順位" not in tmp.columns or "馬タイプ" not in tmp.columns:
        tmp = add_newspaper_features(tmp)

    tmp["_展開脚質"] = tmp.get(
        "脚質", pd.Series("", index=tmp.index)
    ).map(normalize_running_style)
    tmp["_展開AI順位"] = pd.to_numeric(tmp.get("AI順位"), errors="coerce")
    tmp["_展開AI点"] = pd.to_numeric(tmp.get("AI点"), errors="coerce")
    tmp["_展開コース"] = pd.to_numeric(tmp.get("コース指数"), errors="coerce")
    tmp["_展開人気"] = pd.to_numeric(tmp.get("人気"), errors="coerce")
    tmp["_展開オッズ"] = pd.to_numeric(tmp.get("単勝オッズ"), errors="coerce")
    tmp["_展開材料順位"] = pd.to_numeric(tmp.get("材料順位"), errors="coerce")
    tmp["_4角"] = tmp.get("4角予想", pd.Series("", index=tmp.index)).fillna("").astype(str)
    tmp["_馬番数値"] = pd.to_numeric(tmp.get("馬番"), errors="coerce")
    return tmp


def print_venue_pace_summary(df):
    tmp = prepare_venue_scenario(df)
    styles = tmp["_展開脚質"]
    counts = {
        "逃": int(styles.eq("逃").sum()),
        "先": int(styles.eq("先").sum()),
        "差": int(styles.eq("差").sum()),
        "追": int(styles.eq("追").sum()),
    }
    escape_count = counts["逃"]
    early_count = counts["逃"] + counts["先"]

    if escape_count >= 2 or early_count >= 5:
        pace_text = "速くなりそう / 差し浮上"
        favored_styles = ["差", "追"]
    elif early_count <= 2:
        pace_text = "落ち着きそう / 前残り警戒"
        favored_styles = ["逃", "先"]
    else:
        pace_text = "平均ペース / 好位差し互角"
        favored_styles = ["先", "差"]

    pace_holes = tmp[
        tmp["_展開脚質"].isin(favored_styles)
        & (tmp["_展開AI順位"].le(10) | tmp["_展開AI順位"].isna())
        & (
            tmp["_展開人気"].ge(5)
            | tmp["_展開オッズ"].ge(10)
        )
    ].sort_values(
        ["_展開材料順位", "_展開AI順位", "_展開コース"],
        ascending=[True, True, False],
        na_position="last",
    ).head(2)

    hole_numbers = []
    for value in pace_holes["_馬番数値"].dropna():
        hole_numbers.append(circled_number(int(value)))

    print("")
    print("【展開予想】")
    print(
        f"脚質構成：逃{counts['逃']} 先{counts['先']} "
        f"差{counts['差']} 追{counts['追']}"
    )
    print(f"ペース：{pace_text}")
    print(f"有利脚質：{'・'.join(favored_styles)}")
    print(f"展開穴：{''.join(hole_numbers) if hole_numbers else 'なし'}")


def print_venue_race_scenario(df):
    tmp = prepare_venue_scenario(df)

    def numeric(column):
        if column in tmp.columns:
            return pd.to_numeric(tmp[column], errors="coerce")
        return pd.Series(float("nan"), index=tmp.index, dtype="float64")

    ai = numeric("AI点")
    total = numeric("総合評価点").fillna(numeric("_最終印点")).fillna(ai)
    final_order = pd.to_numeric(tmp.get("_最終印順", pd.Series(99, index=tmp.index)), errors="coerce").fillna(99)
    styles = tmp["_展開脚質"]
    mark = tmp.get("最終印", pd.Series("", index=tmp.index)).fillna("").astype(str)

    def horse_label(row):
        number = pd.to_numeric(row.get("馬番"), errors="coerce")
        no_text = str(int(number)) if pd.notna(number) else str(row.get("馬番", "")).strip()
        return f"{no_text} {str(row.get('馬名', '')).strip()}".strip()

    def names_text(pool, limit=None):
        if pool.empty:
            return "該当馬なし"
        if limit:
            pool = pool.head(limit)
        return "、".join(horse_label(row) for _, row in pool.iterrows())

    def sorted_pool(mask):
        pool = tmp[mask].copy()
        if pool.empty:
            return pool
        pool["_表示印順"] = final_order.reindex(pool.index).fillna(99)
        pool["_表示総合"] = total.reindex(pool.index).fillna(0)
        pool["_表示AI"] = ai.reindex(pool.index).fillna(0)
        return pool.sort_values(["_表示印順", "_表示総合", "_表示AI"], ascending=[True, False, False])

    escape_pool = sorted_pool(styles.eq("逃"))
    leader_pool = sorted_pool(styles.eq("先"))
    closer_pool = sorted_pool(styles.eq("差"))
    trailer_pool = sorted_pool(styles.eq("追"))
    center_pool = sorted_pool(mark.isin(["◎", "○"]))
    partner_pool = sorted_pool(mark.isin(["▲", "△"]))
    star_pool = sorted_pool(mark.eq("☆"))

    escape_count = len(escape_pool)
    early_count = len(escape_pool) + len(leader_pool)
    if escape_count >= 2 or early_count >= 5:
        pace_text = "速くなりそう。差し・追込の浮上も警戒。"
    elif escape_count == 1 and early_count <= 3:
        pace_text = "落ち着きやすい流れ。前残りを警戒。"
    else:
        pace_text = "平均ペース想定。好位勢と差し馬は互角。"

    print("【展開】")
    print(f"逃げ・先行候補：{names_text(pd.concat([escape_pool, leader_pool]), 5)}")
    print(f"ペース想定：{pace_text}")
    if not escape_pool.empty:
        lead = escape_pool.iloc[0]
        lead_score = pd.to_numeric(total.reindex([lead.name]).iloc[0], errors="coerce")
        lead_rank = int(final_order.reindex([lead.name]).iloc[0]) if pd.notna(final_order.reindex([lead.name]).iloc[0]) else 99
        if lead_rank <= 2 or (pd.notna(lead_score) and lead_score >= total.quantile(0.75)):
            print(f"展開のポイント：{horse_label(lead)}が楽に運べれば粘り込みまで。")
        else:
            print(f"展開のポイント：{horse_label(lead)}は展開を作れそうだが、能力的に粘り込みには展開の助けが必要。")
    elif not leader_pool.empty:
        print(f"展開のポイント：{horse_label(leader_pool.iloc[0])}が押し出される形。隊列が落ち着けば先行勢に余地。")
    else:
        print("展開のポイント：明確な逃げ馬が少なく、序盤の位置取りが評価を左右。")

    def reason_text(row, role):
        parts = []
        if role == "center":
            parts.append(f"AI点{format_number_for_display(row.get('AI点'))}")
            parts.append(f"総合評価{format_number_for_display(row.get('総合評価点') or row.get('_最終印点'))}")
        material = str(row.get("評価/検討材料") or build_nar_evaluation_material(row))
        class_shift = str(row.get("クラス変動") or "")
        reason = str(row.get("印理由") or "")
        if class_shift:
            parts.append(class_shift)
        for keyword in ["能力上位", "最高指数", "コース実績", "距離実績", "対戦先着", "対戦◎", "展開向く", "同馬場実績"]:
            if keyword in material or keyword in reason:
                parts.append(keyword)
        if str(row.get("展開印", "")).strip() == "展":
            parts.append("展開向く")
        unique = []
        for part in parts:
            if part and part not in unique:
                unique.append(part)
        return "、".join(unique[:4]) if unique else "条件補正込みで評価"

    print("")
    print("【中心馬】")
    if center_pool.empty:
        print("中心馬：該当馬なし")
    else:
        for _, row in center_pool.head(2).iterrows():
            print(f"{row.get('最終印', '')} {horse_label(row)}：{reason_text(row, 'center')}")

    print("")
    print("【相手候補】")
    if partner_pool.empty:
        print("相手候補：該当馬なし")
    else:
        for _, row in partner_pool.head(3).iterrows():
            print(f"{row.get('最終印', '')} {horse_label(row)}：{reason_text(row, 'partner')}")

    print("")
    print("【穴候補】")
    if star_pool.empty:
        print("穴候補：該当馬なし")
    else:
        row = star_pool.iloc[0]
        material = str(row.get("評価/検討材料") or "")
        peak_text = "ピーク指数型" if ("最高指数" in material or "ピーク" in str(row.get("印理由") or "")) else "妙味候補"
        print(f"☆ {horse_label(row)}：{peak_text}。{reason_text(row, 'hole')}")

def print_venue_profile(venue, profile, has_style_html):
    print("【会場別試験評価】")
    if profile:
        print(f"会場判定: {venue}")
        print(f"初期傾向: {profile['headline']}")
        print(f"検証数: {profile['sample_races']}レース（暫定）")
        if not has_style_html:
            print("脚質データ: なし。指数中心で評価します。")
        else:
            print("脚質データ: あり。指数と組み合わせて評価します。")
    else:
        label = venue or "不明"
        print(f"会場判定: {label}（専用傾向は未登録）")
        print("共通評価: AI順位・コース指数・近3走最高で補助評価します。")
    print("注意: 会場評価はAI点へ加算していません。")


def print_race_styles(candidates):
    print("")
    print("【単複型】")
    singles = candidates[candidates["勝負タイプ"].eq("単複型")]
    if singles.empty:
        print("該当なし")
    else:
        for _, row in singles.iterrows():
            horse_no = pd.to_numeric(row.get("馬番"), errors="coerce")
            no_text = str(int(horse_no)) if pd.notna(horse_no) else str(row.get("馬番", ""))
            odds = format_number_for_display(row.get("オッズ"))
            print(f"{no_text} {row.get('馬名', '')} 単勝{odds}倍 - {row.get('評価', '')}")

    print("")
    print("【穴ワイド型】")
    holes = candidates[candidates["勝負タイプ"].eq("穴ワイド型")]
    if holes.empty:
        print("該当なし")
        return
    for _, row in holes.iterrows():
        horse_no = pd.to_numeric(row.get("馬番"), errors="coerce")
        no_text = str(int(horse_no)) if pd.notna(horse_no) else str(row.get("馬番", ""))
        odds = format_number_for_display(row.get("オッズ"))
        print(f"{no_text} {row.get('馬名', '')} 単勝{odds}倍 - {row.get('評価', '')}")


def print_local_single_and_hole(df):
    print("")
    print("★★★★★★★★★★★★★★★★")
    print("★【券種適性メモ】★")
    print("★★★★★★★★★★★★★★★★")
    if df.empty or "材料順位" not in df.columns:
        print("判定できません。")
        return

    ordered = df.sort_values("材料順位").copy()
    focus = ordered.iloc[0]
    focus_no = pd.to_numeric(focus.get("馬番"), errors="coerce")
    focus_no_text = str(int(focus_no)) if pd.notna(focus_no) else str(focus.get("馬番", "")).strip()
    focus_label = f"{focus_no_text} {focus.get('馬名', '')}"
    focus_odds = pd.to_numeric(focus.get("単勝オッズ"), errors="coerce")
    support = pd.to_numeric(focus.get("_指数裏付け数"), errors="coerce")
    support = int(support) if pd.notna(support) else 0
    ai_rank = pd.to_numeric(focus.get("AI順位"), errors="coerce")
    focus_age = extract_age_from_sex_age(focus.get("性齢"))
    style = normalize_running_style(focus.get("脚質", ""))
    corner = str(focus.get("4角予想", ""))
    rear_dependent = style == "追" or "後方" in corner
    weight_change = pd.to_numeric(focus.get("_load_weight_change"), errors="coerce")
    focus_score = pd.to_numeric(focus.get("_会場評価点"), errors="coerce")
    second_score = (
        pd.to_numeric(ordered.iloc[1].get("_会場評価点"), errors="coerce")
        if len(ordered) >= 2
        else None
    )
    material_gap = (
        focus_score - second_score
        if pd.notna(focus_score) and second_score is not None and pd.notna(second_score)
        else None
    )
    running_styles = ordered.get(
        "脚質", pd.Series("", index=ordered.index)
    ).map(normalize_running_style)
    ordered["_表示オッズ"] = pd.to_numeric(
        ordered.get("単勝オッズ"), errors="coerce"
    )
    odds_on_pool = ordered[ordered["_表示オッズ"].lt(2)].sort_values(
        "_表示オッズ", na_position="last"
    )
    odds_on_favorite = odds_on_pool.iloc[0] if not odds_on_pool.empty else None
    odds_on_no = (
        pd.to_numeric(odds_on_favorite.get("馬番"), errors="coerce")
        if odds_on_favorite is not None
        else pd.NA
    )
    escape_count = int(running_styles.eq("逃").sum())
    close_materials = material_gap is not None and material_gap <= 1.5
    pace_uncertain = escape_count == 0 and style in ("差", "追")
    weight_risk = pd.notna(weight_change) and weight_change >= 2
    no_clear_focus = (
        (close_materials and pace_uncertain)
        or (
            weight_risk
            and pd.notna(ai_rank)
            and ai_rank > 1
            and material_gap is not None
            and material_gap <= 2.5
        )
    )

    focus_available = True
    judgment = "見送り"
    if pd.isna(focus_odds):
        print(f"単勝候補：なし（{focus_label}はオッズ未取得）。")
        focus_available = False
    elif focus_odds < 2:
        print(
            f"単勝候補：なし（{focus_label}は単勝{format_number_for_display(focus_odds)}倍。"
            "1倍台のため相手穴確認）。"
        )
        focus_available = False
    elif no_clear_focus:
        reasons = []
        if close_materials:
            reasons.append("上位の材料差が小さい")
        if pace_uncertain:
            reasons.append("逃げ不在で差し展開が読みづらい")
        if weight_risk:
            reasons.append(f"前走比+{format_number_for_display(weight_change)}kg")
        print(f"単勝候補：なし（混戦。{'・'.join(reasons)}）。")
        focus_available = False
    else:
        if (
            focus_age is not None
            and focus_age >= 11
            and not (support == 3 and pd.notna(ai_rank) and ai_rank <= 2)
        ):
            judgment = "見送り"
        elif rear_dependent and not (support >= 2 and pd.notna(ai_rank) and ai_rank <= 3):
            judgment = "見送り"
        elif focus.get("勝負タイプ") == "単複型" and support >= 2:
            judgment = "中心候補"
        elif focus.get("勝負タイプ") == "単複型" or support >= 2:
            judgment = "相手候補"
        else:
            judgment = "見送り"
        if weight_risk and judgment == "中心候補":
            judgment = "相手候補"
        if judgment == "見送り":
            focus_available = False

        weight_note = (
            f"・斤量+{format_number_for_display(weight_change)}kg"
            if weight_risk
            else ""
        )
        if focus_available:
            print(
                f"単勝候補：{circled_number(int(focus_no)) if pd.notna(focus_no) else focus_no_text}"
                f"{focus.get('馬名', '')} 単勝{format_number_for_display(focus_odds)}倍。"
                f"{judgment}（指数{support}/3・{style or '脚質不明'}{weight_note}）。"
            )
        else:
            print(
                f"単勝候補：なし（{focus_label}は"
                f"{judgment}・指数{support}/3・{style or '脚質不明'}）。"
            )

    anchor_no = focus_no
    anchor_label = (
        circled_number(int(focus_no)) if pd.notna(focus_no) else focus_no_text
    )
    if odds_on_favorite is not None:
        odds_on_name = odds_on_favorite.get("馬名", "")
        odds_on_odds = format_number_for_display(odds_on_favorite.get("単勝オッズ"))
        odds_on_text = (
            circled_number(int(odds_on_no)) if pd.notna(odds_on_no)
            else str(odds_on_favorite.get("馬番", "")).strip()
        )
        print(
            f"1倍台人気：{odds_on_text}{odds_on_name} 単勝{odds_on_odds}倍。"
            "ここは相手穴を優先確認。"
        )
        anchor_no = odds_on_no
        anchor_label = odds_on_text

    hole_pool = ordered[
        pd.to_numeric(ordered.get("馬番"), errors="coerce").ne(anchor_no)
    ].copy()
    hole_pool["_穴オッズ"] = pd.to_numeric(
        hole_pool.get("単勝オッズ"), errors="coerce"
    )
    hole_pool["_穴AI順位"] = pd.to_numeric(
        hole_pool.get("AI順位"), errors="coerce"
    )
    hole_pool["_穴裏付け"] = pd.to_numeric(
        hole_pool.get("_指数裏付け数"), errors="coerce"
    ).fillna(0)
    hole_pool["_穴サイン有"] = (
        hole_pool.get("穴サイン", pd.Series("", index=hole_pool.index))
        .fillna("")
        .astype(str)
        .str.strip()
        .ne("")
    )
    hole_pool["_穴年齢"] = hole_pool.get(
        "性齢", pd.Series("", index=hole_pool.index)
    ).map(extract_age_from_sex_age)
    hole_pool = hole_pool[
        hole_pool["_穴オッズ"].ge(10)
        & hole_pool["_穴AI順位"].le(10)
        & (hole_pool["_穴裏付け"].ge(1) | hole_pool["_穴サイン有"])
        & (hole_pool["_穴年齢"].isna() | hole_pool["_穴年齢"].lt(11))
    ]
    if hole_pool.empty:
        print("穴候補：該当なし。")
        return

    hole_limit = 3 if odds_on_favorite is not None else 1
    hole_numbers = []
    for _, hole in hole_pool.head(hole_limit).iterrows():
        hole_no = pd.to_numeric(hole.get("馬番"), errors="coerce")
        hole_no_text = str(int(hole_no)) if pd.notna(hole_no) else str(hole.get("馬番", "")).strip()
        hole_support = pd.to_numeric(hole.get("_指数裏付け数"), errors="coerce")
        hole_support = int(hole_support) if pd.notna(hole_support) else 0
        hole_mark = circled_number(int(hole_no)) if pd.notna(hole_no) else hole_no_text
        hole_numbers.append(hole_mark)
        print(
            f"穴候補：{hole_mark}{hole.get('馬名', '')} "
            f"単勝{format_number_for_display(hole.get('単勝オッズ'))}倍"
            f"（材料順位{int(hole.get('材料順位'))}位・指数{hole_support}/3）。"
        )
    if pd.notna(anchor_no) and hole_numbers:
        print(f"ワイド検討：{anchor_label} - {''.join(hole_numbers)}")


def build_matchup_table(df):
    if "対戦" not in df.columns:
        return pd.DataFrame(columns=["最終印", "馬番", "馬名", "対戦評価", "対戦"])
    source = df[df["対戦"].fillna("").astype(str).str.strip().ne("")].copy()
    if source.empty:
        return pd.DataFrame(columns=["最終印", "馬番", "馬名", "対戦評価", "対戦"])
    if "対戦評価" not in source.columns:
        source["対戦評価"] = source.get("_h2h_label", pd.Series("", index=source.index)).fillna("")
    if "_対戦順" not in source.columns:
        mark_order = {"◎": 0, "○": 1, "▲": 2, "△": 3, "☆": 4}
        source["_対戦順"] = source.get("最終印", pd.Series("", index=source.index)).map(mark_order).fillna(9)
    sort_cols = [col for col in ["_対戦順", "_h2h_score", "補正AI点", "AI点"] if col in source.columns]
    ascending = [True] + [False] * (len(sort_cols) - 1) if sort_cols else []
    if sort_cols:
        source = source.sort_values(sort_cols, ascending=ascending)
    cols = [col for col in ["最終印", "馬番", "馬名", "対戦評価", "対戦"] if col in source.columns]
    return source[cols].reset_index(drop=True)


def print_matchup_table(df):
    matchup = build_matchup_table(df)
    if matchup.empty:
        return
    print("")
    print("【対戦表】")
    try:
        display(format_result_for_output(matchup))
    except Exception:
        display(matchup)


# ---- Keiba AI Mobile wrapper -------------------------------------------------
# The prediction code above is generated from the research notebook's logic
# cells.  The wrapper below feeds in-memory HTML into that logic and converts
# the resulting notebook-style output into PredictionResult.  It intentionally
# does not read or execute .ipynb files at runtime.

import contextlib as _ka_contextlib
import io as _ka_io
from typing import Any as _KaAny

import pandas as _ka_pd

from .models import PredictionResult as _KaPredictionResult
from .prediction_runtime import (
    DisplayCapture as _KaDisplayCapture,
    build_overall_table as _ka_build_overall_table,
    capture_first_dataframe as _ka_capture_first_dataframe,
    capture_text as _ka_capture_text,
    install_notebook_shims as _ka_install_notebook_shims,
    split_attention_horses as _ka_split_attention_horses,
)


def _run_nar_notebook_body(
    html_files: dict[str, str],
    file_names: dict[str, str],
    capture: _KaDisplayCapture,
    *,
    fetch_past_detail: bool = True,
) -> dict[str, _KaAny]:
    globals()["display"] = capture.display
    globals().update({
            "FETCH_PAST_RACE_DETAIL": fetch_past_detail,
            "PAST_RACE_SLEEP_SEC": 0.35,
            "SHOW_CORNER_SCENARIO": True,
            "SHOW_TARGET_HORSE_AUDIT": False,
            "html_from_pc_file": html_files.get("speed", ""),
            "html_from_style_file": html_files.get("style", ""),
            "html_from_odds_file": "",
            "html_from_newspaper_file": html_files.get("newspaper", ""),
            "html_file_name": file_names.get("speed", ""),
            "style_html_file_name": file_names.get("style", ""),
            "odds_html_file_name": "",
            "newspaper_html_file_name": file_names.get("newspaper", ""),
            "html_from_shutuba_file": html_files.get("shutuba", ""),
            "shutuba_html_file_name": file_names.get("shutuba", ""),
    })
    capture.reset()
    #@title 解析実行
    html_input = globals().get("html_from_pc_file", "").strip()
    if not html_input:
        raise ValueError("先に `HTMLファイルをまとめてアップロード` セルで、タイム指数HTMLを含めてアップロードしてください。")

    session = make_session()
    html = html_input
    source_label = f"PC保存HTML: {globals().get('html_file_name', '')}"
    fetch_past_detail = globals().get("FETCH_PAST_RACE_DETAIL", True)
    past_race_sleep_sec = globals().get("PAST_RACE_SLEEP_SEC", 0.35)
    show_corner_scenario = globals().get("SHOW_CORNER_SCENARIO", True)
    OPTIONAL_ODDS_HTML_UI_ENABLED = False  # オッズHTML UIは一時的に非表示
    style_html_input = globals().get("html_from_style_file", "").strip()
    newspaper_html_input = globals().get("html_from_newspaper_file", "").strip()
    shutuba_html_input = globals().get("html_from_shutuba_file", "").strip()
    odds_html_input = globals().get("html_from_odds_file", "").strip() if OPTIONAL_ODDS_HTML_UI_ENABLED else ""

    uploaded_soup = BeautifulSoup(html, "html.parser")
    uploaded_rows = len(uploaded_soup.select("#Speed_List tbody tr.List"))
    if uploaded_rows == 0:
        uploaded_rows = len(uploaded_soup.select("table.SpeedIndex_Table tbody tr"))

    print(f"取得方法: {source_label}")
    print(f"アップロードHTML内のタイム指数行数: {uploaded_rows}")
    if uploaded_rows <= 3:
        print("注意: アップロードしたHTML自体に3頭分程度しか入っていません。保存前のブラウザ画面で全頭表示されているか確認してください。")

    if OPTIONAL_ODDS_HTML_UI_ENABLED:
        print_optional_odds_html_debug(odds_html_input, globals().get("odds_html_file_name", ""))

    result_df, race_info = parse_nar_speed_table(
        html=html,
        session=session,
        fetch_past_detail=fetch_past_detail,
        sleep_sec=past_race_sleep_sec,
    )
    result_df, style_df = apply_nar_style_features(result_df, style_html_input)
    running_style_info = analyze_running_style(result_df)
    result_df = add_newspaper_features(result_df, running_style_info)
    if newspaper_html_input:
        entry_html_label = "競馬新聞HTML"
        result_df, race_info, entry_html_info = apply_nar_newspaper_html_features(result_df, race_info, newspaper_html_input)
    else:
        entry_html_label = "出馬表HTML"
        result_df, race_info, entry_html_info = apply_shutuba_features(result_df, race_info, shutuba_html_input)
    result_df, venue_candidates, detected_venue, venue_profile = apply_venue_profile(
        result_df, race_info, has_style_html=bool(style_html_input)
    )
    result_df = add_final_marks(result_df, running_style_info)
    result_df = refresh_horse_pace_comments(result_df, running_style_info)
    result_df = apply_watch_marks(result_df, race_type="nar")
    result_df = remove_betting_output_columns(result_df)
    if "_最終印点" in result_df.columns:
        result_df["総合評価点"] = pd.to_numeric(result_df["_最終印点"], errors="coerce").round(1)
    result_df = add_purchase_value_columns(result_df)
    result_df = add_audit_evaluation_columns(result_df, race_type="nar")
    result_df = prepare_nar_display_columns(result_df)


    display_cols = ["表示印", "展開印", "馬番", "馬名", "馬年齢", "斤量", "騎手", "オッズ", "脚質", "レース間隔", "AI点", "総合評価", "市場反映勝率", "単勝期待値", "クラス変動", "クラス根拠", "馬場実績", "距離指数", "コース指数", "3走前", "2走前", "前走", "平均指数", "過去1年最高指数", "★最高指数", "★該当走", "★条件", "★最高指数の取得元", "評価/検討材料", "能力評価値", "能力帯", "能力差", "レース難易度", "レース難易度理由", "表示コメント", "raw_score", "ability_display_score", "normalized_ai_score", "ai_rank", "final_mark_score", "market_score", "star_max_index", "star_max_race", "star_max_venue", "star_max_distance", "star_max_surface", "star_max_turn", "star_match_level", "star_max_source", "axis_confidence", "axis_confidence_reason", "ability_band", "ability_gap_level", "race_difficulty", "race_difficulty_reason", "display_comment", "old_final_mark", "old_watch_mark", "hole_candidate", "watch_horse"]
    # Keep result-free parser evidence available to the independent
    # ability/price comparison layer.  These columns are not shown by the
    # legacy table and never change Ver3 scoring.
    display_cols.extend([
        "_current_class_rank", "_current_class_label", "_previous_class_rank",
        "_previous_class_label", "_best_past_class_rank", "_best_past_class_label",
        "_past_class_labels", "_past_runs", "_days_since_last",
        "馬体重", "_body_weight", "_body_weight_change",
        "_current_load_weight", "_previous_load_weight", "_load_weight_change",
        "_ver3_ability_core", "_market_non_ability_adjustment",
        "_current_jockey", "_previous_jockey", "_jockey_changed",
        "厩舎コメント", "新聞コメント", "対戦", "対戦評価", "対戦材料",
    ])
    print(f"レース: {race_info.get('race_name', '')} / {race_info.get('race_data', '')}")
    print(f"抽出頭数: {len(result_df)}")
    print_venue_profile(detected_venue, venue_profile, bool(style_html_input))
    print_venue_pace_summary(result_df)
    if newspaper_html_input:
        print(
            f"競馬新聞HTML: 反映 / 今日の馬場: {race_info.get('going') or '未取得'}"
            f" / 馬体重反映: {entry_html_info.get('body_count', 0)}"
            f" / 前走斤量・騎手反映: {entry_html_info.get('previous_detail_count', 0)}"
        )
    elif shutuba_html_input:
        print(f"{entry_html_label}: 反映 / 今日の馬場: {race_info.get('going') or '未取得'} / 馬体重反映: {entry_html_info.get('body_count', 0)}")
    else:
        print("競馬新聞HTML: 未アップロード")
    if style_html_input:
        style_count = int(result_df["脚質"].astype(str).ne("").sum())
        print(f"脚質HTML内の抽出頭数: {len(style_df)} / 表へ反映: {style_count}")
    else:
        print("脚質HTML: 未アップロード")
    display_cols = [column for column in display_cols if column in result_df.columns]
    print("")
    print("【レース全体表】")
    try:
        display(result_display_styler(result_df[display_cols]))
    except Exception:
        display(format_result_for_output(result_df[display_cols]))
    print("")
    print_ver30_all_horse_rating(result_df, race_type="nar")
    print("")
    print_ver30_attention_horses(result_df, race_type="nar")
    print("")
    ai_confidence_summary = build_ai_confidence_summary(result_df, race_info, detected_venue, venue_profile, race_type="nar")
    print_ver30_ai_race_review(result_df, race_info, running_style_info, ai_confidence_summary, race_type="nar")
    print("")
    print_ver30_betting_structure(result_df, ai_confidence_summary, race_type="nar")
    if globals().get("SHOW_TARGET_HORSE_AUDIT", True):
        print("")
        print_target_horse_adjustment_audit(result_df, horse_no=globals().get("TARGET_HORSE_AUDIT_NO", 12), horse_name_keyword=globals().get("TARGET_HORSE_AUDIT_NAME", "ヤングオーオー"))

    return {
        "result_df": locals().get("result_df"),
        "race_info": locals().get("race_info"),
        "running_style_info": locals().get("running_style_info"),
        "ai_confidence_summary": locals().get("ai_confidence_summary"),
        "display_cols": locals().get("display_cols", []),
    }


def _ka_debug_text(value) -> str:
    if value is None:
        return ""
    try:
        if _ka_pd.isna(value):
            return ""
    except (TypeError, ValueError):
        pass
    return str(value).strip()


def _ka_row_text(row, *keys: str) -> str:
    for key in keys:
        try:
            value = row.get(key)
        except Exception:
            continue
        text = _ka_debug_text(value)
        if text:
            return text
    return ""


def _ka_build_nar_previous_jockey_prediction_trace(result_df, horse_evaluation) -> list[dict[str, str]]:
    if not isinstance(result_df, _ka_pd.DataFrame) or result_df.empty:
        return []

    evaluation_by_number: dict[str, dict[str, str]] = {}
    if isinstance(horse_evaluation, _ka_pd.DataFrame) and not horse_evaluation.empty:
        for _, evaluation_row in horse_evaluation.iterrows():
            number = _ka_row_text(evaluation_row, "馬番", "馬")
            if not number:
                continue
            evaluation_by_number[number] = {
                "app_display_previous_jockey": _ka_row_text(evaluation_row, "騎手詳細"),
                "horse_evaluation_jockey": _ka_row_text(evaluation_row, "騎手"),
            }

    rows: list[dict[str, str]] = []
    for _, row in result_df.iterrows():
        number = _ka_row_text(row, "馬番", "馬")
        if not number:
            continue
        evaluation_info = evaluation_by_number.get(number, {})
        rows.append(
            {
                "horse_number": number,
                "horse_name": _ka_row_text(row, "馬名"),
                "prediction_current_jockey": _ka_row_text(row, "_display_current_jockey", "_current_jockey", "騎手"),
                "prediction_previous_jockey": _ka_row_text(row, "_display_previous_jockey", "_previous_jockey"),
                "prediction_jockey_changed": _ka_row_text(row, "_display_jockey_changed", "_jockey_changed"),
                "app_display_previous_jockey": evaluation_info.get("app_display_previous_jockey", ""),
                "horse_evaluation_jockey": evaluation_info.get("horse_evaluation_jockey", ""),
            }
        )
    return rows


def _ka_build_nar_star_table_trace(table, stage: str) -> list[dict[str, str]]:
    if not isinstance(table, _ka_pd.DataFrame) or table.empty:
        return []

    rows: list[dict[str, str]] = []
    for _, row in table.iterrows():
        year_max_index = _ka_row_text(row, "year_max_index") or _ka_row_text(row, "_year_max_index")
        if not year_max_index and len(row) > 23:
            year_max_index = row.iloc[23]
        star_max_index = _ka_row_text(row, "star_max_index") or _ka_row_text(row, "_star_high")
        if not star_max_index and len(row) > 24:
            star_max_index = row.iloc[24]
        rows.append(
            star_trace_row(
                horse_no=row.iloc[2] if len(row) > 2 else "",
                horse_name=row.iloc[3] if len(row) > 3 else "",
                year_max_index=year_max_index,
                star_max_index=star_max_index,
                star_source=_ka_row_text(row, "star_max_source") or _ka_row_text(row, "_star_high_source"),
                stage_label=stage,
            )
        )
    return rows


def predict_nar_from_html(
    html_files: dict[str, str],
    file_names: dict[str, str] | None = None,
    *,
    fetch_past_detail: bool = True,
) -> _KaPredictionResult:
    file_names = file_names or {}
    capture = _KaDisplayCapture()
    _ka_install_notebook_shims(capture)

    raw_buffer = _ka_io.StringIO()
    with _ka_contextlib.redirect_stdout(raw_buffer):
        state = _run_nar_notebook_body(
            html_files,
            file_names,
            capture,
            fetch_past_detail=fetch_past_detail,
        )

    result_df = state.get("result_df")
    race_info = dict(state.get("race_info") or {})
    race_name = str(race_info.get("race_name") or "")
    globals()["display"] = capture.display

    overall_table = _ka_build_overall_table(result_df, state.get("display_cols"))
    log_star_trace("08 PredictionResult creation", _ka_build_nar_star_table_trace(overall_table, "overall_table"))
    horse_evaluation = _ka_capture_first_dataframe(
        capture,
        lambda: print_ver30_all_horse_rating(result_df, race_type="nar"),
    )
    attention_text = _ka_capture_text(
        capture,
        lambda: print_ver30_attention_horses(result_df, race_type="nar"),
    )
    review_text = _ka_capture_text(
        capture,
        lambda: print_ver30_ai_race_review(
            result_df,
            race_info,
            state.get("running_style_info"),
            state.get("ai_confidence_summary"),
            race_type="nar",
        ),
    )
    betting_text = _ka_capture_text(
        capture,
        lambda: print_ver30_betting_structure(
            result_df,
            state.get("ai_confidence_summary"),
            race_type="nar",
        ),
    )
    debug_info = {
        "condition_fit_sources": extract_condition_fit_sources(result_df),
        "nar_star_trace": get_star_trace(),
        "nar_previous_jockey_trace": _ka_build_nar_previous_jockey_prediction_trace(
            result_df,
            horse_evaluation,
        )
    }

    return _KaPredictionResult(
        race_mode="nar",
        race_name=race_name,
        race_info=race_info,
        overall_table=overall_table,
        horse_evaluation=horse_evaluation,
        attention_horses=_ka_split_attention_horses(attention_text),
        ai_race_review=review_text.strip(),
        betting_structure=betting_text.strip(),
        source_files=dict(file_names),
        status="ok",
        message="PredictionResult generated by Python module.",
        raw_output=raw_buffer.getvalue(),
        debug_info=debug_info,
    )

# -*- coding: utf-8 -*-
from __future__ import annotations

#@title 解析ロジック
import math
import re
import time
from datetime import date
from functools import lru_cache
from io import StringIO
from urllib.parse import urljoin

import pandas as pd
import requests
from bs4 import BeautifulSoup

from .audit_features import add_audit_evaluation_columns
from .condition_fit import extract_condition_fit_sources
from .star_index import build_star_max_result, star_match_level
from .ver3_ability import calculate_ver3_ability_core


USER_AGENT = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 17_0 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) Version/17.0 Mobile/15E148 Safari/604.1"
)

# JRAのハンデ戦では斤量差を見ますが、指数上位をひっくり返し過ぎない補助量に留めます。
LOAD_WEIGHT_INDEX_PER_KG = 0.5
RELATIVE_WEIGHT_INDEX_PER_KG = 0.2
MAX_TOTAL_WEIGHT_ADJUSTMENT = 1.5


def make_session(cookie_header=""):
    session = requests.Session()
    session.headers.update({
        "User-Agent": USER_AGENT,
        "Accept-Language": "ja,en-US;q=0.9,en;q=0.8",
    })
    if cookie_header.strip():
        session.headers.update({"Cookie": cookie_header.strip()})
    return session


def decode_netkeiba_response(response):
    content = response.content
    for enc in ("EUC-JP", "cp932", "utf-8"):
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
    for hidden in clone.select(".Sort_Function_Data_Hidden, script, style"):
        hidden.decompose()
    return text_of(clone)


def parse_int_from_text(text):
    match = re.search(r"-?\d+", text or "")
    return int(match.group(0)) if match else None


def extract_age_from_sex_age(value):
    match = re.search(r"(\d+)", str(value or ""))
    return int(match.group(1)) if match else None


def parse_going_from_text(text):
    text = norm_text(text)
    patterns = [
        r"馬場\s*[:：]?\s*(不良|稍重|重|良)",
        r"(?:芝|ダート|ダ)\s*[:：]?\s*(不良|稍重|重|良)",
    ]
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            return match.group(1)
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
    if days <= 0:
        return "-"
    if days >= 60:
        return "休み明け"
    weeks = max(0, days // 7)
    return f"中{weeks}週"


JRA_TRACK_CODES = {
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

JRA_TRACK_NAMES = list(JRA_TRACK_CODES.values())

def parse_race_id_from_text(text):
    match = re.search(r"race_id=(\d{12})", text or "") or re.search(r"/race/(\d{12})", text or "") or re.search(r"\b(\d{12})\b", text or "")
    return match.group(1) if match else ""


def parse_jra_track_from_race_id(text):
    race_id = parse_race_id_from_text(text)
    if not race_id or len(race_id) < 6:
        return ""
    return JRA_TRACK_CODES.get(race_id[4:6], "")


def parse_jra_racecourse(*texts):
    combined = " ".join(norm_text(str(text)) for text in texts if text)
    track_from_id = parse_jra_track_from_race_id(combined)
    if track_from_id:
        return track_from_id
    for name in JRA_TRACK_NAMES:
        if name in combined:
            return name
    return ""


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
    r"リステッド|Listed|オープン|OPEN|OP|クラス|条件|A級|B級|C級|A1|A2|A3|B1|B2|B3|C1|C2|C3",
    flags=re.IGNORECASE,
)

GRADE_TYPE_RULES = [
    (r"(?:Icon_)?GradeType[_\s-]*1(?!\d)|grade_type_?1(?!\d)", 90, "G1"),
    (r"(?:Icon_)?GradeType[_\s-]*2(?!\d)|grade_type_?2(?!\d)", 80, "G2"),
    (r"(?:Icon_)?GradeType[_\s-]*3(?!\d)|grade_type_?3(?!\d)", 70, "G3"),
]
def race_class_info(text):
    text = norm_text(text).translate(str.maketrans("０１２３４５６７８９", "0123456789"))
    # Grade icon classes can coexist with Open/OP text; treat icons as stronger signals.
    for pattern, rank, label in GRADE_TYPE_RULES:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return rank, label
    for pattern, rank, label in CLASS_RULES:
        if re.search(pattern, text, flags=re.IGNORECASE):
            return rank, label
    return None, ""


def extract_class_text_candidates(soup):
    if soup is None:
        return []
    candidates = []

    def add_text(value):
        text = norm_text(value)
        if text:
            candidates.append(text)

    def add_attrs(node):
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

    def add_node(node, include_attrs=True):
        if node is None:
            return
        add_text(text_of(node))
        if include_attrs:
            add_attrs(node)
            for child in node.select("[class*='Grade'], [class*='grade'], img[alt], img[src], [title], [aria-label]"):
                add_attrs(child)

    # Keep class detection tied to the race header.  Reading every class/src on
    # the page can misread unrelated GradeType icons as the past race grade.
    selectors = [
        ".RaceName",
        ".RaceData01",
        ".RaceData02",
        ".data_intro h1",
        ".data_intro .racedata",
        ".data_intro .smalltxt",
        "h1",
    ]
    for selector in selectors:
        for node in soup.select(selector):
            add_node(node)

    header = soup.select_one(".RaceName, .RaceHead, .RaceHeader, .data_intro")
    if header is not None:
        for node in header.select("[class*='Grade'], [class*='grade'], img[alt], img[src], [title], [aria-label]"):
            add_attrs(node)

    unique = []
    for text in candidates:
        text = norm_text(text)
        if text and text not in unique:
            unique.append(text)
    return unique
def race_class_info_from_soup(soup, *fallback_texts):
    candidates = extract_class_text_candidates(soup)
    candidates += [norm_text(str(text)) for text in fallback_texts if text]
    major_labels = {"G1", "G2", "G3", "Jpn1", "Jpn2", "Jpn3", "重賞", "準重賞", "L"}
    for text in candidates:
        rank, label = race_class_info(text)
        if rank is not None and label in major_labels:
            return rank, label
    for text in candidates:
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



def extract_course_info_from_race_html(past_html):
    soup = BeautifulSoup(past_html, "html.parser")
    candidates = [
        soup.select_one(".data_intro .racedata p span"),
        soup.select_one(".data_intro .racedata span"),
        soup.select_one(".race_data .racedata span"),
        soup.select_one(".RaceData01"),
        soup.select_one(".RaceData"),
    ]
    for candidate in candidates:
        detail_text = text_of(candidate)
        info = parse_course_text(detail_text)
        if info.get("label"):
            return info

    # Fallback for db.netkeiba pages where the course text is present but
    # class names differ from the standard JRA/NAR race detail templates.
    plain = text_of(soup)
    match = re.search(r"(芝|ダ|障)[^。\n\r]{0,30}?(\d{3,4})\s*m", plain)
    if match:
        start = max(0, match.start() - 12)
        end = min(len(plain), match.end() + 20)
        info = parse_course_text(plain[start:end])
        if info.get("label"):
            return info
    return {}


def parse_current_race_info(html):
    soup = BeautifulSoup(html, "html.parser")
    race_name = text_of(soup.select_one(".RaceName"))
    race_data = text_of(soup.select_one(".RaceData01"))
    race_data2 = text_of(soup.select_one(".RaceData02"))
    title_text = text_of(soup.title)
    info = parse_course_text(race_data)
    class_rank, class_label = race_class_info_from_soup(soup, race_name, race_data, race_data2, title_text)
    info.update({
        "race_name": race_name,
        "race_data": race_data,
        "race_data2": race_data2,
        "racecourse": parse_jra_racecourse(html, race_name, race_data, race_data2, title_text),
        "race_date": parse_race_date_from_text(" ".join([race_name, race_data, race_data2, title_text, text_of(soup.select_one(".RaceList_NameBox"))])),
        "class_rank": class_rank,
        "class_label": class_label,
    })
    return info


def parse_index_cell(cell):
    raw = visible_text(cell)
    value = parse_int_from_text(raw.replace("*", ""))
    link = ""
    if cell is not None:
        a = cell.find("a", href=True)
        if a:
            link = urljoin("https://db.netkeiba.com", a["href"])
    info = {"value": value, "raw": raw, "url": link, "race_date": None}
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
    return f"{format_number_for_display(number)}%" if number is not None else ""


def compact_name(text):
    return re.sub(r"\s+", "", text or "")


def extract_jockey_id(link):
    href = link.get("href", "") if link else ""
    match = re.search(r"/jockey/(?:result/recent/)?(\d+)", href)
    return match.group(1) if match else ""


def extract_past_race_results(past_soup):
    table = past_soup.select_one("table.race_table_01, table.ResultsByRaceDetail")
    if table is None:
        return {}

    header_cells = table.select_one("tr").find_all(["td", "th"], recursive=False) if table.select_one("tr") else []
    headers = [visible_text(cell) for cell in header_cells]

    def header_index(label):
        return next(
            (i for i, header in enumerate(headers) if label in re.sub(r"\s+", "", header)),
            None,
        )

    horse_no_index = header_index("馬番")
    horse_name_index = header_index("馬名")
    load_weight_index = header_index("斤量")
    jockey_index = header_index("騎手")

    results = {}
    for row in table.select("tr")[1:]:
        cells = row.find_all(["td", "th"], recursive=False)
        if len(cells) < 4:
            continue
        position = parse_int_from_text(visible_text(cells[0]))
        horse_no_cell = cells[horse_no_index] if horse_no_index is not None and len(cells) > horse_no_index else (cells[2] if len(cells) > 2 else None)
        horse_name_cell = cells[horse_name_index] if horse_name_index is not None and len(cells) > horse_name_index else (cells[3] if len(cells) > 3 else None)
        horse_no = parse_int_from_text(visible_text(horse_no_cell))
        horse_name = visible_text(horse_name_cell)
        if load_weight_index is not None and len(cells) > load_weight_index:
            load_weight = parse_float_from_text(visible_text(cells[load_weight_index]))
        else:
            load_weight = parse_float_from_text(visible_text(cells[5])) if len(cells) > 5 else None
        jockey_cell = cells[jockey_index] if jockey_index is not None and len(cells) > jockey_index else (cells[6] if len(cells) > 6 else None)
        jockey_link = jockey_cell.find("a", href=True) if jockey_cell else None
        jockey = text_of(jockey_link) if jockey_link else visible_text(jockey_cell)
        jockey_id = extract_jockey_id(jockey_link)
        if not horse_name:
            continue
        results[compact_name(horse_name)] = {
            "position": position,
            "horse_no": horse_no,
            "horse_name": horse_name,
            "load_weight": load_weight,
            "jockey": jockey,
            "jockey_id": jockey_id,
        }
    return results


def is_same_condition(current, past):
    return star_match_level(current, past) != "none"


def is_similar_condition(current, past):
    return bool(
        current.get("racecourse")
        and past.get("racecourse")
        and current.get("racecourse") != past.get("racecourse")
        and current.get("surface")
        and current.get("distance")
        and current.get("direction")
        and current.get("surface") == past.get("surface")
        and current.get("distance") == past.get("distance")
        and current.get("direction") == past.get("direction")
    )


def is_same_distance(current, past):
    return bool(
        current.get("surface")
        and current.get("distance")
        and current.get("surface") == past.get("surface")
        and current.get("distance") == past.get("distance")
    )


def format_prev_run(index_value, past_info, same_condition):
    if index_value is None:
        return ""
    label = past_info.get("label") or ""
    racecourse = past_info.get("racecourse") or ""
    display_label = f"{racecourse}{label}" if racecourse or label else ""
    star = "★" if same_condition else ""
    return f"{index_value}/{display_label}{star}" if display_label else f"{index_value}{star}"


def get_race_id_from_url(url):
    match = re.search(r"race_id=(\d+)", url or "")
    if match:
        return match.group(1)
    match = re.search(r"/race/(\d+)", url or "")
    return match.group(1) if match else "race"


def parse_speed_table(html, race_url, session, fetch_past_detail=True, sleep_sec=0.35):
    current = parse_current_race_info(html)
    soup = BeautifulSoup(html, "html.parser")
    table = soup.select_one("#Speed_List table") or soup.select_one("table.SpeedIndex_Table")
    if table is None:
        raise ValueError("タイム指数テーブルが見つかりません。ログイン済みページのHTMLか、正しいタイム指数URLを確認してください。")

    rows = table.select("tbody tr.List") or table.select("tbody tr")

    @lru_cache(maxsize=256)
    def fetch_past_info(url):
        if not fetch_past_detail or not url:
            return {}
        racecourse_from_url = parse_jra_track_from_race_id(url)
        try:
            time.sleep(float(sleep_sec or 0))
            past_html = fetch_html(url, session=session)
            past_soup = BeautifulSoup(past_html, "html.parser")
            info = extract_course_info_from_race_html(past_html)
            past_race_name = (
                text_of(past_soup.select_one(".RaceName"))
                or text_of(past_soup.select_one(".data_intro h1"))
                or text_of(past_soup.select_one("h1"))
            )
            class_rank, class_label = race_class_info_from_soup(
                past_soup,
                past_race_name,
                text_of(past_soup.select_one(".data_intro .racedata")),
                text_of(past_soup.select_one(".RaceData01")),
                text_of(past_soup.select_one(".RaceData02")),
                text_of(past_soup.title),
            )
            info["race_name"] = past_race_name
            info["class_rank"] = class_rank
            info["class_label"] = class_label
            info["racecourse"] = racecourse_from_url or parse_jra_racecourse(url, text_of(past_soup))
            info["race_date"] = parse_race_date_from_text(text_of(past_soup))
            info["results"] = extract_past_race_results(past_soup)
            return info if info.get("label") else {
                "label": "",
                "racecourse": racecourse_from_url,
                "race_date": info.get("race_date"),
                "race_name": past_race_name,
                "class_rank": class_rank,
                "class_label": class_label,
                "results": info.get("results", {}),
                "error": "course text not found",
            }
        except Exception as exc:
            return {"label": "", "racecourse": racecourse_from_url, "error": str(exc)}

    records = []
    for row in rows:
        umaban = visible_text(row.select_one(".sk__umaban"))
        waku = visible_text(row.select_one("td.Waku"))
        horse_cell = row.select_one(".sk__horse_name") or row.select_one(".Horse_Name")
        horse_link = horse_cell.find("a") if horse_cell else None
        horse_name = text_of(horse_link) if horse_link else visible_text(horse_cell)
        if not umaban or not horse_name:
            continue

        sex_age = visible_text(row.select_one(".Txt_C"))
        load_weight = visible_text(row.select_one(".sk__load_weight"))
        jockey_cell = row.select_one(".Jockey")
        jockey_link = jockey_cell.find("a", href=True) if jockey_cell else None
        jockey = text_of(jockey_link) if jockey_link else visible_text(jockey_cell)
        current_jockey_id = extract_jockey_id(jockey_link)
        odds = parse_float_from_text(visible_text(row.select_one(".sk__odds") or row.select_one(".Odds")))
        popularity = parse_int_from_text(visible_text(row.select_one(".sk__ninki") or row.select_one(".Ninki")))
        max_index = parse_index_cell(row.select_one(".sk__max_index") or row.select_one(".MaxIndex") or row.select_one(".Speed_List03"))["value"]
        distance_index = parse_index_cell(row.select_one(".sk__max_distance_index"))["value"]
        course_index = parse_index_cell(row.select_one(".sk__max_course_index"))["value"]

        prev_defs = [("3走前", ".sk__index3"), ("2走前", ".sk__index2"), ("前走", ".sk__index1")]
        prev_values = []
        prev_display = {}
        star_count = 0
        star_values = []
        star_candidate_runs = []
        same_condition_flags = []
        same_distance_values = []
        similar_condition_values = []
        same_going_values = []
        missing_past_labels = 0
        past_fetch_errors = []
        days_since_last = None
        previous_load_weight = None
        previous_jockey = ""
        previous_jockey_id = ""
        previous_class_rank = None
        previous_class_label = ""
        past_class_ranks = []
        past_class_labels = []
        past_runs = []

        for label, selector in prev_defs:
            cell_info = parse_index_cell(row.select_one(selector))
            past_info = fetch_past_info(cell_info["url"])
            past_condition = dict(past_info)
            for key in ("racecourse", "surface", "distance", "direction", "label"):
                if not past_condition.get(key) and cell_info.get(key):
                    past_condition[key] = cell_info.get(key)
            if not past_condition.get("label"):
                past_condition["label"] = cell_info.get("label") or ""
            result_entry = (past_info.get("results") or {}).get(compact_name(horse_name), {})
            if result_entry or cell_info["url"]:
                past_runs.append({
                    "race_id": parse_race_id_from_text(cell_info["url"]),
                    "url": cell_info["url"],
                    "race_date": past_info.get("race_date"),
                    "race_name": past_info.get("race_name", ""),
                    "course_label": past_condition.get("label", ""),
                    "racecourse": past_condition.get("racecourse", ""),
                    "surface": past_condition.get("surface", ""),
                    "distance": past_condition.get("distance"),
                    "direction": past_condition.get("direction", ""),
                    "position": result_entry.get("position"),
                    "label": label,
                    "value": cell_info["value"],
                })
            if past_info.get("class_rank") is not None:
                past_class_ranks.append(past_info.get("class_rank"))
                past_class_labels.append(past_info.get("class_label", ""))
            if cell_info["value"] is not None and not past_info.get("label"):
                missing_past_labels += 1
                if past_info.get("error"):
                    past_fetch_errors.append(f"{label}: {past_info.get('error')}")
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
            prev_display[label] = format_prev_run(cell_info["value"], past_condition, same_cond)
            star_candidate_runs.append({
                "label": label,
                "value": cell_info.get("value"),
                "racecourse": past_condition.get("racecourse", ""),
                "surface": past_condition.get("surface", ""),
                "distance": past_condition.get("distance"),
                "direction": past_condition.get("direction", ""),
            })
            if label == "前走":
                result_entry = (past_info.get("results") or {}).get(compact_name(horse_name), {})
                previous_load_weight = result_entry.get("load_weight")
                previous_jockey = result_entry.get("jockey", "")
                previous_jockey_id = result_entry.get("jockey_id", "")
                days_since_last = days_between(current.get("race_date"), past_info.get("race_date"))
                previous_class_rank = past_info.get("class_rank")
                previous_class_label = past_info.get("class_label", "")

        valid_prev = [v for v in prev_values if v is not None]
        avg3 = round(sum(valid_prev) / len(valid_prev), 1) if valid_prev else None
        trend = None
        if prev_values[0] is not None and prev_values[-1] is not None:
            trend = prev_values[-1] - prev_values[0]
        current_load_weight = parse_float_from_text(load_weight)
        load_weight_change = current_load_weight - previous_load_weight if current_load_weight is not None and previous_load_weight is not None else None
        load_weight_display = format_load_weight_with_change(load_weight, previous_load_weight)
        if current_jockey_id and previous_jockey_id:
            jockey_changed = current_jockey_id != previous_jockey_id
        else:
            jockey_changed = bool(jockey and previous_jockey and compact_name(jockey) != compact_name(previous_jockey))
        jockey_display = f"{jockey}(替)" if jockey_changed else jockey

        star_result = build_star_max_result(current, star_candidate_runs)
        star_high_value = star_result.value
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
            "距離指数": distance_index,
            "コース指数": course_index,
            "3走前": prev_display["3走前"],
            "2走前": prev_display["2走前"],
            "前走": prev_display["前走"],
            "3走平均": avg3,
            "year_max_index": max_index,
            "_year_max_index": max_index,
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
            "_previous_load_weight": previous_load_weight,
            "_current_load_weight": current_load_weight,
            "_load_weight_change": load_weight_change,
            "_load_weight_adjustment": load_weight_index_adjustment(load_weight_change),
            "_current_jockey": jockey,
            "_current_jockey_id": current_jockey_id,
            "_previous_jockey": previous_jockey,
            "_previous_jockey_id": previous_jockey_id,
            "_jockey_changed": jockey_changed,
            "_race_distance": current.get("distance"),
            "_days_since_last": days_since_last,
            "_is_layoff": bool(days_since_last is not None and days_since_last >= 60),
            "_past_runs": past_runs,
            "_missing_past_labels": missing_past_labels,
            "_past_fetch_errors": " / ".join(past_fetch_errors[:3]),
        })

    df = pd.DataFrame(records)
    if df.empty:
        raise ValueError("馬データを抽出できませんでした。ログイン済みページのHTMLを貼り付けているか確認してください。")

    df = add_head_to_head_features(df)
    df = add_condition_context_features(df, current)
    df = add_scores_and_comments(df)
    return df, current


def _h2h_horse_no_text(value):
    number = parse_int_from_text(str(value or ""))
    return str(number) if number is not None else str(value or "").strip()


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

                loser_no = _h2h_horse_no_text(loser["horse_no"])
                winner_no = _h2h_horse_no_text(winner["horse_no"])
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
        h2h_labels.append(label)
        h2h_latest.append(latest_phrase)
        display_values.append(latest_phrase)

    df["_h2h_wins"] = pd.Series(wins)
    df["_h2h_losses"] = pd.Series(losses)
    df["_h2h_score"] = df["_h2h_wins"] - df["_h2h_losses"]
    df["_h2h_label"] = h2h_labels
    df["_h2h_latest"] = h2h_latest
    df["対戦"] = display_values
    return df


def parse_jra_style_table(html):
    if not html:
        return pd.DataFrame()

    soup = BeautifulSoup(html, "html.parser")
    tables = soup.select("table.Data01_Table") or soup.select("table")
    table = next((t for t in tables if t.select_one(".DataTitle_Cell") and t.select("tr.HorseList")), None)
    if table is None:
        return pd.DataFrame()

    headers = [visible_text(th) for th in table.select("thead th")]

    def header_index(name):
        for idx, header in enumerate(headers):
            if header == name:
                return idx
        return None

    win_idx = header_index("勝率")
    rentai_idx = header_index("連対率")
    show_idx = header_index("複勝率")
    single_return_idx = header_index("単勝回収率")
    show_return_idx = header_index("複勝回収率")

    def cell_text(cells, index, fallback=None):
        target = index if index is not None else fallback
        return visible_text(cells[target]) if target is not None and len(cells) > target else ""

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
        odds = parse_float_from_text(visible_text(row.select_one('[id^="odds-1_"]')))
        popularity = parse_int_from_text(visible_text(row.select_one('[id^="ninki-1_"]')))
        records.append({
            "馬番": horse_no,
            "馬名": horse_name,
            "脚質": running_style,
            "脚質勝率": cell_text(cells, win_idx, 8),
            "脚質連対率": cell_text(cells, rentai_idx, 9),
            "脚質複勝率": cell_text(cells, show_idx, 10),
            "脚質単回収": cell_text(cells, single_return_idx, 11),
            "脚質複回収": cell_text(cells, show_return_idx, 12),
            "_脚質HTML単勝オッズ": odds,
            "_脚質HTML人気": popularity,
        })

    style_df = pd.DataFrame(records)
    if style_df.empty:
        return style_df
    return style_df.drop_duplicates("馬番")


def flatten_table_columns(df):
    result = df.copy()
    columns = []
    for col in result.columns:
        if isinstance(col, tuple):
            parts = [str(part) for part in col if str(part) and str(part) != "nan"]
            columns.append(" ".join(parts))
        else:
            columns.append(str(col))
    result.columns = columns
    return result


def find_column_by_keywords(df, *keywords):
    for column in df.columns:
        text = str(column)
        if all(keyword in text for keyword in keywords):
            return column
    return None


def clean_cell_text(value):
    if value is None or pd.isna(value):
        return ""
    text = norm_text(str(value))
    return "" if text.lower() == "nan" else text


def parse_jra_newspaper_html(newspaper_html):
    columns = [
        "馬番", "_新聞馬名", "_新聞単勝オッズ", "_新聞人気", "_新聞脚質",
        "_新聞レース間隔", "_新聞斤量", "_新聞騎手", "_新聞騎手変更", "新聞コメント",
        "_新聞馬体重", "_新聞馬体重値", "_新聞馬体重増減", "_新聞過去走",
        "_新聞今回クラス順位", "_新聞今回クラス", "_新聞前走クラス順位", "_新聞前走クラス",
        "_新聞最高クラス順位", "_新聞最高クラス", "_新聞過去クラス", "_新聞クラス変動",
        "_新聞前走間隔日数",
        "調教コメント", "調教評価", "_調教評価記号", "_調教評価文",
        "推定前半3F", "推定後半3F",
    ]
    if not newspaper_html:
        return pd.DataFrame(columns=columns)
    try:
        tables = [flatten_table_columns(df) for df in pd.read_html(StringIO(newspaper_html))]
    except Exception:
        tables = []

    records = {}

    # HorseList is the pre-race source for the current entry facts. Parse it
    # separately so these facts survive changes to the surrounding tables.
    soup = BeautifulSoup(newspaper_html, "html.parser")
    current_race_date = parse_race_date_from_text(
        " ".join(
            [
                text_of(soup.title),
                str((soup.select_one("meta[name='description']") or {}).get("content", "")),
                str((soup.select_one("meta[property='og:description']") or {}).get("content", "")),
            ]
        )
    )
    current_class_rank, current_class_label = race_class_info_from_soup(soup)
    for horse_row in soup.select("dl.HorseList"):
        horse_no = parse_int_from_text(text_of(horse_row.select_one("dt.Waku_Horse")))
        if horse_no is None:
            continue
        record = records.setdefault(horse_no, {"馬番": horse_no})
        horse_name = text_of(horse_row.select_one(".Horse_Info .Horse02 a"))
        style = text_of(horse_row.select_one(".Horse_Info .Horse06 .Type span"))
        horse06 = text_of(horse_row.select_one(".Horse_Info .Horse06"))
        interval_match = re.search(r"(連闘|中\s*\d+\s*週|休み明け|長期休養)", horse06)
        odds = parse_float_from_text(text_of(horse_row.select_one('[id^="odds-1_"]')))
        popularity = parse_int_from_text(text_of(horse_row.select_one('[id^="ninki-1_"]')))
        jockey_cell = horse_row.select_one("dd.Jockey")
        jockey = text_of(jockey_cell.select_one("a")) if jockey_cell is not None else ""
        jockey_change = text_of(jockey_cell.select_one("a .Change")) if jockey_cell is not None else ""
        jockey = re.sub(r"^(?:替|継)\s*", "", jockey)
        body_weight, body_weight_value, body_weight_change = _jra_newspaper_body_weight(horse_row)
        newspaper_past_runs = _jra_newspaper_past_runs(horse_row, current_race_date)
        previous_run = newspaper_past_runs[0] if newspaper_past_runs else {}
        ranked_runs = [run for run in newspaper_past_runs if run.get("class_rank") is not None]
        best_run = max(ranked_runs, key=lambda run: run.get("class_rank")) if ranked_runs else {}
        previous_class_rank = previous_run.get("class_rank")
        previous_class_label = norm_text(str(previous_run.get("class_label") or ""))
        days_since_last = days_between(current_race_date, previous_run.get("race_date"))
        current_load = None
        if jockey_cell is not None:
            for span in reversed(jockey_cell.find_all("span", recursive=False)):
                value = text_of(span)
                if re.fullmatch(r"\d{2}(?:\.\d)?", value):
                    current_load = parse_float_from_text(value)
                    break
        if horse_name:
            record["_新聞馬名"] = horse_name
        if odds is not None:
            record["_新聞単勝オッズ"] = odds
        if popularity is not None:
            record["_新聞人気"] = popularity
        if style:
            record["_新聞脚質"] = style
        if interval_match:
            record["_新聞レース間隔"] = re.sub(r"\s+", "", interval_match.group(1))
        if current_load is not None:
            record["_新聞斤量"] = current_load
        if jockey:
            record["_新聞騎手"] = jockey
        if jockey_change:
            record["_新聞騎手変更"] = jockey_change
        if body_weight:
            record["_新聞馬体重"] = body_weight
            record["_新聞馬体重値"] = body_weight_value
            record["_新聞馬体重増減"] = body_weight_change
        if newspaper_past_runs:
            record["_新聞過去走"] = newspaper_past_runs
        if current_class_rank is not None:
            record["_新聞今回クラス順位"] = current_class_rank
            record["_新聞今回クラス"] = current_class_label
        if previous_class_rank is not None:
            record["_新聞前走クラス順位"] = previous_class_rank
            record["_新聞前走クラス"] = previous_class_label
        if best_run:
            record["_新聞最高クラス順位"] = best_run.get("class_rank")
            record["_新聞最高クラス"] = best_run.get("class_label")
        past_class_labels = [
            norm_text(str(run.get("class_label") or ""))
            for run in newspaper_past_runs
            if norm_text(str(run.get("class_label") or ""))
        ]
        if past_class_labels:
            record["_新聞過去クラス"] = past_class_labels
        class_shift = class_shift_label(current_class_rank, previous_class_rank)
        if class_shift:
            record["_新聞クラス変動"] = class_shift
        if days_since_last is not None:
            record["_新聞前走間隔日数"] = days_since_last

    for df in tables:
        horse_col = find_column_by_keywords(df, "馬", "番")
        early_col = find_column_by_keywords(df, "前半", "3F")
        late_col = find_column_by_keywords(df, "後半", "3F")
        if horse_col is None or late_col is None:
            continue
        for _, row in df.iterrows():
            horse_no = parse_int_from_text(clean_cell_text(row.get(horse_col)))
            late3f = parse_float_from_text(clean_cell_text(row.get(late_col)))
            early3f = parse_float_from_text(clean_cell_text(row.get(early_col))) if early_col is not None else None
            if horse_no is None or late3f is None:
                continue
            record = records.setdefault(horse_no, {"馬番": horse_no})
            if early3f is not None:
                record["推定前半3F"] = early3f
            record["推定後半3F"] = late3f

    for df in tables:
        horse_col = find_column_by_keywords(df, "馬", "番")
        comment_col = find_column_by_keywords(df, "コメント")
        if horse_col is None or comment_col is None:
            continue
        for _, row in df.iterrows():
            horse_no = parse_int_from_text(clean_cell_text(row.get(horse_col)))
            comment = clean_cell_text(row.get(comment_col))
            if horse_no is None or not comment:
                continue
            record = records.setdefault(horse_no, {"馬番": horse_no})
            record["新聞コメント"] = comment

    for df in tables:
        horse_col = find_column_by_keywords(df, "馬", "番")
        jockey_col = find_column_by_keywords(df, "騎手")
        if horse_col is None or jockey_col is None:
            continue
        for _, row in df.iterrows():
            horse_no = parse_int_from_text(clean_cell_text(row.get(horse_col)))
            jockey = clean_cell_text(row.get(jockey_col))
            if horse_no is None or not jockey:
                continue
            record = records.setdefault(horse_no, {"馬番": horse_no})
            record["_新聞騎手"] = jockey

    for df in tables:
        horse_col = find_column_by_keywords(df, "馬", "番")
        time_col = find_column_by_keywords(df, "調教")
        if horse_col is None or time_col is None:
            continue
        name_col = find_column_by_keywords(df, "馬名")
        date_col = find_column_by_keywords(df, "日付")
        eval_cols = [col for col in df.columns if "評価" in str(col)]
        for horse_no, group in df.groupby(df[horse_col].map(lambda value: parse_int_from_text(clean_cell_text(value)))):
            if horse_no is None or pd.isna(horse_no):
                continue
            horse_no = int(horse_no)
            record = records.setdefault(horse_no, {"馬番": horse_no})
            training_comment = ""
            training_grade = ""
            training_label = ""

            for _, row in group.iterrows():
                row_texts = [clean_cell_text(row.get(col)) for col in df.columns]
                name_text = clean_cell_text(row.get(name_col)) if name_col else ""
                date_text = clean_cell_text(row.get(date_col)) if date_col else ""
                if date_text == "前走" or name_text.endswith("前走") or "前走" in name_text:
                    long_texts = [
                        text for text in row_texts
                        if len(text) >= 18 and not re.search(r"\d{4}/\d{1,2}/\d{1,2}", text)
                    ]
                    if long_texts:
                        training_comment = max(long_texts, key=len)

                for col in eval_cols:
                    value = clean_cell_text(row.get(col))
                    if re.fullmatch(r"[A-CＳS][+-]?", value):
                        training_grade = value.replace("Ｓ", "S")
                    elif value and len(value) <= 12 and value not in {"評価", training_grade}:
                        training_label = value

            if training_comment:
                record["調教コメント"] = training_comment
            if training_grade or training_label:
                record["_調教評価記号"] = training_grade
                record["_調教評価文"] = training_label
                record["調教評価"] = " ".join(part for part in [training_grade, training_label] if part)

    if not records:
        return pd.DataFrame(columns=columns)
    result = pd.DataFrame(records.values())
    for column in columns:
        if column not in result.columns:
            result[column] = ""
    return result[columns].drop_duplicates("馬番")


def _jra_newspaper_body_weight(horse_row):
    text = text_of(horse_row.select_one(".Horse_Info .Horse07 .Weight"))
    match = re.search(r"(\d{3})\s*(?:kg)?\s*\(([+-]?\d+)\)", text, flags=re.I)
    if not match:
        return "", None, None
    value = int(match.group(1))
    change = int(match.group(2))
    display = f"{value}({change:+d})" if change else f"{value}(0)"
    return display, value, change


def _jra_newspaper_past_runs(horse_row, current_race_date):
    result = []
    labels = ("前走", "2走前", "3走前")
    for index, past in enumerate(horse_row.select(".Past_Wrapper li.Past")[:3]):
        grade = past.select_one("[class*='GradeType'], [class*='GradeIcon'], [class*='Icon_Grade']")
        grade_text = text_of(grade)
        if grade_text.upper() == "L":
            grade_text = "(L)"
        grade_classes = " ".join(grade.get("class", [])) if grade is not None else ""
        class_rank, class_label = race_class_info(
            " ".join(
                part
                for part in (grade_classes, grade_text, text_of(past.select_one(".Data03")))
                if part
            )
        )
        result.append(
            {
                "label": labels[index],
                "race_name": text_of(past.select_one(".RaceName")),
                "race_date": _jra_newspaper_month_day(text_of(past.select_one(".Data01")), current_race_date),
                "class_rank": class_rank,
                "class_label": class_label,
                "position": parse_int_from_text(text_of(past.select_one(".Data04 .Num"))),
            }
        )
    return result


def _jra_newspaper_month_day(value, current_race_date):
    match = re.search(r"(\d{1,2})[./-](\d{1,2})", norm_text(str(value or "")))
    if not match or not current_race_date:
        return None
    month = int(match.group(1))
    day = int(match.group(2))
    year = current_race_date.year
    try:
        parsed = date(year, month, day)
    except ValueError:
        return None
    if parsed > current_race_date:
        try:
            parsed = date(year - 1, month, day)
        except ValueError:
            return None
    return parsed


def build_jra_newspaper_materials(row):
    text = " ".join(
        clean_cell_text(row.get(column))
        for column in ["新聞コメント", "調教コメント", "調教評価", "_調教評価記号", "_調教評価文"]
    )
    grade = clean_cell_text(row.get("_調教評価記号"))
    label = clean_cell_text(row.get("_調教評価文"))
    materials = []
    score = 0.0

    def add(text_label, value):
        nonlocal score
        if text_label and text_label not in materials:
            materials.append(text_label)
        score += value

    if grade in {"S", "A"}:
        add(f"調教{grade}", 2.0 if grade == "A" else 2.4)
    elif grade == "B":
        score += 0.3

    if "文句なし" in text:
        add("調教文句なし", 1.6)
    if any(key in text for key in ["好調", "好状態", "状態はいい", "体調は悪くない", "好気配", "元気いっぱい", "態勢整う", "出来は良"]):
        add("状態良", 0.9)
    if any(key in text for key in ["軽量", "軽ハンデ", "50キロ", "51キロ"]):
        add("軽ハンデ", 0.9)
    if re.search(r"小倉.{0,12}(得意|馬場もいい|好相性|悪くない|向き)", text) or re.search(r"(得意|好相性|悪くない).{0,12}小倉", text):
        add("小倉向き", 1.2)
    if "スピードはある" in text:
        add("スピードあり", 0.8)
    if any(key in text for key in ["次につながる", "勝ち方", "巻き返し"]):
        add("前走評価", 0.6)
    if "千二なら" in text or "千二への対応" in text:
        add("距離材料", 0.4)

    if any(key in text for key in ["対応が鍵", "鍵に", "どうか", "どこまでやれるか", "どこまでやれるか楽しみ"]):
        add("重賞未知", -0.8)
    if any(key in text for key in ["重賞に挑戦", "重賞でどこまで"]):
        score -= 0.4
    if "遅れ" in text and "併入" not in text:
        score -= 0.4

    if label and label not in materials and len(label) <= 8:
        materials.append(label)
    return " / ".join(materials[:4]), round(min(max(score, -2.0), 3.0), 2)


def combine_material_text(*texts, limit=5):
    parts = []
    for text in texts:
        for part in re.split(r"\s*/\s*", clean_cell_text(text)):
            if part and part not in parts:
                parts.append(part)
    return " / ".join(parts[:limit])


def build_jra_late3f_materials(df):
    if "推定後半3F" not in df.columns:
        return pd.Series("", index=df.index), pd.Series(0.0, index=df.index)
    late = pd.to_numeric(df.get("推定後半3F"), errors="coerce")
    early = pd.to_numeric(df.get("推定前半3F"), errors="coerce")
    valid_count = int(late.notna().sum())
    if valid_count < 3:
        return pd.Series("", index=df.index), pd.Series(0.0, index=df.index)

    late_rank = late.rank(method="min", ascending=True)
    early_rank = early.rank(method="min", ascending=True)
    styles = df.get("脚質", pd.Series("", index=df.index)).map(normalize_running_style)
    materials = []
    scores = []

    for idx in df.index:
        rank = late_rank.loc[idx]
        style = styles.loc[idx]
        if pd.isna(rank):
            materials.append("")
            scores.append(0.0)
            continue

        score = 0.0
        label = ""
        if rank <= 1:
            if style in ("逃", "先"):
                label = "先行上がり最速"
                score = 1.4
            elif style == "差":
                label = "末脚最速"
                score = 0.9
            else:
                label = "上がり最速も展開待ち"
                score = 0.35
        elif rank <= 3:
            if style in ("逃", "先"):
                label = "先行上がり優秀"
                score = 1.0
            elif style == "差":
                label = "末脚上位"
                score = 0.65
            else:
                label = "上がり上位も展開待ち"
                score = 0.25
        elif rank <= 5:
            label = "末脚注意"
            score = 0.15

        if label and pd.notna(early_rank.loc[idx]) and early_rank.loc[idx] <= 3 and rank <= 3 and style in ("逃", "先"):
            label = combine_material_text(label, "前後半バランス", limit=2)
            score += 0.25

        materials.append(label)
        scores.append(round(min(max(score, -0.5), 1.6), 2))
    return pd.Series(materials, index=df.index), pd.Series(scores, index=df.index)


def apply_jra_newspaper_html_features(df, newspaper_html):
    result = df.copy()
    if not newspaper_html:
        return result
    newspaper_df = parse_jra_newspaper_html(newspaper_html)
    if newspaper_df.empty:
        result["調教評価"] = ""
        result["新聞材料"] = ""
        result["_新聞材料点"] = 0.0
        return result

    result = result.merge(newspaper_df, on="馬番", how="left")
    for column in ["_新聞騎手", "新聞コメント", "調教コメント", "調教評価", "_調教評価記号", "_調教評価文", "推定前半3F", "推定後半3F"]:
        if column not in result.columns:
            result[column] = ""
        result[column] = result[column].fillna("").astype(str)
    if "_新聞騎手" in result.columns:
        newspaper_jockey = result["_新聞騎手"].fillna("").astype(str).str.strip()
        jockey_present = newspaper_jockey.ne("")
        if bool(jockey_present.any()):
            if "騎手" not in result.columns:
                result["騎手"] = ""
            else:
                result["騎手"] = result["騎手"].astype("object")
            result.loc[jockey_present, "騎手"] = newspaper_jockey.loc[jockey_present]

    # Current newspaper facts are copied into their normal display/material
    # columns after Ver3 scoring. They can therefore affect only the independent
    # market/material view, never _ver3_ability_core.
    fact_columns = {
        "_新聞馬名": "馬名",
        "_新聞単勝オッズ": "オッズ",
        "_新聞人気": "人気",
        "_新聞脚質": "脚質",
        "_新聞レース間隔": "レース間隔",
        "_新聞斤量": "斤量",
        "_新聞騎手": "騎手",
        "_新聞馬体重": "馬体重",
    }
    numeric_fact_targets = {"オッズ", "人気", "斤量"}

    def assign_present_fact_values(source, target):
        if source not in result.columns:
            return
        source_values = result[source]
        target_is_numeric = target in numeric_fact_targets or (
            target in result.columns and pd.api.types.is_numeric_dtype(result[target])
        )
        if target_is_numeric:
            values = pd.to_numeric(source_values, errors="coerce")
            present = values.notna()
        else:
            values = source_values.fillna("").astype(str).str.strip()
            present = values.ne("")
        if not bool(present.any()):
            return
        if target not in result.columns:
            result[target] = (
                pd.Series(pd.NA, index=result.index, dtype="Float64")
                if target_is_numeric
                else pd.Series([""] * len(result), index=result.index, dtype="object")
            )
        elif target_is_numeric and pd.api.types.is_integer_dtype(result[target]):
            non_integer_values = values.loc[present].dropna().map(float).mod(1).ne(0).any()
            if bool(non_integer_values):
                result[target] = result[target].astype("Float64")
        elif not target_is_numeric:
            result[target] = result[target].astype("object")
        result.loc[present, target] = values.loc[present]

    for source, target in fact_columns.items():
        assign_present_fact_values(source, target)

    newspaper_scalar_fields = {
        "_新聞馬体重値": "_body_weight",
        "_新聞馬体重増減": "_body_weight_change",
        "_新聞今回クラス順位": "_current_class_rank",
        "_新聞今回クラス": "_current_class_label",
        "_新聞前走クラス順位": "_previous_class_rank",
        "_新聞前走クラス": "_previous_class_label",
        "_新聞最高クラス順位": "_best_past_class_rank",
        "_新聞最高クラス": "_best_past_class_label",
        "_新聞過去クラス": "_past_class_labels",
        "_新聞クラス変動": "_class_shift",
        "_新聞前走間隔日数": "_days_since_last",
    }
    for source, target in newspaper_scalar_fields.items():
        if source not in result.columns:
            continue
        if target not in result.columns:
            result[target] = pd.Series([None] * len(result), index=result.index, dtype="object")
        for index, value in result[source].items():
            if _jra_newspaper_value_present(value) and not _jra_newspaper_value_present(result.at[index, target]):
                result.at[index, target] = value

    if "_新聞過去走" in result.columns:
        if "_past_runs" not in result.columns:
            result["_past_runs"] = pd.Series([[] for _ in range(len(result))], index=result.index, dtype="object")
        result["_past_runs"] = result.apply(
            lambda row: _merge_jra_past_run_evidence(row.get("_past_runs"), row.get("_新聞過去走")),
            axis=1,
        )
    if "_新聞斤量" in result.columns:
        if "_current_load_weight" not in result.columns:
            result["_current_load_weight"] = pd.NA
        load_values = pd.to_numeric(result["_新聞斤量"], errors="coerce")
        load_present = load_values.notna()
        if bool(load_present.any()):
            result.loc[load_present, "_current_load_weight"] = load_values.loc[load_present]
    if "_新聞騎手" in result.columns:
        if "_current_jockey" not in result.columns:
            result["_current_jockey"] = ""
        jockey_values = result["_新聞騎手"].fillna("").astype(str).str.strip()
        jockey_present = jockey_values.ne("")
        if bool(jockey_present.any()):
            result.loc[jockey_present, "_current_jockey"] = jockey_values.loc[jockey_present]
    if "_新聞騎手変更" in result.columns:
        if "_jockey_changed" not in result.columns:
            result["_jockey_changed"] = False
        changed = result["_新聞騎手変更"].fillna("").astype(str).str.contains("替")
        result.loc[changed, "_jockey_changed"] = True

    material_rows = result.apply(build_jra_newspaper_materials, axis=1)
    late3f_materials, late3f_scores = build_jra_late3f_materials(result)
    result["末脚材料"] = late3f_materials
    result["_末脚材料点"] = late3f_scores
    result["新聞材料"] = [
        combine_material_text(base, late3f)
        for base, late3f in zip([item[0] for item in material_rows], late3f_materials)
    ]
    result["_新聞材料点"] = [
        round(min(max(float(base_score) + float(late_score), -2.5), 3.8), 2)
        for base_score, late_score in zip([item[1] for item in material_rows], late3f_scores)
    ]
    return result


def _jra_newspaper_value_present(value):
    if value is None:
        return False
    if isinstance(value, (list, tuple, dict, set)):
        return bool(value)
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    return norm_text(str(value)) not in {"", "nan", "None"}


def _merge_jra_past_run_evidence(existing, newspaper):
    base = [dict(run) for run in existing if isinstance(run, dict)] if isinstance(existing, list) else []
    additions = [dict(run) for run in newspaper if isinstance(run, dict)] if isinstance(newspaper, list) else []
    by_role = {_jra_past_run_role(run): run for run in base if _jra_past_run_role(run)}
    for run in additions:
        role = _jra_past_run_role(run)
        target = by_role.get(role)
        if target is None:
            base.append(run)
            if role:
                by_role[role] = run
            continue
        for key, value in run.items():
            if _jra_newspaper_value_present(value) and not _jra_newspaper_value_present(target.get(key)):
                target[key] = value
    order = {"3走前": 0, "2走前": 1, "前走": 2}
    return sorted(base, key=lambda run: order.get(_jra_past_run_role(run), 9))


def _jra_past_run_role(run):
    label = norm_text(str((run or {}).get("label") or (run or {}).get("key") or (run or {}).get("run_key") or ""))
    aliases = {
        "前走": "前走", "last": "前走", "race1": "前走", "1走前": "前走",
        "2走前": "2走前", "2back": "2走前", "race2": "2走前",
        "3走前": "3走前", "3back": "3走前", "race3": "3走前",
    }
    return aliases.get(label, "")



def extract_oikiri_grade_and_label(text):
    clean = clean_cell_text(text)
    grade = ""
    label = ""
    tokens = re.split(r"\s+", clean)
    for token in tokens:
        token = token.strip()
        if re.fullmatch(r"[ABCDSＳ][+-]?", token):
            grade = token.replace("Ｓ", "S")
        elif token and len(token) <= 12 and not re.search(r"\d", token) and token not in {"評価", "映像", "前走"}:
            if any("\u3040" <= ch <= "\u9fff" for ch in token):
                label = token
    if not grade:
        match = re.search(r"(?:^|\s)([ABCDSＳ][+-]?)(?:\s|$)", clean)
        if match:
            grade = match.group(1).replace("Ｓ", "S")
    if not label and grade:
        label_text = clean.replace(grade, " ")
        label_text = re.sub(r"前走|映像|評価|--|◎|◯|○|▲|△|☆|消|✓", " ", label_text)
        label_parts = [
            part for part in re.split(r"\s+", label_text)
            if part and len(part) <= 12 and not re.search(r"\d", part)
        ]
        if label_parts:
            label = label_parts[-1]
    return grade, label


def extract_oikiri_time_text(text):
    clean = clean_cell_text(text)
    if not clean:
        return "", ""
    has_course = re.search(r"(坂路|坂|CW|ＣＷ|W|ウッド|芝|ダ|ポリ|栗|美|小倉|函館|札幌)", clean)
    numbers = re.findall(r"\d{1,3}\.\d", clean)
    if len(numbers) < 2 and not has_course:
        return "", ""
    if not numbers:
        return "", ""
    last_1f = numbers[-1]
    time_text = clean
    if len(time_text) > 60:
        time_text = " ".join(numbers[-6:])
    return time_text[:80], last_1f


def extract_oikiri_awase_text(text):
    clean = clean_cell_text(text)
    if not clean:
        return ""
    keywords = ["併", "遅れ", "先着", "同入", "併入", "追走", "内", "外", "中", "馬なり", "一杯"]
    if not any(keyword in clean for keyword in keywords):
        return ""
    if len(clean) > 90:
        pieces = []
        for part in re.split(r"\s+|。|、", clean):
            if any(keyword in part for keyword in keywords):
                pieces.append(part)
        clean = " ".join(pieces) if pieces else clean
    return clean[:90]


def parse_jra_oikiri_html(oikiri_html):
    columns = [
        "馬番", "追切評価", "追切短評", "追切時計", "ラスト1F", "併せ内容",
        "_追切評価記号", "_追切評価文",
    ]
    if not oikiri_html:
        return pd.DataFrame(columns=columns)
    try:
        tables = [flatten_table_columns(df) for df in pd.read_html(StringIO(oikiri_html))]
    except Exception:
        tables = []

    records = {}

    def ensure_record(horse_no):
        horse_no = int(horse_no)
        return records.setdefault(horse_no, {"馬番": horse_no})

    for df in tables:
        horse_col = find_column_by_keywords(df, "馬", "番")
        name_col = find_column_by_keywords(df, "馬名")
        eval_cols = [col for col in df.columns if "評価" in str(col)]
        candidate_cols = list(df.columns)

        for _, row in df.iterrows():
            row_texts = [clean_cell_text(row.get(col)) for col in candidate_cols]
            row_text = " ".join(text for text in row_texts if text)
            horse_no = parse_int_from_text(clean_cell_text(row.get(horse_col))) if horse_col else None
            if horse_no is None:
                match = re.match(r"\s*\d{1,2}\s+(\d{1,2})\b", row_text)
                if match:
                    horse_no = int(match.group(1))
            if horse_no is None or horse_no <= 0 or horse_no > 30:
                continue

            record = ensure_record(horse_no)
            grade = ""
            label = ""
            time_text = ""
            last_1f = ""
            awase = ""

            eval_sources = []
            for col in eval_cols:
                value = clean_cell_text(row.get(col))
                if value:
                    eval_sources.append(value)
            if not eval_sources:
                eval_sources = row_texts

            for value in eval_sources:
                g, l = extract_oikiri_grade_and_label(value)
                if g:
                    grade = g
                if l:
                    label = l

            if not grade or not label:
                g, l = extract_oikiri_grade_and_label(row_text)
                grade = grade or g
                label = label or l

            for value in row_texts:
                t, last = extract_oikiri_time_text(value)
                if t and len(t) > len(time_text):
                    time_text = t
                    last_1f = last
                a = extract_oikiri_awase_text(value)
                if a and len(a) > len(awase):
                    awase = a

            if not awase:
                awase = extract_oikiri_awase_text(row_text)
            if not time_text:
                time_text, last_1f = extract_oikiri_time_text(row_text)

            if grade or label:
                record["_追切評価記号"] = grade
                record["_追切評価文"] = label
                record["追切短評"] = label
                record["追切評価"] = " ".join(part for part in [grade, label] if part)
            if time_text:
                record["追切時計"] = time_text
            if last_1f:
                record["ラスト1F"] = last_1f
            if awase:
                record["併せ内容"] = awase

    if not records:
        return pd.DataFrame(columns=columns)
    result = pd.DataFrame(records.values())
    for column in columns:
        if column not in result.columns:
            result[column] = ""
    return result[columns].drop_duplicates("馬番")


def build_jra_oikiri_materials(row):
    text = " ".join(
        clean_cell_text(row.get(column))
        for column in ["追切評価", "追切短評", "追切時計", "ラスト1F", "併せ内容", "_追切評価記号", "_追切評価文"]
    )
    grade = clean_cell_text(row.get("_追切評価記号"))
    label = clean_cell_text(row.get("_追切評価文"))
    awase = clean_cell_text(row.get("併せ内容"))
    last_1f = safe_num(parse_float_from_text(clean_cell_text(row.get("ラスト1F"))), None)
    materials = []
    score = 0.0

    def add(text_label, value):
        nonlocal score
        if text_label and text_label not in materials:
            materials.append(text_label)
        score += value

    if grade in {"S", "A"}:
        add(f"追切{grade}", 2.0 if grade == "A" else 2.4)
    elif grade == "B":
        score += 0.2
    elif grade == "C":
        add("追切C", -0.7)
    elif grade == "D":
        add("追切D", -1.3)

    if "文句なし" in text:
        add("文句なし", 1.3)
    if any(key in text for key in ["好調", "好気配", "力強い", "キビキビ", "態勢整う", "出来は良", "元気一杯", "一歩前進", "末脚良し"]):
        add("動き良", 0.7)
    if any(key in text for key in ["重い", "平凡", "一息", "伸び欠", "物足り", "不安", "精彩欠"]):
        add("動き不安", -0.9)
    if "一番時計" in text:
        add("一番時計", 1.0)
    if last_1f is not None and last_1f <= 11.5:
        add("終い良", 0.5)

    low_class_words = [
        "格下", "下級", "未勝利", "未勝", "三未", "三未勝", "二未", "二未勝",
        "新馬", "1勝", "１勝", "一勝", "1勝C", "１勝Ｃ", "500万",
        "2歳", "２歳", "3歳未勝利", "三歳未勝利",
    ]
    delayed = bool(re.search(r"遅れ(?!ず|なし|ない)", awase))
    if delayed:
        if any(word in awase for word in low_class_words):
            add("格下遅れ", -2.0)
        else:
            add("併せ遅れ", -1.0)
    elif "先着" in awase:
        add("併せ先着", 0.8)
    elif "併入" in awase or "同入" in awase:
        add("併せ互角", 0.2)

    if label and label not in materials and len(label) <= 8:
        materials.append(label)
    return " / ".join(materials[:5]), round(min(max(score, -3.0), 3.0), 2)


def apply_jra_oikiri_html_features(df, oikiri_html):
    result = df.copy()
    if not oikiri_html:
        return result
    oikiri_df = parse_jra_oikiri_html(oikiri_html)
    if oikiri_df.empty:
        result["追切評価"] = ""
        result["追切材料"] = ""
        result["併せ内容"] = ""
        result["_追切材料点"] = 0.0
        return result

    result = result.merge(oikiri_df, on="馬番", how="left")
    for column in ["追切評価", "追切短評", "追切時計", "ラスト1F", "併せ内容", "_追切評価記号", "_追切評価文"]:
        if column not in result.columns:
            result[column] = ""
        result[column] = result[column].fillna("").astype(str)

    material_rows = result.apply(build_jra_oikiri_materials, axis=1)
    result["追切材料"] = [item[0] for item in material_rows]
    result["_追切材料点"] = [item[1] for item in material_rows]
    return result



def normalize_state_material_part(part, grade_text=""):
    text = clean_cell_text(part)
    if not text:
        return ""
    replacements = {
        "調教文句なし": "文句なし",
        "調教S": "S",
        "調教A": "A",
        "追切S": "S",
        "追切A": "A",
        "状態良": "好気配",
        "不安あり": "重賞未知",
    }
    text = replacements.get(text, text)
    if text in {"追切B", "調教B"}:
        return ""
    if text in {"S", "A", "B", "C", "D"} and re.search(r"\b" + re.escape(text) + r"\b", grade_text):
        return ""
    return text


def build_state_material_text(row):
    parts = []
    grade_text = clean_cell_text(row.get("追切評価")) or clean_cell_text(row.get("調教評価"))

    def add(value):
        value = normalize_state_material_part(value, grade_text)
        if not value:
            return
        if grade_text and value in grade_text:
            return
        if value not in parts:
            parts.append(value)

    if grade_text:
        parts.append(grade_text)

    h2h_label = clean_cell_text(row.get("_h2h_label"))
    h2h_latest = clean_cell_text(row.get("_h2h_latest") or row.get("対戦"))
    h2h_score = safe_num(row.get("_h2h_score"), 0) or 0
    if h2h_label == "対戦◎" or h2h_score >= 2:
        source_parts = ["対戦◎"]
    elif h2h_score > 0 or h2h_label == "対戦○" or "先着" in h2h_latest:
        source_parts = ["対戦先着"]
    else:
        source_parts = []

    priority = [
        "格下遅れ", "併せ遅れ", "動き不安", "追切C", "追切D",
        "文句なし", "一番時計", "好気配", "高指数", "コース実績", "コース穴", "小倉向き",
        "先行上がり最速", "先行上がり優秀", "末脚最速", "末脚上位", "上がり最速も展開待ち", "上がり上位も展開待ち", "末脚注意",
        "対戦◎", "対戦先着", "対戦穴",
        "併せ先着", "軽ハンデ", "終い良", "動き良", "併せ互角",
        "距離材料", "前走評価", "スピードあり", "重賞未知",
    ]
    for column in ["追切材料", "新聞材料", "_検討指数材料"]:
        for part in re.split(r"\s*/\s*", clean_cell_text(row.get(column))):
            part = normalize_state_material_part(clean_cell_text(part), grade_text)
            if part and part not in source_parts:
                source_parts.append(part)

    awase = clean_cell_text(row.get("併せ内容"))
    if awase:
        low_class_words = [
            "格下", "下級", "未勝利", "未勝", "三未", "三未勝", "二未", "二未勝",
            "新馬", "1勝", "１勝", "一勝", "1勝C", "１勝Ｃ", "500万",
            "2歳", "２歳", "3歳未勝利", "三歳未勝利",
        ]
        delayed = bool(re.search(r"遅れ(?!ず|なし|ない)", awase))
        if delayed and any(word in awase for word in low_class_words):
            source_parts.append("格下遅れ")
        elif delayed:
            source_parts.append("併せ遅れ")
        elif "先着" in awase:
            source_parts.append("併せ先着")
        elif "併入" in awase or "同入" in awase:
            source_parts.append("併せ互角")

    for key in priority:
        if key in source_parts:
            add(key)
    for part in source_parts:
        if len(parts) >= 4:
            break
        add(part)

    return "/".join(parts[:4])


def apply_state_material_column(df):
    result = df.copy()
    if "性齢" in result.columns:
        result["馬年齢"] = result["性齢"].fillna("").astype(str).map(clean_cell_text)
        result["年齢"] = result["性齢"].map(extract_age_from_sex_age)
    else:
        result["馬年齢"] = ""
        result["年齢"] = ""

    if "単勝オッズ" in result.columns:
        result["オッズ"] = pd.to_numeric(result["単勝オッズ"], errors="coerce")
    elif "オッズ" not in result.columns:
        result["オッズ"] = pd.NA

    def numeric(column):
        if column in result.columns:
            return pd.to_numeric(result[column], errors="coerce")
        return pd.Series(float("nan"), index=result.index, dtype="float64")

    average = numeric("平均指数").fillna(numeric("3走平均"))
    star_high = numeric("★最高指数").fillna(numeric("★最高"))
    distance = numeric("距離指数")
    course = numeric("コース指数")
    ranks = {
        "平均": average.rank(method="min", ascending=False),
        "最高": star_high.rank(method="min", ascending=False),
        "距離": distance.rank(method="min", ascending=False),
        "コース": course.rank(method="min", ascending=False),
    }

    index_labels = []
    field_average_value = safe_num(average.mean(), None)
    odds_for_material = pd.to_numeric(result.get("オッズ"), errors="coerce")
    for idx in result.index:
        labels = []
        course_value = course.loc[idx]
        has_course_record = pd.notna(course_value) and float(course_value) > 0
        if has_course_record:
            labels.append("コース実績")
        top1 = any(pd.notna(series.loc[idx]) and series.loc[idx] <= 1 for series in ranks.values())
        top3 = any(pd.notna(series.loc[idx]) and series.loc[idx] <= 3 for series in ranks.values())
        strong_value = any(
            pd.notna(value) and value >= 100
            for value in [average.loc[idx], star_high.loc[idx], distance.loc[idx], course.loc[idx]]
        )
        if top1 or (top3 and strong_value):
            labels.insert(0, "高指数")
        elif top3:
            labels.insert(0, "指数上位")
        course_hole = (
            pd.notna(ranks["コース"].loc[idx])
            and ranks["コース"].loc[idx] <= 3
            and pd.notna(odds_for_material.loc[idx])
            and odds_for_material.loc[idx] >= 8
            and pd.isna(star_high.loc[idx])
            and pd.notna(average.loc[idx])
            and (
                (field_average_value is not None and average.loc[idx] <= field_average_value - 5)
                or ranks["平均"].loc[idx] >= 6
            )
        )
        if course_hole:
            labels.insert(0, "コース穴")
        index_labels.append("/".join(dict.fromkeys(label for label in labels if label)))
    result["_検討指数材料"] = index_labels

    result["状態材料"] = result.apply(build_state_material_text, axis=1)
    result["調教/評価/検討材料"] = result["状態材料"]
    return result

def prepare_jra_display_columns(df):
    prepared = df.copy()
    if "馬年齢" not in prepared.columns and "性齢" in prepared.columns:
        prepared["馬年齢"] = prepared["性齢"].fillna("").astype(str).map(clean_cell_text)
    if "騎手" not in prepared.columns:
        prepared["騎手"] = "―"
    else:
        jockey = prepared["騎手"].fillna("").astype(str).map(clean_cell_text)
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
    if "調教/評価/検討材料" not in prepared.columns and "状態材料" in prepared.columns:
        prepared["調教/評価/検討材料"] = prepared["状態材料"].fillna("").astype(str)
    prepared["評価根拠"] = prepared.apply(build_jra_evaluation_material, axis=1)
    return prepared

def build_jra_evaluation_material(row):
    parts = []

    def add(label):
        if label and label not in parts:
            parts.append(label)

    material = str(row.get("調教/評価/検討材料") or row.get("状態材料") or "")
    reason = str(row.get("印理由") or "")
    class_shift = str(row.get("クラス変動") or "")
    ability_text = str(row.get("能力") or "")
    h2h_text = str(row.get("対戦") or row.get("_h2h_latest") or "")

    if class_shift:
        add(class_shift)
    for keyword in ["能力上位", "最高指数", "高指数", "コース実績", "距離実績", "小倉向き", "福島向き", "函館向き", "対戦先着", "展開向く", "同馬場実績"]:
        if keyword in material or keyword in reason or keyword in h2h_text:
            add(keyword)
    if "上位" in ability_text:
        add("能力上位")
    if str(row.get("展開印", "")).strip() == "展":
        add("展開向く")
    if "先着" in h2h_text:
        add("対戦先着")

    if not parts:
        for piece in material.replace(" / ", "/").split("/"):
            piece = piece.strip()
            if piece:
                add(piece)
            if len(parts) >= 4:
                break
    return " / ".join(parts[:5])

def apply_jra_style_features(df, style_html):
    df = df.copy()
    if not style_html:
        df["脚質"] = ""
        return df, pd.DataFrame()

    style_df = parse_jra_style_table(style_html)
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

    df["馬場適性"] = df.apply(going_label, axis=1)
    df["クラス変動"] = df.get("_class_shift", pd.Series("", index=df.index)).fillna("")
    return df


def safe_num(value, fallback):
    if value is None:
        return fallback
    if isinstance(value, float) and math.isnan(value):
        return fallback
    return value


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

    raw_scores = []
    ver3_ability_cores = []
    market_non_ability_adjustments = []
    for _, row in df.iterrows():
        avg3 = safe_num(row["3走平均"], field_avg3)
        dist = safe_num(row["距離指数"], avg3)
        course = safe_num(row["コース指数"], avg3)
        latest = safe_num(row["_last"], avg3)
        trend = safe_num(row["_trend"], 0)
        star_high = safe_num(row.get("_star_high"), None)
        similar_condition_high = safe_num(row.get("_similar_condition_high"), None)
        best_recent = safe_num(row["近3走最高"], avg3)
        weight_adjustment = safe_num(row.get("_total_load_weight_adjustment"), 0)

        bonus = 0
        if star_high is not None and pd.notna(field_star_high):
            if star_high >= field_star_high + 8:
                bonus += 1.4
            elif star_high >= field_star_high + 4:
                bonus += 0.9
            if pd.notna(top_star_high) and star_high == top_star_high:
                bonus += 1.2
            if pd.notna(row["3走平均"]) and star_high >= row["3走平均"] + 6:
                bonus += 0.8
        if trend >= 5:
            bonus += 1.0
        elif trend >= 2:
            bonus += 0.5
        if latest >= avg3 + 3:
            bonus += 0.8
        if star_high is None and similar_condition_high is not None:
            if similar_condition_high >= avg3 + 6:
                bonus += 1.2
            elif similar_condition_high >= avg3:
                bonus += 0.7
            else:
                bonus += 0.3

        star_component = star_high if star_high is not None else field_avg3
        ability_core = calculate_ver3_ability_core(
            recent_average=avg3,
            star_index=star_component,
            recent_best=best_recent,
            latest_index=latest,
            distance_index=dist,
            course_index=course,
        )
        raw = ability_core + weight_adjustment + bonus
        raw_scores.append(raw)
        ver3_ability_cores.append(ability_core)
        # Legacy Ver3 compatibility keeps these terms in _raw_score. Market
        # mode reads _ver3_ability_core directly; this adjustment column exists
        # only to audit legacy values and old saved snapshots.
        market_non_ability_adjustments.append(weight_adjustment + bonus)

    df["_raw_score"] = raw_scores
    df["_ver3_ability_core"] = ver3_ability_cores
    df["_market_non_ability_adjustment"] = market_non_ability_adjustments
    if "year_max_index" not in df.columns:
        df["year_max_index"] = df.get("_year_max_index", pd.Series(pd.NA, index=df.index))
    if "過去1年最高指数" not in df.columns:
        df["過去1年最高指数"] = df["year_max_index"]
    min_raw = df["_raw_score"].min()
    max_raw = df["_raw_score"].max()
    if max_raw == min_raw:
        df["AI点"] = 80.0
    else:
        df["AI点"] = (60 + 40 * (df["_raw_score"] - min_raw) / (max_raw - min_raw)).round(1)
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
    df["★該当走"] = df["star_max_race"]
    df["★条件"] = df["star_max_condition"]

    df = df.sort_values(["AI点", "3走平均", "距離指数"], ascending=[False, False, False]).reset_index(drop=True)
    df.insert(0, "AI順位", range(1, len(df) + 1))
    top_avg3 = df["3走平均"].max()
    top_dist = df["距離指数"].max()
    top_course = df["コース指数"].max()
    top_ai_for_comment = df["AI点"].max()
    ages = df["性齢"].map(extract_age_from_sex_age)
    recent_high_numeric = pd.to_numeric(df["近3走最高"], errors="coerce")
    field_recent_high = recent_high_numeric.mean()

    def comment(row):
        parts = []
        star_high = safe_num(row.get("_star_high"), None)
        if star_high is not None and pd.notna(top_star_high) and star_high == top_star_high:
            if row["AI点"] >= top_ai_for_comment - 10:
                parts.append("同条件◎")
            else:
                parts.append("条件揃うも相手強化")
        elif star_high is not None and pd.notna(row["3走平均"]) and star_high >= row["3走平均"] + 6:
            parts.append("条件替わり妙味")
        elif star_high is not None:
            parts.append("条件揃う")

        similar_condition_high = safe_num(row.get("_similar_condition_high"), None)
        if star_high is None and similar_condition_high is not None:
            if pd.notna(row.get("3走平均")) and similar_condition_high >= row["3走平均"]:
                parts.append("他場同条件○")
            else:
                parts.append("他場同条件")

        going_note = str(row.get("馬場適性") or "")
        if going_note and going_note != "同馬場未知":
            parts.append(going_note)

        age = extract_age_from_sex_age(row.get("性齢"))
        relative_load_weight = safe_num(row.get("_relative_load_weight"), None)
        young_lightweight = bool(age == 3 and relative_load_weight is not None and relative_load_weight >= 2)
        if young_lightweight:
            parts.append("軽量3歳注意")
        elif relative_load_weight is not None and relative_load_weight >= 2.5:
            parts.append("軽斤量")

        days_since = row.get("_days_since_last")
        if pd.notna(days_since) and days_since >= 60:
            parts.append("休み明け注意")
        elif pd.notna(days_since) and days_since >= 45:
            parts.append("間隔空き")

        if pd.notna(row["3走平均"]) and row["3走平均"] == top_avg3:
            parts.append("近走最上位")
        elif pd.notna(row["3走平均"]) and row["3走平均"] >= field_avg3 + 3:
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
            elif trend >= 2 or star_high is not None:
                parts.append("3歳穴")
        elif age == 4:
            if popularity is not None and popularity <= 5 and row["AI点"] < top_ai_for_comment - 10:
                parts.append("指数以上の支持")
            elif popularity is not None and popularity <= 5:
                parts.append("4歳上積み")
            elif trend >= 2 or star_high is not None:
                parts.append("4歳穴")

        if trend >= 5:
            parts.append("上昇中")
        elif trend >= 2:
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

    top_ai = df["AI点"].max()
    df["妙味スコア"] = df.apply(
        lambda row: int(row["人気"]) - int(row["AI順位"]) if pd.notna(row.get("人気")) else 0,
        axis=1,
    )

    def recommendation_bonus(row):
        value = safe_num(row["妙味スコア"], 0)
        bonus = max(min(value, 8), -4) * 0.35
        if pd.notna(row.get("人気")) and int(row["人気"]) == 1:
            bonus += 1.0
        if row["AI点"] >= top_ai - 8 and value >= 4:
            bonus += 1.2
        return bonus

    df["推奨点"] = (df["AI点"] + df.apply(recommendation_bonus, axis=1)).round(1)

    def age_recommendation_bonus(row):
        age = extract_age_from_sex_age(row.get("性齢"))
        trend = safe_num(row.get("_trend"), 0)
        relative_load_weight = safe_num(row.get("_relative_load_weight"), 0)
        bonus = 0.0
        if age == 3:
            bonus += 3.0
            if trend >= 2:
                bonus += 1.5
            if relative_load_weight >= 2:
                bonus += 1.5
        elif age == 4:
            bonus += 2.2
            if trend >= 2:
                bonus += 1.2
            if relative_load_weight >= 2:
                bonus += 0.5
        elif age is not None and age >= 11:
            bonus -= 3.5
        elif age is not None and age >= 9:
            bonus -= 1.5
        return bonus

    df["_年齢"] = ages
    df["_age_bonus"] = df.apply(age_recommendation_bonus, axis=1).round(1)
    df["推奨点"] = (df["推奨点"] + df["_age_bonus"]).round(1)

    df["役割"] = "消し寄り"
    selected = set()
    popularity = pd.to_numeric(df["人気"], errors="coerce")
    layoff_mask = df.get("_is_layoff", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    star_high_numeric = pd.to_numeric(df["★最高"], errors="coerce")

    def assign_role(mask, role_name, limit):
        candidates = df[mask & ~df.index.isin(selected)].sort_values(
            ["推奨点", "AI点", "★最高", "3走平均"],
            ascending=[False, False, False, False],
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
            "穴軸": "★",
            "妙味あり": "★",
            "相手有力": "○",
            "押さえ": "△",
            "消し寄り": "",
        }.get(row["役割"], "")

    def betting_note(mark):
        return {
            "◎": "本軸",
            "★": "穴",
            "○": "本線",
            "△": "押さえ",
        }.get(mark, "")

    df["印"] = df.apply(betting_mark, axis=1)
    df["買い目メモ"] = df["印"].map(betting_note)

    final_cols = ["推奨順位", "印", "役割", "買い目メモ", "妙味スコア", "AI順位", "枠", "馬番", "馬名", "性齢", "斤量", "騎手", "単勝オッズ", "人気", "距離指数", "コース指数", "3走前", "2走前", "前走", "3走平均", "過去1年最高指数", "year_max_index", "★最高", "★該当走", "★条件", "★最高指数の取得元", "star_max_index", "star_max_race", "star_max_venue", "star_max_distance", "star_max_surface", "star_max_turn", "star_match_level", "star_max_source", "近3走最高", "AI点", "推奨点", "コメント"]
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


def get_betting_groups(df):
    return {
        "main_axis": df[df["役割"].eq("本軸")]["馬番"].tolist(),
        "value_axis": df[df["役割"].eq("穴軸")]["馬番"].tolist(),
        "value": df[df["役割"].eq("妙味あり")]["馬番"].tolist(),
        "main_partners": df[df["役割"].eq("相手有力")]["馬番"].tolist(),
        "reserves": df[df["役割"].eq("押さえ")]["馬番"].tolist(),
    }


def add_focus_scores(df):
    df = df.copy()

    def numeric_col(frame, column, fallback=0):
        if column in frame.columns:
            return pd.to_numeric(frame[column], errors="coerce").fillna(fallback)
        return pd.Series(fallback, index=frame.index, dtype="float64")

    ai = numeric_col(df, "AI点")
    recommend = numeric_col(df, "推奨点", ai)
    avg3 = numeric_col(df, "3走平均", ai)
    recent_high = numeric_col(df, "近3走最高", avg3)
    star_high = numeric_col(df, "★最高", 0)
    similar_condition_high = numeric_col(df, "_similar_condition_high", 0)
    distance = numeric_col(df, "距離指数", avg3)
    course = numeric_col(df, "コース指数", avg3)
    value_score = numeric_col(df, "妙味スコア", 0)
    popularity = numeric_col(df, "人気", 99)
    layoff = df.get("_is_layoff", pd.Series(False, index=df.index)).fillna(False).astype(bool)
    styles = df["脚質"].map(normalize_running_style) if "脚質" in df.columns else pd.Series("", index=df.index)
    ages = df["性齢"].map(extract_age_from_sex_age) if "性齢" in df.columns else pd.Series(None, index=df.index)
    age_bonus = numeric_col(df, "_age_bonus", 0)

    top_ai = ai.max()
    top_recent = recent_high.max()
    top_star = star_high.max()
    top_distance = distance.max()
    top_course = course.max()
    pop_bonus = (13 - popularity.clip(1, 12)) * 0.25
    distance_high = distance >= top_distance - 6
    course_high = course >= top_course - 6
    front_condition = styles.isin(["逃", "先"]) & (popularity <= 4) & (distance_high | course_high)
    escape_condition = styles.eq("逃") & (popularity <= 4) & distance_high & course_high
    leader_condition = styles.eq("先") & (popularity <= 4) & distance_high & course_high
    focus = (
        recommend
        + (ai >= top_ai - 12).astype(float) * 2.0
        + (recent_high >= top_recent - 6).astype(float) * 1.6
        + (star_high.notna() & (star_high >= top_star - 5)).astype(float) * 2.4
        + similar_condition_high * 0.04
        + front_condition.astype(float) * 2.0
        + escape_condition.astype(float) * 2.0
        + leader_condition.astype(float) * 1.0
        + value_score.clip(-4, 8) * 0.35
        + age_bonus * 0.8
        + pop_bonus
    )
    focus -= layoff.astype(float) * 3.0
    focus -= ((ages.fillna(0) >= 8) & (ai <= top_ai - 25)).astype(float) * 2.0
    df["_focus_score"] = focus.round(1)
    return df


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
    ages = df["性齢"].map(extract_age_from_sex_age) if "性齢" in df.columns else pd.Series(None, index=index)
    trend = numeric_series("_trend", 0).fillna(0)
    relative_load_weight = numeric_series("_relative_load_weight", 0).fillna(0)
    comments = df.get("コメント", pd.Series("", index=index)).astype(str)
    race_comments = df.get("展開コメント", pd.Series("", index=index)).astype(str)
    styles = df["脚質"].map(normalize_running_style) if "脚質" in df.columns else pd.Series("", index=index)

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
        | race_comments.str.contains("展開穴|差し届く|能力で粘る|単騎なら粘る|好位で粘る|前残り警戒", na=False)
    ) & not_too_low
    development = (
        ages.isin([3, 4])
        & (
            popular
            | (trend >= 2)
            | ((ages == 3) & (relative_load_weight >= 2))
            | comments.str.contains("3歳|4歳|指数以上の支持|上昇", na=False)
        )
        & (ai >= top_ai - 40)
    )
    value_hole = (
        value_pop
        & (
            (value_score >= 3)
            | condition
            | similar_condition
            | pace
            | ((recommend >= top_recommend - 12) & not_too_low)
        )
    )

    return {
        "能力上位": ability & ~old_low,
        "同条件": condition & ~old_low,
        "準適性": similar_condition & ~old_low,
        "展開": pace & ~old_low,
        "3-4歳上積み": development,
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


def add_result_view_features(df):
    df = df.copy()
    if "_focus_score" not in df.columns:
        df = add_focus_scores(df)
    if "_候補理由" not in df.columns:
        df["_候補理由"] = get_candidate_reason_texts(df)

    raw_score = pd.to_numeric(df.get("_focus_score"), errors="coerce")
    if raw_score.notna().any() and raw_score.max() != raw_score.min():
        practical_score = 60 + 40 * (raw_score - raw_score.min()) / (raw_score.max() - raw_score.min())
    else:
        practical_score = pd.Series(80.0, index=df.index)
    df["実戦点"] = practical_score.round(1)

    popularity = pd.to_numeric(df.get("人気"), errors="coerce")
    ai = pd.to_numeric(df.get("AI点"), errors="coerce")
    top_ai = ai.max()

    evaluations = []
    materials = []
    for idx, row in df.iterrows():
        reason_text = str(row.get("_候補理由", ""))
        reason_parts = [part for part in reason_text.split("、") if part]
        score = safe_num(row.get("実戦点"), 0)
        pop = popularity.loc[idx] if idx in popularity.index else None
        ai_value = ai.loc[idx] if idx in ai.index else None
        comment = str(row.get("コメント", ""))
        race_comment = str(row.get("展開コメント", ""))

        if not reason_parts:
            reason_parts = [part for part in [comment, race_comment] if part and part != "nan"][:1]

        if "能力上位" in reason_parts and pd.notna(ai_value) and pd.notna(top_ai) and ai_value >= top_ai - 8:
            evaluation = "軸候補"
        elif "3-4歳上積み" in reason_parts and pd.notna(pop) and pop <= 5:
            evaluation = "上積み注意"
        elif "同条件" in reason_parts or "準適性" in reason_parts:
            evaluation = "条件注目"
        elif "展開" in reason_parts:
            evaluation = "展開注目"
        elif "妙味穴" in reason_parts:
            evaluation = "紐穴"
        elif score >= 80:
            evaluation = "相手候補"
        else:
            evaluation = "様子見"

        if row.get("候補") != "✓" and score < 75:
            evaluation = "消し寄り"

        evaluations.append(evaluation)
        materials.append("、".join(reason_parts[:3]))

    df["実戦評価"] = evaluations
    df["馬券材料"] = materials
    return df


def sort_result_for_view(df):
    df = df.copy()
    if "AI順位" in df.columns:
        return df.sort_values(
            ["AI順位", "AI点"],
            ascending=[True, False],
        ).reset_index(drop=True)
    if "実戦点" not in df.columns:
        df = add_result_view_features(df)
    eval_order = df.get("実戦評価", pd.Series("", index=df.index)).map({
        "軸候補": 0,
        "上積み注意": 1,
        "条件注目": 2,
        "展開注目": 3,
        "相手候補": 4,
        "紐穴": 5,
        "様子見": 6,
        "消し寄り": 7,
    }).fillna(9)
    df["_実戦評価順"] = eval_order
    return df.sort_values(
        ["実戦点", "推奨点", "AI点", "_実戦評価順"],
        ascending=[False, False, False, True],
    ).reset_index(drop=True)


def apply_front_condition_bonus(df):
    if "脚質" not in df.columns or "推奨点" not in df.columns:
        return df

    df = df.copy()
    styles = df["脚質"].map(normalize_running_style)
    distance = pd.to_numeric(df.get("距離指数"), errors="coerce")
    course = pd.to_numeric(df.get("コース指数"), errors="coerce")
    popularity = pd.to_numeric(df.get("人気"), errors="coerce")
    star_high = pd.to_numeric(df.get("★最高"), errors="coerce")

    top_distance = distance.max()
    top_course = course.max()
    top_star = star_high.max()
    distance_high = distance.notna() & pd.notna(top_distance) & (distance >= top_distance - 6)
    course_high = course.notna() & pd.notna(top_course) & (course >= top_course - 6)
    star_fit = star_high.notna() & pd.notna(top_star) & (star_high >= top_star - 5)
    popular = popularity.fillna(99) <= 4

    front = styles.isin(["逃", "先"])
    escape = styles.eq("逃")
    leader = styles.eq("先")

    bonus = pd.Series(0.0, index=df.index)
    bonus += (front & popular & (distance_high | course_high)).astype(float) * 1.0
    bonus += (escape & popular & distance_high & course_high).astype(float) * 1.2
    bonus += (leader & popular & distance_high & course_high).astype(float) * 0.6
    bonus += (escape & popular & star_fit).astype(float) * 0.6
    bonus = bonus.clip(upper=2.8)

    df["_front_condition_bonus"] = bonus.round(1)
    df["推奨点"] = (pd.to_numeric(df["推奨点"], errors="coerce").fillna(df["AI点"]) + bonus).round(1)
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

    escape_count = counts["逃"]
    leader_count = counts["先"]
    early_count = escape_count + leader_count
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
    elif escape_count == 1 and early_count <= 3:
        pace = "単騎逃げ注意"
        tendency = "逃げ残り警戒"
        favored = ["逃", "先"]
    elif early_count <= 1:
        pace = "スロー想定"
        tendency = "前残り注意"
        favored = ["逃", "先"]
    elif escape_count >= 3:
        pace = "逃げ激化"
        tendency = "差し浮上"
        favored = ["差", "追"]
    elif escape_count >= 2 or early_count >= 5:
        pace = "速くなりそう"
        tendency = "差し浮上"
        favored = ["差", "追"]
    elif early_count >= 4:
        pace = "先行多め"
        tendency = "好位組の地力勝負"
        favored = ["先", "差"]
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
    if escape_count == 1 and early_count <= 3 and lone_escape_numbers:
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
    recent = pd.to_numeric(df.get("近3走最高"), errors="coerce")
    star_high = pd.to_numeric(df.get("★最高"), errors="coerce")
    distance = pd.to_numeric(df.get("距離指数"), errors="coerce")
    course = pd.to_numeric(df.get("コース指数"), errors="coerce")
    popularity = pd.to_numeric(df.get("人気"), errors="coerce")
    style_win_rate = df.get("脚質勝率", pd.Series("", index=df.index)).apply(format_percent_value)

    top_ai = ai.max()
    top_recent = recent.max()
    top_distance = distance.max()
    top_course = course.max()

    corner_values = []
    stretch_values = []
    comments = []

    for idx, row in df.iterrows():
        style = styles.loc[idx]
        ai_value = ai.loc[idx]
        recent_value = recent.loc[idx]
        star_value = star_high.loc[idx]
        distance_value = distance.loc[idx]
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
        distance_high = pd.notna(distance_value) and pd.notna(top_distance) and distance_value >= top_distance - 6
        course_high = pd.notna(course_value) and pd.notna(top_course) and course_value >= top_course - 5
        condition_fit = pd.notna(star_value)
        value_pop = pd.notna(pop_value) and pop_value >= 5
        popular_front = pd.notna(pop_value) and pop_value <= 4 and (distance_high or course_high)
        style_win_text = style_win_rate.loc[idx]
        horse_no = int(row.get("馬番", 0) or 0)
        is_pace_hole = horse_no in pace_holes

        if style == "逃":
            if front_flow and (win_level or popular_front):
                stretch = "押切候補"
                comment = "前残り有利"
            elif win_level:
                stretch = "押切候補"
                comment = "同型次第" if early_compete else "勝ち負け"
            elif counts.get("逃", 0) == 1 and (strong_level or condition_fit or popular_front):
                stretch = "逃げ粘る"
                comment = "単騎なら粘る"
            elif fast_flow and popular_front:
                stretch = "粘る"
                comment = "同型多いが能力で粘る"
            elif fast_flow and not (strong_level or condition_fit or course_high or distance_high):
                stretch = "粘り課題"
                comment = "流れ厳しい"
            elif strong_level or condition_fit or course_high or distance_high:
                stretch = "粘る"
                comment = "同型次第" if early_compete else "前で運べる"
            else:
                stretch = "粘り込み"
                comment = "流れひとつ"
        elif style == "先":
            if front_flow and (win_level or strong_level or popular_front):
                stretch = "好位粘る"
                comment = "前残り警戒"
            elif fast_flow and win_level:
                stretch = "好位差し"
                comment = "流れに乗る"
            elif fast_flow and (strong_level or recent_high or condition_fit or popular_front):
                stretch = "好位粘る"
                comment = "好位で粘る"
            elif win_level:
                stretch = "勝ち負け"
                comment = "好位から勝負"
            elif strong_level or condition_fit or course_high or distance_high:
                stretch = "粘る"
                comment = "前で運べる"
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


def apply_running_style_features(df):
    info = analyze_running_style(df)
    if not info.get("available"):
        df = df.copy()
        df["4角予想"] = ""
        df["直線評価"] = ""
        df["展開コメント"] = ""
        return df, info
    df = apply_front_condition_bonus(df)
    return add_corner_stretch_features(df, info), info


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
        course_value = safe_num(course.loc[idx], None)
        average_value = safe_num(average.loc[idx], None)
        distance_value = safe_num(distance.loc[idx], None)
        course_rank_value = safe_num(course_rank.loc[idx], None)
        has_course_record = course_value is not None and pd.notna(course_value) and float(course_value) > 0
        if has_course_record:
            score += 1.4
            condition_adjustment += 0.5
            add_unique(reasons, "コース実績")
            if (course_rank_value is not None and pd.notna(course_rank_value) and course_rank_value <= 3) or float(course_value) >= 100:
                score += 1.2
                condition_adjustment += 0.5
                add_unique(reasons, "コース指数上位")
            if average_value is not None and pd.notna(average_value) and float(average_value) >= 100 and float(course_value) >= 95:
                score += 2.0
                condition_adjustment += 1.0
                add_unique(reasons, "平均×コース")
            if (
                style in ("逃", "先")
                and average_value is not None
                and pd.notna(average_value)
                and float(average_value) >= 100
                and float(course_value) >= 100
            ):
                score += 3.2
                condition_adjustment += 1.2
                add_unique(reasons, "軸向き")
            elif (
                style in ("逃", "先")
                and average_value is not None
                and pd.notna(average_value)
                and float(average_value) >= 95
                and float(course_value) >= 95
            ):
                score += 1.6
                condition_adjustment += 0.8
                add_unique(reasons, "前受け実績")
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
    pace = str(info.get("ペース", ""))
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
        course_value = safe_num(course.loc[idx], None)
        average_value = safe_num(average.loc[idx], None)
        distance_value = safe_num(distance.loc[idx], None)
        course_rank_value = safe_num(course_rank.loc[idx], None)
        has_course_record = course_value is not None and pd.notna(course_value) and float(course_value) > 0
        if has_course_record:
            score += 1.4
            condition_adjustment += 0.5
            add_unique(reasons, "コース実績")
            if (course_rank_value is not None and pd.notna(course_rank_value) and course_rank_value <= 3) or float(course_value) >= 100:
                score += 1.2
                condition_adjustment += 0.5
                add_unique(reasons, "コース指数上位")
            if average_value is not None and pd.notna(average_value) and float(average_value) >= 100 and float(course_value) >= 95:
                score += 2.0
                condition_adjustment += 1.0
                add_unique(reasons, "平均×コース")
            if (
                style in ("逃", "先")
                and average_value is not None
                and pd.notna(average_value)
                and float(average_value) >= 100
                and float(course_value) >= 100
            ):
                score += 3.2
                condition_adjustment += 1.2
                add_unique(reasons, "軸向き")
            elif (
                style in ("逃", "先")
                and average_value is not None
                and pd.notna(average_value)
                and float(average_value) >= 95
                and float(course_value) >= 95
            ):
                score += 1.6
                condition_adjustment += 0.8
                add_unique(reasons, "前受け実績")
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
        if (
            pd.notna(high_rank.loc[idx])
            and high_rank.loc[idx] <= 3
            and pop_value is not None
            and pop_value >= 6
        ):
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
            if trend >= 2:
                adjustment += 1.5
            if relative_load_weight >= 2:
                adjustment += 1.5
        elif age == 4:
            adjustment += 2.2
            if trend >= 2:
                adjustment += 1.2
            if relative_load_weight >= 2:
                adjustment += 0.5
        elif age is not None and age >= 11:
            adjustment -= 3.5
        elif age is not None and age >= 9:
            adjustment -= 1.5
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

        stable = (
            avg_rank <= 5
            and row_range is not None
            and pd.notna(range_median)
            and row_range <= range_median
        )
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


def build_class_comparison_table(df, race_name):
    comparison = df.copy()
    race_name = str(race_name or "").translate(
        str.maketrans("０１２３４５６７８９", "0123456789")
    )
    is_one_win = "1勝クラス" in race_name or "500万" in race_name
    is_upper = any(
        label in race_name
        for label in ["2勝クラス", "3勝クラス", "オープン", "OP", "リステッド", "(L)", "重賞", "G1", "G2", "G3"]
    )

    comparison["_比較平均"] = pd.to_numeric(comparison.get("平均指数"), errors="coerce")
    comparison["_比較最高"] = pd.to_numeric(comparison.get("最高指数"), errors="coerce")
    comparison["_比較コース"] = pd.to_numeric(comparison.get("コース指数"), errors="coerce")
    comparison["_比較距離"] = pd.to_numeric(comparison.get("距離指数"), errors="coerce")
    comparison["_比較前走"] = pd.to_numeric(comparison.get("_last"), errors="coerce")
    comparison["_比較AI"] = pd.to_numeric(comparison.get("AI点"), errors="coerce")

    if is_one_win:
        ages = comparison.get("性齢", pd.Series("", index=comparison.index)).map(extract_age_from_sex_age)
        comparison = comparison[ages.eq(3)].copy()
        title = "3歳馬比較表"
        note = "3歳馬だけを抽出し、コース指数を最優先、近3走最高・前走指数を次の比較材料として並べています。"
        sort_columns = ["_比較コース", "_比較最高", "_比較前走", "_比較AI"]
        comparison_columns = [
            "比較順位", "馬番", "馬名", "性齢", "斤量", "騎手", "脚質", "人気", "単勝オッズ",
            "コース指数", "距離指数", "前走", "平均指数", "最高指数", "★最高指数", "AI点",
        ]
    else:
        title = "近走持続力比較表" if is_upper else "近走能力比較表"
        note = "近3走平均を最優先に、最高指数・コース指数・前走指数を比較して並べています。"
        sort_columns = ["_比較平均", "_比較最高", "_比較コース", "_比較前走", "_比較AI"]
        comparison_columns = [
            "比較順位", "馬番", "馬名", "性齢", "斤量", "騎手", "脚質", "人気", "単勝オッズ",
            "平均指数", "最高指数", "コース指数", "距離指数", "前走", "★最高指数", "AI点",
        ]

    comparison = comparison.sort_values(sort_columns, ascending=False, na_position="last").reset_index(drop=True)
    comparison.insert(0, "比較順位", range(1, len(comparison) + 1))
    return comparison, title, note, comparison_columns


def build_axis_suitability_table(df, race_name):
    axis_df = df.copy()
    comparison_df, _, _, _ = build_class_comparison_table(axis_df, race_name)
    comparison_rank = comparison_df.set_index("馬番")["比較順位"].to_dict()

    styles = axis_df.get("脚質", pd.Series("", index=axis_df.index)).map(normalize_running_style)
    escape_count = int(styles.eq("逃").sum())
    early_count = int(styles.isin(["逃", "先"]).sum())
    prev_values = axis_df.get("_prev_values", pd.Series([[]] * len(axis_df), index=axis_df.index))
    ranges = prev_values.map(
        lambda values: (
            max(valid) - min(valid)
            if isinstance(values, list)
            and len(valid := [value for value in values if value is not None]) >= 2
            else None
        )
    )
    range_median = pd.to_numeric(ranges, errors="coerce").median()
    course = pd.to_numeric(axis_df.get("コース指数"), errors="coerce")
    distance = pd.to_numeric(axis_df.get("距離指数"), errors="coerce")
    course_median = course.median()
    distance_median = distance.median()

    axis_scores = []
    reproducibilities = []
    dependencies = []
    evaluations = []
    bet_styles = []
    reasons = []
    comparison_ranks = []

    for idx, row in axis_df.iterrows():
        ai_rank = safe_num(row.get("AI順位"), len(axis_df))
        horse_no = row.get("馬番")
        compare_rank = comparison_rank.get(horse_no)
        values = row.get("_prev_values")
        valid = [value for value in values if value is not None] if isinstance(values, list) else []
        row_range = safe_num(ranges.loc[idx], None)
        latest = safe_num(row.get("_last"), None)
        average = safe_num(row.get("平均指数"), None)
        style = styles.loc[idx]
        score = 0.0
        reason_parts = []

        if ai_rank <= 1:
            score += 4.0
            reason_parts.append("能力最上位")
        elif ai_rank <= 3:
            score += 3.0
            reason_parts.append("能力上位")
        elif ai_rank <= 5:
            score += 2.0
        elif ai_rank <= 8:
            score += 1.0

        if compare_rank is not None:
            if compare_rank <= 1:
                score += 2.0
                reason_parts.append("比較表上位")
            elif compare_rank <= 3:
                score += 1.5
                reason_parts.append("比較表上位")
            elif compare_rank <= 5:
                score += 0.5

        if len(valid) == 3 and valid[0] <= valid[1] <= valid[2]:
            score += 1.5
            reason_parts.append("近走上昇")
        if row_range is not None and pd.notna(range_median) and row_range <= range_median:
            score += 1.0
            reason_parts.append("指数安定")
        if latest is not None and average is not None:
            if latest >= average - 3:
                score += 0.8
            elif latest <= average - 8:
                score -= 1.5
                reason_parts.append("前走低下")

        if pd.notna(course.loc[idx]) and pd.notna(course_median) and course.loc[idx] >= course_median:
            score += 0.5
        if pd.notna(distance.loc[idx]) and pd.notna(distance_median) and distance.loc[idx] >= distance_median:
            score += 0.5

        weight_change = safe_num(row.get("_load_weight_change"), None)
        if weight_change is not None and weight_change <= -1:
            score += 1.0
            reason_parts.append("斤量減")
        elif weight_change is not None and weight_change >= 2:
            score -= 0.5

        age = extract_age_from_sex_age(row.get("性齢"))
        if age is not None and age >= 11:
            score -= 3.5
            reason_parts.append("11歳以上")
        elif age is not None and age >= 9:
            score -= 1.5
            reason_parts.append("高齢注意")

        if style == "逃":
            if escape_count >= 2:
                dependency = "高"
                score -= 3.5
                reason_parts.append("逃げ競合")
            else:
                dependency = "中"
                score += 0.5
                reason_parts.append("単騎候補")
        elif style == "追":
            dependency = "高"
            score -= 2.0
            reason_parts.append("展開待ち")
        elif style == "先":
            dependency = "中" if early_count >= 5 else "低"
            score += -0.5 if early_count >= 5 else 0.5
            if early_count >= 5:
                reason_parts.append("先行競合")
        elif style == "差":
            dependency = "中"
            if early_count >= 5:
                score += 0.8
                reason_parts.append("差し展開")
        else:
            dependency = "中"

        if score >= 8:
            reproducibility = "高"
        elif score >= 5:
            reproducibility = "中"
        else:
            reproducibility = "低"

        if reproducibility == "高" and dependency != "高":
            evaluation = "中心にしやすい"
        elif reproducibility == "高":
            evaluation = "能力高いが展開確認"
        elif reproducibility == "中" and dependency == "低":
            evaluation = "相手軸で考えやすい"
        elif reproducibility == "中":
            evaluation = "相手候補"
        else:
            evaluation = "展開確認"

        odds = safe_num(row.get("単勝オッズ"), None)
        if evaluation == "中心にしやすい":
            if odds is not None and odds >= 8:
                bet_style = "単複 / ワイド・馬連の軸"
            elif odds is not None and odds >= 4:
                bet_style = "単勝・馬連 / ワイドの軸"
            else:
                bet_style = "ワイド・馬連の軸 / 単勝"
        elif evaluation in ("能力高いが展開確認", "相手軸で考えやすい"):
            bet_style = "ワイド・馬連の軸候補"
        elif evaluation == "相手候補":
            bet_style = "ワイド・馬連の相手"
        else:
            bet_style = "展開確認後"

        unique_reasons = []
        for part in reason_parts:
            if part not in unique_reasons:
                unique_reasons.append(part)

        axis_scores.append(round(score, 1))
        comparison_ranks.append(compare_rank)
        reproducibilities.append(reproducibility)
        dependencies.append(dependency)
        evaluations.append(evaluation)
        bet_styles.append(bet_style)
        reasons.append(" / ".join(unique_reasons[:4]))

    axis_df["_軸検討値"] = axis_scores
    axis_df["比較順位"] = comparison_ranks
    axis_df["再現性"] = reproducibilities
    axis_df["展開依存"] = dependencies
    axis_df["中心評価"] = evaluations
    axis_df["向く買い方"] = bet_styles
    axis_df["中心理由"] = reasons
    axis_df = axis_df.sort_values(
        ["_軸検討値", "AI点", "比較順位"],
        ascending=[False, False, True],
        na_position="last",
    ).reset_index(drop=True)
    axis_df.insert(0, "中心順位", range(1, len(axis_df) + 1))
    axis_columns = [
        "中心順位", "馬番", "馬名", "人気", "単勝オッズ", "AI順位", "比較順位",
        "再現性", "展開依存", "中心評価", "向く買い方", "中心理由",
    ]
    return axis_df.head(6), axis_columns


def print_axis_conclusion(axis_df):
    if axis_df.empty:
        print("中心馬を判定できませんでした。")
        return
    row = axis_df.iloc[0]
    horse_no = pd.to_numeric(row.get("馬番"), errors="coerce")
    no_text = str(int(horse_no)) if pd.notna(horse_no) else str(row.get("馬番", "")).strip()
    print(f"中心に考えやすい馬: {no_text} {row.get('馬名', '')}")
    print(
        f"{row.get('中心理由', '')}。再現性は{row.get('再現性', '')}、"
        f"展開依存は{row.get('展開依存', '')}。オッズを踏まえると{row.get('向く買い方', '')}。"
    )


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
    mask = (
        mark.ne("◎")
        & (
            pace_mark.eq("展")
            | value_gap.ge(3).fillna(False)
            | odds.ge(8).fillna(False)
            | class_shift.eq("クラス降級")
            | h2h.str.contains("先着|対戦○|対戦優勢", na=False)
            | reason.str.contains("妙味|展開向く|降級", na=False)
            | comment.str.contains("展開向く|末脚|好位", na=False)
            | distance_rank.le(3).fillna(False)
            | course_rank.le(3).fillna(False)
            | star_rank.le(3).fillna(False)
        )
    )
    pool = tmp[mask].copy()
    if pool.empty:
        return ""
    pool["_妙味差"] = value_gap.reindex(pool.index).fillna(0)
    pool["_妙味AI"] = ai.reindex(pool.index).fillna(0)
    pool["_妙味オッズ"] = odds.reindex(pool.index).fillna(0)
    selected = pool.sort_values(
        ["_妙味差", "_妙味AI", "_妙味オッズ"], ascending=[False, False, False]
    ).head(max(int(limit), 1))
    labels = []
    for _, row in selected.iterrows():
        no = pd.to_numeric(row.get("馬番"), errors="coerce")
        no_text = str(int(no)) if pd.notna(no) else str(row.get("馬番", "")).strip()
        labels.append(f"{no_text} {str(row.get('馬名', '')).strip()}".strip())
    return "オッズ妙味は" + "、".join(labels) + "。人気より展開・クラス・調教/評価/検討材料を優先して確認。"


def print_race_scenario(df):
    tmp = df.copy()
    if tmp.empty:
        print("【レース考察】")
        print("出走データがありません。")
        return
    tmp = prepare_jra_display_columns(tmp)

    def numeric(column):
        if column in tmp.columns:
            return pd.to_numeric(tmp[column], errors="coerce")
        return pd.Series(float("nan"), index=tmp.index, dtype="float64")

    def clean_text(value):
        return str(value or "").replace("\n", " ").strip()

    def norm_style(value):
        try:
            return normalize_running_style(value)
        except Exception:
            text = clean_text(value)
            if "逃" in text:
                return "逃"
            if "先" in text:
                return "先"
            if "追" in text:
                return "追"
            if "差" in text:
                return "差"
            return ""

    ai = numeric("AI点")
    total = numeric("総合評価点").fillna(numeric("_最終印点")).fillna(ai)
    final_order = pd.to_numeric(tmp.get("_最終印順", pd.Series(99, index=tmp.index)), errors="coerce").fillna(99)
    styles = tmp.get("脚質", pd.Series("", index=tmp.index)).map(norm_style)
    mark = tmp.get("最終印", pd.Series("", index=tmp.index)).fillna("").astype(str)

    def row_label(row):
        no = pd.to_numeric(row.get("馬番"), errors="coerce")
        no_text = str(int(no)) if pd.notna(no) else str(row.get("馬番", "")).strip()
        return f"{no_text} {str(row.get('馬名', '')).strip()}".strip()

    def sorted_pool(mask):
        pool = tmp[mask].copy()
        if pool.empty:
            return pool
        pool["_表示印順"] = final_order.reindex(pool.index).fillna(99)
        pool["_表示総合"] = total.reindex(pool.index).fillna(0)
        pool["_表示AI"] = ai.reindex(pool.index).fillna(0)
        return pool.sort_values(["_表示印順", "_表示総合", "_表示AI"], ascending=[True, False, False])

    def names_text(pool, limit=None):
        if pool.empty:
            return "該当馬なし"
        if limit:
            pool = pool.head(limit)
        return "、".join(row_label(row) for _, row in pool.iterrows())

    escape_pool = sorted_pool(styles.eq("逃"))
    leader_pool = sorted_pool(styles.eq("先"))
    closer_pool = sorted_pool(styles.eq("差"))
    trailer_pool = sorted_pool(styles.eq("追"))
    center_pool = sorted_pool(mark.isin(["◎", "○"]))
    partner_pool = sorted_pool(mark.isin(["▲", "△"]))
    star_pool = sorted_pool(mark.eq("✓"))

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
            print(f"展開のポイント：{row_label(lead)}が楽に運べれば粘り込みまで。")
        else:
            print(f"展開のポイント：{row_label(lead)}は展開を作れそうだが、能力的に粘り込みには展開の助けが必要。")
    elif not leader_pool.empty:
        print(f"展開のポイント：{row_label(leader_pool.iloc[0])}が押し出される形。隊列が落ち着けば先行勢に余地。")
    else:
        print("展開のポイント：明確な逃げ馬が少なく、序盤の位置取りが評価を左右。")

    def reason_text(row, role):
        parts = []
        if role == "center":
            parts.append(f"AI点{format_number_for_display(row.get('AI点'))}")
            parts.append(f"総合評価{format_number_for_display(row.get('総合評価点') or row.get('_最終印点'))}")
        material = clean_text(row.get("評価根拠") or row.get("調教/評価/検討材料") or "")
        class_shift = clean_text(row.get("クラス変動"))
        reason = clean_text(row.get("印理由"))
        if class_shift:
            parts.append(class_shift)
        for keyword in ["能力上位", "最高指数", "高指数", "コース実績", "距離実績", "対戦先着", "展開向く", "調教上位", "好気配", "クラス降級"]:
            if keyword in material or keyword in reason:
                parts.append(keyword)
        if clean_text(row.get("展開印")) == "展":
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
            print(f"{row.get('最終印', '')} {row_label(row)}：{reason_text(row, 'center')}")

    print("")
    print("【相手候補】")
    if partner_pool.empty:
        print("相手候補：該当馬なし")
    else:
        for _, row in partner_pool.head(3).iterrows():
            print(f"{row.get('最終印', '')} {row_label(row)}：{reason_text(row, 'partner')}")

    print("")
    print("【穴候補】")
    if star_pool.empty:
        print("穴候補：該当馬なし")
    else:
        row = star_pool.iloc[0]
        material = clean_text(row.get("評価根拠") or row.get("調教/評価/検討材料"))
        peak_text = "ピーク指数型" if ("最高指数" in material or "ピーク" in clean_text(row.get("印理由"))) else "妙味候補"
        print(f"✓ {row_label(row)}：{peak_text}。{reason_text(row, 'hole')}")

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
        for part in [part for part in str(row.get("コメント", "")).split("、") if part]:
            if part not in materials:
                materials.append(part)

        age = extract_age_from_sex_age(row.get("性齢"))
        reason_map = {
            "同条件": "同条件実績",
            "準適性": "他場実績注意",
            "展開": "展開利",
            "妙味穴": "人気薄",
            "3-4歳上積み": f"{age}歳上積み" if age in (3, 4) else "若駒上積み",
        }
        for reason in str(row.get("_候補理由", "")).split("、"):
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
        details = []
        style = normalize_running_style(row.get("脚質"))
        if style:
            details.append(style)
        row_relative_weight = safe_num(row.get("_relative_load_weight"), 0)
        if extract_age_from_sex_age(row.get("性齢")) == 3 and row_relative_weight >= 2:
            details.append("軽量3歳")
        elif row_relative_weight >= 2.5:
            details.append("軽斤量")
        rate = format_percent_value(row.get("脚質勝率"))
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
        elif escape_count == 1 and early_count <= 3:
            row = tmp[styles.eq("逃")].sort_values(["_insight_pace_score", "推奨点"], ascending=[False, False]).iloc[0]
            lines.append(f"展開：前は手薄で{pace_horse_label(row)}の単騎逃げが焦点。楽に運べれば前残りまで。")
            used_horses.add(horse_no(row))
        elif early_count <= 1:
            front_pool = tmp[styles.isin(["逃", "先"])].sort_values(["_insight_pace_score", "推奨点"], ascending=[False, False]).head(2)
            if not front_pool.empty:
                names = "、".join(pace_horse_label(row) for _, row in front_pool.iterrows())
                lines.append(f"展開：前へ行く馬が少なくスロー寄り。{names}の前残りを警戒。")
                mark_used(front_pool)
        else:
            pace_pool = unused(tmp[styles.isin(["先", "差"])]).sort_values(
                ["_insight_pace_score", "推奨点", "AI点"],
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
        & tmp["_候補理由"].str.contains("妙味穴|展開|同条件", na=False)
    ])
    if not value_pool.empty:
        selected = value_pool.assign(_value_score=value_score.loc[value_pool.index]).sort_values(
            ["_value_score", "_insight_course_score", "AI点"],
            ascending=[False, False, False],
        ).head(2)
        descriptions = [f"{horse_label(row)}は{material_text(row)}" for _, row in selected.iterrows()]
        lines.append(f"妙味注目：{'、'.join(descriptions)}。人気より条件と展開を重視。")
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

    escape_count = int((tmp["_脚質正規化"] == "逃").sum())
    leader_count = int((tmp["_脚質正規化"] == "先").sum())
    if escape_count + leader_count >= 4:
        print("隊列メモ：前が忙しくなれば差し浮上")
    elif escape_count <= 1:
        escape = horse_list(tmp["_脚質正規化"].eq("逃"))
        if escape != "なし":
            print(f"隊列メモ：{escape}の単騎逃げに注意")

    focus_mask = tmp["直線評価"].isin(["勝ち負け", "押切候補", "差切警戒", "末脚勝負", "差し浮上", "差し届く"])
    focus = (
        tmp[focus_mask]
        .sort_values(["AI点", "推奨点"], ascending=[False, False])
        .head(4)
    )
    if not focus.empty:
        notes = [f"{circled_number(row['馬番'])}{row['直線評価']}" for _, row in focus.iterrows()]
        print(f"直線注目：{' / '.join(notes)}")


def print_betting_plan(df, budget=1000):
    groups = get_betting_groups(df)
    shape = analyze_race_shape(df)
    stars = shape["波乱度"].count("★")
    main_axis = groups["main_axis"][:1]
    value_axis = groups["value_axis"][:1]
    value = groups["value"][:2]
    main_partners = groups["main_partners"][:3]
    reserves = groups["reserves"][:3]
    budget = int(budget or 1000)
    budget = max(100, budget)

    def make_wide(axis_group, partner_group, max_tickets=None):
        tickets = []
        for a in axis_group:
            for b in partner_group:
                if a == b:
                    continue
                ticket = (int(a), int(b))
                key = tuple(sorted(ticket))
                if key not in [tuple(sorted(t)) for t in tickets]:
                    tickets.append(ticket)
        return tickets[:max_tickets] if max_tickets else tickets

    def make_box_pairs(horses, max_tickets=None):
        tickets = []
        horses = unique_horse_numbers(horses)
        for i, a in enumerate(horses):
            for b in horses[i + 1:]:
                ticket = (int(a), int(b))
                key = tuple(sorted(ticket))
                if key not in [tuple(sorted(t)) for t in tickets]:
                    tickets.append(ticket)
        return tickets[:max_tickets] if max_tickets else tickets

    def make_trio(first_group, second_group, third_group, max_tickets=None):
        tickets = []
        for a in first_group:
            for b in second_group:
                for c in third_group:
                    combo = tuple(sorted({int(a), int(b), int(c)}))
                    if len(combo) == 3 and combo not in tickets:
                        tickets.append(combo)
        return tickets[:max_tickets] if max_tickets else tickets

    def make_trifecta(first_group, second_group, third_group, max_tickets=None):
        tickets = []
        for a in first_group:
            for b in second_group:
                for c in third_group:
                    ticket = (int(a), int(b), int(c))
                    if len(set(ticket)) == 3 and ticket not in tickets:
                        tickets.append(ticket)
        return tickets[:max_tickets] if max_tickets else tickets

    def format_ticket(ticket, separator):
        return separator.join(circled_number(x) for x in ticket)

    total = 0
    printed = False

    def add_block(label, tickets, yen_each, separator="-"):
        nonlocal total, printed
        affordable = max(0, (budget - total) // yen_each)
        selected = tickets[:affordable]
        if not selected:
            return
        subtotal = len(selected) * yen_each
        total += subtotal
        printed = True
        print(f"{label}：{yen_each}円 × {len(selected)}点 = {subtotal}円")
        print(" / ".join(format_ticket(ticket, separator) for ticket in selected))
        print("")

    def add_formation_block(label, first_group, second_group, third_group, tickets, yen_each):
        nonlocal total, printed
        affordable = max(0, (budget - total) // yen_each)
        selected = tickets[:affordable]
        if not selected:
            return
        subtotal = len(selected) * yen_each
        total += subtotal
        printed = True
        print(f"{label}：{yen_each}円 × {len(selected)}点 = {subtotal}円")
        print(f"{format_horses(first_group)} - {format_horses(second_group)} - {format_horses(third_group)}")
        print("")

    print("【買い目提案】")
    print(f"予算：{budget}円")
    if not main_axis and not value_axis:
        candidates = unique_horse_numbers(main_partners, groups["reserves"], limit=5)
        print("見送り寄り：明確な軸なし")
        if len(candidates) >= 2:
            add_block("ワイド候補", make_wide(candidates[:1], candidates[1:], max_tickets=budget // 200), 200)
        if not printed:
            print("買い目なし")
        return

    if stars >= 4:
        print("配分：波乱型（ワイド数点＋3連複）")
        value_main = df[df["役割"].eq("相手有力")].sort_values(
            ["妙味スコア", "単勝オッズ", "推奨点"],
            ascending=[False, False, False],
        )["馬番"].tolist()[:1]
        wide_candidates = unique_horse_numbers(value_axis, value, value_main, main_axis, main_partners, limit=3)
        add_block("ワイド 厚め", make_box_pairs(wide_candidates, max_tickets=3), 100)
        trio_first = unique_horse_numbers(value_axis, main_axis, limit=1) or wide_candidates[:1]
        trio_second = unique_horse_numbers(value, main_partners, main_axis, limit=2)
        trio_third = unique_horse_numbers(main_partners, value, trio_first, reserves, limit=6)
        add_formation_block(
            "3連複 フォーメーション",
            trio_first,
            trio_second,
            trio_third,
            make_trio(trio_first, trio_second, trio_third, max_tickets=12),
            100,
        )
    elif stars == 3:
        print("配分：バランス型（ワイド保険＋3連複）")
        wide_axis = value_axis or main_axis
        wide_partners = [x for x in unique_horse_numbers(main_axis, main_partners, value, limit=4) if x not in wide_axis]
        add_block("ワイド 保険", make_wide(wide_axis[:1], wide_partners, max_tickets=2), 100)
        trio_first = unique_horse_numbers(main_axis, value_axis, limit=2)
        trio_second = unique_horse_numbers(main_partners, value, main_axis, limit=5)
        trio_third = unique_horse_numbers(main_partners, value, reserves, main_axis, value_axis, limit=8)
        add_formation_block(
            "3連複 本線＋穴",
            trio_first,
            trio_second,
            trio_third,
            make_trio(trio_first, trio_second, trio_third, max_tickets=10),
            100,
        )
    else:
        print("配分：堅め型（本軸中心）")
        axis = main_axis or main_partners[:1]
        trio_partners = unique_horse_numbers(main_partners, value, reserves, limit=5)
        add_block("3連複 本線", make_trio(axis, trio_partners, trio_partners, max_tickets=6), 100)
        wide_partners = [x for x in unique_horse_numbers(main_partners, value, reserves, limit=3) if x not in axis]
        add_block("ワイド 保険", make_wide(axis, wide_partners, max_tickets=2), 200)
        add_block("3連単 少点", make_trifecta(axis, unique_horse_numbers(main_partners, value, limit=4), unique_horse_numbers(main_partners, value, reserves, limit=6), max_tickets=4), 100, separator="→")

    if not printed:
        print("買い目なし")
    else:
        print(f"推奨合計：{total}円 / 残り：{budget - total}円")


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
    "比較順位",
    "中心順位",
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


def make_result_cell_styles(df):
    styles = pd.DataFrame("", index=df.index, columns=df.columns)

    if "候補" in df.columns:
        styles.loc[df["候補"].astype(str).eq("✓"), "候補"] = (
            "font-weight: 800; color: #047857; background-color: #ecfdf5;"
        )

    if "実戦評価" in df.columns:
        eval_styles = {
            "軸候補": "font-weight: 800; color: #7f1d1d; background-color: #fee2e2;",
            "上積み注意": "font-weight: 800; color: #075985; background-color: #e0f2fe;",
            "条件注目": "font-weight: 800; color: #92400e; background-color: #fef3c7;",
            "展開注目": "font-weight: 800; color: #065f46; background-color: #d1fae5;",
            "相手候補": "font-weight: 700; color: #1f2937; background-color: #f3f4f6;",
            "紐穴": "font-weight: 800; color: #9d174d; background-color: #fce7f3;",
        }
        for label, css in eval_styles.items():
            styles.loc[df["実戦評価"].astype(str).eq(label), "実戦評価"] = css

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


def build_matchup_table(df):
    if "対戦" not in df.columns:
        return pd.DataFrame(columns=["最終印", "馬番", "馬名", "対戦評価", "対戦"])
    source = df[df["対戦"].fillna("").astype(str).str.strip().ne("")].copy()
    if source.empty:
        return pd.DataFrame(columns=["最終印", "馬番", "馬名", "対戦評価", "対戦"])
    if "対戦評価" not in source.columns:
        source["対戦評価"] = source.get("_h2h_label", pd.Series("", index=source.index)).fillna("")
    if "_対戦順" not in source.columns:
        mark_order = {"◎": 0, "○": 1, "▲": 2, "△": 3, "✓": 5, "☆": 5}
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



def normalize_final_class_shift(value):
    text = str(value or "").strip()
    if text in ("クラス降級", "相手弱化"):
        return "クラス降級"
    if text in ("クラス昇級", "相手強化"):
        return "クラス昇級"
    if text in ("同級", "同級安定", "同級付近"):
        return "同級"
    return text


def final_mark_class_shift(row):
    shift = normalize_final_class_shift(row.get("_class_shift") or row.get("クラス変動") or "")
    if shift:
        return shift
    current_rank = safe_num(row.get("_current_class_rank"), None)
    previous_rank = safe_num(row.get("_previous_class_rank"), None)
    if current_rank is None or previous_rank is None:
        return ""
    diff = current_rank - previous_rank
    if diff >= 8:
        return "クラス昇級"
    if diff <= -8:
        return "クラス降級"
    return "同級"

def final_mark_class_basis(row):
    current_label = str(row.get("_current_class_label") or "").strip()
    previous_label = str(row.get("_previous_class_label") or "").strip()
    best_label = str(row.get("_best_past_class_label") or "").strip()
    shift = final_mark_class_shift(row)
    parts = []
    parts.append(f"今回{current_label}" if current_label else "今回クラス不明")
    if previous_label:
        parts.append(f"前走{previous_label}")
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
    shift = final_mark_class_shift(row)
    score = 0.0
    reasons = []

    if shift == "クラス降級":
        score += 4.0
    elif shift == "クラス昇級":
        score -= 4.0
    elif shift == "同級":
        score += 1.0

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
    AI rank, index ranks, market value, venue, newspaper, and oikiri material
    into the final mark score.
    """
    df = df.copy()
    df = df.drop(columns=["最終印", "展開印", "印理由", "クラス根拠", "_最終印点", "_最終印順", "_穴評価点", "補正AI点", "_course_hole"], errors="ignore")
    if df.empty:
        df["最終印"] = ""
        df["展開印"] = ""
        df["印理由"] = ""
        df["クラス変動"] = ""
        df["クラス根拠"] = ""
        df["補正AI点"] = ""
        df["_course_hole"] = False
        return df

    df["クラス変動"] = df.apply(final_mark_class_shift, axis=1)

    def numeric(column, default=None):
        if column in df.columns:
            return pd.to_numeric(df[column], errors="coerce")
        return pd.Series(default, index=df.index, dtype="float64")

    ai = numeric("AI点", 0).fillna(0)
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
    newspaper_score = numeric("_新聞材料点", 0).fillna(0)
    oikiri_score = numeric("_追切材料点", 0).fillna(0)
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
            (course_rank, "コース指数上位", 2.3, 1.3),
            (star_rank, "同条件実績", 1.8, 1.0),
        ]:
            rank_item = rank_series.loc[idx]
            if pd.notna(rank_item) and rank_item <= 1:
                score += first_bonus
                add_unique(reasons, label)
            elif pd.notna(rank_item) and rank_item <= 3:
                score += top3_bonus
                add_unique(reasons, label)

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
        h2h_positive = h2h_value > 0 or h2h_label in ("対戦◎", "対戦○") or "先着" in h2h_latest
        h2h_strong = h2h_value >= 2 or h2h_label == "対戦◎"
        h2h_negative = h2h_value < 0 or h2h_label == "対戦△" or "敗戦" in h2h_latest
        if h2h_strong:
            add_unique(reasons, "対戦◎")
        elif h2h_positive:
            add_unique(reasons, "対戦先着")
        elif h2h_negative:
            add_unique(reasons, "対戦劣勢")

        style = styles.loc[idx]
        course_value = safe_num(course.loc[idx], None)
        average_value = safe_num(average.loc[idx], None)
        distance_value = safe_num(distance.loc[idx], None)
        course_rank_value = safe_num(course_rank.loc[idx], None)
        average_rank_value = safe_num(average_rank.loc[idx], None)
        star_value = safe_num(star_high.loc[idx], None)
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
        has_course_record = course_value is not None and pd.notna(course_value) and float(course_value) > 0
        if has_course_record:
            score += 1.6
            condition_adjustment += 0.7
            add_unique(reasons, "コース実績")
            if (course_rank_value is not None and pd.notna(course_rank_value) and course_rank_value <= 3) or float(course_value) >= 100:
                score += 1.5
                condition_adjustment += 0.8
                add_unique(reasons, "コース指数上位")
            if average_value is not None and pd.notna(average_value) and float(average_value) >= 100 and float(course_value) >= 95:
                score += 2.0
                condition_adjustment += 1.0
                add_unique(reasons, "平均×コース")
            if (
                style in ("逃", "先")
                and average_value is not None
                and pd.notna(average_value)
                and float(average_value) >= 100
                and float(course_value) >= 100
            ):
                score += 3.2
                condition_adjustment += 1.2
                add_unique(reasons, "軸向き")
            elif (
                style in ("逃", "先")
                and average_value is not None
                and pd.notna(average_value)
                and float(average_value) >= 95
                and float(course_value) >= 95
            ):
                score += 1.6
                condition_adjustment += 0.8
                add_unique(reasons, "前受け実績")
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

        newspaper_value = newspaper_score.loc[idx]
        newspaper_value = float(newspaper_value) if pd.notna(newspaper_value) else 0.0
        if newspaper_value:
            # 競馬新聞は最終印の補助。指数・クラス・展開を主役にするため加点は控えめ。
            newspaper_adjustment = min(max(newspaper_value, -1.0), 2.0) * 0.25
            score += newspaper_adjustment
            condition_adjustment += newspaper_adjustment
        newspaper_material = str(row.get("新聞材料") or "")
        if newspaper_material:
            if "小倉向き" in newspaper_material:
                score += 0.8
                condition_adjustment += 0.4
                add_unique(reasons, "小倉向き")
            if "先行上がり最速" in newspaper_material or "先行上がり優秀" in newspaper_material:
                score += 0.7
                condition_adjustment += 0.3
                add_unique(reasons, "先行上がり優秀")
            elif "末脚最速" in newspaper_material or "末脚上位" in newspaper_material:
                score += 0.35
                condition_adjustment += 0.15
                add_unique(reasons, "末脚上位")
            elif "上がり最速も展開待ち" in newspaper_material or "上がり上位も展開待ち" in newspaper_material:
                add_unique(reasons, "末脚展開待ち")
            for material in re.split(r"\s*/\s*", newspaper_material):
                if material in ("調教A", "調教S", "調教文句なし"):
                    add_unique(reasons, material)
                    break

        oikiri_value = oikiri_score.loc[idx]
        oikiri_value = float(oikiri_value) if pd.notna(oikiri_value) else 0.0
        if oikiri_value:
            # 調教タイムは状態面の補助。併せ遅れなどの明確な不安は少し強めに反映する。
            oikiri_adjustment = min(max(oikiri_value, -3.0), 3.0) * 0.6
            score += oikiri_adjustment
            condition_adjustment += oikiri_adjustment
        oikiri_material = str(row.get("追切材料") or "")
        if oikiri_material:
            for material in oikiri_material.split(" / "):
                if material in ("追切S", "追切A", "文句なし", "一番時計", "格下遅れ", "併せ遅れ"):
                    add_unique(reasons, material)
                    break

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
            layoff_penalty = 3.0
            if rank_value <= 1:
                layoff_penalty += 2.0
            if max(newspaper_value, oikiri_value) >= 2.0:
                layoff_penalty -= 0.5
            if oikiri_value <= -1.0:
                layoff_penalty += 0.8
            layoff_adjustment = max(2.5, layoff_penalty)
            score -= layoff_adjustment
            condition_adjustment -= layoff_adjustment
            if "休み明け割引" in reasons:
                reasons.remove("休み明け割引")
            reasons.insert(0, "休み明け割引")

        class_basis = final_mark_class_basis(row)
        if not reasons:
            add_unique(reasons, "総合評価上位")
        scores.append(round(score, 2))
        reason_texts.append(" / ".join(reasons[:5]))
        class_basis_texts.append(class_basis)
        adjusted_ai_scores.append(round(min(max(base_ai_value + condition_adjustment, 0.0), 100.0), 1))

        hole_score = score
        if value_gap >= 3:
            hole_score += min(value_gap, 8) * 0.8
        if odds_value is not None and odds_value >= 8:
            hole_score += 1.4
        if any(reason in reasons for reason in ("展開材料", "人気以上に評価", "コース実績", "平均×コース", "軸向き", "前受け実績", "対戦◎", "対戦先着")) or str(row.get("クラス変動") or "") == "クラス降級":
            hole_score += 1.4
        if h2h_strong:
            hole_score += 3.0
        elif h2h_positive:
            hole_score += 2.2
        if max(newspaper_value, oikiri_value) >= 2.0:
            hole_score += 0.2
        if course_hole:
            hole_score += 6.0
        course_hole_flags.append(bool(course_hole))
        hole_scores.append(round(hole_score, 2))

    df["_最終印点"] = scores
    df["_穴評価点"] = hole_scores
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

        class_down = hole_shift.isin(["クラス降級", "相手弱化"])
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
        h2h_positive_mask = (
            hole_h2h_score.gt(0)
            | hole_h2h_label.isin(["対戦◎", "対戦○"])
            | hole_h2h_latest.str.contains("先着", na=False)
        )
        h2h_strong_mask = hole_h2h_score.ge(2) | hole_h2h_label.eq("対戦◎")
        h2h_bad_mask = hole_h2h_score.lt(0) | hole_h2h_latest.str.contains("敗戦", na=False)
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
        hole_pool["_☆専用点"] += relative_index_hole_mask.astype(float) * 7.0
        hole_pool["_☆専用点"] += h2h_strong_mask.astype(float) * 7.0
        hole_pool["_☆専用点"] += course_hole_mask.astype(float) * 6.0
        hole_pool["_☆専用点"] += h2h_positive_mask.astype(float) * 5.0
        hole_pool["_☆専用点"] -= h2h_bad_mask.astype(float) * 3.0
        hole_pool["_☆専用点"] += weight_ok.astype(float) * 1.5
        hole_pool["_☆専用点"] += age_ok.astype(float) * 1.0
        hole_pool["_☆専用点"] -= hole_ai_rank.le(3).astype(float) * 10.0
        hole_pool["_☆専用点"] -= hole_weight_change.gt(0).astype(float) * 3.0
        hole_pool["_☆専用点"] -= hole_age.ge(9).fillna(False).astype(float) * 4.0

        special_holes = hole_pool[special_hole_mask | h2h_hole_mask | (course_hole_mask & layoff_ok) | relative_index_hole_mask]
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

    delta_slots = 0
    for idx in remaining:
        if idx == star_idx or idx in assigned or delta_slots <= 0:
            continue
        df.at[idx, "最終印"] = "△"
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

    ai = numeric("AI点", 0).fillna(0)
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
        base_ai_value = float(ai.loc[idx]) if pd.notna(ai.loc[idx]) else 0.0
        reasons = []

        # ===== Ver1.0 legacy notes =====
        # score += AI順位加点
        # score += 平均/最高/距離/コース/★最高の順位加点
        # score += ★最高×条件、近3走最高×条件、距離コース両方、条件軸材料
        # score += 人気以上、人気先行、オッズ妙味
        # Ver2.0では上記を総合評価から除外し、AI点に含まれる能力評価として扱う。

        class_shift = class_shift_for(row)
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
        pace_candidates = df[df["馬番"].astype(int).isin(pace_numbers)].copy()
    else:
        pace_candidates = df[df["印理由"].astype(str).str.contains("展開向く", na=False)].copy()
    if not pace_candidates.empty:
        pace_idx = pace_candidates.sort_values(["_最終印点", "AI点"], ascending=[False, False]).index[0]
        df.at[pace_idx, "展開印"] = "展"

    df["最終印"] = ""
    df["_最終印順"] = pd.NA
    ordered_indices = df.sort_values(["_最終印点", "AI点"], ascending=[False, False]).index.tolist()
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
    return {"△1": "△①", "△2": "△②"}.get(str(role or ""), str(role or ""))


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
        mark_value = str(row.get("_single_odds_mark") or "").strip()
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
            "name": str(row.get("馬名") or "").strip(),
            "odds": _single_odds_from_row(row),
        })
    return horses


def _single_odds_horse_text(horse, include_name=True):
    if not horse:
        return ""
    role = _single_odds_role_display(horse.get("role"))
    odds = _single_odds_format(horse.get("odds"))
    no = str(horse.get("no") or "").strip()
    name = str(horse.get("name") or "").strip()
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
    role = str(role or "")
    if role.startswith("○"):
        return "○"
    if role.startswith("△"):
        return role
    return role


def _ticket_role_text(role):
    role = str(role or "")
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
    return "中央" if str(race_type).lower() == "jra" else "地方"


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
    mark = str(row.get("最終印", "") or "").strip()
    return mark if mark in {"◎", "○", "▲", "△", "✓", "☆"} else ""


def _horse_market_warning(row):
    ai_rank = _horse_rank_value(row, "_馬_AI順位")
    market_rank = _horse_rank_value(row, "_馬_市場順位")
    return ai_rank < 999 and market_rank < 999 and (market_rank - ai_rank) >= 3


def _horse_type_group(horse_type):
    text = str(horse_type or "")
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
    return {"axis": 0, "stable": 1, "hole": 2, "opponent": 3, "fade": 9}.get(_horse_type_group(horse_type), 5)


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

    work["_馬_印"] = work.apply(_horse_mark_value, axis=1)
    work["_馬_市場警戒"] = work.apply(_horse_market_warning, axis=1)
    work["_馬タイプ"] = work.apply(lambda row: _horse_classify(row, race_type), axis=1)
    work["_馬コメント"] = work.apply(lambda row: " / ".join(_horse_comment_items(row, row.get("_馬タイプ"), race_type)[:2]), axis=1)
    work["_馬タイプ優先"] = work["_馬タイプ"].apply(_horse_type_priority)
    return work


def _horse_classify(row, race_type="nar"):
    race_type = str(race_type or "nar").lower()
    mark = _horse_mark_value(row)
    odds = _horse_numeric_value(row, "_馬_単勝")
    ai_rank = _horse_rank_value(row, "_馬_AI順位")
    total_rank = _horse_rank_value(row, "_馬_総合順位")
    market_rank = _horse_rank_value(row, "_馬_市場順位")
    market_warning = _horse_market_warning(row)

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
        no = str(row.get("_馬_馬番") or "").strip()
        if no:
            mapping[no] = {
                "type": row.get("_馬タイプ", ""),
                "comment": row.get("_馬コメント", ""),
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
            "no": str(row.get("_馬_馬番") or ""),
            "name": str(row.get("_馬_馬名") or ""),
            "odds": row.get("_馬_単勝"),
            "type": row.get("_馬タイプ", ""),
            "comment": row.get("_馬コメント", ""),
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
            "馬番": row.get("_馬_馬番", ""),
            "印": row.get("_馬_印", "") or "無印",
            "馬名": row.get("_馬_馬名", ""),
            "単勝": _single_odds_format(row.get("_馬_単勝")),
            "AI順位": _horse_format_rank(row.get("_馬_AI順位")),
            "総合順位": _horse_format_rank(row.get("_馬_総合順位")),
            "市場順位": _horse_format_rank(row.get("_馬_市場順位")),
            "馬タイプ": row.get("_馬タイプ", ""),
            "一言コメント": row.get("_馬コメント", ""),
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
                "馬番": row.get("_馬_馬番", ""),
                "印": row.get("_馬_印", "") or "無印",
                "馬名": row.get("_馬_馬名", ""),
                "単勝": _single_odds_format(row.get("_馬_単勝")),
                "AI順位": _horse_format_rank(row.get("_馬_AI順位")),
                "総合順位": _horse_format_rank(row.get("_馬_総合順位")),
                "市場順位": _horse_format_rank(row.get("_馬_市場順位")),
                "馬タイプ": row.get("_馬タイプ", ""),
                "一言コメント": row.get("_馬コメント", ""),
            })
        if all_rows:
            _ticket_display_collapsible_table("全馬タイプを表示", pd.DataFrame(all_rows))
    except Exception:
        pass


def _ticket_attach_horse_type(horse, horse_type_map):
    no = str((horse or {}).get("no") or "").strip()
    data = horse_type_map.get(no, {})
    horse["type"] = data.get("type", horse.get("type", ""))
    horse["type_comment"] = data.get("comment", horse.get("comment", ""))
    return horse


def _ticket_type_pair_text(left_type, right_type):
    left = str(left_type or "-")
    right = str(right_type or "-")
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
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except Exception:
        pass
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
    try:
        if value is None or pd.isna(value):
            return ""
    except Exception:
        if value is None:
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


def _ver30_bool_value(value):
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except Exception:
        pass
    if isinstance(value, str):
        return value.strip().lower() in {"true", "1", "yes", "y"}
    return bool(value)


def _ver30_load_weight_detail(row):
    current = _ver30_float_value(row.get("_current_load_weight"))
    if current is None:
        current = _ver30_float_value(row.get("斤量"))
    if current is None:
        return "データなし"

    previous = _ver30_float_value(row.get("_previous_load_weight"))
    change = _ver30_float_value(row.get("_load_weight_change"))
    if change is None and previous is not None:
        change = current - previous

    current_text = _ver30_format_kg(current)
    if previous is None or change is None:
        return f"{current_text}（前走データなし）"
    return f"{current_text}（前走比{_ver30_signed_kg(change)}）"


def _ver30_jockey_detail(row):
    current = _ver30_text_value(row.get("_current_jockey")) or _ver30_text_value(row.get("騎手"))
    previous = _ver30_text_value(row.get("_previous_jockey"))
    if not current:
        return "データなし"
    if not previous:
        return f"{current}【前走データなし】"
    if _ver30_bool_value(row.get("_jockey_changed")):
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
    return {"◎": 0, "○": 1, "▲": 2, "△": 3, "✓": 5, "☆": 5, "": 9}.get(str(mark or ""), 9)


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
        type_text = str(row.get("_馬タイプ", ""))
        type_group = _horse_type_group(type_text)
        mark_text = str(row.get("_馬_印", ""))

        ability_level = max(_ver30_rank_level(ai_r, 1), _ver30_rank_level(total_r, 1))
        if type_group == "fade":
            ability_level = min(ability_level, 2)
        ability.append(_ver30_star(ability_level))

        stability_level = _ver30_rank_level(market_r, 1)
        if ai_r is not None and market_r is not None and abs(ai_r - market_r) <= 1:
            stability_level = min(5, stability_level + 1)
        if "休み明け" in str(row.get("レース間隔", "")):
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
        value = str(row.get(key, "") or "")
        if value and value != "nan":
            text_parts.append(value)
    text = " / ".join(text_parts)
    notes = []
    if "クラス降級" in text or str(row.get("クラス変動", "")) == "クラス降級":
        notes.append("クラス降級")
    if "距離" in text or _ver30_num(row, "距離指数") is not None:
        notes.append("距離適性")
    if "コース" in text or _ver30_num(row, "コース指数") is not None:
        notes.append("コース実績")
    if any(word in text for word in ["復調", "上向", "好気配", "動き良", "良化"]):
        notes.append("復調気配")
    if any(word in text for word in ["高指数", "最高指数", "能力上位", "指数上位"]):
        notes.append("高指数")
    if "展開向く" in text or str(row.get("展開印", "")):
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
            "印": _ver30_text_value(row.get("display_mark", "")) or "無印",
            "表示印": _ver30_text_value(row.get("display_mark", "")),
            "馬名": row.get("_馬_馬名", ""),
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
            "能力ランク": _ver30_text_value(row.get("ability_rank", "")) or "-",
            "能力ランク理由": _ver30_text_value(row.get("ability_rank_reason", "")) or "-",
            "勢いランク": _ver30_text_value(row.get("momentum_rank", "")) or "-",
            "勢いスコア": row.get("momentum_score", ""),
            "勢い理由": _ver30_text_value(row.get("momentum_reason", "")) or "-",
            "近3走傾向": _ver30_text_value(row.get("recent3_trend", "")) or "-",
            "総合ランク": _ver30_text_value(row.get("overall_rank", "")) or "-",
            "総合ランク理由": _ver30_text_value(row.get("overall_rank_reason", "")) or "-",
            "AI順位": _ver30_audit_rank_display(row),
            "軸信頼度": _ver30_text_value(row.get("axis_confidence", "")) or "-",
            "軸信頼度理由": _ver30_text_value(row.get("axis_confidence_reason", "")) or "-",
            "能力帯": _ver30_text_value(row.get("ability_band", "")) or "-",
            "能力差": _ver30_text_value(row.get("ability_gap_level", "")) or "-",
            "レース難易度": _ver30_text_value(row.get("race_difficulty", "")) or "-",
            "レース難易度理由": _ver30_text_value(row.get("race_difficulty_reason", "")) or "-",
            "AI点": _ver30_ai_point_display(row),
            "クラス変動": _ver30_class_shift_short(row),
            "チェック項目": _ver30_text_value(row.get("チェック項目", "")) or "-",
            "補足": _ver30_text_value(row.get("補足", "")) or "なし",
        }
        if str(race_type).lower() == "jra":
            base["調教評価"] = _ver30_training_eval_short(row)
            base["厩舎コメント"] = _ver30_stable_comment_short(row)
        else:
            base["対戦評価"] = _ver30_matchup_eval_short(row)
        base.update({
            "評価／検討材料": _ver30_material_tags(row, race_type),
            "馬タイプ": row.get("_馬タイプ", ""),
            "穴候補": _ver30_audit_bool_label(row.get("hole_candidate")),
            "注意馬": _ver30_audit_bool_label(row.get("watch_horse")),
            "表示コメント": _ver30_text_value(row.get("display_comment", "")),
            "一言コメント": _ver30_text_value(row.get("display_comment", "")) or row.get("_Ver30コメント", ""),
        })
        rows.append(base)
    rating_df = pd.DataFrame(rows)
    try:
        display(format_result_for_output(rating_df))
    except Exception:
        display(rating_df)


def _ver30_horse_label(row):
    no = str(row.get("_馬_馬番", "") or "").strip()
    name = str(row.get("_馬_馬名", "") or "").strip()
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
        mark = _ver30_text_value(row.get("display_mark", "")) or "無印"
        print("")
        print(_ver30_horse_label(row))
        print(f"印：{mark}")
        print(_ver30_attention_comment(row))


def _ver30_attention_comment(row):
    horse_type = str(row.get("_馬タイプ", ""))
    value_level = int(row.get("_Ver30妙味Lv", 1) or 1)
    ability_level = int(row.get("_Ver30能力Lv", 1) or 1)
    market_level = int(row.get("_Ver30市場Lv", 1) or 1)
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
        mark = _ver30_text_value(value_horse.get("display_mark", "")) or "無印"
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

    watch_candidate = (~core_mark) & (class_down | single_or_payout_comment | pace_favorable | high_eval | other_watch)
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

    score = pd.to_numeric(result.get("_最終印点"), errors="coerce")
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

    model_weight = np.exp((normalized - normalized.max()) * 2.4)
    model_total = float(model_weight.sum())
    model_probability = (
        model_weight / model_total
        if model_total > 0
        else pd.Series(1.0 / len(result), index=result.index)
    )

    market_raw = pd.Series(0.0, index=result.index, dtype="float64")
    valid_odds = odds.notna() & odds.gt(0)
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

    fair_odds = (1.0 / probability).replace([np.inf, -np.inf], np.nan)
    win_ev = probability * odds
    probability_percent = probability * 100
    probability_rank = probability_percent.rank(method="min", ascending=False)

    style = result.get("脚質", pd.Series("", index=result.index)).fillna("").astype(str)
    pace_mark = result.get("展開印", pd.Series("", index=result.index)).fillna("").astype(str)
    mark = result.get("最終印", pd.Series("", index=result.index)).fillna("").astype(str)
    marked = mark.ne("")
    honmei = mark.eq("◎")
    main_partner = mark.isin(["○", "▲"])
    reserve = mark.eq("△")
    star = mark.eq("✓")
    top3_probability = probability_rank.le(3)
    top5_probability = probability_rank.le(5)
    late_closer_without_pace = style.str.contains("追", na=False) & ~pace_mark.eq("展")

    single_value_mask = (
        marked
        & win_ev.ge(1.18)
        & odds.between(2.0, 20.0, inclusive="both")
        & ~late_closer_without_pace
    )

    h2h_text = result.get("対戦", pd.Series("", index=result.index)).fillna("").astype(str)
    horse_numbers = pd.to_numeric(result.get("馬番"), errors="coerce")
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

    honmei_single_axis_mask = honmei & single_value_mask
    honmei_closer_axis_mask = honmei & late_closer_without_pace
    honmei_axis_mask = honmei & ~honmei_single_axis_mask & ~honmei_closer_axis_mask
    main_single_partner_mask = main_partner & single_value_mask
    main_partner_strong_mask = main_partner & top5_probability & ~main_single_partner_mask
    main_partner_mask = main_partner & ~main_single_partner_mask & ~main_partner_strong_mask
    reserve_single_mask = reserve & single_value_mask
    reserve_mask = reserve & ~reserve_single_mask
    star_single_mask = star & single_value_mask
    star_pace_mask = star & pace_mark.eq("展") & ~star_single_mask
    star_probability_mask = star & top3_probability & ~star_single_mask & ~star_pace_mask
    star_mask = star & ~star_single_mask & ~star_pace_mask & ~star_probability_mask
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
            main_single_partner_mask,
            main_partner_strong_mask,
            main_partner_mask,
            reserve_single_mask,
            reserve_mask,
            star_single_mask,
            star_pace_mask,
            star_probability_mask,
            star_mask,
            h2h_cover_mask,
            probability_partner_mask,
        ],
        [
            "単勝妙味/中心候補",
            "追込注意候補",
            "中心候補",
            "単勝妙味/相手有力",
            "相手有力",
            "印相手候補",
            "単勝妙味/押さえ候補",
            "押さえ候補",
            "単勝妙味/✓穴候補",
            "展開注意候補",
            "✓穴候補有力",
            "✓穴候補",
            "対戦押さえ候補",
            "勝率補助候補",
        ],
        default="見送り",
    )
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


def build_purchase_candidate_table(df, limit=8):
    import re

    source = add_purchase_value_columns(df)
    source = prepare_jra_display_columns(source)
    if source.empty:
        return pd.DataFrame()

    mark = source.get("最終印", pd.Series("", index=source.index)).fillna("").astype(str)
    valid_marks = ["◎", "○", "▲", "△", "✓", "☆"]
    pool = source[mark.isin(valid_marks)].copy()
    columns = [
        "印", "馬番", "馬名", "騎手", "斤量", "脚質", "レース間隔", "オッズ",
        "AI点", "総合評価", "市場反映勝率", "単勝期待値", "評価根拠",
        "調教/評価", "厩舎コメント", "買い方メモ",
    ]
    if pool.empty:
        return pd.DataFrame(columns=columns)

    def clean_text(value):
        text = str(value or "").replace("\n", " ").strip()
        return re.sub(r"\s+", " ", text)

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
        pool["評価根拠"] = pool.apply(build_jra_evaluation_material, axis=1)

    material_sources = [
        "調教/評価/検討材料",
        "評価/検討材料",
        "追切材料",
        "調教評価",
        "新聞材料",
        "状態材料",
    ]
    pool["調教/評価"] = ""
    for column in material_sources:
        if column in pool.columns:
            values = pool[column].map(clean_text)
            pool["調教/評価"] = pool["調教/評価"].where(pool["調教/評価"].astype(str).ne(""), values)

    comment_sources = ["厩舎コメント", "新聞コメント", "馬コメント"]
    pool["厩舎コメント"] = ""
    for column in comment_sources:
        if column in pool.columns:
            values = pool[column].map(clean_text)
            pool["厩舎コメント"] = pool["厩舎コメント"].where(pool["厩舎コメント"].astype(str).ne(""), values)

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

    def state_style(value):
        text = str(value or "").strip()
        if re.match(r"^[SＳＡA]", text):
            return "background-color: #dff4e8; font-weight: 700;"
        if text.startswith("B"):
            return "background-color: #eaf4ff; font-weight: 700;"
        if text.startswith("C"):
            return "background-color: #fff4e2; color: #7a4a00;"
        return ""

    def interval_style(value):
        text = str(value or "")
        if "休み明け" in text or "休" in text:
            return "background-color: #fff4e2; color: #7a4a00;"
        return ""

    try:
        styler = formatted.style
        if "オッズ" in formatted.columns:
            styler = styler.set_properties(subset=["オッズ"], **{"font-weight": "700"})
        if "調教/評価" in formatted.columns:
            styler = styler.applymap(state_style, subset=["調教/評価"])
        if "レース間隔" in formatted.columns:
            styler = styler.applymap(interval_style, subset=["レース間隔"])
        elif "間隔" in formatted.columns:
            styler = styler.applymap(interval_style, subset=["間隔"])
        return styler
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
        ability_text = str(row.get("能力") or "")
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

        oikiri_material = str(row.get("追切材料") or "")
        if "格下遅れ" in oikiri_material or "併せ遅れ" in oikiri_material:
            parts.append("追切の併せ遅れは割引")
        elif any(key in oikiri_material for key in ["追切A", "追切S", "文句なし", "一番時計", "動き良"]):
            parts.append("追切材料良")

        if rank_value is not None and rank_value <= 3:
            parts.append("能力上位")
        elif rank_value is not None and rank_value <= 6:
            parts.append("相手圏")
        elif "下位" in ability_text:
            parts.append("能力面は割引")

        if layoff.loc[idx]:
            material = str(row.get("追切評価") or row.get("追切材料") or row.get("調教評価") or row.get("新聞材料") or "").strip()
            if material:
                parts.append("状態材料あるが休み明け")
            else:
                parts.append("休み明けで信頼度割引")

        if safe_num(star_rank.loc[idx], None) is not None and star_rank.loc[idx] <= 3:
            parts.append("同条件実績あり")
        elif safe_num(distance_rank.loc[idx], None) is not None and distance_rank.loc[idx] <= 3:
            parts.append("距離指数上位")
        elif safe_num(course_rank.loc[idx], None) is not None and course_rank.loc[idx] <= 3:
            parts.append("コース指数上位")

        if (
            pop_value is not None
            and rank_value is not None
            and pop_value - rank_value >= 3
        ) or (odds_value is not None and odds_value >= 8 and rank_value is not None and rank_value <= 8):
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



#@title JRA会場別試験ロジック
# AI点は変更せず、2026年6月21日の会場傾向を独立した補助欄へ表示します。

JRA_VENUE_PROFILES = {
    "東京": {
        "sample_races": 4,
        "headline": "近3走最高を中心に、長い直線で差しの進出を確認",
        "primary": "最高指数",
        "secondary": "コース指数",
    },
    "阪神": {
        "sample_races": 11,
        "headline": "近3走平均と能力の安定性を重視。追込単独は展開待ち",
        "primary": "平均指数",
        "secondary": "距離指数",
    },
    "函館": {
        "sample_races": 11,
        "headline": "最高指数と前走指数を確認。4角好位へ運べる馬を重視",
        "primary": "最高指数",
        "secondary": "_last",
    },
}


def jra_venue_rank(series):
    return pd.to_numeric(series, errors="coerce").rank(
        method="min", ascending=False
    )


def jra_rank_bonus(rank_value, top1=2.5, top3=1.5, top5=0.7):
    if pd.isna(rank_value):
        return 0.0
    if rank_value <= 1:
        return top1
    if rank_value <= 3:
        return top3
    if rank_value <= 5:
        return top5
    return 0.0


def jra_metric_label(column):
    return {
        "最高指数": "近3走最高",
        "平均指数": "近3走平均",
        "コース指数": "コース指数",
        "距離指数": "距離指数",
        "_last": "前走指数",
    }.get(column, column)


def apply_jra_venue_profile(df, race_info, has_style_html=False):
    result = df.copy()
    if "間隔" not in result.columns:
        if "_days_since_last" in result.columns:
            result["間隔"] = result["_days_since_last"].map(format_interval_from_days)
        else:
            result["間隔"] = ""
    if "馬場適性" not in result.columns:
        result["馬場適性"] = ""
    if "クラス変動" not in result.columns:
        result["クラス変動"] = ""
    venue = str(race_info.get("racecourse") or "").strip()
    profile = JRA_VENUE_PROFILES.get(venue)

    ranks = {
        "最高指数": jra_venue_rank(
            result.get("最高指数", pd.Series(index=result.index, dtype="float64"))
        ),
        "平均指数": jra_venue_rank(
            result.get("平均指数", pd.Series(index=result.index, dtype="float64"))
        ),
        "コース指数": jra_venue_rank(
            result.get("コース指数", pd.Series(index=result.index, dtype="float64"))
        ),
        "距離指数": jra_venue_rank(
            result.get("距離指数", pd.Series(index=result.index, dtype="float64"))
        ),
        "_last": jra_venue_rank(
            result.get("_last", pd.Series(index=result.index, dtype="float64"))
        ),
    }
    ai_rank = pd.to_numeric(result.get("AI順位"), errors="coerce")
    styles = result.get(
        "脚質", pd.Series("", index=result.index)
    ).map(normalize_running_style)

    scores = []
    reasons = []
    for idx in result.index:
        score = 0.0
        parts = []
        rank = ai_rank.loc[idx]
        age = extract_age_from_sex_age(result.at[idx, "性齢"])
        if pd.notna(rank):
            if rank <= 3:
                score += 2.0
                parts.append("能力上位")
            elif rank <= 5:
                score += 1.0

        if profile:
            primary = profile["primary"]
            secondary = profile["secondary"]
            primary_rank = ranks[primary].loc[idx]
            secondary_rank = ranks[secondary].loc[idx]
            score += jra_rank_bonus(primary_rank, 3.0, 2.0, 0.8)
            score += jra_rank_bonus(secondary_rank, 2.0, 1.2, 0.5)
            if pd.notna(primary_rank) and primary_rank <= 3:
                parts.append(f"{jra_metric_label(primary)}上位")
            if pd.notna(secondary_rank) and secondary_rank <= 3:
                parts.append(f"{jra_metric_label(secondary)}上位")

            if has_style_html:
                style = styles.loc[idx]
                if venue == "東京":
                    if style == "差":
                        score += 0.6
                        parts.append("差し進出型")
                    elif style == "追":
                        score -= 0.3
                        parts.append("後方展開待ち")
                elif venue == "阪神":
                    if style == "差":
                        score += 0.4
                        parts.append("差し安定")
                    elif style == "逃":
                        score += 0.3
                        parts.append("逃げ残り注意")
                    elif style == "追":
                        score -= 0.5
                        parts.append("追込展開待ち")
                elif venue == "函館":
                    if style == "先":
                        score += 0.9
                        parts.append("好位有利")
                    elif style == "逃":
                        score += 0.7
                        parts.append("前残り警戒")
                    elif style == "差":
                        score += 0.2
                        parts.append("早め進出なら")
                    elif style == "追":
                        score -= 0.8
                        parts.append("後方不利")
        else:
            score += jra_rank_bonus(ranks["最高指数"].loc[idx], 1.5, 1.0, 0.5)
            score += jra_rank_bonus(ranks["_last"].loc[idx], 1.5, 1.0, 0.5)
            if ranks["最高指数"].loc[idx] <= 3:
                parts.append("近3走最高上位")
            if ranks["_last"].loc[idx] <= 3:
                parts.append("前走指数上位")

        if age is not None and age >= 11:
            score -= 3.0
            parts.append("11歳以上注意")
        elif age is not None and age >= 9:
            score -= 1.2
            parts.append("高齢注意")

        unique = []
        for part in parts:
            if part not in unique:
                unique.append(part)
        scores.append(round(score, 1))
        reasons.append(" / ".join(unique[:5]) or "会場別強調なし")

    support_counts = []
    support_texts = []
    for idx in result.index:
        labels = []
        if ranks["最高指数"].loc[idx] <= 5:
            labels.append("近3走最高")
        if ranks["コース指数"].loc[idx] <= 5:
            labels.append("コース")
        if ranks["_last"].loc[idx] <= 5:
            labels.append("前走")
        support_counts.append(len(labels))
        support_texts.append(
            f"{len(labels)}/3（{'・'.join(labels)}）"
            if labels
            else "0/3"
        )

    result["_JRA会場評価点"] = scores
    result["_指数裏付け数"] = support_counts
    result["指数裏付け"] = support_texts
    result["会場理由"] = reasons
    result["会場評価"] = pd.Series(scores, index=result.index).map(
        lambda value: (
            "会場強調"
            if value >= 5
            else "会場注意"
            if value >= 2.5
            else "補助材料少"
        )
    )

    def jra_ticket_type(row):
        odds = pd.to_numeric(row.get("単勝オッズ"), errors="coerce")
        ai_rank = pd.to_numeric(row.get("AI順位"), errors="coerce")
        support_count = pd.to_numeric(row.get("_指数裏付け数"), errors="coerce")
        support_count = int(support_count) if pd.notna(support_count) else 0
        venue_score = pd.to_numeric(row.get("_JRA会場評価点"), errors="coerce")
        style = normalize_running_style(row.get("脚質", ""))
        age = extract_age_from_sex_age(row.get("性齢"))

        if pd.isna(odds):
            return "見送り"
        if age is not None and age >= 11 and support_count < 3:
            return "見送り"
        if odds < 2:
            return "ワイド軸" if support_count >= 2 or (pd.notna(ai_rank) and ai_rank <= 3) else "見送り"
        if (
            pd.notna(ai_rank)
            and ai_rank <= 3
            and support_count >= 2
            and pd.notna(venue_score)
            and venue_score >= 4
        ):
            return "単勝＋ワイド"
        if pd.notna(ai_rank) and ai_rank <= 3 and support_count >= 2:
            return "単勝"
        if odds >= 10 and support_count >= 1 and pd.notna(ai_rank) and ai_rank <= 10:
            return "複勝穴"
        if odds >= 5 and odds <= 30 and support_count >= 1:
            return "ワイド"
        return "見送り"

    def jra_ticket_reason(row):
        kind = row.get("推奨券種", "")
        odds = pd.to_numeric(row.get("単勝オッズ"), errors="coerce")
        ai_rank = pd.to_numeric(row.get("AI順位"), errors="coerce")
        support_count = pd.to_numeric(row.get("_指数裏付け数"), errors="coerce")
        support_count = int(support_count) if pd.notna(support_count) else 0
        venue_score = pd.to_numeric(row.get("_JRA会場評価点"), errors="coerce")
        style = normalize_running_style(row.get("脚質", ""))
        parts = []
        if pd.notna(ai_rank) and ai_rank <= 3:
            parts.append("AI上位")
        if support_count >= 2:
            parts.append(f"指数{support_count}/3")
        if pd.notna(venue_score) and venue_score >= 4:
            parts.append("会場材料")
        if pd.notna(odds):
            if odds < 2:
                parts.append("1倍台")
            elif odds >= 10:
                parts.append("配当妙味")
        if style in ("逃", "先"):
            parts.append("前受け")
        elif style in ("差", "追"):
            parts.append("差し脚")
        if not parts:
            parts.append("強調材料少")
        return f"{kind}：" + " / ".join(parts[:4])

    result["推奨券種"] = result.apply(jra_ticket_type, axis=1)
    result["券種理由"] = result.apply(jra_ticket_reason, axis=1)
    return result, venue, profile


def print_jra_venue_profile(venue, profile, has_style_html):
    print("【JRA会場別試験評価】")
    if profile:
        print(f"会場判定: {venue}")
        print(f"初期傾向: {profile['headline']}")
        print(f"検証数: {profile['sample_races']}レース（2026年6月21日・暫定）")
    else:
        print(f"会場判定: {venue or '不明'}（専用傾向は未登録）")
        print("共通評価: 近3走最高と前走指数を確認します。")
    print(f"脚質データ: {'あり' if has_style_html else 'なし'}")
    print("注意: 会場評価・指数裏付けはAI点へ加算していません。")


def print_jra_single_place_conclusion(axis_df):
    print("")
    print("★★★★★★★★★★★★★★★★")
    print("★【券種適性メモ】★")
    print("★★★★★★★★★★★★★★★★")
    if axis_df.empty:
        print("判定できません。")
        return

    row = axis_df.iloc[0]
    horse_no = pd.to_numeric(row.get("馬番"), errors="coerce")
    no_text = str(int(horse_no)) if pd.notna(horse_no) else str(row.get("馬番", "")).strip()
    horse_label = f"{no_text} {row.get('馬名', '')}"
    odds = pd.to_numeric(row.get("単勝オッズ"), errors="coerce")
    reproducibility = str(row.get("再現性", ""))
    dependency = str(row.get("展開依存", ""))
    evaluation = str(row.get("中心評価", ""))
    age = extract_age_from_sex_age(row.get("性齢"))
    support_count = pd.to_numeric(row.get("_指数裏付け数"), errors="coerce")
    support_count = int(support_count) if pd.notna(support_count) else 0
    axis_df = axis_df.copy()
    axis_df["_表示オッズ"] = pd.to_numeric(axis_df.get("単勝オッズ"), errors="coerce")
    odds_on_pool = axis_df[axis_df["_表示オッズ"].lt(2)].sort_values(
        "_表示オッズ", na_position="last"
    )
    odds_on_favorite = odds_on_pool.iloc[0] if not odds_on_pool.empty else None
    odds_on_no = (
        pd.to_numeric(odds_on_favorite.get("馬番"), errors="coerce")
        if odds_on_favorite is not None
        else pd.NA
    )

    if pd.isna(odds):
        print(f"単勝候補：なし（{horse_label}はオッズ未取得）。")
        judgment = "見送り"
    elif odds < 2:
        print(f"単勝候補：なし（{horse_label}は単勝{format_number_for_display(odds)}倍。1倍台のため相手穴確認）。")
        judgment = "見送り"
    elif (
        age is not None
        and age >= 11
        and not (
            reproducibility == "高"
            and dependency == "低"
            and support_count == 3
        )
    ):
        judgment = "見送り"
    elif (
        evaluation == "中心にしやすい"
        and reproducibility == "高"
        and dependency != "高"
    ):
        judgment = "中心候補"
    elif reproducibility in ("高", "中") and dependency != "高":
        judgment = "相手候補"
    else:
        judgment = "見送り"
    if not pd.isna(odds) and odds >= 2:
        if judgment == "見送り":
            print(
                f"単勝候補：なし（{horse_label}は"
                f"再現性{reproducibility}・展開依存{dependency}）。"
            )
        else:
            print(
                f"単勝候補：{circled_number(int(horse_no)) if pd.notna(horse_no) else no_text}"
                f"{row.get('馬名', '')} 単勝{format_number_for_display(odds)}倍。"
                f"{judgment}（再現性{reproducibility}・展開依存{dependency}）。"
            )

    anchor_no = horse_no
    anchor_label = circled_number(int(horse_no)) if pd.notna(horse_no) else no_text
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

    hole_pool = axis_df[
        pd.to_numeric(axis_df.get("馬番"), errors="coerce").ne(anchor_no)
    ].copy()
    hole_pool["_穴オッズ"] = pd.to_numeric(
        hole_pool.get("単勝オッズ"), errors="coerce"
    )
    hole_pool["_穴AI順位"] = pd.to_numeric(
        hole_pool.get("AI順位"), errors="coerce"
    )
    hole_pool["_穴年齢"] = hole_pool.get(
        "性齢", pd.Series("", index=hole_pool.index)
    ).map(extract_age_from_sex_age)
    hole_pool = hole_pool[
        hole_pool["_穴オッズ"].ge(10)
        & hole_pool["_穴AI順位"].le(10)
        & hole_pool.get("再現性", pd.Series("", index=hole_pool.index)).isin(["高", "中"])
        & (hole_pool["_穴年齢"].isna() | hole_pool["_穴年齢"].lt(11))
    ]
    if (judgment == "見送り" and odds_on_favorite is None) or hole_pool.empty:
        print("穴候補：該当なし。")
        return

    hole_limit = 3 if odds_on_favorite is not None else 1
    hole_numbers = []
    for _, hole in hole_pool.head(hole_limit).iterrows():
        hole_no = pd.to_numeric(hole.get("馬番"), errors="coerce")
        hole_no_text = str(int(hole_no)) if pd.notna(hole_no) else str(hole.get("馬番", "")).strip()
        hole_mark = circled_number(int(hole_no)) if pd.notna(hole_no) else hole_no_text
        hole_numbers.append(hole_mark)
        hole_odds = format_number_for_display(hole.get("単勝オッズ"))
        print(
            f"穴候補：{hole_mark}{hole.get('馬名', '')} 単勝{hole_odds}倍"
            f"（再現性{hole.get('再現性', '')}・展開依存{hole.get('展開依存', '')}）。"
        )
    if pd.notna(anchor_no) and hole_numbers:
        print(f"ワイド検討：{anchor_label} - {''.join(hole_numbers)}")


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


def _run_jra_notebook_body(
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
            "html_file_name": file_names.get("speed", ""),
            "style_html_file_name": file_names.get("style", ""),
            "odds_html_file_name": "",
            "html_from_newspaper_file": html_files.get("newspaper", ""),
            "html_from_oikiri_file": html_files.get("oikiri", ""),
            "newspaper_html_file_name": file_names.get("newspaper", ""),
            "oikiri_html_file_name": file_names.get("oikiri", ""),
    })
    capture.reset()
    #@title 解析実行
    html_input = globals().get("html_from_pc_file", "").strip()
    if not html_input:
        raise ValueError("先に `HTMLファイルをまとめてアップロード` セルで、タイム指数HTMLを含めてアップロードしてください。")

    session = make_session("")
    html = html_input
    source_label = f"PC保存HTML: {globals().get('html_file_name', '')}"
    fetch_past_detail = globals().get("FETCH_PAST_RACE_DETAIL", True)
    past_race_sleep_sec = globals().get("PAST_RACE_SLEEP_SEC", 0.35)
    show_corner_scenario = globals().get("SHOW_CORNER_SCENARIO", True)
    OPTIONAL_ODDS_HTML_UI_ENABLED = False  # オッズHTML UIは一時的に非表示
    style_html_input = globals().get("html_from_style_file", "").strip()
    newspaper_html_input = globals().get("html_from_newspaper_file", "").strip()
    oikiri_html_input = globals().get("html_from_oikiri_file", "").strip()
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

    result_df, race_info = parse_speed_table(
        html=html,
        race_url="",
        session=session,
        fetch_past_detail=fetch_past_detail,
        sleep_sec=past_race_sleep_sec,
    )
    result_df, style_df = apply_jra_style_features(result_df, style_html_input)
    running_style_info = analyze_running_style(result_df)
    result_df = add_newspaper_features(result_df, running_style_info)
    result_df, detected_venue, venue_profile = apply_jra_venue_profile(
        result_df, race_info, has_style_html=bool(style_html_input)
    )
    result_df = apply_jra_newspaper_html_features(result_df, newspaper_html_input)
    result_df = apply_jra_oikiri_html_features(result_df, oikiri_html_input)
    result_df = apply_state_material_column(result_df)
    result_df = add_final_marks(result_df, running_style_info)
    result_df = refresh_horse_pace_comments(result_df, running_style_info)
    result_df = apply_watch_marks(result_df, race_type="jra")
    result_df = remove_betting_output_columns(result_df)
    if "_最終印点" in result_df.columns:
        result_df["総合評価点"] = pd.to_numeric(result_df["_最終印点"], errors="coerce").round(1)
    result_df = add_purchase_value_columns(result_df)
    result_df = add_audit_evaluation_columns(result_df, race_type="jra")
    result_df = prepare_jra_display_columns(result_df)


    display_cols = ["表示印", "展開印", "馬番", "馬名", "馬年齢", "斤量", "騎手", "オッズ", "脚質", "レース間隔", "AI点", "総合評価", "市場反映勝率", "単勝期待値", "クラス変動", "クラス根拠", "馬場実績", "距離指数", "コース指数", "3走前", "2走前", "前走", "平均指数", "過去1年最高指数", "★最高指数", "★該当走", "★条件", "★最高指数の取得元", "調教/評価/検討材料", "能力評価値", "能力帯", "能力差", "レース難易度", "レース難易度理由", "表示コメント", "raw_score", "ability_display_score", "normalized_ai_score", "ai_rank", "final_mark_score", "market_score", "star_max_index", "star_max_race", "star_max_venue", "star_max_distance", "star_max_surface", "star_max_turn", "star_match_level", "star_max_source", "axis_confidence", "axis_confidence_reason", "ability_band", "ability_gap_level", "race_difficulty", "race_difficulty_reason", "display_comment", "old_final_mark", "old_watch_mark", "hole_candidate", "watch_horse"]
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
        "調教評価", "追切評価", "追切内容", "調教コメント", "厩舎コメント", "新聞コメント",
    ])
    print(f"レース: {race_info.get('race_name', '')} / {race_info.get('race_data', '')}")
    print(f"抽出頭数: {len(result_df)}")
    print_jra_venue_profile(detected_venue, venue_profile, bool(style_html_input))
    if style_html_input:
        style_count = int(result_df["脚質"].astype(str).ne("").sum())
        print(f"脚質HTML内の抽出頭数: {len(style_df)} / 表へ反映: {style_count}")
    else:
        print("脚質HTML: 未アップロード")
    if newspaper_html_input:
        newspaper_count = int(result_df.get("新聞材料", pd.Series("", index=result_df.index)).astype(str).ne("").sum())
        print(f"競馬新聞HTML内の抽出頭数: {newspaper_count} / ファイル: {globals().get('newspaper_html_file_name', '')}")
    else:
        print("競馬新聞HTML: 未アップロード")
    if oikiri_html_input:
        oikiri_count = int(result_df.get("追切材料", pd.Series("", index=result_df.index)).astype(str).ne("").sum())
        print(f"調教タイムHTML内の抽出頭数: {oikiri_count} / ファイル: {globals().get('oikiri_html_file_name', '')}")
    else:
        print("調教タイムHTML: 未アップロード")
    missing_past_labels = int(result_df.get("_missing_past_labels", pd.Series(dtype="int64")).sum())
    if fetch_past_detail:
        print(f"距離補完できなかった近走セル数: {missing_past_labels}")
        if missing_past_labels:
            print("注意: 3走前〜前走に指数だけの表示が残る場合があります。Colabから過去レースページへアクセスできない、またはページ構造が違う可能性があります。")
    else:
        print("過去レース詳細補完: OFF（3走前〜前走の距離表示と同条件判定は省略されます）")

    display_cols = [column for column in display_cols if column in result_df.columns]
    print("")
    print("【レース全体表】")
    try:
        display(result_display_styler(result_df[display_cols]))
    except Exception:
        display(format_result_for_output(result_df[display_cols]))
    print("")
    print_ver30_all_horse_rating(result_df, race_type="jra")
    print("")
    print_ver30_attention_horses(result_df, race_type="jra")
    print("")
    ai_confidence_summary = build_ai_confidence_summary(result_df, race_info, detected_venue, venue_profile, race_type="jra")
    print_ver30_ai_race_review(result_df, race_info, running_style_info, ai_confidence_summary, race_type="jra")
    print("")
    print_ver30_betting_structure(result_df, ai_confidence_summary, race_type="jra")

    return {
        "result_df": locals().get("result_df"),
        "race_info": locals().get("race_info"),
        "running_style_info": locals().get("running_style_info"),
        "ai_confidence_summary": locals().get("ai_confidence_summary"),
        "display_cols": locals().get("display_cols", []),
    }


def predict_jra_from_html(
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
        state = _run_jra_notebook_body(
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
    horse_evaluation = _ka_capture_first_dataframe(
        capture,
        lambda: print_ver30_all_horse_rating(result_df, race_type="jra"),
    )
    attention_text = _ka_capture_text(
        capture,
        lambda: print_ver30_attention_horses(result_df, race_type="jra"),
    )
    review_text = _ka_capture_text(
        capture,
        lambda: print_ver30_ai_race_review(
            result_df,
            race_info,
            state.get("running_style_info"),
            state.get("ai_confidence_summary"),
            race_type="jra",
        ),
    )
    betting_text = _ka_capture_text(
        capture,
        lambda: print_ver30_betting_structure(
            result_df,
            state.get("ai_confidence_summary"),
            race_type="jra",
        ),
    )

    return _KaPredictionResult(
        race_mode="jra",
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
        debug_info={"condition_fit_sources": extract_condition_fit_sources(result_df)},
    )

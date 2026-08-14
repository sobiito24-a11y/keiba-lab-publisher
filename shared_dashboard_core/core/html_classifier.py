from __future__ import annotations

import html
import re
from collections import defaultdict
from typing import Iterable
from urllib.parse import parse_qs, urlparse

from .models import ClassifiedHtml, HtmlMeta, RaceMode, UploadBundleValidation


KIND_LABELS = {
    "speed": "タイム指数",
    "shutuba": "出馬表",
    "style": "脚質分析",
    "jockey": "騎手コース成績",
    "newspaper": "競馬新聞",
    "oikiri": "調教",
    "odds": "オッズ",
    "unknown": "不明なHTML",
}

REQUIRED_KINDS: dict[RaceMode, tuple[str, ...]] = {
    "nar": ("speed", "newspaper", "style"),
    "jra": ("speed", "newspaper", "style"),
}

DISPLAY_ORDER: dict[RaceMode, tuple[str, ...]] = {
    "nar": ("speed", "newspaper", "style", "jockey", "shutuba"),
    "jra": ("speed", "newspaper", "style", "jockey", "oikiri"),
}

EVIDENCE_TIERS = (
    ("canonical", "og:url"),
    ("page url",),
    ("body id", "body class", "dom marker", "table id/class"),
    ("title",),
    ("file name",),
)


def required_kinds(mode: RaceMode) -> tuple[str, ...]:
    return REQUIRED_KINDS[mode]


def kind_label(kind: str) -> str:
    return KIND_LABELS.get(kind, kind)


def classify_netkeiba_page_url(value: str) -> str:
    """Classify a netkeiba race-page URL before any HTML is downloaded.

    ``courseanalysis`` is a page family, not one data kind.  cid=2 must be
    resolved as the optional jockey page before the generic cid=1/style
    fallback; otherwise an acquisition client rejects or misroutes it before
    the HTML classifier and parser can run.
    """

    text = html.unescape(str(value or "").strip())
    try:
        parsed = urlparse(text)
    except ValueError:
        return "unknown"
    host = parsed.netloc.lower()
    if host not in {
        "nar.netkeiba.com",
        "nar.sp.netkeiba.com",
        "race.netkeiba.com",
        "race.sp.netkeiba.com",
    }:
        return "unknown"
    path = parsed.path.lower()
    query = parse_qs(parsed.query)
    if path.endswith("/race/data_list.html") and "courseanalysis" in {
        item.lower() for item in query.get("mode", [])
    }:
        cid_values = {item.strip() for item in query.get("cid", [])}
        if "2" in cid_values:
            return "jockey"
        return "style"
    if path.endswith("/race/speed.html"):
        return "speed"
    if path.endswith("/race/newspaper.html"):
        return "newspaper"
    if path.endswith("/race/oikiri.html"):
        return "oikiri"
    if path.endswith("/race/shutuba.html"):
        return "shutuba"
    if "/odds/" in path:
        return "odds"
    return "unknown"


def decode_uploaded_html(data: bytes) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp932", "euc-jp"):
        try:
            return data.decode(encoding)
        except UnicodeDecodeError:
            continue
    return data.decode("utf-8", errors="replace")


def classify_many(files: Iterable[tuple[str, bytes]], mode: RaceMode) -> dict[str, list[ClassifiedHtml]]:
    grouped: dict[str, list[ClassifiedHtml]] = defaultdict(list)
    for file_name, data in files:
        html_text = decode_uploaded_html(data)
        item = classify_html(file_name, html_text, mode)
        grouped[item.kind].append(item)
    return dict(grouped)


def validate_upload_bundle(
    grouped: dict[str, list[ClassifiedHtml]],
    expected_mode: RaceMode,
) -> UploadBundleValidation:
    """Validate one manually uploaded race without parsing ambiguous files."""

    recognized = [
        item
        for kind, items in grouped.items()
        if kind != "unknown"
        for item in items
    ]
    errors: list[str] = []
    warnings: list[str] = []

    for item in recognized:
        if len(item.meta.race_ids) > 1 and not item.meta.race_id:
            errors.append(
                f"{item.file_name}: HTML内部に複数のrace_idがあります（{', '.join(item.meta.race_ids)}）"
            )
        elif not item.meta.race_id:
            errors.append(f"{item.file_name}: HTML内部からrace_idを確認できません")

        if len(item.meta.mode_candidates) > 1 and not item.meta.detected_mode:
            errors.append(
                f"{item.file_name}: JRA/NARの判定情報が競合しています（{', '.join(item.meta.mode_candidates)}）"
            )
        elif not item.meta.detected_mode:
            errors.append(f"{item.file_name}: HTML内部からJRA/NARを確認できません")
        elif item.meta.detected_mode != expected_mode:
            errors.append(
                f"{item.file_name}: {item.meta.detected_mode.upper()} HTMLです。"
                f"現在の{expected_mode.upper()}モードとは一致しません"
            )

    race_ids = tuple(sorted({item.meta.race_id for item in recognized if item.meta.race_id}))
    if len(race_ids) > 1:
        details = ", ".join(
            f"{item.file_name}={item.meta.race_id or '未確認'}" for item in recognized
        )
        errors.append(f"アップロードHTMLのrace_idが一致していません（{details}）")

    detected_modes = tuple(sorted({item.meta.detected_mode for item in recognized if item.meta.detected_mode}))
    if len(detected_modes) > 1:
        details = ", ".join(
            f"{item.file_name}={item.meta.detected_mode.upper() or '未確認'}" for item in recognized
        )
        errors.append(f"アップロードHTMLのJRA/NARが一致していません（{details}）")

    duplicate_kinds = tuple(
        kind for kind, items in grouped.items() if kind != "unknown" and len(items) > 1
    )
    for kind in duplicate_kinds:
        names = ", ".join(item.file_name for item in grouped[kind])
        warnings.append(
            f"{kind_label(kind)}HTMLが複数あります（{names}）。自動上書きせず、画面で選んだ1件だけを使用します"
        )

    unknowns = grouped.get("unknown", [])
    if unknowns:
        warnings.append(
            "不明なHTMLは解析しません（" + ", ".join(item.file_name for item in unknowns) + "）"
        )

    return UploadBundleValidation(
        race_id=race_ids[0] if len(race_ids) == 1 else "",
        detected_mode=detected_modes[0] if len(detected_modes) == 1 else "",
        errors=tuple(_dedupe(errors)),
        warnings=tuple(_dedupe(warnings)),
        duplicate_kinds=duplicate_kinds,
    )


def classify_html(file_name: str, html_text: str, mode: RaceMode) -> ClassifiedHtml:
    meta = extract_meta(file_name, html_text)
    matches = _collect_matches(meta, mode)
    kind = _decide_kind(matches)
    reasons = tuple(matches.get(kind, ())) if kind != "unknown" else ()
    return ClassifiedHtml(
        kind=kind,
        label=kind_label(kind),
        file_name=file_name,
        html_text=html_text,
        meta=meta,
        reasons=reasons,
        all_matches={k: tuple(v) for k, v in matches.items()},
    )


def extract_meta(file_name: str, html_text: str) -> HtmlMeta:
    source = str(html_text or "")
    head = source[:300_000]
    title = _clean_text(_first_tag_text(head, "title"))
    body_id = _attr_from_first_tag(head, "body", "id")
    body_class = _attr_from_first_tag(head, "body", "class")
    canonical = _extract_link_href(head, "canonical")
    og_url = _extract_meta_content(head, "og:url")
    table_markers = tuple(_extract_table_markers(head))
    page_urls = tuple(_extract_page_urls(head))
    dom_markers = tuple(_extract_dom_markers(head))
    race_id, race_ids = _extract_race_identity(
        file_name,
        head,
        canonical,
        og_url,
        page_urls,
    )
    detected_mode, mode_candidates = _extract_mode_identity(
        title,
        body_id,
        body_class,
        canonical,
        og_url,
        page_urls,
    )
    return HtmlMeta(
        file_name=str(file_name or ""),
        title=title,
        canonical=canonical,
        og_url=og_url,
        page_urls=page_urls,
        body_id=body_id,
        body_class=body_class,
        table_markers=table_markers,
        dom_markers=dom_markers,
        race_id=race_id,
        race_ids=race_ids,
        detected_mode=detected_mode,
        mode_candidates=mode_candidates,
    )


def _collect_matches(meta: HtmlMeta, mode: RaceMode) -> dict[str, list[str]]:
    matches: dict[str, list[str]] = defaultdict(list)
    fields = [
        ("canonical", meta.canonical),
        ("og:url", meta.og_url),
        ("page url", " ".join(meta.page_urls)),
        ("body id", meta.body_id),
        ("body class", meta.body_class),
        ("dom marker", " ".join(meta.dom_markers)),
        ("table id/class", " ".join(meta.table_markers)),
        ("title", meta.title),
        ("file name", meta.file_name),
    ]
    for source_name, value in fields:
        for kind, reason in _match_field(source_name, value, mode):
            matches[kind].append(reason)
    return dict(matches)


def _match_field(source_name: str, value: str, mode: RaceMode) -> list[tuple[str, str]]:
    text = str(value or "")
    lower = text.lower()
    found: list[tuple[str, str]] = []

    def add(kind: str, description: str) -> None:
        found.append((kind, f"{source_name}: {description}"))

    if source_name in {"file name", "title"}:
        if "タイム指数" in text:
            add("speed", "タイム指数")
        if "有利な脚質" in text or "脚質 データ分析" in text:
            add("style", "有利な脚質")
        if "得意な騎手" in text or "騎手 データ分析" in text:
            add("jockey", "得意な騎手")
        if "出馬表" in text or "出走表" in text:
            add("shutuba", "出馬表")
        if "競馬新聞" in text:
            add("newspaper", "競馬新聞")
        if "調教" in text or "追い切り" in text:
            add("oikiri", "調教・追い切り")
        if "オッズ" in text or "odds" in lower:
            add("odds", "オッズ")

        if source_name == "file name":
            if re.search(r"(?:^|[_\-.])speed(?:[_\-.]|$)", lower):
                add("speed", "speed")
            if "newspaper" in lower:
                add("newspaper", "newspaper")
            if "courseanalysis" in lower:
                add("style", "courseanalysis")
            if "jockey" in lower:
                add("jockey", "jockey")
            if "oikiri" in lower or "training" in lower or "workout" in lower:
                add("oikiri", "oikiri/training")
            if "shutuba" in lower:
                add("shutuba", "shutuba")

    if source_name in {"canonical", "og:url", "page url"}:
        urls = re.findall(r"https?://[^\s\"']+", html.unescape(text))
        url_kinds = {classify_netkeiba_page_url(url) for url in urls}
        url_kinds.discard("unknown")
        for url_kind in sorted(url_kinds):
            if url_kind in {"style", "jockey"}:
                add(url_kind, "mode=courseanalysis&cid=2" if url_kind == "jockey" else "mode=courseanalysis")
        if "/race/speed.html" in lower or "speed.html" in lower:
            add("speed", "speed.html")
        if "/race/shutuba.html" in lower or "shutuba.html" in lower:
            add("shutuba", "shutuba.html")
        if "newspaper" in lower:
            add("newspaper", "newspaper")
        if "oikiri" in lower:
            add("oikiri", "oikiri")
        if "/odds/" in lower:
            add("odds", "/odds/")

    if source_name in {"body id", "body class"}:
        if text == "Netkeiba_Race_OddsView":
            add("odds", "Netkeiba_Race_OddsView")
        if text in {"Netkeiba_Race_NewsPaper", "Netkeiba_Race_Newspaper"}:
            add("newspaper", text)
        if _contains_any(lower, ("page_race_oikiri", "netkeiba_race_oikiri")):
            add("oikiri", "page_race_oikiri/Netkeiba_Race_Oikiri")
        # ``race_data_list`` is shared by courseanalysis cid=1 (running
        # style) and cid=2 (jockey statistics), so it is not kind evidence.
        # A specific URL, page title, or page-specific DOM must decide it.
        if _contains_any(lower, ("page_race_speed", "netkeiba_race_speed")):
            add("speed", "page_race_speed/Netkeiba_Race_Speed")

    if source_name in {"dom marker", "table id/class"}:
        if _contains_any(text, ("Speed_List", "SpeedIndex_Table", "speed_list")):
            add("speed", "Speed_List/SpeedIndex_Table")
        if _contains_any(text, ("CourseAnalysis", "RaceData_CourseAnalysis", "DataGraphWrap1", "score1")):
            add("style", "CourseAnalysis")
        if _contains_any(text, ("Shutuba_Table", "RaceTable_Shutuba")):
            add("shutuba", "Shutuba_Table")
        if _contains_any(text, ("NewsPaper", "Newspaper", "RaceNewspaper", "riot-shutuba-past")):
            add("newspaper", "NewsPaper table")
        if _contains_any(text, ("Oikiri", "Training", "Workout")):
            add("oikiri", "Oikiri table")
        if _contains_any(text, ("RaceOdds_HorseList_Table", "Odds_Table", "OddsTable")):
            add("odds", "Odds table")

    return found


def _decide_kind(matches: dict[str, list[str]]) -> str:
    if not matches:
        return "unknown"

    for tier in EVIDENCE_TIERS:
        candidates = [
            kind
            for kind, reasons in matches.items()
            if any(
                reason.startswith(f"{source_name}:")
                for source_name in tier
                for reason in reasons
            )
        ]
        candidates = list(dict.fromkeys(candidates))
        if len(candidates) == 1:
            return candidates[0]
        if candidates:
            # Conflicting evidence at one tier is deliberately not resolved by
            # a generic kind fallback. A lower-trust tier must not overrule it.
            return "unknown"
    return "unknown"


def _first_tag_text(source: str, tag_name: str) -> str:
    match = re.search(rf"<{tag_name}\b[^>]*>(.*?)</{tag_name}>", source, flags=re.I | re.S)
    return match.group(1) if match else ""


def _attr_from_first_tag(source: str, tag_name: str, attr_name: str) -> str:
    match = re.search(rf"<{tag_name}\b[^>]*>", source, flags=re.I | re.S)
    if not match:
        return ""
    return _extract_attr(match.group(0), attr_name)


def _extract_link_href(source: str, rel_value: str) -> str:
    for tag in re.findall(r"<link\b[^>]*>", source, flags=re.I | re.S):
        rel = _extract_attr(tag, "rel").lower()
        if rel_value.lower() in rel:
            return _extract_attr(tag, "href")
    return ""


def _extract_meta_content(source: str, property_value: str) -> str:
    for tag in re.findall(r"<meta\b[^>]*>", source, flags=re.I | re.S):
        prop = _extract_attr(tag, "property").lower() or _extract_attr(tag, "name").lower()
        if prop == property_value.lower():
            return _extract_attr(tag, "content")
    return ""


def _extract_table_markers(source: str) -> list[str]:
    markers: list[str] = []
    for tag in re.findall(r"<table\b[^>]*>", source, flags=re.I | re.S):
        for attr in ("id", "class"):
            value = _extract_attr(tag, attr)
            if value:
                markers.extend(part for part in re.split(r"\s+", value) if part)
    return markers


def _extract_page_urls(source: str) -> list[str]:
    urls: list[str] = []
    for tag in re.findall(r"<link\b[^>]*>", source, flags=re.I | re.S):
        rel = _extract_attr(tag, "rel").lower()
        if "alternate" in rel:
            value = _extract_attr(tag, "href")
            if value:
                urls.append(value)
    urls.extend(
        html.unescape(value.strip())
        for value in re.findall(r'''["']@id["']\s*:\s*["']([^"']+)["']''', source, flags=re.I)
        if value.strip()
    )
    return _dedupe(urls)


def _extract_dom_markers(source: str) -> list[str]:
    markers: list[str] = []
    for tag in re.findall(r"<([a-zA-Z][\w:-]*)\b[^>]*>", source, flags=re.I | re.S):
        if "-" in tag:
            markers.append(tag)
    for tag in re.findall(r"<[a-zA-Z][\w:-]*\b[^>]*>", source, flags=re.I | re.S):
        for attr in ("id", "class", "data-is"):
            value = _extract_attr(tag, attr)
            if value:
                markers.extend(part for part in re.split(r"\s+", value) if part)
    return _dedupe(markers)


def _extract_race_identity(
    file_name: str,
    source: str,
    canonical: str,
    og_url: str,
    page_urls: tuple[str, ...],
) -> tuple[str, tuple[str, ...]]:
    tiers = (
        (canonical, og_url),
        page_urls,
        tuple(
            re.findall(
                r'''(?:\brace[_-]?id\b|data-race-id)["']?\s*(?:=|:)\s*["']?(\d{10,14})''',
                html.unescape(source),
                flags=re.I,
            )
        ),
        (file_name,),
    )
    seen: list[str] = []
    for values in tiers:
        tier_ids: list[str] = []
        for value in values:
            tier_ids.extend(re.findall(r"(?:race_id=)?(\d{10,14})", html.unescape(str(value or ""))))
        tier_ids = _dedupe(tier_ids)
        seen.extend(race_id for race_id in tier_ids if race_id not in seen)
        if len(tier_ids) == 1:
            return tier_ids[0], tuple(seen)
        if len(tier_ids) > 1:
            return "", tuple(seen)
    return "", tuple(seen)


def _extract_mode_identity(
    title: str,
    body_id: str,
    body_class: str,
    canonical: str,
    og_url: str,
    page_urls: tuple[str, ...],
) -> tuple[str, tuple[str, ...]]:
    tiers = (
        (canonical, og_url),
        page_urls,
        (body_id, body_class, title),
    )
    seen: list[str] = []
    for values in tiers:
        tier_modes: list[str] = []
        for value in values:
            mode = _mode_from_value(value)
            if mode:
                tier_modes.append(mode)
        tier_modes = _dedupe(tier_modes)
        seen.extend(mode for mode in tier_modes if mode not in seen)
        if len(tier_modes) == 1:
            return tier_modes[0], tuple(seen)
        if len(tier_modes) > 1:
            return "", tuple(seen)
    return "", tuple(seen)


def _mode_from_value(value: str) -> str:
    text = html.unescape(str(value or ""))
    lower = text.lower()
    host = urlparse(text).netloc.lower() if "://" in text else ""
    if host.startswith("nar.") or ".nar." in host:
        return "nar"
    if host in {"race.netkeiba.com", "race.sp.netkeiba.com"}:
        return "jra"
    if "netkeiba_race_nar" in lower or "地方競馬" in text or re.search(r"\bnar\b", lower):
        return "nar"
    if "中央競馬" in text or re.search(r"\bjra\b", lower):
        return "jra"
    return ""


def _extract_attr(tag: str, attr_name: str) -> str:
    match = re.search(
        rf"""\b{re.escape(attr_name)}\s*=\s*(['"])(.*?)\1""",
        tag,
        flags=re.I | re.S,
    )
    return html.unescape(match.group(2).strip()) if match else ""


def _clean_text(value: str) -> str:
    without_tags = re.sub(r"<[^>]+>", " ", value or "")
    return re.sub(r"\s+", " ", html.unescape(without_tags)).strip()


def _contains_any(text: str, needles: tuple[str, ...]) -> bool:
    return any(needle in text for needle in needles)


def _dedupe(values: Iterable[str]) -> list[str]:
    return list(dict.fromkeys(value for value in values if value))

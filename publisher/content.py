from __future__ import annotations

import copy
import re
from collections import defaultdict
from typing import Any, Iterable, Mapping

from .jockey import jockey_display

MARK_ORDER = ("◎", "○", "▲", "△", "☆")
BANNED_WORDS = ("絶対", "鉄板", "必勝", "確実")
INTERNAL_PATTERNS = (r"[＋+]\s*印[◎○▲△☆]", r"[＋+]\s*今回\d+位", r"[＋+]\s*(?:上位|中位|下位)帯", r"印[◎○▲△☆]をプラス材料")
DEFAULT_NOTE_INTRO = """🐴 KEIBA LABへようこそ。

自作競馬AI「KEIBA LAB」による予想と、主のつぶやきです。

能力・近走・展開・コース・調教など、
いろんな材料をAIが全頭比較。

数字だけでは見えにくいレースのポイントを、
主と一緒にゆるく考察しています。

予想はレース前の時点で保存。
当たった日も外れた日も、そのまま残して検証していきます🐾

今日も競馬を楽しみながら、
AIと一緒に予想していきます。"""
DEFAULT_PINNED_POST = """🐴 KEIBA LAB｜AI競馬予想

自作競馬AIの予想を公開検証しています。

能力・近走・展開・コース・調教などから全頭比較。
予想はレース前に保存し、
的中・不的中に関係なくそのまま検証。

全レースの詳しい予想と考察はnoteへ🐾"""

_CIRCLED = {1: "①", 2: "②", 3: "③", 4: "④", 5: "⑤", 6: "⑥", 7: "⑦", 8: "⑧", 9: "⑨", 10: "⑩", 11: "⑪", 12: "⑫", 13: "⑬", 14: "⑭", 15: "⑮", 16: "⑯", 17: "⑰", 18: "⑱"}


def text(value: Any) -> str:
    return "" if value is None else str(value).strip()


def _number_value(value: Any) -> int:
    match = re.search(r"\d+", text(value))
    return int(match.group()) if match else 999


def race_number(race: Mapping[str, Any]) -> str:
    value = text(race.get("race_number"))
    if value:
        return value if value.upper().endswith("R") else f"{value}R"
    race_id = text(race.get("race_id"))
    return f"{int(race_id[-2:])}R" if race_id[-2:].isdigit() else ""


def is_debut_race(race: Mapping[str, Any]) -> bool:
    candidates = [race.get("race_name"), race.get("class"), race.get("race_class")]
    info = race.get("mobile_snapshot", {}).get("race_info", {}) if isinstance(race.get("mobile_snapshot"), Mapping) else {}
    candidates.extend([info.get("race_name"), info.get("class")])
    return any(re.search(r"新馬|メイクデビュー", text(value)) for value in candidates)


def group_by_venue(snapshot: Mapping[str, Any], *, exclude_debut: bool = True) -> dict[str, list[dict[str, Any]]]:
    grouped: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for source in snapshot.get("races") or []:
        race = copy.deepcopy(source)
        if exclude_debut and is_debut_race(race):
            continue
        grouped[text(race.get("venue")) or "会場不明"].append(race)
    for races in grouped.values():
        races.sort(key=lambda item: _number_value(race_number(item)))
    return dict(grouped)


def marked_horses(race: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    order = {mark: index for index, mark in enumerate(MARK_ORDER)}
    selected = [h for h in race.get("horses") or [] if text(h.get("mark")) in order]
    return sorted(selected, key=lambda h: (order[text(h.get("mark"))], _number_value(h.get("horse_no"))))


def marked_map(race: Mapping[str, Any]) -> dict[str, Mapping[str, Any]]:
    return {text(h.get("mark")): h for h in marked_horses(race)}


def value_horses(race: Mapping[str, Any]) -> list[Mapping[str, Any]]:
    return [h for h in race.get("horses") or [] if h.get("value_signal") is True or text(h.get("value_signal")).lower() in {"1", "true", "yes", "妙味あり", "あり", "○"}]


def _horse_label(horse: Mapping[str, Any]) -> str:
    return f"{text(horse.get('horse_no'))}番{text(horse.get('horse_name'))}".strip()


def _top_horse(race: Mapping[str, Any], rank_key: str) -> Mapping[str, Any] | None:
    candidates = [h for h in race.get("horses") or [] if _number_value(h.get(rank_key)) < 999]
    return min(candidates, key=lambda h: _number_value(h.get(rank_key)), default=None)


def _clean_material(value: Any) -> str:
    clean = re.sub(r"^[＋+○△▲☆\-－]\s*", "", text(value))
    return "" if re.fullmatch(r"印[◎○▲△☆]|今回\d+位|(?:上位|中位|下位)帯", clean) else clean


def _materials(horse: Mapping[str, Any]) -> list[str]:
    values: list[Any] = []
    for key in ("decision_material", "plus_materials", "minus_materials"):
        raw = horse.get(key)
        values.extend(raw if isinstance(raw, list) else ([raw] if raw else []))
    result: list[str] = []
    for value in values:
        for part in re.split(r"[／/、]", text(value)):
            clean = _clean_material(part)
            if clean and clean not in result:
                result.append(clean)
    return result


def _condition_phrases(horse: Mapping[str, Any], *, limit: int = 3) -> list[str]:
    phrases = [v for v in _materials(horse) if re.search(r"距離|コース|近走|展開|調教|指数|対戦|実績|適性", v)]
    extras = (
        horse.get("development"), horse.get("course_material"),
        f"状態は{text(horse.get('state'))}" if text(horse.get("state")) not in {"", "-", "不明"} else "",
        horse.get("interval"),
        f"調教{text(horse.get('training_short')).split('｜')[0].strip()}" if text(horse.get("training_short")) not in {"", "-", "対象外", "データなし"} else "",
        f"斤量{text(horse.get('weight'))}" if text(horse.get("weight")) not in {"", "-", "不明"} else "",
    )
    phrases.extend(text(v).lstrip("±+－- ") for v in extras if text(v) not in {"", "-", "なし", "不明"})
    result: list[str] = []
    for item in phrases:
        clean = _clean_material(item)
        if clean and clean not in result:
            result.append(clean)
    return result[:limit]


def _safe_sentence(value: str) -> str:
    clean = "\n".join(" ".join(line.split()) for line in text(value).splitlines())
    for word in BANNED_WORDS:
        clean = clean.replace(word, "有力")
    for pattern in INTERNAL_PATTERNS:
        clean = re.sub(pattern, "", clean)
    return clean


def race_commentary(race: Mapping[str, Any]) -> str:
    marks = marked_map(race)
    honmei, second, third = marks.get("◎"), marks.get("○"), marks.get("▲")
    ability = _top_horse(race, "ability_rank")
    if not honmei:
        return "保存済みの印と各馬の条件材料を比較したいレース。"
    seed = _number_value(race_number(race)) % 3
    condition = f"{text(race.get('surface'))}{text(race.get('distance'))}m" if text(race.get("distance")) else "今回条件"
    if ability and text(ability.get("horse_no")) == text(honmei.get("horse_no")):
        openings = (
            f"能力比較で先頭に立つ{_horse_label(honmei)}が◎。地力を素直に軸に据える構図だ。",
            f"{condition}で能力1位の{_horse_label(honmei)}を本線に取る。相手探しが中心になるレースと見た。",
            f"上位の地力を比べると{_horse_label(honmei)}が一歩リード。今回は能力上位をそのまま◎とした。",
        )
        opening = openings[seed]
    elif ability:
        opening = f"能力値では{_horse_label(ability)}が最上位だが、今回条件まで含めた総合評価は{_horse_label(honmei)}を◎に選んだ。能力順どおりではなく、適性と展開を比べたい一戦。"
    else:
        opening = f"今回条件を総合して{_horse_label(honmei)}を◎に据える。上位印同士の適性差が焦点になる。"
    facts = _condition_phrases(honmei, limit=3)
    rank = _number_value(honmei.get("current_evaluation_rank"))
    main = (f"{_horse_label(honmei)}は総合評価{rank}位。" if rank < 999 else f"{_horse_label(honmei)}は、") + (("、".join(facts) + "を材料に今回の中心として評価した。") if facts else "保存時点の総合比較で中心評価となった。")
    rivals = []
    for mark, horse in (("○", second), ("▲", third)):
        if horse:
            rival_facts = _condition_phrases(horse, limit=2)
            rivals.append(f"{mark}{_horse_label(horse)}は" + ("、".join(rival_facts) if rival_facts else "上位評価の一角"))
    endings = (
        "。◎との差だけでなく、展開ひとつで順序が替わる余地も見ておきたい。",
        f"。{condition}への対応と道中の位置取りを考えると、○▲も相手の有力候補になる。",
        "。本命一頭の評価だけで完結せず、この2頭がどこまで迫れるかをセットで考えたい。",
    )
    comparison = "。".join(rivals) + endings[seed] if rivals else ""
    paragraphs = [opening, main, comparison]
    if ability and text(ability.get("mark")) not in {"◎", "○", "▲"}:
        paragraphs.append(f"なお能力1位の{_horse_label(ability)}は最終印では評価を抑えた。能力値だけで決めない点がこのレースのポイントになる。")
    return "\n\n".join(_safe_sentence(p) for p in paragraphs if p)


def short_commentary(race: Mapping[str, Any]) -> str:
    marks = marked_map(race)
    honmei, second, third = marks.get("◎"), marks.get("○"), marks.get("▲")
    ability = _top_horse(race, "ability_rank")
    if not honmei:
        return "保存済みの印と今回条件を比較したいレース。"
    if ability and text(ability.get("horse_no")) != text(honmei.get("horse_no")):
        first = f"能力比較では{_horse_label(ability)}が上位も、今回条件を含め{_horse_label(honmei)}を中心視。"
    else:
        facts = [item for item in _condition_phrases(honmei, limit=4) if not re.search(r"劣勢|下降|懸念|不利|届かない", item)]
        variants = (
            f"能力と今回条件を総合し{_horse_label(honmei)}を中心視。",
            f"能力上位の{_horse_label(honmei)}を軸に、{facts[0] if facts else '相手関係'}も評価。",
            f"{_horse_label(honmei)}の地力を重視。{facts[0] if facts else '今回条件'}が焦点。",
        )
        first = variants[_number_value(race_number(race)) % 3]
    values = value_horses(race)
    if values:
        focus = next((h for h in values if h in (second, third)), values[0])
        second_line = f"妙味では{text(focus.get('mark')) or '注目'}{_horse_label(focus)}にも注目。"
    elif second and third:
        second_line = f"相手は○{_horse_label(second)}と▲{_horse_label(third)}を比較したい。"
    else:
        second_line = "展開と条件適性のかみ合わせが焦点。"
    return _safe_sentence(first + second_line)


def note_race_section(race: Mapping[str, Any], owner_comment: str = "") -> str:
    mark_lines = [f"{text(h.get('mark'))} {text(h.get('horse_no'))}番 {text(h.get('horse_name'))}" for h in marked_horses(race)]
    lines = [f"### {text(race.get('venue'))}{race_number(race)}", "", *(mark_lines or ["印：保存データ内に該当なし"]), "", "### 🔍 AIレース考察", "", race_commentary(race)]
    honmei = marked_map(race).get("◎")
    if honmei:
        facts = [text(honmei.get("ability_band")) or "-", text(honmei.get("ability_value")) or "-", f"能力{text(honmei.get('ability_rank')) or '-'}位", f"今回{text(honmei.get('current_evaluation_rank')) or '-'}位"]
        reasons = _condition_phrases(honmei, limit=4)
        jockey = jockey_display(honmei)
        if jockey.relationship == "changed": reasons.append(f"騎手は{jockey.text}へ変更")
        lines.extend(["", "### 🎯 注目馬", "", f"◎{_horse_label(honmei)}", "", " / ".join(facts), "", ("、".join(reasons) + "を総合し、最終的に本命とした。") if reasons else "保存時点の総合評価を根拠に本命とした。"])
    values = value_horses(race)
    if values:
        descriptions = []
        for horse in values:
            reasons = _condition_phrases(horse, limit=2)
            descriptions.append(f"{text(horse.get('mark'))}{_horse_label(horse)}は、{('、'.join(reasons) + 'が市場評価との比較材料。') if reasons else '保存時点で市場評価とのずれがある候補。'}買い推奨ではなく、相手候補として注目したい。")
        lines.extend(["", "### 💰 妙味あり", "", *descriptions])
    if text(owner_comment):
        lines.extend(["", "### 【主のひとこと】", "", text(owner_comment)])
    result = "\n".join(lines).rstrip()
    validate_public_content(race, result)
    return result


def note_article(venue: str, races: Iterable[Mapping[str, Any]], race_date: str, *, intro: str = DEFAULT_NOTE_INTRO, owner_comments: Mapping[str, str] | None = None) -> str:
    lines = [text(intro), "", f"## 🐎 {_month_day(race_date)} {venue}競馬｜全レースAI予想"]
    for race in races:
        lines.extend(["", note_race_section(race, (owner_comments or {}).get(text(race.get("race_id")), ""))])
    return "\n".join(lines).rstrip() + "\n"


def note_title(venue: str, race_date: str) -> str:
    return f"【{_month_day(race_date)} {venue}競馬｜KEIBA LAB AI全レース予想】"


def _month_day(value: str) -> str:
    match = re.search(r"\d{4}[-/]?(\d{1,2})[-/]?(\d{1,2})", text(value))
    return f"{int(match.group(1))}/{int(match.group(2))}" if match else text(value)


def x_weighted_length(body: str) -> int:
    urls = re.findall(r"https?://\S+", body)
    return len(re.sub(r"https?://\S+", "", body)) + 23 * len(urls)


def x_post(race: Mapping[str, Any], note_url: str = "") -> str:
    venue, rno = text(race.get("venue")), race_number(race)
    marked = marked_horses(race)
    mark_lines = []
    for horse in marked:
        number = _number_value(horse.get("horse_no"))
        mark_lines.append(f"{text(horse.get('mark'))}{_CIRCLED.get(number, text(horse.get('horse_no')))} {text(horse.get('horse_name'))}")
    lines = [f"【{venue}{rno}｜KEIBA LAB AI予想】", "", *(mark_lines or ["保存データ内に印情報なし"]), "", short_commentary(race)]
    if note_url:
        lines.extend(["", "🐴全レース予想・詳しいAI考察はnote👇", note_url])
    lines.extend(["", f"#{venue}{rno} #AI競馬予想"])
    body = _safe_sentence("\n".join(lines)).strip()
    if x_weighted_length(body) > 280:
        lines = lines[:len(marked) + 2] + ["", short_commentary(race).split("。")[0] + "。"] + (["", "🐴全レース予想はnote👇", note_url] if note_url else []) + ["", f"#{venue}{rno}"]
        body = _safe_sentence("\n".join(lines)).strip()
    if x_weighted_length(body) > 280:
        lines = [f"【{venue}{rno}｜KEIBA LAB AI予想】", "", *(mark_lines or ["保存データ内に印情報なし"])]
        if note_url:
            lines.extend(["", "🐴詳しいAI考察はnote👇", note_url])
        body = _safe_sentence("\n".join(lines)).strip()
    validate_public_content(race, body, require_top3=False)
    if x_weighted_length(body) > 280:
        raise ValueError(f"X投稿文が280文字を超えています: {x_weighted_length(body)}")
    return body


def validate_public_content(race: Mapping[str, Any], body: str, *, require_top3: bool = True) -> None:
    for pattern in INTERNAL_PATTERNS:
        if re.search(pattern, body):
            raise ValueError(f"内部表現が公開文に残っています: {pattern}")
    marks = marked_map(race)
    if require_top3:
        missing = [m for m in ("◎", "○", "▲") if marks.get(m) and text(marks[m].get("horse_name")) not in body]
        if missing:
            raise ValueError(f"考察に上位印が登場しません: {','.join(missing)}")
    for mark, horse in marks.items():
        name = re.escape(text(horse.get("horse_name")))
        wrong = re.search(rf"([◎○▲△☆])\s*\d*番?\s*{name}", body) if name else None
        if wrong and wrong.group(1) != mark:
            raise ValueError(f"最終印と本文が不一致です: {horse.get('horse_name')}")

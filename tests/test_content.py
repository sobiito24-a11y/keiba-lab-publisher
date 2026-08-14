from __future__ import annotations

import copy
import re

from publisher.content import (
    group_by_venue,
    is_debut_race,
    note_article,
    note_race_section,
    short_commentary,
    value_horses,
    x_post,
)
from publisher.snapshot import prediction_signature


def test_debut_exclusion_is_explicit_only(snapshot):
    normal = copy.deepcopy(snapshot["races"][0])
    debut = copy.deepcopy(normal)
    debut["race_id"] = "debut-1"
    debut["race_name"] = "2歳新馬"
    make = copy.deepcopy(normal)
    make["race_id"] = "debut-2"
    make["race_name"] = "メイクデビュー新潟"
    age_only = copy.deepcopy(normal)
    age_only["race_id"] = "age-only"
    age_only["race_name"] = "2歳特別"
    assert is_debut_race(debut)
    assert is_debut_race(make)
    assert not is_debut_race(age_only)
    grouped = group_by_venue({"races": [normal, debut, make, age_only]}, exclude_debut=True)
    assert sum(map(len, grouped.values())) == 2
    assert sum(map(len, group_by_venue({"races": [normal, debut, make, age_only]}, exclude_debut=False).values())) == 4


def test_venue_note_contains_all_races_and_brand(snapshot):
    races = group_by_venue(snapshot)["新潟"]
    note = note_article("新潟", races, "2026-08-09")
    assert "## 🐎 8/9 新潟競馬｜全レースAI予想" in note
    assert "🐴 KEIBA LABへようこそ。" in note
    assert "予想はレース前の時点で保存。" in note
    assert note.count("### 新潟") == len(races)


def test_each_x_post_is_race_specific_and_has_venue_url(snapshot):
    races = group_by_venue(snapshot)["新潟"][:3]
    posts = [x_post(race, "https://note.com/niigata") for race in races]
    assert len(posts) == len(set(posts))
    for race, post in zip(races, posts):
        assert f"【新潟{race['race_number']}｜KEIBA LAB AI予想】" in post
        assert "https://note.com/niigata" in post
        assert f"#新潟{race['race_number']} #AI競馬予想" in post


def test_note_url_is_venue_specific(snapshot):
    grouped = group_by_venue(snapshot)
    sapporo = x_post(grouped["札幌"][0], "https://note.example/sapporo")
    niigata = x_post(grouped["新潟"][0], "https://note.example/niigata")
    assert "sapporo" in sapporo and "niigata" not in sapporo
    assert "niigata" in niigata and "sapporo" not in niigata


def test_value_present_and_absent(snapshot):
    race = copy.deepcopy(snapshot["races"][0])
    for horse in race["horses"]:
        horse["value_signal"] = False
    assert value_horses(race) == []
    assert "💰 妙味あり" not in note_race_section(race)
    race["horses"][0]["value_signal"] = True
    assert value_horses(race) == [race["horses"][0]]
    assert "💰 妙味あり" in note_race_section(race)


def test_missing_marks_and_optional_materials_are_safe(snapshot):
    race = copy.deepcopy(snapshot["races"][0])
    for horse in race["horses"]:
        horse["mark"] = ""
        horse.pop("plus_materials", None)
        horse.pop("minus_materials", None)
        horse.pop("development", None)
        horse.pop("course_material", None)
        horse.pop("training_short", None)
        horse.pop("stable_comment_summary", None)
    note = note_race_section(race)
    post = x_post(race)
    assert "印：保存データ内に該当なし" in note
    assert "保存データ内に印情報なし" in post
    assert short_commentary(race)


def test_no_hype_words(snapshot):
    race = copy.deepcopy(snapshot["races"][0])
    race["horses"][0]["plus_materials"] = ["鉄板級という保存文字列"]
    generated = note_race_section(race) + x_post(race)
    assert not re.search(r"絶対|鉄板|必勝|確実", generated)


def test_content_does_not_mutate_snapshot(snapshot):
    before = prediction_signature(snapshot)
    race = snapshot["races"][0]
    note_race_section(race)
    x_post(race)
    assert prediction_signature(snapshot) == before

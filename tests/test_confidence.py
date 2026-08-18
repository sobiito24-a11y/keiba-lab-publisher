from __future__ import annotations

import copy

import pytest

from publisher.confidence import SHOW_PUBLIC_SS_DEFAULT, assess_confidence, assess_snapshot
from publisher.content import note_article, note_race_section, x_post, x_weighted_length
from publisher.operations import OPERATION_MODES, publication_candidate, visible_x_races
from publisher.snapshot import prediction_signature
from publisher.state import dump_state, load_state, new_state


def make_race(mode: str = "jra") -> dict:
    return {
        "race_id": "test-race-01", "race_mode": mode, "date": "2026-08-18", "venue": "新潟", "race_number": "1R",
        "distance": 1600, "surface": "芝", "race_name": "テスト競走",
        "horses": [
            {
                "horse_no": "1", "horse_name": "テストホース", "mark": "◎", "ability_rank": 1,
                "ability_value": 90.0, "ability_band": "A", "current_evaluation_rank": 1,
                "plus_materials": ["上位帯", "今回2位", "印○", "距離実績◎", "コース実績◎", "近走上昇", "調教B", "継続騎乗"],
                "minus_materials": [], "training_short": "B 動き良好", "jockey_change": "継続",
                "development": "好位確保・展開平均", "state": "上昇", "estimated_position": "先団 → 先団 → 先団",
            },
            {"horse_no": "2", "horse_name": "セカンド", "mark": "○", "ability_rank": 2, "ability_value": 82.0, "current_evaluation_rank": 2, "plus_materials": [], "minus_materials": []},
            {"horse_no": "3", "horse_name": "サード", "mark": "▲", "ability_rank": 3, "ability_value": 80.0, "current_evaluation_rank": 3, "plus_materials": [], "minus_materials": []},
            {"horse_no": "4", "horse_name": "フォース", "mark": "△", "ability_rank": 4, "ability_value": 78.0, "current_evaluation_rank": 4, "plus_materials": [], "minus_materials": []},
            {"horse_no": "5", "horse_name": "フィフス", "mark": "☆", "ability_rank": 5, "ability_value": 76.0, "current_evaluation_rank": 5, "plus_materials": [], "minus_materials": []},
        ],
    }


def test_jra_ss_internal_and_public_s():
    result = assess_confidence(make_race("jra"))
    assert result.internal_confidence_rank == "SS"
    assert result.public_confidence_rank == "S"
    assert SHOW_PUBLIC_SS_DEFAULT is False
    assert assess_confidence(make_race("jra"), show_public_ss=True).public_confidence_rank == "SS"


def test_jra_s_a_and_outside():
    s = make_race("jra"); s["horses"][0]["jockey_change"] = "乗替"; s["horses"][0]["minus_materials"] = ["乗替"]
    assert assess_confidence(s).internal_confidence_rank == "S"
    assert "乗り替わり" in assess_confidence(s).confidence_warning_materials
    a = make_race("jra"); a["horses"][0]["ability_value"] = 83.0
    assert assess_confidence(a).internal_confidence_rank == "A"
    outside = make_race("jra"); outside["horses"][0]["ability_rank"] = 3
    assert assess_confidence(outside).internal_confidence_rank == "対象外"


def test_nar_ss_s_a_and_outside():
    ss = make_race("nar"); ss["horses"][0]["ability_value"] = 95.0; ss["horses"][1]["ability_value"] = 80.0
    assert assess_confidence(ss).internal_confidence_rank == "SS"
    s = make_race("nar"); s["horses"][0]["ability_value"] = 87.0; s["horses"][0]["jockey_change"] = "乗替"; s["horses"][0]["minus_materials"] = ["乗替"]
    assert assess_confidence(s).internal_confidence_rank == "S"
    a = make_race("nar"); a["horses"][0]["ability_value"] = 83.0
    assert assess_confidence(a).internal_confidence_rank == "A"
    outside = make_race("nar"); outside["horses"][0]["ability_rank"] = 3
    assert assess_confidence(outside).internal_confidence_rank == "対象外"


def test_note_s_a_outside_and_warning():
    s = make_race(); s["horses"][0]["jockey_change"] = "乗替"; s["horses"][0]["minus_materials"] = ["乗替"]
    note = note_race_section(s)
    assert "**注目度：S**" in note and "### 【気になるポイント】" in note and "乗り替わり" in note
    a = make_race(); a["horses"][0]["ability_value"] = 83.0
    assert "**注目度：A**" in note_race_section(a) and "### 🐴 注目度S" not in note_race_section(a)
    outside = make_race(); outside["horses"][0]["ability_rank"] = 3
    assert "注目度：" not in note_race_section(outside)
    article = note_article("新潟", [s, a], "2026-08-18")
    assert "## 🐴 本日の注目レース" in article and "- 新潟1R" in article and "Aランクは各レース欄" in article


def test_x_only_s_and_280_limit():
    s = make_race(); body = x_post(s, "https://note.com/example")
    assert "🐴 注目度S" in body and x_weighted_length(body) <= 280
    a = make_race(); a["horses"][0]["ability_value"] = 83.0
    assert "注目度" not in x_post(a, "https://note.com/example")


@pytest.mark.parametrize("mode", [OPERATION_MODES[1], OPERATION_MODES[2]])
def test_busy_and_holiday_candidate_prioritizes_s(mode):
    a = make_race(); a["race_id"] = "a"; a["horses"][0]["ability_value"] = 83.0
    s = make_race(); s["race_id"] = "s"; s["horses"][0]["jockey_change"] = "乗替"; s["horses"][0]["minus_materials"] = ["乗替"]
    assert publication_candidate([a, s])["race_id"] == "s"
    assert visible_x_races(mode, [a, s], "s") == [s]


def test_weekday_keeps_all_races():
    races = [make_race(), make_race()]
    assert visible_x_races(OPERATION_MODES[0], races) == races


def test_state_fields_and_old_v3_compatibility():
    race = make_race("nar")
    race["horses"][0]["ability_value"] = 87.0; race["horses"][0]["jockey_change"] = "乗替"; race["horses"][0]["minus_materials"] = ["乗替"]
    state = new_state({"race_ids": [race["race_id"]], "immutable_prediction_sha256": "x"})
    state["confidence_ranks"] = assess_snapshot([race])
    restored = load_state(dump_state(state))
    row = restored["confidence_ranks"][race["race_id"]]
    assert restored["show_public_ss"] is False
    assert row["internal_confidence_rank"] == "S" and row["public_confidence_rank"] == "S"
    legacy = copy.deepcopy(state); legacy.pop("show_public_ss"); legacy.pop("confidence_ranks")
    restored_legacy = load_state(dump_state(legacy))
    assert restored_legacy["show_public_ss"] is False and restored_legacy["confidence_ranks"] == {}


def test_snapshot_is_never_changed_or_recalculated():
    race = make_race("jra"); snapshot = {"races": [race]}; before = prediction_signature(snapshot)
    assess_snapshot(snapshot["races"]); note_race_section(race); x_post(race)
    assert prediction_signature(snapshot) == before

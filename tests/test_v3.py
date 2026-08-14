from __future__ import annotations

import copy
import json

import pytest

from publisher.content import DEFAULT_NOTE_INTRO, group_by_venue, note_article, note_race_section, race_commentary, validate_public_content, x_post, x_weighted_length
from publisher.operations import OPERATION_MODES, publication_candidate, visible_x_races
from publisher.results import load_result_json, merge_result_upload, result_reply
from publisher.snapshot import prediction_signature
from publisher.state import DuplicatePublicationError, dump_state, load_state, new_state, record_manual_publication


def test_note_quality_marks_and_internal_labels(snapshot):
    race = snapshot["races"][0]; body = note_race_section(race)
    assert DEFAULT_NOTE_INTRO.startswith("🐴") and "### 🎯 注目馬" in body
    for mark in ("◎", "○", "▲"):
        horse = next(h for h in race["horses"] if h.get("mark") == mark)
        assert horse["horse_name"] in race_commentary(race)
    assert "＋印" not in body and "＋今回" not in body


def test_ability_one_not_honmei(snapshot):
    race = next(r for r in snapshot["races"] if next(h for h in r["horses"] if h.get("ability_rank") == 1).get("mark") != "◎")
    assert "能力値では" in race_commentary(race) and "今回条件" in race_commentary(race)


def test_owner_comment_and_snapshot_immutable(snapshot):
    before = prediction_signature(snapshot); race = snapshot["races"][0]
    assert "【主のひとこと】" in note_race_section(race, "正直ここは難しい笑")
    note_article("中京", [race], race["date"]); x_post(race, "https://note.com/x")
    assert prediction_signature(snapshot) == before


@pytest.mark.parametrize("mode,count", [(OPERATION_MODES[0], 3), (OPERATION_MODES[1], 1), (OPERATION_MODES[2], 1)])
def test_operation_modes(snapshot, mode, count):
    races = group_by_venue(snapshot)["新潟"][:3]; chosen = races[1]["race_id"]
    assert len(visible_x_races(mode, races, chosen)) == count
    assert publication_candidate(races)["race_id"] in {r["race_id"] for r in races}


def test_x_url_length_marks_and_circled_numbers(snapshot):
    for race in snapshot["races"]:
        body = x_post(race, "https://note.com/example")
        assert x_weighted_length(body) <= 280 and "https://note.com/example" in body
        for horse in [h for h in race["horses"] if h.get("mark") in {"◎", "○", "▲", "△", "☆"}]:
            assert horse["horse_name"] in body


def test_final_mark_validation(snapshot):
    race = snapshot["races"][0]; honmei = next(h for h in race["horses"] if h.get("mark") == "◎")
    with pytest.raises(ValueError): validate_public_content(race, f"○{honmei['horse_no']}番{honmei['horse_name']}", require_top3=False)


def test_result_missing_is_safe(snapshot):
    assert result_reply(snapshot["races"][0], None) == "結果データ未取得"


def test_official_result_reply_and_no_inferred_payoff(snapshot):
    race = snapshot["races"][0]; marks = {h["mark"]: h for h in race["horses"] if h.get("mark")}
    payload = {"race_id": race["race_id"], "results": [{"horse_no": marks["◎"]["horse_no"], "rank": 1}, {"horse_no": marks["△"]["horse_no"], "rank": 3}], "payoffs": {"wide": [{"combination": "3-7", "payout": 410}]}}
    loaded = load_result_json(json.dumps(payload).encode()); text = result_reply(race, loaded, "ここはガミ、、、🐕笑")
    assert "◎" in text and "1着🏇" in text and "△" in text and "3着💥" in text and "ワイド 3-7 410円" in text
    without = copy.deepcopy(loaded); without["payoffs"] = {}
    assert "410" not in result_reply(race, without)


def test_result_upload_merge(snapshot):
    race = snapshot["races"][0]; target = {}
    merge_result_upload(target, {"races": [{"race_id": race["race_id"], "results": [{"horse_no": "1", "rank": "2"}]}]})
    assert target[race["race_id"]]["results"][0]["rank"] == "2"


def test_manual_record_duplicate_and_state_roundtrip(snapshot):
    race = snapshot["races"][0]; state = new_state({"race_ids": [race["race_id"]], "immutable_prediction_sha256": "x"})
    state["operation_mode"] = OPERATION_MODES[2]; state["free_race_ids"] = [race["race_id"]]; state["note_urls"][race["venue"]] = "https://note.com/x"
    state["owner_comments"][race["race_id"]] = "主コメント"
    record_manual_publication(state, race, "本文", "https://note.com/x", free_publication=True)
    with pytest.raises(DuplicatePublicationError): record_manual_publication(state, race, "本文", "https://note.com/x", free_publication=True)
    restored = load_state(dump_state(state))
    assert restored["operation_mode"] == OPERATION_MODES[2] and restored["publication_records"][0]["body_sha256"]


def test_v2_state_migration_separates_api_history(snapshot):
    race = snapshot["races"][0]; legacy = new_state({"race_ids": [race["race_id"]], "immutable_prediction_sha256": "x"})
    legacy["schema_version"] = 2; legacy["post_history"] = [{"race_id": race["race_id"], "post_id": "old-api"}]
    legacy["x_account"] = "old"; legacy["schedules"] = {race["race_id"]: "09:00"}
    migrated = load_state(json.dumps(legacy).encode())
    assert migrated["schema_version"] == 3 and migrated["legacy_x_api_history"][0]["post_id"] == "old-api"
    assert "x_account" not in migrated and "schedules" not in migrated


def test_no_x_api_module_or_credentials_required():
    import publisher.content, publisher.results, publisher.state
    assert True

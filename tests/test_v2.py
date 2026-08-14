from __future__ import annotations

import copy

import pytest

from publisher.content import DEFAULT_NOTE_INTRO, group_by_venue, note_article, note_race_section, race_commentary, validate_public_content, x_post, x_weighted_length
from publisher.posting import PostingError, XApiClient, XCredentials, post_to_x, validate_post_prerequisites, verify_account
from publisher.snapshot import prediction_signature
from publisher.state import DuplicatePostError, ensure_can_post, new_state, record_post


def test_note_intro_top3_and_no_internal_labels(snapshot):
    race = snapshot["races"][0]
    body = note_race_section(race)
    assert DEFAULT_NOTE_INTRO.startswith("🐴")
    for mark in ("◎", "○", "▲"):
        horse = next(h for h in race["horses"] if h.get("mark") == mark)
        assert horse["horse_name"] in race_commentary(race)
    assert "＋印" not in body and "＋今回" not in body


def test_ability_one_not_honmei_explains_reversal(snapshot):
    race = next(r for r in snapshot["races"] if next(h for h in r["horses"] if h.get("ability_rank") == 1).get("mark") != "◎")
    assert "能力値では" in race_commentary(race)
    assert "今回条件" in race_commentary(race)


def test_owner_comment_and_jra_nar(snapshot, nar_keiba):
    race = snapshot["races"][0]
    assert "【主のひとこと】" in note_race_section(race, "ここは気になる")
    assert "中京" in note_article("中京", [race], race["date"])


def test_x_length_url_and_snapshot_immutable(snapshot):
    before = prediction_signature(snapshot)
    for race in snapshot["races"]:
        body = x_post(race, "https://note.com/example")
        assert x_weighted_length(body) <= 280
        assert "https://note.com/example" in body
    assert prediction_signature(snapshot) == before


def test_final_mark_validation_detects_conflict(snapshot):
    race = snapshot["races"][0]
    honmei = next(h for h in race["horses"] if h.get("mark") == "◎")
    with pytest.raises(ValueError):
        validate_public_content(race, f"○{honmei['horse_no']}番{honmei['horse_name']}", require_top3=False)


class Response:
    def __init__(self, status, payload): self.status_code, self.payload, self.text = status, payload, ""
    def json(self): return self.payload


class Session:
    def __init__(self): self.posts = []
    def get(self, *args, **kwargs): return Response(200, {"data": {"id": "1", "username": "keiba_lab_ai", "name": "KEIBA LAB"}})
    def post(self, *args, **kwargs): self.posts.append(kwargs["json"]); return Response(201, {"data": {"id": "999"}})


def test_x_connection_dry_run_success_and_wrong_account():
    creds = XCredentials("a", "b", "c", "d")
    client = XApiClient(creds, session=Session())
    assert verify_account(client, "@keiba_lab_ai")["username"] == "keiba_lab_ai"
    with pytest.raises(PostingError): verify_account(client, "other")
    assert post_to_x(client, body="preview", dry_run=True) == "dry-run"
    assert client.session.posts == []
    assert post_to_x(client, body="real", dry_run=False) == "999"
    with pytest.raises(PostingError): XCredentials.from_mapping({})
    with pytest.raises(PostingError): validate_post_prerequisites(note_url="", account="keiba_lab_ai")
    with pytest.raises(PostingError): validate_post_prerequisites(note_url="https://note.com/x", account="")


def test_post_history_all_fields_and_duplicate(snapshot):
    state = new_state({"race_ids": ["r1"]})
    record_post(state, "r1", "keiba_lab_ai", "x_race", "投稿済", body="hello", post_id="999", note_url="https://note.com/x")
    row = state["post_history"][0]
    assert row["post_id"] == "999" and row["body_sha256"] and row["note_url"]
    with pytest.raises(DuplicatePostError): ensure_can_post(state, "r1", "keiba_lab_ai", "x_race")
    record_post(state, "r2", "keiba_lab_ai", "x_race", "投稿失敗", body="bad", message="rate limit", http_status=429, retryable=True)
    failed = state["post_history"][-1]
    assert failed["http_status"] == 429 and failed["retryable"] is True


def test_generated_posts_are_not_identical(snapshot):
    races = group_by_venue(snapshot)["新潟"][:3]
    assert len({x_post(r, "https://note.com/x") for r in races}) == 3

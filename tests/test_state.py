from __future__ import annotations

import copy

import pytest

from publisher.snapshot import prediction_signature
from publisher.state import (
    DuplicatePublicationError,
    PublisherStateError,
    dump_state,
    ensure_can_record_publication,
    load_state,
    new_state,
    record_manual_publication,
)


def test_publisher_save_roundtrip_and_manual_edits(loaded):
    state = new_state(loaded.source_info)
    state["note_urls"]["新潟"] = "https://note.example/niigata"
    state["note_drafts"]["新潟"] = "手動編集したnote原稿"
    race_id = loaded.source_info["race_ids"][0]
    state["x_drafts"][race_id] = "手動編集したX原稿"
    state["x_targets"][race_id] = False
    state["operation_mode"] = "🏃 忙しい日"
    state["free_race_ids"] = [race_id]
    restored = load_state(dump_state(state), source_info=loaded.source_info)
    assert restored["note_drafts"] == state["note_drafts"]
    assert restored["x_drafts"] == state["x_drafts"]
    assert restored["x_targets"] == state["x_targets"]
    assert restored["operation_mode"] == state["operation_mode"]
    assert restored["free_race_ids"] == state["free_race_ids"]


def test_publisher_state_does_not_embed_or_change_prediction(loaded):
    snapshot = loaded.snapshot
    before = prediction_signature(snapshot)
    state = new_state(loaded.source_info)
    payload = dump_state(state)
    assert b'"horses"' not in payload
    assert prediction_signature(snapshot) == before
    assert loaded.immutable_prediction_sha256 == before


def test_double_post_prevention(loaded):
    state = new_state(loaded.source_info)
    race_id = loaded.source_info["race_ids"][0]
    race = {"race_id": race_id, "date": "2026-08-09", "venue": "中京", "race_number": "1R"}
    record_manual_publication(state, race, "本文", "https://note.com/x", free_publication=True)
    with pytest.raises(DuplicatePublicationError): ensure_can_record_publication(state, race_id)
    with pytest.raises(DuplicatePublicationError): record_manual_publication(state, race, "本文", "https://note.com/x", free_publication=True)


def test_state_rejects_wrong_source(loaded):
    state = new_state(loaded.source_info)
    wrong = copy.deepcopy(loaded.source_info)
    wrong["immutable_prediction_sha256"] = "0" * 64
    with pytest.raises(PublisherStateError, match="異なる"):
        load_state(dump_state(state), source_info=wrong)


@pytest.mark.parametrize("data", [b"", b"{}", b"not json"])
def test_invalid_state_rejected(data):
    with pytest.raises(PublisherStateError):
        load_state(data)


def test_state_duplicate_race_id_rejected(loaded):
    state = new_state(loaded.source_info)
    state["source"]["race_ids"].append(state["source"]["race_ids"][0])
    with pytest.raises(PublisherStateError, match="重複"):
        dump_state(state)

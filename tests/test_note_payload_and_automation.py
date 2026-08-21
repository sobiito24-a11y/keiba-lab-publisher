from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from publisher.content import group_by_venue
from publisher.note_automation import NoteDraftAutomationError, NoteLoginRequiredError, save_note_draft
from publisher.note_payload import JRA_HEADING_IMAGE, NAR_HEADING_IMAGE, build_prediction_note_payload
from publisher.snapshot import load_prediction, prediction_signature


class FakeNoteDriver:
    def __init__(self, *, fail_step: str = ""):
        self.fail_step = fail_step
        self.steps: list[str] = []
        self.title = ""
        self.body = ""
        self.tags: tuple[str, ...] = ()
        self.image_path: Path | None = None

    def _step(self, name: str) -> None:
        self.steps.append(name)
        if self.fail_step == name:
            if name == "ensure_logged_in":
                raise NoteLoginRequiredError()
            raise NoteDraftAutomationError("fake failure", step=name)

    def open_editor(self) -> None:
        self._step("open_editor")

    def ensure_logged_in(self) -> None:
        self._step("ensure_logged_in")

    def set_heading_image(self, image_path: Path | None) -> None:
        self._step("set_heading_image")
        self.image_path = image_path

    def set_title(self, title: str) -> None:
        self._step("set_title")
        self.title = title

    def set_body(self, body: str) -> None:
        self._step("set_body")
        self.body = body

    def set_tags(self, tags: tuple[str, ...]) -> None:
        self._step("set_tags")
        self.tags = tags

    def save_draft(self) -> str:
        self._step("save_draft")
        return "https://note.com/keiba_lab_ai/n/draft-test"

    def close(self) -> None:
        self.steps.append("close")


def test_jra_prediction_payload_uses_snapshot_body_and_fixed_image(snapshot):
    before = prediction_signature(snapshot)
    races = group_by_venue(snapshot)["新潟"]
    payload = build_prediction_note_payload("新潟", races, "2026-08-09")
    assert payload.article_type == "JRA"
    assert payload.heading_image_path == JRA_HEADING_IMAGE
    assert payload.heading_image_path.exists()
    assert "中央競馬" in payload.tags
    assert "KEIBALAB" in payload.tags
    assert "## 🐎 8/9 新潟競馬｜全レースAI予想" in payload.body
    assert prediction_signature(snapshot) == before


def test_nar_prediction_payload_uses_fixed_local_image(nar_keiba):
    loaded = load_prediction(nar_keiba)
    snapshot = loaded.snapshot
    payload = build_prediction_note_payload("大井", snapshot["races"], "2026-08-14")
    assert payload.article_type == "NAR"
    assert payload.heading_image_path == NAR_HEADING_IMAGE
    assert payload.heading_image_path.exists()
    assert "地方競馬" in payload.tags
    assert "KEIBALAB" in payload.tags
    assert prediction_signature(snapshot) == loaded.immutable_prediction_sha256


def test_note_payload_generation_never_calls_predictors(keiba_data, monkeypatch):
    calls = {"jra": 0, "nar": 0}

    def forbidden(mode):
        def inner(*args, **kwargs):
            calls[mode] += 1
            raise AssertionError("predictor was called")
        return inner

    monkeypatch.setitem(sys.modules, "shared_dashboard_core.core.jra_predictor", SimpleNamespace(predict_jra=forbidden("jra")))
    monkeypatch.setitem(sys.modules, "shared_dashboard_core.core.nar_predictor", SimpleNamespace(predict_nar=forbidden("nar")))
    loaded = load_prediction(keiba_data)
    snapshot = loaded.snapshot
    for venue, races in group_by_venue(snapshot).items():
        build_prediction_note_payload(venue, races, snapshot["races"][0]["date"])
    assert calls == {"jra": 0, "nar": 0}


def test_note_draft_save_success_uses_driver_without_publication(snapshot):
    races = group_by_venue(snapshot)["札幌"]
    payload = build_prediction_note_payload("札幌", races, "2026-08-09")
    driver = FakeNoteDriver()
    result = save_note_draft(payload, driver=driver)
    assert result.status == "draft_saved"
    assert result.url.endswith("draft-test")
    assert driver.title == payload.title
    assert driver.body == payload.body
    assert driver.tags == payload.tags
    assert driver.image_path == payload.heading_image_path
    assert "save_draft" in driver.steps


def test_note_login_required_stops_safely(snapshot):
    races = group_by_venue(snapshot)["札幌"]
    payload = build_prediction_note_payload("札幌", races, "2026-08-09")
    driver = FakeNoteDriver(fail_step="ensure_logged_in")
    with pytest.raises(NoteLoginRequiredError):
        save_note_draft(payload, driver=driver)
    assert driver.steps == ["open_editor", "ensure_logged_in"]


def test_note_ui_change_stops_safely(snapshot):
    races = group_by_venue(snapshot)["札幌"]
    payload = build_prediction_note_payload("札幌", races, "2026-08-09")
    driver = FakeNoteDriver(fail_step="set_title")
    with pytest.raises(NoteDraftAutomationError, match="fake failure"):
        save_note_draft(payload, driver=driver)
    assert "save_draft" not in driver.steps

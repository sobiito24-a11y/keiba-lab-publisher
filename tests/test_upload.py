from __future__ import annotations

import pytest

import app
from publisher.content import group_by_venue, note_article
from publisher.snapshot import load_prediction, prediction_signature
from publisher.upload import KeibaUploadError, read_uploaded_keiba
from shared_dashboard_core.core.prediction_snapshot import KeibaSnapshotError


class FakeUploadedFile:
    def __init__(self, name: str, data: bytes) -> None:
        self.name = name
        self._data = data

    def getvalue(self) -> bytes:
        return self._data


class FakeSessionState(dict):
    def __getattr__(self, name):
        try:
            return self[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def __setattr__(self, name, value):
        self[name] = value


def test_uploaded_file_bytes_loads_existing_jra_keiba(keiba_data):
    upload = read_uploaded_keiba(FakeUploadedFile("2026-08-09_JRA_all_venues.keiba", keiba_data))
    loaded = load_prediction(upload.data)
    snapshot = loaded.snapshot
    assert len(snapshot["races"]) == 32
    assert {race["race_mode"] for race in snapshot["races"]} == {"jra"}
    assert set(group_by_venue(snapshot)) == {"札幌", "新潟", "中京"}


def test_uploaded_file_bytes_loads_nar_keiba(nar_keiba):
    upload = read_uploaded_keiba(FakeUploadedFile("2026-08-14_NAR_大井.keiba", nar_keiba))
    loaded = load_prediction(upload.data)
    snapshot = loaded.snapshot
    assert len(snapshot["races"]) == 1
    assert snapshot["races"][0]["race_mode"] == "nar"
    assert snapshot["races"][0]["venue"] == "大井"


def test_uploaded_file_accepts_iphone_style_filename(keiba_data):
    upload = read_uploaded_keiba(FakeUploadedFile("予想 2026-08-23 JRA.keiba", keiba_data))
    assert upload.filename == "予想 2026-08-23 JRA.keiba"
    assert len(upload.sha256) == 64


def test_non_keiba_upload_stops_before_snapshot_load(keiba_data):
    with pytest.raises(KeibaUploadError, match=".keiba"):
        read_uploaded_keiba(FakeUploadedFile("snapshot.zip", keiba_data))


@pytest.mark.parametrize("data", [b"", b"not-a-keiba-archive"])
def test_corrupt_or_empty_keiba_upload_stops_safely(data):
    upload_file = FakeUploadedFile("broken.keiba", data)
    if data:
        upload = read_uploaded_keiba(upload_file)
        with pytest.raises(KeibaSnapshotError):
            load_prediction(upload.data)
    else:
        with pytest.raises(KeibaUploadError):
            read_uploaded_keiba(upload_file)


def test_same_uploaded_file_does_not_reload_prediction(monkeypatch, keiba_data):
    calls = {"load": 0}
    state = FakeSessionState()

    class FakeLoaded:
        source_info = {"race_ids": ["dummy"]}
        snapshot = {"races": []}

    def fake_load_prediction(data: bytes):
        calls["load"] += 1
        return FakeLoaded()

    monkeypatch.setattr(app.st, "session_state", state)
    monkeypatch.setattr(app, "load_prediction", fake_load_prediction)
    monkeypatch.setattr(app, "new_state", lambda source_info: {})
    monkeypatch.setattr(app, "result_map_from_snapshot", lambda snapshot: {})
    app._get_loaded(keiba_data)
    app._get_loaded(keiba_data)
    assert calls["load"] == 1


def test_prediction_signature_unchanged_after_upload_note_generation(keiba_data):
    upload = read_uploaded_keiba(FakeUploadedFile("2026-08-09_JRA_all_venues.keiba", keiba_data))
    loaded = load_prediction(upload.data)
    snapshot = loaded.snapshot
    before = prediction_signature(snapshot)
    for venue, races in group_by_venue(snapshot).items():
        body = note_article(venue, races, snapshot["races"][0]["date"])
        assert venue in body
    assert prediction_signature(snapshot) == before
    assert prediction_signature(loaded.snapshot) == loaded.immutable_prediction_sha256

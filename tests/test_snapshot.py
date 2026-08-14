from __future__ import annotations

import copy
import hashlib
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from publisher.content import group_by_venue, note_article, x_post
from publisher.snapshot import load_prediction, prediction_signature
from shared_dashboard_core.core.prediction_snapshot import KeibaSnapshotError, UnsupportedSchemaError

from .conftest import archive_for


def test_uses_exact_dashboard_snapshot_loader_copy():
    publisher_copy = Path(__file__).parents[1] / "shared_dashboard_core" / "core" / "prediction_snapshot.py"
    # SHA-256 recorded from the Dashboard source used for this release.
    assert hashlib.sha256(publisher_copy.read_bytes()).hexdigest() == "4c425dff5f955d941af4da0c6a9a00794c448cd1f792af5d0a7ddb68d6cf1b90"


def test_normal_keiba_and_three_real_venues(loaded):
    snapshot = loaded.snapshot
    assert snapshot["schema_version"] == 1
    assert len(snapshot["races"]) == 32
    assert set(group_by_venue(snapshot)) == {"札幌", "新潟", "中京"}
    loaded.assert_unchanged()


def test_jra_and_nar_supported(loaded, nar_keiba):
    assert {race["race_mode"] for race in loaded.snapshot["races"]} == {"jra"}
    nar = load_prediction(nar_keiba)
    assert nar.snapshot["races"][0]["race_mode"] == "nar"
    assert nar.snapshot["races"][0]["venue"] == "大井"


def test_generation_never_calls_predictors(keiba_data, monkeypatch):
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
        note_article(venue, races, snapshot["races"][0]["date"])
        for race in races:
            x_post(race)
    assert calls == {"jra": 0, "nar": 0}


def test_prediction_facts_are_immutable_during_generation(loaded):
    snapshot = loaded.snapshot
    before = prediction_signature(snapshot)
    for venue, races in group_by_venue(snapshot).items():
        note_article(venue, races, snapshot["races"][0]["date"])
        for race in races:
            x_post(race, "https://note.com/keiba-lab")
    assert prediction_signature(snapshot) == before
    assert prediction_signature(loaded.snapshot) == loaded.immutable_prediction_sha256


@pytest.mark.parametrize("data", [b"", b"not-a-zip"])
def test_empty_or_corrupt_keiba_rejected(data):
    with pytest.raises(KeibaSnapshotError):
        load_prediction(data)


@pytest.mark.parametrize("version", [0, 2, 99])
def test_unsupported_schema_rejected(snapshot, version):
    with pytest.raises(UnsupportedSchemaError):
        load_prediction(archive_for(snapshot, schema_version=version))


def test_duplicate_race_id_rejected(snapshot):
    broken = copy.deepcopy(snapshot)
    broken["races"].append(copy.deepcopy(broken["races"][0]))
    with pytest.raises(KeibaSnapshotError, match="重複"):
        load_prediction(archive_for(broken))

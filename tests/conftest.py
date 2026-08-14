from __future__ import annotations

import copy
import hashlib
import io
import json
import zipfile
from pathlib import Path

import pytest

from publisher.snapshot import load_prediction
from shared_dashboard_core.core.prediction_snapshot import build_event_snapshot, keiba_bytes


FIXTURE = Path(__file__).parent / "fixtures" / "2026-08-09_JRA_all_venues.keiba"


@pytest.fixture(scope="session")
def keiba_data() -> bytes:
    return FIXTURE.read_bytes()


@pytest.fixture(scope="session")
def loaded(keiba_data):
    return load_prediction(keiba_data)


@pytest.fixture()
def snapshot(loaded):
    return loaded.snapshot


@pytest.fixture()
def nar_keiba(snapshot) -> bytes:
    race = copy.deepcopy(snapshot["races"][0])
    race["race_id"] = "202647081401"
    race["race_mode"] = "nar"
    race["venue"] = "大井"
    race["date"] = "2026-08-14"
    race["prediction_result"]["race_mode"] = "nar"
    race["prediction_result"]["race_info"]["race_id"] = race["race_id"]
    race["prediction_result"]["race_info"]["venue"] = "大井"
    mobile_info = race.get("mobile_snapshot", {}).get("race_info", {})
    mobile_info["race_id"] = race["race_id"]
    mobile_info["venue"] = "大井"
    return keiba_bytes(build_event_snapshot([race]))


def archive_for(snapshot: dict, *, schema_version=None) -> bytes:
    payload = copy.deepcopy(snapshot)
    if schema_version is not None:
        payload["schema_version"] = schema_version
    body = json.dumps(payload, ensure_ascii=False, separators=(",", ":"), sort_keys=True).encode("utf-8")
    manifest = {
        "format": "keiba-prediction-snapshot",
        "schema_version": payload.get("schema_version"),
        "app_version": payload.get("app_version"),
        "prediction_logic_version": payload.get("prediction_logic_version"),
        "saved_at": payload.get("saved_at"),
        "race_count": len(payload.get("races") or []),
        "snapshot_sha256": hashlib.sha256(body).hexdigest(),
    }
    output = io.BytesIO()
    with zipfile.ZipFile(output, "w", zipfile.ZIP_DEFLATED) as zf:
        zf.writestr("manifest.json", json.dumps(manifest, ensure_ascii=False))
        zf.writestr("snapshot.json", body)
    return output.getvalue()


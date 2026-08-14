from __future__ import annotations

import copy
import hashlib
import json
from dataclasses import dataclass
from typing import Any, Mapping

from shared_dashboard_core.core.prediction_snapshot import load_keiba


IMMUTABLE_PREDICTION_FIELDS = (
    "horse_no",
    "horse_name",
    "ability_value",
    "ability_rank",
    "ability_band",
    "current_evaluation_rank",
    "mark",
    "value_signal",
)


def canonical_json(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def prediction_signature(snapshot: Mapping[str, Any]) -> str:
    """Fingerprint only the saved prediction facts Publisher must never alter."""

    rows: list[dict[str, Any]] = []
    for race in snapshot.get("races") or []:
        horses = []
        for horse in race.get("horses") or []:
            horses.append({key: copy.deepcopy(horse.get(key)) for key in IMMUTABLE_PREDICTION_FIELDS})
        rows.append({"race_id": race.get("race_id"), "horses": horses})
    return hashlib.sha256(canonical_json(rows)).hexdigest()


@dataclass(frozen=True)
class LoadedPrediction:
    file_sha256: str
    snapshot_sha256: str
    immutable_prediction_sha256: str
    _snapshot: dict[str, Any]

    @property
    def snapshot(self) -> dict[str, Any]:
        """Return a copy so UI/editor operations cannot mutate the source Snapshot."""

        return copy.deepcopy(self._snapshot)

    @property
    def source_info(self) -> dict[str, Any]:
        return {
            "file_sha256": self.file_sha256,
            "snapshot_sha256": self.snapshot_sha256,
            "immutable_prediction_sha256": self.immutable_prediction_sha256,
            "format": self._snapshot.get("format"),
            "schema_version": self._snapshot.get("schema_version"),
            "app_version": self._snapshot.get("app_version"),
            "prediction_logic_version": self._snapshot.get("prediction_logic_version"),
            "prediction_created_at": self._snapshot.get("prediction_created_at"),
            "race_ids": [race.get("race_id") for race in self._snapshot.get("races") or []],
        }

    def assert_unchanged(self) -> None:
        current = prediction_signature(self._snapshot)
        if current != self.immutable_prediction_sha256:
            raise RuntimeError("Prediction Snapshotの予想事実が変更されています。")


def load_prediction(data: bytes) -> LoadedPrediction:
    """Load with Dashboard's canonical loader; never restore or run a predictor."""

    snapshot = load_keiba(data)
    frozen = copy.deepcopy(snapshot)
    return LoadedPrediction(
        file_sha256=hashlib.sha256(data).hexdigest(),
        snapshot_sha256=hashlib.sha256(canonical_json(frozen)).hexdigest(),
        immutable_prediction_sha256=prediction_signature(frozen),
        _snapshot=frozen,
    )


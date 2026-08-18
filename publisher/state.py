from __future__ import annotations

import copy
import hashlib
import json
from datetime import datetime
from typing import Any, Mapping

from .operations import OPERATION_MODES

PUBLISHER_FORMAT = "keiba-lab-publisher-state"
PUBLISHER_SCHEMA_VERSION = 3
STATUSES = ("未生成", "原稿生成済", "note URL登録済", "X投稿準備完了", "投稿済", "投稿失敗")


class PublisherStateError(ValueError): pass
class DuplicatePublicationError(PublisherStateError): pass
DuplicatePostError = DuplicatePublicationError


def new_state(source_info: Mapping[str, Any]) -> dict[str, Any]:
    now = datetime.now().isoformat(timespec="seconds")
    return {
        "format": PUBLISHER_FORMAT, "schema_version": 3, "created_at": now, "updated_at": now,
        "source": copy.deepcopy(dict(source_info)), "exclude_debut": True,
        "operation_mode": OPERATION_MODES[0], "publication_mode": "全レース無料", "free_race_ids": [],
        "note_urls": {}, "note_intro": "", "note_generated": {}, "note_drafts": {}, "owner_comments": {},
        "x_generated": {}, "x_drafts": {}, "x_targets": {}, "race_status": {},
        "publication_records": [], "result_data": {}, "result_comments": {}, "result_reply_drafts": {},
        "show_public_ss": False, "confidence_ranks": {},
        "legacy_x_api_history": [],
        "publication_sections": {"free": ["marks", "short_commentary"], "note": ["all_races", "detail_analysis"]},
    }


def dump_state(state: Mapping[str, Any]) -> bytes:
    payload = copy.deepcopy(dict(state)); payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    validate_state(payload)
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")


def load_state(data: bytes, *, source_info: Mapping[str, Any] | None = None) -> dict[str, Any]:
    try: state = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc: raise PublisherStateError("Publisher保存ファイルが破損しています。") from exc
    version = state.get("schema_version")
    if version in {1, 2}: state = _migrate_legacy(state)
    state = _with_defaults(state)
    validate_state(state)
    if source_info and state.get("source", {}).get("immutable_prediction_sha256") != source_info.get("immutable_prediction_sha256"):
        raise PublisherStateError("異なるPrediction Snapshot用のPublisher保存データです。")
    return copy.deepcopy(state)


def _migrate_legacy(state: Mapping[str, Any]) -> dict[str, Any]:
    migrated = copy.deepcopy(dict(state)); defaults = new_state(migrated.get("source") or {})
    old_history = migrated.pop("post_history", []) or []
    migrated["legacy_x_api_history"] = copy.deepcopy(old_history)
    for key, value in defaults.items(): migrated.setdefault(key, copy.deepcopy(value))
    migrated["schema_version"] = 3
    old_mode = migrated.get("publication_mode")
    if old_mode == "1R無料＋全レースnote": migrated["operation_mode"] = OPERATION_MODES[1]
    migrated.pop("x_account", None); migrated.pop("expected_x_account", None); migrated.pop("schedules", None)
    migrated.pop("posting_interval_minutes", None); migrated.pop("scheduler_active", None)
    return migrated


def _with_defaults(state: Mapping[str, Any]) -> dict[str, Any]:
    merged = copy.deepcopy(dict(state))
    defaults = new_state(merged.get("source") or {})
    for key, value in defaults.items():
        merged.setdefault(key, copy.deepcopy(value))
    # SS remains private until a future explicit release.
    if not isinstance(merged.get("show_public_ss"), bool):
        merged["show_public_ss"] = False
    if not isinstance(merged.get("confidence_ranks"), dict):
        merged["confidence_ranks"] = {}
    return merged


def validate_state(state: Mapping[str, Any]) -> None:
    if state.get("format") != PUBLISHER_FORMAT: raise PublisherStateError("Publisher保存形式が不正です。")
    if state.get("schema_version") != 3: raise PublisherStateError("未対応のPublisher schema_versionです。")
    if state.get("operation_mode") not in OPERATION_MODES: raise PublisherStateError("運用モードが不正です。")
    if not isinstance(state.get("show_public_ss", False), bool): raise PublisherStateError("SS公開設定が不正です。")
    if not isinstance(state.get("confidence_ranks", {}), Mapping): raise PublisherStateError("注目度保存形式が不正です。")
    race_ids = state.get("source", {}).get("race_ids") or []
    if len(race_ids) != len(set(race_ids)): raise PublisherStateError("race_idが重複しています。")


def ensure_can_record_publication(state: Mapping[str, Any], race_id: str) -> None:
    if any(str(row.get("race_id")) == str(race_id) for row in state.get("publication_records") or []):
        raise DuplicatePublicationError(f"{race_id} はすでにX投稿済みとして記録されています。")


def record_manual_publication(state: dict[str, Any], race: Mapping[str, Any], body: str, note_url: str, *, free_publication: bool) -> None:
    race_id = str(race.get("race_id") or ""); ensure_can_record_publication(state, race_id)
    now = datetime.now().isoformat(timespec="seconds")
    state.setdefault("publication_records", []).append({
        "date": str(race.get("date") or ""), "venue": str(race.get("venue") or ""), "race_id": race_id,
        "race_number": str(race.get("race_number") or ""), "x_body": body, "published_at": now,
        "note_url": note_url, "free_publication": bool(free_publication),
        "result": copy.deepcopy(state.get("result_data", {}).get(race_id) or {}),
        "result_comment": str(state.get("result_reply_drafts", {}).get(race_id) or ""),
        "body_sha256": hashlib.sha256(body.encode("utf-8")).hexdigest(),
    })
    state.setdefault("race_status", {})[race_id] = "投稿済"


# Compatibility names for legacy callers/tests. These never call X.
def ensure_can_post(state: Mapping[str, Any], race_id: str, account: str = "", post_type: str = "") -> None:
    ensure_can_record_publication(state, race_id)


def record_post(state: dict[str, Any], race_id: str, account: str, post_type: str, status: str, **kwargs: Any) -> None:
    if status != "投稿済":
        raise PublisherStateError("Ver.3ではAPI投稿失敗履歴を記録しません。")
    race = {"race_id": race_id, "date": "", "venue": "", "race_number": ""}
    record_manual_publication(state, race, str(kwargs.get("body") or ""), str(kwargs.get("note_url") or ""), free_publication=False)

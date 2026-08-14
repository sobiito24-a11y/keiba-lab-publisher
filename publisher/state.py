from __future__ import annotations

import copy
import json
from datetime import datetime
from typing import Any, Mapping


PUBLISHER_FORMAT = "keiba-lab-publisher-state"
PUBLISHER_SCHEMA_VERSION = 1
STATUSES = ("未生成", "原稿生成済", "note URL登録済", "X投稿準備完了", "投稿済", "投稿失敗")


class PublisherStateError(ValueError):
    pass


class DuplicatePostError(PublisherStateError):
    pass


def new_state(source_info: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "format": PUBLISHER_FORMAT,
        "schema_version": PUBLISHER_SCHEMA_VERSION,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "updated_at": datetime.now().isoformat(timespec="seconds"),
        "source": copy.deepcopy(dict(source_info)),
        "exclude_debut": True,
        "note_urls": {},
        "note_drafts": {},
        "x_drafts": {},
        "x_targets": {},
        "race_status": {},
        "schedules": {},
        "post_history": [],
        "x_account": "",
        "publication_sections": {
            "free": ["marks", "short_commentary"],
            "paid_reserved": ["detail_analysis", "ability_value", "course_development", "training", "stable_comment"],
        },
    }


def dump_state(state: Mapping[str, Any]) -> bytes:
    payload = copy.deepcopy(dict(state))
    payload["updated_at"] = datetime.now().isoformat(timespec="seconds")
    validate_state(payload)
    return json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True).encode("utf-8")


def load_state(data: bytes, *, source_info: Mapping[str, Any] | None = None) -> dict[str, Any]:
    if not data:
        raise PublisherStateError("Publisher保存ファイルが空です。")
    try:
        state = json.loads(data.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise PublisherStateError("Publisher保存ファイルが破損しています。") from exc
    validate_state(state)
    if source_info and state.get("source", {}).get("immutable_prediction_sha256") != source_info.get("immutable_prediction_sha256"):
        raise PublisherStateError("異なるPrediction Snapshot用のPublisher保存データです。")
    return copy.deepcopy(state)


def validate_state(state: Mapping[str, Any]) -> None:
    if state.get("format") != PUBLISHER_FORMAT:
        raise PublisherStateError("Publisher保存形式が不正です。")
    if state.get("schema_version") != PUBLISHER_SCHEMA_VERSION:
        raise PublisherStateError("未対応のPublisher schema_versionです。")
    if not isinstance(state.get("source"), Mapping):
        raise PublisherStateError("Prediction Snapshot識別情報がありません。")
    race_ids = state.get("source", {}).get("race_ids") or []
    if len(race_ids) != len(set(race_ids)):
        raise PublisherStateError("race_idが重複しています。")
    for value in (state.get("race_status") or {}).values():
        if value not in STATUSES:
            raise PublisherStateError(f"不正な投稿状態です: {value}")


def ensure_can_post(state: Mapping[str, Any], race_id: str, account: str, post_type: str) -> None:
    for entry in state.get("post_history") or []:
        if (
            str(entry.get("race_id")) == str(race_id)
            and str(entry.get("account")) == str(account)
            and str(entry.get("post_type")) == str(post_type)
            and entry.get("status") == "投稿済"
        ):
            raise DuplicatePostError(f"{race_id} は同じアカウント・投稿種別ですでに投稿済みです。")


def record_post(state: dict[str, Any], race_id: str, account: str, post_type: str, status: str, *, message: str = "") -> None:
    if status not in {"投稿済", "投稿失敗"}:
        raise PublisherStateError("投稿履歴のstatusが不正です。")
    if status == "投稿済":
        ensure_can_post(state, race_id, account, post_type)
    state.setdefault("post_history", []).append(
        {
            "race_id": str(race_id),
            "account": str(account),
            "post_type": str(post_type),
            "status": status,
            "posted_at": datetime.now().isoformat(timespec="seconds"),
            "message": message,
        }
    )
    state.setdefault("race_status", {})[str(race_id)] = status

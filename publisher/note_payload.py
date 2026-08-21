from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

from .content import note_article, note_title, text


PROJECT_ROOT = Path(__file__).resolve().parents[1]
NOTE_ASSET_DIR = PROJECT_ROOT / "assets" / "note"
JRA_HEADING_IMAGE = NOTE_ASSET_DIR / "central_keiba_prediction_1280x670.png"
NAR_HEADING_IMAGE = NOTE_ASSET_DIR / "local_keiba_prediction_1280x670.png"

PREDICTION_ARTICLE_TYPES = {"jra": "JRA", "nar": "NAR"}
READING_ARTICLE_TYPE = "読み物"
DEFAULT_BRAND_TAG = "KEIBALAB"


class NotePayloadError(ValueError):
    pass


@dataclass(frozen=True)
class NoteDraftPayload:
    title: str
    body: str
    tags: tuple[str, ...]
    article_type: str
    heading_image_path: Path | None = None
    source: Mapping[str, Any] = field(default_factory=dict)
    scheduled_at: str | None = None

    def validate(self) -> None:
        if not self.title.strip():
            raise NotePayloadError("noteタイトルが空です。")
        if not self.body.strip():
            raise NotePayloadError("note本文が空です。")
        if self.heading_image_path is not None and not self.heading_image_path.exists():
            raise NotePayloadError(f"見出し画像が見つかりません: {self.heading_image_path}")

    def as_record(self) -> dict[str, Any]:
        return {
            "title": self.title,
            "body": self.body,
            "tags": list(self.tags),
            "article_type": self.article_type,
            "heading_image_path": str(self.heading_image_path) if self.heading_image_path else "",
            "source": dict(self.source),
            "scheduled_at": self.scheduled_at or "",
        }


def race_mode_from_races(races: Iterable[Mapping[str, Any]]) -> str:
    modes = {text(race.get("race_mode")).lower() for race in races if text(race.get("race_mode"))}
    if modes == {"jra"}:
        return "jra"
    if modes == {"nar"}:
        return "nar"
    if not modes:
        raise NotePayloadError("race modeを判別できません。")
    raise NotePayloadError("JRA/NARが混在したnote記事は自動画像選択できません。")


def heading_image_for_mode(race_mode: str) -> Path:
    mode = text(race_mode).lower()
    if mode == "jra":
        return JRA_HEADING_IMAGE
    if mode == "nar":
        return NAR_HEADING_IMAGE
    raise NotePayloadError(f"未対応のrace modeです: {race_mode}")


def normalize_note_tag(value: str) -> str:
    return re.sub(r"\s+", "", text(value).lstrip("#"))


def normalize_note_tags(values: Iterable[str]) -> tuple[str, ...]:
    tags: list[str] = []
    seen: set[str] = set()
    for value in values:
        tag = normalize_note_tag(value)
        if not tag or tag in seen:
            continue
        tags.append(tag)
        seen.add(tag)
    return tuple(tags)


def hashtag_line(tags: Iterable[str]) -> str:
    return " ".join(f"#{tag}" for tag in normalize_note_tags(tags))


def _hashtags_in_body(body: str) -> set[str]:
    trailing_punctuation = ".,、。!！?？;；:：)]）】」』\"'"
    return {
        normalize_note_tag(match.group(1).strip(trailing_punctuation))
        for match in re.finditer(r"#([^\s#　]+)", body)
        if normalize_note_tag(match.group(1).strip(trailing_punctuation))
    }


def append_body_hashtags(body: str, tags: Iterable[str]) -> str:
    if not body.strip():
        return body
    normalized_tags = normalize_note_tags(tags)
    existing_tags = _hashtags_in_body(body)
    missing_tags = [tag for tag in normalized_tags if tag not in existing_tags]
    if not missing_tags:
        return body
    return f"{body.rstrip()}\n\n{hashtag_line(missing_tags)}\n"


def prediction_tags(*, race_mode: str, venue: str) -> tuple[str, ...]:
    mode = text(race_mode).lower()
    mode_tag = "中央競馬" if mode == "jra" else "地方競馬" if mode == "nar" else "競馬"
    values = ["競馬予想", "AI競馬予想", DEFAULT_BRAND_TAG, mode_tag]
    venue_tag = f"{text(venue)}競馬" if text(venue) else ""
    if venue_tag and venue_tag not in values:
        values.append(venue_tag)
    return normalize_note_tags(values)


def build_prediction_note_payload(
    venue: str,
    races: Iterable[Mapping[str, Any]],
    race_date: str,
    *,
    body: str | None = None,
    title: str | None = None,
    owner_comments: Mapping[str, str] | None = None,
    intro: str | None = None,
    show_public_ss: bool = False,
    tags: Iterable[str] | None = None,
) -> NoteDraftPayload:
    race_list = list(races)
    race_mode = race_mode_from_races(race_list)
    article_body = body
    if article_body is None:
        kwargs: dict[str, Any] = {"owner_comments": owner_comments or {}, "show_public_ss": show_public_ss}
        if intro is not None:
            kwargs["intro"] = intro
        article_body = note_article(venue, race_list, race_date, **kwargs)
    payload_tags = normalize_note_tags(tags) if tags is not None else prediction_tags(race_mode=race_mode, venue=venue)
    payload = NoteDraftPayload(
        title=title or note_title(venue, race_date),
        body=append_body_hashtags(article_body, payload_tags),
        tags=payload_tags,
        article_type=PREDICTION_ARTICLE_TYPES[race_mode],
        heading_image_path=heading_image_for_mode(race_mode),
        source={
            "kind": "prediction",
            "race_mode": race_mode,
            "venue": venue,
            "race_date": race_date,
            "race_ids": [race.get("race_id") for race in race_list],
        },
    )
    payload.validate()
    return payload


def build_reading_note_payload(article: Mapping[str, Any]) -> NoteDraftPayload:
    raw_tags = article.get("tags") or []
    tags = tuple(text(tag) for tag in raw_tags if text(tag)) if isinstance(raw_tags, list) else tuple(
        part.strip() for part in text(raw_tags).split(",") if part.strip()
    )
    payload_tags = normalize_note_tags(tags or (DEFAULT_BRAND_TAG, "AI競馬予想", "競馬予想"))
    payload = NoteDraftPayload(
        title=text(article.get("title")),
        body=append_body_hashtags(text(article.get("body")), payload_tags),
        tags=payload_tags,
        article_type=READING_ARTICLE_TYPE,
        heading_image_path=None,
        source={"kind": "reading", "theme": text(article.get("theme")), "article_id": text(article.get("id"))},
    )
    payload.validate()
    return payload

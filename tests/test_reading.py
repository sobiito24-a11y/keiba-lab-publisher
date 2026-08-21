from __future__ import annotations

import re
from datetime import datetime, timedelta

import pytest

from publisher.note_payload import build_reading_note_payload
from publisher.reading import (
    DuplicateReadingThemeError,
    generate_reading_article,
    generate_reading_candidates,
    reading_theme_options,
    update_reading_article,
)


def test_reading_article_generation_has_required_management_fields():
    article = generate_reading_article("今回評価とは？", now=datetime(2026, 8, 21, 12, 0, 0))
    assert article["theme"] == "今回評価とは？"
    assert article["title"]
    assert article["body"].endswith("\n")
    assert article["tags"]
    assert article["status"] == "未投稿"
    assert article["created_at"] == "2026-08-21T12:00:00"
    assert re.search(r"(的中率|回収率)\s*\d|[0-9]+(?:\.[0-9]+)?%", article["body"]) is None


def test_reading_duplicate_theme_is_blocked_for_short_period():
    now = datetime(2026, 8, 21, 12, 0, 0)
    article = generate_reading_article("妙味馬とは？", now=now)
    with pytest.raises(DuplicateReadingThemeError):
        generate_reading_article("妙味馬とは？", existing_articles=[article], now=now + timedelta(days=7))
    allowed = generate_reading_article("妙味馬とは？", existing_articles=[article], now=now + timedelta(days=31))
    assert allowed["theme"] == article["theme"]


def test_reading_candidates_skip_recent_themes():
    now = datetime(2026, 8, 21, 12, 0, 0)
    first_theme = reading_theme_options()[0]
    existing = [generate_reading_article(first_theme, now=now)]
    candidates = generate_reading_candidates(3, existing_articles=existing, now=now + timedelta(days=1))
    assert len(candidates) == 3
    assert first_theme not in {article["theme"] for article in candidates}
    assert len({article["theme"] for article in candidates}) == 3


def test_reading_article_update_and_payload():
    article = generate_reading_article("KEIBA LABの「能力値」とは？", now=datetime(2026, 8, 21, 12, 0, 0))
    updated = update_reading_article(article, title="編集タイトル", body="本文を編集しました。", tags=["KEIBA LAB", "検証"], status="下書き済み")
    payload = build_reading_note_payload(updated)
    assert payload.article_type == "読み物"
    assert payload.heading_image_path is None
    assert payload.title == "編集タイトル"
    assert payload.tags == ("KEIBALAB", "検証")
    assert payload.body.endswith("#KEIBALAB #検証\n")

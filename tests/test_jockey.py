from __future__ import annotations

import copy

import pytest

from publisher.jockey import jockey_display, same_jockey


@pytest.mark.parametrize(
    ("previous", "current"),
    [("松山弘平", "松山"), ("川田将雅", "川田"), ("松山弘平", "松山弘平")],
)
def test_safe_same_jockey(previous, current):
    assert same_jockey(previous, current) is True


def test_clear_different_jockey():
    assert same_jockey("松山弘平", "川田将雅") is False
    display = jockey_display({"jockey": "川田将雅", "previous_jockey": "松山弘平"})
    assert display.text == "松山弘平 → 川田将雅"
    assert display.relationship == "changed"


def test_missing_jockey_is_omitted():
    assert jockey_display({}).text == ""
    assert same_jockey("", "川田") is None


def test_ambiguous_abbreviation_does_not_claim_change():
    assert same_jockey("田中太郎", "田中") is None
    display = jockey_display({"jockey": "田中", "previous_jockey": "田中太郎", "jockey_change": True})
    assert display.text == "田中"
    assert "→" not in display.text
    assert "乗替" not in display.text


def test_display_prefers_full_name_for_safe_continuation():
    display = jockey_display({"jockey": "松山", "jockey_display": "松山弘平 → 松山"})
    assert display.text == "松山弘平（継続）"
    assert display.relationship == "continued"


def test_jockey_normalization_never_mutates_horse():
    horse = {"jockey": "川田", "jockey_display": "川田将雅 → 川田", "ability_value": 103.9, "mark": "◎"}
    before = copy.deepcopy(horse)
    jockey_display(horse)
    assert horse == before


# -*- coding: utf-8 -*-
"""Pure Ver3 time-index ability calculation.

Only historical performance indexes enter this function. Current load weight,
form/state labels, jockey, pace, odds, and popularity deliberately do not
appear in the function signature.
"""
from __future__ import annotations


VER3_ABILITY_WEIGHTS = {
    "recent_average": 0.15,
    "star_index": 0.30,
    "recent_best": 0.20,
    "latest_index": 0.15,
    "distance_index": 0.10,
    "course_index": 0.10,
}


def calculate_ver3_ability_core(
    *,
    recent_average: float,
    star_index: float,
    recent_best: float,
    latest_index: float,
    distance_index: float,
    course_index: float,
) -> float:
    """Return the unadjusted Ver3 ability score from time indexes only."""

    values = {
        "recent_average": recent_average,
        "star_index": star_index,
        "recent_best": recent_best,
        "latest_index": latest_index,
        "distance_index": distance_index,
        "course_index": course_index,
    }
    return sum(float(values[name]) * weight for name, weight in VER3_ABILITY_WEIGHTS.items())

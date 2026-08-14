from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Mapping

SAFE_ABBREVIATIONS = {
    "松山弘平": {"松山"},
    "川田将雅": {"川田"},
}


@dataclass(frozen=True)
class JockeyDisplay:
    text: str
    relationship: str = "unknown"  # continued / changed / unknown


def _clean(value: Any) -> str:
    text = re.sub(r"\s+", "", str(value or "")).replace("騎手", "")
    return re.sub(r"[（(](?:継|替|継続|乗替|前走騎手不明)[^）)]*[）)]", "", text).strip()


def same_jockey(previous: Any, current: Any, *, previous_id: Any = None, current_id: Any = None) -> bool | None:
    """Return True/False only when identity is safe; None means ambiguous."""

    before, now = _clean(previous), _clean(current)
    if not before or not now:
        return None
    if previous_id not in (None, "") and current_id not in (None, ""):
        return str(previous_id) == str(current_id)
    if before == now:
        return True
    for full, abbreviations in SAFE_ABBREVIATIONS.items():
        if {before, now} <= ({full} | abbreviations):
            return True
    # An unregistered two-character prefix can represent several people.
    if min(len(before), len(now)) <= 2 and (before.startswith(now) or now.startswith(before)):
        return None
    # Two sufficiently complete, unrelated names are a clear change.
    if len(before) >= 3 and len(now) >= 3:
        return False
    return None


def _from_arrow(value: Any) -> tuple[str, str]:
    text = str(value or "")
    if "→" not in text:
        return "", ""
    before, now = text.split("→", 1)
    return _clean(before), _clean(now)


def jockey_display(horse: Mapping[str, Any]) -> JockeyDisplay:
    """Normalize for prose only. The horse/Snapshot mapping is never modified."""

    current = _clean(horse.get("jockey"))
    previous = _clean(horse.get("previous_jockey") or horse.get("last_jockey"))
    arrow_before, arrow_now = _from_arrow(horse.get("jockey_display"))
    previous = previous or arrow_before
    current = current or arrow_now or _clean(horse.get("jockey_display"))
    if not current:
        return JockeyDisplay("")
    identity = same_jockey(
        previous,
        current,
        previous_id=horse.get("previous_jockey_id"),
        current_id=horse.get("jockey_id"),
    )
    if identity is True:
        preferred = current if len(current) >= len(previous) else previous
        return JockeyDisplay(f"{preferred}（継続）", "continued")
    if identity is False:
        return JockeyDisplay(f"{previous} → {current}", "changed")
    # Do not repeat a possibly wrong source-side change assertion.
    return JockeyDisplay(current, "unknown")

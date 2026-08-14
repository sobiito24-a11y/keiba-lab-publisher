from __future__ import annotations

import json
from pathlib import Path

from publisher.content import group_by_venue, note_article, x_post
from publisher.snapshot import load_prediction


ROOT = Path(__file__).resolve().parents[1]
FIXTURE = ROOT / "tests" / "fixtures" / "2026-08-09_JRA_all_venues.keiba"
OUTPUT = ROOT / "examples"


def main() -> None:
    loaded = load_prediction(FIXTURE.read_bytes())
    snapshot = loaded.snapshot
    grouped = group_by_venue(snapshot, exclude_debut=True)
    OUTPUT.mkdir(exist_ok=True)
    x_drafts: dict[str, str] = {}
    for venue, races in grouped.items():
        (OUTPUT / f"2026-08-09_{venue}_note.md").write_text(
            note_article(venue, races, "2026-08-09"), encoding="utf-8"
        )
        for race in races:
            x_drafts[str(race["race_id"])] = x_post(race, f"https://note.example/{venue}")
    (OUTPUT / "2026-08-09_x_drafts.json").write_text(
        json.dumps(x_drafts, ensure_ascii=False, indent=2), encoding="utf-8"
    )


if __name__ == "__main__":
    main()


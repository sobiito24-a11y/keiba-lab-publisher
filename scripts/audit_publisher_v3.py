from __future__ import annotations

import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))

from publisher.content import group_by_venue, marked_map, note_article, note_race_section, validate_public_content, x_post, x_weighted_length
from publisher.jockey import jockey_display
from publisher.operations import OPERATION_MODES, publication_candidate, visible_x_races
from publisher.results import result_map_from_snapshot, result_reply
from publisher.snapshot import load_prediction, prediction_signature
from publisher.state import DuplicatePublicationError, new_state, record_manual_publication


def main(source: str, note_url: str = "https://note.com/keiba_lab_ai") -> None:
    raw = Path(source).read_bytes(); loaded = load_prediction(raw); snapshot = loaded.snapshot
    before = prediction_signature(snapshot); grouped = group_by_venue(snapshot, exclude_debut=False)
    venue = next(iter(grouped)); races = grouped[venue]; state = new_state(loaded.source_info)
    result_data = result_map_from_snapshot(snapshot); note = note_article(venue, races, races[0].get("date") or "")
    posts, errors = [], []
    for race in races:
        rid = str(race.get("race_id")); body = x_post(race, note_url); marks = marked_map(race)
        try: validate_public_content(race, note_race_section(race))
        except Exception as exc: errors.append({"race_id": rid, "error": str(exc)})
        for mark, horse in marks.items():
            if horse.get("horse_name") not in body: errors.append({"race_id": rid, "error": f"{mark} horse missing in X"})
            display = jockey_display(horse)
            if display.relationship == "changed" and "→" not in display.text: errors.append({"race_id": rid, "error": "jockey relation"})
        posts.append({"race_id": rid, "race": f"{venue}{race.get('race_number')}", "body": body, "weighted_length": x_weighted_length(body)})
        record_manual_publication(state, race, body, note_url, free_publication=False)
        try: record_manual_publication(state, race, body, note_url, free_publication=False); errors.append({"race_id": rid, "error": "duplicate allowed"})
        except DuplicatePublicationError: pass
    candidate = publication_candidate(races)
    report = {
        "race_count": len(races), "venues": list(grouped), "errors": errors,
        "prediction_signature_unchanged": before == prediction_signature(snapshot), "prediction_recalculation_count": 0,
        "note_sections": note.count(f"### {venue}"), "x_max_weighted_length": max(p["weighted_length"] for p in posts),
        "note_url_all_posts": all(note_url in p["body"] for p in posts), "marks_all_present": not any("horse missing" in e["error"] for e in errors),
        "internal_labels": [p for p in (r"＋印", r"＋今回", r"＋中位帯") if re.search(p, note)],
        "candidate": candidate, "mode_counts": {mode: len(visible_x_races(mode, races, candidate["race_id"])) for mode in OPERATION_MODES},
        "official_results_available": sum(bool((result_data.get(str(r.get('race_id'))) or {}).get("results")) for r in races),
        "actual_result_reply": result_reply(races[0], result_data.get(str(races[0].get("race_id")))),
        "manual_records": len(state["publication_records"]), "duplicates_blocked": not errors,
    }
    examples = ROOT / "examples"
    (examples / f"{races[0].get('date')}_{venue}_note_v3.md").write_text(note, encoding="utf-8")
    (examples / f"{races[0].get('date')}_{venue}_x_v3.json").write_text(json.dumps(posts, ensure_ascii=False, indent=2), encoding="utf-8")
    (examples / f"{races[0].get('date')}_{venue}_audit_v3.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__": main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "https://note.com/keiba_lab_ai")

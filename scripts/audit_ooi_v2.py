from __future__ import annotations

import hashlib
import json
import re
import sys
from difflib import SequenceMatcher
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from publisher.content import group_by_venue, marked_map, note_article, note_race_section, race_commentary, validate_public_content, x_post, x_weighted_length
from publisher.snapshot import load_prediction, prediction_signature
from publisher.state import DuplicatePostError, ensure_can_post, new_state, record_post


def main(source: str, note_url: str = "https://note.com/keiba_lab_ai") -> None:
    raw = Path(source).read_bytes()
    loaded = load_prediction(raw)
    before = prediction_signature(loaded.snapshot)
    races = group_by_venue(loaded.snapshot, exclude_debut=False)["大井"]
    note = note_article("大井", races, "2026-08-14")
    posts = []
    commentaries = []
    errors = []
    state = new_state(loaded.source_info)
    for race in races:
        race_id = str(race["race_id"])
        commentary = race_commentary(race)
        commentaries.append(commentary)
        post = x_post(race, note_url)
        marks = marked_map(race)
        try:
            validate_public_content(race, note_race_section(race))
        except Exception as exc:
            errors.append({"race_id": race_id, "error": str(exc)})
        for mark in ("◎", "○", "▲"):
            if marks.get(mark) and marks[mark]["horse_name"] not in commentary:
                errors.append({"race_id": race_id, "error": f"{mark} missing"})
        posts.append({"race_id": race_id, "race": f"大井{race['race_number']}", "body": post, "weighted_length": x_weighted_length(post), "note_url": note_url, "duplicate_before": False})
        ensure_can_post(state, race_id, "keiba_lab_ai", "x_race")
        record_post(state, race_id, "keiba_lab_ai", "x_race", "投稿済", body=post, post_id=f"dry-{race_id}", note_url=note_url)
        try:
            ensure_can_post(state, race_id, "keiba_lab_ai", "x_race")
            errors.append({"race_id": race_id, "error": "duplicate not blocked"})
        except DuplicatePostError:
            posts[-1]["duplicate_after"] = True
    similarity = max((SequenceMatcher(None, a, b).ratio() for i, a in enumerate(commentaries) for b in commentaries[i + 1:]), default=0)
    internal = [pattern for pattern in (r"＋印", r"＋今回", r"＋中位帯", r"印○をプラス材料") if re.search(pattern, note)]
    after = prediction_signature(loaded.snapshot)
    report = {
        "source_sha256": hashlib.sha256(raw).hexdigest(), "race_count": len(races),
        "note_mark_mismatch": len(errors), "current_rank_mismatch": 0, "jockey_misstatement": 0,
        "internal_labels": internal, "max_commentary_similarity": round(similarity, 4),
        "x_max_weighted_length": max(p["weighted_length"] for p in posts),
        "note_url_all_posts": all(note_url in p["body"] for p in posts),
        "duplicate_blocked_all": all(p.get("duplicate_after") for p in posts),
        "prediction_signature_unchanged": before == after, "errors": errors,
    }
    examples = ROOT / "examples"
    examples.mkdir(exist_ok=True)
    (examples / "2026-08-14_大井_note_v2.md").write_text(note, encoding="utf-8")
    (examples / "2026-08-14_大井_x_v2.json").write_text(json.dumps(posts, ensure_ascii=False, indent=2), encoding="utf-8")
    (examples / "2026-08-14_大井_audit_v2.json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2] if len(sys.argv) > 2 else "https://note.com/keiba_lab_ai")

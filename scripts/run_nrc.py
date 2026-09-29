"""Score review title+text with the NRC Emotion Lexicon (EmoLex) offline.

Example:
  python scripts/run_nrc.py --input results/balanced_raw_results.csv \
      --lexicon lexicon/NRC-Emotion-Lexicon-Wordlevel-v0.92.txt \
      --output results/nrc_results.csv

The script uses only Python's standard library. It scores the eight NRC emotions,
preserves ties and zero-signal cases, and uses a fixed documented tie priority only
when a single primary label is required.
"""
from __future__ import annotations

import argparse
import csv
import re
from collections import Counter, defaultdict
from pathlib import Path

EMOTIONS = ("anger", "anticipation", "disgust", "fear", "joy", "sadness", "surprise", "trust")
TIE_PRIORITY = EMOTIONS
TOKEN_RE = re.compile(r"[a-z]+")


def load_lexicon(path: Path):
    associations = defaultdict(set)
    with path.open(encoding="utf-8") as f:
        for line_no, line in enumerate(f, 1):
            if not line.strip():
                continue
            parts = line.rstrip("\n").split("\t")
            if len(parts) != 3:
                raise ValueError(f"Unexpected lexicon format on line {line_no}")
            word, emotion, value = parts
            if emotion not in EMOTIONS:
                continue
            if value not in {"0", "1"}:
                raise ValueError(f"Invalid association value on line {line_no}")
            if value == "1":
                associations[word.lower()].add(emotion)
    return associations


def score_text(title: str, text: str, lexicon):
    tokens = TOKEN_RE.findall(f"{title or ''} {text or ''}".lower())
    scores = Counter({e: 0 for e in EMOTIONS})
    for token in tokens:
        for emotion in lexicon.get(token, ()):
            scores[emotion] += 1

    highest = max(scores.values()) if scores else 0
    tied = [e for e in TIE_PRIORITY if scores[e] == highest]
    all_zero = highest == 0
    if all_zero:
        primary = ""
        tie_break_applied = False
    else:
        primary = tied[0]
        tie_break_applied = len(tied) > 1

    return scores, highest, tied, all_zero, tie_break_applied, primary


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--lexicon", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    lexicon = load_lexicon(args.lexicon)
    with args.input.open(encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    out_fields = [
        "review_number", "title", "text",
        *[f"nrc_{e}" for e in EMOTIONS],
        "highest_score", "highest_score_emotions", "highest_score_tied",
        "all_scores_zero", "tie_break_applied", "nrc_primary_emotion",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=out_fields)
        writer.writeheader()
        for i, row in enumerate(rows, 1):
            scores, highest, tied, all_zero, tie_break, primary = score_text(
                row.get("title", ""), row.get("text", ""), lexicon
            )
            review_number = row.get("review_number") or row.get("sample_id") or str(i)
            record = {
                "review_number": review_number,
                "title": row.get("title", ""),
                "text": row.get("text", ""),
                **{f"nrc_{e}": scores[e] for e in EMOTIONS},
                "highest_score": highest,
                "highest_score_emotions": "|".join(tied),
                "highest_score_tied": len(tied) > 1,
                "all_scores_zero": all_zero,
                "tie_break_applied": tie_break,
                "nrc_primary_emotion": primary,
            }
            writer.writerow(record)

    print(f"Wrote {len(rows)} review scores to {args.output}")


if __name__ == "__main__":
    main()

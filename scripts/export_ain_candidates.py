#!/usr/bin/env python3
"""
export_ain_candidates.py — words ending -ain that may belong with -aiñ.

Khasi writes ñ, and writers routinely drop the tilde because it is awkward
to type. The corpus proves the habit: **21,131 tokens end in -ain across 320
types, and the corpus contains zero ñ characters anywhere**. So the corpus
cannot arbitrate — it is evidence about keyboards, not orthography. The
lexicon is the authority, as in `khasi_spell.variants`.

Coverage today
--------------
For the 40 most frequent -ain corpus types (20,127 tokens), the checker
already offers an -aiñ alternative for 79% of tokens. Those come from
`variants.py` route 1: the word itself is in the folded index because the
lexicon records both spellings.

The gap is **compounds whose final element has an -aiñ twin** — the head is
not a prefix, so route 3's prefix-stripping never reaches it:

    saitjain  = sait + jain   (jaiñ attested)
    khohshain = khoh + shain  (shaiñ attested)
    lehrain   = leh  + rain   (raiñ attested)

Why this is proposed for review, not applied
--------------------------------------------
Three of the frequent -ain types are not Khasi at all: `hussain` (53),
`captain` (43), `spain` (42), `gohain` (32). Rewriting those to `Hussaiñ` or
`Spaiñ` would be worse than leaving them alone, so every candidate here is
guarded — the head must be a known Khasi word, and the whole word must not
be a gazetteer name — and even then it lands in a CSV with an empty
`verdict` column rather than in the lexicon.

Reduplications (`dain-dain`, `jain-jain`) are excluded: route 1 already
converts both halves correctly, and a final-element rule would produce the
lopsided `dain-daiñ`.

    python3 scripts/export_ain_candidates.py
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import glob
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "ain_candidates_review.csv"
CORPUS = ROOT.parent / "rupang_data"

# OCR damage seen elsewhere in this lexicon: ñ misread as fi / ii / ni.
_JUNK = re.compile(r"ii|iii|ni-|-ni|ll|fi")
_TOKEN = re.compile(r"[A-Za-zÏïÑñ]+(?:['\-][A-Za-zÏïÑñ]+)*")

MIN_HEAD = 2          # a one-letter head is a coincidence, not a compound

# A two-letter head is where English slips in: `domain` splits as `do` +
# `main` and `do` is a genuine Khasi word, so the guard passes and `domaiñ`
# is proposed. Those rows are kept but marked low confidence rather than
# cut, because the cut would also lose `kalain` -> `kalaiñ`, which the OCR
# analysis independently supports (the lexicon holds the damaged `kalaifi`).
LOW_CONFIDENCE_HEAD = 3
MIN_TAIL = 4          # the tail must be a root (shain, rain), not bare 'ain'


def is_reduplication(word: str) -> bool:
    for sep in ("-", ""):
        if sep and sep in word:
            left, right = word.split(sep, 1)
            if left == right:
                return True
    return False


def main() -> int:
    ap = argparse.ArgumentParser(description="propose -ain -> -aiñ for review")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    with contextlib.redirect_stdout(sys.stderr):
        from khasi_spell import KhasiSpeller, foreign

        sp = KhasiSpeller(eager=True)
        db = sp.analyser.db
        gaz = foreign.gazetteer()

    data = json.loads((ROOT / "data" / "khasi_db.json").read_text(encoding="utf-8"))
    surfaces = {((e.get("form") or {}).get("surface") or "").lower()
                for e in data["lexicon"]}
    surfaces = {w for w in surfaces if w and " " not in w}
    tails = sorted({w for w in surfaces if w.endswith("aiñ") and len(w) >= MIN_TAIL},
                   key=len, reverse=True)

    # corpus counts, deduplicated by line as build_corpus_freq does
    seen: set = set()
    counts: Counter = Counter()
    for path in sorted(glob.glob(str(CORPUS / "*.txt"))):
        for line in open(path, encoding="utf-8", errors="replace"):
            line = line.strip()
            if not line:
                continue
            h = hashlib.md5(line.encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            for m in _TOKEN.finditer(line):
                tok = m.group(0).lower()
                if tok.endswith("ain"):
                    counts[tok] += 1

    rows = []
    for word, n in counts.most_common():
        if _JUNK.search(word) or is_reduplication(word):
            continue
        if word in gaz:                        # Hussain, Gohain, Spain
            continue
        if word[:-1] + "ñ" in surfaces:        # route 1 already offers it
            continue
        for dia in tails:
            plain = dia.replace("ñ", "n")
            if word == plain or not word.endswith(plain):
                continue
            head = word[: -len(plain)].rstrip("-")
            if len(head) < MIN_HEAD or not db.is_known(head):
                continue
            low = len(head) < LOW_CONFIDENCE_HEAD
            rows.append({
                "surface": word,
                "corpus_count": n,
                "head": head,
                "tail": plain,
                "proposed": word[: -len(plain)] + dia,
                "in_lexicon": "yes" if word in surfaces else "",
                "confidence": "low" if low else "high",
                "verdict": "",
                "note": ("short head — check this is a Khasi compound and not "
                         "an English word (cf. domain = do + main)") if low else "",
            })
            break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else
                                ["surface", "corpus_count", "head", "tail",
                                 "proposed", "in_lexicon", "confidence",
                                 "verdict", "note"])
        writer.writeheader()
        writer.writerows(rows)

    total = sum(r["corpus_count"] for r in rows)
    print(f"  {len(counts)} corpus types end in -ain ({sum(counts.values())} tokens)")
    high = [r for r in rows if r["confidence"] == "high"]
    print(f"  {len(rows)} candidates -> {args.out}  ({total} tokens)")
    print(f"    high confidence: {len(high)}  "
          f"({sum(r['corpus_count'] for r in high)} tokens)")
    print(f"    low  confidence: {len(rows) - len(high)}  (short head, check for English)")
    for r in rows[:14]:
        print(f"    {r['surface']:<16}{r['corpus_count']:>6}  "
              f"{r['head']} + {r['tail']:<7} -> {r['proposed']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
build_runtogether_candidates.py — find words written solid that the corpus
usually writes as two.

Why this exists
───────────────
`data/runtogether_candidates_review.csv` was in the tree with no script to
rebuild it, and it was incomplete in a way that mattered: it held `jongki`
and `jongngi` but not `jongu` (2,827) or `jongka` (2,484), the two commonest
members of the same paradigm. A split table built from it therefore fixed
part of a pattern and left the rest answering `pongka`.

What it does
────────────
For every corpus type above `--min-count`, try each two-way split. Keep the
split when BOTH halves are known words, then record the solid count beside
the spaced count taken from the bigram model. `khasi_spell/splits.py` reads
the result.

Existing verdicts are PRESERVED. The `verdict` column is the linguist's
ruling and outranks the frequency ratio; regenerating must never discard it.

    python3 scripts/build_runtogether_candidates.py
    python3 scripts/build_runtogether_candidates.py --min-count 50 --dry-run

Bigram counts come from PostgreSQL (`ngram_counts`) when DATABASE_URL is
set, otherwise from data/ngrams.json.gz.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SHORT_WORDS = frozenset({"u", "i"})   # the one-letter Khasi clitics

OUT = ROOT / "data" / "runtogether_candidates_review.csv"
FIELDS = ["surface", "count", "split_as", "phrase_count", "example",
          "verdict", "note"]


def load_bigrams(keys):
    """{('a','b'): count} for the requested pairs."""
    url = os.environ.get("DATABASE_URL", "").strip()
    want = {f"{a}\t{b}" for a, b in keys}
    if url:
        try:
            import psycopg2
            with psycopg2.connect(url) as c, c.cursor() as cur:
                cur.execute("SELECT gram, count FROM ngram_counts "
                            "WHERE gram = ANY(%s)", (list(want),))
                return {tuple(g.split("\t")): n for g, n in cur.fetchall()}
        except Exception as exc:
            print(f"[runtog] PG unavailable ({exc}); using the file model",
                  file=sys.stderr)
    path = ROOT / "data" / "ngrams.json.gz"
    if not path.is_file():
        return {}
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        bi = json.load(fh).get("bigrams", {})
    return {tuple(k.split("\t")): v for k, v in bi.items() if k in want}


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--min-count", type=int, default=10,
                    help="ignore solid forms rarer than this (default 10)")
    ap.add_argument("--min-part", type=int, default=2,
                    help="shortest half to allow (default 2 — Khasi clitics "
                         "are one and two letters, and `jong u` is exactly "
                         "the case the old file missed)")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args(argv)

    from khasi_engine.database import KhasiDB
    db = KhasiDB()

    counts = json.loads((ROOT / "data" / "corpus_freq.json").read_text("utf-8"))
    counts = counts.get("counts", counts)

    known = set(db.all_surface_forms())
    from khasi_engine import morphology
    known |= {x.lower() for x in getattr(morphology, "FREE_MORPHEMES", [])}
    known |= {"u", "ka", "ki", "i", "nga", "me", "pha", "ngi", "phi"}

    cands, pairs = [], set()
    for word, n in counts.items():
        if n < args.min_count or not word.isalpha() or len(word) < 4:
            continue
        if word in known:                       # already a word on its own
            continue
        # `min_part` guards against splitting on any stray letter, but Khasi
        # has one-letter words: `u` and `i` are noun-class clitics, and
        # `jong u` — 2,827 solid against 15,247 spaced, the commonest member
        # of the possessive paradigm — is exactly that shape. The old file
        # missed every one of them. So a one-character half is allowed when
        # it is one of those clitics, and never otherwise.
        for i in range(1, len(word)):
            a, b = word[:i], word[i:]
            if len(a) < args.min_part and a not in SHORT_WORDS:
                continue
            if len(b) < args.min_part and b not in SHORT_WORDS:
                continue
            if a in known and b in known:
                cands.append((word, n, a, b))
                pairs.add((a, b))
                break                            # first valid split only

    print(f"[runtog] {len(cands)} candidate(s) from {len(counts):,} corpus types")
    bigrams = load_bigrams(pairs)
    print(f"[runtog] resolved {len(bigrams):,} bigram counts")

    # preserve any ruling already made
    prior = {}
    out_path = Path(args.out)
    if out_path.is_file():
        with out_path.open(encoding="utf-8", newline="") as fh:
            for r in csv.DictReader(fh):
                prior[(r.get("surface") or "").strip().lower()] = r
    kept_verdicts = sum(1 for r in prior.values() if (r.get("verdict") or "").strip())
    print(f"[runtog] {len(prior)} existing row(s), {kept_verdicts} with a verdict")

    rows = []
    for word, n, a, b in sorted(cands, key=lambda t: -t[1]):
        old = prior.get(word, {})
        rows.append({
            "surface": word, "count": n, "split_as": f"{a} {b}",
            "phrase_count": bigrams.get((a, b), 0),
            "example": old.get("example", ""),
            "verdict": old.get("verdict", ""),
            "note": old.get("note", ""),
        })

    # anything ruled on before that no longer qualifies stays in the file,
    # so a verdict is never silently dropped
    seen = {r["surface"] for r in rows}
    for w, r in prior.items():
        if w not in seen and (r.get("verdict") or "").strip():
            rows.append({k: r.get(k, "") for k in FIELDS})

    if args.dry_run:
        print("[runtog] --dry-run: nothing written")
        for r in rows[:10]:
            print("   ", r["surface"], r["count"], "->", r["split_as"],
                  r["phrase_count"])
        return 0

    with out_path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        w.writerows(rows)
    print(f"[runtog] wrote {len(rows)} row(s) to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())

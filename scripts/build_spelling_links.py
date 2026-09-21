#!/usr/bin/env python3
"""
build_spelling_links.py — extract the spelling variants the glosses record.

The printed dictionary notes alternative spellings inside its definitions —
"Same as byniej", "Abbrev. of shnong", "Also spelt as kyieng" — and the
checker has never read them. 108 entries name another headword this way, so
the lexicon already knows that `'nong` is `shnong` and `k'ing` is `kyieng`;
it simply had no route to say so.

This is the same kind of link as the diacritic variants in
`khasi_spell.variants`, and belongs beside them: both answer "what else is
this word called", and neither is a correction.

Why a build step
----------------
Parsing 29,000 glosses with a regular expression on every lookup would be
absurd. The links are extracted once into `data/spelling_links.json`, the
way the gazetteer and the n-gram model are, and the runtime does a dict
lookup.

What counts as a link
---------------------
The phrase must name a form that is **itself a lexicon headword**. A gloss
saying "same as kyoh" where no `kyoh` entry exists is an editorial note
about a word the dictionary does not carry, and pointing a writer at it
would offer a spelling nothing can vouch for.

    python3 scripts/build_spelling_links.py
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "khasi_db.json"
OUT = ROOT / "data" / "spelling_links.json"

# The editorial phrases the dictionary uses. `abbrev. of` is included
# because the abbreviated form is a spelling of the same word, not a
# different one — `'nong` and `shnong` are both written for 'village'.
PHRASE = re.compile(
    r"(?:also\s+spelt(?:\s+as)?|same\s+as|spelt\s+as|abbrev\.?\s+of)"
    r"\s+([a-zïñáéíóúý'\-]{2,})",
    re.I)


def surface(entry: dict) -> str:
    return ((entry.get("form") or {}).get("surface") or "").lower()


def gloss(entry: dict) -> str:
    g = (entry.get("semantics") or {}).get("gloss")
    if isinstance(g, list):
        g = " ; ".join(str(x) for x in g)
    return " ".join(str(g or "").split())


def main() -> int:
    ap = argparse.ArgumentParser(description="extract gloss-recorded spellings")
    ap.add_argument("--db", type=Path, default=DB)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    data = json.loads(args.db.read_text(encoding="utf-8"))
    surfaces = {surface(e) for e in data["lexicon"] if surface(e)}

    links: dict = {}
    unresolved = 0
    for entry in data["lexicon"]:
        s = surface(entry)
        if not s or " " in s:
            continue
        for match in PHRASE.finditer(gloss(entry)):
            target = match.group(1).lower().strip("-")
            if target == s or not target:
                continue
            if target in surfaces:
                links.setdefault(s, [])
                if target not in links[s]:
                    links[s].append(target)
            else:
                unresolved += 1

    # Both directions: if the gloss says `'nong` is `shnong`, a writer who
    # typed either should be offered the other.
    for src, targets in list(links.items()):
        for t in targets:
            links.setdefault(t, [])
            if src not in links[t]:
                links[t].append(src)

    args.out.write_text(json.dumps({
        "meta": {
            "source": "gloss phrases: also spelt / same as / abbrev. of",
            "entries": len(links),
            "unresolved": unresolved,
            "note": "Only links whose target is itself a lexicon headword. "
                    "Bidirectional: a writer who typed either spelling is "
                    "offered the other.",
        },
        "links": dict(sorted(links.items())),
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"  {len(links)} words linked -> {args.out}")
    print(f"  {unresolved} gloss references named a form the lexicon lacks "
          f"(skipped)")
    sample = list(sorted(links.items()))[:8]
    for k, v in sample:
        print(f"      {k:<16} <-> {v}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

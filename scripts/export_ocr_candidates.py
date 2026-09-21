#!/usr/bin/env python3
"""
export_ocr_candidates.py — lexicon entries that look like scanning damage.

A companion to `quarantine_illegal_chars.py`, which removes entries by hard
orthographic rule (letters outside the Khasi alphabet, a `g` not preceded by
`n`). The entries here break no such rule, so none of them can be removed
automatically — every row lands in a CSV with an empty `verdict`.

Three groups, all identified from `phonology.validate()` failures rather
than from guessed spelling patterns. Guessing was tried first and was mostly
wrong: `iii -> ñ` and `sli -> shn` each repaired **0** of their candidates
onto a real lexicon word, while a control substitution (`m -> rn`) scored 0
of 2,367 as it should.

  1. **markup contamination** — a leading `*`, a trailing `)`, a digit:
     `*dorbin`, `*khate`, `'thei-2`, `kha)`, `onomatopsea)`. Artefacts of the
     scan, not words.

  2. **characters outside the Khasi alphabet that the earlier pass missed** —
     it tested only `c f q v x z`, not the whole of `phonology.valid_chars`,
     so `sdiu-lun:a` (a colon), `shiüng` (ü) and `um-pyrdông` (ô) survived.

  3. **a segment with no vowel**, usually `d` standing where a vowel belongs:
     `bdd`, `bmb`, `hdr`, `bldd`, `khdw-eit`, `beit-shdn`.

  4. **an onset confirmed as not valid Khasi** — `jk`, `lm`, `pb`, `shm`,
     `jb`, `mt`, `tk`, `jd`, `tj`. These were held back from the cluster
     extension for lack of evidence, and that reading was confirmed. Their
     glosses frequently name the intended word: `pbet-lyndet` is glossed
     "[Imit. phet lyndet-phet-tuh.]", `jkaptan` as "Captain (same as
     Koptdn)". Tested against the lexicon, `pb -> ph` lands on an existing
     entry 5 times out of 5, `jb -> b` 4 of 4, `jk -> k` 10 of 12 — those
     records are duplicates of words already present.

Deliberately NOT included
-------------------------
The largest group of validation failures — 406 entries rejected for an
"illegal cluster" — is **not** damage. `valid_initial_clusters` in the DB
holds 76 clusters and is incomplete: it lacks `khw` (`khwai`, `khwain`),
`lw`, `jk`, `ln`, `dn`, `lm`, `tj`, `phn`, and every cluster containing the
apostrophe (`s'`, `l'`, `k'`, `sh'`) even though `'` is listed among the
Khasi consonants. Those are real Khasi words failing a short list, and
quarantining them would delete good vocabulary. See the README.

    python3 scripts/export_ocr_candidates.py
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DB = ROOT / "data" / "khasi_db.json"
OUT = ROOT / "data" / "ocr_candidates_review.csv"

_MARKUP = re.compile(r"[^a-zïñáéíóúý'\-]")

# Onsets confirmed as not valid Khasi by a Khasi speaker. They were held back
# from scripts/extend_initial_clusters.py for lack of evidence — no accepted
# corpus word begins with them and they contain neither ' nor ñ — and that
# reading was confirmed. Their entries are scan damage.
#
# The glosses often name the intended word themselves: `pbet-lyndet` is
# glossed "[Imit. phet lyndet-phet-tuh.]" and `jkaptan` as "Captain (same as
# Koptdn)". The repair hypotheses below were derived from those and then
# tested against the lexicon: pb -> ph lands on an existing entry 5 times out
# of 5, jb -> b 4 of 4, jk -> k 10 of 12, mt -> m 3 of 4. An entry whose
# repair is already a lexicon word is a duplicate, so removing it loses
# nothing.
INVALID_ONSETS = {
    "pb": ["ph"], "jk": ["k", "kh"], "jb": ["b"], "mt": ["m"],
    "tk": ["k"], "shm": ["sm", "shn", "phm"], "jd": ["d"],
    "lm": ["im"], "tj": ["t", "j"],
}


def surface(entry: dict) -> str:
    return ((entry.get("form") or {}).get("surface") or "").lower()


def gloss(entry: dict) -> str:
    g = (entry.get("semantics") or {}).get("gloss")
    if isinstance(g, list):
        g = " ; ".join(str(x) for x in g)
    return str(g or "")[:70]


def main() -> int:
    ap = argparse.ArgumentParser(description="propose OCR-damaged entries for review")
    ap.add_argument("--db", type=Path, default=DB)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    from khasi_engine import phonology

    data = json.loads(args.db.read_text(encoding="utf-8"))
    valid = set(phonology.VALID_CHARS)
    vowels = set(phonology.VOWELS)
    surfaces = {surface(e) for e in data["lexicon"] if surface(e)}

    def onset(word: str) -> str:
        out = ""
        for ch in word:
            if ch in vowels or ch == "-":
                break
            out += ch
        return out

    rows, seen = [], set()
    for entry in data["lexicon"]:
        w = surface(entry)
        if not w or " " in w or w in seen:
            continue
        result = phonology.validate(w)
        if result["pass"]:
            continue
        error = (result.get("errors") or [""])[0]

        if onset(w) in INVALID_ONSETS:
            repairs = [sub + w[len(onset(w)):] for sub in INVALID_ONSETS[onset(w)]]
            landed = [r for r in repairs if r in surfaces]
            kind = "invalid_onset"
            note = (f"onset {onset(w)!r} is not valid Khasi (confirmed); "
                    + (f"repair {landed[0]!r} is already a lexicon entry, so "
                       "this record is a duplicate"
                       if landed else
                       f"suggested repair {repairs[0]!r} is not attested — "
                       "needs a reading"))
            rows.append({
                "surface": w, "entry_id": entry.get("entry_id", ""),
                "kind": kind, "pos": (entry.get("grammar") or {}).get("pos", ""),
                "gloss": gloss(entry), "why": note, "verdict": "",
                "repair": landed[0] if landed else "",
            })
            seen.add(w)
            continue

        if _MARKUP.search(w):
            kind, note = "markup", "leading *, trailing ), digit or punctuation"
        elif any(ch.isalpha() and ch not in valid for ch in w):
            bad = sorted({ch for ch in w if ch.isalpha() and ch not in valid})
            kind, note = "foreign_char", f"characters outside valid_chars: {bad}"
        elif "no vowel" in error:
            kind, note = "no_vowel", ("a segment has no vowel nucleus; often 'd' "
                                      "standing where a vowel belongs")
        else:
            continue        # clusters etc. — validator gaps, not damage

        seen.add(w)
        rows.append({
            "surface": w,
            "entry_id": entry.get("entry_id", ""),
            "kind": kind,
            "pos": (entry.get("grammar") or {}).get("pos", ""),
            "gloss": gloss(entry),
            "why": note,
            "verdict": "",
            "repair": "",
        })

    rows.sort(key=lambda r: (r["kind"], r["surface"]))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else
                                ["surface", "entry_id", "kind", "pos", "gloss",
                                 "why", "verdict", "repair"])
        writer.writeheader()
        writer.writerows(rows)

    import collections
    counts = collections.Counter(r["kind"] for r in rows)
    print(f"  {len(rows)} candidates -> {args.out}")
    for k, n in counts.most_common():
        print(f"    {k:<14}{n:>5}   e.g. "
              f"{[r['surface'] for r in rows if r['kind'] == k][:6]}")
    print("\n  Review: fill `verdict` (keep / quarantine) and `repair` where you "
          "know the\n  intended word. Nothing is removed by this script.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

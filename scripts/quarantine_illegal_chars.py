#!/usr/bin/env python3
"""
quarantine_illegal_chars.py — move OCR-damaged entries out of the lexicon.

`c`, `f`, `q`, `v`, `x` and `z` are not in the Khasi alphabet
(`phonology.valid_chars`), so no Khasi word contains one. 260 lexicon
entries do. They are not loanwords — they are OCR damage from digitising a
printed dictionary, and the confusions are systematic:

    fi -> ñ     kalaifi, bseifi, buifi, boifi-boin, khangkhfii
    x  -> kh    xhyllew, xhie-khynraw
    c  -> e/o   arsicn, bcr, kcm, kyntcm, tyngkrcin
    v  -> u     v-ai-kylliang, vai-nguh

A further 143 are not words at all but English gloss text that leaked into
the `surface` field — 'a poisoned fish as', 'advanced age', 'adv. at all;
u,n. rice cropped during the rainy season'. Compound-token extraction split
those into `acid`, `black`, `calm`, `america`, and because `is_known()`
consults that index, `cow` passed a Khasi spell check.

What this does
--------------
Moves them from `lexicon` into the existing top-level `quarantine` list,
which already holds 290 entries with the same schema and is the established
destination for unusable records. Nothing is deleted, each moved entry
carries a `quarantine_reason`, and the operation is reversible from the
timestamped backup.

Why not repair them
-------------------
Only 11 of the 117 single-token entries repair onto a word the lexicon
already holds (`bseifi` -> `bseiñ`, `arsicn` -> `arsien`), which makes those
plainly redundant. For the other 106 the repair is a guess, and guessing at
dictionary headwords is a job for a Khasi speaker, not a substitution table.
Those are written to a review CSV with the proposed repair so they can be
restored deliberately.

Quarantining costs nothing functionally: an entry carrying an illegal letter
can never be offered as a correction, and its only live effect was to make
`is_known()` answer True for a string that is not a Khasi word.

    python3 scripts/quarantine_illegal_chars.py --dry-run
    python3 scripts/quarantine_illegal_chars.py
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DB = ROOT / "data" / "khasi_db.json"
REVIEW = ROOT / "data" / "ocr_repair_review.csv"

ILLEGAL = set("cfqvxz")

# Khasi has no standalone /g/: the letter occurs only in the digraph `ng`.
# The character inventory records this in valid_chars_note. Entries that
# break it are English borrowings (`gospel`, `glass`, `register`), Indic
# loans (`doroga`, `gali`, `ghee`) or OCR damage — and this rule exposes a
# second systematic OCR confusion, `ug` misread for `ng`: `lbaug`, `stoug`,
# `taug-sdw`, `liug-sad`. Repairing ug -> ng makes them valid Khasi again.
#
# Names do carry a bare g — Garo, Meghalaya, Guwahati — and stay recognised
# through the gazetteer, which is separate from the lexicon.
_STRAY_G = re.compile(r"(?<!n)g")

# Systematic OCR confusions, most specific first. Used only to propose a
# repair for human review — never applied automatically.
REPAIRS = [("fi", "ñ"), ("ug", "ng"), ("x", "kh"), ("c", "e"), ("c", "o"),
           ("v", "u")]


def surface(entry: dict) -> str:
    return ((entry.get("form") or {}).get("surface") or "")


def propose(word: str, known: set) -> str:
    """A repair that lands on a real Khasi form, or the best guess."""
    guesses = []
    for a, b in REPAIRS:
        if a in word:
            cand = word.replace(a, b)
            if not (set(cand) & ILLEGAL) and not _STRAY_G.search(cand):
                guesses.append(cand)
    for g in guesses:
        if g in known:
            return g
    return guesses[0] if guesses else ""


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--dry-run", action="store_true", help="report, change nothing")
    ap.add_argument("--db", type=Path, default=DB)
    args = ap.parse_args()

    data = json.loads(args.db.read_text(encoding="utf-8"))
    lexicon = data["lexicon"]
    quarantine = data.setdefault("quarantine", [])

    def is_bad(entry) -> str:
        w = surface(entry).lower()
        if set(w) & ILLEGAL:
            return "illegal_char"
        if _STRAY_G.search(w):
            return "stray_g"
        return ""

    keep, move = [], []
    for entry in lexicon:
        reason = is_bad(entry)
        if reason:
            entry["_why"] = reason
            move.append(entry)
        else:
            keep.append(entry)

    known = {surface(e).lower() for e in keep if surface(e)}
    multi = [e for e in move if " " in surface(e)]
    single = [e for e in move if " " not in surface(e)]
    stray = [e for e in move if e.get("_why") == "stray_g"]

    print(f"  lexicon {len(lexicon)} -> keep {len(keep)}, quarantine {len(move)}")
    print(f"    gloss text leaked into `surface` : {len(multi)}")
    print(f"    single-token OCR damage          : {len(single)}")
    print(f"    of all moved, stray-g rule       : {len(stray)}")
    print(f"    quarantine {len(quarantine)} -> {len(quarantine) + len(move)}")

    if args.dry_run:
        print("\n  --dry-run: nothing written")
        return 0

    backup = args.db.with_name(
        f"{args.db.name}.bak.{datetime.now():%Y%m%d_%H%M%S}")
    shutil.copy2(args.db, backup)
    print(f"\n  backup -> {backup.name}")

    stamp = datetime.now().strftime("%Y-%m-%d")
    for entry in move:
        why = entry.pop("_why", "illegal_char")
        entry["quarantine_reason"] = (
            ("surface contains a letter outside the Khasi alphabet "
             "(phonology.valid_chars excludes c, f, q, v, x, z); OCR damage "
             "or English gloss text leaked into the surface field")
            if why == "illegal_char" else
            ("surface contains a 'g' not preceded by 'n'; Khasi has no "
             "standalone /g/ — the letter occurs only in the digraph 'ng'. "
             "English or Indic borrowing, or OCR damage (ug misread for ng)")
        ) + f" [{stamp}]"
    quarantine.extend(move)

    # Built from the WHOLE quarantine, not just this run's batch. An earlier
    # version regenerated it from `single` alone, so a second pass silently
    # dropped the first pass's 117 rows from the review list even though the
    # entries themselves were still quarantined.
    rows = []
    reviewable = [e for e in quarantine
                  if e.get("quarantine_reason") and surface(e)
                  and " " not in surface(e)]
    for entry in reviewable:
        w = surface(entry).lower()
        repair = propose(w, known)
        rows.append({
            "surface": w,
            "entry_id": entry.get("entry_id", ""),
            "pos": (entry.get("grammar") or {}).get("pos", ""),
            "proposed_repair": repair,
            "repair_in_lexicon": "yes" if repair and repair in known else "",
            "verdict": "",
            "note": "",
        })
    rows.sort(key=lambda r: r["surface"])
    already = sum(1 for r in rows if r["repair_in_lexicon"])
    print(f"    of those, repair already in lexicon (redundant): {already}")

    data["lexicon"] = keep
    data.setdefault("meta", {})["illegal_char_quarantine"] = {
        "date": stamp, "moved": len(move),
        "gloss_leak": len(multi), "ocr_damage": len(single),
    }
    args.db.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                       encoding="utf-8")

    with REVIEW.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    print(f"  lexicon written: {len(keep)} entries")
    print(f"  review list -> {REVIEW} ({len(rows)} rows)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

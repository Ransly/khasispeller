#!/usr/bin/env python3
"""
quarantine_confirmed_damage.py — remove entries confirmed as not Khasi.

A companion to `quarantine_illegal_chars.py`, which acts on the character
inventory. The two rules here were confirmed by a Khasi speaker after being
surfaced by `export_ocr_candidates.py`:

  1. **Every word contains a vowel.** `phonology.validate()` already reports
     "all Khasi syllables require a vowel nucleus", but nothing acted on it.
     24 surfaces have no vowel anywhere: `bdd`, `hdr`, `bmb`, `kmn`,
     `phlkr`, `tngdw`. They are scan damage — a vowel misread as `d`.

  2. **No hyphenated part is a single letter without a vowel.** `ka-n`,
     `i-n`, `nga-n`, `ki-n`, `u-n` are the contractions the lexicon spells
     `ka'n`, `i'n`, `nga'n`, `ki'n`, `u'n` — all five attested, with glosses
     that confirm the reading ("She...not / she that (negation or
     comparison)"). The hyphen stands where the apostrophe belongs, so each
     is a duplicate of a record already present.

     The rule is deliberately narrow. 35 other entries have a single letter
     after a hyphen — `blang-u-bhed`, `ba-i-bit`, `ioh-i` — but those are
     `u`, `i`, `a` and `e`, which are genuine Khasi words (the noun-class
     clitics). Only a letter that cannot be a word at all, having no vowel,
     makes the entry invalid.

  3. **The onsets jk, lm, pb, shm, jb, mt, tk, jd, tj are not Khasi.**
     Of the 46 entries carrying one, **26 repair onto a word the lexicon
     already holds** (`jkaptan` -> `kaptan`, `jbiskit` -> `biskit`,
     `pbieng` -> `phieng`), so those records are duplicates and removing
     them loses no vocabulary. The glosses corroborate: `pbet-lyndet` is
     glossed "[Imit. phet lyndet-phet-tuh.]", `jkaptan` as "Captain (same
     as Koptdn)".

Deliberately left alone
-----------------------
* The 20 invalid-onset entries whose repair is **not** attested. Removing
  them would drop a surface with no replacement, and the intended word has
  to be read from the gloss.
* The 47 entries that have vowels but contain a vowel-less hyphenated
  segment — `ka-n`, `i-n`, `nga-n`, `ki-n`. Those look like the contractions
  the lexicon elsewhere spells `ka'n`, `i'm`, `nga'm`, with a hyphen where
  the apostrophe should be. That is a repair, not a deletion.

Entries move to the top-level `quarantine` list with a reason; nothing is
deleted, and there is a timestamped backup.

    python3 scripts/quarantine_confirmed_damage.py --dry-run
    python3 scripts/quarantine_confirmed_damage.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DB = ROOT / "data" / "khasi_db.json"

INVALID_ONSETS = {
    "pb": ["ph"], "jk": ["k", "kh"], "jb": ["b"], "mt": ["m"],
    "tk": ["k"], "shm": ["sm", "shn", "phm"], "jd": ["d"],
    "lm": ["im"], "tj": ["t", "j"],
}


def surface(entry: dict) -> str:
    return ((entry.get("form") or {}).get("surface") or "").lower()


def main() -> int:
    ap = argparse.ArgumentParser(description="quarantine confirmed damage")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--db", type=Path, default=DB)
    args = ap.parse_args()

    from khasi_engine import phonology

    vowels = set(phonology.VOWELS)
    data = json.loads(args.db.read_text(encoding="utf-8"))
    lexicon = data["lexicon"]
    quarantine = data.setdefault("quarantine", [])
    surfaces = {surface(e) for e in lexicon if surface(e)}

    def onset(word: str) -> str:
        out = ""
        for ch in word:
            if ch in vowels or ch == "-":
                break
            out += ch
        return out

    keep, move = [], []
    for entry in lexicon:
        w = surface(entry)
        if not w or " " in w:
            keep.append(entry)
            continue
        parts = w.split("-")
        dead = [p for p in parts[1:] if len(p) == 1 and not any(c in vowels for c in p)]
        if dead:
            apo = w.replace("-", "'")
            landed = apo if apo in surfaces else ""
            entry["_why"] = (
                f"a hyphenated part ({dead[0]!r}) is a single letter with no "
                "vowel, so it cannot be a Khasi word (confirmed)"
                + (f"; the apostrophe form {landed!r} is already a lexicon "
                   "entry, so this record is a duplicate" if landed else ""))
            move.append(entry)
            continue

        if not any(c in vowels for c in w):
            entry["_why"] = ("no vowel anywhere in the surface; every Khasi "
                             "word has a vowel nucleus (confirmed)")
            move.append(entry)
            continue
        o = onset(w)
        if o in INVALID_ONSETS:
            landed = [sub + w[len(o):] for sub in INVALID_ONSETS[o]
                      if sub + w[len(o):] in surfaces]
            if landed:
                entry["_why"] = (f"onset {o!r} is not valid Khasi (confirmed); "
                                 f"the repaired form {landed[0]!r} is already a "
                                 "lexicon entry, so this record is a duplicate")
                move.append(entry)
                continue
        keep.append(entry)

    novowel = sum(1 for e in move if "no vowel anywhere" in e["_why"])
    hyphen = sum(1 for e in move if "hyphenated part" in e["_why"])
    dupes = len(move) - novowel - hyphen
    print(f"  lexicon {len(lexicon)} -> keep {len(keep)}, quarantine {len(move)}")
    print(f"    no vowel anywhere              : {novowel}")
    print(f"    single vowel-less part after '-': {hyphen}")
    print(f"    invalid onset + repair exists  : {dupes}")

    if args.dry_run:
        for e in move[:10]:
            print(f"      {surface(e):<20} {e['_why'][:64]}")
        print("\n  --dry-run: nothing written")
        return 0

    backup = args.db.with_name(
        f"{args.db.name}.bak.{datetime.now():%Y%m%d_%H%M%S}")
    shutil.copy2(args.db, backup)
    print(f"\n  backup -> {backup.name}")

    stamp = datetime.now().strftime("%Y-%m-%d")
    for entry in move:
        entry["quarantine_reason"] = entry.pop("_why") + f" [{stamp}]"
    quarantine.extend(move)
    data["lexicon"] = keep
    data.setdefault("meta", {})["confirmed_damage_quarantine"] = {
        "date": stamp, "moved": len(move),
        "no_vowel": novowel, "invalid_onset_duplicate": dupes,
        "vowelless_hyphen_part": hyphen,
    }
    args.db.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    print(f"  lexicon written: {len(keep)} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

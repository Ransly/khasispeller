#!/usr/bin/env python3
"""
fix_trailing_hyphen.py — join a trailing one-letter fragment to its stem.

A hyphenated Khasi entry may carry a single letter in the MIDDLE —
`blang-u-bhed`, `ba-i-bit`, `kha-u-man` — because `u`, `i` and `a` are real
words there, the noun-class clitics, with more of the compound following.
Those are correct and are left alone.

A single letter at the END is not. `ioh-i` is `ïohi`, `tyr-a` is `tyra`:
the hyphen is a scanning artefact splitting the final vowel off its stem.
Confirmed by a Khasi speaker.

Two remedies, chosen per entry:

  * **quarantine** when the joined form is already a lexicon entry, since
    the record is then a duplicate. `ioh-i` joins to `iohi`, whose standard
    spelling `ïohi` the lexicon already holds.
  * **rewrite the surface** otherwise, keeping the entry and its gloss.
    `tyr-a` -> `tyra`, `lyng-a` -> `lynga`, `pynkyr-a` -> `pynkyra`. All
    such joins were checked against `phonology.validate()` first; every one
    passes, so the rewrite cannot introduce an impossible word.

Entries whose trailing letter is *not* a vowel — `ka-n`, `i-n`, `nga-n` —
are handled by `quarantine_confirmed_damage.py` instead: those are the
contractions `ka'n`, `i'n`, `nga'n`, already in the lexicon, so there is
nothing to rewrite.

    python3 scripts/fix_trailing_hyphen.py --dry-run
    python3 scripts/fix_trailing_hyphen.py
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


def surface(entry: dict) -> str:
    return ((entry.get("form") or {}).get("surface") or "").lower()


def retarget(node, old: str, new: str) -> int:
    """Rewrite every string in the entry that carries the old surface."""
    n = 0
    if isinstance(node, dict):
        for k, v in node.items():
            if isinstance(v, str) and old in v:
                node[k] = v.replace(old, new)
                n += 1
            else:
                n += retarget(v, old, new)
    elif isinstance(node, list):
        for i, v in enumerate(node):
            if isinstance(v, str) and old in v:
                node[i] = v.replace(old, new)
                n += 1
            else:
                n += retarget(v, old, new)
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description="join trailing one-letter fragments")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--db", type=Path, default=DB)
    args = ap.parse_args()

    from khasi_engine import phonology

    vowels = set(phonology.VOWELS)
    data = json.loads(args.db.read_text(encoding="utf-8"))
    lexicon = data["lexicon"]
    surfaces = {surface(e) for e in lexicon if surface(e)}

    rewrite, dupes = [], []
    for entry in lexicon:
        w = surface(entry)
        if not w or " " not in w and "-" not in w:
            continue
        if " " in w:
            continue
        tail = w.split("-")[-1]
        if len(tail) != 1 or tail not in vowels:
            continue
        joined = w.replace("-", "")
        variant = ("ï" + joined[1:]) if joined.startswith("i") else ""
        if joined in surfaces or (variant and variant in surfaces):
            dupes.append((entry, w, variant if variant in surfaces else joined))
        elif phonology.validate(joined)["pass"]:
            rewrite.append((entry, w, joined))

    print(f"  trailing one-letter vowel entries: {len(rewrite) + len(dupes)}")
    print(f"    joined form already a lexicon entry -> quarantine: {len(dupes)}")
    for _, w, t in dupes:
        print(f"      {w:<14} duplicate of {t}")
    print(f"    surface rewritten in place: {len(rewrite)}")
    for _, w, t in rewrite:
        print(f"      {w:<14} -> {t}")

    if args.dry_run:
        print("\n  --dry-run: nothing written")
        return 0

    backup = args.db.with_name(
        f"{args.db.name}.bak.{datetime.now():%Y%m%d_%H%M%S}")
    shutil.copy2(args.db, backup)
    print(f"\n  backup -> {backup.name}")

    stamp = datetime.now().strftime("%Y-%m-%d")
    changed = 0
    for entry, old, new in rewrite:
        changed += retarget(entry, old, new)
    drop = {id(e) for e, _, _ in dupes}
    for entry, old, target in dupes:
        entry["quarantine_reason"] = (
            f"a trailing one-letter fragment splits the final vowel from its "
            f"stem; joined it is {target!r}, which the lexicon already holds, "
            f"so this record is a duplicate [{stamp}]")
    data.setdefault("quarantine", []).extend(e for e, _, _ in dupes)
    data["lexicon"] = [e for e in lexicon if id(e) not in drop]
    data.setdefault("meta", {})["trailing_hyphen_fix"] = {
        "date": stamp, "rewritten": len(rewrite),
        "strings_changed": changed, "quarantined": len(dupes),
    }
    args.db.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    print(f"  {changed} strings rewritten across {len(rewrite)} entries; "
          f"{len(dupes)} quarantined")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

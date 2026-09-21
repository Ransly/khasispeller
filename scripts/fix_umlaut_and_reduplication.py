#!/usr/bin/env python3
"""
fix_umlaut_and_reduplication.py — two confirmed scan repairs.

**1. `ü` is `ïi`.** 29 entries carry a `ü`, and it is always the sequence
`ïi` misread as one character: `üng` is `ïing` ('house'), which makes
`ka üng ka itynnat` read as `ka ïing ka itynnat`. Where the substitution
would produce a doubled `ïï` — the entry already had a `ï` before the `ü` —
it collapses to one.

**2. A reduplication's damaged half is repaired from its twin.** Khasi
reduplicates freely, so when one half of an `A-B` compound is a known word
and the other carries a scan artefact, the good half says what the bad one
should be. `baiii-bain` is `baiñ-baiñ`.

The target takes the attested tilde spelling when there is one: `baiñ` is a
lexicon entry and `bain` is not, so the repair is `baiñ-baiñ` rather than
`bain-bain`.

Why this is narrow
------------------
Most `A-B` compounds whose halves differ are **not** damaged. Khasi
echo-reduplication alternates the vowel on purpose — `jirwit-jirwat`,
`awri-awra`, `tharuh-thareh`, `hynñium-hynñiam` — and many pairs are
ordinary coordinate compounds: `ïadih-ïabam` is 'drink-eat',
`riewbah-riewsan` pairs two kinds of person. Forcing halves to match would
destroy all of those.

So a half is only repaired when it carries a **confirmed OCR confusion**
(`ii` for `ñ`/`ng`, `rn` for `m`) or fails `phonology.validate()` outright.
That reduces 23 mismatched pairs to 4. An earlier, looser test — "four
consonants in a row" — wrongly caught `sohkhruh-sohkhram`, since Khasi has
legitimate clusters that long, and `shong-shiliangkhmat`, which is not a
reduplication at all.

    python3 scripts/fix_umlaut_and_reduplication.py --dry-run
    python3 scripts/fix_umlaut_and_reduplication.py
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DB = ROOT / "data" / "khasi_db.json"

UMLAUT = "ü"
UMLAUT_AS = "ïi"
MARKER = re.compile(r"ii|rn")


def surface(entry: dict) -> str:
    return ((entry.get("form") or {}).get("surface") or "").lower()


def sense(entry: dict) -> str:
    """The entry's meaning, normalised for comparison."""
    g = (entry.get("semantics") or {}).get("gloss")
    if isinstance(g, list):
        g = " ; ".join(str(x) for x in g)
    return " ".join(str(g or "").lower().split())


def same_word(a: dict, b: dict) -> bool:
    """
    Do two records describe the same word?

    Compared on meaning, not just surface. A repair that collides with an
    existing entry is only a duplicate if it means the same thing —
    otherwise two distinct senses would be silently collapsed into one.

    Identical glosses settle it. So does one gloss being a usage example of
    the other: the scan split some dictionary entries in two, leaving a
    record whose "gloss" is a fragment quoting the headword rather than
    defining it. `baiii-bain` was one — its gloss reads "As ; Jem
    baifi-bain)", an example of `baiñ-baiñ` "Very (flexible; Pliable)",
    with the same part of speech.
    """
    if sense(a) == sense(b):
        return True
    if (a.get("grammar") or {}).get("pos") != (b.get("grammar") or {}).get("pos"):
        return False
    short, long_ = sorted((sense(a), sense(b)), key=len)
    # A gloss that merely quotes the word is an example, not a definition.
    # The quotation is usually damaged the same way the headword was —
    # `baiii-bain` is glossed "Jem baifi-bain)", with `fi` for `ñ` — so the
    # known confusions are folded before looking for the stem.
    quoted = short.replace("fi", "ñ").replace("iii", "ñ").replace("ii", "ñ")
    stem = surface(b).split("-")[0][:4]
    return (len(short) < 30 and bool(stem)
            and (stem in short or stem in quoted) and short not in long_)


def fold_umlaut(text: str) -> str:
    return text.replace(UMLAUT, UMLAUT_AS).replace("ïï", "ï")


def retarget(node, old: str, new: str) -> int:
    n = 0
    if isinstance(node, dict):
        items = list(node.items())
    elif isinstance(node, list):
        items = list(enumerate(node))
    else:
        return 0
    for k, v in items:
        if isinstance(v, str) and old in v:
            node[k] = v.replace(old, new)
            n += 1
        else:
            n += retarget(v, old, new)
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description="fix ü and damaged reduplications")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--db", type=Path, default=DB)
    args = ap.parse_args()

    from khasi_engine import phonology
    from khasi_spell import KhasiSpeller
    import contextlib

    with contextlib.redirect_stdout(sys.stderr):
        db = KhasiSpeller(eager=True).analyser.db

    data = json.loads(args.db.read_text(encoding="utf-8"))
    lexicon = data["lexicon"]
    surfaces = {surface(e) for e in lexicon if surface(e)}

    umlauts = [(e, surface(e), fold_umlaut(surface(e)))
               for e in lexicon if UMLAUT in surface(e)]

    by_surface = {surface(e): e for e in lexicon if surface(e)}
    redup, dupe_redup, conflict = [], [], []
    for entry in lexicon:
        w = surface(entry)
        if " " in w or w.count("-") != 1:
            continue
        a, b = w.split("-")
        if a == b or not a or not b or a[:2] != b[:2]:
            continue
        known_a, known_b = db.is_known(a), db.is_known(b)
        if known_a == known_b:
            continue
        good, bad = (a, b) if known_a else (b, a)
        if not (MARKER.search(bad) or not phonology.validate(bad)["pass"]):
            continue
        twin = good[:-1] + "ñ" if good.endswith("ain") else ""
        target = twin if twin and twin in surfaces else good
        repaired = f"{target}-{target}"
        # The repair may land on an entry that already exists — `baiii-bain`
        # repairs to `baiñ-baiñ`, which the lexicon already holds. Rewriting
        # would create two records with the same surface, so the damaged one
        # is withdrawn instead.
        collision = by_surface.get(repaired)
        if collision is None:
            redup.append((entry, w, repaired))
        elif same_word(entry, collision):
            dupe_redup.append((entry, w, repaired))
        else:
            conflict.append((entry, w, repaired))

    print(f"  ü -> ïi                   : {len(umlauts)} entries")
    for _, old, new in umlauts[:8]:
        print(f"      {old:<38} -> {new}")
    print(f"  reduplication repaired    : {len(redup)} entries")
    for _, old, new in redup:
        print(f"      {old:<38} -> {new}")
    print(f"  reduplication duplicate   : {len(dupe_redup)} quarantined "
          f"(same meaning as the existing entry)")
    for _, old, new in dupe_redup:
        print(f"      {old:<38} duplicate of {new}")
    if conflict:
        print(f"  COLLISION, different meaning: {len(conflict)} left untouched")
        for _, old, new in conflict:
            print(f"      {old:<38} would collide with {new} — needs a reading")

    if args.dry_run:
        print("\n  --dry-run: nothing written")
        return 0

    backup = args.db.with_name(
        f"{args.db.name}.bak.{datetime.now():%Y%m%d_%H%M%S}")
    shutil.copy2(args.db, backup)
    print(f"\n  backup -> {backup.name}")

    changed = 0
    for entry, old, new in umlauts + redup:
        changed += retarget(entry, old, new)
    stamp = datetime.now().strftime("%Y-%m-%d")
    for entry, old, new in dupe_redup:
        entry["quarantine_reason"] = (
            f"a reduplication with a scan-damaged half; repaired it is "
            f"{new!r}, which the lexicon already holds, so this record is a "
            f"duplicate [{stamp}]")
    drop = {id(e) for e, _, _ in dupe_redup}
    data.setdefault("quarantine", []).extend(e for e, _, _ in dupe_redup)
    data["lexicon"] = [e for e in lexicon if id(e) not in drop]
    # any ü left anywhere (variants, structures) folds too
    changed += retarget(data["lexicon"], UMLAUT, UMLAUT_AS)
    data.setdefault("meta", {})["umlaut_reduplication_fix"] = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "umlaut_entries": len(umlauts), "reduplications": len(redup),
        "reduplication_duplicates": len(dupe_redup),
        "strings_changed": changed,
    }
    args.db.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    print(f"  {changed} strings rewritten")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

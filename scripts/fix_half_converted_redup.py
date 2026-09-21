#!/usr/bin/env python3
"""
fix_half_converted_redup.py — withdraw reduplications whose halves disagree.

A Khasi total reduplication repeats the word exactly, so `baiñ-bain` cannot be
right: one half carries the tilde and the other does not. It reached the
lexicon from a different source file (kh_Nn1_0003) than the correct
`baiñ-baiñ` (kh_B_0350), and both survived as separate headwords.

It was not a harmless dead entry. Route 1 of the variant system offers any
attested spelling, so asking about `bain-bain` returned BOTH `baiñ-baiñ` and
`baiñ-bain` — the checker proposing a half-converted spelling as a legitimate
alternative.

An audit of all 29,694 entries for this shape — `a-b` where the halves match
once ï and ñ are folded away but differ as written — finds exactly one.

The distinctive part of its gloss (the example phrase) is carried over to the
surviving entry with the spelling corrected, so nothing is lost. The record
itself is quarantined rather than deleted, as every other withdrawal in this
lexicon has been.

    python3 scripts/fix_half_converted_redup.py --dry-run
    python3 scripts/fix_half_converted_redup.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "khasi_db.json"
FOLD = str.maketrans({"ñ": "n", "ï": "i"})


def _undo_tilde_as_i(half: str) -> str:
    """
    `baini` -> `bain`: the scan read a tilde as a trailing `i`.

    One of the OCR classes this lexicon is known to carry (see the `fi`/`ii`
    -> `ñ` family in the repair notes). It matters here because it produces a
    *near* reduplication rather than an exact one — `baini-bain` looks like
    two different words and so slips past a fold-equality test, while being
    the same damage as `baiñ-bain`.
    """
    return half[:-1] if half.endswith("i") and len(half) > 3 else half


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    data = json.loads(DB.read_text(encoding="utf-8"))
    lex = data["lexicon"]
    by_surface = {(e.get("form") or {}).get("surface"): e for e in lex}

    victims = []
    for e in lex:
        s = (e.get("form") or {}).get("surface") or ""
        if "-" not in s or " " in s:
            continue
        parts = s.split("-")
        if len(parts) != 2:
            continue
        a, b = parts
        if a == b:
            continue
        fa, fb = a.translate(FOLD), b.translate(FOLD)
        matched = fa == fb
        if not matched:
            # the tilde-as-i class: baini-bain is baiñ-baiñ, twice damaged
            matched = (_undo_tilde_as_i(a).translate(FOLD) == fb
                       or _undo_tilde_as_i(b).translate(FOLD) == fa)
        if not matched:
            continue
        # The survivor is whichever spelling is fully, correctly converted.
        # Neither half of `baini-bain` is, so fall back to the tilde form of
        # the undamaged half.
        if fa != a and _undo_tilde_as_i(a) == a:
            keeper = a + "-" + a
        elif fb != b and _undo_tilde_as_i(b) == b:
            keeper = b + "-" + b
        else:
            stem = _undo_tilde_as_i(a) if _undo_tilde_as_i(a) != a else b
            keeper = (stem[:-1] + "ñ") + "-" + (stem[:-1] + "ñ")
        victims.append((e, keeper))

    if not victims:
        print("no half-converted reduplications found")
        return 0

    for e, keeper in victims:
        s = e["form"]["surface"]
        surv = by_surface.get(keeper)
        print(f"  withdraw {e['entry_id']} {s!r}")
        print(f"     keeper {keeper!r} -> "
              f"{surv['entry_id'] if surv else 'NOT IN LEXICON — aborting'}")
        if surv is None:
            print("     refusing to withdraw an entry with no surviving spelling")
            return 1
        extra = [g for g in (e.get("semantics", {}).get("gloss") or [])
                 if g not in (surv.get("semantics", {}).get("gloss") or [])]
        fixed = [g.replace(s, keeper) for g in extra]
        print(f"     carrying over to keeper: {fixed}")

    if args.dry_run:
        print("dry run — nothing written")
        return 0

    backup = DB.with_suffix(f".json.bak.redup_{int(time.time())}")
    shutil.copy2(DB, backup)
    print(f"  backup -> {backup.name}")

    quarantine = data.setdefault("quarantine", [])
    for e, keeper in victims:
        s = e["form"]["surface"]
        surv = by_surface[keeper]
        extra = [g for g in (e.get("semantics", {}).get("gloss") or [])
                 if g not in (surv.get("semantics", {}).get("gloss") or [])]
        surv.setdefault("semantics", {}).setdefault("gloss", []).extend(
            g.replace(s, keeper) for g in extra)
        e["_withdrawn"] = {
            "reason": "half_converted_reduplication",
            "detail": (f"{s!r} repeats a root with only one half carrying the "
                       f"tilde; a total reduplication repeats exactly. "
                       f"Superseded by {keeper!r} ({surv['entry_id']}), whose "
                       f"gloss now carries this entry's example."),
            "superseded_by": surv["entry_id"],
            "withdrawn": time.strftime("%Y-%m-%d"),
        }
        quarantine.append(e)
        lex.remove(e)

    data["meta"].setdefault("changelog", []).append(
        f"v3.13: withdrew {len(victims)} half-converted reduplication(s) "
        f"({', '.join(v[0]['form']['surface'] for v in victims)}) — route 1 was "
        f"offering them as legitimate alternative spellings.")
    data["meta"]["version"] = "v3.13"
    DB.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  lexicon {len(lex)} entries, quarantine {len(quarantine)}")
    print("\nNOTE: re-run migrate_json_to_pg.py and build_db_extras.py.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

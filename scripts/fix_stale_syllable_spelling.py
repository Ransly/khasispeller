#!/usr/bin/env python3
"""
fix_stale_syllable_spelling.py — re-spell stored syllables after the
diacritic corrections.

The problem
-----------
The Word Details panel shows `ïaid` as its headword and then syllabifies it
`ia · id`, with a diphthong nucleus of `ia`. Both spellings cannot be right.

    form.surface           'ïaid'
    phonology.surface      'ïaid'
    phonology.derived      {"syllables": ["ia", "id"],
                            "diphthong_nucleus": ["ia"], ...}

`form.surface` and `phonology.surface` were corrected by the diaeresis
passes; `phonology.derived` was not, so it still holds the spelling the
lexicon used before the correction. `fix_ain_coda_tilde.py` names
`("phonology", "derived", "syllables")` in its KHASI_PATHS and did rewrite
it for the `ain -> aiñ` rule; `fix_ia_diaeresis.py` has no such path list,
so every `ia -> ïa` correction left the derived block behind. 909 entries
are affected, all of them `ïa` words.

What this pass does, and what it deliberately does not
------------------------------------------------------
It copies the CHARACTERS of the corrected surface onto the syllables that
are already recorded, leaving every syllable BOUNDARY exactly where it was.

That restraint matters. Some stored syllabifications are visibly wrong —
`ïam baid` is split `['ia', 'm', 'baid']` with a stray `m`, and
`jingïapart balang` splits `['jing','ia','par','t','ba','lang']` with a
stray `t`. Those are a real defect, but they are a DIFFERENT defect: this
pass is about spelling, and re-syllabifying 909 entries wholesale would mix
a boundary change nobody reviewed into a diacritic repair. The stray-letter
splits are left alone and recorded here so the next pass can find them.

Why a character copy is safe
----------------------------
An entry is only touched when the syllables and the surface hold the SAME
letters in the SAME order, differing only in diacritics — tested by
stripping combining marks from both and requiring equality. Unicode
normalisation preserves length for these characters (`ï` is one character
before and after), so the copy is position-for-position and cannot drop,
add or reorder anything. Any entry failing that test is skipped.

`pattern` (VV.VC) is NOT recomputed. `ï` is a vowel to
`khasi_engine.phonology`, so the classification does not change under
re-spelling — though note the g2p transcribes `ïaid` as /jaic/, treating the
`ï` as a glide, and whether the stored pattern should say CVVC instead is an
open question this pass does not answer.

    python3 scripts/fix_stale_syllable_spelling.py --dry-run
    python3 scripts/fix_stale_syllable_spelling.py
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import unicodedata
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "khasi_db.json"

# Characters that appear in a surface but never inside a stored syllable.
SEPARATORS = re.compile(r"[\s\-',.!()]")


def fold(text: str) -> str:
    """Drop combining marks: ï -> i, ñ -> n, á -> a. Length preserving."""
    return "".join(c for c in unicodedata.normalize("NFD", text)
                   if not unicodedata.combining(c))


def respell(syllables: list, surface: str):
    """Rewrite *syllables* with the characters of *surface*.

    Returns the new list, or None when the two do not hold the same letters
    and the copy would therefore be a guess.
    """
    bare = SEPARATORS.sub("", surface)
    joined = "".join(str(s) for s in syllables)
    if joined == bare:
        return None                      # already correct
    if fold(joined) != fold(bare):
        return None                      # different letters — not ours to fix
    out, i = [], 0
    for syl in syllables:
        n = len(str(syl))
        out.append(bare[i:i + n])
        i += n
    return out if i == len(bare) else None


def respell_nuclei(nuclei: list, corrected: str) -> list:
    """Re-spell each diphthong nucleus from the corrected syllable string."""
    folded = fold(corrected)
    out = []
    for n in nuclei:
        s = str(n)
        j = folded.find(fold(s))
        out.append(corrected[j:j + len(s)] if j >= 0 else s)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description="re-spell stale syllables")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--db", type=Path, default=DB)
    args = ap.parse_args()

    data = json.loads(args.db.read_text(encoding="utf-8"))
    changes, nuc_changes, odd = [], 0, []

    for e in data["lexicon"]:
        surface = ((e.get("form") or {}).get("surface") or "").strip()
        derived = ((e.get("phonology") or {}).get("derived") or {})
        syl = derived.get("syllables")
        if not surface or not isinstance(syl, list) or not syl:
            continue
        new = respell(syl, surface)
        if new is None:
            continue
        changes.append((e.get("entry_id"), surface, list(syl), new))
        # A single-letter syllable that is not a whole word is the
        # stray-letter split described in the docstring — recorded, not fixed.
        if any(len(s) == 1 for s in new) and len(new) > 1:
            odd.append((e.get("entry_id"), surface, new))
        if not args.dry_run:
            derived["syllables"] = new
            nuc = derived.get("diphthong_nucleus")
            if isinstance(nuc, list) and nuc:
                fixed = respell_nuclei(nuc, "".join(new))
                if fixed != nuc:
                    derived["diphthong_nucleus"] = fixed
                    nuc_changes += 1

    print(f"  entries to re-spell            : {len(changes)}")
    print(f"  of those, with a 1-letter split: {len(odd)} (left alone)")

    if args.dry_run:
        for eid, sf, old, new in changes[:15]:
            print(f"    {eid} | {sf}\n        {old}  ->  {new}")
        if odd:
            print("\n  stray single-letter splits, for a later pass:")
            for eid, sf, new in odd[:10]:
                print(f"    {eid} | {sf:24s} {new}")
        print("\n  --dry-run: nothing written")
        return 0

    backup = args.db.with_name(
        f"{args.db.name}.bak.{datetime.now():%Y%m%d_%H%M%S}")
    shutil.copy2(args.db, backup)
    print(f"  backup -> {backup.name}")

    stamp = datetime.now().strftime("%Y-%m-%d")
    data.setdefault("meta", {})["syllable_respell_fix"] = {
        "date": stamp,
        "syllables_respelled": len(changes),
        "diphthong_nuclei_respelled": nuc_changes,
        "single_letter_splits_left": len(odd),
        "note": "phonology.derived kept the pre-diaeresis spelling because "
                "fix_ia_diaeresis.py rewrote the surfaces only. Characters "
                "copied from the corrected surface; syllable boundaries "
                "unchanged.",
    }
    args.db.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    print(f"  re-spelled {len(changes)} syllable lists, "
          f"{nuc_changes} diphthong nuclei")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

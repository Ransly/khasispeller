#!/usr/bin/env python3
"""
fix_syllable_debris.py — a syllable is made of letters.

Two maintainer rulings:

  RULE 1  A digit is not a syllable.
  RULE 2  Punctuation and symbols are not syllables — EXCEPT `'` and `-`,
          which Khasi writes inside words. `'` marks the glottal stop
          (`nga'm`, `'riew`) and `-` joins compound and reduplicated forms
          (`bak-bak`), so both are kept wherever they sit inside a syllable.
          Every other symbol is stripped, and a syllable left empty is
          dropped.

And the consequence of RULE 2 that does the real work:

  RULE 3  A syllable beginning `(` opens a parenthetical, and everything
          from there to the end of the list belongs to it, not to the word.
          The source dictionary prints variants and notes in brackets and
          the syllabifier ran straight through them:

            jingphem   ['jing','phem','(jing','blem',')']
            jingri     ['jing','ri','(jin','ri,','jyn','ri:','cat','tle,',
                        'do','mes','tic','a','ni','mal','s)']

          `jingri` has swallowed an English gloss. Merely stripping the
          bracket would splice the variant INTO the word — `jingphem` would
          syllabify as jing-phem-jing-blem — so the list is truncated at the
          bracket instead.

          A CLOSED bracket is a different thing and is only dropped:
          `['(ka)','kam']`, `['(u)','khla']` carry the noun-class clitic the
          source prints ahead of the headword. Truncating there would leave
          30 records with no syllables at all, which is how the two shapes
          were told apart.

Evidence that the reading is right
----------------------------------
161 records carry a vowel-less syllable. After cleaning, 117 of them have
syllables that concatenate to exactly the letters of their own surface,
which they did not before. That is the check this pass is verified against.

The remaining 44 are left alone deliberately, because each is a DIFFERENT
defect that this ruling does not reach:

  * truncated lists — `ehe hep, u mareh u kulai` is syllabified only as far
    as `['e','he','hep']`; the syllabifier stopped at the comma long ago.
  * a spurious duplicated first syllable — `jingsynshar ka doh` is
    `['jing','jing','syn','shar','ka','doh']`, and `jingïang` is
    `['jing','ji','ngiang']`.
  * `h, h` and `j, j`, dictionary entries FOR the letters h and j. A lone
    consonant is not a syllable, but emptying the list would say less than
    leaving it, and no ruling covers the alphabet entries.
  * `dak 0`, where the digit is in the SURFACE, not only in the syllables.
    RULE 1 takes it out of the syllable list; correcting the headword is a
    separate call.

    python3 scripts/fix_syllable_debris.py --dry-run
    python3 scripts/fix_syllable_debris.py
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "khasi_db.json"

# The only non-letters Khasi writes inside a word.
IN_WORD = "'-"
# Everything a surface may hold that a syllable never does.
SURFACE_SEP = re.compile(r"[\s\-',.!()?:;\"]")


def clean(syllables: list, pattern: str):
    """Apply the three rules. Returns (syllables, pattern, changed)."""
    parts = pattern.split(".") if pattern else []
    aligned = len(parts) == len(syllables)
    out, out_parts = [], []
    for idx, raw in enumerate(map(str, syllables)):
        if raw.startswith("(") and raw.endswith(")"):
            # A self-contained bracket is the noun-class clitic the source
            # prints before a headword — `(ka) kam`, `(u) khla`. It is not
            # part of the word and not a variant: drop this token only.
            continue
        if raw.startswith("("):            # RULE 3 — parenthetical, stop here
            break
        kept = "".join(c for c in raw if c.isalpha() or c in IN_WORD)
        if not kept:                       # RULES 1 and 2 — nothing left
            continue
        out.append(kept)
        if aligned:
            # the pattern loses exactly the symbols the syllable lost
            p = parts[idx]
            out_parts.append(p[:len(kept)] if len(p) > len(kept) else p)
    changed = out != [str(s) for s in syllables]
    return out, (".".join(out_parts) if aligned else pattern), changed


def main() -> int:
    ap = argparse.ArgumentParser(description="strip debris from syllables")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--db", type=Path, default=DB)
    args = ap.parse_args()

    data = json.loads(args.db.read_text(encoding="utf-8"))
    vowels = set("".join((data.get("phonology") or {}).get("vowels_simple") or ""))
    if len(vowels) < 5:
        print(f"  ERROR: phonology.vowels_simple looks wrong: {sorted(vowels)}")
        return 1

    touched, reconstructs, emptied = [], 0, []
    for e in data["lexicon"]:
        derived = ((e.get("phonology") or {}).get("derived") or {})
        syl = derived.get("syllables")
        if not isinstance(syl, list) or not syl:
            continue
        # Every record is examined, not only those with a vowel-less
        # syllable. A first gate on "has a vowel-less member" missed 101
        # lists outright: `(jing` and `ri,` both CONTAIN vowels, so a
        # parenthetical whose every fragment happens to carry one was never
        # looked at. Debris is defined by its characters, not by its vowels.
        if not any((not c.isalpha()) and c not in IN_WORD
                   for s in syl for c in str(s)):
            continue                       # letters, ' and - only — clean
        new, newpat, changed = clean(syl, derived.get("pattern") or "")
        if not changed:
            continue
        if not new:
            emptied.append((e.get("entry_id"),
                            (e.get("form") or {}).get("surface"), syl))
            continue                       # never leave a word with no syllables
        surface = (e.get("form") or {}).get("surface") or ""
        if "".join(new) == SURFACE_SEP.sub("", surface):
            reconstructs += 1
        touched.append((e.get("entry_id"), surface, list(map(str, syl)), new,
                        derived.get("pattern"), newpat))
        if not args.dry_run:
            derived["syllables"] = new
            derived["pattern"] = newpat
            derived["syllable_count"] = len(new)

    print(f"  records cleaned                     : {len(touched)}")
    print(f"    of those, syllables now rebuild the surface: {reconstructs}")
    print(f"  records that would be left with none: {len(emptied)} (skipped)")

    if args.dry_run:
        for eid, sf, old, new, op, np in touched[:14]:
            print(f"    {sf}\n        {old}\n     -> {new}\n        {op}  ->  {np}")
        for eid, sf, syl in emptied:
            print(f"    SKIPPED (would empty): {eid} | {sf} | {syl}")
        print("\n  --dry-run: nothing written")
        return 0

    backup = args.db.with_name(
        f"{args.db.name}.bak.{datetime.now():%Y%m%d_%H%M%S}")
    shutil.copy2(args.db, backup)
    print(f"\n  backup -> {backup.name}")

    data.setdefault("meta", {})["syllable_debris_fix"] = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "records_cleaned": len(touched),
        "rebuild_their_surface": reconstructs,
        "left_with_none_skipped": len(emptied),
        "note": "Digits and symbols removed from syllable lists; ' and - kept "
                "as in-word characters. A syllable opening '(' truncates the "
                "list — the source prints variants and notes in brackets and "
                "the syllabifier ran through them.",
    }
    args.db.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    print(f"  written: {len(data['lexicon'])} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

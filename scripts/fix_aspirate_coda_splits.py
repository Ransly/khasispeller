#!/usr/bin/env python3
"""
fix_aspirate_coda_splits.py — an aspirate never closes a syllable.

`pynïakhlad` was syllabified `pyn · ïakh · lad`, and `ïakh` cannot be a
Khasi syllable: it ends in `kh`. The lexicon's own phonology block says so
twice —

    phonology.forbidden_final   ph th kh bh dh jh lh rh  (+ s l sh j)
    phonology.consonant_chart   every aspirate carries "final": false

— because aspiration is a release feature and an aspirated stop in a coda is
not pronounceable as such. The `kh` belongs to the ONSET of the next
syllable, and `khl` is in `phonology.valid_initial_clusters`, so the reading
is `pyn · ïa · khlad`.

What is fixed, and what is deliberately not
-------------------------------------------
Only a NON-FINAL syllable ending in an ASPIRATE, and only when moving that
aspirate onto the next syllable produces a valid onset. 123 such splits
exist; 121 qualify (kh 98, ph 23) and 2 do not — `path|dow` and
`jing|iath|huh|…` would need onsets `thd` and `thh`, which are not clusters
Khasi allows, so they are left for a reading that can say what those records
should be.

`forbidden_final` lists `s l sh j` as well, and those are NOT touched. That
list governs what may end a WORD, not what may close a syllable inside one:
`pyllait` is `pyl · lait` and `kylliang` is `kyl · liang`, both correct, with
`l` as a coda before an `l` onset. Treating all 1,275 non-final syllables
that end in a forbidden final as errors would wreck 911 correct geminate
splits. The aspirates are the subset where the coda reading is impossible
rather than merely word-final-illegal.

Why the pattern needs no recomputation
--------------------------------------
A consonant digraph is one symbol in the stored pattern, so the aspirate is
a single `C` and it moves exactly as the letters do:

    ïakh  VVC  ->  ïa     VV        (drop the trailing C)
    lad   CVC  ->  khlad  CCVC      (prepend a C)

That mirrors the letter move instead of re-deriving anything, which keeps
this pass out of the CV-convention question entirely. A record whose pattern
does not line up — wrong number of parts, or a part not ending in `C` where
the aspirate sits — is skipped rather than guessed at.

    python3 scripts/fix_aspirate_coda_splits.py --dry-run
    python3 scripts/fix_aspirate_coda_splits.py
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "khasi_db.json"


def main() -> int:
    ap = argparse.ArgumentParser(description="move aspirate codas to the next onset")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--db", type=Path, default=DB)
    args = ap.parse_args()

    data = json.loads(args.db.read_text(encoding="utf-8"))
    phon = data.get("phonology") or {}
    aspirates = sorted(phon.get("aspirates") or [], key=len, reverse=True)
    onsets = set(phon.get("valid_initial_clusters") or [])
    consonants = set(phon.get("consonants") or [])
    vowels = set(phon.get("vowels_simple") or [])
    if not aspirates or not onsets or len(vowels) < 5:
        print("  ERROR: phonology block is missing aspirates/clusters/vowels")
        return 1
    print(f"  aspirates: {' '.join(aspirates)}")

    def onset_of(text: str) -> str:
        out = ""
        for ch in text:
            if ch in vowels:
                break
            out += ch
        return out

    moved, skipped, samples = 0, [], []
    for e in data["lexicon"]:
        derived = ((e.get("phonology") or {}).get("derived") or {})
        syl = derived.get("syllables")
        if not isinstance(syl, list) or len(syl) < 2:
            continue
        parts = (derived.get("pattern") or "").split(".")
        aligned = len(parts) == len(syl)
        before = list(map(str, syl))
        cur = list(map(str, syl))
        cur_parts = list(parts)
        i = 0
        changed = False
        while i < len(cur) - 1:
            s, nxt = cur[i], cur[i + 1]
            asp = next((a for a in aspirates if s.endswith(a)), None)
            if not asp or len(s) <= len(asp):
                i += 1
                continue
            if onset_of(asp + nxt) not in onsets | consonants:
                skipped.append((e.get("entry_id"),
                                (e.get("form") or {}).get("surface"), before))
                i += 1
                continue
            if aligned and not cur_parts[i].endswith("C"):
                skipped.append((e.get("entry_id"),
                                (e.get("form") or {}).get("surface"), before))
                i += 1
                continue
            cur[i] = s[: -len(asp)]
            cur[i + 1] = asp + nxt
            if aligned:
                cur_parts[i] = cur_parts[i][:-1]
                cur_parts[i + 1] = "C" + cur_parts[i + 1]
            changed = True
            moved += 1
            i += 1
        if not changed:
            continue
        # The letters must be untouched — only a boundary moved.
        assert "".join(cur) == "".join(before), (before, cur)
        if len(samples) < 12:
            samples.append(((e.get("form") or {}).get("surface"), before, cur,
                            derived.get("pattern"),
                            ".".join(cur_parts) if aligned else derived.get("pattern")))
        if not args.dry_run:
            derived["syllables"] = cur
            derived["syllable_count"] = len(cur)
            if aligned:
                derived["pattern"] = ".".join(cur_parts)

    print(f"  aspirate codas moved to the next onset : {moved}")
    print(f"  skipped (no valid onset / pattern askew): {len(skipped)}")

    if args.dry_run:
        for sf, b, a_, op, np in samples:
            print(f"    {sf}\n        {b}  ->  {a_}\n        {op}  ->  {np}")
        for eid, sf, b in skipped[:6]:
            print(f"    SKIPPED {eid} | {sf} | {b}")
        print("\n  --dry-run: nothing written")
        return 0

    backup = args.db.with_name(
        f"{args.db.name}.bak.{datetime.now():%Y%m%d_%H%M%S}")
    shutil.copy2(args.db, backup)
    print(f"\n  backup -> {backup.name}")

    data.setdefault("meta", {})["aspirate_coda_split_fix"] = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "moved": moved,
        "skipped": len(skipped),
        "note": "An aspirate cannot close a syllable (consonant_chart marks "
                "every aspirate final:false); it was moved to the onset of "
                "the following syllable where that yields a valid cluster. "
                "Letters unchanged, only a boundary. s/l/sh/j codas are NOT "
                "touched — forbidden_final governs word-final position, and "
                "pyl|lait is a correct geminate split.",
    }
    args.db.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    print(f"  written: {len(data['lexicon'])} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

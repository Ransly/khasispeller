#!/usr/bin/env python3
"""
fix_syllable_splits_and_i_class.py — two maintainer rulings on the stored
phonology.

RULE 1 — a syllable must have a vowel nucleus.
    The stored syllabification strands a coda consonant as a syllable of its
    own: `ïam baid` splits ['ïa', 'm', 'baid'] and `pynskhern da` splits
    ['pyn', 'skher', 'n', 'da']. A consonant alone is not a Khasi syllable,
    so the orphan joins the syllable before it:

        ['ïa', 'm', 'baid']  ->  ['ïam', 'baid']

    Merging two adjacent syllables cannot change the letters or their order —
    it only moves a boundary — so this is safe by construction. The pattern
    is merged at the same index so the two stay in step.

    Only a LETTERS-ONLY orphan is merged. 151 records hold punctuation or a
    digit as a "syllable" (['dod','dod','(dud','dud',')'], ['dak','0'],
    ['shi','eh','!','yn','nai']) — a parenthetical variant or scan debris
    captured into the list. Merging `)` into `long` would manufacture
    `long)`, so those are left for a pass that can read what they should be.

RULE 2 — `ï` is a vowel, not a consonant.
    The stored patterns disagree with themselves: `ïaid` ['ïa','id'] is
    VV.VC, with `ï` as V, while `ïaroh` ['ïa','roh'] is CV.CVC, with `ï` as
    C. The engine already agrees with the first — `ï` is in
    `phonology.VOWELS` and `g2p` tokenises it V — so the C readings are the
    error.

    ONLY the `ï` positions change. Recomputing the whole pattern was measured
    and rejected: it would rewrite 3,942 records, and 3,497 of those turn on
    a different question this ruling does not address — whether a final `w`
    in a diphthong is V (`ksew` is stored CCVV) or C. A third convention,
    `ïing` written VC with `ïi` as one vowel, is likewise left alone; `ï`
    there already counts as V. A syllable whose grapheme count does not match
    its pattern length is skipped rather than guessed at.

RULE 3 — one spelling correction, supplied by the maintainer.
    `jingïapart balang` is an OCR misreading of `jingïapait`. The lexicon
    already carries `jingïapait` (kh_DB_026128, root `pait`) and the paired
    form `jingïapait - jingïapiet`; `pait` heads some twenty entries, while
    `part` is a root in this record alone. Surface, normalised form, root,
    phonology surface, syllable and IPA all move together.

    python3 scripts/fix_syllable_splits_and_i_class.py --dry-run
    python3 scripts/fix_syllable_splits_and_i_class.py
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "khasi_db.json"

# entry_id -> (wrong, right) for RULE 3.
SPELLING = {"kh_DB_001063": ("jingïapart", "jingïapait")}


def _load_g2p():
    import sys
    sys.path.insert(0, str(ROOT))
    from khasi_engine import g2p
    return g2p


def graphemes(text: str, digraphs) -> list[str]:
    """Split into pattern units: a consonant digraph is one, else one char."""
    out, i, s = [], 0, str(text)
    while i < len(s):
        for g in digraphs:
            if s.startswith(g, i):
                out.append(g)
                i += len(g)
                break
        else:
            out.append(s[i])
            i += 1
    return out


def has_vowel(text: str, vowels) -> bool:
    return any(c in vowels for c in str(text))


def merge_orphans(syl: list, pattern: str, vowels):
    """RULE 1. Returns (syllables, pattern) or None when nothing changed."""
    out = [str(s) for s in syl]
    parts = pattern.split(".") if pattern else []
    aligned = len(parts) == len(out)
    changed = False
    i = 0
    while i < len(out):
        orphan = (not has_vowel(out[i], vowels)
                  and out[i].replace("'", "").isalpha() and out[i])
        if orphan and len(out) > 1:
            if i > 0:
                out[i - 1] += out.pop(i)
                if aligned:
                    parts[i - 1] += parts.pop(i)
                changed = True
                i -= 1
                continue
            # a vowel-less FIRST syllable belongs to what follows
            out[0] += out.pop(1)
            if aligned:
                parts[0] += parts.pop(1)
            changed = True
            continue
        i += 1
    if not changed:
        return None
    return out, (".".join(parts) if aligned else pattern)


def fix_i_class(syl: list, pattern: str, digraphs) -> tuple[str, int]:
    """RULE 2. Returns (pattern, flips) — only `ï` positions may change."""
    parts = pattern.split(".") if pattern else []
    if len(parts) != len(syl):
        return pattern, 0
    flips, out = 0, []
    for s, p in zip(syl, parts):
        if "ï" not in str(s):
            out.append(p)
            continue
        g = graphemes(s, digraphs)
        if len(g) != len(p):
            out.append(p)          # a different convention — not ours to touch
            continue
        chars = list(p)
        for j, unit in enumerate(g):
            if unit == "ï" and chars[j] == "C":
                chars[j] = "V"
                flips += 1
        out.append("".join(chars))
    return ".".join(out), flips


def main() -> int:
    ap = argparse.ArgumentParser(description="syllable splits and ï class")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--db", type=Path, default=DB)
    args = ap.parse_args()

    g2p = _load_g2p()
    data = json.loads(args.db.read_text(encoding="utf-8"))

    # Vowels come from the file being edited, NOT from phonology.VOWELS.
    # That constant is bound at import time from a cache which, in
    # PostgreSQL mode, is not populated until a KhasiDB is constructed —
    # until then `phonology.VOWELS` is frozenset("" + "ï"), a set of one.
    # A maintenance script that imports the module without building a
    # database therefore sees almost no vowels, and every syllable looks
    # like a vowel-less orphan. Reading the block straight out of the JSON
    # keeps this pass self-consistent with the data it rewrites.
    vowels = set("".join((data.get("phonology") or {}).get("vowels_simple") or ""))
    if len(vowels) < 5:
        print(f"  ERROR: phonology.vowels_simple looks wrong: {sorted(vowels)}")
        return 1
    print(f"  vowels: {''.join(sorted(vowels))}")
    digraphs = sorted(g2p._CONS_DIGRAPHS, key=len, reverse=True)
    merged, flipped, flips_total, respelt = [], [], 0, []

    for e in data["lexicon"]:
        derived = ((e.get("phonology") or {}).get("derived") or {})
        syl = derived.get("syllables")
        if not isinstance(syl, list) or not syl:
            continue
        pattern = derived.get("pattern") or ""
        before = list(map(str, syl))

        # RULE 3 first, so the merged/flipped work sees the corrected letters.
        eid = e.get("entry_id")
        if eid in SPELLING:
            wrong, right = SPELLING[eid]
            form = e.setdefault("form", {})
            for key in ("surface", "normalized"):
                if isinstance(form.get(key), str):
                    form[key] = form[key].replace(wrong, right)
            lem = e.setdefault("lemma", {})
            if lem.get("root") == "part":
                lem["root"] = "pait"
            if lem.get("root_id") == "root_part":
                lem["root_id"] = "root_pait"
            phon = e.setdefault("phonology", {})
            if isinstance(phon.get("surface"), str):
                phon["surface"] = phon["surface"].replace(wrong, right)
            syl = derived["syllables"] = [
                "pait" if s in ("par", "part") else s
                for s in syl if s != "t"
            ]
            pattern = derived["pattern"] = ".".join(
                p for p, s in zip(pattern.split("."), before) if s != "t"
            ) if len(pattern.split(".")) == len(before) else pattern
            # the merged 'par'+'t' slot is now 'pait' -> CVVC
            parts = pattern.split(".")
            if len(parts) == len(syl):
                parts = ["".join("V" if u in g2p._VOWEL_SINGLE else "C"
                                 for u in graphemes(s, digraphs))
                         if s == "pait" else p
                         for p, s in zip(parts, syl)]
                pattern = derived["pattern"] = ".".join(parts)
            try:
                ipa = phon.get("ipa")
                if isinstance(ipa, dict):
                    words = [g2p.to_ipa(t).strip("/")
                             for t in (form.get("surface") or "").split()]
                    ipa["surface"] = "/" + " ".join(w for w in words if w) + "/"
                    ipa["generated_by"] = g2p.VERSION
            except Exception:
                pass
            # RULE 1 may find nothing left to merge on this entry, and the
            # count lives in its branch, so keep it in step here too.
            derived["syllable_count"] = len(syl)
            respelt.append((eid, form.get("surface"), list(syl), pattern))

        # RULE 1
        got = merge_orphans(syl, pattern, vowels)
        if got:
            syl, pattern = got
            derived["syllables"] = syl
            derived["pattern"] = pattern
            derived["syllable_count"] = len(syl)
            merged.append((e.get("entry_id"),
                           (e.get("form") or {}).get("surface"), before, syl))

        # RULE 2
        newpat, n = fix_i_class(syl, pattern, digraphs)
        if n:
            derived["pattern"] = newpat
            flips_total += n
            flipped.append((e.get("entry_id"),
                            (e.get("form") or {}).get("surface"),
                            pattern, newpat))

    print(f"  RULE 1  orphan codas merged   : {len(merged)} entries")
    print(f"  RULE 2  ï reclassified C -> V : {len(flipped)} entries, "
          f"{flips_total} positions")
    print(f"  RULE 3  spelling corrected    : {len(respelt)} entry")

    if args.dry_run:
        print("\n  --- merges ---")
        for eid, sf, b, a in merged[:12]:
            print(f"    {sf:26s} {b}  ->  {a}")
        print("\n  --- ï reclassified ---")
        for eid, sf, b, a in flipped[:12]:
            print(f"    {sf:26s} {b}  ->  {a}")
        print("\n  --- spelling ---")
        for eid, sf, s, p in respelt:
            print(f"    {eid} | {sf} | {s} | {p}")
        print("\n  --dry-run: nothing written")
        return 0

    backup = args.db.with_name(
        f"{args.db.name}.bak.{datetime.now():%Y%m%d_%H%M%S}")
    shutil.copy2(args.db, backup)
    print(f"\n  backup -> {backup.name}")

    data.setdefault("meta", {})["syllable_split_and_i_class_fix"] = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "orphan_codas_merged": len(merged),
        "i_diaeresis_reclassified_entries": len(flipped),
        "i_diaeresis_positions": flips_total,
        "spelling_corrected": {k: v[1] for k, v in SPELLING.items()},
        "note": "A syllable needs a vowel nucleus, so a stranded coda joins "
                "the syllable before it. ï is a vowel; only ï positions in "
                "the stored pattern changed. Punctuation-as-syllable records "
                "and the w-in-diphthong convention are untouched.",
    }
    args.db.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    print(f"  written: {len(data['lexicon'])} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

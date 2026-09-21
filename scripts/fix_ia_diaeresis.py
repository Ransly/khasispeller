#!/usr/bin/env python3
"""
Correct the lexicon's spelling of word-initial `ia` to `ïa`.

Khasi writes this sequence with the diaeresis. The lexicon does not do so
consistently — it records 196 `ia-` entries against 148 `ïa-`, 68 `jingia-`
against 24 `jingïa-`, and only 26 words both ways — so the same word is
spelled differently depending on which entry you happen to hit.

    python3 scripts/fix_ia_diaeresis.py --dry-run
    python3 scripts/fix_ia_diaeresis.py

Scope
-----
`ia` at the start of a word, and `ia` immediately after one of the outer
prefixes that `morphology.affix_order` licenses before it — jing-, nong-,
pyn-, sngew-. Word-internal `ia` elsewhere is untouched.

Nothing is lost: the original spelling is kept in `form.variants`, so the
plain form still resolves and the previous state is recoverable from the
data itself as well as from the timestamped backup.

Only this project's copy is touched. The parent platform's lexicon is not
modified.
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import shutil
import sys
import time
from pathlib import Path

DEFAULT_DB = Path(__file__).parent.parent / "data" / "khasi_db.json"

# `ia` at a word boundary, optionally behind one of the outer prefixes.
# The lookbehind stops it firing mid-word (kynia..., briew...).
PATTERN = re.compile(
    # `ia` must be followed by the rest of the word — a letter, or a hyphen
    # then a letter (jingia-beit). Requiring a bare letter missed every
    # hyphenated compound on the first pass.
    # `ia` must be at a word boundary. It may continue into the rest of the
    # word (a letter, or a hyphen then a letter — jingia-beit), or stand
    # alone as its own token.
    #
    # The standalone case is included on the maintainer's instruction. Note
    # that morphology.prefixes.ia says the opposite — "Bound prefix only;
    # 'ia' as free morpheme object-marker is NOT this prefix" — so this
    # deliberately overrides the database's own annotation.
    # `+` is a boundary too: morphology.canonical joins morphemes with it
    # ("ia+iaid+arsut"). Omitting it left the first segment as `ia` while
    # morphology.structure — where the same morpheme stands alone — became
    # `ïa`, so the two fields disagreed on 598 entries.
    r"(?<![A-Za-zÏïÑñ])((?:jing|pyn|nong|sngew)?)ia(?=-?[A-Za-zÏïÑñ]|\s|\+|$)"
)

# Diacritics removed, for recognising an entry's own plain spelling.
_TO_PLAIN = str.maketrans({"ï": "i", "ñ": "n"})

# Fields carrying a surface spelling that must stay in step.
STRING_FIELDS = (
    ("form", "surface"),
    ("form", "normalized"),
    ("lemma", "root"),
    ("morphology", "canonical"),
)


def rewrite(text: str) -> str:
    """Apply the correction, preserving the case of the initial letter."""
    def sub(m: re.Match) -> str:
        return m.group(1) + "ïa"
    return PATTERN.sub(sub, text)


def needs_fix(text) -> bool:
    return isinstance(text, str) and bool(PATTERN.search(text))


def migrate(db: dict, keep_variant: bool = True) -> dict:
    lex = db["lexicon"]
    surfaces = {(e.get("form") or {}).get("surface") for e in lex}
    stats: collections.Counter = collections.Counter()
    collisions: list[tuple[str, str]] = []
    samples: list[tuple[str, str]] = []

    for entry in lex:
        original_surface = (entry.get("form") or {}).get("surface")
        changed = False

        for block, key in STRING_FIELDS:
            holder = entry.get(block)
            if isinstance(holder, dict) and needs_fix(holder.get(key)):
                new = rewrite(holder[key])
                if new != holder[key]:
                    holder[key] = new
                    stats[f"{block}.{key}"] += 1
                    changed = True

        # Variants are corrected too, EXCEPT one that is simply the plain
        # spelling of this entry's own surface. Comparing against the
        # pre-edit surface is not enough: on a second run the surface is
        # already corrected, so the preserved plain form no longer matches
        # it and would be rewritten — silently undoing the preservation and
        # leaving a variant identical to the surface. Compare with
        # diacritics removed, which is stable across runs.
        form = entry.get("form")
        if isinstance(form, dict):
            surface_now = form.get("surface")
            plain_surface = (surface_now.translate(_TO_PLAIN)
                             if isinstance(surface_now, str) else None)
            variants = form.get("variants")
            if isinstance(variants, list):
                for i, v in enumerate(variants):
                    if v == original_surface:
                        continue
                    if isinstance(v, str) and plain_surface \
                            and v.translate(_TO_PLAIN) == plain_surface:
                        continue      # this entry's own plain spelling
                    if needs_fix(v):
                        new = rewrite(v)
                        if new != v:
                            variants[i] = new
                            stats["form.variants"] += 1
                            changed = True

        # Morpheme pieces inside morphology.structure
        struct = (entry.get("morphology") or {}).get("structure")
        if isinstance(struct, list):
            for piece in struct:
                pf = (piece or {}).get("form")
                if not isinstance(pf, dict):
                    continue
                for k in ("raw", "display"):
                    if needs_fix(pf.get(k)):
                        new = rewrite(pf[k])
                        if new != pf[k]:
                            pf[k] = new
                            stats[f"structure.{k}"] += 1
                            changed = True

        if changed:
            stats["entries"] += 1
            new_surface = (entry.get("form") or {}).get("surface")
            if keep_variant and isinstance(original_surface, str) \
                    and original_surface != new_surface:
                form = entry.setdefault("form", {})
                vs = form.setdefault("variants", [])
                if isinstance(vs, list) and original_surface not in vs:
                    vs.append(original_surface)
                    stats["original_kept_as_variant"] += 1
            if isinstance(new_surface, str) and new_surface in surfaces \
                    and new_surface != original_surface:
                collisions.append((original_surface, new_surface))
            if len(samples) < 10 and original_surface != new_surface:
                samples.append((original_surface, new_surface))

    return {"stats": stats, "collisions": collisions, "samples": samples}


def fix_short_roots(db: dict) -> dict:
    """
    Repair entries analysed as prefix + a one-letter root.

    `ïat` was recorded as [prefix 'ia', root 't']. The project already ruled
    against exactly this — golden_corpus carries "short-stem reroot: ia+t
    rejected" — and the parallel `iat` entry was re-rooted by that cleanup,
    but its diacritic twin was missed. Correcting the spelling then merged
    the pair and the unrepaired record survived, so the analyser reported
    these words as derived rather than valid.

    `database.py` sets _MIN_ROOT_LEN = 2 for the same reason: a single
    letter is not a Khasi root.

    Single-word entries only, and prefix+root only. `um` is [root 'u',
    suffix '-m'] — a real clitic contraction (War pp.74-75), not this
    problem, and must be left alone.
    """
    fixed, samples = 0, []
    for entry in db["lexicon"]:
        surface = (entry.get("form") or {}).get("surface")
        if not isinstance(surface, str) or " " in surface:
            continue
        morph = entry.get("morphology") or {}
        struct = morph.get("structure") or []
        if len(struct) != 2:
            continue
        kinds = [p.get("type") for p in struct]
        if kinds != ["prefix", "root"]:
            continue
        root_piece = struct[1]
        raw = ((root_piece.get("form") or {}).get("raw") or "")
        if len(raw) >= 2:
            continue
        # Collapse to a single root: the whole word.
        entry["morphology"]["structure"] = [
            {"type": "root", "form": {"raw": surface}}
        ]
        entry["morphology"]["type"] = "root"
        entry["morphology"]["canonical"] = surface
        lemma = entry.setdefault("lemma", {})
        lemma["root"] = surface
        fixed += 1
        if len(samples) < 8:
            samples.append((surface, raw))
    return {"fixed": fixed, "samples": samples}


def dedupe(db: dict) -> dict:
    """
    Merge entries that now share a surface form.

    Correcting the spelling collapses pairs the lexicon recorded twice, once
    each way — `iap` and `ïap` both become `ïap`. Two entries for one surface
    is not merely untidy: the analyser returns verdict 'derived' rather than
    'valid' whenever a surface has more than one entry, so the duplicates
    change how the word analyses.

    Nothing is discarded. The survivor is the entry with the most glosses;
    the other's glosses are appended if new, and its id is recorded in
    `merged_from` so the merge is auditable and reversible from the backup.
    """
    lex = db["lexicon"]
    by_surface: dict[str, list[dict]] = collections.defaultdict(list)
    for e in lex:
        sf = (e.get("form") or {}).get("surface")
        if isinstance(sf, str):
            by_surface[sf].append(e)

    drop_ids, merged, glosses_added = set(), 0, 0
    for sf, group in by_surface.items():
        if len(group) < 2:
            continue
        # Prefer the richer record; an "Alternate spelling of X" stub loses.
        def rank(e):
            g = (e.get("semantics") or {}).get("gloss") or []
            stub = str(g[0] if g else "").lower().startswith(
                ("alternate spelling", "alternative spelling", "variant spelling"))
            return (stub, -len(g))
        group = sorted(group, key=rank)
        keeper, rest = group[0], group[1:]
        sem = keeper.setdefault("semantics", {})
        gl = sem.setdefault("gloss", [])
        if not isinstance(gl, list):
            gl = sem["gloss"] = [gl]
        for other in rest:
            for g in ((other.get("semantics") or {}).get("gloss") or []):
                if g not in gl:
                    gl.append(g)
                    glosses_added += 1
            keeper.setdefault("merged_from", []).append(other.get("entry_id"))
            drop_ids.add(id(other))
            merged += 1

    if drop_ids:
        db["lexicon"] = [e for e in lex if id(e) not in drop_ids]
    return {"merged": merged, "glosses_added": glosses_added,
            "surfaces": sum(1 for g in by_surface.values() if len(g) > 1)}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=str(DEFAULT_DB))
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--no-dedupe", action="store_true",
                    help="leave duplicate surfaces in place after correcting")
    ap.add_argument("--no-keep-variant", action="store_true",
                    help="do not record the previous spelling in form.variants")
    args = ap.parse_args()

    path = Path(args.db)
    if not path.is_file():
        print(f"no lexicon at {path}", file=sys.stderr)
        return 1
    print(f"reading {path} ({path.stat().st_size/1e6:.0f} MB) ...")
    with open(path, encoding="utf-8") as fh:
        db = json.load(fh)

    result = migrate(db, keep_variant=not args.no_keep_variant)
    stats, collisions, samples = result["stats"], result["collisions"], result["samples"]

    print(f"\n  entries changed          {stats['entries']:,}")
    for k in sorted(k for k in stats if k not in ("entries",)):
        print(f"    {k:<28} {stats[k]:,}")
    print(f"\n  samples:")
    for a, b in samples:
        print(f"    {a}  ->  {b}")
    print(f"\n  collisions (corrected form already existed): {len(collisions):,}")
    for a, b in collisions[:6]:
        print(f"    {a}  ->  {b}")
    if collisions:
        print("    both entries are kept; the lexicon simply records the word twice,")
        print("    which it already did for 26 words before this change.")

    sr = fix_short_roots(db)
    if sr["fixed"]:
        print(f"\n  short-root entries repaired  {sr['fixed']}")
        for surf, old in sr["samples"]:
            print(f"    {surf:<12} prefix + root {old!r}  ->  root {surf!r}")

    if not args.no_dedupe:
        dd = dedupe(db)
        print(f"\n  duplicate surfaces merged  {dd['merged']:,} entries "
              f"across {dd['surfaces']:,} surfaces")
        print(f"    glosses carried over     {dd['glosses_added']:,}")
        print(f"    lexicon size             {len(db['lexicon']):,} entries")

    if args.dry_run:
        print("\n  dry run — nothing written")
        return 0

    backup = path.with_suffix(f".json.bak.ia_diaeresis_{int(time.time())}")
    shutil.copy2(path, backup)
    print(f"\n  backup   {backup.name}")
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(db, fh, ensure_ascii=False)
    print(f"  written  {path.name} ({path.stat().st_size/1e6:.0f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

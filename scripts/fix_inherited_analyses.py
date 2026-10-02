#!/usr/bin/env python3
"""
fix_inherited_analyses.py — single words that carry ANOTHER word's analysis.

The bug
-------
The Word Details panel for `katkum` 'according to' showed

    Syllables  kat · ba        Pattern  CVC.CV        Root  katba (adverb)

— the analysis of `katba` 'as long as', the headword eight entries above
it. The entry itself (kh_DB_009786) holds it: lemma.root, morphology.canonical
and morphology.structure all say `katba`, and phonology.derived holds katba's
syllables. Every copy of the lexicon back to the oldest backup has it, so it
came in with the dictionary import, where a word that was not recognised as a
new headword inherited the analysis of the headword before it:

    bel   -> bela, ben, benis       brap -> braw       brop -> brot
    bniat -> bnieij                 bun  -> jyngbam, jynbam

Measured over the 11,890 single-word entries (no space, no comma):

    morphology.type "root" but lemma.root is another word      123
    stored syllables that do not spell the entry's own word    529

Multi-word phrase entries are left alone: their syllables are partial by
construction, and the panel analyses a word, not a phrase. khasi_spell/api.py
now refuses to show any stored syllabification that does not spell the word,
so a phrase's partial list can no longer be shown either.

What this pass does
-------------------
ROOTS. An entry the lexicon itself rules a bare root (morphology.type
"root") is its own root, so lemma.root, lemma.root_id, morphology.canonical
and the single root part of morphology.structure are set to the surface.
Only a root that differs in its LETTERS is touched: `'am-duh` rooted in
`am-duh`, or `ïeng` in `ieng`, is the same word written without a mark.

SYLLABLES. Rebuilt only from what the lexicon already says, never by a
syllabification rule: a 2026-09-10 pass that re-derived patterns
orthographically was withdrawn (see recompute_syllable_patterns.py), because
C/V structure is phonological. In order, for each entry:

  1. same boundaries, missing marks — the stored syllables spell the word once
     ï/ñ/accents are ignored: the word is re-cut at the same boundaries, so
     `ia·shong` becomes `ïa·shong`;
  2. a doubled syllable — `kyn·kyn·bat` loses the repeat when the rest spells
     the word; the stored pattern loses the matching segment;
  3. otherwise each hyphen-separated part is rebuilt on its own:
       a. the part is itself an entry whose syllables spell it: those
          syllables and that entry's pattern;
       b. the part has one vowel group, and the lexicon already reads that
          group as the nucleus of a one-syllable word, so one syllable: the
          part itself, with the pattern the lexicon gives that syllable
          elsewhere (at least 80% of its readings), else no pattern —
          `buiam` stays unresolved, since no one-syllable word has `uia`;
       c. the part splits into two or more entries (each at least two
          letters, with a vowel, with syllables that spell it) and EVERY such
          split gives the same syllables: those, with the pieces' patterns —
          `katkum` = `kat` + `kum` -> kat·kum, CVC.CVC. A cut must sit
          between two consonants or after a vowel, unless it follows the
          word's own prefixes (jing|ai); none may fall inside a leading
          prefix (`jinging` is not ji|nging). A contraction with an inner
          apostrophe (`phin'n`) is not rebuilt.
     A part that none of these settles leaves the entry unresolved.
  4. unresolved: the wrong syllables are withdrawn (empty list, no pattern).
     The panel then shows no syllables rather than another word's.

TRAITS. phonology.derived also holds the traits read off those syllables. For
a rebuilt or withdrawn entry they described the other word too: `bnieij`
carried `bniat`'s diphthong /ia/. A stored diphthong (IPA) is kept only if the
word's own pronunciation, from khasi_engine.g2p, contains it; a stored "long
vowel" is cleared, since vowel length is not written (War 2001 pp.49-54) and
so cannot be checked from the spelling. Re-cut and de-doubled entries keep
theirs: their analysis was the word's own all along.

    python3 scripts/fix_inherited_analyses.py --dry-run
    python3 scripts/fix_inherited_analyses.py
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import shutil
import sys
import unicodedata
from datetime import datetime
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "khasi_db.json"
sys.path.insert(0, str(ROOT))

FOLD = str.maketrans("ïñáéíóúý", "inaeiouy")
VOWELS = set("aeiouyïáéíóúý")
VOWEL_GROUP = re.compile(r"[aeiouyïáéíóúý]+")
PATTERN_SHARE = 0.8


def nfc(s) -> str:
    return unicodedata.normalize("NFC", s or "")


def bare(s) -> str:
    """Lower case, without hyphens, apostrophes or spaces."""
    return re.sub(r"[\s\-'’]", "", nfc(s).lower())


def fold(s) -> str:
    return bare(s).translate(FOLD)


def surface(e) -> str:
    return nfc((e.get("form") or {}).get("surface") or "").strip()


def derived(e) -> dict:
    return (e.get("phonology") or {}).get("derived") or {}


def is_single_word(w: str) -> bool:
    return bool(w) and " " not in w and "," not in w


def spells(syl, word) -> bool:
    return bool(syl) and bare("".join(syl)) == bare(word)


def pattern_parts(pat, n):
    parts = (pat or "").split(".") if pat else []
    return parts if len(parts) == n else None


class Inventory:
    """Syllabifications the lexicon gets right: an entry's syllables spell it."""

    def __init__(self, lexicon, exclude_ids, prefixes=()):
        # Multi-letter prefixes only: `i` would let a split start anywhere.
        self.prefixes = {bare(x) for x in prefixes if len(bare(x)) > 1}
        self.entry = {}                                   # bare word -> (syl, pattern parts | None)
        self.reading = collections.defaultdict(collections.Counter)
        self.nuclei = set()                               # vowel groups of one-syllable words
        for e in lexicon:
            w = surface(e)
            syl = derived(e).get("syllables")
            if (e.get("entry_id") in exclude_ids or not is_single_word(w)
                    or not spells(syl, w)):
                continue
            syl = [nfc(s).lower() for s in syl]
            parts = pattern_parts(derived(e).get("pattern"), len(syl))
            self.entry.setdefault(bare(w), (syl, parts))
            for s, p in zip(syl, parts or []):
                self.reading[s][p] += 1
            if len(syl) == 1 and len(VOWEL_GROUP.findall(bare(w))) == 1:
                self.nuclei.add(fold(VOWEL_GROUP.findall(bare(w))[0]))

    def pattern_of(self, syllable):
        c = self.reading.get(syllable)
        if not c:
            return None
        p, n = c.most_common(1)[0]
        return p if n / sum(c.values()) >= PATTERN_SHARE else None

    def leading_prefixes(self, part):
        """End offsets of the prefixes *part* starts with (jing, jingpyn, ...)."""
        ends, i = [], 0
        while True:
            hit = max((x for x in self.prefixes
                       if part.startswith(x, i) and len(part) > i + len(x)),
                      key=len, default=None)
            if not hit:
                return ends
            i += len(hit)
            ends.append(i)

    def splits(self, part):
        """Every way to write *part* as two or more known entries.

        A cut must be one the syllable structure allows on its face: between
        two consonants (kat|kum) or after a vowel (be|la), unless it follows
        the word's own prefixes, where any cut is a morpheme boundary
        (jing|ai, nong|pyn|ïa|id). And no cut may fall INSIDE a leading
        prefix: `jinging` is not ji|nging.
        """
        ends = self.leading_prefixes(part)
        inside = {k for i, e in enumerate(ends)
                  for k in range(ends[i - 1] + 1 if i else 1, e)}

        def allowed(i, at_prefix):
            if i in inside:
                return False
            if at_prefix:
                return True
            return part[i] not in VOWELS          # the next piece opens on a consonant

        @lru_cache(None)
        def go(i, prev_start):
            if i == len(part):
                return [()]
            # the piece just placed is itself one of the leading prefixes
            at_prefix = i in ends and (prev_start == 0 or prev_start in ends)
            if i and not allowed(i, at_prefix):
                return []
            out = []
            for j in range(i + 2, len(part) + 1):
                piece = part[i:j]
                if piece in self.entry and VOWELS & set(piece):
                    out += [(piece,) + rest for rest in go(j, i)]
            return out
        return [s for s in go(0, 0) if len(s) >= 2]

    def rebuild_part(self, part):
        """(syllables, pattern parts with None for unknown) or None."""
        if part in self.entry:
            syl, parts = self.entry[part]
            return list(syl), list(parts) if parts else [self.pattern_of(s) for s in syl]
        groups = VOWEL_GROUP.findall(part)
        if len(groups) == 1 and fold(groups[0]) in self.nuclei:
            return [part], [self.pattern_of(part)]
        results = set()
        for seg in self.splits(part):
            syl, pats = [], []
            for piece in seg:
                ps, pp = self.entry[piece]
                syl += ps
                pats += pp if pp else [self.pattern_of(s) for s in ps]
            results.add((tuple(syl), tuple(pats)))
        if len({r[0] for r in results}) == 1:
            syl, pats = next(iter(results))
            return list(syl), list(pats)
        return None


def recut(word, syllables, keep_marks=False):
    """*word*'s own letters at the boundaries of *syllables*.

    The syllables must spell the word letter for letter once marks are
    ignored, so each syllable takes as many of the word's letters as it has.
    With *keep_marks*, an apostrophe inside a syllable stays where it was
    (`k'iar` -> `k'ïar`); otherwise syllables hold letters only.
    """
    letters, out, i = bare(word), [], 0
    for s in syllables:
        piece = ""
        for ch in nfc(s).lower():
            if ch in "'’":
                piece += ch if keep_marks else ""
            elif not ch.isspace() and ch != "-":
                piece += letters[i]
                i += 1
        out.append(piece)
    return out


def repair_syllables(e, inv):
    """(kind, syllables, pattern) for an entry whose syllables are wrong."""
    w = surface(e)
    old = [nfc(s) for s in derived(e).get("syllables") or []]
    old_pat = derived(e).get("pattern")

    if fold("".join(old)) == fold(w):
        return "recut", recut(w, old, keep_marks=True), old_pat

    for i in range(1, len(old)):
        if bare(old[i]) == bare(old[i - 1]):
            syl = old[:i] + old[i + 1:]
            if spells(syl, w):
                parts = pattern_parts(old_pat, len(old))
                pat = ".".join(parts[:i] + parts[i + 1:]) if parts else None
                return "doubled", recut(w, syl), pat

    if re.search(r"\w['’]", w):
        return "withdrawn", [], None              # a contraction (phin'n): not rebuilt
    syl, pats = [], []
    for part in (bare(p) for p in re.split(r"-", w)):
        if not part:
            continue
        got = inv.rebuild_part(part)
        if got is None:
            return "withdrawn", [], None
        syl += got[0]
        pats += got[1]
    if not spells(syl, w):
        return "withdrawn", [], None
    pat = ".".join(pats) if all(pats) else None
    return "rebuilt", recut(w, syl), pat


def main() -> int:
    ap = argparse.ArgumentParser(description="single words carrying another word's analysis")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--db", type=Path, default=DB)
    args = ap.parse_args()

    data = json.loads(args.db.read_text(encoding="utf-8"))
    lexicon = data["lexicon"]

    root_bad = [e for e in lexicon
                if is_single_word(surface(e))
                and (e.get("morphology") or {}).get("type") == "root"
                and (e.get("lemma") or {}).get("root")
                and fold(e["lemma"]["root"]) != fold(surface(e))]
    syl_bad = [e for e in lexicon
               if is_single_word(surface(e)) and derived(e).get("syllables")
               and not spells(derived(e)["syllables"], surface(e))]

    # morphology.prefixes is keyed by the prefix itself: {"jing": {...}, ...}
    prefixes = list((data.get("morphology") or {}).get("prefixes") or {})
    inv = Inventory(lexicon, {e.get("entry_id") for e in syl_bad}, prefixes)
    plans = [(e, *repair_syllables(e, inv)) for e in syl_bad]
    tally = collections.Counter(p[1] for p in plans)

    print(f"  single-word entries               : "
          f"{sum(is_single_word(surface(e)) for e in lexicon):,}")
    print(f"  roots set back to the word itself : {len(root_bad)}")
    print(f"  syllables that spell another word : {len(syl_bad)}")
    for kind in ("recut", "doubled", "rebuilt", "withdrawn"):
        print(f"      {kind:<10}: {tally[kind]}")
    print(f"      rebuilt without a pattern: "
          f"{sum(1 for p in plans if p[1] == 'rebuilt' and not p[3])}")

    if args.dry_run:
        for kind in ("recut", "doubled", "rebuilt", "withdrawn"):
            rows = [p for p in plans if p[1] == kind]
            print(f"\n  --- {kind} ({len(rows)}) ---")
            for e, _, syl, pat in rows[:40]:
                print(f"    {e['entry_id']} {surface(e):<22} "
                      f"{'·'.join(derived(e)['syllables']):<22} -> "
                      f"{'·'.join(syl) or '(none)':<24} {pat or ''}")
        print("\n  --- roots ---")
        for e in root_bad[:40]:
            print(f"    {e['entry_id']} {surface(e):<22} root {e['lemma']['root']!r} -> "
                  f"{surface(e)!r}")
        print("\n  --dry-run: nothing written")
        return 0

    from khasi_engine import g2p                  # before the backup: fail early

    backup = args.db.with_name(f"{args.db.name}.bak.{datetime.now():%Y%m%d_%H%M%S}")
    shutil.copy2(args.db, backup)
    print(f"\n  backup -> {backup.name}")

    for e in root_bad:
        w = surface(e)
        lem = e.setdefault("lemma", {})
        lem["root"] = w
        lem["root_id"] = f"root_{w.lower()}"
        morph = e.setdefault("morphology", {})
        morph["canonical"] = w
        roots = [p for p in morph.get("structure") or [] if p.get("type") == "root"]
        if len(morph.get("structure") or []) == 1 and len(roots) == 1:
            roots[0].setdefault("form", {})["raw"] = w

    traits_cleared = []
    for e, kind, syl, pat in plans:
        d = e.setdefault("phonology", {}).setdefault("derived", {})
        d["syllables"] = syl
        d["syllable_count"] = len(syl)
        d["pattern"] = pat
        if kind in ("rebuilt", "withdrawn"):
            ipa = (g2p.convert(surface(e)) or {}).get("surface") or ""
            dn = d.get("diphthong_nucleus")
            if dn and not all(x in ipa for x in (dn if isinstance(dn, list) else [dn])):
                d["diphthong_nucleus"] = None
                traits_cleared.append(e["entry_id"])
            if d.get("has_long_vowel"):
                d["has_long_vowel"] = None
                traits_cleared.append(e["entry_id"])
    print(f"  entries with another word's traits cleared: {len(set(traits_cleared))}")

    stamp = datetime.now().strftime("%Y-%m-%d")
    data.setdefault("meta", {})["inherited_analysis_fix"] = {
        "date": stamp,
        "roots_set_to_surface": sorted(e["entry_id"] for e in root_bad),
        "syllables": {k: sorted(p[0]["entry_id"] for p in plans if p[1] == k)
                      for k in ("recut", "doubled", "rebuilt", "withdrawn")},
        "traits_cleared": sorted(set(traits_cleared)),
        "note": "Single-word entries that carried another word's root or "
                "syllables (an import error: a word inherited the analysis of "
                "the headword before it; katkum showed katba's). Bare roots "
                "now root themselves; syllables rebuilt only from the "
                "lexicon's own syllabifications, else withdrawn. Phrases "
                "untouched. Built by scripts/fix_inherited_analyses.py.",
    }
    args.db.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  written: {args.db.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

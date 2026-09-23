"""
variants.py — alternative spellings that differ only in diacritics.

Khasi writes ï and ñ, and both are awkward to type. Writers routinely drop
them: `iathuh` for `ïathuh`, `ain` for `aiñ`, `jingiathuh` for `jingïathuh`.
The engine already tolerates this — `database.py` folds diacritics at lookup,
so the word is understood and accepted — but tolerance is not guidance. A
writer who wants the standard spelling is never told what it is.

This module closes that gap. It answers "what other spellings of this word
exist?" and returns them as *alternatives*, never as errors. Whichever form
the writer typed stays valid; `canonical` marks the one the lexicon records,
so a caller can show which is the standard form without forcing it.

Both directions are offered. The lexicon itself records 107 words twice,
once with the diacritic and once without (khwain/khwaiñ, luin/luiñ,
niang/ñiang khriat), so neither spelling can be called wrong.

Three routes, because the diacritic can sit in different places:

  1. The whole word. `database.py` maintains a folded index mapping a
     diacritic-free key to the lexicon entries that do carry diacritics, so
     `iathuh` -> `ïathuh` is a single lookup.

  2. The plain spelling, for input that already carries a diacritic —
     `aiñ` -> `ain`. This is the reverse of route 1.

  3. The root of a derived form. `jingïathuh` is not a lexicon entry — it is
     `jing-` + `ïathuh`, assembled by the morphology. Route 1 finds nothing.
     So the word is parsed, the root's variants are looked up, and the
     variant is substituted back into the surface.

  5. A rule for the -ain / -aiñ root, wherever it falls.

  6. A rule for a short spelling whose `ï`-form is recorded under an
     `i`-form root: `ing` is `ïing`, `oh` is `ïoh`. The folded index cannot
     see these, because it strips the diaeresis and so links `ïing` only to
     `iing`, leaving `ing` stranded.

  7. A spelling the dictionary records in its own gloss — "Same as byniej",
     "Abbrev. of shnong", "Also spelt as kyieng". Khasi writes several words
     two ways, a glottal-stop form beside a fuller one: `k'ing` and `kyieng`
     'a wasp', `'nong` and `shnong` 'village', `'lap` and `slap` 'rain'.
     Extracted by `scripts/build_spelling_links.py`. The lexicon records many of these
     only in the plain spelling — `spain` is an entry glossed "Bandage; To
     swathe" and there is no `spaiñ` headword — so no lookup route can
     offer the standard form. Guarded by case, because -ain is also a
     common English and Indic ending: `Spain` the country and `Hussain` the
     name must be left alone, and the corpus shows capitalisation separates
     them reliably (`Hussain` is capitalised in 53 of 53 uses, `spain`
     splits 34 capitalised against 8 lowercase). See `foreign.is_proper_case`.

  4. A morphological rule, for the reciprocal prefix. Lookup routes can only
     offer a spelling the lexicon happens to record, and for this prefix it
     records both inconsistently: 196 `ia-` entries against 148 `ïa-`, 68
     `jingia-` against 24 `jingïa-`, and only 26 words appear both ways. So
     `ïabeit` cannot be offered for `iabeit` by lookup — it is simply not
     listed. The rule supplies it.

     `morphology.prefixes.ia` records the prefix as productive with 664
     entries (`reciprocal_plural_subject`), and `morphology.affix_order`
     gives the stacking order ["jing-", "nong-", "pyn-", "sngew-", "ia-",
     ROOT] — which is where the jing+ia and pyn+ia patterns come from. War
     (2001) writes this prefix `ïa-`.

Which spelling is standard?
---------------------------
The corpus cannot answer this. It contains **zero** tokens with ï or ñ
across 6.67M words — the diacritics were either stripped in preparation or
simply not typed. Corpus frequency would therefore always favour the plain
form, which is evidence about typing habits, not about orthography.

The lexicon is used as the authority instead: it is curated and page-cited
to published grammars and dictionaries. A variant it records exactly is
flagged `canonical`; one reconstructed from a root, or folded from the
input, is not.
"""
from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, asdict
from itertools import combinations
from typing import Any, Optional

# Plain form -> its diacritic counterpart in Khasi orthography.
# ï and ñ are the two characters in regular use; the acute-accented vowels
# appear in the lexicon and mark vowel quality (see War 2001 pp.49-54).
DIACRITIC_OF = {
    "i": "ï", "n": "ñ",
    "a": "á", "e": "é", "o": "ó", "u": "ú", "y": "ý",
}

# Substituting at every position at once explodes combinatorially and yields
# nothing real. Khasi words rarely carry more than two diacritics.
MAX_SUBSTITUTIONS = 2

# The reciprocal / plural-subject prefix. The lexicon spells it both ways;
# War (2001) writes it with the diaeresis. Outer prefixes come from
# morphology.affix_order, which puts ia- innermost before the root.
RECIPROCAL_PLAIN = "ia"
RECIPROCAL_DIACRITIC = "ïa"
# Prefixes that can sit OUTSIDE the reciprocal ia-. Khasi stacks these, so
# they must be treated as a chain rather than a fixed list of single items:
# `jingpynialehkai` is jing- + pyn- + ïa- + lehkai, and a flat list of one-step
# outers matched none of it. Before this, `pynialehkai` was offered
# `pynïalehkai` while `jingpynialehkai` was offered nothing at all, and the
# stacked forms that did work — `jingpynïaid` — worked only because the
# lexicon happened to record them, not by rule.
RECIPROCAL_OUTER_UNITS = ("jing", "pyn", "nong", "sngew")

# `affix_order` in the lexicon records jing- > pyn- > ïa- as the attested
# ordering; three steps covers that with room to spare, and bounds the search.
MAX_RECIPROCAL_OUTER_DEPTH = 3

# The remainder after ia- must look like a real base, or every root that
# merely begins with those letters (iap 'die', iar) gets mangled. Two is
# better than three: it picks up iaai, iaid, iada and their jing-/nong-/pyn-
# derivatives, and a higher share of what it then proposes turns out to be a
# spelling the lexicon already records (22 of 182, against 17 of 171 at
# three) — which is the signal that the extra fires are real.
MIN_RECIPROCAL_BASE = 2

# The -ain / -aiñ pair. Khasi writes ñ and typists drop the tilde, so the
# lexicon records many words only in the plain form: `spain` is a Khasi
# entry glossed "Bandage; To swathe", and there is no `spaiñ` headword to
# find. A lookup route can only offer what the lexicon happens to hold, so
# for those a rule is needed.
#
# Guarded three ways, because -ain is also a common English and Indic
# ending:
#
#   * the word must be **lowercase in context**. Capitalisation marks a
#     name, and the corpus shows it is a reliable signal: `Hussain` is
#     capitalised in 53 of 53 uses and `Gohain` in 32 of 32, while `spain`
#     splits 34 capitalised (the country) against 8 lowercase (the Khasi
#     word). The caller applies this via `foreign.is_proper_case`.
#   * the word must be **recorded in the lexicon**. `plain`, `remain`,
#     `domain`, `maintain` and `detain` are accepted by the morphology gate
#     without being entries; requiring a curated headword excludes all five
#     while keeping `spain`, `pain`, `jain`, `rain`, `main`, `lain`.
#   * the stem must be long enough that `-ain` is an ending rather than the
#     whole word.
AIN_PLAIN = "ain"
AIN_DIACRITIC = "aiñ"
MIN_AIN_STEM = 1

# A compound whose FINAL element carries the tilde: `saitjain` is sait +
# jain, and `jaiñ` is attested, so the compound is `saitjaiñ`. Route 3 walks
# the other way — it strips known prefixes — so it never reaches a root
# sitting at the end, and `saitjain` alone is 890 corpus tokens.
#
# The tail must be a real root rather than the bare word `ain`, or every
# word merely ending in those letters gets rewritten; and the head must be a
# known Khasi word, which is what keeps the rule off `hussain` (`huss`),
# `again` (`ag`) and `spain`-the-country.
MIN_AIN_TAIL = 4
# Three, not two. A two-letter head lets English through: `remain` splits as
# `re` + `main` and `domain` as `do` + `main`, and both `re` and `do` are
# genuine Khasi words, so the guard passes and `remaiñ`/`domaiñ` are
# proposed. All 60 high-confidence candidates in the review export have a
# head of three or more; the 26 it excludes were flagged low-confidence for
# exactly this reason.
MIN_AIN_HEAD = 3

# An attested -aiñ root can also open or sit inside a compound, not only end
# it: `jainsem` is `jaiñ` + `sem`, `jainkup-jainsem` is `jaiñkup-jaiñsem`.
# The tail rule never reaches those, so `jain` alone offered `jaiñ` while
# every compound built on it offered nothing.
#
# Matching is anchored at the start of each hyphen-separated part. That
# alone is not enough — `maintain` really does begin with `main` — so the
# word must also be a lexicon entry, the same guard the headword rule uses.
# `maintain`, `remain`, `domain`, `plain` and `again` are none of them
# curated headwords; `jainsem`, `jainkhor` and `mainker` all are.
#
# The remainder after the root must be a known word of at least two
# characters. One letter is too weak — `bain` + `a` and `dain` + `i` are
# real clitics, but `baiña` and `daiñi` are not words anyone writes.
MIN_AIN_REMAINDER = 2

# Diacritic -> plain, for offering the reverse direction.
_TO_PLAIN = str.maketrans({v: k for k, v in DIACRITIC_OF.items()})


@dataclass
class SpellingVariant:
    """An alternative spelling of an accepted word."""

    variant: str
    source: str          # "lexicon" (whole word) | "root" (derived form)
                         # | "folded" (the diacritic-free spelling)
                         # | "corpus" (a corpus word in lexicon orthography)
    canonical: bool = False   # is this the spelling the lexicon records?
    root: Optional[str] = None
    # English gloss of the variant, "" when the lexicon records none. The
    # reader is being asked to choose between two spellings; the meaning is
    # what tells them the offer is the same word — `jaiñsem` "an outer
    # garment of Khasi women" rather than a near-miss on another entry.
    gloss: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


def _generate(word: str, max_subs: int = MAX_SUBSTITUTIONS) -> list[str]:
    """Diacritic-bearing spellings of *word*, up to `max_subs` substitutions."""
    positions = [i for i, ch in enumerate(word) if ch in DIACRITIC_OF]
    out: list[str] = []
    for n in range(1, min(max_subs, len(positions)) + 1):
        for combo in combinations(positions, n):
            chars = list(word)
            for i in combo:
                chars[i] = DIACRITIC_OF[chars[i]]
            out.append("".join(chars))
    return out


def _candidate_roots(word: str, db: Any) -> list[str]:
    """
    Plausible roots of *word*, by stripping a known prefix.

    Deliberately cheap — string slicing plus a hash lookup. The morphology
    module holds the prefix inventory, loaded from the lexicon data, so this
    stays in step with the grammar without invoking the parser.
    """
    try:
        from khasi_engine import morphology
        prefixes = sorted((morphology.PREFIXES or {}), key=len, reverse=True)
    except Exception:
        prefixes = ["jing", "nong", "pyn", "ia", "ba"]
    out = []
    for pfx in prefixes:
        if len(word) > len(pfx) + 2 and word.startswith(pfx):
            out.append(word[len(pfx):])
    return out


def _invalid_standalone() -> frozenset:
    """Tokens the phonology block marks as meaningless on their own."""
    global _INVALID_CACHE
    if _INVALID_CACHE is None:
        try:
            from khasi_engine import phonology
            raw = phonology._load_db().get("semantically_invalid_standalone") or []
            _INVALID_CACHE = frozenset(str(x).lower() for x in raw)
        except Exception:
            _INVALID_CACHE = frozenset({"ng", "sh", "kh", "ph", "th", "bh", "dh", "jh"})
    return _INVALID_CACHE


_INVALID_CACHE = None


def _outer_chains(word: str) -> list[str]:
    """
    Every chain of stackable prefixes that *word* opens with, shortest first.

    For `jingpynialehkai` this yields "", "jing", "jingpyn" — so the ia- that
    sits behind two prefixes is reached, not just the one behind zero or one.
    Always includes "" so a bare `iabeit` still matches.
    """
    chains = [""]
    frontier = [("", 0)]
    while frontier:
        cur, depth = frontier.pop()
        if depth >= MAX_RECIPROCAL_OUTER_DEPTH:
            continue
        for unit in RECIPROCAL_OUTER_UNITS:
            nxt = cur + unit
            if word.startswith(nxt) and nxt not in chains:
                chains.append(nxt)
                frontier.append((nxt, depth + 1))
    return sorted(chains, key=len)


def _reciprocal_forms(word: str, db: Any) -> list[str]:
    """
    Apply the reciprocal-prefix rule: ia- -> ïa-, in the attested positions.

        ia    + base  ->  ïa    + base      iabeit    -> ïabeit
        jing  + ia... ->  jing  + ïa...     jingiathuh -> jingïathuh
        pyn   + ia... ->  pyn   + ïa...     pyniadei   -> pynïadei

    Guarded so it fires only where ia- really is the prefix. The base must
    be at least MIN_RECIPROCAL_BASE characters and known to the lexicon;
    without that, roots that merely start with those letters — iap 'die',
    iar, iaid — would be rewritten into nonsense.
    """
    out: list[str] = []
    for outer in _outer_chains(word):
        rest = word[len(outer):]
        if not rest.startswith(RECIPROCAL_PLAIN):
            continue
        base = rest[len(RECIPROCAL_PLAIN):]
        probe = base.lstrip("-")          # jingia-beit
        if len(probe) < MIN_RECIPROCAL_BASE or not db.is_known(probe):
            continue
        # db.is_known() accepts sub-phonemic tokens such as a bare digraph,
        # so 'iang' would split as ia- + 'ng' and be rewritten. The phonology
        # block lists exactly these; a base drawn from it is not a base.
        if probe in _invalid_standalone():
            continue
        out.append(outer + RECIPROCAL_DIACRITIC + base)
    return out


def _is_reduplication(word: str) -> bool:
    """`dain-dain`, `jain-jain` — route 1 already converts both halves."""
    for sep in ("-", ""):
        if sep and sep in word:
            left, right = word.split(sep, 1)
            if left == right:
                return True
    return False


def _ain_form(word: str, db: Any) -> Optional[str]:
    """
    `spain` -> `spaiñ`, when the lexicon records the plain spelling.

    Returns None when the word does not end in -ain, when the diacritic
    form is already an entry (route 1 handles that and marks it canonical),
    or when the lexicon does not record this word at all.
    """
    if not word.endswith(AIN_PLAIN) or len(word) - len(AIN_PLAIN) < MIN_AIN_STEM:
        return None
    if word not in db._surface_index:
        return None                       # not a curated Khasi headword
    if _is_reduplication(word):
        # `dain-dain` is a headword, so this route reaches it and would
        # rewrite only the second half — `dain-daiñ`. Route 1 already
        # converts both halves correctly.
        return None
    candidate = word[: -len(AIN_PLAIN)] + AIN_DIACRITIC
    if candidate in db._surface_index:
        return None                       # already attested; route 1 has it
    return candidate


_AIN_ROOTS: Optional[dict] = None


def _ain_root_map(db: Any) -> dict:
    """Plain spelling -> attested -aiñ spelling, longest root first."""
    global _AIN_ROOTS
    if _AIN_ROOTS is None:
        roots = sorted(
            (w for w in db._surface_index
             if w.endswith(AIN_DIACRITIC) and "-" not in w
             and " " not in w and len(w) >= MIN_AIN_TAIL),
            key=len, reverse=True)
        _AIN_ROOTS = {r[:-1] + "n": r for r in roots}
    return _AIN_ROOTS


def _parts_all_known(word: str, db: Any) -> bool:
    """
    A hyphenated token whose every element is a known Khasi word.

    `_ain_elements` used to require a curated headword, which excluded the
    compounds the checker itself accepts. `suk-u-sain` is the case: it is not
    a headword, so route 5 declined it, and `_ain_compound` declined it too
    because that route asks whether the material *before* the -ain root is a
    single known word -- and `suk-u` is not one, though `suk` and `u` both
    are. Between them the two routes covered headwords and two-part
    compounds and missed three-part ones entirely, so `suk-u-sain` was never
    offered `suk-u-saiñ`.

    Requiring every element to be a real word keeps this conservative: it
    admits exactly the tokens `speller._check_hyphenated` already treats as
    valid, and nothing assembled from fragments.

    Named entities are not a concern here. `Spain` and `Hussain` are excluded
    upstream by capitalisation in `_collect_variants`, which is where that
    judgement belongs -- the corpus has `Hussain` capitalised in 53 of 53
    uses, against `spain` the Khasi noun in 8 lowercase.
    """
    if "-" not in word:
        return False
    parts = [p for p in word.split("-") if p]
    return len(parts) > 1 and all(db.is_known(p) for p in parts)


def _ain_elements(word: str, db: Any) -> Optional[str]:
    """
    `jainsem` -> `jaiñsem`, `jainkup-jainsem` -> `jaiñkup-jaiñsem`.

    Substitutes an attested -aiñ root wherever it *opens* a
    hyphen-separated part, so every occurrence is converted rather than the
    first. Returns None when nothing matches.
    """
    if not (word in db._surface_index or _parts_all_known(word, db)):
        return None
    roots = _ain_root_map(db)
    parts = word.split("-")
    changed = False
    for i, part in enumerate(parts):
        hit = None
        for plain, dia in roots.items():
            if part == plain:
                hit = dia
                break
            if (part.startswith(plain)
                    and len(part) - len(plain) >= MIN_AIN_REMAINDER
                    and _spelled_exactly(part[len(plain):], db)):
                hit = dia + part[len(plain):]
                break
        # The root map holds only -aiñ spellings the lexicon actually
        # attests, and most -aiñ forms are not written down: `saiñ` is not a
        # headword, it is derived from the headword `sain` by the same rule
        # route 5 applies to a whole word. Without this second attempt the
        # element route could only ever convert a part whose tilde spelling
        # someone had already curated, which is why `suk-u-sain` stayed
        # unconverted while `sain` on its own was offered `saiñ`.
        if hit is None:
            hit = _ain_form(part, db)
        if hit is not None:
            parts[i] = hit
            changed = True
    out = "-".join(parts)
    return out if changed and out != word else None


def _spelled_exactly(word: str, db: Any) -> bool:
    """True when the lexicon really spells something this way.

    `db.is_known()` deliberately falls back to the orthographic fold, which
    maps ñ->n — so `is_known("main")` and `is_known("tain")` are both True
    purely because `maiñ` and `taiñ` exist. That is right for looking a word
    UP and wrong for licensing a compound seam: it makes every English word
    containing `ain` look like a Khasi compound, and offered `maintaiñ` for
    `maintain`. Same defect the solid-compound splitter had in
    khasi_engine/complex.py, and the same remedy — an exact surface, root
    lemma, or phrase token, never a fold.
    """
    return (word in db._surface_index
            or word in getattr(db, "_root_index", {})
            or word in getattr(db, "_compound_known", set()))


def _ain_compound(word: str, db: Any) -> Optional[str]:
    """
    `saitjain` -> `saitjaiñ`, when the final element is an attested -aiñ root.

    Returns None unless the word ends in a lexicon root whose tilde form is
    recorded, the material before it is itself a known Khasi word, and the
    whole thing is not already a headword or a reduplication.
    """
    if not word.endswith(AIN_PLAIN) or word in db._surface_index:
        return None
    if _is_reduplication(word):
        return None
    for dia in _ain_tails(db):
        plain = dia[:-1] + "n"
        if word == plain or not word.endswith(plain):
            continue
        head = word[: -len(plain)].rstrip("-")
        if len(head) < MIN_AIN_HEAD or not _spelled_exactly(head, db):
            continue
        return word[: -len(plain)] + dia
    return None


_AIN_TAILS: Optional[list] = None


def _ain_tails(db: Any) -> list:
    """Attested -aiñ roots, longest first so the fullest match wins."""
    global _AIN_TAILS
    if _AIN_TAILS is None:
        _AIN_TAILS = sorted(
            (w for w in db._surface_index
             if w.endswith(AIN_DIACRITIC) and len(w) >= MIN_AIN_TAIL
             and " " not in w),
            key=len, reverse=True)
    return _AIN_TAILS


_SHORT_TO_DIACRITIC: Optional[dict] = None


def _short_form_map(db: Any) -> dict:
    """
    Short spellings that have a `ï`-form recorded under an `i`-form root.

    `ing` is written `ïing`, but no lookup route finds it: the folded index
    strips the diaeresis, so it links `ïing` to `iing`, and `ing` folds to
    itself. What connects them is the lexicon's own root field — `ïing` is
    recorded under the root `iing`, whose tail is `ing`.

    (Since 2026-09-08 that root is spelled `ïing`, `ii` being illegal; the
    tail is `ing` either way, so both spellings of the root are accepted.)

    That is the whole rule, and it is deliberately narrow. Sharing a root is
    far too weak on its own: 975 pairs satisfy it, linking `ar` 'two' to
    `khát-ar` and `bakla` to `jingïai-bakla`, which are derivations rather
    than spellings of the same word. Requiring the root to be the `i`-form
    whose tail is the short spelling reduces that to seven:

        eng ew ing oh op or um   ->   ïeng ïew ïing ïoh ïop ïor ïum

    The corpus supports the reading. `ing` occurs 3,167 times in contexts
    that can only be the noun — `sha ing lajong` 'to his own house',
    `hapoh ing` 'inside the house' — even though the lexicon also carries a
    rare verb sense of the same spelling, glossed "to be burnt".
    """
    global _SHORT_TO_DIACRITIC
    if _SHORT_TO_DIACRITIC is None:
        out: dict = {}
        # `all_entries()` yields the flattened shape, where the surface is
        # `surface_form` and the root is `root_lemma` — not the nested
        # form/lemma objects of the JSON on disk.
        for entry in db.all_entries():
            surface = (entry.get("surface_form") or "").lower()
            root = (entry.get("root_lemma") or "").lower()
            if not surface.startswith("ï") or " " in surface or len(root) < 3:
                continue
            if root.startswith("i"):
                short = root[1:]
            elif root.startswith("ï") and root[1] == "i":
                # `ïing` is the one member of this set whose i-form root
                # would be `iing` — an illegal `ii` — so the 2026-09-08 pass
                # had to record it in the ï-form, and requiring an i-form
                # root would drop `ing` -> `ïing` entirely.
                #
                # The condition stays narrow deliberately. Accepting ANY
                # ï-form root re-admits exactly what this rule exists to
                # exclude: `ïar` is recorded under the root `ïar`, so `ar`
                # 'two' would be offered `ïar`. Only a root whose i-form is
                # itself illegal qualifies.
                short = root[1:]
            else:
                continue
            if short in db._surface_index:
                # Several entries can share the root — `ïing`, `ïing-basa`,
                # `ïing bishar`. The answer wanted for `ing` is the bare
                # word, so keep the shortest rather than whichever the
                # lexicon happened to list first.
                prev = out.get(short)
                if prev is None or len(surface) < len(prev):
                    out[short] = surface
        _SHORT_TO_DIACRITIC = out
    return _SHORT_TO_DIACRITIC


def _short_i_form(word: str, db: Any) -> Optional[str]:
    """`ing` -> `ïing`. None when the word has no such recorded form."""
    return _short_form_map(db).get(word)


_SPELLING_LINKS: Optional[dict] = None


def _corpus_fold_map(db: Any) -> dict:
    """{reduced spelling: admitted corpus form}, for corpus words the pool
    admitted with diacritics restored (`iatreilang` -> `ïatreilang`).

    Rebuilt only when the set changes size, which it does once, at start-up.
    """
    forms = getattr(db, "_corpus_forms", None) or ()
    cache = getattr(db, "_corpus_fold_cache", None)
    if cache is None or cache[0] != len(forms):
        m = {}
        for f in forms:
            plain = f.translate(_TO_PLAIN)
            if plain != f:
                m.setdefault(plain, []).append(f)
        cache = (len(forms), m)
        try:
            db._corpus_fold_cache = cache
        except Exception:
            pass
    return cache[1]


def _spelling_links() -> dict:
    """
    Alternative spellings the dictionary records in its own glosses.

    The printed source notes them inside definitions — "Same as byniej",
    "Abbrev. of shnong", "Also spelt as kyieng" — and nothing read them
    until now. 108 entries name another headword this way, so the lexicon
    already knew that `'nong` is `shnong` and `k'ing` is `kyieng`.

    Extracted once by `scripts/build_spelling_links.py`; an empty map when
    that has not been run, so every caller degrades to the other routes.
    """
    global _SPELLING_LINKS
    if _SPELLING_LINKS is None:
        import json
        from pathlib import Path

        path = Path(__file__).resolve().parent.parent / "data" / "spelling_links.json"
        try:
            _SPELLING_LINKS = json.loads(path.read_text(encoding="utf-8"))["links"]
        except Exception:
            _SPELLING_LINKS = {}
    return _SPELLING_LINKS


# "abbrev. of briew", "contr. of kyieng", "short for shnong" — the printed
# dictionary's way of recording that one headword is a CLIPPED form of
# another, not another way of spelling it.
_ABBREV_OF = re.compile(
    r"\b(?:abbrev(?:iation)?|contr(?:action)?|short)\.?\s+(?:of|for)\s+([^\s;,.)]+)",
    re.I)


def _is_abbreviation_link(db, a: str, b: str) -> bool:
    """True when one of *a*, *b* is recorded as an abbreviation of the other.

    `'riew` is glossed "1. (abbrev. of briew); 2. (abbrev. of shriew) an
    arum" — a clipped form with its own entry, not a variant spelling of
    `briew`. Offering it to someone who wrote `briew` says their spelling
    has an alternative when it does not.

    The gloss must NAME the other word, so an entry that merely mentions an
    abbreviation of some third word is unaffected.
    """
    for x, y in ((a, b), (b, a)):
        for row in (db.lookup(x) or []):
            for m in _ABBREV_OF.finditer(str(row.get("english_gloss") or "")):
                if m.group(1).strip().lower().strip("'\u2019") == y.lower().strip("'\u2019"):
                    return True
    return False


def _is_affixed_form(db, word: str) -> bool:
    """True when the lexicon records *word* as built with a SUFFIX.

    `phin` is `phi` + the future suffix `-n` — `morphology.type: "suffixed"`,
    canonical `phi+-n`, glossed "2sg/pl will". `phiñ` is a separate dictionary
    entry glossed "You... not". They are different grammatical forms, not two
    spellings of one word, so offering `phiñ` to someone who wrote `phin`
    changes what the sentence means.

    SUFFIXED only, and the distinction is the point. On a PREFIXED form the
    diacritic belongs to the prefix — `ia-` is the same morpheme as `ïa-`,
    which the lexicon now spells with the diaeresis — so `iathuh` -> `ïathuh`
    corrects the prefix and must keep working. A first version of this test
    included "prefixed" and broke exactly that. On a SUFFIX the substitution
    swaps one morpheme for another: `-n` (future) becomes `-ñ`.

    Twelve entries in the lexicon are `suffixed`, and one plain/diacritic
    pair has a suffixed side — `phin` — so this stays narrow.
    """
    rows = db.lookup(word) or []
    if not rows:
        return False
    rich = (getattr(db, "_enriched_by_id", None) or {}).get(
        rows[0].get("entry_id")) or {}
    return (rich.get("morphology") or {}).get("type") == "suffixed"


def _is_elided_short_form(full: str, candidate: str) -> bool:
    """True when *candidate* is the apostrophe-elided short form of *full*.

    Khasi writes an elided initial consonant with an apostrophe: `syiem`
    "king" is also written `s'iem`, `kyieng` as `k'ing`, `khyndew` as `'dew`.
    Those are short forms of the word, not other ways of spelling it, and the
    maintainer has ruled that a short form is never offered as an alternative
    — the same ruling that removed the "abbrev. of" links.

    The test is deliberately one-directional. Someone who types the SHORT
    form is still shown the full one, which expands rather than abbreviates;
    only the reverse is suppressed.

    A pair where BOTH forms carry an apostrophe is left alone: neither is the
    short form of the other. `'ba-hab` and `'ba-pynthor` are such a pair.
    """
    apos = ("'", "\u2019")
    has_full = any(c in full for c in apos)
    has_cand = any(c in candidate for c in apos)
    return has_cand and not has_full


def find(word: str, analyser: Any, max_subs: int = MAX_SUBSTITUTIONS) -> list[SpellingVariant]:
    """
    Alternative diacritic spellings of *word* that the lexicon attests.

    Returns an empty list when the word already carries its diacritics, or
    when no attested alternative exists. Never reports the input itself.
    """
    w = unicodedata.normalize("NFC", word).lower()
    if not w:
        return []
    db = analyser.db
    found: dict[str, SpellingVariant] = {}

    # --- route 1: the whole word ------------------------------------
    # The folded index holds only entries that actually carry diacritics,
    # so anything it returns is a genuine alternative spelling.
    # An affixed input is skipped here too. The folded index matches on the
    # diacritic-stripped key, so `phin` finds `phiñ` by this route as well as
    # by generation — guarding only the generator left the bug in place.
    affixed = _is_affixed_form(db, w)
    for entry in ([] if affixed else db._folded_index.get(db._fold_key(w), [])):
        surface = (entry.get("surface_form") or "").lower()
        # The engine's fold key also strips apostrophes, so the folded index
        # returns elision variants alongside diacritic ones — 'ka' would
        # offer "k'a", 'im' would offer "i'm". Those are a different kind of
        # variation and not what this module is for, so require that the
        # candidate reduce to exactly the input once diacritics are removed.
        if surface and surface != w and surface.translate(_TO_PLAIN) == w:
            found[surface] = SpellingVariant(surface, "lexicon")

    # Generated candidates that happen to be lexicon entries in their own
    # right. Catches ñ, which the engine's fold map does not cover, so
    # `ain` -> `aiñ` is invisible to route 1.
    # Skipped entirely when the input is an affixed form: its spelling comes
    # from its morphology, so a diacritic substitution lands on a different
    # word. See _is_affixed_form — `phin` (phi + -n, "you will") was being
    # offered `phiñ` ("you... not").
    if not affixed:
        for cand in _generate(w, max_subs):
            if cand != w and cand not in found and cand in db._surface_index:
                found[cand] = SpellingVariant(cand, "lexicon")

    # --- route 1b: WITHDRAWN — the diacritic-free spelling ------------
    # This used to offer the plain spelling to a writer who had typed the
    # diacritic one, on the grounds that the lexicon records 107 words both
    # ways (khwain/khwaiñ, luin/luiñ) so both were "legitimate".
    #
    # They are not equal. In formal Khasi the diacritic spelling is the
    # correct one: `jingïalehkai`, `jaiñsem`. `ï` written as `i` and `ñ`
    # written as `n` are substitutions people make when a keyboard or a
    # font gets in the way, and the doubled lexicon entries are that habit
    # recorded, not evidence of free variation. Offering `jingialehkai` to
    # someone who typed `jingïalehkai` invited them to undo a correct
    # spelling — and "Accept all" once applied exactly that across a whole
    # document.
    #
    # The direction that helps is kept, and it is route 1 above: a writer
    # who types the plain form is still shown the diacritic one. Only the
    # downgrade is gone. The acute vowels come out with it — `á` marks
    # phonemic length, so stripping it loses a contrast rather than a
    # typographic nicety.
    #
    # A diacritic form that is genuinely misspelled is unaffected: it fails
    # the lexicon and is flagged through the normal correction path, which
    # can still suggest the plain form on edit distance if that is what was
    # meant.

    # --- route 2: the root of a derived form -------------------------
    # `jingïathuh` is not a lexicon entry — it is jing- + ïathuh. The root's
    # variants are looked up and substituted back into the surface.
    #
    # Prefix stripping, not analyser.analyse(). A full parse per token cost
    # ~2 s on unparseable words and 15 s on a single sentence; stripping a
    # known prefix and probing the folded index is a handful of hash lookups
    # and finds the same roots for the derived forms that matter here.
    for root in _candidate_roots(w, db):
        root_variants = {
            (e.get("surface_form") or "").lower()
            for e in db._folded_index.get(db._fold_key(root), [])
        }
        root_variants |= {
            c for c in _generate(root, max_subs) if c in db._surface_index
        }
        for rv in root_variants:
            if not rv or rv == root or rv.translate(_TO_PLAIN) != root:
                continue
            # Replace the last occurrence, so a prefix that happens to
            # contain the root's letters is left alone.
            idx = w.rfind(root)
            cand = w[:idx] + rv + w[idx + len(root):]
            if cand != w and cand not in found:
                found[cand] = SpellingVariant(cand, "root", root=rv)

    # --- route 4: the reciprocal prefix, by rule ---------------------
    for cand in _reciprocal_forms(w, db):
        if cand != w and cand not in found:
            found[cand] = SpellingVariant(cand, "rule")

    # --- route 5: the -ain / -aiñ ending, by rule --------------------
    ain = _ain_form(w, db) or _ain_compound(w, db) or _ain_elements(w, db)
    if ain and ain not in found:
        found[ain] = SpellingVariant(ain, "rule")

    # --- route 6: a short spelling of an ï-form ----------------------
    short = _short_i_form(w, db)
    if short and short not in found:
        found[short] = SpellingVariant(short, "rule")

    # --- route 7: a spelling the gloss itself records ----------------
    # `spelling_links.json` was built from three different gloss phrases at
    # once — "same as", "also spelt as" and "abbrev. of" — and only the
    # first two are spelling variants. Roughly half the 206 link pairs are
    # abbreviations, and those are dropped: a clipped form with its own
    # entry is a different word, not another way of writing this one.
    for cand in _spelling_links().get(w, ()):
        if cand != w and cand not in found:
            if _is_abbreviation_link(db, w, cand):
                continue
            # …and the same for an elided short form. `syiem` was offered
            # `s'iem`, whose gloss says "same as syiem" rather than
            # "abbrev. of", so the abbreviation test above let it through.
            # 110 of the 206 link pairs are this shape.
            if _is_elided_short_form(w, cand):
                continue
            found[cand] = SpellingVariant(cand, "gloss")

    # --- route 8: a corpus word the pool admitted with its diacritics --
    # `iatreilang` is how the corpus writes it (422 times); the pool admits
    # it as `ïatreilang`, the only spelling the lexicon uses for ïa- words.
    # The reduced spelling is accepted, so — exactly as for a lexicon word
    # typed without its ï — the diacritic spelling is offered alongside
    # rather than as a correction.
    for cand in _corpus_fold_map(db).get(w, ()):
        if cand != w and cand not in found:
            found[cand] = SpellingVariant(cand, "corpus")

    # A variant is canonical when the lexicon records that exact surface.
    for v in found.values():
        v.canonical = v.variant in db._surface_index
        for m in (db.lookup(v.variant) or []):
            g = (m.get("english_gloss") or "").strip()
            if g:
                v.gloss = g
                break

    # Canonical spellings first, then whole-word evidence over reconstruction.
    order = {"lexicon": 0, "gloss": 1, "root": 2, "folded": 3, "corpus": 4}
    return sorted(found.values(),
                  key=lambda v: (not v.canonical, order.get(v.source, 9), v.variant))

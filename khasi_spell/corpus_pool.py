"""
corpus_pool.py — admit frequent corpus types into the CANDIDATE pool.

Why
---
`pyntreikam` (`pyn-` + `trei` + `kam`) occurs 1,940 times in the corpus and
is not a lexicon entry. The symmetric-delete index is built from
`all_surface_forms()`, so the word is not in the list to be ranked, and
`suggest("pyntreika")` — one deletion away — cannot return it. No cost
model, context model or tie-break retrieves a string that was never a
candidate.

This is the same asymmetry `generate.py` addresses from the morphological
side, arriving from the corpus side instead: the checker's own measurement
put 64% of a 23,254-word target vocabulary outside the candidate pool.

Acceptance
----------
Supply alone left a contradiction: the pool offered words the checker then
rejected. `jyla` was answered with `jylla`, and `jylla` (24,252 corpus
occurrences, not a headword) was flagged the moment the writer accepted it.
The vote's frequency signal reads the lexicon's frequency table, so no
corpus word could ever earn it.

`apply(accept=True)`, the default, closes that: every admitted word, and
the reduced spelling the corpus writes it in, is registered with
`db.register_corpus_acceptance()` and earns the frequency signal in the
vote. `is_known()` is still untouched — it answers "is this in the
lexicon", and screening the candidate pool by it previously cost about 4
points of top-1. Measured on 200 clean corpus sentences, see the README.

Run-together spellings are refused outright. `jongki` is `jong ki` written
solid; `khasi_spell.splits` corrects it to the spaced form, so admitting it
would both offer it as a correction and accept it, undoing the split.

`corpus_freq.py` deliberately does not extend the vocabulary, for good
reason — the corpus holds English, proper nouns and misspellings, and
admitting them as valid Khasi would defeat the point. Nothing here
contradicts that: these forms become *offerable*, never *known*, and the
admission rules below exist to keep the junk out of even that role.

Admission
---------
A corpus type is admitted only if it clears all of:

  * a frequency floor (see ``DEFAULT_FLOOR``)
  * every character in the Khasi alphabet, and none of the letters Khasi
    does not use (``f v z c x q``) — those mark English and scan damage
  * the phonotactic screen
  * absent from ``quarantine``
  * its diacritic-folded form is not already a lexicon entry. The corpus
    holds zero `ï` and zero `ñ`, so every diacritic headword has an
    undiacriticked twin in it; admitting those would give each headword a
    rival plain spelling at distance 0 and quietly undo the ï/ñ work
  * no corpus type within one edit is ``DOMINANCE_RATIO`` times commoner.
    The corpus contains misspellings. A type seen 4 times one edit from a
    type seen 20,000 times is an error in the wild, not a word, and
    admitting it would have the checker suggest typos.
  * it is not a run-together spelling that ``khasi_spell.splits`` corrects
    to two words (`jongki` -> `jong ki`).
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

DEFAULT_PATH = Path(__file__).parent.parent / "data" / "corpus_freq.json"

# Occurrences required before a corpus type may be offered as a correction.
#
# Every one of the eight genuine cases the diagnostic sweep found clears a
# floor of 100, so recall does not set this number — precision does. Measured
# peak RSS (engine only, before the n-gram model): 471 MB with the pool off,
# 474 MB at floor 100, 481 MB at 20, 488 MB at 10, against a 512 MB
# deployment ceiling. At floor 20 the pool is 2,488 forms and 64,260 extra
# index keys against the lexicon's 406,037.
DEFAULT_FLOOR = 20

# How much commoner a one-edit neighbour must be before this type is read as
# a misspelling of it rather than a word in its own right.
#
# NOTE THE DIRECTION: a LOWER number is a STRICTER filter. The neighbour has
# to be `ratio` times commoner to win, so raising this makes domination
# harder to prove and lets MORE types in — measured, floor 20: ratio 5 admits
# 2,358 forms, ratio 10 admits 2,488, ratio 100 admits 2,782.
#
# Calibrated on a real failure. `atreilang` (53) is the corpus's misspelling
# of `iatreilang` (422) with the initial vowel dropped, a ratio of 7.96 — so
# at 10 it was admitted, the checker offered it as a correction for
# `iatreilang`, and `iatreilang` was offered back for it: the user was cycled
# between two spellings, one of which is not a word. Every value at or below
# 6 excludes it and all eight of the genuine cases survive down to 2, so this
# sits at 5 with margin on both sides.
DOMINANCE_RATIO = 5

# Letters Khasi does not use. Their presence marks English or scan damage.
BANNED_LETTERS = frozenset("fvzcxq")

ALPHABET = frozenset("abcdefghijklmnopqrstuvwxyzïñ'-")

# Characters an edit may introduce when looking for a dominant neighbour.
_EDIT_ALPHABET = "abcdefghijklmnopqrstuvwxyz"

_FOLD = str.maketrans({"ï": "i", "ñ": "n"})


def _fold(word: str) -> str:
    return word.translate(_FOLD)


# Prefixes the reciprocal `ïa-` attaches behind. Counted in the lexicon:
# jingïa- 86 / jingia- 0, pynïa- 20 / pynia- 0, nongïa- 17 / nongia- 0.
_PREFIXES_BEFORE_IA = ("jing", "pyn", "nong")


def canonicalise(word: str) -> str:
    """Restore the diacritics the corpus cannot type.

    The corpus holds ZERO `ï` and ZERO `ñ` across 6.67M tokens, so every
    corpus type is written in a reduced orthography. Admitting one verbatim
    put the wrong spelling into the candidate pool: `ïiatreilang` was
    answered with `iatreilang`, and `ïatreilang` — correct — was offered
    `iatreilang` as its correction.

    The lexicon's convention is not a preference, it is absolute. Counted
    over single-word headwords:

        word-initial   ïa-   332     ia-     0
        after jing-    jingïa 86     jingia  0
        after pyn-     pynïa  20     pynia   0
        after nong-    nongïa 17     nongia  0
        word-final     -aiñ  153     -ain    0

    With no counter-examples in 608 cases, restoring is a rewrite to the
    only spelling the lexicon uses, not a guess. `ia` is the reciprocal
    morpheme and is written `ïa` wherever it appears as one.
    """
    w = word.lower()
    if w.startswith("ia"):
        w = "ï" + w[1:]
    else:
        for p in _PREFIXES_BEFORE_IA:
            if w.startswith(p + "ia"):
                w = p + "ï" + w[len(p) + 1:]
                break
    if w.endswith("ain"):
        w = w[:-3] + "aiñ"
    return w


def _ain_confirmed(word: str, db: Any) -> bool:
    """Does the lexicon back `word` -> `word[:-3] + 'aiñ'`?

    True when the word, or its final element, is a headword the lexicon
    records with the tilde (`thawain` -> `thawaiñ`, `saitjain` ->
    `saitjaiñ`). False for names and loans (`hussain`, `risain`).
    """
    try:
        from khasi_spell import variants as _v
        return bool(_v._ain_form(word, db) or _v._ain_compound(word, db)
                    or _v._ain_elements(word, db))
    except Exception:
        return False


def _load_counts(path: Optional[Path] = None) -> dict:
    p = Path(path) if path else DEFAULT_PATH
    if not p.exists():
        raise FileNotFoundError(
            f"no corpus frequency list at {p}. Build one:  "
            f"python3 scripts/build_corpus_freq.py <corpus-dir> -o {p}"
        )
    data = json.loads(p.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "counts" in data:
        data = data["counts"]
    if not isinstance(data, dict):
        raise ValueError(f"unrecognised corpus_freq.json shape: {type(data)}")
    return data


def _quarantined(db: Any) -> set:
    """Surface forms withdrawn from the lexicon. Never re-admit one.

    Read from `_extra_blocks`, where KhasiDB already keeps every top-level
    block it does not manage, quarantine included. Re-reading the file
    instead costs a second parse of the whole lexicon: measured at peak RSS
    850 MB against 481 MB, which would breach the 512 MB deployment ceiling
    on its own. KhasiDB's own load path carries a comment about exactly that
    failure — three transient 300 MB peaks were the original free-tier OOM.

    Empty when the block is absent; `select()` reports the count it used, so
    a caller can tell "nothing quarantined" from "criterion not applied".
    """
    blocks = getattr(db, "_extra_blocks", None) or {}
    entries = blocks.get("quarantine") or []
    if isinstance(entries, dict):
        entries = list(entries.values())
    out: set = set()
    for e in entries:
        if isinstance(e, str):
            out.add(e.lower())
            continue
        if not isinstance(e, dict):
            continue
        form = e.get("form") or {}
        s = (form.get("normalized") or form.get("surface")
             or e.get("surface_form") or "")
        if s:
            out.add(str(s).lower())
    return out


def _dominant_neighbour(word: str, counts: dict, ratio: int) -> Optional[str]:
    """A one-edit corpus type at least *ratio* times commoner, or None."""
    bar = counts.get(word, 0) * ratio
    best = None
    best_c = bar - 1
    for i in range(len(word)):
        cand = word[:i] + word[i + 1:]                      # deletion
        c = counts.get(cand, 0)
        if c > best_c:
            best, best_c = cand, c
    for i in range(len(word) - 1):
        if word[i] == word[i + 1]:
            continue
        cand = word[:i] + word[i + 1] + word[i] + word[i + 2:]   # transposition
        c = counts.get(cand, 0)
        if c > best_c:
            best, best_c = cand, c
    for i in range(len(word)):
        for ch in _EDIT_ALPHABET:
            if ch == word[i]:
                continue
            cand = word[:i] + ch + word[i + 1:]             # substitution
            c = counts.get(cand, 0)
            if c > best_c:
                best, best_c = cand, c
    for i in range(len(word) + 1):
        for ch in _EDIT_ALPHABET:
            cand = word[:i] + ch + word[i:]                 # insertion
            c = counts.get(cand, 0)
            if c > best_c:
                best, best_c = cand, c
    return best


def select(
    db: Any,
    counts: Optional[dict] = None,
    floor: int = DEFAULT_FLOOR,
    ratio: int = DOMINANCE_RATIO,
    runtogether: Optional[set] = None,
) -> tuple[set, dict]:
    """Return (admitted forms, rejection tally). Pure; changes nothing.

    *runtogether* is the set of solid spellings the split table corrects to
    two words; None loads it from ``khasi_spell.splits``.
    """
    from khasi_engine.spell_checker import _phonotactically_ok

    counts = counts if counts is not None else _load_counts()
    if runtogether is None:
        try:
            from khasi_spell import splits
            runtogether = set(splits.load())
        except Exception:
            runtogether = set()
    known = set(db.all_surface_forms())
    folded_lex = {_fold(w) for w in known}
    quarantine = _quarantined(db)

    admitted: set = set()
    tally = {"quarantine_entries": len(quarantine), "below_floor": 0, "in_lexicon": 0, "non_khasi_char": 0,
             "banned_letter": 0, "quarantined": 0, "folds_onto_headword": 0,
             "phonotactic": 0, "dominated": 0, "runtogether": 0,
             "ain_kept_plain": 0,
             "diacritic_restored": 0, "restored_onto_headword": 0,
             "admitted": 0}

    for word, count in counts.items():
        if count < floor:
            tally["below_floor"] += 1
            continue
        w = word.lower()
        # `all_surface_forms()` is the candidate pool; `is_known()` is the
        # wider "is this a Khasi word" question, and they differ — `japan`
        # reaches is_known() through `_compound_tokens` while being filtered
        # out of the surface list as English-in-phrases. Admitting such a
        # word adds nothing (it is already offerable) and would mislabel its
        # provenance as corpus-only.
        if w in known or db.is_known(w):
            tally["in_lexicon"] += 1
            continue
        if not set(w) <= ALPHABET:
            tally["non_khasi_char"] += 1
            continue
        if BANNED_LETTERS & set(w):
            tally["banned_letter"] += 1
            continue
        if w in quarantine:
            tally["quarantined"] += 1
            continue
        if _fold(w) in folded_lex:
            tally["folds_onto_headword"] += 1
            continue
        if not _phonotactically_ok(w):
            tally["phonotactic"] += 1
            continue
        # A solid spelling of two words. The split table offers `jong ki`
        # for `jongki`; admitting `jongki` would offer the error itself as
        # a correction and, with acceptance on, stop flagging it at all.
        if w in runtogether:
            tally["runtogether"] += 1
            continue
        if _dominant_neighbour(w, counts, ratio) is not None:
            tally["dominated"] += 1
            continue
        # Admit the spelling the lexicon uses, not the one the corpus can
        # type. Every filter above ran on the corpus form, because that is
        # what the counts are keyed on and what the neighbours are written
        # in; only the admitted string changes.
        canon = canonicalise(w)
        # `-ain` -> `-aiñ` holds for Khasi words, not for everything that
        # happens to end in -ain. Unguarded, the pool admitted the name
        # `hussaiñ` and the loans `risaiñ` (resign) and `pilaiñ` (plane) —
        # spellings no one writes. Restore the tilde only where the
        # lexicon's own -aiñ rule, the one variants.py uses, agrees.
        if canon.endswith("aiñ") and not w.endswith("aiñ") \
                and not _ain_confirmed(w, db):
            canon = canon[:-1] + "n"
            tally["ain_kept_plain"] += 1
        if canon != w:
            # The restored form may be a headword even where the reduced one
            # was not: `iatrei` is not in the surface list, `ïatrei` is.
            if canon in known or db.is_known(canon):
                tally["restored_onto_headword"] += 1
                continue
            tally["diacritic_restored"] += 1
        admitted.add(canon)

    tally["admitted"] = len(admitted)
    return admitted, tally


def apply(
    checker: Any,
    path: Optional[Path] = None,
    floor: int = DEFAULT_FLOOR,
    ratio: int = DOMINANCE_RATIO,
    accept: bool = True,
) -> dict:
    """Admit qualifying corpus types to *checker*'s candidate pool.

    Registers them with the database as attested-but-not-known, and inserts
    them into the already-built delete index — the index is constructed
    during KhasiSpellChecker.__init__, well before this runs, so it is
    extended in place rather than rebuilt.

    With *accept* (the default) they are also made acceptable: the vote
    credits them with corpus frequency, both in the lexicon's orthography
    (`ïatreilang`, the form offered) and in the reduced one the corpus
    writes (`iatreilang`, accepted but never offered).
    """
    db = checker._db
    counts = _load_counts(path)
    admitted, tally = select(db, counts, floor=floor, ratio=ratio)

    register = getattr(db, "register_corpus_forms", None)
    if register is None:                       # engine too old; do nothing
        return {"admitted": 0, "reason": "db has no register_corpus_forms"}
    register(admitted)

    indexed = 0
    for w in admitted:
        if checker.teach_word(w):
            indexed += 1

    # The accepted set is data and may be shared (KhasiDB is memoised per
    # data source); whether THIS checker consults it is a per-checker flag.
    accepted = 0
    accept_fn = getattr(db, "register_corpus_acceptance", None)
    checker._corpus_accept_on = bool(accept and accept_fn is not None)
    if checker._corpus_accept_on:
        accepted = accept_fn(admitted | {_fold(w) for w in admitted})

    return {
        "floor": floor,
        "dominance_ratio": ratio,
        "corpus_types": len(counts),
        "indexed": indexed,
        "accepted": accepted,
        **tally,
    }

"""
spell_checker.py — Morphology-Gated Spell Checker
Khasi NLP Engine

The spell checker treats the morphological engine as its primary validity
oracle. Before generating any edit-distance candidates it asks: "Does the
confidence vote accept this word?" If yes, the word is correct and spell
checking is skipped entirely.

Decision pipeline for a single word
────────────────────────────────────
Input word  (NFC; the typographic apostrophe read as ASCII)
    │
    ▼
[Taught] a word added with KhasiSpeller.add_word → accepted, "user_dictionary"
    │
    ▼
[Gate 0] Semantically-invalid standalone token (`ng`, `e`)
    │  HIT  → method="semantically_invalid_standalone", is_known=False,
    │         suggestions from a narrow distance-first search
    │  PASS ↓
    ▼
[Gate 1+2] Confidence vote — lexicon 3, affixed morphology 2 (1 when the
    │  root is not a word), corpus frequency 2, phonotactics 1; threshold 3
    │  CLEARS → is_known=True, method="confidence_gate", gate 2
    │           …unless an unrecorded hyphenated compound has a misspelled
    │           part → rejected, method="hyphenated_part", gate 3.
    │           A word ending in y never clears: no Khasi word ends in y.
    │           Nor does an English word the lexicon does not record
    │           (data/english_in_corpus.json): `hospital`, `state`.
    │  BELOW and phonotactically invalid → method="levenshtein_phonotactic",
    │           gate 1, suggestions from a widened search (max_dist=3) plus
    │           banned-letter repair, and the validator's `reason`.
    │           Ranked inside _levenshtein_suggestions on the checker's own
    │           frequency table (corpus counts when applied).
    │  BELOW ↓
    ▼
[Gate 3] Edit-distance candidates
    │  Pool: the symmetric-delete index over the lexicon, the checker's own
    │        index of corpus-pool and taught words, and the morphological
    │        generator (generate.py).
    │  Filter: _offerable() — a vowel, legal letters, no final y, not an
    │          English word, shares material with the input, and
    │          phonotactically valid or attested.
    │  Rank:  (distance, first-letter change, -(frequency + morphology bonus
    │          + shared-onset bonus + attested bonus))
    │  A run-together word gets its two-word split first ("runtogether_split").
    │  Optional: FastText semantic re-ranking — off by default, it
    │            measured WORSE than rule-only ranking
    ▼
Return ranked suggestions

The `method` field takes exactly one of: "empty", "user_dictionary",
"semantically_invalid_standalone", "levenshtein_phonotactic",
"confidence_gate", "hyphenated_part", "english_word", "too_long",
"levenshtein_primary", "levenshtein_speller_combined", "hybrid_fasttext",
"runtogether_split", "none".

Sentence checking
─────────────────
check_sentence() tokenises the input with khasi_engine.tokens and runs each
distinct word through the same decision. Words that pass are silently
skipped; only words the gate rejects surface as corrections, in the
capitalisation of the word they replace.

This ensures that productive morphological forms (pynbha, jingsniew,
kynjat, etc.) are never flagged as misspellings even if they are not
in the static lexicon.
"""

from __future__ import annotations

import inspect
import math
import os
import threading
from functools import lru_cache
from typing import Mapping, Optional, TYPE_CHECKING, Callable

from khasi_engine import tokens as _tokens

if TYPE_CHECKING:
    from khasi_engine.database import KhasiDB

# Characters that are never part of a Khasi word but DO appear inside one in
# scanned or scraped text — the caret in `i^p` for `ïap`, and its relatives.
# Defined with the rest of the tokenisation in khasi_engine.tokens.
_NOISE_CHARS = _tokens.NOISE_CHARS

# Word boundary pattern, shared with every other component (tokens.py).
#
# Noise characters bind letters together exactly as `'` and `-` do, so a
# damaged word is seen WHOLE. Splitting on them was worse than useless: `i^p`
# became the tokens `i` and `p`, and each fragment was then "corrected" on its
# own. The facade used to tokenise without them, and then corrected a
# fragment inside a word this pattern had already flagged — two overlapping
# edits to one word. Accented vowels and the typographic apostrophe are
# letters and joiners here too; without them a dictionary word like
# `sngewrém` was split and each piece "corrected".
_WORD_PATTERN = _tokens.WORD_PATTERN

# Score bonus for suggestions that parse morphologically
# Kept small so morphological validity boosts ranking but cannot override
# a phonemically close match (distance 1) with a distant but valid word.
_MORPHO_VALID_BONUS = 3

# Weight per shared word-initial phoneme when breaking ranking ties.
# Matches the weight _levenshtein_suggestions already uses internally.
_PREFIX_BONUS_WEIGHT = 3

# Bonus for a candidate the lexicon actually records, over one the generator
# built. Both are offerable, but they are not equal evidence: `jingïalehkai`
# is a word someone wrote down and the corpus contains 21 times, while
# `jingailehkai` is a construction that has never been written. Without this
# the two tie on distance and sort alphabetically.
#
# Sized to outweigh the frequency of a rare headword but NOT a whole edit of
# distance, so a closer generated form still wins — the generator exists
# precisely because the lexicon cannot reach every valid derivation.
_ATTESTED_BONUS = 25

# Symmetric-delete index. Set False to restore the full-vocabulary scan.
DELETE_INDEX_ENABLED = True

# Whether the vendored autocorrect Speller still contributes candidates.
#
# It generates every edit of the word and intersects with the expanded
# frequency table (302,695 forms when this was measured; about 194k now that
# the expansion covers single-token roots only) — Norvig generate-and-filter,
# the very thing the delete index replaces. Profiling put it at 95% of suggest()'s runtime once the index
# was in, so it was A/B'd on the 308-item frozen benchmark:
#
#     config              detect   top-1   top-5   ms/word
#     index + Speller      78.6%   60.7%   74.4%       909
#     index only           78.6%   61.7%   74.7%        22
#
# It was contributing noise, not recall: turning it off is 41x faster AND a
# point better on top-1. Off by default. The Speller itself is still built —
# nlp_data supplies the frequency dictionary and the "frequent" confidence
# signal — only its candidate generation is skipped.
SPELLER_CANDIDATES_ENABLED = False

# Distance penalty for a candidate that is not a lexicon headword.
#
# all_surface_forms() includes compound tokens — fragments harvested from
# inside multi-word entries — which have no entry of their own. 'and' is one
# such fragment (English, scraped from a multi-word gloss). Because distance
# dominates the sort absolutely, 'dand' suggested 'and' at distance 1.0 over
# 'dang' at 1.4, even though 'dang' is a real headword used 12,967 times.
#
# A half-edit penalty demotes fragments below a genuine headword. MEASURED
# AND DISABLED: on the 77-item probe it left top-1 unchanged (54.5%) and cost
# top-5 1.3 points, because it also demotes legitimate corrections that only
# appear inside multi-word entries. The character-distance blend below fixed
# the 'dand' case properly. Set to 0.5 to re-enable.
_NON_HEADWORD_PENALTY = 0.0

# Corpus-derived candidates carry their own penalty, for the reason the
# generator's does (0.25 prefix, 0.4 compound): a form the dictionary
# records is better evidence than a string people happen to write, so it
# must win a tie at equal distance. Deliberately below a full edit — a
# corpus form one edit nearer should still beat a headword two edits away.
_CORPUS_FORM_PENALTY = 0.3

# ï -> i, ñ -> n. Used to recognise a candidate that is simply the input
# with its diacritics removed, which is a de-normalisation rather than a
# correction. See _offerable().
_DIACRITIC_FOLD = str.maketrans({"ï": "i", "ñ": "n"})

# Blend the phoneme distance with a character-level Damerau distance, taking
# the smaller. See _damerau_chars.
#
# MEASURED on the 77-item probe: top-1 54.5% -> 59.7%, top-5 68.8% -> 75.3%.
# The largest single ranking gain in this project. Two reasons: digraph
# tokenisation inflates purely typographic slips ('dand' for 'dang' is one
# keystroke but 1.4 phonemes), and the phoneme recurrence never implemented
# transposition at all, so the character distance supplies Damerau's fourth
# operation. Set to False to restore phoneme-only distance.
_USE_CHAR_DISTANCE = True

# Extend candidate generation with forms a morphological generator can build
# from real lexicon roots. See khasi_engine/generate.py for why this exists:
# the vote accepts words the lexicon does not hold, but candidate generation
# only ever offers what the lexicon holds, so the checker can accept
# `pynïarap` and still be unable to propose it as the correction for
# `pynarap`. Measured on a 23,254-word target vocabulary, 50.4% of the
# unreachable targets were words the checker accepts.
#
# ON by default as of 6 September 2026, on the following measurements. A
# generator can only add candidates, so the risk it carries is precision, not
# recall, and precision is what was measured:
#
#   frozen benchmark      unchanged — 0 of 294 items altered. Structurally
#                         guaranteed: its golds are in the candidate pool
#                         100% of the time, and this only emits forms that
#                         are not, so the benchmark cannot see the feature.
#   running text          measured twice, and the larger sample is the one
#                         that counts. On 200 sentences: 100 flags both ways.
#                         On 1,000 sentences (16,274 tokens): 522 -> 530,
#                         3.21% -> 3.26%. So the cost is +8 flags, not zero.
#                         The 8 are jingdonlang, jingiuhroit, jingshahkurup,
#                         jingshahthang, kamawkriah, nongthohkhubor,
#                         pyniengnoh, issue — real Khasi derived forms the
#                         vote was ALREADY rejecting, silently, with an empty
#                         suggestion list. The generator does not create the
#                         wrong rejection, it makes an existing one visible.
#                         That is still a cost to a reader, so it is counted.
#   peak RSS              472 MB both ways
#   latency               32.5 -> 40.3 ms median per sentence (+24%)
#   open vocabulary       on 3,000 pairs whose gold is often absent from the
#                         lexicon: top-1 0.269 -> 0.288, and on the subset
#                         whose gold is unreachable, 0.000 -> 0.031.
#                         62 fixes against 3 breaks, sign test p = 2.5e-15
#
# The gain is measured on synthetic errors only. Disable with
# KHASI_SPELL_MORPH_GEN=0.
MORPH_GENERATOR_ENABLED = os.environ.get(
    "KHASI_SPELL_MORPH_GEN", "1").lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# Morphological validity oracle
# ---------------------------------------------------------------------------

def _is_morphologically_valid(word: str, morph_check: Callable[[str], bool]) -> bool:
    """
    Ask the morphological engine whether *word* is a valid Khasi form.
    Uses the callable injected at construction time so spell_checker.py
    stays decoupled from the phase modules.
    """
    return morph_check(word)


# ---------------------------------------------------------------------------
# Levenshtein fallback  (no external dependencies)
# ---------------------------------------------------------------------------

# Khasi digraphs that count as a single phoneme for edit-distance purposes.
#
# Read from the lexicon's own `phonology.digraphs_as_single`, the list the
# validator and the Learn page use. This tuple used to be hard-coded with
# `dz` (which the data does not list) and without `jh` (which it does), so
# the distance function and the phonology disagreed about what one sound is.
# The literal below is only the fallback for a missing phonology block.
_DIGRAPHS_FALLBACK = ("ng", "sh", "ph", "th", "kh", "bh", "dh", "jh", "lh", "rh")
_DIGRAPH_STATE: dict = {"source": None, "digraphs": _DIGRAPHS_FALLBACK}


def _digraphs() -> tuple:
    """The current digraph inventory, longest first. Follows the data."""
    try:
        from khasi_engine import phonology as _p
        src = _p.DIGRAPHS_AS_SINGLE
    except Exception:                                   # pragma: no cover
        src = None
    if src is not _DIGRAPH_STATE["source"]:
        seq = tuple(sorted(src, key=len, reverse=True)) if src else _DIGRAPHS_FALLBACK
        _DIGRAPH_STATE["source"] = src
        _DIGRAPH_STATE["digraphs"] = seq
        _MULTI_STATE["digraphs"] = None                 # rules depend on units
    return _DIGRAPH_STATE["digraphs"]


# Kept for callers that read the old name; always the data-driven inventory.
_DIGRAPHS = _DIGRAPHS_FALLBACK


def _to_phonemes(word: str) -> list[str]:
    """
    Tokenise a Khasi word into its phoneme sequence.

    Digraphs (sh, ph, kh, ng, …) are treated as single units so that
    edit-distance reflects phonemic similarity rather than raw character
    distance.  Without this, "shnong" (phonemes: sh-n-o-ng) and "shnng"
    (phonemes: sh-n-ng) would appear to differ by 2 characters ('o' inserted,
    'n' deleted) but they actually differ by just 1 phoneme (missing vowel).

    Examples
    --------
    "shnong" → ["sh", "n", "o", "ng"]
    "shnng"  → ["sh", "n", "ng"]
    "khraw"  → ["kh", "r", "a", "w"]
    "lang"   → ["l", "a", "ng"]
    """
    result: list[str] = []
    i = 0
    w = word.lower()
    digraphs = _digraphs()
    while i < len(w):
        matched = False
        for dg in digraphs:
            if w[i:i+len(dg)] == dg:
                result.append(dg)
                i += len(dg)
                matched = True
                break
        if not matched:
            result.append(w[i])
            i += 1
    return result


def _shared_phoneme_prefix(a: str, b: str) -> int:
    """
    Count how many leading phonemes two Khasi words share.

    Used as a tie-breaker in suggestion ranking: when two candidates are
    at the same edit distance, prefer the one that shares more of the
    word-initial phoneme sequence with the input (e.g. "shnong" shares
    sh-n with "shnng", while "shong" only shares sh).
    """
    pa, pb = _to_phonemes(a), _to_phonemes(b)
    count = 0
    for x, y in zip(pa, pb):
        if x == y:
            count += 1
        else:
            break
    return count


# ── Khasi phoneme confusion sets ──────────────────────────────────────────
# Pairs of phonemes that are commonly confused in writing.  Substituting
# one for the other costs less than a full edit (1.0), making the
# corrected word rank higher than a phonetically unrelated candidate at
# the same integer distance.
#
# Every pair here is between SINGLE phonemes, the only kind a substitution
# can see. Pairs that span two phonemes on one side — the vowel nuclei ie/i,
# ie/ei, ie/ia and the ng/nk, ng/gn nasals — are in
# _MULTI_UNIT_CONFUSION_COST below and are opt-in.
#
# Cost 0.3 → dropped mark (accent, diaeresis, tilde)
# Cost 0.4 → very close (ng written g or n)
# Cost 0.7-0.8 → related (voiced/voiceless, nearby nasals)
_CONFUSION_COST: dict[tuple[str, str], float] = {
    # ng ↔ written as g / n  (nasal+velar stop): dieg → dieng, dien → dieng
    ("ng", "g"):  0.4,
    ("g",  "ng"): 0.4,
    ("ng", "n"):  0.4,
    ("n",  "ng"): 0.4,

    # Vowel length / accent confusion
    ("a",  "á"):  0.3,
    ("á",  "a"):  0.3,
    ("e",  "é"):  0.3,
    ("é",  "e"):  0.3,
    ("i",  "í"):  0.3,
    ("í",  "i"):  0.3,
    ("o",  "ó"):  0.3,
    ("ó",  "o"):  0.3,
    ("u",  "ú"):  0.3,
    ("ú",  "u"):  0.3,

    # Voiced ↔ voiceless stop (initial consonant drift): tieng → dieng
    ("d",  "t"):  0.8,
    ("t",  "d"):  0.8,
    ("b",  "p"):  0.8,
    ("p",  "b"):  0.8,
    ("k",  "g"):  0.7,
    ("g",  "k"):  0.7,

    # Nasal variation at word boundary (ŋ ↔ n ↔ ng)
    ("n",  "m"):  0.7,
    ("m",  "n"):  0.7,

    # ── Dropped diacritics ────────────────────────────────────────────────
    # Omitting ï and ñ is the commonest Khasi orthographic slip: the writer
    # knows the word, the keyboard has no key for the mark. The 2026-08-26
    # and 2026-09-08 lexicon passes settled `ïa` and `aiñ` as the correct
    # forms, so `iaid`/`ïaid` and `sain`/`saiñ` are one word written two ways.
    #
    # Cost 0.3, matching the accent pairs above rather than the 0.4 confusion
    # band — and NOT lower, because the lexicon's own phonology says both
    # marks are phonemic, not decorative:
    #
    #   phonology.consonant_chart  ñ is /ɲ/, palatal, distinct from n
    #                              (alveolar) and ng (velar)
    #   phonology.minimal_pairs    nasals_final: tʰaɲ "weave" / tʰaŋ "burn"
    #   phonology.vowels_short     [a e i o u ï] — ï is a vowel in its own
    #                              right, and vowel_length_note records it as
    #                              the central high /ɨ/ with no long counterpart
    #
    # 0.3 is what this table already charges for a dropped accent (a/á), and
    # vowel length is phonemic too (vowel_length_note) — so the same error in
    # kind gets the same price. A word differing only by a mark still ranks
    # below an exact match, which is what the `ïing` "house" / `ing` "burn"
    # split needs.
    ("i",  "ï"):  0.3,
    ("ï",  "i"):  0.3,
    ("n",  "ñ"):  0.3,
    ("ñ",  "n"):  0.3,
    ("y",  "ý"):  0.3,
    ("ý",  "y"):  0.3,
}

# Confusions that span two phonemes on at least one side. `_to_phonemes`
# splits vowel sequences into their letters, so a plain substitution table
# never sees `ie` or `nk` as one unit: these 14 entries sat in the table
# above and never fired (`dien` -> `din` cost a full edit, not 0.4, whatever
# the comments said). They are now applied as multi-unit substitutions inside
# the same recurrence — when switched on. (`ia`/`ïa` was dropped outright:
# the single-phoneme `i`/`ï` pair already prices it identically.)
#
# OFF by default, on measurement (2026-09-30, all else equal):
#
#                          word benchmark        sentence benchmark
#                          top-1    top-5        top-1 (context)
#     off                  73.5%    91.5%        92.0%
#     on                   72.4%    91.5%        92.0%
#
# Three word items lost and nothing gained: the vowel-nucleus pairs pull in
# near-homographs that tie with the right answer and win the tie. Kept, and
# measurable, rather than deleted: set KHASI_SPELL_MULTI_UNIT_CONFUSIONS=1.
_MULTI_UNIT_CONFUSION_COST: dict[tuple[str, str], float] = {
    # ng written as nk / gn:  dienk → dieng, dign → ding
    ("ng", "nk"): 0.5,
    ("nk", "ng"): 0.5,
    ("ng", "gn"): 0.5,
    ("gn", "ng"): 0.5,
    # Vowel nucleus ambiguity: ie / i / e / ei / ia / io
    ("ie", "i"):  0.4,
    ("i",  "ie"): 0.4,
    ("ie", "e"):  0.4,
    ("e",  "ie"): 0.4,
    ("ie", "ei"): 0.5,
    ("ei", "ie"): 0.5,
    ("ie", "ia"): 0.6,
    ("ia", "ie"): 0.6,
    ("ie", "io"): 0.7,
    ("io", "ie"): 0.7,
}


# Rules built from _MULTI_UNIT_CONFUSION_COST, cached per digraph inventory.
_MULTI_STATE: dict = {"digraphs": None, "by_end": {}, "longest": 1}

# Off by default: see the measurement above _MULTI_UNIT_CONFUSION_COST.
_MULTI_UNIT_CONFUSIONS = os.environ.get(
    "KHASI_SPELL_MULTI_UNIT_CONFUSIONS", "0").lower() in ("1", "true", "yes", "on")


def _multi_rules() -> tuple[dict, int]:
    """Multi-phoneme confusion rules, indexed by their last units.

    Returns ({(last_a, last_b): [(A, B, cost), ...]}, longest side).
    Rebuilt when the digraph inventory changes, since that decides how a
    pair splits into units.
    """
    if not _MULTI_UNIT_CONFUSIONS:
        return {}, 1
    digraphs = _digraphs()
    if _MULTI_STATE["digraphs"] is not digraphs:
        by_end: dict = {}
        longest = 1
        for (a, b), cost in _MULTI_UNIT_CONFUSION_COST.items():
            A, B = tuple(_to_phonemes(a)), tuple(_to_phonemes(b))
            if len(A) == 1 and len(B) == 1:
                continue                           # handled by _sub_cost
            by_end.setdefault((A[-1], B[-1]), []).append((A, B, cost))
            longest = max(longest, len(A), len(B))
        _MULTI_STATE.update(digraphs=digraphs, by_end=by_end, longest=longest)
    return _MULTI_STATE["by_end"], _MULTI_STATE["longest"]


def _sub_cost(pa: str, pb: str) -> float:
    """
    Substitution cost between two phonemes.

    Returns a value in (0.0, 1.0]:
      0.0 — identical phonemes (free)
      <1.0 — confused pair (see _CONFUSION_COST)
      1.0 — unrelated phonemes (standard edit cost)
    """
    if pa == pb:
        return 0.0
    return _CONFUSION_COST.get((pa, pb), 1.0)


def _damerau_chars(a: str, b: str, cap: float = 4.0) -> float:
    """
    Character-level Damerau-Levenshtein distance, including transposition.

    Complements the phoneme distance rather than replacing it. Digraph
    tokenisation is right for phonemic similarity but can overstate a purely
    typographic slip: 'dand' for 'dang' is one keystroke, yet as phonemes
    [d,a,n,d] vs [d,a,ng] it costs a substitution plus a deletion (1.4).
    Transposition also lives here — the phoneme recurrence never had it.
    """
    la, lb = len(a), len(b)
    if abs(la - lb) > cap:
        return cap + 1
    prev2: list[float] = []
    prev = list(range(lb + 1))
    for i in range(1, la + 1):
        cur = [float(i)] + [0.0] * lb
        for j in range(1, lb + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            cur[j] = min(prev[j] + 1, cur[j-1] + 1, prev[j-1] + cost)
            if i > 1 and j > 1 and a[i-1] == b[j-2] and a[i-2] == b[j-1]:
                cur[j] = min(cur[j], prev2[j-2] + 1)      # transposition
        prev2, prev = prev, cur
    return prev[lb]


def _phoneme_table(pa: list, pb: list) -> list:
    """Full DP table of the phoneme distance, multi-unit rules included.

    One recurrence shared by `_levenshtein` and `explain_edit`, so the
    explanation a reader sees is the arithmetic the ranking used.
    """
    rules, _ = _multi_rules()
    la, lb = len(pa), len(pb)
    D = [[0.0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        D[i][0] = float(i)
    for j in range(lb + 1):
        D[0][j] = float(j)
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            best = min(D[i-1][j] + 1.0, D[i][j-1] + 1.0,
                       D[i-1][j-1] + _sub_cost(pa[i-1], pb[j-1]))
            for A, B, cost in rules.get((pa[i-1], pb[j-1]), ()):
                la_, lb_ = len(A), len(B)
                if (i >= la_ and j >= lb_
                        and tuple(pa[i-la_:i]) == A and tuple(pb[j-lb_:j]) == B):
                    best = min(best, D[i-la_][j-lb_] + cost)
            D[i][j] = best
    return D


def _phoneme_distance(a: str, b: str) -> float:
    pa, pb = _to_phonemes(a), _to_phonemes(b)
    return _phoneme_table(pa, pb)[len(pa)][len(pb)]


def _levenshtein(a: str, b: str) -> float:
    """
    Phoneme-aware, confusion-weighted Levenshtein distance.

    Tokenises both words into phoneme sequences (digraphs as single units).
    Substitution costs are reduced for phonetically related pairs via
    _CONFUSION_COST (and, when switched on, the two-phoneme pairs in
    _MULTI_UNIT_CONFUSION_COST), so commonly confused spellings produce lower
    distances and rank higher in suggestion lists.

    Returns the smaller of that and the character-level Damerau distance.
    """
    if a == b:
        return 0.0
    phonemic = _phoneme_distance(a, b)
    # Take whichever view sees the smaller error. The phoneme distance
    # keeps its advantage on genuine phonemic confusions (ng/g, ie/i); the
    # character distance rescues typographic slips and transpositions that
    # digraph tokenisation inflates.
    if _USE_CHAR_DISTANCE:
        return min(phonemic, _damerau_chars(a, b))
    return phonemic


def _char_ops(a: str, b: str) -> list[dict]:
    """Character-level Damerau operations turning *a* into *b*."""
    la, lb = len(a), len(b)
    D = [[0] * (lb + 1) for _ in range(la + 1)]
    for i in range(la + 1):
        D[i][0] = i
    for j in range(lb + 1):
        D[0][j] = j
    for i in range(1, la + 1):
        for j in range(1, lb + 1):
            cost = 0 if a[i-1] == b[j-1] else 1
            D[i][j] = min(D[i-1][j] + 1, D[i][j-1] + 1, D[i-1][j-1] + cost)
            if i > 1 and j > 1 and a[i-1] == b[j-2] and a[i-2] == b[j-1]:
                D[i][j] = min(D[i][j], D[i-2][j-2] + 1)
    ops, i, j = [], la, lb
    while i > 0 or j > 0:
        if (i > 1 and j > 1 and a[i-1] == b[j-2] and a[i-2] == b[j-1]
                and a[i-1] != a[i-2] and D[i][j] == D[i-2][j-2] + 1):
            ops.append({"op": "swap", "from": a[i-2:i], "to": b[j-2:j],
                        "cost": 1.0, "known": False})
            i, j = i - 2, j - 2
            continue
        if i > 0 and j > 0 and D[i][j] == D[i-1][j-1] + (0 if a[i-1] == b[j-1] else 1):
            if a[i-1] != b[j-1]:
                ops.append({"op": "substitute", "from": a[i-1], "to": b[j-1],
                            "cost": 1.0, "known": False})
            i, j = i - 1, j - 1
            continue
        if i > 0 and D[i][j] == D[i-1][j] + 1:
            ops.append({"op": "drop", "from": a[i-1], "to": "", "cost": 1.0,
                        "known": False})
            i -= 1
            continue
        ops.append({"op": "add", "from": "", "to": b[j-1], "cost": 1.0,
                    "known": False})
        j -= 1
    return list(reversed(ops))


def explain_edit(a: str, b: str) -> list[dict]:
    """
    Which operations relate *a* to *b*, and were any of them a confusion
    Khasi orthography actually makes?

    The ranking already knows this — `_CONFUSION_COST` is why `dand` proposes
    `dang` ahead of a same-distance alternative — but nothing surfaced it, so
    a reader saw five candidates at "88%, 77%, 67%" with no way to tell that
    the first differs by a nasal the language genuinely confuses while the
    others differ by an unrelated letter.

    The operations come from whichever view produced the distance the
    ranking used. `_levenshtein` takes the smaller of the phoneme distance
    and the character Damerau distance; this used to walk the phoneme
    recurrence only, so `dand` -> `dang` was explained as two operations
    costing 1.4 while it had been ranked at 1.0 as one character change.
    The costs of the returned operations now add up to `_levenshtein(a, b)`.
    `known` marks an operation the confusion matrix prices below a full edit.
    """
    a, b = (a or "").lower(), (b or "").lower()
    if a == b:
        return []
    pa, pb = _to_phonemes(a), _to_phonemes(b)
    D = _phoneme_table(pa, pb)
    if _USE_CHAR_DISTANCE and _damerau_chars(a, b) < D[len(pa)][len(pb)] - 1e-9:
        return _char_ops(a, b)
    rules, _ = _multi_rules()
    ops, i, j = [], len(pa), len(pb)
    while i > 0 or j > 0:
        if i > 0 and j > 0:
            matched = False
            for A, B, cost in rules.get((pa[i-1], pb[j-1]), ()):
                la_, lb_ = len(A), len(B)
                if (i >= la_ and j >= lb_ and tuple(pa[i-la_:i]) == A
                        and tuple(pb[j-lb_:j]) == B
                        and abs(D[i][j] - (D[i-la_][j-lb_] + cost)) < 1e-9):
                    ops.append({"op": "substitute", "from": "".join(A),
                                "to": "".join(B), "cost": round(cost, 2),
                                "known": True})
                    i, j = i - la_, j - lb_
                    matched = True
                    break
            if matched:
                continue
            cost = _sub_cost(pa[i-1], pb[j-1])
            if abs(D[i][j] - (D[i-1][j-1] + cost)) < 1e-9:
                if pa[i-1] != pb[j-1]:
                    ops.append({"op": "substitute", "from": pa[i-1], "to": pb[j-1],
                                "cost": round(cost, 2),
                                "known": (pa[i-1], pb[j-1]) in _CONFUSION_COST})
                i, j = i - 1, j - 1
                continue
        if i > 0 and abs(D[i][j] - (D[i-1][j] + 1)) < 1e-9:
            ops.append({"op": "drop", "from": pa[i-1], "to": "", "cost": 1.0,
                        "known": False})
            i -= 1
            continue
        ops.append({"op": "add", "from": "", "to": pb[j-1], "cost": 1.0,
                    "known": False})
        j -= 1
    return list(reversed(ops))


def _normalise_candidates(word: str) -> list[str]:
    """
    Generate a small set of normalised candidate forms from *word*.

    These candidates bypass edit-distance entirely — they are produced by
    applying known Khasi orthographic normalisation rules directly to the
    input.  If any candidate exists in the lexicon it will be caught by
    is_known() in the suggest() caller.  Here we return them so they can
    be merged into the Levenshtein candidate list.

    Rules applied
    -------------
    1. Doubled-consonant removal: "dinng" → "ding", "dienn" → "dien"
       (users sometimes double a nasal to indicate the ng sound)
    2. Terminal nasal normalisation: word ending in bare "n" → also try "ng"
       and vice versa  (ŋ ↔ n ↔ ng ambiguity)
    3. Vowel-nucleus expansion: terminal "n" after a vowel → try "ng" too
    4. "gn" → "ng" transposition at word end  (dign → ding)
    5. ie/i/e substitution at vowel nucleus  (dien → din, den)
    """
    w = word.lower()
    candidates: list[str] = []

    # Rule 1 — collapse doubled consonants
    # Replace any run of 2+ identical consonant characters with one
    import re as _re
    collapsed = _re.sub(r'([^aáeéiíoóuúyýï])\1+', r'\1', w)
    if collapsed != w:
        candidates.append(collapsed)

    # Rule 2 — terminal n ↔ ng  (ŋ normalisation)
    if w.endswith("ng") and not w.endswith("ang"):
        candidates.append(w[:-2] + "n")     # dieng → dien
    if w.endswith("n") and not w.endswith("ng"):
        candidates.append(w + "g")           # dien  → dieng (then → dieng)

    # Rule 3 — "gn" at word end → "ng"  (dign → ding)
    if w.endswith("gn"):
        candidates.append(w[:-2] + "ng")

    # Rule 4 — strip a lone trailing "g" that might be a half-typed "ng"
    if w.endswith("g") and not w.endswith("ng"):
        candidates.append(w[:-1] + "ng")    # dig → ding (if ding exists)

    # Rule 5 — vowel nucleus ie/i/e interchange  (limited: only word-middle)
    if "ie" in w:
        candidates.append(w.replace("ie", "i", 1))   # dien → din
        candidates.append(w.replace("ie", "e", 1))   # dien → den
    if "ei" in w:
        candidates.append(w.replace("ei", "ie", 1))  # deig → dieg
    # single 'e' between consonants → try 'ie'
    # (only if word doesn't already have 'ie' to avoid loops)
    if "ie" not in w:
        candidates.append(_re.sub(r'(?<=[^aáeéiíoóuúyýï])e(?=[^aáeéiíoóuúyýï])', 'ie', w))

    # Deduplicate and exclude the original word
    seen: set[str] = {w}
    return [c for c in candidates if c and c not in seen and len(c) >= 2]


_NAMED_ENTITIES: Optional[frozenset] = None


def _named_entities() -> frozenset:
    """
    Proper-noun tokens, read from data/gazetteer.json if it has been built.

    Loaded here rather than through `khasi_spell.foreign` so the engine
    keeps no dependency on the facade — this reads a data file sitting
    beside khasi_db.json, which the engine already owns.

    Empty when the file is absent, in which case the illegal-character rule
    applies with no exception at all.
    """
    global _NAMED_ENTITIES
    if _NAMED_ENTITIES is None:
        import json as _json
        from khasi_engine import paths as _paths

        path = _paths.data_file("gazetteer.json")
        try:
            _NAMED_ENTITIES = frozenset(
                _json.loads(path.read_text(encoding="utf-8")).get("tokens") or ())
        except Exception:
            _NAMED_ENTITIES = frozenset()
    return _NAMED_ENTITIES


_ENGLISH_WORDS: Optional[frozenset] = None


def _english_words() -> frozenset:
    """
    English words found in the corpus, read from data/english_in_corpus.json.

    Maintainer ruling 2026-10-01: English words in Khasi text are rejected,
    however often Khasi writers use them. Before this list the corpus pool
    admitted any frequent type spelled with Khasi letters, and the vote then
    accepted it on corpus frequency + phonotactics: `hospital`, `the`,
    `state`, `member` — 355 English types, 0.77% of corpus tokens. A word on
    the list is never accepted unless the lexicon records it, never offered,
    and never admitted to the pool. Built by scripts/build_english_list.py,
    which holds back names and the few Khasi words that share an English
    spelling (`longing`, "household").

    Empty when the file is absent, in which case nothing changes.
    """
    global _ENGLISH_WORDS
    if _ENGLISH_WORDS is None:
        import json as _json
        from khasi_engine import paths as _paths

        path = _paths.data_file("english_in_corpus.json")
        try:
            _ENGLISH_WORDS = frozenset(
                _json.loads(path.read_text(encoding="utf-8")).get("words") or ())
        except Exception:
            _ENGLISH_WORDS = frozenset()
    return _ENGLISH_WORDS


_CORPUS_COUNTS: Optional[dict] = None


def corpus_counts() -> dict:
    """Raw corpus counts (data/corpus_freq.json), in corpus spelling.

    Loaded once and shared: the analyser asks whether a fused compound has
    ever been written solid, and khasi_spell asks whether a capitalised word
    is a known name. Empty when the file is absent, and every caller then
    leaves its rule off.
    """
    global _CORPUS_COUNTS
    if _CORPUS_COUNTS is None:
        import json as _json
        from khasi_engine import paths as _paths
        try:
            _CORPUS_COUNTS = _json.loads(
                _paths.data_file("corpus_freq.json").read_text(encoding="utf-8")
            ).get("counts") or {}
        except Exception:
            _CORPUS_COUNTS = {}
    return _CORPUS_COUNTS


_NAME_ONLY_WORDS: Optional[frozenset] = None


def _name_only_words() -> frozenset:
    """
    English words held back from the English list as names or titles.

    scripts/build_english_list.py holds back a word whose corpus uses are at
    least 95% capitalised: `United` (1,746 of 1,766), `Hoping` (508 of 520),
    `Smit`, `Martin`, `Limited`. Held back, such a word was accepted in any
    case, so lower-case `killing`, `helping` and `traditional` passed as
    Khasi — 52 of them, against the 2026-10-01 ruling that English words are
    rejected. Maintainer decision 2026-10-04: accepted when written with a
    capital, rejected as English in lower case. Words held back as Khasi
    (`longing` "household") or as Khasi derivations (`jingle`) are not here.
    """
    global _NAME_ONLY_WORDS
    if _NAME_ONLY_WORDS is None:
        import json as _json
        from khasi_engine import paths as _paths

        path = _paths.data_file("english_in_corpus.json")
        try:
            held = _json.loads(path.read_text(encoding="utf-8")).get("held_back") or {}
            _NAME_ONLY_WORDS = frozenset(
                w for w, why in held.items() if str(why).startswith("name:"))
        except Exception:
            _NAME_ONLY_WORDS = frozenset()
    return _NAME_ONLY_WORDS


class _DeleteIndex:
    """
    Symmetric-delete index (Brants-style SymSpell) over the spelling vocabulary.

    _levenshtein_suggestions used to compute a weighted phoneme distance
    against every one of the ~16,000 surface forms, which cost ~2.5 s per
    unknown word — the single largest performance problem in the checker.

    The index maps every string obtained by deleting up to `max_dist`
    characters from a vocabulary word back to that word. A query is expanded
    the same way, and anything sharing a deletion is a candidate. That turns
    the search from a full scan into a handful of hash lookups; the existing
    weighted distance then ranks the small candidate set, so ranking
    behaviour is unchanged by construction.

    Measured on this vocabulary: 429,925 keys, +70 MB, 2.2 s to build,
    0.07 ms per lookup.

    Caveat, deliberately not hidden: the index is character-based while the
    distance is a *weighted* phoneme distance where a confusion pair can cost
    as little as 0.3. A candidate at weighted distance 2.0 can therefore be
    more than two character edits away, and such a candidate is invisible to
    a distance-2 index. `verify_recall()` measures what that costs; the
    fallback scan is kept for callers that need exactness.
    """

    __slots__ = ("max_dist", "_index", "_words")

    def __init__(self, words, max_dist: int = 2):
        self.max_dist = max_dist
        # A set, not the sequence as passed: nothing reads it in order, and
        # `add()` needs a membership test that is not a scan of 15k strings.
        self._words = set(words)
        index: dict[str, list[str]] = {}
        for w in words:
            for d in self._deletes(w, max_dist):
                index.setdefault(d, []).append(w)
        self._index = index

    @staticmethod
    def _deletes(word: str, max_dist: int) -> set[str]:
        out = {word}
        frontier = {word}
        for _ in range(max_dist):
            nxt = set()
            for w in frontier:
                for i in range(len(w)):
                    nxt.add(w[:i] + w[i + 1:])
            out |= nxt
            frontier = nxt
        return out

    def candidates(self, word: str) -> set[str]:
        """Vocabulary words plausibly within max_dist character edits."""
        hits: set[str] = set()
        for d in self._deletes(word, self.max_dist):
            hits.update(self._index.get(d, ()))
        hits.discard(word)
        return hits

    def add(self, word: str) -> bool:
        """Index one more word, as the constructor would have.

        The index is built once from the lexicon, so a word taught at
        runtime was invisible to every later search — accepted by the vote,
        never offered as a correction. Returns False when the word is
        already indexed.
        """
        w = (word or "").lower().strip()
        if not w or " " in w or w in self._words:
            return False
        self._words.add(w)
        for d in self._deletes(w, self.max_dist):
            self._index.setdefault(d, []).append(w)
        return True

    def __len__(self) -> int:
        return len(self._index)


# Letters Khasi does not use, and what a writer or a scanner most likely
# meant by them. Substitutions are phonetically motivated — /f/ is written
# `ph` in Khasi loans, /v/ falls together with `b` — and deletion is offered
# too, because a stray banned letter is often simply noise.
#
# `fi` -> `ñ` is not phonetics but this scanner: the tilde over an n is read
# as f followed by i. The lexicon shows it — `kfii` is `kñi`, `kfiia` is
# `kñia` — and both repairs land on real entries.
_BANNED_REPAIR = {
    "f": ("ph", "p", ""), "v": ("b", "w", ""), "z": ("s", "j", ""),
    "c": ("k", "s", ""),  "x": ("ks", "k", ""), "q": ("k", ""),
}
_BANNED_DIGRAPH = (("fii", "ñi"), ("fi", "ñ"))
# A word with several banned letters explodes combinatorially; the cap keeps
# the extra search bounded. Words this damaged rarely have a real answer.
_MAX_BANNED_REPAIRS = 12

# Charged against every candidate reached by repairing a banned letter, so a
# word found directly always wins at the same distance. Without it the
# deletion repair (`f` -> nothing) shortened the input and made every short
# word look close: `brief` lost `briew` to `bri`.
_BANNED_REPAIR_PENALTY = 0.6


# Charged against a candidate that fails phonology.validate(). Large enough
# to move a malformed word below a well-formed one at the same edit distance,
# small enough that it still appears when it is the only answer.
_INVALID_SHAPE_PENALTY = 0.75

@lru_cache(maxsize=65536)
def _phonotactically_ok(word: str) -> bool:
    """phonology.validate() with a cache — it is called per candidate.

    Bounded. The cache used to be a plain module dict that grew by about ten
    entries per unknown word checked and was never emptied, which in a
    long-running service under a 512 MB ceiling is a slow leak. Cleared when
    a checker is built, since a new lexicon may carry new phonology.
    """
    try:
        from khasi_engine import phonology as _p
        return bool((_p.validate(word) or {}).get("pass"))
    except Exception:
        return True


def _repair_banned(word: str) -> list[str]:
    """Plausible Khasi spellings of *word* with its banned letters repaired."""
    out: set = set()
    for bad, good in _BANNED_DIGRAPH:
        if bad in word:
            out.add(word.replace(bad, good))
    for i, ch in enumerate(word):
        for rep in _BANNED_REPAIR.get(ch, ()):
            out.add(word[:i] + rep + word[i + 1:])
    out.discard(word)
    return sorted(x for x in out if x)[:_MAX_BANNED_REPAIRS]


def _phonotactic_reason(word: str) -> Optional[str]:
    """The phonology module's own explanation for rejecting *word*."""
    try:
        from khasi_engine import phonology as _p
        errs = (_p.validate(word) or {}).get("errors") or []
        return "; ".join(str(e) for e in errs) or None
    except Exception:
        return None


def _first_letter_differs(word: str, cand: str) -> int:
    """
    1 when *cand* starts with a different letter from *word*, else 0.

    Used only to break ties between candidates at the same distance, never
    to exclude one: about one benchmark error in six changes the first
    letter (`ylngkha` for `lyngkha`), so a hard rule would make those
    uncorrectable. As a tiebreak it stops `khh` being offered `eh` and `oh`
    — one "sound" away only because the digraph kh counts as a single
    phoneme — ahead of `khoh`, `khah` and `khih`, which keep what was typed.
    The dots on ï and ñ are ignored, so restoring a diacritic costs nothing.
    """
    fold = lambda w: w[:1].lower().replace("ï", "i").replace("ñ", "n")
    return 0 if fold(word) == fold(cand) else 1


def _levenshtein_suggestions(
    word: str,
    db: "KhasiDB",
    max_dist: int = 2,
    top_n: int = 5,
    *,
    freq: Optional[Mapping[str, int]] = None,
    extra_index: Optional["_DeleteIndex"] = None,
    extra_forms: frozenset | set = frozenset(),
    penalised: frozenset | set = frozenset(),
) -> list[tuple[float, str]]:
    """
    Return up to *top_n* (distance, word) pairs within *max_dist* phonemic
    edits of *word*, sorted by (distance, first-letter, -score).

    Returning distances alongside words lets callers preserve phonemic
    proximity as the primary ranking key — morphological validity is only
    used as a secondary tie-breaker, never to override a closer match.

    The keyword arguments carry a checker's own state, which is not stored on
    the shared lexicon: *freq* its frequency table (corpus counts when
    applied), *extra_index* / *extra_forms* the corpus-pool and runtime-taught
    words it may offer, and *penalised* the subset charged
    `_CORPUS_FORM_PENALTY`.
    """
    word = word.lower()
    scored: list[tuple[float, int, float, str]] = []
    seen: set[str] = set()
    freq_data = freq if freq is not None else db.to_freq_dict()
    # Narrow the search space before measuring distance. Falls back to the
    # full scan only when no index is ATTACHED — an attached index that
    # returns nothing is taken at its word, and the search ends there.
    #
    # That is deliberate, not an oversight: rescanning on an empty pool was
    # measured at 4.3 s against 40 ms indexed, and at the max_dist this is
    # called with it recovers nothing, because a word with an empty pool has
    # no lexicon form within two character edits by construction. Zero of
    # the 294 frozen benchmark items reach it. A caller that needs exactness
    # detaches the index instead.
    index = getattr(db, "_delete_index", None)
    if index is not None:
        pool = set(index.candidates(word))
        if extra_index is not None:
            pool |= extra_index.candidates(word)
    else:
        pool = set(db.all_surface_forms()) | set(extra_forms)

    # First pass: include normalised candidates with a small distance boost
    # (they get distance 0.5 so they outrank Levenshtein candidates at dist 1)
    #
    # `seen` is marked only for a candidate actually taken. Marking every
    # normalisation candidate up front blacklisted it from the distance pass
    # below, so a candidate this pass generated but could not attest was
    # silently removed from the search rather than merely not boosted:
    # `ïatreilan` normalises to `ïatreilang` under the terminal n/ng rule,
    # and `ïatreillang` under doubled-consonant collapse, so both marked the
    # right answer seen, declined to score it, and returned nothing at all.
    surface_forms = db.surface_form_set()
    for nc in _normalise_candidates(word):
        if nc in seen or " " in nc:
            continue
        if nc in freq_data or nc in surface_forms or nc in extra_forms:
            seen.add(nc)
            f      = freq_data.get(nc, 0)
            prefix = _shared_phoneme_prefix(word, nc)
            score  = f + prefix * _PREFIX_BONUS_WEIGHT
            scored.append((0.5, _first_letter_differs(word, nc), -score, nc))

    # Second pass: measure distance over the candidate pool.
    for sf in pool:
        # Skip multi-word strings (safety net)
        if " " in sf or not sf:
            continue
        if sf in seen:
            continue
        seen.add(sf)
        d = _levenshtein(word, sf)
        # Fragments from multi-word entries are weaker candidates than
        # headwords; see _NON_HEADWORD_PENALTY.
        if _NON_HEADWORD_PENALTY and sf not in db._surface_index:
            d += _NON_HEADWORD_PENALTY
        if _CORPUS_FORM_PENALTY and sf in penalised:
            d += _CORPUS_FORM_PENALTY
        # A candidate the engine's own phonotactics rejects is ranked below
        # one it accepts. `sbngaifi` was answered with `pbngaiñ` at 81%
        # above `shngaiñ` at 71%, and `pb` is not a permitted Khasi onset —
        # the lexicon carries `pbngaiñ` only because the scan misread it.
        #
        # A PENALTY, not a filter, and the distinction matters: the cluster
        # list is demonstrably incomplete. 107 lexicon forms have an onset
        # it does not list, and many are plainly real — `bmiang` "the
        # margin", `bpei` "hearth; ashes", `bwieng` "the intestines of a
        # bird", `gra` "old brass vessel". Excluding them would lose real
        # vocabulary; demoting them only means a word that IS well formed
        # gets asked first.
        if _INVALID_SHAPE_PENALTY and not _phonotactically_ok(sf):
            d += _INVALID_SHAPE_PENALTY
        if 0 < d <= max_dist:
            f      = freq_data.get(sf, 0)
            # Prefix bonus: candidates sharing more leading phonemes with the
            # input word rank higher when edit distance is tied. Uses the
            # same weight as the final merge in suggest(); the literal 3 here
            # used to drift from _PREFIX_BONUS_WEIGHT, so changing the
            # constant (as the ablation script does) only half took effect.
            prefix = _shared_phoneme_prefix(word, sf)
            score  = f + prefix * _PREFIX_BONUS_WEIGHT
            scored.append((d, _first_letter_differs(word, sf), -score, sf))

    scored.sort()
    return [(d, sf) for d, _, _, sf in scored[:top_n]]


# ---------------------------------------------------------------------------
# Speller factory  (version-aware)
# ---------------------------------------------------------------------------

def _build_speller(freq_dict: dict):
    try:
        from autocorrect import Speller
    except ImportError:
        return None

    try:
        sig    = inspect.signature(Speller.__init__)
        params = set(sig.parameters.keys())

        # expand_morphology=False: the engine expands the table itself, from
        # the lexicon's own prefix inventory (see _expand_frequency_table).
        # The vendored Speller's expansion used its hard-coded prefix list,
        # which treats the free morphemes ba, la, iai and nang as prefixes
        # and knows nothing of sngew- or ïa-.
        if "expand_morphology" in params:
            return Speller(lang="kh", nlp_data=freq_dict, expand_morphology=False)
        if "nlp_data" in params:
            return Speller(lang="kh", nlp_data=freq_dict)

        speller = Speller(lang="kh")
        speller.nlp_data = freq_dict
        return speller

    except Exception as e:
        print(f"[khasi-nlp] Warning: Speller init failed — {e}")
        return None


def _expand_frequency_table(freq: dict) -> int:
    """Add prefix + root combinations to *freq*, in place. Returns the count.

    The table is what corpus frequencies are later mapped onto, and it is the
    whitelist of forms that may earn the vote's "frequent" signal: a string
    is only eligible if it is a lexicon form or a plausible combination. The
    heads come from the lexicon's own morphology block — every prefix marked
    `productive`, with assimilation applied (pyn- + lait -> pyllait), and the
    `free_morphemes` (ba, la, iai, nang, …) that are written solid before a
    root. The vendored Speller used a fixed list that called the free
    morphemes prefixes and omitted sngew- and ïa-.

    Values stay BELOW the frequent bar. They are structural placeholders, not
    observations, and a placeholder must never earn "frequent" on its own —
    only a corpus count can lift a combination over the bar.
    """
    try:
        from khasi_engine import morphology as _morph
        from khasi_engine.assimilation import should_assimilate
    except Exception:                                   # pragma: no cover
        return 0
    heads: list[tuple[str, int, bool]] = []
    for name, info in (getattr(_morph, "PREFIXES", None) or {}).items():
        if info.get("productive"):
            heads.append((name, 2, True))
    for name in ((getattr(_morph, "_MORPH", None) or {}).get("free_morphemes") or {}):
        heads.append((name.lower(), 4, False))
    cap = KhasiSpellChecker._FREQ_HIGH_BAR - 1
    roots = [(w, v) for w, v in freq.items() if w and " " not in w]
    added = 0
    for head, divisor, assimilates in heads:
        for root, value in roots:
            derived = None
            if assimilates:
                applied, surface = should_assimilate(head + "-", root)
                if applied and surface:
                    derived = surface
            derived = derived or head + root
            if derived not in freq:
                freq[derived] = max(1, min(int(value) // divisor, cap))
                added += 1
    return added


# ---------------------------------------------------------------------------
# Main spell checker
# ---------------------------------------------------------------------------

class KhasiSpellChecker:
    """
    Morphology-gated spell checker.

    The morphological engine is the primary validity oracle. The spell
    checker only activates when all morphological phases fail to recognise
    a word. This prevents valid derived forms (pynbha, jingsniew, …) from
    ever being flagged as misspellings.

    Parameters
    ----------
    db          : KhasiDB — the shared lexicon
    morph_check : callable(word: str) -> bool
                  Returns True if the word is morphologically valid.
                  Injected by KhasiAnalyser to avoid a circular import.
    use_hybrid  : bool — enable FastText re-ranking
    model_path  : str  — path to FastText model file
    """

    def __init__(
        self,
        db: "KhasiDB",
        morph_check: Optional[Callable[[str], bool]] = None,
        use_hybrid: bool = False,
        model_path: Optional[str] = None,
    ):
        self._db         = db
        self._morph_check = morph_check   # injected after analyser is constructed
        self._morph_diag  = None          # richer parse oracle, optional
        # FastText re-ranking flag. The model itself is handled by
        # khasi_engine.w2v_client, which prefers the bundled local model and
        # falls back to a remote Space; this object only holds the switch.
        self._hybrid     = False
        # model_path is accepted for backwards compatibility with the v1
        # KhasiAnalyser constructor and ignored: w2v_client resolves the
        # model (KHASI_W2V_MODEL overrides the bundled path).
        _ = model_path

        # ── Per-checker state ─────────────────────────────────────────────
        # Everything a speller adds on top of the lexicon lives HERE, not on
        # the KhasiDB, which is memoised and shared by every speller in the
        # process. Stored on the database, one speller's corpus pool,
        # corpus frequencies and taught words leaked into the next.
        #
        #   _corpus_forms   corpus types the pool admitted: offerable
        #   _corpus_accept  those, plus their reduced spelling: acceptable
        #   _taught         add_word(): acceptable and offerable, any spelling
        #   _extra_index    delete index over _corpus_forms and _taught
        #   _ranking_freq   corpus-scaled lexicon frequencies (gate-1 search)
        self._corpus_forms: set = set()
        self._corpus_accept: set = set()
        self._taught: set = set()
        self._extra_index: Optional[_DeleteIndex] = None
        self._ranking_freq: Optional[Mapping[str, int]] = None
        self._corpus_freq_applied = False
        self._corpus_accept_on = False
        self._lock = threading.RLock()
        _phonotactically_ok.cache_clear()

        freq_dict = dict(db.to_freq_dict())
        _expand_frequency_table(freq_dict)
        self._speller = _build_speller(freq_dict)

        if self._speller:
            n = len(getattr(self._speller, "nlp_data", freq_dict))
            # Report the source actually used, not a hard-coded label. This
            # line said "khasi_db.json" on every run, PostgreSQL included, so
            # it agreed with KhasiDB's own start-up line only by accident and
            # contradicted it the rest of the time — the same failure mode
            # KhasiDB.__init__ warns about a few hundred lines away.
            src = ("PostgreSQL" if getattr(db, "_database_url", None)
                   else getattr(db, "_json_path", "khasi_db.json"))
            print(f"[khasi-nlp] Speller ready — {n} words (source: {src})")
        else:
            print("[khasi-nlp] Speller unavailable — using Levenshtein fallback")

        # The index is a property of the lexicon, so it is built once per
        # database rather than once per speller.
        if getattr(db, "_delete_index", None) is None:
            self.build_delete_index()

        if use_hybrid:
            # Honour the constructor flag for backwards compatibility, but
            # the canonical toggle path is set_remote_w2v_enabled().
            self.set_remote_w2v_enabled(True)

    def build_delete_index(self, max_dist: int = 2) -> None:
        """
        Attach a symmetric-delete index to the database, replacing the
        full-vocabulary scan with hash lookups. Costs ~2 s and ~70 MB at
        start-up and saves ~2.5 s on every unknown word.

        Safe to skip: _levenshtein_suggestions falls back to the scan when
        no index is present, which is what DELETE_INDEX_ENABLED=False gives.
        """
        if not DELETE_INDEX_ENABLED:
            return
        try:
            words = [w for w in self._db.all_surface_forms() if w and " " not in w]
            self._db._delete_index = _DeleteIndex(words, max_dist=max_dist)
            print(f"[khasi-nlp] delete index ready — {len(self._db._delete_index):,} keys "
                  f"over {len(words):,} forms")
        except Exception as e:                     # never fatal
            print(f"[khasi-nlp] delete index unavailable, using full scan — {e}")
            self._db._delete_index = None

    def teach_word(self, word: str) -> bool:
        """Accept *word* and make it reachable as a correction, for THIS checker.

        Returns True when the word was new.

        Used by `KhasiSpeller.add_word`. A taught word is accepted before any
        gate runs, so names and loans that break Khasi phonotactics —
        `Meghalaya` has a bare g, `Congress` a c — can be taught too; they
        were rejected at the phonotactic gate however they were added. It is
        also indexed for candidate search, so a typo of it is corrected to it.

        Stored on the checker, not the shared lexicon: teaching one speller
        used to teach every speller in the process.
        """
        w = _tokens.canonical(word or "").lower().strip()
        if not w or " " in w:
            return False
        with self._lock:
            if w in self._taught:
                return False
            self._taught.add(w)
            self._index_extra(w)
        return True

    # ── Corpus supply and acceptance (per checker) ────────────────────────

    def _index_extra(self, w: str) -> None:
        """Add *w* to this checker's own candidate index."""
        if getattr(self._db, "_delete_index", None) is None:
            return                         # full-scan mode reads the sets directly
        if self._extra_index is None:
            self._extra_index = _DeleteIndex((), max_dist=self._db._delete_index.max_dist)
        self._extra_index.add(w)

    def register_corpus_forms(self, forms) -> int:
        """Admit corpus types as offerable-but-not-known candidates.

        They become correction candidates — without them `pyntreikam`, 1,940
        corpus occurrences and not a headword, is not in the list to be
        ranked. They do NOT become lexicon words: `KhasiDB.is_known()` never
        sees them. Returns the number newly registered.
        """
        n = 0
        with self._lock:
            for f in forms:
                w = _tokens.canonical(f or "").lower()
                if w and " " not in w and w not in self._corpus_forms:
                    self._corpus_forms.add(w)
                    self._index_extra(w)
                    n += 1
        return n

    def register_corpus_acceptance(self, forms) -> int:
        """Let corpus words earn the vote's frequency signal. Returns the count."""
        before = len(self._corpus_accept)
        with self._lock:
            self._corpus_accept.update(
                _tokens.canonical(f).lower() for f in forms if f and " " not in f)
        return len(self._corpus_accept) - before

    def is_corpus_form(self, word: str) -> bool:
        """True when *word* is offerable only because the corpus attests it."""
        return _tokens.canonical(word or "").lower() in self._corpus_forms

    def is_corpus_accepted(self, word: str) -> bool:
        """True when the corpus attests *word* often enough to accept it."""
        return _tokens.canonical(word or "").lower() in self._corpus_accept

    def is_taught(self, word: str) -> bool:
        return _tokens.canonical(word or "").lower() in self._taught

    def is_attested(self, word: str) -> bool:
        """Did the lexicon, the admitted corpus pool or the user record *word*?"""
        w = _tokens.canonical(word or "").lower()
        return (self._db.is_attested(w) or w in self._corpus_forms
                or w in self._taught)

    def candidate_forms(self) -> list[str]:
        """Every single-token form this checker can offer, sorted."""
        extra = (self._corpus_forms | self._taught) - self._db.surface_form_set()
        return sorted(set(self._db.all_surface_forms()) | extra)

    def candidate_form_count(self) -> int:
        """len(candidate_forms()), without building the list."""
        extra = (self._corpus_forms | self._taught) - self._db.surface_form_set()
        return len(self._db.surface_form_set()) + len(extra)

    def set_ranking_frequencies(self, freq: Optional[Mapping[str, int]]) -> None:
        """Frequencies the first-pass candidate search ranks by.

        Set by khasi_spell.corpus_freq so the phonotactic-gate path ranks on
        the same corpus counts as the final merge. Held per checker; it used
        to be written into the shared lexicon's own table.
        """
        self._ranking_freq = freq

    def _search_kwargs(self) -> dict:
        """This checker's state, as `_levenshtein_suggestions` takes it."""
        return {
            "freq": self._ranking_freq,
            "extra_index": self._extra_index,
            "extra_forms": self._corpus_forms | self._taught,
            "penalised": self._corpus_forms,
        }

    def set_morph_check(self, morph_check: Callable[[str], bool]) -> None:
        """
        Inject the morphological validity callable after construction.
        Called by KhasiAnalyser once its own pipeline is ready.
        """
        self._morph_check = morph_check

    def set_morph_diagnostic(self, morph_diag: Callable[[str], dict]) -> None:
        """
        Inject the richer morphological diagnostic callable. Returns a dict
        with at least the keys `valid` and `has_affix`. Used by the
        confidence-scored gate to distinguish "bare-root parse" (weak signal)
        from "real affixation" (medium signal).
        """
        self._morph_diag = morph_diag

    def load_model(self, model_path: Optional[str] = None) -> None:
        """
        Kept for backwards compatibility. Flips the hybrid flag; the model
        itself is resolved by w2v_client — the bundled local model first,
        then a remote Space when KHASI_W2V_URL is set.
        """
        _ = model_path
        self.set_remote_w2v_enabled(True)

    def unload_model(self) -> None:
        """
        Backwards-compatible alias for `set_remote_w2v_enabled(False)`.
        Only the flag changes; `w2v_client.unload_local()` frees a loaded
        local model.
        """
        self.set_remote_w2v_enabled(False)

    def set_remote_w2v_enabled(self, enabled: bool) -> None:
        """Toggle FastText re-ranking on or off.

        The name is historical: the backend is whatever w2v_client resolves,
        which is the bundled LOCAL model when present and a remote Space only
        otherwise.
        """
        if enabled:
            from . import w2v_client
            if not w2v_client.is_configured():
                # Neither a local model nor KHASI_W2V_URL: stay rule-only.
                self._hybrid = False
                return
            self._hybrid = True
            print(f"[khasi-nlp] Spell-checker hybrid ENABLED "
                  f"({w2v_client.active_backend()} backend).")
        else:
            self._hybrid = False
            print("[khasi-nlp] Spell-checker hybrid DISABLED (rule-only).")

    # ------------------------------------------------------------------
    # Sync  (call after any DB mutation)
    # ------------------------------------------------------------------

    def sync_with_db(self) -> None:
        """Rebuild what this checker derives from the lexicon after an edit.

        Used to swap in the raw frequency table only: the expansion, the
        delete index and the morphological generator all kept the old
        lexicon, so an added entry was accepted but never offered. Corpus
        frequencies applied by the facade are not re-applied here.
        """
        freq_dict = dict(self._db.to_freq_dict())
        _expand_frequency_table(freq_dict)
        for w in self._taught:
            freq_dict.setdefault(w, 20)
        if self._speller and hasattr(self._speller, "nlp_data"):
            self._speller.nlp_data = freq_dict
        elif self._speller:
            self._speller = _build_speller(freq_dict)
        self.build_delete_index()
        try:
            from khasi_engine.generate import reset_generator
            reset_generator()
        except Exception:                               # pragma: no cover
            pass
        _phonotactically_ok.cache_clear()

    # ------------------------------------------------------------------
    # Gate helpers
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Run-together splits
    # ------------------------------------------------------------------

    def _split_table(self) -> dict:
        """{solid form: spaced form}, loaded once. See khasi_spell.splits."""
        t = getattr(self, "_splits_cache", None)
        if t is None:
            try:
                from khasi_spell import splits
                t = splits.load()
            except Exception:
                t = {}
            self._splits_cache = t
        return t

    def _split_suggestion(self, word: str) -> Optional[str]:
        """The two-word form of *word*, when the corpus says it is one.

        Edit distance cannot propose a space — every candidate is drawn from
        a vocabulary of single words — so the correct answer for `jongki`
        (`jong ki`, written spaced 17,312 times against 2,642 solid) was not
        merely mis-ranked, it was unreachable. The list instead offered
        `jongka`, which is the same error with a different pronoun, and
        `jongka` was offered `jongki` straight back.
        """
        return self._split_table().get((word or "").lower().strip())

    def _offerable(self, cand: str, word: str = "") -> bool:
        """
        May *cand* be offered as a correction (of *word*, when given)?

        Three conditions, and the first two exist because a correction that
        is itself not a word is worse than no correction. Rejecting `ng`
        and then suggesting `n` and `g` — which the checker also rejects —
        is not help.

          * it contains a vowel. Every Khasi syllable requires a vowel
            nucleus, so a bare consonant is not a candidate word. This is
            what separates `n`/`g`/`t`/`k` from the legitimate one-letter
            clitics `u`, `i`, `a`;
          * it is not itself listed in `semantically_invalid_standalone`;
          * it is phonotactically valid **or** attested in the lexicon.
            `is_attested` rather than `is_known`: `validate()` has a small
            false-reject rate, and a form the lexicon wrote down stays
            offerable even when it is not an acceptable standalone word —
            `shafon` for `shafan`.

        The distance metric makes the vowel rule necessary rather than
        merely tidy. Phoneme-aware distance scores `ng` -> `n` at 0.4,
        because deleting half a digraph is cheap, so bare consonants
        outrank the real answers `nga` and `ngi` at 1.0 and fill the list.

        A fourth condition applies when *word* is supplied: the candidate
        must **share a character with it**. Digraphs are single phonemes, so
        `ph` -> `u` is one substitution and scores 1.0 — exactly the same as
        `ph` -> `phi`, which is the actual word. The one-letter clitics `u`,
        `a`, `i` then win the tie on frequency, and `ph` was answered with
        `['u', 'phi', 'a', 'i', 'ï']`. Only `phi` is a correction of `ph`;
        the rest share nothing with the input at all. Requiring some common
        material is a weak test, but it is exactly the one the distance
        metric cannot make on its own.
        """
        if not cand:
            return False
        from khasi_engine.phonology import (
            VOWELS as _V, SEMANTICALLY_INVALID_STANDALONE as _INVALID,
        )
        low = _tokens.canonical(cand).lower()
        # A word the user taught this checker is offerable whatever its
        # letters — that is the point of teaching `Meghalaya` or `Congress` —
        # though never as a "correction" of itself.
        if low in self._taught:
            return not (word and low == _tokens.canonical(word).lower())
        # A correction that is the input is not a correction. This could not
        # arise while the pool was the lexicon and the gate refused anything
        # in it, but a corpus-derived candidate can be rejected by the vote
        # and present in the pool at once, which offers `pyntreikam` for
        # `pyntreikam` at distance 0 — the top of the list, and useless.
        if word and low == word.lower():
            return False

        # Stripping the input's diacritics is not a correction. `ïatreilang`
        # was answered with `iatreilang` — the same word with the diaeresis
        # removed — which pushes the writer off the only spelling the
        # lexicon uses (`ïatrei` 332 initial cases, plain `ia-` zero).
        #
        # Narrow on purpose: it refuses the fold of the input and nothing
        # else, so a candidate that merely happens to carry fewer diacritics
        # somewhere else is untouched, and `ïiatreilang` -> `ïatreilang`
        # still works. Corpus types are admitted already canonicalised, so
        # this is the backstop rather than the mechanism.
        if word:
            wl = word.lower()
            if low != wl and low == wl.translate(_DIACRITIC_FOLD):
                return False
        if low in _INVALID or not any(ch in _V for ch in low):
            return False

        # Never offer a word we would ourselves flag as a run-together.
        # `jongka` was suggested for `jongki` and `jongki` for `jongka` —
        # both the same error with a different clitic, cycling the writer
        # between two forms neither of which is right.
        if " " not in low and self._split_suggestion(low):
            return False

        # Illegal characters are a HARD constraint, unlike the rest of
        # phonotactics. c, f, q, v, x and z are simply not in the Khasi
        # alphabet (phonology.valid_chars), so no Khasi word contains one —
        # there is nothing for `is_attested` to overrule. That distinction
        # was missed earlier: `shafon` was restored to the suggestions for
        # `shafan` on the grounds that the lexicon attests it, but the
        # lexicon attests it only because English gloss text leaked into 260
        # surface-form fields ('a poisoned fish as', 'advanced age'). It is
        # not a Khasi word and should never have been offered.
        #
        # The exception is named entities. Meghalaya's Garo and Bengali place
        # names use these letters freely — 652 of the 6,609 gazetteer tokens
        # contain one (Achakchiring, Achugre, Adugachol) — and those are real
        # words a writer may well mistype.
        from khasi_engine.phonology import VALID_CHARS as _VALID
        if any(ch.isalpha() and ch not in _VALID for ch in low):
            if low not in _named_entities():
                return False
            return True
        # No Khasi word ends in y — hard for the same reason: the lexicon
        # attests `bodily` and `longingly` only because English text became
        # headwords. Names keep their exemption here too.
        from khasi_engine.phonology import ends_in_forbidden_vowel
        if ends_in_forbidden_vowel(low) and low not in _named_entities():
            return False
        # Nor is an English word, unless the Khasi lexicon records it
        # (maintainer ruling 2026-10-01; see _english_words).
        if (low in _english_words() or low in _name_only_words()) and not self._db.is_known(low):
            return False
        if word:
            w = word.lower()
            if low != w and not (set(low) & set(w)):
                return False
        return self._phonotactically_valid(cand) or self.is_attested(cand)

    def _phonotactically_valid(self, word: str) -> bool:
        from khasi_engine.phonology import is_phonotactically_valid
        return is_phonotactically_valid(word)

    def _morphologically_valid(self, word: str) -> bool:
        """
        Returns True if the morphological engine accepts *word* as valid.
        Falls back to lexicon lookup if the morph_check callable is not
        yet injected.
        """
        if self._morph_check is not None:
            return self._morph_check(word)
        # Fallback: direct lexicon lookup only
        return self._db.is_known(word)

    # ── Confidence-weighted gate ──────────────────────────────────────────
    # Replaces the old "any single positive signal stops the pipeline"
    # cascade with a multi-signal vote. The lexicon dominates because it
    # is curated ground truth; morphology with real affixation is medium
    # evidence; phonotactic validity and corpus frequency contribute
    # partial credit. Threshold 3 means a lexicon hit alone passes, OR
    # combinations of medium/weak signals together pass.
    _W_LEXICON       = 3
    _W_MORPH_AFFIXED = 2
    # An affixed parse whose root is not itself a word. Scores below
    # _W_MORPH_AFFIXED so that phonotactic validity alone no longer carries
    # it over the threshold: 2 + 1 cleared the bar, 1 + 1 does not.
    _W_MORPH_IMPLAUSIBLE = 1
    _W_MORPH_BARE    = 0   # no signal — too easy to satisfy
    _W_PHONOTACTIC   = 1
    _W_FREQUENT      = 2
    _W_THRESHOLD     = 3
    _FREQ_HIGH_BAR   = 10  # min corpus freq to count as "frequent"
    # The longest token suggest() will search for. Candidate generation (the
    # delete index, the compound splitter) grows with the cube of the length:
    # a 1,000-letter token took 9 s and 480 MB, one of 2,000 had not finished
    # after a minute, so a single request could stall or crash the service.
    # No real word comes close — the lexicon's longest single token is 27
    # characters, the corpus's 31 — and at 50 the slowest input measured
    # (a repeated prefix) takes 0.11 s.
    MAX_WORD_CHARS   = 50
    # Open word classes — the ones a productive derivation can be built on.
    # Used by _root_is_substantive() to tell a prefix that is also a content
    # word (sngew VERB) from one that is not (ia PREPOSITION).
    _OPEN_CLASS = frozenset({"NOUN", "VERB", "ADJ", "ADJECTIVE", "ADV", "ADVERB"})
    # Grammar words take no prefix — maintainer ruling 2026-10-04. pyn- + ka,
    # jing- + ki, pyn- + ba and jing- + ne were accepted as derivations. Some
    # of these the lexicon labels as content words (`ba` verb, `ne` and `u`
    # noun, `ym` adverb), so the class test alone cannot refuse them.
    _NO_PREFIX_ROOTS = frozenset({"ka", "ki", "u", "i", "ba", "ne", "ha", "na",
                                  "ia", "ïa", "sha", "la", "ym", "bad"})
    # Lexicon classes that are grammar words. "other" and "phrase" are mixed
    # bags and stay out.
    _CLOSED_CLASS_LABELS = frozenset({"determiner", "pronoun", "preposition",
                                      "conj", "particle", "interjection"})

    def _root_is_substantive(self, root: str) -> bool:
        """A parse's root must be a WORD — not a morpheme, not a fragment.

        `root_known` asks db.is_known(root), which is deliberately generous:
        it also answers True for tokens harvested out of multi-word phrase
        entries, so that words like `shnong` appearing only inside phrases
        stay valid. That generosity leaks into derivation, where two refusals
        are warranted:

        * A CLOSED-CLASS registered prefix is not a root. `ia-`/`ïa-` are
          both the reciprocal prefix and a preposition, so jing-ia-pyn-ia
          "derives" ia from ia. Note the qualifier: 15 of the 18 registered
          prefixes are also words, because Khasi prefixes are grammaticalised
          content words — `sngew` is VERB "to feel", `wan` VERB "to come",
          `nong`/`jing` NOUN. A blanket "a prefix is never a root" rule was
          tried first and broke `nongsngew` (two suite failures); only the
          closed-class prefixes are refused.
        * A root with neither a headword entry nor a gloss is not a word.
          `wo` and `do` reach is_known() only as phrase fragments.

        Deliberately NOT a minimum root length. Khasi has real two-letter
        roots — `ai` (give), `im` (live) — and tests/golden_corpus.json
        records the 2026-06-04 ruling that ai is a free verb root, so a
        length gate would contradict it. Measured: this keeps all 213
        genuine derivations probed and removes 47 of 112 invented ones.
        """
        r = (root or "").lower()
        if not r:
            return False
        if r in self._NO_PREFIX_ROOTS:
            return False
        matches = self._db.lookup(r) or []
        # A word the lexicon records only as a grammar word — determiner,
        # pronoun, preposition, conjunction — is not a root either. Until
        # 2026-10-04 only the closed-class PREFIXES (ia, ïa) were refused, so
        # pyn- + ka and jing- + ki passed. Lookup can fall through to other
        # spellings, so only the root's own entries count.
        own = {(m.get("grammatical_class") or "").strip().lower()
               for m in matches if (m.get("surface_form") or "").lower() == r}
        own.discard("")
        if own and own <= self._CLOSED_CLASS_LABELS:
            return False
        try:
            from khasi_engine import morphology as _morph
            if r in {k.lower() for k in getattr(_morph, "PREFIXES", {})}:
                pos = next((m.get("grammatical_class", "").upper()
                            for m in matches
                            if (m.get("grammatical_class") or "").strip()), "")
                if pos not in self._OPEN_CLASS:
                    return False
        except Exception:
            pass
        if r in getattr(self._db, "_surface_index", {}):
            return True
        return any((m.get("english_gloss") or "").strip() for m in matches)

    def _confidence_score(self, word: str) -> dict:
        """
        Score how confident we are that *word* is a real Khasi word.
        Returns {"score": int, "signals": dict[str,int], "threshold": int}.
        """
        w = word.lower().strip()
        signals: dict[str, int] = {}

        # Strongest: surface form in lexicon
        if self._db.is_known(w):
            signals["in_lexicon"] = self._W_LEXICON

        # Medium / zero: morphology, weighted by whether real affixation
        # was used. Bare-root parses (no prefix/infix/suffix/clitic) earn
        # nothing, since any syllable concatenation tends to satisfy them.
        morph_affixed = False
        morph_valid   = False
        morph_plausible = True      # assume plausible when we cannot tell
        if self._morph_diag is not None:
            try:
                diag = self._morph_diag(w) or {}
                morph_valid   = bool(diag.get("valid"))
                morph_affixed = bool(diag.get("has_affix"))
                if "root_known" in diag:
                    # Whether the root is a word is the ONLY signal used here.
                    # parse_score used to be an alternative way to qualify
                    # (`or score > 0`), but it does not separate the two
                    # populations: a stacked-prefix parse over an invented
                    # root scores a flat 0.15, and 103 of 213 genuine
                    # derivations score exactly 0.15 too. That disjunct let
                    # through 414/414 probed non-words (jingialehka,
                    # jingiamareha, ...) — every one of them accepted on
                    # morphology 2 + phonotactic 1 = the threshold. Dropping
                    # it costs nothing: all 213 genuine derivations qualify
                    # on root_known alone. root_known is then narrowed by
                    # _root_is_substantive(), which refuses a root that is
                    # itself an affix or is only a multi-word phrase
                    # fragment (ia, ïa, wo, do).
                    morph_plausible = (bool(diag["root_known"])
                                       and self._root_is_substantive(
                                           diag.get("root") or ""))
                # …and a well-formed parse still loses to a real word one
                # edit away. `nongihkai` decomposes as nong- + ih + -kai over
                # a genuine root, so every test above passes, yet the writer
                # transposed two letters of `nonghikai` "instructor" — a word
                # the lexicon holds. A transposition is the one typo class
                # that reliably survives this gate: it preserves the letters,
                # so the string stays phonotactically legal and still splits
                # into recognisable morphemes. The analyser already refuses a
                # fused compound on the same grounds; this extends it to the
                # affix path.
                #
                # Demoted rather than dismissed, to _W_MORPH_IMPLAUSIBLE. A
                # form that is genuinely written this often clears the bar
                # anyway on corpus frequency (2) + phonotactics (1), so real
                # but unlisted derivations keep a route to acceptance.
                if diag.get("outranked_by"):
                    morph_plausible = False
            except Exception:
                pass
        elif self._morph_check is not None:
            try:
                morph_valid = bool(self._morph_check(w))
            except Exception:
                pass
        if morph_valid:
            if not morph_affixed:
                weight = self._W_MORPH_BARE
            elif morph_plausible:
                weight = self._W_MORPH_AFFIXED
            else:
                # Affixed, but the root is not a word — the analyser could
                # make this decomposition, not that it should be trusted.
                weight = self._W_MORPH_IMPLAUSIBLE
            if weight:
                signals["morphology"] = weight

        # Weak: phonotactic. By itself, never enough to clear the threshold.
        if self._phonotactically_valid(w):
            signals["phonotactic"] = self._W_PHONOTACTIC

        # Medium: corpus frequency. The autocorrect Speller's nlp_data
        # holds frequencies harvested from the lexicon corpus.
        freq_data = getattr(self._speller, "nlp_data", {}) if self._speller else {}
        if freq_data.get(w, 0) >= self._FREQ_HIGH_BAR:
            signals["frequent"] = self._W_FREQUENT
        elif self._corpus_accept_on and w in self._corpus_accept:
            # A corpus word the lexicon does not record. nlp_data is built
            # from the lexicon, so this signal never reached such a word:
            # `jylla`, seen 24,252 times, scored 1 of the 3 needed. Words
            # land here only after corpus_pool has admitted them — seen at
            # least its floor of times (above _FREQ_HIGH_BAR), Khasi letters
            # only, phonotactically possible, not dominated by a far
            # commoner neighbour, not a run-together spelling.
            #
            # The set and the switch both live on the checker, not the
            # database: KhasiDB is memoised per data source, so every speller
            # in a process shares one, and anything stored there leaked.
            signals["frequent"] = self._W_FREQUENT

        return {
            "score":     sum(signals.values()),
            "signals":   signals,
            "threshold": self._W_THRESHOLD,
        }

    # ------------------------------------------------------------------
    # Core public API
    # ------------------------------------------------------------------

    def is_known(self, word: str) -> bool:
        """
        True if the checker accepts *word* — the same decision `suggest()`
        makes, so the two can never disagree.

        This used to answer "does morphology accept it, or is it in the
        frequency table", a looser rule than the confidence vote: any bare-root
        parse and any of the generated prefix+root strings counted.
        Nothing in the package called it, which is how the divergence went
        unnoticed.
        """
        return bool(self.suggest(word, n=1).get("is_known"))

    # _rank_candidates() was removed 2026-09-30. Its docstring said it was
    # shared by gates 1 and 3; neither called it. It also ranked differently
    # from the live path (no attested bonus, the Speller merged regardless of
    # SPELLER_CANDIDATES_ENABLED), so reading it misdescribed the ranking.

    def _hyphenated_part_veto(self, word_lower: str, conf: dict,
                              in_lexicon: bool) -> Optional[dict]:
        """Reject an accepted hyphenated compound whose part is a misspelling.

        Phase 4 accepts any well-formed two-part hyphenated token as a
        compound without checking that its parts are words, so `man-miay`
        cleared the vote on morphology 2 + phonotactics 1 although `miay` is
        not a word. The text path caught this in the facade and the word
        path did not, so `/word`, `/batch`, `correct()` and the CLI accepted
        what `/check` flagged. The rule now lives here, where every entry
        point passes.

        A compound the lexicon records, or the corpus attests often enough to
        count as frequent, is a word in its own right whatever its parts look
        like (`jrain-jrain`). A part that is not a word but has no suggestion
        is left alone, exactly as before: an unknown part with nothing to
        offer is not evidence of a typo.
        """
        if "-" not in word_lower or in_lexicon or "frequent" in conf.get("signals", {}):
            return None
        parts = word_lower.split("-")
        if any(not p for p in parts):
            return None
        fixed, bad = [], []
        for part in parts:
            if self._db.is_known(part) or part in self._taught:
                fixed.append(part)
                continue
            r = self.suggest(part, n=1)
            if r.get("is_known") or not r.get("suggestions"):
                fixed.append(part)
                continue
            fixed.append(r["suggestions"][0])
            bad.append((part, r["suggestions"][0]))
        if not bad:
            return None
        rebuilt = "-".join(fixed)
        note = "; ".join(f"'{p}' is not a word (did you mean '{s}'?)" for p, s in bad)
        return {
            "is_known":              False,
            "morphologically_valid": False,
            "phonotactically_valid": True,
            "suggestions":           [rebuilt],
            "suggestion_distances":  [round(_levenshtein(word_lower, rebuilt), 3)],
            "method":                "hyphenated_part",
            "in_lexicon":            False,
            "gate_reached":          3,
            "confidence":            {**conf, "veto": note},
            "reason":                note,
        }

    def suggest(self, word: str, n: int = 5) -> dict:
        """
        Morphology-gated spelling decision and suggestions for *word*.

        The input is normalised first: Unicode NFC, and the typographic
        apostrophe (’) read as the ASCII one. Then:

        Taught words — accepted outright (method "user_dictionary").

        Gate 0 — semantically invalid standalone token (`ng`):
            is_known=False, gate 0, with suggestions from a narrow search.

        Gates 1+2 — the confidence vote (lexicon 3, affixed morphology 2 or
            1, corpus frequency 2, phonotactics 1; threshold 3):
            clears → is_known=True, method "confidence_gate", gate 2 — unless
                a hyphenated compound the lexicon does not record has a part
                that is itself a misspelling, which is rejected with the
                repaired compound (method "hyphenated_part", gate 3);
            below, and phonotactically invalid → gate 1, method
                "levenshtein_phonotactic", suggestions from a widened search
                plus banned-letter repair, and the validator's `reason`.

        Gate 3 — below the threshold but well formed: candidates from the
            delete index, the corpus pool and the morphological generator,
            ranked by distance then frequency, morphology and shared onset.
            Optional FastText re-ranking; a run-together split goes first.
        """
        # Strip surrounding punctuation but preserve ' (glottal stop) and - (compound)
        import re as _re
        word = _tokens.canonical(word or "")
        word = _re.sub(r"^[^\w\u00ef\u00f1\u00cf\u00d1'\-]+|[^\w\u00ef\u00f1\u00cf\u00d1'\-]+$", "", word.strip())
        if not word:
            return {"is_known": False, "morphologically_valid": False,
                    "phonotactically_valid": False, "suggestions": [],
                    "method": "empty", "in_lexicon": False, "gate_reached": 0}
        word_lower = word.lower()

        # A Cyrillic or Greek letter drawn like a Latin one (`shn\u043eng`, with a
        # Cyrillic \u043e) leaves the word looking right while it matches nothing.
        # Reported with the letter named, and the Latin spelling offered when
        # that is a word — otherwise its own suggestions.
        latin = _tokens.latinise(word)
        if latin != word:
            import unicodedata as _ud
            inner = self.suggest(latin, n=n)
            if inner.get("is_known"):
                offer, dist = [latin], [0.0]
            else:
                offer = list(inner.get("suggestions") or [])[:n]
                dist = list(inner.get("suggestion_distances") or [])[:len(offer)]
            foreign = sorted({ch for ch in word if _tokens.latinise(ch) != ch})
            named = ", ".join(f"{ch} ({_ud.name(ch, 'letter').title()})" for ch in foreign)
            note = (f"Contains a letter from another alphabet that looks Latin: {named}. "
                    f"Typed with Latin letters it reads '{latin}'.")
            return {
                "is_known":              False,
                "morphologically_valid": False,
                "phonotactically_valid": False,
                "suggestions":           offer,
                "suggestion_distances":  dist,
                "method":                "lookalike_letters",
                "in_lexicon":            False,
                "gate_reached":          3,
                "confidence":            {"score": 0, "signals": {},
                                          "threshold": self._W_THRESHOLD, "veto": note},
                "reason":                note,
            }

        # A word the user taught this checker (KhasiSpeller.add_word) is
        # accepted before any gate: names and loans such as `Meghalaya` break
        # Khasi phonotactics by design, and teaching them is the only way to
        # stop them being flagged.
        if word_lower in self._taught:
            return {
                "is_known":              True,
                "morphologically_valid": False,
                "phonotactically_valid": self._phonotactically_valid(word_lower),
                "suggestions":           [],
                "method":                "user_dictionary",
                "in_lexicon":            self._db.is_known(word_lower),
                "gate_reached":          2,
                "confidence":            {"score": self._W_THRESHOLD,
                                          "signals": {"user_dictionary": self._W_THRESHOLD},
                                          "threshold": self._W_THRESHOLD},
            }

        # Far longer than any Khasi word: answered without a search, which at
        # this length would take minutes (see MAX_WORD_CHARS). A long headword,
        # such as a hyphenated phrase the lexicon records, still passes.
        if len(word_lower) > self.MAX_WORD_CHARS and not self._db.is_known(word_lower):
            note = (f"Too long to check as one word: {len(word_lower)} characters, "
                    f"and the limit is {self.MAX_WORD_CHARS}. Is a space missing?")
            return {
                "is_known":              False,
                "morphologically_valid": False,
                "phonotactically_valid": False,
                "suggestions":           [],
                "suggestion_distances":  [],
                "method":                "too_long",
                "in_lexicon":            False,
                "gate_reached":          3,
                "confidence":            {"score": 0, "signals": {},
                                          "threshold": self._W_THRESHOLD, "veto": note},
                "reason":                note,
            }

        # ── Gate 0: Semantically invalid standalone check ─────────────
        # Tokens like "ng" listed in semantically_invalid_standalone are
        # phonotactically well-formed digraphs that carry no standalone
        # meaning in running text.
        #
        # This used to defer to the lexicon — `and not self._db.is_known(...)`
        # — on the reasoning that the DB might hold them as proper entries.
        # It does: kh_DB_005909 is "ng", glossed "The seventh letter of the
        # Khasi Alphabet". So the escape hatch always fired and the gate
        # never blocked anything, which is how `ng` came to pass a spell
        # check as an ordinary word.
        #
        # A letter-name belongs in a dictionary, but it is not a word of
        # running prose, and `semantically_invalid_standalone` is the
        # curated, explicit statement of exactly that. An incidental
        # headword should not overrule it, so the lexicon check is gone.
        # Membership, not error-string matching. This used to grep the
        # validator's messages for "sub-phonemic unit" / "no standalone",
        # which silently coupled the gate to prose: adding an unrelated rule
        # whose message happened to contain "no standalone" routed ordinary
        # misspellings here, and because check_sentence only reports
        # gate_reached 1 or 3, they stopped being flagged at all —
        # 249/249 detection fell to 214/249. The set is the actual authority.
        from khasi_engine.phonology import (
            SEMANTICALLY_INVALID_STANDALONE as _INVALID_STANDALONE,
        )
        if word_lower in _INVALID_STANDALONE:
            # Rejecting without suggesting is half an answer. A fragment is
            # almost always a truncated or mistyped word — `ng` for `nga`
            # 'I', `ngi` 'we' — so the same narrow, distance-first search
            # Gate 1 uses runs here too. It was returning [] simply because
            # the gate was written to block, not to help.
            # Over-fetch: filtering happens after the search, so asking for
            # exactly n leaves fewer than n. For `ng` the bare consonants
            # `n` and `g` occupy two of five slots at distance 0.4 and are
            # then discarded, which cost the real answer `nga` its place.
            _pairs = _levenshtein_suggestions(word_lower, self._db,
                                              max_dist=3, top_n=n * 4,
                                              **self._search_kwargs())
            _kept = [(d, s) for d, s in _pairs if self._offerable(s, word_lower)][:n]
            _suggs = [s for _, s in _kept]
            return {
                "is_known":              False,
                "morphologically_valid": False,
                "phonotactically_valid": False,
                "suggestions":           _suggs,
                # Blended phoneme/Damerau distance per suggestion, in the same
                # order. Surfaced so a caller can show *how close* a candidate
                # is rather than only its rank; see speller.WordResult.
                "suggestion_distances":  [round(d, 3) for d, _ in _kept],
                "method":                "semantically_invalid_standalone",
                "in_lexicon":            False,
                "gate_reached":          0,
            }

        # ── Gate 1+2: confidence-weighted "is this a real word" decision ──
        # Replaces the old cascade where any single positive signal (lexicon
        # OR morphology OR phonotactic) was enough to declare the input
        # valid. The cascade was too lenient — typos like "shilong" parsed
        # morphologically (shi + long) and skipped spell-check entirely.
        # Now: lexicon presence is the dominant signal (weight 3), morphology
        # WITH actual affixation is medium (weight 2), bare-root parses are
        # ignored, phonotactics is weak (weight 1), and high corpus frequency
        # is medium (weight 2). Threshold 3 means lexicon-hit is alone
        # sufficient, otherwise a combination of medium signals is needed.
        conf = self._confidence_score(word_lower)
        in_lexicon = "in_lexicon" in conf["signals"]
        morph_valid = "morphology" in conf["signals"] or in_lexicon
        phon_valid = "phonotactic" in conf["signals"]

        # No Khasi word ends in y (maintainer ruling 2026-10-01), and unlike
        # the rest of phonotactics the vote may not overrule that. Before the
        # rule, frequent English words cleared it on corpus frequency 2 +
        # phonotactics 1 (party, deputy, history); the lexicon holds a few
        # y-final forms only as scan damage (bodily, longingly — English
        # text in the 1906 dictionary) or as the first half of a hyphenated
        # entry (jaly of jaly-eit). Such a word fails validation, so it goes
        # to gate 1 below, with the validator's reason and suggestions.
        from khasi_engine.phonology import ends_in_forbidden_vowel
        ends_in_y = ends_in_forbidden_vowel(word_lower)
        # English words are rejected in the same way (maintainer ruling
        # 2026-10-01) unless the Khasi lexicon records the spelling: corpus
        # frequency used to carry `hospital` and `state` over the bar, and a
        # spurious parse carried `within` and `tablet`. See _english_words.
        english = not in_lexicon and (
            word_lower in _english_words()
            # A name or title word (`United`, `Hoping`) is English in lower
            # case and accepted with a capital. See _name_only_words.
            or (word_lower in _name_only_words() and not word[:1].isupper()))
        if conf["score"] >= conf["threshold"] and not ends_in_y and not english:
            veto = self._hyphenated_part_veto(word_lower, conf, in_lexicon)
            if veto is not None:
                return veto
            return {
                "is_known":              True,
                "morphologically_valid": morph_valid,
                "phonotactically_valid": phon_valid,
                "suggestions":           [],
                "method":                "confidence_gate",
                "in_lexicon":            in_lexicon,
                "gate_reached":          2,
                "confidence":            conf,
            }

        # An English word is flagged with no suggestion. It is not a
        # misspelling of a Khasi word, so any Khasi string near it is noise,
        # and auto-correction must leave it alone: `state` was being
        # "corrected" to `star`. Checked before the y rule, so `party` says
        # what it is rather than that it ends in y.
        if english:
            note = f"'{word_lower}' is an English word, not a Khasi word"
            if word_lower in _name_only_words():
                note += "; with a capital it is accepted as a name"
            return {
                "is_known":              False,
                "morphologically_valid": False,
                "phonotactically_valid": phon_valid,
                "suggestions":           [],
                "suggestion_distances":  [],
                "method":                "english_word",
                "in_lexicon":            False,
                "gate_reached":          3,
                "confidence":            {**conf, "veto": note},
                "reason":                note,
            }

        # A hyphenated word that breaks the rule through its last part is
        # best answered part by part, the way an accepted compound with a
        # misspelled part is: `man-miay` -> `man-miat`. The lexicon search
        # below finds only whole entries (`man-man`).
        if ends_in_y and "-" in word_lower:
            veto = self._hyphenated_part_veto(word_lower, conf, in_lexicon=False)
            if veto is not None:
                return {**veto, "phonotactically_valid": False}

        # Low-confidence input. If it also fails phonotactics, surface that
        # to the UI as a distinct "phonotactic warning" path so users can
        # see why we're suggesting corrections (keeps the old gate_reached=1
        # signal intact for the frontend's gate-stage badge).
        if not phon_valid:
            # Deliberately NOT the gate-3 ranker. Sharing it was tried and
            # measured worse (gate-1 top-1 61.9% -> 57.1%): the wider radius
            # plus the Speller merge admits more competitors than it does
            # correct answers when the input is already malformed. The
            # narrow, distance-first search wins here.
            # Over-fetched for the same reason as Gate 0: the filter runs
            # after the search, so asking for exactly n under-delivers.
            lev_pairs = _levenshtein_suggestions(word_lower, self._db,
                                                 max_dist=3, top_n=n * 4,
                                                 **self._search_kwargs())
            lev_kept = [(d, s) for d, s in lev_pairs
                        if self._offerable(s, word_lower)][:n]

            # Nothing found, and the word carries a letter Khasi does not
            # use? Repair the letter and look again, rather than only saying
            # what is wrong. `sbngaifi` searched as written returns ZERO
            # candidates at distance 3 — no Khasi word is near a string with
            # an `f` in it — but read as `sbngaiñ` it reaches `shngaiñ`.
            # Blocking without offering anything is a dead end for the
            # writer; this is the difference between "that is not a word"
            # and "did you mean …".
            if any(c in _BANNED_REPAIR for c in word_lower) or "fi" in word_lower:
                # Always, not only when the plain search came back empty.
                # `kfii` searched as written finds `ki`, `kin`, `kali` — poor
                # but non-empty — so a "only if nothing else" guard skipped
                # the repair and missed `kñi`, which is the actual word.
                # Repaired results are merged with the plain ones and the
                # distance decides.
                merged: dict = {d_s[1]: d_s[0] for d_s in lev_kept}
                for repaired in _repair_banned(word_lower):
                    # The repair may BE the word. `kfii` read as `kñi` is a
                    # lexicon entry outright, and searching outward from it
                    # returned its neighbours while missing it.
                    if self._db.is_known(repaired):
                        d0 = _BANNED_REPAIR_PENALTY
                        if repaired not in merged or d0 < merged[repaired]:
                            merged[repaired] = d0
                    for d, cand in _levenshtein_suggestions(
                            repaired, self._db, max_dist=2, top_n=n * 2,
                            **self._search_kwargs()):
                        # A candidate much shorter than what was typed is not
                        # a correction of it. Without this, `xerox` proposed
                        # `er` and `eoi`.
                        if len(cand) < len(word_lower) * 0.6:
                            continue
                        # Offerability is judged against the ORIGINAL word,
                        # so a candidate still has to share material with
                        # what was actually typed — otherwise `xerox` starts
                        # proposing `eoi`.
                        if not self._offerable(cand, word_lower):
                            continue
                        d += _BANNED_REPAIR_PENALTY
                        if cand not in merged or d < merged[cand]:
                            merged[cand] = d
                # Sort on DISTANCE ONLY, and rely on the sort being stable.
                # Sorting the (distance, candidate) tuple broke ties
                # alphabetically and threw away the frequency ranking the
                # search had already applied: `briew` and `brieng` are both
                # 1.0 from `brief`, and `brieng` won on spelling alone.
                # Insertion order here is the direct results first, in their
                # ranked order, then the repaired ones.
                lev_kept = [(d, c) for c, d in
                            sorted(merged.items(), key=lambda kv: kv[1])][:n]
            lev_suggs = [s for _, s in lev_kept]
            return {
                "is_known":              False,
                "morphologically_valid": False,
                "phonotactically_valid": False,
                "suggestions":           lev_suggs,
                "suggestion_distances":  [round(d, 3) for d, _ in lev_kept],
                "method":                "levenshtein_phonotactic",
                "in_lexicon":            False,
                "gate_reached":          1,
                "confidence":            conf,
                # WHY it failed, in the phonology module's own words —
                # "Illegal characters: f — not in Khasi orthography". The
                # reader of a flagged word deserves the actual reason; the
                # panel could otherwise only say "no candidate passed the
                # admissibility filter", which describes our search rather
                # than their spelling. Costs nothing extra: validate() has
                # already run to reach this branch.
                "reason": _phonotactic_reason(word_lower),
            }

        # ── Gate 3: Spell checker ─────────────────────────────────────
        # Strategy: our confusion-weighted Levenshtein (which includes
        # normalised candidates and phoneme-aware distance) ALWAYS runs first.
        # The autocorrect Speller — when available — contributes additional
        # candidates that extend the pool, but never overrides our ranking.
        # This prevents the Speller's frequency-based scoring from promoting
        # morphologically valid but phonetically distant words above the
        # obvious correction.
        suggestions: list[str] = []
        method = "none"

        # Step 1 — our confusion-weighted Levenshtein (always runs first)
        # Returns (distance, word) pairs sorted by (dist, -freq_score).
        # Distance is the PRIMARY ranking signal — morphological validity
        # and frequency are secondary tie-breakers only.
        lev_pairs = _levenshtein_suggestions(word_lower, self._db, top_n=n * 4,
                                             **self._search_kwargs())
        # Phonotactic check is a heuristic; the lexicon is ground truth. If a
        # candidate is recorded in the lexicon as a real Khasi word, never drop
        # it just because the validator's rules don't cover it (place names,
        # loanwords, and reduplications often fail strict phonotactics).
        #
        # `is_attested`, not `is_known`: the question here is "did the lexicon
        # write this down", not "is this an acceptable standalone word". They
        # diverged when the phonotactic screen was added to `is_known`, and
        # using the strict one here dropped `shafon` from the suggestions for
        # `shafan` — a real word the validator happens to reject.
        lev_ok = [(d, s) for d, s in lev_pairs if self._offerable(s, word_lower)]

        # Step 2 — Speller extends the pool with additional candidates.
        # These get a default distance of max_dist (treated as distant matches)
        # so they never displace our close Levenshtein matches.
        speller_extra: list[tuple[float, str]] = []
        lev_words = {s for _, s in lev_ok}
        if self._speller and SPELLER_CANDIDATES_ENABLED:
            try:
                if hasattr(self._speller, "get_suggestions"):
                    raw = self._speller.get_suggestions(word_lower, n=n * 3)
                elif hasattr(self._speller, "get_candidates"):
                    pairs = self._speller.get_candidates(word_lower)
                    pairs.sort(reverse=True)
                    raw = [w for _, w in pairs[: n * 3]]
                else:
                    raw = []
                for s in raw:
                    if self._offerable(s, word_lower) and s not in lev_words:
                        d = _levenshtein(word_lower, s)
                        speller_extra.append((d, s))
            except Exception as e:
                print(f"[khasi-nlp] Speller.suggest error: {e}")

        # Step 2b — morphological generation.
        #
        # Everything above searches the lexicon, so it can only ever offer a
        # word the lexicon holds. The confidence vote does not work that way:
        # it accepts forms the grammar derives, lexicon or no lexicon. That
        # asymmetry means the checker can accept `pynïarap` and be unable to
        # propose it for `pynarap`. Measured over a 23,254-word target
        # vocabulary, half the unreachable targets were words it accepts.
        #
        # Candidates here are built from real glossed headwords by productive
        # affixation and by compounding, and each carries a half-edit penalty
        # so a genuine headword at the same distance still wins.
        gen_extra: list[tuple[float, str]] = []
        if MORPH_GENERATOR_ENABLED:
            try:
                from khasi_engine.generate import get_generator
                gen = get_generator(self._db, _levenshtein)
                known = {c for _, c in lev_ok} | {c for _, c in speller_extra}
                for dist, form, _route in gen.candidates(word_lower, limit=n * 4):
                    if form not in known and self._offerable(form, word_lower):
                        gen_extra.append((dist, form))
            except Exception as e:
                print(f"[khasi-nlp] morphological generator error: {e}")

        # Step 3 — merge and re-rank using (distance, -freq, -morph_bonus)
        # Distance always wins. Frequency and morphological validity only
        # break ties between candidates at the same phonemic distance.
        freq_data = getattr(self._speller, "nlp_data", {}) if self._speller else {}
        all_candidates: list[tuple[float, float, str]] = []
        seen_final: set[str] = set()
        for d, s in lev_ok + speller_extra + gen_extra:
            if s in seen_final:
                continue
            seen_final.add(s)
            freq  = freq_data.get(s, 0)
            bonus = _MORPHO_VALID_BONUS if self._morphologically_valid(s) else 0
            # Shared word-initial phonemes, weighted as in
            # _levenshtein_suggestions. That first pass already computes this
            # bonus, but the merge here used to drop it, so candidates tying
            # on distance AND frequency fell through to alphabetical order.
            # 'pyleng' is the case in point: 'pleng' and 'pylleng' both sit at
            # distance 1 with frequency 20, and 'pleng' won on the letter p.
            # Counting shared onset phonemes (p-y-l = 3 vs p = 1) puts the
            # real correction first.
            prefix = _shared_phoneme_prefix(word_lower, s) * _PREFIX_BONUS_WEIGHT
            # A form the lexicon records beats one the generator invented.
            attested = _ATTESTED_BONUS if self._db.is_known(s) else 0
            all_candidates.append((d, _first_letter_differs(word_lower, s),
                                   -(freq + bonus + prefix + attested), s))

        all_candidates.sort()
        suggestions = [s for _, _, _, s in all_candidates[:n]]
        suggestion_distances = [round(d, 3) for d, _, _, _ in all_candidates[:n]]
        method = "levenshtein_primary" if not speller_extra else "levenshtein_speller_combined"

        # Step 4 — FastText re-ranking when embedding model is loaded
        if self._hybrid and suggestions:
            try:
                _order = {c: i for i, c in enumerate(suggestions)}
                suggestions = self._rerank_hybrid(word_lower, suggestions)
                # Re-ranking permutes the list, so carry the distances with it
                # rather than leaving them silently misaligned.
                suggestion_distances = [suggestion_distances[_order[c]]
                                        for c in suggestions]
                method = "hybrid_fasttext"
            except Exception:
                pass

        # A split outranks every single-word candidate: it is a one-character
        # insertion, and the corpus evidence behind it is far stronger than
        # the edit distance behind the words it displaces. Inserted after the
        # re-rankers so neither can reorder or drop it.
        _split = self._split_suggestion(word_lower)
        if _split:
            suggestions = [_split] + [s for s in suggestions if s != _split]
            suggestion_distances = [1.0] + list(suggestion_distances)
            suggestions = suggestions[:n]
            suggestion_distances = suggestion_distances[:n]
            method = "runtogether_split"

        if not suggestions:
            method = "none"

        return {
            "is_known":           False,
            "morphologically_valid": False,
            "phonotactically_valid": True,
            "suggestions":        suggestions,
            "suggestion_distances": suggestion_distances,
            "method":             method,
            "in_lexicon":         self._db.is_known(word_lower),
            "gate_reached":       3,
            "confidence":         conf,
        }

    # _rank_by_morphology() was removed 2026-08-26: dead since the merge in
    # suggest() started scoring candidates inline. It had no callers and no
    # tests, and its scoring disagreed with the live path (it sorted by
    # frequency alone, ignoring distance), so anyone reading it for the
    # ranking rules would have been misled.

    def _rerank_hybrid(self, word: str, candidates: list[str]) -> list[str]:
        """
        Re-rank candidates using phonemic distance + semantic similarity
        (from the HF-hosted Word2Vec Space) + lexical frequency.

        Scoring formula
        ---------------
        score = phonemic_closeness * DIST_W
              + semantic_similarity * SEM_W
              + log1p(freq) * FREQ_W
              + morph_bonus

        Where phonemic_closeness = 1 / (1 + phonemic_distance) and
        semantic_similarity comes from one POST /score round-trip to the
        HF Space (one call per word — all candidates batched).

        If the Space is unreachable or returns garbage, we degrade
        gracefully to rule-only ranking instead of raising.
        """
        from . import w2v_client

        freq_data = getattr(self._speller, "nlp_data", {}) if self._speller else {}

        DIST_W  = 4.0   # phonemic closeness weight (dominant signal)
        SEM_W   = 3.0   # semantic similarity weight
        FREQ_W  = 0.1   # frequency weight (minor tie-breaker)

        # One batched HTTP call gives us sims for every candidate.
        try:
            sim_map = w2v_client.score(word, list(candidates))
        except (w2v_client.W2vNotConfigured, w2v_client.W2vUpstreamError) as exc:
            # Fall through to rule-only re-rank — better than blowing up
            # the spell-checker mid-request.
            print(f"[khasi-nlp] remote w2v score failed: {exc}")
            sim_map = {}

        scored: list[tuple[float, str]] = []
        for c in candidates:
            dist      = _levenshtein(word, c)
            closeness = 1.0 / (1.0 + dist)
            # Treat OOV / missing as 0 (sentinel -1.0 from w2v_client too).
            raw_sim = sim_map.get(c, 0.0)
            sim     = max(0.0, raw_sim)

            freq  = freq_data.get(c, 1)
            bonus = _MORPHO_VALID_BONUS if self._morphologically_valid(c) else 0

            score = (closeness * DIST_W) + (sim * SEM_W) + (math.log1p(freq) * FREQ_W) + bonus
            scored.append((score, c))

        scored.sort(reverse=True)
        return [c for _, c in scored]

    # ------------------------------------------------------------------
    # Sentence checking
    # ------------------------------------------------------------------

    def check_sentence(self, sentence: str) -> list[dict]:
        """
        Tokenise *sentence* and return corrections for words that fail
        the gate.

        Words the gate accepts are silently skipped — never flagged even if
        absent from the static lexicon. Offsets index *sentence* as given;
        callers that normalise (NFC) must pass the normalised text.

        Suggestions carry the capitalisation of the word they replace, so a
        sentence-initial `Shnng` is corrected to `Shnong`, not `shnong`.
        """
        results = []

        # This used to call self._speller.check_sentence() for its tokeniser,
        # which ran edit-distance-2 correction on every token (up to 18 s on
        # one sentence) and silently pre-filtered what reached the gate. It
        # was removed; see the README section "Sentence latency".
        #
        # One decision per distinct word: suggest() is a pure function of the
        # normalised word for a given checker, and long documents repeat
        # their unknown words many times.
        memo: dict[tuple, dict] = {}
        for match in _WORD_PATTERN.finditer(sentence):
            w    = match.group(0)
            # The capital is part of the key: a name word is accepted as
            # `United` and rejected as `united` (see _name_only_words).
            key  = (_tokens.canonical(w).lower(), w[:1].isupper())
            gate = memo.get(key)
            if gate is None:
                gate = memo[key] = self.suggest(w, n=5)
            # Flag gate=0 (not a word on its own), gate=1 (phonotactically
            # invalid) and gate=3 (unknown, with something to offer). Gate=2
            # (accepted) words are silently skipped.
            #
            # Gate 1 is reported even with nothing to offer; gate 3 is not.
            # The difference is what we know. Gate 1 means the string breaks
            # Khasi phonotactics — `sbngaifi` contains an `f`, which is not in
            # the orthography — so it is wrong whether or not we can fix it.
            # Gate 3 only means "not in the lexicon", which is also true of
            # every legitimate word we have never recorded; flagging those
            # with no suggestion turns a silence into a false alarm.
            #
            # A gate-3 rejection that carries a reason is reported with or
            # without a suggestion, for the same reason as gate 1: the reason
            # is something we know — "an English word" (maintainer ruling
            # 2026-10-01) — not merely "absent from the lexicon". `hospital`
            # has no Khasi neighbour to suggest and was passing silently.
            #
            # `suggestion` is None when nothing can be offered; callers that
            # rewrite text must leave the word alone rather than substitute
            # None.
            if gate["gate_reached"] in (0, 1) or (
                    gate["gate_reached"] == 3
                    and (gate["suggestions"] or gate.get("reason"))):
                suggs = [_tokens.match_case(w, x) for x in gate["suggestions"]]
                results.append({
                    "original":         w,
                    "suggestion":       suggs[0] if suggs else None,
                    "start":            match.start(),
                    "end":              match.end(),
                    "suggestions_top5": suggs,
                    "suggestion_distances": list(gate.get("suggestion_distances") or []),
                    "method":           gate["method"],
                    "morphologically_valid": gate["morphologically_valid"],
                    "phonotactically_valid": gate["phonotactically_valid"],
                    "gate_reached":     gate["gate_reached"],
                    "reason":           gate.get("reason"),
                })
        return results

    def autocorrect(self, word: str) -> str:
        """The best correction for *word*, in its capitalisation; the word
        itself when it is accepted or nothing can be offered."""
        gate = self.suggest(word, n=1)
        if gate["is_known"] or not gate["suggestions"]:
            return word
        return _tokens.match_case(word.strip(), gate["suggestions"][0])

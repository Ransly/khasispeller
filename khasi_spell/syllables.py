"""
syllables.py — syllables by rule, for a word the lexicon has not syllabified.

The lexicon's own syllables are curated per entry (rebuilt against War 2001)
and are always preferred; after them come syllables assembled from the
entries of a derived form's pieces (api._stitch_derived_phonology). Both
need the lexicon to know the word or every piece of it. `ïatreilang` 'work
together' is ïa- + treilang, and treilang (trei + lang) has no entry, so the
Phonology panel showed a letter count and nothing else. This module is the
last resort for such forms.

The conventions are read off the lexicon's own syllabifications rather than
taken from a general theory:

  * Two vowel letters share a syllable; a longer run splits after every
    second letter: jai·aw, ïa·i, ïa·ei.
  * One consonant between vowels opens the next syllable (ba·pher), unless
    the next vowel is ï, which opens a syllable of its own (jing·ïa).
  * Of two or more consonants, the first closes the syllable before it
    (pak·ri, kyn·thei, jing·sngew) when it can end a syllable at all. An
    aspirate or sh cannot, so the cluster opens the next one (ba·khraw).
  * lh and rh are one sound only at the start of a word; inside it they
    split (kyr·hin, jal·hung).
  * A prefix the morphology finds ends a syllable (pyn·i·did, ïa·trei), and
    so does a suffix that has a vowel.

Measured against the 7,955 single words the lexicon syllabifies, each parsed
as if it were unknown: the same syllables for 98.6%, the same number of
syllables for 99.9%. Most of the rest are words the lexicon itself treats
inconsistently (ïa·kmen beside ïak·ren, kyn·ew beside by·niej).

The pattern uses the lexicon's notation — a digraph is one C, a vowel letter
a V, and a w after a vowel is the offglide, V (law CVV); in iew the ie
counts once (riew CVV). Computed from the lexicon's syllables it matches the
lexicon's own pattern for 98.4% of those words. The rest are mostly long
vowels, which the spelling does not mark (War 2001 pp.49-54), and an i
before ñ that the lexicon writes as a glide in 40 words (raiñ CVCC) and as
a vowel in 153 (CVVC), the convention followed here.
"""
from __future__ import annotations

from khasi_engine import phonology as _phon

# Consonants that can close a syllable: the validator's permitted word
# finals, plus h (the glottal stop), w (the diphthong offglide) and the loan
# finals l and s. The validator only warns about l and s at the end of a
# word, and the lexicon's own syllables close on them inside one.
_EXTRA_CODAS = frozenset({"h", "w", "l", "s"})


def _inventory():
    """(vowels, digraphs longest first, codas, letters), read at call time.

    The phonology module rebinds its constants when the PostgreSQL loader
    installs the lexicon's phonology block, so they are not copied at import.
    """
    vowels = frozenset(_phon.VOWELS)
    digraphs = sorted(_phon.DIGRAPHS_AS_SINGLE, key=len, reverse=True)
    codas = (frozenset(_phon.ALLOWED_FINAL_CONSONANTS)
             | frozenset(_phon.ALLOWED_FINAL_DIGRAPHS)
             | frozenset(_phon.ALLOWED_FINAL_SPECIAL)
             | _EXTRA_CODAS)
    letters = frozenset(_phon.VALID_CHARS) - {"-", "'"}
    return vowels, digraphs, codas, letters


def _tokens(s: str, digraphs, word_start: bool = True) -> list[str]:
    out, i = [], 0
    while i < len(s):
        for d in digraphs:
            if s.startswith(d, i) and not (d in ("lh", "rh") and (i or not word_start)):
                out.append(d)
                i += len(d)
                break
        else:
            out.append(s[i])
            i += 1
    return out


def _split(stretch: str, vowels, digraphs, codas, word_start: bool) -> list[str] | None:
    """Syllables of one stretch with no morpheme boundary inside it."""
    runs: list[list] = []                     # [is_vowel, [tokens]]
    for t in _tokens(stretch, digraphs, word_start):
        v = t in vowels
        if runs and runs[-1][0] == v:
            runs[-1][1].append(t)
        else:
            runs.append([v, [t]])
    if not any(v for v, _ in runs):
        return None
    seq: list[tuple[bool, list[str]]] = []
    for v, ts in runs:
        if v:
            seq.extend((True, ts[i:i + 2]) for i in range(0, len(ts), 2))
        else:
            seq.append((False, ts))
    out, cur, seen = [], "", False
    for i, (v, ts) in enumerate(seq):
        if v:
            if seen and seq[i - 1][0]:        # two nuclei side by side
                out.append(cur)
                cur = ""
            cur += "".join(ts)
            seen = True
        elif not seen or i == len(seq) - 1:   # the first onset, the last coda
            cur += "".join(ts)
        elif len(ts) == 1:
            if seq[i + 1][1][0] == "ï":
                out.append(cur + ts[0])
                cur = ""
            else:
                out.append(cur)
                cur = ts[0]
        else:
            k = 1 if ts[0] in codas else 0
            out.append(cur + "".join(ts[:k]))
            cur = "".join(ts[k:])
    out.append(cur)
    return [s for s in out if s]


def syllabify(word: str, prefixes=(), suffix: str = "") -> list[str] | None:
    """*word*'s syllables by rule, or None when the rules cannot say.

    *prefixes* (outermost first) and *suffix* are the morphology's layers;
    each ends a syllable. None for a word with a hyphen, an apostrophe or a
    letter Khasi does not use, and for any stretch without a vowel.
    """
    vowels, digraphs, codas, letters = _inventory()
    w = (word or "").lower()
    if not w or not vowels or any(ch not in letters for ch in w):
        return None
    pieces, rest = [], w
    for p in prefixes or ():
        p = (p or "").strip("-").lower()
        if p and rest.startswith(p) and len(rest) > len(p):
            pieces.append(p)
            rest = rest[len(p):]
    sfx = (suffix or "").strip("-").lower()
    if (sfx and any(c in vowels for c in sfx)
            and rest.endswith(sfx) and len(rest) > len(sfx)):
        pieces += [rest[:-len(sfx)], sfx]
    else:
        pieces.append(rest)
    out: list[str] = []
    for n, piece in enumerate(pieces):
        syl = _split(piece, vowels, digraphs, codas, word_start=(n == 0))
        if not syl:
            return None
        out.extend(syl)
    if "".join(out) != w or not all(any(c in vowels for c in s) for s in out):
        return None
    return out


def pattern(syllables) -> str:
    """The CV pattern of *syllables*, dot-separated, in the lexicon's notation."""
    vowels, digraphs, _, _ = _inventory()
    parts = []
    for n, s in enumerate(syllables):
        t = _tokens(s, digraphs, word_start=(n == 0))
        cv = []
        for i, x in enumerate(t):
            prev = t[i - 1] if i else ""
            nxt = t[i + 1] if i + 1 < len(t) else ""
            if x in vowels:
                if x == "e" and prev == "i" and nxt == "w":
                    continue                  # iew: the ie counts once
                cv.append("V")
            elif x == "w" and prev in vowels and nxt not in vowels:
                cv.append("V")                # the offglide of a diphthong
            else:
                cv.append("C")
        parts.append("".join(cv))
    return ".".join(parts)

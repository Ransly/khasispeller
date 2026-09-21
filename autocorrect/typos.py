# Python 3 Spelling Corrector
#
# Copyright 2014 Jonas McCallum.
# Updated for Python 3, based on Peter Norvig's
# 2007 version: http://norvig.com/spell-correct.html
#
# Khasi digraph + infix support: digraphs and the -yn- infix are treated
# as atomic units so that edits never split them into invalid fragments.
"""
Word based methods and functions

Author: Jonas McCallum
https://github.com/foobarmus/autocorrect

Optimized by: Filip Sondej
https://github.com/fifimajster/autocorrect/

Khasi improvements: digraph-aware tokenisation, -yn- infix detection.
"""

from itertools import chain

from autocorrect.constants import (
    alphabets,
    khasi_digraphs,
    khasi_digraphs_sorted,
    KHASI_INFIX_YN,
)


def _tokenize_khasi(word: str) -> list[str]:
    """
    Split a Khasi word into a list of *phonemic tokens*, treating confirmed
    digraphs/trigraphs and the -yn- infix as single atomic units.

    Pass 1 — greedy left-to-right scan matches digraphs/trigraphs using the
    pre-sorted ``khasi_digraphs_sorted`` constant (longest match wins).

    Pass 2 — collapses a bare ['y','n'] sequence at position index 1-2
    (i.e. immediately after the first consonant) into a single 'yn' infix
    token, provided the word begins with the expected first character ('k').
    This keeps the infix atomic so edit operations never split it.

    Examples
    --------
    >>> _tokenize_khasi("khynmaw")
    ['kh', 'y', 'n', 'm', 'a', 'w']        # 'kh' digraph; no yn-infix here

    >>> _tokenize_khasi("kynhjat")
    ['k', 'yn', 'hj', 'a', 't']             # 'yn' infix collapsed

    >>> _tokenize_khasi("shnong")
    ['sh', 'n', 'o', 'ng']

    >>> _tokenize_khasi("thied")
    ['th', 'ie', 'd']
    """
    # --- Pass 1: digraph/trigraph tokenisation ---
    tokens: list[str] = []
    i = 0
    lower = word.lower()
    while i < len(lower):
        matched = False
        for dg in khasi_digraphs_sorted:
            end = i + len(dg)
            if lower[i:end] == dg:
                tokens.append(word[i:end])      # preserve original casing
                i = end
                matched = True
                break
        if not matched:
            tokens.append(word[i])
            i += 1

    # --- Pass 2: -yn- infix detection ---
    # The infix 'yn' appears after the first consonant in k-initial words.
    # If tokens[1] == 'y' and tokens[2] == 'n' and the word starts with 'k',
    # collapse the two single-char tokens into one 'yn' token.
    infix, first_char = KHASI_INFIX_YN
    if (
        len(tokens) >= 3
        and tokens[0].lower() == first_char
        and tokens[1].lower() == infix[0]      # 'y'
        and tokens[2].lower() == infix[1]      # 'n'
    ):
        tokens = [tokens[0], infix] + tokens[3:]

    return tokens


class Word:
    """
    Container for word-based edit-distance methods.

    For Khasi (lang='kh') the word is first tokenised into phonemic units
    (digraphs and the -yn- infix each count as one unit) before slicing,
    so that edit operations never break multi-character phonemes apart.
    For all other languages the original character-level behaviour is preserved.
    """

    __slots__ = ["slices", "word", "alphabet", "only_replacements", "lang", "_is_khasi"]

    def __init__(self, word: str, lang: str = "en", only_replacements: bool = False):
        """
        Generate slices to assist with typo definitions.

        For English (character-level):
            'the' => (('', 'the'), ('t', 'he'), ('th', 'e'), ('the', ''))

        For Khasi (token-level, digraphs / infix kept intact):
            'khun' => tokens ['kh','u','n']
                   => (([], ['kh','u','n']), (['kh'], ['u','n']), ...)
        """
        self.word = word
        self.lang = lang
        self.alphabet = alphabets[lang]
        self.only_replacements = only_replacements
        self._is_khasi = (lang == "kh")

        if self._is_khasi:
            tokens = _tokenize_khasi(word)
            self.slices = tuple(
                (tokens[:i], tokens[i:]) for i in range(len(tokens) + 1)
            )
        else:
            self.slices = tuple(
                (word[:i], word[i:]) for i in range(len(word) + 1)
            )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _join(self, left, right) -> str:
        """Reconstruct a string from two token lists or two character strings."""
        if self._is_khasi:
            return "".join(left) + "".join(right)
        return left + right

    def _alphabet_units(self):
        """
        Yield replacement units for the current language.

        For Khasi this includes single chars, digraphs, and the 'yn' infix
        so that a missing multi-character phoneme can be inserted or replaced
        in a single edit step.
        """
        yield from self.alphabet
        if self._is_khasi:
            yield from khasi_digraphs
            yield KHASI_INFIX_YN[0]     # 'yn'

    # ------------------------------------------------------------------
    # Edit operations
    # ------------------------------------------------------------------

    def _deletes(self):
        """Delete one token."""
        for a, b in self.slices[:-1]:
            if self._is_khasi:
                yield self._join(a, b[1:])
            else:
                yield a + b[1:]

    def _transposes(self):
        """Swap two adjacent tokens."""
        for a, b in self.slices[:-2]:
            if self._is_khasi:
                yield self._join(a, [b[1], b[0]] + list(b[2:]))
            else:
                yield a + b[1] + b[0] + b[2:]

    def _replaces(self):
        """Replace one token with every alphabet unit."""
        for a, b in self.slices[:-1]:
            for c in self._alphabet_units():
                if self._is_khasi:
                    yield self._join(a, [c] + list(b[1:]))
                else:
                    yield a + c + b[1:]

    def _inserts(self):
        """Insert one alphabet unit at every position."""
        for a, b in self.slices:
            for c in self._alphabet_units():
                if self._is_khasi:
                    yield self._join(a, [c] + list(b))
                else:
                    yield a + c + b

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def typos(self):
        """All strings one edit away from *word*."""
        if self.only_replacements:
            return chain(self._replaces())
        return chain(
            self._deletes(),
            self._transposes(),
            self._replaces(),
            self._inserts(),
        )

    def double_typos(self):
        """All strings two edits away from *word*."""
        return chain.from_iterable(
            Word(e1, lang=self.lang, only_replacements=self.only_replacements).typos()
            for e1 in self.typos()
        )

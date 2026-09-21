"""
khasi_spell — a standalone morphology-gated spellchecker for Khasi.

Khasi is prefix-dominant and productively derivational, so a finite word
list cannot enumerate its valid words: `jingbha`, `pynbha` and
`nonghikai` are all well formed whether or not anyone has listed them.
This checker therefore asks a morphological analyser whether a word is a
legal Khasi formation, and only words that fail that test enter
edit-distance correction.

    from khasi_spell import KhasiSpeller

    sp = KhasiSpeller()
    sp.is_correct("jingpynim")        # True  — derived, not listed
    sp.suggest("lyngdo")              # ['lyngdoh', ...]
    sp.correct_text("Ka shnng ka bha")

Extracted from the MorpSpeller / Khasi NLP Explorer platform. The parent
project is unmodified; this is an independent copy of the spelling path.
"""

from khasi_spell.speller import (
    Correction,
    GATE_ACCEPTED,
    GATE_CHECKED,
    GATE_INVALID_STANDALONE,
    GATE_PHONOTACTIC,
    KhasiSpeller,
    TextResult,
    WordResult,
)

__all__ = [
    "KhasiSpeller",
    "WordResult",
    "TextResult",
    "Correction",
    "GATE_INVALID_STANDALONE",
    "GATE_PHONOTACTIC",
    "GATE_ACCEPTED",
    "GATE_CHECKED",
]
__version__ = "1.0.0"

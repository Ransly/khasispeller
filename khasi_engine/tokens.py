"""
tokens.py — one definition of a Khasi word token, shared by every component.

Five components used to tokenise text with five slightly different regular
expressions: the engine's sentence checker, the facade's hyphen and variant
passes, the context re-ranker, the real-word detector and the web page. They
disagreed in two ways, and both corrupted correct text:

  * Only ï and ñ were listed as letters, although the lexicon's own
    character set also has á é í ó ú ý. A dictionary word such as `sngewrém`
    was cut into `sngewr` and `m`, and each fragment was "corrected".
  * The typographic apostrophe (’), which phones and word processors insert
    automatically, was not a word character. `nga’m` became `nga` + `m`, and
    `m` was "corrected" to `ma`.

On top of that, the engine joined letters across scan debris (`i^p`) and the
facade did not, so the two produced overlapping corrections for one word.

Offsets are always into the text exactly as given: the apostrophe variants
are normalised only when a word is LOOKED UP, never in the text itself.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Iterable, Iterator

# Latin letters, including every precomposed Latin-1 letter: the Khasi
# diacritics ï ñ á é í ó ú ý and their capitals, and the accented letters of
# names and loans. A foreign accented word is now read whole and rejected by
# the phonotactic gate with a reason, instead of being split into fragments
# that each get "corrected".
LETTERS = "A-Za-zÀ-ÖØ-öø-ÿ"

# The straight apostrophe, the typographic right single quote and the
# modifier-letter apostrophe. All three mark the glottal stop / elision.
APOSTROPHES = "'’ʼ"

# Characters that are never part of a Khasi word but appear INSIDE one in
# scanned or scraped text — the caret in `i^p` for `ïap`, and its relatives.
# They bind letters together so a damaged word is seen whole. Measured before
# they were added: no lexicon headword has one of these joining two letters.
NOISE_CHARS = "^~«»•|\\{}[]"

# A word: letters, optionally joined by an internal apostrophe, hyphen or
# scan-noise character. A joiner at either edge is not part of the word.
WORD_PATTERN = re.compile(
    rf"[{LETTERS}]+(?:[{re.escape(APOSTROPHES + NOISE_CHARS)}\-][{LETTERS}]+)*"
)

_APOSTROPHE_MAP = str.maketrans({"’": "'", "ʼ": "'"})

# Opening punctuation a sentence may start with: quotes, brackets, bullets
# and dashes.
_OPENERS = "\"'‘“«([{—–-•*·>"

# A sentence ends at a terminator followed by optional closing quotes or
# brackets. Mirrors scripts/build_ngrams.py, which splits sentences on
# `(?<=[.!?])\s+` and on line breaks.
_TERMINATOR = re.compile(r"[.!?…][\"'’”»)\]]*$")


def nfc(text: str) -> str:
    """Compose Unicode: ï and ñ each have a precomposed and a decomposed form."""
    return unicodedata.normalize("NFC", text or "")


def canonical(word: str) -> str:
    """The spelling to LOOK UP: NFC, with apostrophe variants made ASCII.

    Length-preserving for NFC input, so it never moves an offset.
    """
    return nfc(word).translate(_APOSTROPHE_MAP)


# The corpus behind the frequency list and the n-gram model holds ZERO ï, ñ
# and apostrophes across 6.67M tokens (and almost no acute accents): they
# were stripped in preparation or never typed. A word must be looked up in
# that model as the corpus writes it, or every diacritic and every elision is
# "unseen" and scores at the floor.
_CORPUS_FOLD = str.maketrans({
    "ï": "i", "ñ": "n", "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ý": "y",
    "Ï": "I", "Ñ": "N", "Á": "A", "É": "E", "Í": "I", "Ó": "O", "Ú": "U", "Ý": "Y",
})


def corpus_form(word: str) -> str:
    """*word* spelled as the corpus writes it: no diacritics, no apostrophe.

    `ïaid` -> `iaid`, `nga'm` -> `ngam`. For language-model and corpus
    lookups only; never shown to a reader.
    """
    return canonical(word).translate(_CORPUS_FOLD).replace("'", "")


def tokens(text: str) -> Iterator[re.Match]:
    """Word tokens of *text*, as match objects carrying their offsets."""
    return WORD_PATTERN.finditer(text or "")


def match_case(original: str, replacement: str) -> str:
    """Give *replacement* the capitalisation of *original*.

    The engine's vocabulary is lower case, so a correction applied to
    `Shnng` at the start of a sentence used to come back as `shnong`.
    All-capitals (two or more letters) stays all-capitals; a leading capital
    stays a leading capital; anything else is left as the lexicon spells it.
    """
    if not original or not replacement:
        return replacement
    letters = [c for c in original if c.isalpha()]
    if len(letters) > 1 and all(c.isupper() for c in letters):
        return replacement.upper()
    if letters and letters[0].isupper():
        for i, ch in enumerate(replacement):
            if ch.isalpha():
                return replacement[:i] + ch.upper() + replacement[i + 1:]
    return replacement


def sentence_index(text: str, spans: Iterable[tuple[int, int]]) -> list[int]:
    """Sentence number of each span, in order.

    A new sentence starts after a line break, or after a terminator
    (`. ! ?` and optional closing quotes) followed by whitespace — the same
    boundaries the n-gram model was built with. Used so a word's context
    never reaches across a sentence boundary, where the model has only ever
    seen the sentence-start marker.
    """
    out: list[int] = []
    sent = 0
    prev_end = None
    for start, end in spans:
        if prev_end is not None:
            gap = text[prev_end:start]
            if "\n" in gap or _ends_sentence(gap):
                sent += 1
        out.append(sent)
        prev_end = end
    return out


def _ends_sentence(gap: str) -> bool:
    """True when the text between two words closes a sentence.

    Whitespace or opening punctuation must separate the terminator from the
    next word; `3.5` and `etc.Ka` are not boundaries. Opening quotes of the
    next sentence (`. "Ka`) are stripped before the terminator is tested.
    """
    head = gap.rstrip(_OPENERS + " \t\r\n")
    if head == gap:
        return False
    return bool(_TERMINATOR.search(head))


def starts_sentence(text: str, start: int) -> bool:
    """True when the token at *start* opens a sentence, a line or the text.

    A capital there carries no information about whether the word is a
    name. Counted as openers: the start of the text or of a line (headings,
    list items, verse), and a position after `. ! ? :`. Opening quotes,
    brackets, bullets and dashes may intervene, so `- Shnng`, `"Shnng` and
    `ong: Shnng` all count.
    """
    before = (text or "")[:start]
    line = before.rsplit("\n", 1)[-1]
    head = line.rstrip(_OPENERS + " \t\r")
    if not head.strip():
        return True
    if head == line:                        # glued to the previous character
        return False
    return head.endswith(":") or bool(_TERMINATOR.search(head))

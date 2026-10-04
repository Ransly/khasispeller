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
#
# Full-width Latin letters (\uff53\uff48\uff4e\uff4f\uff4e\uff47), from East Asian input methods and some
# PDFs, are letters too, looked up as ASCII; a word written in them used to
# be skipped unchecked.
LETTERS = "A-Za-zÀ-ÖØ-öø-ÿ\uff21-\uff3a\uff41-\uff5a"

# The straight apostrophe, the typographic right single quote and the
# modifier-letter apostrophe. All three mark the glottal stop / elision.
APOSTROPHES = "'’ʼ"

# Characters that are never part of a Khasi word but appear INSIDE one in
# scanned or scraped text — the caret in `i^p` for `ïap`, and its relatives.
# They bind letters together so a damaged word is seen whole. Measured before
# they were added: no lexicon headword has one of these joining two letters.
NOISE_CHARS = "^~«»•|\\{}[]"

# Invisible characters found inside words copied from PDFs and web pages: the
# soft hyphen, zero-width space, zero-width non-joiner and joiner, word joiner
# and byte-order mark. They split `shn\u00adong` into `shn` + `ong`, and each piece
# was "corrected". A word is now read through them, and they are dropped when
# it is looked up.
INVISIBLES = "\u00ad\u200b\u200c\u200d\u2060\ufeff"

# Cyrillic and Greek letters drawn like Latin ones. One inside a Latin word
# (`shn\u043eng` with a Cyrillic \u043e) leaves it looking right and matching nothing,
# and it was split at that letter. They belong to a word only beside Latin
# letters, so Russian or Greek text stays outside the checker, and they are
# never mapped silently: the spell checker reports them (method
# "lookalike_letters").
_LOOKALIKE_TO_LATIN = {
    # Cyrillic
    "\u0430": "a", "\u0435": "e", "\u043e": "o", "\u0440": "p", "\u0441": "c", "\u0443": "y", "\u0445": "x",
    "\u0456": "i", "\u0458": "j", "\u0455": "s", "\u0501": "d", "\u04bb": "h", "\u04cf": "l",
    "\u0410": "A", "\u0412": "B", "\u0415": "E", "\u041a": "K", "\u041c": "M", "\u041d": "H", "\u041e": "O",
    "\u0420": "P", "\u0421": "C", "\u0422": "T", "\u0425": "X", "\u0406": "I", "\u0408": "J", "\u0405": "S",
    # Greek
    "\u03bf": "o", "\u03b1": "a", "\u03b9": "i", "\u03c1": "p", "\u03c5": "u", "\u03bd": "v", "\u03ba": "k",
    "\u039f": "O", "\u0391": "A", "\u0392": "B", "\u0395": "E", "\u0397": "H", "\u0399": "I", "\u039a": "K",
    "\u039c": "M", "\u039d": "N", "\u03a1": "P", "\u03a4": "T", "\u03a5": "Y", "\u03a7": "X", "\u0396": "Z",
}
LOOKALIKES = "".join(_LOOKALIKE_TO_LATIN)

# A run of letters: Latin letters with look-alikes among them. At most three
# look-alikes may lead before a Latin letter, which keeps the scan linear on
# a long stretch of Cyrillic text.
_RUN = rf"[{LOOKALIKES}]{{0,3}}[{LETTERS}][{LETTERS}{LOOKALIKES}]*"

# A word: runs of letters, optionally joined by an internal apostrophe,
# hyphen, scan-noise character or invisible character. A joiner at either
# edge is not part of the word.
WORD_PATTERN = re.compile(
    rf"{_RUN}(?:[{re.escape(APOSTROPHES + NOISE_CHARS + INVISIBLES)}\-]{_RUN})*"
)

_APOSTROPHE_MAP = str.maketrans({"’": "'", "ʼ": "'"})
# Looked up as typed, less what cannot be seen: apostrophes made ASCII,
# full-width letters made ASCII, invisible characters dropped.
_LOOKUP_MAP = str.maketrans({
    "’": "'", "ʼ": "'",
    **{chr(c): chr(c - 0xFEE0) for c in range(0xFF21, 0xFF3B)},
    **{chr(c): chr(c - 0xFEE0) for c in range(0xFF41, 0xFF5B)},
    **{ch: None for ch in INVISIBLES},
})
_LATIN_MAP = str.maketrans(_LOOKALIKE_TO_LATIN)

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
    """The spelling to LOOK UP: NFC, apostrophe variants and full-width
    letters made ASCII, invisible characters dropped.

    Used only to look words up — offsets always come from the text as given,
    which matters because dropping an invisible character shortens the word.
    """
    return nfc(word).translate(_LOOKUP_MAP)


def latinise(word: str) -> str:
    """*word* with Cyrillic and Greek look-alike letters made Latin.

    `shn\u043eng` (Cyrillic \u043e) -> `shnong`. For reporting and suggesting the Latin
    spelling; canonical() deliberately does not do this, so a word with a
    foreign letter is flagged rather than silently accepted.
    """
    return (word or "").translate(_LATIN_MAP)


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


# Web and email addresses, and letters fused to a number, are not words. In
# 1,000 news lines 41 flags were pieces of addresses (gmail, com, www, mpsc)
# and 84 were units and ordinals (1st, 20th, 79.2mm, 5:30pm).
#
# Each candidate run is found once by a pattern with nothing required after
# it, then tested with string operations. Written as single patterns that
# demand a dot, an @ or a digit at the end, these backtracked across long
# runs: 100 s for 48,000 characters of ka-ka-ka.
_URL_SCHEME = re.compile(r"(?:https?://|www\.)[^\s<>\"']+", re.IGNORECASE)
_CHUNK = re.compile(r"[^\s<>\"'()\[\]{}]+")
_TLDS = frozenset({"com", "org", "net", "in", "gov", "edu", "info", "nic",
                   "co", "io", "me", "uk", "biz", "tv"})
_LABEL = re.compile(r"[\w-]+")
_ALNUM = re.compile(rf"[{LETTERS}0-9]+")
_EDGE = ".,;:!?"


def _address_span(chunk: str):
    """(left, right) trim of *chunk* when it is an email or a bare domain."""
    left = len(chunk) - len(chunk.lstrip(_EDGE))
    core = chunk.strip(_EDGE)
    if not core:
        return None
    if "@" in core:
        user, _, host = core.partition("@")
        if user and "." in host and all(_LABEL.fullmatch(x) for x in host.split(".")):
            return left, left + len(core)
        return None
    host = core.split("/", 1)[0]
    labels = host.split(".")
    if (len(labels) >= 2 and labels[-1].lower() in _TLDS
            and all(_LABEL.fullmatch(x) for x in labels)):
        return left, left + len(core)
    return None


def unchecked_spans(text: str) -> tuple:
    """(skip, digit_words) for *text*.

    `skip` holds the (start, end) of web and email addresses and of letters
    fused to a number: an ordinal or unit (12th, 79.2mm, 5:30pm), a code
    (covid19) or a registration plate (ML04A — capitals, so never a typo).
    `digit_words` holds (start, end, word) for lower- or mixed-case letters
    with digits strictly inside, `shn0ng`, which may be a digit typed for a
    letter; the analyser decides.
    """
    text = text or ""
    skip = [m.span() for m in _URL_SCHEME.finditer(text)]
    for m in _CHUNK.finditer(text):
        if "." in m.group(0) or "@" in m.group(0):
            span = _address_span(m.group(0))
            if span:
                skip.append((m.start() + span[0], m.start() + span[1]))
    digit_words = []
    for m in _ALNUM.finditer(text):
        s = m.group(0)
        if not any(ch.isdigit() for ch in s) or not any(ch.isalpha() for ch in s):
            continue                                   # a word, or a plain number
        inside = any(a <= m.start() < b for a, b in skip)
        if s[0].isalpha() and s[-1].isalpha() and not s.isupper() and not inside:
            digit_words.append((m.start(), m.end(), s))
        else:
            skip.append(m.span())
    return skip, digit_words


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

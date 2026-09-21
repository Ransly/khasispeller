"""
foreign.py — tokens the checker should not pretend to correct.

Running Khasi text is full of words that are not Khasi: place names,
personal names, party acronyms, English borrowings left untranslated. The
gate is right that they are not Khasi words, but being right is not the
same as being useful. On 200 untouched corpus sentences the checker flagged
8.23% of all tokens, and the suggestions read like this:

    East      -> pat        Cherra   -> heriap
    Ardent    -> armet      resign   -> arsien
    Congress  -> ?          Meghalaya-> ?

Every one is noise. An editor that underlines a seventh of a news article,
including every place name in it, is one people switch off — and a
spellchecker that is switched off has 0% accuracy.

What is suppressed
------------------
Three signals. The first two are self-contained; the third consults the
project's own proper-noun lists (see `scripts/build_gazetteer.py`):

  * **a known name** — `Sohra`, `Shillong`, `Lyndem`, `Mukul`. 6,609 tokens
    compiled from the villages / institutions / politicians / parties /
    state lists. This is positive recognition rather than a guess, so it
    works wherever the word appears — including sentence-initially, where
    capitalisation carries no information and the other rules are blind.
    It accounts for 9.1% of the flags that survive the capitalisation rule
    and suppresses **none** of the 558 frozen benchmark errors.

    The list is consulted only for words the gate has *already* rejected,
    which makes its 312 collisions with ordinary Khasi words (`bah`, `bat`,
    `blei`, `dawa`) inert: those are accepted by the lexicon first and never
    reach it.


  * **capitalised mid-sentence** — `Meghalaya`, `Cherra`, `Conrad`. A
    capital that is not sentence-initial is the orthographic marker for a
    proper noun. 42% of all flags.
  * **acronyms** — `MDA`, `AIS`, `HS`, `NDA`. All-caps, length > 1. 15%.

Together they account for 57% of the noise, and on the 558 errors in the
two frozen benchmarks they suppress **none**.

The honest limitation
---------------------
That 0% is partly an artefact: every benchmark error was injected into a
lowercased word, so no capitalised misspelling was ever there to miss.
Suppression does create a genuine blind spot — 12.6% of running text is
mid-sentence capitalised and 1.7% is acronyms, so **14.3% of tokens are no
longer checked**.

What makes that acceptable is not the benchmark but what the checker could
do with those tokens if it kept them. There is no Khasi proper-noun
lexicon, so `Meghlaya` and `Meghalaya` are equally absent from it and
equally unrankable; of the 414 capitalised tokens in the sample only 114
were flagged, and those 114 produced suggestions like `East -> pat`. The
capability given up is one the checker never actually had. Adding a
gazetteer of Khasi place and personal names would be the principled fix,
and would let this suppression be narrowed to "capitalised *and* unknown to
the gazetteer".

Nothing is deleted. Suppressed tokens are returned on `TextResult.skipped`
so a caller can surface them if it wants; they are simply kept out of
`corrections`, which is what gets auto-applied.

Not used: correcting misspelled names
-------------------------------------
The gazetteer makes `Meghlaya -> Meghalaya` look easy, and it was measured
before being built. It does not work. Injecting one edit into 300 gazetteer
names, nearest-match recovered 60.3% of them — but run over the genuine
capitalised tokens in 200 corpus sentences, **18.2% of real names sat within
one edit of a different gazetteer entry** and would have been "corrected"
into it:

    Light -> night        Rangbah -> nangbah      Hindu -> hindi
    Bill  -> hill         Shynrang -> synrang     Kumar -> khmar

`Rangbah` and `Shynrang` are ordinary Khasi words; `Light` and `Hindu` are
simply other names. Corrupting one real name in five to fix three in five
misspelled ones is a bad trade, so the gazetteer is used for recognition
only, never as a correction source.

Not used: an English word list
------------------------------
A third signal, "is this an English word", was measured and is the single
strongest — 55% of flags on its own, 23 points on top of the two above,
taking total suppression from 57% to 80% and the flag rate from 8.23% to
1.64% of tokens. It is not implemented here because it needs a bundled
dictionary. Measured against SCOWL via /usr/share/dict/american-english
(72,735 entries after removing the 867 that are also Khasi words) it would
have hidden 2 of the 558 benchmark errors — `ills` and `katie`, typos of
Khasi words that happen to spell English ones. Coverage scales with list
size (len<=4: 3,974 words, 32% of flags; full list: 55%), so there is no
cheap abridged version. Bundling it is a licensing and size decision for
the project to make, not one to slip in.
"""
from __future__ import annotations

import re
from typing import Optional

# A capital opening a sentence carries no information about word class, so
# the rule needs to know where sentences begin. Anything after one of these
# (plus optional closing quotes/brackets and whitespace) starts a sentence.
_SENTENCE_END = re.compile(r"[.!?][\"')\]]*\s+$")

# Minimum length for an all-caps token to read as an acronym rather than a
# shouted word or a stray initial.
MIN_ACRONYM_LENGTH = 2

CAPITALISED = "capitalised"
ACRONYM = "acronym"
KNOWN_NAME = "name"

_GAZETTEER: Optional[frozenset] = None


def gazetteer() -> frozenset:
    """
    Proper-noun tokens, loaded once from data/gazetteer.json.

    Returns an empty set when the file has not been built, so every caller
    degrades to the capitalisation rules alone rather than failing.
    """
    global _GAZETTEER
    if _GAZETTEER is None:
        import json
        from pathlib import Path

        path = Path(__file__).resolve().parent.parent / "data" / "gazetteer.json"
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            _GAZETTEER = frozenset(data.get("tokens") or ())
        except Exception:
            _GAZETTEER = frozenset()
    return _GAZETTEER


def is_sentence_initial(text: str, start: int) -> bool:
    """True when the token at *start* opens a sentence (or the text)."""
    before = text[:start]
    return not before.strip() or bool(_SENTENCE_END.search(before))


def is_proper_case(token: str, sentence_initial: bool) -> bool:
    """
    Does *token*'s capitalisation mark it as a name?

    True for a capital that is not sentence-initial, and for an acronym.
    Deliberately **case only** — it does not consult the gazetteer, because
    that list collides with 312 ordinary Khasi words (`bah`, `bat`, `blei`,
    `dawa`) and suppressing those would be wrong here.

    Used to decide whether a diacritic alternative should be offered. That
    rewrite is a claim about Khasi orthography and says nothing about a name
    that merely looks Khasi: `Spain` is a country, while `spain` is a Khasi
    word glossed "bandage; to swathe" whose standard spelling carries the
    tilde. The corpus shows case is a reliable discriminator — `Hussain` is
    capitalised in 53 of 53 uses and `Gohain` in 32 of 32, while `spain`
    splits 34 capitalised against 8 lowercase.
    """
    if not token:
        return False
    if token.isupper() and len(token) >= MIN_ACRONYM_LENGTH:
        return True
    return bool(token[:1].isupper()) and not sentence_initial


def classify(token: str, sentence_initial: bool) -> Optional[str]:
    """
    Why *token* should not be corrected, or None to check it normally.

    Returns `ACRONYM`, `CAPITALISED`, or None.
    """
    if not token:
        return None
    # Positive recognition first: a name is a name wherever it stands, and
    # this is the only rule that works sentence-initially.
    if token.lower() in gazetteer():
        return KNOWN_NAME
    if token.isupper() and len(token) >= MIN_ACRONYM_LENGTH:
        return ACRONYM
    if token[:1].isupper() and not sentence_initial:
        return CAPITALISED
    return None

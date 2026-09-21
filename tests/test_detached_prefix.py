"""
A bound prefix written detached from its root.

`jing ia mareh` is `jingïamareh` typed with two stray spaces. The rule that
rejoins these matched a bound prefix plus ONE word, so it stopped at
`jing ia` and offered `jingïa` — a prefix chain with nothing on it, which is
not a word either. The reciprocal sitting between the prefix and the root is
the common case, because `ia-` is itself a prefix.

What keeps this narrow is the BOUND prefix. Only `jing-` and `pyn-` qualify
(they gloss themselves as prefixes in the lexicon); `sngew`, `nong` and `ïa`
are ordinary words, and a rule that joined those would have rewritten 143
lexicon phrases. After `jing`/`pyn`, a following `ia` cannot be the
preposition.

Probe words only — no lexicon content.
"""
import pytest

from khasi_spell import KhasiSpeller


@pytest.fixture(scope="module")
def sp():
    return KhasiSpeller(eager=True)


def _fix(sp, text):
    doc = sp.check_text(text).to_dict()
    return {c["original"]: c["suggestion"] for c in doc.get("corrections", [])}


# The reciprocal detached between prefix and root — the reported case.
@pytest.mark.parametrize("text,original,fixed", [
    ("jing ia mareh",  "jing ia mareh",  "jingïamareh"),
    ("pyn ia mareh",   "pyn ia mareh",   "pynïamareh"),
    ("jing ia lehkai", "jing ia lehkai", "jingïalehkai"),
    ("jing ia thuh",   "jing ia thuh",   "jingïathuh"),
])
def test_detached_reciprocal_is_absorbed(sp, text, original, fixed):
    assert _fix(sp, text).get(original) == fixed


def test_join_restores_the_diacritic(sp):
    """`ia` becomes `ïa` when it joins — the lexicon writes 332 headwords
    with `ïa-` and none with plain `ia-`. Offering `jingiamareh` and leaving
    the variants layer to fix it afterwards is two steps for one slip."""
    assert "ï" in _fix(sp, "jing ia mareh")["jing ia mareh"]


# The two-token form must keep working.
@pytest.mark.parametrize("text,original,fixed", [
    ("jing ialehkai", "jing ialehkai", "jingïalehkai"),
    ("jing mareh",    "jing mareh",    "jingmareh"),
    ("pyn shai",      "pyn shai",      "pynshai"),
])
def test_two_token_join_still_works(sp, text, original, fixed):
    assert _fix(sp, text).get(original) == fixed


def test_root_tagged_other_is_still_joined(sp):
    """WEAK — guards against reaching for a part-of-speech filter here.

    `rap` "to help" is tagged `other` in the lexicon, so `_is_content_word`
    is False for it; generate.py records the same unreliability. A POS gate
    would drop this case.
    """
    assert _fix(sp, "jing rap").get("jing rap") == "jingrap"


# `ia` + a noun-class clitic is preposition + article, not prefix + root.
@pytest.mark.parametrize("text", [
    "ka jing ia ka",   # `jing` the noun, then `ia ka`
    "pyn ia u",
    "jing ia ki",
    "jing ia i",
])
def test_preposition_plus_clitic_is_left_alone(sp, text):
    for original, fixed in _fix(sp, text).items():
        assert " " not in original or "ï" not in fixed, (
            f"{text!r} wrongly joined to {fixed!r}")


# Prefixes that are also ordinary words are never joined.
@pytest.mark.parametrize("text", ["ia ka briew", "sngew pang", "nong shong"])
def test_unbound_prefixes_are_never_joined(sp, text):
    assert _fix(sp, text) == {} or all(
        " " not in o for o in _fix(sp, text))

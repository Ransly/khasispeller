"""Syllables by rule (khasi_spell/syllables.py), one test per convention."""
import pytest

from khasi_spell.syllables import pattern, syllabify


@pytest.mark.parametrize("word, prefixes, expected", [
    ("bapher", [], ["ba", "pher"]),           # one consonant opens the next syllable
    ("jingïa", [], ["jing", "ïa"]),           # ...unless ï follows
    ("pakri", [], ["pak", "ri"]),             # of two, the first closes the syllable
    ("kynthei", [], ["kyn", "thei"]),
    ("jingsngew", [], ["jing", "sngew"]),
    ("bakhraw", [], ["ba", "khraw"]),         # an aspirate cannot close one
    ("kyrhin", [], ["kyr", "hin"]),           # rh splits inside a word
    ("jaiaw", [], ["jai", "aw"]),             # two vowel letters to a syllable
    ("ïai", [], ["ïa", "i"]),
    ("pynidid", ["pyn-"], ["pyn", "i", "did"]),   # a prefix ends a syllable
    ("ïatreilang", ["ïa-"], ["ïa", "trei", "lang"]),
])
def test_syllabify(word, prefixes, expected):
    assert syllabify(word, prefixes) == expected


def test_a_suffix_without_a_vowel_is_not_a_syllable():
    assert syllabify("kam", suffix="-m") == ["kam"]


@pytest.mark.parametrize("word", ["", "bdr", "jiap-jiap", "'tikmie", "cafe"])
def test_no_answer_rather_than_a_guess(word):
    assert syllabify(word) is None


@pytest.mark.parametrize("syllables, expected", [
    (["trei"], "CCVV"),
    (["lang"], "CVC"),                        # ng is one C
    (["law"], "CVV"),                         # w after a vowel is the offglide
    (["riew"], "CVV"),                        # in iew the ie counts once
    (["ïa", "trei", "lang"], "VV.CCVV.CVC"),
])
def test_pattern(syllables, expected):
    assert pattern(syllables) == expected

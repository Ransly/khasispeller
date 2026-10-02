"""
The Khasi alphabet, as the maintainer gave it on 2026-10-01: 7 vowels
(a e i ï o u y) and 16 consonants (b d g h j k l m n ng ñ p r s t w), 23
letters, and no c, f, q, v, x or z. Also the same day's rulings on how a
word may end: in j, h and w, but never in y.

The lexicon's phonology block used to say 27 (`consonant_total`), list 24
(`consonants`) and chart 27, `c` among them; the vendored autocorrect
alphabet carried c f q v x z "for robustness", and the G2P read `c` as a
letter. The tests in the first half need only the code; the rest read the
phonology block, from the JSON file or from PostgreSQL, whichever is
configured.
"""
import os

import pytest

NOT_KHASI = set("cfqvxz")

VOWELS = ["a", "e", "i", "ï", "o", "u", "y"]

CONSONANTS = ["b", "d", "g", "h", "j", "k", "l", "m", "n", "ng", "ñ",
              "p", "r", "s", "t", "w"]

# A consonant letter + h: written with two letters, not letters themselves.
PLUS_H = {"sh", "ph", "bh", "th", "dh", "kh", "jh", "lh", "rh"}


def _foreign(symbols) -> list:
    return [s for s in symbols if NOT_KHASI & set(str(s).lower())]


# ----------------------------------------------------------------------
# Code: no alphabet the checker edits or reads with holds c f q v x z
# ----------------------------------------------------------------------

def test_vendored_autocorrect_alphabet_is_the_khasi_alphabet():
    from autocorrect.constants import alphabets
    letters = alphabets["kh"]
    assert not _foreign(letters)
    # ng is the digraph n + g, so the single letters are the 23 less one.
    assert set(letters.lower()) == set("".join(VOWELS + CONSONANTS))


def test_realword_edit_alphabet_has_no_foreign_letters():
    from khasi_spell import realword
    assert not _foreign(realword._ALPHABET)


def test_g2p_does_not_read_c_as_a_khasi_letter():
    from khasi_engine import g2p
    assert "c" not in g2p._CONS_SINGLE
    # The palatal stop is still produced where Khasi has it: final -it.
    assert g2p.to_ipa("buit") == "/buic/"


# ----------------------------------------------------------------------
# Data: the lexicon's phonology block
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def phon():
    # This module is in conftest's NO_LEXICON_NEEDED so the code checks above
    # run on a fresh clone; the data checks skip themselves instead.
    from khasi_engine import paths
    if not (os.environ.get("DATABASE_URL") or "").strip() \
            and not paths.default_db_path().is_file():
        pytest.skip("lexicon not available (see LICENSE-DATA)")
    # Constructing KhasiDB injects a PostgreSQL phonology block into the
    # phonology module, so the display helpers below read the same data.
    from khasi_engine.database import KhasiDB
    return KhasiDB().phonology


def test_alphabet_is_the_maintainers_list(phon):
    alphabet = phon["alphabet"]
    assert alphabet["vowels"] == VOWELS
    assert alphabet["consonants"] == CONSONANTS
    assert alphabet["letter_total"] == 23
    assert set(alphabet["not_letters"]) == NOT_KHASI


def test_sixteen_consonants(phon):
    assert phon["consonants"] == CONSONANTS
    assert phon["consonant_total"] == len(phon["consonants"]) == 16


def test_chart_holds_exactly_the_sixteen(phon):
    symbols = [it["symbol"] for group in phon["consonant_chart"].values()
               for it in group]
    assert len(symbols) == 16
    assert set(symbols) == set(CONSONANTS)
    for gone in ("c", "'", "y"):
        assert gone not in symbols
    assert not PLUS_H & set(symbols)


def test_two_letter_spellings_are_a_consonant_plus_h(phon):
    rows = phon["consonant_plus_h"]
    assert {r["symbol"] for r in rows} == PLUS_H
    for r in rows:
        assert r["letters"] == [r["symbol"][0], "h"]
        assert r["letters"][0] in CONSONANTS
        assert r["symbol"] not in CONSONANTS
        assert r["final"] is False
        assert r["aspirated"] is (r["symbol"] != "sh")


def test_no_foreign_letter_in_any_inventory(phon):
    for key in ("consonants", "vowels_simple", "diphthongs", "triphthongs",
                "valid_initial_clusters", "digraphs_as_single", "aspirates",
                "forbidden_final", "allowed_final_consonants",
                "allowed_final_digraphs", "allowed_final_special",
                "allowed_geminates", "valid_chars"):
        assert not _foreign(phon.get(key, [])), key
    chart = [it["symbol"] for g in phon["consonant_chart"].values() for it in g]
    assert not _foreign(chart)
    assert not _foreign(phon["alphabet"]["vowels"] + phon["alphabet"]["consonants"])


def test_validator_rejects_foreign_letters(phon):
    from khasi_engine import phonology
    assert not NOT_KHASI & set(phonology.VALID_CHARS)
    for word in ("cow", "fan", "qat", "van", "xam", "zan"):
        assert not phonology.validate(word)["pass"], word


def test_display_reports_sixteen(phon):
    from khasi_engine import phonology
    inv = phonology.get_consonant_inventory()
    assert inv["total"] == 16
    assert len(inv["all_consonants"]) == 16
    assert inv["alphabet"]["vowels"] == VOWELS
    assert {r["symbol"] for r in inv["consonant_plus_h"]} == PLUS_H
    assert phonology.get_full_phonology_display()["summary"]["total_consonants"] == 16


def test_word_finals(phon):
    # j ends native words (siej, biej): allowed, not forbidden, not a loan sign.
    assert "j" in phon["allowed_final_consonants"]
    assert "j" not in phon["forbidden_final"]
    assert phon["loan_final_signals"] == ["l", "s"]
    # y never ends a word; h and w do.
    assert "y" in phon["forbidden_final"]
    assert not {"h", "w"} & set(phon["forbidden_final"])
    rules = phon["rules"]
    assert "j" in rules["final_allowed_singles"]
    assert "y" in rules["final_forbidden"]
    assert not {"h", "w", "j"} & set(rules["final_forbidden"])
    j = [it for g in phon["consonant_chart"].values() for it in g if it["symbol"] == "j"]
    assert j[0]["final"] is True

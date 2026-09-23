"""The word panel's senses: the source dictionary's "[Imit. x-y.]" note.

The 1906 dictionary prints an imitative — its word-collocation — inside the
definition. The API lifts it into its own field so the panel can label it
instead of printing bracketed text as part of the meaning. Synthetic words
only: nothing here is copied from the lexicon.
"""
from khasi_spell import api


def test_imitative_is_lifted_out_of_the_sense():
    out = api._split_senses(["To sit; To settle. [Imit. ab-cd.]"], "verb", "")
    assert out["senses"] == ["To sit", "To settle."]
    assert out["imitatives"] == ["ab-cd"]


def test_a_gloss_that_is_only_an_imitative_keeps_it():
    out = api._split_senses(["[Imit. ef- gh.]"], "verb", "")
    assert "senses" not in out
    assert out["imitatives"] == ["ef-gh"]          # the scan's "ef- gh" closed up


def test_a_note_the_scan_cut_short_is_dropped():
    out = api._split_senses(["A thing. [Imit. ka"], "noun", "")
    assert out["senses"] == ["A thing."]
    assert "imitatives" not in out


def test_the_english_word_imitate_is_not_a_note():
    out = api._split_senses(["Imitate the cry"], "verb", "")
    assert out["senses"] == ["Imitate the cry"]
    assert "imitatives" not in out

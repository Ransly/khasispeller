"""
Run-together split suggestions.

`jongki` is `jong ki` written solid — the corpus writes it spaced 17,312
times against 2,642. The checker was right to flag it and wrong in what it
offered: edit distance cannot propose a space, so every candidate came from
a vocabulary of single words and the one correct answer was unreachable.
What the reader got instead was `jongka` — the same error with a different
clitic — and `jongka` was offered `jongki` straight back.

These tests pin three things: the split is offered and ranked first, it
survives the context re-ranker (which scored a two-word candidate as one
token and buried it), and words that merely happen to divide into two known
words are left alone.

No lexicon content here. `jong` is a grammatical particle and the rest are
corpus frequencies.
"""
import pytest

from khasi_spell import KhasiSpeller, splits


@pytest.fixture(scope="module")
def sp():
    return KhasiSpeller(eager=True)


@pytest.fixture(scope="module")
def table():
    return splits.load()


# The possessive paradigm: `jong` + a pronoun or noun-class clitic.
#
# `jongphi` and `jongpha` are here on the project's Khasi speaker's ruling,
# not on the frequency ratio — one sits at 1.29 (under the 2.0 default) and
# the other at 7 occurrences (under the generator's count floor). They are
# pinned precisely BECAUSE the defaults exclude them: a later change to
# either threshold must not quietly drop two members of a paradigm whose
# other seven qualify on their own.
PARADIGM = ["jongu", "jongka", "jongki", "jongngi", "jongnga", "jongme",
            "jongphi", "jongpha"]

# Words that divide into two known words but are written solid by the
# corpus. `tangba` is 767 solid against 267 spaced; `wanrah` 4,604 against
# 550. A rule that split on decomposability alone would wreck both.
SOLID = ["tangba", "jongno", "wanrah", "iathuh"]


def test_table_loads(table):
    assert table, "split table is empty — is the review CSV present?"


@pytest.mark.parametrize("word", PARADIGM)
def test_paradigm_is_split(table, word):
    head, _, tail = word.partition("jong")
    assert table.get(word) == "jong " + tail


@pytest.mark.parametrize("word", SOLID)
def test_solid_words_are_not_split(table, word):
    assert word not in table, f"{word} is written solid by the corpus"


@pytest.mark.parametrize("word", PARADIGM)
def test_split_is_offered_first(sp, word):
    sug = sp.suggest(word, n=5)
    assert sug, f"{word} produced no suggestion at all"
    assert " " in sug[0], f"{word} -> {sug} (split not first)"


def test_split_survives_context_reranking(sp):
    """The re-ranker scored `jong ngi` as one token, missed every n-gram and
    dropped it to last. check_text is the path the web UI uses, so a fix that
    only works in suggest() reaches nobody."""
    doc = sp.check_text("Ka iing jongngi ka bha bad ka jongki ka sniew").to_dict()
    got = {c["original"]: c["suggestion"] for c in doc["corrections"]}
    assert got.get("jongngi") == "jong ngi"
    assert got.get("jongki") == "jong ki"


def test_no_suggestion_cycles(sp):
    """A -> B -> A leaves the writer toggling between two wrong forms."""
    for word in PARADIGM:
        for s in sp.suggest(word, n=5):
            if " " in s:
                continue
            assert word not in sp.suggest(s, n=5), f"cycle {word} -> {s} -> {word}"


def test_a_runtogether_is_never_itself_a_suggestion(sp, table):
    """If we would flag it, we must not propose it."""
    for word in PARADIGM:
        for s in sp.suggest(word, n=5):
            assert s not in table, f"{word} was offered run-together {s!r}"


def test_verdict_column_overrides_the_ratio(tmp_path):
    """The frequency ratio is the default for unruled rows, never a
    replacement for the linguist's ruling."""
    csv_path = tmp_path / "r.csv"
    csv_path.write_text(
        "surface,count,split_as,phrase_count,example,verdict,note\n"
        "aaabbb,100,aaa bbb,1,,split,\n"        # ratio says no, verdict says yes
        "cccddd,10,ccc ddd,9999,,keep,\n",      # ratio says yes, verdict says no
        encoding="utf-8")
    t = splits.load(csv_path)
    assert t.get("aaabbb") == "aaa bbb"
    assert "cccddd" not in t


def test_missing_file_is_not_an_error(tmp_path):
    assert splits.load(tmp_path / "nope.csv") == {}


@pytest.mark.parametrize("word,spaced", [("jongphi", "jong phi"),
                                         ("jongpha", "jong pha")])
def test_ruled_rows_survive_the_ratio(table, word, spaced):
    """A verdict must outrank the frequency evidence, and must also survive
    the generator rebuilding the file — a ruling that vanished on the next
    regeneration would be worse than no verdict column at all."""
    assert table.get(word) == spaced

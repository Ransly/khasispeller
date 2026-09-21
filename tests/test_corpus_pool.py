"""
Candidate supply from corpus types.

The gap these cover: a word can be ordinary Khasi, appear thousands of times
in the corpus, and still be uncorrectable — because the symmetric-delete
index is built from lexicon surface forms, so a string the lexicon never
recorded is not in the list to be ranked. No cost model retrieves a
candidate that was never a candidate.

This is candidate SUPPLY only. Acceptance is a separate question with a
separate predicate, and these tests assert that the two stayed separate:
admitting a corpus type makes it offerable, never makes it a known word.

No lexicon content appears here. `pyntreikam` and `nongthohkhubor` are
corpus types, absent from `data/khasi_db.json` by construction — that
absence is exactly what is under test.

The lexicon loads once per module (~16-20 s); everything shares one
instance.
"""

import pytest

from khasi_spell import KhasiSpeller


@pytest.fixture(scope="module")
def sp():
    return KhasiSpeller(eager=True)


# Corpus types, none of them lexicon entries. Counts are from
# data/corpus_freq.json at the time of writing.
CORPUS_ONLY = [
    ("pyntreikam", 1940),
    ("nongthohkhubor", 1035),
]


# ----------------------------------------------------------------------
# Supply — the reproduction
# ----------------------------------------------------------------------

def test_corpus_frequent_word_is_reachable(sp):
    """A one-deletion typo of a 1,940-occurrence word must reach it."""
    assert "pyntreikam" in sp.suggest("pyntreika", n=10)


@pytest.mark.parametrize("word,_count", CORPUS_ONLY)
def test_corpus_types_enter_the_candidate_pool(sp, word, _count):
    assert word in set(sp.analyser.db.all_surface_forms())


def test_suggestion_is_never_the_input(sp):
    """A correction that is the input is not a correction.

    Reachable only once corpus types are in the pool: before that the
    engine could not return `pyntreikam` at all, let alone for itself.
    """
    for w in ["pyntreikam", "shnong", "jingpule"]:
        assert w not in sp.suggest(w, n=10)


# ----------------------------------------------------------------------
# The seam — supply must not become acceptance
# ----------------------------------------------------------------------

@pytest.mark.parametrize("word,_count", CORPUS_ONLY)
def test_corpus_types_are_attested_but_not_known(sp, word, _count):
    """`is_attested` answers 'was this written down', `is_known` 'is this a
    word'. Admitting corpus supply must move only the first."""
    db = sp.analyser.db
    assert db.is_attested(word), "corpus type should be offerable"
    assert not db.is_known(word), "corpus type must not become a known word"


@pytest.mark.parametrize("word,_count", CORPUS_ONLY)
def test_corpus_supply_does_not_change_acceptance(sp, word, _count):
    """WEAK — pins the current split, not a desirable end state.

    These are real Khasi words and a user typing one still sees it
    rejected. Fixing that is the acceptance repair, deliberately not done
    here; this test exists so that repair cannot happen by accident as a
    side effect of the supply change.
    """
    assert sp.check(word).is_correct is False


# ----------------------------------------------------------------------
# Traps
# ----------------------------------------------------------------------

def test_diacritic_headwords_keep_no_plain_rival(sp):
    """A corpus type that folds onto a lexicon entry must not enter the
    pool as a rival spelling at distance 0.

    The corpus holds zero `ï` and zero `ñ`, so every diacritic headword has
    an undiacriticked corpus twin. Admitting those as separate forms would
    undo the ï/ñ normalisation work silently.

    The invariant is a shared diacritic fold: no admitted corpus type may
    fold to the same string as a lexicon surface form. Testing it the other
    way round — adding diacritics back and asking `is_known` — does not
    work, because `w.replace("i", "ï")` is a no-op on a word with no `i`
    and the check collapses into `is_known(w)`.
    """
    db = sp.analyser.db
    corpus_forms = getattr(db, "_corpus_forms", set())
    fold = str.maketrans({"ï": "i", "ñ": "n"})
    lexicon_folds = {
        w.translate(fold) for w in db.all_surface_forms()
        if w not in corpus_forms
    }
    collisions = sorted(w for w in corpus_forms
                        if w.translate(fold) in lexicon_folds)
    assert not collisions[:5], (
        "corpus types share a diacritic fold with a lexicon form: %r"
        % collisions[:5]
    )


DIACRITIC_INPUTS = ["ïiatreilang", "ïatreilang"]


FOLD = str.maketrans({"ï": "i", "ñ": "n"})


@pytest.mark.parametrize("word", DIACRITIC_INPUTS)
def test_diacritic_input_is_never_told_to_drop_its_diacritic(sp, word):
    """Stripping the input's diacritics is not a correction.

    `ïatreilang` was answered with `iatreilang` — itself, minus the
    diaeresis — pushing the writer off the only spelling the lexicon uses.
    """
    for s in sp.suggest(word, n=10):
        assert s != word.lower().translate(FOLD), (
            f"{word!r} was offered its own diacritic-stripped form {s!r}"
        )


def test_corpus_types_are_admitted_in_lexicon_orthography(sp):
    """The corpus cannot type `ï`, so its types arrive in a reduced
    orthography and must be canonicalised before entering the pool.

    The lexicon's convention is absolute — 332 headwords begin `ïa-` and
    none begin plain `ia-` — so a pool holding `iatreilang` would offer a
    spelling the dictionary never uses.
    """
    db = sp.analyser.db
    plain = sorted(w for w in getattr(db, "_corpus_forms", set())
                   if w.startswith("ia") or w.endswith("ain"))
    assert not plain[:5], f"un-canonicalised corpus forms in pool: {plain[:5]}"


@pytest.mark.parametrize("typo,gold", [
    ("ïiatreilang", "ïatreilang"),
    ("iatreilang", "ïatreilang"),
    ("jingiatreilang", "jingïatreilang"),
])
def test_reduced_spelling_is_corrected_to_the_diacritic_one(sp, typo, gold):
    assert gold in sp.suggest(typo, n=5)


def test_suggestions_do_not_cycle(sp):
    """Applying a suggestion must not lead back to the input.

    `atreilang` (53 occurrences) is the corpus's misspelling of
    `iatreilang` (422) with the initial vowel dropped — a ratio of 7.96,
    which slipped under a dominance filter set at 10. Both were in the pool,
    each was offered as the correction of the other, and the user was cycled
    between two spellings, one of which is not a word.
    """
    for word in ["iatreilang", "pyntreikam", "shnong"]:
        for s in sp.suggest(word, n=5):
            assert word not in sp.suggest(s, n=5), (
                f"cycle: {word!r} -> {s!r} -> {word!r}"
            )


def test_pool_growth_is_bounded(sp):
    """The delete index costs ~27 keys per form and the deployment ceiling
    is 512 MB. An unbounded pool is an OOM, not a feature."""
    db = sp.analyser.db
    corpus_forms = getattr(db, "_corpus_forms", set())
    assert len(corpus_forms) < 60000, (
        "corpus pool of %d forms will not fit the memory budget"
        % len(corpus_forms)
    )


def test_check_and_check_text_agree_on_corpus_types(sp):
    """WEAK — records a disagreement, and a cost the supply fix creates.

    `_flag_words` reports a gate-3 rejection only when it has suggestions,
    so today these words are rejected by `check()` and silently dropped by
    `check_text()`: no underline, which is the right outcome reached by the
    wrong route. Giving them candidates removes that accident — the word is
    still rejected, but now it is visibly flagged and a correction is
    offered for a word that was never wrong.

    The assertion is therefore deliberately one-directional: it pins that
    the two entry points do not contradict each other, and the docstring
    records that agreement here is currently bought at the price of a false
    alarm. That price is what the acceptance repair has to pay back.
    """
    for word, _ in CORPUS_ONLY:
        single_ok = sp.check(word).is_correct
        flagged = bool(sp.check_text(word).to_dict().get("corrections"))
        if single_ok:
            assert not flagged, f"{word}: accepted alone but flagged in text"

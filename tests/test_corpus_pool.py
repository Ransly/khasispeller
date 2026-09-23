"""
Candidate supply from corpus types.

The gap these cover: a word can be ordinary Khasi, appear thousands of times
in the corpus, and still be uncorrectable — because the symmetric-delete
index is built from lexicon surface forms, so a string the lexicon never
recorded is not in the list to be ranked. No cost model retrieves a
candidate that was never a candidate.

Supply alone left a contradiction — the pool offered words the checker then
rejected (`jyla` -> `jylla`, and `jylla` flagged once accepted). Admitted
words now also pass the confidence vote, credited with corpus frequency.
`is_known()` still refuses them: acceptance comes through the vote, not by
turning corpus types into lexicon words.

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


@pytest.mark.parametrize("word,_count", CORPUS_ONLY + [("jylla", 24252),
                                                       ("wanrah", 4604)])
def test_admitted_corpus_words_are_accepted(sp, word, _count):
    """The acceptance repair. These are ordinary Khasi words the lexicon
    does not record; `jylla` ('state') was rejected at 1 point of the 3
    needed despite 24,252 corpus occurrences."""
    assert sp.check(word).is_correct is True
    assert not sp.check_text(word).to_dict().get("corrections")


def test_offered_corpus_words_are_accepted(sp):
    """Never suggest a word the checker would itself reject."""
    for typo, gold in [("jyla", "jylla"), ("wanra", "wanrah"),
                       ("pyntreika", "pyntreikam")]:
        sug = sp.suggest(typo, n=5)
        assert sug and sug[0] == gold, (typo, sug)
        assert sp.check(gold).is_correct, f"{gold!r} offered, then rejected"


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
    from khasi_spell.corpus_pool import _ain_confirmed
    db = sp.analyser.db
    forms = getattr(db, "_corpus_forms", set())
    plain_ia = sorted(w for w in forms if w.startswith("ia"))
    assert not plain_ia[:5], f"un-canonicalised corpus forms in pool: {plain_ia[:5]}"
    # -ain keeps its plain spelling only where the lexicon's own -aiñ rule
    # does not back the tilde — names and loans such as `hussain`.
    wrongly_plain = sorted(w for w in forms
                           if w.endswith("ain") and _ain_confirmed(w, db))
    assert not wrongly_plain[:5], wrongly_plain[:5]


def test_ain_is_restored_only_where_the_lexicon_agrees(sp):
    """`hussain` is a name and `risain` a loan (resign); `hussaiñ` and
    `risaiñ` are spellings no one writes. `thawain` is a Khasi compound whose
    final element the lexicon records with the tilde."""
    forms = getattr(sp.analyser.db, "_corpus_forms", set())
    assert "hussaiñ" not in forms and "risaiñ" not in forms
    assert "thawaiñ" in forms


def test_misspelled_diacritic_form_is_corrected(sp):
    assert "ïatreilang" in sp.suggest("ïiatreilang", n=5)


@pytest.mark.parametrize("plain,standard", [
    ("iatreilang", "ïatreilang"),
    ("jingiatreilang", "jingïatreilang"),
])
def test_reduced_spelling_is_accepted_with_the_diacritic_offered(sp, plain, standard):
    """The corpus writes every ïa- word without its ï. Treated exactly like a
    lexicon word typed that way (`iathuh`): accepted, with the standard
    spelling offered as an alternative rather than as a correction."""
    assert sp.check(plain).is_correct
    assert standard in [v.variant for v in sp.variants(plain)]


def test_runtogether_spellings_are_not_admitted(sp):
    """`jongki` is `jong ki` written solid. The split table corrects it; the
    pool must neither offer it nor, now, accept it."""
    forms = getattr(sp.analyser.db, "_corpus_forms", set())
    assert "jongki" not in forms
    r = sp.check("jongki")
    assert r.is_correct is False and r.suggestions[0] == "jong ki"


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
    """Both entry points go through the same vote, so an accepted corpus
    word is neither rejected by check() nor underlined by check_text()."""
    for word, _ in CORPUS_ONLY:
        assert sp.check(word).is_correct
        assert not sp.check_text(word).to_dict().get("corrections"), word


def test_acceptance_can_be_switched_off():
    """`corpus_pool_accept=False` restores supply-only behaviour, which is
    how the acceptance repair is measured against its absence."""
    off = KhasiSpeller(eager=True, corpus_pool_accept=False)
    assert off.check("pyntreikam").is_correct is False
    assert "pyntreikam" in off.suggest("pyntreika", n=5)
    del off


def test_pool_off_is_not_honoured_after_a_pooled_speller(sp):
    """WEAK — pins a known weakness, not a desirable end state.

    KhasiDB is memoised per data source, so every speller in a process
    shares one database and one delete index, and corpus_pool writes into
    both. A later `use_corpus_pool=False` speller therefore still offers
    pool words. Acceptance does not leak (its switch is per-checker), but
    supply does. Measure pool-on against pool-off in separate processes.
    """
    off = KhasiSpeller(eager=True, use_corpus_pool=False)
    assert "pyntreikam" in off.suggest("pyntreika", n=5)
    del off

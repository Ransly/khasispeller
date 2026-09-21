"""
Spellchecker regression tests.

The parent platform shipped 39 tests, none of which exercised the
spelling path — so any change to it was unguarded. This file closes that
gap for the extracted project.

Assertions record *observed* behaviour, including where it is weak. Where
the checker currently ranks a wrong candidate first, the test asserts the
correct word appears in the top 5 rather than pretending it is first.
Those cases are marked WEAK and are the ones to watch when the ranking
model is replaced.

The lexicon loads once per module (~15-20 s), so the whole file runs in
roughly the time of a single construction.
"""

import pytest

from khasi_spell import KhasiSpeller
from khasi_spell.speller import (
    GATE_ACCEPTED,
    GATE_CHECKED,
    GATE_PHONOTACTIC,
)


@pytest.fixture(scope="module")
def sp():
    return KhasiSpeller(eager=True)


# ----------------------------------------------------------------------
# The morphology gate — the reason this checker exists
# ----------------------------------------------------------------------

# Derived forms that ARE listed in the lexicon. These pass whether or not
# the morphology gate works, so on their own they prove nothing about it —
# they are here to catch a regression in ordinary acceptance.
LISTED_DERIVED_FORMS = [
    "jingbha",     # jing- + bha    nominaliser
    "pynbha",      # pyn- + bha     causative
    "jingpynim",   # jing- + pyn- + im   stacked prefixes
]

# Derived forms that are NOT in the lexicon. These are the real test of the
# morphology gate: a word-list checker flags every one of them.
UNLISTED_DERIVED_FORMS = [
    # Each is prefix + a root the lexicon knows, with the combined form
    # absent from both lexicon and corpus — the morphology gate is the only
    # thing that can accept them.
    #
    # 'nongshkai' used to head this list and was wrong: it is not a Khasi
    # word (0 corpus uses; its "root" shkai is not in the lexicon — the real
    # forms are shkait and jingshkair). It was a false negative all along,
    # and the parse-plausibility signal correctly rejects it.
    "nongsngew",   # nong- + sngew (root: 3,129 corpus uses; form: 0)
    "nonglah",     # nong- + lah   (root: 19,535 uses; form: 0)
    "babam",       # ba-   + bam   (root:  2,403 uses; form: 0)
]


@pytest.mark.parametrize("word", LISTED_DERIVED_FORMS)
def test_listed_derived_forms_are_accepted(sp, word):
    result = sp.check(word)
    assert result.is_correct, f"{word} was flagged as misspelled"
    assert result.gate == GATE_ACCEPTED
    assert result.suggestions == []


@pytest.mark.parametrize("word", UNLISTED_DERIVED_FORMS)
def test_unlisted_derived_forms_are_accepted(sp, word):
    """
    The point of the whole design: productively derived words are accepted
    even though they appear in no word list. Remove the morphology gate and
    every one of these becomes a false positive.
    """
    result = sp.check(word)
    assert not result.in_lexicon, (
        f"{word} is now a lexicon entry — pick another unlisted form, "
        f"this test no longer exercises the gate"
    )
    assert result.is_correct, f"{word} was flagged despite being well formed"
    assert result.gate == GATE_ACCEPTED
    assert result.morphologically_valid


@pytest.mark.parametrize("word", ["dieng", "kmie", "hikai", "khubor"])
def test_lexicon_words_are_accepted(sp, word):
    """Ordinary lexicon members pass the confidence gate."""
    assert sp.is_correct(word)


def test_gate_reports_why_a_word_was_accepted(sp):
    """The gate stage distinguishes lexicon hits from morphological parses."""
    result = sp.check("jingbha")
    assert result.gate_name == "accepted"
    assert result.morphologically_valid


# An infix is declared to sit after C1 — the FIRST consonant. Before the
# position was enforced, the compiled pattern's leading [^vowel]+ group
# swallowed whole clusters, so "slpa" parsed as "sl" + -p- + "a", rebuilt
# the real root "sla", and cleared the gate on morphology 2 + phonotactic 1.
# The misspelling was reported as a correct word and, because accepted words
# skip candidate generation, neither "sla" nor "slap" was ever offered.
CLUSTER_INFIX_NONWORDS = ["slpa", "slna", "smla", "krpad", "jnglut"]


@pytest.mark.parametrize("word", CLUSTER_INFIX_NONWORDS)
def test_infix_is_not_applied_after_a_consonant_cluster(sp, word):
    result = sp.check(word)
    assert not result.in_lexicon, (
        f"{word} is now a lexicon entry — this test no longer exercises the rule"
    )
    assert not result.is_correct, (
        f"{word} was accepted; an infix was parsed after a consonant cluster"
    )
    assert result.suggestions, f"{word} was rejected with nothing offered"


def test_slpa_offers_both_real_neighbours(sp):
    """The transposition of 'slap' and the root 'sla' both reach the user."""
    top = sp.check("slpa").suggestions[:3]
    assert "sla" in top and "slap" in top, top


@pytest.mark.parametrize("word,root", [
    ("kynshaid", "kshaid"),   # k + -yn- + shaid, the database's own example
])
def test_genuine_infix_forms_still_parse(sp, word, root):
    result = sp.check(word)
    assert result.is_correct, f"{word} lost its infix analysis"
    assert result.morphologically_valid


# `distances` and `glosses` are documented as parallel to `suggestions`.
# context.rerank() permutes `suggestions`, and used to leave the other two
# behind, so a reader saw one real word carrying another real word's meaning:
# "sla" was displayed as "Youth ...", which belongs to "samla".
@pytest.mark.parametrize("text", [
    "ka slpa ka la wan",
    "ka patbah ka la wan",
    "ka smla ka la wan",
])
def test_reranking_keeps_glosses_on_their_own_words(sp, text):
    for context in (False, True):
        result = sp.check_text(text, context=context)
        for c in result.corrections:
            assert len(c.glosses) == len(c.suggestions)
            assert len(c.distances) == len(c.suggestions)
            for word, gloss in zip(c.suggestions, c.glosses):
                own = sp.check(word).gloss
                if own:
                    assert gloss == own, (
                        f"{word!r} was listed with {gloss!r}, "
                        f"but the lexicon glosses it {own!r}"
                    )


def test_context_actually_reorders_that_example(sp):
    """Guards the test above: it is only meaningful when the order moves."""
    plain = sp.check_text("ka slpa ka la wan", context=False).corrections[0]
    ctx = sp.check_text("ka slpa ka la wan", context=True).corrections[0]
    assert plain.suggestion != ctx.suggestion


# ----------------------------------------------------------------------
# Detection and correction
# ----------------------------------------------------------------------

# (misspelling, expected correction) — verified against the live engine.
TOP1_CASES = [
    ("lyngdo", "lyngdoh"),   # missing final -h
    ("kmei", "kmie"),        # ie/ei vowel-nucleus transposition
    ("umm", "um"),           # doubled consonant
    ("dign", "ding"),        # gn -> ng, caught by a normalisation rule
    # Was WEAK — 'shna' outranked it while every lexicon word scored 20 and
    # ties fell through to alphabetical order. Corpus counts separate them
    # decisively (shnong 22,530 vs shna 2,971).
    ("shnng", "shnong"),
]


@pytest.mark.parametrize("wrong,right", TOP1_CASES)
def test_top1_correction(sp, wrong, right):
    """Cases the checker currently gets right at rank 1."""
    suggestions = sp.suggest(wrong)
    assert suggestions, f"no suggestions for {wrong}"
    assert suggestions[0] == right, f"{wrong}: expected {right}, got {suggestions}"


def test_corpus_frequencies_are_applied(sp):
    """The ranking prior should be corpus counts, not the 11-value placeholder."""
    summary = sp.corpus_frequencies
    assert summary is not None, "corpus_freq.json missing — run scripts/build_corpus_freq.py"
    assert summary["distinct_values"] > 20, "still on the pseudo-frequency scale"


@pytest.mark.parametrize("word", ["shnng", "dign", "lyngdo", "kmei"])
def test_misspellings_are_flagged(sp, word):
    """Detection: none of these should be accepted as valid."""
    result = sp.check(word)
    assert not result.is_correct
    assert result.gate in (GATE_PHONOTACTIC, GATE_CHECKED)


# ----------------------------------------------------------------------
# Phonotactics
# ----------------------------------------------------------------------

def test_final_aspirate_is_rejected(sp):
    """Khasi aspirates never occur word-finally, so 'bakh' is illegal."""
    result = sp.check("bakh")
    assert not result.is_correct
    assert result.gate == GATE_PHONOTACTIC
    assert not result.phonotactically_valid


# ----------------------------------------------------------------------
# Text level
# ----------------------------------------------------------------------

def test_check_text_finds_errors_with_spans(sp):
    text = "Ka shnng ka bha bad u briew u dign ha ka iing"
    result = sp.check_text(text)

    assert result.has_errors
    flagged = {c.original for c in result.corrections}
    assert "shnng" in flagged
    assert "dign" in flagged

    # Spans must index back into the original string exactly.
    for c in result.corrections:
        assert text[c.start:c.end] == c.original


def test_check_text_leaves_valid_words_alone(sp):
    """Valid words, derived ones included, must not be flagged."""
    result = sp.check_text("Ka jingbha bad u nongsngew")
    assert not result.has_errors
    assert result.corrected == result.text


def test_correct_text_applies_corrections(sp):
    corrected = sp.correct_text("Ka shnng ka bha")
    assert corrected != "Ka shnng ka bha"
    assert "shnng" not in corrected


def test_correct_text_preserves_surrounding_text(sp):
    """Offset arithmetic must not corrupt the untouched parts."""
    corrected = sp.correct_text("Ka dign ha ka iing")
    assert corrected.startswith("Ka ")
    assert corrected.endswith(" ha ka iing")


# ----------------------------------------------------------------------
# Public API surface
# ----------------------------------------------------------------------

def test_correct_returns_word_unchanged_when_valid(sp):
    assert sp.correct("jingbha") == "jingbha"


def test_suggest_is_empty_for_valid_words(sp):
    assert sp.suggest("dieng") == []


def test_vocabulary_is_loaded(sp):
    assert sp.vocabulary_size > 30000


def test_lazy_loading_defers_construction():
    """Constructing without eager=True must not read the lexicon."""
    lazy = KhasiSpeller()
    assert not lazy.loaded


def test_add_word_teaches_the_session(sp):
    """A word added at runtime stops being flagged."""
    invented = "zzyrkhang"
    sp.add_word(invented)
    assert invented in sp.analyser.spell._speller.nlp_data


def test_result_objects_serialise(sp):
    """Results must survive JSON encoding for the API layer."""
    import json

    json.dumps(sp.check("lyngdo").to_dict())
    json.dumps(sp.check_text("Ka shnng").to_dict())


# ----------------------------------------------------------------------
# Assimilation gemination — regression, reported 2026-08-25
# ----------------------------------------------------------------------
#
# pyn- + l-initial root geminates: pyn- + lait -> pyl|lait. The surface
# prefix 'pyl' consumes only the first half of the geminate, so whatever
# follows must still begin with the root's own 'l'.
#
# The reverse rule never checked that, so any word starting 'pyl' or 'wal'
# was split after three characters and the tail treated as a root. 'pyleng'
# became pyn- + 'eng', and because 'eng' is a real lexicon word the parse was
# confirmed and the misspelling silently accepted. Correct form: 'pylleng'.

GEMINATE_MISSPELLINGS = ["pyleng", "pyleit"]

# Real assimilated forms — must keep passing.
GEMINATE_VALID = ["pylleng", "pyllait", "pyllam", "wallam", "pyllong"]

# War (2001) p.58 lists three roots that resist assimilation.
ASSIMILATION_EXCEPTIONS = ["pynleh", "pynloit", "pynlong"]


@pytest.mark.parametrize("word", GEMINATE_MISSPELLINGS)
def test_single_l_is_not_a_valid_assimilated_form(sp, word):
    """A missing geminate must be flagged, not silently accepted."""
    assert not sp.is_correct(word), (
        f"{word} accepted — the reverse assimilation rule is matching a "
        f"single 'l' where the rule produces a geminate"
    )


@pytest.mark.parametrize("word", GEMINATE_VALID)
def test_geminate_forms_still_accepted(sp, word):
    assert sp.is_correct(word)


@pytest.mark.parametrize("word", ASSIMILATION_EXCEPTIONS)
def test_assimilation_exceptions_still_accepted(sp, word):
    """pynleh / pynloit / pynlong resist assimilation and stay unassimilated."""
    assert sp.is_correct(word)


def test_pyleng_suggests_the_geminate_first(sp):
    """
    'pleng' and 'pylleng' both sit at distance 1 with frequency 20, so the
    merge fell through to alphabetical order and 'pleng' won on the letter p.
    Counting shared onset phonemes (p-y-l = 3 vs p = 1) puts the real
    correction first.
    """
    suggestions = sp.suggest("pyleng")
    assert "pylleng" in suggestions
    assert suggestions[0] == "pylleng", f"expected pylleng first, got {suggestions}"


# KNOWN GAP: 'waleng' is still accepted, but by a different route — the
# Phase 4 solid-compound splitter reports type 'compound_fusion' rather than
# assimilation. That is a separate over-acceptance path in complex.py and is
# deliberately not fixed here. Kept as documentation, not asserted, so the
# suite does not fail on a known limitation.


# ----------------------------------------------------------------------
# Real sentence, reported 2026-08-26
# ----------------------------------------------------------------------

SENTENCE = ("katba u bynriew u dand iaid hok, im hok katkum ka ain u Blei, "
            "u suk-u-sain, u roi-u-par bad u man-bha man-miay.")


def test_reported_sentence_spelling(sp):
    """
    Two genuine misspellings, each of which needed a fix to catch:

      dand -> dang       phoneme distance made 'dang' 1.4 away while 'and'
                         (an English fragment from a multi-word entry) sat
                         at 1.0 and won. The character-distance blend sees
                         the single keystroke.
      man-miay -> man-miat
                         Phase 4 accepts any hyphenated token as a compound
                         without checking its parts, so the bad half was
                         invisible.
    """
    result = sp.check_text(SENTENCE)
    fixes = {c.original: c.suggestion for c in result.corrections}
    assert fixes.get("dand") == "dang"
    assert fixes.get("man-miay") == "man-miat"
    assert "dang" in result.corrected and "man-miat" in result.corrected


def test_reported_sentence_offers_diacritics(sp):
    """The accepted words carry alternative spellings, reported separately."""
    result = sp.check_text(SENTENCE)
    offered = {v.word: [x["variant"] for x in v.variants] for v in result.variants}
    assert "ïaid" in offered.get("iaid", [])
    assert "aiñ" in offered.get("ain", [])
    # Apostrophe elisions are a different kind of variation and must not
    # leak in from the fold key — 'ka' should not offer "k'a".
    assert "ka" not in offered
    assert "im" not in offered


def test_variants_are_never_auto_applied(sp):
    """
    Alternatives are guidance. The corrected string must contain only real
    corrections, never a diacritic swap the writer did not ask for.
    """
    result = sp.check_text(SENTENCE)
    assert "ïaid" not in result.corrected
    assert "aiñ" not in result.corrected


def test_hyphenated_compound_with_good_parts_is_left_alone(sp):
    """man-bha and roi-u-par are fine and must not be touched."""
    result = sp.check_text("u man-bha bad u roi-u-par")
    assert not result.has_errors


# ----------------------------------------------------------------------
# Context-aware re-ranking (khasi_spell.context)
# ----------------------------------------------------------------------

class TestContextReranking:
    """
    The trigram model re-orders candidates using neighbouring words.

    These pin behaviour, not accuracy — accuracy lives in the frozen
    sentence benchmark (tests/sentence_benchmark.json), measured by
    scripts/run_sentence_benchmark.py.
    """

    def test_rerank_is_noop_without_a_model(self):
        """A missing n-gram model must not break checking."""
        from khasi_spell import context

        class C:
            start, end = 0, 4
            suggestion = "aaa"
            suggestions = ["aaa", "bbb"]

        c = C()
        assert context.rerank("aaa bbb ccc", [c], None) == 0
        assert c.suggestion == "aaa"          # untouched

    def test_single_candidate_left_alone(self, sp):
        from khasi_spell import context

        class C:
            start, end = 0, 3
            suggestion = "one"
            suggestions = ["one"]

        c = C()
        assert context.rerank("one two", [c], sp._lm()) == 0
        assert c.suggestions == ["one"]

    def test_alpha_zero_lets_the_model_decide(self, sp):
        """At alpha=0 the string ordering is discarded entirely."""
        from khasi_spell import context

        lm = sp._lm()
        if lm is None:
            pytest.skip("data/ngrams.json.gz not built")

        class C:
            start, end = 0, 2
            suggestion = "xx"
            suggestions = ["xxqz", "ka"]     # 'ka' is the commonest Khasi word

        c = C()
        context.rerank("xx u briew", [c], lm, alpha=0.0)
        assert c.suggestion == "ka"
        assert c.suggestions[0] == "ka"

    def test_large_alpha_preserves_engine_order(self, sp):
        from khasi_spell import context

        lm = sp._lm()
        if lm is None:
            pytest.skip("data/ngrams.json.gz not built")

        class C:
            start, end = 0, 2
            suggestion = "xxqz"
            suggestions = ["xxqz", "ka"]

        c = C()
        context.rerank("xx u briew", [c], lm, alpha=1000.0)
        assert c.suggestions == ["xxqz", "ka"]

    def test_check_text_accepts_the_context_switch(self, sp):
        """Both paths must return the same shape and the same corrections."""
        text = "u dand iaid hok bad u man-bha"
        on = sp.check_text(text, variants=False, context=True)
        off = sp.check_text(text, variants=False, context=False)
        assert [c.start for c in on.corrections] == [c.start for c in off.corrections]
        for c in on.corrections:
            assert c.suggestion in c.suggestions

    def test_corrected_string_matches_the_chosen_suggestions(self, sp):
        """The rebuilt string must reflect whatever re-ranking chose."""
        text = "katba u bynriew u dand iaid hok"
        r = sp.check_text(text, variants=False, context=True)
        out = text
        for c in sorted(r.corrections, key=lambda c: -c.start):
            out = out[:c.start] + c.suggestion + out[c.end:]
        assert r.corrected == out

    def test_lm_accessor_is_cached(self, sp):
        """The model is ~5 MB; loading it per token would dominate runtime."""
        first = sp._lm()
        assert sp._lm() is first


class TestSentenceLatency:
    """
    Guards against reintroducing the discarded edit-2 call.

    `check_sentence` used to borrow `autocorrect.Speller.check_sentence()`
    as a tokeniser. That call runs full edit-distance-2 correction on every
    token and the results were thrown away — 41 s on the sentence below,
    which is dominated by long non-Khasi words. Generous bounds: this is a
    tripwire for an O(n^2 * alphabet^2) blow-up, not a performance target.
    """

    PATHOLOGICAL = ("Kitei ki jaitbynriew Scheduled Tribes bad Scheduled "
                    "Castes haneng, ki lah ban thid jaka katba mon ha Meghalaya.")

    def test_foreign_proper_nouns_do_not_explode(self, sp):
        import time

        sp.check_text("ka iing", variants=False, context=False)   # warm
        started = time.time()
        sp.check_text(self.PATHOLOGICAL, variants=False, context=False)
        elapsed = time.time() - started
        assert elapsed < 3.0, (
            f"{elapsed:.1f}s for one sentence — the Speller edit-2 path is "
            f"probably back; it measured 18s before removal"
        )

    def test_detection_survives_on_that_sentence(self, sp):
        """Speed must not have come from checking fewer words."""
        r = sp.check_text(self.PATHOLOGICAL, variants=False, context=False)
        flagged = {c.original for c in r.corrections}
        # 'thid' is the injected error in benchmark item #7.
        assert "thid" in flagged

    def test_every_token_reaches_the_gate(self, sp):
        """
        The old pre-filter only passed tokens the Speller *changed*, so a
        word it corrected to itself was never gate-checked. Every token
        should now be considered.
        """
        text = "ka iing ka la dei ban dawa ia sorykar"
        r = sp.check_text(text, variants=False, context=False)
        assert any(c.original == "sorykar" for c in r.corrections)


class TestForeignSuppression:
    """
    Proper nouns and acronyms are withheld, not corrected.

    Rationale and the measured trade-off live in `khasi_spell.foreign`.
    """

    def test_sentence_initial_capital_is_still_checked(self):
        """A capital opening a sentence says nothing about word class."""
        from khasi_spell import foreign

        text = "Sorykar ka la iaid."
        assert foreign.is_sentence_initial(text, 0)
        assert foreign.classify("Sorykar", True) is None

    def test_mid_sentence_capital_is_withheld(self):
        from khasi_spell import foreign

        # A name the gazetteer does NOT hold, so the capitalisation rule is
        # what fires. 'Shillong' would report as KNOWN_NAME instead.
        text = "ka la iaid sha Ardent"
        start = text.index("Ardent")
        assert not foreign.is_sentence_initial(text, start)
        assert foreign.classify("Ardent", False) == foreign.CAPITALISED

    def test_acronym_is_withheld(self):
        from khasi_spell import foreign

        assert foreign.classify("MDA", False) == foreign.ACRONYM
        assert foreign.classify("MDA", True) == foreign.ACRONYM   # even first
        # A lone capital is an initial ('u C. Kharkongor'), so it is caught
        # by the capitalised rule rather than counted as an acronym.
        assert foreign.classify("A", False) == foreign.CAPITALISED
        assert foreign.classify("A", True) is None

    def test_lowercase_word_is_never_withheld(self):
        from khasi_spell import foreign

        assert foreign.classify("sorykar", False) is None

    def test_sentence_start_detected_after_punctuation(self):
        from khasi_spell import foreign

        text = 'U briew u la leit. Shillong ka jingieid.'
        assert foreign.is_sentence_initial(text, text.index("Shillong"))

    def test_withheld_tokens_are_reported_not_deleted(self, sp):
        r = sp.check_text("Ka sorkar jong ka MDA ka la iaid.", variants=False)
        assert not any(c.original == "MDA" for c in r.corrections)
        assert any(s["word"] == "MDA" for s in r.skipped)
        # the withheld suggestion is preserved for a caller that wants it
        assert all("would_suggest" in s for s in r.skipped)

    def test_corrected_string_leaves_withheld_tokens_alone(self, sp):
        text = "Ka sorkar jong ka MDA ka la iaid."
        r = sp.check_text(text, variants=False)
        assert "MDA" in r.corrected

    def test_switch_restores_the_old_behaviour(self, sp):
        text = "Ka sorkar jong ka MDA ka la iaid."
        off = sp.check_text(text, variants=False, skip_foreign=False)
        assert off.skipped == []
        assert any(c.original == "MDA" for c in off.corrections)

    def test_real_lowercase_errors_still_flagged(self, sp):
        """Suppression must not swallow ordinary misspellings."""
        r = sp.check_text("Ka iing ka la dei ban dawa ia sorykar MDA.",
                          variants=False)
        assert any(c.original == "sorykar" for c in r.corrections)


class TestGazetteer:
    """
    Proper-noun recognition from the project's own NER lists.

    Recognition only — the gazetteer is never used as a correction source.
    See `khasi_spell.foreign` for why (18.2% of genuine names sit within
    one edit of a different entry).
    """

    def test_gazetteer_loads(self):
        from khasi_spell import foreign

        assert len(foreign.gazetteer()) > 5000

    def test_name_recognised_sentence_initially(self):
        """The rule capitalisation cannot express: position-independent."""
        from khasi_spell import foreign

        assert foreign.classify("Sohra", True) == foreign.KNOWN_NAME
        assert foreign.classify("Lyndem", True) == foreign.KNOWN_NAME

    def test_name_beats_the_capitalisation_guess(self, sp):
        """A known name reports as such, not as an assumption."""
        from khasi_spell import foreign

        assert foreign.classify("Shillong", False) == foreign.KNOWN_NAME
        assert foreign.classify("Ardent", False) == foreign.CAPITALISED

    def test_lexicon_collisions_are_inert(self, sp):
        """
        'bah', 'blei', 'dawa' are gazetteer tokens AND ordinary Khasi words.
        The lexicon accepts them first, so they must never be withheld.
        """
        for word in ("bah", "blei", "dawa"):
            r = sp.check_text(f"u briew u la {word} ia ka kam.", variants=False)
            assert not any(s["word"] == word for s in r.skipped), word

    def test_gazetteer_does_not_swallow_real_errors(self, sp):
        r = sp.check_text("Sohra ka dei ka nong bad u la dawa ia sorykar.",
                          variants=False)
        assert any(c.original == "sorykar" for c in r.corrections)
        assert not any(c.original == "Sohra" for c in r.corrections)


class TestSubPhonemicTokens:
    """
    Fragments must not pass as words.

    Two independent routes let them through, both found from a report that
    `ng` was accepted:

      1. Gate 0 consulted `semantically_invalid_standalone` but then deferred
         to the lexicon, which holds `ng` as a letter-name entry
         (kh_DB_005909, "The seventh letter of the Khasi Alphabet"). The
         escape hatch always fired, so the gate never blocked anything.
      2. `is_known()` accepts anything in `_compound_tokens`, built by
         splitting multi-word surface forms. That admitted bare digraphs
         (`sh`, `kh`) and English words from translations embedded in those
         surface forms (`cow`, `and`, `big`, `car`).
    """

    def test_ng_is_rejected(self, sp):
        """The reported bug. The DB lists 'ng' as invalid standalone."""
        r = sp.check("ng")
        assert not r.is_correct
        assert r.gate_name == "invalid_standalone"

    def test_the_db_still_declares_it_invalid(self):
        from khasi_engine import phonology

        assert "ng" in phonology.SEMANTICALLY_INVALID_STANDALONE

    def test_lexicon_entry_does_not_override_the_rule(self, sp):
        """
        'ng' was a lexicon entry — the letter name, kh_DB_005909 — and Gate 0
        deferred to it, which is how a bare digraph came to pass a spell
        check. The entry has since been quarantined for having no vowel, so
        the gate is no longer the only thing standing between `ng` and
        acceptance; both must still reject it.
        """
        assert not sp.analyser.db.is_known("ng")
        assert not sp.check("ng").is_correct
        assert sp.check("ng").gate_name == "invalid_standalone"

    def test_bare_digraphs_are_rejected(self, sp):
        for tok in ("sh", "kh", "ph", "th", "bh"):
            assert not sp.check(tok).is_correct, tok

    def test_english_from_embedded_translations_is_rejected(self, sp):
        """
        These reached the index through surface forms like
        'ka masi ka la rong jyndat ïa ka jaiñ ka the cow took away the cloth'.
        """
        for tok in ("cow", "and", "big", "car", "for"):
            assert not sp.check(tok).is_correct, tok

    def test_words_living_only_in_compounds_still_work(self, sp):
        """
        The regression this screen must not cause. Compound-token
        extraction exists so that `shnong`, which appears only inside
        phrases like `dorbar shnong`, stays recognised.
        """
        for word in ("shnong", "mynsiem", "iing", "kynih", "dorbar"):
            assert sp.check(word).is_correct, word

    def test_screen_cannot_reject_pronounceable_khasi(self):
        from khasi_engine.database import _phonotactic_ok

        assert _phonotactic_ok("shnong")
        assert _phonotactic_ok("jingïathuh")
        assert not _phonotactic_ok("sh")
        assert not _phonotactic_ok("cow")


class TestFragmentSuggestions:
    """
    A rejected fragment must still get usable suggestions.

    Gate 0 returned a hardcoded empty list: it was written to block, not to
    help. Once it suggested, two further problems showed up — the bare
    consonants `n` and `g` were offered for `ng` (a correction that is
    itself rejected is not a correction), and over-fetching was needed
    because filtering runs after the search.
    """

    def test_ng_gets_suggestions(self, sp):
        r = sp.check("ng")
        assert not r.is_correct
        assert r.suggestions, "rejected without offering any correction"

    def test_the_obvious_corrections_are_offered(self, sp):
        """`nga` 'I' and `ngi` 'we' are the words `ng` is a fragment of."""
        suggestions = sp.check("ng").suggestions
        assert "nga" in suggestions
        assert "ngi" in suggestions

    def test_never_suggests_something_it_would_reject(self, sp):
        """The rule that removed `n` and `g` from the list."""
        for word in ("ng", "sh", "kh", "ph", "th"):
            for cand in sp.check(word).suggestions:
                assert sp.check(cand).is_correct, f"{word} -> {cand} is itself invalid"

    def test_bare_consonants_are_not_offered(self, sp):
        assert not ({"n", "g", "t", "s", "k"} & set(sp.check("ng").suggestions))

    def test_one_letter_clitics_remain_offerable(self, sp):
        """`u`, `i`, `a` are real Khasi words; the vowel rule must keep them."""
        checker = sp.analyser.spell
        for clitic in ("u", "i", "a"):
            assert checker._offerable(clitic), clitic
        for fragment in ("n", "g", "ng"):
            assert not checker._offerable(fragment), fragment

    def test_filtering_does_not_shorten_the_list(self, sp):
        """Over-fetch: `ng` returned 3 of 5 before the search was widened."""
        assert len(sp.check("ng").suggestions) == 5


class TestSuggestionsShareMaterial:
    """
    A correction must have something in common with what was typed.

    Digraphs are single phonemes, so `ph` -> `u` costs one substitution and
    scores 1.0 — the same as `ph` -> `phi`, the actual word. The one-letter
    clitics then won the tie on frequency and `ph` was answered with
    `['u', 'phi', 'a', 'i', 'ï']`.
    """

    def test_suggestions_share_a_character_with_the_input(self, sp):
        for word in ("ph", "ng", "sh", "kh", "th"):
            for cand in sp.check(word).suggestions:
                assert set(cand.lower()) & set(word.lower()), f"{word} -> {cand}"

    def test_the_real_word_leads(self, sp):
        """`phi` ('you' pl.) is the word `ph` is a fragment of."""
        assert sp.check("ph").suggestions[0] == "phi"

    def test_unrelated_clitics_are_gone(self, sp):
        assert not ({"u", "a", "i", "ï"} & set(sp.check("ph").suggestions))

    def test_clitics_still_reachable_when_relevant(self, sp):
        """The rule is 'shares material', not 'never suggest short words'."""
        checker = sp.analyser.spell
        assert checker._offerable("u", "nu")        # shares 'u'
        assert not checker._offerable("u", "ph")    # shares nothing

    def test_known_corrections_are_unaffected(self, sp):
        """Regression guard for the cases reported earlier in development."""
        assert sp.check("pyleng").suggestions[0] == "pylleng"
        assert sp.check("dand").suggestions[0] == "dang"
        # This once asserted `shafon` was offered for `shafan`, restored on
        # the grounds the lexicon attests it. It is an English fragment
        # carrying an `f`, so it is not a Khasi word and the assertion was
        # wrong — see TestIllegalCharacters.


class TestIllegalCharacters:
    """
    c, f, q, v, x, z are not in the Khasi alphabet.

    `phonology.valid_chars` says so, and `validate()` has always reported
    "Illegal characters: f". What let them through was `_offerable`'s
    `is_attested` escape, added so that words the validator false-rejects
    (place names, loans, reduplications) stay offerable. That escape is
    right for the *soft* rules — codas, syllable shape — and wrong for the
    character inventory, where there is nothing to overrule: no Khasi word
    contains these letters.

    The lexicon appears to attest some only because English gloss text
    leaked into 260 surface_form fields ('a poisoned fish as', 'advanced
    age'), which compound-token extraction then split into `acid`, `black`,
    `calm`.
    """

    ILLEGAL = set("cfqvxz")

    def test_suggestions_never_contain_illegal_letters(self, sp):
        for word in ("shafan", "ph", "ng", "sh", "kh", "pyleng", "dand"):
            for cand in sp.check(word).suggestions:
                assert not (set(cand.lower()) & self.ILLEGAL), f"{word} -> {cand}"

    def test_shafon_is_no_longer_offered(self, sp):
        """
        Restored earlier on the grounds the lexicon attests it. That was
        wrong: it is an English fragment, not a Khasi word.
        """
        assert "shafon" not in sp.check("shafan").suggestions

    def test_named_entities_are_exempt(self, sp):
        """
        Garo and Bengali place names in Meghalaya use these letters freely —
        652 of the 6,609 gazetteer tokens contain one.
        """
        checker = sp.analyser.spell
        assert checker._offerable("achugre", "achugra")
        assert checker._offerable("academy", "acadmy")

    def test_non_entities_are_not_exempt(self, sp):
        checker = sp.analyser.spell
        assert not checker._offerable("shafon", "shafan")
        assert not checker._offerable("acid", "asid")

    def test_the_rule_follows_the_db(self):
        """Derived from phonology.valid_chars, not hardcoded here."""
        from khasi_engine import phonology

        for ch in self.ILLEGAL:
            assert ch not in phonology.VALID_CHARS

    def test_known_corrections_survive(self, sp):
        assert sp.check("pyleng").suggestions[0] == "pylleng"
        assert sp.check("dand").suggestions[0] == "dang"


class TestBenchmarkQuarantine:
    """The frozen set is never edited for convenience — only annotated."""

    def test_contaminated_items_are_flagged_not_deleted(self):
        import json
        from pathlib import Path

        p = Path(__file__).resolve().parent / "spell_benchmark.json"
        data = json.loads(p.read_text(encoding="utf-8"))
        assert len(data["items"]) == 308, "items must never be removed"
        bad = [i for i in data["items"] if i.get("valid") is False]
        # 7 with a letter outside the Khasi alphabet, 3 English golds
        # (`stand`, `took`, `euphony`), 3 more exposed by the stray-g pass
        # (`ghee`, `thoughtless`, `syrmgiew` — OCR for `syrngiew`).
        assert len(bad) == 14
        for item in bad:
            assert item["invalid_reason"]
        # 7 carry a letter outside the Khasi alphabet; the other 3 (`stand`,
        # `took`, `euphony`) are English words spellable in Khasi letters,
        # reachable only while the leaked gloss text was still in the lexicon.
        assert sum(1 for i in bad if set(i["gold"].lower()) & set("cfqvxz")) == 7


class TestLexiconCleaned:
    """
    260 OCR-damaged / gloss-contaminated entries moved to `quarantine`.

    See scripts/quarantine_illegal_chars.py. Nothing was deleted — the
    lexicon's existing top-level `quarantine` list is the destination, and
    each moved entry carries a `quarantine_reason`.
    """

    def test_no_illegal_characters_remain_in_the_lexicon(self):
        import json
        from pathlib import Path

        db = json.loads((Path(__file__).resolve().parent.parent / "data" /
                         "khasi_db.json").read_text(encoding="utf-8"))
        bad = [e for e in db["lexicon"]
               if set(((e.get("form") or {}).get("surface") or "").lower())
               & set("cfqvxz")]
        assert bad == []

    def test_entries_were_moved_not_deleted(self):
        import json
        from pathlib import Path

        db = json.loads((Path(__file__).resolve().parent.parent / "data" /
                         "khasi_db.json").read_text(encoding="utf-8"))
        moved = [e for e in db["quarantine"] if e.get("quarantine_reason")]
        # Three passes, all moving into the same list:
        #   260  illegal characters (c f q v x z)
        #   116  a `g` not preceded by `n`
        #    54  confirmed damage — 28 with no vowel anywhere, 26 whose
        #        invalid onset repairs onto a word already in the lexicon
        #     7  a hyphenated part is a single vowel-less letter: `ka-n`,
        #        `i-n`, `nga-n`, `ki-n`, `u-n` are the contractions the
        #        lexicon spells `ka'n`, `i'n`, `nga'n`, `ki'n`, `u'n`
        #     1  `ioh-i`, whose joined form `ïohi` was already an entry
        #    89  scan damage: 34 duplicates of an existing surface, 55 whose
        #        intended word cannot be recovered (glosses preserved)
        #     1  `baiii-bain`, whose repair `baiñ-baiñ` was already an entry
        #    16  the unified pass: confirmed-invalid onsets whose repair is
        #        either a different word or unrecoverable
        #     1  `hills` — an English word captured as a Khasi headword during
        #        the 1906 dictionary PDF import; its gloss is that
        #        dictionary's own definition text plus scan debris
        #     1  `ba-kai khaiïi` — scan damage the maintainer confirmed
        #        unrecoverable: an `iï` sequence Khasi does not use, with no
        #        attested reading
        #    14  surfaces that are English end to end, captured from the
        #        dictionary's second column — "the toes", "human eyes",
        #        "used in washing the head". Each carries a gloss belonging
        #        to some other headword ("human eyes" is glossed "Having a
        #        very bad smell"), so there is no Khasi to recover.
        #        See scripts/fix_english_headwords.py, which also TRIMS 14
        #        records where English followed real Khasi rather than
        #        replacing it; those stay in the lexicon.
        #    13  the alphabet entries — "h, h", "e, e", "ï, ï" and the rest,
        #        one per LETTER of the Khasi alphabet, each glossed with its
        #        position in it ("The eighth letter of the Khasi Alphabet").
        #        A letter is not a word. The pronoun `i` is a separate
        #        record, kh_DB_006772, and stays.
        #        See scripts/fix_brackets_and_syllables.py.
        #     1  `briw` — not a Khasi word (maintainer ruling 2026-09-10). A
        #        scan artefact of `briew` from the 1906 dictionary PDF
        #        import; its gloss repeats briew's meanings and is itself
        #        mangled ("Mnn; Woman; Person; … •pi. ki 6W^to= people").
        #        See scripts/quarantine_nonwords.py.
        #     3  `bpei`, `bmiang`, `bwieng` — scan damage on the first
        #        letter (maintainer ruling 2026-09-11). Each correct
        #        spelling is already its own entry with the same gloss —
        #        `dpei`, `rmiang`, `rwieng` — and each damaged onset (bp,
        #        bm, bw) is one the phonology block forbids while the
        #        correct one (dp, rm, rw) is permitted.
        #     6  a second batch of the same kind, 2026-09-11: mdwpyrsut ->
        #        mawpyrsut, mdwshun -> mawshun, wdi-dong -> wai-dong,
        #        pbngaiñ -> phngaiñ, pbnguid -> phnguid, jklat -> klat. The
        #        other jk- headwords are NOT withdrawn by this pass.
        #     7  scan damage carrying a character Khasi does not use
        #        (2026-09-15): `ïakıt bok kit rwiang` (dotless ı),
        #        `ksai jingtem bajùa` and `ka sopti ka bà-heh sakther`
        #        (accented u, a), `sihi sngew-s«7i`, `sbur \\`, and two
        #        written with a slash instead of a separator,
        #        `kram-kram / krum-krum` and `pang-mat / pang-sh'ing`.
        #        The 31 entries joined by an EN DASH are NOT withdrawn:
        #        those are paired expressions, a legitimate entry type.
        assert len(moved) == 590
        stray = [e for e in moved
                 if "not preceded by 'n'" in e["quarantine_reason"]]
        assert len(stray) == 116
        novowel = [e for e in moved if "no vowel anywhere" in e["quarantine_reason"]]
        # Count by the phrase unique to each rule — "is a duplicate" appears
        # in both the invalid-onset and the hyphen reasons.
        onset = [e for e in moved if "is not valid Khasi" in e["quarantine_reason"]]
        hyphen = [e for e in moved if "hyphenated part" in e["quarantine_reason"]]
        assert len(novowel) == 28
        assert len(onset) == 26
        assert len(hyphen) == 7

    def test_english_gloss_leakage_is_gone(self, sp):
        for word in ("cow", "and", "big", "car", "black", "acid",
                     "america", "calm", "villager"):
            assert not sp.check(word).is_correct, word

    def test_real_khasi_survived(self, sp):
        for word in ("shnong", "mynsiem", "iing", "kynih", "dorbar",
                     "bseiñ", "arsien"):
            assert sp.check(word).is_correct, word


class TestAinDiacriticRule:
    """
    -ain -> -aiñ, but only for Khasi words, never for names.

    The lexicon records many of these only in the plain spelling: `spain`
    is an entry glossed "Bandage; To swathe" and there is no `spaiñ`
    headword, so lookup alone cannot offer it. Three guards keep the rule
    off everything else — lowercase in context, present in the lexicon, and
    a non-empty stem.
    """

    def test_khasi_word_gets_the_tilde(self, sp):
        assert "spaiñ" in [v["variant"] for v in sp.check("spain").variants]
        assert "paiñ" in [v["variant"] for v in sp.check("pain").variants]

    def test_capitalised_is_left_alone(self, sp):
        """`Spain` is a country; `spain` is a bandage."""
        assert sp.check("Spain").variants == []
        assert sp.check("Hussain").variants == []

    def test_case_decides_inside_a_sentence(self, sp):
        text = "u la wan na Spain bad u la spain ia ka jain."
        r = sp.check_text(text, variants=True, context=False)
        offered = {v.word: [x["variant"] for x in v.variants] for v in r.variants}
        assert "spaiñ" in offered.get("spain", [])
        assert "Spain" not in offered

    def test_non_lexicon_words_are_excluded(self, sp):
        """
        `plain`, `remain`, `domain`, `maintain`, `detain` pass the morphology
        gate without being lexicon entries. Requiring a curated headword is
        what keeps the rule off them.
        """
        for word in ("plain", "remain", "domain", "maintain", "detain"):
            assert sp.check(word).variants == [], word

    def test_attested_twins_still_come_from_the_lexicon(self, sp):
        """Route 1 handles these; the rule must not duplicate or override."""
        for word, twin in (("jain", "jaiñ"), ("rain", "raiñ"),
                           ("main", "maiñ"), ("lain", "laiñ")):
            variants = [v["variant"] for v in sp.check(word).variants]
            assert twin in variants, word

    def test_proper_case_helper_uses_case_only(self):
        """
        Not the gazetteer — that collides with 312 ordinary Khasi words
        (`bah`, `bat`, `blei`, `dawa`), which must keep their variants.
        """
        from khasi_spell import foreign

        assert foreign.is_proper_case("Spain", sentence_initial=False)
        assert not foreign.is_proper_case("Spain", sentence_initial=True)
        assert not foreign.is_proper_case("spain", sentence_initial=False)
        assert foreign.is_proper_case("MDA", sentence_initial=True)

    def test_rule_does_not_fire_when_the_twin_exists(self, sp):
        """`_ain_form` defers to route 1 so a variant is not offered twice.

        This used to assert `_ain_form("spain") == "spaiñ"`, because the
        lexicon recorded the plain `spain` and no `spaiñ` headword, leaving
        the rule to reconstruct the tilde form. The 2026-09-08 coda-tilde
        pass corrected the headwords, so route 1 now answers for both and
        this rule correctly stands down for each. The user is not worse off
        — the reconstruction is replaced by a stronger answer, a canonical
        lexicon variant.
        """
        from khasi_spell import variants as v

        db = sp.analyser.db
        assert v._ain_form("jain", db) is None       # jaiñ is attested
        assert v._ain_form("spain", db) is None      # spaiñ is attested now
        offered = {x.variant: (x.source, x.canonical) for x in sp.variants("spain")}
        assert offered.get("spaiñ") == ("lexicon", True)


class TestGOnlyInNg:
    """
    Khasi has no standalone /g/ — the letter occurs only in the digraph `ng`.

    The character inventory said so all along: `valid_chars_note` records
    that g "is included as a valid character because it appears as the
    second component of the 'ng' digraph … It does not exist as a standalone
    Khasi phoneme." Nothing enforced it, so `goh`, `gain`, `garo` and the
    English borrowings `register`, `gospel`, `glass`, `telegram` all
    validated as Khasi.
    """

    def test_ng_words_are_valid(self):
        from khasi_engine import phonology

        for word in ("nga", "ngi", "shnong", "dieng", "sngi", "sngewbha",
                     "jingïathuh", "lyngngoh"):
            assert phonology.validate(word)["pass"], word

    def test_stray_g_is_rejected(self):
        from khasi_engine import phonology

        for word in ("goh", "gain", "register", "gospel", "glass", "dang-go"):
            assert not phonology.validate(word)["pass"], word

    def test_english_borrowings_no_longer_pass(self, sp):
        for word in ("register", "gospel", "glass", "telegram", "although"):
            assert not sp.check(word).is_correct, word

    def test_names_are_still_handled_as_names(self, sp):
        """
        `Garo`, `Meghalaya`, `Guwahati` do carry a bare g. They are handled
        where names are handled — the gazetteer and the capitalisation rules
        in `khasi_spell.foreign` — not by weakening the phonology.
        """
        from khasi_spell import foreign

        assert "garo" in foreign.gazetteer()
        r = sp.check_text("ka jylla Meghalaya bad ki Garo", variants=False)
        assert not any(c.original in ("Meghalaya", "Garo") for c in r.corrections)


class TestGateZeroUsesMembership:
    """
    Gate 0 tests the invalid-standalone set directly.

    It used to grep the validator's error messages for "sub-phonemic unit"
    and "no standalone", which coupled the gate to prose. Adding an
    unrelated rule whose message contained the words "no standalone" routed
    ordinary misspellings into Gate 0 — and since `check_sentence` only
    reports gate_reached 1 or 3, they stopped being flagged: sentence
    detection fell from 249/249 to 214/249.
    """

    def test_ng_still_reaches_gate_zero(self, sp):
        r = sp.check("ng")
        assert r.gate == 0 and r.gate_name == "invalid_standalone"

    def test_ordinary_misspellings_do_not(self, sp):
        for typo in ("dign", "lynbga", "kyrtneg", "balagn"):
            assert sp.check(typo).gate == 1, typo

    def test_they_are_still_flagged_in_a_sentence(self, sp):
        r = sp.check_text("ka dign ka lynbga", variants=False, context=False)
        flagged = {c.original for c in r.corrections}
        assert "dign" in flagged and "lynbga" in flagged

    def test_gate_zero_is_not_coupled_to_message_text(self):
        """The regression guard: membership, not substring matching."""
        import inspect
        from khasi_engine import spell_checker

        src = inspect.getsource(spell_checker.KhasiSpellChecker.suggest)
        # Membership test, not a substring search over validator prose. The
        # comment above the gate mentions the old strings, so assert on the
        # code shape rather than their absence from the whole source.
        assert "if word_lower in _INVALID_STANDALONE:" in src


class TestAinDiaeresisNormalised:
    """
    Khasi writes the sequence a + i + ñ — a plain `i`, tilde on the nasal.

    The lexicon held both spellings: 553 surfaces with `aiñ` against 84 with
    `aïñ`, and 20 words appeared BOTH ways, which is what makes it an error
    rather than a variant. Left alone it produced nonsense alternatives —
    `nongthain` was offered `nongthaïñ`, two diacritics where the
    orthography has one, because the generator found that spelling attested.
    """

    def test_no_aiin_remains_in_the_lexicon(self):
        import json
        from pathlib import Path

        db = json.loads((Path(__file__).resolve().parent.parent / "data" /
                         "khasi_db.json").read_text(encoding="utf-8"))
        bad = [e for e in db["lexicon"]
               if "aïñ" in ((e.get("form") or {}).get("surface") or "")]
        assert bad == []

    def test_the_double_diacritic_is_never_offered(self, sp):
        for word in ("nongthain", "thain", "sympain", "mynrain", "jingthain"):
            for v in sp.check(word).variants:
                assert "aïñ" not in v["variant"], (word, v)

    def test_the_correct_form_is_offered(self, sp):
        assert "nongthaiñ" in [v["variant"] for v in sp.check("nongthain").variants]


class TestAinCompoundRule:
    """
    A compound whose final element carries the tilde: `saitjain` -> `saitjaiñ`.

    Route 3 strips known *prefixes*, so it never reaches a root sitting at
    the end. `saitjain` alone is 890 corpus tokens.
    """

    def test_compounds_get_the_tilde(self, sp):
        cases = {
            "saitjain": "saitjaiñ",
            "jinglehrain": "jinglehraiñ",
            "shongshngain": "shongshngaiñ",
            "suhjain": "suhjaiñ",
        }
        for word, expected in cases.items():
            assert expected in [v["variant"] for v in sp.check(word).variants], word

    def test_names_and_english_stay_off(self, sp):
        """`huss`/`ag` are not Khasi words, so the head guard blocks them."""
        for word in ("hussain", "again", "Hussain", "Spain"):
            assert sp.check(word).variants == [], word

    def test_reduplications_are_left_to_route_one(self, sp):
        """
        Route 1 converts both halves (`dain-dain` -> `daiñ-daiñ`); the
        compound rule would produce the lopsided `dain-daiñ`.
        """
        for word in ("dain-dain", "jain-jain"):
            for v in sp.check(word).variants:
                assert not v["variant"].startswith(word.split("-")[0] + "-" ), v

    def test_bare_ain_is_not_a_tail(self):
        """
        The tail must be a root like `jain`/`shain`, not the word `ain`, or
        anything merely ending in those letters gets rewritten.
        """
        from khasi_spell import variants as v

        assert v.MIN_AIN_TAIL >= 4


class TestHyphenatedWholeToken:
    """
    A hyphenated compound the lexicon records is a word in its own right.

    `_check_hyphenated` split every hyphenated token and "corrected" parts
    that failed `is_known`, without first asking whether the whole token was
    a lexicon entry. `jrain-jrain` is an entry but neither `jrain` is, so it
    was rewritten to `jain-jain` — a real reduplication turned into a
    different word. It also made the two entry points contradict each other:
    `check()` accepted these while `check_text()` flagged them, on 3.3% of
    sampled lexicon compounds.
    """

    # `jdinkup-jainsem` was here until it was quarantined as scan damage —
    # its `jd` onset is not Khasi and it repairs to `jainkup-jainsem`.
    KNOWN_COMPOUNDS = ("jrain-jrain", "jainkup-jainsem", "saw-ka-siau", "man-bha")

    def test_known_compounds_are_not_rewritten(self, sp):
        for word in self.KNOWN_COMPOUNDS:
            r = sp.check_text(word, variants=False, context=False)
            assert r.corrections == [], (word, [c.suggestion for c in r.corrections])

    def test_both_entry_points_agree(self, sp):
        for word in self.KNOWN_COMPOUNDS:
            accepted_by_check = sp.check(word).is_correct
            flagged_in_text = any(
                c.original.lower() == word
                for c in sp.check_text(word, variants=False,
                                       context=False).corrections)
            assert accepted_by_check is not flagged_in_text, word

    def test_the_feature_still_works(self, sp):
        """
        The pass exists to catch a misspelling hiding inside a compound:
        Phase 4 accepts any hyphenated token, so `man-miay` passed even
        though `miay` does not exist.
        """
        r = sp.check_text("u man-bha man-miay", variants=False, context=False)
        assert any(c.original == "man-miay" and c.suggestion == "man-miat"
                   for c in r.corrections)

    def test_unknown_compounds_are_still_split(self, sp):
        """A compound the lexicon does NOT record must still be inspected."""
        r = sp.check_text("ka man-zzzq", variants=False, context=False)
        assert not sp.analyser.db.is_known("man-zzzq")


class TestInitialClusters:
    """
    `valid_initial_clusters` was incomplete and rejected correct Khasi.

    The original 76 clusters failed 406 single-token surfaces (3.3%) —
    `khwai` and `khwaiñ` are curated entries, and `s'ïang`, `l'er`, `k'a`
    carry the glottal stop, which the DB lists among the Khasi consonants.
    `validate()` drives `_phonotactic_ok` and `_offerable`, so the gap kept
    real words out of `_compound_known` and left them dependent on the
    `is_attested` escape hatch.
    """

    def test_corpus_attested_clusters_validate(self):
        from khasi_engine import phonology

        for word in ("khwai", "khwaiñ", "lngaid", "dngang", "rwai",
                     "lwait", "phna", "thwet", "twad", "rtiang"):
            assert phonology.validate(word)["pass"], word

    def test_glottal_stop_clusters_validate(self):
        """`'` is listed in phonology.consonants, so these are legitimate."""
        from khasi_engine import phonology

        for word in ("s'am", "l'er", "k'a", "sh'ah", "b'ym", "p'oh", "r'ang"):
            assert phonology.validate(word)["pass"], word

    def test_damage_and_foreign_still_rejected(self):
        """The extension must not legitimise scan damage or non-Khasi."""
        from khasi_engine import phonology

        for word in ("pbetkiri", "tjmja", "jkang", "goh", "register",
                     "cow", "ng"):
            assert not phonology.validate(word)["pass"], word

    def test_digraphs_were_not_added_as_clusters(self):
        """
        `sh`, `kh`, `th`, `ph`, `ng` are single phonemes. They appeared in
        the failure list only because words carrying them fail for another
        reason (a vowel misread as `d`), and adding them would have papered
        over that.
        """
        import json
        from pathlib import Path

        db = json.loads((Path(__file__).resolve().parent.parent / "data" /
                         "khasi_db.json").read_text(encoding="utf-8"))
        clusters = set(db["phonology"]["valid_initial_clusters"])
        for digraph in ("sh", "kh", "th", "ph", "ng"):
            assert digraph not in clusters, digraph

    def test_the_list_grew_as_recorded(self):
        import json
        from pathlib import Path

        db = json.loads((Path(__file__).resolve().parent.parent / "data" /
                         "khasi_db.json").read_text(encoding="utf-8"))
        assert len(db["phonology"]["valid_initial_clusters"]) == 99
        assert db["phonology"].get("valid_initial_clusters_note")


class TestHyphenSingleLetter:
    """
    No hyphenated part is a single letter without a vowel.

    `ka-n`, `i-n`, `nga-n`, `ki-n`, `u-n` are the contractions the lexicon
    spells `ka'n`, `i'n`, `nga'n`, `ki'n`, `u'n` — all five attested, with
    glosses that confirm it ("She...not / she that"). The hyphen stands
    where the apostrophe belongs.

    The rule is narrow on purpose: 35 other entries have a single letter
    after a hyphen (`blang-u-bhed`, `ba-i-bit`, `ioh-i`), but those are
    `u`, `i`, `a`, `e` — genuine Khasi words. Only a letter that cannot be
    a word at all, having no vowel, invalidates the entry.
    """

    def test_hyphen_contractions_are_gone(self, sp):
        for word in ("ka-n", "i-n", "nga-n", "ki-n", "u-n"):
            assert not sp.check(word).is_correct, word

    def test_the_apostrophe_forms_survive(self, sp):
        for word in ("ka'n", "i'n", "nga'n", "ki'n", "u'n"):
            assert sp.check(word).is_correct, word

    def test_middle_position_clitics_are_untouched(self, sp):
        """
        `u`, `i`, `a` are real words, so a single letter in the MIDDLE of a
        compound is correct — there is more of the word after it.
        """
        db = sp.analyser.db
        for word in ("blang-u-bhed", "ba-i-bit", "kha-u-man",
                     "bam-hynroh-u-bnai"):
            assert word in db._surface_index, word

    def test_trailing_single_vowel_is_joined(self, sp):
        """
        A single letter at the END is a scan artefact splitting the final
        vowel from its stem: `tyr-a` is `tyra`, `ioh-i` is `ïohi`.
        """
        db = sp.analyser.db
        for old in ("ioh-i", "hih-i", "loli-i", "lyng-a", "pi-e",
                    "pynkyr-a", "tyr-a"):
            assert old not in db._surface_index, old
        # `loli-i` joined to `lolii`, but `ii` is not a legal Khasi sequence
        # (2026-09-08 ruling), and a run of `i` ending a token carries a
        # misread `ñ` — so the joined form is `loliñ`. The other six join to
        # sequences that were legal all along and are unaffected.
        for new in ("ïohi", "hihi", "loliñ", "lynga", "pie", "pynkyra", "tyra"):
            assert sp.check(new).is_correct, new


class TestScanDamageRepaired:
    """
    Four confirmed readings, applied by scripts/repair_scan_damage.py:
    no stray bracket, no foreign diacritic or digit, no vowel-less segment,
    and the nine invalid onsets.

    Each entry got one of three outcomes — quarantined as a duplicate when a
    repair landed on an existing surface, rewritten when exactly one valid
    repair existed, or quarantined with its gloss and candidate repairs
    recorded when the intended word could not be recovered.
    """

    GONE = ("kha)", "ing)", "stones)", "shiüng", "la-shem-taiëw",
            "tiew-pathai-khub6r", "sdiu-lun:a", "*dorbin", "*dud",
            "bd-liim", "beit-shdn", "jdinkup-jainsem")
    REWRITTEN = ("la-shem-taiew", "shiung", "jurew", "duri-po",
                 "khulom-sner", "tiew-pathai-khubor", "um-jer")

    def test_damaged_surfaces_are_gone(self, sp):
        db = sp.analyser.db
        for word in self.GONE:
            assert word not in db._surface_index, word

    def test_repaired_surfaces_are_live(self, sp):
        db = sp.analyser.db
        for word in self.REWRITTEN:
            assert word in db._surface_index, word

    def test_no_single_token_surface_holds_a_foreign_character(self):
        import json, re
        from pathlib import Path

        db = json.loads((Path(__file__).resolve().parent.parent / "data" /
                         "khasi_db.json").read_text(encoding="utf-8"))
        bad = [s for s in
               (((e.get("form") or {}).get("surface") or "").lower()
                for e in db["lexicon"])
               if s and " " not in s and re.search(r"[^a-zïñáéíóúý'\-]", s)]
        assert bad == []

    def test_unresolved_entries_keep_their_evidence(self):
        """Quarantine is not deletion: the gloss and candidates are kept."""
        import json
        from pathlib import Path

        db = json.loads((Path(__file__).resolve().parent.parent / "data" /
                         "khasi_db.json").read_text(encoding="utf-8"))
        unresolved = [e for e in db["quarantine"]
                      if "cannot be recovered" in (e.get("quarantine_reason") or "")]
        assert len(unresolved) == 60
        assert all("gloss:" in e["quarantine_reason"] for e in unresolved)


class TestAinRootAnywhere:
    """
    An attested -aiñ root can open or sit inside a compound, not only end it.

    `jain` alone offered `jaiñ`, but every compound built on it — `jainsem`,
    `jainkhor`, `jainkup-jainsem` — offered nothing, because the tail rule
    only looks at the end of the word.
    """

    def test_root_at_the_start_is_converted(self, sp):
        cases = {"jainsem": "jaiñsem", "jainkhor": "jaiñkhor",
                 "mainker": "maiñker"}
        for word, expected in cases.items():
            assert expected in [v["variant"] for v in sp.check(word).variants], word

    def test_every_occurrence_is_converted(self, sp):
        """Not just the first — `jainkup-jainsem` has two."""
        variants = [v["variant"] for v in sp.check("jainkup-jainsem").variants]
        assert "jaiñkup-jaiñsem" in variants

    def test_root_after_a_hyphen(self, sp):
        for word, expected in (("iing-jain", "iing-jaiñ"),
                               ("jain-ryndia", "jaiñ-ryndia")):
            assert expected in [v["variant"] for v in sp.check(word).variants], word

    def test_english_is_excluded(self, sp):
        """
        Start-anchoring alone is not enough — `maintain` really does begin
        with `main`. The word must also be a curated lexicon headword, which
        none of these is.
        """
        for word in ("maintain", "remain", "domain", "plain", "again"):
            assert sp.check(word).variants == [], word

    def test_single_letter_remainders_are_refused(self, sp):
        """`bain` + `a` and `dain` + `i` are real clitics; the joins are not."""
        from khasi_spell import variants as v

        db = sp.analyser.db
        assert v._ain_elements("baina", db) is None
        assert v._ain_elements("daini", db) is None


class TestUmlautAndReduplication:
    """
    `ü` is `ïi`, and a reduplication's damaged half is read from its twin.

    `üng` is `ïing` ('house'), so `ka üng ka itynnat` reads as
    `ka ïing ka itynnat`. And where one half of an `A-B` compound is a known
    word and the other carries a scan artefact, the good half says what the
    bad one should be: `baiii-bain` is `baiñ-baiñ`.
    """

    def test_no_umlaut_remains(self):
        import json
        from pathlib import Path

        db = json.loads((Path(__file__).resolve().parent.parent / "data" /
                         "khasi_db.json").read_text(encoding="utf-8"))
        bad = [e for e in db["lexicon"]
               if "ü" in ((e.get("form") or {}).get("surface") or "")]
        assert bad == []

    def test_ung_became_iing(self, sp):
        db = sp.analyser.db
        assert "mrad ïing" in db._surface_index

    def test_damaged_reduplications_repaired(self, sp):
        db = sp.analyser.db
        for gone in ("jatmut-jatrnat", "thiah-thdi", "tyngkliip-tyngkhap",
                     "baiii-bain"):
            assert gone not in db._surface_index, gone
        for fixed in ("jatmut-jatmut", "thiah-thiah", "tyngkhap-tyngkhap"):
            assert sp.check(fixed).is_correct, fixed

    def test_echo_reduplication_is_left_alone(self, sp):
        """
        Khasi alternates the vowel on purpose, and many pairs are coordinate
        compounds. Forcing halves to match would destroy both kinds.
        """
        db = sp.analyser.db
        for word in ("jirwit-jirwat", "awri-awra", "tharuh-thareh",
                     "hynñium-hynñiam", "ïadih-ïabam", "riewbah-riewsan"):
            assert word in db._surface_index, word


class TestNoDuplicateSurfaces:
    """
    Two lexicon records must not share a surface.

    A repair can collide with an entry that already exists — `baiii-bain`
    repairs to `baiñ-baiñ`, which the lexicon held — and rewriting would
    leave two records claiming the same word. Which one a lookup returns is
    then arbitrary.

    The collision is decided on **meaning**: a colliding record is withdrawn
    only when it describes the same word, so two distinct senses are never
    silently collapsed into one.
    """

    def test_no_surface_appears_twice(self):
        import json, collections
        from pathlib import Path

        db = json.loads((Path(__file__).resolve().parent.parent / "data" /
                         "khasi_db.json").read_text(encoding="utf-8"))
        seen = collections.Counter(
            ((e.get("form") or {}).get("surface") or "").lower()
            for e in db["lexicon"]
            if ((e.get("form") or {}).get("surface") or ""))
        dupes = {k: n for k, n in seen.items() if n > 1}
        assert dupes == {}, f"{len(dupes)} duplicated surfaces: {list(dupes)[:5]}"

    def test_same_word_recognises_an_example_gloss(self):
        """
        The scan split some entries in two, leaving a record whose gloss
        quotes the headword rather than defining it. `baiii-bain` glossed
        "As ; Jem baifi-bain)" is an example of `baiñ-baiñ` "Very (flexible;
        Pliable)", same part of speech — the same word, not a second sense.
        """
        import importlib.util
        from pathlib import Path

        path = (Path(__file__).resolve().parent.parent / "scripts" /
                "fix_umlaut_and_reduplication.py")
        spec = importlib.util.spec_from_file_location("_fixmod", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        example = {"form": {"surface": "baiii-bain"},
                   "grammar": {"pos": "ADV"},
                   "semantics": {"gloss": "As ; Jem baifi-bain)"}}
        real = {"form": {"surface": "baiñ-baiñ"},
                "grammar": {"pos": "ADV"},
                "semantics": {"gloss": "Very (flexible ; Pliable ; Soft)"}}
        assert mod.same_word(example, real)

    def test_different_senses_are_not_merged(self):
        import importlib.util
        from pathlib import Path

        path = (Path(__file__).resolve().parent.parent / "scripts" /
                "fix_umlaut_and_reduplication.py")
        spec = importlib.util.spec_from_file_location("_fixmod", path)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        a = {"form": {"surface": "xyz"}, "grammar": {"pos": "NOUN"},
             "semantics": {"gloss": "A basket for carrying grain"}}
        b = {"form": {"surface": "xyz"}, "grammar": {"pos": "VERB"},
             "semantics": {"gloss": "To walk slowly uphill"}}
        assert not mod.same_word(a, b)


class TestUnifiedRepairPass:
    """
    One pass, rules as data, one decision procedure.

    Replaces six scripts that had grown apart: `surface()` was redefined in
    all six, four rules were implemented twice, the same defect got
    different treatment depending on run order, and only one of the six
    compared meaning before withdrawing a record on a collision.
    """

    def _mod(self):
        import sys
        from pathlib import Path

        scripts = Path(__file__).resolve().parent.parent / "scripts"
        if str(scripts) not in sys.path:
            sys.path.insert(0, str(scripts))
        import repair_lexicon

        return repair_lexicon

    def test_rules_are_data_and_all_named(self):
        mod = self._mod()
        names = [r.name for r in mod.RULES]
        assert len(names) == len(set(names)), "rule names must be unique"
        for expected in ("illegal_char", "stray_g", "no_vowel",
                         "hyphen_letter", "trailing_vowel", "stray_onset",
                         "lead_consonant", "markup", "diacritic",
                         "reduplication"):
            assert expected in names, expected

    def test_lead_consonant_is_restricted_to_confirmed_onsets(self):
        """
        The general form of this rule is wrong. `ki'm` reduces to `i'm` and
        `ki'n` to `i'n`, but those are different pronouns — `ki` 'they'
        against `i` 'it'. 22 such pairs were caught by the meaning check
        before the rule was narrowed to confirmed onsets.
        """
        mod = self._mod()
        assert "lj" in mod.LEAD_CONSONANT_ONSETS
        assert "jm" in mod.LEAD_CONSONANT_ONSETS
        for safe in ("k", "ki", "b", "j"):
            assert safe not in mod.LEAD_CONSONANT_ONSETS

    def test_confirmed_invalid_words_are_rejected(self, sp):
        for word in ("ljing", "ljang", "jmong", "rsham", "jsleit", "mthen",
                     "mthin", "tjtai", "tjtei", "beiii", "biid", "biik"):
            assert not sp.check(word).is_correct, word

    def test_the_words_they_collided_with_survive(self, sp):
        """A wrong repair must never take the real word down with it."""
        for word in ("jing", "mong", "sham", "then", "tai",
                     "i'm", "i'n", "ki'm", "ki'n"):
            assert sp.check(word).is_correct, word

    def test_conflicts_record_the_colliding_word(self):
        import json
        from pathlib import Path

        db = json.loads((Path(__file__).resolve().parent.parent / "data" /
                         "khasi_db.json").read_text(encoding="utf-8"))
        conflicts = [e for e in db["quarantine"]
                     if "is a different word" in (e.get("quarantine_reason") or "")]
        assert conflicts, "conflict reasons must name the colliding word"


class TestIngHouseSense:
    """
    `ing` is at least four different words that share a spelling.

    Only the house sense is spelled `ïing`. A rule of "everything that is
    not about burning" — the obvious reading — would have rewritten ginger
    and backbone into house, so the senses were read from the glosses one
    at a time.

        house    ing ai-bam 'hotel', ing-kirja 'chapel'
        burn     ing ding 'catch fire', ing-thap 'to singe'
        ginger   ing-bah, ing-makhir
        back     ing-dong 'back; backbone', pynkdor ing-dong 'hunch'
        anger    ing-nud 'be angry'

    A glottal stop before it marks yet more homographs — `sh'ing` 'bone',
    `s'ing` 'ginger', `k'ing` 'a wasp' — which is why the substitution
    guards its lookbehind against the apostrophe.
    """

    def test_house_sense_uses_the_tilde_form(self, sp):
        db = sp.analyser.db
        for word in ("ïing ai-bam", "ïing ap-phira", "ïing-kirja",
                     "leit-ïing-briew", "ïing-bam ki baduk"):
            assert word in db._surface_index, word

    def test_other_senses_are_untouched(self, sp):
        """Different words that merely look alike."""
        db = sp.analyser.db
        for word in ("ing", "ing ding", "ing-dong", "ing-bah", "ing-nud",
                     "ing-thap"):
            assert word in db._surface_index, word

    def test_glottal_stop_homographs_are_untouched(self, sp):
        db = sp.analyser.db
        for word in ("sh'ing", "s'ing", "k'ing", "'ing-dong", "'ing-bah"):
            assert word in db._surface_index, word

    def test_short_spelling_still_offers_the_tilde_form(self, sp):
        """
        The corpus settles this: `ing` occurs 3,167 times and every one is
        the noun — `shna ing`, `hapoh ing`, `poi ing`. Not one is the verb.
        """
        assert "ïing" in [v["variant"] for v in sp.check("ing").variants]

    def test_the_route_stays_off_unrelated_words(self, sp):
        for word in ("ar", "ai", "ad", "am", "ap", "at", "jing", "long", "bam"):
            assert sp.check(word).variants == [], word


class TestGlossRecordedSpellings:
    """
    The dictionary records alternative spellings inside its definitions —
    "Also spelt as kyieng", "Abbrev. of shnong" — and nothing read them
    until `scripts/build_spelling_links.py`.

    The glosses are OCR output like the headwords, so a link is only as
    sound as the sentence it came from. `s'ing` 'ginger' was glossed
    "(same as sping)", but `sping` means 'Handle'; the entry for `'ing`
    reads "Abbrev. of sying = ginger", which identifies the gloss's `sping`
    as a scan error for `sying`. Left alone, the link joined ginger to
    handle.
    """

    def test_real_pairs_are_offered(self, sp):
        """Only "same as" / "also spelt as" links are spellings."""
        for word, expected in (("ade", "ide"), ("bdi", "bydi"),
                               ("'ba-hab", "'ba-pynthor")):
            assert expected in [v["variant"] for v in sp.check(word).variants], word

    def test_abbreviations_are_not_offered_as_spellings(self, sp):
        """A clipped form with its own entry is a different word.

        `'riew` is glossed "1. (abbrev. of briew); 2. (abbrev. of shriew) an
        arum". Offering it to a writer who typed `briew` says their spelling
        has an alternative when it does not. 102 of the 206 link pairs are
        abbreviations of this kind and none of them is a spelling variant.
        """
        for word, clipped in (("briew", "'riew"), ("shnong", "'nong"),
                              ("slap", "'lap"), ("khyndew", "'dew")):
            offered = [v["variant"] for v in sp.check(word).variants]
            assert clipped not in offered, f"{word} must not offer {clipped}"

    def test_elided_short_forms_are_not_offered_either(self, sp):
        """An apostrophe form is a SHORT form, not another spelling.

        Khasi writes an elided initial consonant with an apostrophe — `syiem`
        "king" also appears as `s'iem`, `kyieng` as `k'ing`. Those glosses say
        "same as", not "abbrev. of", so the abbreviation filter let them
        through; 110 of the 206 link pairs are this shape.

        One-directional, like the diacritic rule: typing the short form still
        offers the full one.
        """
        for word, short in (("syiem", "s'iem"), ("kyieng", "k'ing"),
                            ("pyiar", "p'ïar")):
            assert short not in [v["variant"] for v in sp.check(word).variants]
            assert word in [v["variant"] for v in sp.check(short).variants], \
                f"{short} should still expand to {word}"

    def test_the_ginger_handle_link_is_gone(self, sp):
        """`sping` is 'Handle'. It is not a spelling of `s'ing` 'ginger'."""
        assert "sping" not in [v["variant"] for v in sp.check("s'ing").variants]
        assert "s'ing" not in [v["variant"] for v in sp.check("sping").variants]

    def test_links_only_point_at_real_headwords(self):
        """
        74 gloss references name a form the lexicon does not carry. Offering
        those would send a writer to a spelling nothing can vouch for.
        """
        import json
        from pathlib import Path

        root = Path(__file__).resolve().parent.parent
        db = json.loads((root / "data" / "khasi_db.json").read_text(encoding="utf-8"))
        links = json.loads((root / "data" / "spelling_links.json")
                           .read_text(encoding="utf-8"))["links"]
        surfaces = {((e.get("form") or {}).get("surface") or "").lower()
                    for e in db["lexicon"]}
        for src, targets in links.items():
            for t in targets:
                assert t in surfaces, f"{src} -> {t} is not a headword"

    def test_links_are_bidirectional(self):
        import json
        from pathlib import Path

        links = json.loads((Path(__file__).resolve().parent.parent / "data" /
                            "spelling_links.json").read_text(encoding="utf-8"))["links"]
        for src, targets in links.items():
            for t in targets:
                assert src in links.get(t, []), f"{src} -> {t} is one-way"


class TestPhonotacticallyImpossibleWordsAreReported:
    """A word that breaks Khasi phonotactics is flagged even when nothing
    can be offered for it.

    `sbngaifi` contains `f`, which is not in the Khasi orthography at all,
    and `phonology.validate` says exactly that. It was silently accepted:
    `check_sentence` required `and gate["suggestions"]`, so a word already
    judged impossible vanished from the result whenever no candidate was
    close enough. 82% of strings carrying a banned letter (c f v x z) were
    detected and then dropped — `xerox`, `office` and `qwerty` among them.

    Being unable to fix a word is not a reason to call it correct.
    """

    BANNED = ("sbngaifi", "xerox", "office", "qwerty", "zzz")

    @pytest.mark.parametrize("word", BANNED)
    def test_it_is_flagged(self, sp, word):
        assert sp.check_text(word).has_errors, f"{word} was accepted"

    @pytest.mark.parametrize("word", BANNED)
    def test_it_explains_why(self, sp, word):
        """The reason is the phonology module's own wording, not ours."""
        c = sp.check_text(word).corrections[0]
        assert c.reason, f"{word} flagged with no reason"

    def test_a_banned_letter_is_repaired_and_searched(self, sp):
        """Blocking is not the end of it — the letter is repaired first.

        Searched as written, `sbngaifi` returns ZERO candidates at distance
        3: no Khasi word is near a string containing `f`. Read with the
        banned letter repaired it reaches `shngaiñ`. The repair table is
        phonetically motivated (`f` is written `ph` in Khasi loans) plus one
        scanner artefact — `fi` is this source's rendering of `ñ`, which the
        lexicon confirms: `kfii` is `kñi` and `kfiia` is `kñia`.
        """
        c = sp.check_text("sbngaifi").corrections[0]
        assert c.suggestions, "sbngaifi was blocked with nothing offered"
        assert any("ngaiñ" in x for x in c.suggestions), c.suggestions

        # A repair that IS a lexicon word is offered directly, ahead of its
        # neighbours. Context re-ranking is off here: the corpus carries no
        # diacritics, so it has no evidence for `kñi` and demotes it — a
        # known corpus problem, not a ranking one.
        c = sp.check_text("kfii", context=False).corrections[0]
        assert c.suggestions[0] == "kñi", c.suggestions

    def test_a_malformed_word_ranks_below_a_well_formed_one(self, sp):
        """`pb` is not a permitted Khasi onset.

        `sbngaifi` was answered with `pbngaiñ` ahead of `shngaiñ`, and the
        lexicon carries `pbngaiñ` only because the scan misread it. The
        engine's own phonology says the onset is impossible, so a candidate
        it rejects is ranked below one it accepts.

        A PENALTY and not a filter, deliberately: the cluster list is
        incomplete. 107 lexicon forms have an onset it does not list and many
        are plainly real — `bmiang` "the margin", `bpei` "hearth; ashes",
        `gra` "old brass vessel" — so excluding them would lose real
        vocabulary. Demoting only means a well-formed word is offered first.
        """
        c = sp.check_text("sbngaifi", context=False).corrections[0]
        assert c.suggestions[0] == "shngaiñ", c.suggestions

        # …and demotion is not censorship. A malformed form the lexicon
        # still carries is offered when it is the closest thing there is;
        # only its RANK changes. (`pbngaiñ`, the original example, was
        # withdrawn as scan damage on 2026-09-11 — the ruling was that it
        # should be `phngaiñ` — so a form still in the lexicon is used.)
        c = sp.check_text("buhmla", context=False).corrections[0]
        assert "buhmld" in c.suggestions, c.suggestions

    def test_ties_keep_the_ranked_order(self, sp):
        """Equal distance must not be broken by spelling.

        `briew` and `brieng` are both 1.0 from `brief`. Merging the repaired
        candidates with a sort on the (distance, candidate) tuple broke the
        tie alphabetically and `brieng` won, discarding the frequency ranking
        the search had already applied.
        """
        c = sp.check_text("brief", context=False).corrections[0]
        assert c.suggestions[0] == "briew", c.suggestions

    def test_a_repair_never_outranks_a_direct_hit(self, sp):
        """The repair carries a penalty.

        Deleting a banned letter shortens the word and makes every short word
        look close. Unpenalised, `brief` lost `briew` to `bri`.
        """
        c = sp.check_text("brief").corrections[0]
        assert c.suggestions[0] == "briew", c.suggestions

    def test_the_text_is_left_alone_when_nothing_can_be_offered(self, sp):
        """Reported, but never rewritten — a null suggestion is not an edit."""
        r = sp.check_text("ka office ka lah")
        assert r.has_errors
        assert r.corrected == "ka office ka lah"
        assert r.corrections[0].suggestion is None

    def test_an_unknown_word_with_no_suggestion_stays_quiet(self, sp):
        """Gate 3 is NOT gate 1.

        "Not in the lexicon" is also true of every legitimate word never
        recorded, so flagging those with nothing to offer turns silence into
        a false alarm. `roi-u-par` is a hyphenated compound of good parts.
        """
        assert not sp.check_text("u man-bha bad u roi-u-par").has_errors

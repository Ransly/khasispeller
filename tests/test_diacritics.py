"""
Khasi diacritics: ï (U+00EF) and ñ (U+00F1).

The lexicon holds 1,272 words with ï and 1,829 with ñ, so these are not
edge cases. The subtle part is Unicode normalisation: each character has a
precomposed and a decomposed spelling, the lexicon and tokeniser use the
precomposed one, and decomposed input used to split mid-word — turning
'jingïathuh' into 'jing', 'i', 'athuh' and then "correcting" the fragment.
macOS and several compose-key setups emit decomposed text.
"""

import unicodedata

import pytest

from khasi_spell import KhasiSpeller


@pytest.fixture(scope="module")
def sp():
    return KhasiSpeller(eager=True)


@pytest.fixture(scope="module")
def words(sp):
    lex = sp.analyser.db.to_freq_dict()
    i_w = [w for w in lex if "ï" in w and " " not in w and "-" not in w]
    n_w = [w for w in lex if "ñ" in w and " " not in w and "-" not in w]
    assert i_w and n_w, "lexicon has no diacritic words — wrong data file?"
    return i_w, n_w


def test_lexicon_contains_diacritic_words(words):
    i_w, n_w = words
    assert len(i_w) > 100
    assert len(n_w) > 100


def test_diacritic_words_are_accepted(sp, words):
    i_w, n_w = words
    for w in i_w[:5] + n_w[:5]:
        assert sp.is_correct(w), f"{w!r} rejected"


def test_uppercase_diacritics_are_accepted(sp, words):
    """Ï and Ñ at the start of a sentence, and in full caps."""
    i_w, n_w = words
    assert sp.is_correct(i_w[0].capitalize())
    assert sp.is_correct(n_w[0].upper())


@pytest.mark.parametrize("form", ["NFC", "NFD"])
def test_both_unicode_forms_are_accepted(sp, words, form):
    """
    The regression this file exists for. NFD input must not be treated as
    a different word from NFC input.
    """
    word = words[0][0]
    assert sp.is_correct(unicodedata.normalize(form, word))


def test_decomposed_text_is_not_corrupted(sp, words):
    """
    Decomposed input used to split mid-word and produce a bogus correction
    on text that was already right. Nothing may be flagged here.
    """
    i_w, n_w = words
    text = unicodedata.normalize("NFD", f"Ka {i_w[0]} bad ka {n_w[0]} ka bha")
    result = sp.check_text(text)
    assert not result.has_errors, \
        f"decomposed text wrongly flagged: {[c.original for c in result.corrections]}"


def test_offsets_index_the_returned_text(sp, words):
    """
    Normalisation can change length, so spans must index the text the
    result carries — not the caller's original string.
    """
    i_w, _ = words
    text = unicodedata.normalize("NFD", f"Ka shnng bad {i_w[0]} ka bha")
    result = sp.check_text(text)
    for c in result.corrections:
        assert result.text[c.start:c.end] == c.original


def test_suggestions_preserve_diacritics(sp, words):
    """A correction may legitimately need to restore ï or ñ."""
    i_w, _ = words
    target = next((w for w in i_w if len(w) >= 5), i_w[0])
    typo = target.replace("ï", "i")
    if sp.is_correct(typo):
        pytest.skip(f"{typo!r} is itself a valid word — no correction expected")
    assert any("ï" in s for s in sp.suggest(typo)), \
        f"no diacritic-bearing suggestion for {typo!r}"


# ----------------------------------------------------------------------
# Alternative spellings — guidance, not correction
# ----------------------------------------------------------------------

def test_accepted_word_still_offers_the_diacritic_spelling(sp):
    """
    The point of the feature. 'jingiathuh' is accepted — the engine folds
    diacritics at lookup — but the lexicon spells it 'jingïathuh', and a
    writer who wants the standard form was never told.
    """
    r = sp.check("jingiathuh")
    assert r.is_correct, "should still be accepted, not turned into an error"
    assert "jingïathuh" in [v["variant"] for v in r.variants]


def test_whole_word_variant_from_the_lexicon(sp):
    r = sp.check("iathuh")
    assert r.is_correct
    assert "ïathuh" in [v["variant"] for v in r.variants]


def test_n_tilde_variant_is_found(sp):
    """
    The engine's fold map covers ï but not ñ, so 'ain' -> 'aiñ' is
    invisible to the folded index and has to be generated.
    """
    assert "aiñ" in [v.variant for v in sp.variants("ain")]


def test_both_spellings_are_valid(sp):
    """
    Neither spelling is an error. The lexicon records 107 words BOTH ways
    (khwain/khwaiñ, luin/luiñ), so calling either one wrong would contradict
    the lexicon itself.
    """
    for a, b in (("ain", "aiñ"), ("khwain", "khwaiñ"), ("iathuh", "ïathuh")):
        assert sp.is_correct(a), f"{a} rejected"
        assert sp.is_correct(b), f"{b} rejected"


def test_the_diacritic_spelling_is_offered_but_never_withdrawn(sp):
    """One direction only: plain -> diacritic, never the reverse.

    In formal Khasi the diacritic spelling is the correct one. Writing `ï`
    as `i` and `ñ` as `n` is a substitution people make when a keyboard or
    a font gets in the way, and the words the lexicon records both ways are
    that habit written down, not free variation. So a writer who types the
    plain form is shown the diacritic one; a writer who already typed the
    diacritic form is shown nothing, because there is nothing to fix.

    This replaces an earlier test that asserted both directions. Offering
    the downgrade invited a correct spelling to be undone, and "Accept all"
    applied exactly that across a whole document.
    """
    for plain, diacritic in (("ain", "aiñ"), ("khwain", "khwaiñ"),
                             ("iathuh", "ïathuh"), ("jingialehkai", "jingïalehkai")):
        offered = [v["variant"] for v in sp.check(plain).variants]
        assert diacritic in offered, f"{plain} -> {diacritic} missing"
        back = [v["variant"] for v in sp.check(diacritic).variants]
        assert plain not in back, f"{diacritic} must not offer {plain}"


def test_canonical_marks_the_lexicon_spelling(sp):
    """The caller must be able to tell which form the lexicon records."""
    by = {v["variant"]: v["canonical"] for v in sp.check("ain").variants}
    assert by.get("aiñ") is True, "aiñ is a lexicon entry and should be canonical"
    # The reverse is no longer offered at all — see
    # test_the_diacritic_spelling_is_offered_but_never_withdrawn — so there
    # is no variant left to carry a canonical flag.
    assert sp.check("aiñ").variants == []


def test_words_with_no_alternative_spelling_offer_nothing(sp):
    """
    A word the lexicon records one way only has no variant to offer.

    `shnong` was in this list until the gloss-recorded spellings were read:
    its own entry says `'nong` is "abbrev. of shnong", so the two are
    spellings of one word and the pair is now offered. So were `briew`
    (`'riew`) and `khlur` (`'lur`) — the abbreviated forms are real. The
    words below carry no such note and no diacritic form.
    """
    for w in ("bha", "dorbar", "mynsiem", "bynriew", "lyngdoh"):
        assert sp.check(w).variants == [], w


def test_gloss_recorded_spellings_are_offered(sp):
    """
    The printed dictionary records alternative spellings inside its
    definitions rather than as separate fields, and those ARE offered —
    but only the ones that are genuinely spellings.

    `spelling_links.json` was built from three gloss phrases at once:
    "same as", "also spelt as" and "abbrev. of". Only the first two name a
    variant spelling. A form the dictionary calls an abbreviation is a
    clipped word with its own entry, so offering it to someone who wrote
    the full form tells them their spelling has an alternative when it does
    not: `briew` was offered `'riew`, glossed "abbrev. of briew".
    """
    for word, expected in (("ade", "ide"), ("bdi", "bydi"),
                           ("'ba-hab", "'ba-pynthor")):
        assert expected in [v["variant"] for v in sp.check(word).variants], word

    # Neither an abbreviation nor an elided short form is a spelling.
    for word, short in (("briew", "'riew"), ("shnong", "'nong"),
                        ("slap", "'lap"), ("khyndew", "'dew"),
                        ("syiem", "s'iem"), ("kyieng", "k'ing"),
                        ("pyiar", "p'ïar")):
        offered = [v["variant"] for v in sp.check(word).variants]
        assert short not in offered, f"{word} must not offer {short}"

    # The expansion direction still helps: typing the short form offers the
    # full one. Only the abbreviating direction is suppressed.
    for short, full in (("s'iem", "syiem"), ("k'ing", "kyieng"),
                        ("p'ïar", "pyiar")):
        assert full in [v["variant"] for v in sp.check(short).variants], short


def test_n_tilde_words_are_accepted_without_the_diacritic(sp):
    """
    Before the fold map covered ñ, 'ain' was reported as MISSPELLED while
    'iathuh' was accepted — one of Khasi's two diacritics was tolerated and
    the other was not.
    """
    r = sp.check("ain")
    assert r.is_correct
    assert r.gate_name == "accepted"


def test_variant_source_is_recorded(sp):
    """Whole-word lexicon evidence is stronger than a rebuilt derived form."""
    by_source = {v.variant: v.source for v in sp.variants("iathuh")}
    assert by_source.get("ïathuh") == "lexicon"
    derived = {v.variant: v.source for v in sp.variants("jingiathuh")}
    assert derived.get("jingïathuh") == "root"


def test_canonical_spellings_are_listed_first(sp):
    """A recorded spelling should outrank a reconstructed or folded one.

    This used to probe `khwaiñ`, which worked because the lexicon recorded
    BOTH `khwain` and `khwaiñ`. The 2026-09-08 coda-tilde pass corrected
    the plain spellings away (a coda n takes the tilde), so `khwaiñ` now
    has only a folded partner and there is no recorded twin to be
    canonical. `iathuh`/`ïathuh` is the same relationship on the diaeresis
    side, where both spellings are still recorded, so it exercises the
    ranking this test is about.
    """
    vs = sp.variants("iathuh")
    assert vs and vs[0].canonical
    assert vs[0].variant == "ïathuh"


def test_diacritic_words_keep_a_real_frequency(sp):
    """
    The corpus has zero ï/ñ tokens, so diacritic words would otherwise be
    demoted to the unattested band. They inherit the count of their plain
    spelling instead.
    """
    nlp = sp.analyser.spell._speller.nlp_data
    assert nlp.get("ïathuh", 0) > 10, "diacritic word demoted to unattested"
    assert abs(nlp.get("ïathuh", 0) - nlp.get("iathuh", 0)) <= 2


# ----------------------------------------------------------------------
# Reciprocal prefix ia- -> ïa-, applied as a morphological rule
# ----------------------------------------------------------------------
#
# The lexicon spells this prefix both ways and cannot be used as the
# authority: 196 `ia-` entries against 148 `ïa-`, 68 `jingia-` against 24
# `jingïa-`, and only 26 words recorded both ways. So `ïabeit` is not
# listed and no lookup can propose it. morphology.prefixes.ia records the
# prefix as productive (664 entries, reciprocal_plural_subject) and
# morphology.affix_order gives the stacking order
# ["jing-", "nong-", "pyn-", "sngew-", "ia-", ROOT]; War (2001) writes it
# with the diaeresis.

@pytest.mark.parametrize("word,expected", [
    ("iabeit", "ïabeit"),          # ia    + base
    ("iakhublei", "ïakhublei"),
    ("iaai", "ïaai"),              # two-character base
    ("jingiathuh", "jingïathuh"),  # jing  + ia + base
    ("jingia-beit", "jingïa-beit"),
    ("pyniadei", "pynïadei"),      # pyn   + ia + base
    ("pynia-bit", "pynïa-bit"),
])
def test_reciprocal_prefix_alternatives(sp, word, expected):
    assert sp.is_correct(word), f"{word} should stay valid"
    offered = [v["variant"] for v in sp.check(word).variants]
    assert expected in offered, f"{word}: expected {expected}, got {offered}"


@pytest.mark.parametrize("word", ["iar", "jingbha", "bha", "shnong"])
def test_reciprocal_rule_does_not_overfire(sp, word):
    """Words that merely begin with those letters must not be rewritten."""
    from khasi_spell.variants import _reciprocal_forms
    assert _reciprocal_forms(word.lower(), sp.analyser.db) == []


def test_a_suffixed_form_is_not_offered_a_diacritic_twin(sp):
    """`phin` is `phi` + the future suffix `-n`, not a spelling of `phiñ`.

    Both are exact lexicon entries and both are pronouns, but they mean
    different things — "2sg/pl will" against "You... not" — so offering one
    for the other changes the grammar of the sentence rather than its
    spelling.

    The rule is SUFFIXES only. On a prefixed form the diacritic belongs to
    the prefix itself: `ia-` and `ïa-` are one morpheme, the lexicon now
    writes it `ïa-`, and `iathuh` -> `ïathuh` must keep working. A first
    version of this guard included prefixes and broke exactly that, which is
    what the two assertions below protect.
    """
    assert [v["variant"] for v in sp.check("phin").variants] == []
    assert [v["variant"] for v in sp.check("phiñ").variants] == []
    # Prefixed forms are untouched — the diacritic corrects the prefix.
    for plain, corrected in (("iathuh", "ïathuh"), ("iang", "ïang"),
                             ("jingiathuh", "jingïathuh")):
        assert corrected in [v["variant"] for v in sp.check(plain).variants], plain
    # And an ordinary doubled pair still upgrades.
    for plain, corrected in (("ain", "aiñ"), ("bein", "beiñ")):
        assert corrected in [v["variant"] for v in sp.check(plain).variants], plain


def test_reciprocal_base_must_not_be_sub_phonemic(sp):
    """
    db.is_known() accepts a bare digraph, so 'iang' would split as ia- +
    'ng' and be rewritten by rule. The phonology block marks 'ng' as
    meaningless standalone, and that is used to reject it.
    """
    from khasi_spell.variants import _reciprocal_forms, _invalid_standalone
    assert "ng" in _invalid_standalone()
    assert _reciprocal_forms("iang", sp.analyser.db) == []
    # It is still offered — but from the lexicon, which records ïang.
    assert "ïang" in [v["variant"] for v in sp.check("iang").variants]


def test_lexicon_is_internally_consistent_on_ia(sp):
    """
    After the 2026-08-26 correction the lexicon should spell this prefix one
    way. The rule's job moved from patching the data to handling user input.

    The standalone `ia` particle is included too, on the maintainer's
    instruction. Note that morphology.prefixes.ia states the opposite —
    "Bound prefix only; 'ia' as free morpheme object-marker is NOT this
    prefix" — so this deliberately overrides the database's annotation.
    """
    db = sp.analyser.db
    plain = [w for w in db._surface_index
             if w == "ia" or w.startswith("ia ") or
             (w.startswith("ia") and " " not in w)]
    assert plain == [], f"uncorrected ia entries remain: {plain[:10]}"
    assert len([w for w in db._surface_index
                if w.startswith("ïa") and " " not in w]) > 300


def test_no_duplicate_surfaces_after_the_correction(sp):
    """
    Correcting the spelling collapsed pairs recorded both ways. Two entries
    for one surface is not cosmetic — the analyser downgrades the verdict
    from 'valid' to 'derived' whenever a surface has more than one entry.
    """
    db = sp.analyser.db
    dupes = [w for w, e in db._surface_index.items() if len(e) > 1]
    assert dupes == [], f"{len(dupes)} duplicate surfaces: {dupes[:6]}"


def test_short_roots_are_not_analysed_as_prefix_plus_a_letter(sp):
    """
    `ïat` was recorded as [prefix 'ia', root 't']. The project had already
    ruled against that ("short-stem reroot: ia+t rejected") and re-rooted
    the plain twin, but the diacritic entry was missed; correcting the
    spelling merged the pair and the unrepaired record survived, which made
    the analyser call these words derived rather than valid.
    """
    for word in ("ïad", "ïai", "ïat", "ïaw"):
        result = sp.analyser.analyse(word)
        assert result.get("verdict") == "valid", f"{word} is {result.get('verdict')}"
        assert (result.get("phase2") or {}).get("root") == word


def test_legitimate_clitic_contractions_are_untouched(sp):
    """
    `um` is [root 'u', suffix '-m'] — the negation clitic (War pp.74-75),
    not a bogus short root. The repair must not have collapsed it.
    """
    for word in ("um", "un"):
        assert sp.is_correct(word)


def test_plain_spelling_still_resolves_to_the_corrected_root(sp):
    """Typing the old spelling must still work and reach the same root."""
    for plain in ("iathuh", "iabeit", "iap", "iaid", "jingiathuh"):
        assert sp.is_correct(plain), f"{plain} rejected after the correction"
        offered = [v["variant"] for v in sp.check(plain).variants]
        assert any("ï" in v for v in offered), f"{plain} offers no corrected form"

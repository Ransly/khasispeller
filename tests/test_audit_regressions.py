"""
Regression tests for the 2026-09-30 audit.

Each test pins one defect that was confirmed by running the checker and then
fixed. None of them was caught by the suite at the time, which is why each
one is here. Words come from the frozen benchmarks, the README's own
examples, or are synthetic; no lexicon glosses are quoted.
"""
import json
import unicodedata

import pytest

from khasi_spell import KhasiSpeller


@pytest.fixture(scope="module")
def sp():
    return KhasiSpeller(eager=True)


# ----------------------------------------------------------------------
# Text that is correct must come back unchanged
# ----------------------------------------------------------------------

@pytest.mark.parametrize("word", ["sngewrém", "úd", "wád"])
def test_accented_lexicon_words_are_not_split(sp, word):
    """The tokeniser listed only ï and ñ as letters, so `sngewrém` became
    `sngewr` + `m` and both pieces were "corrected"."""
    assert sp.check(word).is_correct
    text = f"ka {word} ka bha"
    r = sp.check_text(text)
    assert r.corrections == [], [(c.original, c.suggestion) for c in r.corrections]
    assert r.corrected == text


def test_typographic_apostrophe_is_an_apostrophe(sp):
    """`nga’m` typed on a phone got an `m` -> `ma` correction."""
    for word in ("nga’m", "ngaʼm"):
        assert sp.check(word).is_correct, word
        text = f"u {word} u bha"
        r = sp.check_text(text)
        assert r.corrections == [], word
        assert r.corrected == text          # the writer's own quote survives


def test_scan_debris_gets_one_edit_not_two(sp):
    """The facade tokenised without the engine's noise joiners and corrected
    a fragment inside a word the engine had already flagged."""
    text = "ka i^p-shnong ka"
    r = sp.check_text(text, context=False)
    spans = sorted((c.start, c.end) for c in r.corrections)
    assert all(a[1] <= b[0] for a, b in zip(spans, spans[1:])), spans
    assert "i^pa-shnong" not in r.corrected


def test_sentence_initial_capital_survives_correction(sp):
    assert sp.correct_text("Shnng ka bha") == "Shnong ka bha"
    assert sp.correct("Shnng") == "Shnong"


def test_join_keeps_the_capital(sp):
    r = sp.check_text("Jing ialehkai ka bha")
    assert r.corrections and r.corrections[0].suggestion == "Jingïalehkai"


# ----------------------------------------------------------------------
# Misspellings must not be hidden
# ----------------------------------------------------------------------

@pytest.mark.parametrize("text", [
    "Ka shnong ka bha\nShnng ka bha",       # start of a line
    "Kane ka: Shnng ka bha",                # after a colon
    "- Shnng ka bha",                        # a list item
])
def test_line_initial_capital_is_still_checked(sp, text):
    r = sp.check_text(text)
    assert any(c.original == "Shnng" for c in r.corrections), r.skipped


def test_a_lower_case_name_token_is_checked(sp):
    """The gazetteer matched case-insensitively, so a lower-case typo that
    happened to spell a name was withheld as a "name"."""
    from khasi_spell import foreign
    assert "sokda" in foreign.gazetteer()
    r = sp.check_text("ka sokda ka bha")
    assert any(c.original == "sokda" for c in r.corrections)
    assert not r.skipped
    # Written as a name, mid-sentence, it is still withheld.
    r2 = sp.check_text("u la leit sha Sokda")
    assert any(s["word"] == "Sokda" for s in r2.skipped)


# ----------------------------------------------------------------------
# One decision for every entry point
# ----------------------------------------------------------------------

def test_single_word_and_text_agree_on_a_misspelled_part(sp):
    """`check()` accepted `man-miay` while `check_text()` flagged it."""
    r = sp.check("man-miay")
    assert r.is_correct is False
    assert r.suggestions == ["man-miat"]
    t = sp.check_text("u man-miay u")
    assert [(c.original, c.suggestion) for c in t.corrections] == [("man-miay", "man-miat")]
    assert sp.correct("man-miay") == "man-miat"


def test_recorded_compounds_are_still_accepted(sp):
    """A compound the lexicon records is a word whatever its parts look
    like — `jrain-jrain` is an entry although `jrain` is not — and one whose
    parts are all words passes the part check."""
    for word in ("jrain-jrain", "man-bha"):
        assert sp.check(word).is_correct, word


def test_analyse_reports_the_checkers_decision(sp):
    """analyse() now returns the checker's decision as `accepted`, and says
    in `verdict_desc` when its morphological reading differs from it.

    `man-miay` was the example until 2026-10-01, when a final y became a
    phonological error (no Khasi word ends in y): both readings now reject
    it, so `man-mial` carries the disagreement instead."""
    for word in ("man-mial", "man-miay", "jylla", "shnong", "nongihkai",
                 "nongsngew", "jingkynshaitum"):
        a = sp.analyser.analyse(word)
        assert a["accepted"] == sp.check(word).is_correct, word
        if not a["accepted"] and a["verdict"] in ("valid", "derived"):
            assert "not accepted by the spell checker" in a["verdict_desc"], word
    assert "miat" in sp.analyser.analyse("man-mial")["verdict_desc"]
    assert "do not end in y" in sp.analyser.analyse("man-miay")["verdict_desc"]


def test_analyse_describes_a_corpus_word_as_one(sp):
    a = sp.analyser.analyse("jylla")
    assert a["verdict"] == "valid"
    assert "corpus" in a["verdict_desc"].lower()
    assert "particle" not in a["verdict_desc"].lower()


def test_engine_entry_points_normalise_unicode(sp):
    """/v1/spellcheck and /analyse skipped NFC, so a decomposed `ï` split the
    word and produced a bogus correction."""
    nfd = unicodedata.normalize("NFD", "u jingïathuh u bha")
    out = sp.analyser.check_sentence(nfd)
    assert out["corrections"] == []
    assert out["input"] == unicodedata.normalize("NFC", nfd)
    assert sp.analyser.spell.suggest(unicodedata.normalize("NFD", "jingïathuh"))["is_known"]
    assert sp.analyser.analyse(unicodedata.normalize("NFD", "jingïathuh"))["verdict"] in ("valid", "derived")


def test_joins_are_not_reported_as_accepted(sp):
    r = sp.check_text("ka jing ialehkai ka bha")
    c = r.corrections[0]
    assert c.method == "bound_prefix_spaced" and c.gate == 3


def test_is_known_is_the_same_decision_as_suggest(sp):
    chk = sp.analyser.spell
    for word in ("shnong", "shnng", "man-miay", "jylla", "ng", "cow"):
        assert chk.is_known(word) == chk.suggest(word)["is_known"], word


# ----------------------------------------------------------------------
# Alternative spellings are the same word
# ----------------------------------------------------------------------

@pytest.mark.parametrize("word", ["um", "ew", "oh", "eng"])
def test_short_forms_are_not_linked_to_different_words(sp, word):
    """`um` 'he not' was offered `ïum` 'a little water'; `ew` 'Ah!' was
    offered `ïew` 'market'. Different words, not two spellings."""
    assert ("ï" + word) not in [v["variant"] for v in sp.check(word).variants]


def test_the_supported_short_form_remains(sp):
    assert "ïing" in [v["variant"] for v in sp.check("ing").variants]


# ----------------------------------------------------------------------
# Distance and explanation
# ----------------------------------------------------------------------

@pytest.mark.parametrize("a,b,cost", [
    ("dien", "din", 0.4),        # ie / i
    ("dien", "den", 0.4),        # ie / e
    ("diang", "dieng", 0.6),     # ia / ie
    ("dienk", "dieng", 0.5),     # nk / ng
    ("dign", "ding", 0.5),       # gn / ng
])
def test_multi_phoneme_confusions_apply_when_switched_on(monkeypatch, a, b, cost):
    """16 of the 44 confusion entries could never fire: `_to_phonemes`
    never produces `ie` or `nk` as one unit. They now work — as an opt-in
    table, because switching them on measured one point worse at top-1."""
    from khasi_engine import spell_checker as S
    monkeypatch.setattr(S, "_MULTI_UNIT_CONFUSIONS", True)
    monkeypatch.setitem(S._MULTI_STATE, "digraphs", None)
    try:
        assert abs(S._levenshtein(a, b) - cost) < 1e-9
    finally:
        S._MULTI_STATE["digraphs"] = None


def test_the_live_table_holds_only_single_phoneme_pairs():
    """Every default entry must be one a substitution can actually see."""
    from khasi_engine import spell_checker as S
    for a, b in S._CONFUSION_COST:
        assert len(S._to_phonemes(a)) == 1 and len(S._to_phonemes(b)) == 1, (a, b)
    assert not S._MULTI_UNIT_CONFUSIONS
    assert S._levenshtein("dien", "din") == 1.0


@pytest.mark.parametrize("a,b", [("dand", "dang"), ("dien", "din"),
                                 ("lyngdo", "lyngdoh"), ("nongihkai", "nonghikai")])
def test_explanation_adds_up_to_the_ranking_distance(a, b):
    from khasi_engine.spell_checker import _levenshtein, explain_edit
    ops = explain_edit(a, b)
    assert abs(sum(o["cost"] for o in ops) - _levenshtein(a, b)) < 0.011, ops


def test_digraphs_follow_the_data():
    from khasi_engine import phonology
    from khasi_engine.spell_checker import _to_phonemes
    assert _to_phonemes("jhieh")[0] == "jh" or "jh" not in phonology.DIGRAPHS_AS_SINGLE
    assert "dz" not in _to_phonemes("dzong")


# ----------------------------------------------------------------------
# Language model and real-word detection
# ----------------------------------------------------------------------

def test_one_left_word_is_kept_as_context():
    from khasi_spell.ngram import pad_context
    assert pad_context(["ka"], []) == (("<s>", "ka"), ("</s>", "</s>"))
    assert pad_context(["u", "ka"], ["bha"]) == (("u", "ka"), ("bha", "</s>"))


def test_slot_score_uses_the_single_left_word(sp):
    lm = sp._lm()
    if lm is None:
        pytest.skip("no n-gram model")
    assert lm.slot_score("sorkar", ["ka"], []) == lm.slot_score("sorkar", ["<s>", "ka"], [])
    assert lm.slot_score("sorkar", ["ka"], []) != lm.slot_score("sorkar", [], [])


def test_real_word_margin_applies_to_one_call_only(sp):
    if sp._lm() is None:
        pytest.skip("no n-gram model")
    text = "Ka sorkar ki la pynkiew ia ka tulop jong ki MLA"
    sp.check_realword(text, min_margin=0.5)
    assert sp._realword.min_margin == 8.0
    loose = sp.check_realword(text, min_margin=0.5)
    strict = sp.check_realword(text)
    assert len(loose) >= len(strict)


def test_the_commonest_word_can_be_flagged(sp):
    """A frequency rule hidden in freq_ratio=1.0 required every alternative
    to be at least as common as the typed word, so `ka` — the commonest word
    in the corpus — could never be flagged."""
    if sp._lm() is None:
        pytest.skip("no n-gram model")
    text = ("Hap ban pynkynmaw ia kine ka rangbah, ba kumba 38 snem ka sorkar "
            "Congress ka la pyniaid ia ka jylla Meghalaya.")
    flags = sp.check_realword(text)
    at = text.index("ka rangbah")
    assert any(f.start == at and f.suggestion == "ki" for f in flags), flags


def test_one_language_model_per_speller(sp):
    if sp._lm() is None:
        pytest.skip("no n-gram model")
    sp.check_text("ka shnng ka bha")
    sp.check_realword("ka sorkar ka la wan")
    assert sp._realword._lm is sp._ngram_lm


def test_a_failing_model_does_not_fail_the_check(sp):
    class Broken:
        def slot_score(self, *a, **k):
            raise RuntimeError("database went away")
    saved = sp._ngram_lm
    try:
        sp._ngram_lm = Broken()
        r = sp.check_text("ka shnng ka bha")
        assert r.corrections and r.corrections[0].suggestion == "shnong"
    finally:
        sp._ngram_lm = saved


# ----------------------------------------------------------------------
# Spellers do not leak into each other
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def second(sp):
    """Built AFTER `sp`, in the same process, sharing its lexicon object."""
    return KhasiSpeller(eager=True, use_corpus_freq=False, use_corpus_pool=False)


def test_a_second_speller_leaves_the_first_alone(sp, second):
    nlp = sp.analyser.spell._speller.nlp_data
    probe = ["abi", "abra"]
    before = {w: nlp.get(w) for w in probe}
    KhasiSpeller(eager=True)                     # a third, default speller
    assert {w: nlp.get(w) for w in probe} == before


def test_corpus_frequency_switch_is_per_speller(sp, second):
    from khasi_engine.spell_checker import KhasiSpellChecker
    off = second.analyser.spell
    assert off._ranking_freq is None and not off._corpus_freq_applied
    # pseudo-frequencies, not the log-scaled corpus band
    assert max(off._speller.nlp_data.get(w, 0) for w in ("abi", "abra")) >= 10
    assert sp.analyser.spell._corpus_freq_applied
    assert KhasiSpellChecker._FREQ_HIGH_BAR == 10


def test_corpus_pool_switch_is_per_speller(sp, second):
    """Was pinned WEAK: the pool lived on the shared lexicon, so a later
    `use_corpus_pool=False` speller still offered pool words."""
    assert "pyntreikam" in sp.suggest("pyntreika", n=5)
    assert "pyntreikam" not in second.suggest("pyntreika", n=5)
    assert not second.analyser.spell.is_corpus_form("pyntreikam")


def test_add_word_accepts_names_and_is_per_speller(sp, second):
    """`add_word("Meghalaya")` left the word rejected: the phonotactic gate
    (a bare g) ran first. And a taught word leaked into every speller."""
    assert not second.check("Meghalaya").is_correct
    second.add_word("Meghalaya")
    assert second.check("Meghalaya").is_correct
    assert "meghalaya" in second.suggest("meghalya", n=5)
    assert not sp.check("Meghalaya").is_correct


# ----------------------------------------------------------------------
# The lexicon object
# ----------------------------------------------------------------------

def test_frequency_table_is_read_only(sp):
    table = sp.analyser.db.to_freq_dict()
    with pytest.raises(TypeError):
        table["shnong"] = 1


def test_next_entry_id_follows_the_lexicon_scheme(sp):
    db = sp.analyser.db
    eid = db.next_entry_id()
    assert eid.startswith("kh_DB_") and eid not in db._id_index


def test_stats_count_lexemes(sp):
    st = sp.analyser.db.stats()
    assert st["lexeme_count"] > 0
    assert st["loan_words"] == st["shim_kylliang_forms"]


# ----------------------------------------------------------------------
# HTTP service and CLI
# ----------------------------------------------------------------------

@pytest.fixture(scope="module")
def client():
    from fastapi.testclient import TestClient
    from khasi_spell import api
    with TestClient(api.app) as c:
        yield c


def test_word_route_keeps_parallel_lists(client):
    d = client.post("/word", json={"word": "lyngdo", "n": 3}).json()
    assert len(d["suggestions"]) == 3
    assert len(d["distances"]) == len(d["glosses"]) == 3


def test_realword_route_returns_the_normalised_text(client):
    text = unicodedata.normalize("NFD", "u jingïathuh u bha")
    d = client.post("/realword", json={"text": text}).json()
    assert d["text"] == unicodedata.normalize("NFC", text)


def test_realword_margin_does_not_leak_between_requests(client):
    client.post("/realword", json={"text": "ka sorkar ki la wan", "min_margin": 0.5})
    from khasi_spell import api
    rw = api._speller._realword
    assert rw is None or rw.min_margin == 8.0


def test_service_index_lists_analyse(client):
    assert any("/analyse" in k for k in client.get("/api").json()["endpoints"])


def test_health_reports_resources(client):
    res = client.get("/health").json()["resources"]
    assert res["gazetteer"] and "data_dir" in res


def test_cli_json_is_valid_and_includes_real_word_results(capsys):
    """Two bugs: engine progress lines went to stdout ahead of the JSON, so
    `--json` output did not parse; and `--realword` was dropped under it."""
    from khasi_spell import cli
    cli.main(["--json", "check", "Ka sorkar ki la pynkiew ia ka tulop", "--realword"])
    out = json.loads(capsys.readouterr().out)
    assert "realword" in out or "realword_error" in out


def test_cli_fix_counts_only_applied_corrections(tmp_path, capsys):
    from khasi_spell import cli
    src = tmp_path / "t.txt"
    # `qwertyuiop` breaks the phonotactics and has no candidate at all.
    src.write_text("u qwertyuiop u dand\n", encoding="utf-8")
    cli.main(["file", str(src), "--fix"])
    out = capsys.readouterr().out
    assert "1 correction(s) applied" in out and "1 flagged word(s) left unchanged" in out


@pytest.mark.parametrize("text,word", [
    ("u ïaid sha ka ïing", "ïaid"),
    ("u ïaid sha ka ïing", "ïing"),
    ("u la ïaid sha ka ïew", "ïew"),
    ("U nga'm u la wan", "nga'm"),
])
def test_word_choice_never_asks_to_drop_a_diacritic(sp, text, word):
    """The corpus holds no ï, ñ or apostrophe, so scored in the lexicon's
    spelling every diacritic word floored against its plain twin: `ïaid` was
    flagged with `iaid` at margin 17.8, `ïing` with the illegal `iing`."""
    if sp._lm() is None:
        pytest.skip("no n-gram model")
    flags = sp.check_realword(text, min_margin=3.0)
    assert not [f for f in flags if f.word == word], [(f.word, f.suggestion) for f in flags]


def test_corpus_form_is_how_the_corpus_writes_it():
    from khasi_engine.tokens import corpus_form
    assert corpus_form("ïaid") == "iaid"
    assert corpus_form("nga’m") == "ngam"
    assert corpus_form("jaiñsem") == "jainsem"


# ----------------------------------------------------------------------
# Maintainer rulings, 2026-10-01: how a Khasi word may end
# ----------------------------------------------------------------------

@pytest.mark.parametrize("word", ["party", "history", "bodily", "jaly", "ly"])
def test_no_word_ending_in_y_is_accepted(sp, word):
    # party and history were accepted on corpus frequency 2 + phonotactics 1;
    # bodily (English in the 1906 dictionary) and jaly (the first half of
    # jaly-eit) were accepted because the lexicon index holds them.
    assert not sp.is_correct(word)


@pytest.mark.parametrize("word", ["bak-ly-bak", "ly-ngang", "siej", "biej", "soh", "ksew"])
def test_words_that_may_end_so_are_accepted(sp, word):
    assert sp.is_correct(word)


def test_no_suggestion_ends_in_y(sp):
    for typo in ("parti", "bodili", "jalyi", "histry"):
        assert not [s for s in sp.suggest(typo, n=10) if s.endswith("y")], typo


@pytest.mark.parametrize("damaged,intended",
                         [("balangj", "balang"), ("dohtdongj", "dohtdong")])
def test_withdrawn_scan_damage_is_corrected(sp, damaged, intended):
    # The dictionary's comma read as j; withdrawn 2026-10-01.
    assert not sp.is_correct(damaged)
    assert sp.suggest(damaged)[0] == intended


# ----------------------------------------------------------------------
# Maintainer ruling, 2026-10-01: English words are rejected
# ----------------------------------------------------------------------

@pytest.mark.parametrize("word", ["hospital", "the", "state", "member", "kilo",
                                  "within", "iron", "party"])
def test_english_words_are_rejected_without_a_suggestion(sp, word):
    # hospital, the, state were accepted on corpus frequency; within and iron
    # on a spurious Khasi parse. No Khasi string is a correction of them.
    r = sp.check(word)
    assert not r.is_correct
    assert r.method == "english_word"
    assert r.suggestions == []


@pytest.mark.parametrize("word", ["longing", "shnong", "pyntreikam", "jylla",
                                  "nongthohkhubor", "ka", "bad"])
def test_khasi_words_survive_the_english_list(sp, word):
    # longing is the Khasi "household" and is held back from the list.
    assert sp.is_correct(word)


def test_english_words_are_flagged_in_text_and_left_alone(sp):
    text = "Ka hospital jong ka state ka la plie."
    flagged = [(c.original, c.suggestion, c.method)
               for c in sp.check_text(text, variants=False, context=False).corrections]
    assert flagged == [("hospital", None, "english_word"), ("state", None, "english_word")]
    # Auto-correction must not turn them into Khasi words (`state` -> `star`).
    assert sp.correct_text(text) == text


def test_english_words_are_never_offered(sp):
    from khasi_engine.spell_checker import _english_words
    english = _english_words()
    assert len(english) > 1000
    for typo in ("hospitel", "stat", "membr", "thi", "kilu"):
        assert not set(sp.suggest(typo, n=10)) & english, typo


def test_english_words_stay_out_of_the_corpus_pool(sp):
    from khasi_engine.spell_checker import _english_words
    assert not sp.analyser.spell._corpus_accept & _english_words()


# ----------------------------------------------------------------------
# Word Details: every word shows its own analysis (2026-10-02)
# ----------------------------------------------------------------------

def test_katkum_shows_its_own_analysis(client):
    """The panel for `katkum` showed kat·ba, CVC.CV and root `katba`: at
    import the entry inherited the analysis of the headword above it."""
    d = client.post("/analyse", json={"word": "katkum"}).json()
    assert d["morphology"]["root"] == "katkum"
    assert d["phonology"]["syllables"] == ["kat", "kum"]
    assert d["phonology"]["pattern"] == "CVC.CVC"


def _single_word_records(sp):
    records = (getattr(sp.analyser.db, "_enriched_by_id", None) or {}).values()
    out = [(((r.get("form") or {}).get("surface") or "").strip(), r) for r in records]
    out = [(w, r) for w, r in out if w and " " not in w and "," not in w]
    assert len(out) > 10_000, "the lexicon's records were not loaded"
    return out


def test_stored_syllables_spell_their_own_word(sp):
    """529 single-word entries held another word's syllables (`ben` had
    `bel`'s). Repaired by scripts/fix_inherited_analyses.py."""
    from khasi_spell.api import _syl_letters
    wrong = []
    for w, r in _single_word_records(sp):
        syl = ((r.get("phonology") or {}).get("derived") or {}).get("syllables")
        if syl and "".join(_syl_letters(s) for s in syl) != _syl_letters(w):
            wrong.append((w, syl))
    assert not wrong, wrong[:10]


def test_a_bare_root_is_its_own_root(sp):
    """An entry ruled a bare root named another word as its root: `katkum`
    -> katba, `braw` -> brap. A missing mark (`ïeng` -> ieng) is not that."""
    from khasi_spell.api import _SYL_MARKS, _syl_letters
    def same(a, b):
        return _syl_letters(a).translate(_SYL_MARKS) == _syl_letters(b).translate(_SYL_MARKS)
    wrong = [(w, (r.get("lemma") or {}).get("root"))
             for w, r in _single_word_records(sp)
             if (r.get("morphology") or {}).get("type") == "root"
             and (r.get("lemma") or {}).get("root")
             and not same(r["lemma"]["root"], w)]
    assert not wrong, wrong[:10]


@pytest.mark.parametrize("word, root, marked, form", [
    ("bynriew", "briew", "b‹yn›riew", "-yn-"),
    ("shnong", "shong", "sh‹n›ong", "-n-"),
])
def test_an_infix_is_shown_inside_its_root(client, word, root, marked, form):
    """The panel listed `yn` above `briew` the way it lists a prefix above its
    root, which reads as yn + briew. An infix row now carries its own form
    and what it makes, and the derivation line marks where it goes."""
    m = client.post("/analyse", json={"word": word}).json()["morphology"]
    assert m["root"] == root
    assert m["chain"] == [root, marked]
    (row,) = [a for a in m["affixes"] if a["role"] == "infix"]
    assert row["form"] == form and row["label"].startswith("infix, makes")


@pytest.mark.parametrize("word, surface, lexical", [
    ("jingstad", ["jing", "stad"], ["jing-", "stad"]),                # prefix + root
    ("jingïalehkai", ["jing", "ïa", "lehkai"], ["jing-", "ïa-", "lehkai"]),
    ("pyllait", ["pyl", "lait"], ["pyn-", "lait"]),                   # n -> l before l
    ("bynriew", ["briew", "yn"], ["briew", "-yn-"]),                  # infix
    ("katkum", ["katkum"], ["katkum"]),                               # bare root
])
def test_morphology_maps_surface_to_lexical(client, word, surface, lexical):
    """Word Details shows the word as written above the pieces it is built
    from. A piece that changed is marked, the root carries its own class, and
    the pieces always spell the written word."""
    fm = client.post("/analyse", json={"word": word}).json()["morphology"]["form_map"]
    assert [s["surface"] for s in fm["segments"]] == surface
    assert [s["lexical"] for s in fm["segments"]] == lexical
    if word == "pyllait":
        assert fm["segments"][0]["changed"] and "assimilates" in fm["rule"]
        assert fm["generation"] == ["pyn- + lait", "pyllait"]
    if word == "bynriew":
        assert fm["marked"] == "b‹yn›riew"
    if word == "jingïalehkai":
        assert fm["generation"] == ["lehkai", "ïalehkai", "jingïalehkai"]
    else:
        assert all(not s["changed"] for s in fm["segments"][1:])


def test_the_root_carries_its_own_word_class(client):
    """`stad` was labelled a noun because `jingstad` is one."""
    fm = client.post("/analyse", json={"word": "pynlong"}).json()["morphology"]["form_map"]
    root = fm["segments"][-1]
    assert root["kind"] == "root" and root["lexical"] == "long" and root["pos"] == "verb"


# ----------------------------------------------------------------------
# Phonology for a word the lexicon cannot syllabify (2026-10-04)
# ----------------------------------------------------------------------

def test_a_word_the_lexicon_cannot_syllabify_gets_syllables_by_rule(client):
    """`ïatreilang` showed "well formed" and a letter count: it has no entry,
    and neither has its root treilang (trei + lang), so nothing could be
    assembled. The rules fill the gap and the panel says where they came from."""
    p = client.post("/analyse", json={"word": "ïatreilang"}).json()["phonology"]
    assert p["syllables"] == ["ïa", "trei", "lang"]
    assert p["pattern"] == "VV.CCVV.CVC"
    assert p["syllables_by_rule"] is True
    lexicon = client.post("/analyse", json={"word": "bynriew"}).json()["phonology"]
    assert lexicon["syllables"] == ["byn", "riew"] and "syllables_by_rule" not in lexicon


def test_the_rules_agree_with_the_lexicon(sp):
    """khasi_spell/syllables.py quotes 98.6%: each single word the lexicon
    syllabifies, parsed as if unknown, gets the lexicon's own syllables."""
    import re
    from khasi_engine import morphology
    from khasi_engine import tokens as T
    from khasi_spell.api import _own_syllables
    from khasi_spell.syllables import syllabify
    db = sp.analyser.db
    total = same = 0
    for w, r in _single_word_records(sp):
        w = T.nfc(w).lower()
        gold = _own_syllables(w, ((r.get("phonology") or {}).get("derived") or {}).get("syllables"))
        if not re.fullmatch(r"[a-zïñ]+", w) or not gold or "".join(gold) != w:
            continue
        layers = morphology.parse(w, lambda x: [] if x == w else db.lookup(x)).get("layers") or {}
        got = syllabify(w, [layers[k] for k in ("prefix", "prefix2", "prefix3") if layers.get(k)],
                        layers.get("suffix") or "")
        total += 1
        same += got == gold
    assert total > 7000
    assert same / total >= 0.98, f"{same}/{total}"

"""
FastText semantic re-ranking — local backend.

The model ships in models/fasttext/. Tests that need it are skipped when
gensim is missing or the model file is absent, so the suite still passes
on a checkout without the 786 MB download.

Note that the re-ranker is OFF by default. It measured worse than
rule-only ranking (top-1 53.2% -> 49.4%), so these tests assert that it
is wired correctly, not that it improves results.
"""

import pytest

from khasi_engine import w2v_client
from khasi_spell import KhasiSpeller

gensim = pytest.importorskip("gensim", reason="gensim not installed")

pytestmark = pytest.mark.skipif(
    w2v_client.local_model_path() is None,
    reason="no local FastText model in models/fasttext/",
)


@pytest.fixture(scope="module")
def model():
    return w2v_client.load_local()


# ----------------------------------------------------------------------
# Backend resolution — must not load anything
# ----------------------------------------------------------------------

def test_bundled_model_is_discovered():
    assert w2v_client.local_model_path() is not None
    assert w2v_client.is_configured()


def test_local_backend_preferred_over_remote(monkeypatch):
    """A bundled model wins over a configured Space unless told otherwise."""
    monkeypatch.setenv("KHASI_W2V_URL", "https://example.invalid")
    assert w2v_client.active_backend() == "local"


def test_backend_can_be_forced_remote(monkeypatch):
    monkeypatch.setenv("KHASI_W2V_BACKEND", "remote")
    monkeypatch.setenv("KHASI_W2V_URL", "https://example.invalid")
    assert w2v_client.active_backend() == "remote"


def test_unconfigured_backend_reports_none(monkeypatch, tmp_path):
    monkeypatch.setenv("KHASI_W2V_BACKEND", "remote")
    monkeypatch.delenv("KHASI_W2V_URL", raising=False)
    assert w2v_client.active_backend() is None
    assert not w2v_client.is_configured()


def test_missing_model_override_falls_back(monkeypatch, tmp_path):
    monkeypatch.setenv("KHASI_W2V_MODEL", str(tmp_path / "nope.model"))
    assert w2v_client.local_model_path() is None


# ----------------------------------------------------------------------
# Model properties — documents what was actually shipped
# ----------------------------------------------------------------------

def test_model_shape(model):
    assert model.vector_size == 100
    assert len(model.wv) > 20000
    assert model.sg == 0, "expected CBOW"


def test_info_reports_loaded_model(model):
    info = w2v_client.info()
    assert info["backend"] == "local"
    assert info["loaded"] is True
    assert info["dim"] == 100
    assert info["algorithm"] == "cbow"


# ----------------------------------------------------------------------
# Scoring — the contract spell_checker depends on
# ----------------------------------------------------------------------

def test_score_returns_one_entry_per_candidate(model):
    candidates = ["pylleng", "pleng", "pynleng"]
    scores = w2v_client.score("pyleng", candidates)
    assert set(scores) == set(candidates)
    assert all(isinstance(v, float) for v in scores.values())


def test_score_handles_unseen_words(model):
    """
    Subword composition means FastText scores words it never saw in
    training — the property that makes it plausible for a morphologically
    productive language.
    """
    scores = w2v_client.score("zzyrkhang", ["jingbha", "bha"])
    assert len(scores) == 2


def test_similar_filters_noise(model):
    hits = w2v_client.similar("jingbha", topn=5)
    assert hits
    assert all(w2v_client.is_clean(w) for w, _ in hits)
    assert all(-1.0 <= s <= 1.0 for _, s in hits)


@pytest.mark.parametrize("tok,ok", [
    ("jingbha", True), ("ïing", True), ("shnong", True),
    ("abc123", False), ("with.dot", False), ("a", False), ("", False),
    # QUIRK, inherited verbatim from the Hugging Face Space: the regex
    # ^[a-zïñ'\-]+$ permits an apostrophe, but the punctuation blacklist
    # rejects it first, so Khasi elision forms never survive the filter.
    # Arguably wrong for Khasi, where the apostrophe marks elision
    # ('tikmie < kti + kmie). Kept as-is because is_clean() is used only
    # by the similar() diagnostic, never by score(), so it cannot affect
    # spelling results — and because diverging would put local and remote
    # backends out of step.
    ("u'm", False),
])
def test_is_clean(tok, ok):
    assert w2v_client.is_clean(tok) is ok


# ----------------------------------------------------------------------
# Wiring into the speller
# ----------------------------------------------------------------------

def test_embeddings_off_by_default():
    sp = KhasiSpeller()
    assert sp.embeddings["enabled"] is False


def test_embeddings_can_be_enabled():
    sp = KhasiSpeller(use_embeddings=True, eager=True)
    assert sp.embeddings["enabled"] is True
    assert sp.embeddings["backend"] == "local"


def test_suggestions_still_produced_with_reranking():
    """Re-ranking reorders; it must never empty the candidate list."""
    sp = KhasiSpeller(use_embeddings=True, eager=True)
    suggestions = sp.suggest("lyngdo")
    assert suggestions
    assert "lyngdoh" in suggestions


# ----------------------------------------------------------------------
# Real-word detection — prototype, deliberately not wired into check_text
# ----------------------------------------------------------------------

def test_realword_detector_is_quiet_on_clean_text():
    """
    At the shipped thresholds the detector must stay silent on ordinary
    correct Khasi. It measured 0.7% false positives per sentence; anything
    that raises that makes the feature actively harmful.
    """
    from khasi_spell import KhasiSpeller
    from khasi_spell.realword import RealWordDetector

    sp = KhasiSpeller(eager=True)
    det = RealWordDetector(sp)
    clean = "Ka sorkar ka la pynkiew ia ka tulop jong ki MLA barim ha ka jylla"
    assert det.check(clean) == []


def test_confusion_set_finds_real_neighbours():
    """Edit-then-lookup must return lexicon words, not generated strings."""
    from khasi_spell import KhasiSpeller
    from khasi_spell.realword import RealWordDetector

    sp = KhasiSpeller(eager=True)
    det = RealWordDetector(sp)
    det._ensure()
    alts = det.confusion_set("briew")
    assert alts, "no neighbours found for a common word"
    db = sp.analyser.db
    assert all(db.is_known(a) for a in alts)
    assert "briew" not in alts


# ----------------------------------------------------------------------
# N-gram language model and real-word detection
# ----------------------------------------------------------------------

def _lm_available():
    from khasi_spell.ngram import NgramLM
    return NgramLM()._path.is_file()


ngram_required = pytest.mark.skipif(
    not _lm_available(), reason="data/ngrams.json.gz not built"
)


@ngram_required
def test_lm_loads_and_reports_shape():
    from khasi_spell.ngram import NgramLM
    lm = NgramLM().load()
    info = lm.info()
    assert info["tokens"] > 1_000_000
    assert info["bigrams"] > 10_000
    assert info["trigrams"] > 10_000


@ngram_required
@pytest.mark.parametrize("left,gold,wrong,right", [
    (["ka", "sorkar"], "ka", "ki", ["la", "pynkiew"]),
    (["u"], "briew", "riew", ["u", "la"]),
])
def test_lm_prefers_the_right_word_in_context(left, gold, wrong, right):
    """The whole premise: the LM must score the correct word higher."""
    from khasi_spell.ngram import NgramLM
    lm = NgramLM().load()
    assert lm.slot_score(gold, left, right) > lm.slot_score(wrong, left, right)


@ngram_required
def test_realword_quiet_on_clean_corpus_text():
    """Zero false positives is the shipped operating point. Guard it."""
    from khasi_spell import KhasiSpeller
    sp = KhasiSpeller(eager=True)
    clean = "Ka sorkar ka la pynkiew ia ka tulop jong ki MLA barim ha ka jylla"
    assert sp.check_realword(clean) == []


@ngram_required
def test_realword_catches_a_clitic_swap():
    """
    ka/ki confusion — the commonest real-word error in Khasi.

    The sentence contains TWO instances of 'ki': one wrong (after 'sorkar',
    where the corpus overwhelmingly has 'ka') and one correct ('jong ki
    MLA'). Catching the first while leaving the second alone is the whole
    job — a detector that flags both would be worthless.
    """
    from khasi_spell import KhasiSpeller
    sp = KhasiSpeller(eager=True)
    text = "Ka sorkar ki la pynkiew ia ka tulop jong ki MLA barim ha ka jylla"
    flags = sp.check_realword(text, min_margin=5.0)

    wrong = [f for f in flags if f.word.lower() == "ki" and f.suggestion == "ka"]
    assert wrong, f"clitic swap missed: {[(f.word, f.suggestion) for f in flags]}"
    assert wrong[0].start == text.index("ki"), "flagged the wrong occurrence"
    assert len(wrong) == 1, "the legitimate 'ki' in 'jong ki MLA' was also flagged"


@ngram_required
def test_realword_catches_a_word_swap():
    from khasi_spell import KhasiSpeller
    sp = KhasiSpeller(eager=True)
    flags = sp.check_realword("U riew u la leit sha ka iing jong u", min_margin=6.0)
    assert any(f.suggestion == "briew" for f in flags)

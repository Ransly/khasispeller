"""
PostgreSQL-backed n-gram model.

The file model expands 838,445 counts into Python dicts — 5.1 MB of gzip
becoming 181 MB resident. `PgNgramLM` leaves the counts in the database and
fetches only what a slot needs, batched into one query per slot.

That is only worth having if it is *exactly* equivalent. These tests assert
the PG model returns the same scores as the file model rather than assuming
the arithmetic was copied correctly, and they skip cleanly when no database
is configured, so the suite still runs on a JSON-only checkout.

No lexicon content appears here — the probe words are ordinary Khasi
function words chosen for being common enough to exercise the backoff path.
"""
import math
import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.environ.get("DATABASE_URL", "").strip(),
    reason="no DATABASE_URL — PG n-gram backend not configured",
)


@pytest.fixture(scope="module")
def pair():
    """(file model, pg model), or skip when either is unavailable."""
    from khasi_spell.ngram import NgramLM
    from khasi_spell.ngram_pg import PgNgramLM

    url = os.environ["DATABASE_URL"].strip()
    if not PgNgramLM.table_exists(url):
        pytest.skip("ngram_counts absent or empty — "
                    "run scripts/migrate_ngrams_to_pg.py")
    try:
        f = NgramLM().load()
    except FileNotFoundError:
        pytest.skip("no ngrams.json.gz to compare against")
    return f, PgNgramLM(url).load()


SLOTS = [
    ("briew", ["ka", "sorkar"], ["ka", "la"]),
    ("shnong", ["ha", "ka"], ["ba", "la"]),
    ("bad", [], []),
    ("jingtip", ["ka"], ["ka"]),
    ("zzzznotaword", ["ka", "la"], ["bad", "ka"]),
]


def test_token_total_matches(pair):
    f, p = pair
    assert p.total == f.total


def test_counts_reported_match(pair):
    f, p = pair
    i = p.info()
    assert i["unigrams"] == len(f.uni)
    assert i["bigrams"] == len(f.bi)
    assert i["trigrams"] == len(f.tri)


@pytest.mark.parametrize("word,left,right", SLOTS)
def test_slot_score_identical(pair, word, left, right):
    f, p = pair
    assert p.slot_score(word, left, right) == pytest.approx(
        f.slot_score(word, left, right), rel=1e-12
    )


def test_sentence_logprob_identical(pair):
    f, p = pair
    words = ["ka", "sorkar", "ka", "la", "leh"]
    assert p.sentence_logprob(words) == pytest.approx(
        f.sentence_logprob(words), rel=1e-12
    )


def test_unseen_context_scores_at_floor_both_ways(pair):
    f, p = pair
    args = ("qqzzxx", ["yyww", "vvuu"], ["ttss", "rrqq"])
    assert p.slot_score(*args) == pytest.approx(f.slot_score(*args), rel=1e-12)


def test_one_query_per_slot(pair):
    """The batching is the point: a slot must not cost 15 round trips."""
    _, p = pair
    p._cache.clear()
    before = p.queries
    p.slot_score("briew", ["ka", "sorkar"], ["ka", "la"])
    assert p.queries - before == 1


def test_cache_absorbs_repeat_slots(pair):
    """Candidates competing for one slot share nearly all their keys."""
    _, p = pair
    p._cache.clear()
    p.slot_score("briew", ["ka", "sorkar"], ["ka", "la"])
    after_first = p.queries
    for _ in range(5):
        p.slot_score("briew", ["ka", "sorkar"], ["ka", "la"])
    assert p.queries == after_first, "repeat slot should be served from cache"


def test_missing_gram_is_cached_not_requeried(pair):
    """An unseen context must not be re-queried per candidate."""
    _, p = pair
    p._cache.clear()
    p.prefetch(["deliberately-absent-gram"])
    q = p.queries
    p.prefetch(["deliberately-absent-gram"])
    assert p.queries == q

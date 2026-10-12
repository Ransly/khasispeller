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


def test_a_dropped_connection_is_reopened(pair):
    """A managed database closes idle connections and restarts. Without a
    reconnect every later call raised InterfaceError, and context
    re-ranking — so /check — failed until the process restarted."""
    f, p = pair
    expected = f.slot_score("briew", ["ka", "sorkar"], ["ka", "la"])
    p._cache.clear()
    p._conn.close()
    assert p.slot_score("briew", ["ka", "sorkar"], ["ka", "la"]) == expected


def test_scores_are_safe_under_threads(pair):
    """The service calls slot_score() from several request threads; the
    LRU cache and the single connection are guarded by one lock."""
    from concurrent.futures import ThreadPoolExecutor
    f, p = pair
    p._cache_size = 50          # force constant eviction
    try:
        with ThreadPoolExecutor(8) as ex:
            got = list(ex.map(lambda s: p.slot_score(*s), SLOTS * 20))
        assert got == [f.slot_score(*s) for s in SLOTS * 20]
    finally:
        p._cache_size = 50_000


# ----------------------------------------------------------------------
# Timeouts (audit item 8)
# ----------------------------------------------------------------------
# Every request thread shares this one connection behind one lock, so a
# query that stalled used to stall them all, with no limit.

def test_the_connection_carries_a_statement_timeout(pair, monkeypatch):
    from khasi_spell.ngram_pg import PgNgramLM
    monkeypatch.setenv("KHASI_PG_STATEMENT_TIMEOUT_MS", "1234")
    p = PgNgramLM(os.environ["DATABASE_URL"].strip()).load()
    with p._conn.cursor() as cur:
        cur.execute("SHOW statement_timeout")
        assert cur.fetchone()[0] == "1234ms"


def test_a_query_over_the_timeout_is_cancelled_and_the_connection_survives(pair, monkeypatch):
    import psycopg2
    from khasi_spell.ngram_pg import PgNgramLM
    f, _ = pair
    monkeypatch.setenv("KHASI_PG_STATEMENT_TIMEOUT_MS", "100")
    p = PgNgramLM(os.environ["DATABASE_URL"].strip()).load()
    with pytest.raises(psycopg2.errors.QueryCanceled):
        with p._conn.cursor() as cur:
            cur.execute("SELECT pg_sleep(3)")
    slot = ("briew", ["ka", "sorkar"], ["ka", "la"])
    assert p.slot_score(*slot) == f.slot_score(*slot)


def test_a_timed_out_fetch_fails_once_instead_of_waiting_twice(pair, monkeypatch):
    """A dropped connection is retried once; a timeout is not, because the
    same query would only time out again while holding the shared lock."""
    import psycopg2
    from khasi_spell.ngram_pg import PgNgramLM
    p = PgNgramLM(os.environ["DATABASE_URL"].strip()).load()
    calls = []

    class Cursor:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def execute(self, *a):
            calls.append(a)
            raise psycopg2.errors.QueryCanceled("canceling statement due to statement timeout")

    class Conn:
        closed = False
        def cursor(self): return Cursor()

    p._conn = Conn()
    with pytest.raises(psycopg2.errors.QueryCanceled):
        p._fetch(["ka"])
    assert len(calls) == 1


def test_timeouts_come_from_the_environment(monkeypatch):
    from khasi_engine.database import pg_connect_timeout
    from khasi_spell.ngram_pg import statement_timeout_ms
    monkeypatch.delenv("KHASI_PG_CONNECT_TIMEOUT", raising=False)
    monkeypatch.delenv("KHASI_PG_STATEMENT_TIMEOUT_MS", raising=False)
    assert (pg_connect_timeout(), statement_timeout_ms()) == (10, 10_000)
    monkeypatch.setenv("KHASI_PG_CONNECT_TIMEOUT", "3")
    monkeypatch.setenv("KHASI_PG_STATEMENT_TIMEOUT_MS", "0")
    assert (pg_connect_timeout(), statement_timeout_ms()) == (3, 0)
    monkeypatch.setenv("KHASI_PG_CONNECT_TIMEOUT", "soon")
    assert pg_connect_timeout() == 10

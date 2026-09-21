"""
ngram_pg.py — the trigram model, served from PostgreSQL instead of RAM.

Why
───
`NgramLM` expands ngrams.json.gz into three Python dicts: 5.1 MB on disk,
**181 MB resident** for 838,445 counts. That is a 35x blow-up and almost all
of it is object overhead — the counts themselves are small integers.

Nothing about the scoring needs them resident. `slot_score()` does at most
15 point lookups on exact keys, and — the property this class is built on —
**every one of those keys is computable before any of them is issued**. So
they go to the database in a single round trip instead of fifteen, and the
same Stupid Backoff arithmetic runs against the result.

Batching is what makes this viable rather than merely possible. Issuing the
lookups one at a time would put a network round trip inside the inner loop
of the ranker; a sentence scoring five candidates across twenty slots would
spend hundreds of round trips where this spends twenty.

Drop-in: same public surface as NgramLM (`slot_score`, `sentence_logprob`,
`s_bigram`, `s_trigram`, `info`, `loaded`), so `speller._lm()` can return
either and nothing downstream knows the difference.

Correctness is not approximated. Given the same counts this returns the same
scores as NgramLM — `tests/test_ngram_pg.py` asserts that against the file
model rather than trusting it.
"""
from __future__ import annotations

import json
import math
import os
from collections import OrderedDict
from typing import Optional, Sequence

BOS, EOS = "<s>", "</s>"
BACKOFF = 0.4
FLOOR = 1e-10

# Grams held in memory after being fetched. The corpus is Zipfian and the
# function words that dominate lookups are a tiny set, so a small cache
# absorbs most of the traffic; 50k entries is a few MB against the 181 MB
# this class exists to avoid.
DEFAULT_CACHE = 50_000


class PgNgramLM:
    """Trigram LM with Stupid Backoff, counts fetched from PostgreSQL."""

    def __init__(self, database_url: Optional[str] = None,
                 cache_size: int = DEFAULT_CACHE):
        self._url = database_url or os.environ.get("DATABASE_URL", "").strip()
        self._conn = None
        self._cache: "OrderedDict[str, int]" = OrderedDict()
        self._cache_size = cache_size
        self.total = 0
        self.meta: dict = {}
        self._counts: dict = {}
        self._loaded = False
        # Diagnostics — how well the cache is doing, and how chatty we are.
        self.queries = 0
        self.fetched = 0
        self.hits = 0
        self.misses = 0

    # ------------------------------------------------------------------

    @staticmethod
    def table_exists(database_url: str) -> bool:
        """True when the n-gram tables are present and populated.

        Lets the caller choose a backend without assuming a migration has
        been run — an empty or absent table must fall back to the file, not
        silently score every candidate at the floor.
        """
        try:
            import psycopg2
            with psycopg2.connect(database_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT to_regclass('public.ngram_counts')")
                    if cur.fetchone()[0] is None:
                        return False
                    cur.execute("SELECT 1 FROM ngram_counts LIMIT 1")
                    return cur.fetchone() is not None
        except Exception:
            return False

    def load(self) -> "PgNgramLM":
        """Fetch only the scalars. The counts stay in the database."""
        if self._loaded:
            return self
        import psycopg2
        self._conn = psycopg2.connect(self._url)
        self._conn.autocommit = True
        with self._conn.cursor() as cur:
            cur.execute("SELECT key, value FROM ngram_meta")
            rows = dict(cur.fetchall())
        self.total = int(rows.get("total") or 0)
        self.meta = rows.get("meta") or {}
        self._counts = rows.get("counts") or {}
        if not self.total:
            raise RuntimeError(
                "ngram_meta has no token total — run "
                "scripts/migrate_ngrams_to_pg.py"
            )
        self._loaded = True
        return self

    @property
    def loaded(self) -> bool:
        return self._loaded

    def info(self) -> dict:
        self.load()
        return {
            "path": "postgresql:ngram_counts",
            "tokens": self.total,
            "unigrams": int(self._counts.get("unigrams", 0)),
            "bigrams": int(self._counts.get("bigrams", 0)),
            "trigrams": int(self._counts.get("trigrams", 0)),
            "backend": "postgresql",
            "cache": {"size": len(self._cache), "limit": self._cache_size,
                      "hits": self.hits, "misses": self.misses,
                      "queries": self.queries, "rows_fetched": self.fetched},
            **{k: v for k, v in self.meta.items()
               if k.startswith(("sentences", "min_"))},
        }

    # ------------------------------------------------------------------
    # Fetching
    # ------------------------------------------------------------------

    def _remember(self, gram: str, count: int) -> None:
        self._cache[gram] = count
        self._cache.move_to_end(gram)
        while len(self._cache) > self._cache_size:
            self._cache.popitem(last=False)

    def prefetch(self, grams) -> dict:
        """Resolve *grams* to counts, querying once for whatever is missing.

        A gram the table does not hold is cached as 0, so an unseen context
        is not re-queried on every candidate competing for the same slot —
        which is exactly the case the ranker generates.
        """
        self.load()
        out, missing = {}, []
        for g in grams:
            if g in self._cache:
                self._cache.move_to_end(g)
                out[g] = self._cache[g]
                self.hits += 1
            else:
                missing.append(g)
                self.misses += 1
        if missing:
            self.queries += 1
            with self._conn.cursor() as cur:
                cur.execute(
                    "SELECT gram, count FROM ngram_counts WHERE gram = ANY(%s)",
                    (list(set(missing)),),
                )
                found = dict(cur.fetchall())
            self.fetched += len(found)
            for g in missing:
                c = int(found.get(g, 0))
                self._remember(g, c)
                out[g] = c
        return out

    # ------------------------------------------------------------------
    # Scoring — identical arithmetic to NgramLM, over a fetched dict
    # ------------------------------------------------------------------

    @staticmethod
    def _keys_for_trigram(u: str, v: str, w: str) -> list:
        """Every key s_trigram(u,v,w) can touch, including its backoff path."""
        return [u, v, w, f"{u}\t{v}", f"{v}\t{w}", f"{u}\t{v}\t{w}"]

    def _p_uni(self, w: str, c: dict) -> float:
        n = c.get(w, 0)
        return n / self.total if n and self.total else FLOOR

    def _s_bigram(self, prev: str, w: str, c: dict) -> float:
        c_prev = c.get(prev, 0)
        if c_prev:
            n = c.get(f"{prev}\t{w}", 0)
            if n:
                return n / c_prev
        return BACKOFF * self._p_uni(w, c)

    def _s_trigram(self, u: str, v: str, w: str, c: dict) -> float:
        c_uv = c.get(f"{u}\t{v}", 0)
        if c_uv:
            n = c.get(f"{u}\t{v}\t{w}", 0)
            if n:
                return n / c_uv
        return BACKOFF * self._s_bigram(v, w, c)

    # Public single-shot forms, for parity with NgramLM.
    def s_bigram(self, prev: str, w: str) -> float:
        return self._s_bigram(prev, w, self.prefetch([prev, w, f"{prev}\t{w}"]))

    def s_trigram(self, u: str, v: str, w: str) -> float:
        return self._s_trigram(u, v, w, self.prefetch(self._keys_for_trigram(u, v, w)))

    def slot_score(self, word: str, left: Sequence[str],
                   right: Sequence[str]) -> float:
        """Log score for *word* in a slot — one query, not fifteen."""
        self.load()
        l2, l1 = (list(left) + [BOS, BOS])[-2:] if len(left) < 2 else list(left)[-2:]
        r1, r2 = (list(right) + [EOS, EOS])[:2] if len(right) < 2 else list(right)[:2]

        keys = (self._keys_for_trigram(l2, l1, word)
                + self._keys_for_trigram(l1, word, r1)
                + self._keys_for_trigram(word, r1, r2))
        c = self.prefetch(keys)

        return (math.log(max(self._s_trigram(l2, l1, word, c), FLOOR))
                + math.log(max(self._s_trigram(l1, word, r1, c), FLOOR))
                + math.log(max(self._s_trigram(word, r1, r2, c), FLOOR)))

    def sentence_logprob(self, words: Sequence[str]) -> float:
        self.load()
        padded = [BOS, BOS] + list(words) + [EOS]
        keys = []
        for i in range(2, len(padded)):
            keys += self._keys_for_trigram(padded[i - 2], padded[i - 1], padded[i])
        c = self.prefetch(keys)
        return sum(
            math.log(max(self._s_trigram(padded[i - 2], padded[i - 1],
                                         padded[i], c), FLOOR))
            for i in range(2, len(padded))
        )

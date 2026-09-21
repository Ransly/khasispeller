#!/usr/bin/env python3
"""
migrate_ngrams_to_pg.py — load data/ngrams.json.gz into PostgreSQL.

Why
───
`ngrams.json.gz` is 5.1 MB on disk and **181 MB resident**: json.load expands
838,445 counts into three Python dicts, a 35x blow-up that is pure object
overhead. On a 512 MB instance that is the difference between fitting and
not.

The access pattern does not need the dicts. `slot_score()` performs point
lookups on exact keys — at most 15 per call, all of them computable before
any of them is issued. That is an indexed table, not a hash map that has to
be resident.

    python3 scripts/migrate_ngrams_to_pg.py            # load
    python3 scripts/migrate_ngrams_to_pg.py --dry-run  # parse only
    python3 scripts/migrate_ngrams_to_pg.py --drop     # remove the tables

Reads DATABASE_URL from the environment, like migrate_json_to_pg.py.

Idempotent: ON CONFLICT DO UPDATE, so re-running after rebuilding the model
refreshes counts in place. It does NOT delete grams that disappeared from a
rebuilt model — pass --prune for that, same opt-in shape as the lexicon
migration and for the same reason.
"""
from __future__ import annotations

import argparse
import gzip
import json
import os
import sys
import time
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_MODEL = PROJECT_ROOT / "data" / "ngrams.json.gz"

SCHEMA_SQL = """
-- One row per gram. Keys are the model's own: tab-separated for n>1, so a
-- unigram can never collide with a bigram and a single table suffices.
CREATE TABLE IF NOT EXISTS ngram_counts (
    gram   TEXT     PRIMARY KEY,
    n      SMALLINT NOT NULL,
    count  INTEGER  NOT NULL
);

-- Scalars the model needs on every score (token total) plus build metadata.
CREATE TABLE IF NOT EXISTS ngram_meta (
    key    TEXT  PRIMARY KEY,
    value  JSONB NOT NULL
);

-- The PK already indexes `gram`, which is what point lookups use. `n` is
-- indexed only so info()/diagnostics can count per order without a seq scan.
CREATE INDEX IF NOT EXISTS idx_ngram_n ON ngram_counts (n);
"""

DROP_SQL = "DROP TABLE IF EXISTS ngram_counts; DROP TABLE IF EXISTS ngram_meta;"


def load_model(path: Path) -> dict:
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt", encoding="utf-8") as fh:
        return json.load(fh)


def rows_from(payload: dict):
    """Yield (gram, n, count) for every order in the model."""
    for order, key in ((1, "unigrams"), (2, "bigrams"), (3, "trigrams")):
        for gram, count in payload.get(key, {}).items():
            yield gram, order, int(count)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--batch", type=int, default=5000)
    ap.add_argument("--dry-run", action="store_true",
                    help="Parse and count only — no DB writes")
    ap.add_argument("--drop", action="store_true",
                    help="DROP the n-gram tables and exit (destructive)")
    ap.add_argument("--prune", action="store_true",
                    help="DELETE grams no longer in the model file")
    args = ap.parse_args(argv)

    url = os.environ.get("DATABASE_URL", "").strip()
    if not url and not args.dry_run:
        print("DATABASE_URL is not set.", file=sys.stderr)
        return 2

    path = Path(args.model)
    print(f"[ngrams] Reading {path} …")
    t0 = time.time()
    payload = load_model(path)
    counts = {k: len(payload.get(k, {})) for k in ("unigrams", "bigrams", "trigrams")}
    total_rows = sum(counts.values())
    print(f"[ngrams] Loaded {total_rows:,} grams in {time.time() - t0:.1f}s  {counts}")

    if args.dry_run:
        print("[ngrams] --dry-run: no DB writes")
        return 0

    import psycopg2
    import psycopg2.extras

    conn = psycopg2.connect(url)
    conn.autocommit = False
    try:
        with conn.cursor() as cur:
            if args.drop:
                print("[ngrams] DROPPING ngram tables")
                cur.execute(DROP_SQL)
                conn.commit()
                return 0

            cur.execute(SCHEMA_SQL)

            # Token total excludes sentence markers, matching NgramLM.load().
            uni = payload.get("unigrams", {})
            token_total = sum(v for k, v in uni.items() if k not in ("<s>", "</s>"))
            meta = dict(payload.get("meta", {}))
            for key, value in (("total", token_total), ("meta", meta),
                               ("counts", counts)):
                cur.execute(
                    "INSERT INTO ngram_meta (key, value) VALUES (%s, %s) "
                    "ON CONFLICT (key) DO UPDATE SET value = EXCLUDED.value",
                    (key, json.dumps(value)),
                )

            t0 = time.time()
            batch, done = [], 0
            for row in rows_from(payload):
                batch.append(row)
                if len(batch) >= args.batch:
                    psycopg2.extras.execute_values(
                        cur,
                        "INSERT INTO ngram_counts (gram, n, count) VALUES %s "
                        "ON CONFLICT (gram) DO UPDATE SET count = EXCLUDED.count",
                        batch,
                    )
                    done += len(batch)
                    batch.clear()
                    if done % 100000 == 0:
                        print(f"[ngrams]   {done:,}/{total_rows:,}")
            if batch:
                psycopg2.extras.execute_values(
                    cur,
                    "INSERT INTO ngram_counts (gram, n, count) VALUES %s "
                    "ON CONFLICT (gram) DO UPDATE SET count = EXCLUDED.count",
                    batch,
                )
                done += len(batch)

            if args.prune:
                cur.execute("SELECT gram FROM ngram_counts")
                in_db = {r[0] for r in cur.fetchall()}
                in_file = {g for g, _, _ in rows_from(payload)}
                orphans = in_db - in_file
                print(f"[ngrams] {len(orphans):,} gram(s) absent from the model")
                if orphans:
                    psycopg2.extras.execute_values(
                        cur, "DELETE FROM ngram_counts WHERE gram IN (%s)",
                        [(o,) for o in orphans],
                    )

        conn.commit()
        print(f"[ngrams] Upserted {done:,} rows in {time.time() - t0:.1f}s")
        print(f"[ngrams] token total = {token_total:,}")
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())

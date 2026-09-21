"""
The three data sources must agree.

`khasi_db.json` is the editing source of truth. **PostgreSQL is the single
served source**: every top-level block the engine reads has a table, so a
database is a complete picture of the system on its own.

`khasi_db_extras.json` survives only as a compatibility fallback for a
database created before `quarantine_meta`, `grammar_schema_meta` and
`production_readiness_meta` existed. It is still kept in step, because a
fallback that has silently rotted is worse than none.

That is not hypothetical. `build_db_extras.py` asked to be re-run "whenever
khasi_db.json changes, alongside migrate_json_to_pg.py", and because it was
a step to remember rather than an automatic one, on 2026-09-10 the sidecar
was found 28 quarantine records behind the JSON — every withdrawal made
that day was missing from what production could see. `migrate_json_to_pg.py`
now rebuilds the sidecar itself; these tests are what stop the two drifting
apart again.
"""
import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
MAIN = ROOT / "data" / "khasi_db.json"
SIDECAR = ROOT / "data" / "khasi_db_extras.json"

# Blocks the sidecar exists to carry, i.e. everything PostgreSQL does not.
SIDECAR_BLOCKS = ("quarantine", "grammar_schema",
                  "production_readiness", "demo_examples")


@pytest.fixture(scope="module")
def main_db():
    return json.loads(MAIN.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def sidecar():
    if not SIDECAR.is_file():
        pytest.skip("sidecar not built")
    return json.loads(SIDECAR.read_text(encoding="utf-8"))


@pytest.mark.parametrize("block", SIDECAR_BLOCKS)
def test_sidecar_matches_the_json(main_db, sidecar, block):
    """Every block the sidecar carries must equal the one in the JSON."""
    a, b = main_db.get(block), sidecar.get(block)
    assert a is not None, f"{block} missing from khasi_db.json"
    assert b is not None, f"{block} missing from the sidecar"
    assert len(a) == len(b), (
        f"{block}: khasi_db.json has {len(a)}, the sidecar has {len(b)} — "
        "run scripts/migrate_json_to_pg.py, which rebuilds the sidecar")
    assert json.dumps(a, sort_keys=True, ensure_ascii=False) == \
        json.dumps(b, sort_keys=True, ensure_ascii=False), \
        f"{block} differs between khasi_db.json and the sidecar"


def test_quarantine_ids_are_identical(main_db, sidecar):
    """Named separately: this is the block that actually drifted."""
    a = {e.get("entry_id") for e in main_db.get("quarantine", [])
         if isinstance(e, dict)}
    b = {e.get("entry_id") for e in sidecar.get("quarantine", [])
         if isinstance(e, dict)}
    assert a == b, (f"{len(a - b)} withdrawn entries are in khasi_db.json but "
                    f"not in the sidecar; {len(b - a)} the other way round")


def test_postgres_carries_every_served_block(main_db):
    """PostgreSQL alone must be able to serve the engine.

    Before 2026-09-10 four blocks lived only in the sidecar, which made the
    database an incomplete picture and let the file drift 28 quarantine
    records behind. Each now has a table.
    """
    import os
    if not os.environ.get("DATABASE_URL"):
        pytest.skip("DATABASE_URL not set — JSON mode")
    try:
        import psycopg2
        conn = psycopg2.connect(os.environ["DATABASE_URL"])
    except Exception as exc:                       # pragma: no cover
        pytest.skip(f"PostgreSQL unreachable: {exc}")
    try:
        with conn.cursor() as cur:
            for table, block in (("quarantine_meta", "quarantine"),
                                 ("grammar_schema_meta", "grammar_schema"),
                                 ("production_readiness_meta",
                                  "production_readiness")):
                cur.execute(f"select payload from {table} where id = 1")
                row = cur.fetchone()
                assert row and row[0], (
                    f"{table} is empty — run scripts/migrate_json_to_pg.py")
                assert len(row[0]) == len(main_db[block]), (
                    f"{table} has {len(row[0])} but khasi_db.json has "
                    f"{len(main_db[block])}")
    finally:
        conn.close()


def test_postgres_matches_the_json_when_configured(main_db):
    """Row-for-row parity with the database, when one is configured."""
    import os
    if not os.environ.get("DATABASE_URL"):
        pytest.skip("DATABASE_URL not set — JSON mode")
    # psycopg2, the driver migrate_json_to_pg.py uses. An earlier version of
    # this test imported `psycopg` (v3), which is not installed, so it
    # skipped silently — and a parity test that never runs is worse than
    # none, because it reads as green.
    try:
        import psycopg2
        conn = psycopg2.connect(os.environ["DATABASE_URL"])
    except Exception as exc:                       # pragma: no cover
        pytest.skip(f"PostgreSQL unreachable: {exc}")
    try:
        with conn.cursor() as cur:
            cur.execute("select count(*) from lexicon_entries")
            n_pg = cur.fetchone()[0]
            cur.execute("select entry_id from lexicon_entries")
            pg_ids = {r[0] for r in cur.fetchall()}
    finally:
        conn.close()
    json_ids = {e.get("entry_id") for e in main_db["lexicon"]}
    assert n_pg == len(main_db["lexicon"]), (
        f"PostgreSQL has {n_pg} rows, khasi_db.json has "
        f"{len(main_db['lexicon'])} — re-run scripts/migrate_json_to_pg.py")
    assert pg_ids == json_ids, (
        f"{len(json_ids - pg_ids)} entries are in the JSON but not the "
        f"database; {len(pg_ids - json_ids)} are in the database but not the "
        "JSON (--prune removes those)")

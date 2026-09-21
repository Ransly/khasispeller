#!/usr/bin/env python3
"""
scripts/migrate_json_to_pg.py
─────────────────────────────────────────────────────────────────────────────
ONE-TIME migration: reads data/khasi_db.json and loads every entry into the
PostgreSQL database that Render provisions when DATABASE_URL is set.

Run this ONCE after you have created the Render Postgres database and before
you first deploy (or after adding the DATABASE_URL env var to an existing
deploy).

Usage
─────
  # From your project root:
  DATABASE_URL="postgresql://user:pass@host/db" python scripts/migrate_json_to_pg.py

  # Or if DATABASE_URL is already in your environment:
  python scripts/migrate_json_to_pg.py

  # Custom JSON path:
  DATABASE_URL="..." python scripts/migrate_json_to_pg.py --db data/khasi_db.json

Options
───────
  --db PATH       Path to khasi_db.json  (default: data/khasi_db.json)
  --batch N       Insert batch size      (default: 200)
  --dry-run       Parse + validate only, no DB writes
  --reset         DROP and recreate all tables before migrating (destructive!)
  --prune         DELETE rows whose entry_id is no longer in the JSON

Why --prune exists
──────────────────
This script upserts. It has no way to notice that an entry has been REMOVED
from khasi_db.json, so a withdrawn record stays live in PostgreSQL for ever.

That is not academic. Entries withdrawn to `quarantine` are meant to stop
being offered — `baiñ-bain`, a reduplication with the tilde on only one half,
was quarantined on 6 September 2026 and remained in the database, where the
running service still read it and still offered it as a legitimate
alternative spelling. Deleting it from the JSON changed nothing until the row
went too.

--prune is opt-in because it deletes. It lists every row it would remove and
refuses to run without --db pointing at the file those ids came from.

What it does
────────────
  1. Reads and parses khasi_db.json
  2. Creates tables if they don't exist (idempotent — safe to re-run)
  3. Upserts meta / phonology / morphology singleton blobs
  4. Upserts every lexicon entry in batches using ON CONFLICT DO UPDATE
     so partial runs can be safely resumed
  5. Prints a summary of inserted / updated / skipped counts

Requirements
────────────
  pip install psycopg2-binary
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from typing import Optional

# ── Make sure the project root is on sys.path so we can import khasi_engine ──
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))


# ---------------------------------------------------------------------------
# Schema DDL
# ---------------------------------------------------------------------------

SCHEMA_SQL = """
-- Singleton metadata blobs (one row each, id=1)
CREATE TABLE IF NOT EXISTS db_meta (
    id      INTEGER PRIMARY KEY DEFAULT 1,
    payload JSONB   NOT NULL
);

CREATE TABLE IF NOT EXISTS phonology_meta (
    id      INTEGER PRIMARY KEY DEFAULT 1,
    payload JSONB   NOT NULL
);

CREATE TABLE IF NOT EXISTS morphology_meta (
    id      INTEGER PRIMARY KEY DEFAULT 1,
    payload JSONB   NOT NULL
);

CREATE TABLE IF NOT EXISTS demo_examples_meta (
    id      INTEGER PRIMARY KEY DEFAULT 1,
    payload JSONB   NOT NULL
);

-- The blocks that used to live only in the khasi_db_extras.json sidecar.
-- PostgreSQL is now the single served source, so they belong here beside
-- everything else the engine reads at run time.
CREATE TABLE IF NOT EXISTS quarantine_meta (
    id      INTEGER PRIMARY KEY DEFAULT 1,
    payload JSONB   NOT NULL
);

CREATE TABLE IF NOT EXISTS grammar_schema_meta (
    id      INTEGER PRIMARY KEY DEFAULT 1,
    payload JSONB   NOT NULL
);

CREATE TABLE IF NOT EXISTS production_readiness_meta (
    id      INTEGER PRIMARY KEY DEFAULT 1,
    payload JSONB   NOT NULL
);

-- Main lexicon table
CREATE TABLE IF NOT EXISTS lexicon_entries (
    entry_id        TEXT        PRIMARY KEY,
    surface         TEXT        NOT NULL DEFAULT '',
    normalized      TEXT        NOT NULL DEFAULT '',
    variants        TEXT[]      NOT NULL DEFAULT '{}',
    root            TEXT,
    root_id         TEXT,
    pos             TEXT,
    gender          TEXT,
    clitic          TEXT,
    -- Phase 8: derived from proclitic. `number` ∈ {singular, plural};
    -- `diminutive` is true only for the `i` clitic (other clitics → false;
    -- no clitic → NULL). Populated by apply_proclitic_derivation.py.
    number          TEXT,
    diminutive      BOOLEAN,
    morph_type      TEXT,
    morph_canonical TEXT,
    morph_structure JSONB       NOT NULL DEFAULT '[]',
    morph_features  JSONB       NOT NULL DEFAULT '{}',
    morph_rules     JSONB       NOT NULL DEFAULT '[]',
    phon_surface    TEXT,
    phon_derived    JSONB       NOT NULL DEFAULT '{}',
    phon_flags      JSONB       NOT NULL DEFAULT '{}',
    gloss           TEXT[]      NOT NULL DEFAULT '{}',
    domain          TEXT,
    related         JSONB       NOT NULL DEFAULT '[]',
    better_forms    TEXT[]      NOT NULL DEFAULT '{}',
    origin          TEXT,
    source_file     TEXT,
    loan_origin     TEXT,
    loan_type       TEXT,
    previous_id     TEXT,
    is_valid        BOOLEAN     NOT NULL DEFAULT TRUE,
    confidence      NUMERIC(4,3) NOT NULL DEFAULT 1.0,
    needs_review    BOOLEAN     NOT NULL DEFAULT FALSE,
    review_reasons  TEXT[]      NOT NULL DEFAULT '{}',
    raw_notes       TEXT,
    -- v4.3-only enrichment fields. Stored as targeted columns instead of
    -- a single bloated `enriched_payload` to keep memory low on Render's
    -- 512 MB free tier (full payload duplication added ~300 MB resident).
    -- _load_payload_from_pg merges these back into the enriched shape.
    semantic_features JSONB    NOT NULL DEFAULT '{}',
    ontology_review   JSONB    NOT NULL DEFAULT '{}',
    domain_source     TEXT,
    homograph_group   TEXT,
    gloss_normalized  TEXT,
    created_at      TIMESTAMPTZ NOT NULL DEFAULT now(),
    updated_at      TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Evolve an EXISTING table (CREATE TABLE IF NOT EXISTS is a no-op on it):
-- explicitly ADD the v4.3 targeted columns. Idempotent and safe on a fresh
-- install too. NOTE: enriched_payload is NOT dropped here — see DROP_LEGACY_SQL
-- which runs only after a successful upsert pass so we never lose data on a
-- partial run.
ALTER TABLE lexicon_entries
    ADD COLUMN IF NOT EXISTS semantic_features JSONB NOT NULL DEFAULT '{}',
    ADD COLUMN IF NOT EXISTS ontology_review   JSONB NOT NULL DEFAULT '{}',
    ADD COLUMN IF NOT EXISTS domain_source     TEXT,
    ADD COLUMN IF NOT EXISTS homograph_group   TEXT,
    ADD COLUMN IF NOT EXISTS gloss_normalized  TEXT,
    -- Phase 8 proclitic-derived grammar fields (added 2026-05-24).
    ADD COLUMN IF NOT EXISTS number            TEXT,
    ADD COLUMN IF NOT EXISTS diminutive        BOOLEAN;

-- Indexes for the engine's hot lookup paths
CREATE INDEX IF NOT EXISTS idx_lexicon_surface    ON lexicon_entries (surface);
CREATE INDEX IF NOT EXISTS idx_lexicon_root       ON lexicon_entries (root);
CREATE INDEX IF NOT EXISTS idx_lexicon_pos        ON lexicon_entries (pos);
CREATE INDEX IF NOT EXISTS idx_lexicon_loan       ON lexicon_entries (loan_origin);
CREATE INDEX IF NOT EXISTS idx_lexicon_needs_review ON lexicon_entries (needs_review)
    WHERE needs_review = TRUE;
CREATE INDEX IF NOT EXISTS idx_lexicon_homograph  ON lexicon_entries (homograph_group)
    WHERE homograph_group IS NOT NULL;
"""

# Run ONLY after all upserts succeed — drops the legacy bloated column.
# Kept separate so an interrupted migration (SSL drop, OOM, etc.) never loses
# enriched_payload data before the new targeted columns are populated.
DROP_LEGACY_SQL = """
ALTER TABLE lexicon_entries DROP COLUMN IF EXISTS enriched_payload;
"""

RESET_SQL = """
DROP TABLE IF EXISTS lexicon_entries CASCADE;
DROP TABLE IF EXISTS db_meta CASCADE;
DROP TABLE IF EXISTS phonology_meta CASCADE;
DROP TABLE IF EXISTS morphology_meta CASCADE;
DROP TABLE IF EXISTS demo_examples_meta CASCADE;
DROP TABLE IF EXISTS quarantine_meta CASCADE;
DROP TABLE IF EXISTS grammar_schema_meta CASCADE;
DROP TABLE IF EXISTS production_readiness_meta CASCADE;
"""

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _normalise_url(url: str) -> str:
    """asyncpg-style URLs aren't accepted by psycopg2 — strip the driver tag."""
    if url.startswith("postgresql+asyncpg://"):
        return "postgresql://" + url[len("postgresql+asyncpg://"):]
    return url


def _is_enriched(entry: dict) -> bool:
    return "form" in entry and isinstance(entry.get("form"), dict) and "surface" in entry["form"]


def _entry_to_row(entry: dict) -> dict:
    """
    Convert one enriched (v3.2) entry dict into a flat dict suitable for
    INSERT into lexicon_entries.  Handles both enriched shape (v3.2) and
    flat shape (v2.0) gracefully.
    """
    if not _is_enriched(entry):
        # v2.0 flat shape — adapt inline
        ap   = entry.get("affix_profile") or {}
        mf   = entry.get("morphological_flags") or {}
        pf   = entry.get("phonological_flags") or {}
        gloss_str = entry.get("english_gloss") or ""
        gloss_list = [g.strip() for g in gloss_str.split(";") if g.strip()]
        return {
            "entry_id":        entry.get("entry_id", ""),
            "surface":         entry.get("surface_form", ""),
            "normalized":      entry.get("surface_form", ""),
            "variants":        [],
            "root":            entry.get("root_lemma"),
            "root_id":         None,
            "pos":             entry.get("grammatical_class"),
            "gender":          entry.get("gender") if entry.get("gender") not in ("none", None, "") else None,
            "clitic":          entry.get("clitic") if entry.get("clitic") not in ("none", None, "") else None,
            "morph_type":      "compound" if mf.get("is_compound") else ("prefixed" if ap.get("prefix") else "root"),
            "morph_canonical": None,
            "morph_structure": [],
            "morph_features":  {"reduplication": mf.get("reduplication_type"), "is_tien_kynnoh": bool(mf.get("is_tien_kynnoh"))},
            "morph_rules":     [],
            "phon_surface":    entry.get("surface_form"),
            "phon_derived":    {"pattern": pf.get("syllable_structure", "")},
            "phon_flags":      {},
            "gloss":           gloss_list,
            "domain":          None,
            "related":         list(entry.get("related_entries") or []),
            "better_forms":    [],
            "origin":          entry.get("source"),
            "source_file":     None,
            "loan_origin":     entry.get("loan_origin") if entry.get("loan_origin") not in ("none", None, "") else None,
            "loan_type":       entry.get("shim_kylliang_type"),
            "previous_id":     None,
            "is_valid":        True,
            "confidence":      1.0,
            "needs_review":    False,
            "review_reasons":  [],
            "raw_notes":       entry.get("notes") or None,
        }

    # v3.2 enriched shape
    form  = entry.get("form") or {}
    lemma = entry.get("lemma") or {}
    gram  = entry.get("grammar") or {}
    morph = entry.get("morphology") or {}
    phon  = entry.get("phonology") or {}
    sem   = entry.get("semantics") or {}
    rels  = entry.get("lexical_relations") or {}
    src   = entry.get("source") or {}
    val   = entry.get("validation") or {}

    return {
        "entry_id":        entry["entry_id"],
        "surface":         form.get("surface", ""),
        "normalized":      form.get("normalized", ""),
        "variants":        list(form.get("variants") or []),
        "root":            lemma.get("root"),
        "root_id":         lemma.get("root_id"),
        "pos":             gram.get("pos"),
        "gender":          gram.get("gender"),
        "clitic":          gram.get("clitic"),
        # Phase 8 proclitic-derived fields. Pass through as-is; the load
        # path merges them back into the entry's `grammar` block.
        "number":          gram.get("number"),
        "diminutive":      gram.get("diminutive"),
        "morph_type":      morph.get("type"),
        "morph_canonical": morph.get("canonical"),
        "morph_structure": morph.get("structure") or [],
        "morph_features":  morph.get("features") or {},
        "morph_rules":     morph.get("morphology_rule") or [],
        "phon_surface":    phon.get("surface"),
        "phon_derived":    phon.get("derived") or {},
        "phon_flags":      phon.get("flags") or {},
        "gloss":           list(sem.get("gloss") or []),
        "domain":          sem.get("domain"),
        "related":         list(rels.get("related") or []),
        "better_forms":    list(rels.get("better_forms") or []),
        "origin":          src.get("origin"),
        "source_file":     src.get("source_file"),
        "loan_origin":     src.get("loan_origin"),
        "loan_type":       src.get("loan_type"),
        "previous_id":     src.get("previous_entry_id"),
        "is_valid":        bool(val.get("is_valid", True)),
        # `.get(..., default)` only fires when the key is *missing* — an
        # explicit `null` value still returns None. Phase 8 skeletal
        # entries (kh_AS_*, kh_KK_*) carry validation.confidence: null,
        # so guard the float() against that case before falling back to 1.0.
        "confidence":      float(val["confidence"]) if val.get("confidence") is not None else 1.0,
        "needs_review":    bool(val.get("needs_review", False)),
        "review_reasons":  list(val.get("review_reasons") or []),
        "raw_notes":       entry.get("raw_notes"),
        # v4.3 targeted enrichment fields — kept as small targeted columns
        # rather than a single bloated enriched_payload JSONB. _load_payload_
        # from_pg merges these back into the enriched shape at boot.
        "semantic_features": sem.get("semantic_features") or {},
        "ontology_review":   sem.get("ontology_review") or {},
        "domain_source":     sem.get("domain_source"),
        "homograph_group":   rels.get("homograph_group"),
        "gloss_normalized":  sem.get("gloss_normalized"),
    }


# ---------------------------------------------------------------------------
# Main migration
# ---------------------------------------------------------------------------

def migrate(
    database_url: str,
    db_path: Path,
    batch_size: int = 200,
    dry_run: bool = False,
    reset: bool = False,
    prune: bool = False,
) -> None:
    import psycopg2
    import psycopg2.extras

    url = _normalise_url(database_url)

    # ── Load JSON ────────────────────────────────────────────────────────────
    print(f"[migrate] Reading {db_path} …", flush=True)
    t0 = time.time()
    with open(db_path, encoding="utf-8") as f:
        raw = json.load(f)
    elapsed = time.time() - t0
    entries = raw.get("lexicon", [])
    print(f"[migrate] Loaded {len(entries):,} entries in {elapsed:.1f}s")

    if dry_run:
        print("[migrate] --dry-run: validating rows only (no DB writes)")
        errors = 0
        for i, e in enumerate(entries):
            try:
                row = _entry_to_row(e)
                if not row["entry_id"]:
                    print(f"  [WARN] entry #{i} has no entry_id — skipping")
                    errors += 1
            except Exception as exc:
                print(f"  [ERROR] entry #{i} ({e.get('entry_id', '?')}): {exc}")
                errors += 1
        print(f"[migrate] Dry-run complete. {len(entries) - errors:,} valid, {errors} errors.")
        return

    # ── Connect ──────────────────────────────────────────────────────────────
    print(f"[migrate] Connecting to PostgreSQL …")
    conn = psycopg2.connect(url)
    conn.autocommit = False

    try:
        with conn.cursor() as cur:
            if reset:
                print("[migrate] --reset: dropping existing tables …")
                cur.execute(RESET_SQL)
                conn.commit()
                print("[migrate] Tables dropped.")

            # ── Create schema ────────────────────────────────────────────────
            print("[migrate] Ensuring schema …")
            cur.execute(SCHEMA_SQL)
            conn.commit()
            print("[migrate] Schema ready.")

            # ── Singleton blobs ──────────────────────────────────────────────
            for table, key in (
                ("db_meta",            "meta"),
                ("phonology_meta",     "phonology"),
                ("morphology_meta",    "morphology"),
                ("demo_examples_meta", "demo_examples"),
                # Formerly sidecar-only. See the schema comment.
                ("quarantine_meta",            "quarantine"),
                ("grammar_schema_meta",        "grammar_schema"),
                ("production_readiness_meta",  "production_readiness"),
            ):
                blob = raw.get(key)
                if blob:
                    cur.execute(
                        f"INSERT INTO {table} (id, payload) VALUES (1, %s) "
                        f"ON CONFLICT (id) DO UPDATE SET payload = EXCLUDED.payload",
                        (psycopg2.extras.Json(blob),),
                    )
                    print(f"[migrate] Upserted {table}.")
            conn.commit()

            # ── Lexicon entries in batches ────────────────────────────────────
            COLS = [
                "entry_id", "surface", "normalized", "variants",
                "root", "root_id", "pos", "gender", "clitic",
                "number", "diminutive",   # Phase 8 proclitic-derived
                "morph_type", "morph_canonical", "morph_structure",
                "morph_features", "morph_rules",
                "phon_surface", "phon_derived", "phon_flags",
                "gloss", "domain", "related", "better_forms",
                "origin", "source_file", "loan_origin", "loan_type", "previous_id",
                "is_valid", "confidence", "needs_review", "review_reasons",
                "raw_notes",
                # v4.3 enrichment as targeted columns (replaces enriched_payload)
                "semantic_features", "ontology_review",
                "domain_source", "homograph_group", "gloss_normalized",
            ]
            placeholders = ", ".join(f"%({c})s" for c in COLS)
            update_set   = ", ".join(
                f"{c}=EXCLUDED.{c}" for c in COLS if c != "entry_id"
            )
            SQL = (
                f"INSERT INTO lexicon_entries ({', '.join(COLS)}) "
                f"VALUES ({placeholders}) "
                f"ON CONFLICT (entry_id) DO UPDATE SET {update_set}, updated_at = now()"
            )

            inserted = 0
            skipped  = 0
            t1 = time.time()

            batch: list[dict] = []
            for i, entry in enumerate(entries):
                try:
                    row = _entry_to_row(entry)
                except Exception as exc:
                    print(f"  [SKIP] entry #{i} ({entry.get('entry_id','?')}): {exc}")
                    skipped += 1
                    continue

                if not row["entry_id"]:
                    print(f"  [SKIP] entry #{i}: no entry_id")
                    skipped += 1
                    continue

                # psycopg2 needs Python lists/Json for array/jsonb columns
                row["variants"]        = list(row["variants"] or [])
                row["morph_structure"] = psycopg2.extras.Json(row["morph_structure"])
                row["morph_features"]  = psycopg2.extras.Json(row["morph_features"])
                row["morph_rules"]     = psycopg2.extras.Json(row["morph_rules"])
                row["phon_derived"]    = psycopg2.extras.Json(row["phon_derived"])
                row["phon_flags"]      = psycopg2.extras.Json(row["phon_flags"])
                row["related"]         = psycopg2.extras.Json(row["related"])
                row["gloss"]           = list(row["gloss"] or [])
                row["better_forms"]    = list(row["better_forms"] or [])
                row["review_reasons"]  = list(row["review_reasons"] or [])
                # v4.3 targeted enrichment fields (replaces the bloated
                # enriched_payload approach which doubled resident memory).
                # Saves ~300 MB by storing only the fields the flat columns
                # don't already cover.
                row["semantic_features"] = psycopg2.extras.Json(row.get("semantic_features") or {})
                row["ontology_review"]   = psycopg2.extras.Json(row.get("ontology_review") or {})

                batch.append(row)

                if len(batch) >= batch_size:
                    psycopg2.extras.execute_batch(cur, SQL, batch)
                    conn.commit()
                    inserted += len(batch)
                    batch = []
                    pct = 100 * inserted / len(entries)
                    elapsed2 = time.time() - t1
                    rate = inserted / elapsed2 if elapsed2 > 0 else 0
                    print(
                        f"  {inserted:,}/{len(entries):,} ({pct:.0f}%)  "
                        f"{rate:.0f} rows/s",
                        flush=True,
                    )

            # flush remainder
            if batch:
                psycopg2.extras.execute_batch(cur, SQL, batch)
                conn.commit()
                inserted += len(batch)

            # All upserts succeeded — now safe to drop the legacy column.
            # Doing it here (not in SCHEMA_SQL) means an SSL drop or OOM mid-
            # migration leaves enriched_payload intact for a clean re-run.
            print("[migrate] Dropping legacy enriched_payload column …")
            cur.execute(DROP_LEGACY_SQL)
            conn.commit()
            print("[migrate] Legacy column dropped.")

            # Rows the JSON no longer contains. The upsert above cannot see a
            # deletion, so without this a withdrawn entry stays live in the
            # database and the service keeps serving it.
            cur.execute("SELECT entry_id, surface FROM lexicon_entries")
            in_db = {r[0]: r[1] for r in cur.fetchall()}
            in_json = {e.get("entry_id") for e in entries}
            orphans = sorted(k for k in in_db if k not in in_json)
            if orphans:
                print(f"[migrate] {len(orphans)} row(s) in PostgreSQL are absent "
                      f"from the JSON:")
                for eid in orphans[:20]:
                    print(f"             {eid}  {in_db[eid]!r}")
                if len(orphans) > 20:
                    print(f"             … and {len(orphans) - 20} more")
                if prune:
                    psycopg2.extras.execute_batch(
                        cur, "DELETE FROM lexicon_entries WHERE entry_id = %s",
                        [(eid,) for eid in orphans])
                    conn.commit()
                    print(f"[migrate] pruned {len(orphans)} row(s).")
                else:
                    print("[migrate] NOT deleted. Re-run with --prune to remove "
                          "them, or they stay live in the served lexicon.")
            else:
                print("[migrate] No orphaned rows — PostgreSQL matches the JSON.")

        # The sidecar is the other derived artefact and must move with the
        # database, or the two disagree. `khasi_db.json` holds four blocks
        # PostgreSQL does not carry — quarantine, grammar_schema,
        # production_readiness, demo_examples — and under DATABASE_URL the
        # engine reads them from `data/khasi_db_extras.json`.
        #
        # build_db_extras.py says "re-run this whenever khasi_db.json
        # changes, alongside migrate_json_to_pg.py". Being a step someone had
        # to remember, it was forgotten: on 2026-09-10 the sidecar was found
        # 28 quarantine records behind the JSON, so the served system had an
        # incomplete audit trail. Rebuilding it here makes the coupling
        # automatic instead of remembered.
        try:
            import importlib.util as _ilu
            import sys as _sys
            _spec = _ilu.spec_from_file_location(
                "build_db_extras",
                Path(__file__).with_name("build_db_extras.py"))
            _mod = _ilu.module_from_spec(_spec)
            _spec.loader.exec_module(_mod)
            _argv, _sys.argv = _sys.argv, ["build_db_extras"]
            try:
                print("[migrate] rebuilding the sidecar khasi_db_extras.json …")
                _mod.main()
            finally:
                _sys.argv = _argv
        except Exception as _exc:          # never fail the migration for this
            print(f"[migrate] WARNING: sidecar rebuild failed: {_exc}")
            print("[migrate] run scripts/build_db_extras.py by hand "
                  "before serving.")

        elapsed_total = time.time() - t0
        print(
            f"\n[migrate] ✓ Done in {elapsed_total:.1f}s\n"
            f"          Inserted/updated : {inserted:,}\n"
            f"          Skipped (errors) : {skipped:,}\n"
        )

    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Migrate khasi_db.json → PostgreSQL (one-time setup)"
    )
    parser.add_argument(
        "--db",
        default=str(PROJECT_ROOT / "data" / "khasi_db.json"),
        help="Path to khasi_db.json (default: data/khasi_db.json)",
    )
    parser.add_argument(
        "--batch",
        type=int,
        default=200,
        help="Batch insert size (default: 200)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Parse and validate only — no DB writes",
    )
    parser.add_argument(
        "--prune",
        action="store_true",
        help="DELETE rows whose entry_id is no longer in the JSON "
             "(withdrawn entries; detection always runs, deletion is opt-in)",
    )
    parser.add_argument(
        "--reset",
        action="store_true",
        help="DROP all tables before migrating (destructive!)",
    )
    args = parser.parse_args()

    database_url = os.environ.get("DATABASE_URL", "").strip()
    if not database_url:
        print(
            "ERROR: DATABASE_URL environment variable is not set.\n"
            "Export it first:\n"
            "  export DATABASE_URL='postgresql://user:pass@host/db'\n"
            "  python scripts/migrate_json_to_pg.py",
            file=sys.stderr,
        )
        sys.exit(1)

    db_path = Path(args.db)
    if not db_path.exists():
        print(f"ERROR: JSON file not found: {db_path}", file=sys.stderr)
        sys.exit(1)

    migrate(
        database_url=database_url,
        db_path=db_path,
        batch_size=args.batch,
        dry_run=args.dry_run,
        reset=args.reset,
        prune=args.prune,
    )


if __name__ == "__main__":
    main()

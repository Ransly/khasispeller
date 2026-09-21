#!/usr/bin/env python3
"""
build_db_extras.py — write the sidecar the PostgreSQL path needs.

PostgreSQL carries only `meta`, `phonology`, `morphology` and `lexicon`.
`khasi_db.json` also holds four other top-level blocks, and under
DATABASE_URL the engine looks for them in `data/khasi_db_extras.json`:

    quarantine            834 withdrawn records — the audit trail for every
                          entry removed during the OCR repair, and the thing
                          that makes those withdrawals reversible
    grammar_schema        7 keys
    production_readiness  9 keys
    demo_examples         UI samples

Without the sidecar the engine still starts, but those blocks are silently
absent: `/v2/health` and the quarantine endpoints see nothing, and the
withdrawal evidence is gone from the running system. The file is small
(a fraction of the 67 MB lexicon) precisely so it can be read on a 512 MB
tier without re-parsing the whole database.

Re-run this whenever khasi_db.json changes, alongside migrate_json_to_pg.py.

    python3 scripts/build_db_extras.py
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PG_OWNED = {"meta", "phonology", "morphology", "lexicon"}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", default=str(ROOT / "data" / "khasi_db.json"))
    ap.add_argument("--out", default=str(ROOT / "data" / "khasi_db_extras.json"))
    args = ap.parse_args()

    data = json.loads(Path(args.db).read_text(encoding="utf-8"))
    extras = {k: v for k, v in data.items() if k not in PG_OWNED}
    # `meta` is duplicated deliberately: PostgreSQL is authoritative for it,
    # but keeping a copy here means the sidecar is self-describing if it is
    # ever read on its own.
    extras["_meta_copy"] = {k: data.get("meta", {}).get(k)
                            for k in ("version", "schema_version")}

    out = Path(args.out)
    out.write_text(json.dumps(extras, ensure_ascii=False, indent=1), encoding="utf-8")
    for k, v in extras.items():
        n = len(v) if hasattr(v, "__len__") else "-"
        print(f"  {k:<22} {type(v).__name__:<6} len={n}")
    print(f"wrote {out} ({out.stat().st_size / 1024:.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Run the test suite against a throwaway local PostgreSQL — the backend
production uses — without touching any real database.

    python3 scripts/test_with_local_postgres.py            # whole suite
    python3 scripts/test_with_local_postgres.py -k realword   # pytest args pass through

Why
───
Production reads the lexicon and the n-gram model from PostgreSQL, but the
suite normally runs on data/khasi_db.json, where the 15 PostgreSQL tests skip
themselves. The only database at hand is usually the one in .env, which is not
local, and a test run must never be pointed at it. This starts a private
server instead, so that path is tested too.

What it does
────────────
  1. Starts PostgreSQL on 127.0.0.1:55432 (no password, no Unix socket) with
     its data in a temporary folder.
  2. Loads the lexicon and the n-gram model with the project's own migration
     scripts, exactly as for Render. The lexicon migration also rebuilds
     data/khasi_db_extras.json from data/khasi_db.json, as it always does.
  3. Runs pytest with DATABASE_URL set to that server and .env ignored.
  4. Stops the server and deletes the folder, also on failure.

Needs the PostgreSQL server programs (initdb, pg_ctl — the Ubuntu package
postgresql-16 puts them in /usr/lib/postgresql/16/bin; set PGBIN otherwise),
psycopg2, and the licensed lexicon at data/khasi_db.json (see LICENSE-DATA).
"""
from __future__ import annotations

import glob
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PORT = os.environ.get("KHASI_TEST_PGPORT", "55432")


def _pgbin() -> Path:
    if os.environ.get("PGBIN"):
        return Path(os.environ["PGBIN"])
    found = sorted(glob.glob("/usr/lib/postgresql/*/bin"),
                   key=lambda p: int(Path(p).parent.name) if Path(p).parent.name.isdigit() else 0)
    if found:
        return Path(found[-1])
    on_path = shutil.which("pg_ctl")
    if on_path:
        return Path(on_path).parent
    sys.exit("PostgreSQL server programs not found: install postgresql, or set PGBIN.")


def main() -> int:
    pgbin = _pgbin()
    tmp = Path(tempfile.mkdtemp(prefix="khasi-pg-test-"))
    data = tmp / "data"
    env = {**os.environ,
           "KHASI_SPELL_NO_DOTENV": "1",
           "DATABASE_URL": f"postgresql://khasi@127.0.0.1:{PORT}/khasi_test"}
    started = False
    try:
        subprocess.run([pgbin / "initdb", "-D", data, "-U", "khasi", "--auth=trust",
                        "-E", "UTF8", "--no-locale"], check=True, stdout=subprocess.DEVNULL)
        subprocess.run([pgbin / "pg_ctl", "-D", data, "-l", tmp / "server.log", "-w",
                        "-o", f"-p {PORT} -c listen_addresses=127.0.0.1 "
                              "-c unix_socket_directories=''", "start"],
                       check=True, stdout=subprocess.DEVNULL)
        started = True
        subprocess.run([pgbin / "createdb", "-h", "127.0.0.1", "-p", PORT, "-U", "khasi",
                        "khasi_test"], check=True)
        print(f"[pg-test] private server on 127.0.0.1:{PORT}, data in {tmp}", flush=True)
        for script in ("migrate_json_to_pg.py", "migrate_ngrams_to_pg.py"):
            subprocess.run([sys.executable, ROOT / "scripts" / script],
                           check=True, cwd=ROOT, env=env)
        return subprocess.run([sys.executable, "-m", "pytest", "-q", *sys.argv[1:]],
                              cwd=ROOT, env=env).returncode
    finally:
        if started:
            subprocess.run([pgbin / "pg_ctl", "-D", data, "-m", "fast", "stop"],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())

"""
Shared test configuration.

Most of the suite needs the lexicon, which is licensed separately
(LICENSE-DATA) and is not in the repository. On a fresh clone without it —
no `data/khasi_db.json` (or KHASI_DATA_DIR) and no DATABASE_URL — those tests
are skipped with a reason instead of erroring one by one with
FileNotFoundError. The modules listed in `NO_LEXICON_NEEDED` still run.
"""
import os

import pytest

# Hermetic runs: the service and the CLI read `.env`, and loading it inside
# the test process would switch every later test onto PostgreSQL. Export
# DATABASE_URL explicitly to run the suite against a database.
os.environ.setdefault("KHASI_SPELL_NO_DOTENV", "1")

# The service's rate limit counts requests per visitor, and every test client
# is the same visitor. Left on, a fast run of the API tests would start
# getting 429s. The limiter has its own tests (test_service_limits), and the
# route tests that need it switch it on for themselves.
for _name in ("KHASI_SPELL_RATE_CHECKS_PER_MIN", "KHASI_SPELL_RATE_LOOKUPS_PER_MIN",
              "KHASI_SPELL_RATE_LOOKUPS_PER_HOUR"):
    os.environ.setdefault(_name, "0")

# Test modules that run without the lexicon. test_alphabet's data checks
# skip themselves when it is absent.
NO_LEXICON_NEEDED = {"test_g2p", "test_api_senses", "test_tokens", "test_alphabet",
                     "test_service_limits", "test_page_messages"}


def _lexicon_available() -> bool:
    if (os.environ.get("DATABASE_URL") or "").strip():
        return True
    from khasi_engine import paths
    return paths.default_db_path().is_file()


def pytest_collection_modifyitems(config, items):
    if _lexicon_available():
        return
    skip = pytest.mark.skip(
        reason="lexicon not available: set DATABASE_URL, or provide "
               "data/khasi_db.json (see LICENSE-DATA)")
    for item in items:
        module = item.module.__name__.rsplit(".", 1)[-1] if item.module else ""
        if module not in NO_LEXICON_NEEDED:
            item.add_marker(skip)

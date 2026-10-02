"""
env.py — read a `.env` file into os.environ, with the standard library only.

Used by the HTTP service and the CLI, so both read the lexicon from the same
place. `export DATABASE_URL=...` lives and dies with one shell: run the
service from a second terminal and it silently loads the JSON lexicon
instead, which looks identical until you notice the start-up line.

Values already present in the environment always win, so an explicit
`DATABASE_URL=... uvicorn ...` still overrides the file. Must run BEFORE the
engine modules are imported: phonology, morphology and assimilation decide
at import time whether to read the JSON lexicon.
"""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

_ROOT = Path(__file__).resolve().parent.parent


def load_dotenv(path: Optional[Path] = None) -> Optional[str]:
    """Load *path* (default: `.env` beside pyproject.toml).

    Returns a note naming the keys loaded, or None when nothing was loaded.
    """
    # KHASI_SPELL_NO_DOTENV=1 disables the file, which the test suite sets
    # so a developer's .env cannot switch tests onto PostgreSQL mid-run.
    if (os.environ.get("KHASI_SPELL_NO_DOTENV") or "").strip() not in ("", "0"):
        return None
    p = Path(path) if path else _ROOT / ".env"
    if not p.is_file():
        return None
    loaded = []
    try:
        for line in p.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key.startswith("export "):
                key = key[len("export "):].strip()
            if key and key not in os.environ:
                os.environ[key] = value
                loaded.append(key)
    except OSError as e:
        print(f"[khasi-spell] could not read {p}: {e}")
        return None
    return f"{p.name}: {', '.join(loaded)}" if loaded else None

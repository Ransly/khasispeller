"""
paths.py — where the engine's data files live, decided in one place.

Every module used to build its own `Path(__file__).parent.parent / "data"`.
That works from a source checkout and nowhere else: a non-editable
`pip install .` puts the packages in site-packages without the `data/`
folder, and every optional resource (corpus frequencies, the n-gram model,
the gazetteer, the run-together table, the gloss spelling links) then
switched itself off without a word.

`KHASI_DATA_DIR` points an installed copy at its data. Without it the
project-root `data/` folder is used, exactly as before.

This module also parses the lexicon's small `phonology` and `morphology`
blocks ONCE for the modules that need them at import time. Each of
phonology.py, morphology.py and assimilation.py used to parse the whole
64 MB JSON file on its own, so a JSON-mode start-up read it four times
counting KhasiDB's own load.
"""
from __future__ import annotations

import json
import os
from functools import lru_cache
from pathlib import Path

_PROJECT_ROOT = Path(__file__).resolve().parent.parent


def data_dir() -> Path:
    """The directory holding the lexicon and the corpus-derived resources."""
    override = (os.environ.get("KHASI_DATA_DIR") or "").strip()
    if override:
        return Path(override).expanduser()
    return _PROJECT_ROOT / "data"


def data_file(name: str) -> Path:
    """Path of one file inside `data_dir()`."""
    return data_dir() / name


def default_db_path() -> Path:
    """The JSON lexicon used when no path is given and DATABASE_URL is unset."""
    return data_file("khasi_db.json")


@lru_cache(maxsize=1)
def lexicon_blocks() -> dict:
    """
    The lexicon's `phonology` and `morphology` blocks, parsed once.

    Returns an empty dict when DATABASE_URL is set: KhasiDB then loads both
    blocks from PostgreSQL and injects them into the modules, so reading the
    JSON here would only add a transient ~300 MB peak. Also empty when the
    file is absent, which is the normal state of a fresh clone — the file is
    licensed separately and is not in the repository.
    """
    if os.environ.get("DATABASE_URL"):
        return {}
    path = default_db_path()
    try:
        with open(path, encoding="utf-8") as fh:
            raw = json.load(fh)
    except FileNotFoundError:
        return {}
    except Exception as exc:                      # malformed file: say so once
        print(f"[khasi-nlp] could not read {path}: {exc}")
        return {}
    # Keep only the two small blocks; the lexicon itself is dropped here and
    # KhasiDB reads it when it is constructed.
    return {
        "phonology": raw.get("phonology") or {},
        "morphology": raw.get("morphology") or {},
    }

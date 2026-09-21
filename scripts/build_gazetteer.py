#!/usr/bin/env python3
"""
build_gazetteer.py — compile the proper-noun lists into data/gazetteer.json.

Source: the project's own NER pattern files (villages, institutions,
politicians, political parties, state capitals, states/UTs). These are the
author's data, not a third-party corpus, so nothing here carries an
external licence.

    python3 scripts/build_gazetteer.py
    python3 scripts/build_gazetteer.py --source "/path/to/Patterns Files"

What it does
------------
Entries are one per line and many are multi-word ("Nongbak Rengkil",
"National Institute of Fashion Technology"). The spellchecker works one
token at a time, so entries are split into tokens and only the
**capitalised** ones are kept — that drops the connectives ("of", "and")
which would otherwise enter the gazetteer as name tokens and disable
checking for those ordinary words.

Tokens shorter than MIN_LENGTH are dropped. Single letters and two-letter
fragments come mostly from initials ("A. Andrew Shullai") and collide
heavily with ordinary Khasi words, which are frequently one or two letters.

patterns.jsonl in the same directory is deliberately ignored: it is the
compiled spaCy pattern file, a different artefact with a different shape.
"""
from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

DEFAULT_SOURCE = Path("/home/rans/PhD-Works/NER/Patterns Files")
OUT = ROOT / "data" / "gazetteer.json"

FILES = ["villages.txt", "institution.txt", "politicians.txt",
         "political_parties.txt", "state_cap.txt", "states_ut.txt"]

# Below this, entries are initials and fragments that collide with ordinary
# Khasi words far more often than they identify a name.
MIN_LENGTH = 3

_TOKEN = re.compile(r"[A-Za-zÏïÑñ]+(?:['\-][A-Za-zÏïÑñ]+)*")


def main() -> int:
    ap = argparse.ArgumentParser(description="compile the proper-noun gazetteer")
    ap.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    ap.add_argument("--min-length", type=int, default=MIN_LENGTH)
    args = ap.parse_args()

    if not args.source.is_dir():
        print(f"error: no such directory: {args.source}", file=sys.stderr)
        return 2

    by_file: dict[str, int] = {}
    tokens: dict[str, str] = {}       # token -> the list it came from
    entries = 0

    for name in FILES:
        path = args.source / name
        if not path.is_file():
            print(f"  warning: missing {name}", file=sys.stderr)
            continue
        kind = name.replace(".txt", "")
        before = len(tokens)
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            entries += 1
            for m in _TOKEN.finditer(line):
                tok = m.group(0)
                if not tok[:1].isupper() or len(tok) < args.min_length:
                    continue
                tokens.setdefault(tok.lower(), kind)
        by_file[kind] = len(tokens) - before

    # Report collisions rather than resolving them. A gazetteer token is
    # only ever consulted for a word the gate has already rejected, so a
    # word that is also in the lexicon is accepted by the lexicon first and
    # never reaches this list — the overlap is inert, but worth seeing.
    collisions = []
    try:
        from khasi_spell import KhasiSpeller

        db = KhasiSpeller(eager=True).analyser.db
        collisions = sorted(t for t in tokens if db.is_known(t))
    except Exception as exc:                       # pragma: no cover
        print(f"  (collision check skipped: {exc})", file=sys.stderr)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({
        "meta": {
            "source": str(args.source),
            "files": FILES,
            "entries_read": entries,
            "min_length": args.min_length,
            "tokens": len(tokens),
            "lexicon_collisions": len(collisions),
            "note": "Capitalised tokens from the project's own NER pattern "
                    "files. Consulted only for words the gate has already "
                    "rejected, so lexicon collisions are inert.",
        },
        "tokens": dict(sorted(tokens.items())),
    }, ensure_ascii=False, indent=1), encoding="utf-8")

    print(f"  {entries} entries read from {len(by_file)} files")
    for k, v in by_file.items():
        print(f"    {k:<22}{v:>7} new tokens")
    print(f"  {len(tokens)} tokens -> {OUT}")
    print(f"  {len(collisions)} collide with a lexicon word (inert): "
          f"{', '.join(collisions[:12])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

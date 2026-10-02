#!/usr/bin/env python3
"""
build_english_list.py — the English words of the corpus, in data/english_in_corpus.json.

Why it exists
-------------
Maintainer ruling 2026-10-01: English words in Khasi text are to be
REJECTED — `party`, `deputy`, `hospital` are not Khasi words, however often
Khasi writers use them. They were accepted all the same. The corpus pool
admits a corpus type seen at least 20 times when it is spelled with Khasi
letters and passes phonotactics, and the vote then accepts it on corpus
frequency (2) + phonotactics (1). Nothing asked whether the type is English.
Measured before this list: 355 English corpus types (count >= 20, not in the
lexicon) were accepted, 0.77% of all corpus tokens — `hospital` 2,233,
`the` 2,002, `national`, `state`, `member`. A few more cleared the vote on a
spurious Khasi parse instead (`tablet`, `within`, `winter`).

The checker reads this list three ways (khasi_engine.spell_checker): a word
on it is never accepted unless the lexicon records it, it is never offered
as a suggestion, and corpus_pool does not admit it.

What is on it
-------------
Corpus types seen at least FLOOR times that are in an English dictionary,
contain a vowel and are not Khasi lexicon words. Only that intersection is stored, so the
dictionary itself is not bundled: the source is the SCOWL-derived
/usr/share/dict/american-english and british-english (Debian `wamerican`,
`wbritish`; permissive licence). British spellings matter here — the Khasi
press writes `centre` and `programme`.

What is held back
-----------------
A Khasi word or name can share a spelling with an English one, and
rejecting a real Khasi word is the worse error. Held back, with the reason
recorded in the file:

  * names — a word almost never written in lower case: at least
    NAME_SHARE of its corpus uses capitalised and at most NAME_MAX_LOWER in
    lower case (`Blah`, `Hoping`, `Smit`, `Witting`). Khasi surnames and
    places; capitalised mid-sentence they are skipped by
    khasi_spell.foreign anyway, and holding them back keeps a name that
    opens a sentence from being flagged. Share alone is not enough: English
    words inside names are mostly capitalised too — `district` 84%,
    `national` 95%, `congress` 98% — but each is written in lower case
    hundreds of times as the ordinary English word, so those stay listed;
  * Khasi derivations — a productive Khasi prefix (jing-, pyn-, nong-, ïa-,
    sngew-) on a lexicon word: `jingling` is jing- + ling;
  * KHASI below, by ruling or inspection — `longing` is the Khasi
    "household" (`ki longing ki ba ju ioh ia ka skhim`).

    python3 scripts/build_english_list.py --dry-run
    python3 scripts/build_english_list.py

Needs the lexicon, data/corpus_freq.json and the corpus at ../rupang_data
(for capitalisation).
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import json
import re
import sys
from collections import Counter
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

OUT = ROOT / "data" / "english_in_corpus.json"
CORPUS = ROOT.parent / "rupang_data"
DICTIONARIES = ["/usr/share/dict/american-english", "/usr/share/dict/british-english"]

# The candidate search only reaches corpus types; below this count a type
# is too rare to be accepted on corpus evidence, and the list would mostly
# collect accidental spellings.
FLOOR = 5

# A word is a name here when at least this share of its corpus uses are
# capitalised AND it is written in lower case at most NAME_MAX_LOWER times.
NAME_SHARE = 0.95
NAME_MAX_LOWER = 20

# Single letters are initials, and none can be accepted anyway.
MIN_LENGTH = 2

# A word with no vowel (`km`, `mm`, `sh`) is an abbreviation or a stray
# digraph. It fails the phonotactic gate anyway, and "invalid spelling" with
# its suggestions describes it better than "English word" with none.
VOWELS = set("aeiouy")

# Khasi words that share a spelling with an English word, by ruling or
# inspection. word -> evidence.
KHASI = {
    "longing": "Khasi 'household' (ki longing ki ba ju ioh ia ka skhim); "
               "886 lower-case uses, all in Khasi sentences",
}

_TOKEN = re.compile(r"[A-Za-zÏïÑñ]+(?:['\-][A-Za-zÏïÑñ]+)*")


def english_dictionary() -> set:
    words: set = set()
    for path in DICTIONARIES:
        with open(path, encoding="utf-8", errors="ignore") as fh:
            for line in fh:
                w = line.strip()
                # Lower-case entries only: capitalised ones are names.
                if w.isalpha() and w.islower():
                    words.add(w)
    return words


def capitalisation(words: set) -> dict:
    """word -> (lower-case uses, Title-case uses) across the corpus."""
    lower, title = Counter(), Counter()
    for path in sorted(glob.glob(str(CORPUS / "*.txt"))):
        with open(path, encoding="utf-8", errors="replace") as fh:
            for line in fh:
                for tok in _TOKEN.findall(line):
                    low = tok.lower()
                    if low not in words:
                        continue
                    if tok.islower():
                        lower[low] += 1
                    elif tok[0].isupper() and (len(tok) == 1 or tok[1:].islower()):
                        title[low] += 1
    return {w: (lower[w], title[w]) for w in words}


def main() -> int:
    ap = argparse.ArgumentParser(description="build data/english_in_corpus.json")
    ap.add_argument("--floor", type=int, default=FLOOR)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    with contextlib.redirect_stdout(sys.stderr):
        from khasi_engine.database import KhasiDB
        from khasi_engine import morphology
        from khasi_spell import corpus_pool
        db = KhasiDB()
    counts = corpus_pool._load_counts()
    english = english_dictionary()

    candidates = {w: c for w, c in counts.items()
                  if c >= args.floor and len(w) >= MIN_LENGTH
                  and VOWELS & set(w)
                  and w in english and not db.is_known(w)}
    case = capitalisation(set(candidates))
    prefixes = sorted((p for p, info in (morphology.PREFIXES or {}).items()
                       if info.get("productive")), key=len, reverse=True)

    words, held = {}, {}
    for w, c in sorted(candidates.items()):
        lo, ti = case.get(w, (0, 0))
        if w in KHASI:
            held[w] = f"Khasi word: {KHASI[w]}"
            continue
        if lo + ti and ti / (lo + ti) >= NAME_SHARE and lo <= NAME_MAX_LOWER:
            held[w] = f"name: {ti} of {lo + ti} corpus uses are capitalised"
            continue
        stem = next((w[len(p):] for p in prefixes
                     if w.startswith(p) and len(w) - len(p) >= 2
                     and db.is_known(w[len(p):])), None)
        if stem is not None:
            held[w] = f"Khasi derivation: {w[:len(w) - len(stem)]}- + {stem}"
            continue
        words[w] = c

    print(f"  English dictionary words : {len(english):,}")
    print(f"  corpus types >= {args.floor}, English, not Khasi lexicon: {len(candidates):,}")
    print(f"  on the list              : {len(words):,} "
          f"({sum(words.values()):,} corpus tokens)")
    print(f"  held back                : {len(held):,}")
    for w in sorted(held, key=lambda x: -candidates[x])[:20]:
        print(f"      {w:14} {candidates[w]:>6}  {held[w]}")
    if args.dry_run:
        print("\n  --dry-run: nothing written")
        return 0

    args.out.write_text(json.dumps({
        "meta": {
            "built": date.today().isoformat(),
            "ruling": "maintainer 2026-10-01: English words in Khasi text are rejected",
            "rule": (f"corpus types seen at least {args.floor} times that are in an "
                     "English dictionary, contain a vowel and are not Khasi "
                     "lexicon words"),
            "dictionaries": DICTIONARIES,
            "dictionary_licence": "SCOWL (Debian wamerican/wbritish), permissive",
            "floor": args.floor,
            "name_share": NAME_SHARE,
            "name_max_lower": NAME_MAX_LOWER,
            "words": len(words),
            "held_back": len(held),
            "builder": "scripts/build_english_list.py",
        },
        "words": dict(sorted(words.items())),
        "held_back": dict(sorted(held.items())),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n  wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

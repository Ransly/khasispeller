#!/usr/bin/env python3
"""
export_borrowing_candidates.py — build a review list of likely borrowings.

The checker's remaining noise on clean text is dominated by loanwords the
lexicon does not hold: `lockdown`, `plastik`, `result`, `drama`, `apps`.
They are not Khasi words, so the gate is right to reject them, and they are
not misspellings, so correcting them is wrong. The fix is a curated list of
accepted borrowings — the same shape as the proper-noun gazetteer, which
cost nothing and carries no external licence.

This script proposes candidates. It does not decide: every row lands in a
CSV with an empty `verdict` column for a human to fill in.

Why not just accept frequent corpus words automatically
-------------------------------------------------------
Because "written often" and "written correctly" are different claims, and
conflating them in a spellchecker is self-defeating. Measured on held-out
corpus text, accepting any token with 25+ occurrences would suppress 74% of
the remaining flags — but among them `jongphi`, a genuine run-together of
`jong phi`. It is frequent *because* it is a common error. The frozen
benchmarks score that rule at 1 error hidden in 558, which flatters it:
they contain synthetic single-edit typos, not the run-together and
out-of-lexicon errors that dominate real text.

So frequency is used here only to *rank candidates for review*, never to
accept them.

Selection
---------
A token is proposed when all of these hold:

  * it appears at least MIN_COUNT times in the corpus — rare strings are
    overwhelmingly typos, and there are 32,943 hapax legomena to prove it;
  * the checker currently rejects it, so it is really contributing noise;
  * it is lowercase — capitalised forms are already handled by the
    proper-noun rules in `khasi_spell.foreign`;
  * **no Khasi word sits within one edit of it.** A misspelling is nearly
    always one edit from its target, so requiring distance 2+ removes most
    typos from the candidate pool;
  * **it does not split into two Khasi words.** Run-togethers — `jongki`
    for `jong ki`, `kila` for `ki la` — are frequent enough to rank near the
    top by count, and `near_lexicon` cannot see them: a token made of two
    whole words is nowhere near one edit of any single entry. They are
    written to a separate file, because they are a real error class the
    checker currently handles badly, not borrowings.

Columns
-------
surface, count, near_lexicon, in_english, example, verdict, note

`example` is a real corpus sentence containing the token, which is what
makes a borrowing judgeable at a glance. `in_english` is advisory only —
computed when a system word list is present, blank otherwise.

    python3 scripts/export_borrowing_candidates.py
    python3 scripts/export_borrowing_candidates.py --min-count 10 --limit 500
"""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

MIN_COUNT = 5
OUT = ROOT / "data" / "borrowing_candidates_review.csv"
CORPUS = ROOT.parent / "rupang_data"

_TOKEN = re.compile(r"[A-Za-zÏïÑñ]+(?:['\-][A-Za-zÏïÑñ]+)*")
_ALPHABET = "abdeghijklmnoprstuwyïñ"


# A run-together needs both halves to be real words, and short halves make
# spurious splits: almost any string splits into *something* if two letters
# count. Khasi's clitics are short, though, so this cannot go above 2.
MIN_SPLIT_PART = 2

# A true run-together joins a *function* word to its neighbour: `jong ki`,
# `ki la`. Requiring one half to come from this closed class is what
# separates the errors from ordinary compounding — Khasi builds words like
# `jaitbynriew` (jait + bynriew) and `longkmie` (long + kmie) freely, and
# those are correct. Proclitics, prepositions and common particles only.
# A coincidental split has to be ruled out as well. If `jongki` really is
# `jong ki` run together, that phrase occurs in the corpus constantly; if
# `nepali` were `ne pali`, it would occur too — and it does not, because
# Nepali is a nationality. The separation is absolute on the sample:
#
#     jong ki  17312     ha iing  1192     ne pali   0
#     namar ba  7618     jong phi  633     ne so     0
#     jong ngi  2962     tang ba   267     lano sha  0
#
# so the threshold only has to be above zero; 5 leaves room without
# admitting anything that scored 0.
MIN_SPLIT_BIGRAM = 5

FUNCTION_WORDS = frozenset({
    "ka", "ki", "u", "i", "ia", "ïa", "jong", "la", "ban", "ha", "sha",
    "bad", "ne", "ba", "na", "kum", "da", "hangne", "kaba", "uba", "kiba",
})


def splits_into_words(word: str, db, bigrams: dict) -> str:
    """
    `jong` + `ki` for `jongki`, or "" when the token is not two words joined.

    This is the filter that keeps run-together errors out of the borrowing
    list. They are frequent — `jongki`, `jongphi`, `kila`, `kijong` — and
    they are invisible to `near_lexicon`, because a token made of two whole
    words is nowhere near one edit of any single entry.

    Two conditions beyond "both halves are words": one half must be a
    function word, which rules out ordinary compounding (`jaitbynriew`,
    `longkmie`); and the split phrase must actually occur in the corpus,
    which rules out coincidence (`nepali`, `neso`, `daplin`, `lanosha`).
    """
    w = word.lower()
    for i in range(MIN_SPLIT_PART, len(w) - MIN_SPLIT_PART + 1):
        left, right = w[:i], w[i:]
        if not (db.is_known(left) and db.is_known(right)):
            continue
        if left not in FUNCTION_WORDS and right not in FUNCTION_WORDS:
            continue
        if bigrams.get(f"{left}\t{right}", 0) < MIN_SPLIT_BIGRAM:
            continue                      # the phrase does not exist: coincidence
        return f"{left} {right}"
    return ""


def near_lexicon(word: str, db) -> bool:
    """True when some lexicon word is within one edit — i.e. it reads as a typo."""
    w = word.lower()
    if db.is_known(w):
        return True
    for i in range(len(w)):
        if db.is_known(w[:i] + w[i + 1:]):
            return True
        for c in _ALPHABET:
            if db.is_known(w[:i] + c + w[i + 1:]) or db.is_known(w[:i] + c + w[i:]):
                return True
    return any(db.is_known(w + c) for c in _ALPHABET)


def main() -> int:
    ap = argparse.ArgumentParser(description="propose borrowings for review")
    ap.add_argument("--min-count", type=int, default=MIN_COUNT)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--corpus", type=Path, default=CORPUS)
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()

    english: set[str] = set()
    for p in ("/usr/share/dict/american-english", "/usr/share/dict/british-english"):
        with contextlib.suppress(OSError):
            english |= {w.strip().lower()
                        for w in Path(p).read_text(encoding="utf-8",
                                                   errors="replace").splitlines()
                        if w.strip().isalpha()}

    # Deduplicate lines first: the corpus repeats each line about twice.
    seen: set[str] = set()
    counts: Counter = Counter()
    example: dict[str, str] = {}
    for path in sorted(args.corpus.rglob("*.txt")):
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            line = line.strip()
            if not line:
                continue
            h = hashlib.md5(line.encode()).hexdigest()
            if h in seen:
                continue
            seen.add(h)
            for m in _TOKEN.finditer(line):
                tok = m.group(0).lower()
                counts[tok] += 1
                if tok not in example and 40 < len(line) < 200:
                    example[tok] = line

    print(f"  {len(seen)} deduplicated lines, {len(counts)} types", file=sys.stderr)

    with contextlib.redirect_stdout(sys.stderr):
        from khasi_spell import KhasiSpeller, foreign

        sp = KhasiSpeller(eager=True)
        db = sp.analyser.db
        gaz = foreign.gazetteer()
        lm = sp._lm()
        bigrams = lm.bi if lm is not None else {}
        if not bigrams:
            print("  warning: no n-gram model; run scripts/build_ngrams.py "
                  "or run-together detection will be disabled", file=sys.stderr)

    pool = [(w, n) for w, n in counts.items()
            if n >= args.min_count and w.isalpha() and len(w) >= 3
            and w not in gaz and not db.is_known(w)]
    pool.sort(key=lambda x: -x[1])
    print(f"  {len(pool)} frequent unknown types before filtering", file=sys.stderr)

    rows = []
    runtogether = []
    with contextlib.redirect_stdout(sys.stderr):
        for w, n in pool:
            if near_lexicon(w, db):
                continue                      # reads as a typo, not a borrowing
            split = splits_into_words(w, db, bigrams)
            if split and w in english:
                # `banking` splits as `ban king` and both halves are Khasi
                # words, but it is an English word that happens to. A
                # borrowing, not a run-together.
                split = ""
            if split:
                runtogether.append({"surface": w, "count": n, "split_as": split,
                                    "phrase_count": bigrams.get(
                                        split.replace(" ", "\t"), 0),
                                    "example": example.get(w, "")[:160],
                                    "verdict": "",
                                    # `ia` is both the accusative preposition
                                    # and the productive reciprocal prefix, so
                                    # `iasyllok` may be a legitimate derivation
                                    # (ïasyllok) rather than `ia syllok` joined.
                                    # A Khasi speaker has to decide.
                                    "note": ("check: ia- may be the reciprocal "
                                             "prefix, not the preposition")
                                            if split.startswith("ia ") else ""})
                continue
            if sp.check(w).is_correct:
                continue                      # the gate already accepts it
            rows.append({
                "surface": w,
                "count": n,
                "near_lexicon": "",
                "in_english": "yes" if w in english else "",
                "example": example.get(w, "")[:160],
                "verdict": "",
                "note": "",
            })
            if args.limit and len(rows) >= args.limit:
                break

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]) if rows else
                                ["surface", "count", "near_lexicon", "in_english",
                                 "example", "verdict", "note"])
        writer.writeheader()
        writer.writerows(rows)

    rt_path = args.out.with_name("runtogether_candidates_review.csv")
    with rt_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["surface", "count", "split_as",
                                                "phrase_count", "example",
                                                "verdict", "note"])
        writer.writeheader()
        writer.writerows(runtogether)

    eng = sum(1 for r in rows if r["in_english"])
    print(f"  {len(runtogether)} run-together candidates -> {rt_path}")
    print(f"    top: {', '.join(r['surface'] + '=' + r['split_as'] for r in runtogether[:8])}")
    print(f"  {len(rows)} candidates -> {args.out}")
    print(f"    {eng} ({eng/max(len(rows),1):.0%}) are English-dictionary words")
    print(f"    top by frequency: "
          f"{', '.join(r['surface'] for r in rows[:14])}")
    print(f"\n  Review: fill the `verdict` column (accept / reject), then the "
          f"accepted rows\n  can be compiled the way the gazetteer is.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

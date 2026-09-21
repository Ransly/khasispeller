#!/usr/bin/env python3
"""
Build a corpus word-frequency list for the spellchecker's ranking prior.

The engine's built-in "frequency" is not one: database.py assigns values by
a word's structural role in the lexicon (root 20, surface 12/15, +5 if
morphologically transparent), yielding just 11 distinct values across
34,224 forms. It cannot tell a common word from a rare one. This replaces
it with observed counts.

    python3 scripts/build_corpus_freq.py ../rupang_data -o data/corpus_freq.json

Deduplicates at line level first: the source corpus repeats each line about
twice, which would double every count and distort nothing uniformly but
still misrepresents the distribution.
"""
from __future__ import annotations

import argparse
import collections
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

# Same token pattern the checker uses, so the frequency keys line up with
# what it will look up: letters plus ï/ñ, with word-internal ' and -.
TOKEN = re.compile(r"[A-Za-zÏïÑñ]+"
                   r"(?:['\-][A-Za-zÏïÑñ]+)*")


def iter_lines(paths: list[Path]):
    for p in sorted(paths):
        try:
            with open(p, encoding="utf-8", errors="replace") as fh:
                yield from fh
        except OSError as e:
            print(f"  skipped {p}: {e}", file=sys.stderr)


def build(paths: list[Path], dedupe: bool = True) -> tuple[collections.Counter, dict]:
    counts: collections.Counter = collections.Counter()
    seen: set[str] = set()
    raw_lines = kept_lines = raw_tokens = 0

    for line in iter_lines(paths):
        line = line.strip()
        if not line:
            continue
        raw_lines += 1
        if dedupe:
            h = hashlib.md5(line.encode("utf-8")).hexdigest()
            if h in seen:
                continue
            seen.add(h)
        kept_lines += 1
        words = [unicodedata.normalize("NFC", w).lower() for w in TOKEN.findall(line)]
        raw_tokens += len(words)
        counts.update(words)

    stats = {
        "lines_read": raw_lines,
        "lines_kept": kept_lines,
        "duplication_factor": round(raw_lines / kept_lines, 2) if kept_lines else 0,
        "tokens": raw_tokens,
        "types": len(counts),
        "hapax": sum(1 for c in counts.values() if c == 1),
    }
    return counts, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus", help="directory of .txt files, or a single file")
    ap.add_argument("-o", "--output", default="data/corpus_freq.json")
    ap.add_argument("--min-count", type=int, default=1,
                    help="drop types below this count (default 1 = keep all)")
    ap.add_argument("--no-dedupe", action="store_true")
    args = ap.parse_args()

    root = Path(args.corpus)
    paths = sorted(root.rglob("*.txt")) if root.is_dir() else [root]
    if not paths:
        print(f"no .txt files under {root}", file=sys.stderr)
        return 1
    print(f"reading {len(paths)} file(s) from {root} ...")

    counts, stats = build(paths, dedupe=not args.no_dedupe)
    if args.min_count > 1:
        counts = collections.Counter({w: c for w, c in counts.items()
                                      if c >= args.min_count})
        stats["types_after_min_count"] = len(counts)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as fh:
        json.dump({"meta": {**stats, "source": str(root),
                            "min_count": args.min_count,
                            "deduplicated": not args.no_dedupe},
                   "counts": dict(counts.most_common())},
                  fh, ensure_ascii=False)

    for k, v in stats.items():
        print(f"  {k:<22} {v:,}" if isinstance(v, int) else f"  {k:<22} {v}")
    print(f"  written                {out}  ({out.stat().st_size/1e6:.1f} MB)")
    print(f"  top 12                 {', '.join(w for w, _ in counts.most_common(12))}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
Build a word n-gram language model from the Khasi corpus.

This is the model the real-word error detector needs. Embeddings answer
"is this word vaguely topical", which turned out not to separate genuine
errors from correct words. An n-gram model answers "does this word belong
in this slot", which is the actual question — the Mays, Damerau and Mercer
(1991) formulation.

    python3 scripts/build_ngrams.py ../rupang_data -o data/ngrams.json.gz

Deduplicates lines first, exactly as build_corpus_freq.py does: the source
corpus repeats every line twice, which would double every count.

Pruning
-------
Singleton bigrams and trigrams are dropped by default. Most n-grams occur
once and carry no evidence, and keeping them triples the file for nothing.
Unigrams are kept in full — they are the backoff floor.
"""
from __future__ import annotations

import argparse
import collections
import gzip
import hashlib
import json
import re
import sys
import unicodedata
from pathlib import Path

TOKEN = re.compile(r"[A-Za-zÏïÑñ]+(?:['\-][A-Za-zÏïÑñ]+)*")
BOS, EOS = "<s>", "</s>"

# A line of running prose may hold several sentences.
SENT_SPLIT = re.compile(r"(?<=[.!?])\s+")


def sentences(paths: list[Path], dedupe: bool = True):
    seen: set[str] = set()
    for p in sorted(paths):
        try:
            fh = open(p, encoding="utf-8", errors="replace")
        except OSError as e:
            print(f"  skipped {p}: {e}", file=sys.stderr)
            continue
        with fh:
            for line in fh:
                line = line.strip()
                if not line:
                    continue
                if dedupe:
                    h = hashlib.md5(line.encode("utf-8")).hexdigest()
                    if h in seen:
                        continue
                    seen.add(h)
                for s in SENT_SPLIT.split(line):
                    words = [unicodedata.normalize("NFC", w).lower()
                             for w in TOKEN.findall(s)]
                    if words:
                        yield words


def build(paths: list[Path], dedupe: bool, min_bi: int, min_tri: int):
    uni: collections.Counter = collections.Counter()
    bi: collections.Counter = collections.Counter()
    tri: collections.Counter = collections.Counter()
    n_sent = n_tok = 0

    for words in sentences(paths, dedupe):
        n_sent += 1
        n_tok += len(words)
        padded = [BOS, BOS] + words + [EOS]
        uni.update(words)
        uni[BOS] += 2
        uni[EOS] += 1
        for i in range(1, len(padded)):
            bi[f"{padded[i-1]}\t{padded[i]}"] += 1
        for i in range(2, len(padded)):
            tri[f"{padded[i-2]}\t{padded[i-1]}\t{padded[i]}"] += 1

    raw = {"bigrams": len(bi), "trigrams": len(tri)}
    if min_bi > 1:
        bi = collections.Counter({k: v for k, v in bi.items() if v >= min_bi})
    if min_tri > 1:
        tri = collections.Counter({k: v for k, v in tri.items() if v >= min_tri})

    stats = {
        "sentences": n_sent,
        "tokens": n_tok,
        "unigram_types": len(uni),
        "bigram_types_raw": raw["bigrams"],
        "bigram_types_kept": len(bi),
        "trigram_types_raw": raw["trigrams"],
        "trigram_types_kept": len(tri),
        "min_bigram_count": min_bi,
        "min_trigram_count": min_tri,
    }
    return uni, bi, tri, stats


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("corpus")
    ap.add_argument("-o", "--output", default="data/ngrams.json.gz")
    ap.add_argument("--min-bigram", type=int, default=2)
    ap.add_argument("--min-trigram", type=int, default=2)
    ap.add_argument("--no-dedupe", action="store_true")
    args = ap.parse_args()

    root = Path(args.corpus)
    paths = sorted(root.rglob("*.txt")) if root.is_dir() else [root]
    if not paths:
        print(f"no .txt files under {root}", file=sys.stderr)
        return 1
    print(f"reading {len(paths)} file(s) ...")

    uni, bi, tri, stats = build(paths, not args.no_dedupe,
                                args.min_bigram, args.min_trigram)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    payload = {"meta": {**stats, "source": str(root)},
               "unigrams": dict(uni), "bigrams": dict(bi), "trigrams": dict(tri)}
    opener = gzip.open if out.suffix == ".gz" else open
    with opener(out, "wt", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False)

    for k, v in stats.items():
        print(f"  {k:<20} {v:,}" if isinstance(v, int) else f"  {k:<20} {v}")
    print(f"  written              {out}  ({out.stat().st_size/1e6:.1f} MB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

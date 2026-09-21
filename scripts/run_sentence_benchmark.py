#!/usr/bin/env python3
"""
run_sentence_benchmark.py — measure correction accuracy *in context*.

The word-level benchmark (tests/spell_benchmark.json, run by
run_benchmark.py) asks whether the right correction is found for an
isolated misspelling. This one asks a harder question: when the same
misspelling sits in a real sentence, does the checker put the right word
first?

The two can diverge. A misspelling with several equidistant candidates is
a coin toss in isolation but often unambiguous in context, which is what
the trigram re-ranker in `khasi_spell.context` exploits.

The benchmark is frozen. tests/sentence_benchmark.json holds 250 real
deduplicated corpus sentences, each with one single-edit error injected at
a recorded offset, generated once with a fixed seed. It is frozen for the
same reason the word benchmark is: an earlier probe in this project
regenerated its sample on every run, so consecutive measurements shared
almost no items and the numbers could not be compared. Do not regenerate
it to make a change look better.

Only items the checker actually flags were kept, so this isolates ranking
from detection. `error located` reports how many are still found at the
injected offset — if that falls, detection has regressed and the accuracy
figures below it are being computed over a different set.

    python3 scripts/run_sentence_benchmark.py
    python3 scripts/run_sentence_benchmark.py --json
    python3 scripts/run_sentence_benchmark.py --limit 50     # quick check
"""
from __future__ import annotations

import argparse
import contextlib
import json
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

BENCH = ROOT / "tests" / "sentence_benchmark.json"


def evaluate(sp, items, context: bool):
    """Top-1/top-5/MRR over *items*, with or without the context pass."""
    stats = {"top1": 0, "top5": 0, "located": 0, "rr": 0.0}
    by_kind = defaultdict(lambda: {"n": 0, "top1": 0})
    started = time.time()

    for it in items:
        by_kind[it["kind"]]["n"] += 1
        result = sp.check_text(it["sentence"], variants=False, context=context)
        corr = next((c for c in result.corrections if c.start == it["start"]), None)
        if corr is None:
            continue                     # not flagged: a detection miss
        stats["located"] += 1
        gold, ranked = it["gold"], list(corr.suggestions)
        if corr.suggestion == gold:
            stats["top1"] += 1
            by_kind[it["kind"]]["top1"] += 1
        if gold in ranked[:5]:
            stats["top5"] += 1
        if gold in ranked:
            stats["rr"] += 1.0 / (ranked.index(gold) + 1)

    n = len(items)
    return {
        "n": n,
        "located": stats["located"],
        "top1": stats["top1"] / n,
        "top5": stats["top5"] / n,
        "mrr": stats["rr"] / n,
        "ms_per_sentence": (time.time() - started) / max(n, 1) * 1000,
        "by_kind": {k: {"n": v["n"], "top1": v["top1"] / v["n"]}
                    for k, v in sorted(by_kind.items())},
    }


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("--json", action="store_true", help="machine-readable output")
    ap.add_argument("--limit", type=int, default=None, help="use only the first N items")
    ap.add_argument("--no-compare", action="store_true",
                    help="measure the default configuration only, skipping the A/B")
    args = ap.parse_args()

    if not BENCH.is_file():
        print(f"error: {BENCH} not found", file=sys.stderr)
        return 2
    data = json.loads(BENCH.read_text(encoding="utf-8"))
    # Items flagged valid=false have a gold that is not a Khasi word — they
    # were answerable only while English gloss text sat in the lexicon. Kept
    # in the file so the frozen set is never edited for convenience.
    quarantined = [i for i in data["items"] if i.get("valid") is False]
    valid = [i for i in data["items"] if i.get("valid") is not False]
    items = valid[:args.limit] if args.limit else valid

    # The engine prints progress on load; keep stdout clean for --json.
    with contextlib.redirect_stdout(sys.stderr):
        from khasi_spell import KhasiSpeller

        sp = KhasiSpeller(eager=True)
        on = evaluate(sp, items, context=True)
        off = None if args.no_compare else evaluate(sp, items, context=False)

    if args.json:
        print(json.dumps({"context": on, "isolated": off, "meta": data.get("meta")},
                         ensure_ascii=False, indent=2))
        return 0

    print(f"\nSentence benchmark — {on['n']} frozen items"
          + (f", {len(quarantined)} quarantined" if quarantined else ""))
    print(f"  error located at the injected offset: "
          f"{on['located']}/{on['n']} ({on['located']/on['n']:.1%})")
    print(f"\n  {'configuration':<22}{'top-1':>9}{'top-5':>9}{'MRR':>8}{'ms/sent':>10}")
    if off:
        print(f"  {'isolated (no context)':<22}{off['top1']:>9.1%}"
              f"{off['top5']:>9.1%}{off['mrr']:>8.3f}{off['ms_per_sentence']:>10.0f}")
    print(f"  {'+ context re-ranking':<22}{on['top1']:>9.1%}"
          f"{on['top5']:>9.1%}{on['mrr']:>8.3f}{on['ms_per_sentence']:>10.0f}")
    if off:
        print(f"  {'delta':<22}{on['top1']-off['top1']:>+9.1%}"
              f"{on['top5']-off['top5']:>+9.1%}{on['mrr']-off['mrr']:>+8.3f}")

    print(f"\n  by error kind (top-1):")
    print(f"    {'kind':<8}{'n':>5}" + (f"{'isolated':>11}" if off else "") + f"{'+context':>11}")
    for kind, v in on["by_kind"].items():
        row = f"    {kind:<8}{v['n']:>5}"
        if off:
            row += f"{off['by_kind'][kind]['top1']:>11.1%}"
        print(row + f"{v['top1']:>11.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

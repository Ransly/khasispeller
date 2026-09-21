#!/usr/bin/env python3
"""
Evaluate the spellchecker against the frozen benchmark.

    python3 scripts/run_benchmark.py
    python3 scripts/run_benchmark.py --json

Why frozen: the ad-hoc probe used during development regenerated its sample
from the lexicon on every run, so editing the lexicon silently changed the
test set. Two measurements a day apart shared only 2 of 76 items, which made
a 6-point "regression" look real when nothing had regressed. tests/
spell_benchmark.json fixes the items so results are comparable over time.

Reported metrics
----------------
  detection   share of misspellings the checker declines to accept
  top-1/top-5 share where the intended word is first / anywhere in five
  MRR         mean reciprocal rank of the intended word
  latency     mean ms per word
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

BENCH = Path(__file__).parent.parent / "tests" / "spell_benchmark.json"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", default=str(BENCH))
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--limit", type=int, default=0)
    args = ap.parse_args()

    sys.path.insert(0, str(Path(__file__).parent.parent))
    from khasi_spell import KhasiSpeller

    data = json.load(open(args.benchmark, encoding="utf-8"))
    all_items = data["items"]
    # Items flagged valid=false ask for a gold answer that cannot be a Khasi
    # word — English that reached the lexicon through gloss text leaking into
    # surface_form fields ('shafon', 'acid', 'funeral'). They are kept in the
    # file so the frozen set is never edited for convenience and the
    # contamination stays visible, but scoring against them measures nothing.
    quarantined = [i for i in all_items if i.get("valid") is False]
    items = [i for i in all_items if i.get("valid") is not False]
    items = items[: args.limit] if args.limit else items

    # The engine prints progress to stdout, which corrupts --json output.
    # Send anything it writes during construction to stderr instead.
    import contextlib
    with contextlib.redirect_stdout(sys.stderr):
        sp = KhasiSpeller(eager=True)
    by_kind: dict[str, list[int]] = collections.defaultdict(lambda: [0, 0, 0, 0])
    detected = top1 = top5 = 0
    rr = 0.0
    t0 = time.time()
    misses = []

    for it in items:
        typo, gold = it["typo"], it["gold"]
        r = sp.check(typo)
        kind = it["kind"][:3]
        row = by_kind[kind]
        row[0] += 1
        if not r.is_correct:
            detected += 1
            row[1] += 1
        sug = r.suggestions
        if sug and sug[0] == gold:
            top1 += 1
            row[2] += 1
        elif len(misses) < 12:
            misses.append((typo, gold, sug[:3]))
        if gold in sug:
            top5 += 1
            row[3] += 1
            rr += 1.0 / (sug.index(gold) + 1)

    n = len(items)
    elapsed = (time.time() - t0) / n * 1000
    result = {
        "n": n,
        "detection": round(detected / n, 4),
        "top1": round(top1 / n, 4),
        "top5": round(top5 / n, 4),
        "mrr": round(rr / n, 4),
        "ms_per_word": round(elapsed),
        "by_kind": {k: {"n": v[0], "detected": v[1], "top1": v[2], "top5": v[3]}
                    for k, v in sorted(by_kind.items())},
    }

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    print(f"\n  benchmark: {Path(args.benchmark).name}  ({n} items"
          + (f", {len(quarantined)} quarantined" if quarantined else "") + ")")
    print(f"  {'detection':<14}{result['detection']:>8.1%}")
    print(f"  {'top-1':<14}{result['top1']:>8.1%}")
    print(f"  {'top-5':<14}{result['top5']:>8.1%}")
    print(f"  {'MRR':<14}{result['mrr']:>8.3f}")
    print(f"  {'latency':<14}{result['ms_per_word']:>7} ms/word")
    print(f"\n  {'kind':<8}{'n':>5}{'detect':>9}{'top-1':>9}{'top-5':>9}")
    for k, v in result["by_kind"].items():
        print(f"  {k:<8}{v['n']:>5}{v['detected']/v['n']:>9.1%}"
              f"{v['top1']/v['n']:>9.1%}{v['top5']/v['n']:>9.1%}")
    print(f"\n  sample misses:")
    for typo, gold, sug in misses[:8]:
        print(f"    {typo:<14} want {gold:<14} got {sug}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

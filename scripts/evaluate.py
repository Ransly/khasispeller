#!/usr/bin/env python3
"""
evaluate.py — every evaluation metric the project reports, in one place.

The headline figures elsewhere (`run_benchmark.py`, `run_sentence_benchmark.py`)
answer "did it get the right answer". This answers the questions a reviewer
asks instead: how often does it cry wolf, how much of the loss is retrieval
against ranking, and is a claimed improvement distinguishable from chance.

Three families of metric, and they measure different things
-----------------------------------------------------------
**Detection, as a binary classifier.** A spellchecker decides, for every
token, "is this an error?". That is a classification problem and deserves a
confusion matrix. The negatives matter as much as the positives: a checker
that flags correct words is abandoned regardless of how well it corrects.

  precision   = TP / (TP + FP)   of the words it flagged, how many were errors
  recall      = TP / (TP + FN)   of the errors present, how many it flagged
  F1          = harmonic mean    the single number, when one is needed
  specificity = TN / (TN + FP)   of the correct words, how many it left alone

The negative class is *lexicon words the checker should accept*, sampled
from the corpus so the words are ones people actually write, and matched in
size to the positive class so the figures are not distorted by an imbalance
chosen for convenience.

**Correction quality, conditional on detection.** Accuracy at several k plus
mean reciprocal rank. Reporting top-1 alone hides where the loss is: if
top-1 is far below top-5, the right answer is being found and mis-ranked,
which is a different engineering problem from not finding it at all.

  MRR = mean over items of 1/rank, counting a miss as 0

**Significance.** A paired sign test over the items whose outcome changed.
With frozen benchmarks the same items are scored under both configurations,
so the pairing is exact and the test is the right one. Reporting a delta
without it invites the reasonable objection that the difference is noise.

False positives on running text
-------------------------------
Also reported, and not the same as the specificity above. That measures
behaviour on words the lexicon knows; this measures behaviour on real
sentences, where names, borrowings and unlisted-but-valid words all appear.
It is the number that predicts whether anyone keeps the tool switched on.

    python3 scripts/evaluate.py
    python3 scripts/evaluate.py --json
    python3 scripts/evaluate.py --no-running-text     # skip the slow part
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import hashlib
import json
import math
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

WORD_BENCH = ROOT / "tests" / "spell_benchmark.json"
SENT_BENCH = ROOT / "tests" / "sentence_benchmark.json"
CORPUS = ROOT.parent / "rupang_data"
TOKEN = re.compile(r"[A-Za-zÏïÑñ]+(?:['\-][A-Za-zÏïÑñ]+)*")

SEED = 1          # fixed so the negative sample is the same on every run


def valid_items(path: Path) -> list:
    """Benchmark items minus those flagged `valid: false`."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [i for i in data["items"] if i.get("valid") is not False]


def negative_sample(sp, n: int) -> list:
    """
    Correct words the checker ought to accept.

    Drawn from the corpus rather than from the lexicon at random, so they
    are words people actually write, and capped at `n` to match the positive
    class — an imbalanced split would make precision look like whatever the
    ratio was chosen to be.
    """
    db = sp.analyser.db
    seen, pool = set(), []
    for path in sorted(glob.glob(str(CORPUS / "*.txt"))):
        for line in open(path, encoding="utf-8", errors="replace"):
            for w in TOKEN.findall(line):
                wl = w.lower()
                if wl not in seen and len(wl) > 3 and db.is_known(wl):
                    seen.add(wl)
                    pool.append(wl)
        if len(pool) > 4000:
            break
    random.seed(SEED)
    return random.sample(pool, min(n, len(pool)))


def detection_metrics(sp, errors: list, corrects: list) -> dict:
    tp = sum(1 for i in errors if not sp.check(i["typo"]).is_correct)
    fn = len(errors) - tp
    fp = sum(1 for w in corrects if not sp.check(w).is_correct)
    tn = len(corrects) - fp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0
    return {
        "tp": tp, "fn": fn, "fp": fp, "tn": tn,
        "precision": prec, "recall": rec,
        "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
        "specificity": tn / (tn + fp) if tn + fp else 0.0,
        "accuracy": (tp + tn) / (tp + tn + fp + fn),
    }


def correction_metrics(sp, errors: list, depth: int = 10) -> dict:
    """Rank of the gold answer per item; 0 when it never appears."""
    ranks = []
    for item in errors:
        sug = sp.suggest(item["typo"], n=depth)
        ranks.append(sug.index(item["gold"]) + 1 if item["gold"] in sug else 0)
    n = len(ranks)
    out = {f"top_{k}": sum(1 for r in ranks if 0 < r <= k) / n
           for k in (1, 3, 5, 10)}
    out["mrr"] = sum(1 / r for r in ranks if r) / n
    out["never_retrieved"] = sum(1 for r in ranks if r == 0) / n
    return out


def running_text_fp(sp, n_sentences: int = 200) -> dict:
    """
    Flag rate on untouched corpus sentences — the practical false-positive
    rate, as distinct from specificity on known lexicon words.
    """
    seen, sents = set(), []
    for path in sorted(glob.glob(str(CORPUS / "*.txt"))):
        for line in open(path, encoding="utf-8", errors="replace"):
            for s in re.split(r"(?<=[.!?])\s+", line.strip()):
                if 8 <= len(TOKEN.findall(s)) <= 22:
                    h = hashlib.md5(s.encode()).hexdigest()
                    if h not in seen:
                        seen.add(h)
                        sents.append(s)
        if len(sents) > 25000:
            break
    random.seed(555)
    random.shuffle(sents)
    sample = sents[:n_sentences]
    flagged = tokens = with_flag = 0
    for s in sample:
        r = sp.check_text(s, variants=False, context=False)
        tokens += len(TOKEN.findall(s))
        flagged += len(r.corrections)
        with_flag += 1 if r.corrections else 0
    return {"sentences": len(sample), "tokens": tokens, "flagged": flagged,
            "token_rate": flagged / tokens,
            "sentence_rate": with_flag / len(sample)}


def sign_test(wins: int, losses: int) -> float:
    """
    Two-sided exact sign test. Ties carry no information and are excluded,
    which is what makes this the right test for paired benchmark outcomes.
    """
    n = wins + losses
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(min(wins, losses) + 1))
    return min(1.0, 2 * tail / (2 ** n))


def context_significance(sp) -> dict:
    """Is the contextual re-ranking gain distinguishable from chance?"""
    items = valid_items(SENT_BENCH)
    wins = losses = 0
    for it in items:
        off = sp.check_text(it["sentence"], variants=False, context=False)
        on = sp.check_text(it["sentence"], variants=False, context=True)
        a = next((c for c in off.corrections if c.start == it["start"]), None)
        b = next((c for c in on.corrections if c.start == it["start"]), None)
        ok_off = bool(a and a.suggestion == it["gold"])
        ok_on = bool(b and b.suggestion == it["gold"])
        if ok_on and not ok_off:
            wins += 1
        elif ok_off and not ok_on:
            losses += 1
    return {"fixed": wins, "broken": losses, "n": len(items),
            "p_value": sign_test(wins, losses)}


def main() -> int:
    ap = argparse.ArgumentParser(description="report every evaluation metric")
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--no-running-text", action="store_true")
    ap.add_argument("--no-significance", action="store_true")
    args = ap.parse_args()

    with contextlib.redirect_stdout(sys.stderr):
        from khasi_spell import KhasiSpeller

        sp = KhasiSpeller(eager=True)

    errors = valid_items(WORD_BENCH)
    corrects = negative_sample(sp, len(errors))
    result = {
        "detection": detection_metrics(sp, errors, corrects),
        "correction": correction_metrics(sp, errors),
    }
    if not args.no_running_text:
        result["running_text"] = running_text_fp(sp)
    if not args.no_significance:
        result["context_significance"] = context_significance(sp)

    if args.json:
        print(json.dumps(result, indent=2))
        return 0

    d, c = result["detection"], result["correction"]
    print(f"\n  DETECTION — {d['tp'] + d['fn']} errors vs "
          f"{d['tn'] + d['fp']} correct words")
    print(f"    {'':<14}{'flagged':>10}{'not flagged':>14}")
    print(f"    {'is an error':<14}{d['tp']:>10}{d['fn']:>14}")
    print(f"    {'is correct':<14}{d['fp']:>10}{d['tn']:>14}")
    for k in ("precision", "recall", "f1", "specificity", "accuracy"):
        print(f"    {k:<14}{d[k]:>10.3f}")

    print(f"\n  CORRECTION — rank of the right answer")
    for k in ("top_1", "top_3", "top_5", "top_10", "mrr", "never_retrieved"):
        print(f"    {k:<16}{c[k]:>8.3f}")

    if "running_text" in result:
        r = result["running_text"]
        print(f"\n  FALSE POSITIVES on untouched text")
        print(f"    {r['flagged']} of {r['tokens']} tokens "
              f"({r['token_rate']:.2%}) across {r['sentences']} sentences")
        print(f"    sentences carrying at least one flag: {r['sentence_rate']:.1%}")

    if "context_significance" in result:
        s = result["context_significance"]
        print(f"\n  SIGNIFICANCE of contextual re-ranking")
        print(f"    fixes {s['fixed']}, breaks {s['broken']} of {s['n']} items")
        print(f"    two-sided sign test p = {s['p_value']:.2e}")
    print()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
ablate.py — run one ablated configuration of the checker and report metrics.

Written for the research analysis of 6 September 2026. Every configuration is
produced by patching the shipped modules at import time; nothing in the
repository is edited, so a run leaves the tree exactly as it found it.

    python3 scripts/ablate.py full
    python3 scripts/ablate.py no_morphology
    for c in full lexicon_only no_morphology no_frequency no_phonotactic_vote \
             no_phoneme_distance no_char_distance no_rank_bonus no_offerable; do
        python3 scripts/ablate.py $c > /tmp/abl_$c.json
    done

Why the running-text flag rate is reported alongside precision/recall
---------------------------------------------------------------------
`evaluate.py`'s negative class is drawn subject to `db.is_known(w)`, so it is
defined by lexicon membership. A dictionary-only configuration accepts all of
it by construction and scores specificity 1.000 for free — which means
benchmark precision, specificity and F1 CANNOT show what morphological
acceptance contributes. Three ablations that remove a component score better
on those metrics than the shipped system for exactly this reason.

The flag rate on untouched corpus sentences is not compromised that way: its
sample is whatever real prose contains. It is the measurement that orders the
configurations the architecture predicts, and it is why this script computes
it for every configuration rather than only for the default one.

Output is one JSON object on stdout; engine chatter goes to stderr, so the
output is safe to redirect.
"""
from __future__ import annotations

import contextlib
import glob
import hashlib
import json
import random
import re
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CORPUS = ROOT.parent / "rupang_data"
TOKEN = re.compile(r"[A-Za-zÏïÑñ]+(?:['\-][A-Za-zÏïÑñ]+)*")

# Same seeds as evaluate.py, so the samples are the ones the headline figures
# were computed over and every configuration is paired at the item level.
NEGATIVE_SEED = 1
RUNNING_TEXT_SEED = 555
RUNNING_TEXT_SENTENCES = 200

CONFIGS = (
    "full", "lexicon_only", "no_morphology", "no_frequency",
    "no_phonotactic_vote", "no_phoneme_distance", "no_char_distance",
    "no_rank_bonus", "no_offerable",
)


def valid_items(path: Path) -> list:
    """Benchmark items minus those flagged `valid: false`."""
    data = json.loads(path.read_text(encoding="utf-8"))
    return [i for i in data["items"] if i.get("valid") is not False]


def apply_config(cfg: str, SC, Checker) -> str:
    """Patch the imported modules in place. Returns a human-readable note."""
    if cfg == "full":
        return "unmodified shipped configuration"
    if cfg == "lexicon_only":
        Checker._W_MORPH_AFFIXED = 0
        Checker._W_MORPH_IMPLAUSIBLE = 0
        Checker._W_MORPH_BARE = 0
        Checker._W_PHONOTACTIC = 0
        Checker._W_FREQUENT = 0
        return "accept iff the surface form is in the lexicon (S = 3L, threshold 3)"
    if cfg == "no_morphology":
        Checker._W_MORPH_AFFIXED = 0
        Checker._W_MORPH_IMPLAUSIBLE = 0
        Checker._W_MORPH_BARE = 0
        return "morphological evidence removed from the vote"
    if cfg == "no_frequency":
        Checker._W_FREQUENT = 0
        return "corpus-frequency evidence removed from the vote"
    if cfg == "no_phonotactic_vote":
        Checker._W_PHONOTACTIC = 0
        return "phonotactic evidence removed from the vote"
    if cfg == "no_phoneme_distance":
        SC._levenshtein = lambda a, b: SC._damerau_chars(a, b)
        return "character Damerau-Levenshtein only; phoneme distance removed"
    if cfg == "no_char_distance":
        SC._USE_CHAR_DISTANCE = False
        return "phoneme distance only; character Damerau blend removed"
    if cfg == "no_rank_bonus":
        SC._MORPHO_VALID_BONUS = 0
        SC._PREFIX_BONUS_WEIGHT = 0
        return "ranking tie-breakers removed (morphology and shared-prefix bonuses)"
    if cfg == "no_offerable":
        Checker._offerable = lambda self, cand, word="": True
        return "candidate admissibility filter disabled"
    raise SystemExit(f"unknown configuration {cfg!r}; choose from {', '.join(CONFIGS)}")


def negative_sample(db, n: int) -> list:
    """Correct words the checker ought to accept — evaluate.py's sample."""
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
    random.seed(NEGATIVE_SEED)
    return random.sample(pool, min(n, len(pool)))


def running_text_sample(n: int) -> list:
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
    random.seed(RUNNING_TEXT_SEED)
    random.shuffle(sents)
    return sents[:n]


def main() -> int:
    if len(sys.argv) != 2:
        print(f"usage: {sys.argv[0]} <{'|'.join(CONFIGS)}>", file=sys.stderr)
        return 2
    cfg = sys.argv[1]

    import khasi_engine.spell_checker as SC
    Checker = next((o for o in vars(SC).values()
                    if isinstance(o, type) and hasattr(o, "_W_LEXICON")), None)
    if Checker is None:
        print("could not locate the checker class", file=sys.stderr)
        return 1
    note = apply_config(cfg, SC, Checker)

    with contextlib.redirect_stdout(sys.stderr):
        from khasi_spell import KhasiSpeller
        t0 = time.time()
        sp = KhasiSpeller(eager=True)
        startup = time.time() - t0

    errors = valid_items(ROOT / "tests" / "spell_benchmark.json")
    corrects = negative_sample(sp.analyser.db, len(errors))

    flag_err = [not sp.check(i["typo"]).is_correct for i in errors]
    flag_cor = [not sp.check(w).is_correct for w in corrects]
    tp, fp = sum(flag_err), sum(flag_cor)
    fn, tn = len(errors) - tp, len(corrects) - fp
    prec = tp / (tp + fp) if tp + fp else 0.0
    rec = tp / (tp + fn) if tp + fn else 0.0

    # Rank of the gold answer per item; 0 when it never appears.
    ranks = []
    for item in errors:
        sug = sp.suggest(item["typo"], n=10)
        ranks.append(sug.index(item["gold"]) + 1 if item["gold"] in sug else 0)
    n = len(ranks)

    # Conditional on detection: divides only by items this configuration
    # flagged, which separates ranking quality from acceptance behaviour.
    # Without it, a configuration that flags everything looks like a better
    # ranker purely because it emits more suggestion lists.
    detected = [r for r, f in zip(ranks, flag_err) if f]
    nd = len(detected) or 1

    sample = running_text_sample(RUNNING_TEXT_SENTENCES)
    flagged = tokens = with_flag = 0
    t0 = time.time()
    for s in sample:
        result = sp.check_text(s, variants=False, context=False)
        tokens += len(TOKEN.findall(s))
        flagged += len(result.corrections)
        with_flag += 1 if result.corrections else 0
    ms_sentence = (time.time() - t0) / max(len(sample), 1) * 1000

    print(json.dumps({
        "config": cfg,
        "note": note,
        "startup_s": round(startup, 1),
        "detection": {
            "tp": tp, "fn": fn, "fp": fp, "tn": tn,
            "precision": prec, "recall": rec,
            "f1": 2 * prec * rec / (prec + rec) if prec + rec else 0.0,
            "specificity": tn / (tn + fp) if tn + fp else 0.0,
        },
        "correction": {
            **{f"top_{k}": sum(1 for r in ranks if 0 < r <= k) / n for k in (1, 3, 5, 10)},
            "mrr": sum(1 / r for r in ranks if r) / n,
            "never_retrieved": sum(1 for r in ranks if r == 0) / n,
        },
        "correction_conditional": {
            "n_flagged": len(detected),
            "top_1": sum(1 for r in detected if r == 1) / nd,
            "top_5": sum(1 for r in detected if 0 < r <= 5) / nd,
            "mrr": sum(1 / r for r in detected if r) / nd,
        },
        "running_text": {
            "sentences": len(sample), "tokens": tokens, "flagged": flagged,
            "token_rate": flagged / tokens if tokens else 0.0,
            "sentence_rate": with_flag / len(sample) if sample else 0.0,
            "ms_per_sentence": ms_sentence,
        },
        # Per-item vectors, so a paired sign test against another run needs no
        # re-measurement — the pairing is exact because the items are frozen.
        "ranks": ranks,
        "flag_err": [int(b) for b in flag_err],
    }, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

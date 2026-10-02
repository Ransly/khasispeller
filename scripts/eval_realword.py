#!/usr/bin/env python3
"""
eval_realword.py — precision and recall of real-word error detection.

The figures in khasi_spell/realword.py and the README were measured by a
script that was never committed, so they could not be reproduced or
updated. This one is seeded and self-contained:

  clean     untouched corpus sentences — every flag is a false positive
  swap      one accepted word replaced by a lexicon neighbour one edit away
            (from the detector's own confusion set, corpus-attested)
  clitic    one `ka` or `ki` swapped for the other, reported per direction,
            because a frequency rule used to make `ka` -> `ki` undetectable

Flags are computed once per sentence at the loosest margin and filtered for
each threshold, which gives the same answer as re-running at each margin:
a token is flagged when its best alternative gains at least the margin.

    python3 scripts/eval_realword.py                # table
    python3 scripts/eval_realword.py --json

Needs the corpus at ../rupang_data (as evaluate.py does) and the n-gram
model.
"""
from __future__ import annotations

import argparse
import contextlib
import glob
import hashlib
import json
import random
import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CORPUS = ROOT.parent / "rupang_data"
TOKEN = re.compile(r"[A-Za-zÏïÑñ]+(?:['\-][A-Za-zÏïÑñ]+)*")
MARGINS = (3.0, 5.0, 6.0, 7.0, 8.0, 10.0)
SEED = 20260930


_FOLD = str.maketrans({"ï": "i", "ñ": "n", "á": "a", "é": "e", "í": "i",
                       "ó": "o", "ú": "u", "ý": "y", "\u2019": "'"})


def same_word(a: str, b: str) -> bool:
    """Equal in the corpus's spelling. The detector offers the lexicon's
    standard spelling (`ïaid`), the corpus sentence carries the plain one
    (`iaid`); that is the right suggestion, not a different word."""
    f = lambda w: w.lower().translate(_FOLD).replace("'", "")
    return f(a) == f(b)


def sentences(limit: int = 30000) -> list:
    """Deduplicated corpus sentences of 8-22 tokens, in file order."""
    seen, out = set(), []
    for path in sorted(glob.glob(str(CORPUS / "*.txt"))):
        for line in open(path, encoding="utf-8", errors="replace"):
            for s in re.split(r"(?<=[.!?])\s+", line.strip()):
                if 8 <= len(TOKEN.findall(s)) <= 22:
                    h = hashlib.md5(s.encode()).hexdigest()
                    if h not in seen:
                        seen.add(h)
                        out.append(s)
        if len(out) > limit:
            break
    return out


def flags_at(sp, text: str, floor: float) -> list:
    return sp.check_realword(text, min_margin=floor)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true")
    ap.add_argument("--clean", type=int, default=150)
    ap.add_argument("--swap", type=int, default=80)
    ap.add_argument("--clitic", type=int, default=80)
    args = ap.parse_args()

    with contextlib.redirect_stdout(sys.stderr):
        from khasi_spell import KhasiSpeller
        sp = KhasiSpeller(eager=True)
        sp.check_realword("ka la wan")                 # build the detector
    det = sp._realword
    db = sp.analyser.db
    freq = det._freq
    floor = min(MARGINS)

    pool = sentences()
    rnd = random.Random(SEED)
    rnd.shuffle(pool)
    clean = pool[:args.clean]
    rest = pool[args.clean:]

    # --- false positives on untouched text --------------------------------
    clean_flags = [flags_at(sp, s, floor) for s in clean]
    tokens = sum(len(TOKEN.findall(s)) for s in clean)

    # --- injected word swaps ---------------------------------------------
    swaps = []
    for s in rest:
        if len(swaps) >= args.swap:
            break
        cands = []
        for m in TOKEN.finditer(s):
            w = m.group(0)
            wl = w.lower()
            if len(wl) < 3 or not db.is_known(wl) or w[:1].isupper():
                continue
            alts = [a for a in det.confusion_set(wl) if freq.get(a, 0) > 0]
            if alts:
                cands.append((m, alts))
        if not cands:
            continue
        m, alts = rnd.choice(cands)
        alt = rnd.choice(alts)
        swapped = s[:m.start()] + alt + s[m.end():]
        swaps.append((swapped, m.start(), m.start() + len(alt), m.group(0).lower()))
    swap_flags = [flags_at(sp, t, floor) for t, *_ in swaps]

    # --- clitic swaps, per direction -------------------------------------
    clitics = {"ka->ki": [], "ki->ka": []}
    per_dir = max(1, args.clitic // 2)
    for s in rest[len(swaps):]:
        for m in re.finditer(r"\b(ka|ki)\b", s):
            direction = f"{m.group(1)}->{'ki' if m.group(1) == 'ka' else 'ka'}"
            if len(clitics[direction]) >= per_dir:
                continue
            new = "ki" if m.group(1) == "ka" else "ka"
            clitics[direction].append((s[:m.start()] + new + s[m.end():],
                                       m.start(), m.group(1)))
            break
        if all(len(v) >= per_dir for v in clitics.values()):
            break
    clitic_flags = {d: [flags_at(sp, t, floor) for t, *_ in items]
                    for d, items in clitics.items()}

    rows = []
    for mg in MARGINS:
        fp_tokens = sum(sum(1 for f in fl if f.margin >= mg) for fl in clean_flags)
        fp_sents = sum(1 for fl in clean_flags if any(f.margin >= mg for f in fl))
        hit = total_flags = correct = 0
        for (t, start, end, orig), fl in zip(swaps, swap_flags):
            kept = [f for f in fl if f.margin >= mg]
            total_flags += len(kept)
            at = [f for f in kept if f.start == start]
            if at:
                hit += 1
                correct += 1 if same_word(at[0].suggestion, orig) else 0
        wrong_elsewhere = total_flags - hit
        row = {
            "margin": mg,
            "fp_token_rate": round(fp_tokens / max(tokens, 1), 4),
            "fp_sentence_rate": round(fp_sents / max(len(clean), 1), 4),
            "swap_recall": round(hit / max(len(swaps), 1), 4),
            "swap_correct_suggestion": round(correct / max(len(swaps), 1), 4),
            "swap_precision": round(hit / total_flags, 4) if total_flags else None,
            "swap_flags_elsewhere": wrong_elsewhere,
        }
        for d, items in clitics.items():
            got = sum(1 for (t, start, orig), fl in zip(items, clitic_flags[d])
                      if any(f.start == start and f.margin >= mg for f in fl))
            row[f"clitic_recall_{d}"] = round(got / max(len(items), 1), 4)
        rows.append(row)

    out = {"seed": SEED, "clean_sentences": len(clean), "clean_tokens": tokens,
           "swap_sentences": len(swaps),
           "clitic_sentences": {d: len(v) for d, v in clitics.items()},
           "default_margin": det.min_margin, "rows": rows}
    if args.json:
        print(json.dumps(out, indent=2))
        return 0
    print(f"\n  clean {len(clean)} sentences ({tokens} tokens), swaps {len(swaps)}, "
          f"clitic swaps {out['clitic_sentences']}; default margin {det.min_margin}")
    print(f"\n  {'margin':>6} {'FP tok':>7} {'FP sent':>8} {'recall':>7} "
          f"{'right sugg':>10} {'precision':>9} {'ka->ki':>7} {'ki->ka':>7}")
    for r in rows:
        prec = f"{r['swap_precision']:.0%}" if r["swap_precision"] is not None else "—"
        print(f"  {r['margin']:>6.1f} {r['fp_token_rate']:>7.2%} {r['fp_sentence_rate']:>8.1%} "
              f"{r['swap_recall']:>7.1%} {r['swap_correct_suggestion']:>10.1%} {prec:>9} "
              f"{r['clitic_recall_ka->ki']:>7.1%} {r['clitic_recall_ki->ka']:>7.1%}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

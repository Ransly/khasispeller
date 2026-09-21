"""
context.py — re-rank correction candidates using the surrounding sentence.

The engine corrects each token in isolation. It compares the misspelling
against the lexicon, weights the edits phonemically, and returns the
closest forms. That is the right primary signal, but it is blind to a fact
every reader uses: the neighbouring words say a great deal about which
candidate belongs in the slot.

Deletions show the cost most clearly. Dropping a letter destroys more
string evidence than any other single edit, and several lexicon entries end
up equidistant from the wreckage — string distance alone cannot separate
them. Measured on the frozen sentence benchmark, deletions were the weakest
class at 75.6% top-1; with context they reach 90.2%.

How it works
------------
For each correction, the trigram model scores every candidate as an
occupant of the slot it would fill, given up to two words either side
(`NgramLM.slot_score`). That score is blended with the engine's own
ordering rather than replacing it:

    combined = slot_score(candidate) + ALPHA * (-original_rank)

ALPHA is the exchange rate between the two kinds of evidence: how much
log-probability the model must find to move a candidate up one place. At 0
the string evidence is discarded entirely and accuracy drops back to
roughly the baseline — the model is a useful tiebreaker, not a replacement.
Very large values freeze the original order. The optimum is broad and flat
between 0.5 and 3.0, which is what makes the setting trustworthy: it is a
plateau, not a spike fitted to one sample.

Why blending, not overriding
----------------------------
The corpus is 6.67M tokens — large enough to rank candidates that appear in
it, far too small to have seen every valid trigram. On unseen context every
candidate floors to the same score, the LM term becomes constant, and the
blend falls back to the engine's ordering automatically. No special-casing
is needed for the sparse case, which is the common case.

A known limitation
------------------
The neighbours fed to the model are the words as typed, not as corrected.
In a sentence carrying two errors close together, each is ranked against
the other's misspelling — which the model has never seen, so it floors and
contributes nothing. The pass then degrades to the engine's own ordering
rather than to a wrong answer, so this costs opportunity, not accuracy.
Iterating to a fixed point, or decoding the sentence jointly, would
recover it; neither is justified by a 250-item benchmark on which errors
are injected one per sentence.

Measurement
-----------
250 frozen corpus sentences, one injected single-edit error each, split in
half: ALPHA chosen on one half, reported on the other.

    held-out top-1   86.4% -> 89.6%
    full set         fixes 20, breaks 7   (sign test p = 0.019)

Insertions are the one class that does not benefit (97.1% -> 95.7%): string
evidence there is already near-perfect, so context can only add noise. The
net across all four classes is positive, so no per-class gating is applied
— that would be four more parameters fitted to 250 items.
"""
from __future__ import annotations

import re
from typing import Any, Optional, Sequence

# Exchange rate between context and string evidence, in log units per rank
# position. Chosen on a held-out half of the sentence benchmark; the
# 0.5-3.0 plateau all scores within 1 point of it.
ALPHA = 1.0

# Words either side fed to the trigram model. `slot_score` scores the three
# trigrams containing the slot, so it cannot use more than two.
CONTEXT_WINDOW = 2

# Matches the tokenisation used elsewhere in the package: letters plus the
# two Khasi diacritics, with internal apostrophes and hyphens kept.
_TOKEN = re.compile(r"[A-Za-zÏïÑñ]+(?:['\-][A-Za-zÏïÑñ]+)*")


def _score(lm, cand, left, right):
    """Context score for one candidate, which may be more than one word.

    A split suggestion like `jong ngi` is a single candidate string holding
    two tokens. Passed to slot_score() whole it is looked up as if it were
    one word, every n-gram lookup misses, and it scores at the floor — so
    the re-ranker moved the one correct answer to the bottom of the list.

    Each part is scored in the slot it would actually occupy, with the other
    parts as its immediate context, and the MEAN is returned so a two-word
    candidate competes on the same scale as a one-word one rather than
    carrying twice the (negative) log score.
    """
    parts = cand.split()
    if len(parts) < 2:
        return lm.slot_score(cand, left, right)
    total = 0.0
    for i, part in enumerate(parts):
        total += lm.slot_score(part, list(left) + parts[:i],
                               parts[i + 1:] + list(right))
    return total / len(parts)


def rerank(
    text: str,
    corrections: Sequence[Any],
    lm: Any,
    alpha: float = ALPHA,
) -> int:
    """
    Re-order each correction's suggestions using its sentence context.

    Mutates the `suggestion`, `suggestions`, `distances` and `glosses`
    fields of *corrections* in place and returns the number whose top
    choice changed. A correction with fewer than two candidates is left
    alone — there is nothing to re-rank.

    The three parallel lists are permuted together. They are documented as
    being in the same order as `suggestions`, and a reader decides on the
    gloss, so leaving them behind does not merely mislabel the rows — it
    attaches one real word's meaning to a different real word.

    *lm* is a loaded `NgramLM`. Callers that may not have one should catch
    FileNotFoundError around the load and skip this pass; a missing model
    is a normal state, not an error.
    """
    if not corrections or lm is None:
        return 0

    tokens = [(m.group(0).lower(), m.start()) for m in _TOKEN.finditer(text)]
    by_offset = {start: i for i, (_, start) in enumerate(tokens)}
    changed = 0

    for corr in corrections:
        cands = list(getattr(corr, "suggestions", None) or [])
        if len(cands) < 2:
            continue

        idx = by_offset.get(corr.start)
        if idx is None:
            # Hyphenated parts are corrected at an offset inside a token,
            # so they will not be found. Fall back to the nearest token
            # starting at or before the correction.
            idx = next((i for i in range(len(tokens) - 1, -1, -1)
                        if tokens[i][1] <= corr.start), None)
            if idx is None:
                continue

        left = [w for w, _ in tokens[max(0, idx - CONTEXT_WINDOW):idx]]
        right = [w for w, _ in tokens[idx + 1:idx + 1 + CONTEXT_WINDOW]]

        scored = sorted(
            range(len(cands)),
            key=lambda i: -(_score(lm, cands[i], left, right) + alpha * (-i)),
        )
        reordered = [cands[i] for i in scored]
        if reordered[0] != cands[0]:
            changed += 1
        corr.suggestions = reordered
        corr.suggestion = reordered[0]
        # Carry the parallel lists through the same permutation. A list that
        # is shorter than `cands` keeps its own length rather than being
        # padded, so a caller that supplied none still gets none.
        for attr in ("distances", "glosses"):
            old = getattr(corr, attr, None)
            if not old:
                continue
            setattr(corr, attr, [old[i] for i in scored if i < len(old)])

    return changed

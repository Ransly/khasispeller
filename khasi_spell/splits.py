"""
splits.py — offer the two-word form for a run-together word.

The gap this closes
───────────────────
`jongki` is `jong ki` ("of them") written solid. The checker was right to
flag it — the corpus writes it spaced 17,312 times against 2,642 solid — but
what it OFFERED was `jongka`, `jingngi`, `nongngi`: the nearest single words
in the lexicon. Edit distance cannot propose a space, so the one correction
that is actually right was unreachable, and the whole Khasi possessive
paradigm (`jong` + pronoun, 9,418 corpus occurrences) was answered with
nonsense.

Worse, two of those answers were each other: `jongka` was offered for
`jongki` and `jongki` for `jongka`, cycling the writer between two forms
that are both wrong in the same way.

Evidence
────────
`data/runtogether_candidates_review.csv` already holds the candidates, with
the solid count and the spaced count side by side. A row is read as a
run-together when the spaced form is at least `DEFAULT_RATIO` times commoner.

That threshold is not a linguistic claim, and the distribution shows why it
cannot be: `tangba` (767 solid / 267 spaced) and `jongno` (221 / 36) are
solid words that happen to divide into two known ones, while `iakane`
(147 / 10,691) plainly is not. The ratio separates those two populations,
and everything between 0.96 and 1.65 is genuinely ambiguous.

So the CSV's `verdict` column WINS wherever it is filled in. The ratio is
the default for the rows nobody has ruled on yet, not a replacement for the
ruling. `jongphi` sits at 1.29 and is excluded by default despite the rest
of its paradigm qualifying — mark it `split` in that column to include it.

    verdict = split | yes | runtogether   -> always offer the split
    verdict = keep  | no  | solid | word  -> never offer it
    verdict = (blank)                     -> decide on the ratio
"""
from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Optional

DEFAULT_PATH = (Path(__file__).parent.parent / "data"
                / "runtogether_candidates_review.csv")

# Spaced-to-solid ratio at which a row is read as a run-together. Measured
# against the 53 rows in the file: the solid words sit at 0.16-0.98 and the
# clear run-togethers from 2.24 up.
DEFAULT_RATIO = 2.0

_SPLIT_VERDICTS = {"split", "yes", "y", "runtogether", "run-together", "1", "true"}
_KEEP_VERDICTS = {"keep", "no", "n", "solid", "word", "0", "false"}


def load(path: Optional[Path] = None, ratio: float = DEFAULT_RATIO) -> dict:
    """Return {solid form: spaced form} for every qualifying row.

    Missing file is not an error — the feature simply does not fire, which
    is the behaviour a deployment without the review data should have.
    """
    p = Path(path) if path else DEFAULT_PATH
    if not p.is_file():
        return {}
    out: dict = {}
    try:
        with p.open(encoding="utf-8", newline="") as fh:
            for row in csv.DictReader(fh):
                surface = (row.get("surface") or "").strip().lower()
                split_as = (row.get("split_as") or "").strip().lower()
                if not surface or " " not in split_as:
                    continue
                verdict = (row.get("verdict") or "").strip().lower()
                if verdict in _KEEP_VERDICTS:
                    continue
                if verdict not in _SPLIT_VERDICTS:
                    # Undecided: fall back to the corpus evidence.
                    try:
                        solid = int(row.get("count") or 0)
                        spaced = int(row.get("phrase_count") or 0)
                    except ValueError:
                        continue
                    if not solid or spaced < solid * ratio:
                        continue
                out[surface] = split_as
    except Exception:
        return {}
    return out


def summary(table: dict, path: Optional[Path] = None) -> dict:
    p = Path(path) if path else DEFAULT_PATH
    return {"source": str(p), "available": p.is_file(),
            "ratio": DEFAULT_RATIO, "entries": len(table)}

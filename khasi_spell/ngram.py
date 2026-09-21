"""
ngram.py — word n-gram language model over the Khasi corpus.

Built for real-word error detection, where the question is "does this word
belong in this slot" rather than "is this word roughly on topic". The
embedding-based attempt at the same job failed precisely because cosine
similarity to a context centroid answers the second question.

    python3 scripts/build_ngrams.py ../rupang_data -o data/ngrams.json.gz

    from khasi_spell.ngram import NgramLM
    lm = NgramLM()
    lm.slot_score("briew", ["ka", "sorkar"], ["ka", "la"])

Smoothing: Stupid Backoff (Brants et al., 2007)
----------------------------------------------
    S(w | u, v) = count(u,v,w) / count(u,v)          when the trigram is seen
                = 0.4 * S(w | v)                     otherwise
    S(w | v)    = count(v,w) / count(v)              when the bigram is seen
                = 0.4 * count(w) / N                 otherwise

Chosen over Kneser-Ney deliberately. Stupid Backoff returns a relative
score rather than a normalised probability, which is all that is needed
when comparing candidates competing for the same slot, and it stays well
behaved when the counts are sparse — as they are here, since bigrams and
trigrams seen only once are pruned at build time.
"""
from __future__ import annotations

import gzip
import json
import math
from pathlib import Path
from typing import Optional, Sequence

DEFAULT_PATH = Path(__file__).parent.parent / "data" / "ngrams.json.gz"

BOS, EOS = "<s>", "</s>"
BACKOFF = 0.4          # Brants et al.'s constant; performance is flat near it
FLOOR = 1e-10          # keeps log() finite for words the corpus never saw


class NgramLM:
    """Trigram language model with Stupid Backoff."""

    def __init__(self, path: Optional[str | Path] = None, lazy: bool = True):
        self._path = Path(path) if path else DEFAULT_PATH
        self.uni: dict[str, int] = {}
        self.bi: dict[str, int] = {}
        self.tri: dict[str, int] = {}
        self.meta: dict = {}
        self.total = 0
        self._loaded = False
        if not lazy:
            self.load()

    # ------------------------------------------------------------------

    def load(self) -> "NgramLM":
        if self._loaded:
            return self
        if not self._path.is_file():
            raise FileNotFoundError(
                f"No n-gram model at {self._path}.\n"
                f"Build one:  python3 scripts/build_ngrams.py <corpus-dir> "
                f"-o {self._path}"
            )
        opener = gzip.open if self._path.suffix == ".gz" else open
        with opener(self._path, "rt", encoding="utf-8") as fh:
            payload = json.load(fh)
        self.uni = payload["unigrams"]
        self.bi = payload["bigrams"]
        self.tri = payload["trigrams"]
        self.meta = payload.get("meta", {})
        # Exclude the sentence markers from the token total so unigram
        # probabilities describe real words.
        self.total = sum(v for k, v in self.uni.items() if k not in (BOS, EOS))
        self._loaded = True
        return self

    @property
    def loaded(self) -> bool:
        return self._loaded

    def info(self) -> dict:
        self.load()
        return {"path": str(self._path), "tokens": self.total,
                "unigrams": len(self.uni), "bigrams": len(self.bi),
                "trigrams": len(self.tri), **{k: v for k, v in self.meta.items()
                                              if k.startswith(("sentences", "min_"))}}

    # ------------------------------------------------------------------
    # Scoring
    # ------------------------------------------------------------------

    def _p_uni(self, w: str) -> float:
        c = self.uni.get(w, 0)
        return c / self.total if c and self.total else FLOOR

    def s_bigram(self, prev: str, w: str) -> float:
        """S(w | prev)."""
        c_prev = self.uni.get(prev, 0)
        if c_prev:
            c = self.bi.get(f"{prev}\t{w}", 0)
            if c:
                return c / c_prev
        return BACKOFF * self._p_uni(w)

    def s_trigram(self, u: str, v: str, w: str) -> float:
        """S(w | u, v)."""
        c_uv = self.bi.get(f"{u}\t{v}", 0)
        if c_uv:
            c = self.tri.get(f"{u}\t{v}\t{w}", 0)
            if c:
                return c / c_uv
        return BACKOFF * self.s_bigram(v, w)

    def slot_score(
        self,
        word: str,
        left: Sequence[str],
        right: Sequence[str],
    ) -> float:
        """
        Log score for *word* occupying a slot, given up to two words either
        side.

        Only the three trigrams that actually contain the slot are scored —
        (l2, l1, w), (l1, w, r1) and (w, r1, r2). Everything else in the
        sentence is identical across candidates and would cancel, so scoring
        it would be wasted work and would dilute the margin.
        """
        self.load()
        l2, l1 = (list(left) + [BOS, BOS])[-2:] if len(left) < 2 else list(left)[-2:]
        r1, r2 = (list(right) + [EOS, EOS])[:2] if len(right) < 2 else list(right)[:2]

        total = math.log(max(self.s_trigram(l2, l1, word), FLOOR))
        total += math.log(max(self.s_trigram(l1, word, r1), FLOOR))
        total += math.log(max(self.s_trigram(word, r1, r2), FLOOR))
        return total

    def sentence_logprob(self, words: Sequence[str]) -> float:
        """Log score of a whole word sequence. Diagnostic, not used in ranking."""
        self.load()
        padded = [BOS, BOS] + list(words) + [EOS]
        return sum(math.log(max(self.s_trigram(padded[i-2], padded[i-1], padded[i]),
                                FLOOR))
                   for i in range(2, len(padded)))

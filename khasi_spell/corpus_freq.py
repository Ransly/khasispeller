"""
corpus_freq.py — replace the engine's structural pseudo-frequencies with
observed corpus counts.

Why
---
`database.py` assigns a word's "frequency" from its role in the lexicon:
root 20, surface form 12 or 15, +5 if morphologically transparent, compound
constituents 8-19. Across 34,224 forms that produces **11 distinct values**.
It cannot distinguish a word used ten thousand times from one used once, so
the ranking tie-break and the confidence gate's "frequent" signal are both
running on a placeholder.

This module substitutes counts measured from a real corpus. Build the list
first:

    python3 scripts/build_corpus_freq.py ../rupang_data -o data/corpus_freq.json

Then:

    from khasi_spell import KhasiSpeller, corpus_freq
    sp = KhasiSpeller(eager=True)
    corpus_freq.apply(sp)

Scaling
-------
Counts are log-scaled rather than used raw. The raw range is 1 to ~729,000;
dropped into the ranking key unchanged, frequency would swamp every other
signal — the morphological-validity bonus (3) and the shared-onset bonus
(3 per phoneme) would stop mattering at all, and a common word one edit away
would beat the right rarer word every time. `log2(1 + count) * weight` keeps
frequency in the same order of magnitude as the other terms while preserving
their ordering, which is also what a noisy-channel prior does.

Vocabulary is deliberately NOT extended. The corpus contains English,
proper nouns and misspellings; admitting them as valid Khasi would defeat
the point. Only the values of words the lexicon already knows are updated.
"""
from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Optional

from khasi_engine import paths as _paths

DEFAULT_PATH = _paths.data_file("corpus_freq.json")

# Diacritic -> plain, for inheriting counts (see FOLD_INHERIT below).
_FOLD_TABLE = str.maketrans({
    "ï": "i", "ñ": "n",
    "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ý": "y",
})

# log2(1+count) * WEIGHT. At 2.0: count 1 -> 2, 85 -> 13, 2,971 -> 23,
# 22,530 -> 29. Differences between genuinely commoner and rarer words
# exceed the bonus terms (3), so frequency decides where it should,
# without erasing them.
DEFAULT_WEIGHT = 2.0

# Words the corpus never uses. They are not necessarily bad — the checker
# expands the lexicon's table with prefix+root combinations (about 160k, see
# spell_checker._expand_frequency_table), most of which are
# well formed but simply unattested. Flooring them all to one constant throws
# away the structural signal (root vs generated form) without replacing it,
# so instead their original pseudo-frequency is compressed into a low band
# that sits below anything the corpus actually attests.
DEFAULT_FLOOR = 1

# The corpus contains ZERO tokens with ï or ñ across 6.67M words — the
# diacritics were either stripped in preparation or never typed. Taken
# literally that demotes every one of the ~3,100 diacritic-bearing lexicon
# words to the unattested band, which is an artefact of how the corpus was
# written rather than evidence that the words are rare. So a word with no
# count of its own inherits the count of its diacritic-free spelling,
# discounted slightly to keep the attested form ahead on a genuine tie.
FOLD_INHERIT = True
FOLD_DISCOUNT = 0.9

UNATTESTED_BAND = 3      # unattested values are squeezed into 1..UNATTESTED_BAND
UNATTESTED_MAX_PSEUDO = 25   # the engine's pseudo-frequencies top out around 20


def load(path: Optional[str | Path] = None) -> dict:
    """Load the frequency file. Returns {"meta": {...}, "counts": {...}}."""
    p = Path(path) if path else _paths.data_file("corpus_freq.json")
    if not p.is_file():
        raise FileNotFoundError(
            f"No corpus frequency list at {p}.\n"
            f"Build one:  python3 scripts/build_corpus_freq.py <corpus-dir> -o {p}"
        )
    with open(p, encoding="utf-8") as fh:
        return json.load(fh)


def scale(count: int, weight: float = DEFAULT_WEIGHT) -> int:
    """Log-scale a raw count into the engine's frequency range."""
    return int(round(math.log2(1 + count) * weight))


def _remap(table, counts: dict, weight: float, floor: int) -> dict:
    """Corpus-scaled values for every key of *table*, which is not modified.

    Attested words get `scale(count)`. A diacritic word the corpus cannot
    type inherits its plain spelling's count (FOLD_INHERIT). Anything else
    keeps its structural pseudo-frequency, compressed into the low band so a
    lexicon root still outranks a generated combination while both stay
    below any word the corpus has actually seen.
    """
    out: dict = {}
    for word, pseudo in table.items():
        c = counts.get(word)
        if c is None and FOLD_INHERIT:
            plain = word.translate(_FOLD_TABLE)
            if plain != word and counts.get(plain):
                c = int(counts[plain] * FOLD_DISCOUNT)
        if c is not None:
            out[word] = max(floor, scale(c, weight))
        else:
            p = pseudo if isinstance(pseudo, (int, float)) else 0
            out[word] = max(floor, min(UNATTESTED_BAND,
                                       int(round(p / UNATTESTED_MAX_PSEUDO
                                                 * UNATTESTED_BAND))))
    return out


def apply(
    speller: Any,
    path: Optional[str | Path] = None,
    weight: float = DEFAULT_WEIGHT,
    floor: int = DEFAULT_FLOOR,
    extend_vocabulary: bool = False,
) -> dict:
    """
    Rewrite ONE speller's frequency values from corpus counts.

    Accepts either a `KhasiSpeller` or a raw `KhasiSpellChecker`.

    Only the checker's own tables change: its Speller's `nlp_data`, which
    the vote and the final ranking read, and a separate corpus-scaled copy
    of the lexicon's table for the first-pass candidate search. The shared
    lexicon is never written. It used to be, and because every speller in a
    process shares one lexicon object, a second speller re-applied the
    squash to the already-squashed values of the first — its unattested
    band collapsed from 1-3 to 1 — and a `use_corpus_freq=False` speller
    built afterwards got corpus values anyway.

    Idempotent: a second call on the same checker returns the first call's
    summary and changes nothing.

    Parameters
    ----------
    extend_vocabulary :
        Add corpus words the lexicon lacks. Off by default and rarely what
        you want: the corpus carries English, proper nouns and the writers'
        own misspellings, and admitting them makes the checker accept them.

    Returns a summary dict of what changed.
    """
    # Accept the facade or the engine object.
    analyser = getattr(speller, "analyser", None)
    spell = analyser.spell if analyser is not None else speller
    done = getattr(spell, "_corpus_freq_summary", None)
    if done is not None:
        return done
    inner = getattr(spell, "_speller", None)
    if inner is None or not hasattr(inner, "nlp_data"):
        raise RuntimeError("speller has no nlp_data to update")

    payload = load(path)
    counts: dict[str, int] = payload["counts"]
    nlp = inner.nlp_data

    scaled = _remap(nlp, counts, weight, floor)
    matched = sum(1 for w in nlp if w in counts)
    inherited = sum(1 for w in nlp if w not in counts and FOLD_INHERIT
                    and w.translate(_FOLD_TABLE) != w
                    and counts.get(w.translate(_FOLD_TABLE)))
    missing = len(nlp) - matched - inherited
    nlp.update(scaled)

    # The phonotactic-gate path ranks candidates inside
    # _levenshtein_suggestions, which reads the lexicon's frequency table
    # rather than the Speller's nlp_data. Left on pseudo-frequencies, every
    # lexicon word there scores 20 and ties fall through to alphabetical
    # order — which is how 'shnng' ended up suggesting 'shna' over the far
    # commoner 'shnong'. The checker gets its own corpus-scaled copy.
    db = getattr(spell, "_db", None)
    db_updated = 0
    if db is not None and hasattr(spell, "set_ranking_frequencies"):
        ranking = _remap(db.to_freq_dict(), counts, weight, floor)
        db_updated = sum(1 for w in ranking if w in counts)
        spell.set_ranking_frequencies(ranking)

    added = 0
    if extend_vocabulary:
        for word, c in counts.items():
            if word not in nlp:
                nlp[word] = max(floor, scale(c, weight))
                added += 1

    summary = {
        "vocabulary": len(nlp),
        "matched_in_corpus": matched,
        "absent_from_corpus": missing,
        "inherited_from_folded": inherited,
        "added_to_vocabulary": added,
        "coverage": round(matched / (matched + missing), 4) if (matched + missing) else 0.0,
        "corpus_tokens": payload["meta"].get("tokens"),
        "corpus_types": payload["meta"].get("types"),
        "weight": weight,
        "distinct_values": len(set(nlp.values())),
        "db_freq_updated": db_updated,
    }
    spell._corpus_freq_summary = summary
    spell._corpus_freq_applied = True
    return summary

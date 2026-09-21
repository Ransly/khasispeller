"""
generate.py — on-demand morphological candidate generation.

The problem this solves
-----------------------
The checker uses two different definitions of "is a Khasi word". To decide
whether what you *typed* is valid it asks the grammar: the confidence vote in
spell_checker.py gives an affixed parse 2 of the 3 points needed, so a word
the lexicon has never seen is accepted when morphology derives it. To decide
what you *meant*, candidate generation iterates `db.all_surface_forms()` — a
fixed list of 15,174 strings read out of the lexicon. A word must be in that
list to be offered.

The two definitions disagree, and the disagreement was measured on 6 September
2026 against a 23,254-word target vocabulary: 64.1% of the targets are outside
the candidate pool, and **50.4% of those are words the checker accepts when
typed correctly**. At item level, 29.5% of a 12,000-item sample was flagged as
an error while the intended word was structurally impossible to suggest.

    check("pynïarap")   -> accepted, {morphology 2, phonotactic 1}
    suggest("pynarap")  -> ['pynaram', 'pynrap', 'bynrap', 'pynïap', 'pynap']

The right answer is missing from a list of five, for a word the same process
would have accepted a line earlier.

The analyser runs word -> structure. What is missing is the other direction,
structure -> word. This module supplies it, without an FST and without
enumerating the language.

Why on demand rather than pre-generated
---------------------------------------
Pre-generating is the obvious move and the wrong one. 8,896 roots against 18
prefixes is ~160k forms before compounds; the delete index currently holds
410,207 keys over 15,174 forms for about 70 MB, so an 11x larger pool means
roughly 4.4M keys and several hundred MB on a process that already peaks near
644 MB. The source comments record a 512 MB free-tier OOM as a real event.

So nothing is generated until a token has already failed the gate, and then
only forms within reach of that token. The existing delete index is reused to
find nearby roots, so this module adds **no persistent memory at all**.

Two routes, because the evidence says one is not enough
-------------------------------------------------------
Of the 3,147 accepted-but-unreachable targets measured, only 24.8% begin with
a productive prefix. 69.5% carry no recognised prefix and are overwhelmingly
compounds — `sngewdonkam`, `dohmluh`, `mawmluh`, `kheiñsting`, `hynriewspah`.
A prefix-only generator would address a quarter of the problem.

  route 1  prefix + root   strip a productive prefix, find roots near the
                           residue, re-apply the prefix with assimilation
  route 2  root + root     split the token at each position, keep splits
                           where both halves are near real roots

What guards precision
---------------------
A generator invents forms, so every constraint below exists to stop it
inventing rubbish that competes for suggestion slots:

  * **only productive prefixes.** `jing- pyn- nong- ia- ïa- sngew-` carry
    `productive: true` in the lexicon's own morphology table. The other
    twelve are described there as lexicalised or fossilised inventories
    ("Tiny inventory (3 entries)", "Fossilised in ~14 adverbs"), and
    generating with those would manufacture words that do not exist.
  * **every root must be a real lexicon entry with a gloss.** All 11,952
    single-word entries carry one, so this is equivalent to "the root is a
    dictionary headword". This is where a gloss test belongs — asking
    whether the *generated* form has a gloss would be circular, since a form
    with a lexicon entry would already be in the candidate pool and would
    never have needed generating.
  * **the result must not already be reachable**, so generation only ever
    adds to the pool.
  * **the caller's `_offerable` still applies** — vowel present, legal
    letters, shares material with the input.
  * **a distance penalty** (default 0.5, half an edit) so a real headword
    always outranks a generated form at the same distance.

Part-of-speech is deliberately NOT a gate. The prefix table records
`applies_to` (`pyn-` selects verbs, `jing-` verbs and adjectives), which is
tempting, but the lexicon's own POS labels are not reliable enough to gate on:
`rap` "V. to help" is tagged `other`, and `pharshi` is tagged `noun` although
`ïapharshi` is a real word. A strict POS filter would reject both of the
examples that motivated this module. Compatibility is computed and returned
for ranking and for later measurement, and nothing is dropped on it.
"""
from __future__ import annotations

import os
from typing import Any, Callable, Iterable, Iterator, Optional

from khasi_engine.database import OPEN_CLASSES, canonical_pos

# Prefixes carrying productive: true in morphology.prefixes. Read from the
# database at build time rather than hard-coded here, with this list as the
# fallback when the table is unavailable.
_FALLBACK_PRODUCTIVE = ("jing", "pyn", "nong", "sngew", "ia", "ïa")

# Charged against every generated form so a genuine headword at the same
# distance always wins. Kept to a quarter-edit rather than a half: at 0.5 a
# correct generated form one edit from the input loses to *any* lexicon form
# one edit away, and on the motivating case `ïaparshi` that pushed the right
# answer `ïapharshi` out of the top five entirely.
GENERATED_PENALTY = 0.25

# The compound route is charged more than the prefix route, and the reason is
# evidential rather than arbitrary. A prefix derivation uses a prefix the
# lexicon marks productive plus one glossed headword; a compound concatenates
# two headwords, which is a far weaker claim that the result is a word. Left
# equal, the compound route buried both motivating cases: `ïapharshi` and
# `pynïarap` were generated correctly at 1.25 and then tied with six pieces of
# compound noise apiece (`ïapahshi`, `ïaprashi`, `pyntharap`), which the
# frequency tie-break resolved the wrong way.
COMPOUND_PENALTY = 0.4

# A root shorter than this produces too many spurious splits to be useful.
MIN_ROOT_LEN = 3

# How far the residue may sit from a real root.
MAX_ROOT_DIST = 1.0

# Caps on the search, and the reason for each.
#
# The compound route pairs every near-root of the left half with every
# near-root of the right half at every split point. Uncapped that is
# quadratic in the size of two delete-index buckets, and it measured 227 ms
# per word — fifty times the 4.8 ms baseline, which would be a worse
# regression than the bug it fixes. Keeping only the nearest few roots on
# each side costs almost nothing in coverage, since a root three places down
# the list is already further from the input than the eventual answer.
MAX_ROOTS_PER_SIDE = 4
MIN_COMPOUND_LEN = 6

# How permissive the compound route is allowed to be. Measured on 200
# untouched corpus sentences (see the module docstring's precision note):
#
#   "off"       prefix derivation only
#   "anchored"  one half must be an exact headword of at least
#               MIN_COMPOUND_ANCHOR characters; the other may carry one edit
#
# Requiring *both* halves to be exact is not a third option, it is a no-op:
# if both are exact then their concatenation is the input itself, which is
# discarded, and a compound can only correct anything when one half is
# corrected.
#
# The anchor length is what makes the route safe. Unconstrained, it
# over-generates badly on non-Khasi input, because the checker has no
# language identifier and cannot tell a misspelt Khasi compound from an
# English word. Measured on 200 untouched corpus sentences, an anchor of any
# length produced 10 spurious flags, every one on a borrowing or a piece of
# scan damage:
#
#   elekshon  -> blekshon, jlekshon, khlekshon, klekshon   anchor `shon` (4)
#   meeting   -> meiting, mengting, menting, meting        anchor `ting` (4)
#   kandidet  -> kanidet, jandidet, shandidet              anchor `det`  (3)
#   minimum   -> shinimum                                  anchor `mum`  (3)
#   iiengnuti -> diengnuti, liengnuti, miengnuti           anchor `nuti` (4)
#
# Short roots combine with almost anything, so the anchor carries no
# evidence. Requiring five characters removes all of the above while keeping
# the case the route exists for: `sngwdonkam` -> `sngewdonkam`, anchored on
# `donkam` (6) with one edit on `sngew`.
COMPOUND_MODE = os.environ.get("KHASI_SPELL_COMPOUND_MODE", "anchored").lower()

MIN_COMPOUND_ANCHOR = 5   # the exact half
MIN_COMPOUND_PART = 4     # the half carrying the edit



class MorphGenerator:
    """Generates candidate corrections that the lexicon does not contain."""

    def __init__(self, db: Any, levenshtein: Callable[[str, str], float]):
        self._db = db
        self._lev = levenshtein
        self._surfaces = frozenset(db.all_surface_forms())
        self._roots: dict[str, str] = {}      # form -> grammatical class
        self._prefixes: list[tuple[str, tuple[str, ...], tuple[str, ...]]] = []
        self._memo: dict[tuple, list[tuple[str, float]]] = {}
        # Word-class lookups, cached for the generator's lifetime rather than
        # per call: the same roots recur across every word checked.
        self._content_memo: dict[str, bool] = {}
        self._build()

    # ------------------------------------------------------------------
    def _build(self) -> None:
        """Index the glossed single-word headwords, and the prefix table."""
        for entry in self._db.all_entries():
            form = (entry.get("surface_form") or "").strip().lower()
            if not form or " " in form or len(form) < MIN_ROOT_LEN:
                continue
            if not entry.get("english_gloss"):
                continue
            self._roots.setdefault(form, (entry.get("grammatical_class") or "").lower())

        table = {}
        try:
            table = (self._db.morphology or {}).get("prefixes", {}) or {}
        except Exception:
            table = {}

        from khasi_engine import assimilation as _asm
        surface_variants: dict[str, set[str]] = {}
        for rule in _asm.get_all_rules():
            trigger = (rule.get("trigger_prefix") or "").rstrip("-")
            surface = rule.get("surface_prefix")
            if trigger and surface:
                surface_variants.setdefault(trigger, set()).add(surface)

        names = [k for k, v in table.items() if v.get("productive")] or list(
            _FALLBACK_PRODUCTIVE)
        for name in names:
            spec = table.get(name, {})
            forms = {name} | set(spec.get("variants") or ()) | surface_variants.get(name, set())
            applies = tuple(a.lower() for a in (spec.get("applies_to") or ()))
            self._prefixes.append((name, tuple(sorted(forms, key=len, reverse=True)), applies))

    # ------------------------------------------------------------------
    @property
    def root_count(self) -> int:
        return len(self._roots)

    def _near_roots(self, residue: str, max_dist: float,
                    cap: int = MAX_ROOTS_PER_SIDE) -> list[tuple[str, float]]:
        """
        The *cap* nearest real headwords within *max_dist* of *residue*.

        Results are memoised for the lifetime of one `candidates()` call: the
        prefix and compound routes ask about overlapping substrings of the
        same word, so the same residue is looked up several times.
        """
        if len(residue) < MIN_ROOT_LEN:
            return []
        key = (residue, max_dist, cap)
        hit = self._memo.get(key)
        if hit is not None:
            return hit

        found: list[tuple[str, float]] = []
        if residue in self._roots:
            found.append((residue, 0.0))
        index = getattr(self._db, "_delete_index", None)
        pool: Iterable[str] = index.candidates(residue) if index is not None else ()
        lo, hi = len(residue) - int(max_dist) - 1, len(residue) + int(max_dist) + 1
        for form in pool:
            # Length is a free lower bound on edit distance, and the buckets
            # are large enough that skipping the distance computation here is
            # most of the speed-up.
            if not (lo <= len(form) <= hi) or form == residue:
                continue
            if form not in self._roots:
                continue
            d = self._lev(residue, form)
            if 0 < d <= max_dist:
                found.append((form, d))
        # Distance THEN alphabetical. Sorting on distance alone left ties in
        # whatever order the candidate set happened to iterate in, and `cap`
        # then kept an arbitrary few of them — so the same input produced a
        # different set of suggestions in each process, since Python
        # randomises string hashing. Three consecutive runs of `jingilehkai`
        # gave three different top-5s before this.
        found.sort(key=lambda t: (t[1], t[0]))
        found = found[:cap]
        self._memo[key] = found
        return found

    def _compound_parts(self, word: str, max_root_dist: float, min_anchor: int):
        """Split *word* into two roots, yielding (left, right) pairs.

        Route 2's rules, unchanged and in one place: exactly one half
        carries the edit, the other is an exact root of at least
        *min_anchor* characters, both halves are content words, and neither
        is itself a prefix derivation.

        `min_anchor` is a parameter rather than a constant because a
        prefixed variant of this route was tried — the idea being that a
        productive prefix is independent evidence, so the five-character
        anchor could be relaxed after one matched. It was reverted. It did
        not reach its motivating case (`pyntreika` -> `pyntreikam` needs
        `trei` + a two-character `ka`, below MIN_ROOT_LEN, and `kam` is cut
        by MAX_ROOTS_PER_SIDE's alphabetical tie-break regardless), and at
        an anchor of 3 it re-admitted the documented noise through the
        prefix door: `iiengnuti` -> `iaangnuti`, `iabengnuti`, `iadengnuti`.
        The parameter is kept so the next attempt starts from the
        measurement rather than repeating it.
        """
        if len(word) < MIN_COMPOUND_LEN:
            return
        min_part = min(MIN_COMPOUND_PART, min_anchor)
        for i in range(min_part, len(word) - min_part + 1):
            lefts = self._near_roots(word[:i], max_root_dist)
            if not lefts:
                continue
            rights = self._near_roots(word[i:], max_root_dist)
            for lroot, dl in lefts:
                if len(lroot) < min_part:
                    continue
                for rroot, dr in rights:
                    if len(rroot) < min_part:
                        continue
                    if (dl > 0) == (dr > 0):
                        continue
                    anchor = rroot if dl > 0 else lroot
                    if len(anchor) < min_anchor:
                        continue
                    if not (self._is_content(lroot) and self._is_content(rroot)):
                        continue
                    if (self._starts_with_prefix(lroot)
                            or self._starts_with_prefix(rroot)):
                        continue
                    yield lroot, rroot

    def _starts_with_prefix(self, word: str) -> bool:
        """True when *word* opens with one of the productive prefixes.

        Used to keep the compound route out of route 1's territory: a form
        that is already `jing-` over something is a derivation, not a
        compoundable root.
        """
        w = (word or "").lower().lstrip("'")
        for canon, forms, _applies in self._prefixes:
            for surface in forms:
                s = surface.rstrip("-")
                # Longer than the prefix itself, so the bare prefix as a
                # standalone word is caught but a root that merely happens to
                # equal one is not silently extended.
                if s and w != s and w.startswith(s):
                    return True
            if w == canon.rstrip("-"):
                return True
        return False

    def _is_content(self, word: str) -> bool:
        """True when the lexicon records *word* — spelled that way — as an
        open-class word.

        The spelling check is the point. `db.lookup()` falls through to the
        root-lemma index and then to an orthographic fold, so `jingih`,
        `jingai` and even `jing-` all come back with `pos=noun` by resolving
        onto some `jing…` entry. Taking that at face value let the compound
        route treat them as real halves, which is how `jingailehkai` and
        `jingihlehkai` were manufactured. A compound half has to be a word as
        written, not something that merely resolves to one.
        """
        cached = self._content_memo.get(word)
        if cached is not None:
            return cached
        w = word.lower()
        ok = False
        for m in (self._db.lookup(w) or []):
            if canonical_pos(m.get("grammatical_class")) not in OPEN_CLASSES:
                continue
            if ((m.get("surface_form") or "").lower() == w
                    or (m.get("root_lemma") or "").lower() == w):
                ok = True
                break
        self._content_memo[word] = ok
        return ok

    @staticmethod
    def _join(prefix: str, root: str) -> str:
        """Apply the prefix forwards, letting assimilation fire where it should."""
        from khasi_engine.assimilation import should_assimilate
        applied, surface = should_assimilate(prefix + "-", root)
        return surface if applied and surface else prefix + root

    # ------------------------------------------------------------------
    def candidates(
        self,
        word: str,
        max_dist: float = 2.0,
        max_root_dist: float = MAX_ROOT_DIST,
        limit: int = 40,
    ) -> list[tuple[float, str, str]]:
        """
        Forms derivable from real roots that lie within *max_dist* of *word*.

        Returns `(distance, form, route)` triples, distance already carrying
        GENERATED_PENALTY, sorted nearest first. Forms the lexicon already
        holds are never returned — this only ever extends the pool.
        """
        w = (word or "").lower().strip()
        if len(w) < 4:
            return []
        self._memo.clear()
        out: dict[str, tuple[float, str]] = {}
        penalties = {"prefix": GENERATED_PENALTY, "compound": COMPOUND_PENALTY}

        def offer(form: str, route: str) -> None:
            if not form or form == w or form in self._surfaces:
                return
            d = self._lev(w, form)
            if d <= 0 or d > max_dist:
                return
            d += penalties[route]
            prev = out.get(form)
            if prev is None or d < prev[0]:
                out[form] = (d, route)

        # route 1 — a productive prefix over a root near the residue
        for canon, forms, _applies in self._prefixes:
            for surface in forms:
                # Allow the prefix itself to be mistyped by one character:
                # `jinkerkut` should still reach `jing-`.
                for k in range(max(2, len(surface) - 1), min(len(w) - MIN_ROOT_LEN, len(surface) + 1) + 1):
                    head, residue = w[:k], w[k:]
                    if self._lev(head, surface) > 1.0:
                        continue
                    for root, _d in self._near_roots(residue, max_root_dist):
                        offer(self._join(canon, root), "prefix")


        # route 2 — a bare compound: both halves near a real root, and
        # **at least one of them exactly a root**.
        #
        # Allowing an edit on both halves gives two degrees of freedom for
        # manufacturing a string, which is how `ïaparshi` produced `ïapahshi`
        # and `ïaprashi`. Requiring one side to be exact keeps the cases that
        # matter — `sngwdonkam` is a misspelt `sngew` beside an exact
        # `donkam` — while removing most of the noise.
        #
        # The split rules now live in `_compound_parts`, shared with route 1b
        # above, which passes a lower anchor floor because it has the prefix
        # as evidence. Everything this route enforced it still enforces: the
        # five-character anchor, both halves content words (`jingailehkai`,
        # `jingitlehkai`), and neither half a prefix derivation.
        if COMPOUND_MODE != "off":
            for lroot, rroot in self._compound_parts(
                    w, max_root_dist, MIN_COMPOUND_ANCHOR):
                offer(lroot + rroot, "compound")

        ranked = sorted((d, form, route) for form, (d, route) in out.items())
        return ranked[:limit]


_GENERATOR: Optional[MorphGenerator] = None


def get_generator(db: Any, levenshtein: Callable[[str, str], float]) -> MorphGenerator:
    """Process-wide generator, built on first use."""
    global _GENERATOR
    if _GENERATOR is None or _GENERATOR._db is not db:
        _GENERATOR = MorphGenerator(db, levenshtein)
    return _GENERATOR


def reset_generator() -> None:
    global _GENERATOR
    _GENERATOR = None

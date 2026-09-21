"""
analyser.py — Main Orchestrator
Khasi NLP Engine

The single entry point for all analysis. Instantiate KhasiAnalyser once,
then call analyse(word) or check_sentence(text) as many times as needed.

Pipeline (morphology-gated with phonological spell-through):
  Phase 1 → phonology.validate()               always runs
  Phase 2 → morphology.parse()                 always runs
  Phase 3 → assimilation.check()               always runs
  Phase 4 → complex.detect()                   always runs
  Spell   → spell_checker.suggest()            runs when phases 1-4
                                               do NOT resolve the word,
                                               INCLUDING when Phase 1 fails
                                               (phonotactic errors still get
                                               correction suggestions)

The spell checker receives a morph_check callable that lets it query
the full morphological pipeline before generating any edit-distance
candidates. This means:

  - Morphologically valid words (even those not in the static lexicon,
    such as productive prefix+root forms) are NEVER flagged as misspelled.
  - Spell suggestions are re-ranked by morphological validity: a candidate
    that parses as a valid Khasi word scores higher than one that is merely
    phonotactically legal.
  - Sentence checking skips any token that passes the morphological gate.
  - Phonotactically invalid words (e.g. "bamm") receive correction
    suggestions via Levenshtein distance (e.g. "bam"), rather than being
    blocked by morphological gating.
"""

from __future__ import annotations

import re
from typing import Optional
from pathlib import Path

from khasi_engine.database import KhasiDB, OPEN_CLASSES, canonical_pos
from khasi_engine import phonology, morphology, assimilation, complex as complex_words
from khasi_engine.spell_checker import KhasiSpellChecker


# The Lynnong II noun-class clitics. `ia` in front of one of these is the
# preposition taking its article, not the reciprocal prefix taking a root.
_NOUN_CLASS_CLITICS = frozenset({"ka", "u", "ki", "i"})


class KhasiAnalyser:
    """
    Main analysis engine. Thread-safe after __init__.

    Parameters
    ----------
    db_path    : path to khasi_db.json (default: ../data/khasi_db.json)
    use_hybrid : enable FastText re-ranking in spell checker
    model_path : path to FastText .model file (required if use_hybrid=True)
    """

    def __init__(
        self,
        db_path: Optional[str | Path] = None,
        use_hybrid: bool = False,
        model_path: Optional[str] = None,
    ):
        self.db = KhasiDB(db_path)

        # ── Phase A: derive _PROTECTED_WORDS from the data ─────────────
        # morphology.FREE_MORPHEMES is the data-driven function-word set
        # (built from morphology.free_morphemes JSON block + spec adds);
        # see morphology._rebind_morph_constants. Plus a handful of high-
        # frequency words too short to appear as standalone entries.
        # KhasiDB.__init__ above has already triggered set_morphology_cache
        # on PG mode, so FREE_MORPHEMES is populated by the time we read it.
        self._PROTECTED_WORDS = frozenset(
            set(morphology.FREE_MORPHEMES)
            | {"ai"}   # short common verb not in FREE_MORPHEMES
        )

        # Build the spell checker first (without morph_check yet)
        self.spell = KhasiSpellChecker(
            self.db,
            use_hybrid=use_hybrid,
            model_path=model_path,
        )

        # Inject the morphological validity oracle into the spell checker.
        # This is done after construction to avoid a circular dependency:
        # the spell checker needs the analyser's pipeline, but the analyser
        # needs the spell checker to exist first.
        self.spell.set_morph_check(self._is_morphologically_valid)
        # Richer signal — lets the spell-checker's confidence gate know
        # whether morphology accepted the word via real affixation or as a
        # bare-root concatenation (which is too permissive on its own).
        self.spell.set_morph_diagnostic(self._morphology_diagnostic)

    # ------------------------------------------------------------------
    # Morphological validity oracle  (used by the spell checker)
    # ------------------------------------------------------------------

    # _PROTECTED_WORDS — class-level fallback ONLY; the live value is built
    # in __init__ from morphology.FREE_MORPHEMES so it stays in sync with
    # the JSON. This class-level frozenset is used only if an instance is
    # never constructed (e.g. doctests, isolated imports) and documents the
    # historical minimum set.
    _PROTECTED_WORDS: frozenset = frozenset([
        # Noun-class clitics (Lynnong II)
        "u", "ka", "i", "ki",
        # Common prepositions / particles
        "da", "sha", "na", "ha", "ba", "la",
        # Common pronouns
        "nga", "phi", "ngi",
        # Common verbs too short to be in DB
        "ai", "ia",
    ])

    def _is_morphologically_valid(self, word: str) -> bool:
        """
        Return True if *word* is accepted by at least one of phases 1-4,
        is a known grammatical particle/clitic, OR appears as a recognised
        token in the lexicon (including as a component of multi-word entries).

        For Phase 3 (assimilation), the rule must have root_confirmed=True —
        i.e. the reconstructed root must exist in the lexicon. This prevents
        false positives where the surface prefix pattern matches but the
        resulting root is not a real Khasi word (e.g. 'pyliait' would match
        the n_to_l pattern but reconstruct root 'iait' which is not in the DB).
        """
        w = word.lower().strip()

        # Protected clitics and grammatical particles
        if w in self._PROTECTED_WORDS:
            return True

        # Lexicon presence check — covers words that appear only inside
        # multi-word or compound entries (e.g. "sohmon" in a phrasal entry).
        # db.is_known() checks both surface_index and compound_tokens.
        if self.db.is_known(w):
            return True

        # Phase 1 — must pass for anything else to matter
        if not phonology.validate(w)["pass"]:
            return False

        # Phase 2 — direct match, prefix strip, or infix detection
        p2 = morphology.parse(w, self.db.lookup)
        if p2["pass"]:
            return True

        # Phase 3 — assimilation rule match (root must be confirmed in lexicon)
        p3 = assimilation.check(w, self.db.lookup)
        if p3.get("rule_detected") and p3.get("root_confirmed", False):
            return True

        # Phase 4 — compound, reduplication, tien kynnoh, shim kylliang
        p4 = complex_words.detect(w, self.db.lookup)
        if p4.get("detected"):
            # Check the WHOLE WORD before trusting a split. `compound_fusion`
            # is the one Phase-4 type that is pure inference: it accepts any
            # string that happens to divide into two lexicon entries, without
            # the compound itself being attested anywhere. That is how `suroh`
            # was accepted as sur+oh ("sound"+"cut") when the writer meant
            # `surok` ("road"), and `patbah` as pat+bah when they meant
            # `paitbah` — each one edit from a real word the lexicon already
            # holds, each silently accepted and never offered as a correction.
            #
            # An attested word one edit away is far better evidence than an
            # unattested split, so it wins. The other Phase-4 types are not
            # touched: compound_registered and shim_kylliang are attested in
            # the lexicon, and reduplication is a structural pattern rather
            # than a guess about word boundaries.
            if p4.get("type") == "compound_fusion":
                if self._outranked_by_attested(w) is not None:
                    return False
            return True

        return False

    # Distance at which an attested word outranks a speculative analysis. One
    # edit: `suroh`/`surok` and `patbah`/`paitbah` are both exactly that, and
    # so is `nongihkai`/`nonghikai` — `_levenshtein` counts a transposition as
    # a single edit, so this radius covers the whole Damerau set.
    _ATTESTED_VETO_DIST = 1.0

    def _closer_real_word(self, word: str) -> Optional[str]:
        """The nearest ATTESTED word within `_ATTESTED_VETO_DIST`, or None.

        Uses the symmetric-delete index directly rather than going through
        `spell.suggest()`, which would recurse: suggest() consults this very
        method through the morphology gate.
        """
        cache = self.__dict__.setdefault("_closer_cache", {})
        if word in cache:
            return cache[word]
        index = getattr(self.db, "_delete_index", None)
        if index is None:
            return None
        try:
            from khasi_engine.spell_checker import _levenshtein
        except Exception:
            return None
        best, best_d = None, None
        for cand in index.candidates(word):
            if cand == word or " " in cand or not self.db.is_known(cand):
                continue
            d = _levenshtein(word, cand)
            if d <= self._ATTESTED_VETO_DIST and (best_d is None or d < best_d):
                best, best_d = cand, d
        cache[word] = best
        return best

    def _free_morpheme_plus_root(self, word: str):
        """`word` read as a free morpheme welded to a known root, or None.

        `babam` is `ba` + `bam` — the relativiser written solid against a
        root with 2,403 corpus uses. That is a grammatical construction, not
        a guess about a compound seam, so it must survive the fusion veto
        even though `baibam` sits one edit away. The fused-compound route
        cannot see it: `ba` is a free morpheme, which that route refuses to
        split on, so it offers `bab` + `am` instead — junk that happened to
        be accepted.

        Measured: this readmits none of the vetoed non-words.
        """
        from khasi_engine import morphology as _m
        free = {x.lower() for x in getattr(_m, "FREE_MORPHEMES", [])}
        for i in range(1, len(word) - 1):
            head, rest = word[:i], word[i:]
            if head in free and len(rest) >= 2 and self.db.is_known(rest):
                return head, rest
        return None

    def _outranked_by_attested(self, word: str):
        """The attested word that should outrank a speculative analysis of
        *word*, or None when the analysis stands.

        Used by every route that can accept a word the lexicon does not hold:
        the fused-compound split (`_is_morphologically_valid`), the affixed
        parse (`_morphology_diagnostic`, for the confidence vote) and
        `analyse()`. The question each of them is really asking is the same —
        "is there a real word close enough that the writer more likely meant
        it?" — so they ask it in one place.

        Returns None when nothing real is close enough, when the word is a
        free morpheme plus a root, or when the word is itself attested. That
        last guard matters: without it a lexicon word could be vetoed by its
        own one-edit neighbour (`hok` by `hek`), which is why callers used to
        be responsible for only reaching here after an `is_known` miss.
        """
        w = (word or "").lower().strip()
        if not w or w in self._PROTECTED_WORDS or self.db.is_known(w):
            return None
        if self._free_morpheme_plus_root(w) is not None:
            return None
        return self._closer_real_word(w)

    def _transposition_of_attested(self, word: str):
        """An attested word differing from *word* by one adjacent
        transposition, or None.

        The strict sibling of `_outranked_by_attested`, and the only form of
        veto an affixed parse gets. The distance-1 ball is too wide there: a
        productive derivation sits one substitution away from a DIFFERENT
        derivation all the time, because substituting a letter inside the
        root usually just names another morpheme. `nonglah` (nong- + lah
        "be able") lost to `nongrah` (nong- + rah "carry"), `pynloit` to
        `pynkoit`, `kynshaid` to `kynshait` — each a real word the lexicon
        holds, none of them what the writer meant. `_CONFUSION_COST` makes
        this worse by design, pricing the d/t and ng/g pairs below a full
        edit precisely because Khasi varies there; that is phonemic
        variation, not a typo.

        A transposition carries no such ambiguity. It leaves the letters
        exactly as the writer typed them and only reorders two, so it is a
        slip of the fingers rather than a different morpheme — which is why
        `nongihkai` for `nonghikai` is a spelling error and `nonglah` is not.

        Enumerating the transpositions and asking the lexicon is both exact
        and cheaper than a distance search: len(word)-1 lookups, no index.
        """
        w = (word or "").lower().strip()
        if not w or w in self._PROTECTED_WORDS or self.db.is_known(w):
            return None
        if self._free_morpheme_plus_root(w) is not None:
            return None
        for i in range(len(w) - 1):
            if w[i] == w[i + 1]:
                continue
            cand = w[:i] + w[i + 1] + w[i] + w[i + 2:]
            if self.db.is_known(cand):
                return cand
        return None

    def _is_content_word(self, word: str) -> bool:
        """True when the lexicon records *word* — spelled that way — as an
        open-class word.

        Khasi compounds are built from content words: jaiñ+sem is noun+noun,
        leh+kai verb+noun. An interjection is not compounding material, and
        the 2-letter interjection entries the 1906 dictionary carries (`ho`,
        `ab`) otherwise let almost any string be read as a compound.

        Open classes come from `database.OPEN_CLASSES`, which follows the
        inventory declared in `grammar_schema.pos_inventory`, and the value is
        normalised first — the flat records this reads say "adjective" where
        the schema says ADJ.

        The spelling check matters too: `db.lookup()` falls through to the
        root-lemma index and an orthographic fold, so a non-word can resolve
        onto a real entry and inherit its part of speech.
        """
        w = (word or "").lower()
        for m in (self.db.lookup(w) or []):
            if canonical_pos(m.get("grammatical_class")) not in OPEN_CLASSES:
                continue
            if ((m.get("surface_form") or "").lower() == w
                    or (m.get("root_lemma") or "").lower() == w):
                return True
        return False

    def _morphology_diagnostic(self, word: str) -> dict:
        """
        Richer cousin of _is_morphologically_valid — returns a structured
        signal dict the spell-checker uses for confidence-weighted gating.

        Fields:
          valid       — bool, same answer as _is_morphologically_valid
          has_affix   — bool, True if validity came from real affixation
                        (prefix/infix/suffix/clitic, assimilation rule, or
                        complex word detection). False for bare lexicon
                        hits or protected clitics. This distinction matters
                        because a bare-root parse is too easy to satisfy
                        and was promoting typos like "shilong" to valid.
          reason      — str, which gate fired
          root        — str, the parsed root (morphology path only)
          root_known  — bool, whether that root is itself a known word.
                        A parse whose root is not a word is a decomposition
                        the analyser could make, not one it should trust.
          outranked_by— str, present only when an attested word sits within
                        one edit of an affixed parse. The decomposition may
                        be well formed and still lose to a real word the
                        writer more plausibly meant.
          parse_score — float | None, the parser's own confidence
        """
        w = word.lower().strip()

        if w in self._PROTECTED_WORDS:
            return {"valid": True, "has_affix": False, "reason": "protected"}

        if self.db.is_known(w):
            return {"valid": True, "has_affix": False, "reason": "lexicon"}

        if not phonology.validate(w)["pass"]:
            return {"valid": False, "has_affix": False, "reason": "phonotactic_fail"}

        p2 = morphology.parse(w, self.db.lookup)
        if p2.get("pass"):
            layers = p2.get("layers") or {}
            has_affix = any(k in layers for k in ("prefix", "infix", "suffix", "clitic"))
            # Whether the parse is plausible, not merely possible. A
            # generative analyser will decompose almost any string into
            # affix + remainder; what separates a real derivation from a
            # typo is that its ROOT is a word. Measured on the frozen
            # benchmark, 91% of silently-accepted misspellings had a root
            # that is not in the lexicon, against 17% of genuine derived
            # forms (median parse_score 0.0 vs 2.2).
            #
            # A word root is necessary but not sufficient. `nongihkai` parses
            # as nong- + `ih` ("to see") + -kai over a root the lexicon really
            # does hold, so root_known says yes and the vote scored it
            # morphology 2 + phonotactic 1 = the threshold — accepted in
            # silence, with no suggestion offered. The writer had typed
            # `nonghikai` "instructor" and transposed two letters.
            #
            # The fused-compound route already refuses a split when a real
            # word sits one edit away; an affixed parse deserves the same
            # test, because a productive analyser can decompose a typo just
            # as readily as a word — but only for a transposition, which is
            # the one edit that cannot be a different morpheme. See
            # _transposition_of_attested for the measurement behind that.
            # Reported rather than folded into root_known, which answers a
            # different question and is worth keeping honest for callers
            # that read it.
            root = (layers.get("root") or p2.get("root") or "")
            out = {
                "valid": True,
                "has_affix": has_affix,
                "reason": "morphology",
                "root": root,
                "root_known": bool(root) and self.db.is_known(root),
                "parse_score": (p2.get("affix_info") or {}).get("parse_score"),
            }
            if has_affix:
                outranked = self._transposition_of_attested(w)
                if outranked:
                    out["outranked_by"] = outranked
            return out

        p3 = assimilation.check(w, self.db.lookup)
        if p3.get("rule_detected") and p3.get("root_confirmed", False):
            return {"valid": True, "has_affix": True, "reason": "assimilation"}


        p4 = complex_words.detect(w, self.db.lookup)
        if p4.get("detected"):
            # This branch used to return no `root_known`, and the confidence
            # vote's guard is `if "root_known" in diag` — so for every complex
            # form the guard simply did not run and the parse was assumed
            # plausible, worth the full affixed-morphology weight. That is how
            # `hook` was accepted: split as `ho` + `ok`, morphology 2 +
            # phonotactic 1 = the threshold, with nothing in the lexicon.
            #
            # For a fused compound, "is the root a word" means "are the parts
            # words" — and specifically CONTENT words. `ho` is an interjection
            # ("Int. order") and `ok` an adverb ("Hiccup-like"); a noun built
            # by welding those together is not a plausible reading, whereas
            # jaiñ+sem and leh+kai are noun+noun and verb+noun.
            #
            # This narrows only what the VOTE trusts. The split itself is
            # still reported, so a lexicon word keeps the analysis it had.
            parts = [p for p in (p4.get("components") or []) if p]
            known = None
            if p4.get("type") == "compound_fusion" and parts:
                known = all(self._is_content_word(p) for p in parts)
                # …and the whole word still has to beat a real one. Both
                # halves being content words is not enough on its own:
                # sur+oh and pat+bah are verb+verb and adverb+verb, so they
                # pass the test above, yet the writer meant `surok` and
                # `paitbah` — attested words one edit away. The confidence
                # vote is a second, independent route to acceptance, so the
                # veto has to be applied here as well as in
                # _is_morphologically_valid or the word is still never
                # flagged.
                if known and self._outranked_by_attested(w) is not None:
                    known = False
            out = {"valid": True, "has_affix": True, "reason": "complex"}
            if known is not None:
                out["root"] = parts[0]
                out["root_known"] = known
            return out

        return {"valid": False, "has_affix": False, "reason": "unknown"}

    # ------------------------------------------------------------------
    # Core analysis
    # ------------------------------------------------------------------

    def analyse(self, word: str) -> dict:
        """
        Run all phases on *word* and return a unified result dict.

        The spell checker is only invoked when phases 1-4 do not resolve
        the word, so valid derived forms are never incorrectly suggested
        for correction.

        Returns
        -------
        dict with keys:
            input        : str
            verdict      : str   — "valid" | "invalid" | "derived" | "unknown"
            verdict_desc : str   — human-readable explanation
            phase1       : dict  — phonological validation result
            phase2       : dict  — morphological parse result
            phase3       : dict  — assimilation detection result
            phase4       : dict  — complex word detection result
            spell        : dict  — spell check result (empty when phases resolve)
        """
        import re as _re_a
        # Strip surrounding punctuation EXCEPT apostrophe (') and hyphen (-),
        # which are valid Khasi word characters (glottal stop / compound marker).
        # The original regex used the range '- (apostrophe to hyphen) in the
        # negated class, which accidentally INCLUDED apostrophe in the stripped set.
        word = _re_a.sub(r"^[^\w\u00ef\u00f1\u00cf\u00d1'\-]+|[^\w\u00ef\u00f1\u00cf\u00d1'\-]+$", "", word.strip())
        if not word:
            return {"error": "Empty input"}

        # ── Phases 1–4 always run ────────────────────────────────────
        p1 = phonology.validate(word)
        p2 = morphology.parse(word, self.db.lookup)
        p3 = assimilation.check(word, self.db.lookup)
        p4 = complex_words.detect(word, self.db.lookup)

        # ── Spell checker: only when morphology doesn't resolve ──────
        # For Phase 3 (assimilation), require root_confirmed=True to avoid
        # false positives where the surface prefix pattern matches but the
        # reconstructed root is not a real Khasi word in the lexicon.
        # db.is_known() also catches compound_tokens (words that appear only
        # inside multi-word entries, e.g. "sohmon" in a phrasal entry).
        morph_resolved = (
            p2["pass"]
            or (p3.get("rule_detected") and p3.get("root_confirmed", False))
            or p4.get("detected")
            or (word.lower() in self._PROTECTED_WORDS)
            or self.db.is_known(word.lower())
        )

        # …minus the words a real one outranks. This line is re-deriving the
        # acceptance decision from the raw phase results, which meant it saw
        # none of the vetoes the gate applies: `suroh`, `patbah` and `hook`
        # are all refused by _is_morphologically_valid and flagged by
        # check(), yet came back `derived` with no suggestions from THIS
        # entry point, because `p4.get("detected")` is consulted directly.
        # Both speculative routes are covered — the fused compound and the
        # affixed parse (`nongihkai`) — so that analyse() and check() give
        # the same answer for the same word.
        # Each route asks the same question its own gate asks, so the two
        # cannot drift apart again: the fused compound gets the full
        # one-edit veto that _is_morphologically_valid applies, the affixed
        # parse the transposition-only veto the confidence vote applies.
        # Phase order matters and has to match _is_morphologically_valid's:
        # it returns on p2 before it ever reaches p4, so a word both phases
        # accept is on the affix path, not the fusion path, and gets the
        # affix path's veto. Testing p4 first instead costs
        # `jingkynshaitum` its golden-corpus ruling — it parses as jing- over
        # an OOV remainder AND splits as a fused compound, and the wider
        # fusion veto finds `jingkynshait-um` one edit away.
        if morph_resolved:
            _outranked = None
            if p2.get("pass"):
                if any(k in (p2.get("layers") or {})
                       for k in ("prefix", "infix", "suffix", "clitic")):
                    _outranked = self._transposition_of_attested(word)
            elif p4.get("detected") and p4.get("type") == "compound_fusion":
                _outranked = self._outranked_by_attested(word)
            if _outranked is not None:
                morph_resolved = False

        if not p1["pass"] and morph_resolved:
            # Hyphenated/compound word fails phonotactics when split into
            # segments but IS directly in the DB (e.g. "jingbym-kyrmen").
            # The lexicon match is authoritative — accept as valid/derived.
            sp = {
                "is_known": True,
                "morphologically_valid": True,
                "phonotactically_valid": False,
                "suggestions": [],
                "method": "morphology_gate",
                "in_lexicon": self.db.is_known(word.lower()),
                "gate_reached": 2,
            }
        elif not p1["pass"]:
            # Phonotactically invalid — delegate to the spell checker's
            # Gate 1 which generates Levenshtein suggestions even for
            # phonotactically invalid words (e.g. "bamm" → "bam").
            # This ensures that a correctly identified phonological error
            # is accompanied by actionable correction suggestions.
            sp = self.spell.suggest(word, n=5)
        elif morph_resolved:
            # Word accepted by morphological pipeline — no spell check needed
            sp = {
                "is_known": True,
                "morphologically_valid": True,
                "phonotactically_valid": True,
                "suggestions": [],
                "method": "morphology_gate",
                "in_lexicon": self.db.is_known(word.lower()),
                "gate_reached": 2,
            }
        else:
            # Unknown word — run full spell checker (gate 3)
            sp = self.spell.suggest(word, n=5)

        verdict, verdict_desc = self._compute_verdict(p1, p2, p3, p4, sp, word)

        # ── Meaning-licensing gate (Phase A) ─────────────────────────
        # Selectional check: a single prefix must attach to a base whose PoS is
        # in the prefix's applies_to. If not, the form is NOT a licensed
        # derivation (e.g. prefixing a conjunction/adposition like jing+ban,
        # pyn+da). Conservative — only downgrades a clear violation, and leaves
        # the verdict untouched when it cannot decide (no affix data, base not
        # attested, or a stacked/infixed form).
        licensed, lic_note, lic_hard = self._meaning_licensed(p2)
        if licensed is not None:
            p2["meaning_licensed"] = licensed
            if lic_note:
                p2["meaning_note"] = lic_note
            # HARD downgrade only for unambiguous violations (prefix on a
            # function word). Content-word mismatches stay 'derived' with a soft
            # flag, since the base PoS tag may be wrong / a sense unrecorded.
            if licensed is False and lic_hard and verdict == "derived":
                verdict = "unknown"
                verdict_desc = (
                    "Not a licensed derivation — " + lic_note
                    + ". The prefix does not attach to this base's part of speech."
                )

        # ── The gate's answer wins over the parse's ──────────────────
        # _compute_verdict reads p2["pass"] and reports "derived" without
        # ever consulting `sp`, so a word the checker has just refused still
        # came back as a derivation. The two disagreeing is worse than either
        # answer alone: the UI showed `nongihkai` as a prefixed form of `ih`
        # while the correction `nonghikai` sat unread in the same payload.
        #
        # Downgrade only — a parse is never promoted here, and a word the
        # lexicon holds is never touched.
        if (verdict == "derived"
                and not self.db.is_known(word.lower())
                and sp.get("is_known") is False):
            suggs = sp.get("suggestions") or []
            verdict = "unknown"
            verdict_desc = (
                "Parses as " + verdict_desc[0].lower() + verdict_desc[1:]
                if verdict_desc else "Parses morphologically"
            )
            verdict_desc += (
                ", but is not attested and a known word is closer"
                + (". Did you mean: " + ", ".join(suggs[:3]) + "?" if suggs else ".")
            )

        # Compound/phrasal host entries — only populated when the word resolved
        # via the lexicon as a component token of a multi-word entry. Useful
        # for the frontend to render the host phrases inline.
        compound_hosts = self.db.find_compound_entries(word.lower()) if word else []

        return {
            "input":          word,
            "verdict":        verdict,
            "verdict_desc":   verdict_desc,
            "phase1":         p1,
            "phase2":         p2,
            "phase3":         p3,
            "phase4":         p4,
            "spell":          sp,
            "compound_hosts": compound_hosts,
        }

    def is_multi_word(self, text: str) -> bool:
        """Return True if the input contains multiple whitespace-separated tokens."""
        return len(text.strip().split()) > 1

    def analyse_phrase(self, text: str) -> dict:
        """
        Analyse a multi-word Khasi phrase token by token.

        Each whitespace-separated token is passed through the full four-phase
        pipeline independently. The overall phrase verdict summarises the results:
            all_valid   — every token resolved successfully
            has_errors  — one or more tokens are phonotactically invalid
            has_unknown — one or more tokens are unknown but phonotactically valid
            empty       — input was blank

        Returns
        -------
        dict with keys:
            input        : str
            is_multi_word: True
            tokens       : list[dict]  — per-token analyse() results
            phrase_verdict: str        — "all_valid" | "has_errors" | "has_unknown"
            phrase_desc  : str         — human-readable summary
            corrections  : list[dict]  — tokens with suggestions (unknown/invalid only)
        """
        text = text.strip()
        if not text:
            return {"error": "Empty input", "is_multi_word": True}

        # Split on whitespace, then strip surrounding punctuation from each token
        # so that "ud," is analysed as "ud" (the comma is not part of the word).
        # We preserve the cleaned form as the canonical token for analysis and
        # display; the raw form (with punctuation) is discarded.
        import re as _re
        # Strip surrounding punctuation but preserve ' (glottal stop) and - (compound)
        _PUNCT_STRIP = _re.compile(r"^[^\w\u00ef\u00f1\u00cf\u00d1'\-]+|[^\w\u00ef\u00f1\u00cf\u00d1'\-]+$", _re.UNICODE)
        tokens_raw = [_PUNCT_STRIP.sub("", tok) for tok in text.split() if _PUNCT_STRIP.sub("", tok)]
        token_results = [self.analyse(t) for t in tokens_raw]

        invalid_tokens  = [r for r in token_results if r["verdict"] == "invalid"]
        unknown_tokens  = [r for r in token_results if r["verdict"] == "unknown"]
        valid_tokens    = [r for r in token_results
                           if r["verdict"] in ("valid", "derived")]

        if invalid_tokens:
            phrase_verdict = "has_errors"
            bad = ", ".join(f"'{r['input']}'" for r in invalid_tokens)
            phrase_desc = f"{len(invalid_tokens)} phonotactically invalid token(s): {bad}"
        elif unknown_tokens:
            phrase_verdict = "has_unknown"
            unk = ", ".join(f"'{r['input']}'" for r in unknown_tokens)
            phrase_desc = (
                f"{len(valid_tokens)}/{len(token_results)} token(s) valid — "
                f"unknown token(s): {unk}"
            )
        else:
            phrase_verdict = "all_valid"
            phrase_desc = (
                f"All {len(token_results)} token(s) morphologically valid"
            )

        corrections = [
            {
                "token":        r["input"],
                "verdict":      r["verdict"],
                "verdict_desc": r["verdict_desc"],
                "suggestions":  r["spell"].get("suggestions", []),
                "method":       r["spell"].get("method", ""),
            }
            for r in token_results
            if r["verdict"] in ("invalid", "unknown")
        ]

        return {
            "input":         text,
            "is_multi_word": True,
            "tokens":        token_results,
            "phrase_verdict":phrase_verdict,
            "phrase_desc":   phrase_desc,
            "corrections":   corrections,
        }

    # ------------------------------------------------------------------
    # A bound prefix written with a space after it
    # ------------------------------------------------------------------

    # Letters a Khasi word may contain, for finding the token after a prefix.
    _WORD_RE = r"[A-Za-zÀ-ÿïñÏÑ'’\-]+"

    def _bound_prefixes(self) -> tuple:
        """Productive prefixes that cannot stand alone as a word.

        Derived from the lexicon rather than listed here: a productive prefix
        whose own entry glosses itself as a prefix ("Prefix used to make
        abstract Nouns", "Prefix of many causative verbs") is bound. That
        yields jing- and pyn-.

        The other four productive prefixes are ALSO ordinary words and must
        not be joined to whatever follows them:

            ïa- / ia-  a preposition, "to, for, concerning"
            sngew-     a verb, "feel, hear"
            nong-      a noun, "a square measure"

        Measured on the lexicon's own phrases, a rule that joined any
        productive prefix would have rewritten 143 of them — 107 beginning
        `sngew` (`sngew tieng`, `sngew pang`) and 20 beginning `ïa`, one of
        which is a whole sentence that would collapse to `ïaki`.
        """
        cached = getattr(self, "_bound_pfx_cache", None)
        if cached is not None:
            return cached
        out = []
        for form, info in (getattr(morphology, "PREFIXES", None) or {}).items():
            if not info.get("productive"):
                continue
            # The gloss must OPEN with "Prefix", not merely mention it. A
            # substring search let `ïa-` in — its entry is a preposition whose
            # long gloss says the word elsewhere — and `ïa ki briew` was then
            # rewritten to `ïaki briew`.
            for row in (self.db.lookup(form) or []):
                gloss = str(row.get("english_gloss") or "").strip().lower()
                if gloss.startswith("prefix"):
                    out.append(form)
                    break
        out = tuple(sorted(set(out), key=len, reverse=True)) or ("jing", "pyn")
        self._bound_pfx_cache = out
        return out

    def _canonical_prefix(self, form: str) -> str:
        """The spelling the lexicon prefers for a prefix recorded twice.

        `ia-` and `ïa-` are one prefix under two keys; the `ïa-` record says
        so — "Spelling corrected 2026-08-26: the lexicon now writes this
        prefix 'ïa-'. The plain key is kept so older data and callers still
        resolve." Rather than hard-code that pair, prefixes sharing a
        `function` are treated as spellings of one morpheme and the one
        carrying a diacritic wins, which is the same ruling the variants
        layer applies to whole words.
        """
        cache = self.__dict__.setdefault("_canon_pfx_cache", {})
        if form in cache:
            return cache[form]
        prefixes = getattr(morphology, "PREFIXES", None) or {}
        info = prefixes.get(form) or {}
        fn = info.get("function")
        best = form
        if fn:
            twins = [k for k, v in prefixes.items() if v.get("function") == fn]
            plain = str.maketrans({"ï": "i", "ñ": "n", "á": "a", "é": "e",
                                   "í": "i", "ó": "o", "ú": "u", "ý": "y"})
            for k in twins:
                if k != form and k.translate(plain) == form.translate(plain) \
                        and k != k.translate(plain):
                    best = k
                    break
        cache[form] = best
        return best

    def _canonical_spelling(self, word: str) -> str:
        """Rebuild *word* with the lexicon's preferred affix spellings.

        `jingialehkai` parses as jing- + ia- + lehkai, and `ia-` is the alias
        of `ïa-`, so the word the lexicon writes is `jingïalehkai`. Returns
        *word* unchanged unless the rebuild is a form the engine accepts.
        """
        try:
            parsed = morphology.parse(word, self.db.lookup)
        except Exception:
            return word
        layers = parsed.get("layers") or {}
        root = layers.get("root")
        if not root or not word.endswith(root):
            return word
        root = str(root)
        chain = [str(layers[k]).strip("-")
                 for k in ("prefix", "prefix2", "prefix3") if layers.get(k)]
        if not chain:
            return word
        # The ROOT gets the same treatment. `pyniabeit` parses as pyn- +
        # iabeit with no second prefix, because `ia-` applies to verbs and
        # `beit` is an adjective — the gate is right to decline. But
        # `iabeit` is only reachable by folding, while `ïabeit` is an entry
        # spelled exactly that way, so that is the spelling to offer.
        canon_root = root
        if root not in self.db._surface_index:
            for entry in self.db._folded_index.get(
                    self.db._fold_key(root), []):
                surface = (entry.get("surface_form") or "").lower()
                if surface and surface in self.db._surface_index \
                        and surface.translate(str.maketrans(
                            {"ï": "i", "ñ": "n"})) == root:
                    canon_root = surface
                    break
        rebuilt = "".join(self._canonical_prefix(p) for p in chain) + canon_root
        if rebuilt == word:
            return word
        # Only accept a rebuild that spells the same word — same letters once
        # the diacritics come off — and that the engine actually recognises.
        plain = str.maketrans({"ï": "i", "ñ": "n"})
        if rebuilt.translate(plain) != word.translate(plain):
            return word
        if self.db.is_known(rebuilt) or self._is_morphologically_valid(rebuilt):
            return rebuilt
        return word

    def _split_prefix_corrections(self, text: str) -> list:
        """Find `jing ialehkai` and offer `jingïalehkai`.

        A bound prefix cannot be a word on its own, so a space after one is a
        typing slip rather than a phrase. Both halves are individually valid,
        which is why nothing flagged it before: `jing` is an entry and so is
        the root.

        Two guards keep this narrow. The joined form must be morphologically
        valid — that alone rejects the dashed records like
        `jing – sngew – pangnud`. And a two-word sequence the lexicon records
        as a PHRASE is left alone: `jing sum` and `jing la` are entries, and
        whether those recorded spellings are themselves the same slip is a
        question about the data, not something to decide inside a
        spellchecker.
        """
        out = []
        for pfx in self._bound_prefixes():
            # The reciprocal may sit BETWEEN the bound prefix and the root,
            # detached from both: `jing ia mareh`. Matching only
            # prefix + one word stopped at `jing ia` and offered `jingïa` —
            # a prefix chain with nothing on it, which is not a word either.
            # The bound prefix is what makes this safe: after `jing`/`pyn`
            # the following `ia` cannot be the preposition, it is the
            # reciprocal, so absorbing it does not touch the 143 lexicon
            # phrases that a general "join any productive prefix" rule would
            # have rewritten. Measured: no lexicon phrase matches
            # `jing|pyn  ia  X` at all.
            for m in re.finditer(
                    rf"\b({pfx})\s+(?:(ïa|ia)\s+)?({self._WORD_RE})",
                    text, re.IGNORECASE):
                head, recip, tail = m.group(1), m.group(2), m.group(3)
                parts = [head.lower()]
                if recip:
                    parts.append(recip.lower())
                parts.append(tail.lower())
                # `ia` before a noun-class clitic is the PREPOSITION and its
                # article — `ia ka`, `ia u` — never the reciprocal prefix and
                # a root. `ka jing ia ka` is the noun `jing` followed by
                # exactly that, and absorbing it produced `jingïaka`: a
                # prefix chain welded to a clitic.
                #
                # The four Lynnong II clitics are the test, not a part of
                # speech: this file's own _PROTECTED_WORDS is built from
                # FREE_MORPHEMES and contains `mareh`, a verb, so it rejects
                # the very case this rule exists for; and `_is_content_word`
                # leans on POS tags that generate.py records as unreliable
                # (`rap` "to help" is tagged `other`).
                if recip and tail.lower() in _NOUN_CLASS_CLITICS:
                    continue
                spaced = " ".join(parts)
                if self.db.is_known(spaced):
                    continue                      # a recorded phrase
                joined = "".join(parts)
                if not self._is_morphologically_valid(joined):
                    continue
                # Land on the spelling the lexicon writes, in one step.
                # `jing ialehkai` joins to `jingialehkai`, which parses as
                # jing- + ia- + lehkai — and `ia-` is the alias key for
                # `ïa-`. Rebuilding from the parse with the canonical affix
                # spellings gives `jingïalehkai` directly, instead of
                # offering the plain join and making the variants layer
                # correct it afterwards.
                best = self._canonical_spelling(joined)
                out.append({
                    "original": m.group(0),
                    "suggestion": best,
                    "start": m.start(),
                    "end": m.end(),
                    "suggestions_top5": [best],
                    "method": "bound_prefix_spaced",
                    "morphologically_valid": False,
                    "phonotactically_valid": True,
                    "gate_reached": 2,
                })
        return out

    def _hyphen_break_corrections(self, text: str) -> list:
        """Rejoin a word broken across a line by a hyphen.

        Scanned and scraped text carries the printer's line-break hyphen into
        the middle of a word: `Kiiyn- nah` for `khynnah`. Split on the space,
        each half is meaningless — `kiiyn` was being "corrected" to `kibym`
        and the text came back `kibym- nah`, worse than it went in. Joined,
        `kiiynnah` suggests `khynnah`.

        Only `word-` IMMEDIATELY followed by whitespace fires. The spaced form
        `word - word` is left alone: it is how this lexicon writes paired
        expressions — 1,068 of them, `bha-dur - bha-dar`, `ka bhah - ka bynta`
        — and joining those would destroy a genuine entry type. The attached
        form appears in only 10 lexicon surfaces, every one of them this same
        damage (`ladei- jingmut`, `nar- puh`, `i- ngap`).

        Both readings are tried, because the hyphen is not always spurious:
        dropping it gives `kiiynnah`, keeping it gives `kiiyn-nah`, and
        whichever the lexicon actually recognises wins.
        """
        out = []
        pattern = rf"\b({self._WORD_RE})-[ \t]+({self._WORD_RE})"
        for m in re.finditer(pattern, text):
            left, right = m.group(1), m.group(2)
            best = None
            for joined in (f"{left}{right}".lower(), f"{left}-{right}".lower()):
                if self.db.is_known(joined):
                    best = joined
                    break
                sugg = (self.spell.suggest(joined, n=3).get("suggestions")
                        or [])
                if sugg and best is None:
                    best = sugg[0]
            if not best or best == m.group(0).lower():
                continue
            out.append({
                "original": m.group(0),
                "suggestion": best,
                "start": m.start(),
                "end": m.end(),
                "suggestions_top5": [best],
                "method": "hyphen_line_break",
                "morphologically_valid": False,
                "phonotactically_valid": True,
                "gate_reached": 2,
            })
        return out

    def check_sentence(self, text: str) -> dict:
        """
        Spell-check an entire sentence or paragraph.

        Tokens that are morphologically valid are silently skipped.
        Only genuinely unknown words surface as correction suggestions.

        Returns
        -------
        dict with keys:
            input       : str
            corrections : list[dict]  — {original, suggestion, start, end,
                                          suggestions_top5, method,
                                          morphologically_valid}
            corrected   : str         — auto-corrected version of input
        """
        corrections = self.spell.check_sentence(text)
        # A bound prefix written with a space after it spans TWO tokens, so it
        # is found on the raw text and then any single-token correction inside
        # its span is dropped — otherwise the two would rewrite the same
        # characters and the offsets would slide.
        joins = self._split_prefix_corrections(text)
        joins += self._hyphen_break_corrections(text)
        if joins:
            spans = [(j["start"], j["end"]) for j in joins]
            corrections = [c for c in corrections
                           if not any(c["start"] < e and c["end"] > st
                                      for st, e in spans)]
            corrections = sorted(corrections + joins, key=lambda c: c["start"])
        corrected   = text
        offset      = 0
        for c in corrections:
            # A correction with no suggestion is a word we know is wrong but
            # cannot fix — `sbngaifi` has an `f` in it. It is reported so the
            # reader sees it, and left exactly as written here, because the
            # alternative is substituting None into their text.
            if not c.get("suggestion"):
                continue
            start = c["start"] + offset
            end   = c["end"]   + offset
            corrected = corrected[:start] + c["suggestion"] + corrected[end:]
            offset   += len(c["suggestion"]) - (c["end"] - c["start"])
        return {
            "input":       text,
            "corrections": corrections,
            "corrected":   corrected,
        }

    # ------------------------------------------------------------------
    # DB management  (delegated to KhasiDB, spell checker kept in sync)
    # ------------------------------------------------------------------

    def add_lexicon_entry(self, entry: dict) -> dict:
        if "entry_id" not in entry or not entry["entry_id"]:
            entry["entry_id"] = self.db.next_entry_id()
        self.db.add_entry(entry)
        self.spell.sync_with_db()
        return entry

    def update_lexicon_entry(self, entry_id: str, updates: dict) -> Optional[dict]:
        updated = self.db.update_entry(entry_id, updates)
        if updated:
            self.spell.sync_with_db()
        return updated

    def delete_lexicon_entry(self, entry_id: str) -> bool:
        ok = self.db.delete_entry(entry_id)
        if ok:
            self.spell.sync_with_db()
        return ok

    def save_db(self, path: Optional[str | Path] = None) -> None:
        self.db.save(path)

    def get_stats(self) -> dict:
        return self.db.stats()

    def get_all_entries(self) -> list[dict]:
        return self.db.all_entries()

    # ------------------------------------------------------------------
    # Verdict computation
    # ------------------------------------------------------------------

    _POSMAP = {"adj": "adjective", "adv": "adverb"}
    # Function-word PoS that no prefix's applies_to ever licenses — a prefix on
    # one of these is an unambiguous over-generation (HARD downgrade). A
    # content-word mismatch (e.g. noun where verb is needed) is only a SOFT flag,
    # because the base's PoS tag may simply be wrong / a sense may be unrecorded.
    _FUNCTION_POS = {"conj", "adp", "pron", "det", "part", "intj"}

    # Phase A Rule 3 — gloss class-consistency. For each discovered_rule
    # in morphology.DISCOVERED_RULES, list expected substrings that must
    # appear in the derived form's gloss when the rule applies. This is a
    # data-mining approximation — the rule's own `description` field is
    # what the linguist wrote, but the engine needs concrete tokens to
    # match. Add to this map when a new discovered_rule lands in JSON.
    _RULE3_GLOSS_TERMS: dict[str, set[str]] = {
        "sngew_experiencer_selection": {
            # Core experiencer markers
            "feel", "sense", "experience", "perceive", "stative",
            "emotion", "feeling", "sentiment", "perception",
            # Common stative-emotion glosses observed in lexicon
            "pleased", "displeased", "delight", "happy", "glad",
            "sad", "sorrow", "regret", "angry", "anger", "afraid",
            "fear", "joy", "joyful", "ashamed", "shame",
            "embarrass", "jealous", "confuse", "confused",
            "doubt", "suspicion", "suspect", "disappoint",
            # Cognitive-experiencer glosses
            "understand", "comprehend", "apprehend", "perceive",
            "realize", "recognize", "believe", "think",
            "fed-up", "fed up",
        },
        "shi_unit_classifier": {
            "one", "unit", "single", "counter", "first",
            "once", "a ", "one ", "single ",
        },
        "mar_distributive": {
            "all", "together", "each", "every", "at once",
            "distribute", "distributive", "collectively", "jointly",
        },
    }

    def _meaning_licensed(self, p2: dict) -> tuple[Optional[bool], str, bool]:
        """Phase A selectional licensing for a single-prefix derivation.

        Returns ``(licensed, note, hard)``:
          • ``(True, "", False)``   — licensed
          • ``(False, note, hard)`` — violation; ``hard`` True only when the base
            is a function word (unambiguous), in which case the caller downgrades
            the verdict. ``hard`` False is a soft flag (annotate, keep verdict).
          • ``(None, "", False)``   — undecidable (no affix data, base not
            attested, or a stacked/infixed/compound form): leave verdict unchanged.
        """
        if not p2.get("pass"):
            return None, "", False
        layers = p2.get("layers") or {}
        if layers.get("prefix2") or layers.get("prefix3") or layers.get("infix"):
            return None, "", False  # stacked / infixed — out of scope
        pfx_dash = layers.get("prefix") or (p2.get("affix_info") or {}).get("prefix")
        if not pfx_dash:
            return None, "", False
        pkey = str(pfx_dash).rstrip("-").lower()
        pinfo = morphology.PREFIXES.get(pkey) or {}
        applies = {a.lower() for a in (pinfo.get("applies_to") or [])}
        if not applies:
            return None, "", False
        root = (layers.get("root") or p2.get("root") or "").lower()
        if not root:
            return None, "", False
        matches = self.db.lookup(root)
        if not matches:
            return None, "", False  # base not attested — don't penalise
        base_pos = set()
        for m in matches:
            gc = (m.get("grammatical_class") or "").lower()
            if gc:
                base_pos.add(self._POSMAP.get(gc, gc))
        if not base_pos:
            return None, "", False
        if base_pos & applies:
            # Rule 2 passes. In strict mode, also run Rule 3 — gloss
            # class-consistency — against any matching discovered_rule
            # for this prefix. Soft-flag (hard=False) when the derived
            # form's gloss lacks the expected semantic-class terms.
            if morphology.strict_phase_a():
                derived_glosses = [
                    (m.get("english_gloss") or "").lower()
                    for m in (p2.get("matches") or [])
                ]
                if derived_glosses:
                    for rule in morphology.DISCOVERED_RULES:
                        rid = (rule.get("id") or "").lower()
                        if not rid.startswith(pkey + "_"):
                            continue
                        terms = self._RULE3_GLOSS_TERMS.get(rid)
                        if not terms:
                            break
                        if not any(t in g for g in derived_glosses for t in terms):
                            return False, (
                                f"{pkey}- rule '{rid}' expects gloss with "
                                f"any of {sorted(terms)[:5]}…; "
                                f"got {derived_glosses[0][:60]!r}"
                            ), False
                        break  # one discovered_rule per prefix
            return True, "", False
        hard = bool(base_pos & self._FUNCTION_POS)
        return False, (
            f"{pkey}- attaches to {'/'.join(sorted(applies))}, "
            f"but '{root}' is {'/'.join(sorted(base_pos))}"
        ), hard

    def _compute_verdict(self, p1, p2, p3, p4, sp, word: str = "") -> tuple[str, str]:
        """Determine the overall analysis verdict from all four phases.

        *word* is the original surface input, used to look up host
        compound/phrasal entries when the gate-2 lexicon match fires.
        """

        # Lexicon match takes priority over Phase 1 (phonotactic) failure.
        # Hyphenated compounds like "jingbym-kyrmen" are valid DB entries but
        # Phase 1 rejects them when it splits on hyphens and checks each segment.
        # The spell checker sets gate_reached=2 when morph_resolved — that is
        # our signal to let Phase 2 decide the verdict instead of Phase 1.
        p1_fail_but_db_match = (
            not p1["pass"]
            and sp.get("gate_reached") == 2
            and p2.get("pass")
        )

        if not p1["pass"] and not p1_fail_but_db_match:
            suggs = sp.get("suggestions", [])
            base_msg = "Phonologically invalid — " + "; ".join(p1["errors"][:2])
            if suggs:
                base_msg += ". Did you mean: " + ", ".join(suggs[:3]) + "?"
            return ("invalid", base_msg)

        if p2["pass"]:
            # Use the cleaned layers root (e.g. "kyrmen" for jingbym-kyrmen)
            # rather than the raw DB root_lemma which may still contain the prefix.
            root   = p2.get("layers", {}).get("root") or p2.get("root")
            status = p2.get("status", "")
            gloss  = ""
            suffix_note = ""
            if p2.get("matches"):
                entry = p2["matches"][0]
                gloss = entry.get("english_gloss", "")
                suffix = entry.get("affix_profile", {}).get("suffix")
                if suffix:
                    suffix_note = f" | suffix: {suffix}"

            # Phase A Rule 4 honor: morphology.parse() set this marker when
            # strict mode is on AND the stored compound_roots contained a
            # function-word component. The entry is still in the lexicon
            # (so verdict="valid"), but we don't claim it as a derivation
            # — verdict_desc says "phrase" so the user sees the right
            # interpretation. The marker is never set in default mode.
            phrase_downgrade = (p2.get("affix_info") or {}).get("phrase_downgrade")
            if phrase_downgrade:
                comps = ", ".join(phrase_downgrade.get("components") or [])
                return (
                    "valid",
                    f"Phrase — stored as compound but contains function word(s) "
                    f"({comps}); treated as a phrase per Phase A Rule 4"
                    + (f" — {gloss}" if gloss else ""),
                )
            if status in ("direct_match", "partial_parse"):
                affix         = p2.get("affix_info", {})
                pfx           = affix.get("prefix")
                pfx2          = p2.get("layers", {}).get("prefix2")
                pfx3          = p2.get("layers", {}).get("prefix3")
                inf           = affix.get("infix")
                compound_roots = affix.get("compound_roots", [])

                # Compound word: surface ALL morphemic components including
                # any additional compound roots beyond prefix + primary root
                # e.g. jing-ang-nud → compound_roots=["jing","ang","nud"]
                if compound_roots and len(compound_roots) > 1:
                    chain = " + ".join(compound_roots)
                    return (
                        "derived",
                        f"Compound form — {chain}"
                        + (f" ({pfx} prefix)" if pfx else "")
                        + (f" — {gloss}" if gloss else "")
                        + suffix_note,
                    )
                # Stacked-prefix form (prefix2 present — jing + pyn + root etc.)
                if pfx and pfx2:
                    # Build the full prefix chain using dash forms from layers
                    # (layers always stores dash form e.g. "jing-", "pyn-")
                    layers_ref = p2.get("layers", {})
                    chain_parts = []
                    for lk in ("prefix", "prefix2", "prefix3"):
                        dash = layers_ref.get(lk)
                        if dash:
                            chain_parts.append(dash.rstrip("-"))
                    chain_parts.append(root)
                    chain_str = " + ".join(chain_parts)
                    warn = " [partial — prefix2 not in DB]" if status == "partial_parse" else ""
                    return (
                        "derived",
                        f"Prefixed form — {chain_str}"
                        + (f" — {gloss}" if gloss else "")
                        + suffix_note
                        + warn,
                    )
                # Single-prefix form — use dash form from layers when available
                if pfx:
                    neg = p2.get("layers", {}).get("negation")
                    middle = f" + {neg}" if neg else ""
                    # Prefer layers["prefix"] (dash form) over affix_info["prefix"] (raw key)
                    pfx_display = p2.get("layers", {}).get("prefix") or pfx
                    return (
                        "derived",
                        f"Prefixed form — {pfx_display.rstrip('-')}{middle} + {root}"
                        + (f" — {gloss}" if gloss else "")
                        + suffix_note,
                    )
                # Infixed form
                if inf:
                    return (
                        "derived",
                        f"Infixed form — {inf} in root {root}"
                        + (f" — {gloss}" if gloss else "")
                        + suffix_note,
                    )
                return (
                    "valid",
                    "Direct lexicon match — root: " + str(root)
                    + (f" — {gloss}" if gloss else "")
                    + suffix_note,
                )
            if status in ("derived_form", "recursive_derivation", "lexicalized_recursive_derivation"):
                affix = p2.get("affix_info", {})
                pfx = p2.get("layers", {}).get("prefix")
                pfx2 = p2.get("layers", {}).get("prefix2")
                pfx3 = p2.get("layers", {}).get("prefix3")
                if pfx and pfx2:
                    chain = affix.get("prefix_chain", "?")
                    depth = affix.get("stack_depth", 0)
                    if status == "lexicalized_recursive_derivation":
                        title = "Lexicalized recursive derivation"
                    else:
                        title = f"Stacked prefixes ({depth})"
                    return (
                        "derived",
                        f"{title} — {chain} + {root}"
                        + (f" — {gloss}" if gloss else ""),
                    )
                if pfx:
                    fn = affix.get("prefix_function_label") or affix.get("prefix_function", "")
                    return (
                        "derived",
                        f"Prefixed form — {pfx.rstrip('-')} + {root}"
                        + (f" ({fn})" if fn else "")
                        + (f" — {gloss}" if gloss else "")
                        + suffix_note,
                    )
                return (
                    "derived",
                    f"Derived form — {root}"
                    + (f" — {gloss}" if gloss else "")
                    + suffix_note,
                )
            if status == "prefix_stripped":
                pfx = p2.get("affix_info", {}).get("prefix", "?")
                fn  = p2.get("affix_info", {}).get("prefix_function", "")
                return (
                    "derived",
                    f"Prefixed form — {pfx} + {root}"
                    + (f" ({fn})" if fn else "")
                    + (f" — {gloss}" if gloss else "")
                    + suffix_note,
                )
            if status == "prefix_stacked":
                chain = p2.get("affix_info", {}).get("prefix_chain", "?")
                depth = p2.get("affix_info", {}).get("stack_depth", 0)
                return (
                    "derived",
                    f"Stacked prefixes ({depth}) — {chain} + {root}"
                    + (f" — {gloss}" if gloss else ""),
                )
            if status == "infix_detected":
                inf = p2.get("affix_info", {}).get("infix", "?")
                fn  = p2.get("affix_info", {}).get("infix_function", "")
                return (
                    "derived",
                    f"Infixed form — infix {inf} in root {root}"
                    + (f" ({fn})" if fn else "")
                    + (f" — {gloss}" if gloss else "")
                    + suffix_note,
                )
            if status == "suffix_stripped":
                sfx = p2.get("affix_info", {}).get("suffix", "?")
                fn  = p2.get("affix_info", {}).get("suffix_function", "")
                return (
                    "derived",
                    f"Suffixed form — {root} + {sfx}"
                    + (f" ({fn})" if fn else "")
                    + (f" — {gloss}" if gloss else "")
                    + suffix_note,
                )
            if status == "prefix_phonotactic":
                pfx = p2.get("affix_info", {}).get("prefix", "?")
                fn  = p2.get("affix_info", {}).get("prefix_function", "")
                return (
                    "derived",
                    f"Prefixed form (productive) — {pfx} + {root}"
                    + (f" ({fn})" if fn else "")
                    + " — root phonotactically valid, not yet in lexicon",
                )
            if status == "prefix_stacked_phonotactic":
                chain = p2.get("affix_info", {}).get("prefix_chain", "?")
                depth = p2.get("affix_info", {}).get("stack_depth", 0)
                # Build per-prefix function labels
                fn_parts = []
                for i in range(depth):
                    key = "prefix" if i == 0 else f"prefix{i+1}"
                    fn = p2.get("affix_info", {}).get(f"{key}_function", "")
                    dash = p2.get("affix_info", {}).get(key, "")
                    if fn: fn_parts.append(f"{dash}{fn}")
                fn_str = "; ".join(fn_parts)
                return (
                    "derived",
                    f"Stacked prefixes ({depth}) — {chain} + {root}"
                    + (f" [{fn_str}]" if fn_str else "")
                    + " — root phonotactically valid, not yet in lexicon",
                )

        if p3.get("rule_detected") and p3.get("root_confirmed", False):
            rule  = p3["rule_detected"]
            pfx   = p3.get("reconstructed_prefix", "")
            rroot = p3.get("reconstructed_root", "")
            desc  = p3.get("rule_description", "")
            return (
                "derived",
                f"Assimilation form — rule: {rule}"
                + (f" ({desc})" if desc else "")
                + (f" — underlying: {pfx} + {rroot}" if pfx or rroot else ""),
            )

        if p4.get("detected"):
            p4_type    = p4.get("type", "")
            p4_desc    = p4.get("description", p4_type.replace("_", " "))
            details    = p4.get("details", {})
            components = p4.get("components", [])

            if p4_type == "shim_kylliang":
                src   = details.get("source_language", "unknown")
                itype = details.get("integration_type", "direct")
                gloss = details.get("gloss", "")
                return (
                    "valid",
                    f"Borrowed word (shim kylliang) — source: {src}, integration: {itype}"
                    + (f" — {gloss}" if gloss else ""),
                )

            if p4_type in ("tien_kynnoh", "tien_kynnoh_pdeng", "tien_kynnoh_non_lexical"):
                subtype  = details.get("tien_kynnoh_subtype", "")
                kynnoh   = details.get("kynnoh_constituent", "")
                primary  = details.get("primary_constituent", "")
                gloss    = details.get("gloss", "")
                subtype_str = f" [{subtype}]" if subtype else ""
                pair_str    = (
                    f" — {primary} / {kynnoh}" if primary and kynnoh
                    else f" — {' + '.join(components)}"
                )
                return (
                    "valid",
                    f"Tien Kynnoh{subtype_str}{pair_str}"
                    + (f" — {gloss}" if gloss else ""),
                )

            return (
                "derived",
                f"Complex form — {p4_desc}: {' + '.join(components)}",
            )

        # Spell gate 2 = morphologically valid — could be a protected grammatical
        # particle, a compound-token (word found inside a multi-word DB entry),
        # or a morph parse resolved outside Phase 2/3/4.
        if sp.get("gate_reached") == 2:
            in_lex = sp.get("in_lexicon", False)
            if in_lex:
                # If the word appears as a token inside multi-word entries,
                # surface those phrases so the user sees the actual lexicon
                # entry — not just an abstract "found in a compound" note.
                # Multiple host entries are joined by ", " (up to 3 shown).
                hosts: list[str] = []
                try:
                    if word:
                        hosts = self.db.find_compound_entries(word.lower())
                except Exception:
                    hosts = []
                if hosts:
                    sample = ", ".join(f'"{h}"' for h in hosts[:3])
                    more = f" (+{len(hosts) - 3} more)" if len(hosts) > 3 else ""
                    return (
                        "valid",
                        "Lexicon match — found as component of compound/phrasal entry: "
                        + sample + more,
                    )
                return (
                    "valid",
                    "Lexicon match — found as component of a compound/phrasal entry",
                )
            return (
                "valid",
                "Grammatical particle / protected word — phonotactically valid",
            )

        # Nothing found — word is genuinely unknown
        suggs = sp.get("suggestions", [])
        if suggs:
            return (
                "unknown",
                "Not in lexicon — phonotactically valid. Did you mean: "
                + ", ".join(suggs[:3]) + "?",
            )
        return (
            "unknown",
            "Not in current lexicon — phonotactically valid but no parse found. "
            "Consider adding to DB.",
        )

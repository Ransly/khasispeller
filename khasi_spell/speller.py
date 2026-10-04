"""
speller.py — the public API of the standalone Khasi spellchecker.

This is a thin facade over `khasi_engine`. It exists so that callers have
one stable surface to program against, rather than reaching into the
engine's internals (whose shapes are inherited from the parent platform
and are not guaranteed stable).

Typical use
-----------
    from khasi_spell import KhasiSpeller

    sp = KhasiSpeller()                       # loads the lexicon once
    sp.is_correct("jingbha")                  # True  — morphology gate
    sp.suggest("lyngdo")                      # ['lyngdoh', ...]
    sp.check_text("Ka shnng ka bha")          # per-token findings
    sp.correct_text("Ka shnng ka bha")        # corrected string

Construction cost
-----------------
The lexicon is a 64 MB JSON file (or a PostgreSQL database) and takes
roughly 15-25 seconds to load, so build one `KhasiSpeller` and keep it.
Loading is deferred until the first call, so importing this module is cheap.
"""

from __future__ import annotations

import threading
from collections import OrderedDict
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

from khasi_engine import tokens as _tokens


def _nfc(text: str) -> str:
    """
    Normalise to composed form.

    Khasi's ï and ñ each have two Unicode spellings: precomposed (U+00EF,
    U+00F1) and decomposed (i/n plus a combining mark). The lexicon is
    precomposed, and the tokeniser's character class lists only the
    precomposed forms — so decomposed input splits mid-word. 'jingïathuh'
    becomes 'jing', 'i', 'athuh', and the checker then "corrects" the
    fragment, corrupting text that was right to begin with.

    macOS and several compose-key configurations emit decomposed text, so
    this is not hypothetical. Normalising at the public boundary is the
    cheapest place to fix it: everything downstream then sees one spelling.

    Note that offsets in the returned results index the NORMALISED string,
    which is why check_text() returns that string as `text`. Only NFC is
    applied to the text; the typographic apostrophe is tolerated at lookup
    time instead, so the writer's own quotes survive in `corrected`.
    """
    return _tokens.nfc(text)


# The four gate stages the engine can stop at. Exposed because the stage
# is genuinely informative: it tells you *why* a word was or was not
# flagged, which a bare boolean cannot.
GATE_INVALID_STANDALONE = 0   # sub-phonemic token, e.g. a bare digraph
GATE_PHONOTACTIC        = 1   # violates Khasi phonotactics
GATE_ACCEPTED           = 2   # accepted by the confidence vote
GATE_CHECKED            = 3   # unknown; candidates were generated

_GATE_NAMES = {
    0: "invalid_standalone",
    1: "phonotactically_invalid",
    2: "accepted",
    3: "unknown",
}


@dataclass
class WordResult:
    """Outcome of checking a single word."""

    word: str
    is_correct: bool
    suggestions: list[str] = field(default_factory=list)
    # Blended phoneme/Damerau distance for each suggestion, same order. This
    # is a distance, not a probability: 0 is identical and each whole unit is
    # roughly one edit, with confusable Khasi phonemes costing less than one.
    # Callers that want a "match strength" must say how they derived it.
    distances: list[float] = field(default_factory=list)
    # English gloss per suggestion, same order, "" where the lexicon has none.
    # A candidate can be offered without being a headword — the delete index
    # also draws on root and compound-token indices, and the morphological
    # generator builds forms the lexicon never recorded — so an empty string
    # here is normal and must be rendered as nothing, not as a placeholder.
    glosses: list[str] = field(default_factory=list)
    gate: int = GATE_CHECKED
    gate_name: str = "unknown"
    method: str = ""
    in_lexicon: bool = False
    morphologically_valid: bool = False
    phonotactically_valid: bool = False
    confidence: Optional[dict] = None
    # Alternative diacritic spellings the lexicon attests. Populated even
    # when the word is accepted — writing 'iathuh' is not an error, but the
    # writer may want to know the lexicon spells it 'ïathuh'.
    variants: list[dict] = field(default_factory=list)
    # English gloss of the word ITSELF, "" when the lexicon records none.
    # `glosses` above describes the suggestions; an accepted word has no
    # suggestions, so without this the detail panel could say a word was
    # accepted while showing nothing about what it means.
    gloss: str = ""

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Correction:
    """One flagged token within a longer text."""

    original: str
    suggestion: str
    start: int
    end: int
    suggestions: list[str] = field(default_factory=list)
    distances: list[float] = field(default_factory=list)
    # English gloss per suggestion, same order, "" where the lexicon has none.
    # A candidate can be offered without being a headword — the delete index
    # also draws on root and compound-token indices, and the morphological
    # generator builds forms the lexicon never recorded — so an empty string
    # here is normal and must be rendered as nothing, not as a placeholder.
    glosses: list[str] = field(default_factory=list)
    gate: int = GATE_CHECKED
    method: str = ""
    # Why the word was flagged, in the phonology module's own words —
    # "Illegal characters: f — not in Khasi orthography" — or the checker's:
    # "'hospital' is an English word, not a Khasi word", or a misspelled
    # part of a compound. Carried to the UI so a word with no offerable
    # correction can still explain itself. None otherwise.
    reason: Optional[str] = None
    # A capitalised word checked although it may be a name — see
    # KhasiSpeller._capitalised_typo. Shown to the reader as a possible
    # misspelling, and never applied automatically.
    possible_misspelling: bool = False

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class VariantFlag:
    """An accepted word that has another attested spelling."""

    word: str
    variants: list[dict]
    start: int
    end: int

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class TextResult:
    """Outcome of checking a span of text."""

    text: str
    corrected: str
    corrections: list[Correction] = field(default_factory=list)
    # Accepted words with an alternative spelling. NOT errors — kept apart
    # from `corrections` so nothing here is ever auto-applied.
    variants: list[VariantFlag] = field(default_factory=list)
    # Tokens that failed the gate but were withheld as proper nouns or
    # acronyms. Reported rather than deleted, so a caller can surface them.
    skipped: list[dict] = field(default_factory=list)

    @property
    def has_errors(self) -> bool:
        return bool(self.corrections)

    def to_dict(self) -> dict:
        return {
            "text": self.text,
            "corrected": self.corrected,
            "corrections": [c.to_dict() for c in self.corrections],
            "variants": [v.to_dict() for v in self.variants],
            "skipped": list(self.skipped),
        }


class KhasiSpeller:
    """
    Morphology-gated spellchecker for Khasi.

    Parameters
    ----------
    db_path :
        Path to the lexicon JSON. Defaults to ``data/khasi_db.json`` in the
        project (or in ``KHASI_DATA_DIR`` when set). Ignored for loading when
        ``DATABASE_URL`` is set: the lexicon then comes from PostgreSQL.
    use_embeddings :
        Enable FastText semantic re-ranking. Uses the bundled local model
        (needs gensim) or a remote Space via ``KHASI_W2V_URL``; falls back
        to rule-only ranking if neither is available.

        Default off, and deliberately so: on this project's synthetic probe
        the re-ranker LOWERED top-1 accuracy from 53.2% to 49.4%. FastText
        composes vectors from character n-grams, so orthographic neighbours
        score high whether or not they are the right correction — it partly
        re-measures edit distance while adding noise. Turn it on to
        experiment, not because it is expected to help.
    use_corpus_freq :
        Replace the engine's structural pseudo-frequencies with observed
        corpus counts (``data/corpus_freq.json``). On by default when that
        file is present; silently skipped when it is not.

        The built-in values are not really frequencies — they are assigned
        from a word's role in the lexicon and take only 11 distinct values,
        so they cannot tell a common word from a rare one. Substituting real
        counts improved detection, top-1 and top-5 by 1.3 points each.
    use_corpus_pool :
        Admit frequent corpus types to the *candidate* pool, so a word the
        lexicon does not record can still be offered as a correction. On by
        default. See ``khasi_spell.corpus_pool``.

        Distinct from ``use_corpus_freq``, which changes what known words
        score. This changes which unknown ones are reachable at all:
        `pyntreikam` occurs 1,940 times, is not a headword, and without this
        no typo of it is correctable, because it is not in the list to be
        ranked.
    corpus_pool_accept :
        Also let the admitted corpus words pass the confidence vote, by
        crediting them with corpus frequency. On by default. Without it the
        checker offered words it then rejected — `jyla` was answered with
        `jylla`, which was flagged once accepted. ``is_known()`` is untouched
        either way. Ignored when ``use_corpus_pool`` is off.

        Every one of these switches is per speller. The lexicon object is
        shared by all spellers in a process, so corpus frequencies, the corpus
        pool and taught words live on each speller's own checker and are never
        written into the lexicon.
    corpus_pool_floor :
        Occurrences a corpus type needs before it may be offered. Lower
        admits more vocabulary and more noise.
    eager :
        Load the lexicon during construction instead of on first use.
    """

    def __init__(
        self,
        db_path: Optional[str | Path] = None,
        use_embeddings: bool = False,
        use_corpus_freq: bool = True,
        use_corpus_pool: bool = True,
        corpus_pool_floor: Optional[int] = None,
        corpus_pool_accept: bool = True,
        eager: bool = False,
    ) -> None:
        from khasi_spell import corpus_pool as _cp
        self._db_path = Path(db_path) if db_path else None
        self._use_embeddings = use_embeddings
        self._use_corpus_freq = use_corpus_freq
        self._use_corpus_pool = use_corpus_pool
        self._corpus_pool_accept = corpus_pool_accept
        self._corpus_pool_floor = (_cp.DEFAULT_FLOOR if corpus_pool_floor is None
                                   else corpus_pool_floor)
        self._corpus_pool_summary: Optional[dict] = None
        self._corpus_freq_summary: Optional[dict] = None
        self._corpus_count_table: Optional[dict] = None   # see _corpus_counts
        self._realword: Any = None
        self._ngram_lm: Any = None
        self._lm_lock = threading.Lock()
        # Bounded LRU. A plain dict grew with every distinct token a
        # long-running service was ever sent.
        self._variant_cache: "OrderedDict[str, list]" = OrderedDict()
        self._analyser: Any = None
        self._lm_warned = False
        if eager:
            self._load()

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def _load(self) -> Any:
        if self._analyser is None:
            from khasi_engine.analyser import KhasiAnalyser

            kwargs: dict[str, Any] = {}
            if self._db_path is not None:
                kwargs["db_path"] = str(self._db_path)
            self._analyser = KhasiAnalyser(**kwargs)
            if self._use_embeddings:
                # FastText re-ranking, local or remote. Never fatal:
                # w2v_client raises W2vNotConfigured / W2vUpstreamError,
                # both of which the checker catches and falls back from.
                self._analyser.spell.set_remote_w2v_enabled(True)
            if self._use_corpus_freq:
                from khasi_spell import corpus_freq
                try:
                    self._corpus_freq_summary = corpus_freq.apply(self._analyser.spell)
                except FileNotFoundError:
                    # No corpus list built yet — keep the engine's own values.
                    self._corpus_freq_summary = None
            if self._use_corpus_pool:
                # Candidate SUPPLY, separate from the frequency values above:
                # corpus_freq updates what known words score, this decides
                # which unknown ones may be offered at all. Never fatal.
                from khasi_spell import corpus_pool
                try:
                    self._corpus_pool_summary = corpus_pool.apply(
                        self._analyser.spell, floor=self._corpus_pool_floor,
                        accept=self._corpus_pool_accept,
                    )
                except FileNotFoundError:
                    self._corpus_pool_summary = None
        return self._analyser

    def preload_embeddings(self) -> dict:
        """
        Load the FastText model now rather than on the first correction.

        Scoring is lazy by default, which is right for a library — you only
        pay the ~12 s and ~850 MB if something actually needs re-ranking.
        It is wrong for a server, where that cost lands on whichever user
        happens to send the first request. Long-running processes should
        call this during start-up.

        Returns the backend info dict. Never raises: if no model or gensim
        is available it reports that and the checker stays rule-only.
        """
        from khasi_engine import w2v_client
        if not self._use_embeddings:
            return {"enabled": False, "reason": "use_embeddings=False"}
        try:
            w2v_client.load_local()
        except (w2v_client.W2vNotConfigured, w2v_client.W2vUpstreamError) as exc:
            return {**w2v_client.info(), "enabled": False, "error": str(exc)}
        return self.embeddings

    @property
    def loaded(self) -> bool:
        """True once the lexicon has been read into memory."""
        return self._analyser is not None

    @property
    def analyser(self):
        """
        The underlying `KhasiAnalyser`.

        Escape hatch for callers who need engine internals — morphological
        parses, phonological detail, the raw lexicon. Its response shapes
        are inherited from the parent platform and are not part of this
        package's stable API.
        """
        return self._load()

    @property
    def embeddings(self) -> dict:
        """
        State of the semantic re-ranker: which backend, whether loaded.

        Off by default. Measured on this project's synthetic probe it costs
        3.8 points of top-1 accuracy (53.2% -> 49.4%), so it is opt-in
        rather than automatic. See README "Semantic re-ranking".
        """
        from khasi_engine import w2v_client
        d = w2v_client.info()
        d["enabled"] = bool(self._analyser.spell._hybrid) if self._analyser else False
        return d

    @property
    def corpus_pool(self) -> Optional[dict]:
        """Summary of the corpus-derived candidate pool, or None if not applied.

        Provenance: `admitted` forms are offerable because people write
        them, not because the dictionary records them. Anything showing a
        suggestion to a reader should be able to say which of the two it
        is — `is_corpus_form()` answers that per word.
        """
        self._load()
        return self._corpus_pool_summary

    def is_corpus_form(self, word: str) -> bool:
        """True when *word* is offerable only because the corpus attests it."""
        return self._load().spell.is_corpus_form(word)

    @property
    def corpus_frequencies(self) -> Optional[dict]:
        """Summary of the corpus-frequency substitution, or None if not applied."""
        self._load()
        return self._corpus_freq_summary

    def _lm(self):
        """
        The trigram model, or None when none is available.

        One instance per speller, shared by context re-ranking and real-word
        detection. The detector used to build its own copy, so a service that
        ran both held two ~170 MB models. Loaded once, under a lock, because
        the HTTP service calls this from several request threads.
        """
        if self._ngram_lm is None:
            with self._lm_lock:
                if self._ngram_lm is None:
                    self._ngram_lm = self._build_lm()
        return self._ngram_lm or None

    def _build_lm(self):
        """Pick a backend: PostgreSQL when it has the counts, else the file.

        The file model costs 181 MB resident for 5.1 MB of gzip — 838,445
        counts expanded into Python dicts, nearly all of it object overhead.
        Served from PostgreSQL the counts stay in the database and the same
        arithmetic runs over whatever a slot actually needs.

        PG is chosen only when `ngram_counts` exists AND has rows. An empty
        or missing table falls back to the file rather than scoring every
        candidate at the floor, which would silently disable context
        re-ranking instead of failing.

        Override with KHASI_NGRAM_BACKEND=file|pg.
        """
        import os

        want = os.environ.get("KHASI_NGRAM_BACKEND", "auto").strip().lower()
        url = os.environ.get("DATABASE_URL", "").strip()

        if want in ("pg", "postgres", "postgresql") or (want == "auto" and url):
            from khasi_spell.ngram_pg import PgNgramLM
            if not url:
                print("[khasi-spell] KHASI_NGRAM_BACKEND asked for PostgreSQL "
                      "but DATABASE_URL is not set; using the file model")
            elif PgNgramLM.table_exists(url):
                try:
                    return PgNgramLM(url).load()
                except Exception as exc:
                    print(f"[khasi-spell] PG n-gram backend unavailable "
                          f"({exc}); using the file model")
            elif want != "auto":
                print("[khasi-spell] KHASI_NGRAM_BACKEND asked for PostgreSQL "
                      "but ngram_counts is absent or empty; using the file "
                      "model. Run scripts/migrate_ngrams_to_pg.py")

        from khasi_spell.ngram import NgramLM
        try:
            return NgramLM().load()
        except FileNotFoundError:
            return False   # absent; do not retry every call

    @property
    def language_model(self) -> dict:
        """
        State of the trigram model, shared by real-word detection and
        context re-ranking.

        Reports availability without loading it: the model is read on the
        first `check_text()` (context re-ranking) or `check_realword()`.
        """
        import os
        from khasi_spell.ngram import NgramLM
        lm = self._ngram_lm or None
        if lm is not None:
            return {"available": True, "loaded": True, **lm.info()}
        url = os.environ.get("DATABASE_URL", "").strip()
        want = os.environ.get("KHASI_NGRAM_BACKEND", "auto").strip().lower()
        if url and want != "file":
            from khasi_spell.ngram_pg import PgNgramLM
            if PgNgramLM.table_exists(url):
                return {"available": True, "loaded": False,
                        "path": "postgresql:ngram_counts", "backend": "postgresql"}
        probe = NgramLM()
        return {"available": probe._path.is_file(), "loaded": False,
                "path": str(probe._path), "backend": "file"}

    @property
    def vocabulary_size(self) -> int:
        """
        Size of the frequency table the ranker consults.

        NOT a count of Khasi words, despite the name — 17,803 of its 32,976
        keys are multi-word phrases ("'bai iohnong", "bai kliarwait"), so it
        overstates the vocabulary by about 2.5x. It was being shown in the
        interface as "Khasi word forms", which was wrong. Kept under this name
        because the parent platform's clients read it; use `word_forms` for
        anything user-facing. Cheap: the table is cached on the lexicon.
        """
        return len(self._load().db.to_freq_dict())

    @property
    def word_forms(self) -> int:
        """
        Distinct single-token forms the checker can actually offer.

        The lexicon's `all_surface_forms()` — headwords, root lemmas and the
        tokens harvested from multi-word entries — plus the corpus words this
        speller admitted and any words taught with `add_word()`. Corrections
        are drawn from these, and from forms the morphological generator
        builds on demand from them.
        """
        return self._load().spell.candidate_form_count()

    # ------------------------------------------------------------------
    # Word level
    # ------------------------------------------------------------------

    _GLOSS_SENSES = 3
    _GLOSS_CHARS = 72

    def _glosses(self, words) -> list[str]:
        """
        English gloss for each word, "" when the lexicon records none.

        Trimmed for display: entries carry every sense the printed dictionary
        listed, and `dang` alone runs to eleven ("Just; Meantime; Still; Yet;
        Constipated; ..."), which is a paragraph beside a one-line suggestion.
        The first few senses are the useful part; callers wanting the whole
        thing can look the entry up.
        """
        db = self._load().db
        out: list[str] = []
        for w in words:
            gloss = ""
            try:
                hits = db.lookup(w) or []
                raw = (hits[0].get("english_gloss") or "").strip() if hits else ""
                if raw:
                    senses = [s.strip() for s in raw.split(";") if s.strip()]
                    gloss = "; ".join(senses[: self._GLOSS_SENSES])
                    if len(gloss) > self._GLOSS_CHARS:
                        gloss = gloss[: self._GLOSS_CHARS].rstrip(" ;,") + "…"
            except Exception:
                gloss = ""
            out.append(gloss)
        return out

    def check(self, word: str, n: int = 5) -> WordResult:
        """Check one word and return the full outcome, including why.

        *n* is the number of suggestions. The HTTP service used to call this
        and then `suggest()` again for any other n, running the search twice
        and re-aligning three parallel lists by hand.
        """
        word = _nfc(word)
        raw = self._load().spell.suggest(word, n=n)
        if raw.get("method") == "doubled_prefix" and raw.get("suggestions"):
            raw = dict(raw)
            # Lower case, as a single word's suggestions always are.
            raw["suggestions"], raw["suggestion_distances"] = self._standard_first(
                word.lower(), raw["suggestions"], raw.get("suggestion_distances") or [])
        gate = raw.get("gate_reached", GATE_CHECKED)
        return WordResult(
            word=word,
            is_correct=bool(raw.get("is_known")),
            suggestions=list(raw.get("suggestions") or []),
            distances=list(raw.get("suggestion_distances") or []),
            glosses=self._glosses(raw.get("suggestions") or []),
            gate=gate,
            gate_name=_GATE_NAMES.get(gate, "unknown"),
            method=raw.get("method", ""),
            in_lexicon=bool(raw.get("in_lexicon")),
            morphologically_valid=bool(raw.get("morphologically_valid")),
            phonotactically_valid=bool(raw.get("phonotactically_valid")),
            confidence=raw.get("confidence"),
            # A rejected word keeps only the alternatives the checker accepts
            # (`saitjain` -> `saitjaiñ`), never one as wrong as itself
            # (`jingjingiamareh` -> `jingjingïamareh`).
            variants=[v.to_dict() for v in self._word_variants(word)
                      if raw.get("is_known")
                      or self._load().spell.suggest(v.variant, n=1).get("is_known")],
            gloss=self._glosses([word])[0],
        )

    def _word_variants(self, word: str) -> list:
        """
        Diacritic alternatives for a single word checked on its own.

        A lone word carries no sentence position, so a leading capital can
        only mean a name: `Spain` the country rather than `spain` 'bandage',
        whose standard spelling is `spaiñ`. See `foreign.is_proper_case`.
        """
        from khasi_spell import foreign

        token = _nfc(word).strip()
        if foreign.is_proper_case(token, sentence_initial=False):
            return []
        return self.variants(token)

    def variants(self, word: str) -> list:
        """
        Alternative diacritic spellings of *word* attested by the lexicon.

        Khasi's ï and ñ are awkward to type and routinely dropped. The
        engine already accepts the plain form, so this is guidance rather
        than correction: `iathuh` is valid, and the lexicon also records
        `ïathuh`.

        Returns a list of `SpellingVariant`, empty when the word already
        carries its diacritics or nothing is attested.
        """
        from khasi_spell import variants as _variants
        key = _tokens.canonical(word).lower()
        hit = self._variant_cache.get(key)
        if hit is None:
            hit = _variants.find(word, self._load())
            self._variant_cache[key] = hit
            while len(self._variant_cache) > 8192:
                self._variant_cache.popitem(last=False)
        else:
            self._variant_cache.move_to_end(key)
        return hit

    def is_correct(self, word: str) -> bool:
        """True if the word is accepted as valid Khasi."""
        return self.check(word).is_correct

    def suggest(self, word: str, n: int = 5) -> list[str]:
        """
        Ranked corrections for a word, best first.

        Returns an empty list when the word is already correct, and also
        when it is so malformed that no candidate could be generated.
        Use `check()` if you need to tell those two cases apart.
        """
        return list(self._load().spell.suggest(_nfc(word), n=n).get("suggestions") or [])

    def correct(self, word: str) -> str:
        """The single best correction, in the word's own capitalisation, or
        the word unchanged if it is correct."""
        return self._load().spell.autocorrect(_nfc(word))

    # ------------------------------------------------------------------
    # Text level
    # ------------------------------------------------------------------

    # Tokens exactly as the engine sees them (khasi_engine.tokens): every
    # component tokenises with this one pattern, so a correction's offsets
    # and a variant's offsets always refer to the same word.
    _TOKEN_RE = _tokens.WORD_PATTERN

    # _check_hyphenated() was removed 2026-09-30. It verified the parts of a
    # hyphenated compound here, in the text path only, so `check("man-miay")`
    # accepted what `check_text()` flagged. The rule now lives in the engine
    # (KhasiSpellChecker._hyphenated_part_veto), where every entry point
    # passes. Tokenising here with a narrower pattern than the engine's also
    # let it correct a fragment inside a word the engine had already flagged:
    # `i^p-shnong` came back as `i^pa-shnong`.

    @staticmethod
    def _apply(text: str, corrections) -> str:
        """*text* with each correction's top suggestion spliced in.

        Right to left so earlier offsets stay valid, and never two edits to
        the same characters: an overlap would slide the offsets and corrupt
        the text. A correction with no suggestion (`sbngaifi` has an `f`) is
        reported to the reader and left exactly as written.
        """
        out = text
        floor = None
        for c in sorted(corrections, key=lambda c: -c.start):
            # A possible misspelling may be a name: it is the reader's call.
            if not c.suggestion or c.possible_misspelling:
                continue
            if floor is not None and c.end > floor:
                continue
            out = out[:c.start] + c.suggestion + out[c.end:]
            floor = c.start
        return out

    def _variant_candidate(self, token: str) -> bool:
        """
        Cheap test for whether a token could have a diacritic variant.

        variants() parses the word to reach the root route, and a parse per
        token cost 15 s on one sentence. These are all hash lookups: a token
        that already carries a diacritic, one whose folded key the lexicon
        indexes, or a longer unlisted form that might be derived. Everything
        else cannot produce a variant and is skipped without parsing.
        """
        from khasi_spell import variants as _v
        db = self._load().db
        w = _tokens.canonical(token).lower()
        if any(ch in _v.DIACRITIC_OF.values() for ch in w):
            return True
        if db._folded_index.get(db._fold_key(w)):
            return True
        # A spelling the dictionary records in its own gloss — `k'ing` and
        # `kyieng`, `'nong` and `shnong`. Nothing in the tests below can
        # see these; the link lives in the gloss text. See variants.py
        # route 7.
        if w in _v._spelling_links():
            return True

        # A short spelling whose ï-form is recorded under an i-form root
        # (`ing` -> `ïing`). Invisible to the folded-index test above,
        # because the diaeresis folds to `iing` and never to `ing`, and to
        # the length test below, since these words are short lexicon
        # headwords. See variants.py route 6.
        if _v._short_i_form(w, db) is not None:
            return True

        # An -ain headword the lexicon records only in the plain spelling
        # (`spain` -> `spaiñ`). variants.py route 5 supplies these by rule,
        # and they are lexicon entries, so the length test below skips them.
        # -ain words: a headword the lexicon records only in the plain
        # spelling (`spain` -> `spaiñ`), or a compound whose final element
        # carries the tilde (`saitjain` -> `saitjaiñ`). Both are supplied by
        # rule in variants.py, so neither is caught by the tests above.
        if w.endswith(_v.AIN_PLAIN):
            return True
        # Possibly a derived form whose ROOT carries the diacritic
        # (jingiathuh -> jingïathuh). Only unlisted words need the parse.
        return len(w) >= 6 and w not in db._surface_index

    # A capitalised word mid-sentence, or an all-caps one, is withheld as a
    # presumed name — and so was every capitalised misspelling: `ka Shnongg
    # ba khraw` passed. It is checked after all when it looks like a typo
    # rather than a name: a Khasi word within one change, and the word itself
    # never seen in the corpus, where real names are common (meghalaya 9,651,
    # conrad 2,126). All-caps words shorter than 5 letters stay withheld as
    # acronyms (ITES, ZSU, BAC). Measured 2026-10-04: catches 186 of 235
    # capitalised benchmark errors, 171 with the right word first (none were
    # caught before), and adds 63 flags in 1,000 news lines (0.08% of
    # tokens), every one a rare name — hence "possible", and never applied
    # automatically.
    _CAPS_MAX_DISTANCE = 1.0
    _CAPS_MIN_ALLCAPS = 5

    def _capitalised_typo(self, c: "Correction", why: Optional[str]) -> bool:
        from khasi_spell import foreign
        if why not in (foreign.CAPITALISED, foreign.ACRONYM):
            return False                      # a known name stays withheld
        if why == foreign.ACRONYM and len(c.original) < self._CAPS_MIN_ALLCAPS:
            return False
        if not c.suggestions or not c.distances or c.distances[0] > self._CAPS_MAX_DISTANCE:
            return False
        counts = self._corpus_counts()
        if not counts:
            return False      # without the corpus every rare name would look like a typo
        return counts.get(_tokens.corpus_form(c.original).lower(), 0) == 0

    def _corpus_counts(self) -> dict:
        """Raw corpus counts (data/corpus_freq.json), loaded once; {} if absent.

        The engine's copy (spell_checker.corpus_counts), so the file is held
        in memory once.
        """
        if self._corpus_count_table is None:
            from khasi_engine.spell_checker import corpus_counts
            self._corpus_count_table = corpus_counts()
        return self._corpus_count_table

    def _standard_first(self, typed: str, suggestions: list, distances: list) -> tuple:
        """*suggestions* with the first one's standard spelling put before it.

        A doubled prefix is answered with the word written once, built from
        the letters typed: `jingjingiamareh` -> `jingiamareh`, which the page
        would then offer "also written jingïamareh". In formal Khasi the
        diacritic spelling is the correct one, so it is offered first, with
        the plain form kept second. Only a spelling the checker accepts.
        """
        first = suggestions[0]
        spell = self._load().spell
        for v in self.variants(first):
            if v.variant != first and spell.suggest(v.variant, n=1).get("is_known"):
                std = _tokens.match_case(typed, v.variant)
                rest = [x for x in suggestions if x != std]
                dist = [1.0] + list(distances)
                return [std] + rest, dist[:len(rest) + 1]
        return list(suggestions), list(distances)

    def _settle_rejected_variants(self, flags: list) -> tuple:
        """Split variant offers into true offers and corrections.

        An offer means "this word is correct, and is also written so". A word
        the engine rejected with nothing to suggest is not reported (see
        check_sentence), so it reached the variant pass looking correct and
        was offered a spelling as if it were a word: `jingjingiamareh` showed
        as correct, "also written jingjingïamareh". Now a rejected word keeps
        only the spellings the checker accepts, and those become its
        correction (`suhjain` -> `suhjaiñ`); with none left it gets no offer.
        """
        spell = self._load().spell
        offers, fixes = [], []
        for f in flags:
            if spell.suggest(f.word, n=1).get("is_known"):
                offers.append(f)
                continue
            good = [v["variant"] for v in f.variants
                    if spell.suggest(v["variant"], n=1).get("is_known")]
            if not good:
                continue
            good = [_tokens.match_case(f.word, v) for v in good]
            fixes.append(Correction(
                original=f.word, suggestion=good[0], start=f.start, end=f.end,
                suggestions=good, distances=[1.0] * len(good),
                glosses=self._glosses(good), gate=GATE_CHECKED,
                method="variant_spelling",
                reason=f"Not accepted as written; the dictionary spells it {good[0]}.",
            ))
        return offers, fixes

    def _collect_variants(self, text: str, flagged: set) -> list["VariantFlag"]:
        """
        Accepted words that have another attested spelling.

        Proper nouns are excluded. The diacritic rewrite is a claim about
        Khasi orthography and says nothing about a name that merely looks
        like a Khasi word: `Spain` is a country, but `spain` is a Khasi word
        glossed "bandage; to swathe", and only the second should be offered
        `spaiñ`. Capitalisation is the discriminator, and the corpus shows it
        is a reliable one — `Hussain` is capitalised in 53 of 53 uses and
        `Gohain` in 32 of 32, while `spain` splits 34 capitalised (the
        country) against 8 lowercase (the Khasi word).
        """
        from khasi_spell import foreign

        # The characters inside a flagged span, so each token's test reads its
        # own characters rather than every flag: tested pairwise, a text with
        # thousands of flags cost 36 million comparisons (16 s).
        covered = bytearray(len(text))
        for s, e in flagged:
            s, e = max(s, 0), min(e, len(text))
            if e > s:
                covered[s:e] = b"\x01" * (e - s)

        out: list[VariantFlag] = []
        for m in self._TOKEN_RE.finditer(text):
            token = m.group(0)
            if len(token) < 2 or covered.find(1, m.start(), m.end()) != -1:
                continue
            if foreign.is_proper_case(
                    token, foreign.is_sentence_initial(text, m.start())):
                continue          # a name, not a spelling choice
            if not self._variant_candidate(token):
                continue
            vs = self.variants(token)
            if vs:
                out.append(VariantFlag(
                    word=token, variants=[v.to_dict() for v in vs],
                    start=m.start(), end=m.end(),
                ))
        return out

    def check_text(self, text: str, variants: bool = True,
                   context: bool = True,
                   skip_foreign: bool = True) -> TextResult:
        """
        Check a sentence or paragraph.

        Tokens accepted by the confidence vote are skipped silently, so
        productively derived words never appear as corrections. Every
        suggestion carries the capitalisation of the word it replaces.

        With *context* set (the default), each correction's candidates are
        re-ranked against the neighbouring words in the same sentence using
        the trigram model. See README "Context-aware ranking" for the
        measurement. Set it False to rank every token in isolation. The pass
        is skipped when no model is available, and if the model backend
        fails mid-request (a dropped database connection) the corrections
        keep the engine's own order instead of the request failing.

        With *skip_foreign* set (the default), proper nouns and acronyms are
        withheld from `corrections` and reported on `TextResult.skipped`
        instead. A known name must be capitalised to be withheld, and a
        capital opening a sentence, a line or a quotation does not count as
        a name. See `khasi_spell.foreign` for what this gives up.
        """
        text = _nfc(text)
        raw = self._load().check_sentence(text)
        text = raw.get("input", text)
        corrections = [
            Correction(
                original=c["original"],
                suggestion=c["suggestion"],
                start=c["start"],
                end=c["end"],
                suggestions=list(c.get("suggestions_top5") or []),
                distances=list(c.get("suggestion_distances") or []),
                glosses=self._glosses(c.get("suggestions_top5") or []),
                gate=c.get("gate_reached", GATE_CHECKED),
                method=c.get("method", ""),
                reason=c.get("reason"),
            )
            for c in (raw.get("corrections") or [])
        ]
        corrections.sort(key=lambda c: c.start)
        for c in corrections:
            if c.method == "doubled_prefix" and c.suggestions:
                c.suggestions, c.distances = self._standard_first(
                    c.original, c.suggestions, c.distances)
                c.suggestion = c.suggestions[0]
                c.glosses = self._glosses(c.suggestions)

        # Withhold proper nouns and acronyms. Done before re-ranking so the
        # language model is not asked to score candidates for a token that
        # is not going to be offered anyway.
        skipped: list[dict] = []
        if skip_foreign and corrections:
            from khasi_spell import foreign

            kept = []
            for c in corrections:
                # A letter from another alphabet, or a digit, inside a word is
                # an error even in a name (`Shill\u043eng`, with a Cyrillic
                # \u043e; `Shn0ng`).
                if c.method in ("lookalike_letters", "digit_in_word"):
                    kept.append(c)
                    continue
                why = foreign.classify(
                    c.original, foreign.is_sentence_initial(text, c.start)
                )
                if why is None:
                    kept.append(c)
                elif self._capitalised_typo(c, why):
                    c.possible_misspelling = True
                    c.reason = ("Capitalised, so it may be a name. If it is not, it is "
                                "probably misspelled: it is in neither the dictionary nor "
                                "the corpus, and a Khasi word is one change away.")
                    kept.append(c)
                else:
                    skipped.append({"word": c.original, "start": c.start,
                                    "end": c.end, "reason": why,
                                    "would_suggest": c.suggestion})
            corrections = kept

        # Re-rank each correction's candidates against its neighbours. The
        # engine ranks every token in isolation; the trigram model breaks
        # ties it cannot see. A missing model is a normal state; a failing
        # one (PostgreSQL gone away) must not turn a spelling check into an
        # HTTP 500, so any error leaves the engine's own order in place.
        if context and corrections:
            try:
                from khasi_spell import context as _context

                _context.rerank(text, corrections, self._lm())
            except Exception as exc:
                if not self._lm_warned:
                    print(f"[khasi-spell] context re-ranking skipped: {exc}")
                    self._lm_warned = True

        # No spelling offers inside a web or email address either.
        flags = self._collect_variants(
            text, [(c.start, c.end) for c in corrections]
            + _tokens.unchecked_spans(text)[0]) if variants else []
        flags, fixes = self._settle_rejected_variants(flags)
        if fixes:
            corrections = sorted(corrections + fixes, key=lambda c: c.start)

        # One reconstruction, from the corrections as finally ranked and
        # filtered, so `corrected` can never disagree with `corrections`.
        corrected = self._apply(text, corrections)

        return TextResult(
            text=text,
            corrected=corrected,
            corrections=corrections,
            variants=flags,
            skipped=skipped,
        )

    def correct_text(self, text: str) -> str:
        """Apply every top-ranked correction and return the result."""
        return self.check_text(text).corrected

    # ------------------------------------------------------------------
    # Real-word errors
    # ------------------------------------------------------------------

    def check_realword(self, text: str, min_margin: Optional[float] = None) -> list:
        """
        Find correctly spelled words that look wrong for their context.

        Separate from `check_text()` because it answers a different question
        and carries a different risk. `check_text()` flags strings that are
        not Khasi words; this flags strings that are Khasi words but appear
        to be the wrong ones, using a trigram model over the corpus.

        *min_margin* applies to THIS call only. It used to be written onto
        the shared detector, so in the HTTP service one request's
        sensitivity became every later request's default. Defaults to
        `RealWordDetector.min_margin`; see khasi_spell.realword for the
        measured precision and recall at each setting.

        Returns a list of `RealWordFlag`, with offsets into the NFC text.
        Raises FileNotFoundError with build instructions when no trigram
        model is available.
        """
        from khasi_spell.realword import RealWordDetector

        text = _nfc(text)
        if self._realword is None:
            with self._lm_lock:
                if self._realword is None:
                    self._realword = RealWordDetector(self)
        return self._realword.check(text, min_margin=min_margin)

    # ------------------------------------------------------------------
    # Vocabulary
    # ------------------------------------------------------------------

    def add_word(self, word: str, frequency: int = 20) -> None:
        """
        Teach this speller a word for the session only.

        The word is accepted from then on, whatever its letters — names and
        loans such as `Meghalaya` or `Congress`, which break Khasi
        phonotactics and were still rejected after being "added" — and it
        becomes a correction candidate, so a typo of it is corrected to it.

        Not persisted, and not shared: other spellers in the process do not
        see it. *frequency* is its weight in the ranking tie-break.
        """
        checker = self._load().spell
        normalised = _tokens.canonical(word).lower().strip()
        if not normalised:
            return
        speller = checker._speller
        if speller is not None and hasattr(speller, "add_word"):
            speller.add_word(normalised, frequency)
        checker.teach_word(normalised)
        self._variant_cache.clear()   # a new word can change variants

    def add_words(self, words) -> None:
        """Teach the checker several words for this session only."""
        for w in words:
            self.add_word(w)


__all__ = [
    "KhasiSpeller",
    "WordResult",
    "TextResult",
    "Correction",
    "GATE_INVALID_STANDALONE",
    "GATE_PHONOTACTIC",
    "GATE_ACCEPTED",
    "GATE_CHECKED",
]

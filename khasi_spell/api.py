"""
api.py — optional HTTP service for the Khasi spellchecker.

    pip install -r requirements-api.txt
    uvicorn khasi_spell.api:app --port 8000

Requires FastAPI, which the library itself does not. Keeping the service
optional means the package has no third-party dependencies unless you
actually want a server.

The `POST /v1/spellcheck` route reproduces the request and response shape
of the parent platform's endpoint, so existing clients work unchanged.
Routes without the `/v1` prefix are this package's own, cleaner surface.
"""

from __future__ import annotations

import os
import re
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Literal, Optional

try:
    from fastapi import FastAPI, HTTPException, Response
    from fastapi.middleware.cors import CORSMiddleware
    from fastapi.responses import FileResponse, JSONResponse
    from pydantic import BaseModel, Field
except ImportError as exc:  # pragma: no cover
    raise ImportError(
        "khasi_spell.api needs FastAPI and Pydantic.\n"
        "Install them with:  pip install -r requirements-api.txt"
    ) from exc

from khasi_engine import morphology
from khasi_engine.database import canonical_pos
from khasi_spell.speller import KhasiSpeller


_speller: Optional[KhasiSpeller] = None


def get_speller() -> KhasiSpeller:
    if _speller is None:  # pragma: no cover - lifespan guarantees this
        raise HTTPException(503, "spellchecker not ready")
    return _speller


def _load_dotenv() -> Optional[str]:
    """
    Read `.env` beside the project root into os.environ, if it exists.

    Twelve lines of stdlib rather than a dependency, because the package has
    no mandatory third-party requirement and this is not worth breaking that
    for. Values already present in the environment always win, so an explicit
    `DATABASE_URL=... uvicorn ...` still overrides the file.

    This exists because `export DATABASE_URL=...` lives and dies with one
    shell: run uvicorn from a second terminal and the service silently loads
    the JSON lexicon instead, which looks identical until you notice the
    startup line.
    """
    path = Path(__file__).resolve().parent.parent / ".env"
    if not path.is_file():
        return None
    loaded = []
    try:
        for line in path.read_text(encoding="utf-8").splitlines():
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            key, value = key.strip(), value.strip().strip('"').strip("'")
            if key and key not in os.environ:
                os.environ[key] = value
                loaded.append(key)
    except OSError as e:
        print(f"[khasi-spell] could not read {path}: {e}")
        return None
    return f"{path.name}: {', '.join(loaded)}" if loaded else None


@asynccontextmanager
async def lifespan(app: FastAPI):
    note = _load_dotenv()
    if note:
        print(f"[khasi-spell] loaded {note}")
    print("[khasi-spell] lexicon source: "
          + ("PostgreSQL" if os.environ.get("DATABASE_URL") else "JSON file"))
    # Load the lexicon once at start-up rather than on the first request,
    # so no user ever pays the ~15-20 s cost.
    global _speller
    _speller = KhasiSpeller(
        db_path=os.environ.get("KHASI_DB_PATH") or None,
        # Opt-in: the re-ranker measured WORSE than rule-only ranking on
        # this project's probe (top-1 53.2% -> 49.4%), so it never turns
        # itself on just because a model happens to be present.
        use_embeddings=os.environ.get("KHASI_SPELL_EMBEDDINGS", "").lower()
                       in ("1", "true", "yes", "on"),
        eager=True,
    )
    # Warm the embedding model too, so "startup complete" means it. Left
    # lazy, the first request to need re-ranking would stall ~12 s.
    if _speller._use_embeddings:
        state = _speller.preload_embeddings()
        if state.get("enabled"):
            print(f"[khasi-spell] embeddings ready — {state['backend']} "
                  f"backend, vocab {state.get('vocab_size')}")
        else:
            print(f"[khasi-spell] embeddings unavailable, staying rule-only: "
                  f"{state.get('error', 'not configured')}")
    print(f"[khasi-spell] ready — {_speller.word_forms} word forms "
          f"({_speller.vocabulary_size} frequency-table keys)")
    yield
    _speller = None


app = FastAPI(
    title="Khasi Spellchecker",
    version="1.0.0",
    description="Morphology-gated spelling correction for the Khasi language.",
    lifespan=lifespan,
)

# CORS — by default allow any origin (handy for dev). In production set
# CORS_ORIGINS to a comma-separated list to restrict it, e.g.:
#   CORS_ORIGINS=https://khasi-spell.vercel.app,https://www.khasi-nlp.org
#
# Without this every browser call from another origin is blocked, so the
# service is reachable by curl and by its own bundled page at "/" and by
# nothing else. Same shape as the parent project's app.py so the two behave
# identically once both are deployed.
_cors_origins_env = os.environ.get("CORS_ORIGINS", "*").strip()
_cors_origins = (
    [o.strip() for o in _cors_origins_env.split(",") if o.strip()]
    if _cors_origins_env != "*" else ["*"]
)
app.add_middleware(
    CORSMiddleware,
    allow_origins=_cors_origins,
    allow_methods=["*"],
    allow_headers=["*"],
)
print(f"[khasi-spell] CORS allowed origins: {_cors_origins}")


# ----------------------------------------------------------------------
# Schemas
# ----------------------------------------------------------------------

class SpellCheckRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, examples=["shnng bha"])
    mode: Literal["word", "sentence"] = "sentence"


class WordRequest(BaseModel):
    word: str = Field(..., min_length=1, max_length=100)
    n: int = Field(default=5, ge=1, le=20)


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=100000)
    # Context re-ranking is on by default; exposed so a caller checking
    # word lists or fragments — where neighbours carry no signal — can
    # turn it off.
    context: bool = Field(default=True)
    # Proper nouns and acronyms are withheld by default; they land on
    # `skipped` rather than `corrections`.
    skip_foreign: bool = Field(default=True)


class RealWordRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=100000)
    min_margin: Optional[float] = Field(default=None, ge=0.0, le=50.0)


class BatchRequest(BaseModel):
    words: list[str] = Field(..., min_length=1, max_length=1000)
    n: int = Field(default=5, ge=1, le=20)


# ----------------------------------------------------------------------
# Routes
# ----------------------------------------------------------------------

_STATIC = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
def index():
    """
    The web interface.

    `no-cache` means "revalidate before reusing", not "never cache". Without
    it FileResponse sends only ETag and Last-Modified, and a browser is then
    free to apply heuristic freshness — commonly a fraction of the file's age
    — and serve a stale copy with no request at all. Editing index.html and
    reloading then shows the OLD page, which is indistinguishable from the
    edit not having happened.

    The cost is one conditional request per load, answered with a 304 and no
    body whenever the file has not changed.
    """
    page = _STATIC / "index.html"
    if not page.is_file():  # pragma: no cover - only if the file is deleted
        return JSONResponse(service_info())
    return FileResponse(page, media_type="text/html",
                        headers={"Cache-Control": "no-cache"})


@app.get("/api", include_in_schema=False)
def api_index():
    """Machine-readable service index."""
    return service_info()


def service_info() -> dict:
    """Service index — what this is and where everything lives."""
    ready = _speller is not None and _speller.loaded
    return {
        "service": "khasi-spellchecker",
        "version": app.version,
        "ready": ready,
        # Distinct single-token forms a correction can be drawn from. This is
        # the number to show a user. `vocabulary_size` is the frequency-table
        # size and counts multi-word phrases; it is kept for compatibility.
        "word_forms": _speller.word_forms if ready else 0,
        "vocabulary_size": _speller.vocabulary_size if ready else 0,
        "docs": "/docs",
        "web_ui": "/",
        "endpoints": {
            "GET  /health": "readiness and vocabulary size",
            "POST /word": '{"word": "lyngdo"} -> gate, method, suggestions',
            "POST /check": '{"text": "..."} -> spans + corrected string',
            "POST /correct": '{"text": "..."} -> corrected string only',
            "POST /realword": '{"text": "..."} -> contextually wrong real words',
            "POST /batch": '{"words": [...]} -> up to 1000 words per call',
            "POST /v1/spellcheck": 'parent-platform compatible: {"text","mode"}',
        },
    }


@app.get("/favicon.ico", include_in_schema=False)
def favicon():
    """Browsers request this unprompted; answer quietly instead of 404-ing."""
    return Response(status_code=204)


@app.get("/health")
def health():
    ready = _speller is not None and _speller.loaded
    return {
        "ok": ready,
        # Distinct single-token forms a correction can be drawn from. This is
        # the number to show a user. `vocabulary_size` is the frequency-table
        # size and counts multi-word phrases; it is kept for compatibility.
        "word_forms": _speller.word_forms if ready else 0,
        "vocabulary_size": _speller.vocabulary_size if ready else 0,
        "version": app.version,
        "embeddings": _speller.embeddings if ready else {"enabled": False},
        # Both of these change results silently when absent, so report them.
        "corpus_frequency": (
            {"applied": _speller.corpus_frequencies is not None,
             **({"distinct_values": _speller.corpus_frequencies["distinct_values"],
                 "corpus_tokens": _speller.corpus_frequencies["corpus_tokens"]}
                if _speller.corpus_frequencies else {})}
            if ready else {"applied": False}),
        "language_model": _speller.language_model if ready else {"available": False},
    }


@app.post("/v1/spellcheck")
def spellcheck_v1(req: SpellCheckRequest):
    """
    Compatibility route: same contract as the parent platform's endpoint.

    mode='word'     -> raw engine result for one word
    mode='sentence' -> {input, corrections, corrected}
    """
    sp = get_speller()
    if req.mode == "word":
        return sp.analyser.spell.suggest(req.text.strip(), n=5)
    return sp.analyser.check_sentence(req.text)


@app.post("/word")
def check_word(req: WordRequest):
    """Check one word. Returns the outcome and why it was reached."""
    sp = get_speller()
    payload = sp.check(req.word).to_dict()
    if req.n != 5:
        # `glosses` and `distances` are documented as parallel to
        # `suggestions`. Replacing only the word list left a shorter list of
        # words beside a five-long list of meanings, and any difference in
        # order between suggest() and check() would then print one word's
        # meaning under another. Rebuild all three together, keyed by word.
        words = sp.suggest(req.word, n=req.n)
        at = {w: i for i, w in enumerate(payload.get("suggestions") or [])}
        for key, blank in (("glosses", ""), ("distances", None)):
            prev = payload.get(key) or []
            payload[key] = [
                prev[at[w]] if w in at and at[w] < len(prev) else blank
                for w in words
            ]
        payload["suggestions"] = words
    return payload


@app.post("/check")
def check_text(req: TextRequest):
    """Check a span of text. Returns spans, suggestions and a corrected string."""
    return get_speller().check_text(req.text, context=req.context,
                                        skip_foreign=req.skip_foreign).to_dict()


@app.post("/correct")
def correct_text(req: TextRequest):
    """Return the text with every top-ranked correction applied."""
    sp = get_speller()
    result = sp.check_text(req.text, context=req.context,
                                        skip_foreign=req.skip_foreign)
    # Count what actually changed, not what was flagged. A word the engine
    # knows is wrong but cannot fix — `sbngaifi`, which contains an `f` —
    # is now reported with a null suggestion and left as written, so
    # len(corrections) overstates the edits made to the text.
    applied = sum(1 for c in result.corrections
                  if (c.get("suggestion") if isinstance(c, dict)
                      else getattr(c, "suggestion", None)))
    return {
        "text": result.text,
        "corrected": result.corrected,
        "changes": applied,
        # Flagged but unfixable — the caller can show them without implying
        # the text was edited.
        "flagged_without_suggestion": len(result.corrections) - applied,
    }


# Engine status strings, worded for a reader. The raw values are enum-ish
# (`recursive_derivation`, `lexicalized_recursive_derivation`) and were being
# printed to the panel as-is.
_STATUS_LABEL = {
    "direct_match":            "Found in the lexicon as written",
    "derived_form":            "Derived — one affix over a root the lexicon records",
    "recursive_derivation":    "Derived — stacked affixes over a root",
    "lexicalized_recursive_derivation":
                               "Derived, and recorded in the lexicon in its own right",
    "partial_parse":           "Partly parsed — some material is unaccounted for",
    "infix_detected":          "Infixed form",
    "suffix_stripped":         "Suffixed form",
    "not_found":               "No parse — the analyser could not decompose this form",
}


def _affix_detail(p2: dict) -> dict:
    """Per-affix function, the derivational order, and rejected parses.

    `morphology.parse()` records what every affix DOES — `jing-` is the
    nominalizer, `ïa-` the reciprocal — along with the word classes each
    attaches to, the chain the derivation was built in, and the alternative
    parses it scored and rejected. All of it sat in `affix_info`, `tree` and
    `alternatives`, and none of it was exposed: the panel could name the
    affixes but not say what any of them was for.
    """
    try:
        info = p2.get("affix_info") or {}
        out: dict = {}

        # Affixes in application order, innermost last, each with its label.
        affixes = []
        for key, order in (("prefix", 1), ("prefix2", 2), ("prefix3", 3)):
            form = info.get(key)
            if not form:
                continue
            affixes.append({
                "form":       form,
                "role":       "prefix",
                "label":      info.get(f"{key}_function_label") or "",
                "function":   info.get(f"{key}_function") or "",
                "applies_to": info.get(f"{key}_applies_to") or [],
            })
        for key in ("infix", "suffix", "clitic"):
            form = info.get(key)
            if form:
                affixes.append({
                    "form":       form,
                    "role":       key,
                    "label":      info.get(f"{key}_function_label") or "",
                    "function":   info.get(f"{key}_function") or "",
                    "applies_to": info.get(f"{key}_applies_to") or [],
                })
        if affixes:
            out["affixes"] = affixes

        # "lehkai → ïalehkai → jingïalehkai" — the word built one step at a
        # time, which is the clearest statement of a stacked derivation.
        if info.get("derivational_chain"):
            out["chain"] = [s.strip() for s in
                            str(info["derivational_chain"]).split("→") if s.strip()]

        # The POS the derivation arrives at, which need not be the root's:
        # jing- takes a verb and yields a noun.
        tree = p2.get("tree") or {}
        if tree.get("pos"):
            out["derived_pos"] = tree["pos"]
        if info.get("root_in_lexicon") is not None:
            out["root_in_lexicon"] = bool(info["root_in_lexicon"])

        alts = [x for x in (p2.get("alternatives") or []) if x.get("structure")]
        if alts:
            out["alternatives"] = [
                {"structure": x.get("structure"), "score": x.get("score")}
                for x in alts[:3]
            ]
        if p2.get("status") in _STATUS_LABEL:
            out["status_label"] = _STATUS_LABEL[p2["status"]]
        return out
    except Exception:
        return {}


# Noun-class articles, read from the lexicon's own declared blocks rather than
# restated here. Two blocks carry them and they are worded differently:
#
#   grammar_schema.noun_class_clitics   i -> gender "neuter_diminutive"
#   morphology.clitics                  i -> gender "neuter"
#
# The schema block is the richer of the two and is preferred, with the
# morphology table as the fallback. Either way both features come through:
# the hard-coded table this replaces carried gender only, so `ki` lost its
# number and the panel could not say "common plural".
#
# The lexicon stores an entry's own clitic in `grammar.clitic`; the 1906
# dictionary ALSO printed it inside the definition, so it arrives a second time
# as a bare "U" sense, which `_split_senses` drops.
# Read lazily, not at import: `morphology.CLITICS` is empty until KhasiDB's
# constructor calls set_morphology_cache, and this module is imported first.
_ARTICLES_CACHE: dict[str, str] = {}


def _articles() -> dict[str, str]:
    if _ARTICLES_CACHE:
        return _ARTICLES_CACHE
    block = {}
    if _speller is not None:
        schema = (getattr(_speller.analyser.db, "_extra_blocks", {}) or {}) \
            .get("grammar_schema") or {}
        block = schema.get("noun_class_clitics") or {}
    block = block or (getattr(morphology, "CLITICS", None) or {})
    for form, feats in block.items():
        if not isinstance(feats, dict):
            continue
        parts = [str(feats.get(k, "")).replace("_", "/").strip()
                 for k in ("gender", "number")]
        _ARTICLES_CACHE[str(form).lower()] = " ".join(p for p in parts if p)
    # Fallback only if the block is unavailable (isolated import, no lexicon).
    return _ARTICLES_CACHE or {
        "u": "masculine singular", "ka": "feminine singular",
        "i": "neuter singular", "ki": "common plural"}
# Part-of-speech abbreviations the source prints at the head of a sense.
_POS_ABBREV = {
    "n": "noun", "v": "verb", "a": "adjective", "adj": "adjective",
    "adv": "adverb", "int": "interjection", "pron": "pronoun",
    "prep": "preposition", "conj": "conjunction",
}


# The 1906 dictionary's "[Imit. x-y.]": its preface explains "Imitatives, or
# word-collocations, have been given where necessary", i.e. the jingle pair
# a word enters into ("ab" -> "ab-cd"). It is not part of the meaning,
# so it is lifted out and shown on its own line. A note the scan cut short
# ("[Imit. ka") carries nothing readable and is dropped.
_IMIT = re.compile(r"\[Imit\b[.:]?\s*([^\]]*)(?:\]|$)")
_IMIT_HYPHEN_GAP = re.compile(r"(\w)- (\w)")


def _imitatives(text: str) -> tuple[str, list]:
    """Return (text without its [Imit. …] notes, the collocations found)."""
    found = []
    for m in _IMIT.finditer(text):
        c = _IMIT_HYPHEN_GAP.sub(r"\1-\2", " ".join(m.group(1).split())).strip(" .;,")
        if len(re.findall(r"[A-Za-zÏïÑñ]", c)) >= 4:
            found.append(c)
    return " ".join(_IMIT.sub(" ", text).split()).strip(), found


def _split_senses(gloss_list, pos: str, clitic: str) -> dict:
    """Separate a dictionary gloss into senses, notes and grammar.

    The lexicon keeps the 1906 dictionary's own notation inside the gloss, so
    `lyngdoh` reads "(short o); U; N. druid; Priest; Sacred grove". Only the
    last three are meanings. The first is a pronunciation note and the second
    is the noun-class article — which the entry ALREADY carries in
    `grammar.clitic`, so printing it as a sense states it twice and reads as
    though the word means "U".

    Returns `{"senses": [...], "notes": [...], "imitatives": [...]}`, dropping a
    bare article and a POS abbreviation that merely repeats `grammar.pos`.
    """
    senses, notes, imits = [], [], []
    for raw in (gloss_list or []):
        for part in str(raw).split(";"):
            s, found = _imitatives(part.strip())
            imits.extend(x for x in found if x not in imits)
            if not s:
                continue
            # "(short o)", "(better than dynda)" — the source's own asides.
            if s.startswith("(") and s.endswith(")"):
                notes.append(s[1:-1].strip())
                continue
            # A bare article, already held in grammar.clitic.
            if s.lower().rstrip(".") in _articles():
                continue
            # "N. druid" -> drop the "N." when it only repeats grammar.pos.
            m = re.match(r"^([A-Za-z]{1,4})\.\s+(.*)$", s)
            if m:
                abbrev = _POS_ABBREV.get(m.group(1).lower())
                # Compare canonically: the entry's pos may be either
                # vocabulary ("adjective" or "ADJ"), and a raw lower() on the
                # canonical form gives "adj", which never equals "adjective".
                if abbrev and canonical_pos(abbrev) == canonical_pos(pos):
                    s = m.group(2).strip()
            if s:
                senses.append(s)
    out = {}
    if senses:
        out["senses"] = senses
    if notes:
        out["notes"] = notes
    if imits:
        out["imitatives"] = imits
    arts = _articles()
    if clitic and clitic.lower() in arts:
        out["article"] = clitic.lower()
        # The features the clitics block declares, so the panel can state them
        # instead of carrying its own copy of the paradigm.
        if arts[clitic.lower()]:
            out["article_gloss"] = arts[clitic.lower()]
    return out


def _syllables_of(db, word: str):
    """(syllables, pattern) from *word*'s own lexicon entry, else (None, None)."""
    rows = db.lookup(word) or []
    if not rows:
        return None, None
    rich = (getattr(db, "_enriched_by_id", None) or {}).get(
        rows[0].get("entry_id")) or {}
    derived = (rich.get("phonology") or {}).get("derived") or {}
    return derived.get("syllables"), derived.get("pattern")


def _stitch_derived_phonology(db, word: str, layers: dict):
    """Syllabify a DERIVED form from the pieces the lexicon does carry.

    A productive form usually has no entry of its own. `jingïakhlad` is known
    only because it appears inside the phrase `dak jingïakhlad ne jingkynnoh
    ktien`, so `is_known()` is True while `lookup()` returns nothing — and the
    panel showed "well formed" plus a letter count and no syllables, while
    `pynïakhlad`, which does have an entry, showed the full breakdown. Same
    shape of word, two different panels. 885 entries that DO exist also carry
    no syllabification, and they had the same empty panel.

    The affixes and the root are themselves entries, so the answer is
    assembled from the lexicon rather than invented — `jing` (CVC) + `ïa` (VV)
    + `lehkai`. Each layer is taken in the order morphology reports it:
    prefix, prefix2, …, root.

    Returns (syllables, pattern) ONLY when the assembled syllables spell the
    word exactly; otherwise (None, None) and the caller shows nothing, as
    before. That reconstruction check is what stops this from guessing.
    """
    root = (layers or {}).get("root") or ""
    if not root or not word.endswith(root):
        return None, None
    pieces = []
    for key in ("prefix", "prefix2", "prefix3"):
        v = (layers or {}).get(key)
        if v:
            pieces.append(str(v).strip("-"))
    pieces.append(root)

    syl, pat_parts = [], []
    for piece in pieces:
        ps, pp = _syllables_of(db, piece)
        if not ps:
            return None, None
        syl.extend(ps)
        pat_parts.append(pp or "")
    if "".join(str(x) for x in syl) != word:
        return None, None
    pattern = ".".join(p for p in pat_parts if p) if all(pat_parts) else None
    if pattern and len(pattern.split(".")) != len(syl):
        pattern = None
    return syl, pattern


def _stored_phonology(sp, word: str) -> dict:
    """The lexicon's own phonological analysis of *word*, when it has one.

    Returns `{"syllables": [...], "pattern": "CVC.CVC", "traits": [...]}`, or
    `{}` for a form the lexicon does not carry. `traits` flattens the entry's
    boolean flags into the ones worth showing a reader, already worded —
    the raw dict is nine keys, most of them false most of the time.
    """
    try:
        db = sp.analyser.db
        entry = (db.lookup(word) or [None])[0]

        def _assembled():
            """Fallback used when the lexicon has no syllabification to show."""
            try:
                parsed = morphology.parse(word.lower(), db.lookup)
            except Exception:
                return {}
            layers = parsed.get("layers") or {}
            syl, pat = _stitch_derived_phonology(db, word.lower(), layers)
            if not syl:
                return {}
            out = {"syllables": list(syl),
                   "derived_from": layers.get("root")}
            if pat:
                out["pattern"] = pat
            return out

        if not entry:
            return _assembled()
        # Read the ENRICHED record, not the flat one lookup() returns. The
        # flat shape keeps only `phonological_flags.syllable_structure` and
        # drops the syllabification entirely (see _enriched_to_flat), so the
        # syllables are reachable only through the by-id index — which the
        # PostgreSQL loader populates from the phon_derived/phon_flags columns.
        rich = (getattr(db, "_enriched_by_id", None) or {}).get(entry.get("entry_id")) or {}
        phon = rich.get("phonology") or {}
        derived = phon.get("derived") or {}
        flags = phon.get("flags") or {}
        if not derived.get("syllables"):
            # The entry exists but was never syllabified — 885 records are in
            # that state. Assemble it the same way rather than showing a
            # letter count on its own.
            assembled = _assembled()
            if assembled:
                derived = {"syllables": assembled.get("syllables"),
                           "pattern": assembled.get("pattern"),
                           "diphthong_nucleus": derived.get("diphthong_nucleus"),
                           "has_long_vowel": derived.get("has_long_vowel")}
        out = {}
        if derived.get("syllables"):
            out["syllables"] = list(derived["syllables"])
        if derived.get("pattern"):
            out["pattern"] = derived["pattern"]
        if derived.get("diphthong_nucleus"):
            out["diphthong"] = derived["diphthong_nucleus"]
        traits = []
        if derived.get("has_long_vowel"):    traits.append("long vowel")
        if flags.get("has_diphthong"):       traits.append("diphthong nucleus")
        if flags.get("has_triphthong"):      traits.append("triphthong nucleus")
        if flags.get("has_aspirate_onset"):  traits.append("aspirate onset")
        if flags.get("has_glottal_final"):   traits.append("glottal final")
        if traits:
            out["traits"] = traits
        return out
    except Exception:
        return {}


def _ipa(word: str) -> dict:
    """Rule-based IPA, or an empty dict if the converter is unavailable."""
    try:
        from khasi_engine import g2p
        out = g2p.convert(word) or {}
        return {"surface": out.get("surface", ""),
                # The converter flags forms whose vowel length it cannot
                # recover from spelling — War (2001) pp.49-54 documents length
                # as contrastive but unmarked, so this is a real gap in the
                # orthography, not a defect in the rules.
                "length_uncertain": bool(out.get("needs_review"))}
    except Exception:
        return {}


def _candidate_reasons(sp, word: str) -> list[dict]:
    """Each suggestion with the phoneme operations that relate it to *word*."""
    from khasi_engine.spell_checker import explain_edit
    r = sp.check(word)
    out = []
    for i, cand in enumerate(r.suggestions[:5]):
        ops = explain_edit(word, cand)
        out.append({
            "word": cand,
            "distance": (r.distances[i] if i < len(r.distances) else None),
            "gloss": (r.glosses[i] if i < len(r.glosses) else ""),
            "ops": ops,
            "known_confusion": any(o.get("known") for o in ops),
        })
    return out


@app.post("/analyse")
def analyse_word(req: WordRequest):
    """
    Linguistic analysis of one word: phonology, morphology, assimilation.

    Distinct from /word, which reports the spelling VERDICT and the confidence
    vote behind it. This reports what the word IS — which phonotactic rule it
    breaks, how the morphology decomposes it, whether an assimilation rule
    fired. The engine computes all of this already; nothing exposed it.

    Trimmed on the way out: `phase2.matches` carries the full lexicon record
    for every candidate root, which is far more than a reader needs and would
    dominate the payload.
    """
    sp = get_speller()
    a = sp.analyser.analyse(req.word.strip())
    p1 = a.get("phase1") or {}
    p2 = a.get("phase2") or {}
    p3 = a.get("phase3") or {}
    p4 = a.get("phase4") or {}
    top = (p2.get("matches") or [{}])[0]
    flags = top.get("morphological_flags") or {}
    return {
        "word": a.get("input"),
        "verdict": a.get("verdict"),
        "summary": a.get("verdict_desc"),
        "phonology": {
            # Pronunciation, computed live rather than read from the entry.
            # The PostgreSQL migration does not carry `phonology.ipa` — 0 of
            # 29,692 rows have it — so under DATABASE_URL the stored IPA is
            # simply absent, and a derived form like `jingpynïalehkai` never
            # had one to begin with. The rule engine covers both.
            "ipa": _ipa(a.get("input") or ""),
            "ok": p1.get("pass"),
            "errors": p1.get("errors") or [],
            "warnings": p1.get("warnings") or [],
            "details": p1.get("details") or {},
            # Real phonology, as opposed to the validator's bookkeeping.
            #
            # `details` above carries what the phonotactic CHECKER needed —
            # letter count, hyphen-part count, the initial cluster it tested.
            # The lexicon separately stores the actual analysis of the word:
            # its syllabification, its CV pattern, whether the nucleus is a
            # diphthong, whether it ends in a glottal. Nothing read it, so a
            # panel headed "Phonology" could only show letter counts.
            #
            # Present only for a headword: these fields are curated per entry
            # (rebuilt against War 2001 in the v3.8 pass), not derived live, so
            # a form the lexicon does not carry has no syllabification to show.
            **_stored_phonology(sp, a.get("input") or ""),
        },
        "morphology": {
            "status": p2.get("status"),
            "root": p2.get("root"),
            "layers": p2.get("layers") or {},
            "gloss": top.get("english_gloss") or "",
            "pos": top.get("grammatical_class") or "",
            "gender": (top.get("gender") or "") if top.get("gender") != "none" else "",
            "is_compound": bool(flags.get("is_compound")),
            "compound_roots": flags.get("compound_roots") or [],
            # Senses separated from the source dictionary's own notation, and
            # the noun-class article promoted to the field it belongs in.
            **_split_senses(
                [top.get("english_gloss") or ""],
                top.get("grammatical_class") or "",
                top.get("clitic") or "",
            ),
            # What each affix DOES, the order it was applied in, and what else
            # the parser considered. `layers` alone gives bare strings — a
            # panel could show "jing- + ïa- + lehkai" without being able to say
            # that jing- nominalises and ïa- is the reciprocal, which is the
            # part a reader of Khasi actually wants. See _affix_detail.
            **_affix_detail(p2),
        },
        "assimilation": {
            "rule": p3.get("rule_detected"),
            "description": p3.get("rule_description"),
            "prefix": p3.get("reconstructed_prefix"),
            "root": p3.get("reconstructed_root"),
        },
        # Why each proposal is a plausible correction, not merely a close
        # string: which phonemes differ, and whether the confusion matrix
        # recognises that difference as one Khasi orthography actually makes.
        "candidates": _candidate_reasons(sp, req.word.strip()),
        "compound_fusion": {
            "detected": bool(p4.get("detected")),
            "type": p4.get("type"),
            "description": p4.get("description"),
            "components": p4.get("components") or [],
        },
    }


@app.post("/realword")
def realword(req: RealWordRequest):
    """
    Find correctly spelled words that look wrong for their context.

    Different question from /check, which flags strings that are not Khasi
    words. This flags strings that ARE Khasi words but appear to be the
    wrong ones, scored against a trigram model over the corpus.

    Needs data/ngrams.json.gz; returns 503 with build instructions if absent.
    """
    sp = get_speller()
    try:
        flags = sp.check_realword(req.text, min_margin=req.min_margin)
    except FileNotFoundError as exc:
        raise HTTPException(503, str(exc)) from exc
    return {"text": req.text,
            "flags": [f.to_dict() for f in flags],
            "count": len(flags)}


@app.post("/batch")
def batch(req: BatchRequest):
    """
    Check many words in one call.

    Absent from the parent platform, where document workflows had to make
    one request per token.
    """
    sp = get_speller()
    results = []
    for word in req.words:
        r = sp.check(word)
        results.append({
            "word": word,
            "is_correct": r.is_correct,
            "suggestions": r.suggestions if req.n == 5 else sp.suggest(word, n=req.n),
            "gate": r.gate_name,
        })
    return {"results": results, "count": len(results)}

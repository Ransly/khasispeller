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
from typing import Annotated, Literal, Optional

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

# `.env` first, before any engine module is imported: phonology, morphology
# and assimilation decide at import time whether to read the JSON lexicon,
# and a DATABASE_URL loaded later (it used to be read in the lifespan hook)
# arrived after they had already parsed the 64 MB file.
from khasi_spell.env import load_dotenv as _load_env_file

_DOTENV_NOTE = _load_env_file()

from khasi_engine import morphology                     # noqa: E402
from khasi_engine import tokens as _tokens              # noqa: E402
from khasi_engine.database import canonical_pos         # noqa: E402
from khasi_spell.speller import KhasiSpeller            # noqa: E402
from khasi_spell import syllables as _syllables         # noqa: E402


_speller: Optional[KhasiSpeller] = None


def get_speller() -> KhasiSpeller:
    if _speller is None:  # pragma: no cover - lifespan guarantees this
        raise HTTPException(503, "spellchecker not ready")
    return _speller


@asynccontextmanager
async def lifespan(app: FastAPI):
    if _DOTENV_NOTE:
        print(f"[khasi-spell] loaded {_DOTENV_NOTE}")
    print("[khasi-spell] lexicon source: "
          + ("PostgreSQL" if os.environ.get("DATABASE_URL") else "JSON file"))
    # Where the data files are read from: in the log, which only the
    # operator sees, rather than in /health (see _public).
    from khasi_engine import paths as _paths
    print(f"[khasi-spell] data folder: {_paths.data_dir()}")
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


# The interactive API documentation (/docs, /redoc, /openapi.json) is off
# unless KHASI_SPELL_API_DOCS=1. Public, it mapped every endpoint for anyone
# copying the lexicon's glosses in bulk, and the glosses are licensed
# (LICENSE-DATA). The service index at /api stays.
_API_DOCS = os.environ.get("KHASI_SPELL_API_DOCS", "").strip().lower() in ("1", "true", "yes", "on")

app = FastAPI(
    title="Khasi Spellchecker",
    version="1.0.0",
    description="Morphology-gated spelling correction for the Khasi language.",
    lifespan=lifespan,
    docs_url="/docs" if _API_DOCS else None,
    redoc_url="/redoc" if _API_DOCS else None,
    openapi_url="/openapi.json" if _API_DOCS else None,
)

# CORS — off unless CORS_ORIGINS lists the other sites allowed to call the
# service from a browser, comma-separated, e.g.:
#   CORS_ORIGINS=https://khasi-spell.vercel.app,https://www.khasi-nlp.org
# ("*" allows every site). The bundled page at "/" is this same site and
# needs none of it. The default used to be "*", so any website could have
# its visitors' browsers query the service and its licensed glosses. This
# does not stop scripts; only a rate limit would.
_cors_origins_env = os.environ.get("CORS_ORIGINS", "").strip()
_cors_origins = (["*"] if _cors_origins_env == "*" else
                 [o.strip() for o in _cors_origins_env.split(",") if o.strip()])
if _cors_origins:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=_cors_origins,
        allow_methods=["*"],
        allow_headers=["*"],
    )
print(f"[khasi-spell] CORS allowed origins: {_cors_origins or 'this site only'}")


# ----------------------------------------------------------------------
# Schemas
# ----------------------------------------------------------------------

class SpellCheckRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=10000, examples=["shnng bha"])
    mode: Literal["word", "sentence"] = "sentence"


class WordRequest(BaseModel):
    word: str = Field(..., min_length=1, max_length=100)
    n: int = Field(default=5, ge=1, le=20)


# Upper bound on /check, /correct and /realword input. Checking is
# synchronous and costs tens of milliseconds per unknown word, so a large,
# mostly non-Khasi document held a worker for minutes (20,000 characters of
# English-heavy text took 162 s before the 2026-09-30 speed-ups). 50,000
# characters is a long article; send longer documents in parts.
MAX_TEXT_CHARS = 50_000


class TextRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_CHARS)
    # Context re-ranking is on by default; exposed so a caller checking
    # word lists or fragments — where neighbours carry no signal — can
    # turn it off.
    context: bool = Field(default=True)
    # Proper nouns and acronyms are withheld by default; they land on
    # `skipped` rather than `corrections`.
    skip_foreign: bool = Field(default=True)


class RealWordRequest(BaseModel):
    text: str = Field(..., min_length=1, max_length=MAX_TEXT_CHARS)
    min_margin: Optional[float] = Field(default=None, ge=0.0, le=50.0)


class BatchRequest(BaseModel):
    # Each word capped as /word caps it. Words had no limit here, and before
    # the engine's own cap (KhasiSpellChecker.MAX_WORD_CHARS) one long word
    # could hold the service for minutes.
    words: list[Annotated[str, Field(max_length=100)]] = Field(..., min_length=1, max_length=1000)
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
        "docs": "/docs" if _API_DOCS else None,   # KHASI_SPELL_API_DOCS=1
        "web_ui": "/",
        "endpoints": {
            "GET  /health": "readiness and vocabulary size",
            "POST /word": '{"word": "lyngdo"} -> gate, method, suggestions',
            "POST /analyse": '{"word": "..."} -> phonology, morphology, assimilation',
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


# Keys that say where something lives — a path on this server, a URL — are
# left out of /health. It reported the data folder, the language-model file
# and the embedding model by absolute path; a visitor needs to know what is
# running, not where. The startup log still prints the data folder.
_PRIVATE_KEYS = frozenset({"path", "model_path", "data_dir", "url", "dsn",
                           "database_url", "host"})


def _public(value):
    """*value* without any key in _PRIVATE_KEYS, at any depth."""
    if isinstance(value, dict):
        return {k: _public(v) for k, v in value.items() if k not in _PRIVATE_KEYS}
    if isinstance(value, list):
        return [_public(v) for v in value]
    return value


@app.get("/health")
def health():
    ready = _speller is not None and _speller.loaded
    return _public({
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
        # The optional resources that switch features off when missing. A
        # non-editable install without KHASI_DATA_DIR used to lose all of
        # them without a word; say which are present.
        "resources": _resources() if ready else {},
    })


def _resources() -> dict:
    from khasi_engine import paths
    names = {"corpus_frequencies": "corpus_freq.json",
             "ngram_file": "ngrams.json.gz",
             "gazetteer": "gazetteer.json",
             "runtogether_splits": "runtogether_candidates_review.csv",
             "gloss_spelling_links": "spelling_links.json"}
    return {k: paths.data_file(v).is_file() for k, v in names.items()}


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
    # One search for any n: check() takes n directly, so `suggestions`,
    # `distances` and `glosses` come out of the same ranking and stay
    # parallel. This used to run check() and then suggest() again for any
    # n other than 5, and re-align the three lists by hand.
    return get_speller().check(req.word, n=req.n).to_dict()


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
    # A possible misspelling (a capitalised word that may be a name) is
    # reported but never applied, so it is not a change either.
    possible = sum(1 for c in result.corrections if c.possible_misspelling)
    unfixable = sum(1 for c in result.corrections if not c.suggestion)
    applied = sum(1 for c in result.corrections
                  if c.suggestion and not c.possible_misspelling)
    return {
        "text": result.text,
        "corrected": result.corrected,
        "changes": applied,
        # Flagged but unfixable — the caller can show them without implying
        # the text was edited.
        "flagged_without_suggestion": unfixable,
        "possible_misspellings": possible,
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


# What each infix of the lexicon's morphology.infixes block makes, in words,
# and the class it takes. The engine's one-word label for the first code is
# "verb", which is wrong for `bynriew` 'human being': the code itself says a
# verb OR an abstract noun.
_INFIX_MAKES = {
    "noun_to_verb_or_abstract": ("a verb or an abstract noun", ["noun"]),
    "verb_to_noun":             ("a noun", ["verb"]),
    "verbal_derivation":        ("a verb", []),
    "adj_to_abstract_noun":     ("an abstract noun", ["adjective"]),
}


def _infix_detail(form: str, word: str, root: str) -> dict:
    """An infix row the panel can read, and the word with the infix marked.

    The panel listed `yn` above `briew` exactly as it lists a prefix above
    its root, which reads as yn + briew. An infix sits INSIDE the root —
    b‹yn›riew, sh‹n›ong — so the row carries its own form (-yn-), what the
    lexicon says it makes, and the word with the infix marked, for the
    derivation line (briew → b‹yn›riew). The marking is shown only when the
    infix's own pattern, taken out, gives back the root.
    """
    marker = (form or "").strip("-")
    ix = next((x for x in morphology.INFIXES if x.get("marker") == marker), None)
    if not ix:
        return {}
    fn = ix.get("function") or ""
    makes, takes = _INFIX_MAKES.get(fn, (fn.replace("_", " "), []))
    out = {"form": f"-{marker}-", "function": fn,
           "label": f"infix, makes {makes}", "applies_to": takes}
    pat = ix.get("pattern")
    m = pat.match(word.lower()) if hasattr(pat, "match") and word else None
    if m and root and (m.group(1) + m.group(2)) == root.lower():
        out["marked"] = f"{m.group(1)}‹{marker}›{m.group(2)}"
    return out


def _root_meaning(db, root: str):
    """(first sense, word class) from *root*'s OWN entry, or ("", "").

    The panel used to label the root with the class of the word it was
    looking at — `stad` was shown as a noun because `jingstad` is one.
    """
    rows = db.lookup(root) or []
    if not rows:
        return "", ""
    r = rows[0]
    pos = r.get("grammatical_class") or ""
    senses = _split_senses([r.get("english_gloss") or ""], pos,
                           r.get("clitic") or "").get("senses") or []
    return (senses[0] if senses else ""), pos


def _prefix_label(prefix: str) -> str:
    """What a prefix does, from the lexicon's own registry (pyn- causative)."""
    fn = (morphology.PREFIXES.get(prefix.strip("-")) or {}).get("function") or ""
    return morphology._short_function_label(fn) if fn else ""


def _form_map(word: str, p2: dict, assim: dict, db) -> dict:
    """The word at its two levels: as written (surface) and as built (lexical).

    One segment per morpheme, surface above lexical, so the panel can align
    them and show where they differ — `pyl` above `pyn-` in pyllait, where
    n became l before l. Also the two directions: analysis (surface to
    lexical) and generation (lexical to surface, with the sound change or
    the step-by-step derivation). Four shapes:

        sound change   pyllait  = pyn- + lait   (n -> l before l)
        infix          bynriew  = briew + -yn-  (b‹yn›riew)
        affixes        jingïalehkai = jing- + ïa- + lehkai
        bare root      katkum   = katkum

    Returned only when the pieces spell the written word exactly; anything
    else gets no map, and the panel falls back to its plain listing.
    """
    w = _tokens.nfc(word or "").strip()
    wl = w.lower()
    layers = p2.get("layers") or {}
    info = p2.get("affix_info") or {}
    root = (p2.get("root") or "").strip()

    def seg(surface, lexical, kind, gloss="", pos=""):
        return {"surface": surface, "lexical": lexical, "kind": kind, "gloss": gloss,
                "pos": pos, "changed": surface.lower() != lexical.strip("-").lower()}

    # 1. A sound change where a prefix meets the root.
    if assim.get("rule") and assim.get("prefix") and assim.get("root"):
        pre, rt = assim["prefix"].strip("-"), assim["root"]
        if wl.endswith(rt.lower()) and len(wl) > len(rt):
            g, pos = _root_meaning(db, rt)
            cut = len(w) - len(rt)
            return {"form_map": {
                "segments": [seg(w[:cut], pre + "-", "prefix", _prefix_label(pre)),
                             seg(w[cut:], rt, "root", g, pos)],
                "generation": [pre + "- + " + rt, w],
                "rule": assim.get("description") or ""}}

    # 2. An infix inside the root: written as one word, built from two pieces.
    if layers.get("infix") and root:
        d = _infix_detail(layers["infix"], w, root)
        if d.get("marked"):
            g, pos = _root_meaning(db, root)
            return {"form_map": {
                "segments": [dict(seg(root, root, "root", g, pos), changed=False),
                             dict(seg(d["form"].strip("-"), d["form"], "infix",
                                      d.get("label", "").replace("infix, ", "")),
                                  changed=False)],
                "marked": d["marked"],
                "generation": [root + " + " + d["form"], w]}}

    # 3. Prefixes, outermost first, and any suffix, over a root.
    keys = [k for k in ("prefix", "prefix2", "prefix3") if layers.get(k)]
    suffix = (layers.get("suffix") or "").strip("-")
    if root and (keys or suffix):
        pieces = [layers[k].strip("-") for k in keys] + [root] + ([suffix] if suffix else [])
        if "".join(pieces).lower() == wl:
            at, surface = 0, []
            for p in pieces:                     # the written word's own letters
                surface.append(w[at:at + len(p)])
                at += len(p)
            segs = [seg(surface[i], layers[k].strip("-") + "-", "prefix",
                        info.get(f"{k}_function_label") or _prefix_label(layers[k]))
                    for i, k in enumerate(keys)]
            g, pos = _root_meaning(db, root)
            segs.append(seg(surface[len(keys)], root, "root", g, pos))
            if suffix:
                segs.append(seg(surface[-1], "-" + suffix, "suffix",
                                info.get("suffix_function_label") or ""))
            chain = [s.strip() for s in str(info.get("derivational_chain") or "").split("→")
                     if s.strip()]
            return {"form_map": {
                "segments": segs,
                "generation": chain if len(chain) > 1
                              else [" + ".join(s["lexical"] for s in segs), w]}}

    # 4. A bare root: one piece, the same at both levels.
    if root and root.lower() == wl:
        g, pos = _root_meaning(db, root)
        return {"form_map": {"segments": [seg(w, root, "root", g, pos)]}}
    return {}


def _affix_detail(p2: dict, word: str = "") -> dict:
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
        marked = None
        for key in ("infix", "suffix", "clitic"):
            form = info.get(key)
            if form:
                row = {
                    "form":       form,
                    "role":       key,
                    "label":      info.get(f"{key}_function_label") or "",
                    "function":   info.get(f"{key}_function") or "",
                    "applies_to": info.get(f"{key}_applies_to") or [],
                }
                if key == "infix":
                    detail = _infix_detail(form, word, p2.get("root") or "")
                    marked = detail.pop("marked", None)
                    row.update(detail)
                affixes.append(row)
        if affixes:
            out["affixes"] = affixes

        # "lehkai → ïalehkai → jingïalehkai" — the word built one step at a
        # time, which is the clearest statement of a stacked derivation.
        if info.get("derivational_chain"):
            out["chain"] = [s.strip() for s in
                            str(info["derivational_chain"]).split("→") if s.strip()]
        elif marked:
            out["chain"] = [p2.get("root"), marked]       # briew → b‹yn›riew

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


_SYL_MARKS = str.maketrans("ïñáéíóúý", "inaeiouy")


def _syl_letters(s: str) -> str:
    return re.sub(r"[\s\-'’]", "", _tokens.nfc(s or "").lower())


def _own_syllables(word: str, syllables):
    """*syllables* in *word*'s own letters, or None if they spell another word.

    A stored syllabification is shown only when it spells the word it is shown
    for. `katkum` 'according to' carried `katba`'s kat·ba — 529 single-word
    entries held another word's syllables, an import error repaired by
    scripts/fix_inherited_analyses.py — and nothing checked before display.
    Syllables that differ from the word only by a dropped ï, ñ or accent are
    re-cut from the word itself, so `ia·shong` shows as `ïa·shong`.
    """
    if not syllables:
        return None
    letters = _syl_letters(word)
    joined = "".join(_syl_letters(s) for s in syllables)
    if joined == letters:
        return list(syllables)
    if joined.translate(_SYL_MARKS) != letters.translate(_SYL_MARKS):
        return None
    out, i = [], 0
    for s in syllables:
        n = len(_syl_letters(s))
        out.append(letters[i:i + n])
        i += n
    return out


def _syllables_of(db, word: str):
    """(syllables, pattern) from *word*'s own lexicon entry, else (None, None)."""
    rows = db.lookup(word) or []
    if not rows:
        return None, None
    rich = (getattr(db, "_enriched_by_id", None) or {}).get(
        rows[0].get("entry_id")) or {}
    derived = (rich.get("phonology") or {}).get("derived") or {}
    syl = _own_syllables(word, derived.get("syllables"))
    return (syl, derived.get("pattern")) if syl else (None, None)


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

    Returns `{"syllables": [...], "pattern": "CVC.CVC", "traits": [...]}`.
    `traits` flattens the entry's boolean flags into the ones worth showing
    a reader, already worded — the raw dict is nine keys, most of them false
    most of the time. For a form the lexicon cannot syllabify, even from its
    pieces, the syllables and pattern are worked out by rule and flagged
    `syllables_by_rule`; `{}` only when the rules cannot say either.
    """
    try:
        db = sp.analyser.db
        entry = (db.lookup(word) or [None])[0]

        def _by_rule():
            """Last resort: the syllables by rule — see khasi_spell.syllables.

            `ïatreilang` is ïa- + treilang, and treilang (trei + lang) has no
            entry, so neither the lexicon nor its pieces could syllabify it
            and the panel showed a letter count. Parsed as if the word were
            unknown, so a headword that was never syllabified still gets its
            prefix boundaries: on a direct match the parser reports none.
            """
            w = _tokens.nfc(word).lower()
            try:
                layers = morphology.parse(
                    w, lambda x: [] if x == w else db.lookup(x)).get("layers") or {}
            except Exception:
                layers = {}
            syl = _syllables.syllabify(
                w, [layers[k] for k in ("prefix", "prefix2", "prefix3") if layers.get(k)],
                layers.get("suffix") or "")
            if not syl:
                return {}
            return {"syllables": syl, "pattern": _syllables.pattern(syl),
                    "syllables_by_rule": True}

        def _assembled():
            """Fallback used when the lexicon has no syllabification to show."""
            try:
                parsed = morphology.parse(word.lower(), db.lookup)
            except Exception:
                return _by_rule()
            layers = parsed.get("layers") or {}
            syl, pat = _stitch_derived_phonology(db, word.lower(), layers)
            if not syl:
                return _by_rule()
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
        derived = dict(phon.get("derived") or {})
        flags = phon.get("flags") or {}
        own = _own_syllables(word, derived.get("syllables"))
        if derived.get("syllables") and not own:
            # The stored analysis spells another word (katkum held katba's
            # kat·ba). Its syllables, pattern and the traits read off them
            # describe that word, so none of them is shown.
            derived = {}
        elif own:
            derived["syllables"] = own
        if not derived.get("syllables"):
            # The entry exists but was never syllabified — 885 records are in
            # that state. Assemble it the same way rather than showing a
            # letter count on its own.
            assembled = _assembled()
            if assembled:
                derived = {"syllables": assembled.get("syllables"),
                           "pattern": assembled.get("pattern"),
                           "diphthong_nucleus": derived.get("diphthong_nucleus"),
                           "has_long_vowel": derived.get("has_long_vowel"),
                           "by_rule": assembled.get("syllables_by_rule")}
        out = {}
        if derived.get("syllables"):
            out["syllables"] = list(derived["syllables"])
            if derived.get("by_rule"):
                out["syllables_by_rule"] = True
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
        # The spell checker's decision, the same one /word reports. The
        # verdict above is the analyser's reading of the form; when they
        # differ the summary says why.
        "accepted": a.get("accepted"),
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
            # Advisory notes from the phonotactic check, less the loanword note
            # on a final -l or -s ("Final '-l' in 'shongskul' is not native to
            # Khasi — likely a loanword"), which is shown for no word
            # (maintainer request, 6 October 2026). The engine still makes it;
            # tests/test_phonology_rules.py pins that.
            "warnings": [w for w in (p1.get("warnings") or [])
                         if "likely a loanword" not in w],
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
            # The lexicon's fields are curated per entry (rebuilt against War
            # 2001 in the v3.8 pass). A derived form gets its pieces' entries
            # stitched together; anything else, syllables worked out by rule,
            # flagged `syllables_by_rule` so the panel can say so.
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
            **_affix_detail(p2, a.get("input") or ""),
            # The word at two levels — as written and as built — aligned
            # piece by piece, with any sound change. See _form_map.
            **_form_map(a.get("input") or "", p2,
                        {"rule": p3.get("rule_detected"),
                         "description": p3.get("rule_description"),
                         "prefix": p3.get("reconstructed_prefix"),
                         "root": p3.get("reconstructed_root")},
                        sp.analyser.db),
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
    # The offsets index the NFC-normalised text, so that is the text
    # returned — /check and /correct already did this; /realword echoed the
    # raw input, which differs for decomposed ï/ñ.
    text = _tokens.nfc(req.text)
    try:
        # min_margin applies to this request only; it used to be stored on
        # the shared detector and became every later request's default.
        flags = sp.check_realword(text, min_margin=req.min_margin)
    # The details go to the log, not the response: the exception text named
    # the model file by its path on this server, and a database error can
    # name the database host and user.
    except FileNotFoundError as exc:
        print(f"[khasi-spell] /realword: {exc}")
        raise HTTPException(503, "The word-choice check is not available on this "
                                 "server: its language model is not installed.") from exc
    except Exception as exc:                  # e.g. the n-gram database is down
        print(f"[khasi-spell] /realword: language model unavailable: {exc!r}")
        raise HTTPException(503, "The word-choice check is unavailable right now. "
                                 "Please try again later.") from exc
    return {"text": text,
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
        r = sp.check(word, n=req.n)
        results.append({
            "word": word,
            "is_correct": r.is_correct,
            "suggestions": r.suggestions,
            "gate": r.gate_name,
        })
    return {"results": results, "count": len(results)}

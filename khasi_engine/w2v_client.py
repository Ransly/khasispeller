"""
Word-embedding (FastText) client — semantic re-ranking for the spell checker.

Two backends, resolved in this order:

  1. LOCAL   a gensim FastText model on disk. This project bundles one at
             models/fasttext/word2vec_model.model; override with
             KHASI_W2V_MODEL. Needs gensim. No network, no cold starts.
  2. REMOTE  the Hugging Face Space the parent platform used, via
             KHASI_W2V_URL (+ optional KHASI_W2V_KEY). No gensim needed.

The parent platform had only the remote backend, because its Render host
could not hold the model in memory. This project is not under that
constraint, so local is the default and the remote path is kept for
deployments that are.

`score()` and `is_configured()` keep the signatures the parent used, so
spell_checker.py consumes either backend without knowing which is active.

Environment
-----------
  KHASI_W2V_MODEL   path to a gensim FastText .model file
  KHASI_W2V_BACKEND "local" | "remote" | "auto" (default "auto")
  KHASI_W2V_URL     base URL of the remote Space
  KHASI_W2V_KEY     shared secret for the Space
"""
from __future__ import annotations

import os
import re
import unicodedata
from pathlib import Path
from typing import Optional


class W2vNotConfigured(RuntimeError):
    """No backend available — caller should fall back to rule-only ranking."""


class W2vUpstreamError(RuntimeError):
    """The configured backend failed (bad model file, network, or HTTP error)."""


# Bundled model: <project root>/models/fasttext/word2vec_model.model
_BUNDLED_MODEL = (
    Path(__file__).parent.parent / "models" / "fasttext" / "word2vec_model.model"
)

_model = None            # cached gensim FastText instance
_model_path: Optional[Path] = None


# ---------------------------------------------------------------------------
# Noise filter — mirrors the Hugging Face Space so both backends agree
# ---------------------------------------------------------------------------

_CLEAN_RE = re.compile(r"^[a-zïñ'\-]+$")


def is_clean(tok: str) -> bool:
    """
    True if *tok* looks like a real Khasi word rather than a tokenisation
    artefact. Copied from the Space so local and remote scoring agree on
    what may be surfaced as a suggestion.
    """
    if not tok:
        return False
    tok = unicodedata.normalize("NFC", tok).lower()
    if any(ch.isdigit() for ch in tok):
        return False
    if any(ch in tok for ch in ".,;:!?\"'`/\\()[]{}<>«»“”‘’—_"):
        return False
    if len(tok) < 2 or len(tok) > 25:
        return False
    return bool(_CLEAN_RE.match(tok))


# ---------------------------------------------------------------------------
# Backend resolution
# ---------------------------------------------------------------------------

def _backend_pref() -> str:
    return (os.environ.get("KHASI_W2V_BACKEND") or "auto").strip().lower()


def local_model_path() -> Optional[Path]:
    """The local model that would be used, or None if there isn't one."""
    override = os.environ.get("KHASI_W2V_MODEL")
    if override:
        p = Path(override).expanduser()
        return p if p.is_file() else None
    return _BUNDLED_MODEL if _BUNDLED_MODEL.is_file() else None


def _remote_config() -> Optional[tuple[str, Optional[str]]]:
    url = (os.environ.get("KHASI_W2V_URL") or "").rstrip("/")
    if not url:
        return None
    return url, (os.environ.get("KHASI_W2V_KEY") or None)


def active_backend() -> Optional[str]:
    """Which backend `score()` would use: 'local', 'remote' or None."""
    pref = _backend_pref()
    has_local = local_model_path() is not None
    has_remote = _remote_config() is not None

    if pref == "local":
        return "local" if has_local else None
    if pref == "remote":
        return "remote" if has_remote else None
    if has_local:
        return "local"
    if has_remote:
        return "remote"
    return None


def is_configured() -> bool:
    """True if any backend is available. Does not load the model."""
    return active_backend() is not None


def info() -> dict:
    """Describe the configured backend without loading anything heavy."""
    backend = active_backend()
    d: dict = {"backend": backend, "loaded": _model is not None}
    if backend == "local":
        d["model_path"] = str(local_model_path())
        if _model is not None:
            d["vocab_size"] = len(_model.wv)
            d["dim"] = _model.vector_size
            d["algorithm"] = "skipgram" if _model.sg else "cbow"
    elif backend == "remote":
        d["url"] = _remote_config()[0]
    return d


# ---------------------------------------------------------------------------
# Local backend
# ---------------------------------------------------------------------------

def load_local(path: Optional[str | Path] = None):
    """
    Load and cache the local FastText model.

    Costs roughly 15 s and about 1 GB of resident memory, dominated by the
    763 MB character-n-gram matrix. Called lazily on first scoring request
    unless you invoke it yourself to move the cost to start-up.
    """
    global _model, _model_path

    target = Path(path).expanduser() if path else local_model_path()
    if target is None:
        raise W2vNotConfigured(
            "No local FastText model found. Expected "
            f"{_BUNDLED_MODEL}, or set KHASI_W2V_MODEL."
        )
    if _model is not None and _model_path == target:
        return _model

    try:
        from gensim.models import FastText
    except ImportError as exc:
        raise W2vNotConfigured(
            "gensim is needed for the local embedding backend.\n"
            '  pip install "gensim>=4,<5" "numpy<2"\n'
            "Or set KHASI_W2V_URL to use the remote Space instead."
        ) from exc

    try:
        _model = FastText.load(str(target))
    except Exception as exc:
        raise W2vUpstreamError(f"could not load {target}: {exc}") from exc

    _model_path = target
    return _model


def unload_local() -> None:
    """Drop the cached model and reclaim its memory."""
    global _model, _model_path
    _model = None
    _model_path = None


def _score_local(word: str, candidates: list[str]) -> dict[str, float]:
    model = load_local()
    out: dict[str, float] = {}
    for c in candidates:
        try:
            # FastText composes vectors from character n-grams, so it scores
            # unseen words too — the property that makes it a good fit for a
            # morphologically productive language.
            out[c] = float(model.wv.similarity(word, c))
        except (KeyError, ValueError):
            out[c] = -1.0        # same sentinel the Space returns for OOV
    return out


# ---------------------------------------------------------------------------
# Remote backend
# ---------------------------------------------------------------------------

def _score_remote(word: str, candidates: list[str]) -> dict[str, float]:
    cfg = _remote_config()
    if cfg is None:
        raise W2vNotConfigured("KHASI_W2V_URL is not set")
    url, key = cfg

    try:
        import requests
    except ImportError as exc:
        raise W2vNotConfigured(
            "requests is needed for the remote embedding backend.\n"
            "  pip install requests"
        ) from exc

    headers = {"Content-Type": "application/json"}
    if key:
        headers["X-Api-Key"] = key
    try:
        # 30 s — a cold Space spends ~15 s reloading the model.
        r = requests.post(
            f"{url}/score",
            json={"word": word, "candidates": candidates},
            headers=headers,
            timeout=30,
        )
        r.raise_for_status()
        payload = r.json()
        return {item["text"]: float(item["score"])
                for item in payload.get("ranked", [])}
    except requests.RequestException as e:
        raise W2vUpstreamError(str(e)) from e


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def score(word: str, candidates: list[str]) -> dict[str, float]:
    """
    Cosine similarity between *word* and each candidate.

    Returns {candidate: score}; OOV or unscoreable candidates get -1.0 so
    they sort to the bottom of any blended ranking.

    Raises W2vNotConfigured when no backend is available and
    W2vUpstreamError when the chosen backend fails — spell_checker catches
    both and falls back to rule-only ranking.
    """
    backend = active_backend()
    if backend == "local":
        return _score_local(word, candidates)
    if backend == "remote":
        return _score_remote(word, candidates)
    raise W2vNotConfigured(
        "No embedding backend. Bundle a model at "
        f"{_BUNDLED_MODEL}, set KHASI_W2V_MODEL, or set KHASI_W2V_URL."
    )


def similar(word: str, topn: int = 10, clean: bool = True) -> list[tuple[str, float]]:
    """
    Nearest neighbours of *word* in the embedding space.

    Local backend only — a diagnostic for inspecting what the model has
    actually learned. Raises W2vNotConfigured if no local model is present.
    """
    model = load_local()
    try:
        hits = model.wv.most_similar(word, topn=topn * 3 if clean else topn)
    except KeyError as exc:
        raise W2vUpstreamError(f"{word!r} has no representation: {exc}") from exc
    if clean:
        hits = [(w, s) for w, s in hits if is_clean(w)]
    return [(w, float(s)) for w, s in hits[:topn]]

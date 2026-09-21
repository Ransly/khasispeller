"""
g2p.py — Khasi grapheme-to-phoneme (G2P) converter, v2.

Rule-based orthography → IPA for Sohra-standard Khasi, grounded in
Badaplin War (2001), *Ki Sawa Bad Ki Dur Kyntien Jong Ka Ktien Khasi*,
Lynnong III-VI (pp.37-56, incl. the author's own transcription table).

v2 supersedes the stored g2p_v1 output, fixing the systematic errors that
tests/g2p_calibration_war.json exposed (all at 0% under v1):

  • final -h  = /ʔ/         dih → /diʔ/, bah → /baʔ/, lyngdoh → /lɪŋdɔʔ/
  • final -it/-id = /ic/    leit → /lɔic/, buit → /buic/, jakoid → /dzakɔic/
  • lax i in closed σ = /ɪ/ im → /ɪm/, tip → /tɪp/     (open/ -h stays /i/)
  • prevocalic y = /ʔ/      shyiap → /ʃʔiap/
  • y nucleus = /ɪ/         lyngdoh, kynjai, kyrjaw
  • j = /dz/, aspirates = superscript, ñ = /ɲ/, ng = /ŋ/, sh = /ʃ/

NOT modelled (documented, not guessed):
  • phonemic vowel LENGTH — War pp.49-54: contrastive but unmarked in
    spelling and not recoverable from orthography (kam 'work' /kaːm/ vs
    kam 'stride' /kam/). v2 leaves vowels short and flags such entries.
  • fine e/ɛ and o/ɔ quality where the accent is absent — defaults to the
    open-mid value (War's most common), which is right far more often than
    not but is a known residual error source.

Public API:
    to_ipa(word)  -> str   e.g. "dih" -> "/diʔ/"
    convert(word) -> dict   {"surface","generated_by","needs_review",...}
"""
from __future__ import annotations

import json as _json
import os as _os
from pathlib import Path as _Path

# v2.1 diverges from the parent project's g2p_v2 in one rule: the apostrophe
# is silent (elision mark, War p.84) rather than /ʔ/. Bumped so that any IPA
# this writes is distinguishable from the v2 values stored in the lexicon.
VERSION = "g2p_v2.1"

# ── Length / lexical-quality override lexicon ──────────────────────────────
# Vowel length is phonemic but unmarked in Khasi spelling, so it cannot be
# derived — it is listed word-by-word in data/khasi_g2p_length_lexicon.json
# (seeded from War pp.52-56). Consulted BEFORE the rule engine. Editable
# without touching this module.

def _load_length_lexicon() -> dict:
    path = _Path(__file__).parent.parent / "data" / "khasi_g2p_length_lexicon.json"
    try:
        raw = _json.loads(path.read_text(encoding="utf-8"))
        return {k.lower(): v for k, v in (raw.get("entries") or {}).items()}
    except Exception as e:  # pragma: no cover - defensive
        if not _os.environ.get("KHASI_G2P_QUIET"):
            print(f"[khasi-nlp] g2p: could not load length lexicon — {e}")
        return {}


LENGTH_LEXICON: dict = _load_length_lexicon()

# ── Consonant graphemes → IPA (multi-char first) ───────────────────────────
_CONS_DIGRAPHS = {
    "ng": "ŋ", "sh": "ʃ",
    "ph": "pʰ", "th": "tʰ", "kh": "kʰ", "bh": "bʰ", "dh": "dʰ", "jh": "dzʰ",
    "lh": "lʰ", "rh": "rʰ",
}
_CONS_SINGLE = {
    "p": "p", "b": "b", "t": "t", "d": "d", "k": "k", "g": "g",
    "m": "m", "n": "n", "ñ": "ɲ", "l": "l", "r": "r",
    "s": "s", "h": "h", "w": "w", "j": "dz", "c": "c",
    # The apostrophe is an ELISION MARK, not a glottal stop (War p.84):
    # 'tikmie < kti+kmie, ar'ti < ar+kti. It stands for a deleted segment,
    # so it contributes no phoneme of its own — but it is still kind 'C',
    # so the syllable it closes stays closed for the lax-i rule.
    "'": "",
}

# ── Vowel / glide graphemes → IPA (multi-char first) ───────────────────────
# Diphthongs and vowel digraphs. War pp.52-53. Longest match wins.
_VOWEL_CLUSTERS = {
    # triphthongs
    "iau": "iau", "uai": "uai", "ieu": "eu", "iai": "iai", "iaw": "iau",
    # diphthongs / vowel digraphs
    "iew": "eu", "iw": "iu", "aw": "au", "ew": "ɛu", "eu": "ɛu",
    "ai": "ai", "ei": "ɛi", "oi": "ɔi", "ui": "ui", "iu": "iu",
    "ia": "ia", "ie": "e", "ua": "ua", "au": "au", "ou": "ou",
}
_VOWEL_SINGLE = {
    "a": "a", "á": "a",
    "e": "ɛ", "é": "e",
    "i": "i", "í": "i",
    "o": "ɔ", "ó": "o",
    "u": "u", "ú": "u",
    "y": "ɪ", "ý": "ɪ",
    "ï": "i",
}

_VOWEL_CHARS = set("aeiouáéíóúyýï")

# Ordered grapheme list (longest first) for the tokenizer.
_MULTI = sorted(
    list(_CONS_DIGRAPHS) + list(_VOWEL_CLUSTERS),
    key=len, reverse=True,
)


def _tokenise(w: str) -> list[tuple[str, str]]:
    """Split *w* into (grapheme, kind) tokens; kind in {'C','V'}."""
    toks: list[tuple[str, str]] = []
    i = 0
    n = len(w)
    while i < n:
        matched = False
        for g in _MULTI:
            if w.startswith(g, i):
                kind = "C" if g in _CONS_DIGRAPHS else "V"
                toks.append((g, kind))
                i += len(g)
                matched = True
                break
        if matched:
            continue
        ch = w[i]
        if ch in _CONS_SINGLE:
            toks.append((ch, "C"))
        elif ch in _VOWEL_SINGLE:
            toks.append((ch, "V"))
        else:
            toks.append((ch, "?"))  # unknown char passed through
        i += 1
    return toks


def _to_ipa_tokens(toks: list[tuple[str, str]]) -> list[str]:
    out: list[str] = []
    n = len(toks)
    for idx, (g, kind) in enumerate(toks):
        nxt = toks[idx + 1] if idx + 1 < n else None
        prev = toks[idx - 1] if idx > 0 else None
        is_last = idx == n - 1

        if kind == "C":
            if g == "h":
                # final -h (coda) = glottal stop /ʔ/; onset h = /h/.
                if is_last and prev is not None and prev[1] == "V":
                    out.append("ʔ")
                else:
                    out.append("h")
            elif g in ("t", "d"):
                # word-final t/d after an i-offglide → palatal /c/
                # (leit → lɔic, buit → buic, jakoid → dzakɔic).
                if is_last and prev is not None and prev[1] == "V" and prev[0].endswith("i"):
                    out.append("c")
                else:
                    out.append(_CONS_DIGRAPHS.get(g) or _CONS_SINGLE[g])
            else:
                out.append(_CONS_DIGRAPHS.get(g) or _CONS_SINGLE.get(g, g))
            continue

        if kind == "V":
            if g in ("y", "ï"):
                # prevocalic y / ï = glottal or glide before a vowel;
                # nucleus otherwise.
                if nxt is not None and nxt[1] == "V":
                    out.append("ʔ" if g == "y" else "j")
                else:
                    out.append(_VOWEL_SINGLE[g])
                continue
            base = _VOWEL_CLUSTERS.get(g) or _VOWEL_SINGLE.get(g, g)
            # Lax-i rule: a monophthong /i/ closed by a NON-glottal coda
            # consonant becomes /ɪ/ (im → ɪm, tip → tɪp); open or -h keeps
            # /i/ (ki, dih).
            if base == "i":
                coda = toks[idx + 1] if idx + 1 < n else None
                after = toks[idx + 2] if idx + 2 < n else None
                closed_nonglottal = (
                    coda is not None and coda[1] == "C" and coda[0] != "h"
                    and (after is None or after[1] != "V")
                )
                if closed_nonglottal:
                    base = "ɪ"
            out.append(base)
            continue

        out.append(g)  # unknown
    return out


def to_ipa(word: str) -> str:
    """Return the IPA transcription of *word*, delimited with slashes."""
    w = (word or "").strip().lower()
    if not w:
        return "//"
    # Length / lexical-quality override — phonemic length is not derivable,
    # so an attested word takes its listed IPA verbatim (War pp.52-56).
    hit = LENGTH_LEXICON.get(w)
    if hit and hit.get("ipa"):
        return hit["ipa"]
    # Multi-word / hyphenated: transcribe each part, join with the separator.
    if " " in w or "-" in w:
        import re as _re
        parts = _re.split(r"([ \-])", w)
        body = "".join(
            (to_ipa(p).strip("/") if p not in (" ", "-") else p)
            for p in parts
        )
        return f"/{body}/"
    body = "".join(_to_ipa_tokens(_tokenise(w)))
    return f"/{body}/"


def convert(word: str) -> dict:
    """Return a DB-shaped phonology.ipa payload for *word*."""
    surface = to_ipa(word)
    w = (word or "").strip().lower()
    if w in LENGTH_LEXICON:
        # Length/quality is known from the lexicon — no review needed.
        needs_review = False
    else:
        # Flag for review when the word likely carries phonemic length
        # (not yet in the lexicon) — heuristic: a monosyllabic content word
        # closed by a nasal or liquid, where a non-high vowel often lengthens.
        needs_review = _length_suspect(word)
    return {
        "surface": surface,
        "generated_by": VERSION,
        "needs_review": needs_review,
    }


def _length_suspect(word: str) -> bool:
    """Heuristic flag: could this word carry an unmodelled long vowel?
    Conservative — only monosyllables closed by m/n/ng/r/l, which is where
    War's attested length contrasts cluster. Not a transcription, just a
    review hint for a future length-lexicon pass."""
    w = (word or "").strip().lower()
    if not w or " " in w or "-" in w:
        return False
    toks = _tokenise(w)
    n_vowels = sum(1 for _, k in toks if k == "V")
    if n_vowels != 1:
        return False
    return bool(toks) and toks[-1][0] in ("m", "n", "ng", "r", "l")

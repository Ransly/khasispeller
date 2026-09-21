"""
phonology.py — Phase 1: Phonological Validator
Khasi NLP Engine · Ki Sawa Bad Ki Dur Kyntien Jong Ka Ktien Khasi (Badaplin War, 2001)

All phonological constants and display data are loaded from data/khasi_db.json
under the "phonology" key.  No rules, inventories, charts, minimal pairs,
or inferences are hardcoded here — edit the JSON to change any of them.

JSON keys consumed
──────────────────
Validation (existing):
  vowels_simple, digraphs_as_single, aspirates, forbidden_final,
  allowed_final_consonants, allowed_final_digraphs, allowed_final_special,
  allowed_geminates, valid_chars, diphthongs, valid_initial_clusters

Validation (new):
  semantically_invalid_standalone — list of strings that are phonotactically
  valid (e.g. a recognised digraph such as "ng") but carry no independent
  semantic content and must not be accepted as standalone words.

Display (new — added from Khasi Structure HTM):
  consonant_chart          — manner-grouped list of consonant objects
  consonant_total          — int, total phonemic consonants
  consonant_inferences     — list of descriptive strings
  consonant_distribution_notes — list of strings
  minimal_pairs            — dict of named groups → list of pair objects
  vowel_chart              — list of vowel objects
  vowel_contrasts_initial  — list of contrast-set objects
  vowel_contrasts_medial   — list of contrast-set objects
  triphthongs              — list of strings
  vowel_distribution_notes — list of strings
  phonological_processes   — list of process objects
  syllable_structure       — dict with max_onsets, max_codas, nucleus, notes
"""

from __future__ import annotations
import json as _json
import os as _os
import sys as _sys
from pathlib import Path as _Path


# ---------------------------------------------------------------------------
# Internal loader — single source of truth
# ---------------------------------------------------------------------------

def _db_path() -> _Path:
    return _Path(__file__).parent.parent / "data" / "khasi_db.json"


# Cached phonology section. khasi_db.json is ~77 MB, so re-parsing it on
# every /v1/phonology* request blows past Render's 512 MB free-tier RAM
# limit and the worker is OOM-killed mid-request. The phonology block is
# tiny and immutable for the life of the process, so we parse the file
# once at import time and hand back the same dict thereafter.
_PHON_CACHE: dict | None = None


def _load_db() -> dict:
    """Return the cached phonology section from khasi_db.json.

    On Render (DATABASE_URL set) this short-circuits to an empty dict
    instead of parsing the 77 MB JSON file. The module-level constants
    therefore start as empty frozensets; KhasiDB.__init__ then calls
    set_phonology_cache() with the phonology block it received from
    PostgreSQL, which re-binds the constants in place. This eliminates
    a ~300 MB transient peak at import time on the 512 MB free tier.
    """
    global _PHON_CACHE
    if _PHON_CACHE is not None:
        return _PHON_CACHE
    if _os.environ.get("DATABASE_URL"):
        # KhasiDB will inject the phonology block during lifespan startup
        # via set_phonology_cache(). No request can run before then, so
        # the empty starter state is safe.
        _PHON_CACHE = {}
        return _PHON_CACHE
    try:
        with open(_db_path(), encoding="utf-8") as f:
            _PHON_CACHE = _json.load(f).get("phonology", {}) or {}
    except Exception as e:
        print(f"[khasi-nlp] phonology: could not load khasi_db.json — {e}")
        _PHON_CACHE = {}
    return _PHON_CACHE


def set_phonology_cache(phon: dict) -> None:
    """
    Override the cached phonology dict — used when the lexicon is loaded
    from PostgreSQL (DATABASE_URL set) so the phonology endpoints serve
    the same payload as the engine instead of re-parsing the JSON file.

    Also re-binds the module-level frozenset constants (VOWELS, ASPIRATES,
    etc.) so any function that looks them up via the module globals at
    call time sees fresh data, even if it was imported while the cache
    was still empty.
    """
    global _PHON_CACHE
    _PHON_CACHE = dict(phon) if phon else {}
    _rebind_phon_constants()


def _load_phonology() -> dict:
    """
    Build the frozenset constants needed for validation.
    Falls back to minimal hardcoded values only if the JSON file is missing.
    """
    p = _load_db()
    try:
        return {
            "VOWELS":                   frozenset("".join(p.get("vowels_simple", [])) + "ï"),
            "DIGRAPHS_AS_SINGLE":       frozenset(p.get("digraphs_as_single", [])),
            "ASPIRATES":                frozenset(p.get("aspirates", [])),
            "FORBIDDEN_FINAL":          frozenset(p.get("forbidden_final", [])),
            "ALLOWED_FINAL_CONSONANTS": frozenset("".join(p.get("allowed_final_consonants", []))),
            "ALLOWED_FINAL_DIGRAPHS":   frozenset(p.get("allowed_final_digraphs", [])),
            "ALLOWED_FINAL_SPECIAL":    frozenset(p.get("allowed_final_special", [])),
            "ALLOWED_GEMINATES":        frozenset(p.get("allowed_geminates", [])),
            "VALID_CHARS":              frozenset("".join(p.get("valid_chars", []))),
            "DIPHTHONGS":               frozenset(p.get("diphthongs", [])),
            "SEMANTICALLY_INVALID_STANDALONE": frozenset(p.get("semantically_invalid_standalone", [])),
            # Phonotactic exceptions — words that are accepted by Phase 1 even
            # though they appear to violate the "every syllable needs a vowel
            # nucleus" rule (e.g. 'ym', 'yn', 'pyn'). The list lives in the
            # JSON under phonology.phonotactic_exceptions so it is data-driven
            # and editable without touching this module.
            "PHONOTACTIC_EXCEPTIONS":   frozenset(
                w.lower() for w in p.get("phonotactic_exceptions", [])
            ),
            # Loan-diagnostic final consonants (War Ch.IX §3.5): native Khasi
            # words do not end in -l, -s or -j. Advisory warning only.
            "LOAN_FINAL_SIGNALS":       frozenset(
                s.lower() for s in p.get("loan_final_signals", ["l", "s", "j"])
            ),
        }
    except Exception as e:
        print(f"[khasi-nlp] phonology fallback constants — {e}")
        return {
            "VOWELS":                   frozenset("aáeéiíoóuúyýï"),
            "DIGRAPHS_AS_SINGLE":       frozenset(["ng","sh","ph","th","kh","bh","dh","lh","rh","dz"]),
            "ASPIRATES":                frozenset(["ph","th","kh","bh","dh","lh","rh"]),
            "FORBIDDEN_FINAL":          frozenset(["ph","th","kh","bh","dh","lh","rh","dz"]),
            "ALLOWED_FINAL_CONSONANTS": frozenset("pbdtkcsmnlr"),
            "ALLOWED_FINAL_DIGRAPHS":   frozenset(["ng","sh"]),
            "ALLOWED_FINAL_SPECIAL":    frozenset(["ñ"]),
            "ALLOWED_GEMINATES":        frozenset(["ll","nn"]),
            "VALID_CHARS":              frozenset("abcdefghijklmnoprstuwykáéíóúýïñ-'"),
            "DIPHTHONGS":               frozenset(["ai","ái","aw","ei","ew","iw","ie","ui","úi","oi","ou","au"]),
            "SEMANTICALLY_INVALID_STANDALONE": frozenset(["ng"]),
            "PHONOTACTIC_EXCEPTIONS":   frozenset(),
            "LOAN_FINAL_SIGNALS":       frozenset(["l", "s", "j"]),
        }


def _load_clusters() -> frozenset:
    """
    Read valid_initial_clusters from khasi_db.json.
    Glottal-stop prefix sequences are appended at runtime (apostrophe
    cannot be stored cleanly as JSON keys in all editors).
    """
    p = _load_db()
    clusters = set(p.get("valid_initial_clusters", []))

    # Glottal-stop initial sequences — not stored in JSON
    for c in list("rlnmbdks") + ["sh"]:
        clusters.add(f"'{c}")
    for v in list("aeiouý"):
        clusters.add(f"'{v}")

    return frozenset(clusters)


# ---------------------------------------------------------------------------
# Module-level constants (computed once at import time)
# ---------------------------------------------------------------------------

_PHON = _load_phonology()

VOWELS:                   frozenset = _PHON["VOWELS"]
DIGRAPHS_AS_SINGLE:       frozenset = _PHON["DIGRAPHS_AS_SINGLE"]
ASPIRATES:                frozenset = _PHON["ASPIRATES"]
FORBIDDEN_FINAL:          frozenset = _PHON["FORBIDDEN_FINAL"]
ALLOWED_FINAL_CONSONANTS: frozenset = _PHON["ALLOWED_FINAL_CONSONANTS"]
ALLOWED_FINAL_DIGRAPHS:   frozenset = _PHON["ALLOWED_FINAL_DIGRAPHS"]
ALLOWED_FINAL_SPECIAL:    frozenset = _PHON["ALLOWED_FINAL_SPECIAL"]
ALLOWED_GEMINATES:        frozenset = _PHON["ALLOWED_GEMINATES"]
VALID_CHARS:              frozenset = _PHON["VALID_CHARS"]
DIPHTHONGS:               frozenset = _PHON["DIPHTHONGS"]
SEMANTICALLY_INVALID_STANDALONE: frozenset = _PHON["SEMANTICALLY_INVALID_STANDALONE"]
PHONOTACTIC_EXCEPTIONS:   frozenset = _PHON["PHONOTACTIC_EXCEPTIONS"]
LOAN_FINAL_SIGNALS:       frozenset = _PHON["LOAN_FINAL_SIGNALS"]

VALID_INITIAL_CLUSTERS:   frozenset = _load_clusters()

GLOTTAL_STOP: str = "'"
# Word-final coda validation. Set False to restore pre-2026-08-26 behaviour,
# where no coda check ran at all and 'bamsh' validated.
CODA_CHECK_ENABLED = True

# Deliberately NOT used for validation — see the note below.
VALID_INITIALS: frozenset[str] = frozenset(c for c in VALID_CHARS if c not in "ïñ-")

# ── Why three of these constants stay unconsumed ──────────────────────
# They are loaded so the reference endpoints can publish them, not because
# validation should apply them. Each was checked against the lexicon:
#
#   VALID_INITIALS   excludes 'ï', yet 186 entries begin with it (ïabeit,
#                    ïakhublei …). Applying it would reject every ïa- word.
#                    The constant predates the 2026-08-26 spelling
#                    correction and is simply stale.
#
#   FORBIDDEN_FINAL  is fully covered elsewhere and partly wrong. Of its 14
#                    members, 8 are aspirates already rejected by the
#                    aspirate rule, 3 (j, l, s) are loan signals raised as
#                    warnings, 1 (sh) is caught by the coda check below —
#                    and the remaining 2, 'h' and 'w', are contradicted by
#                    838 and 378 attested entries respectively. Final -h is
#                    the glottal stop, final -w a diphthong offglide.
#
#   DIPHTHONGS       describes nuclei for the reference charts. Nothing in
#                    the validator needs to enumerate them, and a rule that
#                    rejected unlisted vowel sequences would misfire across
#                    hyphenated and multi-syllable forms.
#
# ALLOWED_FINAL_CONSONANTS / _DIGRAPHS / _SPECIAL *are* consumed, by the
# coda check in _check_segment.


def _rebind_phon_constants() -> None:
    """Refresh every module-level frozenset by re-running _load_phonology()
    with the current _PHON_CACHE. Called by set_phonology_cache() after
    KhasiDB injects the PG-loaded phonology block, so the constants —
    which were empty at import time on Render — now hold real data.
    Internal functions look these names up via module globals at call
    time, so they see the refreshed values without any consumer change."""
    g = _sys.modules[__name__].__dict__
    g["_PHON"] = _load_phonology()
    g["VOWELS"]                          = g["_PHON"]["VOWELS"]
    g["DIGRAPHS_AS_SINGLE"]              = g["_PHON"]["DIGRAPHS_AS_SINGLE"]
    g["ASPIRATES"]                       = g["_PHON"]["ASPIRATES"]
    g["FORBIDDEN_FINAL"]                 = g["_PHON"]["FORBIDDEN_FINAL"]
    g["ALLOWED_FINAL_CONSONANTS"]        = g["_PHON"]["ALLOWED_FINAL_CONSONANTS"]
    g["ALLOWED_FINAL_DIGRAPHS"]          = g["_PHON"]["ALLOWED_FINAL_DIGRAPHS"]
    g["ALLOWED_FINAL_SPECIAL"]           = g["_PHON"]["ALLOWED_FINAL_SPECIAL"]
    g["ALLOWED_GEMINATES"]               = g["_PHON"]["ALLOWED_GEMINATES"]
    g["VALID_CHARS"]                     = g["_PHON"]["VALID_CHARS"]
    g["DIPHTHONGS"]                      = g["_PHON"]["DIPHTHONGS"]
    g["SEMANTICALLY_INVALID_STANDALONE"] = g["_PHON"]["SEMANTICALLY_INVALID_STANDALONE"]
    g["PHONOTACTIC_EXCEPTIONS"]          = g["_PHON"]["PHONOTACTIC_EXCEPTIONS"]
    g["LOAN_FINAL_SIGNALS"]              = g["_PHON"]["LOAN_FINAL_SIGNALS"]
    g["VALID_INITIAL_CLUSTERS"]          = _load_clusters()
    g["VALID_INITIALS"]                  = frozenset(c for c in g["VALID_CHARS"] if c not in "ïñ-")


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def validate(word: str) -> dict:
    """
    Run all Phase 1 phonotactic checks on *word*.

    Returns
    -------
    dict with keys:
        pass        : bool   — True if no violations found
        errors      : list   — fatal phonotactic violations
        warnings    : list   — advisory notes
        details     : dict   — metadata (cluster found, length, etc.)
    """
    result: dict = {"pass": True, "errors": [], "warnings": [], "details": {}}

    if not word:
        result["pass"] = False
        result["errors"].append("Empty input")
        return result

    # Strip trailing lexicographic punctuation before phonotactic analysis
    word = (word or "").rstrip("!.,")

    # Phonotactic exceptions — words listed in khasi_db.json under
    # phonology.phonotactic_exceptions are accepted as Phase-1-valid even
    # though their syllable structure does not satisfy the standard rules.
    # In Khasi, certain morphemes use a syllabic consonant (most often 'y'
    # acting as /ɨ/) instead of an orthographic vowel — for example
    #   ym /ɨm/  "not"        yn /ɨn/  "shall"
    #   pyn /pɨn/ causative    bym /bɨm/ "that not"
    # Add or remove entries by editing the JSON; no code changes required.
    if word.lower() in PHONOTACTIC_EXCEPTIONS:
        result["details"]["length"] = len(word)
        result["details"]["segments"] = 1
        result["details"]["phonotactic_exception"] = True
        result["details"]["initial_cluster"] = "syllabic consonant nucleus"
        result["details"]["cluster_valid"] = True
        result["warnings"].append(
            f"'{word}' is a documented phonotactic exception — "
            f"syllabic consonant acts as nucleus (see "
            f"phonology.phonotactic_exceptions in khasi_db.json)"
        )
        return result

    # Semantically invalid standalone check — loaded from khasi_db.json
    # These tokens are phonotactically well-formed (e.g. the digraph "ng")
    # but carry no independent semantic content in Khasi.
    if word.lower() in SEMANTICALLY_INVALID_STANDALONE:
        result["pass"] = False
        result["errors"].append(
            f"'{word}' is phonotactically well-formed but has no standalone "
            f"semantic meaning in Khasi — it is a sub-phonemic unit, not a word"
        )
        return result

    segments = word.lower().split("-")
    result["details"]["length"] = len("".join(segments))
    result["details"]["segments"] = len(segments)

    for seg in segments:
        if not seg:
            continue
        seg_errors, seg_warnings, seg_details = _check_segment(seg)
        result["errors"].extend(seg_errors)
        result["warnings"].extend(seg_warnings)
        result["details"].update(seg_details)

    illegal = set(word.lower()) - VALID_CHARS
    if illegal:
        result["pass"] = False
        result["errors"].append(
            f"Illegal characters: {', '.join(sorted(illegal))} — not in Khasi orthography"
        )

    # `g` occurs only as the second half of the digraph `ng`. The character
    # inventory records this itself: valid_chars_note says "'g' is included
    # as a valid character because it appears as the second component of the
    # 'ng' digraph (velar nasal). It does not exist as a standalone Khasi
    # phoneme." Nothing enforced it, so `goh`, `garo`, `gain` and `again` all
    # validated, and English borrowings like `register`, `gospel`, `glass`
    # and `telegram` were accepted as Khasi words.
    #
    # Loanwords and names do carry a bare g — `Garo`, `Meghalaya`,
    # `Guwahati` — but they are handled where names are handled, by the
    # gazetteer and the capitalisation rules in `khasi_spell.foreign`, not
    # by weakening a phonological constraint.
    stray_g = [i for i, ch in enumerate(word.lower())
               if ch == "g" and (i == 0 or word[i - 1].lower() != "n")]
    if stray_g:
        result["pass"] = False
        result["errors"].append(
            f"'g' in '{word}' is not preceded by 'n' — the letter occurs only "
            f"in the digraph 'ng', never as a Khasi phoneme of its own"
        )

    if result["errors"]:
        result["pass"] = False

    return result


def _check_segment(seg: str) -> tuple[list, list, dict]:
    """Check one hyphen-free segment for phonotactic validity."""
    errors: list[str] = []
    warnings: list[str] = []
    details: dict = {}

    check_seg = seg[1:] if seg.startswith("'") else seg
    if not any(c in VOWELS for c in check_seg):
        errors.append(
            f"Segment '{seg}' contains no vowel — "
            f"all Khasi syllables require a vowel nucleus"
        )

    if seg.startswith("'"):
        details["initial_cluster"] = "' (glottal stop)"
        details["cluster_valid"] = True
        details["glottal_stop_initial"] = True
    elif len(seg) >= 2:
        c1, c2 = seg[0], seg[1]
        if c1 not in VOWELS and c2 not in VOWELS:
            cluster2 = c1 + c2
            cluster3 = seg[:3] if len(seg) >= 3 else ""
            cluster4 = seg[:4] if len(seg) >= 4 else ""

            # ── Bug fix: digraphs are SINGLE phonemes, not two-consonant clusters ──
            # A digraph like 'ng', 'sh', 'ph', 'kh', 'th', 'bh', 'dh', 'jh', 'ny'
            # is ONE phoneme written with two letters.  When it appears at the start
            # of a word we must NOT treat the two letters as a consonant cluster.
            # Instead, treat the digraph as a single unit and look at what follows it.
            if cluster2 in DIGRAPHS_AS_SINGLE:
                # cluster2 is a digraph → single phoneme onset, always valid.
                # Now check if there is an additional consonant after the digraph
                # forming a true digraph+consonant cluster (e.g. 'shl', 'khr', 'phr').
                if cluster3 and cluster3[2] not in VOWELS:
                    # Three-character sequence: digraph + consonant (e.g. 'shl', 'khr')
                    if cluster3 in VALID_INITIAL_CLUSTERS:
                        details["initial_cluster"] = cluster3
                        details["cluster_valid"] = True
                    elif cluster4 and cluster4[2] not in VOWELS and cluster4[3] not in VOWELS:
                        # Four-char: digraph + two consonants (extremely rare; reject)
                        errors.append(
                            f"Invalid onset '{cluster4}' in '{seg}' — "
                            f"no Khasi onset begins with a digraph followed by two consonants"
                        )
                    else:
                        # digraph followed by a single consonant not in clusters list
                        errors.append(
                            f"Invalid onset '{cluster3}' — "
                            f"not a permitted Khasi digraph-cluster onset. "
                            f"Valid digraph clusters include: shl, shn, shr, khl, khn, khr, "
                            f"phr, phl, thr, thl, thn …"
                        )
                else:
                    # Digraph alone is the entire onset — always valid
                    details["initial_cluster"] = cluster2
                    details["cluster_valid"] = True
            elif cluster3 in VALID_INITIAL_CLUSTERS:
                details["initial_cluster"] = cluster3
                details["cluster_valid"] = True
            elif cluster2 in VALID_INITIAL_CLUSTERS:
                details["initial_cluster"] = cluster2
                details["cluster_valid"] = True
            else:
                errors.append(
                    f"Invalid initial cluster '{cluster2}' — "
                    f"not a permitted Khasi onset. "
                    f"Valid clusters include: br, bl, bn, pl, pr, pj, pn, "
                    f"kl, km, kn, kp, kr, ks, kt, kw, ky, "
                    f"sl, sm, sn, sp, sb, sd, sk, sr, sw, "
                    f"tr, tl, tm, tn, tb, td, ty, ml, mr, my, "
                    f"jr, jl, jy, rb, rn, ry, lp, lb, ld, wy …"
                )

    if len(seg) >= 2:
        last2 = seg[-2:]
        if last2 in DIGRAPHS_AS_SINGLE:
            pass
        elif last2[-2] not in VOWELS and last2[-1] not in VOWELS:
            if last2 in ALLOWED_GEMINATES:
                details["final_geminate"] = last2
            else:
                errors.append(
                    f"Final consonant cluster '{last2}' in segment '{seg}' — "
                    f"Khasi disallows consonant clusters in word-final position"
                )

    # Final aspirates are phonologically forbidden word-finally (War 2001
    # p.45: loans deaspirate — sukh → suk, dukh → duk). The aspirate
    # digraphs are ph/th/kh/bh/dh/jh/lh/rh; a bare final -h (soh, duh) is
    # the glottal stop /ʔ/ and is NOT one of these, so it is unaffected.
    #
    # NOTE: the previous "saith fix" skipped this check whenever a VOWEL
    # preceded the aspirate — but a final aspirate in CVCʰ always follows a
    # vowel, so the guard silently disabled the rule (bakh/bath validated).
    # War is unambiguous that no native word ends in an aspirate, and a
    # lexicon audit found zero entries ending in an aspirate digraph, so we
    # flag it unconditionally.
    for asp in ASPIRATES:
        if seg.endswith(asp):
            errors.append(
                f"Aspirated consonant '{asp}' in final position of '{seg}' — "
                f"aspirates are phonologically forbidden word-finally in Khasi"
            )
            break

    # Loan-diagnostic final consonants (War Ch.IX §3.5). Native Khasi words
    # never end in -l, -s or -j; when they appear (bol, bus, garaj) the word
    # is almost certainly a loan. This is advisory, not fatal — the word is
    # still well-formed — so it is a WARNING. Data-driven via
    # phonology.loan_final_signals; falls back to nothing if unset.
    _last = seg[-1] if seg else ""
    if _last and _last in LOAN_FINAL_SIGNALS and _last not in VOWELS:
        # Skip when the final is part of an allowed geminate/digraph handled
        # above (none of l/s/j are, but keep the guard explicit).
        warnings.append(
            f"Final '-{_last}' in '{seg}' is not native to Khasi — "
            f"likely a loanword (War Ch.IX: native words do not end in "
            f"-l, -s or -j)"
        )

    # ── Word-final coda validation ────────────────────────────────────
    # FORBIDDEN_FINAL, ALLOWED_FINAL_CONSONANTS, ALLOWED_FINAL_DIGRAPHS and
    # ALLOWED_FINAL_SPECIAL were loaded from the data and never consulted, so
    # no coda check ran at all and 'bamsh' validated. This consumes them.
    #
    # Driven by the ALLOWED lists rather than FORBIDDEN_FINAL, because that
    # list contradicts the lexicon: it names 'h' and 'w' as forbidden, yet
    # 838 entries end in -h and 378 in -w. Both are legitimate — final -h is
    # the glottal stop /ʔ/ (soh, lyngdoh; see the G2P rule) and final -w is a
    # diphthong offglide (ksew /ksɛu/). Taking the list literally would have
    # rejected 1,216 attested words.
    #
    # l, s and j are left to the loan WARNING above: 148 entries end that way
    # and every one inspected is a borrowing (angel, aspatal, baptis, awaj),
    # so they are marked, not rejected.
    if not CODA_CHECK_ENABLED:
        _final_ok = None
    else:
        _final_ok = (
            set(ALLOWED_FINAL_CONSONANTS)
            | set(ALLOWED_FINAL_DIGRAPHS)
            | set(ALLOWED_FINAL_SPECIAL)
            | set(VOWELS)
            | set(LOAN_FINAL_SIGNALS)  # advisory only — warned above
            | {"h", "w"}               # glottal stop / offglide, see above
        )
    if seg and _final_ok is not None:
        _unit = seg[-2:] if seg[-2:] in DIGRAPHS_AS_SINGLE else seg[-1]
        if _unit not in _final_ok:
            errors.append(
                f"Final '-{_unit}' in '{seg}' is not a permitted Khasi coda — "
                f"allowed finals are {', '.join(sorted(ALLOWED_FINAL_CONSONANTS))}, "
                f"{', '.join(sorted(ALLOWED_FINAL_DIGRAPHS))}, "
                f"{', '.join(sorted(ALLOWED_FINAL_SPECIAL))}, a vowel, -h or -w"
            )

    if seg.endswith("dz"):
        errors.append(
            f"Affricate 'dz' in final position of '{seg}' — "
            f"affricates do not occur word-finally in native Khasi"
        )

    if len(seg) > 12:
        warnings.append(
            f"Segment '{seg}' is unusually long — may be a compound or unanalysed sequence"
        )

    return errors, warnings, details


def is_phonotactically_valid(word: str) -> bool:
    """Quick boolean check — convenience wrapper around validate()."""
    return validate(word)["pass"]


# ---------------------------------------------------------------------------
# Display helpers — all data sourced from khasi_db.json["phonology"]
# ---------------------------------------------------------------------------

def get_consonant_inventory() -> dict:
    """
    Return the full consonant inventory for UI display.
    All data is read from khasi_db.json — no hardcoding.

    JSON keys used:
      consonant_chart, consonant_total, consonant_inferences,
      consonant_distribution_notes, minimal_pairs
    """
    p = _load_db()
    chart = p.get("consonant_chart", {})

    # Derive flat list of all consonants from chart groups (for distribution table)
    all_consonants: list[dict] = []
    for group in chart.values():
        all_consonants.extend(group)

    return {
        "total":                    p.get("consonant_total", len(all_consonants)),
        "chart":                    chart,
        "all_consonants":           all_consonants,
        "minimal_pairs":            p.get("minimal_pairs", {}),
        "distribution": {
            "forbidden_final":      p.get("forbidden_final", []),
            "notes":                p.get("consonant_distribution_notes", []),
        },
        "inferences":               p.get("consonant_inferences", []),
    }


def get_vowel_inventory() -> dict:
    """
    Return the full vowel inventory for UI display.
    All data is read from khasi_db.json — no hardcoding.

    JSON keys used:
      vowel_chart, vowel_contrasts_initial, vowel_contrasts_medial,
      diphthongs, triphthongs, vowel_distribution_notes
    """
    p = _load_db()
    return {
        "monophthongs":             p.get("vowel_chart", []),
        "vowel_contrasts_initial":  p.get("vowel_contrasts_initial", []),
        "vowel_contrasts_medial":   p.get("vowel_contrasts_medial", []),
        "diphthongs":               sorted(p.get("diphthongs", [])),
        "triphthongs":              p.get("triphthongs", []),
        "distribution_notes":       p.get("vowel_distribution_notes", []),
    }


def get_phonological_processes() -> dict:
    """
    Return phonological processes and syllable structure for UI display.
    All data is read from khasi_db.json — no hardcoding.

    JSON keys used:
      phonological_processes, syllable_structure, syllable_templates
    """
    p = _load_db()
    syl = p.get("syllable_structure", {})
    # Merge syllable_templates from top-level (pre-existing key) if not in syllable_structure
    if "templates" not in syl:
        syl = dict(syl)
        syl["templates"] = p.get("syllable_templates", [])
    return {
        "processes":         p.get("phonological_processes", []),
        "syllable_structure": syl,
    }


def get_full_phonology_display() -> dict:
    """
    Aggregate all phonological display data for the frontend.
    Every value originates from khasi_db.json — nothing hardcoded.
    """
    p = _load_db()
    return {
        "consonants": get_consonant_inventory(),
        "vowels":     get_vowel_inventory(),
        "processes":  get_phonological_processes(),
        "summary": {
            "total_consonants": p.get("consonant_total", 24),
            "total_vowels":     len(p.get("vowel_chart", [])),
            "diphthongs":       len(p.get("diphthongs", [])),
            "language":         "Khasi",
            "script":           "Latin (romanised)",
            "source":           "khasi_db.json · phonology",
        },
    }

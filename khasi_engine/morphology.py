"""
morphology.py — Phase 2: Morphological Parser
Khasi NLP Engine · Ki Sawa Bad Ki Dur Kyntien Jong Ka Ktien Khasi (Badaplin War, 2001)

Cascade pipeline:
  1. Direct lexicon lookup
  2. Clitic stripping (u, ka, i, ki)
  2b. Suffix stripping (-to, -n, -m — Ch.VII pp.74–78)
  3. Prefix stripping (jing-, pyn-, nong-, men-, shi-, wan-, kyn-, kyr-, byr-, myn-, mar-, ia-)
  4. Infix detection (-n-, -yn-, -p-, -l-, -yr-)
  5. Returns structured parse with root, affixes, and matched entries

Known gaps relative to the textbook (Ch.VII, Lynnong VII):
  - Free morphemes ba, la, iai, nang were previously misregistered as
    derivational prefixes; they have been removed from PREFIXES.
"""

from __future__ import annotations
import json
import os
import re
import sys
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Load morphological rules from khasi_db.json
# ---------------------------------------------------------------------------
# All prefix, infix, suffix, clitic, and assimilation data lives in
# data/khasi_db.json under the "morphology" key.
# Edit the JSON to add new morphological rules — no Python changes needed.
#
# On Render (DATABASE_URL set) the JSON parse is skipped at import time —
# KhasiDB.__init__ calls set_morphology_cache() during lifespan startup
# with the morphology block it loaded from PostgreSQL, and that triggers
# _rebind_morph_constants() which fills the module globals below.

def _load_morphology() -> dict:
    """Load morphology section from khasi_db.json with fallback.
    Short-circuits to {} on Render so KhasiDB's PG-loaded data is the
    single source of truth instead of triggering a redundant 77 MB parse
    at import time (which was a ~300 MB transient OOM trigger)."""
    if os.environ.get("DATABASE_URL"):
        return {}
    db_path = Path(__file__).parent.parent / "data" / "khasi_db.json"
    try:
        with open(db_path, encoding="utf-8") as f:
            raw = json.load(f)
        return raw.get("morphology", {})
    except Exception as e:
        print(f"[khasi-nlp] Warning: could not load morphology from JSON — {e}")
        return {}


def set_morphology_cache(morph: dict) -> None:
    """Inject the morphology block (called by KhasiDB after PG load).
    Re-binds PREFIXES / INFIXES / SUFFIXES / CLITICS in the module
    globals so internal functions see real data on subsequent calls."""
    g = sys.modules[__name__].__dict__
    g["_MORPH"] = dict(morph) if morph else {}
    _rebind_morph_constants()


def _rebind_morph_constants() -> None:
    """Recompute the derived registries from the current _MORPH cache."""
    g = sys.modules[__name__].__dict__
    m = g.get("_MORPH") or {}
    g["PREFIXES"]        = m.get("prefixes", {})
    g["PREFIXES_SORTED"] = sorted(g["PREFIXES"].keys(), key=len, reverse=True)
    g["INFIXES"]         = _build_infixes(m.get("infixes", {}))
    g["SUFFIXES"]        = list(m.get("suffixes", {}).values())
    g["CLITICS"]         = m.get("clitics", {})

    # ── Phase A — three blocks that were already in the JSON but unread:
    #
    # FREE_MORPHEMES: ba/la/iai/nang are independent particles that
    #   terminate derivation (per Phase A §3b rule 4). Augmented with the
    #   spec's wider stop list (lade, ia, ki, bad, ha…) so the engine
    #   refuses to render them as compound components.
    # AFFIX_ORDER: ratified canonical order jing- > nong- > pyn- > sngew-
    #   > ia- > ROOT (linguistic_extraction §2.3, confidence 0.9).
    #   Exposed as a {prefix_bare: rank} map so the stacked-prefix
    #   recursion can refuse out-of-order stacks.
    # DISCOVERED_RULES: 6 semantic constraints (sngew-experiencer,
    #   shi-classifier, mar-distributive, etc.) fed into the meaning-
    #   licensing gate as approximate gloss-class consistency checks.
    free_morph_keys = set((m.get("free_morphemes") or {}).keys())
    # Phase A §3b explicitly names additional function-word terminators
    # that the JSON's free_morphemes block doesn't yet enumerate.
    # Keeping them in code (not data) until the linguist signs off; they
    # are obvious independent words (pronouns + common prepositions).
    g["FREE_MORPHEMES"] = sorted(free_morph_keys | {
        "lade", "ia", "ki", "ka", "u", "i",   # pronouns / proclitics
        "bad", "ha", "na", "sha", "da",       # prepositions
        "ngi", "phi", "nga",                  # 1pl/2pl/1sg pronouns
    })

    order_list = (m.get("affix_order") or {}).get("order") or []
    # `order_list` may be ["jing-", "nong-", "pyn-", "sngew-", "ia-", "ROOT"].
    # Lower rank = outer/peripheral; higher rank = closer to root. We
    # exclude the ROOT sentinel from the map.
    affix_rank: dict[str, int] = {}
    for i, entry in enumerate(order_list):
        if not entry or entry.upper() == "ROOT":
            continue
        bare = entry.rstrip("-").lower()
        affix_rank[bare] = i
    g["AFFIX_RANK"] = affix_rank
    g["AFFIX_ORDER_FLEXIBLE"] = _load_flexible_pairs(m)

    g["DISCOVERED_RULES"] = list(m.get("discovered_rules") or [])


# Minimum length of a morphological root. Khasi content roots are at least a
# CV(C) syllable (>=2 orthographic chars); a single letter is never a root.
# Guards against junk decompositions (iaroh -> ia- + "r") enabled by the
# alphabet placeholder entries in the lexicon. See roadmap item 3.
_MIN_ROOT_LEN = 2

_MORPH = _load_morphology()

# ---------------------------------------------------------------------------
# Prefix registry  — loaded from khasi_db.json morphology.prefixes
# ---------------------------------------------------------------------------
# To add a new prefix, edit data/khasi_db.json — no Python changes needed.

PREFIXES: dict[str, dict] = _MORPH.get("prefixes", {})

# Sorted longest-first (prevents "pyn" matching before "pynleh" etc.)
PREFIXES_SORTED: list[str] = sorted(PREFIXES.keys(), key=len, reverse=True)

# ---------------------------------------------------------------------------
# Infix registry — loaded from khasi_db.json morphology.infixes
# ---------------------------------------------------------------------------
# The JSON stores pattern strings; we compile them to regex here at load time.

def _build_infixes(raw: dict) -> list[dict]:
    result = []
    for _key, _v in raw.items():
        entry = dict(_v)
        pat_str = entry.get("pattern", "")
        if pat_str:
            entry["pattern"] = re.compile(pat_str)
        result.append(entry)
    # Sort: longer markers first (yn before n, yr before r, etc.)
    result.sort(key=lambda x: len(x.get("marker", "")), reverse=True)
    return result

INFIXES: list[dict] = _build_infixes(_MORPH.get("infixes", {}))

# ---------------------------------------------------------------------------
# Derivation helpers
# ---------------------------------------------------------------------------

def _short_function_label(function: str) -> str:
    mapping = {
        "abstract_noun_from_verb_adj": "nominalizer",
        "causative_benefactive_verb": "causative",
        "causative_verb": "causative",
        "agentive_one_who_does": "agentive",
        "agentive_actor_instrument": "agentive",
        "unit_singular_counter": "counter",
        "directional_come": "verb",
        "nominaliser": "nominalizer",
        "nominaliser_variant": "nominalizer",
        "verbal_derivation": "verb",
        "noun_to_verb_or_abstract": "verb",
        "verb_to_noun": "noun",
        "adj_to_abstract_noun": "noun",
        "adverb_manner": "adverb",
        # myn- suffix/prefix labels
        "past_temporal_marker": "temporal adverb",
        "temporal_adverb_from_noun": "temporal adverb",
        # suffix function labels
        "future_tense_marker": "future tense",
        "negation_of_ym": "negation",
        "negation_or_complementizer": "future tense",   # legacy alias → corrected display
        "multiplicative_numeral": "multiplicative",
        "directional_noun_suffix": "directional",
        # feeling/experiential
        "feeling_state_experiential": "experiential",
        # reciprocal
        "reciprocal_plural_subject": "reciprocal",
    }
    return mapping.get(function, function.replace("_", " "))


def _derive_pos_from_prefix(function: str, current_pos: str | None) -> str:
    mapping = {
        "abstract_noun_from_verb_adj": "noun",
        "nominaliser": "noun",
        "nominaliser_variant": "noun",
        "verb_to_noun": "noun",
        "adj_to_abstract_noun": "noun",
        "agentive_one_who_does": "noun",
        "agentive_actor_instrument": "noun",
        "unit_singular_counter": "noun",
        "causative_benefactive_verb": "verb",
        "causative_verb": "verb",
        "directional_come": "verb",
        "verbal_derivation": "verb",
        "noun_to_verb_or_abstract": "verb",
        "adverb_manner": "adverb",
        "past_temporal_marker": "adverb",
        "temporal_adverb_from_noun": "adverb",
        "feeling_state_experiential": "noun",
        "reciprocal_plural_subject": "verb",
    }
    return mapping.get(function, current_pos or "unknown")


def _build_derivational_chain(prefix_chain: list[str], root: str) -> str:
    forms = [root]
    current = root
    for pfx in reversed(prefix_chain):
        current = PREFIXES[pfx]["dash"].rstrip("-") + current
        forms.append(current)
    return " → ".join(forms)


def _infer_root_pos(prefix_chain: list[str]) -> str | None:
    """Infer the most likely root PoS from stacked prefix applicability."""
    candidates: set[str] | None = None
    for pfx in reversed(prefix_chain):
        applies_to = PREFIXES.get(pfx, {}).get("applies_to")
        if not applies_to:
            continue
        allowed = {item.lower() for item in applies_to}
        if candidates is None:
            candidates = allowed
        else:
            candidates &= allowed
        if not candidates:
            break
    if not candidates:
        return None
    for preferred in ["verb", "adjective", "noun", "adverb", "pronoun"]:
        if preferred in candidates:
            return preferred
    return next(iter(candidates))


def _build_pos_evolution(prefix_chain: list[str], root_class: str | None) -> str | None:
    if not root_class:
        root_class = _infer_root_pos(prefix_chain)
    if not root_class:
        return None
    pos_chain = [root_class]
    current_pos = root_class
    for pfx in reversed(prefix_chain):
        current_pos = _derive_pos_from_prefix(PREFIXES[pfx]["function"], current_pos)
        pos_chain.append(current_pos)
    return " → ".join(pos_chain)


def _final_pos(prefix_chain: list[str], root_class: str | None) -> str | None:
    if root_class is None:
        root_class = _infer_root_pos(prefix_chain)
    evolution = _build_pos_evolution(prefix_chain, root_class)
    if not evolution:
        return root_class
    return evolution.split(" → ")[-1].strip()

# ---------------------------------------------------------------------------
# Suffix registry — loaded from khasi_db.json morphology.suffixes
# ---------------------------------------------------------------------------
SUFFIXES: list[dict] = list(_MORPH.get("suffixes", {}).values())


def _suffix_surface_forms(sfx_info: dict) -> list[str]:
    marker = (sfx_info.get("marker") or "").strip("-")
    forms = [marker] if marker else []

    # NOTE: -m no longer auto-expands to 'um'/'im'. War 2001 pp.74-75: -m
    # (from ym 'not') and -n (from yn 'shall') are contractions onto a CLOSED
    # host set (u, i, ka, ki, nga, ngi, phi, pha, me, ba, la), not productive
    # stem suffixes. The vowel of um/im belongs to the host clitic (u+m, i+m),
    # so stripping '-um' off an open-class stem (shait -> shaitum) invented a
    # word. The valid forms (um, im, kam, kin, ...) are whole lexicon entries
    # and resolve by direct match; _host_restricted_ok() gates the rest.
    return sorted(set(forms), key=len, reverse=True)


def _suffix_hosts(sfx_info: dict) -> set[str]:
    """Closed host set for a host-restricted clitic-contraction suffix."""
    return {h.lower() for h in (sfx_info.get("hosts") or [])}


def _host_restricted_ok(sfx_info: dict, stem: str) -> bool:
    """A host-restricted suffix (-n future / -m negation) may only be stripped
    when the remaining stem is one of its closed hosts. Non-restricted
    suffixes always pass."""
    if not sfx_info.get("host_restricted"):
        return True
    return stem.lower() in _suffix_hosts(sfx_info)


def _find_suffix_entry_by_surface(surface: str) -> dict:
    for sfx in SUFFIXES:
        if surface in _suffix_surface_forms(sfx):
            return sfx
    return {}

# ---------------------------------------------------------------------------
# Clitic registry — loaded from khasi_db.json morphology.clitics
# ---------------------------------------------------------------------------
CLITICS: dict[str, dict] = _MORPH.get("clitics", {})

# ---------------------------------------------------------------------------
# Phase A — three blocks that were already in the JSON but unread by the
# engine. See _rebind_morph_constants() for the canonical population path;
# these module-level initializers populate the same globals when the JSON
# load path runs (local dev / non-PG). On Render (DATABASE_URL set) these
# stay empty until set_morphology_cache() triggers _rebind_morph_constants.
# ---------------------------------------------------------------------------
FREE_MORPHEMES: list[str] = sorted(
    set((_MORPH.get("free_morphemes") or {}).keys())
    | {"lade", "ia", "ki", "ka", "u", "i",
       "bad", "ha", "na", "sha", "da",
       "ngi", "phi", "nga"}
)

AFFIX_RANK: dict[str, int] = {
    _entry.rstrip("-").lower(): _i
    for _i, _entry in enumerate(
        (_MORPH.get("affix_order") or {}).get("order") or []
    )
    if _entry and _entry.upper() != "ROOT"
}

# Affix-order flexible pairs — prefixes that may appear in EITHER order
# relative to each other despite their ranks. War (2001) p.72 attests
# jingïapyndom / jingïapyntieng / jingïapynshit alongside jingpynïakhlad:
# "ï hateng hateng u lah ban ia kylliang jaka bad u pyn" (ï[a]- can at
# times exchange places with pyn-). Overridable from the JSON block
# morphology.affix_order.flexible_pairs = [["pyn","ia"], ...].
def _load_flexible_pairs(m: dict) -> frozenset:
    raw = (m.get("affix_order") or {}).get("flexible_pairs")
    pairs = raw if raw else [["pyn", "ia"]]
    out = set()
    for p in pairs:
        if len(p) == 2:
            a, b = p[0].rstrip("-").lower(), p[1].rstrip("-").lower()
            out.add((a, b))
            out.add((b, a))
    return frozenset(out)


AFFIX_ORDER_FLEXIBLE: frozenset = _load_flexible_pairs(_MORPH)

DISCOVERED_RULES: list[dict] = list(_MORPH.get("discovered_rules") or [])


def strict_phase_a() -> bool:
    """Phase A strict-mode gate. Read at call site so toggling
    STRICT_PHASE_A=1 between requests takes effect without restart.

    When True, downstream rules apply:
      • word-boundary stop on stored `compound_roots` whose components
        include a function word in FREE_MORPHEMES
      • affix-order ranks enforced in the stacked-prefix recursion
      • discovered-rules semantic consistency check in the meaning gate

    When False (default), behaviour is unchanged so the verdict
    distribution can be A/B'd against the Phase A baseline.
    """
    return os.environ.get("STRICT_PHASE_A", "").strip().lower() in (
        "1", "true", "yes", "on",
    )

# ---------------------------------------------------------------------------
# Parser
# ---------------------------------------------------------------------------

def parse(word: str, lexicon_lookup) -> dict:
    """
    Run the full morphological parsing cascade on *word*.

    Parameters
    ----------
    word          : str  — the surface form to analyse
    lexicon_lookup : callable(str) -> list[dict]
                    Function that looks up a word in the lexicon and returns
                    a list of matching entries (empty list = not found).
                    Provided by the DB layer so this module stays stateless.

    Returns
    -------
    dict with keys:
        pass      : bool
        status    : str   — "direct_match" | "derived_form" | "recursive_derivation" | "lexicalized_recursive_derivation" | "partial_parse" | "infix_detected" | "suffix_stripped" | "not_found"
        root      : str | None
        layers    : dict  — {prefix, infix, root, clitic} strings present
        matches   : list  — matched lexicon entries
        affix_info: dict  — metadata about identified affixes
                    (includes parse_score — see _score_parse)
        alternatives : list — competing analyses that scored lower
                       (only when the generative ranker ran)
        tree      : dict  — nested derivation tree (outermost affix first)
    """
    result = _parse_impl(word, lexicon_lookup)
    if result.get("pass"):
        result.setdefault("affix_info", {})
        if "parse_score" not in result["affix_info"]:
            result["affix_info"]["parse_score"] = _score_parse(result)
        result["tree"] = _build_parse_tree(result)
    return result


def _parse_impl(word: str, lexicon_lookup) -> dict:
    """Cascade implementation behind parse(). Step 1 (direct lexicon match,
    which carries maintainer-ruled affix annotations) keeps absolute
    priority; the generative strategies (suffix / prefix / stacked / infix /
    phonotactic) are collected as CANDIDATES and ranked by _score_parse,
    per khasi_morphology_nextgen_architecture_upgrade.md §3."""
    result: dict = {
        "pass": False,
        "status": "not_found",
        "root": None,
        "layers": {},
        "matches": [],
        "affix_info": {},
    }

    w = word.lower().strip()

    # ── Step 1: Direct lookup ──────────────────────────────────────────
    matches = lexicon_lookup(w)
    if matches:
        entry = matches[0]
        affix = entry.get("affix_profile", {})
        layers = {}

        # If the DB entry already has affix annotations, use them.
        # Normalise to the dash form from the PREFIXES registry (e.g. "jing" → "jing-").
        def _pfx_dash(raw: str) -> str:
            """Return the dash form for a prefix raw key, e.g. 'jing' → 'jing-'."""
            if raw in PREFIXES:
                return PREFIXES[raw]["dash"]
            return raw + "-" if not raw.endswith("-") else raw

        if affix.get("prefix"):
            layers["prefix"] = _pfx_dash(affix["prefix"])
        if affix.get("prefix2"):
            layers["prefix2"] = _pfx_dash(affix["prefix2"])
        if affix.get("prefix3"):
            layers["prefix3"] = _pfx_dash(affix["prefix3"])
        if affix.get("infix"):
            layers["infix"] = affix["infix"]

        # If NO affix annotation exists, attempt prefix decomposition
        # on the surface form itself. This handles DB entries like
        # jingthoh where surface_form="jingthoh" but affix_profile
        # has no prefix set. We try to detect the prefix structurally.
        # Skip this structural scan when a suffix annotation already exists
        # (e.g. shisien = shi + -sien) — the entry is a suffix form, not a
        # prefix form, so the prefix scan would produce a wrong decomposition.
        # Respect an explicit single-root ruling: when the stored
        # morphology.structure says the word IS its own root (e.g. mareh,
        # jingud, pynam after the 2026-06-09/10 re-roots), do not let the
        # structural scan below re-derive a prefix decomposition the
        # maintainer has explicitly rejected. The flag is computed at
        # flat-alias time in database._enriched_to_flat.
        _is_ruled_single_root = bool(entry.get("single_root_ruled"))

        if (not _is_ruled_single_root
                and not affix.get("prefix") and not affix.get("infix") and not affix.get("suffix")):
            for pfx in PREFIXES_SORTED:
                if w.startswith(pfx) and len(w) > len(pfx) + 1:
                    candidate_root = w[len(pfx):]
                    root_matches = lexicon_lookup(candidate_root)
                    if root_matches:
                        pfx_info = PREFIXES[pfx]
                        # Prefix Productivity Rule — gates 1 & 3.
                        # 1: prefix must be productive (mar-, kyn-, kyr-, etc.
                        #    are fossilised — reject mar+eh, kyn+tien etc.)
                        # 3: root's grammatical class must appear in applies_to.
                        if not pfx_info.get("productive", False):
                            continue
                        applies_to = [t.lower() for t in (pfx_info.get("applies_to") or [])]
                        if applies_to:
                            eligible = [m for m in root_matches
                                        if (m.get("grammatical_class") or "").lower() in applies_to]
                            if not eligible:
                                continue
                            root_matches = eligible
                        layers["prefix"] = pfx_info["dash"]
                        layers["root"] = candidate_root
                        clitic = entry.get("clitic", "none")
                        if clitic and clitic != "none":
                            layers["clitic"] = clitic
                        result.update({
                            "pass": True,
                            "status": "derived_form",
                            "root": candidate_root,
                            "matches": matches,
                            "layers": layers,
                            "affix_info": {
                                "prefix": pfx_info["dash"],
                                "prefix_function": pfx_info["function"],
                                "prefix_function_label": _short_function_label(pfx_info["function"]),
                                "prefix_applies_to": pfx_info["applies_to"],
                                "derivational_chain": _build_derivational_chain([pfx], candidate_root),
                                "pos_evolution": _build_pos_evolution([pfx], matches[0].get("grammatical_class")),
                                "infix": affix.get("infix"),
                                "assimilation_rule": affix.get("assimilation_rule"),
                            },
                        })
                        return result

        # Guard: root_lemma may itself be a prefixed form when the DB
        # morphology.structure has a "root" part that is still prefixed
        # (e.g. jingpynim → structure root raw = "pynim" instead of "im").
        # In that case root_lemma from lemma.root is correct, but if for any
        # reason the shim propagated the contaminated value, clean it here by
        # stripping any leading known prefix until the remainder is a lexicon
        # root or the shortest possible candidate.
        _raw_root = entry.get("root_lemma", w)
        # Prefix Productivity Rule — gate 1 also applies to this cleanup:
        # only strip PRODUCTIVE prefixes. Without the gate this loop was
        # re-deriving fossilised splits the lexicon cleanup had removed
        # (mareh → eh, kynthoh → thoh, pyrkhat → khat, …).
        # Skipped entirely when the stored structure rules the word a
        # single root.
        if not _is_ruled_single_root:
            for _pfx in PREFIXES_SORTED:
                if not PREFIXES[_pfx].get("productive", False):
                    continue
                if _raw_root.startswith(_pfx) and len(_raw_root) > len(_pfx) + 1:
                    _candidate = _raw_root[len(_pfx):]
                    if lexicon_lookup(_candidate):
                        _raw_root = _candidate
                        break
        layers["root"] = _raw_root
        clitic = entry.get("clitic", "none")
        if clitic and clitic != "none": layers["clitic"] = clitic

        # Fix: when the DB root_lemma is contaminated by the prefix
        # (e.g. root_lemma="jingbym-kyrmen" for surface "jingbym-kyrmen"
        # with prefix="jing"), clean it up by stripping the prefix and any
        # interleaved negation/particle ("bym") to expose the real root.
        #
        # Pattern: prefix annotated AND root_lemma starts with that prefix OR
        # the surface form has the structure prefix + [particle-] + root.
        pfx_raw = affix.get("prefix")
        if pfx_raw and (layers.get("root") or "").lower().startswith(pfx_raw.lower()):
            remainder = layers["root"][len(pfx_raw):]  # strip prefix from root_lemma
            # remainder may be "bym-kyrmen" — split on hyphen and take the last segment
            # as the true root. "bym" is a Khasi negation particle.
            parts = [p.strip() for p in remainder.split("-") if p.strip()]
            # Root-minimality gate (item 3): a Khasi root is never a single
            # letter. Without this, iaroh (root_lemma "iar", prefix "ia")
            # collapsed to root "r" — a junk parse enabled by the alphabet
            # placeholder entries. Only apply the strip when the resulting
            # root is >= _MIN_ROOT_LEN; otherwise keep root_lemma intact.
            if len(parts) >= 2 and len(parts[-1]) >= _MIN_ROOT_LEN:
                # Middle parts are negation/particles, last part is the root
                layers["root"] = parts[-1]
                # Surface the negation particle(s) as a separate layer
                if parts[:-1]:
                    layers["negation"] = "-".join(parts[:-1])
            elif len(parts) == 1 and len(parts[0]) >= _MIN_ROOT_LEN:
                layers["root"] = parts[0]
            # Update affix_info root too
            if "affix_info" not in result:
                result["affix_info"] = {}

        # Fix 1: surface compound_roots so all components are visible.
        # e.g. jing-ang-nud has compound_roots=["jing","ang","nud"] — the
        # "nud" component was previously invisible in the morpheme display.
        morph_flags = entry.get("morphological_flags", {})
        compound_roots = morph_flags.get("compound_roots", [])

        # Phase A Rule 4 — word-boundary stop. When strict mode is on and
        # ANY compound component is a function word (independent particle/
        # pronoun/preposition per FREE_MORPHEMES), the stored "compound" is
        # actually a phrase — Phase A §3b worked example: "jingkhwan lade"
        # is stored compound_roots=["jing","khwan","lade"] but "lade" is a
        # reflexive pronoun, so the entry is `[jingkhwan][lade]` (phrase).
        # We refuse to render the broken structure and tag the marker for
        # the verdict layer to downgrade derived → phrase. Default mode
        # preserves the legacy behaviour so verdicts can be A/B'd.
        #
        # The marker is captured here as a local because the function has
        # multiple return paths below (stacked-prefix branch builds a
        # fresh result via _build_stacked_result); we apply it to whichever
        # result actually leaves the function.
        _phrase_downgrade_marker: Optional[dict] = None
        if (strict_phase_a()
                and morph_flags.get("is_compound")
                and len(compound_roots) > 1):
            _free = set(FREE_MORPHEMES)
            _bad = [c for c in compound_roots if (c or "").lower() in _free]
            if _bad:
                _phrase_downgrade_marker = {
                    "reason": "compound has function-word components per Phase A Rule 4",
                    "components": _bad,
                }
                # Suppress compound rendering — caller treats as phrase.
                morph_flags = {**morph_flags, "is_compound": False}
                compound_roots = []

        if morph_flags.get("is_compound") and len(compound_roots) > 1:
            layers["compound_roots"] = compound_roots
            # For hyphenated compounds, try to identify non-prefix, non-root segments
            # as additional morphemes (e.g. "nud" in jing-ang-nud)
            # The prefix is already in affix_profile; root_lemma is the main root.
            # Additional compound roots beyond prefix+root are labelled compound_N.
            root_lemma = entry.get("root_lemma", "")
            prefix_val = (affix or {}).get("prefix") or ""
            for i, cr in enumerate(compound_roots):
                if cr != root_lemma and cr + "-" != prefix_val and cr != (prefix_val or "").rstrip("-"):
                    if cr not in (root_lemma, "jing", "pyn", "nong", "wan", "kyn",
                                  "kyr", "byr", "myn", "mar", "men", "shi", "ia",
                                  "shyn", "ryn", "pyr", "sngew"):
                        layers[f"compound_{i+1}"] = cr

        prefix_chain = [affix.get(k) for k in ("prefix", "prefix2", "prefix3") if affix.get(k)]
        if prefix_chain:
            deriv_result = _build_stacked_result(prefix_chain, layers["root"], matches, lexicalized=True)
            if clitic and clitic != "none":
                deriv_result["layers"]["clitic"] = clitic
            deriv_result["affix_info"].update({
                "infix": affix.get("infix"),
                "assimilation_rule": affix.get("assimilation_rule"),
                "compound_roots": compound_roots if len(compound_roots) > 1 else [],
            })
            if _phrase_downgrade_marker is not None:
                deriv_result["affix_info"]["phrase_downgrade"] = _phrase_downgrade_marker
            return deriv_result

        # Detect partial parse: the entry was found but the affix
        # decomposition is incomplete (affix_profile.prefix annotated but
        # no prefix2 in layers, yet the surface starts with a second known
        # prefix). Emit status="partial_parse" so callers can distinguish
        # a fully-resolved analysis from a silently truncated one.
        _has_outer_pfx = bool(layers.get("prefix"))
        _has_inner_pfx = bool(layers.get("prefix2"))
        _is_partial = False
        if _has_outer_pfx and not _has_inner_pfx:
            _sfx = w  # surface form (already lowercased)
            _outer_raw = (affix.get("prefix") or "").lower()
            _remainder = _sfx[len(_outer_raw):] if _sfx.startswith(_outer_raw) else ""
            for _p in PREFIXES_SORTED:
                if _remainder.startswith(_p) and len(_remainder) > len(_p) + 1:
                    _is_partial = True
                    break

        # Surface suffix annotation from affix_profile (e.g. for lexicalised
        # inflected forms like ngan, kan, shisien, khnang where the DB entry
        # carries the suffix/infix decomposition but no stacked prefix chain).
        _sfx_from_profile = affix.get("suffix")
        _inf_from_profile = affix.get("infix")
        if _sfx_from_profile and "suffix" not in layers:
            layers["suffix"] = _sfx_from_profile
        if _inf_from_profile and "infix" not in layers:
            layers["infix"] = _inf_from_profile

        # Build suffix_function by looking up the suffix marker in the registry
        _sfx_function = None
        _sfx_function_label = None
        if _sfx_from_profile:
            _sfx_info = _find_suffix_entry_by_surface(_sfx_from_profile.lstrip("-"))
            if not _sfx_info:
                # Also try the dash form as key
                _sfx_info = next(
                    (s for s in SUFFIXES if s.get("marker") == _sfx_from_profile),
                    {}
                )
            _sfx_function = _sfx_info.get("function", "")
            _sfx_function_label = _short_function_label(_sfx_function) if _sfx_function else None

        # Build infix_function for lexicalised infix forms (e.g. khnang)
        _inf_function = None
        if _inf_from_profile:
            _inf_entry = next(
                (ix for ix in INFIXES if f"-{ix.get('marker')}-" == _inf_from_profile),
                {}
            )
            _inf_function = _inf_entry.get("function", "")

        _status = "partial_parse" if _is_partial else "direct_match"
        result.update({
            "pass": True,
            "status": _status,
            "root": layers["root"],
            "matches": matches,
            "layers": layers,
            "affix_info": {
                "prefix": affix.get("prefix"),
                "infix":  _inf_from_profile or affix.get("infix"),
                "infix_function": _inf_function,
                "suffix": _sfx_from_profile,
                "suffix_function": _sfx_function,
                "suffix_function_label": _sfx_function_label,
                "assimilation_rule": affix.get("assimilation_rule"),
                "compound_roots": compound_roots if len(compound_roots) > 1 else [],
                **({"parse_warning": "stacked prefix not fully decomposed — affix_profile.prefix2 missing"} if _is_partial else {}),
                **({"phrase_downgrade": _phrase_downgrade_marker} if _phrase_downgrade_marker else {}),
            },
        })
        return result

    # ── Step 2: Strip clitic and retry ───────────────────────────────
    clitic_found = None
    stripped = w
    for clitic in sorted(CLITICS.keys(), key=len, reverse=True):
        if w.startswith(clitic + " "):
            clitic_found = clitic
            stripped = w[len(clitic) + 1:]
            break

    if clitic_found:
        matches = lexicon_lookup(stripped)
        if matches:
            result.update({
                "pass": True,
                "status": "direct_match",
                "root": matches[0]["root_lemma"],
                "matches": matches,
                "layers": {"clitic": clitic_found, "root": stripped},
                "affix_info": {"clitic": CLITICS[clitic_found]},
            })
            return result

    # ── Steps 2b–5: generative candidate collection + ranking ────────
    # Previously a fixed-order cascade (suffix → prefix → stacked →
    # infix → phonotactic-prefix) that returned the FIRST hit. Every
    # strategy now contributes candidates and _score_parse picks the
    # most plausible analysis (khasi_morphology_nextgen §3 "Morphology
    # Ranking"); losing analyses are kept under result["alternatives"].
    candidates: list[dict] = []
    candidates.extend(_gen_suffix_candidates(stripped, lexicon_lookup))
    candidates.extend(_gen_prefix_candidates(stripped, lexicon_lookup))
    stacked = _try_stacked_prefixes(stripped, lexicon_lookup, max_depth=3)
    if stacked is not None:
        candidates.append(stacked)
    candidates.extend(_gen_infix_candidates(stripped, lexicon_lookup))
    candidates.extend(_gen_phonotactic_candidates(stripped, lexicon_lookup))

    if candidates:
        ranked = _rank_candidates(candidates)
        winner = ranked[0]
        if clitic_found:
            winner["layers"]["clitic"] = clitic_found
            if winner.get("status") == "suffix_stripped" or winner["layers"].get("suffix"):
                winner["affix_info"]["clitic"] = clitic_found
        winner["affix_info"]["parse_score"] = _score_parse(winner)
        if len(ranked) > 1:
            winner["alternatives"] = [_compact_alternative(c) for c in ranked[1:5]]
        result.update(winner)
        return result

    # ── Circumfix check (explicit non-support note for Khasi) ───────
    # Khasi does not use circumfixes (Misra 2022; attested=False in DB).
    # If the word is still unresolved, check whether it might superficially
    # resemble a circumfix form and emit a note so callers understand why
    # it was not parsed as such, rather than silently returning not_found.
    _circumfix_data = _MORPH.get("circumfixes", {})
    if not _circumfix_data.get("attested", True):
        result["affix_info"]["circumfix_note"] = (
            "Khasi does not attest circumfixes; "
            "word could not be resolved by any other affix rule."
        )

    # ── Not found ─────────────────────────────────────────────────────
    result["layers"] = {"surface": stripped}
    if clitic_found:
        result["layers"]["clitic"] = clitic_found
    return result


def get_prefix_info(prefix_dash: str) -> Optional[dict]:
    """Return metadata for a given prefix string like 'jing-'."""
    key = prefix_dash.replace("-", "")
    return PREFIXES.get(key)


def _try_stacked_prefixes(
    word: str,
    lexicon_lookup,
    max_depth: int = 3,
) -> Optional[dict]:
    """
    Recursively strip up to *max_depth* prefixes and check if the
    remainder is a known root.

    Khasi allows productive prefix stacking (textbook pp.70-74):
      depth 2: jing- + pyn- + khlain,  jing- + ia- + bam
      depth 3: jing- + pyn- + ia- + bam

    Two tiers of success:
      1. Root found in lexicon → status="recursive_derivation" (strong)
      2. Root phonotactically valid, not in lexicon → status="recursive_derivation"
         (productive derivation, root not yet in DB)

    Returns a result dict if a valid decomposition is found, else None.
    """
    from khasi_engine.phonology import is_phonotactically_valid

    best_phonotactic: Optional[dict] = None  # fallback if no lexicon match

    def _try_suffix_on(word_candidate: str):
        for sfx_info in SUFFIXES:
            for surface in _suffix_surface_forms(sfx_info):
                if word_candidate.endswith(surface) and len(word_candidate) > len(surface) + 1:
                    base_root = word_candidate[:-len(surface)]
                    # Host-restricted -n/-m only strip onto a clitic host.
                    if not _host_restricted_ok(sfx_info, base_root):
                        continue
                    matches = lexicon_lookup(base_root)
                    if not matches:
                        continue
                    applies_to = sfx_info.get("applies_to") or []
                    if applies_to:
                        def _eligible(m):
                            gclass = (m.get("grammatical_class") or "").lower()
                            clitic = (m.get("clitic") or "none").lower()
                            for tgt in applies_to:
                                t = tgt.lower()
                                if t == "clitic" and clitic and clitic != "none":
                                    return True
                                if t == gclass:
                                    return True
                            return False
                        eligible = [m for m in matches if _eligible(m)]
                        if not eligible:
                            continue
                        matches = eligible
                    return {
                        "root": base_root,
                        "matches": matches,
                        "suffix": f"-{surface}",
                        "suffix_info": sfx_info,
                    }
        return None

    def _recurse(remaining: str, prefixes_so_far: list[str], depth: int):
        nonlocal best_phonotactic
        if depth <= 0:
            return None
        for pfx in PREFIXES_SORTED:
            if remaining.startswith(pfx) and len(remaining) > len(pfx) + 1:
                # Prefix Productivity Rule — gate 1 (productive). A
                # fossilised prefix may never enter a stacking chain.
                if not PREFIXES[pfx].get("productive", False):
                    continue
                # Phase A — affix-order rank enforcement (slice 3). In
                # strict mode, refuse to stack a prefix inside an outer
                # prefix when both appear in the canonical-order list
                # AND the candidate's rank is not strictly greater than
                # the outer's. Order from JSON morphology.affix_order:
                # jing(0) → nong(1) → pyn(2) → sngew(3) → ia(4) → ROOT.
                # Rank-unlisted prefixes (men-, kyn-, mar-, byr-, etc.)
                # are unconstrained — the linguist observation only
                # covered the 5 above with confidence 0.9.
                if strict_phase_a() and prefixes_so_far and AFFIX_RANK:
                    _outer = prefixes_so_far[-1]
                    _o_rank = AFFIX_RANK.get(_outer)
                    _i_rank = AFFIX_RANK.get(pfx)
                    if (_o_rank is not None
                            and _i_rank is not None
                            and _i_rank <= _o_rank
                            # War 2001 p.72: ïa- and pyn- attest both
                            # orders (jingpynïakhlad AND jingïapyndom),
                            # so pairs in AFFIX_ORDER_FLEXIBLE are exempt
                            # from the rank check — but never the same
                            # prefix twice in a row.
                            and not (_outer != pfx
                                     and (_outer, pfx) in AFFIX_ORDER_FLEXIBLE)):
                        continue  # out-of-order stacking; skip candidate
                after = remaining[len(pfx):]
                new_chain = prefixes_so_far + [pfx]

                # Tier 1: remainder is a known lexicon root
                matches = lexicon_lookup(after)
                if matches:
                    # applies_to gate: verify every prefix in the chain is
                    # compatible with the root's grammatical class.
                    # e.g. pyn- (applies_to: verb) must not stack onto a
                    # pronoun root. Filter; skip this decomposition if none
                    # of the matches survive.
                    def _stack_eligible(m):
                        gclass = (m.get("grammatical_class") or "").lower()
                        for _pfx in new_chain:
                            at = PREFIXES[_pfx].get("applies_to") or []
                            if at and gclass not in [t.lower() for t in at]:
                                return False
                        return True
                    eligible = [m for m in matches if _stack_eligible(m)]
                    if not eligible:
                        pass  # fall through; try deeper recursion
                    else:
                        return _build_stacked_result(new_chain, after, eligible)

                suffix_result = _try_suffix_on(after)
                if suffix_result is not None:
                    return _build_stacked_result(
                        new_chain,
                        suffix_result["root"],
                        suffix_result["matches"],
                        suffix=suffix_result["suffix"],
                        suffix_info=suffix_result["suffix_info"],
                    )

                # Record best phonotactic match (most prefixes stripped, valid root)
                if (is_phonotactically_valid(after)
                        and len(new_chain) >= 2
                        and (best_phonotactic is None
                             or len(new_chain) > best_phonotactic["_depth"])):
                    best_phonotactic = _build_stacked_result(
                        new_chain, after, [],
                    )
                    best_phonotactic["_depth"] = len(new_chain)

                # Recurse: try stripping another prefix
                deeper = _recurse(after, new_chain, depth - 1)
                if deeper is not None:
                    return deeper
        return None

    result = _recurse(word, [], max_depth)
    if result is not None:
        return result
    # Return phonotactic fallback (stacked prefixes, root not in lexicon)
    if best_phonotactic is not None:
        best_phonotactic.pop("_depth", None)
        return best_phonotactic
    return None


def _build_stacked_result(
    prefix_chain: list[str],
    root: str,
    matches: list[dict],
    lexicalized: bool = False,
    suffix: str | None = None,
    suffix_info: dict | None = None,
) -> dict:
    """
    Build a morphological parse result for a stacked prefix chain.

    Each prefix is labelled prefix, prefix2, prefix3 … in layers so the
    frontend MorphBlock display can show all layers distinctly.
    affix_info includes the full human-readable chain ("jing- + pyn- + ia-")
    and per-prefix function labels for tooltip display.
    """
    layers: dict = {"root": root}
    affix_info: dict = {}

    for i, pfx in enumerate(prefix_chain):
        pfx_info = PREFIXES[pfx]
        # First prefix uses key "prefix"; subsequent use "prefix2", "prefix3" …
        key = "prefix" if i == 0 else f"prefix{i + 1}"
        layers[key] = pfx_info["dash"]
        affix_info[key] = pfx_info["dash"]
        affix_info[f"{key}_function"] = pfx_info["function"]
        affix_info[f"{key}_function_label"] = _short_function_label(pfx_info["function"])
        affix_info[f"{key}_applies_to"] = pfx_info.get("applies_to", [])

    if suffix:
        layers["suffix"] = suffix
        affix_info["suffix"] = suffix
        if suffix_info:
            affix_info["suffix_function"] = suffix_info.get("function")
            affix_info["suffix_function_label"] = _short_function_label(suffix_info.get("function", ""))
            affix_info["suffix_marker"] = suffix_info.get("marker")

    # Human-readable chain: "jing- + pyn- + ia-"
    chain_str = " + ".join(PREFIXES[p]["dash"] for p in prefix_chain)
    root_class = matches[0].get("grammatical_class") if matches else None
    affix_info["prefix_chain"] = chain_str
    affix_info["stack_depth"] = len(prefix_chain)
    affix_info["root_in_lexicon"] = bool(matches)
    affix_info["derivational_chain"] = _build_derivational_chain(prefix_chain, root)
    affix_info["pos_evolution"] = _build_pos_evolution(prefix_chain, root_class)
    affix_info["derived_pos"] = _final_pos(prefix_chain, root_class)
    affix_info["prefix_functions"] = " ".join(
        f"{PREFIXES[p]['dash']} {_short_function_label(PREFIXES[p]['function'])}"
        for p in prefix_chain
    )

    status = "derived_form" if len(prefix_chain) == 1 else "recursive_derivation"
    if lexicalized and len(prefix_chain) > 1:
        status = "lexicalized_recursive_derivation"

    return {
        "pass": True,
        "status": status,
        "root": root,
        "matches": matches,
        "layers": layers,
        "affix_info": affix_info,
    }


# ---------------------------------------------------------------------------
# Generative candidate collectors (Steps 2b–5 of the old cascade)
# ---------------------------------------------------------------------------
# Each returns a LIST of parse-result dicts instead of returning early, so
# _parse_impl can rank all competing analyses with _score_parse. The gating
# logic (productivity, applies_to, surface-form checks) is unchanged from
# the cascade versions.

def _gen_suffix_candidates(stripped: str, lexicon_lookup) -> list[dict]:
    """Suffix stripping (Ch.VII pp.74–78) as a candidate generator."""
    out: list[dict] = []
    markers = sorted(
        {surface for s in SUFFIXES for surface in _suffix_surface_forms(s)},
        key=len, reverse=True,
    )
    for sfx in markers:
        if not (stripped.endswith(sfx) and len(stripped) > len(sfx) + 1):
            continue
        candidate_root = stripped[:-len(sfx)]
        sfx_info = _find_suffix_entry_by_surface(sfx)
        # Host-restricted clitic contractions (-n future / -m negation) may
        # only strip when the stem is a closed clitic host (u, ka, nga, …).
        # Blocks shait -> shait+(-m) style over-generation (item 5).
        if not _host_restricted_ok(sfx_info, candidate_root):
            continue
        matches = lexicon_lookup(candidate_root)
        if not matches:
            continue
        applies_to = sfx_info.get("applies_to") or []
        if applies_to:
            def _eligible(m):
                gclass = (m.get("grammatical_class") or "").lower()
                # Also handle new-schema entries where grammar.pos carries the PoS
                pos = (m.get("grammar", {}).get("pos") or "").lower()
                clitic = (
                    (m.get("clitic") or "")
                    or (m.get("grammar", {}).get("clitic") or "")
                    or "none"
                ).lower()
                for tgt in applies_to:
                    t = tgt.lower()
                    if t == "clitic" and clitic and clitic != "none":
                        return True
                    if t == "pronoun" and (gclass == "pronoun" or pos == "pron"):
                        return True
                    if t == "numeral" and (
                        gclass in ("numeral", "num") or pos in ("num", "numeral", "num_card")
                    ):
                        return True
                    if t == gclass:
                        return True
                return False
            eligible = [m for m in matches if _eligible(m)]
            if not eligible:
                continue
            matches = eligible

        # If the stripped root itself is a prefixed/derived lexicon entry,
        # preserve its internal affix decomposition and attach the suffix.
        emitted_stacked = False
        for match in matches:
            affix = match.get("affix_profile", {})
            prefix_chain = [affix.get(k) for k in ("prefix", "prefix2", "prefix3") if affix.get(k)]
            if prefix_chain:
                root_lemma = match.get("root_lemma") or candidate_root
                out.append(_build_stacked_result(
                    prefix_chain,
                    root_lemma,
                    [match],
                    suffix=f"-{sfx}",
                    suffix_info=sfx_info,
                ))
                emitted_stacked = True
                break
        if emitted_stacked:
            continue

        out.append({
            "pass": True,
            "status": "suffix_stripped",
            "root": candidate_root,
            "matches": matches,
            "layers": {"suffix": f"-{sfx}", "root": candidate_root},
            "affix_info": {
                "suffix": f"-{sfx}",
                "suffix_function": sfx_info.get("function", ""),
                "suffix_marker": sfx_info.get("marker"),
            },
        })
    return out


def _gen_prefix_candidates(stripped: str, lexicon_lookup) -> list[dict]:
    """Single-prefix stripping (old Step 3) as a candidate generator."""
    out: list[dict] = []
    for pfx in PREFIXES_SORTED:
        if not (stripped.startswith(pfx) and len(stripped) > len(pfx) + 1):
            continue
        candidate_root = stripped[len(pfx):]
        matches = lexicon_lookup(candidate_root)
        if not matches:
            continue
        pfx_info = PREFIXES[pfx]
        # Prefix Productivity Rule — gate 1 (productive).
        if not pfx_info.get("productive", False):
            continue
        # applies_to gate — a prefix that only applies to verbs must not be
        # accepted on a root whose grammatical class is, e.g., pronoun.
        applies_to = pfx_info.get("applies_to") or []
        if applies_to:
            def _pfx_eligible(m):
                gclass = (m.get("grammatical_class") or "").lower()
                for tgt in applies_to:
                    if tgt.lower() == gclass:
                        return True
                return False
            eligible = [m for m in matches if _pfx_eligible(m)]
            if not eligible:
                continue
            matches = eligible
        out.append({
            "pass": True,
            "status": "derived_form",
            "root": candidate_root,
            "matches": matches,
            "layers": {"prefix": pfx_info["dash"], "root": candidate_root},
            "affix_info": {
                "prefix": pfx_info["dash"],
                "prefix_function": pfx_info["function"],
                "prefix_function_label": _short_function_label(pfx_info["function"]),
                "prefix_applies_to": pfx_info["applies_to"],
                "derivational_chain": _build_derivational_chain([pfx], candidate_root),
                "pos_evolution": _build_pos_evolution([pfx], matches[0].get("grammatical_class")),
                "derived_pos": _final_pos([pfx], matches[0].get("grammatical_class")),
            },
        })
        # …and keep going, because a root can itself be transparently
        # derived. `jingïakhlad` stopped at jing- + ïakhlad, while its own
        # siblings read jing- + ïa- + khleh and pyn- + ïa- + khlad — the
        # difference being only that THEY have lexicon entries and reach the
        # lexicalised stacked path, which a productive form never does. So
        # the same stacking is offered here.
        #
        # The gates are the ones the single-prefix case already applies, now
        # to the inner layer: the second prefix must be productive, its
        # `applies_to` must match the inner root's word class, and that inner
        # root must be a lexicon entry in its own right. A root that merely
        # begins with prefix-shaped letters does not qualify.
        #
        # Nothing is dropped: the one-affix reading stays a candidate. The
        # scorer prefers the deeper one on its own terms (+0.5 per affix over
        # a lexicon-confirmed root, less 0.05 for the extra layer), so the
        # ranking is left alone.
        for inner_pfx in PREFIXES_SORTED:
            if not (candidate_root.startswith(inner_pfx)
                    and len(candidate_root) > len(inner_pfx) + 1):
                continue
            inner_info = PREFIXES[inner_pfx]
            if not inner_info.get("productive", False):
                continue
            inner_root = candidate_root[len(inner_pfx):]
            inner_matches = lexicon_lookup(inner_root)
            if not inner_matches:
                continue
            inner_applies = inner_info.get("applies_to") or []
            if inner_applies:
                inner_matches = [
                    m for m in inner_matches
                    if (m.get("grammatical_class") or "").lower()
                    in {t.lower() for t in inner_applies}
                ]
                if not inner_matches:
                    continue
            out.append(_build_stacked_result(
                [pfx, inner_pfx], inner_root, inner_matches))
    return out


# Digraphs count as one consonant. Used only if the phonology module hands
# back an empty set, which happens when the database has not been loaded.
_FALLBACK_DIGRAPHS = frozenset(
    ["ng", "sh", "ph", "th", "kh", "bh", "dh", "lh", "rh", "dz"])


def _first_consonant_unit(word: str) -> str:
    """
    The first consonant of *word*, counting a declared digraph such as "sh"
    or "ng" as the single consonant it is.
    """
    from khasi_engine.phonology import DIGRAPHS_AS_SINGLE
    digraphs = DIGRAPHS_AS_SINGLE or _FALLBACK_DIGRAPHS
    return word[:2] if word[:2].lower() in digraphs else word[:1]


def _gen_infix_candidates(stripped: str, lexicon_lookup) -> list[dict]:
    """Infix detection (old Step 4) as a candidate generator.

    Every infix in the database declares position "after_C1", and every
    example it ships puts the marker after a single consonant: ker->kper,
    sad->snad, shong->shnong, shur->shlur, kjat->kynjat. The compiled
    pattern cannot enforce that on its own, because its leading
    [^vowel]+ group happily swallows a whole cluster. That let a
    misspelling in: "slpa" matched as "sl" + -p- + "a", rebuilt the root
    "sla", found it in the lexicon, and scored morphology 2 + phonotactic
    1 = the acceptance threshold. Khasi does not infix after a two
    consonant onset, so the marker is now required to sit immediately
    after the first consonant unit.
    """
    out: list[dict] = []
    for infix_def in INFIXES:
        if not infix_def["pattern"].match(stripped):
            continue
        marker = infix_def.get("marker") or ""
        c1 = _first_consonant_unit(stripped)
        if not c1 or not stripped[len(c1):].startswith(marker):
            continue
        rest = stripped[len(c1) + len(marker):]
        if not rest:
            continue
        candidate_root = c1 + rest
        matches = lexicon_lookup(candidate_root)
        if not matches:
            continue
        out.append({
            "pass": True,
            "status": "infix_detected",
            "root": candidate_root,
            "matches": matches,
            "layers": {"infix": f"-{infix_def['marker']}-", "root": candidate_root},
            "affix_info": {
                "infix": f"-{infix_def['marker']}-",
                "infix_function": infix_def["function"],
                "infix_examples": infix_def["examples"],
                "c1": c1,
                "remainder": rest,
            },
        })
    return out


def _gen_phonotactic_candidates(stripped: str, lexicon_lookup) -> list[dict]:
    """Phonotactic prefix acceptance (old Step 5) as a candidate generator.

    Accepts productive-prefix + phonotactically-valid remainder even when
    the remainder is not in the lexicon (Khasi prefixes are productive).
    These candidates carry root_in_lexicon=False and score lowest, so a
    lexicon-rooted analysis always outranks them.
    """
    from khasi_engine.phonology import is_phonotactically_valid as _phono_valid
    out: list[dict] = []
    for pfx in PREFIXES_SORTED:
        if not (stripped.startswith(pfx) and len(stripped) > len(pfx) + 1):
            continue
        candidate_root = stripped[len(pfx):]
        if not _phono_valid(candidate_root):
            continue
        pfx_info = PREFIXES[pfx]
        # Productive gate only — no applies_to check because the stem
        # isn't in the lexicon (this is the phonotactic-only fallback).
        if not pfx_info.get("productive", False):
            continue
        out.append({
            "pass": True,
            "status": "derived_form",
            "root": candidate_root,
            "matches": [],
            "layers": {"prefix": pfx_info["dash"], "root": candidate_root},
            "affix_info": {
                "prefix": pfx_info["dash"],
                "prefix_function": pfx_info["function"],
                "prefix_function_label": _short_function_label(pfx_info["function"]),
                "prefix_applies_to": pfx_info["applies_to"],
                "root_in_lexicon": False,
                "derivational_chain": _build_derivational_chain([pfx], candidate_root),
                "derived_pos": _final_pos([pfx], None),
            },
        })
    return out


# ---------------------------------------------------------------------------
# Ranking — khasi_morphology_nextgen_architecture_upgrade.md §3
# ---------------------------------------------------------------------------

def _score_parse(result: dict) -> float:
    """Score a parse candidate for plausibility ranking.

    Heuristics (adapted from the nextgen upgrade doc §3):
      +2.0  exact surface form is a lexicon entry (direct_match/partial_parse)
      +1.0  root found in lexicon
      -0.5  root NOT in lexicon (phonotactic-only derivation)
      +0.5  per affix layer when the root is lexicon-confirmed
      +0.2  per affix layer when it is not (weaker evidence)
      +0.4  PoS-compatibility verified against lexicon entries
      +0.3  productive prefix chain (only productive prefixes are generated)
      -0.3  partial_parse penalty (annotation known incomplete)
      -0.05 per affix beyond the first (prefer the simplest analysis)

    A direct lexicon match therefore always outranks any derivation of the
    same surface (the kynja rule: never prefer kyn+ja over the entry kynja),
    and lexicon-rooted derivations always outrank phonotactic-only ones.
    """
    ai = result.get("affix_info") or {}
    layers = result.get("layers") or {}
    status = result.get("status") or ""

    root_in_lex = ai.get("root_in_lexicon")
    if root_in_lex is None:
        root_in_lex = bool(result.get("matches"))

    n_prefixes = sum(1 for k in layers if k == "prefix" or (k.startswith("prefix") and k[6:].isdigit()))
    n_affixes = n_prefixes + (1 if layers.get("suffix") else 0) + (1 if layers.get("infix") else 0)

    score = 0.0
    if status in ("direct_match", "partial_parse"):
        score += 2.0
    if root_in_lex:
        score += 1.0
        score += 0.5 * n_affixes
        if n_affixes and result.get("matches"):
            score += 0.4
    else:
        score -= 0.5
        score += 0.2 * n_affixes
    if n_prefixes:
        score += 0.3
    if status == "partial_parse":
        score -= 0.3
    score -= 0.05 * max(0, n_affixes - 1)
    return round(score, 3)


def _rank_candidates(candidates: list[dict]) -> list[dict]:
    """Dedupe and sort candidates: best analysis first.

    Tie-breaks after score: lexicon-rooted first, then fewer morphemes
    (Occam), then longer root. Deterministic for stable API output.
    """
    seen: set = set()
    unique: list[dict] = []
    for c in candidates:
        layers = c.get("layers") or {}
        key = (
            c.get("status"),
            c.get("root"),
            tuple(sorted((k, str(v)) for k, v in layers.items())),
        )
        if key in seen:
            continue
        seen.add(key)
        unique.append(c)

    def _key(c: dict):
        ai = c.get("affix_info") or {}
        layers = c.get("layers") or {}
        root_in_lex = ai.get("root_in_lexicon")
        if root_in_lex is None:
            root_in_lex = bool(c.get("matches"))
        n_morph = sum(
            1 for k in layers
            if k == "prefix" or (k.startswith("prefix") and k[6:].isdigit())
            or k in ("suffix", "infix")
        )
        return (-_score_parse(c), not root_in_lex, n_morph, -len(c.get("root") or ""))

    unique.sort(key=_key)
    return unique


def _compact_alternative(c: dict) -> dict:
    """Small serialisable summary of a losing candidate for result["alternatives"]."""
    layers = c.get("layers") or {}
    ai = c.get("affix_info") or {}
    parts = [layers[k].rstrip("-") for k in ("prefix", "prefix2", "prefix3") if layers.get(k)]
    structure = " + ".join(parts + [c.get("root") or ""])
    if layers.get("infix"):
        structure += f" (infix {layers['infix']})"
    if layers.get("suffix"):
        structure += f" + {layers['suffix']}"
    root_in_lex = ai.get("root_in_lexicon")
    if root_in_lex is None:
        root_in_lex = bool(c.get("matches"))
    return {
        "status": c.get("status"),
        "root": c.get("root"),
        "structure": structure,
        "score": _score_parse(c),
        "root_in_lexicon": bool(root_in_lex),
    }


def _build_parse_tree(result: dict) -> dict:
    """Nested derivation tree (nextgen upgrade doc §4), outermost affix on top.

    Example for jingpynim:
        {"prefix": "jing-", "function": "...", "pos": "noun",
         "child": {"prefix": "pyn-", "function": "...", "pos": "verb",
                   "child": {"root": "im", "pos": "verb"}}}
    Suffix/infix attach on the root node (they are root-level operations in
    all attested Khasi derivations — no affix scopes over a suffix).
    """
    layers = result.get("layers") or {}
    matches = result.get("matches") or []
    ai = result.get("affix_info") or {}
    status = result.get("status") or ""

    root = result.get("root") or layers.get("root") or layers.get("surface")
    root_pos = (matches[0].get("grammatical_class") or None) if matches else None

    # For direct-family results the matches describe the SURFACE word, not
    # the root — its PoS must not be copied onto the leaf. Infer the root
    # PoS from the prefix chain's applies_to instead (e.g. jingpynim's
    # leaf im is a verb, not a noun).
    _prefix_chain_bare = [
        str(layers[k]).rstrip("-") for k in ("prefix", "prefix2", "prefix3")
        if layers.get(k)
    ]
    if _prefix_chain_bare and status in (
        "direct_match", "partial_parse", "lexicalized_recursive_derivation",
    ):
        root_pos = _infer_root_pos(_prefix_chain_bare)

    node: dict = {"root": root}
    if root_pos:
        node["pos"] = root_pos
    if layers.get("infix"):
        node["infix"] = layers["infix"]
        if ai.get("infix_function"):
            node["infix_function"] = ai["infix_function"]
    if layers.get("suffix"):
        node["suffix"] = layers["suffix"]
        if ai.get("suffix_function"):
            node["suffix_function"] = ai["suffix_function"]

    current_pos = root_pos
    # layers stores prefixes outermost-first (prefix, prefix2, prefix3);
    # wrap the leaf starting from the innermost.
    for key in ("prefix3", "prefix2", "prefix"):
        dash = layers.get(key)
        if not dash:
            continue
        pinfo = PREFIXES.get(str(dash).rstrip("-"), {})
        function = pinfo.get("function", "")
        current_pos = _derive_pos_from_prefix(function, current_pos) if function else current_pos
        node = {
            "prefix": dash,
            "function": function or None,
            "pos": current_pos,
            "child": node,
        }
    if layers.get("clitic"):
        node = {"clitic": layers["clitic"], "child": node}
    return node

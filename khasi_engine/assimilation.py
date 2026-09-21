"""
assimilation.py — Phase 3: Phonological Assimilation Engine
Khasi NLP Engine · Ki Sawa Bad Ki Dur Kyntien Jong Ka Ktien Khasi (Badaplin War, 2001)

Implements the documented assimilation rules:
  1. n_to_l           : pyn- + l-initial root → geminate pyll-   (pyllait, pyllam)
  2. n_to_l_wan       : wan- + l-initial root → wall-            (wallam)
  3. ng_to_n_alveolar : jing- + alveolar stop (t/d) → jyn-       (jyntah, jyntrei, jyntip)
  4. velar_nasal_to_l : velar-nasal coda + l-initial root → geminate ll
                        documented example: Khon + lung → Khillung (p.57 ex.1)

Known limitations:
  - Exception handling uses hardcoded word lists (e.g. ["pynleh", "pynlong"]).
    The textbook implies exceptions arise from specific phonological environments
    (the /n/ is retained when the following cluster prevents gemination), not from
    arbitrary lexical lists.  A phonological-condition approach would be more
    principled but requires deeper syllable-structure analysis.

Also exposes a reverse-lookup: given a surface form, identify which rule
fired and reconstruct the underlying prefix + root decomposition.
"""

from __future__ import annotations
import json
import os
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# Load assimilation rules from khasi_db.json
# ---------------------------------------------------------------------------
# On Render (DATABASE_URL set) the JSON parse is skipped at import time;
# KhasiDB calls set_assimilation_rules_cache() during lifespan startup
# with the rules it received from PostgreSQL. Internal functions read
# ASSIMILATION_RULES via module globals at call time so they pick up
# the refreshed list on next invocation.

def _load_assimilation_rules() -> list[dict]:
    """Load assimilation_rules from khasi_db.json morphology section.
    Short-circuits to [] on Render to avoid a redundant ~300 MB
    transient parse of the 77 MB JSON file."""
    if os.environ.get("DATABASE_URL"):
        return []
    db_path = Path(__file__).parent.parent / "data" / "khasi_db.json"
    try:
        with open(db_path, encoding="utf-8") as f:
            raw = json.load(f)
        return raw.get("morphology", {}).get("assimilation_rules", [])
    except Exception as e:
        print(f"[khasi-nlp] Warning: could not load assimilation rules — {e}")
        return []


def set_assimilation_rules_cache(rules: list[dict]) -> None:
    """Inject the rules list (called by KhasiDB after PG load).
    Re-binds ASSIMILATION_RULES + _RULES_BY_SURFACE so callers see the
    real data on subsequent invocations."""
    g = sys.modules[__name__].__dict__
    g["ASSIMILATION_RULES"] = list(rules) if rules else []
    g["_RULES_BY_SURFACE"]  = {
        r["surface_prefix"]: r
        for r in g["ASSIMILATION_RULES"]
        if r.get("surface_prefix") is not None
    }


# ---------------------------------------------------------------------------
# Rule registry — loaded from khasi_db.json morphology.assimilation_rules
# ---------------------------------------------------------------------------
# To add or modify assimilation rules, edit data/khasi_db.json.

ASSIMILATION_RULES: list[dict] = _load_assimilation_rules()

# Fast lookup by surface_prefix — only prefix-based rules have a surface_prefix
_RULES_BY_SURFACE: dict[str, dict] = {
    r["surface_prefix"]: r
    for r in ASSIMILATION_RULES
    if r.get("surface_prefix") is not None
}


# ---------------------------------------------------------------------------
# Forward direction: check if a word should trigger assimilation
# ---------------------------------------------------------------------------

def should_assimilate(prefix: str, root: str) -> tuple[bool, str | None]:
    """
    Given a prefix (e.g. 'pyn-') and a root (e.g. 'lait'), determine
    whether assimilation applies and return the expected surface form.

    Returns
    -------
    (True,  surface_form)  — assimilation applies, use surface_form
    (False, None)          — no assimilation for this combination
    """
    for rule in ASSIMILATION_RULES:
        if rule["trigger_prefix"] == prefix:
            if root and root[0] in rule["trigger_initials"]:
                # Exceptions are stored as UNDERLYING forms (pynleh, pynloit,
                # pynlong) — the words where assimilation does NOT apply, so
                # their surface == underlying. The previous code compared the
                # assimilated surface (pylleh) against the exception list and
                # therefore never matched, forcing pynleh → pylleh. Compare
                # the underlying prefix+root instead.
                underlying = prefix.rstrip("-") + root
                if underlying in rule.get("exceptions", []):
                    return False, None
                surface = rule["surface_prefix"] + root
                return True, surface
    return False, None


# ---------------------------------------------------------------------------
# Reverse direction: given a surface form, detect which rule fired
# ---------------------------------------------------------------------------

def check(word: str, lexicon_lookup) -> dict:
    """
    Scan *word* against all assimilation rules and attempt to reverse-engineer
    the underlying prefix + root.

    Parameters
    ----------
    word           : str
    lexicon_lookup : callable(str) -> list[dict]

    Returns
    -------
    dict with keys:
        rule_detected        : str | None
        rule_description     : str | None
        reconstructed_prefix : str | None
        reconstructed_root   : str | None
        surface_prefix       : str | None
        details              : list[str]
        examples             : list[dict]
    """
    result: dict = {
        "rule_detected": None,
        "rule_description": None,
        "reconstructed_prefix": None,
        "reconstructed_root": None,
        "surface_prefix": None,
        "details": [],
        "examples": [],
    }

    w = word.lower().strip()

    # Check against each rule's surface_prefix.
    # Rules without a surface_prefix (e.g. velar_nasal_to_l, which fires at a
    # compound boundary) cannot be detected by surface-prefix scanning — skip them.
    for rule in ASSIMILATION_RULES:
        sp = rule.get("surface_prefix")
        if sp is None:
            continue
        if not w.startswith(sp):
            continue

        # Bail out for known exceptions
        if w in rule.get("exceptions", []):
            continue

        after_surface_prefix = w[len(sp):]
        if not after_surface_prefix:
            continue

        # Reconstruct the root: for n_to_l rules the first char of the
        # root is the same as the assimilated consonant
        candidate_root = _reconstruct_root(rule, after_surface_prefix)
        if not candidate_root:
            continue

        # Confirm root exists in lexicon (strong signal this rule fired)
        matches = lexicon_lookup(candidate_root)
        root_confirmed = bool(matches)

        result.update({
            "rule_detected": rule["id"],
            "rule_description": rule["description"],
            "reconstructed_prefix": rule["trigger_prefix"],
            "reconstructed_root": candidate_root,
            "surface_prefix": sp,
            "details": [
                f"Surface '{w}' → {rule['trigger_prefix']} + '{candidate_root}' "
                f"({'root confirmed in lexicon' if root_confirmed else 'root not in lexicon — may be OOV'})"
            ],
            "examples": rule["examples"],
            "root_confirmed": root_confirmed,
        })
        return result

    return result


def _reconstruct_root(rule: dict, after_surface_prefix: str) -> str | None:
    """
    Given the rule that fired and the string after the surface prefix,
    reconstruct the original root.

    For n_to_l (pyl + lait → root = lait):
        The surface prefix 'pyl' ends with 'l'. The root starts with 'l'.
        In 'pyllait', after 'pyl' we get 'lait' which IS the root.

    For ng_to_n_alveolar (jyn + tah → root = tah):
        After 'jyn' we get 'tah' which IS the root directly.

    For velar_nasal_to_l:
        This rule fires at a compound boundary, not a prefix–root boundary.
        Reconstruction requires Phase 4 (compound detection) first.
        This branch returns None because check() is not the right entry
        point for compound-boundary assimilation.
    """
    rid = rule["id"]

    if rid in ("n_to_l", "n_to_l_wan"):
        # Both rules GEMINATE: pyn- + lait → pyl|lait, wan- + lam → wal|lam.
        # The surface prefix ('pyl' / 'wal') consumes only the FIRST half of
        # the geminate, so whatever follows must still begin with the root's
        # own 'l'. That remainder is the root.
        #
        # The check matters. Without it any word starting 'pyl' or 'wal' is
        # split after three characters and the tail treated as a root, so
        # 'pyleng' reconstructs as pyn- + 'eng' — and because 'eng' is a real
        # lexicon word the parse is confirmed and the misspelling is accepted.
        # The correct form is 'pylleng', with the geminate.
        if not after_surface_prefix.startswith("l"):
            return None
        return after_surface_prefix

    if rid == "ng_to_n_alveolar":
        # jyn + tah → root = tah; after 'jyn' we have 'tah'
        if after_surface_prefix and after_surface_prefix[0] in rule["trigger_initials"]:
            return after_surface_prefix

    if rid == "velar_nasal_to_l":
        # Compound-boundary rule — cannot reconstruct from prefix alone.
        # Handled by Phase 4 (complex.py) compound detection.
        return None

    return None


# ---------------------------------------------------------------------------
# Utility
# ---------------------------------------------------------------------------

def get_all_rules() -> list[dict]:
    """Return the full assimilation rule registry."""
    return ASSIMILATION_RULES

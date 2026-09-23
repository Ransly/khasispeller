"""
database.py — Lexicon Database Layer
Khasi NLP Engine

Single source of truth. khasi_db.json drives BOTH the morphological
analyser AND the spell checker. There is no separate frequency dictionary.

The spell checker reads its word list via to_freq_dict(), derived entirely
from the lexicon. When you add or delete an entry the spell checker's word
list is updated in the same operation.

Schema additions handled here (new in uploaded files):
  - affix_profile.suffix          : str | None   (documented but not yet parsed)
  - morphological_flags.tien_kynnoh_subtype : str  ("pdeng" | "non_lexical" | "unknown")
  - morphological_flags.kynnoh_constituent  : str | None
  - shim_kylliang_type            : str | None   ("direct" | "calque" | "hybrid")
  - reduplication_type values now use long form:
      "total_reduplication", "partial_reduplication_vowel",
      "reduplication_interfix_A",  "reduplication_interfix_B"
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Optional


_DEFAULT_DB_PATH = Path(__file__).parent.parent / "data" / "khasi_db.json"


# ---------------------------------------------------------------------------
# Word classes
#
# The lexicon carries the SAME part of speech under two vocabularies. The
# canonical inventory is declared in `grammar_schema.pos_inventory`:
#
#   NOUN VERB ADJ ADV PRON DET INTJ CONJ ADP PART NUM PHRASE OTHER
#
# and that is what the PostgreSQL `pos` column holds. But `_enriched_to_flat`
# expands them into long lowercase names, so `lookup()` hands callers `noun`,
# `adjective`, `preposition`, `interjection` — except CONJ, which it leaves
# short. Code that tested one vocabulary silently missed entries written in
# the other; normalise before comparing rather than listing both spellings at
# every call site.
# ---------------------------------------------------------------------------
_POS_CANONICAL = {
    "noun": "NOUN", "verb": "VERB",
    "adjective": "ADJ", "adj": "ADJ",
    "adverb": "ADV", "adv": "ADV",
    "pronoun": "PRON", "pron": "PRON",
    "determiner": "DET", "det": "DET", "article": "DET",
    "interjection": "INTJ", "intj": "INTJ",
    "conjunction": "CONJ", "conj": "CONJ",
    "preposition": "ADP", "adp": "ADP",
    "particle": "PART", "part": "PART",
    "numeral": "NUM", "num": "NUM",
    "phrase": "PHRASE", "other": "OTHER",
    # Retired tags, from the schema's own `grammar_schema.pos_remap_history`
    # rather than guessed here. PREFIX is the one that matters: the lexicon
    # holds prefix meta-entries (`jing`, `jing-`) whose glosses read "Prefix
    # forming Abstract Nouns", and the schema retires PREFIX to OTHER, which
    # keeps them out of OPEN_CLASSES and so out of compound seams.
    "pronoun_dem": "PRON", "dem_pron": "PRON",
    "dem_adj": "DET", "art": "DET",
    "prefix": "OTHER",
}

# Open classes — the ones a compound or a productive derivation can be built
# out of. Closed classes (PRON DET INTJ CONJ ADP PART) are function words.
OPEN_CLASSES = frozenset({"NOUN", "VERB", "ADJ", "ADV"})

# ---------------------------------------------------------------------------
# English harvested out of multi-word entries
#
# `all_surface_forms()` splits multi-word entries into tokens so that a Khasi
# word appearing only inside a phrase — `shnong` in "dorbar shnong" — can still
# be offered as a correction. The 1906 dictionary also renders technical
# English with Khasi phrases built on the loanword itself ("kam u mayor" for
# Mayoralty, "riewdkhot ka jury" for Juror), and a handful of records are
# OCR damage whose surface is English outright. Both put English into the
# candidate pool, where `rain` once outranked `raiñ` for the input `raiin`.
#
# `scripts/fix_english_headwords.py` repaired the records that were damaged.
# What is left is genuine entries whose phrases contain an English loanword,
# and those cannot be filtered by rule. Three were measured and all fail:
#
#   phonotactics   rejects 21 of 118 — `jury`, `mayor` and `linen` are all
#                  legal Khasi shapes
#   recurrence     `shnong` occurs in 50 entries and `jury` in 1, but 1,717
#                  of 2,195 genuine phrase-tokens also occur exactly once
#   English list   38 of the 81 remaining are Khasi words that merely collide
#                  with an English spelling — `sui` (black soil), `thur`,
#                  `bot`, `rod`, `ara`, `bani`, `mew`, `nip`, `pram`
#
# So the list is written out by hand. Only tokens confirmed English against
# their source entry are named; anything ambiguous is deliberately left in,
# because dropping a real Khasi word costs a correction the user needs while
# keeping an English one costs a suggestion they can ignore.
#
# Excluded from the CANDIDATE POOL only. The entries themselves are untouched
# and `lookup()` still resolves these tokens, so nothing that was analysable
# stops being analysable.
_ENGLISH_IN_PHRASES = frozenset({
    # English loanwords carrying a Khasi phrase that translates a technical term
    "isthmus", "jury", "linen", "mail", "major", "mayor", "meridian",
    "minister", "multiply", "pastor", "pints", "poetry", "pope", "pound",
    "rabbit", "report", "senator", "stamp", "theater", "ton", "torpedo",
    "whale", "will", "willow", "wine", "list", "pass", "bomb", "ether",
    "hysteria", "roman", "andes", "japan", "panama",
    # tokens of records whose surface is an English fragment
    "away", "bean", "horse", "int", "inter", "kept", "kidney", "like",
    "mouse", "shell", "short", "slope", "spell", "stall", "things", "two",
    "women",
})



def canonical_pos(value: Optional[str]) -> str:
    """Map either POS vocabulary onto the declared inventory. '' when unknown."""
    v = (value or "").strip().lower()
    if not v:
        return ""
    return _POS_CANONICAL.get(v, v.upper())

# ---------------------------------------------------------------------------
# Enriched (v3.2) ↔ flat (v2.0) entry shim
#
# As of 2026-05-08 the canonical data shape is the enriched v3.2 schema
# (form.surface / lemma.root / morphology.structure[] / semantics.gloss[]).
# The morphological engine still expects the flat v2.0 shape it was written
# against (surface_form / root_lemma / affix_profile / morphological_flags).
# We translate at the data layer so the engine modules stay untouched.
#
# This shim is the bridge during the architecture-doc refactor; once the
# engine is ported to read enriched paths directly, it can be removed.
# ---------------------------------------------------------------------------

_POS_ENRICH_TO_FLAT: dict[str, str] = {
    "NOUN":  "noun",
    "VERB":  "verb",
    "ADJ":   "adjective",
    "ADV":   "adverb",
    "PRON":  "pronoun",
    "INTJ":  "interjection",
    "OTHER": "other",
    # Phase A — extended map. The lexicon also carries CONJ/ADP/DET/
    # PART/NUM/PROPN/AUX. Without these entries, _enriched_to_flat()
    # falls through to lowercase-the-original, which works for the
    # `applies_to` gate by accident (no prefix's applies_to ever matches
    # `conj`/`adp`/etc.), but produces inconsistent flat-shape values.
    # Mapping them explicitly makes the flat layer self-consistent.
    "CONJ":  "conj",
    "CCONJ": "conj",
    "SCONJ": "conj",
    "ADP":   "preposition",
    "DET":   "determiner",
    "PART":  "particle",
    "NUM":   "number",
    "PROPN": "proper_noun",
    "AUX":   "auxiliary",
}

_POS_FLAT_TO_ENRICH: dict[str, str] = {v: k for k, v in _POS_ENRICH_TO_FLAT.items()}


def _is_enriched_entry(entry: dict) -> bool:
    """True if the entry uses the v3.2 nested shape."""
    return "form" in entry and isinstance(entry.get("form"), dict) \
        and "surface" in entry["form"]


def _enriched_to_flat(entry: dict) -> dict:
    """Translate one v3.2 entry → the v2.0 flat shape used by the engine."""
    form  = entry.get("form") or {}
    lemma = entry.get("lemma") or {}
    gram  = entry.get("grammar") or {}
    morph = entry.get("morphology") or {}
    phon  = entry.get("phonology") or {}
    sem   = entry.get("semantics") or {}
    src   = entry.get("source") or {}
    rels  = entry.get("lexical_relations") or {}

    prefix = prefix2 = prefix3 = infix = suffix_marker = None
    _prefix_count = 0
    for part in morph.get("structure") or []:
        ptype = part.get("type")
        praw  = (part.get("form") or {}).get("raw")
        if ptype == "prefix":
            if _prefix_count == 0:
                prefix = praw
            elif _prefix_count == 1:
                prefix2 = praw
            elif _prefix_count == 2:
                prefix3 = praw
            _prefix_count += 1
        elif ptype == "infix" and infix is None:
            infix = praw
        elif ptype == "suffix" and suffix_marker is None:
            suffix_marker = praw

    # Also pull from top-level affix_profile if present (used by fix entries
    # and any entry whose morphology.structure lacks suffix/infix parts).
    _raw_affix = entry.get("affix_profile") or {}
    if _raw_affix.get("suffix") and suffix_marker is None:
        suffix_marker = _raw_affix["suffix"]
    if _raw_affix.get("infix") and infix is None:
        infix = _raw_affix["infix"]

    pos_enriched = gram.get("pos") or ""
    pos_flat = _POS_ENRICH_TO_FLAT.get(pos_enriched, pos_enriched.lower())

    gloss_list = sem.get("gloss") or []
    english_gloss = "; ".join(g for g in gloss_list if g)

    morph_type    = morph.get("type")
    redup         = (morph.get("features") or {}).get("reduplication") or "none"
    is_compound   = morph_type == "compound"
    is_tien_kynnoh = bool((morph.get("features") or {}).get("is_tien_kynnoh"))

    # For compounds, the v2.0 schema expected morphological_flags.compound_roots
    # to list the constituent root forms. Build that from morphology.structure[].
    compound_roots: list[str] = []
    if is_compound:
        for part in morph.get("structure") or []:
            if part.get("type") == "root":
                raw = (part.get("form") or {}).get("raw")
                if raw:
                    compound_roots.append(raw)

    # ── root_lemma: prefer the structure's root form when lemma.root is
    # demonstrably wrong. Some v3.2 entries (e.g. kh_DB_001382 nong-batai)
    # set lemma.root to the PREFIX rather than the actual root, which makes
    # the morphology parser report nonsense like "prefix=nong + root=nong".
    # When lemma.root equals the detected prefix, fall back to the first
    # `type=="root"` part in morphology.structure.
    _root_lemma = lemma.get("root", "")
    if prefix and _root_lemma == prefix:
        for part in morph.get("structure") or []:
            if part.get("type") == "root":
                _struct_root = (part.get("form") or {}).get("raw")
                if _struct_root:
                    _root_lemma = _struct_root
                    break

    # Single-root ruling marker (2026-06-10): True when the stored
    # morphology.structure is exactly one root segment — i.e. the
    # maintainer has ruled this word atomic (mareh, jingud, pynam, …).
    # The morphology parser uses this to skip its structural prefix
    # scan and its root_lemma prefix-stripping guard.
    _struct = morph.get("structure") or []
    single_root_ruled = len(_struct) == 1 and _struct[0].get("type") == "root"

    flat: dict = {
        "entry_id":            entry.get("entry_id", ""),
        "kind":                entry.get("kind"),  # Phase A: carry the entry-kind through the flat shape
        "surface_form":        form.get("surface", ""),
        "root_lemma":          _root_lemma,
        "single_root_ruled":   single_root_ruled,
        "grammatical_class":   pos_flat,
        "gender":              gram.get("gender") or "none",
        "clitic":              gram.get("clitic") or "none",
        "english_gloss":       english_gloss,
        "loan_origin":         src.get("loan_origin") or "none",
        "source":              src.get("origin") or "",
        "source_file":         src.get("source_file"),
        "previous_entry_id":   src.get("previous_entry_id"),
        "notes":               entry.get("raw_notes") or "",
        "shim_kylliang_type":  src.get("loan_type"),
        "affix_profile": {
            "prefix":            prefix,
            "prefix2":           prefix2,
            "prefix3":           prefix3,
            "infix":             infix,
            "assimilation_rule": None,
            "suffix":            suffix_marker,
        },
        "morphological_flags": {
            "is_compound":      is_compound,
            "compound_roots":   compound_roots,
            "reduplication_type": redup,
            "is_tien_kynnoh":   is_tien_kynnoh,
            "tien_kynnoh_subtype": "unknown",
            "kynnoh_constituent":  None,
        },
        "phonological_flags": {
            "syllable_structure":  (phon.get("derived") or {}).get("pattern", ""),
            "has_initial_cluster": False,
            "has_final_cluster":   False,
            "aspirated_final":     False,
        },
        "related_entries":     list(rels.get("related") or []),
    }
    return flat


def _build_canonical(structure: list, affix: dict) -> str:
    """Build a linguistically correct canonical representation.

    For infix-derived forms the canonical must express the derivational
    relationship between root and infix using morphophonemic bracket notation,
    NOT a linear string concatenation.  Concatenating the infix raw value with
    the surface form (e.g. "yn+kynjat") is morphologically invalid because the
    infix is not a separable prefix-like morpheme — it is a sub-syllabic
    insertion into the root consonant skeleton.

    Correct notation:
        root = kjat,  infix = -yn-  →  canonical = "kjat + <-yn->"

    For all non-infix forms the existing '+'-join behaviour is preserved.
    """
    infix_marker = affix.get("infix")
    if infix_marker:
        # Root is the first (and only) 'root'-typed part in structure.
        root_raw = next(
            (p["form"]["raw"] for p in structure if p.get("type") == "root" and p.get("form")),
            ""
        )
        return f"{root_raw} + <-{infix_marker}->"
    # Prefix / compound / root forms: join all parts linearly.
    return "+".join(p["form"]["raw"] for p in structure if p.get("form"))


def _normalise_infix_enriched(entry: dict) -> None:
    """Idempotent in-place fix for enriched entries that carry infix morphology.

    Older DB records stored the infix canonical as "infix+surface" (e.g.
    "yn+kynjat") and placed the infix struct before the root.  This function
    silently corrects both issues on load so the engine never surfaces the
    malformed representation, even if the JSON on disk hasn't been migrated.

    Rules applied only when morphology.structure contains an infix part:
      1. Re-order structure so root comes before infix.
      2. Ensure infix form has a `display` key ("-yn-" notation).
      3. Rewrite canonical to "root + <-infix->".
      4. Set morph type to "infixed".
    """
    morph = entry.get("morphology")
    if not isinstance(morph, dict):
        return
    structure = morph.get("structure") or []
    infix_parts = [p for p in structure if p.get("type") == "infix"]
    if not infix_parts:
        return  # nothing to do

    root_parts = [p for p in structure if p.get("type") == "root"]

    # 1. Canonical order: root(s) first, then infix
    other_parts = [p for p in structure if p.get("type") not in ("root", "infix")]
    morph["structure"] = other_parts + root_parts + infix_parts

    # 2. Ensure display key on infix
    for ip in infix_parts:
        raw = (ip.get("form") or {}).get("raw", "")
        if raw:
            ip.setdefault("form", {})["display"] = f"-{raw}-"

    # 3. Fix canonical if it looks like the old "infix+surface" pattern
    infix_raw  = infix_parts[0]["form"]["raw"]
    root_raw   = root_parts[0]["form"]["raw"] if root_parts else ""
    canonical  = morph.get("canonical", "")
    # Detect broken pattern: starts with infix marker, or infix appears before '+'
    if (canonical.startswith(infix_raw + "+") or
            canonical == f"{infix_raw}+{root_raw}"):
        morph["canonical"] = f"{root_raw} + <-{infix_raw}->"

    # 4. morph type
    if morph.get("type") not in ("infixed",):
        morph["type"] = "infixed"


def _flat_to_enriched(flat: dict, *, base: Optional[dict] = None) -> dict:
    """Translate a v2.0 flat entry (e.g. from POST /v1/lexicon) → v3.2.

    `base` is an optional pre-existing enriched entry whose fields are merged
    underneath the translated ones; used by update_entry to preserve nested
    fields (morphology.structure, validation, etc.) that the flat shape can't
    express.
    """
    pos_flat = (flat.get("grammatical_class") or "").lower()
    pos_enriched = _POS_FLAT_TO_ENRICH.get(pos_flat, pos_flat.upper())

    gender = flat.get("gender")
    if gender in (None, "", "none"):
        gender = None

    clitic = flat.get("clitic")
    if clitic in (None, "", "none"):
        clitic = None

    loan_origin = flat.get("loan_origin")
    if loan_origin in (None, "", "none"):
        loan_origin = None

    # Prefer the explicit `glosses` list (added 2026-05-10) when supplied;
    # otherwise fall back to the legacy semicolon-split english_gloss field
    # so older POST payloads keep working unchanged.
    explicit_glosses = flat.get("glosses")
    if isinstance(explicit_glosses, list) and explicit_glosses:
        gloss_list = [str(g).strip() for g in explicit_glosses if str(g).strip()]
    else:
        gloss_str = flat.get("english_gloss") or ""
        gloss_list = [g.strip() for g in gloss_str.split(";") if g.strip()]

    # `domain` / `variants` / `better_forms` use a tri-state convention:
    #   key absent or value is None  → field NOT touched (base value preserved)
    #   value is "" / [] / cleaned-empty → field cleared explicitly
    #   value is non-empty            → field set to the cleaned value
    _DOMAIN_RAW = flat.get("domain")
    if _DOMAIN_RAW is None:
        domain_provided, domain_val = False, None
    elif isinstance(_DOMAIN_RAW, str):
        domain_provided, domain_val = True, (_DOMAIN_RAW.strip() or None)
    else:
        domain_provided, domain_val = True, None

    def _clean_str_list(v):
        if v is None:
            return None  # not provided
        if not isinstance(v, list):
            return []  # malformed → treat as explicit clear
        return [str(x).strip() for x in v if str(x).strip()]

    variants_list     = _clean_str_list(flat.get("variants"))
    better_forms_list = _clean_str_list(flat.get("better_forms"))

    affix = flat.get("affix_profile") or {}
    morph_flags = flat.get("morphological_flags") or {}
    phon_flags = flat.get("phonological_flags") or {}

    structure: list[dict] = []
    if affix.get("prefix"):
        structure.append({"type": "prefix", "form": {"raw": affix["prefix"]}})
    if affix.get("prefix2"):
        structure.append({"type": "prefix", "form": {"raw": affix["prefix2"]}})
    if affix.get("prefix3"):
        structure.append({"type": "prefix", "form": {"raw": affix["prefix3"]}})
    structure.append({"type": "root", "form": {"raw": flat.get("root_lemma", "")}})
    if affix.get("infix"):
        # Infix is appended AFTER root to reflect derivational order:
        # root (kjat) + infix (-yn-) → surface (kynjat).
        # The display key stores the standard morphological bracket notation.
        infix_marker = affix["infix"]
        structure.append({
            "type": "infix",
            "form": {"raw": infix_marker, "display": f"-{infix_marker}-"},
        })

    morph_type = "root"
    if morph_flags.get("is_compound"):
        morph_type = "compound"
    elif affix.get("prefix"):
        morph_type = "prefixed"
    elif affix.get("infix"):
        morph_type = "infixed"
    elif (morph_flags.get("reduplication_type") or "none") != "none":
        morph_type = "reduplicated"

    redup = morph_flags.get("reduplication_type")
    if redup in (None, "", "none"):
        redup = None

    enriched: dict = {
        "entry_id": flat.get("entry_id", ""),
        "form": {
            "surface":    flat.get("surface_form", ""),
            "normalized": flat.get("surface_form", ""),
            # variants: write only when explicitly provided; otherwise leave
            # to the base merge so legacy clients don't wipe existing data.
            **({"variants": variants_list} if variants_list is not None else
               ({"variants": []} if not base else {})),
        },
        "lemma": {
            "root":    flat.get("root_lemma", ""),
            "root_id": None,
        },
        "grammar": {
            "pos":    pos_enriched,
            "gender": gender,
            "clitic": clitic,
        },
        "morphology": {
            "type":      morph_type,
            "structure": structure,
            "canonical": _build_canonical(structure, affix),
            "features": {
                "reduplication":  redup,
                "is_tien_kynnoh": bool(morph_flags.get("is_tien_kynnoh")),
            },
            "morphology_rule": [],
        },
        "phonology": {
            "surface": flat.get("surface_form", ""),
            "derived": {
                "syllables":      [],
                "syllable_count": 0,
                "pattern":        phon_flags.get("syllable_structure", ""),
            },
            "flags": {"exception": False, "needs_review": False},
        },
        "semantics": {
            "gloss":  gloss_list,
            **({"domain": domain_val} if domain_provided else
               ({"domain": None} if not base else {})),
        },
        "lexical_relations": {
            "related":      list(flat.get("related_entries") or []),
            "variants":     [],
            **({"better_forms": better_forms_list} if better_forms_list is not None else
               ({"better_forms": []} if not base else {})),
        },
        "source": {
            "origin":            flat.get("source") or "user",
            "source_file":       flat.get("source_file"),
            "loan_origin":       loan_origin,
            "loan_type":         flat.get("shim_kylliang_type"),
            "previous_entry_id": flat.get("previous_entry_id"),
        },
        "validation": {
            "is_valid":       True,
            "confidence":     1.0,
            "needs_review":   False,
            "review_reasons": [],
        },
        "raw_notes": flat.get("notes") or None,
    }
    if base:
        # Carry over enriched-only nested fields the flat shape can't express.
        # `form` is included so absent variants don't wipe base.form.variants
        # — the conditional spreads above only emit the `variants` key when
        # the payload actually supplies one.
        for key in ("form", "morphology", "phonology", "semantics",
                    "lexical_relations", "source", "validation"):
            if key in base and isinstance(base[key], dict):
                merged = {**base[key], **enriched[key]}
                enriched[key] = merged
    return enriched


def _load_payload_from_pg(database_url: str) -> dict:
    """Fetch meta/phonology/morphology singletons + enriched lexicon from PG.

    Returns a dict shaped like khasi_db.json (meta, phonology, morphology,
    lexicon) but with enriched-shape entries — the caller then runs each
    entry through _enriched_to_flat() so the engine sees v2.0 shape.
    """
    import psycopg2
    import psycopg2.extras

    if database_url.startswith("postgresql+asyncpg://"):
        database_url = "postgresql://" + database_url[len("postgresql+asyncpg://"):]

    payload: dict = {"meta": {}, "phonology": {}, "morphology": {},
                     "lexicon": []}
    with psycopg2.connect(database_url) as conn:
        with conn.cursor(cursor_factory=psycopg2.extras.RealDictCursor) as cur:
            # PostgreSQL is the single served source. The last four tables
            # here used to exist only in `data/khasi_db_extras.json`, which
            # made the database an incomplete picture of the system and left
            # the sidecar free to drift — it was found 28 quarantine records
            # behind the JSON on 2026-09-10. A table that is absent is simply
            # skipped, so an older database still loads and the sidecar can
            # still fill the gap.
            for table, key in (
                ("db_meta", "meta"),
                ("phonology_meta", "phonology"),
                ("morphology_meta", "morphology"),
                ("demo_examples_meta", "demo_examples"),
                ("quarantine_meta", "quarantine"),
                ("grammar_schema_meta", "grammar_schema"),
                ("production_readiness_meta", "production_readiness"),
            ):
                try:
                    cur.execute(f"SELECT payload FROM {table} WHERE id = 1")
                except Exception:
                    conn.rollback()          # table not in this database yet
                    continue
                row = cur.fetchone()
                if row and row["payload"]:
                    payload[key] = row["payload"]
            # v4.3 schema (option-4): merge in the targeted enrichment columns
            # (semantic_features, ontology_review, domain_source,
            # homograph_group, gloss_normalized) instead of loading a single
            # bloated `enriched_payload` JSONB. Saves ~300 MB resident on
            # Render's 512 MB free tier.
            cur.execute("""
                SELECT column_name FROM information_schema.columns
                WHERE table_name = 'lexicon_entries'
                  AND column_name IN ('semantic_features', 'ontology_review',
                                      'domain_source', 'homograph_group',
                                      'gloss_normalized',
                                      'number', 'diminutive')
            """)
            new_cols = {r["column_name"] for r in cur.fetchall()}
            has_v43_cols = (
                "semantic_features" in new_cols
                and "ontology_review" in new_cols
            )
            # Phase 8 proclitic-derived `number` + `diminutive` are detected
            # independently of v4.3 so an older PG without v4.3 can still
            # serve the newer fields if it was migrated against the newer
            # migrate_json_to_pg.py.
            has_proclitic_cols = (
                "number" in new_cols and "diminutive" in new_cols
            )
            extra_select = ""
            if has_v43_cols:
                extra_select += (
                    ", semantic_features, ontology_review, "
                    "domain_source, homograph_group, gloss_normalized"
                )
            if has_proclitic_cols:
                extra_select += ", number, diminutive"
            cur.execute(f"""
                SELECT entry_id, surface, normalized, variants,
                       root, root_id, pos, gender, clitic,
                       morph_type, morph_canonical, morph_structure,
                       morph_features, morph_rules,
                       phon_surface, phon_derived, phon_flags,
                       gloss, domain, related, better_forms,
                       origin, source_file, loan_origin, loan_type, previous_id,
                       is_valid, confidence, needs_review, review_reasons,
                       raw_notes{extra_select}
                FROM lexicon_entries
            """)
            for r in cur:
                entry_dict = {
                    "entry_id": r["entry_id"],
                    "form": {
                        "surface":    r["surface"],
                        "normalized": r["normalized"],
                        "variants":   list(r["variants"] or []),
                    },
                    "lemma":   {"root": r["root"], "root_id": r["root_id"]},
                    "grammar": {"pos": r["pos"], "gender": r["gender"],
                                "clitic": r["clitic"]},
                    "morphology": {
                        "type":           r["morph_type"],
                        "structure":      r["morph_structure"] or [],
                        "canonical":      r["morph_canonical"],
                        "features":       r["morph_features"] or {},
                        "morphology_rule": r["morph_rules"] or [],
                    },
                    "phonology": {
                        "surface": r["phon_surface"],
                        "derived": r["phon_derived"] or {},
                        "flags":   r["phon_flags"] or {},
                    },
                    "semantics": {
                        "gloss":  list(r["gloss"] or []),
                        "domain": r["domain"],
                    },
                    "lexical_relations": {
                        "related":      list(r["related"] or []),
                        "variants":     [],
                        "better_forms": list(r["better_forms"] or []),
                    },
                    "source": {
                        "origin":            r["origin"],
                        "source_file":       r["source_file"],
                        "loan_origin":       r["loan_origin"],
                        "loan_type":         r["loan_type"],
                        "previous_entry_id": r["previous_id"],
                    },
                    "validation": {
                        "is_valid":       r["is_valid"],
                        "confidence":     float(r["confidence"]) if r["confidence"] is not None else 1.0,
                        "needs_review":   r["needs_review"],
                        "review_reasons": list(r["review_reasons"] or []),
                    },
                    "raw_notes": r["raw_notes"],
                }
                # v4.3 enrichment columns merged in (when present in the
                # active schema). Keeps the loaded entry shape identical to
                # the original khasi_db.json — endpoints don't care whether
                # the data came from PG flat columns or v4.3 enrichment cols.
                if has_v43_cols:
                    sf = r["semantic_features"] or {}
                    if sf:
                        entry_dict["semantics"]["semantic_features"] = sf
                    or_ = r["ontology_review"] or {}
                    if or_:
                        entry_dict["semantics"]["ontology_review"] = or_
                    if r["domain_source"]:
                        entry_dict["semantics"]["domain_source"] = r["domain_source"]
                    if r["gloss_normalized"]:
                        entry_dict["semantics"]["gloss_normalized"] = r["gloss_normalized"]
                    if r["homograph_group"]:
                        entry_dict["lexical_relations"]["homograph_group"] = r["homograph_group"]
                # Phase 8: merge proclitic-derived grammar fields back into
                # the entry's `grammar` block so endpoints (and the entry
                # detail panel) see the same shape they'd see in JSON mode.
                if has_proclitic_cols:
                    if r["number"] is not None:
                        entry_dict["grammar"]["number"] = r["number"]
                    if r["diminutive"] is not None:
                        entry_dict["grammar"]["diminutive"] = r["diminutive"]
                payload["lexicon"].append(entry_dict)
    return payload


def _pg_upsert_entry(database_url: str, enriched: dict) -> None:
    """INSERT-or-UPDATE one enriched entry into lexicon_entries."""
    import psycopg2
    import psycopg2.extras

    if database_url.startswith("postgresql+asyncpg://"):
        database_url = "postgresql://" + database_url[len("postgresql+asyncpg://"):]

    form  = enriched.get("form") or {}
    lemma = enriched.get("lemma") or {}
    gram  = enriched.get("grammar") or {}
    morph = enriched.get("morphology") or {}
    phon  = enriched.get("phonology") or {}
    sem   = enriched.get("semantics") or {}
    rels  = enriched.get("lexical_relations") or {}
    src   = enriched.get("source") or {}
    val   = enriched.get("validation") or {}

    row = {
        "entry_id":        enriched["entry_id"],
        "surface":         form.get("surface", ""),
        "normalized":      form.get("normalized", ""),
        "variants":        list(form.get("variants") or []),
        "root":            lemma.get("root"),
        "root_id":         lemma.get("root_id"),
        "pos":             gram.get("pos"),
        "gender":          gram.get("gender"),
        "clitic":          gram.get("clitic"),
        # Phase 8 proclitic-derived. Forward-compatible: when the live PG
        # schema doesn't have these columns yet, the UndefinedColumn
        # fallback below drops them before retrying the upsert.
        "number":          gram.get("number"),
        "diminutive":      gram.get("diminutive"),
        "morph_type":      morph.get("type"),
        "morph_canonical": morph.get("canonical"),
        "morph_structure": psycopg2.extras.Json(morph.get("structure") or []),
        "morph_features":  psycopg2.extras.Json(morph.get("features") or {}),
        "morph_rules":     psycopg2.extras.Json(morph.get("morphology_rule") or []),
        "phon_surface":    phon.get("surface"),
        "phon_derived":    psycopg2.extras.Json(phon.get("derived") or {}),
        "phon_flags":      psycopg2.extras.Json(phon.get("flags") or {}),
        "gloss":           list(sem.get("gloss") or []),
        "domain":          sem.get("domain"),
        "related":         list(rels.get("related") or []),
        "better_forms":    list(rels.get("better_forms") or []),
        "origin":          src.get("origin"),
        "source_file":     src.get("source_file"),
        "loan_origin":     src.get("loan_origin"),
        "loan_type":       src.get("loan_type"),
        "previous_id":     src.get("previous_entry_id"),
        "is_valid":        bool(val.get("is_valid", True)),
        # `.get(..., default)` only fires on missing key — explicit `null`
        # still returns None. Phase 8 skeletal entries carry that null;
        # mirror the guard from scripts/migrate_json_to_pg.py.
        "confidence":      float(val["confidence"]) if val.get("confidence") is not None else 1.0,
        "needs_review":    bool(val.get("needs_review", False)),
        "review_reasons":  list(val.get("review_reasons") or []),
        "raw_notes":       enriched.get("raw_notes"),
        # v4.3 targeted enrichment columns (replaces enriched_payload to
        # save ~300 MB resident memory). Each is small: semantic_features
        # is a 4-key dict (~150 B), the others are short strings or 2-key
        # dicts. Combined cost is < 5% of the row size.
        "semantic_features": psycopg2.extras.Json(sem.get("semantic_features") or {}),
        "ontology_review":   psycopg2.extras.Json(sem.get("ontology_review")   or {}),
        "domain_source":     sem.get("domain_source"),
        "homograph_group":   rels.get("homograph_group"),
        "gloss_normalized":  sem.get("gloss_normalized"),
    }
    cols = list(row.keys())
    placeholders = ", ".join(f"%({c})s" for c in cols)
    update_set = ", ".join(f"{c}=EXCLUDED.{c}" for c in cols if c != "entry_id")
    sql = (
        f"INSERT INTO lexicon_entries ({', '.join(cols)}) VALUES ({placeholders}) "
        f"ON CONFLICT (entry_id) DO UPDATE SET {update_set}, updated_at = now()"
    )
    with psycopg2.connect(database_url) as conn:
        with conn.cursor() as cur:
            try:
                cur.execute(sql, row)
            except psycopg2.errors.UndefinedColumn as exc:
                # Older DB without one of the v4.3 columns — retry with only
                # the columns that actually exist. This makes the upsert
                # forward- and backward-compatible across schema versions.
                conn.rollback()
                cur.execute("""
                    SELECT column_name FROM information_schema.columns
                    WHERE table_name = 'lexicon_entries'
                """)
                live_cols = {r[0] for r in cur.fetchall()}
                cols2 = [c for c in cols if c in live_cols]
                row2 = {c: row[c] for c in cols2}
                placeholders2 = ", ".join(f"%({c})s" for c in cols2)
                update_set2 = ", ".join(f"{c}=EXCLUDED.{c}" for c in cols2 if c != "entry_id")
                sql2 = (
                    f"INSERT INTO lexicon_entries ({', '.join(cols2)}) VALUES ({placeholders2}) "
                    f"ON CONFLICT (entry_id) DO UPDATE SET {update_set2}, updated_at = now()"
                )
                cur.execute(sql2, row2)


def _pg_delete_entry(database_url: str, entry_id: str) -> None:
    import psycopg2
    if database_url.startswith("postgresql+asyncpg://"):
        database_url = "postgresql://" + database_url[len("postgresql+asyncpg://"):]
    with psycopg2.connect(database_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM lexicon_entries WHERE entry_id = %s",
                (entry_id,),
            )

# Frequency weights for the spell checker word list
_FREQ_ROOT         = 20   # root / lemma form
_FREQ_SURFACE      = 12   # surface form that differs from root
_FREQ_SURFACE_EQ   = 15   # surface form that IS the root
_FREQ_MORPHO_BONUS =  5   # extra for morphologically transparent forms

# Frequency weight for tokens extracted from multi-word surface forms
# Scaled logarithmically: a token appearing in 26 compound entries gets
# weight ~18, close to a root (20), while one appearing once gets ~8.
# Formula: round(8 + 12 * log2(count) / log2(max_expected_count))
_FREQ_COMPOUND_BASE     =  8    # weight for a token appearing once
_FREQ_COMPOUND_SCALE    = 12    # additional weight range above base
_FREQ_COMPOUND_LOG_REF  = 50.0  # reference count for max scaling (log2 normaliser)
_FREQ_COMPOUND_MAX      = 19    # hard cap — compound tokens never outrank roots (20)

# All known reduplication type keys (long form used by complex.py)
_REDUPLICATION_TYPES: frozenset[str] = frozenset([
    "total_reduplication",
    "partial_reduplication_vowel",
    "reduplication_interfix_A",
    "reduplication_interfix_B",
    "triplication",
])


def _phonotactic_ok(token: str) -> bool:
    """
    True when *token* could be a Khasi word at all.

    Compound-token extraction splits multi-word surface forms so that words
    living only inside phrases — `shnong` in `dorbar shnong` — stay
    recognisable. 59% of surface forms are multi-word, and some carry an
    English translation alongside the Khasi:

        'ioh sah jit la ka long sone face and honour'
        'ka masi ka la rong jyndat ïa ka jaiñ ka the cow took away the cloth'

    Splitting those put `and`, `face`, `cow`, `big`, `car` into the index,
    and because `is_known()` consults it, they became valid Khasi words —
    `cow` passed a spell check. Bare digraphs arrived the same way: `sh` and
    `kh` are not words, they are single phonemes.

    A phonotactic screen at build time keeps them out. It costs one pass at
    load rather than a check per lookup, and it cannot reject a real Khasi
    word, since being pronounceable in Khasi is what it tests.
    """
    try:
        from khasi_engine.phonology import validate as _v

        return bool(_v(token)["pass"])
    except Exception:
        return True          # never let a phonology failure empty the index


# Process-wide cache of constructed lexicons, keyed by data source.
#
# Building one means a PostgreSQL fetch (or a 67 MB JSON parse) plus a full
# index rebuild, and the API only ever builds ONE — but the test suite built
# NINE per run, because neither KhasiSpeller nor KhasiAnalyser accepts an
# injected db and each constructs its own. Nine rebuilds of a 410,131-key
# index is what made a 6-minute test run freeze the development laptop.
#
# Safe to share: within a process the lexicon is read-only except through
# save()/mutators on the same object, so every holder of a given key sees the
# same authoritative state. Set KHASI_DB_NO_CACHE=1 for an isolated instance,
# or call KhasiDB.clear_cache() to force the next construction to re-read.
_DB_CACHE: dict[tuple, "KhasiDB"] = {}


class KhasiDB:
    """
    In-memory Khasi lexicon.  Single source of truth for both the
    morphological engine and the spell checker.

    Construction is memoised per data source — see _DB_CACHE. Two calls with
    the same db_path and DATABASE_URL return the SAME object, so the lexicon
    is loaded and indexed once per process rather than once per caller.
    """

    @staticmethod
    def _cache_key(db_path: Optional[str | Path]) -> tuple:
        path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        return (str(path), os.environ.get("DATABASE_URL") or "")

    @classmethod
    def clear_cache(cls) -> None:
        """Drop every cached lexicon; the next construction re-reads the
        source. Needed only when the data changed underneath the process
        (a migration, an external edit)."""
        _DB_CACHE.clear()

    def __new__(cls, db_path: Optional[str | Path] = None):
        if os.environ.get("KHASI_DB_NO_CACHE"):
            return super().__new__(cls)
        key = cls._cache_key(db_path)
        cached = _DB_CACHE.get(key)
        if cached is not None:
            return cached
        inst = super().__new__(cls)
        _DB_CACHE[key] = inst
        return inst

    def __init__(self, db_path: Optional[str | Path] = None):
        # Python calls __init__ even when __new__ hands back a cached
        # instance, so re-loading has to be short-circuited here.
        if getattr(self, "_initialised", False):
            return
        # Resolve the data source. DATABASE_URL takes precedence — when set
        # (production on Render), the lexicon is fetched from PostgreSQL and
        # mutations write through to PG. Otherwise the JSON file at
        # `db_path` (or _DEFAULT_DB_PATH) is the source of truth and `save()`
        # rewrites it. Both paths feed the in-memory engine the SAME flat
        # v2.0 shape via _enriched_to_flat().
        self._database_url: Optional[str] = os.environ.get("DATABASE_URL") or None
        path = Path(db_path) if db_path else _DEFAULT_DB_PATH
        self._json_path: Path = path

        if self._database_url:
            print(f"[khasi-nlp] KhasiDB: loading from PostgreSQL "
                  f"({self._database_url.split('@', 1)[-1]})")
            raw = _load_payload_from_pg(self._database_url)
            # PostgreSQL is the single served source: every top-level block
            # the engine reads now has a table, quarantine and grammar_schema
            # included. The sidecar remains ONLY as a compatibility fallback
            # for a database created before those tables existed, and it is
            # merged with setdefault so it can never override the database.
            #
            # It used to be the sole home of four blocks, which made the
            # database an incomplete picture of the system and left the file
            # free to drift — it was found 28 quarantine records behind the
            # JSON on 2026-09-10, so the served system had an incomplete
            # audit trail while every row of the lexicon was correct.
            missing = [k for k in ("quarantine", "grammar_schema",
                                   "production_readiness", "demo_examples")
                       if not raw.get(k)]
            extras_path = path.parent / "khasi_db_extras.json"
            if missing and extras_path.exists():
                try:
                    with open(extras_path, encoding="utf-8") as f:
                        extras = json.load(f)
                    filled = []
                    for k, v in extras.items():
                        if k == "lexicon":
                            continue
                        # `_meta_copy` is the sidecar describing itself, not
                        # a block the engine serves, so it never counts as a
                        # gap in the database.
                        if k not in raw and v and not k.startswith("_"):
                            filled.append(k)
                        raw.setdefault(k, v)
                    # Report only what the sidecar actually supplied. A block
                    # that is EMPTY in the source is skipped by the migration
                    # (`if blob:`), so treating "absent from PostgreSQL" as
                    # "missing" made an empty `demo_examples` look like a
                    # stale database on every start-up.
                    if filled:
                        print(f"[khasi-nlp] KhasiDB: {', '.join(sorted(filled))} "
                              f"not in PostgreSQL — filled from "
                              f"{extras_path.name}. Re-run "
                              f"scripts/migrate_json_to_pg.py to move them "
                              f"into the database.")
                except (json.JSONDecodeError, OSError) as e:
                    print(f"[khasi-nlp] KhasiDB: could not read {extras_path}: {e}")
        else:
            # Say *why*, not just what. A silent fall-back to JSON when the
            # operator believed they were on PostgreSQL is indistinguishable
            # from a working database, and the only visible difference is a
            # line of log nobody reads twice.
            print(f"[khasi-nlp] KhasiDB: loading from JSON file ({path})")
            print("[khasi-nlp] KhasiDB: DATABASE_URL is not set in this "
                  "process, so PostgreSQL is not being used. Export it, or "
                  "put it in a .env file beside pyproject.toml.")
            with open(path, encoding="utf-8") as f:
                raw = json.load(f)

        self._meta: dict       = raw.get("meta", {})
        self._phonology: dict  = raw.get("phonology", {})
        self._morphology: dict = raw.get("morphology", {})
        self._demo_examples: dict = raw.get("demo_examples", {})

        # v4.3 top-level blocks that KhasiDB does NOT mutate but MUST preserve
        # across save() calls. Without this round-trip, any curate write in
        # JSON mode (e.g. accept-suggestion) would rewrite the file using only
        # meta/phonology/morphology/lexicon and silently drop quarantine /
        # production_readiness / grammar_schema. This caused the Health
        # invariants tab + Quarantine list to silently go blank after the
        # first curate click. Stored as a dict of {block_name: raw_value}
        # so adding a new top-level block in a future schema bump is free —
        # it'll be preserved automatically without touching this code.
        _MANAGED_KEYS = {"meta", "phonology", "morphology", "lexicon", "demo_examples"}
        self._extra_blocks: dict = {
            k: v for k, v in raw.items() if k not in _MANAGED_KEYS
        }

        # Share parsed phonology / morphology / assimilation blocks with
        # the engine modules so they don't re-parse the 77 MB JSON file
        # on every cold start. On Render (DATABASE_URL set) the modules
        # short-circuit their JSON load at import time and rely entirely
        # on these injections — saves ~3 × 300 MB transient peaks during
        # uvicorn worker startup, which was the primary 512 MB free-tier
        # OOM trigger.
        try:
            from . import phonology as _phon_mod
            _phon_mod.set_phonology_cache(self._phonology)
        except Exception:
            pass
        try:
            from . import morphology as _morph_mod
            _morph_mod.set_morphology_cache(self._morphology)
        except Exception:
            pass
        try:
            from . import assimilation as _assim_mod
            _assim_mod.set_assimilation_rules_cache(
                (self._morphology or {}).get("assimilation_rules", []) or []
            )
        except Exception:
            pass

        # Translate enriched (v3.2) entries to flat (v2.0) for the engine.
        # We keep the enriched companions in a side-map keyed by entry_id so
        # writes can round-trip back to PG/JSON without losing fields the flat
        # shape can't express. The flat dicts themselves stay clean — the
        # engine and API responses never see the enriched key.
        raw_entries = raw.get("lexicon", [])
        self._lexicon: list[dict] = []
        self._enriched_by_id: dict[str, dict] = {}
        for e in raw_entries:
            if _is_enriched_entry(e):
                _normalise_infix_enriched(e)   # idempotent; no-op for non-infix
                self._enriched_by_id[e.get("entry_id", "")] = e
                self._lexicon.append(_enriched_to_flat(e))
            else:
                self._lexicon.append(e)

        self._surface_index: dict[str, list[dict]] = {}
        self._folded_index:  dict[str, list[dict]] = {}   # accent/apostrophe-folded fallback
        self._root_index:    dict[str, list[dict]] = {}
        self._id_index:      dict[str, dict]       = {}
        self._freq_dict:     dict[str, int]        = {}
        self._compound_tokens: dict[str, int]      = {}  # tokens from multi-word entries
        self._compound_known: set                  = set()  # ...that could be words
        # Frequent corpus types admitted as CANDIDATES — offerable as
        # corrections. Empty unless khasi_spell.corpus_pool registers into
        # it. See is_attested().
        self._corpus_forms: set                    = set()
        # The same admitted words, plus the reduced spelling the corpus
        # writes them in, as ACCEPTABLE: the confidence vote credits them
        # with corpus frequency. Kept apart from _corpus_forms because the
        # two sets differ — `iatreilang` is acceptable (it is how the word
        # is typed) but is never offered; `ïatreilang` is offered. See
        # is_corpus_accepted().
        self._corpus_accept: set                   = set()
        # Maps each compound token → list of multi-word surface_forms it appears in.
        # Lets us answer "which phrase entries contain this token?" without
        # scanning the lexicon every time. Used by the verdict explainer to
        # show a token's host compound/phrasal entries to the user.
        self._compound_entry_index: dict[str, list[str]] = {}

        self._rebuild_indexes()
        # Set last: a construction that raised must not leave a half-built
        # lexicon cached for every later caller to reuse.
        self._initialised = True

    # ------------------------------------------------------------------
    # Index management
    # ------------------------------------------------------------------

    # Orthographic folding for lenient lookup. Khasi text in the wild (SMS,
    # social media) routinely drops the acute-accent quality marks, writes ï
    # as plain i, and omits the glottal-stop apostrophe. Folding lets `kum`
    # find `kúm`, `iing` find `ïing`, and `iuh` find `'iuh` — WITHOUT losing
    # the canonical entries (exact match is always tried first; the folded
    # index is a fallback that may return several entries that fold alike).
    _FOLD_MAP = str.maketrans({
        "á": "a", "é": "e", "í": "i", "ó": "o", "ú": "u", "ý": "y",
        "ï": "i",
        # ñ -> n added in this project (not upstream). Without it 'ain' does
        # not find 'aiñ', so one of Khasi's two diacritics was tolerated at
        # lookup and the other was not. Checked before changing: folding ñ
        # merges 108 keys, and every one is the SAME word the lexicon
        # already records twice — khwain/khwaiñ, luin/luiñ, niang/ñiang.
        # No distinct words collide.
        "ñ": "n",
    })

    @classmethod
    def _fold_key(cls, word: str) -> str:
        return word.lower().translate(cls._FOLD_MAP).replace("'", "")

    def _rebuild_indexes(self) -> None:
        self._surface_index.clear()
        self._folded_index.clear()
        self._root_index.clear()
        self._id_index.clear()
        self._freq_dict.clear()
        self._compound_tokens.clear()
        self._compound_known.clear()
        self._compound_entry_index.clear()
        for entry in self._lexicon:
            self._index_entry(entry)

    def _index_entry(self, entry: dict) -> None:
        # `.get(k, "")` returns the explicit None when the key exists but
        # is null (Phase 8 kh_KK_ stubs carry `lemma: null` which becomes
        # `root_lemma: None` after _enriched_to_flat). Use `or ""` to
        # coalesce both missing-key and explicit-null into an empty string.
        sf   = (entry.get("surface_form") or "").lower()
        root = (entry.get("root_lemma")   or "").lower()
        eid  =  entry.get("entry_id")     or ""

        if sf:
            self._surface_index.setdefault(sf, []).append(entry)
            folded = self._fold_key(sf)
            if folded and folded != sf:
                self._folded_index.setdefault(folded, []).append(entry)
        if root:
            self._root_index.setdefault(root, []).append(entry)
        if eid:
            self._id_index[eid] = entry

        # Compound-token extraction — index individual words from multi-word surface forms
        import re as _re
        if sf and " " in sf:
            for tok in _re.split(r"[\s\-,!]+", sf):
                # Preserve leading apostrophe — marks Khasi glottal stop
                # e.g. "'riewbha" must NOT become "riewbha" (different word)
                tok = tok.rstrip("()!,.'\"")
                tok = tok.lstrip("()!,\"")
                if len(tok) >= 2 and " " not in tok:
                    # Increment raw count, then recompute log-scaled weight
                    count = self._compound_tokens.get(tok, 0) + 1
                    self._compound_tokens[tok] = count
                    # Candidate generation and acceptance are different
                    # questions and need different answers here. Every
                    # extracted token stays available as a *suggestion*,
                    # because `validate()` has a small false-reject rate and
                    # screening the candidate pool cost real accuracy —
                    # `shafon` stopped being offered for `shafan`. Only the
                    # screened subset may answer "is this a Khasi word".
                    if _phonotactic_ok(tok):
                        self._compound_known.add(tok)
                    # Track which multi-word entries contain this token.
                    # Surface forms are stored as-given (preserving original
                    # casing) so the UI can display them naturally.
                    sf_original = (entry.get("surface_form") or "").strip()
                    bucket = self._compound_entry_index.setdefault(tok, [])
                    if sf_original and sf_original not in bucket:
                        bucket.append(sf_original)
                    import math as _math
                    scaled = int(round(
                        _FREQ_COMPOUND_BASE
                        + _FREQ_COMPOUND_SCALE
                        * _math.log2(1 + count)
                        / _math.log2(1 + _FREQ_COMPOUND_LOG_REF)
                    ))
                    weight = min(scaled, _FREQ_COMPOUND_MAX)
                    # Add to freq_dict only if not already a stronger standalone entry
                    if tok not in self._freq_dict or self._freq_dict[tok] < weight:
                        self._freq_dict[tok] = weight
                    # English harvested out of a phrase is removed from this
                    # dict too, but in to_freq_dict() rather than here: the
                    # indexes are built one entry at a time, so a token that
                    # is ALSO a standalone headword may not have been seen
                    # yet, and dropping it now would discard a real word.

        # Frequency dict — spell checker word list
        if root:
            self._freq_dict[root] = max(self._freq_dict.get(root, 0), _FREQ_ROOT)
        if sf:
            affix = entry.get("affix_profile", {})
            has_affix = bool(affix.get("prefix") or affix.get("infix"))
            base_freq = _FREQ_SURFACE_EQ if sf == root else _FREQ_SURFACE
            if has_affix:
                base_freq += _FREQ_MORPHO_BONUS
            self._freq_dict[sf] = max(self._freq_dict.get(sf, 0), base_freq)

    def _unindex_entry(self, entry: dict) -> None:
        # Same coalesce-on-None pattern as _index_entry — see comment there.
        sf   = (entry.get("surface_form") or "").lower()
        root = (entry.get("root_lemma")   or "").lower()
        eid  =  entry.get("entry_id")     or ""

        self._id_index.pop(eid, None)

        for word, index in [(sf, self._surface_index), (root, self._root_index)]:
            if not word or word not in index:
                continue
            index[word] = [e for e in index[word] if e.get("entry_id") != eid]
            if not index[word]:
                del index[word]

        # Recompute frequency for affected words
        for word in {sf, root}:
            if not word:
                continue
            still_used = word in self._surface_index or word in self._root_index
            if not still_used:
                self._freq_dict.pop(word, None)
            else:
                new_freq = max(
                    (_FREQ_SURFACE for _ in self._surface_index.get(word, [])),
                    default=0,
                )
                new_freq = max(new_freq, max(
                    (_FREQ_ROOT for _ in self._root_index.get(word, [])),
                    default=0,
                ))
                self._freq_dict[word] = new_freq

    # ------------------------------------------------------------------
    # Core lookups
    # ------------------------------------------------------------------

    def lookup(self, word: str) -> list[dict]:
        """Look up by surface_form, then root_lemma, then accent/apostrophe-
        folded surface (so diacritic-free input still finds canonical entries).
        Exact matches always take priority over folded ones."""
        w = word.lower()
        results = self._surface_index.get(w, [])
        if results:
            return results
        # Fall back: the word itself may be a root_lemma that appears
        # only as the root of a derived form (e.g. 'tah' is root of 'jyntah'
        # but has no standalone surface_form entry).
        root_hits = self._root_index.get(w, [])
        if root_hits:
            return root_hits
        # Final fallback: orthographic fold (kum→kúm, iing→ïing, iuh→'iuh).
        # Only used when nothing matched exactly, so canonical spellings are
        # never shadowed. The folded index is keyed by the folded form of each
        # accented/apostrophe entry, so a PLAIN query (kum) must also be folded
        # (kum) and looked up here — hence no `folded != w` guard.
        folded = self._fold_key(w)
        hits = self._folded_index.get(folded, [])
        if len(hits) < 2:
            return hits
        # Several spellings can fold together — `iap` folds onto both `ïap`
        # and `'ïap`, because the fold strips the apostrophe. Callers take
        # the first match, so the ORDER decides the answer, and unsorted it
        # is just lexicon insertion order: reordering entries (a dedupe, a
        # withdrawal) silently flipped `iap` from "direct match, root ïap"
        # to a decomposition, breaking a golden-corpus ruling. Rank by how
        # close the spelling is to what was asked for, then alphabetically,
        # so the answer depends on the data and not on the file layout.
        return sorted(
            hits,
            key=lambda m: (
                abs(len((m.get("surface_form") or "")) - len(w)),
                (m.get("surface_form") or ""),
            ),
        )

    def lookup_root(self, root: str) -> list[dict]:
        return self._root_index.get(root.lower(), [])

    def lookup_id(self, entry_id: str) -> Optional[dict]:
        return self._id_index.get(entry_id)

    def is_known(self, word: str) -> bool:
        """
        True if the word has a standalone surface-form entry OR appears as
        a component token of a multi-word entry.  This ensures that high-
        frequency words like "shnong" (village) — which only appear inside
        compound phrases in the lexicon — are still recognised as valid
        rather than being sent to the spell checker.

        The compound route consults `_compound_known`, the phonotactically
        screened subset, not `_compound_tokens`. Splitting multi-word
        surface forms also yields bare digraphs (`sh`, `kh`) and English
        words out of embedded translations (`cow`, `and`, `big`), and
        without the screen those answered True here — `cow` passed a spell
        check. The unscreened dict is still what feeds candidate
        generation; see `_phonotactic_ok`.
        """
        w = word.lower()
        if w in self._surface_index or w in self._compound_known:
            return True
        # Fall back to the accent/apostrophe-folded index, exactly as
        # lookup() does. The two paths disagreed before: lookup() resolved
        # 'iathuh' to the entry spelled 'ïathuh' while is_known() said the
        # word was unknown. That was invisible while both spellings existed
        # as entries, and became a real failure once the lexicon settled on
        # one spelling — every plain-spelled word was suddenly rejected.
        return bool(self._folded_index.get(self._fold_key(w)))

    def find_compound_entries(self, token: str) -> list[str]:
        """
        Return every multi-word lexicon entry (surface_form) that contains
        *token* as one of its components.

        Used by the verdict explainer so a user who looks up a word like
        "sohmon" — which lives only inside a phrasal entry such as
        "u sohmon mynta u dang im pleiñ-pleiñ" — can see that host phrase
        rather than the bare "found in a compound" note. Lookup is O(1).
        """
        return list(self._compound_entry_index.get(token.lower(), []))

    def is_attested(self, word: str) -> bool:
        """
        True if *word* appears anywhere in the lexicon data, screened or not.

        Deliberately more permissive than `is_known()`, and used for a
        different question. `is_known()` answers "is this a Khasi word" and
        so applies the phonotactic screen; this answers "did the lexicon
        ever write this down", which is what candidate filtering needs.

        The distinction has teeth. `shafon` is a real word that appears only
        inside a phrasal entry, and `validate()` rejects it — the validator's
        rules do not cover every place name, loan or reduplication. Filtering
        candidates by `is_known()` therefore dropped it from the suggestions
        for `shafan`, costing ~4 points of top-1 on the word benchmark. It is
        still not accepted as a word on its own; it is merely offerable.
        """
        w = word.lower()
        return (w in self._surface_index or w in self._compound_tokens
                or w in self._root_index or w in self._corpus_forms)

    # ------------------------------------------------------------------
    # Corpus-derived candidate supply
    # ------------------------------------------------------------------

    def register_corpus_forms(self, forms) -> int:
        """Admit *forms* as attested-but-not-known candidate material.

        These are frequent corpus types the lexicon does not record. They
        become offerable as corrections — without them a word like
        `pyntreikam`, 1,940 occurrences and not a headword, is not in the
        list to be ranked and no typo of it can be corrected.

        They deliberately do NOT become known words: `is_known()` does not
        consult this set, so the confidence vote is unchanged and a corpus
        type still has to earn acceptance on its own signals. Candidate
        supply and acceptance are separate questions, which is why they
        have separate predicates.

        Returns the number newly registered.
        """
        before = len(self._corpus_forms)
        self._corpus_forms.update(
            w.lower() for w in forms if w and " " not in w
        )
        return len(self._corpus_forms) - before

    def register_corpus_acceptance(self, forms) -> int:
        """Let *forms* earn the frequency signal in the confidence vote.

        For corpus words `corpus_pool` has already admitted as candidates.
        Without it the pool offered words the checker then rejected: `jyla`
        was answered with `jylla`, and `jylla` — 24,252 corpus occurrences —
        was flagged the moment the writer accepted it. `is_known()` is still
        untouched; the vote reads this set, nothing else does.

        Returns the number newly registered.
        """
        before = len(self._corpus_accept)
        self._corpus_accept.update(
            w.lower() for w in forms if w and " " not in w
        )
        return len(self._corpus_accept) - before

    def is_corpus_accepted(self, word: str) -> bool:
        """True when the corpus attests *word* often enough to accept it."""
        return word.lower() in self._corpus_accept

    def is_corpus_form(self, word: str) -> bool:
        """True when *word* is offerable only because the corpus attests it.

        Provenance: the lexicon did not write this down, people did. Callers
        that show a suggestion to a human should say which of the two they
        are looking at.
        """
        return word.lower() in self._corpus_forms

    def is_known_root(self, root: str) -> bool:
        return root.lower() in self._root_index

    def all_surface_forms(self) -> list[str]:
        """
        Return every word the spell checker should consider as a candidate.

        Includes:
          1. Standalone surface forms (single-word entries)
          2. Root lemmas (single-word)
          3. Individual tokens extracted from multi-word surface forms
             (e.g. "shnong" from "dorbar shnong", "mynsiem" from "long mynsiem")

        This ensures that important Khasi words which only appear inside
        compound or phrasal entries are still reachable by the Levenshtein
        scanner — fixing the class of bugs where a common word like "shnong"
        was never suggested because it had no standalone lexicon entry.
        """
        combined: set[str] = set(self._surface_index.keys())
        combined.update(self._root_index.keys())
        # Harvested phrase tokens, minus the English ones. A token that is
        # ALSO a headword or a root stays regardless: the filter exists to
        # stop English entering the pool through phrase-splitting, not to
        # remove a word the lexicon records in its own right.
        combined.update(t for t in self._compound_tokens
                        if t not in _ENGLISH_IN_PHRASES
                        or t in self._surface_index or t in self._root_index)
        # Corpus-derived candidates. This method defines the candidate pool
        # — it is what the delete index and the fallback scan are built from
        # — so admitting them here is what makes them reachable at all.
        # `is_known()` still refuses them, so nothing here becomes a word.
        combined.update(self._corpus_forms)
        # Exclude multi-word keys (safety net — surface_index shouldn't have them
        # but compound_tokens definitely might if the regex split was incomplete)
        return sorted(w for w in combined if " " not in w)

    def all_entries(self) -> list[dict]:
        return self._lexicon

    # ------------------------------------------------------------------
    # Spell-checker interface  (single source of truth)
    # ------------------------------------------------------------------

    def to_freq_dict(self) -> dict[str, int]:
        """
        Return a word-frequency dict for autocorrect.Speller.
        Derived entirely from khasi_db.json — no separate kh.json used.

        English harvested out of multi-word entries is withheld, for the same
        reason `all_surface_forms()` withholds it: this dict is the speller's
        other candidate source, and filtering only the delete index still left
        `mayor` offered for `mayorr`. Filtering happens here, at read time,
        because the indexes this consults are built one entry at a time and a
        token that is also a standalone headword may not have been indexed yet
        while `_index_entry` is running. See `_ENGLISH_IN_PHRASES`.
        """
        return {w: n for w, n in self._freq_dict.items()
                if w not in _ENGLISH_IN_PHRASES
                or w in self._surface_index or w in self._root_index}

    # ------------------------------------------------------------------
    # Enriched-companion access
    # ------------------------------------------------------------------

    # Cache of stitched multi-word phonology so the lookup work only runs
    # once per entry_id, not on every API call.
    _STITCH_CACHE_ATTR = "_phonology_stitch_cache"

    def _stitch_multiword_phonology(self, enriched: dict) -> dict:
        """When a multi-word entry's phonology.derived only covers some of
        its surface words (a known data-quality issue in the v3.2 dataset
        — e.g. "i syiar iba dang lung jiap-jiap" only carries syllables
        for "i syiar"), look up each whitespace-separated surface token's
        own standalone entry and concatenate their syllables / pattern /
        IPA-friendly count to produce a complete phrase-level phonology.

        Pure read-time augmentation — never mutates the cached enriched
        dict; returns a new dict (or the original when no stitching is
        needed). Editorial fields (morphology.structure, compound_roots)
        are left alone since correct stitching of those needs linguistic
        judgement.
        """
        surface = ((enriched or {}).get("form") or {}).get("surface", "") or ""
        if not surface or " " not in surface:
            return enriched

        tokens = surface.split()
        existing = ((enriched.get("phonology") or {}).get("derived") or {})
        existing_syllables = list(existing.get("syllables") or [])
        existing_pattern_parts = (existing.get("pattern") or "").split(".") \
            if existing.get("pattern") else []

        # Lazy per-instance cache keyed by entry_id.
        cache = getattr(self, self._STITCH_CACHE_ATTR, None)
        if cache is None:
            cache = {}
            setattr(self, self._STITCH_CACHE_ATTR, cache)
        eid = enriched.get("entry_id")
        if eid and eid in cache:
            return cache[eid]

        # ── Per-token expected syllable lookup ─────────────────────────────
        # For each surface token, fetch the standalone entry's syllables and
        # pattern. Track how many tokens the EXISTING syllabification can be
        # explained by, so we preserve correct partial data and only append
        # the missing tail.
        per_token: list[tuple[list[str], list[str]]] = []  # (syllables, pattern_parts)
        for tok in tokens:
            matches = self.lookup(tok.lower())
            tok_syllables: list[str] = []
            tok_pattern_parts: list[str] = []
            if matches:
                tok_enr = self._enriched_by_id.get(matches[0].get("entry_id", ""))
                tok_phon = ((tok_enr or {}).get("phonology") or {}).get("derived") or {}
                tok_syllables = list(tok_phon.get("syllables") or [])
                p = tok_phon.get("pattern") or ""
                if p:
                    tok_pattern_parts = p.split(".")
            if not tok_syllables:
                # Unknown / un-syllabified token — best-effort placeholder.
                tok_syllables = [tok]
                tok_pattern_parts = ["?"]
            per_token.append((tok_syllables, tok_pattern_parts))

        # Walk left to right; the existing data is assumed to cover the
        # FIRST N tokens whose cumulative expected syllable count matches.
        # The remaining tokens get stitched in from per_token.
        running = 0
        covered_tokens = 0
        for sylls, _ in per_token:
            n = len(sylls)
            if running + n <= len(existing_syllables):
                running += n
                covered_tokens += 1
            else:
                break

        # Build the merged output:
        #   prefix = existing data for the covered tokens (preserves any
        #            curated syllabification, e.g. "jingsaw" → "jing.saw")
        #   suffix = stitched per-token for everything after `covered_tokens`
        merged_syllables = existing_syllables[:running] + [
            s for sylls, _ in per_token[covered_tokens:] for s in sylls
        ]
        merged_pattern_parts = existing_pattern_parts[:running] + [
            p for _, parts in per_token[covered_tokens:] for p in parts
        ]

        # Only adopt the merge if it actually adds coverage.
        if len(merged_syllables) <= len(existing_syllables):
            if eid:
                cache[eid] = enriched
            return enriched

        # Deep copy so the cached _enriched_by_id entry stays untouched.
        import copy
        result = copy.deepcopy(enriched)
        derived = result.setdefault("phonology", {}).setdefault("derived", {})
        derived["syllables"] = merged_syllables
        derived["syllable_count"] = len(merged_syllables)
        derived["pattern"] = ".".join(merged_pattern_parts)
        # Marker so downstream consumers can render an "auto-completed" badge.
        derived["_stitched"] = True
        if eid:
            cache[eid] = result
        return result

    def get_enriched(self, entry_id: str) -> Optional[dict]:
        """Return the v3.2 enriched form of an entry by ID, or None.

        API responses use this to attach the full nested object alongside
        the v2.0 flat shape, letting the frontend opt into richer rendering
        (syllable breakdown, semantic domain, validation, etc.) without
        breaking clients that still read flat fields.

        Multi-word entries with incomplete phonology are auto-stitched
        from their per-token standalone entries before being returned.
        """
        enr = self._enriched_by_id.get(entry_id)
        if enr is None:
            return None
        return self._stitch_multiword_phonology(enr)

    def with_enriched(self, flat_entry: dict) -> dict:
        """Shallow-copy a flat entry and attach `_enriched` if available.
        The attached entry goes through the same multi-word phonology
        auto-stitch as `get_enriched`."""
        out = dict(flat_entry)
        eid = flat_entry.get("entry_id")
        if eid:
            enr = self._enriched_by_id.get(eid)
            if enr is not None:
                out["_enriched"] = self._stitch_multiword_phonology(enr)
        return out

    # ------------------------------------------------------------------
    # Write operations  (indexes + freq_dict kept in sync)
    # ------------------------------------------------------------------

    def _invalidate_stitch_cache(self) -> None:
        """Drop any cached phrase-level phonology stitches. Called on every
        write because edits to a single-token entry can change the syllables
        of any multi-word phrase that contains it."""
        cache = getattr(self, self._STITCH_CACHE_ATTR, None)
        if cache:
            cache.clear()

    def add_entry(self, entry: dict) -> None:
        """Add entry to in-memory lexicon. Persists immediately when PG-backed.

        Accepts either v2.0 flat shape (the legacy frontend POST shape) or
        v3.2 enriched shape. Stores the flat shape in memory and the enriched
        shape in PG (translating as needed).
        """
        if _is_enriched_entry(entry):
            enriched = entry
            flat = _enriched_to_flat(entry)
        else:
            flat = entry
            enriched = _flat_to_enriched(entry)
        eid = flat.get("entry_id", "")
        if eid:
            self._enriched_by_id[eid] = enriched
        self._lexicon.append(flat)
        self._index_entry(flat)
        self._invalidate_stitch_cache()
        if self._database_url:
            _pg_upsert_entry(self._database_url, enriched)

    def update_entry(self, entry_id: str, updates: dict) -> Optional[dict]:
        """
        Update fields of an existing entry in-place. Persists immediately
        when PG-backed.

        `updates` is a flat-shape partial dict. The merge happens against
        the in-memory flat entry; the enriched companion is regenerated and
        upserted to PG.
        """
        entry = self._id_index.get(entry_id)
        if not entry:
            return None
        self._unindex_entry(entry)
        self._lexicon = [e for e in self._lexicon if e.get("entry_id") != entry_id]
        merged_flat = {**entry, **updates, "entry_id": entry_id}
        enriched = _flat_to_enriched(
            merged_flat, base=self._enriched_by_id.get(entry_id),
        )
        self._enriched_by_id[entry_id] = enriched
        self._lexicon.append(merged_flat)
        self._index_entry(merged_flat)
        self._invalidate_stitch_cache()
        if self._database_url:
            _pg_upsert_entry(self._database_url, enriched)
        return merged_flat

    def delete_entry(self, entry_id: str) -> bool:
        """Remove entry by ID. Persists immediately when PG-backed."""
        entry = self._id_index.get(entry_id)
        if not entry:
            return False
        self._lexicon = [e for e in self._lexicon if e.get("entry_id") != entry_id]
        self._enriched_by_id.pop(entry_id, None)
        self._unindex_entry(entry)
        self._invalidate_stitch_cache()
        if self._database_url:
            _pg_delete_entry(self._database_url, entry_id)
        return True

    def save(self, path: Optional[str | Path] = None) -> None:
        """Persist the in-memory lexicon to disk.

        No-op when DATABASE_URL is set — PG is the source of truth and the
        write methods already persisted. The JSON file would only end up
        out of sync, so we don't touch it.
        """
        if self._database_url:
            print("[khasi-nlp] save(): PG-backed mode — skipping JSON write")
            return
        out_path = Path(path) if path else self._json_path
        # Reconstruct enriched shape on save so the JSON stays canonical.
        lexicon_to_save = [
            self._enriched_by_id.get(e.get("entry_id", "")) or _flat_to_enriched(e)
            for e in self._lexicon
        ]
        # Order: preserved-but-unmanaged blocks LAST so the file diff stays
        # readable. Putting them after lexicon means a save() that only
        # mutates an entry doesn't reflow the entire file.
        payload = {
            "meta":          self._meta,
            "phonology":     self._phonology,
            "morphology":    self._morphology,
            "lexicon":       lexicon_to_save,
            "demo_examples": self._demo_examples,
        }
        # Preserve every top-level block KhasiDB doesn't itself manage
        # (quarantine, production_readiness, grammar_schema, etc.) so a
        # JSON-mode write never silently drops v4.3 metadata. See
        # _extra_blocks comment in __init__.
        for k, v in self._extra_blocks.items():
            payload[k] = v
        with open(out_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

    # ------------------------------------------------------------------
    # Stats & metadata
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        entries = self._lexicon
        # Honest lexeme count (Phase A): single-word lexical items only.
        # `total_entries` counts every row, including multi-word phrases,
        # idioms and example sentences mis-filed in the lexicon.
        lexeme_count = sum(1 for e in entries if e.get("kind") == "lexeme")
        lexeme_glossed = sum(
            1 for e in entries
            if e.get("kind") == "lexeme" and (e.get("english_gloss") or "").strip()
        )
        return {
            "total_entries":        len(entries),
            "lexeme_count":         lexeme_count,
            "lexeme_glossed_count": lexeme_glossed,
            "kind_counts": {
                k: sum(1 for e in entries if e.get("kind") == k)
                for k in ("lexeme", "compound_or_collocation", "phrase", "example")
            },
            "unique_surface_forms": len(self._surface_index),
            "unique_roots":         len(self._root_index),
            "spell_checker_words":  len(self._freq_dict),
            "prefixed_forms":       sum(
                1 for e in entries if e.get("affix_profile", {}).get("prefix")
            ),
            "infixed_forms":        sum(
                1 for e in entries if e.get("affix_profile", {}).get("infix")
            ),
            "compound_forms":       sum(
                1 for e in entries if e.get("morphological_flags", {}).get("is_compound")
            ),
            "loan_words":           sum(
                1 for e in entries
                if e.get("loan_origin", "none") not in ("none", None, "")
            ),
            "reduplication_forms":  sum(
                1 for e in entries
                if e.get("morphological_flags", {}).get("reduplication_type", "none") != "none"
            ),
            "tien_kynnoh_forms":    sum(
                1 for e in entries
                if e.get("morphological_flags", {}).get("is_tien_kynnoh")
            ),
            "shim_kylliang_forms":  sum(
                1 for e in entries
                if e.get("loan_origin", "none") not in ("none", None, "")
            ),
        }

    @property
    def phonology(self) -> dict:
        return self._phonology

    @property
    def morphology(self) -> dict:
        return self._morphology

    @property
    def meta(self) -> dict:
        return self._meta

    @property
    def demo_examples(self) -> dict:
        return self._demo_examples

    def next_entry_id(self) -> str:
        nums = []
        for eid in (e.get("entry_id", "") for e in self._lexicon):
            try:
                nums.append(int(eid.replace("kh_", "")))
            except ValueError:
                pass
        return f"kh_{max(nums, default=0) + 1:04d}"

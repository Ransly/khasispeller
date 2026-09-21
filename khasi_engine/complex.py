"""
complex.py — Phase 4: Complex Word Processor
Khasi NLP Engine · Ki Sawa Bad Ki Dur Kyntien Jong Ka Ktien Khasi (Badaplin War, 2001)

Detects and analyses:
  1. Total reduplication           — mano-mano, ang-ang, mar-mar
  2. Partial vowel reduplication   — khuk-khak, ngut-nget, jyrwit-jyrwat
  3. Reduplication with interfix A — suki-pa-suki, mano-re-mano
  4. Reduplication with interfix B — thui-ly-thui, jain-ryn-jain
  5. Triplication                  — jrain-rain-rain, khrui-rui-rui
  6. Registered compound words     — tikmie (kti+kmie), liengsuin (lieng+suin)
  7. Hyphenated compound verbs     — ai-bainong, ai-kait
  8. Tien Kynnoh pairs             — khun-kti (poetic/emphatic pairs)
       subtype a: pdeng — both elements have independent meanings (thiah/dem)
       subtype b: non_lexical — second element exists only in the pair (khun/kti)
  9. Shim Kylliang (borrowed words) — ball, drama, phadar, bang (Ch.IX)
       detected via loan_origin field; subtype records the source language
"""

from __future__ import annotations
import re

# ---------------------------------------------------------------------------
# Interfix sets (from textbook Ch.VIII)
# ---------------------------------------------------------------------------

INTERFIXES_A: frozenset[str] = frozenset(["re", "shi", "pa"])
INTERFIXES_B: frozenset[str] = frozenset(["ly", "ryn", "kyn"])
ALL_INTERFIXES: frozenset[str] = INTERFIXES_A | INTERFIXES_B

# ï is a vowel nucleus in Khasi orthography — include in VOWEL_SET
VOWELS: str = "aáeéiíoóuúyýï"
VOWEL_SET: frozenset[str] = frozenset(VOWELS)

# ---------------------------------------------------------------------------
# Tien Kynnoh subtype constants (Ch.X pp.109–113)
# ---------------------------------------------------------------------------
# The textbook distinguishes two classes of tien-kynnoh pairs:
#   pdeng        — both words carry independent lexical meanings; the second
#                  is a synonym or near-synonym from a different register
#                  (e.g. thiah / dem, shong / sah, ïaid / ïeng).
#   non_lexical  — the second element (the kynnoh proper) has no independent
#                  meaning outside the pair; it exists only as the emphatic
#                  echo (e.g. khun / kti, khun / pyrsa, khun / hajar).
TK_SUBTYPE_PDENG       = "pdeng"
TK_SUBTYPE_NON_LEXICAL = "non_lexical"
TK_SUBTYPE_UNKNOWN     = "unknown"

# Descriptions for each type
TYPE_DESCRIPTIONS: dict[str, str] = {
    "total_reduplication":          "Total reduplication — base form repeated identically for emphasis",
    "partial_reduplication_vowel":  "Partial reduplication with vowel ablaut — same consonant skeleton, different vowel",
    "reduplication_interfix_A":     "Reduplication with A-type interfix (re / shi / pa) for intensification",
    "reduplication_interfix_B":     "Reduplication with B-type interfix (ly / ryn / kyn) for intensification",
    "triplication":                 "Triplication — three-part repetition for extreme emphasis",
    "compound_registered":          "Registered compound word (documented in lexicon)",
    "compound_hyphenated":          "Hyphenated compound — two roots joined with hyphen",
    "compound_fusion":              "Fused compound — roots merged with optional sound deletion",
    "tien_kynnoh":                  "Tien Kynnoh — poetic/emphatic word pair (second element may be non-lexical)",
    "tien_kynnoh_pdeng":            "Tien Kynnoh (pdeng subtype) — both elements have independent lexical meanings",
    "tien_kynnoh_non_lexical":      "Tien Kynnoh (non-lexical subtype) — second element exists only in this pair",
    "shim_kylliang":                "Shim Kylliang — borrowed/loan word integrated into Khasi phonology (Ch.IX)",
    "potential_compound":           "Potential compound — hyphenated form, roots not confirmed in lexicon",
}


# ---------------------------------------------------------------------------
# Main detector
# ---------------------------------------------------------------------------

def detect(word: str, lexicon_lookup) -> dict:
    """
    Attempt to classify *word* as a complex morphological form.

    Parameters
    ----------
    word           : str
    lexicon_lookup : callable(str) -> list[dict]

    Returns
    -------
    dict with keys:
        detected      : bool
        type          : str | None
        description   : str | None
        components    : list[str]   — identified base forms / roots
        details       : dict        — type-specific metadata
        db_entry      : list[dict]  — matching lexicon entries if found
    """
    result: dict = {
        "detected": False,
        "type": None,
        "description": None,
        "components": [],
        "details": {},
        "db_entry": [],
    }

    w = word.lower().strip()

    # ── Check DB first (registered compounds / tien kynnoh / loans) ──
    db_entries = lexicon_lookup(w)
    for entry in db_entries:
        mflags = entry.get("morphological_flags", {})

        # ── Shim Kylliang (Ch.IX) — check before tien_kynnoh so loan words
        #    with poetic registers are not misclassified as tien_kynnoh ──
        loan_origin = entry.get("loan_origin", "none")
        if loan_origin and loan_origin not in ("none", None, ""):
            # Determine assimilation subtype from the DB entry notes if present
            notes = entry.get("notes", "")
            result.update({
                "detected": True,
                "type": "shim_kylliang",
                "description": TYPE_DESCRIPTIONS["shim_kylliang"],
                "components": [entry.get("root_lemma", w)],
                "db_entry": db_entries,
                "details": {
                    "source_language": loan_origin,
                    "gloss": entry.get("english_gloss", ""),
                    "notes": notes,
                    # The textbook (Ch.IX pp.94–107) distinguishes:
                    #   "direct"           — phonological adaptation of foreign word
                    #   "calque"           — loan-translation (structural borrowing)
                    #   "hybrid"           — Khasi root + foreign element combined
                    # The entry's loan_origin field records the source language;
                    # the grammatical_class records what class it entered Khasi as.
                    "integration_type": _infer_loan_type(entry),
                },
            })
            return result

        # ── Tien Kynnoh (Ch.X pp.109–113) ────────────────────────────
        if mflags.get("is_tien_kynnoh"):
            # Determine subtype from DB entry
            tk_subtype = mflags.get("tien_kynnoh_subtype", TK_SUBTYPE_UNKNOWN)
            # Infer subtype from constituent fields if not explicitly stored
            if tk_subtype == TK_SUBTYPE_UNKNOWN:
                kynnoh = mflags.get("kynnoh_constituent")
                tk_subtype = _infer_tk_subtype(w, kynnoh, lexicon_lookup)
            type_key = (
                "tien_kynnoh_pdeng"       if tk_subtype == TK_SUBTYPE_PDENG
                else "tien_kynnoh_non_lexical" if tk_subtype == TK_SUBTYPE_NON_LEXICAL
                else "tien_kynnoh"
            )
            result.update({
                "detected": True,
                "type": type_key,
                "description": TYPE_DESCRIPTIONS[type_key],
                "components": [w],
                "db_entry": db_entries,
                "details": {
                    "pair": w,
                    "gloss": entry.get("english_gloss", ""),
                    "tien_kynnoh_subtype": tk_subtype,
                    "kynnoh_constituent": mflags.get("kynnoh_constituent"),
                    "primary_constituent": entry.get("root_lemma", ""),
                },
            })
            return result

        # ── Registered compound ───────────────────────────────────────
        if mflags.get("is_compound"):
            roots    = mflags.get("compound_roots", [])
            deletion = mflags.get("deletion_rule")
            result.update({
                "detected": True,
                "type": "compound_registered",
                "description": TYPE_DESCRIPTIONS["compound_registered"],
                "components": roots,
                "db_entry": db_entries,
                "details": {
                    "roots": roots,
                    "deletion_rule": deletion,
                    "gloss": entry.get("english_gloss", ""),
                },
            })
            return result

        # ── Reduplication ─────────────────────────────────────────────
        if mflags.get("reduplication_type", "none") != "none":
            redupe_type = mflags["reduplication_type"]
            interfix    = mflags.get("interfix")
            result.update({
                "detected": True,
                "type": redupe_type,
                "description": TYPE_DESCRIPTIONS.get(redupe_type, "Reduplication"),
                "components": [entry.get("root_lemma", w)],
                "db_entry": db_entries,
                "details": {
                    "base": entry.get("root_lemma", ""),
                    "interfix": interfix,
                },
            })
            return result

    # ── Pattern-based detection (for words not yet in DB) ────────────
    if "-" in w:
        detected = _detect_from_pattern(w, lexicon_lookup)
        if detected:
            result.update(detected)
            return result

    # ── Solid-compound splitting (War Ch.VIII "Ki Kyntien Khleh") ────
    # Compounding is Khasi's dominant word-formation strategy (65.7% of the
    # lexicon), but most compounds are written SOLID (wanlam, mihngi,
    # donkam) with no hyphen, so the pattern matcher above never sees them.
    # This splitter proposes a two-root decomposition for a solid word not
    # otherwise resolved. It is deliberately CONSERVATIVE (see _split_solid_
    # compound): both parts must be lexicon words, neither a function word,
    # and it never overrides the affix analysers — analyser.py only reaches
    # Phase 4 when Phases 2-3 did not resolve the word.
    if "-" not in w and "'" not in w:
        compound = _split_solid_compound(w, lexicon_lookup)
        if compound:
            result.update(compound)
            return result

    return result



def _infer_loan_type(entry: dict) -> str:
    """
    Infer the integration type of a borrowed word from its DB entry fields.

    The textbook (Ch.IX) describes three patterns:
      direct  — the foreign phoneme sequence is adapted to Khasi phonotactics
                 (final consonant cluster broken, aspiration removed, etc.)
      calque  — the meaning of a foreign phrase is translated using native roots
                 ('korsuhjaiñ' for 'sewing machine')
      hybrid  — a Khasi root is combined with a borrowed element
                 (e.g. Khasi 'soh' + English 'apple' → 'soh-apple')

    In the absence of an explicit field, this function heuristically infers
    the type from loan_origin and the grammatical structure of the entry.
    A dedicated 'shim_kylliang_type' field in the DB would remove the guesswork
    — see the DB schema notes in database.py.
    """
    loan_origin = entry.get("loan_origin", "")
    if "calque" in loan_origin:
        return "calque"
    # Check notes for calque markers
    notes = (entry.get("notes") or "").lower()
    if "calque" in notes or "loan-translation" in notes:
        return "calque"
    # If the surface form contains a hyphen it is likely a hybrid compound
    sf = entry.get("surface_form", "")
    if "-" in sf:
        return "hybrid"
    return "direct"


def _infer_tk_subtype(
    pair_form: str,
    kynnoh_constituent: str | None,
    lexicon_lookup,
) -> str:
    """
    Infer the Tien Kynnoh subtype for a pair that does not have an explicit
    'tien_kynnoh_subtype' field in the DB.

    Logic (derived from Ch.X pp.109–113):
      - If the kynnoh (second element) is independently found in the lexicon,
        both words carry meaning → subtype is 'pdeng'.
      - If the kynnoh is not in the lexicon, it likely exists only inside this
        pair → subtype is 'non_lexical'.
      - If kynnoh_constituent is not stored, attempt to extract it from the
        hyphenated pair form and apply the same test.
    """
    if not kynnoh_constituent and "-" in pair_form:
        parts = pair_form.split("-", 1)
        kynnoh_constituent = parts[1] if len(parts) == 2 else None

    if not kynnoh_constituent:
        return TK_SUBTYPE_UNKNOWN

    in_lexicon = bool(lexicon_lookup(kynnoh_constituent))
    return TK_SUBTYPE_PDENG if in_lexicon else TK_SUBTYPE_NON_LEXICAL


def _detect_from_pattern(w: str, lexicon_lookup) -> dict | None:
    """Pattern-match on the hyphenated structure of the word."""
    parts = w.split("-")

    # ── Triplication: A-B-B where B is a suffix/reduction of A ───────
    if len(parts) == 3 and parts[1] == parts[2]:
        return {
            "detected": True,
            "type": "triplication",
            "description": TYPE_DESCRIPTIONS["triplication"],
            "components": [parts[0], parts[1]],
            "details": {
                "full_form": parts[0],
                "repeated": parts[1],
                "pattern": f"{parts[0]}-{parts[1]}-{parts[1]}",
            },
        }

    # ── Reduplication with interfix: A-interfix-A ─────────────────────
    if len(parts) == 3 and parts[0] == parts[2]:
        mid = parts[1]
        if mid in INTERFIXES_A:
            return {
                "detected": True,
                "type": "reduplication_interfix_A",
                "description": TYPE_DESCRIPTIONS["reduplication_interfix_A"],
                "components": [parts[0]],
                "details": {"base": parts[0], "interfix": mid, "pattern": f"{parts[0]}-{mid}-{parts[0]}"},
            }
        if mid in INTERFIXES_B:
            return {
                "detected": True,
                "type": "reduplication_interfix_B",
                "description": TYPE_DESCRIPTIONS["reduplication_interfix_B"],
                "components": [parts[0]],
                "details": {"base": parts[0], "interfix": mid, "pattern": f"{parts[0]}-{mid}-{parts[0]}"},
            }

    # ── Two-part forms ────────────────────────────────────────────────
    if len(parts) == 2:
        p1, p2 = parts

        # Total reduplication: A-A
        if p1 == p2:
            return {
                "detected": True,
                "type": "total_reduplication",
                "description": TYPE_DESCRIPTIONS["total_reduplication"],
                "components": [p1],
                "details": {"base": p1, "pattern": f"{p1}-{p1}"},
            }

        # Partial vowel reduplication: same consonant skeleton, different vowel
        cs1 = _consonant_skeleton(p1)
        cs2 = _consonant_skeleton(p2)
        if cs1 == cs2 and cs1 and p1 != p2:
            return {
                "detected": True,
                "type": "partial_reduplication_vowel",
                "description": TYPE_DESCRIPTIONS["partial_reduplication_vowel"],
                "components": [p1],
                "details": {
                    "part1": p1,
                    "part2": p2,
                    "shared_consonants": cs1,
                },
            }

        # Hyphenated compound: try to look up both roots
        m1 = lexicon_lookup(p1)
        m2 = lexicon_lookup(p2)
        if m1 or m2:
            return {
                "detected": True,
                "type": "compound_hyphenated",
                "description": TYPE_DESCRIPTIONS["compound_hyphenated"],
                "components": [p1, p2],
                "details": {
                    "root1": p1, "root1_in_lexicon": bool(m1),
                    "root2": p2, "root2_in_lexicon": bool(m2),
                },
            }

        # Potential compound — neither root confirmed
        return {
            "detected": True,
            "type": "potential_compound",
            "description": TYPE_DESCRIPTIONS["potential_compound"],
            "components": [p1, p2],
            "details": {"root1": p1, "root2": p2},
        }

    return None


def _consonant_skeleton(s: str) -> str:
    """Strip all vowels from *s* to get the consonant skeleton."""
    return "".join(c for c in s if c not in VOWEL_SET)


# ---------------------------------------------------------------------------
# Solid-compound splitter (War Ch.VIII)
# ---------------------------------------------------------------------------

# Minimum length of each compound part. Khasi roots can be a single CV(C)
# syllable (um 'water', ja 'rice', doh 'meat'), but allowing 2-char parts
# invites false splits, so we require each part to be a real lexicon word
# AND at least this long. 2 keeps um/ja/soh while excluding bare letters.
_MIN_PART_LEN = 2


def _compound_type(pos1: str, pos2: str) -> str:
    """Label the compound by its constituent PoS pair (War Ch.VIII typology:
    N+N endocentric, V+N / N+V, V+V coordinate, V+ADJ, N+ADJ, etc.)."""
    a = (pos1 or "?").lower()[:1] or "?"
    b = (pos2 or "?").lower()[:1] or "?"
    tag = {"n": "N", "v": "V", "a": "ADJ", "r": "ADV"}
    return f"{tag.get(a, '?')}+{tag.get(b, '?')}"


def _spelled_as(part: str, matches: list[dict]) -> bool:
    """True when *part* is how one of its lexicon matches is actually SPELLED.

    db.lookup() deliberately falls through to an orthographic fold (kum→kúm,
    iing→ïing, iuh→'iuh) so lenient real-world input still finds canonical
    entries. That is right for looking a word UP and wrong for licensing a
    compound seam: `kaa` is not a word, it merely folds onto `ka'a` (Kapok
    tree) once the apostrophe is stripped — which was enough to split
    `lehkaa` into leh + kaa and accept it. A genuine compound of leh + ka'a
    is spelled lehka'a, so a seam has to match a real spelling: the entry's
    own surface form, or its root lemma.
    """
    p = part.lower()
    for m in matches:
        if (m.get("surface_form") or "").lower() == p:
            return True
        if (m.get("root_lemma") or "").lower() == p:
            return True
    return False


def _split_solid_compound(w: str, lexicon_lookup) -> dict | None:
    """Propose a two-root split for a solid (un-hyphenated) compound.

    Conservative acceptance gates — a split is only returned when ALL hold:
      1. Both parts are >= _MIN_PART_LEN characters.
      2. Both parts are attested lexicon words (lexicon_lookup non-empty)
         AND spelled the way the lexicon spells them — an orthographic-fold
         match does not license a seam (see _spelled_as).
      3. Neither part is a function word / free morpheme / bare clitic
         (would make the string a phrase, not a compound — Phase A Rule 4).
      4. Both parts are phonotactically valid on their own.

    When several split points qualify, prefer the one whose parts are BOTH
    open-class content words, then the most balanced split (parts closest in
    length) — this favours mih|ngi over m|ihngi-style artefacts.

    Returns a Phase-4 result dict or None.
    """
    # Lazy imports to avoid a module-level cycle (phonology/morphology import
    # order) and to pick up PG-injected data at call time.
    from khasi_engine import morphology as _morph
    try:
        from khasi_engine.phonology import is_phonotactically_valid as _phono_valid
    except Exception:
        _phono_valid = lambda s: True  # noqa: E731 — fail open if unavailable

    if len(w) < 2 * _MIN_PART_LEN:
        return None

    free = {m.lower() for m in getattr(_morph, "FREE_MORPHEMES", [])}
    clitics = {c.lower() for c in getattr(_morph, "CLITICS", {})}
    blocked = free | clitics
    # Registered prefix morphemes (jing, pyn, nong, mar, kyn, pyr, …). If the
    # first part IS a prefix, this is an affix boundary, not a compound seam —
    # Phase 2 already had first refusal and (for fossilised prefixes like mar-,
    # kyn-) deliberately declined to derive. Resurrecting mar+thoh as a
    # "compound" would contradict that ruling, so we never split there.
    prefix_keys = {k.lower() for k in getattr(_morph, "PREFIXES", {})}

    candidates: list[tuple] = []
    for i in range(_MIN_PART_LEN, len(w) - _MIN_PART_LEN + 1):
        p1, p2 = w[:i], w[i:]
        if p1 in blocked or p2 in blocked:
            continue
        if p1 in prefix_keys:
            continue
        m1 = lexicon_lookup(p1)
        m2 = lexicon_lookup(p2)
        if not (m1 and m2):
            continue
        # Gate 2b: both parts must be spelled the way the lexicon spells
        # them — a folded match is not a compound part. See _spelled_as().
        if not (_spelled_as(p1, m1) and _spelled_as(p2, m2)):
            continue
        if not (_phono_valid(p1) and _phono_valid(p2)):
            continue
        pos1 = (m1[0].get("grammatical_class") or "").lower()
        pos2 = (m2[0].get("grammatical_class") or "").lower()
        open_class = {"noun", "verb", "adjective", "adverb"}
        both_open = pos1 in open_class and pos2 in open_class
        balance = abs(len(p1) - len(p2))
        # Sort key: prefer both-open-class, then balanced, then earlier split.
        candidates.append((not both_open, balance, i, p1, p2, m1, m2, pos1, pos2))

    if not candidates:
        return None

    candidates.sort()
    _, _, _, p1, p2, m1, m2, pos1, pos2 = candidates[0]

    return {
        "detected": True,
        "type": "compound_fusion",
        "description": TYPE_DESCRIPTIONS["compound_fusion"],
        "components": [p1, p2],
        "db_entry": [],
        "details": {
            "root1": p1, "root1_pos": pos1, "root1_gloss": m1[0].get("english_gloss", ""),
            "root2": p2, "root2_pos": pos2, "root2_gloss": m2[0].get("english_gloss", ""),
            "compound_pos_pattern": _compound_type(pos1, pos2),
            "split_confidence": "proposed",
            "alternatives_considered": len(candidates),
        },
    }


def get_type_description(type_key: str) -> str:
    """Return a human-readable description for a complex word type."""
    return TYPE_DESCRIPTIONS.get(type_key, type_key)

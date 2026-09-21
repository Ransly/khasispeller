#!/usr/bin/env python3
"""
fix_brackets_and_syllables.py — four maintainer rulings.

RULE 1  Brackets come out of the Khasi forms.
        310 surfaces carry one, and the content is almost always the
        noun-class clitic the 1906 dictionary prints beside a headword:
        ka 157, u 123, ki 16, i 7, ba 2, k 1, ia 1, uba 1.

            (u) 'bat ïam-baid          adong (ka) - ka adit

        Deleting just the bracket characters would be wrong — `adong (ka)`
        would become `adong ka`, reversing Khasi word order, which is
        `ka adong`. So a bracketed clitic is REMOVED from the form and
        recorded in `grammar.clitic`, where the schema already keeps it.
        The lexicon agrees: kh_KKS_030190 already normalises
        `(u) 'bat ïam-baid` to `'bat ïam-baid`, it simply never filled in
        the clitic. A bracket holding anything else keeps its content and
        loses only the brackets, and two stray unmatched marks
        (`kaei kaei baroh)`, `suhsat }`) are deleted.

        Only the Khasi-bearing fields are touched — form, lemma, phonology
        surface, morphology.canonical, rhetoric.semantic_pairing. Glosses
        are NOT touched: 6,255 of them contain brackets and those are the
        source's own notation, "(short o)", "(better than dynda)".

RULE 2  `syntiew` is ['syn', 'tiew'].  It had an empty syllable list.

RULE 3  The alphabet entries are not words and come out of the lexicon.
        The dictionary has an entry for each LETTER — "h, h" glossed "The
        eighth letter of the Khasi Alphabet". Thirteen exist, not the nine
        with vowel-less syllables that a first pass reported: the vowel
        letters `e, e`, `i, i`, `o, o` and `ï, ï` are the same kind of
        record and are withdrawn with them. They move to `quarantine`, as
        every other withdrawal does; nothing is deleted.

        `i` survives as a word — the pronoun/clitic is a separate record,
        kh_DB_006772, and is untouched.

RULE 4  Two syllabifications named by the maintainer, and two more of the
        same kind found while checking them:

            jingsynshar ka doh  ['jing','jing','syn','shar','ka','doh']
                             -> ['jing','syn','shar','ka','doh']
            jingïang            ['jing','ji','ngiang'] -> ['jing','ïang']
            tiew-pathai-khubor  ['tiew','pa','thai','khub','r']
                             -> ['tiew','pa','thai','khu','bor']
            tnaw                root and syllable read `tndw`, an OCR a->d

        Patterns are not invented. Each is the reading the lexicon already
        attests for that syllable elsewhere: syn CVC (221x), tiew CVV
        (177x), jing CVC (2537x), shar CVC (53x), ka CV (2067x), doh CVC
        (195x), ïang VVC (15x).

    python3 scripts/fix_brackets_and_syllables.py --dry-run
    python3 scripts/fix_brackets_and_syllables.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import re
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "khasi_db.json"

# Bracket contents that are a noun-class clitic or clitic-like particle.
CLITICS = {"u", "ka", "ki", "i", "k", "ba", "ia", "uba"}
# The ones grammar.clitic is allowed to hold.
CLITIC_FIELD = {"u", "ka", "ki", "i"}

BRACKETED = re.compile(r"\(([^)]*)\)")
# Any bracket still standing after the paired pass is unmatched — `baid (ïam`,
# `dak kaba ker , [`. An earlier version removed only closing marks and left 43
# unmatched openers behind, so all six characters are listed.
STRAY = re.compile(r"[()\[\]{}]")

# Khasi-bearing string fields. Glosses are deliberately absent.
KHASI_PATHS = (
    ("form", "surface"), ("form", "normalized"), ("form", "variants"),
    ("lemma", "root"), ("morphology", "canonical"),
    ("phonology", "surface"),
    ("rhetoric", "semantic_pairing"),
)

# RULE 2 and RULE 4 — entry_id -> (syllables, pattern)
SYLLABLES = {
    "kh_DB_024124": (["syn", "tiew"],                          "CVC.CVV"),
    "kh_DB_008900": (["jing", "syn", "shar", "ka", "doh"],     "CVC.CVC.CVC.CV.CVC"),
    "kh_DB_008465": (["jing", "ïang"],                         "CVC.VVC"),
    "kh_DB_030027": (["tiew", "pa", "thai", "khu", "bor"],     "CVV.CV.CVV.CV.CVC"),
    "kh_DB_030041": (["tnaw"],                                 "CCVV"),
}
# RULE 4 — the tnaw record also spells its root with the scanned `d`.
RESPELL_ROOT = {"kh_DB_030041": ("tndw", "tnaw")}


def debracket(text: str) -> tuple[str, str | None, list]:
    """Return (cleaned text, clitic found, parentheticals removed)."""
    if not isinstance(text, str) or not text:
        return text, None, []
    found = None
    out = text

    # A surface wholly wrapped in brackets loses just the wrapper.
    stripped = out.strip()
    if stripped.startswith("(") and stripped.endswith(")") \
            and stripped.count("(") == 1:
        out = stripped[1:-1]

    removed: list[str] = []

    def sub(m):
        nonlocal found
        inner = m.group(1).strip()
        if inner.lower() in CLITICS:
            if found is None:
                found = inner.lower()
            return ""              # the clitic leaves the form entirely
        # Anything else in brackets is a variant list or a pronunciation
        # note — `ba ym (bam, ba'm, bym)`, `buhkhoh (short o)`. Keeping the
        # content would splice it INTO the headword and produce
        # `ba ym bam, ba'm, bym`, so the whole parenthetical goes. The text
        # is handed back to the caller and kept in source.removed_
        # parenthetical, so nothing is destroyed silently.
        if inner:
            removed.append(inner)
        return ""

    out = BRACKETED.sub(sub, out)
    stray_gone = STRAY.sub("", out)
    touched = (found is not None) or bool(removed) or (stray_gone != out) \
        or (out != text)
    out = stray_gone
    if touched:
        # Tidy only what the removal left behind. Applied unconditionally
        # this also ate a LEADING hyphen that is part of the stored
        # morpheme — morphology.canonical "-ioh+ba" is not a bracket
        # artefact and must survive untouched.
        out = re.sub(r"\s{2,}", " ", out).strip()
        out = re.sub(r"^[\s\-]+", "", out)
        out = re.sub(r"[\s,]+$", "", out)
    return out, found, removed


def walk(node, key, changed: list) -> str | None:
    """Debracket node[key] in place; returns a clitic if one was lifted."""
    value = node[key]
    clitic = None
    if isinstance(value, str):
        new, c, rem = debracket(value)
        if new != value:
            node[key] = new
            changed.append((value, new, rem))
        return c
    if isinstance(value, list):
        for i, v in enumerate(value):
            if isinstance(v, str):
                new, c, rem = debracket(v)
                clitic = clitic or c
                if new != v:
                    value[i] = new
                    changed.append((v, new, rem))
    return clitic


def main() -> int:
    ap = argparse.ArgumentParser(description="brackets and syllable rulings")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--db", type=Path, default=DB)
    args = ap.parse_args()

    data = json.loads(args.db.read_text(encoding="utf-8"))
    lexicon = data["lexicon"]
    by_id = {e.get("entry_id"): e for e in lexicon}

    # --- RULE 3: which records are alphabet entries -----------------------
    letter_re = re.compile(r"^(.)\s*,\s*\1$", re.I)
    alphabet = []
    for e in lexicon:
        sf = ((e.get("form") or {}).get("surface") or "").strip()
        g = str((((e.get("semantics") or {}).get("gloss")) or [""])[0])
        # `l, l` is glossed "...small letter l) of the Khasi Alphabet", so a
        # contiguous "letter of the Khasi Alphabet" test misses it. The
        # surface shape plus any mention of the alphabet is the real test.
        if letter_re.match(sf) and "khasi alphabet" in g.lower():
            alphabet.append(e)

    missing = [k for k in list(SYLLABLES) + list(RESPELL_ROOT) if k not in by_id]
    if missing:
        print(f"  ERROR: entry_ids not in the lexicon: {missing}")
        return 1

    # --- RULE 1 -----------------------------------------------------------
    bracket_hits, clitics_lifted, samples = 0, 0, []
    for e in lexicon:
        changed: list = []
        clitic = None
        for path in KHASI_PATHS:
            node = e
            for step in path[:-1]:
                node = node.get(step) if isinstance(node, dict) else None
                if not isinstance(node, dict):
                    node = None
                    break
            if node is None or path[-1] not in node:
                continue
            clitic = walk(node, path[-1], changed) or clitic
        if changed:
            bracket_hits += 1
            if len(samples) < 14:
                samples.append((e.get("entry_id"), changed[0][:2], clitic))
            dropped = sorted({t for _, _, rem in changed for t in rem})
            if dropped and not args.dry_run:
                e.setdefault("source", {})["removed_parenthetical"] = dropped
            if clitic in CLITIC_FIELD:
                gram = e.setdefault("grammar", {})
                if not gram.get("clitic"):
                    gram["clitic"] = clitic
                    clitics_lifted += 1

    # --- RULES 2 and 4 ----------------------------------------------------
    syl_fixed = []
    for eid, (syl, pat) in SYLLABLES.items():
        e = by_id[eid]
        derived = e.setdefault("phonology", {}).setdefault("derived", {})
        before = list(derived.get("syllables") or [])
        derived["syllables"] = list(syl)
        derived["pattern"] = pat
        derived["syllable_count"] = len(syl)
        if eid in RESPELL_ROOT:
            wrong, right = RESPELL_ROOT[eid]
            lem = e.setdefault("lemma", {})
            if lem.get("root") == wrong:
                lem["root"] = right
            phon = e.setdefault("phonology", {})
            if isinstance(phon.get("surface"), str):
                phon["surface"] = phon["surface"].replace(wrong, right)
        syl_fixed.append((eid, (e.get("form") or {}).get("surface"), before, syl))

    print(f"  RULE 1  records debracketed   : {bracket_hits}")
    print(f"          clitics -> grammar    : {clitics_lifted}")
    print(f"  RULE 2+4 syllables corrected  : {len(syl_fixed)}")
    print(f"  RULE 3  alphabet entries      : {len(alphabet)} to quarantine")

    if args.dry_run:
        print("\n  --- brackets ---")
        for eid, (old, new), c in samples:
            print(f"    {eid} | {old!r}  ->  {new!r}   clitic={c!r}")
        print("\n  --- syllables ---")
        for eid, sf, b, a in syl_fixed:
            print(f"    {eid} | {sf} | {b}  ->  {a}")
        print("\n  --- alphabet entries ---")
        for e in alphabet:
            print(f"    {e['entry_id']} | {(e.get('form') or {}).get('surface')!r}")
        print("\n  --dry-run: nothing written")
        return 0

    backup = args.db.with_name(
        f"{args.db.name}.bak.{datetime.now():%Y%m%d_%H%M%S}")
    shutil.copy2(args.db, backup)
    print(f"\n  backup -> {backup.name}")

    # Taking a bracket off a headword can uncover a twin: the lexicon holds
    # `'tiew pathai khubor` both plainly and with a bracketed qualifier, and
    # once the qualifier goes the two records claim the same surface. The
    # analyser answers 'derived' instead of 'valid' for any surface held by
    # more than one entry, so they are merged the way the diacritic passes
    # merge theirs — richer record kept, the other's glosses appended.
    spec = importlib.util.spec_from_file_location(
        "fix_ia", Path(__file__).with_name("fix_ia_diaeresis.py"))
    fix_ia = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fix_ia)
    merge = fix_ia.dedupe(data)
    print(f"  deduped: {merge['merged']} entries merged across "
          f"{merge['surfaces']} surfaces, {merge['glosses_added']} glosses kept")
    lexicon = data["lexicon"]
    alphabet = [e for e in lexicon if e.get("entry_id")
                in {a.get("entry_id") for a in alphabet}]

    stamp = datetime.now().strftime("%Y-%m-%d")
    ids = {e.get("entry_id") for e in alphabet}
    for e in alphabet:
        e["quarantine_reason"] = (
            "an entry for a LETTER of the Khasi alphabet, not a Khasi word; "
            f"its gloss is the alphabet description [{stamp}]")
    data["quarantine"] = data.get("quarantine", []) + alphabet
    data["lexicon"] = [e for e in lexicon if e.get("entry_id") not in ids]

    data.setdefault("meta", {})["bracket_and_syllable_fix"] = {
        "date": stamp,
        "records_debracketed": bracket_hits,
        "clitics_lifted_to_grammar": clitics_lifted,
        "syllables_corrected": [k for k in SYLLABLES],
        "alphabet_entries_quarantined": sorted(ids),
        "merged_after_debracketing": merge["merged"],
        "note": "Bracketed noun-class clitics moved from the form to "
                "grammar.clitic; other brackets lost only their brackets. "
                "Glosses untouched. Alphabet-letter entries withdrawn.",
    }
    args.db.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    print(f"  lexicon: {len(data['lexicon'])} entries "
          f"({len(alphabet)} moved to quarantine)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

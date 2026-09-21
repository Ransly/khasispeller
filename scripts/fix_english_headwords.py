#!/usr/bin/env python3
"""
fix_english_headwords.py — take the English out of Khasi headwords.

The 1906 dictionary is printed in two columns, Khasi and English. In a
number of records the column boundary was misread and the English ran into
the Khasi headword. The motivating case is the one deferred by
`fix_ain_coda_tilde.py`, which skipped it by entry_id rather than repair it:

    balei phi ïaid jaw-jaw ha u slap in the rain

The Khasi ends at `slap` (`ha u slap` already *means* "in the rain"); the
tail is the source's own translation. Because `KhasiDB.all_surface_forms()`
harvests every token of a multi-word entry into the candidate pool, `in`,
`the` and `rain` became offerable Khasi corrections — and `rain` outranked
the correctly spelled `raiñ` for the input `raiin`.

Two repairs, and a rule about which is which
--------------------------------------------
  TRIM        the record has real Khasi followed by English or by the
              dictionary's own notation. Cut the tail; keep the Khasi; put
              the English into `semantics.gloss`, where it belonged.
  QUARANTINE  the surface is English end to end. There is no Khasi to keep,
              and the attached glosses are visibly mismatched OCR debris
              ('human eyes' glossed "Having a very bad smell"). These move
              to the `quarantine` list with a reason, as
              `quarantine_confirmed_damage.py` does. Nothing is deleted.

Why both tables are written out by hand
---------------------------------------
Three automatic rules were measured against the data and all three are
wrong, so none of them decides anything here:

  * "the trailing token is an English word" flags 90 records, and most are
    real: `kam u mayor` (Mayoralty), `nongdie stamp` (Stamp-vendor),
    `jaiñ linen`, `iew shillong`, `ar pints` — Khasi phrases built on an
    English loanword, which is how the dictionary renders technical terms.
    Trimming those destroys genuine entries.
  * "the token recurs in few entries" separates the extremes (`shnong` 50,
    `mynsiem` 76, against `jury` 1) but 1,717 of 2,195 genuine Khasi
    phrase-tokens also occur exactly once, so it would discard most of what
    the harvesting exists to reach.
  * "the token fails Khasi phonotactics" rejects only 21 of the 118 English
    words in the pool. Khasi is permissive; `jury`, `mayor` and `linen` are
    all legal shapes.

What DOES separate them is the presence of English function words
(`in the rain`, `used in washing the head`) or of the dictionary's POS
notation (`a. trilateral`, `ka,n. bewail`) — a Khasi phrase carries
neither. That test produced 25 records; each was then read individually,
and the outcome of that reading is the two tables below. The tables are the
audit trail: no record changes unless it is named here.

    python3 scripts/fix_english_headwords.py --dry-run
    python3 scripts/fix_english_headwords.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "khasi_db.json"

# entry_id -> (kept Khasi surface, English lifted out and added to the gloss)
# "" as the second element means the tail was notation only, with no English
# word to preserve.
TRIM: dict[str, tuple[str, str]] = {
    # --- the dictionary's POS notation ran into the headword --------------
    "kh_DB_004200": ("jingot ïa ka dkhot met",            "mutilation"),
    "kh_DB_005499": ("ba don lai liang",                  "trilateral"),
    "kh_DB_005503": ("ba don lai pnat",                   "tridentate"),
    "kh_DB_005507": ("ba don lai tylli ka kyrteng",       "trinominal"),
    "kh_DB_005508": ("ba don lai tylli ki liang",         "trilateral"),
    "kh_DB_005555": ("dur ka apod sepsngil",              "alas-a-day"),
    "kh_DB_008500": ("jingbuh-nud",                       "bewail"),
    "kh_DB_014147": ("ksieh",                             "otter"),
    "kh_DB_017107": ("leh sting",                         ""),
    "kh_DB_023472": ("snieh mrad",                        "hide, leather, skin"),
    "kh_DB_023571": ("soh jaw",                           "drop"),
    # `ha ka khyrwait a. ha ka khyrwait` — the phrase twice around an `a.`
    "kh_DB_012590": ("ha ka khyrwait",                    ""),
    # --- an English translation ran into the headword --------------------
    "kh_DB_008212": ("balei phi ïaid jaw-jaw ha u slap",  "in the rain"),
    "kh_DB_023745": ("ki lum ki la spong da ka kashem kiw", "with snow"),
}

# entry_id -> surface, for records that are English end to end.
QUARANTINE: dict[str, str] = {
    "kh_DB_024703": "with annas",
    "kh_DB_024758": "the toes",
    "kh_DB_025541": "with shi-",
    "kh_DB_025590": "the whole earth",
    "kh_DB_026064": "used in washing the head",
    "kh_DB_026656": "in the lowlands",
    "kh_DB_026749": "pyre with the dead body",
    "kh_DB_028167": "brilliant like the sun",
    "kh_DB_028182": "thing when asked",
    "kh_DB_028491": "uneral pyre towards the north",
    # Found on a second pass: these carry no English function word and no
    # POS notation, so the detector that produced the list above missed
    # them. Every token is English, and each gloss belongs to some other
    # headword entirely ('human eyes' glossed "Having a very bad smell").
    "kh_DB_025110": "some impending danger",
    "kh_DB_026659": "human eyes",
    "kh_DB_028281": "same thing",
    "kh_DB_028599": "wide open",
}

# Read and left alone, recorded so the next pass does not re-litigate it:
#   kh_DB_013421 `kot a. b.` glossed "Khasi primer". `kot` is "book" and
#   "a. b." reads as the ABC the primer teaches, not as POS notation. It is
#   not clear which, and a wrong trim invents a headword, so it stays.
LEFT_ALONE = {"kh_DB_013421": "kot a. b."}


def surface(e: dict) -> str:
    return ((e.get("form") or {}).get("surface") or "")


def _retrim(value, full: str, keep: str):
    """Delete the dropped tail, leaving the rest of the string untouched.

    Cutting at a prefix and substituting `keep` would be wrong: variants
    carry their own spelling (`... iaid ...` beside `... ïaid ...`) and
    substituting the headword's spelling would flatten that distinction and
    leave a variant identical to the surface. So the tail alone is removed,
    matched both as written and ASCII-folded, and every other character of
    the value survives — including a trailing `?`.
    """
    tail = full[len(keep):]
    if not tail:
        return value
    if isinstance(value, str):
        for bad in (tail, tail.replace("ï", "i")):
            if bad and bad in value:
                return value.replace(bad, "", 1).strip()
        return value
    if isinstance(value, list):
        out, seen = [], set()
        for x in value:
            r = _retrim(x, full, keep)
            # trimming can make two variants collide, and can make a variant
            # equal the surface; neither is worth keeping.
            k = r.strip().lower() if isinstance(r, str) else None
            if k is not None:
                if k in seen or k == keep.strip().lower():
                    continue
                seen.add(k)
            out.append(r)
        return out
    return value


def apply_trims(lexicon: list[dict], report: list[str], todo: set) -> int:
    changed = 0
    for e in lexicon:
        eid = e.get("entry_id")
        if eid not in todo:
            continue
        keep, english = TRIM[eid]
        full = surface(e)
        if not full:
            continue
        form = e.setdefault("form", {})
        form["surface"] = keep
        if "normalized" in form:
            form["normalized"] = _retrim(form["normalized"], full, keep)
        if "variants" in form:
            form["variants"] = _retrim(form["variants"], full, keep)

        phon = e.get("phonology")
        if isinstance(phon, dict):
            if "surface" in phon:
                phon["surface"] = _retrim(phon["surface"], full, keep)
            # The IPA is generated, never string-edited (see
            # fix_ain_coda_tilde.py). Regenerate it from the repaired
            # surface so it does not keep transcribing the English.
            ipa = phon.get("ipa")
            if isinstance(ipa, dict):
                try:
                    from khasi_engine import g2p
                    # to_ipa already returns its own slashes, so they are
                    # stripped before the phrase is wrapped in one pair.
                    words = [g2p.to_ipa(t).strip("/") for t in keep.split()]
                    ipa["surface"] = "/" + " ".join(w for w in words if w) + "/"
                    ipa["generated_by"] = g2p.VERSION
                except Exception:
                    ipa["needs_review"] = True

        rel = e.get("lexical_relations")
        if isinstance(rel, dict) and "variants" in rel:
            rel["variants"] = _retrim(rel["variants"], full, keep)

        # The English belongs in the gloss. Skip it when the gloss already
        # carries it — `bewail` is the first sense of kh_DB_008500 — so the
        # repair never duplicates a sense.
        if english:
            sem = e.setdefault("semantics", {})
            gloss = sem.setdefault("gloss", [])
            if isinstance(gloss, list):
                have = {str(g).strip().lower() for g in gloss}
                if english.lower() not in have:
                    gloss.append(english)
                    sem["gloss_normalized"] = "; ".join(str(g) for g in gloss)
        report.append(f"    {full!r}\n      -> {keep!r}"
                      + (f"   [gloss += {english!r}]" if english else ""))
        changed += 1
    return changed


def main() -> int:
    ap = argparse.ArgumentParser(description="remove English from Khasi headwords")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--db", type=Path, default=DB)
    args = ap.parse_args()

    data = json.loads(args.db.read_text(encoding="utf-8"))
    lexicon = data["lexicon"]

    # Guard: every entry_id named in the tables must exist and must still
    # carry the surface the table was written against. If a previous pass
    # changed one, stop rather than repair something that has moved.
    # The pass is idempotent: an entry already repaired by an earlier run is
    # reported as done rather than treated as an error, so re-running after
    # the table grows only acts on what is left.
    by_id = {e.get("entry_id"): e for e in lexicon}
    done_q = {e.get("entry_id") for e in data.get("quarantine", [])}
    todo_q = {k: v for k, v in QUARANTINE.items() if k not in done_q}
    todo_t = {k: v for k, v in TRIM.items()
              if k in by_id and surface(by_id[k]).strip() != v[0]}
    already = (len(QUARANTINE) - len(todo_q)) + (len(TRIM) - len(todo_t))

    missing = [k for k in list(todo_t) + list(todo_q) if k not in by_id]
    if missing:
        print(f"  ERROR: entry_ids in neither the lexicon nor quarantine: {missing}")
        return 1
    for eid, want in todo_q.items():
        if surface(by_id[eid]).strip() != want:
            print(f"  ERROR: {eid} surface is {surface(by_id[eid])!r}, "
                  f"table expects {want!r}")
            return 1
    for eid, (keep, _) in todo_t.items():
        if not surface(by_id[eid]).startswith(keep):
            print(f"  ERROR: {eid} surface is {surface(by_id[eid])!r}, "
                  f"which does not start with {keep!r}")
            return 1

    report: list[str] = []
    print(f"  TRIM       : {len(todo_t)} to do ({len(TRIM)} in table)")
    print(f"  QUARANTINE : {len(todo_q)} to do ({len(QUARANTINE)} in table)")
    if already:
        print(f"  already repaired by an earlier run: {already}")
    if not todo_t and not todo_q:
        print("\n  nothing to do")
        return 0
    print(f"  left alone : {len(LEFT_ALONE)} ({', '.join(LEFT_ALONE.values())})")

    if args.dry_run:
        print("\n  --- trims ---")
        for eid, (keep, eng) in todo_t.items():
            print(f"    {eid}\n      {surface(by_id[eid])!r}\n      -> {keep!r}"
                  + (f"   [gloss += {eng!r}]" if eng else ""))
        print("\n  --- quarantine ---")
        for eid, s in todo_q.items():
            print(f"    {eid} | {s!r}")
        print("\n  --dry-run: nothing written")
        return 0

    backup = args.db.with_name(
        f"{args.db.name}.bak.{datetime.now():%Y%m%d_%H%M%S}")
    shutil.copy2(args.db, backup)
    print(f"\n  backup -> {backup.name}")

    trimmed = apply_trims(lexicon, report, set(todo_t))
    print("\n  --- trims applied ---")
    for line in report:
        print(line)

    # Trimming can uncover a twin. The dictionary recorded some words twice,
    # once cleanly and once with its own notation attached — `jingbuh-nud`
    # beside `jingbuh-nud, ka,n. bewail` — so cutting the notation makes the
    # two records claim the same surface. That is not cosmetic: the analyser
    # returns 'derived' instead of 'valid' for any surface held by more than
    # one entry. Merge them the way the diacritic passes do, keeping the
    # richer record and appending the other's glosses.
    spec = importlib.util.spec_from_file_location(
        "fix_ia", Path(__file__).with_name("fix_ia_diaeresis.py"))
    fix_ia = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fix_ia)
    merge = fix_ia.dedupe(data)
    print(f"\n  deduped: {merge['merged']} entries merged across "
          f"{merge['surfaces']} surfaces, {merge['glosses_added']} glosses kept")
    lexicon = data["lexicon"]

    stamp = datetime.now().strftime("%Y-%m-%d")
    move = [e for e in lexicon if e.get("entry_id") in todo_q]
    for e in move:
        e["quarantine_reason"] = (
            "the surface is English end to end, captured from the source "
            "dictionary's English column; there is no Khasi to keep and the "
            f"attached gloss does not match it [{stamp}]")
    data["quarantine"] = data.get("quarantine", []) + move
    data["lexicon"] = [e for e in lexicon if e.get("entry_id") not in todo_q]

    data.setdefault("meta", {})["english_headword_fix"] = {
        "date": stamp,
        "trimmed": trimmed,
        "merged_after_trim": merge["merged"],
        "quarantined": len(move),
        "left_alone": sorted(LEFT_ALONE),
        "note": "English text captured into Khasi headwords from the source "
                "dictionary's second column; trimmed surfaces keep the Khasi "
                "and gain the English as a gloss sense.",
    }
    args.db.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    print(f"\n  lexicon: {len(data['lexicon'])} entries "
          f"({len(move)} moved to quarantine)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

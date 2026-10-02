#!/usr/bin/env python3
"""
quarantine_nonwords.py — withdraw headwords the maintainer rules are not
Khasi words.

A table, one row per ruling, so each withdrawal is attributable. Entries
move to the `quarantine` list with a reason, as every other withdrawal in
this project does; nothing is deleted.

Why this is a table and not a rule
----------------------------------
`briw` is a scan artefact of `briew`: it comes from the 1906 dictionary PDF
import, its gloss duplicates `briew`'s meanings, and that gloss is visibly
mangled — "Mnn; Woman; Person; («-) the apple of the eye- •pi. ki 6W^to=
people", where `Mnn` is Man and `•pi. ki 6W^to=` is `pl. ki briew =`.

Searching for that signature — a PDF-import headword one edit from a
headword sourced elsewhere, with OCR debris in its gloss — returns 475
candidates, and they are NOT all damage. `duar` (door), `khamti`, `kyrtong`
(a bull, an ox) read as real words, and `dathew` beside `da-thew` is the
spaced/hyphenated twin pattern this project has ruled is a genuine pair
rather than a duplicate. Real damage is mixed in with them — `kblak-khlak`
for `khlak-khlak`, `eliaw` for `khliaw`, `khlleh` for `khleh` — so the pool
is a review queue, not something to withdraw in bulk.

    python3 scripts/quarantine_nonwords.py --dry-run
    python3 scripts/quarantine_nonwords.py
"""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "khasi_db.json"

# entry_id -> (surface the table was written against, reason)
NOT_WORDS = {
    "kh_DB_024844": (
        "briw",
        "not a Khasi word (maintainer ruling 2026-09-10): a scan artefact of "
        "`briew` from the 1906 dictionary PDF import. Its gloss repeats "
        "briew's meanings and is itself mangled — \"Mnn; Woman; Person; "
        "(«-) the apple of the eye- •pi. ki 6W^to= people\", in which "
        "`Mnn` is Man and `•pi. ki 6W^to=` is `pl. ki briew =`"),

    # Scan damage on the FIRST letter, each confirmed by the maintainer
    # 2026-09-11. In every case the correct spelling is already its own
    # lexicon entry carrying the same gloss, so these are duplicate records
    # with a corrupted headword rather than words needing a rename. Every
    # damaged onset is one the phonology block forbids; every correct one is
    # permitted.
    "kh_DB_025686": (
        "bpei",
        "not a Khasi word (maintainer ruling 2026-09-11): scan damage for "
        "dpei (kh_DB_005285, 'Ash; Ashes; Cinder; Hearth'), which the "
        "lexicon already records. 'bp' is not a permitted Khasi onset; "
        "'dp' is"),
    "kh_DB_027273": (
        "bmiang",
        "not a Khasi word (maintainer ruling 2026-09-11): scan damage for "
        "rmiang (kh_DB_029788), which carries the IDENTICAL gloss 'The "
        "margin; The rim; Border (cloth)'. 'bm' is not a permitted Khasi "
        "onset; 'rm' is"),
    "kh_DB_027334": (
        "bwieng",
        "not a Khasi word (extends the same ruling): scan damage for "
        "rwieng (kh_DB_029825, 'The intestines of a bird') — the same b/r "
        "misreading as bmiang -> rmiang, and the same gloss. 'bw' is not a "
        "permitted Khasi onset; 'rw' is"),

    # Second batch, maintainer rulings 2026-09-11. Same shape as the first:
    # a forbidden onset, and the correct spelling already an entry with the
    # matching gloss.
    "kh_DB_026521": (
        "mdwpyrsut",
        "not a Khasi word (maintainer ruling 2026-09-11): scan damage for "
        "mawpyrsut (kh_DB_029426), same gloss 'Iron-ore'. maw- is the stone "
        "prefix; 'mdw' is not a permitted onset"),
    "kh_DB_026526": (
        "mdwshun",
        "not a Khasi word (maintainer ruling 2026-09-11): scan damage for "
        "mawshun (kh_DB_029427), same gloss 'Limestone'. Same mdw -> maw "
        "misreading as mdwpyrsut"),
    "kh_DB_028605": (
        "wdi-dong",
        "not a Khasi word (maintainer ruling 2026-09-11): scan damage for "
        "wai-dong (kh_DB_030152), same gloss 'Rolled or prepared "
        "betel-nut'. 'wdi' is not a permitted onset; 'wai' is"),
    "kh_DB_026909": (
        "pbngaiñ",
        "not a Khasi word (maintainer ruling 2026-09-11, corrected): scan "
        "damage for ngaiñ (kh_DB_025729), NOT for phngaiñ. The damage is a "
        "spurious leading 'pb', not the pb -> ph substitution seen in "
        "pbnguid -> phnguid; phngaiñ means 'Clear; Not misty', the opposite "
        "of this record's 'Misty', which is what gave the wrong reading "
        "away"),
    "kh_DB_026912": (
        "pbnguid",
        "not a Khasi word (extends the pb -> ph ruling): scan damage for "
        "phnguid (kh_DB_029659, 'To eat gluttonously'), this record reading "
        "'Gluttonously'. Same forbidden 'pb' onset as pbngaiñ"),
    "kh_DB_025284": (
        "jklat",
        "not a Khasi word (maintainer ruling 2026-09-11): scan damage for "
        "klat (kh_DB_012918), same gloss 'Glass; Tumbler; V. turn abruptly "
        "to another direction'. A spurious leading j. The OTHER jk- forms "
        "are deliberately NOT withdrawn — the maintainer is unsure whether "
        "jk is a real onset missing from valid_initial_clusters, and jkap, "
        "jkeng, jking, jkup and the rest carry distinct sensible glosses"),

    # Scan damage at the END of the headword, maintainer rulings 2026-10-01.
    # The 1906 dictionary prints "Balang, ka, n."; the comma after the
    # headword was read as j and the clitic came into the surface with it.
    # Again the correct spelling is already its own entry with the same
    # gloss. They surfaced when final -j stopped being a loan signal.
    "kh_DB_024743": (
        "balangj ka",
        "not a Khasi word (maintainer ruling 2026-10-01): scan damage for "
        "balang (kh_DB_001061), same gloss 'the church; a society "
        "(religious)'. The comma in the 1906 dictionary's 'Balang, ka, n.' "
        "was read as j"),
    "kh_DB_025671": (
        "dohtdongj ka",
        "not a Khasi word (maintainer ruling 2026-10-01): scan damage for "
        "dohtdong (kh_DB_028830), same gloss 'the hump'. The comma after "
        "the headword was read as j, as in balangj"),
    "kh_DB_025583": (
        "diengddwj ka",
        "not a Khasi word (maintainer ruling 2026-10-01): scan damage for "
        "diengdaw (kh_DB_028808), same gloss 'a hard dried wood'. The record "
        "also swallowed the next dictionary entry, diengdoh, which the "
        "lexicon holds on its own (kh_DB_028809)"),
    "kh_DB_027185": (
        "eaj",
        "not a Khasi word (maintainer ruling 2026-10-01): scan damage for "
        "raj (kh_DB_021400), same gloss 'jurisdiction'. The 1906 "
        "dictionary's 'Raj' was read as 'Eaj'"),
}


def main() -> int:
    ap = argparse.ArgumentParser(description="withdraw non-words")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--db", type=Path, default=DB)
    args = ap.parse_args()

    data = json.loads(args.db.read_text(encoding="utf-8"))
    lexicon = data["lexicon"]
    by_id = {e.get("entry_id"): e for e in lexicon}
    done = {e.get("entry_id") for e in data.get("quarantine", [])}

    todo, already, wrong = {}, [], []
    for eid, (surface, reason) in NOT_WORDS.items():
        if eid in done:
            already.append(eid)
            continue
        entry = by_id.get(eid)
        if entry is None:
            wrong.append(f"{eid} is in neither the lexicon nor quarantine")
            continue
        actual = ((entry.get("form") or {}).get("surface") or "").strip().lower()
        if actual != surface:
            wrong.append(f"{eid} is {actual!r}, the table expects {surface!r}")
            continue
        todo[eid] = reason

    if wrong:
        for w in wrong:
            print(f"  ERROR: {w}")
        return 1

    print(f"  to withdraw : {len(todo)}")
    if already:
        print(f"  already done: {len(already)}")
    for eid, reason in todo.items():
        print(f"    {eid} | {NOT_WORDS[eid][0]!r}\n        {reason[:96]}…")

    if args.dry_run:
        print("\n  --dry-run: nothing written")
        return 0
    if not todo:
        print("  nothing to do")
        return 0

    backup = args.db.with_name(
        f"{args.db.name}.bak.{datetime.now():%Y%m%d_%H%M%S}")
    shutil.copy2(args.db, backup)
    print(f"\n  backup -> {backup.name}")

    stamp = datetime.now().strftime("%Y-%m-%d")
    move = [e for e in lexicon if e.get("entry_id") in todo]
    for e in move:
        e["quarantine_reason"] = f"{todo[e['entry_id']]} [{stamp}]"
    data["quarantine"] = data.get("quarantine", []) + move
    data["lexicon"] = [e for e in lexicon if e.get("entry_id") not in todo]
    data.setdefault("meta", {})["nonword_quarantine"] = {
        "date": stamp,
        "moved": len(move),
        "entry_ids": sorted(todo),
    }
    args.db.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    print(f"  lexicon: {len(data['lexicon'])} entries ({len(move)} withdrawn)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

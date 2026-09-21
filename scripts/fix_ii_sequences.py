#!/usr/bin/env python3
"""
fix_ii_sequences.py — Khasi has no `ii`. Repair the two things it hides.

Maintainer ruling (2026-09-08): the valid sequence is `ïi`; plain `ii` and
`iï` are not Khasi. Auditing every `ii` surface showed it is not one error
but two, so this pass applies two rules in a fixed order.

RULE A — `iing` -> `ïing`  (142 surfaces)
    The word for 'house'. Doublet-evidenced the same way the earlier
    diaeresis passes were: both spellings are recorded, so it is an error
    and not a variant —
        iing / ïing        iing rit / ïing rit
        iing sder / ïing sder      iing-kirja / ïing-kirja
    `database.py`'s folded index already treats `ïing` as canonical
    (`iing -> ïing` is one of its worked examples).

    Runs FIRST, because collapsing runs of `i` first would turn `iing`
    into `ing` and destroy the word.

RULE B — a run of two or more `i` is scan damage  (214 surfaces)
    The run collapses to a single `i`; when the run ends a token, its last
    `i` is a misread `ñ`, so the run becomes `iñ`. Maintainer's worked
    examples, all reproduced by this rule:

        biiid       -> bid           run mid-word, collapses
        biin        -> bin           run mid-word, collapses
        biir        -> bir           run mid-word, collapses
        behbeiii    -> behbeiñ       run ends the token -> iñ
        daiii-daiñ  -> daiñ-daiñ     run ends the token -> iñ

    Corroboration: 18 of these repairs land on a word the lexicon already
    holds — biin/bin, biir/bir, khiim/khim, kyntiip/kyntip, khliir/khlir,
    kynjiih/kynjih, jaraiiñ/jaraiñ — and the `iit` family (38 surfaces)
    repairs onto the attested `it`.

    ONE EXCEPTION, where attestation beats the rule:
        nongkhaii -> nongkhaïi, NOT nongkhaiñ
    `nongkhaïi` is a recorded headword, and every pass in this family has
    let an attested spelling win over an inferred one.

Safety
------
Only the Khasi fields listed in KHASI_PATHS are rewritten. `ii` occurs in
English (skiing, radii, shiitake) and, more to the point, provenance text
is full of ordinary prose; the `ain` pass corrupted 432 `remain` before it
was scoped this way. `quarantine` is left alone — it is the audit trail of
what was withdrawn and must keep the spelling it was withdrawn with.

    python3 scripts/fix_ii_sequences.py --dry-run
    python3 scripts/fix_ii_sequences.py
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

RUN = re.compile(r"i{2,}")
ATTESTED_OVERRIDE = {"nongkhaii": "nongkhaïi"}

KHASI_PATHS = (
    ("form", "surface"),
    ("form", "normalized"),
    ("form", "variants"),
    ("lemma", "root"),
    ("lemma", "root_id"),
    ("morphology", "canonical"),
    ("morphology", "structure"),
    ("phonology", "surface"),
    ("phonology", "derived", "syllables"),
    ("lexical_relations", "related"),
    ("lexical_relations", "better_forms"),
)

ENGLISH_PROBES = ("skiing", "radii", "shiitake", "remaiñ", "contaiñ",
                  "certaiñ", "agaiñ", "Hawaii", "hawaii")


def _collapse(word: str) -> str:
    out, i = [], 0
    for m in RUN.finditer(word):
        out.append(word[i:m.start()])
        end = m.end()
        out.append("iñ" if (end == len(word) or word[end] in " -") else "i")
        i = end
    out.append(word[i:])
    return "".join(out)


def fix(text: str) -> str:
    if text in ATTESTED_OVERRIDE:                 # exact-surface override
        return ATTESTED_OVERRIDE[text]
    text = text.replace("iing", "ïing")           # RULE A — must be first
    return _collapse(text)                        # RULE B


def _fix_in(node, key) -> int:
    value = node[key]
    if isinstance(value, str):
        new = fix(value)
        if new != value:
            node[key] = new
            return 1
        return 0
    changed = 0
    if isinstance(value, dict):
        for k in list(value):
            changed += _fix_in(value, k)
    elif isinstance(value, list):
        for i in range(len(value)):
            changed += _fix_in(value, i)
    return changed


def walk(entries) -> int:
    changed = 0
    for entry in entries:
        if not isinstance(entry, dict):
            continue
        for path in KHASI_PATHS:
            node = entry
            for step in path[:-1]:
                node = node.get(step) if isinstance(node, dict) else None
                if not isinstance(node, dict):
                    node = None
                    break
            if node is None or path[-1] not in node:
                continue
            changed += _fix_in(node, path[-1])
    return changed


def main() -> int:
    ap = argparse.ArgumentParser(description="iing -> ïing; collapse ii runs")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--db", type=Path, default=DB)
    args = ap.parse_args()

    data = json.loads(args.db.read_text(encoding="utf-8"))
    before_blob = json.dumps(data, ensure_ascii=False)
    before_counts = {w: before_blob.count(w) for w in ENGLISH_PROBES}

    def surface(e):
        return (e.get("form") or {}).get("surface") or ""

    rule_a = [e for e in data["lexicon"] if "iing" in surface(e)]
    rule_b = [e for e in data["lexicon"]
              if "iing" not in surface(e) and RUN.search(surface(e))]
    print(f"  RULE A  iing -> ïing        : {len(rule_a)} surfaces")
    print(f"  RULE B  collapse ii runs    : {len(rule_b)} surfaces")

    if args.dry_run:
        for e in (rule_a[:6] + rule_b[:14]):
            s = surface(e)
            print(f"      {s:28} ->  {fix(s)}")
        print("\n  --dry-run: nothing written")
        return 0

    backup = args.db.with_name(
        f"{args.db.name}.bak.{datetime.now():%Y%m%d_%H%M%S}")
    shutil.copy2(args.db, backup)
    print(f"  backup -> {backup.name}")

    changed = walk(data["lexicon"])

    spec = importlib.util.spec_from_file_location(
        "fix_ia", Path(__file__).with_name("fix_ia_diaeresis.py"))
    fix_ia = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fix_ia)
    merge = fix_ia.dedupe(data)
    print(f"  deduped: {merge['merged']} entries merged across "
          f"{merge['surfaces']} surfaces, {merge['glosses_added']} glosses kept")

    after_blob = json.dumps(data, ensure_ascii=False)
    damaged = {w: (before_counts[w], after_blob.count(w))
               for w in ENGLISH_PROBES if after_blob.count(w) != before_counts[w]}
    if damaged:
        raise SystemExit(
            f"ABORT: English/provenance text changed (before -> after): "
            f"{damaged}. Nothing written; the file on disk is untouched.")

    stamp = datetime.now().strftime("%Y-%m-%d")
    data.setdefault("meta", {})["ii_sequence_fix"] = {
        "date": stamp, "strings_changed": changed,
        "rule_a_iing": len(rule_a), "rule_b_collapse": len(rule_b),
        "attested_overrides": ATTESTED_OVERRIDE, "merged": merge["merged"],
    }
    args.db.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    print(f"  strings changed: {changed}")
    print(f"  lexicon entries: {len(data['lexicon'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

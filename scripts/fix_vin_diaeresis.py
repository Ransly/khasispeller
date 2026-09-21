#!/usr/bin/env python3
"""
fix_vin_diaeresis.py — normalise `eïñ`/`oïñ`/`uïñ` to `eiñ`/`oiñ`/`uiñ`.

The companion to `fix_ain_diaeresis.py`, which fixed exactly this error for
the `a` vowel only (`WRONG = "aïñ"`) and left the other three behind. The
rule is the same one that script states: Khasi writes the sequence as
**vowel + i + ñ** — a plain `i` in the diphthong, with the tilde on the
nasal alone. A diaeresis marks a vowel that opens its own syllable, so it
cannot sit on the second half of a diphthong.

Evidence this is an error and not a variant, gathered the same way the
`aïñ` pass gathered it — 12 surfaces are recorded BOTH ways:

    lut phoïñ / lut phoiñ            her soïñ / her soiñ
    bha rymmuïñ / bha rymmuiñ        saw huïñ / saw huiñ
    jingkheïñ / jingkheiñ            ïa-peïñ / ïa-peiñ

Counts before this pass: uïñ 59, oïñ 52, eïñ 22.

Two sequences need care and are handled by ordering the replacements:

  * `iïñ` (1 surface, `shleiïñ-shleiïñ`) carries a doubled i. Replacing
    `ïñ`->`iñ` blindly would yield `shleiiñ`. It is collapsed to `iñ`
    FIRST, giving `shleiñ-shleiñ`, which is how the lexicon spells the
    same word elsewhere (`shleïñ` -> `shleiñ`).
  * `lïñ` (1 surface, `leh rymmolïñ`) is NOT touched. There the character
    before the diaeresis is a consonant, so this is scan damage — very
    likely `o` misread as `ol`, since `rymmoïñ` exists — and repairing it
    is a reading of the OCR, not an application of this rule.

As in the `aïñ` pass the rewrite runs over every string in each entry, not
just `form.surface`: variants, normalized forms and morphology structures
all carry surfaces, and the collapse is followed by the shared dedupe so
that pairs recorded both ways do not become two entries for one surface.

    python3 scripts/fix_vin_diaeresis.py --dry-run
    python3 scripts/fix_vin_diaeresis.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import shutil
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "khasi_db.json"

# Order matters — see the `iïñ` note in the module docstring.
REPLACEMENTS = (("iïñ", "iñ"), ("eïñ", "eiñ"), ("oïñ", "oiñ"), ("uïñ", "uiñ"))


def fix(text: str) -> str:
    for wrong, right in REPLACEMENTS:
        text = text.replace(wrong, right)
    return text


def walk(node) -> int:
    """Rewrite in place, returning how many strings changed."""
    changed = 0
    if isinstance(node, dict):
        items = node.items()
    elif isinstance(node, list):
        items = enumerate(node)
    else:
        return 0
    for key, value in list(items):
        if isinstance(value, str):
            new = fix(value)
            if new != value:
                node[key] = new
                changed += 1
        else:
            changed += walk(value)
    return changed


def main() -> int:
    ap = argparse.ArgumentParser(description="normalise eïñ/oïñ/uïñ -> eiñ/oiñ/uiñ")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--db", type=Path, default=DB)
    args = ap.parse_args()

    data = json.loads(args.db.read_text(encoding="utf-8"))

    def surfaces(block):
        return [((e.get("form") or {}).get("surface") or "") for e in block]

    targets = [s for s in surfaces(data["lexicon"]) if fix(s) != s]
    entries = [e for e in data["lexicon"]
               if fix(json.dumps(e, ensure_ascii=False)) != json.dumps(e, ensure_ascii=False)]
    print(f"  lexicon entries touched: {len(entries)}")
    print(f"    of which the surface itself: {len(targets)}")
    untouched = [s for s in surfaces(data["lexicon"]) if "ïñ" in fix(s)]
    print(f"    left alone (consonant before ï): {untouched}")

    if args.dry_run:
        for s in targets[:15]:
            print(f"      {s}  ->  {fix(s)}")
        print("\n  --dry-run: nothing written")
        return 0

    backup = args.db.with_name(
        f"{args.db.name}.bak.{datetime.now():%Y%m%d_%H%M%S}")
    shutil.copy2(args.db, backup)
    print(f"  backup -> {backup.name}")

    changed = walk(data)

    # Collapsing the spellings makes pairs the lexicon recorded both ways
    # collide. Two entries for one surface is not merely untidy: the
    # analyser returns 'derived' rather than 'valid' whenever a surface has
    # more than one entry. Reuse the merge the ia -> ïa pass introduced.
    spec = importlib.util.spec_from_file_location(
        "fix_ia", Path(__file__).with_name("fix_ia_diaeresis.py"))
    fix_ia = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fix_ia)
    merge = fix_ia.dedupe(data)
    print(f"  deduped: {merge['merged']} entries merged across "
          f"{merge['surfaces']} surfaces, {merge['glosses_added']} glosses kept")

    after = [s for s in surfaces(data["lexicon"]) if fix(s) != s]
    stamp = datetime.now().strftime("%Y-%m-%d")
    data.setdefault("meta", {})["vin_diaeresis_fix"] = {
        "date": stamp, "strings_changed": changed,
        "surfaces_before": len(targets), "surfaces_after": len(after),
        "merged": merge["merged"], "left_alone": untouched,
    }
    args.db.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    print(f"  strings changed: {changed}")
    print(f"  surfaces still wrong: {len(after)}")
    print(f"  lexicon entries: {len(data['lexicon'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
fix_ain_diaeresis.py — normalise `aïñ` to `aiñ` across the lexicon.

Khasi writes the sequence as **a + i + ñ**: a plain `i` in the diphthong,
with the tilde on the nasal alone. The lexicon is inconsistent about it —
553 surfaces use `aiñ` and 84 use `aïñ`, and **20 words appear both ways**,
which is direct evidence of an error rather than a variant:

    kyllaïñ bol / kyllaiñ …      khlaïñ mlan / khlaiñ …
    ka kulai ka bakhlaïñ / …     haban da miet ngaïñ / …

Left alone it produces nonsense alternatives. `nongthain` was offered
`nongthaïñ` — two diacritics where the orthography has one — because the
variant generator found that spelling in the lexicon and took it as
attested.

The rewrite runs over every string in each entry, not just `form.surface`:
variants, normalized forms and the morphology structure all carry surfaces,
and the earlier ia -> ïa pass showed what happens when those fall out of
step with each other. `aïñ` cannot occur in an English gloss, so a
whole-entry replacement is safe.

    python3 scripts/fix_ain_diaeresis.py --dry-run
    python3 scripts/fix_ain_diaeresis.py
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

WRONG = "aïñ"
RIGHT = "aiñ"


def walk(node):
    """Rewrite in place, returning how many strings changed."""
    changed = 0
    if isinstance(node, dict):
        for key, value in node.items():
            if isinstance(value, str) and WRONG in value:
                node[key] = value.replace(WRONG, RIGHT)
                changed += 1
            else:
                changed += walk(value)
    elif isinstance(node, list):
        for i, value in enumerate(node):
            if isinstance(value, str) and WRONG in value:
                node[i] = value.replace(WRONG, RIGHT)
                changed += 1
            else:
                changed += walk(value)
    return changed


def main() -> int:
    ap = argparse.ArgumentParser(description="normalise aïñ -> aiñ")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--db", type=Path, default=DB)
    args = ap.parse_args()

    data = json.loads(args.db.read_text(encoding="utf-8"))

    def surfaces(block):
        return [((e.get("form") or {}).get("surface") or "") for e in block]

    before = sum(1 for s in surfaces(data["lexicon"]) if WRONG in s)
    entries = [e for e in data["lexicon"]
               if json.dumps(e, ensure_ascii=False).count(WRONG)]
    print(f"  lexicon entries touching '{WRONG}': {len(entries)}")
    print(f"    of which the surface itself: {before}")

    if args.dry_run:
        for e in entries[:12]:
            s = (e.get("form") or {}).get("surface") or ""
            if WRONG in s:
                print(f"      {s}  ->  {s.replace(WRONG, RIGHT)}")
        print("\n  --dry-run: nothing written")
        return 0

    backup = args.db.with_name(
        f"{args.db.name}.bak.{datetime.now():%Y%m%d_%H%M%S}")
    shutil.copy2(args.db, backup)
    print(f"  backup -> {backup.name}")

    changed = walk(data)

    # Collapsing the spellings makes pairs the lexicon recorded both ways
    # collide — `kyllaïñ bol` and `kyllaiñ bol` become one surface. Two
    # entries for one surface is not merely untidy: the analyser returns
    # 'derived' rather than 'valid' whenever a surface has more than one
    # entry, so duplicates change how the word analyses. The ia -> ïa pass
    # hit this first and its merge is reused here rather than rewritten.
    spec = importlib.util.spec_from_file_location(
        "fix_ia", Path(__file__).with_name("fix_ia_diaeresis.py"))
    fix_ia = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fix_ia)
    merge = fix_ia.dedupe(data)
    print(f"  deduped: {merge['merged']} entries merged across "
          f"{merge['surfaces']} surfaces, {merge['glosses_added']} glosses kept")
    after = sum(1 for s in surfaces(data["lexicon"]) if WRONG in s)
    data.setdefault("meta", {})["ain_diaeresis_fix"] = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "strings_rewritten": changed,
        "entries_touched": len(entries),
        "merged": merge["merged"],
    }
    args.db.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    print(f"  {changed} strings rewritten; surfaces still wrong: {after}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

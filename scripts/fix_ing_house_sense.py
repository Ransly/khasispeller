#!/usr/bin/env python3
"""
fix_ing_house_sense.py — spell the house word `ïing`, and only that word.

`ing` is not one word in this lexicon. It is at least four, and they happen
to share a spelling:

    house    ing ai-bam 'hotel', ing-kirja 'chapel', leit-ing-briew 'to marry'
    burn     ing ding 'catch fire', ing-thap 'to singe'
    ginger   ing-bah, ing-makhir 'species of ginger'
    back     ing-dong 'back; backbone', pynkdor ing-dong 'hunch'
    anger    ing-nud 'be angry', buh ing-tyrkhong 'cherish enmity'

Only the **house** sense is the word spelled `ïing`. A rule of "everything
that is not about burning" would have rewritten ginger and backbone into
house, so the senses are read from the glosses one at a time instead.

Why the house sense needs rewriting at all
------------------------------------------
The lexicon carries all three spellings — `ing`, `iing`, `ïing` — under the
same root `iing`. In 3,167 corpus occurrences of `ing`, every one is the
noun: `shna ing` 'build a house', `hapoh ing` 'inside the house', `poi ing`
'reach home'. Not one is the verb. So where an entry means house, `ïing` is
the spelling, and leaving `ing` there keeps a variant of the same word in
the lexicon under a spelling the orthography does not use.

The entries that mean burn, ginger, back or anger are untouched: they are
different words that merely look alike.

    python3 scripts/fix_ing_house_sense.py --dry-run
    python3 scripts/fix_ing_house_sense.py
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "khasi_db.json"

# The house-sense entries, identified from their glosses and listed
# explicitly rather than matched by pattern — the senses overlap too much
# for a regular expression to separate them safely.
HOUSE_SENSE = {
    "ing ai-bam",
    "ing ap-phira",
    "ing-bam ki baduk",
    "ing lum dorbar u paid ksuid",
    "bam kwai ha ing u blei",
    "ing-kirja",
    "leit-ing-briew",
}

# The apostrophe is in the lookbehind because a glottal stop before `ing`
# marks yet another set of homographs — `sh'ing` 'bone', `s'ing` 'ginger',
# `k'ing` 'a wasp', `'ing-dong` 'the back'. None is the house word, and
# without the guard the substitution would reach inside them.
_TOKEN_ING = re.compile(r"(?<![a-zïñ'])ing(?![a-zïñ])")


def surface(entry: dict) -> str:
    return ((entry.get("form") or {}).get("surface") or "").lower()


def gloss(entry: dict) -> str:
    g = (entry.get("semantics") or {}).get("gloss")
    if isinstance(g, list):
        g = " ; ".join(str(x) for x in g)
    return " ".join(str(g or "").split())


def retarget(node, old: str, new: str) -> int:
    n = 0
    if isinstance(node, dict):
        items = list(node.items())
    elif isinstance(node, list):
        items = list(enumerate(node))
    else:
        return 0
    for k, v in items:
        if isinstance(v, str) and _TOKEN_ING.search(v):
            node[k] = _TOKEN_ING.sub("ïing", v)
            n += 1
        else:
            n += retarget(v, old, new)
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description="spell the house sense ïing")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--db", type=Path, default=DB)
    args = ap.parse_args()

    data = json.loads(args.db.read_text(encoding="utf-8"))
    targets = [e for e in data["lexicon"] if surface(e) in HOUSE_SENSE]
    missing = HOUSE_SENSE - {surface(e) for e in targets}

    print(f"  house-sense entries to rewrite: {len(targets)}")
    for e in targets:
        print(f"      {surface(e):<30} -> "
              f"{_TOKEN_ING.sub('ïing', surface(e)):<30} {gloss(e)[:40]}")
    if missing:
        print(f"  not found (already rewritten?): {sorted(missing)}")

    others = [e for e in data["lexicon"]
              if _TOKEN_ING.search(surface(e)) and surface(e) not in HOUSE_SENSE]
    print(f"\n  left untouched — different words that look alike: {len(others)}")
    for e in others:
        print(f"      {surface(e):<30} {gloss(e)[:46]}")

    if args.dry_run or not targets:
        print("\n  --dry-run: nothing written" if args.dry_run else "\n  nothing to do")
        return 0

    backup = args.db.with_name(
        f"{args.db.name}.bak.{datetime.now():%Y%m%d_%H%M%S}")
    shutil.copy2(args.db, backup)
    print(f"\n  backup -> {backup.name}")

    changed = sum(retarget(e, "ing", "ïing") for e in targets)
    data.setdefault("meta", {})["ing_house_sense_fix"] = {
        "date": datetime.now().strftime("%Y-%m-%d"),
        "entries": len(targets), "strings_rewritten": changed,
        "note": "Only the house sense. The burn, ginger, back and anger "
                "senses share the spelling and are different words.",
    }
    args.db.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    print(f"  {changed} strings rewritten across {len(targets)} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

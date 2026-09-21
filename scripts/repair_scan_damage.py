#!/usr/bin/env python3
"""
repair_scan_damage.py — correct or withdraw entries confirmed as damage.

Four readings confirmed by a Khasi speaker, all of them things the printed
dictionary's scan introduced rather than anything Khasi allows:

  * no stray bracket or punctuation in a word — `kha)`, `ing)`, `sdiu-lun:a`
  * no foreign diacritic and no digit — `shiüng`, `la-shem-taiëw`, `khub6r`
  * no segment without a vowel — `bd-liim`, `beit-shdn`, `khdw-eit`
  * the onsets `jk`, `lm`, `pb`, `shm`, `jb`, `mt`, `tk`, `jd`, `tj`

A leading `*` is treated the same way. It marked a borrowed headword in the
printed dictionary and was captured as if it were a letter: stripping it
lands on an existing entry 22 times out of 24 (`*dorbin` -> `dorbin`,
`*dukandar` -> `dukandar`), which is what identifies it as a marker.

Three outcomes, decided per entry rather than by group:

  1. **quarantine as a duplicate** — a repair lands on a surface the lexicon
     already holds, so the record adds nothing. 34 entries.
  2. **rewrite the surface** — exactly one repair is phonotactically valid
     and it is not yet an entry, so the word is kept with its gloss and the
     correct spelling. 16 entries.
  3. **quarantine unresolved** — several repairs are valid, or none is, so
     the intended word cannot be recovered from the data. The entry moves
     to `quarantine` **with its gloss and the candidate repairs recorded**,
     so a reader can settle it later. Nothing is deleted.

Repairs are principled, not guessed: strip the marker, strip stray
punctuation, fold a foreign diacritic to its base letter, map a digit to the
letter it was misread for, or try each vowel in turn where a `d` stands in a
vowel's place. Each candidate is then checked against the lexicon and
`phonology.validate()`.

    python3 scripts/repair_scan_damage.py --dry-run
    python3 scripts/repair_scan_damage.py
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
sys.path.insert(0, str(ROOT))
DB = ROOT / "data" / "khasi_db.json"

FOLD = {"ü": "u", "ë": "e", "ô": "o", "â": "a", "û": "u", "ê": "e",
        "ö": "o", "ä": "a", "í": "ï"}
DIGIT = {"6": "o", "2": "a", "1": "i", "0": "o", "5": "s"}
_STRAY = re.compile(r"[)(:.–]")
_MARKUP = re.compile(r"[^a-zïñáéíóúý'\-]")

INVALID_ONSETS = ("jk", "lm", "pb", "shm", "jb", "mt", "tk", "jd", "tj")

# Read at build time only, never bundled and never imported by the package.
# Its one job is to stop a rewrite from putting an English word into the
# lexicon: `stones)` and `onomatopsea)` are gloss text captured as
# headwords, and stripping the bracket would leave `stones` and
# `onomatopsea` looking like entries. Absent, the guard simply does not fire.
_ENGLISH_PATHS = ("/usr/share/dict/american-english",
                  "/usr/share/dict/british-english")


def _english() -> set:
    words: set = set()
    for path in _ENGLISH_PATHS:
        try:
            words |= {w.strip() for w in
                      Path(path).read_text(encoding="utf-8",
                                           errors="replace").splitlines()
                      if w.strip().isalpha()}
        except OSError:
            pass
    return words


def surface(entry: dict) -> str:
    return ((entry.get("form") or {}).get("surface") or "").lower()


def gloss(entry: dict) -> str:
    g = (entry.get("semantics") or {}).get("gloss")
    if isinstance(g, list):
        g = " ; ".join(str(x) for x in g)
    return str(g or "")[:60]


def repairs(word: str, vowels: set) -> list[str]:
    out = []
    s = word.lstrip("*")
    if s != word:
        out.append(s)
    t = _STRAY.sub("", word)
    if t != word:
        out.append(t)
    f = "".join(FOLD.get(c, c) for c in word)
    if f != word:
        out.append(f)
    d = "".join(DIGIT.get(c, c) for c in word)
    if d != word:
        out.append(d)
    for i, c in enumerate(word):
        if c == "d":
            out.extend(word[:i] + v + word[i + 1:] for v in "aeiou")
    return out


def retarget(node, old: str, new: str) -> int:
    n = 0
    if isinstance(node, dict):
        items = node.items()
    elif isinstance(node, list):
        items = enumerate(node)
    else:
        return 0
    for k, v in list(items):
        if isinstance(v, str) and old in v:
            node[k] = v.replace(old, new)
            n += 1
        else:
            n += retarget(v, old, new)
    return n


def main() -> int:
    ap = argparse.ArgumentParser(description="repair or withdraw scan damage")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--db", type=Path, default=DB)
    args = ap.parse_args()

    from khasi_engine import phonology

    vowels = set(phonology.VOWELS)
    english = _english()
    data = json.loads(args.db.read_text(encoding="utf-8"))
    lexicon = data["lexicon"]
    surfaces = {surface(e) for e in lexicon if surface(e)}

    def onset(w: str) -> str:
        out = ""
        for ch in w:
            if ch in vowels or ch == "-":
                break
            out += ch
        return out

    def damaged(w: str) -> bool:
        if _MARKUP.search(w):
            return True
        if onset(w) in INVALID_ONSETS:
            return True
        return any(p and not any(c in vowels for c in p) for p in w.split("-"))

    dupes, rewrite, unresolved = [], [], []
    for entry in lexicon:
        w = surface(entry)
        if not w or " " in w or not damaged(w):
            continue
        cands = repairs(w, vowels)
        landed = [c for c in cands if c in surfaces and c != w]
        if landed:
            dupes.append((entry, w, landed[0]))
            continue
        valid = sorted({c for c in cands
                        if c != w and phonology.validate(c)["pass"]
                        and c not in english})
        if len(valid) == 1:
            rewrite.append((entry, w, valid[0]))
        else:
            unresolved.append((entry, w, valid))

    print(f"  quarantine as duplicate : {len(dupes)}")
    print(f"  rewrite the surface     : {len(rewrite)}")
    print(f"  quarantine unresolved   : {len(unresolved)}")
    if args.dry_run:
        for _, w, t in rewrite[:10]:
            print(f"      rewrite  {w:<20} -> {t}")
        for _, w, v in unresolved[:6]:
            print(f"      unresolved {w:<18} candidates={v[:3]}")
        print("\n  --dry-run: nothing written")
        return 0

    backup = args.db.with_name(
        f"{args.db.name}.bak.{datetime.now():%Y%m%d_%H%M%S}")
    shutil.copy2(args.db, backup)
    print(f"\n  backup -> {backup.name}")

    stamp = datetime.now().strftime("%Y-%m-%d")
    changed = 0
    for entry, old, new in rewrite:
        changed += retarget(entry, old, new)
    for entry, w, target in dupes:
        entry["quarantine_reason"] = (
            f"scan damage; repaired it is {target!r}, which the lexicon "
            f"already holds, so this record is a duplicate [{stamp}]")
    for entry, w, cands in unresolved:
        entry["quarantine_reason"] = (
            f"scan damage confirmed as not valid Khasi; the intended word "
            f"cannot be recovered from the data "
            f"(candidate repairs: {cands[:4] or 'none valid'}; "
            f"gloss: {gloss(entry)!r}) [{stamp}]")
    drop = {id(e) for e, _, _ in dupes} | {id(e) for e, _, _ in unresolved}
    data.setdefault("quarantine", []).extend(
        [e for e, _, _ in dupes] + [e for e, _, _ in unresolved])
    data["lexicon"] = [e for e in lexicon if id(e) not in drop]
    data.setdefault("meta", {})["scan_damage_repair"] = {
        "date": stamp, "rewritten": len(rewrite), "strings_changed": changed,
        "quarantined_duplicate": len(dupes),
        "quarantined_unresolved": len(unresolved),
    }
    args.db.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    print(f"  lexicon {len(lexicon)} -> {len(data['lexicon'])}; "
          f"{changed} strings rewritten")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

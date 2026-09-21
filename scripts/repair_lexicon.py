#!/usr/bin/env python3
"""
repair_lexicon.py — one pass over the lexicon, one decision procedure.

This replaces six separate scripts (`quarantine_illegal_chars`,
`quarantine_confirmed_damage`, `repair_scan_damage`, `fix_trailing_hyphen`,
`fix_umlaut_and_reduplication`, `fix_ain_diaeresis`) that had grown apart in
four measurable ways:

  * `surface()` was redefined in all six, `retarget()` in three;
  * four rules were implemented in two scripts each — `no_vowel`,
    `invalid_onset`, `markup`, `diacritic` — so an entry's fate depended on
    which script happened to run first;
  * the same defect class got different treatment: an illegal-character
    entry that *could* be repaired was quarantined, while a structurally
    identical case elsewhere was rewritten;
  * only one of the six compared **meaning** before withdrawing a record on
    a collision, so three of them could silently create a duplicate surface
    or collapse two senses.

Rules are data here, in one ordered list, and every one of them goes through
the same `decide()`. Overlap becomes visible instead of accidental, and the
collision policy applies everywhere rather than in one place.

The decision procedure
----------------------
For each entry the first matching rule proposes repairs, and then:

  1. a repair is already a lexicon surface -> **collision**, settled on
     meaning: the same word is withdrawn as a duplicate. Different senses
     mean the repair is wrong, so the record is withdrawn as unresolved
     instead — never merged into the other entry, and never left live,
     because every rule here detects something confirmed as not Khasi;
  2. exactly one valid repair and no collision -> **rewrite in place**,
     keeping the entry and its gloss;
  3. otherwise -> **quarantine**, carrying the gloss and the candidate
     repairs so a reader can settle it later.

Nothing is ever deleted. Every withdrawn record moves to the top-level
`quarantine` list with a reason.

The rules, and the evidence for each
------------------------------------
`illegal_char`   c f q v x z are not in the Khasi alphabet
                 (phonology.valid_chars). 260 entries; the surviving ones
                 came from English gloss text leaking into surface fields.
`stray_g`        `g` occurs only in the digraph `ng`; valid_chars_note says
                 so. 116 entries. Exposed `ug` misread for `ng`.
`no_vowel`       every Khasi word has a vowel nucleus. 28 entries.
`hyphen_letter`  a hyphenated part that is one vowel-less letter: `ka-n` is
                 the contraction `ka'n`, attested. 7 entries. A single
                 letter *in the middle* is fine — `blang-u-bhed` — because
                 `u`, `i`, `a` are real words there.
`trailing_vowel` a single vowel at the end splits it from its stem:
                 `tyr-a` is `tyra`, `ioh-i` is `ïohi`. 7 entries.
`stray_onset`    jk lm pb shm jb mt tk jd tj are not Khasi onsets. 46
                 entries, 26 of which repaired onto an existing word.
`lead_consonant` NEW. A spurious consonant before the real word:
                 `ljing`->`jing`, `mdong`->`dong`, `mthen`->`then`,
                 `ttyngka`->`tyngka`, `jmong`->`mong`, `tjtai`->`tai`.
                 Identified by the repair landing on an existing entry, so
                 it is provable rather than guessed.
`markup`         a stray bracket, digit or foreign diacritic. A leading `*`
                 marked a borrowed headword in the printed dictionary and
                 was captured as a letter: stripping it lands on an existing
                 entry 22 times out of 24.
`diacritic`      `aïñ` is `aiñ` (553 against 84, 20 words spelled both
                 ways); `ü` is `ïi` (`üng` is `ïing`); `í` is `ï`.
`reduplication`  when one half of an `A-B` compound is a known word and the
                 other carries a scan artefact, the good half says what the
                 bad one should be. Kept narrow: Khasi echo-reduplication
                 alternates the vowel on purpose (`jirwit-jirwat`) and many
                 pairs are coordinate compounds (`ïadih-ïabam` is
                 'drink-eat'), so a half is only repaired when it carries a
                 confirmed OCR confusion or fails validate() outright.

    python3 scripts/repair_lexicon.py --dry-run
    python3 scripts/repair_lexicon.py --rule lead_consonant
    python3 scripts/repair_lexicon.py
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Optional

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
DB = ROOT / "data" / "khasi_db.json"

ILLEGAL = set("cfqvxz")
INVALID_ONSETS = ("jk", "lm", "pb", "shm", "jb", "mt", "tk", "jd", "tj")

# Onsets confirmed as not Khasi, where the word is a real word with a
# spurious consonant in front of it. Restricted to what has actually been
# confirmed, because the general form of this rule — "dropping the first
# letter lands on an existing entry" — is wrong: Khasi has minimal pairs
# that satisfy it. `ki'm` reduces to `i'm` and `ki'n` to `i'n`, but those
# are different pronouns (`ki` 'they' against `i` 'it'), not damage. The
# meaning check caught all 22 such cases; this list stops them arising.
LEAD_CONSONANT_ONSETS = ("tjt", "mdwk", "jsl", "rsh", "mth", "lj", "jm")
LEAD_CONSONANT_WORDS = ("jtiiiikup", "beiii", "biid", "biik")
FOLD = {"ü": "ïi", "ë": "e", "ô": "o", "â": "a", "û": "u", "ê": "e",
        "ö": "o", "ä": "a", "í": "ï"}
DIGIT = {"6": "o", "2": "a", "1": "i", "0": "o", "5": "s"}
STRAY_PUNCT = re.compile(r"[)(:.–]")
NON_KHASI = re.compile(r"[^a-zïñáéíóúý'\-]")
STRAY_G = re.compile(r"(?<!n)g")
OCR_MARKER = re.compile(r"ii|rn")

# Read at build time only — never bundled, never imported by the package.
# Its one job is to stop a rewrite putting an English word into the lexicon.
_ENGLISH_PATHS = ("/usr/share/dict/american-english",
                  "/usr/share/dict/british-english")


# ----------------------------------------------------------------------
# shared primitives — defined once
# ----------------------------------------------------------------------

def surface(entry: dict) -> str:
    return ((entry.get("form") or {}).get("surface") or "").lower()


def gloss(entry: dict) -> str:
    g = (entry.get("semantics") or {}).get("gloss")
    if isinstance(g, list):
        g = " ; ".join(str(x) for x in g)
    return " ".join(str(g or "").split())


def pos(entry: dict) -> Optional[str]:
    return (entry.get("grammar") or {}).get("pos")


def retarget(node, old: str, new: str) -> int:
    """Rewrite every string in the entry that carries the old surface."""
    n = 0
    if isinstance(node, dict):
        items = list(node.items())
    elif isinstance(node, list):
        items = list(enumerate(node))
    else:
        return 0
    for k, v in items:
        if isinstance(v, str) and old in v:
            node[k] = v.replace(old, new)
            n += 1
        else:
            n += retarget(v, old, new)
    return n


def same_word(a: dict, b: dict) -> bool:
    """
    Do two records describe the same word? Compared on meaning.

    Identical glosses settle it. So does one gloss being a usage example of
    the other: the scan split some entries in two, leaving a record that
    quotes the headword instead of defining it. The quotation usually
    carries the same damage the headword did — `baiii-bain` is glossed
    "As ; Jem baifi-bain)" — so the known confusions are folded first.
    """
    ga, gb = gloss(a).lower(), gloss(b).lower()
    if ga == gb:
        return True
    if pos(a) != pos(b):
        return False
    short, long_ = sorted((ga, gb), key=len)
    quoted = short.replace("fi", "ñ").replace("iii", "ñ").replace("ii", "ñ")
    stem = surface(b).split("-")[0][:4]
    return (len(short) < 30 and bool(stem)
            and (stem in short or stem in quoted) and short not in long_)


# ----------------------------------------------------------------------
# rules
# ----------------------------------------------------------------------

@dataclass
class Ctx:
    vowels: set
    surfaces: set
    english: set
    validate: Callable
    is_known: Callable


@dataclass
class Rule:
    name: str
    detect: Callable[[str, Ctx], bool]
    repairs: Callable[[str, Ctx], list]
    note: str = ""


def _onset(word: str, ctx: Ctx) -> str:
    out = ""
    for ch in word:
        if ch in ctx.vowels or ch == "-":
            break
        out += ch
    return out


def _vowelless_parts(word: str, ctx: Ctx) -> list:
    return [p for p in word.split("-")[1:]
            if len(p) == 1 and not any(c in ctx.vowels for c in p)]


def _is_reduplication_damage(word: str, ctx: Ctx):
    if word.count("-") != 1:
        return None
    a, b = word.split("-")
    if a == b or not a or not b or a[:2] != b[:2]:
        return None
    ka, kb = ctx.is_known(a), ctx.is_known(b)
    if ka == kb:
        return None
    good, bad = (a, b) if ka else (b, a)
    if not (OCR_MARKER.search(bad) or not ctx.validate(bad)["pass"]):
        return None
    twin = good[:-1] + "ñ" if good.endswith("ain") else ""
    target = twin if twin and twin in ctx.surfaces else good
    return f"{target}-{target}"


RULES = [
    Rule("illegal_char",
         lambda w, c: bool(set(w) & ILLEGAL),
         lambda w, c: []),
    Rule("stray_g",
         lambda w, c: bool(STRAY_G.search(w)),
         lambda w, c: []),
    Rule("no_vowel",
         lambda w, c: not any(ch in c.vowels for ch in w),
         lambda w, c: []),
    Rule("hyphen_letter",
         lambda w, c: bool(_vowelless_parts(w, c)),
         lambda w, c: [w.replace("-", "'")]),
    Rule("trailing_vowel",
         lambda w, c: (len(w.split("-")[-1]) == 1
                       and w.split("-")[-1] in c.vowels and "-" in w),
         lambda w, c: [w.replace("-", ""),
                       ("ï" + w.replace("-", "")[1:])
                       if w.replace("-", "").startswith("i") else ""]),
    Rule("stray_onset",
         lambda w, c: _onset(w, c) in INVALID_ONSETS,
         lambda w, c: [sub + w[len(_onset(w, c)):]
                       for sub in {"jk": ["k", "kh"], "jb": ["b"], "pb": ["ph"],
                                   "mt": ["m"], "tk": ["k"], "jd": ["d"],
                                   "shm": ["sm", "shn"], "lm": ["im"],
                                   "tj": ["t", "j"]}.get(_onset(w, c), [])]),
    Rule("lead_consonant",
         lambda w, c: (_onset(w, c) in LEAD_CONSONANT_ONSETS
                       or w in LEAD_CONSONANT_WORDS),
         lambda w, c: [w[1:], w[2:]] if len(w) > 3 else [],
         "a spurious consonant before the real word"),
    Rule("markup",
         lambda w, c: bool(NON_KHASI.search(w)),
         lambda w, c: [w.lstrip("*"), STRAY_PUNCT.sub("", w),
                       "".join(FOLD.get(ch, ch) for ch in w).replace("ïï", "ï"),
                       "".join(DIGIT.get(ch, ch) for ch in w)]),
    Rule("diacritic",
         lambda w, c: "aïñ" in w or "ü" in w or "í" in w,
         lambda w, c: [w.replace("aïñ", "aiñ").replace("ü", "ïi")
                        .replace("í", "ï").replace("ïï", "ï")]),
    Rule("reduplication",
         lambda w, c: _is_reduplication_damage(w, c) is not None,
         lambda w, c: [_is_reduplication_damage(w, c)]),
]


# ----------------------------------------------------------------------

def decide(entry, word, cands, ctx, by_surface):
    """The single outcome policy. Returns (outcome, target)."""
    cands = [c for c in dict.fromkeys(cands) if c and c != word]
    for c in cands:
        other = by_surface.get(c)
        if other is not None and other is not entry:
            return ("duplicate", c) if same_word(entry, other) else ("conflict", c)
    valid = [c for c in cands
             if ctx.validate(c)["pass"] and c not in ctx.english]
    if len(valid) == 1:
        return "rewrite", valid[0]
    return "quarantine", (valid or cands)


def main() -> int:
    ap = argparse.ArgumentParser(description="repair the lexicon in one pass")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--rule", help="apply only this rule")
    ap.add_argument("--db", type=Path, default=DB)
    args = ap.parse_args()

    import contextlib
    with contextlib.redirect_stdout(sys.stderr):
        from khasi_engine import phonology
        from khasi_spell import KhasiSpeller
        db_obj = KhasiSpeller(eager=True).analyser.db

    english = set()
    for path in _ENGLISH_PATHS:
        try:
            english |= {w.strip() for w in Path(path).read_text(
                encoding="utf-8", errors="replace").splitlines()
                if w.strip().isalpha()}
        except OSError:
            pass

    data = json.loads(args.db.read_text(encoding="utf-8"))
    lexicon = data["lexicon"]
    by_surface = {surface(e): e for e in lexicon if surface(e)}
    ctx = Ctx(vowels=set(phonology.VOWELS), surfaces=set(by_surface),
              english=english, validate=phonology.validate,
              is_known=db_obj.is_known)

    rules = [r for r in RULES if not args.rule or r.name == args.rule]
    actions = []
    for entry in lexicon:
        w = surface(entry)
        if not w or " " in w:
            continue
        for rule in rules:
            try:
                if not rule.detect(w, ctx):
                    continue
            except Exception:
                continue
            outcome, target = decide(entry, w, rule.repairs(w, ctx),
                                     ctx, by_surface)
            actions.append((rule.name, entry, w, outcome, target))
            break

    import collections
    tally = collections.Counter((a[0], a[3]) for a in actions)
    print(f"  {len(actions)} entries matched a rule\n")
    print(f"  {'rule':<18}{'rewrite':>9}{'duplicate':>11}{'quarantine':>12}{'conflict':>10}")
    for rule in rules:
        row = [tally.get((rule.name, k), 0)
               for k in ("rewrite", "duplicate", "quarantine", "conflict")]
        if any(row):
            print(f"  {rule.name:<18}{row[0]:>9}{row[1]:>11}{row[2]:>12}{row[3]:>10}")
    for name, _, w, outcome, target in actions:
        if outcome in ("rewrite", "conflict"):
            print(f"      [{outcome}] {name}: {w} -> {target}")

    if args.dry_run:
        print("\n  --dry-run: nothing written")
        return 0
    if not actions:
        print("\n  nothing to do")
        return 0

    backup = args.db.with_name(
        f"{args.db.name}.bak.{datetime.now():%Y%m%d_%H%M%S}")
    shutil.copy2(args.db, backup)
    print(f"\n  backup -> {backup.name}")

    stamp = datetime.now().strftime("%Y-%m-%d")
    changed = 0
    drop = set()
    for name, entry, w, outcome, target in actions:
        if outcome == "rewrite":
            changed += retarget(entry, w, target)
        elif outcome == "duplicate":
            entry["quarantine_reason"] = (
                f"{name}: repaired it is {target!r}, which the lexicon already "
                f"holds with the same meaning, so this record is a duplicate "
                f"[{stamp}]")
            drop.add(id(entry))
        elif outcome == "quarantine":
            entry["quarantine_reason"] = (
                f"{name}: not valid Khasi and the intended word cannot be "
                f"recovered (candidates: {target or 'none valid'}; "
                f"gloss: {gloss(entry)[:60]!r}) [{stamp}]")
            drop.add(id(entry))
        elif outcome == "conflict":
            # The repair collides with an entry that means something else,
            # so the repair is wrong. The record is still not valid Khasi,
            # so it is withdrawn rather than left live — with the colliding
            # word named, since that is the evidence against the repair.
            entry["quarantine_reason"] = (
                f"{name}: not valid Khasi; the obvious repair {target!r} is a "
                f"different word, so the intended form is unrecoverable "
                f"(gloss: {gloss(entry)[:60]!r}) [{stamp}]")
            drop.add(id(entry))
    data.setdefault("quarantine", []).extend(
        e for _, e, _, _, _ in actions if id(e) in drop)
    data["lexicon"] = [e for e in lexicon if id(e) not in drop]
    data.setdefault("meta", {})["repair_lexicon"] = {
        "date": stamp, "matched": len(actions), "rewritten": changed,
        "withdrawn": len(drop),
    }
    args.db.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    print(f"  lexicon {len(lexicon)} -> {len(data['lexicon'])}; "
          f"{changed} strings rewritten")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

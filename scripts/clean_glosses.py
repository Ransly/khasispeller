#!/usr/bin/env python3
"""
clean_glosses.py — remove scanning debris from the English glosses.

The lexicon was digitised from a scanned printed dictionary, and some of the
OCR noise is still in the meanings the web page prints: stray `^ \\ « » | ~`,
runs of quote marks, a mangled "[Imit." marker, page numbers at the end of a
column, and — where a scan line ran into the next — a cluster of symbols
followed by text from another entry.

This script only ever edits the text of `semantics.gloss` and
`semantics.gloss_normalized`. It never deletes an entry, never touches a
surface form, and never guesses a missing letter: a symbol that stood for a
letter is removed, not replaced. Every changed string is written, before and
after, to `data/gloss_cleanup_log.json` so the pass can be reviewed and
reversed. That log holds lexicon text, so it is git-ignored.

Why a separate script, not a rule in repair_lexicon.py: that script decides
the fate of whole entries (rewrite / duplicate / quarantine). This one edits
text inside entries and has no outcome other than "cleaned".

Rules, in the order they run (examples are synthetic):

  imit_marker     OCR variants of the dictionary's "[Imit." marker
                  ("[^Imit.", "\\_Imit.", "[ImU.", "[Tmit." …) -> "[Imit.",
                  and a space after a hyphen inside that bracket is closed:
                  "[Imit. ka ab-ka cd.]" stays; "ab- cd" -> "ab-cd".
  label_comma     a caret where the print had a comma after a class label:
                  "Ka^ n. a thing" -> "Ka, n. a thing"; "(xyz)^ word" -> "(xyz), word".
  brace_paren     "{" / "}" used for a parenthesis, when that balances it.
  spillover       a dense cluster of debris symbols (at least four within
                  twelve characters) after substantive text: the rest of the
                  string is scan bleed from a neighbouring line and is cut —
                  unless that rest quotes the entry's own word (an example
                  sentence) or is only a language-of-origin label.
  page_marker     trailing dash runs with a page number: "text — —— — I 123".
  strip_symbols   remaining ^ \\ « » | ~ ° _ are removed.
  quote_runs      runs of two or more quote marks collapse to one.
  debris_only     a gloss element that is at least 40% symbols, or has fewer
                  than three letters left after cleaning, is dropped; an entry left with none gets "[unglossed: scan debris]".

Pass 2 — structure. The glosses were split into a list at every comma, even
inside brackets and numbers, and each piece was capitalised. So a gloss like
"a thing (abc, def)" became ["A thing (abc", "Def)"], and "20,000" became
["20", "000"]. This pass rejoins them:

  join_number     a piece ending in a digit + a piece opening with 2–3 digits
                  -> "20,000" (Indian grouping "2,00,000" too).
  join_bracket    a piece that opens "(" is joined with the following pieces
                  until the bracket closes (at most six), with ", "; the
                  capital the split added is lowered unless the word is a
                  known name in data/gazetteer.json.
  join_square     the same for "[ … ]", joined with a space (at most three).
  join_label      a class label split from its noun: "Ka" + "N. a thing" ->
                  "Ka, n. a thing" (a lone "W" there is a misread "U"); a
                  part-of-speech letter: "V" + "To go" -> "V. to go".
  drop_empty      a piece with no run of two letters or digits ("/", "?/").
  manual          data/gloss_manual_fixes.json, {entry_id: {"gloss": [...],
                  "note": "..."}}: corrections a rule cannot make, applied
                  last and logged like the rest (git-ignored: lexicon text).

gloss_normalized is rebuilt as the pieces joined with "; ", which is how it
was made in the first place. Tokens that no rule can repair — letters mixed
with < > // or capitals inside a word — are listed in data/gloss_review.csv
for a Khasi speaker (git-ignored).

Usage:
    python3 scripts/clean_glosses.py --dry-run     # report only
    python3 scripts/clean_glosses.py               # write, with a backup
"""
from __future__ import annotations

import argparse
import json
import re
import shutil
import sys
from collections import Counter
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "khasi_db.json"
LOG = ROOT / "data" / "gloss_cleanup_log.json"
PLACEHOLDER = "[unglossed: scan debris]"

IMIT_VARIANTS = [
    (re.compile(r"\[\s*\^?\s*[IlT/]m[iU][tL]\b\.?"), "[Imit."),
    (re.compile(r"\\_?\s*[Il]mit\b\.?"), "[Imit."),
    (re.compile(r"\bSJmit\."), "[Imit."),
    (re.compile(r"\bL\^hnit\."), "[Imit."),
    (re.compile(r"\{\s*Imit\."), "[Imit."),
]
IMIT_BRACKET = re.compile(r"\[Imit\.[^\]]*\]?")
HYPHEN_GAP = re.compile(r"(\w)- (\w)")
LABEL_COMMA = re.compile(r"\b([A-Z][a-z]{0,2})\^ ((?:n|a|ad|v|adv|pr|prep|conj|int)\.)")
PAREN_COMMA = re.compile(r"\)\^(?=\s)")
DEBRIS_CHARS = "^\\«»|~°_{}"
QUOTES = "'\"`"
PAGE_MARKER = re.compile(r"(?:\s*[—–-]{1,3}){3,}\s*[IVXl|]{0,4}\s*\d{0,4}\s*$")
# a caret after a digit may stand for a fraction ("2^" for 2½): left for a reader;
# an underscore between letters is part of a name, not debris
SYMBOLS = re.compile(r"(?<!\d)\^|[\\«»|~°]|(?<![A-Za-z0-9])_|_(?![A-Za-z0-9])")
QUOTE_RUN = re.compile(r"[" + re.escape(QUOTES) + r"]{2,}")
SPACES = re.compile(r"\s+")
SPACE_BEFORE_PUNCT = re.compile(r"\s+([,.;:!?\]\)])")
LETTERS = re.compile(r"[A-Za-zÏïÑñ]")
LEADING_PUNCT = re.compile(r"^[\s.,;:\-–—*>]+(?=[A-Za-zÏïÑñ(\"\[])")
DOUBLE_STOP = re.compile(r"(?<!\.)\.\s?\.(?!\.)")


def _spillover_cut(text: str):
    """Index where a dense debris cluster starts after substantive text, or None."""
    marks = [i for i, ch in enumerate(text) if ch in DEBRIS_CHARS or ch in QUOTES]
    for j in range(len(marks)):
        window = [m for m in marks[j:] if m - marks[j] < 12]
        dense = sum(1 for m in window if text[m] in DEBRIS_CHARS)
        if len(window) >= 4 and dense >= 2:
            head = text[:marks[j]]
            if len(LETTERS.findall(head)) >= 8:
                return marks[j]
    return None


LANGUAGES = {"hindi", "bengali", "english", "assamese", "arabic", "persian", "urdu",
             "sanskrit", "portuguese", "nepali", "garo", "khasi"}
WORD = re.compile(r"[A-Za-zÏïÑñ][A-Za-zÏïÑñ'-]*")


def _lev(a: str, b: str) -> int:
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def _keeps_tail(tail: str, headword: str) -> bool:
    """A tail is not scan bleed if it quotes the entry's own word, or is only
    a language-of-origin label."""
    words = [w.lower().strip("'-") for w in WORD.findall(SYMBOLS.sub("", tail))]
    words = [w for w in words if w]
    if words and all(w in LANGUAGES for w in words):
        return True
    parts = [p for p in re.split(r"[-\s]", headword.lower()) if len(p) >= 4] + [headword.lower()]
    return any(_lev(w, p) <= 1 for w in words for p in parts if abs(len(w) - len(p)) <= 1)


def _debris_share(text: str) -> float:
    marks = sum(1 for ch in text if ch in DEBRIS_CHARS or ch in QUOTES)
    visible = sum(1 for ch in text if not ch.isspace())
    return marks / visible if visible else 1.0


def _braces(text: str) -> str:
    if "{" in text and ")" in text[text.index("{"):] and "(" not in text:
        text = text.replace("{", "(", 1)
    if "}" in text and "(" in text[:text.index("}")] and ")" not in text:
        text = text.replace("}", ")", 1)
    return text


MANUAL = ROOT / "data" / "gloss_manual_fixes.json"
REVIEW = ROOT / "data" / "gloss_review.csv"
GAZETTEER = ROOT / "data" / "gazetteer.json"
NUM_TAIL = re.compile(r"\d$")
NUM_HEAD = re.compile(r"^\d{2,3}(?!\d)")
HAS_CONTENT = re.compile(r"[A-Za-zÏïÑñ0-9]{2,}")
CAP_WORD = re.compile(r"^([A-ZÏÑ])([a-zïñ'-]*)")
GARBLED = re.compile(r"\S*[A-Za-z][<>]\S*|\S*[<>][A-Za-z]\S*|\S*//\S*|\b[a-z]+[A-Z]+[A-Za-z]*\b")


def _names() -> set:
    try:
        g = json.loads(GAZETTEER.read_text(encoding="utf-8"))
        return {t.lower() for t in (g.get("tokens") if isinstance(g, dict) else g)}
    except (OSError, ValueError):
        return set()


def _lower_split_capital(piece: str, names: set) -> str:
    m = CAP_WORD.match(piece)
    if m and (m.group(1) + m.group(2)).lower() not in names:
        return m.group(1).lower() + piece[1:]
    return piece


CLASS_LABELS = {"U", "Ka", "Ki", "I"}
POS_LETTERS = {"N", "V"}
NOUN_HEAD = re.compile(r"^N\.?(?:\s+|$)")


def _lower_first(piece: str) -> str:
    """Undo the capital the split added: "Moon" -> "moon", "A thing" -> "a thing"."""
    if not piece[:1].isupper() or piece.startswith(("I ", "I'")):
        return piece
    if piece[1:2].islower() or piece[1:2] in (" ", ""):
        return piece[:1].lower() + piece[1:]
    return piece


def restructure(pieces: list, fired: Counter, names: set) -> list:
    """Rejoin pieces the comma split broke apart; drop empty ones."""
    # 1. numbers: "20" + "000" -> "20,000"
    out = []
    for p in pieces:
        if (out and isinstance(p, str) and isinstance(out[-1], str)
                and NUM_TAIL.search(out[-1]) and NUM_HEAD.match(p)):
            out[-1] = out[-1] + "," + p
            fired["join_number"] += 1
        else:
            out.append(p)
    # 2. brackets: "a thing (abc" + "Def)" -> "a thing (abc, def)"
    res, i = [], 0
    while i < len(out):
        p = out[i]
        if isinstance(p, str) and p.count("(") > p.count(")"):
            j, depth = i, p.count("(") - p.count(")")
            while j + 1 < len(out) and j - i < 6 and depth > 0 and isinstance(out[j + 1], str):
                j += 1
                depth += out[j].count("(") - out[j].count(")")
            if depth <= 0 and j > i:
                joined = p
                for x in out[i + 1:j + 1]:
                    if HAS_CONTENT.search(x) or LETTERS.search(x):
                        joined += ", " + _lower_split_capital(x, names)
                    elif ")" in x:                      # a lone ")" closes, no comma
                        joined += x.strip()
                res.append(joined)
                fired["join_bracket"] += 1
                i = j + 1
                continue
        res.append(p)
        i += 1
    # 2b. square brackets split the same way: "a thing. [Imit." + "Ab-cd.]" ->
    #     "a thing. [Imit. ab-cd.]". Joined with a space — inside a bracket the
    #     break was a line end as often as a comma — and at most three pieces.
    sq, i = [], 0
    while i < len(res):
        p = res[i]
        if isinstance(p, str) and p.count("[") > p.count("]"):
            j, depth = i, p.count("[") - p.count("]")
            while j + 1 < len(res) and j - i < 3 and depth > 0 and isinstance(res[j + 1], str):
                j += 1
                depth += res[j].count("[") - res[j].count("]")
            if depth <= 0 and j > i:
                joined = " ".join([p] + [_lower_split_capital(x, names) for x in res[i + 1:j + 1]])
                sq.append(IMIT_BRACKET.sub(lambda m: HYPHEN_GAP.sub(r"\1-\2", m.group(0)), joined))
                fired["join_square"] += 1
                i = j + 1
                continue
        sq.append(p)
        i += 1
    res = sq
    # 3. labels split from their meaning: "Ka" + "N. hearth" -> "Ka, n. hearth";
    #    "V" + "To swell" -> "V. to swell". A lone "W" before a noun is a misread "U".
    lab, i = [], 0
    while i < len(res):
        p = res[i].strip() if isinstance(res[i], str) else res[i]
        nxt = res[i + 1] if i + 1 < len(res) and isinstance(res[i + 1], str) else None
        if isinstance(p, str) and (p in CLASS_LABELS or p == "W") and nxt is not None:
            rest, k = None, i + 1
            if NOUN_HEAD.match(nxt) and nxt.strip() != "N":
                rest = NOUN_HEAD.sub("", nxt, count=1)
            elif nxt.strip() == "N" and i + 2 < len(res) and isinstance(res[i + 2], str):
                rest, k = res[i + 2], i + 2
            if rest:
                lab.append(f"{'U' if p == 'W' else p}, n. {_lower_first(rest)}")
                fired["join_label"] += 1
                i = k + 1
                continue
        if isinstance(p, str) and p in POS_LETTERS and nxt is not None and HAS_CONTENT.search(nxt):
            lab.append(f"{p}. {_lower_first(nxt)}")
            fired["join_label"] += 1
            i += 2
            continue
        lab.append(res[i])
        i += 1
    # 4. pieces with nothing left to read
    final = []
    for p in lab:
        if (isinstance(p, str) and not p.startswith("[unglossed:") and not HAS_CONTENT.search(p)
                and p.strip() != "I"):             # the pronoun "I" is a meaning, not debris
            fired["drop_empty"] += 1
            continue
        final.append(p)
    return final


def clean(text: str, fired: Counter, headword: str = ""):
    """Return (cleaned, cut_tail). cleaned may be '' for a debris-only string."""
    s, tail = text, ""
    if _debris_share(text) >= 0.4 and len(LETTERS.findall(text)) < 25:
        fired["debris_only"] += 1
        return "", text
    before = s
    for rx, rep in IMIT_VARIANTS:
        s = rx.sub(rep, s)
    s = IMIT_BRACKET.sub(lambda m: HYPHEN_GAP.sub(r"\1-\2", m.group(0)), s)
    if s != before: fired["imit_marker"] += 1
    before = s
    s = LABEL_COMMA.sub(r"\1, \2", s)
    s = PAREN_COMMA.sub("),", s)
    if s != before: fired["label_comma"] += 1
    before = s
    s = _braces(s)
    if s != before: fired["brace_paren"] += 1
    cut = _spillover_cut(s)
    if cut is not None and _keeps_tail(s[cut:], headword):
        fired["spillover_kept"] += 1
        cut = None
    if cut is not None:
        tail = s[cut:]
        s = s[:cut]
        fired["spillover"] += 1
    before = s
    s = PAGE_MARKER.sub("", s)
    if s != before: fired["page_marker"] += 1
    before = s
    s = SYMBOLS.sub("", s)
    s = s.replace("{", "").replace("}", "")
    if s != before: fired["strip_symbols"] += 1
    before = s
    s = QUOTE_RUN.sub('"', s)
    if s != before: fired["quote_runs"] += 1
    s = SPACE_BEFORE_PUNCT.sub(r"\1", SPACES.sub(" ", s)).strip(" ;,")
    s = DOUBLE_STOP.sub(".", LEADING_PUNCT.sub("", s))
    if len(LETTERS.findall(s)) < 3:
        fired["debris_only"] += 1
        return "", tail or text
    return s, tail


def needs_cleaning(text: str) -> bool:
    return bool(SYMBOLS.search(text) or "{" in text or "}" in text
                or QUOTE_RUN.search(text) or PAGE_MARKER.search(text)
                or any(rx.search(text) for rx, _ in IMIT_VARIANTS)
                or LABEL_COMMA.search(text) or PAREN_COMMA.search(text))


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--db", type=Path, default=DB)
    ap.add_argument("--show", type=int, default=0, help="print N before/after examples")
    args = ap.parse_args()

    data = json.loads(args.db.read_text(encoding="utf-8"))
    manual = json.loads(MANUAL.read_text(encoding="utf-8")) if MANUAL.exists() else {}
    surfaces = {((e.get("form") or {}).get("surface") or "").lower() for e in data.get("lexicon", [])}
    names = _names() - surfaces      # a gazetteer token that is also a Khasi word is not a name
    fired, log, review = Counter(), [], []
    for entry in data.get("lexicon", []):
        sem = entry.get("semantics") or {}
        glosses = sem.get("gloss")
        if not isinstance(glosses, list):
            continue
        eid = entry.get("entry_id")
        word = (entry.get("form") or {}).get("surface") or ""
        pieces, tails = [], []
        for g in glosses:                                   # pass 1: debris
            if not isinstance(g, str) or g.startswith("[unglossed:") or not needs_cleaning(g):
                pieces.append(g); continue
            c, tail = clean(g, fired, word)
            if tail:
                tails.append(tail)
            if c:
                pieces.append(c)
        pieces = restructure(pieces, fired, names)          # pass 2: structure
        note = None
        if eid in manual:                                   # hand corrections
            pieces, note = list(manual[eid]["gloss"]), manual[eid].get("note")
            fired["manual"] += 1
        if not pieces:
            pieces = [PLACEHOLDER]
            fired["entry_left_unglossed"] += 1
        for g in pieces:
            bad = GARBLED.findall(g) if isinstance(g, str) else []
            if bad:
                review.append((eid, word, "; ".join(x for x in pieces if isinstance(x, str)), " ".join(bad)))
                break
        if pieces == glosses:
            continue
        normalized = "; ".join(x for x in pieces if isinstance(x, str))
        log.append({"entry_id": eid, "field": "gloss_list", "before": glosses, "after": pieces,
                    "normalized_before": sem.get("gloss_normalized"), "normalized_after": normalized,
                    "removed_tails": tails, "note": note})
        sem["gloss"] = pieces
        sem["gloss_normalized"] = normalized

    print(f"  entries changed: {len(log)}")
    for k, v in fired.most_common():
        print(f"    {k:<22}{v:>6}")
    print(f"  entries with garbled words left for a reader: {len(review)}")
    for r in log[: args.show]:
        print(f"\n  {r['entry_id']}\n    - {r['before']}\n    + {r['after']}"
              + (f"\n    cut: {r['removed_tails']}" if r["removed_tails"] else ""))
    if args.dry_run:
        print("\n  --dry-run: nothing written")
        return 0
    if log:
        backup = args.db.with_name(f"{args.db.name}.bak.{datetime.now():%Y%m%d_%H%M%S}")
        shutil.copy2(args.db, backup)
        args.db.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
        prior = json.loads(LOG.read_text(encoding="utf-8")) if LOG.exists() else []
        stamp = datetime.now().isoformat(timespec="seconds")
        LOG.write_text(json.dumps(prior + [dict(r, run=stamp) for r in log], ensure_ascii=False,
                                  indent=1), encoding="utf-8")
        print(f"\n  backup -> {backup.name}\n  log    -> {LOG.relative_to(ROOT)} ({len(log)} entries)")
    import csv
    with REVIEW.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["entry_id", "surface", "gloss", "garbled", "corrected_gloss", "note"])
        w.writerows([list(r) + ["", ""] for r in review])
    print(f"  review -> {REVIEW.relative_to(ROOT)} ({len(review)} entries)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

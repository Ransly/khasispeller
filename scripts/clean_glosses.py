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
    fired, log = Counter(), []
    for entry in data.get("lexicon", []):
        sem = entry.get("semantics") or {}
        glosses = sem.get("gloss")
        if not isinstance(glosses, list):
            continue
        new_list, changed = [], False
        for g in glosses:
            if not isinstance(g, str) or g.startswith("[unglossed:") or not needs_cleaning(g):
                new_list.append(g); continue
            c, tail = clean(g, fired, (entry.get("form") or {}).get("surface") or "")
            if c != g:
                changed = True
                log.append({"entry_id": entry.get("entry_id"), "field": "gloss",
                            "before": g, "after": c, "removed_tail": tail})
            if c:
                new_list.append(c)
        if not changed:
            continue
        if not new_list:
            new_list = [PLACEHOLDER]
            fired["entry_left_unglossed"] += 1
        sem["gloss"] = new_list
        gn = sem.get("gloss_normalized")
        if isinstance(gn, str) and needs_cleaning(gn):
            c, tail = clean(gn, Counter(), (entry.get("form") or {}).get("surface") or "")
            c = c or PLACEHOLDER
            log.append({"entry_id": entry.get("entry_id"), "field": "gloss_normalized",
                        "before": gn, "after": c, "removed_tail": tail})
            sem["gloss_normalized"] = c

    entries = len({r["entry_id"] for r in log})
    print(f"  gloss strings cleaned: {sum(1 for r in log if r['field'] == 'gloss')} "
          f"in {entries} entries (+{sum(1 for r in log if r['field'] == 'gloss_normalized')} gloss_normalized)")
    for k, v in fired.most_common():
        print(f"    {k:<22}{v:>6}")
    for r in [x for x in log if x["field"] == "gloss"][: args.show]:
        print(f"\n  {r['entry_id']}\n    - {r['before']!r}\n    + {r['after']!r}"
              + (f"\n    cut: {r['removed_tail']!r}" if r["removed_tail"] else ""))
    if args.dry_run:
        print("\n  --dry-run: nothing written")
        return 0
    if not log:
        print("\n  nothing to do"); return 0
    backup = args.db.with_name(f"{args.db.name}.bak.{datetime.now():%Y%m%d_%H%M%S}")
    shutil.copy2(args.db, backup)
    args.db.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    prior = json.loads(LOG.read_text(encoding="utf-8")) if LOG.exists() else []
    stamp = datetime.now().isoformat(timespec="seconds")
    LOG.write_text(json.dumps(prior + [dict(r, run=stamp) for r in log], ensure_ascii=False, indent=1),
                   encoding="utf-8")
    print(f"\n  backup -> {backup.name}\n  log    -> {LOG.relative_to(ROOT)} ({len(log)} changes)")
    return 0


if __name__ == "__main__":
    sys.exit(main())

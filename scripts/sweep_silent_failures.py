#!/usr/bin/env python3
"""
sweep_silent_failures.py -- enumerate the failure class, not the instance.

Read-only diagnostic for the Khasi spellchecker. Nothing is written to the
lexicon; nothing is corrected. It answers one question:

    Which tokens does the checker REJECT while offering NO suggestion --
    and how many of those are valid words the morphology gate should have
    accepted, if it could see compounds inside a derivation?

`iatreilang` (ia- + trei + lang) is the motivating case: not in the lexicon,
residue `treilang` is not a word so the vote scores 1 + 1 = 2 and rejects,
and every lexicon neighbour is 4+ edits away so the index returns an empty
shortlist and `check_sentence` drops the token silently.

Run it over the corpus vocabulary to get the whole class at once, grouped by
structural pattern and ranked by corpus frequency, instead of finding these
one at a time from user reports.

Usage
-----
    python3 sweep_silent_failures.py --corpus-freq data/corpus_freq.json
    python3 sweep_silent_failures.py --corpus-freq data/corpus_freq.json \
        --min-freq 3 --limit 5000 --out silent_failures.csv

Stdlib only, per the project's no-third-party-dependency rule.

NOTE ON THE API
---------------
This script drives KhasiSpeller through the public surface documented in
README.md (`check`, `.confidence`, `.gate_name`, `.suggestions`,
`.analyser.db`). Accessor names on the db object vary, so every lookup goes
through `_resolve` with fallbacks and the script reports which names it
actually bound. Check that report on the first run before trusting the
numbers.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
import time
from collections import Counter, defaultdict

# --------------------------------------------------------------------------
# binding the engine's API defensively
# --------------------------------------------------------------------------


def _resolve(obj, *names):
    """Return the first attribute in `names` that exists and is callable."""
    for n in names:
        fn = getattr(obj, n, None)
        if callable(fn):
            return fn, n
    return None, None


class Engine:
    """Thin adapter over KhasiSpeller so the sweep survives API drift."""

    def __init__(self, eager=True):
        from khasi_spell import KhasiSpeller

        self.sp = KhasiSpeller(eager=eager)
        db = self.sp.analyser.db
        self.db = db

        self._is_known, self.n_known = _resolve(db, "is_known")
        self._is_attested, self.n_attested = _resolve(
            db, "is_attested", "is_known"
        )
        self._all_forms, self.n_forms = _resolve(
            db, "all_surface_forms", "all_forms", "surfaces"
        )
        self._lookup, self.n_lookup = _resolve(db, "lookup", "get", "entry")

    def report_binding(self):
        return {
            "is_known": self.n_known,
            "is_attested": self.n_attested,
            "all_surface_forms": self.n_forms,
            "lookup": self.n_lookup,
        }

    # -- lexicon -----------------------------------------------------------

    def is_known(self, w):
        return bool(self._is_known(w)) if self._is_known else False

    def is_attested(self, w):
        return bool(self._is_attested(w)) if self._is_attested else False

    def all_forms(self):
        return list(self._all_forms()) if self._all_forms else []

    def is_glossed_headword(self, w):
        """A single-word entry with a gloss -- not a compound-token fragment.

        The `_compound_tokens` index is built by splitting multi-word surface
        forms, which is how English gloss text leaked in (`cow`, `and`).
        A recursive splitter must never build on those.
        """
        if not self._lookup:
            return self.is_known(w)
        try:
            e = self._lookup(w)
        except Exception:
            return False
        if not e:
            return False
        if isinstance(e, (list, tuple)):
            e = e[0] if e else None
        if not e:
            return False
        surface = _field(e, "surface", "surface_form", "form") or ""
        # `english_gloss` is what KhasiDB.lookup() actually returns. Without
        # it every lookup came back glossless, so is_glossed_headword() was
        # False for every word in the lexicon, the decomposer never found a
        # root, and the SILENT + ANALYSABLE bucket the script exists to
        # measure was structurally always empty.
        gloss = _field(e, "english_gloss", "gloss", "meaning",
                       "definition", "english") or ""
        return " " not in str(surface).strip() and bool(str(gloss).strip())

    # -- checking ----------------------------------------------------------

    def check(self, word):
        r = self.sp.check(word)
        conf = getattr(r, "confidence", None) or {}
        return {
            "is_correct": bool(getattr(r, "is_correct", False)),
            "gate": getattr(r, "gate_name", None),
            "in_lexicon": bool(getattr(r, "in_lexicon", False)),
            "score": conf.get("score"),
            "threshold": conf.get("threshold"),
            "signals": conf.get("signals") or {},
            "suggestions": list(getattr(r, "suggestions", []) or []),
        }


def _field(entry, *names):
    for n in names:
        if isinstance(entry, dict) and n in entry:
            return entry[n]
        v = getattr(entry, n, None)
        if v is not None:
            return v
    return None


# --------------------------------------------------------------------------
# a standalone recursive analyser -- diagnostic only, touches no engine code
# --------------------------------------------------------------------------

# The six the lexicon marks productive, per README. Verified at runtime
# against morphology.prefixes when that is reachable.
DEFAULT_PRODUCTIVE = ["jing", "pyn", "nong", "sngew", "ia", "ïa"]


class Decomposer:
    """Can this string be built as prefix* + (root | compound of roots)?

    Deliberately a separate implementation from the engine's three splitters
    (complex.py Phase 4, _check_hyphenated, generate.py). The point is to
    measure how much the engine is missing, not to be the fix.
    """

    def __init__(self, engine, prefixes, min_part=3, max_depth=3):
        self.e = engine
        self.prefixes = sorted(prefixes, key=len, reverse=True)
        self.min_part = min_part
        self.max_depth = max_depth
        self._memo = {}

    def analyse(self, word, depth=0):
        """Return a parse tuple, or None. ('root', w) / ('pre', p, sub) /
        ('cmp', left, right)."""
        key = (word, depth)
        if key in self._memo:
            return self._memo[key]
        self._memo[key] = None  # guard against cycles
        res = self._analyse(word, depth)
        self._memo[key] = res
        return res

    def _analyse(self, word, depth):
        if len(word) < self.min_part or depth > self.max_depth:
            return None

        if self.e.is_glossed_headword(word):
            return ("root", word)

        # prefixation
        for p in self.prefixes:
            if word.startswith(p) and len(word) - len(p) >= self.min_part:
                sub = self.analyse(word[len(p):], depth + 1)
                if sub:
                    return ("pre", p, sub)

        # solid compounding -- both halves must analyse
        for i in range(self.min_part, len(word) - self.min_part + 1):
            left, right = word[:i], word[i:]
            if not self.e.is_glossed_headword(left):
                continue
            r = self.analyse(right, depth + 1)
            if r:
                return ("cmp", ("root", left), r)
        return None

    # -- describing a parse ------------------------------------------------

    @staticmethod
    def render(parse):
        if parse is None:
            return ""
        kind = parse[0]
        if kind == "root":
            return parse[1]
        if kind == "pre":
            return "%s- + %s" % (parse[1], Decomposer.render(parse[2]))
        return "%s + %s" % (
            Decomposer.render(parse[1]),
            Decomposer.render(parse[2]),
        )

    @staticmethod
    def shape(parse):
        """A coarse structural label, for grouping the failure class."""
        if parse is None:
            return "unparsed"
        kind = parse[0]
        if kind == "root":
            return "root"
        if kind == "pre":
            inner = Decomposer.shape(parse[2])
            if inner == "root":
                return "prefix + root"
            if inner.startswith("prefix"):
                return "prefix + prefix + ..."
            return "prefix + compound"
        return "compound"

    @staticmethod
    def leaves(parse):
        if parse is None:
            return []
        if parse[0] == "root":
            return [parse[1]]
        if parse[0] == "pre":
            return Decomposer.leaves(parse[2])
        return Decomposer.leaves(parse[1]) + Decomposer.leaves(parse[2])


# --------------------------------------------------------------------------
# input vocabulary
# --------------------------------------------------------------------------


WORD_RE = re.compile(r"[A-Za-zïÏñÑ'\-]+")


def load_corpus_freq(path, min_freq):
    """corpus_freq.json -> [(token, freq)] sorted by frequency desc."""
    with open(path, "r", encoding="utf-8") as fh:
        data = json.load(fh)
    if isinstance(data, dict) and "counts" in data:
        data = data["counts"]
    if not isinstance(data, dict):
        raise SystemExit("unrecognised corpus_freq.json shape: %s" % type(data))
    items = [
        (w, int(c))
        for w, c in data.items()
        if isinstance(c, (int, float)) and int(c) >= min_freq
    ]
    items.sort(key=lambda kv: (-kv[1], kv[0]))
    return items


def load_wordlist(path):
    """One token per line. Optional trailing TAB or comma count."""
    out = []
    with open(path, "r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            parts = re.split(r"[\t,]", line, 1)
            tok = parts[0].strip()
            try:
                freq = int(parts[1].strip()) if len(parts) > 1 else 0
            except (ValueError, IndexError):
                freq = 0
            if tok:
                out.append((tok, freq))
    return out


def load_text(path):
    """Raw Khasi text -> types with in-document counts.

    Use this on writing that actually contains diacritics -- a student
    transcript, an official document, a book chapter. It is the only source
    here that can surface the diacritic half of the failure class from real
    usage rather than by rule.
    """
    with open(path, "r", encoding="utf-8") as fh:
        text = fh.read()
    c = Counter(m.group(0).lower() for m in WORD_RE.finditer(text))
    return sorted(c.items(), key=lambda kv: (-kv[1], kv[0]))


def load_lexicon_forms(engine):
    """Every lexicon surface form. A rejection here is a serious bug."""
    return [(w, 0) for w in engine.all_forms() if w and " " not in w]


def build_probe_set(engine, dec, prefixes, cap=40000):
    """Generate `prefix + compound` forms to probe the gate systematically.

    The bug shape is prefix + (root + root). Corpus sweeps find only what
    people happened to write; this enumerates the shape directly.

    Constrained to stay small and plausible: take lexicon headwords that
    already decompose into two glossed roots, and put each productive
    prefix on the front. `trei` + `lang` is attested as a compound, so
    `ia-`/`ia-` + it is generated whether or not anyone wrote it down.

    These are CANDIDATES, not known-valid words. A rejection here is a
    question to put to a Khasi speaker, not a confirmed bug.
    """
    compounds = []
    for w in engine.all_forms():
        if not w or " " in w or len(w) < 6:
            continue
        p = dec.analyse(w)
        if p and Decomposer.shape(p) == "compound":
            compounds.append(w)
    out = []
    for base in compounds:
        for pre in prefixes:
            if base.startswith(pre):
                continue
            out.append((pre + base, 0))
            if len(out) >= cap:
                return out
    return out


# -- diacritic spellings ---------------------------------------------------

def variant_spellings(word):
    """Rule-based diacritic variants of a plain-spelled token.

    The corpus holds zero i-diaeresis and zero n-tilde across 13.3M tokens,
    so a corpus sweep alone cannot reach the diacritic half of the
    vocabulary -- roughly one headword in eleven. These two rules are the
    documented high-frequency cases and cost one extra check each.

    Word-initial `ia` is written `ia` (README: normalised throughout the
    lexicon); the `-ain` ending is written `-ain`. This is a stopgap, not
    diacritic restoration -- for real coverage, feed --text a document that
    was typed with the diacritics in place.
    """
    out = []
    if word.startswith("ia") and not word.startswith("ïa"):
        out.append("ï" + word[1:])
    if word.endswith("ain"):
        out.append(word[:-3] + "aiñ")
    if word.endswith("ain") and word.startswith("ia"):
        out.append("ï" + word[1:-3] + "aiñ")
    return [v for v in out if v != word]


# --------------------------------------------------------------------------
# main
# --------------------------------------------------------------------------


def main(argv=None):
    ap = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    src = ap.add_argument_group("vocabulary sources (combinable)")
    src.add_argument("--corpus-freq", nargs="?", const="data/corpus_freq.json",
                     help="sweep corpus types (default path "
                          "data/corpus_freq.json). NOTE: the corpus holds no "
                          "diacritics, so this alone cannot reach the "
                          "diacritic-bearing vocabulary")
    src.add_argument("--text", action="append", default=[],
                     help="raw Khasi text file; tokenised. The only source "
                          "that reaches real diacritic usage. Repeatable")
    src.add_argument("--wordlist", action="append", default=[],
                     help="file with one token per line. Repeatable")
    src.add_argument("--lexicon", action="store_true",
                     help="sweep every lexicon surface form; any rejection "
                          "here is a serious bug")
    src.add_argument("--probe", action="store_true",
                     help="generate prefix+compound forms and test the gate "
                          "directly, rather than waiting for someone to "
                          "write one. Results are questions, not bugs")
    src.add_argument("--words", nargs="*", default=None,
                     help="check exactly these words and stop (smoke test)")

    ap.add_argument("--min-freq", type=int, default=2,
                    help="ignore corpus types rarer than this (default 2)")
    ap.add_argument("--limit", type=int, default=0,
                    help="check only the N most frequent types (0 = all)")
    ap.add_argument("--no-variants", action="store_true",
                    help="skip the rule-based diacritic variants")
    ap.add_argument("--min-part", type=int, default=3,
                    help="shortest allowed compound part (default 3; "
                         "generate.py uses 5, which cannot see 'trei')")
    ap.add_argument("--max-depth", type=int, default=3)
    ap.add_argument("--out", default="silent_failures.csv")
    args = ap.parse_args(argv)

    if not any([args.corpus_freq, args.text, args.wordlist,
                args.lexicon, args.probe, args.words]):
        args.corpus_freq = "data/corpus_freq.json"

    print("loading engine ...", flush=True)
    t0 = time.time()
    eng = Engine()
    print("  ready in %.1fs" % (time.time() - t0))
    print("  API bound to: %s" % eng.report_binding())

    prefixes = DEFAULT_PRODUCTIVE
    try:
        decl = eng.db.morphology["prefixes"]
        found = [
            k for k, v in decl.items()
            if isinstance(v, dict) and v.get("productive")
        ]
        if found:
            prefixes = found
    except Exception:
        pass
    print("  productive prefixes: %s" % ", ".join(prefixes))

    dec = Decomposer(eng, prefixes, args.min_part, args.max_depth)

    # -- assemble the vocabulary from every requested source ---------------

    merged, origin = {}, {}

    def add(items, tag):
        n = 0
        for tok, freq in items:
            tok = tok.strip()
            if not tok:
                continue
            if tok not in merged or freq > merged[tok]:
                merged[tok] = freq
            origin.setdefault(tok, tag)
            n += 1
        print("    %-22s %7d types" % (tag, n))

    print("  sources:")
    if args.words:
        add([(w, 0) for w in args.words], "words")
    else:
        if args.corpus_freq:
            add(load_corpus_freq(args.corpus_freq, args.min_freq), "corpus")
        for p in args.text:
            add(load_text(p), "text:%s" % p.split("/")[-1])
        for p in args.wordlist:
            add(load_wordlist(p), "wordlist:%s" % p.split("/")[-1])
        if args.lexicon:
            add(load_lexicon_forms(eng), "lexicon")
        if args.probe:
            add(build_probe_set(eng, dec, prefixes), "probe")

    vocab = sorted(merged.items(), key=lambda kv: (-kv[1], kv[0]))
    if args.limit:
        vocab = vocab[: args.limit]

    if not args.no_variants:
        extra = []
        seen = set(merged)
        for tok, _ in vocab:
            for v in variant_spellings(tok):
                if v not in seen:
                    seen.add(v)
                    origin[v] = "variant-of:%s" % origin.get(tok, "?")
                    extra.append((v, merged.get(tok, 0)))
        if extra:
            print("    %-22s %7d types" % ("diacritic variants", len(extra)))
            vocab = vocab + extra

    print("  vocabulary: %d types total\n" % len(vocab))

    rows = []
    tally = Counter()
    by_shape = defaultdict(list)
    t0 = time.time()

    for i, (word, freq) in enumerate(vocab, 1):
        if i % 500 == 0:
            rate = i / max(time.time() - t0, 1e-9)
            print("  %d/%d  (%.0f/s)" % (i, len(vocab), rate), flush=True)

        try:
            r = eng.check(word)
        except Exception as exc:                      # keep the sweep going
            tally["error"] += 1
            rows.append({
                "token": word, "freq": freq, "source": origin.get(word, ""),
                "bucket": "error",
                "gate": "", "score": "", "n_suggestions": "",
                "shape": "", "parse": "", "leaves_known": "",
                "note": repr(exc)[:200],
            })
            continue

        if r["is_correct"]:
            tally["accepted"] += 1
            continue

        n_sug = len(r["suggestions"])
        parse = dec.analyse(word)
        shape = Decomposer.shape(parse)

        if n_sug == 0 and parse is not None:
            bucket = "SILENT + ANALYSABLE"      # the bug class
        elif n_sug == 0:
            bucket = "silent, unparsed"         # genuinely unknown
        elif parse is not None:
            bucket = "flagged but analysable"   # likely false alarm
        else:
            bucket = "flagged, unparsed"        # probably a real error

        tally[bucket] += 1
        if bucket == "SILENT + ANALYSABLE":
            by_shape[shape].append((word, freq))

        rows.append({
            "token": word,
            "freq": freq,
            "source": origin.get(word, ""),
            "bucket": bucket,
            "gate": r["gate"],
            "score": r["score"],
            "n_suggestions": n_sug,
            "shape": shape,
            "parse": Decomposer.render(parse),
            "leaves_known": sum(
                1 for l in Decomposer.leaves(parse) if eng.is_known(l)
            ),
            "note": "",
        })

    # -- output ------------------------------------------------------------

    rows.sort(key=lambda r: (r["bucket"] != "SILENT + ANALYSABLE",
                             -(r["freq"] or 0)))
    with open(args.out, "w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()) if rows else
                           ["token"])
        w.writeheader()
        w.writerows(rows)

    checked = len(vocab)
    print("\n" + "=" * 64)
    print("SWEEP: %d types checked" % checked)
    print("=" * 64)
    for k, n in tally.most_common():
        print("  %-24s %6d   %5.2f%%" % (k, n, 100.0 * n / max(checked, 1)))

    silent = tally["SILENT + ANALYSABLE"]
    if silent:
        print("\n" + "-" * 64)
        print("THE BUG CLASS: rejected, zero suggestions, but analysable")
        print("-" * 64)
        for shape, items in sorted(
            by_shape.items(), key=lambda kv: -sum(f for _, f in kv[1])
        ):
            occ = sum(f for _, f in items)
            print("\n  %s  --  %d types, %d corpus occurrences"
                  % (shape, len(items), occ))
            for word, freq in sorted(items, key=lambda kv: -kv[1])[:8]:
                print("      %-24s %6d   %s"
                      % (word, freq, Decomposer.render(dec.analyse(word))))
        print("\n  Fix the shape with the most occurrences first.")

    print("\nwrote %s (%d rows)" % (args.out, len(rows)))
    print("""
Caveats, in order of how much they should worry you:

 1. COVERAGE. A corpus sweep finds only words someone wrote in the news
    corpus, which holds no diacritics at all -- so it cannot reach roughly
    one headword in eleven. --text on a document typed WITH diacritics is
    the only honest source for that half; --probe generates the shape
    directly but its output is questions for a Khasi speaker, not bugs.
 2. The decomposer here is deliberately a fourth implementation, separate
    from complex.py, _check_hyphenated and generate.py. It measures the gap;
    it is not the fix. Its min-part is %d, where generate.py uses 5.
 3. Confirm the API binding printed at the top before trusting any count.""" %
          args.min_part)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""
cli.py — command-line interface for the Khasi spellchecker.

    python -m khasi_spell check "katba u bynriew u dand iaid hok, im hok katkum ka ain u Blei, u suk-u-sain, u roi-u-par bad u man-bha man-miay"
    python -m khasi_spell word lyngdo
    python -m khasi_spell fix   "u dand iaid hok"
    python -m khasi_spell file document.txt
    python -m khasi_spell info

Add --json to any command for machine-readable output.

The lexicon takes ~15-20 s to load, so this is a one-shot tool. For
repeated checking, import `KhasiSpeller` and keep the instance, or run
the HTTP service in `khasi_spell.api`.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from khasi_spell import context as _ctx
from khasi_spell.speller import KhasiSpeller


# ANSI colour, suppressed when output is piped.
def _c(code: str, text: str) -> str:
    if not sys.stdout.isatty():
        return text
    return f"\033[{code}m{text}\033[0m"


RED = lambda s: _c("31", s)
GREEN = lambda s: _c("32", s)
YELLOW = lambda s: _c("33", s)
DIM = lambda s: _c("2", s)
BOLD = lambda s: _c("1", s)


def _speller(args) -> KhasiSpeller:
    return KhasiSpeller(
        db_path=args.db,
        use_embeddings=args.embeddings,
        eager=True,
    )


# ----------------------------------------------------------------------
# Commands
# ----------------------------------------------------------------------

def _fmt_variants(variants) -> str:
    """Render alternatives, marking the spelling the lexicon records."""
    parts = []
    for v in variants:
        tag = DIM(" (in lexicon)") if v.get("canonical") else ""
        parts.append(GREEN(v["variant"]) + tag)
    return ", ".join(parts)


def cmd_word(args) -> int:
    sp = _speller(args)
    result = sp.check(args.word)

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0 if result.is_correct else 1

    if result.is_correct:
        why = "in lexicon" if result.in_lexicon else "morphologically valid"
        print(f"{GREEN('OK')}  {BOLD(result.word)}  {DIM('(' + why + ')')}")
        # Accepted, but the lexicon may spell it with diacritics. Not an
        # error — guidance, so it prints after the OK.
        if result.variants:
            print(f"  {DIM('also spelled:')} " + _fmt_variants(result.variants))
        return 0

    print(f"{RED('MISSPELLED')}  {BOLD(result.word)}  {DIM('(' + result.gate_name + ')')}")
    if result.suggestions:
        print("  suggestions:")
        for i, s in enumerate(result.suggestions, 1):
            marker = GREEN("->") if i == 1 else "  "
            print(f"    {marker} {s}")
    else:
        print(DIM("  no suggestions could be generated"))
    if result.variants:
        print(f"  {DIM('also spelled:')} " + _fmt_variants(result.variants))
    return 1


def cmd_check(args) -> int:
    sp = _speller(args)
    result = sp.check_text(args.text, context=not args.no_context,
                          skip_foreign=not args.check_names)

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0 if not result.has_errors else 1

    if not result.has_errors:
        print(GREEN("No spelling errors found."))
        _print_text_variants(result)
        _print_skipped(result)
        if getattr(args, "realword", False):
            try:
                flags = sp.check_realword(args.text, min_margin=getattr(args, 'margin', None))
                if flags:
                    _print_realword(flags)
                    return 1
            except FileNotFoundError as e:
                print(DIM(f"  (real-word check skipped: {e})"))
        return 0

    n = len(result.corrections)
    print(f"{YELLOW(str(n))} issue{'s' if n != 1 else ''} found:\n")
    for c in result.corrections:
        alts = ", ".join(c.suggestions[1:4])
        print(f"  {RED(c.original)} -> {GREEN(c.suggestion)}"
              f"   {DIM('[' + str(c.start) + ':' + str(c.end) + ']')}")
        if alts:
            print(DIM(f"      also: {alts}"))
    print(f"\n{BOLD('Corrected:')} {result.corrected}")
    _print_text_variants(result)
    _print_skipped(result)

    if getattr(args, "realword", False):
        try:
            _print_realword(sp.check_realword(args.text, min_margin=getattr(args, 'margin', None)))
        except FileNotFoundError as e:
            print(DIM(f"  (real-word check skipped: {e})"))
    return 1


def _print_skipped(result) -> None:
    """Proper nouns and acronyms withheld from correction. Not errors."""
    sk = getattr(result, "skipped", None)
    if not sk:
        return
    by_reason: dict[str, set] = {}
    for item in sk:
        by_reason.setdefault(item["reason"], set()).add(item["word"])
    # A recognised name is a stronger statement than a capitalisation guess,
    # so they are labelled separately rather than lumped together.
    labels = {"name": "known names", "capitalised": "assumed proper nouns",
              "acronym": "acronyms"}
    parts = [f"{labels.get(r, r)}: " + ", ".join(sorted(w))
             for r, w in sorted(by_reason.items())]
    print(DIM("\n  not checked — " + "; ".join(parts)))


def _print_text_variants(result) -> None:
    """Accepted words that have another attested spelling. Not errors."""
    if not getattr(result, "variants", None):
        return
    print(f"\n{DIM('alternative spellings (both accepted):')}")
    for v in result.variants:
        print(f"  {v.word} -> " + _fmt_variants(v.variants))


def _print_realword(flags) -> None:
    """Shared renderer — used by `realword` and by `check --realword`."""
    n = len(flags)
    if not n:
        print(DIM("\nNo wrong-word usage detected."))
        return
    print(f"\n{YELLOW(str(n))} possible wrong-word use{'' if n == 1 else 's'}:")
    for f in flags:
        where = f"[{f.start}:{f.end}]  margin {f.margin:.1f}"
        print(f"  {YELLOW(f.word)} -> {GREEN(f.suggestion)}   {DIM(where)}")
    print(DIM("  (contextual guesses, not spelling errors — check before accepting)"))


def cmd_realword(args) -> int:
    sp = _speller(args)
    try:
        flags = sp.check_realword(args.text, min_margin=getattr(args, 'margin', None))
    except FileNotFoundError as e:
        print(f"error: {e}", file=sys.stderr)
        return 2

    if args.json:
        print(json.dumps([f.to_dict() for f in flags], ensure_ascii=False, indent=2))
        return 0 if not flags else 1
    if not flags:
        print(GREEN("No wrong-word usage detected."))
        return 0
    _print_realword(flags)
    return 1


def cmd_fix(args) -> int:
    sp = _speller(args)
    print(sp.check_text(args.text, context=not args.no_context,
                          skip_foreign=not args.check_names).corrected)
    return 0


def cmd_file(args) -> int:
    path = Path(args.path)
    if not path.is_file():
        print(f"error: no such file: {path}", file=sys.stderr)
        return 2

    text = path.read_text(encoding="utf-8")
    sp = _speller(args)
    result = sp.check_text(text, context=not args.no_context,
                          skip_foreign=not args.check_names)

    if args.json:
        print(json.dumps(result.to_dict(), ensure_ascii=False, indent=2))
        return 0 if not result.has_errors else 1

    if args.fix:
        out = Path(args.output) if args.output else path.with_suffix(path.suffix + ".corrected")
        out.write_text(result.corrected, encoding="utf-8")
        print(f"{len(result.corrections)} correction(s) applied -> {out}")
        return 0

    # Report with line numbers, which is what makes file mode useful.
    line_starts = [0]
    for i, ch in enumerate(text):
        if ch == "\n":
            line_starts.append(i + 1)

    def locate(offset: int) -> tuple[int, int]:
        lo, hi = 0, len(line_starts) - 1
        while lo < hi:
            mid = (lo + hi + 1) // 2
            if line_starts[mid] <= offset:
                lo = mid
            else:
                hi = mid - 1
        return lo + 1, offset - line_starts[lo] + 1

    if not result.corrections:
        print(GREEN(f"No spelling errors found in {path}."))
        return 0

    for c in result.corrections:
        line, col = locate(c.start)
        print(f"{path}:{line}:{col}: {RED(c.original)} -> {GREEN(c.suggestion)}")
    print(f"\n{len(result.corrections)} issue(s).")
    return 1


def cmd_info(args) -> int:
    sp = _speller(args)
    stats = sp.analyser.db.stats()
    _lm_state = sp.language_model
    info = {
        "spell_vocabulary": stats.get("spell_checker_words"),
        "lexicon_entries": stats.get("total_entries"),
        "unique_surface_forms": stats.get("unique_surface_forms"),
        "unique_roots": stats.get("unique_roots"),
        "embeddings": sp.embeddings,
        # Both change results silently when missing, so always report them.
        "corpus_frequency": (
            {"applied": True,
             "distinct_values": sp.corpus_frequencies["distinct_values"],
             "corpus_tokens": sp.corpus_frequencies["corpus_tokens"]}
            if sp.corpus_frequencies else
            {"applied": False, "note": "run scripts/build_corpus_freq.py"}),
        "language_model": _lm_state,
        # Silently inert without the n-gram model, so report it explicitly.
        # Probes availability rather than calling _lm(), which would load
        # 170 MB merely to describe the configuration.
        "context_reranking": {
            "enabled": _lm_state["available"],
            "alpha": _ctx.ALPHA,
            "window": _ctx.CONTEXT_WINDOW,
        },
        "db_path": str(args.db) if args.db else "data/khasi_db.json (bundled)",
    }
    if args.json:
        print(json.dumps(info, ensure_ascii=False, indent=2))
        return 0
    print(BOLD("Khasi spellchecker"))
    for k, v in info.items():
        print(f"  {k.replace('_', ' '):<20} {v}")
    return 0


# ----------------------------------------------------------------------
# Entry point
# ----------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="khasi-spell",
        description="Morphology-gated spellchecker for the Khasi language.",
    )
    p.add_argument("--db", metavar="PATH", default=None,
                   help="path to khasi_db.json (default: bundled lexicon)")
    p.add_argument("--embeddings", action="store_true",
                   help="enable FastText re-ranking (bundled model; lowers top-1 — see README)")
    p.add_argument("--json", action="store_true", help="machine-readable output")

    sub = p.add_subparsers(dest="command", required=True)

    sp_word = sub.add_parser("word", help="check a single word")
    sp_word.add_argument("word")
    sp_word.set_defaults(func=cmd_word)

    sp_check = sub.add_parser("check", help="check a sentence or paragraph")
    sp_check.add_argument("text")
    sp_check.add_argument("--realword", action="store_true",
                          help="also look for correctly spelled but wrong words")
    sp_check.add_argument("--margin", type=float, default=None, metavar="X",
                     help="real-word sensitivity, log units; lower finds more "
                          "and is less certain (default 8.0)")
    sp_check.add_argument("--check-names", action="store_true",
                        help="also correct proper nouns and acronyms (noisy)")
    sp_check.add_argument("--no-context", action="store_true",
                        help="rank each word in isolation, ignoring its neighbours")
    sp_check.set_defaults(func=cmd_check)

    sp_rw = sub.add_parser("realword",
                           help="find correctly spelled words used in the wrong place")
    sp_rw.add_argument("text")
    sp_rw.add_argument("--margin", type=float, default=None, metavar="X",
                     help="real-word sensitivity, log units; lower finds more "
                          "and is less certain (default 8.0)")
    sp_rw.set_defaults(func=cmd_realword)

    sp_fix = sub.add_parser("fix", help="print the corrected text")
    sp_fix.add_argument("text")
    sp_fix.add_argument("--check-names", action="store_true",
                        help="also correct proper nouns and acronyms (noisy)")
    sp_fix.add_argument("--no-context", action="store_true",
                        help="rank each word in isolation, ignoring its neighbours")
    sp_fix.set_defaults(func=cmd_fix)

    sp_file = sub.add_parser("file", help="check a text file")
    sp_file.add_argument("path")
    sp_file.add_argument("--fix", action="store_true", help="write a corrected copy")
    sp_file.add_argument("--output", metavar="PATH", help="where to write it")
    sp_file.add_argument("--check-names", action="store_true",
                        help="also correct proper nouns and acronyms (noisy)")
    sp_file.add_argument("--no-context", action="store_true",
                        help="rank each word in isolation, ignoring its neighbours")
    sp_file.set_defaults(func=cmd_file)

    sp_info = sub.add_parser("info", help="show lexicon and configuration")
    sp_info.set_defaults(func=cmd_info)

    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())

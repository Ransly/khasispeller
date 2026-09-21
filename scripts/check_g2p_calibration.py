#!/usr/bin/env python3
"""
check_g2p_calibration.py — measure the DB's G2P against War (2001) pp.55-56.

The lexicon's phonology.ipa.surface values were produced by an internal
`g2p_v1`. This script compares that output to Badaplin War's own phonetic
transcriptions (tests/g2p_calibration_war.json) and reports:

  • overall agreement (exact, after light notation normalisation)
  • per-phenomenon accuracy, so systematic G2P errors are named
    (e.g. final -h → /ʔ/, final -it → /ic/, unmarked vowel length)
  • a per-word diff table

It changes nothing. Use it as a gold-standard regression check before and
after any G2P rework (roadmap item 10b / the phonology items 1-6).

Usage:
    python3 scripts/check_g2p_calibration.py            # human report
    python3 scripts/check_g2p_calibration.py --json     # machine summary
    python3 scripts/check_g2p_calibration.py --strict   # exit 1 if below threshold

Exit status is 0 unless --strict is given and agreement is below --min
(default 0.60), so CI can gate on regressions without failing on the known
current baseline.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FIXTURE = ROOT / "tests" / "g2p_calibration_war.json"
DB_PATH = ROOT / "data" / "khasi_db.json"


def _normalise(ipa: str) -> str:
    """Light normalisation so notation differences don't count as errors.

    Collapses aspirate digraph vs superscript (kh↔kʰ), strips delimiters and
    spaces. Deliberately does NOT normalise length, vowel quality, or final
    consonant place — those ARE the phenomena we want to measure.
    """
    if not ipa:
        return ""
    s = ipa.strip().strip("/|[]").replace(" ", "")
    for plain, sup in (("kh", "kʰ"), ("ph", "pʰ"), ("th", "tʰ"),
                       ("bh", "bʰ"), ("dh", "dʰ"), ("jh", "dzʰ")):
        s = s.replace(sup, plain)
    # unify the two central-vowel notations sometimes used for y
    s = s.replace("ɨ", "ɪ")
    return s


def _load_db_ipa() -> dict:
    raw = json.loads(DB_PATH.read_text(encoding="utf-8"))
    out: dict[str, str] = {}
    for e in raw.get("lexicon", []):
        if not isinstance(e, dict):
            continue
        sf = (e.get("form", {}) or {}).get("surface") or e.get("surface_form") or ""
        ipa = ((e.get("phonology", {}) or {}).get("ipa") or {}).get("surface")
        if sf and ipa and sf.lower() not in out:
            out[sf.lower()] = ipa
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--json", action="store_true", help="emit machine summary")
    ap.add_argument("--strict", action="store_true", help="exit 1 if below --min")
    ap.add_argument("--min", type=float, default=0.60, help="strict threshold")
    ap.add_argument("--live", action="store_true",
                    help="test khasi_engine.g2p.to_ipa directly instead of "
                         "the stored DB IPA (measures the converter, not the "
                         "regenerated data)")
    args = ap.parse_args()

    fixture = json.loads(FIXTURE.read_text(encoding="utf-8"))
    entries = fixture["entries"]
    if args.live:
        import sys as _sys
        _sys.path.insert(0, str(ROOT))
        from khasi_engine import g2p as _g2p
        db_ipa = {e["word"].lower(): _g2p.to_ipa(e["word"]) for e in entries}
    else:
        db_ipa = _load_db_ipa()

    rows = []
    n_match = n_present = 0
    phen_total: dict[str, int] = {}
    phen_ok: dict[str, int] = {}

    for ent in entries:
        w = ent["word"].lower()
        gold = ent["war_ipa"]
        got = db_ipa.get(w)
        present = got is not None
        ok = present and _normalise(got) == _normalise(gold)
        if present:
            n_present += 1
        if ok:
            n_match += 1
        for ph in ent.get("phenomena", []):
            phen_total[ph] = phen_total.get(ph, 0) + 1
            if ok:
                phen_ok[ph] = phen_ok.get(ph, 0) + 1
        rows.append((ent["word"], gold, got or "—", ok, present, ent.get("phenomena", [])))

    total = len(entries)
    agree = n_match / total if total else 0.0

    summary = {
        "source": fixture.get("source"),
        "total": total,
        "present_in_db": n_present,
        "exact_match": n_match,
        "agreement": round(agree, 3),
        "by_phenomenon": {
            ph: {"ok": phen_ok.get(ph, 0), "total": phen_total[ph],
                 "accuracy": round(phen_ok.get(ph, 0) / phen_total[ph], 2)}
            for ph in sorted(phen_total)
        },
    }

    if args.json:
        print(json.dumps(summary, ensure_ascii=False, indent=2))
    else:
        print(f"G2P calibration vs {fixture.get('source')}")
        print(f"  words:        {total}")
        print(f"  present in DB:{n_present}")
        print(f"  exact match:  {n_match}  ({agree:.0%})")
        print()
        print("  per-phenomenon accuracy (lowest = biggest systematic gap):")
        for ph, st in sorted(summary["by_phenomenon"].items(), key=lambda kv: kv[1]["accuracy"]):
            print(f"    {ph:16s} {st['ok']:>2}/{st['total']:<2} {st['accuracy']:.0%}")
        print()
        print(f"  {'word':12s} {'War gold':14s} {'DB g2p_v1':14s} match")
        print("  " + "-" * 52)
        for word, gold, got, ok, present, _ in rows:
            mark = "✓" if ok else ("·" if present else "✗ (missing)")
            print(f"  {word:12s} {gold:14s} {got:14s} {mark}")

    if args.strict and agree < args.min:
        print(f"\nFAIL: agreement {agree:.0%} < threshold {args.min:.0%}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""
extend_initial_clusters.py — add attested onsets to valid_initial_clusters.

`phonology.valid_initial_clusters` holds 76 clusters and is incomplete, so
**406 single-token lexicon surfaces (3.3%) fail validation for being correct
Khasi**. `khwai` and `khwaiñ` are curated entries; `s'ïang`, `l'er`, `k'a`
carry the glottal stop, which the DB lists among the Khasi consonants. The
gap has real downstream effect: `validate()` drives `_phonotactic_ok`, so
those words are kept out of `_compound_known`, and it drives `_offerable`,
where only the `is_attested` escape hatch rescues them.

Evidence, two kinds
-------------------
**Corpus attestation.** The corpus is modern typed text, independent of the
printed dictionary's OCR, so a cluster beginning corpus words the checker
accepts as Khasi is externally confirmed — `rwai`, `lwait`, `khwai`,
`lngaid`, `dngong`, `thwet`, `phngit`.

**Structural, where the corpus cannot testify.** The corpus contains **zero
apostrophes and zero ñ** in 6.67M tokens — both were stripped or never
typed. Silence there is not evidence of absence, so clusters containing `'`
or `ñ` are judged on the DB's own consonant inventory, which lists both, plus
the number of distinct lexicon entries using them.

Deliberately excluded
---------------------
* **digraphs** (`sh`, `kh`, `th`, `ph`, `ng`) — single phonemes, not
  clusters. They surface in the failure list only because the words carrying
  them (`shabdr`, `thaldb`) fail for a different reason: a vowel misread as
  `d`. Adding them would paper over that damage.
* **markup artefacts** (`*d`, `*kh`, `*l`) — from `*dorbin`, `*khate`.
* **onsets with no evidence at all** — `jk`, `lm`, `pb`, `shm`, `jb`, `mt`,
  `tk`, `jd`: no accepted corpus word begins with them and they contain
  neither `'` nor `ñ`. Several appear in entries that look like scan damage
  (`pbet-lyndet`, `tjmja`), so they are left for the review list rather than
  legitimised here.

    python3 scripts/extend_initial_clusters.py --dry-run
    python3 scripts/extend_initial_clusters.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "khasi_db.json"

# Corpus-attested: at least one corpus word beginning with this cluster is
# accepted as Khasi by the checker. (lexicon entries, accepted corpus types)
CORPUS_ATTESTED = {
    "phn": (6, 4), "rw": (6, 4), "lw": (13, 3), "jw": (6, 3), "phng": (5, 3),
    "khw": (9, 2), "lng": (7, 2), "dng": (5, 2), "rt": (5, 2), "thw": (4, 2),
    "tw": (2, 2), "rm": (3, 1), "dr": (2, 1), "shp": (2, 1),
}

# Structural: contains a consonant the corpus cannot testify about.
STRUCTURAL = {
    "s'": 17, "l'": 10, "k'": 7, "sh'": 5, "b'": 4, "p'": 4, "r'": 3,
    "khñ": 5, "kñ": 3,
}


def main() -> int:
    ap = argparse.ArgumentParser(description="extend valid_initial_clusters")
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--db", type=Path, default=DB)
    args = ap.parse_args()

    data = json.loads(args.db.read_text(encoding="utf-8"))
    phon = data["phonology"]
    current = list(phon.get("valid_initial_clusters") or [])
    proposed = sorted(set(CORPUS_ATTESTED) | set(STRUCTURAL))
    new = [c for c in proposed if c not in current]

    print(f"  valid_initial_clusters: {len(current)} -> {len(current) + len(new)}")
    print(f"  adding {len(new)}:")
    for c in new:
        if c in CORPUS_ATTESTED:
            lex, cor = CORPUS_ATTESTED[c]
            print(f"    {c:<6} corpus-attested  ({lex} lexicon entries, "
                  f"{cor} accepted corpus types)")
        else:
            print(f"    {c:<6} structural       ({STRUCTURAL[c]} lexicon entries; "
                  f"corpus has no {'apostrophe' if chr(39) in c else 'ñ'})")

    if args.dry_run:
        print("\n  --dry-run: nothing written")
        return 0

    backup = args.db.with_name(
        f"{args.db.name}.bak.{datetime.now():%Y%m%d_%H%M%S}")
    shutil.copy2(args.db, backup)
    print(f"\n  backup -> {backup.name}")

    phon["valid_initial_clusters"] = sorted(set(current) | set(new))
    phon["valid_initial_clusters_note"] = (
        "Extended " + datetime.now().strftime("%Y-%m-%d") + " by "
        "scripts/extend_initial_clusters.py. The original 76 rejected 406 "
        "single-token surfaces (3.3%) that are correct Khasi. Additions are "
        "either corpus-attested (a corpus word beginning with the cluster is "
        "accepted as Khasi) or structural for clusters containing ' or ñ, "
        "which the corpus cannot testify about — it holds zero of either.")
    args.db.write_text(json.dumps(data, ensure_ascii=False, indent=1),
                       encoding="utf-8")
    print(f"  written: {len(phon['valid_initial_clusters'])} clusters")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

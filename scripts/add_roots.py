#!/usr/bin/env python3
"""
add_roots.py — add `tikna` and `kerkut` as lexicon headwords.

Why these two, and why by hand
------------------------------
The `khasi_test_pairs_v2` evaluation of 6 September 2026 found that 64% of a
23,254-word target vocabulary sits outside the candidate pool, and that the
morphological generator recovers only the part it can rebuild from roots the
lexicon already holds. `jingtikna` and `jingkerkut` were the two cases that
motivated the generator and that it could NOT fix, for the same reason in both
cases: the root is missing, so there is nothing to build from.

    check("jingtikna")   -> rejected, {morphology 1, phonotactic 1} = 2
    check("jingkerkut")  -> rejected, {morphology 1, phonotactic 1} = 2

The vote awards an affixed parse 2 points only when the root is itself a known
word; otherwise 1, which falls a point short of the threshold. Adding the root
is therefore the whole fix — both derived forms start being accepted, and the
generator can reach them, without either of them being listed.

Evidence for each gloss
-----------------------
Neither gloss is invented; both are read off corpus usage. Both entries were
confirmed by a Khasi speaker on 6 September 2026 and are therefore written
with `needs_review: false` and `origin: speaker_confirmed`.

`tikna` — 436 corpus occurrences, no lexicon entry at all. Adjectival/stative,
used under `ba`/`kaba`:

    "ki kot ki sla ba tikna ban pynshisha"
    "ka por kaba tikna (time bound)"        <- the writer's own inline gloss
    "ka kynhun ba tikna ban lam khmat"

`pyntikna` is attested too, which is why the entry is tagged VERB rather than
ADJ: `pyn-` selects verbs (`morphology.prefixes.pyn.applies_to`), so an ADJ
tag would license `jingtikna` but not `pyntikna`. Khasi statives sit in the
verb class throughout this lexicon.

`kerkut` — 38 corpus occurrences. This one is a *spelling* gap rather than a
vocabulary gap: the lexicon already holds `ker-kut` (kh_DB_024962, "To
surround completely") and `ker kut` (kh_DB_014704, "Beleaguer; Besiege"), but
not the unhyphenated spelling writers actually use:

    "kaba la jan shah kerkut ha ki shnong Nepali"
    "ki pulit kiba kerkut ia ka jaka"
    "ka daw jong ka jingkerkut ki shipai Crusaders ha Damietta"

The lexicon already carries the same word under two spellings, so a third is
consistent with existing practice rather than a new modelling decision. The
gloss is inherited from `ker-kut` and the entry is linked back to it.

    python3 scripts/add_roots.py --dry-run
    python3 scripts/add_roots.py
"""
from __future__ import annotations

import argparse
import json
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "khasi_db.json"


def entry(surface: str, root_id: str, gloss: list[str], morph_type: str,
          structure: list[str], canonical: str, syllables: list[str],
          pattern: str, ipa: str, related: list[str], variants: list[str],
          note: str, entry_id: str) -> dict:
    return {
        "entry_id": entry_id,
        "form": {"surface": surface, "normalized": surface, "variants": []},
        "lemma": {"root": surface, "root_id": root_id},
        "grammar": {"pos": "VERB", "gender": None, "clitic": None},
        "morphology": {
            "type": morph_type,
            "structure": [{"type": "root", "form": {"raw": r}} for r in structure],
            "canonical": canonical,
            "features": {"reduplication": None, "is_tien_kynnoh": False},
        },
        "phonology": {
            "surface": surface,
            "derived": {"syllables": syllables, "syllable_count": len(syllables),
                        "pattern": pattern, "diphthong_nucleus": None,
                        "has_long_vowel": False},
            "flags": {"exception": False, "needs_review": False,
                      "has_diphthong": False, "has_triphthong": False,
                      "has_aspirate_onset": False, "has_glottal_final": False},
            "ipa": {"surface": ipa, "generated_by": "manual_v1",
                    "needs_review": False},
        },
        "semantics": {
            "gloss": gloss,
            "domain": "action",
            "semantic_features": {"primary": "action", "secondary": ["process"],
                                  "confidence": 0.4,
                                  "ontology_path": ["event", "action"]},
            "domain_source": "corpus_context_v1",
            "gloss_normalized": "; ".join(gloss),
            "ontology_review": {"status": "accepted", "reason": "speaker_confirmed"},
        },
        "lexical_relations": {"related": related, "variants": variants,
                              "better_forms": []},
        "source": {"origin": "speaker_confirmed", "source_file": "rupang_data",
                   "loan_origin": None, "loan_type": None,
                   "previous_entry_id": None, "note": note},
        "validation": {"is_valid": True, "confidence": 1.0, "needs_review": False, "review_reasons": []},
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    data = json.loads(DB.read_text(encoding="utf-8"))
    existing = {(e.get("form") or {}).get("surface") for e in data["lexicon"]}
    existing_ids = {e.get("entry_id") for e in data["lexicon"]}

    # NOT meta.next_id_sequence — it was stale by 1,800 on 6 September 2026
    # (claimed 28380 against a real maximum of 30182) and assigning from it
    # produced two entries whose ids collided with the existing `tnan` and
    # `tneu`. Derive the next id from the data instead, and repair the counter.
    import re as _re
    used = [int(m.group(1)) for e in data["lexicon"]
            if (m := _re.fullmatch(r"kh_DB_(\d+)", e.get("entry_id") or ""))]
    seq = max(used) + 1
    stale = int(data["meta"].get("next_id_sequence", 0))
    if stale <= max(used):
        print(f"  note: meta.next_id_sequence was {stale}, real max is "
              f"{max(used)} — assigning from {seq} and correcting the counter")
    new = [
        entry(
            surface="tikna", root_id="root_tikna",
            gloss=["To be definite", "To be fixed or settled",
                   "To be certain", "Time-bound"],
            morph_type="root", structure=["tikna"], canonical="tikna",
            syllables=["tik", "na"], pattern="CVC.CV", ipa="/tikna/",
            related=[], variants=[],
            note=("436 corpus occurrences; no dictionary entry. Gloss read from "
                  "corpus usage, including a writer's inline English gloss "
                  "'ka por kaba tikna (time bound)'. Tagged VERB so that pyn- "
                  "(which selects verbs) licenses the attested pyntikna."),
            entry_id=f"kh_DB_{seq:06d}",
        ),
        entry(
            surface="kerkut", root_id="root_kerkut",
            gloss=["To surround completely", "To besiege", "To blockade",
                   "To erect a fort or stockade"],
            morph_type="compound", structure=["ker", "kut"], canonical="ker+kut",
            syllables=["ker", "kut"], pattern="CVC.CVC", ipa="/kɛːr.kut/",
            related=["kh_DB_024962", "kh_DB_014704", "kh_DB_024961"],
            variants=["ker-kut", "ker kut"],
            note=("38 corpus occurrences of the unhyphenated spelling. The "
                  "lexicon already holds ker-kut (kh_DB_024962) and ker kut "
                  "(kh_DB_014704) with the same sense; this adds the spelling "
                  "writers use. Gloss inherited from those entries."),
            entry_id=f"kh_DB_{seq + 1:06d}",
        ),
    ]

    to_add = [e for e in new if e["form"]["surface"] not in existing]
    for e in to_add:
        assert e["entry_id"] not in existing_ids, f"id collision: {e['entry_id']}"
    for e in new:
        if e["form"]["surface"] in existing:
            print(f"  skip {e['form']['surface']!r} — already a surface form")
    for e in to_add:
        print(f"  add  {e['form']['surface']!r} as {e['entry_id']} "
              f"({'; '.join(e['semantics']['gloss'])})")

    if not to_add:
        print("nothing to do")
        return 0
    if args.dry_run:
        print("dry run — nothing written")
        return 0

    backup = DB.with_suffix(f".json.bak.addroots_{int(time.time())}")
    shutil.copy2(DB, backup)
    print(f"  backup -> {backup.name}")

    data["lexicon"].extend(to_add)
    data["meta"]["next_id_sequence"] = seq + len(to_add)
    data["meta"].setdefault("changelog", []).append(
        f"v3.12: added roots {', '.join(e['form']['surface'] for e in to_add)} "
        f"— corpus-attested, glosses corpus-derived, speaker-confirmed. "
        f"Unblocks jingtikna/pyntikna and jingkerkut, which the confidence vote "
        f"was scoring 2 (affixed parse, unknown root) instead of 3.")
    DB.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"  wrote {DB} — lexicon now {len(data['lexicon'])} entries")
    print("\nNOTE: production reads PostgreSQL. Re-migrate after this edit.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

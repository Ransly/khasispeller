#!/usr/bin/env python3
"""
WITHDRAWN — do not run. Kept as a record of a decision that was reversed.

What it did
-----------
On 2026-09-10 this pass rewrote 11,348 stored CV patterns on an ORTHOGRAPHIC
basis — one symbol per letter, with `ng` split into two consonants
(jing -> CVCC) and a coda `w` counted as a consonant (tiew -> CVVC). Only
the aspirate digraphs were collapsed.

Why it was wrong
----------------
C/V classification is phonological, not character-based. A digraph is a fact
about spelling; `sh` is a single phoneme /ʃ/ and so is `ng` /ŋ/ — the
lexicon's own `phonology.consonant_chart` lists both, in an inventory it
totals at 27 consonants. Two letters is not two consonants.

The stored IPA agrees all the way down:

    jing  /dzɪŋ/   CVC     one velar nasal, not /n/+/g/
    sngi  /sŋi/    CCV
    ksew  /ksɛu/   CCVV    coda `w` is realised /u/, the offglide of a
    kaw   /kau/    CVV     diphthong — part of the nucleus, not a coda C
    tiew  /teu/    CVV
    wan   /wan/    CVC     but `w` IS /w/ in ONSET

So the rule for `w` is positional, not a property of the letter, and `ng`
never splits. The pre-existing data already encoded exactly this.

What happened instead
---------------------
The lexicon was restored from `khasi_db.json.bak.20260910_101005`, which
held the phonemic reading, and re-migrated. The `ï` -> V correction made
earlier that day is separate, is NOT part of this reversal, and stands.

Nothing depended on the values either way: the CV pattern is display-only.
It is read by `khasi_spell/api.py` for the Word Details panel and by
`KhasiDB` moving it between the enriched and flat shapes. `phonology.
validate()` — the phonotactic gate — never touches it, and there are zero
reads in spell_checker, generate, complex, assimilation, morphology or
analyser. No word is accepted, rejected, corrected or ranked on it.

Before writing any future pass over these patterns, settle two questions
that this one skated over:

  * is `ï` /ɨ/ or /j/? The stored IPA contradicts itself in the SAME
    environment — ïaroh /jarɔʔ/ against ïap /iap/ — so the g2p cannot
    arbitrate it. `phonology.vowels_simple` lists `ï`, and the maintainer
    ruled it a vowel.
  * does a diphthong occupy one V slot or two? The data says two
    (baid /baic/ -> CVVC, ksew -> CCVV).

A correct pass would derive the pattern from `phonology.ipa` rather than
from the spelling, which is also what makes it self-correcting as the g2p
improves.
"""
print(__doc__)
print("This script is withdrawn and will not run.")
raise SystemExit(2)

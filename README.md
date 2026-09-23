# Khasi Spellchecker

A standalone, morphology-gated spellchecker for **Khasi** (Austroasiatic ·
Khasian), extracted from the MorpSpeller / Khasi NLP Explorer platform.

```python
from khasi_spell import KhasiSpeller

sp = KhasiSpeller()
sp.is_correct("nongsngew")              # True — derived, in no word list
sp.suggest("lyngdo")                    # ['lyngdoh', 'lyngdop', 'lyngtong', ...]
sp.correct_text("Ka shnng ka bha")      # 'Ka shnong ka bha'
```

---

## Why this is not an ordinary spellchecker

Khasi is prefix-dominant and productively derivational. `jing-` nominalises,
`pyn-` causativises, `nong-` forms agentives, and they stack:

```
bha  'good'  →  jingbha   'goodness'      (jing-)
             →  pynbha    'to make good'  (pyn-)
im   'live'  →  jingpynim                 (jing- + pyn-)
```

A finite word list therefore **cannot** enumerate the valid words of Khasi.
Prefixation also triggers assimilation, so the root is sometimes not even
present in the surface form: `pyn-` + `lait` → **pyllait**.

```python
sp.analyser.db.is_known("nongsngew")    # False — not a lexicon entry
sp.is_correct("nongsngew")              # True  — the gate parsed it
```

This checker inverts the usual design. Rather than treating a word list as
ground truth, it asks a **morphological analyser** whether a word is a legal
Khasi formation, and only words that fail that test reach edit-distance
correction. A newly coined but well-formed derivative is never flagged.

## How a word is decided

```
word
 │
 ├─ Gate 0  sub-phonemic token?         → reject outright
 ├─ Gate 1  phonotactically legal?      → if not, widen the search and flag
 ├─ Gate 2  confidence vote             → if it clears the threshold, accept
 │            lexicon hit          +3
 │            morphology w/ affix  +2
 │            phonotactically ok   +1
 │            frequent             +2      threshold: 3
 │
 └─ Gate 3  generate and rank candidates
              phoneme-level weighted Levenshtein (digraphs are one unit)
              40-pair Khasi confusion matrix (ng↔g, ie↔i, accents, voicing)
              5 orthographic normalisation rules
              Norvig-style generator over 304,958 expanded forms
              optional FastText semantic re-ranking
```

`check()` reports which gate a word reached, so you can tell *why* something
was or was not flagged.

## Worked example

```bash
$ khasi-spell check "katba u bynriew u dand iaid hok, im hok katkum ka ain u Blei, u suk-u-sain, u roi-u-par bad u man-bha man-miay"
```

```
2 issues found:

  dand -> dang       [18:22]
  man-miay -> man-miat   [102:110]

Corrected: katba u bynriew u dang iaid hok, im hok katkum ka ain u Blei,
           u suk-u-sain, u roi-u-par bad u man-bha man-miat

alternative spellings (both accepted):
  iaid -> ïaid (in lexicon)
  ain  -> aiñ  (in lexicon)
```

Three different mechanisms are at work, and each needed its own fix:

| | |
|---|---|
| `dand` → `dang` | Digraph tokenisation made `dang` 1.4 phonemes away while `and` — an English fragment inside a multi-word entry — sat at 1.0 and won. The **character-distance blend** sees the single keystroke. |
| `man-miay` → `man-miat` | Phase 4 accepts any hyphenated token as a compound *without checking its parts*, so the bad half was invisible. Parts are now verified. |
| `iaid` → `ïaid`, `ain` → `aiñ` | Not errors. Both spellings are valid; the lexicon records the diacritic form, so it is offered and never applied automatically. |

`sain` → `saiñ` is **not** offered: the lexicon records `sain` and not `saiñ`, so
there is no evidence for the suggestion. A lexicon gap, not a ranking failure.

---

## Running it

Requires **Python 3.9+**. The core library has **no third-party dependencies**,
so the quickest path is to run it in place — no install step at all:

```bash
cd khasi-spellchecker
python3 -m khasi_spell word lyngdo
python3 -m khasi_spell check "katba u bynriew u dand iaid hok, im hok katkum ka ain u Blei, u suk-u-sain, u roi-u-par bad u man-bha man-miay"
python3 -c "from khasi_spell import KhasiSpeller; print(KhasiSpeller().suggest('kmei'))"
```

That works from the project directory because `khasi_spell/`, `khasi_engine/`
and `autocorrect/` sit at the root.

### Installing (optional)

Installing buys you the `khasi-spell` command from any directory.

```bash
python3 -m pip install -U pip setuptools     # see the note below
python3 -m pip install -e .                  # library + CLI
python3 -m pip install -e ".[api]"           # + HTTP service
python3 -m pip install -e ".[dev]"           # + pytest
```

> **The toolchain upgrade is not optional.** This project uses PEP 621
> metadata in `pyproject.toml` and has no `setup.py`. With setuptools < 61 the
> install silently succeeds as `UNKNOWN-0.0.0` with no console script; with
> pip < 21.3 the editable install fails outright with a missing
> `build_editable` hook. Verified working on pip 26.0.1 / setuptools 82.0.1.
> Use a virtual environment if you would rather not upgrade a shared
> interpreter:
>
> ```bash
> python3 -m venv .venv && source .venv/bin/activate
> python3 -m pip install -U pip setuptools && python3 -m pip install -e ".[dev]"
> ```

## Command line

```bash
khasi-spell word lyngdo                      # one word
khasi-spell check "u dand iaid hok"          # sentence, with spans
khasi-spell fix   "u dand iaid hok"          # print corrected text
khasi-spell file document.txt                # file:line:col report
khasi-spell file document.txt --fix          # write a corrected copy
khasi-spell info                             # lexicon, corpus and model status
khasi-spell --json word kmei                 # machine-readable

khasi-spell realword "Ka sorkar ki la ..."   # wrong-word usage
khasi-spell check "..." --realword           # spelling AND word choice
khasi-spell realword "..." --margin 6        # more sensitive, less certain
```

Exit status is `0` when nothing is flagged and `1` when something is, so it
composes in scripts and CI.

## HTTP service

```bash
python3 -m pip install -r requirements-api.txt
python3 -m uvicorn khasi_spell.api:app --port 8000
```

Then open **<http://localhost:8000>** for the web interface, or `/docs` for the
generated API documentation.

### Web interface

A single self-contained HTML page served at `/` — no build step, no CDN, no
JavaScript dependencies, works offline. Four views, each linkable:

| Route | View |
|---|---|
| `/` or `/#!check` | Check text — results summary, per-word table, sidebar detail |
| `/#!word` | Single word — the confidence vote, signal by signal |
| `/#!learn` | Khasi orthography and word formation, from the lexicon's own inventories |
| `/#!about` | How the decision works, and what the checker cannot tell you |
| `/#!example` | Loads and checks the sample sentence — a shareable demo link |

Paste Khasi text, hit **Check**, and misspellings are underlined in place.
Errors, unrecognised words, alternative spellings and withheld proper nouns are
distinguished by colour and listed separately, because they mean different
things: an alternative spelling is *already valid* and is offered, never
applied. Click a suggestion to apply it; nothing is applied automatically.

The single-word view is the interesting one for a morphology-gated checker —
`nongsngew` comes back accepted with the vote shown: not in the lexicon, but
the analyser parsed it, so morphology 2 plus phonotactics 1 clears the
threshold of 3.

**On the "match" percentage.** `/word` and `/check` return, per suggestion, the
blended phoneme/Damerau distance the engine actually ranked on. The percentage
in the interface is derived from that distance *and* the rank position,
because candidates frequently tie on distance alone — every suggestion for
`dand` sits at 1.0, and their order comes from the frequency and morphology
tie-breakers. It is a presentation of the ranking, **not a calibrated
probability**, and the interface says so. Where no distance exists (the
hyphenated-compound path builds a replacement directly) the column shows `—`
rather than a made-up number.

| Route | Purpose |
|---|---|
| `GET  /health` | Readiness, vocabulary, corpus-frequency and language-model state |
| `POST /word` | Check one word; returns gate, method, suggestions |
| `POST /check` | Check text; returns spans and a corrected string |
| `POST /correct` | Corrected text only |
| `POST /realword` | Contextually wrong real words |
| `POST /batch` | Up to 1,000 words in one call |
| `POST /v1/spellcheck` | **Compatibility** — identical contract to the parent platform |
| `GET  /` | Web interface |
| `GET  /api` | This table, as JSON |

The lexicon loads at start-up, so no request pays the load cost.

## Library

```python
sp = KhasiSpeller(eager=True)     # load now instead of on first call

r = sp.check("lyngdo")
r.is_correct        # False
r.suggestions       # ['lyngdoh', ...]
r.gate_name         # 'unknown'  (0 invalid_standalone / 1 phonotactically_invalid
                    #             / 2 accepted / 3 unknown)
r.in_lexicon        # False
r.confidence        # {'score': 1, 'signals': {...}, 'threshold': 3}

t = sp.check_text("u dand iaid hok bad u man-bha man-miay")
t.has_errors                      # True
t.corrections                     # dand -> dang, man-miay -> man-miat
t.corrections[0].start, .end      # exact offsets into the input
t.corrected                       # 'u dang iaid hok bad u man-bha man-miat'
t.variants                        # iaid -> ïaid   (accepted, never auto-applied)

sp.add_word("Meghalaya")          # session-only; not persisted
sp.analyser                       # escape hatch to the full engine
```

`KhasiSpeller` loads ~80 MB of lexicon in about 16 seconds, so **build one and
keep it**. Construction is lazy unless you pass `eager=True`.

### Morphological generation — on by default

The checker accepts words the lexicon does not hold, because the confidence
vote gives a productive derivation on a known root 2 of the 3 points needed.
Candidate *generation*, though, used to search only `all_surface_forms()` —
the lexicon. The two disagreed, and the disagreement was measurable:

```python
sp.check("pynïarap")    # accepted: {morphology 2, phonotactic 1}
sp.suggest("pynarap")   # ['pynaram', 'pynrap', 'bynarap', 'pynarkhap', 'pynap']
```

The right answer is missing from a list of five, for a word the same process
would have accepted a line earlier. Measured against a 23,254-word target
vocabulary: 64% of the targets sat outside the candidate pool, and **half of
those were words the checker accepts when typed correctly**.

`khasi_engine/generate.py` closes part of that. Nothing is generated until a
token has already failed the gate; the existing delete index is reused to find
roots near the residue, so it adds **no persistent memory** (peak RSS 471 MB
against 472 MB with it off). Two routes:

- **prefix** — strip a productive prefix, find roots near the residue, re-apply
  the prefix through `assimilation.should_assimilate`
- **compound** — split at each position, keep splits where one half is an
  **exact** root of at least five characters and the other is within one edit

Only the six prefixes the lexicon marks `productive: true` are used, and every
root must be a glossed single-word headword. Generated candidates carry a
distance penalty (0.25 prefix, 0.4 compound) so a real headword at the same
distance always wins.

| | off | on |
|---|---|---|
| frozen benchmark | 0.840 / 0.670 / 0.806 / 0.729 | **identical, 0 of 294 items changed** |
| running text, 16,274 tokens | 522 flags (3.21%) | 530 flags (3.26%) |
| open vocabulary, unreachable gold | 0.000 top-1 | **0.031** top-1 |
| median latency per sentence | 32.5 ms | 40.3 ms |

> The frozen benchmark **cannot see this feature**. Its gold answers are in the
> candidate pool 100% of the time and the generator only emits forms that are
> not, so it is arithmetically impossible for it to change an answer there.
> That is a fact about the benchmark, not about the generator.

Disable with `KHASI_SPELL_MORPH_GEN=0`. The compound route alone can be turned
off with `KHASI_SPELL_COMPOUND_MODE=off`.

**The honest ceiling.** This recovers about 3% of the unreachable vocabulary,
because the rest cannot be rebuilt from roots the lexicon does not have. Two
roots added by hand — `tikna` and `kerkut`, via `scripts/add_roots.py` — fixed
four derived forms at once (`jingtikna`, `pyntikna`, `jingkerkut`,
`jingker-kut`) with no latency cost. Lexical coverage buys more than
algorithms here.

### Semantic re-ranking (FastText) — bundled, off by default

The FastText model ships with this project at `models/fasttext/`:
**cbow_100**, 27,848 vocab, 100-dim, character n-grams 3–6 over 2M buckets.
Unlike the parent platform — which could not hold it in memory on Render and
proxied to a Hugging Face Space — it runs locally here.

```bash
python3 -m pip install "gensim>=4,<5" "numpy<2"
python3 -c "from khasi_spell import KhasiSpeller; \
            print(KhasiSpeller(use_embeddings=True).suggest('lyngdo'))"
```

Costs ~12 s to load and ~930 MB resident, on top of the lexicon.

> **It is off by default because it measured worse.** On this project's
> 77-item probe, enabling re-ranking dropped top-1 accuracy from **53.2% to
> 49.4%**; top-5 was unchanged, since re-ranking reorders candidates rather
> than finding new ones. Concretely, `lyngdo` → `lyngdoh` became `lyngdo` →
> `lyngdop`, and `kmei` → `kmie` became `kmei` → `kmeng`.
>
> The likely cause is structural: FastText composes vectors from character
> n-grams, so orthographic neighbours score high whether or not they are the
> right correction (`lyngdop` scores 0.903 against `lyngdo`, above `lyngdoh`
> at 0.875). The feature partly re-measures edit distance while adding noise,
> and it is weighted 3.0 against distance's 4.0 — enough to flip decisions.
>
> This is the first measurement of the feature's contribution in either
> project. It answers RQ7 of the project proposal, and the answer is
> currently no.

Backends resolve in order — local model, then remote Space:

| Variable | Effect |
|---|---|
| `KHASI_W2V_MODEL` | Path to a different `.model` file |
| `KHASI_W2V_BACKEND` | `local` \| `remote` \| `auto` (default `auto`) |
| `KHASI_W2V_URL` | Remote Space, used when no local model is present |
| `KHASI_W2V_KEY` | Shared secret for the Space |
| `KHASI_SPELL_EMBEDDINGS` | `1` to enable re-ranking in the HTTP service |

There is also a diagnostic for inspecting what the model learned:

```python
from khasi_engine import w2v_client
w2v_client.similar("jingbha", topn=5)
# [('jingbhasbun', 0.90), ('jingmiat', 0.80), ('jingbhah', 0.79), ...]
```

**The model is gitignored.** At 763 MB the `.npy` is far above GitHub's 100 MB
per-file limit, so it cannot be committed directly — use Git LFS or distribute
it out of band. Everything else works without it; the embedding tests skip
themselves when it is absent.

## PostgreSQL

The lexicon can be served from PostgreSQL instead of the 67 MB JSON file. The
engine picks the source from **`DATABASE_URL`** alone: set, it uses PostgreSQL;
unset, it reads the JSON. Nothing else switches it.

```bash
createdb khasi_spell
export DATABASE_URL="postgresql:///khasi_spell?host=/var/run/postgresql&port=5433"
python3 scripts/migrate_json_to_pg.py      # idempotent; --dry-run first if you like
python3 scripts/build_db_extras.py         # the sidecar — see below
```

`migrate_json_to_pg.py` creates the tables if absent and upserts on
`entry_id`, so re-running after a JSON edit is safe and resumable. `--reset`
drops and recreates everything and is the only destructive option.

**Re-run both scripts after editing `khasi_db.json`.** The database does not
notice the file changing.

### The sidecar is not optional

PostgreSQL carries only `meta`, `phonology`, `morphology` and `lexicon`. Four
other top-level blocks live in `data/khasi_db_extras.json`, which
`build_db_extras.py` writes:

| Block | Size | Why it matters |
|---|---|---|
| `quarantine` | 834 records | the audit trail for every entry withdrawn during the OCR repair — what makes those withdrawals reversible |
| `grammar_schema` | 7 keys | |
| `production_readiness` | 9 keys | |
| `demo_examples` | UI samples | |

Without it the engine still starts and still checks spelling, but those blocks
are **silently absent**. Look for `merged sidecar extras` in the start-up log.

### Why your local run said "loading from JSON file"

Because `export DATABASE_URL=...` lives and dies with one shell. Run
`uvicorn` from a second terminal and the service loads the JSON lexicon
instead — same behaviour, different source, and easy to miss.

Two fixes. Either export it in the same shell:

```bash
export DATABASE_URL="postgresql:///khasi_spell?host=/var/run/postgresql&port=5433"
python3 -m uvicorn khasi_spell.api:app --port 8000
```

or put it in a **`.env`** beside `pyproject.toml`, which the service reads at
start-up (twelve lines of stdlib, no dependency). Anything already in the
environment wins, so an explicit export still overrides the file. `.env` is
gitignored.

The start-up log now names the source either way:

```
[khasi-spell] loaded .env: DATABASE_URL
[khasi-spell] lexicon source: PostgreSQL
[khasi-nlp] KhasiDB: loading from PostgreSQL (…)
[khasi-nlp] KhasiDB: merged sidecar extras (khasi_db_extras.json)
```

and when it falls back it says why, rather than leaving you to spot the
difference.

### Measured, JSON against PostgreSQL

Same 29,694 entries, same engine, same frozen benchmark:

| | JSON | PostgreSQL |
|---|---:|---:|
| precision / recall / F1 | 1.000 / 0.840 / 0.913 | **identical** |
| top-1 / top-5 / MRR | 0.670 / 0.806 / 0.729 | **identical** |
| running-text flag rate | 3.04% | **identical** |
| start-up | 22.1 s | **15.3 s** |
| peak RSS | 471 MB | **510 MB** |

PostgreSQL starts faster because it reads typed columns instead of parsing
67 MB of JSON, and costs ~39 MB more resident. That 510 MB matters on one
deployment target only: the source comments record a 512 MB free-tier OOM on
Render as a real event, and 510 MB is under that ceiling *before* the n-gram
model loads.

## Corpus-derived resources

Two resources are built from a Khasi corpus (200 news files, 13.35M raw
tokens — every line appears exactly twice, so **6.67M after deduplication**).

```bash
python3 scripts/build_corpus_freq.py ../rupang_data -o data/corpus_freq.json
python3 scripts/build_ngrams.py      ../rupang_data -o data/ngrams.json.gz
```

### Corpus frequencies — on by default

The engine's built-in "frequency" is not one: `database.py` assigns values by
a word's role in the lexicon, giving **11 distinct values across 34,224
forms**, unable to tell a common word from a rare one. Real counts replace it:

| | detect | top-1 | top-5 |
|---|---|---|---|
| Pseudo-frequency | 77.9% | 53.2% | 67.5% |
| **Corpus frequency** | **79.2%** | **54.5%** | **68.8%** |

It also fixed a case the checker had always got wrong: `shnng` now suggests
`shnong` (22,530 occurrences) rather than `shna` (2,971).

Disable with `KhasiSpeller(use_corpus_freq=False)`. Only 18% of lexicon forms
are attested in the corpus, and **59,359 corpus types are absent from the
lexicon** — vocabulary expansion is now measurable.

### Corpus words as candidates and as words — on by default

The lexicon derives from a 1973 dictionary; the corpus is modern news. Common
words the dictionary never recorded — `jylla` 'state' (24,252 occurrences),
`wanrah` 'bring' (4,604), `pyntreikam` (1,940), loans such as `elekshon` —
were both **unreachable** as corrections and **rejected** when typed.
`khasi_spell/corpus_pool.py` addresses both.

**Supply.** A corpus type seen at least 20 times enters the candidate pool if
it uses only Khasi letters, passes the phonotactic screen, is not quarantined,
does not fold onto a lexicon headword, is not one edit from a type 5× commoner
(the corpus's own misspellings), and is not a run-together spelling that
`splits.py` corrects (`jongki` → `jong ki`). It is admitted in the lexicon's
orthography: `iatreilang` enters as `ïatreilang`, and `-ain` becomes `-aiñ`
only where the lexicon's own `-aiñ` rule agrees — the name `hussain` and the
loans `risain`, `pilain` stay plain. 2,192 forms are admitted.

**Acceptance.** The vote's frequency signal reads a table built from the
lexicon, so no corpus word could earn it: `jylla` scored 1 of the 3 needed.
Admitted words — and the reduced spelling the corpus writes them in
(`iatreilang`) — now earn it. The reduced spelling is accepted with the
diacritic form offered as an alternative, exactly as for a lexicon word typed
without its `ï`. `is_known()` is unchanged.

Before this, the pool offered words the checker then rejected: `jyla` was
answered with `jylla`, and `jylla` was flagged once accepted.

Measured 23 Sept 2026, acceptance off against on, all else equal:

| | acceptance off | **on** |
|---|---|---|
| flagged on 200 clean corpus sentences | 143 (4.34%) | **105 (3.19%)** |
| sentences carrying a flag | 95 / 200 | **72 / 200** |
| rejected with no suggestion (invisible) | 41 (1.24%) | **10 (0.30%)** |
| proxy set: top suggestion the checker itself rejects | 27 of 517 | **1 of 517** |
| word benchmark (detect / top-1 / top-5) | 98.0 / 73.5 / 91.5 | 98.0 / 73.5 / 91.5 |
| sentence benchmark (located / top-1 with context) | 249 / 92.0% | 248 / 91.6% |

The word benchmark cannot move by construction — its targets are lexicon
words. The one sentence-benchmark item lost is `mynshwa` for `mynshuwa`:
the "error" is the majority spelling in real text (1,413 corpus
occurrences against 864), so the checker now accepts it. The item is left
valid and counted as a miss.

The proxy set is 600 unverified pairs mined from the corpus (a type seen once
or twice, one edit from a type seen 1,000+ times); see
`docs/IMPROVEMENT_ANALYSIS.md`.

Switches: `KhasiSpeller(corpus_pool_accept=False)` keeps supply without
acceptance; `use_corpus_pool=False` turns both off. **Caveat:** `KhasiDB` is
memoised per data source, so every speller in one process shares the pool —
a `use_corpus_pool=False` speller created after a pooled one still offers pool
words (pinned by a `WEAK` test). Compare pool on and off in separate
processes. The acceptance switch is per-checker and does not leak.

### Alternative spellings (ï and ñ)

Khasi writes **ï** and **ñ**, and both are awkward to type, so writers drop
them. **Neither spelling is treated as an error** — the lexicon itself records
107 words both ways (`khwain`/`khwaiñ`, `luin`/`luiñ`, `niang`/`ñiang khriat`),
so calling either one wrong would contradict the lexicon. Both are accepted,
and each offers the other as a choice.

```bash
$ khasi-spell word ain
OK  ain  (morphologically valid)
  also spelled: aiñ (in lexicon)

$ khasi-spell word aiñ
OK  aiñ  (in lexicon)
  also spelled: ain
```

```python
r = sp.check("jingiathuh")
r.is_correct   # True
r.variants     # [{'variant': 'jingïathuh', 'source': 'root', 'canonical': False}]
```

`canonical` marks the spelling the lexicon records exactly, so an interface can
show which is standard without forcing it — the web UI ticks it (✓).

Four routes find them: a folded-index lookup for whole words
(`iathuh` → `ïathuh`), the reverse direction for diacritic input
(`aiñ` → `ain`), root substitution for derived forms, since `jingïathuh` is
not a lexicon entry but `jing-` + `ïathuh`, and a **morphological rule** for
the reciprocal prefix.

#### The reciprocal prefix — corrected in the lexicon

The lexicon used to contradict itself here: 196 `ia-` entries against 148
`ïa-`, 68 `jingia-` against 24 `jingïa-`, only 26 words recorded both ways.
`scripts/fix_ia_diaeresis.py` settled it — word-initial `ia` is now written
`ïa` throughout, including after the outer prefixes `affix_order` licenses
(`jing-`, `pyn-`, `nong-`, `sngew-`) and for the standalone particle.

```bash
python3 scripts/fix_ia_diaeresis.py --dry-run
python3 scripts/fix_ia_diaeresis.py
```

Idempotent, backed up, and the previous spelling is kept in `form.variants`.
Result: **zero plain `ia` surfaces, zero duplicate surfaces**, and both
spellings still accepted — `iathuh` resolves and offers `ïathuh`.

The script does three things, because correcting the spelling exposed two
older problems:

| step | effect |
|---|---|
| Correct the spelling | 1,478 entries across surface, normalized, root, canonical and structure |
| Repair short roots | 4 entries recorded as prefix + a **one-letter root** (`ïat` = `ia` + `t`). The project had already ruled against exactly this — golden_corpus carries "short-stem reroot: ia+t rejected" — and had re-rooted the plain twins, but the diacritic entries were missed |
| Merge duplicates | 85 entries across two passes. The lexicon recorded these words twice, once per spelling; two entries for one surface makes the analyser report `derived` instead of `valid` |

Two engine changes were needed alongside it, **neither fixed upstream**:

- **`is_known()` now consults the folded index**, as `lookup()` always did.
  The two disagreed — `lookup()` resolved `iathuh` to the `ïathuh` entry
  while `is_known()` called the word unknown. Harmless while both spellings
  were entries; once the lexicon settled on one, every plain-spelled word was
  rejected.
- **`morphology.prefixes` gained the `ïa` key.** The parser strips `ïa-` from
  the corrected entries but the registry was still keyed `ia`, so it raised
  `KeyError: 'ïa'`. The plain key is kept for compatibility.

The rule in `variants.py` remains for user input — typing `iabeit` still
offers `ïabeit` — but it no longer has data to patch.

Guarded so it fires only whereGuarded so it fires only where `ia-` really is the prefix: the base must be at
least two characters, known to the lexicon, and not a bare digraph — otherwise
roots that merely begin with those letters (`iap` 'die', `iar`, `iang`) would
be rewritten into nonsense.

**Validation.** Across 12,342 single-word entries the rule fires 180 times.
Of those, **21 predict a spelling the lexicon already records independently**,
and the remaining 159 fill the gap. 95 `ia`-initial words are correctly left
alone. Cost is 2 µs per word.

> **Engine change.** `ñ → n` was added to `_FOLD_MAP` in `database.py`.
> Without it `ain` did not find `aiñ`, so one of Khasi's two diacritics was
> tolerated at lookup and the other was not. Checked before changing: folding
> ñ merges 108 keys, and every one is the same word the lexicon already
> records twice. No distinct words collide. **Not fixed upstream.**

> **The corpus cannot settle which spelling is standard.** It contains
> **zero** tokens with ï or ñ across 6.67M words — they were stripped in
> preparation or never typed. Corpus frequency would therefore always favour
> the plain form, which is evidence about typing habits, not orthography.
>
> That also caused a bug: taken literally, zero counts demoted all ~3,100
> diacritic-bearing lexicon words to the unattested frequency band. They now
> **inherit the count of their plain spelling** (713 words rescued), so
> `ïathuh` ranks alongside `iathuh` rather than below it.

### Real-word error detection

A correctly spelled word in the wrong place is invisible to `check_text()` by
construction. `check_realword()` scores each real word against a trigram
model with Stupid Backoff, and flags it when a one-edit lexicon neighbour
fits the slot substantially better.

```python
sp.check_realword("Ka sorkar ki la pynkiew ia ka tulop jong ki MLA")
# ki -> ka  (margin 5.06)   — and leaves the legitimate 'ki' in 'jong ki MLA'
```

In the web interface this is the **word choice** checkbox on the Text tab, on
by default. Wrong-word flags are underlined in dashed amber, distinct from the
solid red of a misspelling, because they are a weaker claim: one says "this is
not a Khasi word", the other says "this may be the wrong Khasi word". They are
never applied automatically.

Khasi's noun-class clitics (`ka` fem, `ki` pl, `u` masc, `i` dim) are short,
extremely frequent, and almost certainly the commonest real-word error in the
language, so they are included deliberately — a three-character minimum would
exclude every one of them.

| min_margin | false-pos | recall (clitics) |
|---|---|---|
| 6.0 | 4.0% | 41.2% |
| 7.0 | 2.7% | 30.0% |
| **8.0** (default) | **0.0%** | **25.0%** |

Defaults sit at the precision end: wrongly "correcting" a word the writer got
right costs more trust than a missed error is worth. Lower `min_margin` for
more recall.

> **Why a trigram model and not the embeddings.** Both were built and measured
> on the same 150 clean / 80 injected sentences. At an identical 0.7%
> false-positive rate the n-gram scorer reached **33.8% recall against the
> embedding's 3.8%** — nine times better. Cosine-to-centroid asks "is this word
> roughly on topic"; the question is "does this word belong in this slot".
> The embedding scorer is kept as `scorer="embedding"` so the comparison stays
> reproducible.

### Context-aware ranking — on by default

The engine ranks every token in isolation: it compares the misspelling
against the lexicon and returns the closest forms. That is the right primary
signal, but it discards what the neighbouring words say about which candidate
belongs in the slot.

`check_text()` now re-ranks each correction's candidates with the same trigram
model, blending the context score with the engine's own ordering:

```
combined = slot_score(candidate) + ALPHA * (-original_rank)
```

`ALPHA` (1.0) is the exchange rate between the two kinds of evidence — how much
log-probability the model must find to move a candidate up one place. It is not
an override: at `ALPHA = 0` the string evidence is discarded and accuracy falls
back to roughly the baseline. The corpus is far too small to have seen every
valid trigram, so on unseen context every candidate floors to the same score,
the term becomes constant, and the engine's ordering stands automatically.

Measured on `tests/sentence_benchmark.json` — 250 real corpus sentences, one
injected single-edit error each:

| | top-1 | top-5 | ms/sentence |
|---|---|---|---|
| isolated (previous behaviour) | 89.6% | 99.2% | 43 |
| **+ context** | **92.4%** | 99.2% | 46 |

Top-5 is unchanged by construction: re-ordering five candidates cannot change
whether the answer is among them. It is top-1 — the suggestion actually
offered — that moves.

`ALPHA` was chosen on one half of the benchmark and the gain confirmed on the
other (86.4% → 89.6%), so it is not fitted to the numbers reported above. On
the full set context fixes 20 items and breaks 13 — a net gain of 7 in 249,
which a two-sided sign test does **not** distinguish from chance
(*p* = 0.30) — and
the 0.5–3.0 plateau all scores within a point of the chosen value.

**Deletions gain most** — 78.0% → 90.2%, the weakest error class in this
project. Dropping a letter destroys more string evidence than any other single
edit, leaving several lexicon entries equidistant from the wreckage; context is
the only thing left that can separate them. Insertions are the one class that
does not benefit (97.1% → 94.2%), string evidence there being already
near-perfect. No per-class gating is applied — that would be four more
parameters fitted to 250 items.

Two costs, both real:

- **Memory.** The model adds **~170 MB** to a speller instance (475 → 646 MB),
  now paid on the first `check_text()` rather than only by `check_realword()`.
- **Corrected neighbours.** Candidates are scored against the words *as typed*.
  Where two errors sit close together each is ranked against the other's
  misspelling, which the model has never seen, so it floors and contributes
  nothing — the pass degrades to the engine's ordering rather than to a wrong
  answer. Joint decoding would recover this; a benchmark with one error per
  sentence does not justify it.

Turn it off with `check_text(text, context=False)`, `--no-context` on the
`check`/`fix`/`file` commands, or `{"context": false}` on `/check` and
`/correct` — useful when checking word lists or fragments, where neighbours
carry no signal.

---

## What is in here

```
khasi_engine/       the engine, copied unmodified from the parent platform
  spell_checker.py    938  the morphology-gated checker
  morphology.py     1,377  Phase 2 — affix cascade, the validity oracle
  database.py       1,385  lexicon store, word-list source
  analyser.py         895  orchestrator; injects the morphology gate
  phonology.py        539  Phase 1 — phonotactic validation
  complex.py          484  Phase 4 — reduplication, compounds, loans
  assimilation.py     246  Phase 3 — surface/underlying reconstruction
  w2v_client.py        79  optional FastText proxy
autocorrect/        Khasi-adapted Norvig speller (digraph-aware edits)
khasi_spell/        this project's own public surface — facade, CLI, API
  speller.py          the facade: check / suggest / check_text / variants
  context.py          trigram re-ranking of candidates in sentence context
  foreign.py          withholds proper nouns, acronyms and known names
  realword.py         correctly spelled, wrongly chosen words
  variants.py         alternative diacritic spellings (i vs ï, n vs ñ)
  ngram.py            Stupid Backoff trigram model over the corpus
  corpus_freq.py      log-scaled corpus frequencies
  static/index.html   the web interface, single file, no build step
data/               khasi_db.json (lexicon) · corpus_freq.json · ngrams.json.gz · gazetteer.json
scripts/            corpus frequency, n-gram and gazetteer builders,
                    benchmark runners, lexicon quarantine
models/fasttext/    cbow_100 FastText model, 786 MB, gitignored
tests/              the test suite
```

Deliberately **not** carried over from the parent: named-entity recognition,
relation extraction, the curation and governance system, CLDF/LIFT export,
grapheme-to-phoneme conversion, text-to-speech, the React frontend (replaced
here by a single static page), and the v1/v2 REST surface.

`autocorrect/Speller_top5.py` was also left out — it was the only importer of
`gensim`, and nothing imported it.

## Tests

```bash
python3 -m pytest tests -q               # whole suite, ~2 min
python3 -m pytest tests/test_spellchecker.py -q   # spelling only, ~40 s
python3 -m pytest tests -q -k "derived"  # just the morphology gate
```

`pytest` is the only test dependency. If it is not installed:
`python3 -m pip install pytest`.

**135 tests.** 27 are inherited from the parent (ranking and compounds 10,
phonology rules 7, lookup and root gate 5, morphology 3, golden corpus 2).
The other 118 are new (48 spellchecker, 29 embeddings/n-gram, 35 diacritics, 6 coda): the parent had **no** tests on the spelling path at
all, so any change to it was unguarded.

Most of the runtime is the one-off lexicon load; the fixture is module-scoped
so the whole file costs roughly one construction.

Tests in `test_spellchecker.py` assert observed behaviour including its
weaknesses. Where the checker generates the right answer but ranks it below a
competitor, the test asserts top-5 membership and is marked `WEAK` — those
are the cases to tighten when the ranking model is improved.

## Benchmark

```bash
python3 scripts/run_benchmark.py          # ~13 min, 308 items
python3 scripts/run_benchmark.py --limit 50   # quick check
```

`tests/spell_benchmark.json` freezes 308 error/correction pairs — 300
synthetic single-edit corruptions plus 8 cases that were reported bugs during
development (`dand`→`dang`, `shnng`→`shnong`, `pyleng`→`pylleng` …).

> **Why frozen.** The ad-hoc probe used earlier regenerated its sample *from
> the lexicon* on every run, so editing the lexicon silently changed the test
> set. Two measurements a day apart shared **2 of 76 items** — which made a
> 6-point "regression" look real when nothing had regressed. Any number
> quoted from a moving test set is noise. This file makes results comparable
> over time, and it is the smallest piece of the proposal's *KhasiSpell-Bench*
> that could be built today.

It is still synthetic and drawn from the lexicon, so it measures recovery of
known words, not performance on running text. A benchmark built from
authentic Khasi misspellings remains the real goal.

**Current figures** (23 Sept 2026, `scripts/evaluate.py` and the two
benchmark runners):

| | |
|---|---|
| detection | **98.0%** (288 of 294), precision 1.000 — none of 294 correct control words flagged |
| top-1 / top-5 / MRR | **73.5%** (216) / **91.5%** (269) / 0.808 |
| sentence benchmark, top-1 isolated → with context | 87.1% → **91.6%** (21 fixed, 10 broken, sign test p = 0.071) |
| flagged on 200 untouched corpus sentences | **3.19%** (105 of 3,294 tokens) |
| rejected with no suggestion | 0.30% (10 tokens) |
| targets of `khasi_test_pairs_v2.csv` reachable as candidates | 41.5% of 23,254 (35.7% from lexicon forms alone) |
| latency | 47–51 ms per word |

### Sentence benchmark

```bash
python3 scripts/run_sentence_benchmark.py         # ~25 min, A/B with and without context
python3 scripts/run_sentence_benchmark.py --limit 50 --no-compare   # quick check
```

`tests/sentence_benchmark.json` freezes 250 **real deduplicated corpus
sentences**, each with one single-edit error injected at a recorded offset.
It answers a different question from the word benchmark: not "is the right
correction found for an isolated misspelling", but "when that misspelling sits
in a real sentence, is the right word put first". The two diverge exactly where
context matters — a misspelling with several equidistant candidates is a coin
toss in isolation and often unambiguous in a sentence.

Only items the checker actually flags were kept, so the figures isolate
*ranking* from *detection*. The runner reports `error located` for that reason:
if it falls below 250, detection has regressed and the accuracy figures are
being computed over a different set.

Frozen for the same reason as the word benchmark, and drawn from running text
rather than the lexicon — so unlike that one, it does measure performance on
real sentences, though the errors in it are still injected rather than
authentic.

### Fragments are not words

`ng` was accepted as correctly spelled. It is a digraph — one phoneme — and
the lexicon's own `phonology` block lists it under
`semantically_invalid_standalone`. Tracing it turned up **two independent
routes** by which fragments and foreign words were passing as Khasi.

**1. Gate 0 deferred to the lexicon.** It consulted the invalid-standalone
list and then added `and not self._db.is_known(word)`, reasoning that the DB
might hold such tokens as proper entries. It does: `kh_DB_005909` is `ng`,
glossed *"The seventh letter of the Khasi Alphabet"*. So the escape hatch
always fired and the gate never blocked anything it was written to block. A
letter-name belongs in a dictionary but is not a word of running prose, and
the curated list is the explicit statement of that, so the lexicon check is
gone.

**2. `is_known()` accepted anything in `_compound_tokens`.** That index is
built by splitting multi-word surface forms, so that words living only
inside phrases — `shnong` in `dorbar shnong` — stay recognised. 59% of
surface forms are multi-word, and some carry an English translation beside
the Khasi:

```
'ioh sah jit la ka long sone face and honour'
'ka masi ka la rong jyndat ïa ka jaiñ ka the cow took away the cloth on passing'
```

Splitting those put `and`, `face`, `cow`, `big`, `car` into the index, and
because `is_known()` consults it, **`cow` passed a Khasi spell check**. Bare
digraphs arrived the same way: `sh` and `kh` are phonemes, not words.

A phonotactic screen at load time now decides which extracted tokens may
answer "is this a Khasi word".

**Candidates and acceptance are different questions.** Screening both was the
obvious move and it was wrong: `validate()` has a small false-reject rate,
and filtering the *candidate pool* by it cost about 4 points of top-1 —
`shafon` is a real word the validator rejects, and it stopped being offered
as a correction for `shafan`. The two are now separate predicates:

| | question | used for |
|---|---|---|
| `is_known()` | is this an acceptable Khasi word? | acceptance — screened |
| `is_attested()` | did the lexicon write this down? | candidate filtering — unscreened |

Result — `ng`, `sh`, `kh`, `cow`, `and`, `big`, `car` rejected; `shnong`,
`mynsiem`, `kynih`, `dorbar` still accepted; `shafon` offerable again
without being accepted on its own:

| | before | after |
|---|---|---|
| word benchmark top-1 | 67.2% | **67.5%** |
| word benchmark top-5 | 80.5% | **80.8%** |
| sentence benchmark top-1 | 91.6% | 91.6% |
| flagged on clean text | 2.79% | 3.04% |

The flag rate rises because words that were wrongly accepted are now
correctly rejected. 424 of 7,511 extracted tokens fail the screen.

**What it does not fix.** The screen tests pronounceability, not wordhood, so
English that is phonotactically legal Khasi still passes — `day`, `gas` and
`kyn` (a fragment of `kyn-ih`) are still accepted. Removing those needs the
borrowing list, not a phonology rule.

#### The lexicon, cleaned

The root cause of both `cow` passing a spell check and `shafon` being
offered as a correction was the same: **260 lexicon entries whose `surface`
field is not a Khasi word.** They have been moved to the lexicon's existing
top-level `quarantine` list by `scripts/quarantine_illegal_chars.py`.

Two kinds:

**143 are English gloss text that leaked into the surface field** —
`'a poisoned fish as'`, `'advanced age'`, `'adv. at all; u,n. rice cropped
during the rainy season'`. Compound-token extraction split these into
`acid`, `black`, `calm`, `america`, `villager`, and `is_known()` consults
that index.

**117 are OCR damage** from digitising a printed dictionary, and the
confusions are systematic:

```
fi -> ñ    kalaifi, bseifi, buifi, boifi-boin, khangkhfii
x  -> kh   xhyllew, xhie-khynraw
c  -> e/o  arsicn, bcr, kcm, kyntcm, tyngkrcin
v  -> u    v-ai-kylliang, vai-nguh
```

Nothing was deleted. `quarantine` already held 290 entries of the same
schema and is the established destination for unusable records; each moved
entry carries a `quarantine_reason`, and there is a timestamped backup.

**Repairs are proposed, not applied.** Only 11 of the 117 repair onto a word
the lexicon already holds (`bseifi` → `bseiñ`, `arsicn` → `arsien`), which
makes those plainly redundant. For the other 106 the repair is a guess, and
guessing at dictionary headwords is a job for a Khasi speaker — they are
written to `data/ocr_repair_review.csv` with the proposed form and a
`verdict` column.

| | before | after |
|---|---|---|
| lexicon entries | 30,256 | 29,996 |
| quarantine | 290 | 550 |
| compound tokens | 7,511 | 7,207 |
| word benchmark top-1 | 67.5% | **67.5%** |
| sentence benchmark top-1 | 91.6% | **91.6%** |
| flagged on clean text | 3.04% | 3.01% |

Accuracy-neutral, which is the expected result: these entries could never be
offered as corrections, and their only live effect was to make `is_known()`
answer True for strings that are not Khasi words.

**The benchmarks were contaminated from the same source.** Ten word items and
one sentence item have gold answers that are English — `shafon`, `infancy`,
`acid`, `funeral`, `fear`, `defeat`, `tive`, `stand`, `took`, `euphony`,
`there` — with inputs like `infacny`, `ancid`, `fuieral`, `htere`. They were
never Khasi spelling tests. All are flagged `valid: false` with a reason and
**kept in the files**; both runners exclude them from the headline and print
the count. Deleting items from a frozen benchmark to improve a score is
precisely the practice this project avoids.

> Re-migrate to PostgreSQL if you are running the production path — the
> JSON is the source of truth and the database is built from it.

#### The sentence benchmark was scoring correct answers wrong

13 of its golds carried the **pre-normalisation plain `ia-` prefix** —
`nongialam`, `iathuh`, `jingialang`, `pyniaid` — while the lexicon records
only the `ïa-` form. The golds came from corpus text, and the corpus holds
no diacritics at all, so every one of them arrived plain. The benchmark was
therefore marking the checker wrong for proposing the standard spelling:
for `nowgialam` it offered exactly one candidate, `nongïalam`, and scored 0.

Each gold was respelled only after confirming the `ïa` form is a lexicon
entry, and each carries a `corrected` field recording the change. **Typos
are left as typed** — writers do drop the diaeresis, which is what makes the
item realistic.

A single lexicon phrase, `ka jingíakren ka la batur`, also spelled the
prefix with an **acute í** instead of the diaeresis. `jingïakren` is a
separate, correct entry; the stray form reached candidate generation through
compound-token extraction and was being offered in place of the right word.
Six strings corrected.

| | before | after |
|---|---|---|
| sentence top-1 (context) | 91.2% | **92.4%** |
| sentence top-5 | 94.8% | **99.2%** |
| MRR | 0.926 | **0.949** |
| detection | 249/249 | 249/249 |

> **Read that top-5 jump carefully.** It is a measurement correction, not an
> improvement in the checker. Those 13 golds were previously *unreachable* —
> the plain spelling is not in the lexicon, so no ranking could ever have
> produced it. The checker had been finding the right word all along and
> being scored wrong for it. What the corrected figures now show is the real
> remaining weakness: of the 13, only 2 are top-1 while 13 of 13 are in the
> top-5, so these words are **retrieved but mis-ranked** — `ïakren` loses to
> `kren`, `ïalap` to `hap`.

#### One repair pass, rules as data

`scripts/repair_lexicon.py` replaces six scripts that had grown apart in
four measurable ways: `surface()` was redefined in **all six** and
`retarget()` in three; four rules (`no_vowel`, `invalid_onset`, `markup`,
`diacritic`) were implemented in two scripts each, so an entry's fate
depended on which ran first; the same defect class got different treatment,
with an illegal-character entry quarantined where a structurally identical
case elsewhere was rewritten; and **only one of the six compared meaning**
before withdrawing a record on a collision, leaving three able to create a
duplicate surface silently.

Rules are now one ordered list of `(detect, repairs)` pairs, and every one
goes through the same `decide()`:

1. a repair is already a lexicon surface — **collision**, settled on
   meaning: the same word is withdrawn as a duplicate; a different sense
   means the repair is wrong, so the record is withdrawn as unresolved with
   the colliding word named. Never merged, never left live.
2. exactly one valid repair — **rewrite in place**, keeping the gloss.
3. otherwise — **quarantine**, carrying the gloss and the candidates.

> **The meaning check earned its keep immediately.** A new rule for a
> spurious leading consonant — `ljing` for `jing`, `jmong` for `mong` —
> looked provable: the repair lands on an existing entry. Run across the
> lexicon it produced **22 conflicts out of 25**, because Khasi has minimal
> pairs that satisfy it: `ki'm` reduces to `i'm` and `ki'n` to `i'n`, but
> those are different pronouns — `ki` 'they' against `i` 'it'. Had the rule
> shipped in any of the six older scripts, none of which checked meaning, it
> would have silently deleted them.
>
> The rule is now restricted to the onsets confirmed as not occurring in
> Khasi (`tjt`, `mdwk`, `jsl`, `rsh`, `mth`, `lj`, `jm`). Even there the
> repair is usually unrecoverable — `mthen` reduces to `hen`, a different
> word — so 10 of 16 are withdrawn as unresolved rather than repaired.

| | before | after |
|---|---|---|
| lexicon entries | 29,708 | **29,694** |
| quarantined with a reason | 528 | **544** |
| sentence benchmark | 92.4% / 99.2%, 249/249 | unchanged |
| word benchmark | 84.0% detect, 67.0% top-1, 80.6% top-5 | |
| tests | 248 | **253 passing** |

All six original scripts now find nothing to do, which is the check that the
unified pass reproduces them. They are kept for now so the audit trail stays
legible.

#### `ü` is `ïi`, and a reduplication repairs from its twin

**29 entries carried a `ü`**, always the sequence `ïi` misread as one
character: `üng` is `ïing` ('house'), which makes `ka üng ka itynnat` read
as `ka ïing ka itynnat`. Where the substitution would double the diacritic —
the entry already had a `ï` before the `ü` — it collapses to one. No entry
now contains `ü`, and no resulting single-token form fails `validate()`.

**Khasi reduplicates freely**, so when one half of an `A-B` compound is a
known word and the other carries a scan artefact, the good half says what
the bad one should be:

```
baiii-bain        -> baiñ-baiñ   (already an entry, so the record was withdrawn)
jatmut-jatrnat    -> jatmut-jatmut
thiah-thdi        -> thiah-thiah
tyngkliip-tyngkhap-> tyngkhap-tyngkhap
```

The target takes the attested tilde spelling where there is one: `baiñ` is
an entry and `bain` is not.

**This rule had to be kept narrow.** 23 hyphenated pairs have mismatched
halves, and most are not damaged at all — Khasi echo-reduplication
alternates the vowel on purpose (`jirwit-jirwat`, `awri-awra`,
`tharuh-thareh`, `hynñium-hynñiam`), and many pairs are ordinary coordinate
compounds: `ïadih-ïabam` is 'drink-eat', `riewbah-riewsan` pairs two kinds
of person. Forcing halves to match would have destroyed every one of them.

A half is therefore repaired only when it carries a **confirmed OCR
confusion** (`ii` for `ñ`/`ng`, `rn` for `m`) or fails `validate()`
outright, which reduces 23 candidates to 4. An earlier, looser test — "four
consonants in a row" — wrongly caught `sohkhruh-sohkhram`, since Khasi has
clusters that long, and `shong-shiliangkhmat`, which is not a reduplication
at all.

**A collision is decided on meaning, not on the surface.** `baiii-bain`
repairs to `baiñ-baiñ`, which the lexicon already held; rewriting would have
left two records claiming the same word, with lookup returning whichever
came first. The test suite caught that.

Withdrawing the loser is only right when the two records *are* the same
word, so `same_word()` compares the glosses rather than assuming. Identical
senses settle it, and so does one gloss being a usage example of the other —
the scan split some dictionary entries in two, leaving a record that quotes
the headword instead of defining it:

```
withdrawn  baiii-bain  ADV  "As ; Jem baifi-bain)"
surviving  baiñ-baiñ   ADV  "Very (flexible ; Pliable ; Soft ; Supple)"
```

Same part of speech, and the first is an example *of* the second. The
quotation carries the same damage as the headword did — `baifi` for `baiñ` —
so the known confusions are folded before the comparison, which is what
makes the match findable at all. Where the senses genuinely differ, the
entry is left alone and reported rather than merged: two meanings must never
be collapsed into one silently.

A standing test asserts that **no surface appears twice** in the lexicon.

#### An `-aiñ` root anywhere in a compound

`jain` on its own offered `jaiñ`, but every compound built on it —
`jainsem`, `jainkhor`, `jainkup-jainsem` — offered nothing, because the
compound rule only looked at the *end* of the word. `jain` is 'cloth', and
it is the first element of most words that contain it.

The rule now substitutes an attested `-aiñ` root wherever it **opens** a
hyphen-separated part, converting every occurrence rather than the first:

```
jainsem         -> jaiñsem          iing-jain    -> iing-jaiñ
jainkhor        -> jaiñkhor         jain-ryndia  -> jaiñ-ryndia
jainkup-jainsem -> jaiñkup-jaiñsem  mainker      -> maiñker
```

**Start-anchoring is not by itself enough.** `maintain` really does begin
with `main`, so the word must also be a curated lexicon headword — the same
guard the single-word rule uses. `maintain`, `remain`, `domain`, `plain` and
`again` are none of them entries; `jainsem`, `jainkhor` and `mainker` all
are. The remainder after the root must additionally be a known word of at
least two characters: `bain` + `a` and `dain` + `i` are real clitics, but
`baiña` and `daiñi` are not words anyone writes.

37 lexicon surfaces gain a variant they previously lacked. Benchmarks are
unchanged — this adds alternatives for accepted words and touches no
correction path.

#### Scan damage repaired or withdrawn

Four further readings confirmed by a Khasi speaker — no stray bracket in a
word, no foreign diacritic and no digit, no segment without a vowel, and the
nine invalid onsets. A leading `*` joins them: it marked a borrowed headword
in the printed dictionary and was captured as a letter, which is provable
because stripping it lands on an existing entry 22 times out of 24
(`*dorbin` -> `dorbin`, `*dukandar` -> `dukandar`).

`scripts/repair_scan_damage.py` decides an outcome per entry rather than per
group:

| outcome | n | example |
|---|---|---|
| quarantined as a duplicate | 34 | `*dud` -> `dud`, `jdin-ryndia` -> `jain-ryndia` |
| **surface rewritten**, entry kept | 15 | `shiüng` -> `shiung`, `tiew-pathai-khub6r` -> `tiew-pathai-khubor` |
| quarantined unresolved | 55 | `bd-liim` — `ba-`, `be-` and `bi-` all valid |

Repairs are principled rather than guessed: strip the marker, strip stray
punctuation, fold a foreign diacritic to its base letter, map a digit to the
letter it was misread for (`khub6r` -> `khubor`), or try each vowel where a
`d` stands in a vowel's place. Every candidate is then checked against the
lexicon and `validate()`.

**A rewrite must not smuggle English in.** `stones)` and `onomatopsea)` are
gloss text captured as headwords, and stripping the bracket would have left
`stones` and `onomatopsea` looking like entries. The script reads
`/usr/share/dict` at build time purely to refuse such a rewrite — it is
never bundled and never imported by the package, and if absent the guard
simply does not fire. It caught `stones`; it missed `onomatopsea`, which is
itself misspelled and so not in any dictionary.

**Quarantine is not deletion.** The 55 unresolved entries move with their
gloss and their candidate repairs recorded in the reason, so the intended
word can be settled later:

```
"scan damage confirmed as not valid Khasi; the intended word cannot be
 recovered from the data (candidate repairs: ['ba-liim', 'be-liim',
 'bi-liim']; gloss: '?^ ; N. See ...')"
```

| | before | after |
|---|---|---|
| lexicon entries | 29,798 | **29,709** |
| quarantined with a reason | 438 | **527** |
| single-token surfaces with a foreign character | 89 | **0** |
| sentence benchmark | 92.4% / 99.2%, 249/249 | unchanged |
| word benchmark | 84.1% / 67.1% / 80.7% | unchanged |
| flagged on clean text | 3.07% | 3.07% |

#### Confirmed damage removed

Two rules, both surfaced by `export_ocr_candidates.py` and then confirmed by
a Khasi speaker, are applied by `scripts/quarantine_confirmed_damage.py`.
**54 entries moved to `quarantine`**, none deleted.

**Every word contains a vowel** — 28 surfaces have none anywhere: `bdd`,
`hdr`, `bmb`, `kmn`, `phlkr`, `tngdw`. `validate()` already reported "all
Khasi syllables require a vowel nucleus"; nothing acted on it. This also
removed the `ng` entry (the letter name, kh_DB_005909) whose presence was
what let Gate 0 defer to the lexicon in the first place.

**The onsets `jk`, `lm`, `pb`, `shm`, `jb`, `mt`, `tk`, `jd`, `tj` are not
Khasi** — held back from the cluster extension for lack of evidence, and
that reading was confirmed. Of the 46 entries carrying one, **26 repair onto
a word the lexicon already holds**, so those records are duplicates:

```
jkaptan -> kaptan    jbiskit -> biskit    pbieng -> phieng
jkang   -> kang      jbol    -> bol       mtai   -> mai
```

Three independent lines agree: the repair rate (`pb -> ph` 5 of 5, `jb -> b`
4 of 4, `jk -> k` 10 of 12), the glosses (`pbet-lyndet` is glossed "[Imit.
**phet** lyndet-phet-tuh.]", `jkaptan` as "Captain (same as **Koptdn**)"),
and the speaker's judgement.

**No hyphenated part is a single letter without a vowel** — a third
confirmed rule, applied after the first two. `ka-n`, `i-n`, `nga-n`, `ki-n`
and `u-n` are the contractions the lexicon spells `ka'n`, `i'n`, `nga'n`,
`ki'n`, `u'n`, all five attested and glossed to match ("She...not / she
that (negation or comparison)"); the hyphen simply stands where the
apostrophe belongs. `bam-hynroh-n-bnai` duplicates `bam-hynroh-u-bnai`, and
`'thei-2` carries a digit. Seven entries, each with its replacement already
in the lexicon bar the last.

**Position decides.** A single letter in the MIDDLE of a compound is
correct — `blang-u-bhed`, `ba-i-bit`, `kha-u-man`, `bam-hynroh-u-bnai` —
because `u`, `i` and `a` are real words there, the noun-class clitics, with
more of the compound following. 28 such entries are untouched.

A single letter at the **end** is a scan artefact splitting the final vowel
from its stem, and `scripts/fix_trailing_hyphen.py` repairs those seven:

```
tyr-a    -> tyra       lyng-a   -> lynga      pi-e   -> pie
pynkyr-a -> pynkyra    loli-i   -> lolii      hih-i  -> hihi
ioh-i    -> ïohi   (already an entry, so the record was a duplicate)
```

Six had their surface rewritten in place, keeping the entry and its gloss;
`ioh-i` was quarantined because its joined form was already present. Every
join was checked against `validate()` first — all seven pass, so the rewrite
could not introduce an impossible word.

**Left alone deliberately**: the 20 invalid-onset entries whose repair is
*not* attested — removing them would drop a surface with no replacement.

| | before | after |
|---|---|---|
| lexicon entries | 29,860 | **29,798** |
| quarantined with a reason | 376 | **438** |
| word benchmark | 84.1% detect, 80.7% top-5 | unchanged |
| sentence benchmark detection | 249/249 | 249/249 |
| flagged on clean text | 3.07% | 3.07% |

The words the duplicates shadowed are intact: `kang`, `biskit`, `kaptan`,
`phieng`, `mai`, `ben`, `bol`, `king` all still accepted.

#### The phonotactic validator was rejecting correct Khasi

`phonology.valid_initial_clusters` held **76 clusters and was incomplete**.
495 single-token lexicon surfaces failed `validate()`, and **406 of those
failures were legitimate words** — the list lacked `khw` (`khwai`,
`khwaiñ`, both curated entries), `lng`, `dng`, `phn`, `rw`, `lw`, `jw`,
`thw`, and every cluster containing the apostrophe (`s'`, `l'`, `k'`,
`sh'`) even though `'` is listed among the Khasi consonants.

This mattered beyond the validator. `validate()` drives `_phonotactic_ok`,
so those words were kept out of `_compound_known`; and it drives
`_offerable`, where only the `is_attested` escape hatch rescued them. The
escape hatch was doing work the phonology should have done.

**23 clusters added, on two kinds of evidence.**

*Corpus-attested* (14) — the corpus is modern typed text, independent of the
printed dictionary's OCR, so a cluster beginning corpus words the checker
accepts as Khasi is externally confirmed: `rwai`, `lwait`, `khwai`,
`lngaid`, `dngong`, `thwet`, `phngit`, `twad`, `rtiang`.

*Structural* (9) — the corpus holds **zero apostrophes and zero ñ** across
6.67M tokens; both were stripped or never typed. Silence there is not
evidence of absence, so `s'`, `l'`, `k'`, `sh'`, `b'`, `p'`, `r'`, `khñ`
and `kñ` are judged on the DB's own consonant inventory plus the number of
distinct lexicon entries using them (17 for `s'`, 10 for `l'`).

**Deliberately excluded.** Digraphs (`sh`, `kh`, `th`, `ph`, `ng`) are
single phonemes, not clusters — they appeared in the failure list only
because the words carrying them fail for a different reason, a vowel
misread as `d` (`shabdr`, `thaldb`), and adding them would have papered over
that damage. Markup artefacts (`*d`, `*kh`) and onsets with no evidence at
all (`jk`, `lm`, `pb`, `shm`, `jb`, `mt`, `tk`, `jd`) are left for review —
several of those appear only in entries that look like scan damage.

| | before | after |
|---|---|---|
| clusters | 76 | **99** |
| surfaces failing `validate()` | 495 | **340** |
| word benchmark detection | 83.7% | **84.1%** |
| word benchmark top-5 | 80.3% | **80.7%** |
| sentence benchmark | 91.6%, 249/249 | unchanged |
| flagged on clean text | 3.07% | 3.07% |

Damage and non-Khasi still fail: `pbetkiri`, `tjmja`, `jkang`, `goh`,
`register`, `cow`, `ng`.

#### A hyphenated compound is a word

`_check_hyphenated` exists because Phase 4 accepts any hyphenated token
without checking its parts, so `man-miay` passed even though `miay` does
not. It split every hyphenated token and corrected parts that failed
`is_known` — **without first asking whether the whole token was itself a
lexicon entry.**

`jrain-jrain` is an entry; neither `jrain` is. So a real reduplication was
rewritten into a different word:

```
jrain-jrain     -> jain-jain
jdinkup-jainsem -> jainkup-jainsem
saw-ka-siau     -> saw-ka-siat
```

It also made the two entry points contradict each other — `check()` accepted
these while `check_text()` corrected them, so the same word got opposite
verdicts depending on which API was called. Measured on 400 sampled
single-token lexicon words, they disagreed on **13 (3.3%)**; after the whole-
token check, **0**. `man-miay -> man-miat` still works, and compounds the
lexicon does not record are still inspected.

> **The other half of that inconsistency is not fixed, deliberately.** A word
> the checker rejects but cannot suggest for is invisible to `check_text()`:
> `check_sentence` only records a correction when `gate["suggestions"]` is
> non-empty. On 200 clean corpus sentences that hides **126 tokens (3.83%)** —
> more than the visible flag rate — and they are mostly English borrowings and
> lowercase names: `elekshon`, `school`, `percent`, `president`, `meghalaya`,
> `israel`. Surfacing them is arguably right, since a spellchecker should
> underline a word it cannot fix, but it would roughly double visible noise
> and undo much of the suppression work above. That is a product decision,
> not a defect to be quietly patched.
>
> *Update, 23 Sept 2026:* 10 tokens (0.30%). Phonotactically impossible words
> are now flagged rather than dropped, and frequent corpus words are accepted
> — see *Corpus words as candidates and as words*.

#### `-ain` is written `-aiñ`

Khasi writes the sequence **a + i + ñ** — a plain `i` in the diphthong, the
tilde on the nasal alone. Three things were wrong here.

**The lexicon spelled it two ways.** 553 surfaces used `aiñ` and 84 used
`aïñ`, and **20 words appeared both ways**, which is what makes it an error
rather than a variant. Left alone it produced nonsense alternatives:
`nongthain` was offered `nongthaïñ`, two diacritics where the orthography
has one, because the generator found that spelling attested.
`scripts/fix_ain_diaeresis.py` normalises it — 545 strings across 91
entries — and then merges the 20 surfaces that collide as a result, reusing
the `ia` → `ïa` pass's `dedupe()` rather than rewriting it.

**Headwords the lexicon records only plain.** `spain` is an entry glossed
*"Bandage; To swathe"*; there is no `spaiñ` headword, so no lookup route
could offer it. Route 5 supplies it by rule.

**Compounds whose final element carries the tilde.** Route 3 strips known
*prefixes*, so it never reaches a root sitting at the end. `saitjain` alone
is 890 corpus tokens:

```
saitjain    -> saitjaiñ      (sait + jain)
jinglehrain -> jinglehraiñ   (jingleh + rain)
nongthain   -> nongthaiñ     (nong + thain)
```

Guarded four ways, and each guard earned its place:

| guard | what it stops |
|---|---|
| lowercase in context | `Spain` the country, `Hussain`, `Gohain` |
| head is a known Khasi word | `hussain` (`huss`), `again` (`ag`) |
| head ≥ 3 characters | `remain` (`re` + `main`), `domain` (`do` + `main`) |
| not a reduplication | the lopsided `dain-daiñ`; route 1 gives `daiñ-daiñ` |

The corpus settles the case guard: `Hussain` is capitalised in 53 of 53
uses and `Gohain` in 32 of 32, while `spain` splits 34 capitalised (the
country) against 8 lowercase (the Khasi word). The corpus itself contains
**zero ñ characters**, so it is evidence about keyboards, not orthography —
the lexicon remains the authority.

86 further candidates, ranked by corpus frequency and graded by confidence,
are in `data/ain_candidates_review.csv` for the cases the guards decline.

| | result |
|---|---|
| word benchmark | 67.1% top-1, 80.3% top-5 — unchanged |
| sentence benchmark | 91.6%, 249/249 — unchanged |
| lexicon | 29,860 entries, 0 duplicate surfaces |

One benchmark gold, `sngewkhlaïñ`, encoded the spelling now determined to
be wrong. It was respelled rather than quarantined — there *is* a correct
answer — and carries a `corrected` field recording what changed and why.

#### `g` occurs only in `ng`

Khasi has no standalone /g/ — the letter appears only as the second half of
the digraph `ng`. The character inventory said so all along:
`valid_chars_note` records that g *"is included as a valid character because
it appears as the second component of the 'ng' digraph (velar nasal). It
does not exist as a standalone Khasi phoneme."*

Nothing enforced it. `goh`, `gain`, `garo` and the borrowings `register`,
`gospel`, `glass`, `telegram`, `although` all validated as Khasi words.
`phonology.validate()` now rejects a `g` that is not preceded by `n`, and
**116 lexicon entries** carrying one were quarantined alongside the earlier
260.

Names do carry a bare g — `Garo`, `Meghalaya`, `Guwahati` — and they stay
recognised through the gazetteer and the capitalisation rules in
`khasi_spell.foreign`, not by weakening the constraint.

**The rule exposed a second systematic OCR confusion: `ug` for `ng`.**

```
iug-dong               -> ing-dong
heng-heug              -> heng-heng
jiugpynkylla-khongpong -> jingpynkylla-khongpong
```

Repairing `ug` → `ng` makes these valid Khasi again, so the substitution
joined `fi` → `ñ` in the repair table. The review list at
`data/ocr_repair_review.csv` now holds **190 rows** across both passes — it
is rebuilt from the whole quarantine rather than the current batch, after an
earlier version silently dropped the first pass's rows.

> **A fragile coupling this uncovered.** Gate 0 used to decide whether a
> token was sub-phonemic by *grepping the validator's error messages* for
> the strings "sub-phonemic unit" and "no standalone". The first draft of
> this rule produced a message containing the words "no standalone", so
> ordinary misspellings — `dign`, `lynbga`, `balagn` — were routed into
> Gate 0; and because `check_sentence` only reports `gate_reached` 1 or 3,
> they stopped being flagged at all. Sentence detection fell from 249/249 to
> **214/249** and top-1 from 91.6% to 78.3%. The gate now tests membership
> of `semantically_invalid_standalone` directly, which is the actual
> authority, and a test asserts the code shape so the coupling cannot
> return.

| | before | after |
|---|---|---|
| lexicon entries | 29,996 | 29,880 |
| quarantined with a reason | 260 | 376 |
| word benchmark top-1 | 67.5% (298 items) | 67.1% (295 items) |
| sentence benchmark top-1 | 91.6% | **91.6%** |
| tests | 204 | **212 passing** |

Three further benchmark golds were contaminated the same way — `ghee`,
`thoughtless`, and `syrmgiew`, an OCR corruption of `syrngiew`, which is the
real Khasi word and still in the lexicon. They are flagged `valid: false`
and kept, bringing the word benchmark to 13 quarantined of 308.

#### Letters Khasi does not have

`c`, `f`, `q`, `v`, `x` and `z` are not in the Khasi alphabet.
`phonology.valid_chars` has always said so and `validate()` has always
reported *"Illegal characters: f — not in Khasi orthography"*. They reached
the suggestion list through `_offerable`'s `is_attested` escape, which
exists so that words the validator false-rejects — place names, loans,
reduplications — stay offerable.

That escape is right for the **soft** rules (codas, syllable shape) and
wrong for the character inventory, where there is nothing to overrule: no
Khasi word contains these letters. The illegal-character test now runs
first and is absolute.

The lexicon appears to attest some of these words only because English gloss
text leaked into **260 surface_form fields** — `'a poisoned fish as'`,
`'advanced age'`, `'adv. at all; u,n. rice cropped during the rainy season'`
— which compound-token extraction then split into `acid`, `black`, `calm`,
`america`. It is the same contamination behind `cow` passing a spell check,
seen from the other side.

**Named entities are exempt.** Meghalaya's Garo and Bengali place names use
these letters freely — **652 of the 6,609 gazetteer tokens** contain one
(`Achakchiring`, `Achugre`, `Adugachol`, `Adventist`) — and those are real
words a writer may mistype. A candidate carrying an illegal letter is
offered only if the gazetteer holds it.

```
shafan -> ['shalan', 'shajan', 'shanam', 'shabar', 'shaba']   (was: … 'shafon', 'bafan')
```

> **This corrected an earlier fix of mine.** `shafon` had been *restored* to
> this list on the grounds that the lexicon attests it. It does — but only
> through the leaked gloss text above. Attestation was the wrong test; the
> alphabet is not negotiable.

**Seven benchmark items were contaminated the same way.** Their gold answers
are `shafon`, `infancy`, `acid`, `funeral`, `fear`, `defeat`, `tive`, and
their inputs (`infacny`, `ancid`, `fuieral`, `defesat`) are misspellings of
*English* words — they were never Khasi spelling tests. They are flagged
`valid: false` with a reason and **kept in the file**, never deleted, so the
frozen set is not edited for convenience and the contamination stays
auditable; `run_benchmark.py` excludes them from the headline and reports
the count.

| | before | after |
|---|---|---|
| top-1, all 308 items | 67.5% | 65.9% |
| **top-1, 301 valid items** | 67.5% | **67.4%** |
| top-5, 301 valid items | 80.8% | 80.7% |
| sentence benchmark | 91.6% | 91.6% |

The apparent 1.6-point drop is entirely those seven items becoming
unanswerable, which is correct: there is no valid Khasi answer to give. On
everything that was ever a real test, the change is neutral.

#### And a rejection needs a suggestion

Gate 0 returned a hardcoded `"suggestions": []` — it was written to block,
not to help, so `ng` was marked wrong with nothing offered. It now runs the
same narrow, distance-first search Gate 1 uses. Two things had to be fixed
before that was worth having:

**Never suggest what the checker would itself reject.** The first attempt
offered `['n', 'g', 'u', 'na', 'ngi']` for `ng`. `n` and `g` are bare
consonants that the checker rejects — a correction that is itself invalid is
not a correction. They led because phoneme-aware distance scores `ng -> n`
at **0.4**, deleting half a digraph being cheap, so they outranked the real
answers `nga` 'I' and `ngi` 'we' at 1.0.

The six ad-hoc `phonotactically_valid(s) or is_attested(s)` filters scattered
through the suggestion paths are now one predicate, `_offerable()`, which
adds two conditions: the candidate must contain a vowel — every Khasi
syllable requires a vowel nucleus, which is what separates `n`/`g`/`t`/`k`
from the genuine one-letter clitics `u`, `i`, `a` — and it must not itself
be listed in `semantically_invalid_standalone`.

**Over-fetch before filtering.** Gates 0 and 1 asked the search for exactly
`n` candidates and filtered afterwards, so `ng` came back with three
suggestions instead of five, `nga` among the losses. They now request `n * 4`
and truncate after filtering.

**A correction must share material with the input.** Even after the two
fixes above, `ph` was answered with `['u', 'phi', 'a', 'i', 'ï']`. Only
`phi` ('you' pl.) is a correction of `ph`; the others share nothing with it
at all. The cause is again the digraph: `ph` is a single phoneme, so
`ph -> u` is one substitution and scores 1.0 — identical to `ph -> phi` —
and the one-letter clitics then win the tie on raw frequency.

`_offerable()` therefore takes the input word as well, and requires the
candidate to share at least one character with it. It is a weak test, but it
is precisely the one a phoneme-distance metric cannot make for itself. Note
the rule is "shares material", not "never suggest short words" — `u` remains
offerable for `nu`, just not for `ph`.

```
ng -> ['na', 'ngi', 'ong', 'nga', 'ne']
ph -> ['phi', 'eh', 'oh', 'pa', 'pha']
sh -> ['sha', 'shu', 'shi', 'sa', 'eh']
```

Word and sentence benchmarks are unchanged by all of this (67.5% / 91.6%),
which is the point: it corrects behaviour the benchmarks never measured,
because neither contains a sub-phonemic fragment as an input.

### Proper nouns and acronyms — withheld by default

On 200 untouched corpus sentences the checker flagged **8.23% of all
tokens**, and the suggestions were noise:

```
East -> pat      Cherra -> heriap    resign -> arsien    Ardent -> armet
```

An editor that underlines a seventh of a news article, every place name
included, is one people switch off — and a spellchecker that is switched off
has 0% accuracy. Two self-contained signals now withhold the worst of it:

| signal | share of flags | benchmark errors hidden |
|---|---|---|
| known name (gazetteer) | 9.1% of survivors | 0 / 558 |
| capitalised mid-sentence | 42% | 0 / 558 |
| acronym (all-caps, len ≥ 2) | 15% | 0 / 558 |
| **all three** | — | **0 / 558** |

The gazetteer holds **6,609 tokens** compiled by
`scripts/build_gazetteer.py` from the project's own NER pattern files
(villages, institutions, politicians, parties, state lists) — the author's
own data, so no external licence applies. It is positive recognition rather
than a guess, which makes it the only rule that works **sentence-initially**,
where capitalisation carries no information at all: `Sohra ka dei…` is now
recognised instead of flagged. Its 312 collisions with ordinary Khasi words
(`bah`, `bat`, `blei`, `dawa`) are inert — the list is consulted only for
words the gate has already rejected, and those are accepted by the lexicon
first.

Measured effect on untouched corpus text:

| | flagged | sentences with a flag |
|---|---|---|
| before | 271 / 3,294 tokens (8.23%) | 127 / 200 |
| capitalisation + acronym | 97 / 3,294 (2.94%) | 74 / 200 |
| **+ gazetteer** | **92 / 3,294 (2.79%)** | **73 / 200** |

Nothing is deleted. Withheld tokens are returned on `TextResult.skipped`
with the suggestion that *would* have been offered, so a caller can surface
them; they are only kept out of `corrections`, which is what gets applied.
`--check-names` on the CLI, `skip_foreign=False` in the library, or
`{"skip_foreign": false}` on the API restores the old behaviour.

**The honest limitation.** That "0 / 558" is partly an artefact — every
benchmark error was injected into a lowercased word, so no capitalised
misspelling was ever there to miss. Suppression does create a real blind
spot: 12.6% of running text is mid-sentence capitalised and 1.7% is
acronyms, so **14.3% of tokens are no longer checked**. What makes that
acceptable is not the benchmark but what the checker could do with those
tokens anyway. There is no Khasi proper-noun lexicon, so `Meghlaya` and
`Meghalaya` are equally absent from it and equally unrankable. Of the 414
capitalised tokens in the sample only 114 were flagged, and those produced
`East -> pat`. The capability given up is one the checker never had. A
gazetteer of Khasi place and personal names is the principled fix, and would
narrow this to "capitalised *and* unknown to the gazetteer".

**Measured and rejected: correcting misspelled names.** The gazetteer makes
`Meghlaya -> Meghalaya` look easy. It does not work. Injecting one edit into
300 gazetteer names, nearest-match recovered 60.3% — but across the genuine
capitalised tokens in 200 corpus sentences, **18.2% of real names sat within
one edit of a *different* gazetteer entry** and would have been rewritten
into it:

```
Light -> night      Rangbah -> nangbah     Hindu -> hindi
Bill  -> hill       Shynrang -> synrang    Kumar -> khmar
```

`Rangbah` and `Shynrang` are ordinary Khasi words. Corrupting one real name
in five to repair three in five misspelled ones is a bad trade, so the
gazetteer recognises names and never proposes them.

**A further signal, measured but not implemented.** "Is this an English word"
is the strongest of the three — 55% of flags alone, and 23 points on top of
the two above, which would take total suppression to 80% and the flag rate
from 8.23% to **1.64%** of tokens. It needs a bundled dictionary, so it is a
licensing and size decision rather than a code change. Measured against
SCOWL (`/usr/share/dict/american-english`, 72,735 entries after removing the
867 that are also Khasi words) it would have hidden 2 of the 558 benchmark
errors: `ills` and `katie`, typos of Khasi words that happen to spell
English ones. Coverage scales with list size — words of length ≤ 4 (3,974
entries) cover only 32% of flags — so there is no cheap abridged version.

### Sentence latency

Checking a sentence costs about **80 ms**. It used to cost **2,554 ms on
average, with a worst case of 18 seconds**, and the cause was not what it
looked like.

`SpellChecker.check_sentence()` called `autocorrect.Speller.check_sentence()`
to tokenise. That call runs full edit-distance-2 correction on every token to
work out which ones changed — and the engine then discarded those corrections,
recomputing suggestions from `self.suggest()` on the next line. One profiled
sentence, *"Kitei ki jaitbynriew Scheduled Tribes bad Scheduled Castes …"*,
spent 41 s inside it:

```
24 calls   41.2 s   autocorrect/__init__.py:355(<setcomp>)
9,617,988  14.2 s   typos.py:_join
5,006,129  14.6 s   typos.py:_inserts
4,503,395  13.8 s   typos.py:_replaces
```

Edit-2 generation is O(n² · alphabet²), so long unknown words — English proper
nouns above all — explode. That made latency bimodal: ~24 ms for an ordinary
Khasi sentence, seconds for one carrying a foreign name.

The call also acted as a silent **detection pre-filter**: only tokens the
Speller corrected to something *different* were ever passed to the gate, so a
word it happened to correct to itself was never gate-checked. That was a side
effect of borrowing the tokeniser, not a design decision.

It is gone. `word_regexes["kh"]` and the engine's `_WORD_PATTERN` are the same
expression, so spans are unchanged, and this brings the sentence path in line
with `SPELLER_CANDIDATES_ENABLED = False`, which had already removed the
Speller from the word path.

| | flagged on clean text | ms/sentence |
|---|---|---|
| with pre-filter | 77 / 973 tokens (7.91%) | 2,483 |
| without | 81 / 973 tokens (8.32%) | **55** |

Four more tokens per thousand are flagged — words the pre-filter had been
hiding from the gate — for a 45× speedup. Benchmark top-1 and detection are
unchanged (250/250 errors still located).

> **A caution about profiling this.** `cProfile` on the first few benchmark
> sentences reported 145 ms and hid the problem completely, because those
> sentences happen to be ordinary Khasi. The cost is concentrated in a
> minority of inputs, so an average over a small unrepresentative sample says
> nothing. The per-sentence distribution is what exposed it.

## Known limitations

Measured on the parent implementation, which this reproduces exactly:

| | |
|---|---|
| Detection | **84.4%** |
| Top-1 | **67.2%** |
| Top-5 | **80.5%** |
| Latency | **22 ms/word** (was 2,525 ms) |

All figures on the 308-item frozen benchmark. Earlier numbers in this file
came from a sample that moved with the lexicon and are not comparable.

Structural causes, in rough order of impact:

- **No corpus.** Frequencies are assigned from lexicon structure, not observed
  usage — only **11 distinct values** across 34,224 forms. Ranking depends on
  this signal.
- ~~**No transposition.**~~ **Fixed.** The phoneme recurrence still has only
  three operations, but distance is now the smaller of the phoneme distance
  and a character-level *Damerau* distance, which supplies transposition and
  stops digraph tokenisation inflating typographic slips. Worth +5.2 points
  of top-1 and +6.5 of top-5 — the largest single ranking gain here.
- **No context.** Tokens are checked independently, so real-word errors — a
  correctly spelled word in the wrong place — are undetectable by design.
- **No index.** Candidate generation scans all 34,224 forms in Python.
- ~~**Proper nouns are weak.** `shilong` ranks `shilot` above `shillong`.~~
  **No longer reproduces** (23 Sept 2026): `shillong` is first.
- ~~**No word-final coda validation.**~~ **Fixed 2026-08-26.**
  `ALLOWED_FINAL_CONSONANTS`, `_DIGRAPHS` and `_SPECIAL` were loaded from the
  data and never consulted, so no coda check ran and `bamsh` validated. Now
  enforced, rejecting 0.24% of lexicon entries (20 of 8,277) — all of which
  are OCR artefacts or banned letters (`liihv`, `kwiiig`, `sarasg`).
  Driven by the ALLOWED lists, not `forbidden_final`, which named `h` and `w`
  as forbidden while 1,216 entries end that way; those two were removed from
  the data. `VALID_INITIALS` and `DIPHTHONGS` stay unconsumed on purpose —
  the first excludes `ï` and would reject every `ïa-` word. Reasons are
  recorded in `phonology.py`.
- **Phase 4 over-accepts.** The solid-compound splitter reports `waleng` as a
  `compound_fusion` and accepts it. Same class of problem as the assimilation
  bug fixed below, different phase; not yet addressed.

### Fixed here, not in the parent

Two bugs were found and fixed in this copy after extraction. **Neither is
fixed upstream** — port them deliberately if you want them there.

1. **Reverse assimilation ignored gemination** (`assimilation.py`). `pyn-` +
   l-initial root geminates (`pyn-` + `lait` → `pyl|lait`), but the reverse
   rule only checked that a word starts with `pyl`, then treated the tail as
   the root. `pyleng` reconstructed as `pyn-` + `eng`; since `eng` is a real
   word the parse was confirmed and the misspelling accepted. The remainder
   must now start with the root's own `l`. Also affected `wal`-initial words.
2. **Ranking dropped the shared-prefix bonus** (`spell_checker.py`).
   `_levenshtein_suggestions` computes a bonus for shared word-initial
   phonemes, but the final merge discarded it, so candidates tying on both
   distance and frequency fell through to alphabetical order — `pleng` beat
   `pylleng` on the letter p. The bonus is now applied at merge time.

Neither moved the synthetic-probe numbers (77.9 / 53.2 / 67.5, unchanged),
which says more about the probe than about the fixes: random single-edit
corruptions of random lexicon words rarely reproduce either condition. It is
a concrete argument for a benchmark built from authentic errors.

## Relationship to the parent project

Extracted from `project/khasi_nlp_v12_affix_complete/`. The parent is
**unmodified** — this is an independent copy, not a move, and it has its own
copy of the lexicon.

The two will drift. If you fix something in `khasi_engine/` here that also
matters there, port it deliberately; nothing keeps them in sync.

## Licence

Dual, inherited from the parent:

- **Code** — MIT. See `LICENSE`.
- **Lexicon** (`data/khasi_db.json`) — **All Rights Reserved · Permission
  Required**. See `LICENSE-DATA`. Redistribution, ML training, bulk extraction
  and commercial integration need prior written permission.

**Redistributing this repository means redistributing the lexicon.** Settle
that with the rights holders before publishing it. Contact
`ranslyh@gmail.com`.

## Linguistic sources

The phonological and morphological rules are page-cited to primary grammars,
chiefly Badaplin War (2001), *Ki Sawa Bad Ki Dur Kyntien Jong Ka Ktien Khasi*,
with the lexicon deriving largely from E. Bars (1973), *Khasi–English
Dictionary*. Rules live in `data/khasi_db.json`, not in Python, so a linguist
can correct the grammar without touching code.

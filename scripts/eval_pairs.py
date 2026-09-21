"""
eval_pairs.py — evaluate the checker against a wrong/correct word-pair CSV.

    python3 scripts/eval_pairs.py [n_sample] [path/to/pairs.csv]

Written for khasi_test_pairs_v2.csv (116,270 pairs, 23,254 distinct targets).
That file is SYNTHETIC: exactly five corruptions per target, 99.1% single-edit,
and 42.5% of the misspellings contain characters no Khasi writer produces
(the Latin-1 accented vowels, c f q v x z, one Cyrillic yi). It is not an
authentic error corpus and results from it must not be reported as one.

What it measures that tests/spell_benchmark.json cannot
------------------------------------------------------
The frozen benchmark corrupts lexicon words, so the gold answer is in the
candidate search space 100% of the time and top-k measures ranking alone.
Here only 35.9% of targets are reachable, so the run also measures what
happens when the intended word is simply absent from the lexicon — and that
turns out to dominate every aggregate figure.

Results are therefore reported split four ways, and the aggregate is close to
meaningless on its own:

  * gold reachable / not reachable  — the coverage axis
  * misspelling legal / illegal     — illegal ones are free detections

Output is one JSON object with a per-item record list, so any partition can
be recomputed without re-running the 48 ms/item measurement.
"""
import contextlib, csv, collections, json, math, random, sys, time
from pathlib import Path
ROOT=Path("/home/rans/Desktop/MorpSpeller/khasi-spellchecker"); sys.path.insert(0,str(ROOT))
CSVP=Path(sys.argv[2]) if len(sys.argv)>2 else Path(
    "/home/rans/Desktop/MorpSpeller/khasi_test_pairs_v2.csv")
KHASI=set("abdeghijklmnoprstuwy'-ïñáéíóúý")
N_SAMPLE=int(sys.argv[1]) if len(sys.argv)>1 else 12000

rows=list(csv.DictReader(open(CSVP,encoding="utf-8")))
with contextlib.redirect_stdout(sys.stderr):
    from khasi_spell import KhasiSpeller
    sp=KhasiSpeller(eager=True)
db=sp.analyser.db
surf=set(db.all_surface_forms())

# ---- coverage of the target vocabulary -----------------------------------
targets=sorted({r['correct_word'] for r in rows})
cov={"targets":len(targets),
     "in_search_space":sum(1 for t in targets if t.lower() in surf),
     "is_known":sum(1 for t in targets if db.is_known(t.lower())),
     "is_attested":sum(1 for t in targets if db.is_attested(t.lower()))}
print(json.dumps({"coverage":cov}), file=sys.stderr)

def lev(a,b):
    if a==b: return 0
    prev=list(range(len(b)+1))
    for i,ca in enumerate(a,1):
        cur=[i]+[0]*len(b)
        for j,cb in enumerate(b,1):
            cur[j]=min(prev[j]+1,cur[j-1]+1,prev[j-1]+(ca!=cb))
        prev=cur
    return prev[-1]

random.seed(42)
sample=random.sample(rows,min(N_SAMPLE,len(rows)))
recs=[]
t0=time.time()
for i,r in enumerate(sample):
    w,g=r['wrong_word'],r['correct_word']
    res=sp.check(w)
    sug=sp.suggest(w,n=10) if not res.is_correct else []
    rank=sug.index(g)+1 if g in sug else 0
    recs.append({"w":w,"g":g,"flag":not res.is_correct,"gate":res.gate,"rank":rank,
                 "legal":all(c in KHASI for c in w.lower()),
                 "gold_reachable":g.lower() in surf,
                 "d":lev(w.lower(),g.lower()),
                 "top3":sug[:3]})
    if (i+1)%1000==0:
        print(f"  {i+1}/{len(sample)}  {(time.time()-t0)/(i+1)*1000:.0f} ms/item", file=sys.stderr)
json.dump({"coverage":cov,"n":len(recs),"records":recs},
          open("/tmp/claude-1000/-home-rans-Desktop-MorpSpeller/eefe74a3-329c-4ab8-8fca-7473e5f465d5/scratchpad/rg/pairs_raw.json","w"),
          ensure_ascii=False)
print("done", len(recs), f"{time.time()-t0:.0f}s", file=sys.stderr)

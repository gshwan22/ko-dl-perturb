#!/usr/bin/env python3
"""Training-corpus relation profiles (5 conditions, 5k-doc sample, same parser & relation set as outputs)."""
import json, random, sys, os, numpy as np, stanza
random.seed(12345)
N = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
BASE = os.path.expanduser("~/DL-paper/data")
CONDS = {
  "nu0_original": "H/data/C0_original_30k.train.jsonl",
  "nu1_gpt_dlmin": "H/data/C1_gpt_dlmin_30k.train.jsonl",
  "pi-_algo_dlmin": "H/data/C2_algo_dlmin_30k.train.jsonl",
  "pi+_algo_dlmax": "H/data/C3_algo_dlmax_30k.train.jsonl",
  "pi0_algo_dlneu": "H/data/C4_algo_dlneu_30k.train.jsonl",
}
RELS = ["nsubj","obj","obl","nmod","amod","advmod","acl","advcl","ccomp","conj","dislocated","nummod","appos"]
OUT = "H/relation_analysis/corpus_profiles_30k.json"

def turns(r):
    u = next((m["content"] for m in r["messages"] if m["role"] == "user"), "")
    a = next((m["content"] for m in r["messages"] if m["role"] == "assistant"), "")
    return u, a

# sample by user-turn key so all conditions align on the same documents
orig = [json.loads(l) for l in open(CONDS["nu0_original"])]
idx = sorted(random.sample(range(len(orig)), N))
keys = {turns(orig[i])[0]: i for i in idx}
nlp = stanza.Pipeline("ko", processors="tokenize,pos,lemma,depparse", use_gpu=True, verbose=False)

def feats(text):
    doc = nlp(text); dists = {}; alld = []
    for s in doc.sentences:
        if len(s.words) < 5: continue
        for w in s.words:
            if w.deprel in ("punct", "root") or w.head == 0: continue
            d = abs(w.id - w.head); alld.append(d); dists.setdefault(w.deprel, []).append(d)
    if not alld: return None
    return {"mean_dist": float(np.mean(alld)), "rel_mean_dist": {r: float(np.mean(v)) for r, v in dists.items()}}

results = json.load(open(OUT)) if os.path.exists(OUT) else {}
for cond, path in CONDS.items():
    if cond in results: print("skip", cond); continue
    got = {}
    want = set(idx); misaligned = 0
    for i, line in enumerate(open(path)):
        if i not in want: continue
        r = json.loads(line); u, a = turns(r)
        if u != turns(orig[i])[0]:
            misaligned += 1; continue        # 행 정렬 깨진 문서는 제외하고 집계
        f = feats(a) if a.strip() else None
        if f: got[str(i)] = f
        if i >= idx[-1]: break
    print(f"  {cond}: misaligned rows = {misaligned}")
    results[cond] = got; json.dump(results, open(OUT, "w"), ensure_ascii=False)
    print(f"✓ {cond}: {len(got)} docs")

ref = results["nu0_original"]
print(f"\n{'cond':<16}{'docs':>6}{'MDD':>7}{'ΔMDD%':>8}{'σ(ℓ_r)':>8}{'pcorr':>8}{'r(ℓ,Δℓ)':>9}")
for cond, got in results.items():
    common = [i for i in got if i in ref]
    mdd = np.mean([got[i]["mean_dist"] for i in common]); mdd0 = np.mean([ref[i]["mean_dist"] for i in common])
    ell = {r: np.mean([got[i]["rel_mean_dist"][r] for i in common if r in got[i]["rel_mean_dist"]]) for r in RELS}
    ellr = {r: np.mean([ref[i]["rel_mean_dist"][r] for i in common if r in ref[i]["rel_mean_dist"]]) for r in RELS}
    delta = {r: np.mean([got[i]["rel_mean_dist"][r]-ref[i]["rel_mean_dist"][r] for i in common if r in got[i]["rel_mean_dist"] and r in ref[i]["rel_mean_dist"]]) for r in RELS}
    x = [ellr[r] for r in RELS]; y = [ell[r] for r in RELS]; dl = [delta[r] for r in RELS]
    print(f"{cond:<16}{len(common):>6}{mdd:>7.2f}{(mdd-mdd0)/mdd0*100:>8.1f}{np.std(y):>8.2f}{np.corrcoef(x,y)[0,1]:>8.3f}{np.corrcoef(x,dl)[0,1]:>9.2f}")

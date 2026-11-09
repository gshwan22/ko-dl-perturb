#!/usr/bin/env python3
"""Relation-conditioned profile statistics from *_features.json.
   ell_r: mean over docs (containing r) of per-doc mean distance of relation r
   delta_r: mean over docs containing r in BOTH output and reference of (out - ref)
   r(ell_r^ref, delta_r), profile corr(ell_r^out, ell_r^ref), sigma(ell_r^out), normalized variants"""
import json, glob, os, sys, math, re
from collections import defaultdict
import numpy as np

PARSE_DIR = sys.argv[1] if len(sys.argv) > 1 else "H/relation_analysis/parses"
MIN_DOCS = int(sys.argv[2]) if len(sys.argv) > 2 else 30   # relation support threshold
WEIGHTED = len(sys.argv) > 3 and sys.argv[3] == "w"
TOPK = int(sys.argv[4]) if len(sys.argv) > 4 else 0
RELSET = sys.argv[5].split(",") if len(sys.argv) > 5 else None   # 0 = use MIN_DOCS; else top-K relations by reference frequency
ref = json.load(open(f"{PARSE_DIR}/reference_features.json"))

def pearson(x, y):
    x, y = np.array(x, float), np.array(y, float)
    if len(x) < 3 or x.std() == 0 or y.std() == 0: return float('nan')
    return float(np.corrcoef(x, y)[0, 1])

def stats_for(feature_files):
    out_by_rel = defaultdict(list); delta_by_rel = defaultdict(list); ref_by_rel = defaultdict(list)
    n_docs = 0
    for f in feature_files:
        feats = json.load(open(f))
        for doc_id, d in feats.items():
            if doc_id not in ref or d is None or ref[doc_id] is None: continue
            n_docs += 1
            rmd, rref = d["rel_mean_dist"], ref[doc_id]["rel_mean_dist"]
            for r, v in rmd.items():
                w = d.get("rel_counts", {}).get(r, 1) if WEIGHTED else 1
                out_by_rel[r].extend([v] * w)
                if r in rref: delta_by_rel[r].extend([v - rref[r]] * w)
    for doc_id, d in ref.items():
        if d is None: continue
        for r, v in d["rel_mean_dist"].items():
            ref_by_rel[r].extend([v] * (d.get("rel_counts", {}).get(r, 1) if WEIGHTED else 1))
    if RELSET:
        rels = [r for r in RELSET if r in delta_by_rel and len(delta_by_rel[r]) >= 30]
    elif TOPK:
        rels = [r for r, _ in sorted(ref_by_rel.items(), key=lambda kv: -len(kv[1])) if r in delta_by_rel][:TOPK]
    else:
        rels = [r for r in delta_by_rel if len(delta_by_rel[r]) >= MIN_DOCS and len(ref_by_rel[r]) >= MIN_DOCS]
    ell_out = {r: np.mean(out_by_rel[r]) for r in rels}
    ell_ref = {r: np.mean(ref_by_rel[r]) for r in rels}
    delta = {r: np.mean(delta_by_rel[r]) for r in rels}
    xs = [ell_ref[r] for r in rels]
    r_corr = pearson(xs, [delta[r] for r in rels])
    p_corr = pearson([ell_out[r] for r in rels], xs)
    sig = float(np.std([ell_out[r] for r in rels]))
    mean_ell = float(np.mean([ell_out[r] for r in rels]))
    sig_ref = float(np.std(xs))
    return dict(n_docs=n_docs, n_rel=len(rels), r=r_corr, pcorr=p_corr, sigma=sig,
                norm_cv=sig/mean_ell, norm_ref=sig/sig_ref, delta=delta, ell_out=ell_out, ell_ref=ell_ref)

conds = sorted({re.sub(r"_[^_]+_features\.json$", "", os.path.basename(f))
                for f in glob.glob(f"{PARSE_DIR}/C*_features.json")})
print(f"[{PARSE_DIR}] MIN_DOCS={MIN_DOCS} weighted={WEIGHTED} topK={TOPK}")
print(f"{'cond':<16}{'docs':>6}{'rels':>5}{'r(l,Δl)':>9}{'pcorr':>8}{'σ':>7}{'σ/mean':>8}{'σ/σref':>8}")
pooled = {}
for c in conds:
    files = sorted(glob.glob(f"{PARSE_DIR}/{c}_*_features.json"))
    s = stats_for(files); pooled[c] = s
    print(f"{c:<16}{s['n_docs']:>6}{s['n_rel']:>5}{s['r']:>9.2f}{s['pcorr']:>8.3f}{s['sigma']:>7.2f}{s['norm_cv']:>8.3f}{s['norm_ref']:>8.3f}")
    for f in files:   # per-model
        m = re.sub(rf"^{c}_(.+)_features\.json$", r"\1", os.path.basename(f))
        sm = stats_for([f])
        print(f"   {m:<13}{sm['n_docs']:>6}{sm['n_rel']:>5}{sm['r']:>9.2f}{sm['pcorr']:>8.3f}{sm['sigma']:>7.2f}")
ref_rels = [r for r in pooled[conds[0]]['ell_ref']]
print(f"reference σ(ℓ_r) over same rels: {np.std([pooled[conds[0]]['ell_ref'][r] for r in ref_rels]):.2f}")
# selected relations (paper §3.2) for pi+
if "C3_algo_dlmax" in pooled:
    d = pooled["C3_algo_dlmax"]["delta"]
    print("C3 Δℓ_r:", {k: round(d[k], 2) for k in ["dislocated", "nummod", "obj", "conj"] if k in d})

#!/usr/bin/env python3
"""Run on spark-f9f6 from ~/DL-topology/paper_lre:
    python3 scripts/extract_appendix.py
Writes
  tables/tab2_candidates.txt   candidate documents (present in all five 30k corpora) with their five versions
  tables/tabC_threshold.tex    Appendix C: collapse rate at thresholds 0.2 / 0.3 / 0.4, mean over models
  tables/tabD_examples.txt     Appendix D: collapsed outputs from pi+ with the loop onset located
Prints a smoke summary first and aborts if any input is missing."""
import json, os, re, sys, glob
from collections import Counter

HOME = os.path.expanduser("~")
H = f"{HOME}/DL-topology/H"
OUT = "tables"
CONDS = [("C0_original", r"\nuO"), ("C1_gpt_dlmin", r"\nuI"), ("C4_algo_dlneu", r"\piZ"),
         ("C2_algo_dlmin", r"\piM"), ("C3_algo_dlmax", r"\piP")]
MODELS = ["gemma3-4b", "llama31-8b", "mistral-7b", "qwen3-8b", "skt-ax4-light", "exaone35-7.8b"]
THRESH = [0.2, 0.3, 0.4]


def need(p):
    if not os.path.exists(p):
        sys.exit(f"[ABORT] missing: {p}")
    return p


def answer(rec):
    for m in rec["messages"]:
        if m["role"] == "assistant":
            return m["content"]
    return None


def first_sentence(t):
    m = re.search(r"^(.+?(?:다|요|함|음)\.)(\s|$)", t.strip())
    return m.group(1) if m else t.strip().split("\n")[0]


def load_corpus(cond):
    d = {}
    for line in open(need(f"{H}/data/{cond}_30k.train.jsonl")):
        r = json.loads(line)
        d[int(r["meta"]["doc_id"])] = answer(r)
    return d


def uniq_ratio(text):
    tk = text.split()
    return (len(set(tk)) / len(tk)) if tk else 1.0


def loop_onset(text, n=8):
    """index (in chars) where the first repeated n-gram of whitespace tokens begins its second occurrence"""
    tk = text.split()
    seen = {}
    pos, idx = 0, []
    for t in tk:
        pos = text.find(t, pos); idx.append(pos); pos += len(t)
    for i in range(len(tk) - n + 1):
        g = tuple(tk[i:i + n])
        if g in seen:
            return idx[i]
        seen[g] = i
    return None


# ---------- smoke ----------
print("[smoke] corpora:", [os.path.exists(f"{H}/data/{c}_30k.train.jsonl") for c, _ in CONDS])
pred_files = {c: sorted(glob.glob(f"{H}/eval/predictions/{c}_*.jsonl")) for c, _ in CONDS}
for c, _ in CONDS:
    print(f"[smoke] {c}: {len(pred_files[c])} prediction files")
    if not pred_files[c]:
        sys.exit("[ABORT] no prediction files")

# ---------- Table 2 candidates ----------
corp = {c: load_corpus(c) for c, _ in CONDS}
common = set.intersection(*[set(v) for v in corp.values()])
print(f"[tab2] docs present in all five corpora: {len(common)}")
cands = []
for did in common:
    s0 = first_sentence(corp["C0_original"][did])
    n = len(s0.split())
    if 12 <= n <= 22:
        versions = {c: first_sentence(corp[c][did]) for c, _ in CONDS}
        if len({v for v in versions.values()}) == 5:  # all five differ
            cands.append((n, did, versions))
cands.sort()
with open(f"{OUT}/tab2_candidates.txt", "w") as f:
    for n, did, versions in cands[:8]:
        f.write(f"=== doc_id {did}  ({n} eojeol in nu0 first sentence)\n")
        for c, sym in CONDS:
            f.write(f"{sym:<5} {versions[c]}\n")
        f.write("\n")
print(f"[tab2] {min(8, len(cands))} candidates written to {OUT}/tab2_candidates.txt")

# ---------- Appendix C: thresholds ----------
rows, permodel = [], []
for c, sym in CONDS:
    per = {}
    for f in pred_files[c]:
        model = os.path.basename(f)[len(c) + 1:-6]
        ratios = [uniq_ratio(json.loads(l)["prediction"]) for l in open(f) if l.strip()]
        assert len(ratios) >= 990, f"{f}: {len(ratios)} outputs"
        per[model] = [100 * sum(r < t for r in ratios) / len(ratios) for t in THRESH]
    mean = [sum(v[i] for v in per.values()) / len(per) for i in range(3)]
    rows.append((sym, len(per), mean))
    permodel.append((sym, per))
    print(f"[tabC] {sym:<5} n_models={len(per)}  " + "  ".join(f"t{t}: {m:5.1f}" for t, m in zip(THRESH, mean)))
with open(f"{OUT}/tabC_threshold.tex", "w") as f:
    f.write(r"""\begin{table}[h]
\caption{Collapse rate (\%) under three unique-token-ratio thresholds, mean over models (\piZ{} over five models, the others over six). The ordering of conditions is the same at every threshold.}\label{tab:threshold}
\begin{tabular}{@{}lrrr@{}}
\toprule
Condition & $<0.2$ & $<0.3$ & $<0.4$ \\
\midrule
""")
    for sym, n, mean in rows:
        f.write(f"{sym:<5} & {mean[0]:.1f} & {mean[1]:.1f} & {mean[2]:.1f} \\\\\n")
    f.write(r"""\botrule
\end{tabular}
\end{table}
""")
print(f"[tabC] written {OUT}/tabC_threshold.tex")

# ---------- Appendix D: two collapsed examples from pi+ ----------
ex = []
for f in pred_files["C3_algo_dlmax"]:
    model = os.path.basename(f)[len("C3_algo_dlmax") + 1:-6]
    for l in open(f):
        r = json.loads(l)
        p = r["prediction"]
        # strip leaked chat-template prefix "user\n<question>\n" if present
        q = r["input"].strip()
        if p.lstrip().startswith("user") and q in p[:len(q) + 20]:
            p = p[p.find(q) + len(q):].lstrip("\n ")
        if uniq_ratio(p) < 0.3 and len(p) > 300:
            on = loop_onset(p)
            if on and 60 < on < 400:
                ex.append((on, model, r["doc_id"], q, p))
ex.sort(key=lambda x: -x[0])  # longest normal opening first
chosen, used = [], set()
for on, model, did, q, p in ex:
    if model not in used:
        chosen.append((on, model, did, q, p)); used.add(model)
    if len(chosen) == 3:
        break
with open(f"{OUT}/tabD_examples.txt", "w") as f:
    for on, model, did, q, p in chosen:
        f.write(f"=== {model}  doc_id {did}  loop onset at char {on}\n")
        f.write(f"Q: {q}\n")
        f.write("NORMAL : " + p[:on].strip() + "\n")
        f.write("LOOP   : " + p[on:on + 260].strip() + " ...\n\n")
print(f"[tabD] {len(chosen)} examples written to {OUT}/tabD_examples.txt  (candidates found: {len(ex)})")
print("done")

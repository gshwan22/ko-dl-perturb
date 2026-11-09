#!/usr/bin/env python3
"""Figures 1-3 for the LRE manuscript. Run on spark-f9f6 from ~/DL-topology/paper_lre:
    <python with numpy+matplotlib> figures/make_figures.py
Writes figures/fig1_dose_response.pdf, fig2_reorganization.pdf, fig3_transfer.pdf
and prints every plotted number so it can be checked against Tables 3-5.
Conventions: PDF only, no in-figure titles, axis labels are symbols, sans-serif 8 pt,
widths 84 mm (single) / 174 mm (double), models distinguished by marker as well as colour."""
import json, glob, os, re, sys
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HOME = os.path.expanduser("~")
TOPO = f"{HOME}/DL-topology"
DLP = f"{HOME}/DL-paper"
OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)))
PARSES = f"{TOPO}/H/relation_analysis/parses"
CORPUS_JSON = f"{TOPO}/H/relation_analysis/corpus_profiles_30k.json"
RELS = ["nsubj", "obj", "obl", "nmod", "amod", "advmod", "acl", "advcl",
        "ccomp", "conj", "dislocated", "nummod", "appos"]
MODELS = ["gemma3-4b", "llama31-8b", "mistral-7b", "qwen3-8b", "skt-ax4-light"]
MARK = dict(zip(MODELS, ["o", "s", "^", "D", "v"]))
COL = dict(zip(MODELS, ["#1b9e77", "#d95f02", "#7570b3", "#e7298a", "#66a61e"]))
LEVELS = ["0.0", "0.2", "0.4", "0.6", "0.8", "1.0"]
KEYS = ["prediction", "output", "generated", "response", "pred", "text"]
MM = 1 / 25.4

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.sans-serif": ["Arial", "Helvetica", "Liberation Sans", "DejaVu Sans"],
    "font.size": 8, "axes.labelsize": 9, "legend.fontsize": 7,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "pdf.fonttype": 42, "axes.spines.top": False, "axes.spines.right": False,
})


def need(path):
    if not os.path.exists(path):
        sys.exit(f"[ABORT] missing input: {path}")
    return path


def collapse_pct(path):
    n = c = 0
    for line in open(need(path)):
        line = line.strip()
        if not line:
            continue
        r = json.loads(line)
        t = next((r[k] for k in KEYS if isinstance(r.get(k), str)), "")
        tk = t.split()
        if tk:
            n += 1
            c += len(set(tk)) / len(tk) < 0.3
    assert n >= 990, f"{path}: only {n} non-empty outputs"
    return 100.0 * c / n


# ---------------- Figure 1: dose response ----------------
def fig1():
    x = [float(p) for p in LEVELS]
    pi_plus = {m: [collapse_pct(f"{DLP}/evaluation/algo_dlmax/predictions_{m}_p{p}.jsonl") for p in LEVELS] for m in MODELS}
    nu_one = {m: [collapse_pct(f"{DLP}/evaluation/gpt_dlmin/predictions_{m}_p{p}.jsonl") for p in LEVELS] for m in MODELS}
    print("[fig1] collapse %  (pi+ / nu1)  p =", LEVELS)
    for m in MODELS:
        print(f"  {m:<14} pi+ {['%.1f' % v for v in pi_plus[m]]}   nu1 {['%.1f' % v for v in nu_one[m]]}")
    fig, ax = plt.subplots(figsize=(84 * MM, 62 * MM))
    for m in MODELS:
        ax.plot(x, pi_plus[m], marker=MARK[m], color=COL[m], ms=4, lw=1.1, label=m)
    nu_mean = np.mean([nu_one[m] for m in MODELS], axis=0)
    ax.plot(x, nu_mean, color="0.45", ls="--", lw=1.1, marker="x", ms=4, label=r"$\nu_1$ (mean)")
    ax.set_xlabel(r"$p$")
    ax.set_ylabel(r"$C$ (%)")
    ax.set_xticks(x)
    ax.set_ylim(bottom=-0.3)
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig1_dose_response.pdf")
    plt.close(fig)


# ---------------- shared: relation profiles from parsed features ----------------
REF = json.load(open(need(f"{PARSES}/reference_features.json")))


def ref_ell():
    by = {r: [] for r in RELS}
    for d in REF.values():
        if d is None:
            continue
        for r, v in d["rel_mean_dist"].items():
            if r in by:
                by[r].append(v)
    return {r: float(np.mean(by[r])) for r in RELS if by[r]}


def profile(cond, models=None):
    files = sorted(glob.glob(f"{PARSES}/{cond}_*_features.json"))
    if models is not None:
        files = [f for f in files if any(f.endswith(f"{cond}_{m}_features.json") for m in models)]
    assert files, f"no feature files for {cond}"
    out = {r: [] for r in RELS}
    dlt = {r: [] for r in RELS}
    for f in files:
        for doc, d in json.load(open(f)).items():
            if d is None or doc not in REF or REF[doc] is None:
                continue
            rr = REF[doc]["rel_mean_dist"]
            for r, v in d["rel_mean_dist"].items():
                if r not in out:
                    continue
                out[r].append(v)
                if r in rr:
                    dlt[r].append(v - rr[r])
    rels = [r for r in RELS if len(dlt[r]) >= 30]
    ell = {r: float(np.mean(out[r])) for r in rels}
    delta = {r: float(np.mean(dlt[r])) for r in rels}
    return ell, delta


# ---------------- Figure 2: reorganization ----------------
def fig2():
    eref = ref_ell()
    fig, axes = plt.subplots(1, 2, figsize=(174 * MM, 70 * MM), sharey=True)
    panels = [("C3_algo_dlmax", [r"$\pi^{+}$"]), (None, [r"$\nu_0$", r"$\nu_1$"])]
    print("[fig2] Delta ell_r by relation")
    # left: pi+
    ell, delta = profile("C3_algo_dlmax")
    ax = axes[0]
    xs = [eref[r] for r in delta]; ys = [delta[r] for r in delta]
    ax.scatter(xs, ys, s=16, color="#d95f02", zorder=3)
    for r in delta:
        ax.annotate(r, (eref[r], delta[r]), fontsize=6, xytext=(3, 2), textcoords="offset points")
    k, b = np.polyfit(xs, ys, 1)
    gx = np.linspace(min(xs), max(xs), 50)
    ax.plot(gx, k * gx + b, color="0.5", lw=0.8, ls=":")
    print("  pi+ :", {r: round(delta[r], 2) for r in delta}, f" r={np.corrcoef(xs, ys)[0, 1]:.2f}")
    ax.text(0.97, 0.95, r"$\pi^{+}$", transform=ax.transAxes, ha="right", va="top", fontsize=9)
    # right: natural
    ax = axes[1]
    for cond, lab, mk, col in [("C0_original", r"$\nu_0$", "o", "0.2"), ("C1_gpt_dlmin", r"$\nu_1$", "s", "#1b9e77")]:
        ell, delta = profile(cond)
        xs = [eref[r] for r in delta]; ys = [delta[r] for r in delta]
        ax.scatter(xs, ys, s=16, marker=mk, color=col, label=lab, zorder=3, facecolors="none" if mk == "s" else col)
        print(f"  {cond}:", {r: round(delta[r], 2) for r in delta}, f" r={np.corrcoef(xs, ys)[0, 1]:.2f}")
    ax.legend(frameon=False, loc="upper right")
    for ax in axes:
        ax.axhline(0, color="0.6", lw=0.6)
        ax.set_xlabel(r"$\ell_r$")
    axes[0].set_ylabel(r"$\Delta\ell_r$")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig2_reorganization.pdf")
    plt.close(fig)


# ---------------- Figure 3: corpus -> output transfer of sigma(ell_r) ----------------
def corpus_sigma():
    cp = json.load(open(need(CORPUS_JSON)))
    ref = cp["nu0_original"]
    res = {}
    for cond, got in cp.items():
        common = [i for i in got if i in ref]
        ell = []
        for r in RELS:
            v = [got[i]["rel_mean_dist"][r] for i in common if r in got[i]["rel_mean_dist"]]
            if v:
                ell.append(np.mean(v))
        res[cond] = float(np.std(ell))
    return res


def fig3():
    csig = corpus_sigma()
    pairs = [("nu0_original", "C0_original", r"$\nu_0$"), ("nu1_gpt_dlmin", "C1_gpt_dlmin", r"$\nu_1$"),
             ("pi0_algo_dlneu", "C4_algo_dlneu", r"$\pi^{0}$"), ("pi-_algo_dlmin", "C2_algo_dlmin", r"$\pi^{-}$"),
             ("pi+_algo_dlmax", "C3_algo_dlmax", r"$\pi^{+}$")]
    fig, ax = plt.subplots(figsize=(84 * MM, 70 * MM))
    print("[fig3] sigma(ell_r): corpus -> output (pooled; per-model in grey)")
    xs, ys = [], []
    for ck, ok, lab in pairs:
        ell, _ = profile(ok)
        so = float(np.std(list(ell.values())))
        xs.append(csig[ck]); ys.append(so)
        for m in MODELS:
            if glob.glob(f"{PARSES}/{ok}_{m}_features.json"):
                em, _ = profile(ok, [m])
                ax.scatter(csig[ck], np.std(list(em.values())), s=8, color="0.75", zorder=2)
        ax.scatter(csig[ck], so, s=30, color="black", zorder=3)
        ax.annotate(lab, (csig[ck], so), xytext=(5, -2), textcoords="offset points", fontsize=9)
        print(f"  {lab:<12} corpus {csig[ck]:.2f}  output {so:.2f}")
    lo, hi = min(xs + ys) - 0.2, max(xs + ys) + 0.2
    ax.plot([lo, hi], [lo, hi], color="0.6", lw=0.6, ls=":")
    ax.set_xlim(lo, hi); ax.set_ylim(lo, hi)
    ax.set_xlabel(r"$\sigma_{\mathrm{T}}$")
    ax.set_ylabel(r"$\sigma_{\mathrm{G}}$")
    print(f"  Pearson r (pooled, 5 conditions) = {np.corrcoef(xs, ys)[0, 1]:.3f}")
    fig.tight_layout()
    fig.savefig(f"{OUT}/fig3_transfer.pdf")
    plt.close(fig)


if __name__ == "__main__":
    which = sys.argv[1:] or ["1", "2", "3"]
    if "1" in which: fig1()
    if "2" in which: fig2()
    if "3" in which: fig3()
    print("done →", OUT)

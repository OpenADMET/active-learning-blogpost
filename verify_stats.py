"""
verify_stats.py — one-to-one verification of all statistical claims in blogpost.md.

Each labelled block corresponds to a specific comparison made in the text.
Run with:  python verify_stats.py

Requires: results/ directory with run_*.pkl files and setup_*.pkl files.
"""

import pickle
import glob
import numpy as np
from scipy import stats
from scipy.stats import spearmanr


# ── helpers ──────────────────────────────────────────────────────────────────

def load_metric(results_dir, strategy, metric, target_n):
    split = "predefined" if "asap" in results_dir else "random"
    vals = []
    for f in sorted(glob.glob(f"{results_dir}/run_{split}_{strategy}_seed*.pkl")):
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        h = d["result"]["history"]
        step = min(h, key=lambda s: abs(s["n_labeled"] - target_n))
        vals.append(step[metric])
    assert vals, f"No files found: {results_dir}/run_{split}_{strategy}_seed*.pkl"
    return np.array(vals)


def load_sigma_rho(results_dir, strategy, target_n):
    """Spearman rho between ensemble sigma and absolute prediction error on test set."""
    split = "predefined" if "asap" in results_dir else "random"
    setup_file = f"{results_dir}/setup_{split}.pkl"
    with open(setup_file, "rb") as f:
        setup = pickle.load(f)
    y_true = setup["df_test"]["pEC50"].values
    rhos = []
    for f in sorted(glob.glob(f"{results_dir}/run_{split}_{strategy}_seed*.pkl")):
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        h = d["result"]["history"]
        step = min(h, key=lambda s: abs(s["n_labeled"] - target_n))
        y_pred = np.array(step["y_test_pred"])
        y_std  = np.array(step["y_test_std"])
        abs_err = np.abs(y_pred - y_true)
        rhos.append(spearmanr(y_std, abs_err).statistic)
    return np.array(rhos)


def report(label, a, b, paired=True):
    t, p = stats.ttest_rel(a, b) if paired else stats.ttest_ind(a, b)
    sig = "SIGNIFICANT *" if p < 0.05 else "not significant"
    print(f"\n{'─'*60}")
    print(f"  {label}")
    print(f"  A: mean={a.mean():.4f}  SD={a.std(ddof=1):.4f}  values={np.round(a,4).tolist()}")
    print(f"  B: mean={b.mean():.4f}  SD={b.std(ddof=1):.4f}  values={np.round(b,4).tolist()}")
    print(f"  diff(A−B)={a.mean()-b.mean():.4f}  t={t:.3f}  p={p:.4f}  → {sig}")


# ══════════════════════════════════════════════════════════════════════════════
print("=" * 60)
print("MODEL ACCURACY — PXR")
print("=" * 60)
print("(All at n=900 unless noted)")

# [A1] Already in text (reported when user ran it)
print("\n[A1] ChemProp: Exploitation vs Random MAE at n=900")
print("     → already in text: p=0.0007")

# [A2] Already in text
print("\n[A2] CheMeleon: Exploitation vs Random MAE at n=900")
print("     → already in text: p=0.028")

# [A3] CheMeleon vs ChemProp under Exploitation — largest advantage
print("\n[A3] CheMeleon vs ChemProp MAE under Exploitation at n=900")
print("     → text: 'significant under Exploitation (0.07 MAE units, p=0.0006)'")
report("A[CheMeleon Exploitation] vs B[ChemProp Exploitation]",
       load_metric("results/pxr_chemeleon", "Exploitation", "mae", 900),
       load_metric("results/pxr_chemprop",  "Exploitation", "mae", 900))

# [A4] CheMeleon vs ChemProp under EI — smallest advantage
print("\n[A4] CheMeleon vs ChemProp MAE under EI at n=900")
print("     → text: 'not significant under EI (p > 0.19)'")
report("A[CheMeleon EI] vs B[ChemProp EI]",
       load_metric("results/pxr_chemeleon", "EI", "mae", 900),
       load_metric("results/pxr_chemprop",  "EI", "mae", 900))

# [A5] CheMeleon vs ChemProp under Diversity — smallest advantage
print("\n[A5] CheMeleon vs ChemProp MAE under Diversity at n=900")
print("     → text: 'not significant under Diversity (p > 0.19)'")
report("A[CheMeleon Diversity] vs B[ChemProp Diversity]",
       load_metric("results/pxr_chemeleon", "Diversity", "mae", 900),
       load_metric("results/pxr_chemprop",  "Diversity", "mae", 900))

# [A6] ChEMBL warm-start initial advantage at n=0
print("\n[A6] CheMeleon+ChEMBL vs ChemProp+ChEMBL MAE at n=0 (before pool labels)")
print("     → text: '0.83 vs 0.93 (p=0.0002)'")
report("A[CheMeleon+ChEMBL n=0] vs B[ChemProp+ChEMBL n=0]",
       load_metric("results/pxr_chemeleon_chembl", "Exploitation", "mae", 0),
       load_metric("results/pxr_chemprop_chembl",  "Exploitation", "mae", 0))

# [A7] ChEMBL convergence: warm-started vs no-ChEMBL at n=200
print("\n[A7] CheMeleon+ChEMBL vs CheMeleon (no ChEMBL) MAE at n=200")
print("     → text: 'advantage largely disappears by n=200 (p=0.72)'")
report("A[CheMeleon+ChEMBL n=200] vs B[CheMeleon n=200]",
       load_metric("results/pxr_chemeleon_chembl", "Random", "mae", 200),
       load_metric("results/pxr_chemeleon",        "Random", "mae", 200))

# [A8] ChemProp Exploitation lowest Kendall tau vs Random
print("\n[A8] ChemProp Exploitation vs Random Kendall tau at n=900")
print("     → text: 'Exploitation lowest tau ~0.42 (p=0.0006 vs Random)'")
report("A[ChemProp Exploitation ktau] vs B[ChemProp Random ktau]",
       load_metric("results/pxr_chemprop", "Exploitation", "ktau", 900),
       load_metric("results/pxr_chemprop", "Random",       "ktau", 900))

# [A9] CheMeleon vs ChemProp Kendall tau under Exploitation
print("\n[A9] CheMeleon vs ChemProp Kendall tau under Exploitation at n=900")
print("     → text: CheMeleon clustered 0.49-0.50 vs ChemProp ~0.42")
report("A[CheMeleon Exploitation ktau] vs B[ChemProp Exploitation ktau]",
       load_metric("results/pxr_chemeleon", "Exploitation", "ktau", 900),
       load_metric("results/pxr_chemprop",  "Exploitation", "ktau", 900))


# ══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 60)
print("MODEL ACCURACY — MPRO")
print("=" * 60)

# [B1] CheMeleon vs ChemProp Exploitation MAE at n=200
print("\n[B1] CheMeleon vs ChemProp MAE under Exploitation at n=200")
print("     → text: '0.07 units, not significant (p=0.33)'")
report("A[CheMeleon Exploitation n=200] vs B[ChemProp Exploitation n=200]",
       load_metric("results/asap_chemeleon", "Exploitation", "mae", 200),
       load_metric("results/asap_chemprop",  "Exploitation", "mae", 200))

# [B2] CheMeleon vs ChemProp Random MAE at n=200
print("\n[B2] CheMeleon vs ChemProp MAE under Random at n=200")
print("     → text: '0.16 units, significant (p=0.0016)'")
report("A[CheMeleon Random n=200] vs B[ChemProp Random n=200]",
       load_metric("results/asap_chemeleon", "Random", "mae", 200),
       load_metric("results/asap_chemprop",  "Random", "mae", 200))

# [B3] Strategy spread within ChemProp Mpro
print("\n[B3] Strategy spread (max-min across strategies) for Mpro MAE")
print("     → text: ChemProp spread 0.17 at n=200, narrows to <0.05 by n=600; CheMeleon <0.07 throughout")
strategies = ["Exploitation", "Random", "UCB", "EI", "Exploration", "Diversity"]
for n in [200, 400, 600]:
    cp_means = {s: load_metric("results/asap_chemprop",  s, "mae", n).mean() for s in strategies}
    cm_means = {s: load_metric("results/asap_chemeleon", s, "mae", n).mean() for s in strategies}
    print(f"  n={n}: ChemProp spread={max(cp_means.values())-min(cp_means.values()):.3f} "
          f"CheMeleon spread={max(cm_means.values())-min(cm_means.values()):.3f}")

# [B4] CheMeleon vs ChemProp Kendall tau at n=400 by strategy
print("\n[B4] CheMeleon vs ChemProp Kendall tau at n=400 by strategy")
print("     → text: 'CheMeleon significantly higher under UCB, Exploration, Diversity (p<0.05)'")
for s in strategies:
    a = load_metric("results/asap_chemeleon", s, "ktau", 400)
    b = load_metric("results/asap_chemprop",  s, "ktau", 400)
    t, p = stats.ttest_rel(a, b)
    sig = "* " if p < 0.05 else "ns"
    print(f"  {s:<12}: CheMeleon={a.mean():.3f}  ChemProp={b.mean():.3f}  diff={a.mean()-b.mean():.3f}  p={p:.4f} {sig}")


# ══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 60)
print("MODEL UNCERTAINTY — PXR (sigma-error rho)")
print("=" * 60)

# [C1] ChemProp vs CheMeleon rho by strategy at n=900
print("\n[C1] ChemProp vs CheMeleon sigma-error rho at n=900 by strategy")
print("     → text: significant under Exploitation (p=0.0005) and UCB (p=0.005); others p>0.09")
for s in strategies:
    a = load_sigma_rho("results/pxr_chemprop",  s, 900)
    b = load_sigma_rho("results/pxr_chemeleon", s, 900)
    t, p = stats.ttest_rel(a, b)
    sig = "* " if p < 0.05 else "ns"
    print(f"  {s:<12}: ChemProp={a.mean():.3f}  CheMeleon={b.mean():.3f}  diff={a.mean()-b.mean():.3f}  p={p:.4f} {sig}")

# [C2] ChemProp strategy comparison rho at n=900
print("\n[C2] ChemProp rho by strategy at n=900")
print("     → text: Exploitation highest (~0.29), Exploration lowest (~0.16)")
for s in strategies:
    v = load_sigma_rho("results/pxr_chemprop", s, 900)
    print(f"  {s:<12}: mean={v.mean():.3f}  SD={v.std(ddof=1):.3f}")

# [C3] CheMeleon strategy comparison rho at n=900
print("\n[C3] CheMeleon rho by strategy at n=900")
print("     → text: Exploitation lowest (~0.11), Random/Exploration higher (~0.17)")
for s in strategies:
    v = load_sigma_rho("results/pxr_chemeleon", s, 900)
    print(f"  {s:<12}: mean={v.mean():.3f}  SD={v.std(ddof=1):.3f}")

# [C4] ChemProp Exploitation highest rho vs Random
print("\n[C4] ChemProp: Exploitation (highest) vs Random rho at n=900")
print("     → text: 'p=0.0043 vs Random'")
report("A[ChemProp Exploitation rho] vs B[ChemProp Random rho]",
       load_sigma_rho("results/pxr_chemprop", "Exploitation", 900),
       load_sigma_rho("results/pxr_chemprop", "Random",       900))

# [C5] CheMeleon Exploitation lowest rho vs Random
print("\n[C5] CheMeleon: Exploitation (lowest) vs Random rho at n=900")
print("     → text: 'p=0.027 vs Random'")
report("A[CheMeleon Exploitation rho] vs B[CheMeleon Random rho]",
       load_sigma_rho("results/pxr_chemeleon", "Exploitation", 900),
       load_sigma_rho("results/pxr_chemeleon", "Random",       900))


# ══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 60)
print("MODEL UNCERTAINTY — MPRO (sigma-error rho)")
print("=" * 60)

# [D1] CheMeleon vs ChemProp rho at n=200 — comparable, no significant differences
print("\n[D1] CheMeleon vs ChemProp sigma-error rho at n=200 by strategy")
print("     → text: 'comparable, no significant model differences'")
for s in strategies:
    a = load_sigma_rho("results/asap_chemeleon", s, 200)
    b = load_sigma_rho("results/asap_chemprop",  s, 200)
    t, p = stats.ttest_rel(a, b)
    sig = "* " if p < 0.05 else "ns"
    print(f"  {s:<12}: CheMeleon={a.mean():.3f}  ChemProp={b.mean():.3f}  diff={a.mean()-b.mean():.3f}  p={p:.4f} {sig}")

# [D2] Strategy comparison rho at n=200 on Mpro — Random/Diversity tend highest
print("\n[D2] Mpro rho by strategy at n=200 (both models)")
print("     → text: 'Random and Diversity tend to produce highest rho values'")
for model, dr in [("CheMeleon", "results/asap_chemeleon"), ("ChemProp", "results/asap_chemprop")]:
    vals = {s: load_sigma_rho(dr, s, 200) for s in strategies}
    ranked = sorted(vals, key=lambda s: -vals[s].mean())
    print(f"  {model} (ranked): " + "  ".join(f"{s}={vals[s].mean():.3f}" for s in ranked))

print("\n\nDone.")

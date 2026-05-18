"""
verify_stats.py — one-to-one verification of all statistical claims in blogpost.md.

Each labelled block corresponds to a specific comparison made in the text.
Run with:  python verify_stats.py

Requires: results/ directory with run_*.pkl files and setup_*.pkl files.

Sections
--------
[A]  PXR model accuracy (MAE, Kendall τ)
[B]  Mpro model accuracy (MAE, Kendall τ)
[C]  PXR uncertainty (sigma-rho)
[D]  Mpro uncertainty (sigma-rho)
[E]  PXR hit discovery (counts, rates, gaps)
[F]  Mpro hit discovery (counts, rates, gaps)
"""

import pickle
import glob
import numpy as np
from scipy import stats
from scipy.stats import spearmanr


# ── helpers ──────────────────────────────────────────────────────────────────

def load_metric(results_dir, strategy, metric, target_n):
    """Load a scalar metric at a target labeled-pool size across all seeds.

    Parameters
    ----------
    results_dir : str
        Path to a results directory containing run_*.pkl files.
    strategy : str
        Acquisition strategy name (e.g. ``"EI"``, ``"Exploitation"``).
    metric : str
        Key to extract from each history step dict (e.g. ``"mae"``, ``"ktau"``).
    target_n : int
        Target ``n_labeled`` value; the closest step is selected per seed.

    Returns
    -------
    np.ndarray
        1-D array of metric values, one per seed file found.
    """
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
    """Print a t-test comparison between two arrays of metric values.

    Parameters
    ----------
    label : str
        Human-readable description of the comparison printed as a header.
    a : np.ndarray
        Metric values for condition A (one value per seed).
    b : np.ndarray
        Metric values for condition B (one value per seed).
    paired : bool, optional
        If ``True`` (default), use a paired t-test; otherwise unpaired.
    """
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

# [A1] ChemProp: Exploitation vs Random MAE at n=900
print("\n[A1] ChemProp: Exploitation vs Random MAE at n=900")
print("     → text: 'small but statistically significant gap of 0.07 (p=0.0007)'")
report("A[ChemProp Exploitation] vs B[ChemProp Random]",
       load_metric("results/pxr_chemprop", "Exploitation", "mae", 900),
       load_metric("results/pxr_chemprop", "Random",       "mae", 900))

# [A2] CheMeleon: Exploitation vs Random MAE at n=900
print("\n[A2] CheMeleon: Exploitation vs Random MAE at n=900")
print("     → text: 'still significant, 0.02 gap (p=0.028)'")
report("A[CheMeleon Exploitation] vs B[CheMeleon Random]",
       load_metric("results/pxr_chemeleon", "Exploitation", "mae", 900),
       load_metric("results/pxr_chemeleon", "Random",       "mae", 900))

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
# Random is used here (not Exploitation) so both configs receive the same compounds,
# isolating the ChEMBL contribution without confounding from strategy-model interaction.
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


# ══════════════════════════════════════════════════════════════════════════════
# Helper: load pool data and accumulate selected indices up to target_n

def _load_pool_hits(results_dir, hit_threshold):
    """Return (pool_hits_mask, split) for a results directory."""
    split = "predefined" if "asap" in results_dir else "random"
    with open(f"{results_dir}/setup_{split}.pkl", "rb") as f:
        setup = pickle.load(f)
    pool_hits = (setup["df_pool"]["pEC50"] >= hit_threshold).values
    return pool_hits, split


def count_hits_at_n(results_dir, strategy, target_n, hit_threshold):
    """Count hits accumulated up to *target_n* pool labels, per seed.

    Parameters
    ----------
    results_dir : str
        Path to results directory with run_*.pkl files and setup_*.pkl.
    strategy : str
        Acquisition strategy name.
    target_n : int
        Target ``n_labeled``; the closest history step is used.
    hit_threshold : float
        Activity threshold defining a hit.

    Returns
    -------
    np.ndarray
        1-D array of cumulative hit counts, one per seed.
    """
    pool_hits, split = _load_pool_hits(results_dir, hit_threshold)
    counts = []
    for f in sorted(glob.glob(f"{results_dir}/run_{split}_{strategy}_seed*.pkl")):
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        h = d["result"]["history"]
        tgt_step = min(h, key=lambda s: abs(s["n_labeled"] - target_n))
        tgt_k = tgt_step["iteration"]
        idx: list[int] = []
        for step in h:
            if step["iteration"] > tgt_k:
                break
            idx.extend(step["selected_pool_indices"])
        counts.append(int(sum(pool_hits[i] for i in idx)))
    assert counts, f"No files found for {results_dir}/{strategy}"
    return np.array(counts)


def compute_gap_at_first_query(results_dir, strategy, hit_threshold):
    """Compute the hit/non-hit predicted-activity gap at the first query, per seed.

    Reads ``acquisition_scores`` (pool predictions used for acquisition) at
    iteration 1 (the first query from a trained model) and computes
    ``mean(scores[hits]) - mean(scores[non-hits])`` over the unlabeled pool.

    Parameters
    ----------
    results_dir : str
        Path to results directory.
    strategy : str
        Acquisition strategy name.
    hit_threshold : float
        Activity threshold defining a hit.

    Returns
    -------
    np.ndarray
        1-D array of gap values, one per seed.
    """
    pool_hits, split = _load_pool_hits(results_dir, hit_threshold)
    n_pool = len(pool_hits)
    gaps = []
    for f in sorted(glob.glob(f"{results_dir}/run_{split}_{strategy}_seed*.pkl")):
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        h = d["result"]["history"]
        # Iteration 0 = first query: model trained on initial pool, scores on the rest.
        # acquisition_scores at h[0] covers the unlabeled pool *after* removing h[0]'s
        # selected_pool_indices (the initial n_start random compounds).
        if not h:
            continue
        step0 = h[0]
        labeled_mask = np.zeros(n_pool, dtype=bool)
        labeled_mask[step0["selected_pool_indices"]] = True
        unlabeled_idx = np.where(~labeled_mask)[0]
        scores_raw = step0.get("acquisition_scores")
        if scores_raw is None or len(scores_raw) != len(unlabeled_idx):
            gaps.append(float("nan"))
            continue
        scores = np.asarray(scores_raw)
        y_true_unlab = pool_hits[unlabeled_idx]
        if y_true_unlab.sum() == 0 or (~y_true_unlab).sum() == 0:
            gaps.append(float("nan"))
            continue
        gaps.append(float(np.mean(scores[y_true_unlab]) - np.mean(scores[~y_true_unlab])))
    assert gaps, f"No gap data for {results_dir}/{strategy}"
    return np.array(gaps)


# ══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 60)
print("HIT DISCOVERY — PXR")
print("=" * 60)

PXR_HIT_THRESH = 6.0

# [E1] Pool composition
print("\n[E1] PXR pool hit count and rate")
print("     → text: '54 compounds with pEC50 ≥ 6.0 in a pool of 3312, a 1.6% hit rate'")
pool_hits_pxr, _ = _load_pool_hits("results/pxr_chemprop", PXR_HIT_THRESH)
n_hits_pxr = pool_hits_pxr.sum()
n_pool_pxr = len(pool_hits_pxr)
print(f"  Pool hits: {n_hits_pxr}  Pool size: {n_pool_pxr}  Rate: {n_hits_pxr/n_pool_pxr*100:.1f}%")

# [E2] Exploitation vs Random cumulative hits at n=900 (ChemProp)
print("\n[E2] PXR ChemProp Exploitation vs Random cumulative hits at n=900")
print("     → text: 'Exploitation ~70% of pool actives by 27% labeled (n=900), Random ~26%'")
exp_hits_cp = count_hits_at_n("results/pxr_chemprop", "Exploitation", 900, PXR_HIT_THRESH)
rnd_hits_cp = count_hits_at_n("results/pxr_chemprop", "Random",       900, PXR_HIT_THRESH)
print(f"  Exploitation: {exp_hits_cp.tolist()} mean={exp_hits_cp.mean():.1f} ({exp_hits_cp.mean()/n_hits_pxr*100:.0f}% of pool actives)")
print(f"  Random:       {rnd_hits_cp.tolist()} mean={rnd_hits_cp.mean():.1f} ({rnd_hits_cp.mean()/n_hits_pxr*100:.0f}% of pool actives)")

# [E3] ChemProp vs CheMeleon Exploitation hits at n=900
print("\n[E3] PXR ChemProp vs CheMeleon Exploitation hits at n=900")
print("     → text: 'ChemProp 38 vs CheMeleon 32 actives at n=900'")
exp_hits_cm = count_hits_at_n("results/pxr_chemeleon", "Exploitation", 900, PXR_HIT_THRESH)
print(f"  ChemProp:  {exp_hits_cp.tolist()} mean={exp_hits_cp.mean():.1f}")
print(f"  CheMeleon: {exp_hits_cm.tolist()} mean={exp_hits_cm.mean():.1f}")

# [E4] ChEMBL warmstart first-query hits at n=100 (first query, before pool labels)
print("\n[E4] PXR ChEMBL warm-start: hits in first query (n_labeled=0 → first 100 queried)")
print("     → text: 'roughly 10 actives before any pool labels are acquired'")
for model, dr in [("CheMeleon+ChEMBL", "results/pxr_chemeleon_chembl"),
                  ("ChemProp+ChEMBL",  "results/pxr_chemprop_chembl")]:
    pool_hits_m, split_m = _load_pool_hits(dr, PXR_HIT_THRESH)
    first_query_hits = []
    for f in sorted(glob.glob(f"{dr}/run_{split_m}_Exploitation_seed*.pkl")):
        with open(f, "rb") as fh:
            d = pickle.load(fh)
        h = d["result"]["history"]
        # h[0] is n_labeled=0 (ChEMBL only); h[1] is first actual query
        if len(h) > 1:
            idx = h[1]["selected_pool_indices"]
            first_query_hits.append(int(sum(pool_hits_m[i] for i in idx)))
    arr = np.array(first_query_hits)
    print(f"  {model}: {arr.tolist()} mean={arr.mean():.1f}")

# [E5] PXR hit/non-hit predicted-activity gap at first query
print("\n[E5] PXR hit/non-hit predicted-activity gap at first query (Exploitation)")
print("     → text: 'gap 0.64 for ChemProp vs 0.44 for CheMeleon at first query'")
gap_cp = compute_gap_at_first_query("results/pxr_chemprop",  "Exploitation", PXR_HIT_THRESH)
gap_cm = compute_gap_at_first_query("results/pxr_chemeleon", "Exploitation", PXR_HIT_THRESH)
print(f"  ChemProp:  {np.round(gap_cp,3).tolist()} mean={gap_cp.mean():.3f}")
print(f"  CheMeleon: {np.round(gap_cm,3).tolist()} mean={gap_cm.mean():.3f}")

# [E6] Two-to-three times as many hits mid-campaign
print("\n[E6] PXR mid-campaign Exploitation vs Random hit ratio")
print("     → text: 'two to three times as many hits as Random at mid-campaign'")
for n in [500, 700]:
    exp_n = count_hits_at_n("results/pxr_chemprop", "Exploitation", n, PXR_HIT_THRESH)
    rnd_n = count_hits_at_n("results/pxr_chemprop", "Random",       n, PXR_HIT_THRESH)
    ratios = exp_n / np.maximum(rnd_n, 1)
    print(f"  n={n}: Exploitation mean={exp_n.mean():.1f}  Random mean={rnd_n.mean():.1f}  ratio={ratios.mean():.2f}x")


# ══════════════════════════════════════════════════════════════════════════════
print("\n\n" + "=" * 60)
print("HIT DISCOVERY — MPRO")
print("=" * 60)

ASAP_HIT_THRESH = 7.0

# [F1] Pool composition
print("\n[F1] ASAP pool hit count and rate")
print("     → text: '76 actives in a pool of 842 compounds, 9.0%'")
pool_hits_asap, _ = _load_pool_hits("results/asap_chemprop", ASAP_HIT_THRESH)
n_hits_asap = pool_hits_asap.sum()
n_pool_asap = len(pool_hits_asap)
print(f"  Pool hits: {n_hits_asap}  Pool size: {n_pool_asap}  Rate: {n_hits_asap/n_pool_asap*100:.1f}%")

# [F2] CheMeleon Exploitation vs ChemProp hits at n=200
print("\n[F2] ASAP CheMeleon vs ChemProp Exploitation hits at n=200")
print("     → text: 'CheMeleon 55 of 76 pool actives by n=200 vs 47 for ChemProp, 17% difference'")
exp_hits_cm_asap = count_hits_at_n("results/asap_chemeleon", "Exploitation", 200, ASAP_HIT_THRESH)
exp_hits_cp_asap = count_hits_at_n("results/asap_chemprop",  "Exploitation", 200, ASAP_HIT_THRESH)
print(f"  CheMeleon: {exp_hits_cm_asap.tolist()} mean={exp_hits_cm_asap.mean():.1f}")
print(f"  ChemProp:  {exp_hits_cp_asap.tolist()} mean={exp_hits_cp_asap.mean():.1f}")
pct_diff = (exp_hits_cm_asap.mean() - exp_hits_cp_asap.mean()) / exp_hits_cp_asap.mean() * 100
print(f"  CheMeleon advantage: {pct_diff:.0f}%")

# [F3] Random hits at n=200 (model-independent baseline)
print("\n[F3] ASAP Random hits at n=200")
print("     → text: 'Random finds only 17 actives at n=200 regardless of model'")
rnd_hits_cm_asap = count_hits_at_n("results/asap_chemeleon", "Random", 200, ASAP_HIT_THRESH)
rnd_hits_cp_asap = count_hits_at_n("results/asap_chemprop",  "Random", 200, ASAP_HIT_THRESH)
print(f"  CheMeleon Random: {rnd_hits_cm_asap.tolist()} mean={rnd_hits_cm_asap.mean():.1f}")
print(f"  ChemProp  Random: {rnd_hits_cp_asap.tolist()} mean={rnd_hits_cp_asap.mean():.1f}")

# [F4] ASAP hit/non-hit predicted-activity gap at first query
print("\n[F4] ASAP hit/non-hit predicted-activity gap at first query (Exploitation)")
print("     → text: 'CheMeleon gap 1.35 vs ChemProp 0.93 at first query'")
gap_cm_asap = compute_gap_at_first_query("results/asap_chemeleon", "Exploitation", ASAP_HIT_THRESH)
gap_cp_asap = compute_gap_at_first_query("results/asap_chemprop",  "Exploitation", ASAP_HIT_THRESH)
print(f"  CheMeleon: {np.round(gap_cm_asap,3).tolist()} mean={gap_cm_asap.mean():.3f}")
print(f"  ChemProp:  {np.round(gap_cp_asap,3).tolist()} mean={gap_cp_asap.mean():.3f}")

# [F5] Two-to-three times as many hits mid-campaign on Mpro
print("\n[F5] ASAP mid-campaign Exploitation vs Random hit ratio")
print("     → text: 'two to three times as many hits as Random at mid-campaign'")
for n in [200, 400]:
    exp_n = count_hits_at_n("results/asap_chemeleon", "Exploitation", n, ASAP_HIT_THRESH)
    rnd_n = count_hits_at_n("results/asap_chemeleon", "Random",       n, ASAP_HIT_THRESH)
    ratios = exp_n / np.maximum(rnd_n, 1)
    print(f"  n={n}: Exploitation mean={exp_n.mean():.1f}  Random mean={rnd_n.mean():.1f}  ratio={ratios.mean():.2f}x")


print("\n\nDone.")

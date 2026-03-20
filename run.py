#!/usr/bin/env python
"""Active learning pipeline entry point.

Loads the PXR dataset, performs a scaffold split, fits the GTM embedding for
visualization, then runs the active learning loop for the requested strategies
and seeds, saving one pickle file per (strategy, seed) pair.

Usage
-----
    # Default: all strategies × all seeds
    python run.py

    # Pre-compute shared setup once (fast; run before submitting HPC jobs)
    python run.py --setup-only

    # Single HPC job: one (strategy, seed) pair → results/run_<S>_seed<N>.pkl
    python run.py --strategy EI --seed 42

    # All seeds for one strategy (serially)
    python run.py --strategy EI

    # All strategies for one seed (serially)
    python run.py --seed 42

HPC workflow example (SLURM)
-----------------------------
    python run.py --setup-only
    for strat in EI UCB Random Exploitation Exploration Diversity; do
        for seed in 42 43 44 45 46; do
            sbatch --job-name=al_${strat}_${seed} \\
                   --wrap="python run.py --strategy ${strat} --seed ${seed}"
        done
    done
    python analysis.py  # loads setup.pkl + run_*.pkl directly; no merge step needed

Checkpoint format (results/setup.pkl)
--------------------------------------
    {
        "df_pool", "df_test", "df_seed",
        "gtm_coords_pool", "gtm_coords_background",
        "background_smiles", "config",
    }

Per-job format (results/run_<STRATEGY>_seed<N>.pkl)
----------------------------------------------------
    {"strategy": str, "seed": int, "n_start": int, "result": {history, committee}}
"""

import argparse
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from openadmet.models.split.scaffold import ScaffoldSplitter

from src.config import ALConfig, load_config
from src.helpers import (
    run_active_learning,
    smiles_to_gtm,
)

warnings.filterwarnings("ignore")

SETUP_PKL = Path("results/setup.pkl")


# ── Setup phase ────────────────────────────────────────────────────────────────


def run_setup(cfg: ALConfig) -> dict:
    """Load data, scaffold-split, fit GTM, and optionally load external seed data."""
    df = pd.read_parquet(Path(cfg.dataset_path).expanduser())

    background_smiles = list(
        pd.read_csv(cfg.background_path)[cfg.background_smiles_col].values
    )
    background_smiles = [
        x for x in background_smiles if x not in df["OPENADMET_CANONICAL_SMILES"].values
    ]

    print(f"Dataset: {len(df)} compounds")
    print(df["PXR_pEC50"].describe())

    # Activity distribution plot
    fig, ax = plt.subplots(1, dpi=150)
    ax.hist(df["PXR_pEC50"], bins=30, color="teal", alpha=0.7)
    ax.set_title("PXR pEC50 Distribution")
    ax.set_xlabel("pEC50", fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    Path("results").mkdir(exist_ok=True)
    fig.savefig("results/pec50_distribution.png", bbox_inches="tight")
    plt.close(fig)

    # Scaffold split: 80% pool, 20% test. Per-iteration calibration is handled
    # inside run_active_learning by holding out 10% of the current training set.
    splitter = ScaffoldSplitter(
        train_size=0.8, val_size=0.1, test_size=0.1, random_state=42
    )
    X_pool, X_cal_tmp, X_test_tmp, y_pool, y_cal_tmp, y_test_tmp, _ = splitter.split(
        df["OPENADMET_CANONICAL_SMILES"], df["PXR_pEC50"]
    )

    df_pool = pd.DataFrame({"smiles": X_pool, "pEC50": y_pool}).reset_index(drop=True)
    # Combine the val and test splits into a single 20% held-out test set.
    df_test = pd.DataFrame(
        {
            "smiles": list(X_cal_tmp) + list(X_test_tmp),
            "pEC50": list(y_cal_tmp) + list(y_test_tmp),
        }
    ).reset_index(drop=True)

    min_pool_needed = cfg.k_iter * cfg.query_size
    assert len(df_pool) >= min_pool_needed, (
        f"Pool too small: {len(df_pool)} < {min_pool_needed}. "
        "Increase dataset size or reduce K_ITER/M_QUERY."
    )

    print(f"Pool size: \t\t{len(df_pool)}")
    print(f"Test size: \t\t{len(df_test)}")

    # GTM embedding (pool + background, for visualization)
    print("Fitting GTM on pool + background molecules...")
    gtm_model, gtm_coords, resps, llhs = smiles_to_gtm(
        list(df_pool["smiles"].values) + background_smiles,
        device="cpu",
    )
    gtm_coords_pool = gtm_coords[: len(df_pool)]
    gtm_coords_background = gtm_coords[len(df_pool) :]

    # ── External seed data ─────────────────────────────────────────────────────
    df_seed = None
    if cfg.seed_data_path is not None:
        seed_path = Path(cfg.seed_data_path).expanduser()
        if seed_path.suffix in {".parquet", ".pq"}:
            df_seed_raw = pd.read_parquet(seed_path)
        else:
            df_seed_raw = pd.read_csv(seed_path)
        df_seed = (
            df_seed_raw[[cfg.seed_smiles_col, cfg.seed_activity_col]]
            .rename(
                columns={cfg.seed_smiles_col: "smiles", cfg.seed_activity_col: "pEC50"}
            )
            .dropna(subset=["smiles", "pEC50"])
            .reset_index(drop=True)
        )
        # Deduplicate against all main-dataset splits to prevent leakage
        main_smiles = set(df_pool["smiles"]) | set(df_test["smiles"])
        before = len(df_seed)
        df_seed = df_seed[~df_seed["smiles"].isin(main_smiles)].reset_index(drop=True)
        n_removed = before - len(df_seed)
        print(f"Seed data: {len(df_seed)} compounds loaded from {seed_path.name}")
        if n_removed > 0:
            print(
                f"  (removed {n_removed} compounds overlapping with main dataset splits)"
            )

    return {
        "df_pool": df_pool,
        "df_test": df_test,
        "df_seed": df_seed,
        "gtm_coords_pool": gtm_coords_pool,
        "gtm_coords_background": gtm_coords_background,
        "background_smiles": background_smiles,
        "config": cfg,
    }


def load_or_run_setup(config_path: str = "config.yaml") -> dict:
    """Load results/setup.pkl if it exists, otherwise compute and save it."""
    if SETUP_PKL.exists():
        print(f"Loading shared setup from {SETUP_PKL}...")
        with open(SETUP_PKL, "rb") as fh:
            return pickle.load(fh)
    print("No setup.pkl found — running setup now...")
    cfg = load_config(config_path)
    setup = run_setup(cfg)
    with open(SETUP_PKL, "wb") as fh:
        pickle.dump(setup, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Setup saved to {SETUP_PKL}.")
    return setup


# ── Per-job AL run ─────────────────────────────────────────────────────────────


def run_job(strategy: str, seed: int, setup: dict) -> None:
    """Run one (strategy, seed) pair and save results/run_<strategy>_seed<seed>.pkl."""
    cfg: ALConfig = setup["config"]
    out_path = Path(f"results/run_{strategy}_seed{seed}.pkl")
    if out_path.exists():
        print(f"Skipping {strategy!r} seed={seed} (already exists: {out_path}).")
        return

    print(f"\n{'─' * 60}")
    print(f"Strategy: {strategy}  |  Seed: {seed}  |  n_start: {cfg.n_start}")
    print(f"{'─' * 60}")

    result = run_active_learning(
        setup["df_pool"],
        setup["df_test"],
        n_start=cfg.n_start,
        k_iter=cfg.k_iter,
        query_size=cfg.query_size,
        n_models=cfg.n_models,
        max_epochs=cfg.max_epochs,
        seed=seed,
        strategy=strategy,
        verbose=True,
        df_seed=setup.get("df_seed"),
        gtm_coords=setup["gtm_coords_pool"],
    )

    with open(out_path, "wb") as fh:
        pickle.dump(
            {
                "strategy": strategy,
                "seed": seed,
                "n_start": cfg.n_start,
                "result": result,
            },
            fh,
            protocol=pickle.HIGHEST_PROTOCOL,
        )
    print(f"Saved {out_path}.")


# ── Entry point ────────────────────────────────────────────────────────────────


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Active learning pipeline for PXR pEC50 prediction."
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Run data loading, scaffold split, and GTM embedding only; "
        "save results/setup.pkl and exit.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        metavar="PATH",
        help="Path to the YAML experiment config file. Default: config.yaml. "
        "Used only during setup; jobs always use the config frozen in setup.pkl.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Acquisition strategy to run. Omit to run all strategies defined in config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed to use. Omit to run all seeds defined in config.",
    )
    args = parser.parse_args()

    Path("results").mkdir(exist_ok=True)

    # ── Mode 1: setup only ─────────────────────────────────────────────────────
    if args.setup_only:
        cfg = load_config(args.config)
        setup = run_setup(cfg)
        with open(SETUP_PKL, "wb") as fh:
            pickle.dump(setup, fh, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"Setup saved to {SETUP_PKL}.")
        return

    # ── Load or compute shared setup ──────────────────────────────────────────
    setup = load_or_run_setup(args.config)
    cfg: ALConfig = setup["config"]

    # ── Validate per-job CLI args against the frozen config ────────────────────
    if args.strategy is not None and args.strategy not in cfg.strategies:
        raise SystemExit(
            f"ERROR: strategy {args.strategy!r} is not in the configured strategies: "
            f"{cfg.strategies}. Check config.yaml or omit --strategy to run all."
        )
    if args.seed is not None and args.seed not in cfg.seeds:
        raise SystemExit(
            f"ERROR: seed {args.seed} is not in the configured seeds: {cfg.seeds}. "
            f"Check config.yaml or omit --seed to run all."
        )

    # ── Determine (strategy, seed) pairs to run ────────────────────────────────
    strategies_to_run = [args.strategy] if args.strategy else cfg.strategies
    seeds_to_run = [args.seed] if args.seed is not None else cfg.seeds

    # ── Run jobs ───────────────────────────────────────────────────────────────
    for strategy in strategies_to_run:
        for seed in seeds_to_run:
            run_job(strategy, seed, setup)

    print("\nAll jobs complete.")


if __name__ == "__main__":
    main()

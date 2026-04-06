#!/usr/bin/env python
r"""Active learning pipeline entry point.

Loads the PXR dataset, performs the requested split(s), fits the GTM embedding
for visualization, then runs the active learning loop for all requested
(split_type, strategy, seed) combinations, saving one pickle file per triple.

Usage
-----
    # Default: all split_types × all strategies × all seeds
    python run.py

    # Pre-compute shared setups once (run before submitting HPC jobs)
    python run.py --setup-only

    # Single HPC job: one (split_type, strategy, seed) triple
    python run.py --split-type scaffold --strategy EI --seed 42

    # All seeds for one (split_type, strategy) pair (serially)
    python run.py --split-type scaffold --strategy EI

    # All strategies/seeds for one split type (serially)
    python run.py --split-type scaffold

HPC workflow example (SLURM)
-----------------------------
    python run.py --setup-only
    for split in scaffold random; do
        for strat in EI UCB Random Exploitation Exploration Diversity; do
            for seed in 42 43 44 45 46; do
                sbatch --job-name=al_${split}_${strat}_${seed} \\
                       --wrap="python run.py --split-type ${split} \
                               --strategy ${strat} --seed ${seed}"
            done
        done
    done
    python analysis.py

Checkpoint format (results/setup_<split_type>.pkl)
---------------------------------------------------
    {
        "split_type", "df_pool", "df_test", "df_seed",
        "gtm_coords_pool", "config",
    }

Per-job format (results/run_<split_type>_<STRATEGY>_seed<N>.pkl)
-----------------------------------------------------------------
    {"split_type": str, "strategy": str, "seed": int, "n_start": int,
     "result": {history, committee}}
"""

import argparse
import pickle
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from openadmet.models.split.scaffold import ScaffoldSplitter
from openadmet.models.split.sklearn import ShuffleSplitter

from src.config import ALConfig, _VALID_SPLIT_TYPES, load_config
from src.helpers import (
    run_active_learning,
    smiles_to_gtm,
)

warnings.filterwarnings("ignore")


# ── Setup phase ────────────────────────────────────────────────────────────────


def run_setup(cfg: ALConfig, split_type: str) -> dict:
    """Load data, split, fit GTM embedding, and optionally load seed data.

    Reads the dataset from the path specified in ``cfg``, performs an 80/20
    split (scaffold or random) into pool and test sets, fits a GTM on the pool
    for visualization, and (optionally) loads an external seed training dataset
    while deduplicating it against pool and test compounds to prevent leakage.

    Parameters
    ----------
    cfg : ALConfig
        Validated experiment configuration loaded via ``load_config()``.
    split_type : str
        One of ``"scaffold"`` or ``"random"``. Controls whether
        ``ScaffoldSplitter`` or ``ShuffleSplitter`` is used.

    Returns
    -------
    dict
        Setup checkpoint with keys: ``"split_type"``, ``"df_pool"``,
        ``"df_test"``, ``"df_seed"`` (``None`` if no seed data),
        ``"gtm_coords_pool"``, and ``"config"``.

    """
    _ds_path = Path(cfg.dataset_path).expanduser()
    df = (
        pd.read_csv(_ds_path)
        if _ds_path.suffix.lower() == ".csv"
        else pd.read_parquet(_ds_path)
    )

    print(f"Dataset: {len(df)} compounds")
    print(df[cfg.dataset_activity_col].describe())

    # Activity distribution plot (same data regardless of split type; idempotent)
    fig, ax = plt.subplots(1, dpi=150)
    ax.hist(df[cfg.dataset_activity_col], bins=30, color="teal", alpha=0.7)
    ax.set_title("PXR pEC50 Distribution")
    ax.set_xlabel("pEC50", fontweight="bold")
    ax.set_ylabel("Count", fontweight="bold")
    Path("results").mkdir(exist_ok=True)
    fig.savefig("results/pec50_distribution.png", bbox_inches="tight")
    plt.close(fig)

    # Split: 80% pool, 20% test. Per-iteration calibration is handled
    # inside run_active_learning by holding out 10% of the current training set.
    if split_type == "scaffold":
        splitter = ScaffoldSplitter(
            train_size=0.8, val_size=0.0, test_size=0.2, random_state=42
        )
    else:
        splitter = ShuffleSplitter(
            train_size=0.8, val_size=0.0, test_size=0.2, random_state=42
        )
    print(f"Splitting data ({split_type})...")
    X_pool, _, X_test, y_pool, _, y_test, _ = splitter.split(
        df[cfg.dataset_smiles_col], df[cfg.dataset_activity_col]
    )

    df_pool = pd.DataFrame({"smiles": X_pool, "pEC50": y_pool}).reset_index(drop=True)
    df_test = pd.DataFrame({"smiles": X_test, "pEC50": y_test}).reset_index(drop=True)

    min_pool_needed = cfg.k_iter * cfg.query_size
    assert len(df_pool) >= min_pool_needed, (
        f"Pool too small: {len(df_pool)} < {min_pool_needed}. "
        "Increase dataset size or reduce K_ITER/M_QUERY."
    )

    print(f"Pool size: \t\t{len(df_pool)}")
    print(f"Test size: \t\t{len(df_test)}")

    # GTM embedding (pool compounds, for visualization)
    print("Fitting GTM on pool molecules...")
    gtm_model, gtm_coords_pool, resps, llhs = smiles_to_gtm(
        list(df_pool["smiles"].values),
        device="cpu",
    )

    # ── External seed data ─────────────────────────────────────────────────────
    df_seed = None
    if cfg.seed_data_path is not None:
        seed_path = Path(cfg.seed_data_path).expanduser()
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
        "split_type": split_type,
        "df_pool": df_pool,
        "df_test": df_test,
        "df_seed": df_seed,
        "gtm_coords_pool": gtm_coords_pool,
        "config": cfg,
    }


def load_or_run_setup(split_type: str, config_path: str = "config.yaml") -> dict:
    """Load the setup checkpoint for ``split_type``, computing it if absent.

    If ``results/setup_<split_type>.pkl`` exists it is loaded and returned
    directly, allowing HPC worker jobs to skip the expensive setup phase.
    Otherwise ``run_setup`` is called and the result is pickled.

    Parameters
    ----------
    split_type : str
        One of ``"scaffold"`` or ``"random"``.
    config_path : str, optional
        Path to the YAML config file, used only when setup must be computed.

    Returns
    -------
    dict
        Setup checkpoint — same structure as the return value of ``run_setup()``.

    """
    setup_pkl = Path(f"results/setup_{split_type}.pkl")
    if setup_pkl.exists():
        print(f"Loading shared setup from {setup_pkl}...")
        with open(setup_pkl, "rb") as fh:
            return pickle.load(fh)
    print(f"No {setup_pkl} found — running setup now...")
    cfg = load_config(config_path)
    setup = run_setup(cfg, split_type)
    with open(setup_pkl, "wb") as fh:
        pickle.dump(setup, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Setup saved to {setup_pkl}.")
    return setup


# ── Per-job AL run ─────────────────────────────────────────────────────────────


def run_job(strategy: str, seed: int, split_type: str, setup: dict) -> None:
    """Run one (split_type, strategy, seed) triple and write its result to disk.

    Skips the job silently if the output file already exists, making it safe
    to dispatch the same job multiple times (idempotent). On success, writes
    ``results/run_<split_type>_<strategy>_seed<seed>.pkl``.

    Parameters
    ----------
    strategy : str
        Acquisition strategy name (must be present in ``cfg.strategies``).
    seed : int
        Outer random seed for this run (must be present in ``cfg.seeds``).
    split_type : str
        Dataset split type used for this job (``"scaffold"`` or ``"random"``).
    setup : dict
        Shared setup checkpoint as returned by ``load_or_run_setup()``.

    """
    cfg: ALConfig = setup["config"]
    out_path = Path(f"results/run_{split_type}_{strategy}_seed{seed}.pkl")
    if out_path.exists():
        print(f"Skipping {split_type}/{strategy!r} seed={seed} (already exists: {out_path}).")
        return

    print(f"\n{'─' * 60}")
    print(f"Split: {split_type}  |  Strategy: {strategy}  |  Seed: {seed}  |  n_start: {cfg.n_start}")
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
        use_chemeleon=cfg.use_chemeleon,
    )

    with open(out_path, "wb") as fh:
        pickle.dump(
            {
                "split_type": split_type,
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
    """Parse CLI arguments and dispatch active learning jobs.

    Supports three operating modes based on the flags provided:

    - ``--setup-only``: run data loading, splitting, and GTM embedding for all
      configured ``split_types``; save ``results/setup_<split_type>.pkl`` and
      exit. Use this once before submitting HPC jobs.
    - Single job: ``--split-type S --strategy S --seed N`` runs exactly one
      (split_type, strategy, seed) triple.
    - Batch: omit any of the above flags to run all configured combinations.

    All experiment parameters are read from the YAML config file (default:
    ``config.yaml``) during setup and then frozen into the setup pickles;
    worker jobs always use the config stored in the checkpoint.
    """
    parser = argparse.ArgumentParser(
        description="Active learning pipeline for PXR pEC50 prediction."
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Run data loading, splitting, and GTM embedding for all split_types; "
        "save results/setup_<split_type>.pkl and exit.",
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        metavar="PATH",
        help="Path to the YAML experiment config file. Default: config.yaml. "
        "Used only during setup; jobs always use the config frozen in setup_<split_type>.pkl.",
    )
    parser.add_argument(
        "--split-type",
        type=str,
        default=None,
        choices=_VALID_SPLIT_TYPES,
        help="Dataset split type. Omit to run all split_types defined in config.",
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
    cfg = load_config(args.config)

    # ── Mode 1: setup only ─────────────────────────────────────────────────────
    if args.setup_only:
        for split_type in cfg.split_types:
            setup = run_setup(cfg, split_type)
            pkl_path = Path(f"results/setup_{split_type}.pkl")
            with open(pkl_path, "wb") as fh:
                pickle.dump(setup, fh, protocol=pickle.HIGHEST_PROTOCOL)
            print(f"Setup saved to {pkl_path}.")
        return

    # ── Validate per-job CLI args against the config ───────────────────────────
    if args.split_type is not None and args.split_type not in cfg.split_types:
        raise SystemExit(
            f"ERROR: split_type {args.split_type!r} is not in the configured "
            f"split_types: {cfg.split_types}. Check config.yaml or omit --split-type."
        )
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

    # ── Determine (split_type, strategy, seed) combinations to run ────────────
    split_types_to_run = [args.split_type] if args.split_type else cfg.split_types
    strategies_to_run = [args.strategy] if args.strategy else cfg.strategies
    seeds_to_run = [args.seed] if args.seed is not None else cfg.seeds

    # ── Run jobs ───────────────────────────────────────────────────────────────
    for split_type in split_types_to_run:
        setup = load_or_run_setup(split_type, args.config)
        for strategy in strategies_to_run:
            for seed in seeds_to_run:
                run_job(strategy, seed, split_type, setup)

    print("\nAll jobs complete.")


if __name__ == "__main__":
    main()

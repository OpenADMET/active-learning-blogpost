#!/usr/bin/env python
r"""Synthetic oracle active learning pipeline entry point.

Mirrors ``run.py`` exactly, replacing committee training with a fast
:class:`src.synthetic.SyntheticOracle` that knows all true labels and
simulates predictions with configurable MAE and uncertainty quality (Spearman
rank correlation between sigma and |residuals|).  Both parameters ramp linearly
over the campaign from initial to final values.

Results are saved in the **identical pickle format** as ``run.py``, so all
downstream analysis tools (``analysis.py``, ``synthetic_analysis.py``) work
unchanged.

Usage
-----
    # Pre-compute shared setups (reuses existing ChemProp setup pickles if
    # setup_results_path is set in the config, otherwise builds fresh ones)
    python synthetic_run.py --config config/oracle_pxr_rho00_config.yaml --setup-only

    # Run all strategies × all seeds (serial — no GPU required)
    python synthetic_run.py --config config/oracle_pxr_rho00_config.yaml

    # Single job
    python synthetic_run.py --config config/oracle_pxr_rho00_config.yaml \
        --split-type random --strategy EI --seed 42

    # Fast sanity check (forces k_iter=2, seeds=[42])
    python synthetic_run.py --config config/oracle_pxr_rho00_config.yaml --smoke-test

HPC workflow (run all 8 oracle configs × 5 strategies × 5 seeds)
-----------------------------------------------------------------
    for cfg in config/oracle_pxr_*.yaml; do
        python synthetic_run.py --config $cfg --setup-only
        for strat in EI UCB Random Exploitation Exploration; do
            for seed in 42 43 44 45 46; do
                sbatch --wrap="python synthetic_run.py --config $cfg \
                    --strategy $strat --seed $seed"
            done
        done
    done
    for cfg in config/oracle_pxr_*.yaml; do
        python synthetic_analysis.py --config $cfg
    done

Checkpoint format (<results_path>/setup_<split_type>.pkl)
----------------------------------------------------------
    Same as run.py.  If ``data.setup_results_path`` is set in the config,
    setup pickles are loaded from that directory (shared with real runs).

Per-job format (<results_path>/run_<split_type>_<STRATEGY>_seed<N>.pkl)
------------------------------------------------------------------------
    {"split_type": str, "strategy": str, "seed": int, "n_start": int,
     "result": {"history": [...], "committee": None}}
"""

import argparse
import pickle
import warnings
from pathlib import Path

import pandas as pd
import yaml

from src.config import ALConfig, _VALID_SPLIT_TYPES, load_config
from src.synthetic import (
    CachedPredOracle,
    SyntheticOracle,
    generate_prediction_cache,
    load_synthetic_config,
    run_active_learning_synthetic,
)

warnings.filterwarnings("ignore")


# ---------------------------------------------------------------------------
# Setup helpers (thin wrappers around run.py's logic)
# ---------------------------------------------------------------------------


def _load_or_run_setup(
    split_type: str,
    cfg: ALConfig,
    results_dir: Path,
    setup_results_dir: Path | None,
) -> dict:
    """Load setup pickle, trying setup_results_dir before results_dir.

    Parameters
    ----------
    split_type : str
        Split type key (e.g. ``"random"``).
    cfg : ALConfig
        Validated AL config (used only when generating a fresh setup).
    results_dir : Path
        Oracle run output directory.
    setup_results_dir : Path or None
        Optional alternate directory to look for pre-existing setup pickles.

    Returns
    -------
    dict
        Setup checkpoint.
    """
    # Try shared setup pickle first
    for candidate_dir in filter(None, [setup_results_dir, results_dir]):
        pkl = candidate_dir / f"setup_{split_type}.pkl"
        if pkl.exists():
            print(f"Loading shared setup from {pkl}...")
            with open(pkl, "rb") as fh:
                return pickle.load(fh)

    # No existing pickle — build fresh and save to results_dir
    print(f"No setup pickle found — running setup now (split_type={split_type})...")
    # Import here to avoid circular dep; run.py is not a module but a script
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from run import run_setup  # noqa: PLC0415

    setup = run_setup(cfg, split_type, results_dir=results_dir)
    out_pkl = results_dir / f"setup_{split_type}.pkl"
    with open(out_pkl, "wb") as fh:
        pickle.dump(setup, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Setup saved to {out_pkl}.")
    return setup


# ---------------------------------------------------------------------------
# Per-job run
# ---------------------------------------------------------------------------


def run_job(
    strategy: str,
    seed: int,
    split_type: str,
    setup: dict,
    oracle: SyntheticOracle,
    cfg: ALConfig,
    results_dir: Path,
) -> None:
    """Run one (split_type, strategy, seed) triple with the synthetic oracle.

    Skips silently if the output file already exists.

    Parameters
    ----------
    strategy : str
        Acquisition strategy name.
    seed : int
        Outer random seed.
    split_type : str
        Dataset split type.
    setup : dict
        Shared setup checkpoint from :func:`_load_or_run_setup`.
    oracle : SyntheticOracle
        Configured oracle instance.
    cfg : ALConfig
        Validated AL configuration.
    results_dir : Path
        Output directory for result pickles.
    """
    out_path = results_dir / f"run_{split_type}_{strategy}_seed{seed}.pkl"
    if out_path.exists():
        print(f"Skipping {split_type}/{strategy!r} seed={seed} (already exists).")
        return

    print(f"\n{'─' * 60}")
    print(f"Split: {split_type}  |  Strategy: {strategy}  |  Seed: {seed}")
    print(f"{'─' * 60}")

    result = run_active_learning_synthetic(
        setup["df_pool"],
        setup["df_test"],
        oracle=oracle,
        n_start=cfg.n_start,
        k_iter=cfg.k_iter,
        query_size=cfg.query_size,
        seed=seed,
        strategy=strategy,
        verbose=True,
        df_seed=setup.get("df_seed"),
        gtm_coords=setup.get("gtm_coords_pool"),
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


# ---------------------------------------------------------------------------
# Cached oracle helper
# ---------------------------------------------------------------------------


def _load_or_build_cached_oracle(
    oracle_cfg,
    smiles_to_y: dict,
    k_iter: int,
) -> "CachedPredOracle":
    """Load or lazily generate the prediction cache, then return a CachedPredOracle.

    If the cache file already exists it is loaded directly.  Otherwise the
    cache is generated from the configured source run directory and saved
    atomically (so parallel jobs are safe — the second writer is a no-op
    because :func:`generate_prediction_cache` uses an atomic rename).

    Parameters
    ----------
    oracle_cfg : SyntheticConfig
        Must have ``cache_path`` set.
    smiles_to_y : dict
        Ground-truth lookup for all compounds.
    k_iter : int
        Total AL iterations (for rho ramping).

    Returns
    -------
    CachedPredOracle
    """
    cache_path = Path(oracle_cfg.cache_path).expanduser()

    if not cache_path.exists():
        if oracle_cfg.cache_source_results_dir is None:
            raise ValueError(
                f"Cache file {cache_path} does not exist and "
                "'cache_source_results_dir' is not set in the oracle config.  "
                "Either pre-generate the cache or set cache_source_results_dir."
            )
        if oracle_cfg.cache_source_split_type is None:
            raise ValueError(
                "'cache_source_split_type' must be set when generating a cache."
            )
        print(f"Cache not found — generating from {oracle_cfg.cache_source_results_dir} ...")
        generate_prediction_cache(
            results_dir=oracle_cfg.cache_source_results_dir,
            split_type=oracle_cfg.cache_source_split_type,
            strategy=oracle_cfg.cache_source_strategy,
            seeds=oracle_cfg.cache_source_seeds,
            output_path=cache_path,
        )

    with open(cache_path, "rb") as fh:
        pred_cache = pickle.load(fh)

    return CachedPredOracle(
        pred_cache=pred_cache,
        smiles_to_y=smiles_to_y,
        oracle_cfg=oracle_cfg,
        k_iter=k_iter,
        seed=42,
    )


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main() -> None:
    """Parse CLI arguments and dispatch synthetic AL jobs."""
    parser = argparse.ArgumentParser(
        description="Synthetic oracle active learning pipeline."
    )
    parser.add_argument(
        "--config",
        default="config.yaml",
        metavar="PATH",
        help="Path to the YAML experiment config (must include an 'oracle:' section).",
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Build/load setup pickles for all split_types and exit.",
    )
    parser.add_argument(
        "--split-type",
        type=str,
        default=None,
        choices=_VALID_SPLIT_TYPES,
        help="Dataset split type. Omit to run all split_types in config.",
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default=None,
        help="Acquisition strategy. Omit to run all strategies in config.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed. Omit to run all seeds in config.",
    )
    parser.add_argument(
        "--smoke-test",
        action="store_true",
        help="Override k_iter=2 and seeds=[42] for a fast local sanity check.",
    )
    args = parser.parse_args()

    cfg = load_config(args.config)
    oracle_cfg = load_synthetic_config(args.config)

    if args.smoke_test:
        print("SMOKE TEST mode: k_iter=2, seeds=[42]")
        import dataclasses
        cfg = dataclasses.replace(cfg, k_iter=2, seeds=[42])

    results_dir = Path(cfg.results_path).expanduser()
    results_dir.mkdir(parents=True, exist_ok=True)

    # Resolve optional shared setup directory
    raw_cfg_data = yaml.safe_load(open(args.config))["data"]
    setup_results_path = raw_cfg_data.get("setup_results_path")
    setup_results_dir = (
        Path(setup_results_path).expanduser() if setup_results_path else None
    )

    # ── Setup phase ────────────────────────────────────────────────────────
    if args.setup_only:
        for split_type in cfg.split_types:
            _load_or_run_setup(split_type, cfg, results_dir, setup_results_dir)
        return

    # ── Validate CLI args ──────────────────────────────────────────────────
    if args.split_type is not None and args.split_type not in cfg.split_types:
        raise SystemExit(
            f"ERROR: split_type {args.split_type!r} not in config: {cfg.split_types}"
        )
    if args.strategy is not None and args.strategy not in cfg.strategies:
        raise SystemExit(
            f"ERROR: strategy {args.strategy!r} not in config: {cfg.strategies}"
        )
    if args.seed is not None and args.seed not in cfg.seeds:
        raise SystemExit(
            f"ERROR: seed {args.seed} not in config seeds: {cfg.seeds}"
        )

    split_types_to_run = [args.split_type] if args.split_type else cfg.split_types
    strategies_to_run = [args.strategy] if args.strategy else cfg.strategies
    seeds_to_run = [args.seed] if args.seed is not None else cfg.seeds

    # ── Run jobs ───────────────────────────────────────────────────────────
    for split_type in split_types_to_run:
        setup = _load_or_run_setup(split_type, cfg, results_dir, setup_results_dir)

        # Build smiles_to_y from ALL compounds (pool + test + optional seed data)
        smiles_to_y: dict[str, float] = {}
        for df in [setup["df_pool"], setup["df_test"]]:
            smiles_to_y.update(zip(df["smiles"].tolist(), df["pEC50"].tolist()))
        if setup.get("df_seed") is not None:
            df_s: pd.DataFrame = setup["df_seed"]
            smiles_to_y.update(zip(df_s["smiles"].tolist(), df_s["pEC50"].tolist()))

        if oracle_cfg.cache_path is not None:
            oracle = _load_or_build_cached_oracle(
                oracle_cfg, smiles_to_y, cfg.k_iter
            )
        else:
            oracle = SyntheticOracle(
                smiles_to_y=smiles_to_y,
                oracle_cfg=oracle_cfg,
                k_iter=cfg.k_iter,
                seed=42,
            )

        for strategy in strategies_to_run:
            for seed in seeds_to_run:
                run_job(strategy, seed, split_type, setup, oracle, cfg, results_dir)

    print("\nAll synthetic jobs complete.")


if __name__ == "__main__":
    main()

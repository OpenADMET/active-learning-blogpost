"""Experiment configuration loading and validation for the AL pipeline.

All parsing and field validation lives here. The rest of the codebase only
ever sees ``ALConfig`` instances — never raw dicts or YAML strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

# Valid strategy universe — kept in sync with STRATEGIES in src/helpers.py.
# Defined here directly to avoid any circular-import risk.
_VALID_STRATEGIES = ["EI", "UCB", "Random", "Exploitation", "Exploration", "Diversity"]


@dataclass
class ALConfig:
    """Validated experiment configuration loaded from a YAML config file.

    Attributes
    ----------
    dataset_path : str
        Path to the main labelled dataset (CSV or Parquet). Tilde expansion is applied.
    dataset_smiles_col : str
        Column name for SMILES strings in the main dataset.
    dataset_activity_col : str
        Column name for activity values in the main dataset.
    seed_data_path : str or None
        Path to an external seed training dataset (CSV or Parquet), or ``None``
        to skip. These compounds are always in the training set and are never
        queried or evaluated.
    seed_smiles_col : str
        Column name for SMILES in the seed data file.
    seed_activity_col : str
        Column name for activity values in the seed data file.
    strategies : list[str]
        Acquisition strategies to run. Must be a non-empty subset of
        ``["EI", "UCB", "Random", "Exploitation", "Exploration", "Diversity"]``.
    seeds : list[int]
        Outer random seeds that define independent AL runs for error bands.
        Each ``(strategy, seed)`` pair becomes one HPC job.
    k_iter : int
        Number of active learning iterations per run (> 0).
    query_size : int
        Compounds selected from the unlabeled pool per iteration (> 0).
    n_start : int
        Initial labeled pool size drawn randomly before the first AL iteration.
        Set to ``0`` when an external seed dataset bootstraps the committee.
    n_models : int
        Committee size — number of bootstrapped ensemble members (> 0).
    max_epochs : int
        Maximum training epochs per committee member (> 0).
    """

    dataset_path: str
    dataset_smiles_col: str
    dataset_activity_col: str
    seed_data_path: str | None
    seed_smiles_col: str
    seed_activity_col: str
    strategies: list[str]
    seeds: list[int]
    k_iter: int
    query_size: int
    n_start: int
    n_models: int
    max_epochs: int


def load_config(path: str | Path = "config.yaml") -> ALConfig:
    """Load and validate an AL experiment config from a YAML file.

    Parameters
    ----------
    path : str or Path
        Path to the YAML config file. Default is ``"config.yaml"``.

    Returns
    -------
    ALConfig
        Validated configuration dataclass.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If any required field is missing, has the wrong type, or fails a range
        check. All validation errors are collected and reported together.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as fh:
        raw = yaml.safe_load(fh)

    data = raw.get("data", {})
    al = raw.get("active_learning", {})
    tr = raw.get("training", {})

    errors: list[str] = []

    def _require_pos_int(section_dict: dict, key: str, section: str) -> int | None:
        val = section_dict.get(key)
        if val is None:
            errors.append(f"[{section}] '{key}' is required")
            return None
        if not isinstance(val, int) or val <= 0:
            errors.append(
                f"[{section}] '{key}' must be a positive integer, got {val!r}"
            )
            return None
        return val

    # data paths
    dataset_path = data.get("dataset_path")
    if not dataset_path or not isinstance(dataset_path, str):
        errors.append("[data] 'dataset_path' is required and must be a string")
        dataset_path = ""

    dataset_smiles_col = data.get("dataset_smiles_col")
    if not dataset_smiles_col or not isinstance(dataset_smiles_col, str):
        errors.append(
            "[data] 'dataset_smiles_col' is required and must be a non-empty string"
        )
        dataset_smiles_col = ""

    dataset_activity_col = data.get("dataset_activity_col")
    if not dataset_activity_col or not isinstance(dataset_activity_col, str):
        errors.append(
            "[data] 'dataset_activity_col' is required and must be a non-empty string"
        )
        dataset_activity_col = ""

    # seed data (optional)
    seed_data_path = data.get("seed_data_path", None)
    if seed_data_path is not None and not isinstance(seed_data_path, str):
        errors.append("[data] 'seed_data_path' must be a string path or null")
        seed_data_path = None

    seed_smiles_col = data.get("seed_smiles_col", "smiles")
    if not isinstance(seed_smiles_col, str) or not seed_smiles_col:
        errors.append("[data] 'seed_smiles_col' must be a non-empty string")
        seed_smiles_col = "smiles"

    seed_activity_col = data.get("seed_activity_col", "pEC50")
    if not isinstance(seed_activity_col, str) or not seed_activity_col:
        errors.append("[data] 'seed_activity_col' must be a non-empty string")
        seed_activity_col = "pEC50"

    # strategies
    strategies = al.get("strategies")
    if not strategies or not isinstance(strategies, list) or not strategies:
        errors.append("[active_learning] 'strategies' must be a non-empty list")
        strategies = []
    else:
        invalid = [s for s in strategies if s not in _VALID_STRATEGIES]
        if invalid:
            errors.append(
                f"[active_learning] unrecognised strategies: {invalid}. "
                f"Valid options: {_VALID_STRATEGIES}"
            )

    # seeds
    seeds = al.get("seeds")
    if (
        not seeds
        or not isinstance(seeds, list)
        or not all(isinstance(s, int) for s in seeds)
    ):
        errors.append("[active_learning] 'seeds' must be a non-empty list of integers")
        seeds = []

    # positive-int fields
    k_iter = _require_pos_int(al, "k_iter", "active_learning")
    query_size = _require_pos_int(al, "query_size", "active_learning")
    n_models = _require_pos_int(tr, "n_models", "training")
    max_epochs = _require_pos_int(tr, "max_epochs", "training")

    # n_start: 0 is valid (seed-data-only bootstrapping)
    n_start = al.get("n_start")
    if n_start is None:
        errors.append("[active_learning] 'n_start' is required")
        n_start = None
    elif not isinstance(n_start, int) or n_start < 0:
        errors.append(
            f"[active_learning] 'n_start' must be a non-negative integer, got {n_start!r}"
        )
        n_start = None

    if errors:
        raise ValueError(
            "Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    # At this point all values are validated; assert to satisfy static analysis.
    assert k_iter is not None
    assert query_size is not None
    assert n_start is not None
    assert n_models is not None
    assert max_epochs is not None

    return ALConfig(
        dataset_path=dataset_path,
        dataset_smiles_col=dataset_smiles_col,
        dataset_activity_col=dataset_activity_col,
        seed_data_path=seed_data_path,
        seed_smiles_col=seed_smiles_col,
        seed_activity_col=seed_activity_col,
        strategies=strategies,
        seeds=seeds,
        k_iter=k_iter,
        query_size=query_size,
        n_start=n_start,
        n_models=n_models,
        max_epochs=max_epochs,
    )

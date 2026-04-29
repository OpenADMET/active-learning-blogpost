"""Experiment configuration loading and validation for the AL pipeline.

All parsing and field validation lives here. The rest of the codebase only
ever sees ``ALConfig`` instances — never raw dicts or YAML strings.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import yaml

# Valid strategy universe — kept in sync with STRATEGIES in src/helpers.py
# Defined here directly to avoid any circular-import risk
_VALID_STRATEGIES = ["EI", "UCB", "Random", "Exploitation", "Exploration", "Diversity"]

# Valid split-type options
_VALID_SPLIT_TYPES = ["scaffold", "random", "cluster", "predefined"]


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
    dataset_split_seed : int
        Random seed passed to the splitter (``random_state``) for reproducible
        pool/test partitioning. Ignored when ``split_type`` is ``"predefined"``.
        Default is ``42``.
    gtm_seed : int
        Random seed for the GTM (Generative Topographic Map) embedding fitted
        on the pool compounds during setup. Changing this produces a different
        2D chemical-space layout for visualizations but does not affect model
        training or evaluation. Default is ``1234``.
    results_path : str
        Directory where setup pickles, run pickles, and output figures are
        written and read from. Tilde expansion is applied. Default is
        ``"results"``.
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
    split_types : list[str]
        Dataset split methods to run. Must be a non-empty subset of
        ``["scaffold", "random", "cluster", "predefined"]``. Each split type
        produces its own ``setup_<split_type>.pkl`` and set of run pickles,
        enabling side-by-side OOD vs. IID comparisons in ``analysis.py``.
        Use ``"predefined"`` when the dataset already contains a ``split``
        column (or the column named by ``predefined_split_col``) with values
        ``"train"``/``"Train"`` and ``"test"``/``"Test"`` (case-insensitive);
        the pipeline will honour that assignment
        instead of re-splitting.
    predefined_split_col : str
        Name of the column in the dataset that encodes a predetermined
        train/test split. Only used when ``"predefined"`` is in
        ``split_types``. Must contain exactly the values ``"train"``/``"Train"``
        and ``"test"``/``"Test"`` (case-insensitive). Default is ``"split"``.
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
        Hard ceiling on training epochs per committee member (> 0). Early
        stopping will typically trigger well before this limit.
    es_min_delta : float
        Minimum decrease in train loss that counts as an improvement for early
        stopping. Training stops when no improvement exceeds this threshold for
        ``es_patience`` consecutive epochs. Default is ``0.001``.
    es_patience : int
        Number of epochs with no improvement before early stopping triggers
        (> 0). Default is ``15``.
    use_chemeleon : bool
        Whether to initialise each committee member from CheMeleon pretrained
        weights (``from_chemeleon`` argument of ``ChemPropModel``). Set to
        ``False`` to train from random initialisation for ablation comparisons.
        Default is ``True``.
    cluster_method : str
        Clustering algorithm used when ``split_types`` includes ``"cluster"``.
        One of ``"butina"``, ``"kmeans"``, or ``"bemis-murcko"``. Default is
        ``"butina"``.
    cluster_k_clusters : int
        Number of clusters for the k-means clustering method. Ignored when
        ``cluster_method`` is not ``"kmeans"``. Default is ``10``.
    cluster_butina_cutoff : float
        Tanimoto distance threshold for Butina clustering. Must be in (0, 1).
        Ignored when ``cluster_method`` is not ``"butina"``. Default is
        ``0.65``.
    hit_threshold : float
        pEC50 (or pIC50) value at or above which a compound is classified as a
        hit for hit-discovery figures and the distribution animation. Typical
        values: ``6.0`` for PXR, ``7.0`` for ASAP Mpro. Default is ``6.0``.

    """

    dataset_path: str
    dataset_smiles_col: str
    dataset_activity_col: str
    dataset_split_seed: int
    gtm_seed: int
    results_path: str
    seed_data_path: str | None
    seed_smiles_col: str
    seed_activity_col: str
    strategies: list[str]
    split_types: list[str]
    seeds: list[int]
    k_iter: int
    query_size: int
    n_start: int
    n_models: int
    max_epochs: int
    es_min_delta: float
    es_patience: int
    use_chemeleon: bool
    cluster_method: str
    cluster_k_clusters: int
    cluster_butina_cutoff: float
    predefined_split_col: str
    hit_threshold: float


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
    cl = raw.get("clustering", {})

    errors: list[str] = []

    # Helper that validates a config field as a strictly positive integer
    # and appends a human-readable error message if the check fails.
    # All errors are collected before raising so the user sees every problem at once.
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

    # split seed: optional, defaults to 42 for backward compatibility
    dataset_split_seed = data.get("dataset_split_seed", 42)
    if not isinstance(dataset_split_seed, int):
        errors.append(
            f"[data] 'dataset_split_seed' must be an integer, got {dataset_split_seed!r}"
        )
        dataset_split_seed = 42

    # GTM seed: optional, defaults to 1234 for backward compatibility
    gtm_seed = data.get("gtm_seed", 1234)
    if not isinstance(gtm_seed, int):
        errors.append(
            f"[data] 'gtm_seed' must be an integer, got {gtm_seed!r}"
        )
        gtm_seed = 1234

    # results path: optional, defaults to "results" for backward compatibility
    results_path = data.get("results_path", "results")
    if not isinstance(results_path, str) or not results_path:
        errors.append("[data] 'results_path' must be a non-empty string")
        results_path = "results"

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

    # split_types
    split_types = al.get("split_types")
    if not split_types or not isinstance(split_types, list):
        errors.append("[active_learning] 'split_types' must be a non-empty list")
        split_types = []
    else:
        invalid_st = [s for s in split_types if s not in _VALID_SPLIT_TYPES]
        if invalid_st:
            errors.append(
                f"[active_learning] unrecognised split_types: {invalid_st}. "
                f"Valid options: {_VALID_SPLIT_TYPES}"
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

    # early stopping parameters: optional with evidence-based defaults
    es_min_delta = tr.get("es_min_delta", 0.001)
    if not isinstance(es_min_delta, (int, float)) or es_min_delta <= 0:
        errors.append(
            f"[training] 'es_min_delta' must be a positive float, got {es_min_delta!r}"
        )

    es_patience = tr.get("es_patience", 15)
    if not isinstance(es_patience, int) or es_patience <= 0:
        errors.append(
            f"[training] 'es_patience' must be a positive integer, got {es_patience!r}"
        )

    # use_chemeleon: optional bool, defaults to True for backward compatibility
    use_chemeleon = tr.get("use_chemeleon", True)
    if not isinstance(use_chemeleon, bool):
        errors.append(
            f"[training] 'use_chemeleon' must be a boolean (true/false), got {use_chemeleon!r}"
        )

    # clustering parameters: all optional with sensible defaults so existing configs
    # without a 'clustering:' section continue to load without errors
    _valid_cluster_methods = ["butina", "kmeans", "bemis-murcko"]
    cluster_method = cl.get("method", "butina")
    if not isinstance(cluster_method, str) or cluster_method not in _valid_cluster_methods:
        errors.append(
            f"[clustering] 'method' must be one of {_valid_cluster_methods}, "
            f"got {cluster_method!r}"
        )

    cluster_k_clusters = cl.get("k_clusters", 10)
    if not isinstance(cluster_k_clusters, int) or cluster_k_clusters <= 0:
        errors.append(
            f"[clustering] 'k_clusters' must be a positive integer, got {cluster_k_clusters!r}"
        )

    cluster_butina_cutoff = cl.get("butina_cutoff", 0.65)
    if (
        not isinstance(cluster_butina_cutoff, (int, float))
        or not (0 < cluster_butina_cutoff < 1)
    ):
        errors.append(
            f"[clustering] 'butina_cutoff' must be a float in (0, 1), "
            f"got {cluster_butina_cutoff!r}"
        )

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

    # predefined split column: optional, defaults to "split"
    predefined_split_col = data.get("predefined_split_col", "split")
    if not isinstance(predefined_split_col, str) or not predefined_split_col:
        errors.append(
            "[data] 'predefined_split_col' must be a non-empty string"
        )
        predefined_split_col = "split"

    # hit threshold: optional, defaults to 6.0 (PXR convention); ASAP configs should set 7.0
    hit_threshold = data.get("hit_threshold", 6.0)
    if not isinstance(hit_threshold, (int, float)) or hit_threshold <= 0:
        errors.append(
            f"[data] 'hit_threshold' must be a positive number, got {hit_threshold!r}"
        )
        hit_threshold = 6.0

    # cross-field check: n_start=0 requires seed_data_path
    if n_start == 0 and seed_data_path is None:
        errors.append(
            "[active_learning] 'n_start' can only be 0 when 'seed_data_path' is provided "
            "(the committee needs at least some labeled data at iteration 0); "
            "set n_start > 0 or provide a seed dataset."
        )

    if errors:
        raise ValueError(
            "Config validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    # At this point all values are validated; assert to satisfy static analysis
    assert k_iter is not None
    assert query_size is not None
    assert n_start is not None
    assert n_models is not None
    assert max_epochs is not None

    return ALConfig(
        dataset_path=dataset_path,
        dataset_smiles_col=dataset_smiles_col,
        dataset_activity_col=dataset_activity_col,
        dataset_split_seed=dataset_split_seed,
        gtm_seed=gtm_seed,
        results_path=results_path,
        seed_data_path=seed_data_path,
        seed_smiles_col=seed_smiles_col,
        seed_activity_col=seed_activity_col,
        strategies=strategies,
        split_types=split_types,
        seeds=seeds,
        k_iter=k_iter,
        query_size=query_size,
        n_start=n_start,
        n_models=n_models,
        max_epochs=max_epochs,
        es_min_delta=es_min_delta,
        es_patience=es_patience,
        use_chemeleon=use_chemeleon,
        cluster_method=cluster_method,
        cluster_k_clusters=cluster_k_clusters,
        cluster_butina_cutoff=cluster_butina_cutoff,
        predefined_split_col=predefined_split_col,
        hit_threshold=hit_threshold,
    )

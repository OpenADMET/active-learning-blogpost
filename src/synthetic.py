"""Synthetic noisy oracle for active learning simulations.

Replaces ChemProp/CheMeleon committee training with a fast oracle that
knows all true labels and simulates model predictions with configurable:

- **Target MAE**: uniform noise added to predictions; E[|noise|] = configured MAE.
- **Uncertainty quality**: generated sigma with a configurable Spearman rank
  correlation (rho) with prediction residuals, via Gaussian copula mixing.

Both MAE and Spearman rho can ramp linearly from an initial to a final value
over the course of the active learning campaign.

This lets us evaluate acquisition strategies under a controlled range of
uncertainty quality levels, independently of actual model training.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import yaml
from scipy.stats import norm, rankdata
from scipy.stats import spearmanr


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class SyntheticConfig:
    """Oracle hyper-parameters loaded from the ``oracle:`` section of a YAML config.

    Attributes
    ----------
    initial_mae : float
        Target mean absolute error (noise only) at iteration 0.
    final_mae : float
        Target mean absolute error (noise only) at the last iteration.
    initial_uncertainty_rho : float
        Target Spearman rank correlation between predicted sigma and
        |residuals| at iteration 0. Range [-1, 1]; typically in [0, 1].
    final_uncertainty_rho : float
        Target Spearman rho at the final iteration.
    initial_shrinkage : float
        Regression-toward-the-mean factor at iteration 0. Range [0, 1].
        1.0 = unbiased (current behavior); values below 1 compress predictions
        toward the global dataset mean, systematically underpredicting high
        actives as real ML models do when trained on limited data.
    final_shrinkage : float
        Shrinkage factor at the final iteration (typically higher than
        initial_shrinkage, mirroring improved extrapolation with more data).
    noise_heteroscedasticity : float
        Scales noise amplitude by distance above the global mean.  Empirically
        calibrated from real ChemProp/CheMeleon run residuals: ASAP Mpro shows
        strong positive heteroscedasticity (Spearman ρ≈0.60, Q4/Q1=3.6×);
        PXR shows no significant positive heteroscedasticity (ρ≈−0.09).
        Set to 0.0 for PXR configs; ~0.6 for ASAP configs.
        Formula: ``noise_scale = 1 + het * max(y_true − μ, 0) / σ_y``.
    """

    initial_mae: float
    final_mae: float
    initial_uncertainty_rho: float
    final_uncertainty_rho: float
    initial_shrinkage: float = 1.0
    final_shrinkage: float = 1.0
    noise_heteroscedasticity: float = 0.0


def load_synthetic_config(path: str | Path) -> SyntheticConfig:
    """Load the oracle configuration from the ``oracle:`` section of a YAML file.

    Parameters
    ----------
    path : str or Path
        Path to the YAML config file.

    Returns
    -------
    SyntheticConfig
        Validated oracle configuration.

    Raises
    ------
    FileNotFoundError
        If the config file does not exist.
    ValueError
        If the ``oracle:`` section is missing or contains invalid values.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")

    with open(path) as fh:
        raw = yaml.safe_load(fh)

    oracle = raw.get("oracle")
    if oracle is None:
        raise ValueError(
            f"Config file {path} is missing the required 'oracle:' section."
        )

    errors: list[str] = []

    def _get_float(key: str) -> float | None:
        val = oracle.get(key)
        if val is None:
            errors.append(f"[oracle] '{key}' is required")
            return None
        if not isinstance(val, (int, float)):
            errors.append(f"[oracle] '{key}' must be a number, got {val!r}")
            return None
        return float(val)

    initial_mae = _get_float("initial_mae")
    final_mae = _get_float("final_mae")
    initial_rho = _get_float("initial_uncertainty_rho")
    final_rho = _get_float("final_uncertainty_rho")

    # Shrinkage is optional; defaults to 1.0 (unbiased) for backward compatibility.
    def _get_float_optional(key: str, default: float) -> float:
        val = oracle.get(key)
        if val is None:
            return default
        if not isinstance(val, (int, float)):
            errors.append(f"[oracle] '{key}' must be a number, got {val!r}")
            return default
        return float(val)

    initial_shrinkage = _get_float_optional("initial_shrinkage", 1.0)
    final_shrinkage = _get_float_optional("final_shrinkage", 1.0)
    noise_heteroscedasticity = _get_float_optional("noise_heteroscedasticity", 0.0)

    if initial_mae is not None and initial_mae < 0:
        errors.append(f"[oracle] 'initial_mae' must be >= 0, got {initial_mae}")
    if final_mae is not None and final_mae < 0:
        errors.append(f"[oracle] 'final_mae' must be >= 0, got {final_mae}")
    for name, val in [("initial_uncertainty_rho", initial_rho), ("final_uncertainty_rho", final_rho)]:
        if val is not None and not (-1.0 <= val <= 1.0):
            errors.append(f"[oracle] '{name}' must be in [-1, 1], got {val}")
    for name, val in [("initial_shrinkage", initial_shrinkage), ("final_shrinkage", final_shrinkage)]:
        if not (0.0 <= val <= 1.0):
            errors.append(f"[oracle] '{name}' must be in [0, 1], got {val}")
    if noise_heteroscedasticity < 0.0:
        errors.append(
            f"[oracle] 'noise_heteroscedasticity' must be >= 0, got {noise_heteroscedasticity}"
        )

    if errors:
        raise ValueError(
            "Oracle config validation failed:\n" + "\n".join(f"  - {e}" for e in errors)
        )

    assert initial_mae is not None and final_mae is not None
    assert initial_rho is not None and final_rho is not None

    return SyntheticConfig(
        initial_mae=initial_mae,
        final_mae=final_mae,
        initial_uncertainty_rho=initial_rho,
        final_uncertainty_rho=final_rho,
        initial_shrinkage=initial_shrinkage,
        final_shrinkage=final_shrinkage,
        noise_heteroscedasticity=noise_heteroscedasticity,
    )


# ---------------------------------------------------------------------------
# Oracle
# ---------------------------------------------------------------------------


class SyntheticOracle:
    """Simulates model predictions with controlled accuracy and uncertainty quality.

    The oracle knows all true labels (``smiles_to_y`` lookup dict) and produces
    predictions by adding uniform noise calibrated to a target MAE.  Uncertainty
    estimates are generated via a Gaussian copula so their Spearman rank
    correlation with the prediction residuals matches a configurable target rho.

    Both MAE and rho can ramp linearly from an initial to a final value over the
    course of the campaign.

    Parameters
    ----------
    smiles_to_y : dict[str, float]
        Mapping from SMILES string to true activity value.  Should cover all
        compounds that will be passed to :meth:`predict`.
    oracle_cfg : SyntheticConfig
        Oracle hyper-parameters controlling noise and uncertainty quality.
    k_iter : int
        Total number of active learning iterations (used for linear ramping).
    seed : int, optional
        Base random seed for reproducible noise generation.  Default is 42.
    """

    def __init__(
        self,
        smiles_to_y: dict[str, float],
        oracle_cfg: SyntheticConfig,
        k_iter: int,
        seed: int = 42,
    ) -> None:
        self.smiles_to_y = smiles_to_y
        self.oracle_cfg = oracle_cfg
        self.k_iter = max(k_iter, 1)
        self.seed = seed
        # Global mean and std across all oracle-known compounds.
        y_vals = np.array(list(smiles_to_y.values()), dtype=float)
        self.y_global_mean = float(np.mean(y_vals))
        self.y_global_std = float(np.std(y_vals)) or 1.0  # guard div-by-zero

    # ------------------------------------------------------------------
    # Parameter ramping helpers
    # ------------------------------------------------------------------

    def _mae_at(self, iteration: int) -> float:
        """Return the target MAE at a given iteration (linearly ramped)."""
        frac = min(iteration / self.k_iter, 1.0)
        return self.oracle_cfg.initial_mae + (
            self.oracle_cfg.final_mae - self.oracle_cfg.initial_mae
        ) * frac

    def _rho_at(self, iteration: int) -> float:
        """Return the target Spearman rho at a given iteration (linearly ramped)."""
        frac = min(iteration / self.k_iter, 1.0)
        return self.oracle_cfg.initial_uncertainty_rho + (
            self.oracle_cfg.final_uncertainty_rho
            - self.oracle_cfg.initial_uncertainty_rho
        ) * frac

    def _shrinkage_at(self, iteration: int) -> float:
        """Return the shrinkage factor at a given iteration (linearly ramped)."""
        frac = min(iteration / self.k_iter, 1.0)
        return self.oracle_cfg.initial_shrinkage + (
            self.oracle_cfg.final_shrinkage - self.oracle_cfg.initial_shrinkage
        ) * frac

    # ------------------------------------------------------------------
    # Core prediction
    # ------------------------------------------------------------------

    def predict(
        self,
        smiles_list: list[str],
        iteration: int,
        call_id: int = 0,
    ) -> tuple[np.ndarray, np.ndarray]:
        """Predict activity and uncertainty for a list of SMILES strings.

        Looks up true labels, adds uniform noise to achieve the target MAE for
        this iteration, and generates sigma values whose Spearman rank
        correlation with |residuals| matches the target rho.

        Parameters
        ----------
        smiles_list : list[str]
            SMILES strings to predict.
        iteration : int
            Current AL iteration index (0-based), used for ramping.
        call_id : int, optional
            Differentiates multiple predict calls within the same iteration
            (e.g. test evaluation vs pool query) so each gets independent noise.
            Default is 0.

        Returns
        -------
        mean_pred : np.ndarray, shape (n, 1)
            Predicted mean activity values.
        std_pred : np.ndarray, shape (n, 1)
            Predicted standard deviations (uncertainty estimates).
        """
        n = len(smiles_list)
        rng = np.random.RandomState(self.seed + iteration * 10007 + call_id)

        mae = self._mae_at(iteration)
        rho = self._rho_at(iteration)
        shrinkage = self._shrinkage_at(iteration)
        het = self.oracle_cfg.noise_heteroscedasticity

        # Look up true labels
        y_true = np.array([self.smiles_to_y[s] for s in smiles_list], dtype=float)

        # Regression-toward-the-mean: compress predictions toward the global mean.
        # shrinkage=1.0 → unbiased; shrinkage<1 → high actives systematically
        # underpredicted, mirroring real ML models trained on limited data.
        y_biased = self.y_global_mean + shrinkage * (y_true - self.y_global_mean)

        # Heteroscedastic noise: amplitude scales with distance above the global
        # mean, simulating larger errors for rare high-activity compounds that have
        # few training examples.  Empirically calibrated from real run residuals:
        # ASAP shows Q4/Q1 residual ratio ~2–3.6×; PXR shows no significant
        # positive heteroscedasticity (het=0 appropriate for PXR).
        above_mean = np.maximum(y_true - self.y_global_mean, 0.0)
        noise_scale = 1.0 + het * above_mean / self.y_global_std
        noise = rng.uniform(-1.0, 1.0, size=n) * 2.0 * mae * noise_scale
        y_pred = y_biased + noise

        # Sigma with target Spearman rho vs |residuals| (includes bias + noise)
        residuals = np.abs(y_pred - y_true)
        sigma = self._make_sigma(residuals, rho, rng)

        return y_pred.reshape(-1, 1), sigma.reshape(-1, 1)

    def _make_sigma(
        self,
        residuals: np.ndarray,
        rho: float,
        rng: np.random.RandomState,
    ) -> np.ndarray:
        """Generate sigma values with a target Spearman rho vs residuals.

        Uses a Gaussian copula: the mixed latent variable is
        ``z_mix = rho * z_r + sqrt(1 - rho²) * z_ind``, then mapped back
        through the empirical CDF of residuals so the sigma marginal matches
        the residual distribution (realistic scale).

        Parameters
        ----------
        residuals : np.ndarray
            Absolute prediction errors, shape (n,).
        rho : float
            Target Spearman rank correlation between sigma and residuals.
        rng : np.random.RandomState
            Pre-seeded RNG for reproducibility.

        Returns
        -------
        sigma : np.ndarray, shape (n,)
            Uncertainty estimates; always positive.
        """
        n = len(residuals)

        if n == 1:
            # Spearman rho is undefined for a single point; return a constant.
            return np.maximum(residuals, 1e-6)

        # Gaussian copula latent scores
        rank_r = rankdata(residuals, method="average")
        z_r = norm.ppf(rank_r / (n + 1.0))
        z_ind = rng.standard_normal(n)

        # Mix: z_mix has target Spearman rho ≈ rho with residuals
        rho_clipped = float(np.clip(rho, -1.0 + 1e-9, 1.0 - 1e-9))
        z_mix = rho_clipped * z_r + np.sqrt(max(0.0, 1.0 - rho_clipped**2)) * z_ind

        # Map z_mix back to the marginal distribution of residuals
        pct = norm.cdf(z_mix)
        # Avoid exact 0/1 percentiles (can hit boundary of residuals range)
        pct = np.clip(pct, 1e-9, 1.0 - 1e-9)
        sigma = np.quantile(residuals, pct)

        return np.maximum(sigma, 1e-6)

    def calibrate_uncertainty(self, *args, **kwargs) -> None:
        """No-op: oracle uncertainty is controlled by construction."""

    def actual_sigma_error_rho(
        self, smiles_list: list[str], iteration: int, call_id: int = 0
    ) -> float:
        """Compute the actual Spearman rho(sigma, |error|) for diagnostics.

        Parameters
        ----------
        smiles_list : list[str]
            SMILES strings to evaluate.
        iteration : int
            Current AL iteration index.
        call_id : int, optional
            Passed through to :meth:`predict`. Default is 0.

        Returns
        -------
        float
            Achieved Spearman rank correlation between sigma and |error|.
        """
        mean_pred, std_pred = self.predict(smiles_list, iteration, call_id=call_id)
        y_true = np.array([self.smiles_to_y[s] for s in smiles_list], dtype=float)
        residuals = np.abs(mean_pred.flatten() - y_true)
        return float(spearmanr(std_pred.flatten(), residuals).statistic)


# ---------------------------------------------------------------------------
# Acquisition strategies for the oracle
# ---------------------------------------------------------------------------


def _expected_improvement(
    mean: np.ndarray,
    std: np.ndarray,
    best_y: float,
    xi: float = 0.01,
) -> np.ndarray:
    """Expected improvement acquisition scores."""
    z = (mean - best_y - xi) / np.maximum(std, 1e-9)
    return (mean - best_y - xi) * norm.cdf(z) + std * norm.pdf(z)


def _upper_confidence_bound(
    mean: np.ndarray,
    std: np.ndarray,
    kappa: float = 1.0,
) -> np.ndarray:
    """Upper confidence bound acquisition scores."""
    return mean + kappa * std


def query_batch_synthetic(
    oracle: SyntheticOracle,
    smiles_unlabeled: list[str],
    strategy: str,
    best_y: float,
    iteration: int,
    size: int = 100,
    seed: int = 42,
    unlabeled_gtm_coords: np.ndarray | None = None,
    labeled_gtm_coords: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray | None]:
    """Select a batch from the unlabeled pool using a synthetic oracle.

    Model-based strategies (EI, UCB, Exploitation, Exploration) use the
    oracle's predicted mean and sigma directly.  Random uses only the RNG.
    Diversity uses GTM coordinates (no model required).

    Parameters
    ----------
    oracle : SyntheticOracle
        Fitted oracle providing label lookups and uncertainty estimates.
    smiles_unlabeled : list[str]
        SMILES of unlabeled candidate compounds.
    strategy : str
        One of ``"EI"``, ``"UCB"``, ``"Random"``, ``"Exploitation"``,
        ``"Exploration"``.
    best_y : float
        Best observed activity so far (used by EI).
    iteration : int
        Current AL iteration index (for oracle ramping).
    size : int, optional
        Number of compounds to select. Default is 100.
    seed : int, optional
        RNG seed for the random strategy. Default is 42.
    unlabeled_gtm_coords : np.ndarray or None, optional
        GTM coordinates for the unlabeled pool; not used (Diversity excluded).
    labeled_gtm_coords : np.ndarray or None, optional
        GTM coordinates for the labeled set; not used (Diversity excluded).

    Returns
    -------
    top_indices : np.ndarray
        Selected indices relative to the unlabeled pool.
    scores : np.ndarray or None
        Acquisition scores for selected compounds, or ``None`` for Random.
    """
    n_unlabeled = len(smiles_unlabeled)

    if strategy == "Random":
        rng = np.random.RandomState(seed)
        selected = rng.choice(n_unlabeled, size=min(size, n_unlabeled), replace=False)
        return selected, None

    # Oracle predictions on the unlabeled pool (call_id=1 to differ from test eval)
    mean_pred, std_pred = oracle.predict(smiles_unlabeled, iteration, call_id=1)
    mean_1d = mean_pred.flatten()
    std_1d = std_pred.flatten()

    if strategy == "Exploitation":
        scores = mean_1d
    elif strategy == "Exploration":
        scores = std_1d
    elif strategy == "EI":
        scores = _expected_improvement(mean_1d, std_1d, best_y)
    elif strategy == "UCB":
        scores = _upper_confidence_bound(mean_1d, std_1d)
    else:
        raise ValueError(
            f"Unknown strategy {strategy!r}. "
            "Valid options: 'EI', 'UCB', 'Random', 'Exploitation', 'Exploration'."
        )

    top_indices = np.argsort(scores)[::-1][:size]
    return top_indices, scores


# ---------------------------------------------------------------------------
# Evaluation helpers (mirrors src/helpers.evaluate_on_test)
# ---------------------------------------------------------------------------


def _evaluate_oracle_on_test(
    oracle: SyntheticOracle,
    smiles_test: list[str],
    y_test: np.ndarray,
    iteration: int,
    call_id: int = 0,
) -> dict:
    """Evaluate oracle predictions against a held-out test set.

    Mirrors the return signature of ``src.helpers.evaluate_on_test`` so that
    the history records are identical and downstream analysis is unchanged.

    Parameters
    ----------
    oracle : SyntheticOracle
        Trained (or here, parametric) oracle.
    smiles_test : list[str]
        SMILES strings for test compounds.
    y_test : np.ndarray
        Ground-truth activity values, shape (n_test,).
    iteration : int
        Current AL iteration (for ramping parameters).
    call_id : int, optional
        Differentiates multiple calls per iteration. Default is 0.

    Returns
    -------
    dict
        Keys: ``"mae"``, ``"r2"``, ``"ktau"``, ``"spearmanr"``,
        ``"miscal_area"``, ``"y_test_pred"``, ``"y_test_std"``.
    """
    from scipy.stats import kendalltau, spearmanr as sp
    from openadmet.models.eval.uncertainty import UncertaintyMetrics

    mean_pred, std_pred = oracle.predict(smiles_test, iteration, call_id=call_id)
    mean_1d = mean_pred.flatten()
    std_1d = std_pred.flatten()
    y_1d = np.asarray(y_test).flatten()

    residuals = y_1d - mean_1d
    mae = float(np.mean(np.abs(residuals)))
    ss_res = np.sum(residuals**2)
    ss_tot = np.sum((y_1d - np.mean(y_1d)) ** 2)
    r2 = float(1.0 - ss_res / ss_tot) if ss_tot > 0 else 0.0
    ktau = float(kendalltau(y_1d, mean_1d).statistic)
    spearman = float(sp(y_1d, mean_1d).statistic)

    # Miscalibration area via openadmet UncertaintyMetrics
    unc_metrics = UncertaintyMetrics()
    unc_metrics.evaluate(y_1d.reshape(-1, 1), mean_pred, std_pred)
    unc_res = unc_metrics.report()["task_0"]
    miscal_area = float(unc_res["miscal_area"])

    return {
        "mae": mae,
        "r2": r2,
        "ktau": ktau,
        "spearmanr": spearman,
        "miscal_area": miscal_area,
        "y_test_pred": mean_1d,
        "y_test_std": std_1d,
    }


# ---------------------------------------------------------------------------
# Active learning loop
# ---------------------------------------------------------------------------


def run_active_learning_synthetic(
    df_pool: pd.DataFrame,
    df_test: pd.DataFrame,
    oracle: SyntheticOracle,
    n_start: int = 100,
    k_iter: int = 15,
    query_size: int = 20,
    seed: int = 42,
    strategy: str = "Random",
    verbose: bool = True,
    df_seed: pd.DataFrame | None = None,
    gtm_coords: np.ndarray | None = None,
) -> dict:
    """Execute a full active learning loop using the synthetic oracle.

    Mirrors :func:`src.helpers.run_active_learning` exactly — same loop
    structure, same history keys, same output format — but replaces committee
    training with oracle predictions.  This makes all downstream analysis
    (``analysis.py``, ``synthetic_analysis.py``) reusable unchanged.

    Parameters
    ----------
    df_pool : pd.DataFrame
        Candidate pool with columns ``"smiles"`` and ``"pEC50"``.
    df_test : pd.DataFrame
        Held-out test set with columns ``"smiles"`` and ``"pEC50"``.
    oracle : SyntheticOracle
        Configured oracle; must have ``smiles_to_y`` entries for all pool,
        test, and seed compounds.
    n_start : int, optional
        Initial labeled pool size. Default is 100.
    k_iter : int, optional
        Number of AL iterations. Default is 15.
    query_size : int, optional
        Compounds queried per iteration. Default is 20.
    seed : int, optional
        Outer random seed for initial selection and query tie-breaking.
        Default is 42.
    strategy : str, optional
        Acquisition strategy. One of ``"EI"``, ``"UCB"``, ``"Random"``,
        ``"Exploitation"``, ``"Exploration"``. Default is ``"Random"``.
    verbose : bool, optional
        Whether to print per-iteration progress. Default is True.
    df_seed : pd.DataFrame or None, optional
        External seed training data (e.g. ChEMBL). Not queried; not counted
        in ``n_labeled``. Default is None.
    gtm_coords : np.ndarray or None, optional
        Precomputed GTM 2D coordinates for ``df_pool``, shape ``(n_pool, 2)``.
        Stored in history for visualization but not used by the oracle.
        Default is None.

    Returns
    -------
    dict
        ``{"history": list[dict], "committee": None}``
        History entries are identical in structure to those produced by
        :func:`src.helpers.run_active_learning`.
    """
    rng = np.random.RandomState(seed)
    n_total = len(df_pool)
    all_indices = np.arange(n_total)

    labeled_mask = np.zeros(n_total, dtype=bool)
    if n_start > 0:
        initial_idx = rng.choice(all_indices, size=n_start, replace=False)
        labeled_mask[initial_idx] = True

    prev_labeled_mask = np.zeros(n_total, dtype=bool)
    history: list[dict] = []

    smiles_test = df_test["smiles"].tolist()
    y_test_arr = df_test["pEC50"].to_numpy()

    for k in range(k_iter + 1):
        labeled_idx = np.where(labeled_mask)[0]
        unlabeled_idx = np.where(~labeled_mask)[0]
        newly_labeled_idx = np.where(labeled_mask & ~prev_labeled_mask)[0]
        prev_labeled_mask = labeled_mask.copy()

        df_labeled = df_pool.iloc[labeled_idx]

        # best_y across pool-acquired + seed (consistent with real loop)
        if len(df_labeled) > 0 and df_seed is not None and len(df_seed) > 0:
            best_y = max(df_labeled["pEC50"].max(), df_seed["pEC50"].max())
        elif len(df_labeled) > 0:
            best_y = df_labeled["pEC50"].max()
        elif df_seed is not None and len(df_seed) > 0:
            best_y = df_seed["pEC50"].max()
        else:
            raise ValueError(
                "No labeled pool compounds and no seed data at iteration 0. "
                "Either set n_start > 0 or provide df_seed."
            )

        if verbose:
            n_seed_log = len(df_seed) if df_seed is not None else 0
            print(
                f"Iter {k}: {len(df_labeled)} pool-labeled + {n_seed_log} seed. "
                f"Best pEC50: {best_y:.2f}"
            )

        # Evaluate BEFORE calibration (oracle has no calibration, so both are identical)
        res_pre = _evaluate_oracle_on_test(oracle, smiles_test, y_test_arr, iteration=k, call_id=0)
        # No-op calibration step — calibrate_uncertainty does nothing on SyntheticOracle
        oracle.calibrate_uncertainty()
        # Evaluate AFTER calibration (identical to pre for the oracle)
        res = _evaluate_oracle_on_test(oracle, smiles_test, y_test_arr, iteration=k, call_id=0)

        state: dict = {
            "iteration": k,
            "n_labeled": len(df_labeled),
            "n_seed": len(df_seed) if df_seed is not None else 0,
            "best_y": best_y,
            "pool_y_values": df_labeled["pEC50"].values.tolist(),
            "selected_pool_indices": newly_labeled_idx.tolist(),
            "y_test_pred_pre_cal": res_pre["y_test_pred"],
            "y_test_std_pre_cal": res_pre["y_test_std"],
            "miscal_area_pre_cal": res_pre["miscal_area"],
            **res,
        }
        history.append(state)

        if k < k_iter:
            df_unlabeled = df_pool.iloc[unlabeled_idx]
            rel_idx, scores = query_batch_synthetic(
                oracle,
                df_unlabeled["smiles"].tolist(),
                strategy,
                best_y,
                iteration=k,
                size=query_size,
                seed=seed + k,
                unlabeled_gtm_coords=gtm_coords[unlabeled_idx] if gtm_coords is not None else None,
                labeled_gtm_coords=gtm_coords[labeled_idx] if gtm_coords is not None else None,
            )
            abs_idx = unlabeled_idx[rel_idx]
            labeled_mask[abs_idx] = True

            if scores is not None:
                history[-1]["acquisition_scores"] = scores

    return {"history": history, "committee": None}

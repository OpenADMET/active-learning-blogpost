import pathlib
import tempfile

import numpy as np
import pandas as pd
import torch
import useful_rdkit_utils as uru
from chemographykit.gtm import GTM
from chemographykit.utils.molecules import calculate_latent_coords
from lightning import pytorch as pl
from openadmet.models.active_learning.committee import CommitteeRegressor
from openadmet.models.architecture.chemprop import ChemPropModel
from openadmet.models.eval.regression import RegressionMetrics
from openadmet.models.eval.uncertainty import UncertaintyMetrics
from openadmet.models.features.chemprop import ChemPropFeaturizer
from openadmet.models.trainer.lightning import LightningTrainer
from rdkit import Chem
from scipy.spatial.distance import cdist

STRATEGIES = ["EI", "UCB", "Random", "Exploitation", "Exploration", "Diversity"]

STRATEGY_COLORS = {
    "EI": "#2d6a4f",
    "UCB": "#1d3557",
    "Random": "#e76f51",
    "Exploitation": "#9b2226",
    "Exploration": "#7b2d8b",
    "Diversity": "#0077b6",
}

STRATEGY_QUERY_KEYS = {
    "EI": "expected-improvement",
    "UCB": "upper-confidence-bound",
    "Random": None,
    "Exploitation": "exploitation",
    "Exploration": "max-uncertainty-reduction",
    "Diversity": None,
}


def smiles_to_gtm(
    smiles_list,
    num_nodes=36**2,
    num_basis_functions=31**2,
    basis_width=5.726809,
    reg_coeff=543.61223,
    max_iter=300,
    tolerance=0.001,
    standardize=False,
    seed=1234,
    pca_scale=True,
    device="cpu",
):
    """
    Fit a GTM (Generative Topographic Mapping) model to a list of SMILES strings.

    Computes RDKit descriptors, scales them, then fits and projects a GTM model
    to produce 2D coordinates and per-sample responsibilities.

    Parameters
    ----------
    smiles_list : list[str]
        SMILES strings for the molecules to embed.
    num_nodes : int, optional
        Number of grid nodes in the GTM map (total = sqrt(num_nodes) x sqrt(num_nodes)).
        Default is 36**2.
    num_basis_functions : int, optional
        Number of RBF basis functions for GTM. Default is 31**2.
    basis_width : float, optional
        Width of each RBF basis function. Default is 5.726809.
    reg_coeff : float, optional
        Regularization coefficient for GTM. Default is 543.61223.
    max_iter : int, optional
        Maximum number of EM iterations. Default is 300.
    tolerance : float, optional
        EM convergence tolerance. Default is 0.001.
    standardize : bool, optional
        Whether to internally standardize descriptors inside GTM. Default is False.
    seed : int, optional
        Random seed for reproducibility. Default is 1234.
    pca_scale : bool, optional
        Whether to initialize the GTM grid using PCA scaling. Default is True.
    device : str, optional
        Device for GTM computations (e.g., ``"cpu"``, ``"cuda"``, ``"mps"``).
        Default is ``"cpu"``.

    Returns
    -------
    gtm : GTM
        Fitted GTM model object.
    crds_2d : np.ndarray
        2D GTM projections of shape (n_samples, 2).
    resps : np.ndarray
        GTM responsibilities of shape (n_samples, num_nodes).
    llhs : np.ndarray
        Per-sample log-likelihoods of shape (n_samples,).
    """

    # Calculate descriptors
    rdkit_desc = uru.RDKitDescriptors()
    X_desc = np.stack([rdkit_desc.calc_smiles(smi) for smi in smiles_list])

    # Scale
    X_desc, _ = uru.clean_and_scale_descriptors(X_desc)

    # Initialize GTM
    gtm = GTM(
        num_nodes=num_nodes,
        num_basis_functions=num_basis_functions,
        basis_width=basis_width,
        reg_coeff=reg_coeff,
        max_iter=max_iter,
        tolerance=tolerance,
        standardize=standardize,
        seed=seed,
        device=device,
        pca_scale=pca_scale,
    )

    # Convert to tensor if needed
    if not isinstance(X_desc, torch.Tensor):
        X_desc = torch.tensor(X_desc, dtype=torch.float64, device=device)

    # Fit GTM model
    gtm.fit_transform(X_desc)

    # Get responsibilities and log-likelihoods
    resps, llhs = gtm.project(X_desc)
    resps = resps.detach().cpu().numpy()
    llhs = llhs.detach().cpu().numpy()

    # Calculate 2D coordinates from responsibilities
    crds_2d = calculate_latent_coords(resps, correction=True, return_node=True)
    crds_2d = crds_2d.values

    return gtm, crds_2d, resps, llhs


def smiles_to_tmap(
    smiles_list,
    fp_size: int = 2048,
    fp_radius: int = 3,
    lsh_dim: int = 128,
    k: int = 50,
    sl_repeats: int = 2,
    mmm_repeats: int = 2,
    node_size: int = 1,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Compute a TMAP layout for a list of SMILES strings.

    Encodes each molecule as an MHFP fingerprint, indexes all fingerprints
    into an LSH forest, and runs ``tm.layout_from_lsh_forest`` to produce 2D
    node coordinates and minimum-spanning-tree edges.

    Parameters
    ----------
    smiles_list : list[str]
        SMILES of compounds in the pool, in the same row order expected by
        downstream consumers.
    fp_size : int
        MHFP fingerprint size (number of bits). Default 2048.
    fp_radius : int
        MHFP encoding radius. Default 3.
    lsh_dim : int
        Number of LSH permutations. Default 128.
    k : int
        Number of nearest neighbours used by the TMAP layout. Default 50.
    sl_repeats, mmm_repeats : int
        Layout refinement repetitions passed to ``tm.LayoutConfiguration``.
    node_size : int
        Node size hint for ``tm.LayoutConfiguration``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]
        ``(x, y, s, t)`` — node x-coordinates, node y-coordinates, edge
        source indices, and edge target indices, all as numpy arrays of length
        n_compounds (for x/y) or n_edges (for s/t).
    """
    import tmap as tm  # deferred: not available on all platforms
    from mhfp.encoder import MHFPEncoder  # deferred: depends on tmap

    enc = MHFPEncoder(fp_size, fp_radius)
    # NumPy 2.0+ raises OverflowError when the Mersenne prime (2^61-1) used
    # in mhfp's hash arithmetic is larger than the uint32 permutation arrays.
    # Casting to uint64 gives enough headroom for the modular arithmetic.
    enc.permutations_a = enc.permutations_a.astype(np.uint64)
    enc.permutations_b = enc.permutations_b.astype(np.uint64)
    lf = tm.LSHForest(fp_size, lsh_dim)

    fps = []
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is not None:
            fps.append(tm.VectorUint(enc.encode_mol(mol, min_radius=0)))
        else:
            fps.append(tm.VectorUint([0] * fp_size))

    lf.batch_add(fps)
    lf.index()

    cfg = tm.LayoutConfiguration()
    cfg.k = k
    cfg.sl_repeats = sl_repeats
    cfg.mmm_repeats = mmm_repeats
    cfg.node_size = node_size

    x, y, s, t, _ = tm.layout_from_lsh_forest(lf, config=cfg)
    return np.array(x), np.array(y), np.array(s, dtype=int), np.array(t, dtype=int)


def featurize(smiles_list, y_list=None, shuffle=False):
    """Featurize a list of SMILES strings for use with ChemProp.

    Parameters
    ----------
    smiles_list : list[str]
        SMILES strings to featurize.
    y_list : array-like or None, optional
        Target values corresponding to each SMILES string. Pass ``None`` for
        inference-only featurization. Default is None.
    shuffle : bool, optional
        Whether to shuffle the dataset when creating the data loader.
        Default is False.

    Returns
    -------
    loader : DataLoader
        PyTorch data loader ready for model training or inference.
    scaler : object
        Fitted target scaler (used to inverse-transform predictions).
    """
    featurizer = ChemPropFeaturizer(batch_size=64, shuffle=shuffle, n_jobs=0)
    loader, indices, scaler, dataset = featurizer.featurize(smiles_list, y_list)
    return loader, scaler


def build_committee_member(seed=42, max_epochs=20, log_dir=False):
    """Construct a single ChemProp committee member paired with a LightningTrainer.

    Parameters
    ----------
    seed : int, optional
        Random seed passed to ``pl.seed_everything`` for full reproducibility
        across model weights and data-loader shuffling. Default is 42.
    max_epochs : int, optional
        Maximum number of training epochs for the LightningTrainer. Default is 20.
    log_dir : str or False, optional
        Output directory for training logs and checkpoints. Pass ``False`` (or
        ``None``) to disable logging. Default is False.

    Returns
    -------
    model : ChemPropModel
        Initialized (but untrained) ChemProp model.
    trainer : LightningTrainer
        Configured trainer linked to ``model``.
    """

    pl.seed_everything(seed, workers=True)

    # Define the model
    model = ChemPropModel(
        n_tasks=1,
        from_chemeleon=True,
        ffn_hidden_dim=512,
        ffn_hidden_num_layers=3,
        mpnn_lr=1e-4,
        ffn_lr=1e-3,
        mpnn_weight_decay=0,
        ffn_weight_decay=1e-4,
        dropout=0.25,
        batch_norm=False,
        scheduler="plateau",
        reduce_lr_patience=5,
        reduce_lr_factor=0.5,
        monitor_metric="val_loss",
        metric_list=["mae", "rmse"],
    )

    _output_dir = (
        pathlib.Path(log_dir)
        if log_dir
        else pathlib.Path(tempfile.mkdtemp(prefix="al_logs_"))
    )

    # Define the trainer
    trainer = LightningTrainer(
        output_dir=_output_dir,
        max_epochs=max_epochs,
        accelerator="gpu",
        devices=1,
        early_stopping=False,
        gradient_clip_val=0.5,
    )

    # Link them
    trainer.model = model
    return model, trainer


def train_committee(smiles_labeled, y_labeled, n_models=5, seed=42, max_epochs=20):
    """Train a committee of ChemProp models on bootstrapped data.

    Each committee member is trained on a bootstrap resample of the labeled set,
    providing ensemble-based uncertainty estimates via prediction disagreement.

    Parameters
    ----------
    smiles_labeled : pd.Series or list[str]
        SMILES strings for the labeled training compounds.
    y_labeled : pd.Series or np.ndarray
        Target values corresponding to each labeled compound.
    n_models : int, optional
        Number of committee members to train. Default is 5.
    seed : int, optional
        Base random seed; each member uses ``seed + i`` for reproducibility.
        Default is 42.
    max_epochs : int, optional
        Maximum training epochs per committee member. Default is 20.

    Returns
    -------
    committee : CommitteeRegressor
        Assembled committee of trained ChemProp models.
    """
    members = []
    rng = np.random.RandomState(seed)

    for i in range(n_models):
        with tempfile.TemporaryDirectory(prefix="al_logs_") as tmp_dir:
            # Build and seed member first so pl.seed_everything controls dataloader shuffle
            model, trainer = build_committee_member(
                seed=seed + i, max_epochs=max_epochs, log_dir=tmp_dir
            )

            # Bootstrap resampling
            n_samples = len(smiles_labeled)
            boot_idx = rng.choice(n_samples, size=n_samples, replace=True)
            X_boot = smiles_labeled.iloc[boot_idx].tolist()
            y_boot = y_labeled.iloc[boot_idx].values

            # Prepare data
            train_loader, scaler = featurize(X_boot, y_boot, shuffle=True)

            model.build(scaler=scaler)
            trainer.build(no_val=True)

            trainer.train(train_loader, None)

        members.append(model)

    # Assemble committee
    committee = CommitteeRegressor.from_models(members)
    return committee


def query_batch(
    committee,
    smiles_unlabeled,
    strategy,
    best_y,
    size=100,
    seed=42,
    unlabeled_gtm_coords=None,
    labeled_gtm_coords=None,
):
    """
    Select a batch of molecules from the unlabeled pool using the given acquisition strategy.

    Parameters
    ----------
    committee : CommitteeRegressor
        Trained committee used for uncertainty-based strategies.
    smiles_unlabeled : list[str]
        SMILES of unlabeled candidates.
    strategy : str
        Acquisition strategy name. One of ``"EI"``, ``"UCB"``, ``"Random"``,
        ``"Exploitation"``, ``"Exploration"``, or ``"Diversity"``.
    best_y : float
        Best observed label seen so far (used by EI).
    size : int, optional
        Number of compounds to select. Default is 100.
    seed : int, optional
        RNG seed for the random strategy. Default is 42.
    unlabeled_gtm_coords : np.ndarray or None, optional
        GTM (x, y) coordinates for the unlabeled pool; required for Diversity.
        Default is None.
    labeled_gtm_coords : np.ndarray or None, optional
        GTM (x, y) coordinates for the current labeled set; required for Diversity.
        Default is None.

    Returns
    -------
    top_indices : np.ndarray
        Selected indices relative to the unlabeled pool.
    scores : np.ndarray or None
        Acquisition scores for each selected compound, or ``None`` for the
        random strategy.
    """
    n_unlabeled = len(smiles_unlabeled)

    # Special handling for non-model-based strategies
    if strategy == "Random":
        rng = np.random.RandomState(seed)
        selected_idx = rng.choice(
            n_unlabeled, size=min(size, n_unlabeled), replace=False
        )
        return selected_idx, None

    if strategy == "Diversity":
        # When no pool compounds have been labeled yet (e.g. n_start=0 with external
        # seed data), fall back to random so cdist does not receive an empty matrix.
        if labeled_gtm_coords is None or len(labeled_gtm_coords) == 0:
            rng = np.random.RandomState(seed)
            selected_idx = rng.choice(
                n_unlabeled, size=min(size, n_unlabeled), replace=False
            )
            return selected_idx, None
        # Max-min diversity: pick the compounds with the greatest minimum
        # Euclidean distance to any already-labeled compound in GTM space.
        dists = cdist(unlabeled_gtm_coords, labeled_gtm_coords, metric="euclidean")
        min_dists = dists.min(axis=1)  # nearest labeled neighbor for each candidate
        top_indices = np.argsort(min_dists)[::-1][:size]
        return top_indices, min_dists

    # Prepare unlabeled data for prediction
    X_featurized, _ = featurize(smiles_unlabeled)

    # Get acquisition scores from the committee
    qs_key = STRATEGY_QUERY_KEYS[strategy]

    scores = committee.query(
        X_featurized, query_strategy=qs_key, best_y=best_y, xi=0.01
    )

    # Flatten scores before ranking
    scores_1d = np.asarray(scores).ravel()

    # Select top compounds in descending order
    top_indices = np.argsort(scores_1d)[::-1][:size]

    return top_indices, scores_1d


def evaluate_on_test(committee, y_test, X_test_loader=None, smiles_test=None):
    """Evaluate a committee regressor on a held-out test set.

    Parameters
    ----------
    committee : CommitteeRegressor
        Trained committee model to evaluate.
    y_test : array-like
        Ground-truth target values for the test compounds.
    X_test_loader : DataLoader or None, optional
        Pre-computed featurized test loader. When provided ``smiles_test`` is
        ignored and featurization is skipped. Default is None.
    smiles_test : list[str] or pd.Series or None, optional
        SMILES strings for the test compounds. Used only when
        ``X_test_loader`` is None. Default is None.

    Returns
    -------
    results : dict
        Dictionary containing the following keys:

        - ``"mae"`` : float — mean absolute error.
        - ``"r2"`` : float — coefficient of determination.
        - ``"ktau"`` : float — Kendall's tau rank correlation.
        - ``"spearmanr"`` : float — Spearman rank correlation.
        - ``"miscal_area"`` : float — miscalibration area.
        - ``"y_test_pred"`` : np.ndarray — predicted mean values.
        - ``"y_test_std"`` : np.ndarray — predicted standard deviations.
    """
    if X_test_loader is None:
        X_test_loader, _ = featurize(smiles_test)
    y_test_arr = np.array(y_test).reshape(-1, 1)

    # Get predictions
    mean_pred, std_pred = committee.predict(X_test_loader, return_std=True)

    # Regression metrics
    reg_metrics = RegressionMetrics()
    reg_res = reg_metrics.evaluate(y_test_arr, mean_pred)
    task_metrics = reg_res["task_0"]

    # Uncertainty metrics
    unc_metrics = UncertaintyMetrics()
    unc_metrics.evaluate(y_test_arr, mean_pred, std_pred)
    unc_res = unc_metrics.report()["task_0"]

    # Result dictionary
    results = {
        "mae": task_metrics["mae"]["value"],
        "r2": task_metrics["r2"]["value"],
        "ktau": task_metrics["ktau"]["value"],
        "spearmanr": task_metrics["spearmanr"]["value"],
        "miscal_area": unc_res["miscal_area"],
        "y_test_pred": mean_pred.flatten(),
        "y_test_std": std_pred.flatten(),
    }
    return results


def run_active_learning(
    df_pool,
    df_test,
    n_start=100,
    k_iter=15,
    query_size=20,
    n_models=5,
    max_epochs=20,
    seed=42,
    strategy="Random",
    verbose=True,
    df_seed=None,
    gtm_coords=None,
):
    """Execute a full active learning loop for one strategy and seed.

    Starts with a random initial labeled subset (or empty if ``n_start=0``),
    then iteratively queries the unlabeled pool and re-trains the committee for
    ``k_iter`` rounds. Evaluates on the held-out test set after each iteration
    and records full state history.

    Parameters
    ----------
    df_pool : pd.DataFrame
        Candidate pool with columns ``"smiles"`` and ``"pEC50"``.
    df_test : pd.DataFrame
        Held-out test set with columns ``"smiles"`` and ``"pEC50"``.
    n_start : int, optional
        Number of randomly selected pool compounds in the initial labeled set.
        Set to ``0`` when ``df_seed`` is provided to let external data bootstrap
        the committee without consuming pool budget. Default is 100.
    k_iter : int, optional
        Number of active learning iterations after the initial training.
        Default is 15.
    query_size : int, optional
        Number of compounds to select from the pool at each iteration.
        Default is 20.
    n_models : int, optional
        Number of committee members to train at each iteration. Default is 5.
    max_epochs : int, optional
        Maximum training epochs per committee member. Default is 20.
    seed : int, optional
        Base random seed for both initial selection and subsequent queries.
        Default is 42.
    strategy : str, optional
        Acquisition strategy to use. One of ``"EI"``, ``"UCB"``, ``"Random"``,
        ``"Exploitation"``, ``"Exploration"``, or ``"Diversity"``.
        Default is ``"Random"``.
    verbose : bool, optional
        Whether to print progress after each iteration. Default is True.
    df_seed : pd.DataFrame or None, optional
        External seed training data with columns ``"smiles"`` and ``"pEC50"``
        (e.g. ChEMBL PXR data). When provided, these compounds are concatenated
        with the pool-acquired labels at every training step. They are never
        queried, never appear in the test set, and are **not** counted toward
        ``n_labeled`` in the history records. Default is None.
    gtm_coords : np.ndarray or None, optional
        Precomputed GTM 2D coordinates for ``df_pool`` (shape ``(n_pool, 2)``),
        as returned by ``smiles_to_gtm``. When provided the GTM fit is skipped,
        saving significant compute per job. Required for the ``"Diversity"``
        strategy; other strategies pass these coords through to ``query_batch``
        but do not use them. Default is None (GTM is computed on-the-fly).

    Returns
    -------
    dict
        Dictionary with the following keys:

        - ``"history"`` : list[dict] — per-iteration state records, each
          containing iteration index, labeled count, best observed value,
          pool activity values, selected indices, and test metrics.
        - ``"committee"`` : CommitteeRegressor — committee trained on the
          final labeled set.
    """
    # Initialize labeled pool with random subset
    rng = np.random.RandomState(seed)
    n_total = len(df_pool)
    all_indices = np.arange(n_total)

    # Initialize the labeled mask; n_start=0 is valid when df_seed bootstraps the model.
    labeled_mask = np.zeros(n_total, dtype=bool)
    if n_start > 0:
        initial_idx = rng.choice(all_indices, size=n_start, replace=False)
        labeled_mask[initial_idx] = True

    # Container for state history
    history = []

    # Start empty so iteration 0 correctly captures initial_idx as newly selected
    prev_labeled_mask = np.zeros(n_total, dtype=bool)

    # Use precomputed GTM coordinates when provided; otherwise fit on-the-fly.
    # smiles_to_gtm returns (gtm, crds_2d, resps, llhs) — unpack accordingly.
    if gtm_coords is None:
        _, gtm_coords, _, _ = smiles_to_gtm(df_pool["smiles"].tolist())

    # Pre-compute test loader once; reused at every iteration (test set never changes).
    X_test_loader, _ = featurize(df_test["smiles"].tolist())
    y_test_arr_full = df_test["pEC50"].values

    for k in range(k_iter + 1):
        labeled_idx = np.where(labeled_mask)[0]
        unlabeled_idx = np.where(~labeled_mask)[0]

        # Identify which indices are new this iteration for GTM tracking
        newly_labeled_idx = np.where(labeled_mask & ~prev_labeled_mask)[0]
        prev_labeled_mask = labeled_mask.copy()

        # Current labeled data
        df_labeled = df_pool.iloc[labeled_idx]

        # best_y: max over pool-acquired labels; seed data provides the reference
        # when the pool labeled set is still empty (n_start=0).
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

        # Build training set: pool-acquired labels + full external seed data
        if df_seed is not None and len(df_seed) > 0:
            df_train = pd.concat([df_labeled, df_seed], ignore_index=True)
        else:
            df_train = df_labeled

        # Hold out 10% of pool-acquired compounds for per-iteration calibration.
        # Seed data is always kept in the training set (it is never queried and
        # should be fully exploited). Fall back to sampling from df_train only
        # when the pool labeled set is too small to spare any compounds.
        _cal_min_pool = 10  # minimum pool-labeled compounds before we can hold out
        if len(df_labeled) >= _cal_min_pool:
            n_cal = max(1, int(0.1 * len(df_labeled)))
            df_cal_iter = df_labeled.sample(n=n_cal, random_state=seed + k)
            # Remove calibration compounds from the full training DataFrame
            df_train_fit = df_train[~df_train.index.isin(df_cal_iter.index)]
        else:
            # Not enough pool compounds yet — sample cal from the full training set
            # as a fallback (includes seed data).
            n_cal = max(1, int(0.1 * len(df_train)))
            df_cal_iter = df_train.sample(n=n_cal, random_state=seed + k)
            df_train_fit = df_train.drop(df_cal_iter.index)

        if verbose:
            n_seed_log = len(df_seed) if df_seed is not None else 0
            print(
                f"Iter {k}: {len(df_labeled)} pool-labeled + {n_seed_log} seed "
                f"({len(df_train_fit)} train, {len(df_cal_iter)} train_cal). "
                f"Best pEC50: {best_y:.2f}"
            )

        # Train committee on the 90% training subset; seed+k decorrelates bootstrap
        # draws across iterations within a run (avoids the same RNG start point each time).
        committee = train_committee(
            df_train_fit["smiles"],
            df_train_fit["pEC50"],
            n_models=n_models,
            max_epochs=max_epochs,
            seed=seed + k,
        )

        # Evaluate BEFORE calibration (stored for visualization)
        res_pre = evaluate_on_test(
            committee, y_test_arr_full, X_test_loader=X_test_loader
        )

        # Calibrate uncertainty on the held-out 10%
        X_cal_loader, _ = featurize(df_cal_iter["smiles"].tolist())
        y_cal_arr = df_cal_iter["pEC50"].values.reshape(-1, 1)
        committee.calibrate_uncertainty(
            X_cal_loader, y_cal_arr, method="scaling-factor"
        )

        # Evaluate AFTER calibration
        res = evaluate_on_test(committee, y_test_arr_full, X_test_loader=X_test_loader)

        # Record state
        state = {
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

        # Query step (if not last iteration)
        if k < k_iter:
            df_unlabeled = df_pool.iloc[unlabeled_idx]

            # Get indices relative to df_unlabeled
            rel_idx, scores = query_batch(
                committee,
                df_unlabeled["smiles"].tolist(),
                strategy,
                best_y,
                size=query_size,
                seed=seed + k,
                unlabeled_gtm_coords=gtm_coords[unlabeled_idx],
                labeled_gtm_coords=gtm_coords[labeled_idx],
            )

            # Map back to original pool indices
            abs_idx = unlabeled_idx[rel_idx]
            labeled_mask[abs_idx] = True

            if scores is not None:
                history[-1]["acquisition_scores"] = scores

    return {"history": history, "committee": committee}  # return last committee

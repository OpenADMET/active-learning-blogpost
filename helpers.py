import numpy as np
import tmap as tm
import ugtm
from lightning import pytorch as pl
from mhfp.encoder import MHFPEncoder
from openadmet.models.active_learning.committee import CommitteeRegressor
from openadmet.models.architecture.chemprop import ChemPropModel
from openadmet.models.eval.regression import RegressionMetrics
from openadmet.models.eval.uncertainty import UncertaintyMetrics
from openadmet.models.features.chemprop import ChemPropFeaturizer
from openadmet.models.split.scaffold import ScaffoldSplitter
from openadmet.models.trainer.lightning import LightningTrainer
from rdkit import Chem
from scipy.spatial.distance import cdist

from conf import STRATEGY_QUERY_KEYS


def smiles_to_ecfp4(smiles_list, radius=2, n_bits=1024):
    """Convert a list of SMILES to a binary ECFP4 fingerprint matrix."""
    fps = []

    # Initialize the generator (radius 2 is equivalent to ECFP4)
    generator = Chem.rdFingerprintGenerator.GetMorganGenerator(
        radius=radius, fpSize=n_bits
    )

    # Iterate over SMILES
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)

        # Successfully parsed molecule → compute fingerprint
        if mol is not None:
            fp = generator.GetFingerprint(mol)
            fps.append(list(fp))

        # Fall back to an all-zero vector so the array stays rectangular
        else:
            fps.append([0] * n_bits)

    return np.array(fps, dtype=np.float32)


def smiles_to_gtm(
    smiles_list, radius=2, n_bits=1024, k=16, m=4, s=0.3, regul=0.1, niter=200
):
    """Convert a list of SMILES to GTM coordinates."""
    # Compute ECFP4 fingerprints
    X_fp = smiles_to_ecfp4(smiles_list, radius=radius, n_bits=n_bits)

    # Fit GTM on the fingerprints
    gtm_model = ugtm.runGTM(
        X_fp,
        k=k,
        m=m,
        s=s,
        regul=regul,
        niter=niter,
        verbose=False,
    )

    # Return the 2D posterior mean coordinates
    return gtm_model


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


def split_data(
    X,
    y,
    train_size=0.8,
    val_size=0.1,
    test_size=0.1,
    random_state=42,
):
    """Split data into train/val/test sets using scaffold splitting."""
    splitter = ScaffoldSplitter(
        train_size=train_size,
        val_size=val_size,
        test_size=test_size,
        random_state=random_state,
    )

    # Split the data
    # The splitter returns (X_train, X_val, X_test, y_train, y_val, y_test, groups)
    X_pool, X_cal, X_test, y_pool, y_cal, y_test, _ = splitter.split(X, y)
    return X_pool, X_cal, X_test, y_pool, y_cal, y_test


def featurize(smiles_list, y_list=None, shuffle=False):
    """Featurize SMILES for ChemProp."""
    featurizer = ChemPropFeaturizer(batch_size=64, shuffle=shuffle)
    loader, indices, scaler, dataset = featurizer.featurize(smiles_list, y_list)
    return loader, scaler


def build_committee_member(seed=42, max_epochs=20, log_dir=False):
    """Constructs a single ChemPropModel member with a LightningTrainer."""

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

    # Define the trainer
    trainer = LightningTrainer(
        output_dir=log_dir,
        max_epochs=max_epochs,
        accelerator="gpu",  # Use "gpu" if available
        devices=1,
        early_stopping=False,
        gradient_clip_val=0.5,
    )

    # Link them
    trainer.model = model
    return model, trainer


def train_committee(smiles_labeled, y_labeled, n_models=5, seed=42, max_epochs=20):
    """Train N_MODELS committee members on bootstrapped data."""
    members = []
    rng = np.random.RandomState(seed)

    for i in range(n_models):
        # Build and seed member first so pl.seed_everything controls dataloader shuffle
        model, trainer = build_committee_member(
            seed=seed + i, max_epochs=max_epochs, log_dir=None
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
    Select M_QUERY molecules from the unlabeled pool.

    Parameters
    ----------
    committee : CommitteeRegressor
        Trained committee used for uncertainty-based strategies.
    smiles_unlabeled : list[str]
        SMILES of unlabeled candidates.
    strategy : str
        Acquisition strategy name.
    best_y : float
        Best observed label seen so far (used by EI/PI).
    seed : int
        RNG seed for the random strategy.
    unlabeled_gtm_coords : np.ndarray or None
        GTM (x, y) coordinates for the unlabeled pool; required for Diversity.
    labeled_gtm_coords : np.ndarray or None
        GTM (x, y) coordinates for the current labeled set; required for Diversity.

    Returns
    -------
    tuple[np.ndarray, np.ndarray | None]
        Selected indices (relative to unlabeled pool) and acquisition scores (or None).

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
        # Max-min diversity: pick the M compounds with the greatest minimum
        # Euclidean distance to any already-labeled compound in GTM space.
        # This maximises structural dissimilarity between successive batches.
        dists = cdist(unlabeled_gtm_coords, labeled_gtm_coords, metric="euclidean")
        min_dists = dists.min(axis=1)  # nearest labeled neighbor for each candidate
        top_indices = np.argsort(min_dists)[::-1][:size]
        return top_indices, min_dists

    # Prepare unlabeled data for prediction
    X_featurized, _ = featurize(smiles_unlabeled)

    # Get acquisition scores
    # Note: query() returns scores, we must argsort to get indices
    qs_key = STRATEGY_QUERY_KEYS[strategy]

    # Fix kwargs for EI/PI
    scores = committee.query(
        X_featurized, query_strategy=qs_key, best_y=best_y, xi=0.01
    )

    # Flatten scores before ranking
    scores_1d = np.asarray(scores).ravel()

    # Select top M (descending order)
    top_indices = np.argsort(scores_1d)[::-1][:size]

    return top_indices, scores_1d


def evaluate_on_test(committee, smiles_test, y_test):
    """Evaluate committee on held-out test set."""
    X_featurized, _ = featurize(smiles_test)
    y_test_arr = np.array(y_test).reshape(-1, 1)

    # Get predictions
    mean_pred, std_pred = committee.predict(X_featurized, return_std=True)

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
    df_pool, df_test, n_start=100, k_iter=15, seed=42, strategy="Random", verbose=True
):
    """Execute full AL loop for one strategy and seed."""
    # Initialize labeled pool with random subset
    rng = np.random.RandomState(seed)
    n_total = len(df_pool)
    all_indices = np.arange(n_total)

    # Start with N_START labeled indices
    initial_idx = rng.choice(all_indices, size=n_start, replace=False)
    labeled_mask = np.zeros(n_total, dtype=bool)
    labeled_mask[initial_idx] = True

    # Container for state history
    history = []

    # Start empty so iteration 0 correctly captures initial_idx as newly selected
    prev_labeled_mask = np.zeros(n_total, dtype=bool)

    # Precompute GTM coordinates for diversity-based strategy
    gtm_model = smiles_to_gtm(df_pool["smiles"].tolist())
    gtm_coords = gtm_model.matMeans

    for k in range(k_iter + 1):
        labeled_idx = np.where(labeled_mask)[0]
        unlabeled_idx = np.where(~labeled_mask)[0]

        # Identify which indices are new this iteration for GTM tracking
        newly_labeled_idx = np.where(labeled_mask & ~prev_labeled_mask)[0]
        prev_labeled_mask = labeled_mask.copy()

        # Current labeled data
        df_labeled = df_pool.iloc[labeled_idx]
        best_y = df_labeled["pEC50"].max()

        if verbose:
            print(f"Iter {k}: {len(df_labeled)} labeled. Best pEC50: {best_y:.2f}")

        # Train committee
        committee = train_committee(
            df_labeled["smiles"],
            df_labeled["pEC50"],
            seed,
            bootstrap_seed=seed + k,
            al_iter=k,
            run_name=strategy,
        )

        # Evaluate
        res = evaluate_on_test(committee, df_test["smiles"], df_test["pEC50"])

        # Record state
        state = {
            "iteration": k,
            "n_labeled": len(df_labeled),
            "best_y": best_y,
            "pool_y_values": df_labeled["pEC50"].values.tolist(),
            "selected_pool_indices": newly_labeled_idx.tolist(),
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
                seed + k,
                unlabeled_gtm_coords=gtm_coords[unlabeled_idx],
                labeled_gtm_coords=gtm_coords[labeled_idx],
            )

            # Map back to original pool indices
            abs_idx = unlabeled_idx[rel_idx]
            labeled_mask[abs_idx] = True

            if scores is not None:
                history[-1]["acquisition_scores"] = scores

    return {"history": history, "committee": committee}  # return last committee

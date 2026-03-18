# Teaching models to ask: active learning for pEC50 prediction

In drug discovery, the most valuable resource isn't compute, it's data. Synthesizing and assaying a single compound can cost thousands of dollars and take weeks. Yet, most machine learning models are trained as if labels are free, consuming massive random splits of [ChEMBL](https://www.ebi.ac.uk/chembl/) or [Enamine Real](https://enamine.net/compound-collections/real-compounds/real-database).

[**Active learning (AL)**](https://en.wikipedia.org/wiki/Active_learning_(machine_learning)) flips this paradigm. Instead of passively accepting a training set, the model iteratively selects the compounds it finds most confusing (to "explore") or promising (to "exploit"). In this post, we demonstrate how to build an active learning loop using [`openadmet-models`](https://github.com/OpenADMET/openadmet-models), powered by the [**CheMeleon**](https://github.com/JacksonBurns/chemeleon) foundation model. We'll simulate a campaign to find compounds active against the [Pregnane X Receptor (PXR)](https://en.wikipedia.org/wiki/Pregnane_X_receptor), starting with just 100 labeled molecules. By leveraging uncertainty quantification and smart acquisition strategies like expected improvement (EI) and upper confidence bound (UCB), we show that we can reach high predictive accuracy with a fraction of the data required by random screening.

## Why PXR?

PXR is a ligand-activated nuclear transcription factor that functions as the body's primary xenobiotic sensor, highly expressed in the liver and intestines. Structurally, PXR is defined by a uniquely large, flexible, and hydrophobic ligand-binding pocket, which makes it notoriously promiscuous and capable of accommodating a massive variety of chemical scaffolds. When a molecule binds to PXR, it triggers the upregulation of crucial phase I and phase II drug-metabolizing enzymes (most significantly [CYP3A4](https://en.wikipedia.org/wiki/CYP3A4)) as well as phase III efflux transporters. This systemic enzyme induction aggressively accelerates the metabolic clearance of both the offending compound and any co-administered therapies, leading to severe [drug-drug interactions (DDIs)](https://en.wikipedia.org/wiki/Drug_interaction) and sub-therapeutic drug plasma levels. Because of this downstream cascade, PXR is treated as a high-priority absorption, metabolism, excretion, and toxicity ([ADMET](https://en.wikipedia.org/wiki/ADME)) antitarget. We model its binding affinity to computationally flag and optimize away this liability early in the drug design process, preventing compounds that trigger auto-induction or DDIs from advancing to costly clinical trials.

## The label bottleneck in drug discovery

A typical high-throughput screening (HTS) campaign might screen 100,000 compounds, but for lead optimization, we often work with much smaller datasets (hundreds to low thousands). When training predictive models on such small data, the choice of training points matters immensely. A model trained on 100 diverse, informative compounds will outperform one trained on 100 redundant analogues.

Active learning formalizes this intuition. We start with a small "seed" dataset, train an ensemble of models, and then use their consensus (mean) and disagreement (standard deviation) to query a large, unlabeled pool. This is often called [**query-by-committee (QBC)**](https://dl.acm.org/doi/10.1145/130385.130417).

Here, we combine QBC with [**CheMeleon**](https://github.com/JacksonBurns/chemeleon), a graph neural network pretrained on millions of molecules. CheMeleon provides a robust chemical representation even when task-specific labels are scarce, making it an ideal backbone for low-data active learning.

To reproduce the results in this post, install `openadmet-models` by following the [installation instructions](https://docs.openadmet.org/en/latest/installation.html), then run:

```bash
python run.py       # execute the active learning pipeline (~several GPU-hours)
python analysis.py  # generate all figures from results/all_runs.pkl
```

All supporting code lives in `src/`: [src/helpers.py](src/helpers.py) contains the core AL utilities and [src/plots.py](src/plots.py) contains all Plotly/Faerun plotting functions.

## Configuration

The campaign is governed by five parameters, set at the top of `run.py`:

```python
N_START  = 100   # initial labeled pool size
M_QUERY  = 20    # molecules queried per AL iteration
K_ITER   = 15    # number of AL iterations
N_MODELS = 5     # committee size (ensemble members)
SEED     = 42    # global random seed
```

Six acquisition strategies are compared: **EI**, **UCB**, **Random**, **Exploitation**, **Exploration**, and **Diversity**.

## A note on compute

Running a full active learning benchmark is computationally intensive because we retrain the model from scratch at every iteration to simulate a real campaign, repeated for each selection strategy to compare them.

With $k=15$ iterations and 6 strategies, we are performing $15 \times 6 = 90$ full training runs. Each run trains a committee of 5 models. On a standard GPU (e.g. T4 or A10), `run.py` might take several hours to complete. If you want a quick preview, reduce `K_ITER` to `5` (with higher `M_QUERY`) and `N_MODELS` to `3`.

**Note:** The [**CheMeleon**](https://github.com/JacksonBurns/chemeleon) weights (~34MB) will be downloaded from [Zenodo](https://zenodo.org/records/15460715) automatically upon the first run.

## Load in the reference dataset

We won't be running an actual active learning campaign here for obvious reasons, but we can simulate one using a set of real, experimental data. We pretend we don't know the labels and, at each iteration, use what we learned from each synthetic AL cycle to drive next iteration selections. Rinse and repeat.

The dataset is loaded from a Parquet file and an additional background screening library (`_data/octant_screening_compounds.csv`) is read to provide a chemical-space backdrop for the TMAP / GTM visualizations. Background compounds that overlap with the labeled dataset are excluded.

## Splitting the data: scaffold-based held-out test set

Random splitting is dangerously optimistic in drug discovery because it allows structurally similar molecules (analogues) to appear in both training and test sets. A model can "cheat" by memorizing the series rather than learning the [structure-activity relationship (SAR)](https://en.wikipedia.org/wiki/Structure%E2%80%93activity_relationship).

We use a **scaffold split** to separate the data based on [Bemis-Murcko scaffolds](https://practicalcheminformatics.blogspot.com/2021/10/exploratory-data-analysis-with.html). This forces the model to generalize to new chemical series, mimicking a prospective lead optimization scenario. This test set is fixed and held out for the entire active learning loop, ensuring a fair, apples-to-apples comparison across all iterations.

```python
splitter = ScaffoldSplitter(train_size=0.8, val_size=0.1, test_size=0.1, random_state=42)
X_pool, X_cal, X_test, y_pool, y_cal, y_test, _ = splitter.split(
    df["OPENADMET_CANONICAL_SMILES"], df["PXR_pEC50"]
)
```

The 80% training split becomes the **candidate pool** from which the active learner selects new labels, the 10% validation split serves as a post-hoc **calibration set**, and the 10% test split is the fixed evaluation benchmark held out for the entire campaign.

## GTM chemical space embedding

The **Diversity** acquisition strategy selects compounds that are maximally dissimilar from the current labeled set in chemical space. To measure structural distance we use a **Generative Topographic Map (GTM)** — a probabilistic manifold projection that maps high-dimensional ECFP4 fingerprints onto an interpretable 2D grid. Each compound gets a single (x, y) coordinate, and pairwise Euclidean distances in that 2D space serve as a fast, interpretable proxy for molecular dissimilarity.

`run.py` fits the GTM once on the full unlabeled pool **plus** the background screening library before the active learning loop starts, so the embedding is fixed and consistent across all strategies. The resulting coordinates are stored in `results/all_runs.pkl` and reused by `analysis.py` for visualization without recomputation.

## Query-by-committee: how ensemble disagreement guides exploration

Our committee consists of $N=5$ independent [**CheMeleon**](https://github.com/JacksonBurns/chemeleon) models. When we present an unlabeled molecule to the committee, we get 5 predictions.

- The **mean** prediction ($\mu$) is our best guess for activity.
- The **standard deviation** ($\sigma$) represents epistemic uncertainty.

Different acquisition strategies leverage these metrics:
- **Exploitation**: Greedy selection of the highest $\mu$. Finds good compounds fast but can get stuck in local optima.
- **Upper Confidence Bound (UCB)**: $\mu + \beta\sigma$. Optimistically explores regions that *might* be high activity.
- **Expected Improvement (EI)**: Balances $\mu$ and $\sigma$ to calculate the probability of exceeding the current best label $f^*$ (equation below).
- **Exploration**: Pure uncertainty sampling — selects the $m$ compounds with the highest $\sigma$, ignoring predicted activity entirely. Useful as a model-agnostic baseline that maximises coverage of the epistemic uncertainty landscape.
- **Diversity**: Ignores model predictions altogether and instead selects the $m$ compounds that are farthest from the current labeled set in 2D GTM chemical space (max-min Euclidean distance). This enforces structural dissimilarity between batches and prevents the committee from over-sampling a single region of chemical space.

$$EI(\mathbf{x}) = (\mu(\mathbf{x}) - f^* - \xi)\,\Phi(Z) + \sigma(\mathbf{x})\,\phi(Z)$$

where $Z = \dfrac{\mu(\mathbf{x}) - f^* - \xi}{\sigma(\mathbf{x})}$.

To achieve ensemble diversity, we vary the initialization of each ensemble member ([deep ensembling](https://dl.acm.org/doi/10.5555/3295222.3295387)) and [bootstrap](https://en.wikipedia.org/wiki/Bootstrap_aggregating) the labeled set for each member (sampling with replacement). It is precisely this diversity, captured as disagreement between members at prediction time, that gives our committee meaningful epistemic uncertainty to feed into the acquisition function.

We deliberately *aren't* using a held out set for calibration, despite stressing its importance in another [blog post](https://openadmet.ghost.io/concerning-uncertainty/). **Exploration**, **EI**, and **UCB** use σ to rank unlabeled molecules, i.e. the molecule with the highest acquisition score gets queried next. A miscalibrated σ (e.g., overconfident by a constant factor of 2×) will compress all uncertainty estimates equally, but the relative ordering among candidates is preserved. The committee will still correctly identify that compound A is more uncertain than compound B, even if both σ values are off in absolute terms. Scaling factor or [isotonic regression](https://en.wikipedia.org/wiki/Isotonic_regression) fix the scale so that "90% interval" actually covers 90% of outcomes — but it does not reorder the candidates. We thus wait to calibrate after all active learning cycles have been completed, to yield our final model with meaningful uncertainties.

Practically speaking, you also need a held-out calibration set to perform calibration. At every iteration, you'd have to carve off some labeled molecules as a calibration subset rather than using them to train the committee. Early in the loop — say iteration 2 with 140 labeled molecules total — a calibration set of even 30 compounds makes calibration unreliable and weakens the committee.

## The active learning loop

The core logic is encapsulated in [src/helpers.py](src/helpers.py). `featurize` prepares the data, `train_committee` fits a bootstrapped ensemble on the current labeled set, `query_batch` uses the committee's μ and σ (or GTM distances, for **Diversity**) to select the next $m$ molecules, and `evaluate_on_test` scores the updated model on the fixed test set, all bookended by `build_committee_member`, which ensures each ensemble member is constructed consistently. These four steps repeat $k$ times, with each new batch of oracle-labeled molecules feeding back into the next training call. The six strategies are all handled by `query_batch` via a single `strategy` argument.

In each iteration $k$:
1. Train the committee on the current `labeled_pool`.
2. Evaluate on the static `test_set` to track performance.
3. Use the committee to predict on the `unlabeled_pool`.
4. Score the unlabeled molecules with the acquisition function.
5. "Acquire" the top $m$ molecules (reveal their labels).
6. Add them to the `labeled_pool` and repeat.

We track not just accuracy (MAE), but also the *quality* of the molecules we found (their pEC50 values). A good AL strategy should find the potent hits early.

## Running the experiments

Run `python run.py` to execute the loop for all six strategies with a fixed seed. Results are checkpointed to `results/all_runs.pkl` after each strategy, so the run can be safely interrupted and resumed — strategies already present in the checkpoint are skipped automatically.

The checkpoint stores the full per-iteration history for each strategy (test-set metrics, labeled pool snapshot, selected pool indices) together with the pre-fitted GTM coordinates, the scaffold-split DataFrames (`df_pool`, `df_cal`, `df_test`), and the campaign configuration. This self-contained payload means `analysis.py` needs no access to the original dataset.

## Unpacking results

`analysis.py` unpacks the nested `all_runs` dictionary into two tidy long-format DataFrames before plotting:

- **`learning_curve_long`** — one row per (strategy, iteration), carrying the test-set metrics (MAE, R², Kendall's τ, Spearman ρ) and the labeled-pool size `n_labeled`.
- **`pool_history_long`** — one row per (strategy, iteration, compound), recording the pEC50 of every molecule in the labeled pool at each iteration. This lets us track how the distribution of acquired labels shifts over the course of the campaign — a window into *what* each strategy is choosing to label, not just *how well* the model performs.

Because this experiment uses a **single random seed**, there is no cross-seed variance to collapse. `learning_curve_summary` is constructed by renaming the raw metric columns to the `_mean` suffix expected by the plotting functions, then duplicating those values as `_lower` and `_upper` confidence bounds — effectively zero-width bands. This keeps all downstream plotting calls compatible with the multi-seed band-plot API without requiring any changes to the plot helpers.

## Results

Run `python analysis.py` to regenerate all figures. They are written to `results/` as self-contained interactive HTML files.

### Learning curves

*[View interactive figure: Learning curves — MAE](results/learning_curve_mae.html)*

*[View interactive figure: Learning curves — Kendall's τ](results/learning_curve_ktau.html)*

As expected, **EI** and **UCB** outperform **Random** sampling significantly in the early iterations (100–250 labeled molecules). The gap narrows as the pool grows, but the "area under the learning curve" advantage for active learning is substantial.

**Exploitation** initially improves fast but often plateaus or degrades slightly because it over-samples a single chemical series (the "greedy" trap), failing to explore the diversity of the scaffold split.

**Exploration** behaves as a pure uncertainty oracle: it queries regions where the model is most confused, which drives rapid improvement early on but eventually saturates once the model has adequate coverage. Without any signal from predicted activity it can waste queries on low-activity regions.

**Diversity** enforces structural dissimilarity at every step, spreading labels uniformly across chemical space. This prevents the committee from over-committing to a single scaffold and tends to produce well-calibrated models, but it can be slow to find potent hits because it ignores predicted activity entirely.

Crucially, even at $N=100$, the MAE is reasonable (~0.7–0.8), thanks to CheMeleon's pretraining. A randomly initialized model would likely be much worse.

### Hit discovery

*[View interactive figure: Hit discovery curve](results/hit_discovery_curve.html)*

## Navigating chemical space with GTM

Learning curves tell us *how fast* a strategy converges, but they say nothing about *where* in chemical space each strategy chooses to look. A strategy that exploits a single potent scaffold may show an early MAE drop while leaving most of the chemical diversity untouched — a dangerous blind spot in a real campaign.

We reuse the GTM embedding (fit before the active learning loop) for visualization, replaying the **Exploitation** selection history iteration by iteration:

* **Gray** — the full compound pool (all available unlabeled candidates)
* **Blue** — compounds acquired in any *prior* iteration (accumulated history)
* **Red** — compounds selected in the *current* iteration

Use the slider to step through iterations manually, or press **▶ Play** to watch the campaign unfold.

*[View interactive figure: GTM selection animation (Exploitation)](results/gtm_selection_animation_exploitation.html)*

*[View interactive figure: TMAP chemical space (EI)](results/tmap_selection.html)*

## Are our uncertainties trustworthy?

A model with good MAE can still be overconfident. In active learning, this is dangerous: if the model is confident but wrong about an unlabeled region, it may never query it (for **EI** / **UCB**). We tackled the topic of model uncertainty in another OpenADMET blogpost, [Concerning Uncertainty](https://openadmet.ghost.io/concerning-uncertainty/).

We evaluate calibration using the **miscalibration area**. A perfectly calibrated model has e.g. 90% of data points falling within its 90% confidence interval. We can improve this post-hoc using **scaling factor** calibration on a small holdout set (`df_cal`). `analysis.py` takes the final EI committee, calibrates it on `df_cal`, and re-evaluates on `df_test`.

*[View interactive figure: Calibration curve (before / after)](results/calibration_curve.html)*

The plot shows how scaling-factor calibration pulls the calibration curve closer to the diagonal. For early-stage AL, this ensures that the acquisition function (which relies on $\sigma$) is making decisions based on realistic uncertainty estimates, preventing the model from ignoring "unknown unknowns."

## Takeaways

1. **Efficiency**: **EI** and **UCB** consistently reach performance milestones with fewer labels than random sampling. In this simulation, EI often reaches the target MAE with 20–30% fewer compounds.
2. **Exploitation is risky**: Pure greedy strategies converge fast but often stagnate. They find local maxima but fail to learn the global SAR.
3. **Exploration covers unknowns**: Sampling purely by uncertainty ($\sigma$) maps the epistemic landscape quickly but wastes queries on inactive regions once the model matures. It serves as a useful diagnostic — if **Exploration** outperforms **EI**, the committee is under-exploring and would benefit from a higher $\beta$ or $\xi$.
4. **Diversity ensures coverage**: GTM-based max-min selection prevents scaffold collapse and produces the most structurally diverse labeled set. It is the safest strategy when potency information is completely absent, but it sacrifices hit-finding speed.
5. **Foundation Models matter**: CheMeleon provides a strong prior. Even with $N=100$, the models are performant. Active learning is most effective when the base model is already capable of reasoning about chemical structure.
6. **Calibration is (almost) free**: Post-hoc uncertainty calibration significantly improves reliability without retraining, which is crucial for safety-critical decision making, but does require a held-out calibration set.
7. **Recommendation**: For early-stage screening, use **EI**. It naturally balances finding hits (for the project team) and learning the SAR (for the model). Once you have >300 labels and the model is stable, you might switch to pure exploitation or standard screening.

## Limitations and next steps

- **Time Split**: A scaffold split is a good proxy, but a true temporal split (training on data < 2020, testing on > 2020) is the gold standard.
- **Batch Awareness**: We selected the top $m$ compounds independently. Advanced "batch active learning" ensures diversity *within* the query batch to avoid selecting identical analogues. The **Diversity** strategy already addresses this implicitly by maximising structural dissimilarity from the labeled set.

Check out the [`openadmet-models`](https://github.com/OpenADMET/openadmet-models) repository for the full code and more advanced featurizers.

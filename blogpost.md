# Teaching models to ask: active learning for pEC50 prediction

In drug discovery, the most valuable resource isn't compute, it's data. Synthesizing and assaying a single compound can cost thousands of dollars and take weeks. Yet, most machine learning models are trained as if labels are free, consuming massive random splits of [ChEMBL](https://www.ebi.ac.uk/chembl/) or [Enamine Real](https://enamine.net/compound-collections/real-compounds/real-database).

[**Active learning (AL)**](https://en.wikipedia.org/wiki/Active_learning_(machine_learning)) flips this paradigm. Instead of passively accepting a training set, the model iteratively selects the compounds it finds most confusing (to "explore") or promising (to "exploit"). In this post, we demonstrate how to build an active learning loop using [`openadmet-models`](https://github.com/OpenADMET/openadmet-models), powered by the [**CheMeleon**](https://github.com/JacksonBurns/chemeleon) foundation model. We'll simulate a campaign to find compounds active against the [Pregnane X Receptor (PXR)](https://en.wikipedia.org/wiki/Pregnane_X_receptor), bootstrapped from ~600 ChEMBL-derived labels and selecting 100 new compounds per iteration for 20 iterations. By leveraging uncertainty quantification and smart acquisition strategies like expected improvement (EI) and upper confidence bound (UCB), we show that we can reach high predictive accuracy with a fraction of the data required by random screening.

## Why PXR?

PXR is a ligand-activated nuclear transcription factor that functions as the body's primary xenobiotic sensor, highly expressed in the liver and intestines. Structurally, PXR is defined by a uniquely large, flexible, and hydrophobic ligand-binding pocket, which makes it notoriously promiscuous and capable of accommodating a massive variety of chemical scaffolds. When a molecule binds to PXR, it triggers the upregulation of crucial phase I and phase II drug-metabolizing enzymes (most significantly [CYP3A4](https://en.wikipedia.org/wiki/CYP3A4)) as well as phase III efflux transporters. This systemic enzyme induction aggressively accelerates the metabolic clearance of both the offending compound and any co-administered therapies, leading to severe [drug-drug interactions (DDIs)](https://en.wikipedia.org/wiki/Drug_interaction) and sub-therapeutic drug plasma levels. Because of this downstream cascade, PXR is treated as a high-priority absorption, metabolism, excretion, and toxicity ([ADMET](https://en.wikipedia.org/wiki/ADME)) antitarget. We model its binding affinity to computationally flag and optimize away this liability early in the drug design process, preventing compounds that trigger auto-induction or DDIs from advancing to costly clinical trials.

## The label bottleneck in drug discovery

A typical high-throughput screening (HTS) campaign might screen 100,000 compounds, but for lead optimization, we often work with much smaller datasets (hundreds to low thousands). When training predictive models on such small data, the choice of training points matters immensely. A model trained on 100 diverse, informative compounds will outperform one trained on 100 redundant analogues.

Active learning formalizes this intuition. We start with an external "seed" dataset of ~600 ChEMBL-derived PXR measurements to give the committee a meaningful prior, then use their consensus (mean) and disagreement (standard deviation) to query a large, unlabeled pool. This is often called [**query-by-committee (QBC)**](https://dl.acm.org/doi/10.1145/130385.130417).

Here, we combine QBC with [**CheMeleon**](https://github.com/JacksonBurns/chemeleon), a graph neural network pretrained on millions of molecules. CheMeleon provides a robust chemical representation even when task-specific labels are scarce, making it an ideal backbone for low-data active learning.

To reproduce the results in this post, install `openadmet-models` by following the [installation instructions](https://docs.openadmet.org/en/latest/installation.html), then run:

```bash
python run.py       # execute the active learning pipeline (~several GPU-hours)
python analysis.py  # generate all figures from results/all_runs.pkl
```

All supporting code lives in `src/`: [src/helpers.py](src/helpers.py) contains the core AL utilities and [src/plots.py](src/plots.py) contains all Plotly/Faerun plotting functions.

## Configuration

The campaign is governed by a handful of parameters in `config.yaml`:

```yaml
active_learning:
  k_iter: 20       # number of AL iterations per run
  query_size: 100  # compounds selected from the pool per iteration
  n_start: 0       # no initial pool seeding — seed data is used instead
  seeds: [42, 43, 44, 45, 46]

training:
  n_models: 5      # committee size (ensemble members)
```

The `n_start: 0` setting means no compounds are drawn from the unlabeled pool at the start. Instead, the committee is initialized on an external ChEMBL pretraining set of ~600 PXR-activity measurements (`seed_data_path` in `config.yaml`), which is always included in training but never queried or evaluated. This reflects a realistic scenario where historical data is available from the literature or a public database but the prospective assay set has not yet been touched.

The `query_size` of 100 compounds is grounded in experimental throughput: a standard 1536-well microplate can accommodate roughly 100 compounds when each requires a ~13-point dose-response curve to derive a pEC50 value — one full plate worth of data per AL iteration.

Six acquisition strategies are compared: **EI**, **UCB**, **Random**, **Exploitation**, **Exploration**, and **Diversity**.

## A note on compute

Running a full active learning benchmark is computationally intensive because we retrain the model from scratch at every iteration to simulate a real campaign, repeated for each selection strategy to compare them.

With $k=20$ iterations and 6 strategies, we are performing $20 \times 6 = 120$ full training runs. Each run trains a committee of 5 models. On a standard GPU (e.g. T4 or A10), `run.py` might take several hours to complete. If you want a quick preview, reduce `k_iter` to `5` and `n_models` to `3` in `config.yaml`.

**Note:** The [**CheMeleon**](https://github.com/JacksonBurns/chemeleon) weights (~34MB) will be downloaded from [Zenodo](https://zenodo.org/records/15460715) automatically upon the first run.

## Load in the reference dataset

We won't be running an actual active learning campaign here for obvious reasons, but we can simulate one using a set of real, experimental data. We pretend we don't know the labels and, at each iteration, use what we learned from each synthetic AL cycle to drive next iteration selections. Rinse and repeat.

The dataset is loaded from a Parquet file and an additional background screening library (`_data/octant_screening_compounds.csv`) is read to provide a chemical-space backdrop for the TMAP / GTM visualizations. Background compounds that overlap with the labeled dataset are excluded.

## Splitting the data: scaffold-based held-out test set

Random splitting is dangerously optimistic in drug discovery because it allows structurally similar molecules (analogues) to appear in both training and test sets. A model can "cheat" by memorizing the series rather than learning the [structure-activity relationship (SAR)](https://en.wikipedia.org/wiki/Structure%E2%80%93activity_relationship).

We use a **scaffold split** to separate the data based on [Bemis-Murcko scaffolds](https://practicalcheminformatics.blogspot.com/2021/10/exploratory-data-analysis-with.html). This forces the model to generalize to new chemical series, mimicking a prospective lead optimization scenario. This test set is fixed and held out for the entire active learning loop, ensuring a fair, apples-to-apples comparison across all iterations.

```python
splitter = ScaffoldSplitter(train_size=0.8, val_size=0.0, test_size=0.2, random_state=42)
X_pool, _, X_test, y_pool, _, y_test, _ = splitter.split(
    df["OPENADMET_CANONICAL_SMILES"], df["PXR_pEC50"]
)
df_pool = pd.DataFrame({"smiles": X_pool, "pEC50": y_pool})
df_test = pd.DataFrame({"smiles": X_test,  "pEC50": y_test})
```

The 80% training split becomes the **candidate pool** from which the active learner selects new labels, and the 20% test split is the fixed evaluation benchmark held out for the entire campaign.

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

At every iteration, we hold out 10% of the pool-acquired labels as a calibration set (`train_cal`) and use it to fit a **scaling factor** calibrator on the committee's uncertainty estimates. This is a lightweight, single-parameter correction that rescales σ so that coverage intervals better match observed error rates. For the *acquisition function*, this makes no difference: **Exploration**, **EI**, and **UCB** rank unlabeled molecules by σ, and a global scale factor preserves the relative ordering. The committee will still correctly identify that compound A is more uncertain than compound B regardless of whether both σ values are rescaled by the same constant. The calibration matters for the *final model* — so that a "90% interval" actually covers 90% of outcomes — but it does not influence which compounds get queried.

Note that we do not use the held-out `df_test` for calibration. Calibrating on the same scaffold-split test set used to evaluate performance would be circular, and in a real campaign no such external holdout exists. As we discuss in the [calibration section](#are-our-uncertainties-trustworthy), fitting the calibrator on pool-acquired compounds and evaluating it on unseen scaffolds introduces a distribution shift that limits how much the correction can actually help.

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

The checkpoint stores the full per-iteration history for each strategy (test-set metrics, labeled pool snapshot, selected pool indices) together with the pre-fitted GTM coordinates, the scaffold-split DataFrames (`df_pool`, `df_test`), the optional external seed data (`df_seed`), and the campaign configuration. This self-contained payload means `analysis.py` needs no access to the original dataset.

## Unpacking results

`analysis.py` unpacks the nested `all_runs` dictionary into two tidy long-format DataFrames before plotting:

- **`learning_curve_long`** — one row per (strategy, iteration), carrying the test-set metrics (MAE, R², Kendall's τ, Spearman ρ) and the labeled-pool size `n_labeled`.
- **`pool_history_long`** — one row per (strategy, iteration, compound), recording the pEC50 of every molecule in the labeled pool at each iteration. This lets us track how the distribution of acquired labels shifts over the course of the campaign — a window into *what* each strategy is choosing to label, not just *how well* the model performs.

Because this experiment uses a **single random seed**, there is no cross-seed variance to collapse. `learning_curve_summary` is constructed by renaming the raw metric columns to the `_mean` suffix expected by the plotting functions, then duplicating those values as `_lower` and `_upper` confidence bounds — effectively zero-width bands. This keeps all downstream plotting calls compatible with the multi-seed band-plot API without requiring any changes to the plot helpers.

## Results

Run `python analysis.py` to regenerate all figures. They are written to `results/` as self-contained interactive HTML files.

### Learning curves

[![Learning curves — MAE (click for interactive version)](results/learning_curve_mae.png)](results/learning_curve_mae.html)

[![Learning curves — Kendall's τ (click for interactive version)](results/learning_curve_ktau.png)](results/learning_curve_ktau.html)

The strategies converge to similar but not identical terminal performance. On MAE, **Exploitation** is the weakest at 0.55, **UCB** improves slightly to 0.54, and the remaining four strategies — **EI**, **Random**, **Exploration**, and **Diversity** — all land at 0.52. The differences are small, but the pattern is telling: the two strategies that most aggressively target high predicted pEC50 (Exploitation and UCB) pay a modest accuracy penalty, likely because they concentrate labels in a narrow region of chemical space and leave the rest of the SAR undersampled. On Kendall's τ all methods are indistinguishable at ~0.53, meaning the rank-ordering of predictions is equally good regardless of how compounds were selected.

This is likely a consequence of CheMeleon's pretraining and the large per-iteration batch size. When the base representation is already well-suited to the task, the model extracts near-maximum information from almost any labeled set, and the marginal value of *which* compounds to label diminishes. The 100-compound query batch compounds this further: selecting 100 molecules at once is a coarse-grained update that gives every strategy substantial coverage of chemical space at each step, leaving little room for a smarter selector to pull ahead. Active learning's label-efficiency advantage is most pronounced when the base model is data-hungry and queries are small relative to the pool.

Crucially, even at the start of the campaign — armed only with the ~600 ChEMBL pretraining labels — the MAE is reasonable (~0.7–0.8), thanks to CheMeleon's pretraining. A randomly initialized model would likely be much worse.

### Hit discovery

[![Hit discovery curve (click for interactive version)](results/hit_discovery_curve.png)](results/hit_discovery_curve.html)

Where the strategies *do* diverge is in hit-finding. **Exploitation** and **UCB** recover all but one of the actives in the pool, reflecting their shared bias toward high predicted pEC50 — they converge on the potent region of chemical space efficiently. **EI**, **Random**, and **Diversity** recover a similar but slightly smaller fraction, with EI's explore–exploit balance offering no clear advantage over random at this scale. **Exploration** finds the fewest actives: by ignoring predicted activity entirely and querying only by uncertainty, it maps the epistemic landscape of the model rather than the activity landscape of the assay, spending queries on uninformative regions.

## Navigating chemical space with GTM

Learning curves tell us *how fast* a strategy converges, but they say nothing about *where* in chemical space each strategy chooses to look. A strategy that exploits a single potent scaffold may show an early MAE drop while leaving most of the chemical diversity untouched — a dangerous blind spot in a real campaign.

We reuse the GTM embedding (fit before the active learning loop) for visualization, replaying the **Exploitation** selection history iteration by iteration:

* **Gray** — the full compound pool (all available unlabeled candidates)
* **Blue** — compounds acquired in any *prior* iteration (accumulated history)
* **Red** — compounds selected in the *current* iteration

Use the slider to step through iterations manually, or press **▶ Play** to watch the campaign unfold.

[![Exploitation compound selection in GTM chemical space — final state colored by iteration (click for interactive animation)](results/gtm_selection_animation_exploitation.png)](results/gtm_selection_animation_exploitation.html)

[![Active learning selection in TMAP chemical space, EI strategy (click for interactive version)](results/tmap_selection.png)](results/tmap_selection.html)

## Are our uncertainties trustworthy?

A model with good MAE can still be overconfident. In active learning, this is dangerous: if the model is confident but wrong about an unlabeled region, it may never query it (for **EI** / **UCB**). We tackled the topic of model uncertainty in another OpenADMET blogpost, [Concerning Uncertainty](https://openadmet.ghost.io/concerning-uncertainty/).

We evaluate calibration using the **miscalibration area**. A perfectly calibrated model has e.g. 90% of data points falling within its 90% confidence interval. At each active learning iteration, 10% of the pool-acquired labels are held out as a training-phase calibration set (`train_cal`) and used to fit a **scaling factor** calibrator on the committee's uncertainty estimates. `analysis.py` then visualises the before/after calibration curves evaluated on the held-out `df_test`.

[![Uncertainty calibration curve before and after scaling-factor calibration (click for interactive version)](results/calibration_curve.png)](results/calibration_curve.html)

[![Miscalibration area per iteration — before and after calibration (click for interactive version)](results/calibration_area_per_iteration.png)](results/calibration_area_per_iteration.html)

The miscalibration area is nearly identical before and after applying the scaling-factor calibration, and it remains flat across all AL iterations. This is not a failure of the calibration method, but a structural consequence of distribution shift: the scaling factor is fit on a holdout of the *AL-acquired pool*, then evaluated on a scaffold-split test set. Miscalibration on structurally novel scaffolds has a systematically different character from miscalibration on the explored pool, so a global scale correction learned on pool compounds does not transfer. A more flexible method such as isotonic regression would not resolve this either — a calibrator trained on one region of chemical space and applied to another is inherently limited regardless of its flexibility.

The flat trajectory also tells us that the committee's uncertainty structure is essentially fixed by the model architecture and training procedure; more labeled data does not change how the ensemble disagrees. This is expected for deep ensembles trained with bootstrap bagging, where all members share the same inductive bias.

The more informative diagnostic for active learning is whether σ *correlates* with actual prediction error — a ranking question that can be assessed with Spearman ρ between σ and |error|. Absolute coverage (the calibration curve area) is less meaningful when the calibration and evaluation distributions are separated by design.

## Takeaways

1. **Foundation models and large batches flatten label-efficiency gaps**: Strategies converge to near-identical terminal accuracy — all within 0.03 MAE. Exploitation-heavy strategies pay a small accuracy penalty by concentrating labels in a narrow region; the rest reach 0.52 MAE and ~0.53 Kendall's τ. Two factors suppress the advantage of smarter acquisition: CheMeleon's pretraining means any reasonable labeled set produces a capable model, and a 100-compound batch per iteration is a coarse enough update that fine-grained selection strategy differences wash out.
2. **Hit-finding and model accuracy are separable**: Exploitation and UCB recover nearly all actives in the pool despite no accuracy advantage over Random. The acquisition function shapes *what* the model finds, not *how well* it predicts.
3. **Exploration is a poor hit-finder**: Sampling purely by uncertainty ($\sigma$) maps the epistemic landscape of the model but ignores the activity landscape of the assay, spending queries on uninformative low-activity regions. It is best understood as a diagnostic: if Exploration outperforms EI, the committee is under-exploring.
4. **Diversity ensures coverage**: GTM-based max-min selection prevents scaffold collapse and produces the most structurally diverse labeled set. It is the safest strategy when potency information is completely absent, but sacrifices hit-finding speed.
5. **Calibration does not transfer across the scaffold split**: The scaling-factor calibration leaves the miscalibration area unchanged because the calibrator is fit on AL-acquired pool compounds and evaluated on structurally distinct test scaffolds. This is an inherent limitation of post-hoc calibration under distribution shift, not a failure of the method. The uncertainty estimates are still useful for acquisition (relative ordering is preserved), but their absolute coverage on unseen scaffolds should not be trusted.
6. **Recommendation**: For early-stage hit-finding, use **Exploitation** or **UCB** — they find the most actives. For building a generalizable SAR model, all strategies perform equivalently; **Random** is a perfectly defensible baseline. Use **Diversity** only when structural coverage is the explicit goal.

## Limitations and next steps

- **Time Split**: A scaffold split is a good proxy, but a true temporal split (training on data < 2020, testing on > 2020) is the gold standard.
- **Batch Awareness**: We selected the top $m$ compounds independently. Advanced "batch active learning" ensures diversity *within* the query batch to avoid selecting identical analogues. The **Diversity** strategy already addresses this implicitly by maximising structural dissimilarity from the labeled set.

Check out the [`openadmet-models`](https://github.com/OpenADMET/openadmet-models) repository for the full code and more advanced featurizers.

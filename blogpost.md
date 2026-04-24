> Please provide feedback using the comment functionality on GitHub

# Teaching models to ask: active learning for pEC50 prediction

In drug discovery, the most valuable resource is data. Synthesizing and assaying a single compound can cost thousands of dollars and take weeks. Yet, most machine learning models are trained as if labels are free, consuming massive random splits of [ChEMBL](https://www.ebi.ac.uk/chembl/) or [Enamine Real](https://enamine.net/compound-collections/real-compounds/real-database).

[**Active learning (AL)**](https://en.wikipedia.org/wiki/Active_learning_(machine_learning)) flips this paradigm. Instead of passively accepting a training set, the model iteratively selects the compounds it finds most confusing (to "explore") or most promising (to "exploit"). In this post, we build an active learning loop using [`openadmet-models`](https://github.com/OpenADMET/openadmet-models), powered by the [**CheMeleon**](https://github.com/JacksonBurns/chemeleon) foundation model, to simulate a campaign targeting the [Pregnane X Receptor (PXR)](https://en.wikipedia.org/wiki/Pregnane_X_receptor). We compare six acquisition strategies and show that intelligent selection achieves high predictive accuracy and finds active compounds, all with far less data than random screening.

## Selected targets

### PXR

PXR is a ligand-activated nuclear transcription factor that functions as the body's primary xenobiotic sensor, highly expressed in the liver and intestines. Its uniquely large, flexible, and hydrophobic ligand-binding pocket makes it notoriously promiscuous, capable of accommodating a massive variety of chemical scaffolds. When a molecule binds PXR, it induces [CYP3A4](https://en.wikipedia.org/wiki/CYP3A4) and related drug-metabolizing enzymes, accelerating the metabolic clearance of co-administered therapies and causing severe [drug-drug interactions (DDIs)](https://en.wikipedia.org/wiki/Drug_interaction). PXR is therefore treated as a high-priority [ADMET](https://en.wikipedia.org/wiki/ADME) antitarget. We model its binding affinity to flag this liability early in drug design, before compounds advance to costly clinical trials. Our colleagues at [Octant Bio](https://www.octant.bio/) have collected the largest public PXR dataset to date (5x larger than what's in ChEMBL, and all from the same source institution), recently released as part of a [blind challenge](https://openadmet.ghost.io/predicting-pxr-induction-we-have-liftoff/). Crucially, the PXR pool compounds are drawn from an **Enamine diversity deck** — a broad, structurally diverse compound collection selected to maximally cover chemical space, rather than to optimize a particular series.

### SARS-CoV-2 Mpro (ASAP Discovery)

Our second target is the main protease (Mpro) of SARS-CoV-2, an essential viral enzyme and a proven antiviral drug target. The data comes from the [ASAP Discovery](https://asapdiscovery.org/) program, an open-science antiviral drug discovery consortium. Unlike the PXR diversity deck, the ASAP dataset consists of **real-world congeneric series** — tightly clustered chemical matter iterated by medicinal chemists to optimize potency — which is the typical structure of data generated during a focused lead optimization campaign.

These two targets represent contrasting active learning scenarios: PXR tests whether active learning can efficiently navigate a **broad, structurally diverse** chemical space, while ASAP Mpro tests its performance in the **focused, congeneric** setting more common in practice. A key question is whether the same acquisition strategy excels in both regimes, or whether the optimal choice depends on the nature of the compound collection.

## The label bottleneck in drug discovery

In lead optimization, we typically work with hundreds to a few thousand compounds — a small fraction of the chemical space that could be explored. Assaying every candidate is expensive and slow, so *which* compounds we choose to test matters enormously: the composition of the training set shapes both what the model learns and where it generalizes.

Active learning formalizes this intuition. Rather than selecting compounds at random, the model identifies which untested candidates would be most informative to measure next. Our **goal** is to build an accurate activity model while minimizing the number of assays required — finding the most potent compounds and learning the structure-activity landscape as efficiently as possible.

What we are **querying** is a large pool of unlabeled candidate compounds. Their true activity values are hidden from the model; they are revealed only when a compound is nominated for assay, exactly as in a real experimental campaign. We also begin with a foundation of existing measurements from external sources, giving the model a starting foothold before the pool is touched.

**How** the model decides what to query next is the central question of this post. At each iteration, some acquisition strategy scores every unlabeled candidate and selects a batch to assay. We compare six such strategies — ranging from pure exploitation of the model's predictions to pure exploration of uncertain or structurally novel regions — to understand the tradeoffs between finding actives quickly and learning a broadly accurate model.

## Practical considerations

Our benchmark is designed around an as-realistic-as-possible scenario: you have a folder of legacy assay data from the public domain or a related project, a plate of untested candidate compounds, and a busy lab with queues, staff, and resource requirements. The choices below are a best-effort attempt to weight these practical considerations.

### Starting conditions: model initialization and historical data

The earliest iterations of an active learning campaign are the most precarious. With only a handful of labeled compounds, a model trained from scratch has limited reliable signal. Its predictions are essentially noise, and any acquisition strategy built on those predictions is resultingly noisy. Rather than picking a single starting condition, we treat this as a variable and compare three configurations:

1. **ChemProp (random init, no ChEMBL)** — a [ChemProp](https://github.com/chemprop/chemprop) message-passing neural network initialized with random weights and no external pretraining data. This is the true cold-start baseline: the model must learn everything it knows from the compounds queried during the campaign itself.

2. **[CheMeleon](https://github.com/JacksonBurns/chemeleon) (no ChEMBL)** — the same MPNN architecture, but initialized from CheMeleon weights pretrained on millions of molecules. Pretraining instills broadly useful molecular representations that transfer well to novel tasks, giving the model a usable prior before any target-specific data arrives — without requiring any target-relevant historical measurements.

3. **CheMeleon + ChEMBL** — CheMeleon weights further augmented by seeding training with publicly available target-relevant measurements from [ChEMBL](https://www.ebi.ac.uk/chembl/) (~600 entries). This mirrors the real-world scenario in which a practitioner begins a new project armed with both a pretrained backbone and whatever historical assay data is practically available.

Comparing these three conditions lets us disentangle the contributions of *architectural pretraining* and *historical data* to early-campaign performance, and assess whether the additional setup cost of sourcing ChEMBL data is worth it.

### Evaluating generalization

To measure how well the model performs, we hold out a fixed test set before the campaign begins and never touch it during acquisition. The right way to construct that test set depends on the structure of the data.

**For PXR**, the pool is drawn from an Enamine diversity deck — a collection explicitly designed for maximal structural coverage. As a result, the dataset is not very self-similar: random, scaffold, and cluster splits yield comparable model performance, because the test compounds are no more structurally foreign to the training set than they would be under any other partitioning scheme. Given this, we use a simple **random 80/20 split**. The 80% becomes the candidate pool for the active learner; the 20% is the held-out evaluation benchmark used throughout all iterations.

**For ASAP Mpro**, the data was generated in chronological waves of medicinal chemistry iteration, so a meaningful temporal signal exists. We use the **predefined time split** provided with the dataset, which mirrors how the data would have been encountered in a real campaign: earlier compounds for training, later compounds for evaluation. This is a more realistic test of generalization — the model must predict activity for chemical matter synthesized *after* the training cutoff, capturing the true challenge of prospective prediction in drug discovery.

### Query batch size: matching the lab

Many academic demonstrations of active learning query one compound at a time — or at most batches of 10 or 20. This is computationally convenient but experimentally unrealistic. In practice, dose-response assays are run on plates: a standard 1536-well microplate at roughly 13-point dose-response accommodates approximately 100 compounds per run (we'll leave some room for QC and controls). Querying fewer than a plate's worth of compounds per iteration would leave plates partially filled, reducing throughput and complicating scheduling.

We therefore query 100 compounds per iteration — one full plate's equivalent. Even this is conservative: most labs would prefer to fill multiple plates between model retraining cycles, particularly early in a campaign when assay infrastructure is under-utilized. The gap between the batch sizes used in AL benchmarks and what labs would actually run is worth acknowledging; results from single-compound or tiny-batch settings may not transfer directly to experimental practice.

## GTM chemical space embedding

The **Diversity** acquisition strategy selects compounds maximally dissimilar from the current labeled set in chemical space. We measure structural distance with a **Generative Topographic Map (GTM)**, a probabilistic manifold projection that maps high-dimensional ECFP4 fingerprints onto an interpretable 2D grid. Each compound gets a single (x, y) coordinate, and pairwise Euclidean distances serve as a fast proxy for molecular dissimilarity. GTM is also used in downstream visualizations.

## Query-by-committee: how ensemble disagreement guides exploration

Our committee consists of $N=5$ independent [**CheMeleon**](https://github.com/JacksonBurns/chemeleon) models. When we present an unlabeled molecule to the committee, we get 5 predictions.

- The **mean** prediction ($\mu$) is our best guess for activity.
- The **standard deviation** ($\sigma$) represents epistemic uncertainty.

Different acquisition strategies leverage these metrics:
- **Exploitation**: Greedy selection of the highest $\mu$. Finds good compounds fast but can get stuck in local optima.
- **Upper Confidence Bound (UCB)**: $\mu + \beta\sigma$. Optimistically explores regions that *might* be high activity.
- **Expected Improvement (EI)**: Balances $\mu$ and $\sigma$ to calculate the probability of exceeding the current best label $f^*$ (equation below).
- **Exploration**: Pure uncertainty sampling. Selects the $m$ compounds with the highest $\sigma$, ignoring predicted activity. Useful as a baseline that maximizes coverage of epistemic uncertainty.
- **Diversity**: Ignores model predictions altogether and selects the $m$ compounds farthest from the current labeled set in 2D GTM chemical space (max-min Euclidean distance), enforcing structural dissimilarity between batches.

$$EI(\mathbf{x}) = (\mu(\mathbf{x}) - f^* - \xi)\,\Phi(Z) + \sigma(\mathbf{x})\,\phi(Z)$$

where $Z = \dfrac{\mu(\mathbf{x}) - f^* - \xi}{\sigma(\mathbf{x})}$.

To achieve ensemble diversity, each member uses different initialization ([deep ensembling](https://dl.acm.org/doi/10.5555/3295222.3295387)) and trains on a bootstrapped sample of the labeled set. This diversity, captured as disagreement at prediction time, gives the committee meaningful epistemic uncertainty for the acquisition function.

At every iteration, we hold out 10% of the pool-acquired labels to fit a **scaling factor** calibrator on the committee's uncertainty estimates. This rescales σ so that coverage intervals better match observed error rates. Calibration does not affect acquisition: **Exploration**, **EI**, and **UCB** rank candidates by σ, and a global scale factor preserves that ordering. It does, however, shape the *final model's* confidence intervals, which is important for evaluating quality of models produced.

## The active learning loop

The core logic lives in [src/helpers.py](src/helpers.py). `train_committee` fits a bootstrapped ensemble on the current labeled set, `query_batch` selects the next $m$ molecules using the committee's μ and σ (or GTM distances for **Diversity**), and `evaluate_on_test` scores the result on the fixed test set. These steps repeat $k$ times, and all six strategies are dispatched through `query_batch` via a single `strategy` argument.

In each iteration $k$:
1. Train the committee on the current `labeled_pool`.
2. Evaluate on the static `test_set` to track performance.
3. Use the committee to predict on the `unlabeled_pool`.
4. Score the unlabeled molecules with the acquisition function.
5. "Acquire" the top $m$ molecules (reveal their labels).
6. Add them to the `labeled_pool` and repeat.

We track MAE, Kendall's τ, chemical space coverage (GTM- and TMAP- based), uncertainty miscalibration area and correlation with error, as well as actives "found" during the active learning campaign.

## Running the experiments

Run `python run.py` to execute the loop for all six strategies with a fixed seed. Results are checkpointed to `results/all_runs.pkl` after each strategy, so the run can be safely interrupted and resumed. The checkpoint stores per-iteration metrics, labeled pool snapshots, GTM coordinates, split DataFrames, seed data, and campaign configuration, making `analysis.py` self-contained without access to the original dataset.

## Results

Run `python analysis.py` to generate all figures. They are written to `results/` as self-contained interactive HTML files and static SVGs.

### Learning curves

The strategies converge to comparable terminal performance. On MAE, **Exploitation** reaches 0.64, **UCB** 0.59, and the remaining four strategies (**EI**, **Random**, **Exploration**, and **Diversity**) all land at 0.56. The modest accuracy penalty for exploitation-heavy strategies likely reflects their tendency to concentrate labels in a narrow region of chemical space.

[![Learning curves — MAE (click for interactive version)](plots/learning_curve_mae.svg)](plots/learning_curve_mae.html)

*Figure 1. Mean absolute error (MAE, pEC50 units) on the held-out scaffold-split test set as a function of labeled pool size, for each of the six acquisition strategies. Shaded bands show ±1 SD across five random seeds.*

On Kendall's τ all methods are all at approximately ~0.57, meaning the rank-ordering of predictions is about equal regardless of how compounds were selected.

[![Learning curves — Kendall's τ (click for interactive version)](plots/learning_curve_ktau.svg)](plots/learning_curve_ktau.html)

*Figure 2. Kendall's τ rank-correlation between predicted and observed pEC50 on the held-out test set across active learning iterations. Higher values indicate better ranking of compounds by predicted activity. Shaded bands show ±1 SD across five random seeds.*

### Hit discovery

One area where the strategies *do* diverge is in hit-finding. **Exploitation** and **UCB** recover all but one of the actives in the pool, converging on the potent region of chemical space efficiently. **EI**, **Random**, and **Diversity** recover a slightly smaller fraction, with **EI** offering no clear advantage over random at this scale. **Exploration** finds the fewest actives. By querying only by uncertainty, it maps model uncertainty rather than compound potency, spending queries on uninformative regions.

[![Hit discovery curve (click for interactive version)](plots/hit_discovery_curve.svg)](plots/hit_discovery_curve.html)

*Figure 3. Cumulative number of active compounds (pEC50 ≥ 6.3) recovered as a function of labeled pool size for each acquisition strategy. Exploitation- and UCB-based strategies recover the most actives, reflecting their shared bias toward high predicted activity.*

## Navigating chemical space with GTM

Learning curves tell us *how fast* a strategy converges but say nothing about *where* in chemical space it looks. A strategy that exploits a single scaffold may show early MAE gains while leaving most chemical diversity untouched, a potentially dangerous blind spot in a real campaign.

We reuse the GTM embedding (fit before the active learning loop) for visualization, replaying the **Exploitation** selection history iteration by iteration:

* **Gray** — the full compound pool (all available unlabeled candidates)
* **Blue** — compounds acquired in any *prior* iteration (accumulated history)
* **Red** — compounds selected in the *current* iteration

Use the slider to step through iterations manually, or press **▶ Play** to watch the campaign unfold. In the static version below, points are colored by iteration, as indicated by the color bar.

[![Exploitation compound selection in GTM chemical space — final state colored by iteration (click for interactive animation)](plots/gtm_selection_animation_exploitation.svg)](plots/gtm_selection_animation_exploitation.html)

*Figure 4. Final-state GTM embedding of the compound pool for the Exploitation strategy. Each point is a compound projected onto the 2D GTM manifold; color indicates the AL iteration in which it was first selected (viridis scale, earlier iterations darker). Gray points were never selected. Click to open the interactive animation with a per-iteration slider.*

The GTM gives a global view on a smooth 2D lattice. A complementary perspective comes from **TMAP** (Tree-based MAP), which organizes compounds via a **minimum-spanning tree** (MST) over their Morgan fingerprint (radius 2, 1024 bits) similarities. Where the GTM imposes a regular grid, TMAP respects the data topology: edges connect near-neighbors in fingerprint space, and branches trace shared scaffolds or chemical series.

This makes TMAP useful for a question the GTM cannot answer: *does the strategy stay within one branch, or spread across the tree?* A strategy concentrated in one cluster may find actives quickly but leave entire branches unexplored, while one that fans outward covers more scaffolds at the cost of spending queries on uninformative regions. In the figure below, each point is colored by the first AL iteration it was selected (viridis scale, with unselected compounds in light gray). Hover over any point in the interactive version to inspect its SMILES.

[![Active learning selection in TMAP chemical space, Exploitation strategy (click for interactive version)](plots/tmap_selection.svg)](plots/tmap_selection.html)

*Figure 5. TMAP layout of the compound pool for the Exploitation strategy, with the minimum-spanning-tree overlay drawn in gray. Point color encodes the first AL iteration in which each compound was selected (viridis scale); unselected compounds are shown in light gray. Click for the interactive Faerun version with SMILES tooltips.*

## Are our uncertainties trustworthy?

A model with good MAE can still be overconfident. In active learning this is dangerous, since a confident-but-wrong model may never query a whole region of chemical space (for **EI** / **UCB**). We explored model uncertainty more broadly in [Concerning Uncertainty](https://openadmet.ghost.io/concerning-uncertainty/).

We evaluate calibration using the **miscalibration area**, the integrated deviation from perfect coverage (e.g., 90% of observations within the 90% confidence interval).

[![Uncertainty calibration curve before and after scaling-factor calibration (click for interactive version)](plots/calibration_curve.svg)](plots/calibration_curve.html)

*Figure 6. Expected vs. observed coverage curves (calibration plots) before (uncalibrated) and after (calibrated) applying isotonic-regression scaling-factor calibration, evaluated on the scaffold-split test set at the final AL iteration of the Exploitation run. A perfectly calibrated model follows the diagonal.*

[![Miscalibration area per iteration — before and after calibration (click for interactive version)](plots/calibration_area_per_iteration.svg)](plots/calibration_area_per_iteration.html)

*Figure 7. Miscalibration area (integrated deviation from perfect calibration) as a function of AL iteration for all six acquisition strategies, shown before and after scaling-factor calibration. Lower values indicate better-calibrated uncertainty estimates.*

The miscalibration area is nearly identical before and after calibration (Figure 6) and remains flat across all iterations (Figure 7). This is a structural consequence of distribution shift rather than a calibration failure: the scaling factor is fit on pool-acquired compounds and evaluated on scaffolds held out by design, so the correction does not transfer.

The flat trajectory also confirms that the committee's uncertainty structure is set by architecture and training. More labeled data does not change how the ensemble disagrees, as expected for deep ensembles with shared inductive bias.

A more informative diagnostic is whether σ *correlates* with actual prediction error, assessed via Spearman ρ between σ and |error|. Absolute coverage is less meaningful when calibration and evaluation distributions are separated by design.

[![Spearman ρ(σ, |error|) per iteration — uncertainty–error correlation (click for interactive version)](plots/sigma_error_correlation.svg)](plots/sigma_error_correlation.html)

*Figure 8. Spearman rank correlation between predicted uncertainty (σ) and absolute prediction error (|ŷ − y|) on the held-out test set, as a function of labeled pool size, for each acquisition strategy. A positive ρ indicates that σ correctly ranks which test compounds the model is most wrong about. Shaded bands show ±1 SD across five random seeds.*

The correlations are positive and follow a clear gradient by exploitation intensity. **Exploitation** achieves ρ ≈ 0.30, peaking near 0.38 at ~1,700 labeled molecules before declining slightly, **UCB** reaches ~0.21, and the remaining strategies group around 0.17. Non-overlapping bands for **Exploitation** make this curious finding worth of investigation. No strategy reliably ranks which novel test scaffolds will carry the largest errors, but the exploitation-intensity gradient across strategies represents *some* signal.

Together, these results paint a mixed picture. Absolute coverage on novel scaffolds is untrustworthy for all strategies, and **Exploration** finishing last in hit-finding confirms that the uncertainty signal is unreliable at acquisition time. **EI** offers no advantage over **Random**, and **UCB**'s modest hit-finding edge is more plausibly driven by exploitation bias than its variance term.

The σ-error correlation results tell a more structured story. The ordering (**Exploitation** > **UCB** > the σ-influenced strategies) mirrors exploitation intensity precisely, and the separation is statistically robust. One provisional explanation is a training-data concentration effect. By focusing labels in the high-activity region, **Exploitation** leaves test scaffolds consistently far from its training distribution, and structural distance from training tends to produce higher ensemble disagreement. The committee's σ therefore incidentally tracks |error| better than when labels are spread evenly. This remains provisional, but it carries a practical implication: an exploitation-heavy campaign may paradoxically yield better σ-error correlation than one guided explicitly by uncertainty.

## Takeaways

1. **Foundation models and large batches flatten label-efficiency gaps.** All strategies converge within 0.03 MAE of each other. CheMeleon's pretraining means any reasonable labeled set produces a capable model, and 100-compound batches are coarse enough that fine-grained strategy differences wash out.
2. **Hit-finding and model accuracy are separable.** **Exploitation** and **UCB** recover nearly all actives despite no accuracy advantage over Random. The acquisition function shapes *what* the model finds, not *how well* it predicts.
3. **Exploration is a poor hit-finder.** Sampling purely by $\sigma$ maps model uncertainty rather than compound potency. Use it as a diagnostic: if **Exploration** outperforms **EI**, the committee is under-exploring.
4. **Diversity ensures coverage.** GTM-based max-min selection prevents scaffold collapse and produces the most structurally diverse labeled set. It is the safest strategy when potency information is absent, but sacrifices hit-finding speed.
5. **Uncertainty quality depends on training concentration, not uncertainty-guided sampling.** Absolute calibration on novel test scaffolds fails for all strategies. But σ-error correlation follows a clear gradient (**Exploitation** ρ ≈ 0.30, **UCB** ~0.21, remaining strategies ~0.17), tracking exploitation intensity precisely. An exploitation-heavy campaign may paradoxically yield better-ranked uncertainty estimates than one guided explicitly by σ.
6. **Recommendation.** For early-stage hit-finding, use **Exploitation** or **UCB**. For a generalizable SAR model, all strategies perform equivalently and **Random** is a perfectly defensible baseline. Use **Diversity** only when structural coverage is the explicit goal.


## Reproducibility

To reproduce the results in this post, install `openadmet-models` by following the [installation instructions](https://docs.openadmet.org/en/latest/installation.html), then run:

```bash
# Execute the active learning pipeline (~several GPU-hours)
python run.py   

# Generate all figures from results/all_runs.pkl
python analysis.py  
```

All supporting code lives in `src/`: [src/helpers.py](src/helpers.py) contains the core AL utilities and [src/plots.py](src/plots.py) contains all Plotly/Faerun plotting functions. The campaign is governed by parameters specified in [`config.yaml`](config.yaml).

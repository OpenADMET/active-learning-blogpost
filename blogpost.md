# Model, take the wheel: active learning with PXR pEC50 data

In computational drug discovery, the most valuable resource is data. Synthesizing and assaying even a handful of compounds can cost thousands of dollars and take weeks. Yet, many machine learning models are trained as if labels are free, consuming massive random splits of [ChEMBL](https://www.ebi.ac.uk/chembl/). As increasing effort turns towards acquiring assay data **specifically** for the purpose of training machine learning models, a natural question arises \- is there an intelligent way to leverage what we already know about a target to drive our assay campaign towards its target destination?

[**Active learning (AL)**](https://en.wikipedia.org/wiki/Active_learning_\(machine_learning\)) flips the traditional machine learning paradigm. Instead of working with static training and test sets, the model iteratively selects the compounds it is uncertain about  (to "explore") or most promising (to "exploit"). We leverage what the model has learned to guide collection of high impact datapoints, amplifying model improvement and hit finding per $ spent.

In this post, we build an active learning loop using [`openadmet-models`](https://github.com/OpenADMET/openadmet-models), powered by the [**CheMeleon**](https://github.com/JacksonBurns/chemeleon) foundation model, to simulate a campaign targeting the [Pregnane X Receptor (PXR)](https://en.wikipedia.org/wiki/Pregnane_X_receptor). We compare six acquisition strategies and compare their efficacy in predictive accuracy and hit finding, all with far less data than random selection.

## Selected targets

PXR is a ligand-activated nuclear transcription factor that functions as the body's primary xenobiotic sensor, highly expressed in the liver and intestines. Its uniquely large, flexible, and hydrophobic ligand-binding pocket makes it notoriously promiscuous, capable of accommodating a massive variety of chemical scaffolds. When a molecule binds PXR, it induces [CYP3A4](https://en.wikipedia.org/wiki/CYP3A4) and related drug-metabolizing enzymes, accelerating the metabolic clearance of co-administered therapies and causing severe [drug-drug interactions (DDIs)](https://en.wikipedia.org/wiki/Drug_interaction). PXR is therefore treated as a high-priority [ADMET](https://en.wikipedia.org/wiki/ADME) antitarget. We model its binding affinity to flag this liability early in drug design, before compounds advance to costly clinical trials. Our colleagues at [Octant Bio](https://www.octant.bio/) have collected the largest public PXR dataset to date (5x larger than what's in ChEMBL, and all from the same source institution), recently released as part of a [blind challenge](https://openadmet.ghost.io/predicting-pxr-induction-we-have-liftoff/). Crucially, the PXR pool compounds are drawn from an [**Enamine diversity deck**](https://enamine.net/compound-libraries/diversity-libraries/dds-10240) — a broad, structurally diverse compound collection selected to maximally cover chemical space, rather than to optimize a particular series.

Our second target is the [main protease](https://en.wikipedia.org/wiki/3C-like_protease) (Mpro) of [SARS-CoV-2](https://en.wikipedia.org/wiki/SARS-CoV-2), an essential viral enzyme and a proven antiviral drug target. The data comes from the [ASAP Discovery](https://asapdiscovery.org/) program, an open-science antiviral drug discovery consortium. Unlike the PXR diversity deck, the ASAP dataset consists of **real-world congeneric series** — tightly clustered chemical matter iterated by medicinal chemists to optimize potency — which is the typical structure of data generated during a focused lead optimization campaign.

These two targets represent contrasting active learning scenarios: PXR tests whether active learning can efficiently navigate a **broad, structurally diverse** chemical space, while ASAP Mpro tests its performance in the **focused, congeneric** setting more common in practice. A key question is whether the same acquisition strategy excels in both regimes, or whether the optimal choice depends on the nature of the compound collection.

## The label bottleneck in drug discovery

In lead optimization, we typically work with hundreds to a few thousand compounds: a small fraction of chemical space. Assaying every candidate is expensive and slow, so *which* compounds we choose to test matters enormously: the composition of the training set shapes both what the model learns and where it generalizes.

Active learning formalizes this intuition. Rather than selecting compounds at random, the model identifies which untested candidates would be most informative to measure next. Our **goal** is to build an accurate activity model while minimizing the number of assays required — finding the most potent compounds and learning the structure-activity landscape as efficiently as possible.

Toward this end, we **query** from a large pool of “unlabeled” candidate compounds. Their true activity values are hidden from the model; they are revealed only when a compound is nominated for assay, exactly as in a real experimental campaign. We also evaluate with an optional foundation of existing measurements from external sources, giving the model a starting foothold before the pool is touched.

**How** the model decides what to query next is the central question of this post. At each iteration, an acquisition strategy scores every unlabeled candidate and selects a batch to assay. We compare six such strategies, ranging from pure **exploitation** of the model's predictions to pure **exploration** of uncertain or structurally novel regions, to understand the tradeoffs between finding actives quickly and learning a broadly accurate model. And, if possible, both.

## Practical considerations

Our benchmark is designed around an as-realistic-as-possible scenario: you have a folder of legacy assay data from the public domain or a related project, several plates of untested candidate compounds, and a busy lab with queue, staff, and resource requirements. The choices below are a best-effort attempt to balance these practical considerations.

### Cold-starting with a foundation model and public data

The earliest iterations of an active learning campaign are the most precarious. With only a handful of labeled compounds, a model trained from scratch has limited reliable signal. Its predictions are essentially noise, and any acquisition strategy built on those predictions is resultingly noisy. Rather than picking a single starting condition, we treat this as a variable and compare three configurations:

1. **ChemProp (random init, no ChEMBL)** — a [ChemProp](https://github.com/chemprop/chemprop) message-passing neural network initialized with random weights and no external pretraining data. This is the true cold-start baseline: the model must learn everything it knows from the compounds queried during the campaign itself.  
     
2. [**CheMeleon**](https://github.com/JacksonBurns/chemeleon) **(no ChEMBL)** — the same MPNN architecture, but initialized from CheMeleon weights pretrained on millions of molecules. Pretraining instills broadly useful molecular representations that transfer well to novel tasks, giving the model a usable prior before any target-specific data arrives — without requiring any target-relevant historical measurements.  
     
3. **CheMeleon \+ ChEMBL** — CheMeleon weights further augmented by seeding training with publicly available target-relevant measurements from [ChEMBL](https://www.ebi.ac.uk/chembl/) (\~600 entries). This mirrors the real-world scenario in which a practitioner begins a new project armed with both a pretrained backbone and whatever historical assay data is practically available.

Comparing these three conditions lets us disentangle the contributions of *architectural pretraining* and *historical data* to early-campaign performance, and assess whether the additional setup cost of sourcing ChEMBL data is worth it.

### Evaluating generalization

To measure how well the model performs, we hold out a fixed test set before the campaign begins and never touch it during acquisition. The right way to construct that test set depends on the structure of the data.

**For PXR**, the pool is drawn from an Enamine diversity deck, a collection explicitly designed for maximal structural coverage. As a result, the dataset is not very self-similar: random, scaffold, and cluster splits yield comparable model performance, because the test compounds are no more structurally foreign to the training set than they would be under any other partitioning scheme. Given this, we use a simple **random 80/20 split**. The 80% becomes the candidate pool for the active learner; the 20% is the held-out evaluation benchmark used throughout all iterations.

**For ASAP Mpro**, the data was generated in chronological waves of medicinal chemistry iteration, so a meaningful temporal signal exists. We use the **predefined time split** provided with the dataset, which mirrors how the data would have been encountered in a real campaign: earlier compounds for training, later compounds for evaluation. This is a more realistic test of generalization: the model must predict activity for chemical matter synthesized *after* the training cutoff, capturing the true challenge of prospective prediction in drug discovery.

### Query batch size: matching the lab

Many academic demonstrations of active learning query one compound at a time — or at most batches of 10 or 20\. This is computationally convenient but experimentally unrealistic. In practice, dose-response assays are run on plates: a standard 1536-well microplate at roughly 13-point dose-response accommodates approximately 100 compounds per run (we'll “leave some room” for QC and controls). Querying fewer than a plate's worth of compounds per iteration would leave plates partially filled, reducing throughput and complicating scheduling.

We therefore query 100 compounds per iteration, i.e. one full plate's equivalent. Even this is conservative: most labs would prefer to fill multiple plates between model retraining cycles, particularly early in a campaign when assay infrastructure is under-utilized. The gap between the batch sizes used in AL benchmarks and what labs would actually run is worth acknowledging; results from single-compound or tiny-batch settings may not transfer directly to experimental practice.

## GTM chemical space embedding

The **Diversity** acquisition strategy selects compounds maximally dissimilar from the current labeled set in chemical space. We measure structural distance with a **Generative Topographic Map (GTM)**, a probabilistic manifold projection that maps high-dimensional ECFP4 fingerprints onto an interpretable 2D grid. Each compound gets a single (x, y) coordinate, and pairwise Euclidean distances serve as a fast proxy for molecular dissimilarity. **This method does not use any model signal**, instead leveraging data-driven similarity to drive selection**.** We will observe the effects on both the inherently diverse PXR data, as well as the congeneric ASAP Mpro data. GTM is also used in downstream visualizations.

## Query-by-committee: how ensemble disagreement guides exploration

Our committee consists of $N=5$ independent [**CheMeleon**](https://github.com/JacksonBurns/chemeleon) models. To achieve ensemble diversity, each member uses *different initialization* ([deep ensembling](https://dl.acm.org/doi/10.5555/3295222.3295387)) and trains on a [*bootstrapped*](https://en.wikipedia.org/wiki/Bootstrapping_\(statistics\)) *sample* (with replacement) of the labeled set. This diversity, captured as disagreement at prediction time, ostensibly gives the committee an indication of epistemic uncertainty for the acquisition function. When we present an unlabeled molecule to the committee, we get 5 predictions.

- The **mean** prediction ($\\mu$) is our best guess for activity.  
- The **standard deviation** ($\\sigma$) represents epistemic uncertainty.

Different acquisition strategies leverage these metrics:

- **Exploitation**: Greedy selection of the highest $\\mu$. Finds good compounds fast but can get stuck in local optima.  
- **Upper Confidence Bound (UCB)**: $\\mu \+ \\beta\\sigma$. Optimistically explores regions that *might* be high activity (we nominally select $\\beta=2$).  
- **Expected Improvement (EI)**: Balances $\\mu$ and $\\sigma$ to calculate the probability of exceeding the current best label $f^\*$ (equation below).  
- **Exploration**: Pure uncertainty sampling. Selects the $m$ compounds with the highest $\\sigma$, ignoring predicted activity. Useful as a baseline that maximizes coverage of predicted epistemic uncertainty.  
- **Diversity**: Ignores model predictions altogether and selects the $m$ compounds farthest from the current labeled set in 2D GTM chemical space (max-min Euclidean distance), enforcing structural dissimilarity between batches.

![][image1]where ![][image2]

Note that uncertainty calibration, discussed in a previous post [Concerning Uncertainty](https://openadmet.ghost.io/concerning-uncertainty/), does not affect acquisition: **Exploration**, **EI**, and **UCB** rank candidates by σ, and a global scale factor preserves that ordering.

## The active learning loop

In each iteration $k$:

1. Train the committee on the current labeled pool.  
2. Evaluate on the static test set to track performance.  
3. Use the committee to predict on the unlabeled pool.  
4. Score the unlabeled molecules with the acquisition function.  
5. "Acquire" the top $m$ molecules (reveal their labels).  
6. Add them to the labeled pool and repeat.

Per iteration, we track MAE, Kendall's τ, chemical space coverage (GTM- and TMAP- based), uncertainty estimates, and number of active compounds "found" during the active learning campaign.

## Hit discovery

How does acquisition strategy affect how quickly we recover actives? 

### PXR

![][image3]  
*Figure 1\. PXR dataset cumulative number of active compounds (pEC50 ≥ 6.0) recovered as a function of labeled pool size for each acquisition strategy.* 

Across the PXR diversity deck (54 actives in a pool of 3312 compounds, a 1.6% hit rate), acquisition strategy has a dramatic effect on the rate of active discovery. Exploitation is the clear leader throughout the campaign, recovering approximately 70% of all pool actives by the time 25% of the pool has been labeled, compared to roughly 26% for Random sampling. UCB is consistently the second-best strategy, tracking well above Random across the full campaign. Diversity and Random perform nearly identically for hit-finding, as max-min structural diversity sampling does not preferentially seek high-potency regions. Exploration is the weakest strategy for recovering actives, often performing at or below Random throughout the campaign, since it directs queries toward regions of high model uncertainty rather than toward high predicted potency. CheMeleon initialization amplifies the hit-finding advantage of active strategies but does not alter their relative ranking.

### SARS-CoV-2 Mpro

![][image4]  
*Figure 2\. ASAP SARS-Cov-2 Mpro dataset cumulative number of active compounds (pEC50 ≥ 7.0) recovered as a function of labeled pool size for each acquisition strategy.* 

The ASAP Mpro dataset shows the same qualitative ordering of strategies despite a markedly higher base hit rate (76 actives in a pool of 842 compounds, 9.0%). Exploitation again leads throughout the campaign, recovering the majority of pool actives well before 25% of the pool is labeled, while Random trails substantially. Because the congeneric lead series is structurally more homogeneous than the PXR diversity deck, the absolute gap between strategies is compressed relative to the PXR results, but the rank ordering of strategies is fully preserved across both targets and all model configurations. CheMeleon models recover actives more efficiently in early iterations, particularly when warm-started with ChEMBL data.

## Model accuracy

How does acquisition strategy affect model accuracy over the course of the campaign? Does any particular strategy “learn more”, motivating particular experiment design to yield the most accurate models? 

### PXR

![][image5]  
*Figure 3\. PXR dataset mean absolute error (MAE, pEC50 units) on the held-out random-split test set as a function of labeled pool size, for each of the six acquisition strategies. Shaded bands show ±1 SD across five random seeds.*

In contrast to the strong strategy signal in hit discovery, acquisition strategy choice has a comparatively small effect on test-set MAE for PXR. All strategies converge to tightly overlapping error bands throughout the campaign, with differences of less than 0.03 pEC50 units between the best and worst performers at any given labeled pool size. Exploitation exhibits a slight MAE elevation relative to Random because it intentionally biases the labeled set toward actives, providing less coverage of the full activity range. This cost is modest in absolute terms and is far outweighed by the substantial hit-finding advantage that Exploitation provides. This is further substantiated with Kendall's τ, where all methods achieve \~0.55 with heavily overlapped error bands, meaning the rank-ordering of predictions functionally equivalent regardless of acquisition strategy.  
![][image6]  
*Figure 4\. PXR dataset Kendall's τ rank-correlation between predicted and observed pEC50 on the held-out test set across active learning iterations. Higher values indicate better ranking of compounds by predicted activity. Shaded bands show ±1 SD across five random seeds.*

### SARS-CoV-2 Mpro

![][image7]  
*Figure 5\. ASAP SARS-CoV-2 Mpro dataset mean absolute error (MAE, pEC50 units) on the held-out random-split test set as a function of labeled pool size, for each of the six acquisition strategies. Shaded bands show ±1 SD across five random seeds.*  
![][image8]  
*Figure 6\. ASAP SARS-CoV-2 Mpro dataset Kendall's τ rank-correlation between predicted and observed pEC50 on the held-out test set across active learning iterations. Higher values indicate better ranking of compounds by predicted activity. Shaded bands show ±1 SD across five random seeds.*

For ASAP Mpro, model initialization has a substantially larger effect on accuracy than acquisition strategy. CheMeleon reduces test-set MAE by approximately 0.15 to 0.20 pIC50 units relative to ChemProp at equivalent labeled pool sizes, reflecting the benefit of pretrained molecular representations for a congeneric series where transfer of learned chemical patterns is most effective. Within each model type, differences in MAE across strategies remain small, typically less than 0.05 pIC50 units. Kendall's τ reaches approximately 0.63 to 0.64 for CheMeleon and approximately 0.60 for ChemProp, with no meaningful strategy separation in either case.

## Model uncertainty

How does acquisition strategy affect quality of uncertainty estimates? Does a particular acquisition strategy result in the ability to better represent model error? We check how well σ *correlates* with actual prediction error, assessed via Spearman ρ between σ and absolute error.

### PXR

![][image9]  
*Figure 7\. PXR dataset Spearman rank correlation between predicted uncertainty (σ) and absolute prediction error (|ŷ − y|) on the held-out test set, as a function of labeled pool size, for each acquisition strategy. A positive ρ indicates that σ correctly ranks which test compounds the model is most wrong about. Shaded bands show ±1 SD across five random seeds.*

For PXR, the Spearman ρ between predicted uncertainty and absolute prediction error is consistently low across all strategies and model configurations, ranging from approximately 0.10 to 0.26 throughout the campaign. This indicates that the committee's uncertainty estimates are a weak proxy for actual prediction error regardless of how the labeled pool is acquired. Exploitation-focused strategies tend to show slightly lower ρ values, consistent with the committee becoming confidently biased toward the high-potency region it has preferentially sampled. Random and Diversity sampling maintain somewhat higher ρ values because the labeled pool retains broader coverage of the activity distribution. Notably, ρ does not improve monotonically as the labeled pool grows, suggesting that more data alone does not reliably calibrate uncertainty for this diverse compound set.

### SARS-CoV-2 Mpro

![][image10]  
*Figure 7\. ASAP SARS-CoV-2 Mpro dataset Spearman rank correlation between predicted uncertainty (σ) and absolute prediction error (|ŷ − y|) on the held-out test set, as a function of labeled pool size, for each acquisition strategy. A positive ρ indicates that σ correctly ranks which test compounds the model is most wrong about. Shaded bands show ±1 SD across five random seeds.*

The ASAP Mpro dataset shows broadly similar uncertainty behavior. Spearman ρ values remain low throughout the campaign (0.10 to 0.30), and CheMeleon initialization does not systematically improve the correspondence between σ and absolute error relative to ChemProp. As with PXR, Exploitation-focused strategies tend toward lower ρ, while Random and Diversity tend toward modestly higher ρ. The consistent weakness of uncertainty-error correlation across both targets and all model configurations suggests that ensemble disagreement in these committee models is shaped more by training distribution than by intrinsic compound difficulty.

## Takeaways

1. **Foundation models and large batches flatten label-efficiency gaps.** All strategies converge within 0.03 MAE of each other. CheMeleon's pretraining means any reasonable labeled set produces a capable model, and 100-compound batches are coarse enough that fine-grained strategy differences wash out.  
2. **Hit-finding and model accuracy are separable.** **Exploitation** and **UCB** recover nearly all actives despite no accuracy advantage over Random. The acquisition function shapes *what* the model finds, not *how well* it predicts.  
3. **Exploration is a poor hit-finder.** Sampling purely by $\\sigma$ maps model uncertainty rather than compound potency. Use it as a diagnostic: if **Exploration** outperforms **EI**, the committee is under-exploring.  
4. **Diversity ensures coverage.** GTM-based max-min selection prevents scaffold collapse and produces the most structurally diverse labeled set. It is the safest strategy when potency information is absent, but sacrifices hit-finding speed.  
5. **Uncertainty estimates are weak proxies for prediction error.** σ-|error| Spearman ρ remains low (0.10 to 0.30) across both targets, all strategies, and all model configurations. Exploitation biases the labeled pool toward actives and tends to suppress ρ further, while Random and Diversity preserve modestly better calibration. More data alone does not resolve this, as the committee's uncertainty reflects training distribution more than compound difficulty.
6. **Recommendation.** For early-stage hit-finding, use **Exploitation** or **UCB**. For a generalizable SAR model, all strategies perform equivalently and **Random** is a perfectly defensible baseline. Use **Diversity** only when structural coverage is the explicit goal.

## Reproducibility

To reproduce the results in this post, install `openadmet-models` by following the [installation instructions](https://docs.openadmet.org/en/latest/installation.html), clone the blogpost repo, then run:

```shell
# Execute the active learning pipeline (~several GPU-hours)
python run.py   

# Generate all figures from results/all_runs.pkl
python analysis.py  
```

All supporting code lives in `active-learning-blogpost/src/`: [src/helpers.py](http://src/helpers.py) contains the core AL utilities and [src/plots.py](http://src/plots.py) contains all Plotly/Faerun plotting functions. The campaign is governed by parameters specified in [`config.yaml`](http://config.yaml).  
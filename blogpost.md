# Model, take the wheel: active learning with PXR pEC50 data

In computational drug discovery, data is the most valuable resource. Synthesizing and assaying even a handful of compounds can cost thousands of dollars and take weeks. As more effort turns toward acquiring data **specifically** to train machine learning models, a natural question arises. Is there an intelligent way to use what a model already knows to guide the assay campaign toward its destination?

[**Active learning (AL)**](https://en.wikipedia.org/wiki/Active_learning_\(machine_learning\)) flips the traditional machine learning paradigm. Instead of working with static training and test sets, the model iteratively selects the compounds it is uncertain about  (to "explore") or most promising (to "exploit"). We leverage what the model has learned to guide collection of high impact datapoints, amplifying model improvement and hit finding per $ spent.

In this post, we build an active learning loop using [`openadmet-models`](https://github.com/OpenADMET/openadmet-models), powered by [ChemProp](https://chemprop.readthedocs.io/en/latest/) and the [**CheMeleon**](https://github.com/JacksonBurns/chemeleon) foundation model, to simulate a campaign targeting the [Pregnane X Receptor (PXR)](https://en.wikipedia.org/wiki/Pregnane_X_receptor) and the [main protease](https://en.wikipedia.org/wiki/3C-like_protease) (Mpro) of [SARS-CoV-2](https://en.wikipedia.org/wiki/SARS-CoV-2). We compare six acquisition strategies and compare their efficacy in predictive accuracy and hit finding, all with far less data than random selection.

## Selected targets

PXR is a ligand-activated nuclear transcription factor that functions as the body's primary xenobiotic sensor, highly expressed in the liver and intestines. Its uniquely large, flexible, and hydrophobic ligand-binding pocket makes it notoriously promiscuous, capable of accommodating a massive variety of chemical scaffolds. When a molecule binds PXR, it induces [CYP3A4](https://en.wikipedia.org/wiki/CYP3A4) and related drug-metabolizing enzymes, accelerating the metabolic clearance of co-administered therapies and causing severe [drug-drug interactions (DDIs)](https://en.wikipedia.org/wiki/Drug_interaction). PXR is therefore treated as a high-priority [ADMET](https://en.wikipedia.org/wiki/ADME) antitarget. We model its binding affinity to flag this liability early in drug design, before compounds advance to costly clinical trials. Our colleagues at [Octant Bio](https://www.octant.bio/) have collected the largest public PXR dataset to date (5x larger than what's in ChEMBL, and all from the same source institution), recently released as part of a [blind challenge](https://openadmet.ghost.io/predicting-pxr-induction-we-have-liftoff/). Crucially, the PXR pool compounds are drawn from an [**Enamine diversity deck**](https://enamine.net/compound-libraries/diversity-libraries/dds-10240) — a broad, structurally diverse compound collection selected to maximally cover chemical space, rather than to optimize a particular series.

Our second target, SARS-CoV-2 Mpro, is a proven antiviral drug target, with data from the [ASAP Discovery](https://asapdiscovery.org/) open-science consortium. Unlike the PXR diversity deck, the ASAP dataset consists of **real-world congeneric series**, tightly clustered chemical matter iterated by medicinal chemists to optimize potency.

These two targets represent contrasting scenarios. PXR tests efficient navigation of a **broad, structurally diverse** chemical space, while ASAP Mpro tests the **focused, congeneric** setting more common in practice. A key question is whether the same acquisition strategy excels in both regimes.

## The label bottleneck in drug discovery

In lead optimization, we typically work with hundreds to a few thousand compounds: a small fraction of chemical space. Assaying every candidate is expensive and slow, so *which* compounds we choose to test matters enormously: the composition of the training set shapes both what the model learns and where it generalizes.

Active learning formalizes this intuition. Rather than selecting compounds at random, the model identifies which untested candidates would be most informative to measure next. Our **goal** is to build an accurate activity model while minimizing the number of assays required — finding the most potent compounds and learning the structure-activity landscape as efficiently as possible.

We **query** from a large pool of unlabeled candidate compounds whose true activities are hidden until nominated for assay, exactly as in a real campaign. We also evaluate an optional foundation of external measurements to give the model a foothold before any pool labels arrive.

**How** the model decides what to query next is the central question of this post. At each iteration, an acquisition strategy scores every unlabeled candidate and selects a batch to assay. We compare six such strategies, ranging from pure **exploitation** of the model's predictions to pure **exploration** of uncertain or structurally novel regions, to understand the tradeoffs between finding actives quickly and learning a broadly accurate model. And, if possible, both.

## Practical considerations

Our benchmark reflects a realistic scenario: a folder of public legacy assay data, several plates of untested candidates, and a busy lab. The choices below aim to balance these practical constraints.

### Utilizing a foundation model and/or public data

Early iterations are the most precarious. With only a handful of labeled compounds, model predictions are noisy and acquisition strategies built on them are too. Rather than fixing a starting condition, we treat it as a variable in a 2×2 design crossing model initialization against historical data use. All four configurations are run for PXR; for Mpro, ChEMBL does not cover the ASAP series closely enough, so only the two no-ChEMBL configurations are evaluated.

1. **ChemProp (random init, no ChEMBL)** — a [ChemProp](https://github.com/chemprop/chemprop) message-passing neural network initialized with random weights and no external pretraining data. This is the true cold-start baseline: the model must learn everything it knows from the compounds queried during the campaign itself.  
     
2. [**CheMeleon**](https://github.com/JacksonBurns/chemeleon) **(no ChEMBL)** — the same MPNN architecture, but initialized from CheMeleon weights pretrained on millions of molecules. Pretraining instills broadly useful molecular representations that transfer well to novel tasks, giving the model a usable prior before any target-specific data arrives, without requiring any target-relevant historical measurements.  
     
3. **ChemProp \+ ChEMBL** — ChemProp with random initialization, but with training seeded by publicly available target-relevant measurements from [ChEMBL](https://www.ebi.ac.uk/chembl/) (~600 entries). This isolates the contribution of historical data independent of architectural pretraining.

4. **CheMeleon \+ ChEMBL** — CheMeleon weights further augmented by the same ChEMBL seed data. This mirrors the real-world scenario in which a practitioner begins a new project armed with both a pretrained backbone and whatever historical assay data is practically available.

Comparing these conditions disentangles the contributions of architectural pretraining and historical data, and assesses whether sourcing ChEMBL data is worth the additional setup cost.

### Evaluating generalization

To measure how well the model performs, we hold out a fixed test set before the campaign begins and never touch it during acquisition. The right way to construct that test set depends on the structure of the data.

**For PXR**, the pool is drawn from an Enamine diversity deck, a collection explicitly designed for maximal structural coverage. As a result, the dataset is not very self-similar: random, scaffold, and cluster splits yield comparable model performance, because the test compounds are no more structurally foreign to the training set than they would be under any other partitioning scheme. Given this, we use a simple **random 80/20 split**. The 80% becomes the candidate pool for the active learner; the 20% is the held-out evaluation benchmark used throughout all iterations.

**For ASAP Mpro**, the data was generated in chronological waves of medicinal chemistry iteration, so a meaningful temporal signal exists. We use the **predefined time split** provided with the dataset, which mirrors how the data would have been encountered in a real campaign: earlier compounds for training, later compounds for evaluation. This is a more realistic test of generalization: the model must predict activity for chemical matter synthesized *after* the training cutoff, capturing the true challenge of prospective prediction in drug discovery.

### Query batch size: matching the lab

Many academic demonstrations of active learning query one compound at a time — or at most batches of 10 or 20\. This is computationally convenient but experimentally unrealistic. In practice, dose-response assays are run on plates: a standard 1536-well microplate at roughly 13-point dose-response accommodates approximately 100 compounds per run (we'll “leave some room” for QC and controls). Querying fewer than a plate's worth of compounds per iteration would leave plates partially filled, reducing throughput and complicating scheduling.

We therefore query 100 compounds per iteration, i.e. one full plate's equivalent. Even this is conservative: most labs would prefer to fill multiple plates between model retraining cycles, particularly early in a campaign when assay infrastructure is under-utilized. The gap between the batch sizes used in AL benchmarks and what labs would actually run is worth acknowledging; results from single-compound or tiny-batch settings may not transfer directly to experimental practice.

## Query-by-committee: how ensemble disagreement guides exploration

Our committee consists of $N=5$ independent [**CheMeleon**](https://github.com/JacksonBurns/chemeleon) models, each with a different initialization ([deep ensembling](https://dl.acm.org/doi/10.5555/3295222.3295387)) and trained on a [*bootstrapped*](https://en.wikipedia.org/wiki/Bootstrapping_\(statistics\)) sample of the labeled set. Their disagreement at prediction time gives the acquisition function an indication of epistemic uncertainty. Each unlabeled molecule receives 5 predictions.

- The **mean** prediction ($\\mu$) is our best guess for activity.  
- The **standard deviation** ($\\sigma$) represents epistemic uncertainty.

Different acquisition strategies leverage these metrics:

- **Exploitation**: Greedy selection of the highest $\\mu$. Finds good compounds fast but can get stuck in local optima.  
- **Upper Confidence Bound (UCB)**: $\\mu \+ \\beta\\sigma$. Optimistically explores regions that *might* be high activity (we nominally select $\\beta=2$).  
- **Expected Improvement (EI)**: Balances $\\mu$ and $\\sigma$ to calculate the probability of exceeding the current best label $f^\*$ (equation below).  
- **Exploration**: Pure uncertainty sampling. Selects the $m$ compounds with the highest $\\sigma$, ignoring predicted activity. Useful as a baseline that maximizes coverage of predicted epistemic uncertainty.  
- **Diversity**: Ignores model predictions altogether and selects the $m$ compounds farthest from the current labeled set in 2D [generative topological map](https://en.wikipedia.org/wiki/Generative_topographic_map) (GTM) chemical space (max-min Euclidean distance), enforcing structural dissimilarity between batches.

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

Per iteration, we track number of active compounds "found" during the active learning campaign, MAE, Kendall's τ, and model uncertainty estimates.

## Hit discovery

The practical value of active learning is most directly measured by how quickly a campaign recovers actives. At 1.6%, the PXR diversity deck is sparse, leaving ample room for smart acquisition to outpace random sampling. The ASAP Mpro series, at 9.0%, is richer and provides a complementary scenario where even modest strategy advantages translate to large absolute hit count differences. Here we evaluate whether and to what degree different acquisition strategies accelerate active discovery, and whether trends hold across both dataset types.

### PXR

![][image3]  
*Figure 1\. PXR dataset cumulative number of active compounds (pEC50 ≥ 6.0) recovered as a function of labeled pool size for each acquisition strategy.* 

Across the PXR diversity deck (54 actives in a pool of 3312 compounds, a 1.6% hit rate), acquisition strategy has a dramatic effect on the rate of active discovery. **Exploitation** is the clear leader throughout the campaign, recovering approximately 70% of all pool actives by the time 27% of the pool has been labeled with ChemProp, compared to roughly 26% for **Random** sampling over the same interval. **UCB** is consistently the second-best strategy, tracking well above **Random** throughout. **Diversity** and **Random** perform nearly identically for hit-finding, while **Exploration** is the weakest strategy, often performing at or below **Random** since it directs queries toward uncertain regions rather than toward high predicted potency.

Model choice has a modest and perhaps surprising influence on hit-finding for PXR. ChemProp **Exploitation** recovers slightly more hits than CheMeleon **Exploitation** at the same labeled pool size (38 vs 32 actives at n = 900). To understand why, it helps to remember what **Exploitation** actually does. It ranks the unlabeled pool by predicted mean activity alone, with no role for ensemble uncertainty, and queries the top compounds from that list. The question is therefore which model places true actives higher in that ranking after training on the same initial data.

After fitting on the same 100 randomly selected compounds (training set maximum pEC50 approximately 6.46), ChemProp predicts a maximum activity of 8.48 on the unlabeled pool, more than 2 units above anything it was trained on. CheMeleon predicts a maximum of only 6.00, barely reaching the training set ceiling. CheMeleon’s pretrained encoder was trained on physicochemical properties across broad chemical space with no exposure to activity data, producing a smooth, well-structured latent geometry. When the output head is then trained on 100 PXR labels, settled encoder weights constrain how far predictions can extrapolate from that geometry. ChemProp, initialized randomly, faces no such constraint: its encoder and output head co-adapt simultaneously, distorting representations freely to accommodate the highest observed activities and extrapolating well beyond them on similar unseen compounds. The result is a hit-nonhit predicted activity gap of 0.64 units for ChemProp versus 0.44 units for CheMeleon at the first query, despite CheMeleon having higher global rank correlation (Spearman ρ ≈ 0.60 versus 0.53). CheMeleon is a better global ranker, but ChemProp’s aggressive extrapolation concentrates true hits at the very top of the list where **Exploitation** selects, placing approximately 9 true hits in its top-100 versus approximately 6 for CheMeleon. This first-query advantage compounds into a persistent cumulative gap. Tracking the hit-nonhit score gap across subsequent iterations confirms it does not grow as more actives enter the training set, ruling out any feedback from biased accumulation. The mechanism is fixed at initialization.

ChEMBL pretraining provides a meaningful head start in the very first iteration. Under **Exploitation**, both CheMeleon+ChEMBL and ChemProp+ChEMBL identify roughly 10 actives before any pool labels are acquired, compared to zero for their no-ChEMBL counterparts. This advantage comes entirely from a better starting model rather than from the acquisition function. However, it does not persist: warm-started trajectories converge to their no-ChEMBL counterparts within a few hundred pool labels, and cumulative hit counts are comparable from mid-campaign onward. ChEMBL pretraining accelerates early discovery but does not change the ceiling.

### SARS-CoV-2 Mpro

![][image4]  
*Figure 2\. ASAP SARS-Cov-2 Mpro dataset cumulative number of active compounds (pEC50 ≥ 7.0) recovered as a function of labeled pool size for each acquisition strategy.* 

The ASAP Mpro dataset shows the same qualitative ordering of strategies despite a markedly higher base hit rate (76 actives in a pool of 842 compounds, 9.0%). **Exploitation** again leads throughout the campaign, while **Random** trails substantially, and the full strategy ranking is preserved across both model types.

Model choice plays a different role on Mpro than on PXR. CheMeleon **Exploitation** recovers 55 of 76 pool actives by n = 200 versus 47 for ChemProp, a 17% difference from the same number of assays. **Random** finds only 17 actives at n = 200 regardless of model, confirming the benefit is specific to exploitation-driven selection. The mechanism mirrors PXR in reverse. At the very first Mpro query, CheMeleon already assigns a hit-nonhit gap of 1.35 units versus 0.93 for ChemProp. On a congeneric series, pretrained representations immediately encode the structural features associated with potency, while random initialization cannot resolve fine activity distinctions between nearly identical compounds from only 100 training examples. The larger initial gap translates directly into higher first-query precision and a cumulative hit-finding advantage that persists through the early campaign.

The same underlying mechanism explains both observations. **Exploitation** selects by predicted mean, so the model that assigns the most extreme values to true actives wins. On a diverse deck like PXR, a randomly initialized model extrapolates more aggressively from sparse data, outranking actives that a pretrained model conservatively scores near the observed range. On a congeneric series like Mpro, pretrained representations already resolve the subtle distinctions between close analogues from the outset. In both cases the pattern is established at initialization, before any exploitation-driven training bias can take effect. Strategy choice remains the dominant lever, with **Exploitation** consistently recovering two to three times as many hits as **Random** at mid-campaign, but model initialization determines which tool does it more efficiently.

## Model accuracy

Hit-finding efficiency and model quality are not the same objective. A campaign optimized purely for recovering actives concentrates labeled data in the high-potency region, which is exactly what a good QSAR model does not want. A practitioner deploying the resulting model for virtual screening or lead optimization guidance may therefore need a different labeling strategy than one focused on hit recovery. Here we ask whether acquisition strategy produces a meaningful difference in test-set predictive performance, or whether all strategies converge to similar quality regardless of how data was selected.

### PXR

![][image5]  
*Figure 3\. PXR dataset mean absolute error (MAE, pEC50 units) on the held-out random-split test set as a function of labeled pool size, for each of the six acquisition strategies. Shaded bands show ±1 SD across five random seeds.*

Acquisition strategy has a comparatively small effect on test-set MAE for PXR, though the magnitude depends substantially on the model used. For ChemProp, **Exploitation** reaches 0.64 pEC50 MAE at n = 900 compared to 0.57 for **Random**, a small but statistically significant gap of 0.07 units (p=0.0007) driven by the labeled set’s bias toward the active region. For CheMeleon, the same gap narrows to a even smaller, but still statistically significant, 0.02 units (0.57 vs 0.55, p=0.028), reflecting that pretrained representations help the model generalize better from a biased labeled pool. In absolute terms, both penalties remain modest and are far outweighed by **Exploitation**’s hit-finding advantage.

CheMeleon outperforms ChemProp across most strategies, though the magnitude and significance vary. The advantage is statistically significant under **Exploitation** (0.07 MAE units at n = 900, p=0.0006), but not under **EI** or **Diversity** (less than 0.01 units, p > 0.19 for both). When running an **Exploitation** campaign, CheMeleon therefore incurs a substantially smaller accuracy penalty than ChemProp, making it the preferred model backbone when both hit-finding speed and model quality are priorities.

ChEMBL pretraining provides a useful accuracy prior at campaign start. CheMeleon+ChEMBL achieves 0.83 pEC50 MAE before any pool labels are acquired, versus 0.93 for ChemProp+ChEMBL (p=0.0002). This advantage largely disappears by n = 200 pool labels (p=0.72 for CheMeleon+ChEMBL versus CheMeleon alone), as both warm-started configurations converge to their no-ChEMBL counterparts. The ChEMBL benefit is concentrated in the very first iterations.

![][image6]  
*Figure 4\. PXR dataset Kendall's τ rank-correlation between predicted and observed pEC50 on the held-out test set across active learning iterations. Higher values indicate better ranking of compounds by predicted activity. Shaded bands show ±1 SD across five random seeds.*

Kendall’s τ reinforces these conclusions. ChemProp reaches τ ≈ 0.49 and CheMeleon reaches τ ≈ 0.52 by mid-campaign, with strategy bands largely overlapping within each model. **Exploitation** produces the lowest τ for ChemProp (approximately 0.42 at n = 900, p=0.0006 vs **Random**), while CheMeleon strategies cluster tightly between 0.49 and 0.50. Skewing the labeled pool toward actives therefore hurts ranking ability much more for ChemProp than for CheMeleon, likely because pretrained representations already encode broad chemical variation and are less distorted by a narrow training distribution.

### SARS-CoV-2 Mpro

![][image7]  
*Figure 5\. ASAP SARS-CoV-2 Mpro dataset mean absolute error (MAE, pEC50 units) on the held-out random-split test set as a function of labeled pool size, for each of the six acquisition strategies. Shaded bands show ±1 SD across five random seeds.*

For ASAP Mpro, model initialization has a substantially larger effect on accuracy than acquisition strategy. At n = 200, CheMeleon reduces test-set MAE by 0.16 units under **Random** sampling (0.80 vs 0.96, p=0.0016) and by 0.07 units under **Exploitation** (0.73 vs 0.81), though the **Exploitation** gap does not reach significance across seeds (p=0.33), reflecting the high variance of both models under targeted early labeling on this series. The **Random** gap is robust because ChemProp, without pretrained weights, must build its representations from scratch from the most informationally dilute labeling strategy.

Within each model type, strategy-driven MAE differences are larger on Mpro than on PXR at small sample sizes. For ChemProp, the spread across strategies reaches 0.17 pIC50 units at n = 200, with **Exploitation** lowest and **Diversity** highest, narrowing to below 0.05 units by n = 600. For CheMeleon, the spread remains below 0.07 units throughout the campaign.

The CheMeleon accuracy advantage is substantially larger on Mpro (0.08 to 0.16 MAE units) than on PXR (0.01 to 0.07 units), as pretrained representations provide a larger benefit on a focused congeneric series where pretraining patterns are more directly applicable than on a diversity deck.

![][image8]  
*Figure 6\. ASAP SARS-CoV-2 Mpro dataset Kendall’s τ rank-correlation between predicted and observed pEC50 on the held-out test set across active learning iterations. Higher values indicate better ranking of compounds by predicted activity. Shaded bands show ±1 SD across five random seeds.*

Kendall’s τ at n = 400 ranges from 0.62 to 0.65 across strategies for CheMeleon and 0.58 to 0.62 for ChemProp. Unlike PXR, where the dominant τ signal was the intra-model spread driven by **Exploitation** bias, on Mpro the inter-model gap dominates: CheMeleon produces significantly higher τ than ChemProp under **UCB**, **Exploration**, and **Diversity** (p < 0.05 for each), with the remaining strategy comparisons borderline.

## Model uncertainty

Suppose the campaign deliverable is a model whose uncertainty estimates can be trusted, not just a hit list or an accurate regression. Such a model tells a medicinal chemist how much to rely on each prediction, which compounds warrant experimental follow-up because the model genuinely does not know their activity, and which can be deprioritized with confidence. This raises a distinct design question. Is there an acquisition strategy that produces better-ranked uncertainty estimates, and does it align with the strategy that maximizes hit recovery or minimizes MAE?

The strategies most relevant here are those that explicitly use σ in selection, specifically **EI**, **UCB**, and **Exploration**. Each creates a feedback loop between σ quality and compound selection, which could reinforce or degrade the model’s ability to rank its own ignorance. **Random** and **Diversity**, which ignore σ entirely, provide a baseline for whether uncertainty-agnostic labeling helps or hurts. We measure the Spearman rank correlation between σ and absolute prediction error on the held-out test set throughout each campaign. A high ρ means the committee correctly identifies which test compounds it is most wrong about. A low ρ means its expressed confidence is not a reliable guide.

### PXR

![][image9]  
*Figure 7\. PXR dataset Spearman rank correlation between predicted uncertainty (σ) and absolute prediction error (|ŷ − y|) on the held-out test set, as a function of labeled pool size, for each acquisition strategy. A positive ρ indicates that σ correctly ranks which test compounds the model is most wrong about. Shaded bands show ±1 SD across five random seeds.*

For PXR, the Spearman ρ between predicted uncertainty and absolute prediction error is consistently low across all strategies and model configurations, ranging from approximately 0.11 to 0.29 throughout the campaign. This indicates that the committee’s uncertainty estimates are a weak proxy for actual prediction error regardless of how the labeled pool is acquired.

Model choice markedly affects the level of ρ. ChemProp configurations produce ρ values of approximately 0.16 to 0.29, while CheMeleon configurations consistently yield lower values of 0.11 to 0.17. The difference is statistically significant under **Exploitation** and **UCB** (p=0.0005 and p=0.005 respectively), while the remaining strategies show overlapping distributions (p > 0.09). Despite CheMeleon’s substantial advantage in prediction accuracy, its uncertainty estimates are less correlated with actual errors. Because all CheMeleon ensemble members are initialized from the same pretrained weights, bootstrap resampling produces members that remain more similar to one another, compressing the spread in σ and reducing the signal available for ρ to detect.

The direction of the strategy effect also differs between models. For ChemProp, **Exploitation** produces the highest ρ (0.29 at n = 900, p=0.0043 vs **Random**), likely because the biased labeled set creates a sharp contrast between the confidently predicted, well-sampled active region and the uncertain, sparsely labeled inactive region. For CheMeleon, this pattern is reversed, with **Exploitation** producing the lowest ρ (0.11, p=0.027 vs **Random**) while **Random** and **Exploration** maintain modestly higher values.

### SARS-CoV-2 Mpro

![][image10]  
*Figure 7\. ASAP SARS-CoV-2 Mpro dataset Spearman rank correlation between predicted uncertainty (σ) and absolute prediction error (|ŷ − y|) on the held-out test set, as a function of labeled pool size, for each acquisition strategy. A positive ρ indicates that σ correctly ranks which test compounds the model is most wrong about. Shaded bands show ±1 SD across five random seeds.*

The ASAP Mpro dataset shows broadly similar uncertainty behavior, with ρ values ranging from approximately 0.13 to 0.34, modestly higher than PXR. On a congeneric series, training coverage and prediction difficulty are more tightly coupled. A compound the committee has not encountered anything like is simultaneously unfamiliar (high σ) and likely hard to predict (high error), because SAR within a focused series is smooth enough that distance from training data and prediction difficulty tend to co-vary. On the diverse PXR deck, compounds can be structurally well-covered yet behaviorally unpredictable due to activity cliffs and scaffold-specific SAR, decoupling σ from error. Higher ρ on Mpro is most plausibly explained by this difference in data structure rather than by a genuine improvement in uncertainty quality, though we cannot confirm this directly from our analysis.

Unlike PXR, ChemProp and CheMeleon produce comparable ρ on Mpro (both roughly 0.21 to 0.34 depending on strategy at n = 200, with no significant model differences), so the CheMeleon suppression of ρ observed on PXR does not appear on the congeneric series. **Random** and **Diversity** sampling tend to produce the highest ρ values on Mpro for both models, maintaining diverse labeled sets and avoiding the high-potency bias that suppresses ρ under **Exploitation**.

Across both targets, ensemble disagreement reflects training distribution coverage more than compound prediction difficulty, and no acquisition strategy reliably improves this. Each member’s disagreement on a compound reflects how differently it was exposed to that compound’s neighborhood in representation space, not whether the learned SAR is correct. Prediction error depends on SAR complexity, measurement noise, and training data coverage. These quantities measure fundamentally different things, and persistently low ρ is a consequence of using ensemble disagreement as a proxy for epistemic uncertainty rather than a traditional calibration failure.

Bootstrapped deep ensembles were chosen for practical reasons, requiring no architectural changes and having been widely validated in molecular property prediction. The tradeoff is that they measure diversity of initialization and data sampling rather than posterior uncertainty. Approaches better suited to calibrated error ranking include conformal prediction and Gaussian processes on learned representations. **EI** and **UCB** deliver their hit-finding gains by preferentially selecting high-variance compounds, not because σ is a calibrated error predictor. **Exploration**’s consistent failure to find hits despite querying high-σ compounds makes this limitation most apparent.

## Takeaways

**Use Exploitation.** It recovered two to three times as many hits as **Random** at mid-campaign on both PXR and Mpro, at the cost of a modest MAE penalty that is small in absolute terms and far outweighed by the hit-finding gain. If structural coverage rather than potency is the primary objective (e.g., building a broad SAR model before any lead is in hand), use **Diversity** instead. All other strategies, including **EI**, **UCB**, and **Exploration**, do not decisively outperform **Random** on either hit-finding or accuracy and add complexity without a clear payoff.

**Use CheMeleon.** On a congeneric series like Mpro it is the unambiguous choice, delivering 17% more hits under **Exploitation** at n=200 and 0.08 to 0.16 lower MAE. On a diverse deck like PXR, ChemProp's aggressive extrapolation gives it a slight hit-finding edge under **Exploitation** (38 vs 32 actives at n=900), but CheMeleon produces meaningfully more accurate models (0.07 lower MAE under **Exploitation**, p=0.0006) and better rank correlation across all strategies. Unless hit-finding on a diverse deck is the sole objective and model accuracy is irrelevant, CheMeleon is the better backbone.

**Skip ChEMBL unless the campaign will be very short.** The accuracy head-start is real (0.83 vs 0.93 MAE at n=0, p=0.0002), but it fully dissolves by n=200 pool labels (p=0.72). If even a modest initial screening set is affordable, ChEMBL warm-starting provides no lasting benefit.

**Do not rely on ensemble uncertainty for per-compound error ranking.** Spearman ρ between σ and absolute error stays in the 0.10 to 0.30 range regardless of acquisition strategy. The committee is useful for hit-finding through **Exploitation** and **UCB**, but its σ values should not be used to triage individual predictions. If calibrated per-compound uncertainty is a deliverable, consider conformal prediction instead.

## Reproducibility

To reproduce the results in this post, install `openadmet-models` by following the [installation instructions](https://docs.openadmet.org/en/latest/installation.html), clone the blogpost repo, then run:

```shell
# Execute the active learning pipeline (~several GPU-hours)
python run.py   

# Generate all figures from results/all_runs.pkl
python analysis.py  
```

All supporting code lives in `active-learning-blogpost/src/`: [src/helpers.py](http://src/helpers.py) contains the core AL utilities and [src/plots.py](http://src/plots.py) contains all Plotly/Faerun plotting functions. The campaign is governed by parameters specified in [`config.yaml`](http://config.yaml).  
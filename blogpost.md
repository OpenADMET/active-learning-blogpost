# Model, take the wheel: active learning with pEC50 data

By Sean Colby

**In computational drug discovery, data is the most valuable resource, yet synthesizing and assaying even a handful of compounds can cost thousands of dollars and take several weeks. As more effort turns toward acquiring data specifically to train machine learning models, including our work at OpenADMET, a natural question arises: is there an intelligent way to use what a model already knows to guide an assay campaign toward its goal?**

[**Active learning (AL)**](https://en.wikipedia.org/wiki/Active_learning_\(machine_learning\)) flips the traditional machine learning paradigm. Instead of working with static training and test sets, the model iteratively selects the compounds by **acquisition strategy**, ranging from “**exploration”**, selecting compounds it is most uncertain about, to “**exploitation”**, selecting compounds it finds most promising (**“exploitation”**). We leverage what the model has learned to guide the collection of high-impact data points, amplifying model improvement and hit-finding per dollar spent.

In this post, we build an active learning loop using [`openadmet-models`](https://github.com/OpenADMET/openadmet-models), powered by [ChemProp](https://chemprop.readthedocs.io/en/latest/) and the [CheMeleon](https://github.com/JacksonBurns/chemeleon) foundation model, to simulate two campaigns, one targeting the [Pregnane X Receptor (PXR)](https://en.wikipedia.org/wiki/Pregnane_X_receptor) and one targeting the [main protease](https://en.wikipedia.org/wiki/3C-like_protease) (Mpro) of [SARS-CoV-2](https://en.wikipedia.org/wiki/SARS-CoV-2). The former is from a diverse hit screening library, the latter is in the hit-to-lead phase of candidate optimization. We compare six acquisition strategies and assess their efficacy in predictive accuracy and hit-finding.

## Selected targets

PXR is a ligand-activated nuclear transcription factor that serves as the body's primary xenobiotic sensor and is highly expressed in the liver and intestines. Its uniquely large, flexible, and hydrophobic ligand-binding pocket makes it notoriously promiscuous, capable of accommodating a wide variety of chemical scaffolds. When a molecule binds PXR, it can induce  [CYP3A4](https://en.wikipedia.org/wiki/CYP3A4) and related drug-metabolizing enzymes, accelerating the metabolic clearance of co-administered therapies and potentially causing [drug-drug interactions (DDIs)](https://en.wikipedia.org/wiki/Drug_interaction). PXR is therefore treated as a high-priority [ADMET](https://en.wikipedia.org/wiki/ADME) antitarget. We model its activity to flag this liability early in a drug discovery project, before compounds advance to more costly preclinical studies. Our colleagues at [Octant Bio](https://www.octant.bio/) have generated the largest public PXR dataset to date (5x larger than what's in [ChEMBL](https://www.ebi.ac.uk/chembl/) and, unlike ChEMBL, all from the same experimental protocol ), recently released as part of the OpenADMET [PXR Blind Challenge](https://openadmet.ghost.io/predicting-pxr-induction-we-have-liftoff/). Crucially, the PXR compound pool is drawn from an [**Enamine diversity deck**](https://enamine.net/compound-libraries/diversity-libraries/dds-10240): a broad, structurally diverse compound collection selected to maximally cover chemical space, rather than optimized from a particular series.

Our second target, SARS-CoV-2 Mpro, is a proven antiviral drug target, with data from the [ASAP Discovery](https://asapdiscovery.org/) open-science consortium. Unlike the PXR diversity deck, the ASAP dataset consists of a **real-world congeneric series**, tightly clustered chemical matter iteratively optimized by medicinal chemists towards the relevant [target candidate profile](https://wcair.dundee.ac.uk/training/training-resources/drug-discovery-process/target-product-profiles-and-target-candidate-profiles/).

These two targets represent contrasting scenarios. PXR tests efficient navigation of a **broad, structurally diverse** chemical space, while ASAP Mpro tests the **focused, congeneric** setting more commonly seen in later stages of drug discovery projects. A key question is whether the same acquisition strategies excel in both regimes, or if different strategies are preferable for inherently diverse libraries (PXR) versus congeneric series (ASAP Mpro)?

## The label bottleneck in drug discovery

In lead optimization, we typically work with hundreds to a few thousand compounds. Assaying every candidate is expensive and slow, so *which* compounds we choose to test matters enormously. The composition of the training set shapes both what the model learns and where it generalizes.

Active learning formalizes this intuition. Rather than selecting compounds at random, the model identifies which untested candidates would be most informative to measure next. **Our goal is to build an accurate activity model while minimizing the number of assay datapoints required**, finding the most potent compounds and learning the structure-activity landscape as efficiently as possible.

In this study, we **query** a large pool of unlabeled candidate compounds whose true activities are hidden until they are nominated for assay, exactly as in a real campaign. We also evaluate an optional foundation of preliminary training and/or external measurements to give the model a foothold before any pool labels arrive.

**How** the model decides what to query next is the central question of this post. At each iteration, an acquisition strategy scores every unlabeled candidate and selects a batch to assay. We compare six such strategies, ranging from pure **exploitation** of the model's predictions to pure **exploration** of uncertain or structurally novel regions, to understand the trade-offs between quickly finding actives and learning a broadly accurate model. Ideally, if possible, we would select a strategy that achieves both. 

## Practical considerations

Our benchmark attempts to reflect a realistic scenario: a folder of public legacy assay data, several plates of untested candidates, and a busy lab. The choices below aim to balance these practical constraints.

### Utilizing a foundation model and/or public data

Early iterations of active learning are the most precarious. With only a handful of labeled compounds, model predictions are noisy, and acquisition strategies built on them are too. Rather than fixing a starting condition, we vary model initialization and the use of pretraining data: ChemProp versus CheMeleon, and ChEMBL pretraining versus no pretraining (no ChEMBL). All four configurations are run for PXR; for ASAP Mpro, data leakage concerns arise from overlapping compounds, so only the two no-ChEMBL configurations are evaluated.

1. **ChemProp (random init, no ChEMBL)**: a ChemProp message-passing neural network initialized with random weights and no external pretraining data. This is the true cold-start baseline: the model must learn everything it knows from the compounds queried during the campaign itself.  
     
2. **CheMeleon** **(no ChEMBL)**: the same MPNN architecture, but initialized from CheMeleon weights pretrained on millions of molecules. Pretraining instills useful molecular representations that transfer to novel tasks, giving the model a usable prior before any target-specific data arrives and without requiring any target-relevant historical measurements.  
     
3. **ChemProp \+ ChEMBL**: ChemProp with random initialization, but with training seeded by publicly available target-relevant measurements from ChEMBL (\~600 entries for PXR). This isolates the contribution of historical data independent of architectural pretraining.  
     
4. **CheMeleon \+ ChEMBL**: CheMeleon weights are further augmented by the same ChEMBL seed data. This mirrors the real-world scenario in which a practitioner begins a new project armed with both a pretrained backbone and whatever historical assay data is available.

Comparing these conditions disentangles the contributions of architectural pretraining and historical data, and assesses whether sourcing ChEMBL data is worth the additional setup cost.

### Evaluating generalization

To measure how well each model and selection strategy performs in the active learning simulation, we hold out a fixed test set before the campaign begins and do not touch it during acquisition. The correct way to construct that test set depends on the structure of the data.

**For PXR**, the pool is drawn from an Enamine diversity deck, a collection explicitly designed for maximal structural coverage. As a result, the dataset is not very self-similar: random, scaffold, and cluster splits yield comparable model performance, because the test compounds are no more structurally foreign to the training set than they would be under any other partitioning scheme. That is, [Bemis-Murcko](https://doi.org/10.1021/jm9602928) scaffolding produces 96.1% singletons, and [Taylor-Butina](https://doi.org/10.1021/ci9803381) clustering (1024-bit, radius 2 [Morgan fingerprints](https://doi.org/10.1021/ci100050t), Tanimoto distance threshold 0.35) gives 98.6% singletons. Behavior is completely indistinguishable from random at 100% singletons. Given this, we use a simple **random 80/20 split**. The 80% forms the candidate pool for the active learner; the remaining 20% serves as the held-out evaluation benchmark used across all iterations.

**For ASAP Mpro**, the data was generated in chronological waves of medicinal chemistry iteration, so a meaningful temporal signal exists. We use the **predefined time split** provided with the dataset from the ASAP-Polaris-OpenADMET Challenge, which mirrors how the data would have been encountered in a real campaign: earlier compounds for training, later compounds for evaluation. This is a more realistic test of generalization: the model must predict activity for chemical matter synthesized *after* the training cutoff, capturing the true challenge of prospective prediction in drug discovery.

### Query batch size: matching the lab

Many academic demonstrations of active learning query [one](https://doi.org/10.3762/bjoc.20.185) [compound](https://doi.org/10.1016/j.drudis.2014.12.004) [at](https://doi.org/10.1016/j.ddtec.2020.06.001) [a](https://doi.org/10.1021/ci3004682) [time](https://doi.org/10.1287/ijoc.1100.0417), though some [recent](https://doi.org/10.1038/s43588-024-00697-2) [work](https://www.doi.org/10.1021/acs.jcim.2c00602) uses more realistic batch sizes. One-at-a-time selection is computationally convenient and theoretically most performant, but experimentally unrealistic. In practice, dose-response assays are run on plates: a standard 1536-well plate at roughly 13-point dose-response accommodates approximately 100 compounds per run (we'll “leave some room” for QC and controls). Querying fewer than a plate's worth of compounds per iteration would leave plates partially filled, reducing throughput and complicating scheduling.

We therefore query 100 compounds per iteration, i.e., the equivalent of one full plate. Even this is conservative: most labs (including our friends at Octant) would prefer to fill multiple plates between model retraining cycles, particularly early in a campaign when assay infrastructure is underutilized. The gap between the batch sizes used in AL benchmarks and those labs would actually run is worth acknowledging; results from single-compound or tiny-batch settings may not transfer directly to experimental practice.

### Query-by-committee: how ensemble disagreement guides exploration

Our committee consists of N=5 independent models, each with a different initialization ([deep ensembling](https://dl.acm.org/doi/10.5555/3295222.3295387)) and trained on a [*bootstrapped*](https://en.wikipedia.org/wiki/Bootstrapping_\(statistics\)) sample of the labeled set. Their disagreement at prediction time gives the acquisition function an indication of epistemic uncertainty. Each unlabeled molecule receives 5 predictions that we operationalize as:

- The **mean** prediction (μ) is our best guess for activity.  
- The **standard deviation** (σ) represents epistemic uncertainty.

Different acquisition strategies leverage these metrics (see [this resource](https://arxiv.org/abs/1012.2599) for more information):

- **Exploitation**: Greedy selection of the highest μ . Finds good compounds fast but can get stuck in local optima.  
- **Upper Confidence Bound (UCB)**: μ \+ βσ. Optimistically explores regions that *might* have high activity (we nominally select β=2, a common choice).  
- **Expected Improvement (EI)**: Balances μ and σ to calculate the probability of exceeding the current best label *f\** (equation below).  
- **Exploration**: Pure uncertainty sampling. Selects the *m* compounds with the highest μ, ignoring predicted activity. Useful as a baseline that maximizes coverage of predicted epistemic uncertainty.  
- **Diversity**: Ignores model predictions altogether and selects the *m* compounds farthest from the current labeled set in a metric space. Here, a 2D [generative topological map](https://en.wikipedia.org/wiki/Generative_topographic_map) (GTM) calculated from RDKit descriptors is used (max-min Euclidean distance), enforcing physicochemical property dissimilarity between batches.

Note that uncertainty calibration, discussed in a previous post [Concerning Uncertainty](https://openadmet.ghost.io/concerning-uncertainty/), does not affect acquisition: **Exploration**, **EI**, and **UCB** rank candidates by σ, and a global scaling factor preserves that ordering.

## The active learning loop

In each active learning iteration, *k*, we perform the following steps:

1. Train the committee on the current labeled pool.  
2. Evaluate on the static test set to track performance.  
3. Use the committee to predict on the unlabeled pool.  
4. Score the unlabeled molecules with the acquisition function.  
5. "Acquire" the top *m* molecules (reveal their labels).  
6. Add them to the labeled pool and repeat.

Per iteration, we track the number of active compounds "found" during the active learning campaign, as well as MAE, Kendall's τ, and model uncertainty estimates. **Figure 1** interactively visualizes this process for the ASAP Mpro dataset using the **Exploitation** strategy with CheMeleon+ChEMBL.
 
*Figure 1\. Compound selections on the ASAP Mpro dataset embedded in two-dimensional GTM space, shown iteration by iteration for the **Exploitation** strategy with CheMeleon+ChEMBL. Each point is one pool compound projected onto a topographic map of chemical space learned from RDKit physicochemical descriptors. Use the play/pause button to control the animation, or use the slider to manually step through iterations.*

## Hit discovery

The practical value of active learning is most directly measured by how quickly a campaign recovers actives. At only 1.6% active compounds (≥ pEC50 6.0, representative of an early-stage hit-finding threshold), the PXR diversity deck is sparse, leaving ample room for smart acquisition to outpace random sampling. The ASAP Mpro series, at 9.0% active compounds (≥ pEC50 7.0, representative of a hit-to-lead optimization threshold), is richer and provides a complementary scenario. Here, we evaluate whether and to what extent different acquisition strategies accelerate active discovery, and whether trends hold across dataset types and model initializations.

### PXR
 
*Figure 2\. PXR dataset cumulative number of active compounds (pEC50 ≥ 6.0) recovered as a function of labeled pool size for each acquisition strategy.* 

Across the PXR diversity deck (54 compounds with pEC50 ≥ 6.0 in a pool of 3312 compounds, a 1.6% hit rate), choice of acquisition strategy has a dramatic effect on the rate of active discovery (**Figure 2**). **Exploitation** is the clear leader throughout the campaign, recovering approximately 70% of all pool actives by the time 27% of the pool has been labeled with ChemProp, compared to roughly 26% for **Random** sampling over the same interval. **UCB** is consistently the second-best strategy, tracking well above **Random** throughout. **Diversity** and **Random** perform nearly identically for hit-finding, while **Exploration** is the weakest strategy, often performing at or below **Random** since it directs queries toward uncertain regions rather than toward high predicted potency.

Model choice has a modest and perhaps surprising influence on hit-finding for PXR. ChemProp **Exploitation** recovers slightly more hits than CheMeleon **Exploitation** at the same labeled pool size (38 vs 32 actives at n = 900). To understand why, it helps to remember what **Exploitation** actually does. It ranks the unlabeled pool by predicted mean activity alone, with no role for ensemble uncertainty, and queries the top compounds from that list. The question is therefore which model places true actives higher in that ranking after training on the same initial data.

*Figure 3\. PXR dataset predicted pEC50 KDE for hits (pEC50 ≥ 6.0) and non-hits on the unlabeled pool under the **Exploitation** strategy, animated across active learning iterations (5 seeds). ChemProp (top, positive y) and CheMeleon (bottom, negative y) share a common density axis. The annotated gap is the difference between the mean predicted activity of hits and non-hits; a larger gap indicates that the model concentrates true actives at the top of the ranked list from which **Exploitation** selects.*

After fitting on the same 100 randomly selected compounds (training set maximum pEC50 \~6.46), ChemProp predicts a maximum activity of 8.48 on the unlabeled pool, more than 2 units above anything it was trained on. CheMeleon predicts a maximum of only 6.00, barely reaching the training set ceiling. CheMeleon’s pretrained encoder was trained on physicochemical properties across broad chemical space without exposure to activity data, yielding a smooth, well-structured latent geometry. When the output head is then trained on 100 PXR labels, settled encoder weights constrain how far predictions can extrapolate from that geometry. ChemProp, initialized randomly, faces no such constraint: its encoder and output head co-adapt simultaneously, freely distorting representations to accommodate the highest observed activities and extrapolating well beyond them to similar unseen compounds. The result is a hit/non-hit predicted activity gap of 0.64 pEC50 units for ChemProp versus 0.44 pEC50 units for CheMeleon at the first query (**Figure 3**), despite CheMeleon having a higher global rank correlation (Spearman ρ ≈ 0.60 versus 0.53). CheMeleon is a better global ranker, but ChemProp’s more aggressive extrapolation concentrates true hits at the very top of the list, where **Exploitation** selects, placing approximately 9 true hits in its top-100 versus approximately 6 for CheMeleon. This first-query advantage compounds into a persistent cumulative gap. Tracking the hit/non-hit score gap across subsequent iterations confirms that it does not grow as more actives are selected, ruling out any feedback from accumulation biased towards actives.

ChEMBL pretraining provides a meaningful head start in the very first iteration. Under **Exploitation**, both CheMeleon+ChEMBL and ChemProp+ChEMBL identify roughly 10 actives before any pool labels are acquired, compared to zero for their no-ChEMBL counterparts. This advantage stems directly from the ChEMBL warm-start combined with exploitation-driven selection, giving CheMeleon a slight edge over ChemProp. However, it does not persist: warm-started trajectories converge to their no-ChEMBL counterparts within a few hundred pool labels, and cumulative hit counts are comparable from mid-campaign onward. ChEMBL pretraining accelerates early discovery but does not change the ceiling.

### ASAP SARS-CoV-2 Mpro

*Figure 4\. ASAP SARS-Cov-2 Mpro dataset cumulative number of active compounds (pEC50 ≥ 7.0) recovered as a function of labeled pool size for each acquisition strategy.* 

The ASAP Mpro dataset shows the same qualitative ordering of strategies despite a markedly higher base hit rate (**Figure 4**; 76 actives in a pool of 842 compounds, 9.0%). **Exploitation** again leads throughout the campaign, while **Random** trails substantially, and the full strategy ranking is preserved across both model types.
 
*Figure 5\. ASAP Mpro dataset predicted pEC50 KDE for hits (pEC50 ≥ 7.0) and non-hits on the unlabeled pool under the **Exploitation** strategy, animated across active learning iterations (5 seeds). ChemProp (top, positive y) and CheMeleon (bottom, negative y) share a common density axis. The annotated gap is the difference between the mean predicted activity of hits and non-hits; a larger gap indicates that the model concentrates true actives at the top of the ranked list from which **Exploitation** selects.*

Model choice plays a different role on ASAP Mpro than on PXR. CheMeleon **Exploitation** recovers 55 of 76 pool actives by n = 200 versus 47 for ChemProp, a 17% difference from the same number of evaluations. **Random** finds only 17 actives at n = 200 regardless of model, confirming the active-finding benefit is specific to exploitation-driven selection. The mechanism is opposite to that of PXR: at the very first ASAP Mpro query, CheMeleon already assigns a hit/non-hit gap of 1.35 versus 0.93 pEC50 units for ChemProp (**Figure 5**). The larger initial gap translates directly into higher first-query precision and a cumulative advantage in hit-finding that persists throughout the early campaign.

The same underlying mechanism explains both observations. **Exploitation** selects based on the predicted mean, so the model that assigns the most extreme values to true actives wins. On a diverse compound library like PXR, a randomly initialized model extrapolates more aggressively from sparse data, outranking actives that a pretrained model conservatively scores near the observed range. On a congeneric series like Mpro, pretrained representations already capture the subtle distinctions between close analogs from the outset. In both cases, the pattern is established at initialization, before any exploitation-driven training bias can take effect. Strategy choice remains the dominant lever, with **Exploitation** consistently recovering two to three times as many hits as **Random** at mid-campaign, but model initialization (ChemProp versus CheMeleon, ChEMBL versus no ChEMBL) determines which configuration does it more efficiently.

## Model accuracy

Hit-finding efficiency and resulting model quality are not the same objective. A campaign optimized purely for recovering actives concentrates labeled data in the high-potency region, which is ostensibly what a good QSAR model *should* avoid. A practitioner deploying the resulting model for virtual screening or lead optimization guidance may therefore need a different labeling strategy than one focused on hit recovery. Here, we ask whether the acquisition strategy produces a meaningful difference in test-set predictive performance, or whether all strategies converge to similar quality regardless of how the data was selected?

### PXR
 
*Figure 6\. PXR dataset Mean Absolute Error (MAE, pEC50 units) on the held-out random-split test set as a function of labeled pool size, for each of the six acquisition strategies. Shaded bands show ±1 SD across five random seeds.*

The choice of acquisition strategy has a comparatively small effect on test-set MAE for PXR, though the magnitude depends substantially on the model used (**Figure 6**). For ChemProp, **Exploitation** reaches 0.64 pEC50 MAE at n = 900 compared to 0.57 for **Random**, a small but statistically significant gap of 0.07 (p=0.0007) driven by the labeled set’s bias toward the active region. For CheMeleon, the same gap narrows to an even smaller, but still statistically significant, 0.02 (0.57 vs 0.55, p=0.028). In absolute terms, both penalties remain modest and are outweighed by **Exploitation**’s hit-finding advantage.

CheMeleon outperforms ChemProp across most strategies, though the magnitude and significance vary. The advantage is statistically significant under **Exploitation** (0.07 MAE at n = 900, p=0.0006), but not under **EI** or **Diversity** (less than 0.01, p \> 0.19 for both). When running an **Exploitation** campaign, CheMeleon therefore incurs a substantially smaller accuracy penalty than ChemProp, making it the preferred model backbone when both hit-finding speed and model quality are priorities.

ChEMBL pretraining provides a useful accuracy prior at campaign start. CheMeleon+ChEMBL achieves 0.83 pEC50 MAE before any pool labels are acquired, versus 0.93 for ChemProp+ChEMBL (p=0.0002). This advantage largely disappears by n = 200 pool labels (p=0.72 for CheMeleon+ChEMBL versus CheMeleon alone), as both warm-started configurations converge to their no-ChEMBL counterparts. The ChEMBL benefit is concentrated in the very first iterations.  

*Figure 7\. PXR dataset Kendall's τ rank-correlation between predicted and observed pEC50 on the held-out test set across active learning iterations. Higher values indicate better ranking of compounds by predicted activity. Shaded bands show ±1 SD across five random seeds.*

Kendall’s τ reinforces these conclusions (**Figure 7**). ChemProp reaches τ ≈ 0.49, and CheMeleon reaches τ ≈ 0.52 by mid-campaign, with strategy bands largely overlapping within each model. **Exploitation** produces the lowest τ for ChemProp (approximately 0.42 at n = 900, p=0.0006 vs **Random**), while CheMeleon strategies are more tightly clustered (0.49 to 0.50). 

### ASAP SARS-CoV-2 Mpro

*Figure 8\. ASAP SARS-CoV-2 Mpro dataset Mean Absolute Error (MAE, pEC50 units) on the held-out random-split test set as a function of labeled pool size, for each of the six acquisition strategies. Shaded bands show ±1 SD across five random seeds.*

For ASAP Mpro, model initialization has a substantially larger effect on accuracy than the acquisition strategy does (**Figure 8**). At n = 200, CheMeleon reduces test-set MAE by 0.16 under **Random** sampling (0.80 vs 0.96, p=0.0016) and by 0.07 under **Exploitation** (0.73 vs 0.81), though the **Exploitation** gap does not reach significance across seeds (p=0.33), reflecting the high variance of both models under targeted early labeling on this series. The **Random** gap is the cleanest measure of the model effect, because both models receive identical labeled sets under random selection (compound choice does not depend on model predictions), so any performance difference is attributable to model architecture alone, with no confounding from strategy-model interaction.

Within each model type, strategy-driven MAE differences are larger on ASAP Mpro than on PXR at small sample sizes. For ChemProp, the spread across strategies reaches 0.17 pEC50 units at n = 200, with **Exploitation** lowest and **Diversity** highest, narrowing to below 0.05 by n = 600. For CheMeleon, the spread remains below 0.07 throughout the campaign.

The CheMeleon accuracy advantage is substantially larger on Mpro (ΔMAE ranging from 0.08 to 0.16) than on PXR (ΔMAE ranging from 0.01 to 0.07), as pretrained representations provide a larger benefit on a focused congeneric series where pretraining patterns are more directly applicable than on a diversity deck.  

*Figure 9\. ASAP SARS-CoV-2 Mpro dataset Kendall's τ rank-correlation between predicted and observed pEC50 on the held-out test set across active learning iterations. Higher values indicate better ranking of compounds by predicted activity. Shaded bands show ±1 SD across five random seeds.*

Kendall’s τ at n = 400 ranges from 0.62 to 0.65 across strategies for CheMeleon and 0.58 to 0.62 for ChemProp (**Figure 9**). Unlike PXR, where the dominant τ signal was the intra-model spread driven by **Exploitation** bias, on Mpro the inter-model gap dominates: CheMeleon produces significantly higher τ than ChemProp under **UCB**, **Exploration**, and **Diversity** (p \< 0.05 for each), with the remaining strategy comparisons borderline.

In summary, **Exploitation** incurs a small hit to model accuracy on PXR, and this deficit persists throughout the campaign. It is resolved almost entirely by use of CheMeleon, which narrows the **Exploitation** penalty from 0.07 to 0.02 pEC50 units. On ASAP Mpro, there is no such accuracy penalty for **Exploitation**, but the CheMeleon benefit is still clear: MAE is reduced by \~0.20 by the end of the campaign. ChEMBL pretraining helps in early iterations, but the benefit is lost by mid-campaign.

## Model uncertainty

Suppose the campaign deliverable is a model whose uncertainty estimates can be trusted, not just a hit list or an accurate regression. Such a model tells a medicinal chemist how much to rely on each prediction, which compounds warrant experimental follow-up because the model genuinely does not know their activity, and which can be deprioritized with confidence. This raises a distinct design question: is there an acquisition strategy that produces better-ranked uncertainty estimates, and does it align with the strategy that maximizes hit recovery or minimizes MAE? In other words, can active learning help us be more confident in our predictions?

The strategies most relevant here are those that explicitly use σ in selection, specifically **EI**, **UCB**, and **Exploration**. Each creates a feedback loop between σ quality and compound selection, which could reinforce or degrade the model’s ability to rank its own ignorance. **Random** and **Diversity**, which ignore σ entirely, provide a baseline for whether uncertainty-agnostic labeling helps or hurts. 

To measure this, we calculate the Spearman rank correlation (ρ) between σ and absolute prediction error (residuals) on the held-out test set throughout each campaign. A high ρ means the committee correctly identifies which test compounds it is most wrong about. A low ρ means its expressed confidence is not a reliable guide.

### PXR
 
*Figure 10\. PXR dataset Spearman rank correlation between predicted uncertainty (σ) and absolute prediction error (|ŷ − y|) on the held-out test set, as a function of labeled pool size, for each acquisition strategy. A positive ρ indicates that σ correctly ranks which test compounds the model is most wrong about. Shaded bands show ±1 SD across five random seeds.*

For PXR, the Spearman ρ between predicted uncertainty and absolute prediction error is consistently low across all strategies and model configurations, ranging from approximately 0.11 to 0.29 throughout the campaign (**Figure 10**). This indicates that the **committee’s uncertainty estimates are a weak proxy for actual prediction error,** regardless of how the labeled pool is acquired. In other words, the ensemble cannot provide a good estimate of its own uncertainty. 

Model choice markedly affects the level of ρ. ChemProp produces ρ values of approximately 0.16 to 0.29, while CheMeleon consistently yields lower values of 0.11 to 0.17. Comparing ChemProp against CheMeleon at the same strategy, the gap is statistically significant under **Exploitation** and **UCB** (p=0.0005 and p=0.005, respectively), while ChemProp and CheMeleon ρ distributions overlap under the remaining four strategies (p \> 0.09 for each). Despite CheMeleon’s substantial advantage in prediction accuracy, its uncertainty estimates are less correlated with actual errors. **Because all** **CheMeleon ensemble members are initialized from the same pretrained weights**, bootstrap resampling produces members that remain more similar to one another, compressing the spread in σ and reducing the signal available for ρ to detect.

### ASAP SARS-CoV-2 Mpro

*Figure 7\. ASAP SARS-CoV-2 Mpro dataset Spearman rank correlation between predicted uncertainty (σ) and absolute prediction error (|ŷ − y|) on the held-out test set, as a function of labeled pool size, for each acquisition strategy. A positive ρ indicates that σ correctly ranks which test compounds the model is most wrong about. Shaded bands show ±1 SD across five random seeds.*

The ASAP Mpro dataset shows broadly similar uncertainty behavior, with ρ values ranging from approximately 0.13 to 0.34, slightly higher than for PXR. On a congeneric series, training coverage and prediction difficulty are more tightly coupled. A structurally or physicochemically novel compound is simultaneously unfamiliar (high σ) and hard to predict (high error) because  distance from the training data and prediction difficulty tend to covary. On the diverse PXR deck, compounds can be structurally well-covered, yet behaviorally unpredictable due to activity cliffs and scaffold-specific SAR, decoupling σ from error. Higher ρ on Mpro is most plausibly explained by this difference in data structure rather than by a genuine improvement in the quality of uncertainty estimates, though we cannot confirm this directly from our analysis.

Unlike PXR, ChemProp and CheMeleon produce comparable ρ on Mpro (both roughly 0.21 to 0.34 depending on strategy at n = 200, with no significant model differences), so the CheMeleon suppression of ρ observed on PXR does not appear on the congeneric series. **Random** and **Diversity** sampling tend to produce the highest ρ values on Mpro for both models, maintaining diverse labeled sets and avoiding the high-potency bias that suppresses ρ under **Exploitation**.

Across both targets, ensemble disagreement reflects training distribution coverage more than compound prediction difficulty, and no acquisition strategy reliably improves it. Each member’s disagreement on a compound reflects how differently it was exposed to that compound’s neighborhood in representation space, not whether the learned SAR is correct. Prediction error depends on SAR complexity, measurement noise, and training data coverage. These quantities measure fundamentally different things, and persistently low ρ is a consequence of using ensemble disagreement as a proxy for epistemic uncertainty rather than a traditional calibration failure.

Bootstrapped [deep ensembles](https://papers.nips.cc/paper/7219-simple-and-scalable-predictive-uncertainty-estimation-using-deep-ensembles) were chosen for practical reasons: they require no architectural changes and have [been](https://doi.org/10.1038/s41598-025-27167-7) [widely](https://doi.org/10.1021/acs.jcim.5c02381) [validated](https://doi.org/10.1021/acs.jcim.5c01597) in molecular property prediction. The tradeoff is that they measure diversity of initialization and data sampling rather than a more statistically rigorous uncertainty estimate. Approaches better suited to calibrated error ranking include [conformal prediction](https://en.wikipedia.org/wiki/Conformal_prediction) and [Gaussian processes](https://en.wikipedia.org/wiki/Gaussian_process) on learned representations.

## Takeaways

**Use Exploitation.** It recovered two to three times as many hits as **Random** at mid-campaign on both PXR and Mpro, at the cost of a modest MAE penalty that is small in absolute terms and far outweighed by the hit-finding gain. If structural coverage rather than potency is the primary objective (e.g., building a broad SAR model before any lead is in hand), use **Diversity** instead. All other strategies, including **EI**, **UCB**, and **Exploration**, do not decisively outperform **Random** on either hit-finding or accuracy and add complexity without a clear payoff.

**Use CheMeleon or another pretrained model.** For a congeneric series like ASAP Mpro, it is the unambiguous choice, delivering 17% more hits under **Exploitation** at n=200 and 0.08 to 0.16 lower MAE. On a diverse deck like PXR, ChemProp's aggressive extrapolation gives it a slight hit-finding edge under **Exploitation** (38 vs 32 actives at n=900), but CheMeleon produces meaningfully more accurate models (0.07 lower MAE under **Exploitation**, p=0.0006) and better rank correlation across all strategies. Unless hit-finding on a diverse deck is the sole objective, and model accuracy is irrelevant, CheMeleon is the better backbone.

**Skip ChEMBL.** Unless the campaign is very short. The accuracy head-start is real (0.83 vs 0.93 MAE at n=0, p=0.0002), but it fully dissolves by n=200 pool labels (p=0.72). If even a modest initial screening set is affordable, ChEMBL warm-starting provides no lasting benefit.

**Do not rely on ensemble uncertainty.** At least not for the uncertainty surrogate (bootstrapped deep ensembles of ChemProp / CheMeleon) used in this analysis. Spearman ρ between σ and absolute error stays in the 0.10 to 0.30 range regardless of acquisition strategy. We will likely train single models moving forward, rather than ensembles, until we devise an improved strategy for modeling uncertainty.

## Reproducibility

All supporting code lives in `active-learning-blogpost/src/`: `src/helpers.py` contains the core AL utilities, and `src/plots.py` contains all plotting functions. The campaign is governed by parameters specified in `config.yaml`.

To reproduce the results in this post, install `openadmet-models` by following the [installation instructions](https://docs.openadmet.org/en/latest/installation.html), clone the blogpost repo, then run:

```shell
# Execute the active learning pipeline (~several GPU-hours)
python run.py   

# Generate all figures from results/*.pkl
python analysis.py  
```

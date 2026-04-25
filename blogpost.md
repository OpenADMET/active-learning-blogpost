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

In drug discovery, the practical value of active learning is most directly measured by how quickly a campaign recovers active compounds from a screening collection. A model that identifies potent molecules earlier reduces the number of expensive assays needed before a meaningful hit list is assembled. How much leverage an active strategy has depends heavily on the density of hits in the pool. At a 1.6% hit rate, the PXR diversity deck is sparse and leaves ample room for smart acquisition to outpace random sampling. The ASAP Mpro congeneric series, at 9.0%, is richer in actives and provides a complementary scenario where even modest strategy advantages translate to large absolute differences in hits recovered. Here we evaluate whether and to what degree different acquisition strategies accelerate active discovery relative to a random sampling baseline, and whether those trends are consistent across a structurally diverse dataset and a focused congeneric one.

### PXR

![][image3]  
*Figure 1\. PXR dataset cumulative number of active compounds (pEC50 ≥ 6.0) recovered as a function of labeled pool size for each acquisition strategy.* 

Across the PXR diversity deck (54 actives in a pool of 3312 compounds, a 1.6% hit rate), acquisition strategy has a dramatic effect on the rate of active discovery. Exploitation is the clear leader throughout the campaign, recovering approximately 70% of all pool actives by the time 27% of the pool has been labeled with ChemProp, compared to roughly 26% for Random sampling over the same interval. UCB is consistently the second-best strategy, tracking well above Random throughout. Diversity and Random perform nearly identically for hit-finding, while Exploration is the weakest strategy, often performing at or below Random since it directs queries toward uncertain regions rather than toward high predicted potency.

Model choice has a modest and perhaps surprising influence on hit-finding for PXR. ChemProp Exploitation recovers slightly more hits than CheMeleon Exploitation at the same labeled pool size (38 vs 32 actives at n = 900). To understand why, it helps to remember what Exploitation actually does. It ranks the unlabeled pool by predicted mean activity alone, with no role for ensemble uncertainty, and queries the top compounds from that list. The question is therefore which model places true actives higher in that ranking after training on the same initial data.

After fitting on the same 100 randomly selected compounds (training set maximum pEC50 approximately 6.46), ChemProp predicts a maximum activity of 8.48 on the unlabeled pool, more than 2 pEC50 units above anything it was trained on. CheMeleon, by contrast, predicts a maximum of only 6.00, barely reaching the training set ceiling. CheMeleon’s pretrained encoder was trained on Mordred-calculated physicochemical properties across a broad chemical space, with no exposure to any activity data. That pretraining organizes chemical space into a smooth, well-structured latent geometry. When the output head is then trained on 100 PXR activity labels, it can only pull predictions so far from its initialization before training stops, because the encoder weights are already settled and constrain how flexibly the representation adapts. Predictions on unseen compounds therefore remain within the range the head has been trained to map from those fixed representations. ChemProp, initialized from random weights, faces no such constraint. Its encoder and output head co-adapt simultaneously to the 100 training examples, distorting the representation space freely to accommodate the highest observed activities and extrapolating well beyond them on similar unseen compounds. The result is a mean predicted activity gap between true hits and non-hits of 0.64 units for ChemProp versus 0.44 units for CheMeleon at that first query, despite CheMeleon having higher global rank correlation with true activity (Spearman rho approximately 0.60 versus 0.53). CheMeleon is a better global ranker, but ChemProp’s aggressive extrapolation concentrates true hits more tightly at the very top of the list, where Exploitation actually selects. ChemProp places approximately 9 true hits in its top-100 predictions from the first query while CheMeleon places approximately 6, and this first-query advantage compounds into a persistent cumulative gap across the campaign. Tracking the hit-nonhit score gap across subsequent iterations confirms that it does not grow as more actives enter the training set under Exploitation, ruling out any feedback effect from biased training accumulation. The mechanism is fixed at initialization.

Adding ChEMBL pretraining data does not meaningfully shift the hit trajectory on PXR. Warm-started models converge to comparable cumulative hit counts as their no-ChEMBL counterparts after a few hundred pool labels are acquired, indicating that hit-finding here is governed primarily by the acquisition function rather than the starting model weights.

### SARS-CoV-2 Mpro

![][image4]  
*Figure 2\. ASAP SARS-Cov-2 Mpro dataset cumulative number of active compounds (pEC50 ≥ 7.0) recovered as a function of labeled pool size for each acquisition strategy.* 

The ASAP Mpro dataset shows the same qualitative ordering of strategies despite a markedly higher base hit rate (76 actives in a pool of 842 compounds, 9.0%). Exploitation again leads throughout the campaign, while Random trails substantially, and the full strategy ranking is preserved across both model types.

Model choice plays a different role on Mpro than on PXR. CheMeleon Exploitation recovers 55 of 76 pool actives by n = 200 compared to 47 for ChemProp Exploitation, a meaningful 17% more hits from the same number of assays. Random sampling finds only 17 actives at n = 200 regardless of model, confirming that the benefit of CheMeleon is specific to exploitation-driven selection rather than a general improvement in model quality at small sample sizes. The mechanism here is the mirror image of PXR. At the very first Mpro query, before any exploitation bias has entered the training set, CheMeleon already assigns a hit-nonhit predicted activity gap of 1.35 units versus 0.93 units for ChemProp. On a congeneric series where closely related analogues cluster tightly in both chemical and activity space, CheMeleon’s pretrained molecular representations immediately encode which structural features are associated with higher potency. Random initialization lacks this prior and cannot resolve the fine activity distinctions between nearly identical compounds from only 100 training examples. The larger initial gap translates directly into higher first-query precision for CheMeleon and a cumulative hit-finding advantage that persists through the early campaign.

Comparing the two targets, the same underlying mechanism explains both observations. Exploitation selects by predicted mean, so the model that assigns the most extreme predicted values to true actives wins. On a diverse deck like PXR, where actives are scattered across chemical space, a randomly initialized model extrapolates more aggressively from sparse training data and places extreme predictions on potential hits that a pretrained model conservatively scores near the observed range. On a congeneric series like Mpro, where actives are structurally clustered, pretrained representations already resolve the subtle activity distinctions between close analogues and assign a larger score separation from the outset. In both cases the pattern is established at initialization and verified before any exploitation-driven training set bias can take effect. Strategy choice remains the dominant lever on both targets, with Exploitation consistently recovering two to three times as many hits as Random at mid-campaign, but model initialization determines which tool does it more efficiently for a given data type.

## Model accuracy

Recovering actives faster is only one dimension of campaign performance. A complementary question is whether the choice of acquisition strategy influences how well the resulting model generalizes to held-out test compounds. Strategies like Exploitation and EI bias the labeled set toward high-potency regions, which may improve performance near the top of the activity distribution at the expense of accurate prediction across the full range. Random and Diversity sampling, by contrast, maintain broader coverage and might yield better-calibrated global models despite finding fewer hits. We evaluate test-set MAE and Kendall’s τ throughout each campaign to assess whether active strategies produce more accurate models, or whether the label acquisition strategy is largely irrelevant for overall predictive performance.

### PXR

![][image5]  
*Figure 3\. PXR dataset mean absolute error (MAE, pEC50 units) on the held-out random-split test set as a function of labeled pool size, for each of the six acquisition strategies. Shaded bands show ±1 SD across five random seeds.*

In contrast to the strong strategy signal in hit discovery, acquisition strategy choice has a comparatively small effect on test-set MAE for PXR, but the magnitude of that effect depends substantially on the model used. For ChemProp, Exploitation reaches 0.639 pEC50 MAE at n = 900 compared to 0.568 for Random, a gap of 0.071 units driven by the labeled set’s bias toward the active region. For CheMeleon, the same gap narrows to only 0.022 units (0.573 vs 0.551), reflecting that pretrained representations help the model generalize better from a biased labeled pool. In absolute terms, both penalties remain modest and are far outweighed by Exploitation’s hit-finding advantage.

CheMeleon consistently outperforms ChemProp across all strategies. The advantage is largest under Exploitation (0.066 MAE units better at n = 900) and smallest for EI and Diversity (less than 0.01 units). When running an Exploitation campaign, CheMeleon therefore incurs a substantially smaller accuracy penalty than ChemProp, making it the preferred model backbone when both hit-finding speed and model quality are priorities.

Adding ChEMBL pretraining data provides a useful accuracy prior at the very start of the campaign. CheMeleon with ChEMBL achieves 0.83 pEC50 MAE before any pool labels are acquired, compared to 0.93 for ChemProp with ChEMBL. However, this advantage largely disappears by n = 200 pool labels, where both warm-started configurations match their no-ChEMBL counterparts. ChEMBL pretraining is most valuable when the labeled pool is smallest, and its benefit diminishes quickly as pool data accumulates.

Kendall’s τ reinforces these conclusions. ChemProp reaches τ ≈ 0.49 and CheMeleon reaches τ ≈ 0.52 by mid-campaign, with strategy bands largely overlapping within each model. Exploitation produces the lowest τ for ChemProp (approximately 0.42 at n = 900), while CheMeleon strategies are more tightly clustered (0.49 to 0.50), confirming that a biased labeled set depresses rank-ordering performance more severely for a randomly initialized model.  
![][image6]  
*Figure 4\. PXR dataset Kendall's τ rank-correlation between predicted and observed pEC50 on the held-out test set across active learning iterations. Higher values indicate better ranking of compounds by predicted activity. Shaded bands show ±1 SD across five random seeds.*

### SARS-CoV-2 Mpro

![][image7]  
*Figure 5\. ASAP SARS-CoV-2 Mpro dataset mean absolute error (MAE, pEC50 units) on the held-out random-split test set as a function of labeled pool size, for each of the six acquisition strategies. Shaded bands show ±1 SD across five random seeds.*  
![][image8]  
*Figure 6\. ASAP SARS-CoV-2 Mpro dataset Kendall's τ rank-correlation between predicted and observed pEC50 on the held-out test set across active learning iterations. Higher values indicate better ranking of compounds by predicted activity. Shaded bands show ±1 SD across five random seeds.*

For ASAP Mpro, model initialization has a substantially larger effect on accuracy than acquisition strategy. At n = 200, CheMeleon reduces test-set MAE by 0.075 units under Exploitation (0.733 vs 0.808 for ChemProp) and by 0.163 units under Random sampling (0.799 vs 0.962). The gap is largest for Random because ChemProp, without pretrained weights, must build its representations from scratch from the most informationally dilute labeling strategy. Under Exploitation, both models receive targeted labels from the high-potency region, which partially closes the gap.

Within each model type, differences in MAE across strategies remain small, typically less than 0.05 pIC50 units, consistent with the PXR finding. Kendall’s τ reaches approximately 0.62 to 0.66 for CheMeleon and approximately 0.58 to 0.64 for ChemProp at n = 400, with no meaningful strategy separation within either model.

Comparing PXR and Mpro, the magnitude of the CheMeleon accuracy advantage is substantially larger on Mpro (0.08 to 0.16 MAE units) than on PXR (0.01 to 0.07 units). Pretrained molecular representations provide a larger accuracy benefit on a focused congeneric series, where structural patterns encountered during pretraining are more directly applicable to the test distribution, than on a diversity deck where that prior is less transferable.

## Model uncertainty

The acquisition functions used here (EI, UCB, and Exploration) all rely on the committee’s predicted uncertainty σ to guide selection. This means the quality of those decisions is tied directly to how well σ captures actual prediction error. If σ is poorly calibrated, meaning high-σ compounds are not systematically harder to predict, then uncertainty-driven strategies are in effect sampling from noise rather than from meaningful signal. We evaluate this directly by computing the Spearman rank correlation between σ and absolute prediction error on the held-out test set throughout each campaign. A high ρ indicates that the committee correctly identifies which test compounds it is most wrong about. A low ρ indicates that uncertainty and error are largely decoupled, and that acquisition functions relying on σ may not be steering queries as precisely as intended.

### PXR

![][image9]  
*Figure 7\. PXR dataset Spearman rank correlation between predicted uncertainty (σ) and absolute prediction error (|ŷ − y|) on the held-out test set, as a function of labeled pool size, for each acquisition strategy. A positive ρ indicates that σ correctly ranks which test compounds the model is most wrong about. Shaded bands show ±1 SD across five random seeds.*

For PXR, the Spearman ρ between predicted uncertainty and absolute prediction error is consistently low across all strategies and model configurations, ranging from approximately 0.11 to 0.29 throughout the campaign. This indicates that the committee’s uncertainty estimates are a weak proxy for actual prediction error regardless of how the labeled pool is acquired.

Model choice markedly affects the level of ρ. ChemProp configurations produce ρ values of approximately 0.16 to 0.29, while CheMeleon configurations consistently yield lower values of 0.11 to 0.17. Despite CheMeleon’s substantial advantage in prediction accuracy, its uncertainty estimates are less correlated with actual errors. The pretrained representations appear to produce ensembles that are more uniformly confident, reducing the spread in σ needed for ρ to be informative.

The direction of the strategy effect also differs between models. For ChemProp, Exploitation produces the highest ρ (0.29 at n = 900), likely because the biased labeled set creates a sharp contrast between the confidently predicted, well-sampled active region and the uncertain, sparsely labeled inactive region. For CheMeleon, this pattern is reversed, with Exploitation producing the lowest ρ (0.11) while Random and Exploration maintain modestly higher values. ChEMBL warm-starting slightly reduces ρ compared to no-ChEMBL counterparts across both model types, and ρ does not improve monotonically as the labeled pool grows, confirming that more data alone does not resolve the uncertainty calibration issue for this diverse compound set.

### SARS-CoV-2 Mpro

![][image10]  
*Figure 7\. ASAP SARS-CoV-2 Mpro dataset Spearman rank correlation between predicted uncertainty (σ) and absolute prediction error (|ŷ − y|) on the held-out test set, as a function of labeled pool size, for each acquisition strategy. A positive ρ indicates that σ correctly ranks which test compounds the model is most wrong about. Shaded bands show ±1 SD across five random seeds.*

The ASAP Mpro dataset shows broadly similar uncertainty behavior, with ρ values ranging from approximately 0.13 to 0.34. These values are modestly higher than the PXR range, which may reflect the congeneric series having a more structured activity landscape where the committee can better identify compounds that fall outside the chemical variation it has seen.

Unlike PXR, ChemProp and CheMeleon produce comparable ρ on Mpro (both roughly 0.21 to 0.31 depending on strategy at n = 200), so the CheMeleon suppression of ρ observed on PXR does not appear on the congeneric series. Random sampling consistently achieves the highest ρ for both models on Mpro, as it maintains the most diverse labeled set and avoids the high-potency bias that suppresses ρ under Exploitation.

Across both targets and all configurations, the same fundamental limitation holds: ensemble disagreement in these committee models reflects training distribution more than compound difficulty, and no acquisition strategy reliably improves this. Uncertainty-driven strategies such as EI and UCB deliver their hit-finding gains by preferentially selecting high-variance compounds, not because σ is a calibrated error predictor. Exploration’s failure to find hits despite consistently querying high-σ compounds makes this limitation most apparent.

## Takeaways

For **acquisition strategy**, the clearest practical signal from both targets is that **Exploitation** and **UCB** are the preferred choices whenever hit-finding is the primary campaign objective. On the diverse PXR deck, Exploitation recovered roughly three times as many hits as Random by the midpoint of the campaign, and the same pattern held on the congeneric Mpro series, where the advantage was even sharper. The tradeoff is a modest elevation in MAE under Exploitation, but this gap is small in absolute terms (around 0.07 units for ChemProp on PXR, smaller still for CheMeleon) and is far outweighed by the gain in actives found. **Diversity** is the right choice when structural coverage rather than potency is the goal, and it serves as a hedge against scaffold collapse. **Exploration** and **EI** occupy a middle ground and do not decisively outperform **Random** on either hit-finding or accuracy. Random itself is a perfectly reasonable baseline for building a generalizable SAR model, since all strategies converge to similar MAE by the end of the campaign.

For **model initialization**, the data type of the target matters. On the diverse PXR deck, CheMeleon and ChemProp produce comparable hit trajectories under Exploitation, and ChemProp's wider ensemble variance can even give it a slight edge at identifying the relatively rare actives scattered across chemical space. On the congeneric Mpro series, CheMeleon pulls ahead more decisively (55 versus 47 hits at n=200), because a pretrained molecular representation better captures the subtle potency-relevant features that distinguish closely related analogues. Across both targets, CheMeleon produces more accurate models (0.05 to 0.16 lower MAE depending on strategy) and higher rank correlation, so it is the recommended backbone when a generalizable predictive model is also a deliverable of the campaign, not only hit-finding speed.

For **ChEMBL warm-starting**, the benefit is real but short-lived on a diverse dataset like PXR. Starting from ChEMBL provides a meaningful accuracy head-start when the pool-acquired label count is near zero, but both warm-started configurations converge to their no-ChEMBL counterparts by roughly n=200 pool labels. If a campaign can afford even a modest initial screening set, the ChEMBL advantage dissolves quickly. The value of ChEMBL warm-starting is therefore concentrated in the very first iterations, making it most useful when early-stopping a campaign is a possibility or when the labeled pool is expected to remain small.

Taken together, these experiments suggest that the optimal active learning setup shifts with dataset character. On **diverse, low-hit-rate decks** like PXR, acquisition strategy dominates model choice, Exploitation is the clear winner, and ChEMBL warm-starting buys only a short-term accuracy boost. On **congeneric, higher-hit-rate series** like Mpro, a pretrained encoder pays a larger and more sustained dividend, and the margin between acquisition strategies narrows because even a few well-chosen congeneric queries rapidly improve coverage of the relevant chemical series. In both regimes, uncertainty estimates from the committee remain weak absolute predictors of per-compound error (Spearman ρ of 0.10 to 0.30), yet they are directionally useful enough to power effective exploitation-based acquisition throughout the campaign.

## Reproducibility

To reproduce the results in this post, install `openadmet-models` by following the [installation instructions](https://docs.openadmet.org/en/latest/installation.html), clone the blogpost repo, then run:

```shell
# Execute the active learning pipeline (~several GPU-hours)
python run.py   

# Generate all figures from results/all_runs.pkl
python analysis.py  
```

All supporting code lives in `active-learning-blogpost/src/`: [src/helpers.py](http://src/helpers.py) contains the core AL utilities and [src/plots.py](http://src/plots.py) contains all Plotly/Faerun plotting functions. The campaign is governed by parameters specified in [`config.yaml`](http://config.yaml).  
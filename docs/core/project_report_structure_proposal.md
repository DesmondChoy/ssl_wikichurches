# Do Self-Supervised Vision Models Learn What Experts See?

## Working Report Draft

This document converts the earlier report-structure outline into a working draft grounded in the current repository state. It is intended to be readable as an academic report draft rather than as a planning memo, while still keeping visible placeholders for sections whose final wording depends on unsettled figures, tables, or findings.

For a LaTeX-oriented scaffold that can later be moved into the course template, see [`project_report_overleaf_skeleton.tex`](./project_report_overleaf_skeleton.tex).

This draft reflects the current project scope:

- Q1: frozen-model attention alignment
- Q2: fine-tuning effects on attention alignment
- Q3: per-head specialization

It also reflects the current project guidance from the repo: keep the report question-driven, foreground methodology and calibrated interpretation, and avoid turning the main narrative into a route-by-route system walkthrough.

## 1. Abstract

Self-supervised vision models achieve strong downstream performance, but high classification accuracy alone does not reveal whether those models attend to the same visual evidence that human experts consider diagnostically important. This project studies that question in the WikiChurches setting, where expert bounding boxes identify architectural features such as arches, windows, towers, and facade elements that matter for style recognition. We evaluate seven vision models across self-distillation, masked autoencoding, multimodal contrastive pretraining, and a supervised CNN baseline, then measure attention-alignment against 631 expert boxes on 139 annotated church images using IoU, Coverage, MSE, KL divergence, and EMD. The study is organized around three linked questions: how well frozen models align with expert-marked regions, how Linear Probe, LoRA, and Full fine-tuning change that alignment, and whether individual attention heads exhibit descriptive specialization for different architectural features. The current repository already supports the full pipeline for dataset preparation, attention extraction, metric precomputation, fine-tuning analysis, and interactive inspection through a backend and frontend analysis application.

> TODO: Add the final abstract findings sentence once the report locks the headline Q1, Q2, and Q3 claims.

## 2. Project-At-A-Glance Overview

The report studies whether vision models that perform well on style classification also focus on the same architectural evidence that experts use. The current repo state supports a multi-model benchmark, a fine-tuning shift analysis, and a scoped per-head study.

| Study dimension | Current repo-grounded value |
| --- | --- |
| Compared frozen models | 7 |
| Fine-tuning strategies | 3 |
| Alignment metrics | 5 |
| Annotated evaluation images | 139 |
| Expert bounding boxes | 631 |
| Architectural feature types in the ontology | 106 |
| Core research questions | 3 |
| Primary Q3 headline-study models | 4 (`dinov2`, `dinov3`, `mae`, `clip`) |

The project combines a research pipeline and an analysis interface. The pipeline extracts model attention, computes calibrated alignment metrics, stores experiment artifacts, and precomputes cache-backed summaries. The app then exposes those results through Gallery, Image Detail, Dashboard, Compare, Q2, and Q3 surfaces that let the team inspect the same findings at dataset, model, layer, and image level.

> TODO: Insert opening overview table or infographic summarizing the study. Candidate inputs: `README.md`, `docs/core/one_pager_pitch.md`, `outputs/results/active_experiment.json`.

## 3. Introduction and Motivation

A vision model can be correct for the wrong visual reasons. In architectural style recognition, a model that classifies a church as Gothic because it attends to pointed arches and flying buttresses is qualitatively different from one that succeeds because it exploits background regularities, photographer bias, or other shortcuts unrelated to expert reasoning. Accuracy alone cannot distinguish between those cases. For a project framed around trust, interpretability, and the relationship between representation learning and domain knowledge, the question is not only whether a model predicts the right label, but also whether it looks at the right evidence.

WikiChurches provides a strong evaluation setting for that question because it pairs fine-grained architectural-style labels with expert bounding boxes marking characteristic visual features. Those annotations make it possible to compare model attention directly against human expert targets instead of inferring plausibility indirectly from class predictions. They also anchor the evaluation in a domain where meaningful distinctions depend on visually specific, repeated, and semantically rich structures rather than on generic object categories alone.

This report addresses three gaps that the current repo is explicitly designed to study together. First, it compares multiple frozen SSL paradigms and a supervised baseline against the same expert annotations rather than evaluating a single model family in isolation. Second, it asks how attention changes after task adaptation, using shared evaluation images and multiple fine-tuning strategies instead of relying only on frozen-model inspection. Third, it asks whether the model's attention behavior is uniform across heads or whether a smaller subset of heads appears to align better with certain feature types. Taken together, these questions shift the project from a visualization exercise into a domain-grounded evaluation study of what different pretraining and adaptation choices encourage models to attend to.

## 4. Research Questions and Contributions

The study is organized around three linked research questions. Q1 establishes the baseline frozen-model benchmark. Q2 asks whether task-specific adaptation changes that alignment and whether strategy choice matters. Q3 narrows the focus to the descriptive specialization of individual attention heads.

### 4.1 Q1: Frozen-Model Attention Alignment

Do frozen SSL and baseline vision models attend to the same architectural regions that human experts mark as diagnostically important? This is the core benchmark question. It asks whether alignment with expert evidence is already present in the pretraining regime before any task-specific adaptation.

### 4.2 Q2: Fine-Tuning and Attention Shift

How does attention change after adaptation to the style-classification task, and does the strategy matter? This question compares Linear Probe, LoRA, and Full fine-tuning using the same annotated evaluation images. It treats attention shift as both a directional question and a magnitude question: fine-tuning may preserve, improve, or degrade alignment, and the same strategy need not affect every model family in the same way.

### 4.3 Q3: Per-Head Specialization

Do individual attention heads exhibit descriptive specialization for different architectural features, and do the dominant heads change across variants? Q3 is scoped more narrowly than Q1 and Q2. Its goal is not to prove causal explanations for predictions, but to test whether some heads align more strongly than others with expert-marked structures and feature types.

### 4.4 Contributions

- A multi-metric benchmark for comparing expert-alignment across frozen SSL model families and a supervised baseline on the same annotated evaluation set.
- A calibrated Q1 interpretation layer that compares continuous metrics against naive baselines rather than treating raw scores as self-explanatory.
- A Q2 analysis workflow that compares frozen-to-fine-tuned attention shifts across Linear Probe, LoRA, and Full fine-tuning using shared evaluation images and experiment-scoped provenance.
- A scoped Q3 per-head study that reuses the metric pipeline to rank heads and inspect head-feature patterns without overstating causal claims.
- A reproducible pipeline and interactive analysis interface that connect precomputation, experiment artifacts, and qualitative inspection.

## 5. Related Work

### 5.1 Attention Interpretability in Vision Transformers

Prior work has already shown that attention maps in vision transformers can carry useful spatial structure, but the literature is careful about what those maps do and do not prove. DINO popularized the observation that late-layer self-attention can resemble semantic object masks, while attention-rollout work argued that single-layer raw attention may miss how information flows across the full network. Transformer-interpretability work by Chefer et al. further demonstrated that attention visualization alone is not a complete explanation method, which is one reason this project frames its outputs as alignment measurements and descriptive evidence rather than as definitive causal proof.

Representative sources already cited elsewhere in the repo include:

- [Caron et al. (2021), *Emerging Properties in Self-Supervised Vision Transformers*](https://arxiv.org/abs/2104.14294)
- [Abnar and Zuidema (2020), *Quantifying Attention Flow in Transformers*](https://arxiv.org/abs/2005.00928)
- [Chefer et al. (2021), *Transformer Interpretability Beyond Attention Visualization*](https://arxiv.org/abs/2012.09838)

### 5.2 Evaluation Against Human or Expert Annotations

The most relevant methodological precedent is the broader literature that compares model explanations against human- or expert-provided spatial targets. In computer vision, IoU-style localization evaluation, pointing-game style metrics, and plausibility-oriented explanation benchmarks established the basic pattern of comparing model-derived maps against annotated regions. More recent work in medical imaging has applied similar logic to expert annotations and found that model families behave differently when the evaluation target is domain-specific rather than generic. That precedent strengthens the framing of this project as a domain-grounded evaluation study rather than as a new interpretability algorithm.

Representative sources already cited elsewhere in the repo include:

- [Zhou et al. (2016), *Learning Deep Features for Discriminative Localization*](https://arxiv.org/abs/1512.04150)
- [Zhang et al. (2016), *Top-Down Neural Attention by Excitation Backprop*](https://arxiv.org/abs/1511.02668)
- [Choe et al. (2020), *Evaluating Weakly Supervised Object Localization Methods Right*](https://arxiv.org/abs/1910.12449)
- [Chung et al. (2025), *What Should We Learn from Attention Maps? A ViT Study in Medical Imaging*](https://arxiv.org/abs/2503.09535)

### 5.3 Fine-Tuning, Representation Shift, and Attention Drift

The fine-tuning literature suggests that adaptation can reshape pretrained representations in ways that are useful, unstable, or both. Kumar et al. showed that fine-tuning can distort pretrained features relative to linear-probe-style controls, while Biderman et al. argued that LoRA tends to learn less and forget less than full fine-tuning. Work on attention transfer and self-supervised ViT analysis further suggests that attention patterns are not incidental to downstream performance. Q2 builds on this literature by asking not just whether representations change, but whether the change moves attention toward or away from expert-marked architectural evidence.

Representative sources already cited elsewhere in the repo include:

- [Kumar et al. (2022), *Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution*](https://arxiv.org/abs/2202.10054)
- [Biderman et al. (2024), *LoRA Learns Less and Forgets Less*](https://arxiv.org/abs/2405.09673)
- [Park et al. (2023), *What Do Self-Supervised Vision Transformers Learn?*](https://openreview.net/forum?id=azCKuYyS74)
- [Li et al. (2024), *On the Surprising Effectiveness of Attention Transfer for Vision Transformers*](https://arxiv.org/abs/2411.09702)

### 5.4 Attention-Head Specialization

The head-specialization literature, especially Voita et al., established the broader idea that only a subset of heads may carry the most interpretable or task-relevant behavior. Later ViT work extended that intuition to head-level spatial patterns in vision models. Q3 adopts that descriptive framing. It asks whether some heads align more strongly with architectural features than others, not whether one can reduce the full model decision to a single head.

Representative sources already cited elsewhere in the repo include:

- [Voita et al. (2019), *Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting*](https://arxiv.org/abs/1905.09418)
- [Li et al. (2023), *Interpreting Vision Transformer from Head Distribution*](https://doi.org/10.1109/TVCG.2023.3327840)
- [Walmer et al. (2023), *Teaching Matters: Investigating the Role of Supervision in Vision Transformers*](https://openaccess.thecvf.com/content/CVPR2023/papers/Walmer_Teaching_Matters_Investigating_the_Role_of_Supervision_in_Vision_Transformers_CVPR_2023_paper.pdf)
- [Raghu et al. (2021), *Do Vision Transformers See Like Convolutional Neural Networks?*](https://arxiv.org/abs/2108.08810)

### 5.5 Cultural Heritage and Architectural Recognition Context

Architectural heritage and cultural-recognition datasets are less saturated than standard object-recognition benchmarks, yet they are especially appropriate for attention-alignment studies because the diagnostic evidence is often structural, expert-defined, and visually localized. WikiChurches is particularly useful in this regard because it combines style labels with bounding-box annotations of characteristic building parts. The project's novelty therefore lies less in inventing a new metric than in bringing together expert-annotation-grounded evaluation, multiple SSL paradigms, adaptation analysis, and an architecture-focused domain where "looking at the right evidence" is central to the research question.

Representative sources already cited elsewhere in the repo include:

- [Barz and Denzler (2021), *WikiChurches: A Fine-Grained Dataset of Architectural Styles with Real-World Challenges*](https://arxiv.org/abs/2108.06959)
- [Hu et al. (2025), *ASCENT-ViT: Attentive Semantic Concept Explainability for Vision Transformers*](https://www.ijcai.org/proceedings/2025/58)

> TODO: Convert these inline markdown links into the course citation style and final bibliography format. Current source notes: `docs/research/claude_novelty_check.md`, `docs/research/attention_methods.md`, and `docs/reference/metrics_methodology.md`.

## 6. Dataset and Problem Setup

The primary dataset is WikiChurches, a fine-grained architectural-style dataset centered on European church buildings. For this project, the dataset matters not only because it supports classification, but because it includes expert annotations of diagnostically important building parts. Those boxes make it possible to evaluate whether models attend to the same structural evidence that experts use to distinguish styles.

The report uses two related data scopes. The first is the annotated subset used for attention-alignment evaluation: 139 images with 631 expert bounding boxes. The second is the larger style-labeled pool derived from `churches.json`, which supports linear-probe and fine-tuning experiments. The root documentation cites 9,485 images in the official WikiChurches release. The implementation also documents that a local mirror may expose a slightly different raw file count, but the main report should anchor its description to the official release and to the 139-image expert-annotated subset that defines the alignment benchmark.

The annotation file `building_parts.json` defines an ontology of 106 feature types and stores bounding boxes in normalized `left`, `top`, `width`, `height` coordinates. The code clamps negative left or top coordinates to zero, converts boxes into pixel masks at the target heatmap resolution, and combines multiple boxes per image into a union mask for the primary image-level IoU and Coverage calculations. The current project proposal documents the annotated style distribution as Romanesque 51, Gothic 49, Renaissance 22, and Baroque 17, and the implementation maps the corresponding Wikidata style IDs into the working style-classification labels used for training and evaluation.

The current preprocessing strategy keeps the annotated evaluation images out of the primary fine-tuning train and validation splits. This separation matters because Q2 asks how adaptation changes attention on the same expert-annotated pool, so those images should remain evaluation-only in the primary experiment path. For the attention pipeline, the project uses model-appropriate image preprocessing and generates standardized heatmaps at the app's working resolution for metric computation and visualization.

The most important dataset caveat is sparse annotation bias. WikiChurches annotates representative instances of features rather than exhaustively marking every visible instance. As a result, a model can correctly attend to several copies of the same feature while still being penalized by IoU for attending outside the single annotated box. This affects per-bbox interpretation most strongly and should be treated as a documented limitation rather than as a bug in the metric implementation. The current repo addresses this by emphasizing cross-metric interpretation, by distinguishing union-mask from per-bbox views, and by keeping the limitation visible in both methodology and discussion.

## 7. Methodology

The methodology is designed to support a comparative evaluation study rather than a single-model demo. It therefore has to define model coverage, attention extraction rules, metric interpretation, experimental splits, statistical comparisons, and reproducibility safeguards in a way that remains coherent across Q1, Q2, and Q3.

### 7.1 Models and Attention Extraction

The frozen benchmark compares seven models. Six are transformer-based models: `dinov2`, `dinov3`, `mae`, `clip`, `siglip`, and `siglip2`. The seventh, `resnet50`, acts as a supervised CNN baseline. All transformer models use ViT-Base scale backbones in the current implementation. DINOv2 uses a patch-size-14 configuration with 4 register tokens and a 16 x 16 spatial patch grid. DINOv3, MAE, CLIP, SigLIP, and SigLIP2 use patch size 16 and a 14 x 14 patch grid at the standard input resolution. ResNet-50 is handled through a separate Grad-CAM path.

The attention-extraction method depends on model architecture. For DINOv2, DINOv3, MAE, and CLIP, the main methods are CLS attention and attention rollout. CLS attention isolates the class token's attention to patch tokens in a selected layer. Attention rollout composes attention across layers to capture indirect information flow. For SigLIP and SigLIP2, the project uses a mean attention proxy because those models do not expose a CLS-token path equivalent to the DINO, MAE, or CLIP setup. For ResNet-50, the interpretability baseline is Grad-CAM rather than transformer attention.

These choices are important for interpretation. CLS attention and rollout are not interchangeable, and mean attention for the SigLIP family is an interpretability proxy rather than an architecture-native pooling explanation. That distinction becomes especially important in Q3, where the report should avoid mixing architecture-native and proxy-based per-head claims without qualification.

| Model | Training paradigm | Default method | Other supported method(s) |
| --- | --- | --- | --- |
| `dinov2` | Self-distillation | `cls` | `rollout` |
| `dinov3` | Self-distillation with Gram anchoring | `cls` | `rollout` |
| `mae` | Masked autoencoding | `cls` | `rollout` |
| `clip` | Contrastive language-image pretraining | `cls` | `rollout` |
| `siglip` | Sigmoid-based contrastive language-image pretraining | `mean` | None |
| `siglip2` | Sigmoid-based contrastive pretraining with improved dense features | `mean` | None |
| `resnet50` | Supervised CNN baseline | `gradcam` | None |

### 7.2 Alignment Metrics

The project uses five alignment metrics because no single score is sufficient for all of the intended interpretations. IoU and Coverage answer slightly different questions about whether the model's attention lands on expert-marked regions. MSE, KL divergence, and EMD compare the full heatmap against a soft target derived from the boxes and help distinguish overlap from distributional fit.

IoU is the primary threshold-dependent overlap metric. It thresholds the attention heatmap using exact pixel-count percentile selection via `torch.topk` and measures overlap against the union of all boxes for the image. Higher IoU is better. Coverage is threshold-free and measures what fraction of total attention energy falls inside the annotated union mask. Higher Coverage is better. MSE, KL, and EMD are computed against a Gaussian soft-union target derived from the expert boxes. Lower MSE, lower KL, and lower EMD are better. The report should make that direction explicit whenever these metrics first appear because the lower-is-better convention for the continuous metrics affects both Q1 leaderboard interpretation and Q2 delta interpretation.

| Metric | Type | Target representation | Direction |
| --- | --- | --- | --- |
| IoU | Threshold-dependent | Binary union mask | Higher is better |
| Coverage | Threshold-free | Binary union mask with attention energy | Higher is better |
| MSE | Threshold-free | Gaussian soft-union heatmap | Lower is better |
| KL divergence | Threshold-free | Gaussian soft-union distribution | Lower is better |
| EMD | Threshold-free | Gaussian soft-union distribution on shared 8 x 8 support | Lower is better |

### 7.3 Baselines and Calibration

Raw continuous-metric values are difficult to interpret without reference points, because unlike accuracy they do not come with a fixed notion of "chance" or "ceiling." The current project therefore calibrates Q1 continuous metrics against naive baselines: random attention, center Gaussian, saliency prior, and Sobel edge. This matters because a model that merely beats random attention is not necessarily attending to expert-relevant structures in a meaningful way. Stronger evidence comes from beating several naive baselines, including ones that capture generic center or low-level edge biases.

The documented dataset-level baseline references currently used in the repo are shown below. Lower is better for every metric in the table.

| Baseline | MSE | KL | EMD |
| --- | --- | --- | --- |
| Random | 0.3192 | 3.3627 | 0.3468 |
| Center Gaussian | 0.1770 | 2.6317 | 0.2836 |
| Saliency Prior | 0.0957 | 2.6111 | 0.2654 |
| Sobel Edge | 0.0376 | 3.2237 | 0.3137 |

These calibration values make it possible to interpret Q1 results in a more measured way. For example, beating random only is weak evidence, matching center bias suggests generic spatial priors may still dominate, and beating all naive baselines on a metric is stronger support that the model is capturing non-trivial semantic alignment rather than only generic image structure.

### 7.4 Fine-Tuning Protocol

Q2 uses a shared experiment-batch workflow. The primary fine-tuning path trains on the non-annotated style-labeled pool, reuses one shared stratified validation split across all `model x strategy` runs in the same batch, selects the best checkpoint per run by classification validation accuracy, and then evaluates attention alignment on the held-out annotated evaluation images. The 139 bbox-annotated images are excluded from the primary train and validation split and remain the evaluation pool for frozen-vs-fine-tuned comparison.

The current comparison covers three strategies. Linear Probe freezes the backbone and trains only the classification head. LoRA inserts trainable low-rank adapters into attention layers while keeping the backbone largely frozen. Full fine-tuning updates the backbone end-to-end. ResNet-50 is not part of the fine-tuning comparison. The canonical artifact layout is experiment scoped and selected through `outputs/results/active_experiment.json`, which points the app and reporting scripts to the active run matrix and Q2 analysis artifact.

| Strategy | Backbone updates | Intended role in Q2 |
| --- | --- | --- |
| Linear Probe | No | Frozen-backbone control |
| LoRA | Parameter-efficient partial adaptation | Intermediate attention shift |
| Full fine-tuning | Yes, end-to-end | Maximum adaptation capacity |

### 7.5 Q3 Per-Head Scope

Q3 is intentionally narrower than the rest of the report. The most defensible headline scope is the set of architecture-native CLS-token models: `dinov2`, `dinov3`, `mae`, and `clip`. Within that scope, the primary variants are `frozen`, `lora`, and `full`, while `linear_probe` is best treated as a control condition rather than as a main adaptation claim because the backbone does not change. The main Q3 method is `cls`.

The report should explicitly exclude `siglip` and `siglip2` from the primary Q3 claim because their per-head analysis relies on a mean-attention proxy rather than the model's learned pooling head. It should also exclude `resnet50`, which has no transformer attention heads, and avoid claiming that raw per-head attention proves causal feature use. In this report, Q3 is a descriptive head-specialization analysis.

### 7.6 Statistical Analysis

The repo supports paired model comparisons, paired t-tests, Wilcoxon signed-rank tests, bootstrap confidence intervals, Cohen's d for paired differences, and Holm multiple-comparison correction. This statistical layer is especially important in Q2, where the same evaluation images are reused across frozen and fine-tuned conditions and where many model-strategy-metric combinations are compared within shared correction families. The current experiment artifacts serialize correction metadata explicitly, which helps preserve the logic behind headline significance calls instead of leaving it implicit in separate analysis notes.

### 7.7 Methodological Safeguards and Reproducibility

Several design choices strengthen the credibility of the findings. The project uses stable model configuration definitions, exact top-k thresholding for IoU, documented Gaussian-target construction for continuous metrics, explicit dataset-split artifacts for the primary fine-tuning path, and experiment-scoped manifests and run matrices that preserve checkpoint-selection provenance. Combined with cache-backed metric storage and active-experiment pointers, this gives the report a defensible artifact-based workflow rather than a collection of ad hoc screenshots or manually assembled numbers.

## 8. System and Analysis Interface

The software system is the vehicle for running the study and inspecting the results. It should therefore appear in the report as supporting infrastructure rather than as the sole research contribution. At a high level, the system has three layers.

The first layer is the precompute and cache pipeline. It generates frozen and fine-tuned attention heatmaps, feature caches, heatmap images, and the SQLite metrics database that powers leaderboard, progression, Q2, and Q3 queries. The same pipeline also supports per-head cache generation for the scoped Q3 study.

The second layer is the experiment workflow. Fine-tuning scripts write checkpoints, run manifests, split artifacts, experiment ledgers, run matrices, and Q2 summary artifacts into experiment-scoped output directories. This is the operational backbone of the Q2 analysis and the reason the report can describe checkpoint selection, evaluation holdout discipline, and artifact provenance in concrete terms.

The third layer is the analysis interface itself: a FastAPI backend plus a React frontend. The frontend exposes Gallery, Image Detail, Compare, Dashboard, Q2, and Q3 surfaces. The backend resolves cached attention, metrics, and comparison summaries into those views. For the report, the key point is that the app supports the research workflow by making the same quantitative and qualitative evidence inspectable at multiple levels, not that every route is a separate result.

## 9. Results

The current repo already contains enough checked-in artifacts to support a substantive draft narrative for Q1 and Q2, while Q3 remains appropriately more placeholder-heavy. Because the report is still a mixed draft, the sections below treat the existing artifacts as current evidence rather than as permanently frozen final tables.

### 9.1 Q1 Results: Frozen-Model Attention Alignment

The current frozen-model artifacts suggest that DINOv3 is the strongest all-around model on the main leaderboard when the interpretation is calibrated across metrics rather than read from a single score in isolation. In the checked-in `metrics_summary.json`, DINOv3 has the best frozen IoU at layer 11 with `IoU@90 = 0.1327`, the best Coverage at layer 11 with `0.1373`, the best KL divergence at layer 11 with `2.3247`, and the best EMD at layer 11 with `0.2600`. This gives DINOv3 the cleanest cross-metric story among the frozen models in the current artifact set.

The Q1 continuous-baseline summary adds an important calibration layer. It reports that DINOv3 beats all four naive baselines on MSE, KL, and EMD at its best default-method layers. DINOv2 shows a weaker but still credible profile: it beats all four baselines on MSE, but only beats the random and Sobel baselines on KL and EMD. ResNet-50 also forms part of a second tier, with stronger frozen overlap than several SSL models on IoU but a less dominant cross-metric profile than DINOv3.

The most interesting Q1 divergence in the current artifacts comes from the SigLIP family. `siglip` and `siglip2` achieve the best frozen MSE values (`0.0175`) yet fall below the random baseline on EMD. In other words, they look strong if the evaluation emphasizes bounded pointwise deviation from the Gaussian target, but weak if the evaluation emphasizes transport distance. That cross-metric split is exactly why the report should avoid reducing Q1 to a single metric. It suggests that some models may produce heatmaps that are locally smooth or compact while still missing the right spatial distribution in a broader sense.

CLIP presents another instructive contrast. In the frozen state it lags the stronger DINO-family models on IoU and Coverage, and its best IoU occurs early in the network at layer 0 rather than in a late-layer regime. This makes CLIP look less naturally aligned in the frozen benchmark, which in turn makes the Q2 adaptation results more important for interpreting what fine-tuning changes.

> TODO: Insert final Q1 frozen-model comparison table or figure. Candidate inputs: `outputs/cache/metrics_summary.json` and `outputs/results/q1_continuous_baseline_comparison.json`.

### 9.2 Q2 Results: Fine-Tuning Effects on Attention

The current Q2 artifacts support a clear provisional storyline: Linear Probe acts as a near-zero control, while LoRA and Full fine-tuning produce model-dependent attention shifts rather than a uniform "fine-tuning helps" story. In the checked-in experiment `fine_tuning_primary_20260327`, the reference Q2 rows show exactly zero deltas for Linear Probe across all reported metrics because the backbone remains frozen. That behavior is methodologically useful because it confirms that the Q2 pipeline is measuring attention change in the model rather than merely recomputing the same frozen heatmaps under a new label.

The checked-in multi-metric improvement heatmap is already strong enough to include in this mixed draft because it compresses the full strategy comparison into one view and makes the zero-shift Linear Probe control immediately visible.

![Draft Q2 multi-metric improvement heatmap](https://raw.githubusercontent.com/DesmondChoy/ssl_wikichurches/main/outputs/figures/02_all_metrics_improvement_heatmap.png)

*Draft Figure. Sign-normalized Q2 metric deltas for each model and strategy. Blue denotes improvement, red denotes degradation, and asterisks denote significance in the generated artifact. The strongest positive clusters appear in CLIP, MAE, and the SigLIP family, while Linear Probe remains at zero by construction.*

The most dramatic improvement currently appears in CLIP. Full fine-tuning raises CLIP's `IoU@90` from `0.0181` to `0.0745` and raises Coverage from `0.0510` to `0.1047`, while also decreasing KL from `3.6873` to `2.6967` and EMD from `0.4096` to `0.3071`. LoRA also improves CLIP substantially, but not as strongly as Full fine-tuning. This is one of the clearest current examples of a model whose frozen attention looks relatively weak yet whose task-conditioned attention becomes much more expert-aligned after adaptation.

MAE and the SigLIP family also show meaningful improvements under LoRA and Full fine-tuning. For MAE, both strategies improve `IoU@90`, Coverage, KL, and EMD, with LoRA currently producing a particularly strong MSE reduction in the saved artifact. For `siglip` and `siglip2`, LoRA and Full both improve `IoU@90`, Coverage, KL, and EMD, though the absolute frozen baseline remains weaker than the DINO-family models on the overlap metrics. In contrast, the DINO family is more stable. DINOv2 stays close to preserve across most reported metrics, while DINOv3 largely preserves its strong frozen IoU but shows some threshold-free degradation under certain adapted variants in the current artifact set.

This divergence supports an interpretation already noted in the repo's fine-tuning notes: some model families appear to require task-specific adaptation before their attention becomes spatially aligned with expert features, whereas others begin with stronger frozen spatial coherence and therefore move less. The current figure commentary makes the same point more compactly, noting that CLIP and the SigLIP family show clearer improvement trajectories, while DINOv2 and DINOv3 remain comparatively flat.

The preserve/enhance/destroy framing is therefore useful, but it should be reported carefully. The current checked-in figure commentary summarizes `46` enhance, `16` preserve, and `10` destroy outcomes across `72` non-linear-probe model-strategy-metric combinations. That count is a helpful draft summary rather than a substitute for the final table, and the final report should make its counting convention explicit when the figure set is locked.

The checked-in preserve/enhance/destroy figure helps simplify that same result into an easily scannable classification layer and is worth retaining in the draft because it exposes both the dominant improvement pattern and the remaining regression risk.

![Draft Q2 preserve-enhance-destroy summary](https://raw.githubusercontent.com/DesmondChoy/ssl_wikichurches/main/outputs/figures/07_preserve_enhance_destroy.png)

*Draft Figure. Each cell classifies a model-strategy-metric outcome as Enhance, Preserve, or Destroy using the run-matrix logic described in the figure commentary. Enhancement is the dominant outcome in the current artifact set, but the remaining destroy cells show that adaptation can still move attention in the wrong direction.*

The forest-plot visualization adds the statistical layer that the heatmap and categorical summary cannot show on their own, making it easier to distinguish robust movement from small, noisy shifts.

![Draft Q2 forest plot with bootstrap confidence intervals](https://raw.githubusercontent.com/DesmondChoy/ssl_wikichurches/main/outputs/figures/08_forest_plot_ci.png)

*Draft Figure. Mean Q2 deltas with 95% bootstrap confidence intervals for LoRA and Full fine-tuning across six metrics, sign-normalized so rightward always means improvement. This is currently the clearest checked-in figure for showing that several CLIP, MAE, and SigLIP-family gains are not merely anecdotal.*

The draft can also support at least one qualitative example of attention shift rather than relying only on aggregate summaries. The current issue-focused shift map is useful as a provisional example because it shows what a localized redistribution of attention can look like on the architectural facade itself.

![Draft Q2 qualitative attention-shift example](https://raw.githubusercontent.com/DesmondChoy/ssl_wikichurches/main/docs/assets/q2_shift_map_issue_focused.png)

*Draft Figure. Example shift map for a LoRA-adapted model relative to the frozen baseline. Blue indicates regions that gained attention after adaptation and red indicates regions that lost attention. This should remain a supporting figure rather than a headline claim, but it gives the reader a concrete visual intuition for the type of change quantified by the aggregate metrics.*

> TODO: Convert the current draft Q2 figure embeds into final float placement and cross-references in the course template. Current draft assets: `outputs/figures/02_all_metrics_improvement_heatmap.png`, `outputs/figures/07_preserve_enhance_destroy.png`, `outputs/figures/08_forest_plot_ci.png`, and `docs/assets/q2_shift_map_issue_focused.png`.

### 9.3 Q3 Results: Per-Head Specialization

The current repository already supports Q3 data extraction, storage, ranking, and inspection, but the report should keep the findings section narrower and more conservative than Q1 and Q2. The most defensible headline scope is the set of architecture-native CLS-token models: `dinov2`, `dinov3`, `mae`, and `clip`, with `frozen`, `lora`, and `full` as the primary variants. Within that scope, the available pipeline can rank heads by metric, build head-by-feature matrices, store per-image head-feature exemplar rows, and expose them through Dashboard Q3, Image Detail Q3, and the advanced `/q3` workspace.

At the same time, the report should resist the temptation to overstate what Q3 currently proves. The Q3 methodology note is explicit that raw post-softmax self-attention provides a descriptive head-specialization analysis, not a full causal attribution method. It is also explicit that `siglip` and `siglip2` should stay out of the primary headline scope because their per-head analysis uses a mean-attention proxy rather than the models' learned pooling heads.

The most appropriate Q3 result framing in this mixed draft is therefore that the repo already supports the intended head-ranking and head-feature workflow, the primary study scope is defined, and the final narrative should focus on whether late-layer head dominance appears sparse, whether supervision family changes the dominant head set, and whether `lora` or `full` shifts those dominant heads relative to `frozen`. Until the final Q3 figures and aggregated claims are frozen, this section should remain descriptive and explicitly placeholder-heavy.

> TODO: Insert final Q3 head-ranking or head-feature figure. Candidate inputs: `outputs/cache/metrics.db`, `outputs/cache/attention_viz.h5`, and the Q3-specific cache tables documented in `docs/reference/per_head_methodology.md`.

> TODO: Add the finalized Q3 narrative once the team confirms which scoped head-specialization claims are mature enough for the main report.

## 10. Discussion

The discussion should explain why the results are intuitive or surprising, not repeat the tables. The current repo evidence already supports several useful interpretations, though some of them remain provisional until the final figure set is frozen.

### 10.1 Intuitive vs Surprising Findings

One intuitive result is that Linear Probe behaves like a true control for attention change: because the backbone stays frozen, the attention-alignment metrics remain unchanged. Another intuitive result is that stronger task-conditioned adaptation can help models whose frozen attention is not yet strongly expert-aligned. The current CLIP, MAE, and SigLIP-family results fit that pattern well.

A more surprising result is that the strongest frozen continuous score on one metric does not always translate into the most defensible overall alignment story. The SigLIP family illustrates this clearly by pairing excellent frozen MSE with weak frozen EMD. Another surprising result is that the DINO family, especially DINOv2, does not improve dramatically under Q2 even though several other model families do. The working interpretation in the repo is that some models begin with stronger spatial coherence as a byproduct of pretraining, leaving less room for adaptation to improve expert alignment.

### 10.2 What the Results Suggest About Pretraining Objectives

The current artifact set tentatively suggests that pretraining objectives shape not only downstream accuracy but also the ease with which attention becomes expert-aligned. DINOv3's strong frozen Q1 profile suggests that self-distillation can yield attention patterns that are already comparatively close to expert-marked spatial evidence. By contrast, CLIP, MAE, and the SigLIP family appear to gain more from task-specific adaptation in Q2, which may indicate that their frozen attention is less tightly anchored to the expert-marked architectural structures needed for this domain.

This interpretation should remain measured. The report should not claim that one pretraining paradigm is universally "better" at explanation. Instead, it should state that the current WikiChurches benchmark suggests different paradigms start from different spatial priors and respond differently to task adaptation when evaluated against expert evidence.

### 10.3 Practical Implications

From a practical perspective, the current results suggest that model selection for domain adaptation should not be guided by classification accuracy alone. A frozen model with strong expert alignment may be preferable in settings where plausible evidence use matters, while a weaker frozen model may still become a good choice if adaptation reliably improves both task performance and alignment. The Q2 comparison also suggests that LoRA can capture a substantial share of the attention-shift benefit in several models without always requiring the full cost or instability of end-to-end fine-tuning.

### 10.4 Limitations and Threats to Validity

Several limitations should remain explicit in the final report. The annotated evaluation subset is small relative to the full image pool, which limits statistical power and domain breadth. The bounding boxes are expert-guided but not exhaustive, which introduces sparse annotation bias, especially in per-bbox interpretations. Attention itself is also an incomplete explanation signal, so a strong alignment result should be read as evidence of plausibility or spatial correspondence rather than as definitive proof of causal reasoning.

The metric design adds additional interpretive constraints. IoU depends on the thresholding rule, and continuous metrics require calibration against naive baselines to avoid overreading raw numbers. Q3 has its own guardrails: the primary claim should stay within architecture-native CLS-token models, and even there the analysis remains descriptive rather than causal.

> TODO: Add any final threat-to-validity text tied to the exact figure set, especially if the final report includes additional per-feature or per-style tables.

## 11. Conclusion

This project asks whether self-supervised vision models attend to the same architectural evidence that experts consider diagnostically important. Using WikiChurches, expert bounding boxes, and a multi-metric alignment framework, the report studies three linked questions: how well frozen models align with expert evidence, how fine-tuning changes that alignment, and whether individual attention heads exhibit descriptive specialization.

The current repository already supports a strong draft conclusion. The project has a defensible methodology, a calibrated Q1 benchmark, an experiment-scoped Q2 analysis workflow, and a scoped Q3 study design. The draft evidence suggests that model families differ meaningfully both in their frozen spatial alignment and in how much they change under task-specific adaptation. At the same time, the report should remain careful about metric calibration, sparse annotation bias, and the broader limitation that attention is not identical to causal explanation.

> TODO: Add the final conclusion sentence that synthesizes the locked headline findings once the report figures and tables are frozen.

## 12. Appendix

The appendix should absorb material that is useful for reproducibility or supplementary interpretation without overloading the main narrative. Likely appendix items include additional leaderboards, extra qualitative examples, artifact provenance tables, per-metric or per-style supplementary summaries, and implementation details that would distract from the question-driven flow of the main report.

Potential appendix content already has clear repo anchors:

- experiment artifact layout and provenance: `docs/reference/fine_tuning_run_matrix.md`, `outputs/results/active_experiment.json`, `outputs/results/experiments/fine_tuning_primary_20260327/run_matrix.json`
- continuous-metric calibration details: `docs/reference/metrics_methodology.md`, `outputs/results/q1_continuous_baseline_comparison.json`
- supplementary Q2 figures: `outputs/figures/01_val_accuracy_by_model_strategy.png`, `outputs/figures/04_iou_delta_by_percentile.png`, `outputs/figures/05_iou_coverage_mse_kl_emd_radar.png`, `outputs/figures/06_val_accuracy_vs_iou90_delta.png`, and `outputs/figures/09_per_image_delta_strips.png`
- Q3 technical caveats and data layout: `docs/reference/per_head_methodology.md`, `outputs/cache/metrics.db`

> TODO: Insert appendix table for experiment artifact provenance. Candidate inputs: `outputs/results/active_experiment.json` and `outputs/results/experiments/fine_tuning_primary_20260327/run_matrix.json`.

> TODO: Insert appendix table for supplementary Q1/Q2 result artifacts. Candidate inputs: `outputs/results/q1_continuous_baseline_comparison.json`, `outputs/cache/metrics_summary.json`, and `outputs/figures/`.

# Do Self-Supervised Vision Models Learn What Experts See?

**Attention Alignment with Human-Annotated Architectural Features**

ISY5004 Intelligent Sensing Systems Practice Project | Team Size: 3

---

## Problem

Self-supervised learning (SSL) models like DINOv2, MAE, and CLIP learn powerful visual representations without labels. But a critical question remains unanswered: **do these models attend to the same visual features that domain experts consider diagnostic?** A model that classifies Gothic churches by attending to flying buttresses is qualitatively different from one that exploits background correlations --- yet existing benchmarks measure only classification accuracy, not *where* models look.

This project fills that gap by quantitatively measuring alignment between SSL attention patterns and expert-annotated architectural features, using the WikiChurches dataset (Barz & Denzler, NeurIPS 2021).

## Research Questions

| # | Question | Method |
|---|----------|--------|
| **Q1** | Do SSL models attend to expert-identified diagnostic features? | IoU between thresholded attention maps and 631 expert bounding boxes across 6 models and 12 layers |
| **Q2** | Does fine-tuning shift attention toward expert features, and does the strategy matter? | Compare delta-IoU (fine-tuned minus frozen) across Linear Probe vs LoRA vs Full fine-tuning with paired statistical tests |
| **Q3** | Do individual attention heads specialize for different architectural features? | Per-head IoU against expert bounding boxes; rank-based analysis to identify heads consistently aligned with specific feature types |

## Approach

**Dataset:** WikiChurches --- 9,485 church images, 631 expert bounding boxes across 139 annotated churches in 4 architectural styles (Gothic, Romanesque, Renaissance, Baroque).

**Models compared** (all ViT-B for controlled comparison):

| Model | Paradigm | Key Property |
|-------|----------|------------|
| DINOv2 | Self-distillation | Emergent object segmentation |
| DINOv3 | Self-distillation + Gram anchoring | Improved dense features |
| MAE | Masked autoencoding | Pixel-level reconstruction |
| CLIP | Contrastive (language-image, softmax) | Semantic alignment |
| SigLIP 2 | Contrastive (language-image, sigmoid) | Dense feature quality |
| ResNet-50 | Supervised (Grad-CAM) | CNN baseline |

**Methodology:**
1. Extract attention maps (CLS attention, rollout, Grad-CAM) from pretrained models
2. Threshold at multiple percentiles and compute IoU against expert bounding boxes
3. Fine-tune on 4-class style classification (~4,790 images, 139 eval images held out)
4. Re-extract attention and measure delta-IoU to quantify attention shift
5. Analyze per-head IoU to identify heads specialized for architectural features (e.g., windows, portals, towers)

## Preliminary Results

| Model | Best IoU (frozen) | Rank | Key Observation |
|-------|:-:|:-:|-----------------|
| DINOv3 | 0.133 | 1 | Self-distillation leads; best at final layer |
| ResNet-50 | 0.090 | 2 | Supervised CNN competitive via Grad-CAM |
| DINOv2 | 0.082 | 3 | Strong semantic attention in later layers |
| CLIP | 0.049 | 4 | Best at layer 0 --- attention degrades deeper |
| SigLIP 2 | 0.047 | 5 | Mid-layer peak, no CLS token |
| MAE | 0.037 | 6 | Near-uniform across layers; reconstruction != localization |

**Fine-tuning effect (DINOv2):** delta-IoU = +0.009, *p* = 0.031 (Wilcoxon signed-rank, *n* = 139). Small but statistically significant shift toward expert features after full fine-tuning.

## Novelty

No prior work benchmarks multiple SSL paradigms on the same expert-annotated dataset or measures how fine-tuning shifts attention alignment. The closest precedent is Chung et al. (2025, medical imaging), whose finding that "DINO does NOT necessarily outperform supervised/MAE on medical data" strengthens the case for domain-specific evaluation. The delta-IoU methodology for comparing fine-tuning strategies against expert annotations is unexplored in literature. Per-head analysis (Q3) extends Voita et al. (ACL 2019) head specialization framework to vision transformers with domain-specific ground truth --- no existing work computes per-head IoU against expert-annotated architectural features.

## Alignment with ISY5004

| Requirement | How Addressed |
|-------------|---------------|
| Intelligent sensing technique | SSL visual feature learning (5 ViT models + CNN baseline) |
| Image analytics | Attention extraction, feature attribution, localization evaluation |
| Dataset handling | Public dataset with preprocessing, augmentation, data cleaning |
| Experimental comparison | Ablation across 6 models, 4 attention methods, 7 thresholds, 3 fine-tuning strategies |
| System deliverable | Reproducible experimental pipeline with precomputed metrics and interactive analysis tool |

## FAQ

**Why IoU and coverage? What's the justification for these metrics?**

IoU (Intersection over Union) is the standard localization metric used in prior attention-interpretability work (Chefer et al., CVPR 2021). It measures spatial overlap between two binary masks --- in our case, the thresholded attention map and the expert-annotated bounding boxes. Because IoU requires choosing a binarization threshold (e.g., keep the top 10% of attention at the 90th percentile), we sweep 7 percentiles (50--95%) and report best-layer IoU per model at the default 90th percentile.

Coverage complements IoU by being **threshold-free**: it measures what fraction of total attention *energy* (the continuous heatmap values, not a binary mask) falls inside the annotated regions. A model that spreads low-intensity attention everywhere would score high coverage but low IoU, while a model that concentrates attention sharply but in the wrong place would score low on both. Together, the two metrics distinguish "looking at the right place" (IoU) from "how much attention budget lands on expert features" (coverage).

**Metrics change when bounding boxes are selected. What happens under the hood?**

By default, the IoU and coverage scores are computed against the **union of all bounding boxes** for a given image --- a single merged mask representing every expert-annotated feature. When a user selects an individual bounding box in the Annotations panel, the frontend calls a dedicated API endpoint (`GET /metrics/{image_id}/bbox/{bbox_index}`) that computes metrics **on-the-fly for that single bbox only**. The backend loads the cached attention tensor from HDF5, generates a mask from just the selected bbox's coordinates, and computes IoU and coverage against that single-bbox mask instead of the union. This lets users see whether the model attends to a specific architectural feature (e.g., a rose window) versus the aggregate. Deselecting reverts instantly to the precomputed union-of-all-bboxes metric with no re-fetch.

**What is the project scope?**

The project answers three research questions: (Q1) whether frozen SSL models attend to expert-identified features, measured by IoU across 6 models and 12 layers; (Q2) whether fine-tuning shifts attention toward those features, comparing delta-IoU across Linear Probe, LoRA, and Full fine-tuning with paired statistical tests; and (Q3) whether individual attention heads specialize for different architectural feature types, via per-head IoU analysis. The deliverable is a reproducible experimental pipeline with precomputed metrics and an interactive visualization tool for exploring attention alignment across models, layers, and individual features. Out of scope: real-time inference, deployment, transfer to non-architectural domains, and attention-supervised training objectives (e.g., core-tuning, COIN).

## Key References

- Barz & Denzler (NeurIPS 2021) --- WikiChurches dataset
- Caron et al. (ICCV 2021) --- DINO: emergent attention segmentation properties
- Oquab et al. (2023) --- DINOv2; He et al. (CVPR 2022) --- MAE
- Chefer et al. (CVPR 2021) --- Transformer interpretability, IoU evaluation framework
- Voita et al. (ACL 2019) --- Attention head specialization in transformers
- Biderman et al. (TMLR 2024) --- LoRA learns less, forgets less

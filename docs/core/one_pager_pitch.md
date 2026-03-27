# Do Self-Supervised Vision Models Learn What Experts See?

**Attention Alignment with Human-Annotated Architectural Features**

ISY5004 Intelligent Sensing Systems Practice Project | Team Size: 3

---

## Problem

Self-supervised learning (SSL) models like DINOv2, DINOv3, MAE, CLIP, and SigLIP learn powerful visual representations without labels. But a critical question remains unanswered: **do these models attend to the same visual features that domain experts consider diagnostic?** A model that classifies Gothic churches by attending to flying buttresses is qualitatively different from one that exploits background correlations --- yet existing benchmarks measure only classification accuracy, not *where* models look.

This project fills that gap by quantitatively measuring alignment between SSL attention patterns and expert-annotated architectural features, using the WikiChurches dataset (Barz & Denzler, NeurIPS 2021). See the [Project Proposal](project_proposal.md) for the full research design.

## Research Questions

| # | Question | Method |
|---|----------|--------|
| **Q1** | Do SSL models attend to expert-identified diagnostic features? | IoU, MSE, KL divergence, and EMD between attention maps and 631 expert bounding boxes across 7 models and 12 layers |
| **Q2** | Does fine-tuning shift attention toward expert features, and does the strategy matter? | Compare delta-IoU (fine-tuned minus frozen) across Linear Probe vs LoRA vs Full fine-tuning; classify outcomes as Preserve, Enhance, or Destroy using paired Wilcoxon tests with Holm correction |
| **Q3** | Do individual attention heads specialize for different architectural features? | Per-head IoU against expert bounding boxes; rank-based analysis to identify heads consistently aligned with specific feature types *(planned --- see Roadmap)* |

## Approach

**Dataset:** WikiChurches --- 9,485 church images (official release), 631 expert bounding boxes across 139 annotated churches in 4 architectural styles (Gothic, Romanesque, Renaissance, Baroque).

**Models compared** (all ViT-B for controlled comparison, except ResNet-50):

| Model | Paradigm | Key Property |
|-------|----------|------------|
| DINOv2 | Self-distillation | Emergent object segmentation |
| DINOv3 | Self-distillation + Gram anchoring | Improved dense features |
| MAE | Masked autoencoding | Pixel-level reconstruction |
| CLIP | Contrastive VLM (softmax) | Language-image semantic alignment |
| SigLIP | Contrastive VLM (sigmoid) | No CLS token; sigmoid loss |
| SigLIP 2 (`siglip2`) | Contrastive VLM (sigmoid) | Dense feature quality |
| ResNet-50 | Supervised (Grad-CAM) | CNN baseline |

**Methodology:**
1. Extract attention maps (CLS attention, rollout, mean attention for SigLIP/SigLIP 2, and Grad-CAM) from pretrained models (see [Attention Methods Guide](../research/attention_methods.md) for method details)
2. Compute alignment metrics: threshold-dependent IoU at multiple percentiles, plus threshold-free MSE, KL divergence (KL(GT‖attention)), and EMD against Gaussian soft-union ground truth derived from expert bounding boxes
3. Fine-tune on 4-class style classification (~4,790 images) using three strategies: Linear Probe, LoRA, and Full fine-tuning, with one shared non-annotated validation split per experiment batch and the 139 annotated images reserved for final attention evaluation
4. Re-extract attention and measure delta-IoU to classify each model-strategy combination as Preserve (Δ ≈ 0, not significant), Enhance (Δ > 0, significant), or Destroy (Δ < 0, significant) using paired Wilcoxon tests with Holm correction

## Results

**Result provenance:** Q1 frozen metrics come from `outputs/cache/metrics_summary.json`. Publication-safe Q2 numbers should come from the active experiment's `outputs/results/experiments/<experiment_id>/q2_metrics_analysis.json` via `outputs/results/active_experiment.json`. Do not quote legacy top-level `q2_delta_iou_analysis.json` as the primary result source.

### Q1: Frozen Model Leaderboard

| Model | Best IoU (frozen) | Rank | Key Observation |
|-------|:-:|:-:|-----------------|
| DINOv3 | 0.133 | 1 | Self-distillation leads; best at final layer |
| ResNet-50 | 0.090 | 2 | Supervised CNN competitive via Grad-CAM |
| DINOv2 | 0.082 | 3 | Strong semantic attention in later layers |
| CLIP | 0.049 | 4 | Best at layer 0 --- attention degrades deeper |
| SigLIP | 0.047 | 5= | Mid-layer peak, no CLS token |
| SigLIP 2 | 0.047 | 5= | Identical frozen IoU to SigLIP v1 |
| MAE | 0.037 | 7 | Near-uniform across layers; reconstruction ≠ localization |

**Paradigm ranking (frozen):** Unimodal self-distillation (DINOv3 > DINOv2) > supervised baseline (ResNet-50) > multimodal VLMs (CLIP ≈ SigLIP ≈ SigLIP 2) > reconstruction (MAE). Language-image alignment does not improve localization of architectural features.

### Q2: Fine-Tuning Effects --- Preserve / Enhance / Destroy

The interpretation framework is unchanged:

- **Enhance**: fine-tuning moves attention toward expert-marked regions
- **Preserve**: attention stays effectively unchanged
- **Destroy**: fine-tuning moves attention away from expert-marked regions

The concrete category assignments should be regenerated from the active
experiment batch before they are quoted in slides or writeups. For strategy
rationale and hyperparameters, see [Fine-Tuning Methods](../enhancements/fine_tuning_methods.md) and [Run Matrix](../reference/fine_tuning_run_matrix.md).

## Novelty

1. **Multi-paradigm attention benchmark:** First study to compare 7 models spanning 4 SSL paradigms plus a supervised baseline on the same expert-annotated architectural dataset, using both threshold-dependent (IoU) and threshold-free (MSE, KL, EMD) metrics.
2. **Preserve / Enhance / Destroy taxonomy:** Systematic classification of how fine-tuning strategies interact with pre-training paradigms to shift attention alignment. The finding that contrastive VLMs Enhance while self-distillation models Preserve is new. This addresses the open question raised by Chung et al. (2025, medical imaging), whose finding that "DINO does NOT necessarily outperform supervised/MAE on medical data" strengthens the case for domain-specific evaluation.
3. **Delta-IoU methodology:** Paired statistical comparison of attention alignment before and after fine-tuning with Holm-corrected significance testing and effect sizes --- unexplored in prior literature.

Future extension (Q3) would apply Voita et al.'s (ACL 2019) head specialization framework to vision transformers with domain-specific ground truth --- no existing work computes per-head IoU against expert-annotated architectural features.

## Alignment with ISY5004

| Requirement | How Addressed |
|-------------|---------------|
| Intelligent sensing technique | SSL visual feature learning (6 ViT models + CNN baseline) |
| Image analytics | Attention extraction, feature attribution, localization evaluation |
| Dataset handling | Public dataset with preprocessing, augmentation, data cleaning |
| Experimental comparison | Ablation across 7 models, 4 attention methods, 5 metrics (IoU + Coverage + MSE + KL + EMD), 3 fine-tuning strategies |
| System deliverable | Reproducible experimental pipeline with precomputed metrics and interactive analysis tool |

## FAQ

**What metrics are used and why?**

**Threshold-dependent:** IoU (Intersection over Union) is the standard localization metric used in prior attention-interpretability work (Chefer et al., CVPR 2021). It measures spatial overlap between thresholded attention maps and expert bounding boxes. We sweep 5 percentiles (50th--90th) and report best-layer IoU per model at the default 90th percentile.

**Threshold-free:** Coverage measures what fraction of total attention *energy* falls inside annotated regions. MSE compares the normalized attention map against a Gaussian soft-union ground truth derived from expert bounding boxes. KL divergence (KL(GT‖attention)) measures distributional mismatch, heavily penalizing attention in non-expert regions. EMD (Earth Mover's Distance / Wasserstein-1) measures the minimum spatial transport cost between distributions, accounting for *distance* --- a model attending slightly beside a feature is penalized less than one attending to an entirely different region. Together, these five metrics distinguish "looking at the right place" (IoU) from "how attention mass is distributed" (Coverage, MSE, KL, EMD). For detailed definitions, thresholding methodology, worked examples, and known limitations, see the [Metrics Methodology Reference](../reference/metrics_methodology.md).

**Metrics change when bounding boxes are selected. What happens under the hood?**

By default, the IoU and coverage scores are computed against the **union of all bounding boxes** for a given image --- a single merged mask representing every expert-annotated feature. When a user selects an individual bounding box in the Annotations panel, the frontend calls a dedicated API endpoint (`GET /metrics/{image_id}/bbox/{bbox_index}`) that computes metrics **on-the-fly for that single bbox only**. The backend loads the cached attention tensor from HDF5, generates a mask from just the selected bbox's coordinates, and computes IoU and coverage against that single-bbox mask instead of the union. This lets users see whether the model attends to a specific architectural feature (e.g., a rose window) versus the aggregate. Deselecting reverts instantly to the precomputed union-of-all-bboxes metric with no re-fetch.

**What is the project scope?**

The project answers three research questions: (Q1) whether frozen SSL models attend to expert-identified features, measured by IoU and continuous metrics across 7 models and 12 layers; (Q2) whether fine-tuning shifts attention toward those features, classifying outcomes as Preserve, Enhance, or Destroy across Linear Probe, LoRA, and Full fine-tuning with Holm-corrected paired statistical tests; and (Q3, planned) whether individual attention heads specialize for different architectural feature types, via per-head IoU analysis. The deliverable is a reproducible experimental pipeline with precomputed metrics and an interactive visualization tool for exploring attention alignment across models, layers, and individual features. Out of scope: real-time inference, deployment, transfer to non-architectural domains, and attention-supervised training objectives (e.g., core-tuning, COIN).

## Roadmap

| Item | Status | Description |
|------|--------|-------------|
| Q3: Per-head specialization | Planned | Per-head attention extraction exists; full IoU pipeline and statistical analysis pending ([design doc](../enhancements/per_attention_head.md)) |
| Cross-layer aggregation | Planned | Max pooling, depth-weighted mean, ALTI as alternatives to single-layer analysis (Prof. feedback #3) |
| Unimodal vs Multimodal dashboard | Partial | Paradigm grouping applied in analysis; formal dashboard split pending (Prof. feedback #4) |

## Key References

- Barz & Denzler (NeurIPS 2021) --- WikiChurches dataset
- Caron et al. (ICCV 2021) --- DINO: emergent attention segmentation properties
- Oquab et al. (2023) --- DINOv2; He et al. (CVPR 2022) --- MAE
- Chefer et al. (CVPR 2021) --- Transformer interpretability, IoU evaluation framework
- Voita et al. (ACL 2019) --- Attention head specialization in transformers
- Biderman et al. (TMLR 2024) --- LoRA learns less, forgets less

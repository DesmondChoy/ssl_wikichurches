# Do Self-Supervised Vision Models Learn What Experts See?

**Attention Alignment with Human-Annotated Architectural Features**

ISY5004 Intelligent Sensing Systems Practice Project | Team Size: 3

---

## Problem

Self-supervised learning (SSL) models such as DINOv2, DINOv3, MAE, CLIP, SigLIP, and SigLIP 2 learn strong visual representations without dense human supervision. The central question for this project is whether those models attend to the same architectural regions that experts mark as diagnostic, or whether they achieve good downstream performance by leaning on shortcuts that are difficult to justify visually.

This repository answers that question with a quantitative benchmark over expert annotations, a fine-tuning analysis pipeline, and an interactive app for inspecting frozen-model behavior, attention shift after fine-tuning, and Q3 per-head specialization.

## Research Questions

| # | Question | Current method surface |
|---|----------|------------------------|
| **Q1** | Do SSL models attend to expert-identified diagnostic features? | IoU, Coverage, MSE, KL, and EMD across 7 models and 12 layers, with dashboard summaries and image-level drill-down |
| **Q2** | Does fine-tuning shift attention toward expert features, and does the strategy matter? | Frozen-vs-fine-tuned deltas across Linear Probe, LoRA, and Full fine-tuning, summarized through the active experiment |
| **Q3** | Do individual attention heads specialize for different architectural features? | Per-head rankings, head-by-feature heatmaps, exemplars, Image Detail Q3 drill-down, and `/q3-report` |

## Approach

**Dataset:** WikiChurches, using the 139-image expert-annotated subset for attention-alignment evaluation and the style-labeled pool from `churches.json` for fine-tuning.

**Models compared:** DINOv2, DINOv3, MAE, CLIP, SigLIP, SigLIP 2, and ResNet-50.

**Methodology:**

1. Extract attention maps with model-appropriate methods: CLS attention, rollout, mean attention for SigLIP-family models, and Grad-CAM for ResNet-50.
2. Measure alignment with expert annotations using IoU, Coverage, MSE, KL, and EMD.
3. Fine-tune the ViT backbones with Linear Probe, LoRA, and Full strategies while keeping the 139 annotated images out of the primary train/validation split.
4. Summarize Q2 through experiment-scoped artifacts selected by `outputs/results/active_experiment.json`.
5. Analyze Q3 through per-head metrics, head-feature heatmaps, representative exemplars, and shared-context image drill-down.

## Current Results Surface

**Result provenance:** Q1 frozen-model summaries come from the cache-backed frozen metrics pipeline. Q2 summaries come from the active experiment's `q2_metrics_analysis.json`, selected through `outputs/results/active_experiment.json`. Image-level Q2 deltas live in `q2_delta_iou_analysis.json`; the active experiment is the primary source for app, reporting, and documentation workflows.

### Q1

- Frozen-model ranking, layer progression, style breakdown, feature breakdown, and continuous-metric baselines live on **Dashboard Overview**.
- The current report-ready baseline artifacts are:
  - `outputs/results/q1_continuous_baseline_summary.md`
  - `outputs/results/q1_continuous_baseline_comparison.json`

### Q2

- Strategy-aware attention-shift summaries live on **`/q2`**.
- Compare flows let you inspect the same model as `Frozen`, `Linear Probe`, `LoRA`, or `Full Fine-tune` on one typed or selected image.
- Frozen-vs-adapted shift maps render `adapted - frozen` from cached numeric heatmaps.
- Q2 investigation artifacts cover per-style deltas, cross-model image-level correlations, and MAE Renaissance feature-level deltas.
- The canonical artifact is `outputs/results/experiments/<experiment_id>/q2_metrics_analysis.json`.

### Q3

- **Dashboard Q3** is the main discovery surface for per-head specialization.
- **Image Detail Q3** is the single-image drill-down surface for inspecting one selected head or head-feature context.
- **`/q3-report`** is the report-facing surface for head rankings, head-feature matrices, and frozen-to-adapted delta views.

## Novelty

1. **Multi-paradigm attention benchmark:** one evaluation surface comparing 7 models spanning self-distillation, masked autoencoding, multimodal contrastive training, and a supervised CNN baseline.
2. **Multi-metric attention alignment:** one methodology combining threshold-dependent overlap with threshold-free distributional metrics.
3. **Strategy-aware attention-shift analysis:** one Q2 workflow comparing Linear Probe, LoRA, and Full fine-tuning through shared evaluation images and active-experiment provenance.
4. **Per-head architectural feature analysis:** one Q3 workflow connecting per-head metrics, feature heatmaps, exemplars, and image-level inspection.

## Alignment with ISY5004

| Requirement | How the project addresses it |
|-------------|------------------------------|
| Intelligent sensing technique | SSL visual feature learning across multiple vision backbones |
| Image analytics | Attention extraction, feature attribution, localization evaluation, and similarity inspection |
| Dataset handling | Public dataset ingestion, preprocessing, annotation handling, and style-label curation |
| Experimental comparison | Cross-model, cross-method, and cross-strategy comparisons over shared evaluation images |
| System deliverable | Reproducible training/analysis pipeline with an interactive web application |

## FAQ

**What does the app cover?**

The app covers Gallery browsing, Image Detail inspection, model and variant comparison, Dashboard Q1 summaries, the active-experiment-backed Q2 page, and the Q3 per-head workflow across Dashboard, Image Detail, and `/q3-report`.

**Which artifacts are primary?**

Primary operational sources are:

- `outputs/results/active_experiment.json`
- `outputs/results/experiments/<experiment_id>/q2_metrics_analysis.json`
- `outputs/results/experiments/<experiment_id>/run_matrix.json`
- `outputs/results/q1_continuous_baseline_summary.md`
- `outputs/results/q1_continuous_baseline_comparison.json`

**What remains future-facing?**

Future work is centered on broader robustness and presentation layers rather than on enabling the core Q1/Q2/Q3 product surfaces. Examples include alternate cross-layer aggregation schemes, stronger paradigm-group dashboards, entropy or representation-similarity diagnostics, and negative-control experiments.

## Roadmap

| Item | Status | Description |
|------|--------|-------------|
| Cross-layer aggregation | Planned | Compare single-layer analysis against alternatives such as max pooling, depth-weighted means, or ALTI-style aggregation |
| Unimodal vs multimodal dashboard framing | Partial | Paradigm grouping is reflected in analysis and writing; a first-class dashboard split remains optional product polish |
| Representation-shift diagnostics | Planned | Run deeper representation-similarity or forgetting analyses alongside the current Q2 attention-shift outputs |

## Key References

- Barz & Denzler (NeurIPS 2021) — WikiChurches dataset
- Oquab et al. (2023) — DINOv2
- He et al. (CVPR 2022) — MAE
- Chefer et al. (CVPR 2021) — Transformer interpretability and IoU-style evaluation
- Voita et al. (ACL 2019) — Attention head specialization
- Biderman et al. (TMLR 2024) — LoRA learns less and forgets less

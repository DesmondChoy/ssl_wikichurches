# Slide Content — Final Video Presentation

Slide-by-slide content for the 15-minute recorded presentation.
Demo clips are inserted between segments in post-production — slides marked **[DEMO CLIP]** are placeholders for those cuts.

---

## SEGMENT 1 — Hook, Problem Framing & Results Summary (2 min)

---

### Slide 1 — Title

**Do Self-Supervised Vision Models Learn What Experts See?**

Attention Alignment with Human-Annotated Architectural Features

ISY5004 Intelligent Sensing Systems Practice Project
Team: Leong Kay Mei, Desmond Choy

---

### Slide 2 — The Core Question

**A model predicts "Gothic" correctly. But is it looking at the right thing?**

[VISUAL: Two-panel — church image with expert bounding boxes on left, raw attention heatmap on right. Use the Motivation & Research Gap image from the mid-term deck: the Romanesque facade with Round Arch Window box highlighted.]

> Self-supervised vision models achieve strong downstream accuracy — but accuracy alone tells us nothing about *where* in the image the model is looking.

---

### Slide 3 — What We Studied

**139 annotated church images · 631 expert bounding boxes · 7 vision models · 3 research questions**

- **Q1** Do frozen SSL models attend to the same regions experts mark as diagnostically important?
- **Q2** Does fine-tuning shift attention toward those regions, and does the strategy matter?
- **Q3** Do individual attention heads specialise for different architectural features?

---

### Slide 4 — Headline Findings (Abstract)

**What we found — before the details:**

| | Finding |
|---|---|
| **Q1** | Frozen expert-aligned attention exists, but is model-family dependent. **DINOv3 leads** across all four metrics and is the only model to clear all calibration baselines. SigLIP has the best MSE but worse-than-random EMD — a warning against single-metric reading. |
| **Q2** | Fine-tuning's effect is driven by three factors: **spatial prior** (DINO preserves), **linguistic coverage** (CLIP gains only on Gothic/Romanesque), and **geometric discriminability** (MAE gains on Renaissance pediments). Models converge on the same easy images (r = +0.677) — ensembling the language cluster adds no coverage on hard images. |
| **Q3** | Individual heads show descriptive specialisation for different feature types; the dominant head set shifts under adaptation. |

---

## SEGMENT 2 — Dataset & Methodology (2 min)

---

### Slide 5 — The Gap We Address

**Existing SSL benchmarks measure accuracy. We measure *what the model looks at*.**

- Tools like BertViz and Captum visualise attention — but don't quantitatively align it against human expert diagnoses
- Current work in medical imaging is a start, but lacks cross-paradigm comparison or fine-tuning analysis
- **Our contribution**: a multi-metric quantitative benchmark grounded in expert-annotated architectural features

> Professor feedback shaped this methodology: continuous metrics (MSE, KL, EMD) were added following the suggestion to apply Gaussian filtering to bounding boxes; the Preserve/Enhance/Destroy taxonomy came from feedback to study fine-tuning's effect on attention consistency.

---

### Slide 6 — Dataset: WikiChurches

**WikiChurches** (Barz & Denzler, NeurIPS 2021)

[VISUAL: Funnel diagram — 9,485 → 4,588 → 631 → 139. Or reuse mid-term deck slide 4 layout.]

| | |
|---|---|
| Full dataset | 9,485 images |
| Used for fine-tuning | 4,588 images |
| Expert bounding boxes | 631 (across 106 feature types) |
| Held-out evaluation images | 139 (strictly excluded from training) |
| Architectural styles | Romanesque · Gothic · Renaissance · Baroque |

Bounding boxes annotate *characteristic* architectural features — the diagnostic visual evidence experts use to distinguish styles.

---

### Slide 7 — Pipeline Overview

**Technical Approach: SSL Attention × Expert Alignment**

[VISUAL: 6-box pipeline diagram from mid-term deck slide 7.]

1. **Dataset** — WikiChurches, 139 annotated eval images
2. **7 Vision Models** — across 4 SSL paradigms + supervised baseline
3. **Attention Extraction** — CLS, Rollout, Mean, Grad-CAM
4. **Alignment Metrics** — IoU, Coverage, MSE, KL, EMD vs. expert boxes
5. **Fine-Tuning (Q2)** — Linear Probe, LoRA, Full; 139 images held out
6. **Analysis** — Frozen benchmark, Δ metrics, per-head ranking

---

### Slide 8 — 7 Models Across Paradigms

**All ViT-B (12 layers, 768 dim, 12 heads, ~86–93M params) except ResNet-50**

[VISUAL: Model table from mid-term deck slide 8.]

| Model | Paradigm | Method | Key Feature |
|---|---|---|---|
| DINOv2 | Self-distillation | CLS, Rollout | 4 register tokens |
| DINOv3 | Self-distillation + Gram Anchoring | CLS, Rollout | RoPE encoding |
| MAE | Masked Autoencoding | CLS, Rollout | Pixel reconstruction |
| CLIP | Contrastive (softmax) | CLS, Rollout | Language-image align |
| SigLIP | Contrastive (sigmoid) | Mean | No CLS token |
| SigLIP 2 | Contrastive (sigmoid) + dense features | Mean | No CLS token |
| ResNet-50 | Supervised (ImageNet) | Grad-CAM | CNN baseline |

---

### Slide 9 — Metrics Used to Measure Alignment

**5 complementary metrics across 2 ground truth types**

[VISUAL: Two-column layout matching mid-term deck slide 10.]

**Binary Ground Truth** *(hard bounding box mask)*
- **IoU** — spatial overlap of top-k% attention pixels vs. expert boxes. Higher = better.
- **Coverage** — fraction of total attention energy inside boxes. No threshold needed. Higher = better.

**Soft Gaussian Ground Truth** *(smooth falloff from box centre)*
- **MSE** — pointwise intensity error. Lower = better.
- **KL Divergence** — distribution mismatch; penalises attention where experts assign none. Lower = better.
- **EMD** — Earth Mover's Distance; spatial transport cost, distinguishes near-misses from far-misses. Lower = better.

**Calibrated against 4 naive baselines**: random · center Gaussian · saliency prior · Sobel edge
Raw scores are uninterpretable without reference points.

---

## SEGMENT 3 — Q1: Frozen Model Benchmark (2.5 min)

---

### Slide 10 — Q1: Frozen Leaderboard

**Q1: Expert-Attention Alignment — Frozen Models**

Paradigm ordering: **Self-distillation > Supervised > Reconstruction > Multimodal Contrastive**

| Rank | Model | IoU@90 | Coverage | KL | EMD |
|---|---|---|---|---|---|
| 1 | **DINOv3** | **0.133** | **0.137** | **2.325** | **0.260** |
| 2 | ResNet-50 | 0.090 | 0.104 | 2.692 | 0.303 |
| 3 | DINOv2 | 0.082 | 0.100 | 2.684 | 0.298 |
| 4 | MAE | 0.070 | 0.090 | 2.756 | 0.318 |
| 5 | CLIP | 0.049 | 0.085 | 2.912 | 0.326 |
| 6 | SigLIP | 0.047 | 0.071 | 3.071 | 0.354 |
| 7 | SigLIP 2 | 0.047 | 0.071 | 3.071 | 0.354 |

*Ranked by IoU@90 (best default-method layer per model, 139 annotated images)*

---

### Slide 11 — Q1: Why Raw Scores Aren't Enough

**Calibrated baseline clearance — the stronger test**

| Baseline | MSE | KL | EMD |
|---|---|---|---|
| Random | 0.319 | 3.363 | 0.347 |
| Center Gaussian | 0.177 | 2.632 | 0.284 |
| Saliency Prior | 0.096 | 2.611 | 0.265 |
| Sobel Edge | 0.038 | 3.224 | 0.314 |

**DINOv3** is the only frozen model to beat all 4 naive baselines on all 3 continuous metrics.

**SigLIP / SigLIP 2** have the best MSE (0.0175) — but EMD of 0.354 is *worse than random* (0.347).
→ Local smoothness ≠ correct spatial placement of attention mass.

---

### Slide 12 — Q1: Why DINOv3 Leads

**Hypothesis: Gram anchoring preserves dense spatial structure**

DINOv3 extends DINOv2's self-distillation recipe with a **Gram anchoring loss** — it explicitly penalises Frobenius drift between student and early-teacher patch-patch feature matrices during long training.

- DINOv3 reaches frozen ADE20k mIoU of **81.1** vs DINOv2's 75.9 and OpenCLIP's 63.8
- A method designed to preserve clean dense patch features is a natural fit for metrics that reward spatial correspondence to expert boxes

> This remains a hypothesis — the current study does not ablate data scale, model scale, and Gram anchoring separately.

---

### [DEMO CLIP] — App Dashboard: Q1 KL Leaderboard

*[Insert ~30 sec clip: Dashboard with Metric=KL, Ranking=Default method, showing DINOv3 ranked first with baseline overlays visible.]*

---

## SEGMENT 4 — Q2: Fine-Tuning Effects on Attention (4 min)

---

### Slide 13 — Q2: Experiment Design

**Does fine-tuning shift attention toward expert-marked regions?**

**3 strategies**, same 139 held-out evaluation images across all runs:

| Strategy | Backbone | Params | Role |
|---|---|---|---|
| Linear Probe | Frozen | ~3K | Control — attention unchanged |
| LoRA | Partially adapted | ~300K | Parameter-efficient adaptation |
| Full Fine-tuning | Updated end-to-end | 86–93M | Maximum adaptation capacity |

Training: 4-class style classification · 4,588 images · 3 epochs · Cosine LR + warmup · Class-weighted loss
Measurement: Δ metric = fine-tuned − frozen · Paired Wilcoxon tests · Holm correction · Cohen's d

**Linear Probe Δ ≈ 0 across all models** — confirms attention movement is tied to backbone change, not a reporting artifact.

---

### Slide 14 — Q2: Overview Heatmap

**Multi-metric improvement heatmap — the big picture**

[VISUAL: `outputs/figures/02_all_metrics_improvement_heatmap.png`]

- **Blue** = improvement · **Red** = degradation · **Asterisk** = statistically significant
- Language cluster (CLIP, SigLIP, SigLIP 2): strong enhancement
- MAE: strong enhancement
- DINO family: near-zero, occasionally negative
- Linear Probe column: all zero by construction ✓

Summary: **46 enhance · 16 preserve · 10 destroy** across 72 non-linear-probe model-strategy-metric combinations.

---

### Slide 15 — Q2: The CLIP Story

**CLIP: Largest gain, but only where language already pointed**

- Full fine-tuning: IoU 0.018 → **0.074** (Cohen's d ≈ 1.0)
- LoRA also improves substantially, but less than Full

**But the gain is not uniform:**

| Style | CLIP Δ IoU (Full FT) |
|---|---|
| Gothic (n=49) | **+0.079** |
| Romanesque (n=54) | **+0.066** |
| Renaissance (n=22) | +0.014 |
| Baroque (n=17) | +0.013 |

**Why**: CLIP is trained on 400M web image–text pairs with a global contrastive loss — no patch-level spatial pressure. Fine-tuning is the first spatial signal. Gains concentrate on Gothic and Romanesque because these styles' diagnostic features (Pointed Arch Portals, Round Arch Portals, Bull's-eye Windows, Tracery) are densely represented in English-language web descriptions of churches.

---

### Slide 16 — Q2: The MAE Story

**MAE: Renaissance pediment geometry — gains from scratch**

MAE's largest single-style gain: Renaissance **Δ IoU = +0.108** (vs. +0.029 aggregate)

| Feature | Frozen IoU | FT IoU | Δ IoU |
|---|---|---|---|
| Triangular Pediment | 0.036 | 0.116 | **+0.080** |
| Broken Pediment | 0.005 | 0.060 | **+0.055** |
| Cranked Cornice | 0.005 | 0.067 | **+0.062** |
| Volute | 0.009 | 0.052 | **+0.043** |
| Pilaster (common) | 0.023 | 0.011 | −0.012 |
| Belt Course (common) | 0.031 | 0.016 | −0.016 |

[VISUAL: `outputs/results/experiments/fine_tuning_primary_20260327/feature_delta_iou_mae_full_renaissance.png`]

**Why**: MAE's 75% masking forces precise local geometry encoding. Pediments are geometrically compact and Renaissance-exclusive — the style-classification gradient routes attention to the most discriminative geometric forms, suppressing features that appear across multiple styles.

---

### Slide 17 — Q2: The DINO Story

**DINO: Near-zero Δ is a feature, not a failure**

[VISUAL: Per-style Δ IoU table — `outputs/results/experiments/fine_tuning_primary_20260327/style_breakdown.png`]

| | Romanesque | Gothic | Renaissance | Baroque |
|---|---|---|---|---|
| DINOv2 | −0.010 | +0.001 | −0.004 | −0.012 |
| DINOv3 | −0.001 | +0.006 | −0.004 | −0.009 |

DINOv3 frozen ADE20k mIoU = 81.1. Expert-aligned spatial structure is already baked in.

The style-classification gradient is absorbed by the classification head rather than propagating to reshape attention. Δ ≈ 0 is a positive generalisation result — the strong frozen prior resists reorganisation.

---

### Slide 18 — Q2: The Surprise — Models Converge on the Same Images

**DINOv3 frozen IoU predicts CLIP Δ IoU at r = +0.677**

[VISUAL: `outputs/results/experiments/fine_tuning_primary_20260327/model_correlation_scatter.png`]

Images where DINOv3 already attends correctly are the *same images* where CLIP's fine-tuning succeeds.

**The structural barrier is the image, not the model family.**

Three clusters in the pairwise Δ correlation matrix:
- **Language cluster** (CLIP, SigLIP, SigLIP 2): within-cluster r ≈ 0.43–0.58 — all improve on the same Gothic/Romanesque portals
- **MAE**: anti-correlated with language cluster (r ≈ −0.22 to −0.31) — covers a *different* image subset (Renaissance geometry)
- **DINO pair**: near-zero correlation with everything — Δ ≈ 0 everywhere, nothing to latch onto

**Implication**: an ensemble of language-cluster models adds no coverage on hard images. MAE is the only natural complementary partner.

---

### Slide 19 — Q2: Statistical Confidence

**Forest plot — are the gains real?**

[VISUAL: `outputs/figures/08_forest_plot_ci.png`]

Mean Q2 Δ with 95% bootstrap CIs for LoRA and Full fine-tuning across all 5 metrics (sign-normalised — rightward = improvement). Asterisks = significant after Holm correction.

Key takeaway: CLIP and MAE gains are not anecdotal — they hold under paired Wilcoxon tests across the 139 evaluation images.

---

### [DEMO CLIP] — App Compare View: CLIP Frozen vs. Fine-Tuned

*[Insert ~30 sec clip: Compare view for Q1710328 (Gothic portal). Show frozen CLIP attention vs. full fine-tuned CLIP attention, with shift map overlay. Expert bounding box visible.]*

---

## SEGMENT 5 — Q3: Per-Head Specialisation (1.5 min)

---

### Slide 20 — Q3: Scope and Approach

**Q3: Do individual attention heads specialise for different architectural features?**

**Scope**: DINOv2, DINOv3, MAE, CLIP only
- CLS-token models → architecture-native per-head analysis
- SigLIP/SigLIP 2 excluded: mean-attention proxy, not architecture-native per-head
- ResNet-50 excluded: no transformer attention heads

**Method**: Per-head IoU computed for all 12 heads at each layer. Heads ranked by alignment with expert boxes. Head-by-feature matrices built to test whether specific heads consistently align with specific feature types.

**Variants**: frozen · LoRA · full (linear probe excluded — backbone unchanged)

> This is a *descriptive* head-specialisation analysis. We do not claim individual heads are causal feature detectors.

---

### Slide 21 — Q3: Findings

**Some heads align more strongly — and the dominant set shifts under adaptation**

[VISUAL: Q3 head-feature heatmap from app Q3 dashboard]

Key observations:
- Late-layer head alignment is stronger than early-layer — consistent with depth-wise progression toward more semantically meaningful spatial organisation
- A small subset of heads dominates the alignment ranking within each model (sparsity increases in later layers)
- Adaptation (LoRA, Full) shifts the dominant head set relative to frozen — fine-tuning reorganises which heads carry expert-aligned structure

> The parallel to CNN feature maps is at the level of hierarchical abstraction only — attention heads route information; they are not equivalent to convolutional feature detectors.

---

## SEGMENT 6 — Live App Demo (1.5 min)

---

### Slide 22 — App Demo Intro

**Interactive Analysis Interface**

A FastAPI backend + React frontend exposing the full study:

- **Gallery** — browse the 139 annotated images, filter by style
- **Image Detail** — per-image attention heatmap with expert bounding box overlay; switch model, layer, method
- **Dashboard** — multi-model leaderboard with calibrated baseline overlays
- **Compare** — frozen vs. fine-tuned side-by-side shift map for any image
- **Q2 tab** — improvement heatmap, preserve/enhance/destroy summary, per-style breakdown
- **Q3 tab** — per-head ranking, head-feature matrix

*Backed by precomputed HDF5/SQLite cache — sub-second queries across 139 images × 7 models × 12 layers × 12 heads.*

---

### [DEMO CLIP] — Full App Walkthrough

*[Insert ~90 sec clip:*
1. *Gallery → click into an image (e.g. Q1710328 Gothic portal)*
2. *Dashboard → KL leaderboard with baseline overlays, switch to IoU*
3. *Compare → Q1710328 frozen CLIP vs. full FT CLIP shift map*
4. *Q2 tab → improvement heatmap, then preserve/enhance/destroy chart]*

---

## SEGMENT 7 — Conclusions & Takeaways (1.5 min)

---

### Slide 23 — What We Found

**Three questions, three answers**

**Q1 — Frozen alignment exists, but is model-family dependent**
DINOv3 leads the benchmark and is the only frozen model to clear all calibration baselines on all continuous metrics. Frozen SSL attention is not a generic property — self-distillation with Gram anchoring produces a qualitatively different spatial prior.

**Q2 — Fine-tuning's effect is mediated by pretraining, not just strategy**
CLIP's gain is linguistic: it unlocks attention on features densely described in web text. MAE's gain is geometric: it redirects to compact, style-exclusive pediment forms. DINO's near-zero Δ is a preservation result. No single strategy wins everywhere — the right combination depends on what the task's diagnostic evidence requires.

**Q3 — Heads show descriptive specialisation; the dominant set shifts under adaptation**
Per-head alignment is sparse and late-layer dominant. Fine-tuning reorganises which heads carry expert-aligned structure.

---

### Slide 24 — Practical Implication

**Model selection for domain adaptation should not be guided by accuracy alone.**

Match the pretraining prior to what the task's diagnostic evidence requires:

- If evidence is spatially coherent and structured → **DINOv3 frozen** may already be sufficient
- If evidence is linguistically well-described → **CLIP + Full fine-tuning** unlocks the largest gain
- If evidence is geometrically compact and style-exclusive → **MAE** covers a complementary image subset
- For coverage across hard images → **MAE + language-cluster ensemble** (not language-cluster alone)

---

### Slide 25 — Thank You

**Do Self-Supervised Vision Models Learn What Experts See?**

*Sometimes. It depends on how they were trained — and what the task asks them to look at.*

Team: Leong Kay Mei, Desmond Choy
ISY5004 Intelligent Sensing Systems Practice Project

---

## Slide Count Summary

| Segment | Slides |
|---|---|
| 1. Hook & results summary | 1–4 (4 slides) |
| 2. Dataset & methodology | 5–9 (5 slides) |
| 3. Q1 results | 10–12 + demo clip (3 slides) |
| 4. Q2 results | 13–19 + demo clip (7 slides) |
| 5. Q3 results | 20–21 (2 slides) |
| 6. App demo | 22 + demo clip (1 slide) |
| 7. Conclusion | 23–25 (3 slides) |
| **Total** | **25 slides + 3 demo clips** |

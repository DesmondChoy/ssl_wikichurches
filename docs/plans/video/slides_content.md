# Slide Content — Final Video Presentation

Slide-by-slide content for the 15-minute recorded presentation.
The app demo lives entirely in Segment 6 (3 min) — earlier segments have no inline demo clips.

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

**139 annotated church images · 631 expert bounding boxes · 7 vision models · 5 alignment metrics**

Three linked research questions:

1. **Q1** How well do frozen models align with expert-marked regions?
2. **Q2** How do Linear Probe, LoRA, and Full fine-tuning change that alignment?
3. **Q3** Do individual attention heads show descriptive specialisation for different architectural features?

---

### Slide 4 — Headline Findings (Abstract)

**What we found — before the details:**

| | Finding |
|---|---|
| **Q1** | Frozen expert-aligned attention exists, but is model-family dependent. **DINOv3 leads** on IoU@90, Coverage, KL, and EMD, and is the only frozen model to clear all calibrated continuous baselines. **Base SigLIP** is a competitive overlap result at its best layer (IoU = 0.074 at `layer8`, ahead of MAE and CLIP) — *not the same as SigLIP2* (0.047). SigLIP family also has worse-than-random EMD: a warning against single-metric reading. |
| **Q2** | Fine-tuning moves attention unevenly. **CLIP** gains the most (IoU 0.018 → 0.074, *d* ≈ 1.0) but only on Gothic/Romanesque. **MAE** gains on Renaissance pediment geometry. **SigLIP variants** improve from weaker `layer11` baselines — base SigLIP higher in absolute IoU post-FT, SigLIP2 larger in effect size. **DINO** stays flat — its strong frozen prior resists reorganisation. Models converge on the same easy images (r = +0.677). |
| **Q3** | Per-head specialisation is **sparse and family-shaped**. DINOv3 `layer10/head8` stays dominant across Frozen, LoRA, Full. MAE is partly reshaped. CLIP reorganises from an early frozen head to late adapted heads. Strongest head-feature alignment concentrates on portals, arches, and rose windows. |

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

### Slide 10 — Q1: Why a Multi-Metric Benchmark

**Attention alignment is not one thing — each metric catches a different failure mode**

- **IoU@90** — do the model's top-attended pixels land inside the expert annotations?
- **Coverage** — what fraction of total attention energy falls inside annotated regions?
- **MSE / KL / EMD** — does the full heatmap distribution match the Gaussian soft-union target?

A model can place its strongest attention inside the expert boxes, spread attention across the right facade region, or match the overall target distribution — these are related but not identical. Reporting one metric in isolation misleads.

---

### Slide 11 — Q1: Frozen Leaderboard

**Q1: Expert-Attention Alignment — Frozen Models**

Headline ordering: **Self-distillation leads, but SigLIP/SigLIP2 split sharply — they are not the same frozen result**

| Rank | Model | IoU@90 | Coverage | MSE | KL | EMD |
|---|---|---|---|---|---|---|
| 1 | **DINOv3** | **0.133** (L11) | **0.137** | 0.0270 | **2.325** | **0.260** |
| 2 | ResNet-50 | 0.090 (L3) | 0.104 | 0.0242 | 2.692 | 0.303 |
| 3 | DINOv2 | 0.082 (L11) | 0.100 | 0.0209 | 2.684 | 0.298 |
| 4 | **SigLIP** | **0.074** (L8) | 0.076 | 0.01755 | 3.002 | 0.348 |
| 5 | MAE | 0.070 (L11) | 0.090 | 0.0483 | 2.756 | 0.318 |
| 6 | CLIP | 0.049 (L0) | 0.085 | 0.0211 | 2.912 | 0.326 |
| 7 | SigLIP 2 | 0.047 (L4) | 0.071 | **0.01745** | 3.071 | 0.354 |

*Ranked by IoU@90 (best default-method layer per model, 139 annotated images). Base **SigLIP** reaches its peak at `layer8` (0.074) — ahead of MAE and CLIP. **SigLIP 2** stays weaker on overlap (0.047) despite the marginal MSE win.*

---

### Slide 12 — Q1: Why Raw Scores Aren't Enough

**Calibrated baseline clearance — the stronger test**

| Baseline | MSE | KL | EMD |
|---|---|---|---|
| Random | 0.319 | 3.363 | 0.347 |
| Center Gaussian | 0.177 | 2.632 | 0.284 |
| Saliency Prior | 0.096 | 2.611 | 0.265 |
| Sobel Edge | 0.038 | 3.224 | 0.314 |

**DINOv3** is the only frozen model to beat all 4 naive baselines on all 3 continuous metrics.

**SigLIP / SigLIP 2** have the best MSE (≈0.0175) — but EMD of 0.348–0.354 is *worse than random* (0.347).
→ Local smoothness ≠ correct spatial placement of attention mass.
→ And while SigLIP2 has the marginal MSE win, **base SigLIP holds the overlap result** (IoU@90 = 0.074 at `layer8`). Do not collapse the two variants into one family-level frozen claim.

---

### Slide 13 — Q1: Why DINOv3 Leads — Gram Anchoring

**Hypothesis: Gram anchoring preserves dense spatial structure**

DINOv3 extends DINOv2's self-distillation recipe with a **Gram anchoring loss** — it explicitly penalises Frobenius drift between student and early-teacher patch-patch feature matrices during long SSL training.

Two supporting checks:
- **Statistical robustness**: paired gaps to the next-best model are significant (Holm-adjusted *p* < 1.31 × 10⁻⁷)
- **Feature pattern**: DINOv3 aligns best with large coherent parts — *Ornate Portal* (0.214), *Tracery Rose Window* (0.164) — and worst on small ornamentation like *Crocket* and *Fleuron*. Consistent with a better dense spatial prior for prominent structure, not a complete understanding of every fine-grained cue.

> This remains a hypothesis — the current study does not ablate data scale, model scale, and Gram anchoring separately.

---

### Slide 14 — Q1: Dashboard View

**The benchmark, live in the app**

[VISUAL: `docs/final_report/figures/q1_dinov3_react_dashboard_iou_best_available.png`]

DINOv3 ranks first at layer 11 with IoU@90 = 0.133 using CLS attention. The layer-progression panel shows DINOv3's distinctive **late-layer jump** relative to the other models — alignment improves through depth, peaking at the final transformer block.

---

## SEGMENT 4 — Q2: Fine-Tuning Effects on Attention (3.5 min)

---

### Slide 15 — Q2: Experiment Design

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

### Slide 16 — Q2: Overview Heatmap

**Multi-metric improvement heatmap — the big picture**

[VISUAL: `outputs/figures/02_all_metrics_improvement_heatmap.png`]

- **Blue** = improvement · **Red** = degradation · **Asterisk** = statistically significant
- Language cluster (CLIP, SigLIP, SigLIP 2): strong enhancement
- MAE: strong enhancement
- DINO family: near-zero, occasionally negative
- Linear Probe column: all zero by construction ✓

Summary: **46 enhance · 16 preserve · 10 destroy** across 72 non-linear-probe model-strategy-metric combinations.

---

### Slide 17 — Q2: The CLIP Story

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

### Slide 18 — Q2: The MAE Story

**MAE: Renaissance pediment geometry — gains from scratch**

MAE's largest single-style gain: Renaissance **Δ IoU = +0.108** (vs. +0.029 aggregate)

| Feature | Frozen IoU | FT IoU | Δ IoU |
|---|---|---|---|
| Triangular Pediment | 0.036 | 0.116 | **+0.080** |
| Cranked Cornice | 0.005 | 0.067 | **+0.062** |
| Broken Pediment | 0.005 | 0.060 | **+0.055** |
| Volute | 0.009 | 0.052 | **+0.043** |
| Segmental Pediment | 0.013 | 0.054 | **+0.042** |
| Pilaster (common) | 0.023 | 0.011 | −0.012 |
| Belt Course (common) | 0.031 | 0.016 | −0.016 |

[VISUAL: `outputs/results/experiments/fine_tuning_primary_20260327/feature_delta_iou_mae_full_renaissance.png`]

**Why**: MAE's 75% masking forces precise local geometry encoding. Pediments are geometrically compact and Renaissance-exclusive — the style-classification gradient routes attention to the most discriminative geometric forms, suppressing features that appear across multiple styles.

---

### Slide 19 — Q2: The SigLIP Variants

**Weak `layer11` frozen baselines → adaptation sharpens, but variants differ**

Q2 is measured at `layer11` (not each model's Q1 best layer):

| | Frozen IoU@90 (L11) | Full FT IoU@90 (L11) | Δ | Cohen's *d* |
|---|---|---|---|---|
| SigLIP | 0.0364 | **0.0618** | +0.0254 | 0.604 |
| SigLIP 2 | 0.0220 | 0.0519 | **+0.0299** | **0.781** |

- Base SigLIP stays higher in **absolute** IoU@90 post-FT (0.062 vs. 0.052)
- SigLIP 2 moves further in **standardised-effect** terms — consistent with more late-layer adaptation headroom from a weaker starting point and SigLIP 2's denser grounding components
- Q2 is sharpening a weak `layer11` signal, *not* improving each model's best frozen layer (that's the Q1 best-layer story — base SigLIP peaks at `layer8`)

---

### Slide 20 — Q2: The DINO Story

**DINO: Near-zero Δ is a feature, not a failure**

[VISUAL: Per-style Δ IoU table — `outputs/results/experiments/fine_tuning_primary_20260327/style_breakdown.png`]

| | Romanesque | Gothic | Renaissance | Baroque |
|---|---|---|---|---|
| DINOv2 | −0.010 | +0.001 | −0.004 | −0.012 |
| DINOv3 | −0.001 | +0.006 | −0.004 | −0.009 |

DINOv3 frozen ADE20k mIoU = 81.1. Expert-aligned spatial structure is already baked in.

The style-classification gradient is absorbed by the classification head rather than propagating to reshape attention. Δ ≈ 0 is a positive generalisation result — the strong frozen prior resists reorganisation.

---

### Slide 21 — Q2: The Surprise — Models Converge on the Same Images

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

### Slide 22 — Q2: Statistical Confidence

**Forest plot — are the gains real?**

[VISUAL: `outputs/figures/08_forest_plot_ci.png`]

Mean Q2 Δ with 95% bootstrap CIs for LoRA and Full fine-tuning across all 5 metrics (sign-normalised — rightward = improvement). Asterisks = significant after Holm correction.

Key takeaway: CLIP and MAE gains are not anecdotal — they hold under paired Wilcoxon tests across the 139 evaluation images.

---

## SEGMENT 5 — Q3: Per-Head Specialisation (1 min)

---

### Slide 23 — Q3: Scope and Approach

**Q3: Do individual attention heads specialise for different architectural features?**

**Scope**: DINOv2, DINOv3, MAE, CLIP only (architecture-native CLS-token ViTs)
- Same extraction path: layer → head → CLS-to-patch attention map → alignment score
- SigLIP / SigLIP 2 excluded: mean-attention proxy, not native heads
- ResNet-50 excluded: no transformer heads
- Variants: frozen · LoRA · Full (linear probe excluded — backbone unchanged)

> Language throughout: *descriptive specialisation*, not causal proof.

---

### Slide 24 — Q3 View 1: Head Ranking

**Expert-aligned attention is concentrated in a small number of heads**

[VISUAL: `docs/core/assets/q3_head_ranking_transition_map.png`]

- **DINOv3**: `layer10/head8` stays top across Frozen, LoRA, Full; appears in top-3 on 110+ of 139 images in every condition — clearest stability case
- **DINOv2**: same preserved-head pattern at lower absolute alignment
- **MAE**: `layer10/head5` → `layer11/head7` under LoRA → back to `layer10/head5` under Full (partial reshaping)
- **CLIP**: frozen best heads sit in early layers (`layer4/head5` IoU@90 = **0.067**); `layer11` is weak before adaptation (max **0.034**) but becomes the strongest adapted layer — best `layer11` head reaches **0.084 (LoRA)** and **0.105 (Full)**, while `layer4/head5` stays near its frozen score

**Quantitative sparsity** (across the full 12 × 12 layer×head grid):

| Model | Top pair vs. median | Top-5 share of total mean IoU@90 |
|---|---|---|
| DINOv3 | **3.45×** | within 7.3–9.0% band |
| CLIP | **3.22×** | (≈ 2× the 3.5% uniform baseline) |
| MAE | **2.92×** | |
| DINOv2 | **2.69×** | |

Top-5 pairs together hold **7.3%–9.0%** of total mean IoU@90 across all 144 pairs — roughly **2×** what they would hold under uniform spread (3.5%). This is the quantitative content of "sparse".

---

### Slide 25 — Q3 View 2a: Head-Feature Matrix (DINOv3 anchor)

**Linking the ranking head to its strongest feature**

[VISUAL: `docs/core/assets/q3_head_feature_matrix_report_view.png`]

The same `layer10/head8` that leads the DINOv3 ranking also carries the selected matrix cell:
- *Columned Portal* — **IoU@90 = 0.215** (n = 15)
- *Round Arch Portal* — **0.204**
- *Ornate Portal* — **0.181**

This links the ranking evidence to the feature evidence: the dominant head is not only strong overall — it is strongest on **portal-scale structure**.

---

### Slide 26 — Q3 View 2b: Top vs. Bottom Features Across Models

**Strongest features are structural; failures are consistent**

[VISUAL: `docs/core/assets/q3_head_feature_top_bottom_contact_sheet.png`]

For each scoped model's frozen IoU@90 dominant head, top-3 and bottom-3 features (≥3 annotations):

| Model · Head | Strongest | Weakest |
|---|---|---|
| DINOv3 `L10/H8` | Columned Portal, Round Arch Portal, Ornate Portal | Blind Tracery, Crocket, Tabernacle (≈ 0) |
| DINOv2 `L11/H11` | Portal-class (same top end) | Same small-detail boundary |
| MAE `L10/H5` | Large facade parts + **Wimperg** (3rd) | — |
| CLIP `L4/H5` | Portal-class + **Belt Course** | — |

- **MAE → Wimperg** fits the Q2 finding that MAE responds to compact geometric forms
- **CLIP → Belt Course** suggests feature *extent* and clean geometry matter, not only semantic category names
- **Failures aren't random** — across families, weak features are small, thin, repeated, or visually entangled with surrounding masonry

**Caveat to state aloud — metric geometry**: IoU@90 keeps a fixed top-10% attention mask and scores it against a variable-size expert mask. For thin/repeated ornamentation (Crocket, Fleuron, Blind Tracery), the achievable IoU@90 is mechanically capped regardless of where attention lands. Part of the weak-feature shortfall is metric geometry, not head attribution — threshold-free Coverage would help separate the two effects.

**Safe claim**: dominant heads expose spatial patterns compatible with expert-marked **structural parts** — not exact ornament detectors.

---

### Slide 27 — Q3 View 3: Frozen-to-Adapted Delta

**Sharper follow-up: when adaptation changes the dominant head, does the strongest signal stay or move?**

[VISUAL: `docs/core/assets/q3_frozen_adapted_delta_report_view.png`]

**Definition** (state aloud): *expert-aligned signal* = the selected head's normalised CLS-to-patch heatmap scored against expert boxes. For IoU@90, the pipeline keeps the top 10% of heatmap pixels and compares against the expert-box union; the strongest head is the one with the highest mean IoU@90 across the 139 images for that (layer, variant, metric).

**Family-specific answer:**

- **DINO** (stable): DINOv3 `layer10/head8` and DINOv2 `layer11/head11` remain strongest across Frozen, LoRA, Full
- **MAE** (intermediate): Frozen and Full both favour `layer10/head5`; LoRA shifts to `layer11/head7` — parameter-efficient adaptation can move the dominant signal, but not a CLIP-style rewrite
- **CLIP** (clearest reorganisation):
  - **LoRA** within `layer11`: `H11` promoted **#6 → #1**; frozen-best `H4` only falls to **#3**, score **0.080 vs. 0.084** — a new top head without erasing the frozen pattern
  - **Full** within `layer11`: stronger rewrite — `H3` moves **#8 → #1**, `H8` moves **#7 → #2**, `H4` drops to **#4**

**CLIP synthesis**: adaptation strengthens expert alignment in `layer11` (where frozen CLIP is weak, **0.034 max**) *and* reorganises the within-layer ranking of those newly strengthened heads.

**Connecting Q3 back to Q1 and Q2**: DINO already aligns well before fine-tuning, so adaptation leaves its strongest heads mostly unchanged. CLIP and MAE have more room to move — after adaptation, the best-aligned attention comes from **different heads**, not just the same head with a higher score. Q3 gives a head-level explanation for the Q2 result.

---

## SEGMENT 6 — Live App Demo (3 min)

---

### Slide 28 — App Demo Intro

**Interactive Analysis Interface**

*"All of this is backed by precomputed HDF5/SQLite cache — sub-second queries across 139 images × 7 models × 12 layers. Let me show you what that looks like in practice."*

A FastAPI backend + React frontend exposing the full study — six views, each tied to a finding from the talk.

---

### [DEMO CLIP] — Full App Walkthrough (~3 min)

*[End-to-end screen recording covering:*

1. *Gallery → browse the annotated image set; click into Q1710328 (Gothic) to show image detail with expert boxes overlaid*
2. *Image Detail → DINOv3 frozen CLS attention at layer 11 alongside the expert boxes — this is what Q1 measures*
3. *Dashboard → IoU@90 best-available leaderboard with DINOv3 at the top and the layer-progression panel showing the late-layer jump; switch to KL with baseline overlays to show DINOv3 clearing all 4 calibration baselines*
4. *Compare → frozen vs. fine-tuned CLIP shift map on Q1710328 — blue = gained attention after fine-tuning, red = lost; narrate +0.207 Δ IoU on the Ornate Portal feature*
5. *Q2 tab → multi-metric improvement heatmap + preserve/enhance/destroy summary; point out the zero-Δ Linear Probe row as the control*
6. *Q3 tab → head-ranking view for DINOv3 showing `layer10/head8` dominance; head-feature alignment matrix highlighting portal-class features]*

End: *"The app lets us move fluidly between the aggregate benchmark and individual image evidence — that's what makes Q1, Q2, and Q3 interpretable rather than just numbers in a table."*

---

## SEGMENT 7 — Conclusions & Takeaways (1 min)

---

### Slide 29 — What We Found

**Three questions, three answers**

**Q1 — Frozen alignment exists, but is model-family dependent**
DINOv3 is uniquely compatible with expert-marked architectural evidence — the only frozen model clearing all calibration baselines on all continuous metrics.

**Q2 — Fine-tuning's effect is mediated by three factors**
Spatial prior (DINO preserves), linguistic coverage (CLIP gains only on Gothic/Romanesque), and geometric discriminability (MAE gains on Renaissance pediments). No single strategy wins everywhere.

**Q3 — Specialisation is sparse and family-shaped**
DINO preserves dominant heads, MAE is partly reshaped, CLIP reorganises from early frozen to late adapted heads. Strongest alignment concentrates on structural parts, not fine ornamentation.

---

### Slide 30 — Practical Implication

**Model selection for domain adaptation should not be guided by accuracy alone.**

Match the pretraining prior to what the task's diagnostic evidence requires:

- If evidence is spatially coherent and structured → **DINOv3 frozen** may already be sufficient
- If evidence is linguistically well-described → **CLIP + Full fine-tuning** unlocks the largest gain
- If evidence is geometrically compact and style-exclusive → **MAE** covers a complementary image subset
- For coverage across hard images → **MAE + language-cluster ensemble** (not language-cluster alone)

---

### Slide 31 — Thank You

**Do Self-Supervised Vision Models Learn What Experts See?**

*Sometimes. It depends on how they were trained — and what the task asks them to look at.*

Team: Leong Kay Mei, Desmond Choy
ISY5004 Intelligent Sensing Systems Practice Project

---

## Slide Count Summary

| Segment | Slides | Time |
|---|---|---|
| 1. Hook & results summary | 1–4 (4 slides) | 2.0 min |
| 2. Dataset & methodology | 5–9 (5 slides) | 2.0 min |
| 3. Q1 results | 10–14 (5 slides) | 2.5 min |
| 4. Q2 results | 15–22 (8 slides) | 3.5 min |
| 5. Q3 results | 23–27 (5 slides) | 1.0 min |
| 6. App demo | 28 + demo clip (1 slide) | 3.0 min |
| 7. Conclusion | 29–31 (3 slides) | 1.0 min |
| **Total** | **31 slides + 1 demo clip** | **15 min** |

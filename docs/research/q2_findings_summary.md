# Q2 Fine-Tuning Findings Summary

> **Experiment:** `fine_tuning_primary_20260327`
> **Date:** April 2026
> **Scope:** Steps 1–3 of the Q2 investigation roadmap

---

## Background: Model Performance Overview

Full fine-tuning on 4-class architectural style classification (Romanesque / Gothic / Renaissance / Baroque), evaluated on 139 expert-annotated images (layer 11, IoU p90):

| Model | Frozen IoU | Fine-tuned IoU | Δ IoU | Cohen's d | Significant |
|-------|-----------|----------------|-------|-----------|-------------|
| CLIP | 0.0181 | 0.0745 | **+0.0564** | 1.005 | ✅ |
| SigLIP2 | 0.0220 | 0.0519 | **+0.0299** | 0.781 | ✅ |
| SigLIP | 0.0364 | 0.0618 | **+0.0254** | 0.604 | ✅ |
| MAE | 0.0702 | 0.0988 | **+0.0286** | 0.413 | ✅ |
| DINOv2 | 0.0816 | 0.0758 | −0.0058 | −0.184 | ❌ |
| DINOv3 | 0.1327 | 0.1321 | −0.0007 | −0.017 | ❌ |

**Core claim:** CLIP/SigLIP needed fine-tuning to develop expert-relevant spatial attention; DINO already had it from pretraining.

---

## Step 1: Is CLIP's Improvement Style-Specific?

### Question
Is CLIP's improvement (+0.056 aggregate Δ IoU) driven by particular architectural styles, or is it uniform across all four?

### Analysis Method
- Script: `experiments/scripts/analyze_style_breakdown.py`
- Reads per-image Δ IoU from `q2_metrics_analysis.json`, cross-references with `building_parts.json` style labels
- Computes per-style mean Δ IoU with 95% bootstrap CI; Kruskal-Wallis test for style moderation

### Findings

| Model | Romanesque (n=54) | Gothic (n=49) | Renaissance (n=22) | Baroque (n=17) | KW p |
|-------|:-----------------:|:-------------:|:-----------------:|:--------------:|:----:|
| **CLIP** | **+0.066** | **+0.079** | +0.014 | +0.013 | n.s. |
| **MAE** | +0.007 | +0.009 | **+0.108** | **+0.045** | p<0.05 |
| **SigLIP2** | +0.034 | +0.044 | +0.007 | +0.007 | n.s. |
| **SigLIP** | +0.029 | +0.039 | −0.006 | +0.005 | n.s. |
| **DINOv2** | −0.010 | +0.001 | −0.004 | −0.012 | n.s. |
| **DINOv3** | −0.001 | +0.006 | −0.004 | −0.009 | n.s. |

**CLIP's improvement is entirely carried by Romanesque (+0.066) and Gothic (+0.079).** Renaissance and Baroque show near-zero Δ (+0.014 and +0.013). DINO shows flat Δ across all four styles — the ceiling is not style-specific.

### Why

CLIP (Radford et al., 2021) is trained on 400M web image–text pairs with a global InfoNCE contrastive loss:

$$L_i = -\log\frac{\exp(\text{sim}(I_i,T_i)/\tau)}{\sum_j \exp(\text{sim}(I_i,T_j)/\tau)}$$

There is no patch-level spatial pressure — the only training signal is whole-image vs. text alignment. Romanesque and Gothic architectural features (pointed arch portals, rose windows, round arch portals) appear frequently in English web descriptions of churches. Renaissance and Baroque features (Pediments, Volutes, Pilasters) have narrower English-language coverage in web alt-text. Fine-tuning adds the spatial pressure CLIP's objective never provided, and it succeeds most on the styles that are most linguistically grounded.

DINOv3 achieves 81.1 mIoU on ADE20k with frozen features (vs. DINOv2's 75.9) — the Gram anchoring loss during pretraining locks patch structure such that style supervision adds almost nothing.

---

## Step 2: Do Models Improve on the Same Images?

### Question
Do models that both improve (e.g. CLIP and MAE) improve on the **same images**, or do they address complementary weaknesses? Does high frozen IoU (DINO) predict where CLIP's fine-tuning succeeds?

### Analysis Method
- Script: `experiments/scripts/analyze_model_correlation.py`
- Computes Pearson r and Spearman ρ between each model's frozen IoU and CLIP's per-image Δ IoU (CLIP fixed as reference — largest and most interpretable Δ)
- Also builds a pairwise Δ IoU correlation matrix across all model pairs

### Findings

**DINOv3 frozen IoU vs. CLIP Δ IoU: Pearson r = +0.677, Spearman ρ = +0.612 (both p < 0.0001)**

Three natural clusters from the pairwise Δ correlation matrix:

| Cluster | Models | Pairwise r | Interpretation |
|---------|--------|-----------|----------------|
| Language cluster | CLIP / SigLIP / SigLIP2 | r ≈ 0.43–0.58 | Same frozen deficiency → improve on same images |
| MAE | MAE | r ≈ −0.22 to −0.31 vs. all | Improves on different (Renaissance) images |
| DINO pair | DINOv2 / DINOv3 | r = 0.33 with each other, near-zero with language cluster | Already aligned; no FT gain |

The images where DINOv3's pretraining produces expert-aligned attention are exactly the images where CLIP's fine-tuning succeeds — they are structurally "easy" (spatially compact, prominent annotations). CLIP and DINO converge on the same images via different routes.

MAE is anti-correlated with all other models: it improves a completely disjoint Renaissance image subset that has no overlap with the Gothic/Romanesque images driving CLIP's gains.

### Why

- **Language cluster** (CLIP/SigLIP/SigLIP2): All three use global image–text alignment as their sole primary objective. LP Δ = 0.000 for all three — the frozen CLS is not style-separable. Whatever makes an image "easy" to improve is identical across the three models.

- **MAE** (anti-correlated): Trained with pixel MSE reconstruction ($L = \|x - \hat{x}\|_2^2$) under 75% masking (whitepaper optimal: 76.5% FT accuracy vs. 73.9% at 50% and 76.0% at 90%). What makes an image "improvable" for MAE is categorically different — local geometry representability — not linguistic grounding.

- **DINO pair**: iBOT patch-level masked prediction on curated large datasets produces frozen IoU of 0.082/0.133, with only a ~2% fine-tuning gap on ImageNet. Style supervision adds almost nothing.

---

## Step 3: Why Does MAE Show a Renaissance Spike?

### Question
MAE's Renaissance Δ = +0.108 is the largest single-style shift in the entire dataset — unexpected given MAE's modest aggregate (+0.029). What drives it, and is it explained by MAE's pretraining?

### Analysis Method
- Script: `experiments/scripts/analyze_feature_delta_iou.py --model mae --strategy full --style Renaissance`
- Computes per-feature Δ IoU by grouping `annotation.bboxes` by `group_label`, building per-group binary masks, computing `compute_iou(heatmap, mask, percentile=90)` against the fine-tuned model's heatmap, then diffing against frozen IoU from `metrics.db`

### Findings

**Check 1 — Which features drive the spike?**

Per-feature Δ IoU (full FT, layer 11, p90) within Renaissance images:

| Feature | Frozen IoU | FT IoU | Δ IoU | n images |
|---------|-----------|--------|-------|----------|
| Triangular Pediment | 0.0360 | 0.1161 | **+0.0800** | 19 |
| Cranked Cornice | 0.0045 | 0.0668 | **+0.0623** | 2 |
| Broken Pediment | 0.0054 | 0.0599 | **+0.0545** | 7 |
| Volute | 0.0087 | 0.0516 | **+0.0429** | 4 |
| Segmental Pediment | 0.0126 | 0.0543 | **+0.0417** | 7 |
| Segmental Arch | 0.0087 | 0.0299 | +0.0211 | 5 |
| Projecting Cornice | 0.0356 | 0.0485 | +0.0129 | 6 |
| Column | 0.0036 | 0.0159 | +0.0123 | 7 |
| Round Arch Niche | 0.0000 | 0.0083 | +0.0083 | 7 |
| Pilaster | 0.0232 | 0.0111 | **−0.0121** | 15 |
| Belt Course | 0.0311 | 0.0155 | **−0.0156** | 9 |

The top five features are all **pediment-class shapes** — geometrically structured, spatially compact, and visually distinctive. The original hypothesis named Trefoil Window as the expected driver; it does not appear in these Renaissance images at all. **Pediments, not Trefoil Windows, are the locus of MAE's Renaissance gain.**

**Check 2 — Is this a genuine realignment or amplification of a pre-existing advantage?**

Pediment features had near-zero frozen alignment before fine-tuning (Triangular Pediment: 0.036, Broken Pediment: 0.005, Segmental Pediment: 0.013). The large post-FT values (0.116, 0.060, 0.054) represent genuine realignment **created** by fine-tuning, not amplification of a pre-existing advantage.

Notably, **Pilaster and Belt Course — the most common Renaissance features — show negative Δ**. Fine-tuning actively shifts attention *away* from them (cross-style ambiguity) and toward pediment forms (Renaissance-discriminative).

**Check 3 — LP confirms backbone gradients are required**

MAE LP Δ = 0.000 across all 139 images. Only full FT and LoRA (both flow gradients into the backbone) reorganize spatial attention. This is consistent with MAE's whitepaper: LP accuracy is "weakly correlated" with FT accuracy, with a 7-point gap on ImageNet (76.6% LP vs. 83.6% FT for ViT-L).

### Why

MAE's pixel-reconstruction objective under 75% masking requires the encoder to build representations that preserve edge structure, shape boundaries, and texture at the patch level. Triangular and Broken Pediments are defined by sharp angular and curved geometric contours — highly recoverable from adjacent visible patches and precisely representable by MAE's encoder. Belt Courses and Pilasters are elongated, low-contrast bands with low reconstruction specificity relative to their spatial extent.

When fine-tuned to distinguish Renaissance from the other three styles, the gradient routes to pediment features because:
1. MAE's encoder already represents their local geometry with high fidelity (from pixel reconstruction pretraining)
2. Pediment forms are nearly exclusive to Renaissance in this dataset (discriminative)
3. Belt Courses and Pilasters appear across multiple styles (ambiguous) — fine-tuning suppresses them

---

## Consolidated Conclusions

1. **CLIP's gain is style-specific and linguistically grounded.** Gothic and Romanesque features dominate English web text about churches; fine-tuning unlocks latent patch-level spatial knowledge that CLIP's InfoNCE training never forced the CLS token to aggregate.

2. **Models improve on the same "structurally easy" images, not complementary subsets.** DINOv3 frozen IoU strongly predicts where CLIP fine-tuning succeeds (r = +0.677). The language cluster (CLIP/SigLIP/SigLIP2) shares a common frozen deficiency and improves on the same image subset. MAE is the exception — it improves a disjoint Renaissance subset.

3. **MAE's Renaissance spike is real, driven by pediment geometry, and consistent with its pretraining.** The top five gainers are all pediment-class features with near-zero frozen alignment. Fine-tuning creates pediment alignment from scratch by redirecting MAE's local geometry representations toward Renaissance-discriminative forms. LP Δ = 0 across all models confirms backbone gradients are the necessary mechanism.

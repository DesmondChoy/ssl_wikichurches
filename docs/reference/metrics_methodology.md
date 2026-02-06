# Metrics Methodology Reference

This document provides a detailed technical reference for the quantitative metrics used to evaluate attention-annotation alignment in this project. It covers metric definitions, thresholding methodology, ground truth construction, worked examples from live testing, and known limitations.

> **Related documents:**
> - [Attention Methods Guide](../research/attention_methods.md) — Attention extraction methods, model considerations, and the plausibility vs. faithfulness distinction
> - [API Reference](api_reference.md) — REST endpoints for querying metrics (`/api/metrics/...`)
> - [Project Proposal](../core/project_proposal.md) — Research design, hypotheses, and evaluation plan (§3.3, §5)

---

## Table of Contents

1. [Metric Definitions](#metric-definitions)
2. [How Thresholding Works](#how-thresholding-works)
3. [Ground Truth: The Union Mask](#ground-truth-the-union-mask)
4. [Worked Example](#worked-example)
5. [Known Limitations and Design Decisions](#known-limitations-and-design-decisions)
6. [Source Code References](#source-code-references)
7. [Academic References](#academic-references)

---

## Metric Definitions

### IoU (Intersection over Union)

IoU measures the spatial overlap between a **thresholded attention mask** and the **ground truth annotation mask**. It is the standard Jaccard Index applied to binary masks:

```
IoU = |A ∩ G| / |A ∪ G|
```

Where:
- **A** = binary attention mask (pixels above the percentile threshold)
- **G** = binary ground truth mask (union of all bounding boxes)
- **|·|** = pixel count

**Properties:**
- Range: [0, 1], where 1 = perfect overlap
- **Threshold-dependent**: IoU changes when the percentile threshold changes, because a different set of pixels is selected as "high attention"
- Penalizes both false positives (attention outside annotations) and false negatives (annotations without attention)
- Division-by-zero is guarded with ε = 1e-8

**When IoU is high:** The model's high-attention region closely matches the expert-annotated region — few false positives or false negatives.

**When IoU is low:** Either the model attends to the wrong areas (false positives), misses annotated areas (false negatives), or both.

**Non-monotonic relationship with threshold:** IoU does not increase monotonically as the threshold increases. Instead, it follows an inverted-U curve driven by a tradeoff between two error types:

| Threshold regime | Attention area vs annotation area | Dominant error | Effect on IoU |
|------------------|-----------------------------------|---------------|---------------|
| Too low (e.g., p50) | Attention mask much **larger** than annotation | False positives inflate the union | IoU decreases |
| Optimal | Attention mask **≈ matches** annotation area | Balanced | IoU peaks |
| Too high (e.g., p99) | Attention mask much **smaller** than annotation | False negatives — annotations without attention coverage | IoU decreases |

IoU peaks when the attention area is closest to the annotation area *and* the spatial overlap is maximized. The optimal percentile therefore depends on the image: images with larger annotation areas peak at lower percentiles, while images with small annotation areas peak at higher percentiles.

**Interpreting IoU scores — baselines and upper bounds:** Unlike classification accuracy (where 1/*k* is random chance and 1.0 is perfect), IoU does not have fixed reference points. Both the random baseline and theoretical maximum depend on the relative sizes of the attention mask (fraction *a*) and annotation mask (fraction *g*):

| Reference point | Formula | Meaning |
|----------------|---------|---------|
| **Random baseline** | (*a* × *g*) / (*a* + *g* − *a* × *g*) | Expected IoU for uniformly random attention |
| **Theoretical maximum** | min(*a*, *g*) / max(*a*, *g*) | Best possible IoU if the smaller mask is perfectly contained in the larger |

Crucially, when the attention area differs greatly from the annotation area, the theoretical maximum can be far below 1.0. For example, if attention covers 50% of the image but annotations cover only 10%, even perfect spatial alignment yields IoU ≤ 0.20. This means **raw IoU scores should always be interpreted relative to these baselines**, not against an absolute scale of 1.0. See the [worked example](#interpreting-the-numbers) for concrete calculations.

```
Source: src/ssl_attention/metrics/iou.py → compute_iou()
```

### Coverage (Energy-Based Pointing Game)

Coverage measures what fraction of the model's total attention **energy** falls inside the annotated regions. It is equivalent to the Energy-Based Pointing Game (EBPG) metric from Petsiuk et al. (2021):

```
Coverage = Σ(attention × gt_mask) / Σ(attention)
```

Where:
- **attention** = raw (continuous) attention map, clamped to non-negative values
- **gt_mask** = binary ground truth mask
- The numerator sums attention values only inside annotated regions
- The denominator sums all attention values across the entire image

**Properties:**
- Range: [0, 1], where 1 = all attention energy is inside annotated regions
- **Threshold-free**: Coverage does not use any percentile threshold. It operates on the raw continuous attention values, not a binarized mask. Changing the percentile slider in the UI has **no effect** on Coverage.
- For uniformly random attention, Coverage ≈ annotation area fraction (e.g., ~10% if annotations cover 10% of the image)
- Complements IoU by providing a threshold-independent view of alignment

**Why Coverage complements IoU:** IoU requires choosing a threshold, and different thresholds can tell different stories. Coverage sidesteps this entirely — it asks "of all the attention the model distributes, how much lands on expert-annotated regions?" This makes it useful as a stable reference point when comparing across percentile settings.

```
Source: src/ssl_attention/metrics/iou.py → compute_coverage()
```

### Supporting Metrics

| Metric | Definition | Properties |
|--------|-----------|------------|
| **Attention Area** | Fraction of image pixels in the thresholded attention mask | Always equals `(100 - percentile) / 100` (e.g., percentile=90 → 10% attention area) |
| **Annotation Area** | Fraction of image pixels inside the ground truth union mask | Constant for a given image regardless of model/layer/percentile |
| **CorLoc** | Binary: 1 if IoU ≥ 0.5, else 0, averaged across images | Standard WSOL metric (Choe et al., 2020). Computed via `compute_corloc()` |
| **Per-Bbox IoU** | IoU computed per individual bounding box (not the union mask) | Used in Feature Breakdown analysis. Computed via `compute_per_bbox_iou()` |

---

## How Thresholding Works

Attention maps are continuous-valued (each pixel has a floating-point attention value). To compute IoU, we must first binarize them into "high attention" vs "low attention." This project uses **pixel-count percentile thresholding**.

### Pixel-Count Percentile (This Project)

Given an attention map and a percentile *p*:

1. Flatten all attention values into a 1D array
2. Compute the *p*-th percentile value using `torch.quantile(attention, p / 100)`
3. Create a binary mask: pixel = 1 if attention ≥ threshold, else 0

**Example:** At percentile=90, we find the value below which 90% of pixels fall. All pixels at or above this value are marked as "high attention." This always selects exactly the **top 10% of pixels by count**.

```python
# From src/ssl_attention/metrics/iou.py → threshold_attention()
threshold = torch.quantile(attn.flatten().float(), percentile / 100.0)
mask = attn >= threshold
```

### Comparison with Other Thresholding Approaches

Different papers use different thresholding strategies. These are **not interchangeable** — the same raw attention map will produce different binary masks (and therefore different IoU scores) under each method:

| Approach | What it selects | Used by | Key difference |
|----------|----------------|---------|---------------|
| **Pixel-count percentile** (this project) | Top *k*% of pixels by spatial count | This project | Exactly *k*% of pixels are always selected |
| **Cumulative-mass thresholding** | Pixels containing top *k*% of attention mass | Caron et al. (2021), DINO paper | The *number* of pixels varies — a few high-value pixels may contain most of the mass |
| **Mean-value thresholding** | Pixels above the mean attention value | Chefer et al. (2021) | Single fixed threshold, no percentile parameter |

### DINO's Cumulative-Mass Method (for comparison)

The DINO paper (Caron et al., 2021) reports Jaccard similarity using a fundamentally different thresholding approach:

```python
# From DINO visualize_attention.py (simplified)
val, idx = torch.sort(attentions)
val /= torch.sum(val, dim=1, keepdim=True)  # normalize to sum=1
cumval = torch.cumsum(val, dim=1)
th_attn = cumval > (1 - threshold)  # keep pixels with top X% of mass
```

**How it differs:** Instead of keeping a fixed percentage of *pixels*, DINO keeps pixels that collectively account for a given percentage of attention *mass* (sum of values). If attention is concentrated on a few pixels, this selects fewer pixels than the pixel-count method. If attention is diffuse, it selects more.

**Implication:** IoU numbers from this project are **not directly comparable** to IoU numbers reported in the DINO paper, even for the same model and images. Both approaches are mathematically valid, but they answer slightly different questions:
- Pixel-count: "Does the model's top-*k*% most-attended region overlap with annotations?"
- Cumulative-mass: "Does the region containing most of the attention energy overlap with annotations?"

---

## Ground Truth: The Union Mask

### How Multiple Bounding Boxes Are Combined

Each image in the WikiChurches dataset may have multiple bounding boxes (ranging from 1 to 25, with a mean of ~4.5 across 139 images and 631 total bboxes). For the primary metrics (IoU and Coverage), all bounding boxes for an image are combined into a single **union mask** via logical OR:

```python
# From src/ssl_attention/data/annotations.py → ImageAnnotation.get_union_mask()
union_mask = torch.zeros(height, width, dtype=torch.bool)
for bbox in self.bboxes:
    union_mask |= bbox.to_mask(height, width)
```

**Why union?** The research question is "does the model attend to expert-annotated regions?" — the union mask captures *all* annotated regions regardless of feature type. A model that attends to the rose window but not the buttress (both annotated) still receives partial credit via IoU.

### Coordinate System

Bounding box coordinates are stored as normalized values in [0, 1] relative to image dimensions. They are converted to pixel coordinates at the attention map resolution (not the original image resolution):

- Attention resolution depends on the model's patch size (e.g., 16×16 grid for DINOv2's 14px patches on 224×224 input)
- Negative coordinates (found in ~4 images) are clamped to 0
- Very small boxes are guaranteed at least 1 pixel in each dimension

### Per-Bbox IoU (Feature Breakdown)

For the Feature Breakdown analysis, IoU is computed per individual bounding box using `compute_per_bbox_iou()`. This enables questions like "does the model attend to windows more than arches?" The per-bbox results are aggregated by feature type and stored in the `feature_metrics` table of the SQLite cache.

```
Source: src/ssl_attention/metrics/iou.py → compute_per_bbox_iou()
Source: src/ssl_attention/data/annotations.py → BoundingBox.to_mask(), ImageAnnotation.get_union_mask()
```

---

## Worked Example

The following data was collected from live UI testing using image **Q526047_wd0.jpg** (10 bounding boxes, annotation area = 10.3% of image).

### Layer Sweep (DINOv2 / CLS / Percentile=90)

This shows how both metrics evolve across transformer layers:

| Layer | IoU | Coverage | Interpretation |
|-------|-----|----------|----------------|
| L0 | 0.028 | 9.1% | Early layer: near-random alignment |
| L1 | 0.009 | 9.3% | Minimal semantic content |
| L2 | 0.029 | 10.4% | Still low-level features |
| L3 | 0.073 | 12.1% | Slight improvement begins |
| L4 | 0.079 | 10.9% | Mid-level features emerging |
| L5 | 0.012 | 8.3% | Temporary dip (common in ViTs) |
| L6 | 0.042 | 10.0% | Recovery |
| L7 | 0.127 | 13.5% | Semantic attention emerging |
| L8 | 0.096 | 12.8% | Non-monotonic progression |
| L9 | 0.180 | 16.1% | Strong semantic alignment |
| L10 | 0.197 | 17.9% | Near-peak |
| L11 | 0.250 | 20.2% | Best alignment (final layer) |

**Key observations:**
- IoU increases ~9× from L0 to L11 (0.028 → 0.250), confirming that expert-aligned attention emerges in later layers
- Coverage increases ~2.2× (9.1% → 20.2%), showing the model progressively concentrates more energy on annotated regions
- Progression is non-monotonic (dips at L1, L5, L8), consistent with findings in Raghu et al. (2021) about internal ViT representations
- Early-layer Coverage (~9%) is close to the annotation area fraction (10.3%), as expected for near-random attention

### Percentile Sweep (DINOv2 / CLS / Layer 11)

This demonstrates that **Coverage is threshold-free** while IoU depends on the percentile:

| Percentile | Pixels Selected | IoU | Coverage | Attn Area |
|------------|----------------|-----|----------|-----------|
| 90 (Top 10%) | Top 10% | 0.250 | 20.2% | 10.0% |
| 85 (Top 15%) | Top 15% | 0.243 | 20.2% | 15.0% |
| 80 (Top 20%) | Top 20% | 0.229 | 20.2% | 20.0% |
| 75 (Top 25%) | Top 25% | 0.214 | 20.2% | 25.0% |
| 70 (Top 30%) | Top 30% | 0.213 | 20.2% | 30.0% |
| 60 (Top 40%) | Top 40% | 0.209 | 20.2% | 40.0% |
| 50 (Top 50%) | Top 50% | 0.189 | 20.2% | 50.0% |

**Key observations:**
- **Coverage is constant at 20.2% across all 7 percentile values.** This confirms it is threshold-free by design — it operates on raw attention values, not the binarized mask.
- **IoU decreases as more pixels are selected** (0.250 → 0.189). As the attention mask grows beyond the annotation region (10.3% of the image), false positives increase and inflate the union, diluting IoU.
- **Attention Area exactly matches the percentile setting** (percentile=90 → 10% area, percentile=50 → 50% area), confirming the pixel-count percentile implementation is correct.
- **The peak IoU (0.250) occurs at p90, where attention area (10.0%) most closely matches annotation area (10.3%).** This is the inverted-U behavior described in the [IoU definition](#iou-intersection-over-union): the data here only shows the left side of the curve (threshold too low → false positives dominate). At thresholds above p90 (e.g., p95 → 5% area, p99 → 1% area), IoU would decline again as the attention mask becomes too small to cover the annotated region, and false negatives dominate instead.

### Interpreting the Numbers

Raw IoU scores should be read against the random baseline and theoretical maximum for each threshold setting (see [Interpreting IoU scores](#iou-intersection-over-union)). For this image (*g* = 10.3% annotation area):

| Percentile | Attn Area (*a*) | Random IoU | Max IoU | Observed | % of Max |
|------------|-----------------|------------|---------|----------|----------|
| 90 | 10.0% | 0.053 | 0.971 | 0.250 | 25.7% |
| 85 | 15.0% | 0.066 | 0.687 | 0.243 | 35.4% |
| 80 | 20.0% | 0.073 | 0.515 | 0.229 | 44.5% |
| 75 | 25.0% | 0.078 | 0.412 | 0.214 | 51.9% |
| 70 | 30.0% | 0.081 | 0.343 | 0.213 | 62.1% |
| 60 | 40.0% | 0.085 | 0.258 | 0.209 | 81.0% |
| 50 | 50.0% | 0.087 | 0.206 | 0.189 | 91.7% |

**Key insight:** The raw IoU at p50 (0.189) may look low, but it is **91.7% of the theoretical maximum** (0.206) — meaning the model's attention covers nearly all annotated pixels at that threshold. The "low" absolute IoU is a geometric constraint (the union is large when 50% of pixels are selected), not poor model performance. Conversely, at p90, the observed IoU (0.250) is only 25.7% of the theoretical maximum (0.971), indicating substantial room for improvement in spatial precision — the model's top 10% of pixels don't concentrate tightly enough on the annotations.

All observed IoU values are well above their random baselines (3-5×), confirming the model attends to annotated regions at rates far above chance.

### Cross-Model Comparison (Layer 11, Percentile=90)

| Condition | IoU | Coverage |
|-----------|-----|----------|
| DINOv2 / CLS / L11 | 0.250 | 20.2% |
| DINOv2 / Rollout / L11 | 0.062 | 11.9% |
| CLIP / CLS / L11 | 0.024 | 7.9% |

This illustrates that both model choice and attention method significantly affect alignment scores.

---

## Known Limitations and Design Decisions

### 1. Pixel-Count vs Mass Thresholding

This project uses pixel-count percentile thresholding (via `torch.quantile`), while the DINO paper uses cumulative-mass thresholding. As a result:

- **IoU numbers from this project are not directly comparable to DINO paper numbers**, even when evaluating the same DINOv2 model
- Both methods are valid; they answer slightly different questions about attention concentration
- The pixel-count method was chosen for its simplicity and interpretability (percentile=90 always means "top 10% of pixels")

### 2. Coverage Stability with Large Union Masks

When the union mask covers a large fraction of the image (many overlapping bounding boxes), Coverage naturally tends higher because more of the image is "inside" the annotated region. For images where annotations cover, say, 50% of the image, even random attention would yield ~50% Coverage. This is not a bug but should be considered when interpreting Coverage scores — annotation area should be reported alongside Coverage for context.

### 3. Per-Bbox vs Union Metrics

In the Image Detail view, selecting a bounding box switches the displayed IoU and Coverage to **per-bbox metrics** computed against that individual bounding box (via `useBboxMetrics`). When no bbox is selected, metrics show the **union** of all bounding boxes. The UI indicates which mode is active with a green label showing the selected feature name.

The per-bbox metrics are computed on the backend via `compute_per_bbox_iou()` and are also used for the Feature Breakdown analysis on the Dashboard.

### 4. Strict Pointing Game Variant

The CorLoc implementation uses a strict variant (IoU ≥ 0.5) without the 15-pixel tolerance border used in the standard Pointing Game protocol (Zhang et al., 2016). This means CorLoc scores may be slightly lower than those reported in papers using the standard protocol.

---

## Source Code References

| Concept | File | Key Functions |
|---------|------|--------------|
| IoU computation | `src/ssl_attention/metrics/iou.py` | `compute_iou()`, `compute_image_iou()` |
| Percentile thresholding | `src/ssl_attention/metrics/iou.py` | `threshold_attention()` |
| Coverage (EBPG) | `src/ssl_attention/metrics/iou.py` | `compute_coverage()` |
| Per-bbox IoU | `src/ssl_attention/metrics/iou.py` | `compute_per_bbox_iou()` |
| CorLoc | `src/ssl_attention/metrics/iou.py` | `compute_corloc()` |
| Feature type aggregation | `src/ssl_attention/metrics/iou.py` | `aggregate_by_feature_type()` |
| Union mask construction | `src/ssl_attention/data/annotations.py` | `ImageAnnotation.get_union_mask()` |
| Single bbox mask | `src/ssl_attention/data/annotations.py` | `BoundingBox.to_mask()` |
| Metrics database queries | `app/backend/services/metrics_service.py` | `MetricsService` (singleton) |
| Metrics cache generation | `app/precompute/generate_metrics_cache.py` | `compute_metrics_for_model()` |
| Epsilon constant | `src/ssl_attention/config.py` | `EPSILON = 1e-8` |

---

## Academic References

| Paper | Citation | Relevance to Metrics |
|-------|----------|---------------------|
| **DINO** | Caron, M., et al. (2021). Emerging Properties in Self-Supervised Vision Transformers. *ICCV*. [arXiv:2104.14294](https://arxiv.org/abs/2104.14294) | Uses cumulative-mass thresholding + Jaccard similarity for evaluation; our thresholding differs |
| **D-RISE / EBPG** | Petsiuk, V., et al. (2021). Black-box Explanation of Object Detectors via Saliency Maps. *CVPR*. [arXiv:2006.03204](https://arxiv.org/abs/2006.03204) | Defines Energy-Based Pointing Game (EBPG), the basis for our Coverage metric |
| **Attention Rollout** | Abnar, S. & Zuidema, W. (2020). Quantifying Attention Flow in Transformers. *ACL*. [arXiv:2005.00928](https://arxiv.org/abs/2005.00928) | Attention rollout method used as an alternative to single-layer CLS attention |
| **Transformer Interpretability** | Chefer, H., et al. (2021). Transformer Interpretability Beyond Attention Visualization. *CVPR*. [arXiv:2012.09838](https://arxiv.org/abs/2012.09838) | Uses mean-value thresholding; compares Grad-CAM vs attention methods |
| **WSOL Evaluation** | Choe, J., et al. (2020). Evaluating Weakly Supervised Object Localization Methods Right. *CVPR*. [arXiv:1910.12449](https://arxiv.org/abs/1910.12449) | Standardizes CorLoc and IoU evaluation protocols for localization |
| **CAM** | Zhou, B., et al. (2016). Learning Deep Features for Discriminative Localization. *CVPR*. [arXiv:1512.04150](https://arxiv.org/abs/1512.04150) | Introduced IoU-based localization evaluation |
| **Pointing Game** | Zhang, J., et al. (2016). Top-Down Neural Attention by Excitation Backprop. *ECCV*. [arXiv:1511.02668](https://arxiv.org/abs/1511.02668) | Standard Pointing Game protocol (with 15px tolerance) |
| **ERASER Benchmark** | DeYoung, J., et al. (2020). ERASER: A Benchmark to Evaluate Rationalized NLP Models. *ACL*. [arXiv:1911.03429](https://arxiv.org/abs/1911.03429) | Defines plausibility vs faithfulness — our metrics measure plausibility |

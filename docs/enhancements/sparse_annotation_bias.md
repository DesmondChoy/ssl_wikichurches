# Sparse Annotation Bias in Per-Bbox IoU

> **Documented:** February 2026
> **Status:** Documented limitation with mitigation strategies
> **Purpose:** Explain how incomplete annotations inflate perceived false positives in per-bbox IoU, and catalog mitigation approaches ranked by effort and effectiveness

> **Related documents:**
> - [Metrics Methodology — §3 Union Mask](../reference/metrics_methodology.md#ground-truth-the-union-mask), [§5.3 Per-Bbox vs Union](../reference/metrics_methodology.md#3-per-bbox-vs-union-metrics)
> - [Project Proposal — Evaluation Framework](../core/project_proposal.md#research-questions-and-approaches)
> - [Fine-Tuning Methods — Δ IoU Metric](fine_tuning_methods.md)

---

## 1. Problem Statement

### The Round Arch Window Example

Consider image **Q2820485_wd0.jpg**, which depicts a church facade with multiple round arch windows. The WikiChurches dataset annotates a single representative bounding box for "Round Arch Window" — but the facade contains several additional instances of the same feature that are **not annotated**.

When a user clicks the annotated "Round Arch Window" bbox in the Image Detail view, the backend computes IoU against that single bbox mask:

```
GET /api/metrics/{image_id}/bbox/{bbox_index}?model=dinov2&percentile=90
```

The model's attention correctly spreads across **all** round arch windows on the facade — both the annotated one and the unannotated ones. The IoU calculation then treats attention on the unannotated windows as **false positives** (attention outside the ground truth), even though the model is demonstrating correct semantic understanding.

```
                    ┌──────────────────────────────┐
                    │        Church Facade          │
                    │                               │
                    │  ┌─────┐  ┌─────┐  ┌─────┐   │
                    │  │ ⬡⬡⬡ │  │ ⬡⬡⬡ │  │ ⬡⬡⬡ │   │
                    │  │ win │  │ win │  │ win │   │
                    │  └──┬──┘  └──┬──┘  └──┬──┘   │
                    │     │       │       │        │
                    │  annotated  NOT     NOT      │
                    │  (bbox 3)  annotated annotated│
                    └──────────────────────────────┘

  Model attention:  ████████  ████████  ████████   ← correct!
  Ground truth:     ████████  --------  --------   ← sparse
  Per-bbox IoU:     HIGH      PENALIZED PENALIZED
```

**Result:** The per-bbox IoU is deflated because the model is "too good" — it finds all instances of a feature that the dataset only partially annotates.

### Definition

**Sparse annotation bias** occurs when the evaluation metric conflates annotation incompleteness with model error. The WikiChurches dataset annotates *representative* instances of each feature type, not *exhaustive* instances. This design choice is practical for annotation efficiency but introduces a systematic downward bias in per-bbox IoU for features with multiple visible instances.

### Why This Cannot Be Fixed Computationally

The core issue is **missing ground truth**: visible architectural features with no bounding box in the dataset. No amount of code can conjure annotations that don't exist. An initial proposal to "union all bboxes sharing the same feature label" was investigated and found to be nearly ineffective:

- Only **6 of 139 images** (4.3%) contain multiple bboxes with the same label
- Each has exactly **2** same-label bboxes (never 3+)
- The fix would affect **12 of 631 bboxes** (1.9%)

The dominant case — a facade with 5 visible round arch windows where only 1 is annotated — has nothing to union. The remaining mitigations fall into three categories: **alternative metrics** that are robust to sparse annotations, **statistical corrections** that quantify the bias, and **annotation completion** that addresses the root cause.

---

## 2. Why It Matters

### Impact on Per-Bbox Metrics

The bias specifically affects the **per-bbox IoU** pathway — when a user selects an individual bounding box in the Image Detail view, or when the Feature Breakdown analysis aggregates IoU by feature type. The magnitude of the bias depends on:

| Factor | Effect on Bias |
|--------|---------------|
| **Number of unannotated instances** | More unannotated instances → more "false positive" attention → lower IoU |
| **Feature regularity** | Repeating architectural elements (windows, columns, arches) are most affected |
| **Model quality** | Better models are penalized *more* because they correctly attend to all instances |
| **Attention method** | Diffuse methods (Mean, Rollout) spread more attention to unannotated instances than focused methods (CLS) |

### Misleading IoU Scores

Without awareness of this bias, a researcher might incorrectly conclude:

1. **"The model poorly attends to round arch windows"** — when it actually attends to *all* of them, but only one is labeled
2. **"Model A is worse than Model B for windows"** — when Model A simply finds more unannotated instances
3. **"Fine-tuning decreased attention to feature X (negative Δ IoU)"** — when fine-tuning actually *improved* feature detection by finding additional instances

### Implications for Research Conclusions

For the Δ IoU analysis (comparing pre- vs post-fine-tuning attention), sparse annotation bias could mask real improvements. If fine-tuning teaches a model to attend to *all* instances of an architectural feature (a desirable outcome), the per-bbox IoU might paradoxically decrease — leading to a false negative in hypothesis testing.

---

## 3. Current Behavior

### Per-Bbox IoU Path (Affected)

When a user clicks a specific bounding box in the UI, the frontend calls the per-bbox metrics endpoint:

**Endpoint:** `app/backend/routers/metrics.py` — `get_bbox_metrics()` (lines 145–212)

```python
# The endpoint extracts a SINGLE bbox mask
bbox = annotation.bboxes[bbox_index]
bbox_mask = bbox.to_mask(h, w)

# IoU is computed against only this single bbox
iou, attention_area, annotation_area = compute_iou(attention_tensor, bbox_mask, percentile)
```

The core per-bbox computation in the metrics library follows the same pattern:

**Function:** `src/ssl_attention/metrics/iou.py` — `compute_per_bbox_iou()` (lines 317–348)

```python
for bbox in annotation.bboxes:
    bbox_mask = bbox.to_mask(h, w)
    iou, _, _ = compute_iou(attention, bbox_mask, percentile)
    results.append((bbox.label, iou))
```

Each bbox is evaluated in isolation. If two bboxes share the same feature label, their IoU scores are computed independently — attention on bbox B is a "false positive" when evaluating bbox A, even if they represent the same feature type.

### Union IoU Path (Less Affected)

The standard union IoU (shown when no bbox is selected) combines *all* bboxes into a single mask:

**Function:** `src/ssl_attention/data/annotations.py` — `get_union_mask()` (lines 133–149)

```python
union_mask = torch.zeros(height, width, dtype=torch.bool)
for bbox in self.bboxes:
    union_mask |= bbox.to_mask(height, width)
```

The union mask is less affected because it already aggregates all annotated regions. However, it still misses unannotated instances of any feature type — a limitation inherent to the dataset, not the code.

### Feature-Type Aggregation Path (Affected)

The feature-type aggregation groups per-bbox IoU scores by label:

**Function:** `src/ssl_attention/metrics/iou.py` — `aggregate_by_feature_type()` (lines 391–428)

```python
for image_results in per_bbox_results:
    for label, iou in image_results:
        label_ious[label].append(iou)
```

This aggregation inherits the sparse annotation bias from `compute_per_bbox_iou()`. Feature types with commonly repeating instances (windows, columns) will have systematically lower aggregated IoU than singular features (main entrance, bell tower), even if the model attends to both equally well.

---

## 4. Mitigation Strategies

The strategies below are organized into three tiers by effort and effectiveness.

### Tier 1: Use Metrics That Are Robust to Sparse Annotations (No New Annotations Needed)

#### 4.1 Per-Bbox Recall (Recommended — Not Yet Implemented)

The single most effective metric-level mitigation. Instead of asking "how well does attention overlap this bbox?" (IoU, symmetric), ask **"does the model attend to this bbox?"** (recall, asymmetric). Per-bbox recall measures what fraction of a bbox's area is covered by high attention, without caring about attention elsewhere.

```python
def bbox_recall(attention: Tensor, bbox_mask: Tensor, percentile: int) -> float:
    """What fraction of this bbox's area is covered by top-percentile attention?

    Unlike IoU, this metric is completely immune to sparse annotation bias:
    attention on unannotated instances does not affect the score.
    """
    attn_mask = threshold_attention(attention, percentile)
    intersection = (attn_mask & bbox_mask).sum()
    bbox_area = bbox_mask.sum()
    return (intersection / (bbox_area + EPSILON)).item()
```

**Why it works:** If a model attends to all 5 round arch windows on a facade, the recall for the 1 annotated window remains high. Attention on the other 4 unannotated windows is simply ignored.

**Limitation:** No precision component — a model attending uniformly to the entire image gets 100% recall. Should always be reported alongside IoU or Coverage, not as a replacement.

**Implementation path:** Add to `src/ssl_attention/metrics/iou.py` alongside `compute_per_bbox_iou()`. Expose via the existing per-bbox metrics endpoint as an additional field in `IoUResultSchema`.

#### 4.2 Diagnostic: Cross-Metric Divergence Pattern (Already Available)

The project already computes Pointing Game, Coverage, and IoU. When these metrics diverge for a specific bbox, the pattern is a **signature of sparse annotation bias**:

| Metric | Value | Interpretation |
|--------|-------|----------------|
| Pointing Game | **Hit** | Max attention is inside the annotated bbox |
| Coverage | **High** (e.g., 20%) | Reasonable attention energy on annotated regions |
| Per-Bbox IoU | **Low** (e.g., 0.05) | Attention spreads beyond the single annotated bbox |

This pattern means the model correctly attends to the annotated feature (pointing game hit, decent coverage) but also attends to unannotated instances of the same feature (deflating IoU). Flagging this pattern in analysis — rather than treating low per-bbox IoU as model failure — is the simplest mitigation.

**Implementation path:** No code changes needed. Document the interpretation pattern in the metrics methodology guide and UI tooltips.

#### 4.3 Existing Asymmetric Metrics (Already Implemented)

Two metrics already in the codebase are partially or fully robust to this bias:

| Metric | Sparse-Annotation Robust? | Code Reference |
|--------|--------------------------|---------------|
| **Pointing Game** | Fully robust — only checks if the single max-attention point hits a bbox | `src/ssl_attention/metrics/pointing_game.py` — `pointing_game_hit()` (line 53) |
| **Top-K Pointing** | Fully robust — checks if any of the top-K attention points hit a bbox | `src/ssl_attention/metrics/pointing_game.py` — `top_k_pointing_accuracy()` (line 154) |
| **Coverage (EBPG)** | Partially robust — penalizes attention energy outside bboxes, but proportionally (not binary area like IoU) | `src/ssl_attention/metrics/iou.py` — `compute_coverage()` (line 168) |

**Recommendation:** When reporting per-feature-type results, report Pointing Game hit rate alongside IoU. If Pointing Game is high but IoU is low for a feature type, attribute the gap to sparse annotation bias rather than model failure.

### Tier 2: Statistical Corrections (Quantify the Bias)

#### 4.4 Bootstrap with Annotation Dropout

Simulate sparse annotations by randomly dropping 20–30% of bboxes, recompute IoU, and build a confidence interval. This answers: **"How sensitive are our IoU scores to annotation completeness?"**

```python
def bootstrap_iou_under_dropout(
    attention_maps: list[Tensor],
    annotations: list[ImageAnnotation],
    percentile: int = 90,
    dropout_rate: float = 0.3,
    n_bootstrap: int = 1000,
) -> tuple[float, float]:
    """Bootstrap 95% CI for mean IoU under simulated annotation dropout.

    If observed IoU falls within the CI, it is consistent with annotation
    sparsity rather than poor model attention.
    """
    bootstrap_means = []
    for _ in range(n_bootstrap):
        ious = []
        for attn, ann in zip(attention_maps, annotations):
            # Randomly drop bboxes
            n_keep = max(1, int(len(ann.bboxes) * (1 - dropout_rate)))
            kept = random.sample(ann.bboxes, n_keep)
            reduced = ImageAnnotation(image_id=ann.image_id, bboxes=kept, ...)

            h, w = attn.shape[-2:]
            mask = reduced.get_union_mask(h, w).to(attn.device)
            iou, _, _ = compute_iou(attn, mask, percentile)
            ious.append(iou)
        bootstrap_means.append(np.mean(ious))

    return np.percentile(bootstrap_means, 2.5), np.percentile(bootstrap_means, 97.5)
```

**Application to Δ IoU:** For the preliminary fine-tuning result (DINOv2 Δ IoU ≈ +0.009, p=0.031), running a permutation test with annotation dropout would strengthen the claim: "The improvement holds even under simulated annotation incompleteness."

#### 4.5 Report IoU as Percentage of Theoretical Maximum

The [metrics methodology guide](../reference/metrics_methodology.md#interpreting-the-numbers) already computes "% of theoretical max IoU" for different percentile settings. The same logic applies here: when per-bbox IoU is low, report what fraction of the *achievable* maximum it represents, given the size mismatch between attention area and annotation area. This contextualizes "low" IoU values without needing new annotations.

### Tier 3: Annotation Completion (Address the Root Cause)

These approaches create missing ground truth. They require more effort but directly solve the problem.

#### 4.6 SAM-Prompted Completion with Human Review

Use Segment Anything Model (SAM) prompted by existing bboxes to propose unannotated instances, then have a human reviewer accept or reject each proposal.

**Workflow:**
1. For each annotated bbox, use it as a point/box prompt to SAM
2. SAM segments all visually similar regions in the same image
3. Human reviewer accepts valid proposals, rejects false positives (~30 sec per proposal)
4. Accepted proposals become new annotations

**Circularity risk:** Low. SAM is trained on a generic segmentation objective (SA-1B dataset), not on the self-supervised attention objectives used by DINOv2/MAE/CLIP. Using SAM to expand annotations, then evaluating DINOv2 attention against those annotations, is not circular — the training signals are orthogonal.

**Effort estimate:** ~3–4 hours for 139 images (generation is automated; human review is the bottleneck).

**Critical requirement:** Create a holdout subset (~25 images) with complete manual annotations (no model assistance) as a gold-standard evaluation set. Report results on both the SAM-expanded set and the holdout set to demonstrate robustness.

#### 4.7 Manual Re-Annotation of a Holdout Subset

The cleanest approach: select 25 diverse images, annotate *every visible instance* of each feature type (not just representative ones), and use this as a gold-standard evaluation set.

**Why 25 images is sufficient:** With ~4.5 bboxes per image on average, 25 fully-annotated images provide ~112 bbox evaluations — enough for statistical comparison against the 139-image sparse set.

**Effort estimate:** ~8–12 hours of expert annotation time.

---

## 5. Scope and Limitations

### Summary of Mitigation Effectiveness

| Strategy | Addresses Missing GT? | Effort | Best For |
|----------|----------------------|--------|----------|
| **Per-bbox recall** (§4.1) | No — sidesteps the problem | Low (new metric) | Per-feature analysis without false-positive penalty |
| **Cross-metric divergence** (§4.2) | No — diagnoses the problem | Zero | Interpreting existing results correctly |
| **Existing asymmetric metrics** (§4.3) | No — sidesteps the problem | Zero | Corroborating IoU findings |
| **Bootstrap dropout** (§4.4) | No — quantifies the problem | Low (analysis script) | Δ IoU robustness testing |
| **% of theoretical max** (§4.5) | No — contextualizes the problem | Zero | Fairer interpretation of raw IoU |
| **SAM-prompted completion** (§4.6) | **Yes** | Medium (~3–4 hrs) | Expanding annotations with quality control |
| **Manual holdout subset** (§4.7) | **Yes** | High (~8–12 hrs) | Gold-standard validation set |

### What No Strategy Solves

| Limitation | Explanation |
|-----------|-------------|
| **Truly unannotated feature types** | If a visible feature type has *zero* annotations across the entire dataset, no mitigation helps — there is no ground truth to evaluate against. |
| **Cross-image annotation inconsistency** | The same feature type may be annotated in one image but absent from another. Per-image mitigations cannot correct dataset-level inconsistency. |
| **Annotation subjectivity** | Experts may disagree on what constitutes a "round arch window" vs a "pointed arch window." Sparse annotation bias overlaps with inter-annotator disagreement but is a distinct issue. |

### When to Use Which Metric

| Metric | Best For | Limitation |
|--------|----------|-----------|
| **Union IoU** (default) | Overall model-annotation alignment | Dominated by large/numerous bboxes; misses unannotated instances |
| **Per-Bbox IoU** (current) | Feature-specific analysis, fine-grained comparison | Sparse annotation bias for repeating features |
| **Per-Bbox Recall** (proposed) | Feature detection analysis ("does the model find this feature?") | No precision component; ignores attention outside bboxes |
| **Pointing Game** (current) | Quick binary check of attention alignment | Only uses 1 pixel; no partial credit |
| **Coverage** (current) | Threshold-free overall alignment | Still penalizes energy outside bboxes (proportionally, not binary) |

---

## References

- Choe, J., et al. (2020). [Evaluating Weakly Supervised Object Localization Methods Right](https://arxiv.org/abs/1910.12449). *CVPR*. — Discusses evaluation pitfalls including annotation completeness; proposes MaxBoxAccV2 and PxAP as more robust alternatives
- Everingham, M., et al. (2010). The PASCAL Visual Object Classes (VOC) Challenge. *IJCV*. — Per-class IoU evaluation protocol; introduces "difficult" flag for ignore regions
- Petsiuk, V., et al. (2021). [Black-box Explanation of Object Detectors via Saliency Maps](https://arxiv.org/abs/2006.03204). *CVPR*. — Defines Energy-Based Pointing Game (EBPG), the basis for our Coverage metric
- Zhang, J., et al. (2016). Top-Down Neural Attention by Excitation Backprop. *ECCV*. — Original Pointing Game metric, robust to sparse annotations by design
- Bylinskii, Z., et al. (2018). [What Do Different Evaluation Metrics Tell Us About Saliency Models?](https://arxiv.org/abs/1604.03605). *TPAMI*. — Comprehensive comparison of saliency metrics including NSS, AUC-Judd, and KL-divergence
- Yang, S., et al. (2020). [Object Detection as a Positive-Unlabeled Problem](https://arxiv.org/abs/2002.04672). *BMVC*. — Frames missing annotations as unlabeled (not negative), directly relevant to sparse annotation bias
- Kirillov, A., et al. (2023). [Segment Anything](https://arxiv.org/abs/2304.02643). *ICCV*. — Foundation model for prompted segmentation, applicable to annotation completion

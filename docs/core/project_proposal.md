# Project Proposal: Do Self-Supervised Vision Models Learn What Experts See?

## Attention Alignment with Human-Annotated Architectural Features

**Course:** ISY5004 Intelligent Sensing Systems Practice Module  
**Team Size:** 3 members  
**Duration:** 6 weeks

---

## 1. Problem Statement

Self-supervised learning (SSL) models like DINOv2, DINOv3, and MAE learn visual representations without labels and achieve strong performance on downstream tasks. However, a fundamental question remains: **do these models attend to the same visual features that human experts consider diagnostic, or do they exploit statistical shortcuts invisible to humans?** Furthermore, when models are fine-tuned for a specific task, **does their attention shift toward expert-identified features, or do they discover alternative discriminative regions?**

Existing SSL benchmarks measure classification accuracy but do not explain *which* image regions drive predictions. This matters for trust and deployment: a model that correctly classifies Gothic architecture by attending to flying buttresses is qualitatively different from one that exploits dataset-specific background correlations. This project addresses that gap by quantitatively measuring alignment between SSL attention patterns and expert-annotated "characteristic architectural features."

While existing tools like BertViz and Comet ML visualize transformer attention, they do not quantify whether attention aligns with domain expertise. This gap matters for practitioners fine-tuning foundation models: a model achieving high classification accuracy might exploit background correlations rather than attending to diagnostic features. We address this gap with both a quantitative benchmark and an interactive analysis tool.

### Research Questions and Approaches

| Research Question | Approach | Tool/Method |
|-------------------|----------|-------------|
| **Q1:** Do SSL models attend to the same features human experts consider diagnostic? | Compute IoU between thresholded attention maps and expert bounding boxes across 6 models and 12 layers | Attention heatmap overlay, IoU metrics dashboard, model leaderboard |
| **Q2:** Does fine-tuning shift attention toward expert-identified features? | Compare Δ IoU (fine-tuned − frozen) with paired statistical tests on same images | Frozen vs fine-tuned comparison view, attention shift visualization |
| **Q3:** Do individual attention heads specialize for different architectural features, and which heads best align with expert annotations? | Compute per-head IoU separately for each of the 12 attention heads; identify heads with consistently highest alignment using rank-based analysis | Per-head attention selector, head IoU heatmap (head × feature type), head specialization dashboard |
| **Q4:** Does the fine-tuning strategy (Linear Probe vs LoRA vs Full) affect how much attention shifts toward expert features? | Compare Δ IoU across three fine-tuning methods using paired tests; compute Cohen's d effect sizes; analyze catastrophic forgetting via pre-training IoU retention | Fine-tuning method comparison view, Δ IoU bar charts by method, forgetting metrics |

> **Enhancement docs:** Q3 is explored in detail in [Per-Head Attention Visualization](../enhancements/per_attention_head.md). Q2/Q4 fine-tuning strategies are detailed in [Fine-Tuning Methods](../enhancements/fine_tuning_methods.md).

---

## 2. Dataset

**Primary Dataset:** WikiChurches (Barz & Denzler, 2021)

- 9,485 images of European church buildings  
- Hierarchical architectural style labels (Gothic, Baroque, Romanesque, etc.)  
- **631 bounding box annotations** identifying characteristic visual features across 139 churches in 4 style categories

**Why WikiChurches:**

- Expert bounding boxes provide rare ground truth for attention evaluation—most vision datasets lack this  
- Fine-grained style distinctions require attending to specific structural elements (pointed arches, rose windows, flying buttresses)  
- Manageable scale for 6-week timeline  
- Hierarchical labels support secondary analysis of concept granularity

**Dataset Structure (Verified):**

- **Annotations:** `building_parts.json` containing 106 feature types and 631 bounding boxes
- **Coordinate format:** Normalized (0-1) as `left, top, width, height`
- **Style distribution:** Gothic (54), Romanesque (49), Baroque (22), Renaissance (17)
- **Style IDs:** Wikidata Q-IDs (Q46261=Romanesque, Q176483=Gothic, Q236122=Renaissance, Q840829=Baroque)

**Preprocessing:**

- Filter to the 139 churches with bounding box annotations for primary attention-alignment analysis
- Use full dataset (filtered to classes with ≥50 samples) for linear probe evaluation
- Standard augmentation (resize, normalize) without heavy augmentation to preserve architectural detail
- Clamp bounding box coordinates to [0,1] (some edge annotations have small negative values)

---

## 3. System Architecture

### 3.1 Feature Extraction Pipeline

Use pretrained weights from HuggingFace/timm. Models are evaluated both **frozen** (Pass 1) and **fine-tuned** (Pass 2) to compare attention patterns before and after task-specific training:

| Model | Parameters | Training Paradigm | Source |
| :---- | :---- | :---- | :---- |
| DINOv2 ViT-B/14 | 86M | Self-distillation (2023) | `facebook/dinov2-with-registers-base` |
| DINOv3 ViT-B/16 | 86M | Self-distillation + Gram Anchoring (2025) | `facebook/dinov3-vitb16-pretrain-lvd1689m` |
| MAE ViT-B/16 | 86M | Masked autoencoding | `facebook/vit-mae-base` |
| CLIP ViT-B/16 | 86M | Contrastive language-image (softmax) | `openai/clip-vit-base-patch16` |
| SigLIP 2 ViT-B/16 | 86M | Contrastive language-image (sigmoid) + dense features | `google/siglip2-base-patch16-224` |
| ResNet-50 | 23M | Supervised (ImageNet) | `torchvision` |

All models use ViT-B architecture for controlled comparison (ResNet-50 included as supervised baseline).

### 3.2 Attention Extraction

Extract attention maps from each model using multiple methods:

1. **CLS token attention:** Aggregate attention from [CLS] token to spatial patches across heads (DINOv2, DINOv3, MAE, CLIP)
2. **Attention rollout:** Propagate attention through layers to capture indirect dependencies (DINOv2, DINOv3, MAE, CLIP)
3. **Mean attention:** Average attention across patches for models without [CLS] token (SigLIP)
4. **Grad-CAM (baseline):** Gradient-weighted activation maps for ResNet-50 CNN baseline

For each image with expert annotations, generate attention heatmaps at the original image resolution.

### 3.3 Attention-Annotation Alignment

1. Threshold attention maps at 7 percentile levels (top 10%–50% of attention mass)
2. Compute IoU between thresholded regions and expert bounding boxes
3. Compare to baselines (random, center-biased, saliency) — see Section 5

### 3.4 Linear Probe (Sanity Check)

Train linear classifiers on frozen features for 4-class style classification. Confirms models learned discriminative features before analyzing attention.

### 3.5 Fine-Tuning Analysis

Fine-tune each backbone on 4-class style classification (9,485 images), then re-extract attention on the 139 annotated images. Compare Δ IoU (fine-tuned − frozen) per model.

Three fine-tuning strategies are compared (addressing Q4):

| Strategy | Description | Key Parameters |
|:---------|:-----------|:---------------|
| Full fine-tuning | Unfreeze last N backbone layers + classification head | Backbone LR: 1e-5, Head LR: 1e-3 |
| LoRA | Low-rank adapters on backbone attention layers (via HuggingFace PEFT) | rank=8, alpha=32, dropout=0.1 |
| Linear Probe | Freeze backbone, train classification head only | Head LR: 1e-3 |

| Parameter | Value |
|:----------|:------|
| Epochs | 10-20 with early stopping |
| LR schedule | Cosine decay with linear warmup (warmup_ratio=0.1) |
| Gradient clipping | Max norm 1.0 |
| Eval holdout | 139 bbox-annotated images excluded from training split |

### 3.6 Interactive Analysis Tool

Web-based dashboard for exploring attention-annotation alignment:

- **Browser:** 139 annotated churches with model/layer/method selectors
- **Visualization:** Attention heatmap overlay with expert bounding boxes and IoU scores
- **Comparison:** Side-by-side frozen vs fine-tuned, cross-model comparison
- **Metrics:** Model leaderboard by IoU, per-feature breakdown

---

## 4. Ablations

To verify findings are robust to methodological choices:

| Ablation | Variable | Purpose |
|:---------|:---------|:--------|
| Threshold sensitivity | 7 percentile thresholds: top 10%, 15%, 20%, 25%, 30%, 40%, 50% of attention mass | Verify IoU rankings are stable across thresholds |
| Attention method | CLS vs rollout vs mean vs Grad-CAM (per-model where applicable) | Verify findings are not artifacts of extraction method |

---

## 5. Evaluation Plan

### Metrics by Research Question

| RQ | Primary Metric | Statistical Test | Visualization |
|:---|:---------------|:-----------------|:--------------|
| Q1 | Mean IoU (per model, per layer) | Paired t-test across models; bootstrap CIs | Attention heatmaps with bbox overlay |
| Q2 | Δ IoU (fine-tuned − frozen) | Paired t-test (same images) | Side-by-side frozen vs fine-tuned |
| Q3 | Per-head IoU; head specialization index | Rank correlation across heads | Head × feature-type heatmap |
| Q4 | Δ IoU by method; forgetting ratio | Paired t-test with Holm correction | Bar chart by fine-tuning method |

**Secondary metric:** Pointing game accuracy (binary hit: does attention maximum fall within bbox?) with optional 15-pixel tolerance margin per Zhang et al. (2016)

### Baselines

All models compared against:
- Random attention (uniform distribution)
- Center-biased attention (Gaussian prior)
- Low-level saliency (Sobel edges)
- Supervised ResNet-50 (non-SSL reference)

---

## 6. Alignment with ISY5004 Requirements

| Requirement | How Addressed |
| :---- | :---- |
| Intelligent sensing technique | Self-supervised visual feature learning (DINOv2, DINOv3, MAE, CLIP, SigLIP 2) |
| Image/video analytics | Image classification, attention extraction, feature attribution |
| Dataset handling | Public dataset (WikiChurches) with documented preprocessing |
| Experimental comparison | Ablation across 6 models, 4 attention methods, 7 thresholds, multiple layers |
| Literature review | SSL methods, attention visualization, interpretability metrics |
| Practical application | Interactive attention analysis tool for evaluating domain-adapted vision models |
| Final developed system | Web-based dashboard with heatmap explorer, model comparison, and metrics views |
| Commercial solutions comparison | BertViz, Comet ML (visualize but don't quantify); ASCENT-ViT (research, not commercial) |

---

## 7. Timeline (6 Weeks)

| Week | Milestone |
| :---- | :---- |
| 1 | Dataset acquisition, preprocessing, bounding box parsing, feature extraction pipeline |
| 2 | Attention extraction implementation (CLS, rollout, GradCAM), baseline computation |
| 3 | IoU computation pipeline, linear probe training, initial results |
| 4 | Ablation experiments (layers, models, feature categories); **begin fine-tuning** |
| 5 | **Fine-tuning analysis**, statistical comparison (frozen vs fine-tuned), visualization |
| 6 | Report writing, recorded presentation |

---

## 8. Risks and Mitigations

| Risk | Mitigation |
| :---- | :---- |
| Small annotation sample (139 churches) limits statistical power | Report confidence intervals; use bootstrap resampling; acknowledge limitations |
| Attention maps may not reflect true model reasoning | Include GradCAM as gradient-based alternative; triangulate with multiple methods |
| Bounding boxes span only 4 style categories | Frame as focused study; note generalization limitations in discussion |
| Compute constraints (no GPU) | Use ViT-B models only; batch feature extraction; MPS acceleration sufficient |
| IoU may be low across all models | Negative result is still publishable—report honestly what models do attend to |
| DINOv3/SigLIP 2 documentation still sparse | Resolved: both models fully integrated with proper HuggingFace classes (SigLIP using `Siglip2VisionModel`) |
| Fine-tuning may overfit on small style subset | Validation split, early stopping, cosine LR schedule with warmup, gradient clipping, LoRA as parameter-efficient alternative, 139 eval images held out from training. Future work: expand to 5+ classes using unused styles (see Section 11.1) |

---

## 9. Expected Contributions

1. **Benchmark:** Quantitative attention-alignment evaluation on expert-annotated architectural features
2. **Q1:** Empirical comparison of SSL paradigms on expert attention alignment
3. **Q2:** Analysis of how fine-tuning shifts attention toward expert features
4. **Q3:** Identification of attention heads specialized for architectural recognition
5. **Q4:** Trade-off analysis of fine-tuning methods (Linear Probe vs LoRA vs Full)
6. **Deliverable:** Reproducible codebase and interactive analysis tool

---

## 10. Commercial and Research Solutions Comparison

**Visualization Tools:**
- BertViz: Interactive attention head visualization for transformers
- Comet ML: MLOps platform with attention panels
- Hugging Face Transformers: Built-in attention output visualization

These tools visualize attention but do not quantify alignment with domain expertise.

**Research on Attention-Annotation Alignment:**
- ASCENT-ViT (2025): Aligns attention with concept annotations using Pixel TPR
- Medical imaging studies: Evaluate attention against radiologist annotations
- Chefer et al. (2021): Transformer interpretability beyond attention visualization

**Knowledge Gap:** No existing work benchmarks multiple SSL paradigms on the same expert-annotated dataset or measures how fine-tuning shifts attention alignment. We fill this gap with a quantitative benchmark and interactive tool.

---

## 11. Future Extensions

### 11.1 Dataset Expansion: Leveraging Unused Styles

The current implementation uses only 4 architectural styles (Romanesque, Gothic, Renaissance, Baroque), covering 4,790 of 9,346 images (51.3%). The remaining 4,556 images have style labels but are excluded from training. This presents opportunities for future work:

**Current Dataset Utilization:**

| Subset | Images | Purpose |
|--------|--------|---------|
| Annotated (building_parts.json) | 139 | IoU evaluation with expert bounding boxes |
| 4-class labeled (filter_labeled=True) | 4,790 | Fine-tuning and linear probe |
| Other labeled styles | 4,556 | Currently unused |

**Top Unused Styles:**

| Style | Wikidata ID | Images | Notes |
|-------|-------------|--------|-------|
| Neoclassical | Q186363 | 1,587 | Second largest style in dataset |
| Neo-Gothic | Q245188 | 496 | Gothic revival variant |
| Brick Gothic | Q744373 | 486 | Regional Gothic variant |
| Neoclassicism | Q54111 | 350 | Overlaps with Q186363 |
| Baroque Classicism | Q695863 | 272 | Baroque variant |
| Historicism | Q750752 | 215 | 19th century eclectic style |
| Late Gothic | Q930314 | 124 | Temporal Gothic variant |
| Art Nouveau | Q136693 | 122 | Early modern style |

**Expansion Options:**

1. **Add Neoclassical as 5th class** — Instant +1,587 training images with minimal code changes. Neoclassical emerged after Baroque (18th century) and has distinctive visual features (columns, symmetry, pediments).

2. **Merge Gothic variants** — Combine Neo-Gothic, Brick Gothic, and Late Gothic into the Gothic class for +1,106 additional images. Requires careful consideration of whether visual features are sufficiently similar.

3. **Hierarchical classification** — Implement two-level classification (major style → substyle). This would leverage the full dataset while preserving fine-grained distinctions.

4. **Semi-supervised learning** — Use all 9,346 images regardless of style mapping. Unlabeled images could improve learned representations through consistency regularization or pseudo-labeling.

**Implementation Considerations:**

- The annotated subset (139 images) only covers the current 4 styles, so IoU evaluation cannot extend to new classes without additional expert annotations
- Expanding to more classes increases classification difficulty but provides richer evaluation of model capabilities
- Some style boundaries are fuzzy (e.g., Neoclassicism vs Baroque Classicism), which may affect label quality

**Code Changes Required:**

To add Neoclassical as a 5th class, modify `src/ssl_attention/config.py`:

```python
STYLE_MAPPING: dict[str, int] = {
    "Q46261": 0,   # Romanesque
    "Q176483": 1,  # Gothic
    "Q236122": 2,  # Renaissance
    "Q840829": 3,  # Baroque
    "Q186363": 4,  # Neoclassical (NEW)
}

STYLE_NAMES: tuple[str, ...] = ("Romanesque", "Gothic", "Renaissance", "Baroque", "Neoclassical")
NUM_STYLES: int = len(STYLE_MAPPING)
```

---

## References

- Barz, B., & Denzler, J. (2021). WikiChurches: A Fine-Grained Dataset of Architectural Styles with Real-World Challenges. *NeurIPS Datasets and Benchmarks*.  
- Chefer, H., Gur, S., & Wolf, L. (2021). Transformer Interpretability Beyond Attention Visualization. *CVPR*.  
- Caron, M., et al. (2021). Emerging Properties in Self-Supervised Vision Transformers. *ICCV*.  
- He, K., et al. (2022). Masked Autoencoders Are Scalable Vision Learners. *CVPR*.  
- Oquab, M., et al. (2023). DINOv2: Learning Robust Visual Features without Supervision. *arXiv:2304.07193*.  
- Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. *ICML*.  
- Simeoni, O., et al. (2025). DINOv3. *arXiv:2508.10104*.
- Zhai, X., et al. (2023). Sigmoid Loss for Language Image Pre-training. *ICCV*.
- Tschannen, M., et al. (2025). SigLIP 2: A Better Multilingual Vision Language Encoder. *arXiv*.
- Voita, E., et al. (2019). Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting. *ACL*. [arXiv:1905.09418](https://arxiv.org/abs/1905.09418)
- Hu, E.J., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR*. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Biderman, S., et al. (2024). LoRA Learns Less and Forgets Less. *TMLR*. [arXiv:2405.09673](https://arxiv.org/abs/2405.09673)


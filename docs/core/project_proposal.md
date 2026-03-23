# Project Proposal: Do Self-Supervised Vision Models Learn What Experts See?

## Attention Alignment with Human-Annotated Architectural Features

**Course:** ISY5004 Intelligent Sensing Systems Practice Module
**Team Size:** 3 members
**Duration:** 6 weeks

---

## 1. Problem Statement

Self-supervised learning (SSL) models like DINOv2, DINOv3, MAE, CLIP, SigLIP, and SigLIP 2 learn visual representations without labels and achieve strong performance on downstream tasks. However, a fundamental question remains: **do these models attend to the same visual features that human experts consider diagnostic, or do they exploit statistical shortcuts invisible to humans?** Furthermore, when models are fine-tuned for a specific task, **does their attention shift toward expert-identified features, or do they discover alternative discriminative regions?**

Existing SSL benchmarks measure classification accuracy but do not explain *which* image regions drive predictions. This matters for trust and deployment: a model that correctly classifies Gothic architecture by attending to flying buttresses is qualitatively different from one that exploits dataset-specific background correlations. This project addresses that gap by quantitatively measuring alignment between SSL attention patterns and expert-annotated "characteristic architectural features" using both threshold-dependent (IoU) and threshold-free (MSE, KL divergence, EMD) metrics.

Existing interpretability tools address parts of this problem but leave a critical gap. Visualization tools like BertViz (Vig, ACL 2019) render attention weights interactively but provide no quantitative metrics. Attribution frameworks like Captum measure whether explanations are faithful to the model's own decisions (infidelity, sensitivity) but not whether the model attends to features that domain experts consider diagnostic. Recent work in medical imaging has begun evaluating attention against radiologist annotations (Komorowski et al., CVPR 2023), but no existing benchmark compares multiple SSL paradigms on the same expert-annotated dataset, uses both discrete and continuous alignment metrics, or measures how fine-tuning shifts attention alignment. We address this gap with a multi-metric quantitative benchmark and an interactive analysis tool.

### Research Questions and Approaches

| Research Question | Approach | Tool/Method |
|-------------------|----------|-------------|
| **Q1:** Do SSL models attend to the same features human experts consider diagnostic? | Compute IoU, MSE, KL divergence, and EMD between attention maps and expert bounding boxes across 7 models and 12 layers | Attention heatmap overlay, multi-metric dashboard, model leaderboard |
| **Q2:** Does fine-tuning shift attention toward expert-identified features, and does the strategy (Linear Probe vs LoRA vs Full) matter? | Compare Δ IoU (fine-tuned − frozen) with paired statistical tests on same images; classify outcomes as Preserve, Enhance, or Destroy; compare across three fine-tuning methods using paired Wilcoxon tests with Holm correction; compute Cohen's d effect sizes | Frozen vs fine-tuned comparison view, attention shift visualization, Δ IoU bar charts by method |
| **Q3:** Do individual attention heads specialize for different architectural features, and which heads best align with expert annotations? | Compute per-head IoU separately for each of the 12 attention heads; identify heads with consistently highest alignment using rank-based analysis | Per-head attention selector, head IoU heatmap (head × feature type), head specialization dashboard |

> **Enhancement docs:** Q3 is explored in detail in [Per-Head Attention Visualization](../enhancements/per_attention_head.md). Q2 fine-tuning strategies are detailed in [Fine-Tuning Methods](../enhancements/fine_tuning_methods.md).

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

**Known Limitation — Sparse Annotation Bias:** The WikiChurches dataset annotates representative instances of each feature type, not exhaustive ones. When a model attends to all round arch windows on a facade but only one is annotated, per-bbox IoU is deflated by "false positives." This is mitigated through cross-metric validation (Pointing Game and Coverage corroborate IoU findings) and documented in detail in [Sparse Annotation Bias](../enhancements/sparse_annotation_bias.md).

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
| SigLIP ViT-B/16 | 86M | Contrastive language-image (sigmoid) | `google/siglip-base-patch16-224` |
| SigLIP 2 ViT-B/16 | 86M | Contrastive language-image (sigmoid) + dense features | `google/siglip2-base-patch16-224` |
| ResNet-50 | 23M | Supervised (ImageNet) | `torchvision` |

All ViT models use ViT-B architecture (12 layers, 12 heads, 768-dim embeddings) for controlled comparison. ResNet-50 is included as a supervised CNN baseline. DINOv2 uses a 16×16 patch grid (patch_size=14, yielding 256 spatial tokens); all other ViTs use a 14×14 grid (patch_size=16, yielding 196 spatial tokens). DINOv2 and DINOv3 include 4 register tokens.

### 3.2 Attention Extraction

Extract attention maps from each model using multiple methods:

1. **CLS token attention:** Aggregate attention from [CLS] token to spatial patches across heads (DINOv2, DINOv3, MAE, CLIP). Supports head fusion strategies: mean (default), max, or min across the 12 attention heads.
2. **Attention rollout:** Propagate attention through layers to capture indirect dependencies (DINOv2, DINOv3, MAE, CLIP). Recursive layer-wise multiplication following Abnar & Zuidema (2020).
3. **Mean attention:** Average attention across all tokens for models without [CLS] token (SigLIP, SigLIP 2). Supports the same head fusion strategies.
4. **Grad-CAM (baseline):** Gradient-weighted activation maps across all 4 ResNet stages (7×7 final feature grid).

For each image with expert annotations, generate attention heatmaps at the original image resolution.

### 3.3 Attention-Annotation Alignment

Alignment is evaluated using both discrete (threshold-dependent) and continuous (threshold-free) metrics:

#### 3.3a Discrete Metrics

1. **IoU (Intersection over Union):** Threshold attention maps at 5 percentile levels (top 10%, 20%, 30%, 40%, 50% of attention mass) using `torch.topk` for exact pixel counts. Compute IoU between thresholded binary regions and expert bounding box masks. Primary metric reported at the 90th percentile.
2. **Pointing Game:** Binary hit metric — does the single maximum attention point fall within any expert bounding box? Includes optional tolerance margin per Zhang et al. (2016). Extended with top-k variant (do k highest attention points hit bboxes?).
3. **Coverage (Energy):** Threshold-free metric measuring the fraction of total attention energy that falls inside annotated regions. Immune to threshold choice artifacts.

#### 3.3b Continuous Metrics

Continuous metrics compare the raw attention map against a **soft ground truth** generated from expert bounding boxes. Each bounding box is converted to an anisotropic Gaussian heatmap (σ = bbox_size / 4), and multiple boxes per image are combined via pixelwise maximum (soft union).

1. **MSE (Mean Squared Error):** Compare normalized attention map against the Gaussian soft ground truth. Lower is better. Measures absolute deviation between attention intensity and expert-expected focus.
2. **KL Divergence:** Normalize both attention map and ground truth as probability distributions (sum to 1). Compute KL(GT ‖ attention). Heavily penalizes models that assign high attention to areas where experts assign none. Based on methodology from Jain & Wallace (2019).
3. **EMD (Earth Mover's Distance / Wasserstein-1):** Compute the optimal transport cost between attention and ground truth distributions on an 8×8 shared support grid. Accounts for spatial distance — a model attending slightly beside a feature is penalized less than one attending to an entirely different region. Based on methodology from Zhou et al. (2020).

Together, these six metrics distinguish "looking at the right place" (IoU, Pointing Game) from "how attention mass is distributed" (Coverage, MSE, KL, EMD). For detailed definitions, thresholding methodology, and known limitations, see the [Metrics Methodology Reference](../reference/metrics_methodology.md).

### 3.4 Linear Probe (Sanity Check)

Train linear classifiers on frozen features for 4-class style classification. Confirms models learned discriminative features before analyzing attention.

### 3.5 Fine-Tuning Analysis

Fine-tune each backbone on 4-class style classification (using the style-labeled split of ~4,790 images), then re-extract attention on the 139 annotated images (held out from training). Compare Δ IoU (fine-tuned − frozen) per model.

Three fine-tuning strategies are compared (addressing Q2):

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
| Loss | Cross-entropy with class weights (handles style imbalance) |
| Eval holdout | 139 bbox-annotated images excluded from training split |

**Preserve / Enhance / Destroy Taxonomy:** Fine-tuning outcomes are classified using paired Wilcoxon signed-rank tests with Holm-Bonferroni correction for multiple comparisons:

| Category | Definition | Interpretation |
|----------|-----------|----------------|
| **Enhance** | Δ IoU > 0, statistically significant after correction | Fine-tuning shifted attention *toward* expert features |
| **Preserve** | Δ IoU ≈ 0, not significant | Pre-trained attention patterns retained; task solved without attention shift |
| **Destroy** | Δ IoU < 0, statistically significant after correction | Attention drift — fine-tuning shifted focus *away* from expert features |

Effect sizes are reported using Cohen's d (paired d_z) to quantify the magnitude of attention shifts beyond statistical significance.

### 3.6 Interactive Analysis Tool

Web-based dashboard (FastAPI backend + React frontend) for exploring attention-annotation alignment:

- **Image browser:** 139 annotated churches with model/layer/method selectors
- **Visualization:** Attention heatmap overlay with expert bounding boxes and per-metric scores
- **Metrics dashboard:** All 6 metrics (IoU, Coverage, MSE, KL, EMD, Pointing Game) with model leaderboard
- **Fine-tuning comparison:** Side-by-side frozen vs fine-tuned attention with Δ IoU overlay
- **Method comparison:** Linear Probe vs LoRA vs Full comparison across models
- **Per-bbox drill-down:** Select individual bounding boxes to see per-feature metrics computed on-the-fly

---

## 4. Ablations

To verify findings are robust to methodological choices:

| Ablation | Variable | Purpose |
|:---------|:---------|:--------|
| Threshold sensitivity | 5 percentile thresholds: top 10%, 20%, 30%, 40%, 50% of attention mass | Verify IoU rankings are stable across thresholds |
| Attention method | CLS vs rollout vs mean vs Grad-CAM (per-model where applicable) | Verify findings are not artifacts of extraction method |
| Cross-metric validation | IoU vs MSE vs KL vs EMD rankings | Verify model rankings are consistent across discrete and continuous metrics |
| Fine-tuning strategy comparison | Linear Probe vs LoRA vs Full across all models | Paired tests with Holm correction to determine which strategies significantly shift attention |

---

## 5. Evaluation Plan

### Metrics by Research Question

| RQ | Primary Metrics | Statistical Test | Visualization |
|:---|:---------------|:-----------------|:--------------|
| Q1 | IoU (at 90th percentile), MSE, KL divergence, EMD, Coverage, Pointing Game | Paired t-test and Wilcoxon signed-rank across models; bootstrap CIs; Holm-Bonferroni correction for pairwise comparisons | Attention heatmaps with bbox overlay; model leaderboard |
| Q2 | Δ IoU, Δ MSE (fine-tuned − frozen); Preserve/Enhance/Destroy classification | Paired Wilcoxon tests on same 139 images; Holm correction across 18 model-strategy combinations; Cohen's d effect sizes | Side-by-side frozen vs fine-tuned; Δ IoU bar charts by method and model |
| Q3 | Per-head IoU; head specialization index | Rank correlation across heads | Head × feature-type heatmap |

### Baselines

All models compared against:
- Random attention (uniform distribution)
- Center-biased attention (Gaussian prior)
- Low-level saliency (Sobel edges)
- Saliency prior (center bias + border suppression)
- Supervised ResNet-50 (non-SSL reference)

---

## 6. Alignment with ISY5004 Requirements

| Requirement | How Addressed |
| :---- | :---- |
| Intelligent sensing technique | Self-supervised visual feature learning (DINOv2, DINOv3, MAE, CLIP, SigLIP, SigLIP 2) |
| Image/video analytics | Image classification, attention extraction, feature attribution |
| Dataset handling | Public dataset (WikiChurches) with documented preprocessing |
| Experimental comparison | Ablation across 7 models, 4 attention methods, 6 alignment metrics, 5 percentile thresholds, 3 fine-tuning strategies |
| Literature review | SSL methods, attention visualization, interpretability metrics |
| Practical application | Interactive attention analysis tool for evaluating domain-adapted vision models |
| Final developed system | Web-based dashboard with heatmap explorer, model comparison, fine-tuning analysis, and metrics views |
| Commercial solutions comparison | BertViz (visualize only); Captum (faithfulness metrics, not expert alignment); Komorowski et al. (medical imaging only) |

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
| Sparse annotation bias | Representative (not exhaustive) bbox annotations may deflate per-bbox IoU; mitigated via cross-metric validation (Pointing Game + Coverage corroborate IoU findings); documented in [Sparse Annotation Bias](../enhancements/sparse_annotation_bias.md) |
| Documentation drift between model keys and runtime behavior | Mitigate with periodic doc sync against `src/ssl_attention/config.py` and backend validators; keep `siglip` and `siglip2` documented as separate canonical keys |
| Fine-tuning may overfit on small style subset | Validation split, early stopping, cosine LR schedule with warmup, gradient clipping, class-weighted loss, LoRA as parameter-efficient alternative, 139 eval images held out from training. Future work: expand to 5+ classes using unused styles (see Section 11.1) |

---

## 9. Expected Contributions

1. **Multi-metric benchmark:** Quantitative attention-alignment evaluation on expert-annotated architectural features using 6 complementary metrics (IoU, Coverage, MSE, KL, EMD, Pointing Game) — combining threshold-dependent and threshold-free approaches
2. **Q1:** Empirical comparison of 7 models spanning 4 SSL paradigms plus a supervised baseline on expert attention alignment
3. **Q2:** Preserve / Enhance / Destroy taxonomy for classifying how fine-tuning shifts attention toward expert features, with Holm-corrected paired statistical tests and effect sizes across 3 strategies (Linear Probe, LoRA, Full)
4. **Q3:** Identification of attention heads specialized for architectural recognition (planned)
5. **Continuous metrics methodology:** Gaussian soft ground truth generation from expert bounding boxes, enabling distribution-based alignment evaluation (MSE, KL, EMD)
6. **Deliverable:** Reproducible codebase and interactive analysis tool

---

## 10. Commercial and Research Solutions Comparison

**Visualization Tools** (qualitative only):
- BertViz (Vig, ACL 2019): Interactive attention head visualization for transformers
- Hugging Face Transformers: Built-in attention output access
- Ecco (Alammar, ACL 2021): Interactive NLP model interpretability

**Attribution Frameworks** (quantify faithfulness to model, not alignment with expertise):
- Captum (Kokhlikyan et al., 2020): Infidelity, sensitivity metrics for attribution methods
- pytorch-grad-cam (Gil, 2021): Insertion/deletion, ROAD metrics for CAM evaluations
- Chefer et al. (CVPR 2021): Transformer interpretability beyond attention visualization

**Research on Attention-Expert Alignment** (domain-specific, no cross-model benchmark):
- Komorowski et al. (CVPR 2023): ViT explanations evaluated against radiologist bbox annotations
- ASCENT-ViT (2025): Aligns attention with concept annotations using Pixel TPR
- Pointing Game (Zhang et al., ECCV 2016): Binary hit/miss against ground-truth masks

**Knowledge Gap:** Attribution frameworks quantify whether explanations faithfully reflect model decisions, but not whether the model attends to features that domain experts consider diagnostic. The medical imaging studies above evaluate attention against expert annotations, but no existing work benchmarks multiple SSL paradigms on the same expert-annotated dataset using both discrete and continuous alignment metrics, or measures how fine-tuning shifts attention alignment with a formal Preserve/Enhance/Destroy taxonomy. Recent work by Chung et al. (2025) found that "DINO does NOT necessarily outperform supervised/MAE on medical data," strengthening the case for domain-specific evaluation rather than assuming paradigm superiority. We fill this gap with a cross-model, multi-metric quantitative benchmark and interactive analysis tool.

---

## 11. Future Extensions

### 11.1 Dataset Expansion: Leveraging Unused Styles

The current implementation uses only 4 architectural styles (Romanesque, Gothic, Renaissance, Baroque), covering 4,790 of 9,346 style-labeled images (51.3%). The remaining 4,556 labeled images have other styles and are excluded from current training. This presents opportunities for future work:

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

4. **Semi-supervised learning** — Use all available images (9,485 in the official release, plus any local extensions) regardless of style mapping. Unlabeled images could improve learned representations through consistency regularization or pseudo-labeling.

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

### 11.2 Planned Enhancements (from Professor Feedback)

The following enhancements address feedback from the project supervisor (tracked in [GitHub Issue #4](https://github.com/DesmondChoy/ssl_wikichurches/issues/4)):

#### Cross-Layer Aggregation (Issue #4, Item 3)

Current analysis evaluates each of the 12 transformer layers independently. Cross-layer aggregation would construct unified attention maps by combining information across layers:

1. **Max pooling across layers** — Pixel-wise maximum across all 12 layer attention maps. Ensures the final map highlights any pixel strongly attended to by any layer. Captures both early-layer edge/texture signals and deep-layer semantic groupings.

2. **Mean pooling with depth-decay weighting** — Average maps across layers with exponential decay weighting (deeper layers weighted higher). Deep layers contain more semantic, object-level information; weighting prevents noisy early layers from washing out clear semantic signals.

3. **ALTI (Aggregation of Layer-wise Token-to-Token Interactions)** — Recursive backward propagation of attention matrices, tracking token contributions more accurately than standard rollout. State-of-the-art in mechanistic interpretability. Reference: Ferrando et al. (EMNLP 2022).

4. **Learnable linear aggregation** — Train 12 scalar weights to optimally combine layer maps for maximum IoU on a validation set. The learned weights reveal which layers encode the most human-aligned features.

#### Unimodal vs Multimodal Leaderboard (Issue #4, Item 4)

Test the hypothesis: does forcing a model to align images with human language (multimodal training) make its visual attention more aligned with human visual annotations?

1. **Split-category leaderboard** — Group models into Unimodal SSL (DINOv2, DINOv3, MAE) and Multimodal VLMs (CLIP, SigLIP, SigLIP 2) with sub-group averages. ResNet-50 as separate supervised baseline.

2. **Zero-shot text prompting evaluation (VLMs only)** — For CLIP and SigLIP, extract "text-prompted" attention by computing similarity between image patches and architectural feature text (e.g., "A photo of a rose window"). Compare text-guided attention against standard self-attention to test whether VLMs *need* language to find expert features.

3. **Emergent object semantics vs global context** — Analyze whether unimodal models (DINO) achieve higher IoU on small structural features (arches, windows) while multimodal models (CLIP) perform better on full building shapes, as suggested by the DINO and CLIP literature.

### 11.3 Additional Planned Work

- **AUC-Judd / AUC-ROC:** Threshold-independent metric treating raw continuous attention values as confidence scores predicting the binary expert bounding box mask. Provides a single robust alignment score without needing to pick a percentile threshold, complementing the existing IoU-based analysis.

- **CKA (Centered Kernel Alignment):** Measure representational drift across layers between frozen and fine-tuned models. When a model shows attention Destroy behavior (IoU drops), CKA diagnoses *where* in the network catastrophic forgetting occurred (early vs late layers). Reference: Kornblith et al. (ICML 2019).

- **Attention Transfer fine-tuning:** Add a loss penalty that limits how much attention maps can diverge from the frozen model during fine-tuning, testing whether preserving attention patterns mitigates catastrophic forgetting. Reference: SATS (Gupta et al., 2022).

- **Feature-specific forgetting analysis:** Break down the Preserve/Enhance/Destroy taxonomy by specific architectural features (e.g., rose windows vs flying buttresses vs pointed arches). A model might Enhance attention on obvious features but Destroy attention on subtle structural features.

---

## References

- Abnar, S., & Zuidema, W. (2020). Quantifying Attention Flow in Transformers. *ACL*.
- Barz, B., & Denzler, J. (2021). WikiChurches: A Fine-Grained Dataset of Architectural Styles with Real-World Challenges. *NeurIPS Datasets and Benchmarks*.
- Biderman, S., et al. (2024). LoRA Learns Less and Forgets Less. *TMLR*. [arXiv:2405.09673](https://arxiv.org/abs/2405.09673)
- Caron, M., et al. (2021). Emerging Properties in Self-Supervised Vision Transformers. *ICCV*.
- Chefer, H., Gur, S., & Wolf, L. (2021). Transformer Interpretability Beyond Attention Visualization. *CVPR*.
- Ferrando, J., et al. (2022). Measuring the Mixing of Contextual Information in the Transformer. *EMNLP*.
- He, K., et al. (2022). Masked Autoencoders Are Scalable Vision Learners. *CVPR*.
- Hu, E.J., et al. (2022). LoRA: Low-Rank Adaptation of Large Language Models. *ICLR*. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685)
- Jain, S., & Wallace, B. (2019). Attention is Not Explanation. *NAACL*.
- Kornblith, S., et al. (2019). Similarity of Neural Network Representations Revisited. *ICML*.
- Oquab, M., et al. (2023). DINOv2: Learning Robust Visual Features without Supervision. *arXiv:2304.07193*.
- Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. *ICML*.
- Simeoni, O., et al. (2025). DINOv3. *arXiv:2508.10104*.
- Voita, E., et al. (2019). Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting. *ACL*. [arXiv:1905.09418](https://arxiv.org/abs/1905.09418)
- Zhang, J., et al. (2016). Top-Down Neural Attention by Excitation Backprop. *ECCV*.
- Zhai, X., et al. (2023). Sigmoid Loss for Language Image Pre-training. *ICCV*.
- Zhou, Y., et al. (2020). DeepEMD: Few-Shot Image Classification with Differentiable Earth Mover's Distance. *CVPR*.
- Tschannen, M., et al. (2025). SigLIP 2: A Better Multilingual Vision Language Encoder. *arXiv*.

# Fine-Tuning Methods for Studying Attention Shift in SSL Vision Models

> **Research compiled:** February 2026
> **Purpose:** Evaluate fine-tuning strategies to study whether task-specific training redirects SSL model attention toward expert-annotated architectural features

> **Related documents:**
> - [Project Proposal — Q2: Fine-Tuning](../core/project_proposal.md#research-questions-and-approaches)
> - [Implementation Plan — Phase 5: Fine-Tuning Analysis](../core/implementation_plan.md#phase-5-fine-tuning-analysis--in-progress)

## Executive Summary

This document outlines fine-tuning methods suitable for the SSL WikiChurches project. It supports:
- **Research Question 2**: Does fine-tuning shift attention toward expert-identified features, and does the strategy (Linear Probe vs LoRA vs Full) matter?

**Key insight:** The Δ IoU metric (post-fine-tuning IoU − pre-fine-tuning IoU) directly measures whether fine-tuning improves alignment between model attention and human expert annotations.
App/API model keys in this project are `dinov2`, `dinov3`, `mae`, `clip`, `siglip`, `siglip2`, and `resnet50`.
Fine-tuning is supported for `dinov2`, `dinov3`, `mae`, `clip`, `siglip`, and `siglip2` (`resnet50` excluded).
For `siglip`/`siglip2`, attention analysis uses the mean-attention path (no CLS-token method).

---

## 1. Why Fine-Tuning Makes Sense

### The Natural Experiment

1. **Pre-trained models learned general features** — DINOv2, CLIP, MAE, SigLIP, SigLIP 2, etc. were trained on diverse datasets (ImageNet, LAION) without knowledge of architectural styles
2. **Fine-tuning creates task specificity** — Training on style classification forces models to learn which visual features distinguish Romanesque from Gothic, Renaissance from Baroque
3. **Attention as a window into learning** — By comparing attention maps before and after fine-tuning, we can observe *what* the model learns to focus on

### The Hypothesis

> **H0:** Fine-tuning does not significantly change attention alignment with expert annotations (Δ IoU ≈ 0)
> **H1:** Fine-tuning shifts attention toward expert-annotated architectural features (Δ IoU > 0)

---

## 2. Recommended Fine-Tuning Strategy: Tiered Comparison

We recommend a **3-tier comparison** to isolate what drives attention shift:

### Tier 1: Linear Probe (Minimal Intervention)

**What it is:** Freeze the entire pre-trained backbone, train only a classification head.

**Why it's suitable:**
- Establishes a **baseline** where attention maps are *unchanged* from pre-training
- Any IoU differences would be due to model architecture, not fine-tuning
- DINOv2/DINOv3 are specifically designed for this — frozen features perform remarkably well

**Implementation:**
```python
# Freeze backbone, train linear head only
for param in model.backbone.parameters():
    param.requires_grad = False
classifier = nn.Linear(768, num_classes)  # ViT-B hidden dimension
```

**Expected attention shift:** None (Δ IoU ≈ 0)

**Implementation note:** The linear probe uses an `sklearn.Pipeline` that wraps `StandardScaler` + `LogisticRegression`. This ensures the scaler is fit only on each cross-validation training fold, preventing data leakage from test-fold statistics.

**References:**
- [Understanding DINOv2: Engineer's Deep Dive](https://www.lightly.ai/blog/dinov2) — DINOv2 excels as frozen feature extractor
- [DINOv3 Technical Overview](https://arxiv.org/html/2508.10104v1) — "With DINOv3, finetuning is not necessary to obtain strong performance"

---

### Tier 2: LoRA Fine-Tuning (Parameter-Efficient)

**What it is:** Inject trainable low-rank matrices into attention layers while keeping backbone mostly frozen.

**Why it's suitable:**
- Reduces trainable parameters by ~85M while still adapting attention mechanisms
- Modifies *how* attention is computed, allowing measurable Δ IoU
- Preserves most pre-trained representations, avoiding catastrophic forgetting
- 25.8% faster training than full fine-tuning

**Implementation with PEFT:**
```python
from peft import LoraConfig, get_peft_model

config = LoraConfig(
    r=8,                                    # Low-rank dimension
    lora_alpha=32,                          # Scaling factor
    target_modules=["query", "value"],      # Q, V projections in attention
    lora_dropout=0.1,
)
model = get_peft_model(base_model, config)
# Trainable params: ~300K vs ~86M for full fine-tuning
```

**Expected attention shift:** Moderate (measurable Δ IoU)

**References:**
- [Parameter-Efficient Fine-Tuning of DINOv2 for Lung Nodule Classification](https://ieeexplore.ieee.org/document/10635887/) — LoRA achieves 1.7% improvement over standard fine-tuning
- [PEFT Comprehensive Survey (arXiv 2024)](https://arxiv.org/pdf/2403.14608) — Overview of parameter-efficient methods
- [Dynamic Tuning for ViT Adaptation (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/d0241a0fb1fc9be477bdfde5e0da276a-Paper-Conference.pdf)

---

### Tier 3: Full Fine-Tuning (Maximum Adaptation)

**What it is:** Unfreeze all parameters and train end-to-end.

**Why it's suitable:**
- Maximum flexibility for attention to shift toward task-relevant features
- Can cause dramatic attention changes
- Provides an **upper bound** on how much attention *can* shift

**Caution — Catastrophic Forgetting:**
- DINO ViT-Base/16 loses over 70% ImageNet accuracy after just 10 iterations of fine-tuning on CIFAR-100
- May need careful learning rate scheduling (lower LR for backbone, higher for head)
- Consider early stopping based on validation IoU, not just classification accuracy

**Implementation:**
```python
# Differential learning rates
optimizer = torch.optim.AdamW([
    {"params": model.backbone.parameters(), "lr": 1e-5},   # Lower for backbone
    {"params": model.classifier.parameters(), "lr": 1e-4}, # Higher for head
], weight_decay=0.01)
```

**Expected attention shift:** Large (potentially significant Δ IoU)

**References:**
- [Catastrophic Forgetting in SSL ViTs](https://arxiv.org/html/2404.17245v1) — Detailed analysis of forgetting dynamics
- [DINOv2 for Image Classification: Fine-Tuning vs Transfer Learning](https://debuggercafe.com/dinov2-for-image-classification-fine-tuning-vs-transfer-learning/)

---

## 3. Advanced Method: Attention-Supervised Fine-Tuning

This method is particularly relevant because we have expert bounding box annotations that can directly supervise attention.

### The HuMAL Approach (Human-Machine Attention Learning)

Research shows that regularizing model attention to align with human attention improves both interpretability AND accuracy.

**Implementation:**
```python
def attention_alignment_loss(model_attention, expert_bbox_mask):
    """
    Compute loss between model attention and expert bounding box regions.

    Args:
        model_attention: Attention map from model [B, H, W]
        expert_bbox_mask: Binary mask from bounding boxes [B, H, W]
    """
    # Normalize attention map
    attn_map = model_attention.mean(dim=1)  # Average across heads if needed
    attn_map = attn_map / attn_map.sum(dim=(-2, -1), keepdim=True)

    # Create target from bounding boxes (soft mask)
    target = create_mask_from_bboxes(expert_bbox_mask)
    target = target / target.sum(dim=(-2, -1), keepdim=True)

    # Cosine similarity loss
    return 1 - F.cosine_similarity(
        attn_map.flatten(start_dim=1),
        target.flatten(start_dim=1)
    ).mean()

# Combined training loss
total_loss = classification_loss + λ * attention_alignment_loss
```

**Why it's suitable:**
- Directly tests the hypothesis: *Can we force attention toward expert features?*
- Research shows this improves both interpretability AND accuracy
- We already have the bbox annotations needed (631 boxes across 139 images)

**Caveat:** This would create *artificially* high Δ IoU — useful for understanding the *maximum possible* alignment, but should be analyzed separately from standard fine-tuning comparisons.

**References:**
- [HuMAL: Aligning Human and Machine Attention](https://arxiv.org/html/2502.06811v1) — "Regularization on the loss of the last attention layer yielded the best results"
- [Localization-Guided Medical Vision Transformer](https://link.springer.com/chapter/10.1007/978-3-031-92648-8_8) — Uses foreground masks to regularize ViT attribution maps
- [GMAR: Gradient-Driven Multi-Head Attention Rollout](https://arxiv.org/html/2504.19414v1) — Shows not all attention heads are equally meaningful

---

## 4. Additional Methods to Consider

### Core-Tuning (Contrastive-Regularized Fine-Tuning)

Addresses the distribution shift between contrastive pre-training and classification fine-tuning.

**Key insight:** Contrastive pre-training treats same-class images as negatives, pushing them apart in feature space. This hampers fine-tuning.

**Solution:** Mine hard positive/negative pairs during fine-tuning to maintain contrastive structure while learning class boundaries.

**References:**
- [Core-tuning: NeurIPS 2021](https://github.com/Vanint/Core-tuning) — "Promising results on image classification and semantic segmentation"

### Contrastive Initialization (COIN)

Add an intermediate stage before fine-tuning to increase inter-class discrepancy and intra-class compactness.

**References:**
- [Improving Fine-tuning with Contrastive Initialization](https://arxiv.org/abs/2208.00238)

### Block Expansion

Adds new transformer blocks that can be trained while keeping original blocks frozen.

**References:**
- [Parameter Efficient Fine-tuning without Catastrophic Forgetting](https://arxiv.org/html/2404.17245v1)

---

## 5. Data Strategy

### Available Data

| Dataset | Size | Labels | Use |
|---------|------|--------|-----|
| Bbox-annotated images | 139 images, 631 boxes | Expert architectural features | **Evaluation only** |
| Style-labeled (4 classes) | 4,790 images | Romanesque, Gothic, Renaissance, Baroque | Fine-tuning training |
| Full WikiChurches | 9,485 images (official release) | Various style labels | Semi-supervised expansion |

### Recommended Split

```
┌─────────────────────────────────────────────────────────────┐
│  TRAINING SET: 4,651 images                                 │
│  └─ Style-labeled images EXCLUDING bbox-annotated set       │
├─────────────────────────────────────────────────────────────┤
│  VALIDATION SET: ~500 images                                │
│  └─ 10-15% of training set (stratified by style)            │
├─────────────────────────────────────────────────────────────┤
│  EVALUATION SET: 139 images (HELD OUT COMPLETELY)           │
│  └─ Bbox-annotated images for IoU measurement               │
│  └─ Used ONLY for pre/post fine-tuning IoU comparison       │
└─────────────────────────────────────────────────────────────┘
```

### Why This Design

1. **No data leakage:** The 139 bbox images must be completely held out from training
2. **Sufficient training data:** 4,651 images is plenty for ViT fine-tuning (especially with LoRA)
3. **Clean experimental design:** Measure IoU before → Fine-tune → Measure IoU after on *same* 139 images

### Optional: Semi-Supervised Expansion

To leverage all 9,485 images:
- **Pseudo-labeling:** Use confident predictions from fine-tuned model on unlabeled images
- **Consistency regularization:** Enforce consistent predictions across augmented views
- **Self-training:** Iteratively expand training set with high-confidence predictions

**References:**
- [Pseudo-Label Enhancement for WSOD Using Self-Supervised ViT](https://www.sciencedirect.com/science/article/abs/pii/S0950705125000607)

---

## 6. Experimental Design

```
┌─────────────────────────────────────────────────────────────┐
│  PHASE 1: BASELINE MEASUREMENT                              │
│  ├─ Compute IoU for all 6 models (frozen, pre-trained)      │
│  ├─ Record per-image, per-feature-type IoU                  │
│  └─ This establishes "before" measurement                   │
├─────────────────────────────────────────────────────────────┤
│  PHASE 2: FINE-TUNING (3 strategies × 6 models = 18 runs)   │
│  ├─ Linear Probe: Train classification head only            │
│  ├─ LoRA (r=8): Adapt attention Q/V projections             │
│  └─ Full Fine-tune: End-to-end training                     │
├─────────────────────────────────────────────────────────────┤
│  PHASE 3: POST-FINE-TUNING MEASUREMENT                      │
│  ├─ Compute IoU for all 18 fine-tuned variants              │
│  ├─ Calculate Δ IoU = IoU_after - IoU_before                │
│  └─ Compute classification accuracy on held-out set         │
├─────────────────────────────────────────────────────────────┤
│  PHASE 4: STATISTICAL ANALYSIS                              │
│  ├─ Paired t-tests: Is Δ IoU significantly > 0?             │
│  ├─ Effect sizes: Cohen's d for each comparison             │
│  ├─ Bootstrap confidence intervals                          │
│  └─ Holm correction for multiple comparisons                │
├─────────────────────────────────────────────────────────────┤
│  PHASE 5: DETAILED ANALYSIS                                 │
│  ├─ Which models show largest positive Δ IoU?               │
│  ├─ Does fine-tuning method matter (Linear vs LoRA vs Full)?│
│  ├─ Per-feature breakdown: Which architectural features     │
│  │   show most attention improvement?                       │
│  ├─ Layer analysis: At which layer does alignment emerge?   │
│  └─ Attention head analysis: Which heads specialize?        │
└─────────────────────────────────────────────────────────────┘
```

---

## 7. Method Comparison Summary

| Method | Trainable Params | Expected Δ IoU | Forgetting Risk | Compute Cost | Recommended |
|--------|-----------------|----------------|-----------------|--------------|-------------|
| Linear Probe | ~3K | None (baseline) | None | Low | ✅ Yes |
| LoRA (r=8) | ~300K | Moderate | Low | Medium | ✅ Yes |
| Full Fine-tune | ~86M | High | High | High | ✅ Yes |
| Attention-Supervised | ~86M | Very High* | Medium | High | ⚠️ Separate analysis |
| Core-tuning | ~86M | Moderate-High | Low | High | Optional |

*Artificially inflated due to direct supervision

---

## 8. Implementation Recommendations

### Suggested Hyperparameters

```python
# Common settings
common_config = {
    "image_size": 224,
    "batch_size": 32,
    "epochs": 20,
    "warmup_ratio": 0.1,
    "weight_decay": 0.01,
    "scheduler": "cosine",
    "num_classes": 4,  # Romanesque, Gothic, Renaissance, Baroque
}

# Linear Probe
linear_probe_config = {
    **common_config,
    "learning_rate": 1e-3,
    "epochs": 50,  # Can train longer since only head
}

# LoRA Fine-tuning
lora_config = {
    **common_config,
    "learning_rate": 1e-4,
    "lora_r": 8,
    "lora_alpha": 32,
    "lora_dropout": 0.1,
    "target_modules": ["query", "value"],
}

# Full Fine-tuning
full_ft_config = {
    **common_config,
    "backbone_lr": 1e-5,
    "head_lr": 1e-4,
    "gradient_clip": 1.0,
}
```

### Data Augmentation (Training)

```python
train_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ColorJitter(brightness=0.2, contrast=0.2),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])
```

### Evaluation (No Augmentation)

```python
eval_transforms = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                        std=[0.229, 0.224, 0.225]),
])
```

---

## 9. Codebase Integration

### Implementation

Fine-tuning is implemented as a single module rather than the multi-file structure originally proposed:

| Component | Location |
|-----------|----------|
| All fine-tuning logic | `src/ssl_attention/evaluation/fine_tuning.py` |
| Training orchestration | `FineTuner` class |
| Model wrapping + LoRA | `FineTunableModel` class (uses HuggingFace PEFT) |
| Classification head | `ClassificationHead` class |
| Config & results | `FineTuningConfig` / `FineTuningResult` dataclasses |
| Checkpoint loading | `load_finetuned_model()` function |
| Dataset | Reuses existing `FullDataset` with augmentation transforms |

**Implemented methods:** Linear Probe, LoRA (via HF PEFT), Full Fine-tuning

**Not implemented:** Attention-supervised fine-tuning (Section 3), Core-tuning, COIN, Block expansion (Section 4) — these remain research directions.

### Checkpoint Storage

Checkpoints are saved to `outputs/checkpoints/` with a flat naming convention:

```
outputs/checkpoints/
├── {model_name}_linear_probe_finetuned.pt
├── {model_name}_lora_finetuned.pt
└── {model_name}_full_finetuned.pt

outputs/results/
├── fine_tuning_results.json
└── fine_tuning_manifests/
    └── {model_name}_{strategy}_manifest.json
```

The strategy identifiers used throughout the project are `linear_probe`,
`lora`, and `full`.

**Checkpoint discovery**: `load_finetuned_model()` and the fine-tuned
precompute scripts prefer the strategy-aware filenames above. A legacy
`{model}_finetuned.pt` fallback is still accepted for historical **full**
fine-tuning runs only.

**Fine-tuned cache keys**: attention caches, feature caches, and rendered
heatmaps use `{model}_finetuned_{strategy}` keys such as
`dinov2_finetuned_lora` and `clip_finetuned_full`. The strategy-aware
precompute workflows are:

- `generate_attention_cache.py --finetuned --strategies linear_probe lora full`
- `generate_feature_cache.py --finetuned --strategies linear_probe lora full`
- `generate_heatmap_images.py --finetuned --strategies linear_probe lora full`

---


## 10. Fine-tuning Approaches


### Context: What Fine-Tuning Does in This Project

The SSL models (DINOv2, CLIP, etc.) are pretrained vision backbones that produce **attention maps** — heatmaps showing which parts of an image the model "looks at." Fine-tuning on a classification task is **not the goal** — it's an **intervention** that reshapes the model's internal attention. The actual evaluation is always the same: do the attention maps align better or worse with the 631 expert bounding boxes on the 139 annotated images?

```
1. Frozen model     → extract attention → IoU with expert bboxes → "frozen IoU"
2. Fine-tune on classification task (style, country, etc.)
3. Fine-tuned model → extract attention → IoU with expert bboxes → "fine-tuned IoU"
4. Δ IoU = fine-tuned IoU − frozen IoU
```

Each classification task applies a **different learning pressure** on the model's attention:
- **Style →** Pressure to distinguish Romanesque from Gothic → model should attend to style-diagnostic features (pointed arches, rose windows, buttresses)
- **Country →** Pressure to distinguish Germany from France → model might attend to non-architectural cues (landscape, sky, surroundings) → attention likely drifts *away* from expert features
- **More classes →** Finer distinctions (e.g., Gothic vs. Gothic Revival) → must attend to subtler architectural details

The interesting finding isn't "which task gets the best classification accuracy" — it's **how each task reshapes attention relative to the same expert annotations**.

---

### 1. Country Classification (Geographic Proxy)

Classify churches by **country** instead of architectural style.

- **Top classes:** Germany (3,557), France (1,063), UK (943), Spain (717), Italy (538)
- **Why interesting:** Acts as a **negative control** for the Preserve/Enhance/Destroy taxonomy. Country classification likely pushes models toward background/landscape cues rather than architectural features. If country fine-tuning *destroys* attention alignment (Δ IoU drops) while style fine-tuning enhances it, that's strong evidence attention alignment is task-driven rather than incidental.

**Implementation effort: Low**
- Add a `COUNTRY_MAPPING` in `config.py` analogous to `STYLE_MAPPING`
- Extend `FullDataset` to expose country labels by reading `country.id` from `churches.json`
- Update `_compute_class_weights` in `FineTuner` to accept `num_classes` instead of hardcoding `NUM_STYLES`
- `ClassificationHead` and `FineTunableModel` already accept `num_classes` as a parameter
- Δ IoU analysis script is task-agnostic — **no changes needed**

---

### 2. Expanded Style Taxonomy (More Than 4 Classes)

Expand from **4 styles to 8–10** by including additional architectural styles.

- **Candidates:** Gothic Revival (Q186363: 1,636), Modern Architecture (Q245188: 500), Romanesque Revival (Q744373: 492), Neoclassical (Q54111: 381), Brick Gothic (Q695863: 272)
- **Why interesting:** Finer-grained style distinctions force models to attend to subtler discriminative features. If Δ IoU increases more with 10 classes than 4, it suggests the classification objective's **granularity directly modulates attention specificity**.

**Implementation effort: Very Low**
- Expand `STYLE_MAPPING` and `STYLE_NAMES` in `config.py`
- Everything downstream (`FullDataset`, `FineTunableModel`, `ClassificationHead`) derives from `NUM_STYLES = len(STYLE_MAPPING)` automatically
- **No structural code changes needed**

---

### 3. Multi-Label Style Classification

Treat churches with **multiple styles as multi-label** instead of picking the first style.

- **Scope:** 433 of 9,346 churches have >1 style label
- **Why interesting:** A church labeled both Gothic *and* Romanesque should make the model attend to diagnostic features from **both** styles. Tests whether multi-label pressure produces more spatially distributed attention patterns compared to single-label.

**Implementation effort: Medium-High**
- Replace `CrossEntropyLoss` → `BCEWithLogitsLoss` for multi-label loss
- Labels become binary vectors instead of scalar class indices
- Rework `_stratified_split` (can't stratify on a single label)
- Replace accuracy with **per-label F1 / mAP** as training metrics
- Rework `_compute_class_weights` for multi-label weighting

---

### Post-Fine-Tuning Evaluation

The existing `analyze_delta_iou.py` script is **task-agnostic** — it compares attention maps between frozen and fine-tuned models regardless of the classification objective.

#### Key Metrics

| Metric | What it measures | How to interpret |
|--------|-----------------|------------------|
| **Δ IoU** (fine-tuned − frozen) | Whether attention shifts toward/away from expert bboxes | Positive = Enhance, ≈0 = Preserve, Negative = Destroy |
| **Paired statistical test** | Whether Δ IoU is significant after Holm correction | p < 0.05 after correction = confident claim |
| **Cohen's d** | Practical magnitude of the shift | Small (0.2), Medium (0.5), Large (0.8) |
| **Validation accuracy** | How well the model learned the task | Confirms fine-tuning worked before interpreting Δ IoU |

#### Expected Outcomes

- **Country →** Expect **Δ IoU ≤ 0 (Destroy)**. The model should shift attention to non-architectural cues (landscape, urban context). High val accuracy + negative Δ IoU = strong evidence of shortcut learning.
- **Expanded styles →** Expect **Δ IoU ≥ current 4-class Δ IoU (stronger Enhance)**. More classes = more pressure to find discriminative architectural features. Compare Δ IoU magnitude against the existing 4-class baseline.
- **Multi-label →** Expect **more spatially distributed attention** (attention entropy increases). Δ IoU could increase if the model learns to attend to multiple feature types simultaneously rather than just the dominant one.

#### Core Experimental Design

> Same frozen baseline → different fine-tuning objectives → same Δ IoU evaluation on the 139 annotated images. This isolates the effect of the classification task on attention alignment.

#### Observed Results (Q2 Δ IoU)

The following summarizes results from `analyze_delta_iou.py` over all fine-tunable models and strategies (linear probe, LoRA, full), evaluated on the 139 bbox-annotated images at **percentile 90** with Holm-corrected significance. Output: `outputs/results/q2_delta_iou_analysis.json`.

**Linear probe**

- Δ IoU = 0 for every model (CLIP, DINOv2, DINOv3, MAE, SigLIP, SigLIP2). Attention is unchanged because the backbone is frozen; this is the intended baseline.

**Impact of fine-tuning depends on frozen alignment**

- **Models that gain:** CLIP (frozen IoU ≈ 0.018), SigLIP (≈ 0.036), SigLIP2 (≈ 0.022). Full and LoRA both yield significant, often large positive Δ IoU. For CLIP: full +0.069, LoRA +0.063; for SigLIP/SigLIP2: full slightly better than LoRA; both beat linear probe.
- **Models that change little or worsen:** DINOv2 (frozen ≈ 0.082), DINOv3 (≈ 0.133), MAE (≈ 0.033). DINOv2 full gives **negative** Δ IoU (−0.007, significant); LoRA +0.002 (not significant). DINOv3 and MAE show tiny, non-significant deltas. Conclusion: when frozen alignment is already strong, classification fine-tuning does not improve (and can slightly reduce) attention–expert alignment.

**Strategy ordering when fine-tuning helps**

- For CLIP, SigLIP, SigLIP2: **Full ≥ LoRA > linear probe**. Full and LoRA are often not significantly different (e.g. CLIP); when they are (SigLIP, SigLIP2), full wins. LoRA is always significantly better than linear probe when there is a real gain.

**DINOv2: full fine-tuning can hurt alignment**

- Full fine-tuning is significantly worse than both linear probe and LoRA; IoU retention (finetuned / frozen) is **0.91** (only model below 1.0). Full fine-tuning may encourage task shortcuts and move attention away from expert-aligned regions; LoRA is safer for preserving alignment.

**IoU retention (finetuned / frozen) at p90 — full fine-tuning**

- DINOv2: 0.91 (only &lt; 1); DINOv3: 1.02; MAE: 1.11; CLIP: 4.81; SigLIP: 1.98; SigLIP2: 2.65. Large retention gains occur where frozen IoU was low; DINOv2 shows a small drop.

**Summary table (percentile 90)**

| Model    | Strategy     | Frozen IoU | Finetuned IoU | Δ IoU   | Significant?        |
|----------|--------------|------------|---------------|---------|---------------------|
| CLIP     | linear_probe | 0.018      | 0.018         | 0.000   | —                   |
| CLIP     | lora         | 0.018      | 0.082         | +0.063  | Yes                 |
| CLIP     | full         | 0.018      | 0.087         | +0.069  | Yes                 |
| DINOv2   | full         | 0.082      | 0.074         | −0.007  | Yes (worse)         |
| DINOv3   | full / lora  | 0.133      | ~0.134        | ~+0.002 | No                  |
| SigLIP   | full         | 0.036      | 0.072         | +0.036  | Yes                 |
| SigLIP2  | full         | 0.022      | 0.058         | +0.036  | Yes                 |
| MAE      | all          | 0.033      | ~0.036        | ~+0.004 | No                  |

**Takeaways**

- Fine-tuning (LoRA or full) **improves** attention–expert IoU for CLIP, SigLIP, and SigLIP2; it **does not** improve it for DINOv2/DINOv3 and can slightly **worsen** it for DINOv2.
- When improvement occurs, **full and LoRA both help**; full is sometimes better than LoRA; **linear probe never changes alignment**.
- Pre-training family matters: contrastive models (CLIP, SigLIP) start with weaker alignment and benefit from task adaptation; DINO-style models already align well and gain little or lose under classification fine-tuning.
- For DINOv2, **full fine-tuning is the only case where alignment significantly decreases**; LoRA is the safer option if the goal is to preserve or improve expert alignment.

### UI Support: Feature-Local Fine-Tuning Comparison

The shipped UI now exposes two fine-tuning comparison surfaces:

- **Frozen vs Fine-tuned** compares a frozen backbone against one selected strategy, or an auto-discovered legacy fine-tuned variant when strategy is omitted
- **Fine-tuning Method vs Method** compares two strategy-specific variants for the same base model

Both modes support a **feature-focused inspection flow**:

- By default, the user sees the cached attention overlay pair for the full image
- Clicking an expert bounding box switches the view to **bbox-conditioned similarity heatmaps** for both selected variants, matching the gallery-style interaction
- The UI also computes **bbox-local IoU and coverage** for both variants inside the selected expert region
- The page then shows a **feature-local delta** between the two active variants

This matters because the user can move between two complementary views:

- **Global overlay mode:** Where does each selected variant attend overall?
- **Feature-conditioned similarity mode:** What other regions become similar to this selected expert feature?

If the full-image overlays look visually similar, users can still ask a sharper question:

> Did fine-tuning increase attention on this specific architectural cue?

That teaches us something more precise than a global slider:

- **Positive local Δ IoU**: the right-hand variant strengthened alignment on that expert feature
- **Near-zero local Δ IoU**: the compared variants behave similarly on that feature
- **Negative local Δ IoU**: the right-hand variant pulled attention away from that feature

In other words, bbox selection now changes both the **visualization mode** and
the **measurement slice**. This is aligned with the core research framing in
this section: classification is only an intervention, while the real object of
study is how attention shifts relative to expert-defined architectural regions.

## 11. Alternative Perspective: Why NOT to Fine-Tune

Before committing to fine-tuning, consider this counter-argument:

### The "Frozen Features Are Enough" Position

- DINOv3 specifically demonstrates that frozen features + linear probe achieves near-SOTA on many tasks
- Fine-tuning may just teach the model to exploit dataset biases, not learn meaningful features
- If frozen models already align well with expert annotations, fine-tuning may be unnecessary

### Counter-Argument (Why Fine-Tuning Is Still Valuable)

- Even if classification accuracy is similar, attention *patterns* may differ dramatically
- The Δ IoU metric answers a fundamental question about what SSL models learn
- **Negative results are equally informative:** If fine-tuning *doesn't* improve alignment, that tells us the pre-trained features are already task-agnostic representations of architectural semantics

---

## 12. References

### Parameter-Efficient Fine-Tuning
- [Parameter-Efficient Fine-Tuning of DINOv2 for Lung Nodule Classification (IEEE 2024)](https://ieeexplore.ieee.org/document/10635887/)
- [PEFT Comprehensive Survey (arXiv 2024)](https://arxiv.org/pdf/2403.14608)
- [Dynamic Tuning for ViT Adaptation (NeurIPS 2024)](https://proceedings.neurips.cc/paper_files/paper/2024/file/d0241a0fb1fc9be477bdfde5e0da276a-Paper-Conference.pdf)
- [PEFT Beyond LoRA: Advanced Techniques](https://mbrenndoerfer.com/writing/peft-beyond-lora-advanced-parameter-efficient-finetuning-techniques)

### Attention & Interpretability
- [HuMAL: Aligning Human and Machine Attention](https://arxiv.org/html/2502.06811v1)
- [GMAR: Gradient-Driven Multi-Head Attention Rollout](https://arxiv.org/html/2504.19414v1)
- [Mechanistic Interpretability of Fine-Tuned ViTs](https://arxiv.org/abs/2503.18762)
- [Interpretability-Aware Vision Transformer](https://arxiv.org/html/2309.08035v2)
- [Do Vision Transformers See Like Humans?](https://arxiv.org/html/2508.09850)

### DINOv2/DINOv3 Resources
- [Understanding DINOv2: Engineer's Deep Dive](https://www.lightly.ai/blog/dinov2)
- [DINOv3 Technical Deep Dive](https://www.lightly.ai/blog/dinov3)
- [DINOv3 Paper (arXiv 2025)](https://arxiv.org/html/2508.10104v1)
- [Catastrophic Forgetting in SSL ViTs](https://arxiv.org/html/2404.17245v1)
- [Official DINOv2 Repository](https://github.com/facebookresearch/dinov2)

### Contrastive Fine-Tuning
- [Core-tuning: Contrast-Regularized Fine-Tuning (NeurIPS 2021)](https://github.com/Vanint/Core-tuning)
- [Contrastive Initialization (COIN)](https://arxiv.org/abs/2208.00238)
- [Contrastive vs Reconstructive Self-Supervised Learning](https://sslneurips22.github.io/paper_pdfs/paper_50.pdf)

### Weakly Supervised & Attention Guidance
- [Localization-Guided Medical Vision Transformer](https://link.springer.com/chapter/10.1007/978-3-031-92648-8_8)
- [Weakly Supervised Target Detection with Spatial Attention](https://link.springer.com/article/10.1007/s44267-024-00037-y)
- [Attention Regularization Techniques (MIT Deep Learning 2023)](https://deep-learning-mit.github.io/staging/blog/2023/attention-regularization/)
- [AttentionDrop: Novel Regularization for Transformers](https://arxiv.org/pdf/2504.12088)

### General Resources
- [Awesome Transformer Attention (GitHub)](https://github.com/cmhungsteve/Awesome-Transformer-Attention)
- [HuggingFace PEFT Documentation](https://huggingface.co/blog/samuellimabraz/peft-methods)
- [Kili Technology: DINOv2 Fine-Tuning Tutorial](https://kili-technology.com/data-labeling/computer-vision/dinov2-fine-tuning-tutorial-maximizing-accuracy-for-computer-vision-tasks)

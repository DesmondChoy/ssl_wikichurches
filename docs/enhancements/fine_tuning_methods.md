# Fine-Tuning Methods for Studying Attention Shift in SSL Vision Models

> **Research compiled:** February 2026
> **Purpose:** Evaluate fine-tuning strategies to study whether task-specific training redirects SSL model attention toward expert-annotated architectural features

## Executive Summary

This document outlines fine-tuning methods suitable for the SSL WikiChurches project's Research Question 3: *Does fine-tuning on architectural style classification shift model attention toward expert-annotated diagnostic features?*

**Key insight:** The Δ IoU metric (post-fine-tuning IoU − pre-fine-tuning IoU) directly measures whether fine-tuning improves alignment between model attention and human expert annotations.

---

## 1. Why Fine-Tuning Makes Sense

### The Natural Experiment

1. **Pre-trained models learned general features** — DINOv2, CLIP, MAE, etc. were trained on diverse datasets (ImageNet, LAION) without knowledge of architectural styles
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
| Full WikiChurches | 9,485 images | Various style labels | Semi-supervised expansion |

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

### New Files to Create

```
src/ssl_attention/
├── finetune/
│   ├── __init__.py
│   ├── trainer.py          # Training loop with logging
│   ├── lora_adapter.py     # LoRA wrapper for models
│   ├── losses.py           # Classification + attention losses
│   └── schedulers.py       # Learning rate schedulers
├── data/
│   └── style_dataset.py    # Dataset for style classification
```

### Modifications to Existing Code

1. **`VisionBackbone` protocol** — Add `.train()` and `.eval()` mode support
2. **`get_model()` factory** — Support loading fine-tuned checkpoints
3. **Metrics pipeline** — Add Δ IoU computation

### Checkpoint Storage

```
checkpoints/
├── linear_probe/
│   ├── dinov2_linear.pt
│   ├── clip_linear.pt
│   └── ...
├── lora/
│   ├── dinov2_lora_r8.pt
│   └── ...
└── full_ft/
    ├── dinov2_full.pt
    └── ...
```

---

## 10. Alternative Perspective: Why NOT to Fine-Tune

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

## 11. References

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

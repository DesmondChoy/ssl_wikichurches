# Fine-Tuning Methods for SSL Attention Alignment

**Research Date:** February 2026
**Context:** WikiChurches SSL Attention Visualization Platform

---

## Executive Summary

This document evaluates fine-tuning approaches for improving alignment between SSL model attention and expert-annotated architectural features. Before diving into methods, we must address a fundamental question: **Is fine-tuning the right approach?**

**Key Finding:** The answer depends on your objective:
- **If studying intrinsic SSL attention properties** → Don't fine-tune (defeats the purpose)
- **If building a practical architectural feature detector** → Fine-tuning is appropriate
- **If improving attention interpretability** → Consider hybrid approaches

---

## 1. Why Fine-Tune? Understanding the Motivation

### Current State Analysis

Your platform compares 6 SSL models against 631 expert-annotated bounding boxes across 139 church images. The core research question appears to be: *"Do SSL models learn to attend to what architectural experts consider important?"*

### Three Distinct Objectives

| Objective | Fine-Tuning Appropriate? | Rationale |
|-----------|-------------------------|-----------|
| **A. Measure intrinsic attention alignment** | ❌ No | Fine-tuning would confound the measurement—you'd no longer be studying what SSL models learned, but what they can learn |
| **B. Build better architectural feature detector** | ✅ Yes | Leveraging SSL representations + expert annotations creates a practical tool |
| **C. Study how attention shifts with domain adaptation** | ✅ Yes | Measuring before/after attention patterns is scientifically valuable |

### Recommendation

If your goal is **Objective A** (which seems most aligned with comparing models to expert annotations), fine-tuning may not be appropriate. Instead, consider:
- Analyzing existing attention patterns more deeply
- Comparing across more models/methods
- Developing better post-hoc attention interpretation techniques

If pursuing **Objectives B or C**, the methods below are relevant.

---

## 2. Fine-Tuning Methods Suitable for This Context

### 2.1 Parameter-Efficient Fine-Tuning (PEFT) with LoRA

**What it is:** Low-Rank Adaptation freezes the pretrained backbone and injects small trainable matrices into attention layers.

**Why suitable for WikiChurches:**
- Small dataset (139 annotated images) makes full fine-tuning prone to overfitting
- LoRA trains only 0.2-0.9% of parameters, dramatically reducing overfitting risk
- Can leverage full 9,502-image dataset for style classification task
- Memory efficient: runs on consumer GPUs (14-16GB VRAM vs 80GB+ for full fine-tuning)

**How to implement:**
```python
from peft import LoraConfig, get_peft_model

lora_config = LoraConfig(
    r=16,  # Rank of adaptation matrices
    lora_alpha=32,
    target_modules=["qkv"],  # Adapt attention projections
    lora_dropout=0.1,
)
model = get_peft_model(backbone, lora_config)
```

**Trade-offs:**
- ✅ Prevents catastrophic forgetting of SSL-learned features
- ✅ Enables merging weights back for zero-latency inference
- ⚠️ May not significantly change attention patterns (adapts representations, not necessarily attention)

**References:**
- [LoRA for Vision Transformers Benchmark (2025)](https://www.preprints.org/manuscript/202510.2514)
- [DINOv3 LoRA Fine-tuning (GitHub)](https://github.com/RobvanGastel/dinov3-finetune)
- [PEFT Survey and Benchmark (NeurIPS 2024)](https://arxiv.org/html/2402.02242v5)

---

### 2.2 Linear Probing then Fine-Tuning (LP-FT)

**What it is:** Two-stage approach: (1) train only a classification head, (2) then fine-tune the full model with a small learning rate.

**Why suitable for WikiChurches:**
- Prevents distortion of pretrained features that causes OOD degradation
- Research shows LP-FT achieves **+10% better OOD accuracy** than full fine-tuning alone
- Particularly valuable when pretrained features are high-quality (DINOv2/v3 features are excellent)

**How to implement:**
```python
# Stage 1: Linear probe (freeze backbone)
for param in backbone.parameters():
    param.requires_grad = False
train_classifier(epochs=10, lr=1e-3)

# Stage 2: Full fine-tuning (unfreeze)
for param in backbone.parameters():
    param.requires_grad = True
train_full(epochs=5, lr=1e-5)  # Much smaller LR
```

**Trade-offs:**
- ✅ Best of both worlds: ID and OOD performance
- ✅ More stable training dynamics
- ⚠️ Two-stage process requires more careful hyperparameter tuning

**References:**
- [Fine-Tuning can Distort Pretrained Features (ICLR)](https://openreview.net/forum?id=UYneFzXSJWh)
- [DINOv2 Linear Probing Performance](https://arxiv.org/html/2304.07193v2)

---

### 2.3 Attention Supervision Loss (Direct Attention Alignment)

**What it is:** Add an auxiliary loss that directly supervises attention maps to align with bounding box annotations.

**Why suitable for WikiChurches:**
- Directly addresses your core question: attention-to-annotation alignment
- Can use your 631 bounding boxes as ground truth attention targets
- Recent work (TAB, 2024) shows this improves both attention interpretability and task performance

**How to implement:**
```python
def attention_supervision_loss(attention_map, bboxes, image_size):
    """
    Supervise attention to focus on annotated regions.
    """
    # Create target attention from bounding boxes
    target = create_target_attention(bboxes, image_size)

    # Use BCE or MSE loss
    return F.binary_cross_entropy(attention_map, target)

# Combined loss
total_loss = classification_loss + lambda_attn * attention_supervision_loss
```

**Design considerations:**
- **Target creation:** Assign 1.0 to patches overlapping bboxes, 0.0 elsewhere (or soft targets based on IoU)
- **Which attention to supervise:** Last layer CLS attention is most semantic
- **Loss weighting (λ):** Start with 0.1-0.5, tune empirically

**Trade-offs:**
- ✅ Directly optimizes for your evaluation metric
- ✅ Creates interpretable models that attend where humans expect
- ⚠️ May hurt classification performance if attention targets conflict with optimal class discrimination
- ⚠️ Fundamentally changes what the model learns (no longer "self-supervised")

**References:**
- [TAB: Transformer Attention Bottlenecks (2024)](https://arxiv.org/html/2412.18675)
- [Box2Seg: Attention Weighted Loss](https://www.robots.ox.ac.uk/~tvg/publications/2020/box2seg.pdf)

---

### 2.4 Adapter-Based Fine-Tuning

**What it is:** Insert small bottleneck layers between frozen transformer blocks.

**Why suitable for WikiChurches:**
- Even more parameter-efficient than LoRA
- Can adapt attention flow without modifying pretrained weights
- Well-suited for domain adaptation (ImageNet → architectural images)

**How to implement:**
```python
class Adapter(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.down = nn.Linear(dim, dim // reduction)
        self.up = nn.Linear(dim // reduction, dim)
        self.act = nn.GELU()

    def forward(self, x):
        return x + self.up(self.act(self.down(x)))
```

**References:**
- [ViT-Adapter for Segmentation](https://github.com/facebookresearch/dinov2/issues/276)
- [PEFT Methods Explained (Medium)](https://medium.com/@khayyam.h/the-evolution-of-fine-tuning-lora-adapters-and-other-peft-methods-explained-9ed0ac62937a)

---

### 2.5 Depth-Wise Convolution Shortcut

**What it is:** Add lightweight depth-wise convolution modules that bypass transformer blocks, capturing local information that ViTs may miss.

**Why suitable for WikiChurches:**
- Architectural features often have strong local structure (edges, corners, textures)
- Addresses ViT's weakness: "captures global context but overlooks neighboring pixel relationships"
- Minimal parameter overhead (~0.5% additional parameters)

**Trade-offs:**
- ✅ Adds inductive bias beneficial for small datasets
- ✅ Complements global attention with local features
- ⚠️ Architectural modification, not just fine-tuning

**References:**
- [Depth-Wise Convolutions in ViT (ScienceDirect 2024)](https://www.sciencedirect.com/science/article/pii/S0925231224017697)

---

## 3. Alternative Approaches (No Fine-Tuning Required)

If your goal is understanding intrinsic SSL attention rather than building a better detector, consider these alternatives:

### 3.1 Post-Hoc Attention Analysis

**Approach:** Deeper analysis of existing attention patterns without modification.

- **Per-feature-type analysis:** Which architectural features (columns, arches, windows) get highest attention?
- **Attention head specialization:** Do different heads specialize in different feature types?
- **Layer-wise evolution:** How does attention to annotations change across layers?

### 3.2 Weakly-Supervised Attention Enhancement

**Approach:** Use SSL attention + your bounding boxes for pseudo-label generation without retraining.

```python
# DINO attention as seed + bbox propagation
attention_seeds = extract_cls_attention(dino_model, image)
enhanced_masks = propagate_with_bbox_guidance(attention_seeds, bboxes)
```

**References:**
- [Pseudo-Label Enhancement with DINO (2025)](https://www.sciencedirect.com/science/article/abs/pii/S0950705125000607)
- [Upsampling DINOv2 for Localization](https://arxiv.org/html/2410.19836v1)

### 3.3 Attention Steering (Post-Hoc)

**Approach:** Modify attention at inference time without retraining.

- Amplify attention to annotated regions
- Suppress attention to background
- Compare "steered" vs "natural" attention

---

## 4. Recommended Approach for WikiChurches

### If Your Goal is Scientific Understanding (Objective A)

**Don't fine-tune.** Instead:

1. Extend current analysis with per-feature-type breakdowns
2. Analyze attention head specialization across models
3. Compare attention evolution across layers (you have rollout, use it!)
4. Add more models or attention extraction methods for comparison

### If Your Goal is a Better Architectural Feature Detector (Objective B)

**Recommended pipeline:**

1. **Start with LP-FT on style classification** (4 classes, balanced)
   - Uses your full 9,502-image dataset
   - Establishes domain-adapted representations

2. **Add LoRA for efficient adaptation**
   - Train on annotated subset
   - Use style-classification-adapted backbone as starting point

3. **Optional: Add attention supervision loss**
   - Only if attention alignment is the primary goal
   - Careful with loss weighting

### If Your Goal is Studying Attention Shift (Objective C)

**Recommended experiment:**

1. Measure attention alignment metrics before fine-tuning
2. Fine-tune with LP-FT or LoRA
3. Measure attention alignment metrics after fine-tuning
4. Analyze: What features gained/lost attention? Which models changed most?

---

## 5. Implementation Considerations

### Dataset Size Concerns

| Dataset | Size | Suitable Methods |
|---------|------|------------------|
| Annotated subset | 139 images, 631 boxes | LoRA, LP-FT, attention supervision |
| Full dataset | 9,502 images | Full fine-tuning possible, but PEFT still recommended |

With 139 images, **full fine-tuning will almost certainly overfit**. PEFT methods are strongly recommended.

### Computational Requirements

| Method | Parameters Trained | GPU Memory | Training Time |
|--------|-------------------|------------|---------------|
| Full fine-tuning | ~85M (100%) | ~24GB | Hours |
| LoRA (r=16) | ~0.5M (0.6%) | ~8GB | Minutes |
| LP-FT | ~85M staged | ~16GB | ~1 hour |
| Linear probe only | ~3K (0.004%) | ~6GB | Minutes |

### Existing Code Leverage

Your codebase already has:
- `evaluation/fine_tuning.py`: FineTunableModel, FineTuner classes
- `experiments/scripts/fine_tune_models.py`: CLI for training
- Support for freeze_backbone (linear probe mode)
- Differential learning rates for backbone/head

**To add LoRA:** Integrate HuggingFace PEFT library with existing FineTunableModel.

**To add attention supervision:** Modify training loop to extract attention, compute supervision loss.

---

## 6. Summary Table

| Method | Overfitting Risk | Changes Attention? | Preserves SSL Features? | Implementation Effort |
|--------|------------------|-------------------|------------------------|----------------------|
| Full fine-tuning | 🔴 High | Yes | 🔴 No | Low |
| LoRA | 🟢 Low | Moderate | 🟢 Yes | Medium |
| LP-FT | 🟡 Medium | Yes | 🟡 Partially | Low |
| Attention supervision | 🟡 Medium | 🟢 Directly | 🟡 Partially | High |
| Adapters | 🟢 Low | Moderate | 🟢 Yes | Medium |
| No fine-tuning (post-hoc) | N/A | N/A | 🟢 Yes | Low |

---

## 7. Conclusion

**The most important question is not "how to fine-tune" but "why fine-tune."**

If comparing SSL models to human annotations is the research goal, fine-tuning defeats the purpose—you'd be measuring what models *can* learn with supervision, not what they learned self-supervised.

If building a practical tool or studying adaptation dynamics is the goal, **LP-FT combined with LoRA** offers the best balance of:
- Preventing overfitting on small data
- Preserving valuable SSL representations
- Enabling efficient experimentation

**Attention supervision loss** is the most direct approach if attention alignment itself is the optimization target, but it fundamentally changes the nature of the model from "self-supervised" to "weakly-supervised."

---

## References

### Parameter-Efficient Fine-Tuning
- [Parameter-Efficient Fine-Tuning Survey (NeurIPS 2024)](https://arxiv.org/html/2402.02242v5)
- [LoRA Benchmark for Vision Transformers](https://www.preprints.org/manuscript/202510.2514)
- [PEFT Methods Explained](https://medium.com/@khayyam.h/the-evolution-of-fine-tuning-lora-adapters-and-other-peft-methods-explained-9ed0ac62937a)
- [DINOv3 LoRA Fine-tuning](https://github.com/RobvanGastel/dinov3-finetune)

### Linear Probing vs Fine-Tuning
- [Fine-Tuning Distorts Pretrained Features](https://openreview.net/forum?id=UYneFzXSJWh)
- [Benchmarking SSL Pre-training](https://link.springer.com/article/10.1007/s11263-025-02402-w)
- [DINOv2 Paper](https://arxiv.org/html/2304.07193v2)
- [DINOv3 Paper](https://arxiv.org/html/2508.10104v1)

### Attention Supervision
- [TAB: Transformer Attention Bottlenecks](https://arxiv.org/html/2412.18675)
- [Box2Seg: Attention Weighted Loss](https://www.robots.ox.ac.uk/~tvg/publications/2020/box2seg.pdf)

### Weakly Supervised Localization
- [Pseudo-Label Enhancement with DINO](https://www.sciencedirect.com/science/article/abs/pii/S0950705125000607)
- [Upsampling DINOv2 Features](https://arxiv.org/html/2410.19836v1)
- [Unsupervised Object Localization Survey](https://arxiv.org/html/2310.12904v3)

### Small Dataset Training
- [Depth-Wise Convolutions in ViT](https://www.sciencedirect.com/science/article/pii/S0925231224017697)
- [Graph-based Vision Transformer](https://www.nature.com/articles/s41598-025-10408-0)
- [Fine-Tuning ViT with Custom Dataset (Hugging Face)](https://huggingface.co/learn/cookbook/en/fine_tuning_vit_custom_dataset)

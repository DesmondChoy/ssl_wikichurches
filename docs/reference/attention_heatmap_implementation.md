# Attention Heatmap Implementation

This document explains what attention heatmaps are, why they matter for this project, and how they are implemented in this codebase. It also discusses model-specific considerations and potential limitations.

> **Related documents:**
> - [Attention Methods](./attention_methods.md) — Detailed explanation of CLS attention, rollout, mean attention, and Grad-CAM
> - [Project Proposal](../core/project_proposal.md) — Full research design and hypotheses

---

## Table of Contents

1. [What Are Attention Heatmaps?](#what-are-attention-heatmaps)
2. [Why Attention Heatmaps Matter](#why-attention-heatmaps-matter)
3. [Research Questions Addressed](#research-questions-addressed)
4. [Implementation Architecture](#implementation-architecture)
   - [Two-Phase Design](#two-phase-design)
   - [What Happens When You Click a Bounding Box](#what-happens-when-you-click-a-bounding-box)
5. [Model-Specific Appropriateness](#model-specific-appropriateness)
   - [DINO Models (DINOv2, DINOv3)](#dino-models-dinov2-dinov3)
   - [CLIP](#clip)
   - [MAE](#mae)
   - [SigLIP](#siglip)
   - [ResNet-50](#resnet-50)
6. [Summary Table](#summary-table)
7. [References](#references)

---

## What Are Attention Heatmaps?

Attention heatmaps are visualizations that show **where a Vision Transformer (ViT) "looks"** when processing an image. They represent the attention weights from the model's self-attention mechanism as a color-coded overlay on the original image.

In Vision Transformers:
1. An image is divided into patches (e.g., 14×14 or 16×16 pixels each)
2. Each patch becomes a token in a sequence
3. The **attention mechanism** determines how much each token should "attend to" other tokens
4. The **CLS token** (when present) aggregates information from all patches to form a global representation

An attention heatmap visualizes these weights spatially:
- **Red/Yellow** = High attention (the model considers these regions important)
- **Blue** = Low attention (the model ignores these regions)

---

## Why Attention Heatmaps Matter

### The Trust Problem

A model achieving 95% accuracy on architectural style classification could be:
- **Genuinely understanding** — attending to flying buttresses, pointed arches, and rose windows
- **Exploiting shortcuts** — using background correlations, image metadata, or dataset biases

These two scenarios are indistinguishable from accuracy metrics alone. Attention heatmaps provide a window into what the model considers important.

### From Visualization to Quantification

While tools like [BertViz](https://github.com/jessevig/bertviz) and [Comet ML](https://www.comet.ml/) visualize attention, they don't quantify whether attention aligns with domain expertise. Our system goes further by:

1. Overlaying attention on expert-annotated bounding boxes
2. Computing **IoU (Intersection over Union)** between attention regions and expert annotations
3. Enabling systematic comparison across models, layers, and methods

This transforms subjective "the model seems to look at the right things" into quantitative "Model A achieves 0.42 IoU with expert annotations at layer 11."

---

## Research Questions Addressed

Attention heatmaps directly support **Research Question 1** and contribute to all four research questions from the [Project Proposal](../core/project_proposal.md#research-questions-and-approaches):

| RQ | Question | How Attention Heatmaps Help |
|----|----------|----------------------------|
| **Q1** | Do SSL models attend to the same features human experts consider diagnostic? | Overlay heatmaps on expert bounding boxes; compute IoU to quantify alignment |
| **Q2** | Does fine-tuning shift attention toward expert-identified features? | Compare heatmaps before/after fine-tuning; measure Δ IoU |
| **Q3** | Do individual attention heads specialize for different architectural features? | Generate per-head heatmaps; analyze which heads best align with specific feature types |
| **Q4** | Does the fine-tuning strategy affect how much attention shifts? | Compare heatmap changes across Linear Probe, LoRA, and Full fine-tuning |

**Q1 is the primary use case:** The heatmap overlay with bounding boxes is the core visual interface for understanding whether models have learned expert-relevant attention patterns.

---

## Implementation Architecture

### Two-Phase Design

The system uses a **precompute-then-serve** architecture for performance:

```
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 1: OFFLINE PRECOMPUTATION (one-time, runs for hours)      │
│                                                                  │
│   Scripts: app/precompute/generate_attention_cache.py            │
│            app/precompute/generate_feature_cache.py              │
│                                                                  │
│   For each (image × model × layer × method):                    │
│     1. Load model, run forward pass with torch.no_grad()         │
│     2. Extract attention weights from transformer layers         │
│     3. Apply extraction method (CLS, rollout, mean, Grad-CAM)   │
│     4. Save to HDF5 cache (attention_viz.h5, features.h5)       │
│                                                                  │
│   Result: ~50GB of pre-computed attention maps and features      │
└──────────────────────────────────────────────────────────────────┘
                              ↓
┌──────────────────────────────────────────────────────────────────┐
│ PHASE 2: RUNTIME (milliseconds per request)                      │
│                                                                  │
│   User interaction → API request → HDF5 cache lookup → Response │
│                                                                  │
│   NO MODEL INFERENCE AT RUNTIME                                  │
│   All attention data is pre-computed and served from cache       │
└──────────────────────────────────────────────────────────────────┘
```

**Why this design?**
- Model inference is slow (100-500ms per image on GPU)
- Users expect instant feedback when adjusting layers/thresholds
- Pre-computation enables sub-10ms response times

### What Happens When You Click a Bounding Box

When a user clicks on an expert-annotated bounding box, the system computes a **similarity heatmap** showing which image patches are semantically similar to the selected region:

```
┌─────────────────────────────────────────────────────────────────┐
│ STEP 1: User clicks bounding box in React frontend              │
│                                                                 │
│   Component: app/frontend/src/components/attention/             │
│              AttentionViewer.tsx (lines 130-137)                │
│                                                                 │
│   Action: onBboxSelect(selectedBboxIndex)                       │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 2: Frontend sends API request                              │
│                                                                 │
│   POST /api/attention/{image_id}/similarity                     │
│   Body: { left: 0.2, top: 0.3, width: 0.2, height: 0.2 }       │
│   Query: ?model=dinov2&layer=11                                 │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 3: Backend loads PRE-CACHED patch features                 │
│                                                                 │
│   Service: app/backend/services/similarity_service.py           │
│   Source: features.h5 (HDF5 cache)                              │
│                                                                 │
│   Data: (256, 768) tensor for DINOv2                           │
│         = 16×16 patches × 768-dim embeddings                   │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 4: Compute cosine similarity (fast, ~1ms)                  │
│                                                                 │
│   1. Map bbox coordinates → patch indices                       │
│      e.g., bbox covering 4 patches → indices [66, 67, 82, 83]  │
│                                                                 │
│   2. Extract features for those patches                         │
│   3. Compute mean feature vector (query): (1, 768)             │
│   4. L2 normalize query and all patch features                 │
│   5. Cosine similarity: query @ patches.T = (1, 256)           │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 5: Return similarity response                              │
│                                                                 │
│   {                                                             │
│     "similarity": [-0.15, 0.42, ..., 0.08],  // 256 values     │
│     "patch_grid": [16, 16],                                    │
│     "min_similarity": -0.51,                                   │
│     "max_similarity": 0.87,                                    │
│     "bbox_patch_indices": [66, 67, 82, 83]                     │
│   }                                                             │
└─────────────────────────────────────────────────────────────────┘
                              ↓
┌─────────────────────────────────────────────────────────────────┐
│ STEP 6: Frontend renders similarity heatmap                     │
│                                                                 │
│   File: app/frontend/src/utils/renderHeatmap.ts                 │
│                                                                 │
│   1. Create 16×16 canvas (one pixel per patch)                 │
│   2. Map similarity values to Turbo colormap                    │
│   3. Scale up to 224×224 with bilinear interpolation           │
│   4. Overlay on original image with opacity                     │
└─────────────────────────────────────────────────────────────────┘
```

**Key insight:** The similarity computation uses **pre-cached patch features**, not live model inference. The only runtime computation is the cosine similarity calculation, which takes ~1ms.

### Two Types of Heatmaps

| Heatmap Type | Trigger | Data Source | Shows |
|--------------|---------|-------------|-------|
| **Attention Heatmap** | Always visible | `attention_viz.h5` (pre-cached) | Where CLS token attends to patches |
| **Similarity Heatmap** | Click on bounding box | `features.h5` (pre-cached) + runtime cosine similarity | Which patches are semantically similar to selected region |

---

## Model-Specific Appropriateness

Not all attention extraction methods are equally meaningful for all models. The appropriateness depends on how each model was trained and what its attention mechanism represents.

### DINO Models (DINOv2, DINOv3)

| Aspect | Assessment |
|--------|------------|
| **Method** | CLS attention + Rollout |
| **Appropriateness** | **Excellent** |

DINO models are the gold standard for attention visualization. [Caron et al. (2021)](https://arxiv.org/abs/2104.14294) made a remarkable discovery:

> "Self-supervised ViT features contain explicit information about the semantic segmentation of an image, which **does not emerge as clearly with supervised ViTs, nor with convnets**."

The CLS token in DINO is trained via self-distillation to aggregate semantically meaningful information. Its attention weights naturally produce coherent, interpretable maps that resemble segmentation masks—without any segmentation supervision.

**Register tokens:** DINOv2 includes 4 register tokens ([Darcet et al., 2024](https://arxiv.org/abs/2309.16588)) that act as "scratch space" for computation. Our implementation correctly excludes these from attention extraction:

```python
# Sequence: [CLS, reg0, reg1, reg2, reg3, patch0, ..., patch255]
patch_start = 1 + num_registers  # = 5 for DINOv2
cls_to_patches = cls_to_all[:, patch_start:]
```

### CLIP

| Aspect | Assessment |
|--------|------------|
| **Method** | CLS attention + Rollout |
| **Appropriateness** | **Good** (with caveats) |

CLIP's CLS token aggregates visual information for matching with text embeddings. Its attention patterns are meaningful but reflect **text-image alignment** rather than pure visual semantics.

[Walmer et al. (2023)](https://openaccess.thecvf.com/content/CVPR2023/papers/Walmer_Teaching_Matters_Investigating_the_Role_of_Supervision_in_Vision_Transformers_CVPR_2023_paper.pdf) showed that CLIP attention differs from DINO:
- CLIP may attend more to "nameable" objects and features
- Attention patterns are influenced by the text concepts seen during training

**Implication:** CLIP attention alignment with expert annotations tells us about alignment with language-grounded visual concepts, not necessarily the same "visual primitives" that DINO learns.

### MAE

| Aspect | Assessment |
|--------|------------|
| **Method** | CLS attention + Rollout |
| **Appropriateness** | **Questionable** |

MAE (Masked Autoencoder) presents a significant interpretability challenge. While it has a CLS token, the token serves a fundamentally different purpose than in DINO or CLIP.

**The problem:** MAE was trained with **75% masking** to reconstruct missing patches. The CLS token learns to aggregate information for reconstruction, not for semantic aggregation. From [He et al. (2022)](https://openaccess.thecvf.com/content/CVPR2022/papers/He_Masked_Autoencoders_Are_Scalable_Vision_Learners_CVPR_2022_paper.pdf):

> "Unlike contrastive methods, there is no explicit loss function acting on the [CLS] token."

**Research evidence:** [CrossMAE (Fu et al., 2024)](https://crossmae.github.io/) investigated MAE's attention patterns and found they are fundamentally different from discriminative models:
- MAE attention reflects reconstruction objectives, not semantic saliency
- The CLS token doesn't develop the same "aggregation" behavior as DINO

**Our implementation:** We run MAE with `mask_ratio=0.0` (all patches visible), so the mechanics are correct. However, users should understand that MAE's CLS attention may show reconstruction-focused patterns rather than semantic saliency.

**Verdict:** The implementation is technically correct, but the **interpretation may be misleading**. MAE attention should not be directly compared to DINO attention as if they measure the same thing.

### SigLIP

| Aspect | Assessment |
|--------|------------|
| **Method** | Mean attention |
| **Appropriateness** | **Acceptable** (approximation) |

SigLIP uses **Multi-head Attention Pooling (MAP)** instead of a CLS token. This architectural difference requires a different attention extraction approach.

**The challenge:** According to [SigLIP 2 documentation](https://arxiv.org/html/2502.14786v1) and [GitHub discussions](https://github.com/google-research/big_vision/issues/145):
- The MAP head has learned query vectors that aggregate patch information
- There is no single CLS token whose attention we can extract
- The aggregation happens through learned queries, not a fixed token

**Our approach:** We use `extract_mean_attention()`, which computes how much each patch is attended to by all other patches:

```python
# Mean across rows: how much does each position get attended to?
mean_received = attn_fused.mean(dim=1)  # (B, seq)
```

**Limitations:**
- This is an approximation, not the actual attention used by the MAP head
- Better approaches could include extracting the MAP head's query-key attention weights (if accessible)
- Gradient-based methods might better reflect what drives SigLIP's predictions

**Verdict:** The implementation handles the lack of CLS correctly, but mean attention is a **reasonable approximation**, not the ideal solution.

### ResNet-50

| Aspect | Assessment |
|--------|------------|
| **Method** | Grad-CAM |
| **Appropriateness** | **Correct** |

ResNet-50 is a CNN with no attention mechanism. We use Grad-CAM ([Selvaraju et al., 2017](https://arxiv.org/abs/1610.02391)) to generate attention-like heatmaps via gradients.

**Implementation:** See `src/ssl_attention/models/resnet50.py`:

```python
def _compute_gradcam_heatmap(self, layer_name: str, image_size: int) -> Tensor:
    activations = self._activations[layer_name]  # (B, C, H, W)
    gradients = self._gradients[layer_name]      # (B, C, H, W)

    # Global average pooling of gradients -> channel weights
    weights = gradients.mean(dim=(2, 3), keepdim=True)

    # Weighted combination of activation maps
    cam = (weights * activations).sum(dim=1)
    cam = torch.relu(cam)  # Keep positive contributions
```

**Note:** [Chefer et al. (2021)](https://arxiv.org/abs/2012.09838) showed that Grad-CAM is outperformed by attention-based methods for Vision Transformers. We use Grad-CAM only for ResNet-50 (where it's appropriate) and attention-based methods for transformers.

---

## Summary Table

| Model | Attention Method | CLS Token? | Implementation | Appropriateness | Notes |
|-------|------------------|------------|----------------|-----------------|-------|
| **DINOv2** | CLS + Rollout | Yes + 4 registers | Correct | **Excellent** | Gold standard; emergent segmentation |
| **DINOv3** | CLS + Rollout | Yes | Correct | **Excellent** | Same as DINOv2 |
| **CLIP** | CLS + Rollout | Yes | Correct | **Good** | Text-alignment training affects patterns |
| **MAE** | CLS + Rollout | Yes | Correct | **Questionable** | CLS wasn't trained for semantic aggregation |
| **SigLIP** | Mean attention | No (uses MAP) | Approximation | **Acceptable** | Mean attention ≠ MAP head attention |
| **ResNet-50** | Grad-CAM | N/A (CNN) | Correct | **Correct** | Standard approach for CNNs |

### Recommendations for Interpretation

1. **Trust DINO attention most:** DINOv2/v3 attention has strong theoretical and empirical support for semantic meaning
2. **CLIP with context:** Remember CLIP attention reflects text-image training objectives
3. **MAE with caution:** Don't interpret MAE attention as "what the model thinks is important" in the same way as DINO
4. **SigLIP as approximation:** Understand that mean attention is a proxy for the actual pooling mechanism
5. **ResNet as baseline:** Use ResNet Grad-CAM to compare SSL attention against supervised learning

---

## References

### Core Implementation Papers

| Paper | Citation | Relevance |
|-------|----------|-----------|
| **DINO** | Caron, M., et al. (2021). Emerging Properties in Self-Supervised Vision Transformers. *ICCV*. [arXiv:2104.14294](https://arxiv.org/abs/2104.14294) | Discovered emergent segmentation in self-attention |
| **Vision Transformers Need Registers** | Darcet, T., et al. (2024). Vision Transformers Need Registers. *ICLR*. [arXiv:2309.16588](https://arxiv.org/abs/2309.16588) | Explains attention artifacts; motivates register tokens |
| **Attention Rollout** | Abnar, S. & Zuidema, W. (2020). Quantifying Attention Flow in Transformers. *ACL*. [arXiv:2005.00928](https://arxiv.org/abs/2005.00928) | Foundation for rollout implementation |
| **MAE** | He, K., et al. (2022). Masked Autoencoders Are Scalable Vision Learners. *CVPR*. [arXiv:2111.06377](https://arxiv.org/abs/2111.06377) | MAE architecture and training |
| **CrossMAE** | Fu, Y., et al. (2024). CrossMAE: Rethinking Patch Dependence for Masked Autoencoders. [crossmae.github.io](https://crossmae.github.io/) | Analysis of MAE attention patterns |
| **Grad-CAM** | Selvaraju, R.R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. *ICCV*. [arXiv:1610.02391](https://arxiv.org/abs/1610.02391) | CNN visualization method |
| **Transformer Interpretability** | Chefer, H., et al. (2021). Transformer Interpretability Beyond Attention Visualization. *CVPR*. [arXiv:2012.09838](https://arxiv.org/abs/2012.09838) | Compares attention vs Grad-CAM for ViTs |

### Training Paradigm Analysis

| Paper | Citation | Relevance |
|-------|----------|-----------|
| **Teaching Matters** | Walmer, M., et al. (2023). Teaching Matters: Investigating the Role of Supervision in Vision Transformers. *CVPR*. [PDF](https://openaccess.thecvf.com/content/CVPR2023/papers/Walmer_Teaching_Matters_Investigating_the_Role_of_Supervision_in_Vision_Transformers_CVPR_2023_paper.pdf) | Compares attention across training paradigms |
| **SigLIP 2** | Tschannen, M., et al. (2025). SigLIP 2: A Better Multilingual Vision Language Encoder. [arXiv:2502.14786](https://arxiv.org/html/2502.14786v1) | SigLIP architecture details |

### Key Source Files

| File | Purpose |
|------|---------|
| `src/ssl_attention/attention/cls_attention.py` | CLS and mean attention extraction |
| `src/ssl_attention/attention/rollout.py` | Attention rollout implementation |
| `src/ssl_attention/models/resnet50.py` | Grad-CAM implementation for CNNs |
| `app/backend/services/similarity_service.py` | Bbox similarity computation |
| `app/frontend/src/utils/renderHeatmap.ts` | Client-side heatmap rendering |
| `app/precompute/generate_attention_cache.py` | Offline attention precomputation |

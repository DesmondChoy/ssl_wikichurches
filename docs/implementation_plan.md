# SSL WikiChurches Implementation Plan

## Overview

Build a system to compare SSL model attention patterns against 631 expert-annotated bounding boxes on 139 WikiChurches images, measuring whether models attend to the same features human experts consider diagnostic.

**Models:** DINOv2, DINOv3, MAE, CLIP, SigLIP (all ViT-B, frozen)
**Primary Metric:** IoU between thresholded attention and expert bounding boxes
**Platform:** M4 Pro with MPS backend

---

## Project Structure

```
ssl_wikichurches/
├── src/ssl_attention/
│   ├── __init__.py
│   ├── config/
│   │   ├── __init__.py
│   │   ├── base.py              # ExperimentConfig dataclass
│   │   └── models.py            # Model-specific configs
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── protocols.py         # VisionBackbone Protocol, ModelOutput
│   │   ├── registry.py          # Model registry with lazy loading
│   │   ├── base.py              # BaseVisionModel ABC
│   │   ├── dinov2.py            # DINOv2 wrapper (facebook/dinov2-with-registers-base)
│   │   ├── dinov3.py            # DINOv3 wrapper (facebook/dinov3-vitb16-pretrain-lvd1689m)
│   │   ├── mae.py               # MAE wrapper (facebook/vit-mae-base)
│   │   ├── clip_model.py        # CLIP wrapper (openai/clip-vit-base-patch16)
│   │   └── siglip.py            # SigLIP wrapper (google/siglip-base-patch16-224)
│   │
│   ├── attention/
│   │   ├── __init__.py
│   │   ├── cls_attention.py     # CLS token attention extraction
│   │   ├── rollout.py           # Attention rollout implementation
│   │   ├── gradcam.py           # GradCAM for transformers
│   │   └── utils.py             # Upsampling, normalization
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── wikichurches.py      # WikiChurchesDataset, AnnotatedSubset
│   │   ├── transforms.py        # Model-specific preprocessing
│   │   └── annotations.py       # BoundingBox, ImageAnnotation dataclasses
│   │
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── iou.py               # IoU computation (attention vs bbox)
│   │   ├── baselines.py         # Random, center, saliency baselines
│   │   └── statistics.py        # t-tests, bootstrap CIs, effect sizes
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── linear_probe.py      # Linear classifier training
│   │   └── ablations.py         # Layer analysis, model comparison
│   │
│   ├── visualization/
│   │   ├── __init__.py
│   │   ├── heatmaps.py          # Attention heatmap generation
│   │   ├── overlays.py          # Bbox + attention overlay
│   │   └── plots.py             # Statistical plots
│   │
│   ├── cache/
│   │   ├── __init__.py
│   │   └── manager.py           # HDF5 feature caching
│   │
│   └── utils/
│       ├── __init__.py
│       ├── device.py            # MPS/CUDA/CPU handling
│       └── logging.py           # Structured logging
│
├── experiments/
│   ├── configs/
│   │   └── default.yaml         # Main experiment config
│   └── scripts/
│       ├── extract_features.py
│       ├── compute_attention.py
│       ├── run_iou_analysis.py
│       └── train_linear_probe.py
│
├── outputs/                     # Git-ignored
│   ├── cache/
│   ├── results/
│   └── figures/
│
└── tests/
```

---

## Model Details (Latest APIs)

| Model | HuggingFace ID | Class | Patch Size | Notes |
|-------|----------------|-------|------------|-------|
| DINOv2 | `facebook/dinov2-with-registers-base` | `AutoModel` | 14 | Registers give cleaner attention maps |
| DINOv3 | `facebook/dinov3-vitb16-pretrain-lvd1689m` | `DINOv3ViTModel` | 16 | Requires transformers 4.56.0+, uses RoPE |
| MAE | `facebook/vit-mae-base` | `ViTMAEModel` | 16 | Set `mask_ratio=0.0` for extraction |
| CLIP | `openai/clip-vit-base-patch16` | `CLIPVisionModel` | 16 | Vision encoder only |
| SigLIP | `google/siglip-base-patch16-224` | `SiglipVisionModel` | 16 | Uses sigmoid, not softmax |

### Key API Patterns

```python
# All models support:
outputs = model(**inputs, output_attentions=True)
attentions = outputs.attentions  # Tuple of (batch, heads, seq, seq) per layer

# DINOv2/v3 with registers:
# Sequence = [CLS] + [REG1-4] + [patches]
num_registers = model.config.num_register_tokens  # 4
cls_to_patches = attn[:, :, 0, 1 + num_registers:]

# MAE (disable masking):
config = ViTMAEConfig.from_pretrained(model_id, mask_ratio=0.0)
model = ViTMAEModel.from_pretrained(model_id, config=config)
```

---

## Implementation Phases

### Phase 1: Core Infrastructure

1. **Update `pyproject.toml`**
   ```toml
   dependencies = [
       "torch>=2.1.0",
       "torchvision>=0.16.0",
       "transformers>=4.56.0",  # For DINOv3 support
       "timm>=1.0.20",
       "pillow>=10.0.0",
       "h5py>=3.10.0",
       "numpy>=1.26.0",
       "scipy>=1.11.0",
       "matplotlib>=3.8.0",
       "seaborn>=0.13.0",
       "scikit-learn>=1.3.0",
       "pyyaml>=6.0.1",
   ]
   ```

2. **Create protocols** (`src/ssl_attention/models/protocols.py`)
   ```python
   @dataclass
   class ModelOutput:
       cls_token: Tensor           # (B, D)
       patch_tokens: Tensor        # (B, N, D)
       attention_weights: list[Tensor]  # List of (B, H, N+1, N+1)

   @runtime_checkable
   class VisionBackbone(Protocol):
       patch_size: int
       embed_dim: int
       num_layers: int
       def forward(self, images: Tensor) -> ModelOutput: ...
   ```

3. **Implement model wrappers** (separate files):
   - `dinov2.py` - Handle registers, patch size 14
   - `dinov3.py` - Handle RoPE, registers, patch size 16
   - `mae.py` - Disable masking with `mask_ratio=0.0`
   - `clip_model.py` - Vision encoder only
   - `siglip.py` - Vision encoder only

4. **Implement attention extractors**:
   - `cls_attention.py` - CLS to patch attention with head fusion
   - `rollout.py` - Attention rollout through layers
   - `gradcam.py` - Gradient-based baseline

### Phase 2: Data Pipeline

1. **Annotation parsing** (`annotations.py`)
   - `BoundingBox` with `to_mask(H, W)` method
   - `ImageAnnotation` parsing `building_parts.json`
   - Handle normalized (0-1) coordinates

2. **Dataset classes** (`wikichurches.py`)
   - `AnnotatedSubset` - 139 images with bboxes
   - `FullDataset` - 9,485 images for linear probe
   - Per-model preprocessing via registry

3. **HDF5 caching** (`cache/manager.py`)
   - Cache features and attention maps
   - Key: `{model}/{layer}/{image_id}`

### Phase 3: Metrics & Evaluation

1. **IoU computation** (`metrics/iou.py`)
   - Threshold at percentiles (top 10%, 20%, 30%)
   - IoU against bbox union
   - Per-bbox breakdown
   - Pointing game accuracy

2. **Baselines** (`metrics/baselines.py`)
   - Random uniform
   - Center Gaussian
   - Sobel edge saliency

3. **Statistics** (`metrics/statistics.py`)
   - Paired t-test / Wilcoxon
   - Bootstrap CIs
   - Cohen's d effect size

4. **Linear probe** (`evaluation/linear_probe.py`)
   - Train on frozen CLS features
   - 4-class and full hierarchy
   - Accuracy, F1, confusion matrix

### Phase 4: Visualization & Analysis

1. **Heatmaps** (`visualization/heatmaps.py`)
   - Upsample to original resolution
   - Colormap overlay

2. **Comparison plots** (`visualization/plots.py`)
   - Model comparison bar charts with CIs
   - Layer-wise progression
   - Per-feature-category breakdown

---

## Critical Files to Create

| Priority | File | Purpose |
|----------|------|---------|
| 1 | `pyproject.toml` | Add ML dependencies |
| 2 | `src/ssl_attention/models/protocols.py` | Core abstractions |
| 3 | `src/ssl_attention/models/dinov2.py` | DINOv2 wrapper |
| 4 | `src/ssl_attention/models/dinov3.py` | DINOv3 wrapper |
| 5 | `src/ssl_attention/models/mae.py` | MAE wrapper |
| 6 | `src/ssl_attention/models/clip_model.py` | CLIP wrapper |
| 7 | `src/ssl_attention/models/siglip.py` | SigLIP wrapper |
| 8 | `src/ssl_attention/attention/cls_attention.py` | Primary attention method |
| 9 | `src/ssl_attention/data/annotations.py` | Bbox parsing |
| 10 | `src/ssl_attention/metrics/iou.py` | Primary metric |
| 11 | `experiments/configs/default.yaml` | Experiment config |

---

## Verification Plan

1. **Unit tests**
   - Model outputs have correct shapes
   - Attention has expected dimensions per model
   - IoU computes correctly on synthetic data

2. **Smoke test**
   - Run full pipeline on 5 images
   - Verify cache writes/reads work
   - Check attention maps are sensible

3. **Baseline sanity check**
   - Random baseline IoU ~5-10%
   - Center baseline slightly higher
   - Model attention should beat both

4. **Linear probe**
   - >50% accuracy on 4-class task
   - Confirms features are discriminative

5. **Visual inspection**
   - Attention overlays on sample images
   - Compare across models qualitatively

---

## Key Design Decisions

1. **Separate model files** - DINOv2 and DINOv3 have different APIs (registers, RoPE, sequence length)
2. **Protocol-based abstractions** - Unified interface despite API differences
3. **Lazy loading with LRU cache** - Max 1-2 models in memory
4. **HDF5 caching** - Avoid recomputing features
5. **Percentile thresholding** - More robust than fixed thresholds
6. **Registers-aware extraction** - Skip register tokens for DINOv2/v3

---

## Notes

- DINOv2 sequence: 1 CLS + 4 registers + 256 patches (patch 14, 224px)
- DINOv3 sequence: 1 CLS + 4 registers + 196 patches (patch 16, 224px)
- MAE sequence: 1 CLS + 196 patches (with mask_ratio=0)
- CLIP/SigLIP: 1 CLS + 196 patches
- Bounding boxes in `building_parts.json` are normalized (0-1)
- 106 architectural feature types for category analysis
- MPS may need `torch.mps.empty_cache()` between batches

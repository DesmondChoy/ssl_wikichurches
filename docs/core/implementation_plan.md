# SSL WikiChurches Implementation Plan

## Progress Status

| Phase | Description | Status |
|-------|-------------|--------|
| Phase 1 | Core Infrastructure | ✅ Complete |
| Phase 2 | Data Pipeline | ✅ Complete |
| Phase 3 | Metrics & Evaluation | ✅ Complete |
| Phase 4 | Visualization & Analysis | ✅ Complete |
| Phase 5 | Fine-Tuning Analysis | 🔄 In Progress |
| Phase 6 | Interactive Analysis Tool | ✅ Complete |

**Last Updated:** 2026-03-02 (Frozen-vs-fine-tuned precompute/API/frontend integration synced; Phase 5 analytics follow-ups tracked)

---

## Overview

Build a system to compare SSL model attention patterns against 631 expert-annotated bounding boxes on 139 WikiChurches images, measuring whether models attend to the same features human experts consider diagnostic.

**Models:** DINOv2, DINOv3, MAE, CLIP, SigLIP, SigLIP 2 (all ViT-B, evaluated frozen and fine-tuned)
**Research Design:** Two-pass analysis comparing attention patterns before and after task-specific fine-tuning
**Primary Metric:** IoU between thresholded attention and expert bounding boxes
**Platform:** M4 Pro with MPS backend

---

## Project Structure

```
ssl_wikichurches/
├── app/                            # Interactive analysis tool
│   ├── backend/                    # FastAPI backend
│   │   ├── routers/                # API route handlers
│   │   ├── services/               # Business logic
│   │   └── main.py                 # FastAPI entry point
│   ├── frontend/                   # React + TypeScript frontend
│   └── precompute/                 # Pre-computation scripts
│
├── notebooks/                      # Jupyter notebooks
│   └── 01_data_exploration.ipynb   # Dataset exploration with Polars
│
├── src/ssl_attention/
│   ├── __init__.py
│   ├── config.py                  # Centralized configuration
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
│   │   ├── siglip.py            # SigLIP wrapper (google/siglip-base-patch16-224)
│   │   ├── siglip2.py           # SigLIP 2 wrapper (google/siglip2-base-patch16-224)
│   │   └── resnet50.py          # ResNet-50 supervised baseline (Grad-CAM)
│   │
│   ├── attention/
│   │   ├── __init__.py
│   │   ├── cls_attention.py     # CLS token attention extraction
│   │   ├── rollout.py           # Attention rollout implementation
│   │
│   ├── data/
│   │   ├── __init__.py
│   │   ├── wikichurches.py      # WikiChurchesDataset, AnnotatedSubset
│   │   └── annotations.py       # BoundingBox, ImageAnnotation dataclasses
│   │
│   ├── metrics/
│   │   ├── __init__.py
│   │   ├── iou.py               # IoU computation (attention vs bbox)
│   │   ├── pointing_game.py     # Pointing game metric
│   │   ├── baselines.py         # Random, center, saliency baselines
│   │   └── statistics.py        # t-tests, bootstrap CIs, effect sizes
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── linear_probe.py      # Linear classifier training
│   │   └── fine_tuning.py       # Full backbone fine-tuning
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
│       └── device.py            # MPS/CUDA/CPU handling
│
├── experiments/
│   └── scripts/
│       ├── fine_tune_models.py  # Fine-tuning script
│       └── analyze_delta_iou.py # Frozen vs fine-tuned delta analysis
│
├── outputs/                     # Git-ignored
│   ├── cache/
│   ├── checkpoints/
│   └── results/
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
| SigLIP | `google/siglip-base-patch16-224` | `SiglipVisionModel` | 16 | No CLS token; mean attention path |
| SigLIP 2 | `google/siglip2-base-patch16-224` | `Siglip2VisionModel` | 16 | Improved dense features; better for attention alignment |

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

### Phase 1: Core Infrastructure ✅ COMPLETE

1. **Update `pyproject.toml`** ✅
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

2. **Create protocols** (`src/ssl_attention/models/protocols.py`) ✅
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

3. **Implement model wrappers** (separate files): ✅
   - `dinov2.py` - Handle registers, patch size 14 ✅
   - `dinov3.py` - Handle RoPE, registers, patch size 16 ✅
   - `mae.py` - Disable masking with `mask_ratio=0.0` ✅
   - `clip_model.py` - Vision encoder only ✅
   - `siglip.py` - Vision encoder only (SigLIP) ✅
   - `siglip2.py` - Vision encoder only (SigLIP 2) ✅

4. **Implement attention extractors**: ✅
   - `cls_attention.py` - CLS to patch attention with head fusion ✅
   - `rollout.py` - Attention rollout through layers ✅
   - Grad-CAM (in `models/resnet50.py`) - Gradient-based baseline ✅

### Phase 2: Data Pipeline ✅ COMPLETE

1. **Annotation parsing** (`annotations.py`) ✅
   - `BoundingBox` with `to_mask(H, W)` method
   - `ImageAnnotation` parsing `building_parts.json`
   - Handle normalized (0-1) coordinates with clamping

2. **Dataset classes** (`wikichurches.py`) ✅
   - `AnnotatedSubset` - 139 images with bboxes
   - `FullDataset` - 9,502 files in local mirror / 9,485 in official release (use `filter_labeled=True` for 4-style training subset)
   - Per-model preprocessing via registry

3. **HDF5 caching** (`cache/manager.py`) ✅
   - Cache features and attention maps
   - Key: `{model}/{layer}/{image_id}`
   - LRU eviction and corruption detection

### Phase 3: Metrics & Evaluation ✅ COMPLETE

1. **IoU computation** (`metrics/iou.py`) ✅
   - Threshold at percentiles (90/80/70/60/50)
   - IoU against bbox union with coverage metric
   - Per-bbox breakdown
   - CorLoc@50 for literature comparison

2. **Pointing game** (`metrics/pointing_game.py`) ✅
   - Binary hit detection (max attention in bbox)
   - Top-k pointing accuracy
   - Per-feature-type breakdown

3. **Baselines** (`metrics/baselines.py`) ✅
   - Random uniform
   - Center Gaussian
   - Sobel edge saliency
   - Saliency prior (center + border suppression)

4. **Statistics** (`metrics/statistics.py`) ✅
   - Paired t-test / Wilcoxon signed-rank
   - Bootstrap CIs (10k samples)
   - Cohen's d effect size
   - Holm multiple comparison correction

5. **Linear probe** (`evaluation/linear_probe.py`) ✅
   - Train on frozen CLS features (sklearn LogisticRegression)
   - Stratified k-fold cross-validation
   - Accuracy, F1, per-class accuracy, confusion matrix

> **Note:** `compute_corloc()` is implemented but not yet integrated into the precompute pipeline or API. Integration is optional—CorLoc@50 is primarily for literature comparison with DINO papers.

### Phase 4: Visualization & Analysis ✅ COMPLETE

1. **Heatmaps** (`visualization/heatmaps.py`) ✅
   - Upsample to original resolution
   - Colormap overlay with configurable colormaps

2. **Overlays** (`visualization/overlays.py`) ✅
   - Bounding box drawing with labels
   - Attention heatmap overlay on images

3. **Comparison plots** (`visualization/plots.py`) ✅
   - Model comparison bar charts with CIs
   - Layer-wise progression
   - Per-feature-category breakdown
   - Style breakdown charts
   - Scatter plots for coverage vs IoU

### Phase 5: Fine-Tuning Analysis 🔄 IN PROGRESS

1. **Fine-tuning implementation** (`evaluation/fine_tuning.py`) ✅
   - `FineTuningConfig` dataclass for hyperparameters
   - `FineTuningResult` dataclass for training metrics and checkpoint path
   - `FineTunableModel` wrapping SSL backbone + classification head
   - `FineTuner` class with training loop, stratified split, class weighting
   - `ClassificationHead` linear classifier on CLS token
   - Differential learning rates for backbone vs head
   - MPS memory management (`torch.mps.empty_cache()`), checkpoint saving
   - `load_finetuned_model()` for loading trained checkpoints
   - `save_training_results()` for JSON export of training history
   - **LoRA fine-tuning** via HuggingFace PEFT with model-specific target modules (rank=8, alpha=32)
   - **Cosine LR scheduler** with linear warmup (10% warmup ratio)
   - **Gradient clipping** via `clip_grad_norm_` (max_norm=1.0)
   - **Data augmentation** — RandomResizedCrop, HorizontalFlip, ColorJitter (toggleable)

2. **Fine-tuning script** (`experiments/scripts/fine_tune_models.py`) ✅
   - Train single model or all models via CLI flags
   - Configurable hyperparameters (epochs, batch size, learning rates)
   - Head-only training option (`--freeze-backbone`)
   - Fine-tunable model allowlist enforced (`resnet50` excluded)
   - Save checkpoints: `outputs/checkpoints/{model}_finetuned.pt` (linear probe / full) or `{model}_lora_finetuned.pt` (LoRA)
   - Training summary with per-model results
   - `generate_attention_cache.py --finetuned` caches fine-tuned model attention maps
   - `generate_heatmap_images.py --finetuned` renders fine-tuned overlay PNGs under `{model}_finetuned` keys
   - Checkpoint discovery checks both `{model}_finetuned.pt` and `{model}_lora_finetuned.pt`, preferring LoRA when both exist

3. **Comparative analysis** (`experiments/scripts/analyze_delta_iou.py`) ✅
   - Load frozen baseline and fine-tuned models
   - Extract attention on annotated subset
   - Compute Δ IoU per model with per-image breakdown
   - Bootstrap 95% CIs, Cohen's d effect sizes
   - Paired t-test / Wilcoxon (auto-selected based on normality)
   - Holm correction for multiple comparisons across models
   - JSON export of full results

4. **Visualization** 🔄
   - Side-by-side heatmaps (frozen vs fine-tuned) integrated across precompute/API/frontend
   - Attention shift maps (where did attention move?) — tracked in issue #474

> **Note:** Remaining Phase 5 work includes (a) fine-tuned metrics-cache/leaderboard integration and (b) attention shift visualization. See [Fine-Tuning Methods](../enhancements/fine_tuning_methods.md) for detailed research on Linear Probe vs LoRA vs Full fine-tuning approaches.

### Phase 6: Interactive Analysis Tool ✅ COMPLETE

**Technology Choice:** React + FastAPI (full control, production-ready)

1. **Backend** (`app/backend/`) ✅
   - FastAPI with routers for images, attention, metrics, comparison
   - Services for image loading, metrics querying (SQLite), caching
   - Pre-computation scripts for attention maps, heatmaps, and metrics

2. **Frontend** (`app/frontend/`) ✅
   - React + TypeScript + Vite
   - Image browser with style filtering
   - Attention viewer with model/layer selection
   - Model comparison views

3. **Pre-computation Pipeline** ✅
   - `generate_attention_cache.py` - Extract attention to HDF5
   - `generate_heatmap_images.py` - Render heatmap overlays as PNGs
   - `generate_metrics_cache.py` - Compute IoU to SQLite

4. **API Endpoints** ✅
   - `/api/images` - Image listing, filtering, serving
   - `/api/attention` - Heatmap and overlay serving
   - `/api/metrics` - IoU metrics, leaderboard, layer progression
   - `/api/compare` - Model comparison, frozen vs fine-tuned

5. **Representation Similarity Exploration (Utility Feature)** ✅
   - Click on bounding box to compute cosine similarity with all image patches
   - Visualize which regions share similar learned representations
   - Compare feature similarity across models and layers
   - **Files added:**
     - `app/precompute/generate_feature_cache.py` - Cache patch tokens to HDF5
     - `app/backend/services/similarity_service.py` - Bbox-to-patch similarity computation
     - `POST /api/attention/{image_id}/similarity` - Similarity API endpoint
     - `app/frontend/src/components/attention/InteractiveBboxOverlay.tsx` - Clickable bbox overlay
     - `app/frontend/src/utils/renderHeatmap.ts` - Client-side viridis heatmap rendering

6. **Attention Method Selection** ✅
   - Update precompute to generate CLS, rollout, and GradCAM heatmaps
   - Add method parameter to `/api/attention` endpoints
   - Add method selector dropdown to ControlPanel.tsx

7. **Per-Feature-Type Breakdown** ✅
   - Add `/api/metrics/model/{model}/feature_breakdown` endpoint
   - Create FeatureBreakdown.tsx component
   - Display IoU by 106 architectural feature types

8. **Per-Model Layer Counts** ✅
   - `/api/attention/models` returns `num_layers_per_model` dict (e.g., ResNet-50 → 4, ViTs → 12)
   - `viewStore` tracks per-model counts, clamps layer on model switch
   - `LayerSlider` accepts dynamic `maxLayers` prop

9. **Per-Bbox Metrics** ✅
   - `GET /api/metrics/{image_id}/bbox/{bbox_index}` computes IoU/coverage for individual bboxes
   - Frontend switches between per-bbox and union-of-all metrics based on selection
   - Green context indicator in IoUDisplay showing selected bbox name

10. **Max IoU Progress Bar** ✅
    - Shows observed IoU as percentage of theoretical maximum
    - Color-coded: green ≥75%, yellow ≥50%, orange ≥25%, red <25%

11. **UI Polish** ✅
    - Annotations panel moved to left sidebar (3-column layout: annotations | viewer | controls)
    - Bounding boxes shown by default
    - Portal-rendered tooltips (escape overflow-hidden containers)
    - `keepPreviousData` on React Query hooks prevents UI flash during layer animation

12. **Interactive Bbox Similarity Comparison** ✅
   - Add clickable bounding boxes to Model Comparison page (`/compare?type=models`)
   - Clicking a bbox shows similarity heatmaps for both models simultaneously
   - Synchronized selection across both panels for direct comparison
   - **Files added:**
     - `app/frontend/src/components/comparison/SimilarityViewer.tsx` - Bbox overlay + similarity heatmap viewer
     - `app/frontend/src/components/comparison/ModelCompare.tsx` - Updated with synchronized selection
     - Selection info bar, clear button, and colormap legend

---

## Critical Files to Create

| Priority | File | Purpose | Status |
|----------|------|---------|--------|
| 1 | `pyproject.toml` | Add ML dependencies | ✅ Done |
| 2 | `src/ssl_attention/models/protocols.py` | Core abstractions | ✅ Done |
| 3 | `src/ssl_attention/models/dinov2.py` | DINOv2 wrapper | ✅ Done |
| 4 | `src/ssl_attention/models/dinov3.py` | DINOv3 wrapper | ✅ Done |
| 5 | `src/ssl_attention/models/mae.py` | MAE wrapper | ✅ Done |
| 6 | `src/ssl_attention/models/clip_model.py` | CLIP wrapper | ✅ Done |
| 7 | `src/ssl_attention/models/siglip.py` | SigLIP wrapper | ✅ Done |
| 7b | `src/ssl_attention/models/siglip2.py` | SigLIP 2 wrapper | ✅ Done |
| 21 | `src/ssl_attention/models/resnet50.py` | ResNet-50 supervised baseline | ✅ Done |
| 8 | `src/ssl_attention/attention/cls_attention.py` | Primary attention method | ✅ Done |
| 9 | `src/ssl_attention/data/annotations.py` | Bbox parsing | ✅ Done |
| 10 | `src/ssl_attention/data/wikichurches.py` | Dataset classes | ✅ Done |
| 11 | `src/ssl_attention/cache/manager.py` | HDF5 caching | ✅ Done |
| 12 | `src/ssl_attention/metrics/iou.py` | Primary metric | ✅ Done |
| 13 | `src/ssl_attention/metrics/pointing_game.py` | Pointing game metric | ✅ Done |
| 14 | `src/ssl_attention/metrics/baselines.py` | Baseline generators | ✅ Done |
| 15 | `src/ssl_attention/metrics/statistics.py` | Statistical tests | ✅ Done |
| 16 | `src/ssl_attention/evaluation/linear_probe.py` | Linear probe evaluation | ✅ Done |
| 17 | `src/ssl_attention/evaluation/fine_tuning.py` | Fine-tuning wrapper | ✅ Done |
| 18 | `experiments/scripts/fine_tune_models.py` | Training script | ✅ Done |
| 19 | `app/backend/main.py` | Interactive analysis tool backend | ✅ Done |
| 20 | `app/frontend/` | Interactive analysis tool frontend | ✅ Done |

### Additional Phase 1 Files Created

| File | Purpose | Status |
|------|---------|--------|
| `src/ssl_attention/models/base.py` | BaseVisionModel ABC | ✅ Done |
| `src/ssl_attention/models/registry.py` | Model registry with lazy loading | ✅ Done |
| `src/ssl_attention/attention/rollout.py` | Attention rollout implementation | ✅ Done |
| `src/ssl_attention/models/resnet50.py` | Grad-CAM for CNNs (ResNet-50) | ✅ Done |
| `src/ssl_attention/config.py` | Centralized configuration | ✅ Done |
| `src/ssl_attention/utils/device.py` | MPS/CUDA/CPU handling | ✅ Done |

### Additional Phase 2 Files Created

| File | Purpose | Status |
|------|---------|--------|
| `src/ssl_attention/data/__init__.py` | Data module exports | ✅ Done |
| `src/ssl_attention/cache/__init__.py` | Cache module exports | ✅ Done |

### Additional Phase 3 Files Created

| File | Purpose | Status |
|------|---------|--------|
| `src/ssl_attention/metrics/__init__.py` | Metrics module exports | ✅ Done |
| `src/ssl_attention/evaluation/__init__.py` | Evaluation module exports | ✅ Done |

### Additional Phase 4 Files Created

| File | Purpose | Status |
|------|---------|--------|
| `src/ssl_attention/visualization/__init__.py` | Visualization module exports | ✅ Done |
| `src/ssl_attention/visualization/heatmaps.py` | Attention heatmap generation | ✅ Done |
| `src/ssl_attention/visualization/overlays.py` | Bbox + attention overlay | ✅ Done |
| `src/ssl_attention/visualization/plots.py` | Statistical plots | ✅ Done |
| `notebooks/01_data_exploration.ipynb` | Dataset exploration with Polars | ✅ Done |

### Phase 5 Files Created

| File | Purpose | Status |
|------|---------|--------|
| `src/ssl_attention/evaluation/fine_tuning.py` | Fine-tuning wrapper with FineTunableModel | ✅ Done |
| `experiments/scripts/fine_tune_models.py` | CLI training script | ✅ Done |

### Phase 6 Files Created

| File | Purpose | Status |
|------|---------|--------|
| `app/backend/main.py` | FastAPI application entry | ✅ Done |
| `app/backend/config.py` | Backend configuration | ✅ Done |
| `app/backend/schemas/models.py` | Pydantic schemas | ✅ Done |
| `app/backend/routers/` | API route handlers | ✅ Done |
| `app/backend/services/` | Business logic services | ✅ Done |
| `app/precompute/` | Pre-computation scripts | ✅ Done |
| `app/frontend/` | React + TypeScript frontend | ✅ Done |

### Phase 6 Enhancement: Similarity Exploration

| File | Purpose | Status |
|------|---------|--------|
| `app/precompute/generate_feature_cache.py` | Cache CLS + patch tokens to HDF5 | ✅ Done |
| `app/backend/services/similarity_service.py` | Bbox-to-patch cosine similarity | ✅ Done |
| `app/backend/schemas/models.py` | Added BboxInput, SimilarityResponse | ✅ Done |
| `app/frontend/src/components/attention/InteractiveBboxOverlay.tsx` | Clickable bbox SVG overlay | ✅ Done |
| `app/frontend/src/utils/renderHeatmap.ts` | Client-side viridis heatmap | ✅ Done |

### Phase 6 Enhancement: Error Handling & Validation

| File | Purpose | Status |
|------|---------|--------|
| `app/frontend/src/components/ui/ErrorBoundary.tsx` | React error boundary for graceful failure isolation | ✅ Done |
| `app/backend/validators.py` | Centralized parameter validation (model, layer, method) | ✅ Done |
| `app/frontend/src/constants/percentiles.ts` | Shared percentile constants for frontend | ✅ Done |

### Phase 6 Enhancement: Interactive Bbox Comparison

| File | Purpose | Status |
|------|---------|--------|
| `app/frontend/src/components/comparison/SimilarityViewer.tsx` | Bbox overlay + similarity heatmap viewer | ✅ Done |
| `app/frontend/src/components/comparison/ModelCompare.tsx` | Synchronized bbox selection across panels | ✅ Done |

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

6. **Fine-tuning verification**
   - Training converges (loss decreases, val accuracy improves)
   - Fine-tuned models load correctly
   - Attention extraction works on fine-tuned models
   - IoU comparison shows measurable difference

7. **Interactive tool verification**
   - All 139 images load correctly
   - Model/method/layer selectors function
   - Comparison view synchronizes properly
   - Metrics dashboard displays correct values
   - Responsive on target browsers

---

## Key Design Decisions

1. **Separate model files** - DINOv2 and DINOv3 have different APIs (registers, RoPE, sequence length)
2. **Protocol-based abstractions** - Unified interface despite API differences
3. **Lazy loading with LRU cache** - Max 1-2 models in memory
4. **HDF5 caching** - Avoid recomputing features
5. **Percentile thresholding** - More robust than fixed thresholds
6. **Registers-aware extraction** - Skip register tokens for DINOv2/v3
7. **Two-pass evaluation** - Compare frozen and fine-tuned attention to isolate effect of task-specific training

---

## Notes

- DINOv2 sequence: 1 CLS + 4 registers + 256 patches (patch 14, 224px)
- DINOv3 sequence: 1 CLS + 4 registers + 196 patches (patch 16, 224px)
- MAE sequence: 1 CLS + 196 patches (with mask_ratio=0)
- CLIP: 1 CLS + 196 patches
- SigLIP: 196 patches (no CLS token; mean attention method)
- SigLIP 2: 196 patches (no CLS token; mean attention method)

### Dataset Details (Verified)

- **Annotation file:** `building_parts.json` with nested structure:
  - `meta`: 106 feature type definitions with hierarchical parent relationships
  - `annotations`: 139 images with bounding box groups
- **Coordinate format:** `left, top, width, height` (normalized 0-1, some edge values slightly negative—clamp to [0,1])
- **Style IDs:** Wikidata Q-IDs requiring mapping:
  - `Q46261` → Romanesque (54 churches, 39%)
  - `Q176483` → Gothic (49 churches, 35%)
  - `Q236122` → Renaissance (22 churches, 16%)
  - `Q840829` → Baroque (17 churches, 12%)
- **Bbox structure:** `annotations[image_id].bbox_groups[].elements[]` (nested, grouped by related features)

### Technical Notes

- MPS may need `torch.mps.empty_cache()` between batches

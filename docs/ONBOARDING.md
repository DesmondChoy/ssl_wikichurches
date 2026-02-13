# Teammate Onboarding: SSL WikiChurches (Deep Reference)

> **Quick-start version:** See [`docs/core/teammate_onboarding.md`](core/teammate_onboarding.md) for a concise overview with a suggested first-2-days onboarding flow.

Welcome to the project! This document is the comprehensive reference — it covers what we've built, the research questions, where the codebase stands, and what's ahead, with enough technical depth to navigate the implementation confidently.

---

## 1. What This Project Is About

This project is organized around **three research questions**, each with its own methodology, results, and implementation scope:

| # | Research Question | Method | Status |
|---|-------------------|--------|--------|
| **Q1** | Do frozen SSL models attend to expert-identified architectural features? | IoU between thresholded attention maps and 631 expert bounding boxes across 6 models × 12 layers × 7 percentile thresholds | **Answered** — full results available |
| **Q2** | Does fine-tuning shift attention toward expert features, and does the strategy (Linear Probe vs LoRA vs Full) matter? | Delta-IoU (fine-tuned minus frozen) with paired statistical tests (Wilcoxon, Holm correction) | **Partially answered** — infrastructure built, preliminary results for 2 models, final hyperparameter tuning and full multi-strategy comparison still needed |
| **Q3** | Do individual attention heads specialize for different architectural features? | Per-head IoU against expert bounding boxes; rank-based analysis to identify heads aligned with specific feature types | **Not yet started** — design docs written, implementation pending |

We compare 6 vision models against 631 expert-annotated bounding boxes across 139 church images from the [WikiChurches dataset](https://arxiv.org/abs/2108.06959). The benchmark uses the WikiChurches expert annotations to evaluate attention alignment (Q1), measure how fine-tuning shifts that alignment (Q2), and identify whether individual heads develop feature-level specialization (Q3).

### Preliminary Results (Q1)

| Model | Best IoU (frozen, 90th %ile) | Rank | Key Observation |
|-------|:---:|:---:|-----------------|
| DINOv3 | 0.133 | 1 | Self-distillation leads; best at final layer |
| ResNet-50 | 0.090 | 2 | Supervised CNN competitive via Grad-CAM |
| DINOv2 | 0.082 | 3 | Strong semantic attention in later layers |
| CLIP | 0.049 | 4 | Best at layer 0 — attention degrades deeper |
| SigLIP 2 | 0.047 | 5 | Mid-layer peak, no CLS token |
| MAE | 0.037 | 6 | Near-uniform across layers; reconstruction != localization |

### Preliminary Results (Q2 — Delta IoU)

- **DINOv2**: +0.009 delta-IoU at 90th %ile, raw p=0.031 but corrected p=0.052 (Holm) — NOT significant after multiple comparison correction
- **SigLIP**: +0.025 delta-IoU, corrected p < 1e-7 — significant across all percentiles

These are from a single preliminary training run. Final results require proper hyperparameter sweeps.

---

## 2. Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                     Data Flow                           │
│                                                         │
│  Precompute (one-time)  →  HDF5/PNG cache  →  FastAPI  │
│       ↓                        ↓                ↓       │
│  Model inference          Disk storage     REST API     │
│  Attention extraction     ~8-9GB total     26 endpoints │
│  Metrics computation                           ↓        │
│                                          React frontend │
│                                          localhost:5173  │
└─────────────────────────────────────────────────────────┘
```

The key design principle is **precompute + serve**: heavy model inference happens once offline; the web app only reads from pre-computed caches (HDF5 for tensors, PNG for heatmap images, SQLite for metrics).

### Directory Layout

```
ssl_wikichurches/
├── src/ssl_attention/      # Core library (models, attention, metrics, data)
│   ├── models/             #   6 model wrappers + registry + protocols
│   ├── attention/          #   CLS attention, rollout, mean attention
│   ├── metrics/            #   IoU, pointing game, statistics
│   ├── data/               #   Dataset classes, annotation loading
│   ├── evaluation/         #   Fine-tuning (3 strategies), linear probe
│   ├── cache/              #   HDF5 cache manager
│   ├── visualization/      #   Heatmaps, overlays, matplotlib plots
│   ├── utils/              #   Device detection, memory management
│   └── config.py           #   All model configs, constants, paths
│
├── app/
│   ├── backend/            # FastAPI server
│   │   ├── routers/        #   images, attention, metrics, comparison
│   │   ├── services/       #   image_service, attention_service, metrics_service
│   │   └── schemas/        #   Pydantic response models
│   ├── frontend/           # React + Vite + Tailwind
│   │   └── src/
│   │       ├── pages/      #   Home, ImageDetail, Compare, Dashboard
│   │       ├── components/ #   attention/, comparison/, metrics/, ui/
│   │       ├── hooks/      #   React Query hooks for API calls
│   │       ├── api/        #   API client with typed endpoints
│   │       └── constants/  #   Glossary, percentile options
│   └── precompute/         # 4 cache generation scripts
│
├── tests/                  # 273 pytest tests (17 test files across 8 areas)
├── docs/                   # Research docs, API reference, enhancement proposals
├── experiments/            # Training & analysis CLI scripts
├── dataset/                # WikiChurches data (gitignored)
└── outputs/                # Caches & checkpoints (gitignored)
```

---

## 3. The Models

All SSL models use **ViT-Base** (12 layers, 768 hidden dim, 12 heads) for controlled comparison. ResNet-50 serves as a supervised CNN baseline.

| Model | HuggingFace ID | Patch Size | Tokens | Attention Method | Key Detail |
|-------|----------------|:---:|:---:|---------|------------|
| DINOv2 | `facebook/dinov2-with-registers-base` | 14×14 | 256 + CLS + 4 reg | CLS, Rollout | 4 register tokens must be skipped |
| DINOv3 | `facebook/dinov3-vitb16-pretrain-lvd1689m` | 16×16 | 196 + CLS + 4 reg | CLS, Rollout | RoPE positional encoding |
| MAE | `facebook/vit-mae-base` | 16×16 | 196 + CLS | CLS, Rollout | mask_ratio forced to 0 at inference |
| CLIP | `openai/clip-vit-base-patch16` | 16×16 | 196 + CLS | CLS, Rollout | Vision encoder only |
| SigLIP 2 | `google/siglip2-base-patch16-224` | 16×16 | 196 (no CLS!) | Mean | Uses MAP attention pooling; wrapper pins processor to fixed 224×224 |
| ResNet-50 | `torchvision` | N/A | 7×7 grid | Grad-CAM | CNN; 4 layers with hooks |

**Important implementation details:**
- All models implement the `VisionBackbone` protocol and return a standardized `ModelOutput` dataclass
- Models are lazy-loaded via `get_model()` with an LRU cache (max 2 in GPU memory)
- SigLIP special handling is already implemented in `src/ssl_attention/models/siglip.py`: no CLS token (pooler used) and processor pinned to fixed 224×224 for consistent patch grids
- For attention extraction, register tokens (DINOv2/v3) are stripped from the attention weights before computing patch-level attention

---

## 4. How Attention Is Extracted and Evaluated

### Attention Methods

| Method | Used By | How It Works |
|--------|---------|--------------|
| **CLS** | DINOv2, DINOv3, MAE, CLIP | Take the CLS token's attention row → (num_patches,) vector showing where CLS "looks" |
| **Rollout** | DINOv2, DINOv3, MAE, CLIP | Multiply attention matrices across layers to capture indirect information flow (A→B→C) |
| **Mean** | SigLIP | Average attention received by each patch across all heads (no CLS token available) |
| **Grad-CAM** | ResNet-50 | Gradient-weighted class activation maps on conv feature maps |

### Evaluation Metrics

| Metric | What It Measures | Threshold-Dependent? |
|--------|-----------------|:---:|
| **IoU** | Spatial overlap between thresholded attention mask and expert bounding box union mask | Yes (e.g., 90th %ile = top 10%) |
| **Coverage** | Fraction of total attention *energy* falling inside annotated regions | No — uses raw attention values |
| **Pointing Game** | Does the single highest-attention point fall inside a bounding box? | No |
| **CorLoc@50** | Binary: does IoU exceed 0.5 for at least one bbox? | Yes |

**Critical implementation detail (IoU thresholding):** We use `torch.topk()` for exact pixel counts, not `torch.quantile()`. The quantile approach was buggy — tied values at the threshold boundary led to inconsistent mask sizes. This was a significant bug that was caught and fixed (issue #x03).

### Statistical Testing

All model comparisons use:
- **Wilcoxon signed-rank test** (non-parametric, paired)
- **Holm-Bonferroni correction** for multiple comparisons
- **Cohen's d** for effect sizes
- **Bootstrap confidence intervals** (10k samples)

---

## 5. Fine-Tuning System

### The Task

4-class architectural style classification: Romanesque, Gothic, Renaissance, Baroque.

### Dataset Split

- **~4,713 labeled images** from the `Q*_wd0.jpg` subset used by `FullDataset` (9,207 `wd0` images loaded; 9,502 total image files on disk)
- **139 evaluation images** (those with expert bounding boxes) are **explicitly excluded** from training to prevent data leakage
- 80/20 stratified train/val split

### Three Strategies

| Strategy | Config Flag | What Trains | Parameters |
|----------|-------------|-------------|:---:|
| **Linear Probe** | `freeze_backbone=True` | Classification head only | ~3K |
| **LoRA** | `use_lora=True` | Low-rank adapters on Q/V attention projections | ~0.1-1% of backbone |
| **Full** | default | Entire backbone + classification head | ~86M |

### Training Features

- Cosine LR scheduler with linear warmup
- Differential learning rates (backbone: 1e-5, head: 1e-3)
- Class-weighted cross-entropy loss (handles imbalanced style distribution)
- Gradient clipping (max_norm=1.0)
- Data augmentation: RandomResizedCrop, HorizontalFlip, ColorJitter
- Checkpoint saving (best validation accuracy)

### Key Classes

```
FineTuningConfig     — All hyperparameters (dataclass)
FineTunableModel     — Backbone + ClassificationHead wrapper
FineTuner            — Training orchestrator (split, train, validate, save)
FineTuningResult     — Results container (best_val_acc, history, checkpoint_path)
load_finetuned_model — Load checkpoint back for attention re-extraction
```

All in `src/ssl_attention/evaluation/fine_tuning.py`.

---

## 6. The Web Application

### Pages

| Page | URL | Purpose |
|------|-----|---------|
| **Home** | `/` | Image gallery with style filtering, thumbnail grid |
| **Image Detail** | `/image/:id` | Main analysis view — attention heatmap overlay, model/layer/method selection, IoU metrics, interactive bounding boxes |
| **Compare** | `/compare` | Side-by-side model comparison, feature similarity heatmaps |
| **Dashboard** | `/dashboard` | Model leaderboard, aggregate metrics, layer progression charts |

### Key Frontend Components

| Component | What It Does |
|-----------|--------------|
| `AttentionViewer` | Renders attention heatmap overlay on the church image |
| `ControlPanel` | Model, layer, method, percentile selectors |
| `InteractiveBboxOverlay` | Click bounding boxes to get per-feature IoU |
| `IoUDisplay` | Shows IoU score with relative-to-theoretical-max bar |
| `ModelLeaderboard` | Ranked model table by best IoU |
| `SimilarityViewer` | Feature similarity heatmap (click bbox → see similar regions) |
| `FrozenVsFinetuned` | Side-by-side frozen vs fine-tuned comparison (currently hardcoded unavailable pending end-to-end fine-tuned integration) |

### Backend API (26 endpoints)

Four routers:
- **Images** (`/api/images/`) — List, detail, serve image files, thumbnails, bbox overlays
- **Attention** (`/api/attention/`) — Heatmap images, raw attention data, layer progression
- **Metrics** (`/api/metrics/`) — IoU scores, leaderboard, summary, per-bbox metrics, style/feature breakdowns
- **Comparison** (`/api/compare/`) — Feature similarity, frozen vs fine-tuned attention

### Tech Stack

- **Backend:** FastAPI + Pydantic + uvicorn
- **Frontend:** React 19 + TypeScript + Vite + Tailwind CSS + React Query (TanStack Query)
- **State management:** Zustand (simple store for UI state)
- **Data fetching:** React Query with typed API client

---

## 7. Precompute Pipeline

Run these once to generate all caches (10-30 min on M4 Pro with MPS):

```bash
# 1. Attention maps → HDF5
python -m app.precompute.generate_attention_cache --models all

# 2. Feature embeddings → HDF5 (for similarity viewer)
python -m app.precompute.generate_feature_cache --models all

# 3. Heatmap PNGs → outputs/heatmaps/
python -m app.precompute.generate_heatmap_images --colormap viridis

# 4. Metrics → SQLite database
python -m app.precompute.generate_metrics_cache
```

For fine-tuned model attention (needed for frozen vs fine-tuned comparison):
```bash
python -m app.precompute.generate_attention_cache --finetuned --models all
```

---

## 8. Test Suite

**273 tests** across 17 test files (8 test areas), all passing.

| Module | Tests | What's Covered |
|--------|:---:|----------------|
| `test_models/test_model_outputs.py` | All 6 models | ModelOutput shapes, batch consistency, sequence layout |
| `test_models/test_gradcam.py` | ResNet-50 | Grad-CAM hook attachment, heatmap generation |
| `test_attention/test_cls_attention.py` | CLS attention | Head fusion, register skipping, heatmap upsampling |
| `test_attention/test_rollout.py` | Rollout | Layer range, batch independence, identity init |
| `test_attention/test_mean_attention.py` | SigLIP | Mean attention extraction |
| `test_metrics/test_iou.py` | IoU | Thresholding, batch computation, edge cases |
| `test_metrics/test_pointing_game.py` | Pointing game | Hit detection, tolerance, top-k |
| `test_metrics/test_statistics.py` | Statistics | t-tests, Wilcoxon, bootstrap, correction |
| `test_data/test_annotations.py` | Annotations | Bbox parsing, mask generation |
| `test_data/test_wikichurches.py` | Dataset | Image loading, style labels |
| `test_evaluation/test_lora.py` | LoRA | Target module resolution per model |
| `test_services/` | Backend | Validators, metrics service, similarity service |
| `test_backend/` | API | Error responses, comparison validation |
| `test_precompute/` | Scripts | Attention cache generation |

**Notable gap:** No test coverage for the fine-tuning training loop itself (open issue #4up).

### Running Tests

```bash
pytest                    # All tests
pytest -x                 # Stop at first failure
pytest tests/test_metrics # Specific module
pytest --cov              # With coverage
```

---

## 9. Development Workflow

### Getting Started

```bash
# Prerequisites: Python 3.12+, uv, Node.js 18+
uv sync                   # Install Python dependencies
cd app/frontend && npm install && cd ../..

# Download dataset from Google Drive → dataset/
# (link in README.md)

# Generate caches (needs GPU/MPS — 10-30 min)
python -m app.precompute.generate_attention_cache --models all
python -m app.precompute.generate_feature_cache --models all
python -m app.precompute.generate_heatmap_images --colormap viridis
python -m app.precompute.generate_metrics_cache

# Run the app
./dev.sh  # Starts backend :8000 + frontend :5173
```

### Issue Tracking

We use **bd** (beads) for issue tracking:

```bash
bd onboard                # Get started
bd ready                  # Find available work
bd show <id>              # View issue details
bd update <id> --status in_progress  # Claim work
bd close <id>             # Complete work
bd sync                   # Sync with git (exports to JSONL)
```

### Quality Gates

Before committing:
```bash
pytest                           # Tests pass
ruff check src/ app/ tests/      # Linting
ruff format --check src/ app/    # Formatting
cd app/frontend && npx eslint src/ --ext .ts,.tsx  # Frontend lint
```

---

## 10. What's Done and What Needs Work

### Completed (Phases 1-4, 6)

- **Core infrastructure**: 6 model wrappers with standardized protocol, lazy loading, device management
- **Data pipeline**: Annotation parsing, dataset classes (AnnotatedSubset for eval, FullDataset for training), bounding box mask generation
- **Metrics engine**: IoU, coverage, pointing game, CorLoc, all with proper statistical testing (Wilcoxon, bootstrap CI, Holm correction)
- **Full web application**: 4 pages, 26 API endpoints, interactive attention visualization, per-bbox metrics, model comparison, leaderboard dashboard
- **Precompute pipeline**: Attention cache, feature cache, heatmap generation, metrics database
- **Fine-tuning infrastructure**: All 3 strategies (linear probe, LoRA, full) implemented and working

### In Progress (Phase 5 — Fine-Tuning Analysis)

This is where you come in. The fine-tuning *infrastructure* is built, but the research analysis is incomplete:

1. **No saved checkpoints currently** — Preliminary checkpoints were generated with quick single-epoch runs, then deleted. Need proper training with final hyperparameters.

2. **Frozen vs fine-tuned comparison pipeline needs integration** — The compare flow exists in the frontend (`FrozenVsFinetuned` component) and backend (`/api/compare/frozen_vs_finetuned`), but it is not yet wired end-to-end to fine-tuned precompute artifacts (open issue #9ct.1).

3. **Delta IoU analysis exists but needs final runs** — The `experiments/scripts/analyze_delta_iou.py` script and statistical analysis are fully functional, but current results are from preliminary single-epoch training runs, not properly tuned models.

### Open Issues

| Issue | Priority | Description | Blocked By |
|-------|:---:|-------------|------------|
| **#9ct.1** | P2 | Complete frozen-vs-fine-tuned comparison integration pipeline | #9ct (parent audit) |
| **#9ct.5** | P2 | Remove metadata-only image I/O in fine-tuning split prep | #9ct |
| **#9ct.7** | P2 | Reconcile documentation drift | #9ct |
| **#474** | P2 | Add attention shift visualization for frozen vs fine-tuned | Fine-tuning decisions |
| **#9ct.8** | P3 | Remove/consolidate unused cache_service abstraction | #9ct |
| **#4up** | P3 | No test coverage for fine-tuning training loop | Fine-tuning decisions |

### Not Yet Implemented (Q3 — Per-Head Analysis)

Research question Q3 asks whether individual attention heads specialize for specific architectural feature types. Design docs exist at `docs/enhancements/per_attention_head.md` but no code has been written. This would involve:

- Per-head attention extraction (function exists: `get_per_head_attention()`)
- Per-head IoU computation against each bbox
- Head specialization analysis (which heads consistently attend to windows? portals? towers?)
- Head ranking and clustering
- Frontend visualization (head selector still needs to be added in `ControlPanel`)

---

## 11. Known Limitations and Gotchas

### Sparse Annotation Bias

The WikiChurches dataset annotates *representative* instances of each feature type, not *exhaustive* instances. This means a model that correctly attends to all round-arch windows on a facade gets penalized for attending to unannotated windows. See `docs/enhancements/sparse_annotation_bias.md` for full analysis and tiered mitigation strategies.

### IoU Is Upper-Bounded by Area Mismatch

When the attention area (e.g., top 10% = 10% of image) and the annotation area (e.g., 30% of image) differ significantly, the theoretical maximum IoU is much less than 1.0. The frontend shows a "% of theoretical max" bar to contextualize raw IoU scores.

### MPS Backend Limitations

On Apple Silicon (M4 Pro), some operations fall back to CPU:
- `torch.topk` on MPS sometimes needs `PYTORCH_ENABLE_MPS_FALLBACK=1`
- `bfloat16` has limited MPS support — the codebase auto-detects and uses `float32` on MPS

---

## 12. Documentation Map

| Document | Location | Purpose |
|----------|----------|---------|
| README.md | root | Quick start, model table, project structure |
| CLAUDE.md | root | AI assistant instructions, workflow rules |
| One-Pager Pitch | `docs/core/one_pager_pitch.md` | ISY5004 project approval pitch |
| Project Proposal | `docs/core/project_proposal.md` | Full research design |
| Implementation Plan | `docs/core/implementation_plan.md` | Phase tracking, architecture |
| Attention Methods | `docs/research/attention_methods.md` | Background on CLS, rollout, Grad-CAM |
| Novelty Check | `docs/research/claude_novelty_check.md` | Literature novelty assessment |
| API Reference | `docs/reference/api_reference.md` | API routes and schemas documented |
| Metrics Methodology | `docs/reference/metrics_methodology.md` | IoU, coverage with worked examples |
| Heatmap Implementation | `docs/reference/attention_heatmap_implementation.md` | Technical heatmap design |
| Fine-Tuning Methods | `docs/enhancements/fine_tuning_methods.md` | Strategy comparison design |
| Per-Head Analysis | `docs/enhancements/per_attention_head.md` | Q3 design proposal |
| Sparse Annotation Bias | `docs/enhancements/sparse_annotation_bias.md` | Known limitation analysis |

---

## 13. Potential Next Steps

Here's a rough priority ordering of what could move the project forward, organized by which research question each item advances. These are suggestions, not assignments — we should discuss and decide together.

### Q2: Fine-Tuning Effects (Highest Priority — Partially Answered)

1. **Proper fine-tuning runs with final hyperparameters** — Train all 5 SSL models with all 3 strategies (Linear Probe, LoRA, Full), proper epoch counts, and save checkpoints. Current results are from a single preliminary 1-epoch run.

2. **Fix the frozen vs fine-tuned comparison pipeline** (#9ct.1) — The comparison view (backend endpoint + frontend component) exists but is broken. Once checkpoints exist, this needs to work end-to-end.

3. **Full delta-IoU analysis across all models and strategies** — Run `analyze_delta_iou.py` with proper checkpoints to get statistically rigorous results. The analysis script and statistical framework are ready.

4. **Attention shift visualization** (#474) — Visual diff showing how attention maps change after fine-tuning. Depends on checkpoints existing.

5. **Fine-tuning training loop tests** (#4up) — Add test coverage for the training pipeline (currently the only untested major subsystem).

### Q3: Head Specialization (Medium Priority — Not Yet Started)

6. **Per-head attention analysis** — Implement per-head IoU computation against each bbox, head specialization ranking, and frontend visualization. Design docs are ready at `docs/enhancements/per_attention_head.md`. The extraction function (`get_per_head_attention()`) already exists; what's missing is the per-head metrics pipeline and UI.

### Engineering (Lower Priority)

7. **Documentation reconciliation** (#9ct.7) — Sync docs with current code state.
8. **Cache service cleanup** (#9ct.8) — Remove unused abstraction layer.
9. **Performance optimization** (#9ct.5) — Remove unnecessary image loads in fine-tuning data prep.

---

## 14. Quick Reference Commands

```bash
# Development
./dev.sh                          # Start full app (backend + frontend)
pytest                            # Run all tests
bd ready                          # See available issues

# Model exploration (Python REPL)
from ssl_attention.models import get_model
model = get_model("dinov2")       # Loads to MPS/CUDA/CPU
output = model(model.preprocess([image]))  # → ModelOutput

# Fine-tuning
python experiments/scripts/fine_tune_models.py --model dinov2 --epochs 10
python experiments/scripts/fine_tune_models.py --model dinov2 --lora --epochs 10
python experiments/scripts/fine_tune_models.py --model dinov2 --freeze-backbone --epochs 10

# Analysis
python experiments/scripts/analyze_delta_iou.py --models dinov2 siglip
```

---

Questions? Look through the docs, explore the codebase, or just ask. Welcome aboard!

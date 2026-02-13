# SSL WikiChurches Teammate Onboarding Guide


## 1. Why this project exists

This project is organized around three research questions:

1. **Do self-supervised vision models attend to the same architectural features that human experts use?**
2. **Does fine-tuning shift attention toward expert-identified features, and does strategy (Linear Probe vs LoRA vs Full) matter?**
3. **Do individual attention heads specialize for different architectural features?**

The benchmark uses WikiChurches expert annotations (139 images, 631 bounding boxes) and compares multiple model families (DINOv2, DINOv3, MAE, CLIP, SigLIP 2, ResNet-50) with attention alignment metrics (primarily IoU + coverage).

For Q2 fine-tuning, training should rely on the broader WikiChurches style-labeled dataset (9,502 images), with bbox-annotated evaluation images excluded to avoid leakage; the 139 annotated images are primarily for attention-alignment evaluation.

## 2. Current status snapshot (as of February 12, 2026)

### What is working now

- Core Python library under `src/ssl_attention/` is implemented for:
  - Model wrappers and registry
  - Attention extraction (`cls`, `rollout`, `mean`, `gradcam`)
  - Dataset/annotation parsing
  - Metrics (IoU, coverage, pointing game, baselines, statistics)
  - Evaluation utilities (linear probe + fine-tuning module)
- Full app stack exists under `app/`:
  - FastAPI backend routes/services
  - React frontend with gallery/detail/compare/dashboard views
  - Precompute scripts for attention/features/heatmaps/metrics
- Test suite is healthy:
  - `uv run pytest` passed: **273 passed**.
- Dataset is present locally:
  - `dataset/images` contains 9,502 images.
- Precomputed artifacts exist:
  - `outputs/cache/attention_viz.h5`
  - `outputs/cache/features.h5`
  - `outputs/cache/metrics.db`
  - `outputs/cache/heatmaps/...`

### Important current gaps

- `outputs/checkpoints/` is empty (no fine-tuned checkpoints currently present).
- `outputs/results/fine_tuning_results.json` references a checkpoint path that is not present locally.
- Frozen vs fine-tuned product path is only partially integrated:
  - Frontend component `app/frontend/src/components/comparison/FrozenVsFinetuned.tsx` is hardcoded as unavailable.
  - Backend compare endpoint still describes fine-tuned availability as placeholder.
  - `generate_heatmap_images.py` and `generate_metrics_cache.py` currently operate on canonical models, not `_finetuned` model keys.

## 3. Repository map (where to work)

### Core ML library

- `src/ssl_attention/models/`: model wrappers + registry
- `src/ssl_attention/attention/`: attention extraction utilities
- `src/ssl_attention/data/`: datasets + annotations
- `src/ssl_attention/metrics/`: metrics + statistics
- `src/ssl_attention/evaluation/`: linear probe + fine-tuning
- `src/ssl_attention/cache/`: HDF5 cache manager

### App stack

- Backend: `app/backend/`
  - routers: API contracts
  - services: data/cache query logic
  - validators/config/schemas
- Frontend: `app/frontend/src/`
  - pages: `Home`, `ImageDetail`, `Compare`, `Dashboard`
  - `store/viewStore.ts`: global view state
  - `api/client.ts`: backend client interface
  - comparison + attention + metrics components
- Precompute: `app/precompute/`
  - attention cache
  - feature cache
  - heatmap image generation
  - metrics cache generation

### Experiments

- `experiments/scripts/fine_tune_models.py`
- `experiments/scripts/analyze_delta_iou.py`

## 4. Suggested onboarding flow (first 1-2 days)

1. Read these docs first:
   - `README.md`
   - `docs/core/project_proposal.md`
   - `docs/core/implementation_plan.md`
   - `docs/reference/metrics_methodology.md`
2. Run the project health checks:
   - `uv sync`
   - `uv run pytest`
   - `./dev.sh`
3. Open and validate main UX paths:
   - Gallery (`/`)
   - Image detail (`/image/:id`)
   - Compare (`/compare`)
   - Dashboard (`/dashboard`)
4. Inspect precomputed caches and backend health endpoint:
   - `GET /health`
   - Confirm local cache and metrics DB access.

## 5. What has already been done (detailed)

### Data + annotation foundations

- Bounding box parsing, clamping, and mask generation implemented.
- Annotated subset and full dataset classes implemented.
- Style mapping and label filtering implemented.

### Modeling + attention extraction

- ViT wrappers and ResNet baseline wrapper implemented.
- Registers-aware extraction for DINOv2/DINOv3 implemented.
- SigLIP no-CLS handling implemented (mean-style path).
- CLS/rollout/mean/Grad-CAM extraction paths implemented.

### Evaluation + metrics

- IoU/coverage core metrics implemented.
- Per-bbox IoU and feature-type aggregation implemented.
- Pointing game and top-k pointing implemented.
- Baseline generators implemented (random/center/saliency variants).
- Statistical tooling implemented (paired tests, bootstrap CI, effect sizes, multiple-comparison correction).

### Fine-tuning + analysis scripts

- Fine-tuning module supports:
  - full fine-tuning
  - head-only (`freeze_backbone`)
  - LoRA (`peft`)
- Training script and delta-IoU analysis script implemented.

### Productization

- Backend APIs for images/attention/metrics/comparison/similarity implemented.
- Frontend supports:
  - Model/layer/method controls
  - On-image bbox selection
  - Similarity heatmap overlays
  - Leaderboard + feature breakdown views
- End-to-end precompute pipeline exists and is operational for frozen models.

## 6. Recommended next-step roadmap

### Priority 1: Complete frozen vs fine-tuned product path

Goal: make fine-tuned comparisons first-class in API + precompute + UI.

- Add reliable checkpoint lifecycle:
  - define source-of-truth for available checkpoints
  - expose this in backend status endpoint.
- Extend precompute flow to generate heatmaps/metrics for fine-tuned variants.
- Update compare APIs and frontend to use real availability checks (not hardcoded placeholder).
- Add regression tests for fine-tuned cache/heatmap/metrics query paths.

### Priority 2: Strengthen evaluation robustness

Goal: improve conclusions under sparse annotation limitations.

- Add per-bbox recall metric (complements IoU).
- Add sparse-annotation sensitivity analysis (annotation-dropout bootstrap).
- Expose these in API and optionally dashboard.

### Priority 3: Finish dashboard visualization layer

Goal: move from placeholder chart panels to actionable analytics.

- Re-enable/replace charting path currently disabled.
- Add layer-progression and style-breakdown charts with method awareness.
- Ensure consistent behavior across models with different layer counts.

### Priority 4: Operational hardening

Goal: easier collaboration + reproducibility.

- Add CI workflow to run `pytest` automatically.
- Add a small “fresh machine bootstrap” script/checklist.
- Optionally add cached artifact version metadata (dataset hash, cache build params).

## 7. Practical collaboration expectations

- Treat the following as source-of-truth for behavior:
  - backend routers/services
  - frontend API client + hooks
  - precompute scripts
- Before changing model/method behavior, check:
  - `src/ssl_attention/config.py`
  - `app/backend/validators.py`
  - frontend `viewStore` and `ControlPanel`.
- Keep the API + frontend contract aligned (method names, model aliases, layer bounds).
- Prefer adding tests with each change; current suite is fast and reliable.

## 8. High-value first ticket recommendations

1. Implement production-ready frozen-vs-fine-tuned compare flow end-to-end.
2. Add per-bbox recall metric and expose it in detail view metrics.
3. Replace dashboard chart placeholders with working visualizations.
4. Add CI automation for `pytest` and a reproducible bootstrap checklist.

---

If you are joining this repo now, the fastest path to impact is:

- own the fine-tuned comparison completion,
- then strengthen evaluation robustness,
- then improve dashboard analytics for decision-making.

# Do Self-Supervised Vision Models Learn What Experts See?

This repository evaluates whether SSL vision models attend to the same architectural features that human experts mark as diagnostically important in WikiChurches images.

The project centers on three linked questions:

1. **Q1: Attention alignment**. How well do frozen models align with expert annotations across IoU, Coverage, MSE, KL, and EMD?
2. **Q2: Fine-tuning effects**. How do Linear Probe, LoRA, and Full fine-tuning change attention alignment on the same evaluation images?
3. **Q3: Head specialization**. Which individual attention heads align best with specific architectural features, and how does that change across variants?

## Start Here

**Requirements:** Python 3.12+, [uv](https://github.com/astral-sh/uv), Node.js 18+

### 1. Install dependencies

```bash
uv sync
```

Frontend dependencies install on first `./dev.sh` run, or manually from `app/frontend`.

### 2. Choose a dataset path

#### Path A: Annotated subset for the app and evaluation

Use the Google Drive package when you want the 139 annotated images plus `building_parts.json` for the app, cache generation, and the primary attention-alignment workflows.

**[Download annotated subset (Google Drive)](https://drive.google.com/drive/folders/1fsf0k71ADeYCBAwo-dIPntUmpibaoGBr)**

Expected structure:

```text
dataset/
├── images/
│   ├── Q18785543_wd0.jpg
│   ├── Q2034923_wd0.jpg
│   └── ...
└── building_parts.json
```

#### Path B: Official WikiChurches files from Zenodo

Use the downloader when you want selective files from the official WikiChurches release, such as `churches.json`, metadata, the full image archive, or the official `building_parts.json`.

```bash
uv run python scripts/download_wikichurches.py --list
uv run python scripts/download_wikichurches.py --files churches.json image_meta.json building_parts.json
```

The official release contains 9,485 images. The interactive app evaluates the 139-image expert-annotated subset, while fine-tuning draws from the style-labeled pool in `churches.json`.

### 3. Precompute the baseline app artifacts

Generate the frozen-model attention, feature, heatmap, and metrics caches:

```bash
uv run python -m app.precompute.generate_attention_cache --models all
uv run python -m app.precompute.generate_feature_cache --models all
uv run python -m app.precompute.generate_heatmap_images --colormap viridis
uv run python -m app.precompute.generate_metrics_cache
```

Populate the primary Q3 per-head study scope on Dashboard, Image Detail, and `/q3`:

```bash
# Frozen Q3 scope
uv run python -m app.precompute.generate_attention_cache --models dinov2 dinov3 mae clip --per-head
uv run python -m app.precompute.generate_metrics_cache --models dinov2 dinov3 mae clip --per-head

# Fine-tuned Q3 scope
uv run python -m app.precompute.generate_attention_cache --finetuned --models dinov2 dinov3 mae clip --strategies lora full --per-head
uv run python -m app.precompute.generate_metrics_cache --finetuned --models dinov2 dinov3 mae clip --strategies lora full --per-head
```

Populate fine-tuned overlays, similarity features, and heatmaps for the Compare flows after checkpoints exist:

```bash
uv run python -m app.precompute.generate_attention_cache --finetuned --models all --strategies linear_probe lora full
uv run python -m app.precompute.generate_feature_cache --finetuned --models all --strategies linear_probe lora full
uv run python -m app.precompute.generate_heatmap_images --finetuned --models all --strategies linear_probe lora full
uv run python -m app.precompute.generate_metrics_cache --finetuned --models dinov2 dinov3 mae clip siglip siglip2 --strategies linear_probe lora full
```

The full command surface, flags, and utility scripts are documented in [docs/reference/cli_reference.md](docs/reference/cli_reference.md).

### 4. Run the app

#### One-command local development

```bash
./dev.sh
```

This starts:

- backend at `http://127.0.0.1:8000`
- frontend at `http://127.0.0.1:5173`

#### Manual startup

Backend:

```bash
uv run uvicorn app.backend.main:app --reload --port 8000
```

Frontend:

```bash
cd app/frontend
npm install
npm run dev
```

#### Docker

Standard compose run:

```bash
docker compose up
```

This exposes:

- backend at `http://localhost:8000`
- frontend at `http://localhost:3000`

Hot-reload backend profile:

```bash
docker compose --profile dev up backend-dev frontend
```

## App Routes

The current UI surface is organized around these routes:

| Route | Primary role |
|------|--------------|
| `/` | Gallery for browsing annotated images and styles |
| `/image/:imageId` | Image Detail with overlay inspection, metrics progression, and Q3 drill-down |
| `/compare` | Model vs Model and variant comparison workflows on one image |
| `/dashboard` | Q1 overview analysis plus the main Q3 discovery surface |
| `/q2` | Strategy-aware fine-tuning summary sourced from the active experiment |
| `/q3` | Advanced side-by-side Q3 workspace for aligned model comparisons |

## Model Surface

All transformer models use ViT-Base backbones. `resnet50` is the supervised CNN baseline.

| Model key | Backbone | Default method | Other supported methods |
|-----------|----------|----------------|-------------------------|
| `dinov2` | ViT-B/14 | `cls` | `rollout` |
| `dinov3` | ViT-B/16 | `cls` | `rollout` |
| `mae` | ViT-B/16 | `cls` | `rollout` |
| `clip` | ViT-B/16 | `cls` | `rollout` |
| `siglip` | ViT-B/16 | `mean` | — |
| `siglip2` | ViT-B/16 | `mean` | — |
| `resnet50` | CNN | `gradcam` | — |

Fine-tuning is supported for `dinov2`, `dinov3`, `mae`, `clip`, `siglip`, and `siglip2`.

## Fine-Tuning Workflow

Fine-tuning uses the style-labeled pool derived from `churches.json`. The 139 bbox-annotated images stay out of the primary train/validation split and remain the evaluation pool for attention alignment.

### 1. Train one experiment batch

```bash
EXPERIMENT_ID=fine_tuning_primary_20260327

uv run python experiments/scripts/fine_tune_models.py --all --freeze-backbone --epochs 3 --experiment-id "$EXPERIMENT_ID"
uv run python experiments/scripts/fine_tune_models.py --all --lora --epochs 3 --experiment-id "$EXPERIMENT_ID"
uv run python experiments/scripts/fine_tune_models.py --all --epochs 3 --experiment-id "$EXPERIMENT_ID"
```

Primary checkpoints are written to:

- `outputs/checkpoints/<experiment_id>/`
- `outputs/results/experiments/<experiment_id>/fine_tuning_results.json`
- `outputs/results/experiments/<experiment_id>/run_matrix.json`
- `outputs/results/experiments/<experiment_id>/manifests/`
- `outputs/results/experiments/<experiment_id>/splits/`

### 2. Build the Q2 analysis artifact

```bash
uv run python experiments/scripts/analyze_q2_metrics.py \
  --experiment-id "$EXPERIMENT_ID" \
  --models clip dinov2 dinov3 mae siglip siglip2 \
  --strategies linear_probe lora full
```

This writes the canonical active-experiment artifact:

- `outputs/results/experiments/<experiment_id>/q2_metrics_analysis.json`

The compatibility export remains available at:

- `outputs/results/experiments/<experiment_id>/q2_delta_iou_analysis.json`

`outputs/results/active_experiment.json` selects which experiment batch the app, figure scripts, and reporting utilities read by default.

### 3. Refresh Q1 baseline reports

```bash
uv run python experiments/scripts/analyze_q1_continuous_baselines.py
```

This writes:

- `outputs/results/q1_continuous_baseline_comparison.json`
- `outputs/results/q1_continuous_baseline_summary.md`

See [docs/reference/fine_tuning_run_matrix.md](docs/reference/fine_tuning_run_matrix.md) for the full artifact contract and active-experiment layout.

## Developer Commands

| Area | Command |
|------|---------|
| Python tests | `uv run pytest` |
| Python lint | `uv run ruff check .` |
| Python typing | `uv run mypy` |
| Frontend install | `cd app/frontend && npm install` |
| Frontend lint | `cd app/frontend && npm run lint` |
| Frontend build | `cd app/frontend && npm run build` |
| Frontend preview | `cd app/frontend && npm run preview` |
| Frontend E2E | `cd app/frontend && npm run test:e2e` |
| Notebook | `uv run jupyter lab notebooks/01_data_exploration.ipynb` |

## Reporting Utilities

The reporting pipeline reads the active experiment and cache outputs to generate figure and slide assets:

- `uv run python experiments/scripts/generate_run_matrix_figures.py`
- `uv run python experiments/scripts/generate_slide_images.py`
- `cd experiments/scripts && npm install && node create_presentation.js`

Primary generated locations:

- `outputs/figures/`
- `outputs/slides/`

## Documentation Map

| Document | Use it for |
|----------|------------|
| [docs/README.md](docs/README.md) | Documentation index and navigation |
| [docs/user_guide.md](docs/user_guide.md) | Product walkthroughs for Gallery, Compare, Dashboard, Q2, Image Detail, and Q3 |
| [docs/reference/cli_reference.md](docs/reference/cli_reference.md) | Complete command and flag reference |
| [docs/reference/api_reference.md](docs/reference/api_reference.md) | Backend routes, query parameters, and response contracts |
| [docs/reference/fine_tuning_run_matrix.md](docs/reference/fine_tuning_run_matrix.md) | Experiment-scoped artifact layout and active-experiment workflow |

## Project Layout

```text
ssl_wikichurches/
├── app/
│   ├── backend/          # FastAPI backend
│   ├── frontend/         # React + Vite frontend
│   └── precompute/       # Cache generation scripts
├── dataset/              # Local WikiChurches data
├── docs/                 # Current project documentation
├── experiments/          # Fine-tuning, analysis, and reporting scripts
├── outputs/              # Generated caches, checkpoints, figures, and results
├── scripts/              # Dataset and utility scripts
├── src/ssl_attention/    # Core library code
└── tests/                # Pytest test suite
```

## References

- Barz & Denzler (2021). [WikiChurches](https://arxiv.org/abs/2108.06959)
- Oquab et al. (2023). [DINOv2](https://arxiv.org/abs/2304.07193)
- Simeoni et al. (2025). [DINOv3](https://arxiv.org/abs/2508.10104)
- Zhang et al. (2018). [Top-Down Neural Attention by Excitation Backprop](https://link.springer.com/article/10.1007/s11263-017-1059-x)

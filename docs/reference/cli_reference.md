# CLI Reference

This page documents the current command surface for the repository. It uses the parser definitions in the Python scripts, the frontend `package.json`, `docker-compose.yml`, and `dev.sh` as the source of truth.

Run Python commands from the repository root unless a section says otherwise.

## Runtime and Setup

### Environment setup

```bash
uv sync
```

Installs the Python environment defined by `pyproject.toml` and `uv.lock`.

### One-command local app startup

```bash
./dev.sh
```

Purpose:

- starts the FastAPI backend on `127.0.0.1:8000`
- starts the Vite frontend on `127.0.0.1:5173`
- installs frontend dependencies automatically if `app/frontend/node_modules` is missing
- cleans up any existing processes on ports `8000` and `5173`

Classification: primary workflow

### Manual backend startup

```bash
uv run uvicorn app.backend.main:app --reload --port 8000
```

Purpose:

- runs the FastAPI backend directly

Classification: primary workflow

### Frontend scripts

Run these commands from `app/frontend/`:

| Command | Purpose | Classification |
|---------|---------|----------------|
| `npm run dev` | Start the Vite development server | primary workflow |
| `npm run build` | Run `tsc -b` and build the production bundle | primary workflow |
| `npm run lint` | Run ESLint | primary workflow |
| `npm run preview` | Preview the built frontend locally | utility |
| `npm run test:e2e` | Run Playwright tests | primary workflow |

### Docker Compose

Standard app stack:

```bash
docker compose up
```

Purpose:

- starts `backend` on port `8000`
- starts `frontend` on port `3000`

Classification: primary workflow

Dev backend profile:

```bash
docker compose --profile dev up backend-dev frontend
```

Purpose:

- runs the hot-reload backend profile from `docker-compose.yml`
- keeps the frontend on port `3000`

Classification: utility

## Dataset and Utility Scripts

### `scripts/download_wikichurches.py`

Command:

```bash
uv run python scripts/download_wikichurches.py [options]
```

Purpose:

- download files from the official WikiChurches Zenodo record
- resume interrupted downloads when the server supports range requests
- verify downloaded files against expected sizes

Primary outputs:

- files downloaded into the chosen output directory

Canonical examples:

```bash
uv run python scripts/download_wikichurches.py --list
uv run python scripts/download_wikichurches.py --files churches.json image_meta.json building_parts.json
uv run python scripts/download_wikichurches.py --exclude images.zip models.zip -o dataset
uv run python scripts/download_wikichurches.py --verify-only -o dataset
```

Options:

| Flag | Meaning |
|------|---------|
| `-o`, `--output <path>` | Output directory. Defaults to the current directory. |
| `--files <names...>` | Download only the named files from the official file set. |
| `--exclude <names...>` | Exclude named files from the download selection. |
| `--verify-only` | Verify existing files without downloading. |
| `--list` | Print the available official files and exit. |

Classification: primary workflow

### `app.precompute.benchmark_metrics_iteration`

Command:

```bash
uv run python -m app.precompute.benchmark_metrics_iteration [options]
```

Purpose:

- benchmark metadata-only dataset traversal against legacy image-decoding iteration

Primary outputs:

- console report or JSON report only

Canonical examples:

```bash
uv run python -m app.precompute.benchmark_metrics_iteration
uv run python -m app.precompute.benchmark_metrics_iteration --runs 5 --json
```

Options:

| Flag | Meaning |
|------|---------|
| `--dataset-root <path>` | Dataset root to benchmark. Defaults to the configured dataset path. |
| `--runs <int>` | Number of timing runs per traversal mode. Defaults to `3`. |
| `--json` | Emit a JSON payload instead of the human-readable table. |

Classification: utility

## Precompute Scripts

### `app.precompute.generate_attention_cache`

Command:

```bash
uv run python -m app.precompute.generate_attention_cache [options]
```

Purpose:

- generate cached raw attention artifacts for frozen and fine-tuned models
- optionally populate per-head Q3 attention variants

Primary outputs:

- `outputs/cache/attention_viz.h5` by default

Canonical examples:

```bash
uv run python -m app.precompute.generate_attention_cache --models all
uv run python -m app.precompute.generate_attention_cache --models dinov2 --layers 11 --methods cls
uv run python -m app.precompute.generate_attention_cache --models dinov2 dinov3 mae clip --per-head
uv run python -m app.precompute.generate_attention_cache --finetuned --models dinov2 dinov3 mae clip --strategies lora full --per-head
```

Options:

| Flag | Meaning |
|------|---------|
| `--models <names...>` | Models to process. Defaults to `all`. |
| `--layers <ints...>` | Specific layer indices to process. Defaults to every available layer. |
| `--methods <names...>` | Attention methods to compute. Choices: `all`, `cls`, `rollout`, `mean`, `gradcam`. |
| `--cache-path <path>` | Target HDF5 cache path. Defaults to `outputs/cache/attention_viz.h5`. |
| `--no-skip` | Recompute entries even when cached data already exists. |
| `--device <name>` | Override automatic device selection. |
| `--finetuned` | Resolve fine-tuned checkpoints and write fine-tuned cache keys. |
| `--checkpoint-dir <path>` | Directory containing fine-tuned checkpoints. Defaults to `outputs/checkpoints`. |
| `--strategies <names...>` | Fine-tuning strategies in `--finetuned` mode. Choices: `auto`, `all`, `linear_probe`, `lora`, `full`. |
| `--per-head` | Also populate per-head attention variants for supported Q3 methods. |

Classification: primary workflow

### `app.precompute.generate_feature_cache`

Command:

```bash
uv run python -m app.precompute.generate_feature_cache [options]
```

Purpose:

- generate cached patch-level feature tensors used by similarity workflows

Primary outputs:

- `outputs/cache/features.h5` by default

Canonical examples:

```bash
uv run python -m app.precompute.generate_feature_cache --models all
uv run python -m app.precompute.generate_feature_cache --models dinov2 dinov3 mae clip siglip siglip2
uv run python -m app.precompute.generate_feature_cache --finetuned --models dinov2 dinov3 mae clip siglip siglip2 --strategies linear_probe lora full
```

Options:

| Flag | Meaning |
|------|---------|
| `--models <names...>` | Models to process. Defaults to `all`. |
| `--layers <ints...>` | Specific layer indices to process. Defaults to every available layer. |
| `--cache-path <path>` | Target feature cache path. Defaults to `outputs/cache/features.h5`. |
| `--no-skip` | Recompute entries even when feature data already exists. |
| `--device <name>` | Override automatic device selection. |
| `--finetuned` | Resolve fine-tuned checkpoints and write fine-tuned feature keys. |
| `--checkpoint-dir <path>` | Directory containing fine-tuned checkpoints. Defaults to `outputs/checkpoints`. |
| `--strategies <names...>` | Fine-tuning strategies in `--finetuned` mode. Choices: `auto`, `all`, `linear_probe`, `lora`, `full`. |

Classification: primary workflow

### `app.precompute.generate_heatmap_images`

Command:

```bash
uv run python -m app.precompute.generate_heatmap_images [options]
```

Purpose:

- render pure heatmaps, overlays, and bbox overlays from the attention cache

Primary outputs:

- `outputs/cache/heatmaps/` by default

Canonical examples:

```bash
uv run python -m app.precompute.generate_heatmap_images --colormap viridis
uv run python -m app.precompute.generate_heatmap_images --models dinov2 clip --layers 11 --methods cls
uv run python -m app.precompute.generate_heatmap_images --finetuned --models dinov2 dinov3 mae clip siglip siglip2 --strategies linear_probe lora full
```

Options:

| Flag | Meaning |
|------|---------|
| `--models <names...>` | Models to process. Defaults to `all`. |
| `--layers <ints...>` | Specific layer indices to process. Defaults to every available layer. |
| `--methods <names...>` | Attention methods to render. Choices: `all`, `cls`, `rollout`, `mean`, `gradcam`. |
| `--attention-cache <path>` | Source attention cache. Defaults to `outputs/cache/attention_viz.h5`. |
| `--output-dir <path>` | Target directory for rendered PNGs. Defaults to `outputs/cache/heatmaps`. |
| `--colormap <name>` | Colormap choice. Choices: `viridis`, `plasma`, `inferno`, `magma`, `hot`, `jet`. |
| `--alpha <float>` | Overlay transparency. Defaults to `0.5`. |
| `--no-skip` | Re-render existing files instead of skipping them. |
| `--finetuned` | Render heatmaps for fine-tuned cache keys. |
| `--strategies <names...>` | Fine-tuning strategies in `--finetuned` mode. Choices: `auto`, `all`, `linear_probe`, `lora`, `full`. |

Classification: primary workflow

### `app.precompute.generate_metrics_cache`

Command:

```bash
uv run python -m app.precompute.generate_metrics_cache [options]
```

Purpose:

- compute aggregate image metrics from cached attention
- populate the Q1/Q2 metrics tables and optional Q3 per-head tables

Primary outputs:

- `outputs/cache/metrics.db`
- summary exports consumed by the app services

Canonical examples:

```bash
uv run python -m app.precompute.generate_metrics_cache
uv run python -m app.precompute.generate_metrics_cache --models dinov2 clip --layers 11 --methods cls rollout
uv run python -m app.precompute.generate_metrics_cache --percentiles 90 80 70 60 50
uv run python -m app.precompute.generate_metrics_cache --models dinov2 dinov3 mae clip --per-head
uv run python -m app.precompute.generate_metrics_cache --finetuned --models dinov2 dinov3 mae clip siglip siglip2 --strategies linear_probe lora full
uv run python -m app.precompute.generate_metrics_cache --finetuned --models dinov2 dinov3 mae clip --strategies lora full --per-head
uv run python -m app.precompute.generate_metrics_cache --finetuned --models dinov2 dinov3 mae clip --strategies linear_probe --per-head
```

Options:

| Flag | Meaning |
|------|---------|
| `--models <names...>` | Models to process. Defaults to `all`. |
| `--layers <ints...>` | Specific layer indices to process. Defaults to every available layer. |
| `--attention-cache <path>` | Source attention cache. Defaults to `outputs/cache/attention_viz.h5`. |
| `--db-path <path>` | Target SQLite database path. Defaults to `outputs/cache/metrics.db`. |
| `--percentiles <ints...>` | IoU percentile thresholds to compute. Defaults to the configured percentile set. |
| `--methods <names...>` | Explicit attention methods to process. Defaults to all supported methods per model. |
| `--no-skip` | Recompute rows even when database entries already exist. |
| `--finetuned` | Compute metrics for fine-tuned cache keys. |
| `--strategies <names...>` | Fine-tuning strategies in `--finetuned` mode. Defaults to all canonical strategies. |
| `--per-head` | Also compute Q3 per-head metrics from per-head attention variants. |

Classification: primary workflow

Q3 per-head metrics require matching per-head attention cache entries. Run
`generate_attention_cache --per-head` for the same model, strategy, and variant
scope before this command.

## Fine-Tuning and Analysis Scripts

### `experiments/scripts/fine_tune_models.py`

Command:

```bash
uv run python experiments/scripts/fine_tune_models.py [options]
```

Purpose:

- fine-tune one model or the full model set for style classification
- write experiment-scoped checkpoints, manifests, splits, and run-matrix entries

Primary outputs:

- `outputs/checkpoints/<experiment_id>/`
- `outputs/results/experiments/<experiment_id>/fine_tuning_results.json`
- `outputs/results/experiments/<experiment_id>/run_matrix.json`
- `outputs/results/experiments/<experiment_id>/manifests/`
- `outputs/results/experiments/<experiment_id>/splits/`

Canonical examples:

```bash
uv run python experiments/scripts/fine_tune_models.py --model dinov2 --epochs 5
uv run python experiments/scripts/fine_tune_models.py --all --freeze-backbone --epochs 3 --experiment-id fine_tuning_primary_20260327
uv run python experiments/scripts/fine_tune_models.py --all --lora --epochs 3 --experiment-id fine_tuning_primary_20260327
uv run python experiments/scripts/fine_tune_models.py --all --epochs 3 --experiment-id fine_tuning_primary_20260327
```

Options:

| Flag | Meaning |
|------|---------|
| `--model <name>` | Train one fine-tunable model. Mutually exclusive with `--all`. |
| `--all` | Train every fine-tunable model. Mutually exclusive with `--model`. |
| `--epochs <int>` | Number of epochs. Defaults to `10`. |
| `--batch-size <int>` | Training batch size. Defaults to `16`. |
| `--lr-backbone <float>` | Backbone learning rate. Defaults to `1e-5`. |
| `--lr-head <float>` | Classification-head learning rate. Defaults to `1e-3`. |
| `--freeze-backbone` | Train only the head. This is the `linear_probe` condition. |
| `--lora` | Enable LoRA adapters on attention layers. |
| `--lora-rank <int>` | LoRA rank. Defaults to `8`. |
| `--lora-alpha <int>` | LoRA alpha. Defaults to `32`. |
| `--lora-dropout <float>` | LoRA dropout. Defaults to `0.1`. |
| `--seed <int>` | Random seed. Defaults to `42`. |
| `--val-split <float>` | Validation split fraction. Defaults to `0.2`. |
| `--include-annotated-eval` | Use the annotated pool as the local train/validation source when that is the only dataset available. |
| `--val-on-annotated-eval` | Exploratory mode that selects checkpoints on the annotated evaluation pool. |
| `--val-on-random-split` | Deprecated compatibility flag. Random stratified validation is already the primary default path. |
| `--experiment-id <id>` | Explicit experiment batch identifier. Defaults to a timestamped batch id. |

Classification: primary workflow

### `experiments/scripts/analyze_q2_metrics.py`

Command:

```bash
uv run python experiments/scripts/analyze_q2_metrics.py [options]
```

Purpose:

- analyze frozen-versus-fine-tuned attention shifts across IoU, Coverage, MSE, KL, and EMD
- write the active-experiment Q2 analysis artifact and compatibility export

Primary outputs:

- `outputs/results/experiments/<experiment_id>/q2_metrics_analysis.json`
- `outputs/results/experiments/<experiment_id>/q2_delta_iou_analysis.json`
- updates to `outputs/results/active_experiment.json`

Canonical examples:

```bash
uv run python experiments/scripts/analyze_q2_metrics.py --experiment-id fine_tuning_primary_20260327
uv run python experiments/scripts/analyze_q2_metrics.py --experiment-id fine_tuning_primary_20260327 --models clip dinov2 dinov3 mae siglip siglip2 --strategies linear_probe lora full
uv run python experiments/scripts/analyze_q2_metrics.py --experiment-id fine_tuning_primary_20260327 --percentile 90 --layer 11
```

Options:

| Flag | Meaning |
|------|---------|
| `--models <names...>` | Specific models to analyze. Defaults to all fine-tunable models with checkpoints. |
| `--strategies <names...>` | Strategies to analyze. Choices: `linear_probe`, `lora`, `full`. |
| `--percentile <int>` | Single IoU percentile to analyze. Defaults to the full supported percentile set. |
| `--layer <int>` | Layer index to analyze. Defaults to `-1`, which maps to the last layer. |
| `--include-resnet` | Include ResNet-50. The script marks this as unsupported. |
| `--experiment-id <id>` | Experiment batch to analyze. Defaults to the active experiment when available. |
| `--include-exploratory` | Include exploratory runs such as annotated-eval checkpoint selection. |
| `--output <path>` | Custom path for `q2_metrics_analysis.json`. |

Classification: primary workflow

### `experiments/scripts/analyze_q1_continuous_baselines.py`

Command:

```bash
uv run python experiments/scripts/analyze_q1_continuous_baselines.py [options]
```

Purpose:

- compare frozen-model continuous metrics against the documented baseline references

Primary outputs:

- `outputs/results/q1_continuous_baseline_comparison.json`
- `outputs/results/q1_continuous_baseline_summary.md`

Canonical examples:

```bash
uv run python experiments/scripts/analyze_q1_continuous_baselines.py
uv run python experiments/scripts/analyze_q1_continuous_baselines.py --output-dir outputs/results --json-name q1_continuous_baseline_comparison.json --markdown-name q1_continuous_baseline_summary.md
```

Options:

| Flag | Meaning |
|------|---------|
| `--output-dir <path>` | Directory for the generated artifacts. Defaults to `outputs/results`. |
| `--json-name <name>` | JSON filename. |
| `--markdown-name <name>` | Markdown summary filename. |

Classification: primary workflow

### `experiments/scripts/analyze_delta_iou.py`

Command:

```bash
uv run python experiments/scripts/analyze_delta_iou.py [options]
```

Purpose:

- compatibility entrypoint that forwards to `analyze_q2_metrics.py`

Primary outputs:

- same outputs as `analyze_q2_metrics.py`

Canonical usage:

```bash
uv run python experiments/scripts/analyze_delta_iou.py --experiment-id fine_tuning_primary_20260327
```

Options:

- accepts the same CLI surface as `experiments/scripts/analyze_q2_metrics.py`

Classification: compatibility path

### `experiments/scripts/analyze_style_breakdown.py`

Command:

```bash
uv run python experiments/scripts/analyze_style_breakdown.py [options]
```

Purpose:

- summarize Q2 fine-tuned-minus-frozen metric deltas by architectural style from the active experiment
- generate the per-style Q2 figure used in research and report drafts

Primary outputs:

- `outputs/results/experiments/<experiment_id>/style_breakdown.json`
- `outputs/results/experiments/<experiment_id>/style_breakdown.png`

Canonical examples:

```bash
uv run python experiments/scripts/analyze_style_breakdown.py --experiment-id fine_tuning_primary_20260327 --strategy full
uv run python experiments/scripts/analyze_style_breakdown.py --experiment-id fine_tuning_primary_20260327 --strategy lora --metric iou --percentile 90
```

Options:

| Flag | Meaning |
|------|---------|
| `--experiment-id <id>` | Experiment batch to analyze. Defaults to the active experiment. |
| `--strategy <name>` | Strategy to report. Choices: `linear_probe`, `lora`, `full`. Defaults to `full`. |
| `--metric <name>` | Metric to break down by style. Choices: `iou`, `coverage`, `mse`, `kl`, `emd`. Defaults to `iou`. |
| `--percentile <int>` | IoU percentile threshold. Ignored for non-IoU metrics. Defaults to `90`. |
| `--output-json <path>` | Custom JSON output path. Defaults to `<experiment_dir>/style_breakdown.json`. |
| `--output-figure <path>` | Custom figure output path. Defaults to `<experiment_dir>/style_breakdown.png`. |

Classification: analysis workflow

### `experiments/scripts/analyze_model_correlation.py`

Command:

```bash
uv run python experiments/scripts/analyze_model_correlation.py [options]
```

Purpose:

- compare per-image Q2 Δ IoU vectors across model families
- generate scatter and heatmap artifacts for the shared-easy-images / complementarity analysis

Primary outputs:

- `outputs/results/experiments/<experiment_id>/model_correlation.json`
- `outputs/results/experiments/<experiment_id>/model_correlation_scatter.png`
- `outputs/results/experiments/<experiment_id>/model_correlation_heatmap.png`

Canonical examples:

```bash
uv run python experiments/scripts/analyze_model_correlation.py --experiment-id fine_tuning_primary_20260327 --strategy full
uv run python experiments/scripts/analyze_model_correlation.py --experiment-id fine_tuning_primary_20260327 --strategy lora --percentile 90 --layer 11
```

Options:

| Flag | Meaning |
|------|---------|
| `--experiment-id <id>` | Experiment batch to analyze. Defaults to the active experiment. |
| `--strategy <name>` | Strategy for Δ IoU vectors. Choices: `linear_probe`, `lora`, `full`. Defaults to `full`. |
| `--percentile <int>` | IoU percentile threshold. Defaults to `90`. |
| `--layer <int>` | Layer for frozen IoU lookup in `metrics.db`. Defaults to `11`. |
| `--output-json <path>` | Custom JSON output path. Defaults to `<experiment_dir>/model_correlation.json`. |
| `--output-scatter <path>` | Custom scatter figure path. Defaults to `<experiment_dir>/model_correlation_scatter.png`. |
| `--output-heatmap <path>` | Custom heatmap figure path. Defaults to `<experiment_dir>/model_correlation_heatmap.png`. |

Classification: analysis workflow

### `experiments/scripts/analyze_feature_delta_iou.py`

Command:

```bash
uv run python experiments/scripts/analyze_feature_delta_iou.py [options]
```

Purpose:

- compute per-feature Q2 Δ IoU for one model/strategy slice
- optionally restrict the analysis to one architectural style, such as the MAE Renaissance investigation

Primary outputs:

- JSON summary at the requested path or an experiment-scoped default
- optional figure at the requested path or an experiment-scoped default

Canonical examples:

```bash
uv run python experiments/scripts/analyze_feature_delta_iou.py --experiment-id fine_tuning_primary_20260327 --model mae --strategy full --style Renaissance
uv run python experiments/scripts/analyze_feature_delta_iou.py --experiment-id fine_tuning_primary_20260327 --model mae --strategy lora --style Renaissance --min-boxes 2
```

Options:

| Flag | Meaning |
|------|---------|
| `--model <name>` | Model name. Defaults to `mae`. |
| `--strategy <name>` | Fine-tuning strategy. Choices: `full`, `lora`, `linear_probe`. Defaults to `full`. |
| `--experiment-id <id>` | Experiment batch to analyze. Defaults to the active experiment. |
| `--layer <int>` | Attention layer. Defaults to `11`. |
| `--percentile <int>` | IoU percentile threshold. Defaults to `90`. |
| `--style <name>` | Optional architectural style restriction. Choices are the configured style names. |
| `--min-boxes <int>` | Minimum total bbox count for a feature to appear in output. Defaults to `2`. |
| `--output-json <path>` | Custom JSON output path. |
| `--output-figure <path>` | Custom figure output path. |

Classification: analysis workflow

## Reporting and Presentation Scripts

### `experiments/scripts/generate_run_matrix_figures.py`

Command:

```bash
uv run python experiments/scripts/generate_run_matrix_figures.py
```

Purpose:

- generate the publication-style Q2 figure set from the active experiment

Primary outputs:

- figures under `outputs/figures/`
- commentary at `outputs/figures/commentary.txt`

Inputs:

- active experiment `q2_metrics_path`
- active experiment `run_matrix_path`

CLI flags:

- none

Classification: primary workflow

The script reads the active experiment pointer and writes the report-facing Q2
figure set used by the Markdown report, LaTeX report, slide outline, and video
plan.

### `experiments/scripts/generate_slide_images.py`

Command:

```bash
uv run python experiments/scripts/generate_slide_images.py
```

Purpose:

- generate slide-ready PNG assets from cached heatmaps, metrics summaries, and the active experiment

Primary outputs:

- PNG assets under `outputs/slides/`

Inputs:

- `outputs/cache/heatmaps/`
- `outputs/cache/metrics_summary.json`
- active experiment `q2_metrics_path`
- `outputs/figures/` for figure reuse

CLI flags:

- none

Classification: primary workflow

The generated slide PNGs are intermediate assets for the presentation deck.

### `experiments/scripts/create_presentation.js`

Command:

```bash
cd experiments/scripts
npm install
node create_presentation.js
```

Purpose:

- assemble the PPTX presentation from generated slide images and figures

Primary outputs:

- `outputs/slides/presentation.pptx`

Inputs:

- `outputs/slides/`
- `outputs/figures/`

CLI flags:

- none

Classification: primary workflow

The checked-in presentation package lives under `docs/plans/video/`; generated
working assets remain under `outputs/slides/`.

## Developer Quality Commands

| Command | Purpose |
|---------|---------|
| `uv run pytest` | Run the Python test suite |
| `uv run ruff check .` | Run repo-wide Python lint checks |
| `uv run mypy` | Run repo-wide type checks |
| `cd app/frontend && npm run lint` | Run frontend lint checks |
| `cd app/frontend && npm run build` | Build the frontend bundle |
| `cd app/frontend && npm run test:e2e` | Run frontend Playwright tests |

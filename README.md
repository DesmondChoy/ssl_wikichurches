# Do Self-Supervised Vision Models Learn What Experts See?

This repository evaluates whether self-supervised vision models attend to the same architectural features that WikiChurches experts mark as diagnostically important.

The project answers three questions:

1. **Q1: Frozen attention alignment** - how well frozen models align with expert boxes across IoU, Coverage, MSE, KL, and EMD.
2. **Q2: Fine-tuning effects** - how Linear Probe, LoRA, and Full fine-tuning change that alignment.
3. **Q3: Head specialization** - which attention heads align best with specific architectural features.

## Quick Start

Requirements: Python 3.12+, [uv](https://github.com/astral-sh/uv), Node.js 18+.

```bash
uv sync
```

### Use the precomputed submission artifacts

For review, use the precomputed artifact folder instead of rerunning fine-tuning
or cache generation:

1. Download the [Google Drive artifact folder](https://drive.google.com/drive/folders/1pT8VrK6d9h-sZzAr6qhPxvNrVrRi-8Cd?usp=sharing).
2. Copy `dataset/` and `outputs/` from that folder into the repository root.

Expected local structure:

```text
ssl_wikichurches/
├── dataset/
└── outputs/
    ├── cache/
    │   ├── attention_viz.h5
    │   ├── features.h5
    │   ├── metrics.db
    │   ├── metrics_summary.json
    │   └── heatmaps/
    └── checkpoints/
        └── fine_tuning_primary_20260327/
```

On macOS or Linux, from inside the downloaded artifact folder:

```bash
rsync -av dataset outputs /path/to/ssl_wikichurches/
```

Replace `/path/to/ssl_wikichurches/` with the local path to this cloned repo.

With these artifacts in place, the app can run without rerunning
`fine_tune_models.py`, `generate_attention_cache`, `generate_feature_cache`,
`generate_heatmap_images`, or `generate_metrics_cache`.

Run the app:

```bash
./dev.sh
```

This starts the backend at `http://127.0.0.1:8000` and frontend at `http://127.0.0.1:5173`.

### Regenerate artifacts from scratch

If the precomputed artifact folder is unavailable, download one dataset path:

- **Annotated subset:** [Google Drive package](https://drive.google.com/drive/folders/1fsf0k71ADeYCBAwo-dIPntUmpibaoGBr) with 139 images plus `building_parts.json`. Use this for the app, Q1/Q3 cache generation, and expert-alignment evaluation.
- **Official WikiChurches files:** use the downloader when you need `churches.json`, metadata, the full image archive, or the official annotation file.

```bash
uv run python scripts/download_wikichurches.py --list
uv run python scripts/download_wikichurches.py --files churches.json image_meta.json building_parts.json
```

Expected dataset structure:

```text
dataset/
├── images/
│   ├── Q18785543_wd0.jpg
│   └── ...
└── building_parts.json
```

Generate the baseline app caches:

```bash
uv run python -m app.precompute.generate_attention_cache --models all
uv run python -m app.precompute.generate_feature_cache --models all
uv run python -m app.precompute.generate_heatmap_images --colormap viridis
uv run python -m app.precompute.generate_metrics_cache
```

Generate the Q3 per-head cache scope:

```bash
uv run python -m app.precompute.generate_attention_cache --models dinov2 dinov3 mae clip --per-head
uv run python -m app.precompute.generate_metrics_cache --models dinov2 dinov3 mae clip --per-head
uv run python -m app.precompute.generate_attention_cache --finetuned --models dinov2 dinov3 mae clip --strategies lora full --per-head
uv run python -m app.precompute.generate_metrics_cache --finetuned --models dinov2 dinov3 mae clip --strategies lora full --per-head
```

Optional Q3 frozen-backbone control:

```bash
uv run python -m app.precompute.generate_attention_cache --finetuned --models dinov2 dinov3 mae clip --strategies linear_probe --per-head
uv run python -m app.precompute.generate_metrics_cache --finetuned --models dinov2 dinov3 mae clip --strategies linear_probe --per-head
```

Run the app:

```bash
./dev.sh
```

This starts the backend at `http://127.0.0.1:8000` and frontend at `http://127.0.0.1:5173`.

## Submission Artifacts

Start here if you are reviewing the academic submission:

- Final PDF: [docs/final_report/ISY5004_report_final.pdf](docs/final_report/ISY5004_report_final.pdf)
- Report Markdown source: [docs/core/project_report_final.md](docs/core/project_report_final.md)
- Report figures: [docs/final_report/figures/](docs/final_report/figures/)
- Q3 report-view source figures: [docs/core/assets/](docs/core/assets/)
- Video plan, slide outline, script, PDF, and PPTX: [docs/plans/video/](docs/plans/video/)
- Active experiment pointer: [outputs/results/active_experiment.json](outputs/results/active_experiment.json)
- Q2 run matrix: [outputs/results/experiments/fine_tuning_primary_20260327/run_matrix.json](outputs/results/experiments/fine_tuning_primary_20260327/run_matrix.json)
- Q2 analysis: [outputs/results/experiments/fine_tuning_primary_20260327/q2_metrics_analysis.json](outputs/results/experiments/fine_tuning_primary_20260327/q2_metrics_analysis.json)

Git includes the report, figures, Q2 result JSONs, run manifests, split metadata, and active experiment pointer. Git does not include the large local artifacts:

- `dataset/`
- `outputs/cache/`
- `outputs/checkpoints/`
- `outputs/slides/`

For full app-level reproduction of Q1 and Q3, regenerate `outputs/cache/metrics.db` and `outputs/cache/metrics_summary.json` using the cache commands above. For Q2, rerun the experiment workflow below or inspect the checked-in result artifacts.

## App Routes

| Route | Purpose |
|------|---------|
| `/` | Gallery of annotated WikiChurches images |
| `/image/:imageId` | Single-image overlays, annotations, metrics, and Q3 drill-down |
| `/compare` | Frozen model and variant comparisons |
| `/dashboard` | Q1 overview and main Q3 discovery surface |
| `/q2` | Fine-tuning summary from the active experiment |
| `/q3-report` | Report-focused Q3 head ranking, feature matrix, and frozen-to-adapted delta views |

## Q2 Reproduction

The primary experiment ID is `fine_tuning_primary_20260327`.
This path requires the style-labeled pool from `churches.json`, not only the 139-image annotated subset.

```bash
EXPERIMENT_ID=fine_tuning_primary_20260327

uv run python experiments/scripts/fine_tune_models.py --all --freeze-backbone --epochs 3 --experiment-id "$EXPERIMENT_ID"
uv run python experiments/scripts/fine_tune_models.py --all --lora --epochs 3 --experiment-id "$EXPERIMENT_ID"
uv run python experiments/scripts/fine_tune_models.py --all --epochs 3 --experiment-id "$EXPERIMENT_ID"

uv run python experiments/scripts/analyze_q2_metrics.py \
  --experiment-id "$EXPERIMENT_ID" \
  --models clip dinov2 dinov3 mae siglip siglip2 \
  --strategies linear_probe lora full
```

Supplementary Q2 analyses:

```bash
uv run python experiments/scripts/analyze_style_breakdown.py --experiment-id "$EXPERIMENT_ID" --strategy full
uv run python experiments/scripts/analyze_model_correlation.py --experiment-id "$EXPERIMENT_ID" --strategy full
uv run python experiments/scripts/analyze_feature_delta_iou.py --experiment-id "$EXPERIMENT_ID" --model mae --strategy full --style Renaissance
uv run python experiments/scripts/analyze_q1_continuous_baselines.py
```

## Report and Presentation Outputs

Generate the current report-facing figures and presentation assets:

```bash
uv run python experiments/scripts/generate_run_matrix_figures.py
uv run python experiments/scripts/generate_slide_images.py
cd experiments/scripts && npm install && node create_presentation.js
```

The Q3 report route supplies the screenshot-friendly views used by the report and video plan:

- `view=head-ranking` for ranked heads by model, variant, layer, metric, and percentile
- `view=head-feature-matrix` for head-by-feature evidence
- `view=frozen-delta` for frozen-to-LoRA and frozen-to-Full ranking shifts

The full command surface is in [docs/reference/cli_reference.md](docs/reference/cli_reference.md). The experiment artifact contract is in [docs/reference/fine_tuning_run_matrix.md](docs/reference/fine_tuning_run_matrix.md).

## Developer Checks

```bash
uv run ruff check .
uv run mypy
uv run pytest
cd app/frontend && npm run lint && npm run build
```

## Documentation

| Document | Use it for |
|----------|------------|
| [docs/README.md](docs/README.md) | Documentation index |
| [docs/user_guide.md](docs/user_guide.md) | App walkthroughs |
| [docs/reference/cli_reference.md](docs/reference/cli_reference.md) | Command and flag reference |
| [docs/reference/api_reference.md](docs/reference/api_reference.md) | Backend API contracts |
| [docs/reference/fine_tuning_run_matrix.md](docs/reference/fine_tuning_run_matrix.md) | Q2 artifact layout |
| [docs/reference/per_head_methodology.md](docs/reference/per_head_methodology.md) | Q3 per-head method |

## Layout

```text
ssl_wikichurches/
├── app/                 # FastAPI backend, React frontend, cache scripts
├── dataset/             # Local WikiChurches data, not tracked
├── docs/                # Report, references, and user docs
├── experiments/         # Fine-tuning and analysis scripts
├── outputs/             # Results, figures, local caches, checkpoints
├── scripts/             # Dataset and utility scripts
├── src/ssl_attention/   # Core library code
└── tests/               # Pytest suite
```

## References

- Barz & Denzler (2021). [WikiChurches](https://arxiv.org/abs/2108.06959)
- Oquab et al. (2023). [DINOv2](https://arxiv.org/abs/2304.07193)
- Simeoni et al. (2025). [DINOv3](https://arxiv.org/abs/2508.10104)

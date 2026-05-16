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

Download one dataset path:

- **Annotated subset:** [Google Drive package](https://drive.google.com/drive/folders/1fsf0k71ADeYCBAwo-dIPntUmpibaoGBr) with 139 images plus `building_parts.json`. Use this for the app, Q1/Q3 cache generation, and expert-alignment evaluation.
- **Official WikiChurches files:** use the downloader when you need `churches.json`, metadata, the full image archive, or the official annotation file.

```bash
uv run python scripts/download_wikichurches.py --list
uv run python scripts/download_wikichurches.py --files churches.json image_meta.json building_parts.json
```

Expected local structure:

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

Add the Q3 per-head cache scope:

```bash
uv run python -m app.precompute.generate_attention_cache --models dinov2 dinov3 mae clip --per-head
uv run python -m app.precompute.generate_metrics_cache --models dinov2 dinov3 mae clip --per-head
uv run python -m app.precompute.generate_attention_cache --finetuned --models dinov2 dinov3 mae clip --strategies lora full --per-head
uv run python -m app.precompute.generate_metrics_cache --finetuned --models dinov2 dinov3 mae clip --strategies lora full --per-head
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
| `/q3` | Advanced Q3 side-by-side workspace |
| `/q3-report` | Report-focused Q3 head ranking, feature matrix, and delta views |

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

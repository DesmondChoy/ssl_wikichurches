# Do Self-Supervised Vision Models Learn What Experts See?

This project investigates whether SSL models (DINOv2, DINOv3, MAE, CLIP, SigLIP, SigLIP 2) attend to the same visual features human experts consider diagnostic for architectural style classification. Using the WikiChurches dataset, we measure:

1. **Attention alignment** — IoU between model attention and expert annotations
2. **Fine-tuning effects** — Does task-specific fine-tuning shift attention toward expert features, and does the strategy (Linear Probe vs LoRA vs Full) matter?
3. **Head specialization** — Do individual attention heads specialize for different architectural features?

## Quick Start

**Requirements:** Python 3.12+, [uv](https://github.com/astral-sh/uv) (`brew install uv`), Node.js 18+

### 1. Install Dependencies

```bash
uv sync
```

This installs PyTorch, Transformers, FastAPI, and all other dependencies. Model weights (~400MB each) download automatically from HuggingFace Hub on first use.

### 2. Download Dataset

Download from Google Drive and extract to `dataset/`:

**[Download Dataset (Google Drive)](https://drive.google.com/drive/folders/1fsf0k71ADeYCBAwo-dIPntUmpibaoGBr)**

| File/Folder | Size | Contents |
|-------------|------|----------|
| `images/` | ~180 MB | 139 annotated church images |
| `building_parts.json` | 304 KB | Expert bounding box annotations |

<details>
<summary>Expected directory structure</summary>

```
dataset/
├── images/
│   ├── Q18785543_wd0.jpg
│   ├── Q2034923_wd0.jpg
│   └── ... (139 images)
└── building_parts.json
```
</details>

> **Note**: The official WikiChurches release reports **9,485 images** ([Zenodo](https://zenodo.org/records/5166987), [arXiv](https://arxiv.org/abs/2108.06959)). This app uses the 139 images with expert bounding-box annotations for attention-alignment evaluation.

### 3. Pre-compute Caches

Generate attention maps, feature cache, heatmaps, and metrics (10–30 min):

```bash
uv run python -m app.precompute.generate_attention_cache --models all
uv run python -m app.precompute.generate_feature_cache --models all
uv run python -m app.precompute.generate_heatmap_images --colormap viridis
uv run python -m app.precompute.generate_metrics_cache
```

To enable **Frozen vs Fine-tuned** on the Compare page (overlays and bbox similarity heatmaps), also run with `--finetuned` (after training checkpoints; see Fine-Tuning below):

```bash
uv run python -m app.precompute.generate_attention_cache --finetuned --models all --strategies linear_probe lora full
uv run python -m app.precompute.generate_feature_cache --finetuned --models all --strategies linear_probe lora full
uv run python -m app.precompute.generate_heatmap_images --finetuned --models all --strategies linear_probe lora full
```

> **Tip:** Test with a subset first: `--models dinov2 --layers 11`

### 4. Run the App

```bash
./dev.sh  # Starts backend :8000 + frontend :5173
```

Open http://localhost:5173 in your browser.

Key routes after startup:

- `/compare` for `Model vs Model`, `Frozen vs Fine-tuned`, and `Variant vs Variant`
- `/q2` for the strategy-aware multi-metric Q2 attention-shift summary
- `/dashboard` for the base-model leaderboard and layer progression views

<details>
<summary>Alternative: Docker</summary>

```bash
docker compose up
```
</details>

## Models

All models use **ViT-Base** architecture (12 layers, 768 hidden dim, 12 attention heads).

| Model | HuggingFace ID | Architecture | Attention Methods | Training |
|-------|----------------|--------------|-------------------|----------|
| DINOv2 | `facebook/dinov2-with-registers-base` | ViT-B/14 | CLS, Rollout | Self-distillation |
| DINOv3 | `facebook/dinov3-vitb16-pretrain-lvd1689m` | ViT-B/16 | CLS, Rollout | Self-distillation + Gram |
| MAE | `facebook/vit-mae-base` | ViT-B/16 | CLS, Rollout | Masked autoencoding |
| CLIP | `openai/clip-vit-base-patch16` | ViT-B/16 | CLS, Rollout | Contrastive |
| SigLIP | `google/siglip-base-patch16-224` | ViT-B/16 | Mean | Contrastive (sigmoid) |
| SigLIP 2 (`siglip2`) | `google/siglip2-base-patch16-224` | ViT-B/16 | Mean | Contrastive (sigmoid) |
| ResNet-50 | `torchvision` | CNN | Grad-CAM | Supervised (ImageNet) |

**Model keys (CLI/API)**: `dinov2`, `dinov3`, `mae`, `clip`, `siglip`, `siglip2`, `resnet50`.

**Patch sizes**: DINOv2 uses 14×14 patches; other ViTs use 16×16. All maps are upsampled to 224×224 for comparison.

**Attention methods**: CLS uses the [CLS] token row; Rollout multiplies attention across layers. SigLIP/SigLIP 2 use a mean-attention visualization proxy because they do not expose a CLS-token path. ResNet-50 uses Grad-CAM. Metrics are per-method.

## Fine-Tuning

Fine-tune SSL backbones on architectural style classification (all ViT models; `resnet50` is excluded). Three strategies:

| Strategy | Config | What trains |
|----------|--------|-------------|
| **Linear Probe** | `freeze_backbone=True` | Classification head only |
| **LoRA** | `use_lora=True` | Low-rank adapters on attention layers |
| **Full** | (default) | Entire backbone + head |

**Fine-tunable model keys**: `dinov2`, `dinov3`, `mae`, `clip`, `siglip`, `siglip2`.

**Artifact naming**: the primary fine-tuning pipeline now writes experiment-scoped checkpoints to `outputs/checkpoints/<experiment_id>/{model}_{strategy}_finetuned.pt`. Fine-tuned cache keys still use `{model}_finetuned_{strategy}`. A legacy `{model}_finetuned.pt` fallback is still accepted for older full-fine-tuning runs.

```python
from ssl_attention.evaluation import FineTuningConfig, FineTuner, FineTunableModel

# Full fine-tuning (default)
config = FineTuningConfig(model_name="dinov2", num_epochs=10)
tuner = FineTuner(config)
result = tuner.train(FineTunableModel("dinov2"), dataset)

# LoRA fine-tuning (parameter-efficient)
config = FineTuningConfig(model_name="dinov2", use_lora=True, lora_rank=8)
result = tuner.train(FineTunableModel("dinov2"), dataset)
```

See `src/ssl_attention/evaluation/` for full API. The canonical experiment ledger now lives under `outputs/results/experiments/<experiment_id>/`, with `outputs/results/active_experiment.json` pointing the app, figure scripts, and docs refresh tools to the currently selected batch. LoRA support requires the [PEFT](https://github.com/huggingface/peft) library (included in dependencies).

### Fine-tuning Workflow

Fine-tuning runs on the current 4-style labeled subset derived from `churches.json` via `STYLE_MAPPING`. The 139 bbox-annotated images are excluded from the primary train/validation split and used only for final attention-alignment evaluation.

1. **Choose an experiment batch ID** and train checkpoints.

The primary rule is:

- use one shared non-annotated validation split for the whole batch
- choose one checkpoint per `model × strategy` by best classification validation accuracy
- evaluate all attention metrics from that checkpoint on the untouched annotated pool

The example below runs the full primary 18-run batch in three sweeps that all reuse the same shared split artifact:

```bash
EXPERIMENT_ID=fine_tuning_primary_20260327

# Linear probe
uv run python experiments/scripts/fine_tune_models.py --all --freeze-backbone --epochs 3 --experiment-id "$EXPERIMENT_ID"

# LoRA
uv run python experiments/scripts/fine_tune_models.py --all --lora --epochs 3 --experiment-id "$EXPERIMENT_ID"

# Full fine-tuning
uv run python experiments/scripts/fine_tune_models.py --all --epochs 3 --experiment-id "$EXPERIMENT_ID"
```

   Each run writes:

- `outputs/results/experiments/<experiment_id>/splits/<split_id>.json`
- `outputs/results/experiments/<experiment_id>/manifests/<run_id>_manifest.json`
- `outputs/results/experiments/<experiment_id>/run_matrix.json`
- `outputs/results/experiments/<experiment_id>/fine_tuning_results.json`
- `outputs/checkpoints/<experiment_id>/{model}_{strategy}_finetuned.pt`

   `--val-on-annotated-eval` still exists, but it is explicitly marked as `exploratory` in manifests and downstream results. It should not be used for the main experiment. Use `--include-annotated-eval` only if your local dataset contains only the 139 annotated images and you are intentionally running an annotated-only fallback.

2. **Run the fine-tuning attention analysis** (canonical output: `outputs/results/experiments/<experiment_id>/q2_metrics_analysis.json`, consumed by `/api/metrics/q2_summary` and the `/q2` page through `active_experiment.json`):

```bash
uv run python experiments/scripts/analyze_q2_metrics.py \
  --experiment-id "$EXPERIMENT_ID" \
  --models clip dinov2 dinov3 mae siglip siglip2 \
  --strategies linear_probe lora full
```

   The generated artifact includes IoU, coverage, Gaussian MSE, KL, and EMD summaries plus experiment provenance such as `experiment_id`, `split_id`, evaluation image count, git commit SHA, and checkpoint-selection rule. The legacy wrapper `experiments/scripts/analyze_delta_iou.py` still exists for compatibility, but the canonical app-facing artifact is `q2_metrics_analysis.json`.

3. **Precompute for the Compare page** (attention + feature cache + heatmaps for frozen and fine-tuned). Required for overlays and bbox similarity (“Similarity heatmaps are unavailable” appears without feature cache). Run **frozen** first, then **fine-tuned** (same `--strategies` as your checkpoints). Fine-tuned artifacts are written under strategy-aware cache keys such as `{model}_finetuned_lora` and `{model}_finetuned_full`.

```bash
# Frozen
uv run python -m app.precompute.generate_attention_cache --models dinov2 dinov3 mae clip siglip siglip2
uv run python -m app.precompute.generate_feature_cache --models dinov2 dinov3 mae clip siglip siglip2
uv run python -m app.precompute.generate_heatmap_images --models dinov2 dinov3 mae clip siglip siglip2

# Fine-tuned
uv run python -m app.precompute.generate_attention_cache --finetuned --models dinov2 dinov3 mae clip siglip siglip2 --strategies linear_probe lora full
uv run python -m app.precompute.generate_feature_cache --finetuned --models dinov2 dinov3 mae clip siglip siglip2 --strategies linear_probe lora full
uv run python -m app.precompute.generate_heatmap_images --finetuned --models dinov2 dinov3 mae clip siglip siglip2 --strategies linear_probe lora full
```

4. **Build base-model metrics cache** (dashboard APIs):

```bash
uv run python -m app.precompute.generate_metrics_cache
```

5. **Build strategy-aware fine-tuned metrics cache** (required for fine-tuned metric queries in Compare and for downstream analysis workflows):

```bash
uv run python -m app.precompute.generate_metrics_cache --finetuned --models dinov2 dinov3 mae clip siglip siglip2 --strategies linear_probe lora full
```

The dashboard leaderboard and `/api/compare/all_models_summary` still operate on the base `AVAILABLE_MODELS` set. The fine-tuned compare flows and the `/q2` analysis rely on the active experiment's `q2_metrics_analysis.json` plus the fine-tuned attention, feature, heatmap, and strategy-aware metrics caches above, not on a strategy-aware leaderboard surface in the dashboard.

## Data Exploration

Explore the dataset with Jupyter:

```bash
uv run jupyter lab notebooks/01_data_exploration.ipynb
```

## Project Structure

```
ssl_wikichurches/
├── app/
│   ├── backend/          # FastAPI server
│   ├── frontend/         # React + Vite + Tailwind
│   └── precompute/       # Cache generation scripts
├── dataset/              # WikiChurches data (gitignored)
├── docs/                 # Project documentation
│   ├── core/             # Proposal, implementation plan
│   ├── reference/        # API reference, heatmap implementation
│   ├── research/         # Attention methods, novelty analysis
│   └── enhancements/     # Fine-tuning methods, per-head analysis
├── experiments/          # Training & analysis scripts
├── outputs/              # Pre-computed caches (gitignored)
├── scripts/              # Utility scripts
├── src/ssl_attention/    # Core library
└── tests/                # Pytest tests
```

## References

- Barz & Denzler (2021). [WikiChurches](https://arxiv.org/abs/2108.06959)
- Oquab et al. (2023). [DINOv2](https://arxiv.org/abs/2304.07193)
- Simeoni et al. (2025). [DINOv3](https://arxiv.org/abs/2508.10104)
- Zhang et al. (2018). [Top-Down Neural Attention by Excitation Backprop](https://link.springer.com/article/10.1007/s11263-017-1059-x) (Pointing Game)

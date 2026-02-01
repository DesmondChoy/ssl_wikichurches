# SSL Attention Alignment on WikiChurches

Do self-supervised vision models attend to the same features human experts consider diagnostic? This project measures alignment between SSL model attention patterns and 631 expert-annotated architectural features on the WikiChurches dataset.

## Setup

```bash
uv sync
```

Model weights (~400MB each) download automatically from HuggingFace Hub on first use.

## Dataset

Download the [WikiChurches dataset](https://zenodo.org/records/5166987):

```bash
uv run python scripts/download_wikichurches.py -o dataset/
```

### Data Exploration

Explore the dataset structure with the Polars-based notebook:

```bash
uv run jupyter lab notebooks/01_data_exploration.ipynb
```

This notebook loads all 4 data sources (churches, feature types, annotations, style names) and provides exploratory analysis including distributions by century, country, and architectural style.

## Models

All models use **ViT-Base** architecture (12 layers, 768 hidden dim, 12 attention heads).

| Model | HuggingFace ID | Architecture | Training |
|-------|----------------|--------------|----------|
| DINOv2 | `facebook/dinov2-with-registers-base` | ViT-B/14 | Self-distillation |
| DINOv3 | `facebook/dinov3-vitb16-pretrain-lvd1689m` | ViT-B/16 | Self-distillation + Gram |
| MAE | `facebook/vit-mae-base` | ViT-B/16 | Masked autoencoding |
| CLIP | `openai/clip-vit-base-patch16` | ViT-B/16 | Contrastive |
| SigLIP 2 | `google/siglip2-base-patch16-224` | ViT-B/16 | Contrastive (sigmoid) |

**Note on patch sizes**: DINOv2 uses 14×14 patches (256 tokens for 224×224 images) while other models use 16×16 patches (196 tokens). No official DINOv2 ViT-B/16 variant exists. For visualization, all attention maps are upsampled to image resolution, making cross-model comparison valid despite the different native resolutions.

## Fine-Tuning

Fine-tune any SSL model on architectural style classification:

```python
from ssl_attention.evaluation import FineTuningConfig, FineTuner, FineTunableModel
from ssl_attention.data import FullDataset
from ssl_attention.config import DATASET_PATH

config = FineTuningConfig(model_name="dinov2", num_epochs=10)
model = FineTunableModel("dinov2", freeze_backbone=False)
dataset = FullDataset(DATASET_PATH, filter_labeled=True)

tuner = FineTuner(config)
result = tuner.train(model, dataset)
print(f"Best validation accuracy: {result.best_val_acc:.1%}")
```

Checkpoints are saved to `outputs/checkpoints/`. After fine-tuning, compare attention patterns before vs after to measure alignment shift.

## Visualization App

Interactive web app to explore attention patterns across models and layers.

### Pre-computation (one-time)

Before running the app, pre-compute attention maps, feature embeddings, and metrics. This extracts attention and per-layer features from all 5 models × 12 layers × 139 images, then renders heatmaps and computes IoU metrics. Run once; results are cached.

```bash
# Extract attention maps
python -m app.precompute.generate_attention_cache --models all

# Extract per-layer feature embeddings (for similarity analysis)
python -m app.precompute.generate_feature_cache --models all

# Render heatmap overlays
python -m app.precompute.generate_heatmap_images --colormap viridis

# Compute IoU metrics
python -m app.precompute.generate_metrics_cache
```

To test with a subset first:
```bash
python -m app.precompute.generate_attention_cache --models dinov2 --layers 11
python -m app.precompute.generate_feature_cache --models dinov2 --layers 11
```

### Run the App

```bash
# Backend (port 8000)
uvicorn app.backend.main:app --reload

# Frontend (port 5173) - in another terminal
cd app/frontend && npm install && npm run dev
```

Or with Docker:
```bash
docker compose up
```

## References

- Barz & Denzler (2021). [WikiChurches](https://arxiv.org/abs/2108.06959)
- Oquab et al. (2023). [DINOv2](https://arxiv.org/abs/2304.07193)
- Simeoni et al. (2025). [DINOv3](https://arxiv.org/abs/2508.10104)

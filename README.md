# Do Self-Supervised Vision Models Learn What Experts See?

This project investigates whether SSL models (DINOv2, DINOv3, MAE, CLIP, SigLIP 2) attend to the same visual features human experts consider diagnostic for architectural style classification. Using the WikiChurches dataset, we measure:

1. **Attention alignment** — IoU between model attention and expert annotations
2. **Fine-tuning effects** — Does task-specific training shift attention toward expert features?
3. **Representation coherence** — Do patch embeddings cluster around semantic concepts?

## Quick Start

```bash
# 1. Install Python dependencies
uv sync

# 2. Download dataset (see Dataset section below)

# 3. Pre-compute attention maps and metrics (run once)
python -m app.precompute.generate_attention_cache --models all
python -m app.precompute.generate_feature_cache --models all
python -m app.precompute.generate_heatmap_images --colormap viridis
python -m app.precompute.generate_metrics_cache

# 4. Start the app
./dev.sh
```

## Requirements

- **Python 3.12+** — Required for type hints and modern features
- **[uv](https://github.com/astral-sh/uv)** — Fast Python package manager (`brew install uv` or `pip install uv`)
- **Node.js 18+** — For the frontend (check with `node --version`)

## Setup

### 1. Install Python Dependencies

```bash
uv sync
```

This installs all dependencies including PyTorch, Transformers, and FastAPI. Model weights (~400MB each) download automatically from HuggingFace Hub on first use.

### 2. Download Dataset

Download the dataset from Google Drive and extract to `dataset/`:

**[Download Dataset (Google Drive)](https://drive.google.com/drive/folders/TODO_REPLACE_WITH_ACTUAL_LINK)**

The download contains:
| File/Folder | Size | Purpose |
|-------------|------|---------|
| `images/` | ~50 MB | 139 annotated church images |
| `building_parts.json` | 304 KB | Expert bounding box annotations |

Your `dataset/` directory should look like:
```
dataset/
├── images/
│   ├── Q18785543_wd0.jpg
│   ├── Q2034923_wd0.jpg
│   └── ... (139 images)
└── building_parts.json
```

> **Note**: The full WikiChurches dataset (9,502 images) is available on [Zenodo](https://zenodo.org/records/5166987), but this app only uses the 139 images with expert annotations.

### 3. Pre-compute Attention Data

Pre-compute attention maps, feature embeddings, heatmaps, and metrics. This takes 10-30 minutes depending on hardware:

```bash
# Generate attention maps for all models and layers
python -m app.precompute.generate_attention_cache --models all

# Generate feature embeddings
python -m app.precompute.generate_feature_cache --models all

# Generate heatmap images
python -m app.precompute.generate_heatmap_images --colormap viridis

# Calculate IoU metrics
python -m app.precompute.generate_metrics_cache
```

**Tip:** Test with a subset first: `--models dinov2 --layers 11`

### 4. Install Frontend Dependencies

The frontend uses React + Vite + Tailwind. Dependencies are installed automatically when you run `./dev.sh`, but you can also install manually:

```bash
cd app/frontend
npm install
```

### 5. Run the App

```bash
./dev.sh  # Starts backend on :8000 + frontend on :5173
```

Open http://localhost:5173 in your browser.

**Alternative (Docker):**
```bash
docker compose up
```

## Models

All models use **ViT-Base** architecture (12 layers, 768 hidden dim, 12 attention heads).

| Model | HuggingFace ID | Architecture | Training |
|-------|----------------|--------------|----------|
| DINOv2 | `facebook/dinov2-with-registers-base` | ViT-B/14 | Self-distillation |
| DINOv3 | `facebook/dinov3-vitb16-pretrain-lvd1689m` | ViT-B/16 | Self-distillation + Gram |
| MAE | `facebook/vit-mae-base` | ViT-B/16 | Masked autoencoding |
| CLIP | `openai/clip-vit-base-patch16` | ViT-B/16 | Contrastive |
| SigLIP 2 | `google/siglip2-base-patch16-224` | ViT-B/16 | Contrastive (sigmoid) |
| ResNet-50 | `torchvision` | CNN | Supervised (ImageNet) |

**Note on patch sizes**: DINOv2 uses 14×14 patches (256 tokens for 224×224 images) while other models use 16×16 patches (196 tokens). For visualization, all attention maps are upsampled to image resolution, making cross-model comparison valid despite the different native resolutions.

## Fine-Tuning

Fine-tune any SSL model on architectural style classification:

```python
from ssl_attention.evaluation import FineTuningConfig, FineTuner, FineTunableModel

tuner = FineTuner(FineTuningConfig(model_name="dinov2"))
result = tuner.train(FineTunableModel("dinov2"), dataset)
```

See `src/ssl_attention/evaluation/` for full API. Checkpoints save to `outputs/checkpoints/`.

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
├── outputs/              # Pre-computed caches (gitignored)
├── scripts/              # Utility scripts
├── src/ssl_attention/    # Core library
└── tests/                # Pytest tests
```

## References

- Barz & Denzler (2021). [WikiChurches](https://arxiv.org/abs/2108.06959)
- Oquab et al. (2023). [DINOv2](https://arxiv.org/abs/2304.07193)
- Simeoni et al. (2025). [DINOv3](https://arxiv.org/abs/2508.10104)

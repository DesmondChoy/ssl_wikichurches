# Do Self-Supervised Vision Models Learn What Experts See?

This project investigates whether SSL models (DINOv2, DINOv3, MAE, CLIP, SigLIP 2) attend to the same visual features human experts consider diagnostic for architectural style classification. Using the WikiChurches dataset, we measure:

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

> **Note**: The full WikiChurches dataset (9,502 images) is available on [Zenodo](https://zenodo.org/records/5166987), but this app only uses the 139 images with expert annotations.

### 3. Pre-compute Caches

Generate attention maps, feature embeddings, heatmaps, and metrics (10-30 min):

```bash
python -m app.precompute.generate_attention_cache --models all
python -m app.precompute.generate_feature_cache --models all
python -m app.precompute.generate_heatmap_images --colormap viridis
python -m app.precompute.generate_metrics_cache
```

To cache attention from fine-tuned checkpoints (for frozen vs fine-tuned comparison):

```bash
python -m app.precompute.generate_attention_cache --finetuned --models all
```

> **Tip:** Test with a subset first: `--models dinov2 --layers 11`

### 4. Run the App

```bash
./dev.sh  # Starts backend :8000 + frontend :5173
```

Open http://localhost:5173 in your browser.

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
| SigLIP 2 | `google/siglip2-base-patch16-224` | ViT-B/16 | Mean | Contrastive (sigmoid) |
| ResNet-50 | `torchvision` | CNN | Grad-CAM | Supervised (ImageNet) |

**Note on patch sizes**: DINOv2 uses 14×14 patches (256 tokens for 224×224 images) while other models use 16×16 patches (196 tokens). For visualization, all attention maps are upsampled to image resolution, making cross-model comparison valid despite the different native resolutions.

**Note on attention methods**: CLS extracts attention from the [CLS] token row. Rollout multiplies attention matrices across layers to approximate total information flow. SigLIP lacks a CLS token and uses mean received attention (MAP attention pooling). ResNet-50 uses Grad-CAM on convolutional feature maps. Metrics are computed per-method — selecting a different method changes the attention heatmap and all derived metrics.

## Fine-Tuning

Fine-tune any SSL model on architectural style classification. Three strategies are supported:

| Strategy | Config | What trains |
|----------|--------|-------------|
| **Linear Probe** | `freeze_backbone=True` | Classification head only |
| **LoRA** | `use_lora=True` | Low-rank adapters on attention layers |
| **Full** | (default) | Entire backbone + head |

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

See `src/ssl_attention/evaluation/` for full API. Checkpoints save to `outputs/checkpoints/`. LoRA support requires the [PEFT](https://github.com/huggingface/peft) library (included in dependencies).

## Data Exploration

Explore the dataset with Jupyter:

```bash
uv run jupyter lab notebooks/01_data_exploration.ipynb
```

## Project Structure

```
ssl_wikichurches/
├── app/
│   ├── backend/          # FastAPI server (25 REST endpoints)
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

# SSL Attention Alignment on WikiChurches

Do self-supervised vision models attend to the same features human experts consider diagnostic? This project measures alignment between SSL model attention patterns and 631 expert-annotated architectural features on the WikiChurches dataset.

## Models

| Model | Source | Training Paradigm |
|-------|--------|-------------------|
| DINOv2 | `facebook/dinov2-with-registers-base` | Self-distillation |
| DINOv3 | `facebook/dinov3-vitb16-pretrain-lvd1689m` | Self-distillation + Gram Anchoring |
| MAE | `facebook/vit-mae-base` | Masked autoencoding |
| CLIP | `openai/clip-vit-base-patch16` | Contrastive (softmax) |
| SigLIP 2 | `google/siglip2-base-patch16-224` | Contrastive (sigmoid) |

## Setup

```bash
uv sync
```

## Dataset

Download the [WikiChurches dataset](https://zenodo.org/records/5166987):

```bash
# Download all files (~11.8 GB)
uv run python scripts/download_wikichurches.py -o dataset/

# Metadata only (skip images/models)
uv run python scripts/download_wikichurches.py -o dataset/ --exclude images.zip models.zip
```

**Annotated subset:** 139 churches with 631 bounding boxes across 4 styles (Gothic, Romanesque, Baroque, Renaissance) and 106 architectural feature types.

## Project Structure

```
ssl_wikichurches/
├── src/ssl_attention/     # Core library
│   ├── models/            # Model wrappers (DINOv2, DINOv3, MAE, CLIP, SigLIP)
│   ├── attention/         # Attention extraction (CLS, rollout, GradCAM)
│   ├── data/              # Dataset and annotation parsing
│   └── metrics/           # IoU computation and baselines
├── experiments/           # Experiment configs and scripts
├── dataset/               # WikiChurches data (git-ignored)
└── docs/                  # Proposal and implementation plan
```

## References

- Barz & Denzler (2021). [WikiChurches: A Fine-Grained Dataset of Architectural Styles](https://arxiv.org/abs/2108.06959). *NeurIPS Datasets and Benchmarks*.
- Oquab et al. (2023). [DINOv2: Learning Robust Visual Features without Supervision](https://arxiv.org/abs/2304.07193).
- Simeoni et al. (2025). [DINOv3](https://arxiv.org/abs/2508.10104).

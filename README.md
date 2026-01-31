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

## Models

| Model | HuggingFace ID | Training |
|-------|----------------|----------|
| DINOv2 | `facebook/dinov2-with-registers-base` | Self-distillation |
| DINOv3 | `facebook/dinov3-vitb16-pretrain-lvd1689m` | Self-distillation + Gram |
| MAE | `facebook/vit-mae-base` | Masked autoencoding |
| CLIP | `openai/clip-vit-base-patch16` | Contrastive |
| SigLIP 2 | `google/siglip2-base-patch16-224` | Contrastive (sigmoid) |

## References

- Barz & Denzler (2021). [WikiChurches](https://arxiv.org/abs/2108.06959)
- Oquab et al. (2023). [DINOv2](https://arxiv.org/abs/2304.07193)
- Simeoni et al. (2025). [DINOv3](https://arxiv.org/abs/2508.10104)

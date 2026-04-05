# Per-Head Attention Visualization

> **Enhancement proposed:** January 2026
> **Current status:** Shipped in the product surface; some environments may still need cache backfill
> **Purpose:** Provide a concise product-oriented overview of the shipped per-head feature

> **Related documents:**
> - [Project Proposal — Q3: Head Specialization](../core/project_proposal.md#research-questions-and-approaches)
> - [Attention Methods — Head Fusion Strategies](../research/attention_methods.md#head-fusion-strategies)
> - [Q3 Per-Head Attention Methodology](../reference/per_head_methodology.md)
> - [Metrics Methodology](../reference/metrics_methodology.md)

## What Shipped

Per-head attention support is implemented in two places:

- The attention viewer can request raw attention for one head at a time instead of only the fused map.
- The dashboard Q3 panel can show head rankings and head-by-feature matrices from precomputed per-head metrics.

The main operational issue in environments where Q3 still shows a warning is usually **missing cache backfill**, not missing product code. The technical source of truth for extraction rules, model-specific caveats, and interpretability limits is [Q3 Per-Head Attention Methodology](../reference/per_head_methodology.md).

## Why This Matters

Per-head analysis asks a narrower and more useful question than fused attention alone: not just whether a model attends to an annotated region overall, but whether **specific heads** align better with specific architectural features.

That framing is well aligned with prior research:

- [Caron et al. (2021), *Emerging Properties in Self-Supervised Vision Transformers*](https://arxiv.org/abs/2104.14294) showed that DINO-style heads can produce object-aligned attention maps.
- [Walmer et al. (2023), *Teaching Matters*](https://openaccess.thecvf.com/content/CVPR2023/papers/Walmer_Teaching_Matters_Investigating_the_Role_of_Supervision_in_Vision_Transformers_CVPR_2023_paper.pdf) found multiple head archetypes and different local/global behavior across supervision styles.
- [Raghu et al. (2021), *Do Vision Transformers See Like Convolutional Neural Networks?*](https://arxiv.org/abs/2108.08810) showed that ViT layers mix local and global heads differently across depth.
- [Voita et al. (2019), *Analyzing Multi-Head Self-Attention*](https://arxiv.org/abs/1905.09418) provided the broader head-specialization framing that a subset of heads often carries distinct, interpretable roles.

For this project, the practical takeaway is simple: if certain heads repeatedly align with expert annotations for towers, windows, arches, or facade regions, that is useful evidence about head specialization even if it is not a full causal explanation of model decisions.

## Current Behavior

### Supported per-head configurations

| Model family | Per-head method | Status | Notes |
|-------------|-----------------|--------|-------|
| DINOv2 | `cls` | Supported | Uses CLS-to-patch attention |
| DINOv3 | `cls` | Supported | Uses CLS-to-patch attention and skips register tokens |
| MAE | `cls` | Supported | Uses CLS-to-patch attention |
| CLIP (ViT) | `cls` | Supported | Uses class-token attention |
| SigLIP | `mean` | Supported | Uses a per-head mean-attention proxy |
| SigLIP2 | `mean` | Supported | Uses a per-head mean-attention proxy |
| Any model with `rollout` | `rollout` | Not supported | Rollout is a multi-layer aggregate, not the Q3 unit of analysis |
| ResNet-50 | `gradcam` | Not supported | No transformer attention heads |

### Current UI behavior

- The attention viewer only shows the head selector when the selected model and method support per-head viewing **and** the cache actually contains per-head entries for that model.
- The dashboard Q3 panel does not compute per-head metrics on demand. It reads precomputed head-ranking and head-by-feature data from the metrics database.
- If those database rows are missing, the dashboard shows the current warning state telling the user to run the per-head metrics precompute command first.

### Current storage artifacts

- Per-head attention variants are stored in `outputs/cache/attention_viz.h5`.
- Q3 per-head metrics are stored in `outputs/cache/metrics.db`.
- The dedicated Q3 tables are:
  - `head_image_metrics`
  - `head_summary_metrics`
  - `head_feature_metrics`

This is the current shipped storage layout. Earlier draft plans that described per-image `.npy` files are no longer accurate.

## Operational Notes

Q3 needs **both** the per-head attention cache and the per-head metrics database backfill.

### Frozen models

```bash
uv run python -m app.precompute.generate_attention_cache --models dinov2 dinov3 mae clip siglip siglip2 --per-head
uv run python -m app.precompute.generate_metrics_cache --models dinov2 dinov3 mae clip siglip siglip2 --per-head
```

### Fine-tuned variants

```bash
uv run python -m app.precompute.generate_attention_cache --finetuned --models dinov2 dinov3 mae clip siglip siglip2 --strategies linear_probe lora full --per-head
uv run python -m app.precompute.generate_metrics_cache --finetuned --models dinov2 dinov3 mae clip siglip siglip2 --strategies linear_probe lora full --per-head
```

Notes:

- The scripts already enforce the model-method compatibility used by Q3: `cls` for DINOv2, DINOv3, MAE, and CLIP; `mean` for SigLIP and SigLIP2.
- Dashboard variant selections such as `frozen`, `linear_probe`, `lora`, and `full` only work when matching per-head cache entries and per-head metric rows exist for that variant.
- If runtime or storage constraints require chunking, use the same model and strategy sets but process them in smaller batches.

## Interpretation Guardrails

- This feature supports **descriptive head-specialization analysis**, not full causal attribution.
- For SigLIP and SigLIP2, the per-head view is based on a mean-attention proxy derived from the self-attention tensor, not the model's learned pooling head.
- Prior work suggests that late-layer CLIP and fully supervised ViT attention can be less segmentation-friendly than DINO-style object-aligned heads, so cross-model comparisons should be interpreted with care.
- If the goal shifts from "which heads align with these annotations?" to "which computation explains this prediction?", stronger follow-ups would include attention flow, rollout-style analyses, or relevance propagation methods such as those discussed by Abnar and Zuidema and by Chefer et al.

For the detailed reasoning behind these guardrails, use [Q3 Per-Head Attention Methodology](../reference/per_head_methodology.md).

## Historical Note

This file originally held a much longer phase-by-phase implementation proposal for adding per-head support across the backend, frontend, and cache pipeline. That material has been intentionally retired from the main body because the feature is now shipped and the old plan no longer matches the live codebase. The important historical context that remains is the original motivation: expose head-level behavior rather than only fused attention so Q3 can study head specialization directly.

## References

### Head specialization and ViT behavior

- Caron, M., et al. (2021). [*Emerging Properties in Self-Supervised Vision Transformers*](https://arxiv.org/abs/2104.14294)
- Raghu, M., et al. (2021). [*Do Vision Transformers See Like Convolutional Neural Networks?*](https://arxiv.org/abs/2108.08810)
- Walmer, M., et al. (2023). [*Teaching Matters: Investigating the Role of Supervision in Vision Transformers*](https://openaccess.thecvf.com/content/CVPR2023/papers/Walmer_Teaching_Matters_Investigating_the_Role_of_Supervision_in_Vision_Transformers_CVPR_2023_paper.pdf)
- Voita, E., et al. (2019). [*Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned*](https://arxiv.org/abs/1905.09418)

### Cautionary and stronger interpretability methods

- Abnar, S., and Zuidema, W. (2020). [*Quantifying Attention Flow in Transformers*](https://arxiv.org/abs/2005.00928)
- Chefer, H., et al. (2021). [*Transformer Interpretability Beyond Attention Visualization*](https://arxiv.org/abs/2012.09838)

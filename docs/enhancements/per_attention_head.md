# Per-Head Attention Visualization

This note describes the current Q3 product surface and the recommended study framing for per-head analysis.

> **Related documents:**
> - [Project Proposal — Q3: Head Specialization](../core/project_proposal.md#research-questions-and-approaches)
> - [Q3 Per-Head Attention Methodology](../reference/per_head_methodology.md)
> - [Metrics Methodology](../reference/metrics_methodology.md)
> - [Attention Methods — Head Fusion Strategies](../research/attention_methods.md#head-fusion-strategies)

## 1. Product Context

The Q3 surface combines three connected views:

- **Dashboard Q3** for dataset-level discovery through head rankings, head-by-feature heatmaps, inline exemplars, and frozen-to-adapted deltas
- **Image Detail Q3** for single-image drill-down with variant-aware top-head ranking, expandable head gallery, and `Head Attention` vs `Feature Similarity` modes
- **Q3 Report `/q3-report`** for report-facing head ranking, head-feature matrix, and frozen-to-adapted delta layouts

The supporting backend and cache surface includes:

- raw per-head attention requests in the attention API and image viewer
- Q3 head-ranking, image-level head-ranking, head-by-feature matrix, and exemplar endpoints
- per-head cache generation and per-head metrics precompute paths
- variant-aware per-head availability metadata for strict Q3 selectors

## 2. Research Focus

The Q3 capability is broad enough to explore multiple architectures and variant states, but the most defensible headline study stays narrower:

**How do different pretraining supervision families shape head specialization, and how does adaptation change the dominant head set within those families?**

That framing keeps the main claim focused on architecture-native CLS-token models and makes the cache-population requirements tractable.

## 3. Recommended Scope

### Research framing

Focus Q3 on **architecture-native CLS-token models** that let us compare distinct supervision families without relying on proxy-based per-head extraction:

- `dinov2`, `dinov3`: self-supervised self-distillation style models
- `mae`: self-supervised reconstruction-based model
- `clip`: language-supervised contrastive model

This is a stronger academic framing than "all models with any per-head path" because it compares models whose Q3 head maps are all derived from the same basic CLS-to-patch attention mechanism, while still spanning meaningfully different training objectives.

### Primary study scope

- Models: `dinov2`, `dinov3`, `mae`, `clip`
- Variants: `frozen`, `lora`, `full`
- Control: `linear_probe` is optional and should be treated as a sanity-check control rather than a primary research condition, because the backbone stays frozen in that strategy
- Q3 method: `cls`

### Explicit exclusions for this scope

- `siglip` and `siglip2` are out of primary scope because their Q3 per-head analysis uses a mean-attention proxy rather than the model's learned pooling head
- `rollout` remains out of scope because it is a multi-layer aggregate rather than the unit of analysis for head specialization
- `resnet50` remains out of scope because it has no transformer attention heads

## 4. Why This Scope Is Defensible

This scope stays close to existing literature while keeping the implementation tractable.

- [Caron et al. (2021), *Emerging Properties in Self-Supervised Vision Transformers*](https://arxiv.org/abs/2104.14294) supports the idea that DINO-style heads can become semantically meaningful and object-aligned.
- [He et al. (2021), *Masked Autoencoders Are Scalable Vision Learners*](https://arxiv.org/abs/2111.06377) establishes MAE as a self-supervised reconstruction-based ViT and makes it a principled inclusion in a supervision-family comparison.
- [Walmer et al. (2023), *Teaching Matters*](https://openaccess.thecvf.com/content/CVPR2023/papers/Walmer_Teaching_Matters_Investigating_the_Role_of_Supervision_in_Vision_Transformers_CVPR_2023_paper.pdf) is especially relevant because it compares ViTs across supervision regimes and reports diverse head behaviors and local/global processing patterns.
- [Raghu et al. (2021), *Do Vision Transformers See Like Convolutional Neural Networks?*](https://arxiv.org/abs/2108.08810) supports studying layer- and head-level differences in local/global behavior.
- [Voita et al. (2019), *Analyzing Multi-Head Self-Attention*](https://arxiv.org/abs/1905.09418) provides the broader head-specialization framing that a subset of heads can carry distinct functional roles.

In other words, this phase is broad enough to surface meaningful differences, but narrow enough to avoid mixing architecture-native and proxy-based head interpretations in the same headline claim.

## 5. Proposed Hypotheses

### H1. Head specialization is sparse rather than uniform

Within a given model and variant, a small subset of late-layer heads will account for a disproportionate share of top-ranked alignment results.

### H2. Supervision family affects the dominant head set

The dominant heads and feature-preference patterns will differ across:

- DINO-style self-distillation (`dinov2`, `dinov3`)
- reconstruction-based self-supervision (`mae`)
- language-supervised contrastive pretraining (`clip`)

### H3. Adaptation changes which heads dominate

`lora` and `full` fine-tuning will shift or sharpen the dominant head set relative to `frozen`, rather than leaving the same heads equally dominant.

### Control expectation

`linear_probe` should not be treated as a primary research condition because it freezes the backbone. If included, it should behave as a sanity-check control rather than a main adaptation result.

## 6. Product Expectations

### What the app should help a user answer

On the dashboard Q3 panel:

- Which heads dominate for a given model, metric, and layer?
- Do different architectural feature types prefer different heads?
- Which head-feature cells stay dark in the heatmap and still look plausible on exemplar images?
- Does the dominant head set change when moving from `frozen` to `lora` or `full`?
- Do DINOv2, DINOv3, MAE, and CLIP show different head-specialization patterns before moving into report or image-level inspection?

On `/q3-report`:

- Which head-ranking, feature-matrix, or frozen-to-adapted delta view best supports the report narrative?
- Does the selected report view preserve model, variant, layer, metric, percentile, head, and feature context in the URL?

On the image-level attention view:

- What does one specific head actually attend to?
- Are the heads that rank highly in Q3 visually plausible on exemplar images?

### What this phase does not require

- no attempt to make every current model and every variant fully populated
- no claim that raw per-head attention is a full causal explanation method

## 7. Recommended Cache Population Scope

Populate the scoped study conditions first instead of trying to fill every per-head-capable model and variant combination.

### Frozen scope

```bash
uv run python -m app.precompute.generate_attention_cache --models dinov2 dinov3 mae clip --per-head
uv run python -m app.precompute.generate_metrics_cache --models dinov2 dinov3 mae clip --per-head
```

### Fine-tuned study scope

```bash
uv run python -m app.precompute.generate_attention_cache --finetuned --models dinov2 dinov3 mae clip --strategies lora full --per-head
uv run python -m app.precompute.generate_metrics_cache --finetuned --models dinov2 dinov3 mae clip --strategies lora full --per-head
```

### Optional control scope

Run `linear_probe` only if the team explicitly wants a frozen-backbone control comparison:

```bash
uv run python -m app.precompute.generate_attention_cache --finetuned --models dinov2 dinov3 mae clip --strategies linear_probe --per-head
uv run python -m app.precompute.generate_metrics_cache --finetuned --models dinov2 dinov3 mae clip --strategies linear_probe --per-head
```

### Validation target

Success means the Q3 surfaces stop showing missing-data warnings for the scoped study conditions above. Out-of-scope selections can remain unpopulated.

## 8. Storage And API Surfaces

The storage and API layout that supports this direction includes:

- per-head attention variants live in `outputs/cache/attention_viz.h5`
- per-head metrics live in `outputs/cache/metrics.db`
- Q3 table set:
  - `head_image_metrics`
  - `head_summary_metrics`
  - `head_feature_metrics`
  - `head_feature_image_metrics`
- relevant APIs:
  - `/api/attention/{image_id}/raw`
  - `/api/attention/models`
  - `/api/metrics/model/{model}/head_ranking`
  - `/api/metrics/model/{model}/head_feature_matrix`
  - `/api/metrics/model/{model}/head_exemplars`

Dashboard Q3 uses an interactive heatmap plus an inline exemplar panel. `/q3-report` uses a cleaner report-facing matrix view. Feature-cell drill-down is backed by deterministic per-image-per-head-per-feature cache rows so the selected exemplar images match the chosen heatmap cell rather than only the coarse head ranking.

This keeps the main operational task focused on **population and interpretation**.

## 9. Interpretation Guardrails

- This phase supports **descriptive head-specialization analysis**, not causal attribution.
- Cross-family comparisons should be stated carefully: they can reveal different head patterns, but not by themselves prove why a model made a decision.
- `clip` is included as a language-supervised contrastive baseline, not as another self-supervised model.
- `mae` is included specifically because it gives a reconstruction-based self-supervised comparison within the same architecture-native CLS-token analysis family.
- Stronger explanation-heavy follow-up work would require methods like attention flow or relevance propagation rather than raw attention alone.

For the full technical caveats, use [Q3 Per-Head Attention Methodology](../reference/per_head_methodology.md).

## 10. Follow-up Directions

The most useful next steps stay inside the current Q3 framing:

- keep the primary-study cache set populated for `dinov2`, `dinov3`, `mae`, and `clip`
- treat `linear_probe` as an explicit control rather than a headline condition
- keep headline conclusions scoped to descriptive head-specialization analysis rather than causal attribution

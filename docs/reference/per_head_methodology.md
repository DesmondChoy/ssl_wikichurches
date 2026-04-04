# Q3 Per-Head Attention Methodology

This document explains how the project currently computes per-head attention maps for Q3, why that implementation is a defensible baseline, where it departs from architecture-native behavior, and which alternatives would be stronger for explanation-heavy claims.

The goal of Q3 is narrow and specific: identify whether individual attention heads align with expert architectural annotations, and whether some heads align better than others. That is not the same as proving that a head causally drove a prediction. This distinction matters throughout the note.

> **Related documents:**
> - [Project Proposal](../core/project_proposal.md) — Research questions and overall design
> - [Attention Methods](../research/attention_methods.md) — High-level attention extraction background
> - [Metrics Methodology](metrics_methodology.md) — IoU, Coverage, MSE, KL, and EMD definitions
> - [Per-Head Attention Visualization](../enhancements/per_attention_head.md) — Product and UI-oriented enhancement note

---

## Table of Contents

1. [Method Summary](#method-summary)
2. [What the Current Implementation Actually Does](#what-the-current-implementation-actually-does)
3. [Model-Specific Extraction Rules](#model-specific-extraction-rules)
4. [Metric and Ranking Pipeline](#metric-and-ranking-pipeline)
5. [Why This Is a Reasonable Baseline](#why-this-is-a-reasonable-baseline)
6. [Alternatives We Did Not Implement](#alternatives-we-did-not-implement)
7. [Limitations and Interpretation Guardrails](#limitations-and-interpretation-guardrails)
8. [Recommended Wording for Results](#recommended-wording-for-results)
9. [Source Code References](#source-code-references)
10. [External References](#external-references)

---

## Method Summary

The current Q3 pipeline uses **raw post-softmax self-attention weights** returned by the underlying Transformer implementation, isolates one head at one layer, converts that head's patch-level attention into an image-space heatmap, and then computes the same alignment metrics used elsewhere in the project.

In plain language:

1. We ask the model for per-layer attention tensors.
2. We pick one layer and one head.
3. We convert that head into a spatial map.
4. We compare that map against expert annotations with IoU, Coverage, MSE, KL, and EMD.
5. We rank heads by metric, image by image, then aggregate those rankings.

This is a **descriptive head-specialization analysis**. It is not a full causal attribution method.

---

## What the Current Implementation Actually Does

### 1. We use standard attention outputs from the model API

The project reads attention tensors from the model outputs exposed by Hugging Face Transformers. The official model-output documentation defines `attentions` as one tensor per layer with shape:

`(batch_size, num_heads, sequence_length, sequence_length)`

and states that these are the attention weights **after the attention softmax**.

That matters for two reasons:

- We are analyzing the same attention object that standard head-visualization tools expose.
- We are not analyzing pre-softmax scores, value vectors, or output-projection contributions.

Source:
- [Hugging Face Transformers model outputs](https://huggingface.co/docs/transformers/main/en/main_classes/output)

### 2. For CLS-token models, we extract one head's CLS-to-patch row

For `dinov2`, `dinov3`, `mae`, and ViT-based `clip`, the project treats per-head attention as:

`A[l, h, cls, patch]`

where:

- `l` is the chosen layer
- `h` is the chosen head
- `cls` is the class token query
- `patch` ranges over image patches after removing special tokens

The local implementation does this in two equivalent ways:

- `extract_cls_attention(..., head_indices=[h])`, which selects one head and then applies the same logic used for fused CLS attention
- `get_per_head_attention(...)`, which returns all heads at once and could also be indexed downstream

In the precompute pipeline, the project currently uses the first form:

- [`extract_cls_attention()`](/Users/desmondchoy/Projects/ssl_wikichurches/src/ssl_attention/attention/cls_attention.py#L72)
- per-head cache generation in [`generate_attention_cache.py`](/Users/desmondchoy/Projects/ssl_wikichurches/app/precompute/generate_attention_cache.py#L344)

This is not the only possible implementation, but for CLS-token models it is a straightforward and standard one.

### 3. We explicitly skip register tokens where the architecture uses them

For DINO-family models with register tokens, the project removes:

- the CLS token
- any register tokens

before interpreting the remaining values as patch attention.

This matches the documented token layout for DINOv2 and DINOv3:

- DINOv2 docs describe one image-level `CLS` token plus local patch embeddings.
- DINOv3 docs describe one `CLS` token, register tokens, and patch tokens, and expose `num_register_tokens`.

Sources:
- [DINOv2 docs](https://huggingface.co/docs/transformers/model_doc/dinov2)
- [DINOv3 docs](https://huggingface.co/docs/transformers/en/model_doc/dinov3)

### 4. For SigLIP and SigLIP2, we use a patch-attention proxy rather than the model's pooling head

For `siglip` and `siglip2`, the project does **not** have a CLS-token path analogous to DINO, MAE, or ViT-CLIP. Instead, it computes a per-head proxy:

`S[l, h, patch] = mean_q A[l, h, q, patch]`

That is, for one head, it averages how much each patch is attended to by all query positions. In the code this is implemented as the mean across rows of the selected head's attention matrix:

- [`extract_mean_attention()`](/Users/desmondchoy/Projects/ssl_wikichurches/src/ssl_attention/attention/cls_attention.py#L249)

This choice is practical, but it is not architecture-native. The official SigLIP and SigLIP2 implementations use a **learned multi-head attention pooling head** on top of the vision encoder:

- SigLIP: `SiglipMultiheadAttentionPoolingHead`
- SigLIP2: `Siglip2MultiheadAttentionPoolingHead`

Sources:
- [SigLIP source in Transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/modeling_siglip.py)
- [SigLIP2 source in Transformers](https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip2/modeling_siglip2.py)
- [SigLIP docs](https://huggingface.co/docs/transformers/en/model_doc/siglip)
- [SigLIP2 docs](https://huggingface.co/docs/transformers/main/en/model_doc/siglip2)

So the right way to describe the current method is:

- **CLS-token models:** raw per-head CLS-to-patch attention
- **SigLIP family:** per-head mean-attention proxy derived from the attention tensor

### 5. We convert patch attention into image-space heatmaps before scoring

Once the project has a per-head patch vector, it:

1. reshapes it to the patch grid
2. upsamples it to `224 x 224`
3. applies per-sample min-max normalization to `[0, 1]`
4. stores the result in the attention cache as `{method}_head{idx}`

This means the Q3 metrics are computed on **normalized image-space heatmaps**, not directly on the raw patch-probability vector.

Code paths:

- [`attention_to_heatmap()`](/Users/desmondchoy/Projects/ssl_wikichurches/src/ssl_attention/attention/cls_attention.py#L154)
- per-head caching in [`generate_attention_cache.py`](/Users/desmondchoy/Projects/ssl_wikichurches/app/precompute/generate_attention_cache.py#L344)

This is useful for visual consistency and for reusing the existing metric pipeline, but it is also one of the choices that should be made explicit in any methodology writeup.

---

## Model-Specific Extraction Rules

The current implementation supports per-head Q3 only where the method maps cleanly onto head-indexed self-attention tensors:

| Model family | Per-head method used | Current rationale | Architecture-native caveat |
|-------------|----------------------|-------------------|----------------------------|
| DINOv2 | `cls` | Uses CLS-to-patch attention for one head | Good architectural fit |
| DINOv3 | `cls` | Uses CLS-to-patch attention for one head, skips registers | Good architectural fit |
| MAE | `cls` | Uses CLS-to-patch attention for one head | Good architectural fit |
| CLIP (ViT) | `cls` | Uses class-token attention path | Good architectural fit |
| SigLIP | `mean` | Uses per-head mean received attention as a proxy | Not the same as the learned pooling head |
| SigLIP2 | `mean` | Uses per-head mean received attention as a proxy | Not the same as the learned pooling head |
| ResNet-50 | none | No transformer attention heads | Not applicable |

Architecture corroboration:

- DINOv2 docs explicitly describe a whole-image `CLS` token plus patch embeddings.
- DINOv3 docs explicitly describe a `CLS` token, register tokens, and patch tokens.
- MAE official code appends `cls_token` before the Transformer blocks.
- OpenAI CLIP's official code defines `class_embedding` and prepends it to the patch sequence.
- SigLIP and SigLIP2 official code instantiate learned attention pooling heads on the vision side.

Sources:

- [DINOv2 docs](https://huggingface.co/docs/transformers/model_doc/dinov2)
- [DINOv3 docs](https://huggingface.co/docs/transformers/en/model_doc/dinov3)
- [MAE official code](https://github.com/facebookresearch/mae/blob/main/models_mae.py)
- [OpenAI CLIP code](https://github.com/openai/CLIP/blob/main/clip/model.py)
- [SigLIP source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/modeling_siglip.py)
- [SigLIP2 source](https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip2/modeling_siglip2.py)

---

## Metric and Ranking Pipeline

Once a per-head heatmap is cached, the project reuses the existing metric framework.

### Per-head metrics

For every supported `(model, variant, layer, head, image)` combination, the project computes:

- `IoU` at the standard percentile set
- `Coverage` on the raw heatmap
- `MSE`, `KL`, and `EMD` against the Gaussian soft target used elsewhere in the project

Implementation:

- [`compute_per_head_metrics_for_model()`](/Users/desmondchoy/Projects/ssl_wikichurches/app/precompute/generate_metrics_cache.py#L778)

### Storage layout

The project writes three dedicated Q3 tables:

- `head_image_metrics`
- `head_summary_metrics`
- `head_feature_metrics`

This separation is deliberate: per-head Q3 is not folded into the generic Q1/Q2 tables.

### Ranking rule

Heads are ranked **within each image** and **within each metric**:

- higher is better for `iou` and `coverage`
- lower is better for `mse`, `kl`, and `emd`

The project then aggregates:

- `mean_rank`
- `mean_score`
- `std_score`
- `top1_count`
- `top3_count`
- `image_count`

This is important methodologically because it avoids collapsing all metrics into a single composite "best head" score.

### Threshold-free metric convention

The threshold-free metrics are stored under the shared reference percentile key `90` so the schema remains uniform. This is a storage convention, not a claim that those metrics depend on the percentile slider.

Related code:

- [`rank_head_scores()`](/Users/desmondchoy/Projects/ssl_wikichurches/app/precompute/generate_metrics_cache.py#L410)
- Q3 query methods in [`metrics_service.py`](/Users/desmondchoy/Projects/ssl_wikichurches/app/backend/services/metrics_service.py#L1025)

---

## Why This Is a Reasonable Baseline

### It matches what standard attention-visualization tools inspect

Tools like [BertViz](https://github.com/jessevig/bertviz) expose a `head_view` over the model's attention weights. That is strong evidence that "inspect one layer and one head from the returned attention tensor" is a mainstream baseline for head-level analysis.

This matters for Q3 because the research question is about **head specialization**, not necessarily about faithful end-to-end attribution.

### It uses officially exposed outputs

The attention tensor is already returned by the standard model API, with no need to patch model internals. That makes the method:

- reproducible
- architecture-aware
- easy to cache across all images
- consistent across DINOv2, DINOv3, MAE, CLIP, SigLIP, and SigLIP2

### It keeps the unit of analysis aligned with the question

Q3 asks whether **individual heads** specialize. A raw per-head map preserves the unit of analysis directly:

- one layer
- one head
- one image

By contrast, methods that immediately mix layers, gradients, or relevance propagation are often stronger explanation methods, but they are also less direct as a description of one head's own pattern.

---

## Alternatives We Did Not Implement

The current method is not the only way to define per-head attention analysis.

### 1. Direct per-head extraction without the fusion helper

For CLS-token models, our current implementation uses `extract_cls_attention(..., head_indices=[h])`. That is equivalent in spirit to using `get_per_head_attention(...)` and then indexing head `h` afterwards.

This is a code-path choice, not a methodology difference.

### 2. Per-head rollout or attention flow

[Abnar and Zuidema (2020)](https://arxiv.org/abs/2005.00928) argue that raw attention becomes unreliable as explanation evidence because information gets mixed across layers. They propose:

- **attention rollout**
- **attention flow**

These methods propagate attention through the network and, compared to raw attention, yield higher correlation with ablation- and gradient-based importance estimates.

Why we did not use them for the current Q3 baseline:

- they mix information across layers and residual paths
- a "per-head rollout" is possible in principle but is not the most direct description of one head's local pattern
- rollout is better suited to "effective attention through the network" than "what does head 7 in layer 10 look like on its own?"

### 3. Gradient attention rollout

The widely used [jacobgil/vit-explain](https://github.com/jacobgil/vit-explain) repo implements:

- Attention Rollout
- Gradient Attention Rollout

and explicitly describes gradient rollout as an improvement when class-specific explainability is needed.

This is a strong candidate for a future robustness check if Q3 needs to move from descriptive head analysis toward prediction-faithful explanation.

### 4. Relevance propagation and Transformer-specific attribution

[Chefer et al. (2021)](https://arxiv.org/abs/2012.09838) argue that existing methods either use raw attention maps or heuristic propagation, and propose a relevance-propagation method that benchmarks better than prior explainability approaches.

This is the strongest "best practice" alternative for explanation-heavy claims in a methodology section, especially if the claim shifts from:

- "this head aligns with annotations"

to:

- "this head was responsible for the model's use of those regions"

### 5. Query-key or neuron-level analysis

BertViz documents that its more detailed neuron view requires access to query and key vectors, which are **not returned through the standard Hugging Face API**.

This is useful context because it explains why many practical analysis pipelines, including ours, start from the returned attention weights rather than deeper internal tensors.

### 6. Architecture-native pooling-head analysis for SigLIP and SigLIP2

The most important model-specific alternative is on the SigLIP side:

- instead of using per-head mean received attention
- instrument the learned multi-head attention pooling head itself

That would be more faithful to the architecture's final image representation, but it would require custom instrumentation beyond the current generic per-layer self-attention pipeline.

---

## Limitations and Interpretation Guardrails

### 1. Raw attention is not the same as faithful explanation

This is the central limitation.

[Abnar and Zuidema (2020)](https://arxiv.org/abs/2005.00928) explicitly state that attention weights become unreliable explanation probes once information is mixed across layers. [Chefer et al. (2021)](https://arxiv.org/abs/2012.09838) make a similar point from the perspective of transformer-specific explainability.

So Q3 results should be interpreted as:

- **alignment of per-head attention patterns**

not as:

- **proof that a head causally drove the final prediction**

### 2. The current method is single-layer and weight-space only

The implementation analyzes one layer at a time and only uses the attention weights. It does not model:

- residual accumulation across layers
- value vectors
- output projection mixing
- MLP contributions

That is a legitimate simplification for head specialization, but it omits real pathways by which information affects downstream representations.

### 3. SigLIP and SigLIP2 use a heuristic proxy

For the SigLIP family, the current per-head map is **not** the model's learned pooling head. It is a patch-attention proxy derived from the self-attention matrix.

This is useful and comparable across images, but claims about "the best SigLIP head" should be framed more cautiously than equivalent claims for DINOv2, DINOv3, MAE, or ViT-CLIP.

### 4. Heatmap interpolation and normalization change the raw token-space object

The raw per-head patch weights are:

- reshaped to a grid
- upsampled to `224 x 224`
- min-max normalized

This is helpful for overlay visualization and metric reuse, but it means the scored object is an image-space heatmap, not the untouched token-space probability distribution.

### 5. Q3 rankings are metric-specific by design

This is a feature, not a bug, but it should be made explicit.

A head can rank highly by IoU and poorly by EMD, or vice versa, because these metrics encode different notions of alignment:

- overlap
- energy concentration
- pointwise mismatch
- distribution mismatch
- transport cost

The project intentionally does **not** combine them into one "true best head" scalar.

### 6. Unsupported methods are omitted for methodological clarity

The project excludes per-head Q3 for:

- `rollout`
- `gradcam`
- `resnet50`

This is partly architectural and partly methodological:

- `gradcam` and `resnet50` are not transformer-head methods
- `rollout` is a multi-layer post hoc aggregation method, so excluding it keeps Q3 focused on directly inspectable heads

---

## Recommended Wording for Results

For papers, presentations, and internal writeups, the safest wording is:

> We evaluate head specialization using raw per-head attention maps derived from the model's post-softmax self-attention tensors. For CLS-token models, the per-head map is the selected head's CLS-to-patch attention at a fixed layer. For SigLIP-family models, which use a learned attention pooling head rather than a CLS-token attention path, we use mean received patch attention from the selected head as a visualization proxy. These per-head heatmaps are compared against expert annotations using IoU, Coverage, MSE, KL, and EMD. We interpret the resulting rankings as evidence of alignment patterns, not as a full causal explanation of model decisions.

That wording is accurate to the implementation and appropriately conservative.

---

## Source Code References

- Per-head CLS extraction:
  - [`src/ssl_attention/attention/cls_attention.py`](/Users/desmondchoy/Projects/ssl_wikichurches/src/ssl_attention/attention/cls_attention.py)
- Rollout implementation:
  - [`src/ssl_attention/attention/rollout.py`](/Users/desmondchoy/Projects/ssl_wikichurches/src/ssl_attention/attention/rollout.py)
- Per-head cache generation:
  - [`app/precompute/generate_attention_cache.py`](/Users/desmondchoy/Projects/ssl_wikichurches/app/precompute/generate_attention_cache.py)
- Per-head metric computation and ranking:
  - [`app/precompute/generate_metrics_cache.py`](/Users/desmondchoy/Projects/ssl_wikichurches/app/precompute/generate_metrics_cache.py)
- Q3 backend queries:
  - [`app/backend/services/metrics_service.py`](/Users/desmondchoy/Projects/ssl_wikichurches/app/backend/services/metrics_service.py)
- Q3 UI:
  - [`app/frontend/src/components/metrics/Q3HeadAnalysis.tsx`](/Users/desmondchoy/Projects/ssl_wikichurches/app/frontend/src/components/metrics/Q3HeadAnalysis.tsx)

---

## External References

- Hugging Face Transformers. [Model outputs](https://huggingface.co/docs/transformers/main/en/main_classes/output).
- Hugging Face Transformers. [DINOv2 docs](https://huggingface.co/docs/transformers/model_doc/dinov2).
- Hugging Face Transformers. [DINOv3 docs](https://huggingface.co/docs/transformers/en/model_doc/dinov3).
- Facebook Research. [MAE official implementation](https://github.com/facebookresearch/mae/blob/main/models_mae.py).
- Hugging Face Transformers. [CLIP docs](https://huggingface.co/docs/transformers/en/model_doc/clip).
- OpenAI. [CLIP official implementation](https://github.com/openai/CLIP/blob/main/clip/model.py).
- Hugging Face Transformers. [SigLIP docs](https://huggingface.co/docs/transformers/en/model_doc/siglip).
- Hugging Face Transformers. [SigLIP official implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip/modeling_siglip.py).
- Hugging Face Transformers. [SigLIP2 docs](https://huggingface.co/docs/transformers/main/en/model_doc/siglip2).
- Hugging Face Transformers. [SigLIP2 official implementation](https://github.com/huggingface/transformers/blob/main/src/transformers/models/siglip2/modeling_siglip2.py).
- Caron et al. (2021). [Emerging Properties in Self-Supervised Vision Transformers](https://arxiv.org/abs/2104.14294).
- Abnar and Zuidema (2020). [Quantifying Attention Flow in Transformers](https://arxiv.org/abs/2005.00928).
- Chefer, Gur, and Wolf (2021). [Transformer Interpretability Beyond Attention Visualization](https://arxiv.org/abs/2012.09838).
- Vig. [BertViz](https://github.com/jessevig/bertviz).
- Gildenblat. [Explainability for Vision Transformers](https://github.com/jacobgil/vit-explain).

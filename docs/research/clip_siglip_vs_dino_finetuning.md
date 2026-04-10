# Why CLIP/SigLIP Improved But DINO Didn't: A Deep Analysis

> **Research brainstorm — April 2026**
> **Status:** Exploratory. Numbers sourced from `fine_tuning_primary_20260327` experiment, layer 11, IoU p90.

---

## The Core Finding

Fine-tuning on architectural style classification produces dramatically different Δ IoU results depending on the model family:

| Model | Frozen IoU (p90) | Fine-tuned IoU (p90) | Δ IoU | Cohen's d | Significant |
|-------|-----------------|---------------------|-------|-----------|-------------|
| **CLIP** | 0.0181 | 0.0745 | **+0.0564** | 1.005 | ✅ |
| **SigLIP2** | 0.0220 | 0.0519 | **+0.0299** | 0.781 | ✅ |
| **SigLIP** | 0.0364 | 0.0618 | **+0.0254** | 0.604 | ✅ |
| **MAE** | 0.0702 | 0.0988 | **+0.0286** | 0.413 | ✅ |
| **DINOv2** | 0.0816 | 0.0758 | **-0.0058** | -0.184 | ❌ |
| **DINOv3** | 0.1327 | 0.1321 | **-0.0007** | -0.017 | ❌ |

The split is stark: CLIP/SigLIP family and MAE all show significant positive Δ IoU; both DINO variants show essentially zero (or very slightly negative) change.

---

## Part 1: Why CLIP and SigLIP Improved

### 1.1 The Pretraining Objective Explanation

**CLIP** uses global image-text contrastive learning (InfoNCE loss). The model learns to match an entire image to an entire text description. There is no gradient pressure to attend to any particular *spatial region* — only to produce a single image embedding that matches the corresponding caption embedding. Attention can be smeared diffusely across the whole image and the objective would be satisfied as long as the global representation is correct.

Evidence: CLIP frozen IoU at p90 is just 0.0181 — essentially at the level of random chance. This is the lowest of all models, consistent with maximally diffuse, spatially unstructured attention.

**SigLIP** replaces the softmax contrastive loss with a sigmoid (pairwise binary) loss. This avoids the need to normalize across a batch but preserves the same global image-text alignment objective. The lack of a CLS token (mean attention is used as a proxy) compounds the spatial diffuseness. SigLIP frozen IoU = 0.0364 (SigLIP) / 0.0220 (SigLIP2) — both very low.

When you fine-tune these models on a 4-class classification task, the classification objective sends a clear spatial signal for the first time: "to distinguish Gothic from Romanesque, you must attend to pointed arches vs. round ones." The CLS token gradient is forced to route toward spatially discriminative patches. The IoU improvement is the model *learning to look in the right places* for the first time.

### 1.2 CLIP vs. SigLIP: Why CLIP Improves More

CLIP's improvement (d = 1.005) is substantially larger than SigLIP (d = 0.604) or SigLIP2 (d = 0.781). A few candidate explanations:

**Hypothesis A — Language grounding helps spatial alignment.** CLIP was trained on image-text pairs where captions often describe visible features ("a church with tall pointed arches"). Even if CLIP's attention is globally diffuse pre-FT, its representations may have already organized around "nameable" features at the token level, not the attention level. Classification fine-tuning then only needs to "unlock" this spatial organization by routing CLS attention toward already-named-feature-aware patches.

**Hypothesis B — CLS token vs. mean attention.** CLIP has a CLS token whose attention can be sharply redirected by the classification gradient. SigLIP/SigLIP2 use mean attention (averaged over all patch-to-patch attention), which is inherently more diffuse and harder to concentrate. The classification gradient has less leverage on mean attention.

**Hypothesis C — SigLIP's sigmoid loss preserves more pretraining geometry.** The sigmoid loss doesn't push different images apart as aggressively as softmax InfoNCE. Fine-tuning on classification may "override" CLIP representations more completely while SigLIP's representations are more resistant to reshaping.

**Testing Hypothesis A vs. B:** If we could extract the CLS token attention from SigLIP (it doesn't have a natural one, but we could probe the pooling head), and if that shows similar improvement magnitude as CLIP, it would support Hypothesis A. Otherwise Hypothesis B dominates.

### 1.3 LoRA vs. Full Fine-tuning: Model-Specific Patterns

| Model | LoRA Δ | Full Δ | LoRA/Full ratio |
|-------|--------|--------|-----------------|
| CLIP | +0.0275 | +0.0564 | 0.49 |
| SigLIP2 | +0.0242 | +0.0299 | 0.81 |
| SigLIP | +0.0196 | +0.0254 | 0.77 |
| MAE | +0.0311 | +0.0286 | 1.09 |

CLIP uniquely needs full fine-tuning to achieve its full improvement — LoRA only gets it half-way. This is consistent with Biderman et al. (2024): full fine-tuning allows "holistic reshaping" of representations (rank 10-100× higher than LoRA), while LoRA makes targeted changes. CLIP's attention restructuring appears to require holistic reshaping.

MAE is the opposite: LoRA slightly *outperforms* full FT. This may indicate that full fine-tuning begins to overfit or catastrophically forget MAE's spatially rich reconstruction representations, while LoRA's conservative adaptation is sufficient.

---

## Part 2: Why DINO Didn't Improve

### 2.1 The Ceiling and the Pretraining Effect

DINOv3 frozen IoU = 0.1327 — the **highest of all models by a wide margin**. DINOv2 frozen IoU = 0.0816 — also higher than any of the CLIP/SigLIP family before fine-tuning. Fine-tuning takes DINOv3 from 0.1327 to 0.1321 (-0.0007). This near-zero change is not absence of fine-tuning signal — it's evidence that fine-tuning *found nothing to improve*.

DINO's self-distillation objective (student-teacher consistency across views) creates a pressure for the CLS token to aggregate spatially coherent, semantically consistent representations. Caron et al. (2021) documented this: DINO heads produce attention maps that resemble semantic segmentation masks without any supervision. The attention is already concentrated on semantically relevant objects.

When the classification gradient hits a model whose attention is already spatially coherent and semantically grounded, it finds that the existing attention pattern already satisfies "where to look for discriminative features." The gradient signal is absorbed by the classification head rather than propagating back to reshape attention.

### 2.2 The Spectral Structure Argument

Park & Kim (2023) showed that contrastive learning (CL) methods including DINO train attention for longer-range global patterns. The attention entropy in DINO's frozen state is already *low* — heads are concentrated on specific semantic regions. Classification fine-tuning has less entropy to compress; the information content of each attention head is already high.

This is structurally different from CLIP: CLIP's high-entropy diffuse attention has a large compression target, and classification fine-tuning is doing the compression. DINO's attention is already compressed.

### 2.3 The Counter-Intuitive Interpretation: DINO "Not Improving" Is the More Important Finding

**Counter-intuitive claim:** DINO's Δ IoU ≈ 0 is the *more impressive result*, not CLIP's large Δ IoU.

CLIP's large Δ IoU tells us: "CLIP needed classification supervision to develop expert-relevant spatial attention." This is a *dependency* — CLIP's spatial alignment is task-coupled and may not generalize beyond WikiChurches-trained style vocabulary.

DINO's Δ IoU ≈ 0 tells us: "DINO's attention alignment with expert annotations is a pretraining property, not a supervised learning outcome." This is *generalization without supervision* — a fundamentally stronger property.

A practical implication: on a completely different domain (say, medical images or industrial parts), DINO would bring its attention quality with it, while CLIP's fine-tuned spatial alignment might degrade significantly.

### 2.4 The DINOv3 Coverage Decrease Is Unexpected

DINOv3 shows a significant *decrease* in Coverage after full fine-tuning (Δ = -0.0049, significant). Coverage measures threshold-free attention-bbox overlap; a decrease means the fine-tuned model's attention distribution has moved *away* from bbox regions on average.

**Possible explanation:** DINOv3 with Gram anchoring has especially well-distributed attention across all semantic regions. Classification fine-tuning may concentrate attention on the *most discriminative* region per image (the primary style-defining feature), at the expense of coverage over all annotated features. This is a kind of "specificity-coverage tradeoff" where the fine-tuned model is more selective but less comprehensive.

---

## Part 3: Image-Level Analysis

### 3.1 High-Δ and Low-Δ Images for CLIP

From the per-image deltas in the Q2 results, CLIP full fine-tuning shows the following extreme cases:

**Largest positive Δ IoU (CLIP full, p90):**
- Q3580450: +0.239
- Q5886455: +0.223
- Q2443243: +0.209
- Q1710328: +0.207
- Q2858167: +0.204
- Q18785543: +0.173

**Negative Δ IoU cases (where CLIP fine-tuning hurt alignment):**
- Q5690257: -0.081
- Q1502706: -0.021
- Q3670352: -0.024
- Q1619579: -0.036

**Research questions for image-level investigation:**
- Do the high-Δ images share a style? (e.g., do they disproportionately represent Gothic — whose features like pointed arches are more "nameable" in CLIP's visual vocabulary?)
- Do the negative-Δ images show signs that fine-tuning pulled attention to a *different* diagnostic feature than the one annotated? (e.g., model attends to tower but the bbox marks a portal)
- Is there a correlation between frozen IoU and Δ IoU for CLIP? A negative correlation (models with lower frozen IoU gain more) would confirm a regression-to-mean effect. A positive correlation would suggest that fine-tuning amplifies models that were already partially aligned.

### 3.2 The CLIP Layer 10 vs. Layer 11 Non-Monotonic Finding

From `finetuning_results.md`: for a single image (Columned Portal feature), CLIP shows Layer 10 > Layer 11 after fine-tuning (FT IoU 0.208 at L10 vs 0.113 at L11). This non-monotonic pattern suggests:

- CLIP's fine-tuning may restructure attention such that peak feature alignment emerges in an earlier layer than the final layer
- The standard practice of only evaluating at Layer 11 may miss the peak for some images/features
- This could explain why CLIP's aggregate improvement is understated if Layer 11 is not actually the optimal layer post-FT

**Investigation needed:** Run layer-wise IoU across all 12 layers for CLIP frozen vs. fine-tuned, and find where peak IoU occurs per image. If post-FT the peak consistently shifts from Layer 11 to Layer 9/10, this is a significant finding about how CLIP's internal representations are reorganized.

### 3.3 Cross-Model Agreement on "Easy" vs. "Hard" Images

An underexplored angle: when DINO has high frozen IoU on an image, does CLIP also have high Δ IoU on that same image after fine-tuning? If yes, these are "easy" images where the expert annotations happen to cover visually prominent regions that any model recognizes. If no, CLIP and DINO are learning spatially different representations that happen to coincide with expert annotations via different mechanisms.

---

## Part 4: Intuitive vs. Counter-Intuitive Findings

### Intuitive Findings

| Finding | Why it's expected |
|---------|-------------------|
| CLIP improves with FT | CLIP's global objective provides no spatial pressure; FT adds it |
| Linear probe shows Δ = 0 exactly | Frozen backbone = no attention change; confirms experimental control |
| DINOv3 > DINOv2 frozen | DINOv3 adds Gram anchoring which improves dense feature structure |
| CLIP needs full FT more than LoRA | Global attention restructuring requires more parameter updates |
| MAE improves significantly | Reconstruction objective is spatially agnostic; FT adds localization |

### Counter-Intuitive Findings

| Finding | Why it's surprising | Explanation |
|---------|--------------------|----|
| CLIP fine-tuned (0.0745) still worse than DINOv3 frozen (0.1327) | FT should close the gap completely | DINO's pretraining creates a fundamentally different kind of spatial coherence that FT can't fully replicate |
| DINOv3 Coverage slightly *decreases* after full FT | More training should help | FT concentrates attention on most-discriminative region, trading coverage for precision |
| MAE LoRA slightly beats MAE full FT | More trainable params should mean more improvement | Full FT may overfit or cause forgetting of MAE's spatial reconstruction features |
| CLIP's large Cohen's d = 1.005 despite still being lowest IoU in absolute terms | Large effect, still bottom of leaderboard | CLIP starts from an extremely low baseline; the relative gain is large but absolute IoU remains modest |
| Both DINO variants essentially unaffected across all 3 fine-tuning strategies | Surely more parameters (full FT) should do *something* | Confirms spatial attention alignment is pretraining-baked, not fine-tuning-unlockable |

---

## Part 5: Deeper Hypotheses to Test

### H1: Language Grounding Hypothesis (CLIP specific)

CLIP's representations are organized around visual concepts that have text labels. Architectural features like "rose window," "pointed arch," or "buttress" are likely named in CLIP's training captions. The hypothesis: CLIP's *patch features* (not CLS attention) are already highly discriminative for these named features, but the CLS attention doesn't route to them until FT.

**Test:** Extract patch-level feature similarity between CLIP frozen patches and a text embedding of "pointed arch." Measure whether high-similarity patches overlap with bboxes. If yes, the information was always there at the patch level — FT just taught the CLS token to collect it.

### H2: Attention Entropy Hypothesis

DINOv3 frozen CLS attention has lower entropy (more concentrated) than CLIP. FT can only reduce entropy further; DINO is already near-minimum. CLIP starts high-entropy and FT compresses it.

**Test:** Measure Shannon entropy of CLS attention maps at layer 11, frozen vs. fine-tuned. Plot entropy vs. IoU. Prediction: strong negative correlation (lower entropy = higher IoU), with DINO already occupying the low-entropy region.

### H3: Style-Conditioned Improvement Hypothesis

CLIP's language grounding may make it better at improving IoU for Gothic churches (whose features — rose windows, flying buttresses — are more "nameable" in English captions) than for Romanesque churches (whose features — rounded arches, arcading — are more subtle). Fine-tuning on 4-class classification would amplify this.

**Test:** Break down Δ IoU by architectural style (Romanesque / Gothic / Renaissance / Baroque). Check if CLIP's Δ IoU is significantly higher for Gothic than Romanesque. Compare with DINO's per-style frozen IoU distribution to see if DINO already handles all styles equally.

### H4: Feature Type Sensitivity Hypothesis

CLIP may improve more for some feature categories than others. "Towers" and "portals" are highly salient and spatially compact. "Arcading" or "moldings" are spatially distributed and harder to localize. FT may concentrate CLIP attention on the most salient features, improving IoU for towers/portals but potentially hurting for spatially distributed features.

**Test:** Per-feature-type breakdown of Δ IoU for CLIP. Identify which of the 106 feature types show the largest positive and negative Δ.

---

## Part 6: Proposed App Features

### 6.1 Image-Level Δ IoU Explorer

A sortable gallery view filtered by model showing each image with its per-image Δ IoU. Users could:
- Sort by highest/lowest Δ IoU for any model
- Filter by architectural style to test H3
- Click through to the full attention comparison for extreme cases
- See a scatter plot: frozen IoU (x) vs. Δ IoU (y) per image

**Backend needed:** The per-image deltas are already in `q2_metrics_analysis.json`. A new API endpoint `GET /api/metrics/q2/per_image?model=clip&strategy=full&metric=iou&percentile=90` would return them. The frontend needs a new page or panel.

### 6.2 Attention Entropy Visualization

Overlay or panel showing the attention *spread* (entropy or top-k patch concentration) before and after FT. Users could observe directly whether fine-tuning concentrates CLIP attention and whether DINO was already concentrated.

**Implementation:** Compute attention entropy during precompute (`-sum(p log p)` over the 196 patch attention weights). Store in HDF5 alongside attention maps. Expose via API. Render as a single scalar badge next to each attention heatmap.

### 6.3 Layer-Sweep IoU Plot (Post-FT)

Currently the layer progression view shows frozen IoU per layer. Extend it to show both frozen and fine-tuned IoU curves on the same chart, for each strategy. This would expose the non-monotonic CLIP Layer 10 > 11 finding at scale.

**Backend:** The `analyze_q2_metrics.py` script currently runs at Layer 11. Extend it to compute per-layer Δ IoU by running across layers 0-11. Store layer-sweep results in a separate `q2_layer_sweep.json`. Add `GET /api/metrics/q2/layer_sweep?model=clip&strategy=full`.

### 6.4 Cross-Model Image Agreement Heatmap

A matrix visualization: rows = 139 images, columns = models, cell color = Δ IoU. Group rows by style. This would make the H3 style-conditioned hypothesis visually obvious, and would reveal which images are "universally easy" (high Δ IoU across all models) vs. "model-specific" (high for CLIP, low for DINO).

**Frontend:** Table or heatmap component. Similar in spirit to the leaderboard but image-centric rather than model-centric.

### 6.5 Patch-Feature Text Similarity View (CLIP-specific)

For CLIP specifically: show a text query input where the user can type a feature name (e.g., "rose window") and see a heatmap of which patches have the highest cosine similarity to that text embedding. Compare this to the corresponding bbox annotation.

This would directly test H1: if CLIP frozen patch features already align with the bbox via text query, the pretraining knew but the CLS token didn't reflect it.

**Backend:** `POST /api/attention/{image_id}/text_similarity` — input: text query, model=clip; output: 196-dim similarity vector. Requires loading CLIP's text encoder (currently only the vision encoder is used).

### 6.6 Per-Feature-Type Δ IoU Breakdown

Extend the existing `FeatureBreakdown.tsx` component (which shows frozen IoU per feature type) to also show a frozen vs. fine-tuned comparison bar. Highlight feature types where Δ IoU is largest/smallest. Sort by Δ IoU.

**Backend:** `GET /api/metrics/{model}/feature_breakdown?strategy=full` — already returns per-feature data; add Δ metrics.

---

## Part 7: Verification Checklist

Before treating any of these findings as publishable, the following should be confirmed:

- [ ] Run layer-sweep IoU for CLIP to confirm Layer 10 > 11 finding holds across more than one image
- [ ] Per-style breakdown of Δ IoU to test H3 (Gothic vs. Romanesque differential improvement)
- [ ] Per-feature-type Δ IoU to identify which features drive CLIP's improvement
- [ ] Attention entropy measurement to test H2 directly
- [ ] Verify that CLIP's per-image negative-Δ cases (Q5690257 etc.) have a systematic explanation (e.g., confusion between annotated region and other discriminative cue)
- [ ] Check whether CLIP post-FT IoU correlated with DINOv3 frozen IoU across images (H5: shared "easy images")

---

## References

All citations reference papers already cited in the project literature.

- Caron et al. (2021). Emerging Properties in Self-Supervised Vision Transformers. ICCV 2021.
- Park, N., et al. (2023). What Do Self-Supervised Vision Transformers Learn? ICLR 2023.
- Biderman, S., et al. (2024). LoRA Learns Less and Forgets Less. TMLR 2024.
- Darcet, T., et al. (2024). Vision Transformers Need Registers. ICLR 2024.
- Naseer, M. M., et al. (2021). Intriguing Properties of Vision Transformers. NeurIPS 2021. https://arxiv.org/abs/2105.10497
- Park, J., and Kim, J. (2022). How Do Vision Transformers Work? ICLR 2022. https://arxiv.org/abs/2202.06709

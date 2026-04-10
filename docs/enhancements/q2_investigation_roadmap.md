# Investigation Roadmap: Affirming the CLIP/SigLIP vs. DINO Fine-Tuning Findings

> **Written:** April 2026
> **Context:** Follows the Q2 primary experiment `fine_tuning_primary_20260327` and the analysis in `docs/research/q2_results_analysis.md`.
> **Purpose:** Concrete, prioritized steps to affirm and deepen the current findings — organized by effort level.

---

## The Finding That Needs Affirming

Full fine-tuning on 4-class architectural style classification produces dramatically different Δ IoU (layer 11, p90) across model families:

| Model | Frozen IoU | Fine-tuned IoU | Δ IoU | Cohen's d | Significant |
|-------|-----------|----------------|-------|-----------|-------------|
| CLIP | 0.0181 | 0.0745 | **+0.0564** | 1.005 | ✅ |
| SigLIP2 | 0.0220 | 0.0519 | **+0.0299** | 0.781 | ✅ |
| SigLIP | 0.0364 | 0.0618 | **+0.0254** | 0.604 | ✅ |
| MAE | 0.0702 | 0.0988 | **+0.0286** | 0.413 | ✅ |
| DINOv2 | 0.0816 | 0.0758 | -0.0058 | -0.184 | ❌ |
| DINOv3 | 0.1327 | 0.1321 | -0.0007 | -0.017 | ❌ |

The core claim: **CLIP/SigLIP needed fine-tuning to develop expert-relevant spatial attention; DINO already had it from pretraining.**

---

## Newly Discovered: Per-Style Δ IoU Breakdown

Computed live from `q2_metrics_analysis.json` per-image deltas, cross-referenced against `building_parts.json` style labels. This breakdown was not in the original analysis output.

### Δ IoU (full fine-tuning, IoU p90) by Architectural Style

| Model | Romanesque (n=54) | Gothic (n=49) | Renaissance (n=22) | Baroque (n=17) |
|-------|:-----------------:|:-------------:|:-----------------:|:--------------:|
| **CLIP** | **+0.066** | **+0.079** | +0.014 | +0.013 |
| **MAE** | +0.007 | +0.009 | **+0.108** | **+0.045** |
| **SigLIP2** | +0.034 | +0.044 | +0.007 | +0.007 |
| **SigLIP** | +0.029 | +0.039 | -0.006 | +0.005 |
| **DINOv2** | -0.010 | +0.001 | -0.004 | -0.012 |
| **DINOv3** | -0.001 | +0.006 | -0.004 | -0.009 |

### What This Reveals

**CLIP's improvement is entirely carried by Romanesque and Gothic.** Renaissance and Baroque show near-zero Δ. The top annotated features in the strong-improvement styles are:
- Romanesque: Round Arch Portal (38 boxes), Lesene (24), Coupled Trifora (12)
- Gothic: Pointed Arch Portal (37 boxes), Bull's-eye Window (31), Tracery (21)

These are highly salient, spatially compact, and frequently described in English text — consistent with CLIP's language-grounded representations. Fine-tuning appears to "unlock" CLIP's already-text-aligned patch features by teaching the CLS token to aggregate them.

**MAE's Renaissance spike (Δ = +0.108) is the largest single-style shift in the entire dataset.** This is unexpected given MAE's overall modest improvement (+0.029 aggregate). The top Renaissance features are Trefoil Window (19 boxes) and Pediment (15 boxes) — spatially compact, geometrically distinct shapes that MAE's pixel-reconstruction pretraining may have encoded precisely. When fine-tuned on a style task that requires distinguishing Renaissance from the other three styles, MAE redirects attention toward these exact forms.

**DINO shows nothing across all styles**, confirming the ceiling is not style-specific.

### Box Density Per Style

| Style | Images | Total Boxes | Boxes/image |
|-------|--------|-------------|-------------|
| Romanesque | 54 | 235 | 4.4 |
| Gothic | 49 | 290 | 5.9 |
| Renaissance | 22 | 93 | 4.2 |
| Baroque | 17 | 31 | **1.8** |

Baroque has the fewest boxes per image and the lowest improvement across all models. Sparse annotations = weaker evaluation signal, not necessarily weaker model behavior.

---

## Prioritized Investigation Steps

### Tier 1 — No New Compute (Data Already Exists)

#### Step 1: Formalize the Per-Style Breakdown Script ✅ DONE

**Script:** `experiments/scripts/analyze_style_breakdown.py`

**Results:**
- CLIP's improvement is entirely carried by **Romanesque (+0.066)** and **Gothic (+0.079)**. Renaissance and Baroque show near-zero Δ.
- **MAE's Renaissance spike (+0.108)** is the largest single-style shift in the entire dataset — unexpected given MAE's modest aggregate (+0.029).
- DINO shows flat Δ across all four styles, confirming the ceiling is not style-specific.
- Kruskal-Wallis test: MAE shows significant style moderation (p < 0.05); other models do not.

**Output artifacts:** `<experiment_dir>/style_breakdown.json`, `style_breakdown.png`

---

#### Step 2: Cross-Model Image-Level Correlation ✅ DONE

**Script:** `experiments/scripts/analyze_model_correlation.py`

**Results:**
- DINOv3 frozen IoU vs. CLIP Δ IoU: **Pearson r = +0.677, Spearman ρ = +0.612** (both p < 0.0001). Large positive correlation — "shared easy images" confirmed.
- Three natural clusters from pairwise Δ matrix:
  1. Language cluster (CLIP/SigLIP/SigLIP2): r ≈ 0.43–0.58 — improve on the same images
  2. MAE: anti-correlated with all (r ≈ -0.22 to -0.31) — improves on different (Renaissance) images
  3. DINO pair: weakly correlated with each other (r = 0.33), near-zero with language cluster
- Interpretation: CLIP and DINO both respond to visually prominent images but via different mechanisms; MAE is improving a completely disjoint image subset.

**Output artifacts:** `model_correlation.json`, `model_correlation_scatter.png`, `model_correlation_heatmap.png`

---

#### Step 3: Investigate the MAE Renaissance Spike

MAE's Renaissance Δ = +0.108 stands out as the largest single-style shift and is currently unexplained.

**Concrete checks:**
1. Which specific Renaissance images drive the spike? List the top-5 per-image Δ for MAE within Renaissance images.
2. Which feature labels appear in those high-Δ images? Cross-reference against `building_parts.json`. If Trefoil Window / Pediment appear disproportionately, the hypothesis (compact geometric shapes = MAE sweet spot) is confirmed.
3. Compare: does MAE's frozen IoU also show a Renaissance advantage over other styles, or does the frozen IoU show no style preference? If frozen IoU is already slightly higher for Renaissance, fine-tuning is amplifying a pre-existing tendency.

---

### Tier 2 — Re-Run Existing Scripts

#### Step 4: Layer-Sweep Δ IoU for CLIP

`analyze_q2_metrics.py` evaluates at `--layer 11`. Re-run for layers 7–11 for CLIP only to test whether the Layer 10 > 11 non-monotonic pattern (observed on one image in `finetuning_results.md`) holds at the population level.

**Command (approximate):**
```bash
for layer in 7 8 9 10 11; do
    python experiments/scripts/analyze_q2_metrics.py \
        --experiment-id fine_tuning_primary_20260327 \
        --models clip \
        --layer $layer \
        --output-suffix layer_sweep_$layer
done
```

**What to look for:** Does the post-FT IoU peak at layer 10 for CLIP? If yes, all aggregate CLIP results (which use layer 11) are slightly *understated*. This would be a methodological finding worth noting.

**Estimated runtime:** ~5–10 minutes per layer on MPS.

---

#### Step 5: Attention Entropy Measurement

Shannon entropy of the CLS attention weight distribution is a direct measure of attention diffuseness. This tests H2 (entropy hypothesis) from the main brainstorm doc.

**Metric:**
```
H = -sum(p_i * log(p_i))   over the 196 patch attention weights at layer 11
```

**What to add to `analyze_q2_metrics.py`:** During attention extraction, compute entropy alongside IoU. Store as a new metric row in the output JSON.

**Predictions:**
- DINOv3 frozen: lowest entropy (most concentrated)
- CLIP frozen: highest entropy (most diffuse)
- CLIP fine-tuned: entropy decreases substantially
- DINOv3 fine-tuned: entropy essentially unchanged

If these predictions hold, entropy change is a proxy for "how much spatial reorganization fine-tuning caused" — a clean narrative to accompany the Δ IoU results.

---

### Tier 3 — New Experiments

#### Step 6: Country Classification as Negative Control

This is the strongest available experimental control. Fine-tune CLIP on country (Germany/France/UK/Spain/Italy) instead of architectural style, then measure Δ IoU on the same 139 annotated images.

**Prediction:** Δ IoU ≈ 0 or negative. Country labels don't correspond to architectural feature regions — the model should attend to landscape, sky, or urban context rather than portals and windows.

**Why this matters:** If country-CLIP shows Δ ≈ 0 while style-CLIP shows Δ = +0.056, it proves the improvement is **task-driven** (the style label contains spatial information) rather than **parameter-update-driven** (any fine-tuning improves IoU). This is the cleanest possible control for the core claim.

**Implementation effort:** Low. The infrastructure exists completely.
- Add `COUNTRY_MAPPING` to `src/ssl_attention/config.py` (top 5 countries by count)
- Pass `label_source="country"` to `FullDataset` in `fine_tune_models.py`
- Run analysis with existing `analyze_q2_metrics.py`

The enhancement doc `fine_tuning_methods.md` (Section 10.1) already outlines the implementation details.

---

#### Step 7: CLIP Text-Patch Similarity Probe

Load CLIP's text encoder and compute cosine similarity between text embeddings of feature names and frozen patch features across the 139 annotated images.

**Key question:** Do CLIP's *frozen patch features* already align with bbox regions when queried by the feature name, even though CLS attention is diffuse? If yes, fine-tuning's role is to teach the CLS token to aggregate these already-aligned patches — not to create new spatial knowledge.

**Implementation:** ~50 lines using HuggingFace `CLIPModel`:
```python
from transformers import CLIPModel, CLIPProcessor
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch16")
text_inputs = processor(text=["round arch portal", "pointed arch", ...], return_tensors="pt")
text_embeds = model.get_text_features(**text_inputs)  # (num_features, 512)
# patch features already cached in HDF5 from generate_feature_cache.py
patch_sims = torch.cosine_similarity(patch_features, text_embeds[i].unsqueeze(0), dim=-1)
```

Compute IoU between thresholded patch similarity heatmap and the corresponding bbox mask. Compare against CLS attention IoU. If patch-text similarity IoU >> CLS attention IoU (frozen), the "latent spatial knowledge" hypothesis is confirmed.

---

## Summary Table

| Step | Data Needed | Compute Cost | Confidence Value | Status |
|------|-------------|-------------|-----------------|--------|
| 1. Style breakdown script | existing JSON | none | high — formalizes new finding | ✅ Done |
| 2. Cross-model correlation | existing JSON | none | moderate — characterizes mechanism | ✅ Done |
| 3. MAE Renaissance investigation | existing JSON | none | high — explains biggest surprise | ⬜ Pending |
| 4. CLIP layer sweep | re-run script | ~30 min | moderate — may reframe CLIP numbers | ⬜ Pending |
| 5. Attention entropy | modify script | ~10 min | high — direct test of H2 | ⬜ Pending |
| 6. Country classification | new FT run | ~2–3 hrs | **critical** — strongest control | ⬜ Pending |
| 7. CLIP text-patch probe | new code | ~1 hr | high — tests H1 directly | ⬜ Pending |

---

## Open Questions This Roadmap Would Close

| Question | Closed By | Answer |
|----------|-----------|--------|
| Is CLIP's improvement driven by Gothic/Romanesque-specific language grounding? | Step 1 ✅ | Yes — Gothic +0.079, Romanesque +0.066; Renaissance/Baroque near-zero |
| Do CLIP and DINO respond to the same "easy" images? | Step 2 ✅ | Yes — r=+0.677; shared easy images, not complementary mechanisms |
| Why does MAE show a Renaissance spike? | Step 3 ⬜ | Open — hypothesized: compact geometric shapes (Trefoil Window, Pediment) |
| Is CLIP's layer 11 measurement understating the true improvement? | Step 4 ⬜ | Open |
| Does fine-tuning compress CLIP's diffuse attention? | Step 5 ⬜ | Open |
| Is the improvement task-driven or just parameter-update-driven? | Step 6 ⬜ | Open — country classification control not yet run |
| Did CLIP's patch features already encode spatial knowledge pre-FT? | Step 7 ⬜ | Open |

---

## Related Documents

- [q2_results_analysis.md](../research/q2_results_analysis.md) — deep analysis and hypothesis set this roadmap draws from
- [fine_tuning_methods.md](fine_tuning_methods.md) — country classification implementation details (Section 10.1)
- [docs/research/finetuning_results.md](../research/finetuning_results.md) — original Q2 interpretation notes including the Layer 10 observation

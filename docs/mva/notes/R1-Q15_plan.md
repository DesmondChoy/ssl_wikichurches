# Plan: R1-Q15, the layer 6 to 7 alignment step of the DINO models

Status (2026-10-25): done. The human approved the manuscript change. The A15 reply was given in chat for the human to paste into the letter. Tracked in beads issue `ssl_wikichurches-6vep`, closed in `92792a7`. The 2026-10-31 fresh-eyes review corrected two citations in the second paragraph (see below).

The two paragraphs, `analyze_layer_step.py` and `q15_layer_step.json` were committed inside the R1-Q18 commit `8617190`. The R1-Q15 and R1-Q18 sessions shared one git index, and the R1-Q18 commit included the files R1-Q15 had just staged. A separate R1-Q15 commit (`da6d1af`) then added the paragraphs a second time. It was reset before anything was pushed, so `8617190` holds the only copy.

## The item

Reviewer 1, question 15 (`R1-Q15`) in the round-1 reply letter: *XAI Vision round1 reply to all reviewers 1006.docx* in Google Drive (file ID `1z1jEHklwWdebqoxusVEL7Pt5vTpvtQFf`). See [`AGENTS.md`](../AGENTS.md) for the revision workflow.

**Q15 (reviewer):**

> Provide a more in-depth analysis of the abrupt rise in IoU alignment starting at layer 6 for the DINO models, as seen in Figure 3.

**TODO:** To dig into this jump. Can clarify if it just looks like a big jump from the diagram. Trend is still increasing overall, may not need to argue this individual jump.

## Finding

The step is real, not a plotting artifact, so the reply describes it rather than playing it down. For DINOv3 the curve falls from layer 0 to layer 6 before the step, so "the trend is increasing overall" does not hold for it. Layers are numbered from 0, as in Figure 3.

`experiments/scripts/analyze_layer_step.py` reads only `outputs/cache/metrics.db` and writes `outputs/results/q15_layer_step.json`. Every number in the new text comes from it:

| Layer 6 to 7 | DINOv2 | DINOv3 |
|---|---|---|
| Mean IoU@90 | 0.063 to 0.111 | 0.036 to 0.119 |
| Next-largest change between consecutive layers | +0.024 (about half) | +0.018 (less than a quarter) |
| Images where IoU@90 rises | 86% | 90% |
| Images where Coverage rises, KL and EMD fall | 93%, 95%, 94% | 99%, 99%, 97% |
| Images where MSE improves | 37% | 41% |
| Heads whose IoU@90 rises | 10 of 12 | 12 of 12 |

- The step appears at every stored IoU percentile (50 to 90) and grows with the percentile.
- Wilcoxon signed-rank tests, Holm-adjusted across 6 models and 5 metrics: p < 1e-17 for IoU@90, Coverage, KL and EMD in both DINO models. MSE worsens slightly (Holm p 7.8e-4 for DINOv2, 0.16 for DINOv3). Table 3 already shows that MSE ranks the models differently.
- MAE rises from 0.038 to 0.053 over the same layers (64% of images). CLIP, SigLIP and SigLIP2 do not rise significantly.

## Change

Two paragraphs were added to Section 4.3.1 of `round2_v1024_changes.tex`, after Figure 3 and before Section 4.3.2:

1. A description of the step. It is the largest change between consecutive layers, it does not depend on the threshold, it appears in Coverage, KL and EMD but not in MSE, and it covers most images and most heads. MAE, CLIP, SigLIP and SigLIP2 do not show it.
2. An interpretation that cites only references already in the bibliography:
   - `Walmer23CVPR`: DINO ViT-B/16 uses only global heads in its last layers, as do CLIP and supervised ViTs, and its [CLS] attention is centered on the object from the middle layers onward. MAE keeps local heads in its deep layers.
   - `park2023what`: contrastive learning mainly shapes the later layers, and masked image modeling mainly shapes the early ones.
   - `Yamamoto2025`: DINO ViT-S/16 [CLS] attention is closest to human gaze between layers 8 and 10 (counting from 1), which are layers 7 to 9 in our numbering.
   - Caveat: none of these studies reports an abrupt step at a fixed layer, and we did not analyze the mechanism.

Figure 3 is unchanged. R2-Q5 owns its redesign, and `RQ2_layer_iou_new.png` is in `ssl_wikichurches-yvm1`.

## Checks dropped

These checks ran during the investigation, but their results are not in the paper:

- **Share of attention in the top 10% of pixels.** It stays flat from layer 6 to 7 (about 0.22 to 0.23). The cached maps are min-max normalized, so this share depends on the normalization.
- **Overlap of the top-10% regions in consecutive layers.** It is low at every layer. The layer 6 to 7 overlap is the lowest of the late layers only for DINOv3 (0.13), so it does not show that the region moves for both models.
- **CLS-to-register attention shares and token norms.** These need new forward passes and are out of scope.

## Build

Both PDFs build with no undefined citations. In the clean build after R1-Q18, the new text starts on page 18, below Figure 3. Section 4.3.1, Figure 3 and Table 3 keep their numbers.

## Reply letter (A15)

Given in chat on 2026-10-25 for the human to paste:

> A15: We added two paragraphs to Section 4.3.1, after Figure 3. The rise between layers 6 and 7 is the largest change between consecutive layers in both DINO models. DINOv3's mean IoU@90 rises from 0.036 to 0.119, more than four times its next-largest change, and DINOv2's rise is about twice its next-largest change. The step does not come from the IoU@90 threshold. It appears at every percentile from 50 to 90, and Coverage, KL divergence and EMD also improve, each in at least 92% of the 139 images (two-sided Wilcoxon signed-rank test, Holm-adjusted p < 10⁻¹⁷); only MSE becomes slightly worse. It is also not driven by a few images or a single head: IoU@90 rises in 86% (DINOv2) and 90% (DINOv3) of the images, and in 10 and 12 of their 12 heads. Over the same layers MAE rises much less, and CLIP, SigLIP and SigLIP2 do not rise significantly. We relate the step to studies showing that the DINO objective yields global, object-centered [CLS] attention in the later layers, unlike MAE, and that DINO attention is closest to human gaze in comparable layers. Because no prior study reports an abrupt change at a particular layer and we did not analyze the mechanism, the text states that the exact layer may be specific to the two checkpoints we evaluated.

## Fresh-eyes review (2026-10-31)

A review of all Reviewer 1 items checked the interpretation paragraph's citations against the papers. Two phrases changed in Section 4.3.1. No numbers changed, and the reply letter is unchanged, because A15 gives no layer numbers for these studies.

| Citation | Before | After | Why |
|---|---|---|---|
| `Walmer23CVPR` | "...in its last layers and that its..." | "...in its last layers, as do CLIP and supervised ViTs, and that its..." | The paper reports only global heads in the last layers for fully supervised models, CLIP, DINO and MoCo, so the finding is not specific to DINO. |
| `Yamamoto2025` | "in the eighth and ninth of its 12 layers ... layers 7 and 8" | "between the eighth and tenth of its 12 layers ... layers 7 to 9" | The median distance to gaze is lowest at layers 8 and 9 (counting from 1), but the paper names layers 9 and 10 as optimal for this model. The new range covers both. |

The Walmer addition means that global heads alone cannot explain a step that only the DINO models show. The paragraph's closing caveat already says the mechanism was not analyzed, so it is unchanged. `park2023what` was checked and needs no change.

## Out of scope

Frozen CLIP's top-10% attention share rises from about 0.23 at layer 5 to between 0.60 and 0.70 from layer 7 onward. This is direct evidence for the attention-sink explanation in the R1-Q14 paragraph, which currently says token norms were not measured. It belongs to R1-Q14 if anyone wants to add it.

# Plan: R1-Q10, upsampling the attention maps

Status (2026-10-25): done. The human approved the manuscript change, and it is committed in `2784c7d`. The A10 reply was given in chat for the human to paste into the letter. Tracked in beads issue `ssl_wikichurches-nhcl`, closed in `5ca8d5d`.

## The item

Reviewer 1, question 10 (`R1-Q10`) in the round-1 reply letter: *XAI Vision round1 reply to all reviewers 1006.docx* in Google Drive (file ID `1z1jEHklwWdebqoxusVEL7Pt5vTpvtQFf`). See [`AGENTS.md`](../AGENTS.md) for the revision workflow.

**Q10 (reviewer):**

> Justify the choice of bilinear interpolation to upsample the attention maps from the patch-grid resolution to 224×224. Did you test alternative interpolation methods? It could completely change the results.

**A10 (TJ):** Is it linked to Q4? I am not sure why this interpolation would "completely change" the results.

**TODO:** We did not test other interpolation methods. Explain why we need interpolation (resize, normalise to the same resolution to compare with ground truth masks). No need to justify the choice of bilinear interpolation.

## Scope

The work is text only. Per the TODO, the manuscript explains why upsampling is needed and does not argue for bilinear over other methods. That no other method was tested is stated in the reply, not the manuscript.

TJ's question: Q10 is related to Q4 but is a different step. Q4 is the input resize of the image before the model. Q10 is the output upsampling of the attention map after the model. Section 4.1 already described both.

## What the code shows

- `INTERPOLATION_MODE = "bilinear"` (`src/ssl_attention/config.py`) is used by `attention_to_heatmap` (`src/ssl_attention/attention/cls_attention.py`), which reshapes the patch attention to its square grid, upsamples it with `F.interpolate(..., align_corners=False)` and min-max normalizes each map. This matches Section 4.1.
- `app/precompute/generate_metrics_cache.py` builds the Gaussian target at the upsampled map's shape (see the [R1-Q8 note](R1-Q8_plan.md)), so the upsampled grid is shared by all five metrics.
- No run used another interpolation mode.

## Change

All edits are in `round2_v1024_changes.tex`. Two sentences were added to Section 4.1, after "...upsampled to $224 \times 224$ using bilinear interpolation." They say:

1. Why: the patch grids are much coarser than the 224 × 224 masks and differ between backbones.
2. Effect: every attention map is placed on the same grid as the expert masks and Gaussian targets, so each metric compares two maps of the same size in the same coordinate frame.

The patch-grid sentences that follow were left in place ("given below") so the diff stays a pure insertion.

## Build

Both PDFs build. The new text is on clean page 13 and diff page 13 (diff lines 562–566), in Section 4.1.

## Reply letter (A10)

Given in chat on 2026-10-25 for the human to paste:

> A10: We thank the reviewer for this question. Each model produces its attention map on its own patch grid, 16 × 16 for DINOv2 and 14 × 14 for the other five models, whereas the expert masks and Gaussian targets are defined on the 224 × 224 image grid. The attention maps must therefore be brought to the same resolution before the alignment metrics can be computed, and upsampling every map to 224 × 224 places it in the same coordinate frame as the masks. This step is separate from the input resizing described in our reply to Q4, which is applied to the image before it enters the model. We used bilinear interpolation and did not test other interpolation methods. Section 4.1 now explains why the upsampling is needed.

## Out of scope

- An interpolation sensitivity run (nearest, bicubic). The TODO says no other methods were tested and none need to be justified.
- An argument that interpolation cannot change the rankings. It is plausible, since every method interpolates the same patch values, but it is not measured, so it is not claimed.

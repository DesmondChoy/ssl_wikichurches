# Plan: R1-Q8, the standard deviation of the Gaussian soft-union target

Status (2026-10-25): done. The human approved the manuscript change, and it is committed in `78c9653`. The A8 reply was given in chat for the human to paste into the letter. Tracked in beads issue `ssl_wikichurches-rg1x`, closed in `2b18568`.

## The item

Reviewer 1, question 8 (`R1-Q8`) in the round-1 reply letter: *XAI Vision round1 reply to all reviewers 1006.docx* in Google Drive (file ID `1z1jEHklwWdebqoxusVEL7Pt5vTpvtQFf`). See [`AGENTS.md`](../AGENTS.md) for the revision workflow.

**Q8 (reviewer):**

> Why did you use this specific formula used to fit the standard deviations when constructing the Gaussian soft-union target map?

**A8 (TJ):** No note.

**TODO:** Explain the rationale. The purpose is to spread the key points into a continuous map for attention alignment.

## Scope

The work is text only. The reviewer asks why, not for other values, so no sensitivity analysis on σ was added.

## What the code shows

- `gaussian_bbox_heatmap` (`src/ssl_attention/metrics/continuous.py`) sets `sigma = max(side_px / 4, 1)` per axis, matching Eq. 3 in Section 3.4.
- No rationale is recorded. `docs/core/project_proposal.md` states only "σ = bbox_size / 4", and `docs/reference/metrics_methodology.md` does not mention σ. No run tried another σ.
- `app/precompute/generate_metrics_cache.py` builds the target at the attention map's shape, which is the upsampled 224 × 224 map.
- Only MSE, KL divergence and EMD use the Gaussian target. IoU@90 and Coverage use the binary union mask.

## Numbers

Every number in the new text was computed this session:

| Quantity | Value | Source |
|---|---|---|
| Mass within ±2σ, one axis | 0.9545 | `erf(2/√2)` |
| Mass within the box, 2-D | 0.9111 | `erf(2/√2)²` |
| Target value at a box edge | 0.1353 | `exp(-2)` |
| Boxes where the 1-px floor applies (either axis, 224 × 224) | 28 of 631 (11 on x, 20 on y) | `load_annotations(ANNOTATIONS_PATH)`, side < 4 px |
| Median box size at 224 × 224 | 15.6 × 23.3 px | same |

## Change

All edits are in `round2_v1024_changes.tex`. One paragraph was added to Section 3.4, after Eq. 3 (σ) and before the definition of Gᵢ. It covers, in order:

1. Purpose: a continuous target penalizes near-misses less than distant attention.
2. Why /4: box edges at 2σ, about 95% of mass per axis and 91% in 2-D inside the box, and edge value e⁻² ≈ 0.14.
3. Why per box and per axis: the same relative spread for large and small elements, and the box's aspect ratio is kept.
4. The 1-px floor: it stops tiny boxes from becoming single-pixel spikes and applies to 28 of 631 boxes.
5. Scope: σ was fixed in advance and not tuned, and it affects only MSE, KL divergence and EMD.

## Build

Both PDFs build. The new text is on clean and diff pages 9–10.

## Reply letter (A8)

Given in chat on 2026-10-25 for the human to paste:

> A8: We thank the reviewer for this question. The purpose of the Gaussian target is to turn each expert box into a continuous map, so that attention just outside an annotated region is penalized less than attention far from it. We set each axis's standard deviation to a quarter of the box's width or height. The box edges then lie two standard deviations from the centre, so about 95% of the Gaussian's mass along each axis, and about 91% in two dimensions, falls inside the box. The target value at the box edge is e⁻² ≈ 0.14 rather than zero. Because σ is tied to each box's own width and height, the spread is the same relative to the box for large and small elements, and the target keeps the box's aspect ratio. The lower bound of one pixel prevents very small boxes from collapsing to a single-pixel spike; on the 224 × 224 grid it applies to 28 of the 631 boxes along at least one axis. The value of σ was fixed in advance and not tuned. It affects only MSE, KL divergence and EMD, because IoU@90 and Coverage use the binary mask. Section 3.4 now explains this rationale after Eq. 3.

## Out of scope

- A σ sensitivity analysis (for example σ = side/2 or side/6). The reviewer did not ask for one, and it could not change the IoU@90 results.
- A citation for the ±2σ convention. None was checked, so none was added.

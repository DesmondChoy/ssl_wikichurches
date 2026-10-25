# Plan: R1-Q9, the IoU@90 threshold

Status (2026-10-25): done. The human approved the manuscript change, and the A9 reply was given in chat for the human to paste into the letter. Tracked in beads issue `ssl_wikichurches-48ox`, now closed.

## The item

Reviewer 1, question 9 (`R1-Q9`) in the round-1 reply letter: *XAI Vision round1 reply to all reviewers 1006.docx* in Google Drive (file ID `1z1jEHklwWdebqoxusVEL7Pt5vTpvtQFf`). See [`AGENTS.md`](../AGENTS.md) for the revision workflow.

**Q9 (reviewer):**

> Explain why you selected IoU@90, i.e. thresholding the top 10% of pixels. Did you perform a sensitivity analysis using other thresholds?

**A9 (TJ):** We can explain the rationale. If we have other results we can add, otherwise, no need to add results.

**TODO:** Explain, no need to report multiple thresholds, just briefly clarify what we tested and clarify rationale on why p90.

## Scope

The work is text only. Per the TODO, no table or figure of other thresholds is added. The other thresholds are named, and one sentence states what they show.

## What the code shows

- `threshold_attention` (`src/ssl_attention/metrics/iou.py`) keeps exactly `k = round(n × (100 − p) / 100)` pixels with `torch.topk`, so every model is scored on the same pixel count on the 224 × 224 grid.
- `outputs/cache/metrics.db` (`aggregate_metrics`) holds IoU at the 50th, 60th, 70th, 75th, 80th, 85th and 90th percentiles for all six frozen models, all 12 layers and all 139 images. Its p90 values match Table 3.
- The first proposal (`28eebc8`, `docs/project_proposal.md`) planned IoU "at various percentiles (top 10%, 20%, 30%)" and named no primary threshold. The current `docs/core/project_proposal.md` names p90 as primary, but that line dates from the March rewrite (`50a0822`). The text therefore does not claim that p90 was fixed in advance.
- Coverage needs no threshold (Section 3.4 already says it is threshold-free).

## Numbers

Frozen models, each at its best layer for that percentile, mean IoU over 139 images (`aggregate_metrics`; `cls` for DINOv2, DINOv3, MAE and CLIP, `mean` for SigLIP and SigLIP2):

| Percentile | Ranking |
|---|---|
| 90 | DINOv3 0.133, DINOv2 0.118, SigLIP 0.074, MAE 0.070, CLIP 0.051, SigLIP2 0.047 |
| 85 | DINOv3 0.137, DINOv2 0.124, SigLIP 0.087, MAE 0.080, CLIP 0.064, SigLIP2 0.062 |
| 80 | DINOv3 0.138, DINOv2 0.125, SigLIP 0.091, MAE 0.087, CLIP 0.073, SigLIP2 0.071 |
| 75 | DINOv3 0.135, DINOv2 0.125, SigLIP 0.092, MAE 0.090, CLIP 0.079, SigLIP2 0.077 |
| 70 | DINOv3 0.131, DINOv2 0.124, SigLIP 0.092, MAE 0.091, CLIP 0.085, SigLIP2 0.080 |
| 60 | DINOv3 0.123, DINOv2 0.118, MAE 0.092, CLIP 0.091, SigLIP 0.090, SigLIP2 0.083 |
| 50 | DINOv3 0.114, DINOv2 0.111, CLIP 0.094, MAE 0.091, SigLIP 0.090, SigLIP2 0.084 |

DINOv3 and DINOv2 are first and second, and SigLIP2 is last, at every percentile. SigLIP, MAE and CLIP change order. The "102 of 139 images below 10% union area" figure comes from Section 4.2 (R1-Q5, `ce81cb9`).

## Change

All edits are in `round2_v1024_changes.tex`. One paragraph was added to Section 3.4, inside the IoU item, after the sentence that defines IoU@90. It covers, in order:

1. Why p90: the union of the annotations covers less than 10% of the image in 102 of 139 images, so a lower percentile makes the attention mask larger still and lowers the attainable IoU.
2. Why a pixel share: every model is scored on the same number of pixels, whatever its patch grid or attention scale.
3. What was tested: the 50th, 60th, 70th, 75th, 80th and 85th percentiles.
4. What they show: DINOv3 and DINOv2 first and second and SigLIP2 last at every threshold; SigLIP, MAE and CLIP change order.
5. Scope: only IoU@90 is reported, and Coverage is the threshold-free complement.

## Build

Both PDFs build. The new paragraph is on clean page 10 and diff page 10, inside the IoU item of Section 3.4. The reference to Section 4.2 resolves.

## Reply letter (A9)

Given in chat on 2026-10-25 for the human to paste:

> A9: We thank the reviewer for this question. IoU@90 keeps the 10% of pixels with the highest attention. We chose this threshold because the expert-annotated regions are small: their union covers less than 10% of the image in 102 of the 139 images. For these images, a lower threshold selects even more pixels than the annotations cover, which lowers the highest IoU they can reach. Because the threshold is a fixed share of pixels rather than a fixed attention value, every model is scored on the same number of pixels, regardless of its patch grid or the scale of its attention values. We also computed IoU at the 50th, 60th, 70th, 75th, 80th and 85th percentiles. Among the frozen models, DINOv3 and DINOv2 rank first and second and SigLIP2 ranks last at every threshold, whereas the order of SigLIP, MAE and CLIP changes. We therefore report only IoU@90 and use Coverage, which needs no threshold, as a complementary measure. Section 3.4 now explains this choice and the thresholds we tested.

## Out of scope

- A table or appendix of IoU at every percentile. The TODO says not to report multiple thresholds.
- Threshold sensitivity for the RQ2 layer and head analyses and for RQ3 deltas. The reviewer asks about the choice of threshold, and the frozen ranking answers that.
- A 95th-percentile run. The API accepts it, but the cache was not computed at p95.

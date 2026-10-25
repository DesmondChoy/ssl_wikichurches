# Plan: R1-Q16, bold the best value in each column of Table 3

Status (2026-10-25): done. The human approved the manuscript change, and it is committed in `cda5d58`. The A16 reply was given in chat for the human to paste into the letter. Tracked in beads issue `ssl_wikichurches-1pyb`, closed in `22229b0`.

## The item

Reviewer 1, question 16 (`R1-Q16`) in the round-1 reply letter: *XAI Vision round1 reply to all reviewers 1006.docx* in Google Drive (file ID `1z1jEHklwWdebqoxusVEL7Pt5vTpvtQFf`). See [`AGENTS.md`](../AGENTS.md) for the revision workflow.

**Q16 (reviewer):**

> In Table 3, mark the best result in each column in bold to improve readability.

**TODO:** Bold highest numbers.

## Scope

The work is formatting only. No numbers, code, results or figures changed. Table 3 is `table RQ1 result` and keeps its number in the revised manuscript.

## Check

Before bolding, all 30 cells of Table 3 were compared with `aggregate_metrics` in `outputs/cache/metrics.db`: percentile 90, `cls` for DINOv2, DINOv3, MAE and CLIP, `mean` for SigLIP and SigLIP2, best layer chosen separately for each metric. Every cell matches. This mattered because R1-Q4 changed the DINOv2 and CLIP rows.

The best value in each column, with higher better for IoU@90 and Coverage and lower better for the other three:

| Column | Best | Value |
|---|---|---|
| IoU@90 | DINOv3 | 0.133 |
| Coverage | DINOv3 | 0.137 |
| MSE | SigLIP2 | 0.017 |
| KL divergence | DINOv3 | 2.325 |
| EMD | DINOv3 | 0.260 |

No column has a tie. The closest is MSE: SigLIP2 0.01745 against SigLIP 0.01755, which print as 0.017 and 0.018.

## Change

In `round2_v1024_changes.tex`, the five cells above are wrapped in `\textbf`, and the caption ends with "Bold marks the best value in each column." This follows Table 7, whose caption already explains its bold.

## Build

Both PDFs build with no undefined references. In the clean build at `cda5d58`, Table 3 is on page 15 and its layout is unchanged. In the tracked-changes PDF, latexdiff shows each bolded cell as a struck plain value next to a blue bold one. With the struck R1-Q4 values already in the DINOv2 and CLIP rows, the table becomes about 20 pt wider, and the EMD column touches line numbers 675 to 677. The human accepted this.

## Reply letter (A16)

Given in chat on 2026-10-25 for the human to paste:

> A16: We thank the reviewer for this suggestion. In Table 3, the best value in each column is now shown in bold: the highest IoU@90 and Coverage, and the lowest MSE, KL divergence and EMD. The caption explains the bold.

## Out of scope

- The bold MSE value belongs to SigLIP2, which is scored with the mean-attention proxy. The R1-Q6 paragraph in Section 4.2 already names Table 3 among the comparisons this confound affects, so the caption needs no caveat.

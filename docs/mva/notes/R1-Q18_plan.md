# Plan: R1-Q18, replace the Figure 4 screenshot with a LaTeX table

Status (2026-10-25): done. The human approved the manuscript change, and it is committed in `8617190`. The A18 reply was given in chat for the human to paste into the letter. Tracked in beads issue `ssl_wikichurches-ang7`, closed in `f8521a9`.

`8617190` also contains R1-Q15 work: the paragraph after Figure 3, `experiments/scripts/analyze_layer_step.py` and `outputs/results/q15_layer_step.json`. The R1-Q15 session had staged them in the shared git index just before the R1-Q18 commit. In a shared checkout, commit through a private index (`GIT_INDEX_FILE` built with `git read-tree HEAD`) so other sessions' staged files stay out.

## The item

Reviewer 1, question 18 (`R1-Q18`) in the round-1 reply letter: *XAI Vision round1 reply to all reviewers 1006.docx* in Google Drive (file ID `1z1jEHklwWdebqoxusVEL7Pt5vTpvtQFf`). See [`AGENTS.md`](../AGENTS.md) for the revision workflow.

**Q18 (reviewer):**

> Clarify the origin of Figure 4. It looks like it was AI-generated.

**TJ:** We can change this Figure to be a pure Table in Latex.

**TODO:** Just keep the table, add title/details in the figure caption.

## Origin of Figure 4

Figure 4 (`590_fig3.png`, added in `4895800` as "revised fig 3 screenshot") is a screenshot of the head-feature matrix card in the project's React app. The card has a title banner, a legend box, rounded number chips and an "Example only" footer, which explains the reviewer's impression. Its values come from the precomputed metrics cache, not from an image generator.

## Check

All 96 cells match `head_feature_metrics` in `outputs/cache/metrics.db`:

```sql
select feature_name, bbox_count, head, mean_score
from head_feature_metrics
where model = 'dinov3' and layer = 'layer10' and method = 'cls'
  and metric = 'iou' and percentile = 90 and bbox_count >= 3;
```

The 8 rows are the 8 features with the highest H8 score among the 50 features with at least 3 boxes (90 features in total). Each value is the mean per-box IoU@90 over the feature's boxes (`aggregate_feature_scores` in `app/precompute/generate_metrics_cache.py`). H8 at layer 10 is DINOv3's top head in `head_summary_metrics` (0.160). R1-Q4 regenerated only DINOv2 and CLIP, so these values did not change.

## Change

In `round2_v1024_changes.tex`:

1. The `figure` float keeps its label and number. `\includegraphics{590_fig3.png}` is replaced by a booktabs `tabular` with Feature, Boxes and H0 to H11. H8 is bold.
2. The caption drops the colour-scale sentence. It says what a cell is, what Boxes counts, that bold marks the dominant head H8, and how the 8 rows were chosen.
3. In the RQ2 head paragraph, "The dominant head, H8, is highlighted. The colour scale is encoded in the figure, ..." becomes "The dominant head, H8, is shown in bold."

`590_fig3.png` stays because `round2_v1024.tex` still references it.

### Fitting the text width

The text width is 372 pt. Measured widths of the tabular:

| Setting | Width |
|---|---|
| Body font, 6 pt column padding | 569 pt |
| Table font (`\tablebodyfont`, 8 bp), 6 pt | 509 pt |
| Table font, 3 pt, two-line feature names | 391 pt |
| Table font, 2.5 pt, two-line names, no edge padding | 372 pt |
| Table font, 4 pt, two-line names, no edge padding, no leading zeros | 359 pt |

The last row is used. The table uses the class table font, like every other table in the manuscript, splits feature names over two lines, and drops the leading zero (`.215`). Keeping the leading zero would need padding of 2.5 pt or less.

Bold here marks the dominant head, not the best value in a row. In the Tracery Rose Window row, H2 (0.140) is higher than H8 (0.130). The caption states what bold means.

## Build

Both PDFs build with no undefined references. The only overfull box in the clean build is the existing 3.4 pt one in Table 1. Figure 4 is on page 19, and the figures after it keep their numbers (Figures 5 to 9). In the tracked-changes PDF, latexdiff writes each new cell as `\DIFaddFL{.099 }` with a trailing space, which makes Figure 4 about 21 pt wider, so the H11 column touches the line numbers. This matches the accepted Table 3 widening from R1-Q16.

## Reply letter (A18)

Given in chat on 2026-10-25 for the human to paste:

> A18: We thank the reviewer for this comment. Figure 4 was a screenshot of the head-feature matrix view in the interactive analysis tool we developed for this study. Its values were computed by our evaluation code from the expert annotations and the models' attention maps, not produced by a generative model. To remove the ambiguity, we replaced the screenshot with a LaTeX table that holds the same values: the mean IoU@90 of each of the 12 attention heads of frozen DINOv3 at layer 10 for eight architectural features, with the number of expert-annotated boxes for each feature and the dominant head, H8, in bold. The title and notes that were inside the image are now in the caption, which also explains how the eight features were chosen. The figure keeps its number, and the text in Section 4.3.2 now refers to the bold column instead of a colour scale.

## Out of scope

- R1-Q17 (full head-feature matrices in an appendix) is a separate item. Its session reuses this table layout.
- The caption and body keep the existing claim that the dominant head aligns most strongly with portal-scale and facade-scale structures.

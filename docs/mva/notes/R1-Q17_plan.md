# Plan: R1-Q17, complete head-feature matrices in an appendix

Status (2026-10-25): done. The human approved the change. The generator is committed in `17b135e`, the manuscript change in `d128337` and the diff-recipe fix in `fb0be70`. The A17 reply was given in chat for the human to paste into the letter. Tracked in beads issue `ssl_wikichurches-jgzs`, now closed.

## The item

Reviewer 1, question 17 (`R1-Q17`) in the round-1 reply letter: *XAI Vision round1 reply to all reviewers 1006.docx* in Google Drive (file ID `1z1jEHklwWdebqoxusVEL7Pt5vTpvtQFf`). See [`AGENTS.md`](../AGENTS.md) for the revision workflow.

**Q17 (reviewer):**

> Include the complete feature alignment analyses for all other heads in an appendix, as it will complement the work.

**TODO:** Investigate what other alignment analyses are available, add to appendix if possible.

## What the reviewer is pointing at

Baseline Figure 4 (`590_fig3.png`, label `figure rq2_head_feature_matrix`) is the head-feature matrix for frozen DINOv3 at layer 10. It shows IoU@90 for all 12 heads but only 8 of the 50 features with at least three annotations, and its footer says it is an example, not the full matrix.

## What is available

`outputs/cache/metrics.db`, table `head_feature_metrics`, holds the mean per-box score for every layer, head and feature of DINOv2, DINOv3, MAE and CLIP, for the frozen, LoRA and full fine-tuning runs and all five metrics. Each IoU@90 cell is the mean, over a feature's boxes, of the IoU between the head's top 10% of attention pixels and that box. 50 of the 90 features have at least three boxes. The table has no rows for SigLIP or SigLIP2.

All 96 cells of Figure 4 match the database. The dominant frozen heads from `head_summary_metrics` match the Figure 5 caption: DINOv2 L11/H11, DINOv3 L10/H8, MAE L10/H5 and CLIP L4/H5.

## Requested changes

1. A new Appendix B with four tables, one for each model scored with [CLS]-token attention, at the layer of its dominant frozen head. Rows are all 50 features with at least three boxes, sorted by the dominant head. Columns are Boxes and H0 to H11. Values are frozen IoU@90. Bold marks the dominant head's column.
2. One sentence in Section 4.3.2 (page 19 of the clean build), at the end of the Figure 5 paragraph, that points to Appendix B and says the matrices cover the four [CLS] models.
3. `experiments/scripts/generate_paper_head_feature_tables.py` reads `metrics.db` and prints the LaTeX. The output is pasted into `round2_v1024_changes.tex` rather than included with `\input`, because latexdiff does not follow `\input` without `--flatten`.

## Necessary correctness fixes

None found.

## Decisions

- **LaTeX tables, not figures.** R1-Q18 says Figure 4 looks AI-generated, and TJ asked for it to become a LaTeX table. Tables keep the appendix consistent with that change, and R1-Q18 can reuse this generator. No new PNGs are added while `ssl_wikichurches-yvm1` (overwritten baseline PNGs) is open.
- **Dominant layer only.** "All other heads" is read as all 12 heads of the layer that Figures 4 and 5 already use. All 12 layers would need 48 tables.
- **Frozen IoU@90 only.** Figure 4 is frozen IoU@90.
- **Same layout as the R1-Q18 table.** R1-Q18 was rewriting Figure 4 as a LaTeX table in a parallel session while this item ran. The appendix copies its columns (Feature, Boxes, H0 to H11), its bold dominant-head column and its caption wording, so the same matrix has one style. The first proposal bolded the best head in each row instead. That was dropped because two bold conventions for one matrix would confuse readers.
- **The pointer sentence goes in the Figure 5 paragraph.** The Figure 4 paragraph belonged to R1-Q18's edit. The Figure 5 paragraph fits as well, because each dominant head's column is the complete ranking behind Figure 5.

- **3 pt column padding, one wrapped name.** Measured in the template, the text block is 372 by 553 pt, and a table is name + 206 pt + 26 × padding wide. At R1-Q18's 4 pt, 21 of the 50 names would need two lines, and the table would no longer fit on one page. At 3 pt, every name fits on one line except "Coupled Twin Window with Discharging Arch", which the generator splits into two balanced lines.
- **Table numbering.** The `appendices` environment resets the table counter only once, so the new tables first came out as B2 to B5. `\counterwithin*{table}{section}` after `\begin{appendices}` restarts the count in each appendix section.

## Verification

- An independent parser compared every appendix cell with `head_feature_metrics`. All 2,400 values and 200 box counts match, each table has all 50 features, rows are sorted by the dominant head, and only the dominant column is bold.
- The first eight rows of Table B2 are identical to the R1-Q18 Figure 4 table.
- `ruff check .` and `mypy` on the generator pass.
- Clean PDF (built into `build/jgzs/`): Tables B1 to B4 are on pages 33 to 36, one page each, with no overfull boxes and no undefined references. The Appendix B text is on page 32.
- Tracked-changes PDF with the `AGENTS.md` recipe: the tables run about 35 pt into the line numbers and the last rows overlap the page number. latexdiff marks each new cell as `\DIFaddFL{.025 }`. The wavy underline adds depth to every row, and the space before the closing brace widens every cell.
- A prototype with two extra `perl` lines fixes this. One redefines `\DIFaddFL` as blue without the underline, and the other strips that trailing space in table rows. With it, all four tables fit, and the diff-PDF overflows in Table 5 (R1-Q5), Figure 4 (R1-Q18) and Table 7 (R1-Q3) also go away. The human approved it, and the two lines are now in the `AGENTS.md` render recipe (`fb0be70`). The recipe, run as written, builds the tracked-changes PDF with only the five overflows from other sections remaining.

## Reply letter (A17)

Given in chat on 2026-10-25 for the human to paste:

> A17: We thank the reviewer for this suggestion. Figure 4 showed the head-feature matrix for only eight features at one layer of frozen DINOv3. Appendix B now gives the complete matrices for the four models that use [CLS]-token attention, each at the layer of the model's dominant frozen head: DINOv2 layer 11, DINOv3 layer 10, MAE layer 10 and CLIP layer 4 (Tables B1–B4). Each table reports IoU@90 for all 12 heads of that layer and all 50 features with at least three annotated boxes, with the dominant head in bold. These are the dominant heads of Figure 5, so each table also gives the complete feature ranking behind that figure. SigLIP and SigLIP2 use the mean-attention proxy (see our reply to Q6), and Figure 5 reports their feature alignment at the layer level. The values come from the same per-head results as Figures 4 and 5, so no new experiments were needed. Section 4.3.2 now refers to Appendix B.

## Optional / out of scope

- SigLIP and SigLIP2 matrices. Their per-head proxy maps exist in `attention_viz.h5`, but scoring them per feature is a new computation on the proxy that R1-Q6 flagged as a confound.
- All layers, or the LoRA and full fine-tuning runs. The data exists, and a supplementary CSV would suit it better than appendix pages.
- A sentence on how often the dominant head is the best head for a feature (measured: DINOv2 5 of 50, MAE 16, CLIP 22, DINOv3 24). It would be a new claim, so it is left out.

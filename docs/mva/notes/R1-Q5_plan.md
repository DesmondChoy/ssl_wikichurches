# Plan: R1-Q5, region count, region size and feature-level alignment

Status: scope cut to the minimum the comment and TODO ask for (user decision, 2026-10-25). Implementation step 1 is done. `experiments/scripts/analyze_annotation_size_alignment.py` wrote `outputs/results/q5_annotation_size_alignment.json`, and every number below matches it. Both are uncommitted. Manuscript edits wait for `ssl_wikichurches-z74t` to commit. Tracked in beads issue `ssl_wikichurches-1spa`.

## The item

Reviewer 1, question 5 (`R1-Q5`) in the round-1 reply letter: *XAI Vision round1 reply to all reviewers 1006.docx* in Google Drive (file ID `1z1jEHklwWdebqoxusVEL7Pt5vTpvtQFf`). See [`AGENTS.md`](../AGENTS.md) for the revision workflow.

**Q5 (reviewer):**

> As you mentioned, the image is the primary unit of analysis, but an image can contain multiple expert-annotated regions. The paper should analyze how alignment performance changes depending on the number of masked regions per image and their relative size, as larger regions cover more part of the image. Additionally, expand upon the feature-level analysis beyond the last row of Table 4 to show whether models inherently focus better on specific structural elements like arches versus windows.

**A5:** empty. TJ left no note.

**TODO:** Clarify how the dataset was prepared (since some expert-annotated regions overlap).

"Table 4" is `table RQ1 dinov3`, the DINOv3 robustness table (baseline lines 397–409). Its last row is "Weakest features: Crocket; Fleuron; Pinnacle".

## Scope

Each change answers one phrase of the comment or the TODO. Everything else is cut.

| Comment or TODO | Change |
|---|---|
| "how the dataset was prepared (since some expert-annotated regions overlap)" | One paragraph in Implementation Details |
| "number of masked regions per image" | Spearman ρ between per-image IoU@90 and box count, six frozen models, reported in text |
| "their relative size, as larger regions cover more part of the image" | Spearman ρ between per-image IoU@90 and union-mask area, in the same sentences, plus one sentence on why IoU@90 rises with area |
| "expand upon the feature-level analysis beyond the last row of Table 4 … arches versus windows" | One table of IoU@90 by element category for the six models, and one paragraph |
| The last row of Table 4, which the comment names | One necessary correctness fix (below) |

IoU@90 is the only metric. Each model uses its Table 3 IoU@90 layer and default method, which is the setting Table 4 already uses for DINOv3. Significance follows the paper's existing rule, Holm correction across the six models within each comparison.

## What the data show

All numbers below come from `dataset/building_parts.json` and from `outputs/cache/metrics.db` after the R1-Q4 regeneration.

### Annotations (for the TODO)

- The 631 boxes sit in 566 groups. Most groups hold one box. The 60 groups with several boxes describe a composite element through its parts. For example, a Tracery Window group holds separate Pointed Arch Window, Tracery and Trefoil boxes.
- `load_annotations` keeps each box's own label and drops the group label.
- 136 of the 1,420 same-image box pairs overlap. In 54 of them one box lies entirely inside the other. Overlaps occur in 66 of the 139 images. These counts use the normalized box coordinates.
- Images have 1 to 10 boxes (median 4). Their union covers 0.4% to 30.8% of the image (median 4.8%).
- The union mask counts overlapping pixels once (`ImageAnnotation.get_union_mask`). The Gaussian target takes the pixel-wise maximum (baseline lines 303–306). Feature-level scores treat each box separately (`compute_per_bbox_iou`).

The manuscript already mentions the union mask (baseline line 356) and the pixel-wise maximum (line 303). It does not mention the groups or the overlap.

### Count and size versus IoU@90

Spearman ρ over the 139 images. P-values here are unadjusted.

| Model (layer) | ρ(IoU@90, union area) | ρ(IoU@90, box count) |
|---|---:|---:|
| DINOv2 (11) | +0.33 (p < 0.001) | +0.36 (p < 0.001) |
| DINOv3 (11) | +0.40 (p < 0.001) | +0.41 (p < 0.001) |
| MAE (11) | +0.28 (p < 0.001) | +0.31 (p < 0.001) |
| CLIP (0) | +0.60 (p < 0.001) | +0.32 (p < 0.001) |
| SigLIP (8) | +0.42 (p < 0.001) | +0.45 (p < 0.001) |
| SigLIP2 (4) | +0.36 (p < 0.001) | +0.22 (p = 0.009) |

The selected layers reproduce the revised Table 3 IoU@90 column (0.1178, 0.1327, 0.0702, 0.0512, 0.0739, 0.0466).

IoU@90 rises with area partly because it keeps the top 10% of pixels, so a target covering a fraction *g* < 0.1 of the image can reach at most *g* / 0.1. 102 of the 139 images are below 10%. The repo's methodology note states the same bound (`docs/reference/metrics_methodology.md`, "Theoretical maximum").

### Table 4's last row

The weakest features are also the smallest boxes, so their IoU@90 cannot be large.

| Feature | Boxes | Median box area | Max IoU@90 at that area | DINOv3 mean IoU@90 |
|---|---:|---:|---:|---:|
| Pinnacle | 28 | 0.19% | 0.019 | 0.005 |
| Fleuron | 15 | 0.05% | 0.005 | 0.001 |
| Crocket | 6 | 0.02% | 0.002 | 0.000 |

### IoU@90 by element category

Categories are the top-level types in the dataset ontology (`parents` in `building_parts.json`) with at least 20 boxes. Each box counts under every top-level type its feature descends from. For example, a round-arch window counts as both an arch and a window, and 83 boxes are both. Category means are box-weighted means of the per-feature rows in `feature_metrics`, so they need no new inference.

| Category | Boxes | Median box area | DINOv2 | DINOv3 | MAE | CLIP | SigLIP | SigLIP2 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Arch | 177 | 0.67% | 0.041 | 0.045 | 0.031 | 0.013 | 0.038 | 0.018 |
| Window | 161 | 0.74% | 0.052 | 0.062 | 0.032 | 0.018 | 0.030 | 0.025 |
| Column | 88 | 1.29% | 0.030 | 0.043 | 0.025 | 0.013 | 0.036 | 0.009 |
| Ornament | 85 | 0.13% | 0.012 | 0.008 | 0.003 | 0.004 | 0.002 | 0.004 |
| Blind | 58 | 0.56% | 0.027 | 0.030 | 0.015 | 0.010 | 0.010 | 0.008 |
| Tracery | 53 | 1.17% | 0.085 | 0.103 | 0.036 | 0.024 | 0.033 | 0.038 |
| Portal | 47 | 4.03% | 0.090 | 0.115 | 0.091 | 0.026 | 0.141 | 0.030 |
| Pediment | 44 | 2.48% | 0.026 | 0.032 | 0.014 | 0.047 | 0.014 | 0.027 |
| Frieze | 34 | 0.28% | 0.013 | 0.015 | 0.004 | 0.003 | 0.002 | 0.005 |
| Pinnacle | 28 | 0.19% | 0.005 | 0.005 | 0.001 | 0.003 | 0.002 | 0.002 |
| Cornice | 21 | 1.82% | 0.027 | 0.020 | 0.007 | 0.029 | 0.007 | 0.004 |

- Arch and window boxes have similar median areas, so IoU@90 compares them fairly. Windows score at least as high as arches for five models. SigLIP is the exception (0.030 against 0.038).
- Portals score highest for DINOv2, DINOv3, MAE and SigLIP. Pediments score highest for CLIP and tracery for SigLIP2.
- Ornaments and pinnacles, the two smallest categories, are among the three lowest for every model, which matches the area bound. Friezes are third lowest for every model except SigLIP2, where cornices are.

## Changes

### 1. Dataset preparation (TODO)

Add one paragraph to Implementation Details after the dataset paragraph (baseline line 354). Draft with final numbers:

> The annotation file organizes the 631 boxes into 566 groups. Most groups contain a single box, while the 60 groups with several boxes describe a composite element through its parts; for example, a tracery window may be annotated with separate boxes for the pointed-arch window, its tracery, and a trefoil. We used the feature label of each individual box rather than the group label. As a result, boxes can overlap. Of the 1,420 pairs of boxes in the same image, 136 overlap, and in 54 of these one box lies entirely inside the other; such overlaps occur in 66 of the 139 images. Images contain 1 to 10 boxes (median 4), and the union of an image's boxes covers 0.4\% to 30.8\% of the image (median 4.8\%). Overlapping pixels are counted once in the binary union mask and take the pixel-wise maximum in the Gaussian target, so a region annotated twice receives no extra weight. In the feature-level analysis, each box is scored separately.

### 2. Count and size (RQ1 results, after Table 4)

Two or three sentences, no table. Draft structure:

> Because an image can contain several annotated regions, we also examined how frozen alignment depends on the number of boxes per image and on the fraction of the image their union covers. Across the six models, per-image IoU@90 increased with the union area (Spearman $\rho$ from [+0.28] to [+0.60]) and with the number of boxes ($\rho$ from [+0.22] to [+0.45]), with Holm-adjusted $p<$ [X] in all cases. Part of this trend follows from the metric itself. IoU@90 keeps a fixed 10\% of pixels, so for an image whose annotations cover a fraction $g<0.1$ of the image, the attainable IoU@90 is at most $g/0.1$, a bound that applies to [102] of the 139 images.

### 3. Feature level (RQ1 results, after the count and size sentences)

One table (the category table above) and one paragraph. Draft structure:

> Table~X extends the feature-level analysis to all six models by grouping feature types under their top-level ontology categories. Arch and window boxes have similar median areas ([0.67\%] and [0.74\%] of the image), so their IoU@90 values are directly comparable. [Windows score at least as high as arches for five of the six models; SigLIP is the exception.] [Portals score highest for four models.] Ornaments and pinnacles, the two smallest categories, are among the three lowest-scoring categories for every model, which is consistent with the area bound described above.

### 4. Necessary correctness fix: Table 4's last row

**The error.** The comment names Table 4's last row. Its Finding says "Small ornamental details remain challenging", and the matching sentence at baseline line 395 says DINOv3 "remains weak on small decorative elements such as Crockets and Pinnacles". Both read the scores as model weakness. The bound in change 2 shows that boxes of this size cannot score high on IoU@90 (Crocket at most 0.002, Pinnacle at most 0.019).

**Why it affects the requested result.** Changes 2 and 3 state the area bound. Left as written, Table 4's last row would contradict them.

**The fix.** Two wording edits, no new numbers.

- Table 4, last row, Finding cell: "Small ornamental details remain challenging." becomes "These are the smallest annotated parts, so their attainable IoU@90 is low."
- Baseline line 395: "but remains weak on small decorative elements such as Crockets and Pinnacles" becomes "and scores lowest on small decorative elements such as Crockets and Pinnacles, whose small box areas also limit their attainable IoU@90".

## Reply letter (A5), draft

Written into the letter as a green "A5:" paragraph under "A5." on 2026-10-25, after the manuscript change was approved, and confirmed in the stored `.docx`. The TODO line is left for the humans. The new table is Table 5, so the per-style table is now Table 6, and A4's "Tables 3–5" needs updating. Any later table inserted before Table 5 (for example the R2-Q4 hyperparameter table) renumbers both replies.

> In the WikiChurches annotation file, the 631 boxes are organized into 566 groups. Most groups contain a single box, but 60 groups describe a composite element through separate part boxes; for example, a tracery window can be annotated as a pointed-arch window, its tracery and a trefoil. Boxes can overlap both within and across groups. Of the 1,420 pairs of boxes in the same image, 136 overlap, and 54 of these are nested; such overlaps occur in 66 of the 139 images. Overlapping pixels are counted once in the union mask and take the pixel-wise maximum in the Gaussian target. Section 4.1 now describes this preparation.
>
> Section 4.2 now analyzes how alignment depends on the number and size of the annotated regions. For all six frozen models, per-image IoU@90 increases with the number of boxes (Spearman ρ = 0.22–0.45) and with the fraction of the image covered by their union (ρ = 0.28–0.60), and all correlations remain significant after Holm correction. Part of this trend comes from the metric itself. IoU@90 keeps the top 10% of pixels, so an image whose annotations cover less than 10% of its area, as in 102 of the 139 images, cannot reach the maximum score.
>
> Table 5 extends the feature-level analysis of Table 4 to all six models by grouping the features into top-level categories such as arches, windows, portals and ornaments. Arch and window boxes have similar median sizes (0.67% and 0.74% of the image), so the two can be compared directly. Five of the six models align better with windows than with arches, although the difference is small for MAE, and SigLIP shows the opposite. Portals are the best-aligned category for four models, while ornaments and pinnacles, the two smallest categories, are among the three weakest for every model. Because elements this small cannot reach high IoU@90 values, we revised the last row of Table 4 and the related sentence in Section 4.2 to say that their low scores partly reflect their size.

## Implementation

1. Add `experiments/scripts/analyze_annotation_size_alignment.py`, following the existing `analyze_*.py` scripts. It reads `image_metrics` and `feature_metrics` from `outputs/cache/metrics.db` and the annotations, and it does not open `attention_viz.h5`.
   - It selects each frozen model's Table 3 IoU@90 layer from `metrics.db` rather than hard-coding it.
   - It writes `outputs/results/q5_annotation_size_alignment.json` and nothing else. The JSON holds the annotation counts, the ρ values with raw and Holm-adjusted p, the category table and the layer used per model.
   - It fails unless the selected layers reproduce the revised Table 3 IoU@90 column and every model's `feature_metrics` rows sum to 631 boxes.
2. Run `uv run ruff check .` and `uv run mypy`.
3. Edit `round2_v1024_changes.tex`, build both PDFs, and show the change for approval (AGENTS.md workflow steps 3–4).

## Sequencing

- R1-Q4 (`ssl_wikichurches-f9rh`) is closed. The passages this plan edits are unchanged by it.
- `ssl_wikichurches-z74t` is running in Codex and rewrites only the MAE LoRA and Full rows. The script reads only frozen rows, and SQLite readers do not block its writer, so step 1 can run now.
- Make the manuscript edits only after z74t has committed. Otherwise its PDF build and commit could pick up Q5 edits in the same `.tex` file.
- The first run happened while z74t was still writing MAE fine-tuned rows. The script reads only frozen rows, but rerun it once before the manuscript edits so the JSON records the final `metrics.db`.

## Cut

Cut from this item. The comment and the TODO do not ask for these:

- A size-independent measure (box hit rate), its library function and tests.
- Partial correlations, Coverage or any metric other than IoU@90, and a table for the count and size results.
- Rewording the Figure 4 paragraph (baseline line 436), the conclusions (line 511) or the Limitations section.
- The "Strongest features" and "Strongest styles" rows of Table 4, the RQ3 per-style results, and fine-tuned models.

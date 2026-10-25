# Plan: R1-Q4, original image dimensions and resizing

Status (2026-10-25): implemented. Decision: full-image fix, existing checkpoints kept, preprocessing difference disclosed. Code is committed (`d33fcd9` failing test, `d6ce99e` fix, `981d288` figure and analysis generators). The DINOv2 and CLIP caches, the Q2 analysis, the style breakdown, every affected figure and the manuscript text have been regenerated. Waiting for human approval of the manuscript changes before the `.tex`, figures and result files are committed and the reply letter is updated.

## The item

Reviewer 1, question 4 (`R1-Q4`) in the round-1 reply letter: *XAI Vision round1 reply to all reviewers 1006.docx* in Google Drive (file ID `1z1jEHklwWdebqoxusVEL7Pt5vTpvtQFf`). See [`AGENTS.md`](../AGENTS.md) for how the letter is organized and the revision workflow.

**Q4 (reviewer):**

> What are the original dimensions of the images? Resizing to 224×224 can distort the aspect ratio. How did you resize them?

**A4 (TJ's note):** Just clarify the original data format and describe the preprocessing procedure.

**TODO:** Clarify.

Related items: R1-Q10 (bilinear upsampling of attention maps) is a separate item. Do not answer it here.

## What the data and code show

### Evaluation images (139 annotated images, read from `dataset/images/`)

These are the dimensions of the supplied dataset files, not of the camera originals on Wikimedia.

| Property | Value |
|---|---|
| Format | JPEG, RGB |
| Width (px) | 414 to 2,281, median 1,280 |
| Height (px) | 450 to 2,142, median 1,280 (width and height medians are computed separately) |
| Aspect ratio (w/h) | 0.60 to 1.78, median 1.00 |
| Orientation | 71 landscape, 68 portrait, none exactly square (709×708 and 1,281×1,280 are nearly square) |
| Distinct sizes | 88. The most common are 1,706×1,280 (27 images) and 1,280×1,706 (16 images) |

### How each model resizes its input today

Every model uses its own Hugging Face image processor (`preprocess` in `src/ssl_attention/models/base.py`; `FineTunableModel.preprocess` in `src/ssl_attention/evaluation/fine_tuning.py`). Verified on Transformers 5.0.0:

| Model | Geometry | Interpolation |
|---|---|---|
| DINOv2 | Shortest side to 256, keeping the aspect ratio, then center-crop 224×224 (median 57% of the image area kept) | Bicubic |
| CLIP | Shortest side to 224, keeping the aspect ratio, then center-crop 224×224 (median 75% kept) | Bicubic |
| DINOv3 | Resize the full image to 224×224 | Bilinear |
| MAE | Resize the full image to 224×224 | Bilinear |
| SigLIP | Resize the full image to 224×224 | Bicubic |
| SigLIP2 | Resize the full image to 224×224 | Bilinear |

The SigLIP and SigLIP2 wrapper overrides repeat what the checkpoints already do. The NaFlex rationale in the `siglip2.py` docstring does not apply to `google/siglip2-base-patch16-224`.

### The defect

Binary masks and Gaussian targets are always rasterized from normalized box coordinates on the **full-image** 224×224 grid (`BoundingBox.to_mask` in `data/annotations.py`, `annotation_to_gaussian_heatmap` in `metrics/continuous.py`). Attention maps are upsampled from the patch grid without undoing any crop. For DINOv2 and CLIP, the attention covers only the central crop, so the two maps are in different coordinate frames. Boxes that fall inside the crop are still misregistered, because the crop shifts and rescales them.

| Model | Boxes partly outside the crop | Boxes fully outside the crop | Images affected | Median image area kept |
|---|---:|---:|---:|---:|
| DINOv2 | 126 | 30 | 78 / 139 | 57.5% |
| CLIP | 58 | 12 | 48 / 139 | 75.2% |

Evidence that the caches contain the misregistered evaluation:

- Fresh inference with the current processors reproduces cached maps for all six frozen models and for the DINOv2 and CLIP linear-probe, LoRA and full checkpoints, to within float16 rounding (maximum pixel difference below 2.5e-4).
- Recomputed final-layer IoU@90, Coverage, MSE and KL match `metrics.db` exactly. EMD matches to within 5e-8.
- The caches do not record processor provenance, so this is strong sampled evidence, not an exhaustive rerun.

Effect on frozen final-layer IoU@90 (from `ssl_wikichurches-f9rh`, confirmed by Codex):

| Model / method | Current | Full-image resize | Keep crop, remap boxes |
|---|---:|---:|---:|
| DINOv2 CLS | 0.0816 | 0.1178 | 0.1039 |
| DINOv2 rollout | 0.0403 | 0.0496 | 0.0545 |
| CLIP CLS | 0.0181 | 0.0229 | 0.0204 |
| CLIP rollout | 0.0052 | 0.0064 | 0.0076 |

DINOv3 still ranks first at the final layer, but its lead over DINOv2 shrinks from about 0.051 to about 0.015.

Answering Q4 truthfully means describing this pipeline, which would show that DINOv2 and CLIP were scored against misregistered targets. That makes this a **necessary correctness fix** under the editor's request for accurate results.

## Requested changes

1. State the format and dimensions of the evaluation images in Implementation Details.
2. State exactly how images were resized for attention-alignment evaluation, and acknowledge the aspect-ratio distortion.

## Necessary correctness fix

### Policy

For **attention-alignment evaluation**, every model receives the full image resized directly to 224×224 with no crop, using its checkpoint's default interpolation. The reason is protocol design, not which option scores higher: all six models see the same image extent, and the complete annotation set stays in the evaluation. Remapping boxes into each model's crop would score DINOv2 and CLIP on different image regions from the other four models, and it would need a rule for the 30 or 12 boxes that disappear.

**Classification training and validation keep their native preprocessing.** The existing checkpoints are reused. They are learned models evaluated under a common attention-evaluation geometry. The full-image policy changes input pixels, not only the masks, so this has to be disclosed rather than called "unaffected". Specifically:

- Training applies `RandomResizedCrop(224, scale=(0.8, 1.0))` and then the native processor (`fine_tuning.py:1183`). The crop covers 80–100% of the image area at a random aspect ratio between 3:4 and 4:3 (the torchvision default) and is resized to 224×224 without preserving its aspect ratio. Training inputs were therefore already near-full, stretched images. For CLIP, the later center crop removes nothing from the 224×224 square. DINOv2 still resizes to 256 and crops back to 224, which keeps 87.5% of each side.
- Validation, and therefore checkpoint selection, applies the native processors to the original images (`fine_tuning.py:1223`). Selection chose one of 3 epochs. 4 of the 6 DINOv2 and CLIP runs kept epoch 3, the last one.
- The reported classification accuracies remain measurements under that native preprocessing.

### Implementation

1. Add one explicit full-image evaluation preprocessing path. Use it in the DINOv2 and CLIP frozen wrappers, and in `FineTunableModel` **only for attention extraction**. Do not change the processor that `FineTunableModel` uses for training and validation.
2. Route every analysis path through it. `experiments/scripts/analyze_q2_metrics.py:362` calls `model.processor` directly and bypasses `preprocess()`, so regenerating HDF5 and SQLite alone does not refresh the Q2 results.
3. Test behaviour, not shape: every current processor already returns 224×224. Feed a non-square image with distinctive content at its borders and check that the output matches a reference full-image resize, for all six models, on the direct analysis path as well.
4. Store the processor configuration, backend and package versions with the regenerated artifacts.

### Regeneration

1. Attention cache and metrics for `dinov2*` and `clip*`: frozen, linear probe, LoRA and full, all layers, methods and heads. Linear probe can reuse corrected frozen maps only if that reuse is verified, because its backbone is unchanged.
2. RQ1: reselect the best layer **separately for each metric**, which is the rule Table RQ1 uses, before updating the table.
3. The DINOv3 robustness table: recompute next-best comparators, paired gaps and tests.
4. RQ3: adaptation deltas, effect sizes, bootstrap intervals, strategy comparisons and the per-style Kruskal–Wallis results, not only the table cells.
5. Reapply Holm correction across the complete original families (`analyze_q2_metrics.py:878`). Changes to two models can move significance marks for other models.
6. Update the canonical experiment artifacts referenced by `active_experiment.json`. Fig. 6 reads those in preference to the root compatibility copies (`generate_paper_fig6.py:59`).
7. Regenerate PNG overlays with `--no-skip` or with explicit invalidation. `generate_heatmap_images.py:156` skips existing files.

Affected manuscript items: Table RQ1, the DINOv3 robustness table, the layer IoU plot (`RQ2_layer_iou_new.png`), the DINOv2 and CLIP feature panels (`633_fig4.png`), the RQ3 heatmap (`667_fig5.png`), the forest plot (`714_fig6.png`), the per-style table, the head rankings (`787_tbl7_enlarge.png`), and the prose, summary and conclusions that cite them. The DINOv3-only head-feature matrix (`590_fig3.png`) and its descriptive results are unaffected.

## Proposed manuscript text (Implementation Details, around line 356)

Draft only. Finalize after regeneration.

> The 139 evaluation images are JPEG files of varying size, from 414 to 2,281 pixels wide and 450 to 2,142 pixels high (median width and height both 1,280 pixels), with aspect ratios between 0.60 and 1.78. All images were loaded in RGB format. For attention-alignment evaluation, each image was resized directly to $224 \times 224$ pixels without cropping, using each model's default interpolation (bicubic for DINOv2, CLIP and SigLIP; bilinear for DINOv3, MAE and SigLIP2). This anisotropic resize changes the aspect ratio of non-square images, but it retains the full image extent, so no annotated region is excluded by cropping. The bounding boxes are stored in coordinates normalized to the original image, so they were rasterized on the same full-image $224 \times 224$ grid, keeping the attention maps and expert masks spatially registered. Classification training and checkpoint selection used each model's native preprocessing (Section~X); attention maps from the resulting checkpoints were extracted with the full-image resize described here.

Limitations, one or two sentences: the aspect-ratio distortion may affect models pretrained on aspect-preserving crops, aspect-preserving alternatives such as padding were not evaluated, and attention evaluation used a different input geometry from checkpoint selection for DINOv2 and CLIP.

## Decision needed

**Fix approach.** Recommended: the full-image policy above, with the existing checkpoints reused and the difference disclosed. Alternatives: keep the crop and remap boxes (scores differ, see the table), or retrain DINOv2 and CLIP under full-image preprocessing for complete consistency.

**This also decides an R1-Q3 input.** R1-Q3 joins each image's classification prediction to its attention alignment, so both should describe the same input. If the checkpoints are kept, R1-Q3 must choose between predicting on the center crop (matches training and the `best_val_acc` sanity check, but a different view from the attention) and predicting on the full image (matches the attention, but a distribution shift, so the sanity check has to run separately on the crop). Retraining makes all three consistent, but it changes RQ3 again. See "Sequencing with R1-Q4", item 3, in [`R1-Q3_plan.md`](R1-Q3_plan.md).

## Sequencing

1. **Resolve the MAE discrepancy before the Q2 rerun.** For MAE LoRA and MAE Full at layer 11, `metrics.db` (modified Jun 13) and `q2_metrics_analysis.json` (Mar 28) disagree. MAE LoRA IoU@90 is 0.09658 in the DB and 0.10132 in the JSON. The other 16 runs agree within 0.1%. `analyze_q2_metrics.py` recomputes attention from the checkpoints (`load_finetuned_model`, line 473) rather than reading `metrics.db`. It cannot be limited to DINOv2 and CLIP, because Holm correction needs the full 18-run family. Rerunning it therefore overwrites the March JSON, and with it the evidence of which source matches the selected checkpoints. Complete R1-Q3 Step 1 first.

   **The rerun writes more than its output file.** When an experiment ID resolves (it defaults to the active experiment, line 788), the script also overwrites `q2_delta_iou_analysis.json` and rewrites `run_matrix.json` and `outputs/results/active_experiment.json` (lines 944–957, `update_experiment_analysis_outputs`). Passing `--experiment-id` explicitly also calls `refresh_experiment_training_provenance` (line 790), which rewrites `manifests/` and `fine_tuning_results.json`. Passing `--output` does not isolate a run, and a partial run such as `--models mae` would replace the experiment's delta file with that subset. Before and during the rerun:
   - All of these JSON files are tracked by git and unmodified since `924961f` (2026-03-28). Record that commit as the baseline, start with a clean `git status` for `outputs/results/`, and review the diff after the rerun.
   - `outputs/cache/metrics.db` is gitignored. Copy it to a dated backup before regenerating the DINOv2 and CLIP rows.
   - Run only the full 18-run Q2 analysis against the experiment. Do one-off checks, such as the MAE recompute, from a scratch script.
2. **Finish this fix before R1-Q3 finalizes DINOv2 and CLIP.** R1-Q3 reads alignment from the refreshed caches, and the classification input follows the decision above.

## Out of scope

- Letterbox or padding experiments to test sensitivity to aspect ratio.
- R1-Q10 (choice of attention-map interpolation).

## Found while planning

- The `siglip2.py` docstring's NaFlex rationale is wrong for the configured checkpoint. It is harmless, but worth a one-line correction while that file is open.
- `uv run` reinstalls one package on every call, which suggests that `uv.lock` (modified in the working tree) is out of sync with the environment. This is unrelated to Q4.

# Plan: R1-Q3, classification accuracy vs. attention alignment

Status (2026-10-25): Steps 1–6 are done, tracked as beads issue `ssl_wikichurches-4myv`. The manuscript edits and both PDFs are waiting for human approval. The reply-letter text (Step 7) and the commit (Step 8) follow approval. Nothing is committed yet.

Results: `experiments/scripts/analyze_accuracy_alignment.py` wrote `classification_predictions.json` and `accuracy_alignment_analysis.json`. Both checks passed. Recomputed validation accuracy equals `best_val_acc` for all 18 runs, and all 90 run-by-metric alignment means equal the layer-11 rows of `q2_metrics_analysis.json`. `experiments/scripts/generate_paper_accuracy_alignment.py` wrote `RQ3_accuracy_alignment.png`. The manuscript has new Section 4.4.1, Figure 9, Table 7 and Appendix Table A1, plus additions to Implementation Details and Limitations.

## The item

Reviewer 1, question 3 (`R1-Q3`) in the round-1 reply letter: *XAI Vision round1 reply to all reviewers 1006.docx* in Google Drive (file ID `1z1jEHklwWdebqoxusVEL7Pt5vTpvtQFf`). See [`AGENTS.md`](../AGENTS.md) for how the letter is organized, the scope rules and the revision workflow.

**Q3 (reviewer):**

> Although you mentioned that the work focuses on analyzing the human-alignment of the models, it should not completely forget the accuracy domain, as it's a main evaluation metric. In this regard, it would be highly valuable to relate the model's classification correctness to its human-expert attention alignment. Is spatial plausibility correlated with correct prediction?

**A3 (TJ's note):** include both quality and alignment score in one figure. Do we have classification accuracy scores (can we use zero-shot)?

**TODO:** evaluate classification accuracy on the architecture type labels. Plan B is to copy-paste reported performance for the frozen model.

Manuscript files: the baseline is [`round2_v1024.tex`](../round2_v1024.tex) (never edit it). All edits go in [`round2_v1024_changes.tex`](../round2_v1024_changes.tex).

## Dependency on R1-Q7

R1-Q7 says the linear probe is pointless in RQ3 because the backbone is frozen. We confirmed this: in `outputs/results/experiments/fine_tuning_primary_20260327/q2_metrics_analysis.json`, all 54 linear-probe rows have a per-image alignment change of exactly 0.0. Agreed decision for R1-Q7: drop the linear probe as an RQ3 adaptation strategy, but keep its accuracy as the **frozen-model accuracy**.

This plan relies on that decision. The "Frozen" point for each model uses:

- **accuracy** from the linear-probe checkpoint (frozen backbone + trained 4-class head), and
- **alignment** from the frozen model's attention, which is identical to the linear probe's attention.

This replaces TJ's zero-shot idea (only CLIP and SigLIP have text encoders) and Plan B (published numbers come from other datasets and setups). The R1-Q7 manuscript edits are a separate item. Here, label these points "Frozen", not "Linear probe".

## Requested changes

These answer the reviewer's question and TJ's request for one figure.

| Topic | Decision |
|---|---|
| Unit of analysis | Both: a per-model accuracy-vs-IoU@90 figure (TJ's "one figure") and a per-image correct-vs-wrong comparison (the reviewer's literal question). |
| Images used | The 139 annotated evaluation images for both accuracy and alignment, so both come from the same images. |
| Labels | `building_parts.json` annotation labels for the 139 evaluation images. See correctness fix 2. |
| Accuracy metric | Plain accuracy and balanced accuracy on the 139 images. |
| Correctness | Binary only (correct / wrong). |
| Alignment metric | IoU@90 in the main text. Coverage, MSE, KL and EMD in a supplementary table. |
| Attention method and layer | Same as the RQ3 analysis: `cls` for DINOv2, DINOv3, MAE and CLIP; `mean` for SigLIP and SigLIP2; layer 11. |
| Runs | 18: 6 models × {Frozen (linear-probe head), LoRA, Full}. |
| Statistical test | Two-sided Mann-Whitney U per run, comparing the distributions of per-image IoU@90 for correct and wrong images. Holm correction across the 18 runs. |
| Paper placement | No new research question. One short closing subsection in the RQ3 results, plus one sentence in Implementation Details. |

## Necessary correctness fixes

Each fix below addresses a concrete error or inconsistency that would make the requested result wrong if left alone.

### 1. Reconcile the two per-image alignment sources (prerequisite)

**Error.** The per-image alignment values in `outputs/cache/metrics.db` (table `image_metrics`, layer 11) do not match `q2_metrics_analysis.json` for two MAE runs:

| Run | Metric | `q2_metrics_analysis.json` | `metrics.db` | Relative difference |
|---|---|---|---|---|
| MAE LoRA | IoU@90 | 0.10132 | 0.09658 | 4.7% |
| MAE LoRA | Coverage | 0.11506 | 0.11895 | 3.4% |
| MAE LoRA | MSE | 0.07240 | 0.04849 | 33.0% |
| MAE Full | IoU@90 | 0.09875 | 0.09867 | 0.08% |
| MAE Full | IoU@70 | 0.12015 | 0.11696 | 2.7% |
| MAE Full | MSE | 0.08825 | 0.07508 | 14.9% |

The remaining metrics and percentiles for these two runs differ by 0.3% to 5.2%. The other 16 runs agree within 0.1% on every metric and percentile (IoU@90 within about 2e-5 absolute), which is consistent with float noise. `metrics.db` was last modified on Jun 13 2026, after the checkpoints (Mar 27) and the Q2 JSON (Mar 28). That is a lead, not a diagnosis.

**Why it matters.** The per-image join for the correct-vs-wrong comparison uses `metrics.db`, while the paper's RQ3 numbers come from the Q2 analysis. If the MAE rows don't correspond to the selected checkpoints, the MAE points in the figure and the MAE rows in the table would pair predictions from one model with attention from another.

**Fix.** Before joining predictions to alignment:

1. Find which source was produced from the selected checkpoints in `outputs/checkpoints/fine_tuning_primary_20260327/`. For example, regenerate layer-11 metrics for MAE LoRA and MAE Full from those checkpoints and compare with both sources.
2. Use only a source that matches the selected checkpoints, for all 18 runs.
3. Do not assume either source is correct, and do not loosen the verification tolerance to hide the difference.
4. If the Q2 JSON turns out to be the stale one, the paper's RQ3 MAE numbers are affected. Flag that as a separate item. Do not fix it within R1-Q3.

**Resolved (2026-10-25):** `metrics.db` is the stale source. See Step 1 for the evidence and Step 3 for the alignment source now used.

### 2. Use one documented label source per image set

**Error.** Two label sources exist, and they disagree on 4 evaluation images:

- `dataset/building_parts.json`: annotation labels, read by `AnnotatedSubset` in `src/ssl_attention/data/wikichurches.py`.
- `dataset/churches.json`: metadata labels, read by `FullDataset`. Training and validation used these.

| Image | `building_parts.json` | `churches.json` |
|---|---|---|
| `Q1165020_wd0.jpg` | Renaissance | Baroque |
| `Q13098929_wd0.jpg` | Gothic | Romanesque |
| `Q2713844_wd0.jpg` | Romanesque | Renaissance |
| `Q719794_wd0.jpg` | Renaissance | Baroque |

**Fix.**

- **139 evaluation images:** use the `building_parts.json` labels as the authority. They come with the expert bounding boxes, and all 139 images have one of the four styles there (Romanesque 51, Gothic 49, Renaissance 22, Baroque 17).
- **917 validation images:** use the `churches.json` labels, as training did, so the recomputed validation accuracy can be checked against `best_val_acc`.
- Record the 4 conflicts and this choice in the analysis output and in one sentence of Implementation Details. Do not add a sensitivity analysis. None of the 4 images was used for training or validation, so there is no leakage.

**The 125 vs. 139 counts.** All 139 annotated images were held out. Only 125 of them were in the 4,713-image training candidate pool (images whose `churches.json` label is one of the four styles). The other 14 have a four-style label only in `building_parts.json`. Removing the 125 from the pool leaves 4,588 images: 3,671 training and 917 validation. Manuscript line 358 ("After removing the 125 expert-annotated evaluation images") is therefore correct, and no change to it is needed for R1-Q3.

### 3. Verify Frozen alignment at layer 11, not against the RQ1 table

**Error.** The RQ1 table reports each model's best layer per metric ("using the best default-method layer selected separately for each metric"). This analysis uses layer 11. For example, frozen CLIP IoU@90 is 0.049 in the RQ1 table but 0.0181 at layer 11.

**Fix.** Check the Frozen rows against `frozen_mean` at layer 11 in `q2_metrics_analysis.json` (IoU at percentile 90 and the matching method), not against the RQ1 table.

## Sequencing with R1-Q4

R1-Q4 found that DINOv2 and CLIP center-crop their inputs while the expert masks are drawn over the full image resized to 224×224. See [`R1-Q4_plan.md`](R1-Q4_plan.md) and beads issue `ssl_wikichurches-f9rh`. Its fix gives every model the full image with no crop, regenerates the `dinov2*` and `clip*` attention and metrics caches, and reruns the Q2 analysis and paper results. The two plans write to the same files (`outputs/cache/metrics.db` and `q2_metrics_analysis.json`), so their order matters.

### 1. Reconcile MAE before R1-Q4 reruns the Q2 analysis

R1-Q4's step 4 reruns `experiments/scripts/analyze_q2_metrics.py`. That script does not read `metrics.db`. It recomputes attention directly from the selected checkpoints (`load_finetuned_model`, line 473) and overwrites the March `q2_metrics_analysis.json`. Because Holm correction needs the full 18-run family, the rerun covers all runs, including MAE. Either way, the current JSON is lost, and with it the evidence needed to compare the two sources. Therefore, finish Step 1 of this plan before R1-Q4's Q2 rerun. The March outputs are already preserved in git (see Step 1).

The two sources were produced differently, which is useful for Step 1: the JSON (Mar 28) is a direct recompute on the selected checkpoints, while `metrics.db` (Jun 13) came from the separate cache pipeline (`app/precompute/generate_metrics_cache.py`). This is a lead, not a conclusion.

### 2. Finalize DINOv2 and CLIP only after R1-Q4 lands

R1-Q4 changes the per-image alignment of 6 of the 18 runs: DINOv2 and CLIP × Frozen, LoRA, Full. Their figure points, table rows and correct-vs-wrong tests must use the regenerated caches. The verification values for these runs (layer-11 `frozen_mean` and `finetuned_mean`) must come from the rerun Q2 JSON, not the current one.

### 3. Decide which input DINOv2 and CLIP classify

After the R1-Q4 fix, DINOv2 and CLIP attention comes from the full image, but their checkpoints were trained and validated on center crops. The per-image join requires the prediction and the attention to describe the same image input. This decision follows R1-Q4's open decision 2 (keep the checkpoints or retrain):

- **Checkpoints kept, predictions on the center crop.** This matches training, and the Step 2 validation-accuracy sanity check applies as written. The prediction and the attention then come from different views of the same image. State this in one clause in the manuscript.
- **Checkpoints kept, predictions on the full image.** This matches the attention input, but it is a distribution shift for these checkpoints. Recomputed validation accuracy is expected to differ from `best_val_acc`, so the sanity check must run on the center-crop input instead, as a separate check.
- **Checkpoints retrained on the full image.** Prediction, attention and training all use the same input, and the sanity check applies as written. All six DINOv2 and CLIP runs must be retrained, and their RQ3 results change again.

Record the choice in `accuracy_alignment_analysis.json` and in the Implementation Details sentence. DINOv3, MAE, SigLIP and SigLIP2 already resize the full image, so this decision does not affect them.

**Decided (2026-10-25):** R1-Q4 kept the checkpoints. DINOv2 and CLIP predictions use the native center crop (`FineTunableModel.preprocess_for_classification`), so they match training and the Step 2 sanity check applies to all 18 runs. As a robustness check, the script also predicts the 139 images on the full image (`FineTunableModel.preprocess`) for the six DINOv2 and CLIP runs and reruns the correct-vs-wrong tests with those predictions. These results go in the JSON only. The manuscript mentions them in one clause only if a significance decision changes.

### What can start before R1-Q4 lands

- Step 1 (MAE reconciliation).
- Writing the Step 2 script, including the label handling in correctness fix 2.
- Running predictions and the Step 3 analysis for DINOv3, MAE, SigLIP and SigLIP2.

## Approach

### Step 1: Reconcile the alignment source

**Done (2026-10-25), run by the R1-Q4 session:** the Q2 JSON matches the selected checkpoints, and `metrics.db` does not. A read-only scratch script imported `analyze_q2_metrics.py`, loaded `mae_{lora,full}_finetuned.pt` with `load_finetuned_model`, and ran `compute_model_metric_values` at layer 11 on the 139 images. Database values were read from `outputs/cache/metrics.db.2026-10-25-before-crop-fix.bak`. `git status outputs/results/` was clean before and after.

| Run, metric (layer 11, `cls`) | Fresh recompute | Q2 JSON | `metrics.db` |
|---|---|---|---|
| MAE LoRA, IoU@90 | 0.101316 | 0.101316 | 0.096582 |
| MAE Full, IoU@90 | 0.098752 | 0.098752 | 0.098666 |
| MAE LoRA, IoU@70 | 0.123293 | 0.123293 | 0.117882 |
| MAE LoRA, MSE | 0.072402 | 0.072402 | 0.048491 |
| MAE Full, MSE | 0.088251 | 0.088251 | 0.075076 |

Fresh minus JSON was exactly 0 on every metric checked. The stale `mae_finetuned_*` rows in `metrics.db` are filed as beads bug `ssl_wikichurches-z74t` (P1). The MAE processor is unchanged by the R1-Q4 crop fix, so this result holds before and after it. Consequence: Step 3 does not use `metrics.db` (see Step 3).

The original instructions are kept below for the record.

Carry out correctness fix 1. Record which source matches the selected checkpoints, and how this was established, in the analysis output. Do not start Step 3 until this is resolved. Finish this step before R1-Q4 reruns the Q2 analysis (see "Sequencing with R1-Q4", item 1).

1. **Confirm the baseline.** The March results are tracked in git and unchanged since commit `924961f` (2026-03-28): `q2_metrics_analysis.json`, `q2_delta_iou_analysis.json`, `run_matrix.json`, `fine_tuning_results.json`, `manifests/` (all in `outputs/results/experiments/fine_tuning_primary_20260327/`) and `outputs/results/active_experiment.json`. Check that `git status outputs/results/` is clean before running anything, and run `git diff outputs/results/` afterwards. Both should show no changes. `outputs/cache/metrics.db` is gitignored and not covered. This step only reads it, but R1-Q4 backs it up with a dated copy before regenerating its DINOv2 and CLIP rows. Do not treat the existing `metrics.db.bak` as the March state without checking it.
2. **Recompute MAE from a scratch script.** Do not run `analyze_q2_metrics.py` directly, even with `--models mae --strategies lora full`. No flag combination isolates it:
   - `--output` alone: an experiment ID still resolves from the active experiment (line 788), so the script also overwrites `q2_delta_iou_analysis.json` and rewrites `run_matrix.json` and `active_experiment.json` (lines 944–957). An MAE-only run would replace the delta file with MAE rows only and repoint the active experiment.
   - An explicit `--experiment-id`: this also calls `refresh_experiment_training_provenance` (line 790), which rewrites `manifests/`, `fine_tuning_results.json` and `run_matrix.json`.

   Instead, import the script's recompute functions into a scratch script and write all outputs to the scratch directory.
3. **Compare.** Compare the fresh MAE LoRA and MAE Full per-image values with both the archived JSON and `metrics.db`. The source that matches the fresh recompute within float noise is the one to use.

### Step 2: Predict

New script `experiments/scripts/analyze_accuracy_alignment.py`, following the existing `analyze_*.py` scripts:

1. For each of the 18 checkpoints in `outputs/checkpoints/fine_tuning_primary_20260327/`, load the model with `load_finetuned_model` (`src/ssl_attention/evaluation/fine_tuning.py`). Run inference on the 139 annotated images and the 917 validation images. Use the same deterministic preprocessing as the training loop's validation pass (no augmentation). For DINOv2 and CLIP, the input follows the decision in "Sequencing with R1-Q4", item 3.
2. Image IDs come from the split artifact `outputs/results/experiments/fine_tuning_primary_20260327/splits/fine_tuning_primary_20260327__primary__seed42.json` (`annotated_eval_image_ids`, `val_image_ids`). Labels follow correctness fix 2.
3. Save per-image results to `outputs/results/experiments/fine_tuning_primary_20260327/classification_predictions.json`: run ID, image ID, split (`annotated_eval` or `val`), label source, true label, predicted label, correct flag.
4. **Sanity check:** recomputed accuracy on the 917 validation images must match `best_val_acc` in `outputs/results/experiments/fine_tuning_primary_20260327/fine_tuning_results.json` for every run. A mismatch means preprocessing or checkpoint loading is wrong. Stop and investigate before continuing.

### Step 3: Analyze

In the same script:

1. Join the 139-image predictions with per-image alignment on `image_id`: layer 11, the model's method, percentile 90 for IoU. Compute the per-image alignment in this script with `compute_model_metric_values` from `analyze_q2_metrics.py`, for all 18 runs, after R1-Q4's crop fix has landed. Use the frozen model for the Frozen run. Do not read `metrics.db`. Its stale MAE LoRA and Full rows have since been regenerated (`ssl_wikichurches-z74t`, closed), but recomputing keeps every alignment value on the same code path as the paper's RQ3 numbers.
2. Per run, compute:
   - plain and balanced accuracy on the 139 images
   - mean IoU@90 over all 139 images (the figure's y-axis)
   - number of correct and wrong images, and median IoU@90 in each group
   - the Mann-Whitney U p-value, then Holm-adjusted across the 18 runs
3. Repeat the correct-vs-wrong comparison for Coverage, MSE, KL and EMD, with Holm correction per metric, for the supplementary table.
4. Write `outputs/results/experiments/fine_tuning_primary_20260327/accuracy_alignment_analysis.json`. Include the label-conflict list and the alignment-source decision. If a run has fewer than 2 wrong images, report the test as not applicable instead of a p-value.

### Step 4: Figure

One figure, saved in `docs/mva/`:

- x-axis: accuracy on the 139 annotated images. y-axis: mean IoU@90 on the same images.
- 18 points. Draw two arrows per model, Frozen→LoRA and Frozen→Full. LoRA and Full are separate adaptations of the same frozen model, so they are not joined to each other.
- A distinct marker shape per model, as well as colour, consistent with the R2-Q5 accessibility fix.

### Step 5: Manuscript edits (`round2_v1024_changes.tex` only)

Keep the additions small. Do not change the research questions, abstract or introduction.

1. **Implementation Details** (around lines 362–366): one or two sentences. Frozen-model accuracy comes from the frozen backbone with the trained 4-class head. Accuracy is measured on the 139 annotated images using the `building_parts.json` style labels.
2. **RQ3 results:** a short closing subsection, for example "Relation to classification correctness", with the figure, one table and one paragraph. The table has one row per run: accuracy, balanced accuracy, number of correct and wrong images, median IoU@90 for each group, Holm-adjusted p.
   - Describe the test as comparing the distributions of per-image IoU@90 between correct and wrong images.
   - Limit conclusions to the observed association. It is not adjusted for confounders such as style or bounding-box size, and it does not show that alignment causes correct predictions.
3. **Supplementary/appendix:** the correct-vs-wrong table for Coverage, MSE, KL and EMD, next to whatever appendix R1-Q17 creates.
4. **Limitations:** add a sentence about the small number of misclassified images only if the observed predictions show small wrong groups. Base the wording on the actual counts.

Every number must come from `accuracy_alignment_analysis.json`. Never estimate or invent results.

### Step 6: Build and review

Build the clean and tracked-changes PDFs with the commands in [`AGENTS.md`](../AGENTS.md). Check the new subsection, figure and table in both PDFs. Show the change to the human and wait for approval.

### Step 7: Reply letter

After approval, draft one to three sentences for A3 saying what was added and where (subsection, figure and table numbers). Give the text to the human to paste, because the Drive connector cannot write `.docx` content. Leave TJ's note and the TODO in place.

### Step 8: Commit

Commit the `.tex` file, the new figure PNG and the analysis script. Do not commit `build/`.

## Verification

- Step 1 (done): the Q2 JSON matches a fresh recompute on the selected MAE checkpoints exactly; `metrics.db` does not.
- The Step 2 sanity check passes for all 18 runs: recomputed validation accuracy equals `best_val_acc`.
- Frozen rows: mean IoU@90 matches layer-11 `frozen_mean` in `q2_metrics_analysis.json`.
- LoRA and Full rows: mean IoU@90 matches `finetuned_mean` in `q2_metrics_analysis.json` for all 12 runs, using the JSON from R1-Q4's rerun.
- `uv run ruff check .` and `uv run mypy` pass for the new script.

## Optional / out of scope

Optional, not part of the required work:

- A validation-accuracy column (917 images) in the manuscript table. If included, label it "validation accuracy", because these images were used to select the checkpoints. The recomputation is already required as the Step 2 sanity check.

Out of scope for R1-Q3:

- A new research question, or changes to the abstract and introduction.
- Correlation with predicted probability or confidence.
- Confound analysis by style or bounding-box size (bounding-box size belongs to R1-Q5).
- A sensitivity analysis on the 4 label conflicts.
- k-NN or zero-shot classification.
- The R1-Q7 removal of the linear probe from the RQ3 figures and text.

## Flagged for other items

- **RQ3 MAE numbers:** resolved. Step 1 found that the Q2 JSON, which the paper's RQ3 numbers come from, matches the selected checkpoints. The stale rows are in `metrics.db` only (`ssl_wikichurches-z74t`), which also feeds the app.

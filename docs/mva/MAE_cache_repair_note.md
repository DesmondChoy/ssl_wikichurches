# Review note: MAE cache repair outside a reviewer question

Status (2026-10-25): repair verified independently and completed under `ssl_wikichurches-z74t` (closed). The related checkpoint-default fix is complete under `ssl_wikichurches-px47` (closed), commit `3736641`.

**For the manuscript review: z74t changed no manuscript numbers or wording.** The repair corrected stored analysis results that disagreed with the selected MAE checkpoints. The results already used in the paper were correct. Fig. 8 regenerated identically, and the existing RQ3 MAE sentence and Table 5 remain valid. No numerical change in the combined LaTeX diff should be attributed to z74t.

## The item

This was an internal correctness issue discovered while preparing the journal revision, rather than a direct request from a reviewer. It surfaced during the R1-Q4 preprocessing review and was also a prerequisite for the R1-Q3 analysis relating classification correctness to attention alignment. The original concern appears in [`R1-Q4_plan.md`](R1-Q4_plan.md) and [`R1-Q3_plan.md`](R1-Q3_plan.md); this note records its resolution.

The relevant manuscript is [`round2_v1024_changes.tex`](round2_v1024_changes.tex). The submitted baseline, [`round2_v1024.tex`](round2_v1024.tex), was not edited for this repair.

## Why the repair was necessary

The cached MAE LoRA and full fine-tuning attention maps did not all come from the selected checkpoints in `outputs/checkpoints/fine_tuning_primary_20260327/`. The maps combining attention across heads, called **fused maps**, matched superseded checkpoints in the root `outputs/checkpoints/` directory. Their corresponding database metrics therefore described a different trained model from the one selected for the paper.

The [selected experiment's Q2 analysis](../../outputs/results/experiments/fine_tuning_primary_20260327/q2_metrics_analysis.json) computed attention directly from the selected checkpoints and was already correct. The individual-head results were also already correct. This explains how the application cache could disagree with the Q2 analysis while the paper's figures and head rankings remained valid. The root-level `outputs/results/q2_metrics_analysis.json` is an older run; the experiment-scoped file is the comparison source for this repair.

The cache scripts previously defaulted to the root checkpoint directory. Commit `3736641` fixes that default in both [`generate_attention_cache.py`](../../app/precompute/generate_attention_cache.py) and [`generate_feature_cache.py`](../../app/precompute/generate_feature_cache.py). Both now default to the selected March experiment directory. This code fix was recorded separately in px47.

## Which numbers changed

The following are **cache database values, not replacements for numbers printed in the manuscript**. They are mean scores over the same 139 images, at layer 11 using the `cls` attention method. Values are shown to six decimal places.

| MAE variant | Metric | Cache before repair | Repaired cache | Existing Q2 result |
|---|---|---:|---:|---:|
| LoRA | IoU@90 | 0.096582 | 0.101311 | 0.101316 |
| Full fine-tuning | IoU@90 | 0.098666 | 0.098748 | 0.098752 |
| LoRA | MSE | 0.048491 | 0.072402 | 0.072402 |
| Full fine-tuning | MSE | 0.075076 | 0.088251 | 0.088251 |

The repaired cache stores attention in float16, whereas the Q2 analysis evaluates fresh model outputs. Small numerical differences remain after caching. All 18 comparisons covering the five IoU thresholds, coverage, MSE, KL divergence and EMD for the two MAE variants pass the agreed relative tolerance of `1e-4` (0.01%). The largest observed relative difference is approximately 0.00444%.

The pre-repair database was preserved as `outputs/cache/metrics.db.2026-10-25-before-mae-fix.bak`. The existing Q2 JSON was not rerun or overwritten as part of z74t.

## Effect on the manuscript

Figure and table numbers here refer to the manuscript reviewed for z74t; later revisions may renumber them.

| Manuscript item | Source or verification | Outcome for z74t |
|---|---|---|
| Fig. 6, RQ3 adaptation heatmap | Existing experiment-scoped Q2 analysis | No change required. |
| Fig. 7, RQ3 confidence-interval plot | Existing experiment-scoped Q2 analysis | No change required. |
| Table 5, per-style adaptation results | Independently checked against the selected experiment's style analysis | No change required. |
| Fig. 8, dominant layer and head locations | Individual-head scores in `head_summary_metrics` | Regenerated PNG is byte-for-byte identical. |
| RQ3 MAE dominant-head sentence | Dominant heads verified before and after regeneration | Existing wording remains correct. |

For both fine-tuned MAE variants, all 1,584 head-summary means were unchanged after regeneration. The dominant heads remain:

| Setting | Dominant head |
|---|---|
| Frozen MAE | `layer10/head5` |
| MAE LoRA | `layer11/head7` |
| MAE full fine-tuning | `layer10/head5` |

Both the clean and tracked-changes PDFs were rebuilt and the RQ3 sentence and Fig. 8 were inspected on pages 21–22 in those builds. Those page numbers refer to the builds made during z74t; later revisions may move the material.

## Verification and evidence

The MAE LoRA/full attention, metrics, heatmaps and features were regenerated from the selected checkpoints. An independent Claude audit then checked all 139 images on MPS, the Mac's GPU backend. It confirmed that the root checkpoints reproduce the old fused database results, the selected March checkpoints reproduce the old individual-head results, and the repaired caches agree with the selected checkpoints. Ruff, mypy and the 49 relevant tests also passed during the repair.

The regeneration added 1,980,000 MAE rows to `head_feature_image_metrics`, which previously had no rows for these two variants. These are new cache entries, rather than replaced rows or new manuscript results. The independent audit checked their consistency with the existing feature summaries. A test that merely checks for newer row IDs cannot establish replacement for a table with no previous rows for these variants.

The following evidence files are local, gitignored outputs retained in this checkout:

- [Before-repair cache means](../../outputs/cache/z74t-mae-fix-2026-10-25/before-metrics.json) and [the repaired-cache/Q2 comparisons](../../outputs/cache/z74t-mae-fix-2026-10-25/metrics-verification.json).
- [Head rankings and comparison with the backup](../../outputs/cache/z74t-mae-fix-2026-10-25/head-ranking-verification.json).
- [Cache completeness checks](../../outputs/cache/z74t-mae-fix-2026-10-25/completeness-verification.json) and [attention/feature array checks](../../outputs/cache/z74t-mae-fix-2026-10-25/arrays-verification.json). Interpret the row-ID checks with the new-entry distinction above.

The closed z74t issue and its entry in the tracked [Beads interaction log](../../.beads/interactions.jsonl) retain the verified repair summary. The default-directory code change is in `3736641`; the z74t closure record is in `acbb6a6`.

## How to interpret other numerical changes in the LaTeX diff

Other corrections made during the revision can change manuscript numbers. They should be explained under their own items:

- **R1-Q4 / `ssl_wikichurches-f9rh`:** the full-image preprocessing correction for DINOv2 and CLIP changed their attention-alignment evaluation and affected derived comparisons. Its rationale and manuscript scope are documented in [`R1-Q4_plan.md`](R1-Q4_plan.md).
- **`ssl_wikichurches-m0c8`:** the style analysis counted some images in more than one style group. Using one primary style per image corrected Table 5 and the Kruskal–Wallis results. The fix is recorded in `22fb002`, and the corresponding manuscript update is in `d3e8512`.

These are separate from the MAE cache repair. In particular, a change to a MAE per-style number elsewhere in the combined revision should not be explained as an effect of z74t.

## Suggested explanation for the professor

> During revision checks, we found that the stored MAE LoRA/full attention results combined across heads came from superseded checkpoints. We regenerated the affected caches using the selected checkpoints and verified them independently against the existing analyses. The Q2 results and individual-head results already used in the manuscript were correct, so this repair required no changes to manuscript numbers, Fig. 8 or the RQ3 MAE head description. Numerical changes from the separate preprocessing and style-grouping corrections are documented under those items.

This note supplies context for the internal manuscript review. It does not add a response to a reviewer question or update the reply letter.

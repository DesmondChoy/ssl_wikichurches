# Fine-Tuning Run Matrix

This page defines the canonical artifact layout for fine-tuning experiment batches. The generated experiment batch under `outputs/results/experiments/` is the source of truth, and `outputs/results/active_experiment.json` selects which batch the app, figure scripts, and reporting workflows read by default.

## Primary methodology

The primary fine-tuning experiment uses one clean rule:

- train on the non-annotated style-labeled pool
- reuse one shared stratified validation split across every `model × strategy`
  run in the same experiment batch
- choose one checkpoint per run by best classification validation accuracy
- evaluate all attention metrics from that checkpoint on the untouched annotated
  evaluation pool

This keeps checkpoint selection separate from the final attention-alignment
reporting set.

Use `--val-on-annotated-eval` only for explicit exploratory runs. Those runs are marked `exploratory` in manifests and downstream results, and they are separate from the primary source used for the app, figures, slides, and `/q2`.

## Canonical artifact layout

For an experiment ID such as `fine_tuning_primary_20260327`, the pipeline writes:

```text
outputs/
├── checkpoints/
│   └── fine_tuning_primary_20260327/
│       ├── clip_linear_probe_finetuned.pt
│       ├── clip_lora_finetuned.pt
│       └── ...
└── results/
    ├── active_experiment.json
    └── experiments/
        └── fine_tuning_primary_20260327/
            ├── fine_tuning_results.json
            ├── q2_metrics_analysis.json
            ├── q2_delta_iou_analysis.json
            ├── style_breakdown.json
            ├── style_breakdown.png
            ├── model_correlation.json
            ├── model_correlation_scatter.png
            ├── model_correlation_heatmap.png
            ├── feature_delta_iou_mae_full_renaissance.json
            ├── feature_delta_iou_mae_full_renaissance.png
            ├── run_matrix.json
            ├── manifests/
            │   └── <run_id>_manifest.json
            └── splits/
                └── <split_id>.json
```

## Split artifact

Each experiment batch has one shared split artifact for the primary run scope.
It records:

| Field | Meaning |
|---|---|
| `split_id` | Stable identifier for the split within the batch |
| `experiment_id` | Owning experiment batch |
| `seed` | Random seed used to create the split |
| `dataset_root` | Repo-relative dataset root |
| `dataset_version_hint` | Lightweight dataset fingerprint for provenance |
| `policy` | Split policy, usually `random_stratified_excluding_annotated_eval` |
| `exclude_annotated_from_train` | Whether annotated images were removed from train |
| `exclude_annotated_from_val` | Whether annotated images were removed from validation |
| `annotated_eval_image_ids` | Annotated holdout image IDs |
| `train_image_ids` | Non-annotated training image IDs |
| `val_image_ids` | Non-annotated validation image IDs |
| `train_class_counts` | Per-style counts for the training split |
| `val_class_counts` | Per-style counts for the validation split |
| `created_at` | UTC timestamp |

The key fairness property is that every run in the same batch reuses the same
`train_image_ids` and `val_image_ids`.

## Run manifest

Each `model × strategy` run produces one manifest with the fields that matter for
reproducibility and downstream analysis:

| Field | Meaning |
|---|---|
| `run_id` | Stable run identifier within the batch |
| `experiment_id` | Owning experiment batch |
| `run_scope` | `primary` or `exploratory` |
| `model` | Base model key |
| `strategy` | `linear_probe`, `lora`, or `full` |
| `split_id` | Shared split artifact used by the run |
| `checkpoint_path` | Repo-relative checkpoint path |
| `manifest_path` | Repo-relative path to the manifest itself |
| `split_artifact_path` | Repo-relative split artifact path |
| `training_git_commit_sha` | Code revision that produced the checkpoint and training artifact |
| `checkpoint_selection_metric` | Primary rule, currently classification validation accuracy |
| `checkpoint_selection_split` | Human-readable split policy used for selection |
| `selected_epoch` | Epoch of the chosen checkpoint |
| `best_val_score` | Best validation accuracy |
| `split.*` | Train/val/excluded counts and validation source |

## Run matrix

`run_matrix.json` is the single source of truth for figure generation and
documentation-facing reporting.
It contains one entry per run keyed by `run_id`.

Each run entry stores:

| Field | Meaning |
|---|---|
| `experiment_id` | Owning experiment batch |
| `run_id` | Stable run identifier |
| `model` | Base model key |
| `strategy` | Fine-tuning strategy |
| `split_id` | Shared split artifact |
| `checkpoint_path` | Repo-relative checkpoint path |
| `selected_epoch` | Epoch selected for the run |
| `selection_metric` | Checkpoint selection criterion |
| `checkpoint_selection_split` | Split policy used for selection |
| `best_val_score` | Best validation accuracy |
| `manifest_path` | Repo-relative manifest path |
| `analysis_artifact_paths` | Repo-relative analysis outputs for the run batch |
| `run_scope` | `primary` or `exploratory` |
| `training_git_commit_sha` | Code revision that produced the checkpoint for this run |
| `analysis_git_commit_sha` | Code revision that produced the current batch-level Q2 analysis artifact |

The run matrix is designed so figure-generation scripts do not need hardcoded
tables.

## Q2 analysis artifact

The canonical app-facing analysis artifact is:

- `outputs/results/experiments/<experiment_id>/q2_metrics_analysis.json`

It includes:

| Field | Meaning |
|---|---|
| `experiment_id` | Experiment batch being summarized |
| `split_id` | Shared split artifact used for the batch |
| `analysis_git_commit_sha` | Code revision used to generate the saved Q2 analysis artifact |
| `evaluation_image_count` | Number of annotated evaluation images |
| `checkpoint_selection_rule` | Human-readable primary checkpoint rule |
| `result_set_scope` | `primary` or `exploratory` |
| `rows` | Fine-tuned vs frozen metric rows |
| `strategy_comparisons` | Within-model strategy comparisons |

The git commit that later checks these JSON files into the repository is not
serialized inside the artifact itself; use git history for that provenance.

Each cross-model row also records the correction metadata used for the headline
significance call:

| Field | Meaning |
|---|---|
| `correction_method` | Multiple-comparison method applied to the row (`holm`) |
| `correction_family_id` | Stable identifier for the shared `metric + percentile` correction bucket |
| `correction_family_size` | Number of discovered model-strategy rows corrected together in that bucket |

The image-level Q2 delta export lives in `q2_delta_iou_analysis.json`; `/api/metrics/q2_image_deltas` reads it through the active experiment pointer, and legacy delta-only consumers can still read the same file. The broader app and current Q2 summaries read `q2_metrics_analysis.json`.

## Active experiment pointer

`outputs/results/active_experiment.json` is the app-facing selector. It tells the
backend and reporting scripts which experiment batch to read by default.

Expected fields:

| Field | Meaning |
|---|---|
| `experiment_id` | Selected batch |
| `split_id` | Shared split artifact for the batch |
| `run_matrix_path` | Repo-relative path to `run_matrix.json` |
| `fine_tuning_results_path` | Repo-relative path to the batch ledger |
| `q2_metrics_path` | Repo-relative path to the canonical Q2 artifact |
| `q2_delta_iou_path` | Repo-relative path to the image-level Q2 delta export |
| `updated_at` | UTC timestamp |

The backend and reporting helpers resolve these paths through the active pointer
first and fall back to the legacy repository-level paths when the pointer or a
specific keyed path is absent.

## Refresh workflow

Use the same `EXPERIMENT_ID` across all three strategy sweeps:

```bash
EXPERIMENT_ID=fine_tuning_primary_20260327

uv run python experiments/scripts/fine_tune_models.py --all --freeze-backbone --epochs 3 --experiment-id "$EXPERIMENT_ID"
uv run python experiments/scripts/fine_tune_models.py --all --lora --epochs 3 --experiment-id "$EXPERIMENT_ID"
uv run python experiments/scripts/fine_tune_models.py --all --epochs 3 --experiment-id "$EXPERIMENT_ID"

uv run python experiments/scripts/analyze_q2_metrics.py \
  --experiment-id "$EXPERIMENT_ID" \
  --models clip dinov2 dinov3 mae siglip siglip2 \
  --strategies linear_probe lora full
```

Then regenerate the app-facing caches and reporting assets:

```bash
uv run python -m app.precompute.generate_attention_cache --finetuned --models dinov2 dinov3 mae clip siglip siglip2 --strategies linear_probe lora full
uv run python -m app.precompute.generate_feature_cache --finetuned --models dinov2 dinov3 mae clip siglip siglip2 --strategies linear_probe lora full
uv run python -m app.precompute.generate_heatmap_images --finetuned --models dinov2 dinov3 mae clip siglip siglip2 --strategies linear_probe lora full
uv run python -m app.precompute.generate_metrics_cache --finetuned --models dinov2 dinov3 mae clip siglip siglip2 --strategies linear_probe lora full

uv run python experiments/scripts/generate_run_matrix_figures.py
uv run python experiments/scripts/generate_slide_images.py
cd experiments/scripts && npm install && node create_presentation.js
```

The Q2 investigation scripts use the same active experiment and write supplemental artifacts beside the canonical Q2 files:

```bash
uv run python experiments/scripts/analyze_style_breakdown.py --experiment-id "$EXPERIMENT_ID" --strategy full
uv run python experiments/scripts/analyze_model_correlation.py --experiment-id "$EXPERIMENT_ID" --strategy full
uv run python experiments/scripts/analyze_feature_delta_iou.py --experiment-id "$EXPERIMENT_ID" --model mae --strategy full --style Renaissance
```

## Helpful CLI Flags

| Flag | Script | Purpose |
|---|---|---|
| `--experiment-id <id>` | `fine_tune_models.py`, `analyze_q2_metrics.py` | Reuse one named experiment batch across training and analysis |
| `--freeze-backbone` | `fine_tune_models.py` | Run the `linear_probe` control condition |
| `--lora` | `fine_tune_models.py` | Run the LoRA strategy |
| `--include-annotated-eval` | `fine_tune_models.py` | Use the 139 annotated images as the local training/validation pool when they are the only available dataset |
| `--val-on-annotated-eval` | `fine_tune_models.py` | Produce an explicit exploratory batch selected on the annotated pool |
| `--include-exploratory` | `analyze_q2_metrics.py` | Include exploratory runs in the exported Q2 summary |
| `--output <path>` | `analyze_q2_metrics.py` | Write the analysis JSON to a custom location |

## Reporting outputs

The reporting scripts consume the active experiment plus the cache outputs above
to build the figure and slide assets used in presentations and narrative
summaries.

Primary generated locations:

| Output family | Location | Source |
|---|---|---|
| Run-matrix figures | `outputs/figures/` | `generate_run_matrix_figures.py` |
| Figure commentary | `outputs/figures/commentary.txt` | `generate_run_matrix_figures.py` |
| Slide PNG assets | `outputs/slides/` | `generate_slide_images.py` |
| Presentation deck | `outputs/slides/presentation.pptx` | `create_presentation.js` |

## Result storage map

| Result family | Primary storage | Git-tracked? | Notes |
|---|---|---|---|
| Batch selector | `outputs/results/active_experiment.json` | No | Tells the app and reporting scripts which batch is active |
| Split artifact | `outputs/results/experiments/<experiment_id>/splits/<split_id>.json` | No | Shared canonical split for the batch |
| Run manifests | `outputs/results/experiments/<experiment_id>/manifests/` | No | One manifest per `model × strategy` run |
| Run matrix | `outputs/results/experiments/<experiment_id>/run_matrix.json` | No | Single source of truth for figures/docs |
| Batch ledger | `outputs/results/experiments/<experiment_id>/fine_tuning_results.json` | No | Accumulates every run in the batch |
| Checkpoints | `outputs/checkpoints/<experiment_id>/` | No | Trained weights |
| Q2 analysis | `outputs/results/experiments/<experiment_id>/q2_metrics_analysis.json` | No | Consumed by `/api/metrics/q2_summary` and `/q2` |
| Q2 image-level deltas | `outputs/results/experiments/<experiment_id>/q2_delta_iou_analysis.json` | No | Consumed by `/api/metrics/q2_image_deltas` and legacy delta-only consumers |
| Per-style Q2 analysis | `outputs/results/experiments/<experiment_id>/style_breakdown.json`, `style_breakdown.png` | No | Generated by `analyze_style_breakdown.py` |
| Cross-model Q2 correlation | `outputs/results/experiments/<experiment_id>/model_correlation.json`, `model_correlation_scatter.png`, `model_correlation_heatmap.png` | No | Generated by `analyze_model_correlation.py` |
| Feature-level Q2 delta analysis | `outputs/results/experiments/<experiment_id>/feature_delta_iou_mae_full_renaissance.json`, `feature_delta_iou_mae_full_renaissance.png` | No | Generated by `analyze_feature_delta_iou.py` |
| Legacy fallback summary | `outputs/results/q2_metrics_analysis.json` | No | Repository-level fallback path when the active pointer is absent |
| Legacy fallback image-level deltas | `outputs/results/q2_delta_iou_analysis.json` | No | Repository-level fallback path when the active pointer is absent |
| Run-matrix figures | `outputs/figures/` | No | Figure-generation outputs driven by the active experiment |
| Slide assets | `outputs/slides/` | No | Slide-image outputs driven by the active experiment plus cached heatmaps |
| Presentation deck | `outputs/slides/presentation.pptx` | No | PPTX assembled from generated figures and slide images |
| Human-readable reference | `docs/reference/fine_tuning_run_matrix.md` | Yes | This page explains the artifact contract rather than duplicating volatile numbers |

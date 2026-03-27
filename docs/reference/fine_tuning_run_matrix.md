# Fine-Tuning Run Matrix

This page describes the canonical artifact layout for the fine-tuning experiment.
It no longer hardcodes a snapshot table of one historical batch. The source of
truth is the generated experiment batch under `outputs/results/experiments/`,
with `outputs/results/active_experiment.json` selecting which batch the app,
figure scripts, and docs-refresh tooling should read.

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

`--val-on-annotated-eval` still exists for explicit exploratory runs, but those
runs are marked `exploratory` in manifests and downstream results. They are not
the default source for docs, figures, or `/q2`.

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
| `git_commit_sha` | Code revision used for the run |
| `checkpoint_selection_metric` | Primary rule, currently classification validation accuracy |
| `checkpoint_selection_split` | Human-readable split policy used for selection |
| `selected_epoch` | Epoch of the chosen checkpoint |
| `best_val_score` | Best validation accuracy |
| `split.*` | Train/val/excluded counts and validation source |

## Run matrix

`run_matrix.json` is the single source of truth for figures and docs refreshes.
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
| `git_commit_sha` | Code revision |

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
| `git_commit_sha` | Code revision used for the analysis |
| `evaluation_image_count` | Number of annotated evaluation images |
| `checkpoint_selection_rule` | Human-readable primary checkpoint rule |
| `result_set_scope` | `primary` or `exploratory` |
| `results` | Fine-tuned vs frozen metric rows |
| `pairwise_strategy_comparisons` | Within-model strategy comparisons |

`q2_delta_iou_analysis.json` is retained only as a compatibility export for
older consumers.

## Active experiment pointer

`outputs/results/active_experiment.json` is the app-facing selector. It tells the
backend and figure scripts which experiment batch to read by default.

Expected fields:

| Field | Meaning |
|---|---|
| `experiment_id` | Selected batch |
| `split_id` | Shared split artifact for the batch |
| `run_matrix_path` | Repo-relative path to `run_matrix.json` |
| `fine_tuning_results_path` | Repo-relative path to the batch ledger |
| `q2_metrics_path` | Repo-relative path to the canonical Q2 artifact |
| `q2_delta_iou_path` | Repo-relative path to the compatibility export |
| `updated_at` | UTC timestamp |

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

Then regenerate the app-facing caches and figures:

```bash
uv run python -m app.precompute.generate_attention_cache --finetuned --models dinov2 dinov3 mae clip siglip siglip2 --strategies linear_probe lora full
uv run python -m app.precompute.generate_feature_cache --finetuned --models dinov2 dinov3 mae clip siglip siglip2 --strategies linear_probe lora full
uv run python -m app.precompute.generate_heatmap_images --finetuned --models dinov2 dinov3 mae clip siglip siglip2 --strategies linear_probe lora full
uv run python -m app.precompute.generate_metrics_cache --finetuned --models dinov2 dinov3 mae clip siglip siglip2 --strategies linear_probe lora full

uv run python experiments/scripts/generate_run_matrix_figures.py
uv run python experiments/scripts/generate_slide_images.py
```

## Result storage map

| Result family | Primary storage | Git-tracked? | Notes |
|---|---|---|---|
| Batch selector | `outputs/results/active_experiment.json` | No | Tells the app and figure scripts which batch is active |
| Split artifact | `outputs/results/experiments/<experiment_id>/splits/<split_id>.json` | No | Shared canonical split for the batch |
| Run manifests | `outputs/results/experiments/<experiment_id>/manifests/` | No | One manifest per `model × strategy` run |
| Run matrix | `outputs/results/experiments/<experiment_id>/run_matrix.json` | No | Single source of truth for figures/docs |
| Batch ledger | `outputs/results/experiments/<experiment_id>/fine_tuning_results.json` | No | Accumulates every run in the batch |
| Checkpoints | `outputs/checkpoints/<experiment_id>/` | No | Trained weights |
| Q2 analysis | `outputs/results/experiments/<experiment_id>/q2_metrics_analysis.json` | No | Consumed by `/api/metrics/q2_summary` and `/q2` |
| Compatibility export | `outputs/results/experiments/<experiment_id>/q2_delta_iou_analysis.json` | No | Legacy consumer support only |
| Human-readable reference | `docs/reference/fine_tuning_run_matrix.md` | Yes | This page explains the artifact contract rather than duplicating volatile numbers |

#!/usr/bin/env python3
"""Classification correctness versus attention alignment for the RQ3 runs (R1-Q3).

For the 18 primary runs (6 models x Frozen, LoRA, Full) this script predicts the
4-class architectural style of the 139 annotated evaluation images and the 917
validation images, then compares per-image attention alignment (IoU@90,
Coverage, MSE, KL, EMD at layer 11) between correctly and wrongly classified
evaluation images with a two-sided Mann-Whitney U test, Holm-corrected across
runs per metric.

Frozen accuracy comes from the linear-probe checkpoint (frozen backbone plus the
trained 4-class head). Frozen alignment comes from the frozen backbone, whose
attention the linear probe shares.

Stage 1 (inference) writes classification_predictions.json. Stage 2 (analysis)
re-reads that file, rejects it unless every prediction pass covers its split's
images exactly once with consistent labels, and writes
accuracy_alignment_analysis.json, with checks of:
  - recomputed validation accuracy against best_val_acc in fine_tuning_results.json
  - recomputed mean alignment against layer-11 rows of q2_metrics_analysis.json
  - per-image LoRA and Full minus Frozen alignment against per_image_deltas there
The script exits non-zero after writing its outputs if any check fails.

It never reads metrics.db and writes nothing besides its two JSON files.

Usage:
  HF_HUB_OFFLINE=1 uv run python experiments/scripts/analyze_accuracy_alignment.py
  HF_HUB_OFFLINE=1 uv run python experiments/scripts/analyze_accuracy_alignment.py --from-predictions
  HF_HUB_OFFLINE=1 uv run python experiments/scripts/analyze_accuracy_alignment.py \\
      --models siglip --output-dir /path/to/scratch
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from collections.abc import Callable
from dataclasses import asdict, dataclass, replace
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

import numpy as np

os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from scipy import stats
from sklearn.metrics import balanced_accuracy_score
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))
sys.path.insert(0, str(project_root))

from experiments.scripts.analyze_q2_metrics import (  # noqa: E402
    METRIC_CONFIGS,
    AnalysisMetric,
    compute_model_metric_values,
    resolve_layer_index,
)
from ssl_attention.config import DATASET_PATH, DEFAULT_METHOD, STYLE_NAMES  # noqa: E402
from ssl_attention.data.wikichurches import AnnotatedSubset, FullDataset  # noqa: E402
from ssl_attention.evaluation.fine_tuning import (  # noqa: E402
    FineTunableModel,
    FineTuner,
    FineTuningConfig,
    FineTuningSplitArtifact,
    get_checkpoint_filename,
    load_finetuned_model,
)
from ssl_attention.evaluation.fine_tuning_artifacts import (  # noqa: E402
    get_experiment_paths,
    get_git_commit_sha,
    project_path,
    save_json,
)
from ssl_attention.metrics.statistics import multiple_comparison_correction  # noqa: E402
from ssl_attention.models import create_model  # noqa: E402
from ssl_attention.utils.device import clear_memory  # noqa: E402

if TYPE_CHECKING:
    from ssl_attention.models.base import BaseVisionModel

Strategy = Literal["frozen", "lora", "full"]
Split = Literal["annotated_eval", "val"]
InputView = Literal["native", "full_image"]
LabelSource = Literal["building_parts", "churches"]
Q2Field = Literal["frozen_mean", "finetuned_mean"]


@dataclass(frozen=True)
class StrategySpec:
    checkpoint_strategy_id: str
    alignment_from_checkpoint: bool
    q2_reference_field: Q2Field


# The linear probe trains only the head, so Frozen attention is the frozen
# backbone's attention (create_model), exactly as analyze_q2_metrics computes it.
STRATEGIES: dict[Strategy, StrategySpec] = {
    "frozen": StrategySpec("linear_probe", alignment_from_checkpoint=False, q2_reference_field="frozen_mean"),
    "lora": StrategySpec("lora", alignment_from_checkpoint=True, q2_reference_field="finetuned_mean"),
    "full": StrategySpec("full", alignment_from_checkpoint=True, q2_reference_field="finetuned_mean"),
}
STRATEGY_BY_CHECKPOINT_ID: dict[str, Strategy] = {
    spec.checkpoint_strategy_id: strategy for strategy, spec in STRATEGIES.items()
}

LABEL_SOURCES: dict[Split, LabelSource] = {
    "annotated_eval": "building_parts",
    "val": "churches",
}

PREPROCESSORS: dict[InputView, Callable[[FineTunableModel, list[Any]], torch.Tensor]] = {
    "native": FineTunableModel.preprocess_for_classification,
    "full_image": FineTunableModel.preprocess,
}

EXPERIMENT_ID = "fine_tuning_primary_20260327"
MODEL_ORDER = ("dinov2", "dinov3", "mae", "clip", "siglip", "siglip2")
FULL_IMAGE_CHECK_MODELS = frozenset({"dinov2", "clip"})
IOU_PERCENTILE = 90
LAYER = -1
ALPHA = 0.05
MIN_GROUP_SIZE = 2
RELATIVE_TOLERANCE = 1e-4
DELTA_ABS_TOLERANCE = 1e-12

PREDICTIONS_FILENAME = "classification_predictions.json"
ANALYSIS_FILENAME = "accuracy_alignment_analysis.json"
DEFAULT_OUTPUT_DIR = get_experiment_paths(EXPERIMENT_ID).results_dir

ALIGNMENT_SOURCE = (
    "recomputed with analyze_q2_metrics.compute_model_metric_values on whole-image input; "
    "metrics.db not read"
)
CLASSIFICATION_INPUT = (
    "native processor (center crop for dinov2, clip; full-image resize for the others)"
)
FULL_IMAGE_INPUT = "FineTunableModel.preprocess (whole image resized to 224x224, no crop)"
FROZEN_DEFINITION = (
    "accuracy from the linear-probe checkpoint (frozen backbone + trained 4-class head); "
    "alignment from the frozen backbone (create_model)"
)


@dataclass(frozen=True)
class Run:
    run_id: str
    model: str
    strategy: Strategy
    checkpoint_strategy_id: str
    checkpoint_path: str
    best_val_acc: float
    batch_size: int
    attention_method: str


@dataclass(frozen=True)
class Prediction:
    run_id: str
    model: str
    strategy: Strategy
    image_id: str
    split: Split
    input_view: InputView
    label_source: LabelSource
    true_label: str
    predicted_label: str
    correct: bool


@dataclass(frozen=True)
class LabelConflict:
    image_id: str
    building_parts_label: str
    churches_label: str


Alignment = dict[AnalysisMetric, dict[str, float]]


@dataclass(frozen=True)
class PredictionsArtifact:
    experiment_id: str
    split_id: str
    inference_git_commit_sha: str | None
    timestamp: str
    analyzed_layer: int
    iou_percentile: int
    label_conflicts: list[LabelConflict]
    eval_images_without_churches_style: list[str]
    runs: list[Run]
    predictions: list[Prediction]
    alignment: dict[str, Alignment]


@dataclass(frozen=True)
class LabelledImages:
    image_ids: dict[Split, list[str]]
    annotated: AnnotatedSubset
    full: FullDataset
    label_conflicts: list[LabelConflict]
    eval_images_without_churches_style: list[str]


@dataclass(frozen=True)
class ImageRecord:
    image_id: str
    true_label: str
    predicted_label: str
    correct: bool
    metrics: dict[AnalysisMetric, float]


@dataclass(frozen=True)
class MetricComparison:
    metric: AnalysisMetric
    n_correct: int
    n_wrong: int
    correct_median: float | None
    correct_mean: float | None
    wrong_median: float | None
    wrong_mean: float | None
    u_statistic: float | None
    p_value: float | None
    rank_biserial: float | None
    not_applicable_reason: str | None
    holm_p_value: float | None = None
    significant: bool | None = None
    holm_family_size: int | None = None


@dataclass(frozen=True)
class EvalSummary:
    input_view: InputView
    n_images: int
    accuracy: float
    balanced_accuracy: float
    n_correct: int
    n_wrong: int
    mean_iou: float
    comparisons: dict[AnalysisMetric, MetricComparison]
    per_image: list[ImageRecord]


@dataclass(frozen=True)
class ValAccuracyCheck:
    run_id: str
    n_images: int
    n_correct: int
    expected_n_correct: int
    recomputed_accuracy: float
    best_val_acc: float
    passed: bool


@dataclass(frozen=True)
class AlignmentCheck:
    run_id: str
    metric: AnalysisMetric
    q2_field: Q2Field
    recomputed_mean: float
    q2_mean: float
    relative_difference: float
    passed: bool


@dataclass(frozen=True)
class PerImageDeltaCheck:
    run_id: str
    metric: AnalysisMetric
    n_images: int
    n_mismatched: int
    max_abs_difference: float
    passed: bool


def run_sort_key(run: Run) -> tuple[int, int]:
    return MODEL_ORDER.index(run.model), list(STRATEGIES).index(run.strategy)


def load_runs(models: tuple[str, ...]) -> tuple[list[Run], str]:
    """Primary runs for the requested models, plus their shared split ID."""
    results_path = get_experiment_paths(EXPERIMENT_ID).fine_tuning_results_path
    payload = json.loads(results_path.read_text(encoding="utf-8"))

    runs: list[Run] = []
    split_ids: set[str] = set()
    for entry in payload["runs"]:
        model = entry["model_name"]
        if entry["run_scope"] != "primary" or model not in models:
            continue
        checkpoint_strategy_id = entry["strategy_id"]
        expected_name = get_checkpoint_filename(model, checkpoint_strategy_id)
        if Path(entry["checkpoint_path"]).name != expected_name:
            raise ValueError(f"{entry['run_id']}: checkpoint is not {expected_name}")
        split_ids.add(entry["split_id"])
        runs.append(
            Run(
                run_id=entry["run_id"],
                model=model,
                strategy=STRATEGY_BY_CHECKPOINT_ID[checkpoint_strategy_id],
                checkpoint_strategy_id=checkpoint_strategy_id,
                checkpoint_path=entry["checkpoint_path"],
                best_val_acc=float(entry["best_val_acc"]),
                batch_size=int(entry["config"]["batch_size"]),
                attention_method=DEFAULT_METHOD[model].value,
            )
        )

    expected_runs = len(models) * len(STRATEGIES)
    if len(runs) != expected_runs:
        raise ValueError(f"Expected {expected_runs} primary runs, found {len(runs)}")
    if len(split_ids) != 1:
        raise ValueError(f"Runs do not share one split: {sorted(split_ids)}")
    return sorted(runs, key=run_sort_key), split_ids.pop()


def load_split_artifact(split_id: str) -> FineTuningSplitArtifact:
    path = get_experiment_paths(EXPERIMENT_ID).split_artifacts_dir / f"{split_id}.json"
    return FineTuningSplitArtifact(**json.loads(path.read_text(encoding="utf-8")))


def load_labelled_images(split: FineTuningSplitArtifact) -> LabelledImages:
    """Evaluation images with building_parts.json labels, validation images with churches.json labels."""
    annotated = AnnotatedSubset(DATASET_PATH)
    if annotated.image_ids != split.annotated_eval_image_ids:
        raise ValueError("Split annotated_eval_image_ids differ from AnnotatedSubset image IDs")

    # Same construction as fine_tune_models.py, so validation labels match training.
    full = FullDataset(DATASET_PATH, filter_labeled=True)
    full_index = {full.get_metadata(index)["image_id"]: index for index in range(len(full))}

    conflicts: list[LabelConflict] = []
    without_churches_style: list[str] = []
    for sample in annotated:
        image_id = sample["image_id"]
        building_parts_label = sample["style_label"]
        if building_parts_label is None:
            raise ValueError(f"{image_id} has no four-style label in building_parts.json")
        index = full_index.get(image_id)
        churches_label = None if index is None else full.get_metadata(index)["style_label"]
        if churches_label is None:
            without_churches_style.append(image_id)
        elif churches_label != building_parts_label:
            conflicts.append(
                LabelConflict(
                    image_id=image_id,
                    building_parts_label=STYLE_NAMES[building_parts_label],
                    churches_label=STYLE_NAMES[churches_label],
                )
            )

    return LabelledImages(
        image_ids={"annotated_eval": split.annotated_eval_image_ids, "val": split.val_image_ids},
        annotated=annotated,
        full=full,
        label_conflicts=conflicts,
        eval_images_without_churches_style=without_churches_style,
    )


def prediction_passes(model: str) -> list[tuple[Split, InputView]]:
    passes: list[tuple[Split, InputView]] = [("annotated_eval", "native"), ("val", "native")]
    if model in FULL_IMAGE_CHECK_MODELS:
        passes.append(("annotated_eval", "full_image"))
    return passes


def build_loaders(run: Run, images: LabelledImages) -> dict[Split, DataLoader[Any]]:
    """Loaders built from FineTuner's own subset and collate helpers, as in its validation pass."""
    # Only batch_size and seed of this config reach the helpers used here.
    tuner = FineTuner(
        FineTuningConfig(model_name=run.model, experiment_id=EXPERIMENT_ID, batch_size=run.batch_size)
    )
    val_subset = tuner._subset_from_image_ids(
        images.full,
        images.image_ids["val"],
        image_index_map=tuner._build_image_index_map(images.full),
    )
    datasets: dict[Split, Dataset[Any]] = {"annotated_eval": images.annotated, "val": val_subset}
    return {
        split: DataLoader(
            dataset,
            batch_size=tuner.config.batch_size,
            shuffle=False,
            collate_fn=tuner._collate_fn,
            num_workers=0,
        )
        for split, dataset in datasets.items()
    }


def predict_styles(
    model: FineTunableModel,
    loader: DataLoader[Any],
    view: InputView,
    desc: str,
) -> tuple[list[int], list[int]]:
    """Mirror FineTuner's validation pass and return (true, predicted) label indices."""
    preprocess = PREPROCESSORS[view]
    true_labels: list[int] = []
    predicted_labels: list[int] = []
    model.eval()
    with torch.no_grad():
        for batch in tqdm(loader, desc=desc, leave=False):
            logits, _, _ = model(preprocess(model, batch["images"]))
            predicted_labels.extend(logits.argmax(dim=1).tolist())
            true_labels.extend(batch["labels"].tolist())
    return true_labels, predicted_labels


def predict_run(model: FineTunableModel, run: Run, images: LabelledImages) -> list[Prediction]:
    loaders = build_loaders(run, images)
    predictions: list[Prediction] = []
    for split, view in prediction_passes(run.model):
        image_ids = images.image_ids[split]
        true_labels, predicted_labels = predict_styles(
            model, loaders[split], view, desc=f"{run.model}/{run.strategy} {split} {view}"
        )
        if len(true_labels) != len(image_ids):
            raise ValueError(
                f"{run.run_id} {split}: {len(true_labels)} predictions for {len(image_ids)} images"
            )
        predictions.extend(
            Prediction(
                run_id=run.run_id,
                model=run.model,
                strategy=run.strategy,
                image_id=image_id,
                split=split,
                input_view=view,
                label_source=LABEL_SOURCES[split],
                true_label=STYLE_NAMES[true_label],
                predicted_label=STYLE_NAMES[predicted_label],
                correct=true_label == predicted_label,
            )
            for image_id, true_label, predicted_label in zip(
                image_ids, true_labels, predicted_labels, strict=True
            )
        )
    return predictions


def compute_alignment(
    model: BaseVisionModel | FineTunableModel,
    dataset: AnnotatedSubset,
    model_name: str,
) -> Alignment:
    buckets = compute_model_metric_values(model, dataset, model_name, [IOU_PERCENTILE], LAYER)
    return {
        metric: dict(buckets[metric][IOU_PERCENTILE if config.percentile_dependent else None])
        for metric, config in METRIC_CONFIGS.items()
    }


def run_inference(models: tuple[str, ...]) -> PredictionsArtifact:
    runs, split_id = load_runs(models)
    images = load_labelled_images(load_split_artifact(split_id))
    print(
        f"{len(runs)} runs, {len(images.image_ids['annotated_eval'])} evaluation images, "
        f"{len(images.image_ids['val'])} validation images"
    )

    predictions: list[Prediction] = []
    alignment: dict[str, Alignment] = {}
    for model_name in models:
        print(f"\n{model_name}: frozen backbone alignment")
        base_model = create_model(model_name)
        frozen_alignment = compute_alignment(base_model, images.annotated, model_name)
        del base_model
        clear_memory()

        for run in (run for run in runs if run.model == model_name):
            print(f"{model_name}: {run.strategy} ({run.checkpoint_path})")
            finetuned = load_finetuned_model(
                run.model,
                checkpoint_path=project_path(run.checkpoint_path),
                strategy_id=run.checkpoint_strategy_id,
            )
            predictions.extend(predict_run(finetuned, run, images))
            alignment[run.run_id] = (
                compute_alignment(finetuned, images.annotated, model_name)
                if STRATEGIES[run.strategy].alignment_from_checkpoint
                else frozen_alignment
            )
            del finetuned
            clear_memory()

    return PredictionsArtifact(
        experiment_id=EXPERIMENT_ID,
        split_id=split_id,
        inference_git_commit_sha=get_git_commit_sha(),
        timestamp=datetime.now(UTC).isoformat(),
        analyzed_layer=resolve_layer_index(models[0], LAYER),
        iou_percentile=IOU_PERCENTILE,
        label_conflicts=images.label_conflicts,
        eval_images_without_churches_style=images.eval_images_without_churches_style,
        runs=runs,
        predictions=predictions,
        alignment=alignment,
    )


def load_predictions(path: Path) -> PredictionsArtifact:
    raw = json.loads(path.read_text(encoding="utf-8"))
    return PredictionsArtifact(
        experiment_id=raw["experiment_id"],
        split_id=raw["split_id"],
        inference_git_commit_sha=raw["inference_git_commit_sha"],
        timestamp=raw["timestamp"],
        analyzed_layer=raw["analyzed_layer"],
        iou_percentile=raw["iou_percentile"],
        label_conflicts=[LabelConflict(**item) for item in raw["label_conflicts"]],
        eval_images_without_churches_style=raw["eval_images_without_churches_style"],
        runs=[Run(**item) for item in raw["runs"]],
        predictions=[Prediction(**item) for item in raw["predictions"]],
        alignment=raw["alignment"],
    )


def group_stat(values: np.ndarray, statistic: Callable[[np.ndarray], Any]) -> float | None:
    return float(statistic(values)) if values.size else None


def compare_groups(metric: AnalysisMetric, records: list[ImageRecord]) -> MetricComparison:
    """Correct versus wrong images for one metric, before Holm correction."""
    correct = np.array([record.metrics[metric] for record in records if record.correct])
    wrong = np.array([record.metrics[metric] for record in records if not record.correct])
    n_correct, n_wrong = correct.size, wrong.size

    u_statistic: float | None = None
    p_value: float | None = None
    rank_biserial: float | None = None
    not_applicable_reason: str | None = None
    if min(n_correct, n_wrong) < MIN_GROUP_SIZE:
        not_applicable_reason = (
            f"fewer than {MIN_GROUP_SIZE} images in a group "
            f"(correct={n_correct}, wrong={n_wrong})"
        )
    else:
        result = stats.mannwhitneyu(correct, wrong, alternative="two-sided")
        u_statistic = float(result.statistic)
        p_value = float(result.pvalue)
        rank_biserial = 2.0 * u_statistic / (n_correct * n_wrong) - 1.0

    return MetricComparison(
        metric=metric,
        n_correct=n_correct,
        n_wrong=n_wrong,
        correct_median=group_stat(correct, np.median),
        correct_mean=group_stat(correct, np.mean),
        wrong_median=group_stat(wrong, np.median),
        wrong_mean=group_stat(wrong, np.mean),
        u_statistic=u_statistic,
        p_value=p_value,
        rank_biserial=rank_biserial,
        not_applicable_reason=not_applicable_reason,
    )


def summarize_eval(
    predictions: list[Prediction],
    alignment: Alignment,
    view: InputView,
) -> EvalSummary:
    records = [
        ImageRecord(
            image_id=prediction.image_id,
            true_label=prediction.true_label,
            predicted_label=prediction.predicted_label,
            correct=prediction.correct,
            metrics={metric: alignment[metric][prediction.image_id] for metric in METRIC_CONFIGS},
        )
        for prediction in predictions
    ]
    n_correct = sum(record.correct for record in records)
    return EvalSummary(
        input_view=view,
        n_images=len(records),
        accuracy=n_correct / len(records),
        balanced_accuracy=float(
            balanced_accuracy_score(
                [record.true_label for record in records],
                [record.predicted_label for record in records],
            )
        ),
        n_correct=n_correct,
        n_wrong=len(records) - n_correct,
        mean_iou=float(np.mean([record.metrics["iou"] for record in records])),
        comparisons={metric: compare_groups(metric, records) for metric in METRIC_CONFIGS},
        per_image=records,
    )


def apply_holm(comparisons: dict[str, MetricComparison]) -> dict[str, MetricComparison]:
    """Holm across one metric's runs. Runs without an applicable test stay out of the family."""
    tested = {key: c.p_value for key, c in comparisons.items() if c.p_value is not None}
    corrected = multiple_comparison_correction(list(tested.values()), method="holm", alpha=ALPHA)
    updated = dict(comparisons)
    for key, (holm_p_value, significant) in zip(tested, corrected, strict=True):
        updated[key] = replace(
            comparisons[key],
            holm_p_value=holm_p_value,
            significant=significant,
            holm_family_size=len(tested),
        )
    return updated


def holm_family(summaries: dict[str, EvalSummary]) -> dict[str, EvalSummary]:
    corrected = {
        metric: apply_holm({run_id: s.comparisons[metric] for run_id, s in summaries.items()})
        for metric in METRIC_CONFIGS
    }
    return {
        run_id: replace(
            summary,
            comparisons={metric: corrected[metric][run_id] for metric in METRIC_CONFIGS},
        )
        for run_id, summary in summaries.items()
    }


def check_val_accuracy(run: Run, predictions: list[Prediction]) -> ValAccuracyCheck:
    n_images = len(predictions)
    n_correct = sum(prediction.correct for prediction in predictions)
    expected = round(run.best_val_acc * n_images)
    return ValAccuracyCheck(
        run_id=run.run_id,
        n_images=n_images,
        n_correct=n_correct,
        expected_n_correct=expected,
        recomputed_accuracy=n_correct / n_images,
        best_val_acc=run.best_val_acc,
        passed=n_correct == expected,
    )


def q2_metric_rows(q2_rows: list[dict[str, Any]], run: Run, metric: AnalysisMetric) -> list[dict[str, Any]]:
    percentile = IOU_PERCENTILE if METRIC_CONFIGS[metric].percentile_dependent else None
    return [
        row
        for row in q2_rows
        if row["model_name"] == run.model
        and row["metric"] == metric
        and row["percentile"] == percentile
        and row["method"] == run.attention_method
    ]


def q2_reference_mean(q2_rows: list[dict[str, Any]], run: Run, metric: AnalysisMetric) -> float:
    field = STRATEGIES[run.strategy].q2_reference_field
    rows = q2_metric_rows(q2_rows, run, metric)
    # frozen_mean describes the model, so every strategy row carries it.
    if field == "finetuned_mean":
        rows = [row for row in rows if row["strategy_id"] == run.checkpoint_strategy_id]
    values = {float(row[field]) for row in rows}
    if len(values) != 1:
        raise ValueError(f"{run.run_id} {metric}: Q2 {field} is not unique: {sorted(values)}")
    return values.pop()


def check_alignment(
    run: Run,
    alignment: Alignment,
    q2_rows: list[dict[str, Any]],
) -> list[AlignmentCheck]:
    checks: list[AlignmentCheck] = []
    for metric in METRIC_CONFIGS:
        recomputed = float(np.mean(list(alignment[metric].values())))
        reference = q2_reference_mean(q2_rows, run, metric)
        checks.append(
            AlignmentCheck(
                run_id=run.run_id,
                metric=metric,
                q2_field=STRATEGIES[run.strategy].q2_reference_field,
                recomputed_mean=recomputed,
                q2_mean=reference,
                relative_difference=abs(recomputed - reference) / abs(reference),
                passed=math.isclose(recomputed, reference, rel_tol=RELATIVE_TOLERANCE, abs_tol=0.0),
            )
        )
    return checks


def check_per_image_deltas(
    run: Run,
    alignment: Alignment,
    frozen_alignment: Alignment,
    q2_rows: list[dict[str, Any]],
) -> list[PerImageDeltaCheck]:
    """Catch values attached to the wrong image ID, which mean-level checks cannot see."""
    checks: list[PerImageDeltaCheck] = []
    for metric in METRIC_CONFIGS:
        rows = [
            row
            for row in q2_metric_rows(q2_rows, run, metric)
            if row["strategy_id"] == run.checkpoint_strategy_id
        ]
        if len(rows) != 1:
            raise ValueError(f"{run.run_id} {metric}: expected one Q2 row, found {len(rows)}")
        reference: dict[str, float] = rows[0]["per_image_deltas"]
        if set(reference) != set(alignment[metric]):
            raise ValueError(f"{run.run_id} {metric}: image IDs differ from Q2 per_image_deltas")
        pairs = [
            (alignment[metric][image_id] - frozen_alignment[metric][image_id], delta)
            for image_id, delta in reference.items()
        ]
        mismatched = sum(
            not math.isclose(recomputed, delta, rel_tol=RELATIVE_TOLERANCE, abs_tol=DELTA_ABS_TOLERANCE)
            for recomputed, delta in pairs
        )
        checks.append(
            PerImageDeltaCheck(
                run_id=run.run_id,
                metric=metric,
                n_images=len(reference),
                n_mismatched=mismatched,
                max_abs_difference=max(abs(recomputed - delta) for recomputed, delta in pairs),
                passed=mismatched == 0,
            )
        )
    return checks


def validate_predictions(
    artifact: PredictionsArtifact,
    predictions_by_pass: dict[tuple[str, Split, InputView], list[Prediction]],
) -> None:
    """Reject a predictions file whose passes do not cover each split image exactly once."""
    split = load_split_artifact(artifact.split_id)
    expected_ids: dict[Split, list[str]] = {
        "annotated_eval": split.annotated_eval_image_ids,
        "val": split.val_image_ids,
    }
    expected_passes = {
        (run.run_id, split_name, view)
        for run in artifact.runs
        for split_name, view in prediction_passes(run.model)
    }
    if set(predictions_by_pass) != expected_passes:
        raise ValueError(
            f"Prediction passes differ from the expected passes: "
            f"missing {sorted(expected_passes - set(predictions_by_pass))}, "
            f"unexpected {sorted(set(predictions_by_pass) - expected_passes)}"
        )

    true_labels: dict[tuple[Split, str], str] = {}
    for (run_id, split_name, view), predictions in predictions_by_pass.items():
        if [prediction.image_id for prediction in predictions] != expected_ids[split_name]:
            raise ValueError(f"{run_id} {split_name}/{view}: image IDs differ from split {artifact.split_id}")
        for prediction in predictions:
            true_label = true_labels.setdefault((split_name, prediction.image_id), prediction.true_label)
            if (
                prediction.correct != (prediction.true_label == prediction.predicted_label)
                or prediction.label_source != LABEL_SOURCES[split_name]
                or prediction.true_label != true_label
            ):
                raise ValueError(f"{run_id} {split_name}/{view} {prediction.image_id}: inconsistent prediction record")

    eval_ids = set(expected_ids["annotated_eval"])
    for run in artifact.runs:
        for metric in METRIC_CONFIGS:
            if set(artifact.alignment[run.run_id][metric]) != eval_ids:
                raise ValueError(f"{run.run_id} {metric}: alignment image IDs differ from the evaluation images")


def load_q2_rows(analyzed_layer: int) -> list[dict[str, Any]]:
    path = get_experiment_paths(EXPERIMENT_ID).q2_metrics_path
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload["analyzed_layer"] != analyzed_layer:
        raise ValueError(f"{path} is layer {payload['analyzed_layer']}, not {analyzed_layer}")
    rows: list[dict[str, Any]] = payload["rows"]
    return rows


def significance_changes(
    native: dict[str, EvalSummary],
    alternate: dict[str, EvalSummary],
) -> list[dict[str, Any]]:
    changes: list[dict[str, Any]] = []
    for run_id, native_summary in native.items():
        alternate_summary = alternate[run_id]
        for metric in METRIC_CONFIGS:
            before = native_summary.comparisons[metric]
            after = alternate_summary.comparisons[metric]
            if before.significant != after.significant:
                changes.append(
                    {
                        "run_id": run_id,
                        "metric": metric,
                        "alternate_input_view": alternate_summary.input_view,
                        "native_holm_p_value": before.holm_p_value,
                        "alternate_holm_p_value": after.holm_p_value,
                        "native_significant": before.significant,
                        "alternate_significant": after.significant,
                    }
                )
    return changes


def analyze(artifact: PredictionsArtifact) -> dict[str, Any]:
    predictions_by_pass: dict[tuple[str, Split, InputView], list[Prediction]] = {}
    for prediction in artifact.predictions:
        key = (prediction.run_id, prediction.split, prediction.input_view)
        predictions_by_pass.setdefault(key, []).append(prediction)
    validate_predictions(artifact, predictions_by_pass)

    eval_summaries = {
        (run_id, view): summarize_eval(predictions, artifact.alignment[run_id], view)
        for (run_id, split, view), predictions in predictions_by_pass.items()
        if split == "annotated_eval"
    }
    run_ids = [run.run_id for run in artifact.runs]
    native = holm_family({run_id: eval_summaries[(run_id, "native")] for run_id in run_ids})
    alternate = holm_family(
        {
            run_id: eval_summaries.get((run_id, "full_image"), eval_summaries[(run_id, "native")])
            for run_id in run_ids
        }
    )
    full_image_run_ids = [run_id for run_id in run_ids if (run_id, "full_image") in eval_summaries]

    val_checks = [
        check_val_accuracy(run, predictions_by_pass[(run.run_id, "val", "native")])
        for run in artifact.runs
    ]
    q2_rows = load_q2_rows(artifact.analyzed_layer)
    alignment_checks = [
        check
        for run in artifact.runs
        for check in check_alignment(run, artifact.alignment[run.run_id], q2_rows)
    ]
    frozen_run_ids = {run.model: run.run_id for run in artifact.runs if run.strategy == "frozen"}
    delta_checks = [
        check
        for run in artifact.runs
        if STRATEGIES[run.strategy].alignment_from_checkpoint
        for check in check_per_image_deltas(
            run,
            artifact.alignment[run.run_id],
            artifact.alignment[frozen_run_ids[run.model]],
            q2_rows,
        )
    ]
    failures = (
        [{"check": "val_accuracy", **asdict(check)} for check in val_checks if not check.passed]
        + [{"check": "alignment", **asdict(check)} for check in alignment_checks if not check.passed]
        + [{"check": "per_image_delta", **asdict(check)} for check in delta_checks if not check.passed]
    )

    by_run_id = {run.run_id: run for run in artifact.runs}
    return {
        "experiment_id": artifact.experiment_id,
        "split_id": artifact.split_id,
        "analysis_git_commit_sha": get_git_commit_sha(),
        "inference_git_commit_sha": artifact.inference_git_commit_sha,
        "timestamp": datetime.now(UTC).isoformat(),
        "inference_timestamp": artifact.timestamp,
        "analyzed_layer": artifact.analyzed_layer,
        "iou_percentile": artifact.iou_percentile,
        "evaluation_image_count": native[run_ids[0]].n_images,
        "validation_image_count": val_checks[0].n_images,
        "metrics": [asdict(config) for config in METRIC_CONFIGS.values()],
        "label_sources": {
            "annotated_eval": "building_parts.json (AnnotatedSubset, styles[0])",
            "val": "churches.json (FullDataset with filter_labeled=True, as in training)",
        },
        "label_conflicts": [asdict(conflict) for conflict in artifact.label_conflicts],
        "eval_images_without_churches_style": artifact.eval_images_without_churches_style,
        "alignment_source": ALIGNMENT_SOURCE,
        "classification_input": CLASSIFICATION_INPUT,
        "frozen_definition": FROZEN_DEFINITION,
        "test": {
            "name": "Mann-Whitney U",
            "alternative": "two-sided",
            "groups": "correct versus wrong images, native input view, per run and metric",
            "correction": "holm across runs, per metric; runs without an applicable test excluded",
            "alpha": ALPHA,
            "min_group_size": MIN_GROUP_SIZE,
            "rank_biserial": "2U/(n_correct*n_wrong) - 1; positive means correct images have higher values",
        },
        "runs": [
            {
                **asdict(run),
                "annotated_eval": asdict(native[run.run_id]),
                "validation": {
                    "n_images": check.n_images,
                    "n_correct": check.n_correct,
                    "accuracy": check.recomputed_accuracy,
                },
            }
            for run, check in zip(artifact.runs, val_checks, strict=True)
        ],
        "full_image_robustness": {
            "input_view": FULL_IMAGE_INPUT,
            "models": sorted(FULL_IMAGE_CHECK_MODELS),
            "alternate_family": (
                "native runs of the other models plus the full_image runs, "
                "which replace their native counterparts"
            ),
            "runs": [
                {
                    "run_id": run_id,
                    "model": by_run_id[run_id].model,
                    "strategy": by_run_id[run_id].strategy,
                    "native_accuracy": native[run_id].accuracy,
                    "full_image_accuracy": alternate[run_id].accuracy,
                    "accuracy_difference": alternate[run_id].accuracy - native[run_id].accuracy,
                    "annotated_eval": asdict(alternate[run_id]),
                }
                for run_id in full_image_run_ids
            ],
            "significance_changes": significance_changes(native, alternate),
        },
        "verification": {
            "relative_tolerance": RELATIVE_TOLERANCE,
            "val_accuracy": [asdict(check) for check in val_checks],
            "alignment": [asdict(check) for check in alignment_checks],
            "per_image_delta": [asdict(check) for check in delta_checks],
            "failures": failures,
            "passed": not failures,
        },
    }


def format_optional(value: float | None, spec: str) -> str:
    return "n/a" if value is None else format(value, spec)


def print_report(analysis: dict[str, Any]) -> None:
    print(f"\nLabel conflicts (building_parts vs churches): {len(analysis['label_conflicts'])}")
    for conflict in analysis["label_conflicts"]:
        print(
            f"  {conflict['image_id']:<20} {conflict['building_parts_label']:<12} "
            f"{conflict['churches_label']}"
        )

    print(f"\nIoU@{analysis['iou_percentile']}, layer {analysis['analyzed_layer']}, native input")
    print(
        f"{'Run':<16} {'Acc':>6} {'BalAcc':>7} {'Corr':>5} {'Wrong':>6} {'MeanIoU':>8} "
        f"{'MedCorr':>8} {'MedWrong':>9} {'p':>9} {'pHolm':>9} {'r_rb':>6} {'ValAcc':>7}"
    )
    for run in analysis["runs"]:
        summary = run["annotated_eval"]
        iou = summary["comparisons"]["iou"]
        print(
            f"{run['model'] + '/' + run['strategy']:<16} {summary['accuracy']:>6.3f} "
            f"{summary['balanced_accuracy']:>7.3f} {summary['n_correct']:>5} {summary['n_wrong']:>6} "
            f"{summary['mean_iou']:>8.4f} {format_optional(iou['correct_median'], '.4f'):>8} "
            f"{format_optional(iou['wrong_median'], '.4f'):>9} "
            f"{format_optional(iou['p_value'], '.2e'):>9} "
            f"{format_optional(iou['holm_p_value'], '.2e'):>9} "
            f"{format_optional(iou['rank_biserial'], '+.2f'):>6} "
            f"{run['validation']['accuracy']:>7.4f}"
        )

    robustness = analysis["full_image_robustness"]
    if robustness["runs"]:
        print("\nFull-image robustness (139 images, IoU@90)")
        print(f"{'Run':<16} {'AccNative':>9} {'AccFull':>8} {'Diff':>7} {'pHolm(alt)':>11}")
        for entry in robustness["runs"]:
            iou = entry["annotated_eval"]["comparisons"]["iou"]
            print(
                f"{entry['model'] + '/' + entry['strategy']:<16} {entry['native_accuracy']:>9.3f} "
                f"{entry['full_image_accuracy']:>8.3f} {entry['accuracy_difference']:>+7.3f} "
                f"{format_optional(iou['holm_p_value'], '.2e'):>11}"
            )
        print(f"Significance changes vs native family: {len(robustness['significance_changes'])}")
        for change in robustness["significance_changes"]:
            print(
                f"  {change['run_id']} {change['metric']}: "
                f"{change['native_significant']} -> {change['alternate_significant']}"
            )

    verification = analysis["verification"]
    print("\nCheck (a): validation accuracy vs best_val_acc")
    print(f"{'Run':<50} {'n_correct':>9} {'expected':>8} {'n':>4} {'match':>6}")
    for check in verification["val_accuracy"]:
        print(
            f"{check['run_id']:<50} {check['n_correct']:>9} {check['expected_n_correct']:>8} "
            f"{check['n_images']:>4} {str(check['passed']):>6}"
        )

    print(f"\nCheck (b): mean alignment vs q2_metrics_analysis.json (rel tol {RELATIVE_TOLERANCE:g})")
    print(f"{'Run':<50} {'Metric':<9} {'Field':<15} {'Recomputed':>11} {'Q2':>11} {'RelDiff':>9} {'match':>6}")
    for check in verification["alignment"]:
        print(
            f"{check['run_id']:<50} {check['metric']:<9} {check['q2_field']:<15} "
            f"{check['recomputed_mean']:>11.6f} {check['q2_mean']:>11.6f} "
            f"{check['relative_difference']:>9.2e} {str(check['passed']):>6}"
        )

    print("\nCheck (c): per-image (run - Frozen) vs Q2 per_image_deltas")
    print(f"{'Run':<50} {'Metric':<9} {'n':>4} {'mismatched':>10} {'MaxAbsDiff':>11} {'match':>6}")
    for check in verification["per_image_delta"]:
        print(
            f"{check['run_id']:<50} {check['metric']:<9} {check['n_images']:>4} "
            f"{check['n_mismatched']:>10} {check['max_abs_difference']:>11.2e} {str(check['passed']):>6}"
        )

    if verification["failures"]:
        print(f"\nFAILED CHECKS: {len(verification['failures'])}")
        for failure in verification["failures"]:
            detail = ", ".join(f"{key}={value}" for key, value in failure.items() if key != "check")
            print(f"  [{failure['check']}] {detail}")
    else:
        print("\nAll checks passed.")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Relate classification correctness to attention alignment (R1-Q3)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--models",
        nargs="+",
        choices=MODEL_ORDER,
        default=list(MODEL_ORDER),
        help="Models to run inference for. Default: all six.",
    )
    source.add_argument(
        "--from-predictions",
        action="store_true",
        help=f"Skip inference and re-analyze {PREDICTIONS_FILENAME} in the output directory.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help=f"Directory for {PREDICTIONS_FILENAME} and {ANALYSIS_FILENAME}.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    predictions_path = args.output_dir / PREDICTIONS_FILENAME
    analysis_path = args.output_dir / ANALYSIS_FILENAME

    if not args.from_predictions:
        models = tuple(model for model in MODEL_ORDER if model in args.models)
        save_json(predictions_path, asdict(run_inference(models)))
        print(f"\nPredictions saved to: {predictions_path}")

    analysis = analyze(load_predictions(predictions_path))
    save_json(analysis_path, analysis)
    print_report(analysis)
    print(f"\nAnalysis saved to: {analysis_path}")

    if analysis["verification"]["failures"]:
        sys.exit(1)


if __name__ == "__main__":
    main()

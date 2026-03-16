#!/usr/bin/env python3
"""Analyze Q2 attention shift (ΔIoU) across fine-tuning strategies.

This script compares frozen and fine-tuned model attention alignment for
(model, strategy) pairs:
- linear_probe
- lora
- full

Outputs:
- per-image deltas
- per-(model,strategy) statistics (CI, effect size, significance)
- cross-strategy paired comparisons within each model (Holm corrected)
- lightweight forgetting proxy via IoU retention ratio
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from itertools import combinations
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np

# Enable MPS fallback for unsupported ops
os.environ.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")

import torch
from tqdm import tqdm

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from ssl_attention.attention.cls_attention import (  # noqa: E402
    HeadFusion,
    attention_to_heatmap,
    extract_cls_attention,
    extract_mean_attention,
)
from ssl_attention.config import (  # noqa: E402
    DATASET_PATH,
    DEFAULT_METHOD,
    FINETUNE_MODELS,
    FINETUNE_STRATEGIES,
    MODELS,
    AttentionMethod,
)
from ssl_attention.data.wikichurches import AnnotatedSubset  # noqa: E402
from ssl_attention.evaluation.fine_tuning import (  # noqa: E402
    RESULTS_PATH,
    FineTunableModel,
    get_checkpoint_candidates,
    load_finetuned_model,
)
from ssl_attention.metrics.iou import compute_image_iou  # noqa: E402
from ssl_attention.metrics.statistics import (  # noqa: E402
    bootstrap_ci,
    cohens_d,
    multiple_comparison_correction,
    paired_comparison,
)
from ssl_attention.models import create_model  # noqa: E402
from ssl_attention.utils.device import clear_memory  # noqa: E402

if TYPE_CHECKING:
    from ssl_attention.models.base import BaseVisionModel


@dataclass
class PerImageResult:
    image_id: str
    frozen_iou: float
    finetuned_iou: float
    delta_iou: float


@dataclass
class StrategyDeltaIoU:
    model_name: str
    strategy_id: str
    percentile: int
    method: str
    frozen_mean_iou: float
    finetuned_mean_iou: float
    mean_delta_iou: float
    std_delta_iou: float
    delta_ci_lower: float
    delta_ci_upper: float
    cohens_d: float
    p_value: float
    corrected_p_value: float | None
    significant: bool
    test_name: str
    iou_retention_ratio: float | None
    per_image: list[PerImageResult] = field(default_factory=list)
    num_images: int = 0


@dataclass
class StrategyPairComparison:
    model_name: str
    percentile: int
    strategy_a: str
    strategy_b: str
    mean_delta_difference: float
    cohens_d: float
    p_value: float
    corrected_p_value: float | None
    significant: bool
    test_name: str


@dataclass
class AnalysisResults:
    percentiles: list[int]
    models: dict[str, dict[str, dict[int, StrategyDeltaIoU]]]
    strategy_comparisons: dict[str, dict[int, list[StrategyPairComparison]]]
    timestamp: str = ""


def parse_strategy_from_checkpoint_name(checkpoint_path: Path, model_name: str) -> str:
    """Infer strategy from checkpoint filename."""
    stem = checkpoint_path.stem
    for strategy in (s.value for s in FINETUNE_STRATEGIES):
        if stem == f"{model_name}_{strategy}_finetuned":
            return strategy
    # Legacy naming is ambiguous (full vs linear probe). Default to full.
    if stem == f"{model_name}_finetuned":
        return "full"
    return "unknown"


def discover_strategy_checkpoints(
    model_names: list[str],
    strategy_ids: list[str],
) -> dict[str, dict[str, Path]]:
    """Discover checkpoint paths for requested model/strategy pairs."""
    discovered: dict[str, dict[str, Path]] = {}

    for model_name in model_names:
        per_model: dict[str, Path] = {}
        for strategy_id in strategy_ids:
            candidates = get_checkpoint_candidates(model_name, strategy_id=strategy_id)
            checkpoint = next((path for path in candidates if path.exists()), None)
            if checkpoint is not None:
                per_model[strategy_id] = checkpoint
        if per_model:
            discovered[model_name] = per_model

    return discovered


def extract_attention_heatmap(
    model: BaseVisionModel | FineTunableModel,
    pixel_values: torch.Tensor,
    model_name: str,
    layer: int = -1,
) -> torch.Tensor:
    """Extract attention heatmap from frozen or fine-tuned model."""
    config = MODELS[model_name]
    method = DEFAULT_METHOD[model_name]

    output = model.extract_attention(pixel_values) if isinstance(model, FineTunableModel) else model(pixel_values)
    attention_weights = output.attention_weights

    if method == AttentionMethod.CLS:
        attn = extract_cls_attention(
            attention_weights,
            layer=layer,
            num_registers=config.num_registers,
            fusion=HeadFusion.MEAN,
        )
    elif method == AttentionMethod.MEAN:
        attn = extract_mean_attention(
            attention_weights,
            layer=layer,
            fusion=HeadFusion.MEAN,
        )
    elif method == AttentionMethod.GRADCAM:
        raise NotImplementedError(f"Grad-CAM not supported in this script. Model: {model_name}")
    else:
        raise ValueError(f"Unknown attention method: {method}")

    heatmap = attention_to_heatmap(attn, image_size=224, normalize=True)
    return heatmap.squeeze(0)


def compute_model_ious(
    model: BaseVisionModel | FineTunableModel,
    dataset: AnnotatedSubset,
    model_name: str,
    percentiles: list[int],
    layer: int = -1,
) -> dict[int, list[tuple[str, float]]]:
    """Compute IoU for all images across percentiles."""
    results: dict[int, list[tuple[str, float]]] = {p: [] for p in percentiles}

    processor = model.processor
    device = model.device
    dtype = model.dtype

    for sample in tqdm(dataset, desc=f"Processing {model_name}", leave=False):
        image_id = sample["image_id"]
        image = sample["image"]
        annotation = sample["annotation"]

        processed = processor(images=[image], return_tensors="pt")
        pixel_values = processed["pixel_values"].to(device=device, dtype=dtype)

        with torch.no_grad():
            heatmap = extract_attention_heatmap(model, pixel_values, model_name, layer=layer)

        for percentile in percentiles:
            iou_result = compute_image_iou(
                attention=heatmap,
                annotation=annotation,
                image_id=image_id,
                percentile=percentile,
            )
            results[percentile].append((image_id, iou_result.iou))

    return results


def summarize_delta(
    *,
    model_name: str,
    strategy_id: str,
    percentile: int,
    method: str,
    frozen_values: dict[str, float],
    finetuned_values: dict[str, float],
) -> StrategyDeltaIoU:
    """Build summary stats for one model/strategy/percentile."""
    per_image_results: list[PerImageResult] = []
    frozen_arr: list[float] = []
    finetuned_arr: list[float] = []

    for image_id, frozen_iou in frozen_values.items():
        finetuned_iou = finetuned_values[image_id]
        delta = finetuned_iou - frozen_iou
        per_image_results.append(
            PerImageResult(
                image_id=image_id,
                frozen_iou=frozen_iou,
                finetuned_iou=finetuned_iou,
                delta_iou=delta,
            )
        )
        frozen_arr.append(frozen_iou)
        finetuned_arr.append(finetuned_iou)

    frozen_np = np.array(frozen_arr)
    finetuned_np = np.array(finetuned_arr)
    delta_np = finetuned_np - frozen_np

    _, ci_lower, ci_upper = bootstrap_ci(delta_np, statistic="mean")
    comparison = paired_comparison(
        finetuned_np,
        frozen_np,
        model_a_name="finetuned",
        model_b_name="frozen",
        test="auto",
    )

    frozen_mean = float(np.mean(frozen_np))
    finetuned_mean = float(np.mean(finetuned_np))
    iou_retention_ratio = (
        finetuned_mean / frozen_mean if frozen_mean > 0 else None
    )

    return StrategyDeltaIoU(
        model_name=model_name,
        strategy_id=strategy_id,
        percentile=percentile,
        method=method,
        frozen_mean_iou=frozen_mean,
        finetuned_mean_iou=finetuned_mean,
        mean_delta_iou=float(np.mean(delta_np)),
        std_delta_iou=float(np.std(delta_np, ddof=1)),
        delta_ci_lower=ci_lower,
        delta_ci_upper=ci_upper,
        cohens_d=cohens_d(finetuned_np, frozen_np, paired=True),
        p_value=comparison.p_value,
        corrected_p_value=None,
        significant=comparison.significant,
        test_name=comparison.test_name,
        iou_retention_ratio=iou_retention_ratio,
        per_image=per_image_results,
        num_images=len(per_image_results),
    )


def analyze_model_strategy(
    *,
    model_name: str,
    strategy_id: str,
    checkpoint_path: Path,
    dataset: AnnotatedSubset,
    frozen_ious: dict[int, list[tuple[str, float]]],
    percentiles: list[int],
    layer: int,
) -> dict[int, StrategyDeltaIoU]:
    """Analyze one (model, strategy) checkpoint against frozen baseline."""
    method = DEFAULT_METHOD[model_name].value

    print(f"  Loading fine-tuned {model_name} [{strategy_id}] from {checkpoint_path}...")
    finetuned_model = load_finetuned_model(
        model_name,
        checkpoint_path=checkpoint_path,
        strategy_id=strategy_id,
    )

    print("  Computing fine-tuned IoUs...")
    finetuned_ious = compute_model_ious(
        finetuned_model,
        dataset,
        model_name,
        percentiles,
        layer,
    )

    del finetuned_model
    clear_memory()

    results: dict[int, StrategyDeltaIoU] = {}
    for percentile in percentiles:
        frozen_dict = dict(frozen_ious[percentile])
        finetuned_dict = dict(finetuned_ious[percentile])
        results[percentile] = summarize_delta(
            model_name=model_name,
            strategy_id=strategy_id,
            percentile=percentile,
            method=method,
            frozen_values=frozen_dict,
            finetuned_values=finetuned_dict,
        )

    return results


def apply_holm_correction_by_percentile(
    model_results: dict[str, dict[str, dict[int, StrategyDeltaIoU]]],
    percentile: int,
    alpha: float = 0.05,
) -> None:
    """Apply Holm correction across all (model,strategy) p-values for a percentile."""
    keys: list[tuple[str, str]] = []
    p_values: list[float] = []

    for model_name, strategy_results in model_results.items():
        for strategy_id, percentile_results in strategy_results.items():
            if percentile in percentile_results:
                keys.append((model_name, strategy_id))
                p_values.append(percentile_results[percentile].p_value)

    if not p_values:
        return

    corrected = multiple_comparison_correction(p_values, method="holm", alpha=alpha)
    for (model_name, strategy_id), (corrected_p, significant) in zip(keys, corrected, strict=True):
        result = model_results[model_name][strategy_id][percentile]
        result.corrected_p_value = corrected_p
        result.significant = significant


def compare_strategies_within_model(
    *,
    model_name: str,
    strategy_results: dict[str, dict[int, StrategyDeltaIoU]],
    percentiles: list[int],
    alpha: float = 0.05,
) -> dict[int, list[StrategyPairComparison]]:
    """Compute paired strategy comparisons using per-image deltas."""
    output: dict[int, list[StrategyPairComparison]] = {}

    for percentile in percentiles:
        rows: list[StrategyPairComparison] = []
        available_strategies = [
            strategy for strategy in strategy_results if percentile in strategy_results[strategy]
        ]
        for strategy_a, strategy_b in combinations(sorted(available_strategies), 2):
            a_result = strategy_results[strategy_a][percentile]
            b_result = strategy_results[strategy_b][percentile]

            deltas_a = np.array([row.delta_iou for row in a_result.per_image])
            deltas_b = np.array([row.delta_iou for row in b_result.per_image])

            comparison = paired_comparison(
                deltas_a,
                deltas_b,
                model_a_name=strategy_a,
                model_b_name=strategy_b,
                test="auto",
            )

            rows.append(
                StrategyPairComparison(
                    model_name=model_name,
                    percentile=percentile,
                    strategy_a=strategy_a,
                    strategy_b=strategy_b,
                    mean_delta_difference=float(np.mean(deltas_a - deltas_b)),
                    cohens_d=cohens_d(deltas_a, deltas_b, paired=True),
                    p_value=comparison.p_value,
                    corrected_p_value=None,
                    significant=comparison.significant,
                    test_name=comparison.test_name,
                )
            )

        if rows:
            corrected = multiple_comparison_correction(
                [row.p_value for row in rows],
                method="holm",
                alpha=alpha,
            )
            for row, (corrected_p, significant) in zip(rows, corrected, strict=True):
                row.corrected_p_value = corrected_p
                row.significant = significant

        output[percentile] = rows

    return output


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze Δ IoU between frozen and fine-tuned models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        help="Specific models to analyze. Default: all fine-tunable models with checkpoints.",
    )
    parser.add_argument(
        "--strategies",
        nargs="+",
        default=[s.value for s in FINETUNE_STRATEGIES],
        choices=[s.value for s in FINETUNE_STRATEGIES],
        help="Fine-tuning strategies to analyze.",
    )
    parser.add_argument(
        "--percentile",
        type=int,
        default=None,
        help="Single percentile to analyze. Default: [90, 80, 70, 60, 50].",
    )
    parser.add_argument(
        "--layer",
        type=int,
        default=-1,
        help="Layer to extract attention from. Default: -1 (last layer).",
    )
    parser.add_argument(
        "--include-resnet",
        action="store_true",
        help="Include ResNet-50 (unsupported in this script).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Default: outputs/results/q2_delta_iou_analysis.json",
    )

    return parser.parse_args()


def print_summary_table(
    results: dict[str, dict[str, dict[int, StrategyDeltaIoU]]],
    percentile: int,
) -> None:
    print(f"\n{'=' * 112}")
    print(f"Q2 Δ IoU Results (Percentile {percentile})")
    print(f"{'=' * 112}")

    print(
        f"{'Model':<10} {'Strategy':<13} {'Frozen':>8} {'Fine':>8} {'Δ IoU':>8} "
        f"{'95% CI':>18} {'Retain':>8} {'d':>7} {'p(Holm)':>10} {'Sig':>5}"
    )
    print("-" * 112)

    rows: list[StrategyDeltaIoU] = []
    for _model_name, strategy_results in results.items():
        for _strategy_id, percentile_results in strategy_results.items():
            if percentile in percentile_results:
                rows.append(percentile_results[percentile])

    rows.sort(key=lambda r: r.mean_delta_iou, reverse=True)

    for row in rows:
        ci = f"[{row.delta_ci_lower:+.3f}, {row.delta_ci_upper:+.3f}]"
        retention = f"{row.iou_retention_ratio:.3f}" if row.iou_retention_ratio is not None else "n/a"
        p_disp = row.corrected_p_value if row.corrected_p_value is not None else row.p_value
        sig = "***" if row.significant else ""
        print(
            f"{row.model_name:<10} {row.strategy_id:<13} "
            f"{row.frozen_mean_iou:>8.3f} {row.finetuned_mean_iou:>8.3f} {row.mean_delta_iou:>+8.3f} "
            f"{ci:>18} {retention:>8} {row.cohens_d:>+7.2f} {p_disp:>10.4f} {sig:>5}"
        )


def save_results(results: AnalysisResults, output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    payload: dict[str, Any] = {
        "percentiles": results.percentiles,
        "timestamp": results.timestamp,
        "models": {},
        "strategy_comparisons": {},
    }

    for model_name, strategy_results in results.models.items():
        payload["models"][model_name] = {}
        for strategy_id, percentile_results in strategy_results.items():
            payload["models"][model_name][strategy_id] = {}
            for percentile, row in percentile_results.items():
                row_data = asdict(row)
                row_data["per_image_deltas"] = {
                    item["image_id"]: item["delta_iou"] for item in row_data.pop("per_image")
                }
                payload["models"][model_name][strategy_id][str(percentile)] = row_data

    for model_name, percentile_rows in results.strategy_comparisons.items():
        payload["strategy_comparisons"][model_name] = {}
        for percentile, rows in percentile_rows.items():
            payload["strategy_comparisons"][model_name][str(percentile)] = [asdict(row) for row in rows]

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main() -> None:
    from datetime import datetime

    args = parse_args()
    percentiles = [args.percentile] if args.percentile else [90, 80, 70, 60, 50]

    if args.models:
        requested_models = [m for m in args.models if m in FINETUNE_MODELS]
        invalid = [m for m in args.models if m not in FINETUNE_MODELS]
        if invalid:
            print(f"Skipping non-fine-tunable models: {invalid}")
    else:
        requested_models = sorted(FINETUNE_MODELS)

    if args.include_resnet:
        print("ResNet-50 uses Grad-CAM and is not supported by this script. Skipping.")

    discovered = discover_strategy_checkpoints(requested_models, args.strategies)
    if not discovered:
        print("No strategy checkpoints found. Exiting.")
        return

    print(f"Models to analyze: {sorted(discovered.keys())}")
    print(f"Strategies: {args.strategies}")
    print(f"Percentiles: {percentiles}")
    print(f"Layer: {args.layer}")

    print("\nLoading annotated dataset...")
    dataset = AnnotatedSubset(DATASET_PATH)
    print(f"  {len(dataset)} images with expert bounding boxes")

    all_results: dict[str, dict[str, dict[int, StrategyDeltaIoU]]] = {}

    for model_name, strategy_paths in discovered.items():
        print(f"\n{'=' * 64}")
        print(f"Analyzing {model_name.upper()}")
        print(f"{'=' * 64}")

        print(f"  Loading frozen {model_name}...")
        frozen_model = create_model(model_name)

        print("  Computing frozen IoUs...")
        frozen_ious = compute_model_ious(
            frozen_model,
            dataset,
            model_name,
            percentiles,
            args.layer,
        )
        del frozen_model
        clear_memory()

        strategy_results: dict[str, dict[int, StrategyDeltaIoU]] = {}
        for strategy_id in sorted(strategy_paths):
            checkpoint_path = strategy_paths[strategy_id]
            parsed_strategy = parse_strategy_from_checkpoint_name(checkpoint_path, model_name)
            effective_strategy = strategy_id if parsed_strategy == "unknown" else parsed_strategy
            try:
                strategy_results[strategy_id] = analyze_model_strategy(
                    model_name=model_name,
                    strategy_id=effective_strategy,
                    checkpoint_path=checkpoint_path,
                    dataset=dataset,
                    frozen_ious=frozen_ious,
                    percentiles=percentiles,
                    layer=args.layer,
                )
            except Exception as exc:
                print(f"  Error analyzing {model_name}/{strategy_id}: {exc}")

        if strategy_results:
            all_results[model_name] = strategy_results

    if not all_results:
        print("\nNo model/strategy pairs were successfully analyzed.")
        return

    print("\nApplying Holm correction across model/strategy pairs...")
    for percentile in percentiles:
        apply_holm_correction_by_percentile(all_results, percentile)

    strategy_comparisons: dict[str, dict[int, list[StrategyPairComparison]]] = {}
    print("Computing within-model cross-strategy comparisons...")
    for model_name, strategy_results in all_results.items():
        strategy_comparisons[model_name] = compare_strategies_within_model(
            model_name=model_name,
            strategy_results=strategy_results,
            percentiles=percentiles,
        )

    for percentile in percentiles:
        print_summary_table(all_results, percentile)

    results = AnalysisResults(
        percentiles=percentiles,
        models=all_results,
        strategy_comparisons=strategy_comparisons,
        timestamp=datetime.now().isoformat(),
    )

    output_path = args.output or (RESULTS_PATH / "q2_delta_iou_analysis.json")
    save_results(results, output_path)

    print("\nDone!")


if __name__ == "__main__":
    main()

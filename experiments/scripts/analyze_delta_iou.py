#!/usr/bin/env python3
"""Analyze Δ IoU between frozen and fine-tuned model attention alignment.

This script compares attention-annotation IoU before and after fine-tuning
to measure whether task-specific training shifts attention toward expert-
annotated architectural features.

Key metrics:
- Δ IoU = fine-tuned IoU - frozen IoU (per image)
- Paired statistical tests (t-test or Wilcoxon) with Holm correction
- Bootstrap confidence intervals and Cohen's d effect sizes

Usage:
    # All available models (models with checkpoints in outputs/checkpoints/)
    uv run python experiments/scripts/analyze_delta_iou.py

    # Specific models
    uv run python experiments/scripts/analyze_delta_iou.py --models dinov2 siglip

    # Custom percentile threshold
    uv run python experiments/scripts/analyze_delta_iou.py --percentile 80

    # Include ResNet-50 (uses Grad-CAM instead of attention)
    uv run python experiments/scripts/analyze_delta_iou.py --include-resnet

Output:
    - outputs/results/delta_iou_analysis.json: Full analysis results
    - Console: Summary table with Δ IoU, effect sizes, and significance
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

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
    MODELS,
    AttentionMethod,
)
from ssl_attention.data.wikichurches import AnnotatedSubset  # noqa: E402
from ssl_attention.evaluation.fine_tuning import (  # noqa: E402
    CHECKPOINTS_PATH,
    RESULTS_PATH,
    FineTunableModel,
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
    from ssl_attention.data.annotations import ImageAnnotation
    from ssl_attention.models.base import BaseVisionModel


# =============================================================================
# Data Classes
# =============================================================================


@dataclass
class PerImageResult:
    """IoU result for a single image, both variants."""

    image_id: str
    frozen_iou: float
    finetuned_iou: float
    delta_iou: float  # finetuned - frozen


@dataclass
class ModelDeltaIoU:
    """Delta IoU results for a single model.

    Attributes:
        model_name: Name of the model (e.g., 'dinov2').
        percentile: Percentile threshold used for IoU computation.
        method: Attention extraction method used.
        frozen_mean_iou: Mean IoU for frozen model.
        finetuned_mean_iou: Mean IoU for fine-tuned model.
        mean_delta_iou: Mean Δ IoU (fine-tuned - frozen).
        std_delta_iou: Standard deviation of Δ IoU.
        delta_ci_lower: 95% CI lower bound for mean Δ IoU.
        delta_ci_upper: 95% CI upper bound for mean Δ IoU.
        cohens_d: Effect size (positive = improvement).
        p_value: Raw p-value from paired test.
        corrected_p_value: P-value after multiple comparison correction.
        significant: Whether change is significant after correction.
        test_name: Statistical test used.
        per_image: List of per-image results.
        num_images: Number of images analyzed.
    """

    model_name: str
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
    per_image: list[PerImageResult] = field(default_factory=list)
    num_images: int = 0


@dataclass
class AnalysisResults:
    """Full analysis results across all models and percentiles.

    Attributes:
        percentiles: Percentile thresholds analyzed.
        models: Results per model (keyed by model name).
        timestamp: ISO timestamp of analysis.
    """

    percentiles: list[int]
    models: dict[str, dict[int, ModelDeltaIoU]]  # model_name -> percentile -> results
    timestamp: str = ""


# =============================================================================
# Utility Functions
# =============================================================================


def get_available_checkpoints() -> list[str]:
    """Find models with fine-tuned checkpoints.

    Returns:
        List of model names that have checkpoints in outputs/checkpoints/.
    """
    available = []
    for model_name in MODELS.keys():
        checkpoint_path = CHECKPOINTS_PATH / f"{model_name}_finetuned.pt"
        if checkpoint_path.exists():
            available.append(model_name)
    return available


def extract_attention_heatmap(
    model: BaseVisionModel | FineTunableModel,
    pixel_values: torch.Tensor,
    model_name: str,
    layer: int = -1,
) -> torch.Tensor:
    """Extract attention heatmap from any model variant.

    Handles both frozen BaseVisionModel and FineTunableModel uniformly.

    Args:
        model: Model instance (frozen or fine-tuned).
        pixel_values: Preprocessed image tensor.
        model_name: Model name for config lookup.
        layer: Layer to extract attention from.

    Returns:
        Attention heatmap tensor of shape (H, W).
    """
    config = MODELS[model_name]
    method = DEFAULT_METHOD[model_name]

    # Get attention weights
    if isinstance(model, FineTunableModel):
        output = model.extract_attention(pixel_values)
    else:
        output = model(pixel_values)

    attention_weights = output.attention_weights

    # Extract attention based on method
    if method == AttentionMethod.CLS:
        attn = extract_cls_attention(
            attention_weights,
            layer=layer,
            num_registers=config.num_registers,
            fusion=HeadFusion.MEAN,
        )
    elif method == AttentionMethod.MEAN:
        # SigLIP uses mean attention
        attn = extract_mean_attention(
            attention_weights,
            layer=layer,
            fusion=HeadFusion.MEAN,
        )
    elif method == AttentionMethod.GRADCAM:
        # ResNet uses Grad-CAM, need separate handling
        # For now, skip ResNet in main analysis (requires different pipeline)
        raise NotImplementedError(
            f"Grad-CAM not supported in this script. Model: {model_name}"
        )
    else:
        raise ValueError(f"Unknown attention method: {method}")

    # Convert to heatmap
    heatmap = attention_to_heatmap(
        attn,
        image_size=224,
        normalize=True,
    )

    return heatmap.squeeze(0)  # Remove batch dim


def compute_model_ious(
    model: BaseVisionModel | FineTunableModel,
    dataset: AnnotatedSubset,
    model_name: str,
    percentiles: list[int],
    layer: int = -1,
) -> dict[int, list[tuple[str, float]]]:
    """Compute IoU for all images at multiple percentiles.

    Args:
        model: Model instance.
        dataset: AnnotatedSubset with expert bounding boxes.
        model_name: Model name for config lookup.
        percentiles: List of percentile thresholds.
        layer: Layer to extract attention from.

    Returns:
        Dict mapping percentile -> list of (image_id, iou) tuples.
    """
    results: dict[int, list[tuple[str, float]]] = {p: [] for p in percentiles}

    # Use frozen model's processor for consistent preprocessing
    if isinstance(model, FineTunableModel):
        processor = model.processor
        device = model.device
        dtype = model.dtype
    else:
        processor = model.processor
        device = model.device
        dtype = model.dtype

    for sample in tqdm(dataset, desc=f"Processing {model_name}", leave=False):
        image_id = sample["image_id"]
        image = sample["image"]
        annotation: ImageAnnotation = sample["annotation"]

        # Preprocess image
        processed = processor(images=[image], return_tensors="pt")
        pixel_values = processed["pixel_values"].to(device=device, dtype=dtype)

        # Extract attention heatmap
        with torch.no_grad():
            heatmap = extract_attention_heatmap(
                model, pixel_values, model_name, layer=layer
            )

        # Compute IoU at each percentile
        for percentile in percentiles:
            iou_result = compute_image_iou(
                attention=heatmap,
                annotation=annotation,
                image_id=image_id,
                percentile=percentile,
            )
            results[percentile].append((image_id, iou_result.iou))

    return results


def analyze_single_model(
    model_name: str,
    dataset: AnnotatedSubset,
    percentiles: list[int],
    layer: int = -1,
) -> dict[int, ModelDeltaIoU]:
    """Analyze Δ IoU for a single model.

    Args:
        model_name: Name of the model to analyze.
        dataset: AnnotatedSubset with expert bounding boxes.
        percentiles: List of percentile thresholds.
        layer: Layer to extract attention from.

    Returns:
        Dict mapping percentile -> ModelDeltaIoU results.
    """
    method = DEFAULT_METHOD[model_name].value

    # Load frozen model
    print(f"  Loading frozen {model_name}...")
    frozen_model = create_model(model_name)

    # Compute frozen IoUs
    print(f"  Computing frozen IoUs...")
    frozen_ious = compute_model_ious(
        frozen_model, dataset, model_name, percentiles, layer
    )

    # Free frozen model memory
    del frozen_model
    clear_memory()

    # Load fine-tuned model
    print(f"  Loading fine-tuned {model_name}...")
    finetuned_model = load_finetuned_model(model_name)

    # Compute fine-tuned IoUs
    print(f"  Computing fine-tuned IoUs...")
    finetuned_ious = compute_model_ious(
        finetuned_model, dataset, model_name, percentiles, layer
    )

    # Free fine-tuned model memory
    del finetuned_model
    clear_memory()

    # Analyze delta at each percentile
    results: dict[int, ModelDeltaIoU] = {}

    for percentile in percentiles:
        # Build per-image results
        frozen_dict = dict(frozen_ious[percentile])
        finetuned_dict = dict(finetuned_ious[percentile])

        per_image_results: list[PerImageResult] = []
        frozen_values: list[float] = []
        finetuned_values: list[float] = []

        for image_id in frozen_dict:
            frozen_iou = frozen_dict[image_id]
            finetuned_iou = finetuned_dict[image_id]
            delta = finetuned_iou - frozen_iou

            per_image_results.append(
                PerImageResult(
                    image_id=image_id,
                    frozen_iou=frozen_iou,
                    finetuned_iou=finetuned_iou,
                    delta_iou=delta,
                )
            )
            frozen_values.append(frozen_iou)
            finetuned_values.append(finetuned_iou)

        # Convert to numpy for statistics
        import numpy as np

        frozen_arr = np.array(frozen_values)
        finetuned_arr = np.array(finetuned_values)
        delta_arr = finetuned_arr - frozen_arr

        # Compute statistics
        frozen_mean = float(np.mean(frozen_arr))
        finetuned_mean = float(np.mean(finetuned_arr))
        delta_mean = float(np.mean(delta_arr))
        delta_std = float(np.std(delta_arr, ddof=1))

        # Bootstrap CI for delta mean
        _, ci_lower, ci_upper = bootstrap_ci(delta_arr, statistic="mean")

        # Effect size
        effect_size = cohens_d(finetuned_arr, frozen_arr, paired=True)

        # Paired test
        comparison = paired_comparison(
            finetuned_arr,
            frozen_arr,
            model_a_name="finetuned",
            model_b_name="frozen",
            test="auto",
        )

        results[percentile] = ModelDeltaIoU(
            model_name=model_name,
            percentile=percentile,
            method=method,
            frozen_mean_iou=frozen_mean,
            finetuned_mean_iou=finetuned_mean,
            mean_delta_iou=delta_mean,
            std_delta_iou=delta_std,
            delta_ci_lower=ci_lower,
            delta_ci_upper=ci_upper,
            cohens_d=effect_size,
            p_value=comparison.p_value,
            corrected_p_value=None,  # Set later after all models
            significant=comparison.significant,
            test_name=comparison.test_name,
            per_image=per_image_results,
            num_images=len(per_image_results),
        )

    return results


def apply_holm_correction(
    all_results: dict[str, dict[int, ModelDeltaIoU]],
    percentile: int,
    alpha: float = 0.05,
) -> None:
    """Apply Holm correction across models for a given percentile.

    Modifies results in place to set corrected_p_value and significant.

    Args:
        all_results: Results dict mapping model_name -> percentile -> ModelDeltaIoU.
        percentile: Which percentile to correct.
        alpha: Significance threshold.
    """
    # Collect p-values for this percentile
    model_names = list(all_results.keys())
    p_values = [all_results[name][percentile].p_value for name in model_names]

    # Apply Holm correction
    corrected = multiple_comparison_correction(p_values, method="holm", alpha=alpha)

    # Update results
    for i, model_name in enumerate(model_names):
        corrected_p, significant = corrected[i]
        all_results[model_name][percentile].corrected_p_value = corrected_p
        all_results[model_name][percentile].significant = significant


# =============================================================================
# Main
# =============================================================================


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Analyze Δ IoU between frozen and fine-tuned models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )

    parser.add_argument(
        "--models",
        nargs="+",
        type=str,
        help="Specific models to analyze. Default: all models with checkpoints.",
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
        help="Include ResNet-50 (requires separate Grad-CAM pipeline, not yet implemented).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Output JSON path. Default: outputs/results/delta_iou_analysis.json",
    )

    return parser.parse_args()


def print_summary_table(
    results: dict[str, dict[int, ModelDeltaIoU]],
    percentile: int,
) -> None:
    """Print formatted summary table for a percentile."""
    print(f"\n{'='*80}")
    print(f"Δ IoU Analysis Results (Percentile {percentile})")
    print(f"{'='*80}")

    # Header
    print(
        f"{'Model':<10} {'Frozen':>8} {'Fine-tuned':>10} {'Δ IoU':>8} "
        f"{'95% CI':>16} {'Cohen d':>8} {'p-value':>10} {'Sig':>5}"
    )
    print("-" * 80)

    # Sort by delta IoU descending
    sorted_models = sorted(
        results.keys(),
        key=lambda m: results[m][percentile].mean_delta_iou,
        reverse=True,
    )

    for model_name in sorted_models:
        r = results[model_name][percentile]
        ci_str = f"[{r.delta_ci_lower:+.3f}, {r.delta_ci_upper:+.3f}]"
        p_str = (
            f"{r.corrected_p_value:.4f}"
            if r.corrected_p_value is not None
            else f"{r.p_value:.4f}"
        )
        sig_str = "***" if r.significant else ""

        print(
            f"{model_name:<10} {r.frozen_mean_iou:>8.3f} {r.finetuned_mean_iou:>10.3f} "
            f"{r.mean_delta_iou:>+8.3f} {ci_str:>16} {r.cohens_d:>+8.2f} "
            f"{p_str:>10} {sig_str:>5}"
        )

    print("-" * 80)
    print("*** = significant after Holm correction (α=0.05)")


def save_results(
    results: AnalysisResults,
    output_path: Path,
) -> None:
    """Save analysis results to JSON.

    Args:
        results: AnalysisResults to save.
        output_path: Path to save JSON file.
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    data: dict[str, Any] = {
        "percentiles": results.percentiles,
        "timestamp": results.timestamp,
        "models": {},
    }

    for model_name, percentile_results in results.models.items():
        data["models"][model_name] = {}
        for percentile, model_result in percentile_results.items():
            # Convert dataclass to dict, excluding per_image for compactness
            result_dict = asdict(model_result)
            # Simplify per_image to just deltas
            result_dict["per_image_deltas"] = {
                r["image_id"]: r["delta_iou"] for r in result_dict.pop("per_image")
            }
            data["models"][model_name][str(percentile)] = result_dict

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"\nResults saved to: {output_path}")


def main() -> None:
    """Main entry point."""
    from datetime import datetime

    args = parse_args()

    # Determine percentiles
    percentiles = [args.percentile] if args.percentile else [90, 80, 70, 60, 50]

    # Find available checkpoints
    available = get_available_checkpoints()

    if not available:
        print("No fine-tuned checkpoints found in outputs/checkpoints/")
        print("Run experiments/scripts/fine_tune_models.py first to generate checkpoints.")
        print("\nTo run this script without checkpoints (dry run):")
        print("  - The script will skip all models and exit gracefully.")
        return

    # Determine models to analyze
    if args.models:
        # Filter to requested models that have checkpoints
        models_to_analyze = [m for m in args.models if m in available]
        skipped = [m for m in args.models if m not in available]
        if skipped:
            print(f"Skipping models without checkpoints: {skipped}")
    else:
        models_to_analyze = available

    # Exclude ResNet-50 unless explicitly requested
    if not args.include_resnet and "resnet50" in models_to_analyze:
        print("Excluding ResNet-50 (uses Grad-CAM, requires separate pipeline)")
        models_to_analyze.remove("resnet50")

    if not models_to_analyze:
        print("No models to analyze. Exiting.")
        return

    print(f"Models to analyze: {models_to_analyze}")
    print(f"Percentiles: {percentiles}")
    print(f"Layer: {args.layer}")

    # Load dataset
    print("\nLoading annotated dataset...")
    dataset = AnnotatedSubset(DATASET_PATH)
    print(f"  {len(dataset)} images with expert bounding boxes")

    # Analyze each model
    all_results: dict[str, dict[int, ModelDeltaIoU]] = {}

    for model_name in models_to_analyze:
        print(f"\n{'='*60}")
        print(f"Analyzing {model_name.upper()}")
        print(f"{'='*60}")

        try:
            model_results = analyze_single_model(
                model_name=model_name,
                dataset=dataset,
                percentiles=percentiles,
                layer=args.layer,
            )
            all_results[model_name] = model_results
        except Exception as e:
            print(f"  Error analyzing {model_name}: {e}")
            continue

    if not all_results:
        print("\nNo models were successfully analyzed.")
        return

    # Apply Holm correction across models for each percentile
    print("\nApplying Holm correction for multiple comparisons...")
    for percentile in percentiles:
        apply_holm_correction(all_results, percentile)

    # Print summary tables
    for percentile in percentiles:
        print_summary_table(all_results, percentile)

    # Create results object
    analysis_results = AnalysisResults(
        percentiles=percentiles,
        models=all_results,
        timestamp=datetime.now().isoformat(),
    )

    # Save results
    output_path = args.output or (RESULTS_PATH / "delta_iou_analysis.json")
    save_results(analysis_results, output_path)

    print("\nDone!")


if __name__ == "__main__":
    main()

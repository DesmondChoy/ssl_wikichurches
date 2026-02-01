"""Generate pre-rendered heatmap PNG images from cached attention.

Creates overlay images combining original photos with attention heatmaps
for fast serving without runtime model inference.

Directory structure:
    heatmaps/{model}/layer{N}/{method}/{variant}/{image_id}.png

Where:
    - model: dinov2, dinov3, mae, clip, siglip, resnet50
    - method: cls, rollout, mean, gradcam
    - variant: heatmap, overlay, overlay_bbox

Usage:
    python -m app.precompute.generate_heatmap_images --colormap viridis
    python -m app.precompute.generate_heatmap_images --models dinov2 --alpha 0.5
    python -m app.precompute.generate_heatmap_images --models dinov2 --methods cls rollout
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from PIL import Image as PILImage
from tqdm import tqdm

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from ssl_attention.cache import AttentionCache
from ssl_attention.config import (
    CACHE_PATH,
    DATASET_PATH,
    IMAGES_PATH,
    MODELS,
    AttentionMethod,
    DEFAULT_METHOD,
    MODEL_METHODS,
)
from ssl_attention.data import AnnotatedSubset
from ssl_attention.visualization import (
    create_attention_overlay,
    draw_bboxes,
    render_heatmap,
)


def generate_heatmaps_for_model(
    model_name: str,
    dataset: AnnotatedSubset,
    attention_cache: AttentionCache,
    output_dir: Path,
    colormap: str = "viridis",
    alpha: float = 0.5,
    layers: list[int] | None = None,
    methods: list[AttentionMethod] | None = None,
    skip_existing: bool = True,
) -> dict[str, int]:
    """Generate heatmap images for a single model.

    Creates multiple versions per method:
    - Pure heatmap (no overlay)
    - Overlay without bboxes
    - Overlay with bboxes

    Directory structure: {model}/layer{N}/{method}/{variant}/{image_id}.png

    Args:
        model_name: Name of model.
        dataset: Annotated dataset.
        attention_cache: AttentionCache with pre-computed attention.
        output_dir: Directory to save PNG files.
        colormap: Matplotlib colormap name.
        alpha: Overlay transparency.
        layers: Specific layers to process. None = all.
        methods: Specific methods to process. None = all available for model.
        skip_existing: Skip if PNG already exists.

    Returns:
        Dict with statistics.
    """
    stats = {"processed": 0, "skipped": 0, "errors": 0}

    model_config = MODELS[model_name]
    num_layers = model_config.num_layers
    layers_to_process = layers if layers else list(range(num_layers))

    # Determine which methods to process (filter by model compatibility)
    available_methods = MODEL_METHODS.get(model_name, [DEFAULT_METHOD[model_name]])
    if methods:
        methods_to_process = [m for m in methods if m in available_methods]
    else:
        methods_to_process = available_methods

    if not methods_to_process:
        print(f"No compatible methods for {model_name}")
        return stats

    print(f"\nProcessing {model_name} ({len(layers_to_process)} layers, "
          f"{len(methods_to_process)} methods: {[m.value for m in methods_to_process]})")

    # Create output directories for each layer/method combo
    model_dir = output_dir / model_name
    for layer in layers_to_process:
        for method in methods_to_process:
            method_dir = model_dir / f"layer{layer}" / method.value
            (method_dir / "heatmap").mkdir(parents=True, exist_ok=True)
            (method_dir / "overlay").mkdir(parents=True, exist_ok=True)
            (method_dir / "overlay_bbox").mkdir(parents=True, exist_ok=True)

    # Process each image
    for sample in tqdm(dataset, desc=f"{model_name}"):
        image_id = sample["image_id"]
        annotation = sample["annotation"]

        # Load original image
        image_path = IMAGES_PATH / image_id
        if not image_path.exists():
            stats["errors"] += 1
            continue

        try:
            original_img = PILImage.open(image_path).convert("RGB")
            # Resize to standard size for consistency
            original_img = original_img.resize((224, 224), PILImage.Resampling.BILINEAR)
        except Exception as e:
            print(f"\nError loading {image_id}: {e}")
            stats["errors"] += 1
            continue

        for layer in layers_to_process:
            layer_key = f"layer{layer}"

            for method in methods_to_process:
                method_dir = model_dir / layer_key / method.value

                # Define output paths
                heatmap_path = method_dir / "heatmap" / f"{image_id}.png"
                overlay_path = method_dir / "overlay" / f"{image_id}.png"
                bbox_path = method_dir / "overlay_bbox" / f"{image_id}.png"

                # Skip if all exist
                if skip_existing and heatmap_path.exists() and overlay_path.exists() and bbox_path.exists():
                    stats["skipped"] += 3
                    continue

                # Load attention from cache (using method as variant)
                try:
                    attention = attention_cache.load(
                        model_name, layer_key, image_id, variant=method.value
                    )
                except KeyError:
                    # Attention not cached for this combination
                    stats["skipped"] += 3
                    continue

                try:
                    # Generate heatmap image
                    heatmap_img = render_heatmap(attention, colormap=colormap, output_size=(224, 224))

                    # Save pure heatmap
                    if not skip_existing or not heatmap_path.exists():
                        heatmap_img.save(heatmap_path)
                        stats["processed"] += 1
                    else:
                        stats["skipped"] += 1

                    # Save overlay without bboxes
                    if not skip_existing or not overlay_path.exists():
                        overlay = create_attention_overlay(
                            original_img,
                            heatmap_img,
                            annotation=None,
                            alpha=alpha,
                            show_bboxes=False,
                        )
                        overlay.save(overlay_path)
                        stats["processed"] += 1
                    else:
                        stats["skipped"] += 1

                    # Save overlay with bboxes
                    if not skip_existing or not bbox_path.exists():
                        overlay_bbox = create_attention_overlay(
                            original_img,
                            heatmap_img,
                            annotation=annotation,
                            alpha=alpha,
                            show_bboxes=True,
                        )
                        overlay_bbox.save(bbox_path)
                        stats["processed"] += 1
                    else:
                        stats["skipped"] += 1

                except Exception as e:
                    print(f"\nError generating heatmaps for {image_id} {layer_key}/{method.value}: {e}")
                    stats["errors"] += 1

    return stats


def generate_original_with_bboxes(
    dataset: AnnotatedSubset,
    output_dir: Path,
    skip_existing: bool = True,
) -> dict[str, int]:
    """Generate original images with bounding boxes only (no attention).

    Args:
        dataset: Annotated dataset.
        output_dir: Directory to save PNG files.
        skip_existing: Skip if PNG already exists.

    Returns:
        Dict with statistics.
    """
    stats = {"processed": 0, "skipped": 0, "errors": 0}

    originals_dir = output_dir / "originals"
    (originals_dir / "clean").mkdir(parents=True, exist_ok=True)
    (originals_dir / "bbox").mkdir(parents=True, exist_ok=True)

    print("\nProcessing original images with bboxes")

    for sample in tqdm(dataset, desc="originals"):
        image_id = sample["image_id"]
        annotation = sample["annotation"]

        # Load original image
        image_path = IMAGES_PATH / image_id
        if not image_path.exists():
            stats["errors"] += 1
            continue

        try:
            original_img = PILImage.open(image_path).convert("RGB")
            original_img = original_img.resize((224, 224), PILImage.Resampling.BILINEAR)
        except Exception as e:
            print(f"\nError loading {image_id}: {e}")
            stats["errors"] += 1
            continue

        # Save clean original
        clean_path = originals_dir / "clean" / f"{image_id}.png"
        if not skip_existing or not clean_path.exists():
            original_img.save(clean_path)
            stats["processed"] += 1
        else:
            stats["skipped"] += 1

        # Save with bboxes
        bbox_path = originals_dir / "bbox" / f"{image_id}.png"
        if not skip_existing or not bbox_path.exists():
            img_with_bbox = draw_bboxes(original_img, annotation.bboxes)
            img_with_bbox.save(bbox_path)
            stats["processed"] += 1
        else:
            stats["skipped"] += 1

    return stats


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate heatmap PNG images")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Models to process (or 'all')",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=None,
        help="Specific layers to process (default: all 12)",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=["all"],
        choices=["all", "cls", "rollout", "mean", "gradcam"],
        help="Attention methods to process (default: all available per model)",
    )
    parser.add_argument(
        "--attention-cache",
        type=Path,
        default=CACHE_PATH / "attention_viz.h5",
        help="Path to attention cache HDF5",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=CACHE_PATH / "heatmaps",
        help="Directory for output PNGs",
    )
    parser.add_argument(
        "--colormap",
        type=str,
        default="viridis",
        choices=["viridis", "plasma", "inferno", "magma", "hot", "jet"],
        help="Matplotlib colormap",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Overlay transparency (0-1)",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Don't skip existing files",
    )
    args = parser.parse_args()

    # Parse methods
    if "all" in args.methods:
        methods_to_use = None  # Will use all available per model
    else:
        methods_to_use = [AttentionMethod(m) for m in args.methods]

    # Validate
    if not args.attention_cache.exists():
        print(f"Error: Attention cache not found: {args.attention_cache}")
        print("Run generate_attention_cache.py first.")
        return 1

    # Determine models to process
    if "all" in args.models:
        models_to_process = list(MODELS.keys())
    else:
        models_to_process = [m for m in args.models if m in MODELS]

    if not models_to_process:
        print("No valid models specified")
        return 1

    # Setup
    dataset = AnnotatedSubset(DATASET_PATH)
    print(f"Dataset: {len(dataset)} annotated images")

    attention_cache = AttentionCache(args.attention_cache)
    print(f"Attention cache: {args.attention_cache}")

    output_dir = args.output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output dir: {output_dir}")

    # Generate originals with bboxes first
    stats = generate_original_with_bboxes(
        dataset, output_dir, skip_existing=not args.no_skip
    )
    print(f"Originals complete: {stats}")

    # Process each model
    total_stats = {"processed": 0, "skipped": 0, "errors": 0}

    for model_name in models_to_process:
        stats = generate_heatmaps_for_model(
            model_name=model_name,
            dataset=dataset,
            attention_cache=attention_cache,
            output_dir=output_dir,
            colormap=args.colormap,
            alpha=args.alpha,
            layers=args.layers,
            methods=methods_to_use,
            skip_existing=not args.no_skip,
        )

        for key in total_stats:
            total_stats[key] += stats[key]

        print(f"{model_name} complete: {stats}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total processed: {total_stats['processed']}")
    print(f"Total skipped: {total_stats['skipped']}")
    print(f"Total errors: {total_stats['errors']}")

    return 0 if total_stats["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

"""Generate attention cache for all models and layers.

Extracts CLS-to-patch attention for all 139 annotated images across
5 models and 12 layers, storing results in HDF5 format.

Usage:
    python -m app.precompute.generate_attention_cache --models all
    python -m app.precompute.generate_attention_cache --models dinov2 clip
    python -m app.precompute.generate_attention_cache --models dinov2 --layers 10 11
"""

from __future__ import annotations

import argparse
import gc
import sys
from pathlib import Path

import torch
from tqdm import tqdm

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from ssl_attention.attention import extract_cls_attention, attention_to_heatmap
from ssl_attention.cache import AttentionCache
from ssl_attention.config import CACHE_PATH, DATASET_PATH, MODELS
from ssl_attention.data import AnnotatedSubset
from ssl_attention.models import create_model
from ssl_attention.utils import get_device


def generate_attention_for_model(
    model_name: str,
    dataset: AnnotatedSubset,
    cache: AttentionCache,
    layers: list[int] | None = None,
    device: torch.device | str = "cpu",
    skip_existing: bool = True,
) -> dict[str, int]:
    """Generate attention maps for a single model.

    Args:
        model_name: Name of model (e.g., "dinov2").
        dataset: Annotated dataset.
        cache: AttentionCache instance.
        layers: Specific layers to process. None = all layers.
        device: Compute device.
        skip_existing: Skip if already cached.

    Returns:
        Dict with statistics: {"processed": N, "skipped": M, "errors": E}
    """
    stats = {"processed": 0, "skipped": 0, "errors": 0}

    model_config = MODELS[model_name]
    num_layers = model_config.num_layers
    layers_to_process = layers if layers else list(range(num_layers))

    print(f"\n{'='*60}")
    print(f"Processing {model_name} ({len(layers_to_process)} layers)")
    print(f"{'='*60}")

    # Load model
    print(f"Loading {model_name}...")
    model = create_model(model_name)
    model.to(device)
    model.eval()

    try:
        # Process each image
        for sample in tqdm(dataset, desc=f"{model_name}"):
            image_id = sample["image_id"]
            image = sample["image"]

            try:
                # Check if all layers already cached for this image
                if skip_existing:
                    all_cached = all(
                        cache.exists(model_name, f"layer{layer}", image_id)
                        for layer in layers_to_process
                    )
                    if all_cached:
                        stats["skipped"] += len(layers_to_process)
                        continue

                # Run inference once
                with torch.no_grad():
                    preprocessed = model.preprocess([image]).to(device)
                    output = model.forward(preprocessed)

                # Extract attention for each layer
                for layer in layers_to_process:
                    layer_key = f"layer{layer}"

                    if skip_existing and cache.exists(model_name, layer_key, image_id):
                        stats["skipped"] += 1
                        continue

                    # Extract CLS attention for this layer
                    cls_attn = extract_cls_attention(
                        output.attention_weights,
                        layer=layer,
                        num_registers=model.num_registers,
                    )

                    # Convert to 2D heatmap
                    heatmap = attention_to_heatmap(
                        cls_attn,
                        image_size=224,
                        patch_size=model.patch_size,
                    )

                    # Store in cache
                    cache.store(model_name, layer_key, image_id, heatmap.squeeze(0))
                    stats["processed"] += 1

            except Exception as e:
                print(f"\nError processing {image_id}: {e}")
                stats["errors"] += 1

    finally:
        # Free GPU memory
        del model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        elif hasattr(torch.mps, "empty_cache"):
            torch.mps.empty_cache()

    return stats


def main():
    parser = argparse.ArgumentParser(description="Generate attention cache")
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
        "--cache-path",
        type=Path,
        default=CACHE_PATH / "attention_viz.h5",
        help="Path to attention cache HDF5 file",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Don't skip existing cached items",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device (auto-detected if not specified)",
    )
    args = parser.parse_args()

    # Determine models to process
    if "all" in args.models:
        models_to_process = list(MODELS.keys())
    else:
        models_to_process = [m for m in args.models if m in MODELS]
        invalid = [m for m in args.models if m not in MODELS]
        if invalid:
            print(f"Warning: Unknown models ignored: {invalid}")
            print(f"Available: {list(MODELS.keys())}")

    if not models_to_process:
        print("No valid models specified")
        return 1

    # Setup
    device = args.device or get_device()
    print(f"Device: {device}")

    dataset = AnnotatedSubset(DATASET_PATH)
    print(f"Dataset: {len(dataset)} annotated images")

    cache = AttentionCache(args.cache_path)
    print(f"Cache: {args.cache_path}")

    # Process models one at a time to conserve memory
    total_stats = {"processed": 0, "skipped": 0, "errors": 0}

    for model_name in models_to_process:
        stats = generate_attention_for_model(
            model_name=model_name,
            dataset=dataset,
            cache=cache,
            layers=args.layers,
            device=device,
            skip_existing=not args.no_skip,
        )

        for key in total_stats:
            total_stats[key] += stats[key]

        print(f"\n{model_name} complete: {stats}")

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total processed: {total_stats['processed']}")
    print(f"Total skipped: {total_stats['skipped']}")
    print(f"Total errors: {total_stats['errors']}")

    return 0 if total_stats["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

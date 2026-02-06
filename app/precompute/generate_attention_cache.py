"""Generate attention cache for all models and layers.

Extracts attention maps for all 139 annotated images across
multiple models, layers, and attention methods. Stores results in HDF5 format.

Supported methods per model type:
- ViTs with CLS (DINOv2, DINOv3, MAE, CLIP): cls, rollout
- ViTs without CLS (SigLIP): mean
- CNNs (ResNet50): gradcam

Usage (frozen / pretrained models):
    python -m app.precompute.generate_attention_cache --models all
    python -m app.precompute.generate_attention_cache --models dinov2 clip
    python -m app.precompute.generate_attention_cache --models dinov2 --layers 10 11
    python -m app.precompute.generate_attention_cache --models dinov2 --methods cls rollout

Usage (fine-tuned models):
    python -m app.precompute.generate_attention_cache --finetuned --models all
    python -m app.precompute.generate_attention_cache --finetuned --models dinov2
    python -m app.precompute.generate_attention_cache --finetuned --checkpoint-dir path/to/dir
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

from ssl_attention.attention import (
    attention_to_heatmap,
    extract_cls_attention,
    extract_cls_rollout,
    extract_mean_attention,
)
from ssl_attention.cache import AttentionCache
from ssl_attention.config import (
    CACHE_PATH,
    DATASET_PATH,
    DEFAULT_METHOD,
    MODEL_METHODS,
    MODELS,
    AttentionMethod,
)
from ssl_attention.data import AnnotatedSubset
from ssl_attention.evaluation.fine_tuning import (
    CHECKPOINTS_PATH,
    load_finetuned_model,
)
from ssl_attention.models import create_model
from ssl_attention.utils import get_device

# Models that support fine-tuning (all ViTs; ResNet-50 uses a different pipeline)
FINETUNE_MODELS = {"dinov2", "dinov3", "mae", "clip", "siglip"}


def discover_checkpoints(
    checkpoint_dir: Path,
    model_names: list[str] | None = None,
) -> dict[str, Path]:
    """Discover available fine-tuned checkpoints.

    Args:
        checkpoint_dir: Directory containing checkpoint files.
        model_names: Specific models to look for. None = all fine-tunable models.

    Returns:
        Dict mapping model name to checkpoint path for models that have checkpoints.
    """
    candidates = model_names if model_names else sorted(FINETUNE_MODELS)
    found: dict[str, Path] = {}

    for name in candidates:
        if name not in FINETUNE_MODELS:
            print(f"Warning: {name} is not fine-tunable (skipped)")
            continue
        path = checkpoint_dir / f"{name}_finetuned.pt"
        if path.exists():
            found[name] = path
        else:
            print(f"Warning: No checkpoint for {name} at {path}")

    return found


def generate_attention_for_model(
    model_name: str,
    dataset: AnnotatedSubset,
    cache: AttentionCache,
    layers: list[int] | None = None,
    methods: list[AttentionMethod] | None = None,
    device: torch.device | str = "cpu",
    skip_existing: bool = True,
    finetuned: bool = False,
    checkpoint_path: Path | None = None,
) -> dict[str, int]:
    """Generate attention maps for a single model.

    Args:
        model_name: Name of model (e.g., "dinov2").
        dataset: Annotated dataset.
        cache: AttentionCache instance.
        layers: Specific layers to process. None = all layers.
        methods: Specific methods to process. None = all available for model.
        device: Compute device.
        skip_existing: Skip if already cached.
        finetuned: If True, load fine-tuned checkpoint instead of pretrained.
        checkpoint_path: Path to fine-tuned checkpoint (required if finetuned=True).

    Returns:
        Dict with statistics: {"processed": N, "skipped": M, "errors": E}
    """
    stats = {"processed": 0, "skipped": 0, "errors": 0}

    model_config = MODELS[model_name]
    num_layers = model_config.num_layers
    layers_to_process = layers if layers else list(range(num_layers))

    # Cache key: "dinov2" for frozen, "dinov2_finetuned" for fine-tuned
    cache_model_key = f"{model_name}_finetuned" if finetuned else model_name

    # Determine which methods to process (filter by model compatibility)
    available_methods = MODEL_METHODS.get(model_name, [DEFAULT_METHOD[model_name]])
    if methods:
        methods_to_process = [m for m in methods if m in available_methods]
    else:
        methods_to_process = available_methods

    if not methods_to_process:
        print(f"No compatible methods for {model_name}")
        return stats

    mode_label = "fine-tuned" if finetuned else "frozen"
    print(f"\n{'='*60}")
    print(f"Processing {model_name} [{mode_label}] ({len(layers_to_process)} layers, "
          f"{len(methods_to_process)} methods: {[m.value for m in methods_to_process]})")
    print(f"{'='*60}")

    # Load model
    if finetuned:
        print(f"Loading fine-tuned {model_name} from {checkpoint_path}...")
        model = load_finetuned_model(model_name, checkpoint_path, device)
    else:
        print(f"Loading {model_name}...")
        model = create_model(model_name)
        model.to(device)
        model.eval()

    try:
        # Process each image
        for sample in tqdm(dataset, desc=f"{cache_model_key}"):
            image_id = sample["image_id"]
            image = sample["image"]

            try:
                # Check if all layer/method combos already cached
                if skip_existing:
                    all_cached = all(
                        cache.exists(cache_model_key, f"layer{layer}", image_id, variant=method.value)
                        for layer in layers_to_process
                        for method in methods_to_process
                    )
                    if all_cached:
                        stats["skipped"] += len(layers_to_process) * len(methods_to_process)
                        continue

                # Run inference once
                if finetuned:
                    # FineTunableModel.extract_attention() handles no_grad internally
                    pixel_values = model.preprocess([image])
                    output = model.extract_attention(pixel_values)
                else:
                    # CNN models (like ResNet) need gradients for Grad-CAM
                    is_cnn = model_config.num_heads == 1
                    if is_cnn:
                        preprocessed = model.preprocess([image]).to(device)
                        output = model.forward(preprocessed)
                    else:
                        with torch.no_grad():
                            preprocessed = model.preprocess([image]).to(device)
                            output = model.forward(preprocessed)

                # Extract attention for each layer and method
                for layer in layers_to_process:
                    layer_key = f"layer{layer}"

                    for method in methods_to_process:
                        if skip_existing and cache.exists(
                            cache_model_key, layer_key, image_id, variant=method.value
                        ):
                            stats["skipped"] += 1
                            continue

                        # Compute attention based on method
                        if method == AttentionMethod.GRADCAM:
                            # CNN models return pre-computed heatmaps via Grad-CAM
                            heatmap = output.attention_weights[layer]

                        elif method == AttentionMethod.CLS:
                            # Direct CLS token attention
                            cls_attn = extract_cls_attention(
                                output.attention_weights,
                                layer=layer,
                                num_registers=model_config.num_registers,
                            )
                            heatmap = attention_to_heatmap(
                                cls_attn,
                                image_size=224,
                            )

                        elif method == AttentionMethod.ROLLOUT:
                            # Accumulated attention from layers 0→N
                            # end_layer=layer+1 means "include layer N" (exclusive end)
                            cls_attn = extract_cls_rollout(
                                output.attention_weights,
                                num_registers=model_config.num_registers,
                                end_layer=layer + 1,
                            )
                            heatmap = attention_to_heatmap(
                                cls_attn,
                                image_size=224,
                            )

                        elif method == AttentionMethod.MEAN:
                            # Mean attention (for models without CLS token)
                            cls_attn = extract_mean_attention(
                                output.attention_weights,
                                layer=layer,
                            )
                            heatmap = attention_to_heatmap(
                                cls_attn,
                                image_size=224,
                            )

                        else:
                            print(f"Unknown method: {method}")
                            stats["errors"] += 1
                            continue

                        # Store in cache with method as variant
                        cache.store(
                            cache_model_key,
                            layer_key,
                            image_id,
                            heatmap.squeeze(0),
                            variant=method.value,
                        )
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


def main() -> int:
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
        "--methods",
        nargs="+",
        default=["all"],
        choices=["all", "cls", "rollout", "mean", "gradcam"],
        help="Attention methods to compute (default: all available per model)",
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
    parser.add_argument(
        "--finetuned",
        action="store_true",
        help="Generate cache for fine-tuned models (requires checkpoints)",
    )
    parser.add_argument(
        "--checkpoint-dir",
        type=Path,
        default=CHECKPOINTS_PATH,
        help=f"Directory containing fine-tuned checkpoints (default: {CHECKPOINTS_PATH})",
    )
    args = parser.parse_args()

    # Parse methods (None = all available per model)
    methods_to_use = (
        None if "all" in args.methods else [AttentionMethod(m) for m in args.methods]
    )

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

    # For fine-tuned mode, discover available checkpoints and filter models
    checkpoints: dict[str, Path] = {}
    if args.finetuned:
        checkpoints = discover_checkpoints(args.checkpoint_dir, models_to_process)
        if not checkpoints:
            print("No fine-tuned checkpoints found. Nothing to do.")
            return 1
        models_to_process = list(checkpoints.keys())

    # Setup
    device = args.device or get_device()
    mode_label = "FINE-TUNED" if args.finetuned else "FROZEN"
    print(f"Mode: {mode_label}")
    print(f"Device: {device}")

    dataset = AnnotatedSubset(DATASET_PATH)
    print(f"Dataset: {len(dataset)} annotated images")

    cache = AttentionCache(args.cache_path)
    print(f"Cache: {args.cache_path}")

    if args.finetuned:
        print(f"Checkpoints: {args.checkpoint_dir}")
        print(f"Models with checkpoints: {models_to_process}")

    # Process models one at a time to conserve memory
    total_stats = {"processed": 0, "skipped": 0, "errors": 0}

    for model_name in models_to_process:
        stats = generate_attention_for_model(
            model_name=model_name,
            dataset=dataset,
            cache=cache,
            layers=args.layers,
            methods=methods_to_use,
            device=device,
            skip_existing=not args.no_skip,
            finetuned=args.finetuned,
            checkpoint_path=checkpoints.get(model_name),
        )

        for key in total_stats:
            total_stats[key] += stats[key]

        cache_key = f"{model_name}_finetuned" if args.finetuned else model_name
        print(f"\n{cache_key} complete: {stats}")

    print(f"\n{'='*60}")
    print(f"SUMMARY ({mode_label})")
    print(f"{'='*60}")
    print(f"Total processed: {total_stats['processed']}")
    print(f"Total skipped: {total_stats['skipped']}")
    print(f"Total errors: {total_stats['errors']}")

    return 0 if total_stats["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

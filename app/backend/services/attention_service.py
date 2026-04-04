"""Service for accessing raw attention map values."""

from __future__ import annotations

import sys
from pathlib import Path

# Add SSL attention source to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from app.backend.config import ATTENTION_CACHE_PATH, resolve_model_name, split_model_name
from ssl_attention.cache import AttentionCache

# Patch grid sizes per model (based on 224x224 input / patch_size)
# DINOv2: 14px patch_size -> 224/14 = 16x16 grid (256 patches)
# DINOv3, MAE, CLIP, SigLIP: 16px patch_size -> 224/16 = 14x14 grid (196 patches)
# ResNet-50: 7x7 final feature map (49 spatial positions)
MODEL_ATTENTION_GRIDS: dict[str, tuple[int, int]] = {
    "dinov2": (16, 16),  # 256 patches
    "dinov3": (14, 14),  # 196 patches
    "mae": (14, 14),     # 196 patches
    "clip": (14, 14),    # 196 patches
    "siglip": (14, 14),  # 196 patches
    "siglip2": (14, 14),  # 196 patches
    "resnet50": (7, 7),  # 49 feature positions
}


class AttentionService:
    """Service for loading raw attention values from cache."""

    _instance: AttentionService | None = None
    _cache: AttentionCache | None = None
    _per_head_available_models_cache: list[str] | None = None
    _per_head_available_models_signature: tuple[int, int] | None = None

    def __new__(cls) -> AttentionService:
        """Singleton pattern for cache access."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._cache = AttentionCache(ATTENTION_CACHE_PATH)
        return cls._instance

    @property
    def cache(self) -> AttentionCache:
        """Get the attention cache instance."""
        if self._cache is None:
            self._cache = AttentionCache(ATTENTION_CACHE_PATH)
        return self._cache

    def get_attention_grid(self, model: str) -> tuple[int, int]:
        """Get the attention grid dimensions for a model.

        Args:
            model: Model name (e.g., "dinov2").

        Returns:
            Tuple of (rows, cols) for the attention grid.
        """
        if model in MODEL_ATTENTION_GRIDS:
            return MODEL_ATTENTION_GRIDS[model]
        base_model, _, _ = split_model_name(model)
        if base_model in MODEL_ATTENTION_GRIDS:
            return MODEL_ATTENTION_GRIDS[base_model]
        # Default to 14x14
        return (14, 14)

    def exists(self, model: str, layer: str, image_id: str, method: str = "cls") -> bool:
        """Check if attention is cached for given combination.

        Args:
            model: Model name (resolved to canonical).
            layer: Layer key (e.g., "layer11").
            image_id: Image filename.
            method: Attention method (used as variant).

        Returns:
            True if attention is cached.
        """
        cache_model = resolve_model_name(model)
        result: bool = self.cache.exists(cache_model, layer, image_id, variant=method)
        return result

    def resolve_variant(self, method: str, head: int | None = None) -> str:
        """Build the cache variant key for fused or per-head attention."""
        return f"{method}_head{head}" if head is not None else method

    def list_models_with_per_head_cache(self) -> list[str]:
        """Return base-model names that currently have any per-head attention cached."""
        if not self.cache.path.exists():
            self._per_head_available_models_cache = []
            self._per_head_available_models_signature = None
            return []

        stat = self.cache.path.stat()
        signature = (stat.st_mtime_ns, stat.st_size)
        if (
            self._per_head_available_models_cache is not None
            and self._per_head_available_models_signature == signature
        ):
            return list(self._per_head_available_models_cache)

        available_models: set[str] = set()
        for key in self.cache.list_cached():
            variant = key.variant
            base_variant, sep, head_suffix = variant.rpartition("_head")
            if sep and base_variant and head_suffix.isdigit():
                base_model, _, _ = split_model_name(key.model)
                available_models.add(base_model)

        ordered = sorted(available_models)
        self._per_head_available_models_cache = ordered
        self._per_head_available_models_signature = signature
        return list(ordered)

    def get_raw_attention(
        self,
        image_id: str,
        model: str,
        layer: int,
        method: str = "cls",
        head: int | None = None,
    ) -> dict:
        """Load raw attention values from cache.

        Args:
            image_id: Image filename.
            model: Model name.
            layer: Layer number (0-11).
            method: Attention method (cls, rollout, mean, gradcam).
            head: Optional per-head selector.

        Returns:
            Dict with:
                - attention: List of floats (flattened attention values)
                - shape: [rows, cols] grid dimensions
                - min_value: float
                - max_value: float
        """
        layer_key = f"layer{layer}"
        cache_model = resolve_model_name(model)
        variant = self.resolve_variant(method, head)

        # Load from cache
        try:
            attention_tensor = self.cache.load(cache_model, layer_key, image_id, variant=variant)
        except KeyError as e:
            raise ValueError(
                f"Attention not cached for {model}/{layer_key}/{variant}/{image_id}. "
                "Run generate_attention_cache.py first."
            ) from e

        # Get expected grid dimensions
        grid_rows, grid_cols = self.get_attention_grid(cache_model)

        # Flatten attention tensor and convert to list
        attention_flat = attention_tensor.flatten().tolist()

        # Infer actual shape from tensor
        actual_shape = list(attention_tensor.shape)
        if len(actual_shape) == 2:
            shape = actual_shape
        elif len(actual_shape) == 1:
            # 1D tensor - try to infer square grid
            import math
            size = int(math.sqrt(len(attention_flat)))
            shape = [size, size] if size * size == len(attention_flat) else [grid_rows, grid_cols]
        else:
            shape = [grid_rows, grid_cols]

        return {
            "attention": attention_flat,
            "shape": shape,
            "min_value": min(attention_flat),
            "max_value": max(attention_flat),
        }


# Global instance
attention_service = AttentionService()

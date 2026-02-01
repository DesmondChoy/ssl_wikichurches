"""Service for accessing cached attention maps."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

# Add SSL attention source to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from app.backend.config import ATTENTION_CACHE_PATH, resolve_model_name
from ssl_attention.cache import AttentionCache

if TYPE_CHECKING:
    import torch


class CacheService:
    """Service for loading cached attention maps."""

    _instance: CacheService | None = None
    _cache: AttentionCache | None = None

    def __new__(cls) -> CacheService:
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

    def exists(self, model: str, layer: str, image_id: str) -> bool:
        """Check if attention is cached for given combination."""
        cache_model = resolve_model_name(model)
        result: bool = self.cache.exists(cache_model, layer, image_id)
        return result

    def load(self, model: str, layer: str, image_id: str) -> torch.Tensor:
        """Load cached attention map.

        Args:
            model: Model name (e.g., "dinov2").
            layer: Layer identifier (e.g., "layer11").
            image_id: Image filename.

        Returns:
            Attention tensor of shape (H, W).

        Raises:
            KeyError: If attention not cached.
        """
        cache_model = resolve_model_name(model)
        result: torch.Tensor = self.cache.load(cache_model, layer, image_id)
        return result

    def list_cached_images(self, model: str, layer: str) -> list[str]:
        """List all image IDs cached for a model/layer combination."""
        cache_model = resolve_model_name(model)
        keys = self.cache.list_cached(cache_model)
        return [k.image_id for k in keys if k.layer == layer]

    def get_available_layers(self, model: str, image_id: str) -> list[str]:
        """Get all layers with cached attention for an image."""
        cache_model = resolve_model_name(model)
        keys = self.cache.list_cached(cache_model)
        return sorted(set(k.layer for k in keys if k.image_id == image_id))


# Global instance
cache_service = CacheService()

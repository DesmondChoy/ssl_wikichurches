"""Centralized configuration for SSL attention models.

This module provides a single source of truth for all model configurations
and default parameters used throughout the library.

Usage:
    from ssl_attention.config import MODELS, CACHE_MAX_MODELS

    # Access model config
    dinov2_config = MODELS["dinov2"]
    print(dinov2_config.patch_size)  # 14

    # Access constants
    print(CACHE_MAX_MODELS)  # 2

    # Access data paths
    from ssl_attention.config import DATASET_PATH, STYLE_MAPPING
    print(DATASET_PATH)  # Path to WikiChurches dataset
"""

import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class ModelConfig:
    """Configuration for a vision transformer model.

    Attributes:
        model_id: HuggingFace model identifier (e.g., "facebook/dinov2-with-registers-base").
        patch_size: Size of each image patch in pixels (14 or 16).
        embed_dim: Dimension of token embeddings.
        num_layers: Number of transformer layers.
        num_heads: Number of attention heads per layer.
        num_registers: Number of register tokens (0 if none).
        has_cls_token: Whether the model has a CLS token in the sequence.
            SigLIP uses mean pooling instead of CLS.
    """

    model_id: str
    patch_size: int
    embed_dim: int
    num_layers: int
    num_heads: int
    num_registers: int = 0
    has_cls_token: bool = True


# =============================================================================
# Model Configurations
# =============================================================================
# All model specifications in one place for easy comparison and modification.

MODELS: dict[str, ModelConfig] = {
    "dinov2": ModelConfig(
        model_id="facebook/dinov2-with-registers-base",
        patch_size=14,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        num_registers=4,
        has_cls_token=True,
    ),
    "dinov3": ModelConfig(
        model_id="facebook/dinov3-vitb16-pretrain-lvd1689m",
        patch_size=16,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        num_registers=4,
        has_cls_token=True,
    ),
    "mae": ModelConfig(
        model_id="facebook/vit-mae-base",
        patch_size=16,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        num_registers=0,
        has_cls_token=True,
    ),
    "clip": ModelConfig(
        model_id="openai/clip-vit-base-patch16",
        patch_size=16,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        num_registers=0,
        has_cls_token=True,
    ),
    "siglip": ModelConfig(
        model_id="google/siglip2-base-patch16-224",
        patch_size=16,
        embed_dim=768,
        num_layers=12,
        num_heads=12,
        num_registers=0,
        has_cls_token=False,  # SigLIP uses mean pooling, no CLS token
    ),
}


# =============================================================================
# Model Aliases
# =============================================================================
# Alternative names that map to canonical model names.

MODEL_ALIASES: dict[str, str] = {
    "dino": "dinov2",
    "dinov2-reg": "dinov2",
    "vit-mae": "mae",
    "openai-clip": "clip",
    "siglip2": "siglip",
}


# =============================================================================
# Cache Settings
# =============================================================================

# Maximum number of models to keep in memory via LRU cache.
# Set to 2 to avoid GPU memory exhaustion when switching models.
CACHE_MAX_MODELS: int = 2


# =============================================================================
# Attention Module Defaults
# =============================================================================

# Default image size for attention heatmap generation.
# Standard ViT input size is 224x224.
DEFAULT_IMAGE_SIZE: int = 224

# Small epsilon for numerical stability in attention computations.
EPSILON: float = 1e-8

# Interpolation mode for upsampling attention maps to image size.
INTERPOLATION_MODE: str = "bilinear"


# =============================================================================
# Data Paths
# =============================================================================

# Root path to the WikiChurches dataset.
# Can be overridden via SSL_DATASET_PATH environment variable.
# Path: config.py -> ssl_attention -> src -> ssl_wikichurches -> dataset
_PROJECT_ROOT = Path(__file__).parent.parent.parent
_DEFAULT_DATASET_PATH = _PROJECT_ROOT / "dataset"
DATASET_PATH: Path = Path(os.environ.get("SSL_DATASET_PATH", _DEFAULT_DATASET_PATH))

# Derived paths
IMAGES_PATH: Path = DATASET_PATH / "images"
ANNOTATIONS_PATH: Path = DATASET_PATH / "building_parts.json"
CHURCHES_PATH: Path = DATASET_PATH / "churches.json"

# Cache directory for feature/attention storage
CACHE_PATH: Path = _PROJECT_ROOT / "outputs" / "cache"


# =============================================================================
# Style Classification
# =============================================================================

# Wikidata Q-IDs for the 4 main architectural styles in WikiChurches.
# These 4 styles represent the majority of annotated images (142 of 9,502).
# Note: Q46261=Romanesque, Q176483=Gothic (verified from style_names.txt)
STYLE_MAPPING: dict[str, int] = {
    "Q46261": 0,   # Romanesque (54 images in annotated subset)
    "Q176483": 1,  # Gothic (49 images in annotated subset)
    "Q236122": 2,  # Renaissance (22 images in annotated subset)
    "Q840829": 3,  # Baroque (17 images in annotated subset)
}

# Human-readable style names (indexed by STYLE_MAPPING values)
STYLE_NAMES: tuple[str, ...] = ("Romanesque", "Gothic", "Renaissance", "Baroque")

# Number of architectural styles for classification
NUM_STYLES: int = len(STYLE_MAPPING)

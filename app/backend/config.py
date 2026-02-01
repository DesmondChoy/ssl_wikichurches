"""Backend configuration."""

from __future__ import annotations

import os
from pathlib import Path

# Project root (app/ directory is at project_root/app/)
PROJECT_ROOT = Path(__file__).parent.parent.parent

# SSL Attention source
SSL_ATTENTION_SRC = PROJECT_ROOT / "src"

# Dataset paths
DATASET_PATH = Path(os.environ.get("SSL_DATASET_PATH", PROJECT_ROOT / "dataset"))
IMAGES_PATH = DATASET_PATH / "images"
ANNOTATIONS_PATH = DATASET_PATH / "building_parts.json"

# Cache paths (pre-computed data)
CACHE_PATH = Path(os.environ.get("SSL_CACHE_PATH", PROJECT_ROOT / "outputs" / "cache"))
ATTENTION_CACHE_PATH = CACHE_PATH / "attention_viz.h5"
FEATURE_CACHE_PATH = CACHE_PATH / "features.h5"
HEATMAPS_PATH = CACHE_PATH / "heatmaps"
METRICS_DB_PATH = CACHE_PATH / "metrics.db"
METRICS_SUMMARY_PATH = CACHE_PATH / "metrics_summary.json"

# Available models (must match ssl_attention.config.MODELS)
AVAILABLE_MODELS = ["dinov2", "dinov3", "mae", "clip", "siglip2", "resnet50"]

# Number of transformer layers
NUM_LAYERS = 12

# Style names
STYLE_NAMES = ["Romanesque", "Gothic", "Renaissance", "Baroque"]

# API settings
API_PREFIX = "/api"
CORS_ORIGINS = [
    "http://localhost:3000",
    "http://localhost:5173",  # Vite default
    "http://localhost:5174",  # Vite fallback when 5173 is in use
    "http://127.0.0.1:3000",
    "http://127.0.0.1:5173",
    "http://127.0.0.1:5174",
]

# Image settings
THUMBNAIL_SIZE = (128, 128)
STANDARD_IMAGE_SIZE = (224, 224)

# Model name resolution
from ssl_attention.config import MODEL_ALIASES


def resolve_model_name(model: str) -> str:
    """Resolve model alias to canonical name.

    Args:
        model: Model name (may be an alias like 'siglip2').

    Returns:
        Canonical model name (e.g., 'siglip').
    """
    resolved: str = MODEL_ALIASES.get(model, model)
    return resolved

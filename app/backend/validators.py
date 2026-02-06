"""Shared HTTP validation helpers for API endpoints."""

from __future__ import annotations

from fastapi import HTTPException

from app.backend.config import AVAILABLE_MODELS, DEFAULT_METHOD, get_model_num_layers, resolve_model_name
from ssl_attention.config import AttentionMethod


def resolve_default_method(model: str) -> str:
    """Resolve the default attention method for a model.

    Args:
        model: Model name (may be alias like 'siglip2').

    Returns:
        Default method string (e.g., 'cls', 'mean', 'gradcam').
    """
    resolved = resolve_model_name(model)
    return DEFAULT_METHOD.get(resolved, AttentionMethod.CLS).value


def validate_model(model: str) -> None:
    """Validate model name exists in available models.

    Args:
        model: Model name (may be alias).

    Raises:
        HTTPException: If model is not available.
    """
    if model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model: {model}. Available: {AVAILABLE_MODELS}",
        )


def validate_layer_for_model(layer: int, model: str) -> None:
    """Validate layer is within bounds for the given model.

    Args:
        layer: Layer index (0-based).
        model: Model name (may be alias).

    Raises:
        HTTPException: If layer is out of bounds for the model.
    """
    num_layers = get_model_num_layers(resolve_model_name(model))
    if not 0 <= layer < num_layers:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid layer: {layer}. Model '{model}' has {num_layers} layers (0-{num_layers - 1}).",
        )

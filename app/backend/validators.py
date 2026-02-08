"""Shared HTTP validation helpers for API endpoints."""

from __future__ import annotations

from fastapi import HTTPException

from app.backend.config import (
    AVAILABLE_MODELS,
    DEFAULT_METHOD,
    MODEL_METHODS,
    AttentionMethod,
    get_model_num_layers,
    resolve_model_name,
)


def resolve_default_method(model: str) -> str:
    """Resolve the default attention method for a model.

    Args:
        model: Model name (may be alias like 'siglip2').

    Returns:
        Default method string (e.g., 'cls', 'mean', 'gradcam').
    """
    resolved = resolve_model_name(model)
    method: str = DEFAULT_METHOD.get(resolved, AttentionMethod.CLS).value
    return method


def validate_method(model: str, method: str | None) -> str:
    """Validate and resolve attention method for a model.

    Args:
        model: Model name (may be alias like 'siglip2').
        method: Requested method, or None for default.

    Returns:
        Valid method string.

    Raises:
        HTTPException: If method not available for model.
    """
    resolved_model = resolve_model_name(model)
    available = MODEL_METHODS.get(resolved_model, [])

    if method is None:
        default: AttentionMethod = DEFAULT_METHOD.get(resolved_model, AttentionMethod.CLS)
        return str(default.value)

    # Validate requested method
    try:
        method_enum = AttentionMethod(method)
    except ValueError:
        valid_methods = [m.value for m in AttentionMethod]
        raise HTTPException(
            status_code=400,
            detail=f"Invalid method: '{method}'. Valid methods: {valid_methods}",
        ) from None

    if method_enum not in available:
        available_str = [m.value for m in available]
        raise HTTPException(
            status_code=400,
            detail=f"Method '{method}' not available for '{model}'. Available: {available_str}",
        )

    return method


def validate_model(model: str) -> str:
    """Validate model name and return resolved canonical name.

    Args:
        model: Model name (may be alias like 'siglip2').

    Returns:
        Resolved canonical model name (e.g., 'siglip' for 'siglip2').

    Raises:
        HTTPException: If model is not available.
    """
    if model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model: {model}. Available: {AVAILABLE_MODELS}",
        )
    return resolve_model_name(model)


def validate_layer_for_model(layer: int, model: str) -> str:
    """Validate layer is within bounds and return layer key.

    Args:
        layer: Layer index (0-based).
        model: Model name (may be alias).

    Returns:
        Layer key string (e.g., 'layer5').

    Raises:
        HTTPException: If layer is out of bounds for the model.
    """
    num_layers = get_model_num_layers(resolve_model_name(model))
    if not 0 <= layer < num_layers:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid layer: {layer}. Model '{model}' has {num_layers} layers (0-{num_layers - 1}).",
        )
    return f"layer{layer}"

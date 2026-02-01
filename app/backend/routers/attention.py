"""Attention heatmap serving endpoints."""

from __future__ import annotations

from io import BytesIO
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.backend.config import (
    AVAILABLE_MODELS,
    DEFAULT_METHOD,
    MODEL_METHODS,
    NUM_LAYERS,
    AttentionMethod,
    get_model_num_layers,
    resolve_model_name,
)
from app.backend.schemas.models import BboxInput, RawAttentionResponse, SimilarityResponse
from app.backend.services.attention_service import attention_service
from app.backend.services.image_service import image_service
from app.backend.services.similarity_service import similarity_service

router = APIRouter(prefix="/attention", tags=["attention"])


def validate_model(model: str) -> str:
    """Validate model name and return resolved canonical name."""
    if model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model: {model}. Available: {AVAILABLE_MODELS}",
        )
    return resolve_model_name(model)


def validate_layer(layer: int, model: str) -> str:
    """Validate layer number for a specific model and return layer key.

    Args:
        layer: Layer number to validate.
        model: Canonical model name.

    Returns:
        Layer key string (e.g., "layer0").

    Raises:
        HTTPException: If layer is out of range for the model.
    """
    num_layers = get_model_num_layers(model)
    if not 0 <= layer < num_layers:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid layer: {layer}. Model '{model}' has {num_layers} layers (0-{num_layers - 1}).",
        )
    return f"layer{layer}"


def validate_method(model: str, method: str | None) -> str:
    """Validate and resolve attention method for a model.

    Args:
        model: Canonical model name.
        method: Requested method, or None for default.

    Returns:
        Valid method string.

    Raises:
        HTTPException: If method not available for model.
    """
    resolved_model = resolve_model_name(model)
    available = MODEL_METHODS.get(resolved_model, [])

    if method is None:
        # Return default method for this model
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


@router.get("/{image_id}/heatmap")
async def get_heatmap(
    image_id: str,
    model: Annotated[str, Query(description="Model name")] = "dinov2",
    layer: Annotated[int, Query(ge=0, description="Layer number")] = 11,
    method: Annotated[str | None, Query(description="Attention method (cls, rollout, mean, gradcam)")] = None,
) -> StreamingResponse:
    """Get pure attention heatmap (no overlay).

    Returns the attention map rendered with the configured colormap.
    """
    resolved_model = validate_model(model)
    layer_key = validate_layer(layer, resolved_model)
    resolved_method = validate_method(resolved_model, method)

    if not image_service.heatmap_exists(resolved_model, layer_key, image_id, method=resolved_method, variant="heatmap"):
        raise HTTPException(
            status_code=404,
            detail=f"Heatmap not pre-computed for {resolved_model}/{layer_key}/{resolved_method}/{image_id}",
        )

    try:
        img = image_service.load_heatmap(resolved_model, layer_key, image_id, method=resolved_method, variant="heatmap")

        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        return StreamingResponse(
            buf,
            media_type="image/png",
            headers={"Cache-Control": "max-age=86400"},
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None


@router.get("/{image_id}/overlay")
async def get_overlay(
    image_id: str,
    model: Annotated[str, Query(description="Model name")] = "dinov2",
    layer: Annotated[int, Query(ge=0, description="Layer number")] = 11,
    method: Annotated[str | None, Query(description="Attention method (cls, rollout, mean, gradcam)")] = None,
    show_bboxes: Annotated[bool, Query(description="Include bounding boxes")] = False,
) -> StreamingResponse:
    """Get attention heatmap overlaid on original image.

    Args:
        image_id: Image filename.
        model: Model name.
        layer: Layer number (varies by model).
        method: Attention method (cls, rollout, mean, gradcam). Default per model.
        show_bboxes: If True, also draw bounding box annotations.
    """
    resolved_model = validate_model(model)
    layer_key = validate_layer(layer, resolved_model)
    resolved_method = validate_method(resolved_model, method)

    variant = "overlay_bbox" if show_bboxes else "overlay"

    if not image_service.heatmap_exists(resolved_model, layer_key, image_id, method=resolved_method, variant=variant):
        raise HTTPException(
            status_code=404,
            detail=f"Overlay not pre-computed for {resolved_model}/{layer_key}/{resolved_method}/{image_id}",
        )

    try:
        img = image_service.load_heatmap(resolved_model, layer_key, image_id, method=resolved_method, variant=variant)

        buf = BytesIO()
        img.save(buf, format="PNG")
        buf.seek(0)

        return StreamingResponse(
            buf,
            media_type="image/png",
            headers={"Cache-Control": "max-age=86400"},
        )
    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None


@router.get("/{image_id}/raw", response_model=RawAttentionResponse)
async def get_raw_attention(
    image_id: str,
    model: Annotated[str, Query(description="Model name")] = "dinov2",
    layer: Annotated[int, Query(ge=0, description="Layer number")] = 11,
    method: Annotated[str | None, Query(description="Attention method (cls, rollout, mean, gradcam)")] = None,
) -> RawAttentionResponse:
    """Get raw attention values for client-side rendering.

    Returns the attention map as a flat array of values with grid dimensions,
    enabling client-side percentile thresholding and dynamic visualization.

    Args:
        image_id: Image filename.
        model: Model name.
        layer: Layer number (varies by model).
        method: Attention method (cls, rollout, mean, gradcam). Default per model.
    """
    resolved_model = validate_model(model)
    layer_key = validate_layer(layer, resolved_model)
    resolved_method = validate_method(resolved_model, method)

    # Check if attention is cached
    if not attention_service.exists(resolved_model, layer_key, image_id, method=resolved_method):
        raise HTTPException(
            status_code=404,
            detail=f"Attention not cached for {resolved_model}/{layer_key}/{resolved_method}/{image_id}. "
            "Run generate_attention_cache.py first.",
        )

    try:
        result = attention_service.get_raw_attention(
            image_id=image_id,
            model=resolved_model,
            layer=layer,
            method=resolved_method,
        )
        return RawAttentionResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=404, detail=str(e)) from None
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error loading attention: {e}"
        ) from None


@router.get("/{image_id}/layers")
async def get_all_layer_overlays(
    image_id: str,
    model: Annotated[str, Query(description="Model name")] = "dinov2",
    method: Annotated[str | None, Query(description="Attention method (cls, rollout, mean, gradcam)")] = None,
    show_bboxes: Annotated[bool, Query(description="Include bounding boxes")] = False,
) -> dict:
    """Get URLs for attention overlays at all layers.

    Used for layer progression animation.
    """
    resolved_model = validate_model(model)
    resolved_method = validate_method(resolved_model, method)
    num_layers = get_model_num_layers(resolved_model)

    variant = "overlay_bbox" if show_bboxes else "overlay"

    layers = {}
    for layer in range(num_layers):
        layer_key = f"layer{layer}"
        if image_service.heatmap_exists(resolved_model, layer_key, image_id, method=resolved_method, variant=variant):
            layers[layer_key] = (
                f"/api/attention/{image_id}/overlay?"
                f"model={model}&layer={layer}&method={resolved_method}&show_bboxes={show_bboxes}"
            )

    if not layers:
        raise HTTPException(
            status_code=404,
            detail=f"No layers pre-computed for {resolved_model}/{resolved_method}/{image_id}",
        )

    return {
        "image_id": image_id,
        "model": model,
        "method": resolved_method,
        "show_bboxes": show_bboxes,
        "layers": layers,
    }


@router.get("/models")
async def list_models() -> dict:
    """List available models and their configurations including attention methods."""
    # Use original model names as keys (e.g., 'siglip2' not 'siglip')
    # so frontend can look up by the name it uses
    return {
        "models": AVAILABLE_MODELS,
        "num_layers": NUM_LAYERS,  # Legacy: global default for backwards compatibility
        "num_layers_per_model": {
            m: get_model_num_layers(resolve_model_name(m))
            for m in AVAILABLE_MODELS
        },
        "methods": {
            m: [method.value for method in MODEL_METHODS.get(resolve_model_name(m), [])]
            for m in AVAILABLE_MODELS
        },
        "default_methods": {
            m: DEFAULT_METHOD.get(resolve_model_name(m), AttentionMethod.CLS).value
            for m in AVAILABLE_MODELS
        },
    }


@router.post("/{image_id}/similarity", response_model=SimilarityResponse)
async def compute_bbox_similarity(
    image_id: str,
    bbox: BboxInput,
    model: Annotated[str, Query(description="Model name")] = "dinov2",
    layer: Annotated[int, Query(ge=0, description="Layer number")] = 11,
) -> SimilarityResponse:
    """Compute cosine similarity between a bounding box and all image patches.

    This endpoint enables interactive exploration of which image regions have
    similar learned features to a selected architectural element.

    Args:
        image_id: Image filename.
        bbox: Bounding box coordinates (normalized 0-1).
        model: Model name.
        layer: Layer number (varies by model).

    Returns:
        SimilarityResponse with similarity values for each patch.
    """
    resolved_model = validate_model(model)
    validate_layer(layer, resolved_model)

    # Check if features are cached
    if not similarity_service.features_exist(resolved_model, layer, image_id):
        raise HTTPException(
            status_code=404,
            detail=f"Features not pre-computed for {resolved_model}/layer{layer}/{image_id}. "
            "Run generate_feature_cache.py first.",
        )

    try:
        result = similarity_service.compute_similarity(
            image_id=image_id,
            model=resolved_model,
            layer=layer,
            left=bbox.left,
            top=bbox.top,
            width=bbox.width,
            height=bbox.height,
        )
        return SimilarityResponse(**result)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e)) from None
    except Exception as e:
        raise HTTPException(
            status_code=500, detail=f"Error computing similarity: {e}"
        ) from None

"""Attention heatmap serving endpoints."""

from __future__ import annotations

from io import BytesIO
from typing import Annotated

from fastapi import APIRouter, HTTPException, Query
from fastapi.responses import StreamingResponse

from app.backend.config import AVAILABLE_MODELS, NUM_LAYERS
from app.backend.schemas.models import BboxInput, SimilarityResponse
from app.backend.services.image_service import image_service
from app.backend.services.similarity_service import similarity_service

router = APIRouter(prefix="/attention", tags=["attention"])


def validate_model(model: str) -> None:
    """Validate model name."""
    if model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model: {model}. Available: {AVAILABLE_MODELS}",
        )


def validate_layer(layer: int) -> str:
    """Validate layer number and return layer key."""
    if not 0 <= layer < NUM_LAYERS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid layer: {layer}. Must be 0-{NUM_LAYERS - 1}",
        )
    return f"layer{layer}"


@router.get("/{image_id}/heatmap")
async def get_heatmap(
    image_id: str,
    model: Annotated[str, Query(description="Model name")] = "dinov2",
    layer: Annotated[int, Query(ge=0, lt=NUM_LAYERS, description="Layer number")] = 11,
) -> StreamingResponse:
    """Get pure attention heatmap (no overlay).

    Returns the attention map rendered with the configured colormap.
    """
    validate_model(model)
    layer_key = validate_layer(layer)

    if not image_service.heatmap_exists(model, layer_key, image_id, variant="heatmap"):
        raise HTTPException(
            status_code=404,
            detail=f"Heatmap not pre-computed for {model}/{layer_key}/{image_id}",
        )

    try:
        img = image_service.load_heatmap(model, layer_key, image_id, variant="heatmap")

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
    layer: Annotated[int, Query(ge=0, lt=NUM_LAYERS, description="Layer number")] = 11,
    show_bboxes: Annotated[bool, Query(description="Include bounding boxes")] = False,
) -> StreamingResponse:
    """Get attention heatmap overlaid on original image.

    Args:
        image_id: Image filename.
        model: Model name.
        layer: Layer number (0-11).
        show_bboxes: If True, also draw bounding box annotations.
    """
    validate_model(model)
    layer_key = validate_layer(layer)

    variant = "overlay_bbox" if show_bboxes else "overlay"

    if not image_service.heatmap_exists(model, layer_key, image_id, variant=variant):
        raise HTTPException(
            status_code=404,
            detail=f"Overlay not pre-computed for {model}/{layer_key}/{image_id}",
        )

    try:
        img = image_service.load_heatmap(model, layer_key, image_id, variant=variant)

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


@router.get("/{image_id}/layers")
async def get_all_layer_overlays(
    image_id: str,
    model: Annotated[str, Query(description="Model name")] = "dinov2",
    show_bboxes: Annotated[bool, Query(description="Include bounding boxes")] = False,
) -> dict:
    """Get URLs for attention overlays at all layers.

    Used for layer progression animation.
    """
    validate_model(model)

    variant = "overlay_bbox" if show_bboxes else "overlay"

    layers = {}
    for layer in range(NUM_LAYERS):
        layer_key = f"layer{layer}"
        if image_service.heatmap_exists(model, layer_key, image_id, variant=variant):
            layers[layer_key] = f"/api/attention/{image_id}/overlay?model={model}&layer={layer}&show_bboxes={show_bboxes}"

    if not layers:
        raise HTTPException(
            status_code=404,
            detail=f"No layers pre-computed for {model}/{image_id}",
        )

    return {
        "image_id": image_id,
        "model": model,
        "show_bboxes": show_bboxes,
        "layers": layers,
    }


@router.get("/models")
async def list_models() -> dict:
    """List available models and their configurations."""
    return {
        "models": AVAILABLE_MODELS,
        "num_layers": NUM_LAYERS,
    }


@router.post("/{image_id}/similarity", response_model=SimilarityResponse)
async def compute_bbox_similarity(
    image_id: str,
    bbox: BboxInput,
    model: Annotated[str, Query(description="Model name")] = "dinov2",
    layer: Annotated[int, Query(ge=0, lt=NUM_LAYERS, description="Layer number")] = 11,
) -> SimilarityResponse:
    """Compute cosine similarity between a bounding box and all image patches.

    This endpoint enables interactive exploration of which image regions have
    similar learned features to a selected architectural element.

    Args:
        image_id: Image filename.
        bbox: Bounding box coordinates (normalized 0-1).
        model: Model name.
        layer: Layer number (0-11).

    Returns:
        SimilarityResponse with similarity values for each patch.
    """
    validate_model(model)
    validate_layer(layer)

    # Check if features are cached
    if not similarity_service.features_exist(model, layer, image_id):
        raise HTTPException(
            status_code=404,
            detail=f"Features not pre-computed for {model}/layer{layer}/{image_id}. "
            "Run generate_feature_cache.py first.",
        )

    try:
        result = similarity_service.compute_similarity(
            image_id=image_id,
            model=model,
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

"""Comparison endpoints for model vs model analysis."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from app.backend.config import AVAILABLE_MODELS, get_model_num_layers, resolve_model_name
from app.backend.schemas import IoUResultSchema, ModelComparisonSchema
from app.backend.services.image_service import image_service
from app.backend.services.metrics_service import metrics_service

router = APIRouter(prefix="/compare", tags=["comparison"])


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


@router.get("/models", response_model=ModelComparisonSchema)
async def compare_models(
    image_id: str,
    models: Annotated[list[str] | None, Query(description="Models to compare")] = None,
    layer: Annotated[int, Query(ge=0)] = 0,
    percentile: Annotated[int, Query(ge=50, le=95)] = 90,
) -> ModelComparisonSchema:
    """Compare multiple models on a single image.

    Returns IoU results and heatmap URLs for side-by-side comparison.
    """
    # Default models if not specified
    if models is None:
        models = ["dinov2", "clip"]

    # Validate models
    invalid = [m for m in models if m not in AVAILABLE_MODELS]
    if invalid:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid models: {invalid}. Available: {AVAILABLE_MODELS}",
        )

    # Validate layer for all requested models
    for model in models:
        validate_layer_for_model(layer, model)

    # Check image exists
    if not image_service.get_annotation(image_id):
        raise HTTPException(status_code=404, detail=f"Image not found: {image_id}")

    if not metrics_service.db_exists:
        raise HTTPException(
            status_code=503,
            detail="Metrics database not available.",
        )

    layer_key = f"layer{layer}"
    results = []
    heatmap_urls = {}

    for model in models:
        # Get metrics
        metrics = metrics_service.get_image_metrics(image_id, model, layer_key, percentile)
        if metrics:
            results.append(IoUResultSchema(**metrics))

        # Get heatmap URL
        if image_service.heatmap_exists(model, layer_key, image_id, variant="overlay"):
            heatmap_urls[model] = f"/api/attention/{image_id}/overlay?model={model}&layer={layer}"

    if not results:
        raise HTTPException(
            status_code=404,
            detail=f"No metrics found for {image_id}",
        )

    return ModelComparisonSchema(
        image_id=image_id,
        models=models,
        layer=layer_key,
        percentile=percentile,
        results=results,
        heatmap_urls=heatmap_urls,
    )


@router.get("/frozen_vs_finetuned")
async def compare_frozen_vs_finetuned(
    image_id: str,
    model: Annotated[str, Query()] = "dinov2",
    layer: Annotated[int, Query(ge=0)] = 0,
) -> dict:
    """Compare frozen (pretrained) vs fine-tuned model attention.

    Note: Fine-tuned models are not yet available. This endpoint
    returns URLs for the frozen model with placeholder for fine-tuned.
    """
    if model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model: {model}. Available: {AVAILABLE_MODELS}",
        )

    validate_layer_for_model(layer, model)

    if not image_service.get_annotation(image_id):
        raise HTTPException(status_code=404, detail=f"Image not found: {image_id}")

    layer_key = f"layer{layer}"

    # Frozen model (always available after pre-computation)
    frozen_available = image_service.heatmap_exists(model, layer_key, image_id, variant="overlay")

    # Fine-tuned model (placeholder - will be available after Phase 5)
    finetuned_model = f"{model}_finetuned"
    finetuned_available = image_service.heatmap_exists(finetuned_model, layer_key, image_id, variant="overlay")

    return {
        "image_id": image_id,
        "model": model,
        "layer": layer_key,
        "frozen": {
            "available": frozen_available,
            "url": f"/api/attention/{image_id}/overlay?model={model}&layer={layer}" if frozen_available else None,
        },
        "finetuned": {
            "available": finetuned_available,
            "url": f"/api/attention/{image_id}/overlay?model={finetuned_model}&layer={layer}" if finetuned_available else None,
            "note": "Fine-tuned models will be available after Phase 5 training",
        },
    }


@router.get("/layers")
async def compare_layers(
    image_id: str,
    model: Annotated[str, Query()] = "dinov2",
    percentile: Annotated[int, Query(ge=50, le=95)] = 90,
) -> dict:
    """Get IoU progression across layers for layer comparison.

    Used for layer progression animation and analysis.
    """
    if model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model: {model}. Available: {AVAILABLE_MODELS}",
        )

    if not image_service.get_annotation(image_id):
        raise HTTPException(status_code=404, detail=f"Image not found: {image_id}")

    if not metrics_service.db_exists:
        raise HTTPException(
            status_code=503,
            detail="Metrics database not available.",
        )

    # Get per-model layer count
    num_layers = get_model_num_layers(resolve_model_name(model))

    layers_data = []
    for layer in range(num_layers):
        layer_key = f"layer{layer}"
        metrics = metrics_service.get_image_metrics(image_id, model, layer_key, percentile)

        if metrics:
            has_heatmap = image_service.heatmap_exists(model, layer_key, image_id, variant="overlay")
            layers_data.append({
                "layer": layer,
                "layer_key": layer_key,
                "iou": metrics["iou"],
                "coverage": metrics["coverage"],
                "heatmap_url": f"/api/attention/{image_id}/overlay?model={model}&layer={layer}" if has_heatmap else None,
            })

    if not layers_data:
        raise HTTPException(
            status_code=404,
            detail=f"No metrics found for {image_id}",
        )

    # Find best layer
    best = max(layers_data, key=lambda x: x["iou"])

    return {
        "image_id": image_id,
        "model": model,
        "percentile": percentile,
        "layers": layers_data,
        "best_layer": best["layer"],
        "best_iou": best["iou"],
    }


@router.get("/all_models_summary")
async def compare_all_models_summary(
    percentile: Annotated[int, Query(ge=50, le=95)] = 90,
) -> dict:
    """Get summary comparison of all models.

    Returns best layer and IoU for each model.
    """
    if not metrics_service.db_exists:
        raise HTTPException(
            status_code=503,
            detail="Metrics database not available.",
        )

    leaderboard = metrics_service.get_leaderboard(percentile)

    # Get layer progression for each model
    models_data = {}
    for entry in leaderboard:
        model = entry["model"]
        progression = metrics_service.get_layer_progression(model, percentile)
        models_data[model] = {
            "rank": entry["rank"],
            "best_iou": entry["best_iou"],
            "best_layer": entry["best_layer"],
            "layer_progression": dict(zip(progression["layers"], progression["ious"], strict=True)),
        }

    return {
        "percentile": percentile,
        "models": models_data,
        "leaderboard": leaderboard,
    }

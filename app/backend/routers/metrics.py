"""Metrics query endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from app.backend.config import AVAILABLE_MODELS, NUM_LAYERS
from app.backend.schemas import (
    FeatureBreakdownSchema,
    IoUResultSchema,
    LayerProgressionSchema,
    LeaderboardEntry,
    StyleBreakdownSchema,
)
from app.backend.services.metrics_service import metrics_service

router = APIRouter(prefix="/metrics", tags=["metrics"])


def validate_model(model: str) -> None:
    """Validate model name."""
    if model not in AVAILABLE_MODELS:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model: {model}. Available: {AVAILABLE_MODELS}",
        )


@router.get("/leaderboard", response_model=list[LeaderboardEntry])
async def get_leaderboard(
    percentile: Annotated[int, Query(ge=50, le=95)] = 90,
) -> list[LeaderboardEntry]:
    """Get model rankings by best IoU score.

    Returns models ranked by their best IoU at the given percentile.
    """
    if not metrics_service.db_exists:
        raise HTTPException(
            status_code=503,
            detail="Metrics database not available. Run generate_metrics_cache.py first.",
        )

    data = metrics_service.get_leaderboard(percentile)
    return [LeaderboardEntry(**entry) for entry in data]


@router.get("/summary")
async def get_summary() -> dict:
    """Get pre-computed metrics summary.

    Returns overall statistics including leaderboard and per-model best layers.
    """
    summary = metrics_service.get_summary()
    if not summary:
        raise HTTPException(
            status_code=503,
            detail="Metrics summary not available. Run generate_metrics_cache.py first.",
        )
    return summary


@router.get("/{image_id}", response_model=IoUResultSchema)
async def get_image_metrics(
    image_id: str,
    model: Annotated[str, Query()] = "dinov2",
    layer: Annotated[int, Query(ge=0, lt=NUM_LAYERS)] = 11,
    percentile: Annotated[int, Query(ge=50, le=95)] = 90,
) -> IoUResultSchema:
    """Get IoU metrics for a specific image.

    Returns IoU, coverage, and area statistics.
    """
    validate_model(model)
    layer_key = f"layer{layer}"

    if not metrics_service.db_exists:
        raise HTTPException(
            status_code=503,
            detail="Metrics database not available.",
        )

    result = metrics_service.get_image_metrics(image_id, model, layer_key, percentile)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Metrics not found for {image_id} with {model}/{layer_key}",
        )

    return IoUResultSchema(**result)


@router.get("/{image_id}/all_models")
async def get_image_metrics_all_models(
    image_id: str,
    layer: Annotated[int, Query(ge=0, lt=NUM_LAYERS)] = 11,
    percentile: Annotated[int, Query(ge=50, le=95)] = 90,
) -> dict:
    """Get metrics for an image across all models.

    Useful for model comparison on a single image.
    """
    if not metrics_service.db_exists:
        raise HTTPException(
            status_code=503,
            detail="Metrics database not available.",
        )

    layer_key = f"layer{layer}"
    results = {}

    for model in AVAILABLE_MODELS:
        result = metrics_service.get_image_metrics(image_id, model, layer_key, percentile)
        if result:
            results[model] = result

    if not results:
        raise HTTPException(
            status_code=404,
            detail=f"No metrics found for {image_id}",
        )

    return {
        "image_id": image_id,
        "layer": layer_key,
        "percentile": percentile,
        "models": results,
    }


@router.get("/model/{model}/progression", response_model=LayerProgressionSchema)
async def get_layer_progression(
    model: str,
    percentile: Annotated[int, Query(ge=50, le=95)] = 90,
) -> LayerProgressionSchema:
    """Get IoU progression across all layers for a model.

    Shows how attention alignment evolves through transformer layers.
    """
    validate_model(model)

    if not metrics_service.db_exists:
        raise HTTPException(
            status_code=503,
            detail="Metrics database not available.",
        )

    data = metrics_service.get_layer_progression(model, percentile)
    return LayerProgressionSchema(**data)


@router.get("/model/{model}/style_breakdown", response_model=StyleBreakdownSchema)
async def get_style_breakdown(
    model: str,
    layer: Annotated[int, Query(ge=0, lt=NUM_LAYERS)] = 11,
    percentile: Annotated[int, Query(ge=50, le=95)] = 90,
) -> StyleBreakdownSchema:
    """Get IoU breakdown by architectural style.

    Shows how well the model attends to different architectural styles.
    """
    validate_model(model)
    layer_key = f"layer{layer}"

    if not metrics_service.db_exists:
        raise HTTPException(
            status_code=503,
            detail="Metrics database not available.",
        )

    data = metrics_service.get_style_breakdown(model, layer_key, percentile)
    return StyleBreakdownSchema(**data)


@router.get("/model/{model}/feature_breakdown", response_model=FeatureBreakdownSchema)
async def get_feature_breakdown(
    model: str,
    layer: Annotated[int, Query(ge=0, lt=NUM_LAYERS)] = 11,
    percentile: Annotated[int, Query(ge=50, le=95)] = 90,
    min_count: Annotated[int, Query(ge=0)] = 0,
    sort_by: Annotated[str, Query(enum=["mean_iou", "bbox_count", "feature_name", "feature_label"])] = "mean_iou",
) -> FeatureBreakdownSchema:
    """Get IoU breakdown by architectural feature type.

    Shows how well the model attends to different architectural features
    (e.g., windows, doors, arches) across all 106 feature types.
    """
    validate_model(model)
    layer_key = f"layer{layer}"

    if not metrics_service.db_exists:
        raise HTTPException(
            status_code=503,
            detail="Metrics database not available.",
        )

    data = metrics_service.get_feature_breakdown(
        model, layer_key, percentile, sort_by=sort_by, min_count=min_count
    )
    return FeatureBreakdownSchema(**data)


@router.get("/model/{model}/aggregate")
async def get_aggregate_metrics(
    model: str,
    layer: Annotated[int, Query(ge=0, lt=NUM_LAYERS)] = 11,
    percentile: Annotated[int, Query(ge=50, le=95)] = 90,
) -> dict:
    """Get aggregate metrics for a model/layer combination.

    Returns mean, std, median IoU across all images.
    """
    validate_model(model)
    layer_key = f"layer{layer}"

    if not metrics_service.db_exists:
        raise HTTPException(
            status_code=503,
            detail="Metrics database not available.",
        )

    result = metrics_service.get_aggregate_metrics(model, layer_key, percentile)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Aggregate metrics not found for {model}/{layer_key}",
        )

    return result


@router.get("/model/{model}/all_images")
async def get_all_images_metrics(
    model: str,
    layer: Annotated[int, Query(ge=0, lt=NUM_LAYERS)] = 11,
    percentile: Annotated[int, Query(ge=50, le=95)] = 90,
    sort_by: Annotated[str, Query(enum=["iou", "coverage"])] = "iou",
    limit: Annotated[int, Query(ge=1, le=200)] = 139,
) -> dict:
    """Get metrics for all images for a model/layer.

    Returns list of images sorted by IoU or coverage.
    """
    validate_model(model)
    layer_key = f"layer{layer}"

    if not metrics_service.db_exists:
        raise HTTPException(
            status_code=503,
            detail="Metrics database not available.",
        )

    results = metrics_service.get_all_image_metrics(model, layer_key, percentile)

    # Sort
    if sort_by == "coverage":
        results.sort(key=lambda x: x["coverage"], reverse=True)
    # Already sorted by IoU from DB

    # Limit
    results = results[:limit]

    return {
        "model": model,
        "layer": layer_key,
        "percentile": percentile,
        "count": len(results),
        "images": results,
    }

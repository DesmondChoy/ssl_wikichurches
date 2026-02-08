"""Metrics query endpoints."""

from __future__ import annotations

from typing import Annotated

from fastapi import APIRouter, HTTPException, Query

from app.backend.config import AVAILABLE_MODELS, get_model_num_layers, resolve_model_name
from app.backend.schemas import (
    FeatureBreakdownSchema,
    IoUResultSchema,
    LayerProgressionSchema,
    LeaderboardEntry,
    StyleBreakdownSchema,
)
from app.backend.services.attention_service import attention_service
from app.backend.services.image_service import image_service
from app.backend.services.metrics_service import metrics_service
from app.backend.validators import validate_layer_for_model, validate_method, validate_model
from ssl_attention.metrics.iou import compute_coverage, compute_iou

router = APIRouter(prefix="/metrics", tags=["metrics"])


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
    layer: Annotated[int, Query(ge=0)] = 0,
    percentile: Annotated[int, Query(ge=50, le=95)] = 90,
    method: Annotated[str | None, Query(description="Attention method (cls, rollout, mean, gradcam)")] = None,
) -> IoUResultSchema:
    """Get IoU metrics for a specific image.

    Returns IoU, coverage, and area statistics.
    """
    validate_model(model)
    layer_key = validate_layer_for_model(layer, model)
    resolved_method = validate_method(model, method)

    if not metrics_service.db_exists:
        raise HTTPException(
            status_code=503,
            detail="Metrics database not available.",
        )

    result = metrics_service.get_image_metrics(image_id, model, layer_key, percentile, method=resolved_method)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Metrics not found for {image_id} with {model}/{layer_key}",
        )

    return IoUResultSchema(**result)


@router.get("/{image_id}/all_models")
async def get_image_metrics_all_models(
    image_id: str,
    layer: Annotated[int, Query(ge=0)] = 0,
    percentile: Annotated[int, Query(ge=50, le=95)] = 90,
    method: Annotated[str | None, Query(description="Attention method (cls, rollout, mean, gradcam)")] = None,
) -> dict:
    """Get metrics for an image across all models.

    Useful for model comparison on a single image.
    Note: Layer validation is done per-model in the loop since models have different layer counts.
    When method is specified, only models supporting that method are included.
    """
    if not metrics_service.db_exists:
        raise HTTPException(
            status_code=503,
            detail="Metrics database not available.",
        )

    layer_key = f"layer{layer}"
    results = {}

    for model in AVAILABLE_MODELS:
        # Skip models that don't have this layer
        num_layers = get_model_num_layers(resolve_model_name(model))
        if layer >= num_layers:
            continue

        # Skip models that don't support the requested method
        try:
            resolved_method = validate_method(model, method)
        except HTTPException:
            continue

        result = metrics_service.get_image_metrics(image_id, model, layer_key, percentile, method=resolved_method)
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


@router.get("/{image_id}/bbox/{bbox_index}", response_model=IoUResultSchema)
async def get_bbox_metrics(
    image_id: str,
    bbox_index: int,
    model: Annotated[str, Query()] = "dinov2",
    layer: Annotated[int, Query(ge=0)] = 0,
    percentile: Annotated[int, Query(ge=50, le=95)] = 90,
    method: Annotated[str | None, Query(description="Attention method (cls, rollout, mean, gradcam)")] = None,
) -> IoUResultSchema:
    """Get IoU metrics for a specific bounding box.

    Computes IoU and coverage on-the-fly for a single bbox
    against the attention map (rather than the union of all bboxes).
    """
    resolved_model = validate_model(model)
    layer_key = validate_layer_for_model(layer, model)
    resolved_method = validate_method(model, method)

    # Load annotation
    annotation = image_service.get_annotation(image_id)
    if not annotation:
        raise HTTPException(status_code=404, detail=f"Annotation not found for {image_id}")

    # Validate bbox index
    if not 0 <= bbox_index < len(annotation.bboxes):
        raise HTTPException(
            status_code=400,
            detail=f"bbox_index {bbox_index} out of range. Image has {len(annotation.bboxes)} bboxes (0-{len(annotation.bboxes) - 1}).",
        )

    # Load attention tensor from HDF5 cache
    try:
        attention_tensor = attention_service.cache.load(
            resolved_model, layer_key, image_id, variant=resolved_method
        )
    except KeyError:
        raise HTTPException(
            status_code=404,
            detail=f"Attention not cached for {model}/{layer_key}/{resolved_method}/{image_id}. Run precompute first.",
        ) from None

    # Reshape 1D tensors to 2D grid
    if attention_tensor.dim() == 1:
        grid_rows, grid_cols = attention_service.get_attention_grid(resolved_model)
        attention_tensor = attention_tensor.reshape(grid_rows, grid_cols)

    # Generate mask for the specific bbox
    h, w = attention_tensor.shape[-2:]
    bbox = annotation.bboxes[bbox_index]
    bbox_mask = bbox.to_mask(h, w)

    # Compute metrics
    iou, attention_area, annotation_area = compute_iou(attention_tensor, bbox_mask, percentile)
    coverage = compute_coverage(attention_tensor, bbox_mask)

    return IoUResultSchema(
        image_id=image_id,
        model=model,
        layer=layer_key,
        percentile=percentile,
        iou=iou,
        coverage=coverage,
        attention_area=attention_area,
        annotation_area=annotation_area,
        method=resolved_method,
    )


@router.get("/model/{model}/progression", response_model=LayerProgressionSchema)
async def get_layer_progression(
    model: str,
    percentile: Annotated[int, Query(ge=50, le=95)] = 90,
    method: Annotated[str | None, Query(description="Attention method (cls, rollout, mean, gradcam)")] = None,
) -> LayerProgressionSchema:
    """Get IoU progression across all layers for a model.

    Shows how attention alignment evolves through transformer layers.
    """
    validate_model(model)
    resolved_method = validate_method(model, method)

    if not metrics_service.db_exists:
        raise HTTPException(
            status_code=503,
            detail="Metrics database not available.",
        )

    data = metrics_service.get_layer_progression(model, percentile, method=resolved_method)
    return LayerProgressionSchema(**data)


@router.get("/model/{model}/style_breakdown", response_model=StyleBreakdownSchema)
async def get_style_breakdown(
    model: str,
    layer: Annotated[int, Query(ge=0)] = 0,
    percentile: Annotated[int, Query(ge=50, le=95)] = 90,
    method: Annotated[str | None, Query(description="Attention method (cls, rollout, mean, gradcam)")] = None,
) -> StyleBreakdownSchema:
    """Get IoU breakdown by architectural style.

    Shows how well the model attends to different architectural styles.
    """
    validate_model(model)
    layer_key = validate_layer_for_model(layer, model)
    resolved_method = validate_method(model, method)

    if not metrics_service.db_exists:
        raise HTTPException(
            status_code=503,
            detail="Metrics database not available.",
        )

    data = metrics_service.get_style_breakdown(model, layer_key, percentile, method=resolved_method)
    return StyleBreakdownSchema(**data)


@router.get("/model/{model}/feature_breakdown", response_model=FeatureBreakdownSchema)
async def get_feature_breakdown(
    model: str,
    layer: Annotated[int, Query(ge=0)] = 0,
    percentile: Annotated[int, Query(ge=50, le=95)] = 90,
    min_count: Annotated[int, Query(ge=0)] = 0,
    sort_by: Annotated[str, Query(enum=["mean_iou", "bbox_count", "feature_name", "feature_label"])] = "mean_iou",
    method: Annotated[str | None, Query(description="Attention method (cls, rollout, mean, gradcam)")] = None,
) -> FeatureBreakdownSchema:
    """Get IoU breakdown by architectural feature type.

    Shows how well the model attends to different architectural features
    (e.g., windows, doors, arches) across all 106 feature types.
    """
    validate_model(model)
    layer_key = validate_layer_for_model(layer, model)
    resolved_method = validate_method(model, method)

    if not metrics_service.db_exists:
        raise HTTPException(
            status_code=503,
            detail="Metrics database not available.",
        )

    data = metrics_service.get_feature_breakdown(
        model, layer_key, percentile, sort_by=sort_by, min_count=min_count, method=resolved_method
    )
    return FeatureBreakdownSchema(**data)


@router.get("/model/{model}/aggregate")
async def get_aggregate_metrics(
    model: str,
    layer: Annotated[int, Query(ge=0)] = 0,
    percentile: Annotated[int, Query(ge=50, le=95)] = 90,
    method: Annotated[str | None, Query(description="Attention method (cls, rollout, mean, gradcam)")] = None,
) -> dict:
    """Get aggregate metrics for a model/layer combination.

    Returns mean, std, median IoU across all images.
    """
    validate_model(model)
    layer_key = validate_layer_for_model(layer, model)
    resolved_method = validate_method(model, method)

    if not metrics_service.db_exists:
        raise HTTPException(
            status_code=503,
            detail="Metrics database not available.",
        )

    result = metrics_service.get_aggregate_metrics(model, layer_key, percentile, method=resolved_method)
    if not result:
        raise HTTPException(
            status_code=404,
            detail=f"Aggregate metrics not found for {model}/{layer_key}",
        )

    return result


@router.get("/model/{model}/all_images")
async def get_all_images_metrics(
    model: str,
    layer: Annotated[int, Query(ge=0)] = 0,
    percentile: Annotated[int, Query(ge=50, le=95)] = 90,
    sort_by: Annotated[str, Query(enum=["iou", "coverage"])] = "iou",
    limit: Annotated[int, Query(ge=1, le=200)] = 139,
    method: Annotated[str | None, Query(description="Attention method (cls, rollout, mean, gradcam)")] = None,
) -> dict:
    """Get metrics for all images for a model/layer.

    Returns list of images sorted by IoU or coverage.
    """
    validate_model(model)
    layer_key = validate_layer_for_model(layer, model)
    resolved_method = validate_method(model, method)

    if not metrics_service.db_exists:
        raise HTTPException(
            status_code=503,
            detail="Metrics database not available.",
        )

    results = metrics_service.get_all_image_metrics(model, layer_key, percentile, method=resolved_method)

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

"""Pydantic models for API request/response validation."""

from __future__ import annotations

from pydantic import BaseModel, Field


class BoundingBoxSchema(BaseModel):
    """A single bounding box annotation."""

    left: float = Field(..., ge=0, le=1, description="Left edge (0-1)")
    top: float = Field(..., ge=0, le=1, description="Top edge (0-1)")
    width: float = Field(..., ge=0, le=1, description="Width (0-1)")
    height: float = Field(..., ge=0, le=1, description="Height (0-1)")
    label: int = Field(..., ge=0, description="Feature type index")
    label_name: str | None = Field(None, description="Human-readable feature name")


class ImageAnnotationSchema(BaseModel):
    """Complete annotation for an image."""

    image_id: str
    styles: list[str]
    style_names: list[str]
    num_bboxes: int
    bboxes: list[BoundingBoxSchema]


class ImageListItem(BaseModel):
    """Summary info for image list view."""

    image_id: str
    thumbnail_url: str
    styles: list[str]
    style_names: list[str]
    num_bboxes: int


class ImageDetailSchema(BaseModel):
    """Detailed image information."""

    image_id: str
    image_url: str
    thumbnail_url: str
    annotation: ImageAnnotationSchema
    available_models: list[str]


class IoUResultSchema(BaseModel):
    """IoU computation result."""

    image_id: str
    model: str
    layer: str
    percentile: int
    iou: float
    coverage: float
    attention_area: float
    annotation_area: float


class MetricsQueryParams(BaseModel):
    """Query parameters for metrics endpoints."""

    model: str = "dinov2"
    layer: str = "layer0"  # Safe default for all models (some have only 4 layers)
    percentile: int = 90


class LeaderboardEntry(BaseModel):
    """Model ranking entry."""

    rank: int
    model: str
    best_iou: float
    best_layer: str


class LayerProgressionSchema(BaseModel):
    """IoU progression across layers."""

    model: str
    percentile: int
    layers: list[str]
    ious: list[float]
    best_layer: str
    best_iou: float


class StyleBreakdownSchema(BaseModel):
    """IoU breakdown by architectural style."""

    model: str
    layer: str
    percentile: int
    styles: dict[str, float]  # style_name -> mean_iou
    style_counts: dict[str, int]  # style_name -> num_images


class ModelComparisonSchema(BaseModel):
    """Comparison data for multiple models."""

    image_id: str
    models: list[str]
    layer: str
    percentile: int
    results: list[IoUResultSchema]
    heatmap_urls: dict[str, str]  # model -> heatmap URL


class BboxInput(BaseModel):
    """Input for similarity computation - a single bounding box."""

    left: float = Field(..., ge=0, le=1, description="Left edge (0-1)")
    top: float = Field(..., ge=0, le=1, description="Top edge (0-1)")
    width: float = Field(..., ge=0, le=1, description="Width (0-1)")
    height: float = Field(..., ge=0, le=1, description="Height (0-1)")
    label: str | None = Field(None, description="Optional label for the feature")


class SimilarityResponse(BaseModel):
    """Response containing cosine similarity values for all patches."""

    similarity: list[float] = Field(..., description="Similarity values for each patch")
    patch_grid: list[int] = Field(..., description="Grid dimensions [rows, cols]")
    min_similarity: float = Field(..., description="Minimum similarity value")
    max_similarity: float = Field(..., description="Maximum similarity value")
    bbox_patch_indices: list[int] = Field(
        ..., description="Indices of patches within the bbox"
    )


class FeatureIoUEntry(BaseModel):
    """IoU metrics for a single architectural feature type."""

    feature_label: int = Field(..., description="Feature type index (0-105)")
    feature_name: str = Field(..., description="Human-readable feature name")
    mean_iou: float = Field(..., description="Mean IoU across all bboxes of this type")
    std_iou: float = Field(..., description="Standard deviation of IoU")
    bbox_count: int = Field(..., description="Number of bboxes of this type")


class FeatureBreakdownSchema(BaseModel):
    """IoU breakdown by architectural feature type."""

    model: str
    layer: str
    percentile: int
    features: list[FeatureIoUEntry]
    total_feature_types: int = Field(..., description="Total number of feature types returned")


class RawAttentionResponse(BaseModel):
    """Raw attention values for client-side rendering."""

    attention: list[float] = Field(..., description="Flattened attention values (row-major order)")
    shape: list[int] = Field(..., description="Grid dimensions [rows, cols]")
    min_value: float = Field(..., description="Minimum attention value")
    max_value: float = Field(..., description="Maximum attention value")

# API Reference

Complete REST API documentation for the SSL Attention Visualization platform. All endpoints are served under the `/api` prefix by the FastAPI backend.

> **Interactive docs**: When the backend is running, visit `/docs` for the auto-generated Swagger UI or `/redoc` for ReDoc.

---

## Quick Reference

| Method | Path | Description |
|--------|------|-------------|
| GET | `/api/images` | List images with style filter and pagination |
| GET | `/api/images/styles` | List available architectural styles |
| GET | `/api/images/{image_id}` | Get image detail with annotations |
| GET | `/api/images/{image_id}/file` | Serve original image (optional resize) |
| GET | `/api/images/{image_id}/thumbnail` | Serve thumbnail |
| GET | `/api/images/{image_id}/with_bboxes` | Serve image with bounding box overlay |
| GET | `/api/attention/models` | List model configs, methods, layer counts |
| GET | `/api/attention/{image_id}/heatmap` | Pure attention heatmap PNG |
| GET | `/api/attention/{image_id}/overlay` | Heatmap overlaid on original image |
| GET | `/api/attention/{image_id}/raw` | Raw attention values for client rendering |
| GET | `/api/attention/{image_id}/layers` | All layer overlay URLs for a model |
| POST | `/api/attention/{image_id}/similarity` | Bbox cosine similarity across patches |
| GET | `/api/metrics/leaderboard` | Model rankings by best IoU |
| GET | `/api/metrics/summary` | Pre-computed metrics summary |
| GET | `/api/metrics/{image_id}` | Per-image IoU metrics |
| GET | `/api/metrics/{image_id}/bbox/{bbox_index}` | Per-bbox IoU metrics (computed on-the-fly) |
| GET | `/api/metrics/{image_id}/all_models` | Per-image metrics across all models |
| GET | `/api/metrics/model/{model}/progression` | Layer-by-layer IoU progression |
| GET | `/api/metrics/model/{model}/style_breakdown` | IoU by architectural style |
| GET | `/api/metrics/model/{model}/feature_breakdown` | IoU by feature type |
| GET | `/api/metrics/model/{model}/aggregate` | Aggregate stats (mean, std, median) |
| GET | `/api/metrics/model/{model}/all_images` | All image metrics for a model |
| GET | `/api/compare/models` | Side-by-side model comparison |
| GET | `/api/compare/frozen_vs_finetuned` | Frozen vs fine-tuned comparison |
| GET | `/api/compare/layers` | Layer comparison with heatmap URLs |
| GET | `/api/compare/all_models_summary` | Full leaderboard with progressions |
| GET | `/health` | Health check with degraded-mode detection |

---

## Common Parameters

These parameters appear across multiple endpoints:

| Parameter | Type | Values | Default | Description |
|-----------|------|--------|---------|-------------|
| `model` | string | `dinov2`, `dinov3`, `mae`, `clip`, `siglip2`, `resnet50` | `dinov2` | Model name. `siglip2` resolves to canonical name `siglip` internally. |
| `layer` | int | 0–11 (ViTs), 0–3 (ResNet) | varies | Layer index (0-based). Range depends on model. |
| `percentile` | int | 50–95 | `90` | Attention threshold percentile for IoU computation. |
| `method` | string | `cls`, `rollout`, `mean`, `gradcam` | per-model | Attention extraction method. Availability depends on model (see table below). |

### Model / Method / Layer Matrix

| Model | Attention Methods | Default | Layers | Patch Size |
|-------|-------------------|---------|--------|------------|
| `dinov2` | `cls`, `rollout` | `cls` | 12 (0–11) | 14×14 |
| `dinov3` | `cls`, `rollout` | `cls` | 12 (0–11) | 16×16 |
| `mae` | `cls`, `rollout` | `cls` | 12 (0–11) | 16×16 |
| `clip` | `cls`, `rollout` | `cls` | 12 (0–11) | 16×16 |
| `siglip2` | `mean` | `mean` | 12 (0–11) | 16×16 |
| `resnet50` | `gradcam` | `gradcam` | 4 (0–3) | 32×32 |

---

## Images (`/api/images`)

### `GET /api/images`

List all annotated images with optional style filtering and pagination.

**Query Parameters**

| Parameter | Type | Required | Default | Constraints | Description |
|-----------|------|----------|---------|-------------|-------------|
| `style` | string | No | `null` | Must be a valid style name | Filter by architectural style |
| `limit` | int | No | `139` | 1–200 | Maximum number of results |
| `offset` | int | No | `0` | ≥ 0 | Pagination offset |

**Response**: `list[ImageListItem]`

```json
[
  {
    "image_id": "Q2270_0.jpg",
    "thumbnail_url": "/api/images/Q2270_0.jpg/thumbnail",
    "styles": ["Q46261"],
    "style_names": ["Romanesque"],
    "num_bboxes": 5
  }
]
```

---

### `GET /api/images/styles`

List all available architectural styles.

**Parameters**: None

**Response**: `list[str]`

```json
["Romanesque", "Gothic", "Renaissance", "Baroque"]
```

---

### `GET /api/images/{image_id}`

Get detailed information about an image including all bounding box annotations.

**Path Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `image_id` | string | Image filename (e.g., `Q2270_0.jpg`) |

**Response**: `ImageDetailSchema`

```json
{
  "image_id": "Q2270_0.jpg",
  "image_url": "/api/images/Q2270_0.jpg/file",
  "thumbnail_url": "/api/images/Q2270_0.jpg/thumbnail",
  "annotation": {
    "image_id": "Q2270_0.jpg",
    "styles": ["Q46261"],
    "style_names": ["Romanesque"],
    "num_bboxes": 5,
    "bboxes": [
      {
        "left": 0.12,
        "top": 0.34,
        "width": 0.15,
        "height": 0.20,
        "label": 42,
        "label_name": "window"
      }
    ]
  },
  "available_models": ["dinov2", "dinov3", "mae", "clip", "siglip2", "resnet50"]
}
```

**Errors**

| Status | Condition |
|--------|-----------|
| 404 | Image ID not found in annotations or image file missing |

---

### `GET /api/images/{image_id}/file`

Serve the original image as JPEG. Supports optional square resizing.

**Path Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `image_id` | string | Image filename |

**Query Parameters**

| Parameter | Type | Required | Default | Description |
|-----------|------|----------|---------|-------------|
| `size` | int | No | `null` | Resize to `size × size` pixels |

**Response**: `image/jpeg` stream (Cache-Control: 1 hour)

**Errors**

| Status | Condition |
|--------|-----------|
| 404 | Image file not found |

---

### `GET /api/images/{image_id}/thumbnail`

Serve a 128×128 thumbnail of the image.

**Path Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `image_id` | string | Image filename |

**Response**: `image/jpeg` stream (Cache-Control: 24 hours)

**Errors**

| Status | Condition |
|--------|-----------|
| 404 | Image file not found |

---

### `GET /api/images/{image_id}/with_bboxes`

Serve the original image with expert-annotated bounding boxes drawn as overlays.

**Path Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `image_id` | string | Image filename |

**Response**: `image/png` stream (Cache-Control: 24 hours)

**Errors**

| Status | Condition |
|--------|-----------|
| 404 | Bbox image not pre-computed (run `generate_heatmap_images.py` first) |

---

## Attention (`/api/attention`)

### `GET /api/attention/models`

List all available models with their configurations, supported attention methods, and layer counts.

**Parameters**: None

**Response**: `dict`

```json
{
  "models": ["dinov2", "dinov3", "mae", "clip", "siglip2", "resnet50"],
  "num_layers": 12,
  "num_layers_per_model": {
    "dinov2": 12, "dinov3": 12, "mae": 12,
    "clip": 12, "siglip2": 12, "resnet50": 4
  },
  "methods": {
    "dinov2": ["cls", "rollout"],
    "siglip2": ["mean"],
    "resnet50": ["gradcam"]
  },
  "default_methods": {
    "dinov2": "cls",
    "siglip2": "mean",
    "resnet50": "gradcam"
  }
}
```

---

### `GET /api/attention/{image_id}/heatmap`

Get the pure attention heatmap (colormap-rendered, no image underneath).

**Path Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `image_id` | string | Image filename |

**Query Parameters**

| Parameter | Type | Required | Default | Constraints | Description |
|-----------|------|----------|---------|-------------|-------------|
| `model` | string | No | `dinov2` | See [Common Parameters](#common-parameters) | Model name |
| `layer` | int | No | `11` | ≥ 0, within model range | Layer index |
| `method` | string | No | model default | Must be valid for model | Attention method |

**Response**: `image/png` stream (Cache-Control: 24 hours)

**Errors**

| Status | Condition |
|--------|-----------|
| 400 | Invalid model, layer, or method |
| 404 | Heatmap not pre-computed for this combination |

---

### `GET /api/attention/{image_id}/overlay`

Get the attention heatmap overlaid on the original image. Optionally includes bounding box annotations.

**Path Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `image_id` | string | Image filename |

**Query Parameters**

| Parameter | Type | Required | Default | Constraints | Description |
|-----------|------|----------|---------|-------------|-------------|
| `model` | string | No | `dinov2` | See [Common Parameters](#common-parameters) | Model name |
| `layer` | int | No | `11` | ≥ 0, within model range | Layer index |
| `method` | string | No | model default | Must be valid for model | Attention method |
| `show_bboxes` | bool | No | `false` | | Include bounding box annotations on overlay |

**Response**: `image/png` stream (Cache-Control: 24 hours)

**Errors**

| Status | Condition |
|--------|-----------|
| 400 | Invalid model, layer, or method |
| 404 | Overlay not pre-computed for this combination |

---

### `GET /api/attention/{image_id}/raw`

Get raw attention values as a flat numeric array for client-side rendering and dynamic percentile thresholding.

**Path Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `image_id` | string | Image filename |

**Query Parameters**

| Parameter | Type | Required | Default | Constraints | Description |
|-----------|------|----------|---------|-------------|-------------|
| `model` | string | No | `dinov2` | See [Common Parameters](#common-parameters) | Model name |
| `layer` | int | No | `11` | ≥ 0, within model range | Layer index |
| `method` | string | No | model default | Must be valid for model | Attention method |

**Response**: `RawAttentionResponse`

```json
{
  "attention": [0.001, 0.042, 0.087, "..."],
  "shape": [16, 16],
  "min_value": 0.0001,
  "max_value": 0.312
}
```

**Errors**

| Status | Condition |
|--------|-----------|
| 400 | Invalid model, layer, or method |
| 404 | Attention not cached (run `generate_attention_cache.py` first) |
| 500 | Error loading attention data |

---

### `GET /api/attention/{image_id}/layers`

Get overlay URLs for all layers of a model on a given image. Used for layer progression animation.

**Path Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `image_id` | string | Image filename |

**Query Parameters**

| Parameter | Type | Required | Default | Constraints | Description |
|-----------|------|----------|---------|-------------|-------------|
| `model` | string | No | `dinov2` | See [Common Parameters](#common-parameters) | Model name |
| `method` | string | No | model default | Must be valid for model | Attention method |
| `show_bboxes` | bool | No | `false` | | Include bounding boxes in overlays |

**Response**: `dict`

```json
{
  "image_id": "Q2270_0.jpg",
  "model": "dinov2",
  "method": "cls",
  "show_bboxes": false,
  "layers": {
    "layer0": "/api/attention/Q2270_0.jpg/overlay?model=dinov2&layer=0&method=cls&show_bboxes=False",
    "layer1": "/api/attention/Q2270_0.jpg/overlay?model=dinov2&layer=1&method=cls&show_bboxes=False",
    "...": "..."
  }
}
```

**Errors**

| Status | Condition |
|--------|-----------|
| 400 | Invalid model or method |
| 404 | No layers pre-computed for this model/method/image |

---

### `POST /api/attention/{image_id}/similarity`

Compute cosine similarity between a bounding box region and all image patches. Enables interactive exploration of which regions share similar learned features with a selected architectural element.

**Path Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `image_id` | string | Image filename |

**Query Parameters**

| Parameter | Type | Required | Default | Constraints | Description |
|-----------|------|----------|---------|-------------|-------------|
| `model` | string | No | `dinov2` | See [Common Parameters](#common-parameters) | Model name |
| `layer` | int | No | `11` | ≥ 0, within model range | Layer index |

**Request Body**: `BboxInput`

```json
{
  "left": 0.12,
  "top": 0.34,
  "width": 0.15,
  "height": 0.20,
  "label": "window"
}
```

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `left` | float | Yes | 0–1 | Left edge (normalized) |
| `top` | float | Yes | 0–1 | Top edge (normalized) |
| `width` | float | Yes | 0–1 | Width (normalized) |
| `height` | float | Yes | 0–1 | Height (normalized) |
| `label` | string | No | | Optional feature label |

**Response**: `SimilarityResponse`

```json
{
  "similarity": [0.12, 0.45, 0.89, "..."],
  "patch_grid": [16, 16],
  "min_similarity": -0.15,
  "max_similarity": 0.95,
  "bbox_patch_indices": [34, 35, 50, 51]
}
```

**Errors**

| Status | Condition |
|--------|-----------|
| 400 | Invalid model, layer, or bbox coordinates |
| 404 | Features not pre-computed (run `generate_feature_cache.py` first) |
| 500 | Error computing similarity |

---

## Metrics (`/api/metrics`)

All metrics endpoints require the pre-computed metrics database (`metrics.db`). If unavailable, they return **503**.

### `GET /api/metrics/leaderboard`

Get model rankings by best IoU score at a given percentile threshold.

**Query Parameters**

| Parameter | Type | Required | Default | Constraints | Description |
|-----------|------|----------|---------|-------------|-------------|
| `percentile` | int | No | `90` | 50–95 | Attention threshold percentile |

**Response**: `list[LeaderboardEntry]`

```json
[
  {
    "rank": 1,
    "model": "dinov2",
    "best_iou": 0.342,
    "best_layer": "layer11"
  }
]
```

**Errors**

| Status | Condition |
|--------|-----------|
| 503 | Metrics database not available |

---

### `GET /api/metrics/summary`

Get the pre-computed metrics summary including leaderboard and per-model best layers.

**Parameters**: None

**Response**: `dict` (structure defined by `metrics_summary.json`)

**Errors**

| Status | Condition |
|--------|-----------|
| 503 | Metrics summary file not available |

---

### `GET /api/metrics/{image_id}`

Get IoU metrics for a specific image with a given model/layer/percentile.

**Path Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `image_id` | string | Image filename |

**Query Parameters**

| Parameter | Type | Required | Default | Constraints | Description |
|-----------|------|----------|---------|-------------|-------------|
| `model` | string | No | `dinov2` | See [Common Parameters](#common-parameters) | Model name |
| `layer` | int | No | `0` | ≥ 0, within model range | Layer index |
| `percentile` | int | No | `90` | 50–95 | Attention threshold percentile |

**Response**: `IoUResultSchema`

```json
{
  "image_id": "Q2270_0.jpg",
  "model": "dinov2",
  "layer": "layer0",
  "percentile": 90,
  "iou": 0.285,
  "coverage": 0.72,
  "attention_area": 0.15,
  "annotation_area": 0.21
}
```

**Errors**

| Status | Condition |
|--------|-----------|
| 400 | Invalid model or layer |
| 404 | Metrics not found for this image/model/layer combination |
| 503 | Metrics database not available |

---

### `GET /api/metrics/{image_id}/bbox/{bbox_index}`

Get IoU metrics for a specific bounding box (rather than the union of all bboxes). Computed on-the-fly from the attention cache — not pre-cached in the metrics database.

**Path Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `image_id` | string | Image filename |
| `bbox_index` | int | Zero-based bounding box index (0 to `num_bboxes - 1`) |

**Query Parameters**

| Parameter | Type | Required | Default | Constraints | Description |
|-----------|------|----------|---------|-------------|-------------|
| `model` | string | No | `dinov2` | See [Common Parameters](#common-parameters) | Model name |
| `layer` | int | No | `0` | ≥ 0, within model range | Layer index |
| `percentile` | int | No | `90` | 50–95 | Attention threshold percentile |
| `method` | string | No | model default | Must be valid for model | Attention method |

**Response**: `IoUResultSchema`

```json
{
  "image_id": "Q2270_0.jpg",
  "model": "dinov2",
  "layer": "layer0",
  "percentile": 90,
  "iou": 0.185,
  "coverage": 0.42,
  "attention_area": 0.10,
  "annotation_area": 0.04
}
```

**Errors**

| Status | Condition |
|--------|-----------|
| 400 | Invalid model, layer, method, or `bbox_index` out of range |
| 404 | Annotation not found for image, or attention not cached |

---

### `GET /api/metrics/{image_id}/all_models`

Get metrics for a single image across all models. Models that don't support the requested layer *or* the requested method are silently skipped (unlike `/api/compare/models`, which returns 400 on incompatibility).

**Path Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `image_id` | string | Image filename |

**Query Parameters**

| Parameter | Type | Required | Default | Constraints | Description |
|-----------|------|----------|---------|-------------|-------------|
| `layer` | int | No | `0` | ≥ 0 | Layer index (models without this layer are skipped) |
| `percentile` | int | No | `90` | 50–95 | Attention threshold percentile |

**Response**: `dict`

```json
{
  "image_id": "Q2270_0.jpg",
  "layer": "layer0",
  "percentile": 90,
  "models": {
    "dinov2": { "image_id": "...", "model": "dinov2", "layer": "layer0", "iou": 0.285, "..." : "..." },
    "clip": { "..." : "..." }
  }
}
```

**Errors**

| Status | Condition |
|--------|-----------|
| 404 | No metrics found for this image at all |
| 503 | Metrics database not available |

---

### `GET /api/metrics/model/{model}/progression`

Get IoU progression across all layers for a model. Shows how attention alignment evolves through transformer depth.

**Path Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | string | Model name |

**Query Parameters**

| Parameter | Type | Required | Default | Constraints | Description |
|-----------|------|----------|---------|-------------|-------------|
| `percentile` | int | No | `90` | 50–95 | Attention threshold percentile |

**Response**: `LayerProgressionSchema`

```json
{
  "model": "dinov2",
  "percentile": 90,
  "layers": ["layer0", "layer1", "...", "layer11"],
  "ious": [0.12, 0.15, "...", 0.34],
  "best_layer": "layer11",
  "best_iou": 0.342
}
```

**Errors**

| Status | Condition |
|--------|-----------|
| 400 | Invalid model name |
| 503 | Metrics database not available |

---

### `GET /api/metrics/model/{model}/style_breakdown`

Get IoU breakdown by architectural style (Romanesque, Gothic, Renaissance, Baroque).

**Path Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | string | Model name |

**Query Parameters**

| Parameter | Type | Required | Default | Constraints | Description |
|-----------|------|----------|---------|-------------|-------------|
| `layer` | int | No | `0` | ≥ 0, within model range | Layer index |
| `percentile` | int | No | `90` | 50–95 | Attention threshold percentile |

**Response**: `StyleBreakdownSchema`

```json
{
  "model": "dinov2",
  "layer": "layer0",
  "percentile": 90,
  "styles": {
    "Romanesque": 0.31,
    "Gothic": 0.28,
    "Renaissance": 0.25,
    "Baroque": 0.22
  },
  "style_counts": {
    "Romanesque": 54,
    "Gothic": 49,
    "Renaissance": 22,
    "Baroque": 17
  }
}
```

**Errors**

| Status | Condition |
|--------|-----------|
| 400 | Invalid model or layer |
| 503 | Metrics database not available |

---

### `GET /api/metrics/model/{model}/feature_breakdown`

Get IoU breakdown by architectural feature type (e.g., windows, doors, arches) across all 106 feature types.

**Path Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | string | Model name |

**Query Parameters**

| Parameter | Type | Required | Default | Constraints | Description |
|-----------|------|----------|---------|-------------|-------------|
| `layer` | int | No | `0` | ≥ 0, within model range | Layer index |
| `percentile` | int | No | `90` | 50–95 | Attention threshold percentile |
| `min_count` | int | No | `0` | ≥ 0 | Minimum bbox count to include a feature type |
| `sort_by` | string | No | `mean_iou` | `mean_iou`, `bbox_count`, `feature_name`, `feature_label` | Sort order |

**Response**: `FeatureBreakdownSchema`

```json
{
  "model": "dinov2",
  "layer": "layer0",
  "percentile": 90,
  "features": [
    {
      "feature_label": 42,
      "feature_name": "window",
      "mean_iou": 0.35,
      "std_iou": 0.12,
      "bbox_count": 87
    }
  ],
  "total_feature_types": 106
}
```

**Errors**

| Status | Condition |
|--------|-----------|
| 400 | Invalid model or layer |
| 503 | Metrics database not available |

---

### `GET /api/metrics/model/{model}/aggregate`

Get aggregate statistics (mean, std, median IoU) for a model/layer combination across all images.

**Path Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | string | Model name |

**Query Parameters**

| Parameter | Type | Required | Default | Constraints | Description |
|-----------|------|----------|---------|-------------|-------------|
| `layer` | int | No | `0` | ≥ 0, within model range | Layer index |
| `percentile` | int | No | `90` | 50–95 | Attention threshold percentile |

**Response**: `dict` (keys: `mean_iou`, `std_iou`, `median_iou`, etc.)

**Errors**

| Status | Condition |
|--------|-----------|
| 400 | Invalid model or layer |
| 404 | Aggregate metrics not found for this model/layer |
| 503 | Metrics database not available |

---

### `GET /api/metrics/model/{model}/all_images`

Get metrics for all images for a given model/layer, with sorting and pagination.

**Path Parameters**

| Parameter | Type | Description |
|-----------|------|-------------|
| `model` | string | Model name |

**Query Parameters**

| Parameter | Type | Required | Default | Constraints | Description |
|-----------|------|----------|---------|-------------|-------------|
| `layer` | int | No | `0` | ≥ 0, within model range | Layer index |
| `percentile` | int | No | `90` | 50–95 | Attention threshold percentile |
| `sort_by` | string | No | `iou` | `iou`, `coverage` | Sort order |
| `limit` | int | No | `139` | 1–200 | Maximum results |

**Response**: `dict`

```json
{
  "model": "dinov2",
  "layer": "layer0",
  "percentile": 90,
  "count": 139,
  "images": [
    {
      "image_id": "Q2270_0.jpg",
      "iou": 0.42,
      "coverage": 0.85,
      "attention_area": 0.18,
      "annotation_area": 0.22
    }
  ]
}
```

**Errors**

| Status | Condition |
|--------|-----------|
| 400 | Invalid model or layer |
| 503 | Metrics database not available |

---

## Comparison (`/api/compare`)

### `GET /api/compare/models`

Compare multiple models side-by-side on a single image. Returns IoU metrics and heatmap URLs for each model.

> **Method validation**: When a `method` is specified, it is validated against *each* selected model. If any model does not support the requested method, the endpoint returns **400** (strict validation). Use `/api/metrics/{image_id}/all_models` if you want incompatible models to be skipped instead.

**Query Parameters**

| Parameter | Type | Required | Default | Constraints | Description |
|-----------|------|----------|---------|-------------|-------------|
| `image_id` | string | **Yes** | — | Must exist in dataset | Image filename |
| `models` | list[string] | No | `["dinov2", "clip"]` | Each must be valid model | Models to compare (repeat param for multiple) |
| `layer` | int | No | `0` | ≥ 0, within range for all models | Layer index |
| `percentile` | int | No | `90` | 50–95 | Attention threshold percentile |
| `method` | string | No | per-model default | Must be valid for *all* selected models | Attention method |

**Response**: `ModelComparisonSchema`

```json
{
  "image_id": "Q2270_0.jpg",
  "models": ["dinov2", "clip"],
  "layer": "layer0",
  "percentile": 90,
  "results": [
    {
      "image_id": "Q2270_0.jpg",
      "model": "dinov2",
      "layer": "layer0",
      "percentile": 90,
      "iou": 0.285,
      "coverage": 0.72,
      "attention_area": 0.15,
      "annotation_area": 0.21
    }
  ],
  "heatmap_urls": {
    "dinov2": "/api/attention/Q2270_0.jpg/overlay?model=dinov2&layer=0",
    "clip": "/api/attention/Q2270_0.jpg/overlay?model=clip&layer=0"
  }
}
```

**Errors**

| Status | Condition |
|--------|-----------|
| 400 | Invalid model name(s) or layer out of range |
| 404 | Image not found or no metrics available |
| 503 | Metrics database not available |

---

### `GET /api/compare/frozen_vs_finetuned`

Compare frozen (pretrained) vs fine-tuned model attention on a single image.

> **Note**: Fine-tuned models are not yet available. The `finetuned` section returns availability status and a placeholder note.

**Query Parameters**

| Parameter | Type | Required | Default | Constraints | Description |
|-----------|------|----------|---------|-------------|-------------|
| `image_id` | string | **Yes** | — | Must exist in dataset | Image filename |
| `model` | string | No | `dinov2` | See [Common Parameters](#common-parameters) | Model name |
| `layer` | int | No | `0` | ≥ 0, within model range | Layer index |

**Response**: `dict`

```json
{
  "image_id": "Q2270_0.jpg",
  "model": "dinov2",
  "layer": "layer0",
  "frozen": {
    "available": true,
    "url": "/api/attention/Q2270_0.jpg/overlay?model=dinov2&layer=0"
  },
  "finetuned": {
    "available": false,
    "url": null,
    "note": "Fine-tuned models will be available after Phase 5 training"
  }
}
```

**Errors**

| Status | Condition |
|--------|-----------|
| 400 | Invalid model or layer |
| 404 | Image not found |

---

### `GET /api/compare/layers`

Get IoU progression across all layers for a single model/image, with heatmap URLs for each layer. Used for layer comparison views and animations.

**Query Parameters**

| Parameter | Type | Required | Default | Constraints | Description |
|-----------|------|----------|---------|-------------|-------------|
| `image_id` | string | **Yes** | — | Must exist in dataset | Image filename |
| `model` | string | No | `dinov2` | See [Common Parameters](#common-parameters) | Model name |
| `percentile` | int | No | `90` | 50–95 | Attention threshold percentile |

**Response**: `dict`

```json
{
  "image_id": "Q2270_0.jpg",
  "model": "dinov2",
  "percentile": 90,
  "layers": [
    {
      "layer": 0,
      "layer_key": "layer0",
      "iou": 0.12,
      "coverage": 0.45,
      "heatmap_url": "/api/attention/Q2270_0.jpg/overlay?model=dinov2&layer=0"
    }
  ],
  "best_layer": 11,
  "best_iou": 0.342
}
```

**Errors**

| Status | Condition |
|--------|-----------|
| 400 | Invalid model |
| 404 | Image not found or no metrics available |
| 503 | Metrics database not available |

---

### `GET /api/compare/all_models_summary`

Get a full comparison summary of all models: leaderboard rankings plus per-layer IoU progressions.

**Query Parameters**

| Parameter | Type | Required | Default | Constraints | Description |
|-----------|------|----------|---------|-------------|-------------|
| `percentile` | int | No | `90` | 50–95 | Attention threshold percentile |

**Response**: `dict`

```json
{
  "percentile": 90,
  "models": {
    "dinov2": {
      "rank": 1,
      "best_iou": 0.342,
      "best_layer": "layer11",
      "layer_progression": {
        "layer0": 0.12,
        "layer11": 0.34
      }
    }
  },
  "leaderboard": [
    { "rank": 1, "model": "dinov2", "best_iou": 0.342, "best_layer": "layer11" }
  ]
}
```

**Errors**

| Status | Condition |
|--------|-----------|
| 503 | Metrics database not available |

---

## Health & Operations

### `GET /health`

Health check endpoint with degraded-mode detection. Reports the availability of critical backend resources.

> **Note**: This endpoint is at `/health` (not under `/api`).

**Parameters**: None

**Response**: `dict`

```json
{
  "status": "healthy",
  "checks": {
    "annotations_loaded": true,
    "metrics_db_available": true,
    "attention_cache_available": true
  }
}
```

| Field | Type | Description |
|-------|------|-------------|
| `status` | string | `"healthy"` if all checks pass, `"degraded"` if any check fails |
| `checks.annotations_loaded` | bool | Whether `building_parts.json` was parsed and images are available |
| `checks.metrics_db_available` | bool | Whether `metrics.db` exists and is accessible |
| `checks.attention_cache_available` | bool | Whether the HDF5 attention cache exists |

**Startup Validation**: The FastAPI lifespan hook validates four resources on startup — annotations file, attention cache, metrics database, and heatmaps directory. If any are missing, the server starts in **degraded mode** and logs a warning. Endpoints dependent on missing resources return 503 or 404 as appropriate.

---

## Response Schemas

All schemas are defined as Pydantic models in `app/backend/schemas/models.py`.

### `ImageListItem`

| Field | Type | Description |
|-------|------|-------------|
| `image_id` | string | Image filename |
| `thumbnail_url` | string | URL to thumbnail endpoint |
| `styles` | list[string] | Wikidata Q-IDs for architectural styles |
| `style_names` | list[string] | Human-readable style names |
| `num_bboxes` | int | Number of bounding box annotations |

### `ImageDetailSchema`

| Field | Type | Description |
|-------|------|-------------|
| `image_id` | string | Image filename |
| `image_url` | string | URL to full image endpoint |
| `thumbnail_url` | string | URL to thumbnail endpoint |
| `annotation` | ImageAnnotationSchema | Full annotation data |
| `available_models` | list[string] | List of model names |

### `ImageAnnotationSchema`

| Field | Type | Description |
|-------|------|-------------|
| `image_id` | string | Image filename |
| `styles` | list[string] | Wikidata Q-IDs |
| `style_names` | list[string] | Human-readable style names |
| `num_bboxes` | int | Total bounding boxes |
| `bboxes` | list[BoundingBoxSchema] | All bounding box annotations |

### `BoundingBoxSchema`

| Field | Type | Constraints | Description |
|-------|------|-------------|-------------|
| `left` | float | 0–1 | Left edge (normalized) |
| `top` | float | 0–1 | Top edge (normalized) |
| `width` | float | 0–1 | Width (normalized) |
| `height` | float | 0–1 | Height (normalized) |
| `label` | int | ≥ 0 | Feature type index (0–105) |
| `label_name` | string \| null | | Human-readable feature name |

### `IoUResultSchema`

| Field | Type | Description |
|-------|------|-------------|
| `image_id` | string | Image filename |
| `model` | string | Model name |
| `layer` | string | Layer key (e.g., `layer11`) |
| `percentile` | int | Percentile threshold used |
| `iou` | float | Intersection over Union score |
| `coverage` | float | Fraction of annotation area covered by attention |
| `attention_area` | float | Fraction of image area above attention threshold |
| `annotation_area` | float | Fraction of image area covered by annotations |

### `RawAttentionResponse`

| Field | Type | Description |
|-------|------|-------------|
| `attention` | list[float] | Flattened attention values (row-major order) |
| `shape` | list[int] | Grid dimensions `[rows, cols]` |
| `min_value` | float | Minimum attention value |
| `max_value` | float | Maximum attention value |

### `SimilarityResponse`

| Field | Type | Description |
|-------|------|-------------|
| `similarity` | list[float] | Cosine similarity for each patch |
| `patch_grid` | list[int] | Grid dimensions `[rows, cols]` |
| `min_similarity` | float | Minimum similarity value |
| `max_similarity` | float | Maximum similarity value |
| `bbox_patch_indices` | list[int] | Indices of patches within the query bbox |

### `LeaderboardEntry`

| Field | Type | Description |
|-------|------|-------------|
| `rank` | int | Position in ranking |
| `model` | string | Model name |
| `best_iou` | float | Best IoU achieved |
| `best_layer` | string | Layer with best IoU |

### `LayerProgressionSchema`

| Field | Type | Description |
|-------|------|-------------|
| `model` | string | Model name |
| `percentile` | int | Percentile threshold used |
| `layers` | list[string] | Layer keys in order |
| `ious` | list[float] | IoU values per layer |
| `best_layer` | string | Layer key with highest IoU |
| `best_iou` | float | Highest IoU value |

### `StyleBreakdownSchema`

| Field | Type | Description |
|-------|------|-------------|
| `model` | string | Model name |
| `layer` | string | Layer key |
| `percentile` | int | Percentile threshold used |
| `styles` | dict[string, float] | Style name → mean IoU |
| `style_counts` | dict[string, int] | Style name → image count |

### `FeatureBreakdownSchema`

| Field | Type | Description |
|-------|------|-------------|
| `model` | string | Model name |
| `layer` | string | Layer key |
| `percentile` | int | Percentile threshold used |
| `features` | list[FeatureIoUEntry] | Per-feature IoU data |
| `total_feature_types` | int | Total feature types returned |

### `FeatureIoUEntry`

| Field | Type | Description |
|-------|------|-------------|
| `feature_label` | int | Feature type index (0–105) |
| `feature_name` | string | Human-readable feature name |
| `mean_iou` | float | Mean IoU across all bboxes of this type |
| `std_iou` | float | Standard deviation of IoU |
| `bbox_count` | int | Number of bboxes of this type |

### `ModelComparisonSchema`

| Field | Type | Description |
|-------|------|-------------|
| `image_id` | string | Image filename |
| `models` | list[string] | Models compared |
| `layer` | string | Layer key |
| `percentile` | int | Percentile threshold used |
| `results` | list[IoUResultSchema] | Per-model IoU results |
| `heatmap_urls` | dict[string, string] | Model name → heatmap overlay URL |

### `BboxInput`

| Field | Type | Required | Constraints | Description |
|-------|------|----------|-------------|-------------|
| `left` | float | Yes | 0–1 | Left edge (normalized) |
| `top` | float | Yes | 0–1 | Top edge (normalized) |
| `width` | float | Yes | 0–1 | Width (normalized) |
| `height` | float | Yes | 0–1 | Height (normalized) |
| `label` | string \| null | No | | Optional feature label |

---

## Error Handling

All errors follow the standard FastAPI error format:

```json
{
  "detail": "Human-readable error message"
}
```

### Common Error Codes

| Status | Meaning | Common Causes |
|--------|---------|---------------|
| **400** | Bad Request | Invalid model name, layer out of range for model, invalid attention method for model |
| **404** | Not Found | Image ID not in dataset, heatmap/attention/features not pre-computed, metrics not found for combination |
| **500** | Internal Error | Unexpected error during attention loading or similarity computation |
| **503** | Service Unavailable | Metrics database (`metrics.db`) or summary file not generated |

### Centralized Validation

All parameter validation is handled by `app/backend/validators.py`, which provides reusable helpers used across all routers:

| Function | Purpose |
|----------|---------|
| `validate_model(model)` | Resolves aliases (e.g., `siglip2` → `siglip`) and returns 400 if invalid |
| `validate_method(model, method)` | Validates that the method is available for the model; returns 400 if not |
| `validate_layer_for_model(layer, model)` | Checks layer bounds per model; returns 400 if out of range |
| `resolve_default_method(model)` | Returns the default attention method for a model |

### Pre-computation Dependencies

Several endpoints require offline pre-computation before they can serve data:

| Script | Generates | Required By |
|--------|-----------|-------------|
| `generate_heatmap_images.py` | PNG heatmaps/overlays | `/attention/{id}/heatmap`, `/attention/{id}/overlay`, `/attention/{id}/layers`, `/images/{id}/with_bboxes` |
| `generate_attention_cache.py` | HDF5 attention cache | `/attention/{id}/raw` |
| `generate_feature_cache.py` | HDF5 feature cache | `/attention/{id}/similarity` |
| `generate_metrics_cache.py` | SQLite metrics DB + JSON summary | All `/metrics/*` and `/compare/*` endpoints |

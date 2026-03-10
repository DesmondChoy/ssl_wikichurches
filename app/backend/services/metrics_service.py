"""Service for querying pre-computed metrics."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any, Literal

from app.backend.config import (
    DEFAULT_METHOD,
    METRICS_DB_PATH,
    METRICS_SUMMARY_PATH,
    display_model_name,
    get_model_num_layers,
    resolve_model_name,
)
from app.backend.validators import resolve_default_method

if TYPE_CHECKING:
    from collections.abc import Generator


MetricName = Literal["iou", "mse"]

IMAGE_DETAIL_METRICS = (
    {
        "key": "iou",
        "label": "IoU Score",
        "direction": "higher",
        "default_enabled": True,
        "percentile_dependent": True,
    },
    {
        "key": "coverage",
        "label": "Coverage",
        "direction": "higher",
        "default_enabled": True,
        "percentile_dependent": False,
    },
    {
        "key": "mse",
        "label": "MSE",
        "direction": "lower",
        "default_enabled": True,
        "percentile_dependent": False,
    },
)


class MetricsService:
    """Service for querying pre-computed metrics from SQLite."""

    _instance: MetricsService | None = None

    def __new__(cls) -> MetricsService:
        """Singleton pattern."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    @property
    def db_path(self) -> Path:
        """Path to metrics database."""
        return METRICS_DB_PATH

    @property
    def db_exists(self) -> bool:
        """Check if metrics database exists."""
        return self.db_path.exists()

    @contextmanager
    def get_connection(self) -> Generator[sqlite3.Connection, None, None]:
        """Get database connection context manager."""
        if not self.db_exists:
            raise FileNotFoundError(f"Metrics database not found: {self.db_path}")

        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        try:
            yield conn
        finally:
            conn.close()

    def get_image_metrics(
        self,
        image_id: str,
        model: str,
        layer: str,
        percentile: int = 90,
        method: str | None = None,
    ) -> dict | None:
        """Get metrics for a specific image/model/layer combination.

        Returns:
            Dict with iou, coverage, mse, attention_area, annotation_area or None.
        """
        db_model = resolve_model_name(model)
        resolved_method = method if method else resolve_default_method(model)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT iou, coverage, mse, attention_area, annotation_area
                   FROM image_metrics
                   WHERE image_id = ? AND model = ? AND layer = ? AND method = ? AND percentile = ?""",
                (image_id, db_model, layer, resolved_method, percentile),
            )
            row = cursor.fetchone()

            if row:
                mse = row["mse"]
                if mse is None:
                    mse = self._compute_image_mse_from_cache(
                        image_id=image_id,
                        model=model,
                        layer=layer,
                        method=resolved_method,
                    )
                if mse is None:
                    return None

                return {
                    "image_id": image_id,
                    "model": model,  # Return original name for display
                    "layer": layer,
                    "percentile": percentile,
                    "method": resolved_method,
                    "iou": row["iou"],
                    "coverage": row["coverage"],
                    "mse": mse,
                    "attention_area": row["attention_area"],
                    "annotation_area": row["annotation_area"],
                }
            return None

    def get_image_layer_progression(
        self,
        image_id: str,
        model: str,
        percentile: int = 90,
        method: str | None = None,
    ) -> dict[str, Any] | None:
        """Get union-of-bboxes metric progression across all layers for one image."""
        db_model = resolve_model_name(model)
        resolved_method = method if method else resolve_default_method(model)
        layer_points = self._initialize_layer_points(model)

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT layer, iou, coverage, mse
                   FROM image_metrics
                   WHERE image_id = ? AND model = ? AND method = ? AND percentile = ?
                   ORDER BY CAST(SUBSTR(layer, 6) AS INTEGER)""",
                (image_id, db_model, resolved_method, percentile),
            )
            rows = cursor.fetchall()

        if not rows:
            return None

        has_values = False
        for row in rows:
            layer_key = row["layer"]
            if layer_key not in layer_points:
                continue

            mse = row["mse"]
            if mse is None:
                mse = self._compute_image_mse_from_cache(
                    image_id=image_id,
                    model=model,
                    layer=layer_key,
                    method=resolved_method,
                )

            values = layer_points[layer_key]["values"]
            values["iou"] = row["iou"]
            values["coverage"] = row["coverage"]
            values["mse"] = mse
            has_values = has_values or any(value is not None for value in values.values())

        if not has_values:
            return None

        return self._build_image_layer_progression_response(
            image_id=image_id,
            model=model,
            method=resolved_method,
            percentile=percentile,
            selection={
                "mode": "union",
                "bbox_index": None,
                "bbox_label": None,
            },
            layer_points=layer_points,
        )

    def get_bbox_layer_progression(
        self,
        image_id: str,
        model: str,
        bbox_index: int,
        percentile: int = 90,
        method: str | None = None,
    ) -> dict[str, Any] | None:
        """Get bbox-specific metric progression across all layers for one image."""
        annotation = self._get_annotation(image_id)
        if annotation is None:
            return None

        resolved_method = method if method else resolve_default_method(model)
        bbox_label = self._get_bbox_label(annotation, bbox_index)
        layer_points = self._initialize_layer_points(model)
        has_values = False

        for layer_index in range(get_model_num_layers(model)):
            layer_key = f"layer{layer_index}"
            metrics = self._compute_bbox_metrics(
                image_id=image_id,
                model=model,
                layer=layer_key,
                bbox_index=bbox_index,
                percentile=percentile,
                method=resolved_method,
                annotation=annotation,
            )
            if metrics is None:
                continue

            values = layer_points[layer_key]["values"]
            values["iou"] = metrics["iou"]
            values["coverage"] = metrics["coverage"]
            values["mse"] = metrics["mse"]
            has_values = True

        if not has_values:
            return None

        return self._build_image_layer_progression_response(
            image_id=image_id,
            model=model,
            method=resolved_method,
            percentile=percentile,
            selection={
                "mode": "bbox",
                "bbox_index": bbox_index,
                "bbox_label": bbox_label,
            },
            layer_points=layer_points,
        )

    def get_bbox_metrics(
        self,
        image_id: str,
        model: str,
        layer: str,
        bbox_index: int,
        percentile: int = 90,
        method: str | None = None,
    ) -> dict[str, Any] | None:
        """Get bbox-specific metrics for a single layer."""
        annotation = self._get_annotation(image_id)
        if annotation is None:
            return None

        return self._compute_bbox_metrics(
            image_id=image_id,
            model=model,
            layer=layer,
            bbox_index=bbox_index,
            percentile=percentile,
            method=method,
            annotation=annotation,
        )

    def get_aggregate_metrics(
        self,
        model: str,
        layer: str,
        percentile: int = 90,
        method: str | None = None,
    ) -> dict | None:
        """Get aggregate metrics for a model/layer.

        Returns:
            Dict with mean_iou, std_iou, median_iou, mean_coverage, mean_mse,
            std_mse, median_mse, num_images.
        """
        db_model = resolve_model_name(model)
        resolved_method = method if method else resolve_default_method(model)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT mean_iou, std_iou, median_iou, mean_coverage,
                          mean_mse, std_mse, median_mse, num_images
                   FROM aggregate_metrics
                   WHERE model = ? AND layer = ? AND method = ? AND percentile = ?""",
                (db_model, layer, resolved_method, percentile),
            )
            row = cursor.fetchone()

            if row:
                mean_mse = row["mean_mse"]
                std_mse = row["std_mse"]
                median_mse = row["median_mse"]
                if mean_mse is None or std_mse is None or median_mse is None:
                    mean_mse, std_mse, median_mse = self._compute_mse_aggregate_from_images(
                        model=model,
                        layer=layer,
                        percentile=percentile,
                        method=resolved_method,
                    )

                return {
                    "model": model,  # Return original name for display
                    "layer": layer,
                    "percentile": percentile,
                    "method": resolved_method,
                    "mean_iou": row["mean_iou"],
                    "std_iou": row["std_iou"],
                    "median_iou": row["median_iou"],
                    "mean_coverage": row["mean_coverage"],
                    "mean_mse": mean_mse,
                    "std_mse": std_mse,
                    "median_mse": median_mse,
                    "num_images": row["num_images"],
                }
            return None

    def _metric_sql_config(self, metric: MetricName) -> tuple[str, str, str]:
        """Resolve aggregate score column, SQL sort direction, and selector."""
        if metric == "mse":
            return "mean_mse", "ASC", "MIN"
        return "mean_iou", "DESC", "MAX"

    def get_leaderboard(self, percentile: int = 90, metric: MetricName = "iou") -> list[dict]:
        """Get model leaderboard ranked by best score for the selected metric.

        Uses each model's default method for ranking (CLS for ViTs, etc.).

        Returns:
            List of dicts with rank, model, metric, score, best_layer.
        """
        score_column, order_direction, aggregate_fn = self._metric_sql_config(metric)

        # Build a dynamic filter so each model uses its canonical default method
        conditions = []
        params: list[str | int] = []
        for model_name, default_method in DEFAULT_METHOD.items():
            conditions.append("(model = ? AND method = ?)")
            params.extend([model_name, default_method.value])

        if not conditions:
            return []

        filter_clause = " OR ".join(conditions)

        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""SELECT model, {aggregate_fn}({score_column}) as score,
                   (SELECT layer FROM aggregate_metrics a2
                    WHERE a2.model = a1.model AND a2.method = a1.method AND a2.percentile = ?
                      AND a2.{score_column} IS NOT NULL
                    ORDER BY {score_column} {order_direction},
                             CAST(SUBSTR(layer, 6) AS INTEGER) ASC
                    LIMIT 1) as best_layer
                   FROM aggregate_metrics a1
                   WHERE percentile = ? AND {score_column} IS NOT NULL AND ({filter_clause})
                   GROUP BY model
                   ORDER BY score {order_direction}, model ASC""",
                (percentile, percentile, *params),
            )

            return [
                {
                    "rank": i + 1,
                    "model": display_model_name(row["model"]),
                    "metric": metric,
                    "score": row["score"],
                    "best_layer": row["best_layer"],
                }
                for i, row in enumerate(cursor.fetchall())
            ]

    def get_layer_progression(
        self,
        model: str,
        percentile: int = 90,
        method: str | None = None,
        metric: MetricName = "iou",
    ) -> dict:
        """Get metric progression across all layers.

        Returns:
            Dict with model, metric, percentile, layers, scores, best_layer, best_score.
        """
        score_column, _, _ = self._metric_sql_config(metric)
        db_model = resolve_model_name(model)
        resolved_method = method if method else resolve_default_method(model)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                f"""SELECT layer, {score_column} AS score FROM aggregate_metrics
                   WHERE model = ? AND method = ? AND percentile = ?
                     AND {score_column} IS NOT NULL
                   ORDER BY CAST(SUBSTR(layer, 6) AS INTEGER)""",
                (db_model, resolved_method, percentile),
            )
            rows = cursor.fetchall()

            layers = [row["layer"] for row in rows]
            scores = [row["score"] for row in rows]

            best_idx = 0
            if scores:
                comparator = min if metric == "mse" else max
                best_idx = scores.index(comparator(scores))
            best_layer = layers[best_idx] if layers else "layer11"
            best_score = scores[best_idx] if scores else 0.0

            return {
                "model": model,  # Return original name for display
                "metric": metric,
                "percentile": percentile,
                "method": resolved_method,
                "layers": layers,
                "scores": scores,
                "best_layer": best_layer,
                "best_score": best_score,
            }

    def get_style_breakdown(
        self,
        model: str,
        layer: str,
        percentile: int = 90,
        method: str | None = None,
    ) -> dict:
        """Get IoU breakdown by architectural style.

        Returns:
            Dict with model, layer, percentile, styles (name->iou), style_counts.
        """
        db_model = resolve_model_name(model)
        resolved_method = method if method else resolve_default_method(model)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT style_name, mean_iou, num_images FROM style_metrics
                   WHERE model = ? AND layer = ? AND method = ? AND percentile = ?""",
                (db_model, layer, resolved_method, percentile),
            )

            styles = {}
            counts = {}
            for row in cursor.fetchall():
                styles[row["style_name"]] = row["mean_iou"]
                counts[row["style_name"]] = row["num_images"]

            return {
                "model": model,  # Return original name for display
                "layer": layer,
                "percentile": percentile,
                "method": resolved_method,
                "styles": styles,
                "style_counts": counts,
            }

    def get_all_image_metrics(
        self,
        model: str,
        layer: str,
        percentile: int = 90,
        method: str | None = None,
    ) -> list[dict]:
        """Get metrics for all images for a model/layer.

        Returns:
            List of dicts with image_id, iou, coverage, mse.
        """
        db_model = resolve_model_name(model)
        resolved_method = method if method else resolve_default_method(model)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT image_id, iou, coverage, mse, attention_area, annotation_area
                   FROM image_metrics
                   WHERE model = ? AND layer = ? AND method = ? AND percentile = ?
                   ORDER BY iou DESC""",
                (db_model, layer, resolved_method, percentile),
            )
            results = []
            for row in cursor.fetchall():
                mse = row["mse"]
                if mse is None:
                    mse = self._compute_image_mse_from_cache(
                        image_id=row["image_id"],
                        model=model,
                        layer=layer,
                        method=resolved_method,
                    )
                if mse is None:
                    continue

                results.append(
                    {
                        "image_id": row["image_id"],
                        "iou": row["iou"],
                        "coverage": row["coverage"],
                        "mse": mse,
                        "attention_area": row["attention_area"],
                        "annotation_area": row["annotation_area"],
                    }
                )

            return results

    def get_summary(self) -> dict[str, Any] | None:
        """Load pre-computed summary from JSON file."""
        if not METRICS_SUMMARY_PATH.exists():
            return None

        with open(METRICS_SUMMARY_PATH, encoding="utf-8") as f:
            data: dict[str, Any] = json.load(f)
            return data

    def get_feature_breakdown(
        self,
        model: str,
        layer: str,
        percentile: int = 90,
        sort_by: str = "mean_iou",
        min_count: int = 0,
        method: str | None = None,
    ) -> dict:
        """Get IoU breakdown by architectural feature type.

        Args:
            model: Model name.
            layer: Layer key (e.g., "layer11").
            percentile: Percentile threshold.
            sort_by: Field to sort by ("mean_iou", "bbox_count", "feature_name").
            min_count: Minimum bbox count to include a feature.
            method: Attention method. None = model default.

        Returns:
            Dict with model, layer, percentile, features list, total_feature_types.
        """
        db_model = resolve_model_name(model)
        resolved_method = method if method else resolve_default_method(model)
        with self.get_connection() as conn:
            cursor = conn.cursor()

            # Determine sort order
            order_clause = "mean_iou DESC"
            if sort_by == "bbox_count":
                order_clause = "bbox_count DESC"
            elif sort_by == "feature_name":
                order_clause = "feature_name ASC"
            elif sort_by == "feature_label":
                order_clause = "feature_label ASC"

            cursor.execute(
                f"""SELECT feature_label, feature_name, mean_iou, std_iou, bbox_count
                   FROM feature_metrics
                   WHERE model = ? AND layer = ? AND method = ? AND percentile = ? AND bbox_count >= ?
                   ORDER BY {order_clause}""",
                (db_model, layer, resolved_method, percentile, min_count),
            )

            features = [
                {
                    "feature_label": row["feature_label"],
                    "feature_name": row["feature_name"],
                    "mean_iou": row["mean_iou"],
                    "std_iou": row["std_iou"],
                    "bbox_count": row["bbox_count"],
                }
                for row in cursor.fetchall()
            ]

            return {
                "model": model,  # Return original name for display
                "layer": layer,
                "percentile": percentile,
                "method": resolved_method,
                "features": features,
                "total_feature_types": len(features),
            }

    def _compute_image_mse_from_cache(
        self,
        image_id: str,
        model: str,
        layer: str,
        method: str,
    ) -> float | None:
        """Compute MSE directly from cached attention for legacy DB rows."""
        from app.backend.services.attention_service import attention_service
        from app.backend.services.image_service import image_service
        from ssl_attention.metrics import compute_image_mse

        annotation = image_service.get_annotation(image_id)
        if annotation is None:
            return None

        cache_model = resolve_model_name(model)
        try:
            attention = attention_service.cache.load(cache_model, layer, image_id, variant=method)
        except KeyError:
            return None

        if attention.dim() == 1:
            grid_rows, grid_cols = attention_service.get_attention_grid(cache_model)
            attention = attention.reshape(grid_rows, grid_cols)

        return compute_image_mse(attention=attention, annotation=annotation)

    def _compute_bbox_metrics(
        self,
        image_id: str,
        model: str,
        layer: str,
        bbox_index: int,
        percentile: int,
        method: str | None,
        annotation: Any,
    ) -> dict[str, Any] | None:
        """Compute bbox metrics from cached attention for a single layer."""
        from ssl_attention.metrics.continuous import compute_mse, gaussian_bbox_heatmap
        from ssl_attention.metrics.iou import compute_coverage, compute_iou

        resolved_method = method if method else resolve_default_method(model)

        if not 0 <= bbox_index < len(annotation.bboxes):
            raise ValueError(
                f"bbox_index {bbox_index} out of range. "
                f"Image has {len(annotation.bboxes)} bboxes (0-{len(annotation.bboxes) - 1})."
            )

        attention_tensor = self._load_attention_tensor(
            model=model,
            layer=layer,
            image_id=image_id,
            method=resolved_method,
        )
        if attention_tensor is None:
            return None

        bbox = annotation.bboxes[bbox_index]
        height, width = attention_tensor.shape[-2:]
        bbox_mask = bbox.to_mask(height, width)
        bbox_heatmap = gaussian_bbox_heatmap(bbox, height, width, device=attention_tensor.device)
        iou, attention_area, annotation_area = compute_iou(attention_tensor, bbox_mask, percentile)
        coverage = compute_coverage(attention_tensor, bbox_mask)
        mse = compute_mse(attention_tensor, bbox_heatmap)

        return {
            "image_id": image_id,
            "model": model,
            "layer": layer,
            "percentile": percentile,
            "method": resolved_method,
            "iou": iou,
            "coverage": coverage,
            "mse": mse,
            "attention_area": attention_area,
            "annotation_area": annotation_area,
        }

    def _get_annotation(self, image_id: str) -> Any | None:
        """Load image annotation without introducing a module-level import cycle."""
        from app.backend.services.image_service import image_service

        return image_service.get_annotation(image_id)

    def _load_attention_tensor(
        self,
        model: str,
        layer: str,
        image_id: str,
        method: str,
    ) -> Any | None:
        """Load cached attention and normalize it to a 2D tensor."""
        from app.backend.services.attention_service import attention_service

        cache_model = resolve_model_name(model)
        try:
            attention = attention_service.cache.load(cache_model, layer, image_id, variant=method)
        except KeyError:
            return None

        if attention.dim() == 1:
            grid_rows, grid_cols = attention_service.get_attention_grid(cache_model)
            attention = attention.reshape(grid_rows, grid_cols)

        return attention

    def _get_bbox_label(self, annotation: Any, bbox_index: int) -> str:
        """Resolve a human-readable label for a bbox selection."""
        from app.backend.services.image_service import image_service

        if not 0 <= bbox_index < len(annotation.bboxes):
            raise ValueError(
                f"bbox_index {bbox_index} out of range. "
                f"Image has {len(annotation.bboxes)} bboxes (0-{len(annotation.bboxes) - 1})."
            )

        bbox = annotation.bboxes[bbox_index]
        return image_service.get_feature_name(bbox.label) or f"Feature {bbox.label}"

    def _initialize_layer_points(self, model: str) -> dict[str, dict[str, Any]]:
        """Create stable layer entries for the full model depth."""
        num_layers = get_model_num_layers(model)
        return {
            f"layer{layer_index}": {
                "layer": layer_index,
                "layer_key": f"layer{layer_index}",
                "values": self._empty_image_detail_metric_values(),
            }
            for layer_index in range(num_layers)
        }

    def _empty_image_detail_metric_values(self) -> dict[str, float | None]:
        """Build a blank values map for all image-detail metrics."""
        return {str(metric["key"]): None for metric in IMAGE_DETAIL_METRICS}

    def _build_image_layer_progression_response(
        self,
        image_id: str,
        model: str,
        method: str,
        percentile: int,
        selection: dict[str, Any],
        layer_points: dict[str, dict[str, Any]],
    ) -> dict[str, Any]:
        """Assemble the API payload for image-detail layer progression."""
        ordered_layers = [layer_points[f"layer{layer_index}"] for layer_index in range(get_model_num_layers(model))]
        return {
            "image_id": image_id,
            "model": model,
            "method": method,
            "percentile": percentile,
            "selection": selection,
            "metrics": [dict(metric) for metric in IMAGE_DETAIL_METRICS],
            "layers": ordered_layers,
        }

    def _compute_mse_aggregate_from_images(
        self,
        model: str,
        layer: str,
        percentile: int,
        method: str,
    ) -> tuple[float | None, float | None, float | None]:
        """Derive aggregate MSE stats from per-image rows when the DB is not backfilled."""
        import torch

        image_metrics = self.get_all_image_metrics(
            model=model,
            layer=layer,
            percentile=percentile,
            method=method,
        )
        mse_values = [row["mse"] for row in image_metrics if row["mse"] is not None]
        if not mse_values:
            return None, None, None

        mses = torch.tensor(mse_values, dtype=torch.float32)
        return mses.mean().item(), mses.std().item(), mses.median().item()


# Global instance
metrics_service = MetricsService()

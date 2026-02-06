"""Service for querying pre-computed metrics."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.backend.config import (
    DEFAULT_METHOD,
    METRICS_DB_PATH,
    METRICS_SUMMARY_PATH,
    display_model_name,
    resolve_model_name,
)
from app.backend.validators import resolve_default_method

if TYPE_CHECKING:
    from collections.abc import Generator


class MetricsService:
    """Service for querying pre-computed IoU metrics from SQLite."""

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
            Dict with iou, coverage, attention_area, annotation_area or None.
        """
        db_model = resolve_model_name(model)
        resolved_method = method if method else resolve_default_method(model)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT iou, coverage, attention_area, annotation_area
                   FROM image_metrics
                   WHERE image_id = ? AND model = ? AND layer = ? AND method = ? AND percentile = ?""",
                (image_id, db_model, layer, resolved_method, percentile),
            )
            row = cursor.fetchone()

            if row:
                return {
                    "image_id": image_id,
                    "model": model,  # Return original name for display
                    "layer": layer,
                    "percentile": percentile,
                    "method": resolved_method,
                    "iou": row["iou"],
                    "coverage": row["coverage"],
                    "attention_area": row["attention_area"],
                    "annotation_area": row["annotation_area"],
                }
            return None

    def get_aggregate_metrics(
        self,
        model: str,
        layer: str,
        percentile: int = 90,
        method: str | None = None,
    ) -> dict | None:
        """Get aggregate metrics for a model/layer.

        Returns:
            Dict with mean_iou, std_iou, median_iou, mean_coverage, num_images.
        """
        db_model = resolve_model_name(model)
        resolved_method = method if method else resolve_default_method(model)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT mean_iou, std_iou, median_iou, mean_coverage, num_images
                   FROM aggregate_metrics
                   WHERE model = ? AND layer = ? AND method = ? AND percentile = ?""",
                (db_model, layer, resolved_method, percentile),
            )
            row = cursor.fetchone()

            if row:
                return {
                    "model": model,  # Return original name for display
                    "layer": layer,
                    "percentile": percentile,
                    "method": resolved_method,
                    "mean_iou": row["mean_iou"],
                    "std_iou": row["std_iou"],
                    "median_iou": row["median_iou"],
                    "mean_coverage": row["mean_coverage"],
                    "num_images": row["num_images"],
                }
            return None

    def get_leaderboard(self, percentile: int = 90) -> list[dict]:
        """Get model leaderboard ranked by best IoU.

        Uses each model's default method for ranking (CLS for ViTs, etc.).

        Returns:
            List of dicts with rank, model, best_iou, best_layer.
        """
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
                f"""SELECT model, MAX(mean_iou) as best_iou,
                   (SELECT layer FROM aggregate_metrics a2
                    WHERE a2.model = a1.model AND a2.method = a1.method AND a2.percentile = ?
                    ORDER BY mean_iou DESC LIMIT 1) as best_layer
                   FROM aggregate_metrics a1
                   WHERE percentile = ? AND ({filter_clause})
                   GROUP BY model
                   ORDER BY best_iou DESC""",
                (percentile, percentile, *params),
            )

            return [
                {
                    "rank": i + 1,
                    "model": display_model_name(row["model"]),
                    "best_iou": row["best_iou"],
                    "best_layer": row["best_layer"],
                }
                for i, row in enumerate(cursor.fetchall())
            ]

    def get_layer_progression(
        self,
        model: str,
        percentile: int = 90,
        method: str | None = None,
    ) -> dict:
        """Get IoU progression across all layers.

        Returns:
            Dict with model, percentile, layers, ious, best_layer, best_iou.
        """
        db_model = resolve_model_name(model)
        resolved_method = method if method else resolve_default_method(model)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT layer, mean_iou FROM aggregate_metrics
                   WHERE model = ? AND method = ? AND percentile = ?
                   ORDER BY layer""",
                (db_model, resolved_method, percentile),
            )
            rows = cursor.fetchall()

            layers = [row["layer"] for row in rows]
            ious = [row["mean_iou"] for row in rows]

            best_idx = ious.index(max(ious)) if ious else 0
            best_layer = layers[best_idx] if layers else "layer11"
            best_iou = ious[best_idx] if ious else 0.0

            return {
                "model": model,  # Return original name for display
                "percentile": percentile,
                "method": resolved_method,
                "layers": layers,
                "ious": ious,
                "best_layer": best_layer,
                "best_iou": best_iou,
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
            List of dicts with image_id, iou, coverage.
        """
        db_model = resolve_model_name(model)
        resolved_method = method if method else resolve_default_method(model)
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT image_id, iou, coverage, attention_area, annotation_area
                   FROM image_metrics
                   WHERE model = ? AND layer = ? AND method = ? AND percentile = ?
                   ORDER BY iou DESC""",
                (db_model, layer, resolved_method, percentile),
            )

            return [
                {
                    "image_id": row["image_id"],
                    "iou": row["iou"],
                    "coverage": row["coverage"],
                    "attention_area": row["attention_area"],
                    "annotation_area": row["annotation_area"],
                }
                for row in cursor.fetchall()
            ]

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


# Global instance
metrics_service = MetricsService()

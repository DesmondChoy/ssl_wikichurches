"""Service for querying pre-computed metrics."""

from __future__ import annotations

import json
import sqlite3
from contextlib import contextmanager
from pathlib import Path
from typing import TYPE_CHECKING, Any

from app.backend.config import (
    METRICS_DB_PATH,
    METRICS_SUMMARY_PATH,
)

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
    ) -> dict | None:
        """Get metrics for a specific image/model/layer combination.

        Returns:
            Dict with iou, coverage, attention_area, annotation_area or None.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT iou, coverage, attention_area, annotation_area
                   FROM image_metrics
                   WHERE image_id = ? AND model = ? AND layer = ? AND percentile = ?""",
                (image_id, model, layer, percentile),
            )
            row = cursor.fetchone()

            if row:
                return {
                    "image_id": image_id,
                    "model": model,
                    "layer": layer,
                    "percentile": percentile,
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
    ) -> dict | None:
        """Get aggregate metrics for a model/layer.

        Returns:
            Dict with mean_iou, std_iou, median_iou, mean_coverage, num_images.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT mean_iou, std_iou, median_iou, mean_coverage, num_images
                   FROM aggregate_metrics
                   WHERE model = ? AND layer = ? AND percentile = ?""",
                (model, layer, percentile),
            )
            row = cursor.fetchone()

            if row:
                return {
                    "model": model,
                    "layer": layer,
                    "percentile": percentile,
                    "mean_iou": row["mean_iou"],
                    "std_iou": row["std_iou"],
                    "median_iou": row["median_iou"],
                    "mean_coverage": row["mean_coverage"],
                    "num_images": row["num_images"],
                }
            return None

    def get_leaderboard(self, percentile: int = 90) -> list[dict]:
        """Get model leaderboard ranked by best IoU.

        Returns:
            List of dicts with rank, model, best_iou, best_layer.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT model, MAX(mean_iou) as best_iou,
                   (SELECT layer FROM aggregate_metrics a2
                    WHERE a2.model = a1.model AND a2.percentile = ?
                    ORDER BY mean_iou DESC LIMIT 1) as best_layer
                   FROM aggregate_metrics a1
                   WHERE percentile = ?
                   GROUP BY model
                   ORDER BY best_iou DESC""",
                (percentile, percentile),
            )

            return [
                {
                    "rank": i + 1,
                    "model": row["model"],
                    "best_iou": row["best_iou"],
                    "best_layer": row["best_layer"],
                }
                for i, row in enumerate(cursor.fetchall())
            ]

    def get_layer_progression(
        self,
        model: str,
        percentile: int = 90,
    ) -> dict:
        """Get IoU progression across all layers.

        Returns:
            Dict with model, percentile, layers, ious, best_layer, best_iou.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT layer, mean_iou FROM aggregate_metrics
                   WHERE model = ? AND percentile = ?
                   ORDER BY layer""",
                (model, percentile),
            )
            rows = cursor.fetchall()

            layers = [row["layer"] for row in rows]
            ious = [row["mean_iou"] for row in rows]

            best_idx = ious.index(max(ious)) if ious else 0
            best_layer = layers[best_idx] if layers else "layer11"
            best_iou = ious[best_idx] if ious else 0.0

            return {
                "model": model,
                "percentile": percentile,
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
    ) -> dict:
        """Get IoU breakdown by architectural style.

        Returns:
            Dict with model, layer, percentile, styles (name->iou), style_counts.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT style_name, mean_iou, num_images FROM style_metrics
                   WHERE model = ? AND layer = ? AND percentile = ?""",
                (model, layer, percentile),
            )

            styles = {}
            counts = {}
            for row in cursor.fetchall():
                styles[row["style_name"]] = row["mean_iou"]
                counts[row["style_name"]] = row["num_images"]

            return {
                "model": model,
                "layer": layer,
                "percentile": percentile,
                "styles": styles,
                "style_counts": counts,
            }

    def get_all_image_metrics(
        self,
        model: str,
        layer: str,
        percentile: int = 90,
    ) -> list[dict]:
        """Get metrics for all images for a model/layer.

        Returns:
            List of dicts with image_id, iou, coverage.
        """
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                """SELECT image_id, iou, coverage, attention_area, annotation_area
                   FROM image_metrics
                   WHERE model = ? AND layer = ? AND percentile = ?
                   ORDER BY iou DESC""",
                (model, layer, percentile),
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


# Global instance
metrics_service = MetricsService()

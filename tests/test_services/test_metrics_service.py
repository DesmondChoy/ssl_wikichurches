"""Tests for MetricsService layer ordering.

Regression test for issue #9ct.2: layer progression queries must return
layers in numeric order (layer0, layer1, ..., layer11), not lexicographic
order (layer0, layer1, layer10, layer11, layer2, ...).
"""

from __future__ import annotations

import sqlite3
from unittest.mock import patch

import pytest


def _create_test_db(conn: sqlite3.Connection, num_layers: int = 12) -> None:
    """Populate an in-memory SQLite database with synthetic aggregate metrics.

    Creates rows for layers 0..num_layers-1 where mean_iou increases with
    layer index (simulating typical ViT behaviour where later layers attend
    better). Rows are inserted in *reversed* order to stress-test SQL ORDER BY.
    """
    conn.execute(
        """CREATE TABLE aggregate_metrics (
            model TEXT, layer TEXT, method TEXT, percentile INTEGER,
            mean_iou REAL, std_iou REAL, median_iou REAL,
            mean_coverage REAL, num_images INTEGER
        )"""
    )

    # Insert in reversed order to stress-test SQL ORDER BY
    for idx in reversed(range(num_layers)):
        conn.execute(
            """INSERT INTO aggregate_metrics
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "dinov2",
                f"layer{idx}",
                "cls",
                90,
                0.1 + idx * 0.01,  # mean_iou increases with layer
                0.05,
                0.1 + idx * 0.01,
                0.5,
                139,
            ),
        )
    conn.commit()


class TestLayerProgression:
    """Verify get_layer_progression() returns layers in numeric order."""

    @pytest.fixture
    def service(self):
        """Create a MetricsService that queries an in-memory SQLite DB."""
        from app.backend.services.metrics_service import MetricsService

        instance = object.__new__(MetricsService)
        return instance

    @pytest.fixture
    def mem_db(self):
        """In-memory SQLite database with 12 layers (layer0–layer11)."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        _create_test_db(conn, num_layers=12)
        yield conn
        conn.close()

    def test_layers_returned_in_numeric_order(self, service, mem_db):
        """Layers must be ordered layer0, layer1, ..., layer11."""
        with patch.object(type(service), "get_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = lambda _: mem_db
            mock_ctx.return_value.__exit__ = lambda *_: None

            result = service.get_layer_progression(
                model="dinov2", percentile=90, method="cls"
            )

        expected = [f"layer{i}" for i in range(12)]
        assert result["layers"] == expected

    def test_ious_match_layer_order(self, service, mem_db):
        """IoU values must correspond to the correctly-ordered layers."""
        with patch.object(type(service), "get_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = lambda _: mem_db
            mock_ctx.return_value.__exit__ = lambda *_: None

            result = service.get_layer_progression(
                model="dinov2", percentile=90, method="cls"
            )

        # With our synthetic data, IoU increases with layer index
        for i, iou in enumerate(result["ious"]):
            assert iou == pytest.approx(0.1 + i * 0.01, abs=1e-9)

    def test_best_layer_is_highest_numeric(self, service, mem_db):
        """best_layer must be layer11 (highest IoU), not layer9 (lex-max)."""
        with patch.object(type(service), "get_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = lambda _: mem_db
            mock_ctx.return_value.__exit__ = lambda *_: None

            result = service.get_layer_progression(
                model="dinov2", percentile=90, method="cls"
            )

        assert result["best_layer"] == "layer11"
        assert result["best_iou"] == pytest.approx(0.1 + 11 * 0.01, abs=1e-9)

    def test_four_layers_already_correct(self, service):
        """ResNet-50's 4 layers (layer0–layer3) are fine even with string sort."""
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        _create_test_db(conn, num_layers=4)

        with patch.object(type(service), "get_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = lambda _: conn
            mock_ctx.return_value.__exit__ = lambda *_: None

            result = service.get_layer_progression(
                model="dinov2", percentile=90, method="cls"
            )

        expected = [f"layer{i}" for i in range(4)]
        assert result["layers"] == expected
        conn.close()

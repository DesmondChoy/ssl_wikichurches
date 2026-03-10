"""Tests for MetricsService ordering and metric-selection semantics."""

from __future__ import annotations

import sqlite3
from unittest.mock import patch

import pytest
import torch

from ssl_attention.data.annotations import BoundingBox, ImageAnnotation


def _create_progression_db(conn: sqlite3.Connection, num_layers: int = 12) -> None:
    """Populate an in-memory SQLite DB with synthetic aggregate metrics."""
    conn.execute(
        """CREATE TABLE aggregate_metrics (
            model TEXT,
            layer TEXT,
            method TEXT,
            percentile INTEGER,
            mean_iou REAL,
            std_iou REAL,
            median_iou REAL,
            mean_coverage REAL,
            mean_mse REAL,
            std_mse REAL,
            median_mse REAL,
            num_images INTEGER
        )"""
    )

    for idx in reversed(range(num_layers)):
        conn.execute(
            """INSERT INTO aggregate_metrics
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                "dinov2",
                f"layer{idx}",
                "cls",
                90,
                0.1 + idx * 0.01,
                0.05,
                0.1 + idx * 0.01,
                0.5,
                0.30 - idx * 0.01,
                0.02,
                0.30 - idx * 0.01,
                139,
            ),
        )
    conn.commit()


def _create_leaderboard_db(conn: sqlite3.Connection) -> None:
    """Populate rows for leaderboard ranking tests."""
    conn.execute(
        """CREATE TABLE aggregate_metrics (
            model TEXT,
            layer TEXT,
            method TEXT,
            percentile INTEGER,
            mean_iou REAL,
            std_iou REAL,
            median_iou REAL,
            mean_coverage REAL,
            mean_mse REAL,
            std_mse REAL,
            median_mse REAL,
            num_images INTEGER
        )"""
    )

    rows = [
        ("dinov2", "layer0", "cls", 90, 0.40, 0.01, 0.40, 0.5, 0.20, 0.01, 0.20, 139),
        ("dinov2", "layer1", "cls", 90, 0.55, 0.01, 0.55, 0.5, 0.12, 0.01, 0.12, 139),
        ("clip", "layer0", "cls", 90, 0.45, 0.01, 0.45, 0.5, 0.18, 0.01, 0.18, 139),
        ("clip", "layer1", "cls", 90, 0.50, 0.01, 0.50, 0.5, 0.16, 0.01, 0.16, 139),
    ]
    conn.executemany("INSERT INTO aggregate_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    conn.commit()


def _create_image_progression_db(
    conn: sqlite3.Connection,
    model: str = "dinov2",
    num_layers: int = 12,
) -> None:
    """Populate per-image metrics rows for image-detail progression tests."""
    conn.execute(
        """CREATE TABLE image_metrics (
            image_id TEXT,
            model TEXT,
            layer TEXT,
            method TEXT,
            percentile INTEGER,
            iou REAL,
            coverage REAL,
            mse REAL,
            attention_area REAL,
            annotation_area REAL
        )"""
    )

    rows = []
    for idx in reversed(range(num_layers)):
        rows.append(
            (
                "Q123_test.jpg",
                model,
                f"layer{idx}",
                "cls",
                90,
                0.15 + idx * 0.01,
                0.42 + idx * 0.005,
                0.12 - idx * 0.004,
                0.20,
                0.10,
            )
        )
        rows.append(
            (
                "Q123_test.jpg",
                model,
                f"layer{idx}",
                "cls",
                50,
                0.05 + idx * 0.008,
                0.42 + idx * 0.005,
                0.12 - idx * 0.004,
                0.20,
                0.10,
            )
        )

    conn.executemany("INSERT INTO image_metrics VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)", rows)
    conn.commit()


class TestLayerProgression:
    """Verify get_layer_progression() ordering and best-layer logic."""

    @pytest.fixture
    def service(self):
        from app.backend.services.metrics_service import MetricsService

        return object.__new__(MetricsService)

    @pytest.fixture
    def mem_db(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        _create_progression_db(conn, num_layers=12)
        yield conn
        conn.close()

    def test_layers_returned_in_numeric_order(self, service, mem_db):
        with patch.object(type(service), "get_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = lambda _: mem_db
            mock_ctx.return_value.__exit__ = lambda *_: None

            result = service.get_layer_progression(model="dinov2", percentile=90, method="cls")

        assert result["layers"] == [f"layer{i}" for i in range(12)]

    def test_iou_scores_match_layer_order(self, service, mem_db):
        with patch.object(type(service), "get_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = lambda _: mem_db
            mock_ctx.return_value.__exit__ = lambda *_: None

            result = service.get_layer_progression(model="dinov2", percentile=90, method="cls")

        for i, score in enumerate(result["scores"]):
            assert score == pytest.approx(0.1 + i * 0.01, abs=1e-9)

    def test_best_layer_is_highest_numeric_for_iou(self, service, mem_db):
        with patch.object(type(service), "get_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = lambda _: mem_db
            mock_ctx.return_value.__exit__ = lambda *_: None

            result = service.get_layer_progression(model="dinov2", percentile=90, method="cls")

        assert result["best_layer"] == "layer11"
        assert result["best_score"] == pytest.approx(0.1 + 11 * 0.01, abs=1e-9)

    def test_mse_uses_lowest_score_for_best_layer(self, service, mem_db):
        with patch.object(type(service), "get_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = lambda _: mem_db
            mock_ctx.return_value.__exit__ = lambda *_: None

            result = service.get_layer_progression(
                model="dinov2",
                percentile=90,
                method="cls",
                metric="mse",
            )

        assert result["metric"] == "mse"
        assert result["best_layer"] == "layer11"
        assert result["best_score"] == pytest.approx(0.30 - 11 * 0.01, abs=1e-9)


class TestLeaderboardOrdering:
    """Verify leaderboard ordering for selectable metrics."""

    @pytest.fixture
    def service(self):
        from app.backend.services.metrics_service import MetricsService

        return object.__new__(MetricsService)

    def test_mse_leaderboard_sorts_ascending(self, service):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        _create_leaderboard_db(conn)

        with patch.object(type(service), "get_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = lambda _: conn
            mock_ctx.return_value.__exit__ = lambda *_: None

            leaderboard = service.get_leaderboard(percentile=90, metric="mse")

        assert [entry["model"] for entry in leaderboard] == ["dinov2", "clip"]
        assert leaderboard[0]["score"] == pytest.approx(0.12, abs=1e-9)
        assert leaderboard[0]["best_layer"] == "layer1"
        assert all(entry["metric"] == "mse" for entry in leaderboard)
        conn.close()


class TestImageDetailLayerProgression:
    """Verify the extensible image-detail progression helpers."""

    @pytest.fixture
    def service(self):
        from app.backend.services.metrics_service import MetricsService

        return object.__new__(MetricsService)

    @pytest.fixture
    def image_progression_db(self):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        _create_image_progression_db(conn, model="dinov2", num_layers=12)
        yield conn
        conn.close()

    def test_union_progression_layers_are_numeric_and_include_descriptors(self, service, image_progression_db):
        with patch.object(type(service), "get_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = lambda _: image_progression_db
            mock_ctx.return_value.__exit__ = lambda *_: None

            result = service.get_image_layer_progression(
                image_id="Q123_test.jpg",
                model="dinov2",
                percentile=90,
                method="cls",
            )

        assert result is not None
        assert result["selection"]["mode"] == "union"
        assert [metric["key"] for metric in result["metrics"]] == ["iou", "coverage", "mse"]
        assert [metric["direction"] for metric in result["metrics"]] == ["higher", "higher", "lower"]
        assert [metric["percentile_dependent"] for metric in result["metrics"]] == [True, False, False]
        assert [point["layer_key"] for point in result["layers"]] == [f"layer{i}" for i in range(12)]

    def test_union_progression_respects_model_specific_layer_count(self, service):
        conn = sqlite3.connect(":memory:")
        conn.row_factory = sqlite3.Row
        _create_image_progression_db(conn, model="resnet50", num_layers=4)

        with patch.object(type(service), "get_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = lambda _: conn
            mock_ctx.return_value.__exit__ = lambda *_: None

            result = service.get_image_layer_progression(
                image_id="Q123_test.jpg",
                model="resnet50",
                percentile=90,
                method="cls",
            )

        assert result is not None
        assert [point["layer"] for point in result["layers"]] == [0, 1, 2, 3]
        conn.close()

    def test_percentile_changes_only_change_thresholded_metric_values(self, service, image_progression_db):
        with patch.object(type(service), "get_connection") as mock_ctx:
            mock_ctx.return_value.__enter__ = lambda _: image_progression_db
            mock_ctx.return_value.__exit__ = lambda *_: None

            p90 = service.get_image_layer_progression("Q123_test.jpg", "dinov2", percentile=90, method="cls")
            p50 = service.get_image_layer_progression("Q123_test.jpg", "dinov2", percentile=50, method="cls")

        assert p90 is not None and p50 is not None
        assert p90["layers"][0]["values"]["iou"] != p50["layers"][0]["values"]["iou"]
        assert p90["layers"][0]["values"]["coverage"] == pytest.approx(p50["layers"][0]["values"]["coverage"], abs=1e-9)
        assert p90["layers"][0]["values"]["mse"] == pytest.approx(p50["layers"][0]["values"]["mse"], abs=1e-9)

    def test_bbox_progression_returns_metric_values_for_each_layer(self, service):
        annotation = ImageAnnotation(
            image_id="Q123_test.jpg",
            styles=(),
            bboxes=(
                BoundingBox(left=0.25, top=0.25, width=0.5, height=0.5, label=7, group_label=7),
            ),
        )
        attention = torch.tensor(
            [
                [0.05, 0.15, 0.20, 0.10],
                [0.10, 0.70, 0.80, 0.15],
                [0.05, 0.75, 0.90, 0.10],
                [0.02, 0.10, 0.12, 0.03],
            ],
            dtype=torch.float32,
        )

        with (
            patch.object(type(service), "_get_annotation", return_value=annotation),
            patch.object(type(service), "_get_bbox_label", return_value="Window"),
            patch.object(type(service), "_load_attention_tensor", return_value=attention),
        ):
            result = service.get_bbox_layer_progression(
                image_id="Q123_test.jpg",
                model="resnet50",
                bbox_index=0,
                percentile=90,
                method="gradcam",
            )

        assert result is not None
        assert result["selection"] == {"mode": "bbox", "bbox_index": 0, "bbox_label": "Window"}
        assert [point["layer"] for point in result["layers"]] == [0, 1, 2, 3]
        assert all(point["values"]["mse"] is not None for point in result["layers"])

    def test_bbox_progression_rejects_out_of_range_index(self, service):
        annotation = ImageAnnotation(
            image_id="Q123_test.jpg",
            styles=(),
            bboxes=(
                BoundingBox(left=0.25, top=0.25, width=0.5, height=0.5, label=7, group_label=7),
            ),
        )

        with (
            patch.object(type(service), "_get_annotation", return_value=annotation),
            pytest.raises(ValueError, match="bbox_index 3 out of range"),
        ):
            service.get_bbox_layer_progression(
                image_id="Q123_test.jpg",
                model="dinov2",
                bbox_index=3,
                percentile=90,
                method="cls",
            )

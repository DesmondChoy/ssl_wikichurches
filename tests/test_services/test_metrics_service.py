"""Tests for MetricsService ordering and metric-selection semantics."""

from __future__ import annotations

import sqlite3
from unittest.mock import patch

import pytest


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

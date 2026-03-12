"""Tests for metrics-cache schema creation and migration."""

from __future__ import annotations

import sqlite3

from app.precompute.generate_metrics_cache import create_database


def test_create_database_migrates_existing_schema_with_continuous_metric_columns(tmp_path):
    db_path = tmp_path / "metrics.db"
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    cursor.execute(
        """CREATE TABLE image_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            layer TEXT NOT NULL,
            method TEXT NOT NULL,
            image_id TEXT NOT NULL,
            percentile INTEGER NOT NULL,
            iou REAL NOT NULL,
            coverage REAL NOT NULL,
            attention_area REAL NOT NULL,
            annotation_area REAL NOT NULL,
            UNIQUE(model, layer, method, image_id, percentile)
        )"""
    )
    cursor.execute(
        """CREATE TABLE aggregate_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            layer TEXT NOT NULL,
            method TEXT NOT NULL,
            percentile INTEGER NOT NULL,
            mean_iou REAL NOT NULL,
            std_iou REAL NOT NULL,
            median_iou REAL NOT NULL,
            mean_coverage REAL NOT NULL,
            num_images INTEGER NOT NULL,
            UNIQUE(model, layer, method, percentile)
        )"""
    )

    cursor.execute(
        """INSERT INTO image_metrics
           (model, layer, method, image_id, percentile, iou, coverage, attention_area, annotation_area)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("dinov2", "layer0", "cls", "image.jpg", 90, 0.5, 0.6, 0.1, 0.2),
    )
    cursor.execute(
        """INSERT INTO aggregate_metrics
           (model, layer, method, percentile, mean_iou, std_iou, median_iou, mean_coverage, num_images)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        ("dinov2", "layer0", "cls", 90, 0.5, 0.1, 0.5, 0.6, 139),
    )
    conn.commit()
    conn.close()

    migrated = create_database(db_path)
    migrated_cursor = migrated.cursor()

    migrated_cursor.execute("PRAGMA table_info(image_metrics)")
    image_columns = {row[1] for row in migrated_cursor.fetchall()}
    assert "mse" in image_columns
    assert "kl" in image_columns

    migrated_cursor.execute("PRAGMA table_info(aggregate_metrics)")
    aggregate_columns = {row[1] for row in migrated_cursor.fetchall()}
    assert {"mean_mse", "std_mse", "median_mse"}.issubset(aggregate_columns)
    assert {"mean_kl", "std_kl", "median_kl"}.issubset(aggregate_columns)

    migrated_cursor.execute("SELECT COUNT(*) FROM image_metrics")
    assert migrated_cursor.fetchone()[0] == 1
    migrated_cursor.execute("SELECT COUNT(*) FROM aggregate_metrics")
    assert migrated_cursor.fetchone()[0] == 1

    migrated.close()

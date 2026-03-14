"""Generate pre-computed metrics cache for the visualization app.

Computes IoU, coverage, Gaussian-ground-truth MSE, Gaussian-ground-truth
KL divergence, and Gaussian-ground-truth EMD/Wasserstein-1 metrics for all
model/layer/image combinations at multiple percentile thresholds, storing
results in SQLite for fast queries.

Usage:
    python -m app.precompute.generate_metrics_cache
    python -m app.precompute.generate_metrics_cache --models dinov2 clip
"""

from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from pathlib import Path

from tqdm import tqdm

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from app.backend.config import display_model_name
from ssl_attention.cache import AttentionCache
from ssl_attention.config import (
    ANNOTATIONS_PATH,
    CACHE_PATH,
    DATASET_PATH,
    DEFAULT_METHOD,
    MODEL_METHODS,
    MODELS,
    STYLE_MAPPING,
    STYLE_NAMES,
)
from ssl_attention.data import AnnotatedSubset
from ssl_attention.data.annotations import load_annotations_with_features
from ssl_attention.metrics import (
    compute_image_emd,
    compute_image_iou,
    compute_image_kl,
    compute_image_mse,
)
from ssl_attention.metrics.iou import aggregate_by_feature_type, compute_per_bbox_iou

DEFAULT_PERCENTILES = [90, 85, 80, 75, 70, 60, 50]
IMAGE_METRIC_MIGRATIONS = {"mse": "REAL", "kl": "REAL", "emd": "REAL"}
AGGREGATE_METRIC_MIGRATIONS = {
    "mean_mse": "REAL",
    "std_mse": "REAL",
    "median_mse": "REAL",
    "mean_kl": "REAL",
    "std_kl": "REAL",
    "median_kl": "REAL",
    "mean_emd": "REAL",
    "std_emd": "REAL",
    "median_emd": "REAL",
}


def ensure_table_columns(
    conn: sqlite3.Connection,
    table_name: str,
    expected_columns: dict[str, str],
) -> None:
    """Add missing columns to an existing SQLite table in place."""
    cursor = conn.cursor()
    cursor.execute(f"PRAGMA table_info({table_name})")
    existing_columns = {row[1] for row in cursor.fetchall()}

    for column_name, column_type in expected_columns.items():
        if column_name not in existing_columns:
            cursor.execute(f"ALTER TABLE {table_name} ADD COLUMN {column_name} {column_type}")


def create_database(db_path: Path) -> sqlite3.Connection:
    """Create metrics database with schema."""
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Per-image metrics table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS image_metrics (
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
            mse REAL,
            kl REAL,
            emd REAL,
            UNIQUE(model, layer, method, image_id, percentile)
        )
    """)

    # Aggregate metrics table (per model/layer/method)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS aggregate_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            layer TEXT NOT NULL,
            method TEXT NOT NULL,
            percentile INTEGER NOT NULL,
            mean_iou REAL NOT NULL,
            std_iou REAL NOT NULL,
            median_iou REAL NOT NULL,
            mean_coverage REAL NOT NULL,
            mean_mse REAL,
            std_mse REAL,
            median_mse REAL,
            mean_kl REAL,
            std_kl REAL,
            median_kl REAL,
            mean_emd REAL,
            std_emd REAL,
            median_emd REAL,
            num_images INTEGER NOT NULL,
            UNIQUE(model, layer, method, percentile)
        )
    """)

    # Style breakdown table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS style_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            layer TEXT NOT NULL,
            method TEXT NOT NULL,
            style_name TEXT NOT NULL,
            percentile INTEGER NOT NULL,
            mean_iou REAL NOT NULL,
            num_images INTEGER NOT NULL,
            UNIQUE(model, layer, method, style_name, percentile)
        )
    """)

    # Feature breakdown table (per architectural feature type)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS feature_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            layer TEXT NOT NULL,
            method TEXT NOT NULL,
            feature_label INTEGER NOT NULL,
            feature_name TEXT NOT NULL,
            percentile INTEGER NOT NULL,
            mean_iou REAL NOT NULL,
            std_iou REAL NOT NULL,
            bbox_count INTEGER NOT NULL,
            UNIQUE(model, layer, method, feature_label, percentile)
        )
    """)
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_feature_model_layer ON feature_metrics(model, layer, method, percentile)"
    )

    # Indexes for fast queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_image_model_layer ON image_metrics(model, layer, method)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_image_image_id ON image_metrics(image_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_agg_model ON aggregate_metrics(model, method)")

    ensure_table_columns(conn, "image_metrics", IMAGE_METRIC_MIGRATIONS)
    ensure_table_columns(conn, "aggregate_metrics", AGGREGATE_METRIC_MIGRATIONS)

    conn.commit()
    return conn


def compute_metrics_for_model(
    model_name: str,
    dataset: AnnotatedSubset,
    attention_cache: AttentionCache,
    conn: sqlite3.Connection,
    percentiles: list[int],
    layers: list[int] | None = None,
    methods: list[str] | None = None,
    skip_existing: bool = True,
) -> dict[str, int]:
    """Compute metrics for a single model across all its attention methods.

    Args:
        model_name: Name of model.
        dataset: Annotated dataset.
        attention_cache: AttentionCache with pre-computed attention.
        conn: SQLite connection.
        percentiles: List of percentile thresholds.
        layers: Specific layers to process. None = all.
        methods: Specific methods to process. None = all for this model.
        skip_existing: Skip if already in database.

    Returns:
        Dict with statistics.
    """
    import torch

    stats = {"processed": 0, "skipped": 0, "errors": 0}

    model_config = MODELS[model_name]
    num_layers = model_config.num_layers
    layers_to_process = layers if layers else list(range(num_layers))

    # Get all methods for this model (or filter by CLI arg)
    all_methods = [m.value for m in MODEL_METHODS[model_name]]
    methods_to_process = [m for m in methods if m in all_methods] if methods else all_methods

    print(f"\nProcessing {model_name} ({len(layers_to_process)} layers, methods: {methods_to_process})")

    cursor = conn.cursor()

    # Build style mapping for images (metadata-only, no image I/O)
    image_styles: dict[str, str | None] = {}
    for image_id in dataset.image_ids:
        annotation = dataset.annotations[image_id]
        style_name = None
        for style_qid in annotation.styles:
            if style_qid in STYLE_MAPPING:
                style_name = STYLE_NAMES[STYLE_MAPPING[style_qid]]
                break
        image_styles[image_id] = style_name

    for variant in methods_to_process:
        print(f"  Method: {variant}")

        # Process each image (metadata-only, no image I/O)
        for image_id in tqdm(dataset.image_ids, desc=f"{model_name}/{variant}"):
            annotation = dataset.annotations[image_id]

            for layer in layers_to_process:
                layer_key = f"layer{layer}"

                # Load attention from cache
                try:
                    attention = attention_cache.load(model_name, layer_key, image_id, variant=variant)
                except KeyError:
                    stats["skipped"] += len(percentiles)
                    continue

                try:
                    image_mse = compute_image_mse(attention=attention, annotation=annotation)
                    image_kl = compute_image_kl(attention=attention, annotation=annotation)
                    image_emd = compute_image_emd(attention=attention, annotation=annotation)
                except Exception as e:
                    print(f"\nError computing continuous metrics for {image_id} layer{layer}/{variant}: {e}")
                    stats["errors"] += len(percentiles)
                    continue

                for percentile in percentiles:
                    # Check if already exists
                    if skip_existing:
                        cursor.execute(
                            """SELECT mse, kl, emd FROM image_metrics
                               WHERE model=? AND layer=? AND method=? AND image_id=? AND percentile=?""",
                            (model_name, layer_key, variant, image_id, percentile),
                        )
                        existing_row = cursor.fetchone()
                        if (
                            existing_row
                            and existing_row[0] is not None
                            and existing_row[1] is not None
                            and existing_row[2] is not None
                        ):
                            stats["skipped"] += 1
                            continue

                    try:
                        result = compute_image_iou(
                            attention=attention,
                            annotation=annotation,
                            image_id=image_id,
                            percentile=percentile,
                        )

                        # Insert into database
                        cursor.execute(
                            """INSERT OR REPLACE INTO image_metrics
                               (model, layer, method, image_id, percentile, iou, coverage, attention_area, annotation_area, mse, kl, emd)
                               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                            (
                                model_name,
                                layer_key,
                                variant,
                                image_id,
                                percentile,
                                result.iou,
                                result.coverage,
                                result.attention_area,
                                result.annotation_area,
                                image_mse,
                                image_kl,
                                image_emd,
                            ),
                        )

                        stats["processed"] += 1

                    except Exception as e:
                        print(f"\nError computing metrics for {image_id} layer{layer}/{variant}: {e}")
                        stats["errors"] += 1

        conn.commit()

        # Compute aggregate metrics for this method
        print(f"  Computing aggregates for {model_name}/{variant}...")
        for layer in layers_to_process:
            layer_key = f"layer{layer}"

            for percentile in percentiles:
                cursor.execute(
                    """SELECT iou, coverage, mse, kl, emd FROM image_metrics
                       WHERE model=? AND layer=? AND method=? AND percentile=?""",
                    (model_name, layer_key, variant, percentile),
                )
                rows = cursor.fetchall()

                if not rows:
                    continue

                complete_rows = [row for row in rows if row[2] is not None and row[3] is not None and row[4] is not None]
                if not complete_rows:
                    continue

                ious = torch.tensor([r[0] for r in complete_rows])
                coverages = torch.tensor([r[1] for r in complete_rows])
                mses = torch.tensor([r[2] for r in complete_rows], dtype=torch.float32)
                kls = torch.tensor([r[3] for r in complete_rows], dtype=torch.float32)
                emds = torch.tensor([r[4] for r in complete_rows], dtype=torch.float32)

                cursor.execute(
                    """INSERT OR REPLACE INTO aggregate_metrics
                       (model, layer, method, percentile, mean_iou, std_iou, median_iou, mean_coverage,
                        mean_mse, std_mse, median_mse, mean_kl, std_kl, median_kl,
                        mean_emd, std_emd, median_emd, num_images)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                    (
                        model_name,
                        layer_key,
                        variant,
                        percentile,
                        ious.mean().item(),
                        ious.std().item(),
                        ious.median().item(),
                        coverages.mean().item(),
                        mses.mean().item(),
                        mses.std().item(),
                        mses.median().item(),
                        kls.mean().item(),
                        kls.std().item(),
                        kls.median().item(),
                        emds.mean().item(),
                        emds.std().item(),
                        emds.median().item(),
                        len(complete_rows),
                    ),
                )

        conn.commit()

        # Compute style breakdown for this method
        print(f"  Computing style breakdown for {model_name}/{variant}...")
        for layer in layers_to_process:
            layer_key = f"layer{layer}"

            for percentile in percentiles:
                for style_name in STYLE_NAMES:
                    style_images = [img_id for img_id, s in image_styles.items() if s == style_name]
                    if not style_images:
                        continue

                    placeholders = ",".join("?" * len(style_images))
                    cursor.execute(
                        f"""SELECT AVG(iou), COUNT(*) FROM image_metrics
                           WHERE model=? AND layer=? AND method=? AND percentile=? AND image_id IN ({placeholders})""",
                        (model_name, layer_key, variant, percentile, *style_images),
                    )
                    row = cursor.fetchone()

                    if row and row[0] is not None:
                        cursor.execute(
                            """INSERT OR REPLACE INTO style_metrics
                               (model, layer, method, style_name, percentile, mean_iou, num_images)
                               VALUES (?, ?, ?, ?, ?, ?, ?)""",
                            (model_name, layer_key, variant, style_name, percentile, row[0], row[1]),
                        )

        conn.commit()

        # Compute feature breakdown for this method
        print(f"  Computing feature breakdown for {model_name}/{variant}...")
        _, feature_types = load_annotations_with_features(ANNOTATIONS_PATH)
        feature_names = [ft.name for ft in feature_types]

        for layer in layers_to_process:
            layer_key = f"layer{layer}"
            per_bbox_by_pct: dict[int, list[list[tuple[int, float]]]] = {
                p: [] for p in percentiles
            }

            for image_id in dataset.image_ids:
                annotation = dataset.annotations[image_id]
                try:
                    attention = attention_cache.load(model_name, layer_key, image_id, variant=variant)
                except KeyError:
                    continue

                for percentile in percentiles:
                    bbox_ious = compute_per_bbox_iou(attention, annotation, percentile)
                    per_bbox_by_pct[percentile].append(bbox_ious)

            for percentile in percentiles:
                per_bbox_results = per_bbox_by_pct[percentile]
                if not per_bbox_results:
                    continue

                feature_stats = aggregate_by_feature_type(per_bbox_results, feature_names)

                for label, stats_dict in feature_stats.items():
                    feature_name = stats_dict.get("name", f"feature_{label}")
                    cursor.execute(
                        """INSERT OR REPLACE INTO feature_metrics
                           (model, layer, method, feature_label, feature_name, percentile, mean_iou, std_iou, bbox_count)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            model_name,
                            layer_key,
                            variant,
                            label,
                            feature_name,
                            percentile,
                            stats_dict["mean_iou"],
                            stats_dict["std_iou"],
                            int(stats_dict["count"]),
                        ),
                    )

        conn.commit()

    return stats


def export_summary_json(conn: sqlite3.Connection, output_path: Path) -> None:
    """Export summary statistics to JSON for fast frontend loading.

    Uses each model's default method for the summary (same as leaderboard).
    """
    from typing import Any

    cursor = conn.cursor()

    models_data: dict[str, Any] = {}
    leaderboard: list[dict[str, Any]] = []
    leaderboards: dict[str, list[dict[str, Any]]] = {}
    metric_configs = {
        "iou": {"column": "mean_iou", "order": "DESC", "legacy_key": "best_iou"},
        "mse": {"column": "mean_mse", "order": "ASC", "legacy_key": "best_mse"},
        "kl": {"column": "mean_kl", "order": "ASC", "legacy_key": "best_kl"},
        "emd": {"column": "mean_emd", "order": "ASC", "legacy_key": "best_emd"},
    }

    # Get distinct models
    cursor.execute("SELECT DISTINCT model FROM aggregate_metrics")
    models = [row[0] for row in cursor.fetchall()]

    # Build metric-specific leaderboards using each model's default method
    for metric_name, config in metric_configs.items():
        metric_entries: list[dict[str, Any]] = []
        score_column = config["column"]
        order_direction = config["order"]

        for model in models:
            default_method = DEFAULT_METHOD[model].value
            cursor.execute(
                f"""SELECT layer, {score_column}
                    FROM aggregate_metrics
                    WHERE model = ? AND method = ? AND percentile = 90 AND {score_column} IS NOT NULL
                    ORDER BY {score_column} {order_direction}, CAST(SUBSTR(layer, 6) AS INTEGER) ASC
                    LIMIT 1""",
                (model, default_method),
            )
            row = cursor.fetchone()
            if row:
                metric_entries.append(
                    {
                        "model": display_model_name(model),
                        "metric": metric_name,
                        "best_layer": row[0],
                        "best_score": row[1],
                        "method_used": default_method,
                    }
                )

        metric_entries.sort(
            key=lambda entry: (entry["best_score"], entry["model"])
            if order_direction == "ASC"
            else (-entry["best_score"], entry["model"])
        )
        leaderboards[metric_name] = metric_entries

    leaderboard = [
        {
            "model": entry["model"],
            "best_iou": entry["best_score"],
            "method_used": entry["method_used"],
        }
        for entry in leaderboards.get("iou", [])
    ]

    # Get per-model summaries (at default method)
    for model in models:
        default_method = DEFAULT_METHOD[model].value
        metrics_summary: dict[str, Any] = {}

        for metric_name, config in metric_configs.items():
            score_column = config["column"]
            order_direction = config["order"]
            cursor.execute(
                f"""SELECT layer, {score_column} FROM aggregate_metrics
                    WHERE model = ? AND method = ? AND percentile = 90 AND {score_column} IS NOT NULL
                    ORDER BY {score_column} {order_direction}, CAST(SUBSTR(layer, 6) AS INTEGER) ASC
                    LIMIT 1""",
                (model, default_method),
            )
            best_row = cursor.fetchone()
            best_layer = best_row[0] if best_row else "layer11"
            best_score = best_row[1] if best_row else 0

            cursor.execute(
                f"""SELECT layer, {score_column} FROM aggregate_metrics
                    WHERE model = ? AND method = ? AND percentile = 90 AND {score_column} IS NOT NULL
                    ORDER BY CAST(SUBSTR(layer, 6) AS INTEGER)""",
                (model, default_method),
            )
            layer_progression = {row[0]: row[1] for row in cursor.fetchall()}
            metrics_summary[metric_name] = {
                "best_layer": best_layer,
                "best_score": best_score,
                "method_used": default_method,
                "layer_progression": layer_progression,
            }

        iou_summary = metrics_summary["iou"]
        models_data[display_model_name(model)] = {
            "best_layer": iou_summary["best_layer"],
            "best_iou": iou_summary["best_score"],
            "method_used": default_method,
            "layer_progression": iou_summary["layer_progression"],
            "metrics": metrics_summary,
        }

    summary = {
        "ranking_mode": "default_method",
        "models": models_data,
        "leaderboard": leaderboard,
        "leaderboards": leaderboards,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"Exported summary to {output_path}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate metrics cache")
    parser.add_argument(
        "--models",
        nargs="+",
        default=["all"],
        help="Models to process (or 'all')",
    )
    parser.add_argument(
        "--layers",
        nargs="+",
        type=int,
        default=None,
        help="Specific layers to process (default: all 12)",
    )
    parser.add_argument(
        "--attention-cache",
        type=Path,
        default=CACHE_PATH / "attention_viz.h5",
        help="Path to attention cache HDF5",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=CACHE_PATH / "metrics.db",
        help="Path to metrics SQLite database",
    )
    parser.add_argument(
        "--percentiles",
        nargs="+",
        type=int,
        default=DEFAULT_PERCENTILES,
        help="Percentile thresholds to compute",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        default=None,
        help="Specific methods to process (e.g., cls rollout). Default: all for each model.",
    )
    parser.add_argument(
        "--no-skip",
        action="store_true",
        help="Don't skip existing database entries",
    )
    args = parser.parse_args()

    # Validate
    if not args.attention_cache.exists():
        print(f"Error: Attention cache not found: {args.attention_cache}")
        print("Run generate_attention_cache.py first.")
        return 1

    # Determine models to process
    if "all" in args.models:
        models_to_process = list(MODELS.keys())
    else:
        models_to_process = [m for m in args.models if m in MODELS]

    if not models_to_process:
        print("No valid models specified")
        return 1

    # Setup
    dataset = AnnotatedSubset(DATASET_PATH)
    print(f"Dataset: {len(dataset)} annotated images")

    attention_cache = AttentionCache(args.attention_cache)
    print(f"Attention cache: {args.attention_cache}")

    args.db_path.parent.mkdir(parents=True, exist_ok=True)
    conn = create_database(args.db_path)
    print(f"Database: {args.db_path}")

    print(f"Percentiles: {args.percentiles}")

    # Process each model
    total_stats = {"processed": 0, "skipped": 0, "errors": 0}

    for model_name in models_to_process:
        stats = compute_metrics_for_model(
            model_name=model_name,
            dataset=dataset,
            attention_cache=attention_cache,
            conn=conn,
            percentiles=args.percentiles,
            layers=args.layers,
            methods=args.methods,
            skip_existing=not args.no_skip,
        )

        for key in total_stats:
            total_stats[key] += stats[key]

        print(f"{model_name} complete: {stats}")

    # Export summary JSON
    export_summary_json(conn, args.db_path.parent / "metrics_summary.json")

    conn.close()

    print(f"\n{'='*60}")
    print("SUMMARY")
    print(f"{'='*60}")
    print(f"Total processed: {total_stats['processed']}")
    print(f"Total skipped: {total_stats['skipped']}")
    print(f"Total errors: {total_stats['errors']}")

    return 0 if total_stats["errors"] == 0 else 1


if __name__ == "__main__":
    sys.exit(main())

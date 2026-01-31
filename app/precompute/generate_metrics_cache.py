"""Generate pre-computed metrics cache for the visualization app.

Computes IoU and coverage metrics for all model/layer/image combinations
at multiple percentile thresholds, storing results in SQLite for fast queries.

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

from ssl_attention.cache import AttentionCache
from ssl_attention.config import CACHE_PATH, DATASET_PATH, MODELS, STYLE_MAPPING, STYLE_NAMES
from ssl_attention.data import AnnotatedSubset
from ssl_attention.metrics import compute_image_iou

DEFAULT_PERCENTILES = [90, 85, 80, 75, 70, 60, 50]


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
            image_id TEXT NOT NULL,
            percentile INTEGER NOT NULL,
            iou REAL NOT NULL,
            coverage REAL NOT NULL,
            attention_area REAL NOT NULL,
            annotation_area REAL NOT NULL,
            UNIQUE(model, layer, image_id, percentile)
        )
    """)

    # Aggregate metrics table (per model/layer)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS aggregate_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            layer TEXT NOT NULL,
            percentile INTEGER NOT NULL,
            mean_iou REAL NOT NULL,
            std_iou REAL NOT NULL,
            median_iou REAL NOT NULL,
            mean_coverage REAL NOT NULL,
            num_images INTEGER NOT NULL,
            UNIQUE(model, layer, percentile)
        )
    """)

    # Style breakdown table
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS style_metrics (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            model TEXT NOT NULL,
            layer TEXT NOT NULL,
            style_name TEXT NOT NULL,
            percentile INTEGER NOT NULL,
            mean_iou REAL NOT NULL,
            num_images INTEGER NOT NULL,
            UNIQUE(model, layer, style_name, percentile)
        )
    """)

    # Model leaderboard view (best layer per model)
    cursor.execute("""
        CREATE VIEW IF NOT EXISTS model_leaderboard AS
        SELECT
            model,
            MAX(mean_iou) as best_iou,
            (SELECT layer FROM aggregate_metrics a2
             WHERE a2.model = a1.model AND a2.percentile = 90
             ORDER BY mean_iou DESC LIMIT 1) as best_layer
        FROM aggregate_metrics a1
        WHERE percentile = 90
        GROUP BY model
        ORDER BY best_iou DESC
    """)

    # Indexes for fast queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_image_model_layer ON image_metrics(model, layer)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_image_image_id ON image_metrics(image_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_agg_model ON aggregate_metrics(model)")

    conn.commit()
    return conn


def compute_metrics_for_model(
    model_name: str,
    dataset: AnnotatedSubset,
    attention_cache: AttentionCache,
    conn: sqlite3.Connection,
    percentiles: list[int],
    layers: list[int] | None = None,
    skip_existing: bool = True,
) -> dict[str, int]:
    """Compute metrics for a single model.

    Args:
        model_name: Name of model.
        dataset: Annotated dataset.
        attention_cache: AttentionCache with pre-computed attention.
        conn: SQLite connection.
        percentiles: List of percentile thresholds.
        layers: Specific layers to process. None = all.
        skip_existing: Skip if already in database.

    Returns:
        Dict with statistics.
    """
    import torch

    stats = {"processed": 0, "skipped": 0, "errors": 0}

    model_config = MODELS[model_name]
    num_layers = model_config.num_layers
    layers_to_process = layers if layers else list(range(num_layers))

    print(f"\nProcessing {model_name} ({len(layers_to_process)} layers)")

    cursor = conn.cursor()

    # Build style mapping for images
    image_styles: dict[str, str | None] = {}
    for sample in dataset:
        image_id = sample["image_id"]
        annotation = sample["annotation"]
        style_name = None
        for style_qid in annotation.styles:
            if style_qid in STYLE_MAPPING:
                style_name = STYLE_NAMES[STYLE_MAPPING[style_qid]]
                break
        image_styles[image_id] = style_name

    # Collect all results first, then compute aggregates
    all_results: dict[tuple[str, str, int], list] = {}  # (model, layer, percentile) -> list of IoUResults

    for layer in layers_to_process:
        layer_key = f"layer{layer}"

        for percentile in percentiles:
            key = (model_name, layer_key, percentile)
            all_results[key] = []

    # Process each image
    for sample in tqdm(dataset, desc=f"{model_name}"):
        image_id = sample["image_id"]
        annotation = sample["annotation"]

        for layer in layers_to_process:
            layer_key = f"layer{layer}"

            # Load attention from cache
            try:
                attention = attention_cache.load(model_name, layer_key, image_id)
            except KeyError:
                stats["skipped"] += len(percentiles)
                continue

            for percentile in percentiles:
                # Check if already exists
                if skip_existing:
                    cursor.execute(
                        "SELECT 1 FROM image_metrics WHERE model=? AND layer=? AND image_id=? AND percentile=?",
                        (model_name, layer_key, image_id, percentile),
                    )
                    if cursor.fetchone():
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
                           (model, layer, image_id, percentile, iou, coverage, attention_area, annotation_area)
                           VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                        (
                            model_name,
                            layer_key,
                            image_id,
                            percentile,
                            result.iou,
                            result.coverage,
                            result.attention_area,
                            result.annotation_area,
                        ),
                    )

                    all_results[(model_name, layer_key, percentile)].append(result)
                    stats["processed"] += 1

                except Exception as e:
                    print(f"\nError computing metrics for {image_id} layer{layer}: {e}")
                    stats["errors"] += 1

    conn.commit()

    # Compute aggregate metrics
    print(f"Computing aggregates for {model_name}...")
    for layer in layers_to_process:
        layer_key = f"layer{layer}"

        for percentile in percentiles:
            # Fetch all results from DB for this configuration
            cursor.execute(
                "SELECT iou, coverage FROM image_metrics WHERE model=? AND layer=? AND percentile=?",
                (model_name, layer_key, percentile),
            )
            rows = cursor.fetchall()

            if not rows:
                continue

            ious = torch.tensor([r[0] for r in rows])
            coverages = torch.tensor([r[1] for r in rows])

            cursor.execute(
                """INSERT OR REPLACE INTO aggregate_metrics
                   (model, layer, percentile, mean_iou, std_iou, median_iou, mean_coverage, num_images)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (
                    model_name,
                    layer_key,
                    percentile,
                    ious.mean().item(),
                    ious.std().item(),
                    ious.median().item(),
                    coverages.mean().item(),
                    len(rows),
                ),
            )

    conn.commit()

    # Compute style breakdown
    print(f"Computing style breakdown for {model_name}...")
    for layer in layers_to_process:
        layer_key = f"layer{layer}"

        for percentile in percentiles:
            for style_name in STYLE_NAMES:
                # Get image IDs for this style
                style_images = [img_id for img_id, s in image_styles.items() if s == style_name]
                if not style_images:
                    continue

                placeholders = ",".join("?" * len(style_images))
                cursor.execute(
                    f"""SELECT AVG(iou), COUNT(*) FROM image_metrics
                       WHERE model=? AND layer=? AND percentile=? AND image_id IN ({placeholders})""",
                    (model_name, layer_key, percentile, *style_images),
                )
                row = cursor.fetchone()

                if row and row[0] is not None:
                    cursor.execute(
                        """INSERT OR REPLACE INTO style_metrics
                           (model, layer, style_name, percentile, mean_iou, num_images)
                           VALUES (?, ?, ?, ?, ?, ?)""",
                        (model_name, layer_key, style_name, percentile, row[0], row[1]),
                    )

    conn.commit()

    return stats


def export_summary_json(conn: sqlite3.Connection, output_path: Path) -> None:
    """Export summary statistics to JSON for fast frontend loading."""
    from typing import Any

    cursor = conn.cursor()

    models_data: dict[str, Any] = {}
    leaderboard: list[dict[str, Any]] = []

    # Get leaderboard
    cursor.execute(
        """SELECT model, MAX(mean_iou) as best_iou
           FROM aggregate_metrics WHERE percentile = 90
           GROUP BY model ORDER BY best_iou DESC"""
    )
    for row in cursor.fetchall():
        leaderboard.append({"model": row[0], "best_iou": row[1]})

    # Get per-model summaries
    cursor.execute("SELECT DISTINCT model FROM aggregate_metrics")
    models = [row[0] for row in cursor.fetchall()]

    for model in models:
        # Best layer at 90th percentile
        cursor.execute(
            """SELECT layer, mean_iou FROM aggregate_metrics
               WHERE model = ? AND percentile = 90 ORDER BY mean_iou DESC LIMIT 1""",
            (model,),
        )
        row = cursor.fetchone()
        best_layer = row[0] if row else "layer11"
        best_iou = row[1] if row else 0

        # Layer progression
        cursor.execute(
            """SELECT layer, mean_iou FROM aggregate_metrics
               WHERE model = ? AND percentile = 90 ORDER BY layer""",
            (model,),
        )
        layer_progression = {row[0]: row[1] for row in cursor.fetchall()}

        models_data[model] = {
            "best_layer": best_layer,
            "best_iou": best_iou,
            "layer_progression": layer_progression,
        }

    summary = {"models": models_data, "leaderboard": leaderboard}

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
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

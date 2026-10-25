#!/usr/bin/env python3
"""Frozen-model IoU@90 versus annotation count, annotation size and feature category (R1-Q5).

Reads frozen-model rows already in metrics.db and the expert annotations in
building_parts.json. It runs no models and writes nothing except its JSON.

For each frozen model at its best default-method IoU@90 layer (the Table 3
setting), it reports:
  - Spearman correlations of per-image IoU@90 with the number of boxes and
    with the union-mask area, with Holm correction across the six models
  - box-weighted IoU@90 per top-level ontology category
It also summarizes how the annotation boxes are grouped and overlap, and the
attainable IoU@90 of the features in the last row of Table 4.

Outputs:
  - Console summary
  - JSON artifact: outputs/results/q5_annotation_size_alignment.json

Usage:
  uv run python experiments/scripts/analyze_annotation_size_alignment.py
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from collections import Counter
from contextlib import closing
from datetime import UTC, datetime
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from ssl_attention.config import (  # noqa: E402
    ANNOTATIONS_PATH,
    CACHE_PATH,
    DEFAULT_IMAGE_SIZE,
    DEFAULT_METHOD,
)
from ssl_attention.data.annotations import (  # noqa: E402
    BoundingBox,
    FeatureType,
    ImageAnnotation,
    load_annotations_with_features,
)
from ssl_attention.metrics.statistics import multiple_comparison_correction  # noqa: E402

MODEL_ORDER = ("dinov2", "dinov3", "mae", "clip", "siglip", "siglip2")
PERCENTILE = 90
MIN_CATEGORY_BOXES = 20
CONTAINMENT_TOLERANCE = 1e-9

# IoU@90 column of Table 3 in round2_v1024_changes.tex, as printed.
TABLE3_IOU90 = {
    "dinov2": 0.118,
    "dinov3": 0.133,
    "mae": 0.070,
    "clip": 0.051,
    "siglip": 0.074,
    "siglip2": 0.047,
}
TABLE4_WEAKEST_FEATURES = ("Crocket", "Fleuron", "Pinnacle")

DEFAULT_METRICS_DB = CACHE_PATH / "metrics.db"
DEFAULT_OUTPUT = project_root / "outputs" / "results" / "q5_annotation_size_alignment.json"


# ---------------------------------------------------------------------------
# Annotations
# ---------------------------------------------------------------------------

def box_area(bbox: BoundingBox) -> float:
    """Fraction of the evaluation grid covered by the rasterized box."""
    return float(bbox.to_mask(DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE).float().mean())


def union_area(annotation: ImageAnnotation) -> float:
    """Fraction of the evaluation grid covered by the image's union mask."""
    return float(annotation.get_union_mask(DEFAULT_IMAGE_SIZE, DEFAULT_IMAGE_SIZE).float().mean())


def root_categories(feature_types: tuple[FeatureType, ...]) -> list[frozenset[str]]:
    """Top-level ontology types that each feature type descends from (itself if it has no parents)."""

    def ancestors(index: int, seen: set[int]) -> set[int]:
        for parent in feature_types[index].parent_indices:
            if parent not in seen:
                seen.add(parent)
                ancestors(parent, seen)
        return seen

    return [
        frozenset(
            feature_types[i].name
            for i in {ft.index} | ancestors(ft.index, set())
            if not feature_types[i].parent_indices
        )
        for ft in feature_types
    ]


def boxes_overlap(a: BoundingBox, b: BoundingBox) -> bool:
    return min(a.right, b.right) > max(a.left, b.left) and min(a.bottom, b.bottom) > max(a.top, b.top)


def box_contains(outer: BoundingBox, inner: BoundingBox) -> bool:
    # Shared edges can differ by float rounding of left + width, so allow a sub-pixel tolerance.
    return (
        outer.left <= inner.left + CONTAINMENT_TOLERANCE
        and outer.top <= inner.top + CONTAINMENT_TOLERANCE
        and outer.right >= inner.right - CONTAINMENT_TOLERANCE
        and outer.bottom >= inner.bottom - CONTAINMENT_TOLERANCE
    )


def summarize_annotations(annotations: dict[str, ImageAnnotation], raw: dict[str, Any]) -> dict[str, Any]:
    """Count groups, boxes and overlapping box pairs in normalized coordinates."""
    groups = [group for ann in raw["annotations"].values() for group in ann.get("bbox_groups", [])]
    pairs = overlapping = nested = images_with_overlap = 0
    for annotation in annotations.values():
        image_overlaps = 0
        for a, b in combinations(annotation.bboxes, 2):
            pairs += 1
            if boxes_overlap(a, b):
                image_overlaps += 1
                nested += box_contains(a, b) or box_contains(b, a)
        overlapping += image_overlaps
        images_with_overlap += image_overlaps > 0

    counts = [annotation.num_bboxes for annotation in annotations.values()]
    areas = [union_area(annotation) for annotation in annotations.values()]
    return {
        "images": len(annotations),
        "boxes": sum(counts),
        "groups": len(groups),
        "multi_box_groups": sum(len(group.get("elements", [])) > 1 for group in groups),
        "same_image_box_pairs": pairs,
        "overlapping_pairs": overlapping,
        "nested_pairs": nested,
        "images_with_overlap": images_with_overlap,
        "boxes_per_image": {"min": min(counts), "median": float(np.median(counts)), "max": max(counts)},
        "union_area_fraction": {
            "min": float(np.min(areas)),
            "median": float(np.median(areas)),
            "max": float(np.max(areas)),
        },
    }


# ---------------------------------------------------------------------------
# metrics.db
# ---------------------------------------------------------------------------

def select_best_layer(conn: sqlite3.Connection, model: str, method: str) -> tuple[str, float]:
    rows = conn.execute(
        "SELECT layer, AVG(iou) FROM image_metrics WHERE model=? AND method=? AND percentile=? GROUP BY layer",
        (model, method, PERCENTILE),
    ).fetchall()
    if not rows:
        raise SystemExit(f"{model}: no {method} IoU@{PERCENTILE} rows in image_metrics")
    layer, mean_iou = max(rows, key=lambda row: row[1])
    return str(layer), float(mean_iou)


def load_image_rows(
    conn: sqlite3.Connection, model: str, method: str, layer: str
) -> dict[str, tuple[float, float, float]]:
    """Return {image_id: (iou, annotation_area, attention_area)}."""
    rows = conn.execute(
        """SELECT image_id, iou, annotation_area, attention_area FROM image_metrics
           WHERE model=? AND layer=? AND method=? AND percentile=?""",
        (model, layer, method, PERCENTILE),
    ).fetchall()
    return {str(r[0]): (float(r[1]), float(r[2]), float(r[3])) for r in rows}


def load_feature_rows(conn: sqlite3.Connection, model: str, method: str, layer: str) -> dict[int, tuple[float, int]]:
    """Return {feature_label: (mean per-box IoU@90, box count)}."""
    rows = conn.execute(
        """SELECT feature_label, mean_score, bbox_count FROM feature_metrics
           WHERE model=? AND layer=? AND method=? AND metric='iou' AND percentile=?""",
        (model, layer, method, PERCENTILE),
    ).fetchall()
    return {int(r[0]): (float(r[1]), int(r[2])) for r in rows}


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def spearman(x: list[float], y: list[float]) -> dict[str, float]:
    result = stats.spearmanr(x, y)
    return {"rho": float(result.statistic), "p": float(result.pvalue)}


def analyze(conn: sqlite3.Connection) -> dict[str, Any]:
    annotations, feature_types = load_annotations_with_features(ANNOTATIONS_PATH)
    with open(ANNOTATIONS_PATH, encoding="utf-8") as f:
        raw = json.load(f)

    image_ids = sorted(annotations)
    union_areas = {image_id: union_area(annotations[image_id]) for image_id in image_ids}
    box_counts = [annotations[image_id].num_bboxes for image_id in image_ids]
    all_boxes = [bbox for annotation in annotations.values() for bbox in annotation.bboxes]
    boxes_per_label = dict(Counter(bbox.label for bbox in all_boxes))

    roots = root_categories(feature_types)
    category_areas: dict[str, list[float]] = {}
    for bbox in all_boxes:
        for category in roots[bbox.label]:
            category_areas.setdefault(category, []).append(box_area(bbox))
    categories = sorted(
        (name for name, areas in category_areas.items() if len(areas) >= MIN_CATEGORY_BOXES),
        key=lambda name: -len(category_areas[name]),
    )

    models: dict[str, dict[str, Any]] = {}
    category_iou: dict[str, dict[str, float]] = {name: {} for name in categories}
    feature_rows_by_model: dict[str, dict[int, tuple[float, int]]] = {}
    attention_fractions: set[float] = set()

    for model in MODEL_ORDER:
        method = DEFAULT_METHOD[model].value
        layer, mean_iou = select_best_layer(conn, model, method)
        if abs(mean_iou - TABLE3_IOU90[model]) > 5e-4:
            raise SystemExit(
                f"{model}: best {method} IoU@90 is {mean_iou:.4f} at {layer}, "
                f"but Table 3 reports {TABLE3_IOU90[model]:.3f}"
            )

        image_rows = load_image_rows(conn, model, method, layer)
        if set(image_rows) != set(image_ids):
            raise SystemExit(f"{model}/{layer}: image_metrics images differ from the {len(image_ids)} annotated images")
        for image_id, (_, annotation_area, attention_area) in image_rows.items():
            if abs(annotation_area - union_areas[image_id]) > 1e-6:
                raise SystemExit(f"{model}/{layer}/{image_id}: annotation_area does not match the union mask")
            attention_fractions.add(round(attention_area, 6))

        ious = [image_rows[image_id][0] for image_id in image_ids]
        models[model] = {
            "method": method,
            "layer": layer,
            "mean_iou90": mean_iou,
            "union_area": spearman(ious, [union_areas[image_id] for image_id in image_ids]),
            "box_count": spearman(ious, [float(n) for n in box_counts]),
        }

        feature_rows = load_feature_rows(conn, model, method, layer)
        if {label: count for label, (_, count) in feature_rows.items()} != boxes_per_label:
            raise SystemExit(f"{model}/{layer}: feature_metrics box counts differ from the annotations")
        feature_rows_by_model[model] = feature_rows
        for name in categories:
            selected = [(score, count) for label, (score, count) in feature_rows.items() if name in roots[label]]
            category_iou[name][model] = sum(s * n for s, n in selected) / sum(n for _, n in selected)

    for key in ("union_area", "box_count"):
        corrected = multiple_comparison_correction([models[m][key]["p"] for m in MODEL_ORDER], method="holm")
        for model, (p_holm, _) in zip(MODEL_ORDER, corrected, strict=True):
            models[model][key]["p_holm"] = p_holm

    if len(attention_fractions) != 1:
        raise SystemExit(f"Expected one top-10% attention fraction, found {sorted(attention_fractions)}")
    attention_fraction = attention_fractions.pop()

    label_by_name = {ft.name: ft.index for ft in feature_types}
    dinov3_features = feature_rows_by_model["dinov3"]
    table4_weakest = []
    for name in TABLE4_WEAKEST_FEATURES:
        label = label_by_name[name]
        median_area = float(np.median([box_area(bbox) for bbox in all_boxes if bbox.label == label]))
        table4_weakest.append({
            "name": name,
            "boxes": dinov3_features[label][1],
            "median_box_area": median_area,
            "max_iou90_at_median_area": min(median_area, attention_fraction) / max(median_area, attention_fraction),
            "dinov3_iou90": dinov3_features[label][0],
        })

    annotation_summary = summarize_annotations(annotations, raw)
    annotation_summary["top10_attention_fraction"] = attention_fraction
    annotation_summary["images_with_union_below_top10_fraction"] = sum(
        area < attention_fraction for area in union_areas.values()
    )

    return {
        "annotations": annotation_summary,
        "models": models,
        "categories": [
            {
                "name": name,
                "boxes": len(category_areas[name]),
                "median_box_area": float(np.median(category_areas[name])),
                "iou90": category_iou[name],
            }
            for name in categories
        ],
        "table4_weakest_features": table4_weakest,
    }


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_summary(results: dict[str, Any]) -> None:
    summary = results["annotations"]
    print(
        f"\n{summary['boxes']} boxes in {summary['groups']} groups ({summary['multi_box_groups']} with several boxes); "
        f"{summary['overlapping_pairs']}/{summary['same_image_box_pairs']} same-image pairs overlap, "
        f"{summary['nested_pairs']} nested, in {summary['images_with_overlap']} images"
    )

    print(f"\n{'Model':8s} {'Layer':8s} {'IoU@90':>7s} {'rho area':>9s} {'Holm p':>9s} {'rho count':>10s} {'Holm p':>9s}")
    for model, row in results["models"].items():
        area, count = row["union_area"], row["box_count"]
        print(
            f"{model:8s} {row['layer']:8s} {row['mean_iou90']:7.4f} {area['rho']:+9.2f} {area['p_holm']:9.1e} "
            f"{count['rho']:+10.2f} {count['p_holm']:9.1e}"
        )

    print(f"\n{'Category':10s} {'Boxes':>5s} {'Median':>7s} " + " ".join(f"{m:>7s}" for m in MODEL_ORDER))
    for category in results["categories"]:
        print(
            f"{category['name']:10s} {category['boxes']:5d} {100 * category['median_box_area']:6.2f}% "
            + " ".join(f"{category['iou90'][m]:7.3f}" for m in MODEL_ORDER)
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen IoU@90 versus annotation count, size and feature category (R1-Q5)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("--metrics-db", type=Path, default=DEFAULT_METRICS_DB, help="Path to metrics.db.")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="Output JSON path.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    with closing(sqlite3.connect(f"file:{args.metrics_db}?mode=ro", uri=True)) as conn:
        results = analyze(conn)

    results = {
        "generated_at": datetime.now(UTC).isoformat(),
        "inputs": {
            "metrics_db": os.path.relpath(args.metrics_db, project_root),
            "metrics_db_modified": datetime.fromtimestamp(args.metrics_db.stat().st_mtime, UTC).isoformat(),
            "annotations": os.path.relpath(ANNOTATIONS_PATH, project_root),
        },
        "setting": {
            "metric": "iou",
            "percentile": PERCENTILE,
            "layer": "best mean IoU@90 layer with the default method (Table 3)",
            "min_category_boxes": MIN_CATEGORY_BOXES,
        },
        **results,
    }
    print_summary(results)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print(f"\nSaved {args.output}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Paired image-level test of DINOv3's frozen lead over the next-best model (Table RQ1 DINOv3).

For each metric where DINOv3 ranks first, every frozen model is placed at its best
default-method layer (the rule behind metrics_summary.json). DINOv3 is then compared
with the runner-up on the 139 per-image scores using a Wilcoxon signed-rank test, with
Holm correction across those metrics.
"""

from __future__ import annotations

import argparse
import json
import sqlite3
from pathlib import Path
from typing import Any

import numpy as np

from ssl_attention.config import DEFAULT_METHOD
from ssl_attention.metrics.statistics import multiple_comparison_correction, wilcoxon_signed_rank

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "outputs" / "cache" / "metrics.db"
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "results" / "rq1_dinov3_paired_gap.json"
MODELS = ["dinov2", "dinov3", "mae", "clip", "siglip", "siglip2"]
PERCENTILE = 90

# metric -> (aggregate column, per-image column, higher is better)
METRICS = {
    "iou": ("mean_iou", "iou", True),
    "coverage": ("mean_coverage", "coverage", True),
    "mse": ("mean_mse", "mse", False),
    "kl": ("mean_kl", "kl", False),
    "emd": ("mean_emd", "emd", False),
}


def best_layer(conn: sqlite3.Connection, model: str, column: str, higher: bool) -> tuple[str, float]:
    order = "DESC" if higher else "ASC"
    layer, score = conn.execute(
        f"SELECT layer, {column} FROM aggregate_metrics "
        f"WHERE model = ? AND method = ? AND percentile = ? AND {column} IS NOT NULL "
        f"ORDER BY {column} {order}, CAST(SUBSTR(layer, 6) AS INTEGER) ASC LIMIT 1",
        (model, DEFAULT_METHOD[model].value, PERCENTILE),
    ).fetchone()
    return str(layer), float(score)


def per_image(conn: sqlite3.Connection, model: str, layer: str, column: str) -> dict[str, float]:
    rows = conn.execute(
        f"SELECT image_id, {column} FROM image_metrics "
        "WHERE model = ? AND layer = ? AND method = ? AND percentile = ?",
        (model, layer, DEFAULT_METHOD[model].value, PERCENTILE),
    ).fetchall()
    return {image_id: float(value) for image_id, value in rows}


def analyze(conn: sqlite3.Connection) -> dict[str, Any]:
    comparisons: list[dict[str, Any]] = []
    for metric, (agg_column, image_column, higher) in METRICS.items():
        best = {model: best_layer(conn, model, agg_column, higher) for model in MODELS}
        ranked = sorted(MODELS, key=lambda m: best[m][1], reverse=higher)
        if ranked[0] != "dinov3":
            continue
        runner_up = ranked[1]
        a = per_image(conn, "dinov3", best["dinov3"][0], image_column)
        b = per_image(conn, runner_up, best[runner_up][0], image_column)
        image_ids = sorted(a.keys() & b.keys())
        a_values = np.array([a[i] for i in image_ids])
        b_values = np.array([b[i] for i in image_ids])
        _stat, p_value = wilcoxon_signed_rank(a_values, b_values)
        sign = 1.0 if higher else -1.0
        comparisons.append({
            "metric": metric,
            "dinov3_layer": best["dinov3"][0],
            "dinov3_score": best["dinov3"][1],
            "runner_up": runner_up,
            "runner_up_layer": best[runner_up][0],
            "runner_up_score": best[runner_up][1],
            "gap": sign * (best["dinov3"][1] - best[runner_up][1]),
            "dinov3_wins": int(np.sum(sign * (a_values - b_values) > 0)),
            "num_images": len(image_ids),
            "p_value": p_value,
        })
    corrected = multiple_comparison_correction([c["p_value"] for c in comparisons], method="holm")
    for comparison, (p_holm, significant) in zip(comparisons, corrected, strict=True):
        comparison["p_holm"] = p_holm
        comparison["significant"] = significant
    return {"percentile": PERCENTILE, "test": "Wilcoxon signed-rank, Holm", "comparisons": comparisons}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db-path", type=Path, default=DB_PATH)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    with sqlite3.connect(f"file:{args.db_path}?mode=ro", uri=True) as conn:
        results = analyze(conn)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")
    for c in results["comparisons"]:
        print(
            f"{c['metric']:8s} DINOv3 {c['dinov3_layer']} {c['dinov3_score']:.4f} vs {c['runner_up']} "
            f"{c['runner_up_layer']} {c['runner_up_score']:.4f}  gap {c['gap']:+.4f}  "
            f"wins {c['dinov3_wins']}/{c['num_images']}  p_holm {c['p_holm']:.3g}"
        )
    print(args.output)


if __name__ == "__main__":
    main()

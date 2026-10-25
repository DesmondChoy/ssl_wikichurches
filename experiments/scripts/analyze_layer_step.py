#!/usr/bin/env python3
"""Frozen-model alignment change from layer 6 to layer 7 (R1-Q15).

Reads frozen-model rows already in metrics.db. It runs no models and writes
nothing except its JSON.

For each frozen model with its default attention method (the Figure 3
setting), it reports:
  - mean IoU@90 at every layer and the largest change between consecutive layers
  - for the layer 6 to 7 step: the mean of each metric at both layers, the share
    of images that improve, and a two-sided Wilcoxon signed-rank test, with Holm
    correction across all models and metrics
  - mean IoU at layers 6 and 7 for every stored percentile
  - the number of heads whose mean IoU@90 rises from layer 6 to layer 7

Outputs:
  - Console summary
  - JSON artifact: outputs/results/q15_layer_step.json

Usage:
  uv run python experiments/scripts/analyze_layer_step.py
"""

from __future__ import annotations

import argparse
import json
import os
import sqlite3
import sys
from contextlib import closing
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
from scipy import stats

project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root / "src"))

from ssl_attention.config import CACHE_PATH, DEFAULT_METHOD  # noqa: E402
from ssl_attention.metrics.statistics import multiple_comparison_correction  # noqa: E402

MODEL_ORDER = ("dinov2", "dinov3", "mae", "clip", "siglip", "siglip2")
PERCENTILE = 90
NUM_LAYERS = 12
STEP = (6, 7)
# image_metrics column and whether a higher value means better alignment
METRICS = (("iou", True), ("coverage", True), ("mse", False), ("kl", False), ("emd", False))

DEFAULT_METRICS_DB = CACHE_PATH / "metrics.db"
DEFAULT_OUTPUT = project_root / "outputs" / "results" / "q15_layer_step.json"


# ---------------------------------------------------------------------------
# metrics.db
# ---------------------------------------------------------------------------

def load_layer_values(conn: sqlite3.Connection, model: str, method: str, layer: int) -> dict[str, np.ndarray]:
    """Return {metric: per-image values} at one layer, ordered by image_id."""
    columns = ", ".join(name for name, _ in METRICS)
    rows = conn.execute(
        f"""SELECT image_id, {columns} FROM image_metrics
            WHERE model=? AND layer=? AND method=? AND percentile=? ORDER BY image_id""",
        (model, f"layer{layer}", method, PERCENTILE),
    ).fetchall()
    if not rows:
        raise SystemExit(f"{model}/layer{layer}: no {method} rows in image_metrics")
    values = {name: np.array([row[i + 1] for row in rows], dtype=float) for i, (name, _) in enumerate(METRICS)}
    values["image_id"] = np.array([row[0] for row in rows])
    return values


def load_percentile_iou(conn: sqlite3.Connection, model: str, method: str, layer: int) -> dict[int, float]:
    rows = conn.execute(
        "SELECT percentile, mean_iou FROM aggregate_metrics WHERE model=? AND layer=? AND method=? ORDER BY percentile",
        (model, f"layer{layer}", method),
    ).fetchall()
    return {int(p): float(v) for p, v in rows}


def load_head_iou(conn: sqlite3.Connection, model: str, method: str, layer: int) -> list[float]:
    rows = conn.execute(
        """SELECT mean_score FROM head_summary_metrics
           WHERE model=? AND layer=? AND method=? AND metric='iou' AND percentile=? ORDER BY head""",
        (model, f"layer{layer}", method, PERCENTILE),
    ).fetchall()
    return [float(r[0]) for r in rows]


# ---------------------------------------------------------------------------
# Analysis
# ---------------------------------------------------------------------------

def analyze_model(conn: sqlite3.Connection, model: str) -> dict[str, Any]:
    method = DEFAULT_METHOD[model].value
    layers = [load_layer_values(conn, model, method, layer) for layer in range(NUM_LAYERS)]
    for layer, values in enumerate(layers):
        if not np.array_equal(values["image_id"], layers[0]["image_id"]):
            raise SystemExit(f"{model}/layer{layer}: image set differs from layer0")
        aggregate = conn.execute(
            "SELECT mean_iou FROM aggregate_metrics WHERE model=? AND layer=? AND method=? AND percentile=?",
            (model, f"layer{layer}", method, PERCENTILE),
        ).fetchone()
        if aggregate is None or abs(float(aggregate[0]) - values["iou"].mean()) > 5e-4:
            raise SystemExit(f"{model}/layer{layer}: image_metrics mean IoU@90 differs from aggregate_metrics")

    curve = [float(values["iou"].mean()) for values in layers]
    changes = np.diff(curve)
    largest = int(np.argmax(np.abs(changes)))
    second = max((abs(c), i) for i, c in enumerate(changes) if i != largest)

    before, after = layers[STEP[0]], layers[STEP[1]]
    step_metrics = {}
    for name, higher_is_better in METRICS:
        diff = after[name] - before[name]
        gain = diff if higher_is_better else -diff
        step_metrics[name] = {
            "higher_is_better": higher_is_better,
            "mean_before": float(before[name].mean()),
            "mean_after": float(after[name].mean()),
            "share_improved": float(np.mean(gain > 0)),
            "share_worsened": float(np.mean(gain < 0)),
            "p": float(stats.wilcoxon(after[name], before[name]).pvalue),
        }

    percentiles_before = load_percentile_iou(conn, model, method, STEP[0])
    percentiles_after = load_percentile_iou(conn, model, method, STEP[1])

    heads_before = load_head_iou(conn, model, method, STEP[0])
    heads_after = load_head_iou(conn, model, method, STEP[1])
    heads = None
    if heads_before and heads_after:
        heads = {
            "count": len(heads_after),
            "increased": sum(a > b for a, b in zip(heads_after, heads_before, strict=True)),
            "best_before": max(heads_before),
            "best_after": max(heads_after),
        }

    return {
        "method": method,
        "images": len(before["iou"]),
        "mean_iou90_by_layer": curve,
        "largest_change": {"from_layer": largest, "change": float(changes[largest])},
        "second_largest_abs_change": {"from_layer": second[1], "change": float(changes[second[1]])},
        "step": step_metrics,
        "step_iou_by_percentile": {
            str(p): {"before": v, "after": percentiles_after[p]} for p, v in percentiles_before.items()
        },
        "step_heads": heads,
    }


def analyze(conn: sqlite3.Connection) -> dict[str, Any]:
    models = {model: analyze_model(conn, model) for model in MODEL_ORDER}
    keys = [(model, name) for model in MODEL_ORDER for name, _ in METRICS]
    corrected = multiple_comparison_correction([models[m]["step"][n]["p"] for m, n in keys], method="holm")
    for (model, name), (p_holm, _) in zip(keys, corrected, strict=True):
        models[model]["step"][name]["p_holm"] = p_holm
    return {"models": models}


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------

def print_summary(results: dict[str, Any]) -> None:
    for model, row in results["models"].items():
        largest, second = row["largest_change"], row["second_largest_abs_change"]
        print(
            f"\n{model} ({row['method']}): largest change L{largest['from_layer']}->L{largest['from_layer'] + 1} "
            f"{largest['change']:+.3f}; next largest L{second['from_layer']}->L{second['from_layer'] + 1} "
            f"{second['change']:+.3f}"
        )
        print(f"  {'Metric':8s} {'L6':>7s} {'L7':>7s} {'Better':>7s} {'Worse':>7s} {'Holm p':>9s}")
        for name, step in row["step"].items():
            print(
                f"  {name:8s} {step['mean_before']:7.3f} {step['mean_after']:7.3f} {step['share_improved']:7.0%} "
                f"{step['share_worsened']:7.0%} {step['p_holm']:9.1e}"
            )
        print("  IoU by percentile: " + "  ".join(
            f"p{p} {v['before']:.3f}->{v['after']:.3f}" for p, v in row["step_iou_by_percentile"].items()
        ))
        if row["step_heads"]:
            heads = row["step_heads"]
            print(
                f"  Heads with higher IoU@90 at L7: {heads['increased']}/{heads['count']}; "
                f"best head {heads['best_before']:.3f} -> {heads['best_after']:.3f}"
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Frozen-model alignment change from layer 6 to layer 7 (R1-Q15)",
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
        },
        "setting": {
            "percentile": PERCENTILE,
            "step": {"from_layer": STEP[0], "to_layer": STEP[1]},
            "test": "two-sided Wilcoxon signed-rank, Holm-adjusted across models and metrics",
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

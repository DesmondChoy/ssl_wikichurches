#!/usr/bin/env python3
"""Generate the paper's layer-wise frozen IoU@90 figure (RQ2_layer_iou_new.png).

Reads aggregate_metrics from metrics.db using each model's default attention method.
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / "outputs" / "cache" / "metrics.db"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "mva" / "RQ2_layer_iou_new.png"
PERCENTILE = 90
NUM_LAYERS = 12
MARKER = "o"

MODELS = [
    ("dinov2", "DINOv2", "cls", "#D7261E"),
    ("dinov3", "DINOv3", "cls", "#22D35A"),
    ("mae", "MAE", "cls", "#8A2BE2"),
    ("clip", "CLIP", "cls", "#D8C12A"),
    ("siglip", "SigLIP", "mean", "#1FB8D8"),
    ("siglip2", "SigLIP2", "mean", "#D6278A"),
]


def load_layer_iou(conn: sqlite3.Connection, model: str, method: str) -> list[float]:
    rows = dict(
        conn.execute(
            "select layer, mean_iou from aggregate_metrics where model = ? and method = ? and percentile = ?",
            (model, method, PERCENTILE),
        ).fetchall()
    )
    return [float(rows[f"layer{layer}"]) for layer in range(NUM_LAYERS)]


def main() -> None:
    with sqlite3.connect(DB_PATH) as conn:
        series = {key: load_layer_iou(conn, key, method) for key, _label, method, _color in MODELS}

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.4, 4.8))
    layers = list(range(NUM_LAYERS))
    for key, label, _method, color in MODELS:
        ax.plot(layers, series[key], marker=MARKER, markersize=5.5, linewidth=2.0, color=color, label=label)

    y_top = max(0.15, math.ceil(max(max(v) for v in series.values()) / 0.025) * 0.025)
    ax.set_ylim(0.0, y_top)
    ax.set_yticks([i * 0.025 for i in range(round(y_top / 0.025) + 1)])
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.3f"))
    ax.set_xticks(layers, [f"L{layer}" for layer in layers])
    ax.set_xlim(-0.3, NUM_LAYERS - 0.7)
    ax.set_xlabel("Transformer layer", fontsize=14)
    ax.set_ylabel(f"Mean IoU@{PERCENTILE} across 139 annotated images", fontsize=14)
    ax.tick_params(labelsize=14)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#CCCCCC")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=len(MODELS), frameon=False, fontsize=14)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=240, bbox_inches="tight", facecolor="white")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()

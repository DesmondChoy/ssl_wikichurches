#!/usr/bin/env python3
"""Generate the paper's layer-wise frozen IoU@90 figure with distinct markers.

Reads the verified layer-wise results from q15_layer_step.json.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[2]
RESULTS_PATH = PROJECT_ROOT / "outputs" / "results" / "q15_layer_step.json"
OUTPUT_PATH = PROJECT_ROOT / "docs" / "mva" / "RQ2_layer_iou_markers.png"
PERCENTILE = 90
NUM_LAYERS = 12

MARKERS = {
    "dinov2": "X",
    "dinov3": "s",
    "mae": "o",
    "clip": "D",
    "siglip": "v",
    "siglip2": "|",
}

MODELS = [
    ("dinov2", "DINOv2", "cls", "#2E45A8"),
    ("dinov3", "DINOv3", "cls", "#22D35A"),
    ("mae", "MAE", "cls", "#8A2BE2"),
    ("clip", "CLIP", "cls", "#D8C12A"),
    ("siglip", "SigLIP", "mean", "#1FB8D8"),
    ("siglip2", "SigLIP2", "mean", "#D6278A"),
]


def load_layer_iou(results: dict[str, Any], model: str, method: str) -> list[float]:
    model_result = results["models"][model]
    if model_result["method"] != method or model_result["images"] != 139:
        raise ValueError(f"Unexpected source metadata for {model}")
    values = model_result["mean_iou90_by_layer"]
    if len(values) != NUM_LAYERS:
        raise ValueError(f"Expected {NUM_LAYERS} layer values for {model}, found {len(values)}")
    return [float(value) for value in values]


def main() -> None:
    results = json.loads(RESULTS_PATH.read_text())
    series = {key: load_layer_iou(results, key, method) for key, _label, method, _color in MODELS}

    plt.style.use("seaborn-v0_8-whitegrid")
    fig = plt.figure(figsize=(8.4, 4.8))
    # Match the plotting bounds of the submitted reference image.
    ax = fig.add_axes((0.109, 0.245, 0.862, 0.715))
    layers = list(range(NUM_LAYERS))
    for key, label, _method, color in MODELS:
        ax.plot(
            layers,
            series[key],
            marker=MARKERS[key],
            markersize=7.0 if key == "siglip2" else 5.5,
            markeredgewidth=1.8 if key == "siglip2" else 1.0,
            linewidth=2.0,
            color=color,
            label=label,
        )

    y_top = max(0.15, math.ceil(max(max(v) for v in series.values()) / 0.025) * 0.025)
    ax.set_ylim(0.0, y_top)
    ax.set_yticks([i * 0.025 for i in range(round(y_top / 0.025) + 1)])
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.3f"))
    ax.set_xticks(layers, [f"L{layer}" for layer in layers])
    ax.set_xlim(-0.3, NUM_LAYERS - 0.7)
    ax.set_xlabel("Transformer layer", fontsize=14)
    ax.set_ylabel(f"Mean IoU@{PERCENTILE} across 139 annotated images", fontsize=13)
    ax.tick_params(labelsize=14)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#CCCCCC")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=len(MODELS), frameon=False, fontsize=12)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUTPUT_PATH, dpi=240, facecolor="white")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()

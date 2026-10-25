#!/usr/bin/env python3
"""Generate the R1-Q3 accuracy-versus-alignment figure (RQ3_accuracy_alignment.png).

Reads accuracy_alignment_analysis.json only. Each point is one run: x is the
classification accuracy on the 139 annotated images, y is the mean IoU@90 at
layer 11 on the same images. Colour marks the model. Frozen runs are filled
dots, LoRA runs are hollow model shapes, and Full runs are filled model shapes.
Two Frozen dots that would overlap are pushed apart, along the line joining
them, until they just touch; the script prints each offset. With --arrows (the
default), arrows run from each model's Frozen point to its LoRA point (dashed)
and its Full point (solid).

Usage:
  uv run python experiments/scripts/generate_paper_accuracy_alignment.py
  uv run python experiments/scripts/generate_paper_accuracy_alignment.py --no-arrows --output out.png
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.scripts.generate_paper_layer_iou import MODELS  # noqa: E402

DEFAULT_INPUT = (
    PROJECT_ROOT
    / "outputs"
    / "results"
    / "experiments"
    / "fine_tuning_primary_20260327"
    / "accuracy_alignment_analysis.json"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "docs" / "mva" / "RQ3_accuracy_alignment.png"

# The circle is reserved for Frozen runs, so no model uses it as its shape.
MARKERS = {
    "dinov2": "h",
    "dinov3": "s",
    "mae": "^",
    "clip": "D",
    "siglip": "v",
    "siglip2": "P",
}
FROZEN_MARKER = "o"
FROZEN_SIZE = 8
SHAPE_SIZE = 10
LEGEND_SHAPE = "s"
NEUTRAL = "#555555"
FILLED_EDGE = "#333333"


@dataclass(frozen=True)
class StrategyStyle:
    legend_label: str
    hollow: bool
    arrow_linestyle: str | None


STRATEGY_STYLES = {
    "frozen": StrategyStyle("Frozen", hollow=False, arrow_linestyle=None),
    "lora": StrategyStyle("LoRA", hollow=True, arrow_linestyle="--"),
    "full": StrategyStyle("Full fine-tuning", hollow=False, arrow_linestyle="-"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--arrows",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Draw arrows from each Frozen point to its LoRA and Full points.",
    )
    return parser.parse_args()


def marker_style(strategy: str, shape: str, color: str) -> dict[str, Any]:
    style = STRATEGY_STYLES[strategy]
    if strategy == "frozen":
        marker, size = FROZEN_MARKER, FROZEN_SIZE
    else:
        marker, size = shape, SHAPE_SIZE
    return {
        "marker": marker,
        "markersize": size,
        "markerfacecolor": "white" if style.hollow else color,
        "markeredgecolor": color if style.hollow else FILLED_EDGE,
        "markeredgewidth": 2.0 if style.hollow else 0.8,
    }


def separate_overlapping_dots(
    ax: plt.Axes, dots: dict[str, tuple[float, float]]
) -> dict[str, tuple[float, float]]:
    """Push each pair of overlapping Frozen dots apart, along the line joining them, until they just touch.

    The shift is computed in display space, so the axis limits must already be fixed.
    """
    to_display = ax.transData.transform
    to_data = ax.transData.inverted().transform
    touch_px = (FROZEN_SIZE + 1) * ax.figure.dpi / 72
    centers = {key: to_display(xy) for key, xy in dots.items()}
    pairs = [
        (a, b)
        for a, b in itertools.combinations(dots, 2)
        if math.dist(centers[a], centers[b]) < touch_px
    ]
    if len({key for pair in pairs for key in pair}) != 2 * len(pairs):
        raise ValueError("Three or more Frozen dots overlap; pairwise separation cannot show them all.")
    shown = dict(dots)
    for a, b in pairs:
        gap = centers[a] - centers[b]
        length = float(np.hypot(*gap))
        direction = gap / length if length else np.array([0.0, 1.0])
        shift = direction * (touch_px - length) / 2
        for key, center in ((a, centers[a] + shift), (b, centers[b] - shift)):
            x, y = to_data(center)
            shown[key] = (float(x), float(y))
            print(f"Offset Frozen {key} by ({x - dots[key][0]:+.4f}, {y - dots[key][1]:+.4f}) for visibility")
    return shown


def main() -> None:
    args = parse_args()
    analysis = json.loads(args.input.read_text(encoding="utf-8"))
    points = {
        (run["model"], run["strategy"]): (
            run["annotated_eval"]["accuracy"],
            run["annotated_eval"]["mean_iou"],
        )
        for run in analysis["runs"]
    }
    plotted_models = [model for model in MODELS if (model[0], "frozen") in points]

    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8.4, 5.8))
    ax.update_datalim(list(points.values()))
    ax.margins(x=0.08, y=0.12)
    ax.autoscale_view()
    # Fix the limits from the true points before any Frozen dot is moved in display space.
    ax.set_xlim(ax.get_xlim())
    ax.set_ylim(ax.get_ylim())
    shown_frozen = separate_overlapping_dots(ax, {key: points[(key, "frozen")] for key, *_ in plotted_models})
    for key, _label, _method, color in plotted_models:
        frozen_point = shown_frozen[key]
        for strategy, style in STRATEGY_STYLES.items():
            x, y = frozen_point if strategy == "frozen" else points[(key, strategy)]
            if args.arrows and style.arrow_linestyle is not None:
                ax.annotate(
                    "",
                    xy=(x, y),
                    xytext=frozen_point,
                    arrowprops={
                        "arrowstyle": "-|>",
                        "color": color,
                        "linestyle": style.arrow_linestyle,
                        "linewidth": 1.8,
                        "mutation_scale": 14,
                        "shrinkA": 5,
                        "shrinkB": 7,
                    },
                    zorder=2,
                )
            ax.plot(x, y, linestyle="none", zorder=3, **marker_style(strategy, MARKERS[key], color))

    ax.xaxis.set_major_formatter(plt.FormatStrFormatter("%.2f"))
    ax.yaxis.set_major_formatter(plt.FormatStrFormatter("%.3f"))
    ax.set_xlabel("Classification accuracy (139 annotated images)", fontsize=14)
    ax.set_ylabel(
        f"Mean IoU@{analysis['iou_percentile']} (layer {analysis['analyzed_layer']})",
        fontsize=14,
    )
    ax.tick_params(labelsize=14)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("#CCCCCC")

    model_handles = [
        Line2D(
            [],
            [],
            linestyle="none",
            label=label,
            **marker_style("full", MARKERS[key], color),
        )
        for key, label, _method, color in plotted_models
    ]
    strategy_handles = [
        Line2D(
            [],
            [],
            color=NEUTRAL,
            linewidth=1.8,
            linestyle=style.arrow_linestyle if args.arrows and style.arrow_linestyle else "none",
            label=style.legend_label,
            **marker_style(strategy, LEGEND_SHAPE, NEUTRAL),
        )
        for strategy, style in STRATEGY_STYLES.items()
    ]
    model_legend = ax.legend(
        handles=model_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.14),
        ncol=len(model_handles),
        frameon=False,
        fontsize=14,
        handletextpad=0.3,
        columnspacing=1.2,
    )
    ax.add_artist(model_legend)
    strategy_legend = ax.legend(
        handles=strategy_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, -0.24),
        ncol=len(strategy_handles),
        frameon=False,
        fontsize=14,
        handlelength=2.6 if args.arrows else 1.0,
        columnspacing=2.0,
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(
        args.output,
        dpi=240,
        bbox_inches="tight",
        bbox_extra_artists=[model_legend, strategy_legend],
        facecolor="white",
    )
    print(args.output)


if __name__ == "__main__":
    main()

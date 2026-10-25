#!/usr/bin/env python3
"""Generate the R1-Q3 accuracy-versus-alignment figure (RQ3_accuracy_alignment.png).

Reads accuracy_alignment_analysis.json only. Each point is one run: x is the
classification accuracy on the 139 annotated images, y is the mean IoU@90 at
layer 11 on the same images. Arrows run from each model's Frozen point to its
LoRA point (dashed) and its Full point (solid).

Usage:
  uv run python experiments/scripts/generate_paper_accuracy_alignment.py
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

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

MARKERS = {
    "dinov2": "o",
    "dinov3": "s",
    "mae": "^",
    "clip": "D",
    "siglip": "v",
    "siglip2": "P",
}
NEUTRAL = "#555555"
FILLED_EDGE = "#333333"


@dataclass(frozen=True)
class StrategyStyle:
    legend_label: str
    hollow: bool
    arrow_linestyle: str | None


STRATEGY_STYLES = {
    "frozen": StrategyStyle("Frozen (hollow)", hollow=True, arrow_linestyle=None),
    "lora": StrategyStyle("Frozen → LoRA", hollow=False, arrow_linestyle="--"),
    "full": StrategyStyle("Frozen → Full", hollow=False, arrow_linestyle="-"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


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
    for key, _label, _method, color in plotted_models:
        frozen_point = points[(key, "frozen")]
        for strategy, style in STRATEGY_STYLES.items():
            x, y = points[(key, strategy)]
            if style.arrow_linestyle is not None:
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
                        "shrinkA": 7,
                        "shrinkB": 7,
                    },
                    zorder=2,
                )
            ax.plot(
                x,
                y,
                marker=MARKERS[key],
                markersize=10,
                linestyle="none",
                markerfacecolor="white" if style.hollow else color,
                markeredgecolor=color if style.hollow else FILLED_EDGE,
                markeredgewidth=2.0 if style.hollow else 0.8,
                zorder=3,
            )

    ax.margins(x=0.08, y=0.12)
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
            marker=MARKERS[key],
            markersize=10,
            linestyle="none",
            markerfacecolor=color,
            markeredgecolor=FILLED_EDGE,
            markeredgewidth=0.8,
            label=label,
        )
        for key, label, _method, color in plotted_models
    ]
    strategy_handles = [
        Line2D(
            [],
            [],
            marker="o",
            markersize=10,
            linestyle="none",
            markerfacecolor="white",
            markeredgecolor=NEUTRAL,
            markeredgewidth=2.0,
            label=STRATEGY_STYLES["frozen"].legend_label,
        ),
        *(
            Line2D([], [], color=NEUTRAL, linewidth=1.8, linestyle=style.arrow_linestyle, label=style.legend_label)
            for style in STRATEGY_STYLES.values()
            if style.arrow_linestyle is not None
        ),
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
        handlelength=2.6,
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

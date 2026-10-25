#!/usr/bin/env python3
"""Redraw the paper's comparative-study framework as Figure 1.

Usage:
    uv run --no-sync python experiments/scripts/generate_paper_framework.py
    uv run --no-sync python experiments/scripts/generate_paper_framework.py --output /tmp/framework.png

Coordinates follow framework_v2.png's 1275 x 598 pixel grid, with the origin
at the top left. The output doubles both dimensions for print.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patches import Circle, PathPatch
from matplotlib.path import Path as DrawingPath

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "docs" / "mva" / "framework_v3.png"
WIDTH = 1275
HEIGHT = 598
COORDINATE_DPI = 100
OUTPUT_DPI = 200
TEXT_COLOR = "#3D474D"
FONT_SIZE = 17
LINE_WIDTH = 2.5

type Point = tuple[float, float]
type Curve = tuple[Point, Point, Point, Point]
type Leaf = tuple[str, float]


@dataclass(frozen=True)
class Branch:
    """One branch and its leaf labels in the original figure's coordinates."""

    label: str
    colour: str
    side: Literal["left", "right"]
    junction: Point
    leaves: tuple[Leaf, ...]
    curve: Curve


BRANCHES = (
    Branch(
        label="Model",
        colour="#4070FF",
        side="right",
        junction=(941, 138),
        leaves=(
            ("DINOv2, DINOv3", 28),
            ("MAE", 101),
            ("CLIP", 175),
            ("SigLIP, SigLIP2", 249),
        ),
        curve=((661, 265), (695, 191), (747, 138), (815, 138)),
    ),
    Branch(
        label="Attention map",
        colour="#A179F2",
        side="right",
        junction=(1034, 410),
        leaves=(
            ("[CLS] attention", 373),
            ("Mean attention", 447),
        ),
        curve=((668, 334), (707, 385), (755, 410), (815, 410)),
    ),
    Branch(
        label="Dataset",
        colour="#F7578C",
        side="right",
        junction=(959, 572),
        leaves=(("WikiChurches", 572),),
        curve=((655, 335), (694, 500), (737, 572), (815, 572)),
    ),
    Branch(
        label="Adaptation",
        colour="#FFAA00",
        side="left",
        junction=(296, 109),
        leaves=(
            ("LoRA", 72),
            ("Full fine-tuning", 146),
        ),
        curve=((635, 265), (590, 150), (550, 109), (478, 109)),
    ),
    Branch(
        label="Metric",
        colour="#FF4019",
        side="left",
        junction=(349, 418),
        leaves=(
            ("Coverage", 270),
            ("Earth Mover's distance", 344),
            ("Intersection over Union", 418),
            ("Kullback-Leibler divergence", 492),
            ("Mean squared error", 565),
        ),
        curve=((628, 334), (592, 385), (545, 418), (478, 418)),
    ),
)


def draw_curve(ax: Axes, points: Curve, colour: str) -> None:
    """Draw a cubic curve with round ends."""
    path = DrawingPath(
        points,
        [DrawingPath.MOVETO, DrawingPath.CURVE4, DrawingPath.CURVE4, DrawingPath.CURVE4],
    )
    ax.add_patch(
        PathPatch(
            path,
            fill=False,
            edgecolor=colour,
            linewidth=LINE_WIDTH,
            capstyle="round",
        )
    )


def draw_branch(ax: Axes, branch: Branch) -> None:
    """Draw the label, curve, junction and leaves for one branch."""
    junction_x, junction_y = branch.junction
    direction = 1 if branch.side == "right" else -1
    leaf_x = junction_x + direction * 24
    label_x = 834 if branch.side == "right" else junction_x + 39

    draw_curve(ax, branch.curve, branch.colour)
    ax.text(
        label_x,
        junction_y,
        branch.label,
        ha="left",
        va="center",
        fontsize=FONT_SIZE,
        fontweight="bold",
        color=TEXT_COLOR,
    )
    ax.plot(
        [junction_x - direction * 20, junction_x],
        [junction_y, junction_y],
        color=branch.colour,
        linewidth=LINE_WIDTH,
        solid_capstyle="round",
    )

    for text, leaf_y in branch.leaves:
        if len(branch.leaves) == 1:
            ax.plot(
                [junction_x, leaf_x],
                [junction_y, leaf_y],
                color=branch.colour,
                linewidth=LINE_WIDTH,
                solid_capstyle="round",
            )
        else:
            draw_curve(
                ax,
                (
                    branch.junction,
                    (junction_x, leaf_y),
                    (junction_x, leaf_y),
                    (leaf_x, leaf_y),
                ),
                branch.colour,
            )
        ax.text(
            leaf_x + direction * 19,
            leaf_y,
            text,
            ha="left" if branch.side == "right" else "right",
            va="center",
            fontsize=FONT_SIZE,
            color=TEXT_COLOR,
        )

    ax.add_patch(
        Circle(
            branch.junction,
            radius=7,
            facecolor="white",
            edgecolor=branch.colour,
            linewidth=LINE_WIDTH,
            zorder=3,
        )
    )


def generate_figure(output: Path) -> None:
    """Save the framework at exactly 2550 x 1196 pixels."""
    with plt.rc_context(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "savefig.bbox": None,
        }
    ):
        fig = plt.figure(
            figsize=(WIDTH / COORDINATE_DPI, HEIGHT / COORDINATE_DPI),
            dpi=OUTPUT_DPI,
        )
        ax = fig.add_axes((0, 0, 1, 1))
        ax.set_xlim(0, WIDTH)
        ax.set_ylim(HEIGHT, 0)
        ax.set_axis_off()

        for branch in BRANCHES:
            draw_branch(ax, branch)
        ax.text(
            647,
            299,
            "Comparative study",
            ha="center",
            va="center",
            fontsize=FONT_SIZE,
            fontweight="bold",
            color=TEXT_COLOR,
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output, dpi=OUTPUT_DPI, facecolor="white", pad_inches=0)
        plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()
    output = Path(args.output)
    generate_figure(output)
    print(output)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Generate visualizations for the fine-tuning run matrix.

Produces 8 publication-quality figures from Q2 metric deltas and hardcoded
run matrix data, saved to outputs/figures/.

Design principles (Tufte + Datawrapper best practices):
  - Maximize data-ink ratio: minimal gridlines, no chartjunk
  - Fewer colors: gray for context, color only for what matters
  - Muted, colorblind-safe palette (Tableau-inspired)
  - Direct labeling over legends where practical
  - Consistent typography and spacing

Usage:
    python experiments/scripts/generate_run_matrix_figures.py
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import seaborn as sns

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
Q2_JSON = PROJECT_ROOT / "outputs" / "results" / "q2_metrics_analysis.json"
FIGURES_DIR = PROJECT_ROOT / "outputs" / "figures"

# ---------------------------------------------------------------------------
# Design system — muted, colorblind-safe palette
# ---------------------------------------------------------------------------
MODELS = ["clip", "dinov2", "dinov3", "mae", "siglip", "siglip2"]
MODEL_LABELS = {
    "clip": "CLIP",
    "dinov2": "DINOv2",
    "dinov3": "DINOv3",
    "mae": "MAE",
    "siglip": "SigLIP",
    "siglip2": "SigLIP 2",
}
STRATEGIES = ["linear_probe", "lora", "full"]
STRATEGY_LABELS = {
    "linear_probe": "Linear Probe",
    "lora": "LoRA",
    "full": "Full",
}

# Muted Tableau-inspired strategy palette
STRATEGY_COLORS = {
    "linear_probe": "#93b7be",  # muted teal (de-emphasized — baseline strategy)
    "lora": "#4e79a7",          # steel blue
    "full": "#d4764e",          # warm terracotta
}
FROZEN_COLOR = "#8a817c"        # darker warm gray for frozen baselines
TEXT_COLOR = "#4a4a4a"          # dark gray for text (softer than black)
GRID_COLOR = "#cccccc"          # medium-light gray for axes/spines
IMPROVED_COLOR = "#3a7d44"      # forest green (muted)
DEGRADED_COLOR = "#c04e4e"      # muted red
SIG_COLOR = "#2b2b2b"           # near-black for significance markers

# Hardcoded from docs/reference/fine_tuning_run_matrix.md (lines 56-73)
RUN_MATRIX: dict[tuple[str, str], dict[str, Any]] = {
    ("clip", "full"): {"val_acc": 89.6, "best_epoch": 1},
    ("clip", "linear_probe"): {"val_acc": 83.2, "best_epoch": 3},
    ("clip", "lora"): {"val_acc": 84.8, "best_epoch": 3},
    ("dinov2", "full"): {"val_acc": 87.2, "best_epoch": 2},
    ("dinov2", "linear_probe"): {"val_acc": 89.6, "best_epoch": 1},
    ("dinov2", "lora"): {"val_acc": 88.8, "best_epoch": 1},
    ("dinov3", "full"): {"val_acc": 89.6, "best_epoch": 2},
    ("dinov3", "linear_probe"): {"val_acc": 90.4, "best_epoch": 1},
    ("dinov3", "lora"): {"val_acc": 91.2, "best_epoch": 2},
    ("mae", "full"): {"val_acc": 75.2, "best_epoch": 2},
    ("mae", "linear_probe"): {"val_acc": 60.0, "best_epoch": 2},
    ("mae", "lora"): {"val_acc": 72.8, "best_epoch": 3},
    ("siglip", "full"): {"val_acc": 88.0, "best_epoch": 2},
    ("siglip", "linear_probe"): {"val_acc": 85.6, "best_epoch": 1},
    ("siglip", "lora"): {"val_acc": 87.2, "best_epoch": 2},
    ("siglip2", "full"): {"val_acc": 88.8, "best_epoch": 1},
    ("siglip2", "linear_probe"): {"val_acc": 81.6, "best_epoch": 3},
    ("siglip2", "lora"): {"val_acc": 88.8, "best_epoch": 3},
}

HEATMAP_METRICS = [
    ("iou", 90, "IoU@90"),
    ("iou", 50, "IoU@50"),
    ("coverage", None, "Coverage"),
    ("mse", None, "MSE"),
    ("kl", None, "KL"),
    ("emd", None, "EMD"),
]

METRIC_DIRECTION = {
    "iou": "higher",
    "coverage": "higher",
    "mse": "lower",
    "kl": "lower",
    "emd": "lower",
}


# ---------------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------------
def load_q2_data() -> list[dict]:
    """Load Q2 metrics analysis JSON and return the rows list."""
    with open(Q2_JSON) as f:
        data = json.load(f)
    return data["rows"]


def lookup_row(
    rows: list[dict],
    model: str,
    strategy: str,
    metric: str,
    percentile: int | None = None,
) -> dict | None:
    """Find a specific row by (model, strategy, metric, percentile)."""
    for r in rows:
        if (
            r["model_name"] == model
            and r["strategy_id"] == strategy
            and r["metric"] == metric
            and r.get("percentile") == percentile
        ):
            return r
    return None


# ---------------------------------------------------------------------------
# Style helpers
# ---------------------------------------------------------------------------
def setup_style() -> None:
    """Set a clean, minimal style inspired by Tufte + modern dataviz."""
    plt.rcParams.update({
        "figure.dpi": 200,
        "savefig.dpi": 200,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "font.family": "sans-serif",
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.titleweight": "bold",
        "axes.labelsize": 9,
        "axes.labelcolor": TEXT_COLOR,
        "axes.edgecolor": GRID_COLOR,
        "axes.linewidth": 0.8,
        "axes.grid": False,
        "xtick.color": TEXT_COLOR,
        "ytick.color": TEXT_COLOR,
        "xtick.major.size": 3,
        "ytick.major.size": 3,
        "text.color": TEXT_COLOR,
        "legend.frameon": False,
        "legend.fontsize": 9,
    })


def clean_axes(ax: plt.Axes) -> None:
    """Apply minimal Tufte-inspired cleanup to an axes."""
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(GRID_COLOR)
    ax.spines["bottom"].set_color(GRID_COLOR)
    ax.grid(False)


def save_figure(fig: plt.Figure, name: str) -> Path:
    """Save a figure to the figures directory."""
    path = FIGURES_DIR / name
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    return path


# ---------------------------------------------------------------------------
# Figure 1: Grouped Bar Chart — Validation Accuracy
# ---------------------------------------------------------------------------
def fig_validation_accuracy() -> str:
    """Grouped bar chart of validation accuracy per model x strategy."""
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(MODELS))
    width = 0.24

    for i, strat in enumerate(STRATEGIES):
        vals = [RUN_MATRIX[(m, strat)]["val_acc"] for m in MODELS]
        bars = ax.bar(
            x + i * width,
            vals,
            width * 0.9,
            label=STRATEGY_LABELS[strat],
            color=STRATEGY_COLORS[strat],
            edgecolor="white",
            linewidth=0.3,
        )
        for bar, v in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.4,
                f"{v:.1f}",
                ha="center", va="bottom",
                fontsize=7, color=TEXT_COLOR,
            )

    ax.set_xticks(x + width)
    ax.set_xticklabels([MODEL_LABELS[m] for m in MODELS])
    ax.set_ylabel("Validation Accuracy (%)")
    ax.set_title("Validation Accuracy by Model and Fine-Tuning Strategy")
    ax.set_ylim(50, 100)
    ax.legend(loc="upper left", bbox_to_anchor=(1.01, 1), borderaxespad=0)
    clean_axes(ax)

    save_figure(fig, "01_val_accuracy_by_model_strategy.png")

    best_overall = max(RUN_MATRIX.items(), key=lambda kv: kv[1]["val_acc"])
    worst_overall = min(RUN_MATRIX.items(), key=lambda kv: kv[1]["val_acc"])
    return (
        f"Best: {MODEL_LABELS[best_overall[0][0]]} {STRATEGY_LABELS[best_overall[0][1]]} "
        f"({best_overall[1]['val_acc']:.1f}%). "
        f"Worst: {MODEL_LABELS[worst_overall[0][0]]} {STRATEGY_LABELS[worst_overall[0][1]]} "
        f"({worst_overall[1]['val_acc']:.1f}%). "
        "MAE consistently lags; DINOv3 LoRA leads. Full fine-tuning "
        "does not always beat LoRA on classification accuracy."
    )


# ---------------------------------------------------------------------------
# Figure 2: Delta Heatmap Grid
# ---------------------------------------------------------------------------
def fig_delta_heatmap(rows: list[dict]) -> str:
    """2x3 grid of annotated heatmaps — improvement direction normalized."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    for idx, (metric, pctl, title) in enumerate(HEATMAP_METRICS):
        ax = axes[idx]
        direction = METRIC_DIRECTION[metric]

        matrix = np.zeros((len(MODELS), len(STRATEGIES)))
        sig_mask = np.zeros_like(matrix, dtype=bool)
        for mi, m in enumerate(MODELS):
            for si, s in enumerate(STRATEGIES):
                row = lookup_row(rows, m, s, metric, pctl)
                if row:
                    delta = row["mean_delta"]
                    matrix[mi, si] = delta if direction == "higher" else -delta
                    sig_mask[mi, si] = row.get("significant", False)

        vmax = max(abs(matrix.min()), abs(matrix.max())) or 0.01

        # Custom diverging colormap: muted blue-white-red
        sns.heatmap(
            matrix, ax=ax,
            cmap="RdBu", center=0, vmin=-vmax, vmax=vmax,
            annot=True, fmt="+.3f",
            annot_kws={"size": 8, "color": TEXT_COLOR},
            linewidths=1, linecolor="white",
            xticklabels=[STRATEGY_LABELS[s] for s in STRATEGIES],
            yticklabels=[MODEL_LABELS[m] for m in MODELS] if idx % 3 == 0 else False,
            cbar_kws={"shrink": 0.7, "aspect": 15},
        )

        for mi in range(len(MODELS)):
            for si in range(len(STRATEGIES)):
                if sig_mask[mi, si]:
                    ax.text(
                        si + 0.5, mi + 0.82, "*",
                        ha="center", va="center",
                        fontsize=11, fontweight="bold", color=SIG_COLOR,
                    )

        hint = "higher=better" if direction == "higher" else "sign flipped"
        ax.set_title(f"{title}  ({hint})", fontsize=10)

    fig.suptitle(
        "Improvement Heatmaps  |  Blue = improved, Red = degraded  (* = significant)",
        fontsize=12, fontweight="bold", color=TEXT_COLOR, y=1.01,
    )
    fig.tight_layout()
    save_figure(fig, "02_all_metrics_improvement_heatmap.png")

    return (
        "All deltas normalized to improvement direction: positive (blue) = better, "
        "negative (red) = worse. For MSE/KL/EMD the sign is flipped so blue still "
        "means the metric decreased (improved). Linear probe near-zero confirms frozen "
        "backbones don't shift attention. SigLIP shows the most significant improvements."
    )


# ---------------------------------------------------------------------------
# Figure 3: Diverging Bar Chart — Per-Metric Deep Dive
# ---------------------------------------------------------------------------
def fig_diverging_bars(rows: list[dict]) -> str:
    """2x3 faceted horizontal diverging bars, sorted descending, colored by model."""
    active_strategies = ["lora", "full"]

    # One color per model (consistent across subplots)
    model_colors = {
        "clip": "#4e79a7",
        "dinov2": "#f28e2b",
        "dinov3": "#59a14f",
        "mae": "#e15759",
        "siglip": "#76b7b2",
        "siglip2": "#b07aa1",
    }

    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    axes = axes.flatten()

    for idx, (metric, pctl, title) in enumerate(HEATMAP_METRICS):
        ax = axes[idx]
        direction = METRIC_DIRECTION[metric]

        # Collect entries: (label, delta, sig, model_key)
        entries = []
        for m in MODELS:
            for s in active_strategies:
                row = lookup_row(rows, m, s, metric, pctl)
                if row:
                    entries.append({
                        "label": f"{MODEL_LABELS[m]} — {STRATEGY_LABELS[s]}",
                        "delta": row["mean_delta"],
                        "sig": row.get("significant", False),
                        "model": m,
                    })

        # Sort: best performers at top (direction-aware)
        if direction == "higher":
            entries.sort(key=lambda e: e["delta"], reverse=True)
        else:
            entries.sort(key=lambda e: e["delta"], reverse=False)

        labels = [e["label"] for e in entries]
        deltas = np.array([e["delta"] for e in entries])
        colors = [model_colors[e["model"]] for e in entries]
        y_pos = np.arange(len(entries))

        ax.barh(y_pos, deltas, color=colors, height=0.6, edgecolor="white", linewidth=0.3)
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels, fontsize=7)
        ax.invert_yaxis()  # best performer (index 0) at top
        ax.axvline(0, color=TEXT_COLOR, linewidth=0.6)
        hint = "(higher=better)" if direction == "higher" else "(lower=better)"
        ax.set_title(f"{title}  {hint}", fontsize=10)
        clean_axes(ax)

        # Significance markers
        for i, e in enumerate(entries):
            if e["sig"]:
                d = e["delta"]
                offset = abs(d) * 0.1 + 0.0005
                x_pos = d + offset if d >= 0 else d - offset
                ax.text(
                    x_pos, i, "*",
                    ha="center", va="center",
                    fontsize=10, fontweight="bold", color=SIG_COLOR,
                )

    # Model color legend
    from matplotlib.patches import Patch
    legend_handles = [Patch(facecolor=model_colors[m], label=MODEL_LABELS[m]) for m in MODELS]
    fig.legend(
        handles=legend_handles,
        loc="upper center", ncol=6, fontsize=9,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(
        "Per-Metric Deltas: LoRA & Full, best at top  (* = significant)",
        fontsize=12, fontweight="bold", color=TEXT_COLOR, y=1.05,
    )
    fig.tight_layout()
    save_figure(fig, "03_all_metrics_diverging_bars.png")

    return (
        "Bars sorted by delta (largest improvement at top). Colors represent models, "
        "making it easy to spot which models consistently rank high or low. "
        "Linear probe omitted (all near zero). "
        "SigLIP entries cluster near the top across most metrics."
    )


# ---------------------------------------------------------------------------
# Figure 4: IoU Percentile Slope Chart
# ---------------------------------------------------------------------------
def fig_iou_percentile_slopes(rows: list[dict]) -> str:
    """2x3 grid of line charts showing IoU delta across percentiles."""
    percentiles = [50, 60, 70, 80, 90]
    active_strategies = ["lora", "full"]
    line_styles = {"lora": "--", "full": "-"}

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    for mi, model in enumerate(MODELS):
        ax = axes[mi]
        for strat in active_strategies:
            deltas, sigs = [], []
            for p in percentiles:
                row = lookup_row(rows, model, strat, "iou", p)
                if row:
                    deltas.append(row["mean_delta"])
                    sigs.append(row.get("significant", False))
                else:
                    deltas.append(0.0)
                    sigs.append(False)

            ax.plot(
                percentiles, deltas,
                marker="o", linestyle=line_styles[strat],
                color=STRATEGY_COLORS[strat],
                label=STRATEGY_LABELS[strat],
                linewidth=1.8, markersize=4,
            )

            for p, d, sig in zip(percentiles, deltas, sigs):
                if sig:
                    ax.annotate(
                        "*", (p, d),
                        textcoords="offset points", xytext=(0, 7),
                        ha="center", fontsize=10,
                        color=SIG_COLOR, fontweight="bold",
                    )

        ax.axhline(0, color=FROZEN_COLOR, linewidth=0.8, linestyle="-")

        ax.set_title(MODEL_LABELS[model], fontsize=10)
        ax.set_xlabel("Percentile")
        ax.set_ylabel("IoU Delta")
        ax.set_xticks(percentiles)
        clean_axes(ax)

    fig.legend(
        [STRATEGY_LABELS[s] for s in active_strategies],
        loc="upper center", ncol=2, fontsize=10,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(
        "IoU Delta Across Percentile Thresholds  (* = significant)",
        fontsize=12, fontweight="bold", color=TEXT_COLOR, y=1.05,
    )
    fig.tight_layout()
    save_figure(fig, "04_iou_delta_by_percentile.png")

    return (
        "Slopes reveal how fine-tuning affects top-k attention alignment at different "
        "thresholds. Steeper slopes at high percentiles indicate that fine-tuning "
        "particularly shifts the most-attended regions. Flat slopes near zero confirm "
        "linear probe has no effect (omitted)."
    )


# ---------------------------------------------------------------------------
# Figure 5: Radar Chart — Multi-Metric Model Profiles
# ---------------------------------------------------------------------------
def fig_radar_profiles(rows: list[dict]) -> str:
    """2x3 grid of radar charts showing normalized metric profiles."""
    radar_metrics = [
        ("iou", 90, "IoU@90"),
        ("coverage", None, "Coverage"),
        ("mse", None, "MSE"),
        ("kl", None, "KL"),
        ("emd", None, "EMD"),
    ]
    n_metrics = len(radar_metrics)

    all_vals: dict[str, list[float]] = {m[0] + str(m[1]): [] for m in radar_metrics}
    for m in MODELS:
        for s in STRATEGIES:
            for metric, pctl, _ in radar_metrics:
                key = metric + str(pctl)
                row = lookup_row(rows, m, s, metric, pctl)
                if row:
                    all_vals[key].append(row["frozen_mean"])
                    all_vals[key].append(row["finetuned_mean"])

    bounds: dict[str, tuple[float, float]] = {}
    for key, vals in all_vals.items():
        bounds[key] = (min(vals), max(vals)) if vals else (0.0, 1.0)

    def normalize(value: float, metric: str, pctl: int | None) -> float:
        key = metric + str(pctl)
        lo, hi = bounds[key]
        if hi == lo:
            return 0.5
        normed = (value - lo) / (hi - lo)
        if METRIC_DIRECTION[metric] == "lower":
            normed = 1.0 - normed
        return normed

    angles = np.linspace(0, 2 * np.pi, n_metrics, endpoint=False).tolist()
    angles += angles[:1]

    fig, axes = plt.subplots(2, 3, figsize=(14, 10), subplot_kw={"projection": "polar"})
    axes = axes.flatten()

    strat_styles = {
        "lora": {"color": STRATEGY_COLORS["lora"], "linestyle": "--", "linewidth": 1.8},
        "full": {"color": STRATEGY_COLORS["full"], "linestyle": "-", "linewidth": 1.8},
    }

    for mi, model in enumerate(MODELS):
        ax = axes[mi]

        # Frozen baseline
        frozen_vals = []
        for metric, pctl, _ in radar_metrics:
            row = lookup_row(rows, model, "linear_probe", metric, pctl)
            frozen_vals.append(normalize(row["frozen_mean"], metric, pctl) if row else 0.0)
        frozen_vals += frozen_vals[:1]
        ax.plot(angles, frozen_vals, color=FROZEN_COLOR, linewidth=1.8, linestyle="--", label="Frozen")
        ax.fill(angles, frozen_vals, color=FROZEN_COLOR, alpha=0.08)

        # LoRA and Full
        for strat in ["lora", "full"]:
            vals = []
            for metric, pctl, _ in radar_metrics:
                row = lookup_row(rows, model, strat, metric, pctl)
                vals.append(normalize(row["finetuned_mean"], metric, pctl) if row else 0.0)
            vals += vals[:1]
            ax.plot(angles, vals, label=STRATEGY_LABELS[strat], **strat_styles[strat])
            ax.fill(angles, vals, color=strat_styles[strat]["color"], alpha=0.06)

        ax.set_xticks(angles[:-1])
        ax.set_xticklabels([rm[2] for rm in radar_metrics], fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title(MODEL_LABELS[model], fontsize=10, pad=15)
        ax.tick_params(axis="y", labelsize=6, labelcolor=FROZEN_COLOR)
        if mi == 0:
            ax.legend(fontsize=10, loc="upper right", bbox_to_anchor=(1.5, 1.15))

    fig.suptitle(
        "Multi-Metric Radar Profiles (outward = better, normalized [0,1])",
        fontsize=12, fontweight="bold", color=TEXT_COLOR, y=1.01,
    )
    fig.tight_layout()
    save_figure(fig, "05_iou_coverage_mse_kl_emd_radar.png")

    return (
        "Radar charts show each model's metric profile normalized globally. "
        "Outward = better for all axes (lower-is-better metrics are inverted). "
        "Frozen baseline in gray; fine-tuned strategies overlaid. "
        "Models where the colored polygon expands beyond gray show genuine improvement."
    )


# ---------------------------------------------------------------------------
# Figure 6: Faceted Scatter — Val Accuracy vs Attention Delta
# ---------------------------------------------------------------------------
def fig_accuracy_vs_attention(rows: list[dict]) -> str:
    """Faceted scatter (2x3): one panel per model."""
    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()

    strategy_markers = {"linear_probe": "s", "lora": "D", "full": "o"}

    all_x: list[float] = []
    for m in MODELS:
        for strat in STRATEGIES:
            all_x.append(RUN_MATRIX[(m, strat)]["val_acc"])
    x_pad = (max(all_x) - min(all_x)) * 0.15

    for mi, model in enumerate(MODELS):
        ax = axes[mi]
        xs, ys = [], []
        for strat in STRATEGIES:
            val_acc = RUN_MATRIX[(model, strat)]["val_acc"]
            row = lookup_row(rows, model, strat, "iou", 90)
            delta = row["mean_delta"] if row else 0.0
            xs.append(val_acc)
            ys.append(delta)

        for i, strat in enumerate(STRATEGIES):
            ax.scatter(
                xs[i], ys[i],
                c=STRATEGY_COLORS[strat],
                label=STRATEGY_LABELS[strat] if mi == 0 else None,
                s=90, edgecolors="white", linewidth=0.5, zorder=4,
                marker=strategy_markers[strat],
            )
            ax.annotate(
                STRATEGY_LABELS[strat],
                (xs[i], ys[i]),
                textcoords="offset points",
                xytext=(7, -10 if i == 0 else 7),
                fontsize=7, color=STRATEGY_COLORS[strat],
            )

        ax.axhline(0, color=FROZEN_COLOR, linewidth=0.8, linestyle="--", zorder=1)
        ax.set_title(MODEL_LABELS[model], fontsize=10)
        ax.set_xlim(min(all_x) - x_pad, max(all_x) + x_pad)
        y_range = max(ys) - min(ys) if max(ys) != min(ys) else 0.005
        y_pad = y_range * 0.4
        y_lo = min(min(ys) - y_pad, -y_pad * 0.3)
        y_hi = max(max(ys) + y_pad, y_pad * 0.3)
        ax.set_ylim(y_lo, y_hi)
        clean_axes(ax)
        if mi >= 3:
            ax.set_xlabel("Val Accuracy (%)")
        if mi % 3 == 0:
            ax.set_ylabel("IoU@90 Delta")

    fig.legend(
        [STRATEGY_LABELS[s] for s in STRATEGIES],
        loc="upper center", ncol=3, fontsize=10,
        bbox_to_anchor=(0.5, 1.02),
    )
    fig.suptitle(
        "Classification Accuracy vs Attention Alignment Change",
        fontsize=12, fontweight="bold", color=TEXT_COLOR, y=1.05,
    )
    fig.tight_layout()
    save_figure(fig, "06_val_accuracy_vs_iou90_delta.png")

    return (
        "Each panel shows one model's trajectory from Linear Probe (square) through "
        "LoRA (diamond) to Full (circle). Shared axes enable cross-model comparison. "
        "Points above the dashed y=0 line improved attention alignment. "
        "Models like CLIP/SigLIP show clear upward trajectories; DINOv2/DINOv3 stay flat."
    )


# ---------------------------------------------------------------------------
# Figure 7: Best-Epoch Distribution
# ---------------------------------------------------------------------------
def fig_best_epoch_distribution() -> str:
    """Grouped bar chart of best-epoch counts by strategy."""
    epochs = [1, 2, 3]

    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(len(epochs))
    width = 0.24

    for i, strat in enumerate(STRATEGIES):
        counts = []
        for ep in epochs:
            count = sum(
                1 for m in MODELS
                if RUN_MATRIX[(m, strat)]["best_epoch"] == ep
            )
            counts.append(count)

        bars = ax.bar(
            x + i * width, counts, width * 0.9,
            label=STRATEGY_LABELS[strat],
            color=STRATEGY_COLORS[strat],
            edgecolor="white", linewidth=0.3,
        )
        for bar, c in zip(bars, counts):
            if c > 0:
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.05,
                    str(c), ha="center", va="bottom",
                    fontsize=9, fontweight="bold", color=TEXT_COLOR,
                )

    ax.set_xticks(x + width)
    ax.set_xticklabels([f"Epoch {e}" for e in epochs])
    ax.set_ylabel("Number of Models")
    ax.set_title("Best-Epoch Distribution by Fine-Tuning Strategy")
    ax.set_ylim(0, max(6, ax.get_ylim()[1]) + 0.5)
    ax.legend()
    clean_axes(ax)
    ax.yaxis.set_major_locator(mticker.MaxNLocator(integer=True))

    fig.tight_layout()
    save_figure(fig, "07_best_epoch_by_strategy.png")

    strat_mode: dict[str, int] = {}
    for strat in STRATEGIES:
        epoch_counts = {e: 0 for e in epochs}
        for m in MODELS:
            ep = RUN_MATRIX[(m, strat)]["best_epoch"]
            epoch_counts[ep] += 1
        strat_mode[strat] = max(epoch_counts, key=lambda e: epoch_counts[e])

    return (
        f"Modal best epoch: Linear Probe -> {strat_mode['linear_probe']}, "
        f"LoRA -> {strat_mode['lora']}, Full -> {strat_mode['full']}. "
        "Full fine-tuning converges early (epochs 1-2), while LoRA and linear probe "
        "are more spread across epochs, suggesting different optimization dynamics."
    )


# ---------------------------------------------------------------------------
# Figure 8: Frozen vs Fine-Tuned — Cleveland Dot Plot
# ---------------------------------------------------------------------------
def fig_frozen_vs_finetuned(rows: list[dict]) -> str:
    """2x3 dot plot: frozen baseline vs fine-tuned strategies per model.

    Replaces the crowded grouped bar chart with a cleaner Cleveland dot plot.
    Each row = one model. Dots show frozen (gray) and 3 strategy values (colored).
    A thin gray line spans from frozen to the most extreme fine-tuned value.
    """
    fig, axes = plt.subplots(2, 3, figsize=(14, 9))
    axes = axes.flatten()

    dot_config = [
        (None, FROZEN_COLOR, "o", 50, "Frozen"),
        ("linear_probe", STRATEGY_COLORS["linear_probe"], "s", 40, "Linear Probe"),
        ("lora", STRATEGY_COLORS["lora"], "D", 40, "LoRA"),
        ("full", STRATEGY_COLORS["full"], "o", 50, "Full"),
    ]

    for idx, (metric, pctl, title) in enumerate(HEATMAP_METRICS):
        ax = axes[idx]
        direction = METRIC_DIRECTION[metric]
        y_pos = np.arange(len(MODELS))

        for mi, m in enumerate(MODELS):
            # Get frozen value and all fine-tuned values
            base_row = lookup_row(rows, m, "linear_probe", metric, pctl)
            if not base_row:
                continue
            frozen_val = base_row["frozen_mean"]
            all_vals = [frozen_val]

            for strat in STRATEGIES:
                row = lookup_row(rows, m, strat, metric, pctl)
                if row:
                    all_vals.append(row["finetuned_mean"])

            # Span line from min to max value
            ax.plot(
                [min(all_vals), max(all_vals)], [mi, mi],
                color=GRID_COLOR, linewidth=1.5, zorder=1,
            )

        # Plot dots for each category
        for strat, color, marker, size, label in dot_config:
            vals = []
            for m in MODELS:
                lookup_strat = strat if strat else "linear_probe"
                row = lookup_row(rows, m, lookup_strat, metric, pctl)
                if row:
                    vals.append(row["frozen_mean"] if strat is None else row["finetuned_mean"])
                else:
                    vals.append(0.0)

            ax.scatter(
                vals, y_pos,
                c=color, marker=marker, s=size,
                edgecolors="white", linewidth=0.3, zorder=3,
                label=label if idx == 0 else None,
            )

        ax.set_yticks(y_pos)
        ax.set_yticklabels([MODEL_LABELS[m] for m in MODELS] if idx % 3 == 0 else [])
        hint = "higher=better" if direction == "higher" else "lower=better"
        ax.set_title(f"{title}  ({hint})", fontsize=10)
        ax.invert_yaxis()
        clean_axes(ax)

    fig.legend(
        loc="upper center", ncol=4, fontsize=10,
        bbox_to_anchor=(0.5, 1.02),
        columnspacing=2,
    )
    fig.suptitle(
        "Raw Metric Values: Frozen Baseline vs Fine-Tuned Strategies",
        fontsize=12, fontweight="bold", color=TEXT_COLOR, y=1.05,
    )
    fig.tight_layout()
    save_figure(fig, "08_frozen_vs_finetuned_all_metrics.png")

    return (
        "Cleveland dot plot: gray circles = frozen baseline, colored shapes = fine-tuned. "
        "Horizontal span shows the range of change. For higher-is-better metrics, "
        "rightward dots = improvement. For lower-is-better, leftward = improvement. "
        "Much clearer than grouped bars for comparing absolute performance levels."
    )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    setup_style()
    rows = load_q2_data()

    figures = [
        ("01 Validation Accuracy", lambda: fig_validation_accuracy()),
        ("02 Delta Heatmap Grid", lambda: fig_delta_heatmap(rows)),
        ("03 Diverging Bars", lambda: fig_diverging_bars(rows)),
        ("04 IoU Percentile Slopes", lambda: fig_iou_percentile_slopes(rows)),
        ("05 Radar Profiles", lambda: fig_radar_profiles(rows)),
        ("06 Accuracy vs Attention", lambda: fig_accuracy_vs_attention(rows)),
        ("07 Best-Epoch Distribution", lambda: fig_best_epoch_distribution()),
        ("08 Frozen vs Fine-Tuned", lambda: fig_frozen_vs_finetuned(rows)),
    ]

    commentary_lines: list[str] = []
    for name, gen_fn in figures:
        print(f"\n{'='*60}")
        print(f"  {name}")
        print(f"{'='*60}")
        commentary = gen_fn()
        print(f"  {commentary}")
        commentary_lines.append(f"## {name}\n{commentary}\n")

    commentary_path = FIGURES_DIR / "commentary.txt"
    commentary_path.write_text(
        "# Run Matrix Figure Commentary\n\n" + "\n".join(commentary_lines)
    )
    print(f"\n{'='*60}")
    print(f"  All 8 figures + commentary saved to {FIGURES_DIR}/")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()

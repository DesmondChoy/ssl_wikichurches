#!/usr/bin/env python3
"""Generate paper Figure 5, the RQ3 attention-alignment delta heatmap.

Reads the active experiment's q2_metrics_analysis.json.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1 import make_axes_locatable  # type: ignore[import-untyped]

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ACTIVE_EXPERIMENT_PATH = PROJECT_ROOT / 'outputs' / 'results' / 'active_experiment.json'
Q2_JSON_FALLBACK = PROJECT_ROOT / 'outputs' / 'results' / 'q2_metrics_analysis.json'
OUTPUT_PATH = PROJECT_ROOT / 'docs' / 'mva' / 'RQ3_adaptation_heatmap.png'

MODEL_ORDER = ['dinov2', 'dinov3', 'mae', 'clip', 'siglip', 'siglip2']
MODEL_LABELS = {
    'dinov2': 'DINOv2',
    'dinov3': 'DINOv3',
    'mae': 'MAE',
    'clip': 'CLIP',
    'siglip': 'SigLIP',
    'siglip2': 'SigLIP2',
}
STRATEGIES = ['lora', 'full']
STRATEGY_LABELS = {
    'lora': 'LoRA',
    'full': 'Full',
}
METRICS = [
    ('iou', 90, 'IoU@90', 'higher'),
    ('coverage', None, 'Coverage', 'higher'),
    ('mse', None, 'MSE', 'lower'),
    ('kl', None, 'KL', 'lower'),
    ('emd', None, 'EMD', 'lower'),
]
TEXT = '#343a40'
NOTE_BG = '#f7f8fa'
BLUE = '#2166ac'
RED = '#b2182b'
ZERO_EPS = 5e-4


def load_payload() -> dict[str, Any]:
    q2_json = Q2_JSON_FALLBACK
    if ACTIVE_EXPERIMENT_PATH.exists():
        with open(ACTIVE_EXPERIMENT_PATH, encoding='utf-8') as handle:
            active = json.load(handle)
        active_q2_path = active.get('q2_metrics_path')
        if active_q2_path:
            q2_json = PROJECT_ROOT / active_q2_path

    with open(q2_json, encoding='utf-8') as handle:
        payload: dict[str, Any] = json.load(handle)
    payload['_q2_json'] = str(q2_json)
    return payload


def lookup(
    rows: list[dict[str, Any]],
    model: str,
    strategy: str,
    metric: str,
    percentile: int | None,
) -> dict[str, Any]:
    for row in rows:
        if (
            row['model_name'] == model
            and row['strategy_id'] == strategy
            and row['metric'] == metric
            and row.get('percentile') == percentile
        ):
            return row
    raise KeyError(f'Missing row for {model}/{strategy}/{metric}/{percentile}')


def sign_normalized(row: dict[str, Any], direction: str) -> float:
    value = float(row['mean_delta'])
    return -value if direction == 'lower' else value


def fmt(value: float) -> str:
    if abs(value) < ZERO_EPS:
        value = 0.0
    return f'{value:+.3f}'


def draw_heatmap_panel(
    fig: plt.Figure,
    ax: plt.Axes,
    rows: list[dict[str, Any]],
    metric: str,
    percentile: int | None,
    title: str,
    direction: str,
    show_ylabels: bool,
) -> None:
    matrix = np.zeros((len(MODEL_ORDER), len(STRATEGIES)), dtype=float)
    significant = np.zeros_like(matrix, dtype=bool)

    for model_index, model in enumerate(MODEL_ORDER):
        for strategy_index, strategy in enumerate(STRATEGIES):
            row = lookup(rows, model, strategy, metric, percentile)
            matrix[model_index, strategy_index] = sign_normalized(row, direction)
            significant[model_index, strategy_index] = bool(row.get('significant'))

    vmax = float(np.nanmax(np.abs(matrix))) if np.isfinite(matrix).any() else 0.01
    vmax = max(vmax, 0.01)
    image = ax.imshow(matrix, cmap='RdBu', vmin=-vmax, vmax=vmax, aspect='auto')

    ax.set_title(title, fontsize=20, fontweight='bold', color=TEXT, pad=12)
    ax.set_xticks(np.arange(len(STRATEGIES)))
    ax.set_xticklabels([STRATEGY_LABELS[strategy] for strategy in STRATEGIES], fontsize=15, color=TEXT)
    ax.set_yticks(np.arange(len(MODEL_ORDER)))
    if show_ylabels:
        ax.set_yticklabels([MODEL_LABELS[model] for model in MODEL_ORDER], fontsize=15, color=TEXT)
        ax.tick_params(axis='y', length=0, pad=8)
    else:
        ax.set_yticklabels([])
        ax.tick_params(axis='y', left=False, length=0)

    ax.set_xticks(np.arange(-0.5, len(STRATEGIES), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(MODEL_ORDER), 1), minor=True)
    ax.grid(which='minor', color='white', linestyle='-', linewidth=1.6)
    ax.tick_params(which='minor', bottom=False, left=False)
    ax.tick_params(axis='x', labelsize=15, colors=TEXT, pad=7, length=3)
    for spine in ax.spines.values():
        spine.set_visible(False)

    for model_index in range(len(MODEL_ORDER)):
        for strategy_index in range(len(STRATEGIES)):
            value = matrix[model_index, strategy_index]
            label = f'{fmt(value)}{"*" if significant[model_index, strategy_index] else ""}'
            contrast_color = 'white' if abs(value) >= vmax * 0.70 else TEXT
            ax.text(
                strategy_index,
                model_index,
                label,
                ha='center',
                va='center',
                fontsize=15,
                fontweight='bold' if significant[model_index, strategy_index] else 'medium',
                color=contrast_color,
            )

    divider = make_axes_locatable(ax)
    cax = divider.append_axes('right', size='4.8%', pad=0.16)
    cbar = fig.colorbar(image, cax=cax)
    cbar.ax.tick_params(labelsize=16, colors=TEXT, length=3)
    cbar.outline.set_visible(False)  # type: ignore[operator]


def draw_legend_panel(ax: plt.Axes) -> None:
    ax.set_axis_off()
    ax.set_facecolor(NOTE_BG)
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color('#e1e5ea')
    ax.text(0.05, 0.76, 'Legend', fontsize=18, fontweight='bold', color=TEXT, transform=ax.transAxes)

    items = [(0.08, BLUE, 'Improved'), (0.47, RED, 'Degraded')]
    for x, color, label in items:
        ax.add_patch(
            Rectangle((x, 0.54), 0.05, 0.10, transform=ax.transAxes, facecolor=color, edgecolor='none')
        )
        ax.text(x + 0.07, 0.59, label, fontsize=15, fontweight='bold', color=TEXT, va='center', transform=ax.transAxes)

    ax.text(0.08, 0.30, '*', fontsize=20, fontweight='bold', color=TEXT, va='center', transform=ax.transAxes)
    ax.text(
        0.15,
        0.30,
        'significant after Holm correction',
        fontsize=15,
        color=TEXT,
        va='center',
        transform=ax.transAxes,
    )


payload = load_payload()
rows = list(payload['rows'])
fig, axes = plt.subplots(2, 3, figsize=(21, 13.6), dpi=220)
axes_flat = axes.flatten()

for index, (ax, metric_spec) in enumerate(zip(axes_flat[:5], METRICS, strict=True)):
    show_ylabels = index % 3 == 0
    draw_heatmap_panel(fig, ax, rows, *metric_spec, show_ylabels=show_ylabels)

draw_legend_panel(axes_flat[5])
fig.subplots_adjust(left=0.085, right=0.985, top=0.89, bottom=0.08, wspace=0.25, hspace=0.33)
OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUTPUT_PATH, bbox_inches='tight', facecolor='white')
plt.close(fig)
print(OUTPUT_PATH)
print(payload.get('_q2_json'))
print(payload.get('experiment_id'))

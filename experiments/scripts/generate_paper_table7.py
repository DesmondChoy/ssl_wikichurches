#!/usr/bin/env python3
"""Generate the paper's dominant layer-head location figure (RQ3_head_ranking.png).

Reads per-head IoU@90 from metrics.db and SigLIP-family proxy heads from attention_viz.h5.
"""

from __future__ import annotations

import math
import sqlite3
from pathlib import Path

import h5py
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.font_manager import FontProperties
from matplotlib.lines import Line2D

from ssl_attention.config import DATASET_PATH
from ssl_attention.data import AnnotatedSubset
from ssl_attention.metrics.iou import EPSILON

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DB_PATH = PROJECT_ROOT / 'outputs' / 'cache' / 'metrics.db'
H5_PATH = PROJECT_ROOT / 'outputs' / 'cache' / 'attention_viz.h5'
OUT_PATH = PROJECT_ROOT / 'docs' / 'mva' / 'RQ3_head_ranking.png'
PERCENTILE = 90
NUM_LAYERS = 12
NUM_HEADS = 12
TOP_K = 3

MODELS = [
    ('dinov2', 'DINOv2', 'cls', 'native'),
    ('dinov3', 'DINOv3', 'cls', 'native'),
    ('mae', 'MAE', 'cls', 'native'),
    ('clip', 'CLIP', 'cls', 'native'),
    ('siglip', 'SigLIP', 'mean', 'proxy'),
    ('siglip2', 'SigLIP2', 'mean', 'proxy'),
]
VARIANTS = [
    ('frozen', 'Frozen', '#0072B2', 'o', (-0.11, -0.08)),
    ('lora', 'LoRA', '#D55E00', 's', (0.00, 0.10)),
    ('full', 'Full', '#009E73', '^', (0.11, -0.08)),
]
RANK_JITTER = {1: (0.00, 0.00), 2: (-0.06, 0.045), 3: (0.06, -0.045)}

def model_key(base: str, variant: str) -> str:
    return base if variant == 'frozen' else f'{base}_finetuned_{variant}'

def load_native_grid(conn: sqlite3.Connection, key: str, method: str) -> np.ndarray:
    grid = np.full((NUM_HEADS, NUM_LAYERS), np.nan, dtype=np.float32)
    rows = conn.execute(
        """
        SELECT layer, head, mean_score
        FROM head_summary_metrics
        WHERE model = ? AND method = ? AND metric = 'iou' AND percentile = ?
        """,
        (key, method, PERCENTILE),
    ).fetchall()
    for layer, head, score in rows:
        layer_idx = int(str(layer).replace('layer', ''))
        head_idx = int(head)
        if 0 <= layer_idx < NUM_LAYERS and 0 <= head_idx < NUM_HEADS:
            grid[head_idx, layer_idx] = float(score)
    return grid

def prepare_masks() -> tuple[list[str], np.ndarray, np.ndarray, int, int]:
    dataset = AnnotatedSubset(DATASET_PATH)
    image_ids = dataset.image_ids
    height = width = 224
    masks = []
    sums = []
    for image_id in image_ids:
        mask = dataset.annotations[image_id].get_union_mask(height, width).cpu().numpy().astype(bool).ravel()
        masks.append(mask)
        sums.append(int(mask.sum()))
    return image_ids, np.stack(masks, axis=0), np.asarray(sums, dtype=np.int32), height, width

def mean_iou_for_group(group: h5py.Group, image_ids: list[str], masks: np.ndarray, mask_sums: np.ndarray, k: int) -> tuple[float, int]:
    arrays = []
    valid_positions = []
    for idx, image_id in enumerate(image_ids):
        try:
            arrays.append(np.asarray(group[image_id]['attention'], dtype=np.float32).reshape(-1))
            valid_positions.append(idx)
        except KeyError:
            pass
    if not arrays:
        return math.nan, 0
    attn = np.stack(arrays, axis=0)
    top_idx = np.argpartition(attn, -k, axis=1)[:, -k:]
    valid_masks = masks[valid_positions]
    valid_sums = mask_sums[valid_positions]
    intersections = valid_masks[np.arange(len(valid_positions))[:, None], top_idx].sum(axis=1)
    unions = k + valid_sums - intersections
    return float(np.mean(intersections / (unions + EPSILON))), int(len(valid_positions))

def compute_proxy_grid(h5: h5py.File, key: str, image_ids: list[str], masks: np.ndarray, mask_sums: np.ndarray, height: int, width: int) -> np.ndarray:
    grid = np.full((NUM_HEADS, NUM_LAYERS), np.nan, dtype=np.float32)
    k = max(1, round(height * width * (100 - PERCENTILE) / 100.0))
    if key not in h5:
        return grid
    for layer_idx in range(NUM_LAYERS):
        layer_key = f'layer{layer_idx}'
        if layer_key not in h5[key]:
            continue
        for head_idx in range(NUM_HEADS):
            method_key = f'mean_head{head_idx}'
            if method_key not in h5[key][layer_key]:
                continue
            grid[head_idx, layer_idx], _ = mean_iou_for_group(
                h5[key][layer_key][method_key], image_ids, masks, mask_sums, k
            )
    return grid

def top_cells(grid: np.ndarray, k: int) -> list[tuple[int, int, float, int]]:
    valid = np.argwhere(np.isfinite(grid))
    ranked = sorted(((int(h), int(layer), float(grid[h, layer])) for h, layer in valid), key=lambda x: x[2], reverse=True)
    return [(h, layer, score, rank + 1) for rank, (h, layer, score) in enumerate(ranked[:k])]

conn = sqlite3.connect(DB_PATH)
image_ids, masks, mask_sums, height, width = prepare_masks()
grids: dict[tuple[str, str], np.ndarray] = {}
with h5py.File(H5_PATH, 'r') as h5:
    for base, _label, method, mode in MODELS:
        for variant_id, *_ in VARIANTS:
            key = model_key(base, variant_id)
            if mode == 'native':
                grids[(base, variant_id)] = load_native_grid(conn, key, method)
            else:
                print(f'  computing proxy grid for {key}', flush=True)
                grids[(base, variant_id)] = compute_proxy_grid(h5, key, image_ids, masks, mask_sums, height, width)
conn.close()

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'font.size': 7.5,
    'axes.titlesize': 9.0,
    'axes.labelsize': 8.2,
    'xtick.labelsize': 7.2,
    'ytick.labelsize': 7.2,
})
fig, axes = plt.subplots(2, 3, figsize=(5.8, 4.75), constrained_layout=False)
fig.patch.set_facecolor('white')

for idx, (base, label, _method, _mode) in enumerate(MODELS):
    ax = axes.flat[idx]
    ax.set_title(label, fontweight='bold', pad=5)
    ax.set_xlim(-0.85, 12.0)
    ax.set_ylim(-0.85, 12.0)
    ax.set_aspect('equal', adjustable='box')
    ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_yticks([0, 3, 6, 9, 11])
    ax.set_xticks(np.arange(0, 12, 1), minor=True)
    ax.set_yticks(np.arange(0, 12, 1), minor=True)
    ax.grid(which='minor', color='#E9E9E9', linewidth=0.40)
    ax.grid(which='major', color='#CCCCCC', linewidth=0.58)
    ax.tick_params(which='minor', length=0)
    ax.tick_params(which='major', length=2.2, width=0.6, color='#555555')
    for spine in ax.spines.values():
        spine.set_color('#444444')
        spine.set_linewidth(0.7)

    for variant_id, _variant_label, color, marker, (variant_dx, variant_dy) in VARIANTS:
        for head, layer, _score, rank in top_cells(grids[(base, variant_id)], TOP_K):
            rank_dx, rank_dy = RANK_JITTER[rank]
            ax.scatter(
                layer + variant_dx + rank_dx,
                head + variant_dy + rank_dy,
                s=92,
                marker=marker,
                facecolors=color,
                edgecolors=color,
                linewidths=0.0,
                alpha=0.68,
                antialiaseds=False,
                zorder=3,
            )

    if idx in (3, 4, 5):
        ax.set_xlabel('Layer')
    if idx in (0, 3):
        ax.set_ylabel('Head')

fig.suptitle('Layer-Head Locations', x=0.5, y=0.995, fontsize=11.4, fontweight='bold')
fig.text(
    0.5,
    0.932,
    'Top three layer-head locations per training variant, ranked by mean IoU@90.',
    ha='center',
    va='center',
    fontsize=7.35,
    color='#222222',
)

variant_handles = [
    Line2D([0], [0], marker=marker, linestyle='None', label=variant_label, markerfacecolor=color, markeredgecolor=color, markeredgewidth=0.0, markersize=7.4, alpha=0.68)
    for _variant_id, variant_label, color, marker, _offset in VARIANTS
]
legend_font = FontProperties(weight='bold', size=8.25)
fig.legend(
    handles=variant_handles,
    loc='lower center',
    bbox_to_anchor=(0.5, 0.0),
    ncol=3,
    frameon=False,
    prop=legend_font,
    handletextpad=0.62,
    columnspacing=1.55,
)

fig.subplots_adjust(left=0.085, right=0.985, top=0.865, bottom=0.14, wspace=0.30, hspace=0.32)
OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
fig.savefig(OUT_PATH, dpi=500, bbox_inches='tight', facecolor='white')
print(f'Saved {OUT_PATH} ({OUT_PATH.stat().st_size:,} bytes)')

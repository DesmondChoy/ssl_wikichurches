#!/usr/bin/env python3
"""Print the appendix head-feature matrices (reviewer item R1-Q17) as LaTeX tables.

For each CLS model, one table holds the frozen IoU@90 of every head at the dominant head's
layer, one row per feature with at least MIN_BBOXES boxes, read from metrics.db. The first
rows of the DINOv3 table match the head-feature matrix in Figure 4.
"""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from itertools import groupby
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DB_PATH = ROOT / 'outputs' / 'cache' / 'metrics.db'
CLS_MODELS = [('dinov2', 'DINOv2'), ('dinov3', 'DINOv3'), ('mae', 'MAE'), ('clip', 'CLIP')]
MIN_BBOXES = 3
NUM_HEADS = 12
# The 14 columns fit the 372 pt text width only if no feature name is wider than about 20 characters.
MAX_NAME_CHARS = 20


@dataclass(frozen=True)
class FeatureRow:
    label: int
    name: str
    bbox_count: int
    scores: list[float]


@dataclass(frozen=True)
class HeadFeatureMatrix:
    model_key: str
    model_name: str
    layer: str
    dominant_head: int
    rows: list[FeatureRow]


def feature_rows(conn: sqlite3.Connection, key: str, layer: str) -> list[FeatureRow]:
    """Every head's IoU@90 per feature at one layer, indexed by head."""
    records = conn.execute(
        "select feature_label, feature_name, head, bbox_count, mean_score from head_feature_metrics "
        "where model = ? and layer = ? and method = 'cls' and metric = 'iou' and percentile = 90 "
        "and bbox_count >= ? order by feature_label, head",
        (key, layer, MIN_BBOXES),
    ).fetchall()
    rows: list[FeatureRow] = []
    for label, group in groupby(records, key=lambda r: r[0]):
        cells = list(group)
        heads = [c[2] for c in cells]
        if heads != list(range(NUM_HEADS)):
            raise ValueError(f'{key} {layer} feature {label}: heads {heads}, expected 0..{NUM_HEADS - 1}')
        bbox_count = cells[0][3]
        if any(c[3] != bbox_count for c in cells):
            raise ValueError(f'{key} {layer} feature {label}: bbox_count differs across heads')
        rows.append(FeatureRow(label, cells[0][1], bbox_count, [c[4] for c in cells]))
    return rows


def load_matrix(conn: sqlite3.Connection, key: str, name: str) -> HeadFeatureMatrix:
    """Feature rows at the dominant frozen head's layer, sorted by that head."""
    layer, head = conn.execute(
        "select layer, head from head_summary_metrics where model = ? and method = 'cls' "
        "and metric = 'iou' and percentile = 90 order by mean_score desc limit 1",
        (key,),
    ).fetchone()
    rows = sorted(feature_rows(conn, key, layer), key=lambda r: (-r.scores[head], r.label))
    return HeadFeatureMatrix(key, name, layer, head, rows)


def name_cell(name: str) -> str:
    """The name on one line, or split over two lines of near-equal length if it is too long."""
    if len(name) <= MAX_NAME_CHARS:
        return name
    words = name.split()
    split = min(
        range(1, len(words)),
        key=lambda i: abs(len(' '.join(words[:i])) - len(' '.join(words[i:]))),
    )
    return rf"\begin{{tabular}}[c]{{@{{}}l@{{}}}}{' '.join(words[:split])}\\{' '.join(words[split:])}\end{{tabular}}"


def bold_dominant(cells: list[str], dominant_head: int) -> list[str]:
    return [rf'\textbf{{{c}}}' if h == dominant_head else c for h, c in enumerate(cells)]


def render(matrix: HeadFeatureMatrix) -> str:
    dominant = f'H{matrix.dominant_head}'
    caption = (
        f"Head-feature matrix for frozen {matrix.model_name} at layer~{matrix.layer.removeprefix('layer')} "
        f'using IoU@90, with all {len(matrix.rows)} features that have at least three boxes. '
        'Each cell is the mean IoU@90 of one attention head over the expert-annotated boxes of one '
        'feature, and Boxes is the number of these boxes. '
        f'Bold marks {dominant}, the dominant head. Rows are sorted by the {dominant} score.'
    )
    heads = bold_dominant([f'H{h}' for h in range(NUM_HEADS)], matrix.dominant_head)
    col_spec = '@{}lr' + 'c' * NUM_HEADS + '@{}'
    lines = [
        r'\begin{table}[!htbp]',
        r'\centering',
        rf'\caption{{{caption}\label{{table appendix head feature {matrix.model_key}}}}}',
        r'\setlength{\tabcolsep}{3pt}',
        rf'\begin{{tabular}}{{{col_spec}}}',
        r'\toprule',
        ' & '.join(['Feature', 'Boxes', *heads]) + r' \\ \midrule',
    ]
    for row in matrix.rows:
        cells = bold_dominant([f'{s:.3f}'.removeprefix('0') for s in row.scores], matrix.dominant_head)
        lines.append(' & '.join([name_cell(row.name), str(row.bbox_count), *cells]) + r' \\')
    lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


def main() -> None:
    with sqlite3.connect(f'{DB_PATH.as_uri()}?mode=ro', uri=True) as conn:
        matrices = [load_matrix(conn, key, name) for key, name in CLS_MODELS]
    print('\n\n'.join(render(m) for m in matrices))


if __name__ == '__main__':
    main()

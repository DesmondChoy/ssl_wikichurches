#!/usr/bin/env python3
"""Generate paper Figure 4: two strongest and two weakest feature alignments per model.

Rows come from metrics.db (frozen IoU@90 per head and per feature). Crops come from
the cached attention maps in attention_viz.h5.
"""

from __future__ import annotations

import sqlite3
import textwrap
from pathlib import Path
from typing import Any

import h5py
import torch
from PIL import Image, ImageDraw, ImageFont

from ssl_attention.data.annotations import BoundingBox, ImageAnnotation
from ssl_attention.data.wikichurches import AnnotatedSubset
from ssl_attention.metrics.iou import compute_per_bbox_iou

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / 'docs' / 'mva' / '633_fig4.png'

FONT_REG = '/System/Library/Fonts/Supplemental/Arial.ttf'
FONT_BOLD = '/System/Library/Fonts/Supplemental/Arial Bold.ttf'

def font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    return ImageFont.truetype(FONT_BOLD if bold else FONT_REG, size)

F_MODEL = font(34, True)
F_LAYER = font(24)
F_HEADER = font(32, True)
F_TITLE = font(28, True)
F_META = font(20)
F_FOOT = font(21)

TEAL = '#00a99a'
TEAL_LIGHT = '#f1fffc'
RED = '#ef4444'
RED_DARK = '#991b1b'
RED_LIGHT = '#fff7f7'
INK = '#101828'
MUTED = '#52657d'
GRID = '#d5dde8'
BOX_GREEN = '#18bf65'

DB_PATH = ROOT / 'outputs' / 'cache' / 'metrics.db'
CLS_MODELS = [('dinov2', 'DINOv2'), ('dinov3', 'DINOv3'), ('mae', 'MAE'), ('clip', 'CLIP')]
PROXY_MODELS = [('siglip', 'SigLIP'), ('siglip2', 'SigLIP2')]
MIN_BBOXES = 3
FeatureItem = tuple[str, int, str, float]


def feature_items(conn: sqlite3.Connection, table: str, where: str, params: tuple[object, ...]) -> list[FeatureItem]:
    """Two strongest and two weakest IoU@90 features with at least MIN_BBOXES boxes."""
    rows = conn.execute(
        f"select feature_label, feature_name, mean_score from {table} "
        f"where {where} and metric = 'iou' and percentile = 90 and bbox_count >= ?",
        (*params, MIN_BBOXES),
    ).fetchall()
    top = sorted(rows, key=lambda r: (-r[2], r[0]))[:2]
    weak = sorted(rows, key=lambda r: (r[2], r[0]))[:2]
    return [('top', *r) for r in top] + [('weak', *r) for r in weak]


def build_rows(conn: sqlite3.Connection) -> list[dict[str, Any]]:
    """Each model's dominant frozen head (CLS) or best mean-proxy layer (SigLIP family)."""
    rows: list[dict[str, Any]] = []
    for key, name in CLS_MODELS:
        layer, head = conn.execute(
            "select layer, head from head_summary_metrics where model = ? and method = 'cls' "
            "and metric = 'iou' and percentile = 90 order by mean_score desc limit 1",
            (key,),
        ).fetchone()
        items = feature_items(
            conn, 'head_feature_metrics', "model = ? and layer = ? and head = ? and method = 'cls'", (key, layer, head)
        )
        label = f"L{layer.removeprefix('layer')}/H{head}"
        rows.append({'model': name, 'label': label, 'model_key': key, 'layer': layer, 'method': f'cls_head{head}', 'items': items})
    for key, name in PROXY_MODELS:
        (layer,) = conn.execute(
            "select layer from aggregate_metrics where model = ? and method = 'mean' and percentile = 90 "
            "order by mean_iou desc limit 1",
            (key,),
        ).fetchone()
        items = feature_items(conn, 'feature_metrics', "model = ? and layer = ? and method = 'mean'", (key, layer))
        label = f"L{layer.removeprefix('layer')}/mean proxy"
        rows.append({'model': name, 'label': label, 'model_key': key, 'layer': layer, 'method': 'mean', 'items': items})
    return rows


with sqlite3.connect(DB_PATH) as _conn:
    ROWS = build_rows(_conn)
COLS = [('Top 1', TEAL), ('Top 2', TEAL), ('Weak 1', RED_DARK), ('Weak 2', RED_DARK)]

CANVAS_W = 2280
MARGIN_X = 58
ROW_LABEL_W = 178
COL_W = 470
COL_GAP = 42
HEADER_Y = 50
GRID_Y = 118
ROW_H = 382
ROW_GAP = 30
IMG_W = 430
IMG_H = 246
FRAME_PAD_X = (COL_W - IMG_W) // 2
FOOT_H = 58
CANVAS_H = GRID_Y + len(ROWS) * ROW_H + (len(ROWS) - 1) * ROW_GAP + FOOT_H + 34
canvas = Image.new('RGB', (CANVAS_W, CANVAS_H), '#ffffff')
draw = ImageDraw.Draw(canvas)

def text_size(text: str, fnt: ImageFont.FreeTypeFont) -> tuple[float, float]:
    box = draw.textbbox((0, 0), text, font=fnt)
    return box[2] - box[0], box[3] - box[1]

def ellipsize(text: str, fnt: ImageFont.FreeTypeFont, max_w: int) -> str:
    if text_size(text, fnt)[0] <= max_w:
        return text
    ell = '...'
    lo, hi = 0, len(text)
    while lo < hi:
        mid = (lo + hi + 1) // 2
        candidate = text[:mid].rstrip() + ell
        if text_size(candidate, fnt)[0] <= max_w:
            lo = mid
        else:
            hi = mid - 1
    return text[:lo].rstrip() + ell

def wrap_lines(text: str, fnt: ImageFont.FreeTypeFont, max_w: int, max_lines: int = 2) -> list[str]:
    words = text.split()
    lines: list[str] = []
    idx = 0
    for line_no in range(max_lines):
        line = ''
        while idx < len(words):
            candidate = words[idx] if not line else f'{line} {words[idx]}'
            suffix = '...' if line_no == max_lines - 1 and idx < len(words) - 1 else ''
            if text_size(candidate + suffix, fnt)[0] <= max_w:
                line = candidate
                idx += 1
            else:
                break
        if not line and idx < len(words):
            line = ellipsize(words[idx], fnt, max_w)
            idx += 1
        if line:
            lines.append(line)
        if idx >= len(words):
            break
    if idx < len(words) and lines:
        lines[-1] = ellipsize(lines[-1] + ' ' + ' '.join(words[idx:]), fnt, max_w)
    return lines

def choose_best_bbox(
    feature_label: int, annotations: dict[str, ImageAnnotation], h5_group: h5py.Group
) -> tuple[float, str, int, float]:
    rows = []
    for image_id, ann in annotations.items():
        if image_id not in h5_group or not any(b.label == feature_label for b in ann.bboxes):
            continue
        attention = torch.tensor(h5_group[image_id]['attention'][()], dtype=torch.float32)
        vals = [(idx, score) for idx, (label, score) in enumerate(compute_per_bbox_iou(attention, ann, 90)) if label == feature_label]
        if vals:
            image_score = sum(score for _, score in vals) / len(vals)
            best_bbox_index, best_bbox_score = sorted(vals, key=lambda item: (-item[1], item[0]))[0]
            rows.append((image_score, image_id, best_bbox_index, best_bbox_score))
    if not rows:
        raise RuntimeError(f'No candidate for feature {feature_label}')
    return sorted(rows, key=lambda row: (-row[0], row[1]))[0]

def crop_box_for_bbox(bbox: BoundingBox, image_w: int, image_h: int, aspect: float) -> tuple[int, int, int, int]:
    x1 = bbox.left * image_w
    y1 = bbox.top * image_h
    x2 = bbox.right * image_w
    y2 = bbox.bottom * image_h
    bw = max(1.0, x2 - x1)
    bh = max(1.0, y2 - y1)
    cx = (x1 + x2) / 2
    cy = (y1 + y2) / 2
    frac = (bw * bh) / max(1.0, image_w * image_h)
    long_frac = max(bw / image_w, bh / image_h)
    if frac < 0.002:
        scale = 7.2
    elif frac < 0.010:
        scale = 5.0
    elif long_frac < 0.20:
        scale = 3.4
    else:
        scale = 2.35
    cw = max(bw * scale, image_w * 0.24)
    ch = max(bh * scale, image_h * 0.20)
    if cw / ch < aspect:
        cw = ch * aspect
    else:
        ch = cw / aspect
    if cw > image_w:
        cw = image_w
        ch = min(image_h, cw / aspect)
    if ch > image_h:
        ch = image_h
        cw = min(image_w, ch * aspect)
    left = min(max(0.0, cx - cw / 2), image_w - cw)
    top = min(max(0.0, cy - ch / 2), image_h - ch)
    return (int(round(left)), int(round(top)), int(round(left + cw)), int(round(top + ch)))

def draw_crop_with_bbox(
    x: int, y: int, image_id: str, bbox: BoundingBox, border_color: str, bg_color: str
) -> None:
    src = Image.open(ROOT / 'dataset' / 'images' / image_id).convert('RGB')
    sw, sh = src.size
    crop_rect = crop_box_for_bbox(bbox, sw, sh, IMG_W / IMG_H)
    crop = src.crop(crop_rect).resize((IMG_W, IMG_H), Image.Resampling.LANCZOS)
    draw.rounded_rectangle((x - 8, y - 8, x + IMG_W + 8, y + IMG_H + 8), radius=11, fill=bg_color, outline=border_color, width=4)
    canvas.paste(crop, (x, y))
    overlay = ImageDraw.Draw(canvas)
    sx = IMG_W / (crop_rect[2] - crop_rect[0])
    sy = IMG_H / (crop_rect[3] - crop_rect[1])
    bx1 = x + (bbox.left * sw - crop_rect[0]) * sx
    by1 = y + (bbox.top * sh - crop_rect[1]) * sy
    bx2 = x + (bbox.right * sw - crop_rect[0]) * sx
    by2 = y + (bbox.bottom * sh - crop_rect[1]) * sy
    bx1 = max(x + 2, min(x + IMG_W - 2, bx1))
    bx2 = max(x + 2, min(x + IMG_W - 2, bx2))
    by1 = max(y + 2, min(y + IMG_H - 2, by1))
    by2 = max(y + 2, min(y + IMG_H - 2, by2))
    overlay.rectangle((bx1, by1, bx2, by2), outline='white', width=9)
    overlay.rectangle((bx1, by1, bx2, by2), outline=BOX_GREEN, width=5)

first_col_x = MARGIN_X + ROW_LABEL_W + 22
for i, (label, color) in enumerate(COLS):
    x = first_col_x + i * (COL_W + COL_GAP)
    tw, _ = text_size(label, F_HEADER)
    draw.text((x + (COL_W - tw) / 2, HEADER_Y), label, font=F_HEADER, fill=color)

annotated = AnnotatedSubset(ROOT / 'dataset')
image_annotations = annotated.annotations
selected = {}
with h5py.File(ROOT / 'outputs/cache/attention_viz.h5', 'r') as f:
    for row in ROWS:
        group = f[row['model_key']][row['layer']][row['method']]
        for _, feature_label, *_ in row['items']:
            selected[(row['model_key'], row['layer'], row['method'], feature_label)] = choose_best_bbox(feature_label, image_annotations, group)

for r, row in enumerate(ROWS):
    row_y = GRID_Y + r * (ROW_H + ROW_GAP)
    if r > 0:
        draw.line((MARGIN_X, row_y - ROW_GAP // 2, CANVAS_W - MARGIN_X, row_y - ROW_GAP // 2), fill=GRID, width=1)
    label_y = row_y + 102
    draw.text((MARGIN_X, label_y), row['model'], font=F_MODEL, fill=INK)
    draw.text((MARGIN_X, label_y + 42), row['label'], font=F_LAYER, fill=MUTED)
    for c, (kind, feature_label, feature_name, feature_mean) in enumerate(row['items']):
        x0 = first_col_x + c * (COL_W + COL_GAP)
        image_x = x0 + FRAME_PAD_X
        image_y = row_y + 8
        border = TEAL if kind == 'top' else RED
        bg = TEAL_LIGHT if kind == 'top' else RED_LIGHT
        shown_crop, image_id, bbox_index, _bbox_score = selected[(row['model_key'], row['layer'], row['method'], feature_label)]
        draw_crop_with_bbox(image_x, image_y, image_id, image_annotations[image_id].bboxes[bbox_index], border, bg)
        text_x = image_x - 3
        text_y = image_y + IMG_H + 22
        title_lines = wrap_lines(feature_name, F_TITLE, IMG_W + 6, max_lines=2)
        for line_no, line in enumerate(title_lines):
            draw.text((text_x, text_y + line_no * 32), line, font=F_TITLE, fill=INK)
        metric = f'feature mean {feature_mean:.4f} | shown crop {shown_crop:.4f}'
        metric_font = F_META if text_size(metric, F_META)[0] <= IMG_W + 6 else font(18)
        draw.text((text_x, text_y + len(title_lines) * 32 + 10), metric, font=metric_font, fill=MUTED)

footnote = 'Frozen IoU@90 feature cells. SigLIP-family rows use mean-attention proxy maps; DINO/MAE/CLIP rows use CLS-head maps.'
footer_y = CANVAS_H - FOOT_H
for i, line in enumerate(textwrap.wrap(footnote, width=140)):
    draw.text((first_col_x, footer_y + i * 26), line, font=F_FOOT, fill=MUTED)

canvas.save(OUT, optimize=True)
print(OUT.resolve())
print(f'{CANVAS_W}x{CANVAS_H}')

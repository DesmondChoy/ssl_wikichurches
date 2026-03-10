"""Continuous metrics for threshold-free attention alignment.

This module adds Gaussian soft ground-truth generation and mean squared error
evaluation on the cached 224x224 attention heatmaps used by the visualization
app. Unlike percentile-thresholded IoU, these metrics compare dense heatmaps
directly and therefore do not depend on a threshold selection.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import torch
from torch import Tensor

from ssl_attention.config import EPSILON

if TYPE_CHECKING:
    from ssl_attention.data.annotations import BoundingBox, ImageAnnotation


def normalize_attention_for_mse(attention: Tensor) -> Tensor:
    """Normalize an attention heatmap into the repo's [0, 1] heatmap range."""
    normalized = attention.float().nan_to_num(nan=0.0, posinf=1.0, neginf=0.0)

    min_val = normalized.min().item()
    max_val = normalized.max().item()
    if min_val < 0.0 or max_val > 1.0:
        scale = max_val - min_val
        if scale > EPSILON:
            normalized = (normalized - min_val) / scale
        elif max_val <= 0.0:
            normalized = torch.zeros_like(normalized)
        else:
            normalized = torch.ones_like(normalized)

    return normalized.clamp(0.0, 1.0)


def gaussian_bbox_heatmap(
    bbox: BoundingBox,
    height: int,
    width: int,
    *,
    device: torch.device | None = None,
) -> Tensor:
    """Generate an anisotropic Gaussian target heatmap for a single bbox."""
    dtype = torch.float32
    x_coords = torch.arange(width, dtype=dtype, device=device) + 0.5
    y_coords = torch.arange(height, dtype=dtype, device=device) + 0.5
    yy, xx = torch.meshgrid(y_coords, x_coords, indexing="ij")

    bbox_width_px = max((bbox.right - bbox.left) * width, 1.0)
    bbox_height_px = max((bbox.bottom - bbox.top) * height, 1.0)

    center_x = ((bbox.left + bbox.right) * width) / 2.0
    center_y = ((bbox.top + bbox.bottom) * height) / 2.0

    sigma_x = max(bbox_width_px / 4.0, 1.0)
    sigma_y = max(bbox_height_px / 4.0, 1.0)

    exponent = ((xx - center_x) ** 2) / (2.0 * sigma_x**2) + ((yy - center_y) ** 2) / (
        2.0 * sigma_y**2
    )
    heatmap = torch.exp(-exponent)
    max_value = heatmap.max()
    if max_value > 0:
        heatmap = heatmap / max_value
    return heatmap


def soft_union_heatmap(heatmaps: list[Tensor]) -> Tensor:
    """Combine multiple soft targets using pixelwise max."""
    if not heatmaps:
        raise ValueError("soft_union_heatmap requires at least one heatmap")

    union = heatmaps[0]
    for heatmap in heatmaps[1:]:
        union = torch.maximum(union, heatmap)

    max_value = union.max()
    if max_value > 0:
        union = union / max_value
    return union.clamp(0.0, 1.0)


def annotation_to_gaussian_heatmap(
    annotation: ImageAnnotation,
    height: int,
    width: int,
    *,
    device: torch.device | None = None,
) -> Tensor:
    """Generate a soft-union Gaussian heatmap for all bboxes in an annotation."""
    if not annotation.bboxes:
        return torch.zeros((height, width), dtype=torch.float32, device=device)

    heatmaps = [
        gaussian_bbox_heatmap(bbox, height, width, device=device)
        for bbox in annotation.bboxes
    ]
    return soft_union_heatmap(heatmaps)


def compute_mse(attention: Tensor, gt_heatmap: Tensor) -> float:
    """Compute mean squared error between normalized attention and soft GT."""
    if attention.shape != gt_heatmap.shape:
        raise ValueError(
            f"Attention and GT heatmap must have the same shape, got {attention.shape} vs {gt_heatmap.shape}"
        )

    if gt_heatmap.device != attention.device:
        gt_heatmap = gt_heatmap.to(attention.device)

    normalized_attention = normalize_attention_for_mse(attention)
    normalized_gt = gt_heatmap.float().nan_to_num(nan=0.0, posinf=1.0, neginf=0.0).clamp(0.0, 1.0)
    return torch.mean((normalized_attention - normalized_gt) ** 2).item()


def compute_image_mse(attention: Tensor, annotation: ImageAnnotation) -> float:
    """Compute MSE between an attention map and an annotation's Gaussian soft union."""
    height, width = attention.shape[-2:]
    gt_heatmap = annotation_to_gaussian_heatmap(annotation, height, width, device=attention.device)
    return compute_mse(attention, gt_heatmap)

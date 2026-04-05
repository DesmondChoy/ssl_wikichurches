"""Tests for Gaussian soft-target generation and MSE alignment metrics."""

from __future__ import annotations

import pytest
import torch

from ssl_attention.data.annotations import BoundingBox, ImageAnnotation
from ssl_attention.metrics import continuous as continuous_module
from ssl_attention.metrics.continuous import (
    annotation_to_gaussian_heatmap,
    compute_emd,
    compute_image_emd,
    compute_image_kl,
    compute_image_mse,
    compute_kl_divergence,
    compute_mse,
    gaussian_bbox_heatmap,
    prepare_emd_distribution,
    soft_union_heatmap,
)


def _make_annotation(*bbox_specs: tuple[float, float, float, float, int]) -> ImageAnnotation:
    bboxes = tuple(
        BoundingBox(left=left, top=top, width=width, height=height, label=label, group_label=0)
        for left, top, width, height, label in bbox_specs
    )
    return ImageAnnotation(image_id="test.jpg", styles=(), bboxes=bboxes)


class TestGaussianGroundTruth:
    """Verify Gaussian GT construction semantics."""

    def test_single_bbox_produces_centered_unit_heatmap(self):
        annotation = _make_annotation((0.25, 0.25, 0.5, 0.5, 1))

        heatmap = annotation_to_gaussian_heatmap(annotation, 100, 100)

        assert heatmap.shape == (100, 100)
        assert heatmap.min().item() >= 0.0
        assert heatmap.max().item() == 1.0
        assert heatmap[50, 50].item() > 0.95
        assert heatmap[50, 50].item() > heatmap[10, 10].item()

    def test_multiple_bboxes_use_pixelwise_max_soft_union(self):
        bbox1 = BoundingBox(left=0.1, top=0.1, width=0.2, height=0.2, label=1, group_label=0)
        bbox2 = BoundingBox(left=0.7, top=0.7, width=0.2, height=0.2, label=2, group_label=0)
        annotation = ImageAnnotation(image_id="test.jpg", styles=(), bboxes=(bbox1, bbox2))

        combined = annotation_to_gaussian_heatmap(annotation, 100, 100)
        expected = soft_union_heatmap(
            [
                gaussian_bbox_heatmap(bbox1, 100, 100),
                gaussian_bbox_heatmap(bbox2, 100, 100),
            ]
        )

        assert torch.allclose(combined, expected)

    def test_empty_annotations_produce_zero_heatmap(self):
        annotation = ImageAnnotation(image_id="empty.jpg", styles=(), bboxes=())

        heatmap = annotation_to_gaussian_heatmap(annotation, 32, 32)

        assert torch.count_nonzero(heatmap).item() == 0


class TestComputeMse:
    """Verify MSE behaves sensibly for aligned and misaligned attention."""

    def test_identical_attention_and_gt_have_near_zero_mse(self):
        annotation = _make_annotation((0.25, 0.25, 0.5, 0.5, 1))
        gt = annotation_to_gaussian_heatmap(annotation, 64, 64)

        mse = compute_mse(gt, gt)

        assert mse == 0.0

    def test_spatial_shift_increases_mse(self):
        annotation = _make_annotation((0.2, 0.2, 0.3, 0.3, 1))
        gt = annotation_to_gaussian_heatmap(annotation, 64, 64)
        shifted = torch.roll(gt, shifts=10, dims=1)

        assert compute_mse(gt, gt) < compute_mse(shifted, gt)

    def test_uniform_and_random_attention_are_worse_than_aligned_attention(self):
        annotation = _make_annotation((0.3, 0.3, 0.25, 0.25, 1))
        gt = annotation_to_gaussian_heatmap(annotation, 64, 64)
        uniform = torch.full_like(gt, 0.5)
        torch.manual_seed(7)
        random_attention = torch.rand_like(gt)

        aligned = compute_mse(gt, gt)
        uniform_mse = compute_mse(uniform, gt)
        random_mse = compute_mse(random_attention, gt)

        assert aligned < uniform_mse
        assert aligned < random_mse

    def test_empty_annotations_stay_finite(self):
        annotation = ImageAnnotation(image_id="empty.jpg", styles=(), bboxes=())
        attention = torch.rand(32, 32)

        mse = compute_image_mse(attention, annotation)

        assert torch.isfinite(torch.tensor(mse))
        assert 0.0 <= mse <= 1.0


class TestComputeKlDivergence:
    """Verify KL(GT || attention) behaves sensibly and stays finite."""

    def test_identical_distributions_have_near_zero_kl(self):
        annotation = _make_annotation((0.25, 0.25, 0.5, 0.5, 1))
        gt = annotation_to_gaussian_heatmap(annotation, 64, 64)

        kl = compute_kl_divergence(gt, gt)

        assert kl == pytest.approx(0.0, abs=1e-8)

    def test_kl_is_non_negative_and_finite_for_sparse_inputs(self):
        annotation = _make_annotation((0.2, 0.2, 0.3, 0.3, 1))
        gt = annotation_to_gaussian_heatmap(annotation, 32, 32)
        attention = torch.tensor(
            [[float("nan"), -1.0], [float("inf"), 0.0]],
            dtype=torch.float32,
        )

        kl = compute_kl_divergence(attention, gt[:2, :2])

        assert torch.isfinite(torch.tensor(kl))
        assert kl >= 0.0

    def test_controlled_probability_mass_shift_increases_kl(self):
        annotation = _make_annotation((0.2, 0.2, 0.3, 0.3, 1))
        gt = annotation_to_gaussian_heatmap(annotation, 64, 64)
        shifted_small = torch.roll(gt, shifts=4, dims=1)
        shifted_large = torch.roll(gt, shifts=12, dims=1)

        aligned = compute_kl_divergence(gt, gt)
        small_shift = compute_kl_divergence(shifted_small, gt)
        large_shift = compute_kl_divergence(shifted_large, gt)

        assert aligned <= small_shift
        assert small_shift < large_shift

    def test_empty_annotations_stay_finite(self):
        annotation = ImageAnnotation(image_id="empty.jpg", styles=(), bboxes=())
        attention = torch.rand(32, 32)

        kl = compute_image_kl(attention, annotation)

        assert torch.isfinite(torch.tensor(kl))
        assert kl >= 0.0


class TestComputeEmd:
    """Verify EMD/Wasserstein-1 behaves sensibly for spatial mismatches."""

    def test_identical_distributions_have_near_zero_emd(self):
        annotation = _make_annotation((0.25, 0.25, 0.5, 0.5, 1))
        gt = annotation_to_gaussian_heatmap(annotation, 64, 64)

        emd = compute_emd(gt, gt)

        assert emd == pytest.approx(0.0, abs=1e-10)

    def test_larger_spatial_shift_increases_emd(self):
        annotation = _make_annotation((0.2, 0.2, 0.3, 0.3, 1))
        gt = annotation_to_gaussian_heatmap(annotation, 64, 64)
        shifted_small = torch.roll(gt, shifts=4, dims=1)
        shifted_large = torch.roll(gt, shifts=12, dims=1)

        aligned = compute_emd(gt, gt)
        small_shift = compute_emd(shifted_small, gt)
        large_shift = compute_emd(shifted_large, gt)

        assert aligned <= small_shift
        assert small_shift < large_shift

    def test_near_miss_scores_better_than_far_miss(self):
        annotation = _make_annotation((0.3, 0.3, 0.25, 0.25, 1))
        gt = annotation_to_gaussian_heatmap(annotation, 64, 64)
        near_miss = torch.roll(gt, shifts=(2, 2), dims=(0, 1))
        far_miss = torch.roll(gt, shifts=(10, 10), dims=(0, 1))

        assert compute_emd(near_miss, gt) < compute_emd(far_miss, gt)

    def test_falls_back_to_exact_linprog_when_scipy_helper_fails(self, monkeypatch):
        annotation = _make_annotation((0.25, 0.25, 0.5, 0.5, 1))
        gt = annotation_to_gaussian_heatmap(annotation, 64, 64)
        shifted = torch.roll(gt, shifts=6, dims=1)

        monkeypatch.setattr(continuous_module, "wasserstein_distance_nd", lambda *args, **kwargs: None)

        emd = compute_emd(shifted, gt)

        assert torch.isfinite(torch.tensor(emd))
        assert emd > 0.0

    def test_exact_linprog_handles_zero_tail_mass(self, monkeypatch):
        attention = torch.zeros((64, 64), dtype=torch.float32)
        attention[:8, :8] = 1.0
        gt = torch.zeros((64, 64), dtype=torch.float32)
        gt[16:24, 16:24] = 1.0

        # This shape produces a zero-mass final support cell after resizing,
        # which previously made the exact LP fallback numerically infeasible.
        assert prepare_emd_distribution(attention).reshape(-1)[-1].item() == 0.0
        assert prepare_emd_distribution(gt).reshape(-1)[-1].item() == 0.0

        monkeypatch.setattr(continuous_module, "wasserstein_distance_nd", lambda *args, **kwargs: None)

        emd = compute_emd(attention, gt)

        assert torch.isfinite(torch.tensor(emd))
        assert emd > 0.0

    def test_empty_annotations_stay_finite(self):
        annotation = ImageAnnotation(image_id="empty.jpg", styles=(), bboxes=())
        attention = torch.rand(32, 32)

        emd = compute_image_emd(attention, annotation)

        assert torch.isfinite(torch.tensor(emd))
        assert emd >= 0.0

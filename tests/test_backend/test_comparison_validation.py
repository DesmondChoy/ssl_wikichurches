"""Tests for method validation in comparison and all_models endpoints.

Verifies that:
- compare_models returns 400 when a requested method is incompatible with a model.
- all_models silently skips incompatible models instead of falling back to defaults.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from app.backend.main import app

client = TestClient(app)


# Shared test image ID
IMAGE_ID = "Q1234_test"


@pytest.fixture(autouse=True)
def _mock_services():
    """Mock services to avoid needing real data files."""
    mock_annotation = MagicMock()
    mock_annotation.bboxes = []

    with (
        patch("app.backend.routers.comparison.image_service") as mock_img_cmp,
        patch("app.backend.routers.comparison.metrics_service") as mock_met_cmp,
        patch("app.backend.routers.metrics.image_service"),
        patch("app.backend.routers.metrics.metrics_service") as mock_met_all,
    ):
        # comparison router mocks
        mock_img_cmp.get_annotation.return_value = mock_annotation
        mock_img_cmp.heatmap_exists.return_value = False
        mock_met_cmp.db_exists = True
        mock_met_cmp.get_image_metrics.return_value = {
            "image_id": IMAGE_ID,
            "model": "dinov2",
            "layer": "layer0",
            "percentile": 90,
            "iou": 0.5,
            "coverage": 0.6,
            "attention_area": 0.4,
            "annotation_area": 0.3,
            "method": "cls",
        }

        # metrics router mocks
        mock_met_all.db_exists = True
        mock_met_all.get_image_metrics.return_value = {
            "image_id": IMAGE_ID,
            "model": "dinov2",
            "layer": "layer0",
            "percentile": 90,
            "iou": 0.5,
            "coverage": 0.6,
            "attention_area": 0.4,
            "annotation_area": 0.3,
            "method": "cls",
        }

        yield


class TestCompareModelsMethodValidation:
    """compare_models should reject incompatible model+method combos."""

    def test_incompatible_method_returns_400(self):
        """SigLIP does not support rollout — should get 400, not silent fallback."""
        resp = client.get(
            "/api/compare/models",
            params={"image_id": IMAGE_ID, "models": ["siglip2"], "method": "rollout"},
        )
        assert resp.status_code == 400
        assert "not available" in resp.json()["detail"]

    def test_compatible_method_returns_200(self):
        """DINOv2 supports rollout — should succeed."""
        resp = client.get(
            "/api/compare/models",
            params={"image_id": IMAGE_ID, "models": ["dinov2"], "method": "rollout"},
        )
        assert resp.status_code == 200

    def test_resnet_rejects_cls(self):
        """ResNet only supports gradcam — cls should be rejected."""
        resp = client.get(
            "/api/compare/models",
            params={"image_id": IMAGE_ID, "models": ["resnet50"], "method": "cls"},
        )
        assert resp.status_code == 400

    def test_mixed_models_incompatible_method(self):
        """If any model in the list is incompatible, the whole request fails."""
        resp = client.get(
            "/api/compare/models",
            params={
                "image_id": IMAGE_ID,
                "models": ["dinov2", "siglip2"],
                "method": "rollout",
            },
        )
        assert resp.status_code == 400


class TestAllModelsMethodFiltering:
    """all_models should skip incompatible models, not fall back to defaults."""

    def test_rollout_excludes_siglip_and_resnet(self):
        """When method=rollout, siglip2 and resnet50 should be absent."""
        resp = client.get(
            f"/api/metrics/{IMAGE_ID}/all_models",
            params={"method": "rollout"},
        )
        assert resp.status_code == 200
        models = resp.json()["models"]
        assert "siglip2" not in models
        assert "resnet50" not in models

    def test_no_method_includes_all_models(self):
        """Without a method filter, all models should be present."""
        resp = client.get(f"/api/metrics/{IMAGE_ID}/all_models")
        assert resp.status_code == 200
        models = resp.json()["models"]
        # All 6 models should appear (each uses its own default method)
        for m in ("dinov2", "dinov3", "mae", "clip", "siglip2", "resnet50"):
            assert m in models, f"Expected {m} in results when no method filter"

    def test_gradcam_only_includes_resnet(self):
        """method=gradcam should only include resnet50."""
        resp = client.get(
            f"/api/metrics/{IMAGE_ID}/all_models",
            params={"method": "gradcam"},
        )
        assert resp.status_code == 200
        models = resp.json()["models"]
        assert "resnet50" in models
        # ViT models don't support gradcam
        for m in ("dinov2", "dinov3", "mae", "clip", "siglip2"):
            assert m not in models, f"{m} should not appear for method=gradcam"

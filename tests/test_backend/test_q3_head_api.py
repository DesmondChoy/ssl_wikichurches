"""Backend API tests for Q3 per-head endpoints and raw-attention head validation."""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.backend.main import app

client = TestClient(app)

IMAGE_ID = "Q1234_test"


class TestRawAttentionHeadValidation:
    """The raw attention endpoint should validate per-head requests cleanly."""

    def test_rejects_head_for_rollout_method(self):
        with patch("app.backend.routers.attention.attention_service") as mock_attention_service:
            mock_attention_service.exists.return_value = True

            response = client.get(
                f"/api/attention/{IMAGE_ID}/raw",
                params={"model": "dinov2", "layer": 0, "method": "rollout", "head": 3},
            )

        assert response.status_code == 400
        assert "head parameter not supported" in response.json()["detail"]

    def test_rejects_head_for_resnet(self):
        with patch("app.backend.routers.attention.attention_service") as mock_attention_service:
            mock_attention_service.exists.return_value = True

            response = client.get(
                f"/api/attention/{IMAGE_ID}/raw",
                params={"model": "resnet50", "layer": 0, "method": "gradcam", "head": 1},
            )

        assert response.status_code == 400
        assert "head parameter not supported" in response.json()["detail"]

    def test_passes_head_to_attention_service(self):
        payload = {
            "attention": [0.1, 0.2, 0.3, 0.4],
            "shape": [2, 2],
            "min_value": 0.1,
            "max_value": 0.4,
        }
        with patch("app.backend.routers.attention.attention_service") as mock_attention_service:
            mock_attention_service.resolve_variant.return_value = "cls_head5"
            mock_attention_service.exists.return_value = True
            mock_attention_service.get_raw_attention.return_value = payload

            response = client.get(
                f"/api/attention/{IMAGE_ID}/raw",
                params={"model": "dinov2", "layer": 0, "method": "cls", "head": 5},
            )

        assert response.status_code == 200
        assert response.json()["shape"] == [2, 2]
        mock_attention_service.get_raw_attention.assert_called_once_with(
            image_id=IMAGE_ID,
            model="dinov2",
            layer=0,
            method="cls",
            head=5,
        )


class TestQ3MetricsApi:
    """Q3 endpoints should expose metric-generic per-head payloads."""

    def test_head_ranking_endpoint_returns_service_payload(self):
        payload = {
            "model": "dinov2",
            "variant": "frozen",
            "layer": "layer11",
            "method": "cls",
            "metric": "iou",
            "direction": "higher",
            "percentile": 90,
            "supported": True,
            "reason": None,
            "heads": [
                {
                    "head": 3,
                    "mean_score": 0.42,
                    "std_score": 0.04,
                    "mean_rank": 1.2,
                    "top1_count": 6,
                    "top3_count": 11,
                    "image_count": 12,
                }
            ],
        }
        with patch("app.backend.routers.metrics.metrics_service") as mock_metrics_service:
            mock_metrics_service.db_exists = True
            mock_metrics_service.get_head_ranking.return_value = payload

            response = client.get(
                "/api/metrics/model/dinov2/head_ranking",
                params={"layer": 11, "metric": "iou", "percentile": 90, "variant": "frozen"},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["metric"] == "iou"
        assert body["heads"][0]["head"] == 3

    def test_head_feature_matrix_endpoint_returns_unsupported_payload(self):
        payload = {
            "model": "resnet50",
            "variant": "frozen",
            "layer": "layer0",
            "method": None,
            "metric": "coverage",
            "direction": "higher",
            "percentile": 90,
            "supported": False,
            "reason": "Q3 per-head analysis is not supported for model 'resnet50'.",
            "heads": [],
            "features": [],
            "total_feature_types": 0,
        }
        with patch("app.backend.routers.metrics.metrics_service") as mock_metrics_service:
            mock_metrics_service.db_exists = True
            mock_metrics_service.get_head_feature_matrix.return_value = payload

            response = client.get(
                "/api/metrics/model/resnet50/head_feature_matrix",
                params={"layer": 0, "metric": "coverage", "percentile": 90, "variant": "frozen"},
            )

        assert response.status_code == 200
        body = response.json()
        assert body["supported"] is False
        assert body["features"] == []

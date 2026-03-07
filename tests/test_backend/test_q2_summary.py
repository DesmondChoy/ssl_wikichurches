"""Tests for /api/metrics/q2_summary endpoint."""

from __future__ import annotations

from unittest.mock import patch

from fastapi.testclient import TestClient

from app.backend.main import app

client = TestClient(app)


def test_q2_summary_success() -> None:
    payload = {
        "percentiles": [90],
        "timestamp": "2026-03-06T00:00:00",
        "models": {
            "dinov2": {
                "lora": {
                    "90": {
                        "model_name": "dinov2",
                        "strategy_id": "lora",
                        "percentile": 90,
                        "method": "cls",
                        "frozen_mean_iou": 0.3,
                        "finetuned_mean_iou": 0.35,
                        "mean_delta_iou": 0.05,
                        "std_delta_iou": 0.01,
                        "delta_ci_lower": 0.03,
                        "delta_ci_upper": 0.07,
                        "cohens_d": 0.5,
                        "p_value": 0.01,
                        "corrected_p_value": 0.02,
                        "significant": True,
                        "test_name": "paired_ttest",
                        "iou_retention_ratio": 1.16,
                        "num_images": 139,
                        "per_image_deltas": {"img1": 0.1},
                    }
                }
            }
        },
        "strategy_comparisons": {
            "dinov2": {
                "90": [
                    {
                        "model_name": "dinov2",
                        "percentile": 90,
                        "strategy_a": "linear_probe",
                        "strategy_b": "lora",
                        "mean_delta_difference": -0.04,
                        "cohens_d": -0.8,
                        "p_value": 0.01,
                        "corrected_p_value": 0.02,
                        "significant": True,
                        "test_name": "paired_ttest",
                    }
                ]
            }
        },
    }

    with patch("app.backend.routers.metrics.metrics_service.get_q2_summary", return_value=payload):
        response = client.get("/api/metrics/q2_summary", params={"percentile": 90})

    assert response.status_code == 200
    body = response.json()
    assert "models" in body
    assert body["models"]["dinov2"]["lora"]["90"]["strategy_id"] == "lora"


def test_q2_summary_unavailable_returns_503() -> None:
    with patch("app.backend.routers.metrics.metrics_service.get_q2_summary", return_value=None):
        response = client.get("/api/metrics/q2_summary")

    assert response.status_code == 503

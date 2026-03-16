"""Unit tests for strategy-aware Q2 delta-IoU analysis helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[2] / "experiments" / "scripts" / "analyze_delta_iou.py"
    spec = importlib.util.spec_from_file_location("analyze_delta_iou", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_summarize_delta_includes_retention_ratio() -> None:
    module = _load_module()

    result = module.summarize_delta(
        model_name="dinov2",
        strategy_id="lora",
        percentile=90,
        method="cls",
        frozen_values={"a": 0.2, "b": 0.4},
        finetuned_values={"a": 0.3, "b": 0.5},
    )

    assert result.mean_delta_iou > 0
    assert result.iou_retention_ratio is not None
    assert result.iou_retention_ratio > 1.0


def test_compare_strategies_within_model_returns_pairwise_rows() -> None:
    module = _load_module()

    baseline = module.summarize_delta(
        model_name="dinov2",
        strategy_id="linear_probe",
        percentile=90,
        method="cls",
        frozen_values={"a": 0.4, "b": 0.4},
        finetuned_values={"a": 0.41, "b": 0.39},
    )
    lora = module.summarize_delta(
        model_name="dinov2",
        strategy_id="lora",
        percentile=90,
        method="cls",
        frozen_values={"a": 0.4, "b": 0.4},
        finetuned_values={"a": 0.45, "b": 0.44},
    )

    output = module.compare_strategies_within_model(
        model_name="dinov2",
        strategy_results={
            "linear_probe": {90: baseline},
            "lora": {90: lora},
        },
        percentiles=[90],
    )

    assert 90 in output
    assert len(output[90]) == 1
    row = output[90][0]
    assert row.strategy_a == "linear_probe"
    assert row.strategy_b == "lora"
    assert row.corrected_p_value is not None

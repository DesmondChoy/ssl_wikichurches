"""Unit tests for strategy-aware Q2 metric analysis helpers."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path


def _load_module():
    script_path = Path(__file__).resolve().parents[2] / "experiments" / "scripts" / "analyze_q2_metrics.py"
    spec = importlib.util.spec_from_file_location("analyze_q2_metrics", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_summarize_metric_delta_keeps_metric_metadata() -> None:
    module = _load_module()

    result = module.summarize_metric_delta(
        model_name="dinov2",
        strategy_id="lora",
        metric="mse",
        percentile=None,
        method="cls",
        frozen_values={"a": 0.20, "b": 0.40},
        finetuned_values={"a": 0.18, "b": 0.36},
    )

    assert result.metric == "mse"
    assert result.direction == "lower"
    assert result.percentile is None
    assert result.mean_delta < 0


def test_compare_strategies_within_model_returns_pairwise_rows() -> None:
    module = _load_module()

    baseline = module.summarize_metric_delta(
        model_name="dinov2",
        strategy_id="linear_probe",
        metric="iou",
        percentile=90,
        method="cls",
        frozen_values={"a": 0.4, "b": 0.4},
        finetuned_values={"a": 0.41, "b": 0.39},
    )
    lora = module.summarize_metric_delta(
        model_name="dinov2",
        strategy_id="lora",
        metric="iou",
        percentile=90,
        method="cls",
        frozen_values={"a": 0.4, "b": 0.4},
        finetuned_values={"a": 0.45, "b": 0.44},
    )

    output = module.compare_strategies_within_model(
        model_name="dinov2",
        strategy_results={
            "linear_probe": {"iou": {90: baseline}},
            "lora": {"iou": {90: lora}},
        },
        percentiles=[90],
    )

    assert len(output) == 1
    row = output[0]
    assert row.metric == "iou"
    assert row.percentile == 90
    assert row.strategy_a == "linear_probe"
    assert row.strategy_b == "lora"
    assert row.corrected_p_value is not None


def test_apply_holm_correction_excludes_linear_probe_rows() -> None:
    module = _load_module()

    linear_probe = module.summarize_metric_delta(
        model_name="clip",
        strategy_id="linear_probe",
        metric="iou",
        percentile=90,
        method="cls",
        frozen_values={"a": 0.4, "b": 0.4},
        finetuned_values={"a": 0.4, "b": 0.4},
    )
    lora = module.summarize_metric_delta(
        model_name="clip",
        strategy_id="lora",
        metric="iou",
        percentile=90,
        method="cls",
        frozen_values={"a": 0.4, "b": 0.4},
        finetuned_values={"a": 0.45, "b": 0.44},
    )
    full = module.summarize_metric_delta(
        model_name="dinov2",
        strategy_id="full",
        metric="iou",
        percentile=90,
        method="cls",
        frozen_values={"a": 0.4, "b": 0.4},
        finetuned_values={"a": 0.47, "b": 0.46},
    )

    linear_probe.p_value = 0.001
    lora.p_value = 0.01
    full.p_value = 0.02

    linear_probe_significant = linear_probe.significant

    model_results = {
        "clip": {
            "linear_probe": {"iou": {90: linear_probe}},
            "lora": {"iou": {90: lora}},
        },
        "dinov2": {
            "full": {"iou": {90: full}},
        },
    }

    module.apply_holm_correction(model_results, metric="iou", percentile=90)

    expected = module.multiple_comparison_correction([0.01, 0.02], method="holm", alpha=0.05)

    assert linear_probe.corrected_p_value is None
    assert linear_probe.significant is linear_probe_significant
    assert lora.corrected_p_value == expected[0][0]
    assert lora.significant is expected[0][1]
    assert full.corrected_p_value == expected[1][0]
    assert full.significant is expected[1][1]

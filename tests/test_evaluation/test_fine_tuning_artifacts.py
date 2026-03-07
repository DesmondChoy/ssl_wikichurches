"""Tests for strategy-aware fine-tuning artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from ssl_attention.evaluation.fine_tuning import (
    get_checkpoint_candidates,
    get_checkpoint_filename,
    get_finetuned_cache_key,
    infer_strategy_id,
    load_run_manifest,
)


def test_infer_strategy_id() -> None:
    assert infer_strategy_id(freeze_backbone=True, use_lora=False) == "linear_probe"
    assert infer_strategy_id(freeze_backbone=False, use_lora=True) == "lora"
    assert infer_strategy_id(freeze_backbone=False, use_lora=False) == "full"


def test_checkpoint_filename_strategy_aware() -> None:
    assert get_checkpoint_filename("dinov2", "lora") == "dinov2_lora_finetuned.pt"
    assert get_checkpoint_filename("clip", "linear_probe") == "clip_linear_probe_finetuned.pt"


def test_finetuned_cache_key_strategy_aware() -> None:
    assert get_finetuned_cache_key("mae") == "mae_finetuned"
    assert get_finetuned_cache_key("mae", "full") == "mae_finetuned_full"


def test_checkpoint_candidates_include_legacy_for_full_and_linear_probe() -> None:
    full_candidates = get_checkpoint_candidates("dinov2", strategy_id="full")
    lp_candidates = get_checkpoint_candidates("dinov2", strategy_id="linear_probe")

    assert full_candidates[0].name == "dinov2_full_finetuned.pt"
    assert full_candidates[1].name == "dinov2_finetuned.pt"
    assert lp_candidates[0].name == "dinov2_linear_probe_finetuned.pt"
    assert lp_candidates[1].name == "dinov2_finetuned.pt"


def test_load_run_manifest(tmp_path: Path) -> None:
    manifest = {
        "model": "dinov2",
        "strategy": "lora",
        "seed": 42,
        "epochs": 10,
        "checkpoint_path": "outputs/checkpoints/dinov2_lora_finetuned.pt",
        "split": {"train_samples": 10, "val_samples": 2, "excluded_eval_samples": 1, "val_split": 0.2},
    }
    manifest_path = tmp_path / "dinov2_lora_manifest.json"
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

    loaded = load_run_manifest("dinov2", "lora", manifests_dir=tmp_path)

    assert loaded["model"] == "dinov2"
    assert loaded["strategy"] == "lora"


def test_load_run_manifest_missing(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_run_manifest("dinov2", "full", manifests_dir=tmp_path)

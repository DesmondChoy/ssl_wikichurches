from pathlib import Path

from app.precompute.generate_feature_cache import (
    discover_checkpoints,
    discover_checkpoints_by_strategy,
)


def test_discover_checkpoints_by_strategy_returns_strategy_specific_paths(tmp_path: Path) -> None:
    (tmp_path / "dinov2_lora_finetuned.pt").touch()
    (tmp_path / "dinov2_full_finetuned.pt").touch()
    (tmp_path / "clip_finetuned.pt").touch()

    result = discover_checkpoints_by_strategy(
        tmp_path,
        model_names=["dinov2", "clip"],
        strategies=["lora", "full", "linear_probe"],
    )

    assert result["dinov2"]["lora"] == tmp_path / "dinov2_lora_finetuned.pt"
    assert result["dinov2"]["full"] == tmp_path / "dinov2_full_finetuned.pt"
    assert result["clip"]["full"] == tmp_path / "clip_finetuned.pt"
    assert result["clip"]["linear_probe"] == tmp_path / "clip_finetuned.pt"


def test_discover_checkpoints_prefers_lora_then_full_then_linear_probe(tmp_path: Path) -> None:
    (tmp_path / "dinov2_full_finetuned.pt").touch()
    (tmp_path / "dinov2_lora_finetuned.pt").touch()
    (tmp_path / "mae_finetuned.pt").touch()

    result = discover_checkpoints(tmp_path, model_names=["dinov2", "mae"])

    assert result["dinov2"] == tmp_path / "dinov2_lora_finetuned.pt"
    assert result["mae"] == tmp_path / "mae_finetuned.pt"

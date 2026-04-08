"""Tests for attention-service cache availability discovery."""

from __future__ import annotations

from pathlib import Path

import pytest
import torch

from ssl_attention.cache import AttentionCache


class TestQ3VariantPerHeadAvailability:
    """Verify per-head availability discovery stays fast and metadata-only."""

    def test_discovers_availability_from_hdf5_group_structure(
        self,
        monkeypatch: pytest.MonkeyPatch,
        tmp_path: Path,
    ) -> None:
        """Per-head availability should come from cache groups, not dataset enumeration."""
        from app.backend.services.attention_service import attention_service

        cache = AttentionCache(tmp_path / "attention.h5")
        attention = torch.ones((2, 2))

        cache.store("dinov2", "layer0", "Q100.jpg", attention, variant="cls")
        cache.store("dinov2", "layer0", "Q100.jpg", attention, variant="cls_head0")
        cache.store("openai-clip", "layer1", "Q101.jpg", attention, variant="cls_head2")
        cache.store("mae_finetuned_lora", "layer0", "Q102.jpg", attention, variant="cls_head1")
        cache.store("dinov3_finetuned_full", "layer0", "Q103.jpg", attention, variant="cls_head5")
        cache.store("mae_finetuned_linear_probe", "layer0", "Q104.jpg", attention, variant="cls")

        monkeypatch.setattr(attention_service, "_cache", cache)
        monkeypatch.setattr(attention_service, "_per_head_available_models_cache", None)
        monkeypatch.setattr(attention_service, "_q3_variant_availability_cache", None)
        monkeypatch.setattr(attention_service, "_per_head_available_models_signature", None)

        def fail(*_args: object, **_kwargs: object) -> None:
            raise AssertionError("AttentionCache.list_cached() should not be called")

        monkeypatch.setattr(AttentionCache, "list_cached", fail)

        availability = attention_service.list_q3_variant_per_head_availability()

        assert availability == {
            "clip": {
                "frozen": True,
                "linear_probe": False,
                "lora": False,
                "full": False,
            },
            "dinov2": {
                "frozen": True,
                "linear_probe": False,
                "lora": False,
                "full": False,
            },
            "dinov3": {
                "frozen": False,
                "linear_probe": False,
                "lora": False,
                "full": True,
            },
            "mae": {
                "frozen": False,
                "linear_probe": False,
                "lora": True,
                "full": False,
            },
        }
        assert attention_service.list_models_with_per_head_cache() == ["clip", "dinov2", "dinov3", "mae"]

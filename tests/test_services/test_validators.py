"""Tests for app.backend.validators."""

import pytest
from fastapi import HTTPException

from app.backend.validators import resolve_default_method, validate_method


def test_resolve_default_method_cls_models():
    """Models using CLS attention should resolve to 'cls'."""
    for model in ("dinov2", "dinov3", "mae", "clip"):
        assert resolve_default_method(model) == "cls", f"{model} should default to cls"


def test_resolve_default_method_siglip_alias():
    """SigLIP (passed as frontend alias 'siglip2') should resolve to 'mean'."""
    assert resolve_default_method("siglip2") == "mean"


def test_resolve_default_method_resnet():
    """ResNet-50 should resolve to 'gradcam'."""
    assert resolve_default_method("resnet50") == "gradcam"


# --- validate_method tests ---


class TestValidateMethod:
    """Tests for validate_method."""

    def test_none_returns_default_cls(self):
        """method=None should return the model's default method."""
        assert validate_method("dinov2", None) == "cls"
        assert validate_method("dinov3", None) == "cls"
        assert validate_method("mae", None) == "cls"
        assert validate_method("clip", None) == "cls"

    def test_none_returns_default_mean(self):
        """SigLIP should default to 'mean'."""
        assert validate_method("siglip2", None) == "mean"

    def test_none_returns_default_gradcam(self):
        """ResNet should default to 'gradcam'."""
        assert validate_method("resnet50", None) == "gradcam"

    def test_explicit_cls_accepted(self):
        """Explicit 'cls' should be accepted for ViT models."""
        assert validate_method("dinov2", "cls") == "cls"

    def test_explicit_rollout_accepted(self):
        """Explicit 'rollout' should be accepted for ViT models."""
        assert validate_method("dinov2", "rollout") == "rollout"
        assert validate_method("mae", "rollout") == "rollout"

    def test_invalid_method_string(self):
        """Invalid method string should raise HTTPException."""
        with pytest.raises(HTTPException) as exc_info:
            validate_method("dinov2", "invalid_method")
        assert exc_info.value.status_code == 400
        assert "Invalid method" in exc_info.value.detail

    def test_method_not_available_for_model(self):
        """Valid method but not supported by the model should raise HTTPException."""
        with pytest.raises(HTTPException) as exc_info:
            validate_method("resnet50", "cls")
        assert exc_info.value.status_code == 400
        assert "not available" in exc_info.value.detail

    def test_siglip_rejects_cls(self):
        """SigLIP does not support CLS method."""
        with pytest.raises(HTTPException) as exc_info:
            validate_method("siglip2", "cls")
        assert exc_info.value.status_code == 400

    def test_alias_resolution(self):
        """Aliases should be resolved before method validation."""
        # siglip2 is an alias for siglip
        assert validate_method("siglip2", "mean") == "mean"

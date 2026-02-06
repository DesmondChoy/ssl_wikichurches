"""Tests for app.backend.validators."""

from app.backend.validators import resolve_default_method


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

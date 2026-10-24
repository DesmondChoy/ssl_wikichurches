"""Attention inputs must cover the whole image.

Expert boxes are rasterized on the full image resized to 224x224, so a
processor that center-crops would put attention and masks in different frames.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
import torch
from PIL import Image
from torch import nn
from transformers import AutoImageProcessor

from ssl_attention.evaluation.fine_tuning import FineTunableModel
from ssl_attention.models.base import BaseVisionModel
from ssl_attention.models.clip_model import CLIP
from ssl_attention.models.dinov2 import DINOv2
from ssl_attention.models.dinov3 import DINOv3
from ssl_attention.models.mae import MAE
from ssl_attention.models.siglip import SigLIP
from ssl_attention.models.siglip2 import SigLIP2

WRAPPERS = [DINOv2, DINOv3, MAE, CLIP, SigLIP, SigLIP2]
STRIP_FRACTION = 0.1


def _edge_strip_image(width: int, height: int) -> Image.Image:
    """Black image with a white strip along the left and bottom edges."""
    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    pixels[:, : int(width * STRIP_FRACTION)] = 255
    pixels[height - int(height * STRIP_FRACTION) :, :] = 255
    return Image.fromarray(pixels)


def _wrapper_processor(wrapper: type[BaseVisionModel]) -> Any:
    try:
        return wrapper._load_processor(wrapper.__new__(wrapper))
    except OSError as exc:
        pytest.skip(f"{wrapper.model_id} processor not available locally: {exc}")


@pytest.mark.parametrize("wrapper", WRAPPERS, ids=lambda w: w.model_name)
@pytest.mark.parametrize("size", [(600, 300), (300, 600)], ids=["landscape", "portrait"])
def test_processor_keeps_image_edges(wrapper: type[BaseVisionModel], size: tuple[int, int]):
    processor = _wrapper_processor(wrapper)
    pixel_values = processor(images=[_edge_strip_image(*size)], return_tensors="pt")["pixel_values"]

    assert tuple(pixel_values.shape) == (1, 3, 224, 224)
    channel = pixel_values[0, 0].float()
    interior = channel[:150, 60:].mean()
    assert channel[:150, :10].mean() > interior + 1.0, "left edge strip was cropped away"
    assert channel[-10:, 60:].mean() > interior + 1.0, "bottom edge strip was cropped away"


@pytest.mark.parametrize("wrapper", WRAPPERS, ids=lambda w: w.model_name)
def test_fine_tunable_model_keeps_native_processor_for_classification(
    wrapper: type[BaseVisionModel], monkeypatch: pytest.MonkeyPatch
):
    attention_processor = _wrapper_processor(wrapper)
    monkeypatch.setattr(FineTunableModel, "_load_backbone", lambda self: nn.Linear(1, 1))

    model = FineTunableModel(wrapper.model_name, device=torch.device("cpu"))

    assert model.processor.to_dict() == attention_processor.to_dict()
    native = AutoImageProcessor.from_pretrained(wrapper.model_id)
    assert model.classification_processor.to_dict() == native.to_dict()

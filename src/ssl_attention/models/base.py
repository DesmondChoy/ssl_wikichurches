"""Base class for vision model wrappers."""

from abc import ABC, abstractmethod
from contextlib import contextmanager
from typing import Any, Generator

import torch
from PIL import Image
from torch import Tensor, nn
from transformers import AutoConfig, AutoImageProcessor, BatchFeature

from ssl_attention.models.protocols import ModelOutput
from ssl_attention.utils.device import get_device, get_dtype_for_device


class BaseVisionModel(ABC, nn.Module):
    """Abstract base class for SSL vision model wrappers.

    Provides common functionality for device handling, preprocessing,
    and inference context management. Subclasses must implement
    _load_model() and _extract_output().

    Attributes:
        model_name: Short identifier (e.g., 'dinov2').
        model_id: HuggingFace model identifier.
        patch_size: Pixels per patch (14 or 16).
        embed_dim: Token embedding dimension.
        num_layers: Number of transformer layers.
        num_heads: Attention heads per layer.
        num_registers: Register token count (0 if none).
        device: Compute device.
        dtype: Tensor dtype for inference.
    """

    # Subclasses must define these
    model_name: str
    model_id: str
    patch_size: int
    embed_dim: int
    num_layers: int
    num_heads: int
    num_registers: int = 0

    def __init__(
        self,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
    ) -> None:
        """Initialize the model wrapper.

        Args:
            device: Target device. Auto-detects if None.
            dtype: Tensor dtype. Uses optimal for device if None.
        """
        super().__init__()

        self.device = device or get_device()
        self.dtype = dtype or get_dtype_for_device(self.device)

        # Load processor and model
        self.processor = self._load_processor()
        self.model = self._load_model()

        # Move to device and set eval mode
        self.model = self.model.to(device=self.device, dtype=self.dtype)
        self.model.eval()

    def _load_processor(self) -> Any:
        """Load the image processor for this model.

        Returns:
            HuggingFace image processor.
        """
        return AutoImageProcessor.from_pretrained(self.model_id)

    def _load_config(self) -> AutoConfig:
        """Load model config with attention output enabled.

        Returns:
            HuggingFace model config with output_attentions=True.
        """
        config = AutoConfig.from_pretrained(self.model_id)
        config.output_attentions = True
        return config

    @abstractmethod
    def _load_model(self) -> nn.Module:
        """Load the underlying model.

        Subclasses implement this to load their specific architecture.
        Model should NOT be moved to device here (done in __init__).

        Returns:
            The loaded PyTorch model.
        """
        ...

    @abstractmethod
    def _extract_output(self, model_output: Any) -> ModelOutput:
        """Extract standardized output from model-specific output.

        Args:
            model_output: Raw output from the model's forward pass.

        Returns:
            ModelOutput with cls_token, patch_tokens, and attention_weights.
        """
        ...

    def preprocess(self, images: list[Image.Image]) -> Tensor:
        """Preprocess PIL images for model input.

        Args:
            images: List of PIL Images.

        Returns:
            Tensor of shape (B, C, H, W) on the model's device.
        """
        # Most HuggingFace processors return BatchFeature with 'pixel_values'
        processed: BatchFeature = self.processor(images=images, return_tensors="pt")
        pixel_values: Tensor = processed["pixel_values"]
        return pixel_values.to(device=self.device, dtype=self.dtype)

    @contextmanager
    def inference_context(self) -> Generator[None, None, None]:
        """Context manager for inference (no gradients, eval mode).

        Usage:
            with model.inference_context():
                output = model(images)
        """
        was_training = self.model.training
        self.model.eval()
        try:
            with torch.no_grad():
                yield
        finally:
            if was_training:
                self.model.train()

    def forward(self, images: Tensor) -> ModelOutput:
        """Process preprocessed images through the model.

        Args:
            images: Preprocessed images from preprocess(), shape (B, C, H, W).

        Returns:
            ModelOutput with embeddings and attention weights.
        """
        with self.inference_context():
            # Most HuggingFace ViTs need output_attentions=True
            model_output = self.model(
                pixel_values=images,
                output_attentions=True,
            )
            return self._extract_output(model_output)

    @property
    def image_size(self) -> int:
        """Expected input image size (assumes square images)."""
        # Most ViT models use 224x224
        size = getattr(self.processor, "size", {})
        if isinstance(size, dict):
            return size.get("height", size.get("shortest_edge", 224))
        return 224

    @property
    def num_patches_per_side(self) -> int:
        """Number of patches along one dimension."""
        return self.image_size // self.patch_size

    @property
    def total_patches(self) -> int:
        """Total number of image patches."""
        return self.num_patches_per_side ** 2

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"model_name='{self.model_name}', "
            f"device={self.device}, "
            f"dtype={self.dtype}, "
            f"patches={self.total_patches})"
        )

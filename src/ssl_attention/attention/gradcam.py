"""Gradient-weighted Class Activation Mapping (Grad-CAM) for ViT.

Grad-CAM uses gradients to weight feature maps, showing which spatial
locations are important for a particular output. For ViT, we adapt
this by computing gradients with respect to the CLS token representation.

This serves as a gradient-based baseline for comparison with
attention-based methods.

Reference:
    Selvaraju et al. (2017), "Grad-CAM: Visual Explanations from Deep Networks
    via Gradient-based Localization"
    https://arxiv.org/abs/1610.02391
"""

import math
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional as F

from ssl_attention.config import DEFAULT_IMAGE_SIZE, EPSILON, INTERPOLATION_MODE


class GradCAM:
    """Gradient-weighted Class Activation Mapping for Vision Transformers.

    Unlike attention-based methods, Grad-CAM uses gradients to determine
    importance. For SSL models (no classification head), we use the
    CLS token representation as the target.

    Usage:
        >>> gradcam = GradCAM(model.model)  # Pass the underlying model
        >>> with gradcam.capture():
        ...     output = model(images)
        >>> heatmap = gradcam.compute(output.cls_token, patch_tokens_shape)
    """

    def __init__(
        self,
        model: nn.Module,
        target_layer: str | None = None,
    ) -> None:
        """Initialize Grad-CAM.

        Args:
            model: The vision transformer model.
            target_layer: Name of the layer to get activations from.
                If None, uses the last transformer block's output.
        """
        self.model = model
        self.target_layer = target_layer

        # Storage for forward/backward hooks
        self._activations: Tensor | None = None
        self._gradients: Tensor | None = None
        self._handles: list[Any] = []

    def capture(self) -> "GradCAMContext":
        """Context manager for capturing activations and gradients.

        Returns:
            Context manager that registers hooks on entry and removes on exit.
        """
        return GradCAMContext(self)

    def _register_hooks(self) -> None:
        """Register forward and backward hooks on target layer."""
        target = self._find_target_layer()

        def forward_hook(module: nn.Module, input: Any, output: Any) -> None:
            self._activations = output.detach()

        def backward_hook(module: nn.Module, grad_input: Any, grad_output: Any) -> None:
            self._gradients = grad_output[0].detach()

        self._handles.append(target.register_forward_hook(forward_hook))
        self._handles.append(target.register_full_backward_hook(backward_hook))

    def _remove_hooks(self) -> None:
        """Remove registered hooks."""
        for handle in self._handles:
            handle.remove()
        self._handles.clear()

    def _find_target_layer(self) -> nn.Module:
        """Find the target layer for activation extraction.

        Returns:
            The target module.

        Raises:
            ValueError: If target layer not found.
        """
        if self.target_layer is not None:
            # Navigate to specified layer
            parts = self.target_layer.split(".")
            module = self.model
            for part in parts:
                if part.isdigit():
                    module = module[int(part)]
                else:
                    module = getattr(module, part)
            return module

        # Default: find last transformer block
        # Try common ViT structures
        for attr in ["encoder.layer", "blocks", "transformer.resblocks"]:
            try:
                parts = attr.split(".")
                module = self.model
                for part in parts:
                    module = getattr(module, part)
                # Return the last block
                return module[-1]
            except (AttributeError, TypeError):
                continue

        raise ValueError(
            "Could not find target layer automatically. "
            "Please specify target_layer explicitly."
        )

    def compute(
        self,
        target: Tensor,
        num_patches: int,
        image_size: int = DEFAULT_IMAGE_SIZE,
    ) -> Tensor:
        """Compute Grad-CAM heatmap.

        Args:
            target: Target tensor to backprop from (e.g., CLS token).
                Shape (B, D).
            num_patches: Number of image patches.
            image_size: Original image size for upsampling.

        Returns:
            Grad-CAM heatmap of shape (B, H, W).
        """
        if self._activations is None or self._gradients is None:
            raise RuntimeError(
                "No activations/gradients captured. "
                "Use with gradcam.capture(): before forward pass."
            )

        # Compute importance weights (global average of gradients)
        # Gradients shape: (B, seq, D) or similar
        weights = self._gradients.mean(dim=-1, keepdim=True)  # (B, seq, 1)

        # Weight activations
        # Activations shape: (B, seq, D) or similar
        # Handle different activation shapes
        if self._activations.dim() == 3:
            activations = self._activations
        else:
            # Reshape if needed
            activations = self._activations

        # Weighted combination
        cam = (weights * activations).sum(dim=-1)  # (B, seq)

        # Remove CLS token (position 0) if present
        if cam.shape[1] == num_patches + 1:
            cam = cam[:, 1:]  # (B, num_patches)

        # ReLU to keep only positive contributions
        cam = F.relu(cam)

        # Reshape to 2D spatial grid
        patches_per_side = int(math.sqrt(num_patches))
        cam_2d = cam.view(cam.shape[0], patches_per_side, patches_per_side)

        # Upsample to image size
        cam_upsampled = F.interpolate(
            cam_2d.unsqueeze(1),
            size=(image_size, image_size),
            mode=INTERPOLATION_MODE,
            align_corners=False,
        ).squeeze(1)

        # Normalize per sample
        batch_size = cam_upsampled.shape[0]
        flat = cam_upsampled.view(batch_size, -1)
        min_val = flat.min(dim=1, keepdim=True).values.view(batch_size, 1, 1)
        max_val = flat.max(dim=1, keepdim=True).values.view(batch_size, 1, 1)
        cam_upsampled = (cam_upsampled - min_val) / (max_val - min_val + EPSILON)

        return cam_upsampled

    def clear(self) -> None:
        """Clear stored activations and gradients."""
        self._activations = None
        self._gradients = None


class GradCAMContext:
    """Context manager for Grad-CAM activation capture."""

    def __init__(self, gradcam: GradCAM) -> None:
        self.gradcam = gradcam

    def __enter__(self) -> GradCAM:
        self.gradcam._register_hooks()
        return self.gradcam

    def __exit__(self, *args: Any) -> None:
        self.gradcam._remove_hooks()


def compute_gradcam(
    model: nn.Module,
    images: Tensor,
    num_patches: int,
    image_size: int = DEFAULT_IMAGE_SIZE,
    target_layer: str | None = None,
) -> Tensor:
    """Convenience function to compute Grad-CAM in one call.

    This requires gradients, so the model must not be in no_grad mode.

    Args:
        model: The vision transformer model.
        images: Input images tensor (B, C, H, W).
        num_patches: Number of image patches.
        image_size: Original image size for upsampling.
        target_layer: Layer to get activations from.

    Returns:
        Grad-CAM heatmap of shape (B, H, W).

    Note:
        This function enables gradients temporarily. For batch processing
        or more control, use the GradCAM class directly.
    """
    was_training = model.training
    model.eval()

    gradcam = GradCAM(model, target_layer=target_layer)

    # Need gradients for Grad-CAM
    images = images.requires_grad_(True)

    with gradcam.capture():
        output = model(pixel_values=images, output_attentions=False)

        # Use CLS token as target
        if hasattr(output, "last_hidden_state"):
            cls_token = output.last_hidden_state[:, 0, :]
        elif hasattr(output, "pooler_output"):
            cls_token = output.pooler_output
        else:
            raise ValueError("Could not find CLS token in model output")

        # Backward pass (sum of CLS dimensions)
        cls_token.sum().backward()

    heatmap = gradcam.compute(cls_token, num_patches, image_size)

    if was_training:
        model.train()

    return heatmap.detach()

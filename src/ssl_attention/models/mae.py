"""MAE (Masked Autoencoder) model wrapper.

MAE is trained to reconstruct masked patches from visible patches.
For attention analysis, we use mask_ratio=0.0 (no masking) so all
patches are visible and we get full attention patterns.

Sequence structure: [CLS] + [196 patches]
- Patch size: 16 (so 224/16 = 14 patches per side = 196 total)
- No register tokens
"""

from typing import Any

import torch
from torch import Tensor, nn
from transformers import ViTMAEConfig, ViTMAEModel

from ssl_attention.models.base import BaseVisionModel
from ssl_attention.models.protocols import ModelOutput


class MAE(BaseVisionModel):
    """MAE (Masked Autoencoder) wrapper.

    Uses facebook/vit-mae-base which has:
    - 12 transformer layers
    - 12 attention heads
    - 768 embedding dimension
    - 16x16 patches (196 patches for 224x224 images)
    - No register tokens

    Note: We run with mask_ratio=0.0 to get attention over all patches.
    During training MAE masks 75% of patches, but for analysis we need all.

    Example:
        >>> model = MAE()
        >>> images = [Image.open("church.jpg")]
        >>> inputs = model.preprocess(images)
        >>> output = model(inputs)
        >>> print(output.patch_tokens.shape)  # (1, 196, 768)
    """

    model_name = "mae"
    model_id = "facebook/vit-mae-base"
    patch_size = 16
    embed_dim = 768
    num_layers = 12
    num_heads = 12
    num_registers = 0  # MAE has no registers

    def _load_config(self) -> ViTMAEConfig:
        """Load MAE config with attention output enabled and no masking.

        Sets mask_ratio=0.0 so all patches are visible during inference.
        """
        config = ViTMAEConfig.from_pretrained(self.model_id)
        config.output_attentions = True
        config.mask_ratio = 0.0  # No masking for attention analysis
        return config

    def _load_model(self) -> nn.Module:
        """Load MAE from HuggingFace with attention output enabled."""
        config = self._load_config()
        return ViTMAEModel.from_pretrained(self.model_id, config=config)

    # Note: forward() is inherited from base class.
    # mask_ratio=0.0 is set in config, so no masking occurs.

    def _extract_output(self, model_output: Any) -> ModelOutput:
        """Extract standardized output from MAE output.

        MAE returns:
        - last_hidden_state: (B, seq_len, D) where seq_len = 1 + 196 = 197
        - attentions: tuple of L tensors, each (B, H, seq, seq)

        We extract:
        - CLS token: position 0
        - Patch tokens: positions 1 onwards (no registers)
        """
        hidden_states = model_output.last_hidden_state  # (B, 197, 768)
        attentions = model_output.attentions  # tuple of 12 tensors

        # CLS is position 0
        cls_token = hidden_states[:, 0, :]  # (B, 768)

        # Patches start at position 1 (no registers)
        patch_tokens = hidden_states[:, 1:, :]  # (B, 196, 768)

        # Convert attention tuple to list
        attention_weights = list(attentions)

        return ModelOutput(
            cls_token=cls_token,
            patch_tokens=patch_tokens,
            attention_weights=attention_weights,
        )


def create_mae(
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> MAE:
    """Create a MAE model instance.

    Args:
        device: Target device. Auto-detects if None.
        dtype: Tensor dtype. Uses optimal for device if None.

    Returns:
        Configured MAE model.
    """
    return MAE(device=device, dtype=dtype)

"""DINOv3 model wrapper with RoPE and registers.

DINOv3 (also known as DINOv2 v2 or the "LVD" variant) uses:
- Rotary Position Embeddings (RoPE) instead of learned position embeddings
- Register tokens like DINOv2
- Larger training data (LVD-1689M dataset)

Sequence structure: [CLS] + [4 registers] + [196 patches]
- Patch size: 16 (so 224/16 = 14 patches per side = 196 total)
- Register tokens: 4

Note: Requires transformers>=4.56.0 for DINOv3 support.
"""

from typing import Any

import torch
from torch import nn
from transformers import AutoModel

from ssl_attention.models.base import BaseVisionModel
from ssl_attention.models.protocols import ModelOutput

# Suppress trust_remote_code warning
import warnings
warnings.filterwarnings("ignore", message=".*trust_remote_code.*")


class DINOv3(BaseVisionModel):
    """DINOv3 with RoPE and registers wrapper.

    Uses facebook/dinov3-vitb16-pretrain-lvd1689m which has:
    - 12 transformer layers
    - 12 attention heads
    - 768 embedding dimension
    - 16x16 patches (196 patches for 224x224 images)
    - 4 register tokens
    - Rotary Position Embeddings (RoPE)

    Example:
        >>> model = DINOv3()
        >>> images = [Image.open("church.jpg")]
        >>> inputs = model.preprocess(images)
        >>> output = model(inputs)
        >>> print(output.patch_tokens.shape)  # (1, 196, 768)
    """

    model_name = "dinov3"
    model_id = "facebook/dinov3-vitb16-pretrain-lvd1689m"
    patch_size = 16
    embed_dim = 768
    num_layers = 12
    num_heads = 12
    num_registers = 4

    def _load_model(self) -> nn.Module:
        """Load DINOv3 from HuggingFace.

        Uses AutoModel with config that enables attention output.
        """
        config = self._load_config()
        return AutoModel.from_pretrained(self.model_id, config=config, trust_remote_code=True)

    def _extract_output(self, model_output: Any) -> ModelOutput:
        """Extract standardized output from DINOv3 output.

        DINOv3 returns:
        - last_hidden_state: (B, seq_len, D) where seq_len = 1 + 4 + 196 = 201
        - attentions: tuple of L tensors, each (B, H, seq, seq)

        We extract:
        - CLS token: position 0
        - Patch tokens: positions 5 onwards (skip CLS + 4 registers)
        """
        hidden_states = model_output.last_hidden_state  # (B, 201, 768)
        attentions = model_output.attentions  # tuple of 12 tensors

        # CLS is always position 0
        cls_token = hidden_states[:, 0, :]  # (B, 768)

        # Patches start after CLS + registers
        patch_start = 1 + self.num_registers  # 5
        patch_tokens = hidden_states[:, patch_start:, :]  # (B, 196, 768)

        # Convert attention tuple to list
        attention_weights = list(attentions)

        return ModelOutput(
            cls_token=cls_token,
            patch_tokens=patch_tokens,
            attention_weights=attention_weights,
        )


def create_dinov3(
    device: torch.device | None = None,
    dtype: torch.dtype | None = None,
) -> DINOv3:
    """Create a DINOv3 model instance.

    Args:
        device: Target device. Auto-detects if None.
        dtype: Tensor dtype. Uses optimal for device if None.

    Returns:
        Configured DINOv3 model.
    """
    return DINOv3(device=device, dtype=dtype)

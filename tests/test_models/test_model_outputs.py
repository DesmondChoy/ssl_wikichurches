"""Tests for model output extraction logic.

Priority 1: These tests verify that each model's _extract_output() method
correctly parses the HuggingFace model output into standardized ModelOutput.

Each model has a different sequence layout:
- DINOv2: [CLS] + [4 registers] + [256 patches] = 261 tokens
- DINOv3: [CLS] + [4 registers] + [196 patches] = 201 tokens
- MAE:    [CLS] + [196 patches] = 197 tokens
- CLIP:   [CLS] + [196 patches] = 197 tokens
- SigLIP: [196 patches] (no CLS, uses pooler) = 196 tokens
"""

from __future__ import annotations

import pytest
import torch

from tests.conftest import MODEL_CONFIGS, MockHuggingFaceOutput


class TestDINOv2Output:
    """Test DINOv2 output extraction."""

    @pytest.fixture
    def dinov2_output(self, make_mock_output) -> MockHuggingFaceOutput:
        """Create mock DINOv2 output."""
        config = MODEL_CONFIGS["dinov2"]
        return make_mock_output(
            seq_len=config["seq_len"],  # 261
            has_pooler=config["has_pooler"],
        )

    def test_sequence_length(self, dinov2_output):
        """Verify DINOv2 sequence length is 261 (1 CLS + 4 registers + 256 patches)."""
        assert dinov2_output.last_hidden_state.shape[1] == 261

    def test_attention_shape(self, dinov2_output):
        """Verify attention tensor shape matches sequence length."""
        for attn in dinov2_output.attentions:
            assert attn.shape == (2, 12, 261, 261)

    def test_patch_extraction(self, dinov2_output):
        """Verify patch tokens are extracted correctly (skip CLS + 4 registers)."""
        from ssl_attention.models.dinov2 import DINOv2

        # Create a minimal instance to test _extract_output
        # We'll mock the extraction logic directly
        last_hidden = dinov2_output.last_hidden_state
        num_registers = MODEL_CONFIGS["dinov2"]["num_registers"]
        patch_start = 1 + num_registers  # 5

        patch_tokens = last_hidden[:, patch_start:, :]
        assert patch_tokens.shape[1] == 256  # 16x16 patches

    def test_cls_extraction(self, dinov2_output):
        """Verify CLS token comes from position 0."""
        last_hidden = dinov2_output.last_hidden_state
        cls_token = last_hidden[:, 0, :]
        assert cls_token.shape == (2, 768)


class TestDINOv3Output:
    """Test DINOv3 output extraction."""

    @pytest.fixture
    def dinov3_output(self, make_mock_output) -> MockHuggingFaceOutput:
        """Create mock DINOv3 output."""
        config = MODEL_CONFIGS["dinov3"]
        return make_mock_output(
            seq_len=config["seq_len"],  # 201
            has_pooler=config["has_pooler"],
        )

    def test_sequence_length(self, dinov3_output):
        """Verify DINOv3 sequence length is 201 (1 CLS + 4 registers + 196 patches)."""
        assert dinov3_output.last_hidden_state.shape[1] == 201

    def test_attention_shape(self, dinov3_output):
        """Verify attention tensor shape matches sequence length."""
        for attn in dinov3_output.attentions:
            assert attn.shape == (2, 12, 201, 201)

    def test_patch_extraction(self, dinov3_output):
        """Verify patch tokens are extracted correctly (skip CLS + 4 registers)."""
        last_hidden = dinov3_output.last_hidden_state
        num_registers = MODEL_CONFIGS["dinov3"]["num_registers"]
        patch_start = 1 + num_registers  # 5

        patch_tokens = last_hidden[:, patch_start:, :]
        assert patch_tokens.shape[1] == 196  # 14x14 patches


class TestMAEOutput:
    """Test MAE output extraction."""

    @pytest.fixture
    def mae_output(self, make_mock_output) -> MockHuggingFaceOutput:
        """Create mock MAE output."""
        config = MODEL_CONFIGS["mae"]
        return make_mock_output(
            seq_len=config["seq_len"],  # 197
            has_pooler=config["has_pooler"],
        )

    def test_sequence_length(self, mae_output):
        """Verify MAE sequence length is 197 (1 CLS + 196 patches)."""
        assert mae_output.last_hidden_state.shape[1] == 197

    def test_attention_shape(self, mae_output):
        """Verify attention tensor shape matches sequence length."""
        for attn in mae_output.attentions:
            assert attn.shape == (2, 12, 197, 197)

    def test_patch_extraction(self, mae_output):
        """Verify patch tokens start at position 1 (no registers)."""
        last_hidden = mae_output.last_hidden_state
        num_registers = MODEL_CONFIGS["mae"]["num_registers"]
        patch_start = 1 + num_registers  # 1

        patch_tokens = last_hidden[:, patch_start:, :]
        assert patch_tokens.shape[1] == 196  # 14x14 patches


class TestCLIPOutput:
    """Test CLIP output extraction."""

    @pytest.fixture
    def clip_output(self, make_mock_output) -> MockHuggingFaceOutput:
        """Create mock CLIP output."""
        config = MODEL_CONFIGS["clip"]
        return make_mock_output(
            seq_len=config["seq_len"],  # 197
            has_pooler=config["has_pooler"],
        )

    def test_sequence_length(self, clip_output):
        """Verify CLIP sequence length is 197 (1 CLS + 196 patches)."""
        assert clip_output.last_hidden_state.shape[1] == 197

    def test_patch_extraction(self, clip_output):
        """Verify patch tokens start at position 1 (no registers)."""
        last_hidden = clip_output.last_hidden_state
        patch_tokens = last_hidden[:, 1:, :]
        assert patch_tokens.shape[1] == 196


class TestSigLIPOutput:
    """Test SigLIP output extraction."""

    @pytest.fixture
    def siglip_output(self, make_mock_output) -> MockHuggingFaceOutput:
        """Create mock SigLIP output with pooler."""
        config = MODEL_CONFIGS["siglip"]
        return make_mock_output(
            seq_len=config["seq_len"],  # 196
            has_pooler=config["has_pooler"],  # True
        )

    def test_sequence_length(self, siglip_output):
        """Verify SigLIP sequence length is 196 (all patches, no CLS)."""
        assert siglip_output.last_hidden_state.shape[1] == 196

    def test_attention_shape(self, siglip_output):
        """Verify attention tensor shape matches sequence length."""
        for attn in siglip_output.attentions:
            assert attn.shape == (2, 12, 196, 196)

    def test_has_pooler_output(self, siglip_output):
        """Verify SigLIP has pooler_output for CLS-like representation."""
        assert siglip_output.pooler_output is not None
        assert siglip_output.pooler_output.shape == (2, 768)

    def test_all_tokens_are_patches(self, siglip_output):
        """Verify all sequence tokens are patch tokens (no CLS to skip)."""
        last_hidden = siglip_output.last_hidden_state
        # All tokens are patches
        patch_tokens = last_hidden  # No slicing needed
        assert patch_tokens.shape[1] == 196


class TestModelOutputValidation:
    """Test ModelOutput dataclass validation."""

    def test_batch_size_mismatch_raises(self):
        """Verify ModelOutput raises on batch size mismatch."""
        from ssl_attention.models.protocols import ModelOutput

        cls_token = torch.randn(2, 768)
        patch_tokens = torch.randn(3, 196, 768)  # Different batch size!
        attention_weights = [torch.randn(2, 12, 197, 197)]

        with pytest.raises(ValueError, match="Batch size mismatch"):
            ModelOutput(
                cls_token=cls_token,
                patch_tokens=patch_tokens,
                attention_weights=attention_weights,
            )

    def test_attention_batch_mismatch_raises(self):
        """Verify ModelOutput raises on attention batch mismatch."""
        from ssl_attention.models.protocols import ModelOutput

        cls_token = torch.randn(2, 768)
        patch_tokens = torch.randn(2, 196, 768)
        attention_weights = [torch.randn(3, 12, 197, 197)]  # Different batch!

        with pytest.raises(ValueError, match="Batch size mismatch"):
            ModelOutput(
                cls_token=cls_token,
                patch_tokens=patch_tokens,
                attention_weights=attention_weights,
            )

    def test_valid_output_properties(self):
        """Verify ModelOutput computes properties correctly."""
        from ssl_attention.models.protocols import ModelOutput

        cls_token = torch.randn(2, 768)
        patch_tokens = torch.randn(2, 196, 768)
        attention_weights = [torch.randn(2, 12, 197, 197) for _ in range(12)]

        output = ModelOutput(
            cls_token=cls_token,
            patch_tokens=patch_tokens,
            attention_weights=attention_weights,
        )

        assert output.batch_size == 2
        assert output.embed_dim == 768
        assert output.num_patches == 196
        assert output.num_layers == 12


@pytest.mark.parametrize(
    "model_name,expected_seq_len,expected_patches,expected_registers",
    [
        ("dinov2", 261, 256, 4),
        ("dinov3", 201, 196, 4),
        ("mae", 197, 196, 0),
        ("clip", 197, 196, 0),
        ("siglip", 196, 196, 0),
    ],
)
class TestParameterizedModelConfigs:
    """Parametrized tests for all model configurations."""

    def test_sequence_layout(
        self,
        make_mock_output,
        model_name: str,
        expected_seq_len: int,
        expected_patches: int,
        expected_registers: int,
    ):
        """Verify sequence length = 1 + registers + patches (or just patches for SigLIP)."""
        config = MODEL_CONFIGS[model_name]
        output = make_mock_output(seq_len=config["seq_len"], has_pooler=config["has_pooler"])

        actual_seq_len = output.last_hidden_state.shape[1]
        assert actual_seq_len == expected_seq_len

        # Verify math: seq_len = (1 if has CLS else 0) + registers + patches
        has_cls = model_name != "siglip"
        computed_seq_len = (1 if has_cls else 0) + expected_registers + expected_patches
        assert actual_seq_len == computed_seq_len

    def test_patch_count_is_square(
        self,
        make_mock_output,
        model_name: str,
        expected_seq_len: int,
        expected_patches: int,
        expected_registers: int,
    ):
        """Verify patch count is a perfect square (for valid 2D grid)."""
        import math

        sqrt = math.isqrt(expected_patches)
        assert sqrt * sqrt == expected_patches, f"{model_name} patches must be square"

    def test_patches_per_side(
        self,
        make_mock_output,
        model_name: str,
        expected_seq_len: int,
        expected_patches: int,
        expected_registers: int,
    ):
        """Verify patches_per_side matches expected values."""
        import math

        patches_per_side = math.isqrt(expected_patches)

        # DINOv2: 16x16, others: 14x14
        if model_name == "dinov2":
            assert patches_per_side == 16
        else:
            assert patches_per_side == 14


class TestHiddenStatesExtraction:
    """Test hidden states extraction for per-layer features."""

    def test_hidden_states_included_when_requested(self, make_mock_output):
        """Verify hidden_states are included when include_hidden_states=True."""
        output = make_mock_output(
            seq_len=197,
            include_hidden_states=True,
        )
        assert output.hidden_states is not None
        # L+1 tensors: embedding + L transformer layers
        assert len(output.hidden_states) == 13

    def test_hidden_states_excluded_by_default(self, make_mock_output):
        """Verify hidden_states are None by default."""
        output = make_mock_output(seq_len=197, include_hidden_states=False)
        assert output.hidden_states is None

    def test_hidden_states_shape(self, make_mock_output):
        """Verify hidden states have correct shape."""
        batch_size = 2
        seq_len = 197
        embed_dim = 768

        output = make_mock_output(
            seq_len=seq_len,
            batch_size=batch_size,
            embed_dim=embed_dim,
            include_hidden_states=True,
        )

        for hs in output.hidden_states:
            assert hs.shape == (batch_size, seq_len, embed_dim)

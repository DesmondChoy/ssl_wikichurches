"""Fine-tuning module for SSL models on WikiChurches style classification.

This module enables fine-tuning all 5 SSL models (DINOv2, DINOv3, MAE, CLIP, SigLIP)
on the 4-class architectural style classification task. After fine-tuning, attention
patterns can be compared before/after to measure alignment shift.

Key classes:
- FineTuningConfig: Hyperparameters and settings
- FineTuningResult: Training metrics and checkpoint path
- ClassificationHead: Linear classifier on CLS token
- FineTunableModel: Wraps SSL backbone + classification head
- FineTuner: Training orchestrator

Example:
    >>> from ssl_attention.evaluation.fine_tuning import (
    ...     FineTuningConfig, FineTuner, FineTunableModel
    ... )
    >>> from ssl_attention.data import FullDataset
    >>> from ssl_attention.config import DATASET_PATH
    >>>
    >>> config = FineTuningConfig(model_name="dinov2", num_epochs=10)
    >>> model = FineTunableModel(config.model_name, freeze_backbone=False)
    >>> dataset = FullDataset(DATASET_PATH, filter_labeled=True)
    >>>
    >>> tuner = FineTuner(config)
    >>> result = tuner.train(model, dataset)
    >>> print(f"Best validation accuracy: {result.best_val_acc:.1%}")
"""

from __future__ import annotations

import json
import random
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Any

import numpy as np
import torch
from torch import Tensor, nn
from torch.optim import AdamW
from torch.utils.data import DataLoader, Subset
from torchvision import transforms as T
from transformers import AutoImageProcessor, get_cosine_schedule_with_warmup

from ssl_attention.config import (
    MODELS,
    NUM_STYLES,
)
from ssl_attention.models.protocols import ModelOutput
from ssl_attention.utils.device import clear_memory, get_device

if TYPE_CHECKING:
    from ssl_attention.data.wikichurches import FullDataset


# =============================================================================
# Output Directories
# =============================================================================

_PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
CHECKPOINTS_PATH = _PROJECT_ROOT / "outputs" / "checkpoints"
RESULTS_PATH = _PROJECT_ROOT / "outputs" / "results"

# LoRA target modules per model (HF attention projection names differ)
LORA_TARGET_MODULES: dict[str, list[str]] = {
    "dinov2": ["query", "value"],
    "dinov3": ["q_proj", "v_proj"],
    "mae": ["query", "value"],
    "clip": ["q_proj", "v_proj"],
    "siglip": ["q_proj", "v_proj"],
}


# =============================================================================
# Configuration
# =============================================================================


@dataclass
class FineTuningConfig:
    """Configuration for fine-tuning an SSL model.

    Attributes:
        model_name: Name of the SSL model to fine-tune (dinov2, dinov3, mae, clip, siglip).
        num_epochs: Number of training epochs.
        batch_size: Training batch size.
        learning_rate_backbone: Learning rate for backbone parameters.
        learning_rate_head: Learning rate for classification head.
        weight_decay: Weight decay for AdamW optimizer.
        freeze_backbone: If True, only train classification head (linear probe).
        val_split: Fraction of data to use for validation.
        seed: Random seed for reproducibility.
        warmup_ratio: Fraction of total training steps for linear LR warmup.
        max_grad_norm: Maximum gradient norm for clipping (0 disables clipping).
        use_augmentation: Whether to apply training data augmentations.
        use_lora: If True, apply LoRA adapters to backbone attention layers.
        lora_rank: Rank of LoRA decomposition matrices.
        lora_alpha: Scaling factor for LoRA (effective scale = alpha / rank).
        lora_dropout: Dropout probability for LoRA layers.
        lora_target_modules: Attention modules to adapt. Auto-resolved per model if None.
    """

    model_name: str
    num_epochs: int = 10
    batch_size: int = 16
    learning_rate_backbone: float = 1e-5
    learning_rate_head: float = 1e-3
    weight_decay: float = 0.01
    freeze_backbone: bool = False
    val_split: float = 0.2
    seed: int = 42
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    use_augmentation: bool = True
    use_lora: bool = False
    lora_rank: int = 8
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: list[str] | None = field(default=None)

    def __post_init__(self) -> None:
        if self.use_lora and self.freeze_backbone:
            raise ValueError(
                "use_lora=True and freeze_backbone=True are conflicting strategies. "
                "LoRA fine-tunes backbone adapters; freeze_backbone disables all backbone training."
            )
        # Auto-adjust backbone LR for LoRA when user hasn't changed the default
        if self.use_lora and self.learning_rate_backbone == 1e-5:
            self.learning_rate_backbone = 1e-4


@dataclass
class FineTuningResult:
    """Result of fine-tuning an SSL model.

    Attributes:
        model_name: Name of the fine-tuned model.
        best_val_acc: Best validation accuracy achieved.
        best_epoch: Epoch at which best validation accuracy occurred.
        train_history: List of dicts with per-epoch metrics.
        checkpoint_path: Path to saved checkpoint.
        config: The configuration used for training.
    """

    model_name: str
    best_val_acc: float
    best_epoch: int
    train_history: list[dict[str, float]]
    checkpoint_path: Path
    config: dict[str, Any] = field(default_factory=dict)


# =============================================================================
# Model Components
# =============================================================================


class ClassificationHead(nn.Module):
    """Linear classification head for SSL features.

    Takes CLS token (or pooled features) and produces class logits.

    Args:
        embed_dim: Dimension of input features (typically 768).
        num_classes: Number of output classes.
        dropout: Dropout probability before classifier.
    """

    def __init__(
        self,
        embed_dim: int = 768,
        num_classes: int = NUM_STYLES,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()
        self.embed_dim = embed_dim
        self.num_classes = num_classes

        self.head = nn.Sequential(
            nn.Dropout(dropout) if dropout > 0 else nn.Identity(),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, features: Tensor) -> Tensor:
        """Forward pass through classification head.

        Args:
            features: CLS token features of shape (B, embed_dim).

        Returns:
            Logits of shape (B, num_classes).
        """
        result: Tensor = self.head(features)
        return result


class FineTunableModel(nn.Module):
    """Wraps an SSL backbone with a classification head for fine-tuning.

    Supports all 5 SSL models with unified interface:
    - DINOv2: Uses CLS token from dinov2-with-registers
    - DINOv3: Uses CLS token from dinov3 with RoPE
    - MAE: Uses CLS token (mask_ratio=0 for full image)
    - CLIP: Uses CLS token from vision encoder
    - SigLIP: Uses pooler_output (no CLS token in sequence)

    Args:
        model_name: Name of the SSL backbone.
        num_classes: Number of classification classes.
        freeze_backbone: If True, freeze backbone weights.
        device: Target device. Auto-detects if None.
        dtype: Tensor dtype. Uses float32 for MPS compatibility.
    """

    def __init__(
        self,
        model_name: str,
        num_classes: int = NUM_STYLES,
        freeze_backbone: bool = False,
        device: torch.device | None = None,
        dtype: torch.dtype | None = None,
        use_lora: bool = False,
        lora_rank: int = 8,
        lora_alpha: int = 32,
        lora_dropout: float = 0.1,
        lora_target_modules: list[str] | None = None,
    ) -> None:
        super().__init__()

        self.model_name = model_name
        self.num_classes = num_classes
        self.freeze_backbone = freeze_backbone
        self.device = device or get_device()
        # MPS compatibility: use float32
        self.dtype = dtype or torch.float32
        self.use_lora = use_lora
        self.lora_rank = lora_rank
        self.lora_alpha = lora_alpha
        self.lora_dropout = lora_dropout
        self.lora_target_modules = lora_target_modules

        # Load model configuration
        if model_name not in MODELS:
            raise ValueError(f"Unknown model: {model_name}. Available: {list(MODELS.keys())}")
        self._config = MODELS[model_name]

        # Load backbone and processor
        self.backbone = self._load_backbone()
        self.processor = AutoImageProcessor.from_pretrained(self._config.model_id)

        # Apply LoRA adapters before moving to device
        if use_lora:
            self._apply_lora()

        # Create classification head
        self.classifier = ClassificationHead(
            embed_dim=self._config.embed_dim,
            num_classes=num_classes,
        )

        # Move to device
        self.backbone = self.backbone.to(device=self.device, dtype=self.dtype)
        self.classifier = self.classifier.to(device=self.device, dtype=self.dtype)

        # Freeze backbone if requested
        if freeze_backbone:
            self._freeze_backbone()

    def _load_backbone(self) -> nn.Module:
        """Load the SSL backbone model.

        Returns:
            The backbone model with output_attentions enabled.
        """
        from transformers import (
            AutoConfig,
            AutoModel,
            CLIPVisionConfig,
            CLIPVisionModel,
            Siglip2VisionConfig,
            Siglip2VisionModel,
            ViTMAEConfig,
            ViTMAEModel,
        )

        model_id = self._config.model_id

        if self.model_name == "mae":
            mae_config = ViTMAEConfig.from_pretrained(model_id)
            mae_config.output_attentions = True
            mae_config.mask_ratio = 0.0  # No masking for fine-tuning
            return ViTMAEModel.from_pretrained(model_id, config=mae_config)

        elif self.model_name == "clip":
            clip_config = CLIPVisionConfig.from_pretrained(model_id)
            clip_config.output_attentions = True
            return CLIPVisionModel.from_pretrained(model_id, config=clip_config)

        elif self.model_name == "siglip":
            siglip_config = Siglip2VisionConfig.from_pretrained(model_id)
            siglip_config.output_attentions = True
            return Siglip2VisionModel.from_pretrained(model_id, config=siglip_config)

        else:  # dinov2, dinov3
            auto_config = AutoConfig.from_pretrained(model_id)
            auto_config.output_attentions = True
            model: nn.Module = AutoModel.from_pretrained(
                model_id, config=auto_config, trust_remote_code=True
            )
            return model

    def _freeze_backbone(self) -> None:
        """Freeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = False

    def _unfreeze_backbone(self) -> None:
        """Unfreeze all backbone parameters."""
        for param in self.backbone.parameters():
            param.requires_grad = True

    def _apply_lora(self) -> None:
        """Apply LoRA adapters to backbone attention layers.

        Uses HuggingFace PEFT to wrap the backbone with low-rank adapters.
        PEFT auto-freezes non-LoRA backbone params, so get_optimizer_param_groups()
        (which filters by requires_grad) picks up only LoRA + head params.
        """
        try:
            from peft import LoraConfig, get_peft_model
        except ImportError:
            raise ImportError(
                "LoRA requires the 'peft' package. Install it with: pip install peft>=0.7.0"
            ) from None

        # Resolve target modules
        target_modules = self.lora_target_modules
        if target_modules is None:
            if self.model_name not in LORA_TARGET_MODULES:
                raise ValueError(
                    f"No default LoRA target modules for '{self.model_name}'. "
                    f"Available: {list(LORA_TARGET_MODULES.keys())}. "
                    f"Provide lora_target_modules explicitly."
                )
            target_modules = LORA_TARGET_MODULES[self.model_name]

        lora_config = LoraConfig(
            r=self.lora_rank,
            lora_alpha=self.lora_alpha,
            lora_dropout=self.lora_dropout,
            target_modules=target_modules,
            bias="none",
        )

        self.backbone = get_peft_model(self.backbone, lora_config)  # type: ignore[arg-type]

        # Log trainable parameter count
        trainable = sum(p.numel() for p in self.backbone.parameters() if p.requires_grad)
        total = sum(p.numel() for p in self.backbone.parameters())
        print(
            f"LoRA applied: {trainable:,} trainable params / {total:,} total "
            f"({trainable / total:.2%})"
        )

    def _extract_features(self, model_output: Any) -> tuple[Tensor, list[Tensor]]:
        """Extract CLS features and attention weights from backbone output.

        Args:
            model_output: Raw output from backbone forward pass.

        Returns:
            Tuple of (cls_features, attention_weights).
        """
        attentions = list(model_output.attentions)

        if self.model_name == "siglip":
            # SigLIP uses pooler_output (no CLS in sequence)
            cls_features = model_output.pooler_output
        else:
            # DINOv2, DINOv3, MAE, CLIP all have CLS at position 0
            cls_features = model_output.last_hidden_state[:, 0, :]

        return cls_features, attentions

    def preprocess(self, images: list) -> Tensor:
        """Preprocess PIL images for model input.

        Args:
            images: List of PIL Images.

        Returns:
            Tensor of shape (B, C, H, W) on the model's device.
        """
        processed = self.processor(images=images, return_tensors="pt")
        pixel_values: Tensor = processed["pixel_values"]
        return pixel_values.to(device=self.device, dtype=self.dtype)

    def forward(
        self, pixel_values: Tensor
    ) -> tuple[Tensor, Tensor, list[Tensor]]:
        """Forward pass through backbone and classifier.

        Args:
            pixel_values: Preprocessed images of shape (B, C, H, W).

        Returns:
            Tuple of (logits, cls_features, attention_weights).
            - logits: Classification logits of shape (B, num_classes).
            - cls_features: CLS token features of shape (B, embed_dim).
            - attention_weights: List of attention tensors per layer.
        """
        # Forward through backbone
        model_output = self.backbone(
            pixel_values=pixel_values,
            output_attentions=True,
        )

        # Extract features and attention
        cls_features, attention_weights = self._extract_features(model_output)

        # Classify
        logits = self.classifier(cls_features)

        return logits, cls_features, attention_weights

    def extract_attention(self, pixel_values: Tensor) -> ModelOutput:
        """Extract attention in the same format as base models.

        This allows fine-tuned models to be used with existing
        attention analysis metrics (IoU, pointing accuracy).

        Args:
            pixel_values: Preprocessed images of shape (B, C, H, W).

        Returns:
            ModelOutput compatible with attention analysis.
        """
        with torch.no_grad():
            model_output = self.backbone(
                pixel_values=pixel_values,
                output_attentions=True,
            )

        hidden_states = model_output.last_hidden_state
        attentions = list(model_output.attentions)

        if self.model_name == "siglip":
            cls_token = model_output.pooler_output
            patch_tokens = hidden_states  # All positions are patches
        else:
            cls_token = hidden_states[:, 0, :]
            # Skip CLS + registers for patch tokens
            patch_start = 1 + self._config.num_registers
            patch_tokens = hidden_states[:, patch_start:, :]

        return ModelOutput(
            cls_token=cls_token,
            patch_tokens=patch_tokens,
            attention_weights=attentions,
        )

    def get_optimizer_param_groups(
        self,
        lr_backbone: float,
        lr_head: float,
        weight_decay: float = 0.01,
    ) -> list[dict[str, Any]]:
        """Get parameter groups with differential learning rates.

        Args:
            lr_backbone: Learning rate for backbone (smaller to preserve features).
            lr_head: Learning rate for classification head (larger for fast learning).
            weight_decay: Weight decay coefficient.

        Returns:
            List of parameter group dicts for optimizer.
        """
        param_groups = []

        # Backbone parameters (if not frozen)
        if not self.freeze_backbone:
            backbone_params = [p for p in self.backbone.parameters() if p.requires_grad]
            if backbone_params:
                param_groups.append({
                    "params": backbone_params,
                    "lr": lr_backbone,
                    "weight_decay": weight_decay,
                })

        # Head parameters
        head_params = list(self.classifier.parameters())
        param_groups.append({
            "params": head_params,
            "lr": lr_head,
            "weight_decay": weight_decay,
        })

        return param_groups


# =============================================================================
# Training
# =============================================================================


class FineTuner:
    """Training orchestrator for fine-tuning SSL models.

    Handles the full training loop including:
    - Train/validation split (stratified)
    - Class-weighted loss for imbalanced data
    - Cosine LR schedule with linear warmup
    - Gradient clipping for training stability
    - Data augmentation (crop, flip, color jitter)
    - Checkpoint saving (model, optimizer, scheduler)
    - Training history logging

    Args:
        config: Fine-tuning configuration.

    Example:
        >>> config = FineTuningConfig(model_name="dinov2", num_epochs=10)
        >>> tuner = FineTuner(config)
        >>> model = FineTunableModel("dinov2")
        >>> result = tuner.train(model, dataset)
    """

    def __init__(self, config: FineTuningConfig) -> None:
        self.config = config
        self.device = get_device()

        # Set seeds for reproducibility
        self._set_seed(config.seed)

    def _set_seed(self, seed: int) -> None:
        """Set random seeds for reproducibility."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def _compute_class_weights(self, labels: list[int]) -> Tensor:
        """Compute inverse frequency class weights for imbalanced data.

        Args:
            labels: List of class labels.

        Returns:
            Tensor of class weights normalized to sum to num_classes.
        """
        label_counts = np.bincount(labels, minlength=NUM_STYLES)
        # Inverse frequency weighting
        weights = 1.0 / (label_counts + 1e-6)
        # Normalize to sum to num_classes
        weights = weights / weights.sum() * NUM_STYLES
        return torch.tensor(weights, dtype=torch.float32, device=self.device)

    def _stratified_split(
        self,
        dataset: FullDataset,
        val_split: float,
    ) -> tuple[Subset, Subset]:
        """Split dataset into train/val with stratification.

        Args:
            dataset: Full dataset with style_label in samples.
            val_split: Fraction for validation.

        Returns:
            Tuple of (train_subset, val_subset).
        """
        # Get all labels
        all_labels = []
        for i in range(len(dataset)):
            sample = dataset[i]
            label = sample.get("style_label")
            if label is not None:
                all_labels.append((i, label))

        # Group by label
        label_to_indices: dict[int, list[int]] = {}
        for idx, label in all_labels:
            if label not in label_to_indices:
                label_to_indices[label] = []
            label_to_indices[label].append(idx)

        # Split each class
        train_indices = []
        val_indices = []

        for _label, indices in label_to_indices.items():
            random.shuffle(indices)
            n_val = max(1, int(len(indices) * val_split))
            val_indices.extend(indices[:n_val])
            train_indices.extend(indices[n_val:])

        return Subset(dataset, train_indices), Subset(dataset, val_indices)

    def _collate_fn(self, batch: list[dict]) -> dict[str, Any]:
        """Collate function for DataLoader.

        Filters out samples with None labels and returns images as list.
        """
        # Filter samples with valid labels
        valid_batch = [s for s in batch if s.get("style_label") is not None]
        if not valid_batch:
            # Return empty batch markers
            return {"images": [], "labels": torch.tensor([], dtype=torch.long)}

        return {
            "images": [s["image"] for s in valid_batch],
            "labels": torch.tensor(
                [s["style_label"] for s in valid_batch], dtype=torch.long
            ),
        }

    def _build_train_transform(self) -> T.Compose:
        """Build data augmentation transform for training images.

        Returns a transform that applies random crop, flip, and color jitter.
        Outputs PIL images — model.preprocess() handles ToTensor/Normalize.
        """
        return T.Compose([
            T.RandomResizedCrop(224, scale=(0.8, 1.0)),
            T.RandomHorizontalFlip(p=0.5),
            T.ColorJitter(brightness=0.2, contrast=0.2),
        ])

    @staticmethod
    def _augment_train_images(
        images: list, transform: T.Compose
    ) -> list:
        """Apply augmentation transform to a list of PIL images."""
        return [transform(img) for img in images]

    def train(
        self,
        model: FineTunableModel,
        dataset: FullDataset,
    ) -> FineTuningResult:
        """Train the model on the dataset.

        Uses cosine LR schedule with linear warmup, gradient clipping,
        and optional data augmentation for training stability.

        Args:
            model: FineTunableModel to train.
            dataset: FullDataset with labeled images.

        Returns:
            FineTuningResult with training metrics and checkpoint path.
        """
        # Create output directories
        CHECKPOINTS_PATH.mkdir(parents=True, exist_ok=True)
        RESULTS_PATH.mkdir(parents=True, exist_ok=True)

        # Split data
        train_subset, val_subset = self._stratified_split(dataset, self.config.val_split)
        print(f"Train size: {len(train_subset)}, Val size: {len(val_subset)}")

        # Build augmentation transform
        train_transform = self._build_train_transform() if self.config.use_augmentation else None

        # Create data loaders (num_workers=0 for MPS compatibility)
        train_loader = DataLoader(
            train_subset,
            batch_size=self.config.batch_size,
            shuffle=True,
            collate_fn=self._collate_fn,
            num_workers=0,
        )
        val_loader = DataLoader(
            val_subset,
            batch_size=self.config.batch_size,
            shuffle=False,
            collate_fn=self._collate_fn,
            num_workers=0,
        )

        # Compute class weights from training data
        train_labels = [
            dataset[idx]["style_label"]
            for idx in train_subset.indices
            if dataset[idx]["style_label"] is not None
        ]
        class_weights = self._compute_class_weights(train_labels)
        print(f"Class weights: {class_weights.tolist()}")

        # Loss function with class weights
        criterion = nn.CrossEntropyLoss(weight=class_weights)

        # Optimizer with differential learning rates
        param_groups = model.get_optimizer_param_groups(
            lr_backbone=self.config.learning_rate_backbone,
            lr_head=self.config.learning_rate_head,
            weight_decay=self.config.weight_decay,
        )
        optimizer = AdamW(param_groups)

        # LR scheduler: cosine decay with linear warmup
        num_training_steps = self.config.num_epochs * len(train_loader)
        num_warmup_steps = int(self.config.warmup_ratio * num_training_steps)
        scheduler = get_cosine_schedule_with_warmup(
            optimizer,
            num_warmup_steps=num_warmup_steps,
            num_training_steps=num_training_steps,
        )

        # Training loop
        train_history: list[dict[str, float]] = []
        best_val_acc = 0.0
        best_epoch = 0
        if self.config.use_lora:
            checkpoint_path = CHECKPOINTS_PATH / f"{self.config.model_name}_lora_finetuned.pt"
        else:
            checkpoint_path = CHECKPOINTS_PATH / f"{self.config.model_name}_finetuned.pt"

        for epoch in range(self.config.num_epochs):
            # Train
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0

            for batch in train_loader:
                if len(batch["images"]) == 0:
                    continue

                # Apply augmentation (training only)
                images = batch["images"]
                if train_transform is not None:
                    images = self._augment_train_images(images, train_transform)

                # Preprocess and forward
                pixel_values = model.preprocess(images)
                labels = batch["labels"].to(self.device)

                optimizer.zero_grad()
                logits, _, _ = model(pixel_values)
                loss = criterion(logits, labels)
                loss.backward()
                if self.config.max_grad_norm > 0:
                    torch.nn.utils.clip_grad_norm_(
                        model.parameters(), max_norm=self.config.max_grad_norm
                    )
                optimizer.step()
                scheduler.step()

                train_loss += loss.item() * len(labels)
                train_correct += (logits.argmax(dim=1) == labels).sum().item()
                train_total += len(labels)

                # MPS memory management
                if self.device.type == "mps":
                    torch.mps.empty_cache()

            train_loss /= max(train_total, 1)
            train_acc = train_correct / max(train_total, 1)

            # Validate
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for batch in val_loader:
                    if len(batch["images"]) == 0:
                        continue

                    pixel_values = model.preprocess(batch["images"])
                    labels = batch["labels"].to(self.device)

                    logits, _, _ = model(pixel_values)
                    loss = criterion(logits, labels)

                    val_loss += loss.item() * len(labels)
                    val_correct += (logits.argmax(dim=1) == labels).sum().item()
                    val_total += len(labels)

            val_loss /= max(val_total, 1)
            val_acc = val_correct / max(val_total, 1)

            # Log epoch
            current_lr = optimizer.param_groups[0]["lr"]
            epoch_metrics = {
                "epoch": epoch + 1,
                "train_loss": train_loss,
                "train_acc": train_acc,
                "val_loss": val_loss,
                "val_acc": val_acc,
                "lr": current_lr,
            }
            train_history.append(epoch_metrics)

            print(
                f"Epoch {epoch + 1}/{self.config.num_epochs}: "
                f"train_loss={train_loss:.4f}, train_acc={train_acc:.1%}, "
                f"val_loss={val_loss:.4f}, val_acc={val_acc:.1%}, "
                f"lr={current_lr:.2e}"
            )

            # Save best checkpoint
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                best_epoch = epoch + 1
                self._save_checkpoint(
                    model, optimizer, scheduler, epoch + 1, val_acc, checkpoint_path
                )
                print(f"  -> New best! Saved checkpoint to {checkpoint_path}")

        # Final cleanup
        clear_memory()

        return FineTuningResult(
            model_name=self.config.model_name,
            best_val_acc=best_val_acc,
            best_epoch=best_epoch,
            train_history=train_history,
            checkpoint_path=checkpoint_path,
            config=asdict(self.config),
        )

    def _save_checkpoint(
        self,
        model: FineTunableModel,
        optimizer: AdamW,
        scheduler: Any,
        epoch: int,
        val_acc: float,
        path: Path,
    ) -> None:
        """Save model checkpoint.

        Args:
            model: Model to save.
            optimizer: Optimizer state to save.
            scheduler: LR scheduler state to save.
            epoch: Current epoch number.
            val_acc: Validation accuracy at this checkpoint.
            path: Path to save checkpoint.
        """
        checkpoint = {
            "model_name": self.config.model_name,
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scheduler_state_dict": scheduler.state_dict(),
            "val_accuracy": val_acc,
            "config": asdict(self.config),
        }
        torch.save(checkpoint, path)


# =============================================================================
# Loading Fine-tuned Models
# =============================================================================


def load_finetuned_model(
    model_name: str,
    checkpoint_path: Path | None = None,
    device: torch.device | None = None,
) -> FineTunableModel:
    """Load a fine-tuned model from checkpoint.

    Args:
        model_name: Name of the model (dinov2, dinov3, mae, clip, siglip).
        checkpoint_path: Path to checkpoint. If None, uses default path.
        device: Target device. Auto-detects if None.

    Returns:
        FineTunableModel loaded with fine-tuned weights.

    Raises:
        FileNotFoundError: If checkpoint does not exist.
    """
    if checkpoint_path is None:
        checkpoint_path = CHECKPOINTS_PATH / f"{model_name}_finetuned.pt"

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    # Load checkpoint
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Reconstruct config from checkpoint
    config = checkpoint.get("config", {})
    freeze_backbone = config.get("freeze_backbone", False)
    use_lora = config.get("use_lora", False)

    # Create model (LoRA wrapper must be applied before load_state_dict)
    model = FineTunableModel(
        model_name=model_name,
        num_classes=NUM_STYLES,
        freeze_backbone=freeze_backbone,
        device=device,
        use_lora=use_lora,
        lora_rank=config.get("lora_rank", 8),
        lora_alpha=config.get("lora_alpha", 32),
        lora_dropout=config.get("lora_dropout", 0.1),
        lora_target_modules=config.get("lora_target_modules"),
    )

    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    return model


def save_training_results(
    results: list[FineTuningResult],
    output_path: Path | None = None,
) -> None:
    """Save training results to JSON.

    Args:
        results: List of FineTuningResult objects.
        output_path: Path to save JSON. If None, uses default.
    """
    if output_path is None:
        output_path = RESULTS_PATH / "fine_tuning_results.json"

    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert to serializable format
    data = []
    for result in results:
        data.append({
            "model_name": result.model_name,
            "best_val_acc": result.best_val_acc,
            "best_epoch": result.best_epoch,
            "train_history": result.train_history,
            "checkpoint_path": str(result.checkpoint_path),
            "config": result.config,
        })

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=2)

    print(f"Saved training results to {output_path}")

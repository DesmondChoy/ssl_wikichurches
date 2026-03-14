"""Tests for fine-tuning metadata-only split and label bookkeeping."""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import pytest
from PIL import Image

from ssl_attention.config import STYLE_MAPPING
from ssl_attention.data.wikichurches import FullDataset
from ssl_attention.evaluation.fine_tuning import FineTuner, FineTuningConfig


class _NoGetItemFullDataset(FullDataset):
    """Dataset variant that fails if training setup touches image bytes."""

    def __getitem__(self, idx: int):  # type: ignore[override]
        raise AssertionError(f"fine-tuning setup should not index dataset[{idx}]")


@pytest.fixture()
def dataset_root(tmp_path: Path) -> Path:
    """Create a small classification dataset with two labeled classes."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    style_qids = list(STYLE_MAPPING.keys())
    assert len(style_qids) >= 2

    churches = {
        "Q100": {"styles": [style_qids[0]]},
        "Q101": {"styles": [style_qids[0]]},
        "Q102": {"styles": [style_qids[0]]},
        "Q200": {"styles": [style_qids[1]]},
        "Q201": {"styles": [style_qids[1]]},
        "Q202": {"styles": [style_qids[1]]},
        "Q999": {"styles": []},
    }

    for qid in churches:
        Image.new("RGB", (1, 1), "white").save(images_dir / f"{qid}_wd0.jpg")

    with open(tmp_path / "churches.json", "w", encoding="utf-8") as f:
        json.dump(churches, f)

    return tmp_path


@pytest.fixture()
def full_dataset(dataset_root: Path) -> FullDataset:
    return FullDataset(dataset_root)


@pytest.fixture()
def metadata_only_dataset(dataset_root: Path) -> _NoGetItemFullDataset:
    return _NoGetItemFullDataset(dataset_root)


def _label_counter(dataset: FullDataset, indices: list[int]) -> Counter[int]:
    labels = [
        dataset.get_metadata(idx)["style_label"]
        for idx in indices
        if dataset.get_metadata(idx)["style_label"] is not None
    ]
    return Counter(label for label in labels if label is not None)


class TestFineTunerMetadataSetup:
    """Training setup should stay on metadata-only paths."""

    def test_stratified_split_is_deterministic_for_fixed_seed(self, full_dataset: FullDataset) -> None:
        config = FineTuningConfig(model_name="dinov2", seed=123, val_split=0.5)

        tuner_a = FineTuner(config)
        train_a, val_a, excluded_a = tuner_a._stratified_split(full_dataset, config.val_split)

        tuner_b = FineTuner(config)
        train_b, val_b, excluded_b = tuner_b._stratified_split(full_dataset, config.val_split)

        assert train_a.indices == train_b.indices
        assert val_a.indices == val_b.indices
        assert excluded_a == excluded_b == 0

    def test_stratified_split_excludes_eval_ids_and_preserves_label_distribution(
        self,
        full_dataset: FullDataset,
    ) -> None:
        config = FineTuningConfig(model_name="dinov2", seed=7, val_split=0.5)
        tuner = FineTuner(config)
        exclude_image_ids = {"Q100_wd0.jpg", "Q200_wd0.jpg"}

        train_subset, val_subset, n_excluded = tuner._stratified_split(
            full_dataset,
            config.val_split,
            exclude_image_ids=exclude_image_ids,
        )

        train_ids = {full_dataset.get_metadata(idx)["image_id"] for idx in train_subset.indices}
        val_ids = {full_dataset.get_metadata(idx)["image_id"] for idx in val_subset.indices}

        included_indices = [
            idx
            for idx in range(len(full_dataset))
            if full_dataset.get_metadata(idx)["image_id"] not in exclude_image_ids
            and full_dataset.get_metadata(idx)["style_label"] is not None
        ]
        included_counts = _label_counter(full_dataset, included_indices)
        train_counts = _label_counter(full_dataset, train_subset.indices)
        val_counts = _label_counter(full_dataset, val_subset.indices)

        assert n_excluded == 2
        assert exclude_image_ids.isdisjoint(train_ids)
        assert exclude_image_ids.isdisjoint(val_ids)
        assert train_counts + val_counts == included_counts

        for label, count in included_counts.items():
            expected_val = max(1, int(count * config.val_split))
            assert val_counts[label] == expected_val
            assert train_counts[label] == count - expected_val

    def test_collect_labels_for_indices_uses_metadata_only(
        self,
        metadata_only_dataset: _NoGetItemFullDataset,
    ) -> None:
        config = FineTuningConfig(model_name="dinov2", seed=11, val_split=0.5)
        tuner = FineTuner(config)

        labels = tuner._collect_labels_for_indices(
            metadata_only_dataset,
            list(range(len(metadata_only_dataset))),
        )

        assert labels == [0, 0, 0, 1, 1, 1]

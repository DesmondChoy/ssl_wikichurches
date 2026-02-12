"""Tests for AnnotatedSubset.image_ids property.

Verifies that metadata-only iteration (image_ids + annotations) is
consistent with __getitem__ and safe from external mutation.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from PIL import Image

from ssl_attention.data.wikichurches import AnnotatedSubset


@pytest.fixture()
def tiny_dataset(tmp_path: Path) -> AnnotatedSubset:
    """Create a minimal AnnotatedSubset with 3 dummy images."""
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    # Create dummy images (1x1 white pixel)
    ids = ["Q333_wd0.jpg", "Q111_wd0.jpg", "Q222_wd0.jpg"]
    for img_id in ids:
        Image.new("RGB", (1, 1), "white").save(images_dir / img_id)

    # Minimal building_parts.json matching load_annotations schema:
    # top-level "annotations" key, bbox_groups with elements
    annotations = {
        "annotations": {
            "Q333_wd0.jpg": {
                "styles": ["Q46261"],
                "bbox_groups": [
                    {
                        "group_label": 0,
                        "elements": [
                            {"left": 0.1, "top": 0.2, "width": 0.3, "height": 0.4, "label": 0},
                        ],
                    },
                ],
            },
            "Q111_wd0.jpg": {
                "styles": ["Q46261"],
                "bbox_groups": [
                    {
                        "group_label": 0,
                        "elements": [
                            {"left": 0.0, "top": 0.0, "width": 0.5, "height": 0.5, "label": 0},
                        ],
                    },
                ],
            },
            "Q222_wd0.jpg": {
                "styles": [],
                "bbox_groups": [],
            },
        },
    }
    with open(tmp_path / "building_parts.json", "w") as f:
        json.dump(annotations, f)

    return AnnotatedSubset(tmp_path)


class TestImageIdsProperty:
    """Tests for the image_ids property on AnnotatedSubset."""

    def test_image_ids_matches_getitem_order(self, tiny_dataset: AnnotatedSubset) -> None:
        """image_ids[i] equals dataset[i]['image_id'] for all i."""
        for i, image_id in enumerate(tiny_dataset.image_ids):
            assert image_id == tiny_dataset[i]["image_id"]

    def test_image_ids_returns_copy(self, tiny_dataset: AnnotatedSubset) -> None:
        """Mutating the returned list does not affect the dataset."""
        ids = tiny_dataset.image_ids
        ids.clear()

        assert len(tiny_dataset.image_ids) == 3

    def test_image_ids_are_sorted(self, tiny_dataset: AnnotatedSubset) -> None:
        """image_ids are in deterministic sorted order."""
        ids = tiny_dataset.image_ids
        assert ids == sorted(ids)

    def test_annotations_keys_match_image_ids(self, tiny_dataset: AnnotatedSubset) -> None:
        """The set of image_ids equals the set of annotation keys."""
        assert set(tiny_dataset.image_ids) == set(tiny_dataset.annotations.keys())

    def test_length_matches(self, tiny_dataset: AnnotatedSubset) -> None:
        """len(image_ids) equals len(dataset)."""
        assert len(tiny_dataset.image_ids) == len(tiny_dataset)

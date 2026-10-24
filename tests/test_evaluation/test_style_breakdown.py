"""Per-style breakdown must assign each image to exactly one style group."""

from __future__ import annotations

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType

import pytest

ROMANESQUE, GOTHIC = "Q46261", "Q176483"


def _load_module() -> ModuleType:
    script_path = Path(__file__).resolve().parents[2] / "experiments" / "scripts" / "analyze_style_breakdown.py"
    spec = importlib.util.spec_from_file_location("analyze_style_breakdown", script_path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_multi_style_image_counts_once_under_its_primary_style(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    module = _load_module()
    annotations = {
        "annotations": {
            "gothic_first.jpg": {"styles": [GOTHIC, ROMANESQUE]},
            "romanesque.jpg": {"styles": [ROMANESQUE]},
            "unmapped_first.jpg": {"styles": ["Q0", ROMANESQUE]},
        }
    }
    annotations_path = tmp_path / "building_parts.json"
    annotations_path.write_text(json.dumps(annotations))
    monkeypatch.setattr(module, "ANNOTATIONS_PATH", annotations_path)

    grouped = module.group_deltas_by_style(
        {"gothic_first.jpg": 1.0, "romanesque.jpg": 2.0, "unmapped_first.jpg": 3.0},
        module.load_image_styles(),
    )

    assert grouped["Gothic"] == [1.0]
    assert grouped["Romanesque"] == [2.0]
    assert sum(len(deltas) for deltas in grouped.values()) == 2

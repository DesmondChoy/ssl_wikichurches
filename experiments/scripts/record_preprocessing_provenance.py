#!/usr/bin/env python3
"""Record the image preprocessing behind the attention caches and classification checkpoints.

Writes each model's attention-evaluation processor (whole image, no crop) and native
classification processor, plus library versions and the git commit, so regenerated
artifacts can be traced to the exact preprocessing that produced them.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

from transformers import AutoImageProcessor

from ssl_attention.config import FINETUNE_MODELS, MODELS
from ssl_attention.models.base import load_full_image_processor

PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_PATH = PROJECT_ROOT / "outputs" / "results" / "preprocessing_provenance.json"
PROCESSOR_KEYS = ("image_processor_type", "do_resize", "size", "resample", "do_center_crop", "crop_size")


def processor_summary(processor: Any) -> dict[str, Any]:
    config = processor.to_dict()
    return {key: config.get(key) for key in PROCESSOR_KEYS}


def git_commit() -> str:
    return subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, capture_output=True, text=True, check=True
    ).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=OUTPUT_PATH)
    args = parser.parse_args()

    models = {
        name: {
            "model_id": MODELS[name].model_id,
            "attention": processor_summary(load_full_image_processor(MODELS[name].model_id)),
            "classification": processor_summary(AutoImageProcessor.from_pretrained(MODELS[name].model_id)),
        }
        for name in sorted(FINETUNE_MODELS)
    }
    record = {
        "recorded_at": datetime.now(UTC).isoformat(),
        "git_commit": git_commit(),
        "python": platform.python_version(),
        "packages": {pkg: version(pkg) for pkg in ("torch", "torchvision", "transformers", "pillow")},
        "models": models,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(record, indent=2) + "\n")
    print(args.output)


if __name__ == "__main__":
    main()

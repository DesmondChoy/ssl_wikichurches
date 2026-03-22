"""Generate image assets for the mid-project progress presentation.

Reads from pre-computed heatmap cache and metrics data — no model inference.

Output: PNG images in outputs/slides/ for embedding in the PPTX.
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

matplotlib.use("Agg")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[2]
CACHE = ROOT / "outputs" / "cache" / "heatmaps"
ORIGINALS_CLEAN = CACHE / "originals" / "clean"
ORIGINALS_BBOX = CACHE / "originals" / "bbox"
FIGURES = ROOT / "outputs" / "figures"
RESULTS = ROOT / "outputs" / "results"
OUT = ROOT / "outputs" / "slides"
OUT.mkdir(parents=True, exist_ok=True)

# Selected images per style (highest bbox count)
STYLE_IMAGES = {
    "Romanesque": "Q526047_wd0.jpg",
    "Gothic": "Q2981_wd0.jpg",
    "Renaissance": "Q1165020_wd0.jpg",
    "Baroque": "Q1502706_wd0.jpg",
}

# Hero image for title slide + motivation comparison
HERO_IMAGE = "Q206823_wd0.jpg"  # Gothic, 10 bboxes

# Palette (matches project)
STEEL_BLUE = "#4E79A7"
TEAL = "#93B7BE"
TERRACOTTA = "#D4764E"
CHARCOAL = "#2D3436"
WARM_GRAY = "#8A817C"
SUCCESS = "#3A7D44"
FAILURE = "#C04E4E"


def _load_cached_image(model: str, layer: str, method: str, variant: str, image_id: str) -> Image.Image:
    """Load a pre-rendered heatmap image from cache."""
    path = CACHE / model / layer / method / variant / f"{image_id}.png"
    if not path.exists():
        raise FileNotFoundError(f"Cached image not found: {path}")
    return Image.open(path).convert("RGBA")


def _add_label(img: Image.Image, label: str, position: str = "bottom") -> Image.Image:
    """Add a text label bar to the top or bottom of an image."""
    bar_h = 32
    w, h = img.size
    new_h = h + bar_h
    canvas = Image.new("RGBA", (w, new_h), (45, 52, 54, 255))  # charcoal
    y_offset = bar_h if position == "top" else 0
    canvas.paste(img, (0, y_offset))
    draw = ImageDraw.Draw(canvas)
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Helvetica.ttc", 18)
    except (OSError, IOError):
        font = ImageFont.load_default()
    text_y = new_h - bar_h + 6 if position == "bottom" else 6
    bbox = draw.textbbox((0, 0), label, font=font)
    text_w = bbox[2] - bbox[0]
    draw.text(((w - text_w) // 2, text_y), label, fill=(255, 255, 255, 255), font=font)
    return canvas


# ---------------------------------------------------------------------------
# Slide 1: Title hero image (DINOv3 overlay + bboxes on Gothic church)
# ---------------------------------------------------------------------------
def generate_slide1_title():
    print("Generating Slide 1: Title hero image...")
    img = _load_cached_image("dinov3", "layer11", "cls", "overlay_bbox", HERO_IMAGE)
    img.save(OUT / "slide01_title_hero.png")
    print(f"  -> {OUT / 'slide01_title_hero.png'}")


# ---------------------------------------------------------------------------
# Slide 2: Good vs bad attention comparison (DINOv3 vs MAE, same image)
# ---------------------------------------------------------------------------
def generate_slide2_motivation():
    print("Generating Slide 2: Good vs bad attention comparison...")
    good = _load_cached_image("dinov3", "layer11", "cls", "overlay_bbox", HERO_IMAGE)
    bad = _load_cached_image("mae", "layer4", "cls", "overlay_bbox", HERO_IMAGE)

    good_labelled = _add_label(good, "DINOv3 (Self-Distillation) — IoU: 0.133")
    bad_labelled = _add_label(bad, "MAE (Reconstruction) — IoU: 0.037")

    # Side by side with padding
    padding = 16
    w = good_labelled.width + bad_labelled.width + padding
    h = max(good_labelled.height, bad_labelled.height)
    canvas = Image.new("RGBA", (w, h), (248, 249, 250, 255))
    canvas.paste(good_labelled, (0, 0))
    canvas.paste(bad_labelled, (good_labelled.width + padding, 0))

    canvas.save(OUT / "slide02_good_vs_bad.png")
    print(f"  -> {OUT / 'slide02_good_vs_bad.png'}")


# ---------------------------------------------------------------------------
# Slide 4: 2x2 grid of churches by style with bounding boxes
# ---------------------------------------------------------------------------
def generate_slide4_dataset():
    print("Generating Slide 4: Style grid...")
    images = []
    for style, img_id in STYLE_IMAGES.items():
        path = ORIGINALS_BBOX / f"{img_id}.png"
        if not path.exists():
            print(f"  WARNING: {path} not found, skipping {style}")
            continue
        img = Image.open(path).convert("RGBA")
        img = _add_label(img, style, position="bottom")
        images.append(img)

    if len(images) != 4:
        print("  WARNING: Expected 4 images, got", len(images))
        return

    # Resize all to same dimensions
    target_w = max(img.width for img in images)
    target_h = max(img.height for img in images)
    resized = []
    for img in images:
        canvas = Image.new("RGBA", (target_w, target_h), (248, 249, 250, 255))
        x_off = (target_w - img.width) // 2
        y_off = (target_h - img.height) // 2
        canvas.paste(img, (x_off, y_off))
        resized.append(canvas)

    # 2x2 grid
    gap = 12
    grid_w = target_w * 2 + gap
    grid_h = target_h * 2 + gap
    canvas = Image.new("RGBA", (grid_w, grid_h), (248, 249, 250, 255))
    canvas.paste(resized[0], (0, 0))
    canvas.paste(resized[1], (target_w + gap, 0))
    canvas.paste(resized[2], (0, target_h + gap))
    canvas.paste(resized[3], (target_w + gap, target_h + gap))

    canvas.save(OUT / "slide04_style_grid.png")
    print(f"  -> {OUT / 'slide04_style_grid.png'}")


# ---------------------------------------------------------------------------
# Slide 6: Methodology pipeline (4 panels)
# ---------------------------------------------------------------------------
def generate_slide6_pipeline():
    print("Generating Slide 6: Pipeline strip...")
    img_id = HERO_IMAGE

    panels = [
        (ORIGINALS_CLEAN / f"{img_id}.png", "1. Original Image"),
        (CACHE / "dinov3" / "layer11" / "cls" / "heatmap" / f"{img_id}.png", "2. Attention Heatmap"),
        (ORIGINALS_BBOX / f"{img_id}.png", "3. Expert Annotations"),
        (CACHE / "dinov3" / "layer11" / "cls" / "overlay_bbox" / f"{img_id}.png", "4. Compute IoU"),
    ]

    images = []
    for path, label in panels:
        if not path.exists():
            print(f"  WARNING: {path} not found")
            continue
        img = Image.open(path).convert("RGBA")
        img = _add_label(img, label, position="bottom")
        images.append(img)

    if not images:
        return

    # Resize to uniform height
    target_h = max(img.height for img in images)
    gap = 8

    # Add arrow indicators between panels
    total_w = sum(img.width for img in images) + gap * (len(images) - 1)
    canvas = Image.new("RGBA", (total_w, target_h), (248, 249, 250, 255))

    x = 0
    for img in images:
        y = (target_h - img.height) // 2
        canvas.paste(img, (x, y))
        x += img.width + gap

    canvas.save(OUT / "slide06_pipeline.png")
    print(f"  -> {OUT / 'slide06_pipeline.png'}")


# ---------------------------------------------------------------------------
# Slide 13: Frozen IoU vs Delta IoU scatter
# ---------------------------------------------------------------------------
def generate_slide13_scatter():
    print("Generating Slide 13: Frozen IoU vs Delta IoU scatter...")

    # Data from metrics_summary.json and q2_metrics_analysis.json
    models_data = {
        "DINOv3":  {"frozen": 0.1327, "delta": 0.009, "strategy": "LoRA", "sig": False},
        "DINOv2":  {"frozen": 0.0816, "delta": 0.002, "strategy": "LoRA", "sig": False},
        "CLIP":    {"frozen": 0.0485, "delta": 0.063, "strategy": "LoRA", "sig": True},
        "SigLIP":  {"frozen": 0.0466, "delta": 0.036, "strategy": "Full", "sig": True},
        "SigLIP 2": {"frozen": 0.0466, "delta": 0.036, "strategy": "Full", "sig": True},
        "MAE":     {"frozen": 0.0374, "delta": 0.001, "strategy": "LoRA", "sig": False},
    }

    fig, ax = plt.subplots(figsize=(8, 6))

    for name, d in models_data.items():
        color = SUCCESS if d["sig"] else WARM_GRAY
        marker = "D" if d["strategy"] == "LoRA" else "o"
        ax.scatter(d["frozen"], d["delta"], c=color, s=120, marker=marker,
                   edgecolors=CHARCOAL, linewidths=0.8, zorder=5)
        # Label offset to avoid overlap
        x_off, y_off = 0.003, 0.002
        if name == "SigLIP 2":
            y_off = -0.005
        ax.annotate(name, (d["frozen"] + x_off, d["delta"] + y_off),
                    fontsize=11, fontweight="bold", color=CHARCOAL)

    # Quadrant annotations
    ax.axhline(y=0.01, color="#cccccc", linestyle="--", linewidth=0.8)
    ax.text(0.11, 0.055, "Already\naligned", fontsize=9, color="#999999",
            ha="center", style="italic")
    ax.text(0.045, 0.055, "Improvable\nvia fine-tuning", fontsize=9,
            color=SUCCESS, ha="center", style="italic", fontweight="bold")
    ax.text(0.045, -0.002, "Low alignment,\nno improvement", fontsize=9,
            color=FAILURE, ha="center", style="italic")

    ax.set_xlabel("Frozen IoU @ 90th Percentile (Best Layer)", fontsize=12)
    ax.set_ylabel("Best \u0394 IoU (Fine-tuned \u2212 Frozen)", fontsize=12)
    ax.set_title("Pre-training Objective Determines Alignment & Plasticity",
                 fontsize=14, fontweight="bold", pad=15)
    ax.set_xlim(0.02, 0.16)
    ax.set_ylim(-0.008, 0.075)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    from matplotlib.lines import Line2D
    legend_elements = [
        Line2D([0], [0], marker="D", color="w", markerfacecolor=SUCCESS,
               markersize=10, markeredgecolor=CHARCOAL, label="Significant (LoRA)"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=SUCCESS,
               markersize=10, markeredgecolor=CHARCOAL, label="Significant (Full)"),
        Line2D([0], [0], marker="D", color="w", markerfacecolor=WARM_GRAY,
               markersize=10, markeredgecolor=CHARCOAL, label="Not significant"),
    ]
    ax.legend(handles=legend_elements, loc="upper right", framealpha=0.9)

    plt.tight_layout()
    fig.savefig(OUT / "slide13_scatter.png", dpi=200, bbox_inches="tight",
                facecolor="white", edgecolor="none")
    plt.close(fig)
    print(f"  -> {OUT / 'slide13_scatter.png'}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    print(f"Output directory: {OUT}\n")
    generate_slide1_title()
    generate_slide2_motivation()
    generate_slide4_dataset()
    generate_slide6_pipeline()
    generate_slide13_scatter()
    print(f"\nDone! {len(list(OUT.glob('*.png')))} images generated in {OUT}")


if __name__ == "__main__":
    main()

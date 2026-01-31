# WikiChurches Dataset

Download and work with the [WikiChurches dataset](https://zenodo.org/records/5166987) - 72,000+ church images with architectural style labels.

## Setup

```bash
uv sync
```

## Download Dataset

```bash
# Download all files to dataset/ folder (~11.8 GB)
uv run python scripts/download_wikichurches.py -o dataset/

# Download metadata only (skip images/models)
uv run python scripts/download_wikichurches.py -o dataset/ --exclude images.zip models.zip

# List available files
uv run python scripts/download_wikichurches.py --list
```

The script supports **resuming interrupted downloads** - just re-run the command.

## Dataset Contents

Files are downloaded to the `dataset/` folder:

| File | Size | Description |
|------|------|-------------|
| images.zip | 11.5 GB | Church building images |
| models.zip | 250.7 MB | Pre-trained models |
| image_meta.json | 35.9 MB | Image metadata |
| churches.json | 3.2 MB | Church information |
| building_parts.json | 303.9 kB | Building part annotations |
| labels.zip | 252.5 kB | Label files |
| style_names.txt | 3.5 kB | 121 architectural style names |
| parent_child_rel.txt | 925 B | Style hierarchy |

## Citation

> Björn Barz and Joachim Denzler.
> [*WikiChurches: A Fine-Grained Dataset of Architectural Styles with Real-World Challenges.*](https://arxiv.org/pdf/2108.06959)
> arXiv preprint arXiv:2108.06959, 2021.

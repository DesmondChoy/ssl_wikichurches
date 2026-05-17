# Do Self-Supervised Vision Models Learn What Experts See?

This repository evaluates whether self-supervised vision models attend to the same architectural features that WikiChurches experts mark as diagnostically important.

GitHub repo: [DesmondChoy/ssl_wikichurches](https://github.com/DesmondChoy/ssl_wikichurches)

The project answers three questions:

1. **Q1: Frozen attention alignment** - how well frozen models align with expert boxes across IoU, Coverage, MSE, KL, and EMD.
2. **Q2: Fine-tuning effects** - how Linear Probe, LoRA, and Full fine-tuning change that alignment.
3. **Q3: Head specialization** - which attention heads align best with specific architectural features.

## Submission Artifacts

Start here if you are reviewing the academic submission:

- Final PDF: [docs/final_report/ISY5004_report_final.pdf](docs/final_report/ISY5004_report_final.pdf)
- Precomputed data, cache, and checkpoint artifacts: [Google Drive artifact folder](https://drive.google.com/drive/folders/1pT8VrK6d9h-sZzAr6qhPxvNrVrRi-8Cd?usp=sharing)
- Report Markdown source: [docs/core/project_report_final.md](docs/core/project_report_final.md)
- Report figures: [docs/final_report/figures/](docs/final_report/figures/)
- Q3 report-view source figures: [docs/core/assets/](docs/core/assets/)
- Video plan, slide outline, script, PDF, and PPTX: [docs/plans/video/](docs/plans/video/)
- Active experiment pointer: [outputs/results/active_experiment.json](outputs/results/active_experiment.json)
- Q2 run matrix: [outputs/results/experiments/fine_tuning_primary_20260327/run_matrix.json](outputs/results/experiments/fine_tuning_primary_20260327/run_matrix.json)
- Q2 analysis: [outputs/results/experiments/fine_tuning_primary_20260327/q2_metrics_analysis.json](outputs/results/experiments/fine_tuning_primary_20260327/q2_metrics_analysis.json)

The Google Drive artifact folder contains the ignored large artifacts
needed to run the app:

- `dataset/`
- `outputs/cache/`
- `outputs/checkpoints/fine_tuning_primary_20260327/`

With the repo plus that Drive folder, reviewers do **not** need to rerun
fine-tuning, cache generation, heatmap generation, or Q2/Q3 precompute workflows.

## Quick Start For Reviewers

Requirements: Python 3.12+, [uv](https://github.com/astral-sh/uv), Node.js 18+.

```bash
uv sync
```

Download the [Google Drive artifact folder](https://drive.google.com/drive/folders/1pT8VrK6d9h-sZzAr6qhPxvNrVrRi-8Cd?usp=sharing), then copy `dataset/` and `outputs/` from that folder into the repository root.

On macOS or Linux, from inside the downloaded artifact folder:

```bash
rsync -av dataset outputs /path/to/ssl_wikichurches/
```

Replace `/path/to/ssl_wikichurches/` with the local path to this cloned repo.

Expected local structure:

```text
ssl_wikichurches/
├── dataset/
└── outputs/
    ├── cache/
    │   ├── attention_viz.h5
    │   ├── features.h5
    │   ├── metrics.db
    │   ├── metrics_summary.json
    │   └── heatmaps/
    └── checkpoints/
        └── fine_tuning_primary_20260327/
```

Run the app:

```bash
./dev.sh
```

This starts the backend at `http://127.0.0.1:8000` and frontend at `http://127.0.0.1:5173`.

## App Routes

| Route | Purpose |
|------|---------|
| `/` | Gallery of annotated WikiChurches images |
| `/image/:imageId` | Single-image overlays, annotations, metrics, and Q3 drill-down |
| `/compare` | Frozen model and variant comparisons |
| `/dashboard` | Q1 overview and main Q3 discovery surface |
| `/q2` | Fine-tuning summary from the active experiment |
| `/q3-report` | Report-focused Q3 head ranking, feature matrix, and frozen-to-adapted delta views |

## Optional: Regenerate Artifacts From Scratch

Skip this section for normal review. The Google Drive artifact folder already
contains the dataset, app cache, and canonical fine-tuned checkpoints.

Use this section only if you intentionally want to rebuild the local artifacts
instead of using the submitted Drive package.

Download one dataset path:

- **Annotated subset:** [Google Drive package](https://drive.google.com/drive/folders/1fsf0k71ADeYCBAwo-dIPntUmpibaoGBr) with 139 images plus `building_parts.json`. Use this for the app, Q1/Q3 cache generation, and expert-alignment evaluation.
- **Official WikiChurches files:** use the downloader when you need `churches.json`, metadata, the full image archive, or the official annotation file.

```bash
uv run python scripts/download_wikichurches.py --list
uv run python scripts/download_wikichurches.py --files churches.json image_meta.json building_parts.json
```

Expected dataset structure:

```text
dataset/
├── images/
│   ├── Q18785543_wd0.jpg
│   └── ...
└── building_parts.json
```

Generate the baseline app caches:

```bash
uv run python -m app.precompute.generate_attention_cache --models all
uv run python -m app.precompute.generate_feature_cache --models all
uv run python -m app.precompute.generate_heatmap_images --colormap viridis
uv run python -m app.precompute.generate_metrics_cache
```

Generate the Q3 per-head cache scope:

```bash
uv run python -m app.precompute.generate_attention_cache --models dinov2 dinov3 mae clip --per-head
uv run python -m app.precompute.generate_metrics_cache --models dinov2 dinov3 mae clip --per-head
uv run python -m app.precompute.generate_attention_cache --finetuned --models dinov2 dinov3 mae clip --strategies lora full --per-head
uv run python -m app.precompute.generate_metrics_cache --finetuned --models dinov2 dinov3 mae clip --strategies lora full --per-head
```

Optional Q3 frozen-backbone control:

```bash
uv run python -m app.precompute.generate_attention_cache --finetuned --models dinov2 dinov3 mae clip --strategies linear_probe --per-head
uv run python -m app.precompute.generate_metrics_cache --finetuned --models dinov2 dinov3 mae clip --strategies linear_probe --per-head
```

## Optional: Q2 Reproduction

The primary experiment ID is `fine_tuning_primary_20260327`.
This path requires the style-labeled pool from `churches.json`, not only the 139-image annotated subset.
Reviewers do not need this path to inspect the submitted app or report because
the checked-in result artifacts and Drive checkpoints already cover the primary
experiment.

```bash
EXPERIMENT_ID=fine_tuning_primary_20260327

uv run python experiments/scripts/fine_tune_models.py --all --freeze-backbone --epochs 3 --experiment-id "$EXPERIMENT_ID"
uv run python experiments/scripts/fine_tune_models.py --all --lora --epochs 3 --experiment-id "$EXPERIMENT_ID"
uv run python experiments/scripts/fine_tune_models.py --all --epochs 3 --experiment-id "$EXPERIMENT_ID"

uv run python experiments/scripts/analyze_q2_metrics.py \
  --experiment-id "$EXPERIMENT_ID" \
  --models clip dinov2 dinov3 mae siglip siglip2 \
  --strategies linear_probe lora full
```

Supplementary Q2 analyses:

```bash
uv run python experiments/scripts/analyze_style_breakdown.py --experiment-id "$EXPERIMENT_ID" --strategy full
uv run python experiments/scripts/analyze_model_correlation.py --experiment-id "$EXPERIMENT_ID" --strategy full
uv run python experiments/scripts/analyze_feature_delta_iou.py --experiment-id "$EXPERIMENT_ID" --model mae --strategy full --style Renaissance
uv run python experiments/scripts/analyze_q1_continuous_baselines.py
```

## Optional: Report and Presentation Outputs

Use this section only to regenerate report-facing figures and presentation
assets:

```bash
uv run python experiments/scripts/generate_run_matrix_figures.py
uv run python experiments/scripts/generate_slide_images.py
cd experiments/scripts && npm install && node create_presentation.js
```

The Q3 report route supplies the screenshot-friendly views used by the report and video plan:

- `view=head-ranking` for ranked heads by model, variant, layer, metric, and percentile
- `view=head-feature-matrix` for head-by-feature evidence
- `view=frozen-delta` for frozen-to-LoRA and frozen-to-Full ranking shifts

The full command surface is in [docs/reference/cli_reference.md](docs/reference/cli_reference.md). The experiment artifact contract is in [docs/reference/fine_tuning_run_matrix.md](docs/reference/fine_tuning_run_matrix.md).

## Developer Checks

```bash
uv run ruff check .
uv run mypy
uv run pytest
cd app/frontend && npm run lint && npm run build
cd app/frontend && npm run test:e2e
```

## Documentation

| Document | Use it for |
|----------|------------|
| [docs/README.md](docs/README.md) | Documentation index |
| [docs/user_guide.md](docs/user_guide.md) | App walkthroughs |
| [docs/reference/cli_reference.md](docs/reference/cli_reference.md) | Command and flag reference |
| [docs/reference/api_reference.md](docs/reference/api_reference.md) | Backend API contracts |
| [docs/reference/fine_tuning_run_matrix.md](docs/reference/fine_tuning_run_matrix.md) | Q2 artifact layout |
| [docs/reference/per_head_methodology.md](docs/reference/per_head_methodology.md) | Q3 per-head method |

## Layout

```text
ssl_wikichurches/
├── app/                 # FastAPI backend, React frontend, cache scripts
├── dataset/             # Local WikiChurches data, not tracked
├── docs/                # Report, references, and user docs
├── experiments/         # Fine-tuning and analysis scripts
├── outputs/             # Results, figures, local caches, checkpoints
├── scripts/             # Dataset and utility scripts
├── src/ssl_attention/   # Core library code
└── tests/               # Pytest suite
```

## References

The report's related-work discussion explains how each source is used:
[docs/core/project_report_final.md](docs/core/project_report_final.md).

Abnar, S., & Zuidema, W. (2020). Quantifying Attention Flow in Transformers. *Proceedings of the 58th Annual Meeting of the Association for Computational Linguistics (ACL)*. [arXiv:2005.00928](https://arxiv.org/abs/2005.00928)

Barz, B., & Denzler, J. (2021). WikiChurches: A Fine-Grained Dataset of Architectural Styles with Real-World Challenges. *Proceedings of the 35th Conference on Neural Information Processing Systems (NeurIPS) Datasets and Benchmarks Track*. [arXiv:2108.06959](https://arxiv.org/abs/2108.06959)

Biderman, D., Portes, J., Ortiz, J. J. G., Paul, M., Greengard, P., Jennings, C., King, D., Havens, S., Chiley, V., Frankle, J., Blakeney, C., & Cunningham, J. P. (2024). LoRA Learns Less and Forgets Less. *Transactions on Machine Learning Research (TMLR)*. [arXiv:2405.09673](https://arxiv.org/abs/2405.09673)

Caron, M., Touvron, H., Misra, I., Jégou, H., Mairal, J., Bojanowski, P., & Joulin, A. (2021). Emerging Properties in Self-Supervised Vision Transformers. *Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)*. [arXiv:2104.14294](https://arxiv.org/abs/2104.14294)

Chefer, H., Gur, S., & Wolf, L. (2021). Transformer Interpretability Beyond Attention Visualization. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*. [arXiv:2012.09838](https://arxiv.org/abs/2012.09838)

Chung, M., Won, J. B., Kim, G., Kim, Y., & Ozbulak, U. (2024). Evaluating Visual Explanations of Attention Maps for Transformer-based Medical Imaging. *MICCAI 2024 Workshop on Interpretability of Machine Intelligence in Medical Image Computing (iMIMIC)*. [arXiv:2503.09535](https://arxiv.org/abs/2503.09535)

He, K., Chen, X., Xie, S., Li, Y., Dollár, P., & Girshick, R. (2022). Masked Autoencoders Are Scalable Vision Learners. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*, pp. 16000-16009. [arXiv:2111.06377](https://arxiv.org/abs/2111.06377)

Kumar, A., Raghunathan, A., Jones, R., Ma, T., & Liang, P. (2022). Fine-Tuning can Distort Pretrained Features and Underperform Out-of-Distribution. *Proceedings of the 10th International Conference on Learning Representations (ICLR)*. [arXiv:2202.10054](https://arxiv.org/abs/2202.10054)

Li, A. C., Tian, Y., Chen, B., Pathak, D., & Chen, X. (2024). On the Surprising Effectiveness of Attention Transfer for Vision Transformers. *Proceedings of the 38th Conference on Neural Information Processing Systems (NeurIPS)*. [arXiv:2411.09702](https://arxiv.org/abs/2411.09702)

Li, Y., Wang, J., Dai, X., Wang, L., Yeh, C.-C. M., Zheng, Y., Zhang, W., & Ma, K.-L. (2023). How Does Attention Work in Vision Transformers? A Visual Analytics Attempt. *IEEE Transactions on Visualization and Computer Graphics (TVCG)*. doi:10.1109/TVCG.2023.3261935. [arXiv:2303.13731](https://arxiv.org/abs/2303.13731)

Oquab, M., Darcet, T., Moutakanni, T., Vo, H., Szafraniec, M., Khalidov, V., Fernandez, P., Haziza, D., Massa, F., El-Nouby, A., Assran, M., Ballas, N., Galuba, W., Howes, R., Huang, P.-Y., Li, S.-W., Misra, I., Rabbat, M., Sharma, V., Synnaeve, G., Xu, H., Jegou, H., Mairal, J., Labatut, P., Joulin, A., & Bojanowski, P. (2024). DINOv2: Learning Robust Visual Features without Supervision. *Transactions on Machine Learning Research (TMLR)*. [arXiv:2304.07193](https://arxiv.org/abs/2304.07193)

Park, N., Kim, W., Heo, B., Kim, T., & Yun, S. (2023). What Do Self-Supervised Vision Transformers Learn? *Proceedings of the 11th International Conference on Learning Representations (ICLR)*. [arXiv:2305.00729](https://arxiv.org/abs/2305.00729)

Radford, A., Kim, J. W., Hallacy, C., Ramesh, A., Goh, G., Agarwal, S., Sastry, G., Askell, A., Mishkin, P., Clark, J., Krueger, G., & Sutskever, I. (2021). Learning Transferable Visual Models From Natural Language Supervision. *Proceedings of the 38th International Conference on Machine Learning (ICML)*, PMLR vol. 139. [arXiv:2103.00020](https://arxiv.org/abs/2103.00020)

Siméoni, O., Vo, H. V., Seitzer, M., Baldassarre, F., Oquab, M., Jose, C., Khalidov, V., Szafraniec, M., Yi, S., Ramamonjisoa, M., Massa, F., Haziza, D., Wehrstedt, L., Wang, J., Darcet, T., Moutakanni, T., Sentana, L., Roberts, C., Vedaldi, A., Tolan, J., Brandt, J., Couprie, C., Mairal, J., Jégou, H., Labatut, P., & Bojanowski, P. (2025). DINOv3. *arXiv preprint*. [arXiv:2508.10104](https://arxiv.org/abs/2508.10104)

Voita, E., Talbot, D., Moiseev, F., Sennrich, R., & Titov, I. (2019). Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting, the Rest Can Be Pruned. *Proceedings of the 57th Annual Meeting of the Association for Computational Linguistics (ACL)*. [arXiv:1905.09418](https://arxiv.org/abs/1905.09418)

Walmer, M., Suri, S., Gupta, K., & Shrivastava, A. (2023). Teaching Matters: Investigating the Role of Supervision in Vision Transformers. *Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)*. [arXiv:2212.03862](https://arxiv.org/abs/2212.03862)

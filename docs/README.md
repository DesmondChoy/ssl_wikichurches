# Documentation

This directory complements the root [`README.md`](../README.md) with current workflow guides, API and CLI references, methodology notes, and research context for the SSL WikiChurches project.

The current public surface covers:

- seven model keys: `dinov2`, `dinov3`, `mae`, `clip`, `siglip`, `siglip2`, `resnet50`
- app routes: `/`, `/image/:imageId`, `/compare`, `/dashboard`, `/q2`, `/q3`
- experiment-scoped Q2 artifacts selected through `outputs/results/active_experiment.json`

`docs/archive/` contains historical material and is not part of the current-state navigation below.

## Start Here

| Document | Best for |
|----------|----------|
| [../README.md](../README.md) | Setup, dataset paths, app startup, major workflows, and developer commands |
| [user_guide.md](user_guide.md) | Gallery, Image Detail, Compare, Dashboard, Q2, and Q3 product walkthroughs |
| [reference/cli_reference.md](reference/cli_reference.md) | Complete command, option, and script reference |
| [reference/api_reference.md](reference/api_reference.md) | Backend endpoints, query parameters, and response contracts |

## Structure

### [`core/`](core/) — Project Narrative
Current project framing, scope, and implementation status.

| Document | Description |
|----------|-------------|
| [project_req.pdf](core/project_req.pdf) | Original course requirements |
| [project_proposal.md](core/project_proposal.md) | Current project framing, research questions, methodology, and deliverable surface |
| [implementation_plan.md](core/implementation_plan.md) | Architecture, phase tracking, and implementation notes |
| [one_pager_pitch.md](core/one_pager_pitch.md) | Short-form current-state project summary |

### [`reference/`](reference/) — Operational Reference
Current command surfaces, APIs, artifacts, and methodology contracts.

| Document | Description |
|----------|-------------|
| [cli_reference.md](reference/cli_reference.md) | Exhaustive command and flag reference for app runtime, precompute, training, analysis, and reporting scripts |
| [api_reference.md](reference/api_reference.md) | REST API documentation for backend routes and contracts |
| [fine_tuning_run_matrix.md](reference/fine_tuning_run_matrix.md) | Active-experiment flow, experiment artifacts, and report-generation outputs |
| [metrics_methodology.md](reference/metrics_methodology.md) | IoU, Coverage, thresholding, and continuous-metric methodology |
| [per_head_methodology.md](reference/per_head_methodology.md) | Q3 per-head extraction rules, storage layout, and interpretation limits |
| [attention_heatmap_implementation.md](reference/attention_heatmap_implementation.md) | Heatmap implementation design and model-specific visualization notes |

### [`research/`](research/) — Background and Analysis Notes
Research context, novelty framing, and supporting analysis notes.

| Document | Description |
|----------|-------------|
| [attention_methods.md](research/attention_methods.md) | Attention visualization methods, interpretation, and model-specific caveats |
| [finetuning_results.md](research/finetuning_results.md) | Research notes on fine-tuning outcomes and interpretation |
| [claude_novelty_check.md](research/claude_novelty_check.md) | Novelty and related-work notes across the project questions |

### [`enhancements/`](enhancements/) — Scoped Feature Notes
Current implementation notes and follow-up directions for larger feature areas.

| Document | Description |
|----------|-------------|
| [per_attention_head.md](enhancements/per_attention_head.md) | Q3 product framing, scope, and study guidance for per-head analysis |
| [fine_tuning_methods.md](enhancements/fine_tuning_methods.md) | Fine-tuning strategy rationale, methodology, and interpretation notes |
| [sparse_annotation_bias.md](enhancements/sparse_annotation_bias.md) | Annotation sparsity limitations and mitigation guidance |

## Placement Guide

- Background research or related-work synthesis belongs in `research/`.
- Stable workflow, API, artifact, or methodology contracts belong in `reference/`.
- High-level project framing and milestone-oriented documents belong in `core/`.
- Larger feature notes and scoped follow-up directions belong in `enhancements/`.

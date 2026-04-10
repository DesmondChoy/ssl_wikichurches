# Documentation

This directory complements the root [`README.md`](../README.md) with workflow guides, API and methodology references, and research notes for the SSL WikiChurches analysis app. The project compares 7 vision models against expert-annotated architectural features in the 139-image WikiChurches subset with 631 bounding boxes.

Current model keys: `dinov2`, `dinov3`, `mae`, `clip`, `siglip`, `siglip2`, `resnet50`.

## Start Here

| Document | Best for |
|----------|----------|
| [../README.md](../README.md) | Setup, cache generation, fine-tuning commands, and current app routes |
| [user_guide.md](user_guide.md) | Gallery, Compare, Dashboard, Q2, Image Detail, and advanced Q3 workflows |
| [reference/api_reference.md](reference/api_reference.md) | Backend routes, query parameters, payloads, and response contracts |

## Structure

### [`core/`](core/) — Project Foundation
Stable documents defining the vision, requirements, and implementation roadmap.

| Document | Description |
|----------|-------------|
| [project_req.pdf](core/project_req.pdf) | Original course requirements |
| [project_proposal.md](core/project_proposal.md) | Research design, hypotheses, and methodology |
| [implementation_plan.md](core/implementation_plan.md) | Phase tracking, architecture, and progress status |

### [`research/`](research/) — Research & Analysis
Background research, literature analysis, and novelty assessments informing the project direction.

| Document | Description |
|----------|-------------|
| [attention_methods.md](research/attention_methods.md) | Attention visualization methods, metrics, and interpretation guide |
| [claude_novelty_check.md](research/claude_novelty_check.md) | Novelty assessment across research questions |

### [`reference/`](reference/) — Implementation Reference
Technical implementation details and model-specific considerations.

| Document | Description |
|----------|-------------|
| [attention_heatmap_implementation.md](reference/attention_heatmap_implementation.md) | Heatmap implementation architecture, model appropriateness, and two-phase design |
| [api_reference.md](reference/api_reference.md) | Complete REST API documentation for all API endpoints |
| [fine_tuning_run_matrix.md](reference/fine_tuning_run_matrix.md) | Canonical fine-tuning experiment-batch layout, run-matrix contract, and result storage map |
| [metrics_methodology.md](reference/metrics_methodology.md) | IoU, Coverage, thresholding methodology, worked examples, and academic context |
| [per_head_methodology.md](reference/per_head_methodology.md) | Q3 per-head extraction rules, metric pipeline, alternatives, and limitations |

### [`enhancements/`](enhancements/) — Ongoing Enhancements
Current scope notes and follow-up directions for larger feature areas.

| Document | Description | Status |
|----------|-------------|--------|
| [per_attention_head.md](enhancements/per_attention_head.md) | Per-head attention visualization | Mixed (implemented + follow-up work) |
| [fine_tuning_methods.md](enhancements/fine_tuning_methods.md) | Strategy-aware fine-tuning implementation notes, primary-vs-exploratory methodology, and remaining research directions | Mixed (implemented + future work) |

---

## Adding New Documents

- **Background research or literature analysis** → `research/`
- **Implementation-specific technical reference** → `reference/`
- **New feature proposals** → `enhancements/`
- **Major project milestones or reports** → `core/`

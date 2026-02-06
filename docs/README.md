# Documentation

SSL attention visualization platform comparing 6 vision models against expert-annotated architectural features (WikiChurches dataset, 139 images, 631 bounding boxes). See the [Project Proposal](core/project_proposal.md) for the full research design.

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
| [claude_novelty_check.md](research/claude_novelty_check.md) | Novelty assessment across all four research questions |

### [`reference/`](reference/) — Implementation Reference
Technical implementation details and model-specific considerations.

| Document | Description |
|----------|-------------|
| [attention_heatmap_implementation.md](reference/attention_heatmap_implementation.md) | Heatmap implementation architecture, model appropriateness, and two-phase design |

### [`enhancements/`](enhancements/) — Proposed Features
Future improvements not yet implemented. Documents here may be promoted to `core/` or merged into `reference/` once complete.

| Document | Description | Status |
|----------|-------------|--------|
| [per_attention_head.md](enhancements/per_attention_head.md) | Per-head attention visualization | Proposed |
| [fine_tuning_methods.md](enhancements/fine_tuning_methods.md) | Fine-tuning strategies for attention shift analysis | Proposed |

---

## Adding New Documents

- **Background research or literature analysis** → `research/`
- **Implementation-specific technical reference** → `reference/`
- **New feature proposals** → `enhancements/`
- **Major project milestones or reports** → `core/`

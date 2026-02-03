# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

# Agent Instructions

This project uses **bd** (beads) for issue tracking. Run `bd onboard` to get started.

## Quick Reference

```bash
bd ready              # Find available work
bd show <id>          # View issue details
bd update <id> --status in_progress  # Claim work
bd close <id>         # Complete work
bd sync               # Sync with git
```

## Development

```bash
./dev.sh              # Start the frontend React app
pytest                # Run tests
```

## Landing the Plane (Session Completion)

**When ending a work session**, you MUST complete ALL steps below. Work is NOT complete until `git push` succeeds.

**MANDATORY WORKFLOW:**

1. **File issues for remaining work** - Create issues for anything that needs follow-up
2. **Run tests before any commit** - `pytest` must pass before committing code changes
3. **Run quality gates** (if code changed) - Linters, builds
4. **Update issue status** - Close finished work, update in-progress items
5. **PUSH TO REMOTE** - This is MANDATORY:
   ```bash
   git pull --rebase
   bd sync
   git push
   git status  # MUST show "up to date with origin"
   ```
6. **Clean up** - Clear stashes, prune remote branches
7. **Verify** - All changes committed AND pushed
8. **Hand off** - Provide context for next session

**CRITICAL RULES:**
- Work is NOT complete until `git push` succeeds
- NEVER stop before pushing - that leaves work stranded locally
- NEVER say "ready to push when you are" - YOU must push
- If push fails, resolve and retry until it succeeds

---

## Architecture Overview

SSL attention visualization platform comparing 6 vision models against expert-annotated architectural features (WikiChurches dataset, 139 images, 631 bounding boxes).

### Data Flow

```
Precompute (one-time) → HDF5/PNG cache → FastAPI backend → React frontend
```

### Key Directories

| Directory | Purpose |
|-----------|---------|
| `src/ssl_attention/` | Core library: models, attention extraction, metrics, cache, visualization |
| `app/backend/` | FastAPI server (routers: images, attention, metrics, comparison) |
| `app/frontend/` | React + Vite + Tailwind |
| `app/precompute/` | Batch cache generation scripts |

### Models & Attention Methods

| Model | Methods | Notes |
|-------|---------|-------|
| DINOv2 | CLS, Rollout | 14×14 patches, 4 register tokens |
| DINOv3, MAE, CLIP | CLS, Rollout | 16×16 patches |
| SigLIP | Mean | No CLS token |
| ResNet-50 | Grad-CAM | CNN baseline |

### Key Patterns

- **Lazy loading**: Models loaded via `get_model()` with LRU cache (max 2 in GPU memory)
- **Protocol-based**: All models implement `VisionBackbone` protocol returning `ModelOutput`
- **Precompute + serve**: Heavy inference offline, fast cache reads at runtime

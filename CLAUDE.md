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

## Workflow Orchestration

### 1. Plan Node Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately - don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity

### 2. Subagent Strategy
- Use subagents liberally to keep main context window clean
- Offload research, exploration, and parallel analysis to subagents
- For complex problems, throw more compute at it via subagents
- One task per subagent for focused execution

### 3. Self-Improvement Loop
- After ANY correction from the user: update `tasks/lessons.md` with the pattern
- Write rules for yourself that prevent the same mistake
- Ruthlessly iterate on these lessons until mistake rate drops
- Review lessons at session start for relevant project

### 4. Verification Before Done
- Never mark a task complete without proving it works
- Diff behavior between main and your changes when relevant
- Ask yourself: "Would a staff engineer approve this?"
- Run tests, check logs, demonstrate correctness

### 5. Demand Elegance (Balanced)
- For non-trivial changes: pause and ask "is there a more elegant way?"
- If a fix feels hacky: "Knowing everything I know now, implement the elegant solution"
- Skip this for simple, obvious fixes - don't over-engineer
- Challenge your own work before presenting it

### 6. Autonomous Bug Fixing
- When given a bug report: just fix it. Don't ask for hand-holding
- Point at logs, errors, failing tests - then resolve them
- Zero context switching required from the user
- Go fix failing CI tests without being told how

---

## Task Management

1. **Plan First**: Write plan to `tasks/todo.md` with checkable items
2. **Verify Plan**: Check in before starting implementation
3. **Track Progress**: Mark items complete as you go
4. **Explain Changes**: High-level summary at each step
5. **Document Results**: Add review section to `tasks/todo.md`
6. **Capture Lessons**: Update `tasks/lessons.md` after corrections

---

## Core Principles

- **Simplicity First**: Make every change as simple as possible. Impact minimal code.
- **No Laziness**: Find root causes. No temporary fixes. Senior developer standards.
- **Minimal Impact**: Changes should only touch what's necessary. Avoid introducing bugs.

---

## Architecture Overview

SSL attention visualization platform comparing 7 vision models against expert-annotated architectural features (WikiChurches dataset, 139 images, 631 bounding boxes).

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
| SigLIP 2 | Mean | No CLS token |
| ResNet-50 | Grad-CAM | CNN baseline |

### Key Patterns

- **Lazy loading**: Models loaded via `get_model()` with LRU cache (max 2 in GPU memory)
- **Protocol-based**: All models implement `VisionBackbone` protocol returning `ModelOutput`
- **Precompute + serve**: Heavy inference offline, fast cache reads at runtime

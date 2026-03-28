# AGENTS.md

## Scope
- Scope applies to the entire `ssl_wikichurches` repository unless a nested `AGENTS.md` is present.
- Only `AGENTS.md` files at repository and repository subdirectory levels should be used for instruction overrides.

## Primary Tooling
- Use `uv` for Python environment and dependency management.
- Run Python commands from repository root unless a specific file lives in a deeper subdirectory.
- Prefer existing scripts and documented workflows in `README.md` and in scoped subdirectory docs.

## Documentation Source of Truth
- Treat the root `README.md`, `docs/reference/`, `docs/core/project_proposal.md`, `docs/core/implementation_plan.md`, and the live code as the current source of truth.
- Do NOT use `docs/archive/**` for present-day answers or implementation decisions unless the user explicitly asks for historical context.
- There is no current teammate onboarding document in this repo. Do not recreate or cite one for present-day guidance unless the user explicitly asks for historical context.

## Git Workspace Safety
- Do NOT use git worktrees.
- Work only in the main working directory.
- Stay on the current branch by default.
- Do NOT create or switch to a new branch unless the user explicitly tells you to.
- If branch isolation seems safer, ask first instead of deciding unilaterally.

## Build and Test Defaults
- After code changes, run at least targeted tests matching the touched module.
- For Python changes, use `uv run pytest` as the default test runner.
- For Python style checks, use `uv run ruff check .`.
- For static typing checks, use `uv run mypy`.
- For frontend changes under `app/frontend`, use `npm install` (or your project lockfile workflow), `npm run lint`, and `npm run build` from `app/frontend`.

## Issue Tracking (`bd`)
Use beads for task tracking:

- `bd list`
- `bd ready`
- `bd show <id>`
- `bd update <id> --status in_progress`
- `bd close <id>`
- `bd sync`

Before starting work:

- Run `bd list` or `bd ready` to find relevant existing work.
- If the work maps to an existing issue, use that issue ID.
- If no matching issue exists, create a bead before starting implementation.

During implementation:

- Create or reopen tracking artifacts for meaningful work, including follow-up tasks.
- Reference the bead or issue ID in commit messages when relevant.

After completing work:

- Close finished beads with a reason via `bd close <id> -r "reason"`.
- Optionally use `bd close <id> --suggest-next` to surface newly unblocked follow-up work.

## Quality Gate (Before Finishing)
Before finishing work or preparing to commit, run all relevant checks for touched areas:

1. `uv run ruff check .`
2. `uv run mypy`
3. `uv run pytest` (or targeted `pytest` scope for changed modules)
4. Review complete changed files, not only diffs, before committing.
5. For frontend changes: `cd app/frontend && npm run lint` and `cd app/frontend && npm run build`
6. Run the quality skill check from [`.claude/skills/quality/SKILL.md`](./.claude/skills/quality/SKILL.md) (or trigger your local equivalent, e.g., `/quality`) before committing.
7. Remove dead/debug code and verify no regression in changed flows.

## Code and Data Hygiene
- Keep changes scoped to the task at hand.
- Prefer explicit, typed interfaces and avoid adding duplicated logic.
- Validate assumptions with tests when behavior changes.
- Avoid editing generated or cache artifacts unless required by the task.
- Do not hardcode absolute dataset or cache paths.

## Test Failure Policy
- If a test fails due to environment constraints, document the failure and reproduce command with exact error output.
- If external services are unavailable, keep behavior deterministic with clear fallback or failure messaging.

## Collaboration Signals
- Include a short summary of assumptions, chosen approach, and risk areas in handoff notes.
- Keep documentation updates aligned with behavioral changes where appropriate.
- If ambiguity is low-risk and non-blocking, proceed with explicit assumptions and note them.
- If ambiguity affects correctness or design direction, ask one concise clarifying question before continuing.

## Communication Clarity

- **General clarity rule**: Write for immediate understanding on first read. Prefer plain, everyday language, short sentences, and concrete wording over dense or abstract phrasing.
- **Anti-jargon rule**: Do not make the user decode internal shorthand, umbrella terms, or technical jargon when a direct phrase would work. If a technical term is necessary, define it in plain English the first time you use it.

## Workflow Orchestration

### 1. Plan Node Default
- Enter plan mode for ANY non-trivial task (3+ steps or architectural decisions)
- If something goes sideways, STOP and re-plan immediately - don't keep pushing
- Use plan mode for verification steps, not just building
- Write detailed specs upfront to reduce ambiguity
- Before implementation, identify and flag the 3 most critical or technical aspects of the plan
- Use available user-input tools to ask whether the user wants to go deeper on any of them; if yes, keep narrowing to the exact scoped implementation areas they want to understand before coding

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

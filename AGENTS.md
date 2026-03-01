# AGENTS.md

## Scope
- Scope applies to the entire `ssl_wikichurches` repository unless a nested `AGENTS.md` is present.
- Only `AGENTS.md` files at repository and repository subdirectory levels should be used for instruction overrides.

## Primary Tooling
- Use `uv` for Python environment and dependency management.
- Run Python commands from repository root unless a specific file lives in a deeper subdirectory.
- Prefer existing scripts and documented workflows in `README.md` and in scoped subdirectory docs.

## Build and Test Defaults
- After code changes, run at least targeted tests matching the touched module.
- For Python changes, use `uv run pytest` as the default test runner.
- For Python style checks, use `uv run ruff check .`.
- For static typing checks, use `uv run mypy`.
- For frontend changes under `app/frontend`, use `npm install` (or your project lockfile workflow), `npm run lint`, and `npm run build` from `app/frontend`.

## Issue Tracking (`bd`)
Use beads for task tracking:

- `bd ready`
- `bd show <id>`
- `bd update <id> --status in_progress`
- `bd close <id>`
- `bd sync`

Create or reopen tracking artifacts for meaningful work, including follow-up tasks.

## Quality Gate (Before Finishing)
Before finishing work or preparing to commit, run all relevant checks for touched areas:

1. `uv run ruff check .`
2. `uv run mypy`
3. `uv run pytest` (or targeted `pytest` scope for changed modules)
4. For frontend changes: `cd app/frontend && npm run lint` and `cd app/frontend && npm run build`
5. Run the quality skill check from [`.claude/skills/quality/SKILL.md`](./.claude/skills/quality/SKILL.md) (or trigger your local equivalent, e.g., `/quality`) before committing.
6. Remove dead/debug code and verify no regression in changed flows.

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

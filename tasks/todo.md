# Task Todo

## Current Task

- [x] Add the concise technical-focus check-in instruction to `AGENTS.md`
- [x] Mirror the same instruction into `CLAUDE.md`
- [x] Review the diff and run the relevant documentation-session checks
- [ ] Commit and push the update

## Review

- Added the concise pre-implementation "top 3 technical aspects" guidance to both top-level agent instruction files.
- Preserved the follow-up narrowing step so plan discussions can drill into the exact scoped implementation areas the user cares about.
- Verified the documentation-session quality gate with `uv run ruff check .` and `uv run mypy`.

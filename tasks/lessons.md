# Lessons

## 2026-03-10

- Pattern: When a user asks to update repository agent instructions, check all top-level instruction files that serve similar purposes before closing the task.
- Rule: If both `AGENTS.md` and `CLAUDE.md` exist, assume instruction changes likely need to be mirrored unless the user explicitly scopes the request to one file.

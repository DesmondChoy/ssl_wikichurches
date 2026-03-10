# Lessons

## 2026-03-10

- Pattern: When a user asks to update repository agent instructions, check all top-level instruction files that serve similar purposes before closing the task.
- Rule: If both `AGENTS.md` and `CLAUDE.md` exist, assume instruction changes likely need to be mirrored unless the user explicitly scopes the request to one file.
- Pattern: When a user refines planning guidance, preserve the full clarification loop, including any follow-up deep-dive prompts that should happen before implementation.
- Rule: For instruction updates about plan discussions, capture both the initial "top 3 technical aspects" prompt and the follow-up narrowing step so the behavior is explicit and teachable.

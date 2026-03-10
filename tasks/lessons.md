# Lessons

## 2026-03-10

- Pattern: When a user asks to update repository agent instructions, check all top-level instruction files that serve similar purposes before closing the task.
- Rule: If both `AGENTS.md` and `CLAUDE.md` exist, assume instruction changes likely need to be mirrored unless the user explicitly scopes the request to one file.
- Pattern: When a user refines planning guidance, preserve the full clarification loop, including any follow-up deep-dive prompts that should happen before implementation.
- Rule: For instruction updates about plan discussions, capture both the initial "top 3 technical aspects" prompt and the follow-up narrowing step so the behavior is explicit and teachable.
- Pattern: When a verification pass uncovers a concrete regression in a touched flow, treat it as an active bug-fix obligation rather than stopping at documentation.
- Rule: If I can reproduce a regression locally and it falls within the current task scope, I should patch it immediately, then rerun the affected verification before reporting back.
- Pattern: When translating a user's UI idea into a planning/spec issue, avoid adding extra product requirements that were not asked for unless they are true blockers.
- Rule: For planning tickets, keep optional considerations like mobile optimization or CTA relocation out of the core requirements unless the user asked for them or they are necessary to make the flow coherent; otherwise frame them as open questions, not assumptions.

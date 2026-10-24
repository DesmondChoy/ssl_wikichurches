# AGENTS.md

## Scope
- Applies to all work under `app/` (backend and frontend).
- Overrides root-level instructions for app-stack-specific workflows.

## Backend (FastAPI)
- Run API/runtime checks relevant to edited backend files with:
  - `uv run pytest tests/test_services tests/test_backend`
- Keep route, schema, and service behavior changes covered by tests.
- Preserve request/response compatibility for existing frontend integrations.

## Frontend (React + Vite)
- For frontend changes, run:
  - `cd app/frontend && npm install`
  - `cd app/frontend && npm run lint`
  - `cd app/frontend && npm run build`
- Keep API calls typed and tolerant of backend response shape changes.
- Do not commit node artifacts (`node_modules`, `dist`) or lockfiles unless intentionally changed by dependency updates.

## Cross-Stack Consistency
- Update shared constants and contracts together (backend route shapes + frontend consumers).
- If a breaking contract change is introduced, include migration or compatibility handling and tests in both layers.

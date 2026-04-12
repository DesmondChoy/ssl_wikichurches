# Frontend Notes

This directory contains the React + TypeScript + Vite frontend for the SSL WikiChurches analysis app.

Use the root [`README.md`](../../README.md) for setup, app startup, and project-wide workflow guidance. Use [`docs/user_guide.md`](../../docs/user_guide.md) for the product walkthroughs and [`docs/reference/api_reference.md`](../../docs/reference/api_reference.md) for backend contracts.

## Current Pages

The primary page components live under `src/pages/`:

- `Home` for the Gallery route (`/`)
- `ImageDetail` for `/image/:imageId`
- `Compare` for `/compare`
- `Dashboard` for `/dashboard`
- `Q2` for `/q2`
- `Q3` for `/q3`

Q3 follows a Dashboard-first flow: discover dataset-level patterns on `Dashboard`, inspect one image on `ImageDetail`, and use `/q3` for aligned side-by-side model comparisons.

## Frontend Scripts

Run these commands from `app/frontend/`:

| Script | Command | Purpose |
|--------|---------|---------|
| Development server | `npm run dev` | Start the Vite dev server |
| Production build | `npm run build` | Run `tsc -b` and build the Vite bundle |
| Lint | `npm run lint` | Run ESLint across the frontend source tree |
| Preview | `npm run preview` | Serve the built app locally for preview |
| E2E tests | `npm run test:e2e` | Run Playwright end-to-end tests |

## Working Notes

- Keep this file lightweight and frontend-specific.
- Keep shared behavior, route semantics, and current product language in sync with the root docs and live code.
- Use `app/AGENTS.md`, the repo `AGENTS.md`, and `CLAUDE.md` for implementation guidance.

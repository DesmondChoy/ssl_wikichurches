# Task Todo

## Current Task

- [x] Tracker: create and sync Beads task `ssl_wikichurches-7gi` for MSE v1 continuous alignment rollout
- [x] Backend metrics core: add Gaussian GT + MSE helpers under `src/ssl_attention/metrics/`
- [x] Precompute/database: store per-image `mse` plus aggregate `mean_mse`/`std_mse`/`median_mse`, with safe in-place migration for existing DBs
- [x] Backend API/contracts: expose `mse` on image/bbox/comparison payloads and add `metric=iou|mse` support for leaderboard/progression/summary endpoints
- [x] Frontend: surface MSE in image detail and comparison views, plus add dashboard metric selector for leaderboard/layer progression
- [x] Tests: cover metric math, DB migration, API/service behavior, and updated frontend-facing contracts
- [x] Verification: run relevant Python tests/checks, frontend lint/build, and local quality-skill equivalent

## Plan Check-In

- Chosen approach: add MSE as an additive metric alongside existing IoU/coverage, reuse the current 224x224 cached attention heatmaps, and keep style/feature breakdowns IoU-only in v1.
- Critical aspect 1: ground-truth generation must match the plan exactly, including anisotropic Gaussian defaults, pixelwise `max` soft-union across bboxes, `[0, 1]` normalization/clamping, and stable empty-annotation behavior.
- Critical aspect 2: the percentile-keyed cache/database design needs careful migration and write semantics so threshold-free MSE is stored once per `(model, layer, method, image)` but remains queryable from percentile-based tables without breaking existing DBs.
- Critical aspect 3: dashboard endpoints and frontend types currently assume IoU-specific field names and descending rankings, so the contract changes need to stay localized and preserve IoU-only style/feature panels.
- Verification note: the user already provided the target behavior and scope, so implementation will proceed on that basis and validation will focus on proving those exact contract/ordering rules.

## Review

- Implemented Gaussian soft-target generation and threshold-free `compute_mse` / `compute_image_mse` in `src/ssl_attention/metrics/continuous.py`, with anisotropic bbox-derived sigmas, pixelwise-`max` soft union, `[0, 1]` normalization/clamping, and empty-annotation handling.
- Extended `app/precompute/generate_metrics_cache.py` so existing SQLite caches gain `image_metrics.mse` plus `aggregate_metrics.mean_mse/std_mse/median_mse` via `PRAGMA table_info` + `ALTER TABLE`, and existing percentile rows are backfilled instead of skipped when `mse` is missing.
- Updated backend per-image payloads to include `mse`, added single-bbox Gaussian MSE on `/metrics/{image_id}/bbox/{bbox_index}`, and generalized leaderboard/progression/all-models-summary responses to metric-aware `score`/`scores`/`best_score` fields with descending IoU vs ascending MSE semantics.
- Updated the frontend image detail, model comparison summary, dashboard leaderboard, and dashboard layer progression chart to consume and display MSE while keeping style and feature breakdown panels explicitly labeled as IoU-only.
- Added targeted tests for Gaussian GT + MSE math, DB migration, service ranking semantics, backend API payloads, and updated comparison mocks for the additive `mse` field.
- Verification completed:
  - `uv run pytest tests/test_metrics tests/test_services tests/test_backend`
  - `uv run ruff check .`
  - `uv run mypy`
  - `cd app/frontend && npm install`
  - `cd app/frontend && npm run lint`
  - `cd app/frontend && npm run build`
  - local quality-review equivalent via `git status --short`, `git diff HEAD --name-only`, and full-file review of touched modules
- Assumptions preserved:
  - MSE is computed in the existing cached heatmap space and intentionally repeats across percentile rows.
  - Style and feature breakdowns remain IoU-only in v1.
  - `/metrics/summary` remains on its existing IoU-shaped summary cache; the dashboard metric toggle uses `/compare/all_models_summary`.
- Residual notes:
  - `npm install` reported 3 pre-existing frontend dependency vulnerabilities via audit output; dependency remediation was outside this rollout.
  - The frontend build still warns about a large bundle chunk, which this change does not alter.

## Follow-Up Verification

- [x] Tracker: update Beads task `ssl_wikichurches-66s` for UI/database verification
- [x] Launch backend and frontend locally for Playwright inspection
- [x] Spot-check at least two image detail pages and confirm MSE is visible in the UI
- [x] Compare displayed MSE values against `outputs/cache/metrics.db` for the same image/model/layer/method selection
- [x] Sanity-check layout and interaction state so nothing looks out of place in the touched surfaces

### Verification Notes

- Playwright verification used the live dev app at `http://127.0.0.1:5173` with the backend running on `http://127.0.0.1:8000`.
- Image detail page checks:
  - `Q18785543_wd0.jpg` showed `IoU 0.000`, `Coverage 1.4%`, `MSE 0.0090` for the default `dinov2 / cls / Layer 0 / Top 10%` selection, matching the rounded `image_metrics` row in `outputs/cache/metrics.db` for percentile `90`.
  - `Q2034923_wd0.jpg` showed `IoU 0.072`, `Coverage 16.5%`, `MSE 0.0240` for the same default selection, matching the rounded percentile-`90` database row; its `mse` stayed constant across all stored percentiles as intended.
- Interaction check:
  - Selecting the `Round Arch` bbox on `Q2034923_wd0.jpg` updated the displayed metric block and showed a different single-bbox `MSE 0.0134`, confirming the bbox-specific MSE path is wired through the UI.
- Comparison view check:
  - `/compare?image=Q2034923_wd0.jpg` displayed both `IoU` and `MSE` per model. The shown `dinov2` and `dinov3` values matched the rounded `image_metrics` rows for `layer0 / cls / percentile 90`.
- Issue found during UI sanity check:
  - `/dashboard` currently crashes into the route error boundary with `The style prop expects a mapping from style properties to values, not a string`.
  - The stack points into Recharts `BarRectangle`, and the likely trigger is the dashboard style-breakdown chart data using a `style` field name (`app/frontend/src/pages/Dashboard.tsx`) that collides with the SVG `style` prop when the bar rectangles are rendered.

## Dashboard Regression Fix

- [x] Tracker: reopen Beads task `ssl_wikichurches-66s` for the dashboard regression
- [x] Patch the dashboard crash with the minimal frontend fix
- [x] Re-run frontend verification for `/dashboard` in Playwright
- [x] Reconfirm MSE-related surfaces still behave after the patch

### Regression Fix Notes

- Fixed the dashboard crash by renaming the style-breakdown chart data field from `style` to `styleLabel` in `app/frontend/src/pages/Dashboard.tsx`, avoiding a Recharts/SVG prop collision during bar rendering.
- Frontend verification after the patch:
  - `cd app/frontend && npm run lint`
  - `cd app/frontend && npm run build`
  - Playwright check: `/dashboard` now renders successfully in both `IoU` and `MSE` modes with no console errors.
  - Playwright recheck: `/image/Q2034923_wd0.jpg` still shows `MSE (lower better)` with `0.0240`, preserving the previously verified MSE display.

## Playwright Skill Run

- [x] Tracker: create Beads task `ssl_wikichurches-fik` for the full `playwright-testing` skill run
- [x] Start the app using `./dev.sh` as required by the skill
- [x] Phase 1: Navigation & Layout
- [x] Phase 2: Gallery Page
- [x] Phase 3: Image Detail Page
- [x] Phase 4: Compare Page
- [x] Phase 5: Dashboard Page
- [x] Phase 6: Desktop Layout Verification
- [x] Phase 7: Error Handling
- [x] Phase 8: Performance
- [x] Record the detailed per-item results in `tasks/playwright_skill_report.md`

### Playwright Skill Notes

- Followed the skill’s stop-fix-restart-resume workflow for every bug uncovered during verification.
- Bugs fixed during the run:
  - `dev.sh` now launches the backend with `uv run uvicorn ...`, restoring the documented `./dev.sh` startup path.
  - Added a wildcard route and `NotFoundPage` so invalid routes no longer render a blank screen.
  - Gated image-derived metric queries behind successful image-detail fetches so invalid image IDs no longer leave the main panel blank.
  - Improved the gallery’s backend-offline copy so it no longer misleadingly reports zero images and zero boxes.
- Final verification completed:
  - Playwright recheck of `/invalid-route` renders the custom 404 page with recovery links.
  - Playwright recheck of `/image/nonexistent` resolves to the existing load-error fallback after the async request fails.
  - Playwright recheck of backend-offline gallery state shows the new offline summary and refresh guidance.
  - `cd app/frontend && npm install`
  - `cd app/frontend && npm run lint`
  - `cd app/frontend && npm run build`

## Metric Card Formatting

- [x] Tracker: create Beads task `ssl_wikichurches-e1x` for the image-detail metric card formatting tweak
- [x] Reduce the image-detail metric value font size without changing the surrounding layout
- [x] Standardize image-detail metric formatting so IoU and MSE use 3 decimals while coverage remains at 1 decimal place
- [x] Run the relevant frontend verification commands for the touched component

### Metric Card Formatting Notes

- Updated `app/frontend/src/components/metrics/IoUDisplay.tsx` to share formatted display strings for image-detail metrics:
  - `IoU`: 3 decimals
  - `Coverage`: 1 decimal place as a percentage
  - `MSE`: 3 decimals
- Reduced the image-detail metric value typography from the previous `text-2xl` treatment to `text-xl` with tighter tracking so the card stays more balanced.
- Verification completed:
  - `cd app/frontend && npm run lint`
  - `cd app/frontend && npm run build`
  - Playwright check on `/image/Q2034923_wd0.jpg` confirmed the rendered values now appear as `0.072`, `16.5%`, and `0.024`, with `20px` font size on the metric values.

## Image Detail Metric Chart Planning

- [x] Add the presentation framing sentence to repo documentation
- [x] Create P0 Beads issue `ssl_wikichurches-336` for the image-detail-only metric chart and layout refactor
- [x] Review the issue for missing implementation details before coding

### Planning Notes

- Added the presentation framing sentence to `docs/reference/metrics_methodology.md` so the repo explicitly distinguishes the attention heatmap from the metric-over-layer view.
- Created `ssl_wikichurches-336` as a P0 feature issue covering:
  - image-detail-only scope
  - moving View Settings above Annotations on the left
  - dedicating the right column to Metrics
  - replacing static metric cards with a synced layer-metrics line chart
  - metric toggles with all lines enabled by default
  - bbox-aware switching between union and per-bbox progression
  - likely need for a new per-image layer-progression API
- Important details called out in the issue before implementation:
  - remove the existing `Compare Models` CTA below metrics rather than relocating it
  - no separate mobile/responsive optimization track is required for this issue
  - clarify MSE directionality in the chart because lower is better
  - document that percentile changes affect IoU but intentionally leave Coverage and MSE unchanged for a fixed image/model/method
  - use a stable, data-driven y-axis ceiling for the selected metric series so low-valued trends stay readable without axis jitter during playback

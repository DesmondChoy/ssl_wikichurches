# SSL Attention App User Guide

This guide explains how to use the current app to investigate the three project questions:

- **Q1:** Do frozen SSL models attend to the same regions experts marked as diagnostically important?
- **Q2:** How does fine-tuning shift attention, and which strategy helps most?
- **Q3:** Do individual attention heads specialize for different architectural features?

It is written as an app guide, not a results report. For setup, cache generation, and experiment commands, use the root [`README.md`](../README.md).

## App Map

| Page | Route | Best for | How you usually get there |
| --- | --- | --- | --- |
| Gallery | `/` | Browse annotated images and open one example | Top navigation |
| Image Detail | `/image/:imageId` | Inspect one image, its overlays, annotations, and per-image metrics | Click an image or exemplar |
| Compare | `/compare` | Compare frozen models or model variants on one image | Top navigation or quick links |
| Dashboard | `/dashboard` | Overview analysis for Q1 and the main discovery surface for Q3 | Top navigation |
| Q2 | `/q2` | Strategy-aware fine-tuning summary | Dashboard quick action or direct URL |
| Q3 Advanced | `/q3` | Side-by-side comparison of two primary-study Q3 models | Dashboard Q3 action or direct URL |
| Q3 Report | `/q3-report` | Report-focused Q3 head ranking, feature matrix, and delta views | Q3 page action or direct URL |

The top navigation only links to **Gallery**, **Compare**, and **Dashboard**. `Q2`, `Q3 Advanced`, `Q3 Report`, and `Image Detail` are drill-down pages.

## Shared Concepts

- **Metric direction**
  IoU and Coverage are better when higher. MSE, KL, and EMD are better when lower.
- **Percentile**
  The percentile control stays visible across the app for consistency, but in the current implementation it only changes IoU-based thresholding. Coverage, MSE, KL, and EMD are treated as threshold-free.
- **Bounding boxes**
  Clicking a bounding box can do different things depending on the page:
  - In **Image Detail** main tab, it highlights a feature and scopes the contextual metrics view.
  - In **Image Detail Q3** feature-similarity mode and on **Compare**, it becomes the similarity query.
- **Q3 scope**
  The primary Q3 workflow focuses on `dinov2`, `dinov3`, `mae`, and `clip`. `Frozen`, `LoRA`, and `Full Fine-tune` are the headline variants. `Linear Probe` remains available as a control.
- **Active experiment**
  Q2 summary views and any compare-page experiment context read from `outputs/results/active_experiment.json`, not from a hardcoded top-level result file.
- **Context in the URL**
  The app keeps a lot of state in URL parameters, especially for Q2 and Q3. That makes it easier to reopen a specific comparison or share a drill-down state.

## Page Reference

### Gallery (`/`)

Use the Gallery when you want to start from the dataset itself.

- Filter images by architectural style.
- Search image filenames directly.
- Each card shows the image, image ID, style tags, and bounding-box count.
- Clicking a card opens **Image Detail** for that image.

### Image Detail (`/image/:imageId`)

This page has two tabs: **Image Detail** and **Q3**.

#### Image Detail tab

This is the main single-image inspection view.

- **Left column**
  Model, attention method, optional head selector, layer, percentile, bounding-box toggle, and overlay appearance controls.
- **Center column**
  The attention viewer, layer playback controls, and the annotations card.
- **Right column**
  A metrics chart showing layer-by-layer alignment for the current image, model, and method.

Important behavior:

- The right-hand panel is a **metrics progression chart**, not a feature-breakdown panel.
- Clicking a bounding box on the image or in the annotations list keeps that feature selected as context for the current image.
- Head selection only appears when the current model and method support per-head attention.

#### Q3 tab

Use this tab after identifying an interesting head or head-feature pattern in **Dashboard Q3**.

- **Q3 controls**
  Model, variant, rank-by metric, top-head strip, expandable ranked-head gallery, layer slider, and bounding-box toggle.
- **Interpretation mode**
  Switch between **Head Attention** and **Feature Similarity**.
- **Current drill-down state**
  The page keeps the Q3 model, variant, layer, head, and optional feature context loaded.
- **Reset action**
  Use **Q3 defaults** to return to the canonical starting state for the Q3 drill-down workflow.

Important behavior:

- `Linear Probe` is available here, but it is treated as a **control**, not a headline Q3 condition.
- The Q3 tab is a drill-down surface. The main discovery step happens on **Dashboard Q3**.

### Compare (`/compare`)

Type an image filename, pick one from the image list, or open a URL with the canonical `image` query parameter. Older links that use `image_id` are normalized to `image`. After the image is selected, the page supports a frozen-model comparison flow and a generalized variant-comparison flow.

#### Model vs Model

Use this when you want to compare two frozen models on the same image.

- Choose two models.
- If both models support the same selected method, the page uses that shared method.
- If not, each panel falls back to the model's default method.
- Clicking a bounding box shows feature-similarity heatmaps side by side and updates the scope of the metrics panel.

#### Variant vs Variant

Use this when you want to compare any two states of the same base model.

- Choose one model.
- Choose any two variants from `Frozen`, `Linear Probe`, `LoRA`, and `Full Fine-tune`.
- Switch between **Side by side** and **Slider** view.
- When one side is `Frozen` and the other is adapted, a **Shift map** view becomes available and shows `adapted - frozen` directly on the image.
- Use the layer controls to inspect how the comparison changes through the network.
- Click a bounding box to switch from global overlays to bbox-conditioned similarity overlays.

#### Frozen vs Fine-tuned

This is the most common variant comparison preset.

- Choose one model.
- Hold one side on `Frozen`.
- Compare it against `Linear Probe`, `LoRA`, or `Full Fine-tune`.
- Use **Shift map** when you want a single diverging view of where attention increased or decreased after fine-tuning.
- In **Shift map**, the underlying photo is shown in grayscale and dimmed so the red/blue shift colors are easier to read.
- Use the same layer and bbox-conditioned similarity tools as the broader variant workflow.

Important behavior:

- The page supports **any pairwise variant comparison**, and Frozen vs Fine-tuned is one focused case of that broader flow.
- **Side by side** and **Slider** support the broader Variant vs Variant workflow. **Shift map** is only defined for frozen-vs-adapted pairs because it computes a signed change from the frozen baseline.
- When Q2 summary data is available for the selected model and metric, the page can show an **experiment summary** card sourced from the active Q2 analysis.
- The shift-map view is always computed from cached numeric heatmaps, so the selected metric and percentile do not change its colors. They still affect the summary tables and feature-local metrics below the main image.
- A sparse or faint shift map is still a real result. It usually means the adapted model stayed close to the frozen baseline for that image and layer, not that the viewer failed.
- Use **Side by side** or **Slider** when you want to compare the two full overlays. Use **Shift map** when you want the signed change directly.

### Dashboard (`/dashboard`)

This page has two tabs: **Overview** and **Q3**.

#### Overview tab

This is the best starting point for **Q1**.

- **Top controls**
  Metric, percentile, and ranking mode.
- **Model leaderboard**
  Ranks the frozen models for the selected metric.
- **Baseline references**
  For MSE, KL, and EMD, the leaderboard includes the documented baseline lines and legend from the current Q1 continuous-baseline reports.
- **Layer progression**
  Shows how each model's score changes across layers.
- **Style breakdown**
  Breaks the selected model's score down by architectural style.
- **Feature type breakdown**
  Search, sort, and page through the per-feature summary table.
- **Quick actions**
  Jump to Gallery, Compare, or Q2 with the current context partially preserved.

Important behavior:

- The ranking mode can use either each model's **default method** or its **best available method**.
- The Dashboard leaderboard covers the base model set. Strategy-specific fine-tuned variants are surfaced through **Q2** and **Compare**, not as a strategy-aware dashboard leaderboard.

#### Q3 tab

This is the main discovery surface for **Q3**.

- **Workflow context card**
  Shows whether the current selection is inside the primary Q3 workflow or a control condition.
- **Frozen-to-adapted delta panel**
  Compares how head rankings shift from Frozen to LoRA and Full.
- **Single-variant explorer**
  Includes the ranked-head table, head-by-feature heatmap, and inline exemplar loading.
- **Open advanced workspace**
  Sends the current model, variant, layer, metric, percentile, and optional focus into `/q3`.
- **Open report view**
  Sends the current Q3 context into `/q3-report` for the cleaner report-facing layouts.

Important behavior:

- Start here first for Q3. `/q3` is the advanced comparison workspace, not the main entry point.
- Use `/q3-report` when you want the focused report views used for screenshots and video walkthroughs.
- The inline exemplar flow is part of Dashboard Q3, so you can validate a head-feature pattern before leaving the page.

### Q3 Report (`/q3-report`)

Use this page when you want the Q3 evidence in the same focused format used by the academic report figures.

- **Head Ranking** shows the top heads for one model, variant, layer, metric, and percentile.
- **Head-Feature Matrix** connects a selected head to architectural feature labels.
- **Frozen-to-Adapted Delta** shows how the dominant head ranking changes from Frozen to LoRA and Full Fine-tune.

This page is intentionally narrower than Dashboard Q3 and `/q3`. It is built for explanation, screenshots, and reproducible report inspection.

### Q3 Advanced (`/q3`)

Use this page when you want to compare **two primary-study Q3 models side by side** under the same shared context.

- Shared controls keep the two panes aligned on model pair, variant, layer, metric, and percentile.
- Selecting a ranking row or heatmap cell syncs the same head or head-feature focus across both panes.
- A supporting adaptation panel below the two panes reuses the delta view for the primary model.

This page is especially useful once Dashboard Q3 has already narrowed the question to a specific head or feature.

### Q2 (`/q2`)

This is the main summary page for **Q2**.

- Filter by metric, percentile, model, and strategy.
- The header shows the analyzed layer and, when available, active-experiment provenance such as experiment ID, result scope, evaluation image count, and checkpoint-selection rule.
- The main table summarizes the shift from Frozen to each fine-tuning strategy.
- A second section shows pairwise cross-strategy comparisons.
- **Open Variant Compare** launches `/compare` with the current metric and analyzed layer as the starting point.

Important behavior:

- The Q2 page reads from the **active experiment** rather than a single hardcoded result file.
- The compare link opens the same analyzed layer used by the aggregate Q2 summary so the image-level inspection starts from the same context.

## Recommended Workflows

### Q1 Workflow

Use this path when the question is about frozen-model attention quality.

1. Start on **Dashboard Overview**.
2. Use the leaderboard and layer progression chart to identify promising models and layers.
3. Use style breakdown and feature type breakdown to see where a model is strong or weak.
4. Open an image in **Image Detail** to inspect the attention overlay and layer-by-layer metric history.
5. Use **Compare** when you want to see two models on the same image.

If you need report-ready summaries, the current generated artifacts are:

- `outputs/results/q1_continuous_baseline_summary.md`
- `outputs/results/q1_continuous_baseline_comparison.json`

### Q2 Workflow

Use this path when the question is about fine-tuning.

1. Start on **`/q2`**.
2. Read the header first so you know which active experiment and analyzed layer you are looking at.
3. Scan the model-by-strategy delta table.
4. Check the cross-strategy comparison section for direct strategy-vs-strategy differences.
5. Open **Variant Compare** to test whether the aggregate shift is visible on one concrete image.
6. Click one or more bounding boxes in Compare when you want a feature-local explanation instead of a whole-image overlay.

#### Q2 Image-Level Shift Check

Use this sub-workflow when the Q2 summary suggests an interesting strategy and you want to inspect one concrete image.

1. From **`/q2`**, click **Open Variant Compare**.
2. On **Compare**, keep `Comparison Type = Variant vs Variant`.
3. Choose one base model.
4. Set one side to `Frozen`.
5. Set the other side to `Linear Probe`, `LoRA`, or `Full Fine-tune`.
6. Click **Shift map**.
7. Use the layer slider to see whether the shift is strongest earlier or later in the network.
8. Click a bounding box when you want to connect the global shift map to the feature-local delta cards below it.

How to read the shift map:

- `Red` means the adapted model puts more attention there than the frozen model.
- `Blue` means the adapted model puts less attention there than the frozen model.
- The base photo is intentionally grayscale and dimmed in this view so the signed shift colors are easier to see.
- The selected metric and percentile stay visible for context, but they do not recolor the shift map itself.

How to interpret weak versus strong maps:

- A **large, structured red/blue pattern** means fine-tuning noticeably redistributed attention on that image at that layer.
- A **small or sparse patch** means the adapted model changed little relative to Frozen, or changed only in one local region.
- If the shift map looks weak but the lower feature-local delta is strong, the change may be specific to the selected feature rather than obvious at whole-image scale.
- If the shift map looks strong but the feature-local delta is weak or negative, attention may have moved, but not toward the selected expert-marked feature.

Recommended comparison order:

1. Start with **Shift map** to ask, "Did attention move at all, and where?"
2. Use the **feature-local delta** card to ask, "Did that movement help on the selected architectural feature?"
3. Switch to **Side by side** or **Slider** when you need the original two overlays for sanity-checking.

Useful provenance files for this flow are:

- `outputs/results/active_experiment.json`
- `outputs/results/experiments/<experiment_id>/q2_metrics_analysis.json`
- `outputs/results/experiments/<experiment_id>/q2_delta_iou_analysis.json`

The image-level Q2 delta endpoint used for focused examples is `GET /api/metrics/q2_image_deltas?model=<model>&strategy=<strategy>&percentile=<p>&top_k=<n>`.

### Q3 Workflow

Use this path when the question is about head specialization.

1. Start on **Dashboard Q3**.
2. Use the delta panel to see whether adaptation changes which heads look important.
3. Use the ranked-head table and head-by-feature heatmap to pick a candidate head or head-feature pair.
4. Open an inline exemplar, then move into **Image Detail Q3** for a closer qualitative check.
5. Switch between **Head Attention** and **Feature Similarity** to test whether the same head-feature story still holds visually.
6. Open **Q3 Advanced** only when you need the same Q3 focus state on two models at once.

## Tips

- **Start simple**
  If you are unsure where to begin, use `IoU` first for a straightforward overlap story, then switch to `KL` or `EMD` for a stricter distribution-level view.
- **Expect method differences**
  Not every model supports the same attention method, and not every selection supports per-head drill-down.
- **Watch for missing precompute artifacts**
  If a panel says data is unavailable, the usual cause is missing attention, heatmap, feature, or metrics caches for that route.
- **Use the Dashboard for discovery and Image Detail for inspection**
  The Dashboard is best for spotting patterns across models. Image Detail is best for validating what those patterns look like on a real example.
- **Remember that URL state is part of the workflow**
  Q2 and Q3 preserve a lot of context in the query string, so the URL is often the easiest way to revisit the exact same selection later.

# SSL Attention App User Guide

This guide covers how to navigate the React app and use it to investigate research questions, with focused walkthroughs for Q1, Q2, and Q3.

## App Structure at a Glance

| Page | Route | Purpose |
|------|-------|---------|
| **Gallery** | `/` | Browse 139 annotated church images, filter by architectural style |
| **Image Detail** | `/image/{imageId}` | Inspect one image's attention heatmaps, metrics, and annotations |
| **Compare** | `/compare` | Side-by-side comparison of models or fine-tuning variants |
| **Dashboard** | `/dashboard` | System-wide leaderboard, layer progression, and Q3 head analysis |
| **Q3 Advanced** | `/q3` | Side-by-side Q3 comparison workspace for two primary-study models |
| **Q2** | `/q2` | Fine-tuning strategy analysis (frozen vs adapted delta tables) |

The nav bar at the top links to **Gallery**, **Compare**, and **Dashboard**. Q2, Q3 Advanced, and Image Detail are reached by clicking into content from other pages.

---

## Page-by-Page Reference

### Gallery (`/`)

The landing page. Shows all 139 annotated images in a grid.

- **Style filter dropdown** (top-right): Filter by Romanesque, Gothic, Renaissance, or Baroque.
- **Image cards**: Each card shows the image, its ID, style tags, and bounding box count. Click a card to open Image Detail.

### Image Detail (`/image/{imageId}`)

Two tabs: **Image Detail** (default) and **Q3**.

#### Image Detail tab

Three columns:

**Left column (controls):**
- **Model selector**: Choose which model's attention to view (dinov2, clip, siglip, etc.).
- **Attention method**: CLS, rollout, mean, or gradcam (depends on model).
- **Attention head**: "All (Fused)" for the combined map, or select a specific head (0-11).
- **Layer slider**: Slide through transformer layers (L0-L11). Use the play button to animate.
- **Threshold**: Percentile-based filtering (top 10%, 20%, etc.). Only affects IoU-type metrics; threshold-free metrics (MSE, KL, EMD) are unaffected.
- **Show bounding boxes**: Toggle annotation overlay on/off.
- **Heatmap style**: Smooth Gradient, Squares, or Circles.
- **Opacity slider**: Control heatmap transparency.

**Center column (visualization):**
- The attention heatmap renders on the image. Click a bounding box on the image to select it (highlights green).
- Layer animation controls: play/pause, skip to first/prev/next/last layer.

**Right column (metrics):**
- Per-image metrics panel showing IoU, coverage, MSE, KL, EMD at the current layer.
- Feature breakdown and layer progression for the selected model.

#### Q3 tab

Deep per-head inspection. Accessed via `?tab=q3` or from the Dashboard Q3 tab.

- **Current drill-down summary**: Shows which model/variant context you brought in from Dashboard Q3, and whether the selected variant is in the primary Q3 scope or a control.
- **Model/variant/rank-by selectors** scope the analysis to the Q3-supported models and variants.
- **Top heads strip**: Shows the best-ranked heads for the current image context, with an option to expand into the full ranked-head gallery.
- **Interpretation mode toggle**: Switch between Head Attention (view one head's map) and Feature Similarity (click a bbox to query what the model considers similar).
- **Layer slider and bbox toggle**: Step through layers and choose whether annotation boxes stay visible while you inspect the exemplar image.
- **Q3 defaults action**: Resets the page back to the canonical Q3 starting state if you have wandered away from the intended drill-down path.

### Compare (`/compare`)

Requires selecting an image first from the image dropdown.

**Two modes** (toggled at the top):

- **Model vs Model**: Pick two models. Both show attention on the same image. Click a bounding box to see feature similarity heatmaps side-by-side.
- **Variant vs Variant**: Pick one model and any two variants (frozen, linear probe, LoRA, full). Includes a layer slider, a side-by-side/slider view toggle, and feature-local inspection by clicking a bounding box.

In Variant vs Variant mode, the page can also show an expandable experiment summary card sourced from the Q2 aggregate analysis so you can connect the image-level comparison to the batch-level result.

### Dashboard (`/dashboard`)

Two tabs: **Overview** and **Q3**.

#### Overview tab

**Top controls:**
- **Metric dropdown**: IoU, Coverage, MSE, KL, or EMD.
- **Threshold dropdown**: Percentile (only affects IoU; threshold-free metrics ignore this).
- **Ranking mode**: "Default method" ranks each model by its designated method (CLS for DINOv2, mean for SigLIP, etc.). "Best available" picks whichever method scores highest.

**Leaderboard card (left):**
- Horizontal bar chart ranking all 7 frozen models by the selected metric.
- For MSE, KL, and EMD: dashed reference lines show the four documented baselines (Random, Center Gaussian, Saliency Prior, Sobel Edge) with a legend below the chart showing exact values.
- Click a leaderboard row to select that model for the other dashboard panels.

**Layer Progression chart (right):**
- Line chart showing how each model's metric evolves from L0 to L11 (or the model's final layer).

**Style Breakdown (bottom-left):**
- Bar chart showing the selected model's metric broken down by architectural style.

**Feature Type Breakdown (bottom-right):**
- Table of all 90 feature types, sortable by metric score, count, or name.

**Quick Actions:**
- **Compare Models** and **Q2 Analysis** pass the current dashboard context through URL parameters.
- **Browse Images** returns to the Gallery without carrying the dashboard state.

#### Q3 tab

Head specialization explorer. This tab now has two layers:

- **Frozen-to-adapted head delta panel**: Compares how head rankings shift from Frozen to LoRA and Full for the current model/layer/metric context.
- **Single-variant explorer**: Shows the ranked-head table, the head-by-feature heatmap, and inline exemplar images. Clicking through drills down to the Image Detail Q3 tab with full context.
- **Open advanced workspace**: Launches `/q3` with the current model, variant, layer, metric, percentile, and optional head/feature focus carried into a side-by-side comparison page.

### Q3 Advanced (`/q3`)

Advanced pairwise comparison workspace for hypothesis H2.

- **Shared controls**: Keep the primary model, comparison model, variant, layer, metric, and percentile aligned across both panes.
- **Primary pane / Comparison pane**: Reuse the ranked-head table, head-by-feature heatmap, and inline exemplar drill-down from Dashboard Q3, but show two models side by side.
- **Shared focus state**: Selecting a ranking row or heatmap cell syncs the focused head or head-feature context through the URL so the same comparison target can be reopened or shared.
- **Supporting adaptation panel**: Reuses the frozen-to-adapted delta view for the primary model without replacing Dashboard Q3 as the main summary surface.

### Q2 (`/q2`)

**Top controls:**
- Metric, percentile, model (all or specific), and strategy (all, linear probe, LoRA, full).
- The header card also shows the analyzed layer and, when available, the active experiment ID, result scope, evaluation image count, and checkpoint-selection rule.

**Delta table:**
- Shows frozen mean, fine-tuned mean, delta, 95% CI, Cohen's d effect size, and statistical significance for each model/strategy pair.
- The table itself is textual rather than color-coded, so read the signed delta together with whether the selected metric treats higher or lower values as better.

**Cross-strategy comparisons:**
- Pairwise strategy comparisons (e.g., LoRA vs Full) with delta differences and significance.

**"Open Variant Compare" link:**
- Jumps to the Compare page with frozen vs the selected strategy pre-loaded.

---

## Q1 Investigation Walkthrough

**Q1: Do SSL models attend to the same features human experts consider diagnostic?**

This walkthrough shows how to use the app to answer Q1, starting from the high-level leaderboard and drilling down to individual images.

### Step 1: Start at the Dashboard Leaderboard

Go to `/dashboard`. The Overview tab is the Q1 home base.

**What to do:**
1. Set Metric to **MSE** (or KL, or EMD).
2. Keep Ranking on **Default method**.
3. Look at the leaderboard bar chart.

**What to look for:**
- The four dashed baseline reference lines divide the chart into zones. A model bar that falls entirely to the left of all four lines beats every naive baseline on that metric.
- On **MSE**: Most models beat all baselines. This tells you the intensity profiles roughly match the Gaussian targets, but is weak evidence on its own.
- Switch to **KL**: Now only **DINOv3** clears all four lines. Most other models cluster near or past the center/saliency baselines. This means their probability mass distribution is not much better than a simple center-bias heuristic.
- Switch to **EMD**: Again only **DINOv3** clears all baselines. SigLIP and SigLIP2 bars extend past the Random line, meaning their spatial transport cost is worse than chance.

**Key finding to note:** The cross-metric divergence. CLIP and SigLIP look strong on MSE but weak on KL and EMD. DINOv3 is the only model that consistently beats all baselines across all three continuous metrics.

### Step 2: Check IoU for the Threshold-Dependent Story

Switch Metric to **IoU**. The baselines disappear (IoU has a natural 0-1 scale).

**What to look for:**
- DINOv3 leads the IoU leaderboard too, confirming its strength is not metric-specific.
- Compare the model rankings between IoU and MSE. If rankings shift dramatically, that tells you the threshold choice matters for some models.
- Try different thresholds (Top 10%, Top 20%, etc.) and watch how rankings change. Models that are sensitive to threshold may have diffuse attention patterns.

### Step 3: Examine Layer Progression

Still on the Dashboard, look at the **Layer Progression chart** on the right.

**What to look for:**
- Which layers produce the best scores? For most ViT models, later layers (L9-L11) tend to be better, but this varies.
- MAE's progression on MSE often shows a distinctive pattern: high MSE at L0 (reconstruction-trained early layers attend broadly), dropping in later layers.
- ResNet-50 only has 4 layers (gradcam_layer0-3), so its progression is short.
- Cross-model divergence: Do all models improve at the same rate across layers, or do some peak earlier?

### Step 4: Break Down by Architectural Style

Click a model row in the leaderboard (e.g., DINOv3). Look at the **Style Breakdown chart** (bottom-left).

**What to look for:**
- Does the model perform equally well across Romanesque, Gothic, Renaissance, and Baroque?
- Uneven performance by style suggests the model's attention is biased toward certain architectural patterns.
- Compare styles across models: if DINOv3 is strong on Gothic but weak on Baroque, while CLIP shows the opposite, that's an interesting finding about how training objectives shape style sensitivity.

### Step 5: Examine Feature Types

Look at the **Feature Type Breakdown table** (bottom-right).

**What to look for:**
- Sort by the metric score. Which architectural features does the selected model attend to best? Which does it miss?
- Features with high counts but low scores are the most informative failures: the dataset has enough examples, but the model still doesn't attend well.
- Features with high scores but low counts should be treated cautiously: the model may look good on those features by chance.
- Compare feature rankings across models by switching the selected model in the leaderboard.

### Step 6: Drill Down to Individual Images

From the Dashboard, click **Browse Images** in Quick Actions (or go to `/`).

Pick an image that you expect to be interesting (e.g., one with many bounding boxes, or one from a style where a model performed poorly).

**On the Image Detail page:**
1. Set Model to a strong model (e.g., DINOv3) and a weak model (e.g., MAE).
2. Use the layer slider to animate from L0 to L11. Watch how the attention heatmap evolves.
3. Toggle bounding boxes on. Does the attention hotspot overlap with the expert annotations?
4. Try different thresholds to see how much of the attention mass is inside vs outside the boxes.

**What to look for:**
- Qualitative confirmation of the quantitative story. If DINOv3 has the best KL score, its heatmap at the best layer should visually concentrate on the annotated features.
- Attention "leaks": does the model attend to nearby but incorrect regions (e.g., the wall next to a window)? EMD captures this, but seeing it visually helps explain it.
- Layer evolution: at L0, attention is often diffuse or noisy. By L11, it should sharpen on semantic features if the model has learned them.

### Step 7: Side-by-Side Model Comparison

Go to `/compare`. Select the same image. Set comparison type to **Model vs Model**.

**What to do:**
1. Put DINOv3 on the left, CLIP on the right (or any pair you want to compare).
2. Click a bounding box on either side.
3. Both viewers now show feature similarity heatmaps for that bounding box.

**What to look for:**
- Does DINOv3 concentrate similarity tightly around the feature, while CLIP spreads it across the image? This would explain why CLIP has good MSE (right general area) but bad KL (mass in wrong places).
- Try multiple bounding boxes. Some features may be harder for one model than another.

### Step 8: Consult the Analysis Artifacts

The script-generated artifacts provide the structured Q1 findings:

- **`outputs/results/q1_continuous_baseline_summary.md`**: Report-ready markdown with headline findings, cross-metric divergences, per-metric tables, and per-model wrap-ups.
- **`outputs/results/q1_continuous_baseline_comparison.json`**: Machine-readable version with per-model pass/fail against each baseline, surprise tags, and cross-metric synthesis.

These artifacts contain the evidence. The app is where you explore and confirm it visually. The report is where you explain *why*.

---

## Q2 Investigation Walkthrough

**Q2: Does fine-tuning shift attention toward expert features, and which strategy helps most?**

This walkthrough starts from the aggregate Q2 summary, then moves into image-level comparison so you can connect the statistics to what the models actually changed.

### Step 1: Start at the Q2 Summary

Go to `/q2`.

**What to do:**
1. Start with Metric set to **IoU** if you want the most intuitive first pass.
2. Leave Model on **All Models** and Strategy on **All Strategies**.
3. Read the header card before the tables.

**What to look for:**
- The header tells you which layer the Q2 summary was computed at. That matters because the compare link will open at that same layer.
- If the page shows an active experiment ID, result scope, or checkpoint rule, note it. Those fields tell you exactly which fine-tuning batch produced the numbers you are reading.
- If you switch to **MSE**, **KL**, or **EMD**, remember that those are threshold-free metrics. The percentile control stays visible for consistency, but it is not driving those scores.

### Step 2: Read the Model × Strategy Delta Table

Stay on the main Q2 table.

**What to do:**
1. Scan one model at a time across Linear Probe, LoRA, and Full.
2. Then switch metrics and see whether the same strategy stays strongest.

**What to look for:**
- The signed **Delta** is the headline: positive is better for IoU and Coverage, while negative is better for MSE, KL, and EMD.
- The **95% CI** helps you judge stability. Wide intervals mean the apparent gain may be noisy.
- **Effect size** and **Significant** tell you whether a change is likely to be meaningful rather than just numerically different.
- A useful pattern to watch for is divergence by metric: a strategy may improve IoU while barely changing KL or EMD, which suggests sharper overlap without fully fixing the attention distribution.

### Step 3: Compare Strategies Directly

Scroll to **Cross-Strategy Paired Comparisons**.

**What to do:**
1. Focus on one model.
2. Compare pairs such as LoRA vs Full or Linear Probe vs LoRA.

**What to look for:**
- This section answers a different question from the delta table. Instead of asking whether a strategy beats Frozen, it asks whether one fine-tuning strategy beats another.
- Large pairwise differences with weak significance should be treated cautiously.
- If LoRA and Full look similar here, that suggests the simpler adaptation may already capture most of the attention shift.

### Step 4: Jump into Variant Compare

Click **Open Variant Compare** from the Q2 table card.

**What to do:**
1. Keep the pre-loaded model, metric, and layer for your first pass.
2. Compare Frozen on one side with the strategy you care about on the other.
3. Switch between **Side by side** and **Slider** view.

**What to look for:**
- The image-level view shows whether the aggregate delta reflects a visible shift toward the annotated regions.
- If the page shows an **Experiment summary** card, use it as context rather than the final verdict. The real check is whether the visual change on individual images matches the aggregate story.
- If Linear Probe is one of the compared variants, expect smaller changes. That is the control case where the backbone itself did not adapt.

### Step 5: Inspect Feature-Local Changes

Still on `/compare`, click a bounding box or use the feature chips below the viewer.

**What to do:**
1. Select one architectural feature on the image.
2. Read the local metrics and the feature-local delta.
3. Try a second feature on the same image.

**What to look for:**
- Some strategies improve attention on one feature type but not another. That is often more informative than the image-wide average.
- If the selected metric improves locally but the overlay still looks messy, switch to a second metric before drawing a conclusion.
- When bbox-conditioned similarity heatmaps appear, ask whether the fine-tuned variant is more tightly centered on the selected feature than Frozen.

### Step 6: Build the Q2 Conclusion

At this point you should be able to answer Q2 at two levels:

- **Aggregate level:** Which strategy moves the metric most, and is that shift stable enough to trust?
- **Qualitative level:** Do the variant overlays and feature-local comparisons show a clearer focus on expert-annotated regions?

The strongest Q2 conclusions are the ones where the table, the pairwise comparison, and the image-level inspection all point in the same direction.

---

## Q3 Investigation Walkthrough

**Q3: Do individual attention heads specialize for different architectural features?**

This walkthrough begins on Dashboard Q3, where you identify candidate specialized heads, and ends in Image Detail Q3, where you inspect one exemplar image closely.

### Step 1: Start on Dashboard Q3

Go to `/dashboard?tab=q3`.

**What to do:**
1. Start with one of the primary Q3 models.
2. Keep Variant on **Frozen** for your first pass.
3. Pick a metric that matches the kind of specialization you care about.

**What to look for:**
- The **Current Q3 workflow context** card tells you whether you are looking at a primary Q3 path or a control variant.
- If you are new to the page, start with **IoU** for a straightforward overlap story, then revisit with **KL** or **EMD** to see whether the same heads still look strong on stricter distribution-based metrics.
- The selected layer matters. A head that looks unremarkable early in the network may become highly specialized later.

### Step 2: Check the Frozen-to-Adapted Delta Panel

Stay near the top of Dashboard Q3.

**What to do:**
1. Compare the Frozen baseline against **LoRA** and **Full**.
2. Look for heads that move sharply up or down in rank.

**What to look for:**
- This panel tells you whether fine-tuning merely improves the same dominant heads or changes *which* heads matter.
- A promoted head with a meaningful score delta suggests that adaptation may be creating or strengthening specialization.
- If the same heads stay near the top across Frozen, LoRA, and Full, the specialization story is more about stable pretrained structure than fine-tuning.

### Step 3: Use the Ranking Table and Heatmap Together

Scroll to the single-variant explorer.

**What to do:**
1. Read the **Head Ranking** table to find strong heads overall.
2. Then inspect the **Head × Feature Heatmap** to see whether those heads are broadly good or only good for certain features.
3. Use the feature search box if you already care about a specific architectural part.

**What to look for:**
- A head that ranks well overall but lights up only a few feature rows may be genuinely specialized.
- A head that looks decent everywhere may be more general-purpose than feature-specific.
- The heatmap is often the clearest evidence for specialization: darker cells concentrated in a small slice of features are more interesting than a uniformly medium-dark column.

### Step 4: Open Representative Images

Click either **Inspect exemplar** in the ranking table or a specific heatmap cell.

**What to do:**
1. If you click a ranking row, you are asking for a representative image for that head overall.
2. If you click a heatmap cell, you are asking for a representative image for one head-feature pair.
3. In the exemplar panel, open one candidate in Image Detail Q3.

**What to look for:**
- Ranking-based exemplars help you answer, “What does this head usually attend to?”
- Cell-based exemplars help you answer, “What does this head do on *this feature type*?”
- If multiple exemplar images tell the same story, your specialization claim is much stronger.

### Step 5: Inspect the Exemplar in Image Detail Q3

After clicking **Open in Image Detail Q3**, stay inside the Q3 tab on the image page.

**What to do:**
1. Start in **Head Attention** mode.
2. Use the pre-loaded top-head context from Dashboard Q3.
3. Move the layer slider and compare **All (Fused)** with the selected head.

**What to look for:**
- Ask whether the selected head isolates a meaningful architectural part more clearly than the fused view.
- If the same head remains informative across nearby layers, that is a stronger pattern than a one-layer spike.
- The **Current drill-down** panel helps you keep track of which model, variant, and feature context you carried over from the dashboard.

### Step 6: Switch to Feature Similarity Mode

Stay on the same image and flip the interpretation toggle.

**What to do:**
1. Change from **Head Attention** to **Feature Similarity**.
2. Click the relevant bounding box on the image or in the annotations panel.

**What to look for:**
- This mode answers a slightly different question: not just where the head attends, but whether the selected feature pulls up semantically similar regions.
- If the similarity overlay stays tightly aligned with the chosen feature, that supports the idea that the representation around that head-feature context is coherent.
- If the similarity spills into unrelated regions, the head may be less specialized than the dashboard ranking first suggested.

### Step 7: Build the Q3 Conclusion

You have a strong Q3 result when all three layers line up:

- **Delta panel:** shows whether specialization changes under adaptation.
- **Ranking table + heatmap:** identifies which heads and features are linked.
- **Image Detail Q3:** confirms that the selected head-feature story is visible on real exemplar images.

If only one of those layers looks convincing, treat the result as a lead rather than a settled finding.

---

## Tips

- **Threshold-free metrics (MSE, KL, EMD)** ignore the percentile dropdown. The threshold only affects IoU and coverage.
- **"Lower is better"** for MSE, KL, and EMD. **"Higher is better"** for IoU and Coverage. The leaderboard card shows which direction applies.
- **Quick Actions only partially preserve context**: Compare and Q2 keep the current dashboard parameters, while Browse Images goes back to the plain Gallery route.
- **The layer progression chart** does not show baseline reference lines (by design). Baselines are dataset-level constants, not layer-varying, so they belong on the leaderboard where models are compared at their best layer.

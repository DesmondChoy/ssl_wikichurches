# SSL Attention App User Guide

This guide covers how to navigate the React app and use it to investigate research questions, with a focused walkthrough for Q1.

## App Structure at a Glance

| Page | Route | Purpose |
|------|-------|---------|
| **Gallery** | `/` | Browse 139 annotated church images, filter by architectural style |
| **Image Detail** | `/image/{imageId}` | Inspect one image's attention heatmaps, metrics, and annotations |
| **Compare** | `/compare` | Side-by-side comparison of models or fine-tuning variants |
| **Dashboard** | `/dashboard` | System-wide leaderboard, layer progression, and Q3 head analysis |
| **Q2** | `/q2` | Fine-tuning strategy analysis (frozen vs adapted delta tables) |

The nav bar at the top links to **Gallery**, **Compare**, and **Dashboard**. Q2 and Image Detail are reached by clicking into content from other pages.

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

- **Model/variant/metric selectors** scope the analysis to Q3 primary models.
- **Ranked heads**: Shows which heads align best with annotations. Click a head to view its attention.
- **Interpretation mode toggle**: Switch between Head Attention (view one head's map) and Feature Similarity (click a bbox to query what the model considers similar).

### Compare (`/compare`)

Requires selecting an image first from the image dropdown.

**Two modes** (toggled at the top):

- **Model vs Model**: Pick two models. Both show attention on the same image. Click a bounding box to see feature similarity heatmaps side-by-side.
- **Variant vs Variant**: Pick one model, two fine-tuning variants (frozen, linear probe, LoRA, full). Includes a layer slider to animate through layers and see how attention shifts between variants.

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
- Links to Gallery, Compare, Q2, with the current metric/model/layer context passed via URL parameters.

#### Q3 tab

Head specialization explorer. Shows a head-by-feature heatmap, ranked heads, and exemplar images. Clicking through drills down to the Image Detail Q3 tab with full context.

### Q2 (`/q2`)

**Top controls:**
- Metric, percentile, model (all or specific), and strategy (all, linear probe, LoRA, full).

**Delta table:**
- Shows frozen mean, fine-tuned mean, delta, 95% CI, Cohen's d effect size, and statistical significance for each model/strategy pair.
- Color-coded: green for improvement, red for decline.

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

## Tips

- **Threshold-free metrics (MSE, KL, EMD)** ignore the percentile dropdown. The threshold only affects IoU and coverage.
- **"Lower is better"** for MSE, KL, and EMD. **"Higher is better"** for IoU and Coverage. The leaderboard card shows which direction applies.
- **URL parameters are preserved** when navigating between pages via Quick Actions, so your metric/model context carries over.
- **The layer progression chart** does not show baseline reference lines (by design). Baselines are dataset-level constants, not layer-varying, so they belong on the leaderboard where models are compared at their best layer.

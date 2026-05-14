# 15-Minute Video Presentation Plan

## Submission Requirements

- **Format**: One MP4 file per team
- **Length**: 15 minutes
- **No in-person presentation** — the recorded video is the final submission
- Submitted as part of the final zip (alongside source code, report PDF, and LaTeX source files)

---

## Overall Approach

Screen-recorded presentation with voiceover, ending with a live app demo. The app demo is a strong differentiator — use it to ground abstract findings in concrete visuals.

Use the Appendix A easy/hard image pairs from the report as anchor examples throughout:
- **Easy (Gothic)**: Q1710328 — DINOv3 frozen IoU = 0.438, CLIP Δ IoU = +0.207
- **Easy (Romanesque)**: Q2034923 — DINOv3 frozen IoU = 0.403, CLIP Δ IoU = +0.135
- **Hard (Baroque)**: Q694252 — DINOv3 frozen IoU = 0.000, CLIP Δ IoU = −0.005
- **Hard (Renaissance)**: Q1424095 — DINOv3 frozen IoU = 0.002, CLIP Δ IoU = +0.002

---

## Segment-by-Segment Structure

### Segment 1 — Hook & Problem Framing (1.5 min)

Open with: *"A model predicts 'Gothic' correctly — but is it looking at the right thing?"*

- Show a church image with expert bounding boxes vs. a raw attention heatmap side by side
- Establish the core tension: accuracy ≠ expert-aligned evidence use
- One-sentence dataset overview: 139 images, 631 expert boxes, 7 models, 3 research questions

---

### Segment 2 — Dataset & Methodology (2.5 min)

- WikiChurches: fine-grained style labels + expert-annotated architectural features (arches, portals, pediments)
- 7 models across 4 paradigms: self-distillation (DINOv2, DINOv3), masked autoencoding (MAE), language-image contrastive (CLIP, SigLIP, SigLIP2), CNN baseline (ResNet-50)
- 5 alignment metrics — briefly explain the IoU vs. EMD distinction (overlap vs. distributional fit)
- Calibrated baselines (random, center Gaussian, saliency prior, Sobel edge) — explain *why*: raw scores are meaningless without reference points

---

### Segment 3 — Q1: Frozen Model Benchmark (2.5 min)

- Show the leaderboard table: DINOv3 leads on IoU, Coverage, KL, EMD
- Key interpretive point: SigLIP has the best MSE but fails EMD — one-metric reading is misleading
- Explain DINOv3's likely advantage: Gram anchoring preserves dense spatial structure during long SSL training
- DINOv3 is the only frozen model that clears all 4 naive baselines on all 3 continuous metrics
- **App demo (~30 sec)**: live Dashboard showing the KL leaderboard with baseline overlays

---

### Segment 4 — Q2: Fine-Tuning Effects on Attention (4 min)

- Linear Probe = zero attention change by construction (backbone stays frozen — useful sanity check)
- Show the multi-metric improvement heatmap: CLIP and MAE gain most; DINO stays flat
- **CLIP story**: IoU 0.018 → 0.074 (Cohen's d ≈ 1.0), but gains concentrate on Gothic/Romanesque only — linguistic grounding hypothesis (these styles have features densely described in English web text)
- **MAE story**: Renaissance pediment specialization — gains driven by compact, style-exclusive geometry (Triangular Pediment +0.080, Broken Pediment +0.055); most common Renaissance features (Pilaster, Belt Course) show *negative* Δ
- **DINO story**: near-zero Δ across all styles is a *feature*, not a failure — strong frozen spatial prior resists reorganization
- **Surprise cross-model finding**: DINOv3 frozen IoU predicts CLIP Δ IoU at Pearson r = +0.677 — models converge on the same structurally easy images; language-cluster ensemble adds no coverage on hard images; MAE is the only anti-correlated model (natural complementary partner)
- **App demo (~30 sec)**: Compare view showing a frozen vs. fine-tuned CLIP shift map on Q1710328 (Gothic portal)

---

### Segment 5 — Q3: Per-Head Specialization (2 min)

- Scope clearly: DINOv2, DINOv3, MAE, CLIP only (CLS-token models); SigLIP/SigLIP2 excluded (mean-attention proxy); ResNet-50 excluded (no attention heads)
- Language: *descriptive* analysis, not causal proof
- Do some heads align better with specific feature types (portals, windows, towers)?
- Does fine-tuning shift the dominant head set?
- Show Q3 dashboard or head-feature heatmap from the app

---

### Segment 6 — Live App Demo (1.5 min)

Walk through the app:
1. **Gallery** → click into an image detail
2. **Dashboard** → KL leaderboard with baseline overlays
3. **Compare** → frozen vs. fine-tuned shift map (use Q1710328 Gothic portal)
4. **Q2 tab** → improvement heatmap + preserve/enhance/destroy summary

Narration hook: *"All of this is backed by precomputed HDF5/SQLite cache — sub-second queries across 139 images × 7 models × 12 layers."*

---

### Segment 7 — Conclusions & Takeaways (1 min)

Three sentences, one per question:

- **Q1**: Frozen expert alignment exists but is model-family dependent — DINOv3 is uniquely compatible with expert-marked architectural evidence
- **Q2**: Fine-tuning's effect is mediated by three factors: spatial prior (DINO vs. rest), linguistic coverage (CLIP), and geometric discriminability (MAE)
- **Q3**: Some heads show descriptive specialization; dominant heads shift under adaptation

Practical implication: *model selection for domain adaptation shouldn't be guided by accuracy alone — match the pretraining prior to what the task's diagnostic evidence requires.*

---

## Time Budget

| Segment | Time |
|---|---|
| 1. Hook & framing | 1.5 min |
| 2. Dataset & methodology | 2.5 min |
| 3. Q1 results + demo | 2.5 min |
| 4. Q2 results + demo | 4.0 min |
| 5. Q3 results | 2.0 min |
| 6. Live app demo | 1.5 min |
| 7. Conclusion | 1.0 min |
| **Total** | **15 min** |

---

## Key Visual Assets to Prepare

| Asset | Source |
|---|---|
| Church image + expert boxes vs. attention heatmap | App image detail view |
| Q1 leaderboard table | `outputs/cache/metrics_summary.json` |
| Q1 KL dashboard screenshot | App Dashboard (Metric=KL, Ranking=Default method) |
| Q2 multi-metric improvement heatmap | `outputs/figures/02_all_metrics_improvement_heatmap.png` |
| Q2 preserve/enhance/destroy summary | `outputs/figures/07_preserve_enhance_destroy.png` |
| Q2 forest plot with CIs | `outputs/figures/08_forest_plot_ci.png` |
| Q2 per-style Δ IoU breakdown | `outputs/results/experiments/fine_tuning_primary_20260327/style_breakdown.png` |
| Q2 MAE per-feature Renaissance table | `outputs/results/experiments/fine_tuning_primary_20260327/feature_delta_iou_mae_full_renaissance.png` |
| Q2 cross-model correlation scatter | `outputs/results/experiments/fine_tuning_primary_20260327/model_correlation_scatter.png` |
| Q2 shift map example (app Compare view) | Q1710328 (Gothic, easy) |
| Q3 head-feature heatmap | App Q3 dashboard |
| Hard image examples | Q694252 (Baroque), Q1424095 (Renaissance) |

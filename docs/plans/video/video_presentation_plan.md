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

### Segment 1 — Hook, Problem Framing & Results Summary (2 min)

Structure this like the updated abstract: open with the motivating question, break the three research questions out explicitly, then deliver headline findings for all three before going into methodology.

**Hook**: *"A model predicts 'Gothic' correctly — but is it looking at the right thing?"*

- Show a church image with expert bounding boxes vs. a raw attention heatmap side by side
- Establish the core tension: accuracy ≠ expert-aligned evidence use
- Frame the three research questions explicitly (as numbered list, matching the abstract):
  1. How well do frozen models align with expert-marked regions?
  2. How does Linear Probe, LoRA, and Full fine-tuning change that alignment?
  3. Do individual attention heads show descriptive specialisation for different architectural features?
- One-sentence dataset overview: 139 annotated images, 631 expert boxes, 7 models, 5 alignment metrics

**Headline results (deliver before methodology — tell the viewer what they're about to see):**

- **Q1**: Frozen expert-aligned attention exists but is highly model-family dependent. DINOv3 is the only model with consistently strong cross-metric alignment — leading on IoU@90, Coverage, KL, and EMD, and uniquely clearing all calibrated continuous baselines. The SigLIP family illustrates the danger of single-metric reading: best frozen MSE, but EMD worse than random attention.
- **Q2**: Fine-tuning moves attention unevenly. CLIP gains the most (IoU 0.018 → 0.074, Cohen's d ≈ 1.0) but only on Gothic and Romanesque — the styles most densely described in English web text. MAE's largest gain is on Renaissance pediment geometry specifically. DINO stays flat — its strong frozen prior resists reorganisation, and that's a feature. Models converge on the same structurally easy images (DINOv3 frozen IoU predicts CLIP Δ at r = +0.677), so ensembling the language cluster adds no coverage on hard images.
- **Q3**: Per-head specialisation is sparse, descriptive, and family-shaped. DINO-family dominant heads remain stable across adaptation (DINOv3 `layer10/head8` stays top across frozen, LoRA, and Full). MAE is partly reshaped. CLIP reorganises from an earlier frozen head to late-layer adapted heads — head reorganisation tracks the Q2 attention gain story. The strongest head-feature cells cluster on larger structural parts (portals, arches, rose windows); small ornamentation remains hard.

---

### Segment 2 — Dataset & Methodology (2.0 min)

**Research gap framing** (briefly — 2–3 sentences max): existing SSL benchmarks focus on classification accuracy, neglecting *which* image regions drive predictions. Tools like BertViz and Captum visualise attention but don't quantitatively align model attention against human expert diagnoses. This project introduces a multi-metric quantitative benchmark to close that gap.

**Also worth noting**: the professor's mid-term feedback directly shaped methodology — continuous metrics (MSE, KL, EMD) were added alongside IoU following the suggestion to apply Gaussian filtering to bounding boxes; the Preserve/Enhance/Destroy taxonomy came from feedback to study whether fine-tuning preserves or destroys attention consistency. Mentioning this briefly shows methodological responsiveness.

**Dataset**: WikiChurches (Barz & Denzler, NeurIPS 2021) — 9,485 images total, 4,588 used for fine-tuning, 139 expert-annotated images with 631 bounding boxes held out for evaluation. Four architectural styles: Romanesque, Gothic, Renaissance, Baroque. Show the **architecture pipeline diagram** from the mid-term deck (6-box flow: Dataset → 7 Vision Models → Attention Extraction → Alignment Metrics → Fine-Tuning → Analysis).

**Models**: All ViT-B architecture (12 layers, 768 dim, 12 heads, ~86–93M params) except ResNet-50. Use the **model table** from the mid-term deck. Key differentiator per model: DINOv2 (4 register tokens), DINOv3 (Gram anchoring), MAE (pixel reconstruction), CLIP (language-image align), SigLIP/SigLIP2 (no CLS token → Mean attention only), ResNet-50 (Grad-CAM).

**Metrics**: Frame as two ground truth types (from mid-term deck):
- *Binary Ground Truth*: IoU (spatial overlap, threshold-dependent), Coverage (fraction of attention energy inside boxes, threshold-free)
- *Soft Gaussian Ground Truth*: MSE, KL Divergence, EMD — all lower-is-better, compared against a Gaussian heatmap derived from bounding boxes
- Calibrated against 4 naive baselines: random, center Gaussian, saliency prior, Sobel edge — raw scores are meaningless without reference points

---

### Segment 3 — Q1: Frozen Model Benchmark (2.5 min)

**Why a multi-metric benchmark**: Attention alignment is not one thing. A model can place its strongest attention inside the expert boxes, spread attention across the right facade region, or match the overall target distribution — these are related but not identical behaviours. Each metric catches a different failure mode:
- IoU@90: do the model's top-attended pixels land inside the expert annotations?
- Coverage: what fraction of total attention energy falls inside annotated regions?
- MSE, KL, EMD: does the full heatmap distribution match the Gaussian soft-union target?

- Show the leaderboard: **Self-distillation > Supervised > Reconstruction > Multimodal contrastive** (paradigm ordering still valid as headline, but go deeper in narration)
- DINOv3 leads on IoU (0.133), Coverage, KL, EMD. ResNet-50 is second on overlap metrics — the supervised CNN baseline beats all multimodal contrastive models in the frozen setting.
- **Calibrated baseline story**: DINOv3 is the only frozen model to clear all 4 naive baselines on all 3 continuous metrics. Beating random is weak evidence; beating center Gaussian and saliency prior is stronger; beating Sobel edge is the strongest bar.
- SigLIP family: best frozen MSE (0.0175) but EMD worse than random baseline (0.3538 vs 0.3468) — illustrates why single-metric reading misleads. Do not over-interpret SigLIP2 as better than SigLIP.
- **Gram anchoring story**: DINOv3's advantage is hypothesised to come from Gram anchoring, which penalises drift in patch-level feature structure during long SSL training. Two supporting checks: (1) paired gaps are statistically robust (Holm-adjusted p < 1.31×10⁻⁷); (2) DINOv3 aligns best with large coherent parts (Ornate Portal 0.214, Tracery Rose Window 0.164) and worst on small ornamentation (Crocket, Fleuron) — consistent with a model that has a better dense spatial prior for prominent structure, not a complete understanding of every fine-grained cue.
- Show the IoU best-available dashboard screenshot as visual evidence of DINOv3's late-layer jump.

---

### Segment 4 — Q2: Fine-Tuning Effects on Attention (4 min)

- Linear Probe = zero attention change by construction (backbone stays frozen — useful sanity check)
- Show the multi-metric improvement heatmap: CLIP and MAE gain most; DINO stays flat
- **CLIP story**: IoU 0.018 → 0.074 (Cohen's d ≈ 1.0), but gains concentrate on Gothic/Romanesque only — linguistic grounding hypothesis (these styles have features densely described in English web text)
- **MAE story**: Renaissance pediment specialization — gains driven by compact, style-exclusive geometry (Triangular Pediment +0.080, Broken Pediment +0.055); most common Renaissance features (Pilaster, Belt Course) show *negative* Δ
- **DINO story**: near-zero Δ across all styles is a *feature*, not a failure — strong frozen spatial prior resists reorganization
- **Surprise cross-model finding**: DINOv3 frozen IoU predicts CLIP Δ IoU at Pearson r = +0.677 — models converge on the same structurally easy images; language-cluster ensemble adds no coverage on hard images; MAE is the only anti-correlated model (natural complementary partner)

---

### Segment 5 — Q3: Per-Head Specialization (1 min)

**Scope**: DINOv2, DINOv3, MAE, CLIP only — CLS-token models that share the same extraction path (layer → head → CLS-to-patch attention map → alignment score). SigLIP/SigLIP2 excluded (mean-attention proxy, not native heads). ResNet-50 excluded (no transformer heads). Language throughout: *descriptive specialisation*, not causal proof.

**Three views, three results:**

1. **Head Ranking** — Expert-aligned attention is concentrated in a small number of heads, not evenly spread.
   - DINOv3: `layer10/head8` stays top across Frozen, LoRA, and Full; appears in top-3 on 110+ of 139 images in every condition — the clearest stability case
   - DINOv2: same preserved-head pattern at lower absolute alignment
   - MAE: shifts from `layer10/head5` → `layer11/head7` under LoRA, returns to `layer10/head5` under Full — partial reshaping
   - CLIP: frozen best heads sit in early layers (`layer4/head5` at IoU@90 = 0.067), while `layer11` is weak before adaptation (max 0.034) but becomes the strongest adapted layer (LoRA 0.084, Full 0.105) — `layer4/head5` stays near its frozen score
   - **Quantitative sparsity**: in each frozen scoped model the top-ranked (layer, head) pair is 2.7×–3.5× the median pair (DINOv3 3.45×, CLIP 3.22×, MAE 2.92×, DINOv2 2.69×); the top-5 pairs hold 7.3%–9.0% of total mean IoU@90 across all 144 pairs — roughly 2× the 3.5% they would hold under a uniform distribution
   - Show the **Q3 head-ranking transition map** figure

2. **Head-Feature Matrix** — The dominant heads don't just score well overall; their strongest cells cluster on recognisable structural parts, and the weakest cells fail in a consistent way.
   - DINOv3 frozen `layer10/head8`: IoU@90 = 0.215 on Columned Portal, 0.204 on Round Arch Portal, 0.181 on Ornate Portal — weakest on Blind Tracery, Crocket, Tabernacle (≈ 0)
   - Methodology: for each scoped model, take its frozen IoU@90 dominant head from the ranking view and sort features with ≥3 annotations — surface top-3 and bottom-3
   - Pattern holds across all four scoped models. MAE's third-strongest is Wimperg (compact geometry, fits the Q2 MAE story); CLIP's inclusion of Belt Course suggests feature *extent* and clean geometry matter, not only semantic category names
   - Failure cases are not random: across families, weak features are small, thin, repeated, or visually entangled with surrounding masonry
   - **Caveat to flag aloud**: IoU@90 keeps a fixed top-10% attention mask and scores it against a variable-size expert mask — for thin/repeated ornamentation (Crocket, Fleuron, Blind Tracery) the achievable IoU@90 is mechanically capped regardless of where attention falls. Part of the weak-feature shortfall is metric geometry, not head attribution; threshold-free Coverage would help separate the two effects
   - Safe claim: dominant heads expose spatial patterns compatible with expert-marked **structural parts**, not exact ornament detectors
   - Show the **Q3 head-feature matrix** report view, then the **dominant-head top/bottom contact sheet** (`q3_head_feature_top_bottom_contact_sheet.png`) to make the cross-model pattern concrete

3. **Frozen-to-Adapted Delta** — *Sharper follow-up*: when adaptation changes the dominant head, does the strongest expert-aligned signal stay in the same head or move to a different one?
   - **Definition to state aloud**: *expert-aligned signal* = the selected head's normalised CLS-to-patch heatmap scored against expert boxes (top-10% mask vs. expert-box union for IoU@90); strongest head = highest mean IoU@90 across the 139 images for that (layer, variant, metric)
   - **DINO** (stable): DINOv3 `layer10/head8` and DINOv2 `layer11/head11` remain strongest across Frozen, LoRA, Full — strong frozen prior resists reorganisation
   - **MAE** (intermediate): Frozen and Full both favour `layer10/head5`; LoRA shifts to `layer11/head7` — parameter-efficient adaptation can move the dominant signal, but not a full CLIP-style rewrite
   - **CLIP** (clearest reorganisation): within `layer11`, LoRA promotes `H11` from rank **#6 → #1**, with frozen-best `H4` only falling to #3 and remaining close in score (0.080 vs. 0.084) — a new top head without erasing the frozen pattern
   - **CLIP under Full**: stronger rewrite — `H3` moves **#8 → #1**, `H8` moves **#7 → #2**, frozen-best `H4` drops to #4
   - **CLIP synthesis**: adaptation strengthens expert alignment in `layer11` (where frozen CLIP is weak, 0.034 max) *and* reorganises the within-layer ranking of the newly strengthened heads
   - Show the **Q3 frozen-to-adapted delta** figure

**Connecting Q3 back to Q1 and Q2**: DINO-style models already align well before fine-tuning, so adaptation leaves their strongest heads mostly unchanged. CLIP and MAE have more room to move — after adaptation, the best-aligned attention comes from *different heads*, not just from the same head with a higher score. Q3 gives a head-level explanation for the Q2 result.

---

### Segment 6 — Live App Demo (3 min)

Narration hook: *"All of this is backed by precomputed HDF5/SQLite cache — sub-second queries across 139 images × 7 models × 12 layers. Let me show you what that looks like in practice."*

Walk through the app end-to-end, tying each view back to a finding from the talk:

1. **Gallery** → browse the annotated image set; click into an easy image (Q1710328 Gothic) to show the image detail with expert boxes overlaid
2. **Image Detail** → show the DINOv3 frozen CLS attention heatmap at layer 11 alongside the expert boxes — this is what Q1 measures
3. **Dashboard** → IoU@90 best-available leaderboard with DINOv3 at the top and the layer-progression panel showing the late-layer jump; switch to KL metric with baseline overlays to show DINOv3 clearing all 4 calibration baselines
4. **Compare** → frozen vs. fine-tuned CLIP shift map on Q1710328 (Gothic portal) — blue = gained attention after fine-tuning, red = lost; narrate the +0.207 Δ IoU on the Ornate Portal feature
5. **Q2 tab** → multi-metric improvement heatmap + preserve/enhance/destroy summary; point out the zero-Δ Linear Probe row as the control
6. **Q3 tab** → head-ranking view for DINOv3 or CLIP; show a head-feature alignment matrix and point to a head that aligns with portal-class features

End: *"The app lets us move fluidly between the aggregate benchmark and individual image evidence — that's what makes Q1, Q2, and Q3 interpretable rather than just numbers in a table."*

---

### Segment 7 — Conclusions & Takeaways (1 min)

Three sentences, one per question:

- **Q1**: Frozen expert alignment exists but is model-family dependent — DINOv3 is uniquely compatible with expert-marked architectural evidence
- **Q2**: Fine-tuning's effect is mediated by three factors: spatial prior (DINO vs. rest), linguistic coverage (CLIP), and geometric discriminability (MAE)
- **Q3**: Specialisation is sparse and family-shaped — DINO preserves dominant heads, MAE is partly reshaped, CLIP reorganises from early frozen to late adapted heads; strongest alignment concentrates on structural parts, not fine ornamentation

Practical implication: *model selection for domain adaptation shouldn't be guided by accuracy alone — match the pretraining prior to what the task's diagnostic evidence requires.*

---

## Time Budget

| Segment | Time |
|---|---|
| 1. Hook, framing & results summary | 2.0 min |
| 2. Dataset & methodology | 2.0 min |
| 3. Q1 results | 2.5 min |
| 4. Q2 results | 3.5 min |
| 5. Q3 results | 1.0 min |
| 6. Live app demo | 3.0 min |
| 7. Conclusion | 1.0 min |
| **Total** | **15 min** |

---

## Key Visual Assets to Prepare

| Asset | Source |
|---|---|
| Church image + expert boxes vs. attention heatmap | App image detail view |
| Q1 leaderboard table | `outputs/cache/metrics_summary.json` |
| Q1 IoU best-available dashboard screenshot | `docs/final_report/figures/q1_dinov3_react_dashboard_iou_best_available.png` |
| Q1 KL dashboard with baseline overlays | App Dashboard live (Metric=KL, Ranking=Default method) |
| Q2 multi-metric improvement heatmap | `outputs/figures/02_all_metrics_improvement_heatmap.png` |
| Q2 preserve/enhance/destroy summary | `outputs/figures/07_preserve_enhance_destroy.png` |
| Q2 forest plot with CIs | `outputs/figures/08_forest_plot_ci.png` |
| Q2 per-style Δ IoU breakdown | `outputs/results/experiments/fine_tuning_primary_20260327/style_breakdown.png` |
| Q2 MAE per-feature Renaissance table | `outputs/results/experiments/fine_tuning_primary_20260327/feature_delta_iou_mae_full_renaissance.png` |
| Q2 cross-model correlation scatter | `outputs/results/experiments/fine_tuning_primary_20260327/model_correlation_scatter.png` |
| Q2 shift map example (app Compare view) | Q1710328 (Gothic, easy) |
| Q2 ΔIoU across percentile thresholds | `outputs/figures/04_iou_delta_by_percentile.png` |
| Q3 head-ranking transition map | `docs/core/assets/q3_head_ranking_transition_map.png` |
| Q3 head-ranking report view (DINOv3) | `docs/core/assets/q3_head_ranking_report_view.png` |
| Q3 head-feature matrix report view (DINOv3) | `docs/core/assets/q3_head_feature_matrix_report_view.png` |
| Q3 dominant-head top/bottom contact sheet (4 models) | `docs/core/assets/q3_head_feature_top_bottom_contact_sheet.png` |
| Q3 frozen-to-adapted delta (CLIP) | `docs/core/assets/q3_frozen_adapted_delta_report_view.png` |
| Hard image examples | Q694252 (Baroque), Q1424095 (Renaissance) |
| Architecture pipeline diagram | Mid-term deck slide 7 (recreate or screenshot) |
| Model specs table | Mid-term deck slide 8 (ViT-B specs per model) |
| Metrics split diagram | Mid-term deck slide 10 (Binary vs. Soft Gaussian ground truth) |

---

## What Changed from Mid-Term Deck (Do Not Revert)

| Mid-term framing | Final video framing |
|---|---|
| "No single strategy wins everywhere" (tentative) | Specific mechanistic claims: CLIP gains explained by linguistic coverage, MAE by pediment geometry |
| Q1 headline: paradigm ordering only | Paradigm ordering + calibrated baseline clearance story (DINOv3 clears all 4 baselines on all 3 metrics) |
| Q2 results were preliminary | Full per-style breakdown, r=+0.677 cross-model correlation, and easy/hard image framing |
| "Next Steps: complete Q3, cross-layer aggregation" | These are done — conclusion should not repeat mid-term next-steps framing |
| SigLIP2 as improvement over SigLIP | No meaningful frozen difference — treat as same family, warn against over-interpreting |

# Video Script — 12-Minute Narration (Slides Only)
# Do Self-Supervised Vision Models Learn What Experts See?

**Total narrated time**: 12 minutes (Slides 1–33, excluding 3-min demo clip)
**Demo clip**: Slides 34–36 are a separate screen recording (~3 min) — no narration script needed here
**Format**: Screen-recorded voiceover. Read at a measured pace — aim for clarity over speed.

Timing notes per segment are indicative. The 12 minutes is the constraint; adjust pacing within segments to land on time.

---

## Segment 1 — Hook, Problem Framing & Results Summary
**Target: ~2 minutes | Slides 1–4**

---

### Slide 1 — Title
*[~15 seconds]*

> Welcome. This is our ISY5004 Intelligent Sensing Systems Practice Project. I'm Kay Mei, and together with Desmond, we've been investigating whether self-supervised vision models actually learn what architectural experts look at when they identify church styles.

---

### Slide 2 — The Core Question
*[~25 seconds]*

> Here's the core tension we started with. A model predicts "Gothic" correctly — great. But is it looking at pointed arches and flying buttresses, or is it latching onto background regularities that happen to correlate with Gothic churches? These two church images illustrate that: on the left, you see expert bounding boxes marking the diagnostic architectural features. On the right, the raw attention heatmap from a self-supervised model. Accuracy doesn't distinguish these cases — we need a different kind of measurement.

---

### Slide 3 — What We Studied
*[~25 seconds]*

> So we built a benchmark around that question. 139 annotated church images with 631 expert bounding boxes, seven vision models across four self-supervised learning paradigms, and five alignment metrics. We organised the study around three linked research questions: Q1 — how well do frozen models align with expert-marked regions? Q2 — does fine-tuning move that alignment, and does the strategy matter? Q3 — do individual attention heads specialise for different architectural feature types?

---

### Slide 4 — Headline Findings
*[~55 seconds]*

> Before we go into the methodology and evidence, let me tell you what we found — because it will help you read everything that follows.

> For Q1: frozen expert-aligned attention does exist, but it is model-family dependent. DINOv3 leads the default-method benchmark on IoU, Coverage, KL, and EMD, and it is the only frozen model to clear all four naive baselines across MSE, KL, and EMD. It does not win MSE — SigLIP 2 is marginally lowest there, while base SigLIP is actually a competitive overlap result at its best mean-attention layer. That split is the warning: don't collapse SigLIP and SigLIP 2 into one family result, and don't trust a single metric just because it flatters the model.

> For Q2: fine-tuning moves attention unevenly. CLIP gains the most — IoU goes from 0.018 to 0.074, a Cohen's d of approximately 1.0 — but those gains concentrate only on Gothic and Romanesque. MAE's biggest gain is on Renaissance pediment geometry. DINO stays flat, and that near-zero delta is a feature, not a failure. And across all models, the images that are easy for DINOv3 frozen are the same images where CLIP's fine-tuning succeeds — the cross-model correlation there is r equals plus 0.677.

> For Q3: per-head specialisation is sparse and family-shaped. DINOv3's layer 10 head 8 stays dominant across frozen, LoRA, and full fine-tuning. CLIP reorganises substantially from early to late layers. The strongest alignments concentrate on portals, arches, and rose windows — not fine ornamentation.

---

## Segment 2 — Dataset & Methodology
**Target: ~2 minutes | Slides 5–9**

---

### Slide 5 — Dataset & Methodology Section Divider
*[~5 seconds]*

> Let's walk through the setup.

---

### Slide 6 — Dataset: WikiChurches
*[~35 seconds]*

> Our dataset is WikiChurches, introduced by Barz and Denzler in 2021. It's a fine-grained architectural style dataset with four style categories: Romanesque, Gothic, Renaissance, and Baroque. The full dataset has 9,485 images. From those, we used 4,588 for fine-tuning. Critically, 139 images have expert bounding boxes — 631 boxes total across 106 feature types — and we held those 139 images out entirely from all training. Those bounding boxes are the ground truth for all our alignment analysis: they mark the specific architectural features that experts use to distinguish styles.

---

### Slide 7 — Pipeline Overview
*[~35 seconds]*

> Here's the end-to-end pipeline. We start with WikiChurches and our 139 annotated evaluation images. We pass those through seven vision models. From each model we extract attention heatmaps — using CLS attention, rollout, mean attention, or Grad-CAM depending on the model architecture. We then compute five alignment metrics comparing those heatmaps against the expert bounding boxes. In the Q2 branch, we fine-tune each model under three strategies and remeasure alignment. In Q3, we go deeper into the per-head structure of the four native CLS-token models.

---

### Slide 8 — 7 Models Across Paradigms
*[~30 seconds]*

> We evaluated seven models. Six are ViT-Base transformers — DINOv2, DINOv3, MAE, CLIP, SigLIP, and SigLIP 2 — plus ResNet-50 as a supervised CNN baseline. The key architectural distinctions matter here: DINOv2 has four register tokens; DINOv3 adds Gram anchoring; MAE does pixel reconstruction with 75% masking; CLIP uses a global language-image contrastive loss; SigLIP and SigLIP 2 use a sigmoid variant with no CLS token, so they get a mean-attention proxy rather than native per-head attention. ResNet-50 uses Grad-CAM.

---

### Slide 9 — Metrics Used to Measure Alignment
*[~35 seconds]*

> We use five complementary metrics across two ground truth types. For binary ground truth — the hard bounding box mask — we compute IoU, which is the spatial overlap of the model's top-attended pixels versus the expert boxes, and Coverage, which is the fraction of total attention energy inside the boxes without any threshold. For soft Gaussian ground truth — a smooth distribution derived from the boxes — we compute MSE, KL divergence, and EMD, or Earth Mover's Distance. That last one is especially useful because it captures how far the attention mass needs to travel to match the expert target, distinguishing near-misses from far-misses. All continuous metrics are calibrated against four naive baselines: random, center Gaussian, saliency prior, and Sobel edge. Raw scores without these reference points are essentially uninterpretable.

---

## Segment 3 — Q1: Frozen Model Benchmark
**Target: ~2.5 minutes | Slides 10–16**

---

### Slide 10 — Results Section Divider
*[~5 seconds]*

> Now for the results.

---

### Slide 11 — Q1: Frozen Model Benchmark (Overview)
*[~30 seconds]*

> For Q1, the headline story has three parts: DINOv3 is the cleanest frozen alignment result; the leaderboard does not reduce to a tidy paradigm ranking, because ResNet-50 and base SigLIP both sit ahead of MAE and CLIP on sharp overlap; and the SigLIP versus SigLIP 2 split is the best reason to keep the benchmark multi-metric. Let me unpack each of these.

---

### Slide 12 — Why a Multi-Metric Benchmark?
*[~35 seconds]*

> The reason we need five metrics is that attention alignment is not one thing. IoU and Coverage ask whether the model's attention overlaps with the right region. MSE and KL ask whether the overall attention distribution matches a smooth expert-derived target. EMD asks how far the attention mass needs to travel to reach that target — it distinguishes a model that's in the right general area from one that's completely off. And here's the concrete payoff: SigLIP 2 has the lowest MSE, and base SigLIP is almost tied with it, which sounds excellent. But both variants fail the EMD baseline check, and base SigLIP's stronger overlap result does not transfer to SigLIP 2. If we used MSE alone, we would tell the wrong story. All five metrics are necessary.

---

### Slide 13 — Q1: Frozen Leaderboard
*[~30 seconds]*

> Here's the leaderboard. DINOv3 leads on IoU, Coverage, KL, and EMD. ResNet-50 — the supervised CNN baseline — comes second by IoU, and DINOv2 comes third. Then comes the important update: base SigLIP is fourth on IoU at layer 8, ahead of MAE and CLIP, while SigLIP 2 is seventh despite having the lowest MSE. So the correct reading is not "the contrastive family is bad." The correct reading is sharper: DINOv3 is strongest overall, and base SigLIP and SigLIP 2 behave differently enough that we have to report them separately.

---

### Slide 14 — Q1: Why Raw Scores Aren't Enough
*[~30 seconds]*

> This slide makes the calibration argument concrete. The four naive baselines give us reference points: random attention, a center Gaussian, a saliency prior, and a Sobel edge map. DINOv3 is the only frozen model that beats all four baselines on all three continuous metrics — MSE, KL, and EMD. Beating random only is weak; beating Sobel edge, which has some structural signal, is the strongest bar. The SigLIP rows are the cautionary example: SigLIP 2's MSE of 0.01745 and base SigLIP's 0.01755 look impressive, but their EMD scores fail against random. Local smoothness does not equal correct spatial placement of attention mass.

---

### Slide 15 — Why DINOv3 Leads — Gram Anchoring Hypothesis
*[~35 seconds]*

> Why does DINOv3 perform so differently? Our hypothesis is Gram anchoring. DINOv3 extends DINOv2's self-distillation recipe with a Gram-matrix regulariser that explicitly penalises drift in patch-level feature statistics during long SSL training. This is designed to preserve dense spatial structure — exactly the property our alignment metrics reward. Two checks support this: statistically, paired gaps to the next-best model hold at Holm-adjusted p less than 1.31 times ten to the minus seven. Spatially, DINOv3 aligns best with large, coherent architectural parts — Ornate Portal at 0.214, Tracery Rose Window at 0.164 — and worst on small ornamentation like crockets and fleurons. That pattern is exactly what you'd predict from a model with a better dense spatial prior. I want to be clear: this remains a hypothesis — we don't ablate model scale, data scale, and Gram anchoring separately — but the evidence is consistent.

---

### Slide 16 — Q1: Dashboard View
*[~15 seconds]*

> And here's what DINOv3's performance looks like in the app's dashboard. IoU@90 peaks at 0.133 at layer 11 — you can see that distinctive late-layer jump where alignment crystallises in the final transformer blocks. No other frozen model shows this concentration.

---

## Segment 4 — Q2: Fine-Tuning Effects on Attention
**Target: ~3.5 minutes | Slides 17–24**

---

### Slide 17 — Q1: Why DINOv3 Leads (additional)
*[~20 seconds]*

> One more anchor before we move to Q2. The Gram anchoring claim here is not a magic explanation; it is a project-level hypothesis supported by the pattern we actually measured. DINOv3 stays separated from the next-best model in paired image-level checks, and it wins most clearly on large, coherent structures like ornate portals and rose windows. That is exactly where a better dense spatial prior should help.

---

### Slide 18 — Q2: Experiment Design
*[~35 seconds]*

> For Q2, we used three fine-tuning strategies: Linear Probe, LoRA, and Full fine-tuning. Linear Probe keeps the backbone frozen and only trains a classification head — it is our control, and it produces exactly zero delta in alignment across all models and metrics. That's an important sanity check: if alignment moved under Linear Probe, we'd have a reporting artefact. LoRA adapts roughly 300K parameters through low-rank attention projections. Full fine-tuning updates all 86 to 93 million backbone parameters. All three are trained on the same 4,588-image style classification pool, with the 139 annotated images held out throughout. We measure delta as fine-tuned minus frozen, and test significance with paired Wilcoxon tests, Holm correction, and bootstrap confidence intervals.

---

### Slide 19 — Q2: Overview Heatmap
*[~35 seconds]*

> Here's the overall picture from the multi-metric improvement heatmap. Blue cells are improvements, red are degradations, and asterisks mark statistical significance. The strongest positive clusters appear in CLIP, MAE, and the SigLIP variants. The DINO family is near-zero by comparison, and Linear Probe is all zero by construction. The within-cluster language model correlations are r approximately 0.43 to 0.58. MAE is anti-correlated with the language cluster at r approximately negative 0.22 to negative 0.31 — it improves on a different subset of images, which matters for ensembling. One caveat: this is a Q2 layer-11 adaptation story, not a claim that base SigLIP and SigLIP 2 are equivalent in the frozen Q1 benchmark.

---

### Slide 20 — Q2: The CLIP Story
*[~40 seconds]*

> CLIP shows the largest gain in the study. Full fine-tuning takes IoU from 0.018 to 0.074, a Cohen's d of approximately 1.0. LoRA also improves, though less so. But here's the key insight: the gain is not uniform across styles. Gothic images gain plus 0.079, Romanesque plus 0.066. Renaissance and Baroque gain only about 0.013 to 0.014. Why? CLIP was pretrained on 400 million web image-text pairs with a global contrastive loss — there was no patch-level spatial pressure at all. Fine-tuning is the first time CLIP receives a spatial signal. And when it does, the gains concentrate on Gothic and Romanesque because those styles' diagnostic features — pointed arch portals, round arch portals, tracery, bull's-eye windows — are densely described in English-language text about churches. CLIP's attention follows its linguistic grounding.

---

### Slide 21 — Q2: The MAE Story
*[~35 seconds]*

> MAE tells a completely different story. Its largest single-style gain is on Renaissance at plus 0.108 — more than double any of CLIP's per-style gains. And if you look at which features drive that, it's pediment geometry: triangular pediment plus 0.080, cranked cornice plus 0.062, broken pediment plus 0.055. Meanwhile, the two most common Renaissance features — Pilaster and Belt Course — both go negative. The style-classification gradient is routing attention toward the most geometrically discriminative forms and away from the ones that appear across multiple styles. This makes sense for MAE: 75% pixel masking forces precise local geometry encoding, so when fine-tuning arrives, it can redirect that geometric precision toward the most diagnostic shapes.

---

### Slide 22 — Q2: The DINO Story
*[~25 seconds]*

> DINO is the near-zero case across styles and strategies. The report's reading is simple: DINOv3 already has the strongest frozen spatial alignment, so fine-tuning has much less useful work to do. That near-zero delta is not a failure — it is evidence that the useful spatial prior is already there. It is also a warning that more adaptation is not automatically better, because an already strong representation can be disturbed rather than improved.

---

### Slide 23 — Q2: The Surprise — Models Converge on Same Images
*[~35 seconds]*

> Now here's the finding that came as a pleasant surprise. DINOv3's frozen IoU predicts CLIP's per-image delta IoU at Pearson r equals plus 0.677. The images where DINOv3 already attends correctly are the same images where CLIP's fine-tuning succeeds. The structural barrier is often the image, not just the model family. In practice, that means a cluster of language-image models — CLIP, SigLIP, and SigLIP 2 — tends to improve on overlapping image subsets. MAE is the anti-correlated case, which makes it the more interesting complementary partner if you want broader coverage.

---

### Slide 24 — Q2: Statistical Confidence
*[~20 seconds]*

> Finally, the forest plot confirms that these gains are not anecdotal. This shows the mean delta with 95% bootstrap confidence intervals across the reported metric views for LoRA and Full fine-tuning. The asterisks show significance after Holm correction. Several CLIP, MAE, and SigLIP-family gains remain statistically supported over the paired 139-image evaluation set.

---

## Segment 5 — Q3: Per-Head Specialisation
**Target: ~1 minute | Slides 25–29**

---

### Slide 25 — Q3: Scope and Approach
*[~15 seconds]*

> For Q3, we scope to the four native CLS-token ViTs: DINOv2, DINOv3, MAE, and CLIP. SigLIP and ResNet-50 are excluded because their per-head proxies aren't comparable. We're looking at descriptive specialisation — not causal attribution. IoU@90 is the primary head-ranking lens; Coverage and EMD act as robustness checks rather than a forced composite score. Three views make that inspectable: head ranking, a head-feature matrix, and frozen-to-adapted deltas.

---

### Slide 26 — Q3 View 1: Head Ranking
*[~20 seconds]*

> Expert-aligned attention concentrates in a small number of heads. DINOv3's layer 10 head 8 stays top across frozen, LoRA, and full, and appears in the top three on more than 110 of 139 images in every condition. CLIP is the clearest reorganisation: frozen best heads sit in early layers, layer 4 head 5 at 0.067, while layer 11 is weak frozen at 0.034 max but becomes the strongest adapted layer at 0.084 under LoRA and 0.105 under Full. Quantitatively, the top-5 pairs hold about 7 to 9 percent of total mean IoU across all 144 layer-head pairs — roughly 2 times what you'd expect under uniform spread.

---

### Slide 27 — Q3 View 2a: Head-Feature Matrix
*[~10 seconds]*

> The head-feature matrix links the ranking to specific architectural evidence. DINOv3's layer 10 head 8 scores 0.215 on Columned Portal, 0.204 on Round Arch Portal, and 0.181 on Ornate Portal. The dominant head is strongest on portal-scale structure.

---

### Slide 28 — Q3 View 2b: Top vs. Bottom Features Across Models
*[~15 seconds]*

> The pattern holds across all four models. Strongest features are consistently portal-scale or large facade structures. Weakest features — blind tracery, crocket, tabernacle, fleuron — are thin, small, repeated, or visually entangled with masonry. Part of that weakness is also metric geometry: IoU at a fixed 10% threshold mechanically caps the achievable score for small annotation boxes regardless of where attention falls. The safe claim is that dominant heads expose spatial patterns compatible with structural parts — not exact ornament detectors.

---

### Slide 29 — Q3 View 3: Frozen-to-Adapted Delta
*[~15 seconds]*

> The frozen-to-adapted delta shows what happens to the dominant head across variants. DINOv3 is stable — layer 10 head 8 stays top across all three. MAE is intermediate — LoRA shifts the dominant head, Full brings it back. CLIP is the clearest reorganisation: within layer 11, LoRA promotes H11 from rank 6 to rank 1 without erasing the frozen pattern; Full fine-tuning is a stronger rewrite, H3 moves from rank 8 to rank 1. This is the head-level explanation for the Q2 CLIP story: adaptation doesn't just raise scores in layer 11, it reorganises which heads carry the expert-aligned signal.

---

## Segment 6 — Conclusions & Takeaways
**Target: ~1 minute | Slides 30–32**

---

### Slide 30 — Conclusions Section Divider
*[~5 seconds]*

> So, to bring it all together.

---

### Slide 31 — What We Found
*[~35 seconds]*

> Three questions, three answers. For Q1: DINOv3 provides the strongest frozen alignment evidence, leading four of five metrics and uniquely clearing the calibrated continuous baselines. Base SigLIP is a stronger frozen overlap result than SigLIP 2, but both still have late-layer adaptation headroom. For Q2: fine-tuning's effect is mediated by three interacting factors — the pretraining objective's spatial prior, dataset linguistic coverage, and geometric discriminability. No single strategy wins everywhere. For Q3: per-head specialisation is sparse, descriptive, and family-shaped. DINO preserves dominant heads, MAE is partly reshaped, and CLIP reorganises from early frozen to late adapted heads. Strongest alignment concentrates on structural parts, not fine ornamentation.

---

### Slide 32 — Practical Implication
*[~20 seconds]*

> The practical upshot: model selection for domain adaptation should not be guided by accuracy alone. If your task's diagnostic evidence is spatially coherent, DINOv3 frozen may already be sufficient. If it's linguistically described in web text, CLIP with full fine-tuning gives the largest gain. If it's geometrically compact and style-exclusive, MAE covers a complementary image subset. And if you want coverage on hard images, MAE plus a language-cluster model — not the language cluster alone.

---

## Segment 7 — App Demo Lead-In
**Target: ~5 seconds | Slide 33**

---

### Slide 33 — Demo Section Divider
*[~5 seconds]*

> Let me now show you all of this in the app.

---

## [DEMO CLIP — 3 minutes, separate screen recording]

*No narration script — this is a live walkthrough of the React/FastAPI interface.*
*Cover: Gallery → Image Detail → Dashboard (IoU and KL views) → Compare (CLIP shift map) → Q2 tab → `/q3-report` route (head ranking, head-feature matrix, frozen-to-adapted delta)*
*End line: "The app lets us move fluidly between the aggregate benchmark and individual image evidence — that's what makes Q1, Q2, and Q3 interpretable rather than just numbers in a table."*

---

### Slides 34–36 — App Views (shown during demo recording)
*[These slides appear during the demo clip — no separate narration needed]*

---

## Timing Summary

| Segment | Slides | Target Time | Notes |
|---|---|---|---|
| 1. Hook & results summary | 1–4 | 2:00 | Slide 4 is dense — budget ~55 sec |
| 2. Dataset & methodology | 5–9 | 2:00 | Slide 9 metrics is important — don't rush |
| 3. Q1 results | 10–16 | 2:30 | Slide 15 Gram anchoring takes time |
| 4. Q2 results | 17–24 | 3:30 | CLIP and MAE stories are the centrepiece |
| 5. Q3 results | 25–29 | 1:00 | Fast but precise — don't drop the caveats |
| 6. Conclusions | 30–32 | 1:00 | Measured pace, land the key phrases |
| Demo lead-in | 33 | 0:05 | One sentence |
| **Narrated total** | **1–33** | **12:05** | ~5 seconds of buffer |
| Demo clip | 34–36 + recording | 3:00 | Separate file |
| **Grand total** | | **15:05** | |

---

## Delivery Notes

- **Slide 4** (headline findings): This is the hardest slide to pace. The three Q findings together run about 55 seconds — practice this one until it flows naturally.
- **Number precision**: The key numbers to say correctly on air: CLIP IoU 0.018 → 0.074, Cohen's d ≈ 1.0, r = +0.677, p < 1.31 × 10⁻⁷, base SigLIP IoU 0.0739, SigLIP 2 MSE 0.01745, layer 10 head 8, 139 images, 631 boxes.
- **The SigLIP/SigLIP 2 warning** (Slides 12–14): Make sure the audience hears the updated version: base SigLIP is competitive on overlap, SigLIP 2 is best on MSE, and both fail the EMD baseline check. That is the clearest demonstration that the methodology is necessary.
- **Caveats to flag aloud** (don't skip these):
  - Slide 15: Gram anchoring is a hypothesis, not an ablated causal result
  - Slide 28: IoU@90's fixed-mask geometry caps achievable scores for thin annotations
  - Slide 29: Q3 is descriptive specialisation, not causal attribution
- **Pace check points**: You should be approximately at Slide 10 at 4:00, Slide 18 at 6:30, Slide 25 at 10:00, Slide 33 at 12:00.

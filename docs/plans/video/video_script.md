# Video Script — 12-Minute Narration (Slides Only)
# Do Self-Supervised Vision Models Learn What Experts See?

**Total narrated time**: 12 minutes (Slides 1–33, excluding 3-min demo clip)
**Demo clip**: Slides 34–35 are a separate screen recording (~3 min) — no narration script needed here
**Format**: Screen-recorded voiceover. Read at a measured pace — aim for clarity over speed.

Timing notes per segment are indicative. The 12 minutes is the constraint; adjust pacing within segments to land on time.

---

## Segment 1 — Hook, Problem Framing & Results Summary
**Target: ~2 minutes | Slides 1–4**

---

### Slide 1 — Title
*[~15 seconds]*

> Welcome. I'm Kay Mei, and together with Desmond, we've been investigating whether self-supervised vision models actually learn what architectural experts look at when they identify church styles.

---

### Slide 2 — The Core Question
*[~25 seconds]*

> Here's the core tension we started with. A model predicts "Romanesque" correctly — great. But is it deliberately looking at pointed arches, or is it a coincidence? These two church images illustrate that: on the left, you see expert bounding boxes marking the diagnostic architectural features. On the right, the raw attention heatmap from a self-supervised model. Accuracy doesn't distinguish these cases — we need a different kind of measurement.

---

### Slide 3 — What We Studied
*[~25 seconds]*

> So we built a benchmark around that question. We used 139 annotated church images with 631 expert bounding boxes, compared seven vision models across four self-supervised learning paradigms, and measured alignment using five metrics: IoU, Coverage, MSE, KL divergence, and EMD. We organised the study around three linked research questions: Q1 — which frozen models already look in the right places? Q2 — does fine-tuning change where the models look? And Q3 — inside the transformer models, do some attention heads consistently line up with particular architectural features?

---

### Slide 4 — Headline Findings
*[~55 seconds]*

> Before we go into the methodology and evidence, let me tell you what we found — because it will help you read everything that follows.

> For Q1: frozen expert-aligned attention does exist, but it is model-family dependent. DINOv3 leads the benchmark on IoU, Coverage, KL, and EMD, and it is the only frozen model to clear all four naive baselines across MSE, KL, and EMD. It does not win MSE — SigLIP 2 is marginally lowest there — while base SigLIP is actually stronger on direct IoU overlap. That split is the warning: a single metric can make the wrong model look best.

> For Q2: fine-tuning moves attention unevenly. CLIP improves the most — IoU rises from 0.018 to 0.074, but those gains concentrate mainly on Gothic and Romanesque churches. MAE's largest gain is on Renaissance pediment geometry. DINOv3 stays near zero delta, which is not a failure: it was already good at preserving the relevant spatial layout before fine-tuning. And the images that are easy for frozen DINOv3 are also the images where CLIP improves most after fine-tuning.

> For Q3: per-head specialisation is sparse and family-shaped — DINOv3 keeps the same strongest head before and after fine-tuning, while CLIP shifts its strongest signal from earlier layers to later layers. Across models, the clearest matches are on large structures, not tiny decorative details.

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

> Our dataset is WikiChurches, introduced by Barz and Denzler in 2021. It's a fine-grained architectural style dataset with four style categories: Romanesque, Gothic, Renaissance, and Baroque. The full dataset has 9,485 images. From those, we used 4,588 for fine-tuning. 139 of those images have expert bounding boxes, and we held those 139 images out entirely from all training. Those bounding boxes are the ground truth for all our alignment analysis: they mark the specific architectural features that experts use to distinguish styles.

---

### Slide 7 — Pipeline Overview
*[~35 seconds]*

> Here's the end-to-end pipeline. We start with WikiChurches and our 139 annotated evaluation images. We pass those through seven vision models. From each model we extract attention heatmaps — using CLS attention, attention rollout, mean attention, or Grad-CAM depending on the model architecture. We then compute five alignment metrics against the expert boxes. For Q2, we fine-tune the models and remeasure the same metrics. For Q3, we inspect individual attention heads, but only for the transformer models where that head-by-head comparison is valid.

---

### Slide 8 — 7 Models Across Paradigms
*[~30 seconds]*

> We evaluated seven models. Six are ViT-Base transformers — DINOv2, DINOv3, MAE, CLIP, SigLIP, and SigLIP 2 — plus ResNet-50 as a supervised CNN baseline. The paradigms matter here: DINO models learn through self-distillation, MAE learns by reconstructing masked image patches, and CLIP and SigLIP learn from image-text contrastive training. That training history matters, and we'll see it directly in the Q1 and Q2 results.

---

### Slide 9 — Metrics Used to Measure Alignment
*[~35 seconds]*

> We use five complementary metrics because "looking in the right place" has more than one meaning. For binary ground truth, IoU asks whether the model's top-attended pixels overlap with the expert boxes, and Coverage asks how much total attention energy falls inside those boxes. For soft Gaussian ground truth, MSE and KL divergence compare the overall heatmap distribution to a smooth expert target. EMD, or Earth Mover's Distance, measures how far the attention mass would need to move to match that target.

---

## Segment 3 — Q1: Frozen Model Benchmark
**Target: ~2 minutes | Slides 10, 12–16**

---

### Slide 10 — Results Section Divider
*[~5 seconds]*

> Now for the results.

---

### Slide 12 — Why a Multi-Metric Benchmark?
*[~35 seconds]*

> The reason we need five metrics is that attention alignment is not one thing. A heatmap can overlap the expert box, spread attention energy over the right facade area, or match the smooth Gaussian target while still putting mass too far away. That is exactly what the SigLIP results show. SigLIP 2 has the best MSE, and base SigLIP is almost tied with it, so the MSE story looks excellent. But both variants fail the distance-based EMD baseline check, meaning the attention mass is still too far from the expert regions. If we used MSE alone, we would tell the wrong story.

---

### Slide 13 — Q1: Frozen Leaderboard
*[~30 seconds]*

> Here's the leaderboard using IoU@90, our strict top-10-percent overlap score. DINOv3 is first, ResNet-50 is second, and DINOv2 is third. Then comes the important detail: base SigLIP is fourth, ahead of MAE and CLIP, while SigLIP 2 is seventh even though it has the lowest MSE. So we can conclude that DINOv3 is strongest overall, and base SigLIP and SigLIP 2 behave differently enough that we should not merge them into one family result.

---

### Slide 14 — Q1: Why Raw Scores Aren't Enough
*[~30 seconds]*

> This slide shows why raw scores need reference points. We compare each model against random attention, a center Gaussian, a saliency prior, and a Sobel edge map. DINOv3 is the only frozen model that beats all four baselines on all three continuous metrics — MSE, KL, and EMD. The SigLIP rows are the cautionary example: SigLIP 2's MSE of 0.01745 and base SigLIP's 0.01755 look impressive, but their EMD scores fail against random. In plain language, a heatmap can look smooth and still be in the wrong place.

---

### Slide 15 — Why DINOv3 Leads — Gram Anchoring Hypothesis
*[~35 seconds]*

> Why does DINOv3 perform so differently? Our hypothesis is Gram anchoring. In simple terms, DINOv3's training includes a mechanism that helps preserve patch-level spatial structure during long self-supervised training. That is exactly the kind of property our task rewards, because expert boxes mark where architectural parts are in the image. Two checks support this. Statistically, DINOv3 stays clearly ahead of the next-best model after Holm correction. Visually, it works best on large and worse on tiny decorations. This is not a proof that Gram anchoring alone caused the result, but it is a plausible explanation for the pattern.

---

### Slide 16 — Q1: Dashboard View
*[~15 seconds]*

> And here's what DINOv3's performance looks like in the app's dashboard. IoU@90 peaks at 0.133 at layer 11. The important visual pattern is the late-layer jump: the strongest alignment appears in the final transformer blocks, where the model has built a more structured view of the image.

---

## Segment 4 — Q2: Fine-Tuning Effects on Attention
**Target: ~3.5 minutes | Slides 17–24**

---

### Slide 17 — Q2: Experiment Design
*[~35 seconds]*

> For Q2, we used three fine-tuning strategies: Linear Probe, LoRA, and Full fine-tuning. Linear Probe trains only the final classifier, so the visual backbone stays frozen; it is our zero-delta control. LoRA makes a small targeted update through low-rank attention adapters. Full fine-tuning updates the whole visual backbone. All three use the same 4,588-image style classification pool, while the 139 annotated images stay held out. We measure delta as fine-tuned minus frozen, and test significance with paired Wilcoxon tests, Holm correction, and bootstrap confidence intervals.

---

### Slide 18 — Q2: Overview Heatmap
*[~35 seconds]*

> Here's the overall picture from the multi-metric improvement heatmap. Blue cells mean the fine-tuned model moved closer to the expert boxes; red cells mean it moved away; asterisks mark statistically significant shifts. The strongest improvements appear in CLIP, MAE, and the SigLIP variants. The DINO models change very little, and Linear Probe is zero because the visual backbone never changed.

---

### Slide 19 — Q2: The CLIP Story
*[~40 seconds]*

> CLIP shows the largest gain in the study. Full fine-tuning raises IoU from 0.018 to 0.074, with Cohen's d around 1.0. LoRA also improves CLIP, though less strongly. But the key point is that the gain is not uniform across styles: Gothic images gain about +0.079, Romanesque +0.066, while Renaissance and Baroque are close to zero. A plausible reason is CLIP's pretraining: it learns from image-text pairs, so it may already carry useful language-linked concepts for features like pointed arch portals, round arch portals, tracery, and bull's-eye windows. Fine-tuning then redirects that weak frozen attention toward those named architectural structures.

---

### Slide 20 — Q2: The MAE Story
*[~35 seconds]*

> MAE tells a different story. Its largest style-specific gain is on Renaissance, at about +0.108 IoU, and the features driving that gain are pediment shapes: triangular pediment, cranked cornice, and broken pediment. At the same time, common features like pilasters and belt courses do not improve. So MAE is not simply learning "more Renaissance." It is shifting attention toward geometric shapes that are more distinctive for the style. That fits MAE's pretraining, where the model learns by reconstructing missing image patches and therefore has a strong bias toward local geometry.

---

### Slide 21 — Q2: The SigLIP Variants
*[~25 seconds]*

> SigLIP and SigLIP 2 are worth separating. In Q1, base SigLIP outranks SigLIP 2 on IoU. In Q2, the story flips slightly: SigLIP 2 shows a larger standardised effect under full fine-tuning — a plausible reason is that it starts from a weaker frozen baseline, giving adaptation more room to move. Either way, Q2 for both variants is sharpening a weak late-layer signal, not improving on each model's Q1 best layer.

---

### Slide 22 — Q2: The DINO Story
*[~25 seconds]*

> DINO is the near-zero delta case across styles and strategies. The reading is simple: DINOv3 already has the strongest frozen spatial alignment, so fine-tuning has much less useful work to do. The lack of movement is not a failure. It suggests that the useful spatial prior was already there before fine-tuning. It is also a warning that more adaptation is not automatically better, because an already strong representation can be disturbed rather than improved.

---

### Slide 23 — Q2: The Surprise — Models Converge on Same Images
*[~35 seconds]*

> Now here's the finding that came as a pleasant surprise. DINOv3's frozen IoU predicts CLIP's per-image delta IoU at Pearson r equals +0.677. The images where frozen DINOv3 already looks in the right place are also the images where CLIP improves most after fine-tuning. This tells us that some images are structurally easy — large portals, clear arches, coherent facades — and several model families succeed on those same images. MAE is the interesting exception because it improves on a more different subset, especially Renaissance geometry.

---

### Slide 24 — Q2: Statistical Confidence
*[~20 seconds]*

> Finally, the forest plot checks that these gains are not just a few lucky examples. Each point shows the mean delta with a 95% bootstrap confidence interval. The asterisks mark results that remain statistically significant after Holm correction. Several CLIP, MAE, and SigLIP-family gains remain supported over the paired 139-image evaluation set.

---

## Segment 5 — Q3: Per-Head Specialisation
**Target: ~1 minute | Slides 25–29**

---

### Slide 25 — Q3: Scope and Approach
*[~15 seconds]*

> For Q3, we look inside the transformer models. Each transformer layer has multiple attention heads; you can think of a head as one small component that decides how image patches relate to each other. We only analyze DINOv2, DINOv3, MAE, and CLIP here, because they expose comparable CLS-token attention heads. The question is intentionally modest: which heads line up most strongly with expert boxes? This is descriptive specialisation, not causal attribution. We use IoU@90 as the main ranking, then check the pattern with Coverage and EMD.

---

### Slide 26 — Q3 View 1: Head Ranking
*[~20 seconds]*

> The first result is sparse specialisation: not all heads matter equally. Out of 144 possible layer-and-head pairs, a small number carry much more of the expert-aligned signal. In DINOv3, the same head — layer 10, head 8 — remains the strongest before and after fine-tuning, and it is among the top three heads on more than 110 of the 139 images. We get a mixed response for MAE. CLIP behaves differently. Before fine-tuning, its strongest heads are earlier in the network. After fine-tuning, the strongest signal shifts to a later layer. So DINOv3 mostly preserves its internal attention pattern, while CLIP reorganizes.

---

### Slide 27 — Q3 View 2a: Head-Feature Matrix
*[~10 seconds]*

> The head-feature matrix connects those head rankings to actual architectural labels. For DINOv3's strongest head, the best-matching features are all portal-scale structures: columned portals, round arch portals, and ornate portals. So the head is not just strong in the abstract; it lines up with large architectural parts that experts actually marked.

---

### Slide 28 — Q3 View 2b: Top vs. Bottom Features Across Models
*[~15 seconds]*

> The same pattern appears across the four Q3 models. The strongest matches are usually large facade structures: portals, arches, belt courses, and rose-window-scale features. The weakest matches are fine decorations such as blind tracery and crockets. Part of that shortfall is metric geometry — IoU@90's fixed top-10% threshold is mechanically capped for small expert boxes, so threshold-free Coverage would help separate real model difficulty from measurement artefact. The safer claim is that the strongest heads line up better with large structural parts.

---

### Slide 29 — Q3 View 3: Frozen-to-Adapted Delta
*[~15 seconds]*

> The frozen-to-adapted delta view shows what fine-tuning does to the strongest head. DINOv3 is stable: the same head remains strongest. MAE is partly reshaped: LoRA shifts the strongest head, while full fine-tuning brings it back. CLIP changes the most. In layer 11, LoRA promotes a different head to rank 1, and full fine-tuning promotes another. This gives a more concrete explanation for Q2: CLIP improves because fine-tuning changes which internal components carry the expert-aligned signal.

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

> Three questions, three answers. For Q1: DINOv3 gives the strongest evidence that a model can already look near expert-marked regions before fine-tuning, leading on IoU, Coverage, KL, and EMD. Base SigLIP and SigLIP 2 also show why the benchmark must stay multi-metric, because one looks better on overlap and the other looks better on MSE. For Q2: fine-tuning helps most when the model does not already preserve the relevant spatial layout, and the gains depend on which architectural features the model can learn from the style labels. For Q3: per-head specialisation is sparse — the expert-aligned signal is concentrated in a few internal attention heads, not spread evenly everywhere. DINO mostly preserves its internal attention pattern, while CLIP reorganizes after fine-tuning. Across all three questions, the clearest alignments are on large structural parts, not fine ornaments.

---

### Slide 32 — Practical Implication
*[~20 seconds]*

> The practical implication is that model selection for domain adaptation should not be guided by accuracy alone. If the diagnostic evidence is spatially coherent, DINOv3 may already be a strong frozen choice. If the evidence is commonly described in text, CLIP can gain a lot from full fine-tuning. If the evidence is compact geometry, MAE can cover cases that the text-trained models miss. So if we want broader coverage on hard images, the better pairing is MAE plus a text-trained model, not several text-trained models that tend to improve on the same examples.

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

### Slides 34–35 — App Views (shown during demo recording)
*[These slides appear during the demo clip — no separate narration needed]*

---

## Timing Summary

| Segment | Slides | Target Time | Notes |
|---|---|---|---|
| 1. Hook & results summary | 1–4 | 2:00 | Slide 4 is dense — budget ~55 sec |
| 2. Dataset & methodology | 5–9 | 2:00 | Slide 9 metrics is important — don't rush |
| 3. Q1 results | 10, 12–16 | 2:00 | Slide 15 Gram anchoring takes time |
| 4. Q2 results | 17–24 | 3:30 | CLIP, MAE, SigLIP, and DINO stories |
| 5. Q3 results | 25–29 | 1:00 | Fast but precise — don't drop the caveats |
| 6. Conclusions | 30–32 | 1:00 | Measured pace, land the key phrases |
| Demo lead-in | 33 | 0:05 | One sentence |
| **Narrated total** | **1–33** | **12:05** | ~5 seconds of buffer |
| Demo clip | 34–35 + recording | 3:00 | Separate file |
| **Grand total** | | **15:05** | |

---

## Delivery Notes

- **Slide 4** (headline findings): This is the hardest slide to pace. The three Q findings together run about 55 seconds — practice this one until it flows naturally.
- **Number precision**: The key numbers to say correctly on air: CLIP IoU 0.018 → 0.074, Cohen's d ≈ 1.0, r = +0.677, base SigLIP IoU 0.0739, SigLIP 2 MSE 0.01745, layer 10 head 8, 139 images, 631 boxes.
- **The SigLIP/SigLIP 2 warning** (Slides 12–14): Say the warning plainly: base SigLIP is stronger on direct IoU overlap, SigLIP 2 is stronger on MSE, and both fail the distance-based EMD baseline check. That is the clearest demonstration that the methodology is necessary.
- **Caveats to flag aloud** (don't skip these):
  - Slide 15: Gram anchoring is a hypothesis, not an isolated causal proof
  - Slide 28: IoU@90's fixed top-10-percent geometry makes tiny annotation boxes harder to reward
  - Slide 29: Q3 describes where heads align; it does not prove that one head caused the prediction
- **Pace check points**: You should be approximately at Slide 10 at 4:00, Slide 17 at 6:30, Slide 25 at 10:00, Slide 33 at 12:00.

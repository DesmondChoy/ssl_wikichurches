# Understanding Attention in SSL Vision Models

This document explains how attention visualization works in this application and how it helps answer our core research question: **Do self-supervised learning (SSL) models attend to the same visual features that human experts consider diagnostic for architectural style classification?**

> **Related document:** This guide complements the [Project Proposal](../core/project_proposal.md), which details the full research design, hypotheses, and evaluation methodology. Links to specific proposal sections are provided throughout.

---

## Table of Contents

1. [The Research Problem](#the-research-problem)
2. [What is Attention?](#what-is-attention)
3. [Attention Methods in This App](#attention-methods-in-this-app)
   - [CLS Token Attention](#1-cls-token-attention)
   - [Attention Rollout](#2-attention-rollout)
   - [Mean Attention](#3-mean-attention)
   - [Grad-CAM](#4-grad-cam-gradient-weighted-class-activation-mapping)
4. [How We Measure Alignment](#how-we-measure-alignment)
5. [Model-Specific Considerations](#model-specific-considerations)
6. [Interpreting the Visualizations](#interpreting-the-visualizations)
7. [Technical Reference](#technical-reference)
8. [Academic Foundations](#academic-foundations)
9. [References](#references)

---

## The Research Problem

> **See also:** [Project Proposal §1 - Problem Statement](../core/project_proposal.md#1-problem-statement)

When we train neural networks to classify images, they learn to focus on certain visual features. But a critical question remains: **Are these the *right* features?**

Consider a model that correctly identifies Gothic architecture. Is it:
- Attending to **flying buttresses and pointed arches** (the features experts use), or
- Exploiting **dataset biases** like the fact that Gothic churches are often photographed against overcast skies?

This distinction matters enormously for trust and deployment. A model that achieves high accuracy through genuine understanding is fundamentally different from one that exploits statistical shortcuts.

### Our Approach

The [WikiChurches dataset](../core/project_proposal.md#2-dataset) (Barz & Denzler, 2021) provides a rare opportunity: **631 expert-annotated bounding boxes** marking the specific architectural features (rose windows, buttresses, arches, towers) that define each style across 139 churches in 4 architectural styles. By comparing where models "look" to where experts say to look, we can quantitatively measure whether SSL models have learned meaningful visual concepts.

This app visualizes these attention patterns and computes the overlap (IoU) between model attention and expert annotations.

### The Four Research Questions

This application helps answer four interconnected questions from our [research design](../core/project_proposal.md#research-questions-and-approaches):

| # | Research Question | How This App Helps |
|---|-------------------|-------------------|
| 1 | Do SSL models attend to the same features human experts consider diagnostic? | Attention heatmaps + IoU metrics |
| 2 | Does fine-tuning shift attention toward expert-identified features? | Frozen vs. fine-tuned comparison |
| 3 | Do individual attention heads specialize for different architectural features? | Per-head attention selector + per-head IoU analysis |
| 4 | Does the fine-tuning strategy affect how much attention shifts toward expert features? | Fine-tuning method comparison + Δ IoU by method |

### Per-Head Attention Analysis (Research Question 3)

Rather than fusing all 12 attention heads via averaging, Q3 examines each head individually to understand **which heads develop alignment with expert annotations**.

**Key metrics:**
- **Per-head IoU:** Compute IoU for each head separately (head 0, head 1, ..., head 11)
- **Head ranking:** Identify which heads consistently achieve highest IoU across all images
- **Head × feature type matrix:** Analyze whether specific heads specialize for specific architectural features

**Academic foundation:**
- Voita et al. (2019) showed that "a small subset of heads" performs most of the work in transformers
- Caron et al. (2021) demonstrated that DINO heads exhibit diverse specialization patterns
- This analysis extends these findings to measure alignment with domain expertise

### Fine-Tuning Method Comparison (Research Question 4)

Q4 compares three fine-tuning strategies to understand how training approach affects attention shift:

| Method | Trainable Params | Expected Behavior |
|--------|------------------|-------------------|
| **Linear Probe** | ~3K (head only) | No attention change (baseline) |
| **LoRA (r=8)** | ~300K | Moderate attention shift with preserved pre-training |
| **Full Fine-tuning** | ~86M | Maximum attention shift, risk of catastrophic forgetting |

**Academic foundation:**
- Hu et al. (2022) introduced LoRA for parameter-efficient adaptation
- Biderman et al. (2024) showed "LoRA learns less and forgets less" compared to full fine-tuning

---

## What is Attention?

Modern vision models (Vision Transformers, or ViTs) divide images into small patches (14×14 or 16×16 pixels) and process them as a sequence, similar to how language models process words. **Attention** is the mechanism by which the model decides which patches to "look at" when building its understanding of the image.

### The CLS Token

Most ViT models include a special **CLS (class) token** that aggregates information from all patches to form a global image representation. Think of it as a "summary" token that reads from all parts of the image. Where this token attends reveals what the model considers important.

### Attention Heads

Transformers have multiple **attention heads** operating in parallel, each potentially focusing on different aspects of the image. Our visualizations fuse these heads together (typically by averaging) to show the overall attention pattern.

---

## Attention Methods in This App

> **See also:** [Project Proposal §3.2 - Attention Extraction and Visualization](../core/project_proposal.md#32-attention-extraction-and-visualization)

Different methods extract attention information in different ways, each with strengths and limitations. Not all methods work with all models.

The [ablation studies](../core/project_proposal.md#4-ablation-studies) compare these methods to determine which extraction approach best captures expert-relevant regions.

### 1. CLS Token Attention

**What it shows:** Where the CLS token directly attends within a single layer.

**How it works:**
1. Extract the attention weights from a specific transformer layer
2. Look at the row corresponding to the CLS token (position 0)
3. This row contains the attention weight to each image patch
4. Average across all attention heads

**Interpretation:**
- High values (red/yellow) = patches the model considers important for building its global representation
- The pattern varies dramatically by layer (see [Interpreting the Visualizations](#interpreting-the-visualizations))

**Available for:** DINOv2, DINOv3, MAE, CLIP (models with a CLS token)

**Research relevance:** Answers "What does layer N consider important?" Useful for understanding how attention evolves through the network and which layers develop expert-aligned attention.

**Academic basis:** [Caron et al. (2021)](https://arxiv.org/abs/2104.14294) discovered that DINO's self-attention maps "contain explicit information about the semantic segmentation of an image"—attention maps that resemble segmentation masks emerge without any supervision. This remarkable property is why CLS attention is meaningful for our analysis.

```
Source: src/ssl_attention/attention/cls_attention.py
```

### 2. Attention Rollout

**What it shows:** Accumulated attention through all layers, capturing indirect attention paths.

**How it works:**
1. Start with an identity matrix (everything attends to itself)
2. For each layer, multiply by that layer's attention matrix (plus residual connection)
3. The result captures paths like: if layer 1 attends A→B and layer 2 attends B→C, rollout shows A effectively attends to C

**Mathematical formulation:**
```
R₀ = I (identity matrix)
Rᵢ = normalize((Aᵢ + I) @ Rᵢ₋₁)
```

**Interpretation:**
- Shows the "effective attention" considering the full forward pass
- Generally produces more globally coherent patterns than single-layer CLS attention
- Better reflects where information actually flows in the network

**Available for:** DINOv2, DINOv3, MAE, CLIP (Vision Transformers only)

**Research relevance:** Answers "Where does information effectively flow across the entire network?" Useful for understanding the model's overall focus rather than layer-specific patterns.

**Academic basis:** [Abnar & Zuidema (2020)](https://arxiv.org/abs/2005.00928) showed that raw attention weights become increasingly unreliable in deeper layers due to information mixing. Their rollout method achieves **0.71 correlation** with ground-truth importance scores, compared to **0.29 for raw attention**—a significant improvement that justifies our inclusion of this method.

```
Source: src/ssl_attention/attention/rollout.py
```

### 3. Mean Attention

**What it shows:** Average attention received by each patch across all other patches.

**How it works:**
1. Extract attention matrix from a specific layer
2. Average across all rows to see how much each patch is attended to
3. High values indicate "salient" patches that many other patches focus on

**Interpretation:**
- Shows which patches are most attended to overall
- Does not require a CLS token
- Useful for models that use mean pooling instead of CLS pooling

**Available for:** SigLIP (and other models without CLS tokens)

**Research relevance:** SigLIP uses sigmoid loss and mean pooling rather than a CLS token. Mean attention lets us compare its attention patterns to CLS-based models on equal footing.

```
Source: src/ssl_attention/attention/cls_attention.py (extract_mean_attention)
```

### 4. Grad-CAM (Gradient-weighted Class Activation Mapping)

**What it shows:** Which regions most influence the model's prediction, based on gradients.

**How it works:**
1. Perform a forward pass and get class predictions
2. Backpropagate the gradient of a target class through the network
3. Weight feature maps by their gradient importance
4. Produces a spatial map showing which regions drive the prediction

**Interpretation:**
- Unlike attention methods, Grad-CAM is **gradient-based** rather than attention-based
- Shows what matters for the final prediction, not just what the model "looks at"
- Standard technique for CNNs, included as a baseline

**Available for:** ResNet-50 (CNNs without attention mechanisms)

**Research relevance:** Provides a supervised-learning baseline. ResNet-50 was trained on ImageNet with labels, not self-supervised. Comparing its Grad-CAM patterns to SSL attention helps isolate what SSL training paradigms contribute to attention quality.

**Academic basis:** [Selvaraju et al. (2017)](https://arxiv.org/abs/1610.02391) introduced Grad-CAM for CNN interpretability. However, [Chefer et al. (2021)](https://arxiv.org/abs/2012.09838) showed that "Grad-CAM is significantly outperformed by attention maps" for Vision Transformers. This is why we use Grad-CAM only for ResNet-50 (a CNN) and attention-based methods for transformers.

```
Source: src/ssl_attention/attention/gradcam.py
```

---

## How We Measure Alignment

> **See also:** [Project Proposal §3.3 - Attention-Annotation Alignment](../core/project_proposal.md#33-attention-annotation-alignment-primary-metric) and [§5 - Evaluation Plan](../core/project_proposal.md#5-evaluation-plan)

Visualizations are informative but subjective. We need **quantitative metrics** to systematically compare models and answer our research questions.

### IoU (Intersection over Union)

The primary metric measures overlap between model attention and expert bounding boxes:

```
IoU = (Area of Overlap) / (Area of Union)
```

**Process:**
1. Threshold the attention map (e.g., keep only top 10% of values)
2. Convert to a binary mask (1 = high attention, 0 = low)
3. Compare against expert bounding box masks
4. Compute IoU

**Thresholding:** Why "top 10%"?
- Raw attention maps are continuous, not binary
- We threshold at percentiles (90th, 80th, etc.) to focus on the highest-attention regions
- Lower percentiles (e.g., 50th = "top 50%") are more permissive
- The app lets you adjust this to see how IoU changes

**Academic basis:** IoU for attention evaluation was established by [Zhou et al. (2016)](https://arxiv.org/abs/1512.04150) in the original CAM paper, which introduced both IoU and the Pointing Game for evaluating localization. The [ERASER benchmark](https://arxiv.org/abs/1911.03429) (DeYoung et al., 2020) later formalized this as measuring **plausibility**—how well model explanations align with human rationales.

### Baselines

To know if models perform well, we compare against [naive baselines](../core/project_proposal.md#33-attention-annotation-alignment-primary-metric):

| Baseline | Description | Expected IoU |
|----------|-------------|--------------|
| **Random** | Uniform random attention | ~5-10% (proportional to bbox area) |
| **Center** | Gaussian blob in image center | Slightly higher (center bias) |
| **Saliency** | Sobel edge detection | Low-level features |

**Key question:** Do SSL models significantly outperform these baselines? A [negative result](../core/project_proposal.md#8-risks-and-mitigations) (low IoU across all models) is still informative—it tells us what models actually attend to.

### The App Interface

In the metrics dashboard, you can see:
- **Model leaderboard:** Ranked by mean IoU across all images
- **Layer progression:** How IoU changes across transformer layers
- **Per-feature breakdown:** Which architectural elements (windows, arches, towers) each model attends to best

---

## Model-Specific Considerations

> **See also:** [Project Proposal §3.1 - Feature Extraction Pipeline](../core/project_proposal.md#31-feature-extraction-pipeline) for model selection rationale

Each model was chosen to test specific hypotheses about how [training paradigms affect attention](../core/project_proposal.md#4-ablation-studies):
- **DINOv2 vs DINOv3:** Does Gram Anchoring improve attention quality?
- **MAE:** Does reconstruction-based learning produce different attention than discriminative objectives?
- **CLIP vs SigLIP:** Does loss function (softmax vs sigmoid) affect attention alignment?
- **ResNet-50:** Supervised baseline to isolate SSL-specific effects

### Models with CLS Token (DINOv2, DINOv3, MAE, CLIP)

These models support both **CLS Attention** and **Attention Rollout**. You can compare:
- Single-layer attention (how does each layer behave?)
- Accumulated attention (where does information ultimately flow?)

### DINOv2 and DINOv3: Register Tokens

These models include 4 **register tokens** in addition to the CLS token. Registers act as "scratch space" for computation and should be excluded from attention extraction. The app handles this automatically.

**Sequence structure:** `[CLS, REG1, REG2, REG3, REG4, patch1, patch2, ..., patchN]`

**Academic basis:** [Darcet et al. (2024)](https://arxiv.org/abs/2309.16588) discovered that DINOv2 attention maps exhibit artifacts—patches with unusually high attention in low-information areas. Register tokens solve this by providing dedicated storage for global information, producing cleaner attention maps. This is why we use `dinov2-with-registers-base`.

### A Note on "Gram Anchoring" (DINOv3)

You may see references to DINOv3 using "Self-distillation + Gram Anchoring" in the [model table](../core/project_proposal.md#31-feature-extraction-pipeline). This refers to **DINOv3's training objective**, not an attention visualization method available in this app.

Gram Anchoring uses Gram matrix matching during pretraining to encourage consistent style/texture representations. One of our [hypotheses](../core/project_proposal.md#4-ablation-studies) is that this may sharpen attention to semantically meaningful regions compared to DINOv2. We test this by comparing their CLS Attention and Rollout patterns—there is no separate "Gram" attention extraction method.

### DINOv2 vs. DINOv3: Patch Size Difference

| Model | Patch Size | Patches per 224×224 Image |
|-------|------------|---------------------------|
| DINOv2 | 14×14 | 16×16 = 256 patches |
| DINOv3, MAE, CLIP, SigLIP | 16×16 | 14×14 = 196 patches |

This means DINOv2 has finer spatial resolution in its attention maps. For visualization, all maps are upsampled to the full image resolution, making comparison valid.

### SigLIP: No CLS Token

SigLIP uses **mean pooling** instead of a CLS token—it averages all patch representations to get a global representation. Therefore:
- CLS Attention is not available
- Use **Mean Attention** instead

### ResNet-50: No Attention

ResNet is a CNN (Convolutional Neural Network), not a transformer. It has no attention mechanism, so we use:
- **Grad-CAM** for visualization
- 4 stages (layers 0-3) corresponding to ResNet's residual blocks
- 7×7 spatial resolution (49 positions) at the final layer

---

## Interpreting the Visualizations

> **See also:** [Project Proposal §4 - Ablation Studies](../core/project_proposal.md#4-ablation-studies) for the hypotheses we're testing

### Layer Progression

The [layer analysis ablation](../core/project_proposal.md#4-ablation-studies) investigates at what depth expert-aligned attention emerges. Attention patterns change dramatically across layers:

| Layer Depth | Typical Pattern | What It Captures |
|-------------|-----------------|------------------|
| **Early (0-3)** | Diffuse, edge-focused | Low-level features (edges, textures) |
| **Middle (4-8)** | More localized | Mid-level features (shapes, parts) |
| **Late (9-11)** | Concentrated | High-level semantics (objects, regions) |

**Research insight:** Expert-aligned attention typically emerges in later layers, where models develop semantic understanding rather than pixel-level features.

### Color Scale

The app uses the **Turbo colormap** (perceptually uniform rainbow):
- 🔵 **Blue** = Low attention
- 🟢 **Green/Cyan** = Moderate attention
- 🟡 **Yellow** = Higher attention
- 🔴 **Red** = Highest attention

### Comparing Models

When comparing models side-by-side, look for patterns that test our [hypotheses](../core/project_proposal.md#4-ablation-studies):

- **Consistency:** Do all models attend to similar regions?
- **Specificity:** Which model's attention best matches the expert bounding boxes?
- **Failure cases:** When models disagree with experts, what do they attend to instead?
- **Training paradigm effects:** Do self-distillation models (DINO) show more coherent attention than reconstruction models (MAE)?
- **Language supervision:** Do CLIP/SigLIP attend to more "nameable" features?

---

## Technical Reference

### Attention Method Availability by Model

| Model | CLS Attention | Rollout | Mean | Grad-CAM |
|-------|---------------|---------|------|----------|
| DINOv2 | ✅ | ✅ | ❌ | ❌ |
| DINOv3 | ✅ | ✅ | ❌ | ❌ |
| MAE | ✅ | ✅ | ❌ | ❌ |
| CLIP | ✅ | ✅ | ❌ | ❌ |
| SigLIP | ❌ | ❌ | ✅ | ❌ |
| ResNet-50 | ❌ | ❌ | ❌ | ✅ |

### Key Source Files

| File | Description |
|------|-------------|
| `src/ssl_attention/attention/cls_attention.py` | CLS token and mean attention extraction |
| `src/ssl_attention/attention/rollout.py` | Attention rollout implementation |
| `src/ssl_attention/attention/gradcam.py` | Grad-CAM for CNNs |
| `src/ssl_attention/config.py` | Model configurations and method mappings |
| `app/frontend/src/utils/renderHeatmap.ts` | Client-side heatmap rendering |

### Head Fusion Strategies

When combining attention across multiple heads:

| Strategy | Description | Use Case |
|----------|-------------|----------|
| **MEAN** (default) | Average across heads | Democratic, most common |
| **MAX** | Maximum from any head | If any head attends, it counts |
| **MIN** | Minimum across heads | Only where all heads agree |

### API Endpoints

| Endpoint | Description |
|----------|-------------|
| `GET /api/attention/{image_id}/raw` | Raw attention values for client-side rendering |
| `GET /api/attention/{image_id}/heatmap` | Pre-rendered heatmap PNG |
| `POST /api/attention/{image_id}/similarity` | Compute bbox-to-patch similarity |
| `GET /api/metrics/model/{model}` | IoU metrics for a model |

---

## Academic Foundations

This section explains how our approach is grounded in the research literature and how our findings relate to established work.

### Why CLS Attention Works: The DINO Discovery

The original DINO paper made a remarkable discovery that directly motivates our approach:

> "Self-supervised ViT features contain explicit information about the semantic segmentation of an image, which **does not emerge as clearly with supervised ViTs, nor with convnets**."
> — Caron et al. (2021)

This finding—that DINO's self-attention maps naturally resemble segmentation masks without any supervision—is precisely why we can meaningfully compare attention patterns to expert annotations. Our work extends this by asking: *do these emergent attention patterns align with what domain experts consider diagnostic?*

**How our work builds on this:**
- DINO showed attention emerges; we measure whether it aligns with expert knowledge
- DINO used general object segmentation; we use domain-specific architectural features
- DINO was qualitative; we provide quantitative IoU metrics

### Why Attention Rollout Matters

Abnar & Zuidema (2020) demonstrated a critical insight: raw single-layer attention weights are unreliable because information from different tokens becomes increasingly mixed across layers. Their attention rollout method accounts for this by recursively multiplying attention matrices.

Our implementation follows their formulation:
```
R₀ = I (identity matrix)
Rᵢ = normalize((Aᵢ + I) @ Rᵢ₋₁)
```

**Validation from the literature:**
- Raw attention correlation with importance: **0.29 ± 0.39**
- Attention rollout correlation: **0.71 ± 0.24**

This dramatic improvement justifies why we offer rollout as an alternative to single-layer CLS attention.

### The Register Token Solution

Darcet et al. (2024) discovered that DINOv2's attention maps exhibit artifacts—patches with unusually high attention weights in low-information background areas. These artifact tokens were found to contain global image information rather than local patch information.

Their solution—adding learnable "register tokens"—is why we use `facebook/dinov2-with-registers-base`. The registers act as dedicated storage for global information, producing cleaner attention maps for visualization.

**Impact on our work:**
- We specifically use the register-equipped DINOv2 variant
- Our attention extraction correctly excludes register tokens from visualization
- This ensures our IoU measurements reflect meaningful attention, not artifacts

### Why Grad-CAM Has Limitations for Transformers

Chefer et al. (2021) directly compared Grad-CAM to attention-based methods for Vision Transformers:

> "Grad-CAM is significantly outperformed by both attention maps and [our] method, revealing that both methods are more appropriate for ViT interpretability compared to Grad-CAM, which was originally proposed for CNNs."

We include Grad-CAM for ResNet-50 (where it's appropriate) but use attention-based methods for transformers, following this guidance.

### The Attention Explanation Debate: What We Claim (and Don't Claim)

A fundamental question in ML interpretability: **Can we trust attention weights as explanations?** These papers represent opposing views:

| Paper | Core Argument |
|-------|---------------|
| **Jain & Wallace (2019)** | Attention is often uncorrelated with gradient-based importance; different attention patterns can yield identical predictions. Therefore, attention doesn't reliably reveal *why* a model decided something. |
| **Wiegreffe & Pinter (2019)** | Whether attention "explains" depends on your definition. Under certain conditions and tests, attention can be meaningful. It's not a binary yes/no. |
| **DeYoung et al. (2020) - ERASER** | Proposes a crucial distinction: **plausibility** (does the explanation match human intuition?) vs. **faithfulness** (does the explanation reflect what the model actually uses?). |

#### The Plausibility vs. Faithfulness Distinction

This distinction from the ERASER benchmark is critical for understanding what our app measures:

| Concept | Question It Answers | How to Measure | Our Status |
|---------|---------------------|----------------|------------|
| **Plausibility** | Does the model's attention align with where humans think it *should* look? | IoU with expert annotations | ✅ **This is what we measure** |
| **Faithfulness** | Does the model's prediction actually *depend* on the attended regions? | Perturbation tests (mask regions, check if prediction changes) | ⬜ Future work |

#### Why This Matters: Two Scenarios

Consider a model that attends to rose windows when classifying Gothic architecture:

**Scenario A: Plausible AND Faithful**
```
Attention: High on rose windows (IoU = 0.7 with expert bbox) ✓
Perturbation: Masking rose windows changes prediction from "Gothic" to "Uncertain" ✓
Interpretation: Model genuinely uses rose windows for classification
```

**Scenario B: Plausible but NOT Faithful**
```
Attention: High on rose windows (IoU = 0.7 with expert bbox) ✓
Perturbation: Masking rose windows doesn't change prediction ✗
Interpretation: Attention and experts agree, but model actually uses other cues
```

Our current IoU metric cannot distinguish these scenarios. Both would show high alignment scores.

#### What We Are (and Aren't) Claiming

| ❌ What We Do NOT Claim | ✅ What We DO Claim |
|------------------------|---------------------|
| "High attention on buttresses *explains why* the model classified this as Gothic" | "High attention on buttresses *coincides with* where experts say to look for Gothic features" |
| Attention reveals the model's causal reasoning | Attention patterns can be quantitatively compared to expert knowledge |
| Models that attend to expert features are "better" in some absolute sense | Models that attend to expert features are more *aligned with human expertise* |

#### Our Approach: Measuring Coincidence, Not Causation

By measuring **plausibility** (IoU with expert bounding boxes) rather than **faithfulness**, we:

1. **Sidestep the theoretical debate** — We don't need to prove attention is causal
2. **Provide actionable metrics** — IoU is concrete and comparable across models
3. **Answer a useful question** — "Do SSL models develop attention patterns that align with domain expertise?"

This is intellectually honest: we acknowledge that high IoU doesn't prove the model "understands" architectural features the way experts do. But it does tell us whether the model's learned attention patterns happen to focus on the same regions experts consider diagnostic—which is valuable for trust and interpretability.

> **Future work:** Adding faithfulness tests (perturbation-based evaluation) would strengthen our claims by showing whether high-IoU regions are actually used for predictions.

### Evaluation Metrics: Standing on Shoulders

Our evaluation approach draws from established metrics in the literature:

| Metric | Origin | How We Use It |
|--------|--------|---------------|
| **IoU** | Zhou et al. (2016), CAM paper | Primary metric for attention-annotation overlap |
| **Pointing Game** | Zhang et al. (2016) | Complementary binary hit metric |
| **Percentile Thresholding** | Standard practice | We threshold at 90th, 80th, 70th percentiles |
| **Baseline Comparison** | Chefer et al. (2021) | Random, center, saliency baselines |

The ERASER benchmark (DeYoung et al., 2020) established that evaluation should include both **plausibility** (alignment with human rationales) and **faithfulness** (whether highlighted regions actually matter to the model). Our IoU metric measures plausibility; future work could add perturbation-based faithfulness tests.

---

## References

### Core Papers

| Paper | Citation | Relevance |
|-------|----------|-----------|
| **DINO** | Caron, M., et al. (2021). Emerging Properties in Self-Supervised Vision Transformers. *ICCV*. [arXiv:2104.14294](https://arxiv.org/abs/2104.14294) | Discovered emergent segmentation in self-attention |
| **DINOv2** | Oquab, M., et al. (2024). DINOv2: Learning Robust Visual Features without Supervision. *TMLR*. [arXiv:2304.07193](https://arxiv.org/abs/2304.07193) | Foundation model we evaluate |
| **Registers** | Darcet, T., et al. (2024). Vision Transformers Need Registers. *ICLR* (Oral). [arXiv:2309.16588](https://arxiv.org/abs/2309.16588) | Explains artifacts; motivates register-equipped model |
| **Attention Rollout** | Abnar, S. & Zuidema, W. (2020). Quantifying Attention Flow in Transformers. *ACL*. [arXiv:2005.00928](https://arxiv.org/abs/2005.00928) | Our rollout implementation follows this |
| **Transformer Interpretability** | Chefer, H., et al. (2021). Transformer Interpretability Beyond Attention Visualization. *CVPR*. [arXiv:2012.09838](https://arxiv.org/abs/2012.09838) | Compares methods; shows Grad-CAM limitations |
| **Grad-CAM** | Selvaraju, R.R., et al. (2017). Grad-CAM: Visual Explanations from Deep Networks. *ICCV*. [arXiv:1610.02391](https://arxiv.org/abs/1610.02391) | Gradient-based baseline for CNNs |
| **CAM** | Zhou, B., et al. (2016). Learning Deep Features for Discriminative Localization. *CVPR*. [arXiv:1512.04150](https://arxiv.org/abs/1512.04150) | Introduced pointing game evaluation |
| **Specialized Heads** | Voita, E., et al. (2019). Analyzing Multi-Head Self-Attention. *ACL*. [arXiv:1905.09418](https://arxiv.org/abs/1905.09418) | Shows head specialization in transformers |
| **LoRA** | Hu, E.J., et al. (2022). LoRA: Low-Rank Adaptation. *ICLR*. [arXiv:2106.09685](https://arxiv.org/abs/2106.09685) | Parameter-efficient fine-tuning method |
| **LoRA Forgetting** | Biderman, S., et al. (2024). LoRA Learns Less and Forgets Less. *TMLR*. [arXiv:2405.09673](https://arxiv.org/abs/2405.09673) | Forgetting analysis for fine-tuning methods |

### The Attention Debate (Plausibility vs. Faithfulness)

| Paper | Citation | Key Contribution |
|-------|----------|------------------|
| **Attention is not Explanation** | Jain, S. & Wallace, B.C. (2019). *NAACL*. [arXiv:1902.10186](https://arxiv.org/abs/1902.10186) | Showed attention often uncorrelated with importance |
| **Attention is not not Explanation** | Wiegreffe, S. & Pinter, Y. (2019). *EMNLP*. [arXiv:1908.04626](https://arxiv.org/abs/1908.04626) | Argued it depends on definition of "explanation" |
| **ERASER Benchmark** | DeYoung, J., et al. (2020). *ACL*. [arXiv:1911.03429](https://arxiv.org/abs/1911.03429) | Defined plausibility vs. faithfulness—**our IoU measures plausibility** |

### Dataset

| Paper | Citation |
|-------|----------|
| **WikiChurches** | Barz, B. & Denzler, J. (2021). WikiChurches: A Fine-Grained Dataset of Architectural Styles with Real-World Challenges. *NeurIPS Datasets and Benchmarks*. |

---

## Further Reading

### Project Documentation

| Document | Description |
|----------|-------------|
| [Project Proposal](../core/project_proposal.md) | Full research design, hypotheses, and evaluation methodology |
| ↳ [§1 Problem Statement](../core/project_proposal.md#1-problem-statement) | The research questions this app helps answer |
| ↳ [§2 Dataset](../core/project_proposal.md#2-dataset) | WikiChurches dataset details and preprocessing |
| ↳ [§3.1 Models](../core/project_proposal.md#31-feature-extraction-pipeline) | Model selection rationale and training paradigms |
| ↳ [§3.3 IoU Methodology](../core/project_proposal.md#33-attention-annotation-alignment-primary-metric) | How attention-annotation alignment is measured |
| ↳ [§4 Ablation Studies](../core/project_proposal.md#4-ablation-studies) | Hypotheses about model and layer differences |
| ↳ [§9 Expected Contributions](../core/project_proposal.md#9-expected-contributions) | What this research aims to contribute |
| [Implementation Plan](../core/implementation_plan.md) | Technical architecture and phase breakdown |

### Academic References

- **Abnar & Zuidema (2020):** "Quantifying Attention Flow in Transformers" - Original attention rollout paper
- **Chefer et al. (2021):** "Transformer Interpretability Beyond Attention Visualization" - Advanced attribution methods
- **Selvaraju et al. (2017):** "Grad-CAM: Visual Explanations from Deep Networks" - Gradient-based visualization
- **Barz & Denzler (2021):** "WikiChurches" - The dataset providing expert annotations

See the [full references](../core/project_proposal.md#references) in the project proposal.

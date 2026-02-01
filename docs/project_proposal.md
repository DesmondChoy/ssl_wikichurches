# Project Proposal: Do Self-Supervised Vision Models Learn What Experts See?

## Attention Alignment with Human-Annotated Architectural Features

**Course:** ISY5004 Intelligent Sensing Systems Practice Module  
**Team Size:** 3 members  
**Duration:** 6 weeks

---

## 1. Problem Statement

Self-supervised learning (SSL) models like DINOv2, DINOv3, and MAE learn visual representations without labels and achieve strong performance on downstream tasks. However, a fundamental question remains: **do these models attend to the same visual features that human experts consider diagnostic, or do they exploit statistical shortcuts invisible to humans?** Furthermore, when models are fine-tuned for a specific task, **does their attention shift toward expert-identified features, or do they discover alternative discriminative regions?**

A related question emerges: **beyond where models attend, do the learned patch representations encode semantically coherent groupings?** If a model's features for "rose windows" are similar across different images, this suggests the representation has captured meaningful architectural concepts rather than low-level texture statistics.

Existing SSL benchmarks measure classification accuracy but do not explain *which* image regions drive predictions. This matters for trust and deployment: a model that correctly classifies Gothic architecture by attending to flying buttresses is qualitatively different from one that exploits dataset-specific background correlations. This project addresses that gap by quantitatively measuring alignment between SSL attention patterns and expert-annotated "characteristic architectural features."

While existing tools like BertViz and Comet ML visualize transformer attention, they do not quantify whether attention aligns with domain expertise. This gap matters for practitioners fine-tuning foundation models: a model achieving high classification accuracy might exploit background correlations rather than attending to diagnostic features. We address this gap with both a quantitative benchmark and an interactive analysis tool.

### Research Questions and Approaches

| Research Question | Approach | Tool/Method |
|-------------------|----------|-------------|
| Do SSL models attend to the same features human experts consider diagnostic? | Compute IoU between thresholded attention maps and expert bounding boxes across 5 models and 12 layers | Attention heatmap overlay, IoU metrics dashboard, model leaderboard |
| Does fine-tuning shift attention toward expert-identified features? | Compare Δ IoU (fine-tuned − frozen) with paired statistical tests on same images | Frozen vs fine-tuned comparison view, attention shift visualization |
| Do learned representations encode semantically coherent groupings? | Compute cosine similarity between bbox patch features and all image patches | Click-to-compare similarity heatmap, cross-model feature coherence comparison |

---

## 2. Dataset

**Primary Dataset:** WikiChurches (Barz & Denzler, 2021)

- 9,485 images of European church buildings  
- Hierarchical architectural style labels (Gothic, Baroque, Romanesque, etc.)  
- **631 bounding box annotations** identifying characteristic visual features across 139 churches in 4 style categories

**Why WikiChurches:**

- Expert bounding boxes provide rare ground truth for attention evaluation—most vision datasets lack this  
- Fine-grained style distinctions require attending to specific structural elements (pointed arches, rose windows, flying buttresses)  
- Manageable scale for 6-week timeline  
- Hierarchical labels support secondary analysis of concept granularity

**Dataset Structure (Verified):**

- **Annotations:** `building_parts.json` containing 106 feature types and 631 bounding boxes
- **Coordinate format:** Normalized (0-1) as `left, top, width, height`
- **Style distribution:** Gothic (54), Romanesque (49), Baroque (22), Renaissance (17)
- **Style IDs:** Wikidata Q-IDs (Q46261=Romanesque, Q176483=Gothic, Q236122=Renaissance, Q840829=Baroque)

**Preprocessing:**

- Filter to the 139 churches with bounding box annotations for primary attention-alignment analysis
- Use full dataset (filtered to classes with ≥50 samples) for linear probe evaluation
- Standard augmentation (resize, normalize) without heavy augmentation to preserve architectural detail
- Clamp bounding box coordinates to [0,1] (some edge annotations have small negative values)

---

## 3. System Architecture

### 3.1 Feature Extraction Pipeline

Use pretrained weights from HuggingFace/timm. Models are evaluated both **frozen** (Pass 1) and **fine-tuned** (Pass 2) to compare attention patterns before and after task-specific training:

| Model | Parameters | Training Paradigm | Source |
| :---- | :---- | :---- | :---- |
| DINOv2 ViT-B/14 | 86M | Self-distillation (2023) | `facebook/dinov2-with-registers-base` |
| DINOv3 ViT-B/16 | 86M | Self-distillation + Gram Anchoring (2025) | `facebook/dinov3-vitb16-pretrain-lvd1689m` |
| MAE ViT-B/16 | 86M | Masked autoencoding | `facebook/vit-mae-base` |
| CLIP ViT-B/16 | 86M | Contrastive language-image (softmax) | `openai/clip-vit-base-patch16` |
| SigLIP 2 ViT-B/16 | 86M | Contrastive language-image (sigmoid) + dense features | `google/siglip2-base-patch16-224` |

All models use ViT-B architecture for controlled comparison. Feature extraction runs on M4 Pro (MPS backend).

**Model selection rationale:**

- **DINOv2 vs DINOv3:** Direct comparison of Gram Anchoring's effect on attention quality  
- **MAE:** Tests whether pixel-reconstruction objectives produce different attention than discriminative objectives  
- **CLIP vs SigLIP 2:** Tests whether loss function (softmax vs sigmoid) and improved dense features affect attention alignment with experts

### 3.2 Attention Extraction and Visualization

Extract attention maps from each model using multiple methods:

1. **CLS token attention:** Aggregate attention from [CLS] token to spatial patches across heads  
2. **Attention rollout:** Propagate attention through layers to capture indirect dependencies  
3. **GradCAM (baseline):** Gradient-weighted activation maps for comparison with attention-based methods

For each image with expert annotations, generate attention heatmaps at the original image resolution.

### 3.3 Attention-Annotation Alignment (Primary Metric)

Quantify whether high-attention regions overlap with expert-annotated characteristic features:

1. **Threshold attention maps** at various percentiles (top 10%, 20%, 30% of attention mass)  
2. **Compute IoU** between thresholded attention regions and expert bounding boxes  
3. **Compare to baselines:**  
   - Random baseline: IoU expected from uniform random attention  
   - Center baseline: IoU from Gaussian-centered attention (common bias)  
   - Saliency baseline: IoU from low-level saliency (Sobel edge detection)

4. **Pointing game accuracy** - Binary hit metric testing if attention maximum falls within expert bounding box (complementary to IoU)

**Key question:** Do SSL models achieve IoU significantly above these baselines?

### 3.4 Linear Probe Evaluation (Sanity Check)

Train lightweight linear classifiers on frozen features to verify representations are meaningful:

- **Task:** Architectural style classification (4-class for annotated subset; full hierarchy for complete dataset)
- **Metrics:** Top-1 accuracy, per-class F1, confusion matrices
- **Purpose:** Confirm models have learned discriminative features before analyzing attention

### 3.5 Fine-Tuning Analysis (Second Pass)

After evaluating frozen models, fine-tune each backbone on the style classification task:

**Training setup:**
- Dataset: Full 9,485 images with style labels
- Task: 4-class architectural style classification
- Strategy: Freeze backbone initially, then unfreeze last N layers
- Epochs: ~10-20 with early stopping
- Learning rate: 1e-5 to 1e-4 with warmup

**Post-fine-tuning evaluation:**
- Extract attention from fine-tuned models on same 139 annotated images
- Compute IoU with expert bounding boxes
- Compare Δ IoU (fine-tuned - frozen) per model

**Key questions:**
1. Does fine-tuning increase attention-expert alignment?
2. Which SSL paradigm benefits most from task-specific training?
3. Do fine-tuned models attend to features experts didn't annotate?

### 3.6 Interactive Analysis Tool

The primary deliverable is a web-based dashboard for exploring attention-annotation alignment.

**Core Features:**
- Image browser with 139 annotated churches
- Model/method/layer/fine-tuning-state selectors
- Attention heatmap overlay with expert bounding boxes
- IoU score display for current configuration

**Comparison Features:**
- Side-by-side model comparison
- Frozen vs fine-tuned comparison
- Attention shift highlighting

**Metrics Features:**
- Model leaderboard by IoU
- Per-feature-type breakdown
- Statistical comparison view

**Representation Exploration (Enhancement):**
- Click-to-compare: Select an annotated feature to see cosine similarity heatmap
- Visualize which image regions share similar learned representations
- Compare feature coherence across models and layers

---

## 4. Ablation Studies

| Ablation | Variable | Question Addressed |
| :---- | :---- | :---- |
| Model comparison | DINOv2 vs DINOv3 vs MAE vs CLIP vs SigLIP 2 | Do different SSL paradigms produce different attention-expert alignment? |
| Temporal comparison | DINOv2 (2023) vs DINOv3 (2025) | Does Gram Anchoring improve attention quality? |
| Loss function | CLIP (softmax) vs SigLIP 2 (sigmoid) | Does contrastive loss formulation affect attention patterns? |
| Layer analysis | Early vs. middle vs. late ViT layers | At what depth does expert-aligned attention emerge? |
| Attention method | CLS attention vs. rollout vs. GradCAM | Which extraction method best captures expert-relevant regions? |
| Feature category | By annotated feature type (windows, arches, towers, etc.) | Which architectural elements do models attend to most/least? |
| Fine-tuning effect | Frozen vs fine-tuned (same model) | Does task-specific training improve attention-expert alignment? |

**Hypotheses to test:**

1. Self-distillation (DINO) should produce more globally coherent attention than reconstruction (MAE), potentially yielding higher alignment with expert-annotated structural features
2. Language-supervised models (CLIP, SigLIP 2) may attend to "nameable" features more than purely visual SSL models
3. DINOv3's Gram Anchoring may sharpen attention to semantically meaningful regions compared to DINOv2
4. Fine-tuning on style classification will increase IoU with expert bboxes, as experts annotated style-diagnostic features

---

## 5. Evaluation Plan

### Quantitative Metrics

- **Primary:** Mean IoU between attention peaks and expert bounding boxes (per model, per layer)
- **Pointing game:** Binary hit accuracy (max attention in bbox) per model/layer
- **Secondary:** Linear probe accuracy on style classification
- **Fine-tuning effect:** Δ IoU (fine-tuned - frozen) with paired statistical tests
- **Statistical:** Paired t-tests or Wilcoxon signed-rank comparing models; bootstrap confidence intervals for IoU

### Qualitative Analysis

- Attention visualization for representative images showing high/low alignment
- Failure case analysis: When models attend elsewhere, what do they attend to?
- Cross-model comparison: Overlay attention maps from all five models on same images
- Attention shift visualization: Side-by-side frozen vs fine-tuned heatmaps
- Discovery analysis: Do fine-tuned models attend to unannotated-but-relevant features?

**Interactive Demo:** The analysis tool allows users to explore attention across models, layers, and fine-tuning states. Users can compare how different SSL models attend to architectural features, observe attention shift after fine-tuning, and identify which models best align with expert annotations.

### Baselines

- Random attention baseline  
- Center-biased attention baseline  
- Low-level saliency baseline (non-learned)  
- Supervised ResNet-50 (ImageNet pretrained) as non-SSL reference

---

## 6. Alignment with ISY5004 Requirements

| Requirement | How Addressed |
| :---- | :---- |
| Intelligent sensing technique | Self-supervised visual feature learning (DINOv2, DINOv3, MAE, CLIP, SigLIP 2) |
| Image/video analytics | Image classification, attention extraction, feature attribution |
| Dataset handling | Public dataset (WikiChurches) with documented preprocessing |
| Experimental comparison | Ablation across 5 models, 3 attention methods, multiple layers |
| Literature review | SSL methods, attention visualization, interpretability metrics |
| Practical application | Interactive attention analysis tool for evaluating domain-adapted vision models |
| Final developed system | Web-based dashboard with heatmap explorer, model comparison, and metrics views |
| Commercial solutions comparison | BertViz, Comet ML (visualize but don't quantify); ASCENT-ViT (research, not commercial) |

---

## 7. Timeline (6 Weeks)

| Week | Milestone |
| :---- | :---- |
| 1 | Dataset acquisition, preprocessing, bounding box parsing, feature extraction pipeline |
| 2 | Attention extraction implementation (CLS, rollout, GradCAM), baseline computation |
| 3 | IoU computation pipeline, linear probe training, initial results |
| 4 | Ablation experiments (layers, models, feature categories); **begin fine-tuning** |
| 5 | **Fine-tuning analysis**, statistical comparison (frozen vs fine-tuned), visualization |
| 6 | Report writing, recorded presentation |

---

## 8. Risks and Mitigations

| Risk | Mitigation |
| :---- | :---- |
| Small annotation sample (139 churches) limits statistical power | Report confidence intervals; use bootstrap resampling; acknowledge limitations |
| Attention maps may not reflect true model reasoning | Include GradCAM as gradient-based alternative; triangulate with multiple methods |
| Bounding boxes span only 4 style categories | Frame as focused study; note generalization limitations in discussion |
| Compute constraints (no GPU) | Use ViT-B models only; batch feature extraction; MPS acceleration sufficient |
| IoU may be low across all models | Negative result is still publishable—report honestly what models do attend to |
| DINOv3/SigLIP 2 documentation still sparse | Both have HuggingFace weights available; community examples exist |
| Fine-tuning may overfit on small style subset | Use validation split, early stopping, monitor attention maps for degeneration |

---

## 9. Expected Contributions

1. **Quantitative benchmark** for SSL attention alignment with human expert annotations on fine-grained visual recognition
2. **Comparative analysis** of how training paradigm (self-distillation vs. reconstruction vs. contrastive) affects attention patterns
3. **Temporal analysis** comparing DINOv2 (2023) to DINOv3 (2025) on attention quality
4. **Layer-wise analysis** revealing at what depth expert-relevant features emerge
5. **Reproducible codebase** for attention-annotation alignment evaluation
6. **Fine-tuning impact analysis** showing how task-specific training shifts attention patterns relative to expert annotations
7. **Representation coherence analysis** showing whether patch features cluster around semantically similar architectural elements

---

## 10. Commercial and Research Solutions Comparison

**Visualization Tools:**
- BertViz: Interactive attention head visualization for transformers
- Comet ML: MLOps platform with attention panels
- Hugging Face Transformers: Built-in attention output visualization

These tools visualize attention but do not quantify alignment with domain expertise.

**Research on Attention-Annotation Alignment:**
- ASCENT-ViT (2025): Aligns attention with concept annotations using Pixel TPR
- Medical imaging studies: Evaluate attention against radiologist annotations
- Chefer et al. (2021): Transformer interpretability beyond attention visualization

**Knowledge Gap:** No existing work benchmarks multiple SSL paradigms on the same expert-annotated dataset or measures how fine-tuning shifts attention alignment. We fill this gap with a quantitative benchmark and interactive tool.

---

## References

- Barz, B., & Denzler, J. (2021). WikiChurches: A Fine-Grained Dataset of Architectural Styles with Real-World Challenges. *NeurIPS Datasets and Benchmarks*.  
- Chefer, H., Gur, S., & Wolf, L. (2021). Transformer Interpretability Beyond Attention Visualization. *CVPR*.  
- Caron, M., et al. (2021). Emerging Properties in Self-Supervised Vision Transformers. *ICCV*.  
- He, K., et al. (2022). Masked Autoencoders Are Scalable Vision Learners. *CVPR*.  
- Oquab, M., et al. (2023). DINOv2: Learning Robust Visual Features without Supervision. *arXiv:2304.07193*.  
- Radford, A., et al. (2021). Learning Transferable Visual Models From Natural Language Supervision. *ICML*.  
- Simeoni, O., et al. (2025). DINOv3. *arXiv:2508.10104*.  
- Zhai, X., et al. (2023). Sigmoid Loss for Language Image Pre-training. *ICCV*.
- Tschannen, M., et al. (2025). SigLIP 2: A Better Multilingual Vision Language Encoder. *arXiv*.


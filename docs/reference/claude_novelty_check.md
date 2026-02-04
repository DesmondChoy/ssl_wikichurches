# Novelty Assessment: Attention Alignment with Architectural Expertise in Self-Supervised Vision Models

This academic proposal examining whether SSL vision transformers attend to what architectural experts annotate represents a **substantially novel contribution** across all four research questions. While the methodology (IoU measurement, attention maps, fine-tuning analysis) builds on established techniques, the specific combination of SSL models, expert annotations, architectural heritage domain, and comparative fine-tuning analysis fills identifiable gaps in the literature.

## Q1 verdict: Novel application, established methods

The most directly relevant prior work is **Chefer et al. (CVPR 2021)**, which established the IoU and Pointing Game evaluation framework for ViT attention maps against ImageNet segmentation masks. Their follow-up ICCV 2021 paper extended this to bi-modal transformers including CLIP. The **original DINO paper (Caron et al., ICCV 2021)** already demonstrated measuring Jaccard similarity between attention-derived masks and PASCAL VOC12 ground truth, achieving **45.9%** versus only ~23% for supervised ViTs.

The closest contemporary work is **Chung et al. (arXiv 2503.09535, March 2025)**, which computes both IoU and Pointing Game between ViT attention maps and expert bounding boxes on medical imaging datasets. They compare Random, Supervised, DINO, and MAE pretraining—remarkably similar methodology to the proposed work. Their key finding that "DINO pretraining does NOT necessarily outperform supervised/MAE on medical datasets" suggests domain-specific evaluation matters, strengthening the case for architectural heritage studies.

**ASCENT-ViT (IJCAI 2025)** takes a different approach, training attention to align with concept annotations on CUB bird parts and medical kidney tumors. However, this is a training method rather than an evaluation study. **No existing work evaluates attention alignment on WikiChurches or any architectural heritage dataset**, and no work compares DINOv2, DINOv3, MAE, CLIP, and SigLIP 2 in a unified attention-alignment framework. The project's novelty lies in the domain application and model comparison breadth rather than metric innovation.

Methodological recommendations from literature: use both IoU AND Pointing Game (they capture different aspects); consider using **Chefer's gradient-weighted attention rollout** which outperforms raw attention; threshold selection matters (medical paper used top 5% percentile); report per-model results since SSL method effects vary by domain.

## Q2 verdict: Clear methodological gap identified

Literature on fine-tuning effects on attention exists but does not directly measure the proposed Δ IoU methodology. The most relevant paper is **Li et al. (arXiv 2411.09702, November 2024)**, "On the Surprising Effectiveness of Attention Transfer for Vision Transformers," which found that using **only attention patterns from pretraining recovers 77.8% of the performance gap** between training from scratch and full fine-tuning. This demonstrates attention patterns are crucial for transfer learning, but they measured performance gaps rather than alignment with human annotations.

**Park et al. (ICLR 2023)**, "What Do Self-Supervised Vision Transformers Learn?", provides quantitative layer-wise attention analysis comparing contrastive learning (MoCo v3, DINO) versus masked image modeling (MAE, SimMIM). Key finding: CL trains attention for longer-range global patterns while MIM focuses on local texture. They used attention head homogeneity metrics and Fourier analysis—valuable methodology but not expert-annotation alignment.

Several medical imaging papers compare attention-region overlap before/after different pretraining paradigms, but the specific formulation **Δ IoU = IoU(fine-tuned) − IoU(frozen)** for measuring task-induced attention shift toward expert-defined regions does not appear in literature. This represents a clear methodological novelty. The closest precedent is domain adaptation work like **PCaM (arXiv 2506.17232)**, which uses attention guidance loss to direct attention toward task-relevant regions, but this is a training technique rather than evaluation metric.

The project would benefit from citing the attention transfer paper to motivate why attention patterns matter for fine-tuning, then positioning Δ IoU as a way to measure whether this transfer meaningfully aligns with expert knowledge.

## Q3 verdict: Extension of established framework to new domain

**Voita et al. (ACL 2019)**, "Analyzing Multi-Head Self-Attention: Specialized Heads Do the Heavy Lifting," is the foundational paper. They identified three interpretable head types in NMT: positional heads (attend to adjacent tokens), syntactic heads (track dependency relations), and rare word heads. Critically, they validated specialization by comparing against **parsed syntactic ground truth**—methodologically similar to comparing against expert bounding boxes. They found 38 of 48 encoder heads could be pruned with only 0.15 BLEU drop.

For Vision Transformers specifically, **Li et al. (IEEE TVCG 2023)** provides a comprehensive framework including per-head importance via leave-one-out ablation, spatial attention distribution analysis, and autoencoder-based pattern clustering. They distinguish content-agnostic heads (always attend to same position) from content-relevant heads. This is the most methodologically relevant paper for per-head ViT analysis but does not compare against domain-specific expert annotations.

A recent **Neural Networks 2025 paper** on DINO head clusters found three distinct groups: **G1 heads (~20%)** focus on key points within figures resembling human gaze, **G2 heads (~60%)** distribute attention over entire figures with sharp contours, and **G3 heads (~20%)** attend to background. They validated against human eye-tracking data. This provides precedent for categorizing heads by what they attend to, though not against architectural features.

**Basile et al. (arXiv 2510.21518, October 2025)**, "Head Pursuit: Probing Attention Specialization in Multimodal Transformers," uses simultaneous orthogonal matching pursuit to identify heads aligned with interpretable semantic attributes. They found editing ~1% of heads can reliably suppress/enhance targeted concepts. This is methodologically relevant for identifying which heads encode which features.

The proposed per-head IoU against architectural features (windows, portals, towers) extends the Voita paradigm to a new domain with different ground truth. **No existing work computes per-head IoU against expert-annotated domain-specific features** in architectural heritage or any fine-grained recognition context. The main novelty is the domain application and systematic per-feature analysis.

## Q4 verdict: Strongest novelty claim among all questions

**Biderman et al. (TMLR 2024)**, "LoRA Learns Less and Forgets Less," is the primary reference comparing LoRA versus full fine-tuning. Key findings: LoRA substantially underperforms full fine-tuning on target domain but better maintains base model capabilities; full fine-tuning learns perturbations with rank **10-100× greater** than typical LoRA configurations; attention-only LoRA provides no additional benefit over MLP-only LoRA. However, they measured task performance metrics, not attention pattern changes.

**Kumar et al. (ICLR 2022)**, "Fine-Tuning can Distort Pretrained Features," compared Linear Probe versus Full Fine-tuning. Critical finding: fine-tuning achieves 2% higher in-distribution accuracy but **7% lower out-of-distribution accuracy**—fine-tuning distorts pretrained features. They measured feature distortion via cosine similarity and Fisher's discriminant ratio, not attention alignment. Their proposed LP-FT (Linear Probe then Fine-tune) combines benefits of both approaches.

Recent work on LoRA mechanisms includes **CKA similarity heatmaps** comparing SFT versus LoRA representation changes, finding full fine-tuning allows "holistic reshaping" while LoRA "induces more targeted changes and preserves existing structure." The **NTK perspective paper (NeurIPS 2024)** by Tomihari & Sato analyzed training dynamics of LP-FT versus FT versus LoRA, showing LP-FT achieves smaller feature changes from pretrained model.

**No identified work directly compares all three methods (Linear Probe, LoRA, Full Fine-tuning) using attention alignment metrics like IoU.** Existing comparisons focus on performance trade-offs and representation similarity (CKA, cosine), not on whether attention shifts toward expert-defined task-relevant regions. This appears to be the project's strongest novelty claim. The specific research question of whether different fine-tuning strategies differentially affect attention alignment with domain expertise is unexplored.

## WikiChurches dataset and cultural heritage attention studies

The WikiChurches dataset (Barz & Denzler, NeurIPS 2021 Datasets Track) contains **9,485 images** with hierarchical style labels and **631 bounding box annotations** of characteristic visual features for 139 churches. Search results indicate **minimal subsequent usage beyond the original paper**—no follow-up studies from independent research groups, and no attention analysis or interpretability studies using this dataset. This underutilization represents an opportunity.

Related cultural heritage work includes **MonuMAI** (Neurocomputing 2020), which provides expert-annotated architectural elements linked to Hispanic-Muslim, Gothic, Renaissance, and Baroque styles. **HistoNet (2025)** applies hybrid CNN-Transformer-Mamba architectures to architectural heritage with SHAP-based interpretability. The **Brazilian ImageMG dataset** classifies church architectural elements (fronton, door, window, tower) using transfer learning. However, none of these papers analyze SSL model attention against expert annotations.

DINO/DINOv2 have not been systematically applied to architectural heritage attention analysis. The combination of WikiChurches expert bounding boxes with DINO-family attention maps is novel.

## DINO attention properties: What's already known

The DINO literature establishes several key findings relevant to this project. Self-attention heads in the last layer naturally capture **object boundaries without any supervision**, producing attention maps that constitute accurate segmentation masks. DINO achieves **71.4%** on DAVIS-2017 video segmentation without fine-tuning (J&F mean score). The human gaze alignment study found DINO attention closely matches human eye-tracking patterns, with distinct head clusters serving different visual functions.

**DINOv2** combines DINO loss with iBOT Masked Image Modeling for improved patch-level features. PCA of patch features shows semantic part correspondence across poses and styles—potentially valuable for architectural feature analysis. **DINOv3 (2025)** introduces Gram anchoring to address dense feature degradation.

Comparative studies show DINO's attention quality exceeds CLIP's for spatial localization (CLIP lacks spatial awareness with noisy dense features) while CLIP excels at semantic text alignment. **CLIP-DINOiser (ECCV 2024)** uses DINO as teacher to improve CLIP's localization. MAE requires fine-tuning to match DINO's frozen-feature performance but may capture different frequency information.

## Overall novelty assessment by component

| Research Question | Methodology Novelty | Domain Novelty | Gap Strength |
|------------------|---------------------|----------------|--------------|
| Q1: IoU alignment | Low (established) | High (WikiChurches + architecture) | **Moderate** |
| Q2: Δ IoU fine-tuning | **High** (new metric) | High | **Strong** |
| Q3: Per-head specialization | Moderate (extends Voita) | High | **Moderate** |
| Q4: Fine-tuning methods | **High** (no prior comparison) | High | **Strong** |

## Recommended positioning and citations

The project should position itself as bridging three literatures: (1) ViT attention interpretability (Chefer et al., Raghu et al.), (2) SSL pretraining analysis (DINO, Park et al.), and (3) cultural heritage recognition (MonuMAI, WikiChurches). The key differentiator is the expert-annotation-grounded evaluation paradigm.

**Essential citations:** Chefer et al. CVPR 2021 (IoU methodology); Caron et al. ICCV 2021 (DINO attention properties); Barz & Denzler 2021 (WikiChurches); Voita et al. ACL 2019 (head specialization); Biderman et al. 2024 (LoRA vs full FT); Kumar et al. ICLR 2022 (LP vs full FT feature distortion); Chung et al. 2025 (medical attention-expert alignment—closest methodological precedent).

**Methodological enhancements from literature:** Use Chefer's gradient-weighted rollout alongside raw attention; include Pointing Game as complementary metric; consider CKA similarity analysis (Kornblith et al. 2019) for representation changes; apply Li et al.'s head importance framework for head specialization analysis.

## Conclusion: Proceed with confidence

The project fills genuine gaps at the intersection of attention interpretability, fine-tuning analysis, and cultural heritage recognition. Q2 (Δ IoU methodology) and Q4 (three-way fine-tuning comparison) represent the strongest novelty claims. Q1 and Q3 are valuable domain extensions of established methods. The WikiChurches dataset is underutilized and appropriate for this research. No blocking prior work was identified that would undermine the contribution—the closest work (medical imaging attention alignment) actually strengthens the case for domain-specific evaluation studies.
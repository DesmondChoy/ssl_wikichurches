# Project Report Structure Proposal

This document proposes a high-level structure for the final academic project report.

For a LaTeX-oriented version that is easier to move into Overleaf, see
[`project_report_overleaf_skeleton.tex`](./project_report_overleaf_skeleton.tex).

It is intended to help the team organize the report around the current project scope:

- Q1: frozen-model attention alignment
- Q2: fine-tuning effects on attention alignment
- Q3: per-head specialization

It also reflects the current repo state, current experiment artifacts, and professor feedback to focus on:

- a clear opening overview
- calibrated comparisons rather than raw numbers alone
- explanation of why findings are intuitive or surprising
- the current study rather than stretch-goal backlog

## Recommended Shape

The report should read as a research study first, and as a software/system build second.

Suggested main flow:

1. Introduction and motivation
2. Research questions and contributions
3. Related work
4. Dataset and methodology
5. Results
6. Discussion and limitations
7. Conclusion
8. Appendix

## Proposed Sections

### 1. Abstract

- Briefly state the problem: whether SSL vision models attend to the same features that experts mark as diagnostically important.
- State the dataset and task context: WikiChurches, expert bounding boxes, architectural features.
- Summarize the study design at a high level: compare frozen models, analyze fine-tuning shifts, and study per-head specialization.
- Include a placeholder sentence for the main findings once the final results are ready.

### 2. Project-At-A-Glance Overview

- Include a simple infographic or summary block near the start of the report.
- Show the study dimensions in one place:
  - number of models
  - number of fine-tuning strategies
  - number of alignment metrics
  - annotated image count
  - expert bounding box count
  - number of research questions
- Use this section to orient the reader before the detailed methodology.

### 3. Introduction and Motivation

- Explain why classification accuracy alone is not enough for understanding model behavior.
- Introduce the central motivation: models may be correct for the wrong visual reasons.
- Explain why expert-annotated architectural features provide a strong evaluation target.
- End with the central gap this project addresses:
  - cross-model comparison on expert annotations
  - post-fine-tuning attention-shift analysis
  - per-head specialization analysis

### 4. Research Questions and Contributions

- State the research questions clearly and early.
- Keep each question short and easy to scan.

#### 4.1 Q1: Frozen-Model Attention Alignment

- Ask whether frozen SSL and baseline models attend to expert-marked diagnostic regions.
- Frame this as the core benchmark question.

#### 4.2 Q2: Fine-Tuning and Attention Shift

- Ask how attention changes after adaptation.
- Compare Linear Probe, LoRA, and Full fine-tuning.
- Highlight that this section is about both direction and magnitude of attention change.

#### 4.3 Q3: Per-Head Specialization

- Ask whether individual attention heads specialize for different architectural features.
- Keep the scoped study framing clear.

#### 4.4 Contributions

- Summarize the main contributions in a short bullet list.
- Likely include:
  - a multi-metric benchmark
  - calibrated Q1 comparison against baselines
  - Q2 fine-tuning analysis
  - Q3 per-head study
  - reproducible pipeline and interactive analysis tool

### 5. Related Work

- Cover only the most relevant literature clusters.
- Suggested subsections:
  - attention interpretability in vision transformers
  - evaluation against human or expert annotations
  - fine-tuning and representational drift
  - attention-head specialization
  - cultural heritage / architectural recognition context
- Position this project as a domain-grounded evaluation study, not a brand-new interpretability method.

### 6. Dataset and Problem Setup

- Introduce WikiChurches and why it is suitable for this study.
- Describe the annotated subset and style-labeled pool.
- Explain the role of the expert bounding boxes.
- State any key preprocessing or filtering decisions.
- Include important scope and data caveats such as sparse annotation bias.

### 7. Methodology

- This should be one of the main technical sections.
- Focus on the study design and evaluation choices, not only implementation details.

#### 7.1 Models and Attention Extraction

- Briefly describe the compared model families.
- Explain which attention extraction method is used for which model.
- Clarify differences between CLS attention, rollout, mean attention, and Grad-CAM.

#### 7.2 Alignment Metrics

- Explain the purpose of each metric at a high level.
- Distinguish threshold-based metrics from threshold-free metrics.
- Clarify metric direction:
  - higher is better for IoU and Coverage
  - lower is better for MSE, KL, and EMD

#### 7.3 Baselines and Calibration

- Explain why raw continuous-metric numbers need calibration.
- Introduce the naive baselines used for comparison.
- State that this section supports interpretation of Q1 results.

#### 7.4 Fine-Tuning Protocol

- Describe the training/evaluation split logic.
- State clearly that the annotated evaluation images are excluded from the primary train/validation split.
- Summarize the three strategies:
  - Linear Probe
  - LoRA
  - Full fine-tuning
- Explain checkpoint selection at a high level.

#### 7.5 Q3 Per-Head Scope

- Define the narrower scoped study for per-head analysis.
- Explain which models and variants are part of the primary Q3 claim.
- Keep methodology and interpretation guardrails explicit.

#### 7.6 Statistical Analysis

- Briefly describe the paired testing approach.
- Mention multiple-comparison correction and effect sizes.
- Keep this section concise but academically clear.

#### 7.7 Methodological Safeguards and Reproducibility

- Briefly capture the choices that improve credibility of the findings.
- Suggested points:
  - deterministic or stable analysis paths where needed
  - evaluation holdout discipline
  - calibrated baselines
  - explicit experiment provenance
  - artifact-based reporting workflow
- This section helps show that the results are not just visually interesting, but methodologically defensible.

### 8. System and Analysis Interface

- Keep this section shorter than Methods and Results.
- Explain that the software system is the vehicle for running the study and inspecting results.
- Summarize only the major components:
  - metric and cache pipeline
  - fine-tuning experiment pipeline
  - backend and frontend analysis surfaces
- Avoid route-by-route API detail in the main report.

### 9. Results

- This should be the central section of the report.
- Organize it by research question.
- Each subsection should combine quantitative summary with a small amount of qualitative interpretation.

#### 9.1 Q1 Results: Frozen-Model Attention Alignment

- Present the main frozen-model comparison.
- Include calibrated interpretation against baselines.
- Discuss cross-metric agreement and disagreement.
- Use one or two representative examples rather than too many screenshots.

#### 9.2 Q2 Results: Fine-Tuning Effects on Attention

- Make this one of the larger result subsections.
- Compare attention shifts across strategies.
- Discuss which models improve, preserve, or degrade in alignment after fine-tuning.
- Include one or two image-level examples such as shift-map interpretation if useful.

#### 9.3 Q3 Results: Per-Head Specialization

- Present head rankings, head-feature patterns, and a few exemplar observations.
- Keep claims descriptive unless stronger causal evidence is available.
- If findings are still incomplete, keep this subsection appropriately scoped.

### 10. Discussion

- This section should answer the "why" questions raised by the professors.
- It should interpret the results rather than repeat them.

#### 10.1 Intuitive vs Surprising Findings

- Highlight which outcomes matched expectations.
- Highlight which outcomes were counter-intuitive.
- Explain why those results may have occurred.

#### 10.2 What the Results Suggest About Pretraining Objectives

- Compare what different training paradigms appear to encourage.
- Relate observed behavior back to model families where possible.

#### 10.3 Practical Implications

- Explain what these findings could mean for domain adaptation and model selection.
- Keep this brief and grounded in the actual study.

#### 10.4 Limitations and Threats to Validity

- Include:
  - dataset size and domain scope
  - sparse annotation bias
  - the limits of treating attention as explanation
  - any limits in Q3 interpretation

### 11. Conclusion

- Restate the problem and the three research questions.
- Summarize the overall contribution of the study.
- End with a short future-work note only if needed.
- Keep this section concise.

### 12. Appendix

- Put supplementary material here rather than overloading the main narrative.
- Candidates include:
  - extra qualitative examples
  - additional tables
  - implementation details
  - experiment artifact references
  - supplementary figures

## Recommended Emphasis

The report should emphasize:

- research questions
- methodology
- calibrated results
- explanation of findings
- limitations and interpretation

The report should de-emphasize:

- long engineering walkthroughs
- route-by-route app descriptions
- backlog or stretch-goal ideas
- raw result dumps without interpretation

## Practical Writing Guidance

- Keep the main report question-driven rather than feature-driven.
- Use figures that do explanatory work, not just decorative work.
- Prefer a few strong figures with clear captions over many weak figures.
- Make metric direction and interpretation explicit whenever a metric first appears.
- Keep Q3 scoped and honest if the findings are less mature than Q1 and Q2.

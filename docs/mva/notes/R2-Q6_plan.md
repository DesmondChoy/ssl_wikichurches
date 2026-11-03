# Plan: R2-Q6, add a Future Work subsection

Status (2026-11-03): approved prose inserted in `round2_v1024_changes.tex`. Both PDFs build and the rendered subsection is awaiting human review.

## The item

Reviewer 2, question 6 (`R2-Q6`) in the round-1 reply letter requests an explicit discussion of future research addressing the study's limitations.

**Reviewer comment:**

> Although the authors have thoughtfully detailed several limitations of their work, the manuscript currently lacks an explicit discussion on future research perspectives. I suggest including a dedicated "Future Work" subsection that explicitly details how these limitations could be addressed and overcome in subsequent studies, which would add substantial value to the paper.

**TJ's note:** Include two or three recommended future directions.

**Agreed TODO:** Add a Future Work subsection after Limitations, for example generalizing the method to other domains.

## Requested change

Add `\subsection{Future work}` immediately after Section 4.6 (Limitations) and before Section 5 (Conclusions). It is Section 4.7. Use three short paragraphs, each proposing a concrete next step tied to an existing limitation:

| Existing limitation | Future direction |
|---|---|
| Findings evaluated only on architectural images, with untested domain transfer | Expert-annotated benchmarks in medical imaging and pharmaceutical image analysis, such as lesion assessment and image-based drug screening, to test whether model and adaptation patterns persist. |
| Different attention extraction methods and preprocessing geometry | Compare a common extraction method across all the evaluated models, alongside the current model-specific methods, and evaluate aspect-preserving preprocessing with aligned image and annotation coordinates. |
| Spatial alignment does not establish causal reliance | Controlled removal or perturbation of annotated evidence and comparable unannotated regions, measuring prediction changes to test whether aligned regions support the classifier's decisions. |

## Three critical aspects

1. **Connection to the current study:** Each direction addresses a stated limitation, rather than adding unrelated research aims.
2. **Prospective wording:** Describe proposals without suggesting they have been evaluated or will necessarily improve alignment.
3. **Relationship to R2-Q3:** Briefly propose cross-domain validation here, while retaining R2-Q3's detailed explanation of transfer criteria in Limitations. Avoid repeating that passage.

The human narrowed the application examples to medical and pharmaceutical use cases. The cross-domain direction replaces the earlier proposal to expand rare-feature or underrepresented-style samples. Describe these applications as potential future studies requiring task-specific expert annotations, without claiming clinical effectiveness, drug efficacy, or demonstrated commercial value.

## Proposed prose

> One potential direction is to expand the scope beyond architectural datasets. This includes evaluating the approach on expert-annotated datasets in medical imaging and pharmaceutical image analysis. Potential applications include comparing attention with expert-annotated lesions in medical images and regions indicating treatment-related cellular changes in image-based drug screening. These studies would require task-specific expert annotations and would test whether the observed model rankings and adaptation patterns persist across domains with different visual properties.
>
> A second direction is to reduce differences in the evaluation protocol. Applying a common attention extraction method across all the evaluated models, alongside their current methods, would help assess how much the comparisons depend on the readout. Aspect-preserving preprocessing should also be evaluated while keeping the image and annotation coordinates aligned.
>
> Finally, controlled perturbations of expert-annotated regions and comparable unannotated regions could test whether the classifier relies on the evidence with which its attention aligns. Measuring the resulting changes in predictions would complement spatial alignment with evidence about the role of those regions in classification.

## Plan

1. After approval, insert the short subsection in `round2_v1024_changes.tex` only.
2. Build the clean and tracked-changes PDFs with the documented MVA workflow.
3. Inspect the subsection, its numbering, and the transition to Conclusions in both PDFs.
4. Show the rendered change for approval before drafting the A6 reply in chat. Do not edit the reply letter.

## Out of scope

- Performing any of the proposed studies in this revision.
- Adding experiments, numerical results, figures, tables, or unsupported claims.
- Rewriting the existing Limitations or Conclusions sections.

## Implemented change

Section 4.7, “Future work,” contains the three approved paragraphs immediately after Limitations. The wording includes the requested opening and “across all the evaluated models.” No numerical results or experiments were added.

Both PDFs build successfully. The subsection spans clean pages 30–31 and tracked-changes pages 31–32. All four pages were visually inspected, including the heading, paragraph breaks, addition markup, and transition to Conclusions.

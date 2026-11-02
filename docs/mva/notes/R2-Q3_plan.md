# Plan: R2-Q3, generalization beyond the WikiChurches benchmark

Status (2026-11-02): implemented in `round2_v1024_changes.tex`; clean and tracked-changes PDFs build and are awaiting human review.

## The item

Reviewer 2, question 3 (`R2-Q3`) in the round-1 reply letter asks why the study does not include benchmarks from other domains, or asks the authors to explain how the proposed approach might generalize beyond WikiChurches.

**Reviewer comment:**

> Although the authors explicitly acknowledge the size limitations of the WikiChurches benchmark dataset, relying exclusively on a single domain still restricts the generalizability and robustness of the findings. Could the authors clarify why additional benchmarks (e.g., from different domains or with varying visual properties) were not included, or at least discuss how the proposed approach is expected to generalize beyond this specific dataset?

**Agreed TODO:** Elaborate in the discussion how this can be generalized to other domains.

## Current manuscript

Section 4.6, “Limitations,” already says the benchmark is restricted to architectural images and that the findings should not be assumed to generalize directly to medical images or general object-recognition datasets. The reviewer asks for the transfer implications to be clearer. The agreed TODO selects the discussion option; it does not request another benchmark or new experiment.

## Requested change

Add a short clarification in the existing Limitations discussion that separates the reusable measurement procedure from the domain-specific findings:

- The procedure can be applied to other domains when suitable expert-annotated evidence regions are available.
- The regions and visual cues that count as expert evidence are domain-specific, so the method does not establish that the observed model rankings or alignment patterns transfer.
- Transfer should be evaluated on additional expert-annotated benchmarks with different domains and visual properties before making cross-domain claims.

The current results do not establish that model rankings transfer. A future cross-domain study would need to find that attention repeatedly concentrates on each domain's diagnostic regions across images and classes, rather than appearing aligned only because annotated regions are large or common. To claim that the comparative findings transfer, it would also need to observe broadly stable relative model or adaptation patterns across domains, while accounting for differences in annotation size, image scale and visual properties. The exact architectural features, layers or heads need not recur in another domain, as the relevant pattern is attention alignment with that domain's own expert-identified evidence.

Keep this prospective and conditional. Do not claim that generalization has been demonstrated, explain the absence of other datasets with an unsupported rationale, or imply that merely reusing the metrics validates transfer. Keep this addition distinct from the dedicated Future Work subsection requested in R2-Q6, which can later consolidate future directions.

## Plan

1. Edit only the existing Limitations paragraph in `round2_v1024_changes.tex`.
2. Add a concise statement about applying the procedure with domain-specific expert annotations and the need for cross-domain validation.
3. Build the clean and tracked-changes PDFs using the workflow in `docs/mva/AGENTS.md`.
4. Inspect the edited passage in both PDFs, checking that the limitation remains clear and the new text does not overstate transfer.
5. Present the rendered change for approval before drafting the A3 reply. Do not edit the reply letter.

## Implemented change

The Limitations section now distinguishes the reusable measurement procedure from the domain-specific findings. It states that an application to another domain requires expert-identified evidence regions, and that cross-domain validation would need consistent alignment across images and classes, beyond effects from annotation size or frequency. It also states that comparative findings require broadly stable model or adaptation patterns after accounting for differences in annotation size, image scale and visual properties. It does not claim that transfer has been demonstrated.

The clean and tracked-changes PDFs build successfully. The new passage is in the Limitations section on page 30 in the clean PDF; the tracked version marks it as an addition. Visual inspection found the text legible and within the existing paragraph layout.

## Out of scope

- Adding or evaluating another dataset.
- New experiments, metrics, or claims about cross-domain model performance.
- Rewriting the full Limitations section or creating the Future Work subsection before R2-Q6.

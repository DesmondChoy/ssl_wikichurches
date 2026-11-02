# Plan: R2-Q4, summarize implementation settings in a table

Status (2026-11-02): implemented in `round2_v1024_changes.tex`; clean and tracked-changes PDFs build and are awaiting human review. The human chose “training settings only” for table scope.

Punctuation follow-up (2026-11-02): all semicolons in Table 3 were replaced with commas or recast as comma-linked phrases at the human's request. Both PDFs were rebuilt and Table 3 re-inspected.

## The item

Reviewer 2, question 4 (`R2-Q4`) in the round-1 reply letter recommends a dedicated summary table in the “Implementation Details” section with the key hyperparameters and configurations.

**Reviewer comment:**

> To enhance clarity and structure, it is recommended to summarize all key implementation hyperparameters and configurations into a dedicated summary table within the "Implementation Details" section. This will enable readers to easily reference the experimental setup.

**TJ's note:** Move the hyperparameters in the paragraph beginning “All training runs used the AdamW optimizer with class-weighted cross-entropy to…” into a table; this might be clearer.

**Agreed TODO:** Move the hyperparameters in that paragraph into a table.

## Placement and scope

Section 4.1 is `\subsection{Implementation details}`. It already describes the dataset and splits, preprocessing, backbones, frozen classifiers, LoRA and full fine-tuning, training hyperparameters, checkpoint selection, and software/hardware. TJ's note points to the training-settings paragraph, while the reviewer asks for key settings in one place.

Add one concise table in Section 4.1 that summarizes the settings needed to reproduce the downstream training runs. Include the task/split, strategies, optimizer and loss, batch size, epochs and checkpoint rule, learning rates by component/strategy, weight decay, schedule and warmup, gradient clipping, LoRA settings, and training augmentations. Include deterministic validation preprocessing only to clarify its difference from training augmentation. Keep dataset composition, full model descriptions, detailed explanations, and machine/software environment in their existing prose; avoid duplicating them in the table.

## Three critical aspects

1. **Scope:** Keep the table centered on training implementation settings, with the LoRA setup included because it is strategy-specific. Do not turn it into a second copy of all of Section 4.1.
2. **Accuracy:** Copy settings from the existing manuscript and verify them against the relevant configuration/code before moving or condensing text. Preserve distinctions between the classifier-head and backbone learning rates, LoRA and full fine-tuning, and stochastic training augmentation versus deterministic evaluation preprocessing.
3. **Numbering and layout:** The new table will follow the related-work Table 1, so later main-text tables will be renumbered. Keep the table readable in the Springer layout and inspect it in both PDFs. Do not update the reply letter; flag any fixed table references there for the human.

## Plan

1. After approval, verify the settings against the training configuration and existing Section 4.1 prose.
2. Add a compact two-column summary table in Section 4.1. Replace the long optimizer/training-settings passage with a brief lead-in to the table, and remove duplicated values from nearby prose while retaining rationale and workflow details that the table does not cover.
3. Build the clean and tracked-changes PDFs using the workflow in `docs/mva/AGENTS.md`.
4. Inspect the table and nearby prose in both PDFs for legibility, complete settings, and correct numbering/references.
5. Present the rendered change for approval before drafting the A4 reply. Do not edit the reply letter.

## Implemented change

Section 4.1 now has a compact two-column table of training settings, numbered Table 3 in the current clean PDF. It covers the shared train/validation split, task and strategies, optimizer/loss, batch size, epoch/checkpoint rule, learning rates, parameter counts, weight decay, schedule, gradient clipping, LoRA settings, training augmentation and validation preprocessing. The surrounding prose retains the model representations, rationale for the query/value targets, evaluation procedure, and per-strategy selected epochs.

Both PDFs build successfully. The table is on clean page 14 and tracked-changes page 15; visual inspection found it legible and within the page layout. Since it is inserted after existing Tables 1 and 2, following main-text table numbers increase by one. In-text LaTeX references update automatically; fixed table numbers in the reply letter may need manual revision after the final clean build. The reply letter was not edited.

## Out of scope

- New experiments, hyperparameter tuning, or changes to the training configuration.
- New numerical claims not already supported by the manuscript or project configuration.
- Rewriting the rest of Section 4.1 or changing the experimental setup.

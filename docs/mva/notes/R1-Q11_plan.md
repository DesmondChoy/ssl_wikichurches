# Plan: R1-Q11, choice of the LoRA settings

Status (2026-10-25): done. The human approved the manuscript change, and it is committed in `c4b5954`. The A11 reply was given in chat for the human to paste into the letter. Tracked in beads issue `ssl_wikichurches-xwvi`, closed in `0ec334e`.

## The item

Reviewer 1, question 11 (`R1-Q11`) in the round-1 reply letter: *XAI Vision round1 reply to all reviewers 1006.docx* in Google Drive (file ID `1z1jEHklwWdebqoxusVEL7Pt5vTpvtQFf`). See [`AGENTS.md`](../AGENTS.md) for the revision workflow.

**Q11 (reviewer):**

> How did you select the specific LoRA parameters (rank 8, alpha 32) and the target module?

**A11 (TJ):** clarify the details.

**TODO:** Clarify with details.

## Scope

The work is text only. Section 4.1 already stated the values (rank 8, alpha 32, dropout 0.1, no bias, 294,912 adapter parameters, per-model module names) but not why they were used. The change adds that rationale. No code, rerun or rank sweep.

## What the code shows

- The values are the defaults of `FineTuningConfig` and `FineTunableModel` in `src/ssl_attention/evaluation/fine_tuning.py`, set when LoRA was added in `a213ba8` (beads `3km`). The training commands in `docs/reference/fine_tuning_run_matrix.md` do not override them.
- `LORA_TARGET_MODULES` in the same file maps DINOv2 and MAE to `query`/`value` and DINOv3, CLIP, SigLIP and SigLIP2 to `q_proj`/`v_proj`. Both pairs are the query and value projections; the Hugging Face classes name them differently.
- The config saved in each of the six LoRA checkpoints in `outputs/checkpoints/fine_tuning_primary_20260327/` has `lora_rank=8`, `lora_alpha=32`, `lora_dropout=0.1` and `lora_target_modules=None` (so the defaults above). The run manifests do not record LoRA settings, so the checkpoints are the evidence.
- No other LoRA configuration was run.

## Change

All edits are in `round2_v1024_changes.tex`. Four sentences were added to Section 4.1 (Implementation details), after the sentence that names the target modules. They say:

1. The module names differ only because the Hugging Face implementations name the same projections differently.
2. The LoRA settings were fixed before training and not tuned.
3. All six models used the same values, so their LoRA runs differ only in the pretrained backbone.
4. The adapters sit only on the query and value projections, following Hu et al. (`hu2022lora`, already in `xvision.bib`), who found that adapting these two projections performs well for a fixed number of trainable parameters.

The insertion avoids the paragraph's opening sentences, which R1-Q7 rewrote to remove linear probing.

## Build

Both PDFs built. The new text was on clean page 14 and diff page 14 (diff lines 602–608), in Section 4.1.

## Commit

An R1-Q12 session had an uncommitted edit in the same file (line 370) and an uncommitted close entry in `.beads/interactions.jsonl`. Only the R1-Q11 line and the `xwvi` close entry were staged, by writing the index blobs directly, so neither commit contains R1-Q12 work.

## Reply letter (A11)

Given in chat on 2026-10-25 for the human to paste:

> A11: The LoRA settings were not tuned. Rank 8, alpha 32 and dropout 0.1 were fixed before training, and all six models used the same values, so the LoRA runs differ only in the pretrained backbone. Following Hu et al., who found that adapting the query and value projections performs well for a fixed number of trainable parameters, we placed the adapters only on these two projections. The target-module names differ between models (`query`/`value` for DINOv2 and MAE, `q_proj`/`v_proj` for DINOv3, CLIP, SigLIP and SigLIP2) only because the Hugging Face implementations name the same projections differently. Section 4.1 (Implementation details) now states this rationale after the description of the LoRA settings.

## Out of scope

- A rank, alpha or target-module sweep. The TODO asks for clarification, not new experiments.
- A Section 4.6 limitation saying other LoRA settings were not tested. Offered as optional and not requested.
- A hyperparameter summary table. That is R2-Q4.

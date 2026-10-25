# Plan: R1-Q19, make the two model groups unambiguous in the abstract

Status (2026-10-25): done. The human approved the manuscript change, and it is committed in `e337290`. The A19 reply was given in chat for the human to paste into the letter. Tracked in beads issue `ssl_wikichurches-mqgb`, closed in `640cb9f`.

## The item

Reviewer 1, question 19 (`R1-Q19`) in the round-1 reply letter: *XAI Vision round1 reply to all reviewers 1006.docx* in Google Drive (file ID `1z1jEHklwWdebqoxusVEL7Pt5vTpvtQFf`). See [`AGENTS.md`](../AGENTS.md) for the revision workflow.

**Q19 (reviewer):**

> There is a missing comma in the abstract. "...spanning image-only, self-supervised and vision-language models...".

**TODO:** Typo fix.

## Scope

The submitted abstract reads "spanning image-only self-supervised and vision-language models", with no comma after "image-only". The reviewer's quote adds the comma they think is missing.

The six models form two groups, as Section 3.3 and Table 2 state: image-only self-supervised models (DINOv2, DINOv3, MAE) and vision-language models (CLIP, SigLIP, SigLIP2). A comma after "image-only" would describe three groups and suggest that CLIP and the SigLIP models are self-supervised. The manuscript also uses the serial comma throughout, so a three-group list would need a second comma before "and". The fix is therefore to remove the ambiguity, not to add the comma.

## Change

In `round2_v1024_changes.tex`, line 128, "models" is inserted after "self-supervised". The abstract now reads "spanning image-only self-supervised models and vision-language models", which matches the first sentence of Section 3.3. No numbers, figures or other sections changed.

## Build

Both PDFs build with no undefined references. In the clean build at `e337290`, the change is on page 1, lines 033 to 034, and the abstract still ends at line 038. In the tracked-changes PDF, "models" is marked as an insertion on page 1, next to the earlier R1-Q6 and R1-Q7 edits to the same sentence. The diff needs `--append-textcmd=abstract` to mark edits inside `\abstract{...}`.

## Reply letter (A19)

Given in chat on 2026-10-25 for the human to paste:

> A19: The phrase was ambiguous. The six models form two groups: image-only self-supervised models (DINOv2, DINOv3 and MAE) and vision-language models (CLIP, SigLIP and SigLIP2). A comma after "image-only" would describe three groups, so we instead repeated "models" to separate the two groups. The abstract now reads "…spanning image-only self-supervised models and vision-language models…", which matches the wording in Section 3.3.

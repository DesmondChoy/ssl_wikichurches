# Plan: R1-Q12, checkpoint selection across the 3 epochs

Status (2026-10-25): done. The human approved the manuscript change, and it is committed in `a113ec1`. The A12 reply was given in chat for the human to paste into the letter. Tracked in beads issue `ssl_wikichurches-nwpn`, closed in `5afd61c`.

## The item

Reviewer 1, question 12 (`R1-Q12`) in the round-1 reply letter: *XAI Vision round1 reply to all reviewers 1006.docx* in Google Drive (file ID `1z1jEHklwWdebqoxusVEL7Pt5vTpvtQFf`). See [`AGENTS.md`](../AGENTS.md) for the revision workflow.

**Q12 (reviewer):**

> When doing the fine-tuning of the models, you said that you keep the checkpoint of highest accuracy, did you mean it was selected from the 3 training epochs? It should be stated better in the paper.

**A12 (TJ):** no note.

**TODO:** Clarify with details.

## Scope

The work is text only. The submitted manuscript already said the selected epochs "came from epochs 1 through 3", but it did not say that validation ran after every epoch. Only that sentence in Section 4.1 was replaced. No code or results changed.

## What the code shows

- `FineTuner.train` (`src/ssl_attention/evaluation/fine_tuning.py`) evaluates the 917-image validation split after every epoch. It saves the checkpoint when `val_acc > best_val_acc`, so each run keeps its best epoch out of 3. A tie keeps the earlier epoch.
- All 18 manifests in `outputs/results/experiments/fine_tuning_primary_20260327/manifests/` record `classification_validation_accuracy` as the selection metric. Their `selected_epoch` values match `best_epoch` in `fine_tuning_results.json`.
- The 139 expert-annotated images are not part of the validation split and play no part in selection.

Selected epochs:

| Strategy | Epoch 1 | Epoch 2 | Epoch 3 |
|---|---|---|---|
| Full fine-tuning | – | – | all 6 |
| LoRA | – | DINOv2, SigLIP | DINOv3, MAE, CLIP, SigLIP2 |
| Frozen-backbone classifier | DINOv3 | CLIP, MAE | DINOv2, SigLIP, SigLIP2 |

DINOv3's frozen-backbone classifier reached the same validation accuracy (0.735) at epochs 1 and 3, and the strict `>` kept epoch 1.

## Change

All edits are in `round2_v1024_changes.tex`, in the Section 4.1 checkpoint paragraph. One sentence was replaced.

- Before: "The selected epochs came from epochs 1 through 3, depending on the model and adaptation strategy, but every run was trained for the same maximum of 3 epochs."
- After: "Every run was trained for 3 epochs and evaluated on the validation split after each epoch, and the selected checkpoint is the one from the epoch with the highest validation accuracy. This was epoch 3 for all six full fine-tuning runs and for four of the six LoRA runs, epoch 2 for DINOv2 and SigLIP under LoRA, and epoch 1, 2, or 3 for the frozen-backbone classifiers, depending on the model."

The new text uses R1-Q7's term "frozen-backbone classifier". R1-Q7 (`279087a`) rewrote the neighbouring sentences in the same paragraph after this change was committed, and the paragraph now reads consistently.

## Build

Both PDFs build. The changed text is on clean page 14, in Section 4.1 (Implementation details).

## Reply letter (A12)

Given in chat on 2026-10-25 for the human to paste:

> A12: Yes. Each run was trained for 3 epochs, and after every epoch we measured classification accuracy on the shared validation split (917 images not in the expert-annotated subset). We kept the checkpoint from the epoch with the highest validation accuracy, so each run's checkpoint is the best of its 3 epochs. The 139 expert-annotated images were not used for this selection. Epoch 3 was selected for all six full fine-tuning runs and for four of the six LoRA runs. Epoch 2 was selected for DINOv2 and SigLIP under LoRA. For the classifiers trained on the frozen backbones, the selected epoch ranges from 1 to 3 depending on the model. Section 4.1 (Implementation details) now states this procedure and the selected epochs.

## Out of scope

- Per-run validation accuracies or learning curves. The reviewer asked only how the checkpoint was chosen.

# Plan: R1-Q6, the SigLIP mean-attention proxy as a confound

Status (2026-10-25): done. The human approved the manuscript changes, and they are committed in `c05ba48`. The human is pasting the A6 reply into the letter (see "Reply letter (A6)"). Tracked in beads issue `ssl_wikichurches-u5he`.

## The item

Reviewer 1, question 6 (`R1-Q6`) in the round-1 reply letter: *XAI Vision round1 reply to all reviewers 1006.docx* in Google Drive (file ID `1z1jEHklwWdebqoxusVEL7Pt5vTpvtQFf`). See [`AGENTS.md`](../AGENTS.md) for the revision workflow.

**Q6 (reviewer):**

> The comparison with SigLIP and SigLIP2 introduces a confounding variable because they rely on a mean-attention proxy rather than the native [CLS]-token or rollout pathways used for the other models; therefore, the comparisons are not completely between the same values. This discrepancy should be clearly highlighted in the comparative analysis and mentioned in the limitations.

**A6 (TJ):** We don't need to change the experiment, but we need to highlight this as a limitation.

**TODO:** Highlight this limitation.

## Scope

The work is text only, with no new experiments, as TJ asked. The reviewer asks for the point in two places, while the TODO names only one.

| Comment or TODO | Change |
|---|---|
| "clearly highlighted in the comparative analysis" | One paragraph in Section 4.2 (RQ1), after the Table 3 discussion |
| "mentioned in the limitations" / TODO | Two sentences in Section 4.6 |
| The comment's "[CLS]-token or rollout pathways" | One necessary correctness fix in Section 3.3 and Table 2 (below) |

## What the code shows

- `DEFAULT_METHOD` (`src/ssl_attention/config.py`) is `cls` for DINOv2, DINOv3, MAE and CLIP and `mean` for SigLIP and SigLIP2. Table 3, Table 5, Figure 3 (`generate_paper_layer_iou.py`) and Figure 9 all read the default method, so every comparison across the two groups compares [CLS] attention with the proxy.
- `extract_mean_attention` (`src/ssl_attention/attention/cls_attention.py`) returns the mean over query rows of the head-fused attention matrix, which is the average attention each patch receives from all patches. [CLS] attention is a single query row, so the two maps measure different quantities.
- The SigLIP wrappers return the 12 encoder layers' attention, each of shape `(B, H, 196, 196)`. The attention-pooling head that produces `pooler_output` is not among them, so the proxy never reads the pathway that the SigLIP linear probe uses.
- Comparisons within a model use one method throughout: across layers, frozen against adapted (ΔIoU), Table 7's within-run tests, and SigLIP against SigLIP2. The confound does not affect them.

## Rollout: computed, never reported

The comment's "rollout pathways" comes from the baseline's Section 3.3, which said "We compare three attention extraction methods", and from Table 2's rollout row. No reported result uses rollout.

- The project proposal (`docs/core/project_proposal.md`, Ablations) planned "CLS vs rollout vs mean vs Grad-CAM" to verify findings are not artifacts of the extraction method.
- Rollout was added to precompute and the web app in February 2026 (beads `4bs`), and to the metrics cache in `aa25be0` (beads `3f7`).
- The course reports (`docs/core/project_report_final.md`, `docs/final_report/ISY5004_report_final.tex`) used [CLS] as the default method and listed rollout as an "other supported method". The head analysis excluded rollout "because it aggregates layers". `git log -G rollout` across every report and manuscript version shows descriptions only, never a rollout result.
- The journal conversion (`4ab2279`) introduced the "three methods" sentence.
- No beads issue or note records a decision to drop rollout.

`outputs/cache/metrics.db` holds rollout rows for all 16 [CLS]-model variants at all 12 layers, regenerated after the R1-Q4 crop fix. Frozen IoU@90:

| Model | [CLS], best layer | Rollout, best layer | Rollout, layer 11 |
|---|---:|---:|---:|
| DINOv3 | 0.133 (L11) | 0.061 (L0) | 0.060 |
| DINOv2 | 0.118 (L11) | 0.050 (L11) | 0.050 |
| MAE | 0.070 (L11) | 0.043 (L2) | 0.038 |
| CLIP | 0.051 (L0) | 0.051 (L0) | 0.006 |

Rollout at layer 0 equals [CLS] attention at layer 0 for all four models, because removing the [CLS] token and min-max normalizing cancel the identity term and the ½ scale. For DINOv3 and CLIP, rollout's best layer is therefore single-layer attention.

## Changes

All edits are in `round2_v1024_changes.tex`. No numbers changed.

### 1. Necessary correctness fix: Section 3.3 and Table 2

**The error.** Baseline line 260 says the study compares three extraction methods, and Table 2 (baseline line 269) lists rollout. No result reports rollout, and no model is scored with more than one method.

**Why it affects the requested result.** A6 must say the four [CLS] models use [CLS] attention, not "[CLS] or rollout". Left as written, Section 3.3 would contradict the reply.

**The fix.** The paragraph now says each model is scored with one of two methods, defines the proxy, and states that the proxy does not use the SigLIP pooling head. The rollout row is deleted from Table 2. The Related Work mentions of rollout (lines 166, 168 and 179) describe other work and stay. The Abnar and Zuidema citation is still used at line 166.

### 2. Comparative analysis (Section 4.2)

A new paragraph after the Table 3 discussion (baseline line 391) says the cross-model comparisons mix the two methods. It names the affected comparisons (Tables 3 and 5, Figures 3 and 9) and the lower SigLIP MSE. It also says comparisons within a model and between SigLIP and SigLIP2 are not affected.

### 3. Limitations (Section 4.6)

Two sentences in the "metric design choices" point (baseline line 505): differences between the two groups may partly reflect the method rather than the model. We did not evaluate a common method across all six models, such as applying the proxy to the [CLS] models or extracting the SigLIP pooling-head attention.

## Build

Both PDFs build. The checked pages are clean pages 8, 15 and 27–28 and diff pages 8–9, 15 and 27.

latexdiff 1.4.0 comments out the `&` and `\\` of a deleted table row, so the struck Rollout text spilled into the Mean row and widened Table 2 past the margin. A `perl` line in the AGENTS.md render recipe restores them. It uses `\x5c` for the backslash so it runs the same in bash, zsh and fish, and its output was checked byte for byte in zsh and fish.

## Reply letter (A6)

On 2026-10-25 the text below was typed into the letter through Claude in Chrome, as a green "A6:" paragraph between TJ's A6 note and the TODO line, matching A4 and A5. The editor kept it after a reload but reported "Unsaved changes to Drive", and the Drive connector still read the old A6 from the stored `.docx`, as happened with R1-Q3. The human then chose to paste A6 in themselves. If the typed copy is still in the Google Docs editor, remove it so the letter does not hold A6 twice.

> A6: We agree. SigLIP and SigLIP2 have no [CLS] token, so their attention maps come from a mean-attention proxy, whereas DINOv2, DINOv3, MAE and CLIP use [CLS]-token attention in all reported results. Section 3.3 and Table 2 now state that each model is evaluated with one of these two methods and define the proxy. The earlier text also listed attention rollout, which no reported result uses, so we removed it. Section 4.2 now states that comparisons between the two groups (Tables 3 and 5, Figures 3 and 9) mix the two methods and may partly reflect the method rather than the model, whereas comparisons within a model and between SigLIP and SigLIP2 are not affected. Section 4.6 lists this as a limitation. The experiments are unchanged.

## Commit

Other sessions edited `round2_v1024_changes.tex` at the same time (the Figure 9 caption, the line-367 training paragraph, and an RQ3 sentence on the same line as the R1-Q6 limitation). The staged copy was built from `HEAD` with only the four R1-Q6 edits applied, so `c05ba48` holds no text from those sessions and their edits stay in the working tree.

## Out of scope

- Reporting rollout as an extraction-method check. The data is in `metrics.db`, but no reviewer asked for it, and the layer-0 result above would need explaining.
- A like-for-like run (the proxy on the [CLS] models, or SigLIP pooling-head attention). TJ ruled out new experiments, and the cache stores `mean` only for SigLIP.
- A note on Table 3's caption. The Section 4.2 paragraph already covers it.

# Plan: R1-Q14, why CLIP's dominant heads move to the final layer after adaptation

Status (2026-10-25): done. The human approved the manuscript change, and it is committed in `83e9264`. The A14 reply was given in chat for the human to paste into the letter. Tracked in beads issue `ssl_wikichurches-f0cz`, closed in `54f23d0`. The 2026-10-31 fresh-eyes review corrected four points in the paragraph and replaced A14 (see below).

## The item

Reviewer 1, question 14 (`R1-Q14`) in the round-1 reply letter: *XAI Vision round1 reply to all reviewers 1006.docx* in Google Drive (file ID `1z1jEHklwWdebqoxusVEL7Pt5vTpvtQFf`). See [`AGENTS.md`](../AGENTS.md) for the revision workflow.

**Q14 (reviewer):**

> CLIP undergoes the most dramatic attention reorganization after fine-tuning and LoRA, shifting its dominant heads from early to late layers. Provide a deeper theoretical discussion on why this model exhibits such reorganization. Why do you think it's happening?

**TODO:** Extend discussion.

## Scope

The work is text only. The human chose to use only numbers already in `outputs/cache/metrics.db` (`aggregate_metrics` and `head_summary_metrics`, IoU@90, `cls` method). No code, results or figures changed.

## What the data show

- Frozen CLIP's head-averaged IoU@90 is 0.041 at layer 4, 0.012 at layer 5 and at most 0.023 from there to layer 11.
- `layer4/head5` is 0.068 frozen, 0.067 under LoRA and 0.069 under full fine-tuning. The best layer-11 head is 0.041 frozen (H9), 0.099 under LoRA (H7) and 0.130 under full fine-tuning (H3). The early heads stay put, and the last layer gains.
- Under LoRA, layers 0–9 change by at most 0.002, layer 10 rises from 0.017 to 0.025 and layer 11 from 0.023 to 0.053. Full fine-tuning raises them to 0.056 and 0.091.
- MAE also gains most in its later layers, but its best frozen layer is already layer 11. DINOv2 and DINOv3 keep their best heads in layers 10 and 11.

## Change

One paragraph was added to Section 4.5 of `round2_v1024_changes.tex`, after the R1-Q13 paragraph. It argues:

1. CLIP models with this layer layout develop a few high-norm background tokens after layer 6, and these tokens draw the [CLS] attention away from the annotated regions. This may explain the low alignment in the later layers, but not the exact location of the drop at layer 5.
2. Adaptation acts mainly on the last layers, which produce most of CLIP's output and feed the classifier.
3. The other [CLS]-attention models don't reorganize, because their best frozen heads are already late. DINOv2 and DINOv3 are trained with register tokens.
4. Caveat: token norms are not reported, so the mechanism remains a hypothesis.

Four bib entries were added to `xvision.bib`. Their metadata was checked on arXiv and the ICLR and CVPR proceedings pages:

- `Darcet24Registers` (ICLR 2024)
- `Jiang25Registers` (NeurIPS 2025)
- `Yang25ResCLIP` (CVPR 2025)
- `Gandelsman24CLIPDecomp` (ICLR 2024)

The paragraph also cites the existing `Lan24ClearCLIP`.

## Build

Both PDFs build with no undefined citations. The paragraph is on clean page 28 and diff page 29, and the new references are [46]–[49].

There is a known latexdiff quirk. With this paragraph in place, the unchanged end of the R1-Q7 "One additional observation" sentence renders as deleted and re-added on diff page 28. The Gandelsman sentence triggers it. Rewording and `MINWORDSBLOCK` did not remove it. The paragraph opens with "Frozen CLIP's head-averaged..." so that a stray deletion of "SigLIP-family models" does not land inside the new paragraph.

## Unreported check

A token-norm check was run outside the repo on `features.h5` and `attention_viz.h5`. It is not in the manuscript, because `AGENTS.md` requires reported numbers to come from committed code.

- In frozen CLIP, the highest patch norm is 11 times the median at layer 6, peaks at 19 times at layer 7 and falls to 5 times at layer 11 (means over the 139 images; corrected on 2026-10-31 from "11–15 times").
- In 100% of images, the [CLS] attention peak falls on one of the three highest-norm patches. At layer 4 this happens in 1% of images.
- DINOv3 and MAE show no outliers, with ratios of 1.1–1.3.
- After full fine-tuning, the share at layer 11 drops to 83%.

If a reviewer asks for direct evidence, this check could become a small committed script.

## Reply letter (A14)

The human asked for a shorter reply than the first draft. Given in chat on 2026-10-25 for the human to paste:

> A14: We added a paragraph to Section 4.5. Frozen CLIP's alignment drops sharply after layer 4 (Figure 3). Studies of the same ViT-B/16 architecture link this to a few high-norm background tokens that form in the middle layers and draw the [CLS] token's attention, so CLIP's best frozen heads lie in early layers. Adaptation mainly changes the last layers, which produce most of CLIP's output and feed our classifier. The early heads keep their frozen scores, while the best layer-11 head rises from 0.041 to 0.130 under full fine-tuning (Figure 8). The reorganization therefore comes from a stronger final layer, not from attention moving from early to late heads.

## Fresh-eyes review (2026-10-31)

A review of all Reviewer 1 items found four errors in the paragraph. The human chose a wording-only fix, so no results or code changed and the token-norm check stays unreported.

| Problem | Before | After |
|---|---|---|
| The text denied a check that had been run (see "Unreported check") | "We did not measure token norms, so this explanation is consistent with our results but not shown by them." | "We do not report token norms here, so this explanation remains a hypothesis." |
| The mechanism starts after the drop it was used to explain | "from the middle of the network onward" | "after layer 6", plus "Because these tokens appear after the drop at layer 5, the mechanism may explain CLIP's low alignment in the later layers but not the exact location of the drop." |
| The early heads do move under full fine-tuning | "Adaptation leaves the early heads unchanged and strengthens a final layer..." | "Adaptation changes the early heads much less than it strengthens a final layer..." |
| ResCLIP does not support the link | "This fits the location of frozen CLIP's three best heads in layers 2 to 4." | Deleted |

"Same architecture" also became "same layer layout", because Jiang et al. use LAION-2B weights rather than OpenAI's.

**Timing evidence.** Frozen CLIP on the 139 images, from read-only checks of `features.h5` and `attention_viz.h5`. The scripts are not committed, so these numbers are not in the paper.

| Layer | Mean IoU@90 | Max/median patch norm | [CLS] peak on a top-3-norm patch | Top-10% attention share |
|---|---:|---:|---:|---:|
| 4 | 0.041 | 2.1 | 1% | 0.224 |
| 5 | 0.012 | 6.0 | 3% | 0.238 |
| 6 | 0.008 | 11.4 | 100% | 0.387 |
| 7 | 0.009 | 19.2 | 100% | 0.607 |

Alignment falls at layer 5, one layer before the attention sink appears. Jiang et al. report outliers after the layer 6 MLP in OpenCLIP.

**Early heads.** `layer4/head5` stays between 0.067 and 0.069, as the paragraph says. Under LoRA, 99% of the heads in layers 0 to 6 change by at most 0.01. Under full fine-tuning, L2/H2 falls from 0.064 to 0.041, L5/H9 rises from 0.047 to 0.085 and L6/H10 rises from 0.034 to 0.070.

**ResCLIP.** Its layer table favours later windows (2→5 scores below 4→7), and it studies patch-to-patch attention, not [CLS] attention.

**Reply letter.** This A14 replaces the one above. Given in chat on 2026-10-31 for the human to paste:

> A14: We added a paragraph to Section 4.5. Frozen CLIP's alignment drops sharply after layer 4 (Figure 3). Studies of the same ViT-B/16 layout find that a few high-norm background tokens form from layer 6 and draw the [CLS] token's attention, which may explain the low alignment in the later layers, although the first drop comes slightly earlier. Adaptation mainly changes the last layers, which produce most of CLIP's output and feed our classifier. The dominant frozen head, layer 4 head 5, keeps its score, while the best layer-11 head rises from 0.041 to 0.130 under full fine-tuning (Figure 8). The reorganization therefore comes from a stronger final layer, not from attention moving from early to late heads.

**Left unchanged.**

- "Which take over the role of the high-norm tokens". Darcet et al. report these tokens in DINOv2 only from ViT-Large upward, so registers may not be why our ViT-B models lack them.
- "Training-free dense-prediction methods for CLIP correspondingly modify only its final block". This generalizes from the two cited methods.

## Out of scope

- The RQ2 bullet in Section 4.5 still describes CLIP's adaptation-driven reorganization, which is an RQ3 result. It was left unchanged.
- For R1-Q15, the same unreported check shows that the share of DINOv3's top-1% attention pixels falling outside the annotations drops from 0.95 at layer 6 to 0.74 at layer 7. This is a possible lead for the layer-6 rise.

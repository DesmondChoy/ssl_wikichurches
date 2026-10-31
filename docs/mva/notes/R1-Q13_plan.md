# Plan: R1-Q13, language supervision versus image-only SSL, and DINO versus MAE

Status (2026-10-25): done. The human approved the manuscript change, and it is committed in `3a6a468`. The A13 reply was given in chat for the human to paste into the letter. Tracked in beads issue `ssl_wikichurches-x1tf`, closed in `5afd61c`. The 2026-10-31 fresh-eyes review removed one clause from the paragraph (see below).

## The item

Reviewer 1, question 13 (`R1-Q13`) in the round-1 reply letter: *XAI Vision round1 reply to all reviewers 1006.docx* in Google Drive (file ID `1z1jEHklwWdebqoxusVEL7Pt5vTpvtQFf`). See [`AGENTS.md`](../AGENTS.md) for the revision workflow.

**Q13 (reviewer):**

> I would like the authors to expand the discussion: How do you think that the inclusion of language supervision compares to image-only self-supervised models? Within the image-only category, delve deeper into the differences between DINO and MAE paradigms.

**TODO:** Extend the discussion here with one more paragraph should be ok.

## Scope

The work is text only. Every number in the new paragraph already appears in Table 3 (frozen alignment) or Table 7 (frozen and adapted accuracy). No code, results or figures changed.

## Change

One paragraph was added to Section 4.5 of `round2_v1024_changes.tex`, after the "One additional observation" paragraph. It makes these points:

1. Language versus image-only. The comparison uses only the four [CLS]-attention models, which avoids the SigLIP mean-attention confound (R1-Q6). All three image-only models align better than CLIP, but MAE leads CLIP by much less than the DINO models lead MAE. The language models still classify nearly as well as DINO, so language supervision helps classification more than alignment.
2. DINO versus MAE. DINO's loss trains the [CLS] output, and DINOv2 and DINOv3 add patch-level losses. MAE's loss ignores its [CLS] token, which He et al. call an "auxiliary dummy token". The Hugging Face MAE decoder also drops it before the loss (`logits[:, 1:, :]`).
3. Caveat. MAE's low alignment may partly reflect its untrained [CLS] readout rather than missing spatial information.
4. Conclusion. Which tokens the training objective acts on matters more than whether it uses language.

Three bib entries were added to `xvision.bib`. Their metadata was checked on Crossref, the NeurIPS proceedings page and arXiv:

- `Lan24ClearCLIP` (ECCV 2024): CLIP's contrastive training emphasizes global features over local discriminability.
- `Bolya25PE` (NeurIPS 2025): the best embeddings of a contrastive vision-language encoder lie in intermediate layers.
- `Przewiezlikowski25ICCV` (ICCV 2025): attention in masked image models is spread almost uniformly, so the [cls] token does not focus on relevant patches.

The first author's name uses the IEEE spelling without the ogonek, because `sn-jnl` does not load T1 font encoding and `\k{e}` would fail under pdfLaTeX.

## Build

Both PDFs build with no undefined citations. In the clean build at `54f23d0`, the paragraph is on pages 27-28.

## Reply letter (A13)

The human asked for a shorter reply than the first draft. Given in chat on 2026-10-25 for the human to paste:

> A13: We added a paragraph to Section 4.5. To avoid the SigLIP proxy confound (see our reply to Q6), we compare only the four models that use [CLS]-token attention. All three image-only models align better than CLIP, but MAE only slightly, whereas the DINO models lead clearly (Table 3). The language-supervised models still classify nearly as well as DINO and far better than MAE (Table 7), so language supervision helps classification more than alignment. Within the image-only models, DINO's loss trains both the [CLS] token and the patch tokens, while MAE's loss ignores its [CLS] token, which fits MAE's weaker alignment. We conclude that which tokens the training objective acts on matters more than whether it uses language, and we cite three supporting studies.

## Fresh-eyes review (2026-10-31)

A review of all Reviewer 1 items checked the paragraph's citations against the papers. One clause was removed from Section 4.5. The reply letter is unchanged.

**The error.** The paragraph said Bolya et al.'s finding "is consistent with the decline of CLIP's alignment after its earliest layers (Figure 3)". Their best layers are deep, for example layer 47 of 50 for frozen visual Q&A, so the finding concerns layers shortly before the output, not the earliest ones. Figure 3 does not show that pattern either, because CLIP's layer 11 scores higher than its layers 5 to 10.

**The change.** The sentence now ends at "rather than at the output." The Bolya citation stays, so the three new references and A13's "we cite three supporting studies" still hold.

**Left unchanged.**

- He et al. state that the loss is computed on masked patches, not that it ignores the [CLS] output. The manuscript wording is still accurate, because the decoder drops that token's output before the loss.
- The clauses on Lan et al. and Przewiezlikowski et al. closely follow those papers' abstracts. They are short, so they were not reworded.

## Out of scope

- CLIP's head reorganization after adaptation and the register and artifact explanation belong to R1-Q14.
- The layer-6 rise of the DINO models belongs to R1-Q15.
- The `Tschannen25SigLIP2` bib entry lists an author who is not on the arXiv paper and omits several who are. It predates this item and was left unchanged.

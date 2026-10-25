# Plan: R1-Q7, remove the linear probe from the adaptation analysis

Status (2026-10-25): implemented locally under the human handoff, awaiting independent review. The Holm fix was committed before analysis (`51209c8`); the MPS rerun exactly preserves the LoRA and full results at `aa7267f` and produces the three planned significance flips. The human authorized five local commit units, including the separate attention-map count fix, and replaced the approval gates. Do not push or close `ssl_wikichurches-0bpu` or `ssl_wikichurches-3pzx`; verification and deviations are recorded in the issue notes and final handoff.

Line numbers below refer to `round2_v1024_changes.tex` at `aa7267f`. Edits from other items can move them, so the quoted text is what locates each change. R1-Q8, R1-Q9 and R1-Q10 have since committed edits (through `d3a6740`) that move every line after 297 down by three, so line 571 is now 574. All quoted text is still present at `d3a6740`, and those edits added no new mention of linear probing. R1-Q11 (`c4b5954`) and R1-Q12 (`a113ec1`) then edited lines 366 and 370, the two Section 4.1 paragraphs this plan changes. All quoted text is still present at `a113ec1`, and R1-Q12's new sentence already calls these runs "frozen-backbone classifiers".

## The item

Reviewer 1, question 7 (`R1-Q7`) in the round-1 reply letter: *XAI Vision round1 reply to all reviewers 1006.docx* in Google Drive (file ID `1z1jEHklwWdebqoxusVEL7Pt5vTpvtQFf`). See [`AGENTS.md`](../AGENTS.md) for the revision workflow.

**Q7 (reviewer):**

> Regarding the linear probe strategy, since the backbone remains entirely frozen, the attention maps inherently will not change from the frozen baseline (Figure 6 confirms this). I do not see the point of including this analysis, as the model does not change.

**A7 (TJ):** If the encoder is frozen, remove this experiment. Otherwise, clarify whether attention maps can change during this tuning, in which case it cannot be called a linear probe.

**TODO:** Double check if linear probe adaptation changes attention maps. If not, remove this linear probe experiment, and update all figures/diagrams.

The reviewer's "Figure 6" is `667_fig5.png` (`figure RQ3 heatmap`), the RQ3 heatmap whose linear-probe column is all zeros.

## Decisions

| Decision | Outcome |
|---|---|
| Does the linear probe change attention maps? | No. See "What the code and data show". |
| Remove the experiment or keep it? | Remove linear probing as an RQ3 adaptation strategy. Keep its trained head, because it supplies the frozen models' classification predictions in Section 4.4.1. Agreed in the [R1-Q3 plan](R1-Q3_plan.md#dependency-on-r1-q7). |
| Holm family for Figures 6 and 7 | Option A, chosen by the human on 2026-10-25. Correct over the 12 reported LoRA and full comparisons per metric. |
| Fix "three types of attention maps" in the same work? | Yes, bundled (human, 2026-10-25). It stays a separate issue (`ssl_wikichurches-3pzx`) and a separate commit. See "Separate item: attention-map count". |
| Figure 1 source | None is available (human, 2026-10-25). Redraw it with a new script. See "Figure 1 generator". |
| Figure filenames | Figures 1, 6 and 7 get new files, referenced only by the revised manuscript. The baseline keeps its images, and `latexdiff` frames the new figures in the tracked-changes PDF. |

## What the code and data show

The linear probe does not change attention maps. Three independent checks agree.

1. **Code.** With `freeze_backbone=True`, `FineTunableModel._freeze_backbone` sets `requires_grad = False` on every backbone parameter, and `get_optimizer_param_groups` adds no backbone group. All six backbones are ViTs with LayerNorm and no BatchNorm, so training mode updates no running statistics.
2. **Weights.** In all six linear-probe checkpoints in `outputs/checkpoints/fine_tuning_primary_20260327/`, every backbone tensor is bit-identical to the pretrained weights. The check below found 0 differing tensors for DINOv2 (224 tensors), DINOv3 (211), MAE (198), CLIP (199), SigLIP (208) and SigLIP2 (208).
3. **Alignment.** In `outputs/results/experiments/fine_tuning_primary_20260327/q2_metrics_analysis.json`, all 54 linear-probe rows (6 models, 5 metrics, 5 IoU percentiles) have a per-image change of exactly 0.0 on all 139 images. This analysis runs the frozen model and the linear-probe checkpoint as separate forward passes.

Weight check (run from the repository root with `uv run --no-sync python`):

```python
from pathlib import Path
import torch
from ssl_attention.evaluation.fine_tuning import FineTunableModel

for name in ["dinov2", "dinov3", "mae", "clip", "siglip", "siglip2"]:
    path = Path("outputs/checkpoints/fine_tuning_primary_20260327") / f"{name}_linear_probe_finetuned.pt"
    state = torch.load(path, map_location="cpu", weights_only=False)["model_state_dict"]
    saved = {k.removeprefix("backbone."): v for k, v in state.items() if k.startswith("backbone.")}
    pretrained = FineTunableModel(name, freeze_backbone=True, device=torch.device("cpu")).backbone.state_dict()
    assert saved.keys() == pretrained.keys()
    print(name, sum(not torch.equal(saved[k], pretrained[k]) for k in saved))
```

## Requested changes

Each edit removes linear probing as an adaptation strategy. Quoted text is the current wording. An ellipsis (...) marks text left out of a quote. In the Figure 1 row, "Linear probe" is a label inside the image, not LaTeX text.

| Line | Location | Current text | New text |
|---|---|---|---|
| 128 | Abstract | "three adaptation strategies" | "two adaptation strategies" |
| 151 | Introduction, RQ3 question | "how do different adaptation strategies, including linear probing, \textit{Low-Rank Adaptation} (LoRA) \cite{hu2022lora}, and full fine-tuning, affect attention alignment?" | "how do \textit{Low-Rank Adaptation} (LoRA) \cite{hu2022lora} and full fine-tuning affect attention alignment?" |
| 157 | Introduction, third contribution | "three adaptation strategies" | "two adaptation strategies" |
| 231 | Section 3.1, RQ3 motivation | "linear probing preserves the backbone, LoRA introduces parameter-efficient updates, and full fine-tuning ..." | "LoRA introduces parameter-efficient updates, and full fine-tuning ..." |
| 233 | Section 3.1, closing paragraph | "three adaptation strategies" | "two adaptation strategies" |
| 237–238 | Figure 1 and caption | "Linear probe" leaf; "3 model adaptation strategies" | Redrawn figure without the leaf, saved as `framework_v3.png` and referenced by `\includegraphics` (see "Figure 1 generator"); "2 model adaptation strategies" |
| 274 | Section 3.3, adaptation paragraph | "We compare three model adaptation strategies. First, in the \textit{linear probe setting}, ..." | Replacement paragraph below |
| 363 | Section 4.1, adaptation settings | "We evaluated three model adaptation settings, including linear probing, LoRA adaptation, and full fine-tuning. In the linear-probe setting, the pretrained backbone was frozen, and a newly initialized four-class linear classifier was trained on top of the frozen representation." | Replacement sentences below. The [CLS]/pooled sentences and everything from "In the LoRA setting" on stay. |
| 367 | Section 4.1, checkpoint paragraph | "The experiment consists of 18 runs, corresponding to six models and three adaptation strategies." | "The experiment consists of 18 training runs: for each of the six models, one frozen-backbone classifier, one LoRA run, and one full fine-tuning run." |
| 367 | Section 4.1, checkpoint paragraph | "For the frozen models, these predictions come from the linear-probe checkpoint, whose backbone is frozen and therefore produces the frozen model's attention." | "For the frozen models, these predictions come from the frozen-backbone classifier." |
| 475 | Section 4.4, opening | "We compare three adaptation strategies: linear probing, LoRA, and full fine-tuning." | "We compare LoRA and full fine-tuning, each against the frozen model." |
| 479–480 | Figure 6 and caption | Linear-probe column; "for the linear probe, LoRA, and full fine-tuning"; "significant after Holm correction." | Regenerated figure saved as `RQ3_adaptation_heatmap.png` and referenced by `\includegraphics`; "for LoRA and full fine-tuning"; "significant after Holm correction across the 12 model-strategy comparisons for each metric." |
| 485–486 | Figure 7 and caption | "significant after Holm correction." | Regenerated figure saved as `RQ3_adaptation_forest.png` and referenced by `\includegraphics`; "significant after Holm correction across the 12 model-strategy comparisons for each metric." |
| 515 | Figure 8 caption | "The linear probe is omitted because it freezes the backbone and therefore does not create a new attention-head configuration." | Delete the sentence |
| 571 | Section 4.5, after the RQ3 bullet | "Two additional observations clarify the interpretation. First, the linear probe serves as an attention-change control. ... Second, stronger task-conditioned adaptation helps most ..." | "One additional observation clarifies the interpretation. Stronger task-conditioned adaptation helps most ..." |

These stay as they are. Table 1 ("Ours" has a check mark under adaptation). Table 6 (full fine-tuning only, no significance marks). Section 4.4.1, Figure 9, Table 7 and Table A1, which already call these runs "Frozen" and count 18 runs as Frozen, LoRA and Full. The statistical-analysis paragraph (line 346), which says only that Holm adjusts "within each metric family". The training-hyperparameter paragraph (line 365), whose settings apply to all 18 training runs, the frozen-backbone classifiers included.

### Replacement text

Section 3.3 (line 274):

> We compare two model adaptation strategies against the frozen pretrained model. In the \textit{LoRA setting}, low-rank trainable adapters are inserted into selected layers while the original backbone parameters remain frozen, enabling parameter-efficient adaptation. In the \textit{full fine-tuning setting}, all model parameters are updated on the downstream task, allowing the entire visual representation to be optimized for architectural style recognition. To obtain the frozen model's classification predictions, which Section~\ref{section RQ3 correctness} relates to attention alignment, we also train a classification head on its frozen representation. This leaves the backbone and its attention maps unchanged, so we do not treat it as an adaptation strategy.

Section 4.1 (line 363, first two sentences):

> We evaluated two model adaptation settings, LoRA adaptation and full fine-tuning. To obtain classification predictions from each frozen model, we trained a newly initialized four-class linear classifier on top of the frozen backbone. [The two existing sentences on the [CLS]-token and pooled representations stay here.] The backbone weights in these checkpoints are identical to the pretrained weights, and their attention maps give the same per-image alignment scores as the frozen model on all 139 images.

## Necessary correctness fix: the Holm family (Option A)

**The error.** `apply_holm_correction` in `experiments/scripts/analyze_q2_metrics.py` puts every analyzed row for a metric into one family. Each family therefore holds 18 tests, 6 models times linear probe, LoRA and full. The six linear-probe tests have p = 1 by construction. After the linear probe leaves the paper, the asterisks in Figures 6 and 7 would come from a correction over tests the paper does not report.

**The effect.** Re-applying Holm over the 12 LoRA and full rows, with the repository's `multiple_comparison_correction` and the stored raw p-values, changes 3 of 108 significance flags.

| Family | Run | Mean Δ | Holm p (18 tests) | Holm p (12 tests) | In the paper? |
|---|---|---:|---:|---:|---|
| Coverage | DINOv2 LoRA | +0.0021 | 0.132 | 0.019 | Yes, Figures 6 and 7 gain an asterisk |
| IoU@70 | DINOv3 Full | −0.0049 | 0.070 | 0.028 | No |
| IoU@80 | DINOv2 LoRA | +0.0048 | 0.107 | 0.043 | No |

No IoU@90, MSE, KL or EMD flag changes. No sentence in the manuscript claims DINOv2 LoRA Coverage is not significant. The Section 4.4 sentence "The only significant aggregate IoU@90 change among them is a modest LoRA gain for DINOv2" stays true.

## Separate item: attention-map count

This is not part of R1-Q7, and R1-Q7 is complete without it. R1-Q6 is complete too. Its issue is closed, its A6 reply is in the letter, and [its plan](R1-Q6_plan.md) lists its edits to Section 3.3, Table 2, Section 4.2 and Section 4.6. This item is a leftover inconsistency that those edits did not reach. It is tracked as `ssl_wikichurches-3pzx`. The human chose to bundle it into this work (2026-10-25).

**The error.** R1-Q6 removed rollout from Section 3.3 and Table 2, and the A6 reply tells the reviewer each model uses one of two methods. Four places outside R1-Q6's edits still say three: the abstract (line 128), the third contribution (157), the Section 3.1 closing paragraph (233) and the Figure 1 caption (238, "3 types of attention maps"). Figure 1 still shows an "Attention rollout" leaf. All four phrases are still present at `d3a6740`.

**Why it is bundled.** R1-Q7 rewrites the same four sentences and redraws Figure 1. Without this fix, the new Figure 1 generator would draw the rollout leaf that Table 2 no longer lists, and "three types of attention maps" would sit next to the corrected strategy count.

**The fix.** Change "three types of attention maps" to "two types of attention maps", and "3 types" to "2 types" in the caption. Leave the "Attention rollout" leaf out of the Figure 1 data. Commit the text change on its own as a `docs:` commit that says it finishes the rollout removal from R1-Q6, and close `ssl_wikichurches-3pzx`.

**Reply letter.** No change. A6 already tells the reviewer that the paper uses two methods. This fix makes the abstract, the introduction, Section 3.1 and Figure 1 agree with it.

## Figure 1 generator

No source file exists for `framework_v2.png` (human, 2026-10-25). The PNG (1275 × 598 px) has no generator in the repository and no metadata naming the tool that made it. It arrived with the submission commit `4ab2279`. A new script, `experiments/scripts/generate_paper_framework.py`, redraws it in matplotlib so that later edits can be rerun.

**What it draws.** The same mind map, with colours sampled from the current PNG.

| Branch | Colour | Position | Leaves, top to bottom |
|---|---|---|---|
| Model | `#4070FF` | Right, top | DINOv2, DINOv3; MAE; CLIP; SigLIP, SigLIP2 |
| Attention map | `#A179F2` | Right, middle | [CLS] attention; Mean attention |
| Dataset | `#F7578C` | Right, bottom | WikiChurches |
| Adaptation | `#FFAA00` | Left, top | LoRA; Full fine-tuning |
| Metric | `#FF4019` | Left, bottom | Coverage; Earth Mover's distance; Intersection over Union; Kullback-Leibler divergence; Mean squared error |

- "Comparative study" sits in the centre in bold. All text is `#3D474D`. Branch labels are bold and leaves are regular weight.
- Each branch is a curve from the centre to its bold label, then a hollow junction circle, then a bracket that fans out to its leaves, as in the original. Dataset has one leaf, so its leaf continues straight from the junction.
- Compared with the current PNG, two leaves are gone. "Linear probe" under Adaptation goes for R1-Q7, and "Attention rollout" under Attention map goes for the attention-map count fix.
- The current PNG spells SigLIP as "SIgLIP", with a capital I, in both places. The redraw spells it "SigLIP, SigLIP2".
- The original font is unknown. Use a sans-serif font that matplotlib finds on this machine, with DejaVu Sans as the fallback.

**Data shape.** One frozen dataclass per branch holds its label, colour, side, junction position and leaves, and one module-level tuple collects the five branches. The drawing code loops over that tuple, so removing a leaf is a one-line data change. Positions use the original's pixel grid, with the origin at the top left, so the new layout can be checked against the old PNG.

**Structure.** Follow the other `generate_paper_*.py` scripts: a module docstring with usage, an `--output` option, and a default output path in `docs/mva/`.

**Output.** Write `docs/mva/framework_v3.png` and point the revised manuscript's `\includegraphics` at it. `framework_v2.png` stays unchanged, because the baseline `round2_v1024.tex` references it. The new filename also makes `latexdiff` frame the figure in the tracked-changes PDF (its default `--graphics-markup=new-only` frames figures whose `\includegraphics` argument changed). Keep the 1275:598 aspect ratio so the page layout does not move, and render at twice the original pixel size (2550 × 1196) for print.

**Check.** Render a side-by-side image of the old and new figures for the review, and look at the Figure 1 page in the clean PDF.

## Implementation

Steps 1 to 4 do not touch `round2_v1024_changes.tex`, so they can run while another item edits it. Step 5 waits until no other session has uncommitted edits to that file.

### 1. Holm family code and tests

- In `analyze_q2_metrics.py`, add a module constant holding the strategies left out of the Holm family, `frozenset({FineTuningStrategy.LINEAR_PROBE.value})`. A one-line comment gives the reason, which is that these strategies do not change the backbone.
- `apply_holm_correction` skips those rows. A skipped row keeps its raw `p_value` and gets `corrected_p_value=None`, `significant=False`, and `None` for the three `correction_*` fields. The web app already accepts a null `corrected_p_value` (`app/backend/schemas/models.py`, `FrozenVsFinetuned.tsx`).
- The linear-probe rows stay in the JSON, because the web app's Q2 summary filters rows by strategy.
- Two existing tests in `tests/test_evaluation/test_q2_delta_analysis.py` assert the old behavior and must change.
  - `test_apply_holm_correction_includes_all_strategies_in_same_family` becomes a test that the linear-probe row is left out. Give the linear-probe row p = 1.0, as in the real data. With p = 0.001, as now, the LoRA and full corrected values come out the same whether or not the linear probe is in the family, so the test could not detect the defect. With p = 1.0, LoRA's corrected p is 0.02 when the linear probe is excluded and 0.03 when it is included.
  - `test_save_results_serializes_correction_family_metadata` checks the family fields on the LoRA row (family size 1) and checks that the linear-probe row serializes them as `null`.
- The tests assert corrected p-values and family sizes, not only `significant`. The flags alone cannot tell the two families apart.
- Show the human the code diff and the test output. After approval, commit this unit before step 2. The analysis records `git rev-parse HEAD` as `analysis_git_commit_sha` (`get_git_commit_sha` in `src/ssl_attention/evaluation/fine_tuning_artifacts.py`), so the rerun must start from a commit that contains the new correction code.

### 2. Rerun the Q2 analysis

First confirm with `git status` that nothing under `src/`, `experiments/` or `tests/` is uncommitted. Then run `uv run --no-sync python experiments/scripts/analyze_q2_metrics.py` in the background without `--experiment-id`, so it uses the active experiment and does not refresh the training provenance. It rewrites `q2_metrics_analysis.json` and its compatibility copy `q2_delta_iou_analysis.json`. It also records the new analysis paths and commit SHA in `run_matrix.json` and rewrites `outputs/results/active_experiment.json`. All four files are tracked. Then check the following against the committed files at `aa7267f`.

- LoRA and full rows: `p_value`, `mean_delta` and `per_image_deltas` should match. The code path and checkpoints are unchanged, but the script picks the device and precision automatically and does not force deterministic kernels. An exact match is likely on the same machine but not guaranteed. If any value differs, stop and report how large the differences are before regenerating figures, because manuscript numbers taken from this file would need rechecking.
- Linear-probe rows: every per-image delta is 0.0, `corrected_p_value` is null, and `significant` is false.
- Significance flags: exactly the three changes in the table above.
- R1-Q3: copy `classification_predictions.json` to a scratch directory outside the repository, then run `HF_HUB_OFFLINE=1 uv run --no-sync python experiments/scripts/analyze_accuracy_alignment.py --from-predictions --output-dir <scratch>`. It must exit 0, which means checks (b) and (c) pass against the new Q2 file. Its statistics must equal the tracked `accuracy_alignment_analysis.json` apart from `analysis_git_commit_sha` and `timestamp`. The tracked file is not rewritten. The script saves its output before running the checks, so read the exit status, not just the file.

### 3. Regenerate Figures 6 and 7

- `generate_paper_fig5.py` writes Figure 6. Set `STRATEGIES = ['lora', 'full']`, remove the `linear_probe` entry from `STRATEGY_LABELS`, and change `OUTPUT_PATH` to `docs/mva/RQ3_adaptation_heatmap.png`. Check that the six panels still read well with two columns instead of three.
- `generate_paper_fig6.py` writes Figure 7. It already plots LoRA and full only. Change `OUTPUT_PATH` to `docs/mva/RQ3_adaptation_forest.png` and rerun it.
- `667_fig5.png` and `714_fig6.png` stay, because the baseline references them. Restoring their submitted versions is `ssl_wikichurches-yvm1`.
- Check both PNGs. DINOv2 LoRA Coverage carries an asterisk in each, and no other asterisk moves.

### 4. Figure 1

Write `generate_paper_framework.py` as specified in "Figure 1 generator", render the figure, and make the side-by-side comparison. This step does not depend on steps 1 to 3.

### 5. Manuscript text

Apply the table in "Requested changes", including the three new `\includegraphics` filenames. Then apply the attention-map count fix. Before editing, run `git status` to confirm no other session has uncommitted edits to `round2_v1024_changes.tex`.

### 6. Build and check

- Build the clean and tracked-changes PDFs with the commands in [`AGENTS.md`](../AGENTS.md), including the `perl` line for deleted table rows.
- Check the pages with the abstract, the introduction, Figures 1, 6, 7 and 8, Section 3.1, Section 3.3, Section 4.1 and Section 4.5 in both PDFs. In the tracked-changes PDF, Figures 1, 6 and 7 carry the `latexdiff` blue frame.
- Quality gate: `uv run ruff check .`, `uv run mypy`, `uv run pytest tests/test_evaluation/test_q2_delta_analysis.py`, then the full `uv run pytest`, then the `/quality` skill.

### 7. Review, commit and reply

Show the human the regenerated results and figures, the Figure 1 side-by-side, the manuscript diff and the PDF pages, and wait for approval. Unit 1 is already committed in step 1. Commit the rest in order, verifying each unit before the next.

1. `fix:` the Holm family code and its tests (committed in step 1).
2. `fix:` the regenerated Q2 results (both JSON files, `run_matrix.json`, `active_experiment.json`), `generate_paper_fig5.py`, `generate_paper_fig6.py`, `RQ3_adaptation_heatmap.png` and `RQ3_adaptation_forest.png`.
3. `feat:` `generate_paper_framework.py` and `framework_v3.png`, which drops both the "Linear probe" and "Attention rollout" leaves.
4. `docs:` the R1-Q7 manuscript edits and this plan's status.
5. `docs:` the attention-map count fix (four text changes), which closes `ssl_wikichurches-3pzx`.

After approval, give the A7 reply in the chat for the human to paste, with section and figure numbers taken from the latest clean PDF build, then close `ssl_wikichurches-0bpu` (`AGENTS.md`, revision workflow step 5). Draft:

> A7: We agree. Linear probing trains only the classification head, so the backbone and its attention maps do not change. We confirmed this. The backbone weights in all six linear-probe checkpoints are identical to the pretrained weights, and every per-image alignment difference from the frozen model is zero for all five metrics. We therefore removed linear probing as an adaptation strategy. RQ3 now compares LoRA and full fine-tuning against the frozen model (Sections 3.3 and 4.4, Figures 1, 6 and 7). Removing the six linear-probe comparisons from the Holm correction makes one further change significant, a small Coverage gain for DINOv2 under LoRA (Figures 6 and 7). The linear classifier is kept only to provide the frozen models' classification predictions for Section 4.4.1, which relates classification correctness to attention alignment. Section 4.1 states that it leaves the attention maps unchanged.

## Out of scope

- Retraining or new adaptation strategies.
- The pairwise `strategy_comparisons` block in the Q2 JSON. The paper does not report it, and its within-model Holm families are unchanged here.
- `generate_run_matrix_figures.py` and `generate_slide_images.py`, which also read `significant` from the Q2 JSON. Their outputs are course and slide figures, not the manuscript.
- Removing the linear-probe variant from the web app.
- Restoring the five baseline figures that R1-Q3 and R1-Q4 overwrote in place (`633_fig4.png`, `667_fig5.png`, `714_fig6.png`, `787_tbl7_enlarge.png`, `RQ2_layer_iou_new.png`). Tracked as `ssl_wikichurches-yvm1`.

## Codex review (2026-10-25)

Codex reviewed the plan read-only, comparing against `aa7267f`. All eight findings were accepted.

| # | Severity | Finding | Change to the plan |
|---|---|---|---|
| 1 | Blocker | Overwriting `framework_v2.png`, `667_fig5.png` and `714_fig6.png` changes how the baseline renders | Figures 1, 6 and 7 get new filenames. The same problem already affects five figures from R1-Q3 and R1-Q4, filed as `ssl_wikichurches-yvm1`. |
| 2 | Should-fix | The rerun would record the commit SHA of the code before the Holm change | Step 1 is committed before the rerun. |
| 3 | Should-fix | The R1-Q3 replay rewrites a tracked file and saves before checking, and the commit list left out the figure scripts | The replay writes to a scratch directory and the exit status is checked. Both figure scripts are in commit unit 2. |
| 4 | Should-fix | "Three types of attention maps" is not needed for R1-Q7's result | Moved to a separate fix for the attention-map count (`ssl_wikichurches-3pzx`). The human then chose to bundle it as a separate commit. |
| 5 | Should-fix | A7 said the classifier is kept "only to report" accuracy, but its predictions also define the correct and wrong groups in Section 4.4.1, Table 7 and Table A1 | A7 and the Section 3.3 and 4.1 text now say "classification predictions". |
| 6 | Nit | An exact match on rerun is not guaranteed | Step 2 qualifies the expectation and says what to do on a mismatch. |
| 7 | Nit | The 18-test Holm p for DINOv2 LoRA Coverage is 0.13247, which rounds to 0.132, not 0.133 | Fixed. |
| 8 | Nit | Quotes containing "..." are abridged, and "Linear probe" is image text | A note under the edit table says so. |

Codex independently confirmed the rest. The weight check found 0 differing tensors in all six models. All 7,506 per-image deltas in the 54 linear-probe rows are exactly 0. The stored Holm corrections reproduce, and removing the linear probe flips exactly the three listed flags. A linear-probe p of 1.0 in the test fixture detects wrongful inclusion. Null correction fields are safe for the backend, frontend and figure scripts. The edit table covers every manuscript mention. The four rewritten Q2 files are listed correctly. The R1-Q3 checks pass with the proposed correction.

## Found while planning

- In `outputs/cache/metrics.db`, SigLIP linear-probe rows differ from frozen SigLIP in 11,655 of 11,676 rows. The largest per-image IoU difference is 4e-4, and the layer means agree to within 3e-6. The checkpoint weights are identical, so the cause is in cache generation (guess: separate cache runs with nondeterministic kernels). The paper does not use these rows. The other five models match bit for bit.

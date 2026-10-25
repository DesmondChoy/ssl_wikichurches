# AGENTS.md

## Scope
- Applies to `docs/mva/`: the LaTeX source of the journal manuscript *Where Do Vision Foundation Models Look? A Study of Attention Alignment in Image Recognition* (Springer Nature `sn-jnl` template, `sn-mathphys-num` reference style).

## Context
- The paper was submitted and the reviewers returned comments. This folder is where we prepare the revision.
- Reviewer comments and our replies are in the reply letter: [XAI Vision round1 reply to all reviewers 1006.docx](https://docs.google.com/document/d/1z1jEHklwWdebqoxusVEL7Pt5vTpvtQFf/edit) (Drive file ID `1z1jEHklwWdebqoxusVEL7Pt5vTpvtQFf`).
- The letter has three parts: Reply to Editor, Reviewer 1 (Q1–Q19) and Reviewer 2 (Q1–Q6). Each item has a reviewer comment (`Qn`), our reply (`An`, sometimes holding a note from TJ, the corresponding author) and an agreed action (`TODO`). Refer to items as `R1-Q7`, `R2-Q5`.
- Section, figure, table and line references in the comments point to `round2_v1024.tex`.
- The editor asks for accurate results, rewritten overstated conclusions and fully explained limitations. The resubmission includes a clean PDF, a tracked-changes PDF and the point-by-point reply letter.

## Keep revision scope focused
- Keep reviews, plans and edits focused on the requested reviewer item, TJ's note, the agreed `TODO` and the user's instructions. Choose the smallest change that directly answers the request.
- Clearly distinguish **Requested changes**, **Necessary correctness fixes** and **Optional / out of scope**. Tie requested changes to the actual request. For each necessary correctness fix, identify the concrete error and explain why it affects the requested result.
- Keep optional suggestions brief and separate from the required work. Do not add them to implementation steps, acceptance criteria or completion conditions unless the user requests them. They must not block progress.
- Do not add experiments, metrics, confidence intervals, sensitivity checks, confound analyses, figures or manuscript claims merely for completeness. Include them only when requested or needed to correct a concrete error affecting the requested result.
- Reuse valid existing results and workflows. Resolve relevant label, split, data-source and comparison-setting inconsistencies while keeping the scientific scope focused on the request.
- Keep conclusions and limitations tied to observed results. Flag issues belonging to another reviewer item separately instead of expanding the current item.

## Files
- `round2_v1024.tex`: the submitted baseline. Never edit it.
- `round2_v1024_changes.tex`: the revised manuscript. Make every manuscript change here.
- `sn-jnl.cls`, `sn-mathphys-num.bst`, `xvision.bib`, `*.png`: needed to build. Put new figures in this folder. Delete a PNG only when neither `.tex` file references it.
- `build/`: build output (gitignored). Do not commit PDFs from it.
- `notes/`: per-item plans (`R1-Qn_plan.md`) and review notes. Put new plans and notes here, not beside the manuscript.

## Render
Install once with `brew install tectonic latexdiff`. Run everything from `docs/mva/`.

Clean PDF (`build/round2_v1024_changes.pdf`):
```bash
mkdir -p build && tectonic -o build round2_v1024_changes.tex
```

Tracked-changes PDF (`build/round2_v1024_diff.pdf`):
```bash
latexdiff --append-textcmd=abstract round2_v1024.tex round2_v1024_changes.tex 2>/dev/null > build/round2_v1024_diff.tex
perl -pi -e 's/^((?:\x5cDIFdelbeginFL )?\x5cDIFdelFL\{.*?\})%DIFDELCMD < (&|\x5c\x5c)/$1 $2/' build/round2_v1024_diff.tex
tectonic -Z search-path=. -o build build/round2_v1024_diff.tex
```

- Tectonic runs BibTeX and reruns TeX itself. The first build downloads packages.
- Expect and ignore two warnings from the template: `Object @figure.N already defined` and invalid UTF-8 in `algorithm.sty`.
- Without `--append-textcmd=abstract`, edits inside `\abstract{...}` appear in the diff PDF with no markup.
- `-Z search-path=.` lets the diff source in `build/` find the class file and the figures.
- `latexdiff` comments out the `&` and `\\` of a deleted table row, so the struck text spills into the next row and widens the table. The `perl` line restores them. It uses `\x5c` for the backslash so it runs the same in bash, zsh and fish.
- `latexdiff` prints harmless `Missing characters near ... IoU@90` warnings, so its stderr is discarded. If `build/round2_v1024_diff.tex` comes out empty, rerun it without `2>/dev/null`.
- Tectonic uses XeTeX, so line breaks can differ slightly from an Overleaf pdfLaTeX build.
- To check a page visually: `pdftoppm -r 80 -png -f 5 -l 5 build/round2_v1024_changes.pdf /tmp/page`.

## Revision workflow
1. Read the item's `Q`, TJ's note and `TODO` in the reply letter before changing anything.
2. Edit only `round2_v1024_changes.tex`. Every new or changed number must come from code or outputs in this repo. Never estimate or invent results.
3. Build both PDFs and check the change in each.
4. Show the human the change and wait for approval.
5. After approval, write the reply for that item's `An` in the chat. Say what changed and where in the revised manuscript (section, table, figure). The human pastes it into the reply letter.
   - Do not edit the letter yourself, in Google Docs or anywhere else.
   - Take section, figure and table numbers from the latest clean PDF build.
   - Once the reply is in the chat, close the item's beads issue. Do not wait for the reply to appear in the letter.
6. Commit the `.tex` file and any new figures. Do not commit `build/`.

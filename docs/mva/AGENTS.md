# AGENTS.md

## Scope
- Applies to `docs/mva/`: the LaTeX source of the journal manuscript *Where Do Vision Foundation Models Look? A Study of Attention Alignment in Image Recognition* (Springer Nature `sn-jnl` template, `sn-mathphys-num` reference style).

## Context
- The paper was submitted and the reviewers returned comments. This folder is where we prepare the revision.
- Reviewer comments and our replies are in the reply letter: [XAI Vision round1 reply to all reviewers 1006.docx](https://docs.google.com/document/d/1z1jEHklwWdebqoxusVEL7Pt5vTpvtQFf/edit) (Drive file ID `1z1jEHklwWdebqoxusVEL7Pt5vTpvtQFf`).
- The letter has three parts: Reply to Editor, Reviewer 1 (Q1–Q19) and Reviewer 2 (Q1–Q6). Each item has a reviewer comment (`Qn`), our reply (`An`, sometimes holding a note from TJ, the corresponding author) and an agreed action (`TODO`). Refer to items as `R1-Q7`, `R2-Q5`.
- Section, figure, table and line references in the comments point to `round2_v1024.tex`.
- The editor asks for accurate results, rewritten overstated conclusions and fully explained limitations. The resubmission includes a clean PDF, a tracked-changes PDF and the point-by-point reply letter.

## Files
- `round2_v1024.tex`: the submitted baseline. Never edit it.
- `round2_v1024_changes.tex`: the revised manuscript. Make every manuscript change here.
- `sn-jnl.cls`, `sn-mathphys-num.bst`, `xvision.bib`, `*.png`: needed to build. Put new figures in this folder. Delete a PNG only when neither `.tex` file references it.
- `build/`: build output (gitignored). Do not commit PDFs from it.

## Render
Install once with `brew install tectonic latexdiff`. Run everything from `docs/mva/`.

Clean PDF (`build/round2_v1024_changes.pdf`):
```bash
mkdir -p build && tectonic -o build round2_v1024_changes.tex
```

Tracked-changes PDF (`build/round2_v1024_diff.pdf`):
```bash
latexdiff --append-textcmd=abstract round2_v1024.tex round2_v1024_changes.tex 2>/dev/null > build/round2_v1024_diff.tex
tectonic -Z search-path=. -o build build/round2_v1024_diff.tex
```

- Tectonic runs BibTeX and reruns TeX itself. The first build downloads packages.
- Expect and ignore two warnings from the template: `Object @figure.N already defined` and invalid UTF-8 in `algorithm.sty`.
- Without `--append-textcmd=abstract`, edits inside `\abstract{...}` appear in the diff PDF with no markup.
- `-Z search-path=.` lets the diff source in `build/` find the class file and the figures.
- `latexdiff` prints harmless `Missing characters near ... IoU@90` warnings, so its stderr is discarded. If `build/round2_v1024_diff.tex` comes out empty, rerun it without `2>/dev/null`.
- Tectonic uses XeTeX, so line breaks can differ slightly from an Overleaf pdfLaTeX build.
- To check a page visually: `pdftoppm -r 80 -png -f 5 -l 5 build/round2_v1024_changes.pdf /tmp/page`.

## Revision workflow
1. Read the item's `Q`, TJ's note and `TODO` in the reply letter before changing anything.
2. Edit only `round2_v1024_changes.tex`. Every new or changed number must come from code or outputs in this repo. Never estimate or invent results.
3. Build both PDFs and check the change in each.
4. Show the human the change and wait for approval.
5. After approval, update the reply letter. Under that item's `An`, add one to three sentences on what changed and where in the revised manuscript (section, table, figure). Leave TJ's notes and the `TODO` lines for the humans to remove.
   - The letter is a `.docx` in Drive. The Drive connector can read it but cannot write its content. Give the human the exact text to paste, or edit it in the browser only after the human says to.
   - Do not rename, move, or convert the file.
6. Commit the `.tex` file and any new figures. Do not commit `build/`.

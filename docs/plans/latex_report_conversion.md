# LaTeX Report Conversion: Instructions and Workflow

## What Was Done

The full report draft in `docs/core/project_report_final.md` was converted into a
compilable LaTeX document using the course-provided template in `docs/final_report/`.

### Output files

| File | Purpose |
|------|---------|
| `docs/final_report/ISY5004_report_final.tex` | Main LaTeX source (edit this) |
| `docs/final_report/references.bib` | All 23 references in BibTeX format |
| `docs/final_report/figures/` | All figures and screenshots copied here |
| `docs/final_report/ISY5004_report_final.pdf` | Compiled output (8 pages) |

Do not modify `spconf.sty` or `IEEEbib.bst` — these are the course style files.

---

## How to Recompile the PDF Locally

Run the following sequence from the terminal (requires TeX Live or MacTeX):

```bash
cd "docs/final_report"
pdflatex ISY5004_report_final.tex
bibtex ISY5004_report_final
pdflatex ISY5004_report_final.tex
pdflatex ISY5004_report_final.tex
```

You need all four runs in that order:
1. **First pdflatex** — builds the document, generates `.aux`
2. **bibtex** — resolves citations from `references.bib` into `.bbl`
3. **Second pdflatex** — reads BibTeX output, inserts `[1]`-style citation markers
4. **Third pdflatex** — finalises cross-references (figure/table/section numbers)

After minor edits (text only, no new citations or figures), a single `pdflatex` run is enough.

### Using Overleaf instead

1. Upload the entire `docs/final_report/` folder: `.tex`, `.bib`, `.sty`, `.bst`, and the `figures/` subfolder
2. Set the main document to `ISY5004_report_final.tex`
3. Click **Compile** — Overleaf runs the full sequence automatically

---

## Section Mapping: Markdown → LaTeX

| Markdown section | LaTeX section |
|---|---|
| Abstract | `\begin{abstract}` |
| Introduction and Motivation + Research Questions | `\section{Introduction}` |
| Related Work (§5.1–5.5) | `\section{Literature review}` with `\subsection` |
| Dataset + Methodology + System Interface | `\section{Proposed approach}` |
| Results + Discussion | `\section{Experimental results}` |
| Conclusion | `\section{Conclusions and future work}` |
| Author Contributions | `\section{Author contributions}` |
| AI Tool Declaration | `\section{AI Tool Declaration}` |
| Appendix A (image grid) | `\section*{Appendix}` after `\bibliography` |

---

## How to Add a New Reference

1. Add a BibTeX entry to `docs/final_report/references.bib` using the format:
   ```bibtex
   @inproceedings{Key24Short,
     author    = {Last, First and Last2, First2},
     title     = {Paper Title},
     booktitle = {Conference Name},
     year      = {2024},
     pages     = {1--10},
   }
   ```
2. Cite it in the `.tex` body with `\cite{Key24Short}`
3. Recompile with the full 4-step sequence above

---

## How to Add a Figure

1. Copy the PNG into `docs/final_report/figures/`
2. In the `.tex` file, insert:
   ```latex
   \begin{figure}[t]
     \centering
     \includegraphics[width=\columnwidth]{your_figure.png}
     \caption{Your caption here.}
     \label{fig:yourlabel}
   \end{figure}
   ```
3. Reference it in text with `\ref{fig:yourlabel}`

For filenames containing parentheses (the Appendix A screenshots), wrap the
filename in braces: `\includegraphics[...]{{filename(with parens)}.png}`

---

## Outstanding TODOs in the .tex File

Search for `% TODO` in `ISY5004_report_final.tex` to find two placeholders:

1. **Q3 results** (Section 4.3) — insert final head-ranking figure and narrative
   once the Q3 per-head analysis is confirmed
2. **AI Tool Declaration** — fill in which tools were used and for what purpose

---

## Page Budget (8–10 page limit, two-column)

The compiled document is currently **8 pages**. If it grows over 10 pages after
adding Q3 content, apply these trims in order:

1. Shorten the per-style prose in Section 4.4 (Q2 breakdown)
2. Move the baselines calibration table to the appendix
3. Reduce Appendix A from 4 examples to 2 easy + 1 hard

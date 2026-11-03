# Plan: R2-Q2, move the Table 1 dash explanation to a footnote

Status (2026-11-02): implemented in `round2_v1024_changes.tex`; clean and tracked-changes PDFs build and are awaiting human review.

## The item

Reviewer 2, question 2 (`R2-Q2`) in the round-1 reply letter asks that the explanation for the dash symbol in Table 1 be moved from the caption to a dedicated footnote beneath the table.

**Reviewer comment:**

> The explanation for the dash symbol ("- indicates that...") in Table 1 is currently included directly within the main title. To improve layout and readability, it would be cleaner to move this sentence to a dedicated footnote beneath the table.

**Agreed TODO:** Readability fix.

## Scope and review

Table 1 is the related-work comparison labeled `table review` in `round2_v1024_changes.tex`. Its current caption combines the title with the dash definition: “The comparison between this study and existing works. $-$ indicates that it is not explicitly mentioned in the study.” No data or table entries need to change.

This is a presentation-only change. Keep the wording of the definition and the table contents unchanged; shorten the caption to the table title and place the dash definition in a dedicated note directly below the tabular material. The class already wraps each table in `threeparttable` and defines its own `tablenotes` environment, so use that built-in structure without adding another `threeparttable` wrapper. Do not add broader explanations or alter other tables.

## Plan

1. Edit only the Table 1 block in `round2_v1024_changes.tex`.
2. Separate the caption title from the dash explanation and place the latter below the table as a table note/footnote.
3. Build the clean and tracked-changes PDFs using the workflow in `docs/mva/AGENTS.md`.
4. Inspect Table 1 in both PDFs to confirm the note sits beneath the table, is readable, and the table contents and numbering remain unchanged.
5. Present the rendered change for approval before drafting the A2 reply. Do not edit the reply letter.

## Implemented change

The caption now contains only “The comparison between this study and existing works.” The same dash definition appears in a `tablenotes` note below the table. The table entries and label are unchanged. The clean PDF renders the note below the final rule on page 5. The tracked-changes PDF also builds and marks the moved definition as a deletion from the caption and an addition beneath the table.

## Out of scope

- Changing the meaning or wording of the dash definition.
- Changing table data, layout beyond the requested note placement, or the numbering.
- Revising other reviewer items or tables.

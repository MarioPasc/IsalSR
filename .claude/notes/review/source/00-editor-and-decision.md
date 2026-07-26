# Editor decision and submission mechanics

**Manuscript**: TPAMI-2026-05-1699 — "Representation of Directed Acyclic Graphs by Sequences of Instructions for Symbolic Regression"
**Type**: Regular
**Decision**: Major Revision
**Date received**: 2026-07-26
**Deadline**: 2026-09-24
**Handling**: Dr. Joyce Arnold (Administrator, jarnold@computer.org) on behalf of Dr. Kyoung Mu Lee (EIC)
**Addressed to**: Mr. Mario Pascual-González
**Recipients**: mpascual@uma.es, mario.pg02@gmail.com, ezeqlr@lcc.uma.es, elr@uma.es, karlkhader@uma.es

## Associate Editor comment (verbatim, complete)

> REQUIRED: Comments to the Author:
> The submission has received diverse ratings. The editor decides the manuscript should undergo a major revision to address the concerns.

That is the entirety of the AE comment. No additional AE guidance, no attached AE file.

**Reading**: "diverse ratings" is literal — Excellent / Good / Fair across the three reviewers. The AE has delegated entirely to the reviewer comments and offers no steer on which reviewer to prioritise.

## Required deliverables, verbatim from the decision letter

> If you should choose to revise your paper, please prepare a separate document describing how each of the reviewers' comments are responded to in your revision and send it to us by 24-Sep-2026.

> When submitting your revised manuscript, you will be able to respond to the comments made by the reviewer(s) in the space provided. You can use this space to document any changes you make to the original manuscript. In order to expedite the processing of the revised manuscript, please be as specific as possible in your response to the reviewer(s)' questions and comments. You may also upload your responses as separate files for review along with your revision. If you choose to do this, please choose "Summary of Changes" as the file designation.

> Important: Your main manuscript file cannot include colored or highlighted text. Please upload a clean, publication-ready version of your manuscript under the "Formatted (Double Column) Main File - PDF Document Only" file designation. If you would like to include an annotated version of your main manuscript file, please upload it under the "Summary of Changes" file designation.

### Required elements of the revised paper (verbatim list)

> - Abstract
> - Index terms
> - Author affiliation information
> - Main text
> - References
> - Figure captions
> - Table titles
> - Brief biography of each author
> (biographies are not required for concise papers or comments papers)

> Because this is a revision, we request that you add your author bios and photos at this time. This will help ease the transition to pre-prints if your paper is accepted. (Please note that all materials - including references, bios, photos, etc. - must fit within the 12-page limit imposed by the Submission Guidelines.)

### Page limit

> Please be mindful when making your revisions that you still need to maintain the size limitations for papers submitted to TPAMI.

**12 pages**, inclusive of main text, abstract, index terms, illustrations, references, bios and photos.

Current state on disk:

| File | Pages |
|------|-------|
| `article/paper/main.pdf` | 12 |
| `article/supplementary/supplementary.pdf` | 10 |
| `double_blind/paper/main_anonymous.pdf` | 12 |

The main file is already at the ceiling. Every reviewer request that adds text competes with existing content, and R2 separately asks for the paper to be trimmed.

### Literature note (verbatim)

> Please note, some reviewers may have recommended that you discuss additional literature when revising your manuscript. If you feel that the recommended literature does not contribute to the scholarly content of the article or is otherwise irrelevant, please note your concerns in your response to reviewer feedback.

Not applicable in practice: all three reviewers answered the "suggested additional references" box with NA / n/a, and all three rated references "sufficient and appropriate".

## Submission URL

```
https://ieee.atyponrex.com/submission/submissionBoard/REX-PROD-2-079DCF48-C984-416C-813F-BA8DCEE7106E-7EFC19DC-E948-4264-B22D-69CFBE7F6A83-12461/current?idtype=external
```

## Attached-file caveat (verbatim)

> Please note that some reviewers may have included additional comments in a separate file. If a review contains the note "see the attached file" under Section III A - Public Comments, you will need to log on to the submission site to view the file.

**None of the three reviews contains "see the attached file".** All comments are inline in the decision letter. No attachments to retrieve.

## Templates

> Please note that double column will translate more readily into the final publication format. Our peer review double column templates can be found at,
> http://www.computer.org/portal/web/peerreviewjournals/author#templates

Already satisfied: `main.tex` uses `\documentclass[10pt,journal,compsoc]{IEEEtran}`.

## Submitted package as the reviewers saw it

The reviewers had access to more than the main PDF. Evidence:

- R2 refers to "**the embedded preprint**" and "Definition 2.2 in the embedded preprint", and says the two definitions of $\Sigma_{\mathrm{SR}}$ "coexist in the submitted PDF". So the arXiv preprint was part of the submitted package and was read.
- R2 refers to appendix tables by absolute number (Table 5, Tables 6–7, Table 3 of Appendix C), consistent with the supplementary document's own numbering.
- The submission was **double-blind**: `double_blind/paper/main_anonymous.tex` and `double_blind/supplementary/supplementary_anonymous.tex`. Content verified identical to the non-anonymous `article/` versions.
- `previously_published_statement/main.tex` was included, declaring the arXiv preprint and the differences from it. Its Section 2 refers to "supplementary material (Section~S.I)", which does not match the supplementary's actual Appendix A–E numbering.

Implication for anyone answering R2.2: the label-character discrepancy is *between two documents in the package*, not inside the journal manuscript alone.

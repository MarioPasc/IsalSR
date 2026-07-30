# Worked example — R1.3, annotated

The answer to Reviewer 1 comment 3, mapped onto the spine. Source ticket: T07 plus
`T07-appendix/const_normalization_equivariance.md`. Shipped as pages 3–7 of the
letter, with one figure and one table.

The comment:

> 3) The pseudocode (Table 3, Appendix C) opens with a call to
> normalize_const_creation(D), defined only as "redirect all CONST creation edges
> to x1," that appears nowhere else in the paper?

---

## Step 1 output — the number ledger

The ticket was ~1,000 lines with several append-only corrections. What the audit
produced:

**Quotable**

| Number | Meaning | Source |
|---|---|---|
| 14,841 / 0 / 0.00% | S2D corpus, violations before and after | T06 corpus table |
| 49,980 / 0 / 0.00% | synthetic DAGs | T06 corpus table |
| 154,568 / 132,746 / 85.88% | Bingo, violated on arrival | T06 k-table, "all" row |
| 3,890 / 3,890 / 100.00% | UDFS | T06 |
| 0.00 → 27.31 → … → 100.00% | Bingo rate against $k$ | T06 condensed histogram |
| 0 in 37 (method, $k$) cells, 158,458 DAGs | residual after the step | T06 |
| 10⁵, 0 disagreements, 0 edges removed | identity on the hypothesis class | T15 AC-3 |
| 12,176,790 and 234,865 | policy-invariance on real DAGs | T15 AC-4 |
| 169 classes, synthetic ρ 1.040 → 1.042 | where the policies do differ | T15 |
| 15,530 DAGs, 123,240 permutation tests, 0 failures | order-independence | equivariance study |
| 48 → 0 canonicalisation failures | relocation policy vs additive repair | T15 |
| cos(1.5) = 0.0707 vs 1.0 | evaluation not preserved by relocation | T15 |

**Rejected, and why — every one of these sits in a plausible-looking table**

| Number | Why unusable |
|---|---|
| ρ ≈ 288 (bingo), 326–343 (udfs) | recursive re-entry contamination; retracted the same day |
| round-trip 99.39%, ~576,000 failures | same artefact, not a property of the representation |
| first Rule 1 run: 13,394 DAGs, 0 failures | **near-vacuous** — only 24 DAGs (0.18%) actually exercised the rule; explicitly "not accepted" |
| `Const`-free population, 0 equivariance failures | **vacuous** — the generator emits no `Const`, so the step is a guaranteed no-op |
| synthetic drop-arm round-trip 9,338 / 499,642 | comparator artefact, flagged "must not be quoted" |

Five traps in one ticket. The cross-check that validated the replacements:
ρ = 1.7931 from the corrected study against the paper's independently produced
1.793.

## Step 2 — the interview

Proposed to the user, concretely rather than as archetypes:

1. *(chosen, recommended)* Before/after DAG pair on `sin(x₁) + c` — the constant
   arrives as a leaf with no in-edge; the normalisation adds `x₁ → c`. Backed by
   live code. ~30 min.
2. Incidence table over the four populations. Numbers already in the ledger.
   Minutes. **Also adopted** — it answered a different need.
3. Rate-against-$k$ plot. Redundant with the prose series.
4. Prose only.

Both 1 and 2 shipped. They are not substitutes: the figure carries the mechanism,
the table carries the rate.

## Step 3 — the figure

`sin(x₁) + c` was chosen because it is the smallest DAG where the constant is
**not isolated** — it already feeds `+` and is missing only an *incoming* edge,
which is exactly what makes it unencodable. An isolated node would have
misrepresented the problem.

Verified before drawing, on both engines:

| | before | after |
|---|---|---|
| in-degree of `c` | 0 | 1 |
| canonicalisation | refused, identically on both engines | `VkVspv+Ppc`, identical on both |
| eval at x₁ = 1.5 | 2.9974949866 | 2.9974949866 |
| output node | `+` | `+` |

Plus `S2D(VkVspv+Ppc, 1) ≅` the normalised DAG.

## Step 4 — the paragraph map

| Spine | Opens with | Doing |
|---|---|---|
| 1 + 2 + 3 | "The reviewer is correct. The routine is invoked in the first line of the pseudocode of Table 3 and is defined nowhere in the manuscript." | concedes, lists the five deliverables as the length signal, pre-announces that the quoted gloss is wrong |
| 4 | "Every insertion instruction of Σ_SR creates a node together with an edge from the acting pointer" | root cause: no instruction creates a node in isolation, so in-degree 0 means unencodable; ends pointing at Figure 1 |
| — | Figure 1 | before/after, caption below |
| 5 | `rdefn` + `Algorithm 1` | the object the reviewer asked for |
| 6 | "At most \|C\| constants are examined" | worst case, then why it is not approached |
| 7 | "Four properties of the operation are used in the revision. It adds edges and never removes one…" | prose chain, ending on identity-on-the-hypothesis-class |
| 8 | "We measured how often the condition is violated by instrumenting…" | instrument, populations, $N$; result in Table 2 |
| — | Table 2 | caption above |
| 8 (cont.) | "DAGs produced by S2D satisfy the condition by construction…" | says what the table *means*, not what it contains, and why the omission went unnoticed |
| 9 | "Stratifying the Bingo stream by the number of internal nodes $k$ gives…" | mechanism: the profile a constant-terminal cause predicts |
| 10 | "Three further measurements support the properties stated above." | identity, policy-invariance, order-independence, in one paragraph |
| 11 | "We record one limitation of the definition." | the index-order dependence, its three conditions, why they cannot arise here |
| 12 | "We turn to the description quoted by the reviewer." | the relocation is unsound: 48 failures, the cos(1.5) example, non-injectivity |
| 13 | "this correction changes no number in the paper" | closes the paragraph above |
| 14 | "These measurements also answer comment R1.2" | 85.88% / 100% / none |
| 15 | `\changeref{…}` | Section 3, Table 3, Appendix D, Section 6 |

Fifteen moves, **eleven prose paragraphs**, **four display objects** (definition,
algorithm, figure, table), and **zero em-dashes** in the prose. Several moves share
a paragraph: 1–3 open together, and 13 closes the paragraph that carries 12.

Counted from the source, not from memory — the first draft of this page said
"nine paragraphs, three display objects" and both were wrong. The rule in
`SKILL.md` §1 applies to the skill's own documentation too.

## What the answer deliberately does not say

Applying the honesty gate:

**Disclosed** — the submitted relocation policy's three defects, in full, with the
failure counts. The reviewer had quoted that policy's own one-line description, so
silence would have read as concealment, and a round-2 reviewer reading the code
would have found all three.

**Also disclosed** — the index-order limitation of the current definition, which no
reviewer had asked about and which is unreachable in the reported experiments. It
cost one paragraph and forecloses the objection.

**Not narrated** — the sequence of internal code revisions, the ticket and
acceptance-criterion identifiers, dates, which measurement campaign was re-run
after which contamination, and the scripts by name. None of it is manuscript
content, none of it changes a claim, and none of it is visible in the shipped
artefacts.

The distinction is not comfort. It is whether the fact is discoverable from what
the reviewers can see.

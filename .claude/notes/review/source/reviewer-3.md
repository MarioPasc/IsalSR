# Reviewer 3

**Overall rating: Excellent.** Technically sound: **Yes**. Readability: **Easy to read**. Supplementary: accept **as is**.
The most positive of the three, and the shortest. Two comments only, one of which is the same abstract typo R2 found. The single substantive request (R3.1) is about benchmark breadth.

## Comments (verbatim, complete)

> The authors conducted verification using 50 problems, but is there a reason why they did not use the other databases mentioned in section 2.4? I believe the paper would be better if verification results obtained using these databases were included.
>
> There is a typos at the abstract:
> on both methods on both methods: -> on both methods:

That is the whole of R3's public comments.

---

## R3.1 — Why only 50 problems

**Verbatim:**

> The authors conducted verification using 50 problems, but is there a reason why they did not use the other databases mentioned in section 2.4? I believe the paper would be better if verification results obtained using these databases were included.

**Type**: coverage / new experiment. Phrased as a question, not a demand — "is there a reason why" invites a justification as much as new runs.

**What Section 2.4 actually names** — `article/paper/related_work.tex:84–94`, `\subsection{Benchmarking in Symbolic Regression}`, quoted in full because it is the exact object of the question:

> The Nguyen suite~\cite{uy2011} provides $12$ polynomial, trigonometric, and logarithmic targets widely used in GP-based SR.
> Udrescu and Tegmark~\cite{udrescu2020} introduced the AI~Feynman database of $120$ physics equations as a multi-variable alternative,
> and SRBench~\cite{lacava2021} standardised cross-method comparison on over $250$ problems from the Penn Machine Learning Benchmarks.
> Section~\ref{sec:benchmarks} describes the $50$-problem suite used in this evaluation.

So the "other databases" are precisely two:
1. **AI Feynman** — 120 equations; the suite uses 24 of them.
2. **SRBench / PMLB** — 250+ problems; the suite uses none of them.

The Nguyen suite is fully covered (12 of 12). No other database is named in Section 2.4.

**What the paper already offers as justification** — `article/paper/computational_experiments.tex:56–76`, the four predeclared inclusion criteria (`sec:benchmarks`):
> (i)~*published provenance*: the expression and sampling protocol appear in a peer-reviewed source used by at least two independent SR studies;
> (ii)~*operator compatibility*: the target expression is representable within $\Sigma_{\mathrm{SR}}$ …, excluding problems that require operators outside $\Sigma_{\mathrm{SR}}$ (e.g., $\tanh$, $\arctan$, $\mathrm{sgn}$);
> (iii)~*published evidence of structural difficulty*: the expression appears in symbolic-regression studies as a target whose recovery hinges on identifying the correct operator topology rather than fitting numerical constants;
> and (iv)~*complementary coverage*: the candidate adds at least one difficulty axis … not already represented in the suite.

Criterion (ii) is a real filter against much of AI Feynman (many Feynman equations use `arcsin`, `tanh`, `arctan`) and against most of SRBench (PMLB includes black-box real-world datasets with **no ground-truth expression at all**, so criteria (i) and (iii) cannot apply). Criterion (iv) explicitly caps redundant coverage, which is an argument against simply adding more of the same.

**Cost side of the question** — the compute is the binding constraint, and the numbers are already in the manuscript:
- 12-hour (43,200 s) budget per run, `supplementary.tex:562`.
- 1 CPU core + 8–16 GB RAM per run, 15–17 h wallclock allocation, SLURM on Picasso, `supplementary.tex:571–580`.
- Current campaign is $2 \times 2 \times 50 \times 30 = 6{,}000$ runs (as corrected under R2.6) $\approx 7.2 \times 10^4$ core-hours of search budget alone.
- Inspection of `table_supplementary_udfs.tex` shows **most UDFS runs hit the 43,200 s ceiling** — 36 of 50 problems report $T \approx 43{,}200$ s for both variants. UDFS is budget-saturated, so added problems cost the full budget each.
- Scaling to all 120 Feynman equations plus 250 SRBench problems would be $2 \times 2 \times 382 \times 30 \approx 45{,}840$ runs, roughly $7.6\times$ the current campaign.

**Also on the future-work list already** — `article/paper/conclusion.tex:25–28`:
> natural extensions include broader validation on PySR, Operon, DSO and transformer-based methods, **scalability on SRBench**, and the interaction between structural deduplication and explicit diversity-preserving schemes

and `article/paper/discussion.tex:51–52`:
> Extending the evaluation to SR methods that generate larger DAGs is a priority for future work.

So SRBench is already declared future work; R3 is asking for it to be brought forward.

**Tension with R1**: R1's opening explicitly endorses the existing protocol — *"The empirical protocol is rigorous (50 problems, 30 seeds, Demsar-style paired inference)"*. R1 wants a different missing baseline (hash-based dedup), not more problems. R2 wants the paper **shorter**. Any response that expands benchmark coverage has to reconcile these three.

---

## R3.2 — Abstract typo

**Verbatim:**

> There is a typos at the abstract:
> on both methods on both methods: -> on both methods:

**Type**: typo. **Confirmed.** Same defect reported by R2.8(a). `article/paper/main.tex:81`:

> … returns Cohen's $d > 2$ at $p < 10^{-21}$ **on both methods on both methods**: canonicalisation eliminates a mean of $34\%$ …

R3 gives the exact correction: delete one occurrence, keep `on both methods:`.

The same line also contains a double space in "empirical  reduction factor", which neither reviewer mentioned.

Note the abstract is duplicated across documents: `double_blind/paper/main_anonymous.tex` carries the same abstract text, and `previously_published_statement/main.tex:98–115` restates the same headline numbers. Fixes must propagate.

---

## Structured answers (verbatim)

- **A1. Category**: Research/Technology
- **A2. Relevance**: Relevant
- **B1. Significance**: Good
- **B3. Technically sound**: **Yes**
- **B4. Experimental validation**: Lacking in some respects; some cases of interest not tested
- **C1. Title/abstract/keywords appropriate**: Yes
- **C2. References**: References are sufficient and appropriate — suggested references: **n/a**
- **C3. Introduction**: Yes
- **C4. Organization**: Satisfactory
- **C5. Readability**: **Easy to read**
- **C6. Length**: About right
- **C7. Supplemental material**: Yes, as part of the digital library for this submission if accepted
- **C8. Supplemental accept**: **As is**
- **Overall**: **Excellent**

R3 is the only reviewer who marks B3 "Yes" and the only one who accepts the supplementary as is. Their B4 answer ("lacking in some respects") is fully accounted for by R3.1 — coverage, not rigor.

### B2 — significance statement (verbatim, complete)

> This study aims to address the problem of redundant representations in Directed Acyclic Graphs (DAGs) used to represent formula candidates in symbolic regression. In such SR problems, there is an inefficiency during the formula search process where formulas with the same structure are treated as different candidates and evaluated repeatedly simply because their node numbers or order of creation differ. The authors note that in a DAG with k internal nodes, such redundant representations can increase exponentially to a scale of Θ(k!).
>
> To solve this problem, this paper proposes representing the DAG as a 2-tier alphabetic string composed of a structure token and an operation token, thereby eliminating redundancy in isomorphism. Additionally, equivalence candidates are refined using a 1-WL-based subtree hash, and the string is constructed greedily using this as a guide to ensure practical and fast operation. Subtraction and division operations are absorbed into unary operations to reduce non-commutativity.
>
> This paper demonstrated excellent performance results using a relatively simple method by clearly defining the problem. Beyond theoretical verification, the properties of the representation itself were verified through various experiments, including round-trip fidelity, acyclicity, and idempotence. In addition, it presents analysis experiments on costs and scaling, demonstrating highly complete research results.

**One thing to notice in this paragraph**: R3 credits the paper with verifying "round-trip fidelity, acyclicity, and **idempotence**". Those five intrinsic-property experiments belong to the **arXiv preprint**, not the TPAMI manuscript — `previously_published_statement/main.tex:82–89` states plainly that the preprint "validated five intrinsic properties of the representation (round-trip fidelity, acyclicity, canonical invariance, evaluation preservation, search-space reduction)" and that the journal version adds the solver-integration study instead. The word "idempotence" appears nowhere in the TPAMI manuscript or supplementary. R3 has attributed preprint content to the submission — worth knowing when reading their "Excellent", and worth knowing if any response cites those experiments.

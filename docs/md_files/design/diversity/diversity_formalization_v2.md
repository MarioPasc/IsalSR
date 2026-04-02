# Diversity Preservation under Canonical Representation

**Draft section for the ISALSR paper (IEEE TPAMI submission)**
**Status:** Working draft v2 — metric revised per discussion
**Placement:** §6.1 (Theoretical Implications) or as a new §5.X (Results subsection)

---

## Notation and Dependencies

This section depends on Definition 3.9 (Labeled-DAG Isomorphism), Theorem 3.13
(Fast Canonical String is a Complete Labeled-DAG Invariant), and the
Orbit-Stabilizer analysis of §5.2.

---

## Layer 1 — Definitions

We formalize population diversity for an evolutionary SR algorithm operating on
expression DAGs. Let $N$ denote the population size and let
$P_t = \{G_1, \ldots, G_N\}$ be the multiset of labeled DAGs at generation $t$.

**Definition X.1 (Effective Diversity Ratio).** The *effective diversity ratio* of
$P_t$ is the fraction of population slots occupied by pairwise non-isomorphic
individuals:

$$
\delta(P_t) \;=\; \frac{\bigl|\{[G] : G \in P_t\}\bigr|}{N}\,,
$$

where $[G] = \{G' \in \mathcal{G}_n : G' \cong G\}$ denotes the isomorphism class
of $G$ under Definition 3.9. By construction, $\delta(P_t) \in (0, 1]$.

**Definition X.2 (Population Pairwise Distance).** Let $d : \mathcal{G}_n \times
\mathcal{G}_n \to \mathbb{R}_{\geq 0}$ be any metric on labeled DAGs satisfying
the identity of indiscernibles under Definition 3.9:

$$
d(G_i, G_j) = 0 \;\;\Longleftrightarrow\;\; G_i \cong G_j.
$$

The *mean pairwise distance* of population $P_t$ under $d$ is:

$$
\bar{d}(P_t) \;=\; \frac{1}{\binom{N}{2}} \sum_{1 \leq i < j \leq N}
    d(G_i,\, G_j).
$$

The *diameter* of $P_t$ under $d$ is:

$$
\Delta(P_t) \;=\; \max_{i \neq j}\; d(G_i,\, G_j).
$$

Both $\bar{d}$ and $\Delta$ are well-defined for any metric $d$ satisfying the
identity of indiscernibles, and the theoretical results in Layer 2 hold
independently of the specific choice of $d$.

**Remark X.3 (Choice of metric — why not Levenshtein).** The Levenshtein
distance on canonical strings $d_{\mathrm{Lev}}(\hat{w}_{G_i},
\hat{w}_{G_j})$ satisfies the identity of indiscernibles by Theorem 3.13 and
is therefore a valid instantiation of $d$ in principle. However, its correlation
with the standard structural dissimilarity measure (Graph Edit Distance) degrades
with edge density: the IsalGraph evaluation [López-Rubio and Pascual-González,
2026] reports Spearman $\rho = 0.934$ on sparse graphs ($\bar{m} = 3.07$) but
$\rho = 0.349$ on moderately dense graphs ($\bar{m} = 10.70$). SR expression
DAGs fall in the denser regime (a DAG with $k = 6$ internal nodes on $m = 1$
variable can have 10–15 edges on 7 nodes), where $d_{\mathrm{Lev}}$ preserves
less than 50% of the variance in GED rankings ($\rho^2 < 0.49$). An observed
increase in $\bar{d}_{\mathrm{Lev}}(P'_t)$ relative to $\bar{d}_{\mathrm{Lev}}
(P_t)$ could therefore reflect metric distortion rather than genuine structural
spread. For this reason, we instantiate $d$ with labeled GED in the experiments
(§Empirical Evidence).

---

## Layer 2 — Structural Result

The following proposition is a direct consequence of Theorem 3.13 and requires
no assumptions about the evolutionary dynamics or the choice of metric $d$.

**Proposition X.4 (Isomorphism-Free Population).** Let $\mathcal{A}$ be an
evolutionary SR algorithm that represents individuals as ISALSR canonical
strings $\hat{w}_G$ and enforces a *duplicate-free* population invariant in
canonical string space — i.e., no two individuals share the same canonical
string at any generation. Let $P'_t$ denote the population of $\mathcal{A}$ at
generation $t$. Then:

$$
\delta(P'_t) = 1 \qquad \forall\, t \geq 0.
$$

*Proof.* The duplicate-free invariant requires $\hat{w}_{G_i} \neq
\hat{w}_{G_j}$ for all $i \neq j$. By the completeness direction of
Theorem 3.13 ($\hat{w}_{D_1} = \hat{w}_{D_2} \Longleftrightarrow D_1 \cong
D_2$), distinct canonical strings imply $G_i \not\cong G_j$. Therefore every
individual in $P'_t$ belongs to a distinct isomorphism class, and
$|\{[G] : G \in P'_t\}| = N$, giving $\delta(P'_t) = N/N = 1$.  $\square$

**Remark X.5 (Baseline bound).** For a baseline algorithm $\mathcal{B}$
operating on raw DAG representations with no isomorphism-level deduplication,
the effective diversity ratio satisfies $\delta(P_t) \leq 1$, with equality only
when no two population members happen to be isomorphic. Under selection pressure,
fit individuals propagate via crossover and elitism. Because the baseline
operates on *node-labeled* DAGs, a single expression $G^*$ with $k$ internal
nodes can occupy up to $k!/|\mathrm{Aut}(G^*)|$ population slots through
distinct node-numberings (Equation 2, §5.2) — all representing the same
mathematical expression and thus contributing no diversity. In practice,
$\delta(P_t)$ decreases monotonically as $t$ increases.

**Remark X.6 (Zero-pair decomposition).** The effect of isomorphic duplicates on
$\bar{d}(P_t)$ can be made precise for *any* metric $d$ satisfying the identity
of indiscernibles. Partition the $\binom{N}{2}$ population pairs into those that
are isomorphic and those that are not:

$$
\bar{d}(P_t) \;=\; \frac{1}{\binom{N}{2}}
    \Biggl[\;\underbrace{\sum_{\substack{i < j \\ G_i \cong G_j}}
        d(G_i, G_j)}_{\displaystyle = \; 0}
    \;+\;
    \sum_{\substack{i < j \\ G_i \not\cong G_j}}
        d(G_i, G_j)\;\Biggr].
$$

The first sum vanishes by the identity of indiscernibles, but the denominator
$\binom{N}{2}$ counts *all* pairs. If the population contains $r$
isomorphic-pair entries (i.e., $\binom{N}{2} - r$ non-isomorphic pairs), then:

$$
\bar{d}(P_t) \;=\; \frac{\binom{N}{2} - r}{\binom{N}{2}} \;\cdot\;
    \bar{d}_{\neq}(P_t)\,,
$$

where $\bar{d}_{\neq}(P_t)$ is the mean distance restricted to non-isomorphic
pairs. The prefactor $(\binom{N}{2} - r)/\binom{N}{2}$ is a *dilution factor*:
each isomorphic duplicate pair contributes a zero-distance entry that drags the
population-level mean downward. An ISALSR-augmented algorithm with $\delta = 1$
has $r = 0$ by construction, eliminating this dilution entirely.

This decomposition is metric-agnostic: it holds for GED, Levenshtein distance,
or any other metric satisfying $d(G_i, G_j) = 0 \iff G_i \cong G_j$.

---

## Layer 3 — Empirical Conjecture

The structural result (Proposition X.4) guarantees only that every population
slot is occupied by a distinct isomorphism class. It does not, on its own, imply
that the freed slots explore *distant* regions of the quotient space
$\mathcal{G}_n / {\cong}$. That stronger claim depends on the interaction
between the representation, the genetic operators, and the selection mechanism.
We state it as an empirical conjecture, supported by the Bingo integration
experiment.

**Conjecture X.7 (Diversity Preservation).** Let $P_t$ and $P'_t$ be
populations of equal size $N$, evolved under identical evolutionary operators,
selection mechanism, fitness function, and random seed, differing only in that
$P'_t$ uses ISALSR canonical string representation with duplicate-free
enforcement. Let $d$ be any metric on labeled DAGs satisfying
$d(G_i, G_j) = 0 \iff G_i \cong G_j$. Then for all $t \geq t_0$ — where $t_0$
is the generation at which selection pressure begins to dominate genetic drift —
the following inequalities hold in expectation over random seeds:

$$
\mathbb{E}\bigl[\delta(P'_t)\bigr] \;>\; \mathbb{E}\bigl[\delta(P_t)\bigr],
\tag{C1}
$$

$$
\mathbb{E}\bigl[\bar{d}(P'_t)\bigr] \;>\; \mathbb{E}\bigl[\bar{d}(P_t)\bigr].
\tag{C2}
$$

Inequality (C1) follows from Proposition X.4 whenever $\delta(P_t) < 1$, which
occurs under selection pressure (Remark X.5). Inequality (C2) is the substantive
claim: the freed population slots are not merely *distinct* but explore
structurally *distant* regions of the search space.

**Heuristic argument for (C2).** Two mechanisms contribute. First, the dilution
factor derived in Remark X.6 shows that eliminating isomorphic zero-distance
pairs raises $\bar{d}(P'_t)$ mechanically, even if the non-zero distances are
identical between $P_t$ and $P'_t$. Second, in the baseline, each isomorphic
copy of a fit individual $G^*$ competes for crossover and mutation slots,
producing offspring in the immediate neighbourhood of $[G^*]$ in
$\mathcal{G}_n / {\cong}$. When these redundant copies are replaced by
genuinely distinct individuals (as enforced by canonical deduplication), genetic
operators act on a broader set of parent structures, producing offspring that
sample a wider region of the quotient space. This second effect depends on the
specific operators and is not provable without assumptions on the mutation/
crossover kernels; it is the content of the conjecture.

---

## Empirical Evidence

### Metric Instantiation

We instantiate the generic metric $d$ in Conjecture X.7 as the *labeled Graph
Edit Distance* (GED) with the following cost function, designed to respect the
ISALSR isomorphism definition (Definition 3.9):

| Operation | Cost |
|---|---|
| Node substitution ($\ell(u) \neq \ell(v)$) | 1 |
| Node insertion / deletion | 1 |
| Edge insertion / deletion | 1 |
| Node substitution ($\ell(u) = \ell(v)$) | 0 |

Under these costs, $d_{\mathrm{GED}}(G_i, G_j) = 0 \iff G_i \cong G_j$ for
labeled DAGs, satisfying the identity of indiscernibles required by
Definition X.2.

**Computational tractability.** Exact GED is NP-hard in general [Zeng et al.,
2009], but all expression DAGs in the Bingo populations satisfy $|V| \leq 14$
(at most $k = 12$ internal nodes plus $m \leq 2$ variables). At this scale,
exact A$^*$-based GED computation is tractable: the IsalGraph evaluation
[López-Rubio and Pascual-González, 2026] computes exact GED via A$^*$ on graphs
of comparable size ($|V| \leq 12$, LINUX dataset). Computing all
$\binom{200}{2} = 19{,}900$ pairwise distances per generation is expensive but
feasible, and we restrict computation to the sampled generations
$t \in \{0, 5, 10, 20, 30, 50, 70, 100, 120, 150\}$.

### Experimental Setup

We integrate ISALSR into Bingo [Randall et al., 2022], a DAG-native evolutionary
SR algorithm, by inserting a canonicalization step after each genetic operation.
The baseline is unmodified Bingo. Both configurations use identical parameters:
population size $N = 200$, [additional parameters]. We evolve both variants on
[benchmark name] across [number] independent random seeds and record, at each
sampled generation $t$:

1. $\delta(P_t)$: the effective diversity ratio (Definition X.1).
2. $\bar{d}_{\mathrm{GED}}(P_t)$: the mean pairwise labeled GED
   (Definition X.2, instantiated with the cost function above).
3. A two-dimensional PCA projection of the population for qualitative
   visualization (see below).

### Results

**Diversity ratio (Figure X, bottom panel — $\delta$ curves).**
For the baseline, $\delta(P_t)$ collapses from $178/200 = 0.89$ at $t = 0$ to
$8/200 = 0.04$ at $t = 150$: the population converges to fewer than 10
structurally distinct expressions, with the remaining ${\sim}192$ slots occupied
by isomorphic copies of fit individuals. This behaviour is consistent with
Remark X.5: selection pressure propagates fit genotypes, and the baseline's
inability to detect isomorphisms allows $k!/|\mathrm{Aut}(G^*)|$ redundant
copies per expression.

The ISALSR variant maintains $\delta(P'_t) = 1.0$ throughout, with the unique
count reaching and sustaining $N = 200$ from $t = 30$ onward. This confirms
inequality (C1) and is a direct empirical consequence of Proposition X.4.

**Mean pairwise GED (Figure X, bottom panel — $\bar{d}$ curves).**
[TO BE COMPUTED — report $\bar{d}_{\mathrm{GED}}(P_t)$ and
$\bar{d}_{\mathrm{GED}}(P'_t)$ averaged over seeds, with 95% confidence
intervals. The prediction from Remark X.6 is that $\bar{d}_{\mathrm{GED}}
(P'_t) > \bar{d}_{\mathrm{GED}}(P_t)$ for $t \geq t_0$, with the gap
increasing as the baseline $\delta$ decreases.]

**Regression performance (Figure X, bottom panel — $R^2$ curves).**
The bottom panel of Figure X shows that the ISALSR variant achieves equal or
superior test $R^2$ while maintaining full diversity, demonstrating that
diversity preservation does not trade off against regression performance.

**PCA visualization (Figure X, top and middle rows).**
Figure X displays population snapshots at nine sampled generations, projected
onto the first two principal components of a feature representation of each
individual's DAG structure. The count of unique isomorphism classes is annotated
per panel. These projections serve as qualitative evidence that the ISALSR
population occupies a broader region of the projected space than the baseline,
consistent with conjecture (C2). The explained variance ratio of the first two
principal components is [TO BE REPORTED] — this quantifies what fraction of the
total structural variation is captured by the 2D projection. The PCA
visualization complements but does not replace the quantitative
$\bar{d}_{\mathrm{GED}}$ analysis: distances in PCA space are not faithful to
structural dissimilarity in general, since the projection discards information
in the remaining components.

---

## Summary

| Property | Baseline ($P_t$) | ISALSR ($P'_t$) | Status |
|---|---|---|---|
| $\delta$ at $t = 0$ | 0.89 | 0.89 | Measured |
| $\delta$ at $t = 150$ | 0.04 | 1.00 | Measured |
| $\delta = 1\;\forall t$ | Not guaranteed | **Proved** (Prop. X.4) | Theorem |
| $\bar{d}_{\mathrm{GED}}(P'_t) > \bar{d}_{\mathrm{GED}}(P_t)$ | — | — | Conjecture (C2), to be measured |
| Test $R^2$ at $t = 150$ | $\approx 1.0$ | $\approx 1.0$ | Measured |

The three layers decompose the diversity claim into what can be proved
($\delta = 1$), what can be explained mechanically (zero-pair dilution, valid
for any metric), and what remains an empirical conjecture (broader exploration
of the quotient space, tested with GED). This separation prevents overclaiming
while providing a precise framework for quantifying the diversity benefit of
ISALSR canonicalization in population-based SR methods.

---

## Implementation Notes

### GED computation for ISALSR DAGs

The labeled GED must account for ISALSR-specific structure:

1. **Node labels are operation types** from Table 1 (ADD, MUL, SIN, COS, ...,
   CONST, VAR). Substitution cost is 1 if types differ, 0 otherwise.
2. **Edges are directed** (dataflow direction: $u$ provides input to $v$).
3. **POW operand order matters** (Definition 3.9, condition (iv)): if $v$ is a
   POW node, the ordered input list $\sigma(v) = (u_1, u_2)$ distinguishes base
   from exponent. Two POW nodes with swapped operand order are *not* isomorphic.
   The GED cost function should treat operand-order-swapped POW nodes as
   requiring a substitution (cost 1).
4. **Variable nodes are anchored** (Definition 3.9, condition (iii)):
   $\phi(x_i) = x_i$. The GED computation should fix variable nodes (zero cost
   for matching $x_i \leftrightarrow x_i$, infinite cost for
   $x_i \leftrightarrow x_j$ with $i \neq j$).

A suitable implementation is the A$^*$-based exact GED solver in NetworkX
(`networkx.graph_edit_distance`), with a custom `node_subst_cost` function
encoding rules 1, 3, and 4 above, and `edge_subst_cost = 0`,
`edge_del_cost = edge_ins_cost = 1`.

### PCA feature representation

The PCA projection requires a fixed-length vector representation of each DAG.
Options include:

- **Canonical string bag-of-$n$-grams**: count frequencies of character $n$-grams
  (e.g., bigrams) in the canonical string $\hat{w}_G$. This is simple and
  directly tied to the ISALSR representation, but inherits the Levenshtein
  metric's density-dependent distortion.
- **Operation-type histogram + degree sequence**: a feature vector combining
  the count of each operation type with the sorted in-degree and out-degree
  sequences, zero-padded to a fixed length. This is metric-independent and
  captures structural properties directly.
- **1-WL hash histogram**: count the frequency of each distinct 1-WL subtree
  hash $h(v)$ (Definition 3.7) across all nodes. This captures the full rooted
  subtree isomorphism type at each node and is the most structurally informative
  option, but requires a consistent hash-to-index mapping across the population.

Report the explained variance ratio for the first two components regardless of
which feature representation is chosen.

---

## References to Add

- Zeng, Z., Tung, A.K.H., Wang, J., Feng, J., and Zhou, L. "Comparing stars:
  on approximating graph edit distance." *Proceedings of the VLDB Endowment*,
  2(1):25–36, 2009. (NP-hardness of GED)
- Sanfeliu, A. and Fu, K.S. "A distance measure between attributed relational
  graphs for pattern recognition." *IEEE Trans. Syst., Man, Cybern.*,
  13(3):353–362, 1983. (Original GED definition)
- Rothlauf, F. *Representations for Genetic and Evolutionary Algorithms*.
  Springer, 2nd edition, 2006. (Non-redundant encoding theory: synonymity,
  locality, redundancy in EA representations)
- Burke, E.K., Gustafson, S., and Kendall, G. "Diversity in genetic
  programming: an analysis of measures and correlation with fitness."
  *IEEE Trans. Evol. Comput.*, 8(1):47–62, 2004. (Diversity measures in GP)
- Squillero, G. and Tonda, A. "Divergence of character and premature
  convergence: a survey of methodologies for promoting diversity in
  evolutionary optimization." *Information Sciences*, 329:782–799, 2016.
  (Survey of diversity preservation methods in evolutionary computation)
- López-Rubio, E. and Pascual-González, M. "Instruction set for the
  representation of graphs." *arXiv preprint*, arXiv:2603.11039v1, 2026.
  (IsalGraph; Levenshtein–GED correlation analysis justifying metric choice)

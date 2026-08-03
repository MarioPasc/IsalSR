# T04 Appendix — The naive fixed-order-serialisation hash baseline

**Status of this document**: reference specification and results record for the
comparator built in answer to reviewer comment **R1.4**. Written 2026-08-02.

**Parent ticket**: `.claude/notes/review/tasks/T04-naive-hash-dedup-baseline.md`
**Implementation**: `src/isalsr/baselines/{fixed_order_hash,host_native,cardinality}.py`
**Probe evidence**: `~/execs/isalsr/t04_probe/` on Picasso, arrays `1737666`/`1737667`,
retry `1737714`, commit `a4206b8`, seed 0, `max_time` 1500 s, `--constraint=intel`.

> **All numbers in §5 are provisional.** They come from a 28-cell probe at one
> seed on four problems, not from campaign C2. They establish that the arm works
> and what it measures; they are not the numbers that go in the paper.

---

## 1. What the reviewer asked, and what this is

> **R1.4** — *"no comparison against naive hash-based deduplication on a fixed-order
> DAG serialization"*

The comment names one comparator, not a family, and the operative word is
**naive**. The deliverable is therefore a single additional arm that a
practitioner would plausibly reach for before reading our paper: serialise the
DAG in whatever order you already have it in, hash the bytes, and use the hash as
a dedup key. Anything cleverer is a different paper's contribution and would be
answering a question nobody asked.

The scientific claim the arm is built to test is **not** "our method is faster".
It is:

> A fixed-order serialisation is **sound but incomplete** as an isomorphism test.
> It never merges two DAGs that differ, and it fails to merge DAGs that agree.
> The gap between the two is exactly the redundancy that requires an
> isomorphism-invariant canonical form to detect.

Whether that gap is large is an empirical question with a host-dependent answer,
and §5 reports both answers, including the one that is unflattering to us.

---

## 2. Formal definition

### 2.1 The object

A **labeled DAG** is a triple $D = (V, E, \ell)$ together with an input-order map:

- $V = \{0, 1, \dots, n-1\}$, contiguous integer node identifiers;
- $E \subseteq V \times V$ acyclic, with $(u,v) \in E$ read as *"$u$ provides input
  to $v$"*;
- $\ell : V \to \Sigma_{SR} \cup \{\mathrm{VAR}_1, \dots, \mathrm{VAR}_m\}$, the
  node labels, where $\Sigma_{SR}$ is the 12-label decomposed alphabet
  (T16: no `-`, no `/`; `Pow` is the only non-commutative operation);
- $\mathrm{ord}_v$, a total order on the in-neighbours of $v$, significant only
  when $\ell(v)$ is non-commutative.

Two labeled DAGs are **isomorphic**, $D \cong D'$, when there is a bijection
$\varphi : V \to V'$ such that

$$\ell'(\varphi(v)) = \ell(v), \qquad (u,v) \in E \iff (\varphi(u), \varphi(v)) \in E',$$

with $\mathrm{ord}$ preserved at non-commutative nodes, and $\varphi$ the identity
on $\mathrm{VAR}$ nodes — the input variables are pre-numbered and
distinguishable, so they are never permuted.

### 2.2 Fixed-order serialisation

Let $\pi_D : V \to \{0, \dots, n-1\}$ be a node ordering computed **from the
representation of $D$**, not from its isomorphism class. Let $\mathrm{enc}$ be an
injective encoding of an ordered labeled adjacency structure into bytes. Define

$$\sigma_\pi(D) \;=\; \mathrm{enc}\bigl(\pi_D(D)\bigr),$$

that is: relabel every node by its position under $\pi_D$, write out the label
sequence and the adjacency in those coordinates, and encode. The **hash key** is
$h(\sigma_\pi(D))$ for a hash function $h$.

Three orderings are implemented (`fixed_order_hash.py`), forming a ladder of
increasing normalisation power:

| Rung | `SerialisationOrder` | $\pi_D$ | Normalises |
|---|---|---|---|
| 1 | `INSERTION` | node id as stored | nothing |
| 2 | `TOPOLOGICAL` | topological sort, ties by (label, id) | evaluation order |
| 3 | `TOPOLOGICAL_COMMUTATIVE` | as rung 2, plus in-neighbour sets sorted at commutative nodes | + operand order |

### 2.3 Soundness

> **Lemma.** If $\mathrm{enc}$ is injective and $\pi_D$ is a function of the
> representation of $D$, then
> $$\sigma_\pi(D) = \sigma_\pi(D') \;\Longrightarrow\; D \cong D'.$$

*Proof.* Suppose $\sigma_\pi(D) = \sigma_\pi(D')$. By injectivity of
$\mathrm{enc}$, the ordered labeled adjacency structures agree:
$\pi_D(D) = \pi_{D'}(D')$ as labeled digraphs on $\{0,\dots,n-1\}$. Both $\pi_D$
and $\pi_{D'}$ are bijections, so $\varphi := \pi_{D'}^{-1} \circ \pi_D$ is a
bijection $V \to V'$. It preserves labels and adjacency because the two images
coincide coordinatewise, and it fixes $\mathrm{VAR}$ nodes because every rung
places $\mathrm{VAR}$ nodes first in variable index order. Hence $D \cong D'$. $\blacksquare$

**Consequence.** Modulo collisions of $h$ itself, the naive baseline **never
merges two structurally different DAGs**. It is a safe dedup key. This is worth
stating plainly in the response letter: we are not accusing the baseline of being
wrong. It is correct. It is merely partial.

### 2.4 Incompleteness

The converse fails, and this is the entire point. $D \cong D'$ does **not** imply
$\pi_D(D) = \pi_{D'}(D')$, because $\pi$ reads the representation and isomorphism
is precisely what permutes it.

**Worked counter-example (rung 1).** Both DAGs denote $\sin(x_0) + \cos(x_1)$:

| | $D_1$ | $D_2$ |
|---|---|---|
| node 0 | $\mathrm{VAR}\;x_0$ | $\mathrm{VAR}\;x_0$ |
| node 1 | $\mathrm{VAR}\;x_1$ | $\mathrm{VAR}\;x_1$ |
| node 2 | $\mathrm{SIN}$, in: $\{0\}$ | $\mathrm{COS}$, in: $\{1\}$ |
| node 3 | $\mathrm{COS}$, in: $\{1\}$ | $\mathrm{SIN}$, in: $\{0\}$ |
| node 4 | $\mathrm{ADD}$, in: $\{2,3\}$ | $\mathrm{ADD}$, in: $\{2,3\}$ |

$D_1 \cong D_2$ under the transposition $\varphi = (2\;3)$. The insertion-order
label sequences are $[\mathrm{VAR},\mathrm{VAR},\mathrm{SIN},\mathrm{COS},\mathrm{ADD}]$
and $[\mathrm{VAR},\mathrm{VAR},\mathrm{COS},\mathrm{SIN},\mathrm{ADD}]$ —
different bytes, different hash, two dedup entries for one DAG. The canonical
string is identical for both.

Rungs 2 and 3 repair this particular instance (sorting the tie by label
separates $\mathrm{SIN}$ from $\mathrm{COS}$ deterministically), which is why the
ladder exists. They do not repair it in general: rung 3 still breaks residual
ties by node index, so any pair of same-label nodes at the same topological level
that are not automorphic remains order-dependent. Establishing completeness
requires individualisation-refinement over $1$-WL (McKay & Piperno 2014); $1$-WL
alone is provably insufficient (Cai, Fürer & Immerman 1992). **§5 measures the
size of the residual gap rather than arguing about it.**

### 2.5 The arm: host-native serialisation

The arm does not serialise our `LabeledDAG`. It serialises **the host's own
representation**, via a generic record interface (`host_native.py`):

$$\text{record} = (\texttt{host\_key},\ \texttt{host\_tag},\ \texttt{operand\_keys})$$

with one record per host node, emitted in the host's own storage order —
`command_array` rows for Bingo `AGraph`, `CompGraph` nodes for UDFS. The module
is stdlib-only and imports nothing from `experiments/`, so the baseline cannot
accidentally acquire any of our machinery.

`host_native_hash` is the live dedup key (builtin `hash`, process-local);
`host_native_digest` is the stable one (`blake2b`, 8 bytes) for cross-run
artefacts. `HASH_ARM_KEY_MODE = "host_native"` in both
`{bingo,udfs}/isalsr_runner.py`, and `Hash{Bingo,UDFS}Runner.KEY_MODE` selects it.

**Why the host's representation and not ours — this is the load-bearing design
decision.** §3.2 gives the argument and §5.2 gives the measurement that settles it.

---

## 3. Why it was built this way

### 3.1 One arm, not four

An earlier draft proposed four comparator arms spanning the normalisation ladder.
Rejected by Mario, 2026-07-31: the reviewer asked for *a* naive comparator, and a
comparator suite that includes progressively stronger normalisers is no longer
naive — it is us building a competitor and then beating it, which reads as
straw-manning in the opposite direction. The ladder survives as **offline
analysis on a single candidate stream** (§4.2 of the ticket), not as extra arms.

### 3.2 Host-native, not adapter output — and the measurement error that proved it

Both adapters (`agraph_to_labeled_dag`, `compgraph_to_labeled_dag`) **renumber**
nodes as they build our `LabeledDAG`: variables first, then constants, then
internal nodes in topological order. Serialising adapter output therefore
measures a representation that has already been partially canonicalised — by us,
not by the host. A baseline built on it is not the baseline the reviewer
described, and it flatters the naive method.

This was not caught by reasoning. It was caught by Mario rejecting a result:

- **Retracted (2026-07-30)**: measured on adapter output, $\rho_{iso} \approx 1.00$
  — i.e. "isomorphism-invariance buys essentially nothing". Recorded, then
  challenged with *"I think your deduplication is not naive enough"*.
- **Corrected**: measured on host-native representations, $\rho_{iso} = 1.3724$,
  with **92.4 %** of duplicate detections requiring $1$-WL. The adapter's
  renumbering had been collapsing 25.1 % of distinct host representations before
  the "naive" hash ever saw them.

The retracted entry is deliberately left visible in the ticket's work log. §5.2
below reproduces the same effect on Picasso from two independent measurements in
the same probe, which is the strongest form of the evidence.

### 3.3 Shadow counters, and why HyperLogLog

To compare rungs **on one candidate stream** — avoiding the confound that each
arm's dedup changes what the host explores next — the `isalsr` arm carries
parallel counters estimating $|\{\text{distinct keys}\}|$ under each ordering.

Exact sets are not affordable: a 12 h Bingo run emits $\mathcal{O}(10^7)$
candidates, and four exact `set[int]` would add gigabytes to a job that already
needed 128 G. HyperLogLog (Flajolet et al., AofA 2007) at $p = 16$ uses $2^{16}$
one-byte registers = **64 KB per sketch, 256 KB for all four**, independent of
stream length, with standard error $1.04/\sqrt{2^{16}} \approx 0.41\,\%$ — an
order of magnitude below the effects being measured. AC-10 required this be
*measured*, not assumed; see §5.4.

### 3.4 Non-goals

- Not a new SR method, and not a new dedup method. (Advisor constraint 1.)
- Not an argument that the baseline is incorrect. It is correct; see §2.3.
- Not decomposed inside the canonicaliser — `fcs` stays a pure function of $D$.

---

## 4. How it was tested

### 4.1 Offline, against the corpus

| Property | Method |
|---|---|
| Soundness (§2.3) | No serialisation merges non-isomorphic DAGs across the 14,841-DAG corpus, **both backends** |
| Incompleteness (§2.4) | Isomorphic pairs the serialisation separates and the canonical string merges, exhibited in the same suite |
| Rung monotonicity | $\lvert\mathrm{distinct}_1\rvert \ge \lvert\mathrm{distinct}_2\rvert \ge \lvert\mathrm{distinct}_3\rvert \ge \lvert\mathrm{distinct}_{\mathrm{canon}}\rvert$ on the corpus |
| Determinism | `fixed_order_digest` (blake2b) stable across processes; `fixed_order_hash` explicitly documented as process-local |

### 4.2 On Picasso — the standing property probe

Every one of the 28 probe tasks ran SP-1…SP-6 **before** its search, so a
violation costs seconds rather than 25 minutes, and wrote the result as a parsed
JSON artefact rather than a log line. "I checked it" is not evidence.

**All 336 checks passed** (56 evidence files × 6 checks; 28 normal + 28 forced-Python).

| Probe | Establishes | Result |
|---|---|---|
| SP-1 | Provenance: commit + sha256 of 18 sources | 28/28 |
| SP-2 | `.so` mtime post-dates the last C++ edit, and lives outside the repo tree | 28/28 |
| SP-3 | Engine is `native`, **and observed dispatch matches** | 28/28 |
| SP-4 | Alphabet clean on the probe's own candidate stream | 28/28 |
| SP-5 | Both hosts | 28/28 |
| SP-6 | T06's fallback counters live and finite | 28/28 |

**SP-3's negative control is the one that matters**, because it is the check that
was previously passing while proving nothing. `canonical.py:349` used to read
`DEFAULT_BACKEND` directly instead of resolving through `backends.engine()`, so
`ISALSR_ENGINE=python` was *reported* as honoured while C++ kept executing. The
probe now asserts on **observed dispatch** via a call counter on
`_cpp_ext.fast_canonical_string`:

| Run | `reported_engine` | `cpp_actually_invoked` | `reported_matches_observed` |
|---|---|---|---|
| normal ×28 | `cpp` | `true` | `true` |
| `ISALSR_ENGINE=python` ×28 | `python` | **`false`** | `true` |

Build recorded per task: gcc 11.1.0, `x86-64-v3`, AVX2 on, AVX-512 off,
`build_hash 298fc1188bf1b051`.

---

## 5. Results

Four problems × three arms × two hosts, plus four shadow ON/OFF cells. 28/28
cells produced a validating `run_log.json`.

### 5.1 The three-arm comparison (separate runs, host-native key)

$\rho = \text{total candidates} / \text{distinct keys}$. $R = 1 - 1/\rho$ is the
fraction of the stream removed as redundant. The last column,
$(R_{\mathrm{isalsr}} - R_{\mathrm{hash}}) / R_{\mathrm{isalsr}}$, is the share of
all detectable redundancy that **only** the canonical form finds.

| Host | Problem | $\rho_{\text{hash}}$ | $\rho_{\text{isalsr}}$ | $R_{\text{hash}}$ | $R_{\text{isalsr}}$ | needs 1-WL |
|---|---|---|---|---|---|---|
| Bingo | Nguyen-1 | 1.7160 | 1.7757 | 41.73 % | 43.69 % | 4.5 % |
| Bingo | Nguyen-7 | 1.7465 | 1.8249 | 42.74 % | 45.20 % | 5.4 % |
| Bingo | Pagie-1 | 1.7722 | 1.8192 | 43.57 % | 45.03 % | 3.2 % |
| Bingo | I.15.10 | 1.7418 | 1.7972 | 42.59 % | 44.36 % | 4.0 % |
| **Bingo** | **mean** | | | | | **4.3 %** |
| UDFS | Nguyen-1 | **1.0000** | 2.1924 | 0.00 % | 54.39 % | 100 % |
| UDFS | Nguyen-7 | **1.0000** | 2.2426 | 0.00 % | 55.41 % | 100 % |
| UDFS | Pagie-1 | **1.0000** | 1.8359 | 0.00 % | 45.53 % | 100 % |
| UDFS | I.15.10 | **1.0000** | 1.3956 | 0.00 % | 28.35 % | 100 % |
| **UDFS** | **mean** | | | | | **100 %** |

Two opposite results, both real:

- **UDFS: the naive baseline finds nothing at all.** $\rho_{\text{hash}} = 1.0000$
  on all four problems — not "small", *zero* duplicate detections. UDFS enumerates
  systematically and does not re-emit the same `CompGraph` twice, so a hash on its
  own representation is structurally incapable of firing. Every one of the
  28–55 % reductions is attributable to isomorphism-invariance. Consistency check:
  `r2_test` for the hash arm equals the baseline **exactly** in all four UDFS
  cells, which is what must happen if dedup removed nothing.
- **Bingo: the naive baseline captures most of it — this is the concession
  (AC-8).** A plain host-native hash recovers $\approx 42$–$44$ % redundancy
  against the canonical form's $\approx 44$–$45$ %; only **4.3 %** of detectable
  redundancy requires $1$-WL. Bingo is a stochastic GP that re-derives
  *byte-identical* `command_array`s, and those are exactly what a naive hash
  catches. **This must be reported without softening.**

### 5.2 Same-stream shadow ladder — and a caution about it

Adapter-order sketches on the `isalsr` candidate stream, so all rungs see
identical input:

| Host | Problem | $\rho_{\text{ins}}$ | $\rho_{\text{topo}}$ | $\rho_{\text{comm}}$ | $\rho_{\text{isalsr}}$ | comm/isalsr |
|---|---|---|---|---|---|---|
| Bingo | Nguyen-1 | 1.7633 | 1.7577 | 1.7849 | 1.7757 | 1.0052 |
| Bingo | Nguyen-7 | 1.8052 | 1.8038 | 1.8193 | 1.8249 | 0.9969 |
| Bingo | Pagie-1 | 1.8039 | 1.8147 | 1.8153 | 1.8192 | 0.9979 |
| Bingo | I.15.10 | 1.7798 | 1.7728 | 1.7764 | 1.7972 | 0.9885 |
| UDFS | Nguyen-1 | 1.4280 | 2.0018 | 2.0717 | 2.1924 | 0.9450 |
| UDFS | Nguyen-7 | 1.4168 | 2.0528 | 2.1250 | 2.2426 | 0.9476 |
| UDFS | Pagie-1 | 1.3134 | 1.6977 | 1.7368 | 1.8359 | 0.9460 |
| UDFS | I.15.10 | 1.1849 | 1.3763 | 1.3977 | 1.3956 | 1.0015 |

Rung monotonicity holds within sampling error throughout.

> **⚠ These sketches measure adapter output, and they therefore overstate the
> naive baseline — by a factor this probe quantifies exactly.**
>
> On UDFS, the adapter-order rung 3 says a fixed-order serialisation captures
> **94.6 %** of the reduction. The host-native arm in §5.1, on the same host and
> the same problems, says it captures **0 %**. The entire difference is the
> adapter's renumbering, which pre-canonicalises the DAG before the "naive"
> hash sees it.
>
> This is the §3.2 error reproduced from two independent measurements inside one
> probe, and it is the single most important methodological point in this ticket:
> **the object you serialise decides the answer.** Any future analysis that
> reports a fixed-order $\rho$ must state which representation it was computed on.

`shadow_distinct_host_native` — the fourth sketch, which removes this caveat by
measuring the host's own representation on the same stream — is **absent from
this probe**. It was added in commit `a24d73c`, *after* the probe was submitted
at `a4206b8`; at that commit `record_shadow` took no host argument. This is a
provenance fact, not a defect, but it means the same-stream host-native
comparison is **unverified on Picasso** and must be present in C2.

### 5.3 Quality and cost

| Host | Problem | $R^2$ base | $R^2$ hash | $R^2$ isalsr | canon hash (s) | canon isalsr (s) |
|---|---|---|---|---|---|---|
| Bingo | Nguyen-1 | 1.000000 | 1.000000 | 1.000000 | 1.48 | 1.26 |
| Bingo | Nguyen-7 | 1.000000 | 1.000000 | 1.000000 | 32.48 | 101.82 |
| Bingo | Pagie-1 | 0.840833 | 0.180970 | 0.623885 | 49.77 | 93.20 |
| Bingo | I.15.10 | 0.999998 | 0.999997 | 0.999970 | 50.29 | 87.29 |
| UDFS | Nguyen-1 | 0.977508 | 0.977508 | 0.992654 | 0.23 | 0.98 |
| UDFS | Nguyen-7 | 0.998937 | 0.998937 | 0.998937 | 0.23 | 1.02 |
| UDFS | Pagie-1 | −0.001218 | −0.001218 | −0.001218 | 0.10 | 0.36 |
| UDFS | I.15.10 | 0.984536 | 0.984536 | 0.984536 | 0.07 | 0.22 |

The key-computation cost of the naive hash is roughly **2–3× cheaper** than
canonicalisation on Bingo and **3–4× cheaper** on UDFS. That is expected and
should be conceded: a byte hash is cheaper than a graph canonical form. The
paper's efficiency claim is defined on $T_{\text{search}} = \text{wall} -
\text{canon}$ (T01 AC-6) and is insensitive to this by construction.

**Do not read the Pagie-1 $R^2$ column as an arm effect.** Single seed, 25-minute
budget, on a problem no arm solves; the spread (0.18 / 0.62 / 0.84) is seed noise
on a hard problem, not evidence about deduplication. It is recorded because
suppressing an unflattering column is how one gets caught.

### 5.4 AC-10 — shadow counters must not OOM: **PASS**

`MaxRSS` for the ON/OFF pairs, production `--mem` (Bingo 48 G, UDFS 16 G):

| Pair | shadow ON | shadow OFF | Δ |
|---|---|---|---|
| Bingo isalsr Pagie-1 | 393 536 K | 394 496 K | **−0.24 %** |
| UDFS isalsr Pagie-1 | 492 060 K | 492 764 K | **−0.14 %** |
| UDFS isalsr Nguyen-1 | 443 612 K | 422 772 K | +4.9 % |
| Bingo isalsr Nguyen-1 | 2 464 K | 2 100 K | unusable |

No OOM kill anywhere. Two of the three usable deltas are **negative**, so the
+4.9 % is allocator noise, not signal — as it must be: four HLL sketches at
$p=16$ total 256 KB and cannot account for 20 MB. The delta does not grow with
stream length, so the implementation has not silently fallen back to exact sets.
**Pass.**

The Bingo/Nguyen-1 pair is unusable because both tasks ran under SLURM's
accounting sampling interval (28 s and 21 s); Nguyen-1 is trivial for Bingo, so
the short runtimes are convergence, not failure.

Peak RSS across all 28 cells was 350–545 MB against 48 G requested. This does not
refute the historical 128 G fragmentation requirement, which accumulates over
12 h, but at 25 minutes there is no trace of it.

### 5.5 An unrelated campaign-scale defect found by this probe

Task 7 (Bingo baseline Pagie-1) was killed at its wall limit **20 min 33 s after
its search had finished cleanly**. Cause: two unbounded, untimed SymPy calls in
`translator.to_run_log()` — `sympy.simplify(found - true)` in
`solution_recovered`, and one `simplify` **per traversal node** in
`jaccard_index`. Both sat inside `except Exception`, which catches errors but is
blind to slowness. `run_log.json` is written *after* this step, so the run left
no artefact at all.

Re-running the identical cell (same seed, problem, config, and the same unbounded
code) completed in 24:49 with a ~20 s tail. **The pathology is stochastic**:
Bingo stops on wall clock, so the generation reached — and hence the final
individual — varies with node speed and load. It cannot be predicted or
reproduced on demand, only bounded. Because which individual a run ends on
depends on the search trajectory, and the arms change that trajectory, the losses
are **not arm-neutral**: this is a bias risk in the paired design, not merely an
availability risk.

Fixed 2026-08-02: both calls bounded at 300 s, returning `None` (undetermined,
excluded from the aggregate) rather than a fabricated `False`/`0.0`. The
mathematics is unchanged — 20 pinned metric values are byte-identical before and
after (`tests/unit/test_analyzer_metrics.py`). The tempting optimisation of
dropping the per-node `simplify` was **measured and rejected**: it changes
published Jaccard values on the Pythagorean identity (0.1429 → 0.0) and on
cancelling rationals (0.2222 → 0.1), and the counter-example is pinned as a
regression test.

---

## 6. What this does not establish

1. **Nothing here is a paper number.** One seed, four problems, 25-minute
   budgets. C2 supplies the numbers.
2. **No statistical claim.** No dispersion, no CPDT, no critical-difference
   diagram — those need 20 seeds across $D1 \cup D2$.
3. **The same-stream host-native comparison is unverified on Picasso** (§5.2).
4. **$k$-stratification is absent.** The predicted mechanism is that the gap
   widens with internal node count $k$; four problems cannot show that.
5. **UDFS's $\rho_{\text{hash}} = 1.0000$ is exact on four problems, not proven in
   general.** The mechanism (systematic enumeration does not re-emit identical
   representations) predicts it holds broadly, but it is a hypothesis until C2.

---

## 7. Status against the ticket's acceptance criteria

| AC | Requires | Status |
|---|---|---|
| AC-0 | §7 work log maintained | **met** |
| AC-1 | Three serialisations, sound + incomplete + monotone on the 14,841-DAG corpus, both backends | **met** |
| AC-2 | Mode 1 replay: $\rho_{exact}, \rho_{comm}, \rho_{iso}, \rho_{total}$ per method/problem, with dispersion, stratified by $k$ | **partial** — computed offline; dispersion and $k$-stratification need C2 |
| AC-2b | SP-1…SP-6 six-row table per probe; SP-7 on both hosts | **met** (§4.2) |
| AC-3 | Mode 2: the `hash` arm of C2, ≈2,800 runs at 20 seeds | **NOT MET — C2 has not launched** |
| AC-4 | Per-DAG cost on Picasso under C++ and the decomposed alphabet, with $S$ for all three arms | **partial** — §5.3 has costs; $S$ needs C2 |
| AC-5 | Three-arm statistical comparison + critical-difference diagram | **NOT MET — needs C2** |
| AC-6 | Worked isomorphic-pair example | **met** (§2.4) |
| AC-7 | Paper text states sound-but-incomplete explicitly | **NOT MET — response letter not yet written** |
| AC-8 | Report competitiveness without softening | **met** — Bingo 4.3 % recorded in §5.1 |
| AC-9 | §8 filled | open |
| AC-10 | Shadow counters must not OOM, measured on both hosts | **met** (§5.4) |

**This ticket cannot be closed.** Four criteria are unmet and three of them
(AC-3, AC-4, AC-5) are blocked on campaign C2, which is itself gated by T05. AC-7
is blocked on the response-letter pass (`/review-answer`).

What *is* closed is the engineering question the probe was built to answer: the
arm runs correctly on Picasso, on both hosts, under the C++ engine and the
decomposed alphabet, with the shadow counters carrying no memory risk.

### Carried into C2

1. `shadow_distinct_host_native` must be present and non-null in every `isalsr`
   run log — it is the only same-stream measurement free of the §5.2 caveat.
2. `record_shadow`'s bare `except Exception` still hides import errors as zero
   counters. It has produced a silent all-zero result once already. Filed, not
   fixed.
3. `metadata.hardware` carries no `engine` field, so C1.14 cannot be verified
   from a `RunLog` alone. Filed for T02/A7.
4. Report $\rho$ stratified by $k$, and the UDFS $\rho_{\text{hash}} = 1$ result
   across the full problem set — it is either a clean structural claim or it
   breaks somewhere, and both outcomes are worth knowing.

---

## 8. References

- Flajolet, Fusy, Gandouet & Meunier (2007). *HyperLogLog: the analysis of a
  near-optimal cardinality estimation algorithm.* AofA'07, DMTCS Proc. AH, 137–156.
- McKay & Piperno (2014). *Practical graph isomorphism, II.* Journal of Symbolic
  Computation 60, 94–112. DOI:10.1016/j.jsc.2013.09.003
- Cai, Fürer & Immerman (1992). *An optimal lower bound on the number of variables
  for graph identification.* Combinatorica 12(4), 389–410. DOI:10.1007/BF01305232
- Weisfeiler & Leman (1968). *The reduction of a graph to canonical form and the
  algebra which appears therein.* NTI Series 2(9), 12–16.
- Shervashidze, Schweitzer, van Leeuwen, Mehlhorn & Borgwardt (2011).
  *Weisfeiler-Lehman graph kernels.* JMLR 12, 2539–2561.
- Burlacu, Kronberger & Affenzeller (2019/2020). Operon / hash-based tree
  deduplication in GP — applies to **trees**, not DAGs; addressed in prose in the
  response letter.

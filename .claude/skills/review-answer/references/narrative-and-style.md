# The narrative spine and the style contract

Two separate things, both mandatory. The **spine** decides what goes where. The
**contract** decides how it reads. A draft can satisfy one and fail the other.

---

## Part 1 — The narrative spine

Fifteen moves. Skip the ones a comment does not need; never reorder 12 and 13
earlier.

### 1. Concede in the first sentence

The reviewer found something. Say so, factually, and stop. No gratitude ritual,
no cushioning, no restating the comment back at them.

> The reviewer is correct. The routine is invoked in the first line of the
> pseudocode of Table 3 and is defined nowhere in the manuscript. We have
> corrected this.

Three sentences: verdict, the defect stated as fact, the remedy. If the reviewer
is *not* correct, say that just as plainly and give the counter-evidence in the
same paragraph — but check the code first, because they usually are.

### 2. State the stakes, and signal length if the answer is long

A reviewer who does not know why an answer runs two pages reads it as padding.
Tell them what class of thing this is.

> Because the step establishes the reachability condition assumed by Theorems 3.13
> and 3.15 rather than performing cosmetic preprocessing, the revision gives it a
> numbered definition, pseudocode, a validity argument, a complexity bound, and
> the measurements that motivate it; we summarise these below.

The list of deliverables *is* the length signal. Do not write "we answer at
length" or "this is a keystone issue".

### 3. Pre-announce the awkward part

If something the reviewer quoted was wrong, or you are about to disclose a defect,
say now that it is coming and where.

> We also note that the one-line description quoted by the reviewer does not
> correctly characterise the operation, and we address that at the end of this
> response.

This buys the right to defer it to step 12 without looking evasive.

### 4. Root cause — structural, never historical

Explain why the object exists from the mathematics or the design, not from the
project's timeline. "Every insertion instruction creates a node together with an
edge, so a leaf with no in-edge has no encoding" is a root cause. "We added this
in an early version to fix a crash" is history, and does not belong in a
manuscript.

End this move by naming the object and, if a figure exists, pointing at it:
"Figure 1 shows both sides of the step on a small example."

### 5. The formal object, as a display block

Definition, algorithm, theorem statement. Set it off; do not inline a definition
the reviewer will want to quote. Reference it from the prose. This is the one
place where a non-prose block is not a style failure — it is the deliverable the
reviewer asked for.

### 6. Cost or complexity

Only if the object is a procedure. Give the worst case, then say whether it is
approached in practice and why.

> The worst case is O(|C| m (|V|+|E|)). This bound is not approached on host-solver
> output, where the first anchor always succeeds for the reason given below, so the
> cost reduces to |C| acyclicity checks.

### 7. Properties, in prose, best last

Enumerated properties are how you *think*; a paragraph is how you *write*. Chain
them with "It … It … It … Finally, and most importantly for the theory, it …" and
put the property that answers the reviewer's actual worry at the end, where it
lands.

### 8. The measurement

Four things, always: what was instrumented, which populations, what $N$, what
result. Naming the populations is what makes the number checkable.

> We measured how often the condition is violated by instrumenting the point at
> which an expression DAG enters the representation and recording, for each DAG,
> whether some non-variable node lacks a variable ancestor, both before and after
> the step. Four populations were used: …

Wide results go in a captioned table; the prose then says what the table means,
not what it contains.

### 9. The mechanism

A rate on its own invites "how do you know that is the real cause?". A
stratification answers it before it is asked.

> This is the profile implied by the hypothesis that the condition is violated
> precisely when the expression contains at least one constant terminal, since the
> probability that no internal node is a constant decays geometrically in k.

If a competing explanation survives the stratification, say so and say it is
under investigation. Do not present a partial mechanism as settled.

### 10. Supporting measurements, in prose

The second, third and fourth measurements go into one paragraph opened by a
counter — "Three further measurements support the properties stated above" — and
then flow. Resist the urge to label them.

### 11. A limitation, volunteered

State the exact conditions under which the claim fails, then why they cannot
arise in the reported setting. Volunteering this is worth more than the paragraph
costs, and it forecloses the round-2 version of the same objection.

> We record one limitation of the definition. The rule "least index i that does
> not close a cycle" is stated over node indices, and node indices are what an
> isomorphism permutes. If a DAG has … then … The second condition does not occur
> in host-solver output, as noted above, and the first cannot occur in any DAG
> satisfying the hypothesis …

Do not label it "one honest caveat" — the honesty is in the content, and claiming
it in the label undercuts it.

### 12. The awkward part, delivered late

Now that the measurements are on the table, address what the reviewer quoted, or
the defect you pre-announced. Give the concrete failure: the input, the
mechanism, the number.

> An IsalSR string can direct an edge into a variable …, so x₁ may lie downstream
> of a constant; the replacement edge then closes a directed cycle and must be
> refused, while the original in-edge has already been deleted … On 10⁵ randomly
> generated DAGs the relocation produces 48 canonicalisation failures of this
> kind, against none for the operation defined above.

### 13. "No reported number changes"

Say it plainly, once, and only if a policy-invariance measurement supports it.
This is often the single most important sentence in the answer, and it is worth
nothing if it is an assumption.

### 14. Cross-link sibling comments

If two comments share an answer, say so and give the number that closes both.
Reviewers notice when the letter treats one defect as two.

### 15. `\changeref{}` with concrete locations

Section, table, appendix. Not "the manuscript was updated".

---

## Part 2 — The style contract

### Positive rules — do these

| Rule | Why |
|---|---|
| Open with the verdict. | The reviewer is scanning for whether you agreed. |
| Continuous prose; paragraphs carry the argument. | A response letter is a scientific argument, not a status report. |
| One idea per paragraph, stated in its first sentence. | Lets the reviewer skim without losing the thread. |
| Quantify everything: $N$, population, protocol. | "85.88% of Bingo candidates (132,746 / 154,568)" is checkable; "most candidates" is not. |
| Name how each population was produced. | Provenance is what makes a number auditable. |
| Academic "we", active voice, present tense for the revision. | "The revision adds", "We measured", "We have corrected". |
| Display blocks only for genuine objects: definition, algorithm, table, figure. | These are deliverables; everything else is prose. |
| Reuse the manuscript's own terminology and notation. | A new synonym reads as a new concept. |
| Volunteer limitations in your own voice. | Cheaper now than in round 2. |
| Reference every float by `\ref`. | "The table below" breaks when the float moves. |
| End with concrete manuscript locations. | The AE checks that the change exists. |

### Negative rules — never do these

**Structure**

- No `\paragraph{}` outlines, no `(a)`/`(b)`/`(c)` labelled parts.
- No `enumerate` or `description` as the backbone of an answer. A list inside a
  single move is occasionally fine; a list *as* the answer is not.
- No `\textbf{E1 ---}`, `\textbf{Step 2 ---}` style run-in labels.
- No section headings inside a response block.

**Punctuation and rhetoric**

- No em-dashes as a rhetorical device. They are the strongest tell. Use a comma,
  a semicolon, a colon, or two sentences. (The shipped R1.3 answer contains
  **zero** in its prose.)
- No dramatic or "poetic" framing: *The mechanism, not just the rate* · *The
  contrast is the finding* · *load-bearing* · *the keystone* · *worth more than
  being told in round 2* · *this is the whole argument in one line*.
- No significance inflation: *crucially* · *importantly* · *strikingly* ·
  *remarkably* · *it is worth emphasising* · *notably*.
- No self-congratulation on candour: *one honest caveat* · *recorded rather than
  buried* · *we state this plainly rather than waiting to be found*.
- No apology loops. Concede once, in sentence one, and never again.
- Bold for numbers that matter, not for emphasis in running prose.

**Content**

- No implementation jargon in reviewer-facing text. This is the one that slipped
  through review twice: *raises* · *returns False* · *no-op* · *monkey-patch* ·
  *backend* · *adapter* used without definition · bare function or variable names.
  Say what happens mathematically, not what the code does about it.
- No project-internal identifiers: ticket IDs (`T07`, `AC-6`), dates, script
  names, agent or session references, `\begin{comment}` line numbers.
- No development history. See the honesty gate in `SKILL.md` §5.3 for where the
  line sits.
- No promises. Describe what the revision *does*, not what it will do.
- No hedging on verified facts. If it is measured, assert it.
- Never touch `rcomment` content.

---

## Part 3 — Before and after

Every pair below is a real edit made to the shipped R1.3 answer.

**Opening — cut the ritual and the self-commentary**

> ✗ The reviewer is correct, and we thank them for the observation. The routine
> was invoked once, in the first line of the pseudocode of Table 3, and was
> nowhere defined, justified, or analysed. The omission matters more than a
> missing definition usually would, because …

> ✓ The reviewer is correct. The routine is invoked in the first line of the
> pseudocode of Table 3 and is defined nowhere in the manuscript. We have
> corrected this. Because the step establishes the reachability condition assumed
> by Theorems 3.13 and 3.15 rather than performing cosmetic preprocessing, …

**Experiment labels — delete them, keep the content**

> ✗ `\textbf{E2 --- The mechanism, not just the rate.}` Stratifying the Bingo
> stream by …

> ✓ Stratifying the Bingo stream by the number of internal nodes $k$ gives $0.00\%$
> at $k=0$ ($128$ DAGs), …

**A property list becomes a paragraph**

> ✗ `\begin{description}` `\item[N1 (Edge-monotone).]` … `\item[N2 (Idempotent).]`
> … `\end{description}`

> ✓ Four properties of the operation are used in the revision. It adds edges and
> never removes one, so the nodes, the labels and the operand orders are preserved
> …; reachability, once established, cannot be destroyed. It is idempotent, since
> … Finally, and most importantly for the theory, it is the identity on the
> hypothesis class: …

**A framing sentence becomes a factual one**

> ✗ The contrast is the finding, and it explains why the omission escaped us: DAGs
> produced by S2D satisfy the condition by construction …

> ✓ DAGs produced by S2D satisfy the condition by construction, because every node
> is created from an existing one and the m variables are pre-inserted, so a
> variable ancestor propagates by induction. The populations on which we
> originally validated the representation are therefore exactly the populations on
> which the step is inactive, which is why its absence from the manuscript went
> unnoticed.

**Self-congratulatory label removed**

> ✗ `\paragraph{(f) One honest caveat, and where we place the step because of it.}`

> ✓ We record one limitation of the definition.

**Implementation jargon removed (also fixed in the figure)**

> ✗ canonicalisation raises

> ✓ no canonical string exists

The second says what is mathematically true; the first says what the software
does about it, and only a reader of the source knows what "raises" means.

# Correction of five AI Feynman problem definitions (2026-08-02)

**Internal engineering record. Not reviewer-facing.**
Decision: Mario, 2026-08-02. Found while building `benchmarks/datasets/feynman_catalogue.py`
under T05. Nothing about this goes into the manuscript or the response letter.

---

## 1. What was wrong

Five of the 24 problems the suite labels as AI Feynman equations did not encode the
equation their id names. Established by comparing each suite definition against two
independent renderings of the database: the canonical `FeynmanEquations.csv`
(recovered from a Wayback snapshot of the original MIT distribution, 100 base rows
verified) and PMLB's `feynman_*` `metadata.yaml` files. **The two sources agree on all
99 shared equations, 0 mismatches**, so the reference is not in doubt.

| Suite id | Was | Should be | Defect |
|---|---|---|---|
| I.39.10 | `0.5·p_r·V` | `1.5·p_r·V` | coefficient: `1/2` where the database has `3/2` |
| I.12.4 | `q1/(4π·r·c)` | `q1·r/(4π·ε·r³)` | different function: falls as `1/r`, should fall as `1/r²`; the third variable is `ε`, not `c` |
| II.3.24 | `p·r/(4π)` | `Pwr/(4π·r²)` | different function: `r` in the numerator instead of `r²` in the denominator |
| II.11.27 | `n0·e^(−μB/kT) + n0·e^(μB/kT)` | `n·α/(1 − n·α/3)·ε·Ef` | entirely different equation — the implemented target is a paramagnetism-style two-exponential sum, the database's II.11.27 is Clausius–Mossotti polarisation |
| III.17.37 | `f0/√((ω−ω0)² + γ²/4)` | `β·(1 + α·cos θ)` | entirely different equation — the implemented target is a Lorentzian resonance lineshape, which is not in the database at all |

The first three are transcription errors of the named equation. The last two are
unrelated equations carrying a Feynman id.

## 2. What was done

The **target functions were corrected**; the **ids were kept**. Rationale: the paper
has already been reviewed under these ids, and renaming would propagate through the
supplementary tables, the tier docstrings and the bottleneck analysis for no gain.

Files touched:

- `benchmarks/datasets/feynman.py` — `I.39.10`, `I.12.4`, `II.3.24`
- `benchmarks/datasets/hard.py` — `II.11.27`, `III.17.37`, and the module docstring
  (the `sqrt` note no longer applies to `III.17.37`)

Variable ranges were taken from the database rather than carried over:
`II.11.27` now samples `n, α ∈ [0,1]` and `ε, Ef ∈ [1,2]`, which keeps
`1 − n·α/3 ∈ [2/3, 1]` so the denominator never approaches zero.

## 3. Verification

Each corrected definition was evaluated on generated data and compared against the
catalogue formula, lambdified independently through sympy from the database string:

```
I.39.10     nv_ours=2 nv_db=2  finite=True  MATCHES DB
I.12.4      nv_ours=3 nv_db=3  finite=True  MATCHES DB
II.3.24     nv_ours=2 nv_db=2  finite=True  MATCHES DB
II.11.27    nv_ours=4 nv_db=4  finite=True  MATCHES DB
III.17.37   nv_ours=3 nv_db=3  finite=True  MATCHES DB
```

`MATCHES DB` is a relative error below `10⁻¹²`. Arity now agrees with the database on
all five.

## 4. Consequences that are real and are not being papered over

These were named **before** the change was made, not discovered after:

1. **`III.17.37` is no longer a hard problem.** It drops from 4 variables to 3, `k`
   falls to roughly 4, and it no longer requires `sqrt`. The hard tier now contains a
   problem that does not meet the tier's own difficulty rationale.
2. **`II.11.27` is now structurally close to `II.11.28`**, which is already in the
   cherrypicked tier as `1 + n·α/(1 − n·α/3)`. The paper's criterion (iv)
   (complementary coverage) is in tension with having both.
3. **The bottleneck-type analysis no longer applies to these two.**
   `docs/md_files/changes/bottleneck_type_analysis.md` classifies `II.11.27` as
   `none_trivial` and `III.17.37` as `structural`, both derived from the *old*
   targets. Any claim resting on those two rows needs re-deriving from C2 data.
4. **C1↔C2 continuity breaks on five of the 50 D1 problems.** The submitted campaign's
   per-problem numbers for `I.39.10`, `I.12.4`, `II.3.24`, `II.11.27` and `III.17.37`
   were produced on different target functions and are not comparable to C2's. The
   continuity table (`EXECUTION-PLAN.md` §7) must exclude these five rows or it will
   report a spurious shift.
5. `CLAUDE.md` describes `II.11.27` as "paramagnetism, 4 vars, two opposite-sign exp
   branches" and lists it as the primary diversity candidate. That description is now
   stale.

## 5. Open items

- **T09** — Appendix D.1 must document all five by their corrected expressions.
- **T02** — the C1↔C2 continuity table must drop these five rows (consequence 4).
- **CLAUDE.md** — the `II.11.27` diversity-candidate description needs updating, and
  the hard-tier difficulty rationale needs a note about `III.17.37`.
- Whether the hard tier should regain a tenth genuinely-hard problem to replace
  `III.17.37` is **not decided** and is outside T05.

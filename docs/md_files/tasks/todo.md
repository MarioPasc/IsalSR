# IsalSR -- Task Tracking

## Phase 1: Core Data Structures (DONE -- 2026-03-17)
- [x] types.py -- type aliases (NodeId, CdllIndex, InstructionToken, VALID_LABEL_CHARS)
- [x] errors.py -- exception hierarchy (IsalSRError, CycleDetectedError, InvalidTokenError, InvalidDAGError, EvaluationError)
- [x] node_types.py -- NodeType enum (13 types), LABEL_CHAR_MAP, ARITY_MAP, OperationSet
- [x] cdll.py -- CDLL (verbatim from IsalGraph)
- [x] labeled_dag.py -- LabeledDAG (dual adjacency, cycle detection, topological sort, label-aware isomorphism, backtracking support)
- [x] Unit tests: 80 tests (14 CDLL, 44 LabeledDAG, 22 NodeTypes) -- all passing
- [x] Proposal guard: 10/10 PASS alignment with paper hypothesis
- [x] Custom agents: proposal-guard (sonnet), test-runner (haiku), implementation-scientist (opus)
- [x] Custom skill: /test-and-verify

## Phase 2: String-to-DAG + Evaluator (DONE -- 2026-03-17)
- [x] string_to_dag.py -- two-tier tokenizer (V/v compound tokens) + S2D executor with DAG cycle enforcement
- [x] dag_evaluator.py -- topological sort evaluation with protected ops (log, div, sqrt, exp, pow)
- [x] Unit tests: 83 new tests (50 S2D + 33 evaluator) -- all 163 total passing
- [x] End-to-end S2D → evaluate integration tests
- [x] ruff + mypy strict clean
- [x] Proposal guard alignment check

## Phase 3: DAG-to-String + Round-Trip (DONE -- 2026-03-17)
- [x] dag_to_string.py -- greedy D2S with two-char tokens, all B2-B8 fixes, directed out_neighbors
- [x] generate_pairs_sorted_by_sum -- |a|+|b| cost ordering (B2 fix)
- [x] Round-trip tests: 25 tests (13 string-based + 3 fixture + 6 programmatic + 3 evaluation)
- [x] D2S unit tests: 21 tests (pair gen, basics, labels, edges, reachability, complex)
- [x] experiments/plotting_styles.py adapted: IsalSR token colors, two-char token renderer
- [x] All 209 tests passing, ruff clean, mypy strict clean
- [x] Round-trip property verified: S2D(D2S(D)) ~ D for all tested DAGs
- [x] Evaluation preservation: numerical results identical before/after round-trip

## Phase 4: Canonical String — THE CORE CONTRIBUTION (DONE -- 2026-03-17)
- [x] canonical.py -- exhaustive backtracking from x_1, pruned variant with 6-tuple, levenshtein, dag_distance
- [x] compute_structural_tuples -- 6-component directed BFS (in_N1, out_N1, in_N2, out_N2, in_N3, out_N3)
- [x] algorithms/base.py -- D2SAlgorithm ABC
- [x] algorithms/greedy_single.py -- GreedySingleD2S (wraps DAGToString from x_1)
- [x] algorithms/greedy_min.py -- GreedyMinD2S (all VAR nodes, lexmin shortest)
- [x] algorithms/exhaustive.py -- ExhaustiveD2S (wraps canonical_string)
- [x] algorithms/pruned_exhaustive.py -- PrunedExhaustiveD2S (wraps pruned_canonical_string)
- [x] 43 canonical tests: invariance (5), discrimination (4), pruned (5), distance (4), algorithms (7), evaluation (2)
- [x] All 252 tests passing, ruff clean, mypy strict clean
- [x] Canonical invariance verified: isomorphic DAGs → same canonical string
- [x] Canonical discrimination verified: different expressions → different canonicals
- [x] Exhaustive == Pruned on small DAGs confirmed

## Phase 5: Adapters & Evaluation (DONE -- 2026-03-17)
- [x] adapters/base.py -- DAGAdapter ABC (Bridge pattern, Generic[T])
- [x] adapters/networkx_adapter.py -- nx.DiGraph ↔ LabeledDAG with label/var_index/const_value attrs
- [x] adapters/sympy_adapter.py -- SymPy Expr ↔ LabeledDAG (to_sympy, from_sympy, shared subexpr detection)
- [x] evaluation/protected_ops.py -- NumPy-vectorized protected log, div, sqrt, exp, pow, clamp
- [x] evaluation/fitness.py -- R², NRMSE, MSE, evaluate_expression (vectorized DAG eval)
- [x] evaluation/constant_optimizer.py -- BFGS optimization of CONST nodes (scipy.optimize.minimize)
- [x] Integration tests: 26 new (6 NetworkX, 7 SymPy, 10 fitness, 3 constant optimizer)
- [x] All 278 tests passing, ruff clean, mypy strict clean

## Phase 6: Search Operators + Property Tests (DONE -- 2026-03-17)
- [x] Encapsulation fix: LabeledDAG.set_const_value() added, constant_optimizer refactored
- [x] search/operators.py -- token-aware point_mutation, insertion, deletion, subsequence, crossover (1pt + 2pt)
- [x] search/random_search.py -- random_isalsr_string + random_search with canonical deduplication
- [x] search/hill_climbing.py -- multi-restart hill climbing with mandatory canonicalization
- [x] search/population.py -- Population class with tournament selection, evolutionary loop, elitism
- [x] Property tests: 8 Hypothesis tests (roundtrip 2, acyclicity 2, canonical 2, evaluation 2)
- [x] All 286 tests passing, ruff clean, mypy strict clean
- [x] MANDATORY canonicalization enforced in all search algorithms

## Phase 7: Benchmarks & Experiments (DONE -- 2026-03-17)
- [x] evaluation/fitness.py -- added reward() function (Liu2025 Eq. 12: 1/(NRMSE+1))
- [x] benchmarks/datasets/nguyen.py -- 12 Nguyen expressions EXACT from Liu2025 Table 1 + generate_data()
- [x] benchmarks/datasets/feynman.py -- 10 Feynman equations from Liu2025 Table 2 + generate_data()
- [x] benchmarks/datasets/srbench.py -- SRBench metadata (La Cava et al., 2021 NeurIPS)
- [x] experiments/scripts/search_space_analysis.py -- THE KEY EXPERIMENT: O(k!) reduction measurement
- [x] experiments/scripts/run_random_search.py -- CLI benchmark runner
- [x] experiments/scripts/run_hill_climbing.py -- CLI benchmark runner
- [x] experiments/scripts/run_gp.py -- CLI benchmark runner (Population.evolve)
- [x] experiments/scripts/analyze_results.py -- Summary stats + LaTeX tables
- [x] Nguyen benchmark spot-checks verified correct at known values
- [x] All 286 tests passing, ruff clean, mypy strict clean

## PROJECT COMPLETE
All 7 phases implemented. 286 tests, 72+ proposal-guard PASS.
Ready for arXiv preprint and IEEE TPAMI submission.

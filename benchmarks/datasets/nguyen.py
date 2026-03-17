"""Nguyen symbolic regression benchmark definitions.

12 standard Nguyen benchmarks used across SR literature.
Reference: Uy et al. (2011). Semantically-based crossover in GP.

To implement:
    NGUYEN_BENCHMARKS: list of dicts, each containing:
        - name: str (e.g., "Nguyen-1")
        - expression: str (human-readable, e.g., "x^3 + x^2 + x")
        - num_variables: int
        - x_range: tuple[float, float]
        - target_fn: Callable (numpy vectorized)

    Nguyen-1:  x^3 + x^2 + x                    (x in [-1, 1])
    Nguyen-2:  x^4 + x^3 + x^2 + x              (x in [-1, 1])
    Nguyen-3:  x^5 + x^4 + x^3 + x^2 + x        (x in [-1, 1])
    Nguyen-4:  x^6 + x^5 + x^4 + x^3 + x^2 + x  (x in [-1, 1])
    Nguyen-5:  sin(x^2) * cos(x) - 1             (x in [-1, 1])
    Nguyen-6:  sin(x) + sin(x + x^2)             (x in [-1, 1])
    Nguyen-7:  log(x + 1) + log(x^2 + 1)         (x in [0, 2])
    Nguyen-8:  sqrt(x)                            (x in [0, 4])
    Nguyen-9:  sin(x) + sin(y^2)                  (x,y in [-1, 1])
    Nguyen-10: 2 * sin(x) * cos(y)               (x,y in [-1, 1])
    Nguyen-11: x^y                                (x,y in [0, 1])
    Nguyen-12: x^4 - x^3 + y^2/2 - y             (x,y in [-1, 1])
"""

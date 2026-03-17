"""Feynman physics equations benchmark definitions.

50+ equations from the AI Feynman dataset (fundamental physics).
Reference: Udrescu & Tegmark (2020). AI Feynman: A physics-inspired method
for symbolic regression. Science Advances 6(16).

To implement:
    FEYNMAN_BENCHMARKS: list of dicts with name, expression, num_variables,
    variable_ranges, target_fn.

    Selected examples:
    - I.6.20a:  exp(-theta^2 / 2) / sqrt(2*pi)
    - I.9.18:   G * m1 * m2 / (x2 - x1)^2
    - I.12.1:   q1 * q2 / (4*pi*epsilon*r^2)
    - I.15.10:  m0 * v / sqrt(1 - v^2/c^2)
    - II.6.15a: epsilon * E^2 / 2
"""

"""Measure search space reduction from IsalSR canonicalization.

Generates random expression DAGs, computes canonical strings, and measures:
- Number of unique canonical strings vs. total DAGs generated
- Empirical reduction factor vs. theoretical O(k!)
- Distribution of canonical string lengths

This is the core experiment for the paper's central claim.

Usage: python experiments/scripts/search_space_analysis.py --config experiments/configs/nguyen.yaml
"""

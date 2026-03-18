"""Model-agnostic statistical analysis pipeline.

Implements the full statistical framework from the experimental design doc:
- Paired t-test / Wilcoxon signed-rank per problem
- Holm-Bonferroni correction across problems
- Friedman test + Nemenyi post-hoc across methods
- Cohen's d effect size with bootstrap CI
- McNemar's test for solution rate
"""

"""IsalSR experimental model framework.

Three-layer architecture:
1. ModelRunner — runs SR methods with standard interface
2. ResultTranslator — converts model-specific output to unified schema
3. DataAnalyzer — model-agnostic statistical analysis pipeline
"""

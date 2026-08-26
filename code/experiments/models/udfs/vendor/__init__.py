"""Vendored DAG_search (UDFS) package.

Source: https://github.com/kahlmeyer94/DAG_search
License: MIT
Reference: Kahlmeyer et al. (2024). Scaling Up Unbiased Search-based
    Symbolic Regression. IJCAI 2024. DOI: 10.24963/ijcai.2024/471

The DAG_search package is vendored unmodified. All wrapping is done
via subclassing and composition in the parent udfs/ package.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add vendor directory to sys.path so DAG_search can be imported.
_vendor_dir = str(Path(__file__).parent)
if _vendor_dir not in sys.path:
    sys.path.insert(0, _vendor_dir)

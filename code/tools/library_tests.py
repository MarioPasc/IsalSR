"""Print the test files that exercise only the installable library.

A test module belongs to the library suite when it imports nothing from the
research tree (``experiments/``, ``benchmarks/``), which the wheel does not
ship. CI and the release checks run this subset against an installed package.
"""

import re
from pathlib import Path

RESEARCH_IMPORT = re.compile(
    r"^\s*(from|import)\s+(experiments|benchmarks)\b|['\"](experiments|benchmarks)[./]", re.M
)


def main() -> None:
    tests = Path(__file__).resolve().parent.parent / "tests"
    for path in sorted(tests.rglob("test_*.py")):
        if not RESEARCH_IMPORT.search(path.read_text(encoding="utf-8")):
            print(path.relative_to(tests.parent))


if __name__ == "__main__":
    main()

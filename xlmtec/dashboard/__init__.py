"""
xlmtec.dashboard
~~~~~~~~~~~~~~~~~
Evaluation dashboard — compare training runs side by side.

Usage:
    from xlmtec.dashboard import RunComparator, RunReader
    result = RunComparator().compare([Path("output/run1"), Path("output/run2")])
    print(result.winner.name)
"""

from xlmtec.dashboard.reader import RunInfo, RunMetrics, RunReader
from xlmtec.dashboard.comparator import ComparisonResult, RunComparator

__all__ = [
    "RunReader", "RunInfo", "RunMetrics",
    "RunComparator", "ComparisonResult",
]
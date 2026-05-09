"""
Result dataclass for boolean column profiling.

Populated by BooleanProfiler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from .config import BooleanStats


@dataclass
class BooleanProfileResult:
    """
    Boolean profile for all eligible columns.

    Attributes
    ----------
    columns : dict[str, BooleanStats]
        Per-column boolean profiles, keyed by column name.
    analysed_columns : list[str]
        Columns that were actually profiled (after schema intersection
        and eligibility check).
    """

    columns: dict[str, BooleanStats] = field(default_factory=dict)
    analysed_columns: list[str] = field(default_factory=list)

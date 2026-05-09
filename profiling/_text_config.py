"""
Result dataclass for free-text column profiling.

Populated by TextProfiler.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from .config import TextStats


@dataclass
class TextProfileResult:
    """
    Text profile for all eligible columns.

    Attributes
    ----------
    columns : dict[str, TextStats]
        Per-column text profiles, keyed by column name.
    analysed_columns : list[str]
        Columns that were actually profiled (after schema intersection
        and eligibility check).
    """

    columns: dict[str, TextStats] = field(default_factory=dict)
    analysed_columns: list[str] = field(default_factory=list)

"""
Result dataclass for free-text column profiling.

Populated by TextProfiler.
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class TextStats:
    avg_token_count: float = 0.0
    median_token_count: float = 0.0
    vocabulary_size: int = 0
    char_length_min: int = 0
    char_length_max: int = 0
    char_length_mean: float = 0.0
    char_length_median: float = 0.0
    empty_ratio: float = 0.0
    whitespace_ratio: float = 0.0


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

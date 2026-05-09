"""
Result dataclasses for datetime column profiling.

Populated by DatetimeProfiler, which is opt-in via
ProfileConfig.datetime_columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from .config import DatetimeStats


@dataclass
class DatetimeProfileResult:
    """
    Datetime profile for all opted-in columns.

    Attributes
    ----------
    columns : dict[str, ColumnDatetimeProfile]
        Per-column profiles, keyed by column name.
    analysed_columns : list[str]
        Columns that were actually profiled (after schema intersection).
    """

    columns: dict[str, DatetimeStats] = field(default_factory=dict)
    analysed_columns: list[str] = field(default_factory=list)

    def __str__(self) -> str:  # pragma: no cover
        lines = ["=== Datetime Profile ==="]
        for profile in self.columns.values():
            lines.append(str(profile))
        return "\n".join(lines)

"""
Result dataclasses for categorical column profiling.

These complement TabularProfileResult and are populated by
CategoricalProfiler, which is opt-in via ProfileConfig.categorical_columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from .config import CategoricalStats

# ---------------------------------------------------------------------------
# Top-level result
# ---------------------------------------------------------------------------


@dataclass
class CategoricalProfileResult:
    """
    Categorical profile for all opted-in columns.

    Attributes
    ----------
    columns : dict[str, CategoricalColumnProfile]
        Per-column profiles, keyed by column name.
    analysed_columns : list[str]
        Columns that were actually profiled (after schema intersection).
    """

    columns: dict[str, CategoricalStats] = field(default_factory=dict)
    analysed_columns: list[str] = field(default_factory=list)

    def __str__(self) -> str:  # pragma: no cover
        lines = ["=== Categorical Profile ==="]
        for profile in self.columns.values():
            lines.append(str(profile))
        return "\n".join(lines)

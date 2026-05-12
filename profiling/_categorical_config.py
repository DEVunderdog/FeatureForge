"""
Result dataclasses for categorical column profiling.

These complement TabularProfileResult and are populated by
CategoricalProfiler, which is opt-in via ProfileConfig.categorical_columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

# ---------------------------------------------------------------------------
# Categorical stats dataclasses (canonical home — config.py re-exports these)
# ---------------------------------------------------------------------------


class CategoricalFlag(StrEnum):
    MixedType = "mixed_type"
    FreeText = "free_text"
    NearConstant = "near_constant"


@dataclass
class TopValueEntry:
    value: object
    count: int
    percentage: float


@dataclass
class RareCategoryStats:
    threshold_pct: float
    rare_category_count: int = 0
    total_rare_rows: int = 0
    rare_row_percentage: float = 0.0


@dataclass
class ImbalanceMetrics:
    class_ratio: float = 0.0
    shannon_entropy: float = 0.0
    gini_impurity: float = 0.0


@dataclass
class CategoricalStats:
    cardinality: int = 0
    unique_ratio: float = 0.0
    mode_frequency: float = 0.0
    top_values: list[TopValueEntry] = field(default_factory=list)
    rare_categories: RareCategoryStats = field(
        default_factory=lambda: RareCategoryStats(threshold_pct=0.01),
    )
    imbalance: ImbalanceMetrics = field(default_factory=ImbalanceMetrics)
    flags: list[CategoricalFlag] = field(default_factory=list)


CategoricalColumnProfile = CategoricalStats


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

"""
Result dataclasses for categorical column profiling.

These complement TabularProfileResult and are populated by
CategoricalProfiler, which is opt-in via ProfileConfig.categorical_columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class CategoricalKind(StrEnum):
    Nominal = "nominal"
    Ordinal = "ordinal"


class CategoricalFlag(StrEnum):
    MixedType = "mixed_type"
    FreeText = "free_text"
    NearConstant = "near_constant"


# ---------------------------------------------------------------------------
# Per-column result
# ---------------------------------------------------------------------------


@dataclass
class TopValueEntry:
    """One entry in the top-N value counts list."""

    value: object
    count: int
    percentage: float  # fraction of total rows (0–1)


@dataclass
class RareCategoryStats:
    """
    Summary of low-frequency categories.

    Threshold used: categories whose row count < 1 % of total rows.
    """

    threshold_pct: float  # always 0.01
    rare_category_count: int = 0  # distinct categories below threshold
    total_rare_rows: int = 0  # rows belonging to rare categories
    rare_row_percentage: float = 0.0  # total_rare_rows / row_count


@dataclass
class ImbalanceMetrics:
    """
    Three complementary measures of class-distribution skew.

    class_ratio  : max_freq / min_freq  (>10 is a red flag)
    shannon_entropy : -Σ p·log₂(p)     (0 = fully concentrated)
    gini_impurity   : 1 - Σ p²         (0 = perfectly pure)
    """

    class_ratio: float = 0.0
    shannon_entropy: float = 0.0
    gini_impurity: float = 0.0


@dataclass
class CategoricalColumnProfile:
    """
    Full categorical profile for a single column.

    Attributes
    ----------
    column : str
        Column name.
    total_rows : int
        Total rows in the DataFrame (denominator for ratios).
    cardinality : int
        Number of distinct non-null values.
    unique_ratio : float
        cardinality / total_rows.
    kind : Optional[CategoricalKind]
        Ordinal or Nominal — None when detection is inconclusive.
    top_values : list[TopValueEntry]
        Up to 5 most-frequent values with counts and percentages.
    rare_categories : RareCategoryStats
        Statistics about categories below the 1 % frequency threshold.
    imbalance : ImbalanceMetrics
        Class ratio, Shannon entropy, and Gini impurity.
    flags : list[CategoricalFlag]
        Zero or more non-exclusive behavioural flags.
    """

    column: str
    total_rows: int

    # Cardinality
    cardinality: int = 0
    unique_ratio: float = 0.0

    mode_frequency: float = 0.0

    # Semantic kind
    kind: Optional[CategoricalKind] = None
    # Value distribution
    top_values: list[TopValueEntry] = field(default_factory=list)
    rare_categories: RareCategoryStats = field(
        default_factory=lambda: RareCategoryStats(threshold_pct=0.01)
    )
    imbalance: ImbalanceMetrics = field(default_factory=ImbalanceMetrics)

    # Flags
    flags: list[CategoricalFlag] = field(default_factory=list)

    def has_flag(self, flag: CategoricalFlag) -> bool:
        return flag in self.flags

    def __str__(self) -> str:  # pragma: no cover
        lines = [
            f"  Column : {self.column}",
            f"    Kind               : {self.kind or 'unknown'}",
            f"    Cardinality        : {self.cardinality:,}  (unique ratio {self.unique_ratio:.4%})",
            f"    Null count         : {self.null_count:,}",
            f"    Whitespace count   : {self.whitespace_count:,}",
            f"    Effective missing  : {self.effective_missing_count:,}",
        ]
        if self.top_values:
            lines.append("    Top values:")
            for tv in self.top_values:
                lines.append(
                    f"      {str(tv.value)!r:30s}  {tv.count:>8,}  ({tv.percentage:.2%})"
                )
        rc = self.rare_categories
        lines.append(
            f"    Rare categories    : {rc.rare_category_count} categories, "
            f"{rc.total_rare_rows:,} rows ({rc.rare_row_percentage:.2%})  "
            f"[threshold={rc.threshold_pct:.0%}]"
        )
        im = self.imbalance
        lines += [
            f"    Class ratio        : {im.class_ratio:.2f}",
            f"    Shannon entropy    : {im.shannon_entropy:.4f}",
            f"    Gini impurity      : {im.gini_impurity:.4f}",
        ]
        if self.flags:
            lines.append(f"    Flags              : {', '.join(self.flags)}")
        return "\n".join(lines)


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

    columns: dict[str, CategoricalColumnProfile] = field(default_factory=dict)
    analysed_columns: list[str] = field(default_factory=list)

    def __str__(self) -> str:  # pragma: no cover
        lines = ["=== Categorical Profile ==="]
        for profile in self.columns.values():
            lines.append(str(profile))
        return "\n".join(lines)

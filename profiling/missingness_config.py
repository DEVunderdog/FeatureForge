"""
Result dataclasses for missingness profiling.

Populated by MissingnessProfiler, which is always run as part of
StructuralProfiler (non-optional Phase 1 component).
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


class MissingSeverity(StrEnum):
    Minor = "minor"  # < 1%   missing
    Moderate = "moderate"  # 1–5%   missing
    High = "high"  # 5–20%  missing
    Severe = "severe"  # > 20%  missing


class MissingnessFlag(StrEnum):
    FullyNull = "fully_null"  # missing ratio == 1.0 → must drop
    MARSuspect = "mar_suspect"  # correlated missingness with ≥1 other col
    DropCandidate = "drop_candidate"  # >50% of rows missing across the column


# ---------------------------------------------------------------------------
# Per-column result
# ---------------------------------------------------------------------------


@dataclass
class ColumnMissingnessProfile:
    """
    Full missingness profile for a single column.

    Attributes
    ----------
    column : str
        Column name.
    total_rows : int
        Total rows in the DataFrame.
    standard_null_count : int
        Polars-level nulls (None / NaN for floats).
    effective_null_count : int
        Standard nulls + whitespace-only strings + sentinel strings
        ("NA", "NAN", "NULL", "NONE", "?") — i.e. the count used for
        imputation decisions.
    standard_null_ratio : float
        standard_null_count / total_rows.
    effective_null_ratio : float
        effective_null_count / total_rows.
    severity : MissingSeverity
        Derived from effective_null_ratio.
    flags : list[MissingnessFlag]
        Zero or more non-exclusive behavioural flags.
    correlated_with : list[str]
        Columns whose missingness indicator correlates > 0.6 with this
        column's indicator (populated after the correlation matrix pass).
    """

    column: str
    total_rows: int

    standard_null_count: int = 0
    effective_null_count: int = 0
    standard_null_ratio: float = 0.0
    effective_null_ratio: float = 0.0

    severity: Optional[MissingSeverity] = None

    flags: list[MissingnessFlag] = field(default_factory=list)
    correlated_with: list[str] = field(default_factory=list)

    def has_flag(self, flag: MissingnessFlag) -> bool:
        return flag in self.flags

    def __str__(self) -> str:  # pragma: no cover
        lines = [
            f"  Column : {self.column}",
            f"    Standard nulls     : {self.standard_null_count:,}"
            f"  ({self.standard_null_ratio:.2%})",
            f"    Effective nulls    : {self.effective_null_count:,}"
            f"  ({self.effective_null_ratio:.2%})",
            f"    Severity           : {self.severity or 'N/A'}",
        ]
        if self.correlated_with:
            lines.append(f"    MAR correlates with: {', '.join(self.correlated_with)}")
        if self.flags:
            lines.append(f"    Flags              : {', '.join(self.flags)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Row-wise distribution
# ---------------------------------------------------------------------------


@dataclass
class RowMissingnessDistribution:
    """
    Summary of per-row missing-value counts across the scoped columns.

    Attributes
    ----------
    pct_zero_missing : float
        Fraction of rows with 0 missing values.
    pct_one_to_two : float
        Fraction of rows with 1–2 missing values.
    pct_three_to_five : float
        Fraction of rows with 3–5 missing values.
    pct_over_five : float
        Fraction of rows with > 5 missing values.
    pct_over_half_missing : float
        Fraction of rows where > 50% of the scoped columns are missing.
        These rows are candidates for unconditional dropping before imputation.
    drop_candidate_row_count : int
        Absolute count corresponding to pct_over_half_missing.
    """

    pct_zero_missing: float = 0.0
    pct_one_to_two: float = 0.0
    pct_three_to_five: float = 0.0
    pct_over_five: float = 0.0
    pct_over_half_missing: float = 0.0
    drop_candidate_row_count: int = 0

    def __str__(self) -> str:  # pragma: no cover
        return (
            "  Row-wise missingness distribution:\n"
            f"    0 missing          : {self.pct_zero_missing:.2%}\n"
            f"    1–2 missing        : {self.pct_one_to_two:.2%}\n"
            f"    3–5 missing        : {self.pct_three_to_five:.2%}\n"
            f"    >5  missing        : {self.pct_over_five:.2%}\n"
            f"    >50% cols missing  : {self.pct_over_half_missing:.2%}"
            f"  ({self.drop_candidate_row_count:,} rows)"
        )


# ---------------------------------------------------------------------------
# Top-level result
# ---------------------------------------------------------------------------


@dataclass
class MissingnessProfileResult:
    """
    Missingness profile for all analysed columns.

    Attributes
    ----------
    columns : dict[str, ColumnMissingnessProfile]
        Per-column profiles, keyed by column name.
    analysed_columns : list[str]
        Columns that were actually profiled.
    fully_null_columns : list[str]
        Columns where effective_null_ratio == 1.0.  Must be dropped.
    correlation_matrix : dict[str, dict[str, float]]
        Pairwise Pearson correlations between binary missingness indicators.
        Only populated when ≥ 2 columns have at least one missing value.
        Stored as a nested dict: matrix[col_a][col_b] = correlation.
    row_distribution : RowMissingnessDistribution
        Aggregate row-wise missingness summary.
    """

    columns: dict[str, ColumnMissingnessProfile] = field(default_factory=dict)
    analysed_columns: list[str] = field(default_factory=list)
    fully_null_columns: list[str] = field(default_factory=list)
    correlation_matrix: dict[str, dict[str, float]] = field(default_factory=dict)
    row_distribution: RowMissingnessDistribution = field(
        default_factory=RowMissingnessDistribution
    )

    def __str__(self) -> str:  # pragma: no cover
        lines = ["=== Missingness Profile ==="]
        for profile in self.columns.values():
            lines.append(str(profile))
        if self.fully_null_columns:
            lines.append(
                f"\n  Fully-null columns (must drop): "
                f"{', '.join(self.fully_null_columns)}"
            )
        lines.append(str(self.row_distribution))
        return "\n".join(lines)

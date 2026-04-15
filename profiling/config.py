"""
Configuration and result dataclasses for the profiling phase.

ProfileConfig lets callers scope which columns participate in each
sub-analysis. None means "use all columns"; a list restricts the
analysis to just those names.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional


@dataclass
class ProfileConfig:
    """
    Controls which columns are included in each structural analysis.

    Attributes
    ----------
    columns : Optional[list[str]]
        Columns to analyse for every metric. None → all columns.
        Acts as the default for any metric that does not declare its own
        column list.
    duplicate_columns : Optional[list[str]]
        Columns used to determine duplicate rows.
        None → falls back to `columns` (which itself may mean all).
    sparsity_columns : Optional[list[str]]
        Columns included in the overall sparsity calculation.
        None → falls back to `columns`.
    type_detection_columns : Optional[list[str]]
        Columns on which data-type detection is run.
        None → type detection is skipped entirely (opt-in).
    memory_threshold_mb : float
        If total DataFrame memory exceeds this value (in MB) the profiler
        switches to chunked analysis for the metrics that support it.
    chunk_size : int
        Number of rows per chunk when chunked processing is active.
    """

    columns: Optional[list[str]] = None
    duplicate_columns: Optional[list[str]] = None
    sparsity_columns: Optional[list[str]] = None
    type_detection_columns: Optional[list[str]] = None  # opt-in per-column
    memory_threshold_mb: float = 500.0
    chunk_size: int = 100_000

    def resolve_duplicate_columns(self) -> Optional[list[str]]:
        """Return the effective column list for duplicate detection."""
        return (
            self.duplicate_columns
            if self.duplicate_columns is not None
            else self.columns
        )

    def resolve_sparsity_columns(self) -> Optional[list[str]]:
        """Return the effective column list for sparsity calculation."""
        return (
            self.sparsity_columns if self.sparsity_columns is not None else self.columns
        )


# ---------------------------------------------------------------------------
# Result containers
# ---------------------------------------------------------------------------


@dataclass
class MemoryBreakdown:
    """Per-column memory usage when the dataset exceeds the threshold."""

    column_bytes: dict[str, int] = field(default_factory=dict)

    @property
    def sorted_by_usage(self) -> list[tuple[str, int]]:
        return sorted(self.column_bytes.items(), key=lambda x: x[1], reverse=True)

    def top_consumers(self, n: int = 10) -> list[tuple[str, int]]:
        return self.sorted_by_usage[:n]


# ---------------------------------------------------------------------------
# Data-type detection results
# ---------------------------------------------------------------------------


class NumericKind(StrEnum):
    Continuous = "continuous"
    Discrete = "discrete"


class TypeFlag(StrEnum):
    """Non-exclusive flags that can be attached to a column."""

    BooleanCandidate = "boolean_candidate"
    EncodedCategory = "encoded_category"     # low-cardinality int
    IdentifierColumn = "identifier_column"   # near-100 % unique
    SequentialIndex = "sequential_index"     # range(0,n) or range(1,n+1)
    NumericCoerced = "numeric_coerced"       # was object, successfully coerced
    DatetimeCoerced = "datetime_coerced"     # was object, successfully coerced


@dataclass
class ColumnTypeInfo:
    """
    Type-detection result for a single column.

    Attributes
    ----------
    column : str
        Column name.
    original_dtype : str
        Polars dtype string before any coercion.
    inferred_dtype : str
        Dtype after attempted coercion (same as original if no coercion).
    numeric_kind : Optional[NumericKind]
        Whether the column is continuous or discrete (numeric columns only).
    flags : list[TypeFlag]
        Zero or more non-exclusive behavioural flags.
    """

    column: str
    original_dtype: str
    inferred_dtype: str
    numeric_kind: Optional[NumericKind] = None
    flags: list[TypeFlag] = field(default_factory=list)

    def has_flag(self, flag: TypeFlag) -> bool:
        return flag in self.flags


@dataclass
class TabularProfileResult:
    """
    Structural profile of a tabular dataset.

    All metrics respect the column scoping declared in ProfileConfig.
    """

    # Shape
    row_count: int = 0
    column_count: int = 0  # total columns in the DataFrame

    # Memory
    total_memory_bytes: int = 0
    memory_exceeded_threshold: bool = False
    memory_breakdown: Optional[MemoryBreakdown] = None  # populated only when exceeded

    # Duplicates  (scoped to duplicate_columns)
    duplicate_row_count: int = 0
    duplicate_ratio: float = 0.0  # duplicate_row_count / row_count

    # Sparsity  (scoped to sparsity_columns)
    overall_sparsity: float = 0.0  # fraction of cells that are NaN/None

    # Type detection  (scoped to type_detection_columns; empty if not opted-in)
    column_type_info: dict[str, ColumnTypeInfo] = field(default_factory=dict)

    # Metadata
    analysed_columns: list[str] = field(default_factory=list)
    duplicate_scope_columns: list[str] = field(default_factory=list)
    sparsity_scope_columns: list[str] = field(default_factory=list)
    was_chunked: bool = False

    @property
    def total_memory_mb(self) -> float:
        return self.total_memory_bytes / (1024**2)

    def __str__(self) -> str:  # pragma: no cover
        lines = [
            "=== Tabular Structural Profile ===",
            f"  Shape              : {self.row_count:,} rows × {self.column_count:,} cols",
            f"  Memory             : {self.total_memory_mb:.2f} MB"
            + (" [EXCEEDED THRESHOLD]" if self.memory_exceeded_threshold else ""),
            f"  Duplicate rows     : {self.duplicate_row_count:,} ({self.duplicate_ratio:.2%})",
            f"  Overall sparsity   : {self.overall_sparsity:.4%}",
            f"  Chunked processing : {self.was_chunked}",
        ]
        if self.memory_breakdown:
            lines.append("  Top memory consumers:")
            for col, b in self.memory_breakdown.top_consumers():
                lines.append(f"    {col:30s}  {b / (1024**2):.3f} MB")
        if self.column_type_info:
            lines.append("  Type detection:")
            for col, info in self.column_type_info.items():
                flags_str = ", ".join(info.flags) if info.flags else "—"
                kind_str = f"  [{info.numeric_kind}]" if info.numeric_kind else ""
                lines.append(
                    f"    {col:30s}  {info.original_dtype} → {info.inferred_dtype}"
                    f"{kind_str}  flags=[{flags_str}]"
                )
        return "\n".join(lines)
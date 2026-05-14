"""
Configuration and result dataclasses for the profiling phase — Phase 1 redesign.

ProfileConfig controls the structural profiler's behaviour.
Stats dataclasses hold per-column and dataset-level profiling results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional, Union

from ._missingness_config import (
    ColumnMissingnessProfile,
)
from ._correlation_config import (
    CorrelationProfileResult,
)
from ._categorical_config import (
    CategoricalStats,
)
from ._numeric_config import (
    NumericStats,
)
from ._datetime_config import (
    DatetimeStats,
)
from ._boolean_config import BooleanStats
from ._text_config import TextStats
from ._target_config import TargetProfileResult

# ---------------------------------------------------------------------------
# Core enums
# ---------------------------------------------------------------------------


class SemanticType(StrEnum):
    Numeric = "numeric"
    Categorical = "categorical"
    Datetime = "datetime"
    Boolean = "boolean"
    Text = "text"
    Identifier = "identifier"


class Modality(StrEnum):
    Tabular = "tabular"
    # Placeholder slots for future modalities — no implementation yet.
    # Image = "image"
    # TimeSeries = "time_series"


# ---------------------------------------------------------------------------
# Type-detection enums — kept for TypeDetector compatibility
# ---------------------------------------------------------------------------


class NumericKind(StrEnum):
    Continuous = "continuous"
    Discrete = "discrete"


class TypeFlag(StrEnum):
    NumericCoerced = "numeric_coerced"
    DatetimeCoerced = "datetime_coerced"
    BooleanCandidate = "boolean_candidate"
    EncodedCategory = "encoded_category"
    IdentifierColumn = "identifier_column"
    SequentialIndex = "sequential_index"
    FloatSequentialIndex = "float_sequential_index"
    FreeTextCandidate = "free_text_candidate"


# ---------------------------------------------------------------------------
# Column and dataset result containers
# ---------------------------------------------------------------------------

AnyStats = Union[NumericStats, CategoricalStats, DatetimeStats, BooleanStats, TextStats]


@dataclass
class ColumnProfile:
    name: str = ""
    semantic_type: Optional[SemanticType] = None
    type_flags: list[TypeFlag] = field(default_factory=list)
    original_dtype: str = ""
    inferred_dtype: str = ""
    missingness: Optional[ColumnMissingnessProfile] = None
    is_target: bool = False
    stats: Optional[AnyStats] = None


@dataclass
class RowMissingnessDistribution:
    """
    Dataset-level summary of per-row missing-value counts.
    Computed by StructuralProfiler over the full active column set.
    """

    pct_zero_missing: float = 0.0
    pct_one_to_two: float = 0.0
    pct_three_to_five: float = 0.0
    pct_over_five: float = 0.0
    pct_over_half_missing: float = 0.0
    drop_candidate_row_count: int = 0


@dataclass
class MemoryBreakdown:
    column_bytes: dict[str, int] = field(default_factory=dict)

    @property
    def sorted_by_usage(self) -> list[tuple[str, int]]:
        return sorted(self.column_bytes.items(), key=lambda x: x[1], reverse=True)

    def top_consumers(self, n: int = 10) -> list[tuple[str, int]]:
        return self.sorted_by_usage[:n]


@dataclass
class DatasetStats:
    modality: Modality = Modality.Tabular
    row_count: int = 0
    column_count: int = 0
    memory_bytes: int = 0
    memory_breakdown: Optional[MemoryBreakdown] = None
    duplicate_count: int = 0
    duplicate_ratio: float = 0.0
    overall_sparsity: float = 0.0
    was_chunked: bool = False
    missingness_matrix: Optional[dict[str, dict[str, float]]] = None
    row_distribution: RowMissingnessDistribution = field(
        default_factory=RowMissingnessDistribution
    )

    feature_correlation: Optional[CorrelationProfileResult] = None

    target_correlations: dict[str, CorrelationProfileResult] = field(
        default_factory=dict,
    )


@dataclass
class StructuralProfileResult:
    columns: dict[str, ColumnProfile] = field(default_factory=dict)
    dataset: DatasetStats = field(default_factory=DatasetStats)
    targets: dict[str, TargetProfileResult] = field(default_factory=dict)


# ---------------------------------------------------------------------------
# ProfileConfig — clean break from per-profiler column lists
# ---------------------------------------------------------------------------


@dataclass
class ProfileConfig:
    """
    Controls the structural profiler's behaviour.

    Parameters
    ----------
    modality : Modality
        Data modality. Currently only Tabular is implemented.
    target_column : Optional[str]
        Name of the label/target column, if any.
    column_overrides : dict[str, SemanticType]
        Explicit semantic type assignments that override auto-detection.
    exclude_columns : list[str]
        Columns to skip entirely during profiling.
    compute_correlation : bool
        Whether to compute the feature-feature correlation matrix.
    correlation_target_column : Optional[str]
        Column used for feature-target correlation metrics.
    memory_threshold_mb : float
        Memory (MB) above which chunked processing activates.
    chunk_size : int
        Rows per chunk when chunked processing is active.
    """

    modality: Modality = Modality.Tabular
    target_columns: list[str] = field(default_factory=list)
    column_overrides: dict[str, SemanticType] = field(default_factory=dict)
    exclude_columns: list[str] = field(default_factory=list)
    compute_correlation: bool = False
    correlation_target_column: Optional[str] = None
    memory_threshold_mb: float = 500.0
    chunk_size: int = 100_000

    def to_dict(self) -> dict:
        return {
            "modality": str(self.modality),
            "target_columns": list(self.target_columns),
            "column_overrides": {k: str(v) for k, v in self.column_overrides.items()},
            "exclude_columns": list(self.exclude_columns),
            "compute_correlation": self.compute_correlation,
            "correlation_target_column": self.correlation_target_column,
            "memory_threshold_mb": self.memory_threshold_mb,
            "chunk_size": self.chunk_size,
        }

    @classmethod
    def from_dict(cls, data: dict) -> ProfileConfig:
        return cls(
            modality=Modality(data.get("modality", Modality.Tabular)),
            target_column=data.get("target_column"),
            column_overrides={
                k: SemanticType(v) for k, v in data.get("column_overrides", {}).items()
            },
            exclude_columns=list(data.get("exclude_columns", [])),
            compute_correlation=bool(data.get("compute_correlation", False)),
            correlation_target_column=data.get("correlation_target_column"),
            memory_threshold_mb=float(data.get("memory_threshold_mb", 500.0)),
            chunk_size=int(data.get("chunk_size", 100_000)),
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict())

    @classmethod
    def from_json(cls, json_str: str) -> ProfileConfig:
        return cls.from_dict(json.loads(json_str))


@dataclass
class ColumnTypeInfo:
    column: str
    original_dtype: str
    inferred_dtype: str
    numeric_kind: Optional[NumericKind] = None
    flags: list[TypeFlag] = field(default_factory=list)
    semantic_type: Optional[SemanticType] = None

    def has_flag(self, flag: TypeFlag) -> bool:
        return flag in self.flags

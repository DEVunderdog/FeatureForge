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


# ---------------------------------------------------------------------------
# Stats dataclasses
# ---------------------------------------------------------------------------


@dataclass
class MissingnessStats:
    null_count: int = 0
    effective_null_count: int = 0
    effective_null_ratio: float = 0.0
    severity: Optional[str] = None  # "minor" | "moderate" | "high" | "severe"


@dataclass
class PercentileSnapshot:
    p1: Optional[float] = None
    p5: Optional[float] = None
    p25: Optional[float] = None
    p50: Optional[float] = None
    p75: Optional[float] = None
    p95: Optional[float] = None
    p99: Optional[float] = None

    @property
    def iqr(self) -> Optional[float]:
        if self.p25 is not None and self.p75 is not None:
            return self.p75 - self.p25
        return None


@dataclass
class NumericStats:
    mean: Optional[float] = None
    median: Optional[float] = None
    std: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None
    percentiles: PercentileSnapshot = field(default_factory=PercentileSnapshot)
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    skewness_severity: Optional[str] = None  # "normal" | "moderate" | "high" | "severe"
    histogram: list[dict] = field(default_factory=list)
    top_values: list[dict] = field(default_factory=list)
    scale_anomaly: bool = False
    near_constant: bool = False


@dataclass
class CategoricalStats:
    cardinality: int = 0
    unique_ratio: float = 0.0
    is_ordinal: Optional[bool] = None
    top_values: list[dict] = field(default_factory=list)
    rare_category_count: int = 0
    rare_category_ratio: float = 0.0
    whitespace_count: int = 0
    has_mixed_types: bool = False
    entropy: Optional[float] = None
    gini: Optional[float] = None
    class_ratio: Optional[float] = None


@dataclass
class DatetimeStats:
    min_date: Optional[str] = None
    max_date: Optional[str] = None
    date_range_days: Optional[float] = None
    null_rate: float = 0.0
    future_date_count: int = 0
    has_recent_sparsity: bool = False
    inferred_granularity: Optional[str] = None
    temporal_signal: Optional[str] = None


@dataclass
class BooleanStats:
    true_count: int = 0
    false_count: int = 0
    true_ratio: float = 0.0
    false_ratio: float = 0.0
    mode: Optional[bool] = None


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
    is_target: bool = False
    missingness: MissingnessStats = field(default_factory=MissingnessStats)
    stats: Optional[AnyStats] = None


@dataclass
class DatasetStats:
    modality: Modality = Modality.Tabular
    row_count: int = 0
    column_count: int = 0
    memory_bytes: int = 0
    duplicate_count: int = 0
    duplicate_ratio: float = 0.0
    overall_sparsity: float = 0.0
    missingness_matrix: Optional[dict[str, dict[str, float]]] = None
    correlation: Optional[dict[str, dict[str, float]]] = None


@dataclass
class StructuralProfileResult:
    columns: dict[str, ColumnProfile] = field(default_factory=dict)
    dataset: DatasetStats = field(default_factory=DatasetStats)


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
    target_column: Optional[str] = None
    column_overrides: dict[str, SemanticType] = field(default_factory=dict)
    exclude_columns: list[str] = field(default_factory=list)
    compute_correlation: bool = False
    correlation_target_column: Optional[str] = None
    memory_threshold_mb: float = 500.0
    chunk_size: int = 100_000

    def to_dict(self) -> dict:
        return {
            "modality": str(self.modality),
            "target_column": self.target_column,
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
                k: SemanticType(v)
                for k, v in data.get("column_overrides", {}).items()
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

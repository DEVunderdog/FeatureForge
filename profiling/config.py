"""
Configuration and result dataclasses for the profiling phase — Phase 1 redesign.

ProfileConfig controls the structural profiler's behaviour.
Stats dataclasses hold per-column and dataset-level profiling results.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional, Union, List

from ._missingness_config import (
    ColumnMissingnessProfile,
)
from ._correlation_config import (
    CorrelationProfileResult,
)
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
# Stats dataclasses
# ---------------------------------------------------------------------------


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


class SkewSeverity(StrEnum):
    Normal = "normal"  # |skew| <= 0.5
    Moderate = "moderate"  # 0.5 < |skew| <= 1.0
    High = "high"  # 1.0 < |skew| <= 2.0
    Severe = "severe"  # |skew| > 2.0


class KurtosisTag(StrEnum):
    Platykurtic = "platykurtic"  # excess kurtosis < -1  (thin tails)
    Mesokurtic = "mesokurtic"  # -1 <= excess <= 3     (near-normal)
    Leptokurtic = "leptokurtic"  # excess kurtosis > 3   (heavy tails)


class NumericFlag(StrEnum):
    ScaleAnomaly = "scale_anomaly"  # values span 3+ orders of magnitude
    NearConstant = "near_constant"


@dataclass
class NumericTopValueEntry:
    value: float
    count: int
    percentage: float


@dataclass
class HistogramBin:
    lower_bound: float
    upper_bound: float
    count: int
    percentage: float


@dataclass
class NumericStats:
    mean: Optional[float] = None
    median: Optional[float] = None
    mean_median_ratio: Optional[float] = None

    mode: Optional[float] = None
    mode_frequency: float = 0.0
    top_values: list[NumericTopValueEntry] = field(default_factory=list)
    histogram: list[HistogramBin] = field(default_factory=list)

    std: Optional[float] = None
    variance: Optional[float] = None
    min: Optional[float] = None
    max: Optional[float] = None

    percentiles: PercentileSnapshot = field(default_factory=PercentileSnapshot)
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None
    skewness_severity: Optional[SkewSeverity] = None
    kurtosis_tag: Optional[KurtosisTag] = None

    flags: List[NumericFlag] = field(default_factory=list)

    @property
    def iqr(self) -> Optional[float]:
        return self.percentiles.iqr

    def has_flag(self, flag: NumericFlag) -> bool:
        return flag in self.flags


class CategoricalFlag(StrEnum):
    MixedType = "mixed_type"
    FreeText = "free_text"
    NearConstant = "near_constant"


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


class InferredGranularity(StrEnum):
    Yearly = "yearly"  # median gap ≈ 365 days
    Monthly = "monthly"  # median gap ≈ 30 days
    Weekly = "weekly"  # median gap ≈ 7 days
    Daily = "daily"  # median gap ≈ 1 day
    Hourly = "hourly"  # median gap ≈ 1 hour
    Minutely = "minutely"  # median gap ≈ 1 minute
    Secondly = "secondly"  # median gap < 1 minute
    Irregular = "irregular"  # no dominant periodicity


class DatetimeFlag(StrEnum):
    FutureDates = "future_dates"  # values > current date found
    HighGapVariance = "high_gap_variance"  # coefficient of variation > 1.0
    MnarSuspected = "mnar_suspected"  # non-trivial null rate (>5 %)
    RecentDateMissing = "recent_date_missing"  # last 10 % of expected range is sparse


# ---------------------------------------------------------------------------
# Temporal signal audit
# ---------------------------------------------------------------------------


@dataclass
class TemporalSignals:
    """
    Flags which time-derived features are extractable from this column.

    These are signals to guide Phase 3/5 feature engineering — no features
    are created here.

    Attributes
    ----------
    has_year       : bool  – year varies across rows
    has_month      : bool  – month varies across rows
    has_day        : bool  – day-of-month varies across rows
    has_day_of_week: bool  – day-of-week varies (only meaningful for daily+)
    has_hour       : bool  – non-zero hour values present
    has_is_weekend : bool  – weekend signal (True when has_day_of_week is True)
    has_is_month_end: bool – any values fall on the last day of their month
    """

    has_year: bool = False
    has_month: bool = False
    has_day: bool = False
    has_day_of_week: bool = False
    has_hour: bool = False
    has_is_weekend: bool = False
    has_is_month_end: bool = False

    def extractable_features(self) -> list[str]:
        """Return names of features worth extracting downstream."""
        features = []
        if self.has_year:
            features.append("year")
        if self.has_month:
            features.append("month")
        if self.has_day:
            features.append("day_of_month")
        if self.has_day_of_week:
            features.append("day_of_week")
        if self.has_hour:
            features.append("hour")
        if self.has_is_weekend:
            features.append("is_weekend")
        if self.has_is_month_end:
            features.append("is_month_end")
        return features


@dataclass
class DatetimeStats:
    min_date: Optional[str] = None
    max_date: Optional[str] = None
    date_range_days: Optional[float] = None
    future_date_count: int = 0
    inferred_granularity: Optional[InferredGranularity] = None
    median_gap_seconds: Optional[float] = None
    gap_cv: Optional[float] = None
    signals: TemporalSignals = field(default_factory=TemporalSignals)
    flags: list[DatetimeFlag] = field(default_factory=list)

    def has_flag(self, flag: DatetimeFlag) -> bool:
        return flag in self.flags


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
    missingness: Optional[ColumnMissingnessProfile] = field(
        default_factory=ColumnMissingnessProfile
    )
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
    correlation: CorrelationProfileResult = field(
        default_factory=CorrelationProfileResult,
    )
    row_distribution: RowMissingnessDistribution = field(
        default_factory=RowMissingnessDistribution
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


# ---------------------------------------------------------------------------
# Legacy result types — used by old profilers pending redesign
# ---------------------------------------------------------------------------


@dataclass
class ColumnMissingness:
    standard_nulls: int = 0
    effective_nulls: int = 0
    effective_null_ratio: float = 0.0


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


@dataclass
class TabularProfileResult:
    row_count: int = 0
    column_count: int = 0
    total_memory_bytes: int = 0
    memory_exceeded_threshold: bool = False
    memory_breakdown: Optional[MemoryBreakdown] = None
    duplicate_row_count: int = 0
    duplicate_ratio: float = 0.0
    missingness: dict[str, ColumnMissingness] = field(default_factory=dict)
    overall_effective_sparsity: float = 0.0
    column_type_info: dict[str, ColumnTypeInfo] = field(default_factory=dict)
    analysed_columns: list[str] = field(default_factory=list)
    duplicate_scope_columns: list[str] = field(default_factory=list)
    sparsity_scope_columns: list[str] = field(default_factory=list)
    was_chunked: bool = False

    @property
    def total_memory_mb(self) -> float:
        return self.total_memory_bytes / (1024**2)

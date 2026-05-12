"""
Result dataclasses for numeric distribution profiling.

Populated by NumericProfiler, which is opt-in via
ProfileConfig.numeric_columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional, List


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
    Normal = "normal"
    Moderate = "moderate"
    High = "high"
    Severe = "severe"


class KurtosisTag(StrEnum):
    Platykurtic = "platykurtic"
    Mesokurtic = "mesokurtic"
    Leptokurtic = "leptokurtic"


class NumericFlag(StrEnum):
    ScaleAnomaly = "scale_anomaly"
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


ColumnNumericProfile = NumericStats


@dataclass
class NumericProfileResult:
    """
    Numeric distribution profile for all opted-in columns.

    Attributes
    ----------
    columns : dict[str, ColumnNumericProfile]
        Per-column profiles, keyed by column name.
    analysed_columns : list[str]
        Columns that were actually profiled (after schema intersection).
    """

    columns: dict[str, NumericStats] = field(default_factory=dict)
    analysed_columns: list[str] = field(default_factory=list)

    def __str__(self) -> str:  # pragma: no cover
        lines = ["=== Numeric Distribution Profile ==="]
        for profile in self.columns.values():
            lines.append(str(profile))
        return "\n".join(lines)

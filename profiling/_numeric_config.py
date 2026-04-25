"""
Result dataclasses for numeric distribution profiling.

Populated by NumericProfiler, which is opt-in via
ProfileConfig.numeric_columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------


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
class HistogramBin:
    lower_bound: float
    upper_bound: float
    count: int
    percentage: float


@dataclass
class NumericTopValueEntry:
    value: float
    count: int
    percentage: float

# ---------------------------------------------------------------------------
# Per-column result
# ---------------------------------------------------------------------------


@dataclass
class PercentileProfile:
    """
    Full percentile snapshot replacing a box-plot in text form.

    All values are floats; None when the column has no non-null rows.
    """

    p1: Optional[float] = None
    p5: Optional[float] = None
    p25: Optional[float] = None  # Q1
    p50: Optional[float] = None  # median
    p75: Optional[float] = None  # Q3
    p95: Optional[float] = None
    p99: Optional[float] = None

    @property
    def iqr(self) -> Optional[float]:
        """Interquartile range Q3 – Q1."""
        if self.p25 is not None and self.p75 is not None:
            return self.p75 - self.p25
        return None


@dataclass
class ColumnNumericProfile:
    """
    Full numeric distribution profile for a single column.

    Attributes
    ----------
    column : str
        Column name.
    total_rows : int
        Total rows in the DataFrame.
    non_null_count : int
        Rows with a non-null value.

    Central tendency
    ----------------
    mean : float | None
    median : float | None
    mean_median_ratio : float | None
        ``mean / median`` (or inf when median is 0).  A ratio far from 1.0
        is a primary indicator of skew.

    Spread
    ------
    std : float | None
        Population standard deviation (ddof=1).
    variance : float | None
    iqr : float | None
        Q3 − Q1 from the percentile profile.

    Shape
    -----
    skewness : float | None
    kurtosis : float | None
        Excess (Fisher) kurtosis, so 0 ≡ normal distribution.
    skew_severity : SkewSeverity | None
    kurtosis_tag : KurtosisTag | None

    Range
    -----
    min : float | None
    max : float | None

    Percentiles
    -----------
    percentiles : PercentileProfile

    Flags
    -----
    flags : list[NumericFlag]
    """

    column: str
    total_rows: int

    # Non-null count
    non_null_count: int = 0

    # Central tendency
    mean: Optional[float] = None
    median: Optional[float] = None
    mean_median_ratio: Optional[float] = None

    mode: Optional[float] = None
    mode_frequency: float = 0.0
    top_values: list[NumericTopValueEntry] = field(default_factory=list)
    histogram: list[HistogramBin] = field(default_factory=list)

    # Spread
    std: Optional[float] = None
    variance: Optional[float] = None

    # Shape
    skewness: Optional[float] = None
    kurtosis: Optional[float] = None  # excess kurtosis
    skew_severity: Optional[SkewSeverity] = None
    kurtosis_tag: Optional[KurtosisTag] = None

    # Range
    min: Optional[float] = None
    max: Optional[float] = None

    # Percentiles (includes iqr property)
    percentiles: PercentileProfile = field(default_factory=PercentileProfile)

    # Flags
    flags: list[NumericFlag] = field(default_factory=list)

    # Convenience ─────────────────────────────────────────────────────────

    @property
    def iqr(self) -> Optional[float]:
        return self.percentiles.iqr

    def has_flag(self, flag: NumericFlag) -> bool:
        return flag in self.flags

    def __str__(self) -> str:  # pragma: no cover
        def _f(v: Optional[float], fmt: str = ".4f") -> str:
            return f"{v:{fmt}}" if v is not None else "N/A"

        lines = [
            f"  Column : {self.column}",
            f"    Non-null count     : {self.non_null_count:,} / {self.total_rows:,}",
            f"    Mean               : {_f(self.mean)}",
            f"    Median             : {_f(self.median)}",
            f"    Mean/median ratio  : {_f(self.mean_median_ratio)}",
            f"    Std                : {_f(self.std)}",
            f"    Variance           : {_f(self.variance)}",
            f"    IQR                : {_f(self.iqr)}",
            f"    Min / Max          : {_f(self.min)} / {_f(self.max)}",
            f"    Skewness           : {_f(self.skewness)}  [{self.skew_severity or 'N/A'}]",
            f"    Kurtosis (excess)  : {_f(self.kurtosis)}  [{self.kurtosis_tag or 'N/A'}]",
            "    Percentiles:",
            f"      p1={_f(self.percentiles.p1)}  p5={_f(self.percentiles.p5)}"
            f"  p25={_f(self.percentiles.p25)}  p50={_f(self.percentiles.p50)}"
            f"  p75={_f(self.percentiles.p75)}  p95={_f(self.percentiles.p95)}"
            f"  p99={_f(self.percentiles.p99)}",
        ]
        if self.flags:
            lines.append(f"    Flags              : {', '.join(self.flags)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Top-level result
# ---------------------------------------------------------------------------


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

    columns: dict[str, ColumnNumericProfile] = field(default_factory=dict)
    analysed_columns: list[str] = field(default_factory=list)

    def __str__(self) -> str:  # pragma: no cover
        lines = ["=== Numeric Distribution Profile ==="]
        for profile in self.columns.values():
            lines.append(str(profile))
        return "\n".join(lines)

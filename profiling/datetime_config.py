"""
Result dataclasses for datetime column profiling.

Populated by DatetimeProfiler, which is opt-in via
ProfileConfig.datetime_columns.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional
from datetime import datetime


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class InferredGranularity(StrEnum):
    Yearly    = "yearly"     # median gap ≈ 365 days
    Monthly   = "monthly"    # median gap ≈ 30 days
    Weekly    = "weekly"     # median gap ≈ 7 days
    Daily     = "daily"      # median gap ≈ 1 day
    Hourly    = "hourly"     # median gap ≈ 1 hour
    Minutely  = "minutely"   # median gap ≈ 1 minute
    Secondly  = "secondly"   # median gap < 1 minute
    Irregular = "irregular"  # no dominant periodicity


class DatetimeFlag(StrEnum):
    FutureDates       = "future_dates"        # values > current date found
    HighGapVariance   = "high_gap_variance"   # coefficient of variation > 1.0
    MnarSuspected     = "mnar_suspected"      # non-trivial null rate (>5 %)
    RecentDataMissing = "recent_data_missing" # last 10 % of expected range is sparse


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

    has_year:         bool = False
    has_month:        bool = False
    has_day:          bool = False
    has_day_of_week:  bool = False
    has_hour:         bool = False
    has_is_weekend:   bool = False
    has_is_month_end: bool = False

    def extractable_features(self) -> list[str]:
        """Return names of features worth extracting downstream."""
        features = []
        if self.has_year:          features.append("year")
        if self.has_month:         features.append("month")
        if self.has_day:           features.append("day_of_month")
        if self.has_day_of_week:   features.append("day_of_week")
        if self.has_hour:          features.append("hour")
        if self.has_is_weekend:    features.append("is_weekend")
        if self.has_is_month_end:  features.append("is_month_end")
        return features


# ---------------------------------------------------------------------------
# Per-column result
# ---------------------------------------------------------------------------

@dataclass
class ColumnDatetimeProfile:
    """
    Full datetime profile for a single column.

    Attributes
    ----------
    column : str
        Column name.
    total_rows : int
        Total rows in the DataFrame.

    Range
    -----
    min_date : datetime | None
    max_date : datetime | None
    range_days : float | None
        (max_date - min_date).total_seconds() / 86 400.

    Missingness
    -----------
    null_count : int
    null_ratio : float

    Future dates
    ------------
    future_date_count : int
        Count of values strictly greater than the moment profiling ran.
    future_date_context : str
        Human-readable note — e.g. "May indicate data error or valid bookings."

    Granularity
    -----------
    inferred_granularity : InferredGranularity | None
    median_gap_seconds : float | None
        Median pairwise gap between consecutive sorted timestamps.
    gap_cv : float | None
        Coefficient of variation of gaps (std / mean).  High CV → irregular.

    Temporal signals
    ----------------
    signals : TemporalSignals

    Flags
    -----
    flags : list[DatetimeFlag]
    """

    column: str
    total_rows: int

    # Range
    min_date:   Optional[datetime] = None
    max_date:   Optional[datetime] = None
    range_days: Optional[float]    = None

    # Missingness
    null_count: int   = 0
    null_ratio: float = 0.0

    # Future dates
    future_date_count:   int = 0
    future_date_context: str = ""

    # Granularity
    inferred_granularity: Optional[InferredGranularity] = None
    median_gap_seconds:   Optional[float]               = None
    gap_cv:               Optional[float]               = None

    # Temporal signals
    signals: TemporalSignals = field(default_factory=TemporalSignals)

    # Flags
    flags: list[DatetimeFlag] = field(default_factory=list)

    def has_flag(self, flag: DatetimeFlag) -> bool:
        return flag in self.flags

    def __str__(self) -> str:  # pragma: no cover
        def _dt(v: Optional[datetime]) -> str:
            return v.isoformat() if v is not None else "N/A"

        def _f(v: Optional[float], fmt: str = ".4f") -> str:
            return f"{v:{fmt}}" if v is not None else "N/A"

        lines = [
            f"  Column : {self.column}",
            f"    Min date           : {_dt(self.min_date)}",
            f"    Max date           : {_dt(self.max_date)}",
            f"    Range (days)       : {_f(self.range_days, '.1f')}",
            f"    Null count         : {self.null_count:,}  ({self.null_ratio:.2%})",
            f"    Future dates       : {self.future_date_count:,}"
            + (f"  [{self.future_date_context}]" if self.future_date_context else ""),
            f"    Granularity        : {self.inferred_granularity or 'N/A'}",
            f"    Median gap (s)     : {_f(self.median_gap_seconds, '.1f')}",
            f"    Gap CV             : {_f(self.gap_cv)}",
            f"    Temporal signals   : {self.signals.extractable_features() or 'none'}",
        ]
        if self.flags:
            lines.append(f"    Flags              : {', '.join(self.flags)}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# Top-level result
# ---------------------------------------------------------------------------

@dataclass
class DatetimeProfileResult:
    """
    Datetime profile for all opted-in columns.

    Attributes
    ----------
    columns : dict[str, ColumnDatetimeProfile]
        Per-column profiles, keyed by column name.
    analysed_columns : list[str]
        Columns that were actually profiled (after schema intersection).
    """

    columns: dict[str, ColumnDatetimeProfile] = field(default_factory=dict)
    analysed_columns: list[str] = field(default_factory=list)

    def __str__(self) -> str:  # pragma: no cover
        lines = ["=== Datetime Profile ==="]
        for profile in self.columns.values():
            lines.append(str(profile))
        return "\n".join(lines)
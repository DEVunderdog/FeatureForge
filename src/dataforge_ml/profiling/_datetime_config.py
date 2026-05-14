"""
Result dataclasses for datetime column profiling.

Populated by DatetimeProfiler, which is opt-in via
ProfileConfig.datetime_columns.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional


class InferredGranularity(StrEnum):
    Yearly = "yearly"
    Monthly = "monthly"
    Weekly = "weekly"
    Daily = "daily"
    Hourly = "hourly"
    Minutely = "minutely"
    Secondly = "secondly"
    Irregular = "irregular"


class DatetimeFlag(StrEnum):
    FutureDates = "future_dates"
    HighGapVariance = "high_gap_variance"
    MnarSuspected = "mnar_suspected"
    RecentDateMissing = "recent_date_missing"


@dataclass
class TemporalSignals:
    has_year: bool = False
    has_month: bool = False
    has_day: bool = False
    has_day_of_week: bool = False
    has_hour: bool = False
    has_is_weekend: bool = False
    has_is_month_end: bool = False

    def extractable_features(self) -> list[str]:
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

    columns: dict[str, DatetimeStats] = field(default_factory=dict)
    analysed_columns: list[str] = field(default_factory=list)

    def __str__(self) -> str:  # pragma: no cover
        lines = ["=== Datetime Profile ==="]
        for profile in self.columns.values():
            lines.append(str(profile))
        return "\n".join(lines)

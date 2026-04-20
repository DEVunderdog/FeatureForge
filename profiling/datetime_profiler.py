"""
DatetimeProfiler  –  Phase 1 extension: Datetime Column Profiling.

Per-column metrics (opt-in via ProfileConfig.datetime_columns):
  1. Range              – min date, max date, total range in days
  2. Null analysis      – count, ratio, MNAR flag when null_ratio > 5 %
  3. Future dates       – count of values > now, with context note
  4. Granularity        – inferred periodicity from median consecutive gap;
                          high gap-CV flagged as irregular
  5. Temporal signals   – audit which of {year, month, day, day-of-week,
                          hour, is-weekend, is-month-end} vary in the data,
                          to guide downstream feature engineering

Granularity inference bands (median gap in seconds):
  < 90 s        → secondly
  < 3 600 s     → minutely
  < 7 200 s     → hourly
  < 172 800 s   → daily      (< 2 days)
  < 1 209 600 s → weekly     (< 14 days)
  < 5 184 000 s → monthly    (< 60 days)
  else          → yearly

Integration
-----------
Add ``datetime_columns: list[str] | None`` to ProfileConfig, then call::

    from profiling.datetime_profiler import DatetimeProfiler

    dt_profiler = DatetimeProfiler(
        columns=["created_at", "event_time"],
        config=cfg,
    )
    dt_result = dt_profiler.profile(df)

Attach ``dt_result`` to ``StructuralProfileResult`` as
``result.datetime``.
"""

from __future__ import annotations

import warnings
from datetime import datetime, timezone
from typing import Any

import polars as pl

from models.data_structure import DataStructure
from profiling.base import Profiling
from profiling.config import ProfileConfig
from profiling.datetime_config import (
    ColumnDatetimeProfile,
    DatetimeFlag,
    DatetimeProfileResult,
    InferredGranularity,
    TemporalSignals,
)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

# MNAR suspicion: missing rate above this fraction → flag
_MNAR_NULL_RATIO_THRESHOLD: float = 0.05

# Gap coefficient of variation above this → flag as irregular
_HIGH_GAP_CV_THRESHOLD: float = 1.0

# Granularity bands — upper bound (exclusive) in seconds for each label
# Ordered from finest to coarsest.
_GRANULARITY_BANDS: list[tuple[float, InferredGranularity]] = [
    (90.0,          InferredGranularity.Secondly),   # < 1.5 min
    (3_600.0,       InferredGranularity.Minutely),   # < 1 h
    (7_200.0,       InferredGranularity.Hourly),     # < 2 h
    (172_800.0,     InferredGranularity.Daily),      # < 2 days
    (1_209_600.0,   InferredGranularity.Weekly),     # < 14 days
    (5_184_000.0,   InferredGranularity.Monthly),    # < 60 days
]
# Anything ≥ 5_184_000 s → Yearly

# Recent-data sparsity: consider the last this-fraction of the total range
_RECENT_WINDOW_FRACTION: float = 0.10

# Accepted Polars datetime-like dtypes
_DATETIME_DTYPES = {pl.Date, pl.Datetime}


def _is_datetime_dtype(dtype: pl.DataType) -> bool:
    """Return True for Date, Datetime (any time-unit / tz)."""
    return isinstance(dtype, (pl.Date, pl.Datetime))


class DatetimeProfiler(Profiling[DatetimeProfileResult]):
    """
    Datetime distribution profiler for Polars DataFrames.

    Parameters
    ----------
    columns : list[str]
        Columns to profile.  Non-datetime columns are skipped with a warning.
    config : ProfileConfig | None
        Shared profiling configuration.
    """

    def __init__(
        self,
        columns: list[str],
        config: ProfileConfig | None = None,
    ) -> None:
        super().__init__(DataStructure.Tabular, config)
        self._requested_columns = columns

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def profile(self, data: Any) -> DatetimeProfileResult:
        if not isinstance(data, pl.DataFrame):
            raise TypeError(
                f"DatetimeProfiler expects a Polars DataFrame, "
                f"got {type(data).__name__}."
            )
        return self._run(data)

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def _run(self, df: pl.DataFrame) -> DatetimeProfileResult:
        result = DatetimeProfileResult()

        available = self._resolve_columns(df.columns, self._requested_columns)
        result.analysed_columns = available

        n_rows = df.height
        now = datetime.now(tz=timezone.utc)

        for col_name in available:
            series = df[col_name]

            # Attempt coercion for Utf8 columns not yet cast
            if series.dtype in (pl.Utf8, pl.String):
                coerced = series.str.to_datetime(strict=False)
                non_null_after = coerced.drop_nulls().len()
                if non_null_after == 0:
                    warnings.warn(
                        f"DatetimeProfiler: column '{col_name}' could not be "
                        f"coerced to datetime — skipping.",
                        stacklevel=2,
                    )
                    continue
                series = coerced
            elif not _is_datetime_dtype(series.dtype):
                warnings.warn(
                    f"DatetimeProfiler: column '{col_name}' has dtype "
                    f"{series.dtype!s}, which is not a datetime type — skipping.",
                    stacklevel=2,
                )
                continue

            profile = self._profile_column(series, col_name, n_rows, now)
            result.columns[col_name] = profile

        return result

    # ------------------------------------------------------------------
    # Per-column driver
    # ------------------------------------------------------------------

    def _profile_column(
        self,
        series: pl.Series,
        col_name: str,
        n_rows: int,
        now: datetime,
    ) -> ColumnDatetimeProfile:
        profile = ColumnDatetimeProfile(column=col_name, total_rows=n_rows)

        # Normalise to microsecond Datetime (UTC) for uniform arithmetic
        # Date columns are cast to Datetime at midnight UTC.
        if isinstance(series.dtype, pl.Date):
            series = series.cast(pl.Datetime("us", "UTC"))
        elif isinstance(series.dtype, pl.Datetime):
            if series.dtype.time_zone is None:
                series = series.dt.replace_time_zone("UTC")
            else:
                series = series.dt.convert_time_zone("UTC")

        # 1. Missingness
        self._compute_missingness(series, profile, n_rows)

        # Drop nulls for all remaining computations
        clean = series.drop_nulls()

        if clean.len() == 0:
            return profile

        # 2. Range
        self._compute_range(clean, profile)

        # 3. Future dates
        self._check_future_dates(clean, profile, now)

        # 4. Recent data sparsity (needs range, so after _compute_range)
        self._check_recent_data_missing(series, profile, n_rows)

        # 5. Granularity
        self._infer_granularity(clean, profile)

        # 6. Temporal signals
        self._audit_temporal_signals(clean, profile)

        return profile

    # ------------------------------------------------------------------
    # Step 1: Missingness
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_missingness(
        series: pl.Series,
        profile: ColumnDatetimeProfile,
        n_rows: int,
    ) -> None:
        null_count = series.null_count()
        profile.null_count = null_count
        profile.null_ratio = null_count / n_rows if n_rows > 0 else 0.0

        if profile.null_ratio > _MNAR_NULL_RATIO_THRESHOLD:
            profile.flags.append(DatetimeFlag.MnarSuspected)

    # ------------------------------------------------------------------
    # Step 2: Range
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_range(
        clean: pl.Series,
        profile: ColumnDatetimeProfile,
    ) -> None:
        min_ts = clean.min()
        max_ts = clean.max()

        if min_ts is not None:
            profile.min_date = min_ts.replace(tzinfo=timezone.utc) if isinstance(min_ts, datetime) else min_ts
        if max_ts is not None:
            profile.max_date = max_ts.replace(tzinfo=timezone.utc) if isinstance(max_ts, datetime) else max_ts

        if profile.min_date is not None and profile.max_date is not None:
            delta = profile.max_date - profile.min_date
            profile.range_days = delta.total_seconds() / 86_400.0

    # ------------------------------------------------------------------
    # Step 3: Future dates
    # ------------------------------------------------------------------

    @staticmethod
    def _check_future_dates(
        clean: pl.Series,
        profile: ColumnDatetimeProfile,
        now: datetime,
    ) -> None:
        # Cast to Int64 (epoch microseconds) and compare against now scalar
        now_us = int(now.timestamp() * 1_000_000)
        ts_int = clean.cast(pl.Int64)
        future_mask = ts_int > now_us
        future_count = int(future_mask.sum())

        profile.future_date_count = future_count
        if future_count > 0:
            profile.flags.append(DatetimeFlag.FutureDates)
            profile.future_date_context = (
                "Future dates detected. May indicate data entry error "
                "in historical datasets, or valid scheduled/booking records."
            )

    # ------------------------------------------------------------------
    # Step 3b: Recent data sparsity
    # ------------------------------------------------------------------

    @staticmethod
    def _check_recent_data_missing(
        series: pl.Series,
        profile: ColumnDatetimeProfile,
        n_rows: int,
    ) -> None:
        """
        Flag when the last _RECENT_WINDOW_FRACTION of the expected date
        range contains fewer observations than expected.

        We compare density in the recent window vs overall density.
        If the recent window has < 20 % of the expected density → flag.
        """
        if profile.min_date is None or profile.max_date is None:
            return
        if profile.range_days is None or profile.range_days == 0:
            return

        range_seconds = profile.range_days * 86_400.0
        window_seconds = range_seconds * _RECENT_WINDOW_FRACTION

        # Compute cutoff as epoch microseconds
        max_ts_us = int(profile.max_date.timestamp() * 1_000_000)
        window_us = int(window_seconds * 1_000_000)
        cutoff_us = max_ts_us - window_us

        # Cast series to Int64 (epoch microseconds) for comparison
        ts_int = series.cast(pl.Int64)
        recent_mask = ts_int >= cutoff_us
        recent_count = int(recent_mask.sum())

        # Expected count if uniform distribution
        total_non_null = series.drop_nulls().len()
        if total_non_null == 0:
            return
        expected_recent = total_non_null * _RECENT_WINDOW_FRACTION
        density_ratio = recent_count / expected_recent if expected_recent > 0 else 1.0

        if density_ratio < 0.20:
            profile.flags.append(DatetimeFlag.RecentDataMissing)

    # ------------------------------------------------------------------
    # Step 4: Granularity inference
    # ------------------------------------------------------------------

    @staticmethod
    def _infer_granularity(
        clean: pl.Series,
        profile: ColumnDatetimeProfile,
    ) -> None:
        """
        Sort values, compute consecutive gaps in seconds, derive median gap.

        Uses Int64 epoch-microsecond representation for vectorised diff.
        """
        n = clean.len()
        if n < 2:
            profile.inferred_granularity = InferredGranularity.Irregular
            return

        ts_us = clean.sort().cast(pl.Int64)  # microseconds since epoch
        gaps_us = ts_us.diff().drop_nulls()  # consecutive differences

        # Discard zero and negative gaps (exact duplicates or out-of-order noise)
        gaps_us = gaps_us.filter(gaps_us > 0)

        if gaps_us.len() == 0:
            profile.inferred_granularity = InferredGranularity.Irregular
            return

        gaps_s = gaps_us.cast(pl.Float64) / 1_000_000.0  # → seconds

        median_gap_s = float(gaps_s.median())  # type: ignore[arg-type]
        mean_gap_s   = float(gaps_s.mean())    # type: ignore[arg-type]
        std_gap_s    = float(gaps_s.std(ddof=1)) if gaps_s.len() > 1 else 0.0

        profile.median_gap_seconds = median_gap_s

        # Coefficient of variation (robust to skewed gap distributions)
        if mean_gap_s > 0:
            profile.gap_cv = std_gap_s / mean_gap_s
            if profile.gap_cv > _HIGH_GAP_CV_THRESHOLD:
                profile.flags.append(DatetimeFlag.HighGapVariance)
        else:
            profile.gap_cv = 0.0

        # Map median gap to granularity label
        granularity = InferredGranularity.Yearly  # default (coarsest)
        for upper_bound, label in _GRANULARITY_BANDS:
            if median_gap_s < upper_bound:
                granularity = label
                break

        profile.inferred_granularity = granularity

    # ------------------------------------------------------------------
    # Step 5: Temporal signal audit
    # ------------------------------------------------------------------

    @staticmethod
    def _audit_temporal_signals(
        clean: pl.Series,
        profile: ColumnDatetimeProfile,
    ) -> None:
        """
        Check which temporal features vary across rows

        All checks are done via Polars expressions on the full clean series,
        so no Python-level loops are required.
        """
        signals = TemporalSignals()

        years   = clean.dt.year()
        months  = clean.dt.month()
        days    = clean.dt.day()
        dow     = clean.dt.weekday()   # 0=Monday … 6=Sunday
        hours   = clean.dt.hour()

        signals.has_year        = years.n_unique() > 1
        signals.has_month       = months.n_unique() > 1
        signals.has_day         = days.n_unique() > 1
        signals.has_day_of_week = dow.n_unique() > 1
        signals.has_hour        = int(hours.max()) > 0  # type: ignore[arg-type]

        # Weekend signal is only meaningful when day-of-week varies
        if signals.has_day_of_week:
            weekend_mask = dow >= 5  # Saturday=5, Sunday=6
            signals.has_is_weekend = bool(weekend_mask.any())

        # Month-end: day == last day of the respective month
        # We approximate: day == 28/29/30/31 AND next-day's month ≠ current month.
        # Polars has dt.month_end() which returns the last day of the month.
        try:
            month_end_ts = clean.dt.month_end()
            # Strip time component for date-level comparison
            is_month_end_mask = (
                clean.dt.year()  == month_end_ts.dt.year()
            ) & (
                clean.dt.month() == month_end_ts.dt.month()
            ) & (
                clean.dt.day()   == month_end_ts.dt.day()
            )
            signals.has_is_month_end = bool(is_month_end_mask.any())
        except Exception:
            # Fallback: flag if day ≥ 28
            signals.has_is_month_end = bool((days >= 28).any())

        profile.signals = signals
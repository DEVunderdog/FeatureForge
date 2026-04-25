"""
NumericProfiler  –  Phase 1 extension: Numeric Distribution Profiling.

Per-column metrics (opt-in via ProfileConfig.numeric_columns):
  1. Central tendency     – mean, median, mean/median ratio
  2. Spread               – std, variance, IQR (Q3 – Q1)
  3. Skewness & kurtosis  – with severity/tag labels
  4. Range                – min, max
  5. Percentile profile   – p1, p5, p25, p50, p75, p95, p99
  6. Scale-anomaly flag   – values spanning 3+ orders of magnitude

Only numeric Polars dtypes are profiled; string columns in the list are
silently skipped (a warning is produced if the caller passes non-numeric
column names).

Integration
-----------
Add ``numeric_columns: list[str] | None`` to ProfileConfig, then call::

    from profiling.numeric_profiler import NumericProfiler

    num_profiler = NumericProfiler(
        columns=["age", "income", "temperature"],
        config=cfg,
    )
    num_result = num_profiler.profile(df)

Attach ``num_result`` to ``TabularProfileResult`` as
``result.numeric_profile``.
"""

from __future__ import annotations

import warnings
from typing import Any

import polars as pl

from ..models._data_structure import DataStructure
from ._base import Profiling
from .config import ProfileConfig
from ._correlation_profiler import _INT_DTYPES
from ._numeric_config import (
    ColumnNumericProfile,
    KurtosisTag,
    NumericFlag,
    NumericProfileResult,
    PercentileProfile,
    SkewSeverity,
    NumericTopValueEntry,
    HistogramBin,
)

# ---------------------------------------------------------------------------
# Thresholds (documented so callers can see what drives labels / flags)
# ---------------------------------------------------------------------------

# Skewness severity bands (applied to |skewness|)
_SKEW_NORMAL   = 0.5   # |skew| ≤ this  →  normal
_SKEW_MODERATE = 1.0   # |skew| ≤ this  →  moderate
_SKEW_HIGH     = 2.0   # |skew| ≤ this  →  high
#                        |skew| > 2.0   →  severe

# Excess kurtosis bands
_KURT_PLATY_UPPER = -1.0   # excess < this  →  platykurtic
_KURT_LEPTO_LOWER =  3.0   # excess > this  →  leptokurtic
#                            else          →  mesokurtic

# Scale-anomaly: flag when max/min ratio spans ≥ 3 orders of magnitude
_SCALE_ORDERS_OF_MAGNITUDE = 3  # i.e. ratio ≥ 10^3

# Numeric dtypes accepted for profiling
_PROFILED_DTYPES = {
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
    pl.Float32, pl.Float64,
}

# Percentile quantile levels (in order)
_QUANTILE_LEVELS = (0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99)
_NEAR_CONSTANT_THRESHOLD = 0.90
_DISCRETE_MAX_UNIQUE = 20


class NumericProfiler(Profiling[NumericProfileResult]):
    """
    Numeric distribution profiler for Polars DataFrames.

    Parameters
    ----------
    columns : list[str]
        Columns to profile.  Non-numeric or absent columns are skipped
        with a warning; they do not raise.
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

    def profile(self, data: Any) -> NumericProfileResult:
        if not isinstance(data, pl.DataFrame):
            raise TypeError(
                f"NumericProfiler expects a Polars DataFrame, "
                f"got {type(data).__name__}."
            )
        return self._run(data)

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def _run(self, df: pl.DataFrame) -> NumericProfileResult:
        result = NumericProfileResult()

        # Intersect requested columns with the actual schema
        available = self._resolve_columns(df.columns, self._requested_columns)
        result.analysed_columns = available

        n_rows = df.height

        for col_name in available:
            series = df[col_name]

            # Skip non-numeric columns with a warning
            if series.dtype not in _PROFILED_DTYPES:
                warnings.warn(
                    f"NumericProfiler: column '{col_name}' has dtype "
                    f"{series.dtype!s}, which is not numeric — skipping.",
                    stacklevel=2,
                )
                continue

            profile = self._profile_column(series, col_name, n_rows)
            result.columns[col_name] = profile

        return result

    # ------------------------------------------------------------------
    # Per-column driver
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_frequency_and_distribution(
        original_series: pl.Series,
        clean_f64: pl.Series,
        profile: ColumnNumericProfile,
        n_rows: int,
    ) -> None:
        """
        Compute Mode, and depending on whether the feature is continuous or discrete, 
        calculate a 20-bin histogram OR Top-10 value counts.
        """
        if clean_f64.len() == 0:
            return

        vc = clean_f64.value_counts(sort=True)
        col_name = clean_f64.name

        # --- Absolute Mode Frequency ---
        mode_val = float(vc[col_name][0])
        mode_count = int(vc["count"][0])
        mode_freq = mode_count / n_rows if n_rows > 0 else 0.0

        profile.mode = mode_val
        profile.mode_frequency = mode_freq

        if mode_freq > _NEAR_CONSTANT_THRESHOLD:
            profile.flags.append(NumericFlag.NearConstant)

        n_unique = vc.height
        is_discrete = original_series.dtype in _INT_DTYPES or n_unique <= _DISCRETE_MAX_UNIQUE

        if is_discrete:
            # --- Top-10 Distribution (Discrete) ---
            top_rows = min(10, n_unique)
            profile.top_values =[
                NumericTopValueEntry(
                    value=float(vc[col_name][i]),
                    count=int(vc["count"][i]),
                    percentage=int(vc["count"][i]) / n_rows if n_rows > 0 else 0.0
                )
                for i in range(top_rows)
            ]
        else:
            # --- 20-Bin Histogram Distribution (Continuous) ---
            col_min = profile.min
            col_max = profile.max

            if col_min is not None and col_max is not None and col_max > col_min:
                span = col_max - col_min
                
                # Robust math-based binning to avoid Polars API version mismatches
                bin_expr = ((pl.col(col_name) - col_min) / span * 20).floor().cast(pl.Int64)
                bin_expr = pl.when(bin_expr >= 20).then(19).otherwise(bin_expr)  # Clamp max bound inclusive

                df_binned = pl.DataFrame({col_name: clean_f64}).select(bin_idx=bin_expr)
                bin_counts = df_binned.group_by("bin_idx").len()
                
                # Extract into dictionary for fast lookup
                counts_dict = {r[0]: r[1] for r in bin_counts.iter_rows()}

                hist_bins =[]

                for i in range(20):
                    # Calculate bounds for the current bin
                    lower_bound = col_min + (i / 20.0) * span
                    upper_bound = col_min + ((i + 1) / 20.0) * span
                    
                    # Fetch count from dictionary, defaulting to 0
                    b_count = counts_dict.get(i, 0)
                    b_pct = b_count / n_rows if n_rows > 0 else 0.0
                    
                    hist_bins.append(
                        HistogramBin(
                            lower_bound=lower_bound,
                            upper_bound=upper_bound,
                            count=b_count,
                            percentage=b_pct
                        )
                    )
                
                profile.histogram = hist_bins
            elif col_min == col_max and col_min is not None:
                # Edge case: Column is completely uniform but was classified as continuous 
                # (e.g. all 1.5). Just create a single bin.
                profile.histogram =[
                    HistogramBin(
                        lower_bound=col_min,
                        upper_bound=col_max,
                        count=profile.non_null_count,
                        percentage=profile.non_null_count / n_rows if n_rows > 0 else 0.0
                    )
                ]

    def _profile_column(
        self,
        series: pl.Series,
        col_name: str,
        n_rows: int,
    ) -> ColumnNumericProfile:
        profile = ColumnNumericProfile(column=col_name, total_rows=n_rows)

        # Cast to Float64 for uniform arithmetic; avoids integer overflow
        # in variance/skewness calculations.
        f64 = series.cast(pl.Float64)
        clean = f64.drop_nulls()
        profile.non_null_count = clean.len()

        if profile.non_null_count == 0:
            # Nothing to compute — leave all fields as None
            return profile

        # 1. Central tendency
        self._compute_central_tendency(clean, profile)

        self._compute_range(clean, profile)

        self._compute_frequency_and_distribution(series, clean, profile, n_rows)

        # 2 & 5. Percentiles (IQR is derived from these, so compute first)
        self._compute_percentiles(clean, profile)

        # 3. Spread  (std, variance; IQR comes from percentiles)
        self._compute_spread(clean, profile)

        # 4. Shape — skewness and kurtosis
        self._compute_shape(clean, profile)

        # 7. Flags
        self._check_scale_anomaly(clean, profile)

        return profile

    # ------------------------------------------------------------------
    # Step 1: Central tendency
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_central_tendency(
        clean: pl.Series,
        profile: ColumnNumericProfile,
    ) -> None:
        mean   = float(clean.mean())      # type: ignore[arg-type]
        median = float(clean.median())    # type: ignore[arg-type]

        profile.mean   = mean
        profile.median = median

        # Mean/median ratio: primary skew indicator at a glance.
        # Guard against division by zero (e.g. a column of all zeros).
        if median == 0.0:
            profile.mean_median_ratio = float("inf") if mean != 0.0 else 1.0
        else:
            profile.mean_median_ratio = mean / median

    # ------------------------------------------------------------------
    # Step 2: Spread
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_spread(
        clean: pl.Series,
        profile: ColumnNumericProfile,
    ) -> None:
        n = clean.len()
        if n < 2:
            # Std / variance undefined for a single observation
            profile.std      = 0.0
            profile.variance = 0.0
            return

        std = float(clean.std(ddof=1))   # type: ignore[arg-type]
        profile.std      = std
        profile.variance = std ** 2

    # ------------------------------------------------------------------
    # Step 3: Shape — skewness and kurtosis
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_shape(
        clean: pl.Series,
        profile: ColumnNumericProfile,
    ) -> None:
        n = clean.len()
        if n < 3:
            # Skewness / kurtosis require at least 3 observations
            return

        mean = profile.mean
        std  = profile.std

        if std is None or std == 0.0:
            # Constant column — skewness and kurtosis are undefined
            profile.skewness      = 0.0
            profile.kurtosis      = 0.0
            profile.skew_severity = SkewSeverity.Normal
            profile.kurtosis_tag  = KurtosisTag.Mesokurtic
            return

        # Standardised deviations
        z = (clean - mean) / std

        # --- Skewness: Fisher's adjusted (bias-corrected) ---
        # G1 = [n / ((n-1)(n-2))] * Σ z³
        skew_raw = float((z ** 3).sum())
        skewness = (n / ((n - 1) * (n - 2))) * skew_raw

        # --- Excess kurtosis: bias-corrected ---
        # G2 = [n(n+1) / ((n-1)(n-2)(n-3))] * Σ z⁴
        #      - 3(n-1)² / ((n-2)(n-3))
        kurt_raw = float((z ** 4).sum())
        kurtosis = (
            (n * (n + 1)) / ((n - 1) * (n - 2) * (n - 3)) * kurt_raw
            - 3 * (n - 1) ** 2 / ((n - 2) * (n - 3))
        )

        profile.skewness = skewness
        profile.kurtosis = kurtosis

        # Skew severity label
        abs_skew = abs(skewness)
        if abs_skew <= _SKEW_NORMAL:
            profile.skew_severity = SkewSeverity.Normal
        elif abs_skew <= _SKEW_MODERATE:
            profile.skew_severity = SkewSeverity.Moderate
        elif abs_skew <= _SKEW_HIGH:
            profile.skew_severity = SkewSeverity.High
        else:
            profile.skew_severity = SkewSeverity.Severe

        # Kurtosis tag
        if kurtosis < _KURT_PLATY_UPPER:
            profile.kurtosis_tag = KurtosisTag.Platykurtic
        elif kurtosis > _KURT_LEPTO_LOWER:
            profile.kurtosis_tag = KurtosisTag.Leptokurtic
        else:
            profile.kurtosis_tag = KurtosisTag.Mesokurtic

    # ------------------------------------------------------------------
    # Step 4: Range
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_range(
        clean: pl.Series,
        profile: ColumnNumericProfile,
    ) -> None:
        profile.min = float(clean.min())   # type: ignore[arg-type]
        profile.max = float(clean.max())   # type: ignore[arg-type]

    # ------------------------------------------------------------------
    # Step 5: Percentiles
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_percentiles(
        clean: pl.Series,
        profile: ColumnNumericProfile,
    ) -> None:
        # Polars quantile() is O(n log n) once; compute all at once via select
        # to avoid repeated passes.
        quantile_frame = pl.DataFrame({"v": clean}).select(
            [
                pl.col("v").quantile(q, interpolation="linear").alias(f"q{i}")
                for i, q in enumerate(_QUANTILE_LEVELS)
            ]
        )
        row = quantile_frame.row(0)
        # row order: p1, p5, p25, p50, p75, p95, p99
        profile.percentiles = PercentileProfile(
            p1  = row[0],
            p5  = row[1],
            p25 = row[2],
            p50 = row[3],
            p75 = row[4],
            p95 = row[5],
            p99 = row[6],
        )

    # ------------------------------------------------------------------
    # Step 6: Scale-anomaly flag
    # ------------------------------------------------------------------

    @staticmethod
    def _check_scale_anomaly(
        clean: pl.Series,
        profile: ColumnNumericProfile,
    ) -> None:
        """
        Flag when values span ≥ 3 orders of magnitude *on the positive side*.

        Rationale: a column with values like [0.002, 15000] almost certainly
        mixes units or scales, which will mislead distance-based models.

        We use the absolute-value range to handle columns that cross zero
        (e.g. log-returns that go from -0.05 to 500).  Columns whose
        entire range is within [-1, 1] are exempt (percentages, probabilities).
        """
        col_min = profile.min
        col_max = profile.max

        if col_min is None or col_max is None:
            return

        abs_min = abs(col_min)
        abs_max = abs(col_max)

        # Skip all-zero or all-same-sign tiny ranges
        if abs_max == 0.0:
            return

        # Exempt probability / ratio columns
        if abs_max <= 1.0 and abs_min <= 1.0:
            return

        # Compute orders of magnitude
        if abs_min == 0.0:
            # Any non-zero max with a zero minimum → infinite ratio →
            # conservatively flag if max is large enough to be suspicious.
            if abs_max >= 10 ** _SCALE_ORDERS_OF_MAGNITUDE:
                profile.flags.append(NumericFlag.ScaleAnomaly)
            return

        ratio = abs_max / abs_min
        if ratio >= 10 ** _SCALE_ORDERS_OF_MAGNITUDE:
            profile.flags.append(NumericFlag.ScaleAnomaly)
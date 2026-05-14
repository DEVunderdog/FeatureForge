"""
BooleanProfiler  –  Phase 1 extension: Boolean Column Profiling.

Handles columns classified as SemanticType.Boolean, which includes:
  - Native Polars Boolean dtype
  - Integer {0, 1} columns with a Boolean override in ProfileConfig
  - Boolean-string columns ("true"/"false", "yes"/"no", "1"/"0") with override

Per-column metrics:
  1. true_count   – count of non-null truthy values
  2. false_count  – count of non-null falsy values
  3. true_ratio   – true_count / non_null_count  (nulls excluded)
  4. false_ratio  – false_count / non_null_count (nulls excluded)
  5. mode         – most frequent non-null value (True / False), or None if tied

Null values are NOT counted in ratios — missingness is already captured by
the upstream MissingnessProfiler pass and lives in ColumnProfile.missingness.
"""

from __future__ import annotations

import polars as pl

from ._base import ColumnBatchProfiler
from .config import (
    ProfileConfig,
    BooleanStats,
    SemanticType,
)
from ._boolean_config import BooleanProfileResult
from ..models._data_types import _INT_DTYPES

# ---------------------------------------------------------------------------
# String values that represent True / False
# ---------------------------------------------------------------------------

_TRUE_STRINGS: frozenset[str] = frozenset({"true", "yes", "1", "t", "y"})
_FALSE_STRINGS: frozenset[str] = frozenset({"false", "no", "0", "f", "n"})


class BooleanProfiler(ColumnBatchProfiler[BooleanProfileResult]):
    """
    Boolean column profiler for Polars DataFrames.

    A column is eligible when:
      - Its Polars dtype is pl.Boolean, OR
      - Its dtype is an integer with values exclusively in {0, 1}, OR
      - It has a SemanticType.Boolean override in ProfileConfig.column_overrides

    Non-eligible columns in the provided list are silently skipped.

    Parameters
    ----------
    config : ProfileConfig | None
        Shared profiling configuration.
    """

    def __init__(self, config: ProfileConfig | None = None) -> None:
        super().__init__(config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def profile(
        self,
        data: pl.DataFrame,
        columns: list[str],
    ) -> BooleanProfileResult:
        return self._run(data, columns)

    # ------------------------------------------------------------------
    # Eligibility
    # ------------------------------------------------------------------

    def _eligible(self, series: pl.Series) -> bool:
        override = self.config.column_overrides.get(series.name)

        # Explicit override — trust it
        if override == SemanticType.Boolean:
            return True

        # Another override takes precedence over auto-detection
        if override is not None:
            return False

        # Native boolean dtype
        if series.dtype == pl.Boolean:
            return True

        # Integer {0, 1} column — check after dropping nulls
        if series.dtype in _INT_DTYPES:
            clean = series.drop_nulls()
            if clean.len() == 0:
                return False
            unique_vals = set(clean.unique().to_list())
            return unique_vals <= {0, 1}

        return False

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def _run(
        self,
        df: pl.DataFrame,
        columns: list[str],
    ) -> BooleanProfileResult:
        result = BooleanProfileResult()

        available = [
            c
            for c in self._resolve_columns(df.columns, columns)
            if self._eligible(df[c])
        ]
        result.analysed_columns = available

        for col_name in available:
            result.columns[col_name] = self._profile_column(df[col_name], df.height)

        return result

    # ------------------------------------------------------------------
    # Per-column driver
    # ------------------------------------------------------------------

    def _profile_column(
        self,
        series: pl.Series,
        n_rows: int,
    ) -> BooleanStats:
        profile = BooleanStats()

        # Coerce to a clean boolean series (drop nulls)
        bool_series = self._to_bool_series(series)
        non_null_count = bool_series.len()

        if non_null_count == 0:
            return profile

        true_count = int(bool_series.sum())
        false_count = non_null_count - true_count

        profile.true_count = true_count
        profile.false_count = false_count
        profile.true_ratio = true_count / non_null_count
        profile.false_ratio = false_count / non_null_count

        # Mode: True if more trues, False if more falses, None if perfectly tied
        if true_count > false_count:
            profile.mode = True
        elif false_count > true_count:
            profile.mode = False
        else:
            profile.mode = None  # tied — no single mode

        return profile

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _to_bool_series(series: pl.Series) -> pl.Series:
        """
        Return a null-free Boolean Series regardless of the input dtype.

        Handles:
          - pl.Boolean      → drop nulls directly
          - integer {0, 1}  → cast to Boolean, drop nulls
          - string          → map known true/false strings, drop nulls
        """
        if series.dtype == pl.Boolean:
            return series.drop_nulls()

        if series.dtype in _INT_DTYPES:
            return series.cast(pl.Boolean).drop_nulls()

        if series.dtype == pl.Utf8:
            lower = series.str.to_lowercase().str.strip_chars()
            true_mask = lower.is_in(list(_TRUE_STRINGS))
            false_mask = lower.is_in(list(_FALSE_STRINGS))
            known_mask = true_mask | false_mask
            return true_mask.filter(known_mask)

        # Fallback: attempt a cast and drop nulls (covers e.g. Categorical)
        try:
            return series.cast(pl.Boolean).drop_nulls()
        except Exception:
            return pl.Series([], dtype=pl.Boolean)

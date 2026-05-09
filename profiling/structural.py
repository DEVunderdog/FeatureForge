"""
StructuralProfiler  –  unified Phase 1 entry point.
"""

from __future__ import annotations

import math
from typing import Any

import polars as pl

from ._tabular import TabularProfiler
from ._categorical import CategoricalProfiler
from ._datetime_profiler import DatetimeProfiler
from .config import (
    ProfileConfig,
    ColumnProfile,
    StructuralProfileResult,
    RowMissingnessDistribution,
    SemanticType,
    Modality,
)
from ._numeric_profiler import NumericProfiler
from ._missingness_profiler import MissingnessProfiler
from ._target_profiler import TargetProfiler
from ._correlation_profiler import CorrelationProfiler

_ROW_DROP_THRESHOLD = 0.50


class StructuralProfiler:

    def __init__(self, config: ProfileConfig | None = None) -> None:
        self.config = config or ProfileConfig()

        if self.config.modality == Modality.Tabular:
            self.modality_profiler = TabularProfiler(self.config)
        else:
            raise NotImplementedError(
                f"modality {self.config.modality} not supported yet"
            )

    def profile(self, data: Any) -> StructuralProfileResult:
        if not isinstance(data, pl.DataFrame):
            raise TypeError(
                f"StructuralProfiler expects a Polars DataFrame, "
                f"got {type(data).__name__}."
            )

        result = StructuralProfileResult()

        active_cols = [c for c in data.columns if c not in self.config.exclude_columns]

        dataset_stats = self.modality_profiler.profile(data)
        result.dataset = dataset_stats

        missingness_result = MissingnessProfiler(config=self.config).profile(
            data, columns=active_cols
        )

        for col_name in missingness_result.analysed_columns:
            result.columns.setdefault(
                col_name, ColumnProfile(name=col_name)
            ).missingness = missingness_result.columns.get(col_name)

        if missingness_result.correlation_matrix:
            result.dataset.missingness_matrix = missingness_result.correlation_matrix

        # ── 4. Row-missingness distribution ─────────────────────────────
        result.dataset.row_distribution = self._compute_row_distribution(
            df=data,
            cols=active_cols,
            n_rows=data.height,
            overrides=self.config.column_overrides,
        )

        for profiler in [
            NumericProfiler(config=self.config),
            CategoricalProfiler(config=self.config),
            DatetimeProfiler(config=self.config),
        ]:
            sub_result = profiler.profile(data, columns=active_cols)
            for col_name, col_stats in sub_result.columns.items():
                result.columns.setdefault(
                    col_name, ColumnProfile(name=col_name)
                ).stats = col_stats

        if self.config.target_columns is not None:
            for target in self.config.target_columns:
                if target not in data.columns:
                    continue

                target_result = TargetProfiler(
                    target_column=target,
                    config=self.config,
                ).profile(data)

                result.targets[target] = target_result

                col_profile = result.columns.setdefault(
                    target,
                    ColumnProfile(name=target),
                )
                col_profile.is_target = True

                if target_result.numeric_profile is not None:
                    col_profile.stats = target_result.numeric_profile
                elif target_result.categorical_profile is not None:
                    col_profile.stats = target_result.categorical_profile

        # ── 7. Correlation ───────────────────────────────────────────────
        # CHANGE: was never called before.
        if self.config.compute_correlation:
            corr_result = CorrelationProfiler(
                numeric_columns=active_cols,
                categorical_columns=active_cols,
                target_column=self.config.correlation_target_column,
                config=self.config,
            ).profile(data)
            result.dataset.correlation = corr_result

        return result

    @staticmethod
    def _compute_row_distribution(
        df: pl.DataFrame,
        cols: list[str],
        n_rows: int,
        overrides: dict[str, SemanticType],
    ) -> RowMissingnessDistribution:
        from ._missingness_profiler import (
            _sentinel_eligible,
            _inf_eligible,
            _SENTINEL_STRINGS,
        )

        dist = RowMissingnessDistribution()
        if n_rows == 0 or not cols:
            return dist

        n_cols = len(cols)
        per_col_exprs = []

        for col_name in cols:
            dtype = df[col_name].dtype
            override = overrides.get(col_name)
            null_e = pl.col(col_name).is_null()

            if _sentinel_eligible(dtype, override):
                eff = (
                    null_e
                    | (pl.col(col_name).str.strip_chars() == "")
                    | pl.col(col_name).str.to_uppercase().is_in(list(_SENTINEL_STRINGS))
                )
            elif _inf_eligible(dtype):
                eff = (
                    null_e | pl.col(col_name).is_nan() | pl.col(col_name).is_infinite()
                )
            else:
                eff = null_e

            per_col_exprs.append(eff.cast(pl.Int8).alias(col_name))

        row_missing: pl.Series = df.select(per_col_exprs).select(
            pl.sum_horizontal(pl.all()).alias("row_missing")
        )["row_missing"]

        half_threshold = math.ceil(n_cols * _ROW_DROP_THRESHOLD)

        dist.pct_zero_missing = float((row_missing == 0).sum()) / n_rows
        dist.pct_one_to_two = (
            float(((row_missing >= 1) & (row_missing <= 2)).sum()) / n_rows
        )
        dist.pct_three_to_five = (
            float(((row_missing >= 3) & (row_missing <= 5)).sum()) / n_rows
        )
        dist.pct_over_five = float((row_missing > 5).sum()) / n_rows
        dist.drop_candidate_row_count = int((row_missing >= half_threshold).sum())
        dist.pct_over_half_missing = dist.drop_candidate_row_count / n_rows

        return dist

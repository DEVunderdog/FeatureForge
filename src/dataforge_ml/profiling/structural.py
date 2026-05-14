"""
StructuralProfiler  –  unified Phase 1 entry point.

Execution order inside profile(df):
  1. ModalityProfiler      → result.dataset (DatasetStats)
  2. MissingnessProfiler   → ColumnProfile.missingness + dataset.missingness_matrix
  3. Row-missingness dist  → dataset.row_distribution
  4. TypeDetector          → ColumnProfile.semantic_type / type_flags / dtypes
  5. column_overrides      → replace SemanticType on existing ColumnProfiles
  6. ColumnTypeProfiler    → route each column to its profiler by SemanticType;
                             Identifier columns: skip, stats stays None
  7. target_columns        → TargetProfiler; mark ColumnProfile.is_target=True
  8. Correlation           → if compute_correlation=True:
       a. profile_features()  → dataset.feature_correlation  (computed once)
       b. profile_target()    → dataset.target_correlations[target]
                                (once per declared target column)
"""

from __future__ import annotations

import math
from typing import Any

import polars as pl

from ._base import ModalityProfiler, ColumnBatchProfiler
from ._tabular import TabularProfiler
from ._categorical import CategoricalProfiler
from ._datetime_profiler import DatetimeProfiler
from ._numeric_profiler import NumericProfiler
from ._boolean_profiler import BooleanProfiler
from ._text_profiler import TextProfiler
from ._missingness_profiler import MissingnessProfiler
from ._target_profiler import TargetProfiler
from ._correlation_profiler import CorrelationProfiler
from ._type_detector import TypeDetector
from .config import (
    ProfileConfig,
    ColumnProfile,
    StructuralProfileResult,
    RowMissingnessDistribution,
    SemanticType,
    Modality,
)

_ROW_DROP_THRESHOLD = 0.50

# ---------------------------------------------------------------------------
# Registry: SemanticType → ColumnTypeProfiler class
#
# Stateless between profile(series, df) calls, so one instance per
# SemanticType safely handles all columns of that type in one run.
# Add Boolean / Text profilers here when implemented.
# ---------------------------------------------------------------------------
_COLUMN_PROFILER_REGISTRY: dict[SemanticType, type[ColumnBatchProfiler]] = {  # type: ignore[type-arg]
    SemanticType.Numeric: NumericProfiler,
    SemanticType.Categorical: CategoricalProfiler,
    SemanticType.Datetime: DatetimeProfiler,
    SemanticType.Boolean: BooleanProfiler,
    SemanticType.Text: TextProfiler,
}


class StructuralProfiler:

    def __init__(self, config: ProfileConfig | None = None) -> None:
        self.config = config or ProfileConfig()

        if self.config.modality == Modality.Tabular:
            self.modality_profiler: ModalityProfiler = TabularProfiler(self.config)
        else:
            raise NotImplementedError(
                f"modality {self.config.modality} not supported yet"
            )

    # ------------------------------------------------------------------
    # Public entry point
    # ------------------------------------------------------------------

    def profile(self, data: Any) -> StructuralProfileResult:
        if not isinstance(data, pl.DataFrame):
            raise TypeError(
                f"StructuralProfiler expects a Polars DataFrame, "
                f"got {type(data).__name__}."
            )

        result = StructuralProfileResult()

        active_cols = [c for c in data.columns if c not in self.config.exclude_columns]

        # ── 1. Modality profiler ─────────────────────────────────────────
        # Replaces default DatasetStats with the real one (row_count, memory,
        # duplicates, etc.).  Must run before anything writes to result.dataset.
        result.dataset = self.modality_profiler.profile(data)

        # ── 2. Missingness pre-pass ──────────────────────────────────────
        # setdefault creates ColumnProfile entries; subsequent steps mutate
        # the same objects via the same setdefault pattern.
        missingness_result = MissingnessProfiler(config=self.config).profile(
            data, columns=active_cols
        )
        for col_name in missingness_result.analysed_columns:
            cp = result.columns.setdefault(col_name, ColumnProfile(name=col_name))
            cp.missingness = missingness_result.columns.get(col_name)

        if missingness_result.correlation_matrix:
            result.dataset.missingness_matrix = missingness_result.correlation_matrix

        # ── 3. Row-missingness distribution ─────────────────────────────
        result.dataset.row_distribution = self._compute_row_distribution(
            df=data,
            cols=active_cols,
            n_rows=data.height,
            overrides=self.config.column_overrides,
        )

        # ── 4. Type detection ────────────────────────────────────────────
        # setdefault returns the existing ColumnProfile from step 2, so
        # missingness and type info land on the same object.
        type_info = TypeDetector(columns=active_cols).detect(data)
        for col_name, info in type_info.items():
            cp = result.columns.setdefault(col_name, ColumnProfile(name=col_name))
            cp.semantic_type = info.semantic_type
            cp.type_flags = list(info.flags)
            cp.original_dtype = info.original_dtype
            cp.inferred_dtype = info.inferred_dtype

        # ── 5. Apply column_overrides ────────────────────────────────────
        # All active columns are in result.columns by now (steps 2 + 4).
        # Overrides for excluded / non-existent columns are silently ignored.
        for col_name, override_type in self.config.column_overrides.items():
            if col_name in result.columns:
                result.columns[col_name].semantic_type = override_type

        # ── 6. Per-column profiling routed by SemanticType ───────────────
        # Batch all columns of the same SemanticType together and call each
        # profiler once with (df, column_list) — matching the profiler API.
        type_to_cols: dict[SemanticType, list[str]] = {}
        for col_name in active_cols:
            cp = result.columns.get(col_name)
            if cp is None or cp.semantic_type is None:
                continue
            if cp.semantic_type == SemanticType.Identifier:
                continue
            sem_type = cp.semantic_type
            type_to_cols.setdefault(sem_type, []).append(col_name)

        for sem_type, cols in type_to_cols.items():
            profiler_cls = _COLUMN_PROFILER_REGISTRY.get(sem_type)  # type: ignore[arg-type]
            if profiler_cls is None:
                continue
            profiler = profiler_cls(config=self.config)
            try:
                batch = profiler.profile(data, columns=cols)
                for col_name in batch.analysed_columns:
                    if col_name in result.columns:
                        result.columns[col_name].stats = batch.columns.get(col_name)
            except Exception:
                pass

        # ── 7. Target columns ────────────────────────────────────────────
        # TargetProfiler produces target-specific analysis stored in
        # result.targets.  cp.stats is NOT overwritten — step 6 already set it.
        if self.config.target_columns:
            for target in self.config.target_columns:
                if target not in data.columns:
                    continue
                target_result = TargetProfiler(
                    target_column=target,
                    config=self.config,
                ).profile(data)
                result.targets[target] = target_result

                # setdefault returns the existing ColumnProfile.
                cp = result.columns.setdefault(target, ColumnProfile(name=target))
                cp.is_target = True

        # ── 8. Correlation ───────────────────────────────────────────────
        if self.config.compute_correlation:
            # Resolve column lists by detected SemanticType (post-override).
            numeric_cols = [
                c
                for c in active_cols
                if result.columns.get(c)
                and result.columns[c].semantic_type == SemanticType.Numeric
            ]
            categorical_cols = [
                c
                for c in active_cols
                if result.columns.get(c)
                and result.columns[c].semantic_type == SemanticType.Categorical
            ]

            corr_profiler = CorrelationProfiler(
                numeric_columns=numeric_cols,
                categorical_columns=categorical_cols,
                config=self.config,
            )

            # 8a. Feature-feature matrices — computed ONCE, target-independent.
            feature_corr = corr_profiler.profile_features(
                data, numeric_cols
            )
            result.dataset.feature_correlation = feature_corr

            # 8b. Per-target analysis — matrices are NOT recomputed; each call
            #     shallow-copies feature_corr and appends target-specific fields.
            for target in self.config.target_columns:
                if target not in data.columns:
                    continue
                result.dataset.target_correlations[target] = (
                    corr_profiler.profile_target(
                        data, feature_corr, numeric_cols, categorical_cols, target
                    )
                )

        return result

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

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

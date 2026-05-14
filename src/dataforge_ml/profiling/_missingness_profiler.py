"""
MissingnessProfiler  –  Phase 1 extension: Missingness Profiling.

Eligibility model
-----------------
Effective-null detection is based on **dtype first**, with SemanticType
overrides acting only as suppressors, not as enablers:

sentinel-string detection  →  runs when dtype is Utf8/String
                                suppressed if override is Numeric / Datetime / Boolean
                                (those types cannot have meaningful sentinel strings)

Inf / NaN expansion        →  runs when dtype is Float32/Float64
                                never suppressed (Inf in a float column is always
                                effectively missing regardless of semantic label)

column_overrides is SPARSE — most columns will have no entry.
Absence of an override is not a signal; it means "trust the dtype".
"""

from __future__ import annotations


import polars as pl

from ._base import DatasetLevelProfiler
from .config import ProfileConfig, SemanticType
from ._missingness_config import (
    ColumnMissingnessProfile,
    MissingnessFlag,
    MissingnessProfileResult,
    MissingSeverity,
)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

_SEVERITY_MINOR = 0.01
_SEVERITY_MODERATE = 0.05
_SEVERITY_HIGH = 0.20

_MAR_CORRELATION_THRESHOLD = 0.60
_COL_DROP_THRESHOLD = 0.50

_SENTINEL_STRINGS = frozenset({"NA", "NAN", "NULL", "NONE", "?"})

# Overrides that suppress sentinel-string detection on a String column.
# If a column is String but the user says "this is Numeric", treating
# "NA" as a sentinel is correct — but if they say Categorical or Text,
# sentinel detection still makes sense and should run.
_SENTINEL_SUPPRESSING_SEMANTICS = frozenset(
    {
        SemanticType.Numeric,
        SemanticType.Datetime,
        SemanticType.Boolean,
        SemanticType.Identifier,
    }
)


def _sentinel_eligible(dtype: pl.DataType, override: SemanticType | None) -> bool:
    """True when sentinel-string detection should run for this column."""
    if dtype not in (pl.Utf8, pl.String):
        return False
    # Override present and it's a non-text semantic → suppress
    if override is not None and override in _SENTINEL_SUPPRESSING_SEMANTICS:
        return False
    return True


def _inf_eligible(dtype: pl.DataType) -> bool:
    """True when Inf/NaN expansion should run. Always dtype-driven, never suppressed."""
    return dtype in (pl.Float32, pl.Float64)


class MissingnessProfiler(DatasetLevelProfiler[MissingnessProfileResult]):
    """
    Missingness profiler for Polars DataFrames.

    Column scoping
    --------------
    Resolution priority (high → low):
      1. Explicit ``columns`` argument to ``profile()``.
      2. ``config.exclude_columns`` — always removed.
      3. All remaining DataFrame columns.
    """

    def __init__(self, config: ProfileConfig | None = None) -> None:
        super().__init__(config)
        self._config: ProfileConfig = config or ProfileConfig()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def profile(
        self,
        data: pl.DataFrame,
        columns: list[str] | None = None,
    ) -> MissingnessProfileResult:
        return self._run(data, columns)

    # ------------------------------------------------------------------
    # Scope resolution
    # ------------------------------------------------------------------

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def _run(self, df: pl.DataFrame, cols: list[str]) -> MissingnessProfileResult:
        result = MissingnessProfileResult()
        result.analysed_columns = cols
        n_rows = df.height

        if n_rows == 0 or not cols:
            return result

        overrides = self._config.column_overrides  # sparse — most keys absent
        indicator_cols: list[pl.Series] = []

        for col_name in cols:
            override = overrides.get(col_name)  # None for most columns
            col_profile, indicator = self._profile_column(
                series=df[col_name],
                col_name=col_name,
                n_rows=n_rows,
                override=override,
            )
            result.columns[col_name] = col_profile
            indicator_cols.append(indicator)

            ratio = col_profile.effective_null_ratio
            if ratio == 1.0:
                result.fully_null_columns.append(col_name)
                col_profile.flags.append(MissingnessFlag.FullyNull)
            elif ratio > _COL_DROP_THRESHOLD:
                col_profile.flags.append(MissingnessFlag.DropCandidate)

        # ── Missingness correlation matrix ────────────────────────────
        cols_with_missing = [
            c for c in cols if result.columns[c].effective_null_count > 0
        ]
        if len(cols_with_missing) >= 2:
            indicator_frame = pl.DataFrame(
                {s.name: s for s in indicator_cols if s.name in cols_with_missing}
            )
            corr_matrix = self._compute_correlation_matrix(
                indicator_frame, cols_with_missing
            )
            result.correlation_matrix = corr_matrix

            for col_a in cols_with_missing:
                mar_peers = [
                    col_b
                    for col_b, r in corr_matrix.get(col_a, {}).items()
                    if col_b != col_a and r > _MAR_CORRELATION_THRESHOLD
                ]
                if mar_peers:
                    result.columns[col_a].correlated_with = mar_peers
                    if MissingnessFlag.MARSuspect not in result.columns[col_a].flags:
                        result.columns[col_a].flags.append(MissingnessFlag.MARSuspect)

        return result

    # ------------------------------------------------------------------
    # Per-column profiling
    # ------------------------------------------------------------------

    @staticmethod
    def _profile_column(
        series: pl.Series,
        col_name: str,
        n_rows: int,
        override: SemanticType | None = None,  # sparse — None is the common case
    ) -> tuple[ColumnMissingnessProfile, pl.Series]:
        """
        Compute standard + effective null counts for one column.

        Eligibility is dtype-first:
        - sentinel strings  → String dtype, unless override suppresses it
        - Inf/NaN           → Float dtype, always (never suppressed)
        - everything else   → standard Polars null only
        """
        profile = ColumnMissingnessProfile(column=col_name, total_rows=n_rows)
        dtype = series.dtype
        std_null = series.is_null()

        if _sentinel_eligible(dtype, override):
            eff_null = (
                std_null
                | (series.str.strip_chars() == "")
                | series.str.to_uppercase().is_in(list(_SENTINEL_STRINGS))
            )
        elif _inf_eligible(dtype):
            eff_null = std_null | series.is_nan() | series.is_infinite()
        else:
            eff_null = std_null

        std_count = int(std_null.sum())
        eff_count = int(eff_null.sum())

        profile.standard_null_count = std_count
        profile.effective_null_count = eff_count
        profile.standard_null_ratio = std_count / n_rows if n_rows else 0.0
        profile.effective_null_ratio = eff_count / n_rows if n_rows else 0.0

        r = profile.effective_null_ratio
        if r < _SEVERITY_MINOR:
            profile.severity = MissingSeverity.Minor
        elif r < _SEVERITY_MODERATE:
            profile.severity = MissingSeverity.Moderate
        elif r < _SEVERITY_HIGH:
            profile.severity = MissingSeverity.High
        else:
            profile.severity = MissingSeverity.Severe

        indicator = eff_null.cast(pl.Int8).rename(col_name)
        return profile, indicator

    # ------------------------------------------------------------------
    # Correlation matrix
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_correlation_matrix(
        indicator_frame: pl.DataFrame,
        cols: list[str],
    ) -> dict[str, dict[str, float]]:
        import itertools

        matrix: dict[str, dict[str, float]] = {c: {c: 1.0} for c in cols}
        if len(cols) < 2:
            return matrix

        pairs = list(itertools.combinations(cols, 2))
        exprs = [
            pl.corr(col_a, col_b, method="pearson")
            .fill_nan(0.0)
            .fill_null(0.0)
            .alias(f"{col_a}|{col_b}")
            for col_a, col_b in pairs
        ]
        result_row = indicator_frame.select(exprs).to_dicts()[0]

        for (col_a, col_b), r_value in zip(pairs, result_row.values()):
            r = max(-1.0, min(1.0, float(r_value)))
            matrix[col_a][col_b] = r
            matrix[col_b][col_a] = r

        return matrix

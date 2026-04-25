"""
MissingnessProfiler  –  Phase 1 extension: Missingness Profiling.

Always executed as part of StructuralProfiler (not opt-in).

Per-column metrics
------------------
  1. Standard null count & ratio    (Polars-level nulls)
  2. Effective null count & ratio   (+ whitespace-only + sentinel strings)
  3. Severity tier                  (minor / moderate / high / severe)
  4. FullyNull flag                 (ratio == 1.0 → must drop)

Cross-column metrics
--------------------
  5. Missingness correlation matrix
     Pairwise Pearson correlations between binary missingness indicator
     vectors (1=missing, 0=present).  High correlation (> 0.6) between
     two columns suggests Missing-At-Random (MAR) — they go missing
     together, likely due to a shared upstream cause.

Row-wise metrics
----------------
  6. Row-wise missingness distribution
     Buckets: 0 / 1–2 / 3–5 / >5 missing values per row.
     Rows with > 50 % of scoped columns missing are flagged as drop
     candidates before imputation.

Integration
-----------
MissingnessProfiler is wired directly into StructuralProfiler and does not
require any extra ProfileConfig keys — it operates on the same
``analysed_columns`` scope used by TabularProfiler.

    result.missingness = MissingnessProfiler(config=cfg).profile(df, analysed_cols)
"""

from __future__ import annotations

import math
from typing import Any

import polars as pl

from ..models._data_structure import DataStructure
from ._base import Profiling
from .config import ProfileConfig
from ._missingness_config import (
    ColumnMissingnessProfile,
    MissingnessFlag,
    MissingnessProfileResult,
    MissingSeverity,
    RowMissingnessDistribution,
)

# ---------------------------------------------------------------------------
# Thresholds
# ---------------------------------------------------------------------------

_SEVERITY_MINOR = 0.01  # effective_null_ratio <  1 %  → minor
_SEVERITY_MODERATE = 0.05  # effective_null_ratio <  5 %  → moderate
_SEVERITY_HIGH = 0.20  # effective_null_ratio < 20 %  → high
#                             effective_null_ratio >= 20 %  → severe

_MAR_CORRELATION_THRESHOLD = 0.60  # Pearson r > this → MAR suspect
_ROW_DROP_THRESHOLD = 0.50  # > 50 % of scoped cols missing → drop candidate

# String sentinels treated as effective nulls (upper-cased comparison)
_SENTINEL_STRINGS = {"NA", "NAN", "NULL", "NONE", "?"}


class MissingnessProfiler(Profiling[MissingnessProfileResult]):
    """
    Missingness profiler for Polars DataFrames.

    Parameters
    ----------
    config : ProfileConfig | None
        Shared profiling configuration.  ``columns`` scope is passed in
        explicitly at call time (sourced from TabularProfiler's resolved
        columns) so this class is decoupled from config scoping logic.
    """

    def __init__(self, config: ProfileConfig | None = None) -> None:
        super().__init__(DataStructure.Tabular, config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def profile(
        self,
        data: Any,
        columns: list[str] | None = None,
    ) -> MissingnessProfileResult:
        """
        Analyse missingness in *data*.

        Parameters
        ----------
        data : pl.DataFrame
            The frame to analyse.
        columns : list[str] | None
            Columns to include.  If None, all columns are used.
        """
        if not isinstance(data, pl.DataFrame):
            raise TypeError(
                f"MissingnessProfiler expects a Polars DataFrame, "
                f"got {type(data).__name__}."
            )
        cols = self._resolve_columns(data.columns, columns)
        return self._run(data, cols)

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def _run(self, df: pl.DataFrame, cols: list[str]) -> MissingnessProfileResult:
        result = MissingnessProfileResult()
        result.analysed_columns = cols
        n_rows = df.height

        if n_rows == 0 or not cols:
            return result

        # ── 1–4. Per-column pass ─────────────────────────────────────────
        #   Build a boolean "missing indicator" frame at the same time so
        #   we only iterate the data once per column.
        indicator_cols: list[pl.Series] = []  # 1=missing, 0=present  (Int8)

        for col_name in cols:
            series = df[col_name]
            col_profile, indicator = self._profile_column(series, col_name, n_rows)
            result.columns[col_name] = col_profile
            indicator_cols.append(indicator)

            if col_profile.effective_null_ratio == 1.0:
                result.fully_null_columns.append(col_name)
                col_profile.flags.append(MissingnessFlag.FullyNull)

        # ── 5. Missingness correlation matrix ────────────────────────────
        # Only meaningful when ≥ 2 columns have at least one missing value.
        cols_with_any_missing = [
            c for c in cols if result.columns[c].effective_null_count > 0
        ]
        if len(cols_with_any_missing) >= 2:
            indicator_frame = pl.DataFrame(
                {s.name: s for s in indicator_cols if s.name in cols_with_any_missing}
            )
            corr_matrix = self._compute_correlation_matrix(
                indicator_frame, cols_with_any_missing
            )
            result.correlation_matrix = corr_matrix

            # Annotate per-column profiles with MAR suspects
            for col_a in cols_with_any_missing:
                mar_peers: list[str] = []
                for col_b, r in corr_matrix.get(col_a, {}).items():
                    if col_b != col_a and r > _MAR_CORRELATION_THRESHOLD:
                        mar_peers.append(col_b)
                if mar_peers:
                    result.columns[col_a].correlated_with = mar_peers
                    if MissingnessFlag.MARSuspect not in result.columns[col_a].flags:
                        result.columns[col_a].flags.append(MissingnessFlag.MARSuspect)

        # ── 6. Row-wise distribution ──────────────────────────────────────
        result.row_distribution = self._compute_row_distribution(df, cols, n_rows)

        return result

    # ------------------------------------------------------------------
    # Per-column profiling
    # ------------------------------------------------------------------

    @staticmethod
    def _profile_column(
        series: pl.Series,
        col_name: str,
        n_rows: int,
    ) -> tuple[ColumnMissingnessProfile, pl.Series]:
        """
        Compute standard + effective null counts for one column.

        Returns
        -------
        profile : ColumnMissingnessProfile
        indicator : pl.Series  (Int8, 1=missing 0=present, named=col_name)
        """
        profile = ColumnMissingnessProfile(column=col_name, total_rows=n_rows)

        # ── Standard nulls (Polars-level) ────────────────────────────────
        std_null_expr = series.is_null()

        # ── Effective nulls ──────────────────────────────────────────────
        dtype = series.dtype
        if dtype in (pl.Utf8, pl.String):
            eff_null_expr = (
                std_null_expr
                | (series.str.strip_chars() == "")
                | series.str.to_uppercase().is_in(list(_SENTINEL_STRINGS))
            )
        elif dtype in (pl.Float32, pl.Float64):
            eff_null_expr = std_null_expr | series.is_nan() | series.is_infinite()
        else:
            eff_null_expr = std_null_expr

        std_count = int(std_null_expr.sum())
        eff_count = int(eff_null_expr.sum())

        profile.standard_null_count = std_count
        profile.effective_null_count = eff_count
        profile.standard_null_ratio = std_count / n_rows if n_rows else 0.0
        profile.effective_null_ratio = eff_count / n_rows if n_rows else 0.0

        # Severity
        r = profile.effective_null_ratio
        if r < _SEVERITY_MINOR:
            profile.severity = MissingSeverity.Minor
        elif r < _SEVERITY_MODERATE:
            profile.severity = MissingSeverity.Moderate
        elif r < _SEVERITY_HIGH:
            profile.severity = MissingSeverity.High
        else:
            profile.severity = MissingSeverity.Severe

        # Binary indicator: 1 = effectively missing, 0 = present
        indicator = eff_null_expr.cast(pl.Int8).rename(col_name)

        return profile, indicator

    # ------------------------------------------------------------------
    # Correlation matrix
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_correlation_matrix(
        indicator_frame: pl.DataFrame,
        cols: list[str],
    ) -> dict[str, dict[str, float]]:
        """
        Compute the full Pearson correlation matrix between binary
        missingness indicator columns using Polars' parallelised query engine.
        """
        import itertools

        # Initialize the matrix with 1.0 on the diagonal
        matrix: dict[str, dict[str, float]] = {c: {c: 1.0} for c in cols}

        # If there are fewer than 2 columns, no pairs to correlate
        if len(cols) < 2:
            return matrix

        # 1. Build all pairwise correlation expressions
        exprs = []
        pairs = list(itertools.combinations(cols, 2))

        for col_a, col_b in pairs:
            # Using Polars' built-in pearson correlation
            expr = (
                pl.corr(col_a, col_b, method="pearson")
                .fill_nan(0.0)  # Handles 0/0 division from constant columns
                .fill_null(0.0)  # Handles pure null columns
                .alias(f"{col_a}|{col_b}")
            )
            exprs.append(expr)

        # 2. Execute ALL correlations in a single multithreaded pass
        result_row = indicator_frame.select(exprs).to_dicts()[0]

        # 3. Unpack the results into the symmetric matrix
        for (col_a, col_b), r_value in zip(pairs, result_row.values()):
            # Clamp to [-1.0, 1.0] against floating-point drift
            r = max(-1.0, min(1.0, float(r_value)))

            matrix[col_a][col_b] = r
            matrix[col_b][col_a] = r  # symmetry

        return matrix

    # ------------------------------------------------------------------
    # Row-wise distribution
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_row_distribution(
        df: pl.DataFrame,
        cols: list[str],
        n_rows: int,
    ) -> RowMissingnessDistribution:
        """
        For each row count how many of *cols* are effectively null,
        then bucket into the four tiers and compute the drop-candidate %.

        We use Polars' lazy evaluation to build a per-row missing-count
        series without materialising a full boolean frame in Python memory.
        """
        dist = RowMissingnessDistribution()

        if n_rows == 0 or not cols:
            return dist

        n_cols = len(cols)

        # Build per-column effective-null boolean expressions
        per_col_exprs: list[pl.Expr] = []
        for col_name in cols:
            dtype = df[col_name].dtype
            null_expr = pl.col(col_name).is_null()

            if dtype in (pl.Utf8, pl.String):
                eff = (
                    null_expr
                    | (pl.col(col_name).str.strip_chars() == "")
                    | pl.col(col_name).str.to_uppercase().is_in(list(_SENTINEL_STRINGS))
                )
            elif dtype in (pl.Float32, pl.Float64):
                eff = (
                    null_expr
                    | pl.col(col_name).is_nan()
                    | pl.col(col_name).is_infinite()
                )
            else:
                eff = null_expr

            per_col_exprs.append(eff.cast(pl.Int8).alias(col_name))

        # Row-wise sum → per-row missing count
        row_missing_counts: pl.Series = df.select(per_col_exprs).select(
            pl.sum_horizontal(pl.all()).alias("row_missing")
        )["row_missing"]

        zero_mask = row_missing_counts == 0
        one_two_mask = (row_missing_counts >= 1) & (row_missing_counts <= 2)
        three_five_mask = (row_missing_counts >= 3) & (row_missing_counts <= 5)
        over_five_mask = row_missing_counts > 5

        # > 50 % of scoped columns missing
        half_threshold = math.ceil(n_cols * _ROW_DROP_THRESHOLD)
        over_half_mask = row_missing_counts >= half_threshold

        dist.pct_zero_missing = float(zero_mask.sum()) / n_rows
        dist.pct_one_to_two = float(one_two_mask.sum()) / n_rows
        dist.pct_three_to_five = float(three_five_mask.sum()) / n_rows
        dist.pct_over_five = float(over_five_mask.sum()) / n_rows
        dist.drop_candidate_row_count = int(over_half_mask.sum())
        dist.pct_over_half_missing = dist.drop_candidate_row_count / n_rows

        return dist

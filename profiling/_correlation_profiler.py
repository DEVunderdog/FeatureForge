"""
CorrelationProfiler  –  Phase 1 extension: Correlation & Information Structure.

Computes
--------
  1. Pearson correlation matrix          (numeric columns, linear relationships)
  2. Spearman correlation matrix         (numeric columns, monotonic / rank-based)
  3. Near-redundancy flagging            (|r| > 0.95 for either method)
  4. Near-redundancy groups              (union-find clusters → suggested drops)
  5. Feature–target correlation
       • Numeric target   → Pearson r per feature
       • Categorical target → ANOVA F-statistic + eta² per feature
  6. Mutual information                  (MI vs target; classif or regression)

Only numeric Polars dtypes participate in the correlation matrices.
Categorical columns participate in MI and ANOVA steps when a target is given.

Integration
-----------
Wire into StructuralProfiler::

    if self.config.correlation_target_column is not None or self.config.compute_correlation:
        result.correlation = CorrelationProfiler(
            numeric_columns=result.tabular.analysed_columns,   # pre-filtered
            categorical_columns=self.config.categorical_columns or [],
            target_column=self.config.correlation_target_column,
            config=self.config,
        ).profile(data)

Design decisions
----------------
- Spearman is implemented by rank-transforming each column and then
  computing Pearson on the ranks — avoids a scipy dependency in the
  hot path and parallelises naturally inside Polars.
- ANOVA groups are built column-by-column so memory stays bounded even
  for high-cardinality categoricals.
- MI uses sklearn's mutual_info_classif / mutual_info_regression with
  ``n_neighbors=3`` (default); the caller can tune via config.
- Union-find clusters are built from the near-redundant pair list so
  we correctly handle transitive redundancy (A≈B, B≈C → {A,B,C}).
"""

from __future__ import annotations

import itertools
import warnings
from typing import Any, Optional

import polars as pl

from ._base import Profiling
from .config import ProfileConfig
from ._correlation_config import (
    CategoricalTargetCorrelation,
    CorrelationPair,
    CorrelationProfileResult,
    MutualInformationEntry,
    NearRedundancyGroup,
    NumericTargetCorrelation,
    TargetType,
)
from ..models._data_structure import DataStructure

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_NEAR_REDUNDANT_THRESHOLD: float = 0.95   # |r| above this → near-redundant
_TOP_N_FEATURE_TARGET: int = 10           # how many feature-target entries to keep
_MI_N_NEIGHBORS: int = 3                  # sklearn MI n_neighbors hyperparameter

# Polars integer dtypes (for rank transform)
_INT_DTYPES = {
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
}
_NUMERIC_DTYPES = _INT_DTYPES | {pl.Float32, pl.Float64}


# ---------------------------------------------------------------------------
# Union-Find (for near-redundancy clustering)
# ---------------------------------------------------------------------------

class _UnionFind:
    """Minimal union-find / disjoint-set for clustering column pairs."""

    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        if x not in self._parent:
            self._parent[x] = x
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]  # path compression
            x = self._parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra

    def groups(self) -> list[list[str]]:
        """Return all groups with >1 member, each sorted."""
        from collections import defaultdict
        buckets: dict[str, list[str]] = defaultdict(list)
        for x in self._parent:
            buckets[self.find(x)].append(x)
        return [sorted(members) for members in buckets.values() if len(members) > 1]


# ---------------------------------------------------------------------------
# Main profiler
# ---------------------------------------------------------------------------

class CorrelationProfiler(Profiling[CorrelationProfileResult]):
    """
    Correlation and information-structure profiler for Polars DataFrames.

    Parameters
    ----------
    numeric_columns : list[str]
        Numeric columns to include in pairwise matrices and feature-target
        correlation.  Will be intersected with the frame at runtime.
    categorical_columns : list[str]
        Categorical columns to include in MI and ANOVA steps.
        Ignored when no target is provided.
    target_column : str | None
        The supervised target column.  If None, only pairwise matrices
        (steps 1–4) are computed.
    config : ProfileConfig | None
        Shared profiling configuration.
    near_redundant_threshold : float
        |r| threshold above which a pair is flagged as near-redundant.
        Default 0.95.
    top_n_feature_target : int
        How many feature–target entries to keep in the result.  Default 10.
    """

    def __init__(
        self,
        numeric_columns: list[str],
        categorical_columns: Optional[list[str]] = None,
        target_column: Optional[str] = None,
        config: Optional[ProfileConfig] = None,
        near_redundant_threshold: float = _NEAR_REDUNDANT_THRESHOLD,
        top_n_feature_target: int = _TOP_N_FEATURE_TARGET,
    ) -> None:
        super().__init__(DataStructure.Tabular, config)
        self._numeric_columns     = numeric_columns
        self._categorical_columns = categorical_columns or []
        self._target_column       = target_column
        self._threshold           = near_redundant_threshold
        self._top_n               = top_n_feature_target

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def profile(self, data: Any) -> CorrelationProfileResult:
        if not isinstance(data, pl.DataFrame):
            raise TypeError(
                f"CorrelationProfiler expects a Polars DataFrame, "
                f"got {type(data).__name__}."
            )
        return self._run(data)

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def _run(self, df: pl.DataFrame) -> CorrelationProfileResult:
        result = CorrelationProfileResult()

        # ── Resolve numeric columns ──────────────────────────────────────
        all_cols = df.columns
        numeric_cols = [
            c for c in self._resolve_columns(all_cols, self._numeric_columns)
            if df[c].dtype in _NUMERIC_DTYPES
        ]
        result.analysed_numeric_columns = numeric_cols

        # ── 1–4. Pairwise correlation matrices ───────────────────────────
        if len(numeric_cols) >= 2:
            pearson_mat, spearman_mat = self._compute_matrices(df, numeric_cols)
            result.pearson_matrix  = pearson_mat
            result.spearman_matrix = spearman_mat

            pairs = self._build_pairs(numeric_cols, pearson_mat, spearman_mat)
            result.pairwise = pairs
            result.near_redundant_pairs = [p for p in pairs if p.near_redundant]
            result.near_redundancy_groups = self._build_redundancy_groups(
                result.near_redundant_pairs
            )

        # ── 5–6. Target-based analysis ───────────────────────────────────
        if self._target_column and self._target_column in all_cols:
            target_type = self._detect_target_type(df, self._target_column)
            result.target_column = self._target_column
            result.target_type   = target_type

            # Feature columns = numeric + categorical, excluding the target
            feature_numeric = [c for c in numeric_cols if c != self._target_column]
            cat_cols = [
                c for c in self._resolve_columns(all_cols, self._categorical_columns)
                if c != self._target_column
            ]

            if target_type == TargetType.Numeric:
                result.feature_target_numeric = self._feature_target_pearson(
                    df, feature_numeric, self._target_column
                )
                result.mutual_information = self._mutual_information(
                    df, feature_numeric, cat_cols, self._target_column, target_type
                )
            else:  # Categorical
                result.feature_target_categorical = self._feature_target_anova(
                    df, feature_numeric, self._target_column
                )
                result.mutual_information = self._mutual_information(
                    df, feature_numeric, cat_cols, self._target_column, target_type
                )

        return result

    # ------------------------------------------------------------------
    # Step 1–2: Pearson + Spearman matrices
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_matrices(
        df: pl.DataFrame,
        cols: list[str],
    ) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
        """
        Compute full Pearson and Spearman matrices in two Polars passes.

        Spearman is derived by rank-transforming each column (using
        ``rank(method='average')`` to handle ties) and computing Pearson
        on the rank frame — this is mathematically equivalent to the
        classic Spearman formula.
        """
        pearson_mat:  dict[str, dict[str, float]] = {c: {c: 1.0} for c in cols}
        spearman_mat: dict[str, dict[str, float]] = {c: {c: 1.0} for c in cols}

        pairs = list(itertools.combinations(cols, 2))
        if not pairs:
            return pearson_mat, spearman_mat

        # Cast all columns to Float64 once (avoids repeated casts in exprs)
        numeric_frame = df.select(
            [pl.col(c).cast(pl.Float64).alias(c) for c in cols]
        )

        # Rank frame for Spearman
        rank_frame = numeric_frame.select(
            [pl.col(c).rank(method="average").alias(c) for c in cols]
        )

        # Build all correlation expressions in a single .select() pass
        pearson_exprs = [
            pl.corr(col_a, col_b, method="pearson")
              .fill_nan(0.0)
              .fill_null(0.0)
              .alias(f"p|{col_a}|{col_b}")
            for col_a, col_b in pairs
        ]
        spearman_exprs = [
            pl.corr(col_a, col_b, method="pearson")   # pearson on ranks = spearman
              .fill_nan(0.0)
              .fill_null(0.0)
              .alias(f"s|{col_a}|{col_b}")
            for col_a, col_b in pairs
        ]

        p_row = numeric_frame.select(pearson_exprs).row(0)
        s_row = rank_frame.select(spearman_exprs).row(0)

        for i, (col_a, col_b) in enumerate(pairs):
            p_r = max(-1.0, min(1.0, float(p_row[i])))
            s_r = max(-1.0, min(1.0, float(s_row[i])))

            pearson_mat[col_a][col_b] = p_r
            pearson_mat[col_b][col_a] = p_r
            spearman_mat[col_a][col_b] = s_r
            spearman_mat[col_b][col_a] = s_r

        return pearson_mat, spearman_mat

    # ------------------------------------------------------------------
    # Step 3–4: Near-redundancy flagging + grouping
    # ------------------------------------------------------------------

    def _build_pairs(
        self,
        cols: list[str],
        pearson_mat:  dict[str, dict[str, float]],
        spearman_mat: dict[str, dict[str, float]],
    ) -> list[CorrelationPair]:
        pairs: list[CorrelationPair] = []
        for col_a, col_b in itertools.combinations(cols, 2):
            p_r = pearson_mat.get(col_a, {}).get(col_b)
            s_r = spearman_mat.get(col_a, {}).get(col_b)

            near_redundant = (
                (p_r is not None and abs(p_r) > self._threshold)
                or (s_r is not None and abs(s_r) > self._threshold)
            )
            pairs.append(
                CorrelationPair(
                    col_a=col_a,
                    col_b=col_b,
                    pearson_r=p_r,
                    spearman_r=s_r,
                    near_redundant=near_redundant,
                )
            )
        return pairs

    @staticmethod
    def _build_redundancy_groups(
        near_redundant_pairs: list[CorrelationPair],
    ) -> list[NearRedundancyGroup]:
        """
        Union-find clustering of near-redundant columns.

        Within each cluster the first column (alphabetically) is kept;
        all others are in suggested_drop.
        """
        uf = _UnionFind()
        for pair in near_redundant_pairs:
            uf.union(pair.col_a, pair.col_b)

        groups: list[NearRedundancyGroup] = []
        for members in uf.groups():
            groups.append(
                NearRedundancyGroup(
                    columns=members,
                    suggested_drop=members[1:],   # drop all but the first alphabetically
                )
            )
        return groups

    # ------------------------------------------------------------------
    # Step 5a: Feature–target Pearson (numeric target)
    # ------------------------------------------------------------------

    def _feature_target_pearson(
        self,
        df: pl.DataFrame,
        feature_cols: list[str],
        target_col: str,
    ) -> list[NumericTargetCorrelation]:
        """
        Compute Pearson r between each numeric feature and the numeric target.
        Returns the top-N by |r|.
        """
        if not feature_cols:
            return []

        target_series = df[target_col].cast(pl.Float64)

        # Build all feature-vs-target correlation expressions in one pass
        exprs = [
            pl.corr(feat, target_col, method="pearson")
            .fill_nan(0.0)
            .fill_null(0.0) 
            .alias(feat)
            for feat in feature_cols
        ]

        frame = df.select(
            [pl.col(c).cast(pl.Float64) for c in feature_cols]
            + [target_series.alias(target_col)]
        ).select(exprs)

        row = frame.row(0)
        entries = [
            NumericTargetCorrelation(feature=col, pearson_r=float(r))
            for col, r in zip(feature_cols, row)
        ]

        # Sort by |r| descending, keep top-N
        entries.sort(key=lambda e: abs(e.pearson_r or 0.0), reverse=True)
        return entries[: self._top_n]

    # ------------------------------------------------------------------
    # Step 5b: Feature–target ANOVA + eta² (categorical target)
    # ------------------------------------------------------------------

    def _feature_target_anova(
        self,
        df: pl.DataFrame,
        feature_cols: list[str],
        target_col: str,
    ) -> list[CategoricalTargetCorrelation]:
        """
        One-way ANOVA: group numeric features by target category.

        eta² = SS_between / SS_total  (effect size, 0–1).
        Returns the top-N by eta².
        """
        try:
            from scipy.stats import f_oneway  # type: ignore[import]
        except ImportError:
            warnings.warn(
                "scipy is required for ANOVA feature-target analysis. "
                "Install it with: pip install scipy",
                stacklevel=3,
            )
            return []

        if not feature_cols:
            return []

        target_series = df[target_col]
        categories    = target_series.drop_nulls().unique().to_list()

        entries: list[CategoricalTargetCorrelation] = []

        for feat in feature_cols:
            feat_series = df[feat].cast(pl.Float64)
            groups = [
                feat_series.filter(target_series == cat).drop_nulls().to_numpy()
                for cat in categories
            ]
            # Need at least 2 non-empty groups
            non_empty = [g for g in groups if len(g) > 0]
            if len(non_empty) < 2:
                entries.append(CategoricalTargetCorrelation(feature=feat))
                continue

            try:
                f_stat, p_val = f_oneway(*non_empty)
            except Exception:
                entries.append(CategoricalTargetCorrelation(feature=feat))
                continue

            # eta² = SS_between / SS_total
            grand_mean = feat_series.drop_nulls().mean() or 0.0
            ss_total = float(((feat_series.drop_nulls() - grand_mean) ** 2).sum() or 0.0)
            ss_between = sum(
                len(g) * (float(g.mean()) - float(grand_mean)) ** 2
                for g in non_empty
            )
            eta_sq = ss_between / ss_total if ss_total > 0 else 0.0

            entries.append(
                CategoricalTargetCorrelation(
                    feature=feat,
                    f_statistic=float(f_stat),
                    p_value=float(p_val),
                    eta_squared=eta_sq,
                )
            )

        entries.sort(key=lambda e: e.eta_squared or 0.0, reverse=True)
        return entries[: self._top_n]

    # ------------------------------------------------------------------
    # Step 6: Mutual Information
    # ------------------------------------------------------------------

    def _mutual_information(
        self,
        df: pl.DataFrame,
        numeric_feature_cols: list[str],
        categorical_feature_cols: list[str],
        target_col: str,
        target_type: TargetType,
    ) -> list[MutualInformationEntry]:
        """
        Compute MI between all features and the target using sklearn.

        Numeric features are passed as-is (as floats).
        Categorical features are label-encoded (integer codes) before MI
        computation — this is acceptable for MI since MI is invariant to
        monotone label assignments.

        Missing values are imputed with column median (numeric) or mode
        (categorical) to satisfy sklearn's ``no-NaN`` requirement.
        """
        try:
            from sklearn.feature_selection import (  # type: ignore[import]
                mutual_info_classif,
                mutual_info_regression,
            )
        except ImportError:
            warnings.warn(
                "scikit-learn is required for mutual information analysis. "
                "Install it with: pip install scikit-learn",
                stacklevel=3,
            )
            return []

        import numpy as np

        all_feature_cols = numeric_feature_cols + categorical_feature_cols
        if not all_feature_cols:
            return []

        # ── Build feature matrix X ──────────────────────────────────────
        X_cols: list[Any] = []
        discrete_mask: list[bool] = []   # True → discrete feature (for MI)

        for col in numeric_feature_cols:
            series = df[col].cast(pl.Float64)
            median  = series.drop_nulls().median() or 0.0
            filled  = series.fill_null(median).fill_nan(median)
            X_cols.append(filled.to_numpy())
            discrete_mask.append(df[col].dtype in _INT_DTYPES)

        for col in categorical_feature_cols:
            # Cast to string → encode as uint32 category codes
            str_series = df[col].cast(pl.Utf8, strict=False)
            encoded    = str_series.cast(pl.Categorical).to_physical()
            # Fill null codes with 0 (unknown category)
            filled = encoded.fill_null(0)
            X_cols.append(filled.cast(pl.Int64).to_numpy())
            discrete_mask.append(True)

        X = np.column_stack(X_cols).astype(float)

        # ── Build target y ───────────────────────────────────────────────
        if target_type == TargetType.Numeric:
            target_series = df[target_col].cast(pl.Float64)
            median = target_series.drop_nulls().median() or 0.0
            y = target_series.fill_null(median).fill_nan(median).to_numpy()
        else:
            target_str = df[target_col].cast(pl.Utf8, strict=False)
            y = target_str.cast(pl.Categorical).to_physical().fill_null(0).cast(pl.Int64).to_numpy()

        # ── Compute MI ───────────────────────────────────────────────────
        try:
            if target_type == TargetType.Numeric:
                mi_scores = mutual_info_regression(
                    X, y,
                    discrete_features=discrete_mask,
                    n_neighbors=_MI_N_NEIGHBORS,
                    random_state=42,
                )
            else:
                mi_scores = mutual_info_classif(
                    X, y,
                    discrete_features=discrete_mask,
                    n_neighbors=_MI_N_NEIGHBORS,
                    random_state=42,
                )
        except Exception as exc:
            warnings.warn(f"MI computation failed: {exc}", stacklevel=3)
            return []

        # ── Rank and return ──────────────────────────────────────────────
        entries = [
            MutualInformationEntry(feature=col, mi_score=float(score))
            for col, score in zip(all_feature_cols, mi_scores)
        ]
        entries.sort(key=lambda e: e.mi_score, reverse=True)
        for rank, entry in enumerate(entries, start=1):
            entry.rank = rank

        return entries

    # ------------------------------------------------------------------
    # Helper: detect target type
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_target_type(df: pl.DataFrame, target_col: str) -> TargetType:
        """
        Infer whether *target_col* is numeric or categorical.

        Polars numeric dtypes → Numeric.
        Anything else (Utf8, Boolean, Categorical, …) → Categorical.
        """
        return (
            TargetType.Numeric
            if df[target_col].dtype in _NUMERIC_DTYPES
            else TargetType.Categorical
        )
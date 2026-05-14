"""
CorrelationProfiler  –  Phase 1 extension: Correlation & Information Structure.

Two public methods
------------------
profile_features(df, numeric_cols, categorical_cols)
    Computes pairwise Pearson + Spearman matrices and near-redundancy
    groups.  Target-independent — run once per dataset.
    Returns a CorrelationProfileResult with all matrix fields populated
    and all target-specific fields empty.

profile_target(df, feature_result, numeric_cols, categorical_cols, target_col)
    Takes the already-computed feature_result (so matrices are NOT
    recomputed) and adds feature-target Pearson / ANOVA / MI for one
    specific target column.
    Returns a new CorrelationProfileResult that shares the same matrix
    data plus the target-specific fields.

StructuralProfiler calls them like this::

    feature_corr = profiler.profile_features(df, numeric_cols, cat_cols)
    result.dataset.feature_correlation = feature_corr

    for target in config.target_columns:
        target_corr = profiler.profile_target(
            df, feature_corr, numeric_cols, cat_cols, target
        )
        result.dataset.target_correlations[target] = target_corr

The legacy profile() method is preserved for backward compatibility —
it calls profile_features() when no target is given, or profile_target()
after an internal profile_features() call when a target is given.
"""

from __future__ import annotations

import copy
import itertools
import warnings
from typing import Optional

import polars as pl

from ._base import DatasetLevelProfiler
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
from ..models._data_types import _NUMERIC_DTYPES, _INT_DTYPES

_NEAR_REDUNDANT_THRESHOLD: float = 0.95
_TOP_N_FEATURE_TARGET: int = 10
_MI_N_NEIGHBORS: int = 3
_MI_MIN_ROWS: int = 10  # min complete-case rows for a meaningful k-NN MI estimate


# ---------------------------------------------------------------------------
# Union-Find (unchanged)
# ---------------------------------------------------------------------------


class _UnionFind:
    def __init__(self) -> None:
        self._parent: dict[str, str] = {}

    def find(self, x: str) -> str:
        if x not in self._parent:
            self._parent[x] = x
        while self._parent[x] != x:
            self._parent[x] = self._parent[self._parent[x]]
            x = self._parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self._parent[rb] = ra

    def groups(self) -> list[list[str]]:
        from collections import defaultdict

        buckets: dict[str, list[str]] = defaultdict(list)
        for x in self._parent:
            buckets[self.find(x)].append(x)
        return [sorted(m) for m in buckets.values() if len(m) > 1]


# ---------------------------------------------------------------------------
# Main profiler
# ---------------------------------------------------------------------------


class CorrelationProfiler(DatasetLevelProfiler[CorrelationProfileResult]):
    """
    Correlation and information-structure profiler.

    Parameters
    ----------
    numeric_columns : list[str]
        Columns to include in pairwise matrices.
    categorical_columns : list[str]
        Columns to include in MI / ANOVA target steps.
    target_column : str | None
        Used only by the legacy profile() path.
    config : ProfileConfig | None
    near_redundant_threshold : float
    top_n_feature_target : int
    """

    def __init__(
        self,
        numeric_columns: list[str],
        categorical_columns: Optional[list[str]] = None,
        config: Optional[ProfileConfig] = None,
        near_redundant_threshold: float = _NEAR_REDUNDANT_THRESHOLD,
        top_n_feature_target: int = _TOP_N_FEATURE_TARGET,
    ) -> None:
        super().__init__(config)
        self._numeric_columns = numeric_columns
        self._categorical_columns = categorical_columns or []
        self._threshold = near_redundant_threshold
        self._top_n = top_n_feature_target

    # ------------------------------------------------------------------
    # Concrete implementation of the abstract base method
    # ------------------------------------------------------------------

    def profile(self, data: pl.DataFrame) -> CorrelationProfileResult:  # type: ignore[override]
        return self.profile_features(data, self._numeric_columns)

    # ------------------------------------------------------------------
    # Primary API  (called by StructuralProfiler)
    # ------------------------------------------------------------------

    def profile_features(
        self,
        df: pl.DataFrame,
        numeric_cols: list[str],
    ) -> CorrelationProfileResult:
        """
        Compute pairwise feature-feature correlation matrices.

        Pearson + Spearman matrices and near-redundancy groups are filled.
        All target-specific fields are left at their defaults (empty lists /
        None).  Call profile_target() separately for each target column.
        """
        result = CorrelationProfileResult()

        resolved_numeric = [
            c
            for c in numeric_cols
            if c in df.columns and df[c].dtype in _NUMERIC_DTYPES
        ]
        result.analysed_numeric_columns = resolved_numeric

        if len(resolved_numeric) >= 2:
            pearson_mat, spearman_mat = self._compute_matrices(df, resolved_numeric)
            result.pearson_matrix = pearson_mat
            result.spearman_matrix = spearman_mat

            pairs = self._build_pairs(resolved_numeric, pearson_mat, spearman_mat)
            result.pairwise = pairs
            result.near_redundant_pairs = [p for p in pairs if p.near_redundant]
            result.near_redundancy_groups = self._build_redundancy_groups(
                result.near_redundant_pairs
            )

        return result

    def profile_target(
        self,
        df: pl.DataFrame,
        feature_result: CorrelationProfileResult,
        numeric_cols: list[str],
        categorical_cols: list[str],
        target_col: str,
    ) -> CorrelationProfileResult:
        """
        Extend an existing feature_result with target-specific analysis
        for one target column.

        The pairwise matrices are NOT recomputed — they are copied from
        feature_result so the returned object is fully self-contained
        (i.e. safe to store independently and serialise).

        Feature columns for target analysis exclude the target itself,
        so a target that also appears in numeric_cols / categorical_cols
        is automatically excluded from its own feature-target stats.
        """
        if target_col not in df.columns:
            raise ValueError(f"target_col {target_col!r} not found in DataFrame.")

        # Shallow-copy the result so matrix dicts are shared (not duplicated
        # in memory) but the top-level object is independent.
        result = copy.copy(feature_result)
        result.target_column = target_col
        result.target_type = self._detect_target_type(df, target_col)

        # Feature columns: exclude the target from both lists
        feature_numeric = [
            c
            for c in numeric_cols
            if c != target_col and c in df.columns and df[c].dtype in _NUMERIC_DTYPES
        ]
        feature_categorical = [
            c for c in categorical_cols if c != target_col and c in df.columns
        ]

        if result.target_type == TargetType.Numeric:
            result.feature_target_numeric = self._feature_target_pearson(
                df, feature_numeric, target_col
            )
        else:
            result.feature_target_categorical = self._feature_target_anova(
                df, feature_numeric, target_col
            )

        result.mutual_information = self._mutual_information(
            df, feature_numeric, feature_categorical, target_col, result.target_type
        )

        return result

    # ------------------------------------------------------------------
    # Step 1–2: Pearson + Spearman matrices (unchanged)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_matrices(
        df: pl.DataFrame,
        cols: list[str],
    ) -> tuple[dict[str, dict[str, float]], dict[str, dict[str, float]]]:
        pearson_mat: dict[str, dict[str, float]] = {c: {c: 1.0} for c in cols}
        spearman_mat: dict[str, dict[str, float]] = {c: {c: 1.0} for c in cols}

        pairs = list(itertools.combinations(cols, 2))
        if not pairs:
            return pearson_mat, spearman_mat

        numeric_frame = df.select([pl.col(c).cast(pl.Float64).alias(c) for c in cols])
        rank_frame = numeric_frame.select(
            [pl.col(c).rank(method="average").alias(c) for c in cols]
        )

        pearson_exprs = [
            pl.corr(col_a, col_b, method="pearson")
            .fill_nan(0.0)
            .fill_null(0.0)
            .alias(f"p|{col_a}|{col_b}")
            for col_a, col_b in pairs
        ]
        spearman_exprs = [
            pl.corr(col_a, col_b, method="pearson")
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
    # Step 3–4: Near-redundancy (unchanged)
    # ------------------------------------------------------------------

    def _build_pairs(
        self,
        cols: list[str],
        pearson_mat: dict[str, dict[str, float]],
        spearman_mat: dict[str, dict[str, float]],
    ) -> list[CorrelationPair]:
        pairs: list[CorrelationPair] = []
        for col_a, col_b in itertools.combinations(cols, 2):
            p_r = pearson_mat.get(col_a, {}).get(col_b)
            s_r = spearman_mat.get(col_a, {}).get(col_b)
            near_redundant = (p_r is not None and abs(p_r) > self._threshold) or (
                s_r is not None and abs(s_r) > self._threshold
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
        uf = _UnionFind()
        for pair in near_redundant_pairs:
            uf.union(pair.col_a, pair.col_b)
        return [
            NearRedundancyGroup(columns=members, suggested_drop=members[1:])
            for members in uf.groups()
        ]

    # ------------------------------------------------------------------
    # Step 5a: Feature–target Pearson (unchanged)
    # ------------------------------------------------------------------

    def _feature_target_pearson(
        self,
        df: pl.DataFrame,
        feature_cols: list[str],
        target_col: str,
    ) -> list[NumericTargetCorrelation]:
        if not feature_cols:
            return []

        target_series = df[target_col].cast(pl.Float64)
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
        entries.sort(key=lambda e: abs(e.pearson_r or 0.0), reverse=True)
        return entries[: self._top_n]

    # ------------------------------------------------------------------
    # Step 5b: Feature–target ANOVA (unchanged)
    # ------------------------------------------------------------------

    def _feature_target_anova(
        self,
        df: pl.DataFrame,
        feature_cols: list[str],
        target_col: str,
    ) -> list[CategoricalTargetCorrelation]:
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
        categories = target_series.drop_nulls().unique().to_list()
        entries: list[CategoricalTargetCorrelation] = []

        for feat in feature_cols:
            feat_series = df[feat].cast(pl.Float64)
            groups = [
                feat_series.filter(target_series == cat).drop_nulls().to_numpy()
                for cat in categories
            ]
            non_empty = [g for g in groups if len(g) > 0]
            if len(non_empty) < 2:
                entries.append(CategoricalTargetCorrelation(feature=feat))
                continue
            try:
                f_stat, p_val = f_oneway(*non_empty)
            except Exception:
                entries.append(CategoricalTargetCorrelation(feature=feat))
                continue

            grand_mean = feat_series.drop_nulls().mean() or 0.0
            ss_total = float(
                ((feat_series.drop_nulls() - grand_mean) ** 2).sum() or 0.0
            )
            ss_between = sum(
                len(g) * (float(g.mean()) - float(grand_mean)) ** 2 for g in non_empty
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
    # Step 6: Mutual Information (unchanged)
    # ------------------------------------------------------------------

    def _mutual_information(
        self,
        df: pl.DataFrame,
        numeric_feature_cols: list[str],
        categorical_feature_cols: list[str],
        target_col: str,
        target_type: TargetType,
    ) -> list[MutualInformationEntry]:
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

        if not numeric_feature_cols and not categorical_feature_cols:
            return []

        fn = (
            mutual_info_regression
            if target_type == TargetType.Numeric
            else mutual_info_classif
        )

        # Build the target array once; track its null positions separately so
        # each feature can use its own complete-case mask (rows null in either
        # the feature or the target are excluded for that feature only).
        if target_type == TargetType.Numeric:
            target_series = df[target_col].cast(pl.Float64)
            y_full = target_series.to_numpy()
            target_null = (target_series.is_null() | target_series.is_nan()).to_numpy()
        else:
            target_encoded = (
                df[target_col]
                .cast(pl.Utf8, strict=False)
                .cast(pl.Categorical)
                .to_physical()
                .cast(pl.Int64)
            )
            y_full = target_encoded.to_numpy()
            target_null = df[target_col].is_null().to_numpy()

        entries: list[MutualInformationEntry] = []

        for col in numeric_feature_cols:
            series = df[col].cast(pl.Float64)
            feat_null = (series.is_null() | series.is_nan()).to_numpy()
            valid = ~(feat_null | target_null)
            n_valid = int(valid.sum())
            if n_valid < _MI_MIN_ROWS:
                continue
            x = series.to_numpy()[valid].reshape(-1, 1)
            y = y_full[valid]
            try:
                score = float(
                    fn(
                        x,
                        y,
                        discrete_features=[df[col].dtype in _INT_DTYPES],
                        n_neighbors=_MI_N_NEIGHBORS,
                        random_state=42,
                    )[0]
                )
            except Exception as exc:
                warnings.warn(f"MI failed for '{col}': {exc}", stacklevel=3)
                continue
            entries.append(MutualInformationEntry(feature=col, mi_score=score))

        for col in categorical_feature_cols:
            feat_encoded = (
                df[col]
                .cast(pl.Utf8, strict=False)
                .cast(pl.Categorical)
                .to_physical()
                .cast(pl.Int64)
            )
            feat_null = df[col].is_null().to_numpy()
            valid = ~(feat_null | target_null)
            n_valid = int(valid.sum())
            if n_valid < _MI_MIN_ROWS:
                warnings.warn(
                    f"Skipping MI for '{col}': only {n_valid} complete rows "
                    f"(need {_MI_MIN_ROWS}).",
                    stacklevel=3,
                )
                continue
            x = feat_encoded.to_numpy()[valid].reshape(-1, 1).astype(float)
            y = y_full[valid]
            try:
                score = float(
                    fn(
                        x,
                        y,
                        discrete_features=[True],
                        n_neighbors=_MI_N_NEIGHBORS,
                        random_state=42,
                    )[0]
                )
            except Exception as exc:
                warnings.warn(f"MI failed for '{col}': {exc}", stacklevel=3)
                continue
            entries.append(MutualInformationEntry(feature=col, mi_score=score))

        entries.sort(key=lambda e: e.mi_score, reverse=True)
        for rank, entry in enumerate(entries, start=1):
            entry.rank = rank
        return entries

    # ------------------------------------------------------------------
    # Helper
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_target_type(df: pl.DataFrame, target_col: str) -> TargetType:
        return (
            TargetType.Numeric
            if df[target_col].dtype in _NUMERIC_DTYPES
            else TargetType.Categorical
        )

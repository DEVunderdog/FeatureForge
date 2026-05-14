"""
CategoricalProfiler  –  Phase 1 extension: Categorical Column Profiling.

Per-column metrics (opt-in via ProfileConfig.categorical_columns):
  1. Cardinality & unique ratio
  2. Ordinal vs nominal detection
  3. Top-5 value counts with percentages
  4. Rare category analysis          (<1 % frequency threshold)
  5. Whitespace-only value count
  6. Mixed-type flag                 (some values numeric, some not)
  7. Free-text / natural-language flag
        (avg word count >5 OR avg char length >50 OR avg token count >10)
  8. Imbalance metrics
        – class ratio  (max_freq / min_freq)
        – Shannon entropy
        – Gini impurity

Integration
-----------
Add `categorical_columns: list[str] | None` to ProfileConfig, then call::

    from profiling.categorical import CategoricalProfiler

    cat_profiler = CategoricalProfiler(
        columns=["status", "country", "product_type"],
        config=cfg,
    )
    cat_result = cat_profiler.profile(df)

The result is a CategoricalProfileResult; attach it to TabularProfileResult
however suits your downstream pipeline.
"""

from __future__ import annotations

import math

import polars as pl
from ._base import ColumnBatchProfiler
from ._categorical_config import (
    CategoricalProfileResult,
    CategoricalStats,
    TopValueEntry,
    CategoricalFlag,
    RareCategoryStats,
    ImbalanceMetrics,
)
from .config import (
    ProfileConfig,
    SemanticType,
)
from ..models._data_types import _CAT_DTYPES

# ---------------------------------------------------------------------------
# Module-level thresholds (documented so callers can see what drives flags)
# ---------------------------------------------------------------------------

_RARE_THRESHOLD_PCT: float = 0.01  # <1 % of rows → rare
_MIXED_TYPE_MIN_MINOR_PCT: float = 0.05
_MIXED_TYPE_Z_SCORE: float = 1.96

_NEAR_CONSTANT_THRESHOLD: float = 0.90


class CategoricalProfiler(ColumnBatchProfiler[CategoricalProfileResult]):
    """
    Categorical profiler for Polars DataFrames.

    Parameters
    ----------
    columns : list[str]
        Columns to profile. The profiler intersects this list with
        the DataFrame's actual columns at runtime.
    config : ProfileConfig | None
        Shared profiling configuration (used for chunk_size, etc.).

    Usage
    -----
    >>> profiler = CategoricalProfiler(
    ...     columns=["status", "country", "product_type"],
    ... )
    >>> result = profiler.profile(df)
    >>> print(result)
    """

    def __init__(
        self,
        config: ProfileConfig | None = None,
    ) -> None:
        super().__init__(config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def profile(
        self,
        data: pl.DataFrame,
        columns: list[str],
    ) -> CategoricalProfileResult:
        return self._run(data, columns)

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def _eligible(
        self,
        series: pl.Series,
    ) -> bool:
        override = self.config.column_overrides.get(series.name)
        if override == SemanticType.Categorical:
            return True

        if override is not None:
            return False

        return series.dtype in _CAT_DTYPES

    def _run(
        self,
        df: pl.DataFrame,
        columns: list[str],
    ) -> CategoricalProfileResult:
        result = CategoricalProfileResult()

        # Resolve columns against actual schema
        available = [
            c
            for c in self._resolve_columns(df.columns, columns)
            if self._eligible(df[c])
        ]
        result.analysed_columns = available

        n_rows = df.height

        for col_name in available:
            series = df[col_name]
            profile = self._profile_column(series, col_name, n_rows)
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
    ) -> CategoricalStats:
        profile = CategoricalStats()

        # Cast to String for uniform downstream treatment
        str_series = series.cast(pl.Utf8, strict=False)

        # 1. Cardinality
        self._compute_cardinality(str_series, profile, n_rows)

        # 3. Value distribution (top-5, rare categories, imbalance)
        #    Returns the value-count frame for reuse in later steps.
        self._compute_value_distribution(str_series, profile, n_rows)

        # 5. Mixed-type flag
        #    We already know from TypeDetector whether the column was numeric-
        #    coerced; here we detect columns that are *partly* numeric and
        #    partly not — a different (and more expensive) check.
        self._check_mixed_type(str_series, profile)

        return profile

    # ------------------------------------------------------------------
    # Step 1: Cardinality
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_cardinality(
        series: pl.Series,
        profile: CategoricalStats,
        n_rows: int,
    ) -> None:
        cardinality = series.drop_nulls().n_unique()
        profile.cardinality = cardinality
        profile.unique_ratio = cardinality / n_rows if n_rows > 0 else 0.0

    # ------------------------------------------------------------------
    # Step 2: Value distribution
    # ------------------------------------------------------------------

    def _compute_value_distribution(
        self,
        series: pl.Series,
        profile: CategoricalStats,
        n_rows: int,
    ) -> pl.DataFrame:
        """
        Build value-count frame, populate top-5, rare stats, and imbalance.
        Returns the full value-count DataFrame for possible reuse.
        """
        # Exclude nulls and whitespace-only values from distribution stats
        clean = series.filter(
            ~series.is_null()
            & (series.str.strip_chars() != "")
            & ~series.str.to_uppercase().is_in(["NA", "NAN", "NULL", "NONE", "?"])
        )

        if clean.len() == 0:
            return pl.DataFrame({"value": [], "count": []})

        vc = clean.value_counts(sort=True).rename(  # sorted descending by count
            {"count": "count"}
        )  # polars already names it "count"
        # Polars value_counts column name for the values is the series name
        value_col = series.name

        # --- Top-10 ---
        top10_rows = min(10, vc.height)
        profile.top_values = [
            TopValueEntry(
                value=vc[value_col][i],
                count=int(vc["count"][i]),
                percentage=int(vc["count"][i]) / n_rows if n_rows > 0 else 0.0,
            )
            for i in range(top10_rows)
        ]

        profile.mode_frequency = profile.top_values[0].percentage
        if profile.mode_frequency > _NEAR_CONSTANT_THRESHOLD:
            profile.flags.append(CategoricalFlag.NearConstant)

        # --- Rare category analysis ---
        rare_threshold_abs = max(1, math.floor(_RARE_THRESHOLD_PCT * n_rows))
        rare_mask = vc["count"] < rare_threshold_abs
        rare_rows = vc.filter(rare_mask)

        profile.rare_categories = RareCategoryStats(
            threshold_pct=_RARE_THRESHOLD_PCT,
            rare_category_count=rare_rows.height,
            total_rare_rows=(
                int(rare_rows["count"].sum()) if rare_rows.height > 0 else 0
            ),
        )
        profile.rare_categories.rare_row_percentage = (
            profile.rare_categories.total_rare_rows / n_rows if n_rows > 0 else 0.0
        )

        # --- Imbalance metrics ---
        # Class Ratio -> raw distribution
        # Entropy -> randomness / information content
        # Gini -> impurity / misclassification risk
        counts = vc["count"].cast(pl.Float64)
        total = float(counts.sum())
        if total > 0:
            probs = counts / total
            max_freq = float(probs.max())  # type: ignore[arg-type]
            min_freq = float(probs.min())  # type: ignore[arg-type]

            class_ratio = max_freq / min_freq if min_freq > 0 else float("inf")
            entropy = float(-(probs * probs.log(base=2)).fill_nan(0.0).sum())
            gini = float(1.0 - (probs**2).sum())

            profile.imbalance = ImbalanceMetrics(
                class_ratio=class_ratio,
                shannon_entropy=entropy,
                gini_impurity=gini,
            )

        return vc

    # ------------------------------------------------------------------
    # Step 5: Mixed-type flag
    # ------------------------------------------------------------------

    @staticmethod
    def _check_mixed_type(
        series: pl.Series,
        profile: CategoricalStats,
    ) -> None:
        """
        Flag if the column contains both numeric-looking and non-numeric-looking
        values.  We use a regex pre-filter so that the vast majority of
        clearly non-numeric strings are rejected cheaply, and we only
        apply the heavier float-cast check to ambiguous values.
        """

        non_null = series.drop_nulls()
        n_total = non_null.len()

        if n_total == 0:
            return

        numeric_cast = non_null.cast(pl.Float64, strict=False)

        n_numeric = n_total - numeric_cast.null_count()
        n_non_numeric = n_total - n_numeric

        if n_numeric == 0 or n_non_numeric == 0:
            return

        n_minority = min(n_numeric, n_non_numeric)
        p_minority = n_minority / n_total

        z = _MIXED_TYPE_Z_SCORE
        denominator = 1 + (z**2) / n_total
        center = p_minority + (z**2) / (2 * n_total)
        spread = z * math.sqrt(
            (p_minority * (1 - p_minority) + (z**2) / (4 * n_total)) / n_total
        )

        lower_bound = (center - spread) / denominator

        if lower_bound >= _MIXED_TYPE_MIN_MINOR_PCT:
            profile.flags.append(CategoricalFlag.MixedType)

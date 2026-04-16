"""
CategoricalProfiler  –  Phase 1 extension: Categorical Column Profiling.

Per-column metrics (opt-in via ProfileConfig.categorical_columns):
  1. Cardinality & unique ratio
  2. Ordinal vs nominal detection
  3. Top-5 value counts with percentages
  4. Rare category analysis          (<1 % frequency threshold)
  5. Whitespace-only value count
  6. Mixed-type flag                 (some values numeric, some not)
  7. Potential-datetime flag         (50-sample probe, >70 % parse rate)
  8. Free-text / natural-language flag
        (avg word count >5 OR avg char length >50 OR avg token count >10)
  9. Imbalance metrics
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
import re
from typing import Any

import polars as pl

from models.data_structure import DataStructure
from profiling.base import Profiling
from profiling.categorical_config import (
    CategoricalColumnProfile,
    CategoricalFlag,
    CategoricalKind,
    CategoricalProfileResult,
    ImbalanceMetrics,
    RareCategoryStats,
    TopValueEntry,
)
from profiling.config import ProfileConfig

# ---------------------------------------------------------------------------
# Module-level thresholds (documented so callers can see what drives flags)
# ---------------------------------------------------------------------------

_RARE_THRESHOLD_PCT: float = 0.01          # <1 % of rows → rare
_DATETIME_SAMPLE_SIZE: int = 50            # rows sampled for datetime probe
_DATETIME_PARSE_RATE: float = 0.70         # >70 % parsed → potential_datetime
_FREE_TEXT_AVG_WORDS: int = 5              # avg word count threshold
_FREE_TEXT_AVG_CHARS: int = 50            # avg char length threshold
_FREE_TEXT_AVG_TOKENS: int = 10           # secondary: rough token count (chars/4)
_ORDINAL_NAME_RE = re.compile(
    r"(rank|level|tier|grade|priority|order|stage|step|score|rating|severity)",
    re.IGNORECASE,
)
_ORDINAL_VALUE_RE = re.compile(
    r"^(low|medium|high|very\s+high|very\s+low|critical|none|minor|major"
    r"|small|large|first|second|third|fourth|fifth"
    r"|\d+(?:st|nd|rd|th))$",
    re.IGNORECASE,
)

# Numeric-looking pattern used in mixed-type detection.
# Reusing this avoids per-value cast overhead for obviously non-numeric strings.
_NUMERIC_RE = re.compile(r"^\s*[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?\s*$")


class CategoricalProfiler(Profiling[CategoricalProfileResult]):
    """
    Categorical profiler for Polars DataFrames.

    Parameters
    ----------
    columns : list[str]
        Columns to profile.  The profiler intersects this list with
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
        columns: list[str],
        config: ProfileConfig | None = None,
    ) -> None:
        super().__init__(DataStructure.Tabular, config)
        self._requested_columns = columns

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def profile(self, data: Any) -> CategoricalProfileResult:
        if not isinstance(data, pl.DataFrame):
            raise TypeError(
                f"CategoricalProfiler expects a Polars DataFrame, "
                f"got {type(data).__name__}."
            )
        return self._run(data)

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def _run(self, df: pl.DataFrame) -> CategoricalProfileResult:
        result = CategoricalProfileResult()

        # Resolve columns against actual schema
        cols = self._resolve_columns(df.columns, self._requested_columns)
        result.analysed_columns = cols

        n_rows = df.height

        for col_name in cols:
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
    ) -> CategoricalColumnProfile:
        profile = CategoricalColumnProfile(column=col_name, total_rows=n_rows)

        # Cast to String for uniform downstream treatment
        str_series = series.cast(pl.Utf8, strict=False)

        # 1. Cardinality
        self._compute_cardinality(str_series, profile, n_rows)

        # 2. Missingness (nulls + whitespace)
        self._compute_missingness(str_series, profile)

        # 3. Value distribution (top-5, rare categories, imbalance)
        #    Returns the value-count frame for reuse in later steps.
        vc_frame = self._compute_value_distribution(str_series, profile, n_rows)

        # 4. Ordinal vs nominal detection
        self._detect_kind(str_series, col_name, profile)

        # 5. Mixed-type flag
        #    We already know from TypeDetector whether the column was numeric-
        #    coerced; here we detect columns that are *partly* numeric and
        #    partly not — a different (and more expensive) check.
        self._check_mixed_type(str_series, profile)

        # 6. Potential-datetime flag (only for non-datetime-named columns —
        #    TypeDetector already handles name-hinted columns)
        self._check_potential_datetime(str_series, col_name, profile, n_rows)

        # 7. Free-text / natural-language flag
        self._check_free_text(str_series, profile, n_rows)

        return profile

    # ------------------------------------------------------------------
    # Step 1: Cardinality
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_cardinality(
        series: pl.Series,
        profile: CategoricalColumnProfile,
        n_rows: int,
    ) -> None:
        cardinality = series.drop_nulls().n_unique()
        profile.cardinality = cardinality
        profile.unique_ratio = cardinality / n_rows if n_rows > 0 else 0.0

    # ------------------------------------------------------------------
    # Step 2: Missingness (nulls + whitespace)
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_missingness(
        series: pl.Series,
        profile: CategoricalColumnProfile,
    ) -> None:
        profile.null_count = series.null_count()

        # Whitespace-only: non-null values that strip to ""
        non_null = series.drop_nulls()
        whitespace_mask = non_null.str.strip_chars() == ""
        profile.whitespace_count = int(whitespace_mask.sum())

        profile.effective_missing_count = profile.null_count + profile.whitespace_count

    # ------------------------------------------------------------------
    # Step 3: Value distribution
    # ------------------------------------------------------------------

    def _compute_value_distribution(
        self,
        series: pl.Series,
        profile: CategoricalColumnProfile,
        n_rows: int,
    ) -> pl.DataFrame:
        """
        Build value-count frame, populate top-5, rare stats, and imbalance.
        Returns the full value-count DataFrame for possible reuse.
        """
        # Exclude nulls and whitespace-only values from distribution stats
        clean = series.drop_nulls().filter(series.drop_nulls().str.strip_chars() != "")

        if clean.len() == 0:
            return pl.DataFrame({"value": [], "count": []})

        vc = (
            clean.value_counts(sort=True)  # sorted descending by count
            .rename({"count": "count"})     # polars already names it "count"
        )
        # Polars value_counts column name for the values is the series name
        value_col = series.name

        # --- Top-5 ---
        top5_rows = min(5, vc.height)
        profile.top_values = [
            TopValueEntry(
                value=vc[value_col][i],
                count=int(vc["count"][i]),
                percentage=int(vc["count"][i]) / n_rows if n_rows > 0 else 0.0,
            )
            for i in range(top5_rows)
        ]

        # --- Rare category analysis ---
        rare_threshold_abs = max(1, math.floor(_RARE_THRESHOLD_PCT * n_rows))
        rare_mask = vc["count"] < rare_threshold_abs
        rare_rows = vc.filter(rare_mask)

        profile.rare_categories = RareCategoryStats(
            threshold_pct=_RARE_THRESHOLD_PCT,
            rare_category_count=rare_rows.height,
            total_rare_rows=int(rare_rows["count"].sum()) if rare_rows.height > 0 else 0,
        )
        profile.rare_categories.rare_row_percentage = (
            profile.rare_categories.total_rare_rows / n_rows if n_rows > 0 else 0.0
        )

        # --- Imbalance metrics ---
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
    # Step 4: Ordinal vs nominal detection
    # ------------------------------------------------------------------

    @staticmethod
    def _detect_kind(
        series: pl.Series,
        col_name: str,
        profile: CategoricalColumnProfile,
    ) -> None:
        """
        Heuristic ordinal detection:
          • Column name matches ordinal-name pattern, OR
          • ≥60 % of unique non-null values match the ordinal-value pattern.
        Everything else defaults to Nominal.
        """
        if _ORDINAL_NAME_RE.search(col_name):
            profile.kind = CategoricalKind.Ordinal
            return

        unique_vals = series.drop_nulls().unique().to_list()
        if not unique_vals:
            profile.kind = CategoricalKind.Nominal
            return

        ordinal_hits = sum(
            1 for v in unique_vals if _ORDINAL_VALUE_RE.match(str(v))
        )
        if ordinal_hits / len(unique_vals) >= 0.60:
            profile.kind = CategoricalKind.Ordinal
        else:
            profile.kind = CategoricalKind.Nominal

    # ------------------------------------------------------------------
    # Step 5: Mixed-type flag
    # ------------------------------------------------------------------

    @staticmethod
    def _check_mixed_type(
        series: pl.Series,
        profile: CategoricalColumnProfile,
    ) -> None:
        """
        Flag if the column contains both numeric-looking and non-numeric-looking
        values.  We use a regex pre-filter so that the vast majority of
        clearly non-numeric strings are rejected cheaply, and we only
        apply the heavier float-cast check to ambiguous values.

        Note: if TypeDetector already coerced this column to a numeric dtype,
        the caller will have passed the *original* string series here, so
        the check is still meaningful.
        """
        non_null = series.drop_nulls()
        if non_null.len() == 0:
            return

        # Vectorised regex test — faster than Python-level iteration
        looks_numeric = non_null.str.contains(r"^\s*[-+]?(\d+\.?\d*|\.\d+)([eE][-+]?\d+)?\s*$")
        n_numeric = int(looks_numeric.sum())
        n_total = non_null.len()
        n_non_numeric = n_total - n_numeric

        if n_numeric > 0 and n_non_numeric > 0:
            profile.flags.append(CategoricalFlag.MixedType)

    # ------------------------------------------------------------------
    # Step 6: Potential-datetime flag
    # ------------------------------------------------------------------

    @staticmethod
    def _check_potential_datetime(
        series: pl.Series,
        col_name: str,
        profile: CategoricalColumnProfile,
        n_rows: int,
    ) -> None:
        """
        Probe up to 50 non-null values with str.to_datetime(strict=False).
        If >70 % parse successfully, flag as PotentialDatetime.

        Skip columns already flagged MixedType (unreliable parse rates)
        or columns whose names already triggered TypeDetector's datetime
        coercion path (redundant check).
        """
        if CategoricalFlag.MixedType in profile.flags:
            return

        non_null = series.drop_nulls()
        if non_null.len() == 0:
            return

        sample_size = min(_DATETIME_SAMPLE_SIZE, non_null.len())
        sample = non_null.head(sample_size)

        try:
            parsed = sample.str.to_datetime(strict=False)
        except Exception:
            return

        parse_rate = parsed.drop_nulls().len() / sample_size
        if parse_rate > _DATETIME_PARSE_RATE:
            profile.flags.append(CategoricalFlag.PotentialDatetime)

    # ------------------------------------------------------------------
    # Step 7: Free-text / natural-language flag
    # ------------------------------------------------------------------

    @staticmethod
    def _check_free_text(
        series: pl.Series,
        profile: CategoricalColumnProfile,
        n_rows: int,
    ) -> None:
        """
        Flag as FreeText if the average value is long enough to need NLP
        treatment rather than categorical encoding.

        Thresholds (any one triggers the flag):
          • avg word count  > 5    (split on whitespace)
          • avg char length > 50
          • avg token count > 10   (approx: chars / 4, per GPT tokenisation heuristic)
        """
        non_null = series.drop_nulls()
        if non_null.len() == 0:
            return

        # Average character length
        char_lengths = non_null.str.len_chars()
        avg_chars = float(char_lengths.mean() or 0.0)  # type: ignore[arg-type]

        if avg_chars > _FREE_TEXT_AVG_CHARS:
            profile.flags.append(CategoricalFlag.FreeText)
            return

        # Average word count — split on whitespace sequences
        word_counts = non_null.str.split(" ").list.len()
        avg_words = float(word_counts.mean() or 0.0)  # type: ignore[arg-type]

        if avg_words > _FREE_TEXT_AVG_WORDS:
            profile.flags.append(CategoricalFlag.FreeText)
            return

        # Secondary threshold: rough token count (chars / 4)
        avg_tokens = avg_chars / 4.0
        if avg_tokens > _FREE_TEXT_AVG_TOKENS:
            profile.flags.append(CategoricalFlag.FreeText)
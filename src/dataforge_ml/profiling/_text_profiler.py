"""
TextProfiler  –  Phase 1 extension: Free-Text Column Profiling.

Handles columns classified as SemanticType.Text (free-text string columns).
All computation is Polars-native — no external NLP libraries, no language
detection.

Per-column metrics
------------------
1.  avg_token_count    – mean whitespace-split token count across non-null rows
2.  median_token_count – median whitespace-split token count across non-null rows
3.  vocabulary_size    – count of distinct tokens across all non-null values
4.  char_length_min    – shortest non-null string (characters)
5.  char_length_max    – longest non-null string (characters)
6.  char_length_mean   – mean character length across non-null strings
7.  char_length_median – median character length across non-null strings
8.  empty_ratio        – fraction of total rows that are empty strings ("")
9.  whitespace_ratio   – fraction of total rows that are whitespace-only
                         (includes empty strings, since strip → "")

Definitions
-----------
- "token"         : any run of non-whitespace characters produced by
                    str.split_whitespace() semantics, i.e.
                    ``pl.col(c).str.split(" ")`` with empty-string elements
                    filtered out.  We use Polars ``str.count_matches`` on
                    ``r"\\S+"`` which counts exactly these tokens in a single
                    vectorised pass.
- "empty string"  : len == 0 after no stripping.
- "whitespace-only": len == 0 after str.strip_chars().
- Null values are excluded from all per-row metrics and from ratio
  denominators **except** empty_ratio / whitespace_ratio, which are
  computed over total row count (nulls contribute 0, not counted as empty).

Eligibility
-----------
A column is eligible when:
  - It has a SemanticType.Text override in ProfileConfig.column_overrides, OR
  - Its Polars dtype is pl.Utf8 (alias pl.String) and no other override is set.

Integration
-----------
Drop ``TextProfiler`` into the profiler loop in ``structural.py`` alongside
``NumericProfiler``, ``CategoricalProfiler``, ``DatetimeProfiler``, and
``BooleanProfiler``::

    sub_result = TextProfiler(config=self.config).profile(data, columns=active_cols)
    for col_name, col_stats in sub_result.columns.items():
        result.columns.setdefault(col_name, ColumnProfile(name=col_name)).stats = col_stats
"""

from __future__ import annotations

import polars as pl

from ._base import ColumnBatchProfiler
from .config import (
    ProfileConfig,
    TextStats,
    SemanticType,
)
from ._text_config import TextProfileResult

# Regex that counts non-whitespace token runs — used with str.count_matches.
_TOKEN_PATTERN: str = r"\S+"


class TextProfiler(ColumnBatchProfiler[TextProfileResult]):
    """
    Free-text column profiler for Polars DataFrames.

    A column is eligible when:
      - It has a ``SemanticType.Text`` override in
        ``ProfileConfig.column_overrides``, OR
      - Its Polars dtype is ``pl.Utf8`` / ``pl.String`` and no override is set.

    Non-eligible columns are silently skipped.

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
    ) -> TextProfileResult:
        return self._run(data, columns)

    # ------------------------------------------------------------------
    # Eligibility
    # ------------------------------------------------------------------

    def _eligible(self, series: pl.Series) -> bool:
        override = self.config.column_overrides.get(series.name)

        if override == SemanticType.Text:
            return True

        # Any other explicit override takes precedence
        if override is not None:
            return False

        # Native string dtype (pl.Utf8 is the canonical name; pl.String is
        # an alias in newer Polars — check both for cross-version safety)
        return series.dtype in (pl.Utf8, pl.String)

    # ------------------------------------------------------------------
    # Orchestration
    # ------------------------------------------------------------------

    def _run(
        self,
        df: pl.DataFrame,
        columns: list[str],
    ) -> TextProfileResult:
        result = TextProfileResult()

        available = [
            c
            for c in self._resolve_columns(df.columns, columns)
            if self._eligible(df[c])
        ]
        result.analysed_columns = available

        for col_name in available:
            result.columns[col_name] = self._profile_column(
                df[col_name], df.height
            )

        return result

    # ------------------------------------------------------------------
    # Per-column driver
    # ------------------------------------------------------------------

    def _profile_column(
        self,
        series: pl.Series,
        n_rows: int,
    ) -> TextStats:
        profile = TextStats()

        if n_rows == 0:
            return profile

        # ── 1. Empty / whitespace ratios (computed over ALL rows, nulls → 0) ──
        # null rows do not count as empty or whitespace-only.
        non_null_mask = series.is_not_null()
        empty_mask = non_null_mask & (series.str.len_chars() == 0)
        stripped = series.str.strip_chars()
        whitespace_mask = non_null_mask & (stripped.str.len_chars() == 0)

        profile.empty_ratio = float(empty_mask.sum()) / n_rows
        profile.whitespace_ratio = float(whitespace_mask.sum()) / n_rows

        # ── 2. Work on non-null values only from here on ─────────────────────
        non_null = series.drop_nulls()
        n_non_null = non_null.len()

        if n_non_null == 0:
            return profile

        # ── 3. Token counts (whitespace-split, Polars regex count) ────────────
        # str.count_matches counts non-overlapping matches of r"\S+",
        # which is exactly the set of whitespace-delimited tokens.
        token_counts: pl.Series = non_null.str.count_matches(_TOKEN_PATTERN)

        profile.avg_token_count = float(token_counts.mean())  # type: ignore[arg-type]
        profile.median_token_count = float(token_counts.median())  # type: ignore[arg-type]

        # Re-derive cleanly to avoid the chained reference issue above:
        exploded = non_null.str.split(" ").explode().drop_nulls()
        non_empty_tokens = exploded.filter(exploded != "")
        profile.vocabulary_size = non_empty_tokens.n_unique()

        # ── 5. Character-length distribution ─────────────────────────────────
        char_lengths: pl.Series = non_null.str.len_chars().cast(pl.Float64)

        profile.char_length_min = int(char_lengths.min())  # type: ignore[arg-type]
        profile.char_length_max = int(char_lengths.max())  # type: ignore[arg-type]
        profile.char_length_mean = float(char_lengths.mean())  # type: ignore[arg-type]
        profile.char_length_median = float(char_lengths.median())  # type: ignore[arg-type]

        return profile
"""
TypeDetector  –  selective data-type detection for Polars DataFrames.

Detection is opt-in: only columns listed in ProfileConfig.type_detection_columns
are examined.  The detector never mutates the original frame.

Detection pipeline (in order, applied per column):
  1. Numeric coercion   – object/Utf8 columns  →  try cast to Float64
  2. Datetime coercion  – object/Utf8 columns with date-like names/values
  3. Boolean candidate  – int {0,1} or string {"true","false","yes","no",…}
  4. Encoded category   – int with low cardinality (<15 unique values)
  5. Identifier column  – unique ratio > 99 %
  6. Sequential index   – integer column == range(0,n) or range(1,n+1)
  7. Numeric kind       – continuous vs discrete for confirmed numeric cols
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import polars as pl

from .config import ColumnTypeInfo, NumericKind, TypeFlag, SemanticType
from ..models._data_types import _INT_DTYPES, _NUMERIC_DTYPES

if TYPE_CHECKING:
    pass

# Threshold constants
_NUMERIC_COERCE_THRESHOLD = 0.95  # ≥95 % non-null after cast → reclassify
_DATETIME_COERCE_THRESHOLD = 0.80  # ≥80 % non-null after cast → reclassify
_ENCODED_CATEGORY_MAX_UNIQUE = 15  # int with fewer unique values → label-encoded
_ENCODED_CATEGORY_MAX_RATIO = 0.05
_IDENTIFIER_UNIQUE_RATIO = 0.99  # >99 % unique → identifier
_IDENTIFIER_MAX_MEDIAN_LENGTH = 40
_DISCRETE_NUNIQUE_THRESHOLD = 20  # numeric with <20 unique values → discrete

_FREE_TEXT_AVG_WORDS: int = 3
_FREE_TEXT_MEDIAN_CHARS: int = 20
_FREE_TEXT_P90_CHARS: int = 35
_FREE_TEXT_MIN_UNIQUE_RATIO: float = 0.40
_FREE_TEXT_HIGH_UNIQUE_WITH_SPACES: float = 0.70  # unique ratio above which multi-token strings → Text


# Common boolean string values (lowercased)
_BOOL_STRING_SET = {"true", "false", "yes", "no", "t", "f", "0", "1"}

class TypeDetector:
    """
    Run selective type-detection on a Polars DataFrame.

    Parameters
    ----------
    columns : list[str]
        The columns to inspect (already validated against the frame).
    """

    def __init__(self, columns: list[str]) -> None:
        self._columns = columns

    # ------------------------------------------------------------------
    # Public
    # ------------------------------------------------------------------

    def detect(self, df: pl.DataFrame) -> dict[str, ColumnTypeInfo]:
        """
        Return a mapping of column name → ColumnTypeInfo for every
        column in self._columns.
        """
        results: dict[str, ColumnTypeInfo] = {}
        n_rows = df.height

        for col_name in self._columns:
            series = df[col_name]
            original_dtype = str(series.dtype)
            info = ColumnTypeInfo(
                column=col_name,
                original_dtype=original_dtype,
                inferred_dtype=original_dtype,
            )
            working = series

            # 1 & 2: Coercion for string columns
            if series.dtype in (pl.Utf8, pl.String):
                coerced, flag = self._try_numeric_coerce(series, n_rows)
                if coerced is not None:
                    info.inferred_dtype = str(coerced.dtype)
                    info.flags.append(flag)  # type: ignore[arg-type]
                    working = coerced
                    self._check_coerced_encoded_category(working, info)
                else:
                    coerced_dt, flag_dt = self._try_datetime_coerce(
                        series, n_rows
                    )
                    if coerced_dt is not None:
                        info.inferred_dtype = str(coerced_dt.dtype)
                        info.flags.append(flag_dt)  # type: ignore[arg-type]
                        info.semantic_type = SemanticType.Datetime
                        results[col_name] = info
                        continue

            # 3: Boolean candidate
            self._check_boolean_candidate(working, info)
            if TypeFlag.BooleanCandidate in info.flags:
                info.semantic_type = SemanticType.Boolean
                results[col_name] = info
                continue

            # Native datetime types
            if working.dtype in (pl.Date, pl.Datetime, pl.Duration, pl.Time) or isinstance(
                working.dtype, pl.Datetime
            ):
                info.semantic_type = SemanticType.Datetime
                results[col_name] = info
                continue

            # 4–7: Numeric path
            if working.dtype in _NUMERIC_DTYPES:
                if working.dtype in _INT_DTYPES:
                    # EncodedCategory and IdentifierColumn are mutually exclusive:
                    # low-cardinality and near-unique cannot both be true.
                    # Check encoded category first; skip identifier if it matches.
                    self._check_encoded_category(working, info)
                    if TypeFlag.EncodedCategory not in info.flags:
                        self._check_identifier(working, info, n_rows)
                        if TypeFlag.IdentifierColumn in info.flags:
                            self._check_sequential_index(working, info, n_rows)

                if TypeFlag.EncodedCategory in info.flags:
                    info.semantic_type = SemanticType.Categorical
                elif TypeFlag.IdentifierColumn in info.flags:
                    info.semantic_type = SemanticType.Identifier
                else:
                    self._classify_numeric_kind(working, info)
                    info.semantic_type = SemanticType.Numeric

                results[col_name] = info
                continue

            # String path
            if working.dtype in (pl.Utf8, pl.String):
                self._check_free_text(working, info, n_rows)
                if TypeFlag.FreeTextCandidate in info.flags:
                    info.semantic_type = SemanticType.Text
                    results[col_name] = info
                    continue
                self._check_identifier(working, info, n_rows)
                info.semantic_type = (
                    SemanticType.Identifier
                    if TypeFlag.IdentifierColumn in info.flags
                    else SemanticType.Categorical
                )
                results[col_name] = info
                continue

            # Fallback
            info.semantic_type = SemanticType.Text
            results[col_name] = info

        return results

    # ------------------------------------------------------------------
    # Step 1: Numeric coercion
    # ------------------------------------------------------------------

    @staticmethod
    def _try_numeric_coerce(
        series: pl.Series, n_rows: int
    ) -> tuple[pl.Series, TypeFlag] | tuple[None, None]:
        """
        Attempt to cast a Utf8 series to Float64.
        Returns the cast series + flag if success rate ≥ threshold, else (None, None).
        """
        if n_rows == 0:
            return None, None
        try:
            cast = series.cast(pl.Float64, strict=False)
        except Exception:
            return None, None

        non_null = cast.drop_nulls().len()
        # Compare against original non-null count to avoid penalising
        # columns that were already sparse
        original_non_null = series.drop_nulls().len()
        denom = original_non_null if original_non_null > 0 else n_rows
        success_rate = non_null / denom
        if success_rate >= _NUMERIC_COERCE_THRESHOLD:
            return cast, TypeFlag.NumericCoerced
        return None, None

    # ------------------------------------------------------------------
    # Step 2: Datetime coercion
    # ------------------------------------------------------------------

    @staticmethod
    def _try_datetime_coerce(
        series: pl.Series, n_rows: int
    ) -> tuple[pl.Series, TypeFlag] | tuple[None, None]:
        """
        Attempt datetime coercion if the column name looks date-like.
        Returns the parsed series + flag if success rate ≥ threshold.
        """
        if n_rows == 0:
            return None, None

        try:
            cast = series.str.to_datetime(strict=False)
        except Exception:
            return None, None

        original_non_null = series.drop_nulls().len()
        denom = original_non_null if original_non_null > 0 else n_rows
        non_null = cast.drop_nulls().len()
        if denom > 0 and non_null / denom >= _DATETIME_COERCE_THRESHOLD:
            return cast, TypeFlag.DatetimeCoerced
        return None, None

    # ------------------------------------------------------------------
    # Step 3: Boolean candidate
    # ------------------------------------------------------------------

    @staticmethod
    def _check_boolean_candidate(series: pl.Series, info: ColumnTypeInfo) -> None:
        if series.dtype == pl.Boolean:
            info.flags.append(TypeFlag.BooleanCandidate)
            return

        if series.dtype in _INT_DTYPES:
            unique_vals = set(series.drop_nulls().unique().to_list())
            if unique_vals <= {0, 1}:
                info.flags.append(TypeFlag.BooleanCandidate)
        elif series.dtype in (pl.Utf8, pl.String):
            unique_vals_lower = {
                str(v).lower() for v in series.drop_nulls().unique().to_list()
            }
            if unique_vals_lower and unique_vals_lower <= _BOOL_STRING_SET:
                info.flags.append(TypeFlag.BooleanCandidate)

    # ------------------------------------------------------------------
    # Step 4: Encoded category
    # ------------------------------------------------------------------

    @staticmethod
    def _check_coerced_encoded_category(
        series: pl.Series, info: ColumnTypeInfo
    ) -> None:
        """
        Post-coercion low-cardinality check for Float64 series that originated
        as strings. Sets EncodedCategory only when:
        1. All non-null values are whole numbers (the strings were integer-like)
        2. Cardinality passes the same absolute + ratio thresholds as the
            native-integer encoded-category check.

        This distinguishes "1","2","3" (label-encoded → Categorical) from
        "1.5","2.7","3.1" (genuine floats → Numeric).
        """
        if TypeFlag.BooleanCandidate in info.flags:
            return

        valid = series.drop_nulls()
        n_valid = valid.len()
        if n_valid == 0:
            return

        # Whole-number check: reject true floats like 1.5, 2.7
        try:
            as_int = valid.cast(pl.Int64, strict=False)
        except Exception:
            return
        if not (valid == as_int.cast(pl.Float64, strict=False)).all():
            return

        # Cardinality thresholds (same logic as _check_encoded_category)
        n_unique = valid.n_unique()
        min_val = int(valid.min())
        max_val = int(valid.max())
        range_span = (max_val - min_val) + 1
        is_tight_sequence = range_span == n_unique
        absolute_limit = 50 if is_tight_sequence else _ENCODED_CATEGORY_MAX_UNIQUE
        absolute_ok = 0 < n_unique < absolute_limit
        ratio_ok = (n_unique / n_valid) < _ENCODED_CATEGORY_MAX_RATIO

        if (absolute_ok and ratio_ok) or (is_tight_sequence and absolute_ok):
            info.flags.append(TypeFlag.EncodedCategory)

    @staticmethod
    def _check_encoded_category(
        series: pl.Series, info: ColumnTypeInfo
    ) -> None:
        if TypeFlag.BooleanCandidate in info.flags:
            return

        if not series.dtype.is_integer():
            return

        valid_series = series.drop_nulls()
        n_valid = valid_series.len()

        if n_valid == 0:
            return

        n_unique = valid_series.n_unique()

        min_val = valid_series.min()
        max_val = valid_series.max()
        range_span = (max_val - min_val) + 1

        is_tight_sequence = range_span == n_unique

        absolute_limit = 50 if is_tight_sequence else _ENCODED_CATEGORY_MAX_UNIQUE

        absolute_ok = 0 < n_unique < absolute_limit
        ratio_ok = (n_unique / n_valid) < _ENCODED_CATEGORY_MAX_RATIO

        if (absolute_ok and ratio_ok) or (is_tight_sequence and absolute_ok):
            info.flags.append(TypeFlag.EncodedCategory)

    # ------------------------------------------------------------------
    # Step 5: Identifier column
    # ------------------------------------------------------------------

    @staticmethod
    def _check_identifier(series: pl.Series, info: ColumnTypeInfo, n_rows: int) -> None:
        if n_rows == 0:
            return

        n_unique = series.n_unique()
        if n_unique / n_rows <= _IDENTIFIER_UNIQUE_RATIO:
            return

        if series.dtype in (pl.Utf8, pl.String):
            non_null = series.drop_nulls()
            if non_null.len() == 0:
                return

            median_length = non_null.str.len_chars().median()
            if median_length is not None and median_length > _IDENTIFIER_MAX_MEDIAN_LENGTH:
                return

            # Real identifiers are single tokens — no spaces.
            # Sentences and descriptions have median_spaces > 0.
            if float(non_null.str.count_matches(r"\s+").median() or 0.0) > 0:
                return

        info.flags.append(TypeFlag.IdentifierColumn)

    # ------------------------------------------------------------------
    # Step 6: Sequential index
    # ------------------------------------------------------------------

    @staticmethod
    def _check_sequential_index(
        series: pl.Series, info: ColumnTypeInfo, n_rows: int
    ) -> None:
        if n_rows == 0 or TypeFlag.IdentifierColumn not in info.flags:
            # Only bother if already flagged as identifier
            return

        is_float = series.dtype in (pl.Float32, pl.Float64)
        is_int = series.dtype in _INT_DTYPES

        if not (is_float or is_int):
            return

        s_min = series.min()
        s_max = series.max()

        if (s_min != 0 and s_max != n_rows - 1) or (s_min != 1 or s_max != n_rows):
            return

        if is_float:
            series_int = series.cast(pl.Int64)
            if not (series == series_int).all():
                return
            series_to_check = series_int
        else:
            series_to_check = series

        if series_to_check.n_unique() == n_rows:
            flag = (
                TypeFlag.FloatSequentialIndex if is_float else TypeFlag.SequentialIndex
            )
            info.flags.append(flag)

    # ------------------------------------------------------------------
    # Step 7: Numeric kind
    # ------------------------------------------------------------------

    @staticmethod
    def _classify_numeric_kind(series: pl.Series, info: ColumnTypeInfo) -> None:
        # Skip if it's an encoded category (treat as categorical, not numeric)
        if TypeFlag.EncodedCategory in info.flags:
            return

        n_unique = series.drop_nulls().n_unique()

        if series.dtype in _INT_DTYPES:
            info.numeric_kind = NumericKind.Discrete
        elif n_unique < _DISCRETE_NUNIQUE_THRESHOLD:
            info.numeric_kind = NumericKind.Discrete
        else:
            info.numeric_kind = NumericKind.Continuous

    @staticmethod
    def _check_free_text(
        series: pl.Series,
        info: ColumnTypeInfo,
        n_rows: int,
    ) -> None:
        non_null = series.drop_nulls()
        if non_null.len() == 0:
            return

        char_lengths = non_null.str.len_chars()
        median_chars = float(char_lengths.median() or 0.0)
        space_counts = non_null.str.count_matches(r"\s+")
        median_spaces = float(space_counts.median() or 0.0)
        median_words = median_spaces + 1.0
        unique_ratio = series.n_unique() / n_rows if n_rows > 0 else 0.0

        # Multi-word strings of medium length: names, addresses, short descriptions
        if median_chars > _FREE_TEXT_MEDIAN_CHARS and median_spaces >= 1.0:
            info.flags.append(TypeFlag.FreeTextCandidate)
            return

        # Long average word count: sentences, paragraphs
        if median_words > _FREE_TEXT_AVG_WORDS:
            info.flags.append(TypeFlag.FreeTextCandidate)
            return

        p90_chars = float(char_lengths.quantile(0.9) or 0.0)
        if p90_chars > _FREE_TEXT_P90_CHARS and unique_ratio > _FREE_TEXT_MIN_UNIQUE_RATIO:
            info.flags.append(TypeFlag.FreeTextCandidate)
            return

        # High-cardinality multi-token strings that don't meet char thresholds:
        # e.g. short full names like "John Smith", compound tokens
        if unique_ratio >= _FREE_TEXT_HIGH_UNIQUE_WITH_SPACES and median_spaces >= 1.0:
            info.flags.append(TypeFlag.FreeTextCandidate)

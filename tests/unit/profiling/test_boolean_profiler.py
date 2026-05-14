import polars as pl

from dataforge_ml.profiling._boolean_profiler import BooleanProfiler
from dataforge_ml.profiling._boolean_config import BooleanProfileResult, BooleanStats


# ---------------------------------------------------------------------------
# Result type & analysed_columns
# ---------------------------------------------------------------------------


def test_result_type_and_analysed_columns():
    df = pl.DataFrame(
        {
            "flag": pl.Series([True, False, True], dtype=pl.Boolean),
            "score": pl.Series([1.0, 2.0, 3.0], dtype=pl.Float64),
        }
    )
    result = BooleanProfiler().profile(df, ["flag", "score"])
    assert isinstance(result, BooleanProfileResult)
    assert "flag" in result.analysed_columns
    assert "score" not in result.analysed_columns


# ---------------------------------------------------------------------------
# Counts
# ---------------------------------------------------------------------------


def test_true_false_count_sum_equals_non_null_count():
    values = [True, False, True, None, True, False, None]
    df = pl.DataFrame({"flag": pl.Series(values, dtype=pl.Boolean)})
    stats = BooleanProfiler().profile(df, ["flag"]).columns["flag"]
    non_null_count = df["flag"].drop_nulls().len()
    assert stats.true_count + stats.false_count == non_null_count


# ---------------------------------------------------------------------------
# Ratios
# ---------------------------------------------------------------------------


def test_true_ratio_plus_false_ratio_equals_one():
    values = [True, True, False, True, False, True]
    df = pl.DataFrame({"flag": pl.Series(values, dtype=pl.Boolean)})
    stats = BooleanProfiler().profile(df, ["flag"]).columns["flag"]
    assert abs(stats.true_ratio + stats.false_ratio - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# Mode
# ---------------------------------------------------------------------------


def test_tied_column_mode_is_none():
    values = [True] * 5 + [False] * 5
    df = pl.DataFrame({"flag": pl.Series(values, dtype=pl.Boolean)})
    stats = BooleanProfiler().profile(df, ["flag"]).columns["flag"]
    assert stats.mode is None


# ---------------------------------------------------------------------------
# Integer {0, 1} columns
# ---------------------------------------------------------------------------


def test_integer_01_eligible_with_correct_counts_and_ratios():
    values = [1, 0, 1, 1, 0, None]
    df = pl.DataFrame({"bin": pl.Series(values, dtype=pl.Int64)})
    result = BooleanProfiler().profile(df, ["bin"])
    assert "bin" in result.analysed_columns
    stats = result.columns["bin"]
    non_null = [v for v in values if v is not None]
    expected_true = sum(non_null)
    expected_false = len(non_null) - expected_true
    assert stats.true_count == expected_true
    assert stats.false_count == expected_false
    assert abs(stats.true_ratio + stats.false_ratio - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# All-null boolean column
# ---------------------------------------------------------------------------


def test_all_null_boolean_returns_default_stats_without_crashing():
    df = pl.DataFrame({"flag": pl.Series([None, None, None], dtype=pl.Boolean)})
    stats = BooleanProfiler().profile(df, ["flag"]).columns["flag"]
    assert isinstance(stats, BooleanStats)
    assert stats.true_count == 0
    assert stats.false_count == 0

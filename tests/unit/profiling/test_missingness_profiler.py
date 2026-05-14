import polars as pl

from dataforge_ml.profiling._missingness_profiler import MissingnessProfiler


# ---------------------------------------------------------------------------
# null_count equals actual null count in the series
# ---------------------------------------------------------------------------


def test_null_count_equals_actual_null_count():
    values = [1, None, 3, None, None, 6, 7, None]
    df = pl.DataFrame({"x": pl.Series(values, dtype=pl.Int64)})
    profile = MissingnessProfiler().profile(df, ["x"]).columns["x"]
    assert profile.standard_null_count == df["x"].null_count()


# ---------------------------------------------------------------------------
# null_ratio equals null_count / n_rows
# ---------------------------------------------------------------------------


def test_null_ratio_equals_null_count_over_n_rows():
    values = [10, None, 30, None, 50]
    df = pl.DataFrame({"x": pl.Series(values, dtype=pl.Int64)})
    profile = MissingnessProfiler().profile(df, ["x"]).columns["x"]
    expected_ratio = profile.standard_null_count / df.height
    assert abs(profile.standard_null_ratio - expected_ratio) < 1e-10


# ---------------------------------------------------------------------------
# All-null column produces null_ratio == 1.0 without crashing
# ---------------------------------------------------------------------------


def test_all_null_column_ratio_is_one():
    df = pl.DataFrame({"x": pl.Series([None, None, None, None], dtype=pl.Int64)})
    profile = MissingnessProfiler().profile(df, ["x"]).columns["x"]
    assert profile.effective_null_ratio == 1.0


# ---------------------------------------------------------------------------
# Fully populated column has null_count == 0 and null_ratio == 0.0
# ---------------------------------------------------------------------------


def test_fully_populated_column_has_zero_nulls():
    df = pl.DataFrame({"x": pl.Series([1, 2, 3, 4, 5], dtype=pl.Int64)})
    profile = MissingnessProfiler().profile(df, ["x"]).columns["x"]
    assert profile.standard_null_count == 0
    assert profile.standard_null_ratio == 0.0

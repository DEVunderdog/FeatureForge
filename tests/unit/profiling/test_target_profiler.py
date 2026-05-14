import polars as pl
import pytest

from dataforge_ml.profiling._target_profiler import TargetProfiler
from dataforge_ml.profiling._target_config import TargetProblemType


# ---------------------------------------------------------------------------
# Regression: continuous float target
# ---------------------------------------------------------------------------


def test_regression_for_continuous_float_target():
    df = pl.DataFrame(
        {"price": pl.Series([float(i) * 1.5 for i in range(50)], dtype=pl.Float64)}
    )
    result = TargetProfiler(target_column="price").profile(df)
    assert result.problem_type == TargetProblemType.Regression


# ---------------------------------------------------------------------------
# Classification: low-cardinality string target
# ---------------------------------------------------------------------------


def test_classification_for_low_cardinality_string_target():
    vals = ["cat", "dog", "bird"] * 20
    df = pl.DataFrame({"label": pl.Series(vals, dtype=pl.Utf8)})
    result = TargetProfiler(target_column="label").profile(df)
    assert result.problem_type in (
        TargetProblemType.BinaryClassification,
        TargetProblemType.MulticlassClassification,
    )


# ---------------------------------------------------------------------------
# Missing target column raises ValueError
# ---------------------------------------------------------------------------


def test_missing_target_column_raises_value_error():
    df = pl.DataFrame({"feature": pl.Series([1, 2, 3], dtype=pl.Int64)})
    with pytest.raises(ValueError):
        TargetProfiler(target_column="nonexistent").profile(df)

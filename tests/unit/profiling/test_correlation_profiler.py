import polars as pl

from dataforge_ml.profiling._correlation_profiler import CorrelationProfiler
from dataforge_ml.profiling._correlation_config import CorrelationProfileResult


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_df_three_cols():
    """Three numeric columns with moderate (non-trivial) correlations."""
    n = 50
    a = [float(i % 7) for i in range(n)]
    b = [float((i * 3) % 11) for i in range(n)]
    c = [float(i % 13) for i in range(n)]
    return pl.DataFrame(
        {
            "a": pl.Series(a, dtype=pl.Float64),
            "b": pl.Series(b, dtype=pl.Float64),
            "c": pl.Series(c, dtype=pl.Float64),
        }
    )


def _make_df_with_duplicate():
    """Two identical columns plus one independent column."""
    n = 40
    vals = [float(i) for i in range(n)]
    return pl.DataFrame(
        {
            "x": pl.Series(vals, dtype=pl.Float64),
            "x_copy": pl.Series(vals, dtype=pl.Float64),
            "y": pl.Series([v * 0.3 + 7.0 for v in vals], dtype=pl.Float64),
        }
    )


# ---------------------------------------------------------------------------
# Pearson matrix symmetry  (profile_features)
# ---------------------------------------------------------------------------


def test_pearson_matrix_is_symmetric():
    df = _make_df_three_cols()
    cols = ["a", "b", "c"]
    result = CorrelationProfiler(numeric_columns=cols).profile_features(df, cols)
    for col_x in cols:
        for col_y in cols:
            assert result.pearson_matrix[col_x][col_y] == result.pearson_matrix[col_y][col_x]


# ---------------------------------------------------------------------------
# Spearman matrix symmetry  (profile_features)
# ---------------------------------------------------------------------------


def test_spearman_matrix_is_symmetric():
    df = _make_df_three_cols()
    cols = ["a", "b", "c"]
    result = CorrelationProfiler(numeric_columns=cols).profile_features(df, cols)
    for col_x in cols:
        for col_y in cols:
            assert result.spearman_matrix[col_x][col_y] == result.spearman_matrix[col_y][col_x]


# ---------------------------------------------------------------------------
# Every group column appears in at least one near-redundant pairwise entry
# (profile_features)
# ---------------------------------------------------------------------------


def test_near_redundancy_group_columns_have_redundant_pairs():
    df = _make_df_with_duplicate()
    cols = ["x", "x_copy", "y"]
    result = CorrelationProfiler(numeric_columns=cols).profile_features(df, cols)
    for group in result.near_redundancy_groups:
        for col in group.columns:
            has_pair = any(
                (p.col_a == col or p.col_b == col) and p.near_redundant
                for p in result.pairwise
            )
            assert has_pair


# ---------------------------------------------------------------------------
# suggested_drop is a strict subset of the group's columns  (profile_features)
# ---------------------------------------------------------------------------


def test_suggested_drop_is_strict_subset_of_group_columns():
    df = _make_df_with_duplicate()
    cols = ["x", "x_copy", "y"]
    result = CorrelationProfiler(numeric_columns=cols).profile_features(df, cols)
    for group in result.near_redundancy_groups:
        drop_set = set(group.suggested_drop)
        col_set = set(group.columns)
        assert drop_set < col_set  # non-empty and strictly smaller


# ---------------------------------------------------------------------------
# Identical columns produce a near-redundant pair; profile_target also works
# ---------------------------------------------------------------------------


def test_identical_columns_produce_near_redundant_pair():
    df = _make_df_with_duplicate()
    feature_cols = ["x", "x_copy", "y"]
    profiler = CorrelationProfiler(numeric_columns=feature_cols)

    # profile_features: identical pair must be flagged near-redundant
    feature_result = profiler.profile_features(df, feature_cols)
    assert any(
        {p.col_a, p.col_b} == {"x", "x_copy"}
        for p in feature_result.near_redundant_pairs
    )

    # profile_target: second entry point must run without error and attach target info
    target_result = profiler.profile_target(
        df, feature_result, feature_cols, [], "y"
    )
    assert isinstance(target_result, CorrelationProfileResult)
    assert target_result.target_column == "y"

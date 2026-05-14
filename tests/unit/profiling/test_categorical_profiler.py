import polars as pl
import pytest

from dataforge_ml.profiling._categorical import CategoricalProfiler
from dataforge_ml.profiling._categorical_config import (
    CategoricalFlag,
    CategoricalProfileResult,
    CategoricalStats,
    ImbalanceMetrics,
    RareCategoryStats,
    TopValueEntry,
)


# ---------------------------------------------------------------------------
# Result type & column eligibility
# ---------------------------------------------------------------------------


def test_result_type(normal_mixed_df):
    result = CategoricalProfiler().profile(normal_mixed_df, ["category"])
    assert isinstance(result, CategoricalProfileResult)


def test_analysed_columns_only_eligible(normal_mixed_df):
    result = CategoricalProfiler().profile(normal_mixed_df, ["category", "score"])
    assert "score" not in result.analysed_columns
    assert "category" in result.analysed_columns


def test_analysed_columns_matches_columns_dict(normal_mixed_df):
    result = CategoricalProfiler().profile(normal_mixed_df, ["category"])
    assert set(result.analysed_columns) == set(result.columns.keys())


def test_stats_type_per_column(normal_mixed_df):
    result = CategoricalProfiler().profile(normal_mixed_df, ["category"])
    assert isinstance(result.columns["category"], CategoricalStats)


# ---------------------------------------------------------------------------
# Cardinality & unique_ratio
# ---------------------------------------------------------------------------


def test_cardinality_equals_distinct_non_null_count():
    # 5 distinct values, no nulls
    data = ["A", "B", "C", "D", "E"] * 12  # 60 rows, 5 distinct
    df = pl.DataFrame({"cat": pl.Series(data, dtype=pl.Utf8)})
    stats = CategoricalProfiler().profile(df, ["cat"]).columns["cat"]
    assert stats.cardinality == 5


def test_cardinality_excludes_nulls():
    # 4 distinct non-null + some nulls
    data = ["A", "B", "C", "D"] * 10 + [None] * 20
    df = pl.DataFrame({"cat": pl.Series(data, dtype=pl.Utf8)})
    stats = CategoricalProfiler().profile(df, ["cat"]).columns["cat"]
    assert stats.cardinality == 4


def test_unique_ratio_equals_cardinality_over_n_rows():
    data = ["A", "B", "C", "D", "E"] * 12  # 5 distinct, 60 rows
    df = pl.DataFrame({"cat": pl.Series(data, dtype=pl.Utf8)})
    stats = CategoricalProfiler().profile(df, ["cat"]).columns["cat"]
    assert abs(stats.unique_ratio - 5 / 60) < 1e-10


# ---------------------------------------------------------------------------
# top_values
# ---------------------------------------------------------------------------


def test_top_values_at_most_ten_entries():
    # 15 distinct categories → top_values capped at 10
    data = [str(i) for i in range(15)] * 4  # 60 rows, 15 distinct
    df = pl.DataFrame({"cat": pl.Series(data, dtype=pl.Utf8)})
    stats = CategoricalProfiler().profile(df, ["cat"]).columns["cat"]
    assert len(stats.top_values) == 10


def test_top_values_descending_count_order():
    # Uneven counts: A=30, B=20, C=10
    data = ["A"] * 30 + ["B"] * 20 + ["C"] * 10
    df = pl.DataFrame({"cat": pl.Series(data, dtype=pl.Utf8)})
    stats = CategoricalProfiler().profile(df, ["cat"]).columns["cat"]
    counts = [e.count for e in stats.top_values]
    assert counts == sorted(counts, reverse=True)


def test_top_values_entries_are_top_value_entry_type():
    data = ["X", "Y", "Z"] * 20
    df = pl.DataFrame({"cat": pl.Series(data, dtype=pl.Utf8)})
    stats = CategoricalProfiler().profile(df, ["cat"]).columns["cat"]
    for entry in stats.top_values:
        assert isinstance(entry, TopValueEntry)


# ---------------------------------------------------------------------------
# Imbalance metrics
# ---------------------------------------------------------------------------


def test_imbalance_fields_present_for_multi_category():
    data = ["A"] * 30 + ["B"] * 20 + ["C"] * 10
    df = pl.DataFrame({"cat": pl.Series(data, dtype=pl.Utf8)})
    stats = CategoricalProfiler().profile(df, ["cat"]).columns["cat"]
    assert isinstance(stats.imbalance, ImbalanceMetrics)
    assert stats.imbalance.shannon_entropy > 0.0
    assert stats.imbalance.gini_impurity > 0.0
    assert stats.imbalance.class_ratio >= 1.0


def test_imbalance_class_ratio_is_one_for_balanced():
    # Equal counts → max_freq == min_freq → class_ratio = 1.0
    data = ["A"] * 20 + ["B"] * 20 + ["C"] * 20
    df = pl.DataFrame({"cat": pl.Series(data, dtype=pl.Utf8)})
    stats = CategoricalProfiler().profile(df, ["cat"]).columns["cat"]
    assert abs(stats.imbalance.class_ratio - 1.0) < 1e-10


# ---------------------------------------------------------------------------
# NearConstant flag
# ---------------------------------------------------------------------------


def test_near_constant_flag_set():
    # 55/60 = 0.917 > 0.90 → NearConstant
    data = ["A"] * 55 + ["B"] * 5
    df = pl.DataFrame({"cat": pl.Series(data, dtype=pl.Utf8)})
    stats = CategoricalProfiler().profile(df, ["cat"]).columns["cat"]
    assert CategoricalFlag.NearConstant in stats.flags


def test_near_constant_flag_absent_for_balanced():
    data = ["A"] * 20 + ["B"] * 20 + ["C"] * 20
    df = pl.DataFrame({"cat": pl.Series(data, dtype=pl.Utf8)})
    stats = CategoricalProfiler().profile(df, ["cat"]).columns["cat"]
    assert CategoricalFlag.NearConstant not in stats.flags


# ---------------------------------------------------------------------------
# Rare categories
# ---------------------------------------------------------------------------


def test_rare_category_count_correct():
    # 200 rows: threshold_abs = max(1, floor(0.01*200)) = 2
    # "C" appears once → count=1 < 2 → rare
    data = ["A"] * 190 + ["B"] * 9 + ["C"] * 1
    df = pl.DataFrame({"cat": pl.Series(data, dtype=pl.Utf8)})
    stats = CategoricalProfiler().profile(df, ["cat"]).columns["cat"]
    assert stats.rare_categories.rare_category_count == 1


def test_rare_category_count_zero_when_none_rare():
    # All categories appear frequently
    data = ["A"] * 100 + ["B"] * 100
    df = pl.DataFrame({"cat": pl.Series(data, dtype=pl.Utf8)})
    stats = CategoricalProfiler().profile(df, ["cat"]).columns["cat"]
    assert stats.rare_categories.rare_category_count == 0


# ---------------------------------------------------------------------------
# MixedType flag
# ---------------------------------------------------------------------------


def test_mixed_type_flag_set():
    # 10 numeric-looking strings + 50 non-numeric → minority pct ≈ 16.7%
    # Wilson lower bound well above 5% threshold → MixedType
    data = ["apple"] * 50 + ["1", "2", "3", "4", "5", "6", "7", "8", "9", "10"]
    df = pl.DataFrame({"cat": pl.Series(data, dtype=pl.Utf8)})
    stats = CategoricalProfiler().profile(df, ["cat"]).columns["cat"]
    assert CategoricalFlag.MixedType in stats.flags


def test_mixed_type_flag_absent_for_pure_strings():
    data = ["apple", "banana", "cherry"] * 20
    df = pl.DataFrame({"cat": pl.Series(data, dtype=pl.Utf8)})
    stats = CategoricalProfiler().profile(df, ["cat"]).columns["cat"]
    assert CategoricalFlag.MixedType not in stats.flags

import polars as pl
import pytest

from ....splitting._splitter import DataSplitter
from ....splitting._config import FoldResult, SplitResult


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

_N = 100


@pytest.fixture(scope="module")
def df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "feature_a": pl.Series(list(range(_N)), dtype=pl.Float64),
            "feature_b": pl.Series([i * 0.5 for i in range(_N)], dtype=pl.Float64),
            "label": pl.Series(["cat" if i % 2 == 0 else "dog" for i in range(_N)], dtype=pl.Utf8),
        }
    )


@pytest.fixture(scope="module")
def df_no_target() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "x": pl.Series(list(range(_N)), dtype=pl.Float64),
            "y": pl.Series(list(range(_N, _N * 2)), dtype=pl.Float64),
        }
    )


# ---------------------------------------------------------------------------
# Constructor validation
# ---------------------------------------------------------------------------


def test_valid_construction(df):
    splitter = DataSplitter(df, target="label", random_seed=42)
    assert splitter._df is df
    assert splitter._target == "label"
    assert splitter._random_seed == 42


def test_constructor_no_target(df_no_target):
    splitter = DataSplitter(df_no_target)
    assert splitter._target is None
    assert splitter._random_seed is None


def test_constructor_raises_type_error_for_non_polars():
    with pytest.raises(TypeError):
        DataSplitter([[1, 2], [3, 4]])


def test_constructor_raises_type_error_for_numpy_array():
    import numpy as np
    with pytest.raises(TypeError):
        DataSplitter(np.zeros((10, 3)))


def test_constructor_raises_value_error_for_empty_df():
    empty = pl.DataFrame({"x": pl.Series([], dtype=pl.Float64)})
    with pytest.raises(ValueError, match="empty"):
        DataSplitter(empty)


def test_constructor_raises_value_error_for_missing_target(df):
    with pytest.raises(ValueError, match="not found"):
        DataSplitter(df, target="nonexistent_column")


# ---------------------------------------------------------------------------
# random_split — sizes and ratios
# ---------------------------------------------------------------------------


def test_random_split_sizes_sum_to_total(df):
    splitter = DataSplitter(df, target="label", random_seed=0)
    result = splitter.random_split(test_size=0.2)
    assert result.train_size + result.test_size == len(df)


def test_random_split_dataframe_row_counts_match_sizes(df):
    splitter = DataSplitter(df, target="label", random_seed=0)
    result = splitter.random_split(test_size=0.2)
    assert len(result.train) == result.train_size
    assert len(result.test) == result.test_size


def test_random_split_ratios_reflect_actual_proportions(df):
    splitter = DataSplitter(df, target="label", random_seed=0)
    result = splitter.random_split(test_size=0.2)
    total = len(df)
    assert result.train_ratio == pytest.approx(result.train_size / total)
    assert result.test_ratio == pytest.approx(result.test_size / total)


def test_random_split_returns_split_result(df):
    splitter = DataSplitter(df, target="label", random_seed=0)
    result = splitter.random_split(test_size=0.2)
    assert isinstance(result, SplitResult)


# ---------------------------------------------------------------------------
# random_split — stratification
# ---------------------------------------------------------------------------


def test_stratified_split_preserves_class_ratios(df):
    splitter = DataSplitter(df, target="label", random_seed=42)
    result = splitter.random_split(test_size=0.2, stratify=True)
    original_ratio = df["label"].value_counts(sort=True)["count"].to_list()
    train_counts = result.train["label"].value_counts(sort=True)["count"].to_list()
    test_counts = result.test["label"].value_counts(sort=True)["count"].to_list()
    # both splits should have roughly equal class representation (50/50 here)
    train_ratio = train_counts[0] / sum(train_counts)
    test_ratio = test_counts[0] / sum(test_counts)
    assert abs(train_ratio - 0.5) < 0.1
    assert abs(test_ratio - 0.5) < 0.1


def test_stratify_false_produces_valid_split(df_no_target):
    splitter = DataSplitter(df_no_target, random_seed=7)
    result = splitter.random_split(test_size=0.3, stratify=False)
    assert result.train_size + result.test_size == len(df_no_target)


def test_stratify_defaults_true_when_target_set(df):
    splitter = DataSplitter(df, target="label", random_seed=1)
    result = splitter.random_split(test_size=0.2)
    assert result.train_size + result.test_size == len(df)


def test_stratify_defaults_false_when_no_target(df_no_target):
    splitter = DataSplitter(df_no_target, random_seed=1)
    result = splitter.random_split(test_size=0.2)
    assert result.train_size + result.test_size == len(df_no_target)


def test_stratify_true_without_target_raises_value_error(df_no_target):
    splitter = DataSplitter(df_no_target)
    with pytest.raises(ValueError, match="target"):
        splitter.random_split(test_size=0.2, stratify=True)


# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------


def test_same_seed_produces_identical_splits(df):
    s1 = DataSplitter(df, target="label", random_seed=99)
    s2 = DataSplitter(df, target="label", random_seed=99)
    r1 = s1.random_split(test_size=0.2)
    r2 = s2.random_split(test_size=0.2)
    assert r1.train.equals(r2.train)
    assert r1.test.equals(r2.test)


def test_different_seeds_produce_different_splits(df):
    s1 = DataSplitter(df, target="label", random_seed=1)
    s2 = DataSplitter(df, target="label", random_seed=2)
    r1 = s1.random_split(test_size=0.2)
    r2 = s2.random_split(test_size=0.2)
    assert not r1.train.equals(r2.train)


# ---------------------------------------------------------------------------
# No profiling import leakage
# ---------------------------------------------------------------------------


def test_no_profiling_import():
    import splitting._splitter as mod
    import sys
    profiling_modules = [k for k in sys.modules if k.startswith("profiling")]
    # DataSplitter module itself must not have caused profiling to be imported
    assert "profiling" not in mod.__dict__


# ---------------------------------------------------------------------------
# time_split — fixtures
# ---------------------------------------------------------------------------

from datetime import date, timedelta

_BASE = date(2024, 1, 1)
_TIME_N = 50


@pytest.fixture(scope="module")
def time_df() -> pl.DataFrame:
    dates = [_BASE + timedelta(days=i) for i in range(_TIME_N)]
    return pl.DataFrame(
        {
            "date": pl.Series(dates, dtype=pl.Date),
            "value": pl.Series(list(range(_TIME_N)), dtype=pl.Float64),
        }
    )


@pytest.fixture(scope="module")
def time_splitter(time_df) -> DataSplitter:
    return DataSplitter(time_df)


# ---------------------------------------------------------------------------
# time_split — error cases
# ---------------------------------------------------------------------------


def test_time_split_raises_for_missing_column(time_splitter):
    with pytest.raises(ValueError, match="not found"):
        time_splitter.time_split("nonexistent")


def test_time_split_raises_when_neither_arg_provided(time_splitter):
    with pytest.raises(ValueError, match="Either"):
        time_splitter.time_split("date")


# ---------------------------------------------------------------------------
# time_split — fraction mode
# ---------------------------------------------------------------------------


def test_fraction_mode_sizes_sum_to_total(time_df, time_splitter):
    result = time_splitter.time_split("date", test_size=0.2)
    assert result.train_size + result.test_size == len(time_df)


def test_fraction_mode_test_size_is_floor(time_df, time_splitter):
    import math
    result = time_splitter.time_split("date", test_size=0.2)
    assert result.test_size == math.floor(len(time_df) * 0.2)


def test_fraction_mode_no_temporal_leakage(time_splitter):
    result = time_splitter.time_split("date", test_size=0.2)
    max_train = result.train["date"].max()
    min_test = result.test["date"].min()
    assert max_train < min_test


def test_fraction_mode_metadata_accurate(time_df, time_splitter):
    result = time_splitter.time_split("date", test_size=0.2)
    total = len(time_df)
    assert result.train_ratio == pytest.approx(result.train_size / total)
    assert result.test_ratio == pytest.approx(result.test_size / total)


# ---------------------------------------------------------------------------
# time_split — cutoff mode
# ---------------------------------------------------------------------------


def test_cutoff_mode_rows_before_cutoff_are_train(time_df, time_splitter):
    cutoff = _BASE + timedelta(days=40)
    result = time_splitter.time_split("date", cutoff=cutoff)
    assert result.train["date"].max() < cutoff


def test_cutoff_mode_rows_on_or_after_cutoff_are_test(time_df, time_splitter):
    cutoff = _BASE + timedelta(days=40)
    result = time_splitter.time_split("date", cutoff=cutoff)
    assert result.test["date"].min() == cutoff


def test_cutoff_mode_sizes_sum_to_total(time_df, time_splitter):
    cutoff = _BASE + timedelta(days=40)
    result = time_splitter.time_split("date", cutoff=cutoff)
    assert result.train_size + result.test_size == len(time_df)


def test_cutoff_mode_no_temporal_leakage(time_splitter):
    cutoff = _BASE + timedelta(days=25)
    result = time_splitter.time_split("date", cutoff=cutoff)
    assert result.train["date"].max() < result.test["date"].min()


# ---------------------------------------------------------------------------
# time_split — cutoff takes priority over test_size
# ---------------------------------------------------------------------------


def test_cutoff_takes_priority_over_test_size(time_df, time_splitter):
    cutoff = _BASE + timedelta(days=40)
    # test_size=0.5 would give 25 test rows; cutoff=day40 gives 10 test rows
    result_both = time_splitter.time_split("date", test_size=0.5, cutoff=cutoff)
    result_cutoff_only = time_splitter.time_split("date", cutoff=cutoff)
    assert result_both.test.equals(result_cutoff_only.test)
    assert result_both.train.equals(result_cutoff_only.train)


# ---------------------------------------------------------------------------
# kfold — fixtures
# ---------------------------------------------------------------------------

_KFOLD_N = 100
_K = 5


@pytest.fixture(scope="module")
def kfold_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "feature": pl.Series(list(range(_KFOLD_N)), dtype=pl.Float64),
            "label": pl.Series(["A" if i % 2 == 0 else "B" for i in range(_KFOLD_N)], dtype=pl.Utf8),
        }
    )


@pytest.fixture(scope="module")
def kfold_splitter(kfold_df) -> DataSplitter:
    return DataSplitter(kfold_df, target="label", random_seed=42)


@pytest.fixture(scope="module")
def kfold_splitter_no_target(kfold_df) -> DataSplitter:
    return DataSplitter(kfold_df, random_seed=42)


# ---------------------------------------------------------------------------
# kfold — basic structure
# ---------------------------------------------------------------------------


def test_kfold_returns_exactly_k_folds(kfold_splitter):
    folds = kfold_splitter.kfold(_K)
    assert len(folds) == _K


def test_kfold_fold_indices_zero_to_k_minus_one(kfold_splitter):
    folds = kfold_splitter.kfold(_K)
    assert [f.fold_index for f in folds] == list(range(_K))


def test_kfold_returns_fold_result_instances(kfold_splitter):
    folds = kfold_splitter.kfold(_K)
    assert all(isinstance(f, FoldResult) for f in folds)


def test_kfold_sizes_sum_to_total(kfold_df, kfold_splitter):
    folds = kfold_splitter.kfold(_K)
    for fold in folds:
        assert fold.train_size + fold.val_size == len(kfold_df)


def test_kfold_dataframe_row_counts_match_sizes(kfold_splitter):
    folds = kfold_splitter.kfold(_K)
    for fold in folds:
        assert len(fold.train) == fold.train_size
        assert len(fold.val) == fold.val_size


# ---------------------------------------------------------------------------
# kfold — non-overlapping and complete coverage
# ---------------------------------------------------------------------------


def test_kfold_val_sets_non_overlapping(kfold_df, kfold_splitter):
    folds = kfold_splitter.kfold(_K)
    # Collect all row hashes across val sets; no duplicates allowed
    seen = set()
    for fold in folds:
        for row in fold.val.iter_rows():
            assert row not in seen, f"Row {row} appeared in multiple val sets"
            seen.add(row)


def test_kfold_val_sets_cover_all_rows(kfold_df, kfold_splitter):
    folds = kfold_splitter.kfold(_K)
    all_val_rows = set()
    for fold in folds:
        for row in fold.val.iter_rows():
            all_val_rows.add(row)
    all_df_rows = set(kfold_df.iter_rows())
    assert all_val_rows == all_df_rows


# ---------------------------------------------------------------------------
# kfold — stratification
# ---------------------------------------------------------------------------


def test_stratified_kfold_preserves_class_ratios(kfold_splitter):
    folds = kfold_splitter.kfold(_K, stratify=True)
    for fold in folds:
        counts = fold.val["label"].value_counts()["count"].to_list()
        ratio = counts[0] / sum(counts)
        assert abs(ratio - 0.5) < 0.15


def test_kfold_stratify_false_produces_valid_folds(kfold_df, kfold_splitter_no_target):
    folds = kfold_splitter_no_target.kfold(_K, stratify=False)
    assert len(folds) == _K
    for fold in folds:
        assert fold.train_size + fold.val_size == len(kfold_df)


def test_kfold_stratify_defaults_true_when_target_set(kfold_splitter):
    folds = kfold_splitter.kfold(_K)
    assert len(folds) == _K


def test_kfold_stratify_defaults_false_when_no_target(kfold_df, kfold_splitter_no_target):
    folds = kfold_splitter_no_target.kfold(_K)
    assert len(folds) == _K


def test_kfold_stratify_true_without_target_raises(kfold_splitter_no_target):
    with pytest.raises(ValueError, match="target"):
        kfold_splitter_no_target.kfold(_K, stratify=True)

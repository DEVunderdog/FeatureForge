import pytest
from dataforge_ml.profiling.structural import StructuralProfiler
from dataforge_ml.profiling.config import (
    ProfileConfig,
    StructuralProfileResult,
    SemanticType,
)
from dataforge_ml.profiling._numeric_config import NumericStats
from dataforge_ml.profiling._categorical_config import CategoricalStats
from dataforge_ml.profiling._datetime_config import DatetimeStats
from dataforge_ml.profiling._boolean_config import BooleanStats
from dataforge_ml.profiling._text_config import TextStats
from dataforge_ml.profiling._target_config import TargetProfileResult


def test_happy_path(mixed_df):
    config = ProfileConfig(compute_correlation=True)
    result = StructuralProfiler(config).profile(mixed_df)

    assert isinstance(result, StructuralProfileResult)
    assert set(result.columns.keys()) == set(mixed_df.columns)
    for col_profile in result.columns.values():
        assert (
            col_profile.semantic_type is not None
        ), f"column '{col_profile.name}' has no semantic_type"


    assert result.dataset.row_count == mixed_df.height
    assert result.dataset.feature_correlation is not None


def test_no_correlation(mixed_df):
    config = ProfileConfig(compute_correlation=False)
    result = StructuralProfiler(config).profile(mixed_df)

    assert result.dataset.feature_correlation is None


def test_boolean_handoff(mixed_df):
    result = StructuralProfiler(ProfileConfig()).profile(mixed_df)

    cp = result.columns["is_active"]
    assert cp.semantic_type == SemanticType.Boolean
    assert cp.stats is not None
    assert isinstance(cp.stats, BooleanStats)
    assert cp.stats.mode in (True, False, None)


def test_text_handoff(text_df):
    result = StructuralProfiler(ProfileConfig()).profile(text_df)

    cp = result.columns["review"]
    assert cp.semantic_type == SemanticType.Text
    assert cp.stats is not None
    assert isinstance(cp.stats, TextStats)

    assert cp.stats.vocabulary_size > 0
    assert cp.stats.char_length_max >= cp.stats.char_length_min
    assert cp.stats.avg_token_count > 0
    assert 0.0 <= cp.stats.empty_ratio <= 1.0


def test_correlation_consistency(mixed_df):
    config = ProfileConfig(compute_correlation=True)
    result = StructuralProfiler(config).profile(mixed_df)

    fc = result.dataset.feature_correlation
    assert fc is not None

    # age and income are correlated by construction — forward invariant must not be vacuous
    assert len(fc.near_redundant_pairs) >= 1, (
        "expected at least one near-redundant pair (age/income are strongly correlated)"
    )

    # Forward invariant: every near_redundant pair must have both columns co-located
    # in the same NearRedundancyGroup
    for pair in fc.pairwise:
        if not pair.near_redundant:
            continue
        assert any(
            pair.col_a in group.columns and pair.col_b in group.columns
            for group in fc.near_redundancy_groups
        ), (
            f"near_redundant pair ({pair.col_a}, {pair.col_b}) "
            f"not co-located in any NearRedundancyGroup"
        )

    # Backward invariant: every column in a redundancy group must have at least
    # one near_redundant=True pair in pairwise
    for group in fc.near_redundancy_groups:
        for col in group.columns:
            assert any(
                (p.col_a == col or p.col_b == col) and p.near_redundant
                for p in fc.pairwise
            ), (
                f"column '{col}' is in a NearRedundancyGroup but has no "
                f"near_redundant=True pair in pairwise"
            )

    # Matrix symmetry — Pearson
    for col_a, row in fc.pearson_matrix.items():
        for col_b, val in row.items():
            mirror = fc.pearson_matrix.get(col_b, {}).get(col_a)
            assert mirror is not None and abs(val - mirror) < 1e-10, (
                f"Pearson matrix asymmetry: [{col_a}][{col_b}]={val} "
                f"vs [{col_b}][{col_a}]={mirror}"
            )

    # Matrix symmetry — Spearman
    for col_a, row in fc.spearman_matrix.items():
        for col_b, val in row.items():
            mirror = fc.spearman_matrix.get(col_b, {}).get(col_a)
            assert mirror is not None and abs(val - mirror) < 1e-10, (
                f"Spearman matrix asymmetry: [{col_a}][{col_b}]={val} "
                f"vs [{col_b}][{col_a}]={mirror}"
            )

    # Suggested drop is a strict subset of its group's columns
    for group in fc.near_redundancy_groups:
        group_cols = set(group.columns)
        drop_cols = set(group.suggested_drop)
        assert drop_cols < group_cols, (
            f"suggested_drop {drop_cols} is not a strict subset of "
            f"group columns {group_cols}"
        )


def test_column_handoffs(mixed_df):
    result = StructuralProfiler(ProfileConfig()).profile(mixed_df)

    stats_type_for = {
        SemanticType.Numeric: NumericStats,
        SemanticType.Categorical: CategoricalStats,
        SemanticType.Datetime: DatetimeStats,
        SemanticType.Boolean: BooleanStats,
    }

    for name, cp in result.columns.items():
        expected_type = stats_type_for.get(cp.semantic_type)
        if expected_type is None:
            continue

        assert cp.stats is not None, (
            f"column '{name}' has semantic_type={cp.semantic_type} but stats is None"
        )
        assert isinstance(cp.stats, expected_type), (
            f"column '{name}' has semantic_type={cp.semantic_type} "
            f"but stats type is {type(cp.stats).__name__}, expected {expected_type.__name__}"
        )


# ---------------------------------------------------------------------------
# Override: numeric column forced to Categorical via column_overrides
# ---------------------------------------------------------------------------


def test_column_override_changes_stats_type(override_df):
    config = ProfileConfig(column_overrides={"score": SemanticType.Categorical})
    result = StructuralProfiler(config).profile(override_df)
    cp = result.columns["score"]
    assert isinstance(cp.stats, CategoricalStats)


# ---------------------------------------------------------------------------
# Target profiling integration
# ---------------------------------------------------------------------------


def test_target_profiling_integration(target_df):
    config = ProfileConfig(target_columns=["label"])
    result = StructuralProfiler(config).profile(target_df)
    assert "label" in result.targets
    assert isinstance(result.targets["label"], TargetProfileResult)


# ---------------------------------------------------------------------------
# Empty DataFrame does not crash
# ---------------------------------------------------------------------------


def test_empty_dataframe_does_not_crash(empty_df):
    result = StructuralProfiler(ProfileConfig()).profile(empty_df)
    assert isinstance(result, StructuralProfileResult)


# ---------------------------------------------------------------------------
# Numeric handoff: float column produces NumericStats on ColumnProfile
# ---------------------------------------------------------------------------


def test_numeric_handoff(mixed_df):
    result = StructuralProfiler(ProfileConfig()).profile(mixed_df)
    cp = result.columns["income"]
    assert cp.stats is not None
    assert isinstance(cp.stats, NumericStats)


# ---------------------------------------------------------------------------
# Datetime handoff: date column produces DatetimeStats on ColumnProfile
# ---------------------------------------------------------------------------


def test_datetime_handoff(mixed_df):
    result = StructuralProfiler(ProfileConfig()).profile(mixed_df)
    cp = result.columns["joined"]
    assert cp.stats is not None
    assert isinstance(cp.stats, DatetimeStats)


# ---------------------------------------------------------------------------
# Missingness surfaced at column level for columns with nulls
# ---------------------------------------------------------------------------


def test_missingness_surfaced(mixed_df):
    result = StructuralProfiler(ProfileConfig()).profile(mixed_df)
    cp = result.columns["salary"]  # salary has ~10 % nulls by construction
    assert cp.missingness is not None
    assert cp.missingness.standard_null_count > 0

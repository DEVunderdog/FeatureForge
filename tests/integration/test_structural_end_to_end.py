import pytest
from ...profiling.structural import StructuralProfiler
from ...profiling.config import (
    ProfileConfig,
    StructuralProfileResult,
    SemanticType,
)
from ...profiling._numeric_config import NumericStats
from ...profiling._categorical_config import CategoricalStats
from ...profiling._datetime_config import DatetimeStats
from ...profiling._boolean_config import BooleanStats
from ...profiling._text_config import TextStats


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

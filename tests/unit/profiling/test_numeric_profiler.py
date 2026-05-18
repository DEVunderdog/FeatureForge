import polars as pl
import pytest

from dataforge_ml.profiling._numeric_profiler import NumericProfiler
from dataforge_ml.profiling._numeric_config import (
    KurtosisTag,
    NumericFlag,
    NumericProfileResult,
    NumericStats,
    PercentileSnapshot,
    SkewSeverity,
)


# ---------------------------------------------------------------------------
# Result type & column eligibility
# ---------------------------------------------------------------------------


def test_result_type(normal_mixed_df):
    result = NumericProfiler().profile(normal_mixed_df, ["score"])
    assert isinstance(result, NumericProfileResult)


def test_analysed_columns_only_eligible(normal_mixed_df):
    result = NumericProfiler().profile(normal_mixed_df, ["score", "salary"])
    assert "score" in result.analysed_columns
    assert "salary" in result.analysed_columns


def test_analysed_columns_matches_columns_dict(normal_mixed_df):
    result = NumericProfiler().profile(normal_mixed_df, ["score", "salary"])
    assert set(result.analysed_columns) == set(result.columns.keys())


# ---------------------------------------------------------------------------
# Core stats present for a normal float column
# ---------------------------------------------------------------------------


def test_core_stats_non_null_for_float(normal_mixed_df):
    stats = NumericProfiler().profile(normal_mixed_df, ["score"]).columns["score"]
    assert stats.mean is not None
    assert stats.median is not None
    assert stats.std is not None
    assert stats.min is not None
    assert stats.max is not None
    assert stats.mean_median_ratio is not None


def test_min_lte_max(normal_mixed_df):
    stats = NumericProfiler().profile(normal_mixed_df, ["score"]).columns["score"]
    assert stats.min <= stats.max


# ---------------------------------------------------------------------------
# All-null column
# ---------------------------------------------------------------------------


def test_all_null_column_no_crash(all_null_df):
    result = NumericProfiler().profile(all_null_df, ["float_col"])
    assert "float_col" in result.analysed_columns
    stats = result.columns["float_col"]
    assert isinstance(stats, NumericStats)
    assert stats.mean is None
    assert stats.std is None
    assert stats.min is None
    assert stats.max is None


# ---------------------------------------------------------------------------
# Single-value column
# ---------------------------------------------------------------------------


def test_single_value_std_and_skewness_zero(single_value_df):
    stats = NumericProfiler().profile(single_value_df, ["score"]).columns["score"]
    assert stats.std == 0.0
    assert stats.skewness == 0.0


# ---------------------------------------------------------------------------
# ScaleAnomaly flag
# ---------------------------------------------------------------------------


def test_scale_anomaly_flag_set():
    # 0.5 to 5000 → ratio = 10 000 ≥ 10^3 → flag
    df = pl.DataFrame({"v": pl.Series([0.5, 1.0, 1.5, 2.0, 5000.0] * 12, dtype=pl.Float64)})
    stats = NumericProfiler().profile(df, ["v"]).columns["v"]
    assert NumericFlag.ScaleAnomaly in stats.flags


def test_scale_anomaly_flag_absent_normal_range():
    df = pl.DataFrame({"v": pl.Series([10.0, 20.0, 30.0, 40.0, 50.0] * 12, dtype=pl.Float64)})
    stats = NumericProfiler().profile(df, ["v"]).columns["v"]
    assert NumericFlag.ScaleAnomaly not in stats.flags


# ---------------------------------------------------------------------------
# NearConstant flag
# ---------------------------------------------------------------------------


def test_near_constant_flag_set():
    # 55/60 = 0.917 > 0.90 → flag
    data = [5.0] * 55 + [1.0, 2.0, 3.0, 4.0, 6.0]
    df = pl.DataFrame({"v": pl.Series(data, dtype=pl.Float64)})
    stats = NumericProfiler().profile(df, ["v"]).columns["v"]
    assert NumericFlag.NearConstant in stats.flags


def test_near_constant_flag_absent():
    # 30/60 = 0.50 ≤ 0.90 → no flag
    data = [5.0] * 30 + [6.0] * 30
    df = pl.DataFrame({"v": pl.Series(data, dtype=pl.Float64)})
    stats = NumericProfiler().profile(df, ["v"]).columns["v"]
    assert NumericFlag.NearConstant not in stats.flags


# ---------------------------------------------------------------------------
# Skewness severity bands
# ---------------------------------------------------------------------------


def test_skewness_severity_normal():
    # Symmetric uniform 1–60 → |skew| ≈ 0 → Normal
    data = [float(i) for i in range(1, 61)]
    df = pl.DataFrame({"v": pl.Series(data, dtype=pl.Float64)})
    stats = NumericProfiler().profile(df, ["v"]).columns["v"]
    assert stats.skewness_severity == SkewSeverity.Normal


def test_skewness_severity_severe():
    # 57 near-zero values + 3 extreme values → |skew| >> 2.0 → Severe
    data = [0.1] * 57 + [100.0, 200.0, 300.0]
    df = pl.DataFrame({"v": pl.Series(data, dtype=pl.Float64)})
    stats = NumericProfiler().profile(df, ["v"]).columns["v"]
    assert stats.skewness_severity == SkewSeverity.Severe


# ---------------------------------------------------------------------------
# Kurtosis tag bands
# ---------------------------------------------------------------------------


def test_kurtosis_tag_leptokurtic():
    # Mass concentrated at 5.0 with symmetric outliers → excess kurtosis >> 3.0
    data = [5.0] * 54 + [0.1, 0.1, 0.1, 9.9, 9.9, 9.9]
    df = pl.DataFrame({"v": pl.Series(data, dtype=pl.Float64)})
    stats = NumericProfiler().profile(df, ["v"]).columns["v"]
    assert stats.kurtosis_tag == KurtosisTag.Leptokurtic


def test_kurtosis_tag_platykurtic():
    # Uniform over 4 equally-spaced values → excess kurtosis < -1.0
    data = [1.0] * 15 + [4.0] * 15 + [7.0] * 15 + [10.0] * 15
    df = pl.DataFrame({"v": pl.Series(data, dtype=pl.Float64)})
    stats = NumericProfiler().profile(df, ["v"]).columns["v"]
    assert stats.kurtosis_tag == KurtosisTag.Platykurtic


def test_kurtosis_tag_mesokurtic():
    # Bell-curve approximation (discrete triangular) → excess kurtosis in (-1, 3)
    data = [1.0]*3 + [2.0]*7 + [3.0]*12 + [4.0]*16 + [5.0]*12 + [6.0]*7 + [7.0]*3
    df = pl.DataFrame({"v": pl.Series(data, dtype=pl.Float64)})
    stats = NumericProfiler().profile(df, ["v"]).columns["v"]
    assert stats.kurtosis_tag == KurtosisTag.Mesokurtic


# ---------------------------------------------------------------------------
# Percentiles
# ---------------------------------------------------------------------------


def test_percentiles_type_and_all_fields_present(normal_mixed_df):
    stats = NumericProfiler().profile(normal_mixed_df, ["score"]).columns["score"]
    p = stats.percentiles
    assert isinstance(p, PercentileSnapshot)
    for val in (p.p1, p.p5, p.p25, p.p50, p.p75, p.p95, p.p99):
        assert val is not None


def test_percentiles_monotonically_non_decreasing(normal_mixed_df):
    p = NumericProfiler().profile(normal_mixed_df, ["score"]).columns["score"].percentiles
    vals = [p.p1, p.p5, p.p25, p.p50, p.p75, p.p95, p.p99]
    assert vals == sorted(vals)


# ---------------------------------------------------------------------------
# Discrete vs continuous distribution representation
# ---------------------------------------------------------------------------


def test_integer_column_produces_top_values():
    # Int64 dtype always triggers the discrete path
    data = [i % 5 for i in range(60)]
    df = pl.DataFrame({"v": pl.Series(data, dtype=pl.Int64)})
    stats = NumericProfiler().profile(df, ["v"]).columns["v"]
    assert len(stats.top_values) > 0
    assert len(stats.histogram) == 0


def test_continuous_float_produces_histogram():
    # 60 distinct floats → n_unique > _DISCRETE_MAX_UNIQUE (20) → continuous path
    data = [round(i * 0.37, 4) for i in range(60)]
    df = pl.DataFrame({"v": pl.Series(data, dtype=pl.Float64)})
    stats = NumericProfiler().profile(df, ["v"]).columns["v"]
    assert len(stats.histogram) > 0
    assert len(stats.top_values) == 0


def test_histogram_bin_count_adapts_to_data():
    # bins='auto' adapts to data size and spread rather than being fixed at 20.
    # Small dataset → fewer bins (Sturges gives ~6 for n=50).
    # Large dataset → more bins than the small one.
    import numpy as np
    rng = np.random.default_rng(42)

    small = list(rng.normal(0, 1, size=50))
    df_small = pl.DataFrame({"v": pl.Series(small, dtype=pl.Float64)})
    stats_small = NumericProfiler().profile(df_small, ["v"]).columns["v"]
    assert len(stats_small.histogram) < 20

    large = list(rng.normal(0, 1, size=500))
    df_large = pl.DataFrame({"v": pl.Series(large, dtype=pl.Float64)})
    stats_large = NumericProfiler().profile(df_large, ["v"]).columns["v"]
    assert len(stats_large.histogram) > len(stats_small.histogram)


# ---------------------------------------------------------------------------
# Batched scalar stat correctness (issue #53)
# ---------------------------------------------------------------------------


def test_batched_stats_match_single_column_baseline():
    # Profile 10 columns at once; each stat must equal the value produced when
    # profiling that same column in isolation.
    import numpy as np
    rng = np.random.default_rng(0)
    n_cols = 10
    col_names = [f"col_{i}" for i in range(n_cols)]
    data = {c: rng.normal(loc=i, scale=1.0 + i * 0.1, size=200).tolist() for i, c in enumerate(col_names)}
    df = pl.DataFrame({c: pl.Series(v, dtype=pl.Float64) for c, v in data.items()})

    batched = NumericProfiler().profile(df, col_names)

    for col in col_names:
        single = NumericProfiler().profile(df, [col]).columns[col]
        batch_col = batched.columns[col]

        assert pytest.approx(batch_col.mean, rel=1e-9) == single.mean
        assert pytest.approx(batch_col.median, rel=1e-9) == single.median
        assert pytest.approx(batch_col.std, rel=1e-9) == single.std
        assert pytest.approx(batch_col.min, rel=1e-9) == single.min
        assert pytest.approx(batch_col.max, rel=1e-9) == single.max

        bp, sp = batch_col.percentiles, single.percentiles
        for attr in ("p1", "p5", "p25", "p50", "p75", "p95", "p99"):
            assert pytest.approx(getattr(bp, attr), rel=1e-9) == getattr(sp, attr)


def test_batched_50_column_profiling_speed():
    # Regression guard: 50-column profiling should complete well under 10s on
    # any reasonable machine, confirming the single-select path is active.
    import time
    import numpy as np
    rng = np.random.default_rng(1)
    n_cols = 50
    col_names = [f"feat_{i}" for i in range(n_cols)]
    df = pl.DataFrame(
        {c: pl.Series(rng.normal(size=1_000).tolist(), dtype=pl.Float64) for c in col_names}
    )

    t0 = time.perf_counter()
    result = NumericProfiler().profile(df, col_names)
    elapsed = time.perf_counter() - t0

    assert len(result.analysed_columns) == n_cols
    assert elapsed < 10.0, f"50-column profiling took {elapsed:.2f}s — batching may be broken"

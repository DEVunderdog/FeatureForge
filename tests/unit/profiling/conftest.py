from datetime import date, timedelta

import polars as pl
import pytest

_BASE_DATE = date(2023, 1, 1)
_N = 60


@pytest.fixture(scope="session")
def empty_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "score": pl.Series([], dtype=pl.Float64),
            "count": pl.Series([], dtype=pl.Int64),
            "category": pl.Series([], dtype=pl.Utf8),
            "active": pl.Series([], dtype=pl.Boolean),
            "event_date": pl.Series([], dtype=pl.Date),
        }
    )


@pytest.fixture(scope="session")
def all_null_df() -> pl.DataFrame:
    nulls = [None] * _N
    return pl.DataFrame(
        {
            "float_col": pl.Series(nulls, dtype=pl.Float64),
            "int_col": pl.Series(nulls, dtype=pl.Int64),
            "str_col": pl.Series(nulls, dtype=pl.Utf8),
            "bool_col": pl.Series(nulls, dtype=pl.Boolean),
        }
    )


@pytest.fixture(scope="session")
def single_value_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "score": pl.Series([5.0] * _N, dtype=pl.Float64),
            "count": pl.Series([1] * _N, dtype=pl.Int64),
            "category": pl.Series(["X"] * _N, dtype=pl.Utf8),
            "active": pl.Series([True] * _N, dtype=pl.Boolean),
        }
    )


@pytest.fixture(scope="session")
def single_row_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "score": pl.Series([42.0], dtype=pl.Float64),
            "count": pl.Series([7], dtype=pl.Int64),
            "category": pl.Series(["A"], dtype=pl.Utf8),
            "active": pl.Series([True], dtype=pl.Boolean),
            "event_date": pl.Series([_BASE_DATE], dtype=pl.Date),
        }
    )


@pytest.fixture(scope="session")
def normal_mixed_df() -> pl.DataFrame:
    _CATEGORIES = ["A", "B", "C", "D", "E"]

    scores = [round(1.5 + i * 1.7 + (i % 7) * 0.3, 2) for i in range(_N)]
    counts = [i % 20 for i in range(_N)]
    categories = [_CATEGORIES[i % len(_CATEGORIES)] for i in range(_N)]
    active = [i % 2 == 0 for i in range(_N)]
    dates = [_BASE_DATE + timedelta(days=i) for i in range(_N)]
    salary = [None if i % 10 == 0 else round(30_000.0 + i * 500.0, 2) for i in range(_N)]

    return pl.DataFrame(
        {
            "score": pl.Series(scores, dtype=pl.Float64),
            "count": pl.Series(counts, dtype=pl.Int64),
            "category": pl.Series(categories, dtype=pl.Utf8),
            "active": pl.Series(active, dtype=pl.Boolean),
            "event_date": pl.Series(dates, dtype=pl.Date),
            "salary": pl.Series(salary, dtype=pl.Float64),
        }
    )

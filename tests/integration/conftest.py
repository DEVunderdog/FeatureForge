import polars as pl
import pytest


@pytest.fixture(scope="session")
def text_df():
    n = 200
    topics = ["science", "art", "history", "technology", "nature", "music"]
    texts = [
        f"A detailed description covering the topic of {topics[i % len(topics)]} "
        f"with multiple words that comfortably exceed the free-text threshold in row {i}"
        for i in range(n)
    ]
    return pl.DataFrame({"review": pl.Series(texts, dtype=pl.Utf8)})


@pytest.fixture(scope="session")
def mixed_df(rng):
    n = 300

    age = rng.integers(18, 75, size=n)
    income = age * 1200 + rng.normal(0, 5000, size=n)

    salary = rng.normal(50_000, 15_000, size=n).tolist()
    null_mask = rng.random(n) < 0.10
    salary = [None if null_mask[i] else salary[i] for i in range(n)]

    country_choices = ["US", "UK", "CA", "AU", "DE"]
    country = [country_choices[i % len(country_choices)] for i in range(n)]

    names = [f"person_{i}" for i in range(n)]

    is_active = [bool(v) for v in rng.integers(0, 2, size=n)]

    from datetime import date, timedelta
    base = date(2020, 1, 1)
    joined = [base + timedelta(days=int(d)) for d in rng.integers(0, 1460, size=n)]

    return pl.DataFrame({
        "age": pl.Series(age.tolist(), dtype=pl.Int64),
        "income": pl.Series(income.tolist(), dtype=pl.Float64),
        "salary": pl.Series(salary, dtype=pl.Float64),
        "country": pl.Series(country, dtype=pl.Utf8),
        "name": pl.Series(names, dtype=pl.Utf8),
        "is_active": pl.Series(is_active, dtype=pl.Boolean),
        "joined": pl.Series(joined, dtype=pl.Date),
    })

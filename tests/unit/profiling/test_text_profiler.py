import polars as pl

from dataforge_ml.profiling._text_profiler import TextProfiler


# ---------------------------------------------------------------------------
# vocabulary_size > 0 for a column with distinct tokens
# ---------------------------------------------------------------------------


def test_vocabulary_size_positive_for_distinct_tokens():
    df = pl.DataFrame(
        {"text": pl.Series(["apple banana", "cherry", "date elderberry"], dtype=pl.Utf8)}
    )
    stats = TextProfiler().profile(df, ["text"]).columns["text"]
    assert stats.vocabulary_size > 0


# ---------------------------------------------------------------------------
# char_length_max >= char_length_min for any non-empty text column
# ---------------------------------------------------------------------------


def test_char_length_max_gte_min():
    df = pl.DataFrame(
        {"text": pl.Series(["hi", "hello world", "a"], dtype=pl.Utf8)}
    )
    stats = TextProfiler().profile(df, ["text"]).columns["text"]
    assert stats.char_length_max >= stats.char_length_min


# ---------------------------------------------------------------------------
# empty_ratio == 0.0 for no empty strings; == 1.0 for all-empty column
# ---------------------------------------------------------------------------


def test_empty_ratio_absent_and_full():
    df_no_empty = pl.DataFrame(
        {"text": pl.Series(["hello", "world", "foo"], dtype=pl.Utf8)}
    )
    stats_no_empty = TextProfiler().profile(df_no_empty, ["text"]).columns["text"]
    assert stats_no_empty.empty_ratio == 0.0

    df_all_empty = pl.DataFrame(
        {"text": pl.Series(["", "", ""], dtype=pl.Utf8)}
    )
    stats_all_empty = TextProfiler().profile(df_all_empty, ["text"]).columns["text"]
    assert stats_all_empty.empty_ratio == 1.0


# ---------------------------------------------------------------------------
# avg_token_count > 0 for a column with multi-word entries
# ---------------------------------------------------------------------------


def test_avg_token_count_positive_for_multiword():
    df = pl.DataFrame(
        {"text": pl.Series(["the quick brown fox", "hello world", "one two three"], dtype=pl.Utf8)}
    )
    stats = TextProfiler().profile(df, ["text"]).columns["text"]
    assert stats.avg_token_count > 0

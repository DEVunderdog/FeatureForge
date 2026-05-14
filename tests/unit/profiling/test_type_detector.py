import polars as pl

from dataforge_ml.profiling._type_detector import TypeDetector
from dataforge_ml.profiling.config import SemanticType


# ---------------------------------------------------------------------------
# Native pl.Boolean resolves to SemanticType.Boolean
# ---------------------------------------------------------------------------


def test_native_boolean_column_resolves_to_boolean():
    df = pl.DataFrame(
        {"flag": pl.Series([True, False, True, True, False], dtype=pl.Boolean)}
    )
    info = TypeDetector(columns=["flag"]).detect(df)["flag"]
    assert info.semantic_type == SemanticType.Boolean


# ---------------------------------------------------------------------------
# High-cardinality string column resolves to Categorical or Text (not Numeric)
# ---------------------------------------------------------------------------


def test_high_cardinality_string_not_numeric():
    # 80 rows, 40 distinct short strings — high cardinality but below
    # the 99% identifier threshold so it stays Categorical/Text.
    vals = ["item_" + str(i % 40) for i in range(80)]
    df = pl.DataFrame({"name": pl.Series(vals, dtype=pl.Utf8)})
    info = TypeDetector(columns=["name"]).detect(df)["name"]
    assert info.semantic_type in (SemanticType.Categorical, SemanticType.Text)
    assert info.semantic_type != SemanticType.Numeric


# ---------------------------------------------------------------------------
# Identifier space-density guard: multi-word strings must not be Identifier
# ---------------------------------------------------------------------------


def test_unique_sentences_not_classified_as_identifier():
    # 100 unique short sentences — pass the 99% uniqueness threshold but
    # each value contains spaces, so the space-density guard must reject them.
    sentences = [f"The item {i} is ready" for i in range(100)]
    df = pl.DataFrame({"description": pl.Series(sentences, dtype=pl.Utf8)})
    info = TypeDetector(columns=["description"]).detect(df)["description"]
    assert info.semantic_type != SemanticType.Identifier


def test_unique_uuids_classified_as_identifier():
    # 100 unique UUID-like tokens — no spaces, short, 100% unique.
    uuids = [f"3f2a1b-{i:04d}-cd90-ef12" for i in range(100)]
    df = pl.DataFrame({"id": pl.Series(uuids, dtype=pl.Utf8)})
    info = TypeDetector(columns=["id"]).detect(df)["id"]
    assert info.semantic_type == SemanticType.Identifier


def test_unique_short_codes_classified_as_identifier():
    # 100 unique alphanumeric codes — no spaces, 100% unique.
    codes = [f"A{i:03d}" for i in range(100)]
    df = pl.DataFrame({"sku": pl.Series(codes, dtype=pl.Utf8)})
    info = TypeDetector(columns=["sku"]).detect(df)["sku"]
    assert info.semantic_type == SemanticType.Identifier

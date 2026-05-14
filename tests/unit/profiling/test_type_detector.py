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

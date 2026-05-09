import polars as pl

_INT_DTYPES = {
    pl.Int8, pl.Int16, pl.Int32, pl.Int64,
    pl.UInt8, pl.UInt16, pl.UInt32, pl.UInt64,
}

_CAT_DTYPES = {pl.Utf8, pl.String, pl.Categorical, pl.Boolean}

_NUMERIC_DTYPES = _INT_DTYPES | {pl.Float32, pl.Float64}

_DATETIME_DTYPES = {pl.Date, pl.Datetime}

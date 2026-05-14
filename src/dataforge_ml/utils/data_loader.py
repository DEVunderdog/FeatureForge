from __future__ import annotations

import io
from pathlib import Path
from typing import Union
import csv
import chardet
import polars as pl


class UnsupportedFormatError(Exception):
    """Raised when a file extension has no registered loader"""


PathOrBuffer = Union[str, Path, io.IOBase, io.RawIOBase, io.BufferedIOBase]


def _read_raw(source: PathOrBuffer) -> tuple[bytes, str | None]:
    if isinstance(source, (str, Path)):
        path = Path(source)
        if not path.exists():
            raise FileNotFoundError(f"no such file or directory: '{path}'")

        ext = path.suffix.lower()
        raw = path.read_bytes()

        return raw, ext

    pos = source.tell() if hasattr(source, "tell") else None
    raw = source.read()

    if pos is not None:
        try:
            source.seek(pos)
        except Exception:
            pass

    return raw, None


def _detect_encoding(raw: bytes) -> str:
    result = chardet.detect(raw)
    return result.get("encoding") or "utf-8"


def _sniff_csv_delimiter(text: str) -> str:
    sample = text[:4096]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=",;\t|")
        return dialect.delimiter
    except csv.Error:
        return ","



def _load_csv(raw: bytes) -> pl.DataFrame:
    """Load CSV/TSV bytes with auto-detected encoding and delimiter."""
    encoding = _detect_encoding(raw)
    text = raw.decode(encoding, errors="replace")
    delimiter = _sniff_csv_delimiter(text)
    return pl.read_csv(
        io.BytesIO(raw),
        separator=delimiter,
        encoding=encoding,
        infer_schema_length=10_000,
        try_parse_dates=True,
    )


_EXT_LOADERS: dict[str, callable] = {
    ".csv": _load_csv,
    ".tsv": _load_csv,
    ".parquet": lambda raw: pl.read_parquet(io.BytesIO(raw)),
    ".json": lambda raw: pl.read_json(io.BytesIO(raw)),
    ".ndjson": lambda raw: pl.read_ndjson(io.BytesIO(raw)),
    ".jsonl": lambda raw: pl.read_ndjson(io.BytesIO(raw)),
    ".xlsx": lambda raw: pl.read_excel(io.BytesIO(raw)),
    ".xls": lambda raw: pl.read_excel(io.BytesIO(raw)),
    ".arrow": lambda raw: pl.read_ipc(io.BytesIO(raw)),
    ".feather": lambda raw: pl.read_ipc(io.BytesIO(raw)),
}


class DataLoader:
    def __init__(self, fmt: str | None = None) -> None:
        self._fmt_override = fmt.lower() if fmt else None

    def load(
        self,
        source: PathOrBuffer,
        fmt: str | None = None,
    ) -> pl.DataFrame:
        raw, ext_from_path = _read_raw(source)

        resolved_fmt = (fmt or self._fmt_override or ext_from_path or "").lower()

        if resolved_fmt not in _EXT_LOADERS:
            label = resolved_fmt if resolved_fmt else "<unknown>"
            raise UnsupportedFormatError(
                f"Unsupported file format: '{label}'.  "
                f"Supported extensions: {sorted(_EXT_LOADERS)}"
            )

        loader = _EXT_LOADERS[resolved_fmt]
        return loader(raw)


def load(source: PathOrBuffer, fmt: str | None = None) -> pl.DataFrame:
    """Convenience wrapper — equivalent to ``DataLoader().load(source, fmt)``."""
    return DataLoader().load(source, fmt=fmt)

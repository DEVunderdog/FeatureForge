"""
TabularProfiler  –  Phase 1: Structural Profiling for tabular datasets.

Computes:
  • row / column count                (always full dataset)
  • memory usage + per-column breakdown when threshold exceeded
  • duplicate row count & ratio       (scoped to config.duplicate_columns)
  • overall sparsity                  (scoped to config.sparsity_columns)

Chunked processing is activated automatically when the DataFrame's
reported memory exceeds config.memory_threshold_mb. Pandas is used
directly; if the caller has already loaded data via Dask they should
call .compute() first or provide a custom subclass.
"""
from __future__ import annotations

import math
from typing import Any

import pandas as pd

from models.data_structure import DataStructure
from profiling.base import Profiling
from profiling.config import (
    MemoryBreakdown,
    ProfileConfig,
    TabularProfileResult,
)


class TabularProfiler(Profiling[TabularProfileResult]):
    """
    Structural profiler for pandas DataFrames.

    Usage
    -----
    >>> cfg = ProfileConfig(
    ...     duplicate_columns=["user_id", "event_time"],
    ...     sparsity_columns=["age", "income", "postcode"],
    ...     memory_threshold_mb=200,
    ... )
    >>> profiler = TabularProfiler(config=cfg)
    >>> result = profiler.profile(df)
    >>> print(result)
    """

    def __init__(self, config: ProfileConfig | None = None):
        super().__init__(DataStructure.Tabular, config)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def profile(self, data: Any) -> TabularProfileResult:
        if not isinstance(data, pd.DataFrame):
            raise TypeError(
                f"TabularProfiler expects a pandas DataFrame, got {type(data).__name__}."
            )
        return self._run(data)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run(self, df: pd.DataFrame) -> TabularProfileResult:
        result = TabularProfileResult()

        # 1. Shape — always computed on the full frame
        result.row_count = len(df)
        result.column_count = len(df.columns)

        # 2. Memory
        self._analyse_memory(df, result)

        # Decide processing mode AFTER memory analysis so we can log it
        use_chunks = result.memory_exceeded_threshold and result.row_count > 0

        # 3. Resolve column scopes
        all_cols: list[str] = df.columns.tolist()

        analysed_cols = self._resolve_columns(all_cols, self.config.columns)
        dup_cols = self._resolve_columns(
            all_cols, self.config.resolve_duplicate_columns()
        )
        sparsity_cols = self._resolve_columns(
            all_cols, self.config.resolve_sparsity_columns()
        )

        result.analysed_columns = analysed_cols
        result.duplicate_scope_columns = dup_cols
        result.sparsity_scope_columns = sparsity_cols
        result.was_chunked = use_chunks

        if result.row_count == 0:
            return result  # nothing more to do on an empty frame

        # 4. Duplicates & 5. Sparsity
        if use_chunks:
            self._chunked_metrics(df, dup_cols, sparsity_cols, result)
        else:
            self._full_metrics(df, dup_cols, sparsity_cols, result)

        return result

    # ------------------------------------------------------------------
    # Memory analysis
    # ------------------------------------------------------------------

    def _analyse_memory(self, df: pd.DataFrame, result: TabularProfileResult) -> None:
        """
        Populate memory fields on *result*.

        deep=True walks object columns to get the actual heap usage of
        Python strings / dicts stored inside the DataFrame.
        """
        mem_series: pd.Series = df.memory_usage(deep=True)
        # memory_usage returns a Series indexed by column name plus "Index"
        total_bytes: int = int(mem_series.sum())

        result.total_memory_bytes = total_bytes
        threshold_bytes = self.config.memory_threshold_mb * 1024 * 1024
        result.memory_exceeded_threshold = total_bytes > threshold_bytes

        if result.memory_exceeded_threshold:
            # Exclude the synthetic "Index" entry for the breakdown
            col_bytes = {
                col: int(mem_series[col])
                for col in mem_series.index
                if col != "Index" and col in df.columns
            }
            result.memory_breakdown = MemoryBreakdown(column_bytes=col_bytes)

    # ------------------------------------------------------------------
    # Full-frame metrics  (used when memory is below threshold)
    # ------------------------------------------------------------------

    def _full_metrics(
        self,
        df: pd.DataFrame,
        dup_cols: list[str],
        sparsity_cols: list[str],
        result: TabularProfileResult,
    ) -> None:
        result.duplicate_row_count = self._count_duplicates(df, dup_cols)
        result.duplicate_ratio = (
            result.duplicate_row_count / result.row_count if result.row_count else 0.0
        )
        result.overall_sparsity = self._compute_sparsity(df, sparsity_cols)

    # ------------------------------------------------------------------
    # Chunked metrics  (used when memory exceeds threshold)
    # ------------------------------------------------------------------

    def _chunked_metrics(
        self,
        df: pd.DataFrame,
        dup_cols: list[str],
        sparsity_cols: list[str],
        result: TabularProfileResult,
    ) -> None:
        """
        Stream through the DataFrame in row-chunks to keep peak memory low.

        Duplicate detection across chunks works by hashing the dup_cols
        subset and collecting seen hashes — exact semantics match
        pandas drop_duplicates(keep='first').

        Sparsity is accumulated as (missing_cells, total_cells) and
        converted to a fraction at the end.
        """
        chunk_size = self.config.chunk_size
        n_chunks = math.ceil(result.row_count / chunk_size)

        seen_hashes: set[int] = set()
        dup_count = 0
        missing_cells = 0
        total_cells = 0

        for i in range(n_chunks):
            start = i * chunk_size
            chunk: pd.DataFrame = df.iloc[start : start + chunk_size]

            # --- duplicates ---
            if dup_cols:
                sub = chunk[dup_cols]
            else:
                sub = chunk

            for row_tuple in sub.itertuples(index=False, name=None):
                h = hash(row_tuple)
                if h in seen_hashes:
                    dup_count += 1
                else:
                    seen_hashes.add(h)

            # --- sparsity ---
            if sparsity_cols:
                sparsity_chunk = chunk[sparsity_cols]
            else:
                sparsity_chunk = chunk

            missing_cells += int(sparsity_chunk.isna().sum().sum())
            total_cells += sparsity_chunk.size

        result.duplicate_row_count = dup_count
        result.duplicate_ratio = dup_count / result.row_count if result.row_count else 0.0
        result.overall_sparsity = missing_cells / total_cells if total_cells else 0.0

    # ------------------------------------------------------------------
    # Stateless helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _count_duplicates(df: pd.DataFrame, cols: list[str]) -> int:
        subset = cols if cols else None
        # keep=False marks ALL duplicates; keep='first' marks all *but* the
        # first occurrence — we want the number of *redundant* rows.
        return int(df.duplicated(subset=subset, keep="first").sum())

    @staticmethod
    def _compute_sparsity(df: pd.DataFrame, cols: list[str]) -> float:
        sub = df[cols] if cols else df
        if sub.size == 0:
            return 0.0
        return float(sub.isna().sum().sum()) / sub.size
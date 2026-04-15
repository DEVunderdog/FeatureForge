"""
TabularProfiler  –  Phase 1: Structural Profiling for tabular datasets.

All DataFrame operations use Polars (no pandas dependency).

Computes:
  • row / column count                (always full dataset)
  • memory usage + per-column breakdown when threshold exceeded
  • duplicate row count & ratio       (scoped to config.duplicate_columns)
  • overall sparsity                  (scoped to config.sparsity_columns)
  • data-type detection               (scoped to config.type_detection_columns;
                                       skipped entirely when None)

Chunked processing is activated automatically when the DataFrame's
estimated memory exceeds config.memory_threshold_mb.
"""
from __future__ import annotations

import math
from typing import Any

import polars as pl

from models.data_structure import DataStructure
from profiling.base import Profiling
from profiling.config import (
    MemoryBreakdown,
    ProfileConfig,
    TabularProfileResult,
)
from profiling.type_detector import TypeDetector


class TabularProfiler(Profiling[TabularProfileResult]):
    """
    Structural profiler for Polars DataFrames.

    Usage
    -----
    >>> cfg = ProfileConfig(
    ...     duplicate_columns=["user_id", "event_time"],
    ...     sparsity_columns=["age", "income", "postcode"],
    ...     type_detection_columns=["age", "income", "postcode", "created_at"],
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
        if not isinstance(data, pl.DataFrame):
            raise TypeError(
                f"TabularProfiler expects a Polars DataFrame, got {type(data).__name__}."
            )
        return self._run(data)

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _run(self, df: pl.DataFrame) -> TabularProfileResult:
        result = TabularProfileResult()

        # 1. Shape — always computed on the full frame
        result.row_count = df.height
        result.column_count = df.width

        # 2. Memory
        self._analyse_memory(df, result)

        # Decide processing mode AFTER memory analysis
        use_chunks = result.memory_exceeded_threshold and result.row_count > 0

        # 3. Resolve column scopes
        all_cols: list[str] = df.columns

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
            # Still run type detection on empty frame if requested
            self._run_type_detection(df, all_cols, result)
            return result

        # 4. Duplicates & 5. Sparsity
        if use_chunks:
            self._chunked_metrics(df, dup_cols, sparsity_cols, result)
        else:
            self._full_metrics(df, dup_cols, sparsity_cols, result)

        # 6. Type detection (selective — only when type_detection_columns set)
        self._run_type_detection(df, all_cols, result)

        return result

    # ------------------------------------------------------------------
    # Memory analysis
    # ------------------------------------------------------------------

    def _analyse_memory(self, df: pl.DataFrame, result: TabularProfileResult) -> None:
        """
        Populate memory fields on *result*.

        Polars exposes estimated_size() per Series for heap allocation.
        """
        col_bytes: dict[str, int] = {
            col: df[col].estimated_size() for col in df.columns
        }
        total_bytes = sum(col_bytes.values())

        result.total_memory_bytes = total_bytes
        threshold_bytes = self.config.memory_threshold_mb * 1024 * 1024
        result.memory_exceeded_threshold = total_bytes > threshold_bytes

        if result.memory_exceeded_threshold:
            result.memory_breakdown = MemoryBreakdown(column_bytes=col_bytes)

    # ------------------------------------------------------------------
    # Full-frame metrics
    # ------------------------------------------------------------------

    def _full_metrics(
        self,
        df: pl.DataFrame,
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
    # Chunked metrics
    # ------------------------------------------------------------------

    def _chunked_metrics(
        self,
        df: pl.DataFrame,
        dup_cols: list[str],
        sparsity_cols: list[str],
        result: TabularProfileResult,
    ) -> None:
        """
        Stream through the DataFrame in row-chunks to keep peak memory low.

        Duplicate detection: hash the dup_cols subset row-by-row and track
        seen hashes — semantics match keep='first'.
        Sparsity is accumulated as (missing_cells, total_cells).
        """
        chunk_size = self.config.chunk_size
        n_chunks = math.ceil(result.row_count / chunk_size)

        seen_hashes: set[int] = set()
        dup_count = 0
        missing_cells = 0
        total_cells = 0

        for i in range(n_chunks):
            start = i * chunk_size
            end = min(start + chunk_size, result.row_count)
            chunk: pl.DataFrame = df.slice(start, end - start)

            # --- duplicates ---
            sub = chunk.select(dup_cols) if dup_cols else chunk
            for row_tuple in sub.iter_rows():
                h = hash(row_tuple)
                if h in seen_hashes:
                    dup_count += 1
                else:
                    seen_hashes.add(h)

            # --- sparsity ---
            sparsity_chunk = chunk.select(sparsity_cols) if sparsity_cols else chunk
            missing_cells += int(
                sparsity_chunk.select(
                    pl.all().is_null().sum()
                ).row(0)[0]
                if sparsity_chunk.width == 1
                else sum(sparsity_chunk.select(pl.all().is_null().sum()).row(0))
            )
            total_cells += sparsity_chunk.height * sparsity_chunk.width

        result.duplicate_row_count = dup_count
        result.duplicate_ratio = dup_count / result.row_count if result.row_count else 0.0
        result.overall_sparsity = missing_cells / total_cells if total_cells else 0.0

    # ------------------------------------------------------------------
    # Type detection
    # ------------------------------------------------------------------

    def _run_type_detection(
        self,
        df: pl.DataFrame,
        all_cols: list[str],
        result: TabularProfileResult,
    ) -> None:
        """Run TypeDetector on the opted-in columns only."""
        if self.config.type_detection_columns is None:
            return  # opt-in not set — skip entirely

        detection_cols = self._resolve_columns(
            all_cols, self.config.type_detection_columns
        )
        if not detection_cols:
            return

        detector = TypeDetector(detection_cols)
        result.column_type_info = detector.detect(df)

    # ------------------------------------------------------------------
    # Stateless helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _count_duplicates(df: pl.DataFrame, cols: list[str]) -> int:
        """
        Count rows that are duplicates (keeping first occurrence).

        Equivalent to pandas duplicated(subset=cols, keep='first').sum().
        """
        sub = df.select(cols) if cols else df
        # is_duplicated() marks ALL occurrences of a duplicate group.
        # We want only the non-first occurrences, so we subtract the
        # number of unique rows.
        n_unique = sub.unique().height
        return df.height - n_unique

    @staticmethod
    def _compute_sparsity(df: pl.DataFrame, cols: list[str]) -> float:
        sub = df.select(cols) if cols else df
        total = sub.height * sub.width
        if total == 0:
            return 0.0
        missing = sum(sub.select(pl.all().is_null().sum()).row(0))
        return missing / total
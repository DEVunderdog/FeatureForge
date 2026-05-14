"""
Abstract base classes for all structural profilers.

Hierarchy
---------
Profiling[R]                    — root: stores config, provides _resolve_columns
├── ColumnBatchProfiler[R]      — registry tier: __init__(config=None) only;
│   │                             profile(df, columns) processes a typed column batch
│   ├── NumericProfiler
│   ├── CategoricalProfiler
│   ├── DatetimeProfiler
│   ├── BooleanProfiler
│   └── TextProfiler
├── DatasetLevelProfiler[R]     — direct-call tier: may have extra __init__ params;
│   │                             not compatible with the SemanticType registry
│   ├── MissingnessProfiler
│   ├── TargetProfiler
│   └── CorrelationProfiler
└── ModalityProfiler            — dataset-shape tier: profile(df) → DatasetStats
    └── TabularProfiler
"""

from __future__ import annotations

import polars as pl
from abc import abstractmethod, ABC
from typing import Generic, TypeVar

from .config import DatasetStats, ProfileConfig

R = TypeVar("R")


class Profiling(ABC, Generic[R]):
    """
    Root base for all profilers.

    Stores config and provides _resolve_columns. Not instantiated directly —
    use one of the three concrete tier bases below.
    """

    def __init__(self, config: ProfileConfig | None = None):
        self.config = config or ProfileConfig()

    @abstractmethod
    def profile(self, data: pl.DataFrame, **kwargs) -> R: ...

    def _resolve_columns(
        self,
        available: list[str],
        requested: list[str] | None,
    ) -> list[str]:
        if requested is None:
            return list(available)
        available_set = set(available)
        return [c for c in requested if c in available_set]


class ColumnBatchProfiler(Profiling[R]):
    """
    Registry-compatible column profiler.

    Contract
    --------
    - __init__ must accept ONLY config (no extra required params). This allows
      StructuralProfiler to instantiate any registered profiler uniformly via
          profiler_cls(config=self.config)
    - profile(df, columns) receives the full DataFrame and the list of same-type
      column names to process. Returns a result with:
          .columns: dict[str, <Stats>]        — per-column stats
          .analysed_columns: list[str]        — columns actually profiled
    """

    @abstractmethod
    def profile(self, data: pl.DataFrame, columns: list[str]) -> R: ...  # type: ignore[override]


class DatasetLevelProfiler(Profiling[R]):
    """
    Directly-called profiler.

    May have extra __init__ params (e.g. target_column, numeric_columns).
    Never registered in the SemanticType registry — always instantiated
    explicitly with its specific arguments.
    """

    @abstractmethod
    def profile(self, data: pl.DataFrame, **kwargs) -> R: ...


class ModalityProfiler(Profiling[DatasetStats]):
    """
    Dataset-shape profiler.

    One concrete implementation per Modality. Returns DatasetStats covering
    shape, memory, duplicates, sparsity, and chunking metadata.
    profile(df) takes only the DataFrame — no column list needed.
    """

    @abstractmethod
    def profile(self, data: pl.DataFrame, **kwargs) -> DatasetStats: ...  # type: ignore[override]

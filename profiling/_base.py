"""
Abstract base classes for all structural profilers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

import polars as pl

from ..models._data_structure import DataStructure
from .config import DatasetStats, ProfileConfig

R = TypeVar("R")
Stats = TypeVar("Stats")


class Profiling(ABC, Generic[R]):
    """
    Generic structural profiler (legacy base — kept for existing profilers).
    """

    def __init__(self, data_structure: DataStructure, config: ProfileConfig | None = None):
        self.data_structure = data_structure
        self.config = config or ProfileConfig()

    @abstractmethod
    def profile(self, data: Any) -> R: ...

    def _resolve_columns(
        self,
        available: list[str],
        requested: list[str] | None,
    ) -> list[str]:
        if requested is None:
            return list(available)
        available_set = set(available)
        return [c for c in requested if c in available_set]


class ColumnTypeProfiler(ABC, Generic[Stats]):
    """
    Abstract base for per-column semantic-type profilers.

    One concrete implementation exists per SemanticType.
    Receives both the target series and the full DataFrame so
    implementations can compute cross-column statistics when needed.
    """

    @abstractmethod
    def profile(self, series: pl.Series, df: pl.DataFrame) -> Stats: ...


class ModalityProfiler(ABC):
    """
    Abstract base for dataset-level (modality) profilers.

    One concrete implementation exists per Modality.
    Returns a DatasetStats summarising shape, memory, duplicates,
    sparsity, and optional matrices.
    """

    @abstractmethod
    def profile(self, df: pl.DataFrame) -> DatasetStats: ...
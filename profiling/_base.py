"""
Abstract base classes for all structural profilers.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

import polars as pl

from .config import DatasetStats, ProfileConfig

R = TypeVar("R")
Stats = TypeVar("Stats")


class Profiling(ABC, Generic[R]):

    def __init__(self, config: ProfileConfig | None = None):
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

class ModalityProfiler(ABC):
    """
    Abstract base for dataset-level (modality) profilers.

    One concrete implementation exists per Modality.
    Returns a DatasetStats summarising shape, memory, duplicates,
    sparsity, and optional matrices.
    """

    @abstractmethod
    def profile(self, df: pl.DataFrame) -> DatasetStats: ...
"""
Abstract base class for all structural profilers.

Each concrete profiler targets one DataStructure and returns its own
result type. The generic type parameter R lets callers get a properly
typed result back without casting.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from ..models._data_structure import DataStructure
from .config import ProfileConfig

R = TypeVar("R")


class Profiling(ABC, Generic[R]):
    """
    Generic structural profiler.

    Parameters
    ----------
    data_structure : DataStructure
        The kind of data this profiler handles.
    config : ProfileConfig
        Column scoping and threshold configuration.
    """

    def __init__(self, data_structure: DataStructure, config: ProfileConfig | None = None):
        self.data_structure = data_structure
        self.config = config or ProfileConfig()

    @abstractmethod
    def profile(self, data: Any) -> R:
        """
        Analyse *data* and return a structured result.

        Subclasses must override this method. The concrete type of *data*
        is validated inside each subclass (e.g. TabularProfiler checks for
        a pandas DataFrame).
        """
        ...

    # ------------------------------------------------------------------
    # Helpers available to all subclasses
    # ------------------------------------------------------------------

    def _resolve_columns(
        self,
        available: list[str],
        requested: list[str] | None,
    ) -> list[str]:
        """
        Intersect *requested* with *available*, preserving order.

        If *requested* is None all available columns are returned.
        Unknown column names are silently dropped so callers don't need to
        guard against schema drift.
        """
        if requested is None:
            return list(available)
        available_set = set(available)
        return [c for c in requested if c in available_set]
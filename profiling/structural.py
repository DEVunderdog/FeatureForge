"""
StructuralProfiler  –  unified Phase 1 entry point.

Orchestrates TabularProfiler and (optionally) CategoricalProfiler,
returning a single StructuralProfileResult that contains both.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Optional

import polars as pl

from profiling.tabular import TabularProfiler
from profiling.categorical import CategoricalProfiler
from profiling.config import ProfileConfig, TabularProfileResult
from profiling.categorical_config import CategoricalProfileResult


@dataclass
class StructuralProfileResult:
    """Combined result from both profilers."""
    tabular: TabularProfileResult = field(default_factory=TabularProfileResult)
    categorical: Optional[CategoricalProfileResult] = None

    def __str__(self) -> str:
        lines = [str(self.tabular)]
        if self.categorical:
            lines.append(str(self.categorical))
        return "\n\n".join(lines)


class StructuralProfiler:
    """
    Single entry point for Phase 1 structural profiling.

    Usage
    -----
    >>> cfg = ProfileConfig(
    ...     duplicate_columns=["user_id", "event_time"],
    ...     type_detection_columns=["age", "income"],
    ...     categorical_columns=["status", "country"],
    ... )
    >>> profiler = StructuralProfiler(config=cfg)
    >>> result = profiler.profile(df)
    >>> print(result.tabular.duplicate_ratio)
    >>> print(result.categorical.columns["status"].cardinality)
    """

    def __init__(self, config: ProfileConfig | None = None) -> None:
        self.config = config or ProfileConfig()
        self._tabular = TabularProfiler(config=self.config)

    def profile(self, data: Any) -> StructuralProfileResult:
        if not isinstance(data, pl.DataFrame):
            raise TypeError(
                f"StructuralProfiler expects a Polars DataFrame, "
                f"got {type(data).__name__}."
            )

        result = StructuralProfileResult()

        # Always run structural profiling
        result.tabular = self._tabular.profile(data)

        # Run categorical profiling only when opted in
        if self.config.categorical_columns is not None:
            cat_profiler = CategoricalProfiler(
                columns=self.config.categorical_columns,
                config=self.config,
            )
            result.categorical = cat_profiler.profile(data)

        return result
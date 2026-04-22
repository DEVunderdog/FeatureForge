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
from profiling.numeric_config import NumericProfileResult
from profiling.numeric_profiler import NumericProfiler
from profiling.missingness_profiler import MissingnessProfiler
from profiling.missingness_config import MissingnessProfileResult
from profiling.target_config import TargetProfileResult
from profiling.target_profiler import TargetProfiler


@dataclass
class StructuralProfileResult:
    """Combined result from both profilers."""

    tabular: TabularProfileResult = field(default_factory=TabularProfileResult)
    categorical: Optional[CategoricalProfileResult] = None
    numeric: Optional[NumericProfileResult] = None
    missingness: Optional[MissingnessProfileResult] = None
    target: Optional[TargetProfileResult] = None

    def __str__(self) -> str:
        lines = [str(self.tabular)]
        if self.missingness:
            lines.append(str(self.missingness))
        if self.categorical:
            lines.append(str(self.categorical))
        if self.numeric:
            lines.append(str(self.numeric))
        if self.target:
            lines.append(str(self.target))

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
        self._missingness = MissingnessProfiler(config=self.config)

    def profile(self, data: Any) -> StructuralProfileResult:
        if not isinstance(data, pl.DataFrame):
            raise TypeError(
                f"StructuralProfiler expects a Polars DataFrame, "
                f"got {type(data).__name__}."
            )

        result = StructuralProfileResult()

        # Always run structural profiling
        result.tabular = self._tabular.profile(data)

        result.missingness = self._missingness.profile(
            data,
            columns=result.tabular.analysed_columns or None,
        )

        # Run categorical profiling only when opted in
        if self.config.categorical_columns is not None:
            cat_profiler = CategoricalProfiler(
                columns=self.config.categorical_columns,
                config=self.config,
            )
            result.categorical = cat_profiler.profile(data)

        if self.config.numeric_columns is not None:
            result.numeric = NumericProfiler(
                columns=self.config.numeric_columns,
                config=self.config,
            ).profile(data=data)

        if self.config.target_columns is not None:
            target_profiler = TargetProfiler(
                target_column=self.config.target_columns,
                config=self.config,
            )
            result.target = target_profiler.profile(data)

        return result

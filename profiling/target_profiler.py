"""
TargetProfiler  –  Phase 1 extension: Target Variable Profiling.

Performs robust dtype detection to determine the problem framework
(Regression vs Classification) and assesses critical target health metrics:
  1. Target Missingness (Any missingness flags the dataset for row-dropping)
  2. Class Imbalance (For Classification tasks)
  3. Skewness / Normalcy (For Regression tasks)
"""
from __future__ import annotations

from typing import Any

import polars as pl

from models.data_structure import DataStructure
from profiling.base import Profiling
from profiling.config import ProfileConfig
from profiling.target_config import (
    TargetFlag,
    TargetProblemType,
    TargetProfileResult,
)

# Reuse your internal profilers to prevent duplication
from profiling.type_detector import TypeDetector, TypeFlag, NumericKind
from profiling.missingness_profiler import MissingnessProfiler
from profiling.categorical import CategoricalProfiler
from profiling.numeric_profiler import NumericProfiler
from profiling.numeric_config import SkewSeverity

class TargetProfiler(Profiling[TargetProfileResult]):
    """
    Analyzes the target variable to set up downstream ML behavior.
    """

    def __init__(self, target_column: str, config: ProfileConfig | None = None) -> None:
        super().__init__(DataStructure.Tabular, config)
        self.target_column = target_column

    def profile(self, data: Any) -> TargetProfileResult:
        if not isinstance(data, pl.DataFrame):
            raise TypeError(f"TargetProfiler expects a Polars DataFrame, got {type(data).__name__}.")
            
        if self.target_column not in data.columns:
            raise ValueError(f"Target column '{self.target_column}' not found in the DataFrame.")

        return self._run(data)

    def _run(self, df: pl.DataFrame) -> TargetProfileResult:
        series = df[self.target_column]
        n_rows = df.height

        # 1. Type Detection -> Problem Type Mapping
        detector = TypeDetector(columns=[self.target_column])
        type_info = detector.detect(df)[self.target_column]
        problem_type = self._determine_problem_type(series, type_info)

        result = TargetProfileResult(
            column=self.target_column,
            problem_type=problem_type
        )

        if type_info.has_flag(TypeFlag.IdentifierColumn):
            result.flags.append(TargetFlag.IsIdentifier)

        # 2. Target Missingness Check
        # We reuse MissingnessProfiler's static method to get standard + effective nulls
        col_miss_profile, _ = MissingnessProfiler._profile_column(series, self.target_column, n_rows)
        result.missing_count = col_miss_profile.effective_null_count
        result.missing_ratio = col_miss_profile.effective_null_ratio

        if result.missing_count > 0:
            result.flags.append(TargetFlag.ContainsMissing)

        # 3. Problem-Specific Profiling
        if problem_type in (TargetProblemType.BinaryClassification, TargetProblemType.MulticlassClassification):
            self._profile_classification(series, n_rows, result)
        elif problem_type == TargetProblemType.Regression:
            self._profile_regression(series, n_rows, result)

        return result

    def _determine_problem_type(self, series: pl.Series, type_info: Any) -> TargetProblemType:
        """Map TypeDetector results to an ML Problem Type."""
        # 1. Obvious Booleans
        if type_info.has_flag(TypeFlag.BooleanCandidate):
            return TargetProblemType.BinaryClassification
            
        # 2. Low-cardinality integers or strings
        if type_info.has_flag(TypeFlag.EncodedCategory) or series.dtype in (pl.Utf8, pl.String):
            n_unique = series.drop_nulls().n_unique()
            if n_unique == 2:
                return TargetProblemType.BinaryClassification
            return TargetProblemType.MulticlassClassification
            
        # 3. Confirmed numerics
        if type_info.numeric_kind in (NumericKind.Continuous, NumericKind.Discrete):
            return TargetProblemType.Regression
            
        return TargetProblemType.Unknown

    def _profile_classification(self, series: pl.Series, n_rows: int, result: TargetProfileResult) -> None:
        """Generates categorical metrics and checks for class imbalance."""
        cat_profiler = CategoricalProfiler(columns=[self.target_column], config=self.config)
        
        # Internally compute cardinality, top values, and imbalance metrics
        cat_profile = cat_profiler._profile_column(series, self.target_column, n_rows)
        result.categorical_profile = cat_profile

        # Flag Imbalances
        ratio = cat_profile.imbalance.class_ratio
        if ratio > 20.0:
            result.flags.append(TargetFlag.SevereImbalance)
        elif ratio > 5.0:
            result.flags.append(TargetFlag.HighImbalance)

    def _profile_regression(self, series: pl.Series, n_rows: int, result: TargetProfileResult) -> None:
        """Generates numeric metrics and checks for target skewness."""
        num_profiler = NumericProfiler(columns=[self.target_column], config=self.config)
        
        num_profile = num_profiler._profile_column(series, self.target_column, n_rows)
        result.numeric_profile = num_profile

        # Flag Skewness (Highly skewed targets often require Log/Yeo-Johnson transforms)
        if num_profile.skew_severity in (SkewSeverity.High, SkewSeverity.Severe):
            result.flags.append(TargetFlag.HighlySkewed)
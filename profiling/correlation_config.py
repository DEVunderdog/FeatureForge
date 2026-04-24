"""
Result dataclasses for correlation and information-structure profiling.

Populated by CorrelationProfiler, which is opt-in via
ProfileConfig.correlation_target_column (and implicitly by passing
numeric/categorical column lists that are already resolved upstream).

Design notes
------------
- Pearson matrix   : linear relationships between numeric columns.
- Spearman matrix  : monotonic (rank-based) relationships; robust to
                     outliers and non-linearity.
- Near-redundancy  : any pair with |r| > 0.95 flagged — identical signal,
                     one should be dropped before modelling.
- Feature–target   : Pearson for numeric target, ANOVA F / eta² for
                     categorical target.  Top-10 reported.
- Mutual information: MI for all features vs target (classif or regression).
                     Captures non-linear dependencies correlation misses.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Optional


# ---------------------------------------------------------------------------
# Enums
# ---------------------------------------------------------------------------

class CorrelationMethod(StrEnum):
    Pearson  = "pearson"
    Spearman = "spearman"


class TargetType(StrEnum):
    Numeric      = "numeric"      # numeric target  → Pearson + MI regression
    Categorical  = "categorical"  # categorical target → ANOVA/eta² + MI classif


# ---------------------------------------------------------------------------
# Pairwise correlation result
# ---------------------------------------------------------------------------

@dataclass
class CorrelationPair:
    """
    A single entry in the pairwise correlation results.

    Attributes
    ----------
    col_a, col_b : str
        The two column names (col_a < col_b lexicographically,
        so each pair appears exactly once).
    pearson_r : float | None
        Pearson r.  None when fewer than 3 non-null paired observations.
    spearman_r : float | None
        Spearman r.  None under the same condition.
    near_redundant : bool
        True when max(|pearson_r|, |spearman_r|) > threshold (default 0.95).
    """

    col_a: str
    col_b: str
    pearson_r:  Optional[float] = None
    spearman_r: Optional[float] = None
    near_redundant: bool = False


# ---------------------------------------------------------------------------
# Feature–target entries
# ---------------------------------------------------------------------------

@dataclass
class NumericTargetCorrelation:
    """
    Pearson r between one numeric feature and a numeric target.

    Attributes
    ----------
    feature : str
    pearson_r : float | None
    """
    feature:   str
    pearson_r: Optional[float] = None


@dataclass
class CategoricalTargetCorrelation:
    """
    ANOVA-based association between one categorical feature and a numeric
    target (or a numeric feature vs a categorical target when the roles
    are reversed — see CorrelationProfiler docs).

    Attributes
    ----------
    feature : str
    f_statistic : float | None
        One-way ANOVA F-statistic.  Higher F → stronger group separation.
    p_value : float | None
        p-value for the F-test.
    eta_squared : float | None
        Effect size: SS_between / SS_total.  Ranges [0, 1].
        Rule of thumb: 0.01 small, 0.06 medium, 0.14 large.
    """
    feature:     str
    f_statistic: Optional[float] = None
    p_value:     Optional[float] = None
    eta_squared: Optional[float] = None


# ---------------------------------------------------------------------------
# Mutual information
# ---------------------------------------------------------------------------

@dataclass
class MutualInformationEntry:
    """
    MI score for one feature vs the target.

    Attributes
    ----------
    feature : str
    mi_score : float
        Raw MI value (nats, sklearn default).  Not directly comparable
        across datasets — use rank ordering within this dataset.
    rank : int
        1 = highest MI (most informative).
    """
    feature:  str
    mi_score: float = 0.0
    rank:     int   = 0


# ---------------------------------------------------------------------------
# Near-redundancy summary
# ---------------------------------------------------------------------------

@dataclass
class NearRedundancyGroup:
    """
    A cluster of mutually near-redundant columns.

    All pairs within the group exceed the |r| > 0.95 threshold.
    The suggested_drop list contains every column except the first
    alphabetically — a simple, deterministic heuristic.
    """
    columns:       list[str] = field(default_factory=list)
    suggested_drop: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Top-level result
# ---------------------------------------------------------------------------

@dataclass
class CorrelationProfileResult:
    """
    Full correlation and information-structure profile.

    Attributes
    ----------
    analysed_numeric_columns : list[str]
        Numeric columns actually included in the pairwise matrices.
    pairwise : list[CorrelationPair]
        All (col_a, col_b) pairs, each carrying Pearson and Spearman r.
    near_redundant_pairs : list[CorrelationPair]
        Subset of *pairwise* where near_redundant is True.
    near_redundancy_groups : list[NearRedundancyGroup]
        Union-find clusters of near-redundant columns.

    target_column : str | None
        The target column supplied by the caller (may be None when no
        target is provided — only pairwise matrices are then computed).
    target_type : TargetType | None

    feature_target_numeric : list[NumericTargetCorrelation]
        Populated when target is numeric.  Top-10 by |Pearson r|.
    feature_target_categorical : list[CategoricalTargetCorrelation]
        Populated when target is categorical.  Top-10 by eta².
    mutual_information : list[MutualInformationEntry]
        All features ranked by MI vs target.  Empty when no target.

    pearson_matrix : dict[str, dict[str, float]]
        Full symmetric Pearson matrix (numeric columns only).
    spearman_matrix : dict[str, dict[str, float]]
        Full symmetric Spearman matrix (numeric columns only).
    """

    # Column scope
    analysed_numeric_columns: list[str] = field(default_factory=list)

    # Pairwise matrices
    pearson_matrix:  dict[str, dict[str, float]] = field(default_factory=dict)
    spearman_matrix: dict[str, dict[str, float]] = field(default_factory=dict)

    # Pairwise summaries
    pairwise:             list[CorrelationPair] = field(default_factory=list)
    near_redundant_pairs: list[CorrelationPair] = field(default_factory=list)
    near_redundancy_groups: list[NearRedundancyGroup] = field(default_factory=list)

    # Target info
    target_column: Optional[str]       = None
    target_type:   Optional[TargetType] = None

    # Feature–target correlations (top-10 each)
    feature_target_numeric:      list[NumericTargetCorrelation]      = field(default_factory=list)
    feature_target_categorical:  list[CategoricalTargetCorrelation]  = field(default_factory=list)

    # Mutual information (all features, ranked)
    mutual_information: list[MutualInformationEntry] = field(default_factory=list)

    # ------------------------------------------------------------------
    # Convenience helpers
    # ------------------------------------------------------------------

    def top_mi(self, n: int = 10) -> list[MutualInformationEntry]:
        """Return the top-n features by mutual information score."""
        return self.mutual_information[:n]

    def get_pearson(self, col_a: str, col_b: str) -> Optional[float]:
        return self.pearson_matrix.get(col_a, {}).get(col_b)

    def get_spearman(self, col_a: str, col_b: str) -> Optional[float]:
        return self.spearman_matrix.get(col_a, {}).get(col_b)

    def __str__(self) -> str:  # pragma: no cover
        lines = ["=== Correlation & Information Structure ==="]

        lines.append(
            f"  Numeric columns analysed : {len(self.analysed_numeric_columns)}"
        )

        if self.near_redundant_pairs:
            lines.append(
                f"  Near-redundant pairs (|r|>0.95) : {len(self.near_redundant_pairs)}"
            )
            for p in self.near_redundant_pairs:
                lines.append(
                    f"    {p.col_a!r:30s} ↔ {p.col_b!r:30s}"
                    f"  pearson={_fmt(p.pearson_r)}  spearman={_fmt(p.spearman_r)}"
                )
        else:
            lines.append("  Near-redundant pairs : none")

        if self.near_redundancy_groups:
            lines.append("  Near-redundancy groups (suggested drops):")
            for g in self.near_redundancy_groups:
                lines.append(
                    f"    keep={g.columns[0]!r}  drop={g.suggested_drop}"
                )

        if self.target_column:
            lines.append(
                f"\n  Target column : {self.target_column!r}  [{self.target_type}]"
            )

        if self.feature_target_numeric:
            lines.append("  Top feature–target correlations (Pearson):")
            for e in self.feature_target_numeric:
                lines.append(f"    {e.feature:30s}  r={_fmt(e.pearson_r)}")

        if self.feature_target_categorical:
            lines.append("  Top feature–target associations (ANOVA / eta²):")
            for e in self.feature_target_categorical:
                lines.append(
                    f"    {e.feature:30s}  F={_fmt(e.f_statistic)}  "
                    f"p={_fmt(e.p_value, '.4g')}  η²={_fmt(e.eta_squared)}"
                )

        if self.mutual_information:
            lines.append("  Top-10 features by Mutual Information:")
            for e in self.top_mi(10):
                lines.append(f"    #{e.rank:>3}  {e.feature:30s}  MI={e.mi_score:.4f}")

        return "\n".join(lines)


def _fmt(v: Optional[float], spec: str = ".4f") -> str:
    return f"{v:{spec}}" if v is not None else "N/A"
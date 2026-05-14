from __future__ import annotations

from dataclasses import dataclass

import polars as pl


@dataclass
class SplitResult:
    """
    Attributes
    ----------
    train : pl.DataFrame
        Training partition.
    test : pl.DataFrame
        Test/hold-out partition.
    train_size : int
        Number of rows in the training partition.
    test_size : int
        Number of rows in the test partition.
    train_ratio : float
        Fraction of total rows assigned to training (0.0–1.0).
    test_ratio : float
        Fraction of total rows assigned to testing (0.0–1.0).
    """

    train: pl.DataFrame
    test: pl.DataFrame
    train_size: int
    test_size: int
    train_ratio: float
    test_ratio: float


@dataclass
class FoldResult:
    """
    Attributes
    ----------
    train : pl.DataFrame
        Training partition for this fold.
    val : pl.DataFrame
        Validation partition for this fold.
    fold_index : int
        Zero-based index of this fold within the CV run.
    train_size : int
        Number of rows in the training partition.
    val_size : int
        Number of rows in the validation partition.
    """

    train: pl.DataFrame
    val: pl.DataFrame
    fold_index: int
    train_size: int
    val_size: int

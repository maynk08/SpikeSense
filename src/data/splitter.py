"""
Chronological train / validation / test splitting for time-series DataFrames.

Time-series data must NOT be split randomly — future data would leak into
training. This module enforces a strict chronological split: the first
train_ratio of the series becomes training data, the next val_ratio becomes
validation, and the remainder becomes the test set.
"""

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)


@dataclass
class SeriesSplit:
    """Container holding the three chronological splits of a single series."""

    train: pd.DataFrame
    val: pd.DataFrame
    test: pd.DataFrame
    train_ratio: float
    val_ratio: float
    test_ratio: float


def split_series(
    df: pd.DataFrame,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
) -> SeriesSplit:
    """Split a time-series DataFrame chronologically.

    Args:
        df: DataFrame sorted by time (earliest first).
        train_ratio: Fraction of data for training.
        val_ratio: Fraction of data for validation.
        test_ratio: Fraction of data for testing.
            Note: ratios must sum to 1.0 (within floating point tolerance).

    Returns:
        SeriesSplit with .train, .val, .test DataFrames and ratio attributes.
    """
    total = train_ratio + val_ratio + test_ratio
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split ratios must sum to 1.0, got {total:.6f}")

    n = len(df)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    train = df.iloc[:train_end].reset_index(drop=True)
    val = df.iloc[train_end:val_end].reset_index(drop=True)
    test = df.iloc[val_end:].reset_index(drop=True)

    logger.info(
        "Split %d rows → train=%d (%.1f%%), val=%d (%.1f%%), test=%d (%.1f%%)",
        n,
        len(train), 100 * len(train) / n,
        len(val),   100 * len(val) / n,
        len(test),  100 * len(test) / n,
    )
    return SeriesSplit(
        train=train,
        val=val,
        test=test,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
    )


def split_arrays(
    windows: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    train_ratio: float = 0.70,
    val_ratio: float = 0.15,
) -> tuple[
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
    tuple[np.ndarray, np.ndarray, np.ndarray],
]:
    """Chronologically split pre-computed window arrays.

    Returns:
        Three (windows, labels, timestamps) tuples for train, val, test.
    """
    total = train_ratio + val_ratio
    if total >= 1.0:
        raise ValueError("train_ratio + val_ratio must be < 1.0")

    n = len(windows)
    train_end = int(n * train_ratio)
    val_end = train_end + int(n * val_ratio)

    def _slice(arr: np.ndarray, s: slice) -> np.ndarray:
        return arr[s]

    train = (_slice(windows, slice(None, train_end)),
             _slice(labels, slice(None, train_end)),
             _slice(timestamps, slice(None, train_end)))
    val = (_slice(windows, slice(train_end, val_end)),
           _slice(labels, slice(train_end, val_end)),
           _slice(timestamps, slice(train_end, val_end)))
    test = (_slice(windows, slice(val_end, None)),
            _slice(labels, slice(val_end, None)),
            _slice(timestamps, slice(val_end, None)))

    logger.info(
        "Array split → train=%d, val=%d, test=%d windows",
        len(train[0]), len(val[0]), len(test[0]),
    )
    return train, val, test

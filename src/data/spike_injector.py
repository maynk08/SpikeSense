"""
Injects synthetic anomalies into a clean time-series for controlled testing.

Three injection modes:
  - point_spike:  A single value raised sharply above the local baseline.
  - level_shift:  A sudden, sustained change in the mean for a fixed duration.
  - trend_drift:  A gradual linear drift away from the baseline.

All injections return a new DataFrame with updated values and labels.
The ground-truth label column is set to 1 at every injected anomaly step,
giving a perfect ground truth for controlled evaluation.
"""

import logging
from typing import Literal

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

InjectionMode = Literal["point_spike", "level_shift", "trend_drift"]


def inject_spike(
    df: pd.DataFrame,
    index: int,
    mode: InjectionMode = "point_spike",
    magnitude_sigma: float = 4.0,
    duration: int = 20,
    slope_sigma: float = 0.05,
    rolling_window: int = 50,
    value_col: str = "value",
    label_col: str = "label",
    random_seed: int | None = None,
) -> pd.DataFrame:
    """Inject one synthetic anomaly into the series at the given index.

    Args:
        df: Input DataFrame with [timestamp, value, label] columns.
            Labels must already exist (even if all zeros for clean data).
        index: Position in the DataFrame at which to inject the anomaly.
        mode: 'point_spike', 'level_shift', or 'trend_drift'.
        magnitude_sigma: For point_spike and level_shift — how many standard
            deviations (of the local rolling window) the injected value is
            shifted upward from the local mean.
        duration: For level_shift and trend_drift — number of consecutive
            time steps affected.
        slope_sigma: For trend_drift — per-step increase expressed as a
            fraction of the series global standard deviation.
        rolling_window: Size of the rolling window used to estimate the
            local mean and std for spike magnitude calibration.
        value_col: Column name for metric values.
        label_col: Column name for binary anomaly labels.
        random_seed: Optional seed for reproducibility.

    Returns:
        A copy of df with modified values and labels in the anomaly region.
    """
    if random_seed is not None:
        np.random.seed(random_seed)

    df = df.copy()
    n = len(df)

    if index < 0 or index >= n:
        raise IndexError(f"index {index} out of range for DataFrame of length {n}")

    values = df[value_col].values.copy().astype(float)

    # Estimate local statistics using a rolling window centered before the injection point
    start = max(0, index - rolling_window)
    local_slice = values[start:index] if index > start else values[:max(1, index)]
    local_mean = float(np.mean(local_slice)) if len(local_slice) > 0 else float(np.mean(values))
    local_std = float(np.std(local_slice)) if len(local_slice) > 1 else float(np.std(values))
    if local_std < 1e-8:
        local_std = float(np.std(values)) + 1e-8

    global_std = float(np.std(values))
    if global_std < 1e-8:
        global_std = 1.0

    if mode == "point_spike":
        values[index] = local_mean + magnitude_sigma * local_std
        affected = [index]

    elif mode == "level_shift":
        end = min(index + duration, n)
        shift = magnitude_sigma * local_std
        values[index:end] += shift
        affected = list(range(index, end))

    elif mode == "trend_drift":
        end = min(index + duration, n)
        per_step = slope_sigma * global_std
        for step, i in enumerate(range(index, end)):
            values[i] += per_step * (step + 1)
        affected = list(range(index, end))

    else:
        raise ValueError(f"Unknown injection mode: '{mode}'. Use 'point_spike', 'level_shift', or 'trend_drift'.")

    df[value_col] = values
    df.loc[affected, label_col] = 1

    logger.info(
        "Injected %s at index %d (affects %d steps). Local mean=%.4f, std=%.4f",
        mode,
        index,
        len(affected),
        local_mean,
        local_std,
    )
    return df


def inject_multiple(
    df: pd.DataFrame,
    injections: list[dict],
    value_col: str = "value",
    label_col: str = "label",
) -> pd.DataFrame:
    """Apply multiple injections sequentially.

    Args:
        df: Input clean DataFrame.
        injections: List of dicts, each passed as kwargs to inject_spike
            (must include at least 'index' and 'mode').
        value_col: Column name for metric values.
        label_col: Column name for binary anomaly labels.

    Returns:
        DataFrame with all injections applied.

    Example::

        result = inject_multiple(df, [
            {"index": 200, "mode": "point_spike", "magnitude_sigma": 5.0},
            {"index": 400, "mode": "level_shift", "duration": 30},
            {"index": 700, "mode": "trend_drift", "duration": 60},
        ])
    """
    for spec in injections:
        df = inject_spike(df, value_col=value_col, label_col=label_col, **spec)
    return df


def make_clean_copy(df: pd.DataFrame, label_col: str = "label") -> pd.DataFrame:
    """Return a copy of df with all labels set to 0 (for injection experiments)."""
    df = df.copy()
    df[label_col] = 0
    return df

"""
Converts a labeled time-series DataFrame into model-ready feature arrays.

Two output formats are produced:
  1. Sliding-window sequences (shape: [N, window_size, 1]) — for the LSTM Autoencoder.
  2. Statistical feature vectors (shape: [N, n_features]) — for Isolation Forest.

The MinMaxScaler is fit only on the training split to prevent data leakage.
Fitted scaler is returned so it can be saved and reused at inference time.
"""

import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
import yaml
from scipy.stats import kurtosis, skew
from sklearn.preprocessing import MinMaxScaler, StandardScaler

logger = logging.getLogger(__name__)


@dataclass
class PreprocessedData:
    """Container for all arrays produced by the preprocessor."""

    # Raw sliding windows of normalized values: shape [N, window_size]
    windows: np.ndarray
    # LSTM-ready: shape [N, window_size, 1]
    windows_3d: np.ndarray
    # IF-ready statistical features: shape [N, n_features]
    features: np.ndarray
    # Ground truth label for each window (label of the last step in window)
    labels: np.ndarray
    # Timestamps of the last step in each window
    timestamps: np.ndarray
    # Feature names for interpretability
    feature_names: list[str] = field(default_factory=list)
    # Window size used
    window_size: int = 30
    # Stride used
    stride: int = 1


def _extract_window_features(windows: np.ndarray) -> tuple[np.ndarray, list[str]]:
    """Compute statistical features for each window.

    Args:
        windows: shape [N, window_size] of normalized values.

    Returns:
        (feature_matrix of shape [N, n_features], list of feature names)
    """
    means = windows.mean(axis=1)
    stds = windows.std(axis=1)
    mins = windows.min(axis=1)
    maxs = windows.max(axis=1)
    ranges = maxs - mins
    rms = np.sqrt((windows ** 2).mean(axis=1))
    skews = np.array([skew(w) for w in windows])
    kurts = np.array([kurtosis(w) for w in windows])

    feature_matrix = np.column_stack([means, stds, mins, maxs, ranges, rms, skews, kurts])
    feature_names = ["mean", "std", "min", "max", "range", "rms", "skew", "kurtosis"]
    return feature_matrix, feature_names


def create_windows(
    values: np.ndarray,
    labels: np.ndarray,
    timestamps: np.ndarray,
    window_size: int,
    stride: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Slice a 1-D series into overlapping windows.

    The label assigned to each window is the label of its *last* time step.
    This is the standard convention for anomaly detection: a window is
    anomalous if it ends at an anomalous moment.

    Args:
        values: 1-D array of (already scaled) metric values.
        labels: 1-D integer array of ground truth labels.
        timestamps: 1-D array of timestamps (any type).
        window_size: Number of time steps per window.
        stride: Step size between consecutive windows.

    Returns:
        (windows [N, window_size], window_labels [N], window_timestamps [N])
    """
    n = len(values)
    if n < window_size:
        raise ValueError(f"Series length {n} is shorter than window_size {window_size}.")

    indices = range(0, n - window_size + 1, stride)
    windows = np.array([values[i : i + window_size] for i in indices], dtype=np.float32)
    window_labels = np.array([labels[i + window_size - 1] for i in indices], dtype=np.int32)
    window_timestamps = np.array([timestamps[i + window_size - 1] for i in indices])
    return windows, window_labels, window_timestamps


def fit_scaler(
    values: np.ndarray,
    scaler_type: Literal["minmax", "standard"] = "minmax",
) -> MinMaxScaler | StandardScaler:
    """Fit and return a scaler on the provided 1-D array."""
    if scaler_type == "minmax":
        scaler = MinMaxScaler()
    elif scaler_type == "standard":
        scaler = StandardScaler()
    else:
        raise ValueError(f"Unknown scaler type: {scaler_type}")
    scaler.fit(values.reshape(-1, 1))
    return scaler


def preprocess_series(
    df: pd.DataFrame,
    scaler: MinMaxScaler | StandardScaler | None = None,
    window_size: int = 30,
    stride: int = 1,
    scaler_type: Literal["minmax", "standard"] = "minmax",
    value_col: str = "value",
    label_col: str = "label",
    timestamp_col: str = "timestamp",
) -> tuple["PreprocessedData", MinMaxScaler | StandardScaler]:
    """Preprocess a single series DataFrame into model-ready arrays.

    If scaler is None, a new scaler is fit on this series (use for training data).
    If scaler is provided, it is applied without refitting (use for val/test data).

    Args:
        df: DataFrame with [timestamp, value, label] columns.
        scaler: Pre-fitted scaler; None means fit a new one.
        window_size: Sliding window length.
        stride: Window stride.
        scaler_type: 'minmax' or 'standard'.
        value_col: Name of the metric value column.
        label_col: Name of the ground truth label column.
        timestamp_col: Name of the timestamp column.

    Returns:
        (PreprocessedData, fitted_scaler)
    """
    values_raw = df[value_col].values.astype(np.float64)
    labels_raw = df[label_col].values.astype(np.int32)
    timestamps_raw = df[timestamp_col].values

    if scaler is None:
        scaler = fit_scaler(values_raw, scaler_type)
        logger.info("Fitted new %s scaler on %d samples.", scaler_type, len(values_raw))

    values_scaled = scaler.transform(values_raw.reshape(-1, 1)).flatten().astype(np.float32)

    windows, window_labels, window_timestamps = create_windows(
        values_scaled, labels_raw, timestamps_raw, window_size, stride
    )

    features, feature_names = _extract_window_features(windows)
    windows_3d = windows[:, :, np.newaxis]

    data = PreprocessedData(
        windows=windows,
        windows_3d=windows_3d,
        features=features,
        labels=window_labels,
        timestamps=window_timestamps,
        feature_names=feature_names,
        window_size=window_size,
        stride=stride,
    )

    logger.info(
        "Preprocessed %d windows (window_size=%d, stride=%d). Anomalous: %d (%.2f%%)",
        len(windows),
        window_size,
        stride,
        window_labels.sum(),
        100.0 * window_labels.sum() / len(window_labels) if len(window_labels) > 0 else 0,
    )
    return data, scaler


def save_preprocessed(data: PreprocessedData, output_dir: str | Path, prefix: str = "") -> None:
    """Save all arrays from PreprocessedData to disk as .npy files."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    p = f"{prefix}_" if prefix else ""
    np.save(output_dir / f"{p}windows.npy", data.windows)
    np.save(output_dir / f"{p}windows_3d.npy", data.windows_3d)
    np.save(output_dir / f"{p}features.npy", data.features)
    np.save(output_dir / f"{p}labels.npy", data.labels)
    np.save(output_dir / f"{p}timestamps.npy", data.timestamps)
    logger.info("Saved preprocessed data to %s with prefix '%s'", output_dir, prefix)


def load_preprocessed(output_dir: str | Path, prefix: str = "") -> PreprocessedData:
    """Load PreprocessedData from .npy files saved by save_preprocessed."""
    output_dir = Path(output_dir)
    p = f"{prefix}_" if prefix else ""
    return PreprocessedData(
        windows=np.load(output_dir / f"{p}windows.npy"),
        windows_3d=np.load(output_dir / f"{p}windows_3d.npy"),
        features=np.load(output_dir / f"{p}features.npy"),
        labels=np.load(output_dir / f"{p}labels.npy"),
        timestamps=np.load(output_dir / f"{p}timestamps.npy", allow_pickle=True),
    )

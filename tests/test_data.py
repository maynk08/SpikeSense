"""
Unit tests for the data pipeline: loader, preprocessor, spike injector, splitter.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data.loader import load_series
from src.data.preprocessor import (
    PreprocessedData,
    create_windows,
    fit_scaler,
    preprocess_series,
)
from src.data.spike_injector import inject_multiple, inject_spike, make_clean_copy
from src.data.splitter import split_arrays, split_series


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def simple_series() -> pd.DataFrame:
    """A clean 200-point sine-wave series with no anomalies."""
    t = np.linspace(0, 4 * np.pi, 200)
    values = 50 + 10 * np.sin(t) + np.random.default_rng(0).normal(0, 0.5, 200)
    timestamps = pd.date_range("2024-01-01", periods=200, freq="5min")
    return pd.DataFrame({"timestamp": timestamps, "value": values, "label": 0})


@pytest.fixture()
def tmp_csv_and_labels(tmp_path: Path) -> tuple[Path, Path, str]:
    """Write a minimal NAB-style CSV and labels JSON to a temp directory."""
    file_key = "testcategory/test_metric.csv"
    anomaly_ts = "2024-01-01 01:00:00"

    # CSV
    csv_dir = tmp_path / "testcategory"
    csv_dir.mkdir()
    csv_path = csv_dir / "test_metric.csv"
    t = pd.date_range("2024-01-01", periods=100, freq="5min")
    df = pd.DataFrame({"timestamp": t.astype(str), "value": np.random.rand(100) * 100})
    df.to_csv(csv_path, index=False)

    # Labels JSON
    labels_path = tmp_path / "combined_labels.json"
    labels = {file_key: [anomaly_ts]}
    labels_path.write_text(json.dumps(labels))

    return csv_path, labels_path, file_key


@pytest.fixture()
def tmp_csv_and_windows(tmp_path: Path) -> tuple[Path, Path, str]:
    """Write a NAB-style CSV and a combined_windows.json (anomaly bands)."""
    file_key = "testcategory/test_metric.csv"

    csv_dir = tmp_path / "testcategory"
    csv_dir.mkdir()
    csv_path = csv_dir / "test_metric.csv"
    t = pd.date_range("2024-01-01", periods=100, freq="5min")
    df = pd.DataFrame({"timestamp": t.astype(str), "value": np.random.rand(100) * 100})
    df.to_csv(csv_path, index=False)

    # An anomaly window spanning rows 10..20 inclusive (11 rows).
    window = [str(t[10]), str(t[20])]
    labels_path = tmp_path / "combined_windows.json"
    labels_path.write_text(json.dumps({file_key: [window]}))

    return csv_path, labels_path, file_key


# ---------------------------------------------------------------------------
# loader tests
# ---------------------------------------------------------------------------

class TestLoader:
    def test_loads_csv_correctly(self, tmp_csv_and_labels):
        csv_path, labels_path, file_key = tmp_csv_and_labels
        df = load_series(csv_path, labels_path, file_key=file_key)
        assert isinstance(df, pd.DataFrame)
        assert list(df.columns) == ["timestamp", "value", "label"]
        assert df["label"].dtype == int

    def test_label_applied_at_correct_timestamp(self, tmp_csv_and_labels):
        csv_path, labels_path, file_key = tmp_csv_and_labels
        df = load_series(csv_path, labels_path, file_key=file_key)
        # Only the row matching anomaly_ts should be labeled 1
        labeled_rows = df[df["label"] == 1]
        assert len(labeled_rows) == 1
        assert str(labeled_rows.iloc[0]["timestamp"]) == "2024-01-01 01:00:00"

    def test_window_labels_mark_full_band(self, tmp_csv_and_windows):
        csv_path, labels_path, file_key = tmp_csv_and_windows
        df = load_series(csv_path, labels_path, file_key=file_key)
        labeled = df[df["label"] == 1]
        # Every row inside the [row10 .. row20] band is anomalous (11 rows).
        assert len(labeled) == 11
        assert labeled["timestamp"].is_monotonic_increasing
        # Rows outside the band stay normal.
        assert df.loc[:9, "label"].sum() == 0
        assert df.loc[21:, "label"].sum() == 0

    def test_raises_on_missing_csv(self, tmp_path):
        labels_path = tmp_path / "labels.json"
        labels_path.write_text("{}")
        with pytest.raises(FileNotFoundError):
            load_series(tmp_path / "nonexistent.csv", labels_path)

    def test_raises_on_missing_labels(self, tmp_csv_and_labels):
        csv_path, _, _ = tmp_csv_and_labels
        with pytest.raises(FileNotFoundError):
            load_series(csv_path, Path("/nonexistent/labels.json"))

    def test_sorted_by_timestamp(self, tmp_csv_and_labels):
        csv_path, labels_path, file_key = tmp_csv_and_labels
        df = load_series(csv_path, labels_path, file_key=file_key)
        assert df["timestamp"].is_monotonic_increasing

    def test_no_nan_values(self, tmp_csv_and_labels):
        csv_path, labels_path, file_key = tmp_csv_and_labels
        df = load_series(csv_path, labels_path, file_key=file_key)
        assert not df["value"].isna().any()


# ---------------------------------------------------------------------------
# preprocessor tests
# ---------------------------------------------------------------------------

class TestPreprocessor:
    def test_create_windows_shape(self, simple_series):
        values = simple_series["value"].values.astype(np.float32)
        labels = simple_series["label"].values.astype(np.int32)
        timestamps = simple_series["timestamp"].values
        windows, wlabels, wts = create_windows(values, labels, timestamps, window_size=30, stride=1)
        expected_n = len(values) - 30 + 1
        assert windows.shape == (expected_n, 30)
        assert len(wlabels) == expected_n
        assert len(wts) == expected_n

    def test_window_stride(self, simple_series):
        values = simple_series["value"].values.astype(np.float32)
        labels = simple_series["label"].values.astype(np.int32)
        timestamps = simple_series["timestamp"].values
        windows, _, _ = create_windows(values, labels, timestamps, window_size=10, stride=5)
        # Approximate: ceil((200 - 10) / 5) + 1
        assert len(windows) > 0

    def test_scaler_clips_to_01(self, simple_series):
        values = simple_series["value"].values
        scaler = fit_scaler(values, "minmax")
        scaled = scaler.transform(values.reshape(-1, 1)).flatten()
        assert scaled.min() >= 0.0 - 1e-6
        assert scaled.max() <= 1.0 + 1e-6

    def test_preprocess_series_returns_correct_types(self, simple_series):
        data, scaler = preprocess_series(simple_series, window_size=30, stride=1)
        assert isinstance(data, PreprocessedData)
        assert data.windows.ndim == 2
        assert data.windows_3d.ndim == 3
        assert data.windows_3d.shape[2] == 1
        assert data.features.ndim == 2
        assert data.features.shape[1] == 8  # 8 statistical features
        assert len(data.labels) == len(data.windows)

    def test_val_uses_train_scaler(self, simple_series):
        train = simple_series.iloc[:140].reset_index(drop=True)
        val = simple_series.iloc[140:].reset_index(drop=True)
        train_data, scaler = preprocess_series(train, window_size=10)
        val_data, _ = preprocess_series(val, scaler=scaler, window_size=10)
        # Val data may be outside [0,1] due to using train scaler — that's correct
        assert val_data.windows is not None

    def test_raises_on_short_series(self):
        short_df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=5, freq="5min"),
            "value": np.ones(5),
            "label": np.zeros(5, dtype=int),
        })
        with pytest.raises(ValueError, match="shorter than window_size"):
            preprocess_series(short_df, window_size=30)


# ---------------------------------------------------------------------------
# spike injector tests
# ---------------------------------------------------------------------------

class TestSpikeInjector:
    def test_point_spike_raises_value(self, simple_series):
        df = make_clean_copy(simple_series)
        original_value = df.loc[100, "value"]
        result = inject_spike(df, index=100, mode="point_spike", magnitude_sigma=5.0)
        assert result.loc[100, "value"] > original_value

    def test_point_spike_sets_label(self, simple_series):
        df = make_clean_copy(simple_series)
        result = inject_spike(df, index=100, mode="point_spike")
        assert result.loc[100, "label"] == 1
        # All other labels should remain 0
        assert result.loc[result.index != 100, "label"].sum() == 0

    def test_level_shift_duration(self, simple_series):
        df = make_clean_copy(simple_series)
        result = inject_spike(df, index=50, mode="level_shift", duration=15)
        assert result.loc[50:64, "label"].sum() == 15

    def test_trend_drift_is_monotonically_increasing(self, simple_series):
        df = make_clean_copy(simple_series)
        original = df["value"].values.copy()
        result = inject_spike(df, index=50, mode="trend_drift", duration=30, slope_sigma=0.1)
        # Each injected step should be higher than the original
        for i in range(50, 80):
            assert result.loc[i, "value"] >= original[i]

    def test_does_not_modify_original(self, simple_series):
        df = make_clean_copy(simple_series)
        original_values = df["value"].copy()
        inject_spike(df, index=100, mode="point_spike")
        pd.testing.assert_series_equal(df["value"], original_values)

    def test_inject_multiple(self, simple_series):
        df = make_clean_copy(simple_series)
        specs = [
            {"index": 20, "mode": "point_spike"},
            {"index": 80, "mode": "level_shift", "duration": 10},
            {"index": 150, "mode": "trend_drift", "duration": 20},
        ]
        result = inject_multiple(df, specs)
        assert result["label"].sum() > 0

    def test_invalid_mode_raises(self, simple_series):
        with pytest.raises(ValueError, match="Unknown injection mode"):
            inject_spike(simple_series, index=10, mode="invalid_mode")  # type: ignore

    def test_out_of_range_index_raises(self, simple_series):
        with pytest.raises(IndexError):
            inject_spike(simple_series, index=9999, mode="point_spike")


# ---------------------------------------------------------------------------
# splitter tests
# ---------------------------------------------------------------------------

class TestSplitter:
    def test_split_ratios_sum(self, simple_series):
        result = split_series(simple_series, train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
        total = len(result.train) + len(result.val) + len(result.test)
        assert total == len(simple_series)

    def test_chronological_order_preserved(self, simple_series):
        result = split_series(simple_series)
        # Last train timestamp must be before first val timestamp
        assert result.train["timestamp"].iloc[-1] < result.val["timestamp"].iloc[0]
        assert result.val["timestamp"].iloc[-1] < result.test["timestamp"].iloc[0]

    def test_invalid_ratio_raises(self, simple_series):
        with pytest.raises(ValueError, match="sum to 1.0"):
            split_series(simple_series, train_ratio=0.5, val_ratio=0.3, test_ratio=0.1)

    def test_split_arrays_shapes(self):
        windows = np.random.rand(100, 30).astype(np.float32)
        labels = np.zeros(100, dtype=np.int32)
        timestamps = np.arange(100)
        train, val, test = split_arrays(windows, labels, timestamps)
        assert len(train[0]) + len(val[0]) + len(test[0]) == 100

    def test_split_arrays_no_data_leakage(self):
        windows = np.arange(300).reshape(100, 3).astype(np.float32)
        labels = np.zeros(100, dtype=np.int32)
        timestamps = np.arange(100)
        train, val, test = split_arrays(windows, labels, timestamps, train_ratio=0.7, val_ratio=0.15)
        # Train indices must all be < first val index
        train_max_ts = train[2].max()
        val_min_ts = val[2].min()
        assert train_max_ts < val_min_ts

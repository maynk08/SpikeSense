"""
Tests for the Isolation Forest and LSTM Autoencoder model modules.

LSTM tests use a tiny synthetic model (window_size=5, 4 units) to stay fast
without requiring a GPU. The real trained models are tested via load() round-trips.
"""

import json
import tempfile
from pathlib import Path

import numpy as np
import pytest

from src.models import isolation_forest as if_module
from src.models import lstm_autoencoder as lstm_module

MODELS_DIR = Path("models")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_features(n: int = 200, n_features: int = 8, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.standard_normal((n, n_features)).astype(np.float32)


def _make_windows_3d(n: int = 200, window_size: int = 5, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(0, 1, (n, window_size, 1)).astype(np.float32)


def _small_lstm(window_size: int = 5):
    return lstm_module.build_model(
        window_size=window_size,
        encoder_units=4,
        bottleneck_units=4,
        decoder_units=4,
        dropout=0.0,
        learning_rate=0.01,
    )


# ---------------------------------------------------------------------------
# Isolation Forest tests
# ---------------------------------------------------------------------------

class TestIsolationForest:
    def test_train_returns_fitted_model(self):
        features = _make_features(300)
        model = if_module.train(features, n_estimators=10, contamination=0.05)
        # Fitted model should have estimators
        assert len(model.estimators_) == 10

    def test_score_shape(self):
        features = _make_features(300)
        model = if_module.train(features, n_estimators=10)
        scores = if_module.score(model, features)
        assert scores.shape == (300,)
        assert scores.dtype == np.float32

    def test_score_is_higher_for_outliers(self):
        rng = np.random.default_rng(1)
        # Normal cluster around 0
        normal = rng.standard_normal((500, 8)).astype(np.float32)
        # Clear outliers far from 0
        outliers = (rng.standard_normal((10, 8)) * 0.1 + 50).astype(np.float32)
        model = if_module.train(normal, n_estimators=50, contamination=0.02)
        normal_scores = if_module.score(model, normal)
        outlier_scores = if_module.score(model, outliers)
        assert outlier_scores.mean() > normal_scores.mean()

    def test_predict_binary_output(self):
        features = _make_features(200)
        model = if_module.train(features, n_estimators=10, contamination=0.05)
        preds = if_module.predict(model, features)
        assert set(preds).issubset({0, 1})
        assert preds.dtype == np.int32

    def test_predict_flags_roughly_contamination_fraction(self):
        features = _make_features(1000, seed=42)
        contamination = 0.05
        model = if_module.train(features, n_estimators=100, contamination=contamination)
        preds = if_module.predict(model, features)
        flag_rate = preds.mean()
        # Should flag close to contamination% of training data
        assert abs(flag_rate - contamination) < 0.03

    def test_contamination_sweep_returns_all_values(self):
        features = _make_features(200)
        labels = np.zeros(200, dtype=np.int32)
        cvals = [0.01, 0.05, 0.10]
        results = if_module.contamination_sweep(features, features, labels, contamination_values=cvals, n_estimators=10)
        assert len(results) == 3
        returned_cvals = [r["contamination"] for r in results]
        assert returned_cvals == cvals

    def test_save_and_load_roundtrip(self, tmp_path):
        features = _make_features(200)
        model = if_module.train(features, n_estimators=10)
        paths = if_module.save(model, tmp_path, feature_names=["a", "b"])
        assert paths["model"].exists()
        assert paths["metadata"].exists()

        loaded_model, meta = if_module.load(tmp_path)
        # Loaded model should produce same predictions
        original_preds = if_module.predict(model, features)
        loaded_preds = if_module.predict(loaded_model, features)
        np.testing.assert_array_equal(original_preds, loaded_preds)
        assert meta["feature_names"] == ["a", "b"]

    def test_load_raises_on_missing_model(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            if_module.load(tmp_path)

    @pytest.mark.skipif(not MODELS_DIR.exists(), reason="models/ dir not present")
    def test_load_trained_model_from_disk(self):
        model, meta = if_module.load(MODELS_DIR)
        assert model is not None
        assert "contamination" in meta
        assert "feature_names" in meta


# ---------------------------------------------------------------------------
# LSTM Autoencoder tests
# ---------------------------------------------------------------------------

class TestLSTMAutoencoder:
    def test_build_model_output_shape(self):
        model = _small_lstm(window_size=5)
        windows = _make_windows_3d(8, window_size=5)
        output = model.predict(windows, verbose=0)
        assert output.shape == windows.shape

    def test_reconstruction_errors_shape(self):
        model = _small_lstm(window_size=5)
        windows = _make_windows_3d(50, window_size=5)
        errors = lstm_module.compute_reconstruction_errors(model, windows)
        assert errors.shape == (50,)
        assert errors.dtype == np.float32

    def test_reconstruction_errors_are_non_negative(self):
        model = _small_lstm(window_size=5)
        windows = _make_windows_3d(50, window_size=5)
        errors = lstm_module.compute_reconstruction_errors(model, windows)
        assert (errors >= 0).all()

    def test_threshold_percentile(self):
        errors = np.array([0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0], dtype=np.float32)
        t = lstm_module.compute_threshold(errors, percentile=90)
        assert abs(t - 0.91) < 0.01

    def test_predict_binary_output(self):
        model = _small_lstm(window_size=5)
        windows = _make_windows_3d(30, window_size=5)
        errors = lstm_module.compute_reconstruction_errors(model, windows)
        threshold = float(np.percentile(errors, 95))
        preds = lstm_module.predict(model, windows, threshold=threshold)
        assert set(preds).issubset({0, 1})
        assert preds.dtype == np.int32

    def test_normal_windows_have_low_error_after_training(self):
        """After training on normal data, reconstruction error on normal windows
        should be lower than on out-of-distribution windows."""
        rng = np.random.default_rng(0)
        # Simple normal pattern: values oscillating around 0.5
        normal = (0.5 + 0.1 * np.sin(np.linspace(0, 20 * np.pi, 500))).astype(np.float32)
        normal_windows = np.array([
            normal[i:i+5] for i in range(0, 490, 5)
        ])[:, :, np.newaxis].astype(np.float32)

        # Out-of-distribution: random noise
        ood_windows = rng.uniform(0, 1, (20, 5, 1)).astype(np.float32)

        model = _small_lstm(window_size=5)
        lstm_module.train(model, normal_windows, epochs=20, batch_size=8, early_stopping_patience=0)

        normal_errors = lstm_module.compute_reconstruction_errors(model, normal_windows)
        ood_errors = lstm_module.compute_reconstruction_errors(model, ood_windows)
        assert normal_errors.mean() < ood_errors.mean()

    def test_threshold_sweep_returns_correct_count(self):
        errors = np.random.default_rng(0).uniform(0, 1, 200).astype(np.float32)
        labels = np.zeros(200, dtype=np.int32)
        pcts = [80, 90, 95, 99]
        results = lstm_module.threshold_sweep(errors, labels, percentiles=pcts)
        assert len(results) == len(pcts)
        assert all("threshold" in r for r in results)

    def test_save_and_load_roundtrip(self, tmp_path):
        model = _small_lstm(window_size=5)
        windows = _make_windows_3d(20, window_size=5)
        errors = lstm_module.compute_reconstruction_errors(model, windows)
        threshold = lstm_module.compute_threshold(errors)

        lstm_module.save(model, threshold, tmp_path, window_size=5, threshold_percentile=99)
        assert (tmp_path / "lstm_autoencoder.keras").exists()
        assert (tmp_path / "lstm_metadata.json").exists()

        loaded_model, meta = lstm_module.load(tmp_path)
        assert abs(meta["threshold"] - threshold) < 1e-6
        assert meta["window_size"] == 5

        # Loaded model should produce same predictions
        orig_errors = lstm_module.compute_reconstruction_errors(model, windows)
        loaded_errors = lstm_module.compute_reconstruction_errors(loaded_model, windows)
        np.testing.assert_allclose(orig_errors, loaded_errors, rtol=1e-4)

    def test_load_raises_on_missing_model(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            lstm_module.load(tmp_path)

    @pytest.mark.skipif(not MODELS_DIR.exists(), reason="models/ dir not present")
    def test_load_trained_model_from_disk(self):
        model, meta = lstm_module.load(MODELS_DIR)
        assert model is not None
        assert "threshold" in meta
        assert meta["threshold"] > 0
        assert meta["window_size"] == 30

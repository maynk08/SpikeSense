"""
In-process model scoring for the Streamlit dashboard.

This module lets the dashboard run the Isolation Forest + LSTM Autoencoder
models *directly*, without a separate FastAPI backend. It is what makes the
single-service deployment (Streamlit Community Cloud) possible: the dashboard
loads the committed model artifacts once and scores series in-process.

It reuses ``ModelRegistry`` (the same loader the API uses) and returns plain
dicts whose shapes exactly match the JSON the API used to return, so
``dashboard/app.py`` does not need to change how it reads results.

The model load is wrapped in ``functools.lru_cache`` so the multi-MB artifacts
are loaded only once per process.
"""

from __future__ import annotations

import functools
import sys
from pathlib import Path

import numpy as np

# Ensure project root is importable when launched from repo root.
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.api.model_loader import ModelRegistry


@functools.lru_cache(maxsize=1)
def _registry() -> ModelRegistry:
    """Load model artifacts once per process (cached)."""
    reg = ModelRegistry()
    reg.load(ROOT / "config" / "config.yaml")
    return reg


def is_reachable() -> bool:
    """Always True once models load — there is no remote service to reach."""
    try:
        return _registry().loaded
    except Exception:
        return False


def info() -> dict | None:
    """Model metadata, matching the API's /info response fields."""
    reg = _registry()
    return {
        "window_size": reg.window_size,
        "lstm_threshold": reg.lstm_threshold,
        "if_contamination": float(reg.if_meta.get("contamination", 0.0)),
    }


def _score_arrays(scaled: np.ndarray, reg: ModelRegistry) -> list[dict]:
    """Score an (n_windows, window_size) scaled array; return per-window dicts."""
    from src.data.preprocessor import _extract_window_features
    from src.models import isolation_forest as if_module
    from src.models import lstm_autoencoder as lstm_module

    features, _ = _extract_window_features(scaled)
    windows_3d = scaled[:, :, np.newaxis]

    if_scores = if_module.score(reg.if_model, features)
    if_preds = if_module.predict(reg.if_model, features)
    lstm_errors = lstm_module.score(reg.lstm_model, windows_3d)
    lstm_preds = lstm_module.predict(reg.lstm_model, windows_3d, reg.lstm_threshold)

    preds = []
    for i in range(len(scaled)):
        if_pred = bool(if_preds[i])
        lstm_pred = bool(lstm_preds[i])
        preds.append({
            "if_anomaly": if_pred,
            "lstm_anomaly": lstm_pred,
            "combined_union": if_pred or lstm_pred,
            "combined_intersection": if_pred and lstm_pred,
            "scores": {
                "isolation_forest": float(if_scores[i]),
                "lstm_reconstruction_error": float(lstm_errors[i]),
            },
            "alert_sent": False,
            "window_size": reg.window_size,
        })
    return preds


def _batch_response(preds: list[dict]) -> dict:
    return {
        "predictions": preds,
        "n_windows": len(preds),
        "n_anomalies_union": sum(p["combined_union"] for p in preds),
        "n_anomalies_lstm": sum(p["lstm_anomaly"] for p in preds),
        "n_anomalies_if": sum(p["if_anomaly"] for p in preds),
    }


def predict_batch(windows: list[list[float]], series_key: str | None = None) -> dict | None:
    """Score many windows in one vectorized pass (mirrors /predict/batch)."""
    if not windows:
        return _batch_response([])

    reg = _registry()
    raw = np.array(windows, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != reg.window_size:
        return None

    scaler = reg.get_scaler(series_key)
    scaled = scaler.transform(raw.reshape(-1, 1)).reshape(raw.shape).astype(np.float32)
    return _batch_response(_score_arrays(scaled, reg))


def inject_spike(
    series_key: str,
    mode: str = "point_spike",
    magnitude_sigma: float = 4.0,
    duration: int = 20,
) -> dict | None:
    """Inject a synthetic anomaly and score the result (mirrors /demo/inject-spike).

    Note: unlike the API path, this does not fire Discord alerts or persist to a
    database — neither is meaningful in the single-service demo deployment.
    """
    from src.data.loader import load_all_series
    from src.data.preprocessor import create_windows
    from src.data.spike_injector import inject_spike as do_inject, make_clean_copy

    reg = _registry()
    all_series = load_all_series(ROOT / "config" / "config.yaml")
    if series_key not in all_series:
        return None

    df = make_clean_copy(all_series[series_key])
    inject_index = len(df) // 2
    df = do_inject(df, index=inject_index, mode=mode,
                   magnitude_sigma=magnitude_sigma, duration=duration)

    scaler = reg.get_scaler(series_key)
    scaled = scaler.transform(df["value"].values.reshape(-1, 1)).flatten().astype(np.float32)
    labels = df["label"].values.astype(np.int32)
    timestamps = df["timestamp"].values

    windows, window_labels, _ = create_windows(
        scaled, labels, timestamps, window_size=reg.window_size, stride=1,
    )

    batch = _batch_response(_score_arrays(windows, reg))

    return {
        "series_key": series_key,
        "mode": mode,
        "inject_index": inject_index,
        "n_anomalous_windows": int(window_labels.sum()),
        "predictions": batch,
    }


def evaluate() -> dict | None:
    """Return pre-computed evaluation results (mirrors /evaluate)."""
    from src.evaluation.evaluator import load_results

    try:
        results = load_results(str(ROOT / "results"))
    except FileNotFoundError:
        return None

    test_eval = results.get("test_split_evaluation", {})
    spike_scenarios = results.get("spike_scenario_evaluation", [])

    spike_summary = []
    for s in spike_scenarios:
        for m in s.get("metrics", []):
            spike_summary.append({
                "scenario": s["scenario"],
                "model": m["model"],
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
            })

    return {
        "metrics": test_eval.get("metrics", []),
        "dataset_stats": results.get("dataset_stats", {}),
        "detector_config": results.get("model_config", {}),
        "spike_scenario_summary": spike_summary,
    }

"""
Evaluation module for Spike-Sense anomaly detectors.

Computes precision, recall, F1, false-positive rate, and confusion matrix for
Isolation Forest, LSTM Autoencoder, and two combination strategies (union / intersection).

Also supports:
  - Per-scenario evaluation (point_spike / level_shift / trend_drift)
  - Precision-Recall curve generation
  - Batch scoring against pre-processed arrays on disk

Results are saved as structured JSON and CSV for the evaluation notebook and API.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
)

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Core metric computation
# ---------------------------------------------------------------------------

def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    model_name: str = "model",
) -> dict:
    """Compute the full set of binary classification metrics.

    Args:
        y_true: Ground-truth labels (0/1).
        y_pred: Predicted labels (0/1).
        model_name: Label used in the returned dict.

    Returns:
        Dict with keys: model, precision, recall, f1, fpr, support_pos,
        support_neg, n_total, tp, fp, tn, fn.
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    y_pred = np.asarray(y_pred, dtype=np.int32)

    tn, fp, fn, tp = confusion_matrix(y_true, y_pred, labels=[0, 1]).ravel()

    n_pos = int(y_true.sum())
    n_neg = int((1 - y_true).sum())

    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))
    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0

    result = {
        "model": model_name,
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "fpr": round(fpr, 4),
        "tp": int(tp),
        "fp": int(fp),
        "tn": int(tn),
        "fn": int(fn),
        "support_pos": n_pos,
        "support_neg": n_neg,
        "n_total": len(y_true),
        "flag_rate": round(float(y_pred.mean()), 4),
    }
    logger.info(
        "%s → P=%.3f R=%.3f F1=%.3f FPR=%.3f  (TP=%d FP=%d TN=%d FN=%d)",
        model_name, prec, rec, f1, fpr, tp, fp, tn, fn,
    )
    return result


def compute_pr_curve(
    y_true: np.ndarray,
    scores: np.ndarray,
    model_name: str = "model",
) -> dict:
    """Compute Precision-Recall curve points for plotting.

    Args:
        y_true: Ground-truth labels (0/1).
        scores: Continuous anomaly scores (higher = more anomalous).
        model_name: Label for the result.

    Returns:
        Dict with 'model', 'precision', 'recall', 'thresholds', 'avg_precision'.
    """
    y_true = np.asarray(y_true, dtype=np.int32)

    if y_true.sum() == 0:
        logger.warning("PR curve for '%s': no positive labels in y_true.", model_name)
        return {"model": model_name, "precision": [], "recall": [], "thresholds": [], "avg_precision": float("nan")}

    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    avg_prec = float(average_precision_score(y_true, scores))

    return {
        "model": model_name,
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "thresholds": thresholds.tolist(),
        "avg_precision": round(avg_prec, 4),
    }


# ---------------------------------------------------------------------------
# Batch evaluation across all four configurations
# ---------------------------------------------------------------------------

def evaluate_all(
    y_true: np.ndarray,
    if_scores: np.ndarray,
    lstm_errors: np.ndarray,
    if_preds: np.ndarray,
    lstm_preds: np.ndarray,
) -> dict:
    """Evaluate all four prediction configurations.

    Configurations:
      1. Isolation Forest only
      2. LSTM Autoencoder only
      3. Combined Union  (anomaly if EITHER flags)
      4. Combined Intersection (anomaly if BOTH flag)

    Args:
        y_true: Ground-truth labels.
        if_scores: IF anomaly scores (from if_module.score()).
        lstm_errors: LSTM reconstruction errors.
        if_preds: Binary IF predictions.
        lstm_preds: Binary LSTM predictions.

    Returns:
        Dict with keys 'metrics' (list of 4 metric dicts), 'pr_curves' (list of 4),
        'confusion_matrices' (list of 4).
    """
    y_true = np.asarray(y_true, dtype=np.int32)
    union_preds = np.clip(if_preds + lstm_preds, 0, 1)
    inter_preds = (if_preds & lstm_preds).astype(np.int32)

    configs = [
        ("Isolation Forest", if_preds, if_scores),
        ("LSTM Autoencoder", lstm_preds, lstm_errors),
        ("Combined (Union)", union_preds, if_scores + lstm_errors),
        ("Combined (Intersection)", inter_preds, np.minimum(if_scores, lstm_errors)),
    ]

    metrics_list = []
    pr_curves = []
    cm_list = []

    for name, preds, scores in configs:
        metrics_list.append(compute_metrics(y_true, preds, name))
        pr_curves.append(compute_pr_curve(y_true, scores, name))
        cm = confusion_matrix(y_true, preds, labels=[0, 1]).tolist()
        cm_list.append({"model": name, "matrix": cm, "labels": ["Normal", "Anomaly"]})

    return {
        "metrics": metrics_list,
        "pr_curves": pr_curves,
        "confusion_matrices": cm_list,
    }


# ---------------------------------------------------------------------------
# Spike scenario evaluation
# ---------------------------------------------------------------------------

def evaluate_spike_scenarios(
    base_series: np.ndarray,
    if_model,
    lstm_model,
    lstm_threshold: float,
    scaler,
    window_size: int = 30,
) -> list[dict]:
    """Evaluate both models on each of the three synthetic spike types.

    Injects each spike type into a clean copy of base_series, preprocesses,
    scores with both models, and returns per-scenario metrics.

    Args:
        base_series: Clean 1-D float array of raw metric values.
        if_model: Fitted IsolationForest.
        lstm_model: Trained LSTM Autoencoder Keras model.
        lstm_threshold: LSTM anomaly threshold.
        scaler: Fitted MinMaxScaler for this series.
        window_size: Sliding window length.

    Returns:
        List of dicts, one per spike scenario, each containing metrics for
        IF, LSTM, and combined predictions.
    """
    from src.data.spike_injector import inject_spike
    from src.data.preprocessor import create_windows, _extract_window_features
    from src.models import isolation_forest as if_module
    from src.models import lstm_autoencoder as lstm_module

    n = len(base_series)
    inject_index = n // 2  # inject in the middle of the series

    scenarios = [
        ("point_spike",  dict(mode="point_spike",  magnitude_sigma=5.0)),
        ("level_shift",  dict(mode="level_shift",  magnitude_sigma=3.5, duration=20)),
        ("trend_drift",  dict(mode="trend_drift",  slope_sigma=0.08,    duration=50)),
    ]

    results = []
    for scenario_name, inject_kwargs in scenarios:
        # Build a labelled DataFrame for spike injection
        import pandas as pd
        df = pd.DataFrame({
            "timestamp": pd.date_range("2024-01-01", periods=n, freq="5min"),
            "value": base_series.copy(),
            "label": 0,
        })
        df = inject_spike(df, index=inject_index, **inject_kwargs)

        values_scaled = scaler.transform(
            df["value"].values.reshape(-1, 1)
        ).flatten().astype(np.float32)

        labels = df["label"].values.astype(np.int32)
        timestamps = df["timestamp"].values

        windows, window_labels, _ = create_windows(
            values_scaled, labels, timestamps, window_size=window_size, stride=1
        )
        features, _ = _extract_window_features(windows)
        windows_3d = windows[:, :, np.newaxis]

        if_scores = if_module.score(if_model, features)
        if_preds = if_module.predict(if_model, features)
        lstm_errors = lstm_module.score(lstm_model, windows_3d)
        lstm_preds = lstm_module.predict(lstm_model, windows_3d, threshold=lstm_threshold)

        eval_result = evaluate_all(window_labels, if_scores, lstm_errors, if_preds, lstm_preds)
        results.append({
            "scenario": scenario_name,
            "inject_index": inject_index,
            "n_anomalous_windows": int(window_labels.sum()),
            "metrics": eval_result["metrics"],
        })
        logger.info(
            "Scenario '%s': %d anomalous windows", scenario_name, window_labels.sum()
        )

    return results


# ---------------------------------------------------------------------------
# Summary table helpers
# ---------------------------------------------------------------------------

def metrics_to_dataframe(metrics_list: list[dict]) -> pd.DataFrame:
    """Convert a list of metric dicts to a display-ready DataFrame."""
    rows = []
    for m in metrics_list:
        rows.append({
            "Model": m["model"],
            "Precision": m["precision"],
            "Recall": m["recall"],
            "F1": m["f1"],
            "FPR": m["fpr"],
            "TP": m["tp"],
            "FP": m["fp"],
            "TN": m["tn"],
            "FN": m["fn"],
            "Flag Rate": m["flag_rate"],
        })
    return pd.DataFrame(rows)


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_results(results: dict, results_dir: str | Path, filename: str = "evaluation_results.json") -> Path:
    """Save evaluation results dict as formatted JSON."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / filename
    out_path.write_text(json.dumps(results, indent=2))
    logger.info("Evaluation results saved → %s", out_path)
    return out_path


def save_metrics_csv(metrics_list: list[dict], results_dir: str | Path, filename: str = "metrics_summary.csv") -> Path:
    """Save metrics list as a CSV table."""
    results_dir = Path(results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)
    out_path = results_dir / filename
    metrics_to_dataframe(metrics_list).to_csv(out_path, index=False)
    logger.info("Metrics CSV saved → %s", out_path)
    return out_path


def load_results(results_dir: str | Path, filename: str = "evaluation_results.json") -> dict:
    path = Path(results_dir) / filename
    if not path.exists():
        raise FileNotFoundError(f"Results not found: {path}")
    return json.loads(path.read_text())

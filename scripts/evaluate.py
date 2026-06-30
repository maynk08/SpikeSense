"""
Evaluation entry point for Spike-Sense.

Loads trained models and preprocessed test arrays, scores with both detectors,
computes precision/recall/F1 across four configurations, runs spike-injection
scenarios, and saves all results to results/.

Usage:
    python scripts/evaluate.py                       # full evaluation
    python scripts/evaluate.py --config path/to/cfg.yaml
    python scripts/evaluate.py --no-spikes           # skip spike scenarios
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loader import load_all_series, load_config
from src.data.preprocessor import preprocess_series
from src.data.splitter import split_series
from src.evaluation.evaluator import (
    evaluate_all,
    evaluate_spike_scenarios,
    save_metrics_csv,
    save_results,
)
from src.models import isolation_forest as if_module
from src.models import lstm_autoencoder as lstm_module

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Evaluate Spike-Sense anomaly detectors")
    p.add_argument("--config", default="config/config.yaml")
    p.add_argument("--no-spikes", action="store_true", help="Skip spike scenario evaluation")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    models_dir = Path(cfg["api"]["models_dir"])
    results_dir = Path(cfg["evaluation"]["results_dir"])
    results_dir.mkdir(parents=True, exist_ok=True)

    win_cfg = cfg["preprocessing"]
    window_size: int = win_cfg["window_size"]
    stride: int = win_cfg["stride"]
    split_cfg = cfg["data"]["split"]

    lstm_meta_path = models_dir / "lstm_metadata.json"
    lstm_threshold: float = json.loads(lstm_meta_path.read_text())["threshold"]

    # ------------------------------------------------------------------
    # Load models
    # ------------------------------------------------------------------
    logger.info("=== Loading models ===")
    if_model, if_meta = if_module.load(models_dir)
    lstm_model, lstm_meta = lstm_module.load(models_dir)
    logger.info("IF contamination=%.3f, LSTM threshold=%.6f", if_meta["contamination"], lstm_threshold)

    # ------------------------------------------------------------------
    # Load and preprocess test splits for all series
    # ------------------------------------------------------------------
    logger.info("=== Loading and preprocessing test splits ===")
    all_series = load_all_series(args.config)

    # Aggregate test arrays across series (each scaled independently)
    test_windows_all, test_features_all, test_labels_all = [], [], []
    # Keep first series' raw values + scaler for spike scenarios
    spike_series_raw: np.ndarray | None = None
    spike_scaler = None

    for key, df in all_series.items():
        split = split_series(df, split_cfg["train"], split_cfg["val"], split_cfg["test"])
        train_data, scaler = preprocess_series(split.train, scaler=None, window_size=window_size, stride=stride)
        test_data, _ = preprocess_series(split.test, scaler=scaler, window_size=window_size, stride=stride)

        test_windows_all.append(test_data.windows_3d)
        test_features_all.append(test_data.features)
        test_labels_all.append(test_data.labels)

        if spike_series_raw is None:
            spike_series_raw = split.train["value"].values.astype(np.float32)
            # Extend with val+test for a longer series to inject into
            full_vals = df["value"].values.astype(np.float32)
            spike_series_raw = full_vals
            spike_scaler = scaler

    X_test_3d = np.concatenate(test_windows_all, axis=0)
    X_test_feat = np.concatenate(test_features_all, axis=0)
    y_test = np.concatenate(test_labels_all, axis=0)

    logger.info(
        "Test set: %d windows, %d anomalous (%.2f%%)",
        len(y_test), y_test.sum(), 100 * y_test.sum() / max(len(y_test), 1),
    )

    # ------------------------------------------------------------------
    # Score both models on the test split
    # ------------------------------------------------------------------
    logger.info("=== Scoring test split ===")
    if_scores = if_module.score(if_model, X_test_feat)
    if_preds = if_module.predict(if_model, X_test_feat)
    lstm_errors = lstm_module.score(lstm_model, X_test_3d)
    lstm_preds = lstm_module.predict(lstm_model, X_test_3d, threshold=lstm_threshold)

    logger.info("IF flagged: %d / %d (%.2f%%)", if_preds.sum(), len(if_preds), 100 * if_preds.mean())
    logger.info("LSTM flagged: %d / %d (%.2f%%)", lstm_preds.sum(), len(lstm_preds), 100 * lstm_preds.mean())

    # ------------------------------------------------------------------
    # Evaluate all configurations on test split
    # ------------------------------------------------------------------
    logger.info("=== Evaluating all configurations ===")
    test_eval = evaluate_all(y_test, if_scores, lstm_errors, if_preds, lstm_preds)

    # ------------------------------------------------------------------
    # Spike injection scenarios (controlled ground truth)
    # ------------------------------------------------------------------
    spike_results = []
    if not args.no_spikes and spike_series_raw is not None:
        logger.info("=== Running spike injection scenarios ===")
        spike_results = evaluate_spike_scenarios(
            base_series=spike_series_raw,
            if_model=if_model,
            lstm_model=lstm_model,
            lstm_threshold=lstm_threshold,
            scaler=spike_scaler,
            window_size=window_size,
        )

    # ------------------------------------------------------------------
    # Save all results
    # ------------------------------------------------------------------
    full_results = {
        "test_split_evaluation": test_eval,
        "spike_scenario_evaluation": spike_results,
        "dataset_stats": {
            "n_test_windows": int(len(y_test)),
            "n_test_anomalous": int(y_test.sum()),
            "anomaly_rate": round(float(y_test.sum() / max(len(y_test), 1)), 4),
        },
        "model_config": {
            "if_contamination": if_meta.get("contamination"),
            "lstm_threshold": lstm_threshold,
            "lstm_threshold_percentile": lstm_meta.get("threshold_percentile"),
            "window_size": window_size,
        },
    }

    save_results(full_results, results_dir, "evaluation_results.json")
    save_metrics_csv(test_eval["metrics"], results_dir, "metrics_summary.csv")

    # Pretty-print summary
    logger.info("=== Evaluation Summary (Test Split) ===")
    for m in test_eval["metrics"]:
        logger.info(
            "  %-28s  P=%.3f  R=%.3f  F1=%.3f  FPR=%.3f",
            m["model"], m["precision"], m["recall"], m["f1"], m["fpr"],
        )

    if spike_results:
        logger.info("=== Spike Scenario Summary ===")
        for sr in spike_results:
            for m in sr["metrics"]:
                if m["model"] in ("Isolation Forest", "LSTM Autoencoder", "Combined (Union)"):
                    logger.info(
                        "  %-12s | %-28s | P=%.3f  R=%.3f  F1=%.3f",
                        sr["scenario"], m["model"], m["precision"], m["recall"], m["f1"],
                    )

    logger.info("=== Evaluation complete. Results in %s ===", results_dir)


if __name__ == "__main__":
    main()

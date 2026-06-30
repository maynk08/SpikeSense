"""
Training entry point for Spike-Sense.

Runs the full training pipeline for both Isolation Forest and LSTM Autoencoder
across all configured NAB series. Saves model artifacts to models/ and
training diagnostics (contamination sweep, threshold sweep) to results/.

Usage:
    python scripts/train.py                         # use default config
    python scripts/train.py --config path/to/config.yaml
    python scripts/train.py --skip-lstm             # IF only (fast)
    python scripts/train.py --series ec2_cpu_utilization_825cc2.csv  # single file
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
import yaml

# Ensure project root is on sys.path when running as a script
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.data.loader import load_all_series, load_config
from src.data.preprocessor import preprocess_series, save_preprocessed
from src.data.splitter import split_series
from src.models import isolation_forest as if_module
from src.models import lstm_autoencoder as lstm_module

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train Spike-Sense anomaly detectors")
    p.add_argument("--config", default="config/config.yaml", help="Path to config.yaml")
    p.add_argument("--skip-lstm", action="store_true", help="Skip LSTM training")
    p.add_argument("--series", default=None, help="Train on a single series key (filename only)")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    models_dir = Path(cfg["api"]["models_dir"])
    results_dir = Path(cfg["evaluation"]["results_dir"])
    models_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    win_cfg = cfg["preprocessing"]
    window_size: int = win_cfg["window_size"]
    stride: int = win_cfg["stride"]
    scaler_type: str = win_cfg["scaler"]

    split_cfg = cfg["data"]["split"]
    train_r: float = split_cfg["train"]
    val_r: float = split_cfg["val"]
    test_r: float = split_cfg["test"]

    if_cfg = cfg["isolation_forest"]
    lstm_cfg = cfg["lstm"]

    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    logger.info("=== Loading series ===")
    all_series = load_all_series(args.config)

    if args.series:
        all_series = {k: v for k, v in all_series.items() if args.series in k}
        if not all_series:
            logger.error("No series matched filter '%s'", args.series)
            sys.exit(1)

    # Aggregate windows across all series for a single combined model
    # (one model trained on all series generalises better than per-file models)
    train_windows_all, train_features_all, train_labels_all = [], [], []
    val_windows_all, val_features_all, val_labels_all = [], [], []

    # Each series is scaled independently (its own MinMaxScaler fit on its train split).
    # This ensures all windows land in [0, 1] regardless of the metric's raw magnitude
    # (e.g., CPU % vs. network bytes differ by 6+ orders of magnitude).
    # Per-series scalers are saved in processed/ for use at inference time.

    for key, df in all_series.items():
        logger.info("--- Processing: %s ---", key.split("/")[-1])
        split = split_series(df, train_r, val_r, test_r)

        # scaler=None → fit a fresh scaler on this series' train split
        train_data, scaler = preprocess_series(
            split.train,
            scaler=None,
            window_size=window_size,
            stride=stride,
            scaler_type=scaler_type,
        )

        val_data, _ = preprocess_series(
            split.val, scaler=scaler, window_size=window_size, stride=stride
        )
        test_data, _ = preprocess_series(
            split.test, scaler=scaler, window_size=window_size, stride=stride
        )

        # Save per-series processed arrays
        safe_key = key.replace("/", "_").replace(".csv", "")
        processed_dir = Path(cfg["data"]["processed_dir"])
        save_preprocessed(train_data, processed_dir, prefix=f"{safe_key}_train")
        save_preprocessed(val_data, processed_dir, prefix=f"{safe_key}_val")
        save_preprocessed(test_data, processed_dir, prefix=f"{safe_key}_test")

        train_windows_all.append(train_data.windows_3d)
        train_features_all.append(train_data.features)
        train_labels_all.append(train_data.labels)
        val_windows_all.append(val_data.windows_3d)
        val_features_all.append(val_data.features)
        val_labels_all.append(val_data.labels)

    # Stack all series
    X_train_3d = np.concatenate(train_windows_all, axis=0)
    X_train_feat = np.concatenate(train_features_all, axis=0)
    y_train = np.concatenate(train_labels_all, axis=0)
    X_val_3d = np.concatenate(val_windows_all, axis=0)
    X_val_feat = np.concatenate(val_features_all, axis=0)
    y_val = np.concatenate(val_labels_all, axis=0)

    # For LSTM: train on normal windows only
    normal_mask = y_train == 0
    X_train_normal = X_train_3d[normal_mask]
    logger.info(
        "Combined: %d train windows (%d normal), %d val windows",
        len(X_train_3d), len(X_train_normal), len(X_val_3d),
    )

    # ------------------------------------------------------------------
    # Isolation Forest
    # ------------------------------------------------------------------
    logger.info("=== Training Isolation Forest ===")

    # Contamination sweep on val split
    logger.info("Running contamination sweep...")
    sweep_results = if_module.contamination_sweep(
        X_train_feat, X_val_feat, y_val,
        contamination_values=if_cfg["contamination_sweep"],
        n_estimators=if_cfg["n_estimators"],
        random_state=if_cfg["random_state"],
    )
    sweep_path = results_dir / "if_contamination_sweep.json"
    sweep_path.write_text(json.dumps(sweep_results, indent=2))
    logger.info("Contamination sweep saved → %s", sweep_path)

    # Train final model with configured contamination
    if_model = if_module.train(
        X_train_feat,
        n_estimators=if_cfg["n_estimators"],
        contamination=if_cfg["contamination"],
        max_features=if_cfg["max_features"],
        random_state=if_cfg["random_state"],
    )
    feature_names = [
        "mean", "std", "min", "max", "range", "rms", "skew", "kurtosis"
    ]
    if_module.save(if_model, models_dir, feature_names=feature_names)
    logger.info("Isolation Forest training complete.")

    # ------------------------------------------------------------------
    # LSTM Autoencoder
    # ------------------------------------------------------------------
    if args.skip_lstm:
        logger.info("Skipping LSTM training (--skip-lstm flag).")
        return

    logger.info("=== Training LSTM Autoencoder ===")
    lstm_model = lstm_module.build_model(
        window_size=lstm_cfg["window_size"],
        encoder_units=lstm_cfg["encoder_units"],
        bottleneck_units=lstm_cfg["bottleneck_units"],
        decoder_units=lstm_cfg["decoder_units"],
        dropout=lstm_cfg["dropout"],
        learning_rate=lstm_cfg["learning_rate"],
    )
    lstm_model.summary(print_fn=logger.info)

    history = lstm_module.train(
        lstm_model,
        windows_3d=X_train_normal,
        val_windows_3d=X_val_3d,
        epochs=lstm_cfg["epochs"],
        batch_size=lstm_cfg["batch_size"],
        early_stopping_patience=lstm_cfg["early_stopping_patience"],
        random_seed=lstm_cfg["random_seed"],
    )

    # Save training history
    history_path = results_dir / "lstm_training_history.json"
    history_path.write_text(json.dumps({
        "loss": history.history["loss"],
        "val_loss": history.history.get("val_loss", []),
    }, indent=2))
    logger.info("Training history saved → %s", history_path)

    # Compute threshold on training reconstruction errors
    logger.info("Computing anomaly threshold on training errors...")
    train_errors = lstm_module.compute_reconstruction_errors(lstm_model, X_train_normal)
    threshold = lstm_module.compute_threshold(train_errors, lstm_cfg["threshold_percentile"])

    # Run threshold sweep on val split (for the analysis notebook)
    logger.info("Running threshold percentile sweep on val split...")
    val_errors = lstm_module.compute_reconstruction_errors(lstm_model, X_val_3d)
    thr_sweep = lstm_module.threshold_sweep(val_errors, y_val)
    thr_sweep_path = results_dir / "lstm_threshold_sweep.json"
    thr_sweep_path.write_text(json.dumps(thr_sweep, indent=2))
    logger.info("Threshold sweep saved → %s", thr_sweep_path)

    train_error_stats = {
        "mean": float(train_errors.mean()),
        "std": float(train_errors.std()),
        "p50": float(np.percentile(train_errors, 50)),
        "p95": float(np.percentile(train_errors, 95)),
        "p99": float(np.percentile(train_errors, 99)),
        "max": float(train_errors.max()),
    }

    lstm_module.save(
        lstm_model, threshold, models_dir,
        window_size=lstm_cfg["window_size"],
        threshold_percentile=lstm_cfg["threshold_percentile"],
        train_error_stats=train_error_stats,
    )
    logger.info("LSTM Autoencoder training complete.")
    logger.info("=== All training complete. Artifacts in %s ===", models_dir)


if __name__ == "__main__":
    main()

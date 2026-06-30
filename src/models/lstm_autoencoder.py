"""
LSTM Autoencoder for reconstruction-based time-series anomaly detection.

The model learns to reconstruct *normal* windows from the training split.
At inference time, windows the model cannot reconstruct well (high MSE)
are flagged as anomalies. The anomaly threshold is the N-th percentile of
reconstruction errors computed on the training set after fitting.

Architecture:
  Input [batch, window_size, 1]
      → Encoder LSTM (encoder_units) → RepeatVector
      → Decoder LSTM (decoder_units) → TimeDistributed Dense(1)
  Output [batch, window_size, 1]   (reconstruction)

Artifacts saved:
  models/<prefix>_lstm_autoencoder.keras  — the Keras model
  models/<prefix>_lstm_metadata.json      — threshold, window_size, scaler stats
"""

import json
import logging
from pathlib import Path

import numpy as np
import yaml

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Model construction
# ---------------------------------------------------------------------------

def build_model(
    window_size: int = 30,
    encoder_units: int = 64,
    bottleneck_units: int = 32,
    decoder_units: int = 64,
    dropout: float = 0.1,
    learning_rate: float = 0.001,
) -> "keras.Model":  # type: ignore[name-defined]
    """Build and compile the LSTM Autoencoder.

    The bottleneck forces the model to compress the normal temporal pattern
    into a low-dimensional representation. Anomalous patterns that differ
    from the learned normal distribution cannot be reconstructed accurately,
    resulting in high MSE — our anomaly signal.

    Args:
        window_size: Number of time steps per window (must match preprocessing).
        encoder_units: LSTM units in the encoder.
        bottleneck_units: Dense units in the bottleneck layer.
        decoder_units: LSTM units in the decoder.
        dropout: Dropout rate applied after LSTM layers.
        learning_rate: Adam optimizer learning rate.

    Returns:
        Compiled Keras model.
    """
    # Import here so the module can be imported even when TensorFlow is absent
    # (e.g., during unit tests that mock the model).
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers

    inputs = keras.Input(shape=(window_size, 1), name="input")

    # Encoder
    x = layers.LSTM(encoder_units, return_sequences=False, name="encoder_lstm")(inputs)
    x = layers.Dropout(dropout, name="encoder_dropout")(x)
    bottleneck = layers.Dense(bottleneck_units, activation="relu", name="bottleneck")(x)

    # Decoder — repeat bottleneck vector across time steps
    x = layers.RepeatVector(window_size, name="repeat")(bottleneck)
    x = layers.LSTM(decoder_units, return_sequences=True, name="decoder_lstm")(x)
    x = layers.Dropout(dropout, name="decoder_dropout")(x)
    outputs = layers.TimeDistributed(layers.Dense(1), name="reconstruction")(x)

    model = keras.Model(inputs, outputs, name="lstm_autoencoder")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate),
        loss="mse",
    )
    logger.info("Built LSTM Autoencoder: window=%d, enc=%d, bottle=%d, dec=%d",
                window_size, encoder_units, bottleneck_units, decoder_units)
    return model


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    model: "keras.Model",  # type: ignore[name-defined]
    windows_3d: np.ndarray,
    val_windows_3d: np.ndarray | None = None,
    epochs: int = 50,
    batch_size: int = 32,
    early_stopping_patience: int = 5,
    random_seed: int = 42,
) -> "keras.callbacks.History":  # type: ignore[name-defined]
    """Train the autoencoder on normal windows only.

    The model is trained to reconstruct its own input (unsupervised).
    Validation data is used only for early stopping — it should NOT include
    the anomaly labels to avoid any form of supervision leakage.

    Args:
        model: Compiled Keras model from build_model().
        windows_3d: Training windows, shape [N, window_size, 1]. Use
            *normal-only* windows (label==0) from the training split.
        val_windows_3d: Optional validation windows for early stopping.
        epochs: Maximum training epochs.
        batch_size: Mini-batch size.
        early_stopping_patience: Stop if val_loss doesn't improve for this
            many epochs.
        random_seed: NumPy and TF random seed for reproducibility.

    Returns:
        Keras History object (contains loss curves).
    """
    import tensorflow as tf
    from tensorflow import keras

    tf.random.set_seed(random_seed)
    np.random.seed(random_seed)

    callbacks = []
    if val_windows_3d is not None and early_stopping_patience > 0:
        callbacks.append(
            keras.callbacks.EarlyStopping(
                monitor="val_loss",
                patience=early_stopping_patience,
                restore_best_weights=True,
                verbose=1,
            )
        )

    validation_data = (
        (val_windows_3d, val_windows_3d) if val_windows_3d is not None else None
    )

    history = model.fit(
        windows_3d,
        windows_3d,          # target = input (autoencoder)
        epochs=epochs,
        batch_size=batch_size,
        validation_data=validation_data,
        callbacks=callbacks,
        verbose=1,
        shuffle=True,
    )

    logger.info(
        "Training complete. Best val_loss=%.6f at epoch %d",
        min(history.history.get("val_loss", history.history["loss"])),
        len(history.history["loss"]),
    )
    return history


# ---------------------------------------------------------------------------
# Reconstruction error & threshold
# ---------------------------------------------------------------------------

def compute_reconstruction_errors(
    model: "keras.Model",  # type: ignore[name-defined]
    windows_3d: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """Compute per-window mean squared reconstruction error.

    Args:
        model: Trained LSTM Autoencoder.
        windows_3d: Shape [N, window_size, 1].
        batch_size: Inference batch size.

    Returns:
        1-D float32 array of MSE values, one per window.
    """
    reconstructions = model.predict(windows_3d, batch_size=batch_size, verbose=0)
    errors = np.mean(np.square(windows_3d - reconstructions), axis=(1, 2)).astype(np.float32)
    return errors


def compute_threshold(
    train_errors: np.ndarray,
    percentile: float = 99.0,
) -> float:
    """Derive anomaly threshold as a percentile of training reconstruction errors.

    Any window whose error exceeds this threshold is flagged as an anomaly.
    The percentile choice controls the precision/recall tradeoff:
      - Higher percentile → fewer false positives, more false negatives
      - Lower percentile  → more false positives, fewer false negatives

    Args:
        train_errors: Reconstruction errors on the training split.
        percentile: Percentile value (0–100).

    Returns:
        Scalar threshold value.
    """
    threshold = float(np.percentile(train_errors, percentile))
    logger.info(
        "LSTM threshold set at %.1f-th percentile of train errors: %.6f",
        percentile,
        threshold,
    )
    return threshold


def score(
    model: "keras.Model",  # type: ignore[name-defined]
    windows_3d: np.ndarray,
    batch_size: int = 256,
) -> np.ndarray:
    """Return reconstruction error as the anomaly score (higher = more anomalous)."""
    return compute_reconstruction_errors(model, windows_3d, batch_size)


def predict(
    model: "keras.Model",  # type: ignore[name-defined]
    windows_3d: np.ndarray,
    threshold: float,
    batch_size: int = 256,
) -> np.ndarray:
    """Return binary anomaly flags based on reconstruction error threshold.

    Args:
        model: Trained LSTM Autoencoder.
        windows_3d: Shape [N, window_size, 1].
        threshold: Anomaly threshold from compute_threshold().
        batch_size: Inference batch size.

    Returns:
        1-D integer array: 1 = anomaly, 0 = normal.
    """
    errors = score(model, windows_3d, batch_size)
    return (errors > threshold).astype(np.int32)


# ---------------------------------------------------------------------------
# Threshold sweep (for threshold analysis notebook)
# ---------------------------------------------------------------------------

def threshold_sweep(
    errors_val: np.ndarray,
    labels_val: np.ndarray,
    percentiles: list[float] | None = None,
) -> list[dict]:
    """Compute precision/recall at multiple threshold percentiles.

    Args:
        errors_val: Reconstruction errors on the validation split.
        labels_val: Ground-truth labels for the validation split.
        percentiles: List of percentile values to try.

    Returns:
        List of dicts: {'percentile', 'threshold', 'precision', 'recall', 'f1', 'flag_rate'}
    """
    from sklearn.metrics import f1_score, precision_score, recall_score

    if percentiles is None:
        percentiles = [80, 85, 90, 92, 95, 97, 99, 99.5]

    results = []
    for pct in percentiles:
        threshold = float(np.percentile(errors_val, pct))
        preds = (errors_val > threshold).astype(np.int32)
        flag_rate = float(preds.mean())

        if labels_val.sum() == 0:
            prec = rec = f1 = float("nan")
        else:
            prec = float(precision_score(labels_val, preds, zero_division=0))
            rec = float(recall_score(labels_val, preds, zero_division=0))
            f1 = float(f1_score(labels_val, preds, zero_division=0))

        results.append({
            "percentile": pct,
            "threshold": threshold,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "flag_rate": flag_rate,
        })
        logger.info(
            "  pct=%.1f → threshold=%.6f, prec=%.3f, rec=%.3f, f1=%.3f, flagged=%.2f%%",
            pct, threshold, prec, rec, f1, flag_rate * 100,
        )

    return results


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save(
    model: "keras.Model",  # type: ignore[name-defined]
    threshold: float,
    models_dir: str | Path,
    prefix: str = "",
    window_size: int = 30,
    threshold_percentile: float = 99.0,
    train_error_stats: dict | None = None,
) -> dict[str, Path]:
    """Save the Keras model and threshold metadata.

    Args:
        model: Trained LSTM Autoencoder.
        threshold: Computed anomaly threshold value.
        models_dir: Directory to write artifacts into.
        prefix: Optional filename prefix.
        window_size: Window size used during training.
        threshold_percentile: Percentile used to derive the threshold.
        train_error_stats: Optional dict with mean/std of training errors
            for reference in the report.

    Returns:
        Dict with keys 'model' and 'metadata' pointing to saved paths.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    p = f"{prefix}_" if prefix else ""

    model_path = models_dir / f"{p}lstm_autoencoder.keras"
    meta_path = models_dir / f"{p}lstm_metadata.json"

    model.save(model_path)

    meta = {
        "threshold": threshold,
        "threshold_percentile": threshold_percentile,
        "window_size": window_size,
        "train_error_stats": train_error_stats or {},
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    logger.info("Saved LSTM model → %s", model_path)
    logger.info("Saved LSTM metadata → %s", meta_path)
    return {"model": model_path, "metadata": meta_path}


def load(
    models_dir: str | Path,
    prefix: str = "",
) -> tuple["keras.Model", dict]:  # type: ignore[name-defined]
    """Load a saved LSTM Autoencoder and its metadata.

    Returns:
        (Keras model, metadata dict containing 'threshold' and 'window_size')
    """
    import tensorflow as tf
    from tensorflow import keras

    models_dir = Path(models_dir)
    p = f"{prefix}_" if prefix else ""
    model_path = models_dir / f"{p}lstm_autoencoder.keras"
    meta_path = models_dir / f"{p}lstm_metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model = keras.models.load_model(model_path)
    meta: dict = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    logger.info("Loaded LSTM Autoencoder from %s (threshold=%.6f)", model_path, meta.get("threshold", float("nan")))
    return model, meta


def load_config(config_path: str | Path = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)

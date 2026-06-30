"""
Isolation Forest anomaly detector.

Trains on statistical feature vectors derived from sliding windows.
Anomaly scores are the raw IF decision_function output (more negative = more anomalous).
Binary flags are derived by thresholding at the contamination-implied boundary.

Artifacts saved:
  models/<prefix>_isolation_forest.joblib  — the fitted sklearn model
  models/<prefix>_if_metadata.json         — contamination, feature names, threshold
"""

import json
import logging
from pathlib import Path

import joblib
import numpy as np
import yaml
from sklearn.ensemble import IsolationForest
from sklearn.metrics import f1_score, precision_score, recall_score

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train(
    features: np.ndarray,
    n_estimators: int = 200,
    contamination: float = 0.05,
    max_features: float = 1.0,
    random_state: int = 42,
) -> IsolationForest:
    """Fit an Isolation Forest on the provided feature matrix.

    Args:
        features: Shape [N, n_features]. Should be the *training* split only.
        n_estimators: Number of trees.
        contamination: Assumed fraction of anomalies in the training data.
            Controls the decision boundary: setting it too high causes many
            false positives; too low causes misses.
        max_features: Fraction of features to consider per tree split.
        random_state: Reproducibility seed.

    Returns:
        Fitted IsolationForest estimator.
    """
    model = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_features=max_features,
        random_state=random_state,
        n_jobs=-1,
    )
    model.fit(features)
    logger.info(
        "Trained IsolationForest: n_estimators=%d, contamination=%.3f, features=%d",
        n_estimators,
        contamination,
        features.shape[1],
    )
    return model


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def score(model: IsolationForest, features: np.ndarray) -> np.ndarray:
    """Return raw anomaly scores for each window.

    sklearn's decision_function returns:
      - Large positive values → normal (far from the isolation boundary)
      - Negative values → anomalous (easily isolated)

    We negate so that *high* score means *more anomalous* (intuitive convention).

    Args:
        model: Fitted IsolationForest.
        features: Shape [N, n_features].

    Returns:
        1-D array of anomaly scores (higher = more anomalous).
    """
    raw = model.decision_function(features)
    return -raw.astype(np.float32)


def predict(model: IsolationForest, features: np.ndarray) -> np.ndarray:
    """Return binary anomaly flags (1 = anomaly, 0 = normal).

    sklearn's predict() returns +1 for normal and -1 for anomaly.
    We convert to the 0/1 convention used everywhere in this project.

    Args:
        model: Fitted IsolationForest.
        features: Shape [N, n_features].

    Returns:
        1-D integer array of 0s and 1s.
    """
    raw = model.predict(features)          # +1 = normal, -1 = anomaly
    return ((raw == -1).astype(np.int32))


# ---------------------------------------------------------------------------
# Contamination sweep
# ---------------------------------------------------------------------------

def contamination_sweep(
    features_train: np.ndarray,
    features_val: np.ndarray,
    labels_val: np.ndarray,
    contamination_values: list[float] | None = None,
    n_estimators: int = 200,
    random_state: int = 42,
) -> list[dict]:
    """Train IF at multiple contamination values and evaluate on the val split.

    This sweep produces the experimental data needed for the threshold analysis
    notebook — showing how precision/recall trade off as contamination changes.

    Args:
        features_train: Training feature matrix.
        features_val: Validation feature matrix.
        labels_val: Ground-truth labels for the validation split.
        contamination_values: List of contamination values to try.
        n_estimators: Number of trees (constant across sweep).
        random_state: Seed.

    Returns:
        List of result dicts, one per contamination value:
        {'contamination', 'precision', 'recall', 'f1', 'n_flagged'}
    """
    if contamination_values is None:
        contamination_values = [0.01, 0.03, 0.05, 0.08, 0.10, 0.15]

    results = []
    for c in contamination_values:
        model = train(features_train, n_estimators=n_estimators, contamination=c, random_state=random_state)
        preds = predict(model, features_val)
        n_flagged = int(preds.sum())

        # Handle all-zero ground truth gracefully (no labels in val split)
        if labels_val.sum() == 0:
            prec = rec = f1 = float("nan")
        else:
            prec = float(precision_score(labels_val, preds, zero_division=0))
            rec = float(recall_score(labels_val, preds, zero_division=0))
            f1 = float(f1_score(labels_val, preds, zero_division=0))

        results.append({
            "contamination": c,
            "precision": prec,
            "recall": rec,
            "f1": f1,
            "n_flagged": n_flagged,
            "flag_rate": n_flagged / len(preds),
        })
        logger.info(
            "  contamination=%.3f → precision=%.3f, recall=%.3f, f1=%.3f, flagged=%d",
            c, prec, rec, f1, n_flagged,
        )

    return results


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save(
    model: IsolationForest,
    models_dir: str | Path,
    prefix: str = "",
    feature_names: list[str] | None = None,
) -> dict[str, Path]:
    """Save the model and its metadata to disk.

    Args:
        model: Fitted IsolationForest.
        models_dir: Directory to write files into.
        prefix: Optional prefix for artifact filenames.
        feature_names: List of feature names for interpretability.

    Returns:
        Dict with keys 'model' and 'metadata' pointing to saved paths.
    """
    models_dir = Path(models_dir)
    models_dir.mkdir(parents=True, exist_ok=True)
    p = f"{prefix}_" if prefix else ""

    model_path = models_dir / f"{p}isolation_forest.joblib"
    meta_path = models_dir / f"{p}if_metadata.json"

    joblib.dump(model, model_path)

    meta = {
        "contamination": model.contamination,
        "n_estimators": model.n_estimators,
        "max_features": model.max_features,
        "feature_names": feature_names or [],
    }
    meta_path.write_text(json.dumps(meta, indent=2))

    logger.info("Saved IsolationForest model → %s", model_path)
    logger.info("Saved IF metadata → %s", meta_path)
    return {"model": model_path, "metadata": meta_path}


def load(
    models_dir: str | Path,
    prefix: str = "",
) -> tuple[IsolationForest, dict]:
    """Load a saved IsolationForest and its metadata.

    Returns:
        (fitted IsolationForest, metadata dict)
    """
    models_dir = Path(models_dir)
    p = f"{prefix}_" if prefix else ""
    model_path = models_dir / f"{p}isolation_forest.joblib"
    meta_path = models_dir / f"{p}if_metadata.json"

    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    model: IsolationForest = joblib.load(model_path)
    meta: dict = json.loads(meta_path.read_text()) if meta_path.exists() else {}

    logger.info("Loaded IsolationForest from %s", model_path)
    return model, meta


def load_config(config_path: str | Path = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)

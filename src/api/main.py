"""
Spike-Sense FastAPI application.

Endpoints:
  GET  /health            — liveness check
  GET  /info              — model metadata and system info
  POST /predict           — single-window anomaly inference
  POST /predict/batch     — multi-window batch inference
  GET  /evaluate          — pre-computed evaluation results
  POST /demo/inject-spike — demo: inject synthetic anomaly, score and alert

Run locally:
  uvicorn src.api.main:app --reload

The app loads all model artifacts at startup via the lifespan context manager.
No model loading happens per-request.
"""

from __future__ import annotations

import datetime
import logging
from contextlib import asynccontextmanager

import numpy as np
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from src.api.alerting import fire_alert, get_alert_stats, reset_alert_state
from src.api import database as db
from src.api.model_loader import model_registry
from src.api.schemas import (
    AlertRow,
    AlertsResponse,
    BatchPredictRequest,
    BatchPredictResponse,
    EvaluateResponse,
    HealthResponse,
    InfoResponse,
    InjectSpikeRequest,
    MetricRow,
    ModelScores,
    PredictRequest,
    PredictResponse,
    SpikeInjectionResponse,
    StatsResponse,
)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


# ---------------------------------------------------------------------------
# Startup / shutdown
# ---------------------------------------------------------------------------

@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("Loading models and scalers…")
    try:
        db.init_db()
        model_registry.load()
        logger.info("Startup complete.")
    except Exception as exc:
        logger.error("Startup initialisation failed: %s", exc)
    yield
    logger.info("Shutdown.")


app = FastAPI(
    title="Spike-Sense",
    description=(
        "AI-driven anomaly detection for cloud infrastructure metrics. "
        "Combines Isolation Forest and LSTM Autoencoder to detect unusual "
        "patterns in AWS CloudWatch time-series data."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _require_models() -> None:
    if not model_registry.loaded:
        raise HTTPException(status_code=503, detail="Models not yet loaded. Retry in a moment.")


def _score_window(window: list[float], series_key: str | None, persist: bool = True) -> PredictResponse:
    """Score a single window, persist flagged detections, and optionally fire an alert."""
    from src.data.preprocessor import _extract_window_features
    from src.models import isolation_forest as if_module
    from src.models import lstm_autoencoder as lstm_module

    raw = np.array(window, dtype=np.float64)
    scaler = model_registry.get_scaler(series_key)

    # Validate window length matches trained window size
    expected = model_registry.window_size
    if len(raw) != expected:
        raise HTTPException(
            status_code=422,
            detail=f"Window length {len(raw)} does not match model window_size {expected}.",
        )

    scaled = scaler.transform(raw.reshape(-1, 1)).flatten().astype(np.float32)

    # IF features
    window_2d = scaled.reshape(1, -1)
    features, _ = _extract_window_features(window_2d)

    # LSTM window
    window_3d = scaled.reshape(1, expected, 1)

    if_score = float(if_module.score(model_registry.if_model, features)[0])
    if_pred = bool(if_module.predict(model_registry.if_model, features)[0])

    lstm_error = float(lstm_module.score(model_registry.lstm_model, window_3d)[0])
    lstm_pred = bool(lstm_module.predict(model_registry.lstm_model, window_3d, model_registry.lstm_threshold)[0])

    combined_union = if_pred or lstm_pred
    combined_intersection = if_pred and lstm_pred

    alert_sent = False
    if combined_union and persist:
        alert_sent = fire_alert(
            timestamp=datetime.datetime.utcnow().isoformat(),
            value=float(raw[-1]),
            if_score=if_score,
            lstm_error=lstm_error,
            if_anomaly=if_pred,
            lstm_anomaly=lstm_pred,
            series_key=series_key,
        )

        # Persist the detection (and the alert it produced) to the database.
        try:
            pred_id = db.log_prediction(
                series_key=series_key,
                metric_value=float(raw[-1]),
                if_score=if_score,
                lstm_error=lstm_error,
                if_flag=if_pred,
                lstm_flag=lstm_pred,
                combined_union=combined_union,
                combined_intersection=combined_intersection,
            )
            db.log_alert(
                prediction_id=pred_id,
                series_key=series_key,
                metric_value=float(raw[-1]),
                if_score=if_score,
                lstm_error=lstm_error,
                if_flag=if_pred,
                lstm_flag=lstm_pred,
                sent=alert_sent,
            )
        except Exception as exc:  # persistence must never break inference
            logger.warning("Failed to persist detection: %s", exc)

    return PredictResponse(
        if_anomaly=if_pred,
        lstm_anomaly=lstm_pred,
        combined_union=combined_union,
        combined_intersection=combined_intersection,
        scores=ModelScores(
            isolation_forest=if_score,
            lstm_reconstruction_error=lstm_error,
        ),
        alert_sent=alert_sent,
        window_size=expected,
    )


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/health", response_model=HealthResponse, tags=["System"])
def health() -> HealthResponse:
    """Liveness check. Returns 200 when the service is up."""
    if model_registry.loaded:
        return HealthResponse(status="ok", models_loaded=True)
    return HealthResponse(status="degraded", models_loaded=False, detail="Models still loading.")


@app.get("/info", response_model=InfoResponse, tags=["System"])
def info() -> InfoResponse:
    """Return model metadata, threshold values, and available series."""
    _require_models()
    return InfoResponse(
        models_loaded=True,
        window_size=model_registry.window_size,
        if_contamination=float(model_registry.if_meta.get("contamination", 0)),
        lstm_threshold=model_registry.lstm_threshold,
        lstm_threshold_percentile=float(model_registry.lstm_meta.get("threshold_percentile", 99)),
        lstm_train_error_stats=model_registry.lstm_meta.get("train_error_stats", {}),
        series_available=model_registry.series_keys(),
    )


@app.get("/alerts", response_model=AlertsResponse, tags=["History"])
def alerts(limit: int = 20) -> AlertsResponse:
    """Return the most recent anomaly alerts persisted in the database."""
    limit = max(1, min(limit, 200))
    rows = db.recent_alerts(limit=limit)
    return AlertsResponse(alerts=[AlertRow(**r) for r in rows], count=len(rows))


@app.get("/stats", response_model=StatsResponse, tags=["History"])
def stats() -> StatsResponse:
    """Return aggregate detection/alert counts from the database."""
    return StatsResponse(**db.get_stats())


@app.post("/predict", response_model=PredictResponse, tags=["Inference"])
def predict(req: PredictRequest) -> PredictResponse:
    """Score a single metric window with both anomaly detectors.

    The window should contain raw (unscaled) metric values in chronological order.
    The API scales them internally using the per-series scaler.
    """
    _require_models()
    return _score_window(req.window, req.series_key)


@app.post("/predict/batch", response_model=BatchPredictResponse, tags=["Inference"])
def predict_batch(req: BatchPredictRequest) -> BatchPredictResponse:
    """Score multiple windows in a single call.

    Useful for the dashboard to score a full time-series in one request.

    Scoring is fully vectorised: every window is scaled, feature-extracted and
    run through both models in one batched pass each (one IF call, one LSTM
    forward pass), rather than looping per window. This turns a ~4k-window
    series scan from minutes into ~1s. Batch scoring is an exploratory scan, so
    it does not persist or alert per window — that would flood the history.
    Single /predict and /demo/inject-spike persist.
    """
    _require_models()

    from src.data.preprocessor import _extract_window_features
    from src.models import isolation_forest as if_module
    from src.models import lstm_autoencoder as lstm_module

    expected = model_registry.window_size

    if not req.windows:
        return BatchPredictResponse(
            predictions=[], n_windows=0,
            n_anomalies_union=0, n_anomalies_lstm=0, n_anomalies_if=0,
        )

    raw = np.array(req.windows, dtype=np.float64)
    if raw.ndim != 2 or raw.shape[1] != expected:
        raise HTTPException(
            status_code=422,
            detail=f"Each window must have length {expected}; got shape {raw.shape}.",
        )

    scaler = model_registry.get_scaler(req.series_key)
    # Scale all values at once, then restore (n_windows, window_size) shape.
    scaled = scaler.transform(raw.reshape(-1, 1)).reshape(raw.shape).astype(np.float32)

    features, _ = _extract_window_features(scaled)
    windows_3d = scaled[:, :, np.newaxis]

    if_scores = if_module.score(model_registry.if_model, features)
    if_preds = if_module.predict(model_registry.if_model, features)
    lstm_errors = lstm_module.score(model_registry.lstm_model, windows_3d)
    lstm_preds = lstm_module.predict(
        model_registry.lstm_model, windows_3d, model_registry.lstm_threshold
    )

    predictions = []
    for i in range(len(req.windows)):
        if_pred = bool(if_preds[i])
        lstm_pred = bool(lstm_preds[i])
        predictions.append(PredictResponse(
            if_anomaly=if_pred,
            lstm_anomaly=lstm_pred,
            combined_union=if_pred or lstm_pred,
            combined_intersection=if_pred and lstm_pred,
            scores=ModelScores(
                isolation_forest=float(if_scores[i]),
                lstm_reconstruction_error=float(lstm_errors[i]),
            ),
            alert_sent=False,
            window_size=expected,
        ))

    return BatchPredictResponse(
        predictions=predictions,
        n_windows=len(predictions),
        n_anomalies_union=sum(p.combined_union for p in predictions),
        n_anomalies_lstm=sum(p.lstm_anomaly for p in predictions),
        n_anomalies_if=sum(p.if_anomaly for p in predictions),
    )


@app.get("/evaluate", response_model=EvaluateResponse, tags=["Evaluation"])
def evaluate() -> EvaluateResponse:
    """Return pre-computed evaluation results from the last training run.

    Results include precision/recall/F1 for all four model configurations
    and per-scenario spike injection metrics.
    """
    from src.evaluation.evaluator import load_results

    try:
        results = load_results("results")
    except FileNotFoundError:
        raise HTTPException(
            status_code=404,
            detail="Evaluation results not found. Run scripts/evaluate.py first.",
        )

    test_eval = results.get("test_split_evaluation", {})
    metrics_raw = test_eval.get("metrics", [])
    spike_scenarios = results.get("spike_scenario_evaluation", [])

    metrics = [MetricRow(**m) for m in metrics_raw]

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

    return EvaluateResponse(
        metrics=metrics,
        dataset_stats=results.get("dataset_stats", {}),
        detector_config=results.get("model_config", {}),
        spike_scenario_summary=spike_summary,
    )


@app.post("/demo/inject-spike", response_model=SpikeInjectionResponse, tags=["Demo"])
def demo_inject_spike(req: InjectSpikeRequest) -> SpikeInjectionResponse:
    """Inject a synthetic anomaly into a loaded series and score the result.

    This endpoint is designed for live demonstration: it shows how the models
    respond to a controlled anomaly injected mid-series.
    """
    _require_models()

    from src.data.loader import load_all_series
    from src.data.preprocessor import _extract_window_features, create_windows
    from src.data.spike_injector import inject_spike, make_clean_copy
    from src.models import isolation_forest as if_module
    from src.models import lstm_autoencoder as lstm_module

    # Load the raw series
    all_series = load_all_series("config/config.yaml")
    if req.series_key not in all_series:
        raise HTTPException(
            status_code=404,
            detail=f"Series key '{req.series_key}' not found. Available: {list(all_series.keys())}",
        )

    df = make_clean_copy(all_series[req.series_key])
    inject_index = len(df) // 2

    df = inject_spike(
        df,
        index=inject_index,
        mode=req.mode,
        magnitude_sigma=req.magnitude_sigma,
        duration=req.duration,
    )

    scaler = model_registry.get_scaler(req.series_key)
    scaled = scaler.transform(df["value"].values.reshape(-1, 1)).flatten().astype(np.float32)
    labels = df["label"].values.astype(np.int32)
    timestamps = df["timestamp"].values

    windows, window_labels, _ = create_windows(
        scaled, labels, timestamps,
        window_size=model_registry.window_size,
        stride=1,
    )
    features, _ = _extract_window_features(windows)
    windows_3d = windows[:, :, np.newaxis]

    if_scores = if_module.score(model_registry.if_model, features)
    if_preds = if_module.predict(model_registry.if_model, features)
    lstm_errors = lstm_module.score(model_registry.lstm_model, windows_3d)
    lstm_preds = lstm_module.predict(model_registry.lstm_model, windows_3d, model_registry.lstm_threshold)

    predictions = []
    for i in range(len(windows)):
        union = bool(if_preds[i] or lstm_preds[i])
        inter = bool(if_preds[i] and lstm_preds[i])
        predictions.append(PredictResponse(
            if_anomaly=bool(if_preds[i]),
            lstm_anomaly=bool(lstm_preds[i]),
            combined_union=union,
            combined_intersection=inter,
            scores=ModelScores(
                isolation_forest=float(if_scores[i]),
                lstm_reconstruction_error=float(lstm_errors[i]),
            ),
            alert_sent=False,
            window_size=model_registry.window_size,
        ))

    # Fire alert for the first detected anomaly in the batch
    anomaly_indices = [i for i, p in enumerate(predictions) if p.combined_union]
    if anomaly_indices:
        reset_alert_state()
        first_i = anomaly_indices[0]
        last_step = first_i + model_registry.window_size - 1
        predictions[first_i].alert_sent = fire_alert(
            timestamp=str(timestamps[last_step]),
            value=float(df["value"].iloc[last_step]),
            if_score=float(if_scores[first_i]),
            lstm_error=float(lstm_errors[first_i]),
            if_anomaly=bool(if_preds[first_i]),
            lstm_anomaly=bool(lstm_preds[first_i]),
            series_key=req.series_key,
            cooldown_seconds=0,
        )

        # Persist the flagged detections and the alert that fired.
        try:
            pred_id = db.log_prediction(
                series_key=req.series_key,
                metric_value=float(df["value"].iloc[last_step]),
                if_score=float(if_scores[first_i]),
                lstm_error=float(lstm_errors[first_i]),
                if_flag=bool(if_preds[first_i]),
                lstm_flag=bool(lstm_preds[first_i]),
                combined_union=True,
                combined_intersection=bool(if_preds[first_i] and lstm_preds[first_i]),
            )
            db.log_alert(
                prediction_id=pred_id,
                series_key=req.series_key,
                metric_value=float(df["value"].iloc[last_step]),
                if_score=float(if_scores[first_i]),
                lstm_error=float(lstm_errors[first_i]),
                if_flag=bool(if_preds[first_i]),
                lstm_flag=bool(lstm_preds[first_i]),
                sent=predictions[first_i].alert_sent,
            )
        except Exception as exc:
            logger.warning("Failed to persist demo detection: %s", exc)

    batch = BatchPredictResponse(
        predictions=predictions,
        n_windows=len(predictions),
        n_anomalies_union=sum(p.combined_union for p in predictions),
        n_anomalies_lstm=sum(p.lstm_anomaly for p in predictions),
        n_anomalies_if=sum(p.if_anomaly for p in predictions),
    )

    return SpikeInjectionResponse(
        series_key=req.series_key,
        mode=req.mode,
        inject_index=inject_index,
        n_anomalous_windows=int(window_labels.sum()),
        predictions=batch,
    )

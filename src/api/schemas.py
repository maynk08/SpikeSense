"""
Pydantic request / response schemas for the Spike-Sense API.

All public endpoints use these types for input validation and response
serialisation.  FastAPI auto-generates OpenAPI documentation from them.
"""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field, field_validator


# ---------------------------------------------------------------------------
# Request schemas
# ---------------------------------------------------------------------------

class PredictRequest(BaseModel):
    """Single-window inference request."""

    window: list[float] = Field(
        ...,
        min_length=1,
        description="Ordered sequence of raw metric values for the window. "
                    "Must match the window_size the models were trained with (default 30).",
        examples=[[91.9, 94.8, 92.2, 93.7, 91.5, 90.1, 89.4, 92.0,
                   93.1, 91.8, 90.5, 88.9, 91.2, 92.4, 90.8, 89.6,
                   91.0, 92.5, 93.2, 91.7, 90.3, 89.8, 91.5, 92.1,
                   93.0, 91.4, 90.6, 89.9, 91.3, 92.6]],
    )
    series_key: str | None = Field(
        default=None,
        description="Optional NAB series key (e.g. 'realAWSCloudwatch/ec2_cpu_utilization_825cc2.csv'). "
                    "When provided, the API uses the scaler fitted on that series' training data. "
                    "Falls back to a generic scaler when omitted.",
    )

    @field_validator("window")
    @classmethod
    def window_must_not_be_empty(cls, v: list[float]) -> list[float]:
        if len(v) == 0:
            raise ValueError("window must contain at least one value")
        return v


class BatchPredictRequest(BaseModel):
    """Batch inference: multiple windows in one call."""

    windows: list[list[float]] = Field(
        ...,
        min_length=1,
        description="List of windows, each a list of raw metric values.",
    )
    series_key: str | None = Field(default=None)

    @field_validator("windows")
    @classmethod
    def all_windows_same_length(cls, v: list[list[float]]) -> list[list[float]]:
        if not v:
            raise ValueError("windows list must not be empty")
        lengths = {len(w) for w in v}
        if len(lengths) > 1:
            raise ValueError(f"All windows must have the same length. Found lengths: {sorted(lengths)}")
        return v


class InjectSpikeRequest(BaseModel):
    """Request body for the spike injection demo endpoint."""

    series_key: str = Field(
        default="realAWSCloudwatch/ec2_cpu_utilization_825cc2.csv",
        description="Which loaded series to inject into.",
    )
    mode: Literal["point_spike", "level_shift", "trend_drift"] = Field(
        default="point_spike",
        description="Type of synthetic anomaly to inject.",
    )
    magnitude_sigma: float = Field(default=4.0, ge=0.5, le=10.0)
    duration: int = Field(default=20, ge=1, le=200)


# ---------------------------------------------------------------------------
# Response schemas
# ---------------------------------------------------------------------------

class ModelScores(BaseModel):
    """Raw continuous scores from each model (higher = more anomalous)."""

    isolation_forest: float = Field(..., description="IF anomaly score (negated decision function).")
    lstm_reconstruction_error: float = Field(..., description="LSTM mean squared reconstruction error.")


class PredictResponse(BaseModel):
    """Inference response for a single window."""

    if_anomaly: bool = Field(..., description="Isolation Forest prediction.")
    lstm_anomaly: bool = Field(..., description="LSTM Autoencoder prediction.")
    combined_union: bool = Field(..., description="True if EITHER model flags an anomaly.")
    combined_intersection: bool = Field(..., description="True if BOTH models flag an anomaly.")
    scores: ModelScores
    alert_sent: bool = Field(default=False, description="Whether a Discord alert was triggered.")
    window_size: int = Field(..., description="Number of time steps in the scored window.")


class BatchPredictResponse(BaseModel):
    """Inference response for a batch of windows."""

    predictions: list[PredictResponse]
    n_windows: int
    n_anomalies_union: int
    n_anomalies_lstm: int
    n_anomalies_if: int


class MetricRow(BaseModel):
    model: str
    precision: float
    recall: float
    f1: float
    fpr: float
    tp: int
    fp: int
    tn: int
    fn: int
    flag_rate: float


class EvaluateResponse(BaseModel):
    """Pre-computed evaluation results served from results/evaluation_results.json."""

    metrics: list[MetricRow]
    dataset_stats: dict
    detector_config: dict
    spike_scenario_summary: list[dict]


class InfoResponse(BaseModel):
    """Model and system metadata."""

    status: str = "ok"
    models_loaded: bool
    window_size: int
    if_contamination: float
    lstm_threshold: float
    lstm_threshold_percentile: float
    lstm_train_error_stats: dict
    series_available: list[str]


class HealthResponse(BaseModel):
    status: Literal["ok", "degraded"]
    models_loaded: bool
    detail: str = ""


class SpikeInjectionResponse(BaseModel):
    """Response from the demo spike injection endpoint."""

    series_key: str
    mode: str
    inject_index: int
    n_anomalous_windows: int
    predictions: BatchPredictResponse


# ---------------------------------------------------------------------------
# History schemas (database-backed)
# ---------------------------------------------------------------------------

class AlertRow(BaseModel):
    """A single persisted alert record."""

    id: int
    prediction_id: int | None = None
    series_key: str | None = None
    metric_value: float
    detected_by: str = Field(..., description="Which models fired: 'IF', 'LSTM', or 'IF+LSTM'.")
    if_score: float
    lstm_error: float
    sent: bool = Field(..., description="Whether the Discord webhook delivery succeeded.")
    created_at: str | None = None


class AlertsResponse(BaseModel):
    """Recent alerts from the detection history."""

    alerts: list[AlertRow]
    count: int


class StatsResponse(BaseModel):
    """Aggregate counts over the persisted detection/alert history."""

    total_predictions: int
    total_alerts: int
    alerts_sent: int
    anomalies_detected: int

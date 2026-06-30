"""
Integration tests for the FastAPI application.

Uses FastAPI's TestClient (httpx) — no real server is started.
The model_registry is loaded once for the entire test session via a
session-scoped fixture.  Each test that needs a loaded registry uses
the `loaded_client` fixture.
"""

from __future__ import annotations

import pytest
from fastapi.testclient import TestClient

from src.api.alerting import reset_alert_state
from src.api.main import app
from src.api.model_loader import model_registry

WINDOW_SIZE = 30
VALID_WINDOW = [91.0 + i * 0.1 for i in range(WINDOW_SIZE)]
VALID_SERIES_KEY = "realAWSCloudwatch/ec2_cpu_utilization_825cc2.csv"


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture(scope="session")
def loaded_client():
    """TestClient with models pre-loaded — shared across all tests in the session."""
    if not model_registry.loaded:
        model_registry.load()
    reset_alert_state()
    with TestClient(app) as client:
        yield client


@pytest.fixture(autouse=True)
def _reset_alerts():
    """Reset alert state before every test to prevent cross-test contamination."""
    reset_alert_state()
    yield


# ---------------------------------------------------------------------------
# /health
# ---------------------------------------------------------------------------

class TestHealth:
    def test_returns_200(self, loaded_client):
        resp = loaded_client.get("/health")
        assert resp.status_code == 200

    def test_status_ok_when_loaded(self, loaded_client):
        data = resp = loaded_client.get("/health").json()
        assert data["status"] == "ok"
        assert data["models_loaded"] is True

    def test_response_schema(self, loaded_client):
        data = loaded_client.get("/health").json()
        assert "status" in data
        assert "models_loaded" in data


# ---------------------------------------------------------------------------
# /info
# ---------------------------------------------------------------------------

class TestInfo:
    def test_returns_200(self, loaded_client):
        resp = loaded_client.get("/info")
        assert resp.status_code == 200

    def test_contains_window_size(self, loaded_client):
        data = loaded_client.get("/info").json()
        assert data["window_size"] == WINDOW_SIZE

    def test_lstm_threshold_is_positive(self, loaded_client):
        data = loaded_client.get("/info").json()
        assert data["lstm_threshold"] > 0

    def test_series_available_is_list(self, loaded_client):
        data = loaded_client.get("/info").json()
        assert isinstance(data["series_available"], list)
        assert len(data["series_available"]) > 0

    def test_if_contamination_in_range(self, loaded_client):
        data = loaded_client.get("/info").json()
        assert 0 < data["if_contamination"] < 1


# ---------------------------------------------------------------------------
# POST /predict
# ---------------------------------------------------------------------------

class TestPredict:
    def test_valid_request_returns_200(self, loaded_client):
        resp = loaded_client.post("/predict", json={"window": VALID_WINDOW})
        assert resp.status_code == 200

    def test_response_has_required_fields(self, loaded_client):
        data = loaded_client.post("/predict", json={"window": VALID_WINDOW}).json()
        required = {"if_anomaly", "lstm_anomaly", "combined_union", "combined_intersection",
                    "scores", "alert_sent", "window_size"}
        assert required.issubset(data.keys())

    def test_scores_are_floats(self, loaded_client):
        data = loaded_client.post("/predict", json={"window": VALID_WINDOW}).json()
        assert isinstance(data["scores"]["isolation_forest"], float)
        assert isinstance(data["scores"]["lstm_reconstruction_error"], float)

    def test_combined_union_is_bool(self, loaded_client):
        data = loaded_client.post("/predict", json={"window": VALID_WINDOW}).json()
        assert isinstance(data["combined_union"], bool)

    def test_combined_union_implies_either_flag(self, loaded_client):
        data = loaded_client.post("/predict", json={"window": VALID_WINDOW}).json()
        if data["combined_union"]:
            assert data["if_anomaly"] or data["lstm_anomaly"]

    def test_window_size_in_response_matches_model(self, loaded_client):
        data = loaded_client.post("/predict", json={"window": VALID_WINDOW}).json()
        assert data["window_size"] == WINDOW_SIZE

    def test_with_explicit_series_key(self, loaded_client):
        resp = loaded_client.post("/predict", json={
            "window": VALID_WINDOW,
            "series_key": VALID_SERIES_KEY,
        })
        assert resp.status_code == 200

    def test_wrong_window_size_returns_422(self, loaded_client):
        too_short = [1.0] * 10
        resp = loaded_client.post("/predict", json={"window": too_short})
        assert resp.status_code == 422

    def test_empty_window_returns_422(self, loaded_client):
        resp = loaded_client.post("/predict", json={"window": []})
        assert resp.status_code == 422

    def test_very_high_window_triggers_anomaly(self, loaded_client):
        # All values at 999 — far outside any trained distribution.
        # Constant window means IF features are degenerate (std=0, range=0),
        # but the LSTM reconstruction error is very high.
        extreme = [999.0] * WINDOW_SIZE
        data = loaded_client.post("/predict", json={"window": extreme}).json()
        # At least one detector should flag this as anomalous
        assert data["if_anomaly"] or data["lstm_anomaly"]

    def test_lstm_score_is_non_negative(self, loaded_client):
        data = loaded_client.post("/predict", json={"window": VALID_WINDOW}).json()
        assert data["scores"]["lstm_reconstruction_error"] >= 0.0


# ---------------------------------------------------------------------------
# POST /predict/batch
# ---------------------------------------------------------------------------

class TestPredictBatch:
    def test_valid_batch_returns_200(self, loaded_client):
        resp = loaded_client.post("/predict/batch", json={"windows": [VALID_WINDOW, VALID_WINDOW]})
        assert resp.status_code == 200

    def test_response_counts_match(self, loaded_client):
        n = 5
        resp = loaded_client.post("/predict/batch", json={"windows": [VALID_WINDOW] * n})
        data = resp.json()
        assert data["n_windows"] == n
        assert len(data["predictions"]) == n

    def test_anomaly_counts_are_consistent(self, loaded_client):
        resp = loaded_client.post("/predict/batch", json={"windows": [VALID_WINDOW] * 3})
        data = resp.json()
        preds = data["predictions"]
        assert data["n_anomalies_union"] == sum(p["combined_union"] for p in preds)
        assert data["n_anomalies_if"] == sum(p["if_anomaly"] for p in preds)
        assert data["n_anomalies_lstm"] == sum(p["lstm_anomaly"] for p in preds)

    def test_inconsistent_window_lengths_returns_422(self, loaded_client):
        resp = loaded_client.post("/predict/batch", json={
            "windows": [VALID_WINDOW, [1.0] * 10]
        })
        assert resp.status_code == 422

    def test_empty_windows_list_returns_422(self, loaded_client):
        resp = loaded_client.post("/predict/batch", json={"windows": []})
        assert resp.status_code == 422

    def test_extreme_batch_has_anomalies(self, loaded_client):
        # Constant 999 windows: LSTM catches them even if IF features are degenerate
        extreme = [999.0] * WINDOW_SIZE
        resp = loaded_client.post("/predict/batch", json={"windows": [extreme, extreme]})
        data = resp.json()
        # At least one detector should flag both windows
        assert data["n_anomalies_union"] == 2


# ---------------------------------------------------------------------------
# GET /evaluate
# ---------------------------------------------------------------------------

class TestEvaluate:
    def test_returns_200(self, loaded_client):
        resp = loaded_client.get("/evaluate")
        assert resp.status_code == 200

    def test_four_metric_rows(self, loaded_client):
        data = loaded_client.get("/evaluate").json()
        assert len(data["metrics"]) == 4

    def test_metric_row_has_required_fields(self, loaded_client):
        data = loaded_client.get("/evaluate").json()
        row = data["metrics"][0]
        for field in ("model", "precision", "recall", "f1", "fpr", "tp", "fp", "tn", "fn"):
            assert field in row

    def test_model_names_correct(self, loaded_client):
        data = loaded_client.get("/evaluate").json()
        names = {m["model"] for m in data["metrics"]}
        assert "Isolation Forest" in names
        assert "LSTM Autoencoder" in names

    def test_spike_scenario_summary_has_three_scenarios(self, loaded_client):
        data = loaded_client.get("/evaluate").json()
        scenarios = {s["scenario"] for s in data["spike_scenario_summary"]}
        assert {"point_spike", "level_shift", "trend_drift"}.issubset(scenarios)

    def test_dataset_stats_present(self, loaded_client):
        data = loaded_client.get("/evaluate").json()
        assert "n_test_windows" in data["dataset_stats"]

    def test_detector_config_present(self, loaded_client):
        data = loaded_client.get("/evaluate").json()
        assert "lstm_threshold" in data["detector_config"]


# ---------------------------------------------------------------------------
# POST /demo/inject-spike
# ---------------------------------------------------------------------------

class TestDemoInjectSpike:
    def test_valid_request_returns_200(self, loaded_client):
        resp = loaded_client.post("/demo/inject-spike", json={
            "series_key": VALID_SERIES_KEY,
            "mode": "point_spike",
        })
        assert resp.status_code == 200

    def test_response_has_predictions(self, loaded_client):
        resp = loaded_client.post("/demo/inject-spike", json={
            "series_key": VALID_SERIES_KEY,
            "mode": "point_spike",
        })
        data = resp.json()
        assert "predictions" in data
        assert data["predictions"]["n_windows"] > 0

    def test_inject_spike_creates_anomalous_windows(self, loaded_client):
        resp = loaded_client.post("/demo/inject-spike", json={
            "series_key": VALID_SERIES_KEY,
            "mode": "point_spike",
            "magnitude_sigma": 8.0,
        })
        data = resp.json()
        assert data["n_anomalous_windows"] > 0

    def test_invalid_series_key_returns_404(self, loaded_client):
        resp = loaded_client.post("/demo/inject-spike", json={
            "series_key": "nonexistent/series.csv",
            "mode": "point_spike",
        })
        assert resp.status_code == 404

    def test_all_spike_modes_accepted(self, loaded_client):
        for mode in ("point_spike", "level_shift", "trend_drift"):
            resp = loaded_client.post("/demo/inject-spike", json={
                "series_key": VALID_SERIES_KEY,
                "mode": mode,
            })
            assert resp.status_code == 200, f"Failed for mode={mode}"


# ---------------------------------------------------------------------------
# /alerts and /stats (database-backed history)
# ---------------------------------------------------------------------------

class TestHistory:
    def test_stats_returns_200_and_schema(self, loaded_client):
        resp = loaded_client.get("/stats")
        assert resp.status_code == 200
        data = resp.json()
        for key in ("total_predictions", "total_alerts", "alerts_sent", "anomalies_detected"):
            assert key in data
            assert isinstance(data[key], int)

    def test_alerts_returns_list(self, loaded_client):
        resp = loaded_client.get("/alerts")
        assert resp.status_code == 200
        data = resp.json()
        assert "alerts" in data and "count" in data
        assert isinstance(data["alerts"], list)

    def test_injecting_spike_persists_an_alert(self, loaded_client):
        before = loaded_client.get("/stats").json()["total_alerts"]
        # A strong point spike reliably produces a detection + alert.
        loaded_client.post("/demo/inject-spike", json={
            "series_key": VALID_SERIES_KEY,
            "mode": "point_spike",
            "magnitude_sigma": 8.0,
        })
        after = loaded_client.get("/stats").json()["total_alerts"]
        assert after >= before + 1

        alerts = loaded_client.get("/alerts?limit=5").json()["alerts"]
        assert len(alerts) >= 1
        assert alerts[0]["detected_by"] in {"IF", "LSTM", "IF+LSTM"}

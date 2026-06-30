"""
HTTP client for the Spike-Sense FastAPI backend.

All dashboard → API calls go through this module so that the Streamlit app
never constructs raw HTTP requests directly.  Error handling and retries are
centralised here.
"""

from __future__ import annotations

import logging
import os

import requests

logger = logging.getLogger(__name__)

# Base URL is read from env var so it works both locally and in production.
# Default points at the local dev server.
_DEFAULT_BASE_URL = "http://localhost:8000"

# Deployment mode:
#   - If SPIKE_SENSE_API_URL is set, the dashboard talks to a remote FastAPI
#     backend over HTTP (the original two-service architecture).
#   - If it is NOT set, the dashboard scores models in-process via
#     dashboard.local_scoring (single-service deploy, e.g. Streamlit Cloud).
# This lets the same code run standalone for demos and against an API in prod.
_USE_LOCAL = "SPIKE_SENSE_API_URL" not in os.environ


def _base_url() -> str:
    return os.environ.get("SPIKE_SENSE_API_URL", _DEFAULT_BASE_URL).rstrip("/")


def _get(path: str, timeout: int = 10) -> dict | None:
    try:
        resp = requests.get(f"{_base_url()}{path}", timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        logger.error("Cannot reach API at %s — is the backend running?", _base_url())
        return None
    except requests.RequestException as exc:
        logger.error("GET %s failed: %s", path, exc)
        return None


def _post(path: str, payload: dict, timeout: int = 30) -> dict | None:
    try:
        resp = requests.post(f"{_base_url()}{path}", json=payload, timeout=timeout)
        resp.raise_for_status()
        return resp.json()
    except requests.ConnectionError:
        logger.error("Cannot reach API at %s — is the backend running?", _base_url())
        return None
    except requests.RequestException as exc:
        logger.error("POST %s failed: %s", path, exc)
        return None


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def health() -> dict | None:
    """GET /health"""
    return _get("/health")


def info() -> dict | None:
    """GET /info"""
    if _USE_LOCAL:
        from dashboard import local_scoring
        return local_scoring.info()
    return _get("/info")


def predict_batch(windows: list[list[float]], series_key: str | None = None) -> dict | None:
    """POST /predict/batch"""
    if _USE_LOCAL:
        from dashboard import local_scoring
        return local_scoring.predict_batch(windows, series_key=series_key)
    payload: dict = {"windows": windows}
    if series_key:
        payload["series_key"] = series_key
    return _post("/predict/batch", payload, timeout=60)


def evaluate() -> dict | None:
    """GET /evaluate"""
    if _USE_LOCAL:
        from dashboard import local_scoring
        return local_scoring.evaluate()
    return _get("/evaluate")


def inject_spike(
    series_key: str,
    mode: str = "point_spike",
    magnitude_sigma: float = 4.0,
    duration: int = 20,
) -> dict | None:
    """POST /demo/inject-spike"""
    if _USE_LOCAL:
        from dashboard import local_scoring
        return local_scoring.inject_spike(series_key, mode, magnitude_sigma, duration)
    return _post("/demo/inject-spike", {
        "series_key": series_key,
        "mode": mode,
        "magnitude_sigma": magnitude_sigma,
        "duration": duration,
    }, timeout=60)


def is_reachable() -> bool:
    if _USE_LOCAL:
        from dashboard import local_scoring
        return local_scoring.is_reachable()
    result = health()
    return result is not None and result.get("status") == "ok"

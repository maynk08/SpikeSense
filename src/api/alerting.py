"""
Discord webhook alerting with cooldown.

When the scoring engine detects a combined anomaly it calls fire_alert().
A configurable cooldown prevents flooding: if an alert was sent within
the last N seconds the new alert is silently skipped and the function
returns False.

The webhook URL is read from the environment variable DISCORD_WEBHOOK_URL.
If the variable is absent the alert is logged but not sent (safe for local
development without a real Discord server).
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass, field

import requests

logger = logging.getLogger(__name__)

_WEBHOOK_ENV_VAR = "DISCORD_WEBHOOK_URL"
_REQUEST_TIMEOUT_SECONDS = 5


@dataclass
class AlertState:
    """Tracks the last alert time to enforce cooldown."""
    last_alert_time: float = 0.0
    total_sent: int = 0
    total_skipped: int = 0


# Module-level state shared across all requests in the same process
_alert_state = AlertState()


def _format_message(
    timestamp: str,
    value: float,
    if_score: float,
    lstm_error: float,
    if_anomaly: bool,
    lstm_anomaly: bool,
    series_key: str | None,
) -> str:
    models_flagged = []
    if if_anomaly:
        models_flagged.append(f"Isolation Forest (score={if_score:.4f})")
    if lstm_anomaly:
        models_flagged.append(f"LSTM Autoencoder (error={lstm_error:.6f})")

    series_label = series_key.split("/")[-1].replace(".csv", "") if series_key else "unknown"
    flagged_str = " + ".join(models_flagged) if models_flagged else "combined signal"

    return (
        f"**Spike-Sense Anomaly Alert**\n"
        f"Series: `{series_label}`\n"
        f"Metric value: `{value:.4f}`\n"
        f"Detected by: {flagged_str}\n"
        f"Timestamp: `{timestamp}`"
    )


def fire_alert(
    timestamp: str,
    value: float,
    if_score: float,
    lstm_error: float,
    if_anomaly: bool,
    lstm_anomaly: bool,
    series_key: str | None = None,
    cooldown_seconds: int = 60,
) -> bool:
    """Send a Discord webhook alert if the cooldown has elapsed.

    Args:
        timestamp: ISO timestamp string for the anomaly.
        value: Raw (unscaled) metric value at the anomaly point.
        if_score: IF anomaly score.
        lstm_error: LSTM reconstruction error.
        if_anomaly: Whether IF flagged this window.
        lstm_anomaly: Whether LSTM flagged this window.
        series_key: NAB series identifier for context.
        cooldown_seconds: Minimum seconds between consecutive alerts.

    Returns:
        True if an alert was successfully sent, False if skipped or failed.
    """
    now = time.time()
    elapsed = now - _alert_state.last_alert_time

    if elapsed < cooldown_seconds:
        remaining = cooldown_seconds - elapsed
        logger.debug("Alert skipped — cooldown active (%.0fs remaining).", remaining)
        _alert_state.total_skipped += 1
        return False

    message = _format_message(timestamp, value, if_score, lstm_error, if_anomaly, lstm_anomaly, series_key)

    webhook_url = os.environ.get(_WEBHOOK_ENV_VAR, "")
    if not webhook_url:
        logger.warning(
            "Alert not sent — %s not set. Message would have been:\n%s",
            _WEBHOOK_ENV_VAR, message,
        )
        # Update state so cooldown still applies even without a real webhook
        _alert_state.last_alert_time = now
        return False

    try:
        response = requests.post(
            webhook_url,
            json={"content": message},
            timeout=_REQUEST_TIMEOUT_SECONDS,
        )
        response.raise_for_status()
        _alert_state.last_alert_time = now
        _alert_state.total_sent += 1
        logger.info("Discord alert sent (total sent: %d).", _alert_state.total_sent)
        return True
    except requests.RequestException as exc:
        logger.error("Failed to send Discord alert: %s", exc)
        return False


def get_alert_stats() -> dict:
    """Return current alert state for the /info endpoint."""
    return {
        "total_sent": _alert_state.total_sent,
        "total_skipped": _alert_state.total_skipped,
        "last_alert_time": _alert_state.last_alert_time,
        "cooldown_active": (time.time() - _alert_state.last_alert_time) < 60,
    }


def reset_alert_state() -> None:
    """Reset alert state — used in tests to avoid cross-test contamination."""
    _alert_state.last_alert_time = 0.0
    _alert_state.total_sent = 0
    _alert_state.total_skipped = 0

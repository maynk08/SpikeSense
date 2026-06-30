"""
SQLite persistence layer for Spike-Sense.

Two tables capture the runtime history of the scoring engine:

  predictions  — every window the engine flags as anomalous (combined_union).
                 Acts as a detection log: which series, what scores, which
                 models fired, and when.
  alerts       — the subset of detections that actually sent a Discord
                 notification. Cooldown means not every detection notifies,
                 so a prediction has zero-or-many alerts (one-to-many).

Relationship:  predictions (1) ──< (0..N) alerts   [alerts.prediction_id FK]

The database URL is resolved in this order:
  1. env var SPIKE_SENSE_DB_URL   (e.g. sqlite:///:memory: in tests)
  2. config/config.yaml -> api.db_path
  3. default: sqlite:///data/spike_sense.db

SQLite is used for zero-cost, file-based, dependency-light persistence that
commits to the repo and runs anywhere — fitting the project's free-tier hosting.
"""

from __future__ import annotations

import datetime as _dt
import logging
import os
from pathlib import Path

import yaml
from sqlalchemy import (
    Boolean,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    String,
    create_engine,
    func,
    select,
)
from sqlalchemy.orm import (
    DeclarativeBase,
    Mapped,
    mapped_column,
    relationship,
    sessionmaker,
)

logger = logging.getLogger(__name__)

_DB_ENV_VAR = "SPIKE_SENSE_DB_URL"
_DEFAULT_DB_PATH = "data/spike_sense.db"


def _utcnow() -> _dt.datetime:
    return _dt.datetime.now(_dt.timezone.utc)


def _resolve_db_url(config_path: str | Path = "config/config.yaml") -> str:
    """Pick the database URL from env var, then config, then default."""
    env_url = os.environ.get(_DB_ENV_VAR)
    if env_url:
        return env_url

    db_path = _DEFAULT_DB_PATH
    try:
        with open(config_path) as f:
            cfg = yaml.safe_load(f) or {}
        db_path = cfg.get("api", {}).get("db_path", _DEFAULT_DB_PATH)
    except FileNotFoundError:
        pass

    # Ensure the parent directory exists for a file-based SQLite DB.
    parent = Path(db_path).parent
    if parent and not parent.exists():
        parent.mkdir(parents=True, exist_ok=True)
    return f"sqlite:///{db_path}"


# ---------------------------------------------------------------------------
# ORM models
# ---------------------------------------------------------------------------

class Base(DeclarativeBase):
    pass


class Prediction(Base):
    """One flagged window scored by the engine (an anomaly detection record)."""

    __tablename__ = "predictions"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    series_key: Mapped[str | None] = mapped_column(String(128), index=True, nullable=True)
    metric_value: Mapped[float] = mapped_column(Float)          # last raw value in the window
    if_score: Mapped[float] = mapped_column(Float)              # Isolation Forest anomaly score
    lstm_error: Mapped[float] = mapped_column(Float)            # LSTM reconstruction error
    if_flag: Mapped[bool] = mapped_column(Boolean)
    lstm_flag: Mapped[bool] = mapped_column(Boolean)
    combined_union: Mapped[bool] = mapped_column(Boolean)
    combined_intersection: Mapped[bool] = mapped_column(Boolean)
    created_at: Mapped[_dt.datetime] = mapped_column(DateTime, default=_utcnow, index=True)

    alerts: Mapped[list["Alert"]] = relationship(
        back_populates="prediction", cascade="all, delete-orphan"
    )


class Alert(Base):
    """A Discord notification that was actually sent for a detection."""

    __tablename__ = "alerts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    prediction_id: Mapped[int | None] = mapped_column(
        ForeignKey("predictions.id"), index=True, nullable=True
    )
    series_key: Mapped[str | None] = mapped_column(String(128), index=True, nullable=True)
    metric_value: Mapped[float] = mapped_column(Float)
    detected_by: Mapped[str] = mapped_column(String(32))       # "IF", "LSTM", or "IF+LSTM"
    if_score: Mapped[float] = mapped_column(Float)
    lstm_error: Mapped[float] = mapped_column(Float)
    sent: Mapped[bool] = mapped_column(Boolean)                # Discord delivery succeeded
    created_at: Mapped[_dt.datetime] = mapped_column(DateTime, default=_utcnow, index=True)

    prediction: Mapped["Prediction | None"] = relationship(back_populates="alerts")


# ---------------------------------------------------------------------------
# Engine / session management
# ---------------------------------------------------------------------------

_engine = None
_SessionLocal: sessionmaker | None = None


def init_db(config_path: str | Path = "config/config.yaml") -> None:
    """Create the engine (if needed) and ensure tables exist. Idempotent."""
    global _engine, _SessionLocal
    if _engine is None:
        url = _resolve_db_url(config_path)
        # check_same_thread=False lets the FastAPI threadpool share the SQLite file.
        connect_args = {"check_same_thread": False} if url.startswith("sqlite") else {}
        _engine = create_engine(url, connect_args=connect_args, future=True)
        _SessionLocal = sessionmaker(bind=_engine, expire_on_commit=False, future=True)
        logger.info("Database initialised at %s", url)
    Base.metadata.create_all(_engine)


def _session():
    if _SessionLocal is None:
        init_db()
    return _SessionLocal()  # type: ignore[misc]


def _detected_by(if_flag: bool, lstm_flag: bool) -> str:
    parts = []
    if if_flag:
        parts.append("IF")
    if lstm_flag:
        parts.append("LSTM")
    return "+".join(parts) if parts else "none"


# ---------------------------------------------------------------------------
# Write helpers
# ---------------------------------------------------------------------------

def log_prediction(
    *,
    series_key: str | None,
    metric_value: float,
    if_score: float,
    lstm_error: float,
    if_flag: bool,
    lstm_flag: bool,
    combined_union: bool,
    combined_intersection: bool,
) -> int:
    """Persist a flagged-window detection. Returns the new prediction id."""
    with _session() as s:
        row = Prediction(
            series_key=series_key,
            metric_value=metric_value,
            if_score=if_score,
            lstm_error=lstm_error,
            if_flag=if_flag,
            lstm_flag=lstm_flag,
            combined_union=combined_union,
            combined_intersection=combined_intersection,
        )
        s.add(row)
        s.commit()
        return row.id


def log_alert(
    *,
    prediction_id: int | None,
    series_key: str | None,
    metric_value: float,
    if_score: float,
    lstm_error: float,
    if_flag: bool,
    lstm_flag: bool,
    sent: bool,
) -> int:
    """Persist a notification record linked to a detection. Returns alert id."""
    with _session() as s:
        row = Alert(
            prediction_id=prediction_id,
            series_key=series_key,
            metric_value=metric_value,
            detected_by=_detected_by(if_flag, lstm_flag),
            if_score=if_score,
            lstm_error=lstm_error,
            sent=sent,
        )
        s.add(row)
        s.commit()
        return row.id


# ---------------------------------------------------------------------------
# Read helpers
# ---------------------------------------------------------------------------

def recent_alerts(limit: int = 20) -> list[dict]:
    """Return the most recent alerts, newest first, as plain dicts."""
    with _session() as s:
        rows = s.scalars(
            select(Alert).order_by(Alert.created_at.desc(), Alert.id.desc()).limit(limit)
        ).all()
        return [
            {
                "id": r.id,
                "prediction_id": r.prediction_id,
                "series_key": r.series_key,
                "metric_value": r.metric_value,
                "detected_by": r.detected_by,
                "if_score": r.if_score,
                "lstm_error": r.lstm_error,
                "sent": r.sent,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]


def recent_predictions(limit: int = 50) -> list[dict]:
    """Return the most recent flagged detections, newest first."""
    with _session() as s:
        rows = s.scalars(
            select(Prediction).order_by(Prediction.created_at.desc(), Prediction.id.desc()).limit(limit)
        ).all()
        return [
            {
                "id": r.id,
                "series_key": r.series_key,
                "metric_value": r.metric_value,
                "if_score": r.if_score,
                "lstm_error": r.lstm_error,
                "if_flag": r.if_flag,
                "lstm_flag": r.lstm_flag,
                "combined_union": r.combined_union,
                "combined_intersection": r.combined_intersection,
                "created_at": r.created_at.isoformat() if r.created_at else None,
            }
            for r in rows
        ]


def get_stats() -> dict:
    """Aggregate counts for the /stats endpoint and dashboard."""
    with _session() as s:
        n_predictions = s.scalar(select(func.count()).select_from(Prediction)) or 0
        n_alerts = s.scalar(select(func.count()).select_from(Alert)) or 0
        n_sent = s.scalar(select(func.count()).select_from(Alert).where(Alert.sent.is_(True))) or 0
        n_union = s.scalar(
            select(func.count()).select_from(Prediction).where(Prediction.combined_union.is_(True))
        ) or 0
        return {
            "total_predictions": int(n_predictions),
            "total_alerts": int(n_alerts),
            "alerts_sent": int(n_sent),
            "anomalies_detected": int(n_union),
        }


def reset_db() -> None:
    """Drop and recreate all tables. Used in tests."""
    if _engine is None:
        init_db()
    Base.metadata.drop_all(_engine)
    Base.metadata.create_all(_engine)

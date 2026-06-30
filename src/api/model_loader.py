"""
Loads all model artifacts and per-series scalers at API startup.

Everything is loaded once into a singleton ModelRegistry instance and
shared across all requests. This avoids re-loading multi-MB model files
on every request, keeping latency low.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import numpy as np
import yaml
from sklearn.preprocessing import MinMaxScaler

logger = logging.getLogger(__name__)


class ModelRegistry:
    """Singleton holding all loaded model artifacts and scalers."""

    def __init__(self) -> None:
        self.if_model = None
        self.if_meta: dict = {}
        self.lstm_model = None
        self.lstm_meta: dict = {}
        self.lstm_threshold: float = 0.0
        self.window_size: int = 30
        # Per-series scalers: maps series_key → fitted MinMaxScaler
        self.scalers: dict[str, MinMaxScaler] = {}
        self._loaded: bool = False

    @property
    def loaded(self) -> bool:
        return self._loaded

    def load(self, config_path: str | Path = "config/config.yaml") -> None:
        """Load all model artifacts and rebuild per-series scalers.

        Scalers are rebuilt by re-fitting on the training split of each
        configured series. This is fast (< 1s) and avoids persisting scaler
        pickle files separately.

        Args:
            config_path: Path to config.yaml.
        """
        from src.models import isolation_forest as if_module
        from src.models import lstm_autoencoder as lstm_module
        from src.data.loader import load_all_series
        from src.data.splitter import split_series
        from src.data.preprocessor import fit_scaler

        config_path = Path(config_path)
        with open(config_path) as f:
            cfg = yaml.safe_load(f)

        models_dir = Path(cfg["api"]["models_dir"])
        self.window_size = cfg["preprocessing"]["window_size"]
        scaler_type = cfg["preprocessing"]["scaler"]
        split_cfg = cfg["data"]["split"]

        logger.info("Loading Isolation Forest…")
        self.if_model, self.if_meta = if_module.load(models_dir)

        logger.info("Loading LSTM Autoencoder…")
        self.lstm_model, self.lstm_meta = lstm_module.load(models_dir)
        self.lstm_threshold = float(self.lstm_meta["threshold"])

        logger.info("Rebuilding per-series scalers…")
        all_series = load_all_series(config_path)
        for key, df in all_series.items():
            split = split_series(df, split_cfg["train"], split_cfg["val"], split_cfg["test"])
            scaler = fit_scaler(split.train["value"].values.astype(np.float64), scaler_type)
            self.scalers[key] = scaler
            logger.info("  Scaler ready for: %s", key.split("/")[-1])

        self._loaded = True
        logger.info(
            "ModelRegistry ready. IF contamination=%.3f, LSTM threshold=%.6f, %d scalers.",
            self.if_meta.get("contamination", 0),
            self.lstm_threshold,
            len(self.scalers),
        )

    def get_scaler(self, series_key: str | None) -> MinMaxScaler:
        """Return the scaler for the given series key, or the first available."""
        if series_key and series_key in self.scalers:
            return self.scalers[series_key]
        if self.scalers:
            fallback_key = next(iter(self.scalers))
            logger.debug("Series key '%s' not found; using fallback scaler for '%s'.", series_key, fallback_key)
            return self.scalers[fallback_key]
        raise RuntimeError("No scalers available. Did model_registry.load() complete?")

    def series_keys(self) -> list[str]:
        return list(self.scalers.keys())


# Module-level singleton — imported by main.py and test fixtures
model_registry = ModelRegistry()

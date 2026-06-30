"""
Loads NAB-format CSV files and merges ground-truth anomaly labels from combined_labels.json.

Output is a clean DataFrame with columns: [timestamp, value, label]
where label=1 marks a known anomaly timestamp and label=0 marks normal.
"""

import json
import logging
from pathlib import Path

import pandas as pd
import yaml

logger = logging.getLogger(__name__)


def load_config(config_path: str | Path = "config/config.yaml") -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_series(
    csv_path: str | Path,
    labels_path: str | Path,
    file_key: str | None = None,
    timestamp_col: str = "timestamp",
    value_col: str = "value",
    label_col: str = "label",
) -> pd.DataFrame:
    """Load one NAB CSV and attach binary anomaly labels.

    Args:
        csv_path: Path to the NAB metric CSV file.
        labels_path: Path to NAB's combined_labels.json.
        file_key: Key inside combined_labels.json (e.g.
            'realAWSCloudwatch/ec2_cpu_utilization_825cc2.csv').
            Inferred from csv_path relative to data/raw/ if None.
        timestamp_col: Column name for timestamps in the CSV.
        value_col: Column name for metric values in the CSV.
        label_col: Output column name for the binary label.

    Returns:
        DataFrame with columns [timestamp, value, label], sorted by timestamp,
        index reset.
    """
    csv_path = Path(csv_path)
    labels_path = Path(labels_path)

    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")
    if not labels_path.exists():
        raise FileNotFoundError(f"Labels file not found: {labels_path}")

    df = pd.read_csv(csv_path)

    if timestamp_col not in df.columns:
        raise ValueError(f"Expected column '{timestamp_col}' in {csv_path}. Found: {df.columns.tolist()}")
    if value_col not in df.columns:
        raise ValueError(f"Expected column '{value_col}' in {csv_path}. Found: {df.columns.tolist()}")

    df[timestamp_col] = pd.to_datetime(df[timestamp_col])
    df = df[[timestamp_col, value_col]].copy()
    df.columns = ["timestamp", "value"]
    df = df.sort_values("timestamp").reset_index(drop=True)

    # Drop rows where value is NaN
    n_before = len(df)
    df = df.dropna(subset=["value"]).reset_index(drop=True)
    if len(df) < n_before:
        logger.warning("Dropped %d rows with NaN values from %s", n_before - len(df), csv_path.name)

    # Attach labels
    with open(labels_path) as f:
        all_labels: dict[str, list] = json.load(f)

    if file_key is None:
        # Build key from path: take the last two parts (category/filename)
        parts = csv_path.parts
        raw_idx = next((i for i, p in enumerate(parts) if p == "raw"), None)
        if raw_idx is not None and raw_idx + 1 < len(parts):
            file_key = "/".join(parts[raw_idx + 1:])
        else:
            file_key = "/".join(parts[-2:])

    entries = all_labels.get(file_key)
    if entries is None:
        logger.warning("No labels found for key '%s' in %s", file_key, labels_path.name)
        entries = []

    # Two NAB label formats are supported:
    #   - anomaly windows  -> list of [start, end] timestamp pairs (combined_windows.json)
    #   - anomaly points   -> list of single timestamp strings (combined_labels.json)
    # Window labelling marks every row inside a band, matching NAB's official scoring.
    label = pd.Series(0, index=df.index, dtype=int)
    for entry in entries:
        try:
            if isinstance(entry, (list, tuple)) and len(entry) == 2:
                start, end = pd.Timestamp(entry[0]), pd.Timestamp(entry[1])
                label |= ((df["timestamp"] >= start) & (df["timestamp"] <= end)).astype(int)
            else:
                point = pd.Timestamp(entry)
                label |= (df["timestamp"] == point).astype(int)
        except Exception:
            logger.warning("Could not parse label entry %r for %s", entry, file_key)
    df[label_col] = label.astype(int)

    n_anomalies = df[label_col].sum()
    logger.info(
        "Loaded %s: %d rows, %d anomalies (%.2f%%)",
        csv_path.name,
        len(df),
        n_anomalies,
        100.0 * n_anomalies / len(df) if len(df) > 0 else 0,
    )
    return df


def load_all_series(
    config_path: str | Path = "config/config.yaml",
) -> dict[str, pd.DataFrame]:
    """Load all series listed in config.yaml.

    Returns:
        Dict mapping file key (e.g. 'realAWSCloudwatch/ec2_cpu_utilization_825cc2.csv')
        to its labeled DataFrame.
    """
    cfg = load_config(config_path)
    raw_dir = Path(cfg["data"]["raw_dir"])
    labels_file = cfg["data"].get("labels_file", "combined_windows.json")
    labels_path = raw_dir / labels_file
    timestamp_col = cfg["data"]["timestamp_col"]
    value_col = cfg["data"]["value_col"]
    label_col = cfg["data"]["label_col"]

    series: dict[str, pd.DataFrame] = {}
    for file_key in cfg["data"]["files"]:
        csv_path = raw_dir / file_key
        try:
            df = load_series(
                csv_path=csv_path,
                labels_path=labels_path,
                file_key=file_key,
                timestamp_col=timestamp_col,
                value_col=value_col,
                label_col=label_col,
            )
            series[file_key] = df
        except Exception as exc:
            logger.error("Failed to load %s: %s", file_key, exc)

    logger.info("Loaded %d series total.", len(series))
    return series

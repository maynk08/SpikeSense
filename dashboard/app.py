"""
Spike-Sense Streamlit Dashboard

Layout
------
Header        — title, API status indicator
Main area     — Plotly time-series chart with anomaly markers
              — IF score and LSTM reconstruction error sub-charts
Sidebar       — Series selector, model selector, spike injection controls,
                threshold adjuster, alert log
Evaluation    — Collapsible panel: metrics table, confusion matrices, PR summary

Run locally:
    streamlit run dashboard/app.py

The dashboard calls the FastAPI backend for all scoring.  When the backend is
unreachable, a warning banner is shown and the app operates in offline mode
using locally loaded data (no model scores).
"""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure project root is importable when launched from repo root
ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from plotly.subplots import make_subplots

import dashboard.api_client as api

# ---------------------------------------------------------------------------
# Page config (must be first Streamlit call)
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Spike-Sense | Cloud Anomaly Monitor",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
SERIES_LABELS = {
    "realAWSCloudwatch/ec2_cpu_utilization_825cc2.csv": "EC2 CPU #1",
    "realAWSCloudwatch/ec2_cpu_utilization_fe7f93.csv": "EC2 CPU #2",
    "realAWSCloudwatch/ec2_network_in_257a54.csv": "EC2 Network In #1",
    "realAWSCloudwatch/ec2_network_in_5abac7.csv": "EC2 Network In #2",
    "realAWSCloudwatch/rds_cpu_utilization_e47b3b.csv": "RDS CPU #1",
    "realAWSCloudwatch/rds_cpu_utilization_cc0c53.csv": "RDS CPU #2",
}

ANOMALY_COLORS = {
    "Isolation Forest": "#ef4444",
    "LSTM Autoencoder": "#f97316",
    "Combined (Union)": "#dc2626",
    "Combined (Intersection)": "#7c3aed",
}

ALERT_LOG_KEY = "alert_log"
MAX_ALERT_LOG = 20
WINDOW_SIZE = 30


# ---------------------------------------------------------------------------
# Session state helpers
# ---------------------------------------------------------------------------
def _init_state() -> None:
    defaults = {
        ALERT_LOG_KEY: [],
        "last_series_key": None,
        "cached_df": None,
        "batch_results": None,
        "spike_results": None,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


# ---------------------------------------------------------------------------
# Data helpers
# ---------------------------------------------------------------------------
@st.cache_data(ttl=300)
def _load_series(series_key: str) -> pd.DataFrame:
    """Load a NAB series from disk (cached for 5 min)."""
    from src.data.loader import load_series

    raw_dir = ROOT / "data" / "raw"
    labels_path = raw_dir / "combined_labels.json"
    csv_path = raw_dir / series_key
    return load_series(csv_path, labels_path, file_key=series_key)


def _build_windows(df: pd.DataFrame, window_size: int = WINDOW_SIZE) -> list[list[float]]:
    """Slice a series into sliding windows (stride=1) as raw float lists."""
    values = df["value"].values.tolist()
    return [values[i: i + window_size] for i in range(len(values) - window_size + 1)]


def _score_series(df: pd.DataFrame, series_key: str) -> dict | None:
    """Score all windows in the series via the batch predict endpoint."""
    windows = _build_windows(df)
    if not windows:
        return None
    with st.spinner("Scoring time-series with both models…"):
        return api.predict_batch(windows, series_key=series_key)


# ---------------------------------------------------------------------------
# Plotting helpers
# ---------------------------------------------------------------------------
def _make_main_chart(
    df: pd.DataFrame,
    batch: dict | None,
    model_filter: str,
    window_size: int = WINDOW_SIZE,
) -> go.Figure:
    """Build a 3-row Plotly figure: metric + IF score + LSTM error."""
    fig = make_subplots(
        rows=3, cols=1,
        shared_xaxes=True,
        row_heights=[0.55, 0.225, 0.225],
        vertical_spacing=0.04,
        subplot_titles=["Metric Value", "IF Anomaly Score", "LSTM Reconstruction Error"],
    )

    ts = df["timestamp"]
    vals = df["value"]

    # Row 1 — raw metric
    fig.add_trace(
        go.Scatter(x=ts, y=vals, mode="lines", name="Metric",
                   line=dict(color="#3b82f6", width=1.5)),
        row=1, col=1,
    )

    # Overlay ground-truth anomaly markers (if any in data)
    gt_mask = df["label"] == 1
    if gt_mask.any():
        fig.add_trace(
            go.Scatter(
                x=ts[gt_mask], y=vals[gt_mask],
                mode="markers", name="Ground Truth",
                marker=dict(symbol="star", size=14, color="#facc15", line=dict(width=1, color="#1e3a5f")),
            ),
            row=1, col=1,
        )

    if batch:
        preds = batch["predictions"]
        # Window timestamp = last step of each window
        win_ts = ts.iloc[window_size - 1:].reset_index(drop=True)
        win_vals = vals.iloc[window_size - 1:].reset_index(drop=True)
        n = min(len(preds), len(win_ts))

        if_scores = [p["scores"]["isolation_forest"] for p in preds[:n]]
        lstm_errors = [p["scores"]["lstm_reconstruction_error"] for p in preds[:n]]

        # Determine which flag to use for row-1 anomaly markers
        if model_filter == "Isolation Forest":
            flagged = [p["if_anomaly"] for p in preds[:n]]
            color = ANOMALY_COLORS["Isolation Forest"]
        elif model_filter == "LSTM Autoencoder":
            flagged = [p["lstm_anomaly"] for p in preds[:n]]
            color = ANOMALY_COLORS["LSTM Autoencoder"]
        elif model_filter == "Combined (Union)":
            flagged = [p["combined_union"] for p in preds[:n]]
            color = ANOMALY_COLORS["Combined (Union)"]
        else:
            flagged = [p["combined_intersection"] for p in preds[:n]]
            color = ANOMALY_COLORS["Combined (Intersection)"]

        anomaly_ts = win_ts[[bool(f) for f in flagged]]
        anomaly_vals = win_vals[[bool(f) for f in flagged]]

        if not anomaly_ts.empty:
            fig.add_trace(
                go.Scatter(
                    x=anomaly_ts, y=anomaly_vals,
                    mode="markers", name=f"Detected ({model_filter})",
                    marker=dict(symbol="circle", size=8, color=color,
                                line=dict(width=1, color="white")),
                ),
                row=1, col=1,
            )

        # Row 2 — IF score
        fig.add_trace(
            go.Scatter(x=win_ts[:n], y=if_scores, mode="lines", name="IF Score",
                       line=dict(color=ANOMALY_COLORS["Isolation Forest"], width=1)),
            row=2, col=1,
        )
        fig.add_hline(y=0, line_dash="dot", line_color="gray", row=2, col=1)

        # Row 3 — LSTM error
        fig.add_trace(
            go.Scatter(x=win_ts[:n], y=lstm_errors, mode="lines", name="LSTM Error",
                       line=dict(color=ANOMALY_COLORS["LSTM Autoencoder"], width=1)),
            row=3, col=1,
        )

    fig.update_layout(
        height=600,
        template="plotly_dark",
        paper_bgcolor="#0f172a",
        plot_bgcolor="#0f172a",
        font=dict(color="#e2e8f0"),
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=40, b=0),
        hovermode="x unified",
    )
    fig.update_xaxes(gridcolor="#1e293b", showgrid=True)
    fig.update_yaxes(gridcolor="#1e293b", showgrid=True)
    return fig


# ---------------------------------------------------------------------------
# Alert log
# ---------------------------------------------------------------------------
def _add_alert(message: str) -> None:
    log: list = st.session_state[ALERT_LOG_KEY]
    import datetime
    log.insert(0, {"time": datetime.datetime.now().strftime("%H:%M:%S"), "message": message})
    st.session_state[ALERT_LOG_KEY] = log[:MAX_ALERT_LOG]


# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
def _render_sidebar(api_ok: bool, info_data: dict | None) -> tuple[str, str, float, int]:
    """Render controls; return (series_key, model_filter, magnitude, duration)."""
    st.sidebar.title("⚡ Spike-Sense")
    st.sidebar.caption("Cloud Infrastructure Anomaly Monitor")
    st.sidebar.divider()

    # API status
    if api_ok:
        st.sidebar.success("API: Connected", icon="🟢")
    else:
        st.sidebar.error("API: Offline — scores unavailable", icon="🔴")

    st.sidebar.subheader("Series")
    label_to_key = {v: k for k, v in SERIES_LABELS.items()}
    label = st.sidebar.selectbox("Select metric series", list(label_to_key.keys()))
    series_key = label_to_key[label]

    st.sidebar.subheader("Model")
    model_filter = st.sidebar.radio(
        "Show anomalies from",
        ["Combined (Union)", "Isolation Forest", "LSTM Autoencoder", "Combined (Intersection)"],
    )

    st.sidebar.divider()
    st.sidebar.subheader("Spike Injection")
    spike_mode = st.sidebar.selectbox(
        "Anomaly type",
        ["point_spike", "level_shift", "trend_drift"],
        format_func=lambda x: x.replace("_", " ").title(),
    )
    magnitude = st.sidebar.slider("Magnitude (σ)", 1.0, 8.0, 4.0, 0.5)
    duration = st.sidebar.slider("Duration (steps)", 5, 100, 20, 5)

    if st.sidebar.button("🔴 Inject Spike + Score", use_container_width=True, disabled=not api_ok):
        with st.spinner(f"Injecting {spike_mode} and scoring…"):
            result = api.inject_spike(series_key, spike_mode, magnitude, duration)
        if result:
            st.session_state["spike_results"] = result
            n_anom = result["predictions"]["n_anomalies_union"]
            _add_alert(f"Spike injected ({spike_mode}) — {n_anom} anomalous windows detected")
            st.rerun()
        else:
            st.sidebar.error("Injection failed — check API connection.")

    if st.sidebar.button("🔄 Rescore Series", use_container_width=True, disabled=not api_ok):
        st.session_state["batch_results"] = None
        st.session_state["spike_results"] = None
        st.rerun()

    st.sidebar.divider()
    st.sidebar.subheader("Alert Log")
    alert_log: list = st.session_state[ALERT_LOG_KEY]
    if alert_log:
        for entry in alert_log[:5]:
            st.sidebar.caption(f"[{entry['time']}] {entry['message']}")
    else:
        st.sidebar.caption("No alerts yet.")

    return series_key, model_filter, magnitude, duration


# ---------------------------------------------------------------------------
# Evaluation panel
# ---------------------------------------------------------------------------
def _render_evaluation_panel() -> None:
    with st.expander("📊 Model Evaluation Results", expanded=False):
        eval_data = api.evaluate()
        if not eval_data:
            st.warning("Evaluation results unavailable. Ensure the API is running and `results/` exists.")
            return

        st.subheader("Test Split Performance")
        metrics = eval_data.get("metrics", [])
        if metrics:
            df_metrics = pd.DataFrame([{
                "Model": m["model"],
                "Precision": f"{m['precision']:.3f}",
                "Recall": f"{m['recall']:.3f}",
                "F1": f"{m['f1']:.3f}",
                "FPR": f"{m['fpr']:.3f}",
                "TP": m["tp"], "FP": m["fp"], "TN": m["tn"], "FN": m["fn"],
            } for m in metrics])
            st.dataframe(df_metrics, use_container_width=True, hide_index=True)

        st.caption(
            "ℹ️ Test split has 0 labeled anomalies for these NAB series "
            "(all real anomaly timestamps fall in the training window). "
            "Use spike scenarios below for controlled precision/recall evaluation."
        )

        st.subheader("Spike Scenario Results")
        spike_rows = eval_data.get("spike_scenario_summary", [])
        if spike_rows:
            df_spike = pd.DataFrame([{
                "Scenario": r["scenario"].replace("_", " ").title(),
                "Model": r["model"],
                "Precision": f"{r['precision']:.3f}",
                "Recall": f"{r['recall']:.3f}",
                "F1": f"{r['f1']:.3f}",
            } for r in spike_rows
              if r["model"] in ("Isolation Forest", "LSTM Autoencoder", "Combined (Union)")])
            st.dataframe(df_spike, use_container_width=True, hide_index=True)

        st.subheader("Model Configuration")
        cfg = eval_data.get("detector_config", {})
        col1, col2, col3 = st.columns(3)
        col1.metric("IF Contamination", f"{cfg.get('if_contamination', 0):.3f}")
        col2.metric("LSTM Threshold", f"{cfg.get('lstm_threshold', 0):.4f}")
        col3.metric("Window Size", cfg.get('window_size', 30))


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> None:
    _init_state()

    api_ok = api.is_reachable()
    info_data = api.info() if api_ok else None

    # Sidebar
    series_key, model_filter, magnitude, duration = _render_sidebar(api_ok, info_data)

    # Header
    st.markdown(
        "<h1 style='color:#38bdf8;margin-bottom:0'>⚡ Spike-Sense</h1>"
        "<p style='color:#94a3b8;margin-top:2px'>AI-driven anomaly detection for cloud infrastructure metrics</p>",
        unsafe_allow_html=True,
    )

    # KPI row
    if api_ok and info_data:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Model", "IF + LSTM")
        c2.metric("Window Size", info_data.get("window_size", WINDOW_SIZE))
        c3.metric("LSTM Threshold", f"{info_data.get('lstm_threshold', 0):.4f}")
        c4.metric("IF Contamination", f"{info_data.get('if_contamination', 0):.3f}")

    st.divider()

    # Load series data
    if series_key != st.session_state["last_series_key"]:
        st.session_state["cached_df"] = None
        st.session_state["batch_results"] = None
        st.session_state["spike_results"] = None
        st.session_state["last_series_key"] = series_key

    if st.session_state["cached_df"] is None:
        with st.spinner(f"Loading {SERIES_LABELS.get(series_key, series_key)}…"):
            st.session_state["cached_df"] = _load_series(series_key)

    df: pd.DataFrame = st.session_state["cached_df"]

    # Auto-score on first load
    if api_ok and st.session_state["batch_results"] is None and st.session_state["spike_results"] is None:
        st.session_state["batch_results"] = _score_series(df, series_key)

    # Determine which results to show
    if st.session_state["spike_results"]:
        spike_res = st.session_state["spike_results"]
        batch = spike_res["predictions"]
        # Rebuild df with injected values for display
        st.info(
            f"Showing spike-injected series ({spike_res['mode'].replace('_', ' ').title()}). "
            f"**{spike_res['n_anomalous_windows']} anomalous windows** | "
            f"**{batch['n_anomalies_union']} detections (union)**",
            icon="🔴",
        )
    else:
        batch = st.session_state["batch_results"]

    # Main chart
    fig = _make_main_chart(df, batch, model_filter)
    st.plotly_chart(fig, use_container_width=True)

    # Stats below chart
    if batch:
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Windows", batch["n_windows"])
        c2.metric("IF Detections", batch["n_anomalies_if"])
        c3.metric("LSTM Detections", batch["n_anomalies_lstm"])
        c4.metric("Union Detections", batch["n_anomalies_union"])

    st.divider()

    # Evaluation panel
    if api_ok:
        _render_evaluation_panel()

    # Footer
    st.markdown(
        "<p style='color:#475569;font-size:0.8em;text-align:center'>"
        "Spike-Sense — AI Anomaly Detection · "
        "<a href='https://github.com' style='color:#38bdf8'>GitHub</a> · "
        "Powered by Isolation Forest + LSTM Autoencoder</p>",
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()

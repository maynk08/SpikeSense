---
title: Spike-Sense
emoji: ⚡
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
license: mit
---

# ⚡ Spike-Sense

> **AI-driven anomaly detection for cloud infrastructure metrics**

Spike-Sense ingests real AWS CloudWatch time-series data, applies two complementary anomaly detectors — **Isolation Forest** (unsupervised, tree-based) and **LSTM Autoencoder** (deep learning, reconstruction-based) — serves predictions through a REST API, and surfaces everything on an interactive dashboard with live Discord alerts.

---

## Live Demo

| Component | URL |
|---|---|
| 📊 Dashboard | *(Streamlit Cloud — add after deployment)* |
| 🔌 API | *(Render.com — add after deployment)* |
| 📖 API Docs (Swagger) | `<api-url>/docs` |

---

## Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                         DATA LAYER                               │
│                                                                  │
│  NAB Dataset (real AWS CloudWatch CSVs)                          │
│    └─ loader.py  ──►  preprocessor.py  ──►  spike_injector.py   │
│         ↓ labeled DataFrames     ↓ scaled windows               │
└─────────────────────────┬────────────────────────────────────────┘
                          │
┌─────────────────────────▼────────────────────────────────────────┐
│                        MODEL LAYER                               │
│                                                                  │
│  Isolation Forest          LSTM Autoencoder                      │
│  (scikit-learn)            (TensorFlow / Keras)                  │
│  • 200 trees               • Encoder LSTM (64 units)             │
│  • contamination = 0.05    • Bottleneck Dense (32 units)         │
│  • 8 statistical features  • Decoder LSTM (64 units)             │
│                            • Threshold @ 99th pct of train MSE   │
│                                                                  │
│  Artifacts saved to models/ and committed to repo                │
└─────────────────────────┬────────────────────────────────────────┘
                          │  loaded at startup
┌─────────────────────────▼────────────────────────────────────────┐
│              FastAPI SCORING ENGINE  (Render.com)                │
│                                                                  │
│  POST /predict          ─ single-window inference                │
│  POST /predict/batch    ─ full time-series scoring               │
│  GET  /evaluate         ─ pre-computed metrics                   │
│  POST /demo/inject-spike ─ live demo with alert                  │
│                          └─► Discord Webhook Alert               │
└─────────────────────────┬────────────────────────────────────────┘
                          │  HTTP
┌─────────────────────────▼────────────────────────────────────────┐
│           STREAMLIT DASHBOARD  (Streamlit Community Cloud)       │
│                                                                  │
│  • 3-panel Plotly chart (metric + IF score + LSTM error)         │
│  • Anomaly markers per model / combined view                     │
│  • Spike injection controls (point, level shift, trend drift)    │
│  • Evaluation panel (metrics table, spike scenario results)      │
│  • Alert log sidebar                                             │
└──────────────────────────────────────────────────────────────────┘
```

---

## Model Performance

Results on synthetic spike scenarios injected into real AWS CloudWatch data.
*(Real test split has 0 labeled anomalies — all NAB ground-truth anomaly
timestamps fall within the training window for the 3 selected series.)*

### Spike Scenario Evaluation

| Scenario | Model | Precision | Recall | F1 |
|---|---|---|---|---|
| Point Spike | Isolation Forest | 0.003 | **1.000** | 0.007 |
| Point Spike | LSTM Autoencoder | 0.000 | 0.000 | 0.000 |
| Point Spike | Combined (Union) | 0.003 | **1.000** | 0.007 |
| Level Shift | Isolation Forest | 0.024 | 0.350 | 0.046 |
| Level Shift | LSTM Autoencoder | 0.000 | 0.000 | 0.000 |
| Level Shift | Combined (Union) | 0.024 | 0.350 | 0.046 |
| Trend Drift | Isolation Forest | 0.114 | **0.760** | 0.198 |
| Trend Drift | LSTM Autoencoder | 0.106 | 0.180 | 0.133 |
| Trend Drift | Combined (Union) | 0.114 | **0.760** | 0.198 |

**Key insights:**
- IF excels at point anomalies (R=1.00) and trend drift (R=0.76) — fast, no GPU needed
- LSTM catches trend drift more conservatively (lower FPR=0.019 vs IF=0.075)
- Combined Union prioritises recall (catch everything); Intersection prioritises precision
- Low precision is expected for severely imbalanced data (1 anomalous : 3,900+ normal windows)
- LSTM threshold is at the 99th percentile — lowering to 95th improves recall at the cost of more false positives

---

## Project Structure

```
spike-sense/
├── config/
│   └── config.yaml              ← all tunable parameters
├── data/
│   ├── raw/
│   │   ├── realAWSCloudwatch/   ← 6 real AWS CloudWatch CSVs (NAB)
│   │   └── combined_labels.json ← ground-truth anomaly timestamps
│   └── processed/               ← generated by scripts/train.py (gitignored)
├── models/
│   ├── isolation_forest.joblib  ← trained IF (2.2 MB)
│   ├── if_metadata.json
│   ├── lstm_autoencoder.keras   ← trained LSTM (562 KB)
│   └── lstm_metadata.json       ← threshold, window_size, error stats
├── notebooks/                   ← EDA, training, evaluation (Phase 2 deliverable)
├── results/
│   ├── evaluation_results.json  ← full metrics (4 configurations × 3 scenarios)
│   ├── metrics_summary.csv
│   ├── if_contamination_sweep.json
│   ├── lstm_threshold_sweep.json
│   └── lstm_training_history.json
├── scripts/
│   ├── train.py                 ← retrain both models from scratch
│   └── evaluate.py              ← regenerate evaluation_results.json
├── src/
│   ├── data/
│   │   ├── loader.py            ← NAB CSV loader + label merging
│   │   ├── preprocessor.py      ← MinMaxScaler, sliding windows, feature extraction
│   │   ├── spike_injector.py    ← point spike / level shift / trend drift injection
│   │   └── splitter.py          ← chronological train/val/test split
│   ├── models/
│   │   ├── isolation_forest.py  ← train, score, save, load, contamination sweep
│   │   └── lstm_autoencoder.py  ← build, train, threshold, save, load, threshold sweep
│   ├── api/
│   │   ├── main.py              ← FastAPI app (6 endpoints)
│   │   ├── schemas.py           ← Pydantic v2 request/response models
│   │   ├── model_loader.py      ← startup singleton registry
│   │   └── alerting.py          ← Discord webhook with cooldown
│   └── evaluation/
│       └── evaluator.py         ← metrics, PR curves, spike scenarios, persistence
├── dashboard/
│   ├── app.py                   ← Streamlit dashboard
│   └── api_client.py            ← HTTP client wrapper
├── tests/
│   ├── test_data.py             ← 25 tests for data pipeline
│   ├── test_models.py           ← 19 tests for IF + LSTM
│   ├── test_evaluation.py       ← 23 tests for evaluator
│   └── test_api.py              ← 37 tests for all API endpoints
├── .streamlit/config.toml       ← dark theme config
├── render.yaml                  ← Render.com deployment spec
├── Procfile                     ← alternative deployment entrypoint
├── requirements.txt             ← pinned full dependency set
├── requirements-dashboard.txt   ← slim deps for Streamlit Community Cloud
└── pyproject.toml
```

---

## Quick Start (Local)

```bash
# 1. Clone and enter project
git clone https://github.com/<your-username>/spike-sense.git
cd spike-sense

# 2. Create virtual environment and install
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# 3. (Optional) Retrain models — pre-trained artifacts already in models/
python scripts/train.py

# 4. Run the dashboard standalone (scores models in-process — no API needed)
streamlit run dashboard/app.py
# → Dashboard at http://localhost:8501

# --- OR, the two-service setup (FastAPI backend + dashboard) ---

# 4a. Start the API backend
uvicorn src.api.main:app --reload
# → API docs at http://localhost:8000/docs

# 4b. Start the dashboard pointed at the API (separate terminal)
SPIKE_SENSE_API_URL=http://localhost:8000 streamlit run dashboard/app.py

# 5. Run tests
pytest
```

---

## Deployment

The dashboard supports two deployment modes, selected automatically by whether
the `SPIKE_SENSE_API_URL` environment variable is set:

- **Single-service (recommended for demos)** — the dashboard scores the IF +
  LSTM models in-process (`dashboard/local_scoring.py`). One URL, no backend to
  sleep or cold-start. Used when `SPIKE_SENSE_API_URL` is **unset**.
- **Two-service** — a separate FastAPI backend handles scoring and the
  dashboard calls it over HTTP. Used when `SPIKE_SENSE_API_URL` **is** set.

### Single-service → Streamlit Community Cloud (one free URL)

1. Push repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with GitHub
3. **New app** → select this repo, branch `main`
4. Set **Main file path** to `dashboard/app.py`
   *(Streamlit auto-detects `requirements-dashboard.txt`, which includes
   `tensorflow-cpu` for in-process scoring.)*
5. Under **Advanced settings**, set **Python version** to `3.11`
6. **Do not** set `SPIKE_SENSE_API_URL` — leaving it unset enables in-process
   scoring (no separate backend needed)
7. Deploy — dashboard will be live at `https://<app-name>.streamlit.app`

### Two-service (optional) → Render.com + Streamlit Cloud

Deploy the API separately and point the dashboard at it:

1. Create a **Web Service** on [render.com](https://render.com) — it auto-detects
   `render.yaml`. Add `DISCORD_WEBHOOK_URL` in the Render dashboard if you want
   live alerts. API goes live at e.g. `https://spike-sense.onrender.com`.
2. Deploy the dashboard to Streamlit Cloud as above, but **add**
   `SPIKE_SENSE_API_URL=https://spike-sense.onrender.com` in Secrets.

> **Note:** Render's free tier spins down after 15 minutes of inactivity (≈50s
> cold start) and is memory-constrained for TensorFlow — which is why the
> single-service mode above is recommended for demos.

---

## Environment Variables

| Variable | Required | Description |
|---|---|---|
| `DISCORD_WEBHOOK_URL` | No | Discord Incoming Webhook URL for anomaly alerts |
| `SPIKE_SENSE_API_URL` | No | API base URL used by the dashboard (default: `http://localhost:8000`) |

Copy `.env.example` to `.env` for local development.

---

## Dataset

Real AWS CloudWatch metric streams from the [Numenta Anomaly Benchmark (NAB)](https://github.com/numenta/NAB).

| File | Metric | Anomaly Timestamps |
|---|---|---|
| `ec2_cpu_utilization_825cc2.csv` | EC2 CPU % | 2014-04-15 15:44, 2014-04-16 03:34 |
| `ec2_cpu_utilization_fe7f93.csv` | EC2 CPU % | 2014-02-17 06:12, 2014-02-22 00:02, 2014-02-23 15:17 |
| `ec2_network_in_257a54.csv` | EC2 network inbound | 2014-04-15 16:44 |
| `ec2_network_in_5abac7.csv` | EC2 network inbound | 2014-03-10 18:56, 2014-03-12 21:01 |
| `rds_cpu_utilization_e47b3b.csv` | RDS CPU % | 2014-04-13 06:52, 2014-04-18 23:27 |
| `rds_cpu_utilization_cc0c53.csv` | RDS CPU % | 2014-02-25 07:15, 2014-02-27 00:50 |

Each series: 4,032 rows at 5-minute intervals (~14 days).

---

## Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.11 |
| ML | scikit-learn 1.7, TensorFlow 2.16 / Keras |
| API | FastAPI 0.136, Uvicorn, Pydantic v2 |
| Dashboard | Streamlit 1.35, Plotly 5.22 |
| Alerting | Discord Incoming Webhooks |
| Hosting (API) | Render.com free tier |
| Hosting (UI) | Streamlit Community Cloud |
| Data | Numenta Anomaly Benchmark (NAB) |
| Tests | pytest 9.0, pytest-cov |

---

## Future Work

- **Grafana + Prometheus** — production-grade dashboard with persistent metrics storage
- **Real-time CloudWatch streaming** — pull live metrics via `boto3` CloudWatch API
- **Online retraining** — detect model drift and trigger incremental re-fitting
- **Ensemble voting** — learned weights combining IF score and LSTM error
- **Multi-series correlation** — flag when multiple metrics spike together (cascade failure detection)

---

## License

MIT — see [LICENSE](LICENSE)

# Dockerfile for deploying the Spike-Sense dashboard to Hugging Face Spaces.
#
# Single-service deployment: the Streamlit dashboard scores the IF + LSTM
# models in-process (no separate API). HF Spaces gives this container 16 GB
# RAM on the free CPU tier, comfortably handling TensorFlow.
#
# Python is pinned to 3.11 so every dependency installs from a prebuilt wheel
# (TensorFlow 2.16.1, scipy, numpy, scikit-learn all publish cp311 wheels).

FROM python:3.11-slim

# Streamlit serves on 7860 — the port Hugging Face Spaces expects.
ENV PORT=7860 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    # HF mounts the app at /home/user/app; keep caches writable there.
    HOME=/home/user \
    STREAMLIT_SERVER_HEADLESS=true \
    STREAMLIT_BROWSER_GATHER_USAGE_STATS=false

# Create a non-root user (HF Spaces best practice).
RUN useradd -m -u 1000 user
USER user
WORKDIR /home/user/app

# Install dependencies first for better layer caching.
# Uses the lighter dashboard requirements (tensorflow-cpu, no FastAPI).
COPY --chown=user requirements-dashboard.txt ./requirements-dashboard.txt
RUN pip install --upgrade pip && \
    pip install -r requirements-dashboard.txt

# Copy the rest of the project (models, data, config, results, src, dashboard).
COPY --chown=user . .

EXPOSE 7860

# NOTE: SPIKE_SENSE_API_URL is intentionally NOT set, so api_client.py uses
# in-process scoring (dashboard/local_scoring.py) — no backend needed.
CMD ["streamlit", "run", "dashboard/app.py", \
     "--server.port=7860", \
     "--server.address=0.0.0.0", \
     "--server.headless=true"]

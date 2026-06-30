"""
Generate all data-driven figures for the project report from results/*.json.

Every chart is produced from the real artifacts written by scripts/train.py and
scripts/evaluate.py, so the figures in the report reflect the actual run.

Output: report/figures/fig_*.png
Run:    python report/scripts/make_charts.py
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
RESULTS = ROOT / "results"
FIGS = ROOT / "report" / "figures"
FIGS.mkdir(parents=True, exist_ok=True)

plt.rcParams.update({
    "figure.dpi": 150,
    "font.size": 11,
    "font.family": "DejaVu Sans",
    "axes.grid": True,
    "grid.alpha": 0.3,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Consistent colour per model across all charts.
COLORS = {
    "Isolation Forest": "#2563eb",
    "LSTM Autoencoder": "#dc2626",
    "Combined (Union)": "#16a34a",
    "Combined (Intersection)": "#9333ea",
}


def _load(name: str):
    with open(RESULTS / name) as f:
        return json.load(f)


def _save(fig, name: str):
    path = FIGS / name
    fig.tight_layout()
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


def chart_lstm_training():
    h = _load("lstm_training_history.json")
    epochs = range(1, len(h["loss"]) + 1)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, h["loss"], "o-", color="#2563eb", label="Training loss")
    ax.plot(epochs, h["val_loss"], "s-", color="#dc2626", label="Validation loss")
    best = int(np.argmin(h["val_loss"]))
    ax.axvline(best + 1, color="gray", ls="--", alpha=0.6)
    ax.annotate(f"best epoch {best+1}\nval_loss={h['val_loss'][best]:.3f}",
                xy=(best + 1, h["val_loss"][best]), xytext=(best + 1.4, max(h['val_loss']) * 0.7),
                fontsize=9, color="gray")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean squared error (MSE) loss")
    ax.set_title("LSTM Autoencoder Training History")
    ax.legend()
    _save(fig, "fig_lstm_training.png")


def chart_if_contamination_sweep():
    data = _load("if_contamination_sweep.json")
    x = [d["contamination"] for d in data]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, [d["precision"] for d in data], "o-", label="Precision", color="#2563eb")
    ax.plot(x, [d["recall"] for d in data], "s-", label="Recall", color="#dc2626")
    ax.plot(x, [d["f1"] for d in data], "^-", label="F1-score", color="#16a34a")
    best = max(data, key=lambda d: d["f1"])
    ax.axvline(best["contamination"], color="gray", ls="--", alpha=0.6)
    ax.set_xlabel("Contamination parameter")
    ax.set_ylabel("Score")
    ax.set_title("Isolation Forest — Contamination Sweep (validation set)")
    ax.legend()
    _save(fig, "fig_if_contamination_sweep.png")


def chart_lstm_threshold_sweep():
    data = _load("lstm_threshold_sweep.json")
    x = [d["percentile"] for d in data]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(x, [d["precision"] for d in data], "o-", label="Precision", color="#2563eb")
    ax.plot(x, [d["recall"] for d in data], "s-", label="Recall", color="#dc2626")
    ax.plot(x, [d["f1"] for d in data], "^-", label="F1-score", color="#16a34a")
    ax.set_xlabel("Threshold percentile of training reconstruction error")
    ax.set_ylabel("Score")
    ax.set_title("LSTM Autoencoder — Threshold Percentile Sweep (validation set)")
    ax.legend()
    _save(fig, "fig_lstm_threshold_sweep.png")


def chart_test_metrics_bar():
    e = _load("evaluation_results.json")
    metrics = e["test_split_evaluation"]["metrics"]
    models = [m["model"] for m in metrics]
    fig, ax = plt.subplots(figsize=(8.5, 4.5))
    x = np.arange(len(models))
    w = 0.26
    ax.bar(x - w, [m["precision"] for m in metrics], w, label="Precision", color="#2563eb")
    ax.bar(x, [m["recall"] for m in metrics], w, label="Recall", color="#dc2626")
    ax.bar(x + w, [m["f1"] for m in metrics], w, label="F1-score", color="#16a34a")
    for i, m in enumerate(metrics):
        ax.text(i + w, m["f1"] + 0.01, f"{m['f1']:.2f}", ha="center", fontsize=8)
    ax.set_xticks(x)
    ax.set_xticklabels([m.replace(" (", "\n(") for m in models], fontsize=9)
    ax.set_ylabel("Score")
    ax.set_ylim(0, max(m["recall"] for m in metrics) * 1.25)
    ax.set_title("Test-Split Performance by Model Configuration")
    ax.legend()
    _save(fig, "fig_test_metrics_bar.png")


def chart_confusion_matrices():
    e = _load("evaluation_results.json")
    cms = e["test_split_evaluation"]["confusion_matrices"]
    fig, axes = plt.subplots(2, 2, figsize=(8.5, 7.5))
    for ax, cm in zip(axes.flat, cms):
        matrix = np.array(cm["matrix"])
        im = ax.imshow(matrix, cmap="Blues")
        ax.set_title(cm["model"], fontsize=11)
        ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
        ax.set_xticklabels(cm["labels"]); ax.set_yticklabels(cm["labels"])
        ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
        thresh = matrix.max() / 2
        for (r, c), v in np.ndenumerate(matrix):
            ax.text(c, r, f"{v}", ha="center", va="center",
                    color="white" if v > thresh else "black", fontsize=12)
        ax.grid(False)
    fig.suptitle("Confusion Matrices (Test Split)", fontsize=13)
    _save(fig, "fig_confusion_matrices.png")


def chart_pr_curves():
    e = _load("evaluation_results.json")
    curves = e["test_split_evaluation"]["pr_curves"]
    fig, ax = plt.subplots(figsize=(7, 5))
    for c in curves:
        ax.plot(c["recall"], c["precision"], color=COLORS.get(c["model"], None),
                label=f"{c['model']} (AP={c['avg_precision']:.3f})", lw=1.6)
    ax.set_xlabel("Recall"); ax.set_ylabel("Precision")
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    ax.set_title("Precision–Recall Curves (Test Split)")
    ax.legend(fontsize=9)
    _save(fig, "fig_pr_curves.png")


def chart_spike_recall():
    e = _load("evaluation_results.json")
    scenarios = e["spike_scenario_evaluation"]
    models = ["Isolation Forest", "LSTM Autoencoder", "Combined (Union)"]
    names = [s["scenario"].replace("_", " ").title() for s in scenarios]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    x = np.arange(len(scenarios)); w = 0.25
    for j, model in enumerate(models):
        recalls = []
        for s in scenarios:
            row = next((m for m in s["metrics"] if m["model"] == model), None)
            recalls.append(row["recall"] if row else 0)
        ax.bar(x + (j - 1) * w, recalls, w, label=model, color=COLORS.get(model))
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("Recall"); ax.set_ylim(0, 1.1)
    ax.set_title("Synthetic Spike-Scenario Recall by Model")
    ax.legend(fontsize=9)
    _save(fig, "fig_spike_recall.png")


def chart_series_timeline():
    """Illustrative: a series with its labelled anomaly region highlighted."""
    import sys
    sys.path.insert(0, str(ROOT))
    import logging
    logging.disable(logging.CRITICAL)
    from src.data.loader import load_series

    key = "realKnownCause/cpu_utilization_asg_misconfiguration.csv"
    df = load_series(ROOT / "data/raw" / key, ROOT / "data/raw/combined_windows.json", file_key=key)
    n = len(df)
    fig, ax = plt.subplots(figsize=(9, 3.8))
    ax.plot(df["timestamp"], df["value"], color="#334155", lw=0.6, label="CPU utilization")
    anom = df[df["label"] == 1]
    ax.scatter(anom["timestamp"], anom["value"], color="#dc2626", s=6, label="Labelled anomaly window", zorder=3)
    # mark the 85% test boundary
    ax.axvline(df["timestamp"].iloc[int(n * 0.85)], color="#2563eb", ls="--", alpha=0.7)
    ax.text(df["timestamp"].iloc[int(n * 0.85)], df["value"].max() * 0.95, " test split →",
            color="#2563eb", fontsize=9)
    ax.set_xlabel("Time"); ax.set_ylabel("CPU %")
    ax.set_title("Real NAB Series with Labelled Anomaly Window\n(cpu_utilization_asg_misconfiguration)")
    ax.legend(fontsize=9, loc="upper left")
    _save(fig, "fig_series_timeline.png")


def main():
    print("Generating report charts...")
    chart_lstm_training()
    chart_if_contamination_sweep()
    chart_lstm_threshold_sweep()
    chart_test_metrics_bar()
    chart_confusion_matrices()
    chart_pr_curves()
    chart_spike_recall()
    chart_series_timeline()
    print("Done.")


if __name__ == "__main__":
    main()

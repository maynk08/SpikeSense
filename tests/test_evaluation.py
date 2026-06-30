"""
Tests for the evaluation module.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.evaluation.evaluator import (
    compute_metrics,
    compute_pr_curve,
    evaluate_all,
    load_results,
    metrics_to_dataframe,
    save_metrics_csv,
    save_results,
)

RESULTS_DIR = Path("results")


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture()
def perfect_predictions():
    y_true = np.array([0, 0, 0, 1, 1, 0, 0, 1], dtype=np.int32)
    y_pred = y_true.copy()
    return y_true, y_pred


@pytest.fixture()
def mixed_predictions():
    # 5 TP, 2 FP, 3 TN, 1 FN
    y_true = np.array([1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0], dtype=np.int32)
    y_pred = np.array([1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 0], dtype=np.int32)
    return y_true, y_pred


@pytest.fixture()
def all_normal():
    """Edge case: no positive labels."""
    y_true = np.zeros(50, dtype=np.int32)
    y_pred = np.zeros(50, dtype=np.int32)
    return y_true, y_pred


# ---------------------------------------------------------------------------
# compute_metrics tests
# ---------------------------------------------------------------------------

class TestComputeMetrics:
    def test_perfect_predictions(self, perfect_predictions):
        y_true, y_pred = perfect_predictions
        result = compute_metrics(y_true, y_pred)
        assert result["precision"] == 1.0
        assert result["recall"] == 1.0
        assert result["f1"] == 1.0
        assert result["fpr"] == 0.0
        assert result["fp"] == 0
        assert result["fn"] == 0

    def test_mixed_predictions_counts(self, mixed_predictions):
        y_true, y_pred = mixed_predictions
        result = compute_metrics(y_true, y_pred)
        assert result["tp"] == 4
        assert result["fp"] == 2
        assert result["fn"] == 1
        assert result["tn"] == 4

    def test_mixed_predictions_metrics(self, mixed_predictions):
        y_true, y_pred = mixed_predictions
        result = compute_metrics(y_true, y_pred)
        # precision = 4/6 ≈ 0.667
        assert abs(result["precision"] - 4/6) < 0.001
        # recall = 4/5 = 0.8
        assert abs(result["recall"] - 4/5) < 0.001

    def test_all_normal_no_crash(self, all_normal):
        y_true, y_pred = all_normal
        result = compute_metrics(y_true, y_pred)
        assert result["precision"] == 0.0
        assert result["recall"] == 0.0
        assert result["fpr"] == 0.0

    def test_all_flagged_high_fpr(self):
        y_true = np.zeros(100, dtype=np.int32)
        y_pred = np.ones(100, dtype=np.int32)
        result = compute_metrics(y_true, y_pred)
        assert result["fpr"] == 1.0

    def test_model_name_in_result(self, perfect_predictions):
        y_true, y_pred = perfect_predictions
        result = compute_metrics(y_true, y_pred, model_name="TestModel")
        assert result["model"] == "TestModel"

    def test_output_has_all_required_keys(self, mixed_predictions):
        y_true, y_pred = mixed_predictions
        result = compute_metrics(y_true, y_pred)
        required = {"model", "precision", "recall", "f1", "fpr", "tp", "fp", "tn", "fn",
                    "support_pos", "support_neg", "n_total", "flag_rate"}
        assert required.issubset(set(result.keys()))

    def test_flag_rate_matches_pred_mean(self, mixed_predictions):
        y_true, y_pred = mixed_predictions
        result = compute_metrics(y_true, y_pred)
        assert abs(result["flag_rate"] - y_pred.mean()) < 0.001


# ---------------------------------------------------------------------------
# compute_pr_curve tests
# ---------------------------------------------------------------------------

class TestComputePRCurve:
    def test_returns_lists(self):
        y_true = np.array([0, 0, 1, 1, 0], dtype=np.int32)
        scores = np.array([0.1, 0.2, 0.8, 0.9, 0.3], dtype=np.float32)
        result = compute_pr_curve(y_true, scores)
        assert isinstance(result["precision"], list)
        assert isinstance(result["recall"], list)
        assert isinstance(result["thresholds"], list)

    def test_avg_precision_between_0_and_1(self):
        y_true = np.array([0, 0, 0, 1, 1], dtype=np.int32)
        scores = np.array([0.1, 0.2, 0.3, 0.9, 0.8], dtype=np.float32)
        result = compute_pr_curve(y_true, scores)
        assert 0.0 <= result["avg_precision"] <= 1.0

    def test_no_positives_returns_nan(self):
        y_true = np.zeros(20, dtype=np.int32)
        scores = np.random.rand(20).astype(np.float32)
        result = compute_pr_curve(y_true, scores)
        assert result["precision"] == []
        assert result["recall"] == []
        import math
        assert math.isnan(result["avg_precision"])

    def test_perfect_scores_avg_precision_1(self):
        y_true = np.array([0, 0, 0, 1, 1], dtype=np.int32)
        # Perfect scores: anomalies get the highest scores
        scores = np.array([0.1, 0.2, 0.3, 1.0, 0.9], dtype=np.float32)
        result = compute_pr_curve(y_true, scores)
        assert result["avg_precision"] == 1.0


# ---------------------------------------------------------------------------
# evaluate_all tests
# ---------------------------------------------------------------------------

class TestEvaluateAll:
    def test_returns_four_configurations(self):
        y_true = np.array([0, 0, 1, 0, 1], dtype=np.int32)
        if_scores = np.array([0.1, 0.2, 0.9, 0.3, 0.8], dtype=np.float32)
        lstm_errors = np.array([0.01, 0.02, 0.08, 0.03, 0.07], dtype=np.float32)
        if_preds = np.array([0, 0, 1, 0, 1], dtype=np.int32)
        lstm_preds = np.array([0, 0, 1, 0, 0], dtype=np.int32)
        result = evaluate_all(y_true, if_scores, lstm_errors, if_preds, lstm_preds)
        assert len(result["metrics"]) == 4
        assert len(result["pr_curves"]) == 4
        assert len(result["confusion_matrices"]) == 4

    def test_union_catches_more_than_either_alone(self):
        y_true = np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)
        # IF catches index 1, LSTM catches index 3
        if_preds = np.array([0, 1, 0, 0, 0, 0], dtype=np.int32)
        lstm_preds = np.array([0, 0, 0, 1, 0, 0], dtype=np.int32)
        if_scores = if_preds.astype(np.float32)
        lstm_errors = lstm_preds.astype(np.float32)
        result = evaluate_all(y_true, if_scores, lstm_errors, if_preds, lstm_preds)
        models = {m["model"]: m for m in result["metrics"]}
        # Union should have higher recall than either alone
        union_recall = models["Combined (Union)"]["recall"]
        if_recall = models["Isolation Forest"]["recall"]
        lstm_recall = models["LSTM Autoencoder"]["recall"]
        assert union_recall >= if_recall
        assert union_recall >= lstm_recall

    def test_intersection_has_fewer_flags_than_union(self):
        y_true = np.array([0, 1, 0, 1, 0, 1], dtype=np.int32)
        if_preds = np.array([0, 1, 1, 1, 0, 0], dtype=np.int32)
        lstm_preds = np.array([0, 0, 1, 1, 0, 1], dtype=np.int32)
        if_scores = if_preds.astype(np.float32)
        lstm_errors = lstm_preds.astype(np.float32)
        result = evaluate_all(y_true, if_scores, lstm_errors, if_preds, lstm_preds)
        models = {m["model"]: m for m in result["metrics"]}
        union_flags = models["Combined (Union)"]["flag_rate"]
        inter_flags = models["Combined (Intersection)"]["flag_rate"]
        assert inter_flags <= union_flags

    def test_confusion_matrix_shape(self):
        y_true = np.array([0, 0, 1, 1], dtype=np.int32)
        preds = np.array([0, 1, 1, 0], dtype=np.int32)
        scores = preds.astype(np.float32)
        result = evaluate_all(y_true, scores, scores, preds, preds)
        for cm_dict in result["confusion_matrices"]:
            assert len(cm_dict["matrix"]) == 2
            assert len(cm_dict["matrix"][0]) == 2


# ---------------------------------------------------------------------------
# metrics_to_dataframe tests
# ---------------------------------------------------------------------------

class TestMetricsToDataframe:
    def test_returns_dataframe(self):
        metrics = [
            {"model": "A", "precision": 0.8, "recall": 0.7, "f1": 0.75,
             "fpr": 0.05, "tp": 7, "fp": 1, "tn": 19, "fn": 3, "flag_rate": 0.1},
        ]
        df = metrics_to_dataframe(metrics)
        assert isinstance(df, pd.DataFrame)
        assert "Model" in df.columns
        assert "F1" in df.columns

    def test_row_count_matches_input(self):
        metrics = [
            {"model": f"M{i}", "precision": 0.5, "recall": 0.5, "f1": 0.5,
             "fpr": 0.1, "tp": 1, "fp": 1, "tn": 9, "fn": 1, "flag_rate": 0.1}
            for i in range(4)
        ]
        df = metrics_to_dataframe(metrics)
        assert len(df) == 4


# ---------------------------------------------------------------------------
# save / load tests
# ---------------------------------------------------------------------------

class TestPersistence:
    def test_save_and_load_roundtrip(self, tmp_path):
        results = {"metrics": [{"model": "IF", "f1": 0.75}], "extra": 42}
        path = save_results(results, tmp_path)
        assert path.exists()
        loaded = load_results(tmp_path)
        assert loaded["extra"] == 42
        assert loaded["metrics"][0]["f1"] == 0.75

    def test_save_metrics_csv(self, tmp_path):
        metrics = [
            {"model": "IF", "precision": 0.8, "recall": 0.7, "f1": 0.75,
             "fpr": 0.05, "tp": 7, "fp": 1, "tn": 19, "fn": 3, "flag_rate": 0.1},
        ]
        path = save_metrics_csv(metrics, tmp_path)
        assert path.exists()
        df = pd.read_csv(path)
        assert "F1" in df.columns

    def test_load_raises_on_missing_file(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_results(tmp_path, "nonexistent.json")

    @pytest.mark.skipif(not RESULTS_DIR.exists(), reason="results/ not present")
    def test_load_real_results_from_disk(self):
        results = load_results(RESULTS_DIR)
        assert "test_split_evaluation" in results
        assert "spike_scenario_evaluation" in results
        assert "model_config" in results

    @pytest.mark.skipif(not (RESULTS_DIR / "metrics_summary.csv").exists(), reason="metrics CSV not present")
    def test_metrics_csv_has_four_rows(self):
        df = pd.read_csv(RESULTS_DIR / "metrics_summary.csv")
        assert len(df) == 4
        assert set(df["Model"]) == {
            "Isolation Forest", "LSTM Autoencoder",
            "Combined (Union)", "Combined (Intersection)"
        }

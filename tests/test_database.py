"""
Unit tests for the SQLite persistence layer (src/api/database.py).

The autouse _clean_db fixture in conftest.py resets tables before each test,
and conftest points the engine at a throwaway temp database.
"""

from __future__ import annotations

from src.api import database as db


def _make_prediction(union: bool = True, if_flag: bool = True, lstm_flag: bool = False) -> int:
    return db.log_prediction(
        series_key="realAWSCloudwatch/ec2_cpu_utilization_825cc2.csv",
        metric_value=97.5,
        if_score=0.42,
        lstm_error=0.06,
        if_flag=if_flag,
        lstm_flag=lstm_flag,
        combined_union=union,
        combined_intersection=if_flag and lstm_flag,
    )


class TestPersistence:
    def test_log_prediction_returns_id(self):
        pid = _make_prediction()
        assert isinstance(pid, int) and pid > 0

    def test_log_alert_links_to_prediction(self):
        pid = _make_prediction()
        aid = db.log_alert(
            prediction_id=pid,
            series_key="realAWSCloudwatch/ec2_cpu_utilization_825cc2.csv",
            metric_value=97.5,
            if_score=0.42,
            lstm_error=0.06,
            if_flag=True,
            lstm_flag=False,
            sent=False,
        )
        assert aid > 0
        alerts = db.recent_alerts(limit=5)
        assert len(alerts) == 1
        assert alerts[0]["prediction_id"] == pid

    def test_detected_by_label(self):
        pid = _make_prediction(if_flag=True, lstm_flag=True)
        db.log_alert(
            prediction_id=pid, series_key="x", metric_value=1.0,
            if_score=0.1, lstm_error=0.1, if_flag=True, lstm_flag=True, sent=True,
        )
        assert db.recent_alerts(1)[0]["detected_by"] == "IF+LSTM"

    def test_recent_alerts_newest_first(self):
        for _ in range(3):
            pid = _make_prediction()
            db.log_alert(
                prediction_id=pid, series_key="x", metric_value=1.0,
                if_score=0.1, lstm_error=0.1, if_flag=True, lstm_flag=False, sent=False,
            )
        alerts = db.recent_alerts(limit=10)
        ids = [a["id"] for a in alerts]
        assert ids == sorted(ids, reverse=True)

    def test_get_stats_counts(self):
        pid = _make_prediction()
        db.log_alert(
            prediction_id=pid, series_key="x", metric_value=1.0,
            if_score=0.1, lstm_error=0.1, if_flag=True, lstm_flag=False, sent=True,
        )
        stats = db.get_stats()
        assert stats["total_predictions"] == 1
        assert stats["total_alerts"] == 1
        assert stats["alerts_sent"] == 1
        assert stats["anomalies_detected"] == 1

    def test_reset_db_clears_rows(self):
        _make_prediction()
        assert db.get_stats()["total_predictions"] == 1
        db.reset_db()
        assert db.get_stats()["total_predictions"] == 0

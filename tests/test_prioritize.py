from __future__ import annotations

import pandas as pd

from feedback.prioritize import _risk_signals, _row_urgency, build_issue_metrics


def test_risk_signals_detect_payment_and_trust() -> None:
    text = "subscription lene ke baad bhi ads arahe hain, wrong hadith reference"
    s = _risk_signals(text)
    assert s["signal_payment"] == 1
    assert s["signal_trust"] == 1


def test_row_urgency_boosted_for_high_negative_blocker() -> None:
    high = _row_urgency(
        "High",
        "Negative",
        {
            "signal_payment": 1,
            "signal_trust": 1,
            "signal_blocker": 1,
            "signal_religious": 1,
        },
    )
    low = _row_urgency(
        "Low",
        "Positive",
        {
            "signal_payment": 0,
            "signal_trust": 0,
            "signal_blocker": 0,
            "signal_religious": 0,
        },
    )
    assert high > low
    assert 0.0 <= low <= 1.0
    assert 0.0 <= high <= 1.0


def test_build_issue_metrics_growth_prefers_recent_cluster() -> None:
    rows = pd.DataFrame(
        {
            "row_id": [1, 2, 3, 4, 5, 6, 7, 8],
            "cluster_id": [10, 10, 10, 20, 20, 20, 20, 20],
            "parent_id": [1, 1, 1, 2, 2, 2, 2, 2],
            "text_clean": [
                "old complaint",
                "old complaint",
                "old complaint",
                "subscription ads issue",
                "subscription ads issue",
                "subscription ads issue",
                "subscription ads issue",
                "subscription ads issue",
            ],
            "Severity": ["Low", "Low", "Low", "High", "High", "High", "High", "High"],
            "Sentiment": ["Neutral", "Neutral", "Neutral", "Negative", "Negative", "Negative", "Negative", "Negative"],
        }
    )
    art = build_issue_metrics(rows, use_cache=False)
    out = art.cluster_metrics.set_index("cluster_id")
    assert out.loc[20, "priority_score"] > out.loc[10, "priority_score"]

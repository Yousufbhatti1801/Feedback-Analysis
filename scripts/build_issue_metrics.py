"""Build PM-facing issue prioritization metrics.

Run:
    python scripts/build_issue_metrics.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from feedback.ingest import load_feedback
from feedback.normalize import normalize_dataframe
from feedback.prioritize import build_issue_metrics


def main() -> None:
    df = normalize_dataframe(load_feedback())
    clusters = pd.read_parquet(Path(__file__).resolve().parents[1] / "data" / "cache" / "clusters.parquet")
    rows = df.merge(clusters, on="row_id", how="inner")
    artifact = build_issue_metrics(rows, use_cache=False)

    print("Top 15 priority clusters")
    print(
        artifact.cluster_metrics[
            ["cluster_id", "priority_score", "urgency_score", "growth_rate", "count", "explain_reasons"]
        ]
        .head(15)
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()

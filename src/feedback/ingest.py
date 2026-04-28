"""CSV ingestion for the Islam360 feedback export.

The raw export ships with a UTF-8 BOM and an unnamed first column (the row
index from the original export). This module loads it deterministically and
exposes a typed DataFrame with predictable column names.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CSV = REPO_ROOT / "2026-04-24T10-29_export.csv"

EXPECTED_COLUMNS = (
    "Topic",
    "Feedback Comment",
    "Contextual_Subtopic",
    "Granular_Subtopic",
    "Severity",
    "Sentiment",
)

VALID_SEVERITY = {"Low", "Medium", "High"}
VALID_SENTIMENT = {"Positive", "Neutral", "Negative"}


def load_feedback(path: Path | str | None = None) -> pd.DataFrame:
    """Load the feedback CSV with strict schema validation.

    Returns a DataFrame with a clean integer ``row_id`` (taken from the
    unnamed index column in the export) and the six labeled columns above.
    """
    csv_path = Path(path) if path is not None else DEFAULT_CSV
    if not csv_path.exists():
        raise FileNotFoundError(f"Feedback CSV not found at {csv_path}")

    df = pd.read_csv(csv_path, encoding="utf-8-sig", dtype=str, keep_default_na=False)

    if df.columns[0] != "" and df.columns[0] != "Unnamed: 0":
        raise ValueError(
            f"Expected unnamed first column from export, got: {df.columns[0]!r}"
        )
    df = df.rename(columns={df.columns[0]: "row_id"})
    df["row_id"] = df["row_id"].astype(int)

    missing = [c for c in EXPECTED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing expected columns: {missing}")

    bad_severity = set(df["Severity"].unique()) - VALID_SEVERITY
    if bad_severity:
        raise ValueError(f"Unexpected Severity values: {bad_severity}")

    bad_sentiment = set(df["Sentiment"].unique()) - VALID_SENTIMENT
    if bad_sentiment:
        raise ValueError(f"Unexpected Sentiment values: {bad_sentiment}")

    return df


def summarize(df: pd.DataFrame) -> dict:
    """Compact stats dict — useful for sanity prints in scripts and notebooks."""
    return {
        "rows": len(df),
        "topic_counts": df["Topic"].value_counts().to_dict(),
        "severity_counts": df["Severity"].value_counts().to_dict(),
        "sentiment_counts": df["Sentiment"].value_counts().to_dict(),
        "unique_contextual_subtopic": df["Contextual_Subtopic"].nunique(),
        "unique_granular_subtopic": df["Granular_Subtopic"].nunique(),
        "blank_feedback": int((df["Feedback Comment"].str.strip() == "").sum()),
    }


if __name__ == "__main__":
    df = load_feedback()
    stats = summarize(df)
    for k, v in stats.items():
        print(f"{k}: {v}")

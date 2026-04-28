"""Issue prioritization metrics for discovered feedback clusters.

This module is intentionally downstream of semantic clustering. It does *not*
decide issue categories; it scores already-discovered clusters for PM triage.

Outputs include:
- frequency and share
- urgency score
- growth score (time-based if available, otherwise row-order buckets)
- composite priority score
- explainability fields (severity/sentiment mix + lexical risk signals)
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from feedback.cluster import NOISE_LABEL
from feedback.ingest import REPO_ROOT

CACHE_DIR = REPO_ROOT / "data" / "cache"
ISSUE_METRICS_PARQUET = CACHE_DIR / "issue_metrics.parquet"
PARENT_METRICS_PARQUET = CACHE_DIR / "parent_issue_metrics.parquet"

_PAYMENT_TERMS = re.compile(
    r"\b(subscription|premium|paid|payment|pay|refund|charge|charged|billing|"
    r"renew|renewal|card|jazzcash|easypaisa|invoice|trial)\b",
    re.IGNORECASE,
)
_TRUST_TERMS = re.compile(
    r"\b(wrong|incorrect|error|fake|authentic|inauthentic|misleading|"
    r"reference|dalil|proof|sahih|zaeef|daif|weak|haram|halal|"
    r"ghalat|galat|behuda|adult|vulgar|inappropriate)\b",
    re.IGNORECASE,
)
_BLOCKER_TERMS = re.compile(
    r"\b(crash|hang|stuck|freeze|frozen|not work|doesn't work|won't open|"
    r"load fail|blank screen|error|failed|otp|login issue|can't login|"
    r"nahi chal|band ho|ruk jata|atak)\b",
    re.IGNORECASE,
)
_RELIGIOUS_SENSITIVITY_TERMS = re.compile(
    r"\b(quran|qur'an|hadith|hadees|tafseer|dua|allah|namaz|azan|"
    r"surah|aayat|ayat|deen|islam)\b",
    re.IGNORECASE,
)

_SEVERITY_SCORE = {"Low": 0.2, "Medium": 0.55, "High": 1.0}
_SENTIMENT_SCORE = {"Positive": 0.05, "Neutral": 0.35, "Negative": 1.0}


@dataclass
class IssueMetricsArtifact:
    cluster_metrics: pd.DataFrame
    parent_metrics: pd.DataFrame


def _choose_time_col(df: pd.DataFrame) -> str | None:
    """Return first usable datetime-like column if present; else None."""
    candidates = [
        "created_at",
        "timestamp",
        "date",
        "datetime",
        "submitted_at",
        "feedback_time",
    ]
    lower_map = {c.lower(): c for c in df.columns}
    for key in candidates:
        if key in lower_map:
            return lower_map[key]
    return None


def _time_buckets(df: pd.DataFrame, *, n_buckets: int = 8) -> pd.Series:
    """Build temporal buckets from real timestamps when available.

    Fallback: row-order buckets based on ``row_id`` quantiles. This keeps trend
    metrics available even when exports do not include explicit timestamps.
    """
    time_col = _choose_time_col(df)
    if time_col is not None:
        parsed = pd.to_datetime(df[time_col], errors="coerce", utc=True)
        if parsed.notna().sum() >= max(100, int(0.6 * len(df))):
            ranks = parsed.rank(method="first")
            return pd.qcut(ranks, q=n_buckets, labels=False, duplicates="drop").astype(int)

    order_series = df["row_id"] if "row_id" in df.columns else pd.Series(np.arange(len(df)))
    return pd.qcut(order_series.rank(method="first"), q=n_buckets, labels=False, duplicates="drop").astype(int)


def _risk_signals(text: str) -> dict[str, int]:
    text = text or ""
    payment = int(_PAYMENT_TERMS.search(text) is not None)
    trust = int(_TRUST_TERMS.search(text) is not None)
    blocker = int(_BLOCKER_TERMS.search(text) is not None)
    religious = int(_RELIGIOUS_SENSITIVITY_TERMS.search(text) is not None)
    return {
        "signal_payment": payment,
        "signal_trust": trust,
        "signal_blocker": blocker,
        "signal_religious": religious,
    }


def _row_urgency(severity: str, sentiment: str, signals: dict[str, int]) -> float:
    sev = _SEVERITY_SCORE.get(severity, 0.4)
    sent = _SENTIMENT_SCORE.get(sentiment, 0.35)
    payment = signals["signal_payment"]
    trust = signals["signal_trust"]
    blocker = signals["signal_blocker"]
    religious = signals["signal_religious"]

    # Religious term alone is not bad; religious + trust issue is extra risky.
    religious_trust = 1.0 if (religious and trust) else 0.0

    score = (
        0.40 * sev
        + 0.26 * sent
        + 0.12 * payment
        + 0.12 * blocker
        + 0.08 * trust
        + 0.02 * religious_trust
    )
    return float(np.clip(score, 0.0, 1.0))


def _growth_components(cluster_rows: pd.DataFrame, bucket_col: str) -> tuple[float, float, float]:
    counts = (
        cluster_rows.groupby(bucket_col)
        .size()
        .reindex(sorted(cluster_rows[bucket_col].unique()), fill_value=0)
    )
    if len(counts) < 2:
        return 0.0, 0.0, 0.0

    recent = float(counts.iloc[-1])
    previous = float(counts.iloc[-2])
    growth_rate = (recent - previous) / max(previous, 1.0)

    # Positive growth gets mapped to [0,1], shrinkage maps below 0.5.
    growth_score = 1.0 / (1.0 + math.exp(-growth_rate))
    return growth_rate, growth_score, recent


def build_issue_metrics(
    rows: pd.DataFrame,
    *,
    cluster_col: str = "cluster_id",
    parent_col: str = "parent_id",
    text_col: str = "text_clean",
    use_cache: bool = True,
) -> IssueMetricsArtifact:
    """Compute PM-facing prioritization metrics for discovered clusters."""
    if use_cache and ISSUE_METRICS_PARQUET.exists() and PARENT_METRICS_PARQUET.exists():
        return IssueMetricsArtifact(
            cluster_metrics=pd.read_parquet(ISSUE_METRICS_PARQUET),
            parent_metrics=pd.read_parquet(PARENT_METRICS_PARQUET),
        )

    df = rows.copy()
    df = df[df[cluster_col] != NOISE_LABEL].copy()

    signal_frame = df[text_col].fillna("").map(_risk_signals).apply(pd.Series)
    df = pd.concat([df, signal_frame], axis=1)
    df["row_urgency"] = [
        _row_urgency(sev, sent, sig)
        for sev, sent, sig in zip(
            df["Severity"],
            df["Sentiment"],
            signal_frame.to_dict(orient="records"),
        )
    ]
    df["time_bucket"] = _time_buckets(df)

    total_rows = max(len(df), 1)
    cluster_rows: list[dict] = []

    for cid, g in df.groupby(cluster_col):
        count = len(g)
        share = count / total_rows
        urgency = float(g["row_urgency"].mean())
        pct_high = float((g["Severity"] == "High").mean())
        pct_negative = float((g["Sentiment"] == "Negative").mean())

        growth_rate, growth_score, recent = _growth_components(g, "time_bucket")

        impact = np.clip(np.log1p(count) / np.log1p(total_rows), 0.0, 1.0)
        priority = 100.0 * (
            0.55 * urgency
            + 0.25 * growth_score
            + 0.20 * impact
        )

        reasons: list[str] = []
        if pct_high >= 0.2:
            reasons.append("high severity concentration")
        if pct_negative >= 0.5:
            reasons.append("mostly negative sentiment")
        if g["signal_payment"].mean() >= 0.2:
            reasons.append("payment/subscription risk")
        if g["signal_blocker"].mean() >= 0.15:
            reasons.append("blocker/crash language")
        if g["signal_trust"].mean() >= 0.2 and g["signal_religious"].mean() >= 0.2:
            reasons.append("religious trust sensitivity")
        if growth_rate > 0.25:
            reasons.append("rapid recent growth")
        if not reasons:
            reasons.append("frequency-driven")

        cluster_rows.append(
            {
                "cluster_id": int(cid),
                "parent_id": int(g[parent_col].mode().iloc[0]),
                "count": int(count),
                "share": float(share),
                "urgency_score": float(urgency),
                "growth_rate": float(growth_rate),
                "growth_score": float(growth_score),
                "priority_score": float(priority),
                "recent_count": int(recent),
                "pct_high": float(pct_high),
                "pct_negative": float(pct_negative),
                "signal_payment_rate": float(g["signal_payment"].mean()),
                "signal_trust_rate": float(g["signal_trust"].mean()),
                "signal_blocker_rate": float(g["signal_blocker"].mean()),
                "signal_religious_rate": float(g["signal_religious"].mean()),
                "explain_reasons": "; ".join(reasons),
            }
        )

    cluster_metrics = (
        pd.DataFrame(cluster_rows)
        .sort_values(["priority_score", "count"], ascending=[False, False])
        .reset_index(drop=True)
    )

    parent_metrics = (
        cluster_metrics.groupby("parent_id", as_index=False)
        .agg(
            total_count=("count", "sum"),
            mean_priority=("priority_score", "mean"),
            max_priority=("priority_score", "max"),
            mean_urgency=("urgency_score", "mean"),
            mean_growth=("growth_rate", "mean"),
            high_priority_clusters=("priority_score", lambda s: int((s >= 70).sum())),
        )
        .sort_values(["max_priority", "total_count"], ascending=[False, False])
        .reset_index(drop=True)
    )

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cluster_metrics.to_parquet(ISSUE_METRICS_PARQUET, index=False)
    parent_metrics.to_parquet(PARENT_METRICS_PARQUET, index=False)

    return IssueMetricsArtifact(
        cluster_metrics=cluster_metrics,
        parent_metrics=parent_metrics,
    )

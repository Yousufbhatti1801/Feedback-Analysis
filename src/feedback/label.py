"""Cluster labeling for the Phase 3 dashboard.

Two modes:
- ``terms``: deterministic c-TF-IDF top-term labels. Zero cost, no API.
- ``claude``: Claude Sonnet generates a PM-readable label, description,
  example quote, and suspected root cause from MMR-selected representatives
  plus the c-TF-IDF terms. Reads ``ANTHROPIC_API_KEY`` from env.

Output: ``data/cache/labels.parquet`` with one row per cluster_id, plus a
parents table at ``data/cache/parent_labels.parquet`` derived from the
member clusters' labels.
"""

from __future__ import annotations

import json
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize as l2_normalize
from tqdm.auto import tqdm

from feedback.cluster import NOISE_LABEL
from feedback.ingest import REPO_ROOT

CACHE_DIR = REPO_ROOT / "data" / "cache"
LABELS_PARQUET = CACHE_DIR / "labels.parquet"
PARENT_LABELS_PARQUET = CACHE_DIR / "parent_labels.parquet"

CLAUDE_MODEL = "claude-sonnet-4-6"
N_REPRESENTATIVES = 8
N_TFIDF_TERMS = 10
MMR_LAMBDA = 0.5

# A few stopwords on top of sklearn's English list — Roman-Urdu padding tokens
# that appear across many clusters and dilute the c-TF-IDF signal.
EXTRA_STOPWORDS = {
    "app", "apps", "plz", "pls", "please", "kindly", "good", "nice", "best",
    "thanks", "thx", "ka", "ki", "ke", "ko", "se", "hai", "hain", "main",
    "mein", "ko", "aur", "ya", "or", "and", "the", "to", "of", "in", "is",
    "it", "for", "on", "this", "that", "be", "with", "should", "would",
    "could", "can", "may", "must", "have", "has", "had", "are", "was", "were",
}


@dataclass
class ClusterContext:
    cluster_id: int
    size: int
    representatives: list[str]  # MMR-selected raw feedback comments
    top_terms: list[str]
    sentiment_mix: dict[str, int]
    severity_mix: dict[str, int]


# ---------- Representative selection (MMR over centroid) ----------


def mmr_select(
    cluster_vectors: np.ndarray,
    cluster_indices: np.ndarray,
    *,
    k: int = N_REPRESENTATIVES,
    lambda_: float = MMR_LAMBDA,
) -> np.ndarray:
    """Pick ``k`` diverse-yet-central indices via Maximal Marginal Relevance.

    cluster_vectors are L2-normalized so dot product = cosine sim.
    Returns positional indices into ``cluster_indices``.
    """
    if len(cluster_indices) <= k:
        return np.arange(len(cluster_indices))

    centroid = cluster_vectors.mean(axis=0)
    centroid /= np.linalg.norm(centroid) + 1e-12
    rel = cluster_vectors @ centroid  # similarity to centroid

    selected: list[int] = [int(np.argmax(rel))]
    candidates = set(range(len(cluster_indices))) - set(selected)

    while len(selected) < k and candidates:
        cand = np.array(sorted(candidates))
        max_sim_to_selected = (cluster_vectors[cand] @ cluster_vectors[selected].T).max(
            axis=1
        )
        score = lambda_ * rel[cand] - (1 - lambda_) * max_sim_to_selected
        choice = int(cand[np.argmax(score)])
        selected.append(choice)
        candidates.discard(choice)

    return np.array(selected)


# ---------- c-TF-IDF top terms per cluster ----------


def cluster_top_terms(
    df: pd.DataFrame,
    *,
    text_col: str = "text_clean",
    cluster_col: str = "cluster_id",
    n_terms: int = N_TFIDF_TERMS,
) -> dict[int, list[str]]:
    """Aggregate text per cluster, run TF-IDF over the cluster-level corpus.

    Standard c-TF-IDF construction: each cluster becomes one "document",
    term-frequency is counted within the cluster, IDF is over clusters.
    """
    grouped = (
        df[df[cluster_col] != NOISE_LABEL]
        .groupby(cluster_col)[text_col]
        .apply(lambda s: " ".join(s.tolist()))
    )
    if grouped.empty:
        return {}

    stopwords = "english"
    vectorizer = TfidfVectorizer(
        max_features=10_000,
        ngram_range=(1, 2),
        stop_words=stopwords,
        min_df=2,
        token_pattern=r"(?u)\b[A-Za-z]{3,}\b",
    )
    matrix = vectorizer.fit_transform(grouped.values)
    terms = np.array(vectorizer.get_feature_names_out())

    out: dict[int, list[str]] = {}
    for row_idx, cid in enumerate(grouped.index):
        row = matrix.getrow(row_idx).toarray().ravel()
        # Filter our extra stopwords post-hoc.
        order = np.argsort(-row)
        picked: list[str] = []
        for j in order:
            if row[j] <= 0:
                break
            term = terms[j]
            if term in EXTRA_STOPWORDS:
                continue
            if any(part in EXTRA_STOPWORDS for part in term.split()):
                continue
            picked.append(term)
            if len(picked) >= n_terms:
                break
        out[int(cid)] = picked
    return out


# ---------- Build per-cluster context ----------


def build_cluster_contexts(
    df: pd.DataFrame,
    vectors: np.ndarray,
    *,
    text_col: str = "Feedback Comment",
    clean_col: str = "text_clean",
    cluster_col: str = "cluster_id",
) -> list[ClusterContext]:
    """For each cluster, collect MMR representatives, top terms, and label mix."""
    top_terms = cluster_top_terms(df, text_col=clean_col, cluster_col=cluster_col)

    contexts: list[ClusterContext] = []
    for cid, group in df.groupby(cluster_col):
        cid_int = int(cid)
        if cid_int == NOISE_LABEL:
            continue
        idx = group.index.to_numpy()
        cluster_vectors = l2_normalize(vectors[idx])  # ensure normalized
        chosen = mmr_select(cluster_vectors, idx)
        chosen_rows = group.iloc[chosen]
        contexts.append(
            ClusterContext(
                cluster_id=cid_int,
                size=len(group),
                representatives=chosen_rows[text_col].tolist(),
                top_terms=top_terms.get(cid_int, []),
                sentiment_mix=dict(group["Sentiment"].value_counts()),
                severity_mix=dict(group["Severity"].value_counts()),
            )
        )
    contexts.sort(key=lambda c: -c.size)
    return contexts


# ---------- Mode: terms-only label ----------


def label_from_terms(ctx: ClusterContext) -> dict:
    """Deterministic fallback label: top 4 c-TF-IDF terms, joined."""
    terms = ctx.top_terms[:4] if ctx.top_terms else ["(no terms)"]
    short = ", ".join(terms)
    description = (
        f"Cluster of {ctx.size} feedback comments centered on: {short}. "
        f"Top sentiment: {max(ctx.sentiment_mix, key=ctx.sentiment_mix.get)}; "
        f"top severity: {max(ctx.severity_mix, key=ctx.severity_mix.get)}."
    )
    return {
        "cluster_id": ctx.cluster_id,
        "label": short,
        "description": description,
        "example_quote": ctx.representatives[0] if ctx.representatives else "",
        "suspected_root_cause": "",
        "label_mode": "terms",
    }


# ---------- Mode: Claude label ----------


_CLAUDE_PROMPT = """You are summarizing a cluster of similar user feedback comments for a Product Manager of Islam360, a religious mobile app (Quran, Hadith, prayer features). The PM needs labels that name the artifact AND the problem so engineering knows exactly what to fix.

CLUSTER FACTS
- Size: {size} comments
- Top terms (c-TF-IDF, English stems only — comments are mixed Urdu/Roman Urdu/English): {terms}
- Sentiment mix: {sentiment}
- Severity mix: {severity}

REPRESENTATIVE COMMENTS (raw, may be Urdu / Roman Urdu / English):
{representatives}

Return STRICT JSON with exactly these keys:
{{
  "label": "<= 6 English words, names the artifact + problem (good: 'Ads cover Quran verses'; bad: 'Ads issue')",
  "description": "one sentence, plain English, what users are reporting",
  "example_quote": "the most representative single comment from the list above, copied verbatim, max 200 chars",
  "suspected_root_cause": "one short phrase the engineering team can investigate (good: 'subscription state not propagated to ad SDK'; bad: 'unknown')"
}}

Output JSON only — no preamble, no markdown, no code fences."""


def label_with_claude(
    contexts: list[ClusterContext],
    *,
    model: str = CLAUDE_MODEL,
    max_retries: int = 3,
    sleep_between: float = 0.0,
) -> list[dict]:
    """Label every cluster via Claude. Reads ``ANTHROPIC_API_KEY`` from env."""
    if "ANTHROPIC_API_KEY" not in os.environ:
        raise RuntimeError(
            "ANTHROPIC_API_KEY not set. Run: export ANTHROPIC_API_KEY=sk-ant-..."
        )
    from anthropic import Anthropic

    client = Anthropic()
    out: list[dict] = []
    for ctx in tqdm(contexts, desc="claude labeling"):
        prompt = _CLAUDE_PROMPT.format(
            size=ctx.size,
            terms=", ".join(ctx.top_terms) or "(none)",
            sentiment=", ".join(f"{k}={v}" for k, v in ctx.sentiment_mix.items()),
            severity=", ".join(f"{k}={v}" for k, v in ctx.severity_mix.items()),
            representatives="\n".join(
                f"- {r[:300]}" for r in ctx.representatives
            ),
        )
        parsed = _claude_call(client, model, prompt, max_retries=max_retries)
        parsed["cluster_id"] = ctx.cluster_id
        parsed["label_mode"] = "claude"
        out.append(parsed)
        if sleep_between:
            time.sleep(sleep_between)
    return out


def _claude_call(client, model: str, prompt: str, *, max_retries: int) -> dict:
    last_err: Exception | None = None
    for attempt in range(max_retries):
        try:
            resp = client.messages.create(
                model=model,
                max_tokens=400,
                messages=[{"role": "user", "content": prompt}],
            )
            text = resp.content[0].text.strip()
            text = re.sub(r"^```(?:json)?|```$", "", text, flags=re.MULTILINE).strip()
            data = json.loads(text)
            for k in ("label", "description", "example_quote", "suspected_root_cause"):
                data.setdefault(k, "")
            return {k: data[k] for k in ("label", "description", "example_quote", "suspected_root_cause")}
        except Exception as e:
            last_err = e
            time.sleep(1.5 ** attempt)
    raise RuntimeError(f"Claude labeling failed after {max_retries} attempts: {last_err}")


# ---------- Orchestration ----------


def label_clusters(
    df: pd.DataFrame,
    vectors: np.ndarray,
    *,
    mode: Literal["terms", "claude"] = "terms",
    cluster_col: str = "cluster_id",
) -> pd.DataFrame:
    """Build labels for every non-noise cluster."""
    contexts = build_cluster_contexts(df, vectors, cluster_col=cluster_col)
    print(f"[label] {len(contexts)} clusters; mode={mode}")

    if mode == "terms":
        rows = [label_from_terms(c) for c in contexts]
    else:
        claude_rows = label_with_claude(contexts)
        # Attach size and top terms for the dashboard's convenience.
        ctx_by_id = {c.cluster_id: c for c in contexts}
        rows = []
        for r in claude_rows:
            cid = r["cluster_id"]
            r["size"] = ctx_by_id[cid].size
            r["top_terms"] = ", ".join(ctx_by_id[cid].top_terms)
            rows.append(r)

    out = pd.DataFrame(rows)
    if "size" not in out.columns:
        size_by_id = {c.cluster_id: c.size for c in contexts}
        out["size"] = out["cluster_id"].map(size_by_id)
    if "top_terms" not in out.columns:
        terms_by_id = {c.cluster_id: ", ".join(c.top_terms) for c in contexts}
        out["top_terms"] = out["cluster_id"].map(terms_by_id)
    out = out.sort_values("size", ascending=False).reset_index(drop=True)

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(LABELS_PARQUET, index=False)
    print(f"[label] wrote {LABELS_PARQUET.name}")
    return out


def label_parents(clusters_df: pd.DataFrame, labels_df: pd.DataFrame) -> pd.DataFrame:
    """Derive a one-line parent theme label by stitching together its top member clusters."""
    merged = clusters_df.merge(
        labels_df[["cluster_id", "label", "size"]], on="cluster_id", how="left"
    )
    rows = []
    for parent_id, group in merged[merged["parent_id"] != NOISE_LABEL].groupby("parent_id"):
        unique_clusters = (
            group.dropna(subset=["label"])
            .drop_duplicates("cluster_id")
            .sort_values("size", ascending=False)
        )
        top_labels = unique_clusters["label"].head(3).tolist()
        rows.append(
            {
                "parent_id": int(parent_id),
                "rows": int(len(group)),
                "n_clusters": int(unique_clusters["cluster_id"].nunique()),
                "label": " · ".join(top_labels) if top_labels else "(unlabeled)",
            }
        )
    out = pd.DataFrame(rows).sort_values("rows", ascending=False).reset_index(drop=True)
    out.to_parquet(PARENT_LABELS_PARQUET, index=False)
    print(f"[label] wrote {PARENT_LABELS_PARQUET.name}")
    return out


if __name__ == "__main__":
    import argparse

    from feedback.embed import embed_texts
    from feedback.ingest import load_feedback
    from feedback.normalize import normalize_dataframe

    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["terms", "claude"], default="terms")
    args = parser.parse_args()

    df = normalize_dataframe(load_feedback())
    art = embed_texts(df)
    clusters = pd.read_parquet(CACHE_DIR / "clusters.parquet")
    merged = df.merge(clusters, on="row_id")

    labels = label_clusters(merged, art.vectors, mode=args.mode)
    parents = label_parents(clusters, labels)
    print(f"\nTop 10 clusters by size:")
    print(labels[["cluster_id", "size", "label"]].head(10).to_string(index=False))
    print(f"\nParent themes:")
    print(parents.to_string(index=False))

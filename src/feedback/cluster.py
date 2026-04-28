"""Phase 3 clustering pipeline.

Discovers a dynamic taxonomy from the e5-large embeddings:

    embeddings → MinHash dedup → UMAP (10-d) → HDBSCAN → centroid hierarchy

Outputs a single ``clusters.parquet`` with one row per feedback row, carrying
``cluster_id``, ``parent_id``, ``duplicate_group_id``, and the 2-d UMAP
projection used by the dashboard scatter plot.

All randomness is seeded — re-running yields the same partition.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

import hdbscan
import numpy as np
import pandas as pd
import umap
from datasketch import MinHash, MinHashLSH
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import normalize as l2_normalize

from feedback.embed import EmbeddingArtifact
from feedback.ingest import REPO_ROOT

CACHE_DIR = REPO_ROOT / "data" / "cache"
CLUSTERS_PARQUET = CACHE_DIR / "clusters.parquet"

SEED = 42
NOISE_LABEL = -1

# Default knobs from the approved plan.
UMAP_N_COMPONENTS = 10
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.0
HDBSCAN_MIN_CLUSTER_SIZE = 10
HDBSCAN_MIN_SAMPLES = 3
HDBSCAN_SELECTION_METHOD = "eom"
SOFT_ASSIGN_MIN_SIM = 0.0  # always promote; the dashboard filters on `soft_sim` at runtime
N_PARENT_THEMES = 12

MINHASH_NUM_PERM = 128
MINHASH_THRESHOLD = 0.85
NGRAM_SIZE = 5

_TOKEN_RE = re.compile(r"\s+")


# ---------- MinHash near-duplicate detection ----------


def _char_ngrams(text: str, n: int = NGRAM_SIZE) -> set[bytes]:
    """Character n-grams over a normalized string. Empty for very short text."""
    cleaned = _TOKEN_RE.sub(" ", text.lower()).strip()
    if len(cleaned) < n:
        return {cleaned.encode("utf-8")} if cleaned else set()
    return {cleaned[i : i + n].encode("utf-8") for i in range(len(cleaned) - n + 1)}


def _minhash(text: str) -> MinHash:
    m = MinHash(num_perm=MINHASH_NUM_PERM, seed=SEED)
    for gram in _char_ngrams(text):
        m.update(gram)
    return m


def assign_duplicate_groups(
    df: pd.DataFrame,
    *,
    text_col: str = "text_clean",
    id_col: str = "row_id",
    threshold: float = MINHASH_THRESHOLD,
) -> pd.Series:
    """Assign each row a ``duplicate_group_id``. Distinct rows get distinct ids.

    Two rows share a group iff Jaccard similarity on character 5-grams ≥ threshold.
    Group ids are the smallest row_id in the group — stable and human-readable.
    """
    lsh = MinHashLSH(threshold=threshold, num_perm=MINHASH_NUM_PERM)
    minhashes: dict[int, MinHash] = {}

    for row_id, text in zip(df[id_col].tolist(), df[text_col].tolist()):
        m = _minhash(text)
        minhashes[int(row_id)] = m
        lsh.insert(int(row_id), m)

    parent: dict[int, int] = {rid: rid for rid in minhashes}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(a: int, b: int) -> None:
        ra, rb = find(a), find(b)
        if ra == rb:
            return
        if ra < rb:
            parent[rb] = ra
        else:
            parent[ra] = rb

    for row_id, m in minhashes.items():
        for hit in lsh.query(m):
            if hit != row_id:
                union(int(row_id), int(hit))

    return df[id_col].astype(int).map(lambda r: find(int(r)))


# ---------- UMAP + HDBSCAN ----------


def umap_reduce(
    vectors: np.ndarray,
    *,
    n_components: int,
    n_neighbors: int = UMAP_N_NEIGHBORS,
    min_dist: float = UMAP_MIN_DIST,
    seed: int = SEED,
) -> np.ndarray:
    reducer = umap.UMAP(
        n_components=n_components,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=seed,
        verbose=False,
    )
    return reducer.fit_transform(vectors).astype(np.float32)


def hdbscan_cluster(
    reduced: np.ndarray,
    *,
    min_cluster_size: int = HDBSCAN_MIN_CLUSTER_SIZE,
    min_samples: int = HDBSCAN_MIN_SAMPLES,
) -> np.ndarray:
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=min_cluster_size,
        min_samples=min_samples,
        cluster_selection_method=HDBSCAN_SELECTION_METHOD,
        metric="euclidean",
        core_dist_n_jobs=1,
    )
    return clusterer.fit_predict(reduced).astype(np.int32)


# ---------- Soft assignment of noise points to nearest cluster ----------


def soft_assign_noise(
    vectors: np.ndarray,
    cluster_labels: np.ndarray,
    *,
    min_sim: float = SOFT_ASSIGN_MIN_SIM,
) -> tuple[np.ndarray, np.ndarray]:
    """For every row, return ``(soft_cluster_id, similarity)``.

    Non-noise rows get their original cluster + sim to its centroid (typically high).
    Noise rows get the nearest cluster — but only if cosine similarity ≥ ``min_sim``;
    otherwise they stay as ``NOISE_LABEL`` and similarity is the best score we saw.
    The dashboard can filter on the similarity threshold.
    """
    unique = sorted(int(c) for c in np.unique(cluster_labels) if c != NOISE_LABEL)
    if not unique:
        return cluster_labels.copy(), np.zeros(len(vectors), dtype=np.float32)

    centroids = np.stack(
        [vectors[cluster_labels == c].mean(axis=0) for c in unique]
    )
    centroids = l2_normalize(centroids)
    # vectors are already L2-normalized by the embed step.
    sims = vectors @ centroids.T  # (N, K) cosine
    best_idx = sims.argmax(axis=1)
    best_sim = sims[np.arange(len(vectors)), best_idx].astype(np.float32)

    soft = cluster_labels.copy()
    noise_mask = cluster_labels == NOISE_LABEL
    promote = noise_mask & (best_sim >= min_sim)
    soft[promote] = np.array(unique, dtype=cluster_labels.dtype)[best_idx[promote]]
    return soft, best_sim


# ---------- Parent themes via centroid agglomerative clustering ----------


def parent_themes(
    vectors: np.ndarray,
    cluster_labels: np.ndarray,
    *,
    n_parents: int = N_PARENT_THEMES,
) -> dict[int, int]:
    """Group HDBSCAN clusters into ``n_parents`` parent themes.

    Operates on cosine-normalized cluster centroids in the original embedding
    space (not the UMAP space) — that's the most faithful summary of cluster
    semantics. Noise (-1) is its own degenerate parent theme.
    """
    unique = sorted(int(c) for c in np.unique(cluster_labels) if c != NOISE_LABEL)
    if not unique:
        return {NOISE_LABEL: NOISE_LABEL}

    centroids = np.stack(
        [vectors[cluster_labels == c].mean(axis=0) for c in unique]
    )
    centroids = l2_normalize(centroids)

    n_parents_eff = max(1, min(n_parents, len(unique)))
    if n_parents_eff == 1:
        cluster_to_parent = {c: 0 for c in unique}
    else:
        agg = AgglomerativeClustering(
            n_clusters=n_parents_eff,
            metric="cosine",
            linkage="average",
        )
        parent_assignments = agg.fit_predict(centroids)
        cluster_to_parent = {c: int(p) for c, p in zip(unique, parent_assignments)}

    cluster_to_parent[NOISE_LABEL] = NOISE_LABEL
    return cluster_to_parent


# ---------- Pipeline orchestration ----------


@dataclass
class ClusterArtifact:
    df: pd.DataFrame  # row_id, cluster_id, soft_cluster_id, soft_sim, parent_id, ...
    n_clusters: int
    n_parents: int
    noise_fraction: float
    soft_noise_fraction: float
    n_duplicate_groups: int

    def summary(self) -> dict:
        return {
            "rows": len(self.df),
            "n_clusters": self.n_clusters,
            "n_parents": self.n_parents,
            "noise_fraction_strict": round(self.noise_fraction, 4),
            "noise_fraction_soft": round(self.soft_noise_fraction, 4),
            "n_duplicate_groups": self.n_duplicate_groups,
            "duplicate_collapse_pct": round(
                100 * (1 - self.n_duplicate_groups / len(self.df)), 2
            ),
        }


def cluster_pipeline(
    df: pd.DataFrame,
    art: EmbeddingArtifact,
    *,
    text_col: str = "text_clean",
    id_col: str = "row_id",
    n_parents: int = N_PARENT_THEMES,
    use_cache: bool = True,
) -> ClusterArtifact:
    """End-to-end clustering: dedup → UMAP(10) → HDBSCAN → parent themes → UMAP(2).

    Cache key: (corpus hash, parameter signature). Parameter signature is
    embedded in the parquet's pandas attrs so changing knobs invalidates.
    """
    sig = (
        f"h={art.text_hash},p={n_parents},"
        f"mc={HDBSCAN_MIN_CLUSTER_SIZE},ms={HDBSCAN_MIN_SAMPLES},"
        f"sm={HDBSCAN_SELECTION_METHOD},sa={SOFT_ASSIGN_MIN_SIM},"
        f"un={UMAP_N_NEIGHBORS},uc={UMAP_N_COMPONENTS},md={UMAP_MIN_DIST},"
        f"mh={MINHASH_THRESHOLD},ng={NGRAM_SIZE}"
    )

    if use_cache and CLUSTERS_PARQUET.exists():
        cached = pd.read_parquet(CLUSTERS_PARQUET)
        if cached.attrs.get("sig") == sig and len(cached) == len(df):
            print(f"[cluster] cache hit: {CLUSTERS_PARQUET.name}")
            return _from_cached(cached)
        print("[cluster] cache stale — recomputing")

    # Align order: ensure embeddings and df rows are in the same order.
    if not np.array_equal(art.ids, df[id_col].to_numpy(np.int64)):
        # Reindex df to match art.ids ordering.
        df = df.set_index(id_col).loc[art.ids].reset_index()

    print("[cluster] dedup via MinHash LSH ...")
    dup_groups = assign_duplicate_groups(df, text_col=text_col, id_col=id_col)
    n_groups = int(dup_groups.nunique())
    print(f"[cluster]   {len(df)} rows → {n_groups} dedup groups")

    print(f"[cluster] UMAP → {UMAP_N_COMPONENTS}-d ...")
    reduced = umap_reduce(art.vectors, n_components=UMAP_N_COMPONENTS)

    print("[cluster] HDBSCAN ...")
    labels = hdbscan_cluster(reduced)
    n_clusters = int((np.unique(labels) != NOISE_LABEL).sum())
    noise_frac = float((labels == NOISE_LABEL).mean())
    print(f"[cluster]   discovered {n_clusters} clusters, strict noise {noise_frac:.2%}")

    print("[cluster] soft-assigning noise points to nearest cluster ...")
    soft_labels, soft_sims = soft_assign_noise(art.vectors, labels)
    soft_noise_frac = float((soft_labels == NOISE_LABEL).mean())
    print(f"[cluster]   soft noise fraction {soft_noise_frac:.2%} (sim ≥ {SOFT_ASSIGN_MIN_SIM})")

    print("[cluster] UMAP → 2-d projection (for dashboard scatter) ...")
    reduced2 = umap_reduce(art.vectors, n_components=2)

    print("[cluster] parent themes via agglomerative on centroids ...")
    cluster_to_parent = parent_themes(art.vectors, labels, n_parents=n_parents)
    parent_ids = np.array([cluster_to_parent[int(c)] for c in labels], dtype=np.int32)
    soft_parent_ids = np.array(
        [cluster_to_parent.get(int(c), NOISE_LABEL) for c in soft_labels], dtype=np.int32
    )
    n_parents_eff = int(len(set(p for p in cluster_to_parent.values() if p != NOISE_LABEL)))

    out = pd.DataFrame(
        {
            id_col: art.ids,
            "cluster_id": labels,
            "soft_cluster_id": soft_labels,
            "soft_sim": soft_sims,
            "parent_id": parent_ids,
            "soft_parent_id": soft_parent_ids,
            "duplicate_group_id": dup_groups.values.astype(np.int64),
            "umap_x": reduced2[:, 0],
            "umap_y": reduced2[:, 1],
        }
    )
    out.attrs["sig"] = sig

    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out.to_parquet(CLUSTERS_PARQUET, index=False)
    print(f"[cluster] cached: {CLUSTERS_PARQUET.name}")

    return ClusterArtifact(
        df=out,
        n_clusters=n_clusters,
        n_parents=n_parents_eff,
        noise_fraction=noise_frac,
        soft_noise_fraction=soft_noise_frac,
        n_duplicate_groups=n_groups,
    )


def _from_cached(cached: pd.DataFrame) -> ClusterArtifact:
    n_clusters = int((cached["cluster_id"].unique() != NOISE_LABEL).sum())
    noise_frac = float((cached["cluster_id"] == NOISE_LABEL).mean())
    soft_noise_frac = float((cached["soft_cluster_id"] == NOISE_LABEL).mean())
    n_parents = int(
        cached.loc[cached["parent_id"] != NOISE_LABEL, "parent_id"].nunique()
    )
    return ClusterArtifact(
        df=cached,
        n_clusters=n_clusters,
        n_parents=n_parents,
        noise_fraction=noise_frac,
        soft_noise_fraction=soft_noise_frac,
        n_duplicate_groups=int(cached["duplicate_group_id"].nunique()),
    )


if __name__ == "__main__":
    from feedback.embed import embed_texts
    from feedback.ingest import load_feedback
    from feedback.normalize import normalize_dataframe

    df = normalize_dataframe(load_feedback())
    art = embed_texts(df)
    result = cluster_pipeline(df, art)
    for k, v in result.summary().items():
        print(f"  {k}: {v}")

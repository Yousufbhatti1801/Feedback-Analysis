"""Multilingual embedding wrapper for Islam360 feedback.

Model: ``intfloat/multilingual-e5-large`` (1024-d). The e5 family is trained
with a query/passage prefix convention; for retrieval-style and clustering
tasks we use the ``passage:`` prefix on every input. Outputs are L2-normalized
so cosine similarity collapses to a dot product downstream.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

from feedback.ingest import REPO_ROOT

CACHE_DIR = REPO_ROOT / "data" / "cache"
MODEL_NAME = "intfloat/multilingual-e5-large"
EMB_DIM = 1024


@dataclass
class EmbeddingArtifact:
    ids: np.ndarray  # int64, shape (N,)
    vectors: np.ndarray  # float32, shape (N, EMB_DIM)
    model: str
    text_hash: str  # hash of the cleaned-text corpus, for cache invalidation

    @property
    def n(self) -> int:
        return self.vectors.shape[0]


def _pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def _hash_corpus(texts: list[str]) -> str:
    h = hashlib.sha256()
    for t in texts:
        h.update(t.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()[:16]


def _cache_paths(tag: str) -> tuple[Path, Path]:
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    return CACHE_DIR / f"embeddings_{tag}.npy", CACHE_DIR / f"embeddings_{tag}.parquet"


def embed_texts(
    df: pd.DataFrame,
    *,
    text_col: str = "text_clean",
    id_col: str = "row_id",
    tag: str = "main",
    batch_size: int = 64,
    use_cache: bool = True,
) -> EmbeddingArtifact:
    """Embed every row in ``df`` and cache to disk.

    The cache key combines the corpus hash with ``tag``. If you change the
    model, normalization, or input text, the hash changes and the cache misses
    on its own — you don't need to delete files.
    """
    texts = df[text_col].tolist()
    ids = df[id_col].to_numpy(dtype=np.int64)
    text_hash = _hash_corpus(texts)
    npy_path, meta_path = _cache_paths(tag)

    if use_cache and npy_path.exists() and meta_path.exists():
        meta = pd.read_parquet(meta_path)
        if (
            meta.attrs.get("text_hash") == text_hash
            and meta.attrs.get("model") == MODEL_NAME
            and len(meta) == len(texts)
        ):
            vectors = np.load(npy_path)
            print(f"[embed] cache hit: {npy_path.name} ({vectors.shape})")
            return EmbeddingArtifact(
                ids=meta[id_col].to_numpy(np.int64),
                vectors=vectors,
                model=MODEL_NAME,
                text_hash=text_hash,
            )
        print("[embed] cache stale — recomputing")

    device = _pick_device()
    print(f"[embed] loading {MODEL_NAME} on {device}")
    model = SentenceTransformer(MODEL_NAME, device=device)

    prefixed = [f"passage: {t}" for t in texts]
    vectors = model.encode(
        prefixed,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
        convert_to_numpy=True,
    ).astype(np.float32)

    if vectors.shape != (len(texts), EMB_DIM):
        raise RuntimeError(
            f"Unexpected embedding shape {vectors.shape}, expected ({len(texts)}, {EMB_DIM})"
        )

    np.save(npy_path, vectors)
    meta = pd.DataFrame({id_col: ids, text_col: texts})
    meta.attrs["text_hash"] = text_hash
    meta.attrs["model"] = MODEL_NAME
    meta.to_parquet(meta_path, index=False)
    print(f"[embed] cached: {npy_path.name} + {meta_path.name}")

    return EmbeddingArtifact(
        ids=ids, vectors=vectors, model=MODEL_NAME, text_hash=text_hash
    )


def nearest_neighbors(
    artifact: EmbeddingArtifact,
    anchor_indices: list[int] | np.ndarray,
    k: int = 5,
) -> tuple[np.ndarray, np.ndarray]:
    """For each anchor row, return (k+1) nearest indices and similarities.

    Anchors are positional indices into ``artifact.vectors``, not row_ids.
    The first hit is always the anchor itself (sim = 1.0).
    """
    anchors = np.asarray(anchor_indices)
    sims = artifact.vectors[anchors] @ artifact.vectors.T  # (a, N) cosine since L2-normed
    top = np.argpartition(-sims, kth=k, axis=1)[:, : k + 1]
    # sort within the top-k+1
    rows = np.arange(top.shape[0])[:, None]
    order = np.argsort(-sims[rows, top], axis=1)
    top_sorted = top[rows, order]
    sims_sorted = sims[rows, top_sorted]
    return top_sorted, sims_sorted


if __name__ == "__main__":
    from feedback.ingest import load_feedback
    from feedback.normalize import normalize_dataframe

    df = normalize_dataframe(load_feedback())
    artifact = embed_texts(df)
    print(f"vectors: {artifact.vectors.shape}, dtype={artifact.vectors.dtype}")

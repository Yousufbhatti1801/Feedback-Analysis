"""Phase 2 acceptance check: pick 20 random anchors, print 5 nearest neighbors.

Run: .venv/bin/python scripts/spotcheck_neighbors.py
PM-reviewable — the goal is to confirm the embedding space groups
semantically similar feedback together, not to compute a metric.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from feedback.embed import embed_texts, nearest_neighbors
from feedback.ingest import load_feedback
from feedback.normalize import normalize_dataframe

SEED = 42
N_ANCHORS = 20
K = 5
MAX_LEN = 110


def truncate(text: str, n: int = MAX_LEN) -> str:
    text = " ".join(text.split())
    return text if len(text) <= n else text[: n - 1] + "…"


def main() -> None:
    df = normalize_dataframe(load_feedback())
    raw_by_id = df.set_index("row_id")["Feedback Comment"].to_dict()
    art = embed_texts(df)

    rng = np.random.default_rng(SEED)
    anchor_pos = rng.choice(art.n, size=N_ANCHORS, replace=False)
    nn_idx, nn_sim = nearest_neighbors(art, anchor_pos, k=K)

    rows = []
    for i, anchor in enumerate(anchor_pos):
        anchor_id = int(art.ids[anchor])
        print(f"\n=== Anchor {i + 1} (row_id={anchor_id}) ===")
        print(f"  ❯ {truncate(raw_by_id[anchor_id])}")
        for rank, (j, sim) in enumerate(zip(nn_idx[i, 1:], nn_sim[i, 1:]), start=1):
            n_id = int(art.ids[j])
            text = truncate(raw_by_id[n_id])
            print(f"  {rank}. ({sim:.3f})  {text}")
            rows.append(
                {
                    "anchor_pos": int(anchor),
                    "anchor_row_id": anchor_id,
                    "rank": rank,
                    "neighbor_row_id": n_id,
                    "similarity": float(sim),
                    "anchor_text": raw_by_id[anchor_id],
                    "neighbor_text": raw_by_id[n_id],
                }
            )

    out = pd.DataFrame(rows)
    avg_top1 = out[out["rank"] == 1]["similarity"].mean()
    avg_top5 = out["similarity"].mean()
    print(f"\nMean top-1 similarity: {avg_top1:.3f}")
    print(f"Mean top-5 similarity: {avg_top5:.3f}")

    out_path = Path(__file__).resolve().parents[1] / "data" / "cache" / "spotcheck_neighbors.parquet"
    out.to_parquet(out_path, index=False)
    print(f"\nSaved: {out_path.relative_to(out_path.parents[2])}")


if __name__ == "__main__":
    main()

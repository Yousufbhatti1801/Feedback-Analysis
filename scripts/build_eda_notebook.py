"""Generate notebooks/01_eda.ipynb from a flat list of cells.

Keeps the notebook source diff-friendly: edit this file, re-run, commit both.
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf

NB_PATH = Path(__file__).resolve().parents[1] / "notebooks" / "01_eda.ipynb"

CELLS: list[tuple[str, str]] = [
    (
        "md",
        """# Phase 1 — Exploratory Data Analysis

Goal: give the PM concrete evidence about the dataset before any clustering work.

We answer four questions:
1. What's the language mix?
2. How long is a typical feedback comment?
3. How fragmented is the existing `Granular_Subtopic` column?
4. Is the existing label hierarchy internally consistent?

All analysis logic lives in `src/feedback/eda.py` so the same primitives can be
reused by tests and the validation panel later.""",
    ),
    (
        "code",
        """%load_ext autoreload
%autoreload 2

import sys
from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

sns.set_theme(style="whitegrid")

REPO_ROOT = Path.cwd().parent if Path.cwd().name == "notebooks" else Path.cwd()
sys.path.insert(0, str(REPO_ROOT / "src"))

from feedback.ingest import load_feedback, summarize
from feedback import eda

df = load_feedback()
print(f"Loaded {len(df):,} rows, {df.shape[1]} columns")
df.head(3)""",
    ),
    (
        "md",
        "## 1. Existing label distributions\n\nQuick reference for the pre-populated columns.",
    ),
    (
        "code",
        """stats = summarize(df)
for k, v in stats.items():
    print(f"{k}: {v}")""",
    ),
    (
        "code",
        """fig, axes = plt.subplots(1, 3, figsize=(16, 4))
for ax, col, title in zip(
    axes,
    ["Topic", "Severity", "Sentiment"],
    ["Topic", "Severity", "Sentiment"],
):
    df[col].value_counts().plot(kind="bar", ax=ax, color="steelblue")
    ax.set_title(title)
    ax.set_xlabel("")
    ax.tick_params(axis="x", rotation=30)
plt.tight_layout()
plt.show()""",
    ),
    (
        "md",
        "## 2. Feedback length distribution\n\nDrives later decisions: token budget for embeddings, whether to chunk multi-paragraph comments.",
    ),
    (
        "code",
        """ls = eda.length_stats(df)
ls.describe().round(1)""",
    ),
    (
        "code",
        """fig, axes = plt.subplots(1, 2, figsize=(14, 4))
axes[0].hist(np.clip(ls.chars, 0, 1000), bins=60, color="steelblue")
axes[0].set_title("Characters per comment (clipped at 1000)")
axes[0].set_xlabel("chars"); axes[0].set_ylabel("rows")
axes[1].hist(np.clip(ls.words, 0, 200), bins=60, color="seagreen")
axes[1].set_title("Words per comment (clipped at 200)")
axes[1].set_xlabel("words"); axes[1].set_ylabel("rows")
plt.tight_layout()
plt.show()

short = (ls.chars < 5).sum()
very_long = (ls.chars > 500).sum()
print(f"Very short (<5 chars): {short}    Very long (>500 chars): {very_long}")""",
    ),
    (
        "md",
        """## 3. Language mix

Coarse 4-bucket classifier (Urdu script / Roman Urdu / English / mixed) using
Unicode script ranges and a small Roman-Urdu marker list. **Metadata only** —
no transliteration, no language-branched preprocessing. The downstream
multilingual embedding handles the real semantic work; this view exists so the
PM can see the language mix and we can spot-check translation quality later.""",
    ),
    (
        "code",
        """langs = eda.language_distribution(df)
lang_counts = langs.value_counts()
print(lang_counts)
print(f"\\n% non-English: {(langs != 'english').mean() * 100:.1f}%")

fig, ax = plt.subplots(figsize=(10, 4))
lang_counts.plot(kind="bar", ax=ax, color="indianred")
ax.set_title("Detected language buckets")
ax.set_ylabel("rows")
ax.tick_params(axis="x", rotation=0)
plt.tight_layout()
plt.show()""",
    ),
    (
        "md",
        """## 4. `Granular_Subtopic` over-fragmentation

The headline finding for the PM. If most granular labels appear on 1–3 rows, they
are not a useful taxonomy — they are essentially row-level annotations.""",
    ),
    (
        "code",
        """frag = eda.fragmentation_stats(df)
for k, v in frag.items():
    print(f"{k}: {v}")""",
    ),
    (
        "code",
        """granular_counts = df["Granular_Subtopic"].value_counts()
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(np.clip(granular_counts, 0, 20), bins=20, color="darkorange")
ax.set_title("Rows per Granular_Subtopic label (clipped at 20)")
ax.set_xlabel("rows assigned to this label")
ax.set_ylabel("number of labels")
plt.tight_layout()
plt.show()""",
    ),
    (
        "md",
        """### Hierarchy consistency

If `Granular_Subtopic` is a true child of `Contextual_Subtopic`, every granular
label should have exactly one parent. Multiple parents per granular label means
the labels were generated without a consistent hierarchy.""",
    ),
    (
        "code",
        """eda.cross_label_consistency(df)""",
    ),
    (
        "md",
        """## Takeaways for the PM

Findings will be filled in by the executed cells above. Expected story:

- **Language mix** is dominated by Urdu / Roman Urdu / English. This justifies a multilingual embedding (`multilingual-e5-large`) over English-only models.
- **Lengths are short-to-medium**, with a long tail. No chunking needed; raw comment fits in one embedding pass.
- **`Granular_Subtopic` is over-fragmented** — a large fraction of labels apply to ≤3 rows. This confirms the brief: re-derive the taxonomy via clustering rather than treating these as ground truth.
- **Hierarchy is inconsistent** — granular labels with multiple Contextual_Subtopic parents indicate the existing labels were generated row-by-row, not from a controlled taxonomy.

Phase 2 next: normalize text, generate embeddings, cache to Parquet.""",
    ),
]


def main() -> None:
    nb = nbf.v4.new_notebook()
    for kind, source in CELLS:
        if kind == "md":
            nb.cells.append(nbf.v4.new_markdown_cell(source))
        else:
            nb.cells.append(nbf.v4.new_code_cell(source))
    nb.metadata["kernelspec"] = {
        "display_name": "Python 3",
        "language": "python",
        "name": "python3",
    }
    nb.metadata["language_info"] = {"name": "python"}
    NB_PATH.parent.mkdir(parents=True, exist_ok=True)
    nbf.write(nb, NB_PATH)
    print(f"Wrote {NB_PATH.relative_to(NB_PATH.parents[1])}")


if __name__ == "__main__":
    main()

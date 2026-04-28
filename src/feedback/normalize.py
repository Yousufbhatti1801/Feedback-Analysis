"""Light text normalization for embedding input.

Design rules from the project brief:
- Preserve the original raw text — display always uses raw, never the cleaned version.
- No transliteration. The downstream multilingual embedding handles script.
- Strip noise that destroys embedding quality (URLs, repeated whitespace) but
  keep emojis and punctuation, which carry sentiment signal.
"""

from __future__ import annotations

import re
import unicodedata

import pandas as pd

from feedback.eda import detect_language

_URL_RE = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_WHITESPACE_RE = re.compile(r"\s+")
_LEADING_PUNCT_RE = re.compile(r"^[\s.\-_•·]+")


def clean_text(raw: str) -> str:
    """Apply the minimum normalization needed before embedding.

    Steps: NFKC -> URL strip -> whitespace collapse -> strip leading dots/dashes
    (the export has many comments starting with '. ' or '- ', which is noise).
    Returns empty string for blank/whitespace-only input.
    """
    if not raw:
        return ""
    text = unicodedata.normalize("NFKC", raw)
    text = _URL_RE.sub(" ", text)
    text = _WHITESPACE_RE.sub(" ", text)
    text = _LEADING_PUNCT_RE.sub("", text).strip()
    return text


def normalize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Add ``text_clean`` and ``language`` columns and drop blank rows.

    Returns a new DataFrame, indexed by ``row_id``, sorted by row_id, ready
    to feed into the embedding step. Original ``Feedback Comment`` is preserved.
    """
    out = df.copy()
    out["text_clean"] = out["Feedback Comment"].fillna("").map(clean_text)
    out["language"] = out["Feedback Comment"].fillna("").map(detect_language)

    blank_mask = out["text_clean"].str.len() == 0
    if blank_mask.any():
        out = out.loc[~blank_mask].copy()

    return out.sort_values("row_id").reset_index(drop=True)

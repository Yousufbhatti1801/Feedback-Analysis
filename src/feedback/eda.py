"""Phase 1 exploratory analysis primitives.

Functions are kept small and importable so the EDA notebook stays thin and the
same logic can be reused by tests and the validation panel later.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np
import pandas as pd

# Unicode ranges where Urdu/Arabic script lives.
_ARABIC_SCRIPT_RE = re.compile(
    r"[؀-ۿݐ-ݿࢠ-ࣿﭐ-﷿ﹰ-﻿]"
)
_LATIN_LETTER_RE = re.compile(r"[A-Za-z]")

# Common Roman-Urdu markers. Hits any of these → likely Roman Urdu, not English.
# Conservative list: words that are essentially never English noise words.
_ROMAN_URDU_MARKERS = re.compile(
    r"\b("
    r"hai|hain|nahi|nahin|nai|kya|kyun|kyu|kaise|kaisa|kaisi|"
    r"acha|achha|theek|thik|bohat|bahut|zyada|kam|sahi|"
    r"namaz|namaaz|roza|hadees|hadith|tafseer|tafsir|tarjuma|"
    r"masla|masail|dua|duaa|allah|rabb|nabi|"
    r"please|plz|"  # neutral; ignored without other markers
    r"app|apk|"  # neutral
    r"krna|karna|krty|krte|hota|hoti|raha|rahi|gya|gyi|"
    r"mein|main|mai|tum|aap|hum|"
    r"jazak|jazakallah|mashallah|mashaallah|alhamdulillah|inshallah|"
    r"asalam|salam|wassalam"
    r")\b",
    re.IGNORECASE,
)
_ROMAN_URDU_STRONG = re.compile(  # subset that, alone, confirms Roman Urdu
    r"\b("
    r"hai|hain|nahi|nahin|kaise|kaisa|bohat|bahut|"
    r"namaz|namaaz|hadees|tafseer|tarjuma|masla|dua|"
    r"krna|karna|hota|hoti|mein|jazakallah|mashallah|alhamdulillah|inshallah"
    r")\b",
    re.IGNORECASE,
)


@dataclass
class LengthStats:
    chars: pd.Series
    words: pd.Series
    lines: pd.Series

    def describe(self) -> pd.DataFrame:
        return pd.DataFrame(
            {"chars": self.chars, "words": self.words, "lines": self.lines}
        ).describe(percentiles=[0.5, 0.9, 0.95, 0.99])


def length_stats(df: pd.DataFrame, text_col: str = "Feedback Comment") -> LengthStats:
    text = df[text_col].fillna("")
    return LengthStats(
        chars=text.str.len(),
        words=text.str.split().str.len().fillna(0).astype(int),
        lines=text.str.count("\n") + (text.str.len() > 0).astype(int),
    )


def detect_language(text: str) -> str:
    """Coarse 4-bucket classification: blank / urdu_script / roman_urdu / english / mixed.

    Heuristic on purpose — this is metadata only. The downstream multilingual
    embedding does the real semantic work; we just want a PM-readable view of
    the language mix and a way to spot-check translation quality later.
    """
    if not text or not text.strip():
        return "blank"
    has_arabic = _ARABIC_SCRIPT_RE.search(text) is not None
    has_latin = _LATIN_LETTER_RE.search(text) is not None
    if has_arabic and has_latin:
        return "mixed"
    if has_arabic:
        return "urdu_script"
    if has_latin:
        # Strong Roman-Urdu marker → Roman Urdu. Two weak markers → Roman Urdu.
        if _ROMAN_URDU_STRONG.search(text):
            return "roman_urdu"
        if len(_ROMAN_URDU_MARKERS.findall(text)) >= 2:
            return "roman_urdu"
        return "english"
    return "other"  # symbols / digits only


def language_distribution(
    df: pd.DataFrame, text_col: str = "Feedback Comment"
) -> pd.Series:
    """Tag every row with a coarse language bucket. Vectorized; runs instantly."""
    return df[text_col].fillna("").map(detect_language)


def fragmentation_stats(df: pd.DataFrame) -> dict:
    """Quantify the over-fragmentation of Granular_Subtopic.

    PM-visible takeaway: how many granular labels exist for how few rows each.
    """
    granular_counts = df["Granular_Subtopic"].value_counts()
    contextual_counts = df["Contextual_Subtopic"].value_counts()
    return {
        "n_granular_labels": int(granular_counts.size),
        "n_contextual_labels": int(contextual_counts.size),
        "rows_per_granular_mean": float(granular_counts.mean()),
        "rows_per_granular_median": float(granular_counts.median()),
        "granular_singletons": int((granular_counts == 1).sum()),
        "granular_singletons_pct": float((granular_counts == 1).mean() * 100),
        "granular_le_3_rows": int((granular_counts <= 3).sum()),
        "granular_le_3_rows_pct": float((granular_counts <= 3).mean() * 100),
        "rows_per_contextual_mean": float(contextual_counts.mean()),
        "rows_per_contextual_median": float(contextual_counts.median()),
    }


def cross_label_consistency(df: pd.DataFrame) -> pd.DataFrame:
    """For each Granular_Subtopic, how many distinct Contextual_Subtopic parents?

    A healthy hierarchy would have exactly one. Multiple parents per granular
    label is a strong sign the labels were generated row-by-row without a
    consistent taxonomy.
    """
    parents_per_granular = (
        df.groupby("Granular_Subtopic")["Contextual_Subtopic"].nunique().value_counts()
    )
    return parents_per_granular.rename_axis("distinct_parents").to_frame("granular_count")

"""Interactive Islam360 feedback intelligence dashboard.

Run:
    streamlit run app/streamlit_app.py
"""

from __future__ import annotations

import json
import re
import time
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "classified_all_feedbacks.json"

THEME = {
    "bg": "#090f1d",
    "panel": "#101a31",
    "panel_soft": "#121f3b",
    "muted": "#9bb0d6",
    "text": "#edf3ff",
    "accent": "#5a8cff",
    "accent2": "#18c7a4",
    "warn": "#ffb020",
    "danger": "#ff5f6d",
    "violet": "#9a7dff",
}

st.set_page_config(
    page_title="Islam360 Feedback Intelligence",
    layout="wide",
    initial_sidebar_state="expanded",
)

CSS = f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
html, body, [class*="css"] {{ font-family: 'Inter', sans-serif; }}
[data-testid="stAppViewContainer"] {{
    background:
      radial-gradient(1400px 700px at 12% -20%, rgba(90,140,255,0.32), transparent 58%),
      radial-gradient(1000px 500px at 88% -10%, rgba(24,199,164,0.20), transparent 58%),
      {THEME['bg']};
}}
header[data-testid="stHeader"] {{ background: transparent !important; }}
.main .block-container {{ max-width: 1550px; padding-top: 1rem; }}
h1, h2, h3 {{ color: {THEME['text']} !important; letter-spacing: -0.02em; }}
p, li, label, [data-testid="stMarkdownContainer"] {{ color: {THEME['muted']}; }}
[data-testid="metric-container"] {{
    background: linear-gradient(180deg, {THEME['panel']} 0%, {THEME['panel_soft']} 100%);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px;
    padding: 10px;
}}
[data-testid="stMetricValue"] {{ color: {THEME['text']}; font-weight: 800; }}
[data-testid="stDataFrame"] {{
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 12px;
    overflow: hidden;
}}
.help-card {{
    background: linear-gradient(180deg, {THEME['panel']} 0%, {THEME['panel_soft']} 100%);
    border: 1px solid rgba(255,255,255,0.12);
    border-radius: 14px;
    padding: 10px 12px;
}}
</style>
"""

FILTER_FIELDS = [
    "classification",
    "topic",
    "parent_issue",
    "child_issue",
    "sentiment",
    "severity",
    "raw_topic",
]

WORKSPACE_LIMIT = 12  # max issues a PM can pin to the Triage Workspace at once

PM_PRESETS = {
    "All Feedback": {},
    "Trust Escalations": {
        "sentiment": ["Negative"],
        "risk_primary": ["Trust & Authenticity"],
    },
    "Subscription Pain": {
        "risk_primary": ["Monetization Risk"],
        "sentiment": ["Negative", "Neutral"],
    },
    "Quran/Tafseer Critical": {
        "risk_primary": ["Quran / Tafseer"],
        "severity": ["High", "Medium"],
        "sentiment": ["Negative", "Neutral"],
    },
    "Ads Sensitivity": {
        "risk_primary": ["Ads Sensitivity"],
        "sentiment": ["Negative", "Neutral"],
    },
    "Performance Critical": {
        "risk_primary": ["Performance Risk"],
        "severity": ["High", "Medium"],
    },
}

SENSITIVE_KEYWORDS = {
    "Ads Sensitivity": ["ad", "ads", "advert", "advertisement", "advertisements", "vulgar"],
    "Monetization Risk": ["subscription", "subscribed", "payment", "premium", "refund", "billing", "charge", "charged", "paid"],
    "Trust & Authenticity": ["wrong", "incorrect", "authentic", "fake", "sahih", "zaeef", "daeef", "dalil", "proof", "reference"],
    "Quran / Tafseer": ["quran", "qur'an", "quraan", "tafseer", "tafsir", "translation", "tarjuma", "ayah", "ayat", "surah", "sura"],
    "Prayer & Worship": ["prayer", "namaz", "namaaz", "azan", "azaan", "adhan", "dua", "duaa", "salah"],
    "Performance Risk": ["crash", "crashes", "crashing", "hang", "hangs", "hanging", "slow", "freeze", "freezes", "lag", "laggy", "bug", "bugs"],
    "Search & UX": ["search", "ui", "ux", "navigation", "navigate", "find"],
}

# Pre-compile word-boundary patterns once. Substring matching ("ad" in "add") was
# producing massive false positives — virtually every "Add a feature..." comment
# was getting tagged as Ads-related. Word boundaries fix that.
_SENSITIVE_PATTERNS = {
    tag: re.compile(r"\b(?:" + "|".join(re.escape(t) for t in terms) + r")\b", re.IGNORECASE)
    for tag, terms in SENSITIVE_KEYWORDS.items()
}


def clip_text(value: str, limit: int = 170) -> str:
    text = " ".join(str(value or "").split())
    return text if len(text) <= limit else text[: limit - 1] + "…"


def pretty_field(name: str) -> str:
    return name.replace("_", " ").title()


def style_fig(fig, title: str, height: int = 360) -> None:
    fig.update_layout(
        title=title,
        height=height,
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color=THEME["text"], family="Inter"),
        margin=dict(l=8, r=8, t=48, b=8),
        hoverlabel=dict(
            bgcolor="rgba(10,16,29,0.97)",
            bordercolor="rgba(255,255,255,0.2)",
            font=dict(color=THEME["text"]),
        ),
    )


def parse_selection(event: object) -> dict:
    if isinstance(event, dict):
        if "selection" in event and isinstance(event["selection"], dict):
            return event["selection"]
        return event
    return {}


def selected_custom_data(event: object, idx: int = 0) -> str | None:
    selection = parse_selection(event)
    points = selection.get("points", [])
    if not points:
        return None
    custom_data = points[0].get("customdata")
    if isinstance(custom_data, (list, tuple)) and len(custom_data) > idx:
        return str(custom_data[idx])
    return None


def click_aware_chart(fig, base_key: str, *, height_key: str | None = None) -> object:
    """Render a Plotly chart whose on_select fires exactly once per real click.

    Streamlit's ``plotly_chart(on_select="rerun")`` returns the same selection
    event on every rerun until the user makes a new selection. That makes any
    side-effecting handler (e.g., open an issue-detail modal) re-trigger on
    every rerun and lock the user into a popup loop. We dodge that by giving
    the chart a key whose suffix bumps each time we consume a click — the
    next render produces a fresh widget with no stale selection.
    """
    suffix = st.session_state.get(f"_chart_key_suffix_{base_key}", 0)
    return st.plotly_chart(
        fig,
        theme=None,
        use_container_width=True,
        key=f"{base_key}__{suffix}",
        on_select="rerun",
    )


def consume_chart_click(base_key: str) -> None:
    """Bump the chart's widget-key suffix so the consumed click can't re-fire."""
    suffix_key = f"_chart_key_suffix_{base_key}"
    st.session_state[suffix_key] = st.session_state.get(suffix_key, 0) + 1


def record_sensitivity_tags(row: pd.Series) -> list[str]:
    corpus = " ".join(
        [
            str(row.get("topic", "")),
            str(row.get("parent_issue", "")),
            str(row.get("child_issue", "")),
            str(row.get("original_feedback", "")),
        ]
    )
    tags = [tag for tag, pattern in _SENSITIVE_PATTERNS.items() if pattern.search(corpus)]
    return tags or ["General Product"]


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    with DATA_PATH.open("r", encoding="utf-8") as f:
        payload = json.load(f)
    df = pd.DataFrame(payload)
    df["parent_issue"] = df["parent_issue"].fillna("Unassigned / No Parent")
    df["original_feedback"] = df["original_feedback"].fillna("").astype(str)
    df["record_id"] = range(1, len(df) + 1)
    df["is_issue"] = df["classification"].str.lower().isin(["complaint", "issue", "bug", "problem"])
    df["risk_tags"] = df.apply(record_sensitivity_tags, axis=1)
    df["risk_primary"] = df["risk_tags"].map(lambda tags: tags[0])
    df["priority_score"] = (
        df["severity"].map({"Low": 1, "Medium": 2, "High": 3}).fillna(1)
        + df["sentiment"].map({"Positive": 0, "Neutral": 1, "Negative": 2}).fillna(1)
        + df["risk_tags"].map(lambda tags: 2 if "Trust & Authenticity" in tags else 1)
    )
    return df


def ensure_state() -> None:
    if "drill_filters" not in st.session_state:
        st.session_state["drill_filters"] = {}
    if "selected_record_id" not in st.session_state:
        st.session_state["selected_record_id"] = None
    if "active_preset" not in st.session_state:
        st.session_state["active_preset"] = "All Feedback"
    if "active_page" not in st.session_state:
        st.session_state["active_page"] = "Executive Overview"
    if "pinned_issues" not in st.session_state:
        # Each item: {"level": str, "value": str, "ts": int}.
        st.session_state["pinned_issues"] = []


# ---------- Triage Workspace pin/unpin ----------


def pin_issue(level: str, value: str) -> None:
    pins = st.session_state.get("pinned_issues", [])
    if any(p["level"] == level and p["value"] == value for p in pins):
        return
    if len(pins) >= WORKSPACE_LIMIT:
        st.warning(
            f"Workspace is full ({WORKSPACE_LIMIT} pins max). Remove an item first."
        )
        return
    pins.append({"level": level, "value": value, "ts": time.time_ns()})
    st.session_state["pinned_issues"] = pins


def unpin_issue(level: str, value: str) -> None:
    st.session_state["pinned_issues"] = [
        p for p in st.session_state.get("pinned_issues", [])
        if not (p["level"] == level and p["value"] == value)
    ]


def is_pinned(level: str, value: str) -> bool:
    return any(
        p["level"] == level and p["value"] == value
        for p in st.session_state.get("pinned_issues", [])
    )


def clear_workspace() -> None:
    st.session_state["pinned_issues"] = []


def set_drill_filter(name: str, value: str | list[str], *, append: bool = True) -> None:
    if isinstance(value, list):
        values = [str(v) for v in value if str(v).strip()]
    else:
        values = [str(value)] if str(value).strip() else []
    if not values:
        return

    if append and name in st.session_state["drill_filters"]:
        existing = st.session_state["drill_filters"][name]
        merged = list(dict.fromkeys([*existing, *values]))
        st.session_state["drill_filters"][name] = merged
    else:
        st.session_state["drill_filters"][name] = values


def clear_drill_filters() -> None:
    st.session_state["drill_filters"] = {}


# ---------- Issue detail modal ----------
# Click any chart element → open a focused panel showing rank, totals, parent,
# 3 representative samples, and the full numbered list of matching comments.
# A click counter ensures clicking the same element twice still re-opens the modal.

LEVEL_LABEL = {
    "topic": "Topic",
    "parent_issue": "Parent Issue",
    "child_issue": "Feature Request",
    "raw_topic": "Raw Topic",
    "risk_primary": "Risk Domain",
}


def open_issue_detail(level: str, value: str) -> None:
    """Queue an issue-detail modal for (level, value) and rerun."""
    if not value or value in {"None", "nan"}:
        return
    counter = st.session_state.get("_detail_counter", 0) + 1
    st.session_state["_detail_counter"] = counter
    st.session_state["pending_detail"] = {"level": level, "value": value, "n": counter}
    st.rerun()


def _rank_among(df: pd.DataFrame, level: str, value: str) -> tuple[int, int]:
    """Rank ``value`` among all values of ``level`` (1 = most frequent).

    For taxonomy levels, ranking is computed on Suggestion rows only — General
    Reviews would otherwise dominate every level (e.g., "General Other -
    Positive Feedback" tops child_issue counts but isn't an actionable
    request). For risk_primary the rank is over the exploded risk_tags column
    so a row carrying multiple tags counts in each.
    """
    if level == "risk_primary":
        counts = df["risk_tags"].explode().value_counts()
    else:
        scope = df[df["classification"] == "Suggestion"] if "classification" in df.columns else df
        if value in scope[level].astype(str).values:
            counts = scope[level].astype(str).value_counts()
        else:
            # Value isn't an actionable suggestion (e.g., a General Review label
            # clicked from the All-Feedbacks table) — fall back to ranking
            # within the full corpus so we still produce a sensible number.
            counts = df[level].astype(str).value_counts()
    if value not in counts.index:
        return 0, len(counts)
    return list(counts.index).index(value) + 1, len(counts)


def pick_representative_samples(scoped: pd.DataFrame, k: int = 3) -> list[str]:
    """Pick ``k`` diverse-looking sample comments from the scoped DataFrame.

    Prefers mid-length unique comments so the PM sees real evidence rather
    than terse one-word feedback. Falls back to any unique non-empty text.
    """
    texts = scoped["original_feedback"].fillna("").map(lambda s: " ".join(str(s).split()))
    mid = texts.loc[texts.str.len().between(20, 240)].drop_duplicates()
    if len(mid) >= k:
        return mid.sample(n=k, random_state=42).tolist()
    fallback = texts.loc[texts.str.len() > 0].drop_duplicates().head(k).tolist()
    return fallback if fallback else texts.head(k).tolist()


@st.dialog("Issue Detail", width="large")
def _issue_detail_dialog() -> None:
    pending = st.session_state.get("pending_detail")
    if not pending:
        return
    level = pending["level"]
    value = pending["value"]

    df_full = st.session_state.get("_detail_full_df")
    if df_full is None:
        st.warning("No data available.")
        return

    if level == "risk_primary":
        scoped = df_full[df_full["risk_tags"].map(lambda tags: value in tags)]
    else:
        scoped = df_full[df_full[level].astype(str) == value]

    if scoped.empty:
        st.warning(f"No comments found for {LEVEL_LABEL.get(level, level)} = {value}")
        return

    rank, total = _rank_among(df_full, level, value)
    parent = (
        scoped["parent_issue"].mode().iloc[0]
        if "parent_issue" in scoped.columns and not scoped["parent_issue"].mode().empty
        else "—"
    )
    raw_topic_value = (
        scoped["raw_topic"].mode().iloc[0]
        if "raw_topic" in scoped.columns and not scoped["raw_topic"].mode().empty
        else "—"
    )
    high_pct = (scoped["severity"] == "High").mean() * 100
    neg_pct = (scoped["sentiment"] == "Negative").mean() * 100

    text_color = THEME["text"]
    muted_color = THEME["muted"]
    accent_color = THEME["accent"]
    accent2_color = THEME["accent2"]

    # Header
    st.markdown(
        f"<div style='font-size:1.55rem;font-weight:800;color:{text_color};line-height:1.1;'>{value}</div>"
        f"<div style='color:{muted_color};margin-top:4px;font-size:0.92rem;'>"
        f"Rank #{rank} · {len(scoped):,} requests · {parent}</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<div style='height:14px;'></div>", unsafe_allow_html=True)

    # Stat tiles — content adapts to which level was clicked so we never repeat
    # the same field twice (e.g., for a parent_issue click, the value IS the
    # parent, so we show structural counts instead of a duplicate parent tile).
    c1, c2 = st.columns(2)
    c1.metric("RANK", f"#{rank} of {total}")
    c2.metric("TOTAL REQUESTS", f"{len(scoped):,} users")

    c3, c4 = st.columns(2)
    if level == "parent_issue":
        c3.metric("RAW TOPIC", clip_text(raw_topic_value, 32))
        c4.metric("CHILD ISSUES IN GROUP", f"{scoped['child_issue'].nunique():,}")
    elif level == "raw_topic":
        c3.metric("CHILD ISSUES IN TOPIC", f"{scoped['child_issue'].nunique():,}")
        c4.metric("PARENT GROUPS IN TOPIC", f"{scoped['parent_issue'].nunique():,}")
    elif level == "topic":
        c3.metric("PARENT ISSUE", clip_text(parent, 32))
        c4.metric("RAW TOPIC", clip_text(raw_topic_value, 32))
    elif level == "risk_primary":
        c3.metric("PARENT GROUPS HIT", f"{scoped['parent_issue'].nunique():,}")
        c4.metric("CHILD ISSUES HIT", f"{scoped['child_issue'].nunique():,}")
    else:  # child_issue (default)
        c3.metric("PARENT ISSUE", clip_text(parent, 32))
        c4.metric("RAW TOPIC", clip_text(raw_topic_value, 32))

    c6, c7 = st.columns(2)
    c6.metric("HIGH SEVERITY %", f"{high_pct:.0f}%")
    c7.metric("NEGATIVE %", f"{neg_pct:.0f}%")

    # Action row: pin to workspace + filter-dashboard CTA
    btn_pin, btn_filter = st.columns(2)
    pinned = is_pinned(level, value)
    pin_label = "📌 Pinned · Click To Unpin" if pinned else "📌 Pin To Triage Workspace"
    if btn_pin.button(pin_label, use_container_width=True, key="modal_pin_btn"):
        if pinned:
            unpin_issue(level, value)
        else:
            pin_issue(level, value)
        st.rerun()
    if btn_filter.button(
        "Filter Entire Dashboard By This Issue",
        use_container_width=True, type="primary", key="modal_filter_btn",
    ):
        target_field = "risk_primary" if level == "risk_primary" else level
        st.session_state["drill_filters"][target_field] = [value]
        st.session_state["pending_detail"] = None
        st.session_state["active_page"] = "Feedback Evidence Table"
        st.rerun()

    # Sample feedbacks (3) — green-bordered cards, matching the reference design
    st.markdown(
        f"<div style='margin-top:18px;color:{muted_color};font-size:0.78rem;"
        f"letter-spacing:0.1em;font-weight:600;'>SAMPLE USER FEEDBACKS (3)</div>",
        unsafe_allow_html=True,
    )
    samples = pick_representative_samples(scoped, k=3)
    sample_html = []
    for s in samples:
        sample_html.append(
            f"<div style='border-left:3px solid {accent2_color};"
            f"background:rgba(24,199,164,0.06);padding:11px 14px;border-radius:6px;"
            f"margin:6px 0;color:{text_color};font-size:0.92rem;line-height:1.4;'>"
            f"{clip_text(s, 320)}</div>"
        )
    st.markdown("\n".join(sample_html), unsafe_allow_html=True)

    # Full numbered list with chip on the right
    st.markdown(
        f"<div style='margin-top:18px;color:{muted_color};font-size:0.78rem;"
        f"letter-spacing:0.1em;font-weight:600;'>ALL MATCHING FEEDBACKS ({len(scoped):,})</div>",
        unsafe_allow_html=True,
    )
    show_n = min(len(scoped), 200)
    rows_html = []
    for idx, (_, row) in enumerate(scoped.head(show_n).iterrows(), 1):
        comment = clip_text(row.get("original_feedback", ""), 220)
        chip = clip_text(str(row.get("child_issue") or row.get("topic") or ""), 36)
        rows_html.append(
            "<div style='display:flex;gap:14px;align-items:center;"
            "padding:8px 12px;border:1px solid rgba(255,255,255,0.07);"
            "border-radius:8px;margin-bottom:5px;'>"
            f"<div style='color:{muted_color};min-width:26px;font-size:0.8rem;'>{idx}</div>"
            f"<div style='flex:1;color:{text_color};font-size:0.88rem;line-height:1.35;'>{comment}</div>"
            f"<div style='font-size:0.72rem;color:{accent_color};"
            "background:rgba(90,140,255,0.14);border-radius:14px;"
            f"padding:3px 10px;white-space:nowrap;'>{chip}</div>"
            "</div>"
        )
    st.markdown("<div>" + "".join(rows_html) + "</div>", unsafe_allow_html=True)

    if len(scoped) > show_n:
        st.caption(
            f"Showing first {show_n:,} of {len(scoped):,} comments. "
            "Use 'Filter Entire Dashboard By This Issue' for the full set + CSV export."
        )


def drill_and_navigate(filters: dict[str, list[str]], *, target_page: str = "Feedback Evidence Table") -> None:
    for field, values in filters.items():
        set_drill_filter(field, values, append=False)
    st.session_state["active_page"] = target_page
    st.rerun()


def apply_preset(name: str) -> None:
    preset = PM_PRESETS.get(name, {})
    clear_drill_filters()
    for field, values in preset.items():
        set_drill_filter(field, values, append=False)
    st.session_state["active_preset"] = name


def build_filtered_view(df: pd.DataFrame) -> pd.DataFrame:
    """Apply sidebar filters + drill filters and return the narrowed DataFrame.

    UI-state side effects:
      - Renders the entire Filters sidebar.
      - Stores per-field "narrowing" status in ``st.session_state['_active_sidebar_narrowing']``
        so the main-area chip bar can show which sidebar fields are actively filtering.
    """
    out = df.copy()
    narrowing: dict[str, list[str]] = {}

    st.sidebar.markdown("## Filters")

    # ---- Quick jump: fuzzy-find any topic / parent / child issue by name ----
    qj = st.sidebar.text_input(
        "🔍 Quick jump to issue",
        placeholder="search issue name…",
        help="Find any topic / parent issue / child issue by name and open its detail panel.",
        key="quick_jump_input",
    )
    if qj.strip():
        q = qj.strip().lower()
        matches: list[tuple[str, str]] = []
        for level in ("child_issue", "parent_issue", "topic"):
            for v in df[level].dropna().astype(str).unique():
                if q in v.lower():
                    matches.append((level, v))
            if len(matches) >= 50:
                break
        if matches:
            options = [f"[{LEVEL_LABEL.get(l, l)}] {v}" for l, v in matches[:30]]
            picked_label = st.sidebar.selectbox("Matches", options, key="quick_jump_pick")
            if st.sidebar.button("Open Issue Detail", use_container_width=True, key="quick_jump_go"):
                idx = options.index(picked_label)
                lvl, val = matches[idx]
                open_issue_detail(lvl, val)
        else:
            st.sidebar.caption("No matches.")

    st.sidebar.markdown("---")

    # ---- Saved PM presets ----
    preset_name = st.sidebar.selectbox(
        "Saved PM view",
        list(PM_PRESETS.keys()),
        index=list(PM_PRESETS.keys()).index(st.session_state["active_preset"]),
        key="preset_select",
    )
    bcol1, bcol2 = st.sidebar.columns(2)
    if bcol1.button("Apply", use_container_width=True, key="preset_apply"):
        apply_preset(preset_name)
        st.rerun()
    if bcol2.button("Reset", use_container_width=True, key="preset_reset"):
        apply_preset("All Feedback")
        st.rerun()

    st.sidebar.markdown("---")

    # ---- Quick toggles ----
    search = st.sidebar.text_input("Keyword search", placeholder="ads during recitation", key="kw_search")
    only_negative = st.sidebar.checkbox("Only negative sentiment", key="only_neg")
    only_high = st.sidebar.checkbox("Only high severity", key="only_high")
    only_parented = st.sidebar.checkbox("Only rows with parent issue assigned", key="only_parented")
    only_sensitive = st.sidebar.checkbox("Only trust-sensitive", key="only_trust")

    st.sidebar.markdown("---")
    st.sidebar.markdown("**Field filters** (deselect items to narrow)")

    for field in FILTER_FIELDS:
        all_options = sorted(df[field].dropna().astype(str).unique().tolist())
        # Use the FULL universe of options as the default & list — not the post-filter
        # subset — so deselecting one filter doesn't permanently hide options from
        # the others (which used to silently swallow choices on every rerun).
        selected = st.sidebar.multiselect(
            pretty_field(field),
            all_options,
            default=all_options,
            key=f"ms_{field}",
        )
        if selected and len(selected) < len(all_options):
            narrowing[field] = selected
        if selected:
            out = out[out[field].astype(str).isin(selected)]
        else:
            out = out.iloc[0:0]

    if search.strip():
        q = search.lower().strip()
        out = out[
            out["original_feedback"].str.lower().str.contains(q, na=False)
            | out["child_issue"].str.lower().str.contains(q, na=False)
            | out["parent_issue"].str.lower().str.contains(q, na=False)
            | out["topic"].str.lower().str.contains(q, na=False)
        ]
        narrowing["search"] = [search.strip()]
    if only_negative:
        out = out[out["sentiment"] == "Negative"]
        narrowing["only_negative"] = ["true"]
    if only_high:
        out = out[out["severity"] == "High"]
        narrowing["only_high"] = ["true"]
    if only_parented:
        out = out[out["parent_issue"] != "Unassigned / No Parent"]
        narrowing["only_parented"] = ["true"]
    if only_sensitive:
        out = out[out["risk_tags"].map(lambda tags: "Trust & Authenticity" in tags)]
        narrowing["only_sensitive"] = ["true"]

    for key, values in st.session_state["drill_filters"].items():
        if key == "risk_primary":
            out = out[out["risk_tags"].map(lambda tags: any(v in tags for v in values))]
        else:
            out = out[out[key].astype(str).isin(values)]

    st.session_state["_active_sidebar_narrowing"] = narrowing
    return out


def render_active_filters_bar(df_total: int, df_filtered: int) -> None:
    """Show prominent active-filter chips at the top of the main area.

    Combines drill filters (chart clicks) and sidebar narrowing into one bar so
    the PM can always see exactly what's scoping their view, with one-click
    removal for each.
    """
    drill = st.session_state.get("drill_filters", {})
    sidebar_narrow = st.session_state.get("_active_sidebar_narrowing", {})

    has_any = bool(drill) or bool(sidebar_narrow)
    pct = (df_filtered / df_total * 100) if df_total else 0.0
    head_l, head_r = st.columns([5, 1])
    head_l.caption(
        f"**{df_filtered:,}** of {df_total:,} comments in view ({pct:.1f}%) · "
        f"Saved view: **{st.session_state.get('active_preset', 'All Feedback')}**"
    )
    if has_any and head_r.button("Clear All Filters", use_container_width=True, key="clear_all_top"):
        clear_drill_filters()
        st.session_state["active_preset"] = "All Feedback"
        # Reset sidebar widgets that have explicit keys.
        for f in FILTER_FIELDS:
            st.session_state.pop(f"ms_{f}", None)
        for k in ("kw_search", "only_neg", "only_high", "only_parented", "only_trust"):
            st.session_state.pop(k, None)
        st.rerun()

    if not has_any:
        return

    chips: list[tuple[str, str, str]] = []  # (label, key, group)
    for field, values in drill.items():
        chips.append((f"🎯 {pretty_field(field)}: {', '.join(values[:2])}{'…' if len(values) > 2 else ''}", field, "drill"))
    for field, values in sidebar_narrow.items():
        if field in {"only_negative", "only_high", "only_parented", "only_sensitive"}:
            chips.append((f"☑ {field.replace('_', ' ')}", field, "sidebar"))
        elif field == "search":
            chips.append((f"🔎 \"{values[0][:30]}\"", field, "sidebar"))
        else:
            chips.append((f"⚙ {pretty_field(field)}: {len(values)} of {df_total} selected", field, "sidebar"))

    cols = st.columns(min(len(chips), 4) or 1)
    for idx, (label, key, group) in enumerate(chips):
        col = cols[idx % len(cols)]
        if col.button(f"× {label}", key=f"chip_{group}_{key}", use_container_width=True):
            if group == "drill":
                st.session_state["drill_filters"].pop(key, None)
            else:
                # Sidebar widget removal: reset the widget key so its default reapplies.
                if key.startswith("only_") or key == "search":
                    mapping = {
                        "only_negative": "only_neg",
                        "only_high": "only_high",
                        "only_parented": "only_parented",
                        "only_sensitive": "only_trust",
                        "search": "kw_search",
                    }
                    st.session_state.pop(mapping.get(key, key), None)
                else:
                    st.session_state.pop(f"ms_{key}", None)
            st.rerun()


def render_kpis(df: pd.DataFrame) -> None:
    trust_count = int(df["risk_tags"].map(lambda tags: "Trust & Authenticity" in tags).sum())
    monetization_count = int(df["risk_tags"].map(lambda tags: "Monetization Risk" in tags).sum())
    high_count = int((df["severity"] == "High").sum())
    negative_count = int((df["sentiment"] == "Negative").sum())
    top_parent = df["parent_issue"].value_counts().head(1)
    top_child = df["child_issue"].value_counts().head(1)

    c1, c2, c3, c4, c5 = st.columns(5)
    c1.metric("Total Feedback", f"{len(df):,}")
    c2.metric("Negative Sentiment", f"{negative_count:,}")
    c3.metric("High Severity", f"{high_count:,}")
    c4.metric("Trust-Sensitive", f"{trust_count:,}")
    c5.metric("Monetization Risk", f"{monetization_count:,}")

    c6, c7, c8 = st.columns(3)
    c6.metric("Unique Parent Issues", f"{df['parent_issue'].nunique():,}")
    c7.metric("Top Parent Issue", top_parent.index[0] if not top_parent.empty else "-")
    c8.metric("Top Child Issue", clip_text(top_child.index[0], 30) if not top_child.empty else "-")

    b1, b2, b3, b4, b5 = st.columns(5)
    if b1.button("Filter: Negative", use_container_width=True):
        set_drill_filter("sentiment", "Negative", append=False)
        st.rerun()
    if b2.button("Filter: High Severity", use_container_width=True):
        set_drill_filter("severity", "High", append=False)
        st.rerun()
    if b3.button("Filter: Trust-Sensitive", use_container_width=True):
        set_drill_filter("risk_primary", "Trust & Authenticity", append=False)
        st.rerun()
    if b4.button("Filter: Monetization", use_container_width=True):
        set_drill_filter("risk_primary", "Monetization Risk", append=False)
        st.rerun()
    if b5.button("Reset Drill Filters", use_container_width=True):
        clear_drill_filters()
        st.rerun()


def render_pm_guide() -> None:
    with st.expander("PM Quick Guide", expanded=False):
        st.markdown(
            """
            <div class="help-card">
            <b>How to use this dashboard effectively</b><br><br>
            1) Start with a <b>Saved PM view</b> from sidebar (e.g., Trust Escalations).<br>
            2) Click bars/bubbles in charts to drill into issue clusters.<br>
            3) Use <b>Remove &lt;filter&gt;</b> chips to relax scope gradually.<br>
            4) Confirm prioritization in <b>Escalation Queue</b> and <b>Impact vs Urgency</b> matrix.<br>
            5) Validate with real evidence in <b>Feedback Evidence Table</b> before decisions.<br>
            </div>
            """,
            unsafe_allow_html=True,
        )


def overview_tab(df: pd.DataFrame) -> None:
    # Top-of-page treemap: instant volume scan, colored by negative-sentiment rate.
    # The eye finds the biggest red blocks → that's where the PM should look first.
    treemap_df = (
        df.groupby(["raw_topic", "topic"], as_index=False)
        .agg(
            count=("record_id", "count"),
            negative_rate=("sentiment", lambda s: float((s == "Negative").mean())),
            high_severity_rate=("severity", lambda s: float((s == "High").mean())),
        )
        .sort_values("count", ascending=False)
    )
    treemap_df["label"] = treemap_df.apply(
        lambda r: f"{clip_text(r['topic'], 38)} ({r['count']})", axis=1
    )
    fig = px.treemap(
        treemap_df,
        path=["raw_topic", "label"],
        values="count",
        color="negative_rate",
        color_continuous_scale=[THEME["accent2"], THEME["warn"], THEME["danger"]],
        range_color=[0, max(0.5, treemap_df["negative_rate"].max())],
        custom_data=["topic", "raw_topic", "count", "negative_rate", "high_severity_rate"],
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Raw topic: %{customdata[1]}<br>"
            "Comments: %{customdata[2]}<br>"
            "Negative: %{customdata[3]:.1%}<br>"
            "High severity: %{customdata[4]:.1%}<extra></extra>"
        ),
        marker=dict(line=dict(color="rgba(255,255,255,0.08)", width=1)),
    )
    fig.update_layout(coloraxis_colorbar=dict(title="Neg %", thickness=10), clickmode="event+select")
    style_fig(fig, "Topic Volume — Block Size = Comments, Color = Negative Sentiment Rate · Click For Detail", 380)
    evt = click_aware_chart(fig, "overview_treemap")
    selection = parse_selection(evt)
    points = selection.get("points", [])
    if points:
        p = points[0]
        consume_chart_click("overview_treemap")
        # Outer ring (topic) click has a non-empty parent (the raw_topic).
        # Inner ring (raw_topic) click has parent == "".
        if p.get("parent"):
            cd = p.get("customdata") or []
            if cd:
                open_issue_detail("topic", str(cd[0]))
        else:
            open_issue_detail("raw_topic", str(p.get("label") or ""))

    c1, c2 = st.columns(2)
    with c1:
        topic_counts = df["topic"].value_counts().head(12).sort_values()
        top = topic_counts.reset_index()
        top.columns = ["topic", "count"]
        fig = px.bar(
            top,
            x="count",
            y="topic",
            orientation="h",
            color="count",
            color_continuous_scale=[THEME["accent2"], THEME["accent"], THEME["violet"]],
            custom_data=["topic"],
        )
        fig.update_layout(coloraxis_showscale=False, clickmode="event+select")
        style_fig(fig, "Topic Distribution · Click For Detail")
        evt = click_aware_chart(fig, "topic_chart")
        selection = parse_selection(evt)
        points = selection.get("points", [])
        if points:
            cd = points[0].get("customdata") or []
            if cd:
                consume_chart_click("topic_chart")
                open_issue_detail("topic", str(cd[0]))

    with c2:
        parent_counts = df["parent_issue"].value_counts().head(15).sort_values()
        par = parent_counts.reset_index()
        par.columns = ["parent_issue", "count"]
        fig = px.bar(
            par,
            x="count",
            y="parent_issue",
            orientation="h",
            color="count",
            color_continuous_scale=[THEME["accent"], THEME["violet"], THEME["danger"]],
            custom_data=["parent_issue"],
        )
        fig.update_layout(coloraxis_showscale=False, clickmode="event+select")
        style_fig(fig, "Parent Issue Frequency · Click For Detail")
        evt = click_aware_chart(fig, "parent_chart")
        selection = parse_selection(evt)
        points = selection.get("points", [])
        if points:
            cd = points[0].get("customdata") or []
            if cd:
                consume_chart_click("parent_chart")
                open_issue_detail("parent_issue", str(cd[0]))

    c3, c4 = st.columns(2)
    with c3:
        sev = df["severity"].value_counts().reset_index()
        sev.columns = ["severity", "count"]
        fig = px.pie(
            sev,
            names="severity",
            values="count",
            color="severity",
            color_discrete_map={"High": THEME["danger"], "Medium": THEME["warn"], "Low": THEME["accent2"]},
        )
        style_fig(fig, "Severity Split")
        st.plotly_chart(fig, theme=None, use_container_width=True)
    with c4:
        sent = df["sentiment"].value_counts().reset_index()
        sent.columns = ["sentiment", "count"]
        fig = px.pie(
            sent,
            names="sentiment",
            values="count",
            color="sentiment",
            color_discrete_map={"Negative": THEME["danger"], "Neutral": "#94a3b8", "Positive": "#22c55e"},
        )
        style_fig(fig, "Sentiment Split")
        st.plotly_chart(fig, theme=None, use_container_width=True)


def hierarchy_tab(df: pd.DataFrame) -> None:
    # Sunburst — the canonical visual for parent → child taxonomy.
    # Inner ring = parent issue, outer ring = child issue, sized by count.
    # Click a wedge to drill into that branch.
    sun_df = (
        df.groupby(["parent_issue", "child_issue"], as_index=False)
        .agg(
            count=("record_id", "count"),
            negative_rate=("sentiment", lambda s: float((s == "Negative").mean())),
        )
    )
    fig = px.sunburst(
        sun_df,
        path=["parent_issue", "child_issue"],
        values="count",
        color="negative_rate",
        color_continuous_scale=[THEME["accent2"], THEME["warn"], THEME["danger"]],
        range_color=[0, 0.6],
        custom_data=["parent_issue", "child_issue", "count", "negative_rate"],
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[1]}</b><br>"
            "Parent: %{customdata[0]}<br>"
            "Comments: %{customdata[2]}<br>"
            "Negative: %{customdata[3]:.1%}<extra></extra>"
        ),
        insidetextorientation="radial",
        marker=dict(line=dict(color="rgba(255,255,255,0.06)", width=1)),
    )
    fig.update_layout(coloraxis_colorbar=dict(title="Neg %", thickness=10), clickmode="event+select")
    style_fig(fig, "Issue Taxonomy — Inner = Parent, Outer = Child · Click For Detail", 520)
    evt = click_aware_chart(fig, "hierarchy_sunburst")
    selection = parse_selection(evt)
    points = selection.get("points", [])
    if points:
        p = points[0]
        label = str(p.get("label") or "")
        parent_label = str(p.get("parent") or "")
        consume_chart_click("hierarchy_sunburst")
        # Outer ring (child) click has a parent label; inner ring (parent) click does not.
        if parent_label:
            open_issue_detail("child_issue", label)
        else:
            open_issue_detail("parent_issue", label)

    left, right = st.columns([1.8, 1.2])
    with left:
        child_counts = (
            df.groupby(["parent_issue", "child_issue"], as_index=False)
            .size()
            .rename(columns={"size": "count"})
            .sort_values("count", ascending=False)
            .head(30)
        )
        child_counts["label"] = child_counts["child_issue"].map(lambda x: clip_text(x, 55))
        fig = px.bar(
            child_counts.sort_values("count"),
            x="count",
            y="label",
            orientation="h",
            color="parent_issue",
            custom_data=["child_issue", "parent_issue"],
        )
        fig.update_layout(showlegend=False, clickmode="event+select")
        style_fig(fig, "Top Child Issues · Click For Detail", 520)
        evt = click_aware_chart(fig, "child_chart")
        selection = parse_selection(evt)
        points = selection.get("points", [])
        if points:
            cd = points[0].get("customdata") or []
            if cd:
                consume_chart_click("child_chart")
                open_issue_detail("child_issue", str(cd[0]))

        # Severity × sentiment heatmap per parent issue — at-a-glance pain map.
        # Cell color = comments; PM scans columns to find Negative-High concentration.
        sev_sent = (
            df.groupby(["parent_issue", "severity", "sentiment"], as_index=False)
            .size().rename(columns={"size": "count"})
        )
        sev_sent["cell"] = sev_sent["severity"] + " · " + sev_sent["sentiment"]
        pivot = sev_sent.pivot_table(
            index="parent_issue", columns="cell", values="count", fill_value=0,
        )
        # Order columns Negative-first, High-first for left-to-right pain reading.
        col_order = [
            f"{sev} · {sent}"
            for sev in ("High", "Medium", "Low")
            for sent in ("Negative", "Neutral", "Positive")
        ]
        pivot = pivot.reindex(columns=[c for c in col_order if c in pivot.columns])
        pivot = pivot.reindex(pivot.sum(axis=1).sort_values(ascending=False).index)
        fig = px.imshow(
            pivot.values,
            x=pivot.columns.tolist(),
            y=pivot.index.tolist(),
            color_continuous_scale=[THEME["panel"], THEME["accent"], THEME["danger"]],
            aspect="auto",
            text_auto=True,
        )
        fig.update_traces(hovertemplate="%{y}<br>%{x}<br>Count: %{z}<extra></extra>")
        style_fig(fig, "Severity × Sentiment Per Parent Issue (Pain Map)", 360)
        st.plotly_chart(fig, theme=None, use_container_width=True, key="sev_sent_heatmap")

    with right:
        st.markdown("### Drill-Down Detail")
        active = st.session_state["drill_filters"]
        if not active:
            st.info("Click a topic, parent issue, or child issue chart element to open detail.")
            return
        detail_df = df.copy()
        for key, values in active.items():
            detail_df = detail_df[detail_df[key].isin(values)]
        st.write(f"**Selected:** {', '.join([f'{k}={', '.join(v)}' for k, v in active.items()])}")
        st.metric("Matching Feedback Count", f"{len(detail_df):,}")
        st.metric("Unique Child Issues", f"{detail_df['child_issue'].nunique():,}")
        st.metric("Negative %", f"{(detail_df['sentiment'] == 'Negative').mean() * 100:.1f}%")
        st.metric("High Severity %", f"{(detail_df['severity'] == 'High').mean() * 100:.1f}%")
        st.markdown("**Representative Feedback Examples**")
        for text in detail_df["original_feedback"].head(7):
            st.markdown(f"- {clip_text(text, 190)}")


def risk_tab(df: pd.DataFrame) -> None:
    exploded = df[["record_id", "risk_tags", "severity", "sentiment", "parent_issue", "child_issue", "original_feedback"]].explode("risk_tags")
    risk_counts = exploded["risk_tags"].value_counts().reset_index()
    risk_counts.columns = ["risk_domain", "count"]
    fig = px.bar(
        risk_counts.sort_values("count"),
        x="count",
        y="risk_domain",
        orientation="h",
        color="count",
        color_continuous_scale=[THEME["accent"], THEME["violet"], THEME["danger"]],
        custom_data=["risk_domain"],
    )
    fig.update_layout(coloraxis_showscale=False, clickmode="event+select")
    style_fig(fig, "Islamic Trust/Sensitivity Spotlight · Click For Detail")
    evt = click_aware_chart(fig, "risk_chart")
    selection = parse_selection(evt)
    points = selection.get("points", [])
    if points:
        cd = points[0].get("customdata") or []
        if cd:
            consume_chart_click("risk_chart")
            open_issue_detail("risk_primary", str(cd[0]))

    priority = exploded.copy()
    priority["risk_weight"] = priority["risk_tags"].map(
        {
            "Trust & Authenticity": 5,
            "Quran / Tafseer": 4,
            "Prayer & Worship": 4,
            "Ads Sensitivity": 4,
            "Monetization Risk": 3,
            "Performance Risk": 3,
            "Search & UX": 2,
            "General Product": 1,
        }
    ).fillna(1)
    priority["severity_weight"] = priority["severity"].map({"Low": 1, "Medium": 2, "High": 3}).fillna(1)
    priority["sentiment_weight"] = priority["sentiment"].map({"Positive": 1, "Neutral": 2, "Negative": 3}).fillna(2)
    priority["priority_score"] = priority["risk_weight"] * 2 + priority["severity_weight"] + priority["sentiment_weight"]
    queue = (
        priority.groupby(["risk_tags", "parent_issue", "child_issue"], as_index=False)
        .agg(
            total_count=("record_id", "count"),
            high_severity=("severity", lambda s: int((s == "High").sum())),
            negative=("sentiment", lambda s: int((s == "Negative").sum())),
            priority_score=("priority_score", "mean"),
            sample_feedback=("original_feedback", "first"),
        )
        .sort_values(["priority_score", "total_count"], ascending=False)
        .head(30)
    )
    queue["sample_feedback"] = queue["sample_feedback"].map(lambda x: clip_text(x, 130))
    st.markdown("### Escalation Queue")
    st.dataframe(queue, use_container_width=True, hide_index=True)

    matrix = (
        df.groupby(["parent_issue", "child_issue"], as_index=False)
        .agg(
            frequency=("record_id", "count"),
            high_severity_rate=("severity", lambda s: float((s == "High").mean())),
            negative_rate=("sentiment", lambda s: float((s == "Negative").mean())),
            trust_rate=("risk_tags", lambda s: float(sum("Trust & Authenticity" in tags for tags in s) / max(len(s), 1))),
        )
        .sort_values("frequency", ascending=False)
        .head(120)
    )
    matrix["impact"] = matrix["frequency"]
    matrix["urgency"] = (matrix["high_severity_rate"] * 0.45) + (matrix["negative_rate"] * 0.4) + (matrix["trust_rate"] * 0.15)
    matrix["label_short"] = matrix["child_issue"].map(lambda x: clip_text(x, 42))

    fig = px.scatter(
        matrix,
        x="impact",
        y="urgency",
        size="frequency",
        color="urgency",
        color_continuous_scale=[THEME["accent2"], THEME["accent"], THEME["danger"]],
        hover_name="label_short",
        custom_data=["child_issue", "parent_issue", "frequency", "high_severity_rate", "negative_rate", "trust_rate"],
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{customdata[0]}</b><br>"
            "Parent: %{customdata[1]}<br>"
            "Frequency: %{customdata[2]}<br>"
            "High severity: %{customdata[3]:.1%}<br>"
            "Negative: %{customdata[4]:.1%}<br>"
            "Trust-sensitive: %{customdata[5]:.1%}<extra></extra>"
        )
    )
    fig.update_layout(coloraxis_showscale=False, clickmode="event+select")
    style_fig(fig, "Priority Matrix: Impact vs Urgency · Click Point For Detail", 440)
    evt = click_aware_chart(fig, "priority_matrix")
    selection = parse_selection(evt)
    points = selection.get("points", [])
    if points:
        cd = points[0].get("customdata") or []
        if cd:
            consume_chart_click("priority_matrix")
            open_issue_detail("child_issue", str(cd[0]))


def _scope_for_pin(df: pd.DataFrame, pin: dict) -> pd.DataFrame:
    level, value = pin["level"], pin["value"]
    if level == "risk_primary":
        return df[df["risk_tags"].map(lambda tags: value in tags)]
    return df[df[level].astype(str) == value]


def workspace_tab(df: pd.DataFrame) -> None:
    """Pinned-issue triage view — compare multiple issues side by side."""
    pins = st.session_state.get("pinned_issues", [])

    head_l, head_r = st.columns([4, 1])
    head_l.markdown(f"### Triage Workspace · {len(pins)} pinned")
    if pins and head_r.button("Clear Workspace", use_container_width=True):
        clear_workspace()
        st.rerun()

    if not pins:
        st.info(
            "**No issues pinned yet.** Click any chart element across the dashboard "
            "to open its detail panel, then press **📌 Pin To Triage Workspace** to "
            "shortlist it here. You can compare up to "
            f"**{WORKSPACE_LIMIT}** issues side-by-side and bulk-export their feedback."
        )
        return

    # ---- Comparison table: rows = metric, columns = pinned issues ----
    rows: list[dict] = []
    for pin in pins:
        scoped = _scope_for_pin(df, pin)
        if scoped.empty:
            continue
        rank, total = _rank_among(df, pin["level"], pin["value"])
        rows.append({
            "Issue": clip_text(pin["value"], 36),
            "Level": LEVEL_LABEL.get(pin["level"], pin["level"]),
            "Rank": f"#{rank}/{total}",
            "Total": f"{len(scoped):,}",
            "High Sev %": f"{(scoped['severity']=='High').mean()*100:.0f}%",
            "Negative %": f"{(scoped['sentiment']=='Negative').mean()*100:.0f}%",
            "Top Parent": clip_text(
                str(scoped["parent_issue"].mode().iloc[0])
                if not scoped["parent_issue"].mode().empty else "—", 28
            ),
        })
    if rows:
        st.markdown("#### Side-By-Side Comparison")
        st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)

    # ---- Comparative bar chart: pinned issues × negative / high-severity rates ----
    chart_rows = []
    for pin in pins:
        scoped = _scope_for_pin(df, pin)
        if scoped.empty:
            continue
        chart_rows.append({
            "issue": clip_text(pin["value"], 28),
            "metric": "High severity %",
            "pct": (scoped["severity"] == "High").mean() * 100,
        })
        chart_rows.append({
            "issue": clip_text(pin["value"], 28),
            "metric": "Negative %",
            "pct": (scoped["sentiment"] == "Negative").mean() * 100,
        })
    if chart_rows:
        chart_df = pd.DataFrame(chart_rows)
        fig = px.bar(
            chart_df,
            x="pct", y="issue", color="metric", orientation="h",
            barmode="group",
            color_discrete_map={
                "High severity %": THEME["danger"],
                "Negative %": THEME["warn"],
            },
        )
        fig.update_layout(legend=dict(title=None, orientation="h", y=1.08))
        style_fig(fig, "Pain Comparison Across Pinned Issues", 60 + 50 * len(pins))
        st.plotly_chart(fig, theme=None, use_container_width=True, key="workspace_compare")

    # ---- Detail cards for each pinned issue, two per row ----
    st.markdown("#### Pinned Issues")
    text_color = THEME["text"]
    muted_color = THEME["muted"]
    accent2_color = THEME["accent2"]

    for i in range(0, len(pins), 2):
        row_pins = pins[i : i + 2]
        cols = st.columns(2)
        for col, pin in zip(cols, row_pins):
            scoped = _scope_for_pin(df, pin)
            with col:
                rank, total = _rank_among(df, pin["level"], pin["value"])
                parent = (
                    str(scoped["parent_issue"].mode().iloc[0])
                    if not scoped.empty and not scoped["parent_issue"].mode().empty
                    else "—"
                )
                samples = pick_representative_samples(scoped, k=2) if not scoped.empty else []
                sample_html = "".join(
                    f"<div style='border-left:3px solid {accent2_color};"
                    f"background:rgba(24,199,164,0.06);padding:8px 11px;border-radius:6px;"
                    f"margin:4px 0;color:{text_color};font-size:0.85rem;line-height:1.35;'>"
                    f"{clip_text(s, 220)}</div>"
                    for s in samples
                )
                st.markdown(
                    f"<div class='help-card'>"
                    f"<div style='font-size:0.72rem;color:{muted_color};letter-spacing:0.08em;'>"
                    f"{LEVEL_LABEL.get(pin['level'], pin['level']).upper()}</div>"
                    f"<div style='font-size:1.1rem;font-weight:700;color:{text_color};line-height:1.25;margin:2px 0 6px 0;'>"
                    f"{clip_text(pin['value'], 60)}</div>"
                    f"<div style='color:{muted_color};font-size:0.82rem;margin-bottom:6px;'>"
                    f"Rank #{rank}/{total} · {len(scoped):,} requests · {clip_text(parent, 30)}</div>"
                    f"{sample_html}"
                    f"</div>",
                    unsafe_allow_html=True,
                )
                bcol1, bcol2, bcol3 = st.columns(3)
                if bcol1.button("Open", key=f"ws_open_{i}_{pin['value'][:30]}", use_container_width=True):
                    open_issue_detail(pin["level"], pin["value"])
                if bcol2.button("Filter", key=f"ws_filter_{i}_{pin['value'][:30]}", use_container_width=True):
                    st.session_state["drill_filters"][pin["level"]] = [pin["value"]]
                    st.session_state["active_page"] = "Feedback Evidence Table"
                    st.rerun()
                if bcol3.button("Remove", key=f"ws_remove_{i}_{pin['value'][:30]}", use_container_width=True):
                    unpin_issue(pin["level"], pin["value"])
                    st.rerun()

    # ---- Bulk CSV export of every comment matching every pinned issue ----
    st.markdown("---")
    pieces: list[pd.DataFrame] = []
    for pin in pins:
        scoped = _scope_for_pin(df, pin)
        if scoped.empty:
            continue
        chunk = scoped[
            ["record_id", "classification", "topic", "parent_issue", "child_issue",
             "severity", "sentiment", "raw_topic", "original_feedback"]
        ].copy()
        chunk.insert(0, "pinned_level", pin["level"])
        chunk.insert(1, "pinned_value", pin["value"])
        pieces.append(chunk)
    if pieces:
        combined = pd.concat(pieces, ignore_index=True)
        st.download_button(
            f"Export All Pinned Issues' Feedback ({len(combined):,} rows) As CSV",
            data=combined.to_csv(index=False).encode("utf-8"),
            file_name="islam360_triage_workspace.csv",
            mime="text/csv",
            use_container_width=True,
        )


def evidence_tab(df: pd.DataFrame) -> None:
    st.markdown("### Feedback Records")
    cols = [
        "record_id",
        "classification",
        "topic",
        "parent_issue",
        "child_issue",
        "severity",
        "sentiment",
        "raw_topic",
        "risk_primary",
        "original_feedback",
    ]
    table = df[cols].copy()
    table["original_feedback"] = table["original_feedback"].map(lambda x: clip_text(x, 150))
    st.dataframe(table, use_container_width=True, hide_index=True)

    select_col1, select_col2 = st.columns([1.4, 3])
    with select_col1:
        selected_id = st.selectbox(
            "Open record detail",
            options=[None] + df["record_id"].head(500).tolist(),
            format_func=lambda x: "Choose a record id" if x is None else f"Record {x}",
        )
        if selected_id is not None:
            st.session_state["selected_record_id"] = int(selected_id)
    with select_col2:
        rid = st.session_state.get("selected_record_id")
        if rid is not None:
            selected = df[df["record_id"] == rid].head(1)
            if not selected.empty:
                row = selected.iloc[0]
                st.markdown("#### Record Detail")
                st.markdown(
                    f"**Classification:** `{row['classification']}`  |  "
                    f"**Topic:** `{row['topic']}`  |  "
                    f"**Severity:** `{row['severity']}`  |  "
                    f"**Sentiment:** `{row['sentiment']}`"
                )
                st.markdown(f"**Parent issue:** `{row['parent_issue']}`")
                st.markdown(f"**Child issue:** `{row['child_issue']}`")
                st.markdown(f"**Raw topic:** `{row['raw_topic']}`")
                st.markdown(f"**Sensitivity tags:** `{', '.join(row['risk_tags'])}`")
                st.markdown("**Original feedback:**")
                st.info(row["original_feedback"])

    st.download_button(
        "Export filtered records as CSV",
        data=df[cols].to_csv(index=False).encode("utf-8"),
        file_name="islam360_feedback_filtered.csv",
        mime="text/csv",
    )


def main() -> None:
    st.markdown(CSS, unsafe_allow_html=True)
    ensure_state()
    df = load_data()

    # Stash the unfiltered df so the issue-detail dialog always sees ALL comments
    # for the clicked issue, regardless of the sidebar filter scope.
    st.session_state["_detail_full_df"] = df

    # If a chart click queued an issue-detail and we haven't rendered it yet,
    # open the modal. The "n" counter in pending_detail ensures repeated clicks
    # on the same element still re-open the dialog after a previous dismissal.
    pending = st.session_state.get("pending_detail")
    last_rendered = st.session_state.get("_detail_last_rendered")
    if pending and pending != last_rendered:
        st.session_state["_detail_last_rendered"] = pending
        _issue_detail_dialog()

    filtered = build_filtered_view(df)

    st.title("Islam360 Feedback Intelligence Dashboard")
    st.caption("Interactive PM view for issue prioritization, sensitivity risk, and feedback evidence.")
    render_active_filters_bar(len(df), len(filtered))
    render_pm_guide()

    pin_count = len(st.session_state.get("pinned_issues", []))
    workspace_label = (
        f"Triage Workspace ({pin_count})" if pin_count else "Triage Workspace"
    )

    pages = [
        "Executive Overview",
        "Issue Hierarchy Explorer",
        "Islamic Sensitivity & Priority",
        workspace_label,
        "Feedback Evidence Table",
    ]
    # Coerce stale active-page state across renames of the workspace label.
    stored_page = st.session_state.get("active_page", "Executive Overview")
    if stored_page.startswith("Triage Workspace") and stored_page not in pages:
        stored_page = workspace_label
    current_page = st.radio(
        "Dashboard View",
        pages,
        index=pages.index(stored_page) if stored_page in pages else 0,
        horizontal=True,
        key="page_radio",
    )
    st.session_state["active_page"] = current_page

    if filtered.empty and not current_page.startswith("Triage Workspace"):
        st.warning("No records for current filters. Broaden filters or use 'Clear All Filters' above.")
        return

    if current_page == "Executive Overview":
        render_kpis(filtered)
        overview_tab(filtered)
    elif current_page == "Issue Hierarchy Explorer":
        render_kpis(filtered)
        hierarchy_tab(filtered)
    elif current_page == "Islamic Sensitivity & Priority":
        render_kpis(filtered)
        risk_tab(filtered)
    elif current_page.startswith("Triage Workspace"):
        # Workspace operates on the FULL dataset so pinned issues remain
        # visible regardless of the current sidebar filter.
        workspace_tab(df)
    else:
        render_kpis(filtered)
        evidence_tab(filtered)


if __name__ == "__main__":
    main()

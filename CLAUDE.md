# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Authoritative project brief

Product goals, non-negotiable design rules, and the recommended development phases live in [.claude/CLAUDE.md](.claude/CLAUDE.md). Read that first — it is the source of truth for *what* to build and *why*. This file only covers operational facts about the current repo state.

## Current repo state

**Pre-implementation.** There is no source code, no build tooling, no tests, and no package manifest in this repo yet. The only artifact is the raw dataset. Any `make`/`npm`/`pytest` commands do not exist — do not invent them. When starting implementation, set up tooling as part of Phase 2 (Preprocessing) per the brief, and update this file with the real commands once they exist.

## The dataset

Single file at repo root: [2026-04-24T10-29_export.csv](2026-04-24T10-29_export.csv) — 11,258 rows, UTF-8 with BOM.

Columns:
- `` (unnamed index column — row number from the export)
- `Topic` — coarse bucket, 8 values
- `Feedback Comment` — raw user feedback (Urdu / Roman Urdu / English / mixed, often multi-line)
- `Contextual_Subtopic` — 41 unique values
- `Granular_Subtopic` — 6,021 unique values (very long tail; many near-duplicates)
- `Severity` — Low / Medium / High
- `Sentiment` — Positive / Neutral / Negative

Topic distribution: Feature 5320, Other 2904, Bug 1120, Ads 685, Accuracy 527, UI 514, Slow 148, Subscription 40.
Severity: Low 7762, Medium 3029, High 467. Sentiment: Neutral 5660, Negative 3077, Positive 2521.

## Important nuance the brief does not cover

**The CSV is already labeled.** Topic, Contextual_Subtopic, Granular_Subtopic, Severity, and Sentiment are pre-populated (source/method unknown — ask the user before assuming). The brief mandates *dynamic* issue discovery and explicitly forbids relying on hardcoded taxonomies. Reconcile these two facts before designing the pipeline:

- Treat existing labels as a **baseline to evaluate against**, not as ground truth to copy.
- The 6,021 `Granular_Subtopic` values are almost certainly over-fragmented and are a strong candidate for re-clustering / consolidation via embeddings.
- If asked to "use the existing categories," confirm whether the user means "start from these and refine" vs "ignore and rediscover" — the answer changes the architecture.

## Working conventions (from the project brief — reiterated here because they affect every task)

- Do not hardcode issue taxonomies as the primary mechanism — prefer embeddings / clustering / semantic grouping.
- Design for mixed-language (Urdu, Roman Urdu, English) feedback; do not assume clean English.
- Build incrementally by phase (see `.claude/CLAUDE.md` §"Recommended Development Phases"). Do not jump ahead.
- Preserve the original raw `Feedback Comment` text for display even after normalization.
- Keep exploratory analysis separate from the production pipeline.

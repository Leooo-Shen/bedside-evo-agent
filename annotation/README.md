# Annotation Tool

This folder contains a standalone browser-based annotation app for Oracle window outputs.

## Files

- `index.html`: UI shell.
- `styles.css`: layout and visual styles.
- `app.js`: loading, validation, round navigation, annotation capture, and JSON export.
- `output/`: target folder for exported annotated JSON files (choose this folder in your browser save dialog).

## Usage

1. Open `annotation/index.html` in a browser.
2. Choose the patient case folder and click **Load From Folder** (auto-detects `oracle_predictions.json` and `window_contexts.json`).
3. Annotate each valid window round.
4. Draft handling:
   - Auto-save runs locally in browser storage and auto-restores on next load for the same case.
   - Use **Export Draft** / **Import Draft** for manual backup or moving unfinished work across machines.
5. Export annotated JSON and save into `annotation/output/`.

## Features

- Multi-window rounds: one round per valid window.
- Invalid windows are skipped and reported.
- Recommendations are sorted by `rank` with `null` ranks at the end.
- Warning is raised for any unreviewed action/recommendation or unanswered omission question.
- Each window starts with a **Skip this window?** switch (default: No). If set to Yes, that page is excluded from required annotation checks.
- Context source priority:
  - `window_contexts.json` (if provided)
  - `oracle_predictions.json` raw event fields
- The "Previous k-hour Observation" section uses the configured Oracle context `history_hours` (with per-window prompt metadata fallback).
- "Previous k-hour Observation" event source priority:
  - `window_contexts.window_contexts[i].oracle_context_history_events` (preferred)
  - `window_contexts.window_contexts[i].history_events`
  - prompt section fallback parse (`previous_events_current_window`) when structured events are unavailable
- Event cards display raw Oracle evidence IDs (`event_id`, sourced from raw `event_idx`) when available.
- In Task 3, each added recommendation action requires selecting an ICU action category first, then writing free text (exported as `icu_action_category`).
- In Task 3, evidence event IDs for each added recommendation action are optional; if provided, they are exported as `evidence_event_ids`.
- Supports loading previously exported annotated JSON files and continuing annotation.
- Output annotation is stored per window under `window_outputs[i].annotation`.
- **Independent Event Filters**: Both `Current events` and `Previous k-hour Observation` include separate type filters (Vitals, Medications, Procedures, Labs, etc.), each preserving original chronological order. Filters reset when switching windows.

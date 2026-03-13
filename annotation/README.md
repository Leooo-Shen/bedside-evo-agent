# Annotation Tool

This folder contains a standalone browser-based annotation app for Oracle window outputs.

## Files

- `index.html`: UI shell.
- `styles.css`: layout and visual styles.
- `app.js`: loading, validation, round navigation, annotation capture, and JSON export.
- `output/`: target folder for exported annotated JSON files (choose this folder in your browser save dialog).

## Usage

1. Open `annotation/index.html` in a browser.
2. Load `oracle_predictions.json`.
3. Optionally load matching `window_contexts.json` (preferred source for discharge summary, ICU output context, and window events).
4. Annotate each valid window round.
5. Export annotated JSON and save into `annotation/output/`.

## Features

- Multi-window rounds: one round per valid window.
- Invalid windows are skipped and reported.
- Recommendations are sorted by `rank` with `null` ranks at the end.
- Warning is raised for any unreviewed action/recommendation or unanswered omission question.
- Context source priority:
  - `window_contexts.json` (if provided)
  - `oracle_predictions.json` raw event fields
- The "Previous k-hour Observation" section uses the configured Oracle context `history_hours` (with per-window prompt metadata fallback).
- Output annotation is stored per window under `window_outputs[i].annotation`.
- **Independent Event Filters**: Both `Current events` and `Previous k-hour Observation` include separate type filters (Vitals, Medications, Procedures, Labs, etc.), each preserving original chronological order. Filters reset when switching windows.

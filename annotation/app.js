"use strict";

const REQUIRED_ORACLE_OUTPUT_KEYS = [
  "patient_status",
  "action_evaluations",
  "recommendations",
  "overall_window_summary",
];
const DOMAIN_KEYS = ["hemodynamics", "respiratory", "renal_metabolic", "neurology"];
const PLOTTABLE_VITAL_LABEL_PATTERNS = [
  /\bheart rate\b/i,
  /\brespiratory rate\b/i,
  /\bo2 saturation\b/i,
  /\bspo2\b/i,
  /\bblood pressure\b/i,
  /\bart bp\b/i,
  /\bmean arterial pressure\b/i,
  /\bmap\b/i,
  /\btemperature\b/i,
  /\bfi[o0]2\b/i,
  /\binspired o2 fraction\b/i,
  /\bo2 flow\b/i,
  /\btidal volume\b/i,
  /\bminute volume\b/i,
  /\bairway pressure\b/i,
  /\bpeep\b/i,
  /\betco2\b/i,
  /\bco2 pressure\b/i,
  /\bo2 pressure\b/i,
  /\bcentral venous pressure\b/i,
  /\bpulmonary artery pressure\b/i,
  /\bintra cranial pressure\b/i,
  /\bcerebral perfusion pressure\b/i,
  /\bsv[o0]2\b/i,
];
const MIN_VITAL_STAY_POINTS_FOR_PLOT = 2;
const MIN_VITAL_STAY_WINDOW_COVERAGE_FOR_PLOT = 0.1;

const VITAL_GUIDELINE_RULES = [
  { pattern: /\bheart\s*rate\b/i, low: 60, high: 100, unit: "bpm" },
  { pattern: /\bblood\s*pressure\b.*\bsystolic\b/i, low: 90, high: 120, unit: "mmHg" },
  { pattern: /\bblood\s*pressure\b.*\bdiastolic\b/i, low: 60, high: 80, unit: "mmHg" },
  {
    pattern:
      /\b(mean\s*arterial\s*pressure|blood\s*pressure\s*mean|arterial\s*blood\s*pressure\s*mean|non\s*invasive\s*blood\s*pressure\s*mean|map)\b/i,
    low: 70,
    high: 100,
    unit: "mmHg",
  },
  {
    pattern: /\b(respiratory\s*rate|resp\s*rate|insp\s*\/\s*min)\b/i,
    low: 12,
    high: 20,
    unit: "insp/min",
  },
  {
    pattern: /\b(spo2|o2\s*saturation|oxygen\s*saturation|pulseox(?:ymetry)?)\b/i,
    low: 95,
    high: 100,
    unit: "%",
  },
];

const STATUS_CORRECTION_OPTIONS = ["improving", "stable", "deteriorating", "insufficient_data"];
const ACTION_CORRECTION_OPTIONS = ["appropriate", "suboptimal", "potentially_harmful", "not_enough_context"];
const ACTION_DISAGREE_REASONS = [
  "Oracle missed a patient-specific contraindication",
  "Oracle over-relies on hindsight",
  "Incorrect dose or rate judgment",
  "Poor timing",
  "No reasonable alternative was available",
  "Oracle's underlying patient status assessment is wrong",
  "Other",
];
const RECOMMENDATION_DISAGREE_REASONS = [
  "Poor timing",
  "Not a high enough priority",
  "Contraindicated in this patient",
  "Already being performed",
  "Other",
];
const OMISSION_CATEGORIES = [
  "Fluid management",
  "Vasopressor adjustment",
  "Enhanced monitoring",
  "Consultation",
  "Goal-directed therapy",
  "Other",
];

const state = {
  loaded: false,
  predictionsData: null,
  windowContextsData: null,
  sourceFiles: {
    oracle_predictions: null,
    window_contexts: null,
  },
  rounds: [],
  invalidWindows: [],
  annotations: {},
  currentRoundIndex: 0,
  currentEventTypeFilter: "all",
  historyEventTypeFilter: "all",
  loadedAtMs: 0,
  errorHighlights: {},
  globalMessage: {
    type: "",
    text: "",
  },
  vitalPlotStats: {
    totalWindows: 0,
    byLabel: new Map(),
  },
};

let loaderView = null;
let workspaceView = null;
let loaderMessageEl = null;
let globalMessageEl = null;
let runMetaEl = null;
let leftPanelEl = null;
let centerPanelEl = null;
let rightPanelEl = null;
let windowSelectEl = null;
let prevWindowBtn = null;
let nextWindowBtn = null;

window.addEventListener("DOMContentLoaded", () => {
  loaderView = document.getElementById("loader-view");
  workspaceView = document.getElementById("workspace-view");
  loaderMessageEl = document.getElementById("loader-message");
  globalMessageEl = document.getElementById("global-message");
  runMetaEl = document.getElementById("run-meta");
  leftPanelEl = document.getElementById("left-panel");
  centerPanelEl = document.getElementById("center-panel");
  rightPanelEl = document.getElementById("right-panel");
  windowSelectEl = document.getElementById("window-select");
  prevWindowBtn = document.getElementById("prev-window-btn");
  nextWindowBtn = document.getElementById("next-window-btn");

  document.getElementById("load-btn").addEventListener("click", onLoadFiles);
  document.getElementById("export-btn").addEventListener("click", onExport);
  prevWindowBtn.addEventListener("click", () => stepWindow(-1));
  nextWindowBtn.addEventListener("click", () => stepWindow(1));
  windowSelectEl.addEventListener("change", onWindowSelectChange);

  workspaceView.addEventListener("change", onWorkspaceChange);
  workspaceView.addEventListener("input", onWorkspaceInput);
});

async function onLoadFiles() {
  const predictionsInput = document.getElementById("predictions-file");
  const windowContextInput = document.getElementById("window-context-file");

  const predictionsFile = predictionsInput.files && predictionsInput.files[0] ? predictionsInput.files[0] : null;
  const windowContextFile = windowContextInput.files && windowContextInput.files[0] ? windowContextInput.files[0] : null;

  if (!predictionsFile) {
    setLoaderMessage("Please select oracle_predictions.json.", "warn");
    return;
  }

  try {
    setLoaderMessage("Loading files...", "info");
    const predictionsData = await readJsonFile(predictionsFile);
    const windowContextsData = windowContextFile ? await readJsonFile(windowContextFile) : null;

    initializeSession({
      predictionsData,
      windowContextsData,
      sourceFiles: {
        oracle_predictions: predictionsFile.name,
        window_contexts: windowContextFile ? windowContextFile.name : null,
      },
    });
    setLoaderMessage("", "");
  } catch (error) {
    setLoaderMessage(`Failed to load JSON: ${error.message}`, "warn");
  }
}

function initializeSession({ predictionsData, windowContextsData, sourceFiles }) {
  if (!isObject(predictionsData)) {
    throw new Error("oracle_predictions.json must contain a top-level JSON object.");
  }
  if (!Array.isArray(predictionsData.window_outputs)) {
    throw new Error("oracle_predictions.json is missing window_outputs[].");
  }

  const windowContextByIndex = buildWindowContextByIndex(windowContextsData);

  const rounds = [];
  const invalidWindows = [];
  const annotations = {};

  predictionsData.window_outputs.forEach((windowOutput, arrayIndex) => {
    const windowIndex = toInt(windowOutput && windowOutput.window_index, arrayIndex + 1);
    const validation = validateWindowOutput(windowOutput);

    if (!validation.valid) {
      invalidWindows.push({
        array_index: arrayIndex,
        window_index: windowIndex,
        reasons: validation.reasons,
      });
      return;
    }

    const normalizedActions = normalizeActions(windowOutput.oracle_output.action_evaluations);
    const normalizedRecommendations = normalizeRecommendations(windowOutput.oracle_output.recommendations);
    const contextEntry = resolveWindowContextEntry(windowIndex, arrayIndex, windowContextByIndex);
    const eventsResolution = resolveCurrentWindowEvents({
      windowOutput,
      contextEntry,
    });
    const rawEventsResolution = resolveRawWindowEvents({
      windowOutput,
      contextEntry,
    });
    const sectionResolution = resolveContextSections({
      contextEntry,
    });
    const historyEvents = resolveHistoryEvents({
      windowOutput,
      contextEntry,
    });

    const round = {
      arrayIndex,
      windowIndex,
      windowOutput,
      oracleOutput: windowOutput.oracle_output,
      actions: normalizedActions,
      recommendations: normalizedRecommendations,
      events: eventsResolution.events,
      eventSource: eventsResolution.source,
      rawEvents: rawEventsResolution.events,
      rawEventSource: rawEventsResolution.source,
      historyEvents,
      contextSections: sectionResolution.sections,
      contextSource: sectionResolution.source,
      windowKey: buildWindowKey(predictionsData.subject_id, predictionsData.icu_stay_id, windowIndex),
    };

    rounds.push(round);
    annotations[arrayIndex] = buildInitialAnnotation(round);
  });

  if (rounds.length === 0) {
    throw new Error("No valid windows found in this file.");
  }

  state.loaded = true;
  state.predictionsData = predictionsData;
  state.windowContextsData = windowContextsData;
  state.sourceFiles = sourceFiles;
  state.rounds = rounds;
  state.invalidWindows = invalidWindows;
  state.annotations = annotations;
  state.currentRoundIndex = 0;
  state.currentEventTypeFilter = "all";
  state.historyEventTypeFilter = "all";
  state.loadedAtMs = Date.now();
  state.errorHighlights = {};
  state.vitalPlotStats = buildStayVitalPlotStats(rounds);

  const invalidText = invalidWindows.length
    ? `Loaded ${rounds.length} valid windows; skipped ${invalidWindows.length} invalid window(s).`
    : `Loaded ${rounds.length} valid windows.`;
  setGlobalMessage(invalidText, invalidWindows.length ? "warn" : "info");

  loaderView.classList.add("hidden");
  workspaceView.classList.remove("hidden");
  renderWorkspace();
}

function validateWindowOutput(windowOutput) {
  const reasons = [];
  if (!isObject(windowOutput)) {
    return { valid: false, reasons: ["window entry is not an object"] };
  }

  const oracleOutput = windowOutput.oracle_output;
  if (!isObject(oracleOutput)) {
    reasons.push("oracle_output missing or not an object");
    return { valid: false, reasons };
  }

  REQUIRED_ORACLE_OUTPUT_KEYS.forEach((key) => {
    if (!(key in oracleOutput)) {
      reasons.push(`missing oracle_output.${key}`);
    }
  });

  if (!isObject(oracleOutput.patient_status)) {
    reasons.push("oracle_output.patient_status is not an object");
  }

  const overall = isObject(oracleOutput.patient_status) ? oracleOutput.patient_status.overall : null;
  if (!isObject(overall)) {
    reasons.push("oracle_output.patient_status.overall is not an object");
  } else {
    if (!safeString(overall.label)) {
      reasons.push("oracle_output.patient_status.overall.label missing");
    }
    if (!safeString(overall.rationale)) {
      reasons.push("oracle_output.patient_status.overall.rationale missing");
    }
  }

  if (!Array.isArray(oracleOutput.action_evaluations)) {
    reasons.push("oracle_output.action_evaluations is not an array");
  }
  if (!Array.isArray(oracleOutput.recommendations)) {
    reasons.push("oracle_output.recommendations is not an array");
  }
  if (typeof oracleOutput.overall_window_summary !== "string") {
    reasons.push("oracle_output.overall_window_summary is not a string");
  }

  return {
    valid: reasons.length === 0,
    reasons,
  };
}

function normalizeActions(actionEvaluations) {
  const safeActions = Array.isArray(actionEvaluations) ? actionEvaluations : [];
  const normalized = safeActions.map((rawAction, index) => {
    const action = isObject(rawAction) ? rawAction : {};
    const overall = isObject(action.overall) ? action.overall : {};
    const contextual = isObject(action.contextual_appropriateness) ? action.contextual_appropriateness : {};
    const guideline = isObject(action.guideline_adherence) ? action.guideline_adherence : {};

    const actionId = safeString(action.action_id) || `ACTION_${index + 1}`;
    const description = safeString(action.action_description) || "No description";
    const overallLabel = normalizeLabel(overall.label) || normalizeLabel(contextual.label);

    return {
      originalIndex: index,
      action_id: actionId,
      action_description: description,
      overall_label: overallLabel || "unknown",
      overall_rationale: safeString(overall.rationale),
      contextual,
      guideline,
    };
  });

  normalized.sort((a, b) => {
    const rankA = actionSeverityRank(a.overall_label);
    const rankB = actionSeverityRank(b.overall_label);
    if (rankA !== rankB) {
      return rankA - rankB;
    }
    return a.originalIndex - b.originalIndex;
  });

  return normalized;
}

function normalizeRecommendations(recommendations) {
  const safeRecs = Array.isArray(recommendations) ? recommendations : [];
  const normalized = safeRecs.map((rawRec, index) => {
    const rec = isObject(rawRec) ? rawRec : {};
    return {
      originalIndex: index,
      rank: toIntOrNull(rec.rank),
      urgency: safeString(rec.urgency) || "",
      action: safeString(rec.action) || "No action title",
      action_description: safeString(rec.action_description) || "",
      rationale: safeString(rec.rationale) || "",
    };
  });

  normalized.sort((a, b) => {
    const rankA = a.rank === null ? Number.POSITIVE_INFINITY : a.rank;
    const rankB = b.rank === null ? Number.POSITIVE_INFINITY : b.rank;
    if (rankA !== rankB) {
      return rankA - rankB;
    }
    return a.originalIndex - b.originalIndex;
  });

  normalized.forEach((rec, idx) => {
    rec.position_index = idx;
  });

  return normalized;
}

function resolveCurrentWindowEvents({ windowOutput, contextEntry }) {
  if (contextEntry && Array.isArray(contextEntry.current_events)) {
    return { events: cloneData(contextEntry.current_events), source: "window_contexts.current_events" };
  }

  if (Array.isArray(windowOutput.raw_current_events) && windowOutput.raw_current_events.length > 0) {
    return { events: cloneData(windowOutput.raw_current_events), source: "raw_current_events" };
  }

  const metadataEvents = isObject(windowOutput.window_metadata) ? windowOutput.window_metadata.current_events : null;
  if (Array.isArray(metadataEvents) && metadataEvents.length > 0) {
    return { events: cloneData(metadataEvents), source: "window_metadata.current_events" };
  }

  return { events: [], source: "none" };
}

function resolveRawWindowEvents({ windowOutput, contextEntry }) {
  if (Array.isArray(windowOutput.raw_current_events) && windowOutput.raw_current_events.length > 0) {
    return { events: cloneData(windowOutput.raw_current_events), source: "raw_current_events" };
  }

  if (contextEntry && Array.isArray(contextEntry.current_events) && contextEntry.current_events.length > 0) {
    return { events: cloneData(contextEntry.current_events), source: "window_contexts.current_events" };
  }

  const metadataEvents = isObject(windowOutput.window_metadata) ? windowOutput.window_metadata.current_events : null;
  if (Array.isArray(metadataEvents) && metadataEvents.length > 0) {
    return { events: cloneData(metadataEvents), source: "window_metadata.current_events" };
  }

  return { events: [], source: "none" };
}

function resolveHistoryEvents({ windowOutput, contextEntry }) {
  if (contextEntry && Array.isArray(contextEntry.history_events)) {
    return cloneData(contextEntry.history_events);
  }
  const historyEvents = windowOutput && Array.isArray(windowOutput.history_events) ? windowOutput.history_events : [];
  return cloneData(historyEvents);
}

function resolveContextSections({ contextEntry }) {
  const empty = {
    sections: {
      icu_discharge_summary: "",
      icu_trajectory_context_window: "",
      history_events_current_window: "",
      current_observation_window: "",
    },
    source: "none",
  };

  if (contextEntry && isObject(contextEntry.prompt_sections)) {
    return {
      sections: {
        icu_discharge_summary: safeString(contextEntry.prompt_sections.icu_discharge_summary),
        icu_trajectory_context_window: safeString(contextEntry.prompt_sections.icu_trajectory_context_window),
        history_events_current_window: safeString(contextEntry.prompt_sections.history_events_current_window),
        current_observation_window: safeString(contextEntry.prompt_sections.current_observation_window),
      },
      source: "window_contexts.prompt_sections",
    };
  }
  return empty;
}

function buildWindowContextByIndex(windowContextsData) {
  const map = new Map();
  if (!isObject(windowContextsData) || !Array.isArray(windowContextsData.window_contexts)) {
    return map;
  }

  windowContextsData.window_contexts.forEach((item, arrayIndex) => {
    if (!isObject(item)) {
      return;
    }
    const index = toInt(item.window_index, arrayIndex + 1);
    map.set(index, item);
  });
  return map;
}

function resolveWindowContextEntry(windowIndex, arrayIndex, contextMap) {
  if (!contextMap || contextMap.size === 0) {
    return null;
  }
  const candidates = [windowIndex, windowIndex - 1, arrayIndex + 1, arrayIndex];
  for (const candidate of candidates) {
    if (contextMap.has(candidate)) {
      return contextMap.get(candidate);
    }
  }
  return null;
}

function buildInitialAnnotation(round) {
  return {
    window_key: round.windowKey,
    task1_status: {
      verdict: null,
      correction_label: null,
      correction_domains: [],
      comment: null,
    },
    task2_actions: round.actions.map((action) => ({
      action_id: action.action_id,
      verdict: null,
      correction_label: null,
      disagree_reasons: [],
      comment: null,
    })),
    task3_recommendations: round.recommendations.map((rec) => ({
      rank: rec.rank,
      position_index: rec.position_index,
      verdict: null,
      disagree_reasons: [],
      comment: null,
    })),
    task3_omission: {
      has_omission: null,
      categories: [],
      description: null,
    },
  };
}

function renderWorkspace() {
  renderTopBar();
  renderLeftPanel();
  renderCenterPanel();
  renderRightPanel();
  attachEventFilterListeners();
  renderGlobalMessage();
}

function renderTopBar() {
  const round = getCurrentRound();
  const subjectId = safeString(state.predictionsData.subject_id);
  const icuStayId = safeString(state.predictionsData.icu_stay_id);
  const completion = getWindowCompletionSummary();

  runMetaEl.innerHTML = `
    <div class="run-meta-line">
      <strong>Patient:</strong> ${escapeHtml(subjectId || "-")}
      &nbsp;|&nbsp;
      <strong>ICU stay:</strong> ${escapeHtml(icuStayId || "-")}
      &nbsp;|&nbsp;
      <strong>Window key:</strong> ${escapeHtml(round.windowKey)}
    </div>
    <div class="run-meta-line">
      <strong>Total valid windows:</strong> ${state.rounds.length}
      &nbsp;|&nbsp;
      <strong>Invalid skipped:</strong> ${state.invalidWindows.length}
      &nbsp;|&nbsp;
      <strong>Finished:</strong> ${completion.finished}
      &nbsp;|&nbsp;
      <strong>Todo:</strong> ${completion.todo}
    </div>
  `;

  windowSelectEl.innerHTML = state.rounds
    .map((item, idx) => {
      const label = `Window ${item.windowIndex} (${idx + 1}/${state.rounds.length})`;
      const selected = idx === state.currentRoundIndex ? "selected" : "";
      return `<option value="${idx}" ${selected}>${escapeHtml(label)}</option>`;
    })
    .join("");

  prevWindowBtn.disabled = state.currentRoundIndex === 0;
  nextWindowBtn.disabled = state.currentRoundIndex === state.rounds.length - 1;
}

function renderLeftPanel() {
  const round = getCurrentRound();
  const annotation = getCurrentAnnotation();
  const metadata = isObject(round.windowOutput.window_metadata) ? round.windowOutput.window_metadata : {};
  const stats = getRoundReviewStats(round, annotation);
  const icuOutcome = getIcuOutcomeInfo(state.predictionsData);
  const dischargeSummaryText = safeString(round.contextSections.icu_discharge_summary);
  const promptHistoryView = parsePromptHistorySection(round.contextSections.history_events_current_window);
  const historyHours = resolveHistoryHours(round, promptHistoryView.historyHours);
  const historyEventsForDisplay = promptHistoryView.hasSection
    ? promptHistoryView.events
    : (Array.isArray(round.historyEvents) ? round.historyEvents : []);
  const filteredHistoryEvents = filterEventsByType(historyEventsForDisplay, state.historyEventTypeFilter);
  const historyTypeCounts = getEventTypeCounts(historyEventsForDisplay);
  const historyWindowLabel = historyHours !== null
    ? `${formatCompactNumber(historyHours)}-hour`
    : "history";
  const dischargeSummaryHtml = formatDischargeSummaryHtml(dischargeSummaryText);
  const historyEventsHtml = renderTimelineEvents(
    filteredHistoryEvents,
    `<div class="summary-card">No events match this filter in previous ${historyWindowLabel} window.</div>`
  );

  leftPanelEl.innerHTML = `
    <h2>Context</h2>
    <div class="summary-card">
      <div><strong>Round</strong>: ${state.currentRoundIndex + 1} / ${state.rounds.length}</div>
      <div><strong>Window index</strong>: ${round.windowIndex}</div>
      <div><strong>Range</strong>: ${escapeHtml(formatWindowRange(metadata))}</div>
      <div><strong>Hours since ICU admit</strong>: ${escapeHtml(formatNumber(metadata.hours_since_admission))}</div>
      <div>
        <strong>ICU outcome</strong>:
        <span class="outcome-text outcome-${escapeHtml(icuOutcome.toneClass)}">${escapeHtml(icuOutcome.label)}</span>
      </div>
    </div>

    <div class="summary-card">
      <strong>ICU discharge summary</strong>
      ${dischargeSummaryHtml}
    </div>

    <div class="summary-card">
      <div class="section-header-row">
        <strong>Previous ${historyWindowLabel} Observation</strong>
        ${renderEventTypeFilter({ counts: historyTypeCounts, scope: "history", compact: true })}
      </div>
      <div class="inline-meta">Events: ${filteredHistoryEvents.length} shown / ${historyEventsForDisplay.length} total</div>
      <div class="timeline-list">${historyEventsHtml}</div>
    </div>

    <div class="summary-card">
      <strong>Review progress</strong>
      <div class="stat-grid">
        <div class="stat-pill">Task 1: ${stats.task1Done ? "Reviewed" : "Pending"}</div>
        <div class="stat-pill">Actions: ${stats.reviewedActions}/${stats.totalActions}</div>
        <div class="stat-pill">Recs: ${stats.reviewedRecs}/${stats.totalRecs}</div>
        <div class="stat-pill">Omission Q: ${stats.omissionDone ? "Answered" : "Pending"}</div>
      </div>
    </div>
  `;
}

function renderCenterPanel() {
  const round = getCurrentRound();
  const currentEvents = Array.isArray(round.events) ? round.events : [];
  const rawEvents = Array.isArray(round.rawEvents) ? round.rawEvents : [];
  const eventsForDisplay = rawEvents.length > 0 ? rawEvents : currentEvents;
  const eventsSourceForDisplay = rawEvents.length > 0 ? round.rawEventSource : round.eventSource;
  const historyEvents = getPlotHistoryEvents(state.currentRoundIndex);
  const timelineStartMs = getTimelineStartMs();

  const vitalSeries = buildVitalSeries(historyEvents, currentEvents);
  const sparklineHtml = vitalSeries.length
    ? vitalSeries
      .map((series, idx) => {
        return renderSparklineCard({
          metricKey: series.metricKey || `vital-${idx + 1}`,
          metricConfig: series.metricConfig,
          historyPoints: series.history,
          currentPoints: series.current,
          timelineStartMs,
        });
      })
      .join("")
    : `<div class="summary-card">No vital trends available (no numeric data or ICU-stay frequency too low).</div>`;

  const filteredEvents = filterEventsByType(eventsForDisplay, state.currentEventTypeFilter);
  const eventListHtml = renderTimelineEvents(
    filteredEvents,
    `<div class="summary-card">No events available for this filter in current window.</div>`
  );

  const eventTypeCounts = getEventTypeCounts(eventsForDisplay);
  const sourceLabel = formatEventSource(eventsSourceForDisplay);

  centerPanelEl.innerHTML = `
    <h2>Current Observation Window</h2>
    <div class="summary-card">
      <strong>Window summary</strong>
      <div>${escapeHtml(safeString(round.oracleOutput.overall_window_summary) || "-")}</div>
    </div>
    <div class="inline-meta">Vital plots include all prior-window history (grey) and current-window points (blue).</div>
    <div class="sparkline-grid">${sparklineHtml}</div>
    <div class="section-header-row">
      <h3 class="subsection-title">Current events</h3>
      ${renderEventTypeFilter({ counts: eventTypeCounts, scope: "current", compact: true })}
    </div>
    <div class="inline-meta">Source: ${escapeHtml(sourceLabel)} | Events: ${eventsForDisplay.length}</div>
    <div class="timeline-list">${eventListHtml}</div>
  `;
}

function renderTimelineEvents(events, emptyHtml) {
  const safeEvents = Array.isArray(events) ? events : [];
  if (safeEvents.length === 0) {
    return emptyHtml || `<div class="summary-card">No events available.</div>`;
  }
  return safeEvents
    .map((event) => {
      const eventType = classifyEventType(event);
      const eventValue = formatEventValue(event);
      return `
        <article class="timeline-event event-${eventType}">
          <div class="event-top">
            <span class="time">${escapeHtml(formatTime(event.time || event.start_time || ""))}</span>
            <span class="code">${escapeHtml(safeString(event.code) || "UNKNOWN")}</span>
          </div>
          <div>${escapeHtml(formatEventDescription(event))}</div>
          ${eventValue ? `<div class="inline-meta">Value: ${escapeHtml(eventValue)}</div>` : ""}
        </article>
      `;
    })
    .join("");
}

function renderRightPanel() {
  const round = getCurrentRound();
  const annotation = getCurrentAnnotation();
  const task1Locked = annotation.task1_status.verdict === null;
  const patientStatus = round.oracleOutput.patient_status;
  const overall = isObject(patientStatus.overall) ? patientStatus.overall : {};

  rightPanelEl.innerHTML = `
    <h2>Annotation Tasks</h2>
    ${renderTask1(patientStatus, overall, annotation)}
    ${renderTask2(round, annotation, task1Locked)}
    ${renderTask3(round, annotation, task1Locked)}
  `;
}

function renderTask1(patientStatus, overall, annotation) {
  const task1 = annotation.task1_status;
  const domains = isObject(patientStatus.domains) ? patientStatus.domains : {};
  const overallLabel = normalizeLabel(overall.label) || "unknown";
  const overallTone = overallToneClass(overallLabel);
  const verdictErrorClass = hasCurrentHighlight("task1.verdict") ? "error-highlight" : "";
  const correctionErrorClass = hasCurrentHighlight("task1.correction_label") ? "error-highlight" : "";

  const domainCards = DOMAIN_KEYS.map((domainKey) => {
    const domain = isObject(domains[domainKey]) ? domains[domainKey] : {};
    const label = normalizeLabel(domain.label) || "unknown";
    const keySignals = Array.isArray(domain.key_signals) ? domain.key_signals.join("; ") : "";
    return `
      <div class="domain-card">
        <div><strong>${escapeHtml(domainKey)}</strong> <span class="badge badge-${escapeHtml(label)}">${escapeHtml(label)}</span></div>
        <div class="inline-meta">${escapeHtml(safeString(keySignals))}</div>
        <div>${escapeHtml(safeString(domain.rationale) || "-")}</div>
      </div>
    `;
  }).join("");

  const disagreeFields = task1.verdict === "disagree"
    ? `
      <div class="control-group ${correctionErrorClass}">
        <label for="task1-correction-label">Correct label</label>
        <select id="task1-correction-label" name="task1-correction-label">
          <option value="">Select label</option>
          ${STATUS_CORRECTION_OPTIONS.map((item) => {
            const selected = task1.correction_label === item ? "selected" : "";
            return `<option value="${item}" ${selected}>${item}</option>`;
          }).join("")}
        </select>
      </div>

      <div class="control-group">
        <label>Domains where you disagree</label>
        <div class="tag-grid">
          ${DOMAIN_KEYS.map((domainKey) => {
            const checked = task1.correction_domains.includes(domainKey) ? "checked" : "";
            return `
              <label class="tag-item">
                <input type="checkbox" data-task1-domain="${domainKey}" ${checked} />
                ${escapeHtml(domainKey)}
              </label>
            `;
          }).join("")}
        </div>
      </div>

      <div class="control-group">
        <label for="task1-comment">Comment</label>
        <textarea id="task1-comment" placeholder="Optional">${escapeHtml(safeString(task1.comment))}</textarea>
      </div>
    `
    : "";

  return `
    <section class="task-section" id="task1-section">
      <h3 class="task-title">Task 1: Patient Status</h3>
      <div class="summary-card overall-card overall-${escapeHtml(overallTone)}">
        <div class="inline-meta"><strong>overall:</strong> <span class="badge badge-${escapeHtml(overallLabel)}">${escapeHtml(
          overallLabel
        )}</span></div>
        <div>${escapeHtml(safeString(overall.rationale) || "-")}</div>
      </div>

      <details>
        <summary><strong>View domain details</strong></summary>
        ${domainCards}
      </details>

      <div class="verdict-row ${verdictErrorClass}">
        ${renderVerdictChips("task1-verdict", task1.verdict)}
      </div>

      ${disagreeFields}
    </section>
  `;
}

function renderTask2(round, annotation, locked) {
  const actionCards = round.actions
    .map((action, index) => {
      const actionAnn = annotation.task2_actions[index];
      const overallTone = overallToneClass(action.overall_label);
      const correctionErrorClass = hasCurrentHighlight(`action.${index}.correction_label`) ? "error-highlight" : "";
      const reasonErrorClass = hasCurrentHighlight(`action.${index}.disagree_reasons`) ? "error-highlight" : "";
      const cardErrorClass = correctionErrorClass || reasonErrorClass ? "error-highlight" : "";
      const disagreementFields = actionAnn.verdict === "disagree"
        ? `
          <div class="control-group ${correctionErrorClass}">
            <label for="action-correction-${index}">Correct overall label</label>
            <select id="action-correction-${index}" data-action-correction-index="${index}">
              <option value="">Select label</option>
              ${ACTION_CORRECTION_OPTIONS.map((option) => {
                const selected = actionAnn.correction_label === option ? "selected" : "";
                return `<option value="${option}" ${selected}>${option}</option>`;
              }).join("")}
            </select>
          </div>

          <div class="control-group ${reasonErrorClass}">
            <label>Reasons</label>
            <div class="tag-grid">
              ${ACTION_DISAGREE_REASONS.map((reason) => {
                const checked = actionAnn.disagree_reasons.includes(reason) ? "checked" : "";
                return `
                  <label class="tag-item">
                    <input type="checkbox" data-action-reason-index="${index}" data-reason="${escapeHtml(reason)}" ${checked} />
                    ${escapeHtml(reason)}
                  </label>
                `;
              }).join("")}
            </div>
          </div>

          <div class="control-group">
            <label for="action-comment-${index}">Comment</label>
            <textarea id="action-comment-${index}" data-action-comment-index="${index}" placeholder="Optional">${escapeHtml(
              safeString(actionAnn.comment)
            )}</textarea>
          </div>
        `
        : "";

      return `
        <article class="action-card overall-card overall-${escapeHtml(overallTone)} ${cardErrorClass}" id="action-card-${index}">
          <div><strong>${escapeHtml(action.action_id)}</strong> - ${escapeHtml(action.action_description)}</div>
          <div class="inline-meta"><strong>overall:</strong> <span class="badge badge-${escapeHtml(action.overall_label)}">${escapeHtml(
            action.overall_label
          )}</span></div>
          <div>${escapeHtml(action.overall_rationale || "No rationale provided.")}</div>

          <details>
            <summary>Details</summary>
            <div><strong>Guideline adherence:</strong> ${escapeHtml(normalizeLabel(action.guideline.label) || "-")}</div>
            <div>${escapeHtml(safeString(action.guideline.rationale) || "-")}</div>
            <div><strong>Reference:</strong> ${escapeHtml(safeString(action.guideline.guideline_reference) || "-")}</div>
            <hr>
            <div><strong>Contextual appropriateness:</strong> ${escapeHtml(
              normalizeLabel(action.contextual.label) || "-"
            )}</div>
            <div>${escapeHtml(safeString(action.contextual.rationale) || "-")}</div>
            ${safeString(action.contextual.hindsight_caveat)
              ? `<div class="inline-meta"><em>Hindsight caveat: ${escapeHtml(action.contextual.hindsight_caveat)}</em></div>`
              : ""}
          </details>

          <div class="verdict-row">
            ${renderVerdictChips(`action-verdict-${index}`, actionAnn.verdict, {
              "data-action-index": String(index),
            })}
          </div>

          ${disagreementFields}
        </article>
      `;
    })
    .join("");

  return `
    <section class="task-section" id="task2-section">
      <h3 class="task-title">Task 2: Action Evaluation</h3>
      <div class="review-note">No routine-tier suppression. All listed actions are available for review.</div>
      ${actionCards || `<div class="summary-card">No actions in this window.</div>`}
      ${locked ? '<div class="task-locked-overlay">Complete Task 1 to unlock Task 2</div>' : ""}
    </section>
  `;
}

function renderTask3(round, annotation, locked) {
  const recommendationCards = round.recommendations
    .map((rec, index) => {
      const recAnn = annotation.task3_recommendations[index];
      const reasonErrorClass = hasCurrentHighlight(`recommendation.${index}.disagree_reasons`) ? "error-highlight" : "";
      const disagreementFields = recAnn.verdict === "disagree"
        ? `
          <div class="control-group ${reasonErrorClass}">
            <label>Reasons</label>
            <div class="tag-grid">
              ${RECOMMENDATION_DISAGREE_REASONS.map((reason) => {
                const checked = recAnn.disagree_reasons.includes(reason) ? "checked" : "";
                return `
                  <label class="tag-item">
                    <input type="checkbox" data-rec-reason-index="${index}" data-reason="${escapeHtml(reason)}" ${checked} />
                    ${escapeHtml(reason)}
                  </label>
                `;
              }).join("")}
            </div>
          </div>

          <div class="control-group">
            <label for="rec-comment-${index}">Comment</label>
            <textarea id="rec-comment-${index}" data-rec-comment-index="${index}" placeholder="Optional">${escapeHtml(
              safeString(recAnn.comment)
            )}</textarea>
          </div>
        `
        : "";

      return `
        <article class="rec-card ${reasonErrorClass}">
          <div><strong>${escapeHtml(formatRecommendationRank(rec))} ${escapeHtml(rec.action)}</strong> ${rec.urgency ? `[${escapeHtml(
            rec.urgency
          )}]` : ""}</div>
          ${rec.action_description ? `<div>${escapeHtml(rec.action_description)}</div>` : ""}
          ${rec.rationale ? `<div class="inline-meta">Rationale: ${escapeHtml(rec.rationale)}</div>` : ""}

          <div class="verdict-row">
            ${renderVerdictChips(`rec-verdict-${index}`, recAnn.verdict, {
              "data-rec-index": String(index),
            })}
          </div>

          ${disagreementFields}
        </article>
      `;
    })
    .join("");

  const omission = annotation.task3_omission;
  const omissionBlock = `
    <div class="summary-card">
      <strong>Critical omission question</strong>
      <p>Within this window, was there an important intervention that should have been performed but was not - and that Oracle also did not recommend?</p>
      <div class="verdict-row">
        <label class="verdict-chip"><input type="radio" name="omission-verdict" value="false" ${
          omission.has_omission === false ? "checked" : ""
        } /> No obvious omission</label>
        <label class="verdict-chip"><input type="radio" name="omission-verdict" value="true" ${
          omission.has_omission === true ? "checked" : ""
        } /> Yes</label>
        <label class="verdict-chip"><input type="radio" name="omission-verdict" value="null" ${
          omission.has_omission === null ? "checked" : ""
        } /> Not answered</label>
      </div>

      ${
        omission.has_omission === true
          ? `
        <div class="control-group">
          <label>Categories</label>
          <div class="tag-grid">
            ${OMISSION_CATEGORIES.map((category) => {
              const checked = omission.categories.includes(category) ? "checked" : "";
              return `
                <label class="tag-item">
                  <input type="checkbox" data-omission-category="${escapeHtml(category)}" ${checked} />
                  ${escapeHtml(category)}
                </label>
              `;
            }).join("")}
          </div>
        </div>

        <div class="control-group ${hasCurrentHighlight("task3_omission.description") ? "error-highlight" : ""}">
          <label for="omission-description">Description</label>
          <textarea id="omission-description" placeholder="Required when omission is Yes">${escapeHtml(
            safeString(omission.description)
          )}</textarea>
        </div>
      `
          : ""
      }
    </div>
  `;

  return `
    <section class="task-section" id="task3-section">
      <h3 class="task-title">Task 3: Recommendation Review</h3>
      <div class="review-note">Recommendations are sorted by rank; null ranks are shown last.</div>
      ${recommendationCards || `<div class="summary-card">No recommendations in this window.</div>`}
      ${omissionBlock}
      ${locked ? '<div class="task-locked-overlay">Complete Task 1 to unlock Task 3</div>' : ""}
    </section>
  `;
}

function renderVerdictChips(name, selectedVerdict, extraAttributes = {}) {
  const options = [
    { value: "agree", label: "Agree" },
    { value: "disagree", label: "Disagree" },
    { value: "uncertain", label: "Uncertain" },
  ];

  const attrs = Object.entries(extraAttributes)
    .map(([k, v]) => `${k}="${escapeHtml(v)}"`)
    .join(" ");

  return options
    .map((option) => {
      const checked = selectedVerdict === option.value ? "checked" : "";
      return `
        <label class="verdict-chip">
          <input type="radio" name="${escapeHtml(name)}" value="${option.value}" ${attrs} ${checked} />
          ${escapeHtml(option.label)}
        </label>
      `;
    })
    .join("");
}

function renderSparklineCard({ metricKey, metricConfig, historyPoints, currentPoints, timelineStartMs }) {
  const config = isObject(metricConfig) ? metricConfig : {};
  const title = safeString(config.title || metricKey || "Vital");
  const unit = safeString(config.unit || "");
  const healthyRange = Array.isArray(config.healthy_range) ? config.healthy_range : null;

  const history = Array.isArray(historyPoints) ? historyPoints : [];
  const current = Array.isArray(currentPoints) ? currentPoints : [];
  const noCurrentValueNote = current.length === 0
    ? `<div class="sparkline-note warn">No current value in this window.</div>`
    : "";

  if (history.length === 0 && current.length === 0) {
    return `
      <div class="sparkline-card">
        <div class="sparkline-title">${escapeHtml(title)}${unit ? ` (${escapeHtml(unit)})` : ""}</div>
        <div class="sparkline-empty">No data in this window or history</div>
        ${noCurrentValueNote}
      </div>
    `;
  }

  const combined = [
    ...history.map((point) => ({ ...point, source: "history" })),
    ...current.map((point) => ({ ...point, source: "current" })),
  ].sort(compareTrendPoints);

  const numericPoints = combined.filter((point) => Number.isFinite(point.y));
  if (numericPoints.length === 0) {
    return `
      <div class="sparkline-card">
        <div class="sparkline-title">${escapeHtml(title)}${unit ? ` (${escapeHtml(unit)})` : ""}</div>
        <div class="sparkline-empty">No numeric data</div>
        ${noCurrentValueNote}
      </div>
    `;
  }

  const hasTimelineAnchor = Number.isFinite(timelineStartMs);
  const finitePointTimes = numericPoints.map((point) => point.timeMs).filter((time) => Number.isFinite(time));
  const fallbackAnchorMs = finitePointTimes.length > 0 ? finitePointTimes[0] : Number.NaN;
  const anchorMs = hasTimelineAnchor ? timelineStartMs : fallbackAnchorMs;

  const points = [];
  let previousXHour = 0;
  numericPoints.forEach((point, index) => {
    let xHour;
    if (Number.isFinite(point.timeMs) && Number.isFinite(anchorMs)) {
      xHour = (point.timeMs - anchorMs) / 3600000;
    } else if (Number.isFinite(anchorMs)) {
      xHour = previousXHour + 0.01;
    } else {
      xHour = index;
    }
    previousXHour = xHour;
    points.push({
      ...point,
      xHour,
      xIndex: index,
    });
  });
  const observedMin = Math.min(...points.map((point) => point.y));
  const observedMax = Math.max(...points.map((point) => point.y));

  let xMin = Math.min(...points.map((point) => point.xHour));
  let xMax = Math.max(...points.map((point) => point.xHour));
  if (xMin === xMax) {
    xMin -= 0.5;
    xMax += 0.5;
  }
  const xPad = (xMax - xMin) * 0.04;
  xMin -= xPad;
  xMax += xPad;

  const yValues = points.map((point) => point.y);
  const healthyLow = healthyRange && healthyRange.length >= 2 ? toNumber(healthyRange[0]) : Number.NaN;
  const healthyHigh = healthyRange && healthyRange.length >= 2 ? toNumber(healthyRange[1]) : Number.NaN;
  const hasHealthyRange = Number.isFinite(healthyLow) && Number.isFinite(healthyHigh) && healthyLow < healthyHigh;
  if (hasHealthyRange) {
    yValues.push(healthyLow, healthyHigh);
  }

  let yMin = Math.min(...yValues);
  let yMax = Math.max(...yValues);
  if (yMin === yMax) {
    yMin -= 1;
    yMax += 1;
  }
  const yPad = (yMax - yMin) * 0.08;
  yMin -= yPad;
  yMax += yPad;

  const width = 360;
  const height = 130;
  const plotLeft = 44;
  const plotRight = width - 12;
  const plotTop = 12;
  const plotBottom = height - 36;
  const plotWidth = plotRight - plotLeft;
  const plotHeight = plotBottom - plotTop;

  const mapX = (xHour) => plotLeft + (plotWidth * (xHour - xMin)) / (xMax - xMin || 1);
  const mapY = (value) => plotBottom - (plotHeight * (value - yMin)) / (yMax - yMin || 1);
  const toPolyline = (arr) => arr.map((point) => `${mapX(point.xHour).toFixed(2)},${mapY(point.y).toFixed(2)}`).join(" ");

  const historyPlotPoints = points.filter((point) => point.source === "history");
  const currentPlotPoints = points.filter((point) => point.source === "current");
  const currentWindowMissingNote = currentPlotPoints.length === 0
    ? `<div class="sparkline-note warn">No current value in this window.</div>`
    : "";

  const historyPolyline = toPolyline(historyPlotPoints);
  const currentPolyline = toPolyline(currentPlotPoints);

  const historyCircles = historyPlotPoints
    .map((point) => {
      return `<circle cx="${mapX(point.xHour).toFixed(2)}" cy="${mapY(point.y).toFixed(2)}" r="2.8" fill="#94a3b8" />`;
    })
    .join("");
  const currentCircles = currentPlotPoints
    .map((point) => {
      return `<circle cx="${mapX(point.xHour).toFixed(2)}" cy="${mapY(point.y).toFixed(2)}" r="3.2" fill="#0f5ea8" />`;
    })
    .join("");

  const firstCurrent = currentPlotPoints[0];
  const divider =
    firstCurrent && historyPlotPoints.length > 0
      ? `<line x1="${mapX(firstCurrent.xHour).toFixed(2)}" y1="${plotTop}" x2="${mapX(firstCurrent.xHour).toFixed(
          2
        )}" y2="${plotBottom}" stroke="#d1d5db" stroke-width="1" stroke-dasharray="3,3" />`
      : "";

  const xTicks = buildLinearTicks(xMin, xMax, 5);
  const yTicks = buildLinearTicks(yMin, yMax, 5);
  const xTickEls = xTicks
    .map((tick) => {
      const x = mapX(tick);
      return `
        <line x1="${x.toFixed(2)}" y1="${plotBottom}" x2="${x.toFixed(2)}" y2="${(plotBottom + 4).toFixed(
          2
        )}" stroke="#64748b" stroke-width="1" />
        <text x="${x.toFixed(2)}" y="${(plotBottom + 14).toFixed(2)}" text-anchor="middle" fill="#475569" font-size="7">${escapeHtml(
          formatHourTick(tick)
        )}</text>
      `;
    })
    .join("");
  const yTickEls = yTicks
    .map((tick) => {
      const y = mapY(tick);
      return `
        <line x1="${(plotLeft - 4).toFixed(2)}" y1="${y.toFixed(2)}" x2="${plotLeft.toFixed(2)}" y2="${y.toFixed(
          2
        )}" stroke="#64748b" stroke-width="1" />
        <line x1="${plotLeft.toFixed(2)}" y1="${y.toFixed(2)}" x2="${plotRight.toFixed(2)}" y2="${y.toFixed(
          2
        )}" stroke="#e2e8f0" stroke-width="0.8" />
        <text x="${(plotLeft - 6).toFixed(2)}" y="${(y + 3).toFixed(2)}" text-anchor="end" fill="#475569" font-size="7">${escapeHtml(
          formatAxisValue(tick)
        )}</text>
      `;
    })
    .join("");

  let healthyBand = "";
  if (hasHealthyRange) {
    const lowClamped = Math.max(yMin, Math.min(yMax, healthyLow));
    const highClamped = Math.max(yMin, Math.min(yMax, healthyHigh));
    const bandTop = mapY(highClamped);
    const bandBottom = mapY(lowClamped);
    const bandHeight = Math.max(0, bandBottom - bandTop);
    healthyBand = `
      <rect x="${plotLeft}" y="${bandTop.toFixed(2)}" width="${plotWidth.toFixed(2)}" height="${bandHeight.toFixed(
        2
      )}" fill="rgba(34, 197, 94, 0.14)" />
      <line x1="${plotLeft}" y1="${bandTop.toFixed(2)}" x2="${plotRight}" y2="${bandTop.toFixed(
        2
      )}" stroke="#16a34a" stroke-width="1" stroke-dasharray="4,3" />
      <line x1="${plotLeft}" y1="${bandBottom.toFixed(2)}" x2="${plotRight}" y2="${bandBottom.toFixed(
        2
      )}" stroke="#16a34a" stroke-width="1" stroke-dasharray="4,3" />
    `;
  }

  const xAxisLabel = hasTimelineAnchor ? "Time (h, 0 = first window start)" : "Time (h)";
  const healthyLegend = hasHealthyRange
    ? `<span class="legend-item"><span class="legend-dot reference"></span> healthy (${formatAxisValue(
        healthyLow
      )}-${formatAxisValue(healthyHigh)}${unit ? ` ${escapeHtml(unit)}` : ""})</span>`
    : "";

  return `
    <div class="sparkline-card">
      <div class="sparkline-title">${escapeHtml(title)}${unit ? ` (${escapeHtml(unit)})` : ""} (min ${formatAxisValue(
    observedMin
  )}, max ${formatAxisValue(observedMax)})</div>
      <div class="sparkline-legend">
        <span class="legend-item"><span class="legend-dot history"></span> history (${historyPlotPoints.length})</span>
        <span class="legend-item"><span class="legend-dot current"></span> current (${currentPlotPoints.length})</span>
        ${healthyLegend}
      </div>
      ${currentWindowMissingNote}
      <svg width="100%" viewBox="0 0 ${width} ${height}" preserveAspectRatio="none" role="img" aria-label="${escapeHtml(
        title
      )} trend plot">
        ${healthyBand}
        ${yTickEls}
        <line x1="${plotLeft}" y1="${plotBottom}" x2="${plotRight}" y2="${plotBottom}" stroke="#334155" stroke-width="1.2" />
        <line x1="${plotLeft}" y1="${plotTop}" x2="${plotLeft}" y2="${plotBottom}" stroke="#334155" stroke-width="1.2" />
        ${xTickEls}
        ${divider}
        ${historyPlotPoints.length > 1 ? `<polyline fill="none" stroke="#94a3b8" stroke-width="1.8" points="${historyPolyline}" />` : ""}
        ${currentPlotPoints.length > 1 ? `<polyline fill="none" stroke="#0f5ea8" stroke-width="2.2" points="${currentPolyline}" />` : ""}
        ${historyCircles}
        ${currentCircles}
        <text x="${((plotLeft + plotRight) / 2).toFixed(2)}" y="${(height - 8).toFixed(
          2
        )}" text-anchor="middle" fill="#334155" font-size="9">${escapeHtml(xAxisLabel)}</text>
        <text x="12" y="${((plotTop + plotBottom) / 2).toFixed(
          2
        )}" text-anchor="middle" fill="#334155" font-size="9" transform="rotate(-90 12 ${(
    (plotTop + plotBottom) / 2
  ).toFixed(2)})">Value</text>
      </svg>
    </div>
  `;
}

function buildLinearTicks(minValue, maxValue, count) {
  if (!Number.isFinite(minValue) || !Number.isFinite(maxValue)) {
    return [];
  }
  const safeCount = Math.max(2, toInt(count, 5));
  const span = maxValue - minValue;
  if (Math.abs(span) < 1e-9) {
    return [minValue];
  }
  const ticks = [];
  for (let index = 0; index < safeCount; index += 1) {
    ticks.push(minValue + (span * index) / (safeCount - 1));
  }
  return ticks;
}

function formatHourTick(value) {
  const n = toNumber(value);
  if (!Number.isFinite(n)) {
    return "-";
  }
  const rounded = Math.round(n * 10) / 10;
  return rounded.toFixed(1);
}

function formatAxisValue(value) {
  const n = toNumber(value);
  if (!Number.isFinite(n)) {
    return "-";
  }
  const abs = Math.abs(n);
  if (abs >= 100) {
    return String(Math.round(n));
  }
  if (abs >= 10) {
    return (Math.round(n * 10) / 10).toFixed(1);
  }
  return (Math.round(n * 100) / 100).toFixed(2);
}

function formatDischargeSummaryHtml(dischargeSummaryText) {
  const text = safeString(dischargeSummaryText);
  if (!text) {
    return `<div class="inline-meta">No ICU discharge summary context available.</div>`;
  }

  const normalized = normalizeDischargeSummaryText(text);
  if (!normalized) {
    return `<div class="inline-meta">No ICU discharge summary context available.</div>`;
  }

  const sections = parseDischargeSummarySections(normalized);

  if (sections.length <= 1) {
    return `<pre class="context-pre">${escapeHtml(normalized)}</pre>`;
  }

  return `
    <div class="context-section-list">
      ${sections
        .map((section, index) => {
          const titleHtml = section.title
            ? `<div class="context-section-title">${escapeHtml(section.title)}</div>`
            : "";
          const separator = index < sections.length - 1 ? `<div class="context-section-separator"></div>` : "";
          return `
            <div class="context-section-block">
              ${titleHtml}
              <pre class="context-pre">${escapeHtml(section.body)}</pre>
            </div>
            ${separator}
          `;
        })
        .join("")}
    </div>
  `;
}

function normalizeDischargeSummaryText(text) {
  let normalized = safeString(text).replace(/\r\n/g, "\n");
  if (!normalized) {
    return "";
  }

  // Preserve existing blocks first.
  normalized = normalized.replace(/\n{3,}/g, "\n\n");

  // Convert common uppercase banner lines into section headings.
  normalized = normalized.replace(/={3,}\s*([A-Z][A-Z /]+?)\s*={3,}/g, "\n\n## $1\n");

  // Convert inline issue bullets (e.g., "# COPD ...") into their own headings.
  normalized = normalized.replace(/(^|\s)#\s*([A-Za-z][^\n#]{2,80})/g, "\n\n## $2\n");

  // Add section breaks before common discharge-summary headings even if inline.
  const headingRegex = new RegExp(
    String.raw`\b(` +
      [
        "MEDICINE\\s+Allergies",
        "Allergies",
        "Attending",
        "Chief Complaint",
        "Major Surgical or Invasive Procedure",
        "History of Present Illness",
        "Past Medical History",
        "Social History",
        "Family History",
        "Physical Exam(?:ination)?",
        "Admission Physical Exam",
        "Discharge Physical Examination",
        "Pertinent Results",
        "Admission Labs",
        "Imaging/?Procedures?",
        "Brief Hospital Course",
        "Summary",
        "Acute Issues",
        "Chronic Issues",
        "Medications on Admission",
        "Discharge Medications",
        "Discharge Disposition",
        "Discharge Diagnosis",
        "Discharge Condition",
        "Discharge Instructions",
        "Followup Instructions",
      ].join("|") +
      String.raw`)\s*:`,
    "gi"
  );
  normalized = normalized.replace(headingRegex, (_, heading) => `\n\n## ${safeString(heading)}:\n`);

  // Improve readability of long medication lists written inline.
  normalized = normalized.replace(/(^|[^\n])(\d+\.\s+)/g, (match, prefix, numbered) => `${prefix}\n${numbered}`);

  // Clean up excessive empty lines introduced by normalization.
  normalized = normalized.replace(/\n{3,}/g, "\n\n");
  return normalized.trim();
}

function parseDischargeSummarySections(normalizedText) {
  const lines = normalizedText.split("\n");
  const sections = [];
  let current = { title: null, bodyLines: [] };

  const pushCurrent = () => {
    const body = current.bodyLines.join("\n").trim();
    if (!current.title && !body) {
      return;
    }
    sections.push({
      title: current.title ? current.title.trim() : null,
      body: body || "-",
    });
  };

  lines.forEach((line) => {
    const headingMatch = line.match(/^##\s+(.+)$/);
    if (headingMatch) {
      pushCurrent();
      current = {
        title: headingMatch[1].replace(/:\s*$/, "").trim(),
        bodyLines: [],
      };
      return;
    }
    current.bodyLines.push(line);
  });

  pushCurrent();
  return sections.filter((section) => section.title || section.body);
}

function onWindowSelectChange(event) {
  const selectedIndex = toInt(event.target.value, state.currentRoundIndex);
  if (selectedIndex === null) {
    return;
  }
  state.currentRoundIndex = clamp(selectedIndex, 0, state.rounds.length - 1);
  state.currentEventTypeFilter = "all";
  state.historyEventTypeFilter = "all";
  renderWorkspace();
}

function stepWindow(offset) {
  state.currentRoundIndex = clamp(state.currentRoundIndex + offset, 0, state.rounds.length - 1);
  state.currentEventTypeFilter = "all";
  state.historyEventTypeFilter = "all";
  renderWorkspace();
}

function onWorkspaceChange(event) {
  if (!state.loaded) {
    return;
  }

  const target = event.target;
  const annotation = getCurrentAnnotation();
  const previousTask1Verdict = annotation.task1_status.verdict;
  let shouldRender = false;
  clearCurrentRoundErrorHighlights();

  if (target.name === "task1-verdict") {
    annotation.task1_status.verdict = target.value;
    if (target.value !== "disagree") {
      annotation.task1_status.correction_label = null;
      annotation.task1_status.correction_domains = [];
      annotation.task1_status.comment = null;
    }
    shouldRender = true;

    if (previousTask1Verdict === null && annotation.task1_status.verdict !== null) {
      setTimeout(() => {
        const task2 = document.getElementById("task2-section");
        if (task2) {
          task2.scrollIntoView({ behavior: "smooth", block: "start" });
        }
      }, 80);
    }
  }

  if (target.name === "task1-correction-label") {
    annotation.task1_status.correction_label = target.value || null;
  }

  if (target.dataset.task1Domain) {
    toggleSelection(annotation.task1_status.correction_domains, target.dataset.task1Domain, target.checked);
  }

  if (target.name.startsWith("action-verdict-")) {
    const actionIndex = toInt(target.dataset.actionIndex, null);
    if (actionIndex !== null && annotation.task2_actions[actionIndex]) {
      annotation.task2_actions[actionIndex].verdict = target.value;
      if (target.value !== "disagree") {
        annotation.task2_actions[actionIndex].correction_label = null;
        annotation.task2_actions[actionIndex].disagree_reasons = [];
        annotation.task2_actions[actionIndex].comment = null;
      }
      shouldRender = true;
    }
  }

  if (target.dataset.actionCorrectionIndex) {
    const actionIndex = toInt(target.dataset.actionCorrectionIndex, null);
    if (actionIndex !== null && annotation.task2_actions[actionIndex]) {
      annotation.task2_actions[actionIndex].correction_label = target.value || null;
    }
  }

  if (target.dataset.actionReasonIndex) {
    const actionIndex = toInt(target.dataset.actionReasonIndex, null);
    if (actionIndex !== null && annotation.task2_actions[actionIndex]) {
      toggleSelection(
        annotation.task2_actions[actionIndex].disagree_reasons,
        target.dataset.reason || "",
        target.checked
      );
    }
  }

  if (target.name.startsWith("rec-verdict-")) {
    const recIndex = toInt(target.dataset.recIndex, null);
    if (recIndex !== null && annotation.task3_recommendations[recIndex]) {
      annotation.task3_recommendations[recIndex].verdict = target.value;
      if (target.value !== "disagree") {
        annotation.task3_recommendations[recIndex].disagree_reasons = [];
        annotation.task3_recommendations[recIndex].comment = null;
      }
      shouldRender = true;
    }
  }

  if (target.dataset.recReasonIndex) {
    const recIndex = toInt(target.dataset.recReasonIndex, null);
    if (recIndex !== null && annotation.task3_recommendations[recIndex]) {
      toggleSelection(
        annotation.task3_recommendations[recIndex].disagree_reasons,
        target.dataset.reason || "",
        target.checked
      );
    }
  }

  if (target.name === "omission-verdict") {
    if (target.value === "true") {
      annotation.task3_omission.has_omission = true;
    } else if (target.value === "false") {
      annotation.task3_omission.has_omission = false;
      annotation.task3_omission.categories = [];
      annotation.task3_omission.description = null;
    } else {
      annotation.task3_omission.has_omission = null;
      annotation.task3_omission.categories = [];
      annotation.task3_omission.description = null;
    }
    shouldRender = true;
  }

  if (target.dataset.omissionCategory) {
    toggleSelection(annotation.task3_omission.categories, target.dataset.omissionCategory, target.checked);
  }

  if (shouldRender) {
    renderWorkspace();
  }
}

function onWorkspaceInput(event) {
  if (!state.loaded) {
    return;
  }

  const target = event.target;
  const annotation = getCurrentAnnotation();
  clearCurrentRoundErrorHighlights();

  if (target.id === "task1-comment") {
    annotation.task1_status.comment = safeString(target.value) || null;
  }

  if (target.dataset.actionCommentIndex) {
    const actionIndex = toInt(target.dataset.actionCommentIndex, null);
    if (actionIndex !== null && annotation.task2_actions[actionIndex]) {
      annotation.task2_actions[actionIndex].comment = safeString(target.value) || null;
    }
  }

  if (target.dataset.recCommentIndex) {
    const recIndex = toInt(target.dataset.recCommentIndex, null);
    if (recIndex !== null && annotation.task3_recommendations[recIndex]) {
      annotation.task3_recommendations[recIndex].comment = safeString(target.value) || null;
    }
  }

  if (target.id === "omission-description") {
    annotation.task3_omission.description = safeString(target.value) || null;
  }
}

function onExport() {
  if (!state.loaded) {
    return;
  }

  const validation = collectValidationReport();

  if (validation.blocking.length > 0) {
    setErrorHighlightsFromBlockingDetails(validation.blockingDetails);
    const firstBlocked = validation.blockingDetails[0];
    if (firstBlocked) {
      const firstBlockedRoundIndex = state.rounds.findIndex((round) => round.arrayIndex === firstBlocked.arrayIndex);
      if (firstBlockedRoundIndex >= 0) {
        state.currentRoundIndex = firstBlockedRoundIndex;
      }
    }
    renderWorkspace();

    const message = [`Cannot export yet (${validation.blocking.length} blocking issue(s)):`]
      .concat(validation.blocking.slice(0, 12).map((issue) => `- ${issue}`))
      .join("\n");
    window.alert(message);
    setGlobalMessage(message.replace(/\n/g, " "), "warn");
    return;
  }

  if (validation.warnings.length > 0) {
    const warningMessage = [`There are ${validation.warnings.length} review warning(s). Export anyway?`]
      .concat(validation.warnings.slice(0, 10).map((item) => `- ${item}`))
      .join("\n");
    if (!window.confirm(warningMessage)) {
      return;
    }
  }

  const nowIso = new Date().toISOString();
  const durationSeconds = Math.max(0, Math.floor((Date.now() - state.loadedAtMs) / 1000));

  const output = cloneData(state.predictionsData);
  state.rounds.forEach((round) => {
    const annotation = state.annotations[round.arrayIndex];
    const warnings = validation.windowWarnings[round.arrayIndex] || {
      unreviewed_actions: [],
      unreviewed_recommendations: [],
      omission_unanswered: false,
    };

    output.window_outputs[round.arrayIndex].annotation = {
      window_key: round.windowKey,
      annotated_at: nowIso,
      duration_seconds: durationSeconds,
      task1_status: {
        verdict: annotation.task1_status.verdict,
        correction_label: annotation.task1_status.correction_label,
        correction_domains: annotation.task1_status.correction_domains.slice(),
        comment: annotation.task1_status.comment,
      },
      task2_actions: annotation.task2_actions.map((entry) => ({
        action_id: entry.action_id,
        verdict: entry.verdict,
        correction_label: entry.correction_label,
        disagree_reasons: entry.disagree_reasons.slice(),
        comment: entry.comment,
      })),
      task3_recommendations: annotation.task3_recommendations.map((entry) => ({
        rank: entry.rank,
        position_index: entry.position_index,
        verdict: entry.verdict,
        disagree_reasons: entry.disagree_reasons.slice(),
        comment: entry.comment,
      })),
      task3_omission: {
        has_omission: annotation.task3_omission.has_omission,
        categories: annotation.task3_omission.categories.slice(),
        description: annotation.task3_omission.description,
      },
      review_warnings: warnings,
    };
  });

  output.annotation_run = {
    annotated_at: nowIso,
    valid_windows: state.rounds.length,
    invalid_windows: state.invalidWindows.map((item) => item.window_index),
    source_files: {
      oracle_predictions: state.sourceFiles.oracle_predictions,
      window_contexts: state.sourceFiles.window_contexts,
    },
  };

  const filename = `${safeString(output.subject_id) || "subject"}_${safeString(output.icu_stay_id) || "stay"}_annotated.json`;
  state.errorHighlights = {};
  downloadJson(output, filename);
  setGlobalMessage(`Exported ${filename}`, "info");
}

function collectValidationReport() {
  const blocking = [];
  const blockingDetails = [];
  const warnings = [];
  const windowWarnings = {};

  state.rounds.forEach((round) => {
    const annotation = state.annotations[round.arrayIndex];
    const labelPrefix = `${round.windowKey}`;
    const roundIssues = collectRoundBlockingIssues(round, annotation);
    roundIssues.forEach((issue) => {
      const fullMessage = `${labelPrefix}: ${issue.message}`;
      blocking.push(fullMessage);
      blockingDetails.push({
        arrayIndex: round.arrayIndex,
        token: issue.token,
        message: fullMessage,
      });
    });

    const unreviewedActions = annotation.task2_actions
      .filter((action) => action.verdict === null)
      .map((action) => action.action_id);
    const unreviewedRecommendations = annotation.task3_recommendations
      .filter((rec) => rec.verdict === null)
      .map((rec) => formatRecommendationId(rec));
    const omissionUnanswered = annotation.task3_omission.has_omission === null;

    windowWarnings[round.arrayIndex] = {
      unreviewed_actions: unreviewedActions,
      unreviewed_recommendations: unreviewedRecommendations,
      omission_unanswered: omissionUnanswered,
    };

    if (unreviewedActions.length > 0) {
      warnings.push(`${labelPrefix}: unreviewed actions (${unreviewedActions.join(", ")})`);
    }
    if (unreviewedRecommendations.length > 0) {
      warnings.push(`${labelPrefix}: unreviewed recommendations (${unreviewedRecommendations.join(", ")})`);
    }
    if (omissionUnanswered) {
      warnings.push(`${labelPrefix}: critical omission question is unanswered.`);
    }
  });

  return {
    blocking,
    blockingDetails,
    warnings,
    windowWarnings,
  };
}

function collectRoundBlockingIssues(round, annotation) {
  const issues = [];

  if (!annotation.task1_status.verdict) {
    issues.push({
      token: "task1.verdict",
      message: "Task 1 verdict is missing.",
    });
  }

  if (annotation.task1_status.verdict === "disagree" && !annotation.task1_status.correction_label) {
    issues.push({
      token: "task1.correction_label",
      message: "Task 1 disagree requires a corrected label.",
    });
  }

  annotation.task2_actions.forEach((action, actionIndex) => {
    if (action.verdict !== "disagree") {
      return;
    }
    if (!action.correction_label) {
      issues.push({
        token: `action.${actionIndex}.correction_label`,
        message: `Action ${action.action_id} disagree requires corrected label.`,
      });
    }
    if (!Array.isArray(action.disagree_reasons) || action.disagree_reasons.length === 0) {
      issues.push({
        token: `action.${actionIndex}.disagree_reasons`,
        message: `Action ${action.action_id} disagree requires at least one reason.`,
      });
    }
  });

  annotation.task3_recommendations.forEach((rec, recIndex) => {
    if (rec.verdict === "disagree" && (!Array.isArray(rec.disagree_reasons) || rec.disagree_reasons.length === 0)) {
      issues.push({
        token: `recommendation.${recIndex}.disagree_reasons`,
        message: `Recommendation ${formatRecommendationId(rec)} disagree requires at least one reason.`,
      });
    }
  });

  if (annotation.task3_omission.has_omission === true && !safeString(annotation.task3_omission.description)) {
    issues.push({
      token: "task3_omission.description",
      message: "Omission description is required when omission is Yes.",
    });
  }

  return issues;
}

function getWindowCompletionSummary() {
  let finished = 0;
  state.rounds.forEach((round) => {
    const annotation = state.annotations[round.arrayIndex];
    const issues = collectRoundBlockingIssues(round, annotation);
    if (issues.length === 0) {
      finished += 1;
    }
  });
  return {
    finished,
    todo: Math.max(0, state.rounds.length - finished),
  };
}

function getRoundReviewStats(round, annotation) {
  const reviewedActions = annotation.task2_actions.filter((entry) => entry.verdict !== null).length;
  const reviewedRecs = annotation.task3_recommendations.filter((entry) => entry.verdict !== null).length;

  return {
    task1Done: annotation.task1_status.verdict !== null,
    reviewedActions,
    totalActions: round.actions.length,
    reviewedRecs,
    totalRecs: round.recommendations.length,
    omissionDone: annotation.task3_omission.has_omission !== null,
  };
}

function getPlotHistoryEvents(roundPosition) {
  const currentRound = state.rounds[roundPosition];
  const currentStartMs = getWindowStartMs(currentRound);
  const priorEvents = collectPriorEvents(roundPosition);
  if (currentStartMs === null) {
    return priorEvents;
  }
  return priorEvents.filter((event) => {
    const timeMs = parseEventTimeMs(event);
    return timeMs === null || timeMs < currentStartMs;
  });
}

function getRecentHistoryEvents(roundPosition, minutes) {
  const currentRound = state.rounds[roundPosition];
  const currentStartMs = getWindowStartMs(currentRound);
  const priorEvents = collectPriorEvents(roundPosition);
  if (priorEvents.length === 0) {
    return [];
  }

  if (currentStartMs === null) {
    const previousRound = roundPosition > 0 ? state.rounds[roundPosition - 1] : null;
    return previousRound && Array.isArray(previousRound.events) ? previousRound.events.slice() : [];
  }

  const windowMs = Math.max(1, Number(minutes) || 30) * 60 * 1000;
  const lowerBound = currentStartMs - windowMs;
  return priorEvents
    .filter((event) => {
      const timeMs = parseEventTimeMs(event);
      return timeMs !== null && timeMs >= lowerBound && timeMs < currentStartMs;
    })
    .sort((a, b) => {
      const ta = parseEventTimeMs(a) || 0;
      const tb = parseEventTimeMs(b) || 0;
      return ta - tb;
    });
}

function parsePromptHistorySection(sectionText) {
  const parsed = {
    hasSection: false,
    historyHours: null,
    events: [],
  };

  const text = safeString(sectionText);
  if (!text) {
    return parsed;
  }

  parsed.hasSection = true;
  const lines = text.split("\n").map((line) => safeString(line)).filter((line) => line.length > 0);
  lines.forEach((line) => {
    const hourMatch = line.match(/^History duration \(hours\):\s*([0-9]+(?:\.[0-9]+)?)$/i);
    if (hourMatch) {
      const value = toNumber(hourMatch[1]);
      if (Number.isFinite(value)) {
        parsed.historyHours = value;
      }
      return;
    }

    const eventMatch = line.match(/^HX\d+\.\s+(.+)$/);
    if (!eventMatch) {
      return;
    }
    parsed.events.push(parsePromptEventLine(eventMatch[1], line));
  });

  return parsed;
}

function parsePromptEventLine(eventText, rawLine) {
  const text = safeString(eventText);
  const fallback = {
    time: "",
    code: "UNKNOWN",
    code_specifics: text,
    text_value: null,
    numeric_value: null,
    raw_line: safeString(rawLine) || text,
  };

  if (!text) {
    return fallback;
  }

  const match = text.match(/^(\d{4}-\d{2}-\d{2}\s+\d{2}:\d{2}(?::\d{2})?)\s+([A-Z0-9_]+)\s*(.*)$/);
  if (!match) {
    return fallback;
  }

  const time = match[1];
  const code = match[2];
  const details = safeString(match[3]);
  let codeSpecifics = details;
  let numericValue = null;
  let textValue = null;

  if (details.includes("=")) {
    const eqIndex = details.indexOf("=");
    codeSpecifics = safeString(details.slice(0, eqIndex));
    const rhs = safeString(details.slice(eqIndex + 1));
    const rhsNumber = toNumber(rhs);
    if (/^-?\d+(?:\.\d+)?$/.test(rhs) && Number.isFinite(rhsNumber)) {
      numericValue = rhsNumber;
    } else if (rhs) {
      textValue = rhs;
    }
  }

  return {
    time,
    code,
    code_specifics: codeSpecifics,
    text_value: textValue,
    numeric_value: numericValue,
    raw_line: safeString(rawLine) || text,
  };
}

function resolveHistoryHours(round, parsedHistoryHours) {
  const globalCandidates = [
    state.windowContextsData && state.windowContextsData.history_hours,
    state.windowContextsData && state.windowContextsData.context_history_hours,
    state.windowContextsData
      && isObject(state.windowContextsData.oracle_context)
      && state.windowContextsData.oracle_context.history_hours,
  ];

  for (const candidate of globalCandidates) {
    const value = toNumber(candidate);
    if (Number.isFinite(value)) {
      return value;
    }
  }

  const perWindowCandidates = [
    round && round.windowOutput && isObject(round.windowOutput.window_metadata)
      ? round.windowOutput.window_metadata.history_hours
      : null,
    parsedHistoryHours,
  ];
  for (const candidate of perWindowCandidates) {
    const value = toNumber(candidate);
    if (Number.isFinite(value)) {
      return value;
    }
  }
  return null;
}

function collectPriorEvents(roundPosition) {
  const events = [];
  const signatures = new Set();

  for (let index = 0; index < roundPosition; index += 1) {
    const round = state.rounds[index];
    const roundEvents = Array.isArray(round.events) ? round.events : [];
    roundEvents.forEach((event) => {
      const signature = buildEventSignature(event);
      if (signatures.has(signature)) {
        return;
      }
      signatures.add(signature);
      events.push(event);
    });
  }

  return events;
}

function buildEventSignature(event) {
  return [
    safeString(event.time || event.start_time),
    safeString(event.code),
    safeString(event.code_specifics),
    safeString(event.numeric_value),
    safeString(event.text_value),
  ].join("|");
}

function getWindowStartMs(round) {
  const metadata = isObject(round && round.windowOutput && round.windowOutput.window_metadata)
    ? round.windowOutput.window_metadata
    : {};
  const startCandidate = metadata.window_start_time || round?.windowOutput?.current_window_start || "";
  return parseEventTimeMs({ time: startCandidate });
}

function getTimelineStartMs() {
  if (!Array.isArray(state.rounds) || state.rounds.length === 0) {
    return null;
  }

  const firstRoundStart = getWindowStartMs(state.rounds[0]);
  if (Number.isFinite(firstRoundStart)) {
    return firstRoundStart;
  }

  const finiteStarts = state.rounds
    .map((round) => getWindowStartMs(round))
    .filter((value) => Number.isFinite(value));
  if (finiteStarts.length > 0) {
    return Math.min(...finiteStarts);
  }

  return null;
}

function buildStayVitalPlotStats(rounds) {
  const safeRounds = Array.isArray(rounds) ? rounds : [];
  const byLabel = new Map();
  const signatures = new Set();

  safeRounds.forEach((round, roundPosition) => {
    const roundEvents = Array.isArray(round.events) ? round.events : [];
    roundEvents.forEach((event, eventIndex) => {
      const signature = buildEventSignature(event);
      if (signatures.has(signature)) {
        return;
      }
      signatures.add(signature);

      const normalizedEvent = {
        event,
        source: "stay",
        order: eventIndex,
        time: safeString(event.time || event.start_time),
        timeMs: parseEventTimeMs(event),
      };
      if (!isVitalEvent(normalizedEvent)) {
        return;
      }

      const label = extractVitalLabel(normalizedEvent);
      if (!label) {
        return;
      }

      const numeric = extractEventNumericValue(normalizedEvent);
      if (!Number.isFinite(numeric)) {
        return;
      }

      const labelKey = normalizeLabel(label);
      if (!labelKey) {
        return;
      }

      if (!byLabel.has(labelKey)) {
        byLabel.set(labelKey, {
          pointCount: 0,
          windows: new Set(),
        });
      }

      const stats = byLabel.get(labelKey);
      stats.pointCount += 1;
      stats.windows.add(round.windowIndex ?? roundPosition);
    });
  });

  return {
    totalWindows: safeRounds.length,
    byLabel,
  };
}

function shouldPlotVitalForStayFrequency(labelKey) {
  const statsRoot = state.vitalPlotStats;
  if (!statsRoot || !(statsRoot.byLabel instanceof Map)) {
    return true;
  }

  const totalWindows = toInt(statsRoot.totalWindows, 0);
  if (totalWindows <= 0) {
    return true;
  }

  const stats = statsRoot.byLabel.get(labelKey);
  if (!stats) {
    return false;
  }

  const pointCount = toInt(stats.pointCount, 0);
  const windowCount = stats.windows instanceof Set ? stats.windows.size : 0;
  const coverage = windowCount / totalWindows;
  const minPointCount = Math.min(MIN_VITAL_STAY_POINTS_FOR_PLOT, Math.max(1, totalWindows));

  return pointCount >= minPointCount && coverage >= MIN_VITAL_STAY_WINDOW_COVERAGE_FOR_PLOT;
}

function normalizeEventsForTrend(events, source) {
  const safeEvents = Array.isArray(events) ? events : [];
  return safeEvents.map((event, index) => {
    return {
      event,
      source,
      order: index,
      time: safeString(event.time || event.start_time),
      timeMs: parseEventTimeMs(event),
    };
  });
}

function isVitalEvent(normalizedEvent) {
  const event = normalizedEvent && normalizedEvent.event ? normalizedEvent.event : {};
  const code = normalizeLabel(event.code);
  if (!code) {
    return false;
  }
  return code === "vitals" || code.includes("vital");
}

function extractVitalLabel(normalizedEvent) {
  const event = normalizedEvent && normalizedEvent.event ? normalizedEvent.event : {};
  const direct = safeString(event.code_specifics);
  if (direct) {
    return direct;
  }

  const fallback = safeString(event.text_value);
  if (!fallback) {
    return "";
  }
  const firstLine = fallback.split("\n")[0] || "";
  return safeString(firstLine);
}

function labelMatchesAnyPattern(label, patterns) {
  const normalized = normalizeLabel(label);
  if (!normalized || !Array.isArray(patterns) || patterns.length === 0) {
    return false;
  }
  return patterns.some((pattern) => pattern instanceof RegExp && pattern.test(normalized));
}

function extractUnitFromVitalLabel(label) {
  const text = safeString(label);
  if (!text) {
    return "";
  }

  const commaParts = text.split(",").map((part) => safeString(part)).filter(Boolean);
  if (commaParts.length > 1) {
    return commaParts[commaParts.length - 1];
  }

  const parenMatch = text.match(/\(([^)]+)\)\s*$/);
  if (parenMatch && safeString(parenMatch[1])) {
    return safeString(parenMatch[1]);
  }

  return "";
}

function resolveVitalGuideline(label) {
  const normalized = normalizeLabel(label);
  if (!normalized) {
    return null;
  }
  for (const rule of VITAL_GUIDELINE_RULES) {
    if (rule.pattern.test(normalized)) {
      return {
        low: toNumber(rule.low),
        high: toNumber(rule.high),
        unit: safeString(rule.unit),
      };
    }
  }
  return null;
}

function buildMetricKeyFromLabel(label) {
  const normalized = normalizeLabel(label).replace(/[^a-z0-9]+/g, "_");
  return normalized.replace(/^_+|_+$/g, "") || "vital";
}

function buildVitalMetricConfig(label) {
  const originalLabel = safeString(label) || "Vital";
  const guideline = resolveVitalGuideline(originalLabel);
  const inferredUnit = extractUnitFromVitalLabel(originalLabel);
  const unit = safeString((guideline && guideline.unit) || inferredUnit);
  const escapedUnit = unit ? escapeRegex(unit) : "";
  const title = escapedUnit
    ? safeString(originalLabel.replace(new RegExp(`,\\s*${escapedUnit}$`, "i"), "")) || originalLabel
    : originalLabel;

  return {
    title,
    unit,
    healthy_range:
      guideline && Number.isFinite(guideline.low) && Number.isFinite(guideline.high)
        ? [guideline.low, guideline.high]
        : null,
  };
}

function rankVitalSeries(entry) {
  const points = [...entry.history, ...entry.current];
  const values = points.map((point) => point.y).filter((value) => Number.isFinite(value));
  const uniqueTimes = new Set(
    points.map((point) => {
      if (Number.isFinite(point.timeMs)) {
        return `t:${point.timeMs}`;
      }
      return `o:${safeString(point.source) || "na"}:${point.order}`;
    })
  ).size;
  const valueSpan = values.length ? Math.max(...values) - Math.min(...values) : 0;
  return {
    score: points.length * 10 + uniqueTimes + (valueSpan > 0 ? 1 : 0),
    totalPoints: points.length,
    currentPoints: entry.current.length,
    uniqueTimes,
  };
}

function extractEventNumericValue(normalizedEvent) {
  const event = normalizedEvent && normalizedEvent.event ? normalizedEvent.event : {};
  const direct = toNumber(event.numeric_value);
  if (Number.isFinite(direct)) {
    return direct;
  }

  const textCandidates = [event.text_value, event.code_specifics, event.raw_line];
  for (const candidate of textCandidates) {
    const parsed = extractFirstNumericFromText(candidate);
    if (Number.isFinite(parsed)) {
      return parsed;
    }
  }
  return Number.NaN;
}

function extractFirstNumericFromText(value) {
  const text = safeString(value);
  if (!text) {
    return Number.NaN;
  }
  const match = text.match(/-?\d+(?:\.\d+)?/);
  if (!match) {
    return Number.NaN;
  }
  const parsed = Number.parseFloat(match[0]);
  return Number.isFinite(parsed) ? parsed : Number.NaN;
}

function compareTrendPoints(a, b) {
  const aHasTime = Number.isFinite(a.timeMs);
  const bHasTime = Number.isFinite(b.timeMs);
  if (aHasTime && bHasTime && a.timeMs !== b.timeMs) {
    return a.timeMs - b.timeMs;
  }
  if (aHasTime && !bHasTime) {
    return -1;
  }
  if (!aHasTime && bHasTime) {
    return 1;
  }
  const orderDelta = (a.order || 0) - (b.order || 0);
  if (orderDelta !== 0) {
    return orderDelta;
  }
  const sourceRank = (item) => (item && item.source === "current" ? 1 : 0);
  return sourceRank(a) - sourceRank(b);
}

function parseEventTimeMs(event) {
  const candidate = safeString(event && (event.time || event.start_time));
  if (!candidate) {
    return null;
  }

  const normalized = candidate.includes("T") ? candidate : candidate.replace(" ", "T");
  const parsed = Date.parse(normalized);
  if (!Number.isNaN(parsed)) {
    return parsed;
  }
  return null;
}

function formatHistoryEventLine(event) {
  const time = formatTime(event.time || event.start_time || "");
  const code = safeString(event.code) || "UNKNOWN";
  const description = safeString(event.code_specifics) || safeString(event.text_value) || "-";
  const value = formatEventValue(event);
  const valuePart = value ? ` = ${value}` : "";
  return `${time} ${code} ${description}${valuePart}`;
}

function buildVitalSeries(historyEvents, currentEvents) {
  const seriesByLabel = new Map();
  const normalizedHistory = normalizeEventsForTrend(historyEvents, "history");
  const normalizedCurrent = normalizeEventsForTrend(currentEvents, "current");

  const addEventToSeries = (normalizedEvent) => {
    if (!isVitalEvent(normalizedEvent)) {
      return;
    }
    const label = extractVitalLabel(normalizedEvent);
    if (!label) {
      return;
    }
    const numeric = extractEventNumericValue(normalizedEvent);
    if (!Number.isFinite(numeric)) {
      return;
    }

    const labelKey = normalizeLabel(label);
    if (!seriesByLabel.has(labelKey)) {
      seriesByLabel.set(labelKey, {
        label,
        labelKey,
        metricKey: buildMetricKeyFromLabel(label),
        history: [],
        current: [],
      });
    }
    const seriesEntry = seriesByLabel.get(labelKey);
    const point = {
      y: numeric,
      timeMs: normalizedEvent.timeMs,
      order: normalizedEvent.order,
      label: formatTime(normalizedEvent.time),
      source: normalizedEvent.source,
    };

    if (normalizedEvent.source === "current") {
      seriesEntry.current.push(point);
    } else {
      seriesEntry.history.push(point);
    }
  };

  normalizedHistory.forEach(addEventToSeries);
  normalizedCurrent.forEach(addEventToSeries);

  const allWithData = Array.from(seriesByLabel.values()).filter((entry) => entry.current.length > 0 || entry.history.length > 0);
  if (allWithData.length === 0) {
    return [];
  }

  const selected = allWithData.filter((entry) => shouldPlotVitalForStayFrequency(entry.labelKey));
  if (selected.length === 0) {
    return [];
  }

  selected.forEach((entry) => {
    entry.history.sort(compareTrendPoints);
    entry.current.sort(compareTrendPoints);
  });

  const ranked = selected
    .map((entry) => ({
      entry,
      rank: rankVitalSeries(entry),
    }))
    .sort((a, b) => {
      if (a.rank.score !== b.rank.score) {
        return b.rank.score - a.rank.score;
      }
      if (a.rank.currentPoints !== b.rank.currentPoints) {
        return b.rank.currentPoints - a.rank.currentPoints;
      }
      if (a.rank.totalPoints !== b.rank.totalPoints) {
        return b.rank.totalPoints - a.rank.totalPoints;
      }
      if (a.rank.uniqueTimes !== b.rank.uniqueTimes) {
        return b.rank.uniqueTimes - a.rank.uniqueTimes;
      }
      const aPreferred = labelMatchesAnyPattern(a.entry.label, PLOTTABLE_VITAL_LABEL_PATTERNS) ? 1 : 0;
      const bPreferred = labelMatchesAnyPattern(b.entry.label, PLOTTABLE_VITAL_LABEL_PATTERNS) ? 1 : 0;
      if (aPreferred !== bPreferred) {
        return bPreferred - aPreferred;
      }
      return a.entry.label.localeCompare(b.entry.label, undefined, { sensitivity: "base" });
    });

  return ranked.map((item) => {
    return {
      metricKey: item.entry.metricKey,
      metricConfig: buildVitalMetricConfig(item.entry.label),
      history: item.entry.history,
      current: item.entry.current,
    };
  });
}

function filterEventsByType(events, filterType) {
  if (filterType === "all") {
    return events;
  }
  return events.filter((event) => classifyEventType(event) === filterType);
}

function getEventTypeCounts(events) {
  const counts = {
    all: events.length,
    vital: 0,
    medication: 0,
    procedure: 0,
    neuro: 0,
    vent: 0,
    lab: 0,
    disposition: 0,
  };
  events.forEach((event) => {
    const type = classifyEventType(event);
    if (counts.hasOwnProperty(type)) {
      counts[type]++;
    }
  });
  return counts;
}

function renderEventTypeFilter({ counts, scope = "current", compact = false }) {
  const filterOptions = [
    { value: "all", label: "All", count: counts.all },
    { value: "vital", label: "Vitals", count: counts.vital },
    { value: "medication", label: "Medications", count: counts.medication },
    { value: "procedure", label: "Procedures", count: counts.procedure },
    { value: "lab", label: "Labs", count: counts.lab },
    { value: "neuro", label: "Neuro", count: counts.neuro },
    { value: "vent", label: "Ventilation", count: counts.vent },
    { value: "disposition", label: "Disposition", count: counts.disposition },
  ];

  const buttons = filterOptions
    .filter((opt) => opt.value === "all" || opt.count > 0)
    .map((opt) => {
      const isActive = resolveFilterValue(scope) === opt.value;
      const activeClass = isActive ? "active" : "";
      return (
        `<button class="filter-btn ${activeClass}" data-filter="${opt.value}" data-scope="${scope}" type="button">`
        + `${opt.label} (${opt.count})`
        + `</button>`
      );
    })
    .join("");

  const compactClass = compact ? "compact" : "";
  return (
    `<div class="event-filter-bar ${compactClass}" data-scope="${scope}">`
    + `<span class="event-filter-label">Filter</span>`
    + buttons
    + `</div>`
  );
}

function attachEventFilterListeners() {
  const filterButtons = document.querySelectorAll(".filter-btn");
  filterButtons.forEach((btn) => {
    btn.addEventListener("click", (event) => {
      const button = event.currentTarget;
      const filterType = safeString(button.getAttribute("data-filter")) || "all";
      const scope = safeString(button.getAttribute("data-scope")) || "current";
      if (scope === "history") {
        state.historyEventTypeFilter = filterType;
      } else {
        state.currentEventTypeFilter = filterType;
      }
      renderWorkspace();
    });
  });
}

function resolveFilterValue(scope) {
  if (scope === "history") {
    return state.historyEventTypeFilter;
  }
  return state.currentEventTypeFilter;
}

function classifyEventType(event) {
  const code = normalizeLabel(event.code || "");
  const specifics = normalizeLabel(event.code_specifics || "");
  const textValue = normalizeLabel(event.text_value || "");
  const combined = `${code} ${specifics} ${textValue}`;

  if (combined.includes("vital")) {
    if (specifics.includes("gcs") || specifics.includes("rass") || specifics.includes("mental")) {
      return "neuro";
    }
    if (combined.includes("fio2") || combined.includes("peep") || combined.includes("vent")) {
      return "vent";
    }
    return "vital";
  }

  if (combined.includes("drug") || combined.includes("med") || combined.includes("iv")) {
    return "medication";
  }

  if (
    combined.includes("procedure")
    || combined.includes("line")
    || combined.includes("catheter")
    || combined.includes("intub")
    || combined.includes("bronch")
    || combined.includes("dialysis")
  ) {
    return "procedure";
  }

  if (combined.includes("lab") || combined.includes("report") || combined.includes("exam")) {
    return "lab";
  }

  if (
    combined.includes("transfer")
    || combined.includes("enter")
    || combined.includes("leave")
    || combined.includes("service")
  ) {
    return "disposition";
  }

  return "lab";
}

function formatEventSource(source) {
  switch (safeString(source)) {
    case "raw_current_events":
      return "oracle_predictions.raw_current_events";
    case "window_contexts.current_events":
      return "window_contexts.current_events";
    case "window_metadata.current_events":
      return "window_metadata.current_events";
    default:
      return "none";
  }
}

function actionSeverityRank(label) {
  const normalized = normalizeLabel(label);
  if (normalized === "potentially_harmful") {
    return 0;
  }
  if (normalized === "suboptimal") {
    return 1;
  }
  if (normalized === "appropriate") {
    return 2;
  }
  return 3;
}

function overallToneClass(label) {
  const normalized = normalizeLabel(label);
  if (normalized === "improving" || normalized === "appropriate") {
    return "positive";
  }
  if (normalized === "deteriorating" || normalized === "potentially_harmful") {
    return "danger";
  }
  if (normalized === "suboptimal") {
    return "warning";
  }
  return "neutral";
}

function buildWindowKey(subjectId, icuStayId, windowIndex) {
  const subject = safeString(subjectId) || "unknown_subject";
  const stay = safeString(icuStayId) || "unknown_stay";
  return `${subject}_${stay}_w${windowIndex}`;
}

function formatWindowRange(metadata) {
  const start = safeString(metadata.window_start_time);
  const end = safeString(metadata.window_end_time);
  if (!start && !end) {
    return "-";
  }
  return `${formatTime(start)} - ${formatTime(end)}`;
}

function formatTime(value) {
  if (!value) {
    return "-";
  }
  const asString = safeString(value);
  const tsMatch = asString.match(/(\d{2}:\d{2})(?::\d{2})?/);
  if (tsMatch) {
    return tsMatch[1];
  }

  const parsed = new Date(asString);
  if (Number.isNaN(parsed.getTime())) {
    return asString;
  }

  const hh = String(parsed.getHours()).padStart(2, "0");
  const mm = String(parsed.getMinutes()).padStart(2, "0");
  return `${hh}:${mm}`;
}

function formatEventDescription(event) {
  const parts = [];
  const specifics = safeString(event.code_specifics);
  const textValue = safeString(event.text_value);

  if (specifics) {
    parts.push(specifics);
  }
  if (!specifics && textValue) {
    parts.push(textValue);
  }
  if (parts.length === 0) {
    parts.push("-");
  }
  return parts.join(" | ");
}

function formatEventValue(event) {
  const numeric = toNumber(event.numeric_value);
  if (Number.isFinite(numeric)) {
    return numeric.toFixed(2);
  }
  return safeString(event.text_value) || "";
}

function formatRecommendationRank(rec) {
  if (rec.rank === null) {
    return `(unranked, pos ${rec.position_index + 1})`;
  }
  return `#${rec.rank}`;
}

function formatRecommendationId(rec) {
  if (rec.rank === null) {
    return `rank:null@pos:${rec.position_index}`;
  }
  return `rank:${rec.rank}`;
}

function formatNumber(value) {
  const number = toNumber(value);
  return Number.isFinite(number) ? number.toFixed(2) : "-";
}

function formatCompactNumber(value) {
  const number = toNumber(value);
  if (!Number.isFinite(number)) {
    return "-";
  }
  const rounded = Math.round(number * 10) / 10;
  if (Number.isInteger(rounded)) {
    return String(rounded);
  }
  return rounded.toFixed(1);
}

function getIcuOutcomeInfo(predictionsData) {
  const trajectoryMetadata = isObject(predictionsData && predictionsData.trajectory_metadata)
    ? predictionsData.trajectory_metadata
    : {};
  const survived = trajectoryMetadata.survived;
  if (survived === true) {
    return { label: "Survived after ICU", toneClass: "survived" };
  }
  if (survived === false) {
    return { label: "Died after ICU", toneClass: "died" };
  }

  const normalized = normalizeLabel(survived);
  if (["true", "1", "yes", "y", "survived", "alive"].includes(normalized)) {
    return { label: "Survived after ICU", toneClass: "survived" };
  }
  if (["false", "0", "no", "n", "died", "dead", "deceased"].includes(normalized)) {
    return { label: "Died after ICU", toneClass: "died" };
  }

  if (safeString(trajectoryMetadata.death_time)) {
    return { label: "Died after ICU", toneClass: "died" };
  }
  return { label: "Unknown", toneClass: "unknown" };
}

function setLoaderMessage(text, type) {
  loaderMessageEl.className = `message${type ? ` ${type}` : ""}`;
  loaderMessageEl.textContent = text;
}

function setGlobalMessage(text, type) {
  state.globalMessage = { text, type };
  renderGlobalMessage();
}

function renderGlobalMessage() {
  globalMessageEl.className = `message${state.globalMessage.type ? ` ${state.globalMessage.type}` : ""}`;
  globalMessageEl.textContent = state.globalMessage.text || "";
}

function getCurrentRound() {
  return state.rounds[state.currentRoundIndex];
}

function getCurrentAnnotation() {
  const round = getCurrentRound();
  return state.annotations[round.arrayIndex];
}

function hasCurrentHighlight(token) {
  const round = getCurrentRound();
  const highlights = state.errorHighlights[round.arrayIndex];
  if (!highlights) {
    return false;
  }
  return highlights.has(token);
}

function setErrorHighlightsFromBlockingDetails(blockingDetails) {
  const nextHighlights = {};
  blockingDetails.forEach((detail) => {
    if (!nextHighlights[detail.arrayIndex]) {
      nextHighlights[detail.arrayIndex] = new Set();
    }
    nextHighlights[detail.arrayIndex].add(detail.token);
  });
  state.errorHighlights = nextHighlights;
}

function clearCurrentRoundErrorHighlights() {
  const round = getCurrentRound();
  if (!round) {
    return;
  }
  delete state.errorHighlights[round.arrayIndex];
}

function onWorkspaceError(error) {
  setGlobalMessage(error.message || "Unexpected annotation error.", "warn");
}

function readJsonFile(file) {
  return new Promise((resolve, reject) => {
    const reader = new FileReader();
    reader.onload = () => {
      try {
        const parsed = JSON.parse(String(reader.result));
        resolve(parsed);
      } catch (error) {
        reject(new Error(`Invalid JSON in ${file.name}: ${error.message}`));
      }
    };
    reader.onerror = () => {
      reject(new Error(`Unable to read ${file.name}`));
    };
    reader.readAsText(file);
  });
}

function downloadJson(payload, fileName) {
  const json = JSON.stringify(payload, null, 2);
  const blob = new Blob([json], { type: "application/json" });
  const url = URL.createObjectURL(blob);

  const anchor = document.createElement("a");
  anchor.href = url;
  anchor.download = fileName;
  document.body.appendChild(anchor);
  anchor.click();
  document.body.removeChild(anchor);

  setTimeout(() => {
    URL.revokeObjectURL(url);
  }, 500);
}

function toggleSelection(items, item, shouldInclude) {
  const idx = items.indexOf(item);
  if (shouldInclude && idx === -1) {
    items.push(item);
    return;
  }
  if (!shouldInclude && idx >= 0) {
    items.splice(idx, 1);
  }
}

function toInt(value, defaultValue) {
  if (value === null || value === undefined || value === "") {
    return defaultValue;
  }
  const n = Number.parseInt(String(value), 10);
  return Number.isNaN(n) ? defaultValue : n;
}

function toIntOrNull(value) {
  const n = Number.parseInt(String(value), 10);
  return Number.isNaN(n) ? null : n;
}

function toNumber(value) {
  const n = Number.parseFloat(String(value));
  return Number.isNaN(n) ? Number.NaN : n;
}

function safeString(value) {
  if (value === null || value === undefined) {
    return "";
  }
  return String(value).trim();
}

function normalizeLabel(value) {
  return safeString(value).toLowerCase();
}

function escapeRegex(value) {
  return safeString(value).replace(/[.*+?^${}()|[\]\\]/g, "\\$&");
}

function clamp(value, min, max) {
  return Math.min(Math.max(value, min), max);
}

function cloneData(value) {
  return JSON.parse(JSON.stringify(value));
}

function isObject(value) {
  return value !== null && typeof value === "object" && !Array.isArray(value);
}

function escapeHtml(value) {
  return safeString(value)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/\"/g, "&quot;")
    .replace(/'/g, "&#39;");
}

window.addEventListener("error", (event) => {
  onWorkspaceError(event.error || new Error(event.message));
});

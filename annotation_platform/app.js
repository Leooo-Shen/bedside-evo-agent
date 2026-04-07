"use strict";

const REQUIRED_ORACLE_OUTPUT_KEYS = [
  "patient_assessment",
  "action_review",
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
const EVENT_TYPE_OPTIONS = [
  { value: "vital", label: "Vitals" },
  { value: "medication", label: "Medications" },
  { value: "procedure", label: "Procedures" },
  { value: "lab", label: "Labs" },
  { value: "note", label: "Notes" },
  { value: "diagnosis", label: "Diagnosis" },
  { value: "neuro", label: "Neuro" },
  { value: "vent", label: "Ventilation" },
  { value: "disposition", label: "Disposition" },
  { value: "other", label: "Other" },
];
const EVENT_CODE_TYPE_MAP = {
  VITALS: "vital",
  BODY_INPUT: "vital",
  BODY_OUTPUT: "vital",
  DRUG_START: "medication",
  DRUG_STOP: "medication",
  DRUG_PRESCRIPTION: "medication",
  DRUG_PREVIOUSLY: "medication",
  PROCEDURE: "procedure",
  ECG: "procedure",
  ECHO: "procedure",
  XRAY: "procedure",
  LAB_TEST: "lab",
  MICROBIOLOGY_TEST: "lab",
  DIAGNOSIS: "diagnosis",
  COMPLAINT: "diagnosis",
  TRANSFER: "disposition",
  SERVICE: "disposition",
  ENTER_HOSPITALIZATION: "disposition",
  LEAVE_HOSPITALIZATION: "disposition",
  ENTER_ED: "disposition",
  LEAVE_ED: "disposition",
  ENTER_ICU: "disposition",
  LEAVE_ICU: "disposition",
  OTHER_EVENT: "other",
};
const VENT_EVENT_PATTERN = /\b(fio2|peep|vent(?:ilator|ilation)?|tidal volume|minute volume|etco2|airway pressure|intubat(?:ion|ed|e)?|extubat(?:ion|ed|e)?)\b/i;
const NEURO_EVENT_PATTERN = /\b(gcs|rass|cam-icu|mental status|neurolog(?:y|ic)|pupil(?:s)?|delirium|sedation score)\b/i;
const VITAL_EVENT_PATTERN = /\b(vital|heart rate|blood pressure|resp(?:iratory)? rate|spo2|oxygen saturation|temperature|mean arterial pressure|map)\b/i;
const MEDICATION_EVENT_PATTERN = /\b(drug|medication|infusion|vasopressor|antibiotic|analgesic|sedat(?:ion|ive)|insulin)\b/i;
const PROCEDURE_EVENT_PATTERN = /\b(procedure|line|catheter|bronch|dialysis|ecg|echo|xray|ultrasound|ct scan|mri)\b/i;
const LAB_EVENT_PATTERN = /\b(lab(?:_test)?|microbiology|culture|blood gas|electrolyte|creatinine|lactate|hemoglobin|report|exam)\b/i;
const DIAGNOSIS_EVENT_PATTERN = /\b(diagnosis|diagnoses|dx|problem list|complaint|condition)\b/i;
const DISPOSITION_EVENT_PATTERN = /\b(transfer|enter|leave|admit|admission|discharge|service)\b/i;

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
const ACTION_CORRECTION_OPTIONS = ["best_practice", "acceptable", "potentially_harmful", "insufficient_data"];
const ACTION_URGENCY_OPTIONS = [
  { value: "immediate", label: "immediate" },
  { value: "within_1h", label: "within_1h" },
  { value: "within_6h", label: "within_6h" },
  { value: "not_sure", label: "not sure" },
];
const RECOMMENDATION_AGREED_URGENCY_OPTIONS = [
  { value: "urgent", label: "Urgent" },
  { value: "nice_to_have", label: "Nice to have" },
  { value: "optional", label: "Optional" },
];
const RECOMMENDATION_AGREED_DOSE_OPTIONS = [
  { value: "low", label: "low" },
  { value: "standard", label: "standard" },
  { value: "slightly_high", label: "slightly high" },
  { value: "very_high", label: "very high" },
];
const FREE_TEXT_CONFIDENCE_OPTIONS = [
  { value: "high", label: "high" },
  { value: "medium", label: "medium" },
  { value: "low", label: "low" },
];
const ICU_ACTION_CATEGORIES = [
  "Observation / No Immediate Action",
  "Further Assessment & Diagnostics",
  "Respiratory Support",
  "Hemodynamic & Fluid Management",
  "Infection & Antimicrobial Management",
  "Sedation, Analgesia & Neurological Management",
  "Renal Support",
  "Metabolic & Nutritional Management",
  "Transfusion & Blood Products",
  "Procedures & Surgical Interventions",
  "Preventive Care",
  "Escalation & Goals of Care",
  "Others",
];
const MAX_ADDITIONAL_ACTION_EVIDENCE_IDS = 5;
const VITAL_TREND_RELATIVE_CHANGE_THRESHOLD = 0.25;
const VITAL_TREND_BASELINE_RANGE_FRACTION_THRESHOLD = 0.15;
const VITAL_TREND_BASELINE_FRACTION_THRESHOLD = 0.2;
const VERDICT_VALUES = new Set(["agree", "disagree", "uncertain"]);
const STATUS_CORRECTION_VALUES = new Set(STATUS_CORRECTION_OPTIONS);
const ACTION_CORRECTION_VALUES = new Set(ACTION_CORRECTION_OPTIONS);
const ACTION_URGENCY_VALUES = new Set(ACTION_URGENCY_OPTIONS.map((option) => option.value));
const RECOMMENDATION_URGENCY_VALUES = new Set(RECOMMENDATION_AGREED_URGENCY_OPTIONS.map((option) => option.value));
const RECOMMENDATION_DOSE_VALUES = new Set(RECOMMENDATION_AGREED_DOSE_OPTIONS.map((option) => option.value));
const AUTO_SAVE_DEBOUNCE_MS = 900;
const DRAFT_STORAGE_PREFIX = "oracle_annotation_draft_v1";

const state = {
  loaded: false,
  predictionsData: null,
  windowContextsData: null,
  fullWindowContextsData: null,
  sourceFiles: {
    oracle_predictions: null,
    window_contexts: null,
    full_window_contexts: null,
  },
  rounds: [],
  invalidWindows: [],
  annotations: {},
  currentRoundIndex: 0,
  currentEventTypeFilter: "all",
  historyEventTypeFilter: "all",
  loadedAtMs: 0,
  draftKey: null,
  autoSaveTimerId: null,
  autoSaveErrorShown: false,
  errorHighlights: {},
  globalMessage: {
    type: "",
    text: "",
  },
  vitalPlotStats: {
    totalWindows: 0,
    byLabel: new Map(),
  },
  dischargeSummaryOpenSectionsByRound: {},
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
let draftFileInputEl = null;

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
  draftFileInputEl = document.getElementById("draft-file-input");

  document.getElementById("load-folder-btn").addEventListener("click", onLoadFolder);
  document.getElementById("import-draft-btn").addEventListener("click", onImportDraftClick);
  document.getElementById("export-draft-btn").addEventListener("click", onExportDraft);
  document.getElementById("export-btn").addEventListener("click", onExport);
  prevWindowBtn.addEventListener("click", () => stepWindow(-1));
  nextWindowBtn.addEventListener("click", () => stepWindow(1));
  windowSelectEl.addEventListener("change", onWindowSelectChange);
  draftFileInputEl.addEventListener("change", onDraftFileSelected);

  workspaceView.addEventListener("click", onWorkspaceClick);
  workspaceView.addEventListener("change", onWorkspaceChange);
  workspaceView.addEventListener("input", onWorkspaceInput);
  window.addEventListener("beforeunload", () => {
    flushAutoSaveNow();
  });
});

async function onLoadFolder() {
  const folderInput = document.getElementById("case-folder-file");
  const selectedFiles = folderInput && folderInput.files ? Array.from(folderInput.files) : [];
  if (selectedFiles.length === 0) {
    setLoaderMessage("Please select a case folder first.", "warn");
    return;
  }

  const predictionsFile = resolvePredictionsFileFromFolder(selectedFiles);
  if (!predictionsFile) {
    setLoaderMessage(
      "Could not find oracle_predictions.json (or *_annotated.json) in selected folder.",
      "warn"
    );
    return;
  }

  const windowContextFile = resolveWindowContextsFileFromFolder(selectedFiles, predictionsFile);
  const fullWindowContextFile = resolveFullWindowContextsFileFromFolder(selectedFiles, predictionsFile);
  await loadSessionFromFiles({
    predictionsFile,
    windowContextFile,
    fullWindowContextFile,
    loadingMessage: "Loading case folder...",
  });
}

async function loadSessionFromFiles({
  predictionsFile,
  windowContextFile = null,
  fullWindowContextFile = null,
  loadingMessage = "Loading files...",
}) {
  try {
    setLoaderMessage(loadingMessage, "info");
    const predictionsData = await readJsonFile(predictionsFile);
    const windowContextsData = windowContextFile ? await readJsonFile(windowContextFile) : null;
    const fullWindowContextsData = fullWindowContextFile ? await readJsonFile(fullWindowContextFile) : null;

    initializeSession({
      predictionsData,
      windowContextsData,
      fullWindowContextsData,
      sourceFiles: {
        oracle_predictions: fileDisplayName(predictionsFile),
        window_contexts: windowContextFile ? fileDisplayName(windowContextFile) : null,
        full_window_contexts: fullWindowContextFile ? fileDisplayName(fullWindowContextFile) : null,
      },
    });
    setLoaderMessage("", "");
  } catch (error) {
    setLoaderMessage(`Failed to load JSON: ${error.message}`, "warn");
  }
}

function resolvePredictionsFileFromFolder(files) {
  const allFiles = Array.isArray(files) ? files : [];
  if (allFiles.length === 0) {
    return null;
  }

  const exact = allFiles.filter((file) => safeString(file.name) === "oracle_predictions.json");
  if (exact.length > 0) {
    return pickBestRelativePathFile(exact);
  }

  const annotated = allFiles.filter((file) => /_annotated\.json$/i.test(safeString(file.name)));
  if (annotated.length > 0) {
    return pickBestRelativePathFile(annotated);
  }
  return null;
}

function resolveWindowContextsFileFromFolder(files, predictionsFile) {
  const allFiles = Array.isArray(files) ? files : [];
  const candidates = allFiles.filter((file) => safeString(file.name) === "window_contexts.json");
  if (candidates.length === 0) {
    return null;
  }
  if (candidates.length === 1) {
    return candidates[0];
  }

  const predictionsParent = fileParentPath(predictionsFile);
  if (!predictionsParent) {
    return pickBestRelativePathFile(candidates);
  }

  const sameParent = candidates.find((file) => fileParentPath(file) === predictionsParent);
  return sameParent || pickBestRelativePathFile(candidates);
}

function resolveFullWindowContextsFileFromFolder(files, predictionsFile) {
  const allFiles = Array.isArray(files) ? files : [];
  const candidates = allFiles.filter((file) => safeString(file.name) === "full_window_contexts.json");
  if (candidates.length === 0) {
    return null;
  }
  if (candidates.length === 1) {
    return candidates[0];
  }

  const predictionsParent = fileParentPath(predictionsFile);
  if (!predictionsParent) {
    return pickBestRelativePathFile(candidates);
  }

  const sameParent = candidates.find((file) => fileParentPath(file) === predictionsParent);
  return sameParent || pickBestRelativePathFile(candidates);
}

function fileParentPath(file) {
  const relative = fileDisplayName(file);
  const slashIndex = relative.lastIndexOf("/");
  return slashIndex >= 0 ? relative.slice(0, slashIndex) : "";
}

function fileDisplayName(file) {
  if (!file) {
    return "";
  }
  return safeString(file.webkitRelativePath) || safeString(file.name);
}

function pickBestRelativePathFile(files) {
  const safeFiles = Array.isArray(files) ? files.slice() : [];
  if (safeFiles.length === 0) {
    return null;
  }
  safeFiles.sort((a, b) => fileDisplayName(a).localeCompare(fileDisplayName(b), undefined, { sensitivity: "base" }));
  return safeFiles[0];
}

function initializeSession({ predictionsData, windowContextsData, fullWindowContextsData, sourceFiles }) {
  if (!isObject(predictionsData)) {
    throw new Error("oracle_predictions.json must contain a top-level JSON object.");
  }
  if (!Array.isArray(predictionsData.window_outputs)) {
    throw new Error("oracle_predictions.json is missing window_outputs[].");
  }

  flushAutoSaveNow();
  clearAutoSaveTimer();
  const windowContextByIndex = buildWindowContextByIndex(windowContextsData);

  const rounds = [];
  const invalidWindows = [];
  const annotations = {};
  let importedAnnotationCount = 0;

  predictionsData.window_outputs.forEach((windowOutput, arrayIndex) => {
    const windowIndex = toInt(windowOutput && windowOutput.window_index, arrayIndex + 1);
    const metadata = isObject(windowOutput && windowOutput.window_metadata) ? windowOutput.window_metadata : {};
    const sourceWindowIndex = toInt(
      windowOutput && windowOutput.source_window_index,
      toInt(metadata.source_window_index, windowIndex - 1)
    );
    const sourceWindowPosition = toInt(
      windowOutput && windowOutput.source_window_position,
      toInt(metadata.source_window_position, sourceWindowIndex + 1)
    );
    const validation = validateWindowOutput(windowOutput);

    if (!validation.valid) {
      invalidWindows.push({
        array_index: arrayIndex,
        window_index: windowIndex,
        reasons: validation.reasons,
      });
      return;
    }

    const actionReview = isObject(windowOutput.oracle_output.action_review) ? windowOutput.oracle_output.action_review : {};
    const normalizedActions = normalizeActions(actionReview.evaluations);
    const normalizedRecommendations = normalizeRecommendations(actionReview.red_flags);
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
      sourceWindowIndex,
      sourceWindowPosition,
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
    const rawImportedAnnotation = isObject(windowOutput.annotation) ? windowOutput.annotation : null;
    if (rawImportedAnnotation) {
      importedAnnotationCount += 1;
    }
    annotations[arrayIndex] = normalizeAnnotationForRound(round, rawImportedAnnotation);
  });

  if (rounds.length === 0) {
    throw new Error("No valid windows found in this file.");
  }

  state.loaded = true;
  state.predictionsData = predictionsData;
  state.windowContextsData = windowContextsData;
  state.fullWindowContextsData = fullWindowContextsData;
  state.sourceFiles = sourceFiles;
  state.rounds = rounds;
  state.invalidWindows = invalidWindows;
  state.annotations = annotations;
  state.currentRoundIndex = 0;
  state.currentEventTypeFilter = "all";
  state.historyEventTypeFilter = "all";
  state.loadedAtMs = Date.now();
  state.draftKey = buildDraftStorageKey(predictionsData);
  state.autoSaveErrorShown = false;
  state.errorHighlights = {};
  state.vitalPlotStats = buildStayVitalPlotStats(rounds, fullWindowContextsData);
  state.dischargeSummaryOpenSectionsByRound = {};
  state.currentRoundIndex = 0;

  let restoredDraftCount = 0;
  if (importedAnnotationCount === 0) {
    restoredDraftCount = restoreDraftFromLocalStorage();
  }
  persistDraftToLocalStorage();

  const statusMessages = [];
  statusMessages.push(invalidWindows.length
    ? `Loaded ${rounds.length} valid windows; skipped ${invalidWindows.length} invalid window(s).`
    : `Loaded ${rounds.length} valid windows.`);
  if (importedAnnotationCount > 0) {
    statusMessages.push(`Recovered existing annotations from file for ${importedAnnotationCount} window(s).`);
  }
  if (restoredDraftCount > 0) {
    statusMessages.push(`Auto-restored local draft for ${restoredDraftCount} window(s).`);
  }
  if (fullWindowContextsData && Array.isArray(fullWindowContextsData.window_contexts)) {
    statusMessages.push(`Loaded full timeline contexts (${fullWindowContextsData.window_contexts.length} windows).`);
  }
  setGlobalMessage(statusMessages.join(" "), invalidWindows.length ? "warn" : "info");

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

  if (!isObject(oracleOutput.patient_assessment)) {
    reasons.push("oracle_output.patient_assessment is not an object");
  }

  const overall = isObject(oracleOutput.patient_assessment) ? oracleOutput.patient_assessment.overall : null;
  if (!isObject(overall)) {
    reasons.push("oracle_output.patient_assessment.overall is not an object");
  } else {
    if (!safeString(overall.label)) {
      reasons.push("oracle_output.patient_assessment.overall.label missing");
    }
    if (!safeString(overall.rationale)) {
      reasons.push("oracle_output.patient_assessment.overall.rationale missing");
    }
  }

  if (!isObject(oracleOutput.action_review)) {
    reasons.push("oracle_output.action_review is not an object");
  } else {
    if (!Array.isArray(oracleOutput.action_review.evaluations)) {
      reasons.push("oracle_output.action_review.evaluations is not an array");
    }
    if (!Array.isArray(oracleOutput.action_review.red_flags)) {
      reasons.push("oracle_output.action_review.red_flags is not an array");
    }
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

    const actionId = safeString(action.action_id) || `ACTION_${index + 1}`;
    const description = safeString(action.action_name)
      || safeString(action.action)
      || safeString(action.action_description)
      || "No description";
    const overallLabel = normalizeLabel(action.label);
    const overallRationale = safeString(action.rationale);

    return {
      originalIndex: index,
      action_id: actionId,
      action_description: description,
      overall_label: overallLabel || "unknown",
      overall_rationale: overallRationale,
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
    const action = safeString(rec.contraindicated_action)
      || safeString(rec.action)
      || safeString(rec.action_name)
      || safeString(rec.title)
      || "No action title";
    const rationale = safeString(rec.reason)
      || safeString(rec.rationale)
      || safeString(rec.justification)
      || safeString(rec.reasoning)
      || "";
    return {
      originalIndex: index,
      rank: toIntOrNull(rec.rank),
      urgency: safeString(rec.urgency) || "",
      action,
      action_description: rationale,
      rationale,
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
  if (contextEntry && Array.isArray(contextEntry.oracle_context_history_events)) {
    return cloneData(contextEntry.oracle_context_history_events);
  }
  if (contextEntry && Array.isArray(contextEntry.history_events)) {
    return cloneData(contextEntry.history_events);
  }
  const historyEvents = windowOutput && Array.isArray(windowOutput.history_events) ? windowOutput.history_events : [];
  return cloneData(historyEvents);
}

function hasRenderableStructuredHistoryEvents(events) {
  const safeEvents = Array.isArray(events) ? events : [];
  return safeEvents.some((event) => {
    if (!isObject(event)) {
      return false;
    }
    const time = safeString(event.time || event.start_time);
    const code = safeString(event.code);
    const details = safeString(event.code_specifics);
    const eventId = resolveEventEvidenceId(event);
    return Boolean(time || code || details || eventId);
  });
}

function resolveContextSections({ contextEntry }) {
  const empty = {
    sections: {
      icu_discharge_summary: "",
      icu_trajectory_context_window: "",
      previous_events_current_window: "",
      current_observation_window: "",
    },
    source: "none",
  };

  if (contextEntry && isObject(contextEntry.prompt_sections)) {
    return {
      sections: {
        icu_discharge_summary: safeString(contextEntry.prompt_sections.icu_discharge_summary),
        icu_trajectory_context_window: safeString(contextEntry.prompt_sections.icu_trajectory_context_window),
        previous_events_current_window: safeString(contextEntry.prompt_sections.previous_events_current_window),
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

function resolveSourceWindowIndexFromContextEntry(contextEntry, fallbackWindowIndex = null) {
  if (!isObject(contextEntry)) {
    return toInt(fallbackWindowIndex, null);
  }
  const metadata = isObject(contextEntry.window_metadata) ? contextEntry.window_metadata : {};
  return toInt(
    contextEntry.source_window_index,
    toInt(metadata.source_window_index, toInt(fallbackWindowIndex, null))
  );
}

function buildInitialAnnotation(round) {
  return {
    window_key: round.windowKey,
    skip_annotation: false,
    task1_status: {
      selected_label: null,
      selected_risks: [],
      additional_risks: [],
    },
    task2_actions: round.actions.map((action) => ({
      action_id: action.action_id,
      selected_label: null,
      selected_urgency: null,
    })),
    task3_recommendations: round.recommendations.map((rec) => ({
      rank: rec.rank,
      position_index: rec.position_index,
      verdict: null,
      agreed_urgency: null,
      agreed_dose: null,
    })),
    task3_additional_recommendations: {
      actions: [],
    },
    task3_red_flag_actions: {
      selected_actions: [],
      additional_actions: [],
    },
    task4_patient_insights: {
      insights: [],
    },
  };
}

function normalizeAnnotationForRound(round, rawAnnotation) {
  const normalized = buildInitialAnnotation(round);
  if (!isObject(rawAnnotation)) {
    return normalized;
  }

  const skipAnnotation = normalizeNullableBoolean(rawAnnotation.skip_annotation);
  normalized.skip_annotation = skipAnnotation === true;

  const rawTask1 = isObject(rawAnnotation.task1_status) ? rawAnnotation.task1_status : {};
  const activeRiskNames = getOracleActiveRiskNames(round.oracleOutput.patient_assessment);
  const activeRiskNameSet = new Set(activeRiskNames.map((name) => normalizeLabel(name)));
  normalized.task1_status.selected_label = normalizeEnumValue(rawTask1.selected_label, STATUS_CORRECTION_VALUES);
  normalized.task1_status.selected_risks = normalizeRiskNameList(rawTask1.selected_risks)
    .filter((riskName) => activeRiskNameSet.has(normalizeLabel(riskName)));
  normalized.task1_status.additional_risks = normalizeRiskNameList(rawTask1.additional_risks);

  const rawTask2 = Array.isArray(rawAnnotation.task2_actions) ? rawAnnotation.task2_actions : [];
  const rawTask2ByActionId = new Map();
  rawTask2.forEach((entry) => {
    if (!isObject(entry)) {
      return;
    }
    const actionId = safeString(entry.action_id);
    if (actionId) {
      rawTask2ByActionId.set(actionId, entry);
    }
  });
  normalized.task2_actions = normalized.task2_actions.map((entry, index) => {
    const rawEntry = rawTask2ByActionId.get(entry.action_id) || (isObject(rawTask2[index]) ? rawTask2[index] : null);
    if (!rawEntry) {
      return entry;
    }
    const selectedLabel = normalizeEnumValue(rawEntry.selected_label, ACTION_CORRECTION_VALUES);
    const selectedUrgency = normalizeEnumValue(rawEntry.selected_urgency, ACTION_URGENCY_VALUES);
    return {
      ...entry,
      selected_label: selectedLabel,
      selected_urgency: (
        selectedLabel === "best_practice" || selectedLabel === "acceptable"
          ? selectedUrgency
          : null
      ),
    };
  });

  const rawTask3 = Array.isArray(rawAnnotation.task3_recommendations) ? rawAnnotation.task3_recommendations : [];
  normalized.task3_recommendations = normalized.task3_recommendations.map((entry, index) => {
    const rawEntry = isObject(rawTask3[index]) ? rawTask3[index] : null;
    if (!rawEntry) {
      return entry;
    }
    return {
      ...entry,
      verdict: normalizeEnumValue(rawEntry.verdict, VERDICT_VALUES),
      agreed_urgency: normalizeEnumValue(rawEntry.agreed_urgency, RECOMMENDATION_URGENCY_VALUES),
      agreed_dose: normalizeEnumValue(rawEntry.agreed_dose, RECOMMENDATION_DOSE_VALUES),
    };
  });

  const rawAdditional = isObject(rawAnnotation.task3_additional_recommendations)
    ? rawAnnotation.task3_additional_recommendations
    : {};
  normalized.task3_additional_recommendations.actions = getTextConfidenceItems(rawAdditional.actions);

  const rawRedFlags = isObject(rawAnnotation.task3_red_flag_actions) ? rawAnnotation.task3_red_flag_actions : {};
  const oracleRedFlagActions = getOracleContraindicatedActionNames(round.recommendations);
  const oracleRedFlagActionSet = new Set(oracleRedFlagActions.map((action) => normalizeLabel(action)));
  normalized.task3_red_flag_actions.selected_actions = normalizeRedFlagActionList(rawRedFlags.selected_actions)
    .filter((action) => oracleRedFlagActionSet.has(normalizeLabel(action)));
  normalized.task3_red_flag_actions.additional_actions = normalizeRedFlagActionList(rawRedFlags.additional_actions);

  const rawInsights = isObject(rawAnnotation.task4_patient_insights) ? rawAnnotation.task4_patient_insights : {};
  normalized.task4_patient_insights.insights = getTextConfidenceItems(rawInsights.insights);

  return normalized;
}

function normalizeEnumValue(value, allowedValues) {
  const text = safeString(value);
  if (!text) {
    return null;
  }
  return allowedValues instanceof Set && allowedValues.has(text) ? text : null;
}

function normalizeNullableBoolean(value) {
  if (value === true || value === false) {
    return value;
  }
  const normalized = normalizeLabel(value);
  if (["true", "1", "yes", "y"].includes(normalized)) {
    return true;
  }
  if (["false", "0", "no", "n"].includes(normalized)) {
    return false;
  }
  return null;
}

function renderWorkspace() {
  renderTopBar();
  renderLeftPanel();
  renderCenterPanel();
  renderRightPanel();
  attachEventFilterListeners();
  attachDischargeSummarySectionListeners();
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
      const annotation = state.annotations[item.arrayIndex];
      const skipSuffix = annotation && annotation.skip_annotation === true ? " [Skipped]" : "";
      const label = `Window ${item.windowIndex} (${idx + 1}/${state.rounds.length})${skipSuffix}`;
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
  const patientContext = getPatientContextInfo(state.predictionsData, metadata);
  const dischargeSummaryText = safeString(round.contextSections.icu_discharge_summary);
  const promptHistoryView = parsePromptHistorySection(round.contextSections.previous_events_current_window);
  const historyHours = resolveHistoryHours(round, promptHistoryView.historyHours);
  const structuredHistoryEvents = Array.isArray(round.historyEvents) ? round.historyEvents : [];
  const hasStructuredHistoryTimelineEvents = hasRenderableStructuredHistoryEvents(structuredHistoryEvents);
  const hasParsedPromptHistoryEvents = promptHistoryView.hasSection && promptHistoryView.events.length > 0;
  const historyEventsForDisplayBase = hasStructuredHistoryTimelineEvents
    ? structuredHistoryEvents
    : (hasParsedPromptHistoryEvents ? promptHistoryView.events : structuredHistoryEvents);
  const historyEventsForDisplay = withOracleEventIds(historyEventsForDisplayBase);
  const filteredHistoryEvents = filterEventsByType(historyEventsForDisplay, state.historyEventTypeFilter);
  const historyTypeCounts = getEventTypeCounts(historyEventsForDisplay);
  const historyWindowLabel = historyHours !== null
    ? `${formatCompactNumber(historyHours)}-hour`
    : "history";
  const dischargeSummaryHtml = formatDischargeSummaryHtml(dischargeSummaryText, round.arrayIndex);
  const historyEventsHtml = renderTimelineEvents(
    filteredHistoryEvents,
    `<div class="summary-card">No events match this filter in previous ${historyWindowLabel} window.</div>`,
    { scope: "history" }
  );

  leftPanelEl.innerHTML = `
    <h2>Context</h2>
    <div class="summary-card">
      <div><strong>Round</strong>: ${state.currentRoundIndex + 1} / ${state.rounds.length}</div>
      <div><strong>Window index</strong>: ${round.windowIndex}</div>
      <div><strong>Range</strong>: ${escapeHtml(formatWindowRange(metadata))}</div>
      <div><strong>Hours since ICU admit</strong>: ${escapeHtml(formatNumber(metadata.hours_since_admission))}</div>
      <div><strong>Age</strong>: ${escapeHtml(patientContext.age)}</div>
      <div><strong>Gender</strong>: ${escapeHtml(patientContext.gender)}</div>
      <div><strong>Total ICU Stay hours</strong>: ${escapeHtml(patientContext.totalIcuStayHours)}</div>
      <div>
        <strong>ICU outcome</strong>:
        <span class="outcome-text outcome-${escapeHtml(patientContext.icuOutcome.toneClass)}">${escapeHtml(
          patientContext.icuOutcome.label
        )}</span>
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
      ${
        stats.isSkipped
          ? `
        <div class="stat-grid">
          <div class="stat-pill stat-pill-skip">Skipped: this window does not require annotation.</div>
        </div>
      `
          : `
        <div class="stat-grid">
          <div class="stat-pill">Task 1: ${stats.task1Done ? "Reviewed" : "Pending"}</div>
          <div class="stat-pill">Actions: ${stats.reviewedActions}/${stats.totalActions}</div>
          <div class="stat-pill">More recommended actions: ${stats.otherActionsCount}</div>
          <div class="stat-pill">Task 3 (optional): ${stats.task3InsightCount > 0 ? `${stats.task3InsightCount} filled` : "Not filled"}</div>
        </div>
      `
      }
    </div>
  `;
}

function renderCenterPanel() {
  const round = getCurrentRound();
  const currentEvents = Array.isArray(round.events) ? round.events : [];
  const rawEvents = Array.isArray(round.rawEvents) ? round.rawEvents : [];
  const promptCurrentView = parsePromptCurrentSection(round.contextSections.current_observation_window);
  const eventsForDisplayBase = promptCurrentView.hasSection
    ? promptCurrentView.events
    : (rawEvents.length > 0 ? rawEvents : currentEvents);
  const eventsForDisplay = withOracleEventIds(eventsForDisplayBase);
  const eventsSourceForDisplay = promptCurrentView.hasSection
    ? "window_contexts.prompt_sections.current_observation_window"
    : (rawEvents.length > 0 ? round.rawEventSource : round.eventSource);
  const historyEvents = getPlotHistoryEvents(state.currentRoundIndex);
  const previousWindowEvents = getPreviousWindowEvents(state.currentRoundIndex);
  const previousWindowVitalsByLabel = buildLatestVitalValueByLabel(previousWindowEvents);
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
    `<div class="summary-card">No events available for this filter in current window.</div>`,
    {
      scope: "current",
      previousWindowVitalsByLabel,
    }
  );

  const eventTypeCounts = getEventTypeCounts(eventsForDisplay);
  const sourceLabel = formatEventSource(eventsSourceForDisplay);

  centerPanelEl.innerHTML = `
    <h2>Current Observation Window</h2>
    <div class="current-window-grid">
      <section class="current-window-section">
        <h3 class="subsection-title">Vital trends</h3>
        <div class="inline-meta">Vital plots include all prior-window history (grey) and current-window points (blue).</div>
        <div class="sparkline-grid">${sparklineHtml}</div>
      </section>
      <section class="current-window-section">
        <div class="section-header-row">
          <h3 class="subsection-title">Current events</h3>
          ${renderEventTypeFilter({ counts: eventTypeCounts, scope: "current", compact: true })}
        </div>
        <div class="inline-meta">Source: ${escapeHtml(sourceLabel)} | Events: ${eventsForDisplay.length}</div>
        <div class="timeline-list">${eventListHtml}</div>
      </section>
    </div>
  `;
}

function renderTimelineEvents(events, emptyHtml, options = {}) {
  const safeEvents = Array.isArray(events) ? events : [];
  if (safeEvents.length === 0) {
    return emptyHtml || `<div class="summary-card">No events available.</div>`;
  }
  const scope = safeString(options.scope) || "generic";
  const previousWindowVitalsByLabel = options.previousWindowVitalsByLabel instanceof Map
    ? options.previousWindowVitalsByLabel
    : new Map();

  return safeEvents
    .map((event) => {
      const eventType = classifyEventType(event);
      const eventValue = formatEventValue(event);
      const eventId = resolveEventEvidenceId(event);
      const anomalyInfo = getVitalTimelineAnomalyInfo(event, {
        scope,
        previousWindowVitalsByLabel,
      });
      const anomalyClasses = anomalyInfo && Array.isArray(anomalyInfo.classes)
        ? anomalyInfo.classes.join(" ")
        : "";
      const anomalyNote = anomalyInfo && safeString(anomalyInfo.note)
        ? `<div class="inline-meta anomaly-note">${escapeHtml(anomalyInfo.note)}</div>`
        : "";
      return `
        <article class="timeline-event event-${eventType} ${anomalyClasses}">
          <div class="event-top">
            <span class="time">${escapeHtml(formatTime(event.time || event.start_time || ""))}</span>
            ${eventId ? `<span class="event-id">${escapeHtml(eventId)}</span>` : ""}
            <span class="code">${escapeHtml(safeString(event.code) || "UNKNOWN")}</span>
          </div>
          <div>${escapeHtml(formatEventDescription(event))}</div>
          ${eventValue ? `<div class="inline-meta">Value: ${escapeHtml(eventValue)}</div>` : ""}
          ${anomalyNote}
        </article>
      `;
    })
    .join("");
}

function renderRightPanel() {
  const round = getCurrentRound();
  const annotation = getCurrentAnnotation();
  const isSkipped = annotation.skip_annotation === true;
  const patientStatus = round.oracleOutput.patient_assessment;

  rightPanelEl.innerHTML = `
    <h2>Annotation Tasks</h2>
    ${renderSkipControl(annotation)}
    ${
      isSkipped
        ? `
      <section class="task-section skip-window-note">
        <h3 class="task-title">This window is marked as skipped</h3>
        <div class="inline-meta">No annotation is required for this page.</div>
      </section>
    `
        : `
      ${renderTask1(patientStatus, annotation)}
      ${renderTask12RiskFactors(patientStatus, annotation)}
      ${renderTask21ActionEvaluation(round, annotation)}
      ${renderTask22RedFlags(round, annotation)}
      ${renderTask3PatientInsights(annotation)}
    `
    }
  `;
}

function renderSkipControl(annotation) {
  const isSkipped = annotation.skip_annotation === true;
  return `
    <section class="task-section">
      <h3 class="task-title">Skip this window?</h3>
      <div class="verdict-row">
        <label class="verdict-chip"><input type="radio" name="window-skip-toggle" value="false" ${
          isSkipped ? "" : "checked"
        } /> No (default)</label>
        <label class="verdict-chip"><input type="radio" name="window-skip-toggle" value="true" ${
          isSkipped ? "checked" : ""
        } /> Yes, skip this page</label>
      </div>
      <div class="inline-meta">If skipped, this window is excluded from required annotation checks.</div>
    </section>
  `;
}

function renderTask1(patientStatus, annotation) {
  const task1 = annotation.task1_status;
  const overall = isObject(patientStatus && patientStatus.overall) ? patientStatus.overall : {};
  const overallLabel = normalizeLabel(overall.label) || "unknown";
  const overallTone = overallToneClass(overallLabel);
  const labelErrorClass = hasCurrentHighlight("task1.selected_label") ? "error-highlight" : "";
  const selectedLabel = normalizeEnumValue(task1.selected_label, STATUS_CORRECTION_VALUES);

  return `
    <section class="task-section" id="task1-section">
      <h3 class="task-title">Task 1: Patient Status</h3>
      <div class="summary-card overall-card overall-${escapeHtml(overallTone)}">
        <div class="inline-meta"><strong>Overall status:</strong> <span class="badge badge-${escapeHtml(overallLabel)}">${escapeHtml(
          overallLabel
        )}</span></div>
        <div class="inline-meta"><strong>Reason:</strong> ${escapeHtml(safeString(overall.rationale) || "-")}</div>
      </div>

      <div class="verdict-row ${labelErrorClass}">
        ${renderOptionChips("task1-label", STATUS_CORRECTION_OPTIONS, selectedLabel)}
      </div>
    </section>
  `;
}

function renderTask12RiskFactors(patientStatus, annotation) {
  const task1 = annotation.task1_status;
  const predefinedRisks = getOracleActiveRiskNames(patientStatus);
  const selectedRiskNames = normalizeRiskNameList(task1.selected_risks);
  const selectedRiskKeySet = new Set(selectedRiskNames.map((riskName) => normalizeLabel(riskName)));
  const additionalRisks = ensureStringList(task1, "additional_risks");
  const riskOptions = predefinedRisks.length > 0
    ? `
      <div class="risk-chip-grid">
        ${
          predefinedRisks.map((riskName) => {
            const isSelected = selectedRiskKeySet.has(normalizeLabel(riskName));
            const selectedClass = isSelected ? "is-selected" : "";
            return `
              <button
                type="button"
                class="risk-chip ${selectedClass}"
                data-task1-predefined-risk-name="${escapeHtml(riskName)}"
              >${escapeHtml(riskName)}</button>
            `;
          }).join("")
        }
      </div>
    `
    : `<div class="inline-meta">No oracle active risks in this window.</div>`;

  return `
    <section class="task-section" id="task1-2-section">
      <h3 class="task-title">Task 1.2: Potential Risk Factors</h3>
      <div class="summary-card">
        ${riskOptions}
        <div class="task1-risk-list">
          ${
            additionalRisks.map((riskText, index) => `
              <div class="task1-risk-row">
                <input
                  type="text"
                  value="${escapeHtml(riskText)}"
                  data-task1-additional-risk-index="${index}"
                  placeholder="e.g., AKI progression"
                />
                <button type="button" class="secondary-btn" data-task1-additional-risk-remove-index="${index}">Remove</button>
              </div>
            `).join("")
          }
        </div>
        <button type="button" class="secondary-btn" data-task1-additional-risk-add="1">Add more risks</button>
      </div>
    </section>
  `;
}

function renderTask21ActionEvaluation(round, annotation) {
  const additionalRecs = annotation.task3_additional_recommendations;
  const additionalActions = getTextConfidenceItems(additionalRecs.actions);
  const actionCards = round.actions
    .map((action, index) => {
      const actionAnn = annotation.task2_actions[index] || { selected_label: null, selected_urgency: null };
      const overallTone = overallToneClass(action.overall_label);
      const labelErrorClass = hasCurrentHighlight(`action.${index}.selected_label`) ? "error-highlight" : "";
      const urgencyErrorClass = hasCurrentHighlight(`action.${index}.selected_urgency`) ? "error-highlight" : "";
      const cardErrorClass = (labelErrorClass || urgencyErrorClass) ? "error-highlight" : "";
      const selectedLabel = normalizeEnumValue(actionAnn.selected_label, ACTION_CORRECTION_VALUES);
      const selectedUrgency = normalizeEnumValue(actionAnn.selected_urgency, ACTION_URGENCY_VALUES);
      const needsUrgency = selectedLabel === "best_practice" || selectedLabel === "acceptable";

      return `
        <article class="action-card overall-card overall-${escapeHtml(overallTone)} ${cardErrorClass}" id="action-card-${index}">
          <div><strong>${escapeHtml(action.action_id)}</strong> - ${escapeHtml(action.action_description)}</div>
          <div class="inline-meta"><strong>Overall label:</strong> <span class="badge badge-${escapeHtml(action.overall_label)}">${escapeHtml(
            action.overall_label
          )}</span></div>
          <div class="inline-meta"><strong>Reason:</strong> ${escapeHtml(action.overall_rationale || "No rationale provided.")}</div>

          <div class="verdict-row ${labelErrorClass}">
            ${renderOptionChips(`action-label-${index}`, ACTION_CORRECTION_OPTIONS, selectedLabel, {
              "data-action-index": String(index),
            })}
          </div>
          ${
            needsUrgency
              ? `
                <div class="inline-meta"><strong>Urgency</strong></div>
                <div class="verdict-row ${urgencyErrorClass}">
                  ${renderOptionChips(`action-urgency-${index}`, ACTION_URGENCY_OPTIONS, selectedUrgency, {
                    "data-action-urgency-index": String(index),
                  })}
                </div>
              `
              : ""
          }
        </article>
      `;
    })
    .join("");

  const additionalActionsBlock = `
    <div class="summary-card">
      <strong>More Recommended Actions?</strong>
      <div class="control-group ${hasCurrentHighlight("task3_additional_recommendations.actions") ? "error-highlight" : ""}">
        <label>Additional action(s)</label>
        <div class="inline-meta">Choose category first, then write free text. Evidence event IDs are optional (if provided, use comma-separated IDs like 12345).</div>
        <div class="additional-actions-list">
          ${
            additionalActions.map((actionItem, actionIndex) => {
              const categoryToken = `task3_additional_recommendations.actions.${actionIndex}.category`;
              const evidenceToken = `task3_additional_recommendations.actions.${actionIndex}.evidence_event_ids`;
              const categoryErrorClass = hasCurrentHighlight(categoryToken) ? "error-highlight" : "";
              const evidenceErrorClass = hasCurrentHighlight(evidenceToken) ? "error-highlight" : "";
              const categoryValue = normalizeAdditionalActionCategory(actionItem.category);
              return `
                <div class="additional-action-block">
                  <div class="additional-action-block-row">
                    <select
                      class="additional-action-category-select ${categoryErrorClass}"
                      data-additional-action-category-index="${actionIndex}"
                    >
                      <option value="">Select category first</option>
                      ${ICU_ACTION_CATEGORIES.map((categoryOption) => {
                        const selected = categoryValue === categoryOption ? "selected" : "";
                        return `<option value="${escapeHtml(categoryOption)}" ${selected}>${escapeHtml(categoryOption)}</option>`;
                      }).join("")}
                    </select>
                    <select class="additional-action-confidence-select" data-additional-action-confidence-index="${actionIndex}">
                      <option value="">confidence</option>
                      ${FREE_TEXT_CONFIDENCE_OPTIONS.map((option) => {
                        const selected = actionItem.confidence === option.value ? "selected" : "";
                        return `<option value="${escapeHtml(option.value)}" ${selected}>${escapeHtml(option.label)}</option>`;
                      }).join("")}
                    </select>
                    <button
                      type="button"
                      class="secondary-btn additional-action-remove-btn"
                      data-additional-action-remove-index="${actionIndex}"
                    >Remove block</button>
                  </div>
                  <div class="additional-action-block-row">
                    <input
                      type="text"
                      class="additional-action-text-input"
                      ${categoryValue ? "" : "disabled"}
                      value="${escapeHtml(actionItem.text)}"
                      data-additional-action-index="${actionIndex}"
                      placeholder="${categoryValue ? "Enter one action" : "Select category first"}"
                    />
                    <input
                      type="text"
                      class="evidence-ids-input ${evidenceErrorClass}"
                      value="${escapeHtml(actionItem.evidence_event_ids_text || "")}"
                      data-additional-action-evidence-index="${actionIndex}"
                      placeholder="Evidence IDs (optional): 12345"
                    />
                  </div>
                </div>
              `;
            }).join("")
          }
        </div>
        <button type="button" class="secondary-btn" data-additional-action-add="1">Add action</button>
      </div>
    </div>
  `;

  return `
    <section class="task-section" id="task2-1-section">
      <h3 class="task-title">Task 2.1: Action Evaluation</h3>
      ${actionCards || `<div class="summary-card">No actions in this window.</div>`}
      ${additionalActionsBlock}
    </section>
  `;
}

function renderTask22RedFlags(round, annotation) {
  const predefinedRedFlags = getOracleContraindicatedActionNames(round.recommendations);
  const selectedActions = normalizeRedFlagActionList(annotation.task3_red_flag_actions.selected_actions);
  const selectedActionSet = new Set(selectedActions.map((action) => normalizeLabel(action)));
  const additionalActions = ensureStringList(annotation.task3_red_flag_actions, "additional_actions");
  const redFlagOptions = predefinedRedFlags.length > 0
    ? `
      <div class="risk-chip-grid">
        ${
          predefinedRedFlags.map((actionText) => {
            const isSelected = selectedActionSet.has(normalizeLabel(actionText));
            const selectedClass = isSelected ? "is-selected" : "";
            return `
              <button
                type="button"
                class="risk-chip ${selectedClass}"
                data-task22-predefined-red-flag-action="${escapeHtml(actionText)}"
              >${escapeHtml(actionText)}</button>
            `;
          }).join("")
        }
      </div>
    `
    : `<div class="inline-meta">No oracle red flag actions in this window.</div>`;

  return `
    <section class="task-section" id="task2-2-section">
      <h3 class="task-title">Task 2.2: Red Flags</h3>
      <div class="summary-card">
        ${redFlagOptions}
        <div class="task1-risk-list">
          ${
            additionalActions.map((actionText, actionIndex) => `
              <div class="task1-risk-row">
                <input
                  type="text"
                  value="${escapeHtml(actionText)}"
                  data-task22-additional-red-flag-index="${actionIndex}"
                  placeholder="e.g., Avoid aggressive BP lowering"
                />
                <button type="button" class="secondary-btn" data-task22-additional-red-flag-remove-index="${actionIndex}">Remove</button>
              </div>
            `).join("")
          }
        </div>
        <button type="button" class="secondary-btn" data-task22-additional-red-flag-add="1">Add red flag actions</button>
      </div>
    </section>
  `;
}

function renderTask3PatientInsights(annotation) {
  const insights = getTextConfidenceItems(
    isObject(annotation.task4_patient_insights) ? annotation.task4_patient_insights.insights : []
  );
  const insightsErrorClass = hasCurrentHighlight("task4_patient_insights.insights") ? "error-highlight" : "";

  return `
    <section class="task-section" id="task3-section">
      <h3 class="task-title">Task 3: Patient Insights (Optional)</h3>
      <div class="review-note">Use 1-2 sentences per item to describe what is uniquely notable about this patient vs average patients.</div>
      <div class="control-group ${insightsErrorClass}">
        <label>Unique patient insights</label>
        <div class="additional-actions-list">
          ${
            insights.map((insightItem, index) => {
              return `
                <div class="additional-action-row">
                  <input
                    type="text"
                    value="${escapeHtml(insightItem.text)}"
                    data-patient-insight-index="${index}"
                    placeholder="e.g., Most unusual feature compared with average patients"
                  />
                  <select data-patient-insight-confidence-index="${index}">
                    <option value="">confidence</option>
                    ${FREE_TEXT_CONFIDENCE_OPTIONS.map((option) => {
                      const selected = insightItem.confidence === option.value ? "selected" : "";
                      return `<option value="${escapeHtml(option.value)}" ${selected}>${escapeHtml(option.label)}</option>`;
                    }).join("")}
                  </select>
                  <button type="button" class="secondary-btn" data-patient-insight-remove-index="${index}">Remove</button>
                </div>
              `;
            }).join("")
          }
        </div>
        <button type="button" class="secondary-btn" data-patient-insight-add="1">Add insight</button>
      </div>
    </section>
  `;
}

function renderOptionChips(name, options, selectedValue, extraAttributes = {}) {
  const normalizedOptions = (Array.isArray(options) ? options : []).map((option) => {
    if (isObject(option)) {
      return {
        value: safeString(option.value),
        label: safeString(option.label || option.value),
      };
    }
    const value = safeString(option);
    return { value, label: value };
  }).filter((option) => option.value.length > 0);
  const attrs = Object.entries(extraAttributes)
    .map(([k, v]) => `${k}="${escapeHtml(v)}"`)
    .join(" ");

  return normalizedOptions
    .map((option) => {
      const checked = selectedValue === option.value ? "checked" : "";
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

function formatDischargeSummaryHtml(dischargeSummaryText, roundArrayIndex = null) {
  const text = safeString(dischargeSummaryText);
  if (!text) {
    return `<div class="inline-meta">No ICU discharge summary context available.</div>`;
  }

  const normalized = normalizeDischargeSummaryText(text);
  if (!normalized) {
    return `<div class="inline-meta">No ICU discharge summary context available.</div>`;
  }

  const parsedSections = parseDischargeSummarySections(normalized);
  const sections = parsedSections.length > 0
    ? parsedSections
    : [{ title: "Summary", body: normalized }];
  const openState = getDischargeSummaryOpenSections(roundArrayIndex);

  return `
    <div class="context-section-list">
      ${sections
        .map((section, index) => {
          const sectionTitle = safeString(section.title) || (sections.length === 1 ? "Summary" : `Section ${index + 1}`);
          const sectionBody = safeString(section.body) || "-";
          const isOpen = Boolean(openState[index]);
          return `
            <details
              class="context-section-card discharge-section-card"
              data-discharge-section-index="${index}"
              ${isOpen ? "open" : ""}
            >
              <summary class="context-section-summary">
                <span class="context-section-title">${escapeHtml(sectionTitle)}</span>
              </summary>
              <pre class="context-pre">${escapeHtml(sectionBody)}</pre>
            </details>
          `;
        })
        .join("")}
    </div>
  `;
}

function getDischargeSummaryOpenSections(roundArrayIndex) {
  if (roundArrayIndex === null || roundArrayIndex === undefined) {
    return {};
  }
  const openState = state.dischargeSummaryOpenSectionsByRound[roundArrayIndex];
  return isObject(openState) ? openState : {};
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

  // Promote common issue block labels to section headings.
  normalized = normalized.replace(/\b(ACTIVE ISSUES|INACTIVE ISSUES|TRANSITIONAL ISSUES)\b/gi, "\n\n## $1:\n");

  // Convert only standalone, short hash-heading lines (not inline "# ...").
  normalized = normalized
    .split("\n")
    .map((line) => normalizeDischargeSummaryHashHeadingLine(line))
    .join("\n");

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

function normalizeDischargeSummaryHashHeadingLine(line) {
  const asString = line === null || line === undefined ? "" : String(line);
  const headingMatch = asString.match(/^\s*#\s*([A-Za-z][A-Za-z0-9/&()'+.,\- ]{1,64})\s*:?\s*$/);
  if (!headingMatch) {
    return asString;
  }
  const title = safeString(headingMatch[1]).replace(/:\s*$/, "");
  if (!title) {
    return asString;
  }
  return `## ${title}`;
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
  scheduleAutoSave();
  renderWorkspace();
}

function stepWindow(offset) {
  state.currentRoundIndex = clamp(state.currentRoundIndex + offset, 0, state.rounds.length - 1);
  state.currentEventTypeFilter = "all";
  state.historyEventTypeFilter = "all";
  scheduleAutoSave();
  renderWorkspace();
}

function onWorkspaceClick(event) {
  if (!state.loaded) {
    return;
  }

  const target = event.target;
  const annotation = getCurrentAnnotation();
  clearCurrentRoundErrorHighlights();
  let shouldRender = false;

  if (target.dataset.task1PredefinedRiskName !== undefined) {
    const riskName = safeString(target.dataset.task1PredefinedRiskName);
    if (riskName) {
      const selectedRisks = normalizeRiskNameList(annotation.task1_status.selected_risks);
      const targetKey = normalizeLabel(riskName);
      const existingIndex = selectedRisks.findIndex((item) => normalizeLabel(item) === targetKey);
      if (existingIndex >= 0) {
        selectedRisks.splice(existingIndex, 1);
      } else {
        selectedRisks.push(riskName);
      }
      annotation.task1_status.selected_risks = selectedRisks;
      shouldRender = true;
    }
  }

  if (target.dataset.task1AdditionalRiskAdd) {
    const additionalRisks = ensureStringList(annotation.task1_status, "additional_risks");
    additionalRisks.push("");
    shouldRender = true;
  }

  if (target.dataset.task1AdditionalRiskRemoveIndex) {
    const removeIndex = toInt(target.dataset.task1AdditionalRiskRemoveIndex, null);
    const additionalRisks = ensureStringList(annotation.task1_status, "additional_risks");
    if (removeIndex !== null && removeIndex >= 0 && removeIndex < additionalRisks.length) {
      additionalRisks.splice(removeIndex, 1);
      shouldRender = true;
    }
  }

  if (target.dataset.additionalActionAdd) {
    const items = ensureTextConfidenceList(annotation.task3_additional_recommendations, "actions");
    items.push(buildTextConfidenceItem());
    shouldRender = true;
  }

  if (target.dataset.additionalActionRemoveIndex) {
    const removeIndex = toInt(target.dataset.additionalActionRemoveIndex, null);
    const items = ensureTextConfidenceList(annotation.task3_additional_recommendations, "actions");
    if (removeIndex !== null && Array.isArray(items) && removeIndex >= 0 && removeIndex < items.length) {
      items.splice(removeIndex, 1);
      shouldRender = true;
    }
  }

  if (target.dataset.task22PredefinedRedFlagAction !== undefined) {
    const actionText = safeString(target.dataset.task22PredefinedRedFlagAction);
    if (actionText) {
      const selectedActions = normalizeRedFlagActionList(annotation.task3_red_flag_actions.selected_actions);
      const actionKey = normalizeLabel(actionText);
      const existingIndex = selectedActions.findIndex((item) => normalizeLabel(item) === actionKey);
      if (existingIndex >= 0) {
        selectedActions.splice(existingIndex, 1);
      } else {
        selectedActions.push(actionText);
      }
      annotation.task3_red_flag_actions.selected_actions = selectedActions;
      shouldRender = true;
    }
  }

  if (target.dataset.task22AdditionalRedFlagAdd) {
    const additionalActions = ensureStringList(annotation.task3_red_flag_actions, "additional_actions");
    additionalActions.push("");
    shouldRender = true;
  }

  if (target.dataset.task22AdditionalRedFlagRemoveIndex) {
    const removeIndex = toInt(target.dataset.task22AdditionalRedFlagRemoveIndex, null);
    const additionalActions = ensureStringList(annotation.task3_red_flag_actions, "additional_actions");
    if (removeIndex !== null && removeIndex >= 0 && removeIndex < additionalActions.length) {
      additionalActions.splice(removeIndex, 1);
      shouldRender = true;
    }
  }

  if (target.dataset.patientInsightAdd) {
    if (!isObject(annotation.task4_patient_insights)) {
      annotation.task4_patient_insights = { insights: [] };
    }
    const items = ensureTextConfidenceList(annotation.task4_patient_insights, "insights");
    items.push(buildTextConfidenceItem());
    shouldRender = true;
  }

  if (target.dataset.patientInsightRemoveIndex) {
    const removeIndex = toInt(target.dataset.patientInsightRemoveIndex, null);
    const items = ensureTextConfidenceList(annotation.task4_patient_insights, "insights");
    if (removeIndex !== null && Array.isArray(items) && removeIndex >= 0 && removeIndex < items.length) {
      items.splice(removeIndex, 1);
      shouldRender = true;
    }
  }

  if (shouldRender) {
    scheduleAutoSave();
    renderWorkspace();
  }
}

function onWorkspaceChange(event) {
  if (!state.loaded) {
    return;
  }

  const target = event.target;
  const annotation = getCurrentAnnotation();
  let shouldRender = false;
  clearCurrentRoundErrorHighlights();

  if (target.name === "window-skip-toggle") {
    annotation.skip_annotation = target.value === "true";
    shouldRender = true;
  }

  if (target.name === "task1-label") {
    annotation.task1_status.selected_label = normalizeEnumValue(target.value, STATUS_CORRECTION_VALUES);
    shouldRender = true;
  }

  if (target.name.startsWith("action-label-")) {
    const actionIndex = toInt(target.dataset.actionIndex, null);
    if (actionIndex !== null && annotation.task2_actions[actionIndex]) {
      const selectedLabel = normalizeEnumValue(target.value, ACTION_CORRECTION_VALUES);
      annotation.task2_actions[actionIndex].selected_label = selectedLabel;
      if (selectedLabel !== "best_practice" && selectedLabel !== "acceptable") {
        annotation.task2_actions[actionIndex].selected_urgency = null;
      }
      shouldRender = true;
    }
  }

  if (target.name.startsWith("action-urgency-")) {
    const actionIndex = toInt(target.dataset.actionUrgencyIndex, null);
    if (actionIndex !== null && annotation.task2_actions[actionIndex]) {
      annotation.task2_actions[actionIndex].selected_urgency = normalizeEnumValue(target.value, ACTION_URGENCY_VALUES);
      shouldRender = true;
    }
  }

  if (target.name.startsWith("rec-verdict-")) {
    const recIndex = toInt(target.dataset.recIndex, null);
    if (recIndex !== null && annotation.task3_recommendations[recIndex]) {
      annotation.task3_recommendations[recIndex].verdict = target.value;
      if (target.value !== "agree") {
        annotation.task3_recommendations[recIndex].agreed_urgency = null;
        annotation.task3_recommendations[recIndex].agreed_dose = null;
      }
      shouldRender = true;
    }
  }

  if (target.dataset.recAgreeUrgencyIndex) {
    const recIndex = toInt(target.dataset.recAgreeUrgencyIndex, null);
    if (recIndex !== null && annotation.task3_recommendations[recIndex]) {
      annotation.task3_recommendations[recIndex].agreed_urgency = target.value || null;
    }
  }

  if (target.dataset.recAgreeDoseIndex) {
    const recIndex = toInt(target.dataset.recAgreeDoseIndex, null);
    if (recIndex !== null && annotation.task3_recommendations[recIndex]) {
      annotation.task3_recommendations[recIndex].agreed_dose = target.value || null;
    }
  }

  if (target.dataset.additionalActionConfidenceIndex) {
    const actionIndex = toInt(target.dataset.additionalActionConfidenceIndex, null);
    const items = ensureTextConfidenceList(annotation.task3_additional_recommendations, "actions");
    if (actionIndex !== null && actionIndex >= 0 && actionIndex < items.length) {
      items[actionIndex].confidence = normalizeConfidence(target.value);
    }
  }

  if (target.dataset.additionalActionCategoryIndex) {
    const actionIndex = toInt(target.dataset.additionalActionCategoryIndex, null);
    const items = ensureTextConfidenceList(annotation.task3_additional_recommendations, "actions");
    if (actionIndex !== null && actionIndex >= 0 && actionIndex < items.length) {
      items[actionIndex].category = normalizeAdditionalActionCategory(target.value);
      shouldRender = true;
    }
  }

  if (target.dataset.patientInsightConfidenceIndex) {
    const insightIndex = toInt(target.dataset.patientInsightConfidenceIndex, null);
    const items = ensureTextConfidenceList(annotation.task4_patient_insights, "insights");
    if (insightIndex !== null && insightIndex >= 0 && insightIndex < items.length) {
      items[insightIndex].confidence = normalizeConfidence(target.value);
    }
  }

  scheduleAutoSave();
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

  if (target.dataset.task1AdditionalRiskIndex) {
    const riskIndex = toInt(target.dataset.task1AdditionalRiskIndex, null);
    const additionalRisks = ensureStringList(annotation.task1_status, "additional_risks");
    if (riskIndex !== null && riskIndex >= 0 && riskIndex < additionalRisks.length) {
      additionalRisks[riskIndex] = target.value;
    }
  }

  if (target.dataset.additionalActionIndex) {
    const actionIndex = toInt(target.dataset.additionalActionIndex, null);
    const items = ensureTextConfidenceList(annotation.task3_additional_recommendations, "actions");
    if (actionIndex !== null && Array.isArray(items) && actionIndex >= 0 && actionIndex < items.length) {
      items[actionIndex].text = target.value;
    }
  }

  if (target.dataset.additionalActionEvidenceIndex) {
    const actionIndex = toInt(target.dataset.additionalActionEvidenceIndex, null);
    const items = ensureTextConfidenceList(annotation.task3_additional_recommendations, "actions");
    if (actionIndex !== null && Array.isArray(items) && actionIndex >= 0 && actionIndex < items.length) {
      items[actionIndex].evidence_event_ids_text = target.value;
    }
  }

  if (target.dataset.task22AdditionalRedFlagIndex) {
    const actionIndex = toInt(target.dataset.task22AdditionalRedFlagIndex, null);
    const additionalActions = ensureStringList(annotation.task3_red_flag_actions, "additional_actions");
    if (actionIndex !== null && actionIndex >= 0 && actionIndex < additionalActions.length) {
      additionalActions[actionIndex] = target.value;
    }
  }

  if (target.dataset.patientInsightIndex) {
    const insightIndex = toInt(target.dataset.patientInsightIndex, null);
    const items = ensureTextConfidenceList(annotation.task4_patient_insights, "insights");
    if (insightIndex !== null && Array.isArray(items) && insightIndex >= 0 && insightIndex < items.length) {
      items[insightIndex].text = target.value;
    }
  }

  scheduleAutoSave();
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
    setGlobalMessage(message.replace(/\n/g, " "), "warn");
    return;
  }

  if (validation.warnings.length > 0) {
    const warningMessage = [`There are ${validation.warnings.length} review warning(s). Exported with warnings:`]
      .concat(validation.warnings.slice(0, 10).map((item) => `- ${item}`))
      .join("\n");
    setGlobalMessage(warningMessage.replace(/\n/g, " "), "warn");
  }

  const nowIso = new Date().toISOString();
  const durationSeconds = Math.max(0, Math.floor((Date.now() - state.loadedAtMs) / 1000));

  const output = cloneData(state.predictionsData);
  state.rounds.forEach((round) => {
    const annotation = state.annotations[round.arrayIndex];
    const activeRiskSet = new Set(
      getOracleActiveRiskNames(round.oracleOutput.patient_assessment).map((name) => normalizeLabel(name))
    );
    const oracleRedFlagSet = new Set(
      getOracleContraindicatedActionNames(round.recommendations).map((action) => normalizeLabel(action))
    );
    const warnings = validation.windowWarnings[round.arrayIndex] || {
      unreviewed_actions: [],
      unreviewed_recommendations: [],
    };

    output.window_outputs[round.arrayIndex].annotation = {
      window_key: round.windowKey,
      skip_annotation: annotation.skip_annotation === true,
      annotated_at: nowIso,
      duration_seconds: durationSeconds,
      task1_status: {
        selected_label: normalizeEnumValue(annotation.task1_status.selected_label, STATUS_CORRECTION_VALUES),
        selected_risks: normalizeRiskNameList(annotation.task1_status.selected_risks)
          .filter((riskName) => activeRiskSet.has(normalizeLabel(riskName))),
        additional_risks: normalizeRiskNameList(annotation.task1_status.additional_risks),
      },
      task2_actions: annotation.task2_actions.map((entry) => {
        const selectedLabel = normalizeEnumValue(entry.selected_label, ACTION_CORRECTION_VALUES);
        const selectedUrgency = normalizeEnumValue(entry.selected_urgency, ACTION_URGENCY_VALUES);
        return {
          action_id: entry.action_id,
          selected_label: selectedLabel,
          selected_urgency: (
            selectedLabel === "best_practice" || selectedLabel === "acceptable"
              ? selectedUrgency
              : null
          ),
        };
      }),
      task3_recommendations: annotation.task3_recommendations.map((entry) => ({
        rank: entry.rank,
        position_index: entry.position_index,
        verdict: entry.verdict,
        agreed_urgency: entry.agreed_urgency,
        agreed_dose: entry.agreed_dose,
      })),
      task3_additional_recommendations: {
        actions: exportAdditionalRecommendationItems(annotation.task3_additional_recommendations.actions),
      },
      task3_red_flag_actions: {
        selected_actions: normalizeRedFlagActionList(annotation.task3_red_flag_actions.selected_actions)
          .filter((action) => oracleRedFlagSet.has(normalizeLabel(action))),
        additional_actions: normalizeRedFlagActionList(annotation.task3_red_flag_actions.additional_actions),
      },
      task4_patient_insights: {
        insights: exportTextConfidenceItems(annotation.task4_patient_insights.insights, "insight"),
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
      full_window_contexts: state.sourceFiles.full_window_contexts,
    },
  };

  const filename = `${safeString(output.subject_id) || "subject"}_${safeString(output.icu_stay_id) || "stay"}_annotated.json`;
  state.errorHighlights = {};
  flushAutoSaveNow();
  downloadJson(output, filename);
  setGlobalMessage(`Exported ${filename}`, "info");
}

function onImportDraftClick() {
  if (!state.loaded || !draftFileInputEl) {
    return;
  }
  draftFileInputEl.value = "";
  draftFileInputEl.click();
}

async function onDraftFileSelected(event) {
  if (!state.loaded) {
    return;
  }
  const file = event && event.target && event.target.files && event.target.files[0]
    ? event.target.files[0]
    : null;
  if (!file) {
    return;
  }

  try {
    const draftPayload = await readJsonFile(file);
    const appliedCount = applyDraftSnapshot(draftPayload);
    if (appliedCount <= 0) {
      throw new Error("No draft annotations were applied to this case.");
    }
    persistDraftToLocalStorage();
    renderWorkspace();
    setGlobalMessage(`Imported draft (${appliedCount} window(s) restored).`, "info");
  } catch (error) {
    setGlobalMessage(`Failed to import draft: ${error.message}`, "warn");
  } finally {
    if (draftFileInputEl) {
      draftFileInputEl.value = "";
    }
  }
}

function onExportDraft() {
  if (!state.loaded) {
    return;
  }
  const snapshot = buildDraftSnapshot();
  const filename = `${safeString(snapshot.subject_id) || "subject"}_${safeString(snapshot.icu_stay_id) || "stay"}_annotation_draft.json`;
  flushAutoSaveNow();
  downloadJson(snapshot, filename);
  setGlobalMessage(`Exported draft ${filename}`, "info");
}

function buildDraftStorageKey(predictionsData) {
  const subjectId = safeString(predictionsData && predictionsData.subject_id) || "unknown_subject";
  const icuStayId = safeString(predictionsData && predictionsData.icu_stay_id) || "unknown_stay";
  const totalWindows = Array.isArray(predictionsData && predictionsData.window_outputs)
    ? predictionsData.window_outputs.length
    : 0;
  return `${DRAFT_STORAGE_PREFIX}:${subjectId}:${icuStayId}:${totalWindows}`;
}

function buildDraftSnapshot() {
  return {
    draft_version: 1,
    saved_at: new Date().toISOString(),
    draft_key: state.draftKey,
    subject_id: state.predictionsData ? safeString(state.predictionsData.subject_id) : "",
    icu_stay_id: state.predictionsData ? safeString(state.predictionsData.icu_stay_id) : "",
    total_windows: state.rounds.length,
    current_round_index: state.currentRoundIndex,
    source_files: cloneData(state.sourceFiles),
    annotations: cloneData(state.annotations),
  };
}

function scheduleAutoSave() {
  if (!state.loaded || !state.draftKey) {
    return;
  }
  clearAutoSaveTimer();
  state.autoSaveTimerId = window.setTimeout(() => {
    persistDraftToLocalStorage();
  }, AUTO_SAVE_DEBOUNCE_MS);
}

function clearAutoSaveTimer() {
  if (state.autoSaveTimerId !== null) {
    window.clearTimeout(state.autoSaveTimerId);
    state.autoSaveTimerId = null;
  }
}

function flushAutoSaveNow() {
  if (!state.loaded || !state.draftKey) {
    return false;
  }
  clearAutoSaveTimer();
  return persistDraftToLocalStorage();
}

function persistDraftToLocalStorage() {
  if (!state.loaded || !state.draftKey) {
    return false;
  }
  try {
    const snapshot = buildDraftSnapshot();
    localStorage.setItem(state.draftKey, JSON.stringify(snapshot));
    return true;
  } catch (error) {
    if (!state.autoSaveErrorShown) {
      state.autoSaveErrorShown = true;
      setGlobalMessage(`Auto-save failed: ${error.message}`, "warn");
    }
    return false;
  }
}

function restoreDraftFromLocalStorage() {
  if (!state.draftKey) {
    return 0;
  }
  try {
    const raw = localStorage.getItem(state.draftKey);
    if (!raw) {
      return 0;
    }
    const snapshot = JSON.parse(raw);
    return applyDraftSnapshot(snapshot);
  } catch (error) {
    return 0;
  }
}

function applyDraftSnapshot(snapshot) {
  if (!isObject(snapshot) || !isObject(snapshot.annotations)) {
    return 0;
  }
  if (!isDraftCompatibleWithCurrentSession(snapshot)) {
    throw new Error("Draft file does not match current patient/case.");
  }

  const annotationsByIndex = snapshot.annotations;
  const annotationsByWindowKey = new Map();
  Object.values(annotationsByIndex).forEach((entry) => {
    if (!isObject(entry)) {
      return;
    }
    const windowKey = safeString(entry.window_key);
    if (windowKey) {
      annotationsByWindowKey.set(windowKey, entry);
    }
  });

  let appliedCount = 0;
  state.rounds.forEach((round) => {
    const rawByIndex = annotationsByIndex[round.arrayIndex] || annotationsByIndex[String(round.arrayIndex)];
    const rawByWindowKey = annotationsByWindowKey.get(round.windowKey);
    const rawAnnotation = isObject(rawByIndex) ? rawByIndex : (isObject(rawByWindowKey) ? rawByWindowKey : null);
    if (!rawAnnotation) {
      return;
    }
    state.annotations[round.arrayIndex] = normalizeAnnotationForRound(round, rawAnnotation);
    appliedCount += 1;
  });

  const nextRoundIndex = toInt(snapshot.current_round_index, state.currentRoundIndex);
  state.currentRoundIndex = clamp(nextRoundIndex, 0, state.rounds.length - 1);
  state.errorHighlights = {};
  return appliedCount;
}

function isDraftCompatibleWithCurrentSession(snapshot) {
  const snapshotSubject = safeString(snapshot.subject_id);
  const snapshotStay = safeString(snapshot.icu_stay_id);
  const currentSubject = safeString(state.predictionsData && state.predictionsData.subject_id);
  const currentStay = safeString(state.predictionsData && state.predictionsData.icu_stay_id);
  if (snapshotSubject && currentSubject && snapshotSubject !== currentSubject) {
    return false;
  }
  if (snapshotStay && currentStay && snapshotStay !== currentStay) {
    return false;
  }
  const snapshotWindows = toInt(snapshot.total_windows, state.rounds.length);
  if (snapshotWindows !== state.rounds.length) {
    return false;
  }
  return true;
}

function collectValidationReport() {
  const blocking = [];
  const blockingDetails = [];
  const warnings = [];
  const windowWarnings = {};

  state.rounds.forEach((round) => {
    const annotation = state.annotations[round.arrayIndex];
    const labelPrefix = `${round.windowKey}`;
    const isSkipped = annotation.skip_annotation === true;
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

    const unreviewedActions = isSkipped
      ? []
      : annotation.task2_actions.filter((action) => action.selected_label === null).map((action) => action.action_id);

    windowWarnings[round.arrayIndex] = {
      unreviewed_actions: unreviewedActions,
      unreviewed_recommendations: [],
    };

    if (!isSkipped && unreviewedActions.length > 0) {
      warnings.push(`${labelPrefix}: unreviewed actions (${unreviewedActions.join(", ")})`);
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
  if (annotation.skip_annotation === true) {
    return issues;
  }

  if (!annotation.task1_status.selected_label) {
    issues.push({
      token: "task1.selected_label",
      message: "Task 1 label is missing.",
    });
  }

  annotation.task2_actions.forEach((entry, actionIndex) => {
    const selectedLabel = normalizeEnumValue(entry && entry.selected_label, ACTION_CORRECTION_VALUES);
    if (selectedLabel) {
      if (selectedLabel === "best_practice" || selectedLabel === "acceptable") {
        const selectedUrgency = normalizeEnumValue(entry && entry.selected_urgency, ACTION_URGENCY_VALUES);
        if (!selectedUrgency) {
          const actionId = safeString(entry && entry.action_id) || `#${actionIndex + 1}`;
          issues.push({
            token: `action.${actionIndex}.selected_urgency`,
            message: `Task 2.1 urgency is missing for action ${actionId}.`,
          });
        }
      }
      return;
    }
    const actionId = safeString(entry && entry.action_id) || `#${actionIndex + 1}`;
    issues.push({
      token: `action.${actionIndex}.selected_label`,
      message: `Task 2.1 label is missing for action ${actionId}.`,
    });
  });

  const additionalActions = getTextConfidenceItems(annotation.task3_additional_recommendations.actions);
  additionalActions.forEach((item, actionIndex) => {
    if (!safeString(item.text)) {
      return;
    }
    const category = normalizeAdditionalActionCategory(item.category);
    if (!category) {
      issues.push({
        token: `task3_additional_recommendations.actions.${actionIndex}.category`,
        message: `Additional action ${actionIndex + 1} requires selecting a category.`,
      });
    }
    const evidenceParse = parseEvidenceEventIdsInput(item.evidence_event_ids_text || "");
    if (evidenceParse.invalid.length > 0) {
      issues.push({
        token: `task3_additional_recommendations.actions.${actionIndex}.evidence_event_ids`,
        message: `Additional action ${actionIndex + 1} has invalid event IDs: ${evidenceParse.invalid.join(", ")}.`,
      });
    }
    if (evidenceParse.valid.length > MAX_ADDITIONAL_ACTION_EVIDENCE_IDS) {
      issues.push({
        token: `task3_additional_recommendations.actions.${actionIndex}.evidence_event_ids`,
        message:
          `Additional action ${actionIndex + 1} allows at most `
          + `${MAX_ADDITIONAL_ACTION_EVIDENCE_IDS} evidence event IDs.`,
      });
    }
  });

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
  const isSkipped = annotation.skip_annotation === true;
  if (isSkipped) {
    return {
      isSkipped: true,
      task1Done: true,
      reviewedActions: round.actions.length,
      totalActions: round.actions.length,
      otherActionsCount: 0,
      task3InsightCount: 0,
    };
  }

  const reviewedActions = annotation.task2_actions.filter((entry) => entry.selected_label !== null).length;
  const otherActionsCount = getTextConfidenceItems(annotation.task3_additional_recommendations.actions)
    .filter((item) => item.text.length > 0)
    .length;
  const task3InsightCount = getTextConfidenceItems(
    isObject(annotation.task4_patient_insights) ? annotation.task4_patient_insights.insights : []
  ).filter((item) => item.text.length > 0).length;

  return {
    isSkipped: false,
    task1Done: annotation.task1_status.selected_label !== null,
    reviewedActions,
    totalActions: round.actions.length,
    otherActionsCount,
    task3InsightCount,
  };
}

function getPlotHistoryEvents(roundPosition) {
  const currentRound = state.rounds[roundPosition];
  const currentStartMs = getWindowStartMs(currentRound);
  const fullTimelinePriorEvents = collectPriorEventsFromFullTimeline(roundPosition);
  const priorEvents = fullTimelinePriorEvents === null ? collectPriorEvents(roundPosition) : fullTimelinePriorEvents;
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
  const fullTimelinePriorEvents = collectPriorEventsFromFullTimeline(roundPosition);
  const priorEvents = fullTimelinePriorEvents === null ? collectPriorEvents(roundPosition) : fullTimelinePriorEvents;
  if (priorEvents.length === 0) {
    return [];
  }

  if (currentStartMs === null) {
    return getPreviousWindowEvents(roundPosition);
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

function getPreviousWindowEvents(roundPosition) {
  if (!Number.isFinite(roundPosition) || roundPosition < 0) {
    return [];
  }
  const fullTimelinePreviousEvents = getPreviousWindowEventsFromFullTimeline(roundPosition);
  if (fullTimelinePreviousEvents !== null) {
    return fullTimelinePreviousEvents;
  }
  if (roundPosition <= 0) {
    return [];
  }
  const previousRound = state.rounds[roundPosition - 1];
  return previousRound && Array.isArray(previousRound.events) ? previousRound.events.slice() : [];
}

function buildLatestVitalValueByLabel(events) {
  const latestByLabel = new Map();
  const normalizedEvents = normalizeEventsForTrend(events, "previous_window");

  normalizedEvents.forEach((normalizedEvent) => {
    if (!isVitalEvent(normalizedEvent)) {
      return;
    }
    const label = extractVitalLabel(normalizedEvent);
    const labelKey = normalizeLabel(label);
    if (!labelKey) {
      return;
    }
    const numeric = extractEventNumericValue(normalizedEvent);
    if (!Number.isFinite(numeric)) {
      return;
    }

    const point = {
      timeMs: normalizedEvent.timeMs,
      order: normalizedEvent.order,
      source: normalizedEvent.source,
    };
    const existing = latestByLabel.get(labelKey);
    if (!existing || compareTrendPoints(existing.point, point) < 0) {
      latestByLabel.set(labelKey, {
        value: numeric,
        point,
      });
    }
  });

  return latestByLabel;
}

function getVitalTimelineAnomalyInfo(event, { scope = "generic", previousWindowVitalsByLabel = new Map() } = {}) {
  const normalizedEvent = {
    event,
    source: scope || "timeline",
    order: 0,
    time: safeString(event && (event.time || event.start_time)),
    timeMs: parseEventTimeMs(event),
  };
  if (!isVitalEvent(normalizedEvent)) {
    return null;
  }

  const label = extractVitalLabel(normalizedEvent);
  const labelKey = normalizeLabel(label);
  if (!labelKey) {
    return null;
  }
  const numericValue = extractEventNumericValue(normalizedEvent);
  if (!Number.isFinite(numericValue)) {
    return null;
  }

  const guideline = resolveVitalGuideline(label);
  const classes = [];
  const notes = [];

  if (guideline && Number.isFinite(guideline.low) && Number.isFinite(guideline.high)) {
    const outOfRange = numericValue < guideline.low || numericValue > guideline.high;
    if (outOfRange) {
      classes.push("event-anomaly", "event-anomaly-range");
      const rangeText = `${formatCompactNumber(guideline.low)}-${formatCompactNumber(guideline.high)}`
        + (safeString(guideline.unit) ? ` ${safeString(guideline.unit)}` : "");
      notes.push(`Abnormal vital: outside healthy range (${rangeText})`);
    }
  }

  if (scope === "current" && previousWindowVitalsByLabel instanceof Map) {
    const baseline = previousWindowVitalsByLabel.get(labelKey);
    if (baseline && Number.isFinite(baseline.value)) {
      const delta = numericValue - baseline.value;
      const absDelta = Math.abs(delta);
      const relativeDelta = absDelta / Math.max(Math.abs(baseline.value), 1);
      const absThreshold = resolveVitalTrendAbsoluteThreshold(guideline, baseline.value);
      if (absDelta >= absThreshold && relativeDelta >= VITAL_TREND_RELATIVE_CHANGE_THRESHOLD) {
        classes.push("event-anomaly", delta > 0 ? "event-anomaly-trend-up" : "event-anomaly-trend-down");
        const direction = delta > 0 ? "sharp rise" : "sharp drop";
        const percent = Math.round(relativeDelta * 100);
        notes.push(`Abnormal vital: ${direction} vs previous window (${percent}%)`);
      }
    }
  }

  if (classes.length === 0 || notes.length === 0) {
    return null;
  }
  return {
    classes: Array.from(new Set(classes)),
    note: notes.join(" | "),
  };
}

function resolveVitalTrendAbsoluteThreshold(guideline, baselineValue) {
  if (guideline && Number.isFinite(guideline.low) && Number.isFinite(guideline.high)) {
    const rangeWidth = Math.abs(guideline.high - guideline.low);
    return Math.max(1, rangeWidth * VITAL_TREND_BASELINE_RANGE_FRACTION_THRESHOLD);
  }
  return Math.max(1, Math.abs(baselineValue) * VITAL_TREND_BASELINE_FRACTION_THRESHOLD);
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

    const legacyMatch = line.match(/^(HX\d+)\.\s+(.+)$/i);
    if (legacyMatch) {
      parsed.events.push(parsePromptEventLine(legacyMatch[2], line, legacyMatch[1]));
      return;
    }

    const bulletMatch = line.match(/^-\s+(.+)$/);
    if (bulletMatch) {
      parsed.events.push(parsePromptEventLine(bulletMatch[1], line, ""));
      return;
    }

    if (isPromptRawEventLine(line)) {
      parsed.events.push(parsePromptEventLine(line, line, ""));
    }
  });

  return parsed;
}

function parsePromptCurrentSection(sectionText) {
  const parsed = {
    hasSection: false,
    events: [],
  };

  const text = safeString(sectionText);
  if (!text) {
    return parsed;
  }

  parsed.hasSection = true;
  const lines = text.split("\n").map((line) => safeString(line)).filter((line) => line.length > 0);
  lines.forEach((line) => {
    const legacyMatch = line.match(/^(CW\d+)\.\s+(.+)$/i);
    if (legacyMatch) {
      parsed.events.push(parsePromptEventLine(legacyMatch[2], line, legacyMatch[1]));
      return;
    }

    const bulletMatch = line.match(/^-\s+(.+)$/);
    if (bulletMatch) {
      parsed.events.push(parsePromptEventLine(bulletMatch[1], line, ""));
      return;
    }

    if (isPromptRawEventLine(line)) {
      parsed.events.push(parsePromptEventLine(line, line, ""));
    }
  });
  return parsed;
}

function isPromptRawEventLine(line) {
  const text = safeString(line);
  if (!text) {
    return false;
  }
  return /^(?:\[\d+\]\s+|\d+\s+)?\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)/.test(text);
}

function parsePromptEventLine(eventText, rawLine, eventId = "") {
  const text = safeString(eventText);
  let normalizedEventId = normalizeEventEvidenceId(eventId);
  let body = text;

  const prefixedIdMatch = body.match(/^event_id\s*=\s*(\d+)\s+(.+)$/i);
  if (prefixedIdMatch) {
    normalizedEventId = normalizeEventEvidenceId(prefixedIdMatch[1]) || normalizedEventId;
    body = safeString(prefixedIdMatch[2]);
  }

  const leadingBracketIdMatch = body.match(/^\[(\d+)\]\s+(.+)$/);
  if (leadingBracketIdMatch) {
    const maybeTimestampPayload = safeString(leadingBracketIdMatch[2]);
    if (looksLikeEventTimestamp(maybeTimestampPayload)) {
      normalizedEventId = normalizeEventEvidenceId(leadingBracketIdMatch[1]) || normalizedEventId;
      body = maybeTimestampPayload;
    }
  }

  const leadingNumericIdMatch = body.match(/^(\d+)\s+(.+)$/);
  if (leadingNumericIdMatch) {
    const maybeTimestampPayload = safeString(leadingNumericIdMatch[2]);
    if (looksLikeEventTimestamp(maybeTimestampPayload)) {
      normalizedEventId = normalizeEventEvidenceId(leadingNumericIdMatch[1]) || normalizedEventId;
      body = maybeTimestampPayload;
    }
  }

  const fallback = {
    time: "",
    code: "UNKNOWN",
    code_specifics: body,
    text_value: null,
    numeric_value: null,
    event_id: normalizedEventId,
    raw_line: safeString(rawLine) || body,
  };

  if (!body) {
    return fallback;
  }

  const match = body.match(/^(\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?))\s+([A-Z0-9_]+)\s*(.*)$/);
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
    event_id: normalizedEventId,
    raw_line: safeString(rawLine) || body,
  };
}

function looksLikeEventTimestamp(value) {
  const text = safeString(value);
  if (!text) {
    return false;
  }
  return /^\d{4}-\d{2}-\d{2}(?:[ T]\d{2}:\d{2}(?::\d{2})?)/.test(text);
}

function resolveHistoryHours(round, parsedHistoryHours) {
  const metadata = round && round.windowOutput && isObject(round.windowOutput.window_metadata)
    ? round.windowOutput.window_metadata
    : null;
  const perWindowCandidates = [
    parsedHistoryHours,
    metadata ? metadata.history_hours : null,
    metadata ? metadata.context_history_hours : null,
  ];
  for (const candidate of perWindowCandidates) {
    const value = toNumber(candidate);
    if (Number.isFinite(value)) {
      return value;
    }
  }

  const globalCandidates = [
    state.windowContextsData && state.windowContextsData.history_hours,
    state.windowContextsData && state.windowContextsData.context_history_hours,
    state.predictionsData && state.predictionsData.history_hours,
    state.predictionsData && state.predictionsData.context_history_hours,
    state.windowContextsData
      && isObject(state.windowContextsData.oracle_context)
      && state.windowContextsData.oracle_context.history_hours,
    state.predictionsData
      && isObject(state.predictionsData.oracle_context)
      && state.predictionsData.oracle_context.history_hours,
  ];

  for (const candidate of globalCandidates) {
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

function collectFullTimelineWindowEntries(fullWindowContextsData = state.fullWindowContextsData) {
  if (!isObject(fullWindowContextsData) || !Array.isArray(fullWindowContextsData.window_contexts)) {
    return [];
  }

  const entries = fullWindowContextsData.window_contexts
    .map((item, arrayIndex) => {
      if (!isObject(item)) {
        return null;
      }
      const sourceWindowIndex = resolveSourceWindowIndexFromContextEntry(
        item,
        toInt(item.window_index, arrayIndex + 1) - 1
      );
      if (!Number.isFinite(sourceWindowIndex)) {
        return null;
      }
      const currentEvents = Array.isArray(item.current_events) ? item.current_events : [];
      return {
        sourceWindowIndex,
        arrayIndex,
        currentEvents,
        contextEntry: item,
      };
    })
    .filter((item) => item !== null);

  entries.sort((a, b) => {
    if (a.sourceWindowIndex !== b.sourceWindowIndex) {
      return a.sourceWindowIndex - b.sourceWindowIndex;
    }
    return a.arrayIndex - b.arrayIndex;
  });
  return entries;
}

function collectPriorEventsFromFullTimeline(roundPosition) {
  const currentRound = state.rounds[roundPosition];
  if (!currentRound || !Number.isFinite(currentRound.sourceWindowIndex)) {
    return null;
  }

  const fullEntries = collectFullTimelineWindowEntries();
  if (fullEntries.length === 0) {
    return null;
  }

  const events = [];
  const signatures = new Set();
  fullEntries.forEach((entry) => {
    if (entry.sourceWindowIndex >= currentRound.sourceWindowIndex) {
      return;
    }
    entry.currentEvents.forEach((event) => {
      const signature = buildEventSignature(event);
      if (signatures.has(signature)) {
        return;
      }
      signatures.add(signature);
      events.push(event);
    });
  });
  return events;
}

function getPreviousWindowEventsFromFullTimeline(roundPosition) {
  const currentRound = state.rounds[roundPosition];
  if (!currentRound || !Number.isFinite(currentRound.sourceWindowIndex)) {
    return null;
  }
  const targetSourceWindowIndex = currentRound.sourceWindowIndex - 1;
  if (targetSourceWindowIndex < 0) {
    return [];
  }

  const fullEntries = collectFullTimelineWindowEntries();
  if (fullEntries.length === 0) {
    return null;
  }
  const matched = fullEntries.find((entry) => entry.sourceWindowIndex === targetSourceWindowIndex);
  return matched ? matched.currentEvents.slice() : [];
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

function getFullContextWindowStartMs(contextEntry) {
  if (!isObject(contextEntry)) {
    return null;
  }
  const metadata = isObject(contextEntry.window_metadata) ? contextEntry.window_metadata : {};
  const startCandidate = (
    metadata.window_start_time
    || contextEntry.current_window_start
    || contextEntry.current_window_start_time
    || ""
  );
  return parseEventTimeMs({ time: startCandidate });
}

function getTimelineStartMs() {
  const fullEntries = collectFullTimelineWindowEntries();
  if (fullEntries.length > 0) {
    const fullContextStarts = fullEntries
      .map((entry) => getFullContextWindowStartMs(entry.contextEntry))
      .filter((value) => Number.isFinite(value));
    if (fullContextStarts.length > 0) {
      return Math.min(...fullContextStarts);
    }
  }

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

function buildStayVitalPlotStats(rounds, fullWindowContextsData = null) {
  let safeRounds = Array.isArray(rounds) ? rounds : [];
  const fullEntries = collectFullTimelineWindowEntries(fullWindowContextsData);
  if (fullEntries.length > 0) {
    safeRounds = fullEntries.map((entry) => ({
      windowIndex: entry.sourceWindowIndex + 1,
      events: entry.currentEvents,
    }));
  }

  const byLabel = new Map();
  const orderByLabel = new Map();
  let nextStableOrder = 0;
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
        orderByLabel.set(labelKey, nextStableOrder);
        nextStableOrder += 1;
      }

      const stats = byLabel.get(labelKey);
      stats.pointCount += 1;
      stats.windows.add(round.windowIndex ?? roundPosition);
    });
  });

  return {
    totalWindows: safeRounds.length,
    byLabel,
    orderByLabel,
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

function getNonInvasiveBloodPressureOrderPriority(label) {
  const normalized = normalizeLabel(label);
  if (!/\bnon[\s-]*invasive\s+blood\s*pressure\b/.test(normalized)) {
    return null;
  }
  if (/\bsystolic\b/.test(normalized)) {
    return 0;
  }
  if (/\bdiastolic\b/.test(normalized)) {
    return 1;
  }
  if (/\bmean\b/.test(normalized)) {
    return 2;
  }
  return null;
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

function withOracleEventIds(events) {
  const safeEvents = Array.isArray(events) ? events : [];
  return safeEvents.map((event, index) => {
    const normalizedEvent = isObject(event) ? { ...event } : {};
    const existingId = resolveEventEvidenceId(event);
    normalizedEvent.event_id = existingId || `${index + 1}`;
    return normalizedEvent;
  });
}

function resolveEventEvidenceId(event) {
  if (!isObject(event)) {
    return "";
  }
  const candidates = [
    event.event_id,
    event.evidence_event_id,
    event.event_ref,
    event.event_reference,
  ];
  for (const candidate of candidates) {
    const normalized = normalizeEventEvidenceId(candidate);
    if (normalized) {
      return normalized;
    }
  }
  return "";
}

function normalizeEventEvidenceId(value) {
  const raw = safeString(value);
  if (!raw) {
    return "";
  }
  const compact = raw
    .replace(/[，；]/g, ",")
    .replace(/^[([{]+|[)\].,;:]+$/g, "")
    .replace(/\s+/g, "");

  const prefixedMatch = compact.match(/^event_id[:=]?(\d+)$/i);
  if (prefixedMatch) {
    return `${Number.parseInt(prefixedMatch[1], 10)}`;
  }

  if (/^\d+$/.test(compact)) {
    return `${Number.parseInt(compact, 10)}`;
  }

  const rangeMatch = compact.match(/^(\d+)-(\d+)$/);
  if (!rangeMatch) {
    return "";
  }
  const startNumber = Number.parseInt(rangeMatch[1], 10);
  const endNumber = Number.parseInt(rangeMatch[2], 10);
  if (!Number.isFinite(startNumber) || !Number.isFinite(endNumber) || startNumber < 1 || endNumber < startNumber) {
    return "";
  }
  return `${startNumber}-${endNumber}`;
}

function parseEvidenceEventIdsInput(value) {
  const raw = safeString(value);
  if (!raw) {
    return { valid: [], invalid: [] };
  }

  const tokens = raw
    .split(/[\s,;，；]+/)
    .map((token) => safeString(token))
    .filter((token) => token.length > 0);

  const valid = [];
  const invalid = [];
  const seen = new Set();

  tokens.forEach((token) => {
    const normalized = normalizeEventEvidenceId(token);
    if (!normalized) {
      invalid.push(token);
      return;
    }
    if (!seen.has(normalized)) {
      seen.add(normalized);
      valid.push(normalized);
    }
  });

  return { valid, invalid };
}

function formatHistoryEventLine(event) {
  const eventId = resolveEventEvidenceId(event);
  const time = formatTime(event.time || event.start_time || "");
  const code = safeString(event.code) || "UNKNOWN";
  const description = safeString(event.code_specifics) || safeString(event.text_value) || "-";
  const value = formatEventValue(event);
  const valuePart = value ? ` = ${value}` : "";
  const idPart = eventId ? `${eventId} ` : "";
  return `${idPart}${time} ${code} ${description}${valuePart}`;
}

function buildVitalSeries(historyEvents, currentEvents) {
  const stableOrderByLabel = (
    state.vitalPlotStats
    && state.vitalPlotStats.orderByLabel instanceof Map
  )
    ? state.vitalPlotStats.orderByLabel
    : null;
  const resolveStableOrder = (labelKey) => {
    if (!stableOrderByLabel) {
      return Number.POSITIVE_INFINITY;
    }
    const value = stableOrderByLabel.get(labelKey);
    return Number.isFinite(value) ? value : Number.POSITIVE_INFINITY;
  };

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
      const aStableOrder = resolveStableOrder(a.entry.labelKey);
      const bStableOrder = resolveStableOrder(b.entry.labelKey);
      if (aStableOrder !== bStableOrder) {
        return aStableOrder - bStableOrder;
      }
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
      const aNonInvasiveBpPriority = getNonInvasiveBloodPressureOrderPriority(a.entry.label);
      const bNonInvasiveBpPriority = getNonInvasiveBloodPressureOrderPriority(b.entry.label);
      if (
        aNonInvasiveBpPriority !== null &&
        bNonInvasiveBpPriority !== null &&
        aNonInvasiveBpPriority !== bNonInvasiveBpPriority
      ) {
        return aNonInvasiveBpPriority - bNonInvasiveBpPriority;
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
  const counts = { all: events.length };
  EVENT_TYPE_OPTIONS.forEach((option) => {
    counts[option.value] = 0;
  });
  events.forEach((event) => {
    const type = classifyEventType(event);
    if (Object.prototype.hasOwnProperty.call(counts, type)) {
      counts[type]++;
    }
  });
  return counts;
}

function renderEventTypeFilter({ counts, scope = "current", compact = false }) {
  const filterOptions = [{ value: "all", label: "All", count: counts.all }]
    .concat(EVENT_TYPE_OPTIONS.map((option) => ({
      value: option.value,
      label: option.label,
      count: counts[option.value] || 0,
    })));

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

function attachDischargeSummarySectionListeners() {
  const sectionCards = document.querySelectorAll(".discharge-section-card");
  sectionCards.forEach((card) => {
    card.addEventListener("toggle", (event) => {
      const detailEl = event.currentTarget;
      const round = getCurrentRound();
      if (!round || !detailEl) {
        return;
      }
      const sectionIndex = toInt(detailEl.getAttribute("data-discharge-section-index"), null);
      if (sectionIndex === null) {
        return;
      }
      if (!isObject(state.dischargeSummaryOpenSectionsByRound[round.arrayIndex])) {
        state.dischargeSummaryOpenSectionsByRound[round.arrayIndex] = {};
      }
      state.dischargeSummaryOpenSectionsByRound[round.arrayIndex][sectionIndex] = detailEl.open === true;
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
  const normalizedEvent = isObject(event) ? event : {};
  const codeRaw = safeString(normalizedEvent.code);
  const codeUpper = codeRaw.toUpperCase();
  const specifics = safeString(normalizedEvent.code_specifics);
  const textValue = safeString(normalizedEvent.text_value);
  const combined = normalizeLabel(`${codeRaw} ${specifics} ${textValue}`);

  const mappedByCode = EVENT_CODE_TYPE_MAP[codeUpper] || null;
  const canSubClassify = !mappedByCode || ["vital", "procedure", "lab", "other"].includes(mappedByCode);
  if (codeUpper.startsWith("NOTE_")) {
    return "note";
  }
  if (canSubClassify && VENT_EVENT_PATTERN.test(combined)) {
    return "vent";
  }
  if (canSubClassify && NEURO_EVENT_PATTERN.test(combined)) {
    return "neuro";
  }

  if (mappedByCode) {
    return mappedByCode;
  }

  if (codeUpper.startsWith("META_")) {
    return "other";
  }

  if (DISPOSITION_EVENT_PATTERN.test(combined)) {
    return "disposition";
  }
  if (MEDICATION_EVENT_PATTERN.test(combined)) {
    return "medication";
  }
  if (PROCEDURE_EVENT_PATTERN.test(combined)) {
    return "procedure";
  }
  if (DIAGNOSIS_EVENT_PATTERN.test(combined)) {
    return "diagnosis";
  }
  if (LAB_EVENT_PATTERN.test(combined)) {
    return "lab";
  }
  if (VITAL_EVENT_PATTERN.test(combined)) {
    return "vital";
  }

  return "other";
}

function formatEventSource(source) {
  switch (safeString(source)) {
    case "window_contexts.prompt_sections.current_observation_window":
      return "window_contexts.prompt_sections.current_observation_window";
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
  if (normalized === "acceptable") {
    return 1;
  }
  if (normalized === "best_practice") {
    return 2;
  }
  return 3;
}

function overallToneClass(label) {
  const normalized = normalizeLabel(label);
  if (normalized === "improving" || normalized === "best_practice") {
    return "positive";
  }
  if (normalized === "deteriorating" || normalized === "potentially_harmful") {
    return "danger";
  }
  if (normalized === "stable" || normalized === "acceptable" || normalized === "suboptimal") {
    return "info";
  }
  if (normalized === "insufficient_data" || normalized === "not_enough_context") {
    return "neutral";
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

function getPatientContextInfo(predictionsData, windowMetadata = {}) {
  const trajectoryMetadata = isObject(predictionsData && predictionsData.trajectory_metadata)
    ? predictionsData.trajectory_metadata
    : {};
  const windowMeta = isObject(windowMetadata) ? windowMetadata : {};

  const ageCandidates = [
    trajectoryMetadata.age,
    trajectoryMetadata.age_at_admission,
    windowMeta.age,
  ];
  let ageText = "-";
  for (const candidate of ageCandidates) {
    const ageValue = toNumber(candidate);
    if (Number.isFinite(ageValue)) {
      ageText = `${formatCompactNumber(ageValue)} years`;
      break;
    }
  }

  const genderRaw = safeString(trajectoryMetadata.gender) || safeString(windowMeta.gender);
  let genderText = "-";
  if (genderRaw) {
    const normalized = normalizeLabel(genderRaw);
    if (!["none", "nan", "null", "nat", "unknown"].includes(normalized)) {
      if (normalized === "m" || normalized === "male") {
        genderText = "Male";
      } else if (normalized === "f" || normalized === "female") {
        genderText = "Female";
      } else {
        genderText = genderRaw;
      }
    }
  }

  const stayHourCandidates = [
    trajectoryMetadata.total_icu_stay_hours,
    trajectoryMetadata.icu_duration_hours,
    windowMeta.total_icu_stay_hours,
    windowMeta.total_icu_duration_hours,
  ];
  let totalIcuStayHoursText = "-";
  for (const candidate of stayHourCandidates) {
    const hours = toNumber(candidate);
    if (Number.isFinite(hours)) {
      totalIcuStayHoursText = `${formatCompactNumber(hours)} hours`;
      break;
    }
  }

  return {
    age: ageText,
    gender: genderText,
    totalIcuStayHours: totalIcuStayHoursText,
    icuOutcome: getIcuOutcomeInfo(predictionsData, windowMeta),
  };
}

function getIcuOutcomeInfo(predictionsData, windowMetadata = {}) {
  const trajectoryMetadata = isObject(predictionsData && predictionsData.trajectory_metadata)
    ? predictionsData.trajectory_metadata
    : {};
  const windowMeta = isObject(windowMetadata) ? windowMetadata : {};

  const explicitOutcome = safeString(trajectoryMetadata.icu_outcome) || safeString(windowMeta.icu_outcome);
  const explicitNormalized = normalizeLabel(explicitOutcome);
  if (["survived after icu", "survived", "alive"].includes(explicitNormalized)) {
    return { label: "Survived after ICU", toneClass: "survived" };
  }
  if (["died after icu", "died", "dead", "deceased"].includes(explicitNormalized)) {
    return { label: "Died after ICU", toneClass: "died" };
  }

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

  const fallbackSurvived = windowMeta.survived;
  if (fallbackSurvived === true) {
    return { label: "Survived after ICU", toneClass: "survived" };
  }
  if (fallbackSurvived === false) {
    return { label: "Died after ICU", toneClass: "died" };
  }

  const fallbackNormalized = normalizeLabel(fallbackSurvived);
  if (["true", "1", "yes", "y", "survived", "alive"].includes(fallbackNormalized)) {
    return { label: "Survived after ICU", toneClass: "survived" };
  }
  if (["false", "0", "no", "n", "died", "dead", "deceased"].includes(fallbackNormalized)) {
    return { label: "Died after ICU", toneClass: "died" };
  }

  if (safeString(trajectoryMetadata.death_time) || safeString(windowMeta.death_time)) {
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
        const parsed = parseJsonWithRecovery(String(reader.result));
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

function parseJsonWithRecovery(rawText) {
  const text = String(rawText || "");
  try {
    return JSON.parse(text);
  } catch (originalError) {
    const sanitized = sanitizeNonStandardJsonNumbers(text);
    if (sanitized.replaced === 0) {
      throw originalError;
    }
    try {
      return JSON.parse(sanitized.text);
    } catch (recoveryError) {
      throw originalError;
    }
  }
}

function sanitizeNonStandardJsonNumbers(text) {
  const input = String(text || "");
  if (!input) {
    return { text: input, replaced: 0 };
  }

  let output = "";
  let replaced = 0;
  let inString = false;
  let escaped = false;

  for (let index = 0; index < input.length;) {
    const char = input[index];

    if (inString) {
      output += char;
      if (escaped) {
        escaped = false;
      } else if (char === "\\") {
        escaped = true;
      } else if (char === "\"") {
        inString = false;
      }
      index += 1;
      continue;
    }

    if (char === "\"") {
      inString = true;
      output += char;
      index += 1;
      continue;
    }

    if (matchesJsonNumberTokenAt(input, index, "-Infinity")) {
      output += "null";
      replaced += 1;
      index += "-Infinity".length;
      continue;
    }
    if (matchesJsonNumberTokenAt(input, index, "Infinity")) {
      output += "null";
      replaced += 1;
      index += "Infinity".length;
      continue;
    }
    if (matchesJsonNumberTokenAt(input, index, "NaN")) {
      output += "null";
      replaced += 1;
      index += "NaN".length;
      continue;
    }

    output += char;
    index += 1;
  }

  return { text: output, replaced };
}

function matchesJsonNumberTokenAt(input, index, token) {
  if (!input.startsWith(token, index)) {
    return false;
  }
  const left = index > 0 ? input[index - 1] : "";
  const right = input[index + token.length] || "";
  return isJsonTokenBoundaryChar(left) && isJsonTokenBoundaryChar(right);
}

function isJsonTokenBoundaryChar(char) {
  if (!char) {
    return true;
  }
  return /[\s,[\]{}:]/.test(char);
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

function normalizeConfidence(value) {
  const normalized = normalizeLabel(value);
  if (normalized === "high" || normalized === "medium" || normalized === "low") {
    return normalized;
  }
  return null;
}

function normalizeAdditionalActionCategory(value) {
  const raw = safeString(value);
  if (!raw) {
    return "";
  }
  return ICU_ACTION_CATEGORIES.includes(raw) ? raw : "";
}

function normalizeRiskNameList(items) {
  const safeItems = Array.isArray(items) ? items : [];
  const normalized = [];
  const seen = new Set();
  safeItems.forEach((item) => {
    const riskName = safeString(item);
    if (!riskName) {
      return;
    }
    const riskKey = normalizeLabel(riskName);
    if (seen.has(riskKey)) {
      return;
    }
    seen.add(riskKey);
    normalized.push(riskName);
  });
  return normalized;
}

function normalizeRedFlagActionList(items) {
  const safeItems = Array.isArray(items) ? items : [];
  const normalized = [];
  const seen = new Set();
  safeItems.forEach((item) => {
    const actionText = safeString(item);
    if (!actionText) {
      return;
    }
    const actionKey = normalizeLabel(actionText);
    if (seen.has(actionKey)) {
      return;
    }
    seen.add(actionKey);
    normalized.push(actionText);
  });
  return normalized;
}

function getOracleActiveRiskNames(patientAssessment) {
  const activeRisks = (isObject(patientAssessment) && Array.isArray(patientAssessment.active_risks))
    ? patientAssessment.active_risks
    : [];
  const riskNames = [];
  const seen = new Set();
  activeRisks.forEach((risk) => {
    const riskName = isObject(risk) ? safeString(risk.risk_name) : safeString(risk);
    if (!riskName) {
      return;
    }
    const riskKey = normalizeLabel(riskName);
    if (seen.has(riskKey)) {
      return;
    }
    seen.add(riskKey);
    riskNames.push(riskName);
  });
  return riskNames;
}

function getOracleContraindicatedActionNames(recommendations) {
  const safeRecommendations = Array.isArray(recommendations) ? recommendations : [];
  const actionNames = [];
  const seen = new Set();
  safeRecommendations.forEach((rec) => {
    const actionText = isObject(rec)
      ? safeString(rec.action || rec.contraindicated_action || rec.action_name)
      : safeString(rec);
    if (!actionText) {
      return;
    }
    const actionKey = normalizeLabel(actionText);
    if (seen.has(actionKey)) {
      return;
    }
    seen.add(actionKey);
    actionNames.push(actionText);
  });
  return actionNames;
}

function ensureStringList(container, key) {
  if (!isObject(container)) {
    return [];
  }
  const source = Array.isArray(container[key]) ? container[key] : [];
  const normalized = source.map((item) => (item === null || item === undefined ? "" : String(item)));
  container[key] = normalized;
  return normalized;
}

function buildTextConfidenceItem(text = "", confidence = null, evidenceEventIdsText = "", category = "") {
  return {
    text: safeString(text),
    confidence: normalizeConfidence(confidence),
    evidence_event_ids_text: safeString(evidenceEventIdsText),
    category: normalizeAdditionalActionCategory(category),
  };
}

function normalizeTextConfidenceItem(item) {
  if (isObject(item)) {
    const textValue = safeString(item.text || item.action || item.insight || item.value || item.description);
    let evidenceEventIdsText = safeString(item.evidence_event_ids_text || "");
    if (!evidenceEventIdsText) {
      if (Array.isArray(item.evidence_event_ids)) {
        evidenceEventIdsText = item.evidence_event_ids.map((id) => safeString(id)).filter((id) => id.length > 0).join(", ");
      } else if (Array.isArray(item.event_evidence_ids)) {
        evidenceEventIdsText = item.event_evidence_ids.map((id) => safeString(id)).filter((id) => id.length > 0).join(", ");
      } else if (Array.isArray(item.evidence_event_refs)) {
        evidenceEventIdsText = item.evidence_event_refs.map((id) => safeString(id)).filter((id) => id.length > 0).join(", ");
      } else {
        evidenceEventIdsText = safeString(item.evidence_event_ids || item.event_evidence_ids || item.evidence_event_refs || "");
      }
    }
    const categoryValue = (
      item.category
      || item.action_category
      || item.icu_action_category
      || item.category_label
    );
    return buildTextConfidenceItem(textValue, item.confidence, evidenceEventIdsText, categoryValue);
  }
  return buildTextConfidenceItem(safeString(item), null, "", "");
}

function getTextConfidenceItems(items) {
  const safeItems = Array.isArray(items) ? items : [];
  return safeItems.map((item) => normalizeTextConfidenceItem(item));
}

function ensureTextConfidenceList(container, key) {
  if (!isObject(container)) {
    return [];
  }
  const normalized = getTextConfidenceItems(container[key]);
  container[key] = normalized;
  return normalized;
}

function exportTextConfidenceItems(items, keyName) {
  return getTextConfidenceItems(items)
    .filter((item) => item.text.length > 0)
    .map((item) => ({
      [keyName]: item.text,
      confidence: item.confidence,
    }));
}

function exportAdditionalRecommendationItems(items) {
  return getTextConfidenceItems(items)
    .filter((item) => item.text.length > 0)
    .map((item) => {
      const evidenceParse = parseEvidenceEventIdsInput(item.evidence_event_ids_text || "");
      return {
        icu_action_category: normalizeAdditionalActionCategory(item.category),
        action: item.text,
        confidence: item.confidence,
        evidence_event_ids: evidenceParse.valid,
      };
    });
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

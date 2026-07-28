/**
 * Browser-side controller for the COABarrett document review application.
 *
 * WHAT THIS FILE DOES FOR THE APP
 * --------------------------------
 * Python performs OCR, metadata extraction, SDAT lookups, PDF writing, and file
 * movement. This JavaScript file controls what the reviewer sees and sends the
 * reviewer's choices to the Flask routes in app.py.
 *
 * The browser keeps a temporary `state` object so it can draw the document queue,
 * selected PDF, and metadata form without reloading the whole page. The saved
 * version still lives on the server in `.review_state/documents.json`; after every
 * important request, the server response replaces or updates this temporary copy.
 *
 * NORMAL DATA FLOW
 * ----------------
 *   User clicks or edits a control
 *       -> this file builds a JSON request
 *       -> app.py validates and processes it
 *       -> Flask returns updated settings/documents
 *       -> applyState() or replaceDocument() updates the browser
 *       -> renderList() and renderSelectedDocument() redraw the interface
 *
 * TWO POLLING LOOPS ARE INTENTIONAL
 * ---------------------------------
 * `/api/scan-progress` supplies scan messages and measured stage timings.
 * During Mass Scan, `/api/state` is also polled so each finished PDF appears for
 * review before the entire input folder is complete.
 */

const state = { documents: [], settings: {}, selectedId: null };

/**
 * Find one HTML element by ID.
 *
 * The review interface accesses the same controls many times. This short alias keeps
 * those lookups readable while still returning the normal `HTMLElement` (or null)
 * produced by `document.getElementById()`.
 *
 * @param {string} id HTML element ID from templates/index.html.
 * @returns {HTMLElement|null}
 */
const $ = (id) => document.getElementById(id);

const fields = {
  inputFolder: $('inputFolder'),
  outputFolder: $('outputFolder'),
  configPath: $('configPath'),
  projectCode: $('projectCode'),
  county: $('county'),
  dpi: $('dpi'),
  ocrDevice: $('ocrDevice'),
  scanMode: $('scanMode'),
  inPlace: $('inPlace'),
  lot: $('lot'),
  address: $('address'),
  taxMap: $('taxMap'),
  parcel: $('parcel'),
  taxId: $('taxId'),
  section: $('section'),
  editProjectCode: $('editProjectCode'),
  editDocumentType: $('editDocumentType'),
  customDocumentType: $('customDocumentType'),
  copyFile: $('copyFile'),
  saveText: $('saveText'),
};

let browseTargetField = null;
const folderBrowserElements = {
  inputFolder: {
    panel: $('inputFolderBrowser'),
    currentPath: $('inputBrowseCurrentPath'),
    list: $('inputBrowseFolderList'),
    useButton: $('inputUseFolderButton'),
  },
  outputFolder: {
    panel: $('outputFolderBrowser'),
    currentPath: $('outputBrowseCurrentPath'),
    list: $('outputBrowseFolderList'),
    useButton: $('outputUseFolderButton'),
  },
};

let scanProgressTimer = null;
let scanElapsedTimer = null;
let scanStartedAt = null;
let renderedProgressCount = 0;
let liveStateTimer = null;

// ============================================================================
// SCAN PROGRESS DISPLAY AND TIMERS
// ============================================================================

/**
 * Prepare the progress card for a brand-new scan.
 *
 * This function changes only browser elements: it reveals the panel, removes any
 * previous failure styling, resets the elapsed time, clears old messages, and sets
 * `renderedProgressCount` back to zero. It is called before the scan POST begins so
 * messages from an earlier run cannot be mixed with the new run.
 */
function resetScanProgressPanel() {
  const panel = $('scanProgressPanel');
  panel.classList.remove('hidden', 'failed');
  $('scanProgressStatus').textContent = 'Starting scan...';
  $('scanElapsed').textContent = '0.0 s';
  $('scanProgressMessages').innerHTML = '';
  renderedProgressCount = 0;
}

/**
 * Draw one progress snapshot returned by `GET /api/scan-progress`.
 *
 * The server returns the complete message history on every poll. To avoid drawing
 * duplicate rows, this function starts at `renderedProgressCount` and appends only
 * messages the browser has not displayed yet. It also updates failure styling and
 * the latest status message. When the local timer is inactive, it uses the elapsed
 * value measured by Python.
 *
 * @param {Object} data Current progress state created by scan_status.py.
 */
function renderScanProgress(data) {
  const panel = $('scanProgressPanel');
  panel.classList.toggle('failed', Boolean(data.failed));
  // The elapsed timer is updated locally by the browser so a delayed Flask
  // progress response cannot freeze or jump the visible timer. The server's
  // elapsed value is only used when no local scan clock is active.
  if (scanStartedAt === null) {
    $('scanElapsed').textContent = `${Number(data.elapsed || 0).toFixed(1)} s`;
  }

  const messages = data.messages || [];
  const container = $('scanProgressMessages');
  for (let index = renderedProgressCount; index < messages.length; index += 1) {
    const message = messages[index];
    const row = document.createElement('div');
    row.className = 'scan-progress-message';

    const elapsed = document.createElement('span');
    elapsed.className = 'scan-progress-time';
    elapsed.textContent = `${Number(message.elapsed || 0).toFixed(1)} s`;

    const text = document.createElement('span');
    text.textContent = message.text || '';
    row.append(elapsed, text);
    container.appendChild(row);
  }
  renderedProgressCount = messages.length;
  if (messages.length) $('scanProgressStatus').textContent = messages[messages.length - 1].text;
  container.scrollTop = container.scrollHeight;
}

/**
 * Perform one scan-progress polling request.
 *
 * `requestJson()` fetches the latest server snapshot and `renderScanProgress()`
 * paints it. A temporary polling failure is deliberately ignored because the main
 * `/api/scan` request reports the actionable error; losing one progress refresh
 * should not interrupt OCR that is already running on the server.
 *
 * @returns {Promise<void>}
 */
async function pollScanProgress() {
  try {
    renderScanProgress(await requestJson('/api/scan-progress'));
  } catch (_error) {
    // The main scan request reports actionable errors. Keep polling quietly.
  }
}

/**
 * Keep the visible scan timer moving smoothly between server responses.
 *
 * The elapsed time is always recalculated from `scanStartedAt`, rather than adding
 * 0.1 seconds on every interval. Recalculating prevents timer drift when the browser
 * is busy and an interval callback runs late. This timer is visual only; Python's
 * performance timings remain the authoritative measurements.
 */
function updateLocalScanElapsed() {
  if (scanStartedAt === null) return;

  // Date.now() measures from the original start time rather than incrementing a
  // counter. This prevents timer drift when the browser or OCR work delays an
  // individual interval callback.
  const elapsedSeconds = Math.max(0, (Date.now() - scanStartedAt) / 1000);
  $('scanElapsed').textContent = `${elapsedSeconds.toFixed(1)} s`;
}

/**
 * Start the two browser timers used while a scan is active.
 *
 * One interval asks Flask for real progress every 300 ms. The second updates only
 * the visible elapsed clock every 100 ms. Existing intervals are cleared first so
 * clicking Scan again can never create multiple polling loops.
 */
function startScanProgressPolling() {
  resetScanProgressPanel();

  clearInterval(scanProgressTimer);
  clearInterval(scanElapsedTimer);

  scanStartedAt = Date.now();
  updateLocalScanElapsed();

  // Progress information still comes from Flask, but elapsed time is maintained
  // independently in the browser so slow progress responses cannot stall it.
  pollScanProgress();
  scanProgressTimer = setInterval(pollScanProgress, 300);
  scanElapsedTimer = setInterval(updateLocalScanElapsed, 100);
}

/**
 * Stop scan-progress timers and paint the final server snapshot.
 *
 * The last local elapsed value is displayed before `scanStartedAt` is cleared, then
 * one final poll captures completion or failure messages produced near the end of
 * the scan. The caller passes `failed: true` when the main scan request rejected.
 *
 * @param {Object} options Stop options.
 * @param {boolean} [options.failed=false] Keep the panel styled as a failure.
 * @returns {Promise<void>}
 */
async function stopScanProgressPolling({ failed = false } = {}) {
  clearInterval(scanProgressTimer);
  clearInterval(scanElapsedTimer);
  scanProgressTimer = null;
  scanElapsedTimer = null;

  // Paint the final locally measured time before disabling the local clock.
  updateLocalScanElapsed();
  scanStartedAt = null;

  await pollScanProgress();
  if (failed) $('scanProgressPanel').classList.add('failed');
  setTimeout(() => $('scanProgressPanel').classList.add('hidden'), failed ? 0 : 0);
}

// ============================================================================
// SERVER-BACKED FOLDER PICKER
// ============================================================================

/**
 * Open the app's folder picker for either the input or output folder field.
 *
 * Web pages cannot browse the Windows filesystem directly. This function records
 * which field is being edited, hides the other picker, and asks `loadBrowseFolder()`
 * to request a directory listing from Flask. The selected path is written back to
 * the field identified by `targetFieldId`.
 *
 * @param {'inputFolder'|'outputFolder'} targetFieldId Field that will receive the path.
 * @param {string} [startPath=''] Optional folder to display first.
 * @returns {Promise<void>}
 */
async function openFolderBrowser(targetFieldId, startPath = '') {
  const browser = folderBrowserElements[targetFieldId];
  if (!browser) return;

  browseTargetField = targetFieldId;
  Object.entries(folderBrowserElements).forEach(([fieldId, elements]) => {
    elements.panel.classList.toggle('hidden', fieldId !== targetFieldId);
  });

  await loadBrowseFolder(startPath || fields[targetFieldId].value, targetFieldId);
  browser.panel.classList.remove('hidden');
}

/**
 * Close one folder-picker panel and clear its active target.
 *
 * Checking the target ID prevents the input-folder picker from accidentally clearing
 * the output-folder selection context, or vice versa.
 *
 * @param {'inputFolder'|'outputFolder'} targetFieldId Picker to close.
 */
function closeFolderBrowser(targetFieldId) {
  const browser = folderBrowserElements[targetFieldId];
  if (browser) browser.panel.classList.add('hidden');
  if (browseTargetField === targetFieldId) browseTargetField = null;
}

/**
 * Display one level of the server's filesystem in the custom folder picker.
 *
 * The function requests `/api/browse-folders`, clears the old buttons, adds a `..`
 * button for the parent, and creates one button for every child directory. Clicking
 * a directory calls this function again, which is how the reviewer moves through
 * the tree. Clicking “Use this folder” writes the current path into the target form
 * field and immediately persists an output-folder change for an active review batch.
 *
 * @param {string} path Directory path Flask should list.
 * @param {'inputFolder'|'outputFolder'|null} [targetFieldId=browseTargetField]
 * @returns {Promise<void>}
 */
async function loadBrowseFolder(path, targetFieldId = browseTargetField) {
  const browser = folderBrowserElements[targetFieldId];
  if (!browser) return;

  const browseUrl = `/api/browse-folders?path=${encodeURIComponent(path || '')}`;
  const data = await requestJson(browseUrl);

  browser.currentPath.textContent = data.current;
  browser.list.innerHTML = '';

  if (data.parent) {
    const up = document.createElement('button');
    up.type = 'button';
    up.textContent = '..';
    up.title = 'Go to parent folder';
    up.onclick = () => loadBrowseFolder(data.parent, targetFieldId);
    browser.list.appendChild(up);
  }

  data.folders.forEach((folder) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = folder.name;
    button.title = folder.path;
    button.onclick = () => loadBrowseFolder(folder.path, targetFieldId);
    browser.list.appendChild(button);
  });

  browser.useButton.onclick = async () => {
    fields[targetFieldId].value = data.current;
    closeFolderBrowser(targetFieldId);
    if (targetFieldId === 'outputFolder' && state.documents.length) {
      try {
        await saveOutputFolder();
      } catch (error) {
        showToast(error.message, true);
      }
    }
  };
}

// ============================================================================
// USER FEEDBACK AND HTTP COMMUNICATION
// ============================================================================

/**
 * Show a temporary success or error message without interrupting the workflow.
 *
 * All actions reuse one toast element, so frequent autosaves do not leave a stack
 * of old notifications. Assigning the class also removes the previous `hidden`
 * class; a timeout hides the message after 3.8 seconds.
 *
 * @param {string} message Text the reviewer should see.
 * @param {boolean} [isError=false] Apply error styling when true.
 */
function showToast(message, isError = false) {
  const toast = $('toast');
  toast.textContent = message;
  toast.className = `toast ${isError ? 'error' : ''}`;
  setTimeout(() => toast.classList.add('hidden'), 3800);
}


/**
 * Send an API request using the response and error rules shared by the whole page.
 *
 * This is the front end's single networking gateway. It applies the JSON content
 * type, lets the caller override options such as method/body, reads the response as
 * text, and delegates decoding to `parseJsonResponse()`. Non-success HTTP statuses
 * become normal JavaScript `Error` objects, allowing every caller to use the same
 * try/catch and toast pattern.
 *
 * @param {string} url Flask route to request.
 * @param {RequestInit} [options={}] Fetch options such as method and body.
 * @returns {Promise<Object>} Decoded response, or an empty object for an empty body.
 * @throws {Error} When the server reports failure or returns unreadable data.
 */
async function requestJson(url, options = {}) {
  const response = await fetch(url, {
    headers: { 'Content-Type': 'application/json' },
    ...options,
  });

  const bodyText = await response.text();
  const data = parseJsonResponse(bodyText, response);

  if (!response.ok) {
    throw new Error(data?.error || `Request failed with ${response.status}`);
  }

  return data || {};
}

/**
 * Decode a Flask response body and preserve useful errors when it is not JSON.
 *
 * Successful API routes normally return JSON. Flask or a proxy may instead return
 * an HTML error page. If JSON parsing fails, HTML tags and repeated whitespace are
 * removed so the reviewer sees a readable server message instead of `Unexpected
 * token <`.
 *
 * @param {string} bodyText Raw response body read by `requestJson()`.
 * @param {Response} response Fetch response, used for status fallback text.
 * @returns {Object|null} Parsed JSON, or null for an empty response.
 * @throws {Error} When a nonempty body cannot be parsed as JSON.
 */
function parseJsonResponse(bodyText, response) {
  if (!bodyText) return null;

  try {
    return JSON.parse(bodyText);
  } catch (_error) {
    const plainText = bodyText.replace(/<[^>]*>/g, ' ').replace(/\s+/g, ' ').trim();
    throw new Error(plainText || `Server returned ${response.status} ${response.statusText}`);
  }
}

/**
 * Translate a saved document status into the label shown in the queue and review pane.
 *
 * Python stores machine-friendly values such as `needs_review` and `lookup_only`.
 * Keeping the display translation here lets the API retain stable values while the
 * interface uses spacing and capitalization that make sense to a person.
 *
 * @param {Object} document Review record returned by Flask.
 * @returns {string} Human-readable status label.
 */
function statusLabel(document) {
  const labels = { filed: 'Filed', ready: 'Ready', needs_review: 'Needs review', lookup_only: 'Lookup only' };
  return labels[document.status] || 'Needs review';
}

/**
 * Return the full review record represented by `state.selectedId`.
 *
 * The queue stores only an ID as the current selection because document objects are
 * replaced after saves and polling. Looking up the object each time avoids holding a
 * stale reference after the server sends a newer version of that record.
 *
 * @returns {Object|undefined} Selected document, or undefined when none is available.
 */
function selectedDocument() {
  return state.documents.find((document) => document.id === state.selectedId);
}

/**
 * Replace one updated record inside the browser's local queue.
 *
 * A single-document PATCH may return only the changed document rather than the full
 * application state. This function finds that record by stable ID and replaces it in
 * place, preserving queue order and the current selection.
 *
 * @param {Object} updatedDocument Authoritative record returned by Flask.
 */
function replaceDocument(updatedDocument) {
  const index = state.documents.findIndex((document) => document.id === updatedDocument.id);
  if (index >= 0) state.documents[index] = updatedDocument;
}

/**
 * Prevent duplicate submissions while an asynchronous action is running.
 *
 * Scan and batch-filing operations can take long enough for a user to click twice.
 * Disabling the button protects the server from duplicate work, while swapping the
 * label explains what is happening. Callers restore the ready state in `finally`.
 *
 * @param {HTMLButtonElement} button Button being controlled.
 * @param {boolean} isLoading Whether its action is currently running.
 * @param {string} loadingText Label shown while disabled.
 * @param {string} readyText Label shown when enabled.
 */
function setButtonLoading(button, isLoading, loadingText, readyText) {
  button.disabled = isLoading;
  button.textContent = isLoading ? loadingText : readyText;
}

/** Centralized document validation used by the warning banner, document
 * queue, and review form. Keeping all three views on the same validation result
 * prevents one part of the interface from reporting different issues than
 * another. */
const REQUIRED_METADATA_FIELDS = [
  { key: 'lot', label: 'Lot' },
  { key: 'address', label: 'Address' },
  { key: 'project_code', label: 'Project code' },
  { key: 'document_type', label: 'Document type' },
  { key: 'tax_map', label: 'Tax map' },
  { key: 'parcel', label: 'Parcel' },
  { key: 'tax_id', label: 'Tax ID' },
];

// ============================================================================
// CENTRALIZED REVIEW VALIDATION
// ============================================================================

/**
 * Decide whether a metadata value should count as missing in the review interface.
 *
 * OCR and Python use placeholders such as `Unknown Lot`, `Project`, and `Document`.
 * A nonempty placeholder is still unusable for filing, so this function mirrors the
 * server's rules before highlighting fields or enabling workflow decisions.
 *
 * @param {string} key Metadata field name.
 * @param {*} value Value currently stored for that field.
 * @returns {boolean} True when the reviewer still needs to supply a real value.
 */
function isMissingMetadataValue(key, value) {
  const normalized = String(value ?? '').trim();
  if (!normalized) return true;

  const lowered = normalized.toLowerCase();
  if (lowered.startsWith('unknown')) return true;
  if (key === 'project_code' && lowered === 'project') return true;
  if (key === 'document_type' && lowered === 'document') return true;
  return false;
}

/**
 * Convert several server warning/error formats into one clean string array.
 *
 * Older and newer records may store an issue as one string, an array of strings, or
 * an object with `message`, `error`, or `detail`. Normalizing at this boundary keeps
 * the validation and rendering code simple and backward-compatible with saved state.
 *
 * @param {*} value Warning or error value from a document record.
 * @returns {string[]} Nonempty messages ready to display.
 */
function normalizeIssueMessages(value) {
  if (!value) return [];
  if (Array.isArray(value)) {
    return value
      .map((item) => typeof item === 'string' ? item : item?.message || item?.error || JSON.stringify(item))
      .map((item) => String(item || '').trim())
      .filter(Boolean);
  }
  if (typeof value === 'object') {
    const message = value.message || value.error || value.detail;
    return message ? [String(message).trim()] : [];
  }
  const message = String(value).trim();
  return message ? [message] : [];
}

/**
 * Safely read and trim one document's classified or reviewer-selected type.
 *
 * Centralizing this small lookup ensures duplicate detection treats missing metadata
 * objects and extra whitespace the same way everywhere.
 *
 * @param {Object} document Review record.
 * @returns {string} Trimmed document type, or an empty string.
 */
function documentTypeValue(document) {
  return String(document?.metadata?.document_type || '').trim();
}

/**
 * Precompute batch-wide facts needed to validate every queue item consistently.
 *
 * In Batch mode, two permanent drawings with the same document type are suspicious,
 * so documents are grouped by a case-insensitive type and duplicate IDs are recorded.
 * Lookup-only SDAT sheets are excluded. Mass mode intentionally skips this rule because
 * each PDF is an independent job and repeated types are expected.
 *
 * Computing these groups once per render avoids repeating the same whole-queue scan for
 * every individual document.
 *
 * @param {Object[]} [documents=state.documents] Queue to analyze.
 * @returns {{duplicateIds:Set, duplicateTypesById:Map, scanMode:string}}
 */
function buildValidationContext(documents = state.documents || []) {
  const duplicateIds = new Set();
  const duplicateTypesById = new Map();
  const scanMode = String(state.settings?.scan_mode || 'batch').toLowerCase();

  // Duplicate types are intentionally ignored in Mass mode because each PDF is
  // treated as an independent filing job.
  if (scanMode !== 'mass') {
    const typeGroups = new Map();
    documents
      .filter((document) => !document.is_lookup_document)
      .forEach((document) => {
        const type = documentTypeValue(document);
        if (isMissingMetadataValue('document_type', type)) return;
        const key = type.toLowerCase();
        if (!typeGroups.has(key)) typeGroups.set(key, { type, documents: [] });
        typeGroups.get(key).documents.push(document);
      });

    typeGroups.forEach(({ type, documents: groupedDocuments }) => {
      if (groupedDocuments.length < 2) return;
      groupedDocuments.forEach((document) => {
        duplicateIds.add(document.id);
        duplicateTypesById.set(document.id, type);
      });
    });
  }

  return { duplicateIds, duplicateTypesById, scanMode };
}

/**
 * Build the complete review-readiness result for one document.
 *
 * The function checks all required metadata, adds Batch-mode duplicate-type problems,
 * normalizes warnings/errors supplied by Python, and converts failure statuses into an
 * explicit error. Its returned field set drives red highlights, while the issue list
 * drives queue tooltips and severity styling. Using one result prevents the queue and
 * form from disagreeing about what needs attention.
 *
 * @param {Object} document Record being validated.
 * @param {Object} [context=buildValidationContext()] Precomputed batch facts.
 * @returns {Object} Missing fields, issues, severity, and convenience flags.
 */
function getDocumentValidationState(document, context = buildValidationContext()) {
  const metadata = document?.metadata || {};
  const missingFields = REQUIRED_METADATA_FIELDS.filter(({ key }) =>
    isMissingMetadataValue(key, metadata[key])
  );
  const duplicateType = context.duplicateTypesById.get(document?.id) || null;
  const warnings = [
    ...normalizeIssueMessages(document?.warnings),
    ...normalizeIssueMessages(document?.warning),
  ];
  const errors = [
    ...normalizeIssueMessages(document?.errors),
    ...normalizeIssueMessages(document?.error),
  ];
  const status = String(document?.status || '').toLowerCase();
  const statusError = ['error', 'failed', 'failure'].includes(status)
    ? `Document status: ${status}`
    : null;
  if (statusError) errors.push(statusError);

  const issues = [];
  if (duplicateType) issues.push({ kind: 'duplicate', message: `Duplicate document type “${duplicateType}”` });
  if (missingFields.length) {
    issues.push({
      kind: 'missing',
      message: `Missing ${missingFields.map(({ label }) => label).join(', ')}`,
      fields: missingFields.map(({ key }) => key),
    });
  }
  warnings.forEach((message) => issues.push({ kind: 'warning', message }));
  errors.forEach((message) => issues.push({ kind: 'error', message }));

  return {
    missingFields,
    missingFieldKeys: new Set(missingFields.map(({ key }) => key)),
    duplicateType,
    warnings,
    errors,
    issues,
    hasIssues: issues.length > 0,
    severity: errors.length || duplicateType ? 'error' : issues.length ? 'warning' : 'none',
  };
}

/**
 * Create the short label used for a document button in the review queue.
 *
 * The label combines document type and lot because those are the quickest way to tell
 * related engineering drawings apart. Missing placeholders are converted to explicit
 * `Unknown` text instead of being shown as if they were valid metadata.
 *
 * @param {Object} item Review record.
 * @returns {string} Queue label such as `Site Plan - Lot 104`.
 */
function suggestedDocumentLabel(item) {
  const rawType = String(item.metadata?.document_type || '').trim();
  const rawLot = String(item.metadata?.lot || '').trim();
  const type = isMissingMetadataValue('document_type', rawType) ? 'Unknown Type' : rawType;
  const lot = isMissingMetadataValue('lot', rawLot) ? 'Unknown Lot' : `Lot ${rawLot}`;
  return `${type} - ${lot}`;
}

// ============================================================================
// DOCUMENT QUEUE, PDF VIEWER, AND REVIEW FORM RENDERING
// ============================================================================

/**
 * Rebuild the left-hand document queue from the current browser state.
 *
 * For each record, this function calculates validation, creates a button, marks the
 * active selection, and attaches a click handler that calls `selectDocument()`. When
 * no records remain, it switches to the empty-state screen and loads the application's
 * missing-PDF placeholder into the viewer.
 *
 * Rebuilding is simpler and safer than trying to patch many individual DOM rows after
 * scans, saves, and filing actions that may all change the queue.
 */
function renderList() {
  const list = $('documentList');
  list.innerHTML = '';
  const visibleDocuments = state.documents || [];
  $('queueCount').textContent = String(visibleDocuments.length);
  const validationContext = buildValidationContext(visibleDocuments);

  visibleDocuments.forEach((item) => {
    const validation = getDocumentValidationState(item, validationContext);
    const button = document.createElement('button');
    button.className = `doc-row ${item.id === state.selectedId ? 'active' : ''} ${validation.hasIssues ? 'has-issues' : ''}`;
    button.setAttribute('aria-label', `${suggestedDocumentLabel(item)}${validation.hasIssues ? ', needs attention' : ''}`);
    if (validation.hasIssues) button.title = validation.issues.map((issue) => issue.message).join('\n');

    const strong = document.createElement('strong');
    strong.textContent = suggestedDocumentLabel(item);
    button.appendChild(strong);
    button.addEventListener('click', () => selectDocument(item.id));
    list.appendChild(button);
  });

  if (!visibleDocuments.length) {
    $('emptyState').classList.remove('hidden');
    $('reviewPane').classList.add('hidden');
    const pdfFrame = $('pdfFrame');
    pdfFrame.dataset.documentId = '';
    pdfFrame.src = '/documents/missing/pdf';
  }
}

/**
 * Highlight exactly which controls need reviewer attention for the selected PDF.
 *
 * Metadata names used by Python (`tax_map`) are mapped to HTML element IDs (`taxMap`).
 * The surrounding label/card receives a CSS class and the control receives an
 * `aria-invalid` value, so the same problem is visible to both sighted users and
 * assistive technology.
 *
 * @param {Object} document Selected review record.
 */
function renderMissingMetadataHighlights(document) {
  const validation = getDocumentValidationState(document, buildValidationContext());
  const ids = {
    lot: 'lot', address: 'address', tax_map: 'taxMap', parcel: 'parcel',
    tax_id: 'taxId', project_code: 'editProjectCode', document_type: 'editDocumentType'
  };

  Object.entries(ids).forEach(([key, id]) => {
    const field = $(id);
    if (!field) return;
    const container = key === 'document_type'
      ? field.closest('.document-type-card')
      : field.closest('label');
    if (container) container.classList.toggle('metadata-missing', validation.missingFieldKeys.has(key));
    field.setAttribute('aria-invalid', validation.missingFieldKeys.has(key) ? 'true' : 'false');
  });
}

/**
 * Populate the PDF viewer and review form for one selected record.
 *
 * The viewer URL points to `document_pdf()` in app.py. Its `src` is changed only when
 * the document ID changes; otherwise Mass Scan polling would repeatedly reload the
 * PDF and reset the reviewer's page and zoom. Metadata fields, custom document-type
 * controls, missing-field highlights, and filing restrictions are then synchronized
 * with the selected record.
 *
 * @param {Object} document Record returned by `selectedDocument()`.
 */
function renderSelectedDocument(document) {
  $('emptyState').classList.add('hidden');
  $('reviewPane').classList.remove('hidden');
  const pdfFrame = $('pdfFrame');
  const pdfUrl = `/documents/${document.id}/pdf`;

  // Live mass-scan polling refreshes the application state several times per
  // second. Reassigning an iframe's src, even to the same URL, can make the
  // browser reload the PDF and reset the user's page/zoom position. Only load
  // the PDF when the selected document actually changes.
  if (pdfFrame.dataset.documentId !== String(document.id)) {
    pdfFrame.dataset.documentId = String(document.id);
    pdfFrame.src = pdfUrl;
  }

  $('documentTitle').textContent = document.source_name;
  $('documentStatus').textContent = statusLabel(document);

  fields.lot.value = document.metadata.lot || '';
  fields.address.value = document.metadata.address || '';
  fields.taxMap.value = document.metadata.tax_map || '';
  fields.parcel.value = document.metadata.parcel || '';
  fields.taxId.value = document.metadata.tax_id || '';
  fields.section.value = document.metadata.section || '';
  fields.editProjectCode.value = document.metadata.project_code || '';
  if (fields.editDocumentType) {
    const documentType = String(document.metadata.document_type || 'Field Notes').trim();
    const knownOption = Array.from(fields.editDocumentType.options)
      .some((option) => option.value === documentType);

    fields.editDocumentType.value = knownOption ? documentType : '__custom__';
    if (fields.customDocumentType) {
      fields.customDocumentType.value = knownOption ? '' : documentType;
    }
    updateCustomDocumentTypeInterface();
  }
  renderMissingMetadataHighlights(document);
  $('fileButton').disabled = document.status === 'filed' || document.is_lookup_document;
  $('fileButton').title = document.is_lookup_document
    ? 'Lookup-only documents are removed after the permanent batch is filed.'
    : '';
}

/**
 * Make one queue record active and redraw both master and detail views.
 *
 * Saving only the ID in state allows later API responses to replace the underlying
 * object. `renderList()` updates active-button styling, and
 * `renderSelectedDocument()` fills the form and viewer from the newest record.
 *
 * @param {string} id Stable document ID assigned by app.py.
 */
function selectDocument(id) {
  state.selectedId = id;
  const document = selectedDocument();
  renderList();
  if (document) renderSelectedDocument(document);
}

/**
 * Make a complete Flask state response the browser's new source of truth.
 *
 * Settings are copied into scan controls, documents replace the local queue, and an
 * invalid selection falls back to the first remaining record. Normal calls redraw the
 * selected PDF and form. During live Mass Scan polling, `preserveReview` redraws only
 * the queue so a background refresh cannot overwrite text the reviewer is currently
 * typing into the form.
 *
 * @param {Object} data State object returned by `/api/state`, `/api/scan`, or filing.
 * @param {Object} [options={}] Rendering options.
 * @param {boolean} [options.preserveReview=false] Do not overwrite the active form.
 */
function applyState(data, options = {}) {
  state.documents = data.documents || [];
  state.settings = data.settings || {};

  fields.inputFolder.value = state.settings.input_folder || '';
  fields.outputFolder.value = state.settings.output_folder || '';
  fields.configPath.value = state.settings.config_path || '';
  fields.projectCode.value = state.settings.project_code_override || state.settings.project_code || '';
  if (fields.county) fields.county.value = state.settings.county || 'Calvert';
  fields.dpi.value = state.settings.dpi || 300;
  if (fields.ocrDevice) fields.ocrDevice.value = state.settings.ocr_device || 'auto';
  if (fields.scanMode) fields.scanMode.value = state.settings.scan_mode || 'batch';
  if (fields.inPlace) fields.inPlace.checked = Boolean(state.settings.in_place);
  updateInPlaceInterface();

  if (!state.documents.some((document) => document.id === state.selectedId)) {
    state.selectedId = state.documents[0]?.id || null;
  }

  // During a live mass scan, state polling should refresh the queue and warning
  // summary without replacing values the reviewer is currently typing. The
  // review form is refreshed only when the selected document changes or when a
  // deliberate save/select action calls applyState without preserveReview.
  if (options.preserveReview && state.selectedId) {
    renderList();
    return;
  }

  if (state.selectedId)
    selectDocument(state.selectedId);
  else
    renderList();
}

// ============================================================================
// REQUEST PAYLOADS AND SERVER-SYNCHRONIZED WORKFLOWS
// ============================================================================

/**
 * Persist a changed output folder without resubmitting all scan settings.
 *
 * The dedicated PATCH route updates saved settings and any current batch records that
 * depend on the output root. The normalized path returned by Python is written back to
 * both local state and the form, which is important on Windows where path resolution
 * may change the typed representation.
 *
 * @returns {Promise<string>} Absolute output folder accepted by the server.
 */
async function saveOutputFolder() {
  const data = await requestJson('/api/settings/output-folder', {
    method: 'PATCH',
    body: JSON.stringify({ output_folder: fields.outputFolder.value }),
  });
  state.settings.output_folder = data.output_folder;
  fields.outputFolder.value = data.output_folder;
  return data.output_folder;
}

/**
 * Read the scan form and build the request shape expected by `scan_settings()` in app.py.
 *
 * DOM values are strings by default, so DPI is converted to a number and checkboxes to
 * booleans. Defaults protect older pages or missing optional controls. This function
 * gathers data only; validation and filesystem checks remain on the server.
 *
 * @returns {Object} JSON-serializable scan settings.
 */
function scanPayload() {
  return {
    input_folder: fields.inputFolder.value,
    output_folder: fields.outputFolder.value,
    config_path: fields.configPath.value,
    project_code: fields.projectCode.value,
    county: fields.county ? fields.county.value : 'Frederick',
    dpi: Number(fields.dpi.value),
    ocr_device: fields.ocrDevice ? fields.ocrDevice.value : 'auto',
    scan_mode: fields.scanMode ? fields.scanMode.value : 'batch',
    in_place: fields.inPlace ? fields.inPlace.checked : false,
  };
}

/**
 * Build the PATCH body for edits to the currently selected document.
 *
 * It gathers visible metadata, resolves the custom document-type option, preserves any
 * manually edited folder/file name, and tells Python whether those names should be
 * regenerated. `changedField` is especially important: app.py uses it to decide when a
 * Tax ID or address change should trigger a fresh SDAT lookup and Batch synchronization.
 *
 * @param {boolean} [autoFolder=false] Regenerate the suggested destination folder.
 * @param {boolean} [autoFileName=false] Regenerate the suggested PDF name.
 * @param {string} [changedField=''] Metadata key that initiated autosave.
 * @returns {Object} JSON-serializable document update.
 */
function updatePayload(autoFolder = false, autoFileName = false, changedField = '') {
  return {
    lot: fields.lot.value,
    address: fields.address.value,
    tax_map: fields.taxMap.value,
    parcel: fields.parcel.value,
    tax_id: fields.taxId.value,
    section: fields.section.value,
    project_code: fields.editProjectCode.value,
    document_type: fields.editDocumentType
      ? (fields.editDocumentType.value === '__custom__'
        ? String(fields.customDocumentType?.value || '').trim()
        : fields.editDocumentType.value)
      : '',
    folder_name: selectedDocument()?.folder_name || '',
    file_name: selectedDocument()?.file_name || '',
    auto_folder: autoFolder,
    auto_file_name: autoFileName,
    changed_field: changedField,
  };
}

/**
 * Restore saved settings and pending review records when the page first opens.
 *
 * `/api/state` reads `.review_state/documents.json` through state_store.py. Passing the
 * response to `applyState()` reconstructs the queue, scan controls, selected PDF, and
 * metadata form after a browser refresh or application restart.
 *
 * @returns {Promise<void>}
 */
async function loadState() {
  applyState(await requestJson('/api/state'));
}

/**
 * Publish completed Mass Scan documents to the queue while OCR continues.
 *
 * Mass mode appends each finished PDF to server state independently. This polling tick
 * reloads that state with `preserveReview: true`, allowing the reviewer to begin work
 * without waiting for the whole folder. A toast appears only when the queue grows.
 * Temporary read failures are ignored because the main scan request owns error reporting.
 *
 * @returns {Promise<void>}
 */
async function pollLiveScanState() {
  try {
    const data = await requestJson('/api/state');
    const previousCount = state.documents.length;
    applyState(data, { preserveReview: true });
    if (state.documents.length > previousCount) {
      showToast(`${state.documents.length} document${state.documents.length === 1 ? '' : 's'} ready for review.`);
    }
  } catch (_error) {
    // The main scan request reports actionable errors. A temporary state read
    // failure should not stop OCR or the progress timer.
  }
}

/**
 * Start the Mass Scan queue-refresh loop.
 *
 * An immediate poll avoids waiting for the first interval, then Flask state is checked
 * every 500 ms. Clearing an existing interval first guarantees only one loop is active.
 */
function startLiveStatePolling() {
  clearInterval(liveStateTimer);
  pollLiveScanState();
  liveStateTimer = setInterval(pollLiveScanState, 500);
}

/**
 * Stop publishing incremental Mass Scan state after the scan finishes or fails.
 *
 * Clearing the saved handle is important because an orphaned interval would keep
 * requesting `/api/state` and could unexpectedly redraw the queue during later work.
 */
function stopLiveStatePolling() {
  clearInterval(liveStateTimer);
  liveStateTimer = null;
}

/**
 * Run the complete browser side of the Scan PDFs button workflow.
 *
 * The function gathers settings, locks the button, starts progress polling, and starts
 * live queue polling only for Mass mode. It then POSTs to `/api/scan`; the returned state
 * becomes the new review queue. `finally` always stops timers and restores the button,
 * even when OCR or validation fails, so the interface cannot remain permanently busy.
 *
 * @returns {Promise<void>}
 */
async function scan() {
  const button = $('scanButton');
  const payload = scanPayload();
  setButtonLoading(button, true, 'Scanning...', 'Scan PDFs');
  startScanProgressPolling();
  if (payload.scan_mode === 'mass') startLiveStatePolling();
  let scanFailed = false;

  try {
    const data = await requestJson('/api/scan', {
      method: 'POST',
      body: JSON.stringify(payload),
    });
    state.selectedId = state.selectedId || data.documents?.[0]?.id || null;
    applyState(data);
    showToast(`Scanned ${state.documents.length} PDF${state.documents.length === 1 ? '' : 's'}.`);
  } catch (error) {
    scanFailed = true;
    showToast(error.message, true);
  } finally {
    stopLiveStatePolling();
    await stopScanProgressPolling({ failed: scanFailed });
    setButtonLoading(button, false, 'Scanning...', 'Scan PDFs');
  }
}

/**
 * Save the review form for the selected document and apply the authoritative response.
 *
 * The PATCH request is handled by `api_update_document()` in app.py, where metadata may
 * be revalidated, SDAT-refreshed, and synchronized across a Batch. A Batch-wide edit can
 * return a complete state object; an isolated edit may return one document. This function
 * supports both response shapes and returns the newest selected record to its caller.
 *
 * @param {boolean} [autoFolder=false] Ask Python to rebuild the destination folder.
 * @param {boolean} [autoFileName=false] Ask Python to rebuild the PDF name.
 * @param {string} [changedField=''] Field responsible for the save.
 * @returns {Promise<Object|null>} Updated selected record, or null with no selection.
 */
async function saveCurrent(autoFolder = false, autoFileName = false, changedField = '') {
  const document = selectedDocument();
  if (!document) return null;

  const updated = await requestJson(`/api/documents/${document.id}`, {
    method: 'PATCH',
    body: JSON.stringify(updatePayload(autoFolder, autoFileName, changedField)),
  });

  if (updated.documents) {
    applyState(updated);
    return selectedDocument();
  }

  replaceDocument(updated);
  selectDocument(updated.id);
  return updated;
}

/**
 * Save and file the currently selected permanent PDF.
 *
 * Saving first ensures the server receives the latest form values. The filing request
 * then supplies output/copy/text/in-place options to `file_document_to_output()`. Flask
 * returns a queue with the completed record removed; `applyState()` selects the next
 * available PDF. Lookup-only helper documents are blocked earlier by the disabled button.
 *
 * @returns {Promise<void>}
 */
async function fileCurrent() {
  const document = await saveCurrent(false, false);
  if (!document) return;

  try {
    const filed = await requestJson(`/api/documents/${document.id}/file`, {
      method: 'POST',
      body: JSON.stringify({
        folder_name: document.folder_name || '',
        file_name: document.file_name || '',
        output_folder: fields.outputFolder.value,
        copy: fields.copyFile.checked,
        save_text: fields.saveText.checked,
        in_place: fields.inPlace ? fields.inPlace.checked : false,
      }),
    });

    // The server returns the active review queue with the filed document
    // removed. applyState keeps the current selection when possible and selects
    // the next available document when the filed document was active.
    applyState(filed);
    showToast(fields.inPlace?.checked ? 'Metadata saved to the original PDF.' : `Filed to ${filed.filed?.filed_path || 'the output folder'}`);
  } catch (error) {
    showToast(error.message, true);
  }
}

/**
 * Ask Flask to file every eligible record in the active review batch.
 *
 * The button remains disabled while app.py validates the packet, writes PDF metadata,
 * moves or copies files, records tracker information, and removes completed state. The
 * interface is updated only from the server response, and `finally` restores the control
 * even if one document causes the batch operation to fail.
 *
 * @returns {Promise<void>}
 */
async function fileAll() {
  const button = $('fileAllButton');
  setButtonLoading(button, true, 'Filing...', 'File Batch');

  try {
    const data = await requestJson('/api/file-all', {
      method: 'POST',
      body: JSON.stringify({
        output_folder: fields.outputFolder.value,
        copy: fields.copyFile.checked,
        save_text: fields.saveText.checked,
        in_place: fields.inPlace ? fields.inPlace.checked : false,
      }),
    });

    applyState(data);
    showToast(fields.inPlace?.checked ? 'Metadata saved to the original PDFs.' : `Filed ${state.documents.length} PDF${state.documents.length === 1 ? '' : 's'} as one batch.`);
    state.documents = []
    renderList()
  } catch (error) {
    showToast(error.message, true);
  } finally {
    setButtonLoading(button, false, 'Filing...', 'File Batch');
  }
}

// ============================================================================
// FORM AUTOSAVE AND MODE-SPECIFIC CONTROLS
// ============================================================================

/**
 * Translate HTML control IDs into the snake_case metadata keys used by Python.
 *
 * Most IDs already match their API keys, but names such as `taxMap` and
 * `editDocumentType` do not. Keeping this mapping in one place lets generic autosave
 * code tell app.py exactly which property changed.
 *
 * @param {string} id HTML element ID.
 * @returns {string} Metadata key expected in the PATCH payload.
 */
function metadataFieldName(id) {
  const names = { taxMap: 'tax_map', taxId: 'tax_id', editProjectCode: 'project_code', editDocumentType: 'document_type' };
  return names[id] || id;
}

/**
 * Attach the same metadata-save behavior to a group of form controls.
 *
 * On each `change` event, the listener converts the control ID to an API field name and
 * calls `saveCurrent()` with the requested name-regeneration rules. Errors are routed to
 * the shared toast. Registration removes repetitive event-handler code and keeps all
 * property fields consistent.
 *
 * @param {string[]} ids Form control IDs to watch.
 * @param {boolean} autoFolder Rebuild folder suggestions after these fields change.
 * @param {boolean} autoFileName Rebuild filename suggestions after these fields change.
 */
function registerAutoSave(ids, autoFolder, autoFileName) {
  ids.forEach((id) => {
    const element = $(id);
    if (!element) return;
    element.addEventListener('change', () => {
      saveCurrent(autoFolder, autoFileName, metadataFieldName(id))
        .catch((error) => showToast(error.message, true));
    });
  });
}

registerAutoSave(['lot', 'address', 'taxMap', 'parcel', 'taxId', 'section', 'editProjectCode'], true, true);

/**
 * Switch the document-type editor between a configured type and free text.
 *
 * Selecting the special `__custom__` option reveals, enables, and requires the custom
 * text input. Choosing a configured type hides and disables that input so stale custom
 * text cannot accidentally be submitted. Focus is deferred to the next animation frame
 * so it occurs after the control becomes visible.
 */
function updateCustomDocumentTypeInterface() {
  if (!fields.editDocumentType || !fields.customDocumentType) return;

  const isCustom = fields.editDocumentType.value === '__custom__';
  const control = $('customDocumentTypeControl');
  if (control) control.classList.toggle('hidden', !isCustom);
  fields.customDocumentType.disabled = !isCustom;
  fields.customDocumentType.required = isCustom;

  if (isCustom) {
    window.requestAnimationFrame(() => fields.customDocumentType.focus());
  }
}

if (fields.editDocumentType) {
  fields.editDocumentType.addEventListener('change', () => {
    updateCustomDocumentTypeInterface();
    if (fields.editDocumentType.value !== '__custom__') {
      saveCurrent(true, true, 'document_type')
        .catch((error) => showToast(error.message, true));
    }
  });
}

if (fields.customDocumentType) {
  fields.customDocumentType.addEventListener('change', () => {
    const customType = fields.customDocumentType.value.trim();
    if (!customType) {
      showToast('Enter a custom document type before saving.', true);
      fields.customDocumentType.focus();
      return;
    }
    saveCurrent(true, true, 'document_type')
      .catch((error) => showToast(error.message, true));
  });
}


/**
 * Reconfigure filing controls when In-Place mode is selected.
 *
 * In-Place mode writes metadata back to each original PDF, so an output folder and copy
 * option no longer apply. Those controls are disabled and any open output-folder picker
 * is closed. Turning the mode off restores normal destination-tree filing controls.
 */
function updateInPlaceInterface() {
  const enabled = Boolean(fields.inPlace?.checked);
  fields.outputFolder.disabled = enabled;
  const browseButton = document.querySelector(
    'button[onclick="openFolderBrowser(\'outputFolder\')"]'
  );
  if (browseButton) browseButton.disabled = enabled;
  fields.copyFile.disabled = enabled;
  if (enabled) closeFolderBrowser('outputFolder');
}

fields.outputFolder.addEventListener('change', () => {
  if (!state.documents.length) return;
  saveOutputFolder()
    .then(() => showToast('Output folder updated for the current batch.'))
    .catch((error) => showToast(error.message, true));
});

if (fields.inPlace) fields.inPlace.addEventListener('change', updateInPlaceInterface);

// These listeners are the three main entry points initiated by button clicks.
// Each named function owns its complete async workflow and error handling.
$('scanButton').addEventListener('click', scan);
$('fileButton').addEventListener('click', fileCurrent);
$('fileAllButton').addEventListener('click', fileAll);


// Initial page startup: reconstruct the interface from the server's saved state.
// A startup failure uses the same visible error channel as later API actions.
loadState().catch((error) => showToast(error.message, true));



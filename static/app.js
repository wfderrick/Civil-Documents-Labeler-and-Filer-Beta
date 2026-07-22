/**
 * Browser-side controller for the COABarrett document review application.
 *
 * Responsibilities include loading server state, rendering the queue and review
 * form, managing the PDF viewer, validating metadata, polling scan progress, and
 * sending explicit user edits or filing commands back to Flask. Keep DOM-only
 * concerns here; extraction and file-system behavior belong in Python services.
 */

const state = { documents: [], settings: {}, selectedId: null };

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
  lot: $('lot'),
  address: $('address'),
  taxMap: $('taxMap'),
  parcel: $('parcel'),
  taxId: $('taxId'),
  section: $('section'),
  editProjectCode: $('editProjectCode'),
  editDocumentType: $('editDocumentType'),
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

/**
 * resetScanProgressPanel. This helper is kept small so UI state changes remain traceable.
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
 * renderScanProgress. This helper is kept small so UI state changes remain traceable.
 * @param {*} data Value supplied by the caller.
 * @returns {*} Computed value or asynchronous result used by the interface.
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
 * pollScanProgress. This helper is kept small so UI state changes remain traceable.
 */
async function pollScanProgress() {
  try {
    renderScanProgress(await requestJson('/api/scan-progress'));
  } catch (_error) {
    // The main scan request reports actionable errors. Keep polling quietly.
  }
}

/**
 * updateLocalScanElapsed. This helper is kept small so UI state changes remain traceable.
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
 * startScanProgressPolling. This helper is kept small so UI state changes remain traceable.
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
 * stopScanProgressPolling. This helper is kept small so UI state changes remain traceable.
 * @param {*} { failed Value supplied by the caller.
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

/**
 * openFolderBrowser. This helper is kept small so UI state changes remain traceable.
 * @param {*} targetFieldId Value supplied by the caller.
 * @param {*} startPath Value supplied by the caller.
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
 * closeFolderBrowser. This helper is kept small so UI state changes remain traceable.
 * @param {*} targetFieldId Value supplied by the caller.
 */
function closeFolderBrowser(targetFieldId) {
  const browser = folderBrowserElements[targetFieldId];
  if (browser) browser.panel.classList.add('hidden');
  if (browseTargetField === targetFieldId) browseTargetField = null;
}

/**
 * loadBrowseFolder. This helper is kept small so UI state changes remain traceable.
 * @param {*} path Value supplied by the caller.
 * @param {*} targetFieldId Value supplied by the caller.
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

/**
 * showToast. This helper is kept small so UI state changes remain traceable.
 * @param {*} message Value supplied by the caller.
 * @param {*} isError Value supplied by the caller.
 */
function showToast(message, isError = false) {
  const toast = $('toast');
  toast.textContent = message;
  toast.className = `toast ${isError ? 'error' : ''}`;
  setTimeout(() => toast.classList.add('hidden'), 3800);
}


/**The requestJson() function returns the current state data takes in a url and 
using the fetch() function processes the returned data from the given url. This 
function makes a GET /api/state request to the Flask object which then calls the 
api_state() function in app.py to return a Response object as a json file 
holding the current settings and document metadata. */
/**
 * requestJson. This helper is kept small so UI state changes remain traceable.
 * @param {*} url Value supplied by the caller.
 * @param {*} options Value supplied by the caller.
 * @returns {*} Computed value or asynchronous result used by the interface.
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

/**The parseJSONResponse() function returns an object holding the information from
the bodytext parameter. The function first checks that bodytext is valid and 
then trys to parse it as a JSON using the JSON.parse() function. If successful 
the object returned by the parse() function is returned otherwise an error is 
thrown.*/
/**
 * parseJsonResponse. This helper is kept small so UI state changes remain traceable.
 * @param {*} bodyText Value supplied by the caller.
 * @param {*} response Value supplied by the caller.
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

/**The statusLabel() function returns a beautified version of the current status
label stored in the status field in the document parameter.*/
function statusLabel(document) {
  const labels = { filed: 'Filed', ready: 'Ready', needs_review: 'Needs review', lookup_only: 'Lookup only' };
  return labels[document.status] || 'Needs review';
}

/**The selectedDocument() function returns the document in the documents property
in the state object that has the same id as the id stored in the selectedId 
property in the state object. It uses the find() function which returns the 
first value in an array for which the predicate is true.*/
/**
 * selectedDocument. This helper is kept small so UI state changes remain traceable.
 */
function selectedDocument() {
  return state.documents.find((document) => document.id === state.selectedId);
}

/**
 * replaceDocument. This helper is kept small so UI state changes remain traceable.
 * @param {*} updatedDocument Value supplied by the caller.
 */
function replaceDocument(updatedDocument) {
  const index = state.documents.findIndex((document) => document.id === updatedDocument.id);
  if (index >= 0) state.documents[index] = updatedDocument;
}

/**The setButtonLoading() function controls the appearance and function of the
 * Scan PDF's and File Batch buttons. When the second parameter is set to True 
 * it prevents the user from clicking the button. Also based on the second 
 * parameter the button either displays the loading text meaning the process 
 * initiated by the button is currently happening or the ready text meaning the 
 * process has either not been started or it's finished. 
*/
/**
 * setButtonLoading. This helper is kept small so UI state changes remain traceable.
 * @param {*} button Value supplied by the caller.
 * @param {*} isLoading Value supplied by the caller.
 * @param {*} loadingText Value supplied by the caller.
 * @param {*} readyText Value supplied by the caller.
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

/**
 * isMissingMetadataValue. This helper is kept small so UI state changes remain traceable.
 * @param {*} key Value supplied by the caller.
 * @param {*} value Value supplied by the caller.
 * @returns {*} Computed value or asynchronous result used by the interface.
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
 * normalizeIssueMessages. This helper is kept small so UI state changes remain traceable.
 * @param {*} value Value supplied by the caller.
 * @returns {*} Computed value or asynchronous result used by the interface.
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
 * documentTypeValue. This helper is kept small so UI state changes remain traceable.
 * @param {*} document Value supplied by the caller.
 */
function documentTypeValue(document) {
  return String(document?.metadata?.document_type || '').trim();
}

/**
 * buildValidationContext. This helper is kept small so UI state changes remain traceable.
 * @param {*} documents Value supplied by the caller.
 * @returns {*} Computed value or asynchronous result used by the interface.
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
 * suggestedDocumentLabel. This helper is kept small so UI state changes remain traceable.
 * @param {*} item Value supplied by the caller.
 */
function suggestedDocumentLabel(item) {
  const rawType = String(item.metadata?.document_type || '').trim();
  const rawLot = String(item.metadata?.lot || '').trim();
  const type = isMissingMetadataValue('document_type', rawType) ? 'Unknown Type' : rawType;
  const lot = isMissingMetadataValue('lot', rawLot) ? 'Unknown Lot' : `Lot ${rawLot}`;
  return `${type} - ${lot}`;
}

/**
 * renderList. This helper is kept small so UI state changes remain traceable.
 * @returns {*} Computed value or asynchronous result used by the interface.
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

/**The renderSelectedDocument() function makes the document and review pane for 
that document given by the document parameter, visible to the user. It begins by
adding the hidden class to the emptyState <div> to hide it and doing the 
opposite to the reviewPane <div> to reveal it. The <iframe> element with 
id=pdfFrame is then changed which tells the browser to make a GET request to the
given url. This url corresponds to a decorator in app.py for the document_pdf()
fucntion which returns a response object with the requested document in pdf 
form. The <iframe> element displays that given pdf. The documentTitle <div> is
updated with the source_name field in the document parameter. The documentStatus
<div> is updated with the statusLabel() function with a nice version of the 
documents current status. Each field is updated with the document parameter's
metadata.*/
/**
 * renderMissingMetadataHighlights. This helper is kept small so UI state changes remain traceable.
 * @param {*} document Value supplied by the caller.
 * @returns {*} Computed value or asynchronous result used by the interface.
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
 * renderSelectedDocument. This helper is kept small so UI state changes remain traceable.
 * @param {*} document Value supplied by the caller.
 * @returns {*} Computed value or asynchronous result used by the interface.
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
  if (fields.editDocumentType) fields.editDocumentType.value = document.metadata.document_type || 'Field Notes';
  renderMissingMetadataHighlights(document);
  $('fileButton').disabled = document.status === 'filed' || document.is_lookup_document;
  $('fileButton').title = document.is_lookup_document
    ? 'Lookup-only documents are removed after the permanent batch is filed.'
    : '';
}

/**The selectDocument() function updates the current selected document, displays 
the document, and displays the review panel for the document. First the current
state has its selected id updated to the id parameter. Then the document 
variable is set to the return of selectedDocument() which returns the 
information for the document in the state object with the same id as the 
selectedId. renderList() updates the button for the selected document to have 
the active class which changes its appearance so you can tell which document is
selected. renderSelectedDocument() displays the document as a pdf and the review 
panel for editing the document information.*/
/**
 * selectDocument. This helper is kept small so UI state changes remain traceable.
 * @param {*} id Value supplied by the caller.
 */
function selectDocument(id) {
  state.selectedId = id;
  const document = selectedDocument();
  renderList();
  if (document) renderSelectedDocument(document);
}

/**The applyState() function updates the state and HTML fields from the data 
 * parameter, then adds button to access the current documents and show the 
 * current selected document, adds a red banner at the top of the screen to 
 * display any problems within the batch, and display the selected document in
 * pdf form and the review panel to edit document metadata. It does this by 
 * taking in the data parameter and updating the state.document and 
 * state.settings properties to be the same as data.documents and data.settings
 * respectively. Then each field is updated from the state properties these 
 * include input_folder and dpi. Once all of those are updated the warnings are
 * shown by renderBatchWarnings(). If state.selectedId isn't null 
 * selectDocument() adds button to access all of the documents in the document 
 * property, displays the document as a pdf and shows the review panel. 
 * Otherwise page is rendered with no documents.
*/
/**
 * applyState. This helper is kept small so UI state changes remain traceable.
 * @param {*} data Value supplied by the caller.
 * @param {*} options Value supplied by the caller.
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

/**
 * saveOutputFolder. This helper is kept small so UI state changes remain traceable.
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

/**The scanPayload() function returns a javascript object containing the 
 * settings relevant to the scanning process which are input folder, 
 * ouput folder, config path, project code, dpi, and ocr device. These are taken
 * from the current html fields for each.
*/
/**
 * scanPayload. This helper is kept small so UI state changes remain traceable.
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
  };
}

/**
 * updatePayload. This helper is kept small so UI state changes remain traceable.
 * @param {*} autoFolder Value supplied by the caller.
 * @param {*} autoFileName Value supplied by the caller.
 * @param {*} changedField Value supplied by the caller.
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
    document_type: fields.editDocumentType ? fields.editDocumentType.value : '',
    folder_name: selectedDocument()?.folder_name || '',
    file_name: selectedDocument()?.file_name || '',
    auto_folder: autoFolder,
    auto_file_name: autoFileName,
    changed_field: changedField,
  };
}

/**The loadState() function updates the current appearance and state of the app.
 * First it calls requestJson() with the /api/state url which tells the browser 
 * to make a GET request to Flask. The matching decorator in app.py calls the 
 * api_state() function to return a Response object containing json with the 
 * information about the current state from the documents.json file. Once 
 * requestJson gives its output it is passed into applyState() which takes the 
 * json as its data parameter and updates all of the current html fields and the
 * state in app.js along with adding the visual changes of the document buttons,
 * warning banner, document pdf, and review panel.
*/
/**
 * loadState. This helper is kept small so UI state changes remain traceable.
 */
async function loadState() {
  applyState(await requestJson('/api/state'));
}

/**The scan() function */
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
 * startLiveStatePolling. This helper is kept small so UI state changes remain traceable.
 */
function startLiveStatePolling() {
  clearInterval(liveStateTimer);
  pollLiveScanState();
  liveStateTimer = setInterval(pollLiveScanState, 500);
}

/**
 * stopLiveStatePolling. This helper is kept small so UI state changes remain traceable.
 */
function stopLiveStatePolling() {
  clearInterval(liveStateTimer);
  liveStateTimer = null;
}

/**
 * scan. This helper is kept small so UI state changes remain traceable.
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
 * saveCurrent. This helper is kept small so UI state changes remain traceable.
 * @param {*} autoFolder Value supplied by the caller.
 * @param {*} autoFileName Value supplied by the caller.
 * @param {*} changedField Value supplied by the caller.
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
 * fileCurrent. This helper is kept small so UI state changes remain traceable.
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
      }),
    });

    // The server returns the active review queue with the filed document
    // removed. applyState keeps the current selection when possible and selects
    // the next available document when the filed document was active.
    applyState(filed);
    showToast(`Filed to ${filed.filed?.filed_path || 'the output folder'}`);
  } catch (error) {
    showToast(error.message, true);
  }
}

/**
 * fileAll. This helper is kept small so UI state changes remain traceable.
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
      }),
    });

    applyState(data);
    showToast(`Filed ${state.documents.length} PDF${state.documents.length === 1 ? '' : 's'} as one batch.`);
    state.documents = []
    renderList()
  } catch (error) {
    showToast(error.message, true);
  } finally {
    setButtonLoading(button, false, 'Filing...', 'File Batch');
  }
}

/**
 * metadataFieldName. This helper is kept small so UI state changes remain traceable.
 * @param {*} id Value supplied by the caller.
 */
function metadataFieldName(id) {
  const names = { taxMap: 'tax_map', taxId: 'tax_id', editProjectCode: 'project_code', editDocumentType: 'document_type' };
  return names[id] || id;
}

/**
 * registerAutoSave. This helper is kept small so UI state changes remain traceable.
 * @param {*} ids Value supplied by the caller.
 * @param {*} autoFolder Value supplied by the caller.
 * @param {*} autoFileName Value supplied by the caller.
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

registerAutoSave(['lot', 'address', 'taxMap', 'parcel', 'taxId', 'section', 'editProjectCode', 'editDocumentType'], true, true);

fields.outputFolder.addEventListener('change', () => {
  if (!state.documents.length) return;
  saveOutputFolder()
    .then(() => showToast('Output folder updated for the current batch.'))
    .catch((error) => showToast(error.message, true));
});

/*Calls the scan() function whenever the Scan PDF's button is clicked.*/
$('scanButton').addEventListener('click', scan);

/*Calls the fileCurrent() function whenever the File PDF button is clicked.*/
$('fileButton').addEventListener('click', fileCurrent);

/*Calls the fileAll() function whenever the File Batch button is clicked.*/
$('fileAllButton').addEventListener('click', fileAll);


/*This is called the first time the app is opened by the user to update the 
visuals and settings. 
*/
loadState().catch((error) => showToast(error.message, true));



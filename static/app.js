const state = { documents: [], settings: {}, selectedId: null };

const $ = (id) => document.getElementById(id);

const fields = {
  inputFolder: $('inputFolder'),
  outputFolder: $('outputFolder'),
  configPath: $('configPath'),
  projectCode: $('projectCode'),
  dpi: $('dpi'),
  ocrDevice: $('ocrDevice'),
  lot: $('lot'),
  address: $('address'),
  taxMap: $('taxMap'),
  parcel: $('parcel'),
  taxId: $('taxId'),
  section: $('section'),
  editProjectCode: $('editProjectCode'),
  editDocumentType: $('editDocumentType'),
  folderName: $('folderName'),
  fileName: $('fileName'),
  copyFile: $('copyFile'),
  saveText: $('saveText'),
};

let browseTargetField = null;

let scanProgressTimer = null;
let scanElapsedTimer = null;
let scanStartedAt = null;
let renderedProgressCount = 0;

function resetScanProgressPanel() {
  const panel = $('scanProgressPanel');
  panel.classList.remove('hidden', 'failed');
  $('scanProgressStatus').textContent = 'Starting scan...';
  $('scanElapsed').textContent = '0.0 s';
  $('scanProgressMessages').innerHTML = '';
  renderedProgressCount = 0;
}

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

async function pollScanProgress() {
  try {
    renderScanProgress(await requestJson('/api/scan-progress'));
  } catch (_error) {
    // The main scan request reports actionable errors. Keep polling quietly.
  }
}

function updateLocalScanElapsed() {
  if (scanStartedAt === null) return;

  // Date.now() measures from the original start time rather than incrementing a
  // counter. This prevents timer drift when the browser or OCR work delays an
  // individual interval callback.
  const elapsedSeconds = Math.max(0, (Date.now() - scanStartedAt) / 1000);
  $('scanElapsed').textContent = `${elapsedSeconds.toFixed(1)} s`;
}

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

async function openFolderBrowser(targetFieldId, startPath = '') {
  browseTargetField = targetFieldId;
  await loadBrowseFolder(startPath || fields[targetFieldId].value);
  $('folderBrowserModal').classList.remove('hidden');
}

async function loadBrowseFolder(path) {
  const data = await requestJson(`/api/browse-folders?path=${encodeURIComponent(path || '')}`);

  $('browseCurrentPath').textContent = data.current;
  $('browseFolderList').innerHTML = '';

  if (data.parent) {
    const up = document.createElement('button');
    up.textContent = '..';
    up.onclick = () => loadBrowseFolder(data.parent);
    $('browseFolderList').appendChild(up);
  }

  data.folders.forEach((folder) => {
    const button = document.createElement('button');
    button.textContent = folder.name;
    button.onclick = () => loadBrowseFolder(folder.path);
    $('browseFolderList').appendChild(button);
  });

  $('useFolderButton').onclick = async () => {
    fields[browseTargetField].value = data.current;
    $('folderBrowserModal').classList.add('hidden');
    if (browseTargetField === 'outputFolder' && state.documents.length) {
      try {
        await saveOutputFolder();
      } catch (error) {
        showToast(error.message, true);
      }
    }
  };
}

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
function selectedDocument() {
  return state.documents.find((document) => document.id === state.selectedId);
}

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
function setButtonLoading(button, isLoading, loadingText, readyText) {
  button.disabled = isLoading;
  button.textContent = isLoading ? loadingText : readyText;
}

/**The batchwarnings() function returns the possible problems for the document
 * batch in the state object. First it goes through each document that is not
 * a Lookup Only document. For every document it checks that the document has an 
 * acceptable type. If it does the function checks if the typeGroups Map 
 * variable has an element with a key matching the type, if it doesn't then an
 * element is added with that type as the key. Then the document name is added 
 * to the element with the corresponding type key in typegroups. For each 
 * document the required fields are looped through, and the metadata is checked 
 * to see if that document has the required fields. If they don't filter keeps
 * the required fields and returns them. map() takes the output of filter() and 
 * returns just the labels. This return is what missing is set to. If there is 
 * anything in missing it is added to the warnings variable along with the 
 * document the missing information correspsonds to. Finally, each element in 
 * typeGroups is reviewed. If any elements have more then one value in their 
 * list a warning is added to the beginning of warnings with the duplicate
 * document type and the file names. The warning list is returned.
*/
function batchWarnings() {
  const warnings = [];
  const typeGroups = new Map();
  const required = [
    ['lot', 'Lot'], ['address', 'Address'], ['project_code', 'Project code'],
    ['document_type', 'Document type'], ['tax_map', 'Tax map'],
    ['parcel', 'Parcel'], ['tax_id', 'Tax ID']
  ];

  state.documents.filter((document) => !document.is_lookup_document).forEach((document) => {
    const type = (document.metadata?.document_type || '').trim();
    if (type && type !== 'Document' && !type.toLowerCase().startsWith('unknown')) {
      if (!typeGroups.has(type)) typeGroups.set(type, []);
      typeGroups.get(type).push(document.source_name);
    }
    const missing = required
      .filter(([key]) => {
        const value = String(document.metadata?.[key] || '').trim();
        return !value || value.toLowerCase().startsWith('unknown') || value === 'Project' || value === 'Document';
      })
      .map(([, label]) => label);
    if (missing.length) warnings.push(`${document.source_name}: missing ${missing.join(', ')}`);
  });

  typeGroups.forEach((files, type) => {
    if (files.length > 1) warnings.unshift(`Duplicate document type “${type}”: ${files.join(', ')}`);
  });
  return warnings;
}

/**The renderBatchWarnings() function displays any errors that are occuring in 
 * the current batch of documents held in state. It begins by pointing banner to 
 * the batchWarning <div>. Then it checks that it exists if it doesn't the 
 * function returns nothing. Next it sets warnings to the return of 
 * batchWarnings() which returns a list of strings with unknown document type, 
 * multiple documents with the same type, and/or documents with missing 
 * information. If the length of warnings is 0 the batchWarning <div> has hidden
 * added to its class list so its not visible to the user anymore. The <div> 
 * also has the HTML within it cleared so there are no leftover warning messages 
 * which aren't supposed to be there. If the <div> exists and there are warnings
 * then all of the warning are added to an unordered list within the 
 * batchWarning <div>. Finally the hidden class is removed from the <div> to 
 * make it visible to the user.
*/
function renderBatchWarnings() {
  const banner = $('batchWarning');
  if (!banner) return;
  const warnings = batchWarnings();
  if (!warnings.length) {
    banner.classList.add('hidden');
    banner.innerHTML = '';
    return;
  }
  banner.innerHTML = `<strong>Batch needs attention</strong><ul>${warnings.map((item) => `<li>${item}</li>`).join('')}</ul>`;
  banner.classList.remove('hidden');
}

/**The renderlist() function adds buttons to access and edit documents with 
information is stored in documents.json. The list variable stores a pointer to 
the <div> element with the documentList id in index.html. .innerhtml is then 
called to remove any leftover items inside the <div> with id=documentList. The 
visibleDocuments variable is set to a pointer to the documents property of the 
state object. The textContent() function is called on the <span> with 
id=queuecount in order to set the text within it to the number of documents in 
the state object via visibleDocuments.length. For every document a button is 
created by the createElement() function which generates different HTML elements 
based on the parameter passed in. That button has its class set to 
doc-row.active which makes the CSS update the color of the button to show that 
this specific document is the current active document. A span is then added to
the internals of the button with its name and status with the innerHTML 
function. Next the button is set so that whenever it is clicked the 
selectDocument() function is called on that specific items id using the 
addEventListener() function. Finally the buttons created in the forEach loop are 
added to the list of children which belong to the <div> with id=documentList 
with the .appendChild() function. */
function renderList() {
  if (state.selectedId){
    const list = $('documentList');
  list.innerHTML = '';
  const visibleDocuments = state.documents;
  $('queueCount').textContent = String(visibleDocuments.length);

  visibleDocuments.forEach((item) => {
    const button = document.createElement('button');
    button.className = `doc-row ${item.id === state.selectedId ? 'active' : ''}`;
    button.innerHTML = `<strong>${item.source_name}</strong><span>${statusLabel(item)}</span>`;
    button.addEventListener('click', () => selectDocument(item.id));
    list.appendChild(button);
  });
  }else{
    $('emptyState').classList.remove('hidden');
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
function renderSelectedDocument(document) {
  $('emptyState').classList.add('hidden');
  $('reviewPane').classList.remove('hidden');
  $('pdfFrame').src = `/documents/${document.id}/pdf`;
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
  fields.folderName.value = document.folder_name || '';
  fields.fileName.value = document.file_name || '';
  $('ocrText').textContent = document.ocr_text || '';
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
function applyState(data) {
  state.documents = data.documents || [];
  state.settings = data.settings || {};

  fields.inputFolder.value = state.settings.input_folder || '';
  fields.outputFolder.value = state.settings.output_folder || '';
  fields.configPath.value = state.settings.config_path || '';
  fields.projectCode.value = state.settings.project_code_override || state.settings.project_code || '';
  fields.dpi.value = state.settings.dpi || 300;
  if (fields.ocrDevice) fields.ocrDevice.value = state.settings.ocr_device || 'auto';

  if (!state.documents.some((document) => document.id === state.selectedId)) {
    state.selectedId = state.documents[0]?.id || null;
  }

  renderBatchWarnings();
  if (state.selectedId) 
    selectDocument(state.selectedId);
  else
    renderList();
}

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
function scanPayload() {
  return {
    input_folder: fields.inputFolder.value,
    output_folder: fields.outputFolder.value,
    config_path: fields.configPath.value,
    project_code: fields.projectCode.value,
    dpi: Number(fields.dpi.value),
    ocr_device: fields.ocrDevice ? fields.ocrDevice.value : 'auto',
  };
}

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
    folder_name: fields.folderName.value,
    file_name: fields.fileName.value,
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
async function loadState() {
  applyState(await requestJson('/api/state'));
}

/**The scan() function */
async function scan() {
  const button = $('scanButton');
  setButtonLoading(button, true, 'Scanning...', 'Scan PDFs');
  startScanProgressPolling();
  let scanFailed = false;

  try {
    const data = await requestJson('/api/scan', {
      method: 'POST',
      body: JSON.stringify(scanPayload()),
    });
    state.selectedId = data.documents?.[0]?.id || null;
    applyState(data);
    showToast(`Scanned ${state.documents.length} PDF${state.documents.length === 1 ? '' : 's'}.`);
  } catch (error) {
    scanFailed = true;
    showToast(error.message, true);
  } finally {
    await stopScanProgressPolling({ failed: scanFailed });
    setButtonLoading(button, false, 'Scanning...', 'Scan PDFs');
  }
}

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

async function fileCurrent() {
  const document = await saveCurrent(false, false);
  if (!document) return;

  try {
    const filed = await requestJson(`/api/documents/${document.id}/file`, {
      method: 'POST',
      body: JSON.stringify({
        folder_name: fields.folderName.value,
        file_name: fields.fileName.value,
        output_folder: fields.outputFolder.value,
        copy: fields.copyFile.checked,
        save_text: fields.saveText.checked,
      }),
    });

    replaceDocument(filed);
    selectDocument(filed.id);
    showToast(`Filed to ${filed.filed_path}`);
  } catch (error) {
    showToast(error.message, true);
  }
}

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

function metadataFieldName(id) {
  const names = { taxMap: 'tax_map', taxId: 'tax_id', editProjectCode: 'project_code', editDocumentType: 'document_type' };
  return names[id] || id;
}

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
registerAutoSave(['folderName', 'fileName'], false, false);

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



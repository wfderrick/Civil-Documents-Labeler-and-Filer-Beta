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

  $('useFolderButton').onclick = () => {
    fields[browseTargetField].value = data.current;
    $('folderBrowserModal').classList.add('hidden');
  };
}

function showToast(message, isError = false) {
  const toast = $('toast');
  toast.textContent = message;
  toast.className = `toast ${isError ? 'error' : ''}`;
  setTimeout(() => toast.classList.add('hidden'), 3800);
}


/*The requestJson() function returns the current state data takes in a url and 
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

/*The parseJSONResponse() function returns an object holding the information from
the bodytext parameter. The function first checks that bodytext is valid and 
then trys to parse it as a JSON using the JSON.parse() function. If successful 
the object returned by the parse() function is returned otehrwise an error is 
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

function statusLabel(document) {
  const labels = { filed: 'Filed', ready: 'Ready', needs_review: 'Needs review', lookup_only: 'Lookup only' };
  return labels[document.status] || 'Needs review';
}

/*The selectedDocument() function returns the document in the documents property
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

function setButtonLoading(button, isLoading, loadingText, readyText) {
  button.disabled = isLoading;
  button.textContent = isLoading ? loadingText : readyText;
}

function batchWarnings() {
  const warnings = [];
  const typeGroups = new Map();
  const required = [
    ['lot', 'Lot'], ['address', 'Address'], ['project_code', 'Project code'],
    ['document_type', 'Document type'], ['tax_map', 'Tax map'],
    ['parcel', 'Parcel'], ['tax_id', 'Tax ID'], ['section', 'Section'],
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

/*The renderlist() function adds buttons to access and edit documents with 
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
}

/*The renderSelectedDocument() function makes the document and review pane for 
that document given by the document parameter, visible to the user. It begins by
adding the hidden class to the emptyState <div> to hide it and doing the 
opposite to the reviewPane <div> to reveal it. Then the src for the pdfFrame 
<iframe> element is set to the pdf at the address corresponding to the documents 
folder */
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

/**/
function selectDocument(id) {
  state.selectedId = id;
  const document = selectedDocument();
  renderList();
  if (document) renderSelectedDocument(document);
}

/*The applyState() function updates the values held in the index.html elements
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

  renderList();
  renderBatchWarnings();
  if (state.selectedId) selectDocument(state.selectedId);
}

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

/*The loatState() function updates the fields in */
async function loadState() {
  applyState(await requestJson('/api/state'));
}

async function scan() {
  const button = $('scanButton');
  setButtonLoading(button, true, 'Scanning...', 'Scan PDFs');

  try {
    const data = await requestJson('/api/scan', {
      method: 'POST',
      body: JSON.stringify(scanPayload()),
    });
    state.selectedId = data.documents?.[0]?.id || null;
    applyState(data);
    showToast(`Scanned ${state.documents.length} PDF${state.documents.length === 1 ? '' : 's'}.`);
  } catch (error) {
    showToast(error.message, true);
  } finally {
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
        copy: fields.copyFile.checked,
        save_text: fields.saveText.checked,
      }),
    });

    applyState(data);
    showToast(`Filed ${state.documents.length} PDF${state.documents.length === 1 ? '' : 's'} as one batch.`);
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

$('scanButton').addEventListener('click', scan);
$('fileButton').addEventListener('click', fileCurrent);
$('fileAllButton').addEventListener('click', fileAll);

/*This is the final connection between python, html, and javascript.
*/
loadState().catch((error) => showToast(error.message, true));



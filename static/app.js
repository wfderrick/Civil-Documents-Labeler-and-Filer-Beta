const state = { documents: [], settings: {}, selectedId: null };

const $ = (id) => document.getElementById(id);

const fields = {
  inputFolder: $('inputFolder'),
  outputFolder: $('outputFolder'),
  configPath: $('configPath'),
  projectCode: $('projectCode'),
  documentType: $('documentType'),
  lang: $('lang'),
  dpi: $('dpi'),
  ocrDevice: $('ocrDevice'),
  gpuDeviceId: $('gpuDeviceId'),
  ocrWorkers: $('ocrWorkers'),
  ocrThreadsPerWorker: $('ocrThreadsPerWorker'),
  lot: $('lot'),
  address: $('address'),
  editProjectCode: $('editProjectCode'),
  editDocumentType: $('editDocumentType'),
  taxMap: $('taxMap'),
  parcel: $('parcel'),
  taxId: $('taxId'),
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
  const labels = { filed: 'Filed', ready: 'Ready', needs_review: 'Needs review' };
  return labels[document.status] || 'Needs review';
}

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

function renderList() {
  const list = $('documentList');
  list.innerHTML = '';
  $('queueCount').textContent = String(state.documents.length);

  state.documents.forEach((item) => {
    const button = document.createElement('button');
    button.className = `doc-row ${item.id === state.selectedId ? 'active' : ''}`;
    button.innerHTML = `<strong>${item.source_name}</strong><span>${statusLabel(item)}</span>`;
    button.addEventListener('click', () => selectDocument(item.id));
    list.appendChild(button);
  });
}

function renderSelectedDocument(document) {
  $('emptyState').classList.add('hidden');
  $('reviewPane').classList.remove('hidden');
  $('pdfFrame').src = `/documents/${document.id}/pdf`;
  $('documentTitle').textContent = document.source_name;
  $('documentStatus').textContent = statusLabel(document);

  fields.lot.value = document.metadata.lot || '';
  fields.address.value = document.metadata.address || '';
  fields.editProjectCode.value = document.metadata.project_code || '';
  fields.editDocumentType.value = document.metadata.document_type || '';
  fields.taxMap.value = document.metadata.tax_map || '';
  fields.parcel.value = document.metadata.parcel || '';
  fields.taxId.value = document.metadata.tax_id || '';
  fields.folderName.value = document.folder_name || '';
  fields.fileName.value = document.file_name || '';
  $('ocrText').textContent = document.ocr_text || '';
  $('fileButton').disabled = document.status === 'filed';
}

function selectDocument(id) {
  state.selectedId = id;
  const document = selectedDocument();
  renderList();
  if (document) renderSelectedDocument(document);
}

function applyState(data) {
  state.documents = data.documents || [];
  state.settings = data.settings || {};

  fields.inputFolder.value = state.settings.input_folder || '';
  fields.outputFolder.value = state.settings.output_folder || '';
  fields.configPath.value = state.settings.config_path || '';
  fields.projectCode.value = state.settings.project_code || 'Project';
  fields.documentType.value = state.settings.document_type || 'Document';
  fields.lang.value = state.settings.lang || 'en';
  fields.dpi.value = state.settings.dpi || 300;
  if (fields.ocrDevice) fields.ocrDevice.value = state.settings.ocr_device || 'auto';
  if (fields.gpuDeviceId) fields.gpuDeviceId.value = state.settings.gpu_device_id ?? 0;
  if (fields.ocrWorkers) fields.ocrWorkers.value = state.settings.ocr_workers || 1;
  if (fields.ocrThreadsPerWorker) fields.ocrThreadsPerWorker.value = state.settings.ocr_threads_per_worker || 4;

  if (!state.documents.some((document) => document.id === state.selectedId)) {
    state.selectedId = state.documents[0]?.id || null;
  }

  renderList();
  if (state.selectedId) selectDocument(state.selectedId);
}

function scanPayload() {
  return {
    input_folder: fields.inputFolder.value,
    output_folder: fields.outputFolder.value,
    config_path: fields.configPath.value,
    project_code: fields.projectCode.value,
    document_type: fields.documentType.value,
    lang: fields.lang.value,
    dpi: Number(fields.dpi.value),
    ocr_device: fields.ocrDevice ? fields.ocrDevice.value : 'auto',
    gpu_device_id: fields.gpuDeviceId ? Number(fields.gpuDeviceId.value) : 0,
    parallel_ocr: fields.ocrDevice ? fields.ocrDevice.value !== 'gpu' : false,
    ocr_workers: fields.ocrWorkers ? Number(fields.ocrWorkers.value) : 1,
    ocr_threads_per_worker: fields.ocrThreadsPerWorker ? Number(fields.ocrThreadsPerWorker.value) : 4,
  };
}

function updatePayload(autoFolder = false, autoFileName = false) {
  return {
    lot: fields.lot.value,
    address: fields.address.value,
    project_code: fields.editProjectCode.value,
    document_type: fields.editDocumentType.value,
    tax_map: fields.taxMap.value,
    parcel: fields.parcel.value,
    tax_id: fields.taxId.value,
    folder_name: fields.folderName.value,
    file_name: fields.fileName.value,
    auto_folder: autoFolder,
    auto_file_name: autoFileName,
  };
}

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

async function saveCurrent(autoFolder = false, autoFileName = false) {
  const document = selectedDocument();
  if (!document) return null;

  const updated = await requestJson(`/api/documents/${document.id}`, {
    method: 'PATCH',
    body: JSON.stringify(updatePayload(autoFolder, autoFileName)),
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

function registerAutoSave(ids, autoFolder, autoFileName) {
  ids.forEach((id) => {
    $(id).addEventListener('change', () => {
      saveCurrent(autoFolder, autoFileName).catch((error) => showToast(error.message, true));
    });
  });
}

registerAutoSave(['lot', 'address', 'editProjectCode', 'editDocumentType', 'taxMap', 'parcel', 'taxId'], true, true);
registerAutoSave(['folderName', 'fileName'], false, false);

$('scanButton').addEventListener('click', scan);
$('fileButton').addEventListener('click', fileCurrent);
$('fileAllButton').addEventListener('click', fileAll);

loadState().catch((error) => showToast(error.message, true));



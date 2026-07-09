"""IMPORTS AND CONSTANTS SECTION:"""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for

from pipeline_fuzzy_ocr_number_fix import (
    ExtractedMetadata,
    choose_batch_metadata_by_vote,
    enrich_metadata_with_sdat,
    extract_project_code_from_output_folder,
    load_config,
    merge_batch_metadata,
    make_ocr,
    ocr_pdf_batch,
    safe_path_part,
    unique_path,
)

try:
    from paddleocr import PaddleOCR
except Exception:  # pragma: no cover - shown in the UI at runtime
    PaddleOCR = None


APP_DIR = Path(__file__).resolve().parent
STATE_FILE = APP_DIR / ".review_state" / "documents.json"
DEFAULT_CONFIG_PATH = APP_DIR / "config.json"
DEFAULT_STATE: dict[str, Any] = {"settings": {}, "documents": []}
REQUIRED_METADATA_FIELDS = ("lot", "address", "project_code", "document_type")
OPTIONAL_METADATA_FIELDS = ("tax_map", "parcel", "tax_id")

app = Flask("ocr_pipeline_gpu_optimized")
ocr_engine = None
ocr_language = None

"""---------------------------------------------------------------------------------------"""

"""FUNCTION DEFINITION SECTION"""

def api_error(message: str, status_code: int = 500):
    return jsonify({"error": message}), status_code

"""The read_state() function returns all of the current settings and 
document metadata stored in the documents.json file which is in the 
.review_state folder in the project directory. If that file has not
been created yet it returns a default dictionary with empty settings
and documents."""
def read_state() -> dict[str, Any]:
    if not STATE_FILE.exists():
        return dict(DEFAULT_STATE)
    with STATE_FILE.open("r", encoding="utf-8") as state_file:
        return json.load(state_file)


def write_state(state: dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with STATE_FILE.open("w", encoding="utf-8") as state_file:
        json.dump(state, state_file, indent=2)


def get_ocr(lang: str, ocr_device: str = "auto", gpu_device_id: int = 0):
    global ocr_engine, ocr_language
    cache_key = f"{lang}|{ocr_device}|{gpu_device_id}"
    if PaddleOCR is None:
        raise RuntimeError(
            "PaddleOCR is not installed. Run: python -m pip install -r requirements.txt"
        )
    if ocr_engine is None or ocr_language != cache_key:
        ocr_engine = make_ocr(
            lang=lang, ocr_device=ocr_device, gpu_device_id=gpu_device_id
        )
        ocr_language = cache_key
    return ocr_engine


def is_unknown(value: str) -> bool:
    return (
        not value
        or value.lower().startswith("unknown")
        or value in {"Project", "Document"}
    )


""""""
def suggested_folder(metadata: dict[str, str]) -> str:
    return safe_path_part(
        f"Lot {metadata.get('lot', '')} - {metadata.get('address', '')}",
        "Unknown Lot - Unknown Address",
    )


def suggested_filename(metadata: dict[str, str], source_name: str) -> str:
    stem = f"{metadata.get('document_type', '')} - Lot {metadata.get('lot', '')}"
    return safe_path_part(stem, Path(source_name).stem) + ".pdf"


def document_status(metadata: dict[str, str]) -> str:
    return (
        "needs_review"
        if any(
            is_unknown(metadata.get(field, "")) for field in REQUIRED_METADATA_FIELDS
        )
        else "ready"
    )


"""The normalize_document() function returns a checked/corrected document
dictionary. The folder_name, file_name, and status fields are checked, and 
if empty or wrong corrected via the suggested_folder(), 
suggested_filename(), and document_status() functions to match the metadata 
gathered from the file."""
def normalize_document(document: dict[str, Any]) -> dict[str, Any]:
    metadata = document["metadata"]
    document.setdefault("folder_name", suggested_folder(metadata))
    document.setdefault(
        "file_name", suggested_filename(metadata, document["source_name"])
    )
    document["status"] = document_status(metadata)
    return document


def find_document(state: dict[str, Any], document_id: str) -> dict[str, Any] | None:
    return next(
        (doc for doc in state.get("documents", []) if doc.get("id") == document_id),
        None,
    )


""""""


def json_payload() -> dict[str, Any]:
    return request.get_json(force=True) or {}


def resolve_folder(value: str) -> Path:
    return Path(value).expanduser().resolve()


"""Using the data pulled using the get_json function from the request object from"""


def scan_settings(payload: dict[str, Any]) -> dict[str, Any]:
    input_folder_raw = (payload.get("input_folder") or "").strip()
    output_folder_raw = (payload.get("output_folder") or "").strip()

    # Keep advanced/default settings in code/config instead of exposing them in the UI.
    return {
        "input_folder": str(resolve_folder(input_folder_raw)) if input_folder_raw else "",
        "output_folder": str(resolve_folder(output_folder_raw)) if output_folder_raw else "",
        "config_path": (payload.get("config_path") or str(DEFAULT_CONFIG_PATH)).strip(),
        "project_code": (payload.get("project_code") or "").strip(),
        "project_code_override": (payload.get("project_code") or "").strip(),
        "document_type": "Document",
        "lang": "en",
        "dpi": int(payload.get("dpi") or 300),
        "ocr_device": payload.get("ocr_device") or "auto",
        "gpu_device_id": 0,
        "parallel_ocr": False,
        "ocr_workers": 1,
        "ocr_threads_per_worker": 4,
    }

def scan_batch(
    input_folder: Path, ocr, config: dict[str, Any], settings: dict[str, Any]
) -> list[dict[str, Any]]:
    """OCR all PDFs as one related packet and share metadata across the packet.

    OCR can run in parallel. Each worker creates its own PaddleOCR engine, so
    PaddleOCR is not shared across processes. The original lot-search technique
    remains inside the metadata extractor.
    """
    pdfs = sorted(
        path
        for path in input_folder.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    )
    if not pdfs:
        return []

    ocr_device = settings.get("ocr_device", "auto")
    gpu_device_id = int(settings.get("gpu_device_id") or 0)

    # A single GPU should run one OCR engine. Parallel worker processes are kept
    # for CPU fallback, not for one-GPU OCR.
    if str(ocr_device).lower() == "gpu":
        workers = 1
    else:
        workers = (
            settings.get("ocr_workers", 1) if settings.get("parallel_ocr", False) else 1
        )
        workers = max(1, min(int(workers or 1), len(pdfs)))
    threads_per_worker = int(settings.get("ocr_threads_per_worker") or 4)
    print("Begin ocring documents")
    scanned = ocr_pdf_batch(
        pdfs,
        dpi=settings["dpi"],
        lang=settings["lang"],
        workers=workers,
        threads_per_worker=threads_per_worker,
        existing_ocr=ocr if workers == 1 else None,
        ocr_device=ocr_device,
        gpu_device_id=gpu_device_id,
    )
    print("Finished ocring documents")
    print("Begin merging metadata across documents")
    shared_metadata, _votes = choose_batch_metadata_by_vote(
        scanned_documents=scanned,
        config=config,
        default_project_code=settings["project_code"],
        default_document_type=settings["document_type"],
    )
    print("Finished merging metadata across documents")
    documents: list[dict[str, Any]] = []
    print("Begin normalizing documents")
    for scanned_document in scanned:
        metadata = asdict(
            merge_batch_metadata(
                document_text=scanned_document["ocr_text"],
                config=config,
                default_project_code=settings["project_code"],
                default_document_type=settings["document_type"],
                shared_metadata=shared_metadata,
            )
        )
        documents.append(
            normalize_document(
                {
                    "id": uuid.uuid4().hex,
                    "source_path": scanned_document["source_path"],
                    "source_name": scanned_document["source_name"],
                    "ocr_text": scanned_document["ocr_text"],
                    "metadata": metadata,
                    "filed_path": "",
                }
            )
        )
    print("Finished normalizing documents")
    if documents:
        shared_folder = suggested_folder(documents[0]["metadata"])
        for document in documents:
            document["folder_name"] = shared_folder
            document["file_name"] = suggested_filename(
                document["metadata"], document["source_name"]
            )
            normalize_document(document)

    return documents


def refresh_document_names(
    document: dict[str, Any], auto_folder: bool = True, auto_file_name: bool = True
) -> dict[str, Any]:
    metadata = document["metadata"]
    if auto_folder:
        document["folder_name"] = suggested_folder(metadata)
    if auto_file_name:
        document["file_name"] = suggested_filename(metadata, document["source_name"])
    return normalize_document(document)


def load_config_from_state(state: dict[str, Any]) -> dict[str, Any]:
    settings = state.get("settings", {})
    config_path = Path(settings.get("config_path") or DEFAULT_CONFIG_PATH).resolve()
    return load_config(config_path if config_path.exists() else None)


def metadata_from_dict(metadata: dict[str, Any]) -> ExtractedMetadata:
    return ExtractedMetadata(
        lot=str(metadata.get("lot", "Unknown Lot")),
        address=str(metadata.get("address", "Unknown Address")),
        project_code=str(metadata.get("project_code", "Project")),
        document_type=str(metadata.get("document_type", "Document")),
        tax_map=str(metadata.get("tax_map", "")),
        parcel=str(metadata.get("parcel", "")),
        tax_id=str(metadata.get("tax_id", "")),
    )


def refresh_batch_address_from_sdat(state: dict[str, Any]) -> str | None:
    """Use the current batch tax map/parcel/tax ID to refresh the address for all documents."""
    documents = state.get("documents", [])
    if not documents:
        return None

    config = load_config_from_state(state)
    if not config.get("sdat_lookup", True):
        return None

    seed_metadata = metadata_from_dict(documents[0].get("metadata", {}))
    batch_text = "\n".join(document.get("ocr_text", "") for document in documents)
    enriched = enrich_metadata_with_sdat(seed_metadata, batch_text, config)

    if is_unknown(enriched.address):
        return None

    for batch_document in documents:
        batch_document["metadata"]["address"] = enriched.address
        refresh_document_names(batch_document, auto_folder=True, auto_file_name=True)

    return enriched.address


def apply_document_update(
    state: dict[str, Any], document: dict[str, Any], payload: dict[str, Any]
) -> dict[str, Any]:
    metadata = document["metadata"]

    # These are batch-level values. If the user corrects one file, apply the
    # correction to every document in the current batch.
    shared_field_names = ("lot", "address", "tax_map", "parcel", "tax_id", "project_code")
    shared_updates = {field: payload[field] for field in shared_field_names if field in payload}
    changed_field = payload.get("changed_field", "")

    if shared_updates:
        for batch_document in state.get("documents", []):
            batch_document["metadata"].update(shared_updates)
            refresh_document_names(batch_document, auto_folder=True, auto_file_name=True)

    # If the user edits a property identifier, refresh the official SDAT address
    # once and apply that address to every document in the batch.
    if changed_field in {"tax_map", "parcel", "tax_id"}:
        refresh_batch_address_from_sdat(state)

    # Keep non-shared fields document-specific if they are ever posted by older UI/state.
    for field in (*REQUIRED_METADATA_FIELDS, *OPTIONAL_METADATA_FIELDS):
        if field in payload and field not in shared_updates:
            metadata[field] = payload[field]

    if payload.get("auto_folder"):
        document["folder_name"] = suggested_folder(metadata)
    elif "folder_name" in payload:
        document["folder_name"] = safe_path_part(
            payload["folder_name"], suggested_folder(metadata)
        )

    if payload.get("auto_file_name"):
        document["file_name"] = suggested_filename(metadata, document["source_name"])
    elif "file_name" in payload:
        stem = Path(payload["file_name"]).stem
        document["file_name"] = (
            safe_path_part(stem, Path(document["source_name"]).stem) + ".pdf"
        )

    return normalize_document(document)


def file_document_to_output(
    document: dict[str, Any],
    output_folder: Path,
    copy_file: bool = True,
    save_text: bool = False,
    folder_name: str | None = None,
    file_name: str | None = None,
) -> dict[str, Any]:
    source_path = Path(document["source_path"])
    if not source_path.exists():
        raise FileNotFoundError(f"Source PDF no longer exists: {source_path}")

    resolved_folder = safe_path_part(
        folder_name or document.get("folder_name", ""), "Unknown Lot - Unknown Address"
    )
    file_stem = Path(file_name or document.get("file_name", source_path.name)).stem
    resolved_file_name = safe_path_part(file_stem, source_path.stem) + ".pdf"

    destination_folder = output_folder / resolved_folder
    destination_folder.mkdir(parents=True, exist_ok=True)
    destination = unique_path(destination_folder / resolved_file_name)

    if copy_file:
        shutil.copy2(source_path, destination)
    else:
        shutil.move(str(source_path), destination)

    if save_text:
        destination.with_suffix(".txt").write_text(
            document.get("ocr_text", ""), encoding="utf-8"
        )

    document.update(
        {
            "folder_name": resolved_folder,
            "file_name": resolved_file_name,
            "filed_path": str(destination),
            "status": "filed",
        }
    )
    return document

"""-----------------------------------------------------------------------------------------------"""

"""FLASK APP COMMUNICATION SECTION"""

@app.errorhandler(Exception)
def handle_unexpected_error(error):
    if request.path.startswith("/api/"):
        app.logger.exception("API request failed")
        return api_error(str(error) or "Unexpected server error")
    raise error

"""Using the @app.get("/") decorator index() is called when a user opens
or reloads the browser address. index() returns the render_template()
function imported from Flask with index.html as the only parameter.
When render_template() is called it automatically searches the project
directory(aka the drectory app.py is in) for a folder named templates
containing HTML files index.html is the chosen name for this app, but
it can be named anything. Render_template then opens and runs the HTML
contained in index.html."""
@app.get("/")
def index():
    return render_template("index.html")

@app.get("/favicon.ico")
def favicon():
    return "", 204


"""The api_state() function returns a Response object holding the current 
settings and documents. The read_state() function retrieves the settings
and documents from the documents.json file in the .review_state folder in 
project directory. """
@app.get("/api/state")
def api_state():
    state = read_state()
    state["documents"] = [
        normalize_document(document) for document in state.get("documents", [])
    ]
    return jsonify(state)


@app.get("/api/browse-folders")
def api_browse_folders():
    requested = request.args.get("path") or str(Path.home())
    current = Path(requested).expanduser().resolve()

    if not current.exists() or not current.is_dir():
        return api_error(f"Folder not found: {current}", 400)

    folders = []
    for path in sorted(current.iterdir()):
        if path.is_dir():
            folders.append({"name": path.name, "path": str(path)})

    return jsonify(
        {
            "current": str(current),
            "parent": str(current.parent) if current.parent != current else "",
            "folders": folders,
        }
    )


@app.post("/api/scan")
def api_scan():
    settings = scan_settings(json_payload())
    input_folder = Path(settings["input_folder"])
    output_folder = Path(settings["output_folder"])

    if not settings["input_folder"]:
        return api_error("Input folder is required.", 400)
    if not settings["output_folder"]:
        return api_error("Output folder is required.", 400)
    if not input_folder.is_dir():
        return api_error(f"Input folder not found: {input_folder}", 400)

    output_folder.mkdir(parents=True, exist_ok=True)
    config_path = (
        Path(settings["config_path"]).resolve() if settings["config_path"] else None
    )
    config = load_config(config_path if config_path and config_path.exists() else None)
    manual_project_code = (settings.get("project_code") or "").strip()
    if manual_project_code:
        settings["project_code"] = safe_path_part(manual_project_code.upper(), "Project")
        settings["project_code_override"] = settings["project_code"]
    else:
        settings["project_code"] = extract_project_code_from_output_folder(
            output_folder,
            config,
            "Project",
        )
        settings["project_code_override"] = ""
    use_single_engine = (
        str(settings.get("ocr_device", "auto")).lower() == "gpu"
        or not settings.get("parallel_ocr", False)
        or int(settings.get("ocr_workers", 1)) <= 1
    )
    ocr = (
        get_ocr(
            settings["lang"],
            settings.get("ocr_device", "auto"),
            int(settings.get("gpu_device_id") or 0),
        )
        if use_single_engine
        else None
    )

    state = {
        "settings": settings,
        "documents": scan_batch(input_folder, ocr, config, settings),
    }

    write_state(state)
    return jsonify(state)


@app.patch("/api/documents/<document_id>")
def api_update_document(document_id: str):
    state = read_state()
    document = find_document(state, document_id)
    if not document:
        return api_error("Document not found", 404)

    updated = apply_document_update(state, document, json_payload())
    write_state(state)
    return jsonify(
        {
            "settings": state.get("settings", {}),
            "documents": [
                normalize_document(doc) for doc in state.get("documents", [])
            ],
            "updated": updated,
        }
    )


@app.post("/api/documents/<document_id>/file")
def api_file_document(document_id: str):
    payload = request.get_json(silent=True) or {}
    state = read_state()
    document = find_document(state, document_id)
    if not document:
        return api_error("Document not found", 404)

    output_folder = (
        Path(state.get("settings", {}).get("output_folder", "")).expanduser().resolve()
    )
    if not str(output_folder):
        return api_error("Output folder is not configured.", 400)

    try:
        filed = file_document_to_output(
            document,
            output_folder,
            copy_file=payload.get("copy", True),
            save_text=payload.get("save_text", False),
            folder_name=payload.get("folder_name"),
            file_name=payload.get("file_name"),
        )
    except FileNotFoundError:
        return api_error("File not located in specified input folder anymore.", 400)

    write_state(state)
    return jsonify(filed)


@app.post("/api/file-all")
def api_file_all_documents():
    payload = request.get_json(silent=True) or {}
    state = read_state()
    output_folder = (
        Path(state.get("settings", {}).get("output_folder", "")).expanduser().resolve()
    )
    if not str(output_folder):
        return api_error("Output folder is not configured.", 400)

    documents = state.get("documents", [])
    if not documents:
        return api_error("No documents to file.", 400)

    copy_file = payload.get("copy", True)
    save_text = payload.get("save_text", False)
    shared_folder = documents[0].get("folder_name") or suggested_folder(
        documents[0]["metadata"]
    )

    filed_documents = []
    try:
        for document in documents:
            filed_documents.append(
                file_document_to_output(
                    document,
                    output_folder,
                    copy_file=copy_file,
                    save_text=save_text,
                    folder_name=shared_folder,
                )
            )
    except FileNotFoundError as error:
        return api_error(str(error), 400)

    write_state(state)
    return jsonify(
        {"settings": state.get("settings", {}), "documents": filed_documents}
    )


@app.get("/documents/<document_id>/pdf")
def document_pdf(document_id: str):
    try:
        document = find_document(read_state(), document_id)
        if document:
            return send_file(
                Path(document["source_path"]),
                mimetype="application/pdf",
                as_attachment=False,
            )
        return redirect(url_for("index"))
    except FileNotFoundError:
        return send_file(
            Path("file-cant-found-2.pdf"),
            mimetype="application/pdf",
            as_attachment=False,
        )


"""Where it all begins. app is the Flask object created in the imports
and constants section. This object connects all of the files together.
The run function runs app.py on a local development server. For this
project this was the easiest way to create a user interface to interact
with the documents being processed. Once this is created it is stagnant
until a user opens the the browser with the address 
http://127.0.0.1:5055. Once the user does this app uses the 
@app.get("/") decorator to call the index() function. This means that 
when the browser requests GET http://localhost:5055/(Which happens as 
soon as you open the above address) the app object searches through
defined routes and finds @app.get("/") pointing to the index() function
and knows to call it."""
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5055, debug=True)

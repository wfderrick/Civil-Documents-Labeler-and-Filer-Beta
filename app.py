"""IMPORTS AND CONSTANTS SECTION:"""

from __future__ import annotations

import threading
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request, send_file

from ocr_service import make_ocr, ocr_pdf_batch

from metadata_extraction import load_config

from pipeline import (
    LOOKUP_DOCUMENT_TYPE,
    choose_batch_metadata_by_vote,
    merge_batch_metadata,
    safe_path_part,
)

from document_service import (
    apply_document_update,
    file_document_to_output,
    find_document,
    suggested_filename,
    suggested_folder,
    sync_document_metadata,
)
from scan_status import (
    add_scan_progress,
    finish_scan_progress,
    reset_scan_progress,
    scan_progress_snapshot,
)
from state_store import (
    read_state,
    write_state,
    update_output_folder_setting,
)
from tracker import append_batch_tracker

try:
    from paddleocr import PaddleOCR
except Exception:  # pragma: no cover - shown in the UI at runtime
    PaddleOCR = None


APP_DIR = Path(__file__).resolve().parent
STATE_FILE = APP_DIR / ".review_state" / "documents.json"
TRACKER_DIR = Path(r"C:\ocr tracker")
TRACKER_FILE = TRACKER_DIR / "filed_batches.csv"
DEFAULT_CONFIG_PATH = APP_DIR / "config.json"
DEFAULT_STATE: dict[str, Any] = {"settings": {}, "documents": []}
REQUIRED_METADATA_FIELDS = ("lot", "address", "project_code", "document_type")
OPTIONAL_METADATA_FIELDS = ("tax_map", "parcel", "tax_id", "section")

app = Flask("ocr_pipeline_gpu_optimized")
ocr_engine = None
ocr_language = None

_SCAN_PROGRESS_LOCK = threading.Lock()
_SCAN_PROGRESS: dict[str, Any] = {
    "active": False,
    "finished": False,
    "failed": False,
    "started_at": 0.0,
    "messages": [],
}


"""---------------------------------------------------------------------------------------"""

"""FUNCTION DEFINITION SECTION"""


def api_error(message: str, status_code: int = 500):
    """The api_error() function returns a Response object and integer holding
    an error message as a json and status code."""
    return jsonify({"error": message}), status_code


def get_ocr(lang: str, ocr_device: str = "auto", gpu_device_id: int = 0):
    """The get_ocr() function returns a pointer to a PaddleOCR object. If there
    isn't already a selected ocr_engine a new one is created using the
    make_ocr() fucntion."""
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


def json_payload() -> dict[str, Any]:
    """The json_payload() function returns the fields given in the body of the
    POST or PATCH request from the browser these include scan settings and
    document metadata. This is done using the build in get_json() function for
    the request object imported from Flask.
    """
    return request.get_json(force=True) or {}


def resolve_folder(value: str) -> Path:
    """The resolve_folder() function returns a fully expanded and resolved
    path ~. If the tilde is used expanduser() replaces with the users home
    directory. resolve() then removes any relative paths such as
    documents/.../project1 to ensure an absolute path."""
    return Path(value).expanduser().resolve()


def scan_settings(payload: dict[str, Any]) -> dict[str, Any]:
    """The scan_settings() function returns a dictionary with the current scan
    settings set by the payload parameter.

    input folder, output folder, config path, project code, project code, dpi,
    and ocr device are all pulled from the parameter the get() function. The
    other settings document type, language, gpu device id, parrallel ocr, ocr
    workers, and threads per worker are all kept as defaults which can be
    changed by the user manually in this function if necessary. Both input and
    output folders are put through the  resolve_folder() function to make sure
    it is a valid path."""
    input_folder_raw = (payload.get("input_folder") or "").strip()
    output_folder_raw = (payload.get("output_folder") or "").strip()

    return {
        "input_folder": (
            str(resolve_folder(input_folder_raw)) if input_folder_raw else ""
        ),
        "output_folder": (
            str(resolve_folder(output_folder_raw)) if output_folder_raw else ""
        ),
        "config_path": (payload.get("config_path") or str(DEFAULT_CONFIG_PATH)).strip(),
        "project_code": (payload.get("project_code") or "").strip(),
        "project_code_override": (payload.get("project_code") or "").strip(),
        "document_type": "Field Notes",
        "lang": "en",
        "dpi": int(payload.get("dpi") or 300),
        "ocr_device": payload.get("ocr_device") or "auto",
        "gpu_device_id": 0,
        "parallel_ocr": False,
        "ocr_workers": 1,
        "ocr_threads_per_worker": 4,
    }


def scan_batch(
    input_folder: Path,
    ocr,
    config: dict[str, Any],
    settings: dict[str, Any],
    progress_callback=None,
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
    report = progress_callback or (lambda _message: None)
    report(f"Found {len(pdfs)} PDF{'s' if len(pdfs) != 1 else ''} to scan.")
    report("Beginning OCR processing.")
    scanned = ocr_pdf_batch(
        pdfs,
        dpi=settings["dpi"],
        lang=settings["lang"],
        workers=workers,
        threads_per_worker=threads_per_worker,
        existing_ocr=ocr if workers == 1 else None,
        ocr_device=ocr_device,
        gpu_device_id=gpu_device_id,
        progress_callback=report,
    )
    report("Finished OCR processing.")
    report("Beginning metadata voting and SDAT enrichment.")
    shared_metadata, metadata_votes = choose_batch_metadata_by_vote(
        scanned_documents=scanned,
        config=config,
        default_project_code=settings["project_code"],
        default_document_type=settings["document_type"],
    )
    report("Finished metadata voting and SDAT enrichment.")
    documents: list[dict[str, Any]] = []
    report("Preparing documents for review.")
    for scanned_document, metadata_vote in zip(scanned, metadata_votes):
        is_lookup = metadata_vote.document_type == LOOKUP_DOCUMENT_TYPE
        final_metadata = (
            metadata_vote
            if is_lookup
            else merge_batch_metadata(
                document_text=scanned_document["ocr_text"],
                config=config,
                default_project_code=settings["project_code"],
                default_document_type=settings["document_type"],
                shared_metadata=shared_metadata,
                document_metadata=metadata_vote,
            )
        )
        documents.append(
            sync_document_metadata(
                {
                    "id": uuid.uuid4().hex,
                    "source_path": scanned_document["source_path"],
                    "source_name": scanned_document["source_name"],
                    "ocr_text": scanned_document["ocr_text"],
                    "ocr_pages": scanned_document.get("ocr_pages", []),
                    "metadata": asdict(final_metadata),
                    "is_lookup_document": is_lookup,
                    "filed_path": "",
                }
            )
        )
    report("Finished preparing documents for review.")
    normal_documents = [
        document for document in documents if not document.get("is_lookup_document")
    ]
    if normal_documents:
        shared_folder = suggested_folder(normal_documents[0]["metadata"])
        for document in normal_documents:
            document["folder_name"] = shared_folder
            document["file_name"] = suggested_filename(
                document["metadata"], document["source_name"]
            )
            sync_document_metadata(document)

    return documents


def _folder_project_and_section(output_folder: Path) -> tuple[str, str]:
    """The _folder_project_and_section() function returns the project code and
    section taken from the output_folder parameter. It splits the parameter on
    the . and the - to determine the project code and section and returns both.
    """
    name = output_folder.name.strip()
    if "." not in name:
        return name, ""
    project_code, section = name.split(".", 1)
    try:
        section, extra = section.split("-", 1)
        return project_code.strip(), section.strip()

    finally:
        return project_code.strip(), section.strip()


"""-----------------------------------------------------------------------------------------------"""

"""FLASK APP COMMUNICATION SECTION"""


@app.errorhandler(Exception)
def handle_unexpected_error(error):
    if request.path == "/api/scan":
        finish_scan_progress(failed=True, message=f"Scan failed: {error}")
    if request.path.startswith("/api/"):
        app.logger.exception("API request failed")
        return api_error(str(error) or "Unexpected server error")
    raise error


@app.get("/")
def index():
    """Using the @app.get("/") decorator index() is called when a user opens
    or reloads the browser address. index() returns the render_template()
    function imported from Flask with index.html as the only parameter.
    When render_template() is called it automatically searches the project
    directory(aka the drectory app.py is in) for a folder named templates
    containing HTML files index.html is the chosen name for this app, but
    it can be named anything. Render_template then opens and runs the HTML
    contained in index.html."""
    return render_template("index.html")


@app.get("/favicon.ico")
def favicon():
    return "", 204


@app.get("/api/state")
def api_state():
    """The api_state() function returns a Response object holding the current
    settings and documents. The read_state() function retrieves the settings
    and documents from the documents.json file in the .review_state folder in
    project directory. Each document is then checked and corrected version based
    on the metadata contained in the documents.json file. jsonify() is then
    called on the state variable which returns a Response object containing the
    json pulled from documents.json."""
    state = read_state()
    state["documents"] = [
        sync_document_metadata(document) for document in state.get("documents", [])
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


@app.patch("/api/settings/output-folder")
def api_update_output_folder():
    state = read_state()
    try:
        output_folder = update_output_folder_setting(
            state,
            (request.get_json(silent=True) or {}).get("output_folder", ""),
        )
    except (OSError, ValueError) as error:
        return api_error(str(error), 400)
    write_state(state)
    return jsonify(
        {
            "output_folder": str(output_folder),
            "settings": state.get("settings", {}),
        }
    )


@app.get("/api/scan-progress")
def api_scan_progress():
    return jsonify(scan_progress_snapshot())


@app.post("/api/scan")
def api_scan():
    """Scan the selected batch while publishing progress to the browser."""
    reset_scan_progress()
    add_scan_progress("Starting scan request.")
    settings = scan_settings(json_payload())
    input_folder = Path(settings["input_folder"])
    output_folder = Path(settings["output_folder"])

    if not settings["input_folder"]:
        finish_scan_progress(
            failed=True, message="Scan failed: Input folder is required."
        )
        return api_error("Input folder is required.", 400)
    if not settings["output_folder"]:
        finish_scan_progress(
            failed=True, message="Scan failed: Output folder is required."
        )
        return api_error("Output folder is required.", 400)
    if not input_folder.is_dir():
        finish_scan_progress(
            failed=True,
            message=f"Scan failed: Input folder not found: {input_folder}",
        )
        return api_error(f"Input folder not found: {input_folder}", 400)

    output_folder.mkdir(parents=True, exist_ok=True)
    config_path = (
        Path(settings["config_path"]).resolve() if settings["config_path"] else None
    )
    config = load_config(config_path if config_path and config_path.exists() else None)
    detected_project_code, detected_section = _folder_project_and_section(output_folder)
    settings["section"] = detected_section
    manual_project_code = (settings.get("project_code") or "").strip()
    if manual_project_code:
        settings["project_code"] = safe_path_part(
            manual_project_code.upper(), "Project"
        )
        settings["project_code_override"] = settings["project_code"]
    else:
        settings["project_code"] = safe_path_part(detected_project_code, "Project")
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
        "documents": scan_batch(
            input_folder,
            ocr,
            config,
            settings,
            progress_callback=add_scan_progress,
        ),
    }
    if settings.get("section"):
        for document in state["documents"]:
            document["metadata"]["section"] = settings["section"]

    write_state(state)
    finish_scan_progress(
        message=f"Scan complete. {len(state['documents'])} document(s) ready for review."
    )
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
                sync_document_metadata(doc) for doc in state.get("documents", [])
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
    if document.get("is_lookup_document"):
        return api_error(
            "Lookup-only SDAT records are removed after the batch is filed.",
            400,
        )

    try:
        output_folder = update_output_folder_setting(
            state,
            payload.get("output_folder")
            or state.get("settings", {}).get("output_folder", ""),
        )
    except (OSError, ValueError) as error:
        return api_error(str(error), 400)

    try:
        filed = file_document_to_output(
            document,
            output_folder,
            copy_file=payload.get("copy", False),
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
    try:
        output_folder = update_output_folder_setting(
            state,
            payload.get("output_folder")
            or state.get("settings", {}).get("output_folder", ""),
        )
    except (OSError, ValueError) as error:
        return api_error(str(error), 400)
    documents = state.get("documents", [])
    normal_documents = [doc for doc in documents if not doc.get("is_lookup_document")]
    lookup_documents = [doc for doc in documents if doc.get("is_lookup_document")]
    if not normal_documents:
        return api_error("No permanent documents to file.", 400)
    shared_folder = normal_documents[0].get("folder_name") or suggested_folder(
        normal_documents[0]["metadata"]
    )
    filed_documents = []
    try:
        for document in normal_documents:
            filed_documents.append(
                file_document_to_output(
                    document,
                    output_folder,
                    copy_file=payload.get("copy", False),
                    save_text=payload.get("save_text", False),
                    folder_name=shared_folder,
                )
            )
    except FileNotFoundError as error:
        return api_error(str(error), 400)

    try:
        append_batch_tracker(normal_documents, output_folder, filed_documents)
    except OSError as error:
        return api_error(
            f"Documents were filed, but the tracker could not be updated: {error}",
            500,
        )

    # Delete lookup-only source records only after every permanent document succeeds.
    for lookup in lookup_documents:
        source = Path(lookup.get("source_path", ""))
        if source.exists():
            try:
                from send2trash import send2trash

                send2trash(str(source))
            except Exception:
                source.unlink(missing_ok=True)
    state["documents"] = []
    write_state(state)
    return jsonify({"settings": state.get("settings", {}), "documents": []})


@app.get("/documents/<document_id>/pdf")
def document_pdf(document_id: str):
    """The document_pdf() function returns a response object which holds a pdf
    with either a selected document or file-not-found.pdf if there is
    an error. It begins by trying to run find_document() with parameters
    read_state() and document_id. If a document is found matching those
    parameters it's information is returned otherwise None is returned. If that
    is successful the function returns a Response object via the send_file()
    function with the path to the document, document type, and as_attachment
    parameters. If this sends an error or if the find_document() function
    returned none the file-not-found.pdf file in the project directory is
    displayed via the send_file() function."""
    try:
        document = find_document(read_state(), document_id)
        if document:
            return send_file(
                Path(document["source_path"]),
                mimetype="application/pdf",
                as_attachment=False,
            )
        return send_file(
            Path("file-not-found.pdf"),
            mimetype="application/pdf",
            as_attachment=False,
        )
    except FileNotFoundError:
        return send_file(
            Path("file-not-found.pdf"),
            mimetype="application/pdf",
            as_attachment=False,
        )


"""Where it all begins. app is the Flask object created in the imports and constants section. This object connects all of the files together. The run function runs app.py on a local development server. For this project this was the easiest way to create a user interface to interact with the documents being processed. Once this is created it is stagnant until a user opens the the browser with the address http://127.0.0.1:5055. Once the user does this app uses the @app.get("/") decorator to call the index() function. This means that 
when the browser requests GET http://localhost:5055/(Which happens as soon as you open the above address) the app object searches through defined routes and finds @app.get("/") pointing to the index() function and knows to call it."""
if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5055, debug=True)

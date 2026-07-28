"""Flask entry point and traffic controller for the application.

This file connects the browser interface to the OCR, metadata, SDAT,
review-state, and filing modules. It does not contain the low-level OCR
or matching algorithms itself. Instead, each route validates a browser
request, calls the appropriate service functions, and turns the result
back into JSON that ``static/app.js`` can display.

Main scan flow:
    Browser POST /api/scan
        -> validate folders and settings
        -> load configuration
        -> start/reuse PaddleOCR
        -> scan in Batch or Mass mode
        -> save review records to .review_state/documents.json
        -> browser polls /api/state and /api/scan-progress

Main filing flow:
    Browser POST /api/documents/<id>/file or /api/file-all
        -> validate the reviewed document
        -> call document_service.file_document_to_output()
        -> write PDF metadata and move/copy the file
        -> remove completed items from the review queue

Beginner note:
    Flask decorators such as ``@app.get`` and ``@app.post`` connect a
    URL to the Python function directly below the decorator. The browser
    never calls most service functions itself; it calls these routes,
    and the routes call the service functions on its behalf."""

from __future__ import annotations

import threading
import time
import uuid
from dataclasses import asdict
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request, send_file

from document_service import (
    apply_document_update,
    file_document_to_output,
    find_document,
    suggested_filename,
    suggested_folder,
    sync_document_metadata,
)
from metadata_extraction import load_config
from ocr_service import get_cached_ocr, ocr_pdf_batch
from pipeline import (
    LOOKUP_DOCUMENT_TYPE,
    choose_batch_metadata_by_vote,
    merge_batch_metadata,
)
from scan_status import (
    add_scan_progress,
    finish_scan_progress,
    reset_scan_progress,
    scan_progress_snapshot,
    set_scan_timing,
)
from state_store import (
    append_document,
    clear_documents,
    read_state,
    remove_document,
    replace_state,
    update_document,
    update_output_folder,
)
from tracker import append_batch_tracker

APP_DIR = Path(__file__).resolve().parent
STATE_FILE = APP_DIR / ".review_state" / "documents.json"
TRACKER_DIR = Path(r"C:\ocr tracker")
TRACKER_FILE = TRACKER_DIR / "filed_batches.csv"
DEFAULT_CONFIG_PATH = APP_DIR / "config.json"
DEFAULT_STATE: dict[str, Any] = {"settings": {}, "documents": []}
REQUIRED_METADATA_FIELDS = ("lot", "address", "project_code", "document_type")
OPTIONAL_METADATA_FIELDS = ("tax_map", "parcel", "tax_id", "section")

app = Flask("ocr_pipeline_gpu_optimized")

_SCAN_PROGRESS_LOCK = threading.Lock()
_SCAN_PROGRESS: dict[str, Any] = {
    "active": False,
    "finished": False,
    "failed": False,
    "started_at": 0.0,
    "messages": [],
}


# ============================================================================
# INTERNAL HELPERS AND SCAN WORKFLOWS
# ============================================================================
# These functions are called by the Flask routes later in this file. They do not
# become browser URLs until a route-decorated function calls them.

def api_error(message: str, status_code: int = 500):
    """Create the error response format expected by the browser.
    
    Every JavaScript request looks for an ``error`` key when a request
    fails. Centralizing that shape here keeps all Flask routes consistent
    and pairs the JSON body with the correct HTTP status code."""
    return jsonify({"error": message}), status_code


def json_payload() -> dict[str, Any]:
    """Read a JSON request body and always return a dictionary.
    
    Scan settings and document edits arrive from ``static/app.js`` as
    JSON. ``force=True`` asks Flask to decode that body even when the
    browser's content-type header is imperfect. An empty body becomes
    ``{}``, allowing callers to use ``payload.get(...)`` safely."""
    return request.get_json(force=True) or {}


def resolve_folder(value: str) -> Path:
    """Convert a folder typed in the browser into one unambiguous Path.
    
    ``expanduser`` changes ``~`` into the current user's home directory.
    ``resolve`` converts relative pieces such as ``..`` into an absolute
    location. The app stores that stable absolute path in its state file
    so later scan and filing requests refer to the same folder."""
    return Path(value).expanduser().resolve()


def scan_settings(payload: dict[str, Any]) -> dict[str, Any]:
    """Translate the browser's scan form into the settings used by Python.
    
    The web form sends strings, checkboxes, and optional values. This
    function trims text, resolves folder paths, converts DPI to an integer,
    fills internal defaults that are not exposed in the interface, and
    normalizes the scan mode to lowercase.
    
    The returned dictionary is saved with the review state and passed to
    both Batch and Mass scan functions. Keeping this conversion in one
    place prevents each scan path from interpreting the same form differently."""
    input_folder_raw = (payload.get("input_folder") or "").strip()
    output_folder_raw = (payload.get("output_folder") or "").strip()

    return {
        "input_folder": (
            str(resolve_folder(input_folder_raw)) if input_folder_raw else ""
        ),
        "output_folder": (
            str(resolve_folder(output_folder_raw)) if output_folder_raw else ""
        ),
        "config_path": (
            payload.get("config_path") or str(DEFAULT_CONFIG_PATH)
        ).strip(),
        "project_code": (payload.get("project_code") or "").strip(),
        "project_code_override": (payload.get("project_code") or "").strip(),
        "county": (payload.get("county") or "Calvert").strip(),
        "document_type": "Field Notes",
        "lang": "en",
        "dpi": int(payload.get("dpi") or 300),
        "ocr_device": payload.get("ocr_device") or "auto",
        "gpu_device_id": 0,
        "parallel_ocr": False,
        "ocr_workers": 1,
        "ocr_threads_per_worker": 4,
        "scan_mode": (payload.get("scan_mode") or "batch").strip().lower(),
        "in_place": bool(payload.get("in_place", False)),
    }


def scan_batch(
    input_folder: Path,
    ocr,
    config: dict[str, Any],
    settings: dict[str, Any],
    progress_callback=None,
) -> list[dict[str, Any]]:
    """Process every PDF in a folder as one related project packet.
    
    App role:
        Batch mode assumes the PDFs describe the same property. It OCRs
        the whole set, lets repeated property values vote, enriches the
        shared result with SDAT, and prepares one review record per PDF.
    
    How it works:
        1. Collect PDFs in stable filename order.
        2. Choose one GPU worker or the configured number of CPU workers.
        3. Run OCR and retain text plus page coordinates.
        4. Call ``choose_batch_metadata_by_vote`` for classification,
           shared-field voting, visual correction, and SDAT enrichment.
        5. Merge shared property fields into each permanent drawing while
           preserving its own document type.
        6. Give the packet one shared destination folder and generate each
           PDF's suggested filename and review status.
    
    The returned dictionaries are ready to be written to the state store;
    this function does not move the original PDFs."""
    # STEP 1 - Freeze the packet order.
    # Sorting makes OCR, voting, and the review list deterministic. Without
    # this, filesystem order could change which document wins an equal vote.
    pdfs = sorted(
        path
        for path in input_folder.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    )
    if not pdfs:
        return []

    ocr_device = settings.get("ocr_device", "auto")
    gpu_device_id = int(settings.get("gpu_device_id") or 0)

    # A single GPU should run one OCR engine. Parallel worker processes afre kept
    # for CPU fallback, not for one-GPU OCR.
    if str(ocr_device).lower() == "gpu":
        workers = 1
    else:
        workers = (
            settings.get("ocr_workers", 1)
            if settings.get("parallel_ocr", False)
            else 1
        )
        workers = max(1, min(int(workers or 1), len(pdfs)))
    threads_per_worker = int(settings.get("ocr_threads_per_worker") or 4)
    report = progress_callback or (lambda _message: None)
    report(f"Found {len(pdfs)} PDF{'s' if len(pdfs) != 1 else ''} to scan.")
    report("Beginning OCR processing.")
    ocr_started = time.perf_counter()
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
    ocr_elapsed = time.perf_counter() - ocr_started
    set_scan_timing("ocr_total", ocr_elapsed)
    report(f"Finished OCR processing in {ocr_elapsed:.2f} seconds.")
    report("Beginning metadata voting and SDAT enrichment.")
    metadata_started = time.perf_counter()
    def performance_output(message: str) -> None:
        # Print every profiler line to the terminal for easy copy/paste, while
        # also showing it in the application's progress feed.
        """Send one profiler line to both places a developer can observe it.
        
        Printing makes the detailed report easy to copy from the terminal.
        Forwarding the same line to ``report`` also exposes it in the browser's
        progress panel. This nested callback is passed into the batch pipeline."""
        print(message, flush=True)
        report(message)

    # STEP 3 - Turn separate OCR opinions into one property record.
    # The pipeline keeps each document type independent but votes on fields
    # that should be common to every drawing in this project packet.
    shared_metadata, metadata_votes = choose_batch_metadata_by_vote(
        scanned_documents=scanned,
        config=config,
        default_project_code=settings["project_code"],
        default_document_type=settings["document_type"],
        performance_callback=performance_output,
    )
    metadata_elapsed = time.perf_counter() - metadata_started
    set_scan_timing("metadata_and_sdat", metadata_elapsed)
    report(
        f"Finished metadata voting and SDAT enrichment in {metadata_elapsed:.2f} seconds."
    )
    # STEP 4 - Convert service-layer dataclasses into JSON-friendly review
    # records. These dictionaries are what the browser edits and what the
    # state store persists between requests.
    documents: list[dict[str, Any]] = []
    review_started = time.perf_counter()
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
    normal_documents = [
        document
        for document in documents
        if not document.get("is_lookup_document")
    ]
    if normal_documents:
        shared_folder = suggested_folder(normal_documents[0]["metadata"])
        for document in normal_documents:
            document["folder_name"] = shared_folder
            document["file_name"] = suggested_filename(
                document["metadata"], document["source_name"]
            )

    # Synchronize each record once, after shared names and folders are assigned.
    # Older versions synchronized normal documents twice during every batch.
    for document in documents:
        sync_document_metadata(document)

    review_elapsed = time.perf_counter() - review_started
    set_scan_timing("review_record_preparation", review_elapsed)
    report(f"Finished preparing documents for review in {review_elapsed:.2f} seconds.")
    return documents


def scan_mass(
    input_folder: Path,
    ocr,
    config: dict[str, Any],
    settings: dict[str, Any],
    progress_callback=None,
    document_ready_callback=None,
) -> list[dict[str, Any]]:
    """Process a folder as independent PDF jobs and publish each result early.
    
    Mass mode is designed for a folder containing unrelated properties.
    Each PDF is OCRed, classified, SDAT-checked, named, and appended to
    the persistent review queue before the next PDF begins. The browser
    can therefore review completed items while later files are still scanning.
    
    Unlike Batch mode, this function disables cross-document voting and
    duplicate-type correction. It also requests strict address matching so
    an ambiguous SDAT result cannot assign one parcel's Tax ID to another job."""
    pdfs = sorted(
        path
        for path in input_folder.iterdir()
        if path.is_file() and path.suffix.lower() == ".pdf"
    )
    if not pdfs:
        return []

    report = progress_callback or (lambda _message: None)
    publish = document_ready_callback or (lambda _document: None)
    documents: list[dict[str, Any]] = []
    total = len(pdfs)
    report(f"Found {total} PDF{'s' if total != 1 else ''} to mass scan.")

    for number, pdf_path in enumerate(pdfs, start=1):
        report(f"Mass scan document {number} of {total}: {pdf_path.name}")
        scanned_document = ocr_pdf_batch(
            [pdf_path],
            dpi=settings["dpi"],
            lang=settings["lang"],
            workers=1,
            threads_per_worker=int(
                settings.get("ocr_threads_per_worker") or 4
            ),
            existing_ocr=ocr,
            ocr_device=settings.get("ocr_device", "auto"),
            gpu_device_id=int(settings.get("gpu_device_id") or 0),
            progress_callback=report,
        )[0]

        shared_metadata, metadata_votes = choose_batch_metadata_by_vote(
            scanned_documents=[scanned_document],
            config=config,
            default_project_code=settings["project_code"],
            default_document_type=settings["document_type"],
            resolve_duplicate_document_types=False,
            strict_independent_lookup=True,
        )
        metadata_vote = metadata_votes[0]
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
        document = sync_document_metadata(
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
        if settings.get("section"):
            document["metadata"]["section"] = settings["section"]

        if not is_lookup:
            document["folder_name"] = suggested_folder(document["metadata"])
            document["file_name"] = suggested_filename(
                document["metadata"], document["source_name"]
            )
            sync_document_metadata(document)

        documents.append(document)
        publish(document)
        report(f"Ready for review: {pdf_path.name}")

    return documents


def _folder_project_and_section(output_folder: Path) -> tuple[str, str]:
    """Infer project and section labels from the selected project folder name.
    
    COA folders may follow ``PROJECT.SECTION-Description``. The text before
    the first period becomes the project code; the text after it, but before
    an optional dash, becomes the section. A folder without a period supplies
    only a project code.
    
    ``api_scan`` uses these values when the user leaves the project-code box
    blank, reducing duplicate data entry while preserving a manual override."""
    name = output_folder.name.strip()

    if "." not in name:
        return name, ""

    project_code, section = name.split(".", 1)
    section = section.split("-", 1)[0]

    return project_code.strip(), section.strip()


# ============================================================================
# FLASK ROUTES: THE BROWSER'S ENTRY POINTS INTO PYTHON
# ============================================================================
# Each @app.get/@app.post/@app.patch decorator connects a URL used by app.js to
# the function directly below it. Route functions validate requests and delegate
# the actual OCR, metadata, state, or filing work to the service modules.

@app.errorhandler(Exception)
def handle_unexpected_error(error):
    """Turn uncaught server exceptions into responses the interface can understand.
    
    Flask calls this function only when a route did not handle an exception
    itself. Scan failures also mark the shared progress record as failed so
    polling stops with a useful message. API routes receive JSON because the
    JavaScript client cannot render Flask's normal HTML error page. Non-API
    errors are re-raised so Flask retains its usual debugging behavior."""
    if request.path == "/api/scan":
        finish_scan_progress(failed=True, message=f"Scan failed: {error}")
    if request.path.startswith("/api/"):
        app.logger.exception("API request failed")
        return api_error(str(error) or "Unexpected server error")
    raise  # noqa: PLE0704


@app.get("/")
def index():
    """Serve the application's single review page.
    
    Opening ``http://127.0.0.1:5055/`` reaches this route. Flask loads
    ``templates/index.html``; that page then loads ``static/app.js`` and
    ``static/styles.css`` and begins requesting the saved application state."""
    return render_template("index.html")


@app.get("/favicon.ico")
def favicon():
    """Answer the browser's automatic favicon request without logging a 404.
    
    The app does not currently ship an icon, so a successful empty response
    is quieter and has no effect on scanning or review behavior."""
    return "", 204


@app.get("/api/state")
def api_state():
    """Return the latest settings and review queue to the browser.
    
    State is read from disk, then every document is re-synchronized so its
    suggested names and status reflect its current metadata. The browser uses
    this endpoint at startup and while Mass Scan is publishing new documents."""
    state = read_state()
    state["documents"] = [
        sync_document_metadata(document)
        for document in state.get("documents", [])
    ]
    return jsonify(state)


@app.get("/api/browse-folders")
def api_browse_folders():
    """Provide one level of filesystem information for the folder-picker dialog.
    
    The route resolves the requested directory, rejects missing or non-folder
    paths, lists child directories only, and also returns the parent path.
    JavaScript repeatedly calls this endpoint as the user navigates; no files
    are created, moved, or modified here."""
    # Folder browsing is a read-only GET operation. Read the path from the
    # query string; browsers do not reliably support request bodies on GET.
    path = request.args.get("path", "")
    current = Path(path or ".").expanduser().resolve()

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
    """Validate and persist an output folder chosen in the browser.
    
    ``state_store.update_output_folder`` creates the directory when needed
    and performs the state update under the file lock. The route converts
    path-related failures into a 400 response so the interface can show the
    reason without crashing the server."""
    try:
        state, output_folder = update_output_folder(
            (request.get_json(silent=True) or {}).get("output_folder", "")
        )
    except (OSError, ValueError) as error:
        return api_error(str(error), 400)
    return jsonify(
        {
            "output_folder": str(output_folder),
            "settings": state.get("settings", {}),
        }
    )


@app.get("/api/scan-progress")
def api_scan_progress():
    """Return a read-only snapshot of the active scan's messages and timings.
    
    The JavaScript progress timer polls this small endpoint frequently. The
    snapshot is detached from the internal dictionary, so JSON serialization
    cannot mutate the live scan state."""
    return jsonify(scan_progress_snapshot())


@app.post("/api/scan")
def api_scan():
    """Orchestrate an entire scan request from form submission to review queue.
    
    This is the main server-side entry point for scanning. It deliberately
    coordinates other modules rather than implementing OCR itself.
    
    Processing stages:
        1. Reset progress and normalize the submitted settings.
        2. Validate input/output folders and create the output directory.
        3. Load configuration and derive project/section from the folder name.
        4. Reuse or create an OCR engine with the requested device settings.
        5. Run ``scan_batch`` or ``scan_mass`` according to the selected mode.
        6. Persist the resulting queue, record total timing, and mark progress
           finished so browser polling can stop.
    
    Early validation failures update both the HTTP response and progress panel.
    In Mass mode, completed documents are appended during the scan, so the final
    state is re-read from disk instead of relying on an old in-memory copy."""

    # Prepares for the scanning process by gathering settings, input folder,
    # output folder, resetting the scan progress window and starting a new scan
    # progress window.
    # REQUEST PHASE 1 - Start a new observable scan. Resetting before any
    # validation means even an invalid folder produces a complete progress
    # story instead of leaving the browser spinner in an unknown state.
    reset_scan_progress()
    scan_started = time.perf_counter()
    add_scan_progress("Starting scan request.")
    settings = scan_settings(json_payload())
    input_folder = Path(settings["input_folder"])
    in_place = bool(settings.get("in_place", False))
    output_folder = (
        Path(settings["output_folder"])
        if settings["output_folder"]
        else input_folder
    )

    # Ensures that the given input and output folders are in settings and that
    # input is a directory. If not a Response object is returned with an error
    # message and the 400 error code. If the given output folder doesn't exist
    # yet one is created.
    if not settings["input_folder"]:
        finish_scan_progress(
            failed=True, message="Scan failed: Input folder is required."
        )
        return api_error("Input folder is required.", 400)
    if not in_place and not settings["output_folder"]:
        finish_scan_progress(
            failed=True,
            message="Scan failed: Output folder is required unless In-Place is enabled.",
        )
        return api_error(
            "Output folder is required unless In-Place is enabled.", 400
        )
    if not input_folder.is_dir():
        finish_scan_progress(
            failed=True,
            message=f"Scan failed: Input folder not found: {input_folder}",
        )
        return api_error(f"Input folder not found: {input_folder}", 400)

    if not in_place:
        output_folder.mkdir(parents=True, exist_ok=True)

    # Gets the path to the config file if one is given.
    config_path = (
        Path(settings["config_path"]).resolve()
        if settings["config_path"]
        else None
    )

    # If there is a config file it is loaded into a python dictionary.
    # REQUEST PHASE 2 - Load matching rules only after paths are valid.
    # The configuration cache makes repeat scans cheap, while a changed file
    # is detected from its timestamp and size.
    config_started = time.perf_counter()
    config = load_config(
        config_path if config_path and config_path.exists() else None
    )
    set_scan_timing("config_load", time.perf_counter() - config_started)

    # Gets the project code and section from the output folder name. Then updates
    # the section in settins to match the detected section.
    naming_folder = input_folder if in_place else output_folder
    detected_project_code, detected_section = _folder_project_and_section(
        naming_folder
    )
    config["default_county"] = settings.get("county", "")

    settings["section"] = detected_section

    # Attempts to get a manually entered project code from the current settings.
    # If there is one the letters in it are converted to uppercase and the
    # override for project code is set to match the manual project code. If
    # there isn't a manual project code in settings the detected project code is
    # entered into settings and the override it set to an empty string.
    manual_project_code = (settings.get("project_code") or "").strip()
    if manual_project_code:
        settings["project_code"] = manual_project_code.upper()
        settings["project_code_override"] = settings["project_code"]
    else:
        settings["project_code"] = (
            (detected_project_code or "").upper().strip()
        )
        settings["project_code_override"] = ""

    scan_mode = settings.get("scan_mode", "batch")
    if scan_mode not in {"batch", "mass"}:
        scan_mode = "batch"
        settings["scan_mode"] = scan_mode

    # Sequential scans reuse a process-lifetime OCR engine from ocr_service.
    # Parallel CPU batch workers still initialize one private engine per
    # process.
    use_main_process_engine = (
        scan_mode == "mass"
        or str(settings.get("ocr_device", "auto")).lower() == "gpu"
        or not settings.get("parallel_ocr", False)
        or int(settings.get("ocr_workers", 1)) <= 1
    )
    engine_started = time.perf_counter()
    ocr = (
        get_cached_ocr(
            lang=settings["lang"],
            cpu_threads=int(settings.get("ocr_threads_per_worker") or 4),
            ocr_device=settings.get("ocr_device", "auto"),
            gpu_device_id=int(settings.get("gpu_device_id") or 0),
        )
        if use_main_process_engine
        else None
    )
    engine_elapsed = time.perf_counter() - engine_started
    set_scan_timing("ocr_engine_ready", engine_elapsed)
    add_scan_progress(f"OCR engine ready in {engine_elapsed:.2f} seconds.")

    # REQUEST PHASE 4 - Choose the persistence model. Batch produces a full
    # queue and replaces state once. Mass publishes one document at a time,
    # so it clears the old queue first and then appends atomically.
    state = {"settings": settings, "documents": []}
    if scan_mode == "mass":
        # Clear the previous queue immediately. Atomic state writes let the
        # browser safely poll this file while documents are appended.
        replace_state(settings=settings, documents=[])

        scan_mass(
            input_folder,
            ocr,
            config,
            settings,
            progress_callback=add_scan_progress,
            document_ready_callback=append_document,
        )
    else:
        state["documents"] = scan_batch(
            input_folder,
            ocr,
            config,
            settings,
            progress_callback=add_scan_progress,
        )
        if settings.get("section"):
            for document in state["documents"]:
                document["metadata"]["section"] = settings["section"]
        replace_state(settings=settings, documents=state["documents"])
    final_state = read_state() if scan_mode == "mass" else state
    total_elapsed = time.perf_counter() - scan_started
    set_scan_timing("scan_total", total_elapsed)
    add_scan_progress(f"Performance summary: total scan time {total_elapsed:.2f} seconds.")
    finish_scan_progress(
        message=f"Scan complete. {len(final_state['documents'])} document(s)\
          ready for review."
    )
    return jsonify(final_state)


"""
@app.post("/api/scan")
async def api_scan_trigger():
    task = asyncio.create_task(api_scan())
    await task
    return jsonify(read_state())
"""


@app.patch("/api/documents/<document_id>")
def api_update_document(document_id: str):
    """Apply one review-form edit to the matching live document.
    
    The route delegates all business rules to ``apply_document_update`` inside
    ``state_store.update_document``. That locked transaction is important in
    Mass mode: a new OCR result can be appended while the user edits an older
    result, and neither change should overwrite the other.
    
    The response includes the full refreshed queue plus the specifically
    updated record so JavaScript can redraw both the list and detail form."""
    payload = json_payload()
    try:
        state, updated = update_document(
            document_id,
            lambda latest_state, document: apply_document_update(
                latest_state, document, payload
            ),
        )
    except KeyError:
        return api_error("Document not found", 404)
    return jsonify(
        {
            "settings": state.get("settings", {}),
            "documents": [
                sync_document_metadata(doc)
                for doc in state.get("documents", [])
            ],
            "updated": updated,
        }
    )


@app.post("/api/documents/<document_id>/file")
def api_file_document(document_id: str):
    """File one reviewed PDF and remove only that item from the active queue.
    
    The route rejects lookup-only SDAT helper PDFs because they are not
    permanent project documents. It then resolves In-Place versus output-tree
    filing, calls ``file_document_to_output`` to write metadata and move/copy
    the PDF, and atomically removes the completed record.
    
    Re-reading the latest state during removal protects documents that a Mass
    Scan may have appended while this filing request was running."""
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

    in_place = bool(
        payload.get(
            "in_place", state.get("settings", {}).get("in_place", False)
        )
    )
    output_folder = Path(state.get("settings", {}).get("output_folder") or ".")
    if not in_place:
        try:
            state, output_folder = update_output_folder(
                payload.get("output_folder")
                or state.get("settings", {}).get("output_folder", "")
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
            in_place=in_place,
        )
    except FileNotFoundError:
        return api_error(
            "File not located in specified input folder anymore.", 400
        )

    # A successfully filed document no longer needs review. Re-read the latest
    # state so documents completed by an active mass scan are preserved, remove
    # only the filed document, and persist the shortened review queue.
    latest_state, _ = remove_document(document_id)
    return jsonify(
        {
            "settings": latest_state.get("settings", {}),
            "documents": latest_state.get("documents", []),
            "filed": filed,
        }
    )


@app.post("/api/file-all")
def api_file_all_documents():
    """File every permanent document currently waiting for review.
    
    All normal documents share the first document's destination folder in
    Batch mode. Each PDF is processed by the same single-document filing
    function, then a CSV audit row is appended. Lookup-only SDAT printouts are
    discarded only after every permanent file succeeds; In-Place mode keeps
    all source PDFs and skips the external tracker.
    
    The review queue is cleared only after the filing loop completes, avoiding
    a state that claims work is finished when a file operation actually failed."""
    payload = request.get_json(silent=True) or {}
    state = read_state()
    in_place = bool(
        payload.get(
            "in_place", state.get("settings", {}).get("in_place", False)
        )
    )
    output_folder = Path(state.get("settings", {}).get("output_folder") or ".")
    if not in_place:
        try:
            state, output_folder = update_output_folder(
                payload.get("output_folder")
                or state.get("settings", {}).get("output_folder", "")
            )
        except (OSError, ValueError) as error:
            return api_error(str(error), 400)
    documents = state.get("documents", [])
    normal_documents = [
        doc for doc in documents if not doc.get("is_lookup_document")
    ]
    lookup_documents = [
        doc for doc in documents if doc.get("is_lookup_document")
    ]
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
                    in_place=in_place,
                )
            )
    except FileNotFoundError as error:
        return api_error(str(error), 400)

    if not in_place:
        try:
            append_batch_tracker(
                normal_documents, output_folder, filed_documents
            )
        except OSError as error:
            return api_error(
                f"Documents were filed, but the tracker could not be updated: \
                {error}",
                500,
            )

    # Standard filing removes lookup-only helper PDFs after the permanent
    # documents succeed. In-Place mode preserves every scanned source file.
    if not in_place:
        for lookup in lookup_documents:
            source = Path(lookup.get("source_path", ""))
            if source.exists():
                try:
                    from send2trash import send2trash

                    send2trash(str(source))
                except Exception:  # noqa: BLE001
                    source.unlink(missing_ok=True)
    state = clear_documents()
    return jsonify({"settings": state.get("settings", {}), "documents": []})


@app.get("/documents/<document_id>/pdf")
def document_pdf(document_id: str):
    """Stream a source PDF into the embedded browser viewer.
    
    The URL contains the stable review-record ID rather than a filesystem path.
    The function looks up that ID in state and passes the stored source path to
    Flask's ``send_file``. Missing records or deleted source files fall back to
    ``file-not-found.pdf`` so the viewer still receives a valid PDF response."""
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


if __name__ == "__main__":
    """Where it all begins. app is the Flask object created in the imports and 
constants section. This object connects all of the files together. The run 
function runs app.py on a local development server. For this project this was 
the easiest way to create a user interface to interact with the documents being 
processed. Once this is created it is stagnant until a user opens the the
browser with the address http://127.0.0.1:5055. Once the user does this app uses
the @app.get("/") decorator to call the index() function. This means that when 
the browser requests GET http://localhost:5055/(Which happens as soon as you 
open the above address) the app object searches through defined routes and finds
 @app.get("/") pointing to the index() function and knows to call it."""
    app.run(host="127.0.0.1", port=5055, debug=True)

"""IMPORTS AND CONSTANTS SECTION:"""

from __future__ import annotations

import csv
import json
import shutil
import threading
import time
import uuid
from datetime import datetime
import fitz

try:
    import pikepdf
except Exception:  # optional, metadata still works without it
    pikepdf = None
from dataclasses import asdict, replace
from pathlib import Path
from typing import Any

from flask import Flask, jsonify, render_template, request, send_file

from pipeline import (
    ExtractedMetadata,
    LOOKUP_DOCUMENT_TYPE,
    choose_batch_metadata_by_vote,
    _lookup_by_tax_id,
    enrich_metadata_with_sdat,
    lookup_maryland_property_by_address,
    metadata_from_sdat_record,
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


def reset_scan_progress() -> None:
    with _SCAN_PROGRESS_LOCK:
        _SCAN_PROGRESS.update({
            "active": True,
            "finished": False,
            "failed": False,
            "started_at": time.perf_counter(),
            "messages": [],
        })


def add_scan_progress(message: str) -> None:
    text = str(message or "").strip()
    if not text:
        return
    with _SCAN_PROGRESS_LOCK:
        elapsed = max(0.0, time.perf_counter() - float(_SCAN_PROGRESS.get("started_at") or 0.0))
        _SCAN_PROGRESS.setdefault("messages", []).append({
            "text": text,
            "elapsed": round(elapsed, 2),
        })


def finish_scan_progress(*, failed: bool = False, message: str = "") -> None:
    if message:
        add_scan_progress(message)
    with _SCAN_PROGRESS_LOCK:
        _SCAN_PROGRESS["active"] = False
        _SCAN_PROGRESS["finished"] = True
        _SCAN_PROGRESS["failed"] = failed


def scan_progress_snapshot() -> dict[str, Any]:
    with _SCAN_PROGRESS_LOCK:
        started_at = float(_SCAN_PROGRESS.get("started_at") or 0.0)
        elapsed = max(0.0, time.perf_counter() - started_at) if started_at else 0.0
        return {
            "active": bool(_SCAN_PROGRESS.get("active")),
            "finished": bool(_SCAN_PROGRESS.get("finished")),
            "failed": bool(_SCAN_PROGRESS.get("failed")),
            "elapsed": round(elapsed, 2),
            "messages": list(_SCAN_PROGRESS.get("messages", [])),
        }

"""---------------------------------------------------------------------------------------"""

"""FUNCTION DEFINITION SECTION"""


def api_error(message: str, status_code: int = 500):
    """The api_error() function returns a Response object and integer holding 
    an error message as a json and status code. """
    return jsonify({"error": message}), status_code


def read_state() -> dict[str, Any]:
    """The read_state() function returns all of the current settings and
    document metadata stored in the documents.json file which is in the
    .review_state folder in the project directory. If that file has not
    been created yet it returns a default dictionary with empty settings
    and documents."""
    if not STATE_FILE.exists():
        return dict(DEFAULT_STATE)
    with STATE_FILE.open("r", encoding="utf-8") as state_file:
        return json.load(state_file)


def write_state(state: dict[str, Any]) -> None:
    STATE_FILE.parent.mkdir(parents=True, exist_ok=True)
    with STATE_FILE.open("w", encoding="utf-8") as state_file:
        json.dump(state, state_file, indent=2)




def update_output_folder_setting(state: dict[str, Any], raw_value: str) -> Path:
    """Validate and persist a new output folder for the current review batch."""
    value = str(raw_value or "").strip()
    if not value:
        raise ValueError("Output folder is required.")
    output_folder = Path(value).expanduser().resolve()
    output_folder.mkdir(parents=True, exist_ok=True)
    state.setdefault("settings", {})["output_folder"] = str(output_folder)
    return output_folder


def append_batch_tracker(
    documents: list[dict[str, Any]],
    output_folder: Path,
    filed_documents: list[dict[str, Any]],
) -> None:
    """Append one compact CSV record for a successfully filed batch."""
    if not documents or not filed_documents:
        return

    metadata = documents[0].get("metadata", {})
    destination_folder = Path(filed_documents[0]["filed_path"]).parent
    row = {
        "lot_number": metadata.get("lot", ""),
        "address": metadata.get("address", ""),
        "location_filed": str(destination_folder),
        "time_filed": datetime.now().astimezone().isoformat(timespec="seconds"),
        "project_code": metadata.get("project_code", ""),
        "section": metadata.get("section", ""),
        "file_count": len(filed_documents),
        "files_filed": "|".join(Path(doc["filed_path"]).name for doc in filed_documents),
    }

    TRACKER_DIR.mkdir(parents=True, exist_ok=True)
    fieldnames = list(row)
    needs_header = not TRACKER_FILE.exists() or TRACKER_FILE.stat().st_size == 0
    with TRACKER_FILE.open("a", newline="", encoding="utf-8") as tracker:
        writer = csv.DictWriter(tracker, fieldnames=fieldnames)
        if needs_header:
            writer.writeheader()
        writer.writerow(row)


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


def suggested_folder(metadata: dict[str, str]) -> str:
    """The suggested_folder() function returns a string with a suggested folder name
    based on the metadata parameter. The folder name follows the naming conventions
    Lot # - Address(ex: Lot 1 - 34 Jibsail Street). After the lot and address
    information are pulled from the metadata parameter they are passed into the
    safe_path_part() function imported from pipeline.py to ensure they contain only
    allowed characters and remove extra spaces."""
    return safe_path_part(
        f"Lot {metadata.get('lot', '')} - {metadata.get('address', '')}",
        "Unknown Lot - Unknown Address",
    )


def suggested_filename(metadata: dict[str, str], source_name: str) -> str:
    """The suggested_filename() function returns a string with a suggested file name
    based on the metadata parameter. The file name follows the naming conventions
    Document Type - Lot #(ex: Site Plan - Lot 1). After the lot and address
    information are pulled from the metadata parameter they are passed into the
    safe_path_part() function imported from pipeline.py to ensure they contain only
    allowed characters and remove extra spaces."""
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


def normalize_document(document: dict[str, Any]) -> dict[str, Any]:
    """The normalize_document() function returns a checked/corrected document
    dictionary. The folder_name, file_name, and status fields are checked, and
    if empty or wrong corrected via the suggested_folder(),
    suggested_filename(), and document_status() functions to match the metadata
    gathered from the file."""
    metadata = document["metadata"]
    document.setdefault("folder_name", suggested_folder(metadata))
    document.setdefault(
        "file_name", suggested_filename(metadata, document["source_name"])
    )
    document["status"] = (
        "lookup_only"
        if document.get("is_lookup_document")
        else document_status(metadata)
    )
    return document


def find_document(state: dict[str, Any], document_id: str) -> dict[str, Any] | None:
    """The find_document() function returns the document in the state parameter
    with id matching the document_id parameter or None if there aren't any 
    documents in the state parameter with an id that matches the document_id 
    parameter."""
    return next(
        (doc for doc in state.get("documents", []) if doc.get("id") == document_id),
        None,
    )


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
    input_folder: Path, ocr, config: dict[str, Any], settings: dict[str, Any],
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
            normalize_document(
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
        section=str(metadata.get("section", "")),
    )


def _folder_project_and_section(output_folder: Path) -> tuple[str, str]:
    """The _folder_project_and_section() function returns the project code and 
    section taken from the output_folder parameter. It splits the parameter on
    the . and the - to determine the project code and section and returns both."""
    name = output_folder.name.strip()
    if "." not in name:
        return name, ""
    project_code, section = name.split(".", 1)
    try:
        section, extra = section.split("-", 1)
        return project_code.strip(), section.strip()

    finally:
        return project_code.strip(), section.strip()


def refresh_batch_property_fields_from_sdat(
    state: dict[str, Any], changed_field: str
) -> dict[str, str] | None:
    """Validate the edited field with SDAT, then synchronize all property fields."""
    documents = [
        doc for doc in state.get("documents", []) if not doc.get("is_lookup_document")
    ]
    if not documents:
        return None
    config = load_config_from_state(state)
    if not config.get("sdat_lookup", True):
        return None

    seed = metadata_from_dict(documents[0].get("metadata", {}))
    county = str(config.get("default_county", "") or "")
    records: list[dict[str, Any]] = []

    if changed_field == "tax_id":
        # Fast exact lookup. A typed Tax ID is not allowed to overwrite the batch
        # unless SDAT confirms it.
        records = _lookup_by_tax_id(seed.tax_id, county)
    elif changed_field == "address":
        records = lookup_maryland_property_by_address(seed.address, county=county, limit=25)
    else:
        # For map/parcel/section, ignore stale stronger identifiers and query only
        # the edited property identifiers. Empty text avoids a full OCR-text join.
        query_seed = replace(seed, tax_id="", address="Unknown Address")
        enriched = enrich_metadata_with_sdat(query_seed, "", config)
        if enriched != query_seed:
            values = {
                field: getattr(enriched, field)
                for field in ("lot", "address", "tax_map", "parcel", "tax_id", "section")
            }
            for batch_document in documents:
                batch_document["metadata"].update(values)
                refresh_document_names(batch_document, auto_folder=True, auto_file_name=True)
            return values
        return None

    if not records:
        return None

    enriched = metadata_from_sdat_record(seed, records[0])
    values = {
        field: getattr(enriched, field)
        for field in ("lot", "address", "tax_map", "parcel", "tax_id", "section")
    }
    for batch_document in documents:
        batch_document["metadata"].update(values)
        refresh_document_names(batch_document, auto_folder=True, auto_file_name=True)
    return values


def apply_document_update(
    state: dict[str, Any], document: dict[str, Any], payload: dict[str, Any]
) -> dict[str, Any]:
    metadata = document["metadata"]

    # These are batch-level values. If the user corrects one file, apply the
    # correction to every document in the current batch.
    shared_field_names = (
        "lot",
        "address",
        "tax_map",
        "parcel",
        "tax_id",
        "section",
        "project_code",
    )
    shared_updates = {
        field: payload[field] for field in shared_field_names if field in payload
    }
    changed_field = payload.get("changed_field", "")

    if shared_updates:
        for batch_document in state.get("documents", []):
            batch_document["metadata"].update(shared_updates)
            refresh_document_names(
                batch_document, auto_folder=True, auto_file_name=True
            )

    # If the user edits a property identifier, refresh the official SDAT address
    # once and apply that address to every document in the batch.
    if changed_field in {"tax_map", "parcel", "tax_id", "address", "section"}:
        refresh_batch_property_fields_from_sdat(state, changed_field)

    # Keep non-shared fields document-specific if they are ever posted by older UI/state.
    for field in (*REQUIRED_METADATA_FIELDS, *OPTIONAL_METADATA_FIELDS):
        if field in payload and field not in shared_updates:
            metadata[field] = payload[field]

    # Keep the lookup-only behavior synchronized with the editable document type.
    # Lookup-only documents stay visible in the review queue, but they are not
    # filed and are removed only after the permanent batch files successfully.
    document["is_lookup_document"] = (
        metadata.get("document_type") == LOOKUP_DOCUMENT_TYPE
    )

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


def metadata_keyword_text(document: dict[str, Any]) -> str:
    metadata = document.get("metadata", {})
    custom_text = {
        "lot": metadata.get("lot", ""),
        "address": metadata.get("address", ""),
        "project_code": metadata.get("project_code", ""),
        "document_type": metadata.get("document_type", ""),
        "tax_map": metadata.get("tax_map", ""),
        "parcel": metadata.get("parcel", ""),
        "tax_id": metadata.get("tax_id", ""),
        "section": metadata.get("section", ""),
        "source_name": document.get("source_name", ""),
        "filed_at": datetime.now().isoformat(timespec="seconds"),
    }
    return "; ".join(f"{key}={value}" for key, value in custom_text.items() if value)


def _ocr_item_pdf_rect(
    item: dict[str, Any],
    x_scale: float,
    y_scale: float,
) -> fitz.Rect | None:
    """Convert one OCR item's pixel geometry to a PDF-point rectangle."""
    raw = item.get("bbox") or item.get("polygon")
    if raw is None:
        return None
    if hasattr(raw, "tolist"):
        raw = raw.tolist()

    try:
        # Preferred normalized bbox format: [x0, y0, x1, y1].
        if len(raw) == 4 and all(isinstance(value, (int, float)) for value in raw):
            x0, y0, x1, y1 = [float(value) for value in raw]
        else:
            # Compatibility with four-point PaddleOCR polygons.
            points = [point for point in raw if isinstance(point, (list, tuple)) and len(point) >= 2]
            if not points:
                return None
            xs = [float(point[0]) for point in points]
            ys = [float(point[1]) for point in points]
            x0, y0, x1, y1 = min(xs), min(ys), max(xs), max(ys)
    except (TypeError, ValueError):
        return None

    rect = fitz.Rect(x0 * x_scale, y0 * y_scale, x1 * x_scale, y1 * y_scale)
    if rect.is_empty or rect.width <= 0 or rect.height <= 0:
        return None
    return rect


def add_paddle_searchable_text_layer(pdf_path: Path, document: dict[str, Any]) -> None:
    """Add selectable, invisible text using the stored PaddleOCR geometry.

    PaddleOCR coordinates are measured in rendered-image pixels. They are
    mapped back to PDF points using the image dimensions saved during OCR.
    The text uses PDF render mode 3, so it remains invisible while still being
    searchable and selectable in browsers and PDF readers.
    """
    ocr_pages = document.get("ocr_pages") or []
    if not ocr_pages:
        return

    font = fitz.Font("helv")
    inserted = 0

    with fitz.open(pdf_path) as pdf:
        for page_data in ocr_pages:
            try:
                page_index = int(page_data.get("page_index", 0))
                page = pdf[page_index]
                image_width = float(page_data.get("image_width") or 0)
                image_height = float(page_data.get("image_height") or 0)
            except (IndexError, TypeError, ValueError):
                continue

            if image_width <= 0 or image_height <= 0:
                continue

            x_scale = float(page.rect.width) / image_width
            y_scale = float(page.rect.height) / image_height

            for item in page_data.get("items", []):
                text = str(item.get("text", "")).strip()
                if not text:
                    continue

                rect = _ocr_item_pdf_rect(item, x_scale, y_scale)
                if rect is None:
                    continue

                # Fit invisible text to the OCR rectangle.  The baseline is
                # derived from Helvetica's ascender / descender rather than
                # being placed at the bottom of the box, which keeps selection
                # geometry aligned with the detected line.
                natural_width = max(font.text_length(text, fontsize=1), 0.01)
                height_size = rect.height / max(font.ascender - font.descender, 0.01)
                width_size = (rect.width * 0.98) / natural_width
                font_size = min(max(min(height_size, width_size), 1.0), 72.0)
                baseline_y = rect.y0 + (font.ascender * font_size)
                baseline = fitz.Point(rect.x0, baseline_y)

                try:
                    page.insert_text(
                        baseline,
                        text,
                        fontsize=font_size,
                        fontname="helv",
                        render_mode=3,
                        overlay=True,
                    )
                    inserted += 1
                except Exception:
                    continue

        if inserted:
            # Incremental save is fast and preserves the scanned page content.
            pdf.saveIncr()


def write_standard_pdf_metadata(pdf_path: Path, document: dict[str, Any]) -> None:
    metadata = document.get("metadata", {})
    with fitz.open(pdf_path) as pdf:
        pdf.set_metadata(
            {
                **pdf.metadata,  # type: ignore
                "title": f"{metadata.get('document_type', '')} - Lot {metadata.get('lot', '')}",
                "subject": metadata.get("address", ""),
                "keywords": metadata_keyword_text(document),
                "creator": "COA Barrett File Identifier and Sorter",
            }
        )
        pdf.saveIncr()


def write_xmp_metadata(pdf_path: Path, document: dict[str, Any]) -> None:
    """Write structured XMP metadata with a custom COA namespace."""
    if pikepdf is None:
        return

    metadata = document.get("metadata", {})
    namespace = "https://coabarrett.local/ns/ocr-file-sorter/1.0/"

    try:
        with pikepdf.Pdf.open(pdf_path, allow_overwriting_input=True) as pdf:
            with pdf.open_metadata(set_pikepdf_as_editor=True) as meta:
                try:
                    meta.register_xml_namespace("coa", namespace)
                except Exception:
                    pass

                title = f"{metadata.get('document_type', '')} - Lot {metadata.get('lot', '')}".strip(
                    " -"
                )
                if title:
                    meta["dc:title"] = title
                if metadata.get("address"):
                    meta["dc:description"] = metadata.get("address", "")
                meta["pdf:Keywords"] = metadata_keyword_text(document)

                custom_fields = {
                    "coa:Lot": metadata.get("lot", ""),
                    "coa:Address": metadata.get("address", ""),
                    "coa:ProjectCode": metadata.get("project_code", ""),
                    "coa:DocumentType": metadata.get("document_type", ""),
                    "coa:TaxMap": metadata.get("tax_map", ""),
                    "coa:Parcel": metadata.get("parcel", ""),
                    "coa:TaxID": metadata.get("tax_id", ""),
                    "coa:Section": metadata.get("section", ""),
                    "coa:OriginalFileName": document.get("source_name", ""),
                    "coa:FiledAt": datetime.now().isoformat(timespec="seconds"),
                    "coa:Application": "COA Barrett File Identifier and Sorter",
                }
                for key, value in custom_fields.items():
                    if value:
                        meta[key] = str(value)
            pdf.save(pdf_path)
    except Exception as exc:
        print(f"Could not write XMP metadata to {pdf_path}: {exc}")


def write_pdf_metadata(pdf_path: Path, document: dict[str, Any]) -> None:
    """Write standard metadata, structured XMP, and a PaddleOCR text layer."""
    try:
        add_paddle_searchable_text_layer(pdf_path, document)
    except Exception as exc:
        print(f"Could not add PaddleOCR searchable text layer to {pdf_path}: {exc}")

    try:
        write_standard_pdf_metadata(pdf_path, document)
    except Exception as exc:
        print(f"Could not write standard PDF metadata to {pdf_path}: {exc}")

    write_xmp_metadata(pdf_path, document)


def file_document_to_output(
    document: dict[str, Any],
    output_folder: Path,
    copy_file: bool = False,
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

    write_pdf_metadata(destination, document)

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


@app.patch("/api/settings/output-folder")
def api_update_output_folder():
    state = read_state()
    try:
        output_folder = update_output_folder_setting(
            state, (request.get_json(silent=True) or {}).get("output_folder", "")
        )
    except (OSError, ValueError) as error:
        return api_error(str(error), 400)
    write_state(state)
    return jsonify({"output_folder": str(output_folder), "settings": state.get("settings", {})})


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
        finish_scan_progress(failed=True, message="Scan failed: Input folder is required.")
        return api_error("Input folder is required.", 400)
    if not settings["output_folder"]:
        finish_scan_progress(failed=True, message="Scan failed: Output folder is required.")
        return api_error("Output folder is required.", 400)
    if not input_folder.is_dir():
        finish_scan_progress(
            failed=True, message=f"Scan failed: Input folder not found: {input_folder}"
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
        settings["project_code"] = safe_path_part(
            detected_project_code,
            "Project"
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
        "documents": scan_batch(
            input_folder, ocr, config, settings, progress_callback=add_scan_progress
        ),
    }
    if settings.get("section"):
        for document in state["documents"]:
            document["metadata"]["section"] = settings["section"]

    write_state(state)
    finish_scan_progress(message=f"Scan complete. {len(state['documents'])} document(s) ready for review.")
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
    if document.get("is_lookup_document"):
        return api_error(
            "Lookup-only SDAT records are removed after the batch is filed.", 400
        )

    try:
        output_folder = update_output_folder_setting(
            state, payload.get("output_folder") or state.get("settings", {}).get("output_folder", "")
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
            state, payload.get("output_folder") or state.get("settings", {}).get("output_folder", "")
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
        return api_error(f"Documents were filed, but the tracker could not be updated: {error}", 500)

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
    return jsonify(
        {"settings": state.get("settings", {}), "documents": []}
    )


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

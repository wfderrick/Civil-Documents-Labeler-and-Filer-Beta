"""IMPORTS AND CONSTANTS SECTION:"""

from __future__ import annotations

import json
import shutil
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

from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for

from pipeline import (
    ExtractedMetadata,
    LOOKUP_DOCUMENT_TYPE,
    choose_batch_metadata_by_vote,
    enrich_metadata_with_sdat,
    lookup_maryland_property_by_address,
    metadata_from_sdat_record,
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
OPTIONAL_METADATA_FIELDS = ("tax_map", "parcel", "tax_id", "section")

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


"""The suggested_folder() function returns a string with a suggested folder name
based on the metadata parameter. The folder name follows the naming conventions
Lot # - Address(ex: Lot 1 - 34 Jibsail Street). After the lot and address 
information are pulled from the metadata parameter they are passed into the 
safe_path_part() function imported from pipeline.py to ensure they contain only 
allowed characters and remove extra spaces."""
def suggested_folder(metadata: dict[str, str]) -> str:
    return safe_path_part(
        f"Lot {metadata.get('lot', '')} - {metadata.get('address', '')}",
        "Unknown Lot - Unknown Address",
    )


"""The suggested_filename() function returns a string with a suggested file name
based on the metadata parameter. The file name follows the naming conventions
Document Type - Lot #(ex: Site Plan - Lot 1). After the lot and address 
information are pulled from the metadata parameter they are passed into the 
safe_path_part() function imported from pipeline.py to ensure they contain only 
allowed characters and remove extra spaces."""
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
    document["status"] = "lookup_only" if document.get("is_lookup_document") else document_status(metadata)
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
    shared_metadata, metadata_votes = choose_batch_metadata_by_vote(
        scanned_documents=scanned,
        config=config,
        default_project_code=settings["project_code"],
        default_document_type=settings["document_type"],
    )
    print("Finished merging metadata across documents")
    documents: list[dict[str, Any]] = []
    print("Begin normalizing documents")
    for scanned_document, metadata_vote in zip(scanned, metadata_votes):
        is_lookup = metadata_vote.document_type == LOOKUP_DOCUMENT_TYPE
        final_metadata = metadata_vote if is_lookup else merge_batch_metadata(
            document_text=scanned_document["ocr_text"],
            config=config,
            default_project_code=settings["project_code"],
            default_document_type=settings["document_type"],
            shared_metadata=shared_metadata,
            document_metadata=metadata_vote,
        )
        documents.append(normalize_document({
            "id": uuid.uuid4().hex,
            "source_path": scanned_document["source_path"],
            "source_name": scanned_document["source_name"],
            "ocr_text": scanned_document["ocr_text"],
            "ocr_pages": scanned_document.get("ocr_pages", []),
            "metadata": asdict(final_metadata),
            "is_lookup_document": is_lookup,
            "filed_path": "",
        }))
    print("Finished normalizing documents")
    normal_documents = [document for document in documents if not document.get("is_lookup_document")]
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
    name = output_folder.name.strip()
    if "." not in name:
        return name, ""
    project_code, section = name.split(".", 1)
    return project_code.strip(), section.strip()


def refresh_batch_property_fields_from_sdat(state: dict[str, Any], changed_field: str) -> dict[str, str] | None:
    """Use the edited property field as authoritative and synchronize the batch."""
    documents = [doc for doc in state.get("documents", []) if not doc.get("is_lookup_document")]
    if not documents:
        return None
    config = load_config_from_state(state)
    if not config.get("sdat_lookup", True):
        return None
    seed = metadata_from_dict(documents[0].get("metadata", {}))
    batch_text = "\n".join(document.get("ocr_text", "") for document in documents)

    if changed_field == "address":
        records = lookup_maryland_property_by_address(seed.address, county=str(config.get("default_county", "") or ""))
        enriched = metadata_from_sdat_record(seed, records[0]) if records else seed
    else:
        # Tax ID has highest priority; clear stale address/map/parcel before querying it.
        if changed_field == "tax_id":
            seed = replace(seed, address="Unknown Address", tax_map="", parcel="", lot="Unknown Lot", section="")
        elif changed_field in {"tax_map", "parcel", "section"}:
            seed = replace(seed, tax_id="", address="Unknown Address")
        enriched = enrich_metadata_with_sdat(seed, batch_text, config)

    values = {field: getattr(enriched, field) for field in ("lot", "address", "tax_map", "parcel", "tax_id", "section")}
    if not any(not is_unknown(str(value)) for value in values.values()):
        return None
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
    shared_field_names = ("lot", "address", "tax_map", "parcel", "tax_id", "section", "project_code")
    shared_updates = {field: payload[field] for field in shared_field_names if field in payload}
    changed_field = payload.get("changed_field", "")

    if shared_updates:
        for batch_document in state.get("documents", []):
            batch_document["metadata"].update(shared_updates)
            refresh_document_names(batch_document, auto_folder=True, auto_file_name=True)

    # If the user edits a property identifier, refresh the official SDAT address
    # once and apply that address to every document in the batch.
    if changed_field in {"tax_map", "parcel", "tax_id", "address", "section"}:
        refresh_batch_property_fields_from_sdat(state, changed_field)

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


def add_paddle_searchable_text_layer(pdf_path: Path, document: dict[str, Any]) -> None:
    """Add an invisible searchable text layer using PaddleOCR bounding boxes.

    The OCR boxes are stored in image-pixel coordinates. This maps them back to
    PDF page points using the image/page dimensions captured during OCR.
    """
    ocr_pages = document.get("ocr_pages") or []
    if not ocr_pages:
        return

    with fitz.open(pdf_path) as pdf:
        for page_data in ocr_pages:
            try:
                page_index = int(page_data.get("page_index", 0))
                page = pdf[page_index]
            except Exception:
                continue

            image_width = float(page_data.get("image_width") or 0)
            image_height = float(page_data.get("image_height") or 0)
            if image_width <= 0 or image_height <= 0:
                continue

            x_scale = float(page.rect.width) / image_width
            y_scale = float(page.rect.height) / image_height

            for item in page_data.get("items", []):
                text = str(item.get("text", "")).strip()
                bbox = item.get("bbox") or []
                if not text or len(bbox) != 4:
                    continue

                try:
                    x0, y0, x1, y1 = [float(value) for value in bbox]
                except Exception:
                    continue

                rect = fitz.Rect(x0 * x_scale, y0 * y_scale, x1 * x_scale, y1 * y_scale)
                if rect.is_empty or rect.width <= 0 or rect.height <= 0:
                    continue

                # Keep text invisible but searchable. render_mode=3 is invisible text.
                font_size = max(1.0, min(rect.height * 0.85, 14.0))
                try:
                    result = page.insert_textbox(
                        rect,
                        text,
                        fontsize=font_size,
                        fontname="helv",
                        render_mode=3,
                        overlay=True,
                    )
                    if result < 0:
                        # Fallback for very tight OCR boxes.
                        page.insert_text(
                            (rect.x0, rect.y1),
                            text,
                            fontsize=font_size,
                            fontname="helv",
                            render_mode=3,
                            overlay=True,
                        )
                except Exception:
                    continue

        pdf.saveIncr()


def write_standard_pdf_metadata(pdf_path: Path, document: dict[str, Any]) -> None:
    metadata = document.get("metadata", {})
    with fitz.open(pdf_path) as pdf:
        pdf.set_metadata({
            **pdf.metadata, # type: ignore
            "title": f"{metadata.get('document_type', '')} - Lot {metadata.get('lot', '')}",
            "subject": metadata.get("address", ""),
            "keywords": metadata_keyword_text(document),
            "creator": "COA Barrett File Identifier and Sorter",
        })
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

                title = f"{metadata.get('document_type', '')} - Lot {metadata.get('lot', '')}".strip(" -")
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
    detected_project_code, detected_section = _folder_project_and_section(output_folder)
    settings["section"] = detected_section
    manual_project_code = (settings.get("project_code") or "").strip()
    if manual_project_code:
        settings["project_code"] = safe_path_part(manual_project_code.upper(), "Project")
        settings["project_code_override"] = settings["project_code"]
    else:
        settings["project_code"] = safe_path_part(
            detected_project_code or extract_project_code_from_output_folder(output_folder, config, "Project"),
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
    if settings.get("section"):
        for document in state["documents"]:
            document["metadata"]["section"] = settings["section"]

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
    if document.get("is_lookup_document"):
        return api_error("Lookup-only SDAT records are removed after the batch is filed.", 400)

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
    output_folder = Path(state.get("settings", {}).get("output_folder", "")).expanduser().resolve()
    documents = state.get("documents", [])
    normal_documents = [doc for doc in documents if not doc.get("is_lookup_document")]
    lookup_documents = [doc for doc in documents if doc.get("is_lookup_document")]
    if not normal_documents:
        return api_error("No permanent documents to file.", 400)
    shared_folder = normal_documents[0].get("folder_name") or suggested_folder(normal_documents[0]["metadata"])
    filed_documents = []
    try:
        for document in normal_documents:
            filed_documents.append(file_document_to_output(
                document, output_folder, copy_file=payload.get("copy", True),
                save_text=payload.get("save_text", False), folder_name=shared_folder,
            ))
    except FileNotFoundError as error:
        return api_error(str(error), 400)

    # Delete lookup-only source records only after every permanent document succeeds.
    for lookup in lookup_documents:
        source = Path(lookup.get("source_path", ""))
        if source.exists():
            try:
                from send2trash import send2trash
                send2trash(str(source))
            except Exception:
                source.unlink(missing_ok=True)
    state["documents"] = filed_documents
    write_state(state)
    return jsonify({"settings": state.get("settings", {}), "documents": filed_documents})


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
